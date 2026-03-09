use crate::quickscorer::{QuickRouteMeta, QuickScorerEngine};
use crate::{
    config::Config,
    schema::{ReasonItem, ScoreResponse, TimingsUs},
    util::{mix_u64, now_us},
};

use anyhow::Context;
use bytes::Bytes;
use risk_quickscorer_standalone_l2::{StandaloneL2Runtime, StandaloneL2Scratch};
use std::cell::Cell;
use std::cell::RefCell;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

#[derive(Debug)]
struct RateBudget {
    limit_per_sec: u64,
    start: Instant,
    sec: AtomicU64,
    count: AtomicU64,
}

impl RateBudget {
    fn new(limit_per_sec: u64) -> Self {
        Self {
            limit_per_sec,
            start: Instant::now(),
            sec: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    #[inline]
    fn now_sec(&self) -> u64 {
        self.start.elapsed().as_secs()
    }

    fn try_acquire(&self) -> bool {
        if self.limit_per_sec == 0 {
            return true;
        }

        let now = self.now_sec();
        let cur = self.sec.load(Ordering::Relaxed);
        if cur != now
            && self
                .sec
                .compare_exchange(cur, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
        {
            self.count.store(0, Ordering::Relaxed);
        }

        let n = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        n <= self.limit_per_sec
    }
}

thread_local! {
    static TLS_L2_SAMPLE_RNG: Cell<u64> = const { Cell::new(0) };
    static TLS_STANDALONE_L2_SCRATCH: RefCell<Option<StandaloneL2Scratch>> = const { RefCell::new(None) };
}

fn tls_rand_u64() -> u64 {
    TLS_L2_SAMPLE_RNG.with(|c| {
        let mut state = c.get();
        if state == 0 {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            let salt = (std::process::id() as u64).wrapping_mul(0x9e3779b97f4a7c15);
            let addr = (&*c as *const Cell<u64> as usize) as u64;
            state = now ^ salt ^ addr;
        }
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let out = mix_u64(state);
        c.set(state);
        out
    })
}

#[derive(Debug, Clone, Copy)]
pub enum StandaloneL2TauMode {
    Request,
    Fixed(f32),
}

#[derive(Debug)]
struct L2Control {
    rate_budget: Option<RateBudget>,
    max_queue_waterline: f64,
    min_remaining_us: u64,
    queue_wait_budget_us: u64,
    sample_base_ppm: u32,
    sample_min_ppm: u32,
    sample_dyn_ppm: AtomicU64,
    sample_dyn_enable: bool,
    sample_last_update_ms: AtomicU64,
    sample_update_ctr: AtomicU64,
    sample_waterline_target: f64,
    sample_waterline_hi: f64,
    sample_waterline_lo: f64,
    waterline_cached_bits: AtomicU64,
    start: Instant,
}

impl L2Control {
    fn from_env(cfg: &Config) -> Self {
        fn env_u64(key: &str) -> Option<u64> {
            std::env::var(key).ok().and_then(|s| s.parse::<u64>().ok())
        }
        fn env_f64(key: &str) -> Option<f64> {
            std::env::var(key).ok().and_then(|s| s.parse::<f64>().ok())
        }
        fn clamp01(v: f64) -> f64 {
            v.clamp(0.0, 1.0)
        }

        let max_triggers_per_sec = env_u64("ROUTER_L2_MAX_TRIGGERS_PER_SEC").unwrap_or(0);
        let rate_budget = if max_triggers_per_sec > 0 {
            Some(RateBudget::new(max_triggers_per_sec))
        } else {
            None
        };

        let max_queue_waterline = env_f64("ROUTER_L2_MAX_QUEUE_WATERLINE").unwrap_or(1.0);
        let default_min_remaining_us = ((cfg.slo_p99_ms * 1000) / 2).max(1_000);
        let min_remaining_us =
            env_u64("ROUTER_L2_MIN_REMAINING_US").unwrap_or(default_min_remaining_us);
        let queue_wait_budget_us = env_u64("ROUTER_L2_QUEUE_WAIT_BUDGET_US").unwrap_or(0);

        let base_ppm = env_u64("ROUTER_L2_SAMPLE_PPM")
            .or_else(|| {
                env_f64("ROUTER_L2_SAMPLE_RATIO")
                    .map(|v| (v.clamp(0.0, 1.0) * 1_000_000.0).round() as u64)
            })
            .unwrap_or(300_000)
            .min(1_000_000) as u32;

        let mut min_ppm = env_f64("ROUTER_L2_SAMPLE_MIN_RATIO")
            .map(|v| (v.clamp(0.0, 1.0) * 1_000_000.0).round() as u32)
            .unwrap_or_else(|| ((base_ppm as f64) * 0.1).round() as u32);
        if min_ppm > base_ppm {
            min_ppm = base_ppm;
        }

        let sample_dyn_enable = env_u64("ROUTER_L2_DYN_ENABLE").unwrap_or(1) != 0;
        let sample_waterline_target =
            clamp01(env_f64("ROUTER_L2_WATERLINE_TARGET").unwrap_or(0.60));
        let mut sample_waterline_hi = clamp01(env_f64("ROUTER_L2_WATERLINE_HI").unwrap_or(0.85));
        let mut sample_waterline_lo = clamp01(env_f64("ROUTER_L2_WATERLINE_LO").unwrap_or(0.40));
        if sample_waterline_lo > sample_waterline_hi {
            std::mem::swap(&mut sample_waterline_lo, &mut sample_waterline_hi);
        }

        metrics::gauge!("router_l2_sample_ratio").set(base_ppm as f64 / 1_000_000.0);

        Self {
            rate_budget,
            max_queue_waterline,
            min_remaining_us,
            queue_wait_budget_us,
            sample_base_ppm: base_ppm,
            sample_min_ppm: min_ppm,
            sample_dyn_ppm: AtomicU64::new(base_ppm as u64),
            sample_dyn_enable,
            sample_last_update_ms: AtomicU64::new(0),
            sample_update_ctr: AtomicU64::new(0),
            sample_waterline_target,
            sample_waterline_hi,
            sample_waterline_lo,
            waterline_cached_bits: AtomicU64::new(0.0f64.to_bits()),
            start: Instant::now(),
        }
    }

    #[inline]
    fn sample_ppm(&self) -> u32 {
        if self.sample_dyn_enable {
            self.sample_dyn_ppm.load(Ordering::Relaxed).min(1_000_000) as u32
        } else {
            self.sample_base_ppm
        }
    }

    #[inline]
    fn sample_ratio(&self) -> f64 {
        self.sample_ppm() as f64 / 1_000_000.0
    }

    #[inline]
    fn sample_base_ratio(&self) -> f64 {
        self.sample_base_ppm as f64 / 1_000_000.0
    }

    #[inline]
    fn sample_dyn_ratio(&self) -> f64 {
        self.sample_dyn_ppm.load(Ordering::Relaxed).min(1_000_000) as f64 / 1_000_000.0
    }

    #[inline]
    fn now_ms(&self) -> u64 {
        self.start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
    }

    fn feedback_overload(&self) {
        if !self.sample_dyn_enable {
            return;
        }
        let ctr = self.sample_update_ctr.fetch_add(1, Ordering::Relaxed) + 1;
        if ctr & 31 != 0 {
            return;
        }

        let now_ms = self.now_ms();
        let last = self.sample_last_update_ms.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last) < 50 {
            return;
        }
        if self
            .sample_last_update_ms
            .compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }

        let cur = self.sample_dyn_ppm.load(Ordering::Relaxed) as u32;
        let next = ((cur as f64) * 0.7).round() as u32;
        let next = next.max(self.sample_min_ppm).min(self.sample_base_ppm);
        if next < cur {
            self.sample_dyn_ppm.store(next as u64, Ordering::Relaxed);
            metrics::gauge!("router_l2_sample_ratio").set(next as f64 / 1_000_000.0);
            crate::batched_counter!("router_l2_feedback_overload_total").increment(1);
        }
    }

    fn feedback_relax(&self) {
        if !self.sample_dyn_enable {
            return;
        }
        let ctr = self.sample_update_ctr.fetch_add(1, Ordering::Relaxed) + 1;
        if ctr & 127 != 0 {
            return;
        }

        let now_ms = self.now_ms();
        let last = self.sample_last_update_ms.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last) < 200 {
            return;
        }
        if self
            .sample_last_update_ms
            .compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }

        let cur = self.sample_dyn_ppm.load(Ordering::Relaxed) as u32;
        if cur >= self.sample_base_ppm {
            return;
        }
        let next = ((cur as f64) * 1.08).round() as u32;
        let next = next.max(self.sample_min_ppm).min(self.sample_base_ppm);
        if next > cur {
            self.sample_dyn_ppm.store(next as u64, Ordering::Relaxed);
            metrics::gauge!("router_l2_sample_ratio").set(next as f64 / 1_000_000.0);
            crate::batched_counter!("router_l2_feedback_relax_total").increment(1);
        }
    }

    fn feedback_waterline(&self, waterline: f64) {
        self.waterline_cached_bits
            .store(waterline.to_bits(), Ordering::Relaxed);

        if self.rate_budget.as_ref().is_some_and(|b| !b.try_acquire()) {
            self.feedback_overload();
            return;
        }

        if waterline >= self.sample_waterline_hi {
            self.feedback_overload();
        } else if waterline <= self.sample_waterline_lo {
            self.feedback_relax();
        } else {
            let target = self.sample_waterline_target;
            let dist = (waterline - target).abs();
            if dist < 0.03 && (tls_rand_u64() & 0x3f) == 0 {
                self.feedback_relax();
            }
        }
    }
}

#[derive(Clone)]
pub struct L2ControlView {
    inner: Arc<L2Control>,
}

impl L2ControlView {
    #[inline]
    pub fn sample_ratio(&self) -> f64 {
        self.inner.sample_ratio()
    }

    #[inline]
    pub fn sample_base_ratio(&self) -> f64 {
        self.inner.sample_base_ratio()
    }

    #[inline]
    pub fn sample_dyn_ratio(&self) -> f64 {
        self.inner.sample_dyn_ratio()
    }

    #[inline]
    pub fn sample_waterline_target(&self) -> f64 {
        self.inner.sample_waterline_target
    }

    #[inline]
    pub fn sample_waterline_hi(&self) -> f64 {
        self.inner.sample_waterline_hi
    }

    #[inline]
    pub fn sample_waterline_lo(&self) -> f64 {
        self.inner.sample_waterline_lo
    }

    #[inline]
    pub fn feedback_overload(&self) {
        self.inner.feedback_overload();
    }

    #[inline]
    pub fn feedback_waterline(&self, waterline: f64) {
        self.inner.feedback_waterline(waterline);
    }

    #[inline]
    pub fn max_queue_waterline(&self) -> f64 {
        self.inner.max_queue_waterline
    }

    #[inline]
    pub fn min_remaining_us(&self) -> u64 {
        self.inner.min_remaining_us
    }

    #[inline]
    pub fn queue_wait_budget_us(&self) -> u64 {
        self.inner.queue_wait_budget_us
    }
}

#[inline]
fn record_serialize_metrics(resp: &mut ScoreResponse) {
    let ser_hist = crate::sampled_histogram!("stage_serialize_us");
    if ser_hist.enabled() {
        let t_ser = Instant::now();
        let _ = serde_json::to_vec(resp);
        resp.timings_us.serialize = now_us(t_ser);
        ser_hist.record(resp.timings_us.serialize as f64);
    }
}

#[inline]
fn record_e2e_metrics(t0: Instant) {
    crate::sampled_histogram!("e2e_us").record(now_us(t0) as f64);
}

#[derive(Clone)]
pub struct AppCore {
    pub cfg: Config,
    pub quick: Option<Arc<QuickScorerEngine>>,
    l2_ctrl: Arc<L2Control>,
}

impl AppCore {
    pub fn new(cfg: Config) -> Self {
        let l2_ctrl = Arc::new(L2Control::from_env(&cfg));
        Self {
            cfg,
            quick: None,
            l2_ctrl,
        }
    }

    pub fn new_with_quickscorer_bundle<P: AsRef<Path>>(
        cfg: Config,
        bundle_dir: P,
    ) -> anyhow::Result<Self> {
        let l2_ctrl = Arc::new(L2Control::from_env(&cfg));
        let quick = Arc::new(QuickScorerEngine::load(bundle_dir.as_ref())?);

        Ok(Self {
            cfg,
            quick: Some(quick),
            l2_ctrl,
        })
    }

    #[inline]
    pub fn quick_dims(&self) -> Option<(usize, usize)> {
        self.quick.as_ref().map(|q| (q.l1_dim(), q.l2_dim()))
    }

    pub fn l2_ctrl(&self) -> L2ControlView {
        L2ControlView {
            inner: Arc::clone(&self.l2_ctrl),
        }
    }

    pub async fn score_quick_dense_bytes_async(
        &self,
        parse_us: u64,
        row_bytes_le: Bytes,
    ) -> anyhow::Result<ScoreResponse> {
        self.score_quick_dense_bytes_with_meta_async(parse_us, row_bytes_le, None)
            .await
    }

    pub async fn score_quick_dense_bytes_with_meta_async(
        &self,
        parse_us: u64,
        row_bytes_le: Bytes,
        route_meta: Option<QuickRouteMeta>,
    ) -> anyhow::Result<ScoreResponse> {
        let t0 = Instant::now();

        let mut timings = TimingsUs::default();
        timings.parse = parse_us;

        let quick = self
            .quick
            .as_ref()
            .context("quickscorer not enabled: use new_with_quickscorer_bundle")?;

        let out =
            quick.predict_from_l1_bytes_with_meta(row_bytes_le.as_ref(), route_meta.as_ref())?;
        timings.feature = out.feature_us;
        timings.l1 = out.l1_us;
        timings.l2 = out.l2_us;
        timings.router = out.router_us;

        crate::sampled_histogram!("stage_feature_us").record(timings.feature as f64);
        crate::sampled_histogram!("stage_l1_us").record(timings.l1 as f64);
        crate::sampled_histogram!("stage_router_us").record(timings.router as f64);
        crate::sampled_histogram!("stage_l2_us").record(timings.l2 as f64);

        if out.used_l2 {
            crate::batched_counter!("router_l2_trigger_total").increment(1);
        } else {
            crate::batched_counter!("router_l2_trigger_total").increment(0);
        }

        let mut resp = ScoreResponse {
            trace_id: Uuid::new_v4(),
            score: out.final_score as f64,
            decision: out.decision,
            reason: vec![
                ReasonItem {
                    signal: "l1_score".into(),
                    value: out.l1_score as f64,
                    baseline_p95: 0.0,
                    direction: "info".into(),
                },
                ReasonItem {
                    signal: "l2_score".into(),
                    value: out.l2_score.unwrap_or(0.0) as f64,
                    baseline_p95: 0.0,
                    direction: "info".into(),
                },
            ],
            timings_us: timings,
        };

        record_serialize_metrics(&mut resp);
        record_e2e_metrics(t0);
        Ok(resp)
    }

    pub async fn score_quick_dense_bytes_with_standalone_bench_l2_async(
        &self,
        parse_us: u64,
        row_bytes_le: Bytes,
        route_meta: Option<QuickRouteMeta>,
        standalone_l2: &StandaloneL2Runtime,
        sidecar_row_idx: usize,
        tau_mode: StandaloneL2TauMode,
    ) -> anyhow::Result<ScoreResponse> {
        let t0 = Instant::now();

        let mut timings = TimingsUs::default();
        timings.parse = parse_us;

        let quick = self
            .quick
            .as_ref()
            .context("quickscorer not enabled: use new_with_quickscorer_bundle")?;

        let l1_out = quick.predict_l1_only_from_bytes(row_bytes_le.as_ref())?;
        timings.l1 = l1_out.l1_us;
        timings.router = l1_out.router_us;
        crate::sampled_histogram!("stage_l1_us").record(timings.l1 as f64);
        crate::sampled_histogram!("stage_router_us").record(timings.router as f64);

        let (decision, final_score, l2_score) = if l1_out.passed {
            crate::batched_counter!("router_l2_trigger_total").increment(0);
            (crate::schema::Decision::Allow, l1_out.l1_score, None)
        } else {
            crate::batched_counter!("router_l2_trigger_total").increment(1);
            let tau = match tau_mode {
                StandaloneL2TauMode::Request => route_meta
                    .as_ref()
                    .and_then(|m| m.l2_tau_used)
                    .context("benchmark-only standalone L2 mode requires l2_tau_used in request")?,
                StandaloneL2TauMode::Fixed(v) => v,
            };
            let fold_id = route_meta.as_ref().map(|m| m.fold_id).unwrap_or(0);
            let t_l2 = Instant::now();
            let l2_out = TLS_STANDALONE_L2_SCRATCH.with(|cell| -> anyhow::Result<_> {
                let mut slot = cell.borrow_mut();
                if slot.is_none() {
                    *slot = Some(standalone_l2.new_scratch());
                }
                let scratch = slot.as_mut().expect("scratch initialized");
                standalone_l2.predict_l2_row_by_index_with_scratch(
                    sidecar_row_idx % standalone_l2.feat_rows(),
                    tau,
                    fold_id,
                    scratch,
                )
            })?;
            timings.l2 = now_us(t_l2);
            crate::sampled_histogram!("stage_l2_us").record(timings.l2 as f64);
            let decision = if l2_out.reject {
                crate::schema::Decision::Deny
            } else {
                crate::schema::Decision::ManualReview
            };
            (decision, l2_out.score, Some(l2_out.score))
        };

        let mut resp = ScoreResponse {
            trace_id: Uuid::new_v4(),
            score: final_score as f64,
            decision,
            reason: vec![
                ReasonItem {
                    signal: "l1_score".into(),
                    value: l1_out.l1_score as f64,
                    baseline_p95: 0.0,
                    direction: "info".into(),
                },
                ReasonItem {
                    signal: "l2_score".into(),
                    value: l2_score.unwrap_or(0.0) as f64,
                    baseline_p95: 0.0,
                    direction: "info".into(),
                },
            ],
            timings_us: timings,
        };

        record_serialize_metrics(&mut resp);
        record_e2e_metrics(t0);
        Ok(resp)
    }
}
