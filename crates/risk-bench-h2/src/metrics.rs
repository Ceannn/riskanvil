use crate::config::{Args, BenchMode, SchedulerMode, WorkloadMode};
use hdrhistogram::Histogram;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Copy, Debug, Serialize)]
pub enum Verdict {
    Pass,
    ClientLimited,
    ServerLimited,
}

#[derive(Default, Clone, Copy)]
pub struct AttemptBatch {
    pub attempted: u64,
    pub dropped_conn_queue_full: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BatchDone {
    pub client_qwait_us: u64,
    pub service_rtt_us: u64,
    pub e2e_us: u64,
    pub ok: u64,
    pub err: u64,
    pub timeout: u64,
    pub http_2xx: u64,
    pub http_429: u64,
    pub http_5xx: u64,
    pub qsb2_samples: u64,
    pub rsk1_samples: u64,
    pub used_l2: u64,
    pub decision_counts: [u64; 5],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WorkerDone {
    pub stream_error_count: u64,
    pub conn_error_count: u64,
    pub reconnect_attempts: u64,
    pub reconnect_budget_exhausted: u64,
    pub force_timeout_all_count: u64,
    pub worker_exit_with_inflight_count: u64,
    pub active_live_connections_max: u64,
    pub client_drop_queue_full: u64,
    pub bench_limited_transitions: u64,
}

pub struct WorkerSnapshot {
    pub attempted: u64,
    pub dropped_conn_queue_full: u64,
    pub ok: u64,
    pub err: u64,
    pub timeout: u64,
    pub http_2xx: u64,
    pub http_429: u64,
    pub http_5xx: u64,
    pub qsb2_samples: u64,
    pub rsk1_samples: u64,
    pub used_l2: u64,
    pub decision_counts: [u64; 5],
    pub batch_client_qwait_us: Histogram<u64>,
    pub batch_service_rtt_us: Histogram<u64>,
    pub batch_e2e_us: Histogram<u64>,
}

impl WorkerSnapshot {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            attempted: 0,
            dropped_conn_queue_full: 0,
            ok: 0,
            err: 0,
            timeout: 0,
            http_2xx: 0,
            http_429: 0,
            http_5xx: 0,
            qsb2_samples: 0,
            rsk1_samples: 0,
            used_l2: 0,
            decision_counts: [0; 5],
            batch_client_qwait_us: new_hist()?,
            batch_service_rtt_us: new_hist()?,
            batch_e2e_us: new_hist()?,
        })
    }

    pub fn reset(&mut self) -> anyhow::Result<()> {
        *self = Self::new()?;
        Ok(())
    }
}

pub enum AggEvent {
    WorkerSnapshot(WorkerSnapshot),
    WorkerDone(WorkerDone),
}

pub struct StatsAgg {
    pub attempted: u64,
    pub ok: u64,
    pub err: u64,
    pub timeout: u64,
    pub dropped: u64,
    pub drop_conn_queue_full: u64,
    pub http_2xx: u64,
    pub http_429: u64,
    pub http_5xx: u64,
    pub qsb2_samples: u64,
    pub rsk1_samples: u64,
    pub used_l2: u64,
    pub decision_counts: [u64; 5],
    pub stream_error_count: u64,
    pub conn_error_count: u64,
    pub reconnect_attempts: u64,
    pub reconnect_budget_exhausted: u64,
    pub force_timeout_all_count: u64,
    pub worker_exit_with_inflight_count: u64,
    pub active_live_connections_max: u64,
    pub client_drop_queue_full: u64,
    pub bench_limited_transitions: u64,
    pub batch_client_qwait_us: Histogram<u64>,
    pub batch_service_rtt_us: Histogram<u64>,
    pub batch_e2e_us: Histogram<u64>,
}

#[derive(Serialize)]
pub struct SummaryJson {
    pub config: SummaryConfig,
    pub counts: SummaryCounts,
    pub batch_latency_us: SummaryBatchLatency,
    pub verdict: Verdict,
    pub client_limited_reasons: Vec<String>,
    pub server_limited_reasons: Vec<String>,
}

#[derive(Serialize)]
pub struct SummaryConfig {
    pub url: String,
    pub mode: BenchMode,
    pub workload: WorkloadMode,
    pub scheduler: SchedulerMode,
    pub rps: u64,
    pub duration_s: u64,
    pub warmup_s: u64,
    pub workers: usize,
    pub worker_cpus: Option<Vec<usize>>,
    pub conns_per_worker: usize,
    pub max_inflight_per_conn: usize,
    pub throughput_batch_size: usize,
    pub batch_records: usize,
    pub payload_rows: usize,
    pub dense_dim: usize,
    pub route_meta: bool,
    pub transport: &'static str,
}

#[derive(Serialize)]
pub struct SummaryCounts {
    pub target_rps: u64,
    pub target_batch_rps: f64,
    pub attempted: u64,
    pub ok: u64,
    pub err: u64,
    pub timeout: u64,
    pub dropped: u64,
    pub drop_conn_queue_full: u64,
    pub under_target_rps: f64,
    pub under_target_pct: f64,
    pub http_2xx: u64,
    pub http_429: u64,
    pub http_5xx: u64,
    pub qsb2_samples: u64,
    pub rsk1_samples: u64,
    pub used_l2: u64,
    pub decision_allow: u64,
    pub decision_deny: u64,
    pub decision_manual_review: u64,
    pub decision_degrade_allow: u64,
    pub decision_unknown: u64,
    pub attempted_rps: f64,
    pub ok_rps: f64,
    pub attempted_batch_rps: f64,
    pub ok_batch_rps: f64,
    pub stream_error_count: u64,
    pub conn_error_count: u64,
    pub reconnect_attempts: u64,
    pub reconnect_budget_exhausted: u64,
    pub force_timeout_all_count: u64,
    pub worker_exit_with_inflight_count: u64,
    pub active_live_connections_max: u64,
    pub client_drop_queue_full: u64,
    pub bench_limited_transitions: u64,
}

#[derive(Serialize)]
pub struct LatQuantiles {
    pub p50: u64,
    pub p95: u64,
    pub p99: u64,
}

#[derive(Serialize)]
pub struct SummaryBatchLatency {
    pub client_qwait: LatQuantiles,
    pub service_rtt: LatQuantiles,
    pub e2e: LatQuantiles,
}

#[derive(Serialize)]
pub struct WindowRow {
    pub elapsed_s: f64,
    pub target_rps: f64,
    pub target_batch_rps: f64,
    pub attempted_rps: f64,
    pub ok_rps: f64,
    pub attempted_batch_rps: f64,
    pub ok_batch_rps: f64,
    pub under_target_rps: f64,
    pub batch_e2e_p99_us: u64,
}

impl StatsAgg {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            attempted: 0,
            ok: 0,
            err: 0,
            timeout: 0,
            dropped: 0,
            drop_conn_queue_full: 0,
            http_2xx: 0,
            http_429: 0,
            http_5xx: 0,
            qsb2_samples: 0,
            rsk1_samples: 0,
            used_l2: 0,
            decision_counts: [0; 5],
            stream_error_count: 0,
            conn_error_count: 0,
            reconnect_attempts: 0,
            reconnect_budget_exhausted: 0,
            force_timeout_all_count: 0,
            worker_exit_with_inflight_count: 0,
            active_live_connections_max: 0,
            client_drop_queue_full: 0,
            bench_limited_transitions: 0,
            batch_client_qwait_us: new_hist()?,
            batch_service_rtt_us: new_hist()?,
            batch_e2e_us: new_hist()?,
        })
    }

    pub fn record_snapshot(&mut self, snapshot: &WorkerSnapshot) -> anyhow::Result<()> {
        self.attempted += snapshot.attempted;
        self.dropped += snapshot.dropped_conn_queue_full;
        self.drop_conn_queue_full += snapshot.dropped_conn_queue_full;
        self.client_drop_queue_full += snapshot.dropped_conn_queue_full;
        self.ok += snapshot.ok;
        self.err += snapshot.err;
        self.timeout += snapshot.timeout;
        self.http_2xx += snapshot.http_2xx;
        self.http_429 += snapshot.http_429;
        self.http_5xx += snapshot.http_5xx;
        self.qsb2_samples += snapshot.qsb2_samples;
        self.rsk1_samples += snapshot.rsk1_samples;
        self.used_l2 += snapshot.used_l2;
        for (dst, src) in self
            .decision_counts
            .iter_mut()
            .zip(snapshot.decision_counts.iter())
        {
            *dst += *src;
        }
        self.batch_client_qwait_us
            .add(&snapshot.batch_client_qwait_us)?;
        self.batch_service_rtt_us
            .add(&snapshot.batch_service_rtt_us)?;
        self.batch_e2e_us.add(&snapshot.batch_e2e_us)?;
        Ok(())
    }

    pub fn record_worker_done(&mut self, done: &WorkerDone) {
        self.stream_error_count += done.stream_error_count;
        self.conn_error_count += done.conn_error_count;
        self.reconnect_attempts += done.reconnect_attempts;
        self.reconnect_budget_exhausted += done.reconnect_budget_exhausted;
        self.force_timeout_all_count += done.force_timeout_all_count;
        self.worker_exit_with_inflight_count += done.worker_exit_with_inflight_count;
        self.active_live_connections_max =
            self.active_live_connections_max.max(done.active_live_connections_max);
        self.client_drop_queue_full += done.client_drop_queue_full;
        self.bench_limited_transitions += done.bench_limited_transitions;
    }

    pub fn reset_window(&mut self) -> anyhow::Result<()> {
        *self = Self::new()?;
        Ok(())
    }
}

pub fn write_window_header(path: &PathBuf) -> anyhow::Result<File> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "elapsed_s,target_rps,target_batch_rps,attempted_rps,ok_rps,attempted_batch_rps,ok_batch_rps,under_target_rps,batch_e2e_p99_us"
    )?;
    Ok(file)
}

pub fn write_window_row(file: &mut File, row: &WindowRow) -> anyhow::Result<()> {
    writeln!(
        file,
        "{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}",
        row.elapsed_s,
        row.target_rps,
        row.target_batch_rps,
        row.attempted_rps,
        row.ok_rps,
        row.attempted_batch_rps,
        row.ok_batch_rps,
        row.under_target_rps,
        row.batch_e2e_p99_us
    )?;
    Ok(())
}

pub fn make_window_row(
    stats: &StatsAgg,
    elapsed: Duration,
    target_rps: u64,
    batch_records: usize,
) -> WindowRow {
    let secs = elapsed.as_secs_f64().max(1e-9);
    let attempted_rps = stats.attempted as f64 / secs;
    let target_rps_f = target_rps as f64;
    let batch_rps = stats.batch_e2e_us.len() as f64 / secs;
    let batch_div = batch_records.max(1) as f64;
    WindowRow {
        elapsed_s: secs,
        target_rps: target_rps_f,
        target_batch_rps: target_rps_f / batch_div,
        attempted_rps,
        ok_rps: stats.ok as f64 / secs,
        attempted_batch_rps: batch_rps,
        ok_batch_rps: batch_rps,
        under_target_rps: (target_rps_f - attempted_rps).max(0.0),
        batch_e2e_p99_us: hist_q(&stats.batch_e2e_us, 0.99),
    }
}

pub fn build_summary(
    args: &Args,
    worker_cpus: Option<Vec<usize>>,
    payload_rows: usize,
    dense_dim: usize,
    route_meta: bool,
    total: &StatsAgg,
) -> SummaryJson {
    let elapsed = Duration::from_secs(args.duration.max(1));
    let (verdict, client_limited_reasons, server_limited_reasons) = verdict_for(args, total);
    let attempted_rps = total.attempted as f64 / elapsed.as_secs_f64();
    let ok_rps = total.ok as f64 / elapsed.as_secs_f64();
    let batch_div = args.batch_records.max(1) as f64;
    let under_target_rps = (args.qps as f64 - attempted_rps).max(0.0);
    let under_target_pct = if args.qps == 0 {
        0.0
    } else {
        under_target_rps / args.qps as f64
    };
    SummaryJson {
        config: SummaryConfig {
            url: args.url.clone(),
            mode: args.mode,
            workload: args.workload,
            scheduler: args.scheduler,
            rps: args.qps,
            duration_s: args.duration,
            warmup_s: args.warmup,
            workers: args.workers,
            worker_cpus,
            conns_per_worker: args.conns_per_worker,
            max_inflight_per_conn: args.max_inflight_per_conn,
            throughput_batch_size: args.throughput_batch_size,
            batch_records: args.batch_records,
            payload_rows,
            dense_dim,
            route_meta,
            transport: "h2c_http2",
        },
        counts: SummaryCounts {
            target_rps: args.qps,
            target_batch_rps: args.qps as f64 / batch_div,
            attempted: total.attempted,
            ok: total.ok,
            err: total.err,
            timeout: total.timeout,
            dropped: total.dropped,
            drop_conn_queue_full: total.drop_conn_queue_full,
            under_target_rps,
            under_target_pct,
            http_2xx: total.http_2xx,
            http_429: total.http_429,
            http_5xx: total.http_5xx,
            qsb2_samples: total.qsb2_samples,
            rsk1_samples: total.rsk1_samples,
            used_l2: total.used_l2,
            decision_allow: total.decision_counts[0],
            decision_deny: total.decision_counts[1],
            decision_manual_review: total.decision_counts[2],
            decision_degrade_allow: total.decision_counts[3],
            decision_unknown: total.decision_counts[4],
            attempted_rps,
            ok_rps,
            attempted_batch_rps: attempted_rps / batch_div,
            ok_batch_rps: ok_rps / batch_div,
            stream_error_count: total.stream_error_count,
            conn_error_count: total.conn_error_count,
            reconnect_attempts: total.reconnect_attempts,
            reconnect_budget_exhausted: total.reconnect_budget_exhausted,
            force_timeout_all_count: total.force_timeout_all_count,
            worker_exit_with_inflight_count: total.worker_exit_with_inflight_count,
            active_live_connections_max: total.active_live_connections_max,
            client_drop_queue_full: total.client_drop_queue_full,
            bench_limited_transitions: total.bench_limited_transitions,
        },
        batch_latency_us: SummaryBatchLatency {
            client_qwait: quantiles(&total.batch_client_qwait_us),
            service_rtt: quantiles(&total.batch_service_rtt_us),
            e2e: quantiles(&total.batch_e2e_us),
        },
        verdict,
        client_limited_reasons,
        server_limited_reasons,
    }
}

pub fn write_summary(path: &PathBuf, summary: &SummaryJson) -> anyhow::Result<()> {
    let text = serde_json::to_string_pretty(summary)?;
    std::fs::write(path, text)?;
    Ok(())
}

fn new_hist() -> anyhow::Result<Histogram<u64>> {
    Histogram::new_with_bounds(1, 60_000_000, 3).map_err(Into::into)
}

fn hist_q(hist: &Histogram<u64>, q: f64) -> u64 {
    if hist.len() == 0 {
        0
    } else {
        hist.value_at_quantile(q)
    }
}

fn quantiles(hist: &Histogram<u64>) -> LatQuantiles {
    LatQuantiles {
        p50: hist_q(hist, 0.50),
        p95: hist_q(hist, 0.95),
        p99: hist_q(hist, 0.99),
    }
}

fn verdict_for(args: &Args, total: &StatsAgg) -> (Verdict, Vec<String>, Vec<String>) {
    let mut client_limited = Vec::new();
    let mut server_limited = Vec::new();
    if args.scheduler == SchedulerMode::SoftPace {
        let target_total = args.qps.max(1) as f64 * args.duration.max(1) as f64;
        let attempted_ratio = (total.attempted as f64) / target_total;
        if attempted_ratio < 0.95 {
            client_limited.push("under_target".to_string());
        }
    }
    if total.drop_conn_queue_full > 0 {
        client_limited.push("drop_conn_queue_full".to_string());
    }
    if total.reconnect_budget_exhausted > 0 {
        client_limited.push("reconnect_budget_exhausted".to_string());
    }
    if hist_q(&total.batch_client_qwait_us, 0.99) > hist_q(&total.batch_service_rtt_us, 0.99) * 2
        && hist_q(&total.batch_client_qwait_us, 0.99) > 1_000
    {
        client_limited.push("client_qwait_p99_gt_service_rtt_p99".to_string());
    }
    if total.http_429 > 0 {
        server_limited.push("http_429".to_string());
    }
    if total.http_5xx > 0 {
        server_limited.push("http_5xx".to_string());
    }
    if total.timeout > total.ok / 10 && total.timeout > 100 {
        server_limited.push("timeouts".to_string());
    }
    if !server_limited.is_empty() {
        return (Verdict::ServerLimited, client_limited, server_limited);
    }
    if !client_limited.is_empty() {
        return (Verdict::ClientLimited, client_limited, server_limited);
    }
    (Verdict::Pass, client_limited, server_limited)
}
