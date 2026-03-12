use std::{path::Path, sync::Arc};

use anyhow::Context;
use axum::http::StatusCode;
use risk_core::{
    config::Config,
    pipeline::{AppCore, StandaloneL2TauMode},
};
use risk_quickscorer_standalone_l2::StandaloneL2Runtime;
use tokio::sync::Semaphore as TokioSemaphore;
use tracing::info;

use crate::wire::{
    batch128_shape_for_content_len, encode_batch_aggregate_ack, parse_batch128_refs,
    validate_batch128_header, Batch128Shape,
};

#[derive(Clone, Copy, Debug)]
pub enum BatchJobKind {
    Score,
    Null,
    ParseOnly,
}

pub type BatchReply = Result<[u8; 40], (StatusCode, String)>;

#[derive(Clone)]
pub struct BatchApp {
    pub core: Arc<AppCore>,
    pub standalone_l2_bench: Option<Arc<StandaloneL2Runtime>>,
    pub standalone_l2_tau_mode: Option<StandaloneL2TauMode>,
    pub bench_in_flight: Arc<TokioSemaphore>,
    pub expected_dim: usize,
}

#[derive(Clone, Debug)]
pub struct AppInitConfig {
    pub bundle_dir: String,
    pub max_in_flight: usize,
    pub l2_bench_mode: Option<String>,
    pub l2_bench_feat_bin: Option<String>,
    pub l2_bench_tau_mode: String,
    pub l2_bench_fixed_tau: Option<f32>,
}

pub fn build_batch_app(cfg: &AppInitConfig) -> anyhow::Result<BatchApp> {
    let core = AppCore::new_with_quickscorer_bundle(Config::default(), &cfg.bundle_dir)
        .context("init AppCore(bundle)")?;

    let standalone_l2_bench = if cfg.l2_bench_mode.as_deref() == Some("standalone-sidecar") {
        let runtime = if let Some(path) = cfg.l2_bench_feat_bin.as_deref() {
            Arc::new(StandaloneL2Runtime::load_with_feat_bin_override(
                Path::new(&cfg.bundle_dir),
                Some(Path::new(path)),
            )?)
        } else {
            Arc::new(StandaloneL2Runtime::load(Path::new(&cfg.bundle_dir))?)
        };
        Some(runtime)
    } else {
        None
    };

    let standalone_l2_tau_mode = if standalone_l2_bench.is_some() {
        Some(match cfg.l2_bench_tau_mode.as_str() {
            "fixed" => StandaloneL2TauMode::Fixed(
                cfg.l2_bench_fixed_tau
                    .context("--l2-bench-fixed-tau required when --l2-bench-tau-mode=fixed")?,
            ),
            _ => StandaloneL2TauMode::Request,
        })
    } else {
        None
    };

    let quick = core
        .quick
        .as_ref()
        .context("quickscorer not enabled: start with --bundle-dir")?;
    let dbg = quick.debug_info();
    info!(
        "QuickScorer enabled: backend={} l1_dim={} l2_dim={} l1_thr={} fold={} gb_target={} segmented={}",
        dbg.backend,
        dbg.l1_dim,
        dbg.l2_dim,
        dbg.l1_threshold,
        dbg.l2_default_fold,
        dbg.l2_gb_target,
        dbg.l2_segmented
    );

    let (expected_dim, _) = core
        .quick_dims()
        .context("quickscorer dims unavailable after bundle load")?;

    Ok(BatchApp {
        core: Arc::new(core),
        standalone_l2_bench,
        standalone_l2_tau_mode,
        bench_in_flight: Arc::new(TokioSemaphore::new(cfg.max_in_flight.max(1))),
        expected_dim,
    })
}

impl BatchApp {
    pub fn expected_shape(&self, body_len: usize) -> Result<Batch128Shape, (StatusCode, String)> {
        batch128_shape_for_content_len(self.expected_dim, body_len).ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                format!(
                    "batchplane only supports fixed batch128 bodies, got content_len={body_len}"
                ),
            )
        })
    }

    pub fn execute_job(&self, kind: BatchJobKind, body: &[u8]) -> BatchReply {
        let shape = self.expected_shape(body.len())?;
        match kind {
            BatchJobKind::Score => self.score_batch128(shape, body),
            BatchJobKind::Null => self.null_batch128(shape, body),
            BatchJobKind::ParseOnly => self.parse_only_batch128(shape, body),
        }
    }

    fn score_batch128(&self, shape: Batch128Shape, body: &[u8]) -> BatchReply {
        let refs = parse_batch128_refs(body, shape.has_route_meta, shape.record_bytes)
            .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
        let _permit = self
            .bench_in_flight
            .clone()
            .try_acquire_owned()
            .map_err(|_| (StatusCode::TOO_MANY_REQUESTS, "overloaded".to_string()))?;

        let (used_l2_count, decision_counts) = if let (Some(rt), Some(tau_mode)) = (
            self.standalone_l2_bench.as_ref(),
            self.standalone_l2_tau_mode,
        ) {
            self.core
                .score_standalone_batch128_http_ack_lite(&refs.rows, &refs.metas, rt, tau_mode)
                .map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("standalone batch128 inference failed: {e:#}"),
                    )
                })?
        } else {
            self.core
                .score_quick_dense_batch128_http_ack_lite(&refs.rows, &refs.metas)
                .map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("quick batch128 inference failed: {e:#}"),
                    )
                })?
        };

        Ok(encode_batch_aggregate_ack(
            128,
            128,
            used_l2_count,
            decision_counts,
        ))
    }

    fn null_batch128(&self, shape: Batch128Shape, body: &[u8]) -> BatchReply {
        validate_batch128_header(body, shape.has_route_meta, shape.record_bytes)
            .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
        Ok(encode_batch_aggregate_ack(128, 128, 0, [128, 0, 0, 0, 0]))
    }

    fn parse_only_batch128(&self, shape: Batch128Shape, body: &[u8]) -> BatchReply {
        validate_batch128_header(body, shape.has_route_meta, shape.record_bytes)
            .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
        let _refs = parse_batch128_refs(body, shape.has_route_meta, shape.record_bytes)
            .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
        Ok(encode_batch_aggregate_ack(128, 128, 0, [128, 0, 0, 0, 0]))
    }
}
