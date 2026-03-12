mod app;
mod compat_http;
mod completion_window;
mod exp_unix;
mod metrics;
mod score_plane;
mod wire;

use std::sync::Arc;

use anyhow::Context;
use app::{build_batch_app, AppInitConfig};
use clap::Parser;
use compat_http::CompatMode;
use metrics::StageSampler;
use score_plane::{ScorePlane, ScoreRouting};
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug, Clone)]
#[command(name = "tokio-server-batchplane", version, about)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:28091")]
    listen_compat: String,

    #[arg(long, default_value = "/tmp/tokio-server-batchplane.sock")]
    listen_exp_unix: String,

    #[arg(long, value_name = "DIR")]
    bundle_dir: String,

    #[arg(long, default_value_t = 2)]
    io_workers: usize,

    #[arg(long, default_value_t = 4)]
    score_workers: usize,

    #[arg(long, default_value_t = 4096)]
    score_queue_capacity: usize,

    #[arg(long, default_value_t = 4096)]
    max_in_flight: usize,

    #[arg(long, default_value_t = 128)]
    compat_max_inflight_per_conn: usize,

    #[arg(long, default_value_t = 128)]
    exp_max_inflight_per_conn: usize,

    #[arg(long, default_value_t = 4)]
    compat_submit_burst: usize,

    #[arg(long, default_value_t = 4)]
    exp_submit_burst: usize,

    #[arg(long, default_value_t = 8)]
    compat_write_burst: usize,

    #[arg(long, default_value_t = 8)]
    compat_read_budget: usize,

    #[arg(long, default_value_t = 32)]
    compat_completion_budget: usize,

    #[arg(long, default_value_t = 8)]
    exp_write_burst: usize,

    #[arg(long, default_value_t = 0)]
    stage_sample_every: u64,

    #[arg(long, default_value_t = false)]
    strict_single_inflight: bool,

    #[arg(long, value_parser = ["sticky", "round-robin"], default_value = "sticky")]
    score_routing: String,

    #[arg(
        long,
        value_parser = ["sticky-fastpath", "shard-v2", "shard-v3"],
        default_value = "sticky-fastpath"
    )]
    compat_mode: String,

    #[arg(long, value_parser = ["standalone-sidecar"])]
    l2_bench_mode: Option<String>,

    #[arg(long)]
    l2_bench_feat_bin: Option<String>,

    #[arg(long, value_parser = ["request", "fixed"], default_value = "request")]
    l2_bench_tau_mode: String,

    #[arg(long)]
    l2_bench_fixed_tau: Option<f32>,
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,tokio_server_batchplane=info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

async fn async_main(args: Args) -> anyhow::Result<()> {
    init_tracing();
    info!(
        io_workers = args.io_workers,
        score_workers = args.score_workers,
        strict_single_inflight = args.strict_single_inflight,
        score_routing = %args.score_routing,
        compat_mode = %args.compat_mode,
        compat_listen = %args.listen_compat,
        exp_unix = %args.listen_exp_unix,
        "starting tokio-server-batchplane"
    );

    let app = Arc::new(build_batch_app(&AppInitConfig {
        bundle_dir: args.bundle_dir.clone(),
        max_in_flight: args.max_in_flight,
        l2_bench_mode: args.l2_bench_mode.clone(),
        l2_bench_feat_bin: args.l2_bench_feat_bin.clone(),
        l2_bench_tau_mode: args.l2_bench_tau_mode.clone(),
        l2_bench_fixed_tau: args.l2_bench_fixed_tau,
    })?);

    let score_plane = Arc::new(ScorePlane::new(
        args.score_workers.max(1),
        args.score_queue_capacity.max(1),
        match args.score_routing.as_str() {
            "round-robin" => ScoreRouting::RoundRobin,
            _ => ScoreRouting::StickyByConn,
        },
        app.clone(),
    ));
    let stage_sampler = Arc::new(StageSampler::new(args.stage_sample_every));

    tokio::try_join!(
        compat_http::run(
            args.listen_compat.clone(),
            app.clone(),
            score_plane.clone(),
            args.compat_max_inflight_per_conn.max(1),
            args.compat_submit_burst.max(1),
            args.compat_write_burst.max(1),
            args.compat_read_budget.max(1),
            args.compat_completion_budget.max(1),
            args.strict_single_inflight,
            stage_sampler.clone(),
            match args.compat_mode.as_str() {
                "shard-v2" => CompatMode::ShardV2,
                "shard-v3" => CompatMode::ShardV3,
                _ => CompatMode::StickyFastpath,
            },
        ),
        exp_unix::run(
            args.listen_exp_unix.clone(),
            app,
            score_plane,
            args.exp_max_inflight_per_conn.max(1),
            args.exp_submit_burst.max(1),
            args.exp_write_burst.max(1),
            args.strict_single_inflight,
            stage_sampler,
        ),
    )?;

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(args.io_workers.max(1))
        .max_blocking_threads(args.io_workers.max(1))
        .build()
        .context("build tokio runtime")?;
    rt.block_on(async_main(args))
}
