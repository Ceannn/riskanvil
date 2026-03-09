use bytes::{Bytes, BytesMut};
use clap::Parser;
use futures_lite::{AsyncReadExt, AsyncWriteExt};
use glommio::net::TcpListener;
use glommio::sync::{Permit, Semaphore};
use glommio::{LocalExecutorBuilder, Placement};

use std::io::Write as _;

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use risk_core::{
    config::Config, pipeline::{AppCore, StandaloneL2TauMode}, quickscorer::QuickRouteMeta, schema::Decision, util::now_us,
};
use risk_quickscorer_standalone_l2::StandaloneL2Runtime;

use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use std::collections::HashSet;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod mamba_runtime;
use mamba_runtime::{
    mlockall, CircuitBreakerConfig, FallbackMode, HugepagePolicy, MambaRuntime, MemoryPolicy,
};

static TRACE_ID_SEQ: AtomicU64 = AtomicU64::new(1);
const GLOMMIO_IO_MEMORY_BYTES: usize = 1 << 20;
const GLOMMIO_CONN_STASH_BYTES: usize = 8 * 1024;
const GLOMMIO_CONN_READ_BUF_BYTES: usize = 8 * 1024;
const GLOMMIO_CONN_OUT_BYTES: usize = 512;
const GLOMMIO_CONN_STASH_RETAIN_LIMIT_BYTES: usize = 32 * 1024;
const GLOMMIO_CONN_OUT_RETAIN_LIMIT_BYTES: usize = 4 * 1024;

#[derive(Clone, Debug)]
struct DenseRequest {
    payload: Bytes,
    route_meta: Option<QuickRouteMeta>,
}

#[derive(Clone)]
struct AppState {
    core: Arc<AppCore>,
    prom: PrometheusHandle,
    in_flight: Arc<AtomicUsize>,
    max_in_flight: usize,
    standalone_l2_bench: Option<Arc<StandaloneL2Runtime>>,
    standalone_l2_tau_mode: Option<StandaloneL2TauMode>,
    mamba: Option<Arc<MambaRuntime>>,
    mamba_policy_hash: String,
}

#[derive(Debug, Deserialize)]
struct MambaRequest {
    input_ids: Vec<i64>,
    #[serde(default, rename = "static")]
    static_feat: Vec<f32>,
    #[serde(default)]
    static_total: Vec<f32>,
    trace_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct MambaTimingsUs {
    parse: u64,
    queue: u64,
    assemble: u64,
    kernel: u64,
    post: u64,
    serialize: u64,
    other: u64,
}

impl Default for MambaTimingsUs {
    fn default() -> Self {
        Self {
            parse: 0,
            queue: 0,
            assemble: 0,
            kernel: 0,
            post: 0,
            serialize: 0,
            other: 0,
        }
    }
}

#[derive(Debug, Serialize)]
struct MambaResponse {
    trace_id: String,
    margin: f32,
    logits: [f32; 2],
    dispatch: String,
    math_backend: String,
    schema_hash: String,
    policy_hash: String,
    fallback: Option<String>,
    timings_us: MambaTimingsUs,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about)]
struct Args {
    /// 监听地址
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: String,

    /// 可选：启用 QuickScorer bundle（目录根）
    #[arg(long)]
    bundle_dir: Option<String>,

    /// HTTP 层最大并发（超出直接 429），用于保护服务端排队不会失控
    #[arg(long, default_value_t = 4096)]
    max_in_flight: usize,

    /// 启动后预热次数（保留占位；当前 QuickScorer 无显式 warmup）
    #[arg(long, default_value_t = 100)]
    warmup_iters: usize,

    /// Benchmark-only synthetic L2 mode: standalone-sidecar
    #[arg(long, value_parser = ["standalone-sidecar"])]
    l2_bench_mode: Option<String>,

    /// Optional override for standalone L2 89-dim feat_bin
    #[arg(long)]
    l2_bench_feat_bin: Option<String>,

    /// Tau source for benchmark-only standalone L2: request|fixed
    #[arg(long, value_parser = ["request", "fixed"], default_value = "request")]
    l2_bench_tau_mode: String,

    /// Fixed tau used when --l2-bench-tau-mode=fixed
    #[arg(long)]
    l2_bench_fixed_tau: Option<f32>,

    /// 可选：启用 Mamba stateless 推理（bundle 目录）
    #[arg(long)]
    mamba_bundle_dir: Option<String>,

    /// Mamba dispatch: auto|scalar|avx2|avx512
    #[arg(long, default_value = "auto")]
    mamba_dispatch: String,

    /// Mamba math backend: fast|fast_wild|exact
    #[arg(long, default_value = "fast")]
    mamba_math: String,

    /// Mamba scratch pool size (per-process)
    #[arg(long, default_value_t = 2)]
    mamba_scratch_pool: usize,

    /// Mamba auto-dispatch microbench iters
    #[arg(long, default_value_t = 20)]
    mamba_auto_iters: usize,

    /// Mamba auto-dispatch warmup iters
    #[arg(long, default_value_t = 5)]
    mamba_auto_warmup: usize,

    /// Circuit breaker max consecutive errors
    #[arg(long, default_value_t = 5)]
    mamba_cb_max_errors: u64,

    /// Circuit breaker error rate threshold (0..1)
    #[arg(long, default_value_t = 0.05)]
    mamba_cb_error_rate: f64,

    /// Circuit breaker window (ms)
    #[arg(long, default_value_t = 1000)]
    mamba_cb_window_ms: u64,

    /// Circuit breaker trip duration (ms)
    #[arg(long, default_value_t = 5000)]
    mamba_cb_trip_ms: u64,

    /// Fallback strategy: exact|zero
    #[arg(long, default_value = "exact", value_parser = ["exact", "zero"])]
    mamba_fallback: String,

    /// Enable mlockall(MCL_CURRENT|MCL_FUTURE) for the process
    #[arg(long, default_value_t = false)]
    mamba_mlock: bool,

    /// Fail startup if mlockall fails
    #[arg(long, default_value_t = false)]
    mamba_mlock_required: bool,

    /// Pretouch weights + scratch to reduce page faults
    #[arg(long, default_value_t = false)]
    mamba_pretouch: bool,

    /// madvise hugepage policy: never|madvise
    #[arg(long, default_value = "never", value_parser = ["never", "madvise"])]
    mamba_hugepage: String,

    /// madvise(MADV_WILLNEED) on weights + scratch
    #[arg(long, default_value_t = false)]
    mamba_willneed: bool,
}
fn main() -> anyhow::Result<()> {
    // tracing

    let args = Args::parse();
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("info".parse().unwrap()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // metrics recorder（进程内全局一次）
    let prom = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install prometheus recorder");

    // Pre-register key router metrics so `/metrics | grep router_l2` works even before the first hit.
    risk_core::batched_counter!("router_l2_trigger_total").increment(0);
    risk_core::batched_counter!("router_l2_skipped_budget_total").increment(0);
    risk_core::batched_counter!("router_l2_skipped_deadline_budget_total").increment(0);
    risk_core::batched_counter!("router_l2_skipped_rate_total").increment(0);
    risk_core::batched_counter!("router_l2_skipped_sample_total").increment(0);
    risk_core::batched_counter!("router_l2_skipped_waterline_total").increment(0);
    risk_core::batched_counter!("router_timeout_before_l2_total").increment(0);
    risk_core::batched_counter!("router_deadline_miss_total").increment(0);
    risk_core::batched_counter!("router_l2_feedback_overload_total").increment(0);
    risk_core::batched_counter!("router_l2_feedback_relax_total").increment(0);

    let mut cfg = Config::default();

    // 与 tokio server 保持一致：允许用环境变量覆盖端到端预算（用于超时/降级/背压）。
    if let Ok(v) = std::env::var("SLO_P99_MS") {
        if let Ok(ms) = v.parse::<u64>() {
            cfg.slo_p99_ms = ms;
            tracing::info!("SLO_P99_MS override: {} ms", ms);
        }
    }

    let core = if let Some(dir) = args.bundle_dir.as_deref() {
        tracing::info!("QuickScorer enabled, loading bundle_dir={}", dir);
        Arc::new(AppCore::new_with_quickscorer_bundle(cfg, dir)?)
    } else {
        tracing::info!("QuickScorer disabled");
        Arc::new(AppCore::new(cfg))
    };

    let mamba_policy_hash = read_policy_hash().unwrap_or_else(|e| {
        tracing::warn!("mamba policy hash unavailable: {}", e);
        "none".to_string()
    });

    let mamba = if let Some(dir) = args.mamba_bundle_dir.as_deref() {
        tracing::info!(
            "mamba enabled, bundle_dir={} dispatch={} math={}",
            dir,
            args.mamba_dispatch,
            args.mamba_math
        );
        if args.mamba_mlock {
            match mlockall() {
                Ok(()) => tracing::info!("mamba mlockall enabled"),
                Err(e) => {
                    if args.mamba_mlock_required {
                        return Err(anyhow::anyhow!(e));
                    }
                    tracing::warn!("mlockall failed: {}", e);
                }
            }
        }
        let hugepage = match args.mamba_hugepage.as_str() {
            "madvise" => HugepagePolicy::Madvise,
            _ => HugepagePolicy::Never,
        };
        let mem_policy = MemoryPolicy {
            pretouch: args.mamba_pretouch,
            hugepage,
            willneed: args.mamba_willneed,
        };
        let cb_cfg = CircuitBreakerConfig {
            max_errors: args.mamba_cb_max_errors,
            error_rate: args.mamba_cb_error_rate,
            window_ms: args.mamba_cb_window_ms,
            trip_ms: args.mamba_cb_trip_ms,
        };
        let fallback = match args.mamba_fallback.as_str() {
            "exact" => FallbackMode::Exact,
            "zero" => FallbackMode::DefaultZero,
            _ => FallbackMode::Exact,
        };
        let runtime = MambaRuntime::load(
            Path::new(dir),
            &args.mamba_dispatch,
            &args.mamba_math,
            args.mamba_scratch_pool,
            args.mamba_auto_iters,
            args.mamba_auto_warmup,
            mem_policy,
            cb_cfg,
            fallback,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        tracing::info!(
            "mamba ready: schema_hash={} dispatch={} math={} kernel_path={} layout={} mlp_impl={}",
            runtime.schema_hash(),
            runtime.dispatch_label(),
            runtime.math_label(),
            runtime.kernel_path_label(),
            runtime.layout_label(),
            runtime.mlp_impl_label()
        );
        Some(Arc::new(runtime))
    } else {
        None
    };

    let state = AppState {
        core,
        prom,
        in_flight: Arc::new(AtomicUsize::new(0)),
        max_in_flight: args.max_in_flight,
        standalone_l2_bench: if args.l2_bench_mode.as_deref() == Some("standalone-sidecar") {
            Some(if let Some(path) = args.l2_bench_feat_bin.as_deref() {
                Arc::new(StandaloneL2Runtime::load_with_feat_bin_override(
                    Path::new(args.bundle_dir.as_deref().unwrap_or_default()),
                    Some(Path::new(path)),
                )?)
            } else {
                Arc::new(StandaloneL2Runtime::load(
                    Path::new(args.bundle_dir.as_deref().unwrap_or_default()),
                )?)
            })
        } else {
            None
        },
        standalone_l2_tau_mode: if args.l2_bench_mode.as_deref() == Some("standalone-sidecar") {
            Some(match args.l2_bench_tau_mode.as_str() {
                "fixed" => StandaloneL2TauMode::Fixed(
                    args.l2_bench_fixed_tau.ok_or_else(|| {
                        anyhow::anyhow!(
                            "--l2-bench-fixed-tau is required when --l2-bench-tau-mode=fixed"
                        )
                    })?,
                ),
                _ => StandaloneL2TauMode::Request,
            })
        } else {
            None
        },
        mamba,
        mamba_policy_hash,
    };

    // ====== Startup: print QuickScorer backend/dim/threshold ======
    if let Some(q) = state.core.quick.as_ref() {
        let dbg = q.debug_info();
        tracing::info!(
            "QuickScorer loaded: backend={} l1_dim={} l2_dim={} l1_thr={} fold={} gb_target={} segmented={} seg_cols={:?}",
            dbg.backend,
            dbg.l1_dim,
            dbg.l2_dim,
            dbg.l1_threshold,
            dbg.l2_default_fold,
            dbg.l2_gb_target,
            dbg.l2_segmented,
            dbg.l2_seg_cols
        );
    } else if args.bundle_dir.is_some() {
        tracing::warn!("QuickScorer not enabled");
    }

    // QuickScorer currently has no explicit warmup hook.

    if let Ok(cpus) = get_self_affinity_cpus() {
        tracing::info!("process affinity cpus={:?}", cpus);
    }

    // ✅ thread-per-core：优先在“当前进程允许的 CPU 集合”里，扣掉 XGB pool pin 过的 CPU，剩下的给 Glommio IO executors。
    //    如果扣完为空，则退化为使用全部允许 CPU（runtime 与 XGB 可能共享核，但至少不会跑到 taskset 之外）。
    let io_cpus = select_glommio_io_cpus_from_env()?;
    let shards = io_cpus.len().max(1);
    let per_core_in_flight = std::cmp::max(1, (args.max_in_flight + shards - 1) / shards);

    tracing::info!(
        "glommio thread-per-core: io_shards={} io_cpus={:?} global_max_in_flight={} per_core_max_in_flight={}",
        shards,
        io_cpus,
        args.max_in_flight,
        per_core_in_flight
    );

    let addr = args.listen.clone();
    let mut handles = Vec::new();

    if io_cpus.is_empty() {
        // 极端兜底：无法解析 affinity / cpu 列表
        let st = state.clone();
        let addr2 = addr.clone();
        let per_core_in_flight2 = per_core_in_flight as u64;
        let h = LocalExecutorBuilder::new(Placement::Unbound)
            .name("risk-glommio-io")
            // This server is network-bound; the default 10 MiB registered buffer
            // per shard needlessly burns RLIMIT_MEMLOCK and triggers io_uring
            // ENOMEM warnings under multi-shard startup.
            .io_memory(GLOMMIO_IO_MEMORY_BYTES)
            .spawn(move || async move {
                run_accept_loop(
                    addr2,
                    st,
                    Rc::new(Semaphore::new(per_core_in_flight2)),
                    true,
                )
                .await
            })
            .unwrap();
        handles.push(h);
    } else {
        for (idx, cpu) in io_cpus.into_iter().enumerate() {
            let st = state.clone();
            let addr2 = addr.clone();
            let per_core_in_flight2 = per_core_in_flight as u64;
            let name = format!("risk-glommio-io-cpu{}", cpu);
            let h = LocalExecutorBuilder::new(Placement::Fixed(cpu))
                .name(&name)
                // Keep registered io_uring buffers small; this binary does not
                // do heavy storage I/O and otherwise 6 shards can exceed the
                // default memlock budget on WSL/Linux.
                .io_memory(GLOMMIO_IO_MEMORY_BYTES)
                .spawn(move || async move {
                    // glommio::net::TcpListener::bind() 会设置 SO_REUSEPORT，允许多 executor 同地址并行 accept。
                    run_accept_loop(
                        addr2,
                        st,
                        Rc::new(Semaphore::new(per_core_in_flight2)),
                        idx == 0,
                    )
                    .await
                })
                .unwrap();
            handles.push(h);
        }
    }

    // 阻塞主线程（任意一个 executor panic/退出都会让 join 返回，便于暴露问题）
    for h in handles {
        h.join().unwrap();
    }

    Ok(())
}

async fn run_accept_loop(addr: String, state: AppState, core_sem: Rc<Semaphore>, log_listen: bool) {
    let listener = TcpListener::bind(addr.as_str()).expect("bind failed");
    if log_listen {
        tracing::info!("risk-server-glommio listening on http://{}", addr);
    }

    loop {
        match listener.accept().await {
            Ok(stream) => {
                let _ = stream.set_nodelay(true);

                let st = state.clone();
                let sem = Rc::clone(&core_sem);
                glommio::spawn_local(async move {
                    if let Err(e) = handle_conn(stream, st, sem).await {
                        tracing::debug!("conn error: {}", e);
                    }
                })
                .detach();
            }
            Err(e) => {
                tracing::warn!("accept error: {}", e);
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ReqMeta {
    method: String,
    path: String,
    content_len: Option<usize>,
    chunked: bool,
    expect_100: bool,
    want_close: bool,
}

async fn handle_conn(
    mut stream: glommio::net::TcpStream,
    st: AppState,
    core_sem: Rc<Semaphore>,
) -> anyhow::Result<()> {
    let mut stash: BytesMut = BytesMut::with_capacity(GLOMMIO_CONN_STASH_BYTES);
    let mut buf = vec![0u8; GLOMMIO_CONN_READ_BUF_BYTES];
    // 连接级复用：避免每个请求都重新分配响应 buffer
    let mut out: Vec<u8> = Vec::with_capacity(GLOMMIO_CONN_OUT_BYTES);

    loop {
        // 1) 读到 header 结束
        while find_double_crlf(stash.as_ref()).is_none() {
            let n = stream.read(&mut buf).await?;
            if n == 0 {
                return Ok(());
            }
            stash.extend_from_slice(&buf[..n]);
            if stash.len() > 1024 * 1024 {
                anyhow::bail!("header too large");
            }
        }

        let header_end = find_double_crlf(stash.as_ref()).unwrap();

        // 2) 解析 header（用短作用域避免借用冲突）
        let meta = {
            let head = std::str::from_utf8(&stash[..header_end])?;
            parse_request_head(head)
        };

        // 3) 如果有 Expect: 100-continue，先回 100
        if meta.expect_100 {
            stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").await?;
            stream.flush().await?;
        }

        let body_start = header_end + 4;

        // 4) 读 body（content-length 或 chunked）
        let body: Bytes = if meta.chunked {
            // stash 里可能还没有完整 chunked body，继续读直到能解析
            loop {
                if let Some((end, body_vec)) = try_parse_chunked(stash.as_ref(), body_start) {
                    // drop consumed bytes (header + chunks)
                    let _ = stash.split_to(end);
                    break Bytes::from(body_vec);
                }
                let n = stream.read(&mut buf).await?;
                if n == 0 {
                    anyhow::bail!("eof while reading chunked body");
                }
                stash.extend_from_slice(&buf[..n]);
            }
        } else {
            let need = meta.content_len.unwrap_or(0);
            while stash.len() < body_start + need {
                let n = stream.read(&mut buf).await?;
                if n == 0 {
                    anyhow::bail!("eof while reading body");
                }
                stash.extend_from_slice(&buf[..n]);
            }

            if need == 0 {
                // drop header only
                let _ = stash.split_to(body_start);
                Bytes::new()
            } else {
                // Split without copying: [header + body] is removed from stash, leaving keep-alive tail.
                let total = body_start + need;
                let mut head_body = stash.split_to(total);
                head_body.split_off(body_start).freeze()
            }
        };

        // 6) 路由
        let keep_alive = !meta.want_close;

        if meta.method == "GET" && meta.path == "/health" {
            write_http(&mut stream, &mut out, 200, "text/plain", b"ok", keep_alive).await?;
        } else if meta.method == "GET" && meta.path == "/debug/backend" {
            // Lightweight introspection for verifying L1/L2 wiring, schema, and thresholds.
            let quick = st.core.quick.as_ref();
            let l2_ctrl = st.core.l2_ctrl();

            let body = json!({
                "quickscorer": quick.map(|q| q.debug_info()),
                "router_l2": {
                    "sample_ratio": l2_ctrl.sample_ratio(),
                    "sample_base_ratio": l2_ctrl.sample_base_ratio(),
                    "sample_dyn_ratio": l2_ctrl.sample_dyn_ratio(),
                    "waterline_target": l2_ctrl.sample_waterline_target(),
                    "waterline_hi": l2_ctrl.sample_waterline_hi(),
                    "waterline_lo": l2_ctrl.sample_waterline_lo(),
                },
            });
            let buf = serde_json::to_vec(&body)?;
            write_http(
                &mut stream,
                &mut out,
                200,
                "application/json",
                &buf,
                keep_alive,
            )
            .await?;
        } else if meta.method == "POST" && meta.path == "/score_mamba" {
            let t0 = std::time::Instant::now();
            let t_parse = std::time::Instant::now();

            let req: MambaRequest = match serde_json::from_slice(body.as_ref()) {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!("invalid json body: {}", e);
                    write_http(
                        &mut stream,
                        &mut out,
                        400,
                        "text/plain",
                        msg.as_bytes(),
                        keep_alive,
                    )
                    .await?;
                    continue;
                }
            };
            let mut timings = MambaTimingsUs::default();
            timings.parse = now_us(t_parse);

            let mamba = match st.mamba.as_ref() {
                Some(m) => m.clone(),
                None => {
                    write_http(
                        &mut stream,
                        &mut out,
                        503,
                        "text/plain",
                        b"mamba not enabled",
                        keep_alive,
                    )
                    .await?;
                    continue;
                }
            };

            let t_queue = std::time::Instant::now();
            let (_g, _p) = match try_acquire_inflight(&st, core_sem.as_ref()) {
                Some(g) => g,
                None => {
                    write_http(
                        &mut stream,
                        &mut out,
                        429,
                        "text/plain",
                        b"overloaded",
                        keep_alive,
                    )
                    .await?;
                    continue;
                }
            };
            timings.queue = now_us(t_queue);

            let t_assemble = std::time::Instant::now();
            let static_total = if !req.static_total.is_empty() {
                req.static_total
            } else {
                req.static_feat
            };
            if req.input_ids.len() != mamba.input_ids_len()
                || static_total.len() != mamba.static_dim_total()
            {
                let msg = format!(
                    "input shape mismatch: input_ids={} static_total={}",
                    req.input_ids.len(),
                    static_total.len()
                );
                write_http(
                    &mut stream,
                    &mut out,
                    400,
                    "text/plain",
                    msg.as_bytes(),
                    keep_alive,
                )
                .await?;
                continue;
            }
            timings.assemble = now_us(t_assemble);

            let t_kernel = std::time::Instant::now();
            let mamba_out = match mamba.score(&req.input_ids, &static_total) {
                Ok(v) => v,
                Err(e) => {
                    let msg = format!("mamba error: {}", e);
                    write_http(
                        &mut stream,
                        &mut out,
                        500,
                        "text/plain",
                        msg.as_bytes(),
                        keep_alive,
                    )
                    .await?;
                    continue;
                }
            };
            timings.kernel = now_us(t_kernel);

            let t_post = std::time::Instant::now();
            let trace_id = req
                .trace_id
                .unwrap_or_else(|| TRACE_ID_SEQ.fetch_add(1, Ordering::Relaxed).to_string());
            let mut resp = MambaResponse {
                trace_id,
                margin: mamba_out.margin,
                logits: mamba_out.logits,
                dispatch: mamba_out.dispatch,
                math_backend: mamba_out.math,
                schema_hash: mamba.schema_hash().to_string(),
                policy_hash: st.mamba_policy_hash.clone(),
                fallback: mamba_out.fallback_reason,
                timings_us: MambaTimingsUs::default(),
            };
            timings.post = now_us(t_post);

            let t_ser = std::time::Instant::now();
            let _body = serde_json::to_vec(&resp)?;
            timings.serialize = now_us(t_ser);

            let total_us = now_us(t0);
            let known = timings.parse
                + timings.assemble
                + timings.kernel
                + timings.post
                + timings.serialize;
            timings.other = total_us.saturating_sub(known);
            resp.timings_us = timings;

            let body = serde_json::to_vec(&resp)?;
            write_http(
                &mut stream,
                &mut out,
                200,
                "application/json",
                &body,
                keep_alive,
            )
            .await?;
        } else if meta.method == "POST" && meta.path == "/score_dense_f32_bin" {
            // ✅ bench2/bench3 主压测口：dense f32le -> RSK1 (48B)
            let t0 = std::time::Instant::now();

            // HTTP 层背压：超出并发上限就直接 429
            let (_g, _p) = match try_acquire_inflight(&st, core_sem.as_ref()) {
                Some(g) => g,
                None => {
                    write_http(
                        &mut stream,
                        &mut out,
                        429,
                        "text/plain",
                        b"overloaded",
                        keep_alive,
                    )
                    .await?;
                    continue;
                }
            };

            let (dim, _) = match st.core.quick_dims() {
                Some(v) => v,
                None => {
                    write_http(
                        &mut stream,
                        &mut out,
                        500,
                        "text/plain",
                        b"quickscorer not enabled",
                        keep_alive,
                    )
                    .await?;
                    continue;
                }
            };

            // parse payload (raw bytes or RVEC header)
            let t_parse = std::time::Instant::now();
            let req = match parse_dense_payload_le(&body, dim) {
                Ok(p) => p,
                Err(msg) => {
                    write_http(
                        &mut stream,
                        &mut out,
                        400,
                        "text/plain",
                        msg.as_bytes(),
                        keep_alive,
                    )
                    .await?;
                    continue;
                }
            };
            let parse_us = now_us(t_parse);

            let trace_id = TRACE_ID_SEQ.fetch_add(1, Ordering::Relaxed);
            let sidecar_row_idx = req
                .route_meta
                .as_ref()
                .map(|m| m.row_idx as usize)
                .unwrap_or(trace_id as usize);
            let resp = match if let (Some(rt), Some(tau_mode)) =
                (st.standalone_l2_bench.as_ref(), st.standalone_l2_tau_mode)
            {
                st.core
                    .score_quick_dense_bytes_with_standalone_bench_l2_async(
                        parse_us,
                        req.payload,
                        req.route_meta,
                        rt,
                        sidecar_row_idx,
                        tau_mode,
                    )
                    .await
            } else {
                st.core
                    .score_quick_dense_bytes_with_meta_async(parse_us, req.payload, req.route_meta)
                    .await
            } {
                Ok(resp) => resp,
                Err(e) => {
                    let msg = format!("quickscorer inference failed: {:#}", e);
                    write_http(
                        &mut stream,
                        &mut out,
                        500,
                        "text/plain",
                        msg.as_bytes(),
                        keep_alive,
                    )
                    .await?;
                    continue;
                }
            };
            let decision_u8 = match resp.decision {
                Decision::Allow => 0,
                Decision::Deny => 1,
                Decision::ManualReview => 2,
                Decision::DegradeAllow => 3,
            };

            // serialize (RSK1)
            let ser_hist = risk_core::sampled_histogram!("stage_serialize_us");
            // NOTE: 序列化耗时本身不能“精确地”写回同一个包（会有轻微递归依赖）。
            // 这里用两步：先用 serialize_us=0 编码，然后把测得的 serialize_us patch 回固定 offset(44..48)。
            let (mut rsk1, ser_us) = if ser_hist.enabled() {
                let t_ser = std::time::Instant::now();
                let rsk1 = encode_rsk1(
                    trace_id,
                    resp.score as f32,
                    decision_u8,
                    resp.timings_us.parse,
                    resp.timings_us.feature,
                    resp.timings_us.router,
                    resp.timings_us.l1,
                    resp.timings_us.l2,
                    0,
                );
                (rsk1, now_us(t_ser))
            } else {
                (
                    encode_rsk1(
                        trace_id,
                        resp.score as f32,
                        decision_u8,
                        resp.timings_us.parse,
                        resp.timings_us.feature,
                        resp.timings_us.router,
                        resp.timings_us.l1,
                        resp.timings_us.l2,
                        0,
                    ),
                    0,
                )
            };
            if ser_hist.enabled() {
                patch_rsk1_serialize_us(&mut rsk1, ser_us);
                ser_hist.record(ser_us as f64);
            }
            risk_core::sampled_histogram!("e2e_us").record(now_us(t0) as f64);

            write_http(
                &mut stream,
                &mut out,
                200,
                "application/octet-stream",
                &rsk1,
                keep_alive,
            )
            .await?;
        } else if meta.method == "POST" && meta.path == "/score_dense_f32_bin_v2" {
            let t0 = std::time::Instant::now();

            let (_g, _p) = match try_acquire_inflight(&st, core_sem.as_ref()) {
                Some(g) => g,
                None => {
                    write_http(
                        &mut stream,
                        &mut out,
                        429,
                        "text/plain",
                        b"overloaded",
                        keep_alive,
                    )
                    .await?;
                    continue;
                }
            };

            let (dim, _) = match st.core.quick_dims() {
                Some(v) => v,
                None => {
                    write_http(
                        &mut stream,
                        &mut out,
                        500,
                        "text/plain",
                        b"quickscorer not enabled",
                        keep_alive,
                    )
                    .await?;
                    continue;
                }
            };

            let t_parse = std::time::Instant::now();
            let req = match parse_dense_payload_le(&body, dim) {
                Ok(p) => p,
                Err(msg) => {
                    write_http(
                        &mut stream,
                        &mut out,
                        400,
                        "text/plain",
                        msg.as_bytes(),
                        keep_alive,
                    )
                    .await?;
                    continue;
                }
            };
            let parse_us = now_us(t_parse);
            let trace_id = TRACE_ID_SEQ.fetch_add(1, Ordering::Relaxed);
            let sidecar_row_idx = req
                .route_meta
                .as_ref()
                .map(|m| m.row_idx as usize)
                .unwrap_or(trace_id as usize);
            let resp = match if let (Some(rt), Some(tau_mode)) =
                (st.standalone_l2_bench.as_ref(), st.standalone_l2_tau_mode)
            {
                st.core
                    .score_quick_dense_bytes_with_standalone_bench_l2_async(
                        parse_us,
                        req.payload,
                        req.route_meta,
                        rt,
                        sidecar_row_idx,
                        tau_mode,
                    )
                    .await
            } else {
                st.core
                    .score_quick_dense_bytes_with_meta_async(parse_us, req.payload, req.route_meta)
                    .await
            } {
                Ok(resp) => resp,
                Err(e) => {
                    let msg = format!("quickscorer inference failed: {:#}", e);
                    write_http(
                        &mut stream,
                        &mut out,
                        500,
                        "text/plain",
                        msg.as_bytes(),
                        keep_alive,
                    )
                    .await?;
                    continue;
                }
            };
            let decision_u8 = match resp.decision {
                Decision::Allow => 0,
                Decision::Deny => 1,
                Decision::ManualReview => 2,
                Decision::DegradeAllow => 3,
            };
            let qsb2 = encode_qsb2(
                trace_id,
                resp.score as f32,
                decision_u8,
                resp.timings_us.l2 > 0,
            );
            let timings_header = encode_timings_header(
                resp.timings_us.parse,
                resp.timings_us.feature,
                resp.timings_us.router,
                resp.timings_us.l1,
                resp.timings_us.l2,
                resp.timings_us.serialize,
            );
            risk_core::sampled_histogram!("e2e_us").record(now_us(t0) as f64);
            write_http_with_headers(
                &mut stream,
                &mut out,
                200,
                "application/octet-stream",
                &qsb2,
                keep_alive,
                &[("X-Risk-Timings-Us", timings_header.as_str())],
            )
            .await?;
        } else if meta.method == "GET" && meta.path == "/metrics" {
            let text = st.prom.render();
            write_http(
                &mut stream,
                &mut out,
                200,
                "text/plain; version=0.0.4",
                text.as_bytes(),
                keep_alive,
            )
            .await?;
        } else {
            write_http(
                &mut stream,
                &mut out,
                404,
                "text/plain",
                b"not found",
                keep_alive,
            )
            .await?;
        }

        maybe_shrink_conn_buffers(&mut stash, &mut out);

        if meta.want_close {
            return Ok(());
        }
    }
}

#[inline]
fn maybe_shrink_conn_buffers(stash: &mut BytesMut, out: &mut Vec<u8>) {
    if stash.capacity() > GLOMMIO_CONN_STASH_RETAIN_LIMIT_BYTES {
        let mut trimmed = BytesMut::with_capacity(GLOMMIO_CONN_STASH_BYTES);
        if !stash.is_empty() {
            trimmed.extend_from_slice(stash.as_ref());
        }
        *stash = trimmed;
    }
    if out.capacity() > GLOMMIO_CONN_OUT_RETAIN_LIMIT_BYTES {
        *out = Vec::with_capacity(GLOMMIO_CONN_OUT_BYTES);
    }
}

/// RAII guard for global in-flight accounting.
struct InFlightGuard {
    ctr: Arc<AtomicUsize>,
}
impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.ctr.fetch_sub(1, Ordering::Relaxed);
    }
}

fn try_acquire_inflight<'a>(
    st: &AppState,
    core_sem: &'a Semaphore,
) -> Option<(InFlightGuard, Permit<'a>)> {
    // 先拿全局 token，再拿本核 token；本核拿不到就回滚全局。
    let cur = st.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
    if st.max_in_flight > 0 && cur > st.max_in_flight {
        st.in_flight.fetch_sub(1, Ordering::Relaxed);
        return None;
    }

    let g = InFlightGuard {
        ctr: Arc::clone(&st.in_flight),
    };

    match core_sem.try_acquire_permit(1) {
        Ok(p) => Some((g, p)),
        Err(_) => {
            // 回滚全局
            drop(g);
            None
        }
    }
}

/// 从环境变量 / 进程 affinity 里推导 Glommio IO executors 应该绑定的 CPU 列表。
///
/// 规则：
/// 1) 先读当前进程允许的 CPU（比如通过 taskset/cpuset 限制）。
/// 2) 再读 XGB pool 的 pin 列表（XGB_L1_POOL_PIN_CPUS / XGB_L2_POOL_PIN_CPUS / XGB_POOL_PIN_CPUS）。
/// 3) IO_Cpus = Allowed - XGB_Pinned；如果为空，就退化为 Allowed（避免跑到 taskset 之外）。
fn select_glommio_io_cpus_from_env() -> anyhow::Result<Vec<usize>> {
    let allowed = get_self_affinity_cpus()?;
    if allowed.is_empty() {
        return Ok(Vec::new());
    }

    let mut pinned: HashSet<usize> = HashSet::new();
    pinned.extend(parse_cpu_env("XGB_L1_POOL_PIN_CPUS"));
    pinned.extend(parse_cpu_env("XGB_L2_POOL_PIN_CPUS"));
    pinned.extend(parse_cpu_env("XGB_POOL_PIN_CPUS"));

    let io: Vec<usize> = allowed
        .iter()
        .copied()
        .filter(|cpu| !pinned.contains(cpu))
        .collect();

    if io.is_empty() {
        Ok(allowed)
    } else {
        Ok(io)
    }
}

fn parse_cpu_env(var: &str) -> HashSet<usize> {
    match std::env::var(var) {
        Ok(s) => parse_cpu_csv(&s),
        Err(_) => HashSet::new(),
    }
}

fn parse_cpu_csv(s: &str) -> HashSet<usize> {
    let mut out = HashSet::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if let Ok(v) = p.parse::<usize>() {
            out.insert(v);
        }
    }
    out
}

fn get_self_affinity_cpus() -> anyhow::Result<Vec<usize>> {
    #[cfg(target_os = "linux")]
    {
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            let rc = libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
            if rc != 0 {
                return Err(anyhow::anyhow!(std::io::Error::last_os_error()));
            }

            let mut cpus = Vec::new();
            for cpu in 0..(libc::CPU_SETSIZE as usize) {
                if libc::CPU_ISSET(cpu, &set) {
                    cpus.push(cpu);
                }
            }
            cpus.sort_unstable();
            Ok(cpus)
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        let n = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Ok((0..n).collect())
    }
}

fn read_policy_hash() -> anyhow::Result<String> {
    let path = std::env::var("MAMBA_POLICY_PATH")
        .unwrap_or_else(|_| "gate_policy_v1_mainline.json".to_string());
    let data = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&data);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Parse dense payload.
/// - raw: exactly dim*4 bytes
/// - RVEC: magic="RVEC"(4) + ver(u16=1) + flags(u16=0) + dim(u32) + reserved(u32) + payload(f32*dim)
fn parse_dense_payload_le(body: &Bytes, expected_dim: usize) -> Result<DenseRequest, String> {
    let expected_len = expected_dim
        .checked_mul(4)
        .ok_or_else(|| "expected_dim too large".to_string())?;

    // raw fast path
    if body.len() == expected_len {
        return Ok(DenseRequest {
            payload: body.clone(),
            route_meta: None,
        });
    }

    // RVEC header
    let b = body.as_ref();
    if b.len() < 16 {
        return Err(format!("body too short: {} < 16", b.len()));
    }
    if &b[0..4] != b"RVEC" {
        return Err("invalid dense payload (missing magic RVEC)".into());
    }
    let ver = u16::from_le_bytes([b[4], b[5]]);
    let flags = u16::from_le_bytes([b[6], b[7]]);
    if flags != 0 {
        return Err(format!("unsupported RVEC flags: {}", flags));
    }
    let dim = u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize;
    if dim != expected_dim {
        return Err(format!(
            "RVEC dim mismatch: got {}, expected {}",
            dim, expected_dim
        ));
    }
    match ver {
        1 => {
            let need = 16 + expected_len;
            if body.len() != need {
                return Err(format!(
                    "invalid dense payload size: got {}, expected {}",
                    body.len(),
                    need
                ));
            }
            Ok(DenseRequest {
                payload: body.slice(16..),
                route_meta: None,
            })
        }
        2 => {
            let need = 32 + expected_len;
            if body.len() != need {
                return Err(format!(
                    "invalid dense payload size: got {}, expected {}",
                    body.len(),
                    need
                ));
            }
            let fold_id = i32::from_le_bytes([b[12], b[13], b[14], b[15]]);
            let seg_prod_amtbin = u32::from_le_bytes([b[16], b[17], b[18], b[19]]);
            let transaction_id =
                u64::from_le_bytes([b[20], b[21], b[22], b[23], b[24], b[25], b[26], b[27]]);
            let row_idx = u32::from_le_bytes([b[28], b[29], b[30], b[31]]);
            Ok(DenseRequest {
                payload: body.slice(32..),
                route_meta: Some(QuickRouteMeta {
                    row_idx,
                    transaction_id,
                    fold_id,
                    seg_prod_amtbin,
                    l2_tau_used: None,
                }),
            })
        }
        3 => {
            let need = 40 + expected_len;
            if body.len() != need {
                return Err(format!(
                    "invalid dense payload size: got {}, expected {}",
                    body.len(),
                    need
                ));
            }
            let fold_id = i32::from_le_bytes([b[12], b[13], b[14], b[15]]);
            let seg_prod_amtbin = u32::from_le_bytes([b[16], b[17], b[18], b[19]]);
            let transaction_id =
                u64::from_le_bytes([b[20], b[21], b[22], b[23], b[24], b[25], b[26], b[27]]);
            let row_idx = u32::from_le_bytes([b[28], b[29], b[30], b[31]]);
            let l2_tau_used = f32::from_le_bytes([b[32], b[33], b[34], b[35]]);
            Ok(DenseRequest {
                payload: body.slice(40..),
                route_meta: Some(QuickRouteMeta {
                    row_idx,
                    transaction_id,
                    fold_id,
                    seg_prod_amtbin,
                    l2_tau_used: Some(l2_tau_used),
                }),
            })
        }
        _ => Err(format!("unsupported RVEC version: {}", ver)),
    }
}

/// Encode 48-byte RSK1 response.
/// Layout (compatible with risk-bench2 decoder):
/// - magic 'RSK1'
/// - ver(u16)=1
/// - flags(u16)
/// - trace_id(u64)
/// - score(f32)
/// - decision(u8) + pad[3]
/// - timings[6] u32: parse, feature, router, l1, l2, serialize
fn encode_rsk1(
    trace_id: u64,
    score: f32,
    decision: u8,
    parse_us: u64,
    feature_us: u64,
    router_us: u64,
    l1_us: u64,
    l2_us: u64,
    serialize_us: u64,
) -> Vec<u8> {
    let mut flags: u16 = 0;
    if l2_us > 0 {
        flags |= 1;
    }
    if decision == 3 {
        flags |= 2;
    }

    #[inline]
    fn clamp_u32(x: u64) -> u32 {
        (x.min(u32::MAX as u64)) as u32
    }

    let mut out = Vec::with_capacity(48);
    out.extend_from_slice(b"RSK1");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.extend_from_slice(&trace_id.to_le_bytes());
    out.extend_from_slice(&score.to_le_bytes());
    out.push(decision);
    out.extend_from_slice(&[0u8; 3]);
    out.extend_from_slice(&clamp_u32(parse_us).to_le_bytes());
    out.extend_from_slice(&clamp_u32(feature_us).to_le_bytes());
    out.extend_from_slice(&clamp_u32(router_us).to_le_bytes());
    out.extend_from_slice(&clamp_u32(l1_us).to_le_bytes());
    out.extend_from_slice(&clamp_u32(l2_us).to_le_bytes());
    out.extend_from_slice(&clamp_u32(serialize_us).to_le_bytes());

    debug_assert_eq!(out.len(), 48);
    out
}

#[inline]
fn encode_qsb2(trace_id: u64, score: f32, decision: u8, used_l2: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(24);
    out.extend_from_slice(b"QSB2");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.push(decision);
    out.push(if used_l2 { 1 } else { 0 });
    out.extend_from_slice(&trace_id.to_le_bytes());
    out.extend_from_slice(&score.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(out.len(), 24);
    out
}

fn encode_timings_header(
    parse_us: u64,
    feature_us: u64,
    router_us: u64,
    l1_us: u64,
    l2_us: u64,
    serialize_us: u64,
) -> String {
    format!(
        "{},{},{},{},{},{}",
        parse_us, feature_us, router_us, l1_us, l2_us, serialize_us
    )
}

#[inline]
fn patch_rsk1_serialize_us(buf: &mut [u8], serialize_us: u64) {
    if buf.len() < 48 {
        return;
    }
    let v = (serialize_us.min(u32::MAX as u64) as u32).to_le_bytes();
    // timings[5] offset: 24 + 5*4 = 44
    buf[44..48].copy_from_slice(&v);
}

fn parse_request_head(head: &str) -> ReqMeta {
    let mut lines = head.split("\r\n");
    let first = lines.next().unwrap_or("");
    let mut it = first.split_whitespace();
    let method = it.next().unwrap_or("").to_string();
    let path = it.next().unwrap_or("").to_string();

    let mut content_len: Option<usize> = None;
    let mut chunked = false;
    let mut expect_100 = false;
    let mut want_close = false;

    for line in lines {
        let line = line.trim();

        if let Some(v) = line
            .strip_prefix("Content-Length:")
            .or_else(|| line.strip_prefix("content-length:"))
        {
            content_len = v.trim().parse::<usize>().ok();
        }

        if let Some(v) = line
            .strip_prefix("Transfer-Encoding:")
            .or_else(|| line.strip_prefix("transfer-encoding:"))
        {
            if v.split(',')
                .any(|t| t.trim().eq_ignore_ascii_case("chunked"))
            {
                chunked = true;
            }
        }

        if line.eq_ignore_ascii_case("Expect: 100-continue") {
            expect_100 = true;
        }

        if line.eq_ignore_ascii_case("Connection: close") {
            want_close = true;
        }
    }

    ReqMeta {
        method,
        path,
        content_len,
        chunked,
        expect_100,
        want_close,
    }
}

fn find_double_crlf(b: &[u8]) -> Option<usize> {
    b.windows(4).position(|w| w == b"\r\n\r\n")
}

async fn write_http(
    stream: &mut glommio::net::TcpStream,
    scratch: &mut Vec<u8>,
    code: u16,
    ctype: &str,
    body: &[u8],
    keep_alive: bool,
) -> anyhow::Result<()> {
    write_http_with_headers(stream, scratch, code, ctype, body, keep_alive, &[]).await
}

async fn write_http_with_headers(
    stream: &mut glommio::net::TcpStream,
    scratch: &mut Vec<u8>,
    code: u16,
    ctype: &str,
    body: &[u8],
    keep_alive: bool,
    extra_headers: &[(&str, &str)],
) -> anyhow::Result<()> {
    let status = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        429 => "Too Many Requests",
        503 => "Service Unavailable",
        500 => "Internal Server Error",
        _ => "OK",
    };

    let conn = if keep_alive { "keep-alive" } else { "close" };

    // 关键：复用 scratch，合并 header+body，一次性写出，避免 Nagle/delayed-ack 造成 ~40ms 抖动 + 避免每请求分配
    scratch.clear();
    // Vec<u8> 实现了 std::io::Write，所以 write! 不会额外分配 String。
    write!(scratch, "HTTP/1.1 {} {}\r\n", code, status)?;
    write!(scratch, "Content-Type: {}\r\n", ctype)?;
    write!(scratch, "Content-Length: {}\r\n", body.len())?;
    write!(scratch, "Connection: {}\r\n", conn)?;
    for (name, value) in extra_headers {
        write!(scratch, "{}: {}\r\n", name, value)?;
    }
    scratch.extend_from_slice(b"\r\n");
    scratch.extend_from_slice(body);

    stream.write_all(scratch).await?;
    Ok(())
}

fn find_crlf(b: &[u8]) -> Option<usize> {
    b.windows(2).position(|w| w == b"\r\n")
}

/// 尝试从 stash[start..] 解析 chunked body。
/// 成功则返回 (consumed_end_index, body_bytes)。
fn try_parse_chunked(stash: &[u8], start: usize) -> Option<(usize, Vec<u8>)> {
    let mut i = start;
    let mut out = Vec::new();

    loop {
        // chunk size line
        let rel = find_crlf(&stash.get(i..)?)?;
        let line_end = i + rel;
        let line = std::str::from_utf8(&stash[i..line_end]).ok()?.trim();
        let size = usize::from_str_radix(line, 16).ok()?;
        let mut j = line_end + 2; // skip \r\n

        if size == 0 {
            // 终止块：期待 "\r\n"
            if stash.len() < j + 2 {
                return None;
            }
            if &stash[j..j + 2] != b"\r\n" {
                return None;
            }
            j += 2;
            return Some((j, out));
        }

        // need chunk data + trailing \r\n
        if stash.len() < j + size + 2 {
            return None;
        }
        out.extend_from_slice(&stash[j..j + size]);
        j += size;

        if &stash[j..j + 2] != b"\r\n" {
            return None;
        }
        i = j + 2;
    }
}
