use std::sync::atomic::{AtomicU64, Ordering};
use std::{net::SocketAddr, path::Path, str::FromStr, sync::Arc, time::Instant};

use anyhow::Context;
use axum::{
    body::Bytes,
    error_handling::HandleErrorLayer,
    extract::State,
    http,
    http::{header, HeaderName, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use risk_core::{
    config::Config,
    pipeline::AppCore,
    pipeline::StandaloneL2TauMode,
    quickscorer::QuickRouteMeta,
    schema::{Decision, ScoreResponse},
};
use risk_quickscorer_standalone_l2::StandaloneL2Runtime;
use tokio::sync::Semaphore as TokioSemaphore;
use tower::{BoxError, ServiceBuilder};
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, TraceLayer};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

const HDR_RISK_TIMINGS_US: HeaderName = HeaderName::from_static("x-risk-timings-us");

#[derive(Clone, Debug)]
struct DenseRequest {
    payload: Bytes,
    route_meta: Option<QuickRouteMeta>,
}

#[derive(Parser, Debug)]
#[command(name = "risk-server-tokio", version, about)]
struct Args {
    /// Listen address, e.g. 127.0.0.1:8080
    #[arg(long, default_value = "127.0.0.1:8080")]
    listen: String,

    /// QuickScorer bundle root (required)
    #[arg(long, value_name = "DIR")]
    bundle_dir: String,

    /// Max in-flight HTTP requests (tower concurrency_limit)
    #[arg(long, default_value_t = 4096)]
    max_in_flight: usize,

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

    /// Benchmark-only h2c listen addr for risk-bench3, e.g. 127.0.0.1:19092
    #[arg(long)]
    bench3_h2_listen: Option<String>,
}

#[derive(Clone)]
struct AppState {
    core: Arc<AppCore>,
    prom: PrometheusHandle,
    standalone_l2_bench: Option<Arc<StandaloneL2Runtime>>,
    standalone_l2_tau_mode: Option<StandaloneL2TauMode>,
    bench_in_flight: Arc<TokioSemaphore>,
}

static TRACE_ID_SEQ: AtomicU64 = AtomicU64::new(1);

fn env_usize(name: &str, default_value: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default_value)
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,tower_http=info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

fn install_prometheus() -> PrometheusHandle {
    PrometheusBuilder::new()
        .add_global_label("service", "risk-server-tokio")
        .install_recorder()
        .expect("install prometheus recorder")
}

async fn health() -> &'static str {
    "ok"
}

async fn metrics(State(st): State<AppState>) -> String {
    st.prom.render()
}

async fn debug_backend(State(st): State<AppState>) -> Response {
    let quick = st.core.quick.as_ref();
    let l2_ctrl = st.core.l2_ctrl();

    let body = serde_json::json!({
        "quickscorer": quick.map(|q| q.debug_info()),
        "router_l2": {
            "sample_ratio": l2_ctrl.sample_ratio(),
            "sample_base_ratio": l2_ctrl.sample_base_ratio(),
            "sample_dyn_ratio": l2_ctrl.sample_dyn_ratio(),
            "waterline_target": l2_ctrl.sample_waterline_target(),
            "waterline_hi": l2_ctrl.sample_waterline_hi(),
            "waterline_lo": l2_ctrl.sample_waterline_lo(),
        },
        "benchmark_l2": st.standalone_l2_bench.as_ref().map(|rt: &Arc<StandaloneL2Runtime>| serde_json::json!({
            "enabled": true,
            "feat_rows": rt.feat_rows(),
            "l2_dim": rt.l2_dim(),
            "tau_mode": match st.standalone_l2_tau_mode {
                Some(StandaloneL2TauMode::Request) => "request",
                Some(StandaloneL2TauMode::Fixed(_)) => "fixed",
                None => "none",
            }
        })).unwrap_or_else(|| serde_json::json!({
            "enabled": false
        })),
    });

    (StatusCode::OK, Json(body)).into_response()
}

/// tower 的 load_shed / concurrency_limit 早拒绝会走到这里。
/// 我们统一变成 429 overloaded（不再出现你日志里的 503 latency=0ms）。

fn parse_dense_payload_le(body: &Bytes, expected_dim: usize) -> Result<DenseRequest, String> {
    let b = body.as_ref();

    // Fast path: raw payload (no header)
    let raw_len = expected_dim
        .checked_mul(4)
        .ok_or_else(|| "expected_dim too large".to_string())?;
    if b.len() == raw_len {
        return Ok(DenseRequest {
            payload: body.clone(),
            route_meta: None,
        });
    }

    // Headered payload
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
            "dense dim mismatch: got {}, expected {}",
            dim, expected_dim
        ));
    }
    match ver {
        1 => {
            let need = 16 + raw_len;
            if b.len() != need {
                return Err(format!(
                    "invalid payload size: got {}, expected {}",
                    b.len(),
                    need
                ));
            }
            Ok(DenseRequest {
                payload: body.slice(16..),
                route_meta: None,
            })
        }
        2 => {
            let need = 32 + raw_len;
            if b.len() != need {
                return Err(format!(
                    "invalid payload size: got {}, expected {}",
                    b.len(),
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
            let need = 40 + raw_len;
            if b.len() != need {
                return Err(format!(
                    "invalid payload size: got {}, expected {}",
                    b.len(),
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

async fn score_dense_binary_request(
    st: &AppState,
    body: Bytes,
) -> Result<(u64, ScoreResponse), (StatusCode, String)> {
    let t_parse = Instant::now();
    let Some((expected_dim, _)) = st.core.quick_dims() else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "quickscorer not enabled: start server with --bundle-dir".to_string(),
        ));
    };

    let req = parse_dense_payload_le(&body, expected_dim)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    let trace_id = TRACE_ID_SEQ.fetch_add(1, Ordering::Relaxed);
    let parse_us = t_parse.elapsed().as_micros() as u64;

    let resp = if let (Some(rt), Some(tau_mode)) =
        (st.standalone_l2_bench.as_ref(), st.standalone_l2_tau_mode)
    {
        let sidecar_row_idx = req
            .route_meta
            .as_ref()
            .map(|m| m.row_idx as usize)
            .unwrap_or(trace_id as usize);
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
    }
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("quickscorer inference failed: {:#}", e),
        )
    })?;

    Ok((trace_id, resp))
}

fn decision_to_u8(d: &Decision) -> u8 {
    match d {
        Decision::Allow => 0,
        Decision::Deny => 1,
        Decision::ManualReview => 2,
        Decision::DegradeAllow => 3,
    }
}

/// RSK1 binary response layout (48 bytes, little-endian):
/// - magic[4] = "RSK1"
/// - version(u16)=1
/// - flags(u16): bit0=l2_path, bit1=degraded
/// - trace_id(u64)
/// - score(f32)
/// - decision(u8)
/// - pad[3]
/// - timings_us[6](u32): parse/feature/router/l1/l2/serialize
fn encode_rsk1_response(trace_id: u64, resp: &ScoreResponse) -> Vec<u8> {
    let mut out = Vec::with_capacity(48);
    out.extend_from_slice(b"RSK1");
    out.extend_from_slice(&1u16.to_le_bytes());

    let mut flags: u16 = 0;
    if resp.timings_us.l2 > 0 {
        flags |= 1 << 0;
    }
    if matches!(resp.decision, Decision::DegradeAllow) {
        flags |= 1 << 1;
    }
    out.extend_from_slice(&flags.to_le_bytes());

    out.extend_from_slice(&trace_id.to_le_bytes());
    out.extend_from_slice(&(resp.score as f32).to_le_bytes());
    out.push(decision_to_u8(&resp.decision));
    out.extend_from_slice(&[0u8; 3]);

    #[inline]
    fn clamp_u32(x: u64) -> u32 {
        if x > u32::MAX as u64 {
            u32::MAX
        } else {
            x as u32
        }
    }

    let ts = &resp.timings_us;
    for v in [ts.parse, ts.feature, ts.router, ts.l1, ts.l2, ts.serialize] {
        out.extend_from_slice(&clamp_u32(v).to_le_bytes());
    }

    debug_assert!(
        out.len() == 48,
        "RSK1 response must be 48 bytes, got {}",
        out.len()
    );
    out
}

/// QSB2 binary response layout (24 bytes, little-endian):
/// - magic[4] = "QSB2"
/// - version(u16)=1
/// - decision(u8)
/// - flags(u8): bit0=l2_path
/// - trace_id(u64)
/// - score(f32)
/// - reserved(u32)=0
fn encode_qsb2_response(trace_id: u64, resp: &ScoreResponse) -> Vec<u8> {
    let mut out = Vec::with_capacity(24);
    out.extend_from_slice(b"QSB2");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.push(decision_to_u8(&resp.decision));
    let flags = if resp.timings_us.l2 > 0 { 1u8 } else { 0u8 };
    out.push(flags);
    out.extend_from_slice(&trace_id.to_le_bytes());
    out.extend_from_slice(&(resp.score as f32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    debug_assert!(out.len() == 24, "QSB2 response must be 24 bytes");
    out
}

fn encode_timings_header_value(resp: &ScoreResponse) -> Option<HeaderValue> {
    let ts = &resp.timings_us;
    HeaderValue::from_str(&format!(
        "{},{},{},{},{},{}",
        ts.parse, ts.feature, ts.router, ts.l1, ts.l2, ts.serialize
    ))
    .ok()
}

async fn handle_h2_bench_stream(
    req: http::Request<h2::RecvStream>,
    mut respond: h2::server::SendResponse<Bytes>,
    st: AppState,
) -> anyhow::Result<()> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    if method != Method::POST {
        let response = http::Response::builder()
            .status(StatusCode::METHOD_NOT_ALLOWED)
            .header(header::CONTENT_TYPE, "text/plain")
            .body(())
            .context("build h2 method-not-allowed response")?;
        let mut send = respond
            .send_response(response, false)
            .context("send h2 method-not-allowed headers")?;
        send.send_data(Bytes::from_static(b"method not allowed"), true)
            .context("send h2 method-not-allowed body")?;
        return Ok(());
    }

    if path != "/score_dense_f32_bin" && path != "/score_dense_f32_bin_v2" {
        let response = http::Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header(header::CONTENT_TYPE, "text/plain")
            .body(())
            .context("build h2 not-found response")?;
        let mut send = respond
            .send_response(response, false)
            .context("send h2 not-found headers")?;
        send.send_data(Bytes::from_static(b"not found"), true)
            .context("send h2 not-found body")?;
        return Ok(());
    }

    let _permit = st
        .bench_in_flight
        .clone()
        .acquire_owned()
        .await
        .context("acquire h2 bench permit")?;

    let mut body_stream = req.into_body();
    let mut body = bytes::BytesMut::with_capacity(4096);
    while let Some(chunk) = body_stream.data().await {
        let chunk = chunk.context("read h2 request body")?;
        body.extend_from_slice(&chunk);
        let _ = body_stream.flow_control().release_capacity(chunk.len());
    }
    let body = body.freeze();

    let outcome = match path.as_str() {
        "/score_dense_f32_bin" => score_dense_binary_request(&st, body).await.map(|(trace_id, resp)| {
            let bin = encode_rsk1_response(trace_id, &resp);
            let headers: Vec<(HeaderName, HeaderValue)> = vec![(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            )];
            (StatusCode::OK, headers, Bytes::from(bin))
        }),
        "/score_dense_f32_bin_v2" => score_dense_binary_request(&st, body).await.map(|(trace_id, resp)| {
            let mut headers = vec![(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            )];
            if let Some(v) = encode_timings_header_value(&resp) {
                headers.push((HDR_RISK_TIMINGS_US, v));
            }
            (StatusCode::OK, headers, Bytes::from(encode_qsb2_response(trace_id, &resp)))
        }),
        _ => unreachable!("path already validated"),
    };

    let (status, headers, body) = match outcome {
        Ok(ok) => ok,
        Err((status, msg)) => (
            status,
            vec![(header::CONTENT_TYPE, HeaderValue::from_static("text/plain"))],
            Bytes::from(msg),
        ),
    };

    let mut builder = http::Response::builder().status(status);
    for (name, value) in headers {
        builder = builder.header(name, value);
    }
    let response = builder.body(()).context("build h2 response")?;
    let mut send = respond.send_response(response, false).context("send h2 headers")?;
    send.send_data(body, true).context("send h2 body")?;
    Ok(())
}

async fn handle_h2_bench_conn(
    stream: tokio::net::TcpStream,
    st: AppState,
) -> anyhow::Result<()> {
    let mut conn = h2::server::handshake(stream)
        .await
        .context("h2 server handshake")?;
    while let Some(result) = conn.accept().await {
        let (req, respond) = result.context("accept h2 stream")?;
        let st2 = st.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_h2_bench_stream(req, respond, st2).await {
                warn!(error = %e, "h2 bench stream failed");
            }
        });
    }
    Ok(())
}

async fn run_h2_bench_listener(addr: SocketAddr, st: AppState) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("bench3 h2c listening on h2c://{}", addr);
    loop {
        let (stream, peer) = listener.accept().await?;
        stream.set_nodelay(true).ok();
        let st2 = st.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_h2_bench_conn(stream, st2).await {
                warn!(peer = %peer, error = %e, "h2 bench conn failed");
            }
        });
    }
}

/// ✅ Dense f32 直传 + Binary response（application/octet-stream）
/// 请求体为 raw f32le 或带 RVEC header 的 dense payload。
async fn score_dense_f32_bin(State(st): State<AppState>, body: Bytes) -> Response {
    match score_dense_binary_request(&st, body).await {
        Ok((trace_id, resp)) => {
            // 让 timings.serialize 代表二进制序列化时间（而不是 core 侧的 JSON to_vec 计时）。
            let t_ser = Instant::now();
            // 先用旧值编码，拿到真实编码开销后再写回再编码一遍会多一次分配。
            // 我们这里走“单次编码”：先估计 serialize_us=0，编码后写回到 header 里的 timings.serialize。
            // 为了保持简单，直接把 serialize_us 记到 metrics 上，不再写回 body。
            // （bench 端依然能从 stage_p99 里看到 serialize 的数量级，且目前 serialize 占比极小）
            let bin = encode_rsk1_response(trace_id, &resp);
            let _serialize_us = t_ser.elapsed().as_micros() as u64;
            let mut r = Response::new(axum::body::Body::from(bin));
            *r.status_mut() = StatusCode::OK;
            r.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            );
            // trace_id 只在错误时打日志，避免热路径噪声；客户端会拿到 trace_id。
            r
        }
        Err((status, msg)) => {
            error!(error = %msg, "score_dense_f32_bin failed");
            (status, msg).into_response()
        }
    }
}

async fn score_dense_f32_bin_v2(State(st): State<AppState>, body: Bytes) -> Response {
    match score_dense_binary_request(&st, body).await {
        Ok((trace_id, resp)) => {
            let bin = encode_qsb2_response(trace_id, &resp);
            let mut r = Response::new(axum::body::Body::from(bin));
            *r.status_mut() = StatusCode::OK;
            r.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            );
            if let Some(v) = encode_timings_header_value(&resp) {
                r.headers_mut().insert(HDR_RISK_TIMINGS_US, v);
            }
            r
        }
        Err((status, msg)) => {
            error!(error = %msg, "score_dense_f32_bin_v2 failed");
            (status, msg).into_response()
        }
    }
}

async fn handle_tower_overload(err: BoxError) -> Response {
    warn!(error = %err, "request rejected by middleware");
    (StatusCode::TOO_MANY_REQUESTS, "overloaded").into_response()
}

async fn async_main(
    args: Args,
    worker_threads: usize,
    max_blocking_threads: usize,
) -> anyhow::Result<()> {
    init_tracing();
    info!(
        "tokio runtime: worker_threads={} max_blocking_threads={}",
        worker_threads, max_blocking_threads
    );

    let prom = install_prometheus();

    let mut cfg = Config::default();
    if let Ok(v) = std::env::var("SLO_P99_MS") {
        if let Ok(ms) = v.parse::<u64>() {
            cfg.slo_p99_ms = ms;
        }
    }

    let core = AppCore::new_with_quickscorer_bundle(cfg, &args.bundle_dir)
        .context("init AppCore(bundle)")?;

    let standalone_l2_bench = if args.l2_bench_mode.as_deref() == Some("standalone-sidecar") {
        let runtime = if let Some(path) = args.l2_bench_feat_bin.as_deref() {
            Arc::new(StandaloneL2Runtime::load_with_feat_bin_override(
                Path::new(&args.bundle_dir),
                Some(Path::new(path)),
            )?)
        } else {
            Arc::new(StandaloneL2Runtime::load(Path::new(&args.bundle_dir))?)
        };
        Some(runtime)
    } else {
        None
    };
    let standalone_l2_tau_mode = if standalone_l2_bench.is_some() {
        Some(match args.l2_bench_tau_mode.as_str() {
            "fixed" => StandaloneL2TauMode::Fixed(
                args.l2_bench_fixed_tau
                    .context("--l2-bench-fixed-tau is required when --l2-bench-tau-mode=fixed")?,
            ),
            _ => StandaloneL2TauMode::Request,
        })
    } else {
        None
    };

    if let Some(q) = core.quick.as_ref() {
        let dbg = q.debug_info();
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
    } else {
        warn!("QuickScorer not enabled");
    }

    let st = AppState {
        core: Arc::new(core),
        prom,
        standalone_l2_bench,
        standalone_l2_tau_mode,
        bench_in_flight: Arc::new(TokioSemaphore::new(args.max_in_flight.max(1))),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/debug/backend", get(debug_backend))
        .route("/score_dense_f32_bin", post(score_dense_f32_bin))
        .route("/score_dense_f32_bin_v2", post(score_dense_f32_bin_v2))
        .with_state(st.clone())
        .layer(
            ServiceBuilder::new()
                // ✅ 关键：HandleErrorLayer 必须包在最外层，才能把 overload 变成 429
                .layer(HandleErrorLayer::new(handle_tower_overload))
                .layer(tower::load_shed::LoadShedLayer::new())
                .layer(tower::limit::ConcurrencyLimitLayer::new(args.max_in_flight))
                .layer(
                    TraceLayer::new_for_http()
                        .make_span_with(DefaultMakeSpan::new().include_headers(false))
                        .on_failure(DefaultOnFailure::new()),
                ),
        );

    let addr = SocketAddr::from_str(&args.listen).context("invalid --listen")?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!("listening on http://{}", addr);

    if let Some(h2_addr) = args.bench3_h2_listen.as_ref() {
        let h2_addr = SocketAddr::from_str(h2_addr).context("invalid --bench3-h2-listen")?;
        let h2_state = st.clone();
        tokio::spawn(async move {
            if let Err(e) = run_h2_bench_listener(h2_addr, h2_state).await {
                error!(error = %e, "h2 bench listener failed");
            }
        });
    }

    axum::serve(listener, app.into_make_service())
        .await
        .context("server failed")?;

    Ok(())
}

fn main() -> anyhow::Result<()> {
    // 继续支持你现在的环境变量启动方式
    let worker_threads = env_usize("TOKIO_WORKER_THREADS", 4);
    let max_blocking_threads = env_usize("TOKIO_MAX_BLOCKING_THREADS", 4);

    let args = Args::parse();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .worker_threads(worker_threads)
        .max_blocking_threads(max_blocking_threads)
        .build()
        .context("build tokio runtime")?;

    rt.block_on(async_main(args, worker_threads, max_blocking_threads))
}
