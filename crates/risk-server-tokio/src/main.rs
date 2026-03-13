use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    convert::Infallible,
    io,
    io::{Read, Write},
    net::SocketAddr,
    path::Path,
    str::FromStr,
    sync::{mpsc as std_mpsc, Arc},
    time::Instant,
};

use anyhow::Context;
use axum::{
    body::{Body as AxumBody, Bytes},
    error_handling::HandleErrorLayer,
    extract::State,
    http,
    http::{header, HeaderName, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::BytesMut;
use clap::{Parser, ValueEnum};
use http_body_util::BodyExt;
use hyper::{body::Incoming, server::conn::http1, service::service_fn};
use hyper_util::rt::TokioIo;
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use mio::{Events, Interest, Poll, Token, Waker};
use risk_core::{
    config::Config,
    pipeline::AppCore,
    pipeline::StandaloneL2TauMode,
    quickscorer::QuickRouteMeta,
    schema::{Decision, ScoreResponse},
};
use risk_quickscorer_standalone_l2::StandaloneL2Runtime;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::{mpsc, Semaphore as TokioSemaphore};
use tower::{BoxError, ServiceBuilder, ServiceExt};
use tower_http::trace::{DefaultMakeSpan, DefaultOnFailure, TraceLayer};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

const HDR_RISK_TIMINGS_US: HeaderName = HeaderName::from_static("x-risk-timings-us");
const H1_BATCH_ACK_PREFIX: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Length: 40\r\nContent-Type: application/octet-stream\r\n\r\n";
const H1_TEXT_PREFIX_CONTENT_TYPE: &str = "text/plain; charset=utf-8";
const H1_BATCH_PEEK_BYTES: usize = 1024;
const MAX_H1_BATCH_HEADER_BYTES: usize = 4096;
const H1_BATCH_ACK_BODY_LEN: usize = 40;
const H1_BATCH_BINARY_REPLY_LEN: usize = H1_BATCH_ACK_PREFIX.len() + H1_BATCH_ACK_BODY_LEN;
const H1_BATCH_WAKE_TOKEN: Token = Token(0);

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

    /// Sample 1/N throughput h2 bench requests for server-side stage timing metrics
    #[arg(long, default_value_t = 1024)]
    bench3_h2_sample_rate: usize,

    /// Batch dataplane implementation for fixed batch128 HTTP/1.1 requests
    #[arg(long, value_enum, default_value_t = H1BatchDataplaneMode::MioShard)]
    h1_batch_dataplane: H1BatchDataplaneMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum H1BatchDataplaneMode {
    TokioShard,
    MioShard,
}

#[derive(Clone)]
struct AppState {
    core: Arc<AppCore>,
    prom: PrometheusHandle,
    standalone_l2_bench: Option<Arc<StandaloneL2Runtime>>,
    standalone_l2_tau_mode: Option<StandaloneL2TauMode>,
    bench_in_flight: Arc<TokioSemaphore>,
    bench_h2_sample_rate: usize,
    bench_h2_sample_seq: Arc<AtomicU64>,
}

static TRACE_ID_SEQ: AtomicU64 = AtomicU64::new(1);
static BENCH_H2_REQ_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy)]
enum H1BatchFastPath {
    Score,
    Null,
    ParseOnly,
}

#[derive(Clone, Copy)]
struct Batch128Shape {
    has_route_meta: bool,
    record_bytes: usize,
    content_len: usize,
}

struct H1BatchParsedRequest {
    fast_path: Option<H1BatchFastPath>,
    method: Method,
    content_len: usize,
    body: Bytes,
}

enum H1BatchReply {
    Binary(Bytes),
    Text(StatusCode, String),
}

struct MioShardHandle {
    tx: std_mpsc::Sender<std::net::TcpStream>,
    waker: Arc<Waker>,
}

struct MioBatchConn {
    stream: mio::net::TcpStream,
    read_buf: BytesMut,
    write_buf: Vec<u8>,
    write_off: usize,
    close_after_write: bool,
}

impl MioBatchConn {
    fn new(stream: mio::net::TcpStream, max_request_len: usize) -> Self {
        Self {
            stream,
            read_buf: BytesMut::with_capacity(max_request_len),
            write_buf: Vec::with_capacity(H1_BATCH_BINARY_REPLY_LEN.max(256)),
            write_off: 0,
            close_after_write: false,
        }
    }

    fn has_pending_write(&self) -> bool {
        self.write_off < self.write_buf.len()
    }

    fn clear_write_buf(&mut self) {
        self.write_buf.clear();
        self.write_off = 0;
        self.close_after_write = false;
    }
}

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

/// Early rejections from tower load_shed / concurrency_limit end up here.
/// Normalize them to HTTP 429.

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

fn encode_qsb2_record(trace_id: u64, resp: &ScoreResponse, out: &mut Vec<u8>) {
    out.extend_from_slice(b"QSB2");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.push(decision_to_u8(&resp.decision));
    let flags = if resp.timings_us.l2 > 0 { 1u8 } else { 0u8 };
    out.push(flags);
    out.extend_from_slice(&trace_id.to_le_bytes());
    out.extend_from_slice(&(resp.score as f32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
}

fn encode_batch_aggregate_ack(
    record_count: u32,
    ok_count: u32,
    used_l2_count: u32,
    decision_counts: [u32; 5],
) -> [u8; 40] {
    let mut out = [0u8; 40];
    out[0..4].copy_from_slice(b"RBA1");
    out[4..6].copy_from_slice(&1u16.to_le_bytes());
    out[6..8].copy_from_slice(&0u16.to_le_bytes());
    out[8..12].copy_from_slice(&record_count.to_le_bytes());
    out[12..16].copy_from_slice(&ok_count.to_le_bytes());
    out[16..20].copy_from_slice(&used_l2_count.to_le_bytes());
    for (i, count) in decision_counts.into_iter().enumerate() {
        let off = 20 + i * 4;
        out[off..off + 4].copy_from_slice(&count.to_le_bytes());
    }
    out
}

fn build_h1_text_reply(status: StatusCode, msg: &str) -> Vec<u8> {
    let reason = status.canonical_reason().unwrap_or("Error");
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nContent-Type: {}\r\n\r\n",
        status.as_u16(),
        reason,
        msg.len(),
        H1_TEXT_PREFIX_CONTENT_TYPE
    );
    let mut out = Vec::with_capacity(head.len() + msg.len());
    out.extend_from_slice(head.as_bytes());
    out.extend_from_slice(msg.as_bytes());
    out
}

fn fill_h1_binary_reply(write_buf: &mut Vec<u8>, ack: [u8; 40]) {
    write_buf.clear();
    write_buf.extend_from_slice(H1_BATCH_ACK_PREFIX);
    write_buf.extend_from_slice(&ack);
}

fn parse_batch_dense_payload_le(
    body: &Bytes,
    expected_dim: usize,
) -> Result<(bool, usize, usize), String> {
    if body.len() < 16 || &body[0..4] != b"RBH1" {
        return Err("bad batch magic".to_string());
    }
    let version = u16::from_le_bytes([body[4], body[5]]);
    if version != 1 {
        return Err(format!("unsupported batch version: {}", version));
    }
    let flags = u16::from_le_bytes([body[6], body[7]]);
    let record_count = u32::from_le_bytes([body[8], body[9], body[10], body[11]]) as usize;
    let record_bytes = u32::from_le_bytes([body[12], body[13], body[14], body[15]]) as usize;
    if record_count == 0 {
        return Err("record_count must be > 0".to_string());
    }
    let has_route_meta = (flags & 1) != 0;
    let expect_record_bytes = expected_dim
        .checked_mul(4)
        .and_then(|n| n.checked_add(if has_route_meta { 40 } else { 0 }))
        .ok_or_else(|| "record_bytes overflow".to_string())?;
    if record_bytes != expect_record_bytes {
        return Err(format!(
            "record_bytes mismatch: got {} expected {}",
            record_bytes, expect_record_bytes
        ));
    }
    let expect_len = 16usize
        .checked_add(
            record_count
                .checked_mul(record_bytes)
                .ok_or_else(|| "batch length overflow".to_string())?,
        )
        .ok_or_else(|| "batch length overflow".to_string())?;
    if body.len() != expect_len {
        return Err(format!(
            "batch body len mismatch: got {} expected {}",
            body.len(),
            expect_len
        ));
    }
    Ok((has_route_meta, record_count, record_bytes))
}

fn batch128_shape_for_content_len(
    expected_dim: usize,
    content_len: usize,
) -> Option<Batch128Shape> {
    let raw_record_bytes = expected_dim.checked_mul(4)?;
    let route_record_bytes = raw_record_bytes.checked_add(40)?;
    let raw_len = 16usize.checked_add(128usize.checked_mul(raw_record_bytes)?)?;
    if content_len == raw_len {
        return Some(Batch128Shape {
            has_route_meta: false,
            record_bytes: raw_record_bytes,
            content_len,
        });
    }
    let route_len = 16usize.checked_add(128usize.checked_mul(route_record_bytes)?)?;
    if content_len == route_len {
        return Some(Batch128Shape {
            has_route_meta: true,
            record_bytes: route_record_bytes,
            content_len,
        });
    }
    None
}

fn validate_batch128_header(
    body: &[u8],
    has_route_meta: bool,
    record_bytes: usize,
) -> Result<(), String> {
    if body.len() < 16 {
        return Err("body too short for batch header".to_string());
    }
    if &body[0..4] != b"RBH1" {
        return Err("bad batch magic".to_string());
    }
    let version = u16::from_le_bytes([body[4], body[5]]);
    if version != 1 {
        return Err(format!("unsupported batch version: {}", version));
    }
    let flags = u16::from_le_bytes([body[6], body[7]]);
    let expect_flags = if has_route_meta { 1u16 } else { 0u16 };
    if flags != expect_flags {
        return Err(format!(
            "batch flags mismatch: got {} expected {}",
            flags, expect_flags
        ));
    }
    let record_count = u32::from_le_bytes([body[8], body[9], body[10], body[11]]) as usize;
    if record_count != 128 {
        return Err(format!(
            "record_count mismatch: got {} expected 128",
            record_count
        ));
    }
    let got_record_bytes = u32::from_le_bytes([body[12], body[13], body[14], body[15]]) as usize;
    if got_record_bytes != record_bytes {
        return Err(format!(
            "record_bytes mismatch: got {} expected {}",
            got_record_bytes, record_bytes
        ));
    }
    Ok(())
}

fn parse_batch_record_ref<'a>(
    record: &'a [u8],
    has_route_meta: bool,
) -> Result<(&'a [u8], Option<QuickRouteMeta>), String> {
    if !has_route_meta {
        return Ok((record, None));
    }
    if record.len() < 40 || &record[0..4] != b"RVEC" {
        return Err("batch record missing RVEC header".to_string());
    }
    let ver = u16::from_le_bytes([record[4], record[5]]);
    if ver != 3 {
        return Err(format!("unsupported batch RVEC version: {}", ver));
    }
    let flags = u16::from_le_bytes([record[6], record[7]]);
    if flags != 0 {
        return Err(format!("unsupported batch RVEC flags: {}", flags));
    }
    let dim = u32::from_le_bytes([record[8], record[9], record[10], record[11]]) as usize;
    let payload_len = dim
        .checked_mul(4)
        .ok_or_else(|| "batch record payload overflow".to_string())?;
    if record.len() != 40 + payload_len {
        return Err(format!(
            "batch record len mismatch: got {} expected {}",
            record.len(),
            40 + payload_len
        ));
    }
    let fold_id = i32::from_le_bytes([record[12], record[13], record[14], record[15]]);
    let seg_prod_amtbin = u32::from_le_bytes([record[16], record[17], record[18], record[19]]);
    let transaction_id = u64::from_le_bytes([
        record[20], record[21], record[22], record[23], record[24], record[25], record[26],
        record[27],
    ]);
    let row_idx = u32::from_le_bytes([record[28], record[29], record[30], record[31]]);
    let l2_tau_used = f32::from_le_bytes([record[32], record[33], record[34], record[35]]);
    Ok((
        &record[40..],
        Some(QuickRouteMeta {
            row_idx,
            transaction_id,
            fold_id,
            seg_prod_amtbin,
            l2_tau_used: Some(l2_tau_used),
        }),
    ))
}

struct Batch128Refs<'a> {
    rows: [&'a [u8]; 128],
    metas: [Option<QuickRouteMeta>; 128],
}

fn parse_batch128_refs<'a>(
    body: &'a [u8],
    has_route_meta: bool,
    record_bytes: usize,
) -> Result<Batch128Refs<'a>, String> {
    let mut rows = std::array::from_fn(|_| &body[0..0]);
    let mut metas = [None; 128];
    if has_route_meta {
        for i in 0..128 {
            let off = 16 + i * record_bytes;
            let rec = &body[off..off + record_bytes];
            if rec.len() < 40 || &rec[0..4] != b"RVEC" {
                return Err("batch record missing RVEC header".to_string());
            }
            let ver = u16::from_le_bytes([rec[4], rec[5]]);
            if ver != 3 {
                return Err(format!("unsupported batch RVEC version: {}", ver));
            }
            let flags = u16::from_le_bytes([rec[6], rec[7]]);
            if flags != 0 {
                return Err(format!("unsupported batch RVEC flags: {}", flags));
            }
            let dim = u32::from_le_bytes([rec[8], rec[9], rec[10], rec[11]]) as usize;
            let payload_len = dim
                .checked_mul(4)
                .ok_or_else(|| "batch record payload overflow".to_string())?;
            if rec.len() != 40 + payload_len {
                return Err(format!(
                    "batch record len mismatch: got {} expected {}",
                    rec.len(),
                    40 + payload_len
                ));
            }
            rows[i] = &rec[40..];
            metas[i] = Some(QuickRouteMeta {
                fold_id: i32::from_le_bytes([rec[12], rec[13], rec[14], rec[15]]),
                seg_prod_amtbin: u32::from_le_bytes([rec[16], rec[17], rec[18], rec[19]]),
                transaction_id: u64::from_le_bytes([
                    rec[20], rec[21], rec[22], rec[23], rec[24], rec[25], rec[26], rec[27],
                ]),
                row_idx: u32::from_le_bytes([rec[28], rec[29], rec[30], rec[31]]),
                l2_tau_used: Some(f32::from_le_bytes([rec[32], rec[33], rec[34], rec[35]])),
            });
        }
    } else {
        for i in 0..128 {
            let off = 16 + i * record_bytes;
            rows[i] = &body[off..off + record_bytes];
        }
    }
    Ok(Batch128Refs { rows, metas })
}

async fn score_dense_batch_binary_request_v1(
    st: &AppState,
    body: Bytes,
    throughput_mode: bool,
) -> Result<Bytes, (StatusCode, String)> {
    let Some((expected_dim, _)) = st.core.quick_dims() else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "quickscorer not enabled: start server with --bundle-dir".to_string(),
        ));
    };
    let (_has_route_meta, record_count, record_bytes) =
        parse_batch_dense_payload_le(&body, expected_dim)
            .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    let mut out = Vec::with_capacity(16 + record_count * 24);
    out.extend_from_slice(b"RBR1");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(record_count as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for i in 0..record_count {
        let off = 16 + i * record_bytes;
        let rec = body.slice(off..off + record_bytes);
        let (trace_id, resp) = score_dense_binary_request(st, rec).await?;
        if throughput_mode && should_sample_bench_h2(st) {
            record_bench_h2_sample_metrics(&resp);
        }
        encode_qsb2_record(trace_id, &resp, &mut out);
    }
    Ok(Bytes::from(out))
}

async fn score_dense_batch_binary_request_http_ack_v1(
    st: &AppState,
    body: Bytes,
) -> Result<Bytes, (StatusCode, String)> {
    let Some((expected_dim, _)) = st.core.quick_dims() else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "quickscorer not enabled: start server with --bundle-dir".to_string(),
        ));
    };
    let (_has_route_meta, record_count, record_bytes) =
        parse_batch_dense_payload_le(&body, expected_dim)
            .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    if record_count == 128 {
        return score_dense_batch_binary_request_http_ack_v1_batch128(
            st,
            body,
            _has_route_meta,
            record_bytes,
        );
    }
    let mut used_l2_count = 0u32;
    let mut decision_counts = [0u32; 5];
    st.core.quick.as_ref().ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "quickscorer not enabled: start server with --bundle-dir".to_string(),
        )
    })?;

    if let (Some(rt), Some(tau_mode)) = (st.standalone_l2_bench.as_ref(), st.standalone_l2_tau_mode)
    {
        let mut pending_l2 = Vec::with_capacity(record_count / 4 + 1);
        for i in 0..record_count {
            let off = 16 + i * record_bytes;
            let rec = &body[off..off + record_bytes];
            let (row_bytes, route_meta) = parse_batch_record_ref(rec, _has_route_meta)
                .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
            let l1_out = st
                .core
                .predict_quick_l1_only_bytes(row_bytes)
                .map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("quickscorer l1 inference failed: {:#}", e),
                    )
                })?;
            if l1_out.passed {
                decision_counts[0] += 1;
            } else {
                let sidecar_row_idx = route_meta.as_ref().map(|m| m.row_idx as usize).unwrap_or(i);
                pending_l2.push((route_meta, sidecar_row_idx, l1_out));
            }
        }
        for (route_meta, sidecar_row_idx, l1_out) in pending_l2 {
            let lite = st
                .core
                .score_standalone_l2_from_l1_batch_lite(
                    l1_out,
                    route_meta.as_ref(),
                    rt,
                    sidecar_row_idx,
                    tau_mode,
                )
                .map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("quickscorer standalone batch l2 failed: {:#}", e),
                    )
                })?;
            used_l2_count += u32::from(lite.used_l2);
            let idx = match lite.decision {
                Decision::Allow => 0,
                Decision::Deny => 1,
                Decision::ManualReview => 2,
                Decision::DegradeAllow => 3,
            };
            decision_counts[idx] += 1;
        }
    } else {
        for i in 0..record_count {
            let off = 16 + i * record_bytes;
            let rec = &body[off..off + record_bytes];
            let (row_bytes, route_meta) = parse_batch_record_ref(rec, _has_route_meta)
                .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
            let lite = st
                .core
                .score_quick_dense_bytes_with_meta_batch_lite(row_bytes, route_meta.as_ref())
                .map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("quickscorer batch inference failed: {:#}", e),
                    )
                })?;
            used_l2_count += u32::from(lite.used_l2);
            let idx = match lite.decision {
                Decision::Allow => 0,
                Decision::Deny => 1,
                Decision::ManualReview => 2,
                Decision::DegradeAllow => 3,
            };
            decision_counts[idx] += 1;
        }
    }
    Ok(Bytes::copy_from_slice(&encode_batch_aggregate_ack(
        record_count as u32,
        record_count as u32,
        used_l2_count,
        decision_counts,
    )))
}

fn score_dense_batch_null_request_http_ack_v1(
    st: &AppState,
    body: Bytes,
) -> Result<Bytes, (StatusCode, String)> {
    let Some((expected_dim, _)) = st.core.quick_dims() else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "quickscorer not enabled: start server with --bundle-dir".to_string(),
        ));
    };
    let (_has_route_meta, record_count, _record_bytes) =
        parse_batch_dense_payload_le(&body, expected_dim)
            .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    Ok(Bytes::copy_from_slice(&encode_batch_aggregate_ack(
        record_count as u32,
        record_count as u32,
        0,
        [record_count as u32, 0, 0, 0, 0],
    )))
}

fn score_dense_batch_parse_only_request_http_ack_v1(
    st: &AppState,
    body: Bytes,
) -> Result<Bytes, (StatusCode, String)> {
    let Some((expected_dim, _)) = st.core.quick_dims() else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "quickscorer not enabled: start server with --bundle-dir".to_string(),
        ));
    };
    let (has_route_meta, record_count, record_bytes) =
        parse_batch_dense_payload_le(&body, expected_dim)
            .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    for i in 0..record_count {
        let off = 16 + i * record_bytes;
        let rec = &body[off..off + record_bytes];
        let _ = parse_batch_record_ref(rec, has_route_meta)
            .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    }
    Ok(Bytes::copy_from_slice(&encode_batch_aggregate_ack(
        record_count as u32,
        record_count as u32,
        0,
        [record_count as u32, 0, 0, 0, 0],
    )))
}

fn score_dense_batch_binary_request_http_ack_v1_batch128_fixed(
    st: &AppState,
    body: &[u8],
    has_route_meta: bool,
    record_bytes: usize,
) -> Result<[u8; 40], (StatusCode, String)> {
    let refs = parse_batch128_refs(body, has_route_meta, record_bytes)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;

    st.core.quick.as_ref().ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "quickscorer not enabled: start server with --bundle-dir".to_string(),
        )
    })?;

    let (used_l2_count, decision_counts) = if let (Some(rt), Some(tau_mode)) =
        (st.standalone_l2_bench.as_ref(), st.standalone_l2_tau_mode)
    {
        st.core
            .score_standalone_batch128_http_ack_lite(&refs.rows, &refs.metas, rt, tau_mode)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("quickscorer standalone batch128 inference failed: {:#}", e),
                )
            })?
    } else {
        st.core
            .score_quick_dense_batch128_http_ack_lite(&refs.rows, &refs.metas)
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("quickscorer batch128 inference failed: {:#}", e),
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

fn score_dense_batch_binary_request_http_ack_v1_batch128(
    st: &AppState,
    body: Bytes,
    has_route_meta: bool,
    record_bytes: usize,
) -> Result<Bytes, (StatusCode, String)> {
    score_dense_batch_binary_request_http_ack_v1_batch128_fixed(
        st,
        body.as_ref(),
        has_route_meta,
        record_bytes,
    )
    .map(|ack| Bytes::copy_from_slice(&ack))
}

fn batch_binary_response(bin: Bytes) -> Response {
    let mut r = Response::new(AxumBody::from(bin));
    *r.status_mut() = StatusCode::OK;
    r.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    r
}

fn text_status_response(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, msg.into()).into_response()
}

fn h1_batch_fast_path(path: &str) -> Option<H1BatchFastPath> {
    match path {
        "/score_dense_f32_batch_v1" => Some(H1BatchFastPath::Score),
        "/score_dense_f32_batch_null_v1" => Some(H1BatchFastPath::Null),
        "/score_dense_f32_batch_parseonly_v1" => Some(H1BatchFastPath::ParseOnly),
        _ => None,
    }
}

fn parse_peek_h1_batch_fast_path(buf: &[u8]) -> Option<H1BatchFastPath> {
    let line_end = buf.windows(2).position(|w| w == b"\r\n")?;
    let line = std::str::from_utf8(&buf[..line_end]).ok()?;
    let mut parts = line.split_whitespace();
    let method = parts.next()?;
    if method != "POST" {
        return None;
    }
    let path = parts.next()?;
    h1_batch_fast_path(path)
}

fn parse_peek_h1_batch_dataplane_target(
    buf: &[u8],
    expected_dim: usize,
) -> Option<H1BatchFastPath> {
    let header_end = find_header_end(buf)?;
    let header = std::str::from_utf8(&buf[..header_end]).ok()?;
    let mut lines = header.split("\r\n");
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    if parts.next()? != "POST" {
        return None;
    }
    let fast_path = h1_batch_fast_path(parts.next()?)?;
    let content_len = lines.find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if !name.eq_ignore_ascii_case("content-length") {
            return None;
        }
        value.trim().parse::<usize>().ok()
    })?;
    batch128_shape_for_content_len(expected_dim, content_len)?;
    Some(fast_path)
}

async fn maybe_peek_h1_batch_fast_path(
    stream: &tokio::net::TcpStream,
) -> std::io::Result<Option<H1BatchFastPath>> {
    let mut buf = [0u8; H1_BATCH_PEEK_BYTES];
    let n = stream.peek(&mut buf).await?;
    if n == 0 {
        return Ok(None);
    }
    Ok(parse_peek_h1_batch_fast_path(&buf[..n]))
}

async fn maybe_peek_h1_batch_dataplane_target(
    stream: &tokio::net::TcpStream,
    expected_dim: usize,
) -> std::io::Result<Option<H1BatchFastPath>> {
    let mut buf = [0u8; H1_BATCH_PEEK_BYTES];
    let n = stream.peek(&mut buf).await?;
    if n == 0 {
        return Ok(None);
    }
    Ok(parse_peek_h1_batch_dataplane_target(
        &buf[..n],
        expected_dim,
    ))
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn parse_h1_batch_request_from_buf(
    buf: &mut BytesMut,
) -> Result<Option<H1BatchParsedRequest>, String> {
    let Some(header_end) = find_header_end(buf.as_ref()) else {
        if buf.len() > MAX_H1_BATCH_HEADER_BYTES {
            return Err("request header too large".to_string());
        }
        return Ok(None);
    };

    let header_bytes = &buf[..header_end];
    let header_str =
        std::str::from_utf8(header_bytes).map_err(|e| format!("bad request header: {e}"))?;
    let mut lines = header_str.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| "missing request line".to_string())?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| "missing request method".to_string())?
        .parse::<Method>()
        .map_err(|e| format!("bad request method: {e}"))?;
    let path = parts
        .next()
        .ok_or_else(|| "missing request path".to_string())?;
    let _version = parts
        .next()
        .ok_or_else(|| "missing request http version".to_string())?;

    let mut content_len = None;
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                content_len = Some(
                    value
                        .trim()
                        .parse::<usize>()
                        .map_err(|e| format!("bad content-length: {e}"))?,
                );
            }
        }
    }
    let content_len = content_len.ok_or_else(|| "missing content-length".to_string())?;
    let fast_path = h1_batch_fast_path(path);
    let total_len = (header_end + 4)
        .checked_add(content_len)
        .ok_or_else(|| "request length overflow".to_string())?;
    if buf.len() < total_len {
        return Ok(None);
    }

    let req_bytes = buf.split_to(total_len).freeze();
    let body = req_bytes.slice((header_end + 4)..total_len);
    Ok(Some(H1BatchParsedRequest {
        fast_path,
        method,
        content_len,
        body,
    }))
}

async fn write_h1_batch_reply(
    stream: &mut tokio::net::TcpStream,
    reply: H1BatchReply,
) -> anyhow::Result<()> {
    match reply {
        H1BatchReply::Binary(bin) => {
            debug_assert_eq!(bin.len(), 40, "batch ack must stay fixed-size");
            stream.write_all(H1_BATCH_ACK_PREFIX).await?;
            stream.write_all(bin.as_ref()).await?;
        }
        H1BatchReply::Text(status, msg) => {
            let reason = status.canonical_reason().unwrap_or("Error");
            let head = format!(
                "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nContent-Type: {}\r\n\r\n",
                status.as_u16(),
                reason,
                msg.len(),
                H1_TEXT_PREFIX_CONTENT_TYPE
            );
            stream.write_all(head.as_bytes()).await?;
            stream.write_all(msg.as_bytes()).await?;
        }
    }
    Ok(())
}

async fn serve_h1_batch_dataplane_conn(
    mut stream: tokio::net::TcpStream,
    st: AppState,
) -> anyhow::Result<()> {
    let Some((expected_dim, _)) = st.core.quick_dims() else {
        write_h1_batch_reply(
            &mut stream,
            H1BatchReply::Text(
                StatusCode::INTERNAL_SERVER_ERROR,
                "quickscorer not enabled: start server with --bundle-dir".to_string(),
            ),
        )
        .await?;
        return Ok(());
    };

    let max_route_record_bytes = expected_dim
        .checked_mul(4)
        .and_then(|n| n.checked_add(40))
        .ok_or_else(|| anyhow::anyhow!("batch route record bytes overflow"))?;
    let max_body_len = 16usize
        .checked_add(
            128usize
                .checked_mul(max_route_record_bytes)
                .ok_or_else(|| anyhow::anyhow!("batch max body length overflow"))?,
        )
        .ok_or_else(|| anyhow::anyhow!("batch max body length overflow"))?;
    let max_request_len = MAX_H1_BATCH_HEADER_BYTES
        .checked_add(max_body_len)
        .ok_or_else(|| anyhow::anyhow!("batch max request length overflow"))?;

    let mut buf = BytesMut::with_capacity(max_request_len);

    loop {
        let req = loop {
            match parse_h1_batch_request_from_buf(&mut buf) {
                Ok(Some(req)) => break req,
                Ok(None) => {
                    if buf.len() >= max_request_len {
                        write_h1_batch_reply(
                            &mut stream,
                            H1BatchReply::Text(
                                StatusCode::BAD_REQUEST,
                                "request too large for batch data plane".to_string(),
                            ),
                        )
                        .await?;
                        return Ok(());
                    }
                    let n = stream.read_buf(&mut buf).await?;
                    if n == 0 {
                        if buf.is_empty() {
                            return Ok(());
                        }
                        write_h1_batch_reply(
                            &mut stream,
                            H1BatchReply::Text(
                                StatusCode::BAD_REQUEST,
                                "truncated h1 batch request".to_string(),
                            ),
                        )
                        .await?;
                        return Ok(());
                    }
                }
                Err(msg) => {
                    write_h1_batch_reply(
                        &mut stream,
                        H1BatchReply::Text(StatusCode::BAD_REQUEST, msg),
                    )
                    .await?;
                    return Ok(());
                }
            }
        };

        if req.method != Method::POST {
            write_h1_batch_reply(
                &mut stream,
                H1BatchReply::Text(
                    StatusCode::METHOD_NOT_ALLOWED,
                    "method not allowed".to_string(),
                ),
            )
            .await?;
            return Ok(());
        }

        let Some(fast_path) = req.fast_path else {
            write_h1_batch_reply(
                &mut stream,
                H1BatchReply::Text(StatusCode::NOT_FOUND, "not found".to_string()),
            )
            .await?;
            return Ok(());
        };

        let _permit = match st.bench_in_flight.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                write_h1_batch_reply(
                    &mut stream,
                    H1BatchReply::Text(StatusCode::TOO_MANY_REQUESTS, "overloaded".to_string()),
                )
                .await?;
                return Ok(());
            }
        };

        let reply =
            execute_h1_batch_fast_request(&st, fast_path, Some(req.content_len), req.body).await;
        let should_close = matches!(&reply, H1BatchReply::Text(_, _));
        write_h1_batch_reply(&mut stream, reply).await?;
        if should_close {
            return Ok(());
        }
    }
}

fn spawn_h1_batch_dataplane_tokio_shards(
    shard_count: usize,
    st: AppState,
) -> Vec<mpsc::UnboundedSender<std::net::TcpStream>> {
    let mut senders = Vec::with_capacity(shard_count.max(1));
    for shard_idx in 0..shard_count.max(1) {
        let (tx, mut rx) = mpsc::unbounded_channel::<std::net::TcpStream>();
        let st2 = st.clone();
        std::thread::Builder::new()
            .name(format!("h1-batch-dp-{shard_idx}"))
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("build h1 batch dataplane runtime");
                rt.block_on(async move {
                    let mut tasks = tokio::task::JoinSet::new();
                    loop {
                        tokio::select! {
                            Some(std_stream) = rx.recv() => {
                                let st3 = st2.clone();
                                tasks.spawn(async move {
                                    match tokio::net::TcpStream::from_std(std_stream) {
                                        Ok(stream) => {
                                            if let Err(e) = serve_h1_batch_dataplane_conn(stream, st3).await {
                                                warn!(error = %e, shard = shard_idx, "h1 batch dataplane conn failed");
                                            }
                                        }
                                        Err(e) => {
                                            warn!(error = %e, shard = shard_idx, "convert batch dataplane stream failed");
                                        }
                                    }
                                });
                            }
                            Some(joined) = tasks.join_next(), if !tasks.is_empty() => {
                                if let Err(e) = joined {
                                    warn!(error = %e, shard = shard_idx, "h1 batch dataplane task join failed");
                                }
                            }
                            else => break,
                        }
                    }
                    while let Some(joined) = tasks.join_next().await {
                        if let Err(e) = joined {
                            warn!(error = %e, shard = shard_idx, "h1 batch dataplane task join failed");
                        }
                    }
                });
            })
            .expect("spawn h1 batch dataplane shard");
        senders.push(tx);
    }
    senders
}

fn queue_mio_text_reply(conn: &mut MioBatchConn, status: StatusCode, msg: impl Into<String>) {
    let msg = msg.into();
    conn.clear_write_buf();
    conn.write_buf = build_h1_text_reply(status, &msg);
    conn.close_after_write = true;
}

fn queue_mio_binary_reply(conn: &mut MioBatchConn, ack: [u8; 40]) {
    conn.clear_write_buf();
    fill_h1_binary_reply(&mut conn.write_buf, ack);
}

fn mio_conn_interest(conn: &MioBatchConn) -> Interest {
    if conn.has_pending_write() {
        Interest::READABLE.add(Interest::WRITABLE)
    } else {
        Interest::READABLE
    }
}

fn read_mio_batch_conn(conn: &mut MioBatchConn, max_request_len: usize) -> io::Result<bool> {
    let mut scratch = [0u8; 8192];
    loop {
        match conn.stream.read(&mut scratch) {
            Ok(0) => return Ok(true),
            Ok(n) => {
                if conn.read_buf.len().saturating_add(n) > max_request_len {
                    queue_mio_text_reply(
                        conn,
                        StatusCode::BAD_REQUEST,
                        "request too large for batch data plane",
                    );
                    return Ok(false);
                }
                conn.read_buf.extend_from_slice(&scratch[..n]);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(false),
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::ConnectionReset
                        | io::ErrorKind::BrokenPipe
                        | io::ErrorKind::UnexpectedEof
                ) =>
            {
                return Ok(true)
            }
            Err(e) => return Err(e),
        }
    }
}

fn write_mio_batch_conn(conn: &mut MioBatchConn) -> io::Result<bool> {
    while conn.has_pending_write() {
        match conn.stream.write(&conn.write_buf[conn.write_off..]) {
            Ok(0) => return Ok(true),
            Ok(n) => conn.write_off += n,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(false),
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::ConnectionReset
                        | io::ErrorKind::BrokenPipe
                        | io::ErrorKind::UnexpectedEof
                ) =>
            {
                return Ok(true)
            }
            Err(e) => return Err(e),
        }
    }

    if conn.write_off == conn.write_buf.len() {
        let should_close = conn.close_after_write;
        conn.clear_write_buf();
        if should_close {
            return Ok(true);
        }
    }
    Ok(false)
}

fn process_one_mio_batch_request(
    conn: &mut MioBatchConn,
    st: &AppState,
    expected_dim: usize,
) -> io::Result<bool> {
    let req = match parse_h1_batch_request_from_buf(&mut conn.read_buf) {
        Ok(Some(req)) => req,
        Ok(None) => return Ok(false),
        Err(msg) => {
            queue_mio_text_reply(conn, StatusCode::BAD_REQUEST, msg);
            return Ok(true);
        }
    };

    if req.method != Method::POST {
        queue_mio_text_reply(conn, StatusCode::METHOD_NOT_ALLOWED, "method not allowed");
        return Ok(true);
    }
    let Some(fast_path) = req.fast_path else {
        queue_mio_text_reply(conn, StatusCode::NOT_FOUND, "not found");
        return Ok(true);
    };
    let Some(shape) = batch128_shape_for_content_len(expected_dim, req.content_len) else {
        queue_mio_text_reply(
            conn,
            StatusCode::BAD_REQUEST,
            "batch dataplane only supports fixed batch128 requests",
        );
        return Ok(true);
    };

    let _permit = match st.bench_in_flight.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            queue_mio_text_reply(conn, StatusCode::TOO_MANY_REQUESTS, "overloaded");
            return Ok(true);
        }
    };

    match execute_h1_batch_fixed_request(st, fast_path, shape, req.body.as_ref()) {
        Ok(ack) => queue_mio_binary_reply(conn, ack),
        Err((status, msg)) => queue_mio_text_reply(conn, status, msg),
    }
    Ok(true)
}

fn spawn_h1_batch_dataplane_mio_shards(
    shard_count: usize,
    st: AppState,
) -> anyhow::Result<Vec<MioShardHandle>> {
    let Some((expected_dim, _)) = st.core.quick_dims() else {
        anyhow::bail!("quickscorer not enabled: start server with --bundle-dir");
    };

    let max_route_record_bytes = expected_dim
        .checked_mul(4)
        .and_then(|n| n.checked_add(40))
        .ok_or_else(|| anyhow::anyhow!("batch route record bytes overflow"))?;
    let max_body_len = 16usize
        .checked_add(
            128usize
                .checked_mul(max_route_record_bytes)
                .ok_or_else(|| anyhow::anyhow!("batch max body length overflow"))?,
        )
        .ok_or_else(|| anyhow::anyhow!("batch max body length overflow"))?;
    let max_request_len = MAX_H1_BATCH_HEADER_BYTES
        .checked_add(max_body_len)
        .ok_or_else(|| anyhow::anyhow!("batch max request length overflow"))?;

    let mut handles = Vec::with_capacity(shard_count.max(1));
    for shard_idx in 0..shard_count.max(1) {
        let (tx, rx) = std_mpsc::channel::<std::net::TcpStream>();
        let st2 = st.clone();
        let (waker_tx, waker_rx) = std_mpsc::channel::<Arc<Waker>>();
        std::thread::Builder::new()
            .name(format!("h1-batch-dp-mio-{shard_idx}"))
            .spawn(move || {
                let mut poll = Poll::new().expect("create mio poll");
                let waker = Arc::new(
                    Waker::new(poll.registry(), H1_BATCH_WAKE_TOKEN)
                        .expect("create mio batch shard waker"),
                );
                waker_tx
                    .send(waker.clone())
                    .expect("send mio batch shard waker");

                let mut events = Events::with_capacity(1024);
                let mut conns: Vec<Option<MioBatchConn>> = Vec::new();
                let mut free_slots = Vec::new();

                loop {
                    if let Err(e) = poll.poll(&mut events, None) {
                        warn!(error = %e, shard = shard_idx, "mio batch dataplane poll failed");
                        continue;
                    }

                    for event in events.iter() {
                        if event.token() == H1_BATCH_WAKE_TOKEN {
                            loop {
                                match rx.try_recv() {
                                    Ok(std_stream) => {
                                        if let Err(e) = std_stream.set_nonblocking(true) {
                                            warn!(error = %e, shard = shard_idx, "set batch dataplane stream nonblocking failed");
                                            continue;
                                        }
                                        let mut stream = mio::net::TcpStream::from_std(std_stream);
                                        let slot = free_slots.pop().unwrap_or_else(|| {
                                            conns.push(None);
                                            conns.len() - 1
                                        });
                                        let token = Token(slot + 1);
                                        if let Err(e) = poll
                                            .registry()
                                            .register(&mut stream, token, Interest::READABLE)
                                        {
                                            warn!(error = %e, shard = shard_idx, "register batch dataplane stream failed");
                                            if slot + 1 == conns.len() {
                                                conns.pop();
                                            } else {
                                                free_slots.push(slot);
                                            }
                                            continue;
                                        }
                                        conns[slot] = Some(MioBatchConn::new(stream, max_request_len));
                                    }
                                    Err(std_mpsc::TryRecvError::Empty) => break,
                                    Err(std_mpsc::TryRecvError::Disconnected) => return,
                                }
                            }
                            continue;
                        }

                        let slot = event.token().0.saturating_sub(1);
                        let Some(conn) = conns.get_mut(slot).and_then(Option::as_mut) else {
                            continue;
                        };

                        let mut should_close =
                            event.is_error() || event.is_read_closed() || event.is_write_closed();

                        if !should_close && conn.has_pending_write() && event.is_writable() {
                            match write_mio_batch_conn(conn) {
                                Ok(close) => should_close = close,
                                Err(e) => {
                                    warn!(error = %e, shard = shard_idx, "write batch dataplane conn failed");
                                    should_close = true;
                                }
                            }
                        }

                        if !should_close && !conn.has_pending_write() && event.is_readable() {
                            match read_mio_batch_conn(conn, max_request_len) {
                                Ok(close) => should_close = close,
                                Err(e) => {
                                    warn!(error = %e, shard = shard_idx, "read batch dataplane conn failed");
                                    should_close = true;
                                }
                            }
                        }

                        if !should_close && !conn.has_pending_write() {
                            loop {
                                let queued = match process_one_mio_batch_request(conn, &st2, expected_dim) {
                                    Ok(queued) => queued,
                                    Err(e) => {
                                        warn!(error = %e, shard = shard_idx, "process batch dataplane request failed");
                                        should_close = true;
                                        break;
                                    }
                                };
                                if !queued {
                                    break;
                                }
                                match write_mio_batch_conn(conn) {
                                    Ok(close) => {
                                        should_close = close;
                                        if should_close || conn.has_pending_write() {
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        warn!(error = %e, shard = shard_idx, "flush batch dataplane reply failed");
                                        should_close = true;
                                        break;
                                    }
                                }
                            }
                        }

                        if should_close {
                            if let Some(mut conn) = conns[slot].take() {
                                let _ = poll.registry().deregister(&mut conn.stream);
                                free_slots.push(slot);
                            }
                            continue;
                        }

                        let interest = mio_conn_interest(conn);
                        if let Err(e) = poll
                            .registry()
                            .reregister(&mut conn.stream, Token(slot + 1), interest)
                        {
                            warn!(error = %e, shard = shard_idx, "reregister batch dataplane conn failed");
                            if let Some(mut conn) = conns[slot].take() {
                                let _ = poll.registry().deregister(&mut conn.stream);
                                free_slots.push(slot);
                            }
                        }
                    }
                }
            })
            .expect("spawn mio h1 batch dataplane shard");

        let waker = waker_rx.recv().expect("receive mio batch shard waker");
        handles.push(MioShardHandle { tx, waker });
    }
    Ok(handles)
}

async fn read_fixed_body(mut body: Incoming, content_len: usize) -> Result<Bytes, String> {
    let mut buf = BytesMut::with_capacity(content_len);
    while let Some(frame) = body.frame().await {
        let frame = frame.map_err(|e| format!("read request body failed: {e}"))?;
        let Ok(data) = frame.into_data() else {
            continue;
        };
        if buf.len().saturating_add(data.len()) > content_len {
            return Err(format!(
                "request body exceeded content-length: {} > {}",
                buf.len() + data.len(),
                content_len
            ));
        }
        buf.extend_from_slice(&data);
    }
    if buf.len() != content_len {
        return Err(format!(
            "request body truncated: got {} expected {}",
            buf.len(),
            content_len
        ));
    }
    Ok(buf.freeze())
}

fn score_dense_batch_null_request_http_ack_v1_batch128_fixed(
    body: &[u8],
    has_route_meta: bool,
    record_bytes: usize,
) -> Result<[u8; 40], (StatusCode, String)> {
    validate_batch128_header(body, has_route_meta, record_bytes)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    Ok(encode_batch_aggregate_ack(128, 128, 0, [128, 0, 0, 0, 0]))
}

fn score_dense_batch_parse_only_request_http_ack_v1_batch128_fixed(
    body: Bytes,
    has_route_meta: bool,
    record_bytes: usize,
) -> Result<[u8; 40], (StatusCode, String)> {
    validate_batch128_header(body.as_ref(), has_route_meta, record_bytes)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    let refs = parse_batch128_refs(body.as_ref(), has_route_meta, record_bytes)
        .map_err(|msg| (StatusCode::BAD_REQUEST, msg))?;
    let _ = refs;
    Ok(encode_batch_aggregate_ack(128, 128, 0, [128, 0, 0, 0, 0]))
}

fn execute_h1_batch_fixed_request(
    st: &AppState,
    fast_path: H1BatchFastPath,
    shape: Batch128Shape,
    body: &[u8],
) -> Result<[u8; 40], (StatusCode, String)> {
    match fast_path {
        H1BatchFastPath::Score => score_dense_batch_binary_request_http_ack_v1_batch128_fixed(
            st,
            body,
            shape.has_route_meta,
            shape.record_bytes,
        ),
        H1BatchFastPath::Null => score_dense_batch_null_request_http_ack_v1_batch128_fixed(
            body,
            shape.has_route_meta,
            shape.record_bytes,
        ),
        H1BatchFastPath::ParseOnly => {
            score_dense_batch_parse_only_request_http_ack_v1_batch128_fixed(
                Bytes::copy_from_slice(body),
                shape.has_route_meta,
                shape.record_bytes,
            )
        }
    }
}

async fn execute_h1_batch_fast_request(
    st: &AppState,
    fast_path: H1BatchFastPath,
    content_len: Option<usize>,
    body: Bytes,
) -> H1BatchReply {
    let Some((expected_dim, _)) = st.core.quick_dims() else {
        return H1BatchReply::Text(
            StatusCode::INTERNAL_SERVER_ERROR,
            "quickscorer not enabled: start server with --bundle-dir".to_string(),
        );
    };

    let outcome = if let Some(shape) =
        content_len.and_then(|len| batch128_shape_for_content_len(expected_dim, len))
    {
        execute_h1_batch_fixed_request(st, fast_path, shape, body.as_ref())
            .map(|ack| Bytes::copy_from_slice(&ack))
    } else {
        match fast_path {
            H1BatchFastPath::Score => score_dense_batch_binary_request_http_ack_v1(st, body).await,
            H1BatchFastPath::Null => score_dense_batch_null_request_http_ack_v1(st, body),
            H1BatchFastPath::ParseOnly => {
                score_dense_batch_parse_only_request_http_ack_v1(st, body)
            }
        }
    };

    match outcome {
        Ok(bin) => H1BatchReply::Binary(bin),
        Err((status, msg)) => H1BatchReply::Text(status, msg),
    }
}

async fn handle_h1_batch_fast_request(
    st: AppState,
    fast_path: H1BatchFastPath,
    req: http::Request<Incoming>,
) -> Response {
    if req.method() != Method::POST {
        return text_status_response(StatusCode::METHOD_NOT_ALLOWED, "method not allowed");
    }

    let _permit = match st.bench_in_flight.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => return text_status_response(StatusCode::TOO_MANY_REQUESTS, "overloaded"),
    };

    let Some((expected_dim, _)) = st.core.quick_dims() else {
        return text_status_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "quickscorer not enabled: start server with --bundle-dir",
        );
    };

    let content_len = req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    let (_, body) = req.into_parts();
    let body = if let Some(shape) =
        content_len.and_then(|len| batch128_shape_for_content_len(expected_dim, len))
    {
        match read_fixed_body(body, shape.content_len).await {
            Ok(body) => body,
            Err(msg) => return text_status_response(StatusCode::BAD_REQUEST, msg),
        }
    } else {
        match body.collect().await {
            Ok(collected) => collected.to_bytes(),
            Err(e) => {
                return text_status_response(
                    StatusCode::BAD_REQUEST,
                    format!("read request body failed: {e}"),
                )
            }
        }
    };

    match execute_h1_batch_fast_request(&st, fast_path, content_len, body).await {
        H1BatchReply::Binary(bin) => batch_binary_response(bin),
        H1BatchReply::Text(status, msg) => {
            error!(error = %msg, "h1 batch fast path failed");
            text_status_response(status, msg)
        }
    }
}

async fn serve_http1_conn(
    stream: tokio::net::TcpStream,
    st: AppState,
    app: Router,
) -> anyhow::Result<()> {
    let svc = service_fn(move |req: http::Request<Incoming>| {
        let st = st.clone();
        let app = app.clone();
        async move {
            if let Some(fast_path) = h1_batch_fast_path(req.uri().path()) {
                return Ok::<_, Infallible>(handle_h1_batch_fast_request(st, fast_path, req).await);
            }

            let (parts, body) = req.into_parts();
            let req = http::Request::from_parts(parts, AxumBody::new(body));
            let resp = app
                .oneshot(req)
                .await
                .expect("axum router service is infallible");
            Ok::<_, Infallible>(resp)
        }
    });

    http1::Builder::new()
        .keep_alive(true)
        .serve_connection(TokioIo::new(stream), svc)
        .await
        .context("serve h1 connection")?;
    Ok(())
}

fn encode_timings_header_value(resp: &ScoreResponse) -> Option<HeaderValue> {
    let ts = &resp.timings_us;
    HeaderValue::from_str(&format!(
        "{},{},{},{},{},{}",
        ts.parse, ts.feature, ts.router, ts.l1, ts.l2, ts.serialize
    ))
    .ok()
}

fn is_throughput_bench_request<B>(req: &http::Request<B>) -> bool {
    req.headers()
        .get("x-risk-bench-mode")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("throughput"))
        .unwrap_or(false)
}

fn should_sample_bench_h2(st: &AppState) -> bool {
    let rate = st.bench_h2_sample_rate.max(1) as u64;
    let seq = st.bench_h2_sample_seq.fetch_add(1, Ordering::Relaxed);
    seq % rate == 0
}

fn record_bench_h2_sample_metrics(resp: &ScoreResponse) {
    let ts = &resp.timings_us;
    metrics::counter!("bench3_h2_samples_total").increment(1);
    metrics::histogram!("bench3_h2_stage_parse_us").record(ts.parse as f64);
    metrics::histogram!("bench3_h2_stage_feature_us").record(ts.feature as f64);
    metrics::histogram!("bench3_h2_stage_router_us").record(ts.router as f64);
    metrics::histogram!("bench3_h2_stage_l1_us").record(ts.l1 as f64);
    metrics::histogram!("bench3_h2_stage_l2_us").record(ts.l2 as f64);
    metrics::histogram!("bench3_h2_stage_serialize_us").record(ts.serialize as f64);
    metrics::histogram!("bench3_h2_server_total_us").record(
        ts.parse as f64
            + ts.feature as f64
            + ts.router as f64
            + ts.l1 as f64
            + ts.l2 as f64
            + ts.serialize as f64,
    );
    if ts.l2 > 0 {
        metrics::counter!("bench3_h2_used_l2_total").increment(1);
    }
}

async fn handle_h2_bench_stream(
    req: http::Request<h2::RecvStream>,
    mut respond: h2::server::SendResponse<Bytes>,
    st: AppState,
) -> anyhow::Result<()> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let throughput_mode = is_throughput_bench_request(&req);
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

    if path != "/score_dense_f32_bin"
        && path != "/score_dense_f32_bin_v2"
        && path != "/score_dense_f32_batch_v1"
    {
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

    let body_capacity = req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4096);
    let mut body_stream = req.into_body();
    let mut body = bytes::BytesMut::with_capacity(body_capacity);
    while let Some(chunk) = body_stream.data().await {
        let chunk = chunk.context("read h2 request body")?;
        body.extend_from_slice(&chunk);
        let _ = body_stream.flow_control().release_capacity(chunk.len());
    }
    let body = body.freeze();

    let outcome = match path.as_str() {
        "/score_dense_f32_bin" => {
            score_dense_binary_request(&st, body)
                .await
                .map(|(trace_id, resp)| {
                    if throughput_mode && should_sample_bench_h2(&st) {
                        record_bench_h2_sample_metrics(&resp);
                    }
                    let bin = encode_rsk1_response(trace_id, &resp);
                    let headers: Vec<(HeaderName, HeaderValue)> = vec![(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("application/octet-stream"),
                    )];
                    (StatusCode::OK, headers, Bytes::from(bin))
                })
        }
        "/score_dense_f32_bin_v2" => {
            score_dense_binary_request(&st, body)
                .await
                .map(|(trace_id, resp)| {
                    if throughput_mode && should_sample_bench_h2(&st) {
                        record_bench_h2_sample_metrics(&resp);
                    }
                    let mut headers = vec![(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("application/octet-stream"),
                    )];
                    if !throughput_mode {
                        if let Some(v) = encode_timings_header_value(&resp) {
                            headers.push((HDR_RISK_TIMINGS_US, v));
                        }
                    }
                    (
                        StatusCode::OK,
                        headers,
                        Bytes::from(encode_qsb2_response(trace_id, &resp)),
                    )
                })
        }
        "/score_dense_f32_batch_v1" => {
            score_dense_batch_binary_request_v1(&st, body, throughput_mode)
                .await
                .map(|bin| {
                    let headers: Vec<(HeaderName, HeaderValue)> = vec![(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("application/octet-stream"),
                    )];
                    (StatusCode::OK, headers, bin)
                })
        }
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
    let mut send = respond
        .send_response(response, false)
        .context("send h2 headers")?;
    send.send_data(body, true).context("send h2 body")?;
    Ok(())
}

async fn handle_h2_bench_conn(stream: tokio::net::TcpStream, st: AppState) -> anyhow::Result<()> {
    let mut conn = h2::server::handshake(stream)
        .await
        .context("h2 server handshake")?;
    let mut tasks = tokio::task::JoinSet::new();
    while let Some(result) = conn.accept().await {
        let (req, respond) = result.context("accept h2 stream")?;
        let st2 = st.clone();
        tasks.spawn(async move {
            if let Err(e) = handle_h2_bench_stream(req, respond, st2).await {
                warn!(error = %e, "h2 bench stream failed");
            }
        });
        while tasks.len() > 1024 {
            let _ = tasks.join_next().await;
        }
    }
    while tasks.join_next().await.is_some() {}
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

/// Dense f32 request path with a binary response.
/// The request body is raw f32le or an RVEC-framed dense payload.
async fn score_dense_f32_bin(State(st): State<AppState>, body: Bytes) -> Response {
    match score_dense_binary_request(&st, body).await {
        Ok((trace_id, resp)) => {
            // Keep timings.serialize aligned with binary encoding time.
            let t_ser = Instant::now();
            // Use a single encode pass and record serialize_us via metrics.
            let bin = encode_rsk1_response(trace_id, &resp);
            let _serialize_us = t_ser.elapsed().as_micros() as u64;
            let mut r = Response::new(axum::body::Body::from(bin));
            *r.status_mut() = StatusCode::OK;
            r.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            );
            // Keep trace_id in the response and avoid logging it on the hot path.
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

async fn score_dense_f32_batch_v1(State(st): State<AppState>, body: Bytes) -> Response {
    match score_dense_batch_binary_request_http_ack_v1(&st, body).await {
        Ok(bin) => {
            let mut r = Response::new(axum::body::Body::from(bin));
            *r.status_mut() = StatusCode::OK;
            r.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            );
            r
        }
        Err((status, msg)) => {
            error!(error = %msg, "score_dense_f32_batch_v1 failed");
            (status, msg).into_response()
        }
    }
}

async fn score_dense_f32_batch_null_v1(State(st): State<AppState>, body: Bytes) -> Response {
    match score_dense_batch_null_request_http_ack_v1(&st, body) {
        Ok(bin) => {
            let mut r = Response::new(axum::body::Body::from(bin));
            *r.status_mut() = StatusCode::OK;
            r.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            );
            r
        }
        Err((status, msg)) => {
            error!(error = %msg, "score_dense_f32_batch_null_v1 failed");
            (status, msg).into_response()
        }
    }
}

async fn score_dense_f32_batch_parseonly_v1(State(st): State<AppState>, body: Bytes) -> Response {
    match score_dense_batch_parse_only_request_http_ack_v1(&st, body) {
        Ok(bin) => {
            let mut r = Response::new(axum::body::Body::from(bin));
            *r.status_mut() = StatusCode::OK;
            r.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/octet-stream"),
            );
            r
        }
        Err((status, msg)) => {
            error!(error = %msg, "score_dense_f32_batch_parseonly_v1 failed");
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
        bench_h2_sample_rate: args.bench3_h2_sample_rate.max(1),
        bench_h2_sample_seq: Arc::new(AtomicU64::new(
            BENCH_H2_REQ_SEQ.fetch_add(1, Ordering::Relaxed),
        )),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/debug/backend", get(debug_backend))
        .route("/score_dense_f32_bin", post(score_dense_f32_bin))
        .route("/score_dense_f32_bin_v2", post(score_dense_f32_bin_v2))
        .route("/score_dense_f32_batch_v1", post(score_dense_f32_batch_v1))
        .route(
            "/score_dense_f32_batch_null_v1",
            post(score_dense_f32_batch_null_v1),
        )
        .route(
            "/score_dense_f32_batch_parseonly_v1",
            post(score_dense_f32_batch_parseonly_v1),
        )
        .with_state(st.clone())
        .layer(
            ServiceBuilder::new()
                // HandleErrorLayer must stay outermost so overload becomes HTTP 429.
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
    let batch_expected_dim = st.core.quick_dims().map(|(dim, _)| dim);
    let h1_batch_tokio_shards = if args.h1_batch_dataplane == H1BatchDataplaneMode::TokioShard {
        Some(spawn_h1_batch_dataplane_tokio_shards(
            worker_threads.max(1),
            st.clone(),
        ))
    } else {
        None
    };
    let h1_batch_mio_shards = if args.h1_batch_dataplane == H1BatchDataplaneMode::MioShard {
        Some(spawn_h1_batch_dataplane_mio_shards(
            worker_threads.max(1),
            st.clone(),
        )?)
    } else {
        None
    };
    let mut next_h1_batch_shard = 0usize;

    if let Some(h2_addr) = args.bench3_h2_listen.as_ref() {
        let h2_addr = SocketAddr::from_str(h2_addr).context("invalid --bench3-h2-listen")?;
        let h2_state = st.clone();
        tokio::spawn(async move {
            if let Err(e) = run_h2_bench_listener(h2_addr, h2_state).await {
                error!(error = %e, "h2 bench listener failed");
            }
        });
    }

    loop {
        let (stream, peer) = listener.accept().await.context("accept h1 connection")?;
        stream.set_nodelay(true).ok();
        if let Some(h1_batch_shards) = h1_batch_tokio_shards.as_ref() {
            if let Some(_) = maybe_peek_h1_batch_fast_path(&stream)
                .await
                .context("peek h1 batch path")?
            {
                let shard_idx = next_h1_batch_shard % h1_batch_shards.len().max(1);
                next_h1_batch_shard = next_h1_batch_shard.wrapping_add(1);
                let std_stream = stream
                    .into_std()
                    .context("convert h1 batch stream to std")?;
                if h1_batch_shards[shard_idx].send(std_stream).is_err() {
                    return Err(anyhow::anyhow!(
                        "h1 batch dataplane shard {shard_idx} is closed"
                    ));
                }
                continue;
            }
        } else if let (Some(h1_batch_shards), Some(expected_dim)) =
            (h1_batch_mio_shards.as_ref(), batch_expected_dim)
        {
            if let Some(_) = maybe_peek_h1_batch_dataplane_target(&stream, expected_dim)
                .await
                .context("peek h1 batch dataplane target")?
            {
                let shard_idx = next_h1_batch_shard % h1_batch_shards.len().max(1);
                next_h1_batch_shard = next_h1_batch_shard.wrapping_add(1);
                let std_stream = stream
                    .into_std()
                    .context("convert h1 batch stream to std")?;
                h1_batch_shards[shard_idx]
                    .tx
                    .send(std_stream)
                    .map_err(|_| {
                        anyhow::anyhow!("h1 batch dataplane shard {shard_idx} is closed")
                    })?;
                h1_batch_shards[shard_idx]
                    .waker
                    .wake()
                    .context("wake h1 batch dataplane shard")?;
                continue;
            }
        }
        let st2 = st.clone();
        let app2 = app.clone();
        tokio::spawn(async move {
            if let Err(e) = serve_http1_conn(stream, st2, app2).await {
                warn!(peer = %peer, error = %e, "h1 connection failed");
            }
        });
    }
}

fn main() -> anyhow::Result<()> {
    // Keep the existing environment-variable startup path.
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
