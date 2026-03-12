use anyhow::{anyhow, bail, Context, Result};
use bytes::Bytes;
use clap::{Parser, ValueEnum};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use crossbeam_queue::ArrayQueue;
use futures_util::stream::{FuturesUnordered, StreamExt};
use hdrhistogram::Histogram;
use http::Uri;
use memmap2::Mmap;
use mio::net::TcpStream;
use mio::{Events, Interest, Poll, Token};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::cmp::min;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{self, IoSlice, Read, Write};
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::PathBuf;
use std::str;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tokio::runtime::Builder as TokioRuntimeBuilder;
use tokio::sync::mpsc as tokio_mpsc;

const MAX_RESPONSE_BYTES: usize = 64 * 1024;
const FLUSH_BATCH: usize = 256;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about)]
struct Args {
    #[arg(long)]
    url: String,

    #[arg(long)]
    dense_file: String,

    #[arg(long)]
    route_meta_tsv: Option<String>,

    #[arg(long, value_enum, default_value_t = WorkloadMode::Corpus)]
    workload: WorkloadMode,

    #[arg(long)]
    payload_file: Option<String>,

    #[arg(long)]
    dense_dim: usize,

    #[arg(long, default_value_t = 10_000)]
    rps: u64,

    #[arg(long, default_value_t = 20)]
    duration: u64,

    #[arg(long, default_value_t = 0)]
    warmup: u64,

    #[arg(long, default_value_t = 0)]
    workers: usize,

    #[arg(long)]
    worker_cpus: Option<String>,

    #[arg(long)]
    pacer_cpu: Option<usize>,

    #[arg(long, default_value_t = 32)]
    conns_per_worker: usize,

    #[arg(long, default_value_t = 1)]
    max_inflight_per_conn: usize,

    #[arg(long, default_value_t = 2000)]
    timeout_ms: u64,

    #[arg(long, value_enum, default_value_t = PacerKind::Poisson)]
    pacer: PacerKind,

    #[arg(long, default_value_t = 200)]
    window_ms: u64,

    #[arg(long)]
    window_csv: Option<PathBuf>,

    #[arg(long)]
    summary_json: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = ProtocolMode::Auto)]
    protocol: ProtocolMode,

    #[arg(long, value_enum, default_value_t = BenchMode::Latency)]
    mode: BenchMode,

    #[arg(long, value_enum, default_value_t = BatchMode::Off)]
    batch_mode: BatchMode,

    #[arg(long, default_value_t = 64)]
    batch_records: usize,

    #[arg(long, default_value_t = 256)]
    throughput_batch_size: usize,

    #[arg(long, default_value_t = false)]
    progress: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
enum PacerKind {
    Fixed,
    Poisson,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
enum ProtocolMode {
    Auto,
    Qsb2,
    Rsk1,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
enum TransportKind {
    Http,
    H2c,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, PartialEq, Eq)]
enum BenchMode {
    Latency,
    Throughput,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, PartialEq, Eq)]
enum BatchMode {
    Off,
    Http1,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, PartialEq, Eq)]
enum WorkloadMode {
    Corpus,
    Ceiling,
}

#[derive(Clone, Copy, Debug, Serialize)]
enum Verdict {
    Pass,
    ClientLimited,
    ServerLimited,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct StageTimingsUs {
    parse: u32,
    feature: u32,
    router: u32,
    l1: u32,
    l2: u32,
    serialize: u32,
}

#[derive(Clone, Copy, Debug)]
struct Qsb2 {
    decision: u8,
    flags: u8,
    timings: StageTimingsUs,
}

#[derive(Clone, Copy, Debug)]
struct Rsk1 {
    decision: u8,
    flags: u16,
    timings: StageTimingsUs,
}

#[derive(Clone, Copy, Debug)]
enum BodyDecoded {
    Qsb2(Qsb2),
    Rsk1(Rsk1),
    BatchAck(BatchAck),
}

#[derive(Clone, Copy, Debug)]
struct BatchAck {
    record_count: u32,
    ok_count: u32,
    used_l2_count: u32,
    decision_counts: [u32; 5],
}

#[derive(Clone, Copy, Debug)]
struct HttpResponseMeta {
    status_code: u16,
    timings_header: Option<StageTimingsUs>,
}

#[derive(Clone, Copy, Debug)]
struct ReqToken {
    row_idx: usize,
    t_sched: Instant,
    record: bool,
    rows: u32,
}

#[derive(Clone, Copy, Debug)]
struct QueuedReq {
    row_idx: usize,
    t_sched: Instant,
    t_issue: Instant,
    record: bool,
    rows: u32,
}

#[derive(Clone, Copy, Debug)]
struct ActiveReq {
    row_idx: usize,
    t_sched: Instant,
    t_issue: Instant,
    record: bool,
    t_send_done: Option<Instant>,
    deadline: Instant,
    route_header: [u8; 40],
    route_header_len: usize,
}

#[derive(Clone, Debug)]
struct H2SendCmd {
    request_id: u32,
    body: Bytes,
}

#[derive(Debug)]
enum H2RespEvent {
    Completed {
        request_id: u32,
        meta: HttpResponseMeta,
        body: BodyDecoded,
    },
    Failed {
        request_id: u32,
    },
}

#[derive(Clone, Copy, Debug)]
struct H2InflightReq {
    active: ActiveReq,
    conn_idx: usize,
}

struct H2ThroughputConn {
    sender: h2::client::SendRequest<Bytes>,
}

#[derive(Clone, Copy, Debug)]
struct ThroughputInflightReq {
    conn_idx: usize,
    batch_id: Option<u64>,
    deadline: Instant,
}

#[derive(Debug, Default)]
struct ThroughputQuota {
    warmup: AtomicU64,
    record: AtomicU64,
}

#[derive(Clone)]
enum WorkloadSource {
    Corpus {
        payload: PayloadCorpus,
        route_meta: Option<RouteMetaCorpus>,
    },
    Ceiling {
        body: Arc<Bytes>,
    },
}

#[derive(Clone, Copy, Debug, Serialize)]
struct CompletedSample {
    status_code: u16,
    e2e_us: u64,
    queue_delay_us: u64,
    send_delay_us: u64,
    server_rtt_us: u64,
    timeout: bool,
    used_l2: bool,
    decision: u8,
    qsb2: bool,
    rsk1: bool,
    stage: StageTimingsUs,
}

#[derive(Clone, Copy, Debug)]
struct ThroughputResponse {
    status_code: u16,
    timeout: bool,
    rows: u32,
    used_l2_count: u32,
    decision_counts: [u32; 5],
    qsb2_samples: u32,
    rsk1_samples: u32,
}

type H2ThroughputRespFuture = std::pin::Pin<
    Box<dyn std::future::Future<Output = (usize, u32, Result<ThroughputResponse>)> + Send>,
>;

#[derive(Clone, Debug)]
struct ThroughputBatchState {
    issued_at: Instant,
    outstanding: u32,
    sealed: bool,
    ok: u64,
    err: u64,
    timeout: u64,
    http_2xx: u64,
    http_429: u64,
    http_5xx: u64,
    qsb2_samples: u64,
    rsk1_samples: u64,
    used_l2: u64,
    decision_counts: [u64; 5],
}

#[derive(Clone, Debug)]
struct ThroughputBatchDone {
    latency_us: u64,
    ok: u64,
    err: u64,
    timeout: u64,
    http_2xx: u64,
    http_429: u64,
    http_5xx: u64,
    qsb2_samples: u64,
    rsk1_samples: u64,
    used_l2: u64,
    decision_counts: [u64; 5],
}

#[derive(Default)]
struct HttpBatchWorkerStats {
    attempted: AtomicU64,
    dropped_conn_queue_full: AtomicU64,
    dropped_after_attempt: AtomicU64,
    ok: AtomicU64,
    err: AtomicU64,
    timeout: AtomicU64,
    http_2xx: AtomicU64,
    http_429: AtomicU64,
    http_5xx: AtomicU64,
    qsb2_samples: AtomicU64,
    rsk1_samples: AtomicU64,
    used_l2: AtomicU64,
    decision_counts: [AtomicU64; 5],
    batch_latency_us: Mutex<Vec<u64>>,
    done: AtomicBool,
}

#[derive(Default)]
struct HttpBatchWorkerLocal {
    attempted: u64,
    dropped_conn_queue_full: u64,
    dropped_after_attempt: u64,
    ok: u64,
    err: u64,
    timeout: u64,
    http_2xx: u64,
    http_429: u64,
    http_5xx: u64,
    qsb2_samples: u64,
    rsk1_samples: u64,
    used_l2: u64,
    decision_counts: [u64; 5],
    batch_latency_us: Vec<u64>,
}

impl HttpBatchWorkerLocal {
    fn record_dropped_after_attempt(&mut self, dropped_after_attempt: u64) {
        self.dropped_after_attempt += dropped_after_attempt;
    }

    fn record_batch(&mut self, batch: ThroughputBatchDone) {
        self.ok += batch.ok;
        self.err += batch.err;
        self.timeout += batch.timeout;
        self.http_2xx += batch.http_2xx;
        self.http_429 += batch.http_429;
        self.http_5xx += batch.http_5xx;
        self.qsb2_samples += batch.qsb2_samples;
        self.rsk1_samples += batch.rsk1_samples;
        self.used_l2 += batch.used_l2;
        for (dst, src) in self
            .decision_counts
            .iter_mut()
            .zip(batch.decision_counts.into_iter())
        {
            *dst += src;
        }
        self.batch_latency_us.push(batch.latency_us.max(1));
    }

    fn flush_into(&mut self, shared: &HttpBatchWorkerStats) {
        shared
            .attempted
            .fetch_add(self.attempted, Ordering::Relaxed);
        shared
            .dropped_conn_queue_full
            .fetch_add(self.dropped_conn_queue_full, Ordering::Relaxed);
        shared
            .dropped_after_attempt
            .fetch_add(self.dropped_after_attempt, Ordering::Relaxed);
        shared.ok.fetch_add(self.ok, Ordering::Relaxed);
        shared.err.fetch_add(self.err, Ordering::Relaxed);
        shared.timeout.fetch_add(self.timeout, Ordering::Relaxed);
        shared.http_2xx.fetch_add(self.http_2xx, Ordering::Relaxed);
        shared.http_429.fetch_add(self.http_429, Ordering::Relaxed);
        shared.http_5xx.fetch_add(self.http_5xx, Ordering::Relaxed);
        shared
            .qsb2_samples
            .fetch_add(self.qsb2_samples, Ordering::Relaxed);
        shared
            .rsk1_samples
            .fetch_add(self.rsk1_samples, Ordering::Relaxed);
        shared.used_l2.fetch_add(self.used_l2, Ordering::Relaxed);
        for (dst, src) in shared
            .decision_counts
            .iter()
            .zip(self.decision_counts.iter())
        {
            dst.fetch_add(*src, Ordering::Relaxed);
        }
        if !self.batch_latency_us.is_empty() {
            let mut lat = shared.batch_latency_us.lock().expect("batch latency mutex");
            lat.extend(self.batch_latency_us.drain(..));
        }
        *self = Self::default();
    }
}

#[derive(Debug)]
enum AggEvent {
    AttemptBatch {
        attempted: u64,
        dropped_conn_queue_full: u64,
    },
    CompletionBatch {
        items: Vec<CompletedSample>,
    },
    ThroughputBatch {
        batch: ThroughputBatchDone,
    },
    ThroughputBatches {
        items: Vec<ThroughputBatchDone>,
    },
    WorkerDone,
}

#[derive(Clone)]
struct PayloadCorpus {
    mmap: Arc<Mmap>,
    row_bytes: usize,
    rows: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct RouteMetaEntry {
    row_idx: u32,
    transaction_id: u64,
    fold_id: i32,
    seg_prod_amtbin: u32,
    l2_tau_used: f32,
}

#[derive(Clone)]
struct RouteMetaCorpus {
    rows: Arc<Vec<RouteMetaEntry>>,
}

impl PayloadCorpus {
    fn load(path: &str, dense_dim: usize) -> Result<Self> {
        let row_bytes = dense_dim
            .checked_mul(4)
            .ok_or_else(|| anyhow!("dense_dim too large"))?;
        let file = File::open(path).with_context(|| format!("open dense file: {path}"))?;
        let mmap = unsafe { Mmap::map(&file).with_context(|| format!("mmap dense file: {path}"))? };
        if mmap.len() < row_bytes {
            bail!(
                "dense file too small: len={} row_bytes={}",
                mmap.len(),
                row_bytes
            );
        }
        if mmap.len() % row_bytes != 0 {
            bail!(
                "dense file size not multiple of row_bytes: len={} row_bytes={}",
                mmap.len(),
                row_bytes
            );
        }
        let rows = mmap.len() / row_bytes;
        Ok(Self {
            mmap: Arc::new(mmap),
            row_bytes,
            rows,
        })
    }

    #[inline]
    fn row_slice(&self, idx: usize) -> &[u8] {
        let i = idx % self.rows;
        let off = i * self.row_bytes;
        &self.mmap[off..off + self.row_bytes]
    }
}

impl RouteMetaCorpus {
    fn load(path: &str, expected_rows: usize) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read route meta: {path}"))?;
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| anyhow!("empty route_meta.tsv: {path}"))?;
        let cols: Vec<&str> = header.split('\t').collect();
        let idx_row = find_col(&cols, "row_idx")?;
        let idx_tx = find_col(&cols, "TransactionID")?;
        let idx_fold = find_col(&cols, "fold_id")?;
        let idx_seg = find_col(&cols, "seg_prod_amtbin")?;
        let idx_tau = find_col(&cols, "l2_tau_used")?;
        let mut rows = vec![RouteMetaEntry::default(); expected_rows];
        let mut seen = vec![false; expected_rows];
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() <= idx_seg {
                continue;
            }
            let row_idx: usize = parts[idx_row]
                .trim()
                .parse()
                .with_context(|| format!("bad row_idx in route_meta: {}", parts[idx_row]))?;
            if row_idx >= expected_rows {
                bail!(
                    "route_meta row_idx {} out of range {}",
                    row_idx,
                    expected_rows
                );
            }
            rows[row_idx] = RouteMetaEntry {
                row_idx: row_idx as u32,
                transaction_id: parts[idx_tx].trim().parse().with_context(|| {
                    format!("bad TransactionID in route_meta: {}", parts[idx_tx])
                })?,
                fold_id: parts[idx_fold]
                    .trim()
                    .parse()
                    .with_context(|| format!("bad fold_id in route_meta: {}", parts[idx_fold]))?,
                seg_prod_amtbin: parts[idx_seg].trim().parse().with_context(|| {
                    format!("bad seg_prod_amtbin in route_meta: {}", parts[idx_seg])
                })?,
                l2_tau_used: parts[idx_tau].trim().parse().with_context(|| {
                    format!("bad l2_tau_used in route_meta: {}", parts[idx_tau])
                })?,
            };
            seen[row_idx] = true;
        }
        if let Some((idx, _)) = seen.iter().enumerate().find(|(_, ok)| !**ok) {
            bail!("route_meta missing row_idx {}", idx);
        }
        Ok(Self {
            rows: Arc::new(rows),
        })
    }

    #[inline]
    fn row(&self, idx: usize) -> RouteMetaEntry {
        self.rows[idx % self.rows.len()]
    }
}

impl WorkloadSource {
    fn payload_rows(&self) -> usize {
        match self {
            WorkloadSource::Corpus { payload, .. } => payload.rows,
            WorkloadSource::Ceiling { .. } => 1,
        }
    }

    fn dense_dim(&self, fallback: usize) -> usize {
        match self {
            WorkloadSource::Corpus { payload, .. } => payload.row_bytes / 4,
            WorkloadSource::Ceiling { .. } => fallback,
        }
    }

    fn route_meta_enabled(&self) -> bool {
        matches!(
            self,
            WorkloadSource::Corpus {
                route_meta: Some(_),
                ..
            }
        )
    }

    fn build_body(&self, seq: u64, worker_id: usize) -> Bytes {
        match self {
            WorkloadSource::Corpus {
                payload,
                route_meta,
            } => {
                let row_idx =
                    (((seq as usize).wrapping_mul(1_315_423_911)) ^ worker_id) % payload.rows;
                build_payload_bytes(payload, route_meta.as_ref(), row_idx)
            }
            WorkloadSource::Ceiling { body } => body.as_ref().clone(),
        }
    }
}

fn load_workload_source(args: &Args) -> Result<WorkloadSource> {
    match args.workload {
        WorkloadMode::Corpus => {
            let payload = PayloadCorpus::load(&args.dense_file, args.dense_dim)?;
            let route_meta = if let Some(path) = args.route_meta_tsv.as_ref() {
                Some(RouteMetaCorpus::load(path, payload.rows)?)
            } else {
                None
            };
            Ok(WorkloadSource::Corpus {
                payload,
                route_meta,
            })
        }
        WorkloadMode::Ceiling => {
            let path = args
                .payload_file
                .as_ref()
                .context("--payload-file is required when --workload ceiling")?;
            let body = std::fs::read(path).with_context(|| format!("read payload file: {path}"))?;
            Ok(WorkloadSource::Ceiling {
                body: Arc::new(Bytes::from(body)),
            })
        }
    }
}

#[derive(Clone)]
struct RequestTemplate {
    prefix: Arc<Vec<u8>>,
}

impl RequestTemplate {
    fn new(target: &Target, content_len: usize) -> Self {
        let path_and_query = target
            .path_and_query
            .as_ref()
            .expect("http request template requires path");
        let host_header = target
            .host_header
            .as_ref()
            .expect("http request template requires host header");
        let mut req = Vec::with_capacity(256);
        req.extend_from_slice(b"POST ");
        req.extend_from_slice(path_and_query.as_bytes());
        req.extend_from_slice(b" HTTP/1.1\r\nHost: ");
        req.extend_from_slice(host_header.as_bytes());
        req.extend_from_slice(
            b"\r\nContent-Type: application/octet-stream\r\nConnection: keep-alive\r\nContent-Length: ",
        );
        req.extend_from_slice(content_len.to_string().as_bytes());
        req.extend_from_slice(b"\r\n\r\n");
        Self {
            prefix: Arc::new(req),
        }
    }
}

#[derive(Clone, Debug)]
struct Target {
    transport: TransportKind,
    host_header: Option<String>,
    path_and_query: Option<String>,
    addr: SocketAddr,
}

fn parse_target(url: &str) -> Result<Target> {
    let uri: Uri = url
        .parse()
        .with_context(|| format!("invalid --url: {url}"))?;
    let scheme = uri.scheme_str().unwrap_or("http");
    let transport = match scheme {
        "http" => TransportKind::Http,
        "h2c" => TransportKind::H2c,
        _ => bail!("bench3 supports only http:// and h2c:// URLs, got scheme={scheme}"),
    };
    let host = uri.host().ok_or_else(|| anyhow!("url missing host"))?;
    let port = match transport {
        TransportKind::Http => uri.port_u16().unwrap_or(80),
        TransportKind::H2c => uri.port_u16().unwrap_or(8080),
    };
    let path_and_query = match transport {
        TransportKind::Http | TransportKind::H2c => Some(
            uri.path_and_query()
                .map(|v| v.as_str().to_string())
                .unwrap_or_else(|| "/".to_string()),
        ),
    };
    let host_header = Some(if port == 80 {
        host.to_string()
    } else {
        format!("{host}:{port}")
    });
    let addr = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("resolve {host}:{port}"))?
        .next()
        .ok_or_else(|| anyhow!("resolve {host}:{port}: no addresses"))?;
    Ok(Target {
        transport,
        host_header,
        path_and_query,
        addr,
    })
}

#[derive(Default)]
struct ResponseParser {
    buf: Vec<u8>,
    header_len: usize,
    content_len: usize,
    status_code: u16,
    timings_header: Option<StageTimingsUs>,
    headers_parsed: bool,
}

impl ResponseParser {
    fn with_capacity() -> Self {
        Self {
            buf: Vec::with_capacity(4096),
            ..Default::default()
        }
    }

    fn reset(&mut self) {
        self.buf.clear();
        self.header_len = 0;
        self.content_len = 0;
        self.status_code = 0;
        self.timings_header = None;
        self.headers_parsed = false;
    }

    fn read_from(
        &mut self,
        stream: &mut TcpStream,
    ) -> io::Result<Option<(HttpResponseMeta, BodyDecoded)>> {
        let mut tmp = [0u8; 4096];
        loop {
            match stream.read(&mut tmp) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "server closed connection",
                    ))
                }
                Ok(n) => {
                    self.buf.extend_from_slice(&tmp[..n]);
                    if self.buf.len() > MAX_RESPONSE_BYTES {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "response too large",
                        ));
                    }
                    if let Some(done) = self.try_finish()? {
                        return Ok(Some(done));
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(None),
                Err(e) => return Err(e),
            }
        }
    }

    fn read_batch_ack_from(
        &mut self,
        stream: &mut TcpStream,
    ) -> io::Result<Option<(HttpResponseMeta, BatchAck)>> {
        let mut tmp = [0u8; 4096];
        loop {
            match stream.read(&mut tmp) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "server closed connection",
                    ))
                }
                Ok(n) => {
                    self.buf.extend_from_slice(&tmp[..n]);
                    if self.buf.len() > MAX_RESPONSE_BYTES {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "response too large",
                        ));
                    }
                    if let Some(done) = self.try_finish_batch_ack()? {
                        return Ok(Some(done));
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(None),
                Err(e) => return Err(e),
            }
        }
    }

    fn parse_headers_generic(&mut self, header_bytes: &[u8]) -> io::Result<()> {
        let header_str = str::from_utf8(header_bytes).map_err(invalid_data)?;
        let mut lines = header_str.split("\r\n");
        let status_line = lines
            .next()
            .ok_or_else(|| invalid_data("missing status line"))?;
        self.status_code = parse_status_code(status_line)?;
        self.content_len = 0;
        self.timings_header = None;
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                let value = value.trim();
                if name.eq_ignore_ascii_case("content-length") {
                    self.content_len = value
                        .parse::<usize>()
                        .map_err(|e| invalid_data(e.to_string()))?;
                } else if name.eq_ignore_ascii_case("x-risk-timings-us") {
                    self.timings_header = decode_timings_csv(value);
                }
            }
        }
        self.headers_parsed = true;
        Ok(())
    }

    fn try_finish(&mut self) -> io::Result<Option<(HttpResponseMeta, BodyDecoded)>> {
        if !self.headers_parsed {
            if let Some(hdr_end) = find_subsequence(&self.buf, b"\r\n\r\n") {
                self.header_len = hdr_end + 4;
                let header_bytes = self.buf[..hdr_end].to_vec();
                self.parse_headers_generic(&header_bytes)?;
            } else {
                return Ok(None);
            }
        }

        if self.headers_parsed && self.buf.len() >= self.header_len + self.content_len {
            let meta = HttpResponseMeta {
                status_code: self.status_code,
                timings_header: self.timings_header,
            };
            let body = &self.buf[self.header_len..self.header_len + self.content_len];
            let decoded = decode_body(body)?;
            return Ok(Some((meta, decoded)));
        }
        Ok(None)
    }

    fn try_finish_batch_ack(&mut self) -> io::Result<Option<(HttpResponseMeta, BatchAck)>> {
        if !self.headers_parsed {
            if let Some(hdr_end) = find_subsequence(&self.buf, b"\r\n\r\n") {
                self.header_len = hdr_end + 4;
                let header_bytes = &self.buf[..hdr_end];
                if let Some((status_code, content_len)) =
                    parse_batch_ack_headers_fast(header_bytes)?
                {
                    self.status_code = status_code;
                    self.content_len = content_len;
                    self.timings_header = None;
                    self.headers_parsed = true;
                } else {
                    let header_bytes = header_bytes.to_vec();
                    self.parse_headers_generic(&header_bytes)?;
                }
            } else {
                return Ok(None);
            }
        }

        if self.headers_parsed && self.buf.len() >= self.header_len + self.content_len {
            let meta = HttpResponseMeta {
                status_code: self.status_code,
                timings_header: None,
            };
            let body = &self.buf[self.header_len..self.header_len + self.content_len];
            if let Some(ack) = decode_batch_ack_body(body) {
                return Ok(Some((meta, ack)));
            }
            match decode_body(body)? {
                BodyDecoded::BatchAck(ack) => Ok(Some((meta, ack))),
                _ => Err(invalid_data("expected batch ack response")),
            }
        } else {
            Ok(None)
        }
    }
}

#[derive(Default)]
struct WriteState {
    prefix_off: usize,
    route_header_off: usize,
    body_off: usize,
}

struct Conn {
    token: Token,
    stream: TcpStream,
    pending: VecDeque<QueuedReq>,
    active: Option<ActiveReq>,
    write_state: WriteState,
    reading: bool,
    parser: ResponseParser,
}

impl Conn {
    fn load(&self) -> usize {
        self.pending.len() + usize::from(self.active.is_some())
    }
}

struct BatchWriteState {
    prefix_off: usize,
    body_off: usize,
}

impl Default for BatchWriteState {
    fn default() -> Self {
        Self {
            prefix_off: 0,
            body_off: 0,
        }
    }
}

struct BatchActiveReq {
    start_row_idx: usize,
    t_sched: Instant,
    deadline: Instant,
    record: bool,
    rows: u32,
    batch_records: usize,
    batch_header: [u8; 16],
}

struct BatchConn {
    token: Token,
    stream: TcpStream,
    pending: VecDeque<QueuedReq>,
    active: Option<BatchActiveReq>,
    write_state: BatchWriteState,
    reading: bool,
    parser: ResponseParser,
}

impl BatchConn {
    fn load(&self) -> usize {
        self.pending.len() + usize::from(self.active.is_some())
    }
}

fn discard_batch_rows(local_pending: &mut VecDeque<QueuedReq>, conns: &mut [BatchConn]) -> u64 {
    let mut dropped_rows = 0u64;
    while let Some(req) = local_pending.pop_front() {
        if req.record {
            dropped_rows += req.rows.max(1) as u64;
        }
    }
    for conn in conns.iter_mut() {
        while let Some(req) = conn.pending.pop_front() {
            if req.record {
                dropped_rows += req.rows.max(1) as u64;
            }
        }
    }
    dropped_rows
}

struct StatsAgg {
    attempted: u64,
    ok: u64,
    err: u64,
    timeout: u64,
    dropped: u64,
    drop_inflight_cap: u64,
    drop_conn_queue_full: u64,
    dropped_after_attempt: u64,
    http_2xx: u64,
    http_429: u64,
    http_5xx: u64,
    qsb2_samples: u64,
    rsk1_samples: u64,
    used_l2: u64,
    decision_counts: [u64; 5],
    queue_delay_us: Histogram<u64>,
    send_delay_us: Histogram<u64>,
    server_rtt_us: Histogram<u64>,
    e2e_us: Histogram<u64>,
    stage_parse: Histogram<u64>,
    stage_feature: Histogram<u64>,
    stage_router: Histogram<u64>,
    stage_l1: Histogram<u64>,
    stage_l2: Histogram<u64>,
    stage_serialize: Histogram<u64>,
    batch_e2e_us: Histogram<u64>,
}

impl StatsAgg {
    fn new() -> Result<Self> {
        Ok(Self {
            attempted: 0,
            ok: 0,
            err: 0,
            timeout: 0,
            dropped: 0,
            drop_inflight_cap: 0,
            drop_conn_queue_full: 0,
            dropped_after_attempt: 0,
            http_2xx: 0,
            http_429: 0,
            http_5xx: 0,
            qsb2_samples: 0,
            rsk1_samples: 0,
            used_l2: 0,
            decision_counts: [0; 5],
            queue_delay_us: new_hist()?,
            send_delay_us: new_hist()?,
            server_rtt_us: new_hist()?,
            e2e_us: new_hist()?,
            stage_parse: new_hist()?,
            stage_feature: new_hist()?,
            stage_router: new_hist()?,
            stage_l1: new_hist()?,
            stage_l2: new_hist()?,
            stage_serialize: new_hist()?,
            batch_e2e_us: new_hist()?,
        })
    }

    fn record_attempts(&mut self, attempted: u64, dropped_conn_queue_full: u64) {
        self.attempted += attempted;
        self.dropped += dropped_conn_queue_full;
        self.drop_conn_queue_full += dropped_conn_queue_full;
    }

    fn record_dropped_after_attempt(&mut self, dropped_after_attempt: u64) {
        self.dropped += dropped_after_attempt;
        self.dropped_after_attempt += dropped_after_attempt;
    }

    fn record_completion(&mut self, item: &CompletedSample) {
        if item.timeout {
            self.timeout += 1;
        } else if (200..300).contains(&item.status_code) {
            self.ok += 1;
            self.http_2xx += 1;
        } else {
            self.err += 1;
            if item.status_code == 429 {
                self.http_429 += 1;
            } else if item.status_code >= 500 {
                self.http_5xx += 1;
            }
        }

        if item.qsb2 {
            self.qsb2_samples += 1;
        }
        if item.rsk1 {
            self.rsk1_samples += 1;
        }
        if item.used_l2 {
            self.used_l2 += 1;
        }
        self.decision_counts[decision_bucket(item.decision)] += 1;

        let _ = self.queue_delay_us.record(item.queue_delay_us.max(1));
        let _ = self.send_delay_us.record(item.send_delay_us.max(1));
        let _ = self.server_rtt_us.record(item.server_rtt_us.max(1));
        let _ = self.e2e_us.record(item.e2e_us.max(1));

        if item.stage.parse > 0
            || item.stage.feature > 0
            || item.stage.router > 0
            || item.stage.l1 > 0
            || item.stage.l2 > 0
            || item.stage.serialize > 0
        {
            let _ = self.stage_parse.record((item.stage.parse as u64).max(1));
            let _ = self
                .stage_feature
                .record((item.stage.feature as u64).max(1));
            let _ = self.stage_router.record((item.stage.router as u64).max(1));
            let _ = self.stage_l1.record((item.stage.l1 as u64).max(1));
            let _ = self.stage_l2.record((item.stage.l2 as u64).max(1));
            let _ = self
                .stage_serialize
                .record((item.stage.serialize as u64).max(1));
        }
    }

    fn record_throughput_batch(&mut self, batch: &ThroughputBatchDone) {
        self.ok += batch.ok;
        self.err += batch.err;
        self.timeout += batch.timeout;
        self.http_2xx += batch.http_2xx;
        self.http_429 += batch.http_429;
        self.http_5xx += batch.http_5xx;
        self.qsb2_samples += batch.qsb2_samples;
        self.rsk1_samples += batch.rsk1_samples;
        self.used_l2 += batch.used_l2;
        for (dst, src) in self
            .decision_counts
            .iter_mut()
            .zip(batch.decision_counts.iter())
        {
            *dst += *src;
        }
        let _ = self.batch_e2e_us.record(batch.latency_us.max(1));
    }

    fn reset_window(&mut self) -> Result<()> {
        *self = Self::new()?;
        Ok(())
    }
}

#[derive(Serialize)]
struct SummaryJson {
    config: SummaryConfig,
    counts: SummaryCounts,
    latency_us: SummaryLatency,
    batch_latency_us: LatQuantiles,
    stage_p99_us: StageP99,
    verdict: Verdict,
    client_limited_reasons: Vec<String>,
    server_limited_reasons: Vec<String>,
}

#[derive(Serialize)]
struct SummaryConfig {
    url: String,
    rps: u64,
    duration_s: u64,
    warmup_s: u64,
    workers: usize,
    worker_cpus: Option<Vec<usize>>,
    pacer_cpu: Option<usize>,
    conns_per_worker: usize,
    max_inflight_per_conn: usize,
    raw_batch_size: usize,
    stats_sample_rate: usize,
    payload_rows: usize,
    dense_dim: usize,
    route_meta: bool,
    protocol: ProtocolMode,
    bench_mode: BenchMode,
    batch_mode: BatchMode,
    batch_records: usize,
    workload: WorkloadMode,
    transport: &'static str,
    raw_version: Option<String>,
}

#[derive(Serialize)]
struct SummaryCounts {
    offered: u64,
    attempted: u64,
    ok: u64,
    err: u64,
    timeout: u64,
    dropped: u64,
    drop_inflight_cap: u64,
    drop_conn_queue_full: u64,
    dropped_after_attempt: u64,
    http_2xx: u64,
    http_429: u64,
    http_5xx: u64,
    qsb2_samples: u64,
    rsk1_samples: u64,
    used_l2: u64,
    decision_allow: u64,
    decision_deny: u64,
    decision_manual_review: u64,
    decision_degrade_allow: u64,
    decision_unknown: u64,
    attempted_rps: f64,
    ok_rps: f64,
    target_rps: u64,
    offered_but_not_sent: u64,
    offered_but_not_sent_rps: f64,
    offered_but_not_sent_pct: f64,
    under_target_rps: f64,
    under_target_pct: f64,
    attempted_batch_rps: f64,
    ok_batch_rps: f64,
}

#[derive(Serialize)]
struct LatQuantiles {
    p50: u64,
    p95: u64,
    p99: u64,
}

#[derive(Serialize)]
struct SummaryLatency {
    queue_delay: LatQuantiles,
    send_delay: LatQuantiles,
    server_rtt: LatQuantiles,
    e2e: LatQuantiles,
}

#[derive(Serialize)]
struct StageP99 {
    parse: u64,
    feature: u64,
    router: u64,
    l1: u64,
    l2: u64,
    serialize: u64,
}

fn new_hist() -> Result<Histogram<u64>> {
    Histogram::new_with_bounds(1, 60_000_000, 3).map_err(Into::into)
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn invalid_data<E: ToString>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, e.to_string())
}

fn find_col(cols: &[&str], name: &str) -> Result<usize> {
    cols.iter()
        .position(|x| *x == name)
        .ok_or_else(|| anyhow!("missing column '{name}'"))
}

fn parse_status_code(line: &str) -> io::Result<u16> {
    let mut parts = line.split_whitespace();
    let proto = parts
        .next()
        .ok_or_else(|| invalid_data("missing HTTP version"))?;
    if !proto.starts_with("HTTP/1.") {
        return Err(invalid_data(format!("unsupported status line: {line}")));
    }
    let code = parts
        .next()
        .ok_or_else(|| invalid_data("missing status code"))?
        .parse::<u16>()
        .map_err(invalid_data)?;
    Ok(code)
}

fn decode_timings_csv(s: &str) -> Option<StageTimingsUs> {
    let mut it = s.split(',');
    Some(StageTimingsUs {
        parse: it.next()?.parse().ok()?,
        feature: it.next()?.parse().ok()?,
        router: it.next()?.parse().ok()?,
        l1: it.next()?.parse().ok()?,
        l2: it.next()?.parse().ok()?,
        serialize: it.next()?.parse().ok()?,
    })
}

fn trim_ascii_ws(mut bytes: &[u8]) -> &[u8] {
    while let Some((&b, rest)) = bytes.split_first() {
        if matches!(b, b' ' | b'\t' | b'\r') {
            bytes = rest;
        } else {
            break;
        }
    }
    while let Some((&b, rest)) = bytes.split_last() {
        if matches!(b, b' ' | b'\t' | b'\r') {
            bytes = rest;
        } else {
            break;
        }
    }
    bytes
}

fn starts_with_ascii_case_insensitive(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.len() >= needle.len()
        && haystack[..needle.len()]
            .iter()
            .zip(needle.iter())
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
}

fn parse_batch_ack_headers_fast(header_bytes: &[u8]) -> io::Result<Option<(u16, usize)>> {
    let mut lines = header_bytes.split(|&b| b == b'\n');
    let status_line = trim_ascii_ws(
        lines
            .next()
            .ok_or_else(|| invalid_data("missing status line"))?,
    );
    if status_line.len() < 12 || !status_line.starts_with(b"HTTP/1.") || status_line[8] != b' ' {
        return Ok(None);
    }
    let code = std::str::from_utf8(&status_line[9..12])
        .map_err(invalid_data)?
        .parse::<u16>()
        .map_err(invalid_data)?;
    let mut content_len = None;
    for line in lines {
        let line = trim_ascii_ws(line);
        if starts_with_ascii_case_insensitive(line, b"content-length:") {
            let value = trim_ascii_ws(&line["content-length:".len()..]);
            content_len = Some(
                std::str::from_utf8(value)
                    .map_err(invalid_data)?
                    .parse::<usize>()
                    .map_err(invalid_data)?,
            );
            break;
        }
    }
    Ok(content_len.map(|len| (code, len)))
}

fn decode_batch_ack_body(buf: &[u8]) -> Option<BatchAck> {
    if buf.len() != 40 || &buf[0..4] != b"RBA1" {
        return None;
    }
    let version = u16::from_le_bytes([buf[4], buf[5]]);
    if version != 1 {
        return None;
    }
    let rd = |off: usize| -> u32 {
        u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    };
    Some(BatchAck {
        record_count: rd(8),
        ok_count: rd(12),
        used_l2_count: rd(16),
        decision_counts: [rd(20), rd(24), rd(28), rd(32), rd(36)],
    })
}

fn decode_body(buf: &[u8]) -> io::Result<BodyDecoded> {
    if let Some(ack) = decode_batch_ack_body(buf) {
        return Ok(BodyDecoded::BatchAck(ack));
    }
    if buf.len() == 24 && &buf[0..4] == b"QSB2" {
        return Ok(BodyDecoded::Qsb2(Qsb2 {
            decision: buf[6],
            flags: buf[7],
            timings: StageTimingsUs::default(),
        }));
    }
    if buf.len() == 48 && &buf[0..4] == b"QSB2" {
        let rd = |off: usize| -> u32 {
            u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
        };
        return Ok(BodyDecoded::Qsb2(Qsb2 {
            decision: buf[6],
            flags: buf[7],
            timings: StageTimingsUs {
                parse: rd(24),
                feature: rd(28),
                router: rd(32),
                l1: rd(36),
                l2: rd(40),
                serialize: rd(44),
            },
        }));
    }
    if buf.len() == 48 && &buf[0..4] == b"RSK1" {
        let rd = |off: usize| -> u32 {
            u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
        };
        return Ok(BodyDecoded::Rsk1(Rsk1 {
            decision: buf[20],
            flags: u16::from_le_bytes([buf[6], buf[7]]),
            timings: StageTimingsUs {
                parse: rd(24),
                feature: rd(28),
                router: rd(32),
                l1: rd(36),
                l2: rd(40),
                serialize: rd(44),
            },
        }));
    }
    Err(invalid_data("unsupported response body"))
}

fn decision_bucket(decision: u8) -> usize {
    match decision {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        _ => 4,
    }
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

fn parse_cpu_list(spec: &Option<String>) -> Result<Option<Vec<usize>>> {
    let Some(spec) = spec.as_ref() else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, b)) = part.split_once('-') {
            let start: usize = a
                .trim()
                .parse()
                .with_context(|| format!("bad cpu range: {part}"))?;
            let end: usize = b
                .trim()
                .parse()
                .with_context(|| format!("bad cpu range: {part}"))?;
            if end < start {
                bail!("bad cpu range: {part}");
            }
            out.extend(start..=end);
        } else {
            out.push(part.parse().with_context(|| format!("bad cpu: {part}"))?);
        }
    }
    if out.is_empty() {
        bail!("empty cpu list");
    }
    Ok(Some(out))
}

#[cfg(target_os = "linux")]
fn pin_current_thread(cpu: Option<usize>) -> Result<()> {
    let Some(cpu) = cpu else {
        return Ok(());
    };
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if rc != 0 {
            return Err(io::Error::last_os_error()).context("sched_setaffinity");
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_cpu: Option<usize>) -> Result<()> {
    Ok(())
}

fn connect_stream(addr: SocketAddr) -> Result<TcpStream> {
    let std_stream = std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(50))
        .with_context(|| format!("connect {addr}"))?;
    std_stream.set_nodelay(true).ok();
    std_stream
        .set_nonblocking(true)
        .context("set_nonblocking")?;
    Ok(TcpStream::from_std(std_stream))
}

fn register_conn(poll: &Poll, stream: &mut TcpStream, token: Token, writable: bool) -> Result<()> {
    let interest = if writable {
        Interest::READABLE | Interest::WRITABLE
    } else {
        Interest::READABLE
    };
    poll.registry()
        .register(stream, token, interest)
        .context("register conn")?;
    Ok(())
}

fn reregister_conn(
    poll: &Poll,
    stream: &mut TcpStream,
    token: Token,
    writable: bool,
) -> Result<()> {
    let interest = if writable {
        Interest::READABLE | Interest::WRITABLE
    } else {
        Interest::READABLE
    };
    poll.registry()
        .reregister(stream, token, interest)
        .context("reregister conn")?;
    Ok(())
}

fn build_completion(
    active: &ActiveReq,
    done_at: Instant,
    meta: HttpResponseMeta,
    body: BodyDecoded,
) -> CompletedSample {
    let t_send_done = active.t_send_done.unwrap_or(active.t_issue);
    let stage = match body {
        BodyDecoded::Qsb2(q) => {
            if q.timings.parse > 0
                || q.timings.feature > 0
                || q.timings.router > 0
                || q.timings.l1 > 0
                || q.timings.l2 > 0
                || q.timings.serialize > 0
            {
                q.timings
            } else {
                meta.timings_header.unwrap_or_default()
            }
        }
        BodyDecoded::Rsk1(r) => r.timings,
        BodyDecoded::BatchAck(_) => StageTimingsUs::default(),
    };
    let (decision, used_l2, qsb2, rsk1) = match body {
        BodyDecoded::Qsb2(q) => (q.decision, (q.flags & 1) != 0, true, false),
        BodyDecoded::Rsk1(r) => (r.decision, (r.flags & 1) != 0, false, true),
        BodyDecoded::BatchAck(_) => (255, false, false, false),
    };
    CompletedSample {
        status_code: meta.status_code,
        e2e_us: done_at.duration_since(active.t_sched).as_micros() as u64,
        queue_delay_us: active.t_issue.duration_since(active.t_sched).as_micros() as u64,
        send_delay_us: t_send_done.duration_since(active.t_issue).as_micros() as u64,
        server_rtt_us: done_at.duration_since(t_send_done).as_micros() as u64,
        timeout: false,
        used_l2,
        decision,
        qsb2,
        rsk1,
        stage,
    }
}

fn build_timeout(active: &ActiveReq, now: Instant) -> CompletedSample {
    let t_send_done = active.t_send_done.unwrap_or(active.t_issue);
    CompletedSample {
        status_code: 0,
        e2e_us: now.duration_since(active.t_sched).as_micros() as u64,
        queue_delay_us: active.t_issue.duration_since(active.t_sched).as_micros() as u64,
        send_delay_us: t_send_done.duration_since(active.t_issue).as_micros() as u64,
        server_rtt_us: now.duration_since(t_send_done).as_micros() as u64,
        timeout: true,
        used_l2: false,
        decision: 255,
        qsb2: false,
        rsk1: false,
        stage: StageTimingsUs::default(),
    }
}

fn throughput_response_from_decoded(status_code: u16, body: BodyDecoded) -> ThroughputResponse {
    match body {
        BodyDecoded::Qsb2(q) => {
            let mut decision_counts = [0u32; 5];
            decision_counts[decision_bucket(q.decision)] = 1;
            ThroughputResponse {
                status_code,
                timeout: false,
                rows: 1,
                used_l2_count: u32::from((q.flags & 1) != 0),
                decision_counts,
                qsb2_samples: 1,
                rsk1_samples: 0,
            }
        }
        BodyDecoded::Rsk1(r) => {
            let mut decision_counts = [0u32; 5];
            decision_counts[decision_bucket(r.decision)] = 1;
            ThroughputResponse {
                status_code,
                timeout: false,
                rows: 1,
                used_l2_count: u32::from((r.flags & 1) != 0),
                decision_counts,
                qsb2_samples: 0,
                rsk1_samples: 1,
            }
        }
        BodyDecoded::BatchAck(ack) => ThroughputResponse {
            status_code,
            timeout: false,
            rows: ack.record_count,
            used_l2_count: ack.used_l2_count,
            decision_counts: ack.decision_counts,
            qsb2_samples: 0,
            rsk1_samples: 0,
        },
    }
}

fn throughput_timeout_response(rows: u32) -> ThroughputResponse {
    ThroughputResponse {
        status_code: 0,
        timeout: true,
        rows,
        used_l2_count: 0,
        decision_counts: [0, 0, 0, 0, rows],
        qsb2_samples: 0,
        rsk1_samples: 0,
    }
}

fn update_throughput_batch(state: &mut ThroughputBatchState, resp: ThroughputResponse) {
    state.outstanding = state.outstanding.saturating_sub(1);
    if resp.timeout {
        state.timeout += resp.rows as u64;
    } else if (200..300).contains(&resp.status_code) {
        state.ok += resp.rows as u64;
        state.http_2xx += resp.rows as u64;
    } else {
        state.err += resp.rows as u64;
        if resp.status_code == 429 {
            state.http_429 += resp.rows as u64;
        } else if resp.status_code >= 500 {
            state.http_5xx += resp.rows as u64;
        }
    }
    state.qsb2_samples += resp.qsb2_samples as u64;
    state.rsk1_samples += resp.rsk1_samples as u64;
    state.used_l2 += resp.used_l2_count as u64;
    for (dst, src) in state
        .decision_counts
        .iter_mut()
        .zip(resp.decision_counts.iter())
    {
        *dst += *src as u64;
    }
}

fn maybe_finalize_throughput_batch(
    out: &mut Vec<ThroughputBatchDone>,
    batch_id: u64,
    now: Instant,
    batches: &mut HashMap<u64, ThroughputBatchState>,
) {
    let done = batches
        .get(&batch_id)
        .map(|batch| batch.sealed && batch.outstanding == 0)
        .unwrap_or(false);
    if !done {
        return;
    }
    if let Some(batch) = batches.remove(&batch_id) {
        out.push(ThroughputBatchDone {
            latency_us: now.duration_since(batch.issued_at).as_micros() as u64,
            ok: batch.ok,
            err: batch.err,
            timeout: batch.timeout,
            http_2xx: batch.http_2xx,
            http_429: batch.http_429,
            http_5xx: batch.http_5xx,
            qsb2_samples: batch.qsb2_samples,
            rsk1_samples: batch.rsk1_samples,
            used_l2: batch.used_l2,
            decision_counts: batch.decision_counts,
        });
    }
}

fn maybe_flush_throughput_batches(tx: &Sender<AggEvent>, items: &mut Vec<ThroughputBatchDone>) {
    if items.is_empty() {
        return;
    }
    if items.len() == 1 {
        let batch = items.pop().expect("one throughput batch");
        let _ = tx.send(AggEvent::ThroughputBatch { batch });
        return;
    }
    let flushed = std::mem::take(items);
    let _ = tx.send(AggEvent::ThroughputBatches { items: flushed });
}

fn take_quota(quota: &AtomicU64, max_take: usize) -> usize {
    if max_take == 0 {
        return 0;
    }
    let max_take = max_take as u64;
    let mut cur = quota.load(Ordering::Acquire);
    loop {
        if cur == 0 {
            return 0;
        }
        let take = cur.min(max_take);
        match quota.compare_exchange_weak(cur, cur - take, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return take as usize,
            Err(next) => cur = next,
        }
    }
}

fn build_payload_bytes(
    corpus: &PayloadCorpus,
    route_meta: Option<&RouteMetaCorpus>,
    row_idx: usize,
) -> Bytes {
    if let Some(route_meta) = route_meta {
        let mut out = Vec::with_capacity(40 + corpus.row_bytes);
        out.extend_from_slice(&encode_rvec_v3_route_header(
            route_meta.row(row_idx),
            corpus.row_bytes / 4,
        ));
        out.extend_from_slice(corpus.row_slice(row_idx));
        Bytes::from(out)
    } else {
        Bytes::copy_from_slice(corpus.row_slice(row_idx))
    }
}

fn record_len(corpus: &PayloadCorpus, route_meta: Option<&RouteMetaCorpus>) -> usize {
    corpus.row_bytes + route_meta.map(|_| 40).unwrap_or(0)
}

fn encode_batch_header(record_count: usize, record_bytes: usize, has_route_meta: bool) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(b"RBH1");
    out[4..6].copy_from_slice(&1u16.to_le_bytes());
    let flags = if has_route_meta { 1u16 } else { 0u16 };
    out[6..8].copy_from_slice(&flags.to_le_bytes());
    out[8..12].copy_from_slice(&(record_count as u32).to_le_bytes());
    out[12..16].copy_from_slice(&(record_bytes as u32).to_le_bytes());
    out
}

async fn connect_h2_stream(addr: SocketAddr) -> Result<tokio::net::TcpStream> {
    let stream = tokio::time::timeout(
        Duration::from_millis(200),
        tokio::net::TcpStream::connect(addr),
    )
    .await
    .with_context(|| format!("connect timeout {addr}"))?
    .with_context(|| format!("connect {addr}"))?;
    stream.set_nodelay(true).ok();
    Ok(stream)
}

async fn open_h2_throughput_conn(addr: SocketAddr) -> Result<H2ThroughputConn> {
    let stream = connect_h2_stream(addr).await?;
    let (sender, conn) = h2::client::handshake(stream)
        .await
        .context("h2 throughput handshake")?;
    tokio::spawn(async move {
        let _ = conn.await;
    });
    Ok(H2ThroughputConn { sender })
}

async fn decode_h2_response(
    response: h2::client::ResponseFuture,
) -> Result<(HttpResponseMeta, BodyDecoded)> {
    let response = response.await.context("await h2 response")?;
    let status_code = response.status().as_u16();
    let timings_header = response
        .headers()
        .get("x-risk-timings-us")
        .and_then(|v| v.to_str().ok())
        .and_then(decode_timings_csv);
    let mut body_stream = response.into_body();
    let mut body = bytes::BytesMut::with_capacity(128);
    while let Some(chunk) = body_stream.data().await {
        let chunk = chunk.context("read h2 response body")?;
        body.extend_from_slice(&chunk);
        let _ = body_stream.flow_control().release_capacity(chunk.len());
    }
    let decoded = decode_body(&body).context("decode h2 response body")?;
    Ok((
        HttpResponseMeta {
            status_code,
            timings_header,
        },
        decoded,
    ))
}

async fn decode_h2_response_throughput(
    response: h2::client::ResponseFuture,
) -> Result<ThroughputResponse> {
    let response = response.await.context("await h2 response")?;
    let status_code = response.status().as_u16();
    let mut body_stream = response.into_body();
    let mut body = bytes::BytesMut::with_capacity(64);
    while let Some(chunk) = body_stream.data().await {
        let chunk = chunk.context("read h2 response body")?;
        body.extend_from_slice(&chunk);
        let _ = body_stream.flow_control().release_capacity(chunk.len());
    }
    let decoded = decode_body(&body).context("decode h2 throughput response body")?;
    Ok(throughput_response_from_decoded(status_code, decoded))
}

async fn drive_h2_connection(
    addr: SocketAddr,
    uri: Arc<str>,
    mut cmd_rx: tokio_mpsc::UnboundedReceiver<H2SendCmd>,
    resp_tx: tokio_mpsc::UnboundedSender<H2RespEvent>,
) -> Result<()> {
    let mut pending: Option<H2SendCmd> = None;
    loop {
        let cmd = match pending.take() {
            Some(cmd) => cmd,
            None => match cmd_rx.recv().await {
                Some(cmd) => cmd,
                None => return Ok(()),
            },
        };

        let stream = match connect_h2_stream(addr).await {
            Ok(stream) => stream,
            Err(_) => {
                let _ = resp_tx.send(H2RespEvent::Failed {
                    request_id: cmd.request_id,
                });
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue;
            }
        };

        let (sender, conn) = match h2::client::handshake(stream).await {
            Ok(parts) => parts,
            Err(_) => {
                let _ = resp_tx.send(H2RespEvent::Failed {
                    request_id: cmd.request_id,
                });
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue;
            }
        };
        tokio::spawn(async move {
            let _ = conn.await;
        });
        pending = Some(cmd);

        loop {
            let cmd = match pending.take() {
                Some(cmd) => cmd,
                None => match cmd_rx.recv().await {
                    Some(cmd) => cmd,
                    None => return Ok(()),
                },
            };

            let request = http::Request::builder()
                .method("POST")
                .uri(uri.as_ref())
                .header("content-type", "application/octet-stream")
                .body(())
                .context("build h2 request")?;
            let request_id = cmd.request_id;
            let mut ready = match sender.clone().ready().await {
                Ok(ready) => ready,
                Err(_) => {
                    let _ = resp_tx.send(H2RespEvent::Failed { request_id });
                    break;
                }
            };
            match ready.send_request(request, false) {
                Ok((response, mut send_stream)) => {
                    if send_stream.send_data(cmd.body, true).is_err() {
                        let _ = resp_tx.send(H2RespEvent::Failed { request_id });
                        break;
                    }
                    let resp_tx2 = resp_tx.clone();
                    tokio::spawn(async move {
                        let event = match decode_h2_response(response).await {
                            Ok((meta, body)) => H2RespEvent::Completed {
                                request_id,
                                meta,
                                body,
                            },
                            Err(_) => H2RespEvent::Failed { request_id },
                        };
                        let _ = resp_tx2.send(event);
                    });
                }
                Err(_) => {
                    let _ = resp_tx.send(H2RespEvent::Failed { request_id });
                    break;
                }
            }
        }
    }
}

async fn worker_loop_h2c_async(
    worker_id: usize,
    args: Args,
    target: Target,
    corpus: PayloadCorpus,
    route_meta: Option<RouteMetaCorpus>,
    queue: Arc<ArrayQueue<ReqToken>>,
    events_tx: Sender<AggEvent>,
    pacer_done: Arc<AtomicBool>,
) -> Result<()> {
    let timeout = Duration::from_millis(args.timeout_ms.max(1));
    let uri = Arc::<str>::from(format!(
        "http://{}{}",
        target
            .host_header
            .as_ref()
            .expect("h2c target requires host header"),
        target
            .path_and_query
            .as_ref()
            .expect("h2c target requires path"),
    ));
    let (resp_tx, mut resp_rx) = tokio_mpsc::unbounded_channel::<H2RespEvent>();
    let mut cmd_txs = Vec::with_capacity(args.conns_per_worker);
    for _ in 0..args.conns_per_worker.max(1) {
        let (cmd_tx, cmd_rx) = tokio_mpsc::unbounded_channel::<H2SendCmd>();
        cmd_txs.push(cmd_tx);
        let resp_tx2 = resp_tx.clone();
        let addr = target.addr;
        let uri2 = uri.clone();
        tokio::spawn(async move {
            let _ = drive_h2_connection(addr, uri2, cmd_rx, resp_tx2).await;
        });
    }
    drop(resp_tx);

    let mut inflight = HashMap::<u32, H2InflightReq>::with_capacity(
        args.conns_per_worker
            .saturating_mul(args.max_inflight_per_conn.max(1)),
    );
    let mut inflight_counts = vec![0usize; args.conns_per_worker.max(1)];
    let local_pending_cap = args
        .conns_per_worker
        .saturating_mul(args.max_inflight_per_conn.max(1))
        .saturating_mul(2)
        .max(1);
    let mut local_pending: VecDeque<QueuedReq> = VecDeque::with_capacity(local_pending_cap);
    let mut next_conn_rr = worker_id % args.conns_per_worker.max(1);
    let mut next_request_id = ((worker_id as u32) << 20).wrapping_add(1);
    let mut batch = Vec::with_capacity(FLUSH_BATCH);
    let mut last_flush = Instant::now();
    let mut drain_started: Option<Instant> = None;

    loop {
        while local_pending.len() < local_pending_cap {
            let Some(tok) = queue.pop() else {
                break;
            };
            local_pending.push_back(QueuedReq {
                row_idx: tok.row_idx,
                t_sched: tok.t_sched,
                t_issue: Instant::now(),
                record: tok.record,
                rows: tok.rows,
            });
        }

        while let Ok(event) = resp_rx.try_recv() {
            match event {
                H2RespEvent::Completed {
                    request_id,
                    meta,
                    body,
                } => {
                    if let Some(req) = inflight.remove(&request_id) {
                        inflight_counts[req.conn_idx] =
                            inflight_counts[req.conn_idx].saturating_sub(1);
                        if req.active.record {
                            batch.push(build_completion(&req.active, Instant::now(), meta, body));
                        }
                    }
                }
                H2RespEvent::Failed { request_id } => {
                    if let Some(req) = inflight.remove(&request_id) {
                        inflight_counts[req.conn_idx] =
                            inflight_counts[req.conn_idx].saturating_sub(1);
                        if req.active.record {
                            batch.push(build_timeout(&req.active, Instant::now()));
                        }
                    }
                }
            }
        }

        while let Some(req) = local_pending.pop_front() {
            let mut selected = None;
            for step in 0..cmd_txs.len() {
                let idx = (next_conn_rr + step) % cmd_txs.len();
                if inflight_counts[idx] < args.max_inflight_per_conn.max(1) {
                    selected = Some(idx);
                    next_conn_rr = (idx + 1) % cmd_txs.len();
                    break;
                }
            }
            let Some(conn_idx) = selected else {
                local_pending.push_front(req);
                break;
            };

            let now = Instant::now();
            let body = build_payload_bytes(&corpus, route_meta.as_ref(), req.row_idx);
            let active = ActiveReq {
                row_idx: req.row_idx,
                t_sched: req.t_sched,
                t_issue: req.t_issue,
                record: req.record,
                t_send_done: Some(now),
                deadline: now + timeout,
                route_header: [0u8; 40],
                route_header_len: 0,
            };
            let request_id = next_request_id;
            next_request_id = next_request_id.wrapping_add(1);
            if cmd_txs[conn_idx]
                .send(H2SendCmd { request_id, body })
                .is_err()
            {
                if active.record {
                    batch.push(build_timeout(&active, Instant::now()));
                }
                continue;
            }
            inflight.insert(request_id, H2InflightReq { active, conn_idx });
            inflight_counts[conn_idx] += 1;
        }

        let now = Instant::now();
        let expired: Vec<u32> = inflight
            .iter()
            .filter_map(|(request_id, req)| (now >= req.active.deadline).then_some(*request_id))
            .collect();
        for request_id in expired {
            if let Some(req) = inflight.remove(&request_id) {
                inflight_counts[req.conn_idx] = inflight_counts[req.conn_idx].saturating_sub(1);
                if req.active.record {
                    batch.push(build_timeout(&req.active, now));
                }
            }
        }

        if batch.len() >= FLUSH_BATCH || last_flush.elapsed() >= Duration::from_millis(2) {
            maybe_flush_batch(&events_tx, &mut batch);
            last_flush = Instant::now();
        }

        let all_idle = queue.is_empty() && local_pending.is_empty() && inflight.is_empty();
        if pacer_done.load(Ordering::Acquire) && all_idle {
            break;
        }

        if pacer_done.load(Ordering::Acquire) && queue.is_empty() && local_pending.is_empty() {
            let started = drain_started.get_or_insert_with(Instant::now);
            if started.elapsed() >= timeout + Duration::from_millis(100) {
                let now = Instant::now();
                for (_, req) in inflight.drain() {
                    inflight_counts[req.conn_idx] = inflight_counts[req.conn_idx].saturating_sub(1);
                    if req.active.record {
                        batch.push(build_timeout(&req.active, now));
                    }
                }
                break;
            }
        } else {
            drain_started = None;
        }

        if let Ok(Some(event)) =
            tokio::time::timeout(Duration::from_millis(1), resp_rx.recv()).await
        {
            match event {
                H2RespEvent::Completed {
                    request_id,
                    meta,
                    body,
                } => {
                    if let Some(req) = inflight.remove(&request_id) {
                        inflight_counts[req.conn_idx] =
                            inflight_counts[req.conn_idx].saturating_sub(1);
                        if req.active.record {
                            batch.push(build_completion(&req.active, Instant::now(), meta, body));
                        }
                    }
                }
                H2RespEvent::Failed { request_id } => {
                    if let Some(req) = inflight.remove(&request_id) {
                        inflight_counts[req.conn_idx] =
                            inflight_counts[req.conn_idx].saturating_sub(1);
                        if req.active.record {
                            batch.push(build_timeout(&req.active, Instant::now()));
                        }
                    }
                }
            }
        }
    }

    maybe_flush_batch(&events_tx, &mut batch);
    Ok(())
}

async fn worker_loop_h2c_throughput_async(
    worker_id: usize,
    worker_count: usize,
    args: Args,
    target: Target,
    workload: WorkloadSource,
    quota: Arc<ThroughputQuota>,
    events_tx: Sender<AggEvent>,
    pacer_done: Arc<AtomicBool>,
) -> Result<()> {
    let timeout = Duration::from_millis(args.timeout_ms.max(1));
    let uri = Arc::<str>::from(format!(
        "http://{}{}",
        target
            .host_header
            .as_ref()
            .expect("h2c target requires host header"),
        target
            .path_and_query
            .as_ref()
            .expect("h2c target requires path"),
    ));
    let mut conns = Vec::with_capacity(args.conns_per_worker.max(1));
    for _ in 0..args.conns_per_worker.max(1) {
        conns.push(open_h2_throughput_conn(target.addr).await?);
    }
    let mut pending_responses = FuturesUnordered::<H2ThroughputRespFuture>::new();

    let mut inflight = HashMap::<u32, ThroughputInflightReq>::with_capacity(
        args.conns_per_worker
            .saturating_mul(args.max_inflight_per_conn.max(1)),
    );
    let mut inflight_counts = vec![0usize; conns.len()];
    let local_pending_cap = args
        .conns_per_worker
        .saturating_mul(args.max_inflight_per_conn.max(1))
        .saturating_mul(8)
        .max(1);
    let mut local_warmup_tokens = 0usize;
    let mut local_record_tokens = 0usize;
    let mut next_conn_rr = worker_id % args.conns_per_worker.max(1);
    let mut next_request_id = ((worker_id as u32) << 20).wrapping_add(1);
    let mut next_batch_id = ((worker_id as u64) << 32).wrapping_add(1);
    let mut next_seq = worker_id as u64;
    let mut open_batch_id: Option<u64> = None;
    let mut open_batch_issued = 0usize;
    let batch_size = args.throughput_batch_size.max(1);
    let mut batches = HashMap::<u64, ThroughputBatchState>::new();
    let mut drain_started: Option<Instant> = None;
    let mut batch_flush = Vec::with_capacity(32);
    let mut last_batch_flush = Instant::now();

    loop {
        let local_total = local_warmup_tokens + local_record_tokens;
        if local_total < local_pending_cap {
            let refill_cap = local_pending_cap - local_total;
            local_record_tokens += take_quota(&quota.record, refill_cap);
            let local_total = local_warmup_tokens + local_record_tokens;
            if local_total < local_pending_cap {
                local_warmup_tokens += take_quota(&quota.warmup, local_pending_cap - local_total);
            }
        }

        while let Ok(Some((conn_idx, request_id, result))) =
            tokio::time::timeout(Duration::ZERO, pending_responses.next()).await
        {
            if let Some(req) = inflight.remove(&request_id) {
                inflight_counts[req.conn_idx] = inflight_counts[req.conn_idx].saturating_sub(1);
                if let Some(batch_id) = req.batch_id {
                    if let Some(batch) = batches.get_mut(&batch_id) {
                        match result {
                            Ok(response) => update_throughput_batch(batch, response),
                            Err(_) => {
                                update_throughput_batch(batch, throughput_timeout_response(1))
                            }
                        }
                    }
                    maybe_finalize_throughput_batch(
                        &mut batch_flush,
                        batch_id,
                        Instant::now(),
                        &mut batches,
                    );
                }
            }
            let _ = conn_idx;
        }

        while local_record_tokens > 0 || local_warmup_tokens > 0 {
            let mut selected = None;
            for step in 0..conns.len() {
                let idx = (next_conn_rr + step) % conns.len();
                if inflight_counts[idx] < args.max_inflight_per_conn.max(1) {
                    selected = Some(idx);
                    next_conn_rr = (idx + 1) % conns.len();
                    break;
                }
            }
            let Some(conn_idx) = selected else {
                break;
            };

            let now = Instant::now();
            let record = if local_record_tokens > 0 {
                local_record_tokens -= 1;
                true
            } else {
                local_warmup_tokens -= 1;
                false
            };
            let body = workload.build_body(next_seq, worker_id);
            next_seq = next_seq.wrapping_add(worker_count as u64);
            let request_id = next_request_id;
            next_request_id = next_request_id.wrapping_add(1);

            let batch_id = if record {
                let batch_id = match open_batch_id {
                    Some(id) => id,
                    None => {
                        let id = next_batch_id;
                        next_batch_id = next_batch_id.wrapping_add(1);
                        open_batch_id = Some(id);
                        open_batch_issued = 0;
                        batches.insert(
                            id,
                            ThroughputBatchState {
                                issued_at: now,
                                outstanding: 0,
                                sealed: false,
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
                            },
                        );
                        id
                    }
                };
                open_batch_issued += 1;
                if let Some(batch) = batches.get_mut(&batch_id) {
                    batch.outstanding += 1;
                }
                if open_batch_issued >= batch_size {
                    if let Some(batch) = batches.get_mut(&batch_id) {
                        batch.sealed = true;
                    }
                    open_batch_id = None;
                    open_batch_issued = 0;
                }
                Some(batch_id)
            } else {
                None
            };

            let request = http::Request::builder()
                .method("POST")
                .uri(uri.as_ref())
                .header("content-type", "application/octet-stream")
                .header("content-length", body.len())
                .header("x-risk-bench-mode", "throughput")
                .body(())
                .context("build h2 throughput request")?;

            let sender = &conns[conn_idx].sender;
            let mut ready = match sender.clone().ready().await {
                Ok(ready) => ready,
                Err(_) => {
                    conns[conn_idx] = open_h2_throughput_conn(target.addr).await?;
                    if let Some(batch_id) = batch_id {
                        if let Some(batch) = batches.get_mut(&batch_id) {
                            update_throughput_batch(batch, throughput_timeout_response(1));
                        }
                        maybe_finalize_throughput_batch(
                            &mut batch_flush,
                            batch_id,
                            Instant::now(),
                            &mut batches,
                        );
                    }
                    continue;
                }
            };
            let (response, mut send_stream) = match ready.send_request(request, false) {
                Ok(parts) => parts,
                Err(_) => {
                    conns[conn_idx] = open_h2_throughput_conn(target.addr).await?;
                    if let Some(batch_id) = batch_id {
                        if let Some(batch) = batches.get_mut(&batch_id) {
                            update_throughput_batch(batch, throughput_timeout_response(1));
                        }
                        maybe_finalize_throughput_batch(
                            &mut batch_flush,
                            batch_id,
                            Instant::now(),
                            &mut batches,
                        );
                    }
                    continue;
                }
            };
            if send_stream.send_data(body, true).is_err() {
                conns[conn_idx] = open_h2_throughput_conn(target.addr).await?;
                if let Some(batch_id) = batch_id {
                    if let Some(batch) = batches.get_mut(&batch_id) {
                        update_throughput_batch(batch, throughput_timeout_response(1));
                    }
                    maybe_finalize_throughput_batch(
                        &mut batch_flush,
                        batch_id,
                        Instant::now(),
                        &mut batches,
                    );
                }
                continue;
            }

            inflight_counts[conn_idx] += 1;
            inflight.insert(
                request_id,
                ThroughputInflightReq {
                    conn_idx,
                    batch_id,
                    deadline: now + timeout,
                },
            );
            pending_responses.push(Box::pin(async move {
                (
                    conn_idx,
                    request_id,
                    decode_h2_response_throughput(response).await,
                )
            }));
        }

        let now = Instant::now();
        let expired: Vec<u32> = inflight
            .iter()
            .filter_map(|(request_id, req)| (now >= req.deadline).then_some(*request_id))
            .collect();
        for request_id in expired {
            if let Some(req) = inflight.remove(&request_id) {
                inflight_counts[req.conn_idx] = inflight_counts[req.conn_idx].saturating_sub(1);
                if let Some(batch_id) = req.batch_id {
                    if let Some(batch) = batches.get_mut(&batch_id) {
                        update_throughput_batch(batch, throughput_timeout_response(1));
                    }
                    maybe_finalize_throughput_batch(&mut batch_flush, batch_id, now, &mut batches);
                }
            }
        }

        let quota_empty =
            quota.warmup.load(Ordering::Acquire) == 0 && quota.record.load(Ordering::Acquire) == 0;
        let all_idle = quota_empty
            && local_warmup_tokens == 0
            && local_record_tokens == 0
            && inflight.is_empty();
        if pacer_done.load(Ordering::Acquire) && all_idle {
            break;
        }

        if pacer_done.load(Ordering::Acquire)
            && quota_empty
            && local_warmup_tokens == 0
            && local_record_tokens == 0
        {
            let started = drain_started.get_or_insert_with(Instant::now);
            if started.elapsed() >= timeout + Duration::from_millis(100) {
                let now = Instant::now();
                let inflight_ids: Vec<u32> = inflight.keys().copied().collect();
                for request_id in inflight_ids {
                    if let Some(req) = inflight.remove(&request_id) {
                        inflight_counts[req.conn_idx] =
                            inflight_counts[req.conn_idx].saturating_sub(1);
                        if let Some(batch_id) = req.batch_id {
                            if let Some(batch) = batches.get_mut(&batch_id) {
                                update_throughput_batch(batch, throughput_timeout_response(1));
                            }
                            maybe_finalize_throughput_batch(
                                &mut batch_flush,
                                batch_id,
                                now,
                                &mut batches,
                            );
                        }
                    }
                }
                break;
            }
        } else {
            drain_started = None;
        }

        if let Ok(Some((_conn_idx, request_id, result))) =
            tokio::time::timeout(Duration::from_millis(1), pending_responses.next()).await
        {
            if let Some(req) = inflight.remove(&request_id) {
                inflight_counts[req.conn_idx] = inflight_counts[req.conn_idx].saturating_sub(1);
                if let Some(batch_id) = req.batch_id {
                    if let Some(batch) = batches.get_mut(&batch_id) {
                        match result {
                            Ok(response) => update_throughput_batch(batch, response),
                            Err(_) => {
                                update_throughput_batch(batch, throughput_timeout_response(1))
                            }
                        }
                    }
                    maybe_finalize_throughput_batch(
                        &mut batch_flush,
                        batch_id,
                        Instant::now(),
                        &mut batches,
                    );
                }
            }
        }

        if batch_flush.len() >= 32 || last_batch_flush.elapsed() >= Duration::from_millis(2) {
            maybe_flush_throughput_batches(&events_tx, &mut batch_flush);
            last_batch_flush = Instant::now();
        }
    }

    if let Some(batch_id) = open_batch_id.take() {
        if let Some(batch) = batches.get_mut(&batch_id) {
            batch.sealed = true;
        }
        maybe_finalize_throughput_batch(&mut batch_flush, batch_id, Instant::now(), &mut batches);
    }

    let remaining_batch_ids: Vec<u64> = batches.keys().copied().collect();
    for batch_id in remaining_batch_ids {
        if let Some(batch) = batches.get_mut(&batch_id) {
            batch.sealed = true;
        }
        maybe_finalize_throughput_batch(&mut batch_flush, batch_id, Instant::now(), &mut batches);
    }
    maybe_flush_throughput_batches(&events_tx, &mut batch_flush);

    Ok(())
}

fn worker_loop_h2c(
    worker_id: usize,
    worker_count: usize,
    cpu: Option<usize>,
    args: Args,
    target: Target,
    workload: WorkloadSource,
    queue: Arc<ArrayQueue<ReqToken>>,
    throughput_quota: Option<Arc<ThroughputQuota>>,
    events_tx: Sender<AggEvent>,
    pacer_done: Arc<AtomicBool>,
) -> Result<()> {
    pin_current_thread(cpu)?;
    if args.progress {
        eprintln!(
            "[bench3] worker={} start cpu={:?} transport=h2c",
            worker_id, cpu
        );
    }
    let rt = TokioRuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .context("build worker h2 runtime")?;
    match args.mode {
        BenchMode::Latency => {
            let (corpus, route_meta) = match workload {
                WorkloadSource::Corpus {
                    payload,
                    route_meta,
                } => (payload, route_meta),
                WorkloadSource::Ceiling { .. } => {
                    bail!("latency mode requires --workload corpus")
                }
            };
            rt.block_on(worker_loop_h2c_async(
                worker_id, args, target, corpus, route_meta, queue, events_tx, pacer_done,
            ))
        }
        BenchMode::Throughput => rt.block_on(worker_loop_h2c_throughput_async(
            worker_id,
            worker_count,
            args,
            target,
            workload,
            throughput_quota.expect("throughput worker requires quota ingress"),
            events_tx,
            pacer_done,
        )),
    }
}

fn encode_rvec_v3_route_header(meta: RouteMetaEntry, dim: usize) -> [u8; 40] {
    let mut out = [0u8; 40];
    out[0..4].copy_from_slice(b"RVEC");
    out[4..6].copy_from_slice(&3u16.to_le_bytes());
    out[6..8].copy_from_slice(&0u16.to_le_bytes());
    out[8..12].copy_from_slice(&(dim as u32).to_le_bytes());
    out[12..16].copy_from_slice(&meta.fold_id.to_le_bytes());
    out[16..20].copy_from_slice(&meta.seg_prod_amtbin.to_le_bytes());
    out[20..28].copy_from_slice(&meta.transaction_id.to_le_bytes());
    out[28..32].copy_from_slice(&meta.row_idx.to_le_bytes());
    out[32..36].copy_from_slice(&meta.l2_tau_used.to_le_bytes());
    out
}

fn maybe_flush_batch(tx: &Sender<AggEvent>, batch: &mut Vec<CompletedSample>) {
    if batch.is_empty() {
        return;
    }
    let items = std::mem::take(batch);
    let _ = tx.send(AggEvent::CompletionBatch { items });
}

fn flush_write(
    conn: &mut Conn,
    corpus: &PayloadCorpus,
    req_tpl: &RequestTemplate,
) -> io::Result<bool> {
    let Some(active) = conn.active else {
        return Ok(false);
    };
    let body = corpus.row_slice(active.row_idx);
    loop {
        let prefix_rem = &req_tpl.prefix[conn.write_state.prefix_off..];
        let route_rem =
            &active.route_header[conn.write_state.route_header_off..active.route_header_len];
        let body_rem = &body[conn.write_state.body_off..];
        if prefix_rem.is_empty() && route_rem.is_empty() && body_rem.is_empty() {
            return Ok(true);
        }
        let wrote = if !prefix_rem.is_empty() && !route_rem.is_empty() && !body_rem.is_empty() {
            let bufs = [
                IoSlice::new(prefix_rem),
                IoSlice::new(route_rem),
                IoSlice::new(body_rem),
            ];
            conn.stream.write_vectored(&bufs)?
        } else if !prefix_rem.is_empty() && !route_rem.is_empty() {
            let bufs = [IoSlice::new(prefix_rem), IoSlice::new(route_rem)];
            conn.stream.write_vectored(&bufs)?
        } else if !route_rem.is_empty() && !body_rem.is_empty() {
            let bufs = [IoSlice::new(route_rem), IoSlice::new(body_rem)];
            conn.stream.write_vectored(&bufs)?
        } else if !prefix_rem.is_empty() {
            conn.stream.write(prefix_rem)?
        } else if !route_rem.is_empty() {
            conn.stream.write(route_rem)?
        } else {
            conn.stream.write(body_rem)?
        };

        if wrote == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "socket write returned 0",
            ));
        }
        let prefix_advance = min(wrote, prefix_rem.len());
        conn.write_state.prefix_off += prefix_advance;
        let route_advance = min(wrote.saturating_sub(prefix_advance), route_rem.len());
        conn.write_state.route_header_off += route_advance;
        let body_advance = wrote.saturating_sub(prefix_advance + route_advance);
        conn.write_state.body_off += body_advance;

        if wrote < prefix_rem.len() + route_rem.len() + body_rem.len() {
            return Ok(false);
        }
    }
}

fn start_next_send(
    conn: &mut Conn,
    poll: &Poll,
    timeout: Duration,
    corpus: &PayloadCorpus,
    route_meta: Option<&RouteMetaCorpus>,
) -> Result<()> {
    if conn.active.is_some() {
        return Ok(());
    }
    let Some(next) = conn.pending.pop_front() else {
        return Ok(());
    };
    let (route_header, route_header_len) = if let Some(route_meta) = route_meta {
        (
            encode_rvec_v3_route_header(route_meta.row(next.row_idx), corpus.row_bytes / 4),
            40,
        )
    } else {
        ([0u8; 40], 0)
    };
    conn.active = Some(ActiveReq {
        row_idx: next.row_idx,
        t_sched: next.t_sched,
        t_issue: next.t_issue,
        record: next.record,
        t_send_done: None,
        deadline: Instant::now() + timeout,
        route_header,
        route_header_len,
    });
    conn.write_state = WriteState::default();
    conn.reading = false;
    conn.parser.reset();
    reregister_conn(poll, &mut conn.stream, conn.token, true)?;
    Ok(())
}

fn reconnect_conn(conn: &mut Conn, poll: &Poll, target: &Target) -> Result<()> {
    poll.registry()
        .deregister(&mut conn.stream)
        .context("deregister conn")?;
    conn.stream = connect_stream(target.addr)?;
    conn.reading = false;
    conn.parser.reset();
    conn.write_state = WriteState::default();
    register_conn(poll, &mut conn.stream, conn.token, false)?;
    Ok(())
}

fn reconnect_batch_conn(conn: &mut BatchConn, poll: &Poll, target: &Target) -> Result<()> {
    poll.registry()
        .deregister(&mut conn.stream)
        .context("deregister batch conn")?;
    conn.stream = connect_stream(target.addr)?;
    conn.reading = false;
    conn.parser.reset();
    conn.write_state = BatchWriteState::default();
    register_conn(poll, &mut conn.stream, conn.token, false)?;
    Ok(())
}

fn flush_write_batch(
    conn: &mut BatchConn,
    req_tpl: &RequestTemplate,
    corpus: &PayloadCorpus,
    route_meta: Option<&RouteMetaCorpus>,
) -> io::Result<bool> {
    let Some(active) = conn.active.as_ref() else {
        return Ok(true);
    };
    let prefix = req_tpl.prefix.as_slice();
    let prefix_rem = &prefix[conn.write_state.prefix_off..];
    let rec_len = record_len(corpus, route_meta);
    let total_body_len = 16 + rec_len * active.batch_records;
    let mut body_off = conn.write_state.body_off.min(total_body_len);
    let mut route_headers = [[0u8; 40]; 128];
    let mut bufs = Vec::with_capacity(2 + active.batch_records.saturating_mul(2));
    if !prefix_rem.is_empty() {
        bufs.push(IoSlice::new(prefix_rem));
    }
    if body_off < 16 {
        bufs.push(IoSlice::new(&active.batch_header[body_off..]));
        body_off = 0;
    } else {
        body_off -= 16;
    }

    let start_record = body_off / rec_len.max(1);
    let first_record_off = body_off % rec_len.max(1);
    if let Some(route_meta) = route_meta {
        let dim = corpus.row_bytes / 4;
        for (record_idx, header) in route_headers
            .iter_mut()
            .enumerate()
            .take(active.batch_records)
        {
            let row_idx = (active.start_row_idx + record_idx) % corpus.rows;
            *header = encode_rvec_v3_route_header(route_meta.row(row_idx), dim);
        }
        for record_idx in start_record..active.batch_records {
            let row_idx = (active.start_row_idx + record_idx) % corpus.rows;
            let mut rec_off = if record_idx == start_record {
                first_record_off
            } else {
                0
            };
            if rec_off < 40 {
                bufs.push(IoSlice::new(&route_headers[record_idx][rec_off..]));
                rec_off = 0;
            } else {
                rec_off -= 40;
            }
            let row = corpus.row_slice(row_idx);
            bufs.push(IoSlice::new(&row[rec_off..]));
        }
    } else {
        for record_idx in start_record..active.batch_records {
            let row_idx = (active.start_row_idx + record_idx) % corpus.rows;
            let rec_off = if record_idx == start_record {
                first_record_off
            } else {
                0
            };
            let row = corpus.row_slice(row_idx);
            bufs.push(IoSlice::new(&row[rec_off..]));
        }
    }

    let wrote = conn.stream.write_vectored(&bufs)?;
    if wrote == 0 {
        return Err(io::Error::new(
            io::ErrorKind::WriteZero,
            "socket write returned 0",
        ));
    }

    let mut left = wrote;
    let take_prefix = left.min(prefix_rem.len());
    conn.write_state.prefix_off += take_prefix;
    left -= take_prefix;
    let take_body = left.min(total_body_len.saturating_sub(conn.write_state.body_off));
    conn.write_state.body_off += take_body;

    Ok(conn.write_state.prefix_off >= prefix.len() && conn.write_state.body_off >= total_body_len)
}

fn start_next_send_batch(
    conn: &mut BatchConn,
    poll: &Poll,
    timeout: Duration,
    corpus: &PayloadCorpus,
    route_meta: Option<&RouteMetaCorpus>,
    batch_records: usize,
) -> Result<()> {
    if conn.active.is_some() {
        return Ok(());
    }
    let Some(next) = conn.pending.pop_front() else {
        return Ok(());
    };
    let rows = next.rows.max(1) as usize;
    let batch_records = rows.min(batch_records.max(1));
    let batch_header = encode_batch_header(
        batch_records,
        record_len(corpus, route_meta),
        route_meta.is_some(),
    );
    conn.active = Some(BatchActiveReq {
        start_row_idx: next.row_idx,
        t_sched: next.t_sched,
        deadline: Instant::now() + timeout,
        record: next.record,
        rows: batch_records as u32,
        batch_records,
        batch_header,
    });
    conn.write_state = BatchWriteState::default();
    conn.reading = false;
    conn.parser.reset();
    reregister_conn(poll, &mut conn.stream, conn.token, true)?;
    Ok(())
}

fn throughput_done_from_batch_ack(
    active: &BatchActiveReq,
    done_at: Instant,
    meta: HttpResponseMeta,
    ack: BatchAck,
) -> ThroughputBatchDone {
    let rows = if (200..300).contains(&meta.status_code) {
        ack.ok_count as u64
    } else {
        0
    };
    ThroughputBatchDone {
        latency_us: done_at.duration_since(active.t_sched).as_micros() as u64,
        ok: rows,
        err: if (200..300).contains(&meta.status_code) {
            0
        } else {
            active.rows as u64
        },
        timeout: 0,
        http_2xx: rows,
        http_429: if meta.status_code == 429 {
            active.rows as u64
        } else {
            0
        },
        http_5xx: if meta.status_code >= 500 {
            active.rows as u64
        } else {
            0
        },
        qsb2_samples: 0,
        rsk1_samples: 0,
        used_l2: ack.used_l2_count as u64,
        decision_counts: [
            ack.decision_counts[0] as u64,
            ack.decision_counts[1] as u64,
            ack.decision_counts[2] as u64,
            ack.decision_counts[3] as u64,
            ack.decision_counts[4] as u64,
        ],
    }
}

fn throughput_done_from_batch_timeout(
    active: &BatchActiveReq,
    now: Instant,
) -> ThroughputBatchDone {
    ThroughputBatchDone {
        latency_us: now.duration_since(active.t_sched).as_micros() as u64,
        ok: 0,
        err: 0,
        timeout: active.rows as u64,
        http_2xx: 0,
        http_429: 0,
        http_5xx: 0,
        qsb2_samples: 0,
        rsk1_samples: 0,
        used_l2: 0,
        decision_counts: [0, 0, 0, 0, active.rows as u64],
    }
}

fn worker_loop_http(
    worker_id: usize,
    cpu: Option<usize>,
    args: Args,
    target: Target,
    corpus: PayloadCorpus,
    route_meta: Option<RouteMetaCorpus>,
    req_tpl: RequestTemplate,
    queue: Arc<ArrayQueue<ReqToken>>,
    events_tx: Sender<AggEvent>,
    pacer_done: Arc<AtomicBool>,
) -> Result<()> {
    pin_current_thread(cpu)?;
    if args.progress {
        eprintln!("[bench3] worker={} start cpu={:?}", worker_id, cpu);
    }
    let mut poll = Poll::new().context("mio poll")?;
    let mut events = Events::with_capacity(1024);
    let timeout = Duration::from_millis(args.timeout_ms.max(1));

    let mut conns = Vec::with_capacity(args.conns_per_worker);
    for i in 0..args.conns_per_worker {
        let mut stream = connect_stream(target.addr)?;
        let token = Token(i);
        register_conn(&poll, &mut stream, token, false)?;
        conns.push(Conn {
            token,
            stream,
            pending: VecDeque::with_capacity(args.max_inflight_per_conn.max(1)),
            active: None,
            write_state: WriteState::default(),
            reading: false,
            parser: ResponseParser::with_capacity(),
        });
    }

    let mut local_pending: VecDeque<QueuedReq> = VecDeque::with_capacity(
        args.conns_per_worker
            .saturating_mul(args.max_inflight_per_conn.max(1))
            .saturating_mul(4),
    );
    let mut next_conn_rr = worker_id % args.conns_per_worker.max(1);
    let mut batch = Vec::with_capacity(FLUSH_BATCH);
    let mut last_flush = Instant::now();
    let mut drain_started: Option<Instant> = None;

    loop {
        while let Some(tok) = queue.pop() {
            local_pending.push_back(QueuedReq {
                row_idx: tok.row_idx,
                t_sched: tok.t_sched,
                t_issue: Instant::now(),
                record: tok.record,
                rows: tok.rows,
            });
        }

        while let Some(req) = local_pending.pop_front() {
            let mut selected = None;
            for step in 0..conns.len() {
                let idx = (next_conn_rr + step) % conns.len();
                if conns[idx].load() < args.max_inflight_per_conn.max(1) {
                    selected = Some(idx);
                    next_conn_rr = (idx + 1) % conns.len();
                    break;
                }
            }
            let Some(idx) = selected else {
                local_pending.push_front(req);
                break;
            };
            conns[idx].pending.push_back(req);
            start_next_send(
                &mut conns[idx],
                &poll,
                timeout,
                &corpus,
                route_meta.as_ref(),
            )?;
        }

        let now = Instant::now();
        for conn in &mut conns {
            if let Some(active) = conn.active {
                if active.t_send_done.is_some() && now >= active.deadline {
                    if active.record {
                        batch.push(build_timeout(&active, now));
                    }
                    conn.active = None;
                    conn.pending.clear();
                    reconnect_conn(conn, &poll, &target)?;
                    start_next_send(conn, &poll, timeout, &corpus, route_meta.as_ref())?;
                }
            }
        }

        if batch.len() >= FLUSH_BATCH || last_flush.elapsed() >= Duration::from_millis(2) {
            maybe_flush_batch(&events_tx, &mut batch);
            last_flush = Instant::now();
        }

        let all_idle = queue.is_empty()
            && local_pending.is_empty()
            && conns
                .iter()
                .all(|c| c.active.is_none() && c.pending.is_empty());
        if pacer_done.load(Ordering::Acquire) && all_idle {
            break;
        }

        if pacer_done.load(Ordering::Acquire) && queue.is_empty() && local_pending.is_empty() {
            let started = drain_started.get_or_insert_with(Instant::now);
            if started.elapsed() >= timeout + Duration::from_millis(100) {
                let now = Instant::now();
                for conn in &mut conns {
                    if let Some(active) = conn.active.take() {
                        if active.record {
                            batch.push(build_timeout(&active, now));
                        }
                    }
                    conn.pending.clear();
                    conn.reading = false;
                    conn.parser.reset();
                }
                break;
            }
        } else {
            drain_started = None;
        }

        poll.poll(&mut events, Some(Duration::from_millis(2)))
            .context("poll")?;

        for ev in events.iter() {
            let idx = ev.token().0;
            if idx >= conns.len() {
                continue;
            }
            let conn = &mut conns[idx];
            if ev.is_writable()
                && conn.active.is_some()
                && conn.active.unwrap().t_send_done.is_none()
            {
                match flush_write(conn, &corpus, &req_tpl) {
                    Ok(true) => {
                        if let Some(active) = conn.active.as_mut() {
                            active.t_send_done = Some(Instant::now());
                            active.deadline = Instant::now() + timeout;
                        }
                        conn.reading = true;
                        reregister_conn(&poll, &mut conn.stream, conn.token, false)?;
                    }
                    Ok(false) => {}
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                    Err(_) => {
                        let active = conn.active.take();
                        conn.pending.clear();
                        reconnect_conn(conn, &poll, &target)?;
                        if let Some(active) = active {
                            if active.record {
                                batch.push(build_timeout(&active, Instant::now()));
                            }
                        }
                        start_next_send(conn, &poll, timeout, &corpus, route_meta.as_ref())?;
                    }
                }
            }

            if ev.is_readable() && conn.reading {
                match conn.parser.read_from(&mut conn.stream) {
                    Ok(Some((meta, body))) => {
                        let done_at = Instant::now();
                        if let Some(active) = conn.active.take() {
                            if active.record {
                                batch.push(build_completion(&active, done_at, meta, body));
                            }
                        }
                        conn.parser.reset();
                        conn.reading = false;
                        start_next_send(conn, &poll, timeout, &corpus, route_meta.as_ref())?;
                    }
                    Ok(None) => {}
                    Err(_) => {
                        let active = conn.active.take();
                        conn.pending.clear();
                        reconnect_conn(conn, &poll, &target)?;
                        if let Some(active) = active {
                            if active.record {
                                batch.push(build_timeout(&active, Instant::now()));
                            }
                        }
                        start_next_send(conn, &poll, timeout, &corpus, route_meta.as_ref())?;
                    }
                }
            }
        }
    }

    maybe_flush_batch(&events_tx, &mut batch);
    if args.progress {
        eprintln!("[bench3] worker={} done", worker_id);
    }
    Ok(())
}

fn worker_loop_http_batch(
    worker_id: usize,
    cpu: Option<usize>,
    args: Args,
    target: Target,
    corpus: PayloadCorpus,
    route_meta: Option<RouteMetaCorpus>,
    req_tpl: RequestTemplate,
    queue: Arc<ArrayQueue<ReqToken>>,
    shared_stats: Arc<HttpBatchWorkerStats>,
    pacer_done: Arc<AtomicBool>,
) -> Result<()> {
    pin_current_thread(cpu)?;
    if args.progress {
        eprintln!(
            "[bench3] worker={} start cpu={:?} transport=http1-batch",
            worker_id, cpu
        );
    }
    let mut poll = Poll::new().context("mio poll")?;
    let mut events = Events::with_capacity(1024);
    let timeout = Duration::from_millis(args.timeout_ms.max(1));

    let mut conns = Vec::with_capacity(args.conns_per_worker);
    for i in 0..args.conns_per_worker {
        let mut stream = connect_stream(target.addr)?;
        let token = Token(i);
        register_conn(&poll, &mut stream, token, false)?;
        conns.push(BatchConn {
            token,
            stream,
            pending: VecDeque::with_capacity(args.max_inflight_per_conn.max(1)),
            active: None,
            write_state: BatchWriteState::default(),
            reading: false,
            parser: ResponseParser::with_capacity(),
        });
    }

    let mut local_pending: VecDeque<QueuedReq> = VecDeque::with_capacity(
        args.conns_per_worker
            .saturating_mul(args.max_inflight_per_conn.max(1))
            .saturating_mul(4),
    );
    let mut next_conn_rr = worker_id % args.conns_per_worker.max(1);
    let mut drain_started: Option<Instant> = None;
    let mut local_stats = HttpBatchWorkerLocal::default();
    let mut last_local_flush = Instant::now();

    loop {
        let pacing_done = pacer_done.load(Ordering::Acquire);
        if !pacing_done {
            while let Some(tok) = queue.pop() {
                local_pending.push_back(QueuedReq {
                    row_idx: tok.row_idx,
                    t_sched: tok.t_sched,
                    t_issue: Instant::now(),
                    record: tok.record,
                    rows: tok.rows,
                });
            }
        }

        while let Some(req) = local_pending.pop_front() {
            let mut selected = None;
            for step in 0..conns.len() {
                let idx = (next_conn_rr + step) % conns.len();
                if conns[idx].load() < args.max_inflight_per_conn.max(1) {
                    selected = Some(idx);
                    next_conn_rr = (idx + 1) % conns.len();
                    break;
                }
            }
            let Some(idx) = selected else {
                local_pending.push_front(req);
                break;
            };
            conns[idx].pending.push_back(req);
            start_next_send_batch(
                &mut conns[idx],
                &poll,
                timeout,
                &corpus,
                route_meta.as_ref(),
                args.batch_records,
            )?;
        }

        let now = Instant::now();
        for conn in &mut conns {
            if let Some(active) = conn.active.as_ref() {
                if now >= active.deadline {
                    let active = conn.active.take();
                    conn.pending.clear();
                    reconnect_batch_conn(conn, &poll, &target)?;
                    if let Some(active) = active {
                        if active.record {
                            local_stats
                                .record_batch(throughput_done_from_batch_timeout(&active, now));
                        }
                    }
                    start_next_send_batch(
                        conn,
                        &poll,
                        timeout,
                        &corpus,
                        route_meta.as_ref(),
                        args.batch_records,
                    )?;
                }
            }
        }

        let all_idle = (pacing_done || queue.is_empty())
            && local_pending.is_empty()
            && conns
                .iter()
                .all(|c| c.active.is_none() && c.pending.is_empty());
        if pacing_done && all_idle {
            break;
        }

        if pacing_done {
            let started = drain_started.get_or_insert_with(Instant::now);
            if started.elapsed() >= timeout + Duration::from_millis(100) {
                let now = Instant::now();
                local_stats.record_dropped_after_attempt(discard_batch_rows(
                    &mut local_pending,
                    &mut conns,
                ));
                for conn in &mut conns {
                    if let Some(active) = conn.active.take() {
                        if active.record {
                            local_stats
                                .record_batch(throughput_done_from_batch_timeout(&active, now));
                        }
                    }
                    conn.reading = false;
                    conn.parser.reset();
                }
                break;
            }
        } else {
            drain_started = None;
        }

        poll.poll(&mut events, Some(Duration::from_millis(2)))
            .context("poll")?;

        for ev in events.iter() {
            let idx = ev.token().0;
            if idx >= conns.len() {
                continue;
            }
            let conn = &mut conns[idx];
            if ev.is_writable() && conn.active.is_some() {
                match flush_write_batch(conn, &req_tpl, &corpus, route_meta.as_ref()) {
                    Ok(true) => {
                        if let Some(active) = conn.active.as_mut() {
                            active.deadline = Instant::now() + timeout;
                        }
                        conn.reading = true;
                        reregister_conn(&poll, &mut conn.stream, conn.token, false)?;
                    }
                    Ok(false) => {}
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                    Err(_) => {
                        let active = conn.active.take();
                        conn.pending.clear();
                        reconnect_batch_conn(conn, &poll, &target)?;
                        if let Some(active) = active {
                            if active.record {
                                local_stats.record_batch(throughput_done_from_batch_timeout(
                                    &active,
                                    Instant::now(),
                                ));
                            }
                        }
                        start_next_send_batch(
                            conn,
                            &poll,
                            timeout,
                            &corpus,
                            route_meta.as_ref(),
                            args.batch_records,
                        )?;
                    }
                }
            }

            if ev.is_readable() && conn.reading {
                match conn.parser.read_batch_ack_from(&mut conn.stream) {
                    Ok(Some((meta, ack))) => {
                        let done_at = Instant::now();
                        if let Some(active) = conn.active.take() {
                            if active.record {
                                let batch =
                                    throughput_done_from_batch_ack(&active, done_at, meta, ack);
                                local_stats.record_batch(batch);
                            }
                        }
                        conn.parser.reset();
                        conn.reading = false;
                        start_next_send_batch(
                            conn,
                            &poll,
                            timeout,
                            &corpus,
                            route_meta.as_ref(),
                            args.batch_records,
                        )?;
                    }
                    Ok(None) => {}
                    Err(_) => {
                        let active = conn.active.take();
                        conn.pending.clear();
                        reconnect_batch_conn(conn, &poll, &target)?;
                        if let Some(active) = active {
                            if active.record {
                                local_stats.record_batch(throughput_done_from_batch_timeout(
                                    &active,
                                    Instant::now(),
                                ));
                            }
                        }
                        start_next_send_batch(
                            conn,
                            &poll,
                            timeout,
                            &corpus,
                            route_meta.as_ref(),
                            args.batch_records,
                        )?;
                    }
                }
            }
        }

        if last_local_flush.elapsed() >= Duration::from_millis(100) {
            local_stats.flush_into(&shared_stats);
            last_local_flush = Instant::now();
        }
    }
    local_stats.flush_into(&shared_stats);
    shared_stats.done.store(true, Ordering::Release);
    if args.progress {
        eprintln!("[bench3] worker={} done transport=http1-batch", worker_id);
    }
    Ok(())
}

fn exp_interval_us(rng: &mut SmallRng, lambda: f64) -> u64 {
    let mut u = rng.gen::<f64>();
    if u <= 0.0 {
        u = 1e-12;
    }
    let dt_us = -u.ln() / lambda;
    dt_us.max(1.0) as u64
}

fn sleep_until(deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let remain = deadline - now;
        if remain > Duration::from_micros(200) {
            thread::sleep(remain - Duration::from_micros(100));
        } else {
            std::hint::spin_loop();
        }
    }
}

fn pacer_loop(
    args: Args,
    worker_queues: Vec<Arc<ArrayQueue<ReqToken>>>,
    corpus_rows: usize,
    events_tx: Sender<AggEvent>,
    pacer_done: Arc<AtomicBool>,
) -> Result<()> {
    pin_current_thread(args.pacer_cpu)?;
    if args.progress {
        eprintln!(
            "[bench3] pacer start cpu={:?} rps={} warmup={}s duration={}s",
            args.pacer_cpu, args.rps, args.warmup, args.duration
        );
    }
    if args.batch_mode == BatchMode::Http1 {
        let start = Instant::now();
        let warmup_end = start + Duration::from_secs(args.warmup);
        let end = start + Duration::from_secs(args.warmup + args.duration);
        let epoch_us = 100u64;
        let mut next_at = start;
        let rows_per_token = args.batch_records.max(1) as u64;
        let target_tokens_per_sec = (args.rps.max(1) as f64 / rows_per_token as f64).max(1.0);
        let tokens_per_epoch = target_tokens_per_sec * (epoch_us as f64 / 1_000_000.0);
        let mut token_carry = 0.0f64;
        let mut rr = 0usize;
        let mut attempted_rows = 0u64;
        let mut dropped_rows = 0u64;
        let mut req_id = 0u64;
        let mut last_progress = Instant::now();

        while Instant::now() < end {
            sleep_until(next_at);
            let now = Instant::now();
            let record = now >= warmup_end;
            token_carry += tokens_per_epoch;
            let issue = token_carry.floor() as usize;
            token_carry -= issue as f64;

            for _ in 0..issue {
                let tok = ReqToken {
                    row_idx: (((req_id as usize).wrapping_mul(1315423911)) ^ rr) % corpus_rows,
                    t_sched: now,
                    record,
                    rows: rows_per_token as u32,
                };
                req_id = req_id.wrapping_add(1);
                let queue = &worker_queues[rr % worker_queues.len()];
                rr = rr.wrapping_add(1);

                if record {
                    attempted_rows += rows_per_token;
                }
                if queue.push(tok).is_err() && record {
                    dropped_rows += rows_per_token;
                }
            }

            if attempted_rows >= 2048 || dropped_rows > 0 {
                let _ = events_tx.send(AggEvent::AttemptBatch {
                    attempted: attempted_rows,
                    dropped_conn_queue_full: dropped_rows,
                });
                attempted_rows = 0;
                dropped_rows = 0;
            }

            if args.progress && last_progress.elapsed() >= Duration::from_secs(1) {
                eprintln!(
                    "[bench3] pacer progress req_id={} attempted_rows={} dropped_rows={} mode=http1-batch",
                    req_id, attempted_rows, dropped_rows
                );
                last_progress = Instant::now();
            }

            next_at += Duration::from_micros(epoch_us);
        }

        if attempted_rows > 0 || dropped_rows > 0 {
            let _ = events_tx.send(AggEvent::AttemptBatch {
                attempted: attempted_rows,
                dropped_conn_queue_full: dropped_rows,
            });
        }
        pacer_done.store(true, Ordering::Release);
        if args.progress {
            eprintln!("[bench3] pacer done req_id={} mode=http1-batch", req_id);
        }
        return Ok(());
    }
    let total_duration = Duration::from_secs(args.warmup + args.duration);
    let warmup_end = Instant::now() + Duration::from_secs(args.warmup);
    let end = Instant::now() + total_duration;
    let mut next_at = Instant::now();
    let rows_per_token = if args.batch_mode == BatchMode::Http1 {
        args.batch_records.max(1) as u64
    } else {
        1
    };
    let target_tokens_per_sec = if args.batch_mode == BatchMode::Http1 {
        (args.rps.max(1) as f64 / rows_per_token as f64).max(1.0)
    } else {
        args.rps.max(1) as f64
    };
    let lambda = target_tokens_per_sec / 1_000_000.0;
    let fixed_us = (1_000_000.0 / target_tokens_per_sec).max(1.0) as u64;
    let mut rng = SmallRng::seed_from_u64(0x51f15eed_u64);
    let mut rr = 0usize;
    let mut attempted_batch = 0u64;
    let mut dropped_batch = 0u64;
    let mut req_id = 0u64;
    let mut last_progress = Instant::now();

    while Instant::now() < end {
        sleep_until(next_at);
        let now = Instant::now();
        let record = now >= warmup_end;
        let tok = ReqToken {
            row_idx: (((req_id as usize).wrapping_mul(1315423911)) ^ rr) % corpus_rows,
            t_sched: now,
            record,
            rows: rows_per_token as u32,
        };
        req_id = req_id.wrapping_add(1);
        let queue = &worker_queues[rr % worker_queues.len()];
        rr = rr.wrapping_add(1);

        if !record && args.batch_mode == BatchMode::Http1 {
            let dt_us = match args.pacer {
                PacerKind::Fixed => fixed_us,
                PacerKind::Poisson => exp_interval_us(&mut rng, lambda),
            };
            next_at += Duration::from_micros(dt_us);
            continue;
        }

        if record {
            attempted_batch += rows_per_token;
        }
        if queue.push(tok).is_err() && record {
            dropped_batch += rows_per_token;
        }
        if attempted_batch >= 256 {
            let _ = events_tx.send(AggEvent::AttemptBatch {
                attempted: attempted_batch,
                dropped_conn_queue_full: dropped_batch,
            });
            attempted_batch = 0;
            dropped_batch = 0;
        }

        if args.progress && last_progress.elapsed() >= Duration::from_secs(1) {
            eprintln!(
                "[bench3] pacer progress req_id={} attempted_batch={} dropped_batch={}",
                req_id, attempted_batch, dropped_batch
            );
            last_progress = Instant::now();
        }

        let dt_us = match args.pacer {
            PacerKind::Fixed => fixed_us,
            PacerKind::Poisson => exp_interval_us(&mut rng, lambda),
        };
        next_at += Duration::from_micros(dt_us);
    }

    if attempted_batch > 0 || dropped_batch > 0 {
        let _ = events_tx.send(AggEvent::AttemptBatch {
            attempted: attempted_batch,
            dropped_conn_queue_full: dropped_batch,
        });
    }
    pacer_done.store(true, Ordering::Release);
    if args.progress {
        eprintln!("[bench3] pacer done req_id={}", req_id);
    }
    Ok(())
}

fn pacer_loop_http_batch_soft(
    args: Args,
    worker_queues: Vec<Arc<ArrayQueue<ReqToken>>>,
    worker_stats: Vec<Arc<HttpBatchWorkerStats>>,
    corpus_rows: usize,
    pacer_done: Arc<AtomicBool>,
) -> Result<()> {
    pin_current_thread(args.pacer_cpu)?;
    let start = Instant::now();
    let warmup_end = start + Duration::from_secs(args.warmup);
    let end = start + Duration::from_secs(args.warmup + args.duration);
    let tick = Duration::from_millis(1);
    let batch_rows = args.batch_records.max(1) as u64;
    let rows_per_tick_fp = args.rps.max(1).saturating_mul(1_000);
    let denom = 1_000_000u64;
    let mut next_tick = start;
    let mut fp_acc = 0u64;
    let mut rr = 0usize;
    let mut seq = 0u64;

    while Instant::now() < end {
        sleep_until(next_tick);
        next_tick += tick;
        fp_acc = fp_acc.saturating_add(rows_per_tick_fp);
        while fp_acc >= denom.saturating_mul(batch_rows) {
            let worker_idx = rr % worker_queues.len();
            rr = rr.wrapping_add(1);
            let row_idx = (((seq as usize).wrapping_mul(args.batch_records.max(1))) ^ worker_idx)
                % corpus_rows;
            let record = Instant::now() >= warmup_end;
            let tok = ReqToken {
                row_idx,
                t_sched: Instant::now(),
                record,
                rows: batch_rows as u32,
            };
            if worker_queues[worker_idx].push(tok).is_ok() {
                if record {
                    worker_stats[worker_idx]
                        .attempted
                        .fetch_add(batch_rows, Ordering::Relaxed);
                }
            } else if record {
                worker_stats[worker_idx]
                    .dropped_conn_queue_full
                    .fetch_add(batch_rows, Ordering::Relaxed);
            }
            fp_acc -= denom.saturating_mul(batch_rows);
            seq = seq.wrapping_add(1);
        }
    }

    pacer_done.store(true, Ordering::Release);
    Ok(())
}

fn pacer_loop_throughput(
    args: Args,
    worker_quotas: Vec<Arc<ThroughputQuota>>,
    events_tx: Sender<AggEvent>,
    pacer_done: Arc<AtomicBool>,
) -> Result<()> {
    pin_current_thread(args.pacer_cpu)?;
    if args.progress {
        eprintln!(
            "[bench3] pacer start cpu={:?} rps={} warmup={}s duration={}s mode=throughput",
            args.pacer_cpu, args.rps, args.warmup, args.duration
        );
    }
    let start = Instant::now();
    let warmup_end = start + Duration::from_secs(args.warmup);
    let end = start + Duration::from_secs(args.warmup + args.duration);
    let mut next_at = start;
    let lambda = args.rps.max(1) as f64 / 1_000_000.0;
    let fixed_us = (1_000_000.0 / args.rps.max(1) as f64).max(1.0) as u64;
    let mut rng = SmallRng::seed_from_u64(0x51f15eed_u64);
    let mut rr = 0usize;
    let mut attempted_batch = 0u64;
    let mut dropped_batch = 0u64;
    let local_cap = args
        .conns_per_worker
        .saturating_mul(args.max_inflight_per_conn.max(1))
        .saturating_mul(8)
        .max(1) as u64;
    let mut req_id = 0u64;
    let mut last_progress = Instant::now();

    while Instant::now() < end {
        sleep_until(next_at);
        let now = Instant::now();
        let record = now >= warmup_end;
        let quota = &worker_quotas[rr % worker_quotas.len()];
        rr = rr.wrapping_add(1);
        if record {
            attempted_batch += 1;
        }
        let counter = if record { &quota.record } else { &quota.warmup };
        let prev = counter.fetch_add(1, Ordering::AcqRel);
        if prev >= local_cap {
            counter.fetch_sub(1, Ordering::AcqRel);
            if record {
                dropped_batch += 1;
            }
        }
        req_id = req_id.wrapping_add(1);

        if attempted_batch >= 256 {
            let _ = events_tx.send(AggEvent::AttemptBatch {
                attempted: attempted_batch,
                dropped_conn_queue_full: dropped_batch,
            });
            attempted_batch = 0;
            dropped_batch = 0;
        }

        if args.progress && last_progress.elapsed() >= Duration::from_secs(1) {
            eprintln!(
                "[bench3] pacer progress req_id={} attempted_batch={} dropped_batch={} mode=throughput",
                req_id, attempted_batch, dropped_batch
            );
            last_progress = Instant::now();
        }

        let dt_us = match args.pacer {
            PacerKind::Fixed => fixed_us,
            PacerKind::Poisson => exp_interval_us(&mut rng, lambda),
        };
        next_at += Duration::from_micros(dt_us);
    }

    if attempted_batch > 0 || dropped_batch > 0 {
        let _ = events_tx.send(AggEvent::AttemptBatch {
            attempted: attempted_batch,
            dropped_conn_queue_full: dropped_batch,
        });
    }
    pacer_done.store(true, Ordering::Release);
    if args.progress {
        eprintln!("[bench3] pacer done req_id={} mode=throughput", req_id);
    }
    Ok(())
}

fn write_window_csv_header(w: &mut dyn Write) -> Result<()> {
    writeln!(
        w,
        "t_ms,attempted,ok,err,timeout,dropped,drop_inflight_cap,drop_conn_queue_full,http_2xx,http_429,http_5xx,queue_p99_us,send_p99_us,server_rtt_p99_us,e2e_p99_us,batch_e2e_p99_us,stage_router_p99_us,stage_l1_p99_us,stage_l2_p99_us"
    )?;
    Ok(())
}

fn write_window_row(w: &mut dyn Write, t_ms: u128, agg: &StatsAgg) -> Result<()> {
    writeln!(
        w,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        t_ms,
        agg.attempted,
        agg.ok,
        agg.err,
        agg.timeout,
        agg.dropped,
        agg.drop_inflight_cap,
        agg.drop_conn_queue_full,
        agg.http_2xx,
        agg.http_429,
        agg.http_5xx,
        hist_q(&agg.queue_delay_us, 0.99),
        hist_q(&agg.send_delay_us, 0.99),
        hist_q(&agg.server_rtt_us, 0.99),
        hist_q(&agg.e2e_us, 0.99),
        hist_q(&agg.batch_e2e_us, 0.99),
        hist_q(&agg.stage_router, 0.99),
        hist_q(&agg.stage_l1, 0.99),
        hist_q(&agg.stage_l2, 0.99)
    )?;
    Ok(())
}

fn drain_http_batch_worker_stats(
    page: &HttpBatchWorkerStats,
    total: &mut StatsAgg,
    window: &mut StatsAgg,
) {
    let attempted = page.attempted.swap(0, Ordering::Relaxed);
    let dropped = page.dropped_conn_queue_full.swap(0, Ordering::Relaxed);
    if attempted > 0 || dropped > 0 {
        total.record_attempts(attempted, dropped);
        window.record_attempts(attempted, dropped);
    }
    let dropped_after_attempt = page.dropped_after_attempt.swap(0, Ordering::Relaxed);
    if dropped_after_attempt > 0 {
        total.record_dropped_after_attempt(dropped_after_attempt);
        window.record_dropped_after_attempt(dropped_after_attempt);
    }
    macro_rules! drain_counter {
        ($field:ident) => {{
            let v = page.$field.swap(0, Ordering::Relaxed);
            total.$field += v;
            window.$field += v;
        }};
    }
    drain_counter!(ok);
    drain_counter!(err);
    drain_counter!(timeout);
    drain_counter!(http_2xx);
    drain_counter!(http_429);
    drain_counter!(http_5xx);
    drain_counter!(qsb2_samples);
    drain_counter!(rsk1_samples);
    drain_counter!(used_l2);
    for ((dst_total, dst_window), src) in total
        .decision_counts
        .iter_mut()
        .zip(window.decision_counts.iter_mut())
        .zip(page.decision_counts.iter())
    {
        let v = src.swap(0, Ordering::Relaxed);
        *dst_total += v;
        *dst_window += v;
    }
    let mut lats = page.batch_latency_us.lock().expect("batch latency mutex");
    for lat in lats.drain(..) {
        let lat = lat.max(1);
        let _ = total.batch_e2e_us.record(lat);
        let _ = window.batch_e2e_us.record(lat);
    }
}

fn run_aggregator_http_batch_pull(
    args: &Args,
    pages: &[Arc<HttpBatchWorkerStats>],
    start: Instant,
) -> Result<StatsAgg> {
    let mut total = StatsAgg::new()?;
    let mut window = StatsAgg::new()?;
    let mut next_window = if args.window_ms > 0 {
        Some(start + Duration::from_millis(args.window_ms))
    } else {
        None
    };

    let mut window_writer: Option<Box<dyn Write + Send>> = if let Some(path) =
        args.window_csv.as_ref()
    {
        let mut f: Box<dyn Write + Send> = Box::new(
            File::create(path).with_context(|| format!("create window csv: {}", path.display()))?,
        );
        write_window_csv_header(&mut *f)?;
        Some(f)
    } else {
        None
    };

    loop {
        let mut all_done = true;
        for page in pages {
            drain_http_batch_worker_stats(page, &mut total, &mut window);
            all_done &= page.done.load(Ordering::Acquire);
        }

        if let Some(deadline) = next_window {
            if Instant::now() >= deadline {
                if let Some(w) = window_writer.as_mut() {
                    write_window_row(&mut **w, start.elapsed().as_millis(), &window)?;
                }
                window.reset_window()?;
                next_window = Some(deadline + Duration::from_millis(args.window_ms));
            }
        }

        if all_done {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }

    if let Some(w) = window_writer.as_mut() {
        write_window_row(&mut **w, start.elapsed().as_millis(), &window)?;
    }
    Ok(total)
}

fn run_aggregator(
    args: &Args,
    rx: Receiver<AggEvent>,
    worker_count: usize,
    start: Instant,
) -> Result<StatsAgg> {
    let mut total = StatsAgg::new()?;
    let mut window = StatsAgg::new()?;
    let mut done_workers = 0usize;
    let mut next_window = if args.window_ms > 0 {
        Some(start + Duration::from_millis(args.window_ms))
    } else {
        None
    };

    let mut window_writer: Option<Box<dyn Write + Send>> = if let Some(path) =
        args.window_csv.as_ref()
    {
        let mut f: Box<dyn Write + Send> = Box::new(
            File::create(path).with_context(|| format!("create window csv: {}", path.display()))?,
        );
        write_window_csv_header(&mut *f)?;
        Some(f)
    } else {
        None
    };

    while done_workers < worker_count {
        let timeout = next_window
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
            .unwrap_or_else(|| Duration::from_millis(100));
        match rx.recv_timeout(timeout) {
            Ok(AggEvent::AttemptBatch {
                attempted,
                dropped_conn_queue_full,
            }) => {
                total.record_attempts(attempted, dropped_conn_queue_full);
                window.record_attempts(attempted, dropped_conn_queue_full);
            }
            Ok(AggEvent::CompletionBatch { items }) => {
                for item in &items {
                    total.record_completion(item);
                    window.record_completion(item);
                }
            }
            Ok(AggEvent::ThroughputBatch { batch }) => {
                total.record_throughput_batch(&batch);
                window.record_throughput_batch(&batch);
            }
            Ok(AggEvent::ThroughputBatches { items }) => {
                for batch in &items {
                    total.record_throughput_batch(batch);
                    window.record_throughput_batch(batch);
                }
            }
            Ok(AggEvent::WorkerDone) => {
                done_workers += 1;
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }

        if let Some(deadline) = next_window {
            if Instant::now() >= deadline {
                if let Some(w) = window_writer.as_mut() {
                    write_window_row(&mut **w, start.elapsed().as_millis(), &window)?;
                }
                window.reset_window()?;
                next_window = Some(deadline + Duration::from_millis(args.window_ms));
            }
        }
    }

    if let Some(w) = window_writer.as_mut() {
        write_window_row(&mut **w, start.elapsed().as_millis(), &window)?;
    }
    Ok(total)
}

fn pick_workers(args: &Args, worker_cpus: &Option<Vec<usize>>) -> usize {
    if let Some(cpus) = worker_cpus {
        return cpus.len().max(1);
    }
    if args.workers > 0 {
        return args.workers;
    }
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).max(1))
        .unwrap_or(1)
}

fn verdict_for(args: &Args, total: &StatsAgg) -> (Verdict, Vec<String>, Vec<String>) {
    let attempted_rps = total.attempted as f64 / (args.duration as f64).max(1.0);
    let offered_rps =
        (total.attempted + total.drop_conn_queue_full) as f64 / (args.duration as f64).max(1.0);
    let queue_p99 = match args.mode {
        BenchMode::Latency => hist_q(&total.queue_delay_us, 0.99),
        BenchMode::Throughput => hist_q(&total.batch_e2e_us, 0.99),
    };
    let target_period_us = ((1_000_000.0) / args.rps.max(1) as f64).max(1.0) as u64;

    let mut client = Vec::new();
    let mut server = Vec::new();

    if total.drop_inflight_cap > 0 {
        client.push(format!("drop_inflight_cap={}", total.drop_inflight_cap));
    }
    if attempted_rps < (args.rps as f64) * 0.98 {
        if args.batch_mode == BatchMode::Http1 {
            client.push(format!(
                "under_target_rows={:.1} offered_rows={:.1}",
                (args.rps as f64 - attempted_rps).max(0.0),
                offered_rps
            ));
        } else {
            client.push(format!("attempted_rps={:.1} (<98% target)", attempted_rps));
        }
    }
    if args.batch_mode != BatchMode::Http1 && total.drop_conn_queue_full > 0 {
        client.push(format!(
            "drop_conn_queue_full={}",
            total.drop_conn_queue_full
        ));
    }
    if args.batch_mode == BatchMode::Http1 && total.dropped_after_attempt > 0 {
        client.push(format!(
            "dropped_after_attempt={}",
            total.dropped_after_attempt
        ));
    }
    if args.batch_mode == BatchMode::Off
        && (queue_p99 >= 5_000 || queue_p99 >= target_period_us.saturating_mul(10))
    {
        let reason = match args.mode {
            BenchMode::Latency => "queue_delay_p99",
            BenchMode::Throughput => "batch_e2e_p99",
        };
        client.push(format!("{reason}={}us", queue_p99));
    }

    if total.timeout > 0 {
        server.push(format!("timeout={}", total.timeout));
    }
    if total.http_429 > 0 {
        server.push(format!("http_429={}", total.http_429));
    }
    if total.http_5xx > 0 {
        server.push(format!("http_5xx={}", total.http_5xx));
    }

    let verdict = if !client.is_empty() {
        Verdict::ClientLimited
    } else if !server.is_empty() {
        Verdict::ServerLimited
    } else {
        Verdict::Pass
    };
    (verdict, client, server)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let worker_cpus = parse_cpu_list(&args.worker_cpus)?;
    let workers = pick_workers(&args, &worker_cpus);
    let target = parse_target(&args.url)?;
    if args.batch_mode == BatchMode::Http1 {
        if target.transport != TransportKind::Http {
            bail!("--batch-mode http1 requires http:// target");
        }
        if args.mode != BenchMode::Throughput {
            bail!("--batch-mode http1 requires --mode throughput");
        }
        if args.workload != WorkloadMode::Corpus {
            bail!("--batch-mode http1 currently supports only --workload corpus");
        }
    }
    if args.mode == BenchMode::Throughput && target.transport != TransportKind::H2c {
        if args.batch_mode != BatchMode::Http1 {
            bail!("--mode throughput currently supports only h2c:// targets, or http:// with --batch-mode http1");
        }
    }
    if args.mode == BenchMode::Latency && args.workload != WorkloadMode::Corpus {
        bail!("--mode latency currently supports only --workload corpus");
    }
    let workload = load_workload_source(&args)?;
    let (corpus, route_meta) = match &workload {
        WorkloadSource::Corpus {
            payload,
            route_meta,
        } => (Some(payload.clone()), route_meta.clone()),
        WorkloadSource::Ceiling { .. } => (None, None),
    };
    let req_tpl = if target.transport == TransportKind::Http {
        let corpus = corpus
            .as_ref()
            .context("http transport requires corpus workload")?;
        Some(RequestTemplate::new(
            &target,
            if args.batch_mode == BatchMode::Http1 {
                16 + record_len(corpus, route_meta.as_ref()) * args.batch_records.max(1)
            } else {
                corpus.row_bytes + route_meta.as_ref().map(|_| 40).unwrap_or(0)
            },
        ))
    } else {
        None
    };
    let target_transport = target.transport;
    let payload_rows = workload.payload_rows();
    let workload_dense_dim = workload.dense_dim(args.dense_dim);
    let workload_route_meta = workload.route_meta_enabled();

    eprintln!(
        "[bench3] url={} transport={:?} mode={:?} rps={} warmup={}s duration={}s workers={} conns_per_worker={} max_inflight_per_conn={} payload_rows={} dim={} route_meta={} protocol={:?}",
        args.url,
        target.transport,
        args.mode,
        args.rps,
        args.warmup,
        args.duration,
        workers,
        args.conns_per_worker,
        args.max_inflight_per_conn,
        payload_rows,
        workload_dense_dim,
        workload_route_meta,
        args.protocol
    );

    let queue_depth_scale = match target_transport {
        TransportKind::Http => {
            if args.batch_mode == BatchMode::Http1 {
                8
            } else {
                1
            }
        }
        TransportKind::H2c => 8,
    };
    let per_worker_cap = args
        .conns_per_worker
        .saturating_mul(args.max_inflight_per_conn.max(1))
        .saturating_mul(queue_depth_scale)
        .max(1);
    let use_http_batch_pull =
        args.mode == BenchMode::Throughput && args.batch_mode == BatchMode::Http1;
    let (events_tx, events_rx) = crossbeam_channel::unbounded::<AggEvent>();
    let pacer_done = Arc::new(AtomicBool::new(false));
    let mut worker_queues = Vec::with_capacity(workers);
    let mut throughput_quotas = Vec::with_capacity(workers);
    let mut http_batch_stats = Vec::with_capacity(workers);
    let mut worker_handles = Vec::with_capacity(workers);

    for worker_id in 0..workers {
        let use_http_queue = args.mode == BenchMode::Latency || args.batch_mode == BatchMode::Http1;
        let queue = if use_http_queue {
            let queue = Arc::new(ArrayQueue::new(per_worker_cap.max(1)));
            worker_queues.push(queue.clone());
            Some(queue)
        } else {
            None
        };
        let throughput_quota =
            if args.mode == BenchMode::Throughput && args.batch_mode != BatchMode::Http1 {
                let quota = Arc::new(ThroughputQuota::default());
                throughput_quotas.push(quota.clone());
                Some(quota)
            } else {
                None
            };
        let batch_stats = if use_http_batch_pull {
            let stats = Arc::new(HttpBatchWorkerStats::default());
            http_batch_stats.push(stats.clone());
            Some(stats)
        } else {
            None
        };
        let args2 = args.clone();
        let target2 = target.clone();
        let corpus2 = corpus.clone();
        let route_meta2 = route_meta.clone();
        let workload2 = workload.clone();
        let req_tpl2 = req_tpl.clone();
        let tx2 = events_tx.clone();
        let done2 = pacer_done.clone();
        let cpu = worker_cpus
            .as_ref()
            .and_then(|cpus| cpus.get(worker_id).copied());
        let name = format!("bench3-w{worker_id}");
        let h = thread::Builder::new()
            .name(name)
            .spawn(move || {
                let res = match target2.transport {
                    TransportKind::Http => {
                        let corpus = corpus2
                            .clone()
                            .expect("http worker requires corpus workload");
                        if args2.batch_mode == BatchMode::Http1 {
                            worker_loop_http_batch(
                                worker_id,
                                cpu,
                                args2,
                                target2,
                                corpus,
                                route_meta2,
                                req_tpl2.expect("http batch worker requires request template"),
                                queue.expect("http batch worker requires queue ingress"),
                                batch_stats.expect("http batch worker requires shared stats"),
                                done2,
                            )
                        } else {
                            worker_loop_http(
                                worker_id,
                                cpu,
                                args2,
                                target2,
                                corpus,
                                route_meta2,
                                req_tpl2.expect("http worker requires request template"),
                                queue.expect("http worker requires queue ingress"),
                                tx2.clone(),
                                done2,
                            )
                        }
                    }
                    TransportKind::H2c => worker_loop_h2c(
                        worker_id,
                        workers,
                        cpu,
                        args2,
                        target2,
                        workload2,
                        queue.unwrap_or_else(|| Arc::new(ArrayQueue::new(1))),
                        throughput_quota,
                        tx2.clone(),
                        done2,
                    ),
                };
                let _ = tx2.send(AggEvent::WorkerDone);
                if let Err(ref e) = res {
                    eprintln!("[bench3] worker {worker_id} failed: {e:#}");
                }
                res
            })
            .context("spawn worker")?;
        worker_handles.push(h);
    }

    let start = Instant::now();
    let pacer_args = args.clone();
    let pacer_done2 = pacer_done.clone();
    let tx2 = events_tx.clone();
    let http_batch_stats_for_pacer = http_batch_stats.clone();
    let pacer_handle = thread::Builder::new()
        .name("bench3-pacer".to_string())
        .spawn(move || {
            let res = match pacer_args.mode {
                BenchMode::Latency => pacer_loop(
                    pacer_args,
                    worker_queues,
                    payload_rows,
                    tx2,
                    pacer_done2.clone(),
                ),
                BenchMode::Throughput => {
                    if pacer_args.batch_mode == BatchMode::Http1 {
                        if use_http_batch_pull {
                            pacer_loop_http_batch_soft(
                                pacer_args,
                                worker_queues,
                                http_batch_stats_for_pacer,
                                payload_rows,
                                pacer_done2.clone(),
                            )
                        } else {
                            pacer_loop(
                                pacer_args,
                                worker_queues,
                                payload_rows,
                                tx2,
                                pacer_done2.clone(),
                            )
                        }
                    } else {
                        pacer_loop_throughput(
                            pacer_args,
                            throughput_quotas,
                            tx2,
                            pacer_done2.clone(),
                        )
                    }
                }
            };
            pacer_done2.store(true, Ordering::Release);
            if let Err(ref e) = res {
                eprintln!("[bench3] pacer failed: {e:#}");
            }
            res
        })
        .context("spawn pacer")?;

    drop(events_tx);
    let total = if use_http_batch_pull {
        run_aggregator_http_batch_pull(&args, &http_batch_stats, start)?
    } else {
        run_aggregator(&args, events_rx, workers, start)?
    };

    pacer_handle
        .join()
        .map_err(|_| anyhow!("pacer thread panicked"))??;
    for h in worker_handles {
        h.join().map_err(|_| anyhow!("worker thread panicked"))??;
    }

    let attempted_rps = total.attempted as f64 / (args.duration as f64).max(1.0);
    let ok_rps = total.ok as f64 / (args.duration as f64).max(1.0);
    let offered = total.attempted + total.drop_conn_queue_full;
    let offered_rps = offered as f64 / (args.duration as f64).max(1.0);
    let offered_but_not_sent = total.drop_conn_queue_full;
    let offered_but_not_sent_rps = offered_but_not_sent as f64 / (args.duration as f64).max(1.0);
    let offered_but_not_sent_pct = if offered == 0 {
        0.0
    } else {
        offered_but_not_sent as f64 / offered as f64
    };
    let under_target_rps = (args.rps as f64 - attempted_rps).max(0.0);
    let under_target_pct = if args.rps == 0 {
        0.0
    } else {
        under_target_rps / args.rps as f64
    };
    let attempted_batch_rps = if args.batch_mode == BatchMode::Http1 {
        attempted_rps / (args.batch_records.max(1) as f64)
    } else {
        0.0
    };
    let ok_batch_rps = if args.batch_mode == BatchMode::Http1 {
        ok_rps / (args.batch_records.max(1) as f64)
    } else {
        0.0
    };
    let queue_q = quantiles(&total.queue_delay_us);
    let send_q = quantiles(&total.send_delay_us);
    let server_q = quantiles(&total.server_rtt_us);
    let e2e_q = quantiles(&total.e2e_us);
    let batch_q = quantiles(&total.batch_e2e_us);
    let stage = StageP99 {
        parse: hist_q(&total.stage_parse, 0.99),
        feature: hist_q(&total.stage_feature, 0.99),
        router: hist_q(&total.stage_router, 0.99),
        l1: hist_q(&total.stage_l1, 0.99),
        l2: hist_q(&total.stage_l2, 0.99),
        serialize: hist_q(&total.stage_serialize, 0.99),
    };
    let (verdict, client_reasons, server_reasons) = verdict_for(&args, &total);

    println!(
        "[bench3 rps={}] offered={} attempted={} ok={} err={} timeout={} dropped={} offered_rps={:.1} attempted_rps={:.1} ok_rps={:.1}",
        args.rps,
        offered,
        total.attempted,
        total.ok,
        total.err,
        total.timeout,
        total.dropped,
        offered_rps,
        attempted_rps,
        ok_rps
    );
    if args.batch_mode == BatchMode::Http1 {
        println!(
            "[bench3] batch_rps: attempted={:.1} ok={:.1} batch_records={}",
            attempted_batch_rps, ok_batch_rps, args.batch_records
        );
    }
    match args.mode {
        BenchMode::Latency => println!(
            "[bench3] latency_us: queue(p50={} p95={} p99={}) send(p50={} p95={} p99={}) server_rtt(p50={} p95={} p99={}) e2e(p50={} p95={} p99={})",
            queue_q.p50, queue_q.p95, queue_q.p99,
            send_q.p50, send_q.p95, send_q.p99,
            server_q.p50, server_q.p95, server_q.p99,
            e2e_q.p50, e2e_q.p95, e2e_q.p99
        ),
        BenchMode::Throughput => println!(
            "[bench3] batch_latency_us: e2e(p50={} p95={} p99={})",
            batch_q.p50, batch_q.p95, batch_q.p99
        ),
    }
    println!(
        "[bench3] status: 2xx={} 429={} 5xx={} timeout={} drop_inflight_cap={} drop_conn_queue_full={}",
        total.http_2xx, total.http_429, total.http_5xx, total.timeout, total.drop_inflight_cap, total.drop_conn_queue_full
    );
    if args.batch_mode == BatchMode::Http1 {
        println!(
            "[bench3] batch_drop: offered_but_not_sent={} dropped_after_attempt={}",
            total.drop_conn_queue_full, total.dropped_after_attempt
        );
    }
    println!(
        "[bench3] protocol: rsk1_samples={} qsb2_samples={} used_l2={} decisions={{allow:{}, deny:{}, manual_review:{}, degrade_allow:{}, unknown:{}}}",
        total.rsk1_samples,
        total.qsb2_samples,
        total.used_l2,
        total.decision_counts[decision_bucket(0)],
        total.decision_counts[decision_bucket(1)],
        total.decision_counts[decision_bucket(2)],
        total.decision_counts[decision_bucket(3)],
        total.decision_counts[decision_bucket(255)],
    );
    match args.mode {
        BenchMode::Latency => println!(
            "[bench3] stage_p99(us): parse={} feature={} router={} l1={} l2={} serialize={}",
            stage.parse, stage.feature, stage.router, stage.l1, stage.l2, stage.serialize
        ),
        BenchMode::Throughput => {
            println!("[bench3] stage_p99(us): client_disabled use_server_side_metrics=true")
        }
    }
    match verdict {
        Verdict::Pass => println!("[bench3] verdict: PASS"),
        Verdict::ClientLimited => println!(
            "[bench3] verdict: CLIENT_LIMITED reasons={}",
            client_reasons.join(", ")
        ),
        Verdict::ServerLimited => println!(
            "[bench3] verdict: SERVER_LIMITED reasons={}",
            server_reasons.join(", ")
        ),
    }

    if let Some(path) = args.summary_json.as_ref() {
        let summary = SummaryJson {
            config: SummaryConfig {
                url: args.url.clone(),
                rps: args.rps,
                duration_s: args.duration,
                warmup_s: args.warmup,
                workers,
                worker_cpus,
                pacer_cpu: args.pacer_cpu,
                conns_per_worker: args.conns_per_worker,
                max_inflight_per_conn: args.max_inflight_per_conn,
                raw_batch_size: 0,
                stats_sample_rate: 0,
                payload_rows,
                dense_dim: workload_dense_dim,
                route_meta: workload_route_meta,
                protocol: args.protocol,
                bench_mode: args.mode,
                batch_mode: args.batch_mode,
                batch_records: args.batch_records,
                workload: args.workload,
                transport: match target_transport {
                    TransportKind::Http => "http1_keepalive_manual",
                    TransportKind::H2c => "h2c_http2",
                },
                raw_version: None,
            },
            counts: SummaryCounts {
                offered,
                attempted: total.attempted,
                ok: total.ok,
                err: total.err,
                timeout: total.timeout,
                dropped: total.dropped,
                drop_inflight_cap: total.drop_inflight_cap,
                drop_conn_queue_full: total.drop_conn_queue_full,
                dropped_after_attempt: total.dropped_after_attempt,
                http_2xx: total.http_2xx,
                http_429: total.http_429,
                http_5xx: total.http_5xx,
                qsb2_samples: total.qsb2_samples,
                rsk1_samples: total.rsk1_samples,
                used_l2: total.used_l2,
                decision_allow: total.decision_counts[decision_bucket(0)],
                decision_deny: total.decision_counts[decision_bucket(1)],
                decision_manual_review: total.decision_counts[decision_bucket(2)],
                decision_degrade_allow: total.decision_counts[decision_bucket(3)],
                decision_unknown: total.decision_counts[decision_bucket(255)],
                attempted_rps,
                ok_rps,
                target_rps: args.rps,
                offered_but_not_sent,
                offered_but_not_sent_rps,
                offered_but_not_sent_pct,
                under_target_rps,
                under_target_pct,
                attempted_batch_rps,
                ok_batch_rps,
            },
            latency_us: SummaryLatency {
                queue_delay: match args.mode {
                    BenchMode::Latency => queue_q,
                    BenchMode::Throughput => LatQuantiles {
                        p50: 0,
                        p95: 0,
                        p99: 0,
                    },
                },
                send_delay: match args.mode {
                    BenchMode::Latency => send_q,
                    BenchMode::Throughput => LatQuantiles {
                        p50: 0,
                        p95: 0,
                        p99: 0,
                    },
                },
                server_rtt: match args.mode {
                    BenchMode::Latency => server_q,
                    BenchMode::Throughput => LatQuantiles {
                        p50: 0,
                        p95: 0,
                        p99: 0,
                    },
                },
                e2e: match args.mode {
                    BenchMode::Latency => e2e_q,
                    BenchMode::Throughput => LatQuantiles {
                        p50: 0,
                        p95: 0,
                        p99: 0,
                    },
                },
            },
            batch_latency_us: match args.mode {
                BenchMode::Latency => LatQuantiles {
                    p50: 0,
                    p95: 0,
                    p99: 0,
                },
                BenchMode::Throughput => batch_q,
            },
            stage_p99_us: match args.mode {
                BenchMode::Latency => stage,
                BenchMode::Throughput => StageP99 {
                    parse: 0,
                    feature: 0,
                    router: 0,
                    l1: 0,
                    l2: 0,
                    serialize: 0,
                },
            },
            verdict,
            client_limited_reasons: client_reasons,
            server_limited_reasons: server_reasons,
        };
        std::fs::write(path, serde_json::to_vec_pretty(&summary)?)
            .with_context(|| format!("write summary json: {}", path.display()))?;
    }

    Ok(())
}
