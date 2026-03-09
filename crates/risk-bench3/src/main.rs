use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use crossbeam_queue::ArrayQueue;
use hdrhistogram::Histogram;
use http::Uri;
use memmap2::Mmap;
use mio::net::TcpStream;
use mio::{Events, Interest, Poll, Token};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;
use std::cmp::min;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, IoSlice, Read, Write};
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::PathBuf;
use std::str;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

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
}

#[derive(Clone, Copy, Debug)]
struct QueuedReq {
    row_idx: usize,
    t_sched: Instant,
    t_issue: Instant,
    record: bool,
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

#[derive(Debug)]
enum AggEvent {
    AttemptBatch {
        attempted: u64,
        dropped_conn_queue_full: u64,
    },
    CompletionBatch {
        items: Vec<CompletedSample>,
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

#[derive(Clone)]
struct RequestTemplate {
    prefix: Arc<Vec<u8>>,
}

impl RequestTemplate {
    fn new(target: &Target, content_len: usize) -> Self {
        let mut req = Vec::with_capacity(256);
        req.extend_from_slice(b"POST ");
        req.extend_from_slice(target.path_and_query.as_bytes());
        req.extend_from_slice(b" HTTP/1.1\r\nHost: ");
        req.extend_from_slice(target.host_header.as_bytes());
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
    host_header: String,
    path_and_query: String,
    addr: SocketAddr,
}

fn parse_target(url: &str) -> Result<Target> {
    let uri: Uri = url
        .parse()
        .with_context(|| format!("invalid --url: {url}"))?;
    let scheme = uri.scheme_str().unwrap_or("http");
    if scheme != "http" {
        bail!("only http:// URLs are supported in bench3, got scheme={scheme}");
    }
    let host = uri.host().ok_or_else(|| anyhow!("url missing host"))?;
    let port = uri.port_u16().unwrap_or(80);
    let path_and_query = uri
        .path_and_query()
        .map(|v| v.as_str().to_string())
        .unwrap_or_else(|| "/".to_string());
    let host_header = if port == 80 {
        host.to_string()
    } else {
        format!("{host}:{port}")
    };
    let addr = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("resolve {host}:{port}"))?
        .next()
        .ok_or_else(|| anyhow!("resolve {host}:{port}: no addresses"))?;
    Ok(Target {
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

    fn try_finish(&mut self) -> io::Result<Option<(HttpResponseMeta, BodyDecoded)>> {
        if !self.headers_parsed {
            if let Some(hdr_end) = find_subsequence(&self.buf, b"\r\n\r\n") {
                self.header_len = hdr_end + 4;
                let header_bytes = &self.buf[..hdr_end];
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

struct StatsAgg {
    attempted: u64,
    ok: u64,
    err: u64,
    timeout: u64,
    dropped: u64,
    drop_inflight_cap: u64,
    drop_conn_queue_full: u64,
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
        })
    }

    fn record_attempts(&mut self, attempted: u64, dropped_conn_queue_full: u64) {
        self.attempted += attempted;
        self.dropped += dropped_conn_queue_full;
        self.drop_conn_queue_full += dropped_conn_queue_full;
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
    payload_rows: usize,
    dense_dim: usize,
    route_meta: bool,
    protocol: ProtocolMode,
    transport: &'static str,
}

#[derive(Serialize)]
struct SummaryCounts {
    attempted: u64,
    ok: u64,
    err: u64,
    timeout: u64,
    dropped: u64,
    drop_inflight_cap: u64,
    drop_conn_queue_full: u64,
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

fn decode_body(buf: &[u8]) -> io::Result<BodyDecoded> {
    if buf.len() == 24 && &buf[0..4] == b"QSB2" {
        return Ok(BodyDecoded::Qsb2(Qsb2 {
            decision: buf[6],
            flags: buf[7],
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
    let std_stream =
        std::net::TcpStream::connect(addr).with_context(|| format!("connect {addr}"))?;
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
        BodyDecoded::Qsb2(_) => meta.timings_header.unwrap_or_default(),
        BodyDecoded::Rsk1(r) => r.timings,
    };
    let (decision, used_l2, qsb2, rsk1) = match body {
        BodyDecoded::Qsb2(q) => (q.decision, (q.flags & 1) != 0, true, false),
        BodyDecoded::Rsk1(r) => (r.decision, (r.flags & 1) != 0, false, true),
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

fn worker_loop(
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
    let total_duration = Duration::from_secs(args.warmup + args.duration);
    let warmup_end = Instant::now() + Duration::from_secs(args.warmup);
    let end = Instant::now() + total_duration;
    let mut next_at = Instant::now();
    let lambda = args.rps.max(1) as f64 / 1_000_000.0;
    let fixed_us = (1_000_000.0 / args.rps.max(1) as f64).max(1.0) as u64;
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
        };
        req_id = req_id.wrapping_add(1);
        let queue = &worker_queues[rr % worker_queues.len()];
        rr = rr.wrapping_add(1);

        if record {
            attempted_batch += 1;
        }
        if queue.push(tok).is_err() && record {
            dropped_batch += 1;
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

fn write_window_csv_header(w: &mut dyn Write) -> Result<()> {
    writeln!(
        w,
        "t_ms,attempted,ok,err,timeout,dropped,drop_inflight_cap,drop_conn_queue_full,http_2xx,http_429,http_5xx,queue_p99_us,send_p99_us,server_rtt_p99_us,e2e_p99_us,stage_router_p99_us,stage_l1_p99_us,stage_l2_p99_us"
    )?;
    Ok(())
}

fn write_window_row(w: &mut dyn Write, t_ms: u128, agg: &StatsAgg) -> Result<()> {
    writeln!(
        w,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
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
        hist_q(&agg.stage_router, 0.99),
        hist_q(&agg.stage_l1, 0.99),
        hist_q(&agg.stage_l2, 0.99)
    )?;
    Ok(())
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
    let target_period_us = ((1_000_000.0) / args.rps.max(1) as f64).max(1.0) as u64;
    let attempted_rps = total.attempted as f64 / (args.duration as f64).max(1.0);
    let queue_p99 = hist_q(&total.queue_delay_us, 0.99);

    let mut client = Vec::new();
    let mut server = Vec::new();

    if total.drop_inflight_cap > 0 {
        client.push(format!("drop_inflight_cap={}", total.drop_inflight_cap));
    }
    if total.drop_conn_queue_full > 0 {
        client.push(format!(
            "drop_conn_queue_full={}",
            total.drop_conn_queue_full
        ));
    }
    if attempted_rps < (args.rps as f64) * 0.98 {
        client.push(format!("attempted_rps={:.1} (<98% target)", attempted_rps));
    }
    if queue_p99 >= 5_000 || queue_p99 >= target_period_us.saturating_mul(10) {
        client.push(format!("queue_delay_p99={}us", queue_p99));
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
    let corpus = PayloadCorpus::load(&args.dense_file, args.dense_dim)?;
    let route_meta = if let Some(path) = args.route_meta_tsv.as_ref() {
        Some(RouteMetaCorpus::load(path, corpus.rows)?)
    } else {
        None
    };
    let req_tpl = RequestTemplate::new(
        &target,
        corpus.row_bytes + route_meta.as_ref().map(|_| 40).unwrap_or(0),
    );

    eprintln!(
        "[bench3] url={} rps={} warmup={}s duration={}s workers={} conns_per_worker={} max_inflight_per_conn={} payload_rows={} dim={} route_meta={} protocol={:?}",
        args.url,
        args.rps,
        args.warmup,
        args.duration,
        workers,
        args.conns_per_worker,
        args.max_inflight_per_conn,
        corpus.rows,
        args.dense_dim,
        route_meta.is_some(),
        args.protocol
    );

    let per_worker_cap = args
        .conns_per_worker
        .saturating_mul(args.max_inflight_per_conn.max(1));
    let (events_tx, events_rx) = crossbeam_channel::unbounded::<AggEvent>();
    let pacer_done = Arc::new(AtomicBool::new(false));
    let mut worker_queues = Vec::with_capacity(workers);
    let mut worker_handles = Vec::with_capacity(workers);

    for worker_id in 0..workers {
        let queue = Arc::new(ArrayQueue::new(per_worker_cap.max(1)));
        worker_queues.push(queue.clone());
        let args2 = args.clone();
        let target2 = target.clone();
        let corpus2 = corpus.clone();
        let route_meta2 = route_meta.clone();
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
                let res = worker_loop(
                    worker_id,
                    cpu,
                    args2,
                    target2,
                    corpus2,
                    route_meta2,
                    req_tpl2,
                    queue,
                    tx2.clone(),
                    done2,
                );
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
    let pacer_handle = thread::Builder::new()
        .name("bench3-pacer".to_string())
        .spawn(move || {
            let res = pacer_loop(
                pacer_args,
                worker_queues,
                corpus.rows,
                tx2,
                pacer_done2.clone(),
            );
            pacer_done2.store(true, Ordering::Release);
            if let Err(ref e) = res {
                eprintln!("[bench3] pacer failed: {e:#}");
            }
            res
        })
        .context("spawn pacer")?;

    drop(events_tx);
    let total = run_aggregator(&args, events_rx, workers, start)?;

    pacer_handle
        .join()
        .map_err(|_| anyhow!("pacer thread panicked"))??;
    for h in worker_handles {
        h.join().map_err(|_| anyhow!("worker thread panicked"))??;
    }

    let attempted_rps = total.attempted as f64 / (args.duration as f64).max(1.0);
    let ok_rps = total.ok as f64 / (args.duration as f64).max(1.0);
    let queue_q = quantiles(&total.queue_delay_us);
    let send_q = quantiles(&total.send_delay_us);
    let server_q = quantiles(&total.server_rtt_us);
    let e2e_q = quantiles(&total.e2e_us);
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
        "[bench3 rps={}] attempted={} ok={} err={} timeout={} dropped={} attempted_rps={:.1} ok_rps={:.1}",
        args.rps, total.attempted, total.ok, total.err, total.timeout, total.dropped, attempted_rps, ok_rps
    );
    println!(
        "[bench3] latency_us: queue(p50={} p95={} p99={}) send(p50={} p95={} p99={}) server_rtt(p50={} p95={} p99={}) e2e(p50={} p95={} p99={})",
        queue_q.p50, queue_q.p95, queue_q.p99,
        send_q.p50, send_q.p95, send_q.p99,
        server_q.p50, server_q.p95, server_q.p99,
        e2e_q.p50, e2e_q.p95, e2e_q.p99
    );
    println!(
        "[bench3] http: 2xx={} 429={} 5xx={} timeout={} drop_inflight_cap={} drop_conn_queue_full={}",
        total.http_2xx, total.http_429, total.http_5xx, total.timeout, total.drop_inflight_cap, total.drop_conn_queue_full
    );
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
    println!(
        "[bench3] stage_p99(us): parse={} feature={} router={} l1={} l2={} serialize={}",
        stage.parse, stage.feature, stage.router, stage.l1, stage.l2, stage.serialize
    );
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
                payload_rows: corpus.rows,
                dense_dim: args.dense_dim,
                route_meta: route_meta.is_some(),
                protocol: args.protocol,
                transport: "http1_keepalive_manual",
            },
            counts: SummaryCounts {
                attempted: total.attempted,
                ok: total.ok,
                err: total.err,
                timeout: total.timeout,
                dropped: total.dropped,
                drop_inflight_cap: total.drop_inflight_cap,
                drop_conn_queue_full: total.drop_conn_queue_full,
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
            },
            latency_us: SummaryLatency {
                queue_delay: queue_q,
                send_delay: send_q,
                server_rtt: server_q,
                e2e: e2e_q,
            },
            stage_p99_us: stage,
            verdict,
            client_limited_reasons: client_reasons,
            server_limited_reasons: server_reasons,
        };
        std::fs::write(path, serde_json::to_vec_pretty(&summary)?)
            .with_context(|| format!("write summary json: {}", path.display()))?;
    }

    Ok(())
}
