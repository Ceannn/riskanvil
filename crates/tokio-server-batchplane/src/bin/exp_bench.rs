use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Serialize;
use std::collections::VecDeque;
use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

const RBH1_MAGIC: &[u8; 4] = b"RBH1";
const RBA1_MAGIC: &[u8; 4] = b"RBA1";
const RVEC_MAGIC: &[u8; 4] = b"RVEC";
const RVEC_VERSION: u32 = 3;
const RBH1_VERSION: u16 = 1;
const BATCH_HEADER_LEN: usize = 16;
const ROUTE_HEADER_LEN: usize = 40;
const ACK_LEN: usize = 40;
const ROW_BYTES_PER_VALUE: usize = 4;
const DECISION_BUCKETS: usize = 5;

#[derive(Debug, Parser, Clone)]
#[command(name = "exp_bench")]
#[command(about = "Experimental Unix-socket batch128 throughput bench for tokio-server-batchplane")]
struct Args {
    #[arg(long)]
    socket_path: PathBuf,

    #[arg(long)]
    dense_file: PathBuf,

    #[arg(long)]
    route_meta_tsv: PathBuf,

    #[arg(long)]
    dense_dim: usize,

    #[arg(long, default_value_t = 300_000)]
    rps: u64,

    #[arg(long, default_value_t = 5)]
    duration: u64,

    #[arg(long, default_value_t = 2)]
    warmup: u64,

    #[arg(long, default_value_t = 1)]
    workers: usize,

    #[arg(long, default_value_t = 1)]
    conns_per_worker: usize,

    #[arg(long, default_value_t = 128)]
    max_inflight_per_conn: usize,

    #[arg(long, default_value_t = 128)]
    batch_records: usize,

    #[arg(long, default_value_t = 2_000)]
    timeout_ms: u64,

    #[arg(long)]
    summary_json: Option<PathBuf>,
}

#[derive(Clone)]
struct PayloadCorpus {
    row_bytes: usize,
    row_count: usize,
    data: Arc<Vec<u8>>,
}

impl PayloadCorpus {
    fn load(path: &Path, dense_dim: usize) -> Result<Self> {
        if dense_dim == 0 {
            bail!("--dense-dim must be > 0");
        }
        let data = fs::read(path)
            .with_context(|| format!("failed to read dense corpus {}", path.display()))?;
        let row_bytes = dense_dim
            .checked_mul(ROW_BYTES_PER_VALUE)
            .context("dense_dim overflow while computing row bytes")?;
        if row_bytes == 0 {
            bail!("row bytes must be > 0");
        }
        if data.len() % row_bytes != 0 {
            bail!(
                "dense corpus size {} is not a multiple of row_bytes {}",
                data.len(),
                row_bytes
            );
        }
        let row_count = data.len() / row_bytes;
        if row_count == 0 {
            bail!("dense corpus has no rows");
        }
        Ok(Self {
            row_bytes,
            row_count,
            data: Arc::new(data),
        })
    }

    fn row_bytes(&self, row_idx: usize) -> &[u8] {
        let start = row_idx * self.row_bytes;
        &self.data[start..start + self.row_bytes]
    }
}

#[derive(Clone)]
struct RouteMetaEntry {
    transaction_id: u64,
    fold_id: i32,
    seg_prod_amtbin: u32,
    row_idx: u32,
    l2_tau_used: f32,
}

#[derive(Clone)]
struct RouteMetaCorpus {
    entries: Arc<Vec<RouteMetaEntry>>,
}

impl RouteMetaCorpus {
    fn load(path: &Path, expected_rows: usize) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read route meta tsv {}", path.display()))?;
        let mut lines = text.lines();
        let header = lines.next().context("route meta tsv is empty")?;
        let columns: Vec<&str> = header.split('\t').collect();
        let row_idx_col = find_col(&columns, "row_idx")?;
        let txn_col = find_col(&columns, "TransactionID")?;
        let fold_col = find_col(&columns, "fold_id")?;
        let seg_col = find_col(&columns, "seg_prod_amtbin")?;
        let tau_col = find_col(&columns, "l2_tau_used")?;

        let mut indexed = vec![None; expected_rows];
        for (line_no, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            let row_idx: usize = fields
                .get(row_idx_col)
                .context("missing row_idx column")?
                .parse()
                .with_context(|| format!("invalid row_idx at TSV line {}", line_no + 2))?;
            if row_idx >= expected_rows {
                bail!(
                    "route meta row_idx {} out of range for dense corpus rows {}",
                    row_idx,
                    expected_rows
                );
            }
            let entry = RouteMetaEntry {
                transaction_id: fields
                    .get(txn_col)
                    .context("missing TransactionID column")?
                    .parse()
                    .with_context(|| {
                        format!("invalid TransactionID at TSV line {}", line_no + 2)
                    })?,
                fold_id: fields
                    .get(fold_col)
                    .context("missing fold_id column")?
                    .parse()
                    .with_context(|| format!("invalid fold_id at TSV line {}", line_no + 2))?,
                seg_prod_amtbin: fields
                    .get(seg_col)
                    .context("missing seg_prod_amtbin column")?
                    .parse()
                    .with_context(|| {
                        format!("invalid seg_prod_amtbin at TSV line {}", line_no + 2)
                    })?,
                l2_tau_used: fields
                    .get(tau_col)
                    .context("missing l2_tau_used column")?
                    .parse()
                    .with_context(|| format!("invalid l2_tau_used at TSV line {}", line_no + 2))?,
                row_idx: row_idx as u32,
            };
            indexed[row_idx] = Some(entry);
        }

        let mut entries = Vec::with_capacity(expected_rows);
        for (row_idx, entry) in indexed.into_iter().enumerate() {
            entries.push(
                entry.with_context(|| format!("missing route meta for row_idx {}", row_idx))?,
            );
        }
        Ok(Self {
            entries: Arc::new(entries),
        })
    }

    fn entry(&self, row_idx: usize) -> &RouteMetaEntry {
        &self.entries[row_idx]
    }
}

#[derive(Default, Clone, Copy)]
struct ConnStats {
    offered_rows: u64,
    attempted_rows: u64,
    ok_rows: u64,
    timeout_rows: u64,
    dropped_after_attempt_rows: u64,
    used_l2: u64,
    decision_counts: [u64; DECISION_BUCKETS],
}

impl ConnStats {
    fn merge(&mut self, other: &Self) {
        self.offered_rows += other.offered_rows;
        self.attempted_rows += other.attempted_rows;
        self.ok_rows += other.ok_rows;
        self.timeout_rows += other.timeout_rows;
        self.dropped_after_attempt_rows += other.dropped_after_attempt_rows;
        self.used_l2 += other.used_l2;
        for (dst, src) in self
            .decision_counts
            .iter_mut()
            .zip(other.decision_counts.iter())
        {
            *dst += *src;
        }
    }
}

#[derive(Default)]
struct LatencyStats {
    samples_micros: Vec<u64>,
}

impl LatencyStats {
    fn record(&mut self, latency: Duration) {
        self.samples_micros
            .push(latency.as_micros().min(u128::from(u64::MAX)) as u64);
    }

    fn merge(&mut self, mut other: Self) {
        self.samples_micros.append(&mut other.samples_micros);
    }

    fn quantiles(&mut self) -> Quantiles {
        self.samples_micros.sort_unstable();
        Quantiles {
            p50: pick_quantile(&self.samples_micros, 0.50),
            p95: pick_quantile(&self.samples_micros, 0.95),
            p99: pick_quantile(&self.samples_micros, 0.99),
        }
    }
}

#[derive(Default, Serialize)]
struct Quantiles {
    p50: u64,
    p95: u64,
    p99: u64,
}

struct InflightBatch {
    sent_at: Instant,
    deadline: Instant,
    rows: u64,
}

#[derive(Debug, Clone, Copy)]
struct BatchAck {
    ok_count: u32,
    used_l2_count: u32,
    decision_counts: [u32; DECISION_BUCKETS],
}

#[derive(Serialize)]
struct SummaryJson {
    config: SummaryConfig,
    counts: SummaryCounts,
    batch_latency_us: Quantiles,
}

#[derive(Serialize)]
struct SummaryConfig {
    socket_path: String,
    dense_file: String,
    route_meta_tsv: String,
    dense_dim: usize,
    rps: u64,
    warmup_sec: u64,
    duration_sec: u64,
    workers: usize,
    conns_per_worker: usize,
    max_inflight_per_conn: usize,
    batch_records: usize,
    timeout_ms: u64,
}

#[derive(Serialize)]
struct SummaryCounts {
    offered_rows: u64,
    attempted_rows: u64,
    ok_rows: u64,
    timeout_rows: u64,
    dropped_after_attempt_rows: u64,
    used_l2: u64,
    decision_counts: [u64; DECISION_BUCKETS],
    offered_rps: f64,
    attempted_rps: f64,
    ok_rps: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.batch_records != 128 {
        bail!(
            "exp_bench only supports --batch-records 128 right now, got {}",
            args.batch_records
        );
    }
    if args.workers == 0 {
        bail!("--workers must be > 0");
    }
    if args.conns_per_worker == 0 {
        bail!("--conns-per-worker must be > 0");
    }
    if args.max_inflight_per_conn == 0 {
        bail!("--max-inflight-per-conn must be > 0");
    }

    let payloads = Arc::new(PayloadCorpus::load(&args.dense_file, args.dense_dim)?);
    let route_meta = Arc::new(RouteMetaCorpus::load(
        &args.route_meta_tsv,
        payloads.row_count,
    )?);
    let total_conns = args
        .workers
        .checked_mul(args.conns_per_worker)
        .context("workers * conns_per_worker overflow")?;
    let total_batch_rps = args.rps as f64 / args.batch_records as f64;
    let per_conn_batch_rps = total_batch_rps / total_conns as f64;
    if per_conn_batch_rps <= 0.0 {
        bail!("computed per-connection batch rps must be > 0");
    }

    let send_interval = Duration::from_secs_f64(1.0 / per_conn_batch_rps);
    let warmup = Duration::from_secs(args.warmup);
    let bench = Duration::from_secs(args.duration);
    let timeout = Duration::from_millis(args.timeout_ms);
    let start_at = Instant::now() + Duration::from_millis(250);
    let warmup_end = start_at + warmup;
    let stop_send_at = warmup_end + bench;

    let mut handles = Vec::with_capacity(total_conns);
    for conn_idx in 0..total_conns {
        let socket_path = args.socket_path.clone();
        let payloads = Arc::clone(&payloads);
        let route_meta = Arc::clone(&route_meta);
        let max_inflight = args.max_inflight_per_conn;
        let batch_records = args.batch_records;
        let worker_id = conn_idx / args.conns_per_worker;
        handles.push(thread::spawn(move || {
            run_conn(
                conn_idx,
                worker_id,
                &socket_path,
                payloads,
                route_meta,
                batch_records,
                max_inflight,
                start_at,
                warmup_end,
                stop_send_at,
                timeout,
                send_interval,
            )
        }));
    }

    let mut totals = ConnStats::default();
    let mut latencies = LatencyStats::default();
    for handle in handles {
        let (stats, latency_stats) = handle
            .join()
            .map_err(|_| anyhow::anyhow!("exp_bench worker panicked"))??;
        totals.merge(&stats);
        latencies.merge(latency_stats);
    }

    let measurement_secs = bench.as_secs_f64();
    let quantiles = latencies.quantiles();
    let offered_rps = totals.offered_rows as f64 / measurement_secs;
    let attempted_rps = totals.attempted_rows as f64 / measurement_secs;
    let ok_rps = totals.ok_rows as f64 / measurement_secs;
    let attempted_batch_rps = attempted_rps / args.batch_records as f64;
    let ok_batch_rps = ok_rps / args.batch_records as f64;

    println!(
        "[exp_bench rps={}] offered={} attempted={} ok={} timeout={} dropped_after_attempt={} offered_rps={:.1} attempted_rps={:.1} ok_rps={:.1}",
        args.rps,
        totals.offered_rows,
        totals.attempted_rows,
        totals.ok_rows,
        totals.timeout_rows,
        totals.dropped_after_attempt_rows,
        offered_rps,
        attempted_rps,
        ok_rps
    );
    println!(
        "[exp_bench] batch_rps: attempted={:.1} ok={:.1} batch_records={}",
        attempted_batch_rps, ok_batch_rps, args.batch_records
    );
    println!(
        "[exp_bench] batch_latency_us: e2e(p50={} p95={} p99={})",
        quantiles.p50, quantiles.p95, quantiles.p99
    );
    println!(
        "[exp_bench] protocol: used_l2={} decisions={{allow:{}, deny:{}, review:{}, degrade_allow:{}, unknown:{}}}",
        totals.used_l2,
        totals.decision_counts[0],
        totals.decision_counts[1],
        totals.decision_counts[2],
        totals.decision_counts[3],
        totals.decision_counts[4],
    );

    if let Some(path) = &args.summary_json {
        let summary = SummaryJson {
            config: SummaryConfig {
                socket_path: args.socket_path.display().to_string(),
                dense_file: args.dense_file.display().to_string(),
                route_meta_tsv: args.route_meta_tsv.display().to_string(),
                dense_dim: args.dense_dim,
                rps: args.rps,
                warmup_sec: args.warmup,
                duration_sec: args.duration,
                workers: args.workers,
                conns_per_worker: args.conns_per_worker,
                max_inflight_per_conn: args.max_inflight_per_conn,
                batch_records: args.batch_records,
                timeout_ms: args.timeout_ms,
            },
            counts: SummaryCounts {
                offered_rows: totals.offered_rows,
                attempted_rows: totals.attempted_rows,
                ok_rows: totals.ok_rows,
                timeout_rows: totals.timeout_rows,
                dropped_after_attempt_rows: totals.dropped_after_attempt_rows,
                used_l2: totals.used_l2,
                decision_counts: totals.decision_counts,
                offered_rps,
                attempted_rps,
                ok_rps,
            },
            batch_latency_us: quantiles,
        };
        let json =
            serde_json::to_vec_pretty(&summary).context("failed to serialize summary json")?;
        fs::write(path, json)
            .with_context(|| format!("failed to write summary json {}", path.display()))?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_conn(
    conn_idx: usize,
    worker_id: usize,
    socket_path: &Path,
    payloads: Arc<PayloadCorpus>,
    route_meta: Arc<RouteMetaCorpus>,
    batch_records: usize,
    max_inflight: usize,
    start_at: Instant,
    warmup_end: Instant,
    stop_send_at: Instant,
    timeout: Duration,
    send_interval: Duration,
) -> Result<(ConnStats, LatencyStats)> {
    let mut stream = UnixStream::connect(socket_path)
        .with_context(|| format!("failed to connect unix socket {}", socket_path.display()))?;
    stream
        .set_nonblocking(true)
        .context("failed to set unix socket nonblocking")?;

    let mut seq: u64 = conn_idx as u64 * 1_000_000;
    let mut next_send = start_at;
    let mut inflight = VecDeque::with_capacity(max_inflight + 1);
    let mut read_buf = Vec::with_capacity(ACK_LEN * 8);
    let mut stats = ConnStats::default();
    let mut latencies = LatencyStats::default();

    loop {
        let now = Instant::now();
        if now >= next_send && now < stop_send_at {
            if inflight.len() < max_inflight {
                let record_indices =
                    build_record_indices(seq, worker_id, payloads.row_count, batch_records);
                if now >= warmup_end {
                    stats.offered_rows += batch_records as u64;
                }
                let req = build_batch_body(&payloads, &route_meta, &record_indices)?;
                write_all_nonblocking(&mut stream, &req)?;
                if now >= warmup_end {
                    stats.attempted_rows += batch_records as u64;
                    inflight.push_back(InflightBatch {
                        sent_at: now,
                        deadline: now + timeout,
                        rows: batch_records as u64,
                    });
                }
                seq += 1;
                next_send += send_interval;
                continue;
            }
        }

        read_available(&mut stream, &mut read_buf)?;
        while read_buf.len() >= ACK_LEN {
            let ack = decode_batch_ack(&read_buf[..ACK_LEN])?;
            read_buf.drain(..ACK_LEN);
            if let Some(batch) = inflight.pop_front() {
                stats.ok_rows += u64::from(ack.ok_count);
                stats.used_l2 += u64::from(ack.used_l2_count);
                for (dst, src) in stats
                    .decision_counts
                    .iter_mut()
                    .zip(ack.decision_counts.iter())
                {
                    *dst += u64::from(*src);
                }
                latencies.record(batch.sent_at.elapsed());
            }
        }

        let now = Instant::now();
        while let Some(front) = inflight.front() {
            if front.deadline > now {
                break;
            }
            let expired = inflight.pop_front().expect("front existed");
            stats.timeout_rows += expired.rows;
        }

        if now >= stop_send_at && inflight.is_empty() {
            break;
        }
        if now >= stop_send_at {
            let all_expired = inflight
                .front()
                .is_none_or(|entry| entry.deadline <= Instant::now());
            if all_expired {
                break;
            }
        }

        thread::sleep(Duration::from_micros(50));
    }

    for pending in inflight {
        stats.dropped_after_attempt_rows += pending.rows;
    }
    Ok((stats, latencies))
}

fn write_all_nonblocking(stream: &mut UnixStream, buf: &[u8]) -> Result<()> {
    let mut offset = 0;
    while offset < buf.len() {
        match stream.write(&buf[offset..]) {
            Ok(0) => bail!("unix socket closed while writing"),
            Ok(written) => offset += written,
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_micros(50));
            }
            Err(err) => return Err(err).context("unix socket write failed"),
        }
    }
    Ok(())
}

fn read_available(stream: &mut UnixStream, read_buf: &mut Vec<u8>) -> Result<()> {
    let mut scratch = [0u8; 4096];
    loop {
        match stream.read(&mut scratch) {
            Ok(0) => break,
            Ok(read) => {
                read_buf.extend_from_slice(&scratch[..read]);
                if read < scratch.len() {
                    break;
                }
            }
            Err(err) if err.kind() == ErrorKind::WouldBlock => break,
            Err(err) => return Err(err).context("unix socket read failed"),
        }
    }
    Ok(())
}

fn build_record_indices(
    seq: u64,
    worker_id: usize,
    row_count: usize,
    batch_records: usize,
) -> Vec<usize> {
    let mut indices = Vec::with_capacity(batch_records);
    let base = seq.wrapping_mul(batch_records as u64) ^ worker_id as u64;
    for i in 0..batch_records {
        indices.push((base as usize + i) % row_count);
    }
    indices
}

fn build_batch_body(
    payloads: &PayloadCorpus,
    route_meta: &RouteMetaCorpus,
    record_indices: &[usize],
) -> Result<Vec<u8>> {
    let record_count = record_indices.len();
    let record_bytes = ROUTE_HEADER_LEN
        .checked_add(payloads.row_bytes)
        .context("record bytes overflow")?;
    let total_body_len = BATCH_HEADER_LEN
        .checked_add(
            record_count
                .checked_mul(record_bytes)
                .context("body size overflow")?,
        )
        .context("body size overflow")?;
    let mut body = Vec::with_capacity(total_body_len);
    body.extend_from_slice(&encode_batch_header(
        record_count as u32,
        record_bytes as u32,
        true,
    ));
    let mut route_header = [0u8; ROUTE_HEADER_LEN];
    for &row_idx in record_indices {
        let meta = route_meta.entry(row_idx);
        encode_rvec_v3_route_header(&mut route_header, meta, payloads.row_bytes / 4);
        body.extend_from_slice(&route_header);
        body.extend_from_slice(payloads.row_bytes(row_idx));
    }
    Ok(body)
}

fn encode_batch_header(record_count: u32, record_bytes: u32, has_route_meta: bool) -> [u8; 16] {
    let mut header = [0u8; BATCH_HEADER_LEN];
    header[0..4].copy_from_slice(RBH1_MAGIC);
    header[4..6].copy_from_slice(&RBH1_VERSION.to_le_bytes());
    header[6..8].copy_from_slice(&(if has_route_meta { 1u16 } else { 0u16 }).to_le_bytes());
    header[8..12].copy_from_slice(&record_count.to_le_bytes());
    header[12..16].copy_from_slice(&record_bytes.to_le_bytes());
    header
}

fn encode_rvec_v3_route_header(
    dst: &mut [u8; ROUTE_HEADER_LEN],
    meta: &RouteMetaEntry,
    dim: usize,
) {
    dst[0..4].copy_from_slice(RVEC_MAGIC);
    dst[4..6].copy_from_slice(&(RVEC_VERSION as u16).to_le_bytes());
    dst[6..8].copy_from_slice(&0u16.to_le_bytes());
    dst[8..12].copy_from_slice(&(dim as u32).to_le_bytes());
    dst[12..16].copy_from_slice(&meta.fold_id.to_le_bytes());
    dst[16..20].copy_from_slice(&meta.seg_prod_amtbin.to_le_bytes());
    dst[20..28].copy_from_slice(&meta.transaction_id.to_le_bytes());
    dst[28..32].copy_from_slice(&meta.row_idx.to_le_bytes());
    dst[32..36].copy_from_slice(&meta.l2_tau_used.to_le_bytes());
    dst[36..40].fill(0);
}

fn decode_batch_ack(bytes: &[u8]) -> Result<BatchAck> {
    if bytes.len() != ACK_LEN {
        bail!(
            "expected {} bytes for RBA1 ack, got {}",
            ACK_LEN,
            bytes.len()
        );
    }
    if &bytes[0..4] != RBA1_MAGIC {
        bail!("invalid RBA1 magic");
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != 1 {
        bail!("unsupported RBA1 version {}", version);
    }
    let body_len = u16::from_le_bytes([bytes[6], bytes[7]]);
    if body_len != 0 {
        bail!("unexpected RBA1 body length {}", body_len);
    }
    let _record_count = u32::from_le_bytes(bytes[8..12].try_into().expect("slice len checked"));
    let ok_count = u32::from_le_bytes(bytes[12..16].try_into().expect("slice len checked"));
    let used_l2_count = u32::from_le_bytes(bytes[16..20].try_into().expect("slice len checked"));
    let mut decision_counts = [0u32; DECISION_BUCKETS];
    for (idx, slot) in decision_counts.iter_mut().enumerate() {
        let start = 20 + idx * 4;
        *slot = u32::from_le_bytes(
            bytes[start..start + 4]
                .try_into()
                .expect("slice len checked"),
        );
    }
    Ok(BatchAck {
        ok_count,
        used_l2_count,
        decision_counts,
    })
}

fn find_col(columns: &[&str], name: &str) -> Result<usize> {
    columns
        .iter()
        .position(|col| *col == name)
        .with_context(|| format!("missing {} column in route meta tsv", name))
}

fn pick_quantile(sorted: &[u64], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let clamped = q.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f64 * clamped).round() as usize;
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_header_round_trip_shape() {
        let header = encode_batch_header(128, 552, true);
        assert_eq!(&header[0..4], RBH1_MAGIC);
        assert_eq!(u16::from_le_bytes([header[4], header[5]]), RBH1_VERSION);
        assert_eq!(u16::from_le_bytes([header[6], header[7]]), 1);
        assert_eq!(u32::from_le_bytes(header[8..12].try_into().unwrap()), 128);
        assert_eq!(u32::from_le_bytes(header[12..16].try_into().unwrap()), 552);
    }

    #[test]
    fn ack_decode_reads_all_buckets() {
        let mut ack = [0u8; ACK_LEN];
        ack[0..4].copy_from_slice(RBA1_MAGIC);
        ack[4..6].copy_from_slice(&1u16.to_le_bytes());
        ack[6..8].copy_from_slice(&0u16.to_le_bytes());
        ack[8..12].copy_from_slice(&128u32.to_le_bytes());
        ack[12..16].copy_from_slice(&128u32.to_le_bytes());
        ack[16..20].copy_from_slice(&33u32.to_le_bytes());
        for i in 0..DECISION_BUCKETS {
            let start = 20 + i * 4;
            ack[start..start + 4].copy_from_slice(&(10 + i as u32).to_le_bytes());
        }
        let decoded = decode_batch_ack(&ack).unwrap();
        assert_eq!(decoded.ok_count, 128);
        assert_eq!(decoded.used_l2_count, 33);
        assert_eq!(decoded.decision_counts, [10, 11, 12, 13, 14]);
    }

    #[test]
    fn quantiles_pick_expected_elements() {
        let mut values = vec![5, 2, 9, 1, 7];
        values.sort_unstable();
        assert_eq!(pick_quantile(&values, 0.0), 1);
        assert_eq!(pick_quantile(&values, 0.5), 5);
        assert_eq!(pick_quantile(&values, 1.0), 9);
    }
}
