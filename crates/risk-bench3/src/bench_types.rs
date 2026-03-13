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

