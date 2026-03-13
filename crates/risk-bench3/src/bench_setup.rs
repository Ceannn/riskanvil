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
        if let Some(done) = self.try_finish()? {
            return Ok(Some(done));
        }
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
        if let Some(done) = self.try_finish_batch_ack()? {
            return Ok(Some(done));
        }
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
            self.consume_parsed(self.header_len + self.content_len);
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
                self.consume_parsed(self.header_len + self.content_len);
                return Ok(Some((meta, ack)));
            }
            match decode_body(body)? {
                BodyDecoded::BatchAck(ack) => {
                    self.consume_parsed(self.header_len + self.content_len);
                    Ok(Some((meta, ack)))
                }
                _ => Err(invalid_data("expected batch ack response")),
            }
        } else {
            Ok(None)
        }
    }

    fn consume_parsed(&mut self, parsed_len: usize) {
        if parsed_len >= self.buf.len() {
            self.buf.clear();
        } else {
            let remain = self.buf.len() - parsed_len;
            self.buf.copy_within(parsed_len.., 0);
            self.buf.truncate(remain);
        }
        self.header_len = 0;
        self.content_len = 0;
        self.status_code = 0;
        self.timings_header = None;
        self.headers_parsed = false;
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
    writing: Option<BatchActiveReq>,
    inflight: VecDeque<BatchActiveReq>,
    write_state: BatchWriteState,
    parser: ResponseParser,
}

impl BatchConn {
    fn load(&self) -> usize {
        self.pending.len() + self.inflight.len() + usize::from(self.writing.is_some())
    }

    fn wants_write(&self) -> bool {
        self.writing.is_some() || !self.pending.is_empty()
    }

    fn oldest_deadline(&self) -> Option<Instant> {
        self.inflight
            .front()
            .map(|active| active.deadline)
            .or_else(|| self.writing.as_ref().map(|active| active.deadline))
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

