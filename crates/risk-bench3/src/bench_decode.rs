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

