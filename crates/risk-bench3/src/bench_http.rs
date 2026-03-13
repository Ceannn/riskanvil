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
    conn.writing = None;
    conn.inflight.clear();
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
    let Some(active) = conn.writing.as_ref() else {
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

fn prime_next_write_batch(
    conn: &mut BatchConn,
    timeout: Duration,
    corpus: &PayloadCorpus,
    route_meta: Option<&RouteMetaCorpus>,
    batch_records: usize,
) {
    if conn.writing.is_some() {
        return;
    }
    let Some(next) = conn.pending.pop_front() else {
        return;
    };
    let rows = next.rows.max(1) as usize;
    let batch_records = rows.min(batch_records.max(1));
    let batch_header = encode_batch_header(
        batch_records,
        record_len(corpus, route_meta),
        route_meta.is_some(),
    );
    conn.writing = Some(BatchActiveReq {
        start_row_idx: next.row_idx,
        t_sched: next.t_sched,
        deadline: Instant::now() + timeout,
        record: next.record,
        rows: batch_records as u32,
        batch_records,
        batch_header,
    });
    conn.write_state = BatchWriteState::default();
}

fn update_batch_conn_interest(conn: &mut BatchConn, poll: &Poll) -> Result<()> {
    let token = conn.token;
    let writable = conn.wants_write();
    reregister_conn(poll, &mut conn.stream, token, writable)
}

fn record_batch_pending_drop(conn: &mut BatchConn, local_stats: &mut HttpBatchWorkerLocal) {
    let mut dropped_rows = 0u64;
    while let Some(req) = conn.pending.pop_front() {
        if req.record {
            dropped_rows += req.rows.max(1) as u64;
        }
    }
    if dropped_rows > 0 {
        local_stats.record_dropped_after_attempt(dropped_rows);
    }
}

fn record_batch_inflight_timeouts(
    conn: &mut BatchConn,
    now: Instant,
    local_stats: &mut HttpBatchWorkerLocal,
) {
    if let Some(active) = conn.writing.take() {
        if active.record {
            local_stats.record_batch(throughput_done_from_batch_timeout(&active, now));
        }
    }
    while let Some(active) = conn.inflight.pop_front() {
        if active.record {
            local_stats.record_batch(throughput_done_from_batch_timeout(&active, now));
        }
    }
    conn.write_state = BatchWriteState::default();
}

fn fail_batch_conn(
    conn: &mut BatchConn,
    poll: &Poll,
    target: &Target,
    now: Instant,
    local_stats: &mut HttpBatchWorkerLocal,
) -> Result<()> {
    record_batch_pending_drop(conn, local_stats);
    record_batch_inflight_timeouts(conn, now, local_stats);
    reconnect_batch_conn(conn, poll, target)
}

fn start_next_send_batch(
    conn: &mut BatchConn,
    poll: &Poll,
    timeout: Duration,
    corpus: &PayloadCorpus,
    route_meta: Option<&RouteMetaCorpus>,
    batch_records: usize,
) -> Result<()> {
    prime_next_write_batch(conn, timeout, corpus, route_meta, batch_records);
    update_batch_conn_interest(conn, poll)
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
            writing: None,
            inflight: VecDeque::with_capacity(args.max_inflight_per_conn.max(1)),
            write_state: BatchWriteState::default(),
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
            if conn
                .oldest_deadline()
                .is_some_and(|deadline| now >= deadline)
            {
                fail_batch_conn(conn, &poll, &target, now, &mut local_stats)?;
            }
        }

        let all_idle = (pacing_done || queue.is_empty())
            && local_pending.is_empty()
            && conns
                .iter()
                .all(|c| c.writing.is_none() && c.inflight.is_empty() && c.pending.is_empty());
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
                    record_batch_inflight_timeouts(conn, now, &mut local_stats);
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
            if ev.is_readable() && !conn.inflight.is_empty() {
                loop {
                    match conn.parser.read_batch_ack_from(&mut conn.stream) {
                        Ok(Some((meta, ack))) => {
                            let done_at = Instant::now();
                            let Some(active) = conn.inflight.pop_front() else {
                                fail_batch_conn(conn, &poll, &target, done_at, &mut local_stats)?;
                                break;
                            };
                            if active.record {
                                let batch =
                                    throughput_done_from_batch_ack(&active, done_at, meta, ack);
                                local_stats.record_batch(batch);
                            }
                        }
                        Ok(None) => break,
                        Err(_) => {
                            fail_batch_conn(
                                conn,
                                &poll,
                                &target,
                                Instant::now(),
                                &mut local_stats,
                            )?;
                            break;
                        }
                    }
                }
            }

            if ev.is_writable() {
                loop {
                    prime_next_write_batch(
                        conn,
                        timeout,
                        &corpus,
                        route_meta.as_ref(),
                        args.batch_records,
                    );
                    if conn.writing.is_none() {
                        break;
                    }
                    match flush_write_batch(conn, &req_tpl, &corpus, route_meta.as_ref()) {
                        Ok(true) => {
                            if let Some(mut active) = conn.writing.take() {
                                active.deadline = Instant::now() + timeout;
                                conn.inflight.push_back(active);
                            }
                            conn.write_state = BatchWriteState::default();
                        }
                        Ok(false) => break,
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                        Err(_) => {
                            fail_batch_conn(
                                conn,
                                &poll,
                                &target,
                                Instant::now(),
                                &mut local_stats,
                            )?;
                            break;
                        }
                    }
                }
            }

            update_batch_conn_interest(conn, &poll)?;
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

