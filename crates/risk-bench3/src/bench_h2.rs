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

