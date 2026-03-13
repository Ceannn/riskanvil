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

