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

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_batch_aggregate_ack_for_test(
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
        for (idx, count) in decision_counts.iter().enumerate() {
            let off = 20 + idx * 4;
            out[off..off + 4].copy_from_slice(&count.to_le_bytes());
        }
        out
    }

    fn encode_http_batch_ack_response(record_count: u32, ok_count: u32) -> Vec<u8> {
        let ack =
            encode_batch_aggregate_ack_for_test(record_count, ok_count, 0, [ok_count, 0, 0, 0, 0]);
        let mut out = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\n\r\n",
            ack.len()
        )
        .into_bytes();
        out.extend_from_slice(&ack);
        out
    }

    #[test]
    fn batch_ack_parser_consumes_multiple_responses_from_one_buffer() {
        let mut parser = ResponseParser::with_capacity();
        let mut bytes = encode_http_batch_ack_response(128, 128);
        bytes.extend_from_slice(&encode_http_batch_ack_response(128, 64));
        parser.buf.extend_from_slice(&bytes);

        let (_, first) = parser
            .try_finish_batch_ack()
            .expect("first parse ok")
            .expect("first response present");
        assert_eq!(first.record_count, 128);
        assert_eq!(first.ok_count, 128);
        assert!(!parser.buf.is_empty());

        let (_, second) = parser
            .try_finish_batch_ack()
            .expect("second parse ok")
            .expect("second response present");
        assert_eq!(second.record_count, 128);
        assert_eq!(second.ok_count, 64);
        assert!(parser.buf.is_empty());
    }
}
