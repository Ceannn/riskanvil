use crate::config::{Args, BenchMode, SchedulerMode};
use crate::dataset::{Target, WorkloadSource};
use crate::metrics::{AggEvent, BatchDone, WorkerDone, WorkerSnapshot};
use crate::protocol::{decode_h2_response, error_response, timeout_response, ThroughputResponse};
use anyhow::{Context, Result};
use bytes::{Bytes, BytesMut};
use futures_util::{
    future::poll_fn,
    stream::{FuturesUnordered, StreamExt},
};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tokio::runtime::Builder as TokioRuntimeBuilder;
use tracing::{debug, info_span, trace, warn, Instrument};

const RECONNECT_COOLDOWN_MS: u64 = 100;
const RECONNECT_JITTER_MS: u64 = 25;
const CONN_DEAD_AFTER_FAILURES: u32 = 8;
const SNAPSHOT_EVERY_MS: u64 = 100;

#[derive(Clone, Copy)]
struct BatchState {
    planned_at: Instant,
    first_sent_at: Option<Instant>,
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

struct InflightReq {
    conn_idx: usize,
    batch_id: Option<u64>,
    deadline: Instant,
    body: Bytes,
    weight: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConnPhase {
    Healthy,
    CoolingDown,
    Reconnecting,
    Dead,
}

#[derive(Clone, Copy, Debug)]
enum SendErrKind {
    StreamError,
    ConnBroken,
    LocalPressure,
}

struct ConnState {
    sender: Option<h2::client::SendRequest<Bytes>>,
    inflight: usize,
    phase: ConnPhase,
    driver_alive: Arc<AtomicBool>,
    next_retry_at: Instant,
    consecutive_conn_failures: u32,
    reconnect_attempts: u64,
    last_error_kind: Option<SendErrKind>,
}

impl ConnState {
    fn healthy(
        sender: h2::client::SendRequest<Bytes>,
        driver_alive: Arc<AtomicBool>,
        now: Instant,
    ) -> Self {
        Self {
            sender: Some(sender),
            inflight: 0,
            phase: ConnPhase::Healthy,
            driver_alive,
            next_retry_at: now,
            consecutive_conn_failures: 0,
            reconnect_attempts: 0,
            last_error_kind: None,
        }
    }

    fn is_sendable(&self, max_inflight_per_conn: usize) -> bool {
        self.phase == ConnPhase::Healthy
            && self.sender.is_some()
            && self.driver_alive.load(Ordering::Acquire)
            && self.inflight < max_inflight_per_conn.max(1)
    }
}

#[derive(Default)]
struct WorkerLocalStats {
    stream_error_count: u64,
    conn_error_count: u64,
    reconnect_attempts: u64,
    reconnect_budget_exhausted: u64,
    force_timeout_all_count: u64,
    worker_exit_with_inflight_count: u64,
    active_live_connections_max: u64,
    client_drop_queue_full: u64,
    bench_limited_transitions: u64,
}

struct ReconnectBudget {
    window_start: Instant,
    used: usize,
    limit: usize,
}

impl ReconnectBudget {
    fn new(now: Instant, limit: usize) -> Self {
        Self {
            window_start: now,
            used: 0,
            limit: limit.max(1),
        }
    }

    fn try_acquire(&mut self, now: Instant) -> bool {
        if now.duration_since(self.window_start) >= Duration::from_secs(1) {
            self.window_start = now;
            self.used = 0;
        }
        if self.used >= self.limit {
            return false;
        }
        self.used += 1;
        true
    }
}

type RespFuture = futures_util::future::BoxFuture<'static, (usize, anyhow::Result<ThroughputResponse>)>;

pub fn spawn_worker(
    worker_id: usize,
    worker_count: usize,
    cpu: Option<usize>,
    args: Args,
    target: Target,
    workload: WorkloadSource,
    tx: crossbeam_channel::Sender<AggEvent>,
    stop: Arc<AtomicBool>,
) -> std::thread::JoinHandle<Result<()>> {
    thread::spawn(move || {
        pin_current_thread(cpu)?;
        let rt = TokioRuntimeBuilder::new_current_thread()
            .enable_all()
            .build()
            .context("build worker runtime")?;
        let span = info_span!("bench_h2_worker", worker_id, cpu);
        rt.block_on(
            worker_loop(
                worker_id,
                worker_count,
                args,
                target,
                workload,
                tx,
                stop,
            )
            .instrument(span),
        )
    })
}

async fn worker_loop(
    worker_id: usize,
    worker_count: usize,
    args: Args,
    target: Target,
    workload: WorkloadSource,
    tx: crossbeam_channel::Sender<AggEvent>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    let conn_count = args.conns_per_worker.max(1);
    let slot_count = conn_count
        .saturating_mul(args.max_inflight_per_conn.max(1))
        .max(args.throughput_batch_size.max(1));
    let request_weight = match args.mode {
        BenchMode::Single => 1u32,
        BenchMode::Batch => args.batch_records.max(1) as u32,
    };
    let body_len = workload.body_len_for(args.mode, args.batch_records);
    let started = Instant::now();
    let warmup_end = started + Duration::from_secs(args.warmup);
    let stop_at = warmup_end + Duration::from_secs(args.duration);
    let timeout = Duration::from_millis(args.timeout_ms.max(1));
    let soft_budget_cap = slot_count
        .max(args.throughput_batch_size.saturating_mul(2))
        .max(256) as f64;
    let qps_share = split_qps(args.qps, worker_count, worker_id);

    let mut conns = Vec::with_capacity(conn_count);
    for _ in 0..conn_count {
        conns.push(open_conn(&args, &target, started).await?);
    }
    trace!(worker_id, conn_count, slot_count, qps_share, "worker initialized");

    let mut free_slots = Vec::with_capacity(slot_count);
    let mut inflight = Vec::with_capacity(slot_count);
    let mut slot_bufs = match workload {
        WorkloadSource::Corpus { .. } => {
            let mut bufs = Vec::with_capacity(slot_count);
            for _ in 0..slot_count {
                let mut buf = BytesMut::with_capacity(body_len);
                buf.resize(body_len, 0);
                bufs.push(Some(buf));
            }
            Some(bufs)
        }
        WorkloadSource::Ceiling { .. } => None,
    };
    for slot_idx in 0..slot_count {
        free_slots.push(slot_idx);
        inflight.push(None);
    }

    let mut pending_responses: FuturesUnordered<RespFuture> = FuturesUnordered::new();
    let mut batches = HashMap::<u64, BatchState>::new();
    let mut open_batch_id: Option<u64> = None;
    let mut open_batch_len = 0usize;
    let mut next_batch_id = 1u64;
    let mut next_seq = 0u64;
    let mut next_conn_rr = worker_id % conn_count;
    let mut soft_budget = 0.0f64;
    let mut last_budget_refill = started;
    let mut last_snapshot_flush = started;
    let mut scheduling = true;
    let mut drain_started: Option<Instant> = None;
    let mut worker_stats = WorkerLocalStats::default();
    let mut reconnect_budget =
        ReconnectBudget::new(started, args.conns_per_worker.max(1).saturating_mul(4));
    let mut bench_limited_active = false;
    let mut snapshot = WorkerSnapshot::new()?;

    loop {
        let now = Instant::now();
        if scheduling && (stop.load(Ordering::Acquire) || now >= stop_at) {
            scheduling = false;
            drain_started = Some(now);
            debug!(worker_id, inflight = pending_responses.len(), "worker entered drain-only");
            if let Some(batch_id) = open_batch_id.take() {
                if let Some(batch) = batches.get_mut(&batch_id) {
                    batch.sealed = true;
                }
                open_batch_len = 0;
            }
        }

        refresh_connection_states(
            &args,
            &target,
            &mut conns,
            now,
            &mut reconnect_budget,
            &mut worker_stats,
        )
        .await;
        worker_stats.active_live_connections_max = worker_stats
            .active_live_connections_max
            .max(active_live_connections(&conns) as u64);

        if scheduling {
            match args.scheduler {
                SchedulerMode::SoftPace => {
                    let budget_before = soft_budget;
                    refill_soft_budget(
                        &mut soft_budget,
                        &mut last_budget_refill,
                        qps_share,
                        soft_budget_cap,
                        now,
                    );
                    if now >= warmup_end {
                        let dropped =
                            saturated_budget_drop(budget_before, soft_budget, soft_budget_cap);
                        if dropped > 0 {
                            snapshot.dropped_conn_queue_full += dropped;
                            worker_stats.client_drop_queue_full += dropped;
                            trace!(worker_id, dropped, "soft budget saturated");
                        }
                    }
                }
                SchedulerMode::Burst => {
                    soft_budget = soft_budget_cap;
                    last_budget_refill = now;
                }
            }
        }

        reap_timeouts(
            now,
            &mut inflight,
            &mut conns,
            &mut batches,
            &mut slot_bufs,
            &mut free_slots,
            body_len,
            &mut snapshot,
        );

        while let Ok(Some((slot_idx, result))) =
            tokio::time::timeout(Duration::from_millis(0), pending_responses.next()).await
        {
            handle_response_result(
                slot_idx,
                result,
                Instant::now(),
                &mut inflight,
                &mut conns,
                &mut batches,
                &mut slot_bufs,
                &mut free_slots,
                body_len,
                &mut snapshot,
                &mut worker_stats,
            );
        }

        if !scheduling && pending_responses.is_empty() && all_slots_free(&inflight) {
            break;
        }

        if !scheduling {
            if let Some(drain_started) = drain_started {
                if drain_started.elapsed() >= timeout {
                    let forced = timeout_remaining_requests(
                        Instant::now(),
                        &mut inflight,
                        &mut conns,
                        &mut batches,
                        &mut slot_bufs,
                        &mut free_slots,
                        body_len,
                        &mut snapshot,
                    );
                    if forced > 0 {
                        worker_stats.force_timeout_all_count += forced as u64;
                        worker_stats.worker_exit_with_inflight_count += forced as u64;
                        warn!(worker_id, forced, "drain grace expired with inflight requests");
                    }
                    break;
                }
            }
        }

        let mut sent_any = false;
        if scheduling {
            loop {
                if args.scheduler == SchedulerMode::SoftPace && soft_budget < request_weight as f64 {
                    break;
                }
                let Some(conn_idx) = select_conn(&conns, args.max_inflight_per_conn, next_conn_rr) else {
                    note_bench_limited(&mut bench_limited_active, &mut worker_stats);
                    break;
                };
                let Some(slot_idx) = free_slots.pop() else {
                    note_bench_limited(&mut bench_limited_active, &mut worker_stats);
                    break;
                };
                next_conn_rr = (conn_idx + 1) % conn_count;

                let body = build_slot_body(
                    args.mode,
                    &workload,
                    worker_id,
                    next_seq,
                    slot_idx,
                    body_len,
                    args.batch_records,
                    &mut slot_bufs,
                )?;
                let send_now = Instant::now();

                match send_one(&mut conns[conn_idx], &target, body.clone()).await {
                    Ok(response) => {
                        let record = send_now >= warmup_end;
                        let batch_id = if record {
                            Some(start_or_extend_batch(
                                &mut batches,
                                &mut open_batch_id,
                                &mut open_batch_len,
                                &mut next_batch_id,
                                send_now,
                                args.throughput_batch_size,
                            ))
                        } else {
                            None
                        };
                        conns[conn_idx].inflight += 1;
                        inflight[slot_idx] = Some(InflightReq {
                            conn_idx,
                            batch_id,
                            deadline: send_now + timeout,
                            body,
                            weight: request_weight,
                        });
                        pending_responses.push(Box::pin(async move {
                            (slot_idx, decode_h2_response(response).await)
                        }));
                        soft_budget = (soft_budget - request_weight as f64).max(0.0);
                        next_seq = next_seq.wrapping_add(request_weight as u64);
                        if record {
                            snapshot.attempted += request_weight as u64;
                        }
                        sent_any = true;
                        bench_limited_active = false;
                    }
                    Err(SendErrKind::ConnBroken) => {
                        mark_conn_broken(
                            &mut conns[conn_idx],
                            send_now,
                            SendErrKind::ConnBroken,
                            &mut worker_stats,
                        );
                        debug!(worker_id, conn_idx, "connection marked broken after send failure");
                        restore_body_slot(
                            slot_idx,
                            Some(body),
                            &mut slot_bufs,
                            &mut free_slots,
                            body_len,
                        );
                    }
                    Err(SendErrKind::LocalPressure) => {
                        restore_body_slot(
                            slot_idx,
                            Some(body),
                            &mut slot_bufs,
                            &mut free_slots,
                            body_len,
                        );
                        note_bench_limited(&mut bench_limited_active, &mut worker_stats);
                        trace!(worker_id, conn_idx, "local pressure blocked send");
                        break;
                    }
                    Err(SendErrKind::StreamError) => {
                        worker_stats.stream_error_count += 1;
                        debug!(worker_id, conn_idx, "stream error while sending");
                        restore_body_slot(
                            slot_idx,
                            Some(body),
                            &mut slot_bufs,
                            &mut free_slots,
                            body_len,
                        );
                    }
                }
            }
        }

        if sent_any {
            continue;
        }

        tokio::select! {
            maybe = pending_responses.next(), if !pending_responses.is_empty() => {
                if let Some((slot_idx, result)) = maybe {
                    handle_response_result(
                        slot_idx,
                        result,
                        Instant::now(),
                        &mut inflight,
                        &mut conns,
                        &mut batches,
                        &mut slot_bufs,
                        &mut free_slots,
                        body_len,
                        &mut snapshot,
                        &mut worker_stats,
                    );
                }
            }
            _ = tokio::time::sleep(Duration::from_millis(1)) => {}
        }

        flush_snapshot_if_needed(&tx, &mut snapshot, &mut last_snapshot_flush, Instant::now())?;
    }

    if let Some(batch_id) = open_batch_id.take() {
        if let Some(batch) = batches.get_mut(&batch_id) {
            batch.sealed = true;
        }
        maybe_finalize_batch(&mut snapshot, batch_id, Instant::now(), &mut batches);
    }
    let batch_ids: Vec<u64> = batches.keys().copied().collect();
    for batch_id in batch_ids {
        if let Some(batch) = batches.get_mut(&batch_id) {
            batch.sealed = true;
        }
        maybe_finalize_batch(&mut snapshot, batch_id, Instant::now(), &mut batches);
    }
    flush_snapshot(&tx, &mut snapshot)?;

    let _ = tx.send(AggEvent::WorkerDone(WorkerDone {
        stream_error_count: worker_stats.stream_error_count,
        conn_error_count: worker_stats.conn_error_count,
        reconnect_attempts: worker_stats.reconnect_attempts,
        reconnect_budget_exhausted: worker_stats.reconnect_budget_exhausted,
        force_timeout_all_count: worker_stats.force_timeout_all_count,
        worker_exit_with_inflight_count: worker_stats.worker_exit_with_inflight_count,
        active_live_connections_max: worker_stats.active_live_connections_max,
        client_drop_queue_full: worker_stats.client_drop_queue_full,
        bench_limited_transitions: worker_stats.bench_limited_transitions,
    }));
    Ok(())
}

async fn open_conn(args: &Args, target: &Target, now: Instant) -> Result<ConnState> {
    let stream = tokio::net::TcpStream::connect(target.addr)
        .await
        .with_context(|| format!("connect {}", target.addr))?;
    stream.set_nodelay(true).ok();
    let mut builder = h2::client::Builder::new();
    builder.initial_window_size(args.initial_stream_window);
    builder.initial_connection_window_size(args.initial_conn_window);
    builder.initial_max_send_streams(args.initial_max_send_streams);
    let (sender, connection) = builder.handshake(stream).await.context("h2 handshake")?;
    let alive = Arc::new(AtomicBool::new(true));
    let alive_bg = alive.clone();
    tokio::spawn(
        async move {
            let _ = connection.await;
            alive_bg.store(false, Ordering::Release);
        }
        .instrument(info_span!("bench_h2_conn_driver")),
    );
    Ok(ConnState::healthy(sender, alive, now))
}

async fn send_one(
    conn: &mut ConnState,
    target: &Target,
    body: Bytes,
) -> Result<h2::client::ResponseFuture, SendErrKind> {
    if conn.phase != ConnPhase::Healthy {
        return Err(SendErrKind::LocalPressure);
    }
    let Some(sender) = conn.sender.as_mut() else {
        return Err(SendErrKind::ConnBroken);
    };
    if !conn.driver_alive.load(Ordering::Acquire) {
        return Err(SendErrKind::ConnBroken);
    }
    let request = http::Request::builder()
        .method("POST")
        .uri(&target.full_uri)
        .version(http::Version::HTTP_2)
        .header("content-type", "application/octet-stream")
        .header("content-length", body.len())
        .header("x-risk-bench-mode", "throughput")
        .body(())
        .map_err(|_| SendErrKind::LocalPressure)?;
    poll_fn(|cx| sender.poll_ready(cx))
        .await
        .map_err(|_| SendErrKind::ConnBroken)?;
    let (response, mut send_stream) = sender
        .send_request(request, false)
        .map_err(|_| SendErrKind::ConnBroken)?;
    send_stream
        .send_data(body, true)
        .map_err(|_| SendErrKind::ConnBroken)?;
    Ok(response)
}

fn build_slot_body(
    mode: BenchMode,
    workload: &WorkloadSource,
    worker_id: usize,
    seq: u64,
    slot_idx: usize,
    body_len: usize,
    batch_records: usize,
    slot_bufs: &mut Option<Vec<Option<BytesMut>>>,
) -> Result<Bytes> {
    match slot_bufs {
        Some(bufs) => {
            let mut buf = bufs[slot_idx]
                .take()
                .unwrap_or_else(|| BytesMut::with_capacity(body_len));
            if buf.len() != body_len {
                buf.resize(body_len, 0);
            }
            workload.fill_body_for_into(mode, seq, worker_id, batch_records, &mut buf[..])?;
            Ok(buf.freeze())
        }
        None => Ok(workload.build_body_for(mode, seq, worker_id, batch_records)),
    }
}

fn restore_body_slot(
    slot_idx: usize,
    body: Option<Bytes>,
    slot_bufs: &mut Option<Vec<Option<BytesMut>>>,
    free_slots: &mut Vec<usize>,
    body_len: usize,
) {
    if let Some(bufs) = slot_bufs {
        let mut restored = match body.and_then(|body| body.try_into_mut().ok()) {
            Some(buf) => buf,
            None => {
                let mut buf = BytesMut::with_capacity(body_len);
                buf.resize(body_len, 0);
                buf
            }
        };
        if restored.len() != body_len {
            restored.resize(body_len, 0);
        }
        bufs[slot_idx] = Some(restored);
    }
    free_slots.push(slot_idx);
}

fn release_slot(
    slot_idx: usize,
    inflight: &mut [Option<InflightReq>],
    conns: &mut [ConnState],
    slot_bufs: &mut Option<Vec<Option<BytesMut>>>,
    free_slots: &mut Vec<usize>,
    body_len: usize,
) -> Option<InflightReq> {
    let req = inflight.get_mut(slot_idx)?.take()?;
    conns[req.conn_idx].inflight = conns[req.conn_idx].inflight.saturating_sub(1);
    restore_body_slot(
        slot_idx,
        Some(req.body.clone()),
        slot_bufs,
        free_slots,
        body_len,
    );
    Some(req)
}

fn refill_soft_budget(
    budget: &mut f64,
    last_refill: &mut Instant,
    qps_share: u64,
    cap: f64,
    now: Instant,
) {
    if qps_share == 0 {
        return;
    }
    let elapsed = now.duration_since(*last_refill).as_secs_f64();
    *last_refill = now;
    let added = elapsed * qps_share as f64;
    *budget = (*budget + added).min(cap);
}

fn saturated_budget_drop(before: f64, after: f64, cap: f64) -> u64 {
    if before >= cap && after >= cap {
        0
    } else {
        0
    }
}

fn active_live_connections(conns: &[ConnState]) -> usize {
    conns
        .iter()
        .filter(|conn| conn.phase == ConnPhase::Healthy && conn.sender.is_some())
        .count()
}

fn note_bench_limited(active: &mut bool, stats: &mut WorkerLocalStats) {
    if !*active {
        stats.bench_limited_transitions += 1;
        *active = true;
    }
}

fn start_or_extend_batch(
    batches: &mut HashMap<u64, BatchState>,
    open_batch_id: &mut Option<u64>,
    open_batch_len: &mut usize,
    next_batch_id: &mut u64,
    now: Instant,
    throughput_batch_size: usize,
) -> u64 {
    let batch_id = open_batch_id.unwrap_or_else(|| {
        let batch_id = *next_batch_id;
        *next_batch_id += 1;
        batches.insert(
            batch_id,
            BatchState {
                planned_at: now,
                first_sent_at: None,
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
        *open_batch_id = Some(batch_id);
        batch_id
    });
    *open_batch_len += 1;
    if let Some(batch) = batches.get_mut(&batch_id) {
        batch.outstanding += 1;
        if batch.first_sent_at.is_none() {
            batch.first_sent_at = Some(now);
        }
        if *open_batch_len >= throughput_batch_size.max(1) {
            batch.sealed = true;
            *open_batch_id = None;
            *open_batch_len = 0;
        }
    }
    batch_id
}

fn mark_conn_broken(
    conn: &mut ConnState,
    now: Instant,
    err: SendErrKind,
    stats: &mut WorkerLocalStats,
) {
    conn.sender = None;
    conn.phase = if conn.consecutive_conn_failures + 1 >= CONN_DEAD_AFTER_FAILURES {
        ConnPhase::Dead
    } else {
        ConnPhase::CoolingDown
    };
    conn.driver_alive.store(false, Ordering::Release);
    conn.consecutive_conn_failures = conn.consecutive_conn_failures.saturating_add(1);
    conn.last_error_kind = Some(err);
    conn.next_retry_at = now
        + Duration::from_millis(
            RECONNECT_COOLDOWN_MS
                + ((conn.consecutive_conn_failures as u64 * 17) % RECONNECT_JITTER_MS.max(1)),
        );
    stats.conn_error_count += 1;
}

async fn refresh_connection_states(
    args: &Args,
    target: &Target,
    conns: &mut [ConnState],
    now: Instant,
    reconnect_budget: &mut ReconnectBudget,
    stats: &mut WorkerLocalStats,
) {
    for conn in conns.iter_mut() {
        if conn.phase == ConnPhase::Healthy && !conn.driver_alive.load(Ordering::Acquire) {
            mark_conn_broken(conn, now, SendErrKind::ConnBroken, stats);
            debug!("connection driver exited while healthy");
        }

        if matches!(conn.phase, ConnPhase::CoolingDown | ConnPhase::Reconnecting)
            && conn.inflight == 0
            && now >= conn.next_retry_at
        {
            if !reconnect_budget.try_acquire(now) {
                stats.reconnect_budget_exhausted += 1;
                conn.next_retry_at = now + Duration::from_millis(RECONNECT_COOLDOWN_MS);
                trace!("reconnect budget exhausted");
                continue;
            }
            conn.phase = ConnPhase::Reconnecting;
            conn.reconnect_attempts += 1;
            stats.reconnect_attempts += 1;
            trace!(attempt = conn.reconnect_attempts, "attempting reconnect");
            match open_conn(args, target, now).await {
                Ok(new_conn) => {
                    *conn = new_conn;
                    debug!("reconnect succeeded");
                }
                Err(_) => {
                    conn.phase = if conn.consecutive_conn_failures + 1 >= CONN_DEAD_AFTER_FAILURES {
                        ConnPhase::Dead
                    } else {
                        ConnPhase::CoolingDown
                    };
                    conn.consecutive_conn_failures = conn.consecutive_conn_failures.saturating_add(1);
                    conn.next_retry_at = now
                        + Duration::from_millis(
                            RECONNECT_COOLDOWN_MS
                                + ((conn.consecutive_conn_failures as u64 * 17)
                                    % RECONNECT_JITTER_MS.max(1)),
                        );
                    stats.conn_error_count += 1;
                    debug!("reconnect failed");
                }
            }
        }
    }
}

fn handle_response_result(
    slot_idx: usize,
    result: anyhow::Result<ThroughputResponse>,
    now: Instant,
    inflight: &mut [Option<InflightReq>],
    conns: &mut [ConnState],
    batches: &mut HashMap<u64, BatchState>,
    slot_bufs: &mut Option<Vec<Option<BytesMut>>>,
    free_slots: &mut Vec<usize>,
    body_len: usize,
    snapshot: &mut WorkerSnapshot,
    stats: &mut WorkerLocalStats,
) {
    let Some(req) = release_slot(
        slot_idx,
        inflight,
        conns,
        slot_bufs,
        free_slots,
        body_len,
    ) else {
        return;
    };

    let response = match result {
        Ok(resp) => resp,
        Err(_) => {
            let conn = &mut conns[req.conn_idx];
            if conn.driver_alive.load(Ordering::Acquire) && conn.phase == ConnPhase::Healthy {
                conn.last_error_kind = Some(SendErrKind::StreamError);
                stats.stream_error_count += 1;
                error_response(req.weight)
            } else {
                mark_conn_broken(conn, now, SendErrKind::ConnBroken, stats);
                error_response(req.weight)
            }
        }
    };

    if let Some(batch_id) = req.batch_id {
        if let Some(batch) = batches.get_mut(&batch_id) {
            update_batch(batch, response);
            batch.outstanding = batch.outstanding.saturating_sub(1);
        }
        maybe_finalize_batch(snapshot, batch_id, now, batches);
    }
}

fn timeout_remaining_requests(
    now: Instant,
    inflight: &mut [Option<InflightReq>],
    conns: &mut [ConnState],
    batches: &mut HashMap<u64, BatchState>,
    slot_bufs: &mut Option<Vec<Option<BytesMut>>>,
    free_slots: &mut Vec<usize>,
    body_len: usize,
    snapshot: &mut WorkerSnapshot,
) -> usize {
    let mut forced = 0usize;
    for slot_idx in 0..inflight.len() {
        if let Some(req) = release_slot(slot_idx, inflight, conns, slot_bufs, free_slots, body_len) {
            forced += 1;
            if let Some(batch_id) = req.batch_id {
                if let Some(batch) = batches.get_mut(&batch_id) {
                    update_batch(batch, timeout_response(req.weight));
                    batch.outstanding = batch.outstanding.saturating_sub(1);
                }
                maybe_finalize_batch(snapshot, batch_id, now, batches);
            }
        }
    }
    forced
}

fn all_slots_free(inflight: &[Option<InflightReq>]) -> bool {
    inflight.iter().all(Option::is_none)
}

fn select_conn(conns: &[ConnState], max_inflight_per_conn: usize, rr: usize) -> Option<usize> {
    let mut best: Option<(usize, usize)> = None;
    for step in 0..conns.len() {
        let idx = (rr + step) % conns.len();
        if conns[idx].is_sendable(max_inflight_per_conn) {
            let inflight = conns[idx].inflight;
            match best {
                Some((_, best_inflight)) if inflight >= best_inflight => {}
                _ => best = Some((idx, inflight)),
            }
        }
    }
    best.map(|(idx, _)| idx)
}

fn update_batch(batch: &mut BatchState, resp: ThroughputResponse) {
    if resp.timeout {
        batch.timeout += resp.rows as u64;
    } else if (200..300).contains(&resp.status_code) {
        batch.ok += resp.rows as u64;
        batch.http_2xx += resp.rows as u64;
    } else {
        batch.err += resp.rows as u64;
        if resp.status_code == 429 {
            batch.http_429 += resp.rows as u64;
        } else if resp.status_code >= 500 {
            batch.http_5xx += resp.rows as u64;
        }
    }
    if resp.qsb2_samples > 0 {
        batch.qsb2_samples += resp.qsb2_samples as u64;
    }
    if resp.rsk1_samples > 0 {
        batch.rsk1_samples += resp.rsk1_samples as u64;
    }
    batch.used_l2 += resp.used_l2_count as u64;
    for (dst, src) in batch
        .decision_counts
        .iter_mut()
        .zip(resp.decision_counts.iter())
    {
        *dst += *src as u64;
    }
}

fn maybe_finalize_batch(
    snapshot: &mut WorkerSnapshot,
    batch_id: u64,
    now: Instant,
    batches: &mut HashMap<u64, BatchState>,
) {
    let done = batches
        .get(&batch_id)
        .map(|b| b.sealed && b.outstanding == 0)
        .unwrap_or(false);
    if !done {
        return;
    }
    if let Some(batch) = batches.remove(&batch_id) {
        let first_sent_at = batch.first_sent_at.unwrap_or(batch.planned_at);
        record_batch_done(
            snapshot,
            BatchDone {
            client_qwait_us: first_sent_at.duration_since(batch.planned_at).as_micros() as u64,
            service_rtt_us: now.duration_since(first_sent_at).as_micros() as u64,
            e2e_us: now.duration_since(batch.planned_at).as_micros() as u64,
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
        },
        );
    }
}

fn reap_timeouts(
    now: Instant,
    inflight: &mut [Option<InflightReq>],
    conns: &mut [ConnState],
    batches: &mut HashMap<u64, BatchState>,
    slot_bufs: &mut Option<Vec<Option<BytesMut>>>,
    free_slots: &mut Vec<usize>,
    body_len: usize,
    snapshot: &mut WorkerSnapshot,
) {
    for slot_idx in 0..inflight.len() {
        let expired = inflight[slot_idx]
            .as_ref()
            .map(|req| now >= req.deadline)
            .unwrap_or(false);
        if !expired {
            continue;
        }
        if let Some(req) = release_slot(slot_idx, inflight, conns, slot_bufs, free_slots, body_len)
        {
            if let Some(batch_id) = req.batch_id {
                if let Some(batch) = batches.get_mut(&batch_id) {
                    update_batch(batch, timeout_response(req.weight));
                    batch.outstanding = batch.outstanding.saturating_sub(1);
                }
                maybe_finalize_batch(snapshot, batch_id, now, batches);
            }
        }
    }
}

fn record_batch_done(snapshot: &mut WorkerSnapshot, batch: BatchDone) {
    snapshot.ok += batch.ok;
    snapshot.err += batch.err;
    snapshot.timeout += batch.timeout;
    snapshot.http_2xx += batch.http_2xx;
    snapshot.http_429 += batch.http_429;
    snapshot.http_5xx += batch.http_5xx;
    snapshot.qsb2_samples += batch.qsb2_samples;
    snapshot.rsk1_samples += batch.rsk1_samples;
    snapshot.used_l2 += batch.used_l2;
    for (dst, src) in snapshot
        .decision_counts
        .iter_mut()
        .zip(batch.decision_counts.iter())
    {
        *dst += *src;
    }
    let _ = snapshot
        .batch_client_qwait_us
        .record(batch.client_qwait_us.max(1));
    let _ = snapshot
        .batch_service_rtt_us
        .record(batch.service_rtt_us.max(1));
    let _ = snapshot.batch_e2e_us.record(batch.e2e_us.max(1));
}

fn flush_snapshot_if_needed(
    tx: &crossbeam_channel::Sender<AggEvent>,
    snapshot: &mut WorkerSnapshot,
    last_snapshot_flush: &mut Instant,
    now: Instant,
) -> Result<()> {
    if now.duration_since(*last_snapshot_flush) < Duration::from_millis(SNAPSHOT_EVERY_MS) {
        return Ok(());
    }
    flush_snapshot(tx, snapshot)?;
    *last_snapshot_flush = now;
    Ok(())
}

fn flush_snapshot(
    tx: &crossbeam_channel::Sender<AggEvent>,
    snapshot: &mut WorkerSnapshot,
) -> Result<()> {
    let should_send = snapshot.attempted > 0
        || snapshot.dropped_conn_queue_full > 0
        || snapshot.ok > 0
        || snapshot.err > 0
        || snapshot.timeout > 0
        || snapshot.batch_e2e_us.len() > 0;
    if !should_send {
        return Ok(());
    }
    let outbound = std::mem::replace(snapshot, WorkerSnapshot::new()?);
    let _ = tx.send(AggEvent::WorkerSnapshot(outbound));
    Ok(())
}

fn split_qps(total_qps: u64, worker_count: usize, worker_id: usize) -> u64 {
    if worker_count == 0 {
        return total_qps;
    }
    let base = total_qps / worker_count as u64;
    let extra = (worker_id as u64) < (total_qps % worker_count as u64);
    base + u64::from(extra)
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
            return Err(std::io::Error::last_os_error()).context("sched_setaffinity");
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_cpu: Option<usize>) -> Result<()> {
    Ok(())
}
