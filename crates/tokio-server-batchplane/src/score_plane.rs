use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

use axum::http::StatusCode;
use bytes::Bytes;
use crossbeam_channel::{bounded, Sender, TrySendError};
use crossbeam_queue::ArrayQueue;
use mio::Waker;
use tokio::sync::mpsc;
use tracing::warn;

use crate::{
    app::{BatchApp, BatchJobKind, BatchReply},
    metrics::{duration_to_us, StageTimingsUs},
};

pub enum JobBody {
    Bytes(Bytes),
    Vec(Vec<u8>),
}

impl JobBody {
    fn as_slice(&self) -> &[u8] {
        match self {
            JobBody::Bytes(body) => body.as_ref(),
            JobBody::Vec(body) => body.as_slice(),
        }
    }

    fn into_vec(self) -> Option<Vec<u8>> {
        match self {
            JobBody::Bytes(_) => None,
            JobBody::Vec(body) => Some(body),
        }
    }
}

struct BatchJob {
    conn_id: u64,
    seq: u64,
    kind: BatchJobKind,
    body: JobBody,
    parse_us: u32,
    read_to_submit_us: u32,
    first_byte_to_header_done_us: u32,
    header_done_to_credit_acquired_us: u32,
    credit_acquired_to_body_done_us: u32,
    body_done_to_submit_us: u32,
    inflight_depth: u16,
    body_fill_in_progress: u8,
    pending_complete_not_submitted: u8,
    submitted_at: Instant,
    completion_tx: CompletionTarget,
}

pub struct CompletionEntry {
    pub conn_id: u64,
    pub slot_token: usize,
    pub seq: u64,
    pub reply: BatchReply,
    pub recycled_body: Option<Vec<u8>>,
    pub completed_at: Instant,
    pub submitted_at: Instant,
    pub timings: StageTimingsUs,
}

enum CompletionTarget {
    Tokio(mpsc::Sender<CompletionEntry>),
    Shard {
        queue: Arc<ArrayQueue<CompletionEntry>>,
        waker: Arc<Waker>,
        slot_token: usize,
    },
}

pub struct ScorePlane {
    workers: Vec<Sender<BatchJob>>,
    routing: ScoreRouting,
    next_worker: AtomicUsize,
}

#[derive(Clone, Copy, Debug)]
pub enum ScoreRouting {
    StickyByConn,
    RoundRobin,
}

impl ScorePlane {
    pub fn new(
        worker_count: usize,
        queue_capacity: usize,
        routing: ScoreRouting,
        app: Arc<BatchApp>,
    ) -> Self {
        let mut workers = Vec::with_capacity(worker_count.max(1));
        for worker_idx in 0..worker_count.max(1) {
            let (tx, rx) = bounded::<BatchJob>(queue_capacity.max(1));
            let app2 = app.clone();
            std::thread::Builder::new()
                .name(format!("batchplane-score-{worker_idx}"))
                .spawn(move || {
                    while let Ok(job) = rx.recv() {
                        let score_start = Instant::now();
                        let queue_wait_us =
                            duration_to_us(score_start.duration_since(job.submitted_at));
                        let reply = app2.execute_job(job.kind, job.body.as_slice());
                        let recycled_body = job.body.into_vec();
                        let completed_at = Instant::now();
                        let entry = CompletionEntry {
                            conn_id: job.conn_id,
                            slot_token: 0,
                            seq: job.seq,
                            reply,
                            recycled_body,
                            completed_at,
                            submitted_at: job.submitted_at,
                            timings: StageTimingsUs {
                                parse: job.parse_us,
                                read_to_submit: job.read_to_submit_us,
                                first_byte_to_header_done: job.first_byte_to_header_done_us,
                                header_done_to_credit_acquired: job
                                    .header_done_to_credit_acquired_us,
                                credit_acquired_to_body_done: job
                                    .credit_acquired_to_body_done_us,
                                body_done_to_submit: job.body_done_to_submit_us,
                                queue_wait: queue_wait_us,
                                score: duration_to_us(completed_at.duration_since(score_start)),
                                completion_wait: 0,
                                write: 0,
                                total_residency: 0,
                                inflight_depth: job.inflight_depth,
                                body_fill_in_progress: job.body_fill_in_progress,
                                pending_complete_not_submitted: job
                                    .pending_complete_not_submitted,
                            },
                        };
                        match job.completion_tx {
                            CompletionTarget::Tokio(tx) => {
                                if let Err(err) = tx.blocking_send(entry) {
                                    warn!(worker_idx, error = %err, "completion receiver dropped");
                                }
                            }
                            CompletionTarget::Shard {
                                queue,
                                waker,
                                slot_token,
                            } => {
                                let mut entry = entry;
                                entry.slot_token = slot_token;
                                if let Err(_entry) = queue.push(entry) {
                                    warn!(worker_idx, "completion queue full");
                                } else if let Err(err) = waker.wake() {
                                    warn!(worker_idx, error = %err, "completion wake failed");
                                }
                            }
                        }
                    }
                })
                .expect("spawn batchplane score worker");
            workers.push(tx);
        }
        Self {
            workers,
            routing,
            next_worker: AtomicUsize::new(0),
        }
    }

    pub fn submit_tokio(
        &self,
        conn_id: u64,
        seq: u64,
        kind: BatchJobKind,
        body: JobBody,
        parse_us: u32,
        read_to_submit_us: u32,
        first_byte_to_header_done_us: u32,
        header_done_to_credit_acquired_us: u32,
        credit_acquired_to_body_done_us: u32,
        body_done_to_submit_us: u32,
        inflight_depth: u16,
        body_fill_in_progress: u8,
        pending_complete_not_submitted: u8,
        completion_tx: mpsc::Sender<CompletionEntry>,
    ) -> Result<(), BatchReply> {
        self.submit_inner(
            conn_id,
            seq,
            kind,
            body,
            parse_us,
            read_to_submit_us,
            first_byte_to_header_done_us,
            header_done_to_credit_acquired_us,
            credit_acquired_to_body_done_us,
            body_done_to_submit_us,
            inflight_depth,
            body_fill_in_progress,
            pending_complete_not_submitted,
            CompletionTarget::Tokio(completion_tx),
        )
    }

    pub fn submit_shard(
        &self,
        conn_id: u64,
        seq: u64,
        kind: BatchJobKind,
        body: JobBody,
        parse_us: u32,
        read_to_submit_us: u32,
        first_byte_to_header_done_us: u32,
        header_done_to_credit_acquired_us: u32,
        credit_acquired_to_body_done_us: u32,
        body_done_to_submit_us: u32,
        inflight_depth: u16,
        body_fill_in_progress: u8,
        pending_complete_not_submitted: u8,
        completion_queue: Arc<ArrayQueue<CompletionEntry>>,
        completion_waker: Arc<Waker>,
    ) -> Result<(), BatchReply> {
        self.submit_inner(
            conn_id,
            seq,
            kind,
            body,
            parse_us,
            read_to_submit_us,
            first_byte_to_header_done_us,
            header_done_to_credit_acquired_us,
            credit_acquired_to_body_done_us,
            body_done_to_submit_us,
            inflight_depth,
            body_fill_in_progress,
            pending_complete_not_submitted,
            CompletionTarget::Shard {
                queue: completion_queue,
                waker: completion_waker,
                slot_token: 0,
            },
        )
    }

    pub fn submit_shard_slot(
        &self,
        conn_id: u64,
        slot_token: usize,
        seq: u64,
        kind: BatchJobKind,
        body: JobBody,
        parse_us: u32,
        read_to_submit_us: u32,
        first_byte_to_header_done_us: u32,
        header_done_to_credit_acquired_us: u32,
        credit_acquired_to_body_done_us: u32,
        body_done_to_submit_us: u32,
        inflight_depth: u16,
        body_fill_in_progress: u8,
        pending_complete_not_submitted: u8,
        completion_queue: Arc<ArrayQueue<CompletionEntry>>,
        completion_waker: Arc<Waker>,
    ) -> Result<(), BatchReply> {
        self.submit_inner(
            conn_id,
            seq,
            kind,
            body,
            parse_us,
            read_to_submit_us,
            first_byte_to_header_done_us,
            header_done_to_credit_acquired_us,
            credit_acquired_to_body_done_us,
            body_done_to_submit_us,
            inflight_depth,
            body_fill_in_progress,
            pending_complete_not_submitted,
            CompletionTarget::Shard {
                queue: completion_queue,
                waker: completion_waker,
                slot_token,
            },
        )
    }

    fn submit_inner(
        &self,
        conn_id: u64,
        seq: u64,
        kind: BatchJobKind,
        body: JobBody,
        parse_us: u32,
        read_to_submit_us: u32,
        first_byte_to_header_done_us: u32,
        header_done_to_credit_acquired_us: u32,
        credit_acquired_to_body_done_us: u32,
        body_done_to_submit_us: u32,
        inflight_depth: u16,
        body_fill_in_progress: u8,
        pending_complete_not_submitted: u8,
        completion_tx: CompletionTarget,
    ) -> Result<(), BatchReply> {
        let worker_idx = match self.routing {
            ScoreRouting::StickyByConn => (conn_id as usize) % self.workers.len().max(1),
            ScoreRouting::RoundRobin => {
                self.next_worker.fetch_add(1, Ordering::Relaxed) % self.workers.len().max(1)
            }
        };
        let job = BatchJob {
            conn_id,
            seq,
            kind,
            body,
            parse_us,
            read_to_submit_us,
            first_byte_to_header_done_us,
            header_done_to_credit_acquired_us,
            credit_acquired_to_body_done_us,
            body_done_to_submit_us,
            inflight_depth,
            body_fill_in_progress,
            pending_complete_not_submitted,
            submitted_at: Instant::now(),
            completion_tx,
        };
        match self.workers[worker_idx].try_send(job) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(Err((
                StatusCode::TOO_MANY_REQUESTS,
                "score queue overloaded".to_string(),
            ))),
            Err(TrySendError::Disconnected(_)) => {
                warn!(worker_idx, "score worker disconnected");
                Err(Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "score worker disconnected".to_string(),
                )))
            }
        }
    }

    pub fn preserves_conn_order(&self) -> bool {
        matches!(self.routing, ScoreRouting::StickyByConn)
    }
}
