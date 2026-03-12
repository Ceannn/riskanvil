use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tracing::info;

#[derive(Clone, Copy, Debug, Default)]
pub struct StageTimingsUs {
    pub parse: u32,
    pub read_to_submit: u32,
    pub first_byte_to_header_done: u32,
    pub header_done_to_credit_acquired: u32,
    pub credit_acquired_to_body_done: u32,
    pub body_done_to_submit: u32,
    pub queue_wait: u32,
    pub score: u32,
    pub completion_wait: u32,
    pub write: u32,
    pub total_residency: u32,
    pub inflight_depth: u16,
    pub body_fill_in_progress: u8,
    pub pending_complete_not_submitted: u8,
}

pub fn duration_to_us(d: Duration) -> u32 {
    d.as_micros().min(u128::from(u32::MAX)) as u32
}

pub struct StageSampler {
    every: u64,
    seen: AtomicU64,
}

impl StageSampler {
    pub fn new(every: u64) -> Self {
        Self {
            every,
            seen: AtomicU64::new(0),
        }
    }

    pub fn record(&self, transport: &'static str, conn_id: u64, seq: u64, timings: StageTimingsUs) {
        if self.every == 0 {
            return;
        }
        let idx = self.seen.fetch_add(1, Ordering::Relaxed) + 1;
        if idx % self.every != 0 {
            return;
        }
        info!(
            transport,
            conn_id,
            seq,
            parse_us = timings.parse,
            read_to_submit_us = timings.read_to_submit,
            first_byte_to_header_done_us = timings.first_byte_to_header_done,
            header_done_to_credit_acquired_us = timings.header_done_to_credit_acquired,
            credit_acquired_to_body_done_us = timings.credit_acquired_to_body_done,
            body_done_to_submit_us = timings.body_done_to_submit,
            queue_wait_us = timings.queue_wait,
            score_us = timings.score,
            completion_wait_us = timings.completion_wait,
            write_us = timings.write,
            total_residency_us = timings.total_residency,
            inflight_depth = timings.inflight_depth,
            body_fill_in_progress = timings.body_fill_in_progress,
            pending_complete_not_submitted = timings.pending_complete_not_submitted,
            "batchplane stage sample"
        );
    }
}
