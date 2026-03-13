use serde::{Deserialize, Serialize};

/// Runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// End-to-end budget used by router deadline and fallback logic.
    pub slo_p99_ms: u64,

    /// L1 uncertainty band. Samples in (low, high) may escalate to deeper stages.
    pub l1_uncertain_low: f64,
    pub l1_uncertain_high: f64,

    /// Decision thresholds.
    pub deny_threshold: f64,
    pub review_threshold: f64,

    /// Feature store window sizes, in seconds.
    pub win_60s: u64,
    pub win_300s: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            slo_p99_ms: 10,
            l1_uncertain_low: 0.35,
            l1_uncertain_high: 0.65,
            deny_threshold: 0.85,
            review_threshold: 0.65,
            win_60s: 60,
            win_300s: 300,
        }
    }
}
