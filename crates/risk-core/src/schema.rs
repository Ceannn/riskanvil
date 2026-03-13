use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Allow,
    ManualReview,
    Deny,
    /// Conservative fallback when time or budget runs out.
    DegradeAllow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasonItem {
    pub signal: String,
    pub value: f64,
    pub baseline_p95: f64,
    pub direction: String, // "risk_up" / "risk_down" / "info"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreResponse {
    pub trace_id: Uuid,
    pub score: f64,
    pub decision: Decision,
    pub reason: Vec<ReasonItem>,
    /// Per-stage timings in microseconds.
    pub timings_us: TimingsUs,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimingsUs {
    pub parse: u64,
    pub feature: u64,
    pub router: u64,
    #[serde(default, alias = "xgb")]
    pub l1: u64,
    pub l2: u64,
    pub serialize: u64,
}
