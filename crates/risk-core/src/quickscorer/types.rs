use crate::schema::Decision;
use serde::Serialize;

#[derive(Debug, Clone, Copy, Serialize)]
pub struct QuickRouteMeta {
    pub row_idx: u32,
    pub transaction_id: u64,
    pub fold_id: i32,
    pub seg_prod_amtbin: u32,
    pub l2_tau_used: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuickDebugInfo {
    pub backend: &'static str,
    pub l1_dim: usize,
    pub l2_dim: usize,
    pub l1_threshold: f32,
    pub l2_default_fold: i32,
    pub l2_gb_target: String,
    pub l2_segmented: bool,
    pub l2_seg_cols: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct QuickPredictOutput {
    pub l1_score: f32,
    pub l2_score: Option<f32>,
    pub final_score: f32,
    pub decision: Decision,
    pub used_l2: bool,
    pub feature_us: u64,
    pub l1_us: u64,
    pub l2_us: u64,
    pub router_us: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct QuickL1PredictOutput {
    pub l1_score: f32,
    pub passed: bool,
    pub l1_us: u64,
    pub router_us: u64,
}
