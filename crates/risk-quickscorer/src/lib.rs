use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::Cell;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

mod exp_l2_zen4;
mod l2_exp_v1;
mod l2_kernel_zen4;
mod qs_exact;

const MAGIC_SOA: &[u8] = b"L1SOAv1\0";
const MAGIC_BND: &[u8] = b"L1BNDv1\0";
const MAGIC_FEA_V1: &[u8] = b"L1FEATv1\0";
const MAGIC_FEA_V2: &[u8] = b"L1FEATv2\0";
const MAGIC_RNK: &[u8] = b"L1RANKv1\0";
const MAGIC_ORD: &[u8] = b"L1ORDv1\0";
const MAGIC_PRF: &[u8] = b"L1PRFv1\0";
const MAGIC_PRFC: &[u8] = b"L1PRFC1\0";
const MAGIC_HCL: &[u8] = b"L2HCLv1\0";
const MAGIC_RMETA: &[u8] = b"L2RMTv1\0";
const MAGIC_DMETA: &[u8] = b"L2DSPv1\0";

#[derive(Parser, Debug)]
#[command(name = "rust_quickl1")]
#[command(about = "Rust L1 traversal runtime with dual-mode inference")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Infer(InferArgs),
    DispatchL2Exact(DispatchL2ExactArgs),
    QsL2Exact(QsL2ExactArgs),
    QsL2Fast(QsL2FastArgs),
    QsL2PrefixCal(QsL2PrefixCalArgs),
    CompileHotPrefix(CompileHotPrefixArgs),
    RepackL2PrefixV2(RepackL2PrefixV2Args),
    RepackL2RuntimeV2(RepackL2RuntimeV2Args),
    RepackL2RuntimeV3(RepackL2RuntimeV3Args),
    RepackL2ExpV1(RepackL2ExpV1Args),
    RepackL2KernelZen4V1(RepackL2KernelZen4V1Args),
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
enum InferMode {
    RouteFast,
    MarginExact,
    RouteExactReordered,
    MarginExactReordered,
    RouteApprox,
    L2RouteExactReordered,
    L2RouteApprox,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum RankMode {
    RankPack,
    Float,
}

fn default_rank_mode() -> RankMode {
    RankMode::RankPack
}

fn default_zero_string() -> String {
    "zero".to_string()
}

fn parse_prefix_direct_kernel(raw: &str) -> Result<PrefixDirectKernel> {
    match raw {
        "qs" => Ok(PrefixDirectKernel::Qs),
        "hot_exact_96" => Ok(PrefixDirectKernel::HotExact96),
        "hot_exact_128" => Ok(PrefixDirectKernel::HotExact128),
        "hot_exact_192" => Ok(PrefixDirectKernel::HotExact192),
        "hot_exact_256" => Ok(PrefixDirectKernel::HotExact256),
        "hot_exact_384" => Ok(PrefixDirectKernel::HotExact384),
        other => bail!("unsupported direct_kernel: {}", other),
    }
}

fn prefix_direct_kernel_str(kernel: PrefixDirectKernel) -> &'static str {
    match kernel {
        PrefixDirectKernel::Qs => "qs",
        PrefixDirectKernel::HotExact96 => "hot_exact_96",
        PrefixDirectKernel::HotExact128 => "hot_exact_128",
        PrefixDirectKernel::HotExact192 => "hot_exact_192",
        PrefixDirectKernel::HotExact256 => "hot_exact_256",
        PrefixDirectKernel::HotExact384 => "hot_exact_384",
    }
}

fn parse_prefix_certifier_kind(raw: &str) -> Result<PrefixCertifierKind> {
    match raw {
        "table_v1" => Ok(PrefixCertifierKind::TableV1),
        "atlas_v1" => Ok(PrefixCertifierKind::AtlasV1),
        "mlp_v1" => Ok(PrefixCertifierKind::MlpV1),
        other => bail!("unsupported certifier_kind: {}", other),
    }
}

fn prefix_certifier_kind_str(kind: PrefixCertifierKind) -> &'static str {
    match kind {
        PrefixCertifierKind::TableV1 => "table_v1",
        PrefixCertifierKind::AtlasV1 => "atlas_v1",
        PrefixCertifierKind::MlpV1 => "mlp_v1",
    }
}

#[derive(Parser, Debug)]
struct InferArgs {
    #[arg(long)]
    soa: PathBuf,
    #[arg(long)]
    bounds: PathBuf,
    #[arg(long)]
    policy: PathBuf,
    #[arg(long)]
    model_json: Option<PathBuf>,
    #[arg(long)]
    threshold_override: Option<f32>,
    #[arg(long)]
    feat_bin: PathBuf,
    #[arg(long)]
    rank_pack: Option<PathBuf>,
    #[arg(long)]
    tree_order: Option<PathBuf>,
    #[arg(long)]
    approx_policy: Option<PathBuf>,
    #[arg(long)]
    route_meta: Option<PathBuf>,
    #[arg(long)]
    prefix16_pack: Option<PathBuf>,
    #[arg(long)]
    prefix32_pack: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "route-fast")]
    mode: InferMode,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 256)]
    chunk_rows: usize,
    #[arg(long, default_value_t = 4096)]
    parallel_min_rows: usize,
    #[arg(long, default_value_t = 1)]
    bound_check_every: usize,
    #[arg(long, default_value_t = 0.0)]
    eps: f32,
    #[arg(long, default_value_t = 0.0)]
    bound_guard: f32,
    #[arg(long)]
    max_rows: Option<usize>,
    #[arg(long)]
    max_trees: Option<usize>,
    #[arg(long)]
    out_tsv: Option<PathBuf>,
    #[arg(long)]
    stats_json: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    fast_no_stats: bool,
}

#[derive(Parser, Debug)]
struct DispatchL2ExactArgs {
    #[arg(long)]
    feat_bin: PathBuf,
    #[arg(long)]
    dispatch_meta: PathBuf,
    #[arg(long)]
    global_model_dir: PathBuf,
    #[arg(long, default_value = "tau")]
    global_threshold_mode: String,
    #[arg(long)]
    expert_manifest: PathBuf,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 128)]
    chunk_rows: usize,
    #[arg(long, default_value_t = 4096)]
    parallel_min_rows: usize,
    #[arg(long)]
    max_rows: Option<usize>,
    #[arg(long)]
    out_tsv: Option<PathBuf>,
    #[arg(long)]
    stats_json: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct QsL2ExactArgs {
    #[arg(long)]
    qs_pack: PathBuf,
    #[arg(long)]
    feat_bin: PathBuf,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 128)]
    chunk_rows: usize,
    #[arg(long, default_value_t = 4096)]
    parallel_min_rows: usize,
    #[arg(long)]
    max_rows: Option<usize>,
    #[arg(long)]
    out_tsv: Option<PathBuf>,
    #[arg(long)]
    stats_json: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct QsL2FastArgs {
    #[arg(long)]
    qs_pack: PathBuf,
    #[arg(long)]
    feat_bin: PathBuf,
    #[arg(long)]
    route_meta: PathBuf,
    #[arg(long)]
    soa: PathBuf,
    #[arg(long)]
    bounds: PathBuf,
    #[arg(long)]
    tree_order: PathBuf,
    #[arg(long)]
    model_json: PathBuf,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 128)]
    chunk_rows: usize,
    #[arg(long, default_value_t = 4096)]
    parallel_min_rows: usize,
    #[arg(long)]
    max_rows: Option<usize>,
    #[arg(long)]
    out_tsv: Option<PathBuf>,
    #[arg(long)]
    stats_json: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    shadow_only: bool,
}

#[derive(Parser, Debug)]
struct QsL2PrefixCalArgs {
    #[arg(long)]
    bundle_manifest: Option<PathBuf>,
    #[arg(long)]
    qs_pack: Option<PathBuf>,
    #[arg(long)]
    calibration_json: Option<PathBuf>,
    #[arg(long)]
    variant_key: Option<String>,
    #[arg(long)]
    feat_bin: Option<PathBuf>,
    #[arg(long)]
    route_meta: Option<PathBuf>,
    #[arg(long)]
    soa: Option<PathBuf>,
    #[arg(long)]
    bounds: Option<PathBuf>,
    #[arg(long)]
    tree_order: Option<PathBuf>,
    #[arg(long)]
    model_json: Option<PathBuf>,
    #[arg(long)]
    direct_kernel: Option<String>,
    #[arg(long)]
    certifier_kind: Option<String>,
    #[arg(long)]
    certifier_json: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = 128)]
    chunk_rows: usize,
    #[arg(long, default_value_t = 4096)]
    parallel_min_rows: usize,
    #[arg(long)]
    max_rows: Option<usize>,
    #[arg(long)]
    out_tsv: Option<PathBuf>,
    #[arg(long)]
    stats_json: Option<PathBuf>,
    #[arg(long)]
    trace_jsonl: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    shadow_only: bool,
}

#[derive(Parser, Debug)]
struct CompileHotPrefixArgs {
    #[arg(long)]
    prefix_pack: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct RepackL2RuntimeV2Args {
    #[arg(long)]
    qs_pack: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct RepackL2PrefixV2Args {
    #[arg(long)]
    bundle_manifest: PathBuf,
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct RepackL2RuntimeV3Args {
    #[arg(long)]
    bundle_manifest: PathBuf,
}

#[derive(Parser, Debug)]
struct RepackL2ExpV1Args {
    #[arg(long)]
    bundle_manifest: PathBuf,
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct RepackL2KernelZen4V1Args {
    #[arg(long)]
    bundle_manifest: PathBuf,
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Debug)]
struct SoaModel {
    n_features: usize,
    n_trees: usize,
    node_count: usize,
    tree_roots: Vec<u32>,
    fidx: Vec<i32>,
    left: Vec<u32>,
    right: Vec<u32>,
    missing: Vec<u32>,
    thr: Vec<f32>,
    leaf: Vec<f32>,
    is_leaf: Vec<u8>,
}

#[derive(Debug)]
struct Bounds {
    n_trees: usize,
    suffix_min: Vec<f32>,
    suffix_max: Vec<f32>,
}

#[derive(Debug)]
struct TreeOrder {
    n_trees: usize,
    order: Vec<u32>,
    suffix_min: Vec<f32>,
    suffix_max: Vec<f32>,
}

#[derive(Debug)]
struct PrefixPack {
    n_hot_features: usize,
    n_trees: usize,
    node_count: usize,
    hot_global_fidx: Vec<u32>,
    tree_roots: Vec<u32>,
    fidx: Vec<i32>,
    left: Vec<u32>,
    right: Vec<u32>,
    missing: Vec<u32>,
    thr: Vec<f32>,
    leaf: Vec<f32>,
    is_leaf: Vec<u8>,
}

const HOT_COMPILED_NODE_LEAF: u16 = 1 << 15;
const HOT_COMPILED_NODE_FIDX_MASK: u16 = HOT_COMPILED_NODE_LEAF - 1;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct HotCompiledPrefixNode {
    thr: f32,
    leaf: f32,
    left: u16,
    right: u16,
    missing: u16,
    hot_fidx_flags: u16,
}

impl HotCompiledPrefixNode {
    #[inline(always)]
    fn is_leaf(self) -> bool {
        (self.hot_fidx_flags & HOT_COMPILED_NODE_LEAF) != 0
    }

    #[inline(always)]
    fn hot_fidx(self) -> usize {
        (self.hot_fidx_flags & HOT_COMPILED_NODE_FIDX_MASK) as usize
    }
}

#[derive(Debug)]
struct HotCompiledPrefixPack {
    n_hot_features: usize,
    n_trees: usize,
    hot_global_fidx: Vec<u32>,
    tree_roots: Vec<u16>,
    nodes: Vec<HotCompiledPrefixNode>,
}

#[derive(Debug)]
struct RouteMeta {
    n_rows: usize,
    ids: Vec<i64>,
    fold_id: Vec<i32>,
    tau_used: Vec<f32>,
    active: Vec<u8>,
    exact_positive: Vec<u8>,
}

#[derive(Debug)]
struct DispatchMeta {
    n_rows: usize,
    ids: Vec<i64>,
    slot_id: Vec<i32>,
    tau_used: Vec<f32>,
    active: Vec<u8>,
    exact_positive: Vec<u8>,
}

#[derive(Debug, Deserialize)]
struct SegmentExpertsManifest {
    #[serde(default)]
    selected_experts: Vec<SegmentExpertEntry>,
}

#[derive(Debug, Deserialize)]
struct SegmentExpertEntry {
    seg_key: String,
    model_dir: PathBuf,
    #[serde(default = "default_zero_string")]
    threshold_mode: String,
}

#[derive(Debug, Clone, Copy)]
enum DispatchThresholdMode {
    Tau,
    Zero,
}

#[derive(Debug)]
struct LoadedDispatchSlot {
    slot: i32,
    label: String,
    threshold_mode: DispatchThresholdMode,
    model: SoaModel,
    plan: TreeOrder,
    base_score: f32,
}

#[derive(Debug, Serialize)]
struct DispatchSlotStats {
    slot: i32,
    label: String,
    rows: usize,
    rows_per_sec: f64,
    elapsed_sec: f64,
    avg_visited_trees: f64,
    p99_visited_trees: i32,
    route_reject_rate: f64,
}

#[derive(Debug, Serialize)]
struct DispatchStats {
    n_rows: usize,
    threads: usize,
    parallel_enabled: bool,
    feature_format: String,
    nan_free: bool,
    rows_per_sec: f64,
    elapsed_sec: f64,
    avg_visited_trees: f64,
    p50_visited_trees: i32,
    p90_visited_trees: i32,
    p99_visited_trees: i32,
    route_reject_rate: f64,
    route_active_rate: f64,
    rss_peak_mb: f64,
    slots_used: usize,
    slot_details: Vec<DispatchSlotStats>,
    visit_hist: Vec<u64>,
}

#[derive(Debug, Serialize)]
struct QsExactStats {
    n_rows: usize,
    n_features: usize,
    n_trees: usize,
    n_blocks: usize,
    threads: usize,
    parallel_enabled: bool,
    feature_format: String,
    nan_free: bool,
    elapsed_sec: f64,
    rows_per_sec: f64,
    rss_peak_mb: f64,
    avg_blocks_per_row: f64,
    avg_blocks_per_tree: f64,
    resolved_early_rate: f64,
}

#[derive(Debug, Serialize)]
struct QsFastStats {
    n_rows: usize,
    n_features: usize,
    n_trees: usize,
    n_blocks: usize,
    threads: usize,
    parallel_enabled: bool,
    feature_format: String,
    nan_free: bool,
    shadow_only: bool,
    elapsed_sec: f64,
    rows_per_sec: f64,
    rss_peak_mb: f64,
    route_active_rate: f64,
    shadow_reject_rate: f64,
    final_reject_rate: f64,
    direct_reject_rate: f64,
    direct_refer_rate: f64,
    fallback_rate: f64,
    avg_blocks_per_row: f64,
    avg_blocks_per_tree: f64,
    resolved_early_rate: f64,
    exact_avg_visited_trees: f64,
    exact_p99_visited_trees: i32,
    exact_visit_hist: Vec<u64>,
}

#[derive(Debug, Deserialize)]
struct PrefixCalFile {
    format: String,
    checkpoints: Vec<usize>,
    variants: Vec<PrefixCalVariant>,
}

#[derive(Debug, Deserialize)]
struct PrefixCalVariant {
    key: String,
    ref_q: f32,
    rej_q: f32,
    tables: Vec<PrefixCheckpointTable>,
}

#[derive(Debug, Deserialize)]
struct PrefixCheckpointTable {
    checkpoint: usize,
    gap_edges: Vec<f32>,
    global_ref_hi: Vec<f32>,
    global_rej_lo: Vec<f32>,
    #[serde(default)]
    tau_edges: Vec<f32>,
    fold_tables: Vec<PrefixFoldTable>,
    #[serde(default)]
    tau_fold_tables: Vec<PrefixTauFoldTable>,
}

#[derive(Debug, Deserialize)]
struct PrefixFoldTable {
    fold_id: i32,
    ref_hi: Vec<f32>,
    rej_lo: Vec<f32>,
    #[serde(default)]
    counts: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct PrefixTauFoldTable {
    fold_id: i32,
    tau_bin: usize,
    ref_hi: Vec<f32>,
    rej_lo: Vec<f32>,
    #[serde(default)]
    counts: Vec<u32>,
}

#[derive(Debug)]
struct LoadedPrefixFold {
    ref_hi: Vec<f32>,
    rej_lo: Vec<f32>,
}

#[derive(Debug, Clone)]
struct LoadedPrefixAtlasCentroids16 {
    w0: [f32; 16],
    w1: [f32; 16],
    w2: [f32; 16],
    w3: [f32; 16],
    bias: [f32; 16],
}

#[derive(Debug, Clone)]
struct LoadedPrefixCheckpoint {
    checkpoint: usize,
    gap_edges: Vec<f32>,
    global_ref_hi: Vec<f32>,
    global_rej_lo: Vec<f32>,
    tau_edges: Vec<f32>,
    fold_ids: Vec<i32>,
    fold_lut_base: i32,
    fold_slot_lut: Vec<i16>,
    fold_ref_hi: Vec<f32>,
    fold_rej_lo: Vec<f32>,
    tau_bin_count: usize,
    tau_fold_mask: Vec<u8>,
    tau_ref_hi: Vec<f32>,
    tau_rej_lo: Vec<f32>,
}

#[derive(Debug, Clone)]
struct LoadedPrefixCal {
    variant_key: String,
    ref_q: f32,
    rej_q: f32,
    checkpoints: Vec<usize>,
    tables: Vec<LoadedPrefixCheckpoint>,
}

#[derive(Debug, Deserialize)]
struct PrefixCalBundleManifest {
    format: String,
    pack_path: Option<String>,
    calibration_json: Option<String>,
    feat_bin: Option<String>,
    route_meta: Option<String>,
    soa_bin: Option<String>,
    bounds_bin: Option<String>,
    tree_order_bin: Option<String>,
    model_json: Option<String>,
    selected_variant: Option<String>,
    selected_exact_variant: Option<String>,
    selected_lossy_variant: Option<String>,
    direct_kernel: Option<String>,
    certifier_kind: Option<String>,
    certifier_json: Option<String>,
    atlas_bin: Option<String>,
    atlas_format: Option<String>,
    v4_mode: Option<String>,
    order_router_json: Option<String>,
    packet_manifest_json: Option<String>,
    anchor_manifest_json: Option<String>,
    rescue_manifest_json: Option<String>,
    directional_calibration_json: Option<String>,
    rescue_router_json: Option<String>,
    compiled_prefix_json: Option<String>,
    lossy_sidecar_json: Option<String>,
    hot_exact_prefix_limit: Option<usize>,
    telemetry_schema_version: Option<u32>,
    hot_exact_prefix_96_pack: Option<String>,
    hot_exact_prefix_128_pack: Option<String>,
    hot_exact_prefix_192_pack: Option<String>,
    hot_exact_prefix_256_pack: Option<String>,
    hot_exact_prefix_384_pack: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PrefixOrderRouterFile {
    format: String,
    tau_edges: Vec<f32>,
    bundles: Vec<PrefixOrderRouteEntry>,
}

#[derive(Debug, Deserialize)]
struct PrefixOrderRouteEntry {
    tau_bin: usize,
    bundle_manifest: String,
    #[serde(default)]
    label: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PrefixPacketManifestFile {
    format: String,
    bundle_manifest: String,
    packet_family: String,
    #[serde(default)]
    mixed_frac: Option<f32>,
    #[serde(default)]
    scout_frac: Option<f32>,
    #[serde(default)]
    scheduler_policy_json: Option<String>,
    packet_size: usize,
    n_packets: usize,
    max_steps: usize,
    packet_summary_tsv: String,
    #[serde(default)]
    compiled_packet_packs: Vec<PrefixPacketPackEntry>,
}

#[derive(Debug, Deserialize)]
struct PrefixPacketPackEntry {
    packet_id: usize,
    pack_path: String,
}

#[derive(Debug, Deserialize)]
struct PrefixAnchorRescueManifestFile {
    format: String,
    anchor_limit: usize,
    rescue_limit: usize,
    reject_bundle_manifest: String,
    refer_bundle_manifest: String,
}

#[derive(Debug, Deserialize)]
struct RescueRouterFile {
    format: String,
    tau_edges: Vec<f32>,
    gap_edges: Vec<f32>,
    rows: Vec<RescueRouterRow>,
    #[serde(default)]
    defaults: Vec<RescueRouterDefaultRow>,
}

#[derive(Debug, Deserialize)]
struct RescueRouterRow {
    #[serde(default)]
    anchor_checkpoint: usize,
    tau_bin: usize,
    fold_id: i32,
    shadow_side: i8,
    gap_bin: usize,
    action: String,
    #[serde(default)]
    row_count: usize,
}

#[derive(Debug, Deserialize)]
struct RescueRouterDefaultRow {
    #[serde(default)]
    anchor_checkpoint: usize,
    tau_bin: usize,
    shadow_side: i8,
    action: String,
}

#[derive(Debug, Deserialize)]
struct PacketSchedulerPolicyFile {
    format: String,
    packet_manifest_json: String,
    packet_family: String,
    packet_size: usize,
    n_packets: usize,
    max_steps: usize,
    tau_bin_count: usize,
    gap_regime_count: usize,
    delta_regime_count: usize,
    ranking_top_k: usize,
    #[serde(default)]
    oracle_depth: Option<usize>,
    #[serde(default)]
    oracle_top_k: Option<usize>,
    #[serde(default)]
    oracle_second_weight: Option<f32>,
    tau_edges: Vec<f32>,
    gap_edges_by_step: Vec<Vec<f32>>,
    delta_edges_by_step: Vec<Vec<f32>>,
    policy_rows: Vec<PacketSchedulerPolicyRow>,
}

#[derive(Debug, Deserialize)]
struct PacketSchedulerPolicyRow {
    step: usize,
    tau_bin: usize,
    gap_regime: usize,
    delta_regime: usize,
    row_count: usize,
    top_packet: usize,
    #[serde(default)]
    top_score: Option<f32>,
    ranking: Vec<usize>,
    #[serde(default)]
    oracle_depth: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefixV4Mode {
    MultiOrderV1,
    PacketSchedulerV1,
    AnchorRescueV1,
    AnchorRescueLossyV1,
    CompiledPrefixV1,
    ConformalBandV1,
}

include!("prefix.rs");
include!("prefix_v2.rs");
include!("l2_kernel_zen4/mod.rs");
include!("prefix_exec.rs");
include!("l1.rs");
include!("commands.rs");
include!("runtime.rs");
