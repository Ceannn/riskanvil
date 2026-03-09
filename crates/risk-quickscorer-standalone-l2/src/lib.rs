use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::Cell;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

pub mod qs_exact;

const MAGIC_SOA: &[u8] = b"L1SOAv1\0";
const MAGIC_BND: &[u8] = b"L1BNDv1\0";
const MAGIC_FEA_V1: &[u8] = b"L1FEATv1\0";
const MAGIC_FEA_V2: &[u8] = b"L1FEATv2\0";
const MAGIC_RNK: &[u8] = b"L1RANKv1\0";
const MAGIC_ORD: &[u8] = b"L1ORDv1\0";
const MAGIC_PRF: &[u8] = b"L1PRFv1\0";
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

#[derive(Parser, Debug, Clone)]
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

#[derive(Debug)]
struct LoadedPrefixAtlasCentroids16 {
    w0: [f32; 16],
    w1: [f32; 16],
    w2: [f32; 16],
    w3: [f32; 16],
    bias: [f32; 16],
}

#[derive(Debug)]
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

#[derive(Debug)]
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

fn parse_prefix_v4_mode(raw: &str) -> Result<PrefixV4Mode> {
    match raw {
        "multi_order_v1" => Ok(PrefixV4Mode::MultiOrderV1),
        "packet_scheduler_v1" => Ok(PrefixV4Mode::PacketSchedulerV1),
        "anchor_rescue_v1" => Ok(PrefixV4Mode::AnchorRescueV1),
        "anchor_rescue_lossy_v1" => Ok(PrefixV4Mode::AnchorRescueLossyV1),
        "compiled_prefix_v1" => Ok(PrefixV4Mode::CompiledPrefixV1),
        "conformal_band_v1" => Ok(PrefixV4Mode::ConformalBandV1),
        other => bail!("unsupported v4_mode: {}", other),
    }
}

#[derive(Debug)]
struct LoadedPrefixRuntime {
    pack: qs_exact::QsPack,
    calibration: LoadedPrefixCal,
    atlas_certifier: Option<LoadedPrefixAtlas>,
    mlp_certifier: Option<LoadedMlpCertifier>,
    hot_pack: Option<PrefixPack>,
    compiled_hot_pack: Option<HotCompiledPrefixPack>,
    hot_checkpoint_layout: Option<HotCheckpointLayout>,
    plan: TreeOrder,
    direct_kernel: PrefixDirectKernel,
    certifier_kind: PrefixCertifierKind,
    hot_exact_prefix_limit: usize,
    telemetry_schema_version: u32,
    packet_scheduler: Option<LoadedPacketScheduler>,
    anchor_rescue: Option<Box<LoadedAnchorRescueRuntime>>,
}

#[derive(Debug)]
struct LoadedPrefixOrderRoute {
    tau_bin: usize,
    label: String,
    runtime: LoadedPrefixRuntime,
}

#[derive(Debug, Default)]
struct HotCheckpointLayout {
    hot_values: Vec<usize>,
    hot_indices: Vec<usize>,
    late_values: Vec<usize>,
    late_indices: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefixDirectKernel {
    Qs,
    HotExact96,
    HotExact128,
    HotExact192,
    HotExact256,
    HotExact384,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefixCertifierKind {
    TableV1,
    AtlasV1,
    MlpV1,
}

#[derive(Debug)]
struct LoadedPacket {
    packet_id: usize,
    bank_id: i8,
    tree_indices: Vec<u32>,
    tree_count: usize,
    block_cost: f32,
    compiled_pack: Option<PrefixPack>,
}

#[derive(Debug, Deserialize)]
struct PrefixAtlasFile {
    format: String,
    feature_schema_version: u32,
    variant_key: String,
    tau_edges: Vec<f32>,
    cluster_count: usize,
    min_cell_count: u32,
    safe_ref_max_reject_rate: f32,
    safe_rej_min_reject_rate: f32,
    checkpoints: Vec<PrefixAtlasCheckpointFile>,
}

#[derive(Debug, Deserialize)]
struct PrefixAtlasCheckpointFile {
    checkpoint: usize,
    #[serde(default)]
    gap_edges: Vec<f32>,
    feature_mean: Vec<f32>,
    feature_inv_std: Vec<f32>,
    centroids: Vec<Vec<f32>>,
    cells: Vec<PrefixAtlasCellFile>,
}

#[derive(Debug, Deserialize)]
struct PrefixAtlasCellFile {
    tau_bin: usize,
    cluster_id: usize,
    reject_rate: Vec<f32>,
    counts: Vec<u32>,
}

#[derive(Debug)]
struct LoadedPrefixAtlasCheckpoint {
    checkpoint: usize,
    gap_edges: Vec<f32>,
    shared_gap_edges: bool,
    feature_mean: [f32; 4],
    feature_inv_std: [f32; 4],
    cluster_count: usize,
    tau_bin_count: usize,
    centroid_w0: Vec<f32>,
    centroid_w1: Vec<f32>,
    centroid_w2: Vec<f32>,
    centroid_w3: Vec<f32>,
    centroid_bias: Vec<f32>,
    centroids16: Option<Box<LoadedPrefixAtlasCentroids16>>,
    decision_grid: Vec<u8>,
}

#[derive(Debug)]
struct LoadedPrefixAtlas {
    feature_schema_version: u32,
    variant_key: String,
    tau_edges: Vec<f32>,
    cluster_count: usize,
    min_cell_count: u32,
    safe_ref_max_reject_rate: f32,
    safe_rej_min_reject_rate: f32,
    checkpoints: Vec<LoadedPrefixAtlasCheckpoint>,
}

#[derive(Debug)]
struct LoadedPacketPolicy {
    packet_family: String,
    packet_size: usize,
    n_packets: usize,
    max_steps: usize,
    tau_edges: Vec<f32>,
    gap_edges_by_step: Vec<Vec<f32>>,
    delta_edges_by_step: Vec<Vec<f32>>,
    state_rankings: HashMap<(usize, usize, usize, usize), Vec<u16>>,
    step_tau_defaults: HashMap<(usize, usize), Vec<u16>>,
    step_defaults: HashMap<usize, Vec<u16>>,
}

#[derive(Debug)]
struct LoadedPacketScheduler {
    packet_family: String,
    packet_size: usize,
    max_steps: usize,
    packets: Vec<LoadedPacket>,
    policy: LoadedPacketPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RescueAction {
    Fallback,
    RejectRescue,
    ReferRescue,
}

#[derive(Debug)]
struct LoadedRescueRouter {
    tau_edges: Vec<f32>,
    gap_edges: Vec<f32>,
    actions: HashMap<(usize, usize, i32, i8, usize), RescueAction>,
    defaults: HashMap<(usize, usize, i8), RescueAction>,
}

#[derive(Debug)]
struct LoadedAnchorRescueRuntime {
    reject_rescue: Box<LoadedPrefixRuntime>,
    refer_rescue: Box<LoadedPrefixRuntime>,
    router: LoadedRescueRouter,
    anchor_limit: usize,
    rescue_limit: usize,
}

#[derive(Debug)]
struct ResolvedPrefixCalBundle {
    qs_pack: PathBuf,
    calibration_json: PathBuf,
    variant_key: String,
    feat_bin: PathBuf,
    route_meta: PathBuf,
    soa: PathBuf,
    bounds: PathBuf,
    tree_order: PathBuf,
    model_json: PathBuf,
    direct_kernel: PrefixDirectKernel,
    certifier_kind: PrefixCertifierKind,
    certifier_json: Option<PathBuf>,
    atlas_bin: Option<PathBuf>,
    atlas_format: Option<String>,
    hot_exact_prefix_limit: Option<usize>,
    telemetry_schema_version: u32,
    hot_exact_prefix_pack: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct QsPrefixCalStats {
    variant_key: String,
    ref_q: f32,
    rej_q: f32,
    checkpoints: Vec<usize>,
    n_rows: usize,
    n_features: usize,
    n_trees: usize,
    n_blocks: usize,
    threads: usize,
    parallel_enabled: bool,
    feature_format: String,
    nan_free: bool,
    shadow_only: bool,
    v4_mode: String,
    direct_kernel: String,
    certifier_kind: String,
    hot_exact_prefix_limit: usize,
    telemetry_schema_version: u32,
    elapsed_sec: f64,
    rows_per_sec: f64,
    rss_peak_mb: f64,
    route_active_rate: f64,
    shadow_reject_rate: f64,
    final_reject_rate: f64,
    direct_reject_rate: f64,
    direct_refer_rate: f64,
    direct_decision_rate: f64,
    fallback_rate: f64,
    avg_blocks_per_row: f64,
    avg_blocks_per_tree: f64,
    avg_prefix_trees_used: f64,
    resolved_early_rate: f64,
    exact_avg_visited_trees: f64,
    exact_p99_visited_trees: i32,
    checkpoint_exit_counts: Vec<u64>,
    direct_rows_by_checkpoint: Vec<u64>,
    shadow_mismatch_by_checkpoint: Vec<u64>,
    fallback_entry_checkpoint_counts: Vec<u64>,
    exact_visit_hist: Vec<u64>,
    router_choice_counts: Vec<u64>,
    router_tau_bin_counts: Vec<u64>,
    anchor_exit_counts: Vec<u64>,
    rescue_route_counts: Vec<u64>,
    rescue_exit_counts: Vec<u64>,
    tail_continuation_rows: u64,
    shadow_mismatch_by_stage: Vec<u64>,
}

#[derive(Debug, Deserialize)]
struct MlpCertifierFile {
    format: String,
    feature_schema_version: u32,
    variant_key: String,
    checkpoint_values: Vec<usize>,
    fold_values: Vec<i32>,
    tau_edges: Vec<f32>,
    checkpoint_limit: usize,
    mean: Vec<f32>,
    inv_std: Vec<f32>,
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: f32,
    thresholds: Vec<MlpCheckpointThreshold>,
}

#[derive(Debug, Deserialize)]
struct MlpCheckpointThreshold {
    checkpoint: usize,
    tau_ref: f32,
    tau_rej: f32,
}

#[derive(Debug)]
struct LoadedMlpCheckpoint {
    tau_ref: f32,
    tau_rej: f32,
}

#[derive(Debug)]
struct LoadedMlpCertifier {
    feature_schema_version: u32,
    variant_key: String,
    checkpoint_values: Vec<usize>,
    fold_values: Vec<i32>,
    tau_edges: Vec<f32>,
    checkpoint_limit: usize,
    mean: Vec<f32>,
    inv_std: Vec<f32>,
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<Vec<f32>>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: f32,
    threshold_by_checkpoint: HashMap<usize, LoadedMlpCheckpoint>,
}

#[derive(Debug)]
struct DispatchSlotRun {
    scores: Vec<f32>,
    passes: Vec<bool>,
    visits: Vec<i32>,
    stats: DispatchSlotStats,
    visit_hist: Vec<u64>,
    reject_cnt: usize,
}

#[derive(Debug, Deserialize)]
struct ApproxPolicy {
    #[serde(default)]
    checkpoints: Vec<usize>,
    #[serde(default)]
    tau_ref: Vec<Option<f32>>,
    #[serde(default)]
    tau_pass: Vec<Option<f32>>,
    #[serde(default)]
    tau_reject: Vec<Option<f32>>,
    #[serde(default)]
    used_checkpoints: Vec<usize>,
    #[serde(default)]
    used_tau_ref: Vec<Option<f32>>,
    #[serde(default)]
    used_tau_pass: Vec<Option<f32>>,
    #[serde(default)]
    used_tau_reject: Vec<Option<f32>>,
    #[serde(default)]
    k_hot: usize,
    #[serde(default = "default_rank_mode")]
    rank_mode: RankMode,
    #[serde(default)]
    checkpoint_exit_counts: Vec<u64>,
    #[serde(default)]
    #[allow(dead_code)]
    compacted: bool,
    fallback_mode: String,
    calibration_manifest: Value,
}

#[derive(Debug)]
struct RankPack {
    n_features: usize,
    node_thr_rank: Vec<i32>,
    feat_offset: Vec<u32>,
    feat_count: Vec<u32>,
    threshold_values: Vec<f32>,
}

#[derive(Debug)]
enum FeatureStorage {
    Owned {
        x: Vec<f32>,
        ids: Vec<i64>,
        y: Vec<u8>,
    },
    Mmap {
        mmap: Mmap,
        x_offset: usize,
        ids_offset: usize,
        y_offset: usize,
    },
}

#[derive(Debug)]
struct FeatureBatch {
    n_rows: usize,
    n_cols: usize,
    storage: FeatureStorage,
    format_tag: String,
    nan_free: bool,
}

#[derive(Serialize)]
struct InferStats {
    n_rows: usize,
    n_trees: usize,
    threshold: f32,
    base_score: f32,
    mode: InferMode,
    threads: usize,
    parallel_enabled: bool,
    rank_pack_enabled: bool,
    tree_order_enabled: bool,
    approx_policy_enabled: bool,
    feature_format: String,
    nan_free: bool,
    fast_no_stats: bool,
    bound_check_every: usize,
    max_trees: usize,
    eps: f32,
    bound_guard: f32,
    route_pass_rate: f64,
    route_active_rate: f64,
    avg_visited_trees: f64,
    p50_visited_trees: i32,
    p90_visited_trees: i32,
    p99_visited_trees: i32,
    visit_hist: Vec<u64>,
    full_eval_rate: f64,
    approx_coverage_rate: f64,
    approx_pass_rate: f64,
    approx_refer_rate: f64,
    approx_fallback_rate: f64,
    approx_k_hot: usize,
    approx_rank_mode: String,
    approx_checkpoint_exit_counts: Vec<u64>,
    prefix_pack_kind: String,
    elapsed_sec: f64,
    rows_per_sec: f64,
    rss_peak_mb: f64,
}

#[derive(Clone, Copy, Debug)]
struct RowMeta {
    approx_pass: bool,
    approx_refer: bool,
    approx_checkpoint_idx: i32,
}

impl Default for RowMeta {
    fn default() -> Self {
        Self {
            approx_pass: false,
            approx_refer: false,
            approx_checkpoint_idx: -1,
        }
    }
}

#[derive(Default)]
struct FastAgg {
    pass_cnt: usize,
    active_cnt: usize,
    visit_sum: u64,
    full_cnt: usize,
    approx_pass_cnt: usize,
    approx_ref_cnt: usize,
    checkpoint_counts: Vec<u64>,
    visit_hist: Vec<u64>,
}

fn le_u32(buf: &[u8], off: &mut usize) -> Result<u32> {
    if *off + 4 > buf.len() {
        bail!("buffer underflow for u32");
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[*off..*off + 4]);
    *off += 4;
    Ok(u32::from_le_bytes(b))
}

fn le_i32(buf: &[u8], off: &mut usize) -> Result<i32> {
    Ok(le_u32(buf, off)? as i32)
}

fn le_u64(buf: &[u8], off: &mut usize) -> Result<u64> {
    if *off + 8 > buf.len() {
        bail!("buffer underflow for u64");
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[*off..*off + 8]);
    *off += 8;
    Ok(u64::from_le_bytes(b))
}

fn le_f32(buf: &[u8], off: &mut usize) -> Result<f32> {
    if *off + 4 > buf.len() {
        bail!("buffer underflow for f32");
    }
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[*off..*off + 4]);
    *off += 4;
    Ok(f32::from_le_bytes(b))
}

fn le_i64(buf: &[u8], off: &mut usize) -> Result<i64> {
    if *off + 8 > buf.len() {
        bail!("buffer underflow for i64");
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&buf[*off..*off + 8]);
    *off += 8;
    Ok(i64::from_le_bytes(b))
}

fn le_bytes<'a>(buf: &'a [u8], off: &mut usize, len: usize) -> Result<&'a [u8]> {
    if *off + len > buf.len() {
        bail!("buffer underflow for bytes len={}", len);
    }
    let out = &buf[*off..*off + len];
    *off += len;
    Ok(out)
}

fn le_string(buf: &[u8], off: &mut usize) -> Result<String> {
    let len = le_u32(buf, off)? as usize;
    let raw = le_bytes(buf, off, len)?;
    Ok(std::str::from_utf8(raw)
        .context("invalid utf-8 string in binary blob")?
        .to_string())
}

fn le_f32_vec(buf: &[u8], off: &mut usize, len: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push(le_f32(buf, off)?);
    }
    Ok(out)
}

fn mmap_readonly(path: &PathBuf) -> Result<Mmap> {
    let file = File::open(path).with_context(|| format!("open mmap file failed: {}", path.display()))?;
    let mmap =
        unsafe { Mmap::map(&file) }.with_context(|| format!("mmap failed: {}", path.display()))?;
    Ok(mmap)
}

fn load_soa(path: &PathBuf) -> Result<SoaModel> {
    let buf = fs::read(path).with_context(|| format!("read soa failed: {}", path.display()))?;
    if buf.len() < MAGIC_SOA.len() + 12 {
        bail!("soa too short");
    }
    if &buf[0..MAGIC_SOA.len()] != MAGIC_SOA {
        bail!("invalid soa magic");
    }
    let mut off = MAGIC_SOA.len();
    let n_features = le_u32(&buf, &mut off)? as usize;
    let n_trees = le_u32(&buf, &mut off)? as usize;
    let n_nodes = le_u32(&buf, &mut off)? as usize;

    let mut tree_offsets = Vec::with_capacity(n_trees + 1);
    for _ in 0..(n_trees + 1) {
        tree_offsets.push(le_u32(&buf, &mut off)?);
    }

    let mut fidx = Vec::with_capacity(n_nodes);
    let mut left = Vec::with_capacity(n_nodes);
    let mut right = Vec::with_capacity(n_nodes);
    let mut missing = Vec::with_capacity(n_nodes);
    let mut thr = Vec::with_capacity(n_nodes);
    let mut leaf = Vec::with_capacity(n_nodes);
    let mut is_leaf = Vec::with_capacity(n_nodes);

    for _ in 0..n_nodes {
        fidx.push(le_i32(&buf, &mut off)?);
        left.push(le_u32(&buf, &mut off)?);
        right.push(le_u32(&buf, &mut off)?);
        missing.push(le_u32(&buf, &mut off)?);
        thr.push(le_f32(&buf, &mut off)?);
        leaf.push(le_f32(&buf, &mut off)?);
        if off + 4 > buf.len() {
            bail!("soa truncated at node flags");
        }
        is_leaf.push(buf[off]);
        off += 4;
    }

    Ok(SoaModel {
        n_features,
        n_trees,
        node_count: n_nodes,
        tree_roots: tree_offsets[..n_trees].to_vec(),
        fidx,
        left,
        right,
        missing,
        thr,
        leaf,
        is_leaf,
    })
}

fn load_bounds(path: &PathBuf) -> Result<Bounds> {
    let buf = fs::read(path).with_context(|| format!("read bounds failed: {}", path.display()))?;
    if buf.len() < MAGIC_BND.len() + 4 {
        bail!("bounds too short");
    }
    if &buf[0..MAGIC_BND.len()] != MAGIC_BND {
        bail!("invalid bounds magic");
    }
    let mut off = MAGIC_BND.len();
    let n_trees = le_u32(&buf, &mut off)? as usize;

    for _ in 0..n_trees {
        let _ = le_f32(&buf, &mut off)?;
    }
    for _ in 0..n_trees {
        let _ = le_f32(&buf, &mut off)?;
    }

    let mut suffix_min = Vec::with_capacity(n_trees + 1);
    for _ in 0..(n_trees + 1) {
        suffix_min.push(le_f32(&buf, &mut off)?);
    }
    let mut suffix_max = Vec::with_capacity(n_trees + 1);
    for _ in 0..(n_trees + 1) {
        suffix_max.push(le_f32(&buf, &mut off)?);
    }

    Ok(Bounds {
        n_trees,
        suffix_min,
        suffix_max,
    })
}

fn load_tree_order(path: &PathBuf) -> Result<TreeOrder> {
    let buf =
        fs::read(path).with_context(|| format!("read tree order failed: {}", path.display()))?;
    if buf.len() < MAGIC_ORD.len() + 4 {
        bail!("tree order too short");
    }
    if &buf[0..MAGIC_ORD.len()] != MAGIC_ORD {
        bail!("invalid tree order magic");
    }
    let mut off = MAGIC_ORD.len();
    let n_trees = le_u32(&buf, &mut off)? as usize;
    let mut order = Vec::with_capacity(n_trees);
    for _ in 0..n_trees {
        order.push(le_u32(&buf, &mut off)?);
    }
    for _ in 0..n_trees {
        let _ = le_f32(&buf, &mut off)?;
    }
    for _ in 0..n_trees {
        let _ = le_f32(&buf, &mut off)?;
    }
    let mut suffix_min = Vec::with_capacity(n_trees + 1);
    for _ in 0..(n_trees + 1) {
        suffix_min.push(le_f32(&buf, &mut off)?);
    }
    let mut suffix_max = Vec::with_capacity(n_trees + 1);
    for _ in 0..(n_trees + 1) {
        suffix_max.push(le_f32(&buf, &mut off)?);
    }
    Ok(TreeOrder {
        n_trees,
        order,
        suffix_min,
        suffix_max,
    })
}

fn load_prefix_pack(path: &PathBuf) -> Result<PrefixPack> {
    let mmap = mmap_readonly(path)?;
    let buf = &mmap[..];
    if buf.len() < MAGIC_PRF.len() + 12 {
        bail!("prefix pack too short");
    }
    if &buf[0..MAGIC_PRF.len()] != MAGIC_PRF {
        bail!("invalid prefix pack magic");
    }
    let mut off = MAGIC_PRF.len();
    let n_hot_features = le_u32(&buf, &mut off)? as usize;
    let n_trees = le_u32(&buf, &mut off)? as usize;
    let n_nodes = le_u32(&buf, &mut off)? as usize;

    let mut hot_global_fidx = Vec::with_capacity(n_hot_features);
    for _ in 0..n_hot_features {
        hot_global_fidx.push(le_u32(&buf, &mut off)?);
    }
    let mut tree_roots = Vec::with_capacity(n_trees);
    for _ in 0..n_trees {
        tree_roots.push(le_u32(&buf, &mut off)?);
    }
    let mut fidx = Vec::with_capacity(n_nodes);
    let mut left = Vec::with_capacity(n_nodes);
    let mut right = Vec::with_capacity(n_nodes);
    let mut missing = Vec::with_capacity(n_nodes);
    let mut thr = Vec::with_capacity(n_nodes);
    let mut leaf = Vec::with_capacity(n_nodes);
    let mut is_leaf = Vec::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        fidx.push(le_i32(&buf, &mut off)?);
        left.push(le_u32(&buf, &mut off)?);
        right.push(le_u32(&buf, &mut off)?);
        missing.push(le_u32(&buf, &mut off)?);
        thr.push(le_f32(&buf, &mut off)?);
        leaf.push(le_f32(&buf, &mut off)?);
        if off + 4 > buf.len() {
            bail!("prefix pack truncated at node flags");
        }
        is_leaf.push(buf[off]);
        off += 4;
    }
    Ok(PrefixPack {
        n_hot_features,
        n_trees,
        node_count: n_nodes,
        hot_global_fidx,
        tree_roots,
        fidx,
        left,
        right,
        missing,
        thr,
        leaf,
        is_leaf,
    })
}

fn compile_hot_prefix_pack(pack: &PrefixPack) -> Result<HotCompiledPrefixPack> {
    if pack.n_hot_features > u16::MAX as usize {
        bail!(
            "compiled hot prefix pack requires n_hot_features <= {} got {}",
            u16::MAX,
            pack.n_hot_features
        );
    }
    if pack.n_hot_features > HOT_COMPILED_NODE_FIDX_MASK as usize {
        bail!(
            "compiled hot prefix pack requires n_hot_features <= {} got {}",
            HOT_COMPILED_NODE_FIDX_MASK,
            pack.n_hot_features
        );
    }
    if pack.node_count > u16::MAX as usize {
        bail!(
            "compiled hot prefix pack requires node_count <= {} got {}",
            u16::MAX,
            pack.node_count
        );
    }
    let mut nodes = Vec::with_capacity(pack.node_count);
    for idx in 0..pack.node_count {
        let raw_hot_fidx = *pack.fidx.get(idx).unwrap_or(&0i32);
        let hot_fidx = if raw_hot_fidx < 0 {
            0u16
        } else {
            raw_hot_fidx as u16
        };
        let hot_fidx_flags = if pack.is_leaf[idx] != 0 {
            HOT_COMPILED_NODE_LEAF
        } else {
            hot_fidx
        };
        nodes.push(HotCompiledPrefixNode {
            thr: pack.thr[idx],
            leaf: pack.leaf[idx],
            left: pack.left[idx] as u16,
            right: pack.right[idx] as u16,
            missing: pack.missing[idx] as u16,
            hot_fidx_flags,
        });
    }
    Ok(HotCompiledPrefixPack {
        n_hot_features: pack.n_hot_features,
        n_trees: pack.n_trees,
        hot_global_fidx: pack.hot_global_fidx.clone(),
        tree_roots: pack.tree_roots.iter().map(|&v| v as u16).collect(),
        nodes,
    })
}

fn load_route_meta(path: &PathBuf) -> Result<RouteMeta> {
    let buf =
        fs::read(path).with_context(|| format!("read route meta failed: {}", path.display()))?;
    if buf.len() < MAGIC_RMETA.len() + 4 {
        bail!("route meta too short");
    }
    if &buf[0..MAGIC_RMETA.len()] != MAGIC_RMETA {
        bail!("invalid route meta magic");
    }
    let mut off = MAGIC_RMETA.len();
    let n_rows = le_u32(&buf, &mut off)? as usize;
    let mut ids = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        ids.push(le_i64(&buf, &mut off)?);
    }
    let mut fold_id = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        fold_id.push(le_i32(&buf, &mut off)?);
    }
    let mut tau_used = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        tau_used.push(le_f32(&buf, &mut off)?);
    }
    if off + (n_rows * 2) > buf.len() {
        bail!("route meta truncated");
    }
    let active = buf[off..off + n_rows].to_vec();
    off += n_rows;
    let exact_positive = buf[off..off + n_rows].to_vec();
    Ok(RouteMeta {
        n_rows,
        ids,
        fold_id,
        tau_used,
        active,
        exact_positive,
    })
}

fn load_dispatch_meta(path: &PathBuf) -> Result<DispatchMeta> {
    let buf =
        fs::read(path).with_context(|| format!("read dispatch meta failed: {}", path.display()))?;
    if buf.len() < MAGIC_DMETA.len() + 4 {
        bail!("dispatch meta too short");
    }
    if &buf[0..MAGIC_DMETA.len()] != MAGIC_DMETA {
        bail!("invalid dispatch meta magic");
    }
    let mut off = MAGIC_DMETA.len();
    let n_rows = le_u32(&buf, &mut off)? as usize;
    let mut ids = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        ids.push(le_i64(&buf, &mut off)?);
    }
    let mut slot_id = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        slot_id.push(le_i32(&buf, &mut off)?);
    }
    let mut tau_used = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        tau_used.push(le_f32(&buf, &mut off)?);
    }
    if off + (n_rows * 2) > buf.len() {
        bail!("dispatch meta truncated");
    }
    let active = buf[off..off + n_rows].to_vec();
    off += n_rows;
    let exact_positive = buf[off..off + n_rows].to_vec();
    Ok(DispatchMeta {
        n_rows,
        ids,
        slot_id,
        tau_used,
        active,
        exact_positive,
    })
}

fn load_prefix_calibration(path: &PathBuf, variant_key: &str) -> Result<LoadedPrefixCal> {
    let txt = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let raw: PrefixCalFile =
        serde_json::from_str(&txt).with_context(|| format!("parse {}", path.display()))?;
    if raw.format != "L2PrefixCalV1" && raw.format != "L2PrefixCalV2" {
        bail!("unsupported prefix calibration format: {}", raw.format);
    }
    if raw.checkpoints.is_empty() {
        bail!("prefix calibration has no checkpoints");
    }
    let variant = raw
        .variants
        .into_iter()
        .find(|v| v.key == variant_key)
        .with_context(|| format!("variant_key={} not found in {}", variant_key, path.display()))?;
    if variant.tables.len() != raw.checkpoints.len() {
        bail!(
            "variant {} checkpoint table count mismatch: manifest={} tables={}",
            variant.key,
            raw.checkpoints.len(),
            variant.tables.len()
        );
    }

    let mut tables = Vec::with_capacity(variant.tables.len());
    for table in variant.tables {
        if table.gap_edges.len() != 257 {
            bail!(
                "checkpoint {} gap_edges len must be 257, got {}",
                table.checkpoint,
                table.gap_edges.len()
            );
        }
        if table.global_ref_hi.len() != 256 || table.global_rej_lo.len() != 256 {
            bail!(
                "checkpoint {} global band len mismatch: ref={} rej={}",
                table.checkpoint,
                table.global_ref_hi.len(),
                table.global_rej_lo.len()
            );
        }
        let fold_count = table.fold_tables.len();
        let mut fold_ids = Vec::with_capacity(fold_count);
        let mut fold_ref_hi = Vec::with_capacity(fold_count * 256);
        let mut fold_rej_lo = Vec::with_capacity(fold_count * 256);
        for fold in table.fold_tables {
            if fold.ref_hi.len() != 256 || fold.rej_lo.len() != 256 {
                bail!(
                    "checkpoint {} fold {} band len mismatch: ref={} rej={}",
                    table.checkpoint,
                    fold.fold_id,
                    fold.ref_hi.len(),
                    fold.rej_lo.len()
                );
            }
            fold_ids.push(fold.fold_id);
            fold_ref_hi.extend_from_slice(&fold.ref_hi);
            fold_rej_lo.extend_from_slice(&fold.rej_lo);
        }
        let (fold_lut_base, fold_slot_lut) = if let (Some(min_id), Some(max_id)) =
            (fold_ids.iter().min().copied(), fold_ids.iter().max().copied())
        {
            let span = (max_id as i64) - (min_id as i64) + 1;
            if span > 0 && span <= 4096 {
                let mut lut = vec![-1i16; span as usize];
                for (slot, fold_id) in fold_ids.iter().copied().enumerate() {
                    lut[(fold_id - min_id) as usize] = slot as i16;
                }
                (min_id, lut)
            } else {
                (0, Vec::new())
            }
        } else {
            (0, Vec::new())
        };
        let tau_bin_count = table.tau_edges.len().saturating_sub(1);
        let tau_slots = fold_count * tau_bin_count;
        let mut tau_fold_mask = vec![0u8; tau_slots];
        let mut tau_ref_hi = vec![0.0f32; tau_slots * 256];
        let mut tau_rej_lo = vec![0.0f32; tau_slots * 256];
        for tau_fold in table.tau_fold_tables {
            if tau_fold.ref_hi.len() != 256 || tau_fold.rej_lo.len() != 256 {
                bail!(
                    "checkpoint {} fold {} tau_bin {} band len mismatch: ref={} rej={}",
                    table.checkpoint,
                    tau_fold.fold_id,
                    tau_fold.tau_bin,
                    tau_fold.ref_hi.len(),
                    tau_fold.rej_lo.len()
                );
            }
            let fold_slot = fold_ids
                .iter()
                .position(|&id| id == tau_fold.fold_id)
                .with_context(|| {
                    format!(
                        "checkpoint {} tau_fold references unknown fold_id {}",
                        table.checkpoint, tau_fold.fold_id
                    )
                })?;
            if tau_fold.tau_bin >= tau_bin_count {
                bail!(
                    "checkpoint {} fold {} tau_bin {} out of range {}",
                    table.checkpoint,
                    tau_fold.fold_id,
                    tau_fold.tau_bin,
                    tau_bin_count
                );
            }
            let slot = fold_slot * tau_bin_count + tau_fold.tau_bin;
            tau_fold_mask[slot] = 1;
            let off = slot * 256;
            tau_ref_hi[off..off + 256].copy_from_slice(&tau_fold.ref_hi);
            tau_rej_lo[off..off + 256].copy_from_slice(&tau_fold.rej_lo);
        }
        tables.push(LoadedPrefixCheckpoint {
            checkpoint: table.checkpoint,
            gap_edges: table.gap_edges,
            global_ref_hi: table.global_ref_hi,
            global_rej_lo: table.global_rej_lo,
            tau_edges: table.tau_edges,
            fold_ids,
            fold_lut_base,
            fold_slot_lut,
            fold_ref_hi,
            fold_rej_lo,
            tau_bin_count,
            tau_fold_mask,
            tau_ref_hi,
            tau_rej_lo,
        });
    }

    Ok(LoadedPrefixCal {
        variant_key: variant.key,
        ref_q: variant.ref_q,
        rej_q: variant.rej_q,
        checkpoints: raw.checkpoints,
        tables,
    })
}

fn load_mlp_certifier(path: &PathBuf) -> Result<LoadedMlpCertifier> {
    let txt = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let raw: MlpCertifierFile =
        serde_json::from_str(&txt).with_context(|| format!("parse {}", path.display()))?;
    if raw.format != "L2PrefixCalMlpV1" {
        bail!("unsupported mlp certifier format: {}", raw.format);
    }
    if raw.mean.len() != raw.inv_std.len() {
        bail!(
            "mlp normalization length mismatch: mean={} inv_std={}",
            raw.mean.len(),
            raw.inv_std.len()
        );
    }
    if raw.w1.len() != raw.b1.len() || raw.w2.len() != raw.b2.len() {
        bail!(
            "mlp hidden layer shape mismatch: w1={} b1={} w2={} b2={}",
            raw.w1.len(),
            raw.b1.len(),
            raw.w2.len(),
            raw.b2.len()
        );
    }
    let input_dim = raw.mean.len();
    for row in &raw.w1 {
        if row.len() != input_dim {
            bail!("mlp w1 input dim mismatch: expected={} got={}", input_dim, row.len());
        }
    }
    for row in &raw.w2 {
        if row.len() != raw.w1.len() {
            bail!("mlp w2 input dim mismatch: expected={} got={}", raw.w1.len(), row.len());
        }
    }
    if raw.w3.len() != raw.w2.len() {
        bail!(
            "mlp w3 input dim mismatch: expected={} got={}",
            raw.w2.len(),
            raw.w3.len()
        );
    }
    let mut threshold_by_checkpoint = HashMap::new();
    for item in raw.thresholds {
        threshold_by_checkpoint.insert(
            item.checkpoint,
            LoadedMlpCheckpoint {
                tau_ref: item.tau_ref,
                tau_rej: item.tau_rej,
            },
        );
    }
    Ok(LoadedMlpCertifier {
        feature_schema_version: raw.feature_schema_version,
        variant_key: raw.variant_key,
        checkpoint_values: raw.checkpoint_values,
        fold_values: raw.fold_values,
        tau_edges: raw.tau_edges,
        checkpoint_limit: raw.checkpoint_limit,
        mean: raw.mean,
        inv_std: raw.inv_std,
        w1: raw.w1,
        b1: raw.b1,
        w2: raw.w2,
        b2: raw.b2,
        w3: raw.w3,
        b3: raw.b3,
        threshold_by_checkpoint,
    })
}

fn build_loaded_atlas_checkpoint(
    checkpoint: usize,
    gap_edges: Vec<f32>,
    feature_mean: [f32; 4],
    feature_inv_std: [f32; 4],
    centroids: &[[f32; 4]],
    tau_bin_count: usize,
    decision_grid: Vec<u8>,
) -> LoadedPrefixAtlasCheckpoint {
    let cluster_count = centroids.len();
    let mut centroid_w0 = Vec::with_capacity(cluster_count);
    let mut centroid_w1 = Vec::with_capacity(cluster_count);
    let mut centroid_w2 = Vec::with_capacity(cluster_count);
    let mut centroid_w3 = Vec::with_capacity(cluster_count);
    let mut centroid_bias = Vec::with_capacity(cluster_count);
    for centroid in centroids {
        centroid_w0.push(2.0 * centroid[0]);
        centroid_w1.push(2.0 * centroid[1]);
        centroid_w2.push(2.0 * centroid[2]);
        centroid_w3.push(2.0 * centroid[3]);
        centroid_bias.push(
            -((centroid[0] * centroid[0])
                + (centroid[1] * centroid[1])
                + (centroid[2] * centroid[2])
                + (centroid[3] * centroid[3])),
        );
    }
    let centroids16 = if cluster_count == 16 {
        let mut out = LoadedPrefixAtlasCentroids16 {
            w0: [0.0; 16],
            w1: [0.0; 16],
            w2: [0.0; 16],
            w3: [0.0; 16],
            bias: [0.0; 16],
        };
        out.w0.copy_from_slice(&centroid_w0);
        out.w1.copy_from_slice(&centroid_w1);
        out.w2.copy_from_slice(&centroid_w2);
        out.w3.copy_from_slice(&centroid_w3);
        out.bias.copy_from_slice(&centroid_bias);
        Some(Box::new(out))
    } else {
        None
    };
    LoadedPrefixAtlasCheckpoint {
        checkpoint,
        gap_edges,
        shared_gap_edges: false,
        feature_mean,
        feature_inv_std,
        cluster_count,
        tau_bin_count,
        centroid_w0,
        centroid_w1,
        centroid_w2,
        centroid_w3,
        centroid_bias,
        centroids16,
        decision_grid,
    }
}

fn load_prefix_atlas_json(path: &PathBuf) -> Result<LoadedPrefixAtlas> {
    let txt = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let raw: PrefixAtlasFile =
        serde_json::from_str(&txt).with_context(|| format!("parse {}", path.display()))?;
    if raw.format != "L2PrefixAtlasV1" {
        bail!("unsupported prefix atlas format: {}", raw.format);
    }
    if raw.tau_edges.len() < 2 {
        bail!("prefix atlas must provide tau edges");
    }
    let tau_bin_count = raw.tau_edges.len() - 1;
    let mut checkpoints = Vec::with_capacity(raw.checkpoints.len());
    for table in raw.checkpoints {
        if table.feature_mean.len() != 4 || table.feature_inv_std.len() != 4 {
            bail!(
                "atlas checkpoint {} feature normalization length mismatch: mean={} inv_std={}",
                table.checkpoint,
                table.feature_mean.len(),
                table.feature_inv_std.len()
            );
        }
        let feature_mean = [
            table.feature_mean[0],
            table.feature_mean[1],
            table.feature_mean[2],
            table.feature_mean[3],
        ];
        let feature_inv_std = [
            table.feature_inv_std[0],
            table.feature_inv_std[1],
            table.feature_inv_std[2],
            table.feature_inv_std[3],
        ];
        let mut centroids = Vec::with_capacity(table.centroids.len());
        for centroid in table.centroids {
            if centroid.len() != 4 {
                bail!(
                    "atlas checkpoint {} centroid length mismatch: expected=4 got={}",
                    table.checkpoint,
                    centroid.len()
                );
            }
            centroids.push([centroid[0], centroid[1], centroid[2], centroid[3]]);
        }
        let cluster_count = centroids.len();
        let mut decision_grid = vec![0u8; tau_bin_count * cluster_count * 256];
        for cell in table.cells {
            if cell.reject_rate.len() != 256 || cell.counts.len() != 256 {
                bail!(
                    "atlas checkpoint {} tau_bin {} cluster {} cell len mismatch: reject_rate={} counts={}",
                    table.checkpoint,
                    cell.tau_bin,
                    cell.cluster_id,
                    cell.reject_rate.len(),
                    cell.counts.len()
                );
            }
            if cell.tau_bin >= tau_bin_count {
                bail!(
                    "atlas checkpoint {} tau_bin {} out of range {}",
                    table.checkpoint,
                    cell.tau_bin,
                    tau_bin_count
                );
            }
            if cell.cluster_id >= cluster_count {
                bail!(
                    "atlas checkpoint {} cluster_id {} out of range {}",
                    table.checkpoint,
                    cell.cluster_id,
                    cluster_count
                );
            }
            let base = ((cell.tau_bin * cluster_count) + cell.cluster_id) * 256;
            for gap_bin in 0..256usize {
                let count = cell.counts[gap_bin];
                let reject_rate = cell.reject_rate[gap_bin];
                let decision = if count < raw.min_cell_count {
                    0u8
                } else if reject_rate <= raw.safe_ref_max_reject_rate {
                    1u8
                } else if reject_rate >= raw.safe_rej_min_reject_rate {
                    2u8
                } else {
                    0u8
                };
                decision_grid[base + gap_bin] = decision;
            }
        }
        checkpoints.push(build_loaded_atlas_checkpoint(
            table.checkpoint,
            table.gap_edges,
            feature_mean,
            feature_inv_std,
            &centroids,
            tau_bin_count,
            decision_grid,
        ));
    }
    Ok(LoadedPrefixAtlas {
        feature_schema_version: raw.feature_schema_version,
        variant_key: raw.variant_key,
        tau_edges: raw.tau_edges,
        cluster_count: raw.cluster_count,
        min_cell_count: raw.min_cell_count,
        safe_ref_max_reject_rate: raw.safe_ref_max_reject_rate,
        safe_rej_min_reject_rate: raw.safe_rej_min_reject_rate,
        checkpoints,
    })
}

fn load_prefix_atlas_bin(path: &PathBuf) -> Result<LoadedPrefixAtlas> {
    let buf = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    if buf.len() < 12 {
        bail!("prefix atlas bin too short");
    }
    let mut off = 0usize;
    let magic = le_bytes(&buf, &mut off, 8)?;
    if magic != b"L2ATLSV2" {
        bail!("unsupported prefix atlas binary magic");
    }
    let version = le_u32(&buf, &mut off)?;
    if version != 2 {
        bail!("unsupported prefix atlas binary version: {}", version);
    }
    let feature_schema_version = le_u32(&buf, &mut off)?;
    let variant_key = le_string(&buf, &mut off)?;
    let tau_edge_count = le_u32(&buf, &mut off)? as usize;
    if tau_edge_count < 2 {
        bail!("prefix atlas bin must provide tau edges");
    }
    let tau_edges = le_f32_vec(&buf, &mut off, tau_edge_count)?;
    let cluster_count = le_u32(&buf, &mut off)? as usize;
    let min_cell_count = le_u32(&buf, &mut off)?;
    let safe_ref_max_reject_rate = le_f32(&buf, &mut off)?;
    let safe_rej_min_reject_rate = le_f32(&buf, &mut off)?;
    let checkpoint_count = le_u32(&buf, &mut off)? as usize;
    let tau_bin_count = tau_edge_count - 1;
    let mut checkpoints = Vec::with_capacity(checkpoint_count);
    for _ in 0..checkpoint_count {
        let checkpoint = le_u32(&buf, &mut off)? as usize;
        let cluster_count_cp = le_u32(&buf, &mut off)? as usize;
        let gap_edge_count = le_u32(&buf, &mut off)? as usize;
        let feature_mean_v = le_f32_vec(&buf, &mut off, 4)?;
        let feature_inv_std_v = le_f32_vec(&buf, &mut off, 4)?;
        let gap_edges = le_f32_vec(&buf, &mut off, gap_edge_count)?;
        let centroid_w0 = le_f32_vec(&buf, &mut off, cluster_count_cp)?;
        let centroid_w1 = le_f32_vec(&buf, &mut off, cluster_count_cp)?;
        let centroid_w2 = le_f32_vec(&buf, &mut off, cluster_count_cp)?;
        let centroid_w3 = le_f32_vec(&buf, &mut off, cluster_count_cp)?;
        let centroid_bias = le_f32_vec(&buf, &mut off, cluster_count_cp)?;
        let centroids16 = if cluster_count_cp == 16 {
            let mut out = LoadedPrefixAtlasCentroids16 {
                w0: [0.0; 16],
                w1: [0.0; 16],
                w2: [0.0; 16],
                w3: [0.0; 16],
                bias: [0.0; 16],
            };
            out.w0.copy_from_slice(&centroid_w0);
            out.w1.copy_from_slice(&centroid_w1);
            out.w2.copy_from_slice(&centroid_w2);
            out.w3.copy_from_slice(&centroid_w3);
            out.bias.copy_from_slice(&centroid_bias);
            Some(Box::new(out))
        } else {
            None
        };
        let grid_len = le_u32(&buf, &mut off)? as usize;
        if grid_len != tau_bin_count * cluster_count_cp * 256 {
            bail!(
                "atlas bin checkpoint {} grid len mismatch: got={} expected={}",
                checkpoint,
                grid_len,
                tau_bin_count * cluster_count_cp * 256
            );
        }
        let decision_grid = le_bytes(&buf, &mut off, grid_len)?.to_vec();
        checkpoints.push(LoadedPrefixAtlasCheckpoint {
            checkpoint,
            gap_edges,
            shared_gap_edges: false,
            feature_mean: [
                feature_mean_v[0],
                feature_mean_v[1],
                feature_mean_v[2],
                feature_mean_v[3],
            ],
            feature_inv_std: [
                feature_inv_std_v[0],
                feature_inv_std_v[1],
                feature_inv_std_v[2],
                feature_inv_std_v[3],
            ],
            cluster_count: cluster_count_cp,
            tau_bin_count,
            centroid_w0,
            centroid_w1,
            centroid_w2,
            centroid_w3,
            centroid_bias,
            centroids16,
            decision_grid,
        });
    }
    if off != buf.len() {
        bail!("prefix atlas bin trailing bytes: off={} len={}", off, buf.len());
    }
    Ok(LoadedPrefixAtlas {
        feature_schema_version,
        variant_key,
        tau_edges,
        cluster_count,
        min_cell_count,
        safe_ref_max_reject_rate,
        safe_rej_min_reject_rate,
        checkpoints,
    })
}

fn resolve_bundle_path(
    cli_value: Option<&PathBuf>,
    manifest_value: Option<&String>,
    manifest_path: Option<&PathBuf>,
    field: &str,
) -> Result<PathBuf> {
    if let Some(path) = cli_value {
        return Ok(path.clone());
    }
    if let Some(raw) = manifest_value {
        let raw_path = PathBuf::from(raw);
        let path = if let Some(base_path) = manifest_path {
            let candidate = resolve_relative_to(base_path, raw);
            if candidate.exists() || raw_path.is_absolute() {
                candidate
            } else {
                raw_path
            }
        } else {
            raw_path
        };
        return Ok(path);
    }
    bail!("missing required prefix-cal field: {}", field)
}

fn resolve_bundle_path_optional(
    cli_value: Option<&PathBuf>,
    manifest_value: Option<&String>,
    manifest_path: Option<&PathBuf>,
) -> Option<PathBuf> {
    if let Some(path) = cli_value {
        return Some(path.clone());
    }
    manifest_value.map(|raw| {
        let raw_path = PathBuf::from(raw);
        if let Some(base_path) = manifest_path {
            let candidate = resolve_relative_to(base_path, raw);
            if candidate.exists() || raw_path.is_absolute() {
                candidate
            } else {
                raw_path
            }
        } else {
            raw_path
        }
    })
}

fn resolve_prefix_cal_bundle(args: &QsL2PrefixCalArgs) -> Result<ResolvedPrefixCalBundle> {
    let manifest_path = args.bundle_manifest.as_ref();
    let manifest = if let Some(path) = &args.bundle_manifest {
        let txt = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let parsed: PrefixCalBundleManifest =
            serde_json::from_str(&txt).with_context(|| format!("parse {}", path.display()))?;
        if parsed.format != "L2PrefixCalBundleV1" {
            bail!("unsupported prefix-cal bundle format: {}", parsed.format);
        }
        Some(parsed)
    } else {
        None
    };

    let manifest_ref = manifest.as_ref();
    let variant_key = if let Some(key) = &args.variant_key {
        key.clone()
    } else if let Some(raw) = manifest_ref.and_then(|m| m.selected_exact_variant.as_ref()) {
        raw.clone()
    } else if let Some(raw) = manifest_ref.and_then(|m| m.selected_variant.as_ref()) {
        raw.clone()
    } else {
        bail!("missing required prefix-cal field: variant_key or selected_variant")
    };

    let direct_kernel = if let Some(raw) = &args.direct_kernel {
        parse_prefix_direct_kernel(raw)?
    } else if let Some(raw) = manifest_ref.and_then(|m| m.direct_kernel.as_ref()) {
        parse_prefix_direct_kernel(raw)?
    } else {
        PrefixDirectKernel::Qs
    };
    let certifier_kind = if let Some(raw) = &args.certifier_kind {
        parse_prefix_certifier_kind(raw)?
    } else if let Some(raw) = manifest_ref.and_then(|m| m.certifier_kind.as_ref()) {
        parse_prefix_certifier_kind(raw)?
    } else {
        PrefixCertifierKind::TableV1
    };
    let hot_exact_prefix_limit = if let Some(limit) = manifest_ref.and_then(|m| m.hot_exact_prefix_limit) {
        Some(limit)
    } else {
        match direct_kernel {
            PrefixDirectKernel::Qs => None,
            PrefixDirectKernel::HotExact96 => Some(96),
            PrefixDirectKernel::HotExact128 => Some(128),
            PrefixDirectKernel::HotExact192 => Some(192),
            PrefixDirectKernel::HotExact256 => Some(256),
            PrefixDirectKernel::HotExact384 => Some(384),
        }
    };
    let hot_exact_prefix_pack = match direct_kernel {
        PrefixDirectKernel::Qs => None,
        PrefixDirectKernel::HotExact96 => resolve_bundle_path_optional(
            None,
            manifest_ref.and_then(|m| m.hot_exact_prefix_96_pack.as_ref()),
            manifest_path,
        ),
        PrefixDirectKernel::HotExact128 => resolve_bundle_path_optional(
            None,
            manifest_ref.and_then(|m| m.hot_exact_prefix_128_pack.as_ref()),
            manifest_path,
        ),
        PrefixDirectKernel::HotExact192 => resolve_bundle_path_optional(
            None,
            manifest_ref.and_then(|m| m.hot_exact_prefix_192_pack.as_ref()),
            manifest_path,
        ),
        PrefixDirectKernel::HotExact256 => resolve_bundle_path_optional(
            None,
            manifest_ref.and_then(|m| m.hot_exact_prefix_256_pack.as_ref()),
            manifest_path,
        ),
        PrefixDirectKernel::HotExact384 => resolve_bundle_path_optional(
            None,
            manifest_ref.and_then(|m| m.hot_exact_prefix_384_pack.as_ref()),
            manifest_path,
        ),
    };

    Ok(ResolvedPrefixCalBundle {
        qs_pack: resolve_bundle_path(
            args.qs_pack.as_ref(),
            manifest_ref.and_then(|m| m.pack_path.as_ref()),
            manifest_path,
            "pack_path",
        )?,
        calibration_json: resolve_bundle_path(
            args.calibration_json.as_ref(),
            manifest_ref.and_then(|m| m.calibration_json.as_ref()),
            manifest_path,
            "calibration_json",
        )?,
        variant_key,
        feat_bin: resolve_bundle_path(
            args.feat_bin.as_ref(),
            manifest_ref.and_then(|m| m.feat_bin.as_ref()),
            manifest_path,
            "feat_bin",
        )?,
        route_meta: resolve_bundle_path(
            args.route_meta.as_ref(),
            manifest_ref.and_then(|m| m.route_meta.as_ref()),
            manifest_path,
            "route_meta",
        )?,
        soa: resolve_bundle_path(
            args.soa.as_ref(),
            manifest_ref.and_then(|m| m.soa_bin.as_ref()),
            manifest_path,
            "soa_bin",
        )?,
        bounds: resolve_bundle_path(
            args.bounds.as_ref(),
            manifest_ref.and_then(|m| m.bounds_bin.as_ref()),
            manifest_path,
            "bounds_bin",
        )?,
        tree_order: resolve_bundle_path(
            args.tree_order.as_ref(),
            manifest_ref.and_then(|m| m.tree_order_bin.as_ref()),
            manifest_path,
            "tree_order_bin",
        )?,
        model_json: resolve_bundle_path(
            args.model_json.as_ref(),
            manifest_ref.and_then(|m| m.model_json.as_ref()),
            manifest_path,
            "model_json",
        )?,
        direct_kernel,
        certifier_kind,
        certifier_json: resolve_bundle_path_optional(
            args.certifier_json.as_ref(),
            manifest_ref.and_then(|m| m.certifier_json.as_ref()),
            manifest_path,
        ),
        atlas_bin: resolve_bundle_path_optional(
            None,
            manifest_ref.and_then(|m| m.atlas_bin.as_ref()),
            manifest_path,
        ),
        atlas_format: manifest_ref.and_then(|m| m.atlas_format.as_ref()).cloned(),
        hot_exact_prefix_limit,
        telemetry_schema_version: manifest_ref
            .and_then(|m| m.telemetry_schema_version)
            .unwrap_or(2),
        hot_exact_prefix_pack,
    })
}

fn load_prefix_runtime(
    resolved: &ResolvedPrefixCalBundle,
    model: &SoaModel,
    bounds: &Bounds,
) -> Result<LoadedPrefixRuntime> {
    let pack = qs_exact::load_qs_pack(&resolved.qs_pack)?;
    let calibration = load_prefix_calibration(&resolved.calibration_json, &resolved.variant_key)?;
    let atlas_certifier = match resolved.certifier_kind {
        PrefixCertifierKind::TableV1 | PrefixCertifierKind::MlpV1 => None,
        PrefixCertifierKind::AtlasV1 => {
            let mut loaded = if let Some(path) = resolved.atlas_bin.as_ref() {
                load_prefix_atlas_bin(path)?
            } else {
                let path = resolved
                    .certifier_json
                    .as_ref()
                    .context("atlas_v1 selected without certifier_json/atlas_bin")?;
                load_prefix_atlas_json(path)?
            };
            if loaded.variant_key != calibration.variant_key {
                bail!(
                    "atlas certifier variant mismatch: bundle/calibration={} atlas={}",
                    calibration.variant_key,
                    loaded.variant_key
                );
            }
            if loaded.checkpoints.len() != calibration.checkpoints.len() {
                bail!(
                    "atlas checkpoint count mismatch: calibration={} atlas={}",
                    calibration.checkpoints.len(),
                    loaded.checkpoints.len()
                );
            }
            for (slot, checkpoint) in calibration.checkpoints.iter().enumerate() {
                if loaded.checkpoints[slot].checkpoint != *checkpoint {
                    bail!(
                        "atlas checkpoint mismatch at slot {}: calibration={} atlas={}",
                        slot,
                        checkpoint,
                        loaded.checkpoints[slot].checkpoint
                    );
                }
                if loaded.checkpoints[slot].gap_edges.is_empty() {
                    loaded.checkpoints[slot].gap_edges =
                        calibration.tables[slot].gap_edges.clone();
                }
                loaded.checkpoints[slot].shared_gap_edges =
                    loaded.checkpoints[slot].gap_edges == calibration.tables[slot].gap_edges;
            }
            Some(loaded)
        }
    };
    let mlp_certifier = match resolved.certifier_kind {
        PrefixCertifierKind::TableV1 | PrefixCertifierKind::AtlasV1 => None,
        PrefixCertifierKind::MlpV1 => {
            let path = resolved
                .certifier_json
                .as_ref()
                .context("mlp_v1 selected without certifier_json")?;
            let loaded = load_mlp_certifier(path)?;
            if loaded.variant_key != calibration.variant_key {
                bail!(
                    "mlp certifier variant mismatch: bundle/calibration={} certifier={}",
                    calibration.variant_key,
                    loaded.variant_key
                );
            }
            Some(loaded)
        }
    };
    let hot_pack = if resolved.direct_kernel == PrefixDirectKernel::Qs {
        None
    } else {
        let path = resolved
            .hot_exact_prefix_pack
            .as_ref()
            .context("hot exact direct kernel selected without hot prefix pack")?;
        Some(load_prefix_pack(path)?)
    };
    let compiled_hot_pack = match hot_pack.as_ref() {
        Some(pack) => Some(compile_hot_prefix_pack(pack)?),
        None => None,
    };
    let raw_tree_order = load_tree_order(&resolved.tree_order)?;
    let plan = materialize_tree_plan(bounds, Some(&raw_tree_order), None)?;
    if pack.n_features() != model.n_features {
        bail!(
            "feature count mismatch: soa={} qs_pack={}",
            model.n_features,
            pack.n_features()
        );
    }
    let hot_checkpoint_layout =
        build_hot_checkpoint_layout(&calibration, hot_pack.as_ref(), &pack, resolved.direct_kernel);
    Ok(LoadedPrefixRuntime {
        pack,
        calibration,
        atlas_certifier,
        mlp_certifier,
        hot_pack,
        compiled_hot_pack,
        hot_checkpoint_layout,
        plan,
        direct_kernel: resolved.direct_kernel,
        certifier_kind: resolved.certifier_kind,
        hot_exact_prefix_limit: resolved.hot_exact_prefix_limit.unwrap_or(0),
        telemetry_schema_version: resolved.telemetry_schema_version,
        packet_scheduler: None,
        anchor_rescue: None,
    })
}

fn build_hot_checkpoint_layout(
    calibration: &LoadedPrefixCal,
    hot_pack: Option<&PrefixPack>,
    pack: &qs_exact::QsPack,
    direct_kernel: PrefixDirectKernel,
) -> Option<HotCheckpointLayout> {
    if !matches!(
        direct_kernel,
        PrefixDirectKernel::HotExact96
            | PrefixDirectKernel::HotExact128
            | PrefixDirectKernel::HotExact192
            | PrefixDirectKernel::HotExact256
            | PrefixDirectKernel::HotExact384
    ) {
        return None;
    }
    let hot_pack = hot_pack?;
    let hot_limit = hot_pack.n_trees.min(pack.n_trees());
    let mut out = HotCheckpointLayout::default();
    for (idx, checkpoint) in calibration.checkpoints.iter().copied().enumerate() {
        if checkpoint <= hot_limit {
            out.hot_values.push(checkpoint);
            out.hot_indices.push(idx);
        } else {
            out.late_values.push(checkpoint);
            out.late_indices.push(idx);
        }
    }
    Some(out)
}

fn resolve_relative_to(base_path: &PathBuf, raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_relative() {
        base_path
            .parent()
            .map(|p| p.join(path.clone()))
            .unwrap_or(path)
    } else {
        path
    }
}

fn resolve_relative_compat(base_path: &PathBuf, raw: &str) -> PathBuf {
    let raw_path = PathBuf::from(raw);
    if raw_path.is_absolute() {
        return raw_path;
    }
    let candidate = resolve_relative_to(base_path, raw);
    if candidate.exists() {
        candidate
    } else {
        raw_path
    }
}

fn parse_packet_bank(raw: &str) -> i8 {
    match raw {
        "scout" => 2,
        "reject" => 1,
        "refer" => 0,
        _ => -1,
    }
}

fn load_packet_summary_tsv(path: &PathBuf) -> Result<Vec<LoadedPacket>> {
    let txt = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut lines = txt.lines();
    let header = lines
        .next()
        .context("packet_summary.tsv missing header")?
        .split('\t')
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut col = HashMap::new();
    for (idx, name) in header.iter().enumerate() {
        col.insert(name.as_str(), idx);
    }
    let get_idx = |name: &'static str| -> Result<usize> {
        col.get(name)
            .copied()
            .with_context(|| format!("packet_summary.tsv missing column {}", name))
    };
    let packet_id_idx = get_idx("packet_id")?;
    let packet_bank_idx = get_idx("packet_bank")?;
    let tree_count_idx = get_idx("tree_count")?;
    let block_cost_idx = get_idx("block_cost")?;
    let tree_pos_list_idx = get_idx("tree_pos_list")?;
    let mut packets = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        let packet_id = fields
            .get(packet_id_idx)
            .context("packet row missing packet_id")?
            .parse::<usize>()
            .context("parse packet_id")?;
        let tree_count = fields
            .get(tree_count_idx)
            .context("packet row missing tree_count")?
            .parse::<usize>()
            .context("parse tree_count")?;
        let block_cost = fields
            .get(block_cost_idx)
            .context("packet row missing block_cost")?
            .parse::<f32>()
            .context("parse block_cost")?;
        let bank_id = parse_packet_bank(
            fields
                .get(packet_bank_idx)
                .copied()
                .context("packet row missing packet_bank")?,
        );
        let tree_indices = fields
            .get(tree_pos_list_idx)
            .copied()
            .unwrap_or("")
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<u32>().context("parse tree_pos_list value"))
            .collect::<Result<Vec<_>>>()?;
        packets.push(LoadedPacket {
            packet_id,
            bank_id,
            tree_indices,
            tree_count,
            block_cost,
            compiled_pack: None,
        });
    }
    packets.sort_by_key(|packet| packet.packet_id);
    Ok(packets)
}

fn load_packet_policy_json(path: &PathBuf) -> Result<LoadedPacketPolicy> {
    let txt = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let raw: PacketSchedulerPolicyFile =
        serde_json::from_str(&txt).with_context(|| format!("parse {}", path.display()))?;
    if raw.format != "L2PacketSchedulerPolicyV1" {
        bail!("unsupported packet scheduler policy format: {}", raw.format);
    }
    let _ = (
        &raw.packet_manifest_json,
        raw.tau_bin_count,
        raw.gap_regime_count,
        raw.delta_regime_count,
        raw.ranking_top_k,
        raw.oracle_depth,
        raw.oracle_top_k,
        raw.oracle_second_weight,
    );
    let mut state_rankings = HashMap::new();
    let mut step_tau_defaults: HashMap<(usize, usize), (usize, Vec<u16>)> = HashMap::new();
    let mut step_defaults: HashMap<usize, (usize, Vec<u16>)> = HashMap::new();
    for row in raw.policy_rows {
        let mut ranking = Vec::with_capacity(row.ranking.len());
        for packet_id in row.ranking {
            if packet_id >= raw.n_packets {
                bail!(
                    "packet policy ranking entry {} outside n_packets={}",
                    packet_id,
                    raw.n_packets
                );
            }
            ranking.push(packet_id as u16);
        }
        let key = (row.step, row.tau_bin, row.gap_regime, row.delta_regime);
        state_rankings.insert(key, ranking.clone());
        let step_tau_key = (row.step, row.tau_bin);
        match step_tau_defaults.get(&step_tau_key) {
            Some((best_count, _)) if *best_count >= row.row_count => {}
            _ => {
                step_tau_defaults.insert(step_tau_key, (row.row_count, ranking.clone()));
            }
        }
        match step_defaults.get(&row.step) {
            Some((best_count, _)) if *best_count >= row.row_count => {}
            _ => {
                step_defaults.insert(row.step, (row.row_count, ranking));
            }
        }
    }
    Ok(LoadedPacketPolicy {
        packet_family: raw.packet_family,
        packet_size: raw.packet_size,
        n_packets: raw.n_packets,
        max_steps: raw.max_steps,
        tau_edges: raw.tau_edges,
        gap_edges_by_step: raw.gap_edges_by_step,
        delta_edges_by_step: raw.delta_edges_by_step,
        state_rankings,
        step_tau_defaults: step_tau_defaults
            .into_iter()
            .map(|(k, (_cnt, ranking))| (k, ranking))
            .collect(),
        step_defaults: step_defaults
            .into_iter()
            .map(|(k, (_cnt, ranking))| (k, ranking))
            .collect(),
    })
}

fn load_v4_packet_scheduler(
    manifest_path: &PathBuf,
    parsed: &PrefixCalBundleManifest,
) -> Result<LoadedPacketScheduler> {
    let packet_manifest_path = resolve_bundle_path(
        None,
        parsed.packet_manifest_json.as_ref(),
        Some(manifest_path),
        "packet_manifest_json",
    )?;
    let txt = fs::read_to_string(&packet_manifest_path)
        .with_context(|| format!("read {}", packet_manifest_path.display()))?;
    let raw: PrefixPacketManifestFile = serde_json::from_str(&txt)
        .with_context(|| format!("parse {}", packet_manifest_path.display()))?;
    if raw.format != "L2PrefixPacketManifestV1" {
        bail!("unsupported packet manifest format: {}", raw.format);
    }
    let _ = (&raw.bundle_manifest, raw.mixed_frac, raw.scout_frac);
    let packet_summary_path = resolve_relative_compat(&packet_manifest_path, &raw.packet_summary_tsv);
    let scheduler_policy_path = if let Some(raw_path) = raw.scheduler_policy_json.as_ref() {
        resolve_relative_compat(&packet_manifest_path, raw_path)
    } else {
        packet_manifest_path
            .parent()
            .map(|p| p.join("scheduler_policy.json"))
            .unwrap_or_else(|| PathBuf::from("scheduler_policy.json"))
    };
    let mut packets = load_packet_summary_tsv(&packet_summary_path)?;
    if packets.len() != raw.n_packets {
        bail!(
            "packet count mismatch: manifest={} loaded={}",
            raw.n_packets,
            packets.len()
        );
    }
    for entry in raw.compiled_packet_packs {
        if entry.packet_id >= packets.len() {
            bail!(
                "compiled packet entry packet_id={} outside n_packets={}",
                entry.packet_id,
                packets.len()
            );
        }
        let pack_path = resolve_relative_compat(&packet_manifest_path, &entry.pack_path);
        packets[entry.packet_id].compiled_pack = Some(load_prefix_pack(&pack_path)?);
    }
    let policy = load_packet_policy_json(&scheduler_policy_path)?;
    if policy.n_packets != raw.n_packets {
        bail!(
            "packet policy n_packets mismatch: manifest={} policy={}",
            raw.n_packets,
            policy.n_packets
        );
    }
    if policy.max_steps != raw.max_steps {
        bail!(
            "packet policy max_steps mismatch: manifest={} policy={}",
            raw.max_steps,
            policy.max_steps
        );
    }
    let _ = manifest_path;
    Ok(LoadedPacketScheduler {
        packet_family: raw.packet_family,
        packet_size: raw.packet_size,
        max_steps: raw.max_steps,
        packets,
        policy,
    })
}

fn parse_rescue_action(raw: &str) -> Result<RescueAction> {
    match raw {
        "fallback" => Ok(RescueAction::Fallback),
        "run_reject_rescue" => Ok(RescueAction::RejectRescue),
        "run_refer_rescue" => Ok(RescueAction::ReferRescue),
        other => bail!("unsupported rescue action: {}", other),
    }
}

fn load_rescue_router(path: &PathBuf) -> Result<LoadedRescueRouter> {
    let txt = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let raw: RescueRouterFile =
        serde_json::from_str(&txt).with_context(|| format!("parse {}", path.display()))?;
    if raw.format != "L2AnchorRescueRouterV1" && raw.format != "L2AnchorRescueRouterV2" {
        bail!("unsupported rescue router format: {}", raw.format);
    }
    let is_v2 = raw.format == "L2AnchorRescueRouterV2";
    let mut actions = HashMap::new();
    for row in raw.rows {
        actions.insert(
            (
                if is_v2 { row.anchor_checkpoint } else { 0 },
                row.tau_bin,
                row.fold_id,
                row.shadow_side,
                row.gap_bin,
            ),
            parse_rescue_action(&row.action)?,
        );
    }
    let mut defaults = HashMap::new();
    for row in raw.defaults {
        defaults.insert(
            (
                if is_v2 { row.anchor_checkpoint } else { 0 },
                row.tau_bin,
                row.shadow_side,
            ),
            parse_rescue_action(&row.action)?,
        );
    }
    Ok(LoadedRescueRouter {
        tau_edges: raw.tau_edges,
        gap_edges: raw.gap_edges,
        actions,
        defaults,
    })
}

fn load_v4_anchor_rescue(
    args: &QsL2PrefixCalArgs,
    manifest_path: &PathBuf,
    parsed: &PrefixCalBundleManifest,
    model: &SoaModel,
    bounds: &Bounds,
) -> Result<(LoadedPrefixRuntime, String)> {
    let selected_variant = if let Some(key) = &args.variant_key {
        key.clone()
    } else if let Some(raw) = parsed.selected_exact_variant.as_ref() {
        raw.clone()
    } else if let Some(raw) = parsed.selected_variant.as_ref() {
        raw.clone()
    } else {
        bail!(
            "missing required anchor-rescue field: variant_key or selected_exact_variant"
        );
    };

    let anchor_manifest_path = resolve_bundle_path(
        None,
        parsed.anchor_manifest_json.as_ref(),
        Some(manifest_path),
        "anchor_manifest_json",
    )?;
    let rescue_manifest_path = resolve_bundle_path(
        None,
        parsed.rescue_manifest_json.as_ref(),
        Some(manifest_path),
        "rescue_manifest_json",
    )?;
    let rescue_router_path = resolve_bundle_path(
        None,
        parsed.rescue_router_json.as_ref(),
        Some(manifest_path),
        "rescue_router_json",
    )?;

    let anchor_args = QsL2PrefixCalArgs {
        bundle_manifest: Some(anchor_manifest_path.clone()),
        qs_pack: None,
        calibration_json: None,
        variant_key: Some(selected_variant.clone()),
        feat_bin: None,
        route_meta: None,
        soa: None,
        bounds: None,
        tree_order: None,
        model_json: None,
        direct_kernel: None,
        certifier_kind: None,
        certifier_json: None,
        threads: args.threads,
        chunk_rows: args.chunk_rows,
        parallel_min_rows: args.parallel_min_rows,
        max_rows: args.max_rows,
        out_tsv: None,
        stats_json: None,
        trace_jsonl: None,
        shadow_only: args.shadow_only,
    };
    let anchor_resolved = resolve_prefix_cal_bundle(&anchor_args)?;
    let mut anchor_runtime = load_prefix_runtime(&anchor_resolved, model, bounds)?;

    let rescue_txt = fs::read_to_string(&rescue_manifest_path)
        .with_context(|| format!("read {}", rescue_manifest_path.display()))?;
    let rescue_manifest: PrefixAnchorRescueManifestFile =
        serde_json::from_str(&rescue_txt)
            .with_context(|| format!("parse {}", rescue_manifest_path.display()))?;
    if rescue_manifest.format != "L2AnchorRescueManifestV1" {
        bail!(
            "unsupported anchor-rescue manifest format: {}",
            rescue_manifest.format
        );
    }

    let reject_bundle_manifest = resolve_relative_compat(
        &rescue_manifest_path,
        &rescue_manifest.reject_bundle_manifest,
    );
    let refer_bundle_manifest = resolve_relative_compat(
        &rescue_manifest_path,
        &rescue_manifest.refer_bundle_manifest,
    );

    let child_from_manifest = |bundle_manifest: PathBuf| QsL2PrefixCalArgs {
        bundle_manifest: Some(bundle_manifest.clone()),
        qs_pack: None,
        calibration_json: None,
        variant_key: Some(selected_variant.clone()),
        feat_bin: None,
        route_meta: None,
        soa: None,
        bounds: None,
        tree_order: None,
        model_json: None,
        direct_kernel: None,
        certifier_kind: None,
        certifier_json: None,
        threads: args.threads,
        chunk_rows: args.chunk_rows,
        parallel_min_rows: args.parallel_min_rows,
        max_rows: args.max_rows,
        out_tsv: None,
        stats_json: None,
        trace_jsonl: None,
        shadow_only: args.shadow_only,
    };

    let reject_resolved = resolve_prefix_cal_bundle(&child_from_manifest(reject_bundle_manifest))?;
    let refer_resolved = resolve_prefix_cal_bundle(&child_from_manifest(refer_bundle_manifest))?;
    let reject_runtime = load_prefix_runtime(&reject_resolved, model, bounds)?;
    let refer_runtime = load_prefix_runtime(&refer_resolved, model, bounds)?;
    let router = load_rescue_router(&rescue_router_path)?;

    anchor_runtime.anchor_rescue = Some(Box::new(LoadedAnchorRescueRuntime {
        reject_rescue: Box::new(reject_runtime),
        refer_rescue: Box::new(refer_runtime),
        router,
        anchor_limit: rescue_manifest.anchor_limit,
        rescue_limit: rescue_manifest.rescue_limit,
    }));
    Ok((anchor_runtime, selected_variant))
}

fn load_v4_order_router(
    args: &QsL2PrefixCalArgs,
    manifest_path: &PathBuf,
    parsed: &PrefixCalBundleManifest,
    model: &SoaModel,
    bounds: &Bounds,
) -> Result<(Vec<f32>, Vec<LoadedPrefixOrderRoute>, String, u32)> {
    let order_router_path = resolve_bundle_path(
        None,
        parsed.order_router_json.as_ref(),
        Some(manifest_path),
        "order_router_json",
    )?;
    let txt = fs::read_to_string(&order_router_path)
        .with_context(|| format!("read {}", order_router_path.display()))?;
    let router: PrefixOrderRouterFile =
        serde_json::from_str(&txt).with_context(|| format!("parse {}", order_router_path.display()))?;
    if router.format != "L2PrefixOrderRouterV1" {
        bail!("unsupported prefix order router format: {}", router.format);
    }
    let selected_variant = if let Some(key) = &args.variant_key {
        key.clone()
    } else if let Some(raw) = parsed.selected_exact_variant.as_ref() {
        raw.clone()
    } else if let Some(raw) = parsed.selected_variant.as_ref() {
        raw.clone()
    } else {
        bail!(
            "missing required V4 prefix-cal field: variant_key or selected_exact_variant"
        );
    };
    let telemetry_schema_version = parsed.telemetry_schema_version.unwrap_or(3);
    let base_dir = manifest_path
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(PathBuf::new);
    let mut routes = Vec::with_capacity(router.bundles.len());
    for entry in router.bundles.iter() {
        let child_manifest = {
            let path = PathBuf::from(&entry.bundle_manifest);
            if path.is_relative() {
                base_dir.join(path)
            } else {
                path
            }
        };
        let child_args = QsL2PrefixCalArgs {
            bundle_manifest: Some(child_manifest.clone()),
            qs_pack: None,
            calibration_json: None,
            variant_key: Some(selected_variant.clone()),
            feat_bin: None,
            route_meta: None,
            soa: None,
            bounds: None,
            tree_order: None,
            model_json: None,
            direct_kernel: None,
            certifier_kind: None,
            certifier_json: None,
            threads: args.threads,
            chunk_rows: args.chunk_rows,
            parallel_min_rows: args.parallel_min_rows,
            max_rows: args.max_rows,
            out_tsv: None,
            stats_json: None,
            trace_jsonl: None,
            shadow_only: args.shadow_only,
        };
        let resolved = resolve_prefix_cal_bundle(&child_args)?;
        let runtime = load_prefix_runtime(&resolved, model, bounds)?;
        routes.push(LoadedPrefixOrderRoute {
            tau_bin: entry.tau_bin,
            label: entry
                .label
                .clone()
                .unwrap_or_else(|| format!("tau_bin_{}", entry.tau_bin)),
            runtime,
        });
    }
    routes.sort_by_key(|item| item.tau_bin);
    Ok((router.tau_edges, routes, selected_variant, telemetry_schema_version))
}

#[inline(always)]
fn prefix_gap_bin(edges: &[f32], gap: f32) -> usize {
    let mut lo = 0usize;
    let mut hi = edges.len();
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if edges[mid] <= gap {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo.saturating_sub(1).min(edges.len().saturating_sub(2))
}

#[inline(always)]
fn prefix_gap_bin_guard(edges: &[f32], gap: f32, bin: usize) -> (usize, usize) {
    let mut lo = bin;
    let mut hi = bin;
    let tol = 1e-6f32.max(8.0 * f32::EPSILON * gap.abs().max(1.0));
    if bin > 0 {
        let edge_lo = edges[bin];
        if (gap - edge_lo).abs() <= tol {
            lo = bin - 1;
        }
    }
    if bin + 1 < edges.len().saturating_sub(1) {
        let edge_hi = edges[bin + 1];
        if (edge_hi - gap).abs() <= tol {
            hi = bin + 1;
        }
    }
    (lo, hi)
}

#[inline(always)]
fn lookup_prefix_bands_from_guarded_slices(
    ref_hi: &[f32],
    rej_lo: &[f32],
    guard_lo: usize,
    guard_hi: usize,
) -> (f32, f32) {
    let mut ref_bound = ref_hi[guard_lo];
    let mut rej_bound = rej_lo[guard_lo];
    for idx in (guard_lo + 1)..=guard_hi {
        ref_bound = ref_bound.max(ref_hi[idx]);
        rej_bound = rej_bound.min(rej_lo[idx]);
    }
    (ref_bound, rej_bound)
}

#[inline(always)]
fn lookup_prefix_bands_from_slices(
    ref_hi: &[f32],
    rej_lo: &[f32],
    gap_edges: &[f32],
    gap: f32,
) -> (f32, f32) {
    let bin = prefix_gap_bin(gap_edges, gap);
    let (guard_lo, guard_hi) = prefix_gap_bin_guard(gap_edges, gap, bin);
    lookup_prefix_bands_from_guarded_slices(ref_hi, rej_lo, guard_lo, guard_hi)
}

#[inline(always)]
fn prefix_fold_slot(table: &LoadedPrefixCheckpoint, fold_id: i32) -> Option<usize> {
    if !table.fold_slot_lut.is_empty() {
        let offset = (fold_id as i64) - (table.fold_lut_base as i64);
        if offset >= 0 {
            let offset = offset as usize;
            if offset < table.fold_slot_lut.len() {
                let slot = table.fold_slot_lut[offset];
                if slot >= 0 {
                    return Some(slot as usize);
                }
                return None;
            }
        }
        return None;
    }
    match table.fold_ids.as_slice() {
        [] => None,
        [a] => {
            if *a == fold_id {
                Some(0)
            } else {
                None
            }
        }
        [a, b] => {
            if *a == fold_id {
                Some(0)
            } else if *b == fold_id {
                Some(1)
            } else {
                None
            }
        }
        [a, b, c] => {
            if *a == fold_id {
                Some(0)
            } else if *b == fold_id {
                Some(1)
            } else if *c == fold_id {
                Some(2)
            } else {
                None
            }
        }
        [a, b, c, d] => {
            if *a == fold_id {
                Some(0)
            } else if *b == fold_id {
                Some(1)
            } else if *c == fold_id {
                Some(2)
            } else if *d == fold_id {
                Some(3)
            } else {
                None
            }
        }
        ids => {
            for (idx, id) in ids.iter().copied().enumerate() {
                if id == fold_id {
                    return Some(idx);
                }
            }
            None
        }
    }
}

#[inline(always)]
fn lookup_prefix_bands_dense(
    table: &LoadedPrefixCheckpoint,
    fold_slot: Option<usize>,
    tau_bin: Option<usize>,
    guard_lo: usize,
    guard_hi: usize,
) -> (f32, f32) {
    if let Some(slot) = fold_slot {
        if let Some(tb) = tau_bin {
            if table.tau_bin_count != 0 && !table.tau_ref_hi.is_empty() {
                let tau_slot = slot * table.tau_bin_count + tb.min(table.tau_bin_count - 1);
                if table.tau_fold_mask[tau_slot] != 0 {
                    let off = tau_slot * 256;
                    return lookup_prefix_bands_from_guarded_slices(
                        &table.tau_ref_hi[off..off + 256],
                        &table.tau_rej_lo[off..off + 256],
                        guard_lo,
                        guard_hi,
                    );
                }
            }
        }
        if !table.fold_ref_hi.is_empty() {
            let off = slot * 256;
            return lookup_prefix_bands_from_guarded_slices(
                &table.fold_ref_hi[off..off + 256],
                &table.fold_rej_lo[off..off + 256],
                guard_lo,
                guard_hi,
            );
        }
    }
    lookup_prefix_bands_from_guarded_slices(
        &table.global_ref_hi,
        &table.global_rej_lo,
        guard_lo,
        guard_hi,
    )
}

#[inline(always)]
fn lookup_prefix_bands(table: &LoadedPrefixCheckpoint, fold_id: i32, gap: f32) -> (f32, f32) {
    let bin = prefix_gap_bin(&table.gap_edges, gap);
    let (guard_lo, guard_hi) = prefix_gap_bin_guard(&table.gap_edges, gap, bin);
    lookup_prefix_bands_dense(table, prefix_fold_slot(table, fold_id), None, guard_lo, guard_hi)
}

#[inline(always)]
fn lookup_prefix_bands_with_tau(
    table: &LoadedPrefixCheckpoint,
    fold_id: i32,
    tau_used: f32,
    gap: f32,
) -> (f32, f32) {
    let bin = prefix_gap_bin(&table.gap_edges, gap);
    let (guard_lo, guard_hi) = prefix_gap_bin_guard(&table.gap_edges, gap, bin);
    let tau_bin = if table.tau_bin_count == 0 {
        None
    } else {
        Some(prefix_gap_bin(&table.tau_edges, tau_used))
    };
    lookup_prefix_bands_dense(
        table,
        prefix_fold_slot(table, fold_id),
        tau_bin,
        guard_lo,
        guard_hi,
    )
}

#[inline(always)]
fn lookup_prefix_bands_with_tau_bin(
    table: &LoadedPrefixCheckpoint,
    fold_id: i32,
    tau_bin: usize,
    gap: f32,
) -> (f32, f32) {
    lookup_prefix_bands_with_tau_bin_slot(table, prefix_fold_slot(table, fold_id), tau_bin, gap)
}

#[inline(always)]
fn lookup_prefix_bands_with_tau_bin_slot(
    table: &LoadedPrefixCheckpoint,
    fold_slot: Option<usize>,
    tau_bin: usize,
    gap: f32,
) -> (f32, f32) {
    let bin = prefix_gap_bin(&table.gap_edges, gap);
    let (guard_lo, guard_hi) = prefix_gap_bin_guard(&table.gap_edges, gap, bin);
    lookup_prefix_bands_dense(
        table,
        fold_slot,
        if table.tau_bin_count == 0 { None } else { Some(tau_bin) },
        guard_lo,
        guard_hi,
    )
}

#[inline(always)]
fn mlp_tau_bin(edges: &[f32], tau_used: f32) -> usize {
    if edges.len() < 2 {
        return 0;
    }
    prefix_gap_bin(edges, tau_used)
}

#[inline(always)]
fn atlas_tau_bin(edges: &[f32], tau_used: f32) -> usize {
    if edges.len() < 2 {
        return 0;
    }
    prefix_gap_bin(edges, tau_used).min(edges.len().saturating_sub(2))
}

#[inline(always)]
fn atlas_cluster_id_16_scalar(
    centroids: &LoadedPrefixAtlasCentroids16,
    feat: [f32; 4],
) -> usize {
    let f0 = feat[0];
    let f1 = feat[1];
    let f2 = feat[2];
    let f3 = feat[3];
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    macro_rules! upd {
        ($idx:expr) => {{
            let score = f0 * centroids.w0[$idx]
                + f1 * centroids.w1[$idx]
                + f2 * centroids.w2[$idx]
                + f3 * centroids.w3[$idx]
                + centroids.bias[$idx];
            if score > best_score {
                best_score = score;
                best_idx = $idx;
            }
        }};
    }
    upd!(0);
    upd!(1);
    upd!(2);
    upd!(3);
    upd!(4);
    upd!(5);
    upd!(6);
    upd!(7);
    upd!(8);
    upd!(9);
    upd!(10);
    upd!(11);
    upd!(12);
    upd!(13);
    upd!(14);
    upd!(15);
    best_idx
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn atlas_cluster_id_16_avx2(
    centroids: &LoadedPrefixAtlasCentroids16,
    feat: [f32; 4],
) -> usize {
    use std::arch::x86_64::{
        _mm256_cmp_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_movemask_ps,
        _mm256_set1_ps, _mm256_storeu_ps, _CMP_EQ_OQ,
    };

    let f0 = _mm256_set1_ps(feat[0]);
    let f1 = _mm256_set1_ps(feat[1]);
    let f2 = _mm256_set1_ps(feat[2]);
    let f3 = _mm256_set1_ps(feat[3]);

    let score_lo = {
        let mut acc = _mm256_loadu_ps(centroids.bias.as_ptr());
        acc = _mm256_fmadd_ps(f0, _mm256_loadu_ps(centroids.w0.as_ptr()), acc);
        acc = _mm256_fmadd_ps(f1, _mm256_loadu_ps(centroids.w1.as_ptr()), acc);
        acc = _mm256_fmadd_ps(f2, _mm256_loadu_ps(centroids.w2.as_ptr()), acc);
        _mm256_fmadd_ps(f3, _mm256_loadu_ps(centroids.w3.as_ptr()), acc)
    };
    let score_hi = {
        let mut acc = _mm256_loadu_ps(centroids.bias.as_ptr().add(8));
        acc = _mm256_fmadd_ps(f0, _mm256_loadu_ps(centroids.w0.as_ptr().add(8)), acc);
        acc = _mm256_fmadd_ps(f1, _mm256_loadu_ps(centroids.w1.as_ptr().add(8)), acc);
        acc = _mm256_fmadd_ps(f2, _mm256_loadu_ps(centroids.w2.as_ptr().add(8)), acc);
        _mm256_fmadd_ps(f3, _mm256_loadu_ps(centroids.w3.as_ptr().add(8)), acc)
    };

    let best_vec = _mm256_max_ps(score_lo, score_hi);
    let mut best_arr = [0.0f32; 8];
    _mm256_storeu_ps(best_arr.as_mut_ptr(), best_vec);
    let mut best_score = f32::NEG_INFINITY;
    for &score in &best_arr {
        if score > best_score {
            best_score = score;
        }
    }

    let best_broadcast = _mm256_set1_ps(best_score);
    let lo_mask = _mm256_movemask_ps(_mm256_cmp_ps(score_lo, best_broadcast, _CMP_EQ_OQ));
    if lo_mask != 0 {
        return lo_mask.trailing_zeros() as usize;
    }
    let hi_mask = _mm256_movemask_ps(_mm256_cmp_ps(score_hi, best_broadcast, _CMP_EQ_OQ));
    8 + hi_mask.trailing_zeros() as usize
}

#[inline(always)]
fn atlas_cluster_id_16(table: &LoadedPrefixAtlasCheckpoint, feat: [f32; 4]) -> Option<usize> {
    let centroids = table.centroids16.as_ref()?;
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Some(unsafe { atlas_cluster_id_16_avx2(centroids, feat) });
        }
    }
    Some(atlas_cluster_id_16_scalar(centroids, feat))
}

#[inline(always)]
fn atlas_cluster_id(
    table: &LoadedPrefixAtlasCheckpoint,
    gap: f32,
    delta_prev_checkpoint: f32,
    delta_from_64: f32,
) -> Option<usize> {
    if table.cluster_count == 0 {
        return None;
    }
    let feat = [
        (gap - table.feature_mean[0]) * table.feature_inv_std[0],
        (gap.abs() - table.feature_mean[1]) * table.feature_inv_std[1],
        (delta_prev_checkpoint - table.feature_mean[2]) * table.feature_inv_std[2],
        (delta_from_64 - table.feature_mean[3]) * table.feature_inv_std[3],
    ];
    if table.cluster_count == 16 {
        return atlas_cluster_id_16(table, feat);
    }
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for idx in 0..table.cluster_count {
        let score = feat[0] * table.centroid_w0[idx]
            + feat[1] * table.centroid_w1[idx]
            + feat[2] * table.centroid_w2[idx]
            + feat[3] * table.centroid_w3[idx]
            + table.centroid_bias[idx];
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    Some(best_idx)
}

#[inline(always)]
fn lookup_prefix_atlas(
    table: &LoadedPrefixAtlasCheckpoint,
    tau_bin: usize,
    gap_bin: usize,
    gap: f32,
    delta_prev_checkpoint: f32,
    delta_from_64: f32,
) -> Option<bool> {
    let cluster_id = atlas_cluster_id(table, gap, delta_prev_checkpoint, delta_from_64)?;
    if cluster_id >= table.cluster_count {
        return None;
    }
    let tau_bin = tau_bin.min(table.tau_bin_count.saturating_sub(1));
    let idx = ((tau_bin * table.cluster_count) + cluster_id) * 256 + gap_bin.min(255);
    let decision = table.decision_grid[idx];
    match decision {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    }
}

fn mlp_forward(cert: &LoadedMlpCertifier, input: &[f32]) -> f32 {
    let mut h1 = vec![0.0f32; cert.b1.len()];
    for (row_idx, (row, bias)) in cert.w1.iter().zip(cert.b1.iter()).enumerate() {
        let mut acc = *bias;
        for (w, x) in row.iter().zip(input.iter()) {
            acc += *w * *x;
        }
        h1[row_idx] = acc.max(0.0);
    }
    let mut h2 = vec![0.0f32; cert.b2.len()];
    for (row_idx, (row, bias)) in cert.w2.iter().zip(cert.b2.iter()).enumerate() {
        let mut acc = *bias;
        for (w, x) in row.iter().zip(h1.iter()) {
            acc += *w * *x;
        }
        h2[row_idx] = acc.max(0.0);
    }
    let mut logit = cert.b3;
    for (w, x) in cert.w3.iter().zip(h2.iter()) {
        logit += *w * *x;
    }
    1.0 / (1.0 + (-logit).exp())
}

fn build_mlp_input(
    cert: &LoadedMlpCertifier,
    checkpoint: usize,
    fold_id: i32,
    tau_used: f32,
    prefix_score: f32,
    gap: f32,
    delta_prev_checkpoint: f32,
    delta_from_64: f32,
    evals_per_tree: f32,
    trees_used: usize,
    resolved_early_rate_so_far: f32,
) -> Vec<f32> {
    let checkpoint_dim = cert.checkpoint_values.len();
    let fold_dim = cert.fold_values.len();
    let tau_dim = cert.tau_edges.len().saturating_sub(1).max(1);
    let mut out = vec![0.0f32; checkpoint_dim + fold_dim + tau_dim + 8];

    for (idx, cp) in cert.checkpoint_values.iter().enumerate() {
        if *cp == checkpoint {
            out[idx] = 1.0;
            break;
        }
    }
    let mut cursor = checkpoint_dim;
    for (idx, fold) in cert.fold_values.iter().enumerate() {
        if *fold == fold_id {
            out[cursor + idx] = 1.0;
            break;
        }
    }
    cursor += fold_dim;
    let tau_bin = mlp_tau_bin(&cert.tau_edges, tau_used).min(tau_dim.saturating_sub(1));
    out[cursor + tau_bin] = 1.0;
    cursor += tau_dim;
    out[cursor] = prefix_score;
    out[cursor + 1] = gap;
    out[cursor + 2] = gap.abs();
    out[cursor + 3] = delta_prev_checkpoint;
    out[cursor + 4] = delta_from_64;
    out[cursor + 5] = evals_per_tree;
    out[cursor + 6] = trees_used as f32 / 2048.0;
    out[cursor + 7] = resolved_early_rate_so_far;

    for i in 0..out.len() {
        out[i] = (out[i] - cert.mean[i]) * cert.inv_std[i];
    }
    out
}

#[derive(Debug, Default)]
struct PrefixRowShadow {
    prefix_score: f32,
    trees_used: usize,
    work_evals: u64,
    resolved_early_trees: u64,
    shadow_reject: bool,
    shadow_route_score: f32,
    fallback_used: bool,
    direct_checkpoint: usize,
    fallback_entry_checkpoint: usize,
    checkpoint_scores: Vec<f32>,
    checkpoint_work_evals: Vec<u32>,
    checkpoint_deltas: Vec<f32>,
    checkpoint_resolved_early: Vec<u32>,
    selected_packets: Vec<u16>,
    anchor_trees_used: usize,
    anchor_direct_checkpoint: usize,
    rescue_trees_used: usize,
    rescue_direct_checkpoint: usize,
    rescue_route: u8,
}

#[inline(always)]
fn maybe_prefix_certify_atlas_fast(
    table: &LoadedPrefixCheckpoint,
    atlas_table: &LoadedPrefixAtlasCheckpoint,
    row_fold_slot: Option<usize>,
    row_tau: f32,
    row_tau_bin: usize,
    row_calibration_tau_bin: usize,
    prefix_score: f32,
    delta_prev_checkpoint: f32,
    delta_from_64: f32,
) -> Option<(bool, f32)> {
    let gap = row_tau - prefix_score;
    let band_bin = prefix_gap_bin(&table.gap_edges, gap);
    let (guard_lo, guard_hi) = prefix_gap_bin_guard(&table.gap_edges, gap, band_bin);
    let (ref_hi, rej_lo) = lookup_prefix_bands_dense(
        table,
        row_fold_slot,
        if table.tau_bin_count == 0 {
            None
        } else {
            Some(row_calibration_tau_bin)
        },
        guard_lo,
        guard_hi,
    );
    if ref_hi < gap {
        return Some((false, prefix_score + ref_hi));
    }
    if rej_lo >= gap {
        return Some((true, prefix_score + rej_lo));
    }
    let atlas_gap_bin = if atlas_table.shared_gap_edges {
        band_bin.min(255)
    } else {
        prefix_gap_bin(&atlas_table.gap_edges, gap).min(255)
    };
    if let Some(is_reject) = lookup_prefix_atlas(
        atlas_table,
        row_tau_bin,
        atlas_gap_bin,
        gap,
        delta_prev_checkpoint,
        delta_from_64,
    ) {
        return Some((
            is_reject,
            if is_reject {
                row_tau + 1e-6
            } else {
                row_tau - 1e-6
            },
        ));
    }
    None
}

#[inline(always)]
fn maybe_prefix_certify(
    certifier_kind: PrefixCertifierKind,
    atlas: Option<&LoadedPrefixAtlas>,
    mlp: Option<&LoadedMlpCertifier>,
    table: &LoadedPrefixCheckpoint,
    checkpoint_slot: usize,
    checkpoint: usize,
    row_fold: i32,
    row_fold_slot: Option<usize>,
    row_tau: f32,
    row_tau_bin: usize,
    row_calibration_tau_bin: usize,
    prefix_score: f32,
    work_evals: u64,
    trees_used: usize,
    resolved_early_trees: u64,
    delta_prev_checkpoint: f32,
    delta_from_64: f32,
) -> Option<(bool, f32)> {
    let gap = row_tau - prefix_score;
    let (ref_hi, rej_lo) = lookup_prefix_bands_with_tau_bin_slot(
        table,
        row_fold_slot,
        row_calibration_tau_bin,
        gap,
    );
    if ref_hi < gap {
        return Some((false, prefix_score + ref_hi));
    }
    if rej_lo >= gap {
        return Some((true, prefix_score + rej_lo));
    }
    if certifier_kind == PrefixCertifierKind::AtlasV1 {
        if let Some(cert) = atlas {
            let atlas_table = cert.checkpoints.get(checkpoint_slot)?;
            let gap_bin = if atlas_table.shared_gap_edges {
                prefix_gap_bin(&table.gap_edges, gap).min(255)
            } else {
                prefix_gap_bin(&atlas_table.gap_edges, gap).min(255)
            };
            if let Some(is_reject) = lookup_prefix_atlas(
                atlas_table,
                row_tau_bin,
                gap_bin,
                gap,
                delta_prev_checkpoint,
                delta_from_64,
            ) {
                return Some((
                    is_reject,
                    if is_reject {
                        row_tau + 1e-6
                    } else {
                        row_tau - 1e-6
                    },
                ));
            }
        }
    }
    if certifier_kind == PrefixCertifierKind::MlpV1 {
        if let Some(cert) = mlp {
            if checkpoint <= cert.checkpoint_limit {
                if let Some(thresholds) = cert.threshold_by_checkpoint.get(&checkpoint) {
                    let evals_per_tree = if trees_used == 0 {
                        0.0
                    } else {
                        work_evals as f32 / trees_used as f32
                    };
                    let resolved_rate = if trees_used == 0 {
                        0.0
                    } else {
                        resolved_early_trees as f32 / trees_used as f32
                    };
                    let input = build_mlp_input(
                        cert,
                        checkpoint,
                        row_fold,
                        row_tau,
                        prefix_score,
                        gap,
                        delta_prev_checkpoint,
                        delta_from_64,
                        evals_per_tree,
                        trees_used,
                        resolved_rate,
                    );
                    let p_reject = mlp_forward(cert, &input);
                    if p_reject <= thresholds.tau_ref {
                        return Some((false, row_tau - 1e-6));
                    }
                    if p_reject >= thresholds.tau_rej {
                        return Some((true, row_tau + 1e-6));
                    }
                }
            }
        }
    }
    None
}

fn run_prefix_shadow_row_from_atlas(
    pack: &qs_exact::QsPack,
    hot_pack: Option<&PrefixPack>,
    hot_checkpoint_layout: Option<&HotCheckpointLayout>,
    direct_kernel: PrefixDirectKernel,
    atlas: &LoadedPrefixAtlas,
    calibration: &LoadedPrefixCal,
    feat: &[f32],
    row_tau: f32,
    row_fold: i32,
    ranks: &mut [u8],
    missing: &mut [u8],
    hot_cache: Option<&mut LazyHotFeatureCache>,
    init_score: f32,
    trace_enabled: bool,
) -> Result<PrefixRowShadow> {
    let cp_len = calibration.checkpoints.len();
    let mut out = PrefixRowShadow {
        checkpoint_scores: if trace_enabled {
            vec![f32::NAN; cp_len]
        } else {
            Vec::new()
        },
        checkpoint_work_evals: if trace_enabled { vec![0u32; cp_len] } else { Vec::new() },
        checkpoint_deltas: if trace_enabled { vec![0.0f32; cp_len] } else { Vec::new() },
        checkpoint_resolved_early: if trace_enabled {
            vec![0u32; cp_len]
        } else {
            Vec::new()
        },
        ..Default::default()
    };
    let mut last_checkpoint_score = init_score;
    let mut score_at_64: Option<f32> = None;
    let direct_decision = Cell::new(None::<(bool, f32, usize)>);
    let atlas_tau_bin = atlas_tau_bin(&atlas.tau_edges, row_tau);
    let calibration_tau_bin = if let Some(first_table) = calibration.tables.first() {
        if first_table.tau_edges.len() >= 2 {
            prefix_gap_bin(&first_table.tau_edges, row_tau)
        } else {
            0usize
        }
    } else {
        0usize
    };

    let mut handle_checkpoint =
        |global_cp_idx: usize, checkpoint: usize, score: f32, work_evals: u64, resolved_early: u64| {
            let delta_prev = if global_cp_idx == 0 {
                score - init_score
            } else {
                score - last_checkpoint_score
            };
            let delta_from_64 = if let Some(score64) = score_at_64 {
                score - score64
            } else {
                0.0
            };
            if trace_enabled {
                out.checkpoint_scores[global_cp_idx] = score;
                out.checkpoint_work_evals[global_cp_idx] =
                    work_evals.min(u32::MAX as u64) as u32;
                out.checkpoint_deltas[global_cp_idx] = delta_prev;
                out.checkpoint_resolved_early[global_cp_idx] =
                    resolved_early.min(u32::MAX as u64) as u32;
            }
            if checkpoint == 64 {
                score_at_64 = Some(score);
            }
            last_checkpoint_score = score;
            if let Some((is_reject, route_score)) = maybe_prefix_certify_atlas_fast(
                &calibration.tables[global_cp_idx],
                &atlas.checkpoints[global_cp_idx],
                prefix_fold_slot(&calibration.tables[global_cp_idx], row_fold),
                row_tau,
                atlas_tau_bin,
                calibration_tau_bin,
                score,
                delta_prev,
                delta_from_64,
            ) {
                direct_decision.set(Some((is_reject, route_score, checkpoint)));
                return true;
            }
            false
        };

    match direct_kernel {
        PrefixDirectKernel::Qs => {
            let mut qs_cp_idx = 0usize;
            let prefix_row = qs_exact::prefix_until_from(
                pack,
                feat,
                ranks,
                missing,
                &calibration.checkpoints,
                0,
                init_score,
                0,
                0,
                |checkpoint, prefix_score, work_evals, resolved_early| {
                    let global_cp_idx = qs_cp_idx;
                    qs_cp_idx += 1;
                    handle_checkpoint(
                        global_cp_idx,
                        checkpoint,
                        prefix_score,
                        work_evals,
                        resolved_early,
                    )
                },
            )?;
            out.prefix_score = prefix_row.prefix_score;
            out.trees_used = prefix_row.trees_used;
            out.work_evals = prefix_row.block_evals;
            out.resolved_early_trees = prefix_row.resolved_early_trees;
        }
        PrefixDirectKernel::HotExact96
        | PrefixDirectKernel::HotExact128
        | PrefixDirectKernel::HotExact192
        | PrefixDirectKernel::HotExact256
        | PrefixDirectKernel::HotExact384 => {
            let hot_pack = hot_pack.context("hot exact direct kernel selected without hot pack")?;
            let hot_cache = hot_cache.context("missing lazy hot cache")?;
            let checkpoint_layout = hot_checkpoint_layout
                .context("missing hot checkpoint layout for hot exact direct kernel")?;
            let hot_limit = hot_pack.n_trees.min(pack.n_trees());

            let mut score = pack.base_score;
            let mut work_evals = 0u64;
            let mut resolved_early = 0u64;
            let mut trees_used = 0usize;
            if !checkpoint_layout.hot_values.is_empty() {
                let mut hot_pos = 0usize;
                let (hot_score, hot_trees_used, hot_work, hot_resolved) = unsafe {
                    hot_exact_prefix_until(
                        hot_pack,
                        hot_cache,
                        feat,
                        &checkpoint_layout.hot_values,
                        init_score,
                        |checkpoint, prefix_score, node_evals, resolved| {
                            let global_cp_idx = checkpoint_layout.hot_indices[hot_pos];
                            hot_pos += 1;
                            handle_checkpoint(
                                global_cp_idx,
                                checkpoint,
                                prefix_score,
                                node_evals,
                                resolved,
                            )
                        },
                    )
                }?;
                score = hot_score;
                work_evals = hot_work;
                resolved_early = hot_resolved;
                trees_used = hot_trees_used;
            }
            if direct_decision.get().is_none() && !checkpoint_layout.late_values.is_empty() {
                let mut late_pos = 0usize;
                let late_row = qs_exact::prefix_until_from(
                    pack,
                    feat,
                    ranks,
                    missing,
                    &checkpoint_layout.late_values,
                    hot_limit,
                    score,
                    work_evals,
                    resolved_early,
                    |checkpoint, prefix_score, qs_work, resolved| {
                        let global_cp_idx = checkpoint_layout.late_indices[late_pos];
                        late_pos += 1;
                        handle_checkpoint(
                            global_cp_idx,
                            checkpoint,
                            prefix_score,
                            qs_work,
                            resolved,
                        )
                    },
                )?;
                score = late_row.prefix_score;
                work_evals = late_row.block_evals;
                resolved_early = late_row.resolved_early_trees;
                trees_used = late_row.trees_used;
            }
            out.prefix_score = score;
            out.trees_used = trees_used;
            out.work_evals = work_evals;
            out.resolved_early_trees = resolved_early;
        }
    }

    if let Some((is_reject, route_score, checkpoint)) = direct_decision.get() {
        out.shadow_reject = is_reject;
        out.shadow_route_score = route_score;
        out.direct_checkpoint = checkpoint;
        out.fallback_used = false;
        out.fallback_entry_checkpoint = 0;
    } else {
        out.shadow_reject = false;
        out.shadow_route_score = out.prefix_score;
        out.direct_checkpoint = 0;
        out.fallback_used = true;
        out.fallback_entry_checkpoint = *calibration.checkpoints.last().unwrap_or(&0usize);
    }

    Ok(out)
}

fn run_prefix_shadow_row_from(
    pack: &qs_exact::QsPack,
    hot_pack: Option<&PrefixPack>,
    hot_checkpoint_layout: Option<&HotCheckpointLayout>,
    direct_kernel: PrefixDirectKernel,
    certifier_kind: PrefixCertifierKind,
    atlas: Option<&LoadedPrefixAtlas>,
    mlp: Option<&LoadedMlpCertifier>,
    calibration: &LoadedPrefixCal,
    feat: &[f32],
    row_tau: f32,
    row_fold: i32,
    ranks: &mut [u8],
    missing: &mut [u8],
    hot_cache: Option<&mut LazyHotFeatureCache>,
    init_score: f32,
    trace_enabled: bool,
) -> Result<PrefixRowShadow> {
    if certifier_kind == PrefixCertifierKind::AtlasV1 {
        if let Some(atlas) = atlas {
            if mlp.is_none() {
                return run_prefix_shadow_row_from_atlas(
                    pack,
                    hot_pack,
                    hot_checkpoint_layout,
                    direct_kernel,
                    atlas,
                    calibration,
                    feat,
                    row_tau,
                    row_fold,
                    ranks,
                    missing,
                    hot_cache,
                    init_score,
                    trace_enabled,
                );
            }
        }
    }
    let cp_len = calibration.checkpoints.len();
    let mut out = PrefixRowShadow {
        checkpoint_scores: if trace_enabled {
            vec![f32::NAN; cp_len]
        } else {
            Vec::new()
        },
        checkpoint_work_evals: if trace_enabled { vec![0u32; cp_len] } else { Vec::new() },
        checkpoint_deltas: if trace_enabled { vec![0.0f32; cp_len] } else { Vec::new() },
        checkpoint_resolved_early: if trace_enabled {
            vec![0u32; cp_len]
        } else {
            Vec::new()
        },
        ..Default::default()
    };
    let mut last_checkpoint_score = init_score;
    let mut score_at_64: Option<f32> = None;
    let direct_decision = Cell::new(None::<(bool, f32, usize)>);
    let atlas_tau_bin = if let Some(cert) = atlas {
        atlas_tau_bin(&cert.tau_edges, row_tau)
    } else {
        0usize
    };
    let calibration_tau_bin = if let Some(first_table) = calibration.tables.first() {
        if first_table.tau_edges.len() >= 2 {
            prefix_gap_bin(&first_table.tau_edges, row_tau)
        } else {
            0usize
        }
    } else {
        0usize
    };

    let mut handle_checkpoint =
        |global_cp_idx: usize, checkpoint: usize, score: f32, work_evals: u64, resolved_early: u64| {
            let delta_prev = if global_cp_idx == 0 {
                score - init_score
            } else {
                score - last_checkpoint_score
            };
            let delta_from_64 = if let Some(score64) = score_at_64 {
                score - score64
            } else {
                0.0
            };
            if trace_enabled {
                out.checkpoint_scores[global_cp_idx] = score;
                out.checkpoint_work_evals[global_cp_idx] =
                    work_evals.min(u32::MAX as u64) as u32;
                out.checkpoint_deltas[global_cp_idx] = delta_prev;
                out.checkpoint_resolved_early[global_cp_idx] =
                    resolved_early.min(u32::MAX as u64) as u32;
            }
            if checkpoint == 64 {
                score_at_64 = Some(score);
            }
            last_checkpoint_score = score;
            if let Some((is_reject, route_score)) = maybe_prefix_certify(
                certifier_kind,
                atlas,
                mlp,
                &calibration.tables[global_cp_idx],
                global_cp_idx,
                checkpoint,
                row_fold,
                prefix_fold_slot(&calibration.tables[global_cp_idx], row_fold),
                row_tau,
                atlas_tau_bin,
                calibration_tau_bin,
                score,
                work_evals,
                checkpoint,
                resolved_early,
                delta_prev,
                delta_from_64,
            ) {
                direct_decision.set(Some((is_reject, route_score, checkpoint)));
                return true;
            }
            false
        };

    match direct_kernel {
        PrefixDirectKernel::Qs => {
            let mut qs_cp_idx = 0usize;
            let prefix_row = qs_exact::prefix_until_from(
                pack,
                feat,
                ranks,
                missing,
                &calibration.checkpoints,
                0,
                init_score,
                0,
                0,
                |checkpoint, prefix_score, work_evals, resolved_early| {
                    let global_cp_idx = qs_cp_idx;
                    qs_cp_idx += 1;
                    handle_checkpoint(
                        global_cp_idx,
                        checkpoint,
                        prefix_score,
                        work_evals,
                        resolved_early,
                    )
                },
            )?;
            out.prefix_score = prefix_row.prefix_score;
            out.trees_used = prefix_row.trees_used;
            out.work_evals = prefix_row.block_evals;
            out.resolved_early_trees = prefix_row.resolved_early_trees;
        }
        PrefixDirectKernel::HotExact96
        | PrefixDirectKernel::HotExact128
        | PrefixDirectKernel::HotExact192
        | PrefixDirectKernel::HotExact256
        | PrefixDirectKernel::HotExact384 => {
            let hot_pack = hot_pack.context("hot exact direct kernel selected without hot pack")?;
            let hot_cache = hot_cache.context("missing lazy hot cache")?;
            let checkpoint_layout = hot_checkpoint_layout
                .context("missing hot checkpoint layout for hot exact direct kernel")?;
            let hot_limit = hot_pack.n_trees.min(pack.n_trees());

            let mut score = pack.base_score;
            let mut work_evals = 0u64;
            let mut resolved_early = 0u64;
            let mut trees_used = 0usize;
            if !checkpoint_layout.hot_values.is_empty() {
                let mut hot_pos = 0usize;
                let (hot_score, hot_trees_used, hot_work, hot_resolved) = unsafe {
                    hot_exact_prefix_until(
                        hot_pack,
                        hot_cache,
                        feat,
                        &checkpoint_layout.hot_values,
                        init_score,
                        |checkpoint, prefix_score, node_evals, resolved| {
                            let global_cp_idx = checkpoint_layout.hot_indices[hot_pos];
                            hot_pos += 1;
                            handle_checkpoint(
                                global_cp_idx,
                                checkpoint,
                                prefix_score,
                                node_evals,
                                resolved,
                            )
                        },
                    )
                }?;
                score = hot_score;
                work_evals = hot_work;
                resolved_early = hot_resolved;
                trees_used = hot_trees_used;
            }
            if direct_decision.get().is_none() && !checkpoint_layout.late_values.is_empty() {
                let mut late_pos = 0usize;
                let late_row = qs_exact::prefix_until_from(
                    pack,
                    feat,
                    ranks,
                    missing,
                    &checkpoint_layout.late_values,
                    hot_limit,
                    score,
                    work_evals,
                    resolved_early,
                    |checkpoint, prefix_score, qs_work, resolved| {
                        let global_cp_idx = checkpoint_layout.late_indices[late_pos];
                        late_pos += 1;
                        handle_checkpoint(
                            global_cp_idx,
                            checkpoint,
                            prefix_score,
                            qs_work,
                            resolved,
                        )
                    },
                )?;
                score = late_row.prefix_score;
                work_evals = late_row.block_evals;
                resolved_early = late_row.resolved_early_trees;
                trees_used = late_row.trees_used;
            }
            out.prefix_score = score;
            out.trees_used = trees_used;
            out.work_evals = work_evals;
            out.resolved_early_trees = resolved_early;
        }
    }

    if let Some((is_reject, route_score, checkpoint)) = direct_decision.get() {
        out.shadow_reject = is_reject;
        out.shadow_route_score = route_score;
        out.direct_checkpoint = checkpoint;
        out.fallback_used = false;
        out.fallback_entry_checkpoint = 0;
    } else {
        out.shadow_reject = false;
        out.shadow_route_score = out.prefix_score;
        out.direct_checkpoint = 0;
        out.fallback_used = true;
        out.fallback_entry_checkpoint = *calibration.checkpoints.last().unwrap_or(&0usize);
    }

    Ok(out)
}

fn run_prefix_shadow_row(
    pack: &qs_exact::QsPack,
    hot_pack: Option<&PrefixPack>,
    hot_checkpoint_layout: Option<&HotCheckpointLayout>,
    direct_kernel: PrefixDirectKernel,
    certifier_kind: PrefixCertifierKind,
    atlas: Option<&LoadedPrefixAtlas>,
    mlp: Option<&LoadedMlpCertifier>,
    calibration: &LoadedPrefixCal,
    feat: &[f32],
    row_tau: f32,
    row_fold: i32,
    ranks: &mut [u8],
    missing: &mut [u8],
    hot_cache: Option<&mut LazyHotFeatureCache>,
    trace_enabled: bool,
) -> Result<PrefixRowShadow> {
    run_prefix_shadow_row_from(
        pack,
        hot_pack,
        hot_checkpoint_layout,
        direct_kernel,
        certifier_kind,
        atlas,
        mlp,
        calibration,
        feat,
        row_tau,
        row_fold,
        ranks,
        missing,
        hot_cache,
        pack.base_score,
        trace_enabled,
    )
}

fn run_prefix_shadow_row_single_route_compiled(
    runtime: &LoadedPrefixRuntime,
    feat: &[f32],
    row_tau: f32,
    row_fold: i32,
    nan_free: bool,
    ranks: &mut [u8],
    missing: &mut [u8],
    hot_buf: Option<&mut HotFeatureBuf>,
    hot_cache: Option<&mut LazyHotFeatureCache>,
) -> Result<PrefixRowShadow> {
    let Some(compiled_hot_pack) = runtime.compiled_hot_pack.as_ref() else {
        return run_prefix_shadow_row(
            &runtime.pack,
            runtime.hot_pack.as_ref(),
            runtime.hot_checkpoint_layout.as_ref(),
            runtime.direct_kernel,
            runtime.certifier_kind,
            runtime.atlas_certifier.as_ref(),
            runtime.mlp_certifier.as_ref(),
            &runtime.calibration,
            feat,
            row_tau,
            row_fold,
            ranks,
            missing,
            hot_cache,
            false,
        );
    };
    let Some(hot_checkpoint_layout) = runtime.hot_checkpoint_layout.as_ref() else {
        return run_prefix_shadow_row(
            &runtime.pack,
            runtime.hot_pack.as_ref(),
            runtime.hot_checkpoint_layout.as_ref(),
            runtime.direct_kernel,
            runtime.certifier_kind,
            runtime.atlas_certifier.as_ref(),
            runtime.mlp_certifier.as_ref(),
            &runtime.calibration,
            feat,
            row_tau,
            row_fold,
            ranks,
            missing,
            hot_cache,
            false,
        );
    };
    let Some(atlas) = runtime.atlas_certifier.as_ref() else {
        return run_prefix_shadow_row(
            &runtime.pack,
            runtime.hot_pack.as_ref(),
            runtime.hot_checkpoint_layout.as_ref(),
            runtime.direct_kernel,
            runtime.certifier_kind,
            runtime.atlas_certifier.as_ref(),
            runtime.mlp_certifier.as_ref(),
            &runtime.calibration,
            feat,
            row_tau,
            row_fold,
            ranks,
            missing,
            hot_cache,
            false,
        );
    };
    if runtime.certifier_kind != PrefixCertifierKind::AtlasV1
        || !matches!(
            runtime.direct_kernel,
            PrefixDirectKernel::HotExact96
                | PrefixDirectKernel::HotExact128
                | PrefixDirectKernel::HotExact192
                | PrefixDirectKernel::HotExact256
                | PrefixDirectKernel::HotExact384
        )
    {
        return run_prefix_shadow_row(
            &runtime.pack,
            runtime.hot_pack.as_ref(),
            runtime.hot_checkpoint_layout.as_ref(),
            runtime.direct_kernel,
            runtime.certifier_kind,
            runtime.atlas_certifier.as_ref(),
            runtime.mlp_certifier.as_ref(),
            &runtime.calibration,
            feat,
            row_tau,
            row_fold,
            ranks,
            missing,
            hot_cache,
            false,
        );
    }
    let mut out = PrefixRowShadow::default();
    let mut last_checkpoint_score = runtime.pack.base_score;
    let mut score_at_64: Option<f32> = None;
    let direct_decision = Cell::new(None::<(bool, f32, usize)>);
    let atlas_tau_bin = atlas_tau_bin(&atlas.tau_edges, row_tau);
    let calibration_tau_bin = if let Some(first_table) = runtime.calibration.tables.first() {
        if first_table.tau_edges.len() >= 2 {
            prefix_gap_bin(&first_table.tau_edges, row_tau)
        } else {
            0usize
        }
    } else {
        0usize
    };

    let mut handle_checkpoint =
        |global_cp_idx: usize, checkpoint: usize, score: f32, work_evals: u64, resolved_early: u64| {
            let delta_prev = if global_cp_idx == 0 {
                score - runtime.pack.base_score
            } else {
                score - last_checkpoint_score
            };
            let delta_from_64 = if let Some(score64) = score_at_64 {
                score - score64
            } else {
                0.0
            };
            if checkpoint == 64 {
                score_at_64 = Some(score);
            }
            last_checkpoint_score = score;
            if let Some((is_reject, route_score)) = maybe_prefix_certify_atlas_fast(
                &runtime.calibration.tables[global_cp_idx],
                &atlas.checkpoints[global_cp_idx],
                prefix_fold_slot(&runtime.calibration.tables[global_cp_idx], row_fold),
                row_tau,
                atlas_tau_bin,
                calibration_tau_bin,
                score,
                delta_prev,
                delta_from_64,
            ) {
                direct_decision.set(Some((is_reject, route_score, checkpoint)));
                return true;
            }
            let _ = work_evals;
            let _ = resolved_early;
            false
        };

    let mut score = runtime.pack.base_score;
    let mut work_evals = 0u64;
    let mut resolved_early = 0u64;
    let mut trees_used = 0usize;
    if !hot_checkpoint_layout.hot_values.is_empty() {
        let hot_buf = hot_buf.context("missing hot feature buffer for compiled hot kernel")?;
        let mut hot_pos = 0usize;
        let (hot_score, hot_trees_used, hot_work, hot_resolved) = unsafe {
            if nan_free {
                hot_exact_prefix_until_compiled_nomiss(
                    compiled_hot_pack,
                    hot_buf,
                    feat,
                    &hot_checkpoint_layout.hot_values,
                    runtime.pack.base_score,
                    |checkpoint, prefix_score, node_evals, resolved| {
                        let global_cp_idx = hot_checkpoint_layout.hot_indices[hot_pos];
                        hot_pos += 1;
                        handle_checkpoint(
                            global_cp_idx,
                            checkpoint,
                            prefix_score,
                            node_evals,
                            resolved,
                        )
                    },
                )
            } else {
                hot_exact_prefix_until_compiled(
                    compiled_hot_pack,
                    hot_buf,
                    feat,
                    &hot_checkpoint_layout.hot_values,
                    runtime.pack.base_score,
                    |checkpoint, prefix_score, node_evals, resolved| {
                        let global_cp_idx = hot_checkpoint_layout.hot_indices[hot_pos];
                        hot_pos += 1;
                        handle_checkpoint(
                            global_cp_idx,
                            checkpoint,
                            prefix_score,
                            node_evals,
                            resolved,
                        )
                    },
                )
            }
        }?;
        score = hot_score;
        work_evals = hot_work;
        resolved_early = hot_resolved;
        trees_used = hot_trees_used;
    }
    if direct_decision.get().is_none() && !hot_checkpoint_layout.late_values.is_empty() {
        let hot_limit = compiled_hot_pack.n_trees.min(runtime.pack.n_trees());
        let mut late_pos = 0usize;
        let late_row = qs_exact::prefix_until_from(
            &runtime.pack,
            feat,
            ranks,
            missing,
            &hot_checkpoint_layout.late_values,
            hot_limit,
            score,
            work_evals,
            resolved_early,
            |checkpoint, prefix_score, qs_work, resolved| {
                let global_cp_idx = hot_checkpoint_layout.late_indices[late_pos];
                late_pos += 1;
                handle_checkpoint(global_cp_idx, checkpoint, prefix_score, qs_work, resolved)
            },
        )?;
        score = late_row.prefix_score;
        work_evals = late_row.block_evals;
        resolved_early = late_row.resolved_early_trees;
        trees_used = late_row.trees_used;
    }

    out.prefix_score = score;
    out.trees_used = trees_used;
    out.work_evals = work_evals;
    out.resolved_early_trees = resolved_early;
    if let Some((is_reject, route_score, checkpoint)) = direct_decision.get() {
        out.shadow_reject = is_reject;
        out.shadow_route_score = route_score;
        out.direct_checkpoint = checkpoint;
        out.fallback_used = false;
        out.fallback_entry_checkpoint = 0;
    } else {
        out.shadow_reject = false;
        out.shadow_route_score = out.prefix_score;
        out.direct_checkpoint = 0;
        out.fallback_used = true;
        out.fallback_entry_checkpoint = *runtime
            .calibration
            .checkpoints
            .last()
            .unwrap_or(&0usize);
    }
    Ok(out)
}

fn lookup_rescue_action(
    router: &LoadedRescueRouter,
    anchor_checkpoint: usize,
    row_tau: f32,
    row_fold: i32,
    prefix_score: f32,
) -> RescueAction {
    let tau_bin = mlp_tau_bin(&router.tau_edges, row_tau).min(router.tau_edges.len().saturating_sub(2));
    let gap = row_tau - prefix_score;
    let gap_bin = prefix_gap_bin(&router.gap_edges, gap).min(router.gap_edges.len().saturating_sub(2));
    let shadow_side = if prefix_score >= row_tau { 1 } else { -1 };
    if let Some(action) =
        router
            .actions
            .get(&(anchor_checkpoint, tau_bin, row_fold, shadow_side, gap_bin))
    {
        return *action;
    }
    if let Some(action) = router.defaults.get(&(anchor_checkpoint, tau_bin, shadow_side)) {
        return *action;
    }
    RescueAction::Fallback
}

fn rescue_route_code(action: RescueAction) -> u8 {
    match action {
        RescueAction::Fallback => 1,
        RescueAction::RejectRescue => 2,
        RescueAction::ReferRescue => 3,
    }
}

fn run_anchor_rescue_shadow_row(
    runtime: &LoadedPrefixRuntime,
    anchor_rescue: &LoadedAnchorRescueRuntime,
    feat: &[f32],
    row_tau: f32,
    row_fold: i32,
    ranks: &mut [u8],
    missing: &mut [u8],
    hot_cache: Option<&mut LazyHotFeatureCache>,
    trace_enabled: bool,
) -> Result<PrefixRowShadow> {
    let mut out = run_prefix_shadow_row(
        &runtime.pack,
        runtime.hot_pack.as_ref(),
        runtime.hot_checkpoint_layout.as_ref(),
        runtime.direct_kernel,
        runtime.certifier_kind,
        runtime.atlas_certifier.as_ref(),
        runtime.mlp_certifier.as_ref(),
        &runtime.calibration,
        feat,
        row_tau,
        row_fold,
        ranks,
        missing,
        hot_cache,
        trace_enabled,
    )?;
    out.anchor_trees_used = out.trees_used;
    out.anchor_direct_checkpoint = out.direct_checkpoint;
    if !out.fallback_used {
        return Ok(out);
    }

    let action = lookup_rescue_action(
        &anchor_rescue.router,
        out.fallback_entry_checkpoint,
        row_tau,
        row_fold,
        out.prefix_score,
    );
    out.rescue_route = rescue_route_code(action);
    if action == RescueAction::Fallback {
        return Ok(out);
    }

    let rescue_runtime = match action {
        RescueAction::RejectRescue => anchor_rescue.reject_rescue.as_ref(),
        RescueAction::ReferRescue => anchor_rescue.refer_rescue.as_ref(),
        RescueAction::Fallback => unreachable!(),
    };
    let mut rescue_hot_cache = rescue_runtime
        .hot_pack
        .as_ref()
        .map(|pack| LazyHotFeatureCache::new(pack.n_hot_features));
    let rescue_shadow = run_prefix_shadow_row_from(
        &rescue_runtime.pack,
        rescue_runtime.hot_pack.as_ref(),
        rescue_runtime.hot_checkpoint_layout.as_ref(),
        rescue_runtime.direct_kernel,
        rescue_runtime.certifier_kind,
        rescue_runtime.atlas_certifier.as_ref(),
        rescue_runtime.mlp_certifier.as_ref(),
        &rescue_runtime.calibration,
        feat,
        row_tau,
        row_fold,
        ranks,
        missing,
        rescue_hot_cache.as_mut(),
        out.prefix_score,
        false,
    )?;

    out.prefix_score = rescue_shadow.prefix_score;
    out.trees_used = out.anchor_trees_used + rescue_shadow.trees_used;
    out.work_evals += rescue_shadow.work_evals;
    out.resolved_early_trees += rescue_shadow.resolved_early_trees;
    out.rescue_trees_used = rescue_shadow.trees_used;
    out.rescue_direct_checkpoint = rescue_shadow.direct_checkpoint;
    let wrong_side_direct = !rescue_shadow.fallback_used
        && ((action == RescueAction::RejectRescue && !rescue_shadow.shadow_reject)
            || (action == RescueAction::ReferRescue && rescue_shadow.shadow_reject));
    if wrong_side_direct {
        out.shadow_reject = false;
        out.shadow_route_score = out.prefix_score;
        out.fallback_used = true;
        out.direct_checkpoint = 0;
        out.rescue_direct_checkpoint = 0;
        out.fallback_entry_checkpoint = *rescue_runtime
            .calibration
            .checkpoints
            .last()
            .unwrap_or(&0usize);
    } else {
        out.shadow_reject = rescue_shadow.shadow_reject;
        out.shadow_route_score = rescue_shadow.shadow_route_score;
        out.fallback_used = rescue_shadow.fallback_used;
        out.direct_checkpoint = rescue_shadow.direct_checkpoint;
        out.fallback_entry_checkpoint = rescue_shadow.fallback_entry_checkpoint;
    }
    Ok(out)
}

fn lookup_packet_policy_ranking<'a>(
    policy: &'a LoadedPacketPolicy,
    step: usize,
    tau_bin: usize,
    gap_bin: usize,
    delta_bin: usize,
) -> Option<&'a [u16]> {
    if let Some(ranking) = policy
        .state_rankings
        .get(&(step, tau_bin, gap_bin, delta_bin))
    {
        return Some(ranking.as_slice());
    }
    if let Some(ranking) = policy.step_tau_defaults.get(&(step, tau_bin)) {
        return Some(ranking.as_slice());
    }
    policy.step_defaults.get(&step).map(|v| v.as_slice())
}

fn run_packet_shadow_row(
    runtime: &LoadedPrefixRuntime,
    scheduler: &LoadedPacketScheduler,
    feat: &[f32],
    row_tau: f32,
    row_fold: i32,
    ranks: &mut [u8],
    missing: &mut [u8],
    trace_enabled: bool,
) -> Result<PrefixRowShadow> {
    let cp_len = runtime.calibration.checkpoints.len();
    let mut out = PrefixRowShadow {
        checkpoint_scores: if trace_enabled {
            vec![f32::NAN; cp_len]
        } else {
            Vec::new()
        },
        checkpoint_work_evals: if trace_enabled { vec![0u32; cp_len] } else { Vec::new() },
        checkpoint_deltas: if trace_enabled { vec![0.0f32; cp_len] } else { Vec::new() },
        checkpoint_resolved_early: if trace_enabled {
            vec![0u32; cp_len]
        } else {
            Vec::new()
        },
        ..Default::default()
    };
    if runtime.calibration.tables.len() != scheduler.max_steps {
        bail!(
            "packet scheduler calibration length mismatch: tables={} max_steps={}",
            runtime.calibration.tables.len(),
            scheduler.max_steps
        );
    }
    qs_exact::quantize_into(&runtime.pack, feat, ranks, missing);
    let tau_bin = mlp_tau_bin(&scheduler.policy.tau_edges, row_tau)
        .min(scheduler.policy.tau_edges.len().saturating_sub(2));
    let mut used_packets = vec![false; scheduler.packets.len()];
    let mut prefix_score = runtime.pack.base_score;
    let mut trees_used = 0usize;
    let mut work_evals = 0u64;
    let mut resolved_early_trees = 0u64;
    let mut prev_delta = 0.0f32;
    let calibration_tau_bin = if let Some(first_table) = runtime.calibration.tables.first() {
        if first_table.tau_edges.len() >= 2 {
            prefix_gap_bin(&first_table.tau_edges, row_tau)
        } else {
            0usize
        }
    } else {
        0usize
    };
    let mut selected_packets = Vec::with_capacity(scheduler.max_steps);

    for step in 0..scheduler.max_steps {
        let gap = row_tau - prefix_score;
        let gap_bin = prefix_gap_bin(&scheduler.policy.gap_edges_by_step[step], gap);
        let delta_bin =
            prefix_gap_bin(&scheduler.policy.delta_edges_by_step[step], prev_delta);
        let ranking = lookup_packet_policy_ranking(
            &scheduler.policy,
            step,
            tau_bin,
            gap_bin,
            delta_bin,
        );
        let mut selected_packet: Option<&LoadedPacket> = None;
        if let Some(ranking) = ranking {
            for &packet_id in ranking {
                let idx = packet_id as usize;
                if idx < used_packets.len() && !used_packets[idx] {
                    selected_packet = Some(&scheduler.packets[idx]);
                    break;
                }
            }
        }
        if selected_packet.is_none() {
            for packet in scheduler.packets.iter() {
                if !used_packets[packet.packet_id] {
                    selected_packet = Some(packet);
                    break;
                }
            }
        }
        let packet = selected_packet.context("packet scheduler exhausted all packets")?;
        used_packets[packet.packet_id] = true;
        selected_packets.push(packet.packet_id.min(u16::MAX as usize) as u16);
        let (packet_score, packet_blocks, packet_resolved) =
            if let Some(compiled_pack) = packet.compiled_pack.as_ref() {
                let (score, _node_evals, resolved) =
                    unsafe { hot_exact_score_full(compiled_pack, feat)? };
                let block_proxy = packet.block_cost.max(0.0).round() as u64;
                (score, block_proxy, resolved)
            } else {
                qs_exact::score_tree_indices_from_quantized(
                    &runtime.pack,
                    &packet.tree_indices,
                    ranks,
                    missing,
                )?
            };
        prefix_score += packet_score;
        trees_used += packet.tree_count;
        work_evals += packet_blocks;
        resolved_early_trees += packet_resolved;
        if trace_enabled {
            out.checkpoint_scores[step] = prefix_score;
            out.checkpoint_work_evals[step] =
                work_evals.min(u32::MAX as u64) as u32;
            out.checkpoint_deltas[step] = packet_score;
            out.checkpoint_resolved_early[step] =
                resolved_early_trees.min(u32::MAX as u64) as u32;
        }
        if let Some((is_reject, route_score)) = maybe_prefix_certify(
            PrefixCertifierKind::TableV1,
            None,
            None,
            &runtime.calibration.tables[step],
            step,
            runtime.calibration.checkpoints[step],
            row_fold,
            prefix_fold_slot(&runtime.calibration.tables[step], row_fold),
            row_tau,
            0,
            calibration_tau_bin,
            prefix_score,
            work_evals,
            trees_used,
            resolved_early_trees,
            packet_score,
            0.0,
        ) {
            out.prefix_score = prefix_score;
            out.trees_used = trees_used;
            out.work_evals = work_evals;
            out.resolved_early_trees = resolved_early_trees;
            out.shadow_reject = is_reject;
            out.shadow_route_score = route_score;
            out.fallback_used = false;
            out.direct_checkpoint = runtime.calibration.checkpoints[step];
            out.fallback_entry_checkpoint = 0;
            out.selected_packets = selected_packets;
            return Ok(out);
        }
        prev_delta = packet_score;
    }

    out.prefix_score = prefix_score;
    out.trees_used = trees_used;
    out.work_evals = work_evals;
    out.resolved_early_trees = resolved_early_trees;
    out.shadow_reject = false;
    out.shadow_route_score = prefix_score;
    out.fallback_used = true;
    out.direct_checkpoint = 0;
    out.fallback_entry_checkpoint = *runtime.calibration.checkpoints.last().unwrap_or(&0usize);
    out.selected_packets = selected_packets;
    Ok(out)
}

#[inline(always)]
fn run_exact_continuation(
    runtime: &LoadedPrefixRuntime,
    model: &SoaModel,
    feat: &[f32],
    row_tau: f32,
    shadow_row: &PrefixRowShadow,
    nan_free: bool,
    ranks: &mut [u8],
    missing: &mut [u8],
) -> Result<(f32, bool, i32, RowMeta)> {
    if let Some(anchor_rescue) = runtime.anchor_rescue.as_ref() {
        qs_exact::quantize_into(&runtime.pack, feat, ranks, missing);
        let mut executed = vec![false; runtime.plan.order.len()];
        for pos in 0..shadow_row.anchor_trees_used.min(executed.len()) {
            executed[pos] = true;
        }
        if shadow_row.rescue_route == 2 || shadow_row.rescue_route == 3 {
            let rescue_runtime = if shadow_row.rescue_route == 2 {
                anchor_rescue.reject_rescue.as_ref()
            } else {
                anchor_rescue.refer_rescue.as_ref()
            };
            let mut anchor_pos_by_tree = HashMap::new();
            for (pos, tree_id) in runtime.plan.order.iter().copied().enumerate() {
                anchor_pos_by_tree.insert(tree_id, pos);
            }
            for &tree_id in rescue_runtime
                .plan
                .order
                .iter()
                .take(shadow_row.rescue_trees_used)
            {
                if let Some(pos) = anchor_pos_by_tree.get(&tree_id) {
                    executed[*pos] = true;
                }
            }
        }
        let remaining = runtime
            .plan
            .order
            .iter()
            .enumerate()
            .filter_map(|(pos, _)| (!executed[pos]).then_some(pos as u32))
            .collect::<Vec<_>>();
        let (remaining_score, _, _) = qs_exact::score_tree_indices_from_quantized(
            &runtime.pack,
            &remaining,
            ranks,
            missing,
        )?;
        let score = shadow_row.prefix_score + remaining_score;
        let reject = score >= row_tau;
        Ok((score, reject, remaining.len() as i32, RowMeta::default()))
    } else if let Some(packet_runtime) = runtime.packet_scheduler.as_ref() {
        let mut used_packets = vec![false; packet_runtime.packets.len()];
        for &packet_id in shadow_row.selected_packets.iter() {
            let idx = packet_id as usize;
            if idx >= used_packets.len() {
                bail!(
                    "invalid selected packet id {} for n_packets={}",
                    idx,
                    used_packets.len()
                );
            }
            used_packets[idx] = true;
        }
        let mut score = shadow_row.prefix_score;
        let mut visited = 0i32;
        for packet in packet_runtime.packets.iter() {
            if used_packets[packet.packet_id] {
                continue;
            }
            let (packet_score, _, _) = if let Some(compiled_pack) = packet.compiled_pack.as_ref() {
                unsafe { hot_exact_score_full(compiled_pack, feat)? }
            } else {
                qs_exact::score_tree_indices_from_quantized(
                    &runtime.pack,
                    &packet.tree_indices,
                    ranks,
                    missing,
                )?
            };
            score += packet_score;
            visited += packet.tree_count as i32;
        }
        Ok((score, score >= row_tau, visited, RowMeta::default()))
    } else {
        let fallback_start_tree_idx = shadow_row.trees_used;
        let fallback_init_score = shadow_row.prefix_score;
        if nan_free {
            unsafe {
                traverse_float_nomiss_from(
                    model,
                    feat,
                    row_tau,
                    fallback_init_score,
                    InferMode::L2RouteExactReordered,
                    &runtime.plan,
                    None,
                    0.0,
                    0.0,
                    1,
                    fallback_start_tree_idx,
                )
            }
        } else {
            traverse_generic_from(
                model,
                None,
                None,
                feat,
                row_tau,
                fallback_init_score,
                InferMode::L2RouteExactReordered,
                &runtime.plan,
                None,
                0.0,
                0.0,
                1,
                fallback_start_tree_idx,
            )
        }
    }
}

#[inline(always)]
unsafe fn hot_exact_score_full(hot_pack: &PrefixPack, feat: &[f32]) -> Result<(f32, u64, u64)> {
    let mut score = 0.0f32;
    let mut node_evals = 0u64;
    let resolved_early = 0u64;

    for pos in 0..hot_pack.n_trees {
        let mut idx = *hot_pack.tree_roots.get_unchecked(pos) as usize;
        loop {
            if *hot_pack.is_leaf.get_unchecked(idx) != 0 {
                score += *hot_pack.leaf.get_unchecked(idx);
                break;
            }
            let hot_fidx = *hot_pack.fidx.get_unchecked(idx) as usize;
            let global_fidx = *hot_pack.hot_global_fidx.get_unchecked(hot_fidx) as usize;
            let x = *feat.get_unchecked(global_fidx);
            node_evals += 1;
            idx = if x.is_nan() {
                *hot_pack.missing.get_unchecked(idx) as usize
            } else if x < *hot_pack.thr.get_unchecked(idx) {
                *hot_pack.left.get_unchecked(idx) as usize
            } else {
                *hot_pack.right.get_unchecked(idx) as usize
            };
        }
    }

    Ok((score, node_evals, resolved_early))
}

fn parse_dispatch_threshold_mode(raw: &str) -> Result<DispatchThresholdMode> {
    match raw {
        "tau" => Ok(DispatchThresholdMode::Tau),
        "zero" => Ok(DispatchThresholdMode::Zero),
        other => bail!("unsupported threshold_mode: {}", other),
    }
}

fn load_dispatch_slot(slot: i32, label: String, model_dir: &PathBuf, threshold_mode: DispatchThresholdMode) -> Result<LoadedDispatchSlot> {
    let soa = load_soa(&model_dir.join("pack/model_soa.bin"))?;
    let bounds = load_bounds(&model_dir.join("pack/bounds.bin"))?;
    let raw_tree_order = load_tree_order(&model_dir.join("tree_order.bin"))?;
    let plan = materialize_tree_plan(&bounds, Some(&raw_tree_order), None)?;
    let base_score = parse_base_score(&model_dir.join("xgb_model.json"))?;
    Ok(LoadedDispatchSlot {
        slot,
        label,
        threshold_mode,
        model: soa,
        plan,
        base_score,
    })
}

fn load_approx_policy(path: &PathBuf) -> Result<ApproxPolicy> {
    let txt = fs::read_to_string(path)
        .with_context(|| format!("read approx policy failed: {}", path.display()))?;
    let mut policy: ApproxPolicy =
        serde_json::from_str(&txt).context("parse approx policy json failed")?;
    let has_used = !policy.used_checkpoints.is_empty()
        || !policy.used_tau_ref.is_empty()
        || !policy.used_tau_pass.is_empty()
        || !policy.used_tau_reject.is_empty();
    let used_pos = if !policy.used_tau_reject.is_empty() {
        policy.used_tau_reject.len()
    } else {
        policy.used_tau_pass.len()
    };
    let raw_pos = if !policy.tau_reject.is_empty() {
        policy.tau_reject.len()
    } else {
        policy.tau_pass.len()
    };
    if has_used {
        if policy.used_checkpoints.len() != policy.used_tau_ref.len()
            || policy.used_checkpoints.len() != used_pos
        {
            bail!("approx policy compacted length mismatch");
        }
    } else if policy.checkpoints.len() != policy.tau_ref.len()
        || policy.checkpoints.len() != raw_pos
    {
        bail!("approx policy length mismatch");
    }
    if !has_used {
        policy.used_checkpoints = policy.checkpoints.clone();
        policy.used_tau_ref = policy.tau_ref.clone();
        if !policy.tau_reject.is_empty() {
            policy.used_tau_reject = policy.tau_reject.clone();
        } else {
            policy.used_tau_pass = policy.tau_pass.clone();
        }
    } else if policy.used_tau_reject.is_empty() && !policy.used_tau_pass.is_empty() {
        // pass-style positive threshold
    } else if policy.used_tau_pass.is_empty() && !policy.used_tau_reject.is_empty() {
        // reject-style positive threshold
    }
    if policy.k_hot == 0 {
        policy.k_hot = policy.used_checkpoints.iter().copied().max().unwrap_or(0);
    }
    if policy.checkpoint_exit_counts.len() != policy.used_checkpoints.len() {
        policy.checkpoint_exit_counts = vec![0u64; policy.used_checkpoints.len()];
    }
    Ok(policy)
}

impl ApproxPolicy {
    #[inline(always)]
    fn used_tau_positive(&self) -> &[Option<f32>] {
        if !self.used_tau_reject.is_empty() {
            &self.used_tau_reject
        } else {
            &self.used_tau_pass
        }
    }
}

fn load_rank_pack(path: &PathBuf) -> Result<RankPack> {
    let buf =
        fs::read(path).with_context(|| format!("read rank pack failed: {}", path.display()))?;
    if buf.len() < MAGIC_RNK.len() + 12 {
        bail!("rank pack too short");
    }
    if &buf[0..MAGIC_RNK.len()] != MAGIC_RNK {
        bail!("invalid rank pack magic");
    }
    let mut off = MAGIC_RNK.len();
    let n_features = le_u32(&buf, &mut off)? as usize;
    let n_nodes = le_u32(&buf, &mut off)? as usize;
    let n_values = le_u32(&buf, &mut off)? as usize;

    let mut feat_offset = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        feat_offset.push(le_u32(&buf, &mut off)?);
    }
    let mut feat_count = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        feat_count.push(le_u32(&buf, &mut off)?);
    }
    let mut node_thr_rank = Vec::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        node_thr_rank.push(le_i32(&buf, &mut off)?);
    }
    let mut threshold_values = Vec::with_capacity(n_values);
    for _ in 0..n_values {
        threshold_values.push(le_f32(&buf, &mut off)?);
    }

    Ok(RankPack {
        n_features,
        node_thr_rank,
        feat_offset,
        feat_count,
        threshold_values,
    })
}

fn load_features_v1(buf: &[u8], off: usize, n_rows: usize, n_cols: usize) -> Result<FeatureBatch> {
    let mut cursor = off;
    let n_feat = n_rows
        .checked_mul(n_cols)
        .context("feature size overflow")?;
    let mut x = Vec::with_capacity(n_feat);
    for _ in 0..n_feat {
        x.push(le_f32(buf, &mut cursor)?);
    }
    let mut ids = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        ids.push(le_i64(buf, &mut cursor)?);
    }
    let mut y = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        if cursor >= buf.len() {
            bail!("feature bin v1 truncated on labels");
        }
        y.push(buf[cursor]);
        cursor += 1;
    }
    Ok(FeatureBatch {
        n_rows,
        n_cols,
        storage: FeatureStorage::Owned { x, ids, y },
        format_tag: "v1-owned".to_string(),
        nan_free: true,
    })
}

fn load_features(path: &PathBuf) -> Result<FeatureBatch> {
    let file = fs::File::open(path)
        .with_context(|| format!("open feat bin failed: {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("mmap feat bin failed: {}", path.display()))?;
    let buf = &mmap[..];
    if buf.len() < MAGIC_FEA_V1.len() + 8 {
        bail!("feature bin too short");
    }

    if &buf[0..MAGIC_FEA_V1.len()] == MAGIC_FEA_V1 {
        let mut off = MAGIC_FEA_V1.len();
        let n_rows = le_u32(buf, &mut off)? as usize;
        let n_cols = le_u32(buf, &mut off)? as usize;
        return load_features_v1(buf, off, n_rows, n_cols);
    }

    if &buf[0..MAGIC_FEA_V2.len()] == MAGIC_FEA_V2 {
        let mut off = MAGIC_FEA_V2.len();
        let n_rows = le_u32(buf, &mut off)? as usize;
        let n_cols = le_u32(buf, &mut off)? as usize;
        let x_offset = le_u64(buf, &mut off)? as usize;
        let ids_offset = le_u64(buf, &mut off)? as usize;
        let y_offset = le_u64(buf, &mut off)? as usize;
        return Ok(FeatureBatch {
            n_rows,
            n_cols,
            storage: FeatureStorage::Mmap {
                mmap,
                x_offset,
                ids_offset,
                y_offset,
            },
            format_tag: "v2-mmap".to_string(),
            nan_free: true,
        });
    }

    bail!("unknown feature bin magic")
}

impl FeatureBatch {
    fn row(&self, row_idx: usize) -> &[f32] {
        match &self.storage {
            FeatureStorage::Owned { x, .. } => {
                let st = row_idx * self.n_cols;
                &x[st..st + self.n_cols]
            }
            FeatureStorage::Mmap { mmap, x_offset, .. } => {
                let byte_off = *x_offset + row_idx * self.n_cols * 4;
                unsafe {
                    std::slice::from_raw_parts(
                        mmap.as_ptr().add(byte_off) as *const f32,
                        self.n_cols,
                    )
                }
            }
        }
    }

    fn id(&self, row_idx: usize) -> i64 {
        match &self.storage {
            FeatureStorage::Owned { ids, .. } => ids[row_idx],
            FeatureStorage::Mmap {
                mmap, ids_offset, ..
            } => {
                let byte_off = *ids_offset + row_idx * 8;
                unsafe { (mmap.as_ptr().add(byte_off) as *const i64).read_unaligned() }
            }
        }
    }

    fn label(&self, row_idx: usize) -> u8 {
        match &self.storage {
            FeatureStorage::Owned { y, .. } => y[row_idx],
            FeatureStorage::Mmap { mmap, y_offset, .. } => mmap[*y_offset + row_idx],
        }
    }
}

fn load_threshold(policy_path: &PathBuf, override_thr: Option<f32>) -> Result<f32> {
    if let Some(v) = override_thr {
        return Ok(v);
    }
    let txt = fs::read_to_string(policy_path)
        .with_context(|| format!("read policy failed: {}", policy_path.display()))?;
    let v: Value = serde_json::from_str(&txt).context("parse policy json failed")?;
    let thr = v
        .get("decision")
        .and_then(|d| d.get("threshold"))
        .and_then(|t| t.as_f64())
        .context("policy decision.threshold missing")?;
    Ok(thr as f32)
}

fn parse_base_score(model_json_path: &PathBuf) -> Result<f32> {
    let txt = fs::read_to_string(model_json_path)
        .with_context(|| format!("read model json failed: {}", model_json_path.display()))?;
    let v: Value = serde_json::from_str(&txt).context("parse model json failed")?;
    let raw = v
        .get("learner")
        .and_then(|x| x.get("learner_model_param"))
        .and_then(|x| x.get("base_score"))
        .context("model learner.learner_model_param.base_score missing")?;
    let as_str = if let Some(s) = raw.as_str() {
        s.to_string()
    } else {
        raw.to_string()
    };
    let cleaned = as_str
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();
    cleaned
        .parse::<f32>()
        .with_context(|| format!("failed to parse base_score: {}", as_str))
}

fn peak_rss_mb() -> f64 {
    let Ok(txt) = fs::read_to_string("/proc/self/status") else {
        return 0.0;
    };
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb = rest
                .split_whitespace()
                .next()
                .and_then(|x| x.parse::<f64>().ok())
                .unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

fn active_threads(requested: usize) -> usize {
    if requested > 0 {
        return requested;
    }
    std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1)
}

pub(crate) fn build_thread_pool(requested: usize) -> Result<Option<rayon::ThreadPool>> {
    let n = active_threads(requested);
    if n <= 1 {
        return Ok(None);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .context("build rayon thread pool failed")?;
    Ok(Some(pool))
}

pub(crate) fn install_in_pool<T, F>(pool: Option<&rayon::ThreadPool>, f: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    if let Some(pool) = pool {
        pool.install(f)
    } else {
        f()
    }
}

fn is_route_mode(mode: InferMode) -> bool {
    matches!(
        mode,
        InferMode::RouteFast
            | InferMode::RouteExactReordered
            | InferMode::L2RouteExactReordered
    )
}

fn is_l2_route_mode(mode: InferMode) -> bool {
    matches!(mode, InferMode::L2RouteExactReordered | InferMode::L2RouteApprox)
}

fn is_approx_mode(mode: InferMode) -> bool {
    matches!(mode, InferMode::RouteApprox | InferMode::L2RouteApprox)
}

fn score_col_name(mode: InferMode) -> &'static str {
    match mode {
        InferMode::MarginExact | InferMode::MarginExactReordered => "l1_score",
        InferMode::RouteFast | InferMode::RouteExactReordered | InferMode::RouteApprox => "prefix_margin",
        InferMode::L2RouteExactReordered | InferMode::L2RouteApprox => "route_score",
    }
}

fn decision_col_name(mode: InferMode) -> &'static str {
    if is_l2_route_mode(mode) {
        "l2_decision"
    } else {
        "l1_decision"
    }
}

fn positive_label(mode: InferMode) -> &'static str {
    if is_l2_route_mode(mode) {
        "REJECT"
    } else {
        "PASS"
    }
}

fn negative_label(mode: InferMode) -> &'static str {
    "REFER"
}

fn materialize_tree_plan(
    bounds: &Bounds,
    tree_order: Option<&TreeOrder>,
    max_trees: Option<usize>,
) -> Result<TreeOrder> {
    let limit = max_trees.unwrap_or(bounds.n_trees).min(bounds.n_trees);
    if let Some(ord) = tree_order {
        if ord.n_trees != bounds.n_trees {
            bail!(
                "tree order mismatch: tree_order={} bounds={}",
                ord.n_trees,
                bounds.n_trees
            );
        }
        return Ok(TreeOrder {
            n_trees: limit,
            order: ord.order[..limit].to_vec(),
            suffix_min: ord.suffix_min[..=limit].to_vec(),
            suffix_max: ord.suffix_max[..=limit].to_vec(),
        });
    }
    Ok(TreeOrder {
        n_trees: limit,
        order: (0..limit as u32).collect(),
        suffix_min: bounds.suffix_min[..=limit].to_vec(),
        suffix_max: bounds.suffix_max[..=limit].to_vec(),
    })
}

struct RankCache {
    values: Vec<u32>,
    stamps: Vec<u32>,
    epoch: u32,
}

struct HotFeatureBuf {
    values: Vec<f32>,
}

struct LazyHotFeatureCache {
    values: Vec<f32>,
    stamps: Vec<u32>,
    epoch: u32,
}

impl HotFeatureBuf {
    fn new(n_hot_features: usize) -> Self {
        Self {
            values: vec![0.0; n_hot_features],
        }
    }

    #[inline(always)]
    unsafe fn fill_from_row(&mut self, pack: &PrefixPack, feat: &[f32]) {
        self.fill_from_global_u32(&pack.hot_global_fidx, feat);
    }

    #[inline(always)]
    unsafe fn fill_from_global_u32(&mut self, hot_global_fidx: &[u32], feat: &[f32]) {
        let mut i = 0usize;
        while i < hot_global_fidx.len() {
            let g = *hot_global_fidx.get_unchecked(i) as usize;
            *self.values.get_unchecked_mut(i) = *feat.get_unchecked(g);
            i += 1;
        }
    }
}

impl LazyHotFeatureCache {
    fn new(n_hot_features: usize) -> Self {
        Self {
            values: vec![0.0; n_hot_features],
            stamps: vec![0u32; n_hot_features],
            epoch: 1,
        }
    }

    #[inline(always)]
    fn next_row(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamps.fill(0);
            self.epoch = 1;
        }
    }

    #[inline(always)]
    unsafe fn get(&mut self, pack: &PrefixPack, feat: &[f32], hot_fidx: usize) -> f32 {
        if *self.stamps.get_unchecked(hot_fidx) == self.epoch {
            return *self.values.get_unchecked(hot_fidx);
        }
        let global_fidx = *pack.hot_global_fidx.get_unchecked(hot_fidx) as usize;
        let x = *feat.get_unchecked(global_fidx);
        *self.values.get_unchecked_mut(hot_fidx) = x;
        *self.stamps.get_unchecked_mut(hot_fidx) = self.epoch;
        x
    }
}

impl RankCache {
    fn new(n_features: usize) -> Self {
        Self {
            values: vec![0u32; n_features],
            stamps: vec![0u32; n_features],
            epoch: 1,
        }
    }

    #[inline(always)]
    fn next_row(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamps.fill(0);
            self.epoch = 1;
        }
    }
}

#[inline(always)]
fn upper_bound_auto(vals: &[f32], x: f32) -> u32 {
    if vals.len() <= 16 {
        let mut i = 0usize;
        while i < vals.len() {
            if vals[i] > x {
                return i as u32;
            }
            i += 1;
        }
        vals.len() as u32
    } else {
        let mut lo = 0usize;
        let mut hi = vals.len();
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if vals[mid] <= x {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo as u32
    }
}

#[inline(always)]
unsafe fn cached_rank(pack: &RankPack, cache: &mut RankCache, feat: &[f32], fidx: usize) -> u32 {
    let stamp = *cache.stamps.get_unchecked(fidx);
    if stamp == cache.epoch {
        return *cache.values.get_unchecked(fidx);
    }
    let x = *feat.get_unchecked(fidx);
    let st = *pack.feat_offset.get_unchecked(fidx) as usize;
    let ct = *pack.feat_count.get_unchecked(fidx) as usize;
    let vals = pack.threshold_values.get_unchecked(st..st + ct);
    let rank = upper_bound_auto(vals, x);
    *cache.values.get_unchecked_mut(fidx) = rank;
    *cache.stamps.get_unchecked_mut(fidx) = cache.epoch;
    rank
}

#[inline(always)]
unsafe fn hot_exact_prefix_until<F>(
    hot_pack: &PrefixPack,
    cache: &mut LazyHotFeatureCache,
    feat: &[f32],
    checkpoints: &[usize],
    init_score: f32,
    mut on_checkpoint: F,
) -> Result<(f32, usize, u64, u64)>
where
    F: FnMut(usize, f32, u64, u64) -> bool,
{
    if checkpoints.is_empty() {
        bail!("hot exact checkpoints must not be empty");
    }
    let mut prev = 0usize;
    for &cp in checkpoints {
        if cp == 0 || cp > hot_pack.n_trees {
            bail!(
                "invalid hot exact checkpoint {} for n_trees={}",
                cp,
                hot_pack.n_trees
            );
        }
        if cp <= prev {
            bail!("hot exact checkpoints must be strictly increasing");
        }
        prev = cp;
    }

    cache.next_row();
    let max_checkpoint = *checkpoints.last().unwrap_or(&0usize);
    let mut score = init_score;
    let mut node_evals = 0u64;
    let resolved_early = 0u64;
    let mut next_checkpoint_idx = 0usize;

    for pos in 0..max_checkpoint {
        let mut idx = *hot_pack.tree_roots.get_unchecked(pos) as usize;
        loop {
            if *hot_pack.is_leaf.get_unchecked(idx) != 0 {
                score += *hot_pack.leaf.get_unchecked(idx);
                break;
            }
            let hot_fidx = *hot_pack.fidx.get_unchecked(idx) as usize;
            let x = cache.get(hot_pack, feat, hot_fidx);
            node_evals += 1;
            idx = if x.is_nan() {
                *hot_pack.missing.get_unchecked(idx) as usize
            } else if x < *hot_pack.thr.get_unchecked(idx) {
                *hot_pack.left.get_unchecked(idx) as usize
            } else {
                *hot_pack.right.get_unchecked(idx) as usize
            };
        }
        let completed_trees = pos + 1;
        if completed_trees == checkpoints[next_checkpoint_idx] {
            if on_checkpoint(completed_trees, score, node_evals, resolved_early) {
                return Ok((score, completed_trees, node_evals, resolved_early));
            }
            next_checkpoint_idx += 1;
            if next_checkpoint_idx >= checkpoints.len() {
                return Ok((score, completed_trees, node_evals, resolved_early));
            }
        }
    }
    bail!("failed to reach final hot exact checkpoint");
}

#[inline(always)]
unsafe fn hot_exact_prefix_until_compiled_nomiss<F>(
    hot_pack: &HotCompiledPrefixPack,
    hot_buf: &mut HotFeatureBuf,
    feat: &[f32],
    checkpoints: &[usize],
    init_score: f32,
    mut on_checkpoint: F,
) -> Result<(f32, usize, u64, u64)>
where
    F: FnMut(usize, f32, u64, u64) -> bool,
{
    if checkpoints.is_empty() {
        bail!("compiled hot exact checkpoints must not be empty");
    }
    let mut prev = 0usize;
    for &cp in checkpoints {
        if cp == 0 || cp > hot_pack.n_trees {
            bail!(
                "invalid compiled hot exact checkpoint {} for n_trees={}",
                cp,
                hot_pack.n_trees
            );
        }
        if cp <= prev {
            bail!("compiled hot exact checkpoints must be strictly increasing");
        }
        prev = cp;
    }

    hot_buf.fill_from_global_u32(&hot_pack.hot_global_fidx, feat);
    let hot_feat = hot_buf.values.as_slice();
    let max_checkpoint = *checkpoints.last().unwrap_or(&0usize);
    let mut score = init_score;
    let mut node_evals = 0u64;
    let resolved_early = 0u64;
    let mut next_checkpoint_idx = 0usize;

    for pos in 0..max_checkpoint {
        let mut idx = *hot_pack.tree_roots.get_unchecked(pos) as usize;
        loop {
            let node = *hot_pack.nodes.get_unchecked(idx);
            if node.is_leaf() {
                score += node.leaf;
                break;
            }
            let x = *hot_feat.get_unchecked(node.hot_fidx());
            node_evals += 1;
            idx = if x < node.thr {
                node.left as usize
            } else {
                node.right as usize
            };
        }
        let completed_trees = pos + 1;
        if completed_trees == checkpoints[next_checkpoint_idx] {
            if on_checkpoint(completed_trees, score, node_evals, resolved_early) {
                return Ok((score, completed_trees, node_evals, resolved_early));
            }
            next_checkpoint_idx += 1;
            if next_checkpoint_idx >= checkpoints.len() {
                return Ok((score, completed_trees, node_evals, resolved_early));
            }
        }
    }
    bail!("failed to reach final compiled hot exact checkpoint");
}

#[inline(always)]
unsafe fn hot_exact_prefix_until_compiled<F>(
    hot_pack: &HotCompiledPrefixPack,
    hot_buf: &mut HotFeatureBuf,
    feat: &[f32],
    checkpoints: &[usize],
    init_score: f32,
    mut on_checkpoint: F,
) -> Result<(f32, usize, u64, u64)>
where
    F: FnMut(usize, f32, u64, u64) -> bool,
{
    if checkpoints.is_empty() {
        bail!("compiled hot exact checkpoints must not be empty");
    }
    let mut prev = 0usize;
    for &cp in checkpoints {
        if cp == 0 || cp > hot_pack.n_trees {
            bail!(
                "invalid compiled hot exact checkpoint {} for n_trees={}",
                cp,
                hot_pack.n_trees
            );
        }
        if cp <= prev {
            bail!("compiled hot exact checkpoints must be strictly increasing");
        }
        prev = cp;
    }

    hot_buf.fill_from_global_u32(&hot_pack.hot_global_fidx, feat);
    let hot_feat = hot_buf.values.as_slice();
    let max_checkpoint = *checkpoints.last().unwrap_or(&0usize);
    let mut score = init_score;
    let mut node_evals = 0u64;
    let resolved_early = 0u64;
    let mut next_checkpoint_idx = 0usize;

    for pos in 0..max_checkpoint {
        let mut idx = *hot_pack.tree_roots.get_unchecked(pos) as usize;
        loop {
            let node = *hot_pack.nodes.get_unchecked(idx);
            if node.is_leaf() {
                score += node.leaf;
                break;
            }
            let x = *hot_feat.get_unchecked(node.hot_fidx());
            node_evals += 1;
            idx = if x.is_nan() {
                node.missing as usize
            } else if x < node.thr {
                node.left as usize
            } else {
                node.right as usize
            };
        }
        let completed_trees = pos + 1;
        if completed_trees == checkpoints[next_checkpoint_idx] {
            if on_checkpoint(completed_trees, score, node_evals, resolved_early) {
                return Ok((score, completed_trees, node_evals, resolved_early));
            }
            next_checkpoint_idx += 1;
            if next_checkpoint_idx >= checkpoints.len() {
                return Ok((score, completed_trees, node_evals, resolved_early));
            }
        }
    }
    bail!("failed to reach final compiled hot exact checkpoint");
}

#[inline(always)]
fn maybe_exact_exit(
    score: f32,
    threshold: f32,
    suffix_min: f32,
    suffix_max: f32,
    eps: f32,
    bound_guard: f32,
) -> Option<bool> {
    let lower = (score as f64) + (suffix_min as f64);
    let upper = (score as f64) + (suffix_max as f64);
    let pass_cut = (threshold + eps + bound_guard) as f64;
    let ref_cut = (threshold - eps - bound_guard) as f64;
    if lower >= pass_cut {
        Some(true)
    } else if upper < ref_cut {
        Some(false)
    } else {
        None
    }
}

#[inline(always)]
fn maybe_approx_exit(
    score: f32,
    row_threshold: f32,
    visited: usize,
    policy: &ApproxPolicy,
    cp_idx: &mut usize,
) -> Option<(bool, RowMeta)> {
    if *cp_idx >= policy.used_checkpoints.len() || policy.used_checkpoints[*cp_idx] != visited {
        return None;
    }
    let idx = *cp_idx;
    *cp_idx += 1;
    if let Some(tau_ref) = policy.used_tau_ref[idx] {
        if (score - row_threshold) <= tau_ref {
            return Some((
                false,
                RowMeta {
                    approx_pass: false,
                    approx_refer: true,
                    approx_checkpoint_idx: idx as i32,
                },
            ));
        }
    }
    if let Some(tau_pos) = policy.used_tau_positive()[idx] {
        if (score - row_threshold) >= tau_pos {
            return Some((
                true,
                RowMeta {
                    approx_pass: true,
                    approx_refer: false,
                    approx_checkpoint_idx: idx as i32,
                },
            ));
        }
    }
    None
}

#[inline(always)]
unsafe fn traverse_approx_float_nomiss_l1_hot(
    model: &SoaModel,
    feat: &[f32],
    threshold: f32,
    base_score: f32,
    plan: &TreeOrder,
    approx_policy: &ApproxPolicy,
    eps: f32,
    bound_guard: f32,
    bound_check_every: usize,
) -> Result<(f32, bool, i32, RowMeta)> {
    let mut score = base_score;
    let mut visited = 0i32;
    let mut cp_idx = 0usize;
    let checkpoints = approx_policy.used_checkpoints.as_slice();
    let tau_ref = approx_policy.used_tau_ref.as_slice();
    let tau_pos = approx_policy.used_tau_positive();
    let mut next_cp = checkpoints.get(0).copied().unwrap_or(usize::MAX);
    let k_hot = approx_policy.k_hot.min(plan.n_trees);

    for pos in 0..k_hot {
        let tree_idx = *plan.order.get_unchecked(pos) as usize;
        let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
        loop {
            if *model.is_leaf.get_unchecked(idx) != 0 {
                score += *model.leaf.get_unchecked(idx);
                break;
            }
            let fidx = *model.fidx.get_unchecked(idx) as usize;
            let x = *feat.get_unchecked(fidx);
            idx = if x < *model.thr.get_unchecked(idx) {
                *model.left.get_unchecked(idx) as usize
            } else {
                *model.right.get_unchecked(idx) as usize
            };
        }
        let m_end = pos + 1;
        visited = m_end as i32;
        if m_end == next_cp {
            let checkpoint_slot = cp_idx;
            cp_idx += 1;
            if let Some(tau) = *tau_ref.get_unchecked(checkpoint_slot) {
                if (score - threshold) <= tau {
                    return Ok((
                        score,
                        false,
                        visited,
                        RowMeta {
                            approx_pass: false,
                            approx_refer: true,
                            approx_checkpoint_idx: checkpoint_slot as i32,
                        },
                    ));
                }
            }
            if let Some(tau) = *tau_pos.get_unchecked(checkpoint_slot) {
                if (score - threshold) >= tau {
                    return Ok((
                        score,
                        true,
                        visited,
                        RowMeta {
                            approx_pass: true,
                            approx_refer: false,
                            approx_checkpoint_idx: checkpoint_slot as i32,
                        },
                    ));
                }
            }
            next_cp = checkpoints.get(cp_idx).copied().unwrap_or(usize::MAX);
        }
    }

    if bound_check_every == 1 {
        for pos in k_hot..plan.n_trees {
            let tree_idx = *plan.order.get_unchecked(pos) as usize;
            let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
            loop {
                if *model.is_leaf.get_unchecked(idx) != 0 {
                    score += *model.leaf.get_unchecked(idx);
                    break;
                }
                let fidx = *model.fidx.get_unchecked(idx) as usize;
                let x = *feat.get_unchecked(fidx);
                idx = if x < *model.thr.get_unchecked(idx) {
                    *model.left.get_unchecked(idx) as usize
                } else {
                    *model.right.get_unchecked(idx) as usize
                };
            }
            let m_end = pos + 1;
            visited = m_end as i32;
            if let Some(pass) = maybe_exact_exit(
                score,
                threshold,
                *plan.suffix_min.get_unchecked(m_end),
                *plan.suffix_max.get_unchecked(m_end),
                eps,
                bound_guard,
            ) {
                return Ok((score, pass, visited, RowMeta::default()));
            }
        }
    } else {
        for pos in k_hot..plan.n_trees {
            let tree_idx = *plan.order.get_unchecked(pos) as usize;
            let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
            loop {
                if *model.is_leaf.get_unchecked(idx) != 0 {
                    score += *model.leaf.get_unchecked(idx);
                    break;
                }
                let fidx = *model.fidx.get_unchecked(idx) as usize;
                let x = *feat.get_unchecked(fidx);
                idx = if x < *model.thr.get_unchecked(idx) {
                    *model.left.get_unchecked(idx) as usize
                } else {
                    *model.right.get_unchecked(idx) as usize
                };
            }
            let m_end = pos + 1;
            visited = m_end as i32;
            if ((m_end % bound_check_every) == 0) || (m_end == plan.n_trees) {
                if let Some(pass) = maybe_exact_exit(
                    score,
                    threshold,
                    *plan.suffix_min.get_unchecked(m_end),
                    *plan.suffix_max.get_unchecked(m_end),
                    eps,
                    bound_guard,
                ) {
                    return Ok((score, pass, visited, RowMeta::default()));
                }
            }
        }
    }

    Ok((score, score >= threshold, visited, RowMeta::default()))
}

#[inline(always)]
unsafe fn traverse_approx_hot_float_nomiss(
    hot_pack: &PrefixPack,
    hot_buf: &mut HotFeatureBuf,
    model: &SoaModel,
    feat: &[f32],
    threshold: f32,
    base_score: f32,
    plan: &TreeOrder,
    approx_policy: &ApproxPolicy,
    eps: f32,
    bound_guard: f32,
    bound_check_every: usize,
) -> Result<(f32, bool, i32, RowMeta)> {
    hot_buf.fill_from_row(hot_pack, feat);
    let hot_feat = hot_buf.values.as_slice();
    let mut score = base_score;
    let mut visited = 0i32;
    let mut cp_idx = 0usize;
    let hot_trees = hot_pack.n_trees.min(approx_policy.k_hot.min(plan.n_trees));

    for pos in 0..hot_trees {
        let mut idx = *hot_pack.tree_roots.get_unchecked(pos) as usize;
        loop {
            if *hot_pack.is_leaf.get_unchecked(idx) != 0 {
                score += *hot_pack.leaf.get_unchecked(idx);
                break;
            }
            let fidx = *hot_pack.fidx.get_unchecked(idx) as usize;
            let x = *hot_feat.get_unchecked(fidx);
            idx = if x < *hot_pack.thr.get_unchecked(idx) {
                *hot_pack.left.get_unchecked(idx) as usize
            } else {
                *hot_pack.right.get_unchecked(idx) as usize
            };
        }
        let m_end = pos + 1;
        visited = m_end as i32;
        if let Some((pass, meta)) =
            maybe_approx_exit(score, threshold, m_end, approx_policy, &mut cp_idx)
        {
            return Ok((score, pass, visited, meta));
        }
    }

    let k_hot = approx_policy.k_hot.min(plan.n_trees);
    for pos in hot_trees..k_hot {
        let tree_idx = *plan.order.get_unchecked(pos) as usize;
        let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
        loop {
            if *model.is_leaf.get_unchecked(idx) != 0 {
                score += *model.leaf.get_unchecked(idx);
                break;
            }
            let fidx = *model.fidx.get_unchecked(idx) as usize;
            let x = *feat.get_unchecked(fidx);
            idx = if x < *model.thr.get_unchecked(idx) {
                *model.left.get_unchecked(idx) as usize
            } else {
                *model.right.get_unchecked(idx) as usize
            };
        }
        let m_end = pos + 1;
        visited = m_end as i32;
        if let Some((pass, meta)) =
            maybe_approx_exit(score, threshold, m_end, approx_policy, &mut cp_idx)
        {
            return Ok((score, pass, visited, meta));
        }
    }

    for pos in k_hot..plan.n_trees {
        let tree_idx = *plan.order.get_unchecked(pos) as usize;
        let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
        loop {
            if *model.is_leaf.get_unchecked(idx) != 0 {
                score += *model.leaf.get_unchecked(idx);
                break;
            }
            let fidx = *model.fidx.get_unchecked(idx) as usize;
            let x = *feat.get_unchecked(fidx);
            idx = if x < *model.thr.get_unchecked(idx) {
                *model.left.get_unchecked(idx) as usize
            } else {
                *model.right.get_unchecked(idx) as usize
            };
        }
        let m_end = pos + 1;
        visited = m_end as i32;
        if ((m_end % bound_check_every) == 0) || (m_end == plan.n_trees) {
            if let Some(pass) = maybe_exact_exit(
                score,
                threshold,
                *plan.suffix_min.get_unchecked(m_end),
                *plan.suffix_max.get_unchecked(m_end),
                eps,
                bound_guard,
            ) {
                return Ok((score, pass, visited, RowMeta::default()));
            }
        }
    }

    Ok((score, score >= threshold, visited, RowMeta::default()))
}

#[inline(always)]
unsafe fn traverse_approx_rank_nomiss(
    model: &SoaModel,
    pack: &RankPack,
    cache: &mut RankCache,
    feat: &[f32],
    threshold: f32,
    base_score: f32,
    plan: &TreeOrder,
    approx_policy: &ApproxPolicy,
    eps: f32,
    bound_guard: f32,
    bound_check_every: usize,
) -> Result<(f32, bool, i32, RowMeta)> {
    let mut score = base_score;
    let mut visited = 0i32;
    let mut cp_idx = 0usize;
    let k_hot = approx_policy.k_hot.min(plan.n_trees);

    for pos in 0..k_hot {
        let tree_idx = *plan.order.get_unchecked(pos) as usize;
        let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
        loop {
            if *model.is_leaf.get_unchecked(idx) != 0 {
                score += *model.leaf.get_unchecked(idx);
                break;
            }
            let fidx = *model.fidx.get_unchecked(idx) as usize;
            let thr_rank = *pack.node_thr_rank.get_unchecked(idx);
            if thr_rank < 0 {
                bail!("invalid rank pack: split node missing threshold rank");
            }
            let rank = cached_rank(pack, cache, feat, fidx);
            idx = if rank <= thr_rank as u32 {
                *model.left.get_unchecked(idx) as usize
            } else {
                *model.right.get_unchecked(idx) as usize
            };
        }
        let m_end = pos + 1;
        visited = m_end as i32;
        if let Some((pass, meta)) =
            maybe_approx_exit(score, threshold, m_end, approx_policy, &mut cp_idx)
        {
            return Ok((score, pass, visited, meta));
        }
    }

    for pos in k_hot..plan.n_trees {
        let tree_idx = *plan.order.get_unchecked(pos) as usize;
        let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
        loop {
            if *model.is_leaf.get_unchecked(idx) != 0 {
                score += *model.leaf.get_unchecked(idx);
                break;
            }
            let fidx = *model.fidx.get_unchecked(idx) as usize;
            let thr_rank = *pack.node_thr_rank.get_unchecked(idx);
            if thr_rank < 0 {
                bail!("invalid rank pack: split node missing threshold rank");
            }
            let rank = cached_rank(pack, cache, feat, fidx);
            idx = if rank <= thr_rank as u32 {
                *model.left.get_unchecked(idx) as usize
            } else {
                *model.right.get_unchecked(idx) as usize
            };
        }
        let m_end = pos + 1;
        visited = m_end as i32;
        if ((m_end % bound_check_every) == 0) || (m_end == plan.n_trees) {
            if let Some(pass) = maybe_exact_exit(
                score,
                threshold,
                *plan.suffix_min.get_unchecked(m_end),
                *plan.suffix_max.get_unchecked(m_end),
                eps,
                bound_guard,
            ) {
                return Ok((score, pass, visited, RowMeta::default()));
            }
        }
    }

    Ok((score, score >= threshold, visited, RowMeta::default()))
}

#[inline(always)]
unsafe fn traverse_approx_float_nomiss(
    model: &SoaModel,
    feat: &[f32],
    threshold: f32,
    base_score: f32,
    plan: &TreeOrder,
    approx_policy: &ApproxPolicy,
    eps: f32,
    bound_guard: f32,
    bound_check_every: usize,
) -> Result<(f32, bool, i32, RowMeta)> {
    let mut score = base_score;
    let mut visited = 0i32;
    let mut cp_idx = 0usize;
    let k_hot = approx_policy.k_hot.min(plan.n_trees);

    for pos in 0..k_hot {
        let tree_idx = *plan.order.get_unchecked(pos) as usize;
        let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
        loop {
            if *model.is_leaf.get_unchecked(idx) != 0 {
                score += *model.leaf.get_unchecked(idx);
                break;
            }
            let fidx = *model.fidx.get_unchecked(idx) as usize;
            let x = *feat.get_unchecked(fidx);
            idx = if x < *model.thr.get_unchecked(idx) {
                *model.left.get_unchecked(idx) as usize
            } else {
                *model.right.get_unchecked(idx) as usize
            };
        }
        let m_end = pos + 1;
        visited = m_end as i32;
        if let Some((pass, meta)) =
            maybe_approx_exit(score, threshold, m_end, approx_policy, &mut cp_idx)
        {
            return Ok((score, pass, visited, meta));
        }
    }

    for pos in k_hot..plan.n_trees {
        let tree_idx = *plan.order.get_unchecked(pos) as usize;
        let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
        loop {
            if *model.is_leaf.get_unchecked(idx) != 0 {
                score += *model.leaf.get_unchecked(idx);
                break;
            }
            let fidx = *model.fidx.get_unchecked(idx) as usize;
            let x = *feat.get_unchecked(fidx);
            idx = if x < *model.thr.get_unchecked(idx) {
                *model.left.get_unchecked(idx) as usize
            } else {
                *model.right.get_unchecked(idx) as usize
            };
        }
        let m_end = pos + 1;
        visited = m_end as i32;
        if ((m_end % bound_check_every) == 0) || (m_end == plan.n_trees) {
            if let Some(pass) = maybe_exact_exit(
                score,
                threshold,
                *plan.suffix_min.get_unchecked(m_end),
                *plan.suffix_max.get_unchecked(m_end),
                eps,
                bound_guard,
            ) {
                return Ok((score, pass, visited, RowMeta::default()));
            }
        }
    }

    Ok((score, score >= threshold, visited, RowMeta::default()))
}

fn traverse_approx_generic(
    model: &SoaModel,
    rank_pack: Option<&RankPack>,
    mut cache: Option<&mut RankCache>,
    feat: &[f32],
    threshold: f32,
    base_score: f32,
    plan: &TreeOrder,
    approx_policy: &ApproxPolicy,
    eps: f32,
    bound_guard: f32,
    bound_check_every: usize,
) -> Result<(f32, bool, i32, RowMeta)> {
    let mut score = base_score;
    let mut visited = 0i32;
    let mut cp_idx = 0usize;
    let k_hot = approx_policy.k_hot.min(plan.n_trees);

    for pos in 0..k_hot {
        let tree_idx = plan.order[pos] as usize;
        let mut idx = model.tree_roots[tree_idx] as usize;
        loop {
            if model.is_leaf[idx] != 0 {
                score += model.leaf[idx];
                break;
            }
            let fidx = model.fidx[idx] as usize;
            let x = feat[fidx];
            let next = if x.is_nan() {
                model.missing[idx]
            } else if let (Some(pack), Some(cache_ref)) = (rank_pack, cache.as_deref_mut()) {
                let thr_rank = pack.node_thr_rank[idx];
                if thr_rank < 0 {
                    bail!("invalid rank pack: split node missing threshold rank");
                }
                let rank = unsafe { cached_rank(pack, cache_ref, feat, fidx) };
                if rank <= thr_rank as u32 {
                    model.left[idx]
                } else {
                    model.right[idx]
                }
            } else if x < model.thr[idx] {
                model.left[idx]
            } else {
                model.right[idx]
            };
            idx = next as usize;
        }
        let m_end = pos + 1;
        visited = m_end as i32;
        if let Some((pass, meta)) =
            maybe_approx_exit(score, threshold, m_end, approx_policy, &mut cp_idx)
        {
            return Ok((score, pass, visited, meta));
        }
    }

    for pos in k_hot..plan.n_trees {
        let tree_idx = plan.order[pos] as usize;
        let mut idx = model.tree_roots[tree_idx] as usize;
        loop {
            if model.is_leaf[idx] != 0 {
                score += model.leaf[idx];
                break;
            }
            let fidx = model.fidx[idx] as usize;
            let x = feat[fidx];
            let next = if x.is_nan() {
                model.missing[idx]
            } else if let (Some(pack), Some(cache_ref)) = (rank_pack, cache.as_deref_mut()) {
                let thr_rank = pack.node_thr_rank[idx];
                if thr_rank < 0 {
                    bail!("invalid rank pack: split node missing threshold rank");
                }
                let rank = unsafe { cached_rank(pack, cache_ref, feat, fidx) };
                if rank <= thr_rank as u32 {
                    model.left[idx]
                } else {
                    model.right[idx]
                }
            } else if x < model.thr[idx] {
                model.left[idx]
            } else {
                model.right[idx]
            };
            idx = next as usize;
        }
        let m_end = pos + 1;
        visited = m_end as i32;
        if ((m_end % bound_check_every) == 0) || (m_end == plan.n_trees) {
            if let Some(pass) = maybe_exact_exit(
                score,
                threshold,
                plan.suffix_min[m_end],
                plan.suffix_max[m_end],
                eps,
                bound_guard,
            ) {
                return Ok((score, pass, visited, RowMeta::default()));
            }
        }
    }

    Ok((score, score >= threshold, visited, RowMeta::default()))
}

#[inline(always)]
unsafe fn traverse_rank_nomiss(
    model: &SoaModel,
    pack: &RankPack,
    cache: &mut RankCache,
    feat: &[f32],
    threshold: f32,
    base_score: f32,
    mode: InferMode,
    plan: &TreeOrder,
    approx_policy: Option<&ApproxPolicy>,
    eps: f32,
    bound_guard: f32,
    bound_check_every: usize,
) -> Result<(f32, bool, i32, RowMeta)> {
    let mut score = base_score;
    let mut visited = 0i32;
    let mut cp_idx = 0usize;

    for pos in 0..plan.n_trees {
        let tree_idx = *plan.order.get_unchecked(pos) as usize;
        let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
        loop {
            if *model.is_leaf.get_unchecked(idx) != 0 {
                score += *model.leaf.get_unchecked(idx);
                break;
            }
            let fidx = *model.fidx.get_unchecked(idx) as usize;
            let thr_rank = *pack.node_thr_rank.get_unchecked(idx);
            if thr_rank < 0 {
                bail!("invalid rank pack: split node missing threshold rank");
            }
            let rank = cached_rank(pack, cache, feat, fidx);
            idx = if rank <= thr_rank as u32 {
                *model.left.get_unchecked(idx) as usize
            } else {
                *model.right.get_unchecked(idx) as usize
            };
        }

        let m_end = pos + 1;
        visited = m_end as i32;
        if is_route_mode(mode) && (((m_end % bound_check_every) == 0) || (m_end == plan.n_trees)) {
            if let Some(pass) = maybe_exact_exit(
                score,
                threshold,
                *plan.suffix_min.get_unchecked(m_end),
                *plan.suffix_max.get_unchecked(m_end),
                eps,
                bound_guard,
            ) {
                return Ok((score, pass, visited, RowMeta::default()));
            }
        }

        if matches!(mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
            if let Some(policy) = approx_policy {
                if let Some((pass, meta)) =
                    maybe_approx_exit(score, threshold, m_end, policy, &mut cp_idx)
                {
                    return Ok((score, pass, visited, meta));
                }
            }
        }
    }

    Ok((score, score >= threshold, visited, RowMeta::default()))
}

#[inline(always)]
unsafe fn traverse_float_nomiss_from(
    model: &SoaModel,
    feat: &[f32],
    threshold: f32,
    init_score: f32,
    mode: InferMode,
    plan: &TreeOrder,
    approx_policy: Option<&ApproxPolicy>,
    eps: f32,
    bound_guard: f32,
    bound_check_every: usize,
    start_tree_idx: usize,
) -> Result<(f32, bool, i32, RowMeta)> {
    let mut score = init_score;
    let mut visited = start_tree_idx as i32;
    let mut cp_idx = 0usize;

    if start_tree_idx >= plan.n_trees {
        return Ok((score, score >= threshold, visited, RowMeta::default()));
    }

    for pos in start_tree_idx..plan.n_trees {
        let tree_idx = *plan.order.get_unchecked(pos) as usize;
        let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
        loop {
            if *model.is_leaf.get_unchecked(idx) != 0 {
                score += *model.leaf.get_unchecked(idx);
                break;
            }
            let fidx = *model.fidx.get_unchecked(idx) as usize;
            let x = *feat.get_unchecked(fidx);
            idx = if x < *model.thr.get_unchecked(idx) {
                *model.left.get_unchecked(idx) as usize
            } else {
                *model.right.get_unchecked(idx) as usize
            };
        }

        let m_end = pos + 1;
        visited = m_end as i32;
        if is_route_mode(mode) && (((m_end % bound_check_every) == 0) || (m_end == plan.n_trees)) {
            if let Some(pass) = maybe_exact_exit(
                score,
                threshold,
                *plan.suffix_min.get_unchecked(m_end),
                *plan.suffix_max.get_unchecked(m_end),
                eps,
                bound_guard,
            ) {
                return Ok((score, pass, visited, RowMeta::default()));
            }
        }

        if matches!(mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
            if let Some(policy) = approx_policy {
                if let Some((pass, meta)) =
                    maybe_approx_exit(score, threshold, m_end, policy, &mut cp_idx)
                {
                    return Ok((score, pass, visited, meta));
                }
            }
        }
    }

    Ok((score, score >= threshold, visited, RowMeta::default()))
}

#[inline(always)]
unsafe fn traverse_float_nomiss(
    model: &SoaModel,
    feat: &[f32],
    threshold: f32,
    base_score: f32,
    mode: InferMode,
    plan: &TreeOrder,
    approx_policy: Option<&ApproxPolicy>,
    eps: f32,
    bound_guard: f32,
    bound_check_every: usize,
) -> Result<(f32, bool, i32, RowMeta)> {
    traverse_float_nomiss_from(
        model,
        feat,
        threshold,
        base_score,
        mode,
        plan,
        approx_policy,
        eps,
        bound_guard,
        bound_check_every,
        0,
    )
}

#[inline(always)]
fn traverse_generic_from(
    model: &SoaModel,
    rank_pack: Option<&RankPack>,
    mut cache: Option<&mut RankCache>,
    feat: &[f32],
    threshold: f32,
    init_score: f32,
    mode: InferMode,
    plan: &TreeOrder,
    approx_policy: Option<&ApproxPolicy>,
    eps: f32,
    bound_guard: f32,
    bound_check_every: usize,
    start_tree_idx: usize,
) -> Result<(f32, bool, i32, RowMeta)> {
    let mut score = init_score;
    let mut visited = start_tree_idx as i32;
    let mut cp_idx = 0usize;
    if start_tree_idx >= plan.n_trees {
        return Ok((score, score >= threshold, visited, RowMeta::default()));
    }
    for pos in start_tree_idx..plan.n_trees {
        let tree_idx = plan.order[pos] as usize;
        let mut idx = model.tree_roots[tree_idx] as usize;
        loop {
            if model.is_leaf[idx] != 0 {
                score += model.leaf[idx];
                break;
            }
            let fidx = model.fidx[idx] as usize;
            let x = feat[fidx];
            let next = if x.is_nan() {
                model.missing[idx]
            } else if let (Some(pack), Some(cache_ref)) = (rank_pack, cache.as_deref_mut()) {
                let thr_rank = pack.node_thr_rank[idx];
                if thr_rank < 0 {
                    bail!("invalid rank pack: split node missing threshold rank");
                }
                let rank = unsafe { cached_rank(pack, cache_ref, feat, fidx) };
                if rank <= thr_rank as u32 {
                    model.left[idx]
                } else {
                    model.right[idx]
                }
            } else if x < model.thr[idx] {
                model.left[idx]
            } else {
                model.right[idx]
            };
            idx = next as usize;
        }

        let m_end = pos + 1;
        visited = m_end as i32;
        if is_route_mode(mode) && (((m_end % bound_check_every) == 0) || (m_end == plan.n_trees)) {
            if let Some(pass) = maybe_exact_exit(
                score,
                threshold,
                plan.suffix_min[m_end],
                plan.suffix_max[m_end],
                eps,
                bound_guard,
            ) {
                return Ok((score, pass, visited, RowMeta::default()));
            }
        }

        if matches!(mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
            if let Some(policy) = approx_policy {
                if let Some((pass, meta)) =
                    maybe_approx_exit(score, threshold, m_end, policy, &mut cp_idx)
                {
                    return Ok((score, pass, visited, meta));
                }
            }
        }
    }
    Ok((score, score >= threshold, visited, RowMeta::default()))
}

#[inline(always)]
fn traverse_generic(
    model: &SoaModel,
    rank_pack: Option<&RankPack>,
    cache: Option<&mut RankCache>,
    feat: &[f32],
    threshold: f32,
    base_score: f32,
    mode: InferMode,
    plan: &TreeOrder,
    approx_policy: Option<&ApproxPolicy>,
    eps: f32,
    bound_guard: f32,
    bound_check_every: usize,
) -> Result<(f32, bool, i32, RowMeta)> {
    traverse_generic_from(
        model,
        rank_pack,
        cache,
        feat,
        threshold,
        base_score,
        mode,
        plan,
        approx_policy,
        eps,
        bound_guard,
        bound_check_every,
        0,
    )
}

fn run_kernel(
    model: &SoaModel,
    plan: &TreeOrder,
    batch: &FeatureBatch,
    route_meta: Option<&RouteMeta>,
    rank_pack: Option<&RankPack>,
    approx_policy: Option<&ApproxPolicy>,
    prefix16_pack: Option<&PrefixPack>,
    prefix32_pack: Option<&PrefixPack>,
    n: usize,
    threshold: f32,
    base_score: f32,
    mode: InferMode,
    eps: f32,
    bound_guard: f32,
    bound_check_every: usize,
    threads: usize,
    chunk_rows: usize,
    parallel_min_rows: usize,
    thread_pool: Option<&rayon::ThreadPool>,
) -> Result<(
    Vec<f32>,
    Vec<bool>,
    Vec<i32>,
    bool,
    usize,
    usize,
    usize,
    Vec<u64>,
)> {
    let mut scores = vec![0.0f32; n];
    let mut passes = vec![false; n];
    let mut visits = vec![0i32; n];
    let par_threads = active_threads(threads);
    let parallel_enabled = par_threads > 1 && n >= parallel_min_rows;
    let checkpoint_len = approx_policy
        .map(|p| p.used_checkpoints.len())
        .unwrap_or(0usize);
    let hot_prefix_pack = if matches!(mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
        approx_policy.and_then(|p| {
            let first_cp = p.used_checkpoints.first().copied().unwrap_or(usize::MAX);
            match p.rank_mode {
                RankMode::Float if first_cp <= 16 => prefix16_pack,
                RankMode::Float if first_cp <= 32 => prefix32_pack,
                _ => None,
            }
        })
    } else {
        None
    };
    let use_l1_route_approx_hot = matches!(mode, InferMode::RouteApprox)
        && batch.nan_free
        && route_meta.is_none()
        && rank_pack.is_none()
        && hot_prefix_pack.is_none()
        && approx_policy.is_some();

    let mut work = || -> Result<(usize, usize, Vec<u64>)> {
        if parallel_enabled {
            let counts = scores
                .par_chunks_mut(chunk_rows)
                .zip(passes.par_chunks_mut(chunk_rows))
                .zip(visits.par_chunks_mut(chunk_rows))
                .enumerate()
                .map(|(chunk_idx, ((score_chunk, pass_chunk), visit_chunk))| -> Result<(usize, usize, Vec<u64>)> {
                    let start = chunk_idx * chunk_rows;
                    let mut rank_cache = if let Some(pack) = rank_pack {
                        Some(RankCache::new(pack.n_features))
                    } else {
                        None
                    };
                    let mut hot_buf = hot_prefix_pack.map(|p| HotFeatureBuf::new(p.n_hot_features));
                    let mut approx_pass_cnt = 0usize;
                    let mut approx_ref_cnt = 0usize;
                    let mut checkpoint_counts = vec![0u64; checkpoint_len];
                    for local_idx in 0..score_chunk.len() {
                        let row_idx = start + local_idx;
                        let row_active = route_meta
                            .map(|m| m.active[row_idx] != 0)
                            .unwrap_or(true);
                        if !row_active {
                            score_chunk[local_idx] = f32::NAN;
                            pass_chunk[local_idx] = false;
                            visit_chunk[local_idx] = 0;
                            continue;
                        }
                        let row_threshold = route_meta
                            .map(|m| m.tau_used[row_idx])
                            .unwrap_or(threshold);
                        let feat = batch.row(row_idx);
                        let (score, pass, visit, meta) = if matches!(mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
                            if use_l1_route_approx_hot {
                                unsafe {
                                    traverse_approx_float_nomiss_l1_hot(
                                        model,
                                        feat,
                                        row_threshold,
                                        base_score,
                                        plan,
                                        approx_policy.unwrap(),
                                        eps,
                                        bound_guard,
                                        bound_check_every,
                                    )
                                }?
                            } else {
                                match (batch.nan_free, rank_pack, approx_policy, hot_prefix_pack) {
                                (true, None, Some(policy), Some(pack)) => unsafe {
                                    traverse_approx_hot_float_nomiss(
                                        pack,
                                        hot_buf.as_mut().unwrap(),
                                        model,
                                        feat,
                                        row_threshold,
                                        base_score,
                                        plan,
                                        policy,
                                        eps,
                                        bound_guard,
                                        bound_check_every,
                                    )
                                }?,
                                (true, Some(pack), Some(policy), _) => {
                                    let cache = rank_cache.as_mut().unwrap();
                                    cache.next_row();
                                    unsafe {
                                        traverse_approx_rank_nomiss(
                                        model,
                                        pack,
                                        cache,
                                        feat,
                                        row_threshold,
                                        base_score,
                                        plan,
                                        policy,
                                            eps,
                                            bound_guard,
                                            bound_check_every,
                                        )
                                    }?
                                }
                                (true, None, Some(policy), None) => unsafe {
                                    traverse_approx_float_nomiss(
                                        model,
                                        feat,
                                        row_threshold,
                                        base_score,
                                        plan,
                                        policy,
                                        eps,
                                        bound_guard,
                                        bound_check_every,
                                    )
                                }?,
                                (_, _, Some(policy), _) => {
                                    if let Some(cache) = rank_cache.as_mut() {
                                        cache.next_row();
                                    }
                                    traverse_approx_generic(
                                        model,
                                        rank_pack,
                                        rank_cache.as_mut(),
                                        feat,
                                        row_threshold,
                                        base_score,
                                        plan,
                                        policy,
                                        eps,
                                        bound_guard,
                                        bound_check_every,
                                    )?
                                }
                                (_, _, None, _) => bail!("route-approx missing approx policy"),
                                }
                            }
                        } else {
                            match (batch.nan_free, rank_pack) {
                                (true, Some(pack)) => {
                                    let cache = rank_cache.as_mut().unwrap();
                                    cache.next_row();
                                    unsafe {
                                    traverse_rank_nomiss(
                                        model,
                                        pack,
                                        cache,
                                        feat,
                                        row_threshold,
                                        base_score,
                                        mode,
                                            plan,
                                            approx_policy,
                                            eps,
                                            bound_guard,
                                            bound_check_every,
                                        )
                                    }?
                                }
                                (true, None) => unsafe {
                                    traverse_float_nomiss(
                                        model,
                                        feat,
                                        row_threshold,
                                        base_score,
                                        mode,
                                        plan,
                                        approx_policy,
                                        eps,
                                        bound_guard,
                                        bound_check_every,
                                    )
                                }?,
                                (_, _) => {
                                    if let Some(cache) = rank_cache.as_mut() {
                                        cache.next_row();
                                    }
                                    traverse_generic(
                                        model,
                                        rank_pack,
                                        rank_cache.as_mut(),
                                        feat,
                                        row_threshold,
                                        base_score,
                                        mode,
                                        plan,
                                        approx_policy,
                                        eps,
                                        bound_guard,
                                        bound_check_every,
                                    )?
                                }
                            }
                        };
                        if meta.approx_pass {
                            approx_pass_cnt += 1;
                        }
                        if meta.approx_refer {
                            approx_ref_cnt += 1;
                        }
                        if meta.approx_checkpoint_idx >= 0 {
                            checkpoint_counts[meta.approx_checkpoint_idx as usize] += 1;
                        }
                        score_chunk[local_idx] = score;
                        pass_chunk[local_idx] = pass;
                        visit_chunk[local_idx] = visit;
                    }
                    Ok((approx_pass_cnt, approx_ref_cnt, checkpoint_counts))
                })
                .try_reduce(
                    || (0usize, 0usize, vec![0u64; checkpoint_len]),
                    |a, b| {
                        let mut merged = a.2;
                        for (dst, src) in merged.iter_mut().zip(b.2.iter()) {
                            *dst += *src;
                        }
                        Ok((a.0 + b.0, a.1 + b.1, merged))
                    },
                )?;
            Ok(counts)
        } else {
            let mut rank_cache = if let Some(pack) = rank_pack {
                Some(RankCache::new(pack.n_features))
            } else {
                None
            };
            let mut hot_buf = hot_prefix_pack.map(|p| HotFeatureBuf::new(p.n_hot_features));
            let mut approx_pass_cnt = 0usize;
            let mut approx_ref_cnt = 0usize;
            let mut checkpoint_counts = vec![0u64; checkpoint_len];
            for row_idx in 0..n {
                let row_active = route_meta.map(|m| m.active[row_idx] != 0).unwrap_or(true);
                if !row_active {
                    scores[row_idx] = f32::NAN;
                    passes[row_idx] = false;
                    visits[row_idx] = 0;
                    continue;
                }
                let row_threshold = route_meta.map(|m| m.tau_used[row_idx]).unwrap_or(threshold);
                let feat = batch.row(row_idx);
                let (score, pass, visit, meta) = if matches!(mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
                    if use_l1_route_approx_hot {
                        unsafe {
                            traverse_approx_float_nomiss_l1_hot(
                                model,
                                feat,
                                row_threshold,
                                base_score,
                                plan,
                                approx_policy.unwrap(),
                                eps,
                                bound_guard,
                                bound_check_every,
                            )
                        }?
                    } else {
                        match (batch.nan_free, rank_pack, approx_policy, hot_prefix_pack) {
                        (true, None, Some(policy), Some(pack)) => unsafe {
                            traverse_approx_hot_float_nomiss(
                                pack,
                                hot_buf.as_mut().unwrap(),
                                model,
                                feat,
                                row_threshold,
                                base_score,
                                plan,
                                policy,
                                eps,
                                bound_guard,
                                bound_check_every,
                            )
                        }?,
                        (true, Some(pack), Some(policy), _) => {
                            let cache = rank_cache.as_mut().unwrap();
                            cache.next_row();
                            unsafe {
                                traverse_approx_rank_nomiss(
                                model,
                                pack,
                                cache,
                                feat,
                                row_threshold,
                                base_score,
                                plan,
                                policy,
                                    eps,
                                    bound_guard,
                                    bound_check_every,
                                )
                            }?
                        }
                        (true, None, Some(policy), None) => unsafe {
                            traverse_approx_float_nomiss(
                                model,
                                feat,
                                row_threshold,
                                base_score,
                                plan,
                                policy,
                                eps,
                                bound_guard,
                                bound_check_every,
                            )
                        }?,
                        (_, _, Some(policy), _) => {
                            if let Some(cache) = rank_cache.as_mut() {
                                cache.next_row();
                            }
                            traverse_approx_generic(
                                model,
                                rank_pack,
                                rank_cache.as_mut(),
                                feat,
                                row_threshold,
                                base_score,
                                plan,
                                policy,
                                eps,
                                bound_guard,
                                bound_check_every,
                            )?
                        }
                        (_, _, None, _) => bail!("route-approx missing approx policy"),
                        }
                    }
                } else {
                    match (batch.nan_free, rank_pack) {
                        (true, Some(pack)) => {
                            let cache = rank_cache.as_mut().unwrap();
                            cache.next_row();
                            unsafe {
                                traverse_rank_nomiss(
                                    model,
                                    pack,
                                    cache,
                                    feat,
                                    row_threshold,
                                    base_score,
                                    mode,
                                    plan,
                                    approx_policy,
                                    eps,
                                    bound_guard,
                                    bound_check_every,
                                )
                            }?
                        }
                        (true, None) => unsafe {
                            traverse_float_nomiss(
                                model,
                                feat,
                                row_threshold,
                                base_score,
                                mode,
                                plan,
                                approx_policy,
                                eps,
                                bound_guard,
                                bound_check_every,
                            )
                        }?,
                        (_, _) => {
                            if let Some(cache) = rank_cache.as_mut() {
                                cache.next_row();
                            }
                            traverse_generic(
                                model,
                                rank_pack,
                                rank_cache.as_mut(),
                                feat,
                                row_threshold,
                                base_score,
                                mode,
                                plan,
                                approx_policy,
                                eps,
                                bound_guard,
                                bound_check_every,
                            )?
                        }
                    }
                };
                if meta.approx_pass {
                    approx_pass_cnt += 1;
                }
                if meta.approx_refer {
                    approx_ref_cnt += 1;
                }
                if meta.approx_checkpoint_idx >= 0 {
                    checkpoint_counts[meta.approx_checkpoint_idx as usize] += 1;
                }
                scores[row_idx] = score;
                passes[row_idx] = pass;
                visits[row_idx] = visit;
            }
            Ok((approx_pass_cnt, approx_ref_cnt, checkpoint_counts))
        }
    };
    let (approx_pass_cnt, approx_ref_cnt, checkpoint_counts) =
        install_in_pool(thread_pool, work)?;

    Ok((
        scores,
        passes,
        visits,
        parallel_enabled,
        par_threads,
        approx_pass_cnt,
        approx_ref_cnt,
        checkpoint_counts,
    ))
}

fn pct(v: &[i32], q: f64) -> i32 {
    if v.is_empty() {
        return 0;
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    let idx = ((s.len() as f64 - 1.0) * q).round() as usize;
    s[idx.min(s.len() - 1)]
}

fn histogram_from_visits(visits: &[i32], max_trees: usize) -> Vec<u64> {
    let mut hist = vec![0u64; max_trees + 1];
    for &v in visits {
        let idx = if v <= 0 {
            0usize
        } else {
            (v as usize).min(max_trees)
        };
        hist[idx] += 1;
    }
    hist
}

fn pct_from_hist(hist: &[u64], q: f64, n: usize) -> i32 {
    if n == 0 {
        return 0;
    }
    let target = ((n as f64 - 1.0) * q).round() as u64;
    let mut acc = 0u64;
    for (i, c) in hist.iter().enumerate() {
        acc += *c;
        if acc > target {
            return i as i32;
        }
    }
    (hist.len().saturating_sub(1)) as i32
}

fn run_kernel_fast_approx(
    model: &SoaModel,
    plan: &TreeOrder,
    batch: &FeatureBatch,
    route_meta: Option<&RouteMeta>,
    approx_policy: &ApproxPolicy,
    prefix16_pack: Option<&PrefixPack>,
    prefix32_pack: Option<&PrefixPack>,
    n: usize,
    threshold: f32,
    base_score: f32,
    eps: f32,
    bound_guard: f32,
    bound_check_every: usize,
    threads: usize,
    chunk_rows: usize,
    parallel_min_rows: usize,
    thread_pool: Option<&rayon::ThreadPool>,
) -> Result<(FastAgg, bool, usize)> {
    let par_threads = active_threads(threads);
    let parallel_enabled = par_threads > 1 && n >= parallel_min_rows;
    let hot_prefix_pack = {
        let first_cp = approx_policy
            .used_checkpoints
            .first()
            .copied()
            .unwrap_or(usize::MAX);
        match approx_policy.rank_mode {
            RankMode::Float if first_cp <= 16 => prefix16_pack,
            RankMode::Float if first_cp <= 32 => prefix32_pack,
            _ => None,
        }
    };
    let hist_len = plan.n_trees + 1;
    let checkpoint_len = approx_policy.used_checkpoints.len();
    let use_l1_route_approx_hot = batch.nan_free && route_meta.is_none() && hot_prefix_pack.is_none();

    let work = || -> Result<FastAgg> {
        if parallel_enabled {
            let chunk_starts: Vec<usize> = (0..n).step_by(chunk_rows).collect();
            chunk_starts
                .into_par_iter()
                .map(|start| -> Result<FastAgg> {
                    let mut agg = FastAgg {
                        checkpoint_counts: vec![0u64; checkpoint_len],
                        visit_hist: vec![0u64; hist_len],
                        ..Default::default()
                    };
                    let mut hot_buf = hot_prefix_pack.map(|p| HotFeatureBuf::new(p.n_hot_features));
                    let end = (start + chunk_rows).min(n);
                    for row_idx in start..end {
                        let row_active = route_meta
                            .map(|m| m.active[row_idx] != 0)
                            .unwrap_or(true);
                        if !row_active {
                            agg.visit_hist[0] += 1;
                            continue;
                        }
                        agg.active_cnt += 1;
                        let row_threshold = route_meta
                            .map(|m| m.tau_used[row_idx])
                            .unwrap_or(threshold);
                        let feat = batch.row(row_idx);
                        let (score, pass, visit, meta) = if use_l1_route_approx_hot {
                            unsafe {
                                traverse_approx_float_nomiss_l1_hot(
                                    model,
                                    feat,
                                    row_threshold,
                                    base_score,
                                    plan,
                                    approx_policy,
                                    eps,
                                    bound_guard,
                                    bound_check_every,
                                )
                            }?
                        } else {
                            match (batch.nan_free, hot_prefix_pack) {
                                (true, Some(pack)) => unsafe {
                                    traverse_approx_hot_float_nomiss(
                                        pack,
                                        hot_buf.as_mut().unwrap(),
                                        model,
                                        feat,
                                        row_threshold,
                                        base_score,
                                        plan,
                                        approx_policy,
                                        eps,
                                        bound_guard,
                                        bound_check_every,
                                    )
                                }?,
                                (true, None) => unsafe {
                                    traverse_approx_float_nomiss(
                                        model,
                                        feat,
                                        row_threshold,
                                        base_score,
                                        plan,
                                        approx_policy,
                                        eps,
                                        bound_guard,
                                        bound_check_every,
                                    )
                                }?,
                                (_, _) => traverse_approx_generic(
                                    model,
                                    None,
                                    None,
                                    feat,
                                    row_threshold,
                                    base_score,
                                    plan,
                                    approx_policy,
                                    eps,
                                    bound_guard,
                                    bound_check_every,
                                )?,
                            }
                        };
                        if pass {
                            agg.pass_cnt += 1;
                        }
                        let visit_usize = visit.max(0) as usize;
                        agg.visit_sum += visit_usize as u64;
                        if visit_usize == plan.n_trees {
                            agg.full_cnt += 1;
                        }
                        if meta.approx_pass {
                            agg.approx_pass_cnt += 1;
                        }
                        if meta.approx_refer {
                            agg.approx_ref_cnt += 1;
                        }
                        if meta.approx_checkpoint_idx >= 0 {
                            agg.checkpoint_counts[meta.approx_checkpoint_idx as usize] += 1;
                        }
                        if visit_usize < agg.visit_hist.len() {
                            agg.visit_hist[visit_usize] += 1;
                        }
                        let _ = score;
                    }
                    Ok(agg)
                })
                .try_reduce(
                    || FastAgg {
                        checkpoint_counts: vec![0u64; checkpoint_len],
                        visit_hist: vec![0u64; hist_len],
                        ..Default::default()
                    },
                    |mut a, b| {
                        a.pass_cnt += b.pass_cnt;
                        a.active_cnt += b.active_cnt;
                        a.visit_sum += b.visit_sum;
                        a.full_cnt += b.full_cnt;
                        a.approx_pass_cnt += b.approx_pass_cnt;
                        a.approx_ref_cnt += b.approx_ref_cnt;
                        for (dst, src) in a.checkpoint_counts.iter_mut().zip(b.checkpoint_counts.iter()) {
                            *dst += *src;
                        }
                        for (dst, src) in a.visit_hist.iter_mut().zip(b.visit_hist.iter()) {
                            *dst += *src;
                        }
                        Ok(a)
                    },
                )
        } else {
            let mut agg = FastAgg {
                checkpoint_counts: vec![0u64; checkpoint_len],
                visit_hist: vec![0u64; hist_len],
                ..Default::default()
            };
            let mut hot_buf = hot_prefix_pack.map(|p| HotFeatureBuf::new(p.n_hot_features));
            for row_idx in 0..n {
                let row_active = route_meta.map(|m| m.active[row_idx] != 0).unwrap_or(true);
                if !row_active {
                    agg.visit_hist[0] += 1;
                    continue;
                }
                agg.active_cnt += 1;
                let row_threshold = route_meta.map(|m| m.tau_used[row_idx]).unwrap_or(threshold);
                let feat = batch.row(row_idx);
                let (score, pass, visit, meta) = if use_l1_route_approx_hot {
                    unsafe {
                        traverse_approx_float_nomiss_l1_hot(
                            model,
                            feat,
                            row_threshold,
                            base_score,
                            plan,
                            approx_policy,
                            eps,
                            bound_guard,
                            bound_check_every,
                        )
                    }?
                } else {
                    match (batch.nan_free, hot_prefix_pack) {
                        (true, Some(pack)) => unsafe {
                            traverse_approx_hot_float_nomiss(
                                pack,
                                hot_buf.as_mut().unwrap(),
                                model,
                                feat,
                                row_threshold,
                                base_score,
                                plan,
                                approx_policy,
                                eps,
                                bound_guard,
                                bound_check_every,
                            )
                        }?,
                        (true, None) => unsafe {
                            traverse_approx_float_nomiss(
                                model,
                                feat,
                                row_threshold,
                                base_score,
                                plan,
                                approx_policy,
                                eps,
                                bound_guard,
                                bound_check_every,
                            )
                        }?,
                        (_, _) => traverse_approx_generic(
                            model,
                            None,
                            None,
                            feat,
                            row_threshold,
                            base_score,
                            plan,
                            approx_policy,
                            eps,
                            bound_guard,
                            bound_check_every,
                        )?,
                    }
                };
                if pass {
                    agg.pass_cnt += 1;
                }
                let visit_usize = visit.max(0) as usize;
                agg.visit_sum += visit_usize as u64;
                if visit_usize == plan.n_trees {
                    agg.full_cnt += 1;
                }
                if meta.approx_pass {
                    agg.approx_pass_cnt += 1;
                }
                if meta.approx_refer {
                    agg.approx_ref_cnt += 1;
                }
                if meta.approx_checkpoint_idx >= 0 {
                    agg.checkpoint_counts[meta.approx_checkpoint_idx as usize] += 1;
                }
                if visit_usize < agg.visit_hist.len() {
                    agg.visit_hist[visit_usize] += 1;
                }
                let _ = score;
            }
            Ok(agg)
        }
    };

    let agg = install_in_pool(thread_pool, work)?;
    Ok((agg, parallel_enabled, par_threads))
}

fn run_dispatch_exact_slot(
    slot: &LoadedDispatchSlot,
    batch: &FeatureBatch,
    dispatch_meta: &DispatchMeta,
    row_indices: &[usize],
    threads: usize,
    chunk_rows: usize,
    parallel_min_rows: usize,
) -> Result<(DispatchSlotRun, bool, usize)> {
    let n = row_indices.len();
    let mut scores = vec![0.0f32; n];
    let mut passes = vec![false; n];
    let mut visits = vec![0i32; n];
    let par_threads = active_threads(threads);
    let parallel_enabled = par_threads > 1 && n >= parallel_min_rows;
    let t0 = Instant::now();

    let mut work = || -> Result<()> {
        if parallel_enabled {
            scores
                .par_chunks_mut(chunk_rows)
                .zip(passes.par_chunks_mut(chunk_rows))
                .zip(visits.par_chunks_mut(chunk_rows))
                .enumerate()
                .try_for_each(|(chunk_idx, ((score_chunk, pass_chunk), visit_chunk))| -> Result<()> {
                    let start = chunk_idx * chunk_rows;
                    let row_chunk = &row_indices[start..(start + score_chunk.len())];
                    for local_idx in 0..score_chunk.len() {
                        let row_idx = row_chunk[local_idx];
                        let row_threshold = match slot.threshold_mode {
                            DispatchThresholdMode::Tau => dispatch_meta.tau_used[row_idx],
                            DispatchThresholdMode::Zero => 0.0,
                        };
                        let feat = batch.row(row_idx);
                        let (score, pass, visit, _) = unsafe {
                            traverse_float_nomiss(
                                &slot.model,
                                feat,
                                row_threshold,
                                slot.base_score,
                                InferMode::L2RouteExactReordered,
                                &slot.plan,
                                None,
                                0.0,
                                0.0,
                                1,
                            )
                        }?;
                        score_chunk[local_idx] = score;
                        pass_chunk[local_idx] = pass;
                        visit_chunk[local_idx] = visit;
                    }
                    Ok(())
                })?;
        } else {
            for (local_idx, &row_idx) in row_indices.iter().enumerate() {
                let row_threshold = match slot.threshold_mode {
                    DispatchThresholdMode::Tau => dispatch_meta.tau_used[row_idx],
                    DispatchThresholdMode::Zero => 0.0,
                };
                let feat = batch.row(row_idx);
                let (score, pass, visit, _) = unsafe {
                    traverse_float_nomiss(
                        &slot.model,
                        feat,
                        row_threshold,
                        slot.base_score,
                        InferMode::L2RouteExactReordered,
                        &slot.plan,
                        None,
                        0.0,
                        0.0,
                        1,
                    )
                }?;
                scores[local_idx] = score;
                passes[local_idx] = pass;
                visits[local_idx] = visit;
            }
        }
        Ok(())
    };
    let pool = build_thread_pool(threads)?;
    install_in_pool(pool.as_ref(), &mut work)?;
    let elapsed = t0.elapsed().as_secs_f64();
    let visit_hist = histogram_from_visits(&visits, slot.plan.n_trees);
    let reject_cnt = passes.iter().filter(|&&x| x).count();
    let stats = DispatchSlotStats {
        slot: slot.slot,
        label: slot.label.clone(),
        rows: n,
        rows_per_sec: n as f64 / elapsed.max(1e-9),
        elapsed_sec: elapsed,
        avg_visited_trees: visits.iter().map(|&v| v as f64).sum::<f64>() / n.max(1) as f64,
        p99_visited_trees: pct_from_hist(&visit_hist, 0.99, n),
        route_reject_rate: reject_cnt as f64 / n.max(1) as f64,
    };
    Ok((
        DispatchSlotRun {
            scores,
            passes,
            visits,
            stats,
            visit_hist,
            reject_cnt,
        },
        parallel_enabled,
        par_threads,
    ))
}

fn write_dispatch_output_tsv(
    path: &PathBuf,
    batch: &FeatureBatch,
    n: usize,
    scores: &[f32],
    passes: &[bool],
    visits: &[i32],
) -> Result<()> {
    let mut w = BufWriter::new(
        fs::File::create(path).with_context(|| format!("create {}", path.display()))?,
    );
    writeln!(
        w,
        "TransactionID\tisFraud\troute_score\tl2_decision\tvisited_trees"
    )?;
    for i in 0..n {
        let decision = if passes[i] { "REJECT" } else { "REFER" };
        writeln!(
            w,
            "{}\t{}\t{:.9}\t{}\t{}",
            batch.id(i),
            batch.label(i),
            scores[i],
            decision,
            visits[i]
        )?;
    }
    Ok(())
}

fn write_qs_output_tsv(
    path: &PathBuf,
    batch: &FeatureBatch,
    n: usize,
    scores: &[f32],
) -> Result<()> {
    let mut w = BufWriter::new(
        fs::File::create(path).with_context(|| format!("create {}", path.display()))?,
    );
    writeln!(w, "TransactionID\tisFraud\tl2_score")?;
    for i in 0..n {
        writeln!(w, "{}\t{}\t{:.9}", batch.id(i), batch.label(i), scores[i])?;
    }
    Ok(())
}

fn write_qs_fast_output_tsv(
    path: &PathBuf,
    batch: &FeatureBatch,
    n: usize,
    lower: &[f32],
    upper: &[f32],
    shadow_reject: &[bool],
    final_reject: &[bool],
    fallback_used: &[u8],
    route_scores: &[f32],
    block_evals: &[u32],
    exact_visits: &[i32],
) -> Result<()> {
    let mut w = BufWriter::new(
        fs::File::create(path).with_context(|| format!("create {}", path.display()))?,
    );
    writeln!(
        w,
        "TransactionID\tisFraud\tlower_bound\tupper_bound\tshadow_decision\tl2_decision\tfallback_used\troute_score\tqs_block_evals\texact_visited_trees"
    )?;
    for i in 0..n {
        let shadow = if shadow_reject[i] { "REJECT" } else { "REFER" };
        let final_decision = if final_reject[i] { "REJECT" } else { "REFER" };
        writeln!(
            w,
            "{}\t{}\t{:.9}\t{:.9}\t{}\t{}\t{}\t{:.9}\t{}\t{}",
            batch.id(i),
            batch.label(i),
            lower[i],
            upper[i],
            shadow,
            final_decision,
            fallback_used[i],
            route_scores[i],
            block_evals[i],
            exact_visits[i]
        )?;
    }
    Ok(())
}

fn write_qs_prefix_output_tsv(
    path: &PathBuf,
    batch: &FeatureBatch,
    n: usize,
    prefix_scores: &[f32],
    trees_used: &[u16],
    shadow_reject: &[bool],
    final_reject: &[bool],
    fallback_used: &[u8],
    route_scores: &[f32],
    block_evals: &[u32],
    exact_visits: &[i32],
    direct_checkpoints: &[u16],
    fallback_entry_checkpoints: &[u16],
    router_choice: &[u16],
    router_tau_bin: &[u16],
) -> Result<()> {
    let mut w = BufWriter::new(
        fs::File::create(path).with_context(|| format!("create {}", path.display()))?,
    );
    writeln!(
        w,
        "TransactionID\tisFraud\tprefix_score\tprefix_trees_used\tshadow_decision\tl2_decision\tfallback_used\tdirect_checkpoint\tfallback_entry_checkpoint\trouter_choice\trouter_tau_bin\troute_score\tqs_block_evals\texact_visited_trees"
    )?;
    for i in 0..n {
        let shadow = if shadow_reject[i] { "REJECT" } else { "REFER" };
        let final_decision = if final_reject[i] { "REJECT" } else { "REFER" };
        writeln!(
            w,
            "{}\t{}\t{:.9}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{}\t{}",
            batch.id(i),
            batch.label(i),
            prefix_scores[i],
            trees_used[i],
            shadow,
            final_decision,
            fallback_used[i],
            direct_checkpoints[i],
            fallback_entry_checkpoints[i],
            router_choice[i],
            router_tau_bin[i],
            route_scores[i],
            block_evals[i],
            exact_visits[i]
        )?;
    }
    Ok(())
}

fn write_qs_prefix_trace_jsonl(
    path: &PathBuf,
    batch: &FeatureBatch,
    route_meta: &RouteMeta,
    n: usize,
    checkpoints: &[usize],
    prefix_scores: &[f32],
    trees_used: &[u16],
    final_reject: &[bool],
    fallback_used: &[u8],
    router_choice: &[u16],
    router_tau_bin: &[u16],
    checkpoint_scores: &[f32],
    checkpoint_work_evals: &[u32],
    checkpoint_deltas: &[f32],
    checkpoint_resolved_early: &[u32],
) -> Result<()> {
    let cp_len = checkpoints.len();
    let mut w = BufWriter::new(
        fs::File::create(path).with_context(|| format!("create {}", path.display()))?,
    );
    for row_idx in 0..n {
        if route_meta.active[row_idx] == 0 {
            continue;
        }
        let st = row_idx * cp_len;
        let ed = st + cp_len;
        let payload = serde_json::json!({
            "TransactionID": batch.id(row_idx),
            "isFraud": batch.label(row_idx),
            "fold_id": route_meta.fold_id[row_idx],
            "tau_used": route_meta.tau_used[row_idx],
            "exact_final_label": if final_reject[row_idx] { 1 } else { 0 },
            "prefix_score": prefix_scores[row_idx],
            "prefix_trees_used": trees_used[row_idx],
            "fallback_used": fallback_used[row_idx],
            "router_choice": router_choice[row_idx],
            "router_tau_bin": router_tau_bin[row_idx],
            "checkpoints": checkpoints,
            "checkpoint_scores": checkpoint_scores[st..ed],
            "checkpoint_work_evals": checkpoint_work_evals[st..ed],
            "checkpoint_deltas": checkpoint_deltas[st..ed],
            "checkpoint_resolved_early": checkpoint_resolved_early[st..ed],
        });
        writeln!(w, "{}", serde_json::to_string(&payload)?)?;
    }
    Ok(())
}

fn run_dispatch_l2_exact(args: DispatchL2ExactArgs) -> Result<()> {
    let batch = load_features(&args.feat_bin)?;
    let dispatch_meta = load_dispatch_meta(&args.dispatch_meta)?;
    if dispatch_meta.n_rows != batch.n_rows {
        bail!(
            "dispatch meta row mismatch: dispatch_meta={} feat_bin={}",
            dispatch_meta.n_rows,
            batch.n_rows
        );
    }
    for i in 0..batch.n_rows {
        if batch.id(i) != dispatch_meta.ids[i] {
            bail!(
                "dispatch meta TransactionID mismatch at row {}: feat_bin={} dispatch_meta={}",
                i,
                batch.id(i),
                dispatch_meta.ids[i]
            );
        }
    }

    let expert_manifest_txt = fs::read_to_string(&args.expert_manifest)
        .with_context(|| format!("read {}", args.expert_manifest.display()))?;
    let expert_manifest: SegmentExpertsManifest =
        serde_json::from_str(&expert_manifest_txt).context("parse expert manifest failed")?;

    let mut slots = Vec::new();
    slots.push(load_dispatch_slot(
        0,
        "global".to_string(),
        &args.global_model_dir,
        parse_dispatch_threshold_mode(&args.global_threshold_mode)?,
    )?);
    for (idx, expert) in expert_manifest.selected_experts.iter().enumerate() {
        slots.push(load_dispatch_slot(
            (idx + 1) as i32,
            format!("seg:{}", expert.seg_key),
            &expert.model_dir,
            parse_dispatch_threshold_mode(&expert.threshold_mode)?,
        )?);
    }
    let slot_pos: HashMap<i32, usize> = slots
        .iter()
        .enumerate()
        .map(|(i, s)| (s.slot, i))
        .collect();

    let n = args.max_rows.unwrap_or(batch.n_rows).min(batch.n_rows);
    let mut row_buckets = vec![Vec::<usize>::new(); slots.len()];
    let mut scores = vec![f32::NAN; n];
    let mut passes = vec![false; n];
    let mut visits = vec![0i32; n];
    let mut active_cnt = 0usize;
    let mut total_reject = 0usize;
    let mut slot_details = Vec::new();
    let mut merged_hist = vec![0u64; 1];
    let mut parallel_enabled_any = false;
    let mut used_threads = active_threads(args.threads);
    let t0 = Instant::now();

    for row_idx in 0..n {
        let active = dispatch_meta.active[row_idx] != 0;
        if !active {
            continue;
        }
        active_cnt += 1;
        let slot = dispatch_meta.slot_id[row_idx];
        let pos = *slot_pos
            .get(&slot)
            .with_context(|| format!("unknown dispatch slot id {}", slot))?;
        row_buckets[pos].push(row_idx);
    }

    for (slot_idx, row_indices) in row_buckets.iter().enumerate() {
        if row_indices.is_empty() {
            continue;
        }
        let (run, parallel_enabled, slot_threads) = run_dispatch_exact_slot(
            &slots[slot_idx],
            &batch,
            &dispatch_meta,
            row_indices,
            args.threads,
            args.chunk_rows.max(1),
            args.parallel_min_rows.max(1),
        )?;
        parallel_enabled_any |= parallel_enabled;
        used_threads = used_threads.max(slot_threads);
        for (local_idx, &row_idx) in row_indices.iter().enumerate() {
            scores[row_idx] = run.scores[local_idx];
            passes[row_idx] = run.passes[local_idx];
            visits[row_idx] = run.visits[local_idx];
        }
        total_reject += run.reject_cnt;
        if merged_hist.len() < run.visit_hist.len() {
            merged_hist.resize(run.visit_hist.len(), 0);
        }
        for (dst, src) in merged_hist.iter_mut().zip(run.visit_hist.iter()) {
            *dst += *src;
        }
        slot_details.push(run.stats);
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let avg_visited = if n == 0 {
        0.0
    } else {
        visits.iter().map(|&v| v as f64).sum::<f64>() / n as f64
    };
    let stats = DispatchStats {
        n_rows: n,
        threads: used_threads,
        parallel_enabled: parallel_enabled_any,
        feature_format: batch.format_tag.clone(),
        nan_free: batch.nan_free,
        rows_per_sec: n as f64 / elapsed.max(1e-9),
        elapsed_sec: elapsed,
        avg_visited_trees: avg_visited,
        p50_visited_trees: pct_from_hist(&merged_hist, 0.50, n),
        p90_visited_trees: pct_from_hist(&merged_hist, 0.90, n),
        p99_visited_trees: pct_from_hist(&merged_hist, 0.99, n),
        route_reject_rate: total_reject as f64 / n.max(1) as f64,
        route_active_rate: active_cnt as f64 / n.max(1) as f64,
        rss_peak_mb: peak_rss_mb(),
        slots_used: slot_details.len(),
        slot_details,
        visit_hist: merged_hist,
    };

    if let Some(path) = &args.out_tsv {
        write_dispatch_output_tsv(path, &batch, n, &scores, &passes, &visits)?;
    }
    if let Some(path) = &args.stats_json {
        let txt = serde_json::to_string_pretty(&stats)?;
        fs::write(path, txt).with_context(|| format!("write {}", path.display()))?;
    }

    println!(
        "mode=l2-exact-dispatch rows={} threads={} parallel={} rows/s={:.1} avg_visited={:.1} slots_used={} format={}",
        stats.n_rows,
        stats.threads,
        stats.parallel_enabled,
        stats.rows_per_sec,
        stats.avg_visited_trees,
        stats.slots_used,
        stats.feature_format,
    );

    Ok(())
}

fn run_qs_l2_exact(args: QsL2ExactArgs) -> Result<()> {
    let pack = qs_exact::load_qs_pack(&args.qs_pack)?;
    let batch = load_features(&args.feat_bin)?;
    let n = args.max_rows.unwrap_or(batch.n_rows).min(batch.n_rows);

    let thread_pool = build_thread_pool(args.threads)?;
    let t0 = Instant::now();
    let (scores, parallel_enabled, used_threads, agg) = qs_exact::run_qs_exact(
        &pack,
        &batch,
        n,
        args.threads,
        args.chunk_rows.max(1),
        args.parallel_min_rows.max(1),
        thread_pool.as_ref(),
    )?;
    let elapsed = t0.elapsed().as_secs_f64();

    if let Some(path) = &args.out_tsv {
        write_qs_output_tsv(path, &batch, n, &scores)?;
    }

    let stats = QsExactStats {
        n_rows: n,
        n_features: pack.n_features,
        n_trees: pack.n_trees,
        n_blocks: pack.n_blocks,
        threads: used_threads,
        parallel_enabled,
        feature_format: batch.format_tag.clone(),
        nan_free: batch.nan_free,
        elapsed_sec: elapsed,
        rows_per_sec: n as f64 / elapsed.max(1e-9),
        rss_peak_mb: peak_rss_mb(),
        avg_blocks_per_row: agg.total_block_evals as f64 / n.max(1) as f64,
        avg_blocks_per_tree: agg.total_block_evals as f64 / (n.max(1) * pack.n_trees).max(1) as f64,
        resolved_early_rate: agg.resolved_early_trees as f64 / (n.max(1) * pack.n_trees).max(1) as f64,
    };

    if let Some(path) = &args.stats_json {
        let txt = serde_json::to_string_pretty(&stats)?;
        fs::write(path, txt).with_context(|| format!("write {}", path.display()))?;
    }

    println!(
        "mode=l2-qs-exact rows={} threads={} parallel={} rows/s={:.1} avg_blocks/row={:.1} resolved_early_rate={:.6} format={}",
        stats.n_rows,
        stats.threads,
        stats.parallel_enabled,
        stats.rows_per_sec,
        stats.avg_blocks_per_row,
        stats.resolved_early_rate,
        stats.feature_format,
    );

    Ok(())
}

fn run_qs_l2_fast(args: QsL2FastArgs) -> Result<()> {
    let pack = qs_exact::load_qs_pack(&args.qs_pack)?;
    let batch = load_features(&args.feat_bin)?;
    let route_meta = load_route_meta(&args.route_meta)?;
    let model = load_soa(&args.soa)?;
    let bounds = load_bounds(&args.bounds)?;
    let raw_tree_order = load_tree_order(&args.tree_order)?;
    let plan = materialize_tree_plan(&bounds, Some(&raw_tree_order), None)?;
    let base_score = parse_base_score(&args.model_json)?;

    if pack.n_features() != batch.n_cols {
        bail!(
            "feature count mismatch: feat_bin={} qs_pack={}",
            batch.n_cols,
            pack.n_features()
        );
    }
    if route_meta.n_rows != batch.n_rows {
        bail!(
            "route meta row mismatch: route_meta={} feat_bin={}",
            route_meta.n_rows,
            batch.n_rows
        );
    }
    for i in 0..batch.n_rows {
        if batch.id(i) != route_meta.ids[i] {
            bail!(
                "route meta TransactionID mismatch at row {}: feat_bin={} route_meta={}",
                i,
                batch.id(i),
                route_meta.ids[i]
            );
        }
    }

    let n = args.max_rows.unwrap_or(batch.n_rows).min(batch.n_rows);
    let mut lower = vec![f32::NAN; n];
    let mut upper = vec![f32::NAN; n];
    let mut shadow_reject = vec![false; n];
    let mut final_reject = vec![false; n];
    let mut fallback_used = vec![0u8; n];
    let mut route_scores = vec![f32::NAN; n];
    let mut block_evals = vec![0u32; n];
    let mut exact_visits = vec![0i32; n];

    let par_threads = active_threads(args.threads);
    let parallel_enabled = par_threads > 1 && n >= args.parallel_min_rows.max(1);
    let hist_len = plan.n_trees + 1;
    let thread_pool = build_thread_pool(args.threads)?;
    let t0 = Instant::now();

    let mut work = || -> Result<(usize, usize, usize, usize, usize, usize, u64, u64, u64, Vec<u64>)> {
        if parallel_enabled {
            lower
                .par_chunks_mut(args.chunk_rows.max(1))
                .zip(upper.par_chunks_mut(args.chunk_rows.max(1)))
                .zip(shadow_reject.par_chunks_mut(args.chunk_rows.max(1)))
                .zip(final_reject.par_chunks_mut(args.chunk_rows.max(1)))
                .zip(fallback_used.par_chunks_mut(args.chunk_rows.max(1)))
                .zip(route_scores.par_chunks_mut(args.chunk_rows.max(1)))
                .zip(block_evals.par_chunks_mut(args.chunk_rows.max(1)))
                .zip(exact_visits.par_chunks_mut(args.chunk_rows.max(1)))
                .enumerate()
                .map(
                    |(
                        chunk_idx,
                        (((((((lower_chunk, upper_chunk), shadow_chunk), final_chunk), fallback_chunk), score_chunk), block_chunk), visit_chunk),
                    )| -> Result<(usize, usize, usize, usize, usize, usize, u64, u64, u64, Vec<u64>)> {
                        let start = chunk_idx * args.chunk_rows.max(1);
                        let mut ranks = vec![0u8; pack.n_features()];
                        let mut missing = vec![0u8; pack.n_features()];
                        let mut active_cnt = 0usize;
                        let mut shadow_reject_cnt = 0usize;
                        let mut final_reject_cnt = 0usize;
                        let mut direct_reject_cnt = 0usize;
                        let mut direct_refer_cnt = 0usize;
                        let mut fallback_cnt = 0usize;
                        let mut total_block_evals = 0u64;
                        let mut resolved_early_trees = 0u64;
                        let mut exact_visit_sum = 0u64;
                        let mut exact_hist = vec![0u64; hist_len];

                        for local_idx in 0..lower_chunk.len() {
                            let row_idx = start + local_idx;
                            let active = route_meta.active[row_idx] != 0;
                            if !active {
                                shadow_chunk[local_idx] = false;
                                final_chunk[local_idx] = false;
                                fallback_chunk[local_idx] = 0;
                                block_chunk[local_idx] = 0;
                                visit_chunk[local_idx] = 0;
                                score_chunk[local_idx] = f32::NAN;
                                lower_chunk[local_idx] = f32::NAN;
                                upper_chunk[local_idx] = f32::NAN;
                                continue;
                            }

                            active_cnt += 1;
                            let feat = batch.row(row_idx);
                            let interval = qs_exact::interval_row(&pack, feat, &mut ranks, &mut missing)?;
                            let row_tau = route_meta.tau_used[row_idx];
                            let shadow_is_reject = interval.lower >= row_tau;
                            let direct_refer = interval.upper < row_tau;

                            lower_chunk[local_idx] = interval.lower;
                            upper_chunk[local_idx] = interval.upper;
                            block_chunk[local_idx] = interval.block_evals.min(u32::MAX as u64) as u32;
                            total_block_evals += interval.block_evals;
                            resolved_early_trees += interval.resolved_early_trees;
                            shadow_chunk[local_idx] = shadow_is_reject;
                            if shadow_is_reject {
                                shadow_reject_cnt += 1;
                            }

                            if shadow_is_reject {
                                direct_reject_cnt += 1;
                                final_reject_cnt += 1;
                                final_chunk[local_idx] = true;
                                fallback_chunk[local_idx] = 0;
                                score_chunk[local_idx] = interval.lower;
                                visit_chunk[local_idx] = 0;
                            } else if direct_refer {
                                direct_refer_cnt += 1;
                                final_chunk[local_idx] = false;
                                fallback_chunk[local_idx] = 0;
                                score_chunk[local_idx] = interval.upper;
                                visit_chunk[local_idx] = 0;
                            } else {
                                fallback_cnt += 1;
                                fallback_chunk[local_idx] = 1;
                                if args.shadow_only {
                                    final_chunk[local_idx] = false;
                                    score_chunk[local_idx] = interval.upper;
                                    visit_chunk[local_idx] = 0;
                                } else {
                                    let (exact_score, exact_reject_row, visit, _) = if batch.nan_free {
                                        unsafe {
                                            traverse_float_nomiss(
                                                &model,
                                                feat,
                                                row_tau,
                                                base_score,
                                                InferMode::L2RouteExactReordered,
                                                &plan,
                                                None,
                                                0.0,
                                                0.0,
                                                1,
                                            )
                                        }?
                                    } else {
                                        traverse_generic(
                                            &model,
                                            None,
                                            None,
                                            feat,
                                            row_tau,
                                            base_score,
                                            InferMode::L2RouteExactReordered,
                                            &plan,
                                            None,
                                            0.0,
                                            0.0,
                                            1,
                                        )?
                                    };
                                    if exact_reject_row {
                                        final_reject_cnt += 1;
                                    }
                                    let visit_usize = visit.max(0) as usize;
                                    if visit_usize < exact_hist.len() {
                                        exact_hist[visit_usize] += 1;
                                    }
                                    exact_visit_sum += visit_usize as u64;
                                    final_chunk[local_idx] = exact_reject_row;
                                    score_chunk[local_idx] = exact_score;
                                    visit_chunk[local_idx] = visit;
                                }
                            }
                        }

                        Ok((
                            active_cnt,
                            shadow_reject_cnt,
                            final_reject_cnt,
                            direct_reject_cnt,
                            direct_refer_cnt,
                            fallback_cnt,
                            total_block_evals,
                            resolved_early_trees,
                            exact_visit_sum,
                            exact_hist,
                        ))
                    },
                )
                .try_reduce(
                    || (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0u64, 0u64, 0u64, vec![0u64; hist_len]),
                    |a, b| {
                        let mut hist = a.9;
                        for (dst, src) in hist.iter_mut().zip(b.9.iter()) {
                            *dst += *src;
                        }
                        Ok((
                            a.0 + b.0,
                            a.1 + b.1,
                            a.2 + b.2,
                            a.3 + b.3,
                            a.4 + b.4,
                            a.5 + b.5,
                            a.6 + b.6,
                            a.7 + b.7,
                            a.8 + b.8,
                            hist,
                        ))
                    },
                )
        } else {
            let mut ranks = vec![0u8; pack.n_features()];
            let mut missing = vec![0u8; pack.n_features()];
            let mut active_cnt = 0usize;
            let mut shadow_reject_cnt = 0usize;
            let mut final_reject_cnt = 0usize;
            let mut direct_reject_cnt = 0usize;
            let mut direct_refer_cnt = 0usize;
            let mut fallback_cnt = 0usize;
            let mut total_block_evals = 0u64;
            let mut resolved_early_trees = 0u64;
            let mut exact_visit_sum = 0u64;
            let mut exact_hist = vec![0u64; hist_len];

            for row_idx in 0..n {
                let active = route_meta.active[row_idx] != 0;
                if !active {
                    continue;
                }

                active_cnt += 1;
                let feat = batch.row(row_idx);
                let interval = qs_exact::interval_row(&pack, feat, &mut ranks, &mut missing)?;
                let row_tau = route_meta.tau_used[row_idx];
                let shadow_is_reject = interval.lower >= row_tau;
                let direct_refer = interval.upper < row_tau;

                lower[row_idx] = interval.lower;
                upper[row_idx] = interval.upper;
                block_evals[row_idx] = interval.block_evals.min(u32::MAX as u64) as u32;
                total_block_evals += interval.block_evals;
                resolved_early_trees += interval.resolved_early_trees;
                shadow_reject[row_idx] = shadow_is_reject;
                if shadow_is_reject {
                    shadow_reject_cnt += 1;
                }

                if shadow_is_reject {
                    direct_reject_cnt += 1;
                    final_reject_cnt += 1;
                    final_reject[row_idx] = true;
                    route_scores[row_idx] = interval.lower;
                } else if direct_refer {
                    direct_refer_cnt += 1;
                    final_reject[row_idx] = false;
                    route_scores[row_idx] = interval.upper;
                } else {
                    fallback_cnt += 1;
                    fallback_used[row_idx] = 1;
                    if args.shadow_only {
                        final_reject[row_idx] = false;
                        route_scores[row_idx] = interval.upper;
                    } else {
                        let (exact_score, exact_reject_row, visit, _) = if batch.nan_free {
                            unsafe {
                                traverse_float_nomiss(
                                    &model,
                                    feat,
                                    row_tau,
                                    base_score,
                                    InferMode::L2RouteExactReordered,
                                    &plan,
                                    None,
                                    0.0,
                                    0.0,
                                    1,
                                )
                            }?
                        } else {
                            traverse_generic(
                                &model,
                                None,
                                None,
                                feat,
                                row_tau,
                                base_score,
                                InferMode::L2RouteExactReordered,
                                &plan,
                                None,
                                0.0,
                                0.0,
                                1,
                            )?
                        };
                        if exact_reject_row {
                            final_reject_cnt += 1;
                        }
                        let visit_usize = visit.max(0) as usize;
                        if visit_usize < exact_hist.len() {
                            exact_hist[visit_usize] += 1;
                        }
                        exact_visit_sum += visit_usize as u64;
                        final_reject[row_idx] = exact_reject_row;
                        route_scores[row_idx] = exact_score;
                        exact_visits[row_idx] = visit;
                    }
                }
            }

            Ok((
                active_cnt,
                shadow_reject_cnt,
                final_reject_cnt,
                direct_reject_cnt,
                direct_refer_cnt,
                fallback_cnt,
                total_block_evals,
                resolved_early_trees,
                exact_visit_sum,
                exact_hist,
            ))
        }
    };
    let (
        active_cnt,
        shadow_reject_cnt,
        final_reject_cnt,
        direct_reject_cnt,
        direct_refer_cnt,
        fallback_cnt,
        total_block_evals,
        resolved_early_trees,
        exact_visit_sum,
        exact_visit_hist,
    ) = install_in_pool(thread_pool.as_ref(), work)?;
    let elapsed = t0.elapsed().as_secs_f64();

    if let Some(path) = &args.out_tsv {
        write_qs_fast_output_tsv(
            path,
            &batch,
            n,
            &lower,
            &upper,
            &shadow_reject,
            &final_reject,
            &fallback_used,
            &route_scores,
            &block_evals,
            &exact_visits,
        )?;
    }

    let active_f = active_cnt.max(1) as f64;
    let fallback_n = fallback_cnt.max(1);
    let stats = QsFastStats {
        n_rows: n,
        n_features: pack.n_features(),
        n_trees: pack.n_trees(),
        n_blocks: pack.n_blocks(),
        threads: par_threads,
        parallel_enabled,
        feature_format: batch.format_tag.clone(),
        nan_free: batch.nan_free,
        shadow_only: args.shadow_only,
        elapsed_sec: elapsed,
        rows_per_sec: n as f64 / elapsed.max(1e-9),
        rss_peak_mb: peak_rss_mb(),
        route_active_rate: active_cnt as f64 / n.max(1) as f64,
        shadow_reject_rate: shadow_reject_cnt as f64 / active_f,
        final_reject_rate: final_reject_cnt as f64 / active_f,
        direct_reject_rate: direct_reject_cnt as f64 / active_f,
        direct_refer_rate: direct_refer_cnt as f64 / active_f,
        fallback_rate: fallback_cnt as f64 / active_f,
        avg_blocks_per_row: total_block_evals as f64 / active_f,
        avg_blocks_per_tree: total_block_evals as f64 / ((active_cnt.max(1) * pack.n_trees()).max(1) as f64),
        resolved_early_rate: resolved_early_trees as f64
            / ((active_cnt.max(1) * pack.n_trees()).max(1) as f64),
        exact_avg_visited_trees: exact_visit_sum as f64 / fallback_n as f64,
        exact_p99_visited_trees: pct_from_hist(&exact_visit_hist, 0.99, fallback_cnt),
        exact_visit_hist,
    };

    if let Some(path) = &args.stats_json {
        let txt = serde_json::to_string_pretty(&stats)?;
        fs::write(path, txt).with_context(|| format!("write {}", path.display()))?;
    }

    println!(
        "mode=l2-qs-fast rows={} threads={} parallel={} rows/s={:.1} active_rate={:.4} fallback_rate={:.4} avg_blocks/row={:.1}",
        stats.n_rows,
        stats.threads,
        stats.parallel_enabled,
        stats.rows_per_sec,
        stats.route_active_rate,
        stats.fallback_rate,
        stats.avg_blocks_per_row,
    );

    Ok(())
}

fn run_qs_l2_prefix_cal_single_route_hot(
    runtime: &LoadedPrefixRuntime,
    model: &SoaModel,
    batch: &FeatureBatch,
    route_meta: &RouteMeta,
    args: &QsL2PrefixCalArgs,
    n: usize,
    hist_len: usize,
    parallel_enabled: bool,
    thread_pool: Option<&rayon::ThreadPool>,
    prefix_scores: &mut [f32],
    trees_used: &mut [u16],
    shadow_reject: &mut [bool],
    final_reject: &mut [bool],
    fallback_used: &mut [u8],
    direct_checkpoints: &mut [u16],
    fallback_entry_checkpoints: &mut [u16],
    route_scores: &mut [f32],
    block_evals: &mut [u32],
    exact_visits: &mut [i32],
) -> Result<(usize, usize, usize, usize, usize, usize, u64, u64, u64, u64, Vec<u64>)> {
    #[derive(Default)]
    struct L2SingleRouteAgg {
        active_cnt: usize,
        shadow_reject_cnt: usize,
        final_reject_cnt: usize,
        direct_reject_cnt: usize,
        direct_refer_cnt: usize,
        fallback_cnt: usize,
        total_block_evals: u64,
        resolved_early_trees: u64,
        total_prefix_trees: u64,
    }

    impl L2SingleRouteAgg {
        #[inline(always)]
        fn merge(&mut self, other: Self) {
            self.active_cnt += other.active_cnt;
            self.shadow_reject_cnt += other.shadow_reject_cnt;
            self.final_reject_cnt += other.final_reject_cnt;
            self.direct_reject_cnt += other.direct_reject_cnt;
            self.direct_refer_cnt += other.direct_refer_cnt;
            self.fallback_cnt += other.fallback_cnt;
            self.total_block_evals += other.total_block_evals;
            self.resolved_early_trees += other.resolved_early_trees;
            self.total_prefix_trees += other.total_prefix_trees;
        }
    }

    struct L2SingleRouteScratch {
        ranks: Vec<u8>,
        missing: Vec<u8>,
        hot_buf: Option<HotFeatureBuf>,
        hot_cache: Option<LazyHotFeatureCache>,
    }

    impl L2SingleRouteScratch {
        #[inline]
        fn new(runtime: &LoadedPrefixRuntime, n_features: usize) -> Self {
            Self {
                ranks: vec![0u8; n_features],
                missing: vec![0u8; n_features],
                hot_buf: runtime
                    .compiled_hot_pack
                    .as_ref()
                    .map(|pack| HotFeatureBuf::new(pack.n_hot_features)),
                hot_cache: runtime
                    .hot_pack
                    .as_ref()
                    .map(|pack| LazyHotFeatureCache::new(pack.n_hot_features)),
            }
        }
    }

    let n_features = runtime.pack.n_features();
    let chunk_rows = args.chunk_rows.max(1);

    let work = || -> Result<L2SingleRouteAgg> {
            if parallel_enabled {
                prefix_scores
                    .par_chunks_mut(chunk_rows)
                    .zip(trees_used.par_chunks_mut(chunk_rows))
                    .zip(shadow_reject.par_chunks_mut(chunk_rows))
                    .zip(final_reject.par_chunks_mut(chunk_rows))
                    .zip(fallback_used.par_chunks_mut(chunk_rows))
                    .zip(direct_checkpoints.par_chunks_mut(chunk_rows))
                    .zip(fallback_entry_checkpoints.par_chunks_mut(chunk_rows))
                    .zip(route_scores.par_chunks_mut(chunk_rows))
                    .zip(block_evals.par_chunks_mut(chunk_rows))
                    .zip(exact_visits.par_chunks_mut(chunk_rows))
                    .enumerate()
                    .map_init(
                        || L2SingleRouteScratch::new(runtime, n_features),
                        |scratch,
                         (
                            chunk_idx,
                            (((((((((prefix_chunk, trees_chunk), shadow_chunk), final_chunk), fallback_chunk), direct_cp_chunk), fallback_cp_chunk), score_chunk), block_chunk), visit_chunk),
                        )| -> Result<L2SingleRouteAgg> {
                            let start = chunk_idx * chunk_rows;
                            let mut agg = L2SingleRouteAgg::default();

                            for local_idx in 0..prefix_chunk.len() {
                                let row_idx = start + local_idx;
                                if route_meta.active[row_idx] == 0 {
                                    continue;
                                }

                                agg.active_cnt += 1;
                                let feat = batch.row(row_idx);
                                let row_tau = route_meta.tau_used[row_idx];
                                let row_fold = route_meta.fold_id[row_idx];
                                let shadow_row = run_prefix_shadow_row_single_route_compiled(
                                    runtime,
                                    feat,
                                    row_tau,
                                    row_fold,
                                    batch.nan_free,
                                    &mut scratch.ranks,
                                    &mut scratch.missing,
                                    scratch.hot_buf.as_mut(),
                                    scratch.hot_cache.as_mut(),
                                )?;

                                prefix_chunk[local_idx] = shadow_row.prefix_score;
                                trees_chunk[local_idx] =
                                    shadow_row.trees_used.min(u16::MAX as usize) as u16;
                                block_chunk[local_idx] =
                                    shadow_row.work_evals.min(u32::MAX as u64) as u32;
                                direct_cp_chunk[local_idx] =
                                    shadow_row.direct_checkpoint.min(u16::MAX as usize) as u16;
                                fallback_cp_chunk[local_idx] = shadow_row
                                    .fallback_entry_checkpoint
                                    .min(u16::MAX as usize) as u16;
                                agg.total_block_evals += shadow_row.work_evals;
                                agg.resolved_early_trees += shadow_row.resolved_early_trees;
                                agg.total_prefix_trees += shadow_row.trees_used as u64;

                                if !shadow_row.fallback_used {
                                    let is_reject = shadow_row.shadow_reject;
                                    shadow_chunk[local_idx] = is_reject;
                                    final_chunk[local_idx] = is_reject;
                                    score_chunk[local_idx] = shadow_row.shadow_route_score;
                                    if is_reject {
                                        agg.shadow_reject_cnt += 1;
                                        agg.final_reject_cnt += 1;
                                        agg.direct_reject_cnt += 1;
                                    } else {
                                        agg.direct_refer_cnt += 1;
                                    }
                                } else {
                                    agg.fallback_cnt += 1;
                                    fallback_chunk[local_idx] = 1;
                                    score_chunk[local_idx] = shadow_row.prefix_score;
                                    if args.shadow_only {
                                        final_chunk[local_idx] = false;
                                    } else {
                                        let (exact_score, exact_reject_row, visit, _) =
                                            run_exact_continuation(
                                                runtime,
                                                model,
                                                feat,
                                                row_tau,
                                                &shadow_row,
                                                batch.nan_free,
                                                &mut scratch.ranks,
                                                &mut scratch.missing,
                                            )?;
                                        if exact_reject_row {
                                            agg.final_reject_cnt += 1;
                                        }
                                        final_chunk[local_idx] = exact_reject_row;
                                        score_chunk[local_idx] = exact_score;
                                        visit_chunk[local_idx] = visit;
                                    }
                                }
                            }

                            Ok(agg)
                        },
                    )
                    .try_reduce(
                        L2SingleRouteAgg::default,
                        |a, b| {
                            let mut acc = a;
                            acc.merge(b);
                            Ok(acc)
                        },
                    )
            } else {
                let mut scratch = L2SingleRouteScratch::new(runtime, n_features);
                let mut agg = L2SingleRouteAgg::default();

                for row_idx in 0..n {
                    if route_meta.active[row_idx] == 0 {
                        continue;
                    }

                    agg.active_cnt += 1;
                    let feat = batch.row(row_idx);
                    let row_tau = route_meta.tau_used[row_idx];
                    let row_fold = route_meta.fold_id[row_idx];
                    let shadow_row = run_prefix_shadow_row_single_route_compiled(
                        runtime,
                        feat,
                        row_tau,
                        row_fold,
                        batch.nan_free,
                        &mut scratch.ranks,
                        &mut scratch.missing,
                        scratch.hot_buf.as_mut(),
                        scratch.hot_cache.as_mut(),
                    )?;

                    prefix_scores[row_idx] = shadow_row.prefix_score;
                    trees_used[row_idx] = shadow_row.trees_used.min(u16::MAX as usize) as u16;
                    block_evals[row_idx] = shadow_row.work_evals.min(u32::MAX as u64) as u32;
                    direct_checkpoints[row_idx] =
                        shadow_row.direct_checkpoint.min(u16::MAX as usize) as u16;
                    fallback_entry_checkpoints[row_idx] =
                        shadow_row.fallback_entry_checkpoint.min(u16::MAX as usize) as u16;
                    agg.total_block_evals += shadow_row.work_evals;
                    agg.resolved_early_trees += shadow_row.resolved_early_trees;
                    agg.total_prefix_trees += shadow_row.trees_used as u64;

                    if !shadow_row.fallback_used {
                        let is_reject = shadow_row.shadow_reject;
                        shadow_reject[row_idx] = is_reject;
                        final_reject[row_idx] = is_reject;
                        route_scores[row_idx] = shadow_row.shadow_route_score;
                        if is_reject {
                            agg.shadow_reject_cnt += 1;
                            agg.final_reject_cnt += 1;
                            agg.direct_reject_cnt += 1;
                        } else {
                            agg.direct_refer_cnt += 1;
                        }
                    } else {
                        agg.fallback_cnt += 1;
                        fallback_used[row_idx] = 1;
                        route_scores[row_idx] = shadow_row.prefix_score;
                        if args.shadow_only {
                            final_reject[row_idx] = false;
                        } else {
                            let (exact_score, exact_reject_row, visit, _) =
                                run_exact_continuation(
                                    runtime,
                                    model,
                                    feat,
                                    row_tau,
                                    &shadow_row,
                                    batch.nan_free,
                                    &mut scratch.ranks,
                                    &mut scratch.missing,
                                )?;
                            if exact_reject_row {
                                agg.final_reject_cnt += 1;
                            }
                            final_reject[row_idx] = exact_reject_row;
                            route_scores[row_idx] = exact_score;
                            exact_visits[row_idx] = visit;
                        }
                    }
                }

                Ok(agg)
            }
        };

    let agg = install_in_pool(thread_pool, work)?;
    let mut exact_visit_sum = 0u64;
    let mut exact_visit_hist = vec![0u64; hist_len];
    if !args.shadow_only {
        for (fallback_flag, visit) in fallback_used.iter().zip(exact_visits.iter()) {
            if *fallback_flag == 0 {
                continue;
            }
            let visit_usize = (*visit).max(0) as usize;
            if visit_usize < exact_visit_hist.len() {
                exact_visit_hist[visit_usize] += 1;
            }
            exact_visit_sum += visit_usize as u64;
        }
    }
    Ok((
        agg.active_cnt,
        agg.shadow_reject_cnt,
        agg.final_reject_cnt,
        agg.direct_reject_cnt,
        agg.direct_refer_cnt,
        agg.fallback_cnt,
        agg.total_block_evals,
        agg.resolved_early_trees,
        agg.total_prefix_trees,
        exact_visit_sum,
        exact_visit_hist,
    ))
}

fn run_qs_l2_prefix_cal_stats(args: QsL2PrefixCalArgs) -> Result<QsPrefixCalStats> {
    let parsed_manifest = if let Some(path) = &args.bundle_manifest {
        let txt = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let parsed: PrefixCalBundleManifest =
            serde_json::from_str(&txt).with_context(|| format!("parse {}", path.display()))?;
        Some(parsed)
    } else {
        None
    };

    let (
        batch,
        route_meta,
        model,
        runtimes,
        tau_router_edges,
        v4_mode,
    ): (
        FeatureBatch,
        RouteMeta,
        SoaModel,
        Vec<LoadedPrefixOrderRoute>,
        Vec<f32>,
        String,
    ) =
        if let (Some(path), Some(parsed)) = (&args.bundle_manifest, parsed_manifest.as_ref()) {
            match parsed.format.as_str() {
                "L2PrefixCalBundleV1" => {
                    let resolved = resolve_prefix_cal_bundle(&args)?;
                    let batch = load_features(&resolved.feat_bin)?;
                    let route_meta = load_route_meta(&resolved.route_meta)?;
                    let model = load_soa(&resolved.soa)?;
                    let bounds = load_bounds(&resolved.bounds)?;
                    let runtime = load_prefix_runtime(&resolved, &model, &bounds)?;
                    (
                        batch,
                        route_meta,
                        model,
                        vec![LoadedPrefixOrderRoute {
                            tau_bin: 0,
                            label: "single".to_string(),
                            runtime,
                        }],
                        vec![-f32::MAX, f32::MAX],
                        "single_order_v1".to_string(),
                    )
                }
                "L2PrefixCalBundleV4" => {
                    let mode = parse_prefix_v4_mode(
                        parsed
                            .v4_mode
                            .as_ref()
                            .context("missing required prefix-cal V4 field: v4_mode")?,
                    )?;
                    let feat_bin = resolve_bundle_path(
                        args.feat_bin.as_ref(),
                        parsed.feat_bin.as_ref(),
                        Some(path),
                        "feat_bin",
                    )?;
                    let route_meta_path = resolve_bundle_path(
                        args.route_meta.as_ref(),
                        parsed.route_meta.as_ref(),
                        Some(path),
                        "route_meta",
                    )?;
                    let soa_path = resolve_bundle_path(
                        args.soa.as_ref(),
                        parsed.soa_bin.as_ref(),
                        Some(path),
                        "soa_bin",
                    )?;
                    let bounds_path = resolve_bundle_path(
                        args.bounds.as_ref(),
                        parsed.bounds_bin.as_ref(),
                        Some(path),
                        "bounds_bin",
                    )?;
                    let batch = load_features(&feat_bin)?;
                    let route_meta = load_route_meta(&route_meta_path)?;
                    let model = load_soa(&soa_path)?;
                    let bounds = load_bounds(&bounds_path)?;
                    match mode {
                        PrefixV4Mode::MultiOrderV1 => {
                            let (tau_edges, routes, _selected_variant, _telemetry_schema_version) =
                                load_v4_order_router(&args, path, parsed, &model, &bounds)?;
                            (
                                batch,
                                route_meta,
                                model,
                                routes,
                                tau_edges,
                                "multi_order_v1".to_string(),
                            )
                        }
                        PrefixV4Mode::PacketSchedulerV1 => {
                            let child_args = QsL2PrefixCalArgs {
                                bundle_manifest: None,
                                qs_pack: resolve_bundle_path_optional(
                                    None,
                                    parsed.pack_path.as_ref(),
                                    Some(path),
                                ),
                                calibration_json: resolve_bundle_path_optional(
                                    None,
                                    parsed.calibration_json.as_ref(),
                                    Some(path),
                                ),
                                variant_key: args
                                    .variant_key
                                    .clone()
                                    .or_else(|| parsed.selected_exact_variant.clone())
                                    .or_else(|| parsed.selected_variant.clone()),
                                feat_bin: Some(feat_bin.clone()),
                                route_meta: Some(route_meta_path.clone()),
                                soa: Some(soa_path.clone()),
                                bounds: Some(bounds_path.clone()),
                                tree_order: resolve_bundle_path_optional(
                                    None,
                                    parsed.tree_order_bin.as_ref(),
                                    Some(path),
                                ),
                                model_json: resolve_bundle_path_optional(
                                    None,
                                    parsed.model_json.as_ref(),
                                    Some(path),
                                ),
                                direct_kernel: parsed.direct_kernel.clone(),
                                certifier_kind: parsed.certifier_kind.clone(),
                                certifier_json: resolve_bundle_path_optional(
                                    None,
                                    parsed.certifier_json.as_ref(),
                                    Some(path),
                                ),
                                threads: args.threads,
                                chunk_rows: args.chunk_rows,
                                parallel_min_rows: args.parallel_min_rows,
                                max_rows: args.max_rows,
                                out_tsv: None,
                                stats_json: None,
                                trace_jsonl: None,
                                shadow_only: args.shadow_only,
                            };
                            let resolved = resolve_prefix_cal_bundle(&child_args)?;
                            let mut runtime = load_prefix_runtime(&resolved, &model, &bounds)?;
                            runtime.packet_scheduler = Some(load_v4_packet_scheduler(path, parsed)?);
                            (
                                batch,
                                route_meta,
                                model,
                                vec![LoadedPrefixOrderRoute {
                                    tau_bin: 0,
                                    label: "packet_scheduler".to_string(),
                                    runtime,
                                }],
                                vec![-f32::MAX, f32::MAX],
                                "packet_scheduler_v1".to_string(),
                            )
                        }
                        PrefixV4Mode::AnchorRescueV1 | PrefixV4Mode::AnchorRescueLossyV1 => {
                            let (runtime, _selected_variant) =
                                load_v4_anchor_rescue(&args, path, parsed, &model, &bounds)?;
                            let mode_name = if mode == PrefixV4Mode::AnchorRescueLossyV1 {
                                "anchor_rescue_lossy_v1"
                            } else {
                                "anchor_rescue_v1"
                            };
                            (
                                batch,
                                route_meta,
                                model,
                                vec![LoadedPrefixOrderRoute {
                                    tau_bin: 0,
                                    label: "anchor_rescue".to_string(),
                                    runtime,
                                }],
                                vec![-f32::MAX, f32::MAX],
                                mode_name.to_string(),
                            )
                        }
                        other => bail!(
                            "V4 mode {:?} not yet supported in qs-l2-prefix-cal runtime",
                            other
                        ),
                    }
                }
                other => bail!("unsupported prefix-cal bundle format: {}", other),
            }
        } else {
            let resolved = resolve_prefix_cal_bundle(&args)?;
            let batch = load_features(&resolved.feat_bin)?;
            let route_meta = load_route_meta(&resolved.route_meta)?;
            let model = load_soa(&resolved.soa)?;
            let bounds = load_bounds(&resolved.bounds)?;
            let runtime = load_prefix_runtime(&resolved, &model, &bounds)?;
            (
                batch,
                route_meta,
                model,
                vec![LoadedPrefixOrderRoute {
                    tau_bin: 0,
                    label: "single".to_string(),
                    runtime,
                }],
                vec![-f32::MAX, f32::MAX],
                "single_order_v1".to_string(),
            )
        };

    if runtimes.is_empty() {
        bail!("no prefix runtime routes loaded");
    }
    let first_runtime = &runtimes[0].runtime;
    let checkpoints = first_runtime.calibration.checkpoints.clone();
    let cp_len = checkpoints.len();
    if cp_len == 0 {
        bail!("prefix calibration has no checkpoints");
    }
    let base_n_features = first_runtime.pack.n_features();
    let base_n_trees = first_runtime.pack.n_trees();
    let hist_len = first_runtime.plan.n_trees + 1;
    let telemetry_schema_version = if runtimes.len() > 1 { 4 } else { first_runtime.telemetry_schema_version };
    for route in runtimes.iter() {
        if route.runtime.pack.n_features() != base_n_features {
            bail!("multi-order route feature count mismatch");
        }
        if route.runtime.pack.n_trees() != base_n_trees {
            bail!("multi-order route tree count mismatch");
        }
        if route.runtime.calibration.checkpoints != checkpoints {
            bail!("multi-order route checkpoint ladder mismatch");
        }
        if route.runtime.plan.n_trees + 1 != hist_len {
            bail!("multi-order route exact plan length mismatch");
        }
    }
    if base_n_features != batch.n_cols {
        bail!(
            "feature count mismatch: feat_bin={} qs_pack={}",
            batch.n_cols,
            base_n_features
        );
    }
    if route_meta.n_rows != batch.n_rows {
        bail!(
            "route meta row mismatch: route_meta={} feat_bin={}",
            route_meta.n_rows,
            batch.n_rows
        );
    }
    for i in 0..batch.n_rows {
        if batch.id(i) != route_meta.ids[i] {
            bail!(
                "route meta TransactionID mismatch at row {}: feat_bin={} route_meta={}",
                i,
                batch.id(i),
                route_meta.ids[i]
            );
        }
    }
    if model.n_features != batch.n_cols {
        bail!(
            "soa feature count mismatch: soa={} feat_bin={}",
            model.n_features,
            batch.n_cols
        );
    }
    let router_bin_count = tau_router_edges.len().saturating_sub(1).max(1);
    let mut router_lookup = vec![usize::MAX; router_bin_count];
    for (idx, route) in runtimes.iter().enumerate() {
        if route.tau_bin >= router_lookup.len() {
            bail!(
                "router tau_bin {} exceeds available tau bins {}",
                route.tau_bin,
                router_lookup.len()
            );
        }
        router_lookup[route.tau_bin] = idx;
    }
    for slot in router_lookup.iter_mut() {
        if *slot == usize::MAX {
            if runtimes.len() == 1 {
                *slot = 0;
            } else {
                bail!("missing router bundle assignment for one or more tau bins");
            }
        }
    }

    let n = args.max_rows.unwrap_or(batch.n_rows).min(batch.n_rows);
    let has_anchor_rescue = runtimes
        .iter()
        .any(|route| route.runtime.anchor_rescue.is_some());
    let mut prefix_scores = vec![f32::NAN; n];
    let mut trees_used = vec![0u16; n];
    let mut shadow_reject = vec![false; n];
    let mut final_reject = vec![false; n];
    let mut fallback_used = vec![0u8; n];
    let mut route_scores = vec![f32::NAN; n];
    let mut block_evals = vec![0u32; n];
    let mut exact_visits = vec![0i32; n];
    let mut direct_checkpoints = vec![0u16; n];
    let mut fallback_entry_checkpoints = vec![0u16; n];
    let mut router_choice = vec![0u16; n];
    let mut router_tau_bin = vec![0u16; n];
    let trace_enabled = args.trace_jsonl.is_some();
    let mut anchor_direct_checkpoints = vec![0u16; n];
    let mut rescue_direct_checkpoints = vec![0u16; n];
    let mut rescue_route_choice = vec![0u8; n];
    let mut checkpoint_scores_flat = if trace_enabled {
        vec![f32::NAN; n * cp_len]
    } else {
        Vec::new()
    };
    let mut checkpoint_work_evals_flat = if trace_enabled {
        vec![0u32; n * cp_len]
    } else {
        Vec::new()
    };
    let mut checkpoint_deltas_flat = if trace_enabled {
        vec![0.0f32; n * cp_len]
    } else {
        Vec::new()
    };
    let mut checkpoint_resolved_flat = if trace_enabled {
        vec![0u32; n * cp_len]
    } else {
        Vec::new()
    };

    let par_threads = active_threads(args.threads);
    let parallel_enabled =
        !trace_enabled && par_threads > 1 && n >= args.parallel_min_rows.max(1);
    let thread_pool = build_thread_pool(args.threads)?;
    let use_single_route_hot =
        !trace_enabled
            && runtimes.len() == 1
            && runtimes[0].runtime.anchor_rescue.is_none()
            && runtimes[0].runtime.packet_scheduler.is_none();
    let t0 = Instant::now();

    let mut work =
        || -> Result<(usize, usize, usize, usize, usize, usize, u64, u64, u64, u64, Vec<u64>)> {
            if parallel_enabled {
                prefix_scores
                    .par_chunks_mut(args.chunk_rows.max(1))
                    .zip(trees_used.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(shadow_reject.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(final_reject.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(fallback_used.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(direct_checkpoints.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(anchor_direct_checkpoints.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(rescue_direct_checkpoints.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(rescue_route_choice.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(fallback_entry_checkpoints.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(router_choice.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(router_tau_bin.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(route_scores.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(block_evals.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(exact_visits.par_chunks_mut(args.chunk_rows.max(1)))
                    .enumerate()
                    .map(
                        |(chunk_idx, chunk)| -> Result<(usize, usize, usize, usize, usize, usize, u64, u64, u64, u64, Vec<u64>)> {
                            let (
                                ((((((((((((((prefix_chunk, trees_chunk), shadow_chunk), final_chunk), fallback_chunk), direct_cp_chunk), anchor_cp_chunk), rescue_cp_chunk), rescue_route_chunk), fallback_cp_chunk), router_choice_chunk), router_tau_chunk), score_chunk), block_chunk), visit_chunk)
                            ) = chunk;
                            let start = chunk_idx * args.chunk_rows.max(1);
                            let mut ranks = vec![0u8; base_n_features];
                            let mut missing = vec![0u8; base_n_features];
                            let mut hot_buf_by_route: Vec<Option<HotFeatureBuf>> = runtimes
                                .iter()
                                .map(|route| {
                                    route.runtime.compiled_hot_pack.as_ref().map(|pack| {
                                        HotFeatureBuf::new(pack.n_hot_features)
                                    })
                                })
                                .collect();
                            let mut hot_cache_by_route: Vec<Option<LazyHotFeatureCache>> = runtimes
                                .iter()
                                .map(|route| {
                                    route.runtime.hot_pack.as_ref().map(|pack| {
                                        LazyHotFeatureCache::new(pack.n_hot_features)
                                    })
                                })
                                .collect();
                            let mut active_cnt = 0usize;
                            let mut shadow_reject_cnt = 0usize;
                            let mut final_reject_cnt = 0usize;
                            let mut direct_reject_cnt = 0usize;
                            let mut direct_refer_cnt = 0usize;
                            let mut fallback_cnt = 0usize;
                            let mut total_block_evals = 0u64;
                            let mut resolved_early_trees = 0u64;
                            let mut total_prefix_trees = 0u64;
                            let mut exact_visit_sum = 0u64;
                            let mut exact_hist = vec![0u64; hist_len];

                            for local_idx in 0..prefix_chunk.len() {
                                let row_idx = start + local_idx;
                                let active = route_meta.active[row_idx] != 0;
                                if !active {
                                    shadow_chunk[local_idx] = false;
                                    final_chunk[local_idx] = false;
                                    fallback_chunk[local_idx] = 0;
                                    direct_cp_chunk[local_idx] = 0;
                                    anchor_cp_chunk[local_idx] = 0;
                                    rescue_cp_chunk[local_idx] = 0;
                                    rescue_route_chunk[local_idx] = 0;
                                    fallback_cp_chunk[local_idx] = 0;
                                    router_choice_chunk[local_idx] = 0;
                                    router_tau_chunk[local_idx] = 0;
                                    block_chunk[local_idx] = 0;
                                    visit_chunk[local_idx] = 0;
                                    score_chunk[local_idx] = f32::NAN;
                                    prefix_chunk[local_idx] = f32::NAN;
                                    trees_chunk[local_idx] = 0;
                                    continue;
                                }

                                active_cnt += 1;
                                let feat = batch.row(row_idx);
                                let row_tau = route_meta.tau_used[row_idx];
                                let row_fold = route_meta.fold_id[row_idx];
                                let tau_bin = mlp_tau_bin(&tau_router_edges, row_tau)
                                    .min(router_lookup.len().saturating_sub(1));
                                let route_idx = router_lookup[tau_bin];
                                let runtime = &runtimes[route_idx].runtime;
                                router_choice_chunk[local_idx] =
                                    route_idx.min(u16::MAX as usize) as u16;
                                router_tau_chunk[local_idx] =
                                    tau_bin.min(u16::MAX as usize) as u16;
                                let shadow_row = if let Some(anchor_rescue) =
                                    runtime.anchor_rescue.as_ref()
                                {
                                    run_anchor_rescue_shadow_row(
                                        runtime,
                                        anchor_rescue,
                                        feat,
                                        row_tau,
                                        row_fold,
                                        &mut ranks,
                                        &mut missing,
                                        hot_cache_by_route[route_idx].as_mut(),
                                        false,
                                    )?
                                } else if let Some(packet_runtime) =
                                    runtime.packet_scheduler.as_ref()
                                {
                                    run_packet_shadow_row(
                                        runtime,
                                        packet_runtime,
                                        feat,
                                        row_tau,
                                        row_fold,
                                        &mut ranks,
                                        &mut missing,
                                        false,
                                    )?
                                } else {
                                    run_prefix_shadow_row_single_route_compiled(
                                        runtime,
                                        feat,
                                        row_tau,
                                        row_fold,
                                        batch.nan_free,
                                        &mut ranks,
                                        &mut missing,
                                        hot_buf_by_route[route_idx].as_mut(),
                                        hot_cache_by_route[route_idx].as_mut(),
                                    )?
                                };

                                prefix_chunk[local_idx] = shadow_row.prefix_score;
                                trees_chunk[local_idx] =
                                    shadow_row.trees_used.min(u16::MAX as usize) as u16;
                                block_chunk[local_idx] =
                                    shadow_row.work_evals.min(u32::MAX as u64) as u32;
                                direct_cp_chunk[local_idx] =
                                    shadow_row.direct_checkpoint.min(u16::MAX as usize) as u16;
                                anchor_cp_chunk[local_idx] = shadow_row
                                    .anchor_direct_checkpoint
                                    .min(u16::MAX as usize) as u16;
                                rescue_cp_chunk[local_idx] = shadow_row
                                    .rescue_direct_checkpoint
                                    .min(u16::MAX as usize) as u16;
                                rescue_route_chunk[local_idx] = shadow_row.rescue_route;
                                fallback_cp_chunk[local_idx] = shadow_row
                                    .fallback_entry_checkpoint
                                    .min(u16::MAX as usize) as u16;
                                total_block_evals += shadow_row.work_evals;
                                resolved_early_trees += shadow_row.resolved_early_trees;
                                total_prefix_trees += shadow_row.trees_used as u64;

                                if !shadow_row.fallback_used {
                                    let is_reject = shadow_row.shadow_reject;
                                    shadow_chunk[local_idx] = is_reject;
                                    final_chunk[local_idx] = is_reject;
                                    score_chunk[local_idx] = shadow_row.shadow_route_score;
                                    if is_reject {
                                        shadow_reject_cnt += 1;
                                        final_reject_cnt += 1;
                                        direct_reject_cnt += 1;
                                    } else {
                                        direct_refer_cnt += 1;
                                    }
                                } else {
                                    fallback_cnt += 1;
                                    fallback_chunk[local_idx] = 1;
                                    shadow_chunk[local_idx] = false;
                                    score_chunk[local_idx] = shadow_row.prefix_score;
                                    if args.shadow_only {
                                        final_chunk[local_idx] = false;
                                    } else {
                                        let (exact_score, exact_reject_row, visit, _) =
                                            run_exact_continuation(
                                                runtime,
                                                &model,
                                                feat,
                                                row_tau,
                                                &shadow_row,
                                                batch.nan_free,
                                                &mut ranks,
                                                &mut missing,
                                            )?;
                                        if exact_reject_row {
                                            final_reject_cnt += 1;
                                        }
                                        let visit_usize = visit.max(0) as usize;
                                        if visit_usize < exact_hist.len() {
                                            exact_hist[visit_usize] += 1;
                                        }
                                        exact_visit_sum += visit_usize as u64;
                                        final_chunk[local_idx] = exact_reject_row;
                                        score_chunk[local_idx] = exact_score;
                                        visit_chunk[local_idx] = visit;
                                    }
                                }
                            }

                            Ok((
                                active_cnt,
                                shadow_reject_cnt,
                                final_reject_cnt,
                                direct_reject_cnt,
                                direct_refer_cnt,
                                fallback_cnt,
                                total_block_evals,
                                resolved_early_trees,
                                total_prefix_trees,
                                exact_visit_sum,
                                exact_hist,
                            ))
                        },
                    )
                    .try_reduce(
                        || {
                            (
                                0usize,
                                0usize,
                                0usize,
                                0usize,
                                0usize,
                                0usize,
                                0u64,
                                0u64,
                                0u64,
                                0u64,
                                vec![0u64; hist_len],
                            )
                        },
                        |a, b| {
                            let mut hist = a.10;
                            for (dst, src) in hist.iter_mut().zip(b.10.iter()) {
                                *dst += *src;
                            }
                            Ok((
                                a.0 + b.0,
                                a.1 + b.1,
                                a.2 + b.2,
                                a.3 + b.3,
                                a.4 + b.4,
                                a.5 + b.5,
                                a.6 + b.6,
                                a.7 + b.7,
                                a.8 + b.8,
                                a.9 + b.9,
                                hist,
                            ))
                        },
                    )
            } else {
                let mut ranks = vec![0u8; base_n_features];
                let mut missing = vec![0u8; base_n_features];
                let mut hot_buf_by_route: Vec<Option<HotFeatureBuf>> = runtimes
                    .iter()
                    .map(|route| {
                        route.runtime
                            .compiled_hot_pack
                            .as_ref()
                            .map(|pack| HotFeatureBuf::new(pack.n_hot_features))
                    })
                    .collect();
                let mut hot_cache_by_route: Vec<Option<LazyHotFeatureCache>> = runtimes
                    .iter()
                    .map(|route| {
                        route.runtime.hot_pack.as_ref().map(|pack| {
                            LazyHotFeatureCache::new(pack.n_hot_features)
                        })
                    })
                    .collect();
                let mut active_cnt = 0usize;
                let mut shadow_reject_cnt = 0usize;
                let mut final_reject_cnt = 0usize;
                let mut direct_reject_cnt = 0usize;
                let mut direct_refer_cnt = 0usize;
                let mut fallback_cnt = 0usize;
                let mut total_block_evals = 0u64;
                let mut resolved_early_trees = 0u64;
                let mut total_prefix_trees = 0u64;
                let mut exact_visit_sum = 0u64;
                let mut exact_hist = vec![0u64; hist_len];

                for row_idx in 0..n {
                    let active = route_meta.active[row_idx] != 0;
                    if !active {
                        continue;
                    }

                    active_cnt += 1;
                    let feat = batch.row(row_idx);
                    let row_tau = route_meta.tau_used[row_idx];
                    let row_fold = route_meta.fold_id[row_idx];
                    let tau_bin = mlp_tau_bin(&tau_router_edges, row_tau)
                        .min(router_lookup.len().saturating_sub(1));
                    let route_idx = router_lookup[tau_bin];
                    let runtime = &runtimes[route_idx].runtime;
                    let shadow_row = if let Some(anchor_rescue) =
                        runtime.anchor_rescue.as_ref()
                    {
                        run_anchor_rescue_shadow_row(
                            runtime,
                            anchor_rescue,
                            feat,
                            row_tau,
                            row_fold,
                            &mut ranks,
                            &mut missing,
                            hot_cache_by_route[route_idx].as_mut(),
                            trace_enabled,
                        )?
                    } else if let Some(packet_runtime) =
                        runtime.packet_scheduler.as_ref()
                    {
                        run_packet_shadow_row(
                            runtime,
                            packet_runtime,
                            feat,
                            row_tau,
                            row_fold,
                            &mut ranks,
                            &mut missing,
                            trace_enabled,
                        )?
                    } else if !trace_enabled {
                        run_prefix_shadow_row_single_route_compiled(
                            runtime,
                            feat,
                            row_tau,
                            row_fold,
                            batch.nan_free,
                            &mut ranks,
                            &mut missing,
                            hot_buf_by_route[route_idx].as_mut(),
                            hot_cache_by_route[route_idx].as_mut(),
                        )?
                    } else {
                        run_prefix_shadow_row(
                            &runtime.pack,
                            runtime.hot_pack.as_ref(),
                            runtime.hot_checkpoint_layout.as_ref(),
                            runtime.direct_kernel,
                            runtime.certifier_kind,
                            runtime.atlas_certifier.as_ref(),
                            runtime.mlp_certifier.as_ref(),
                            &runtime.calibration,
                            feat,
                            row_tau,
                            row_fold,
                            &mut ranks,
                            &mut missing,
                            hot_cache_by_route[route_idx].as_mut(),
                            trace_enabled,
                        )?
                    };

                    prefix_scores[row_idx] = shadow_row.prefix_score;
                    trees_used[row_idx] = shadow_row.trees_used.min(u16::MAX as usize) as u16;
                    block_evals[row_idx] = shadow_row.work_evals.min(u32::MAX as u64) as u32;
                    direct_checkpoints[row_idx] =
                        shadow_row.direct_checkpoint.min(u16::MAX as usize) as u16;
                    anchor_direct_checkpoints[row_idx] =
                        shadow_row.anchor_direct_checkpoint.min(u16::MAX as usize) as u16;
                    rescue_direct_checkpoints[row_idx] =
                        shadow_row.rescue_direct_checkpoint.min(u16::MAX as usize) as u16;
                    rescue_route_choice[row_idx] = shadow_row.rescue_route;
                    fallback_entry_checkpoints[row_idx] =
                        shadow_row.fallback_entry_checkpoint.min(u16::MAX as usize) as u16;
                    router_choice[row_idx] = route_idx.min(u16::MAX as usize) as u16;
                    router_tau_bin[row_idx] = tau_bin.min(u16::MAX as usize) as u16;
                    total_block_evals += shadow_row.work_evals;
                    resolved_early_trees += shadow_row.resolved_early_trees;
                    total_prefix_trees += shadow_row.trees_used as u64;
                    if trace_enabled {
                        let st = row_idx * cp_len;
                        let ed = st + cp_len;
                        checkpoint_scores_flat[st..ed]
                            .copy_from_slice(&shadow_row.checkpoint_scores);
                        checkpoint_work_evals_flat[st..ed]
                            .copy_from_slice(&shadow_row.checkpoint_work_evals);
                        checkpoint_deltas_flat[st..ed]
                            .copy_from_slice(&shadow_row.checkpoint_deltas);
                        checkpoint_resolved_flat[st..ed]
                            .copy_from_slice(&shadow_row.checkpoint_resolved_early);
                    }

                    if !shadow_row.fallback_used {
                        let is_reject = shadow_row.shadow_reject;
                        shadow_reject[row_idx] = is_reject;
                        final_reject[row_idx] = is_reject;
                        route_scores[row_idx] = shadow_row.shadow_route_score;
                        if is_reject {
                            shadow_reject_cnt += 1;
                            final_reject_cnt += 1;
                            direct_reject_cnt += 1;
                        } else {
                            direct_refer_cnt += 1;
                        }
                    } else {
                        fallback_cnt += 1;
                        fallback_used[row_idx] = 1;
                        shadow_reject[row_idx] = false;
                        route_scores[row_idx] = shadow_row.prefix_score;
                        if args.shadow_only {
                            final_reject[row_idx] = false;
                        } else {
                            let (exact_score, exact_reject_row, visit, _) =
                                run_exact_continuation(
                                    runtime,
                                    &model,
                                    feat,
                                    row_tau,
                                    &shadow_row,
                                    batch.nan_free,
                                    &mut ranks,
                                    &mut missing,
                                )?;
                            if exact_reject_row {
                                final_reject_cnt += 1;
                            }
                            let visit_usize = visit.max(0) as usize;
                            if visit_usize < exact_hist.len() {
                                exact_hist[visit_usize] += 1;
                            }
                            exact_visit_sum += visit_usize as u64;
                            final_reject[row_idx] = exact_reject_row;
                            route_scores[row_idx] = exact_score;
                            exact_visits[row_idx] = visit;
                        }
                    }
                }

                Ok((
                    active_cnt,
                    shadow_reject_cnt,
                    final_reject_cnt,
                    direct_reject_cnt,
                    direct_refer_cnt,
                    fallback_cnt,
                    total_block_evals,
                    resolved_early_trees,
                    total_prefix_trees,
                    exact_visit_sum,
                    exact_hist,
                ))
            }
        };
    let (
        active_cnt,
        shadow_reject_cnt,
        final_reject_cnt,
        direct_reject_cnt,
        direct_refer_cnt,
        fallback_cnt,
        total_block_evals,
        resolved_early_trees,
        total_prefix_trees,
        exact_visit_sum,
        exact_visit_hist,
    ) = if use_single_route_hot {
        run_qs_l2_prefix_cal_single_route_hot(
            &runtimes[0].runtime,
            &model,
            &batch,
            &route_meta,
            &args,
            n,
            hist_len,
            parallel_enabled,
            thread_pool.as_ref(),
            &mut prefix_scores,
            &mut trees_used,
            &mut shadow_reject,
            &mut final_reject,
            &mut fallback_used,
            &mut direct_checkpoints,
            &mut fallback_entry_checkpoints,
            &mut route_scores,
            &mut block_evals,
            &mut exact_visits,
        )?
    } else {
        install_in_pool(thread_pool.as_ref(), work)?
    };
    let elapsed = t0.elapsed().as_secs_f64();

    let mut checkpoint_exit_counts = vec![0u64; cp_len];
    let mut direct_rows_by_checkpoint = vec![0u64; cp_len];
    let mut shadow_mismatch_by_checkpoint = vec![0u64; cp_len];
    let mut fallback_entry_checkpoint_counts = vec![0u64; cp_len];
    let mut router_choice_counts = vec![0u64; runtimes.len().max(1)];
    let mut router_tau_bin_counts = vec![0u64; router_lookup.len().max(1)];
    let mut anchor_exit_counts = vec![0u64; cp_len];
    let mut rescue_route_counts = vec![0u64; 4];
    let mut rescue_exit_counts = vec![0u64; 2];
    let mut tail_continuation_rows = 0u64;
    let mut shadow_mismatch_by_stage = vec![0u64; 2];
    let mut checkpoint_to_idx = HashMap::new();
    for (idx, checkpoint) in checkpoints.iter().copied().enumerate() {
        checkpoint_to_idx.insert(checkpoint as u16, idx);
    }
    for row_idx in 0..n {
        if route_meta.active[row_idx] != 0 {
            let route_idx = router_choice[row_idx] as usize;
            if route_idx < router_choice_counts.len() {
                router_choice_counts[route_idx] += 1;
            }
            let tau_bin = router_tau_bin[row_idx] as usize;
            if tau_bin < router_tau_bin_counts.len() {
                router_tau_bin_counts[tau_bin] += 1;
            }
        }
        let exact_reject = route_meta.exact_positive[row_idx] != 0;
        if has_anchor_rescue {
            let anchor_cp = anchor_direct_checkpoints[row_idx];
            if anchor_cp != 0 {
                if let Some(idx) = checkpoint_to_idx.get(&anchor_cp) {
                    checkpoint_exit_counts[*idx] += 1;
                    direct_rows_by_checkpoint[*idx] += 1;
                    anchor_exit_counts[*idx] += 1;
                    if shadow_reject[row_idx] != exact_reject {
                        shadow_mismatch_by_checkpoint[*idx] += 1;
                        shadow_mismatch_by_stage[0] += 1;
                    }
                }
            } else {
                let rescue_route = rescue_route_choice[row_idx] as usize;
                if rescue_route < rescue_route_counts.len() {
                    rescue_route_counts[rescue_route] += 1;
                }
                let rescue_cp = rescue_direct_checkpoints[row_idx];
                if rescue_cp != 0 {
                    if rescue_route_choice[row_idx] == 2 {
                        rescue_exit_counts[0] += 1;
                    } else if rescue_route_choice[row_idx] == 3 {
                        rescue_exit_counts[1] += 1;
                    }
                    if shadow_reject[row_idx] != exact_reject {
                        shadow_mismatch_by_stage[1] += 1;
                    }
                } else if fallback_used[row_idx] != 0 {
                    tail_continuation_rows += 1;
                }
                let fallback_cp = fallback_entry_checkpoints[row_idx];
                if let Some(idx) = checkpoint_to_idx.get(&fallback_cp) {
                    checkpoint_exit_counts[*idx] += 1;
                    fallback_entry_checkpoint_counts[*idx] += 1;
                }
            }
        } else {
            let direct_cp = direct_checkpoints[row_idx];
            if direct_cp != 0 {
                if let Some(idx) = checkpoint_to_idx.get(&direct_cp) {
                    checkpoint_exit_counts[*idx] += 1;
                    direct_rows_by_checkpoint[*idx] += 1;
                    if shadow_reject[row_idx] != exact_reject {
                        shadow_mismatch_by_checkpoint[*idx] += 1;
                    }
                }
            } else {
                let fallback_cp = fallback_entry_checkpoints[row_idx];
                if let Some(idx) = checkpoint_to_idx.get(&fallback_cp) {
                    checkpoint_exit_counts[*idx] += 1;
                    fallback_entry_checkpoint_counts[*idx] += 1;
                }
            }
        }
    }

    if let Some(path) = &args.out_tsv {
        write_qs_prefix_output_tsv(
            path,
            &batch,
            n,
            &prefix_scores,
            &trees_used,
            &shadow_reject,
            &final_reject,
            &fallback_used,
            &route_scores,
            &block_evals,
            &exact_visits,
            &direct_checkpoints,
            &fallback_entry_checkpoints,
            &router_choice,
            &router_tau_bin,
        )?;
    }
    if let Some(path) = &args.trace_jsonl {
        write_qs_prefix_trace_jsonl(
            path,
            &batch,
            &route_meta,
            n,
            &checkpoints,
            &prefix_scores,
            &trees_used,
            &final_reject,
            &fallback_used,
            &router_choice,
            &router_tau_bin,
            &checkpoint_scores_flat,
            &checkpoint_work_evals_flat,
            &checkpoint_deltas_flat,
            &checkpoint_resolved_flat,
        )?;
    }

    let active_f = active_cnt.max(1) as f64;
    let fallback_n = fallback_cnt.max(1);
    let direct_decision_cnt = direct_reject_cnt + direct_refer_cnt;
    let stats = QsPrefixCalStats {
        variant_key: first_runtime.calibration.variant_key.clone(),
        ref_q: first_runtime.calibration.ref_q,
        rej_q: first_runtime.calibration.rej_q,
        checkpoints: checkpoints.clone(),
        n_rows: n,
        n_features: base_n_features,
        n_trees: base_n_trees,
        n_blocks: first_runtime.pack.n_blocks(),
        threads: par_threads,
        parallel_enabled,
        feature_format: batch.format_tag.clone(),
        nan_free: batch.nan_free,
        shadow_only: args.shadow_only,
        v4_mode: v4_mode,
        direct_kernel: prefix_direct_kernel_str(first_runtime.direct_kernel).to_string(),
        certifier_kind: prefix_certifier_kind_str(first_runtime.certifier_kind).to_string(),
        hot_exact_prefix_limit: first_runtime.hot_exact_prefix_limit,
        telemetry_schema_version,
        elapsed_sec: elapsed,
        rows_per_sec: n as f64 / elapsed.max(1e-9),
        rss_peak_mb: peak_rss_mb(),
        route_active_rate: active_cnt as f64 / n.max(1) as f64,
        shadow_reject_rate: shadow_reject_cnt as f64 / active_f,
        final_reject_rate: final_reject_cnt as f64 / active_f,
        direct_reject_rate: direct_reject_cnt as f64 / active_f,
        direct_refer_rate: direct_refer_cnt as f64 / active_f,
        direct_decision_rate: direct_decision_cnt as f64 / active_f,
        fallback_rate: fallback_cnt as f64 / active_f,
        avg_blocks_per_row: total_block_evals as f64 / active_f,
        avg_blocks_per_tree: total_block_evals as f64
            / ((active_cnt.max(1) * base_n_trees).max(1) as f64),
        avg_prefix_trees_used: total_prefix_trees as f64 / active_f,
        resolved_early_rate: resolved_early_trees as f64 / total_prefix_trees.max(1) as f64,
        exact_avg_visited_trees: exact_visit_sum as f64 / fallback_n as f64,
        exact_p99_visited_trees: pct_from_hist(&exact_visit_hist, 0.99, fallback_cnt),
        checkpoint_exit_counts,
        direct_rows_by_checkpoint,
        shadow_mismatch_by_checkpoint,
        fallback_entry_checkpoint_counts,
        exact_visit_hist,
        router_choice_counts,
        router_tau_bin_counts,
        anchor_exit_counts,
        rescue_route_counts,
        rescue_exit_counts,
        tail_continuation_rows,
        shadow_mismatch_by_stage,
    };

    Ok(stats)
}

fn run_qs_l2_prefix_cal(args: QsL2PrefixCalArgs) -> Result<()> {
    let stats = run_qs_l2_prefix_cal_stats(args.clone())?;
    if let Some(path) = &args.stats_json {
        let txt = serde_json::to_string_pretty(&stats)?;
        fs::write(path, txt).with_context(|| format!("write {}", path.display()))?;
    }

    println!(
        "mode=l2-qs-prefix-cal variant={} rows={} threads={} parallel={} rows/s={:.1} direct_rate={:.4} fallback_rate={:.4} avg_prefix_trees={:.1}",
        stats.variant_key,
        stats.n_rows,
        stats.threads,
        stats.parallel_enabled,
        stats.rows_per_sec,
        stats.direct_decision_rate,
        stats.fallback_rate,
        stats.avg_prefix_trees_used,
    );

    Ok(())
}

fn write_output_tsv(
    path: &PathBuf,
    batch: &FeatureBatch,
    n: usize,
    mode: InferMode,
    scores: &[f32],
    passes: &[bool],
    visits: &[i32],
) -> Result<()> {
    let mut w = BufWriter::new(
        fs::File::create(path).with_context(|| format!("create {}", path.display()))?,
    );
    let score_col = score_col_name(mode);
    let decision_col = decision_col_name(mode);
    writeln!(
        w,
        "TransactionID\tisFraud\t{}\t{}\tvisited_trees",
        score_col, decision_col
    )?;
    let pos_label = positive_label(mode);
    let neg_label = negative_label(mode);
    for i in 0..n {
        let decision = if passes[i] { pos_label } else { neg_label };
        writeln!(
            w,
            "{}\t{}\t{:.9}\t{}\t{}",
            batch.id(i),
            batch.label(i),
            scores[i],
            decision,
            visits[i]
        )?;
    }
    Ok(())
}

fn run_infer(args: InferArgs) -> Result<()> {
    let model = load_soa(&args.soa)?;
    let bounds = load_bounds(&args.bounds)?;
    let batch = load_features(&args.feat_bin)?;
    let rank_pack = if let Some(path) = &args.rank_pack {
        Some(load_rank_pack(path)?)
    } else {
        None
    };
    let raw_tree_order = if let Some(path) = &args.tree_order {
        Some(load_tree_order(path)?)
    } else {
        None
    };
    let prefix16_pack = if let Some(path) = &args.prefix16_pack {
        Some(load_prefix_pack(path)?)
    } else {
        None
    };
    let prefix32_pack = if let Some(path) = &args.prefix32_pack {
        Some(load_prefix_pack(path)?)
    } else {
        None
    };
    let approx_policy = if let Some(path) = &args.approx_policy {
        Some(load_approx_policy(path)?)
    } else {
        None
    };
    let route_meta = if let Some(path) = &args.route_meta {
        Some(load_route_meta(path)?)
    } else {
        None
    };
    let threshold = if matches!(args.mode, InferMode::L2RouteExactReordered | InferMode::L2RouteApprox) {
        args.threshold_override.unwrap_or(0.0)
    } else {
        load_threshold(&args.policy, args.threshold_override)?
    };
    let base_score = if let Some(p) = &args.model_json {
        parse_base_score(p)?
    } else {
        0.0
    };

    if model.n_features != batch.n_cols {
        bail!(
            "feature count mismatch: model={} input={}",
            model.n_features,
            batch.n_cols
        );
    }
    if model.n_trees != bounds.n_trees {
        bail!(
            "tree count mismatch: model={} bounds={}",
            model.n_trees,
            bounds.n_trees
        );
    }
    if let Some(pack) = &rank_pack {
        if pack.n_features != model.n_features {
            bail!(
                "rank pack feature mismatch: rank_pack={} model={}",
                pack.n_features,
                model.n_features
            );
        }
        if pack.node_thr_rank.len() != model.node_count {
            bail!(
                "rank pack node mismatch: rank_pack={} model={}",
                pack.node_thr_rank.len(),
                model.node_count
            );
        }
    }
    if let Some(meta) = &route_meta {
        if meta.n_rows != batch.n_rows {
            bail!(
                "route meta row mismatch: route_meta={} feat_bin={}",
                meta.n_rows,
                batch.n_rows
            );
        }
        if meta.ids.len() != batch.n_rows
            || meta.fold_id.len() != batch.n_rows
            || meta.tau_used.len() != batch.n_rows
            || meta.active.len() != batch.n_rows
            || meta.exact_positive.len() != batch.n_rows
        {
            bail!("route meta internal length mismatch");
        }
        for i in 0..batch.n_rows {
            if batch.id(i) != meta.ids[i] {
                bail!(
                    "route meta TransactionID mismatch at row {}: feat_bin={} route_meta={}",
                    i,
                    batch.id(i),
                    meta.ids[i]
                );
            }
        }
    }
    if matches!(
        args.mode,
        InferMode::RouteExactReordered
            | InferMode::MarginExactReordered
            | InferMode::RouteApprox
            | InferMode::L2RouteExactReordered
            | InferMode::L2RouteApprox
    ) && raw_tree_order.is_none()
    {
        bail!("mode {:?} requires --tree-order", args.mode);
    }
    if matches!(args.mode, InferMode::RouteApprox | InferMode::L2RouteApprox)
        && approx_policy.is_none()
    {
        bail!("mode {:?} requires --approx-policy", args.mode);
    }
    if args.fast_no_stats && args.out_tsv.is_some() {
        bail!("--fast-no-stats cannot be combined with --out-tsv");
    }
    if matches!(args.mode, InferMode::L2RouteExactReordered | InferMode::L2RouteApprox)
        && args.route_meta.is_none()
    {
        bail!("mode {:?} requires --route-meta", args.mode);
    }
    if let Some(policy) = &approx_policy {
        if !matches!(args.mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
            bail!("--approx-policy is only valid with --mode route-approx");
        }
        if policy.fallback_mode != "route-exact-reordered" {
            bail!("unsupported approx fallback_mode: {}", policy.fallback_mode);
        }
        let _ = &policy.calibration_manifest;
    }

    let effective_rank_pack = if matches!(args.mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
        if let Some(policy) = &approx_policy {
            match policy.rank_mode {
                RankMode::RankPack => {
                    if rank_pack.is_none() {
                        bail!("route-approx rank-pack policy requires --rank-pack");
                    }
                    rank_pack.as_ref()
                }
                RankMode::Float => None,
            }
        } else {
            None
        }
    } else {
        rank_pack.as_ref()
    };

    let plan = materialize_tree_plan(&bounds, raw_tree_order.as_ref(), args.max_trees)?;
    if let Some(policy) = &approx_policy {
        let mx = policy
            .k_hot
            .max(policy.used_checkpoints.iter().copied().max().unwrap_or(0));
        if mx > plan.n_trees {
            bail!(
                "approx checkpoint {} exceeds effective tree count {}",
                mx,
                plan.n_trees
            );
        }
    }

    let n = args.max_rows.unwrap_or(batch.n_rows).min(batch.n_rows);
    let thread_pool = build_thread_pool(args.threads)?;
    let prefix_pack_kind = if matches!(args.mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
        if prefix16_pack.is_some()
            && approx_policy
                .as_ref()
                .map(|p| {
                    p.used_checkpoints.first().copied().unwrap_or(usize::MAX) <= 16
                        && matches!(p.rank_mode, RankMode::Float)
                })
                .unwrap_or(false)
        {
            "prefix16"
        } else if prefix32_pack.is_some()
            && approx_policy
                .as_ref()
                .map(|p| {
                    p.used_checkpoints.first().copied().unwrap_or(usize::MAX) <= 32
                        && matches!(p.rank_mode, RankMode::Float)
                })
                .unwrap_or(false)
        {
            "prefix32"
        } else {
            ""
        }
    } else {
        ""
    };

    let stats = if args.fast_no_stats && matches!(args.mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
        let policy = approx_policy.as_ref().unwrap();
        let t0 = Instant::now();
        let (agg, parallel_enabled, used_threads) = run_kernel_fast_approx(
            &model,
            &plan,
            &batch,
            route_meta.as_ref(),
            policy,
            prefix16_pack.as_ref(),
            prefix32_pack.as_ref(),
            n,
            threshold,
            base_score,
            args.eps,
            args.bound_guard,
            args.bound_check_every.max(1),
            args.threads,
            args.chunk_rows.max(1),
            args.parallel_min_rows.max(1),
            thread_pool.as_ref(),
        )?;
        let elapsed = t0.elapsed().as_secs_f64();
        let approx_cov = (agg.approx_pass_cnt + agg.approx_ref_cnt) as f64 / n as f64;
        InferStats {
            n_rows: n,
            n_trees: plan.n_trees,
            threshold,
            base_score,
            mode: args.mode,
            threads: used_threads,
            parallel_enabled,
            rank_pack_enabled: effective_rank_pack.is_some(),
            tree_order_enabled: raw_tree_order.is_some(),
            approx_policy_enabled: approx_policy.is_some(),
            feature_format: batch.format_tag.clone(),
            nan_free: batch.nan_free,
            fast_no_stats: true,
            bound_check_every: args.bound_check_every.max(1),
            max_trees: plan.n_trees,
            eps: args.eps,
            bound_guard: args.bound_guard,
            route_pass_rate: agg.pass_cnt as f64 / n as f64,
            route_active_rate: agg.active_cnt as f64 / n as f64,
            avg_visited_trees: agg.visit_sum as f64 / n as f64,
            p50_visited_trees: pct_from_hist(&agg.visit_hist, 0.50, n),
            p90_visited_trees: pct_from_hist(&agg.visit_hist, 0.90, n),
            p99_visited_trees: pct_from_hist(&agg.visit_hist, 0.99, n),
            visit_hist: agg.visit_hist.clone(),
            full_eval_rate: agg.full_cnt as f64 / n as f64,
            approx_coverage_rate: approx_cov,
            approx_pass_rate: agg.approx_pass_cnt as f64 / n as f64,
            approx_refer_rate: agg.approx_ref_cnt as f64 / n as f64,
            approx_fallback_rate: 1.0 - approx_cov,
            approx_k_hot: policy.k_hot,
            approx_rank_mode: match policy.rank_mode {
                RankMode::RankPack => "rank-pack",
                RankMode::Float => "float",
            }
            .to_string(),
            approx_checkpoint_exit_counts: agg.checkpoint_counts,
            prefix_pack_kind: prefix_pack_kind.to_string(),
            elapsed_sec: elapsed,
            rows_per_sec: n as f64 / elapsed.max(1e-9),
            rss_peak_mb: peak_rss_mb(),
        }
    } else {
        let t0 = Instant::now();
        let (
            scores,
            passes,
            visits,
            parallel_enabled,
            used_threads,
            approx_pass_cnt,
            approx_ref_cnt,
            checkpoint_counts,
        ) = run_kernel(
            &model,
            &plan,
            &batch,
            route_meta.as_ref(),
            effective_rank_pack,
            approx_policy.as_ref(),
            prefix16_pack.as_ref(),
            prefix32_pack.as_ref(),
            n,
            threshold,
            base_score,
            args.mode,
            args.eps,
            args.bound_guard,
            args.bound_check_every.max(1),
            args.threads,
            args.chunk_rows.max(1),
            args.parallel_min_rows.max(1),
            thread_pool.as_ref(),
        )?;
        let elapsed = t0.elapsed().as_secs_f64();

        if let Some(path) = &args.out_tsv {
            write_output_tsv(path, &batch, n, args.mode, &scores, &passes, &visits)?;
        }

        let pass_cnt = passes.iter().filter(|&&x| x).count();
        let full_cnt = visits
            .iter()
            .filter(|&&v| v as usize == plan.n_trees)
            .count();
        let avg_visit = visits.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let approx_cov = (approx_pass_cnt + approx_ref_cnt) as f64 / n as f64;
        InferStats {
            n_rows: n,
            n_trees: plan.n_trees,
            threshold,
            base_score,
            mode: args.mode,
            threads: used_threads,
            parallel_enabled,
            rank_pack_enabled: effective_rank_pack.is_some(),
            tree_order_enabled: raw_tree_order.is_some(),
            approx_policy_enabled: approx_policy.is_some(),
            feature_format: batch.format_tag.clone(),
            nan_free: batch.nan_free,
            fast_no_stats: false,
            bound_check_every: args.bound_check_every.max(1),
            max_trees: plan.n_trees,
            eps: args.eps,
            bound_guard: args.bound_guard,
            route_pass_rate: pass_cnt as f64 / n as f64,
            route_active_rate: route_meta
                .as_ref()
                .map(|m| m.active[..n].iter().filter(|&&x| x != 0).count() as f64 / n as f64)
                .unwrap_or(1.0),
            avg_visited_trees: avg_visit,
            p50_visited_trees: pct(&visits, 0.50),
            p90_visited_trees: pct(&visits, 0.90),
            p99_visited_trees: pct(&visits, 0.99),
            visit_hist: histogram_from_visits(&visits, plan.n_trees),
            full_eval_rate: full_cnt as f64 / n as f64,
            approx_coverage_rate: approx_cov,
            approx_pass_rate: approx_pass_cnt as f64 / n as f64,
            approx_refer_rate: approx_ref_cnt as f64 / n as f64,
            approx_fallback_rate: 1.0 - approx_cov,
            approx_k_hot: approx_policy.as_ref().map(|p| p.k_hot).unwrap_or(0),
            approx_rank_mode: approx_policy
                .as_ref()
                .map(|p| match p.rank_mode {
                    RankMode::RankPack => "rank-pack",
                    RankMode::Float => "float",
                })
                .unwrap_or("")
                .to_string(),
            approx_checkpoint_exit_counts: checkpoint_counts,
            prefix_pack_kind: prefix_pack_kind.to_string(),
            elapsed_sec: elapsed,
            rows_per_sec: n as f64 / elapsed.max(1e-9),
            rss_peak_mb: peak_rss_mb(),
        }
    };

    if let Some(path) = &args.stats_json {
        let txt = serde_json::to_string_pretty(&stats)?;
        fs::write(path, txt).with_context(|| format!("write {}", path.display()))?;
    }

    println!(
        "mode={:?} rows={} threads={} parallel={} rows/s={:.1} avg_visited={:.1} full_eval_rate={:.6} format={} rank_pack={} fast_no_stats={} prefix_pack={}",
        stats.mode,
        stats.n_rows,
        stats.threads,
        stats.parallel_enabled,
        stats.rows_per_sec,
        stats.avg_visited_trees,
        stats.full_eval_rate,
        stats.feature_format,
        stats.rank_pack_enabled,
        stats.fast_no_stats,
        stats.prefix_pack_kind
    );

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Infer(args) => run_infer(args),
        Command::DispatchL2Exact(args) => run_dispatch_l2_exact(args),
        Command::QsL2Exact(args) => run_qs_l2_exact(args),
        Command::QsL2Fast(args) => run_qs_l2_fast(args),
        Command::QsL2PrefixCal(args) => run_qs_l2_prefix_cal(args),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StandaloneL2Output {
    pub score: f32,
    pub reject: bool,
    pub visited_trees: i32,
    pub trees_used: usize,
    pub fallback_used: bool,
}

pub struct StandaloneL2Scratch {
    ranks: Vec<u8>,
    missing: Vec<u8>,
    hot_buf: Option<HotFeatureBuf>,
    hot_cache: Option<LazyHotFeatureCache>,
}

impl StandaloneL2Scratch {
    fn new(runtime: &LoadedPrefixRuntime) -> Self {
        let n_features = runtime.pack.n_features();
        Self {
            ranks: vec![0u8; n_features],
            missing: vec![0u8; n_features],
            hot_buf: runtime
                .compiled_hot_pack
                .as_ref()
                .map(|pack| HotFeatureBuf::new(pack.n_hot_features)),
            hot_cache: runtime
                .hot_pack
                .as_ref()
                .map(|pack| LazyHotFeatureCache::new(pack.n_hot_features)),
        }
    }
}

pub struct StandaloneL2Runtime {
    runtime: LoadedPrefixRuntime,
    model: SoaModel,
    batch: FeatureBatch,
}

#[derive(Debug, Clone, Copy)]
pub struct StandaloneQsPrefixCalBenchConfig {
    pub threads: usize,
    pub chunk_rows: usize,
    pub parallel_min_rows: usize,
    pub max_rows: Option<usize>,
}

impl Default for StandaloneQsPrefixCalBenchConfig {
    fn default() -> Self {
        Self {
            threads: 1,
            chunk_rows: 128,
            parallel_min_rows: 4096,
            max_rows: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StandaloneQsPrefixCalBenchStats {
    pub n_rows: usize,
    pub rows_per_sec: f64,
    pub route_active_rows: usize,
    pub deny_rows: usize,
    pub manual_review_rows: usize,
    pub direct_decision_rows: usize,
    pub fallback_rows: usize,
}

impl StandaloneL2Runtime {
    pub fn load(bundle_dir: &Path) -> Result<Self> {
        Self::load_with_feat_bin_override(bundle_dir, None)
    }

    pub fn load_with_feat_bin_override(
        bundle_dir: &Path,
        feat_bin_override: Option<&Path>,
    ) -> Result<Self> {
        let bundle_manifest = find_l2_bundle_manifest(bundle_dir)
            .with_context(|| format!("find L2 prefix-cal bundle manifest under {}", bundle_dir.display()))?;
        let txt = fs::read_to_string(&bundle_manifest)
            .with_context(|| format!("read {}", bundle_manifest.display()))?;
        let manifest: PrefixCalBundleManifest = serde_json::from_str(&txt)
            .with_context(|| format!("parse {}", bundle_manifest.display()))?;
        if manifest.format != "L2PrefixCalBundleV1" {
            bail!(
                "unsupported prefix-cal bundle format in {}: {}",
                bundle_manifest.display(),
                manifest.format
            );
        }
        let args = QsL2PrefixCalArgs {
            bundle_manifest: Some(bundle_manifest.clone()),
            qs_pack: manifest
                .pack_path
                .as_ref()
                .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
            calibration_json: manifest
                .calibration_json
                .as_ref()
                .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
            variant_key: manifest
                .selected_exact_variant
                .clone()
                .or_else(|| manifest.selected_variant.clone()),
            feat_bin: manifest
                .feat_bin
                .as_ref()
                .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
            route_meta: manifest
                .route_meta
                .as_ref()
                .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
            soa: manifest
                .soa_bin
                .as_ref()
                .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
            bounds: manifest
                .bounds_bin
                .as_ref()
                .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
            tree_order: manifest
                .tree_order_bin
                .as_ref()
                .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
            model_json: manifest
                .model_json
                .as_ref()
                .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
            direct_kernel: manifest.direct_kernel.clone(),
            certifier_kind: manifest.certifier_kind.clone(),
            certifier_json: manifest
                .certifier_json
                .as_ref()
                .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
            out_tsv: None,
            stats_json: None,
            trace_jsonl: None,
            shadow_only: false,
            threads: 0,
            chunk_rows: 128,
            parallel_min_rows: 4096,
            max_rows: None,
        };
        let mut resolved = resolve_prefix_cal_bundle(&args)?;
        resolved.qs_pack = prefer_bundle_root_existing(bundle_dir, &resolved.qs_pack);
        resolved.calibration_json = prefer_bundle_root_existing(bundle_dir, &resolved.calibration_json);
        resolved.feat_bin = prefer_bundle_root_existing(bundle_dir, &resolved.feat_bin);
        resolved.route_meta = prefer_bundle_root_existing(bundle_dir, &resolved.route_meta);
        resolved.soa = prefer_bundle_root_existing(bundle_dir, &resolved.soa);
        resolved.bounds = prefer_bundle_root_existing(bundle_dir, &resolved.bounds);
        resolved.tree_order = prefer_bundle_root_existing(bundle_dir, &resolved.tree_order);
        resolved.model_json = prefer_bundle_root_existing(bundle_dir, &resolved.model_json);
        resolved.certifier_json = resolved
            .certifier_json
            .take()
            .map(|p| prefer_bundle_root_existing(bundle_dir, &p));
        resolved.atlas_bin = resolved
            .atlas_bin
            .take()
            .map(|p| prefer_bundle_root_existing(bundle_dir, &p));
        resolved.hot_exact_prefix_pack = resolved
            .hot_exact_prefix_pack
            .take()
            .map(|p| prefer_bundle_root_existing(bundle_dir, &p));
        if let Some(path) = feat_bin_override {
            resolved.feat_bin = path.to_path_buf();
        }
        let batch = load_features(&resolved.feat_bin)?;
        let model = load_soa(&resolved.soa)?;
        let bounds = load_bounds(&resolved.bounds)?;
        let runtime = load_prefix_runtime(&resolved, &model, &bounds)?;
        Ok(Self {
            runtime,
            model,
            batch,
        })
    }

    pub fn new_scratch(&self) -> StandaloneL2Scratch {
        StandaloneL2Scratch::new(&self.runtime)
    }

    pub fn l2_dim(&self) -> usize {
        self.runtime.pack.n_features()
    }

    pub fn feat_rows(&self) -> usize {
        self.batch.n_rows
    }

    pub fn predict_l2_row_nomiss_with_scratch(
        &self,
        row: &[f32],
        tau: f32,
        fold_id: i32,
        scratch: &mut StandaloneL2Scratch,
    ) -> Result<StandaloneL2Output> {
        let shadow_row = run_prefix_shadow_row_single_route_compiled(
            &self.runtime,
            row,
            tau,
            fold_id,
            true,
            &mut scratch.ranks,
            &mut scratch.missing,
            scratch.hot_buf.as_mut(),
            scratch.hot_cache.as_mut(),
        )?;

        if !shadow_row.fallback_used {
            return Ok(StandaloneL2Output {
                score: shadow_row.shadow_route_score,
                reject: shadow_row.shadow_reject,
                visited_trees: 0,
                trees_used: shadow_row.trees_used,
                fallback_used: false,
            });
        }

        let (score, reject, visited_trees, _) = run_exact_continuation(
            &self.runtime,
            &self.model,
            row,
            tau,
            &shadow_row,
            true,
            &mut scratch.ranks,
            &mut scratch.missing,
        )?;
        Ok(StandaloneL2Output {
            score,
            reject,
            visited_trees,
            trees_used: shadow_row.trees_used,
            fallback_used: true,
        })
    }

    pub fn predict_l2_row_by_index_with_scratch(
        &self,
        row_idx: usize,
        tau: f32,
        fold_id: i32,
        scratch: &mut StandaloneL2Scratch,
    ) -> Result<StandaloneL2Output> {
        ensure!(
            row_idx < self.batch.n_rows,
            "standalone feat row out of bounds: row_idx={} n_rows={}",
            row_idx,
            self.batch.n_rows
        );
        self.predict_l2_row_nomiss_with_scratch(self.batch.row(row_idx), tau, fold_id, scratch)
    }
}

pub fn benchmark_qs_l2_prefix_cal(
    bundle_dir: &Path,
    cfg: StandaloneQsPrefixCalBenchConfig,
) -> Result<StandaloneQsPrefixCalBenchStats> {
    let bundle_manifest = find_l2_bundle_manifest(bundle_dir)
        .with_context(|| format!("find L2 prefix-cal bundle manifest under {}", bundle_dir.display()))?;
    let txt = fs::read_to_string(&bundle_manifest)
        .with_context(|| format!("read {}", bundle_manifest.display()))?;
    let manifest: PrefixCalBundleManifest = serde_json::from_str(&txt)
        .with_context(|| format!("parse {}", bundle_manifest.display()))?;

    let args = QsL2PrefixCalArgs {
        bundle_manifest: Some(bundle_manifest.clone()),
        qs_pack: manifest
            .pack_path
            .as_ref()
            .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
        calibration_json: manifest
            .calibration_json
            .as_ref()
            .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
        variant_key: manifest
            .selected_exact_variant
            .clone()
            .or_else(|| manifest.selected_variant.clone()),
        feat_bin: manifest
            .feat_bin
            .as_ref()
            .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
        route_meta: manifest
            .route_meta
            .as_ref()
            .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
        soa: manifest
            .soa_bin
            .as_ref()
            .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
        bounds: manifest
            .bounds_bin
            .as_ref()
            .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
        tree_order: manifest
            .tree_order_bin
            .as_ref()
            .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
        model_json: manifest
            .model_json
            .as_ref()
            .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
        direct_kernel: manifest.direct_kernel.clone(),
        certifier_kind: manifest.certifier_kind.clone(),
        certifier_json: manifest
            .certifier_json
            .as_ref()
            .map(|raw| resolve_bundle_path_root_first(bundle_dir, &bundle_manifest, raw)),
        out_tsv: None,
        stats_json: None,
        trace_jsonl: None,
        shadow_only: false,
        threads: cfg.threads,
        chunk_rows: cfg.chunk_rows,
        parallel_min_rows: cfg.parallel_min_rows,
        max_rows: cfg.max_rows,
    };
    let prev_cwd = std::env::current_dir().context("read current working directory")?;
    std::env::set_current_dir(bundle_dir)
        .with_context(|| format!("set current dir to {}", bundle_dir.display()))?;
    let stats = run_qs_l2_prefix_cal_stats(args);
    std::env::set_current_dir(&prev_cwd)
        .with_context(|| format!("restore current dir to {}", prev_cwd.display()))?;
    let stats = stats?;
    let active_rows = ((stats.route_active_rate * stats.n_rows as f64).round() as usize).min(stats.n_rows);
    let deny_rows = ((stats.final_reject_rate * active_rows as f64).round() as usize).min(active_rows);
    let fallback_rows = ((stats.fallback_rate * active_rows as f64).round() as usize).min(active_rows);
    let direct_decision_rows =
        ((stats.direct_decision_rate * active_rows as f64).round() as usize).min(active_rows);
    Ok(StandaloneQsPrefixCalBenchStats {
        n_rows: stats.n_rows,
        rows_per_sec: stats.rows_per_sec,
        route_active_rows: active_rows,
        deny_rows,
        manual_review_rows: active_rows.saturating_sub(deny_rows),
        direct_decision_rows,
        fallback_rows,
    })
}

fn find_l2_bundle_manifest(bundle_dir: &Path) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    for entry in WalkDir::new(bundle_dir).follow_links(true) {
        let entry = match entry {
            Ok(v) => v,
            Err(_) => continue,
        };
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.file_name() != "manifest.json" {
            continue;
        }
        let path = entry.path();
        let Ok(txt) = fs::read_to_string(path) else {
            continue;
        };
        if txt.contains("\"format\":\"L2PrefixCalBundleV1\"")
            || txt.contains("\"format\": \"L2PrefixCalBundleV1\"")
        {
            candidates.push(path.to_path_buf());
        }
    }
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .context("no L2PrefixCalBundleV1 manifest found")
}

fn resolve_bundle_path_root_first(bundle_dir: &Path, manifest_path: &Path, raw: &str) -> PathBuf {
    let raw_path = PathBuf::from(raw);
    if raw_path.is_absolute() {
        return raw_path;
    }
    let root_candidate = bundle_dir.join(&raw_path);
    if root_candidate.exists() {
        return root_candidate;
    }
    manifest_path
        .parent()
        .map(|p| p.join(raw_path.clone()))
        .unwrap_or(raw_path)
}

fn prefer_bundle_root_existing(bundle_dir: &Path, current: &Path) -> PathBuf {
    if current.exists() || current.is_absolute() {
        return current.to_path_buf();
    }
    let mut components = current.components();
    let trimmed = if matches!(components.next(), Some(std::path::Component::Normal(_))) {
        current
    } else {
        current
    };
    let candidate = bundle_dir.join(trimmed);
    if candidate.exists() {
        candidate
    } else {
        current.to_path_buf()
    }
}
