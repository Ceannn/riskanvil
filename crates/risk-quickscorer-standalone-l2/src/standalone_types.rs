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

const LATE_IF_TREE_NODE_LEAF: u8 = 1 << 7;
const LATE_IF_TREE_NODE_FIDX_MASK: u8 = LATE_IF_TREE_NODE_LEAF - 1;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct LateIfTreeNode {
    thr: f32,
    leaf: f32,
    left: u8,
    right: u8,
    fidx_flags: u8,
    _pad: u8,
}

impl LateIfTreeNode {
    #[inline(always)]
    fn is_leaf(self) -> bool {
        (self.fidx_flags & LATE_IF_TREE_NODE_LEAF) != 0
    }

    #[inline(always)]
    fn fidx(self) -> usize {
        (self.fidx_flags & LATE_IF_TREE_NODE_FIDX_MASK) as usize
    }
}

#[derive(Debug)]
struct LateIfTreeProgram {
    start_tree_idx: usize,
    n_trees: usize,
    tree_node_offs: Vec<u32>,
    nodes: Vec<LateIfTreeNode>,
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
    late_iftree_program: Option<LateIfTreeProgram>,
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
