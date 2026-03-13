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
    hot_stage_plan: Option<HotStagePlan>,
    certify_plan: Option<LoadedPrefixCertifyPlan>,
    late_segment_runtime: Option<LateSegmentRuntime>,
    prefix_v2: Option<LoadedPrefixV2Runtime>,
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
    late_feature_ids: Vec<u16>,
    late_feature_offsets: Vec<u32>,
    late_cut_starts: Vec<u32>,
    late_cut_ends: Vec<u32>,
}

#[derive(Debug, Default, Clone)]
struct HotStagePlan {
    checkpoints: Vec<usize>,
    feature_offsets: Vec<u32>,
    hot_feature_ids: Vec<u16>,
}

#[derive(Debug, Clone)]
struct LoadedPrefixCertifyCheckpoint {
    table: LoadedPrefixCheckpoint,
    atlas: LoadedPrefixAtlasCheckpoint,
}

#[derive(Debug, Clone)]
struct LoadedPrefixCertifyPlan {
    tau_edges: Vec<f32>,
    checkpoints: Vec<LoadedPrefixCertifyCheckpoint>,
}

#[derive(Debug)]
struct LoadedLateSegment {
    checkpoint: usize,
    global_cp_idx: usize,
    start_tree_idx: usize,
    local_to_global_fid: Vec<u16>,
    pack: qs_exact::QsPack,
}

#[derive(Debug, Default)]
struct LateSegmentRuntime {
    segments: Vec<LoadedLateSegment>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LateSegmentManifest {
    format: String,
    hot_tree_count: usize,
    segments: Vec<LateSegmentMeta>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LateSegmentMeta {
    checkpoint: usize,
    global_cp_idx: usize,
    start_tree_idx: usize,
    pack_file: String,
    local_to_global_fid: Vec<u16>,
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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
    used_tau_ref_dense: Vec<f32>,
    #[serde(default)]
    used_tau_positive_dense: Vec<f32>,
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

fn le_u16(buf: &[u8], off: &mut usize) -> Result<u16> {
    if *off + 2 > buf.len() {
        bail!("u16 out of bounds");
    }
    let v = u16::from_le_bytes([buf[*off], buf[*off + 1]]);
    *off += 2;
    Ok(v)
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
    let mmap = mmap_readonly(path)?;
    let buf = &mmap[..];
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
    let mmap = mmap_readonly(path)?;
    let buf = &mmap[..];
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
    let mmap = mmap_readonly(path)?;
    let buf = &mmap[..];
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

fn default_compiled_hot_pack_path(path: &PathBuf) -> PathBuf {
    let mut out = path.clone();
    let file_name = path
        .file_name()
        .and_then(|x| x.to_str())
        .map(|x| format!("{}.compiled.bin", x))
        .unwrap_or_else(|| "hot_prefix.compiled.bin".to_string());
    out.set_file_name(file_name);
    out
}

fn default_qs_pack_v2_path(path: &PathBuf) -> PathBuf {
    let mut out = path.clone();
    let file_name = path
        .file_name()
        .and_then(|x| x.to_str())
        .map(|x| format!("{}.v2.bin", x))
        .unwrap_or_else(|| "qs.pack.v2.bin".to_string());
    out.set_file_name(file_name);
    out
}

fn default_hot_checkpoint_layout_path(path: &PathBuf) -> PathBuf {
    let mut out = path.clone();
    let file_name = path
        .file_name()
        .and_then(|x| x.to_str())
        .map(|x| format!("{}.hot_layout.bin", x))
        .unwrap_or_else(|| "hot_prefix.hot_layout.bin".to_string());
    out.set_file_name(file_name);
    out
}

fn default_late_segment_manifest_path(path: &PathBuf) -> PathBuf {
    let mut out = path.clone();
    let file_name = path
        .file_name()
        .and_then(|x| x.to_str())
        .map(|x| format!("{}.late_segments.json", x))
        .unwrap_or_else(|| "late_segments.json".to_string());
    out.set_file_name(file_name);
    out
}

fn default_late_segment_pack_path(path: &PathBuf, checkpoint: usize) -> PathBuf {
    let mut out = path.clone();
    let file_name = path
        .file_name()
        .and_then(|x| x.to_str())
        .map(|x| format!("{}.late_cp_{}.v2.bin", x, checkpoint))
        .unwrap_or_else(|| format!("late_cp_{}.v2.bin", checkpoint));
    out.set_file_name(file_name);
    out
}

fn default_certify_plan_path(path: &PathBuf) -> PathBuf {
    let mut out = path.clone();
    let file_name = path
        .file_name()
        .and_then(|x| x.to_str())
        .map(|x| format!("{}.certify.bin", x))
        .unwrap_or_else(|| "l2_certify.bin".to_string());
    out.set_file_name(file_name);
    out
}

fn save_compiled_hot_prefix_pack(path: &PathBuf, pack: &HotCompiledPrefixPack) -> Result<()> {
    let mut out = Vec::with_capacity(
        MAGIC_PRFC.len()
            + 12
            + (pack.n_hot_features * 4)
            + (pack.n_trees * 2)
            + (pack.nodes.len() * 16),
    );
    out.extend_from_slice(MAGIC_PRFC);
    out.extend_from_slice(&(pack.n_hot_features as u32).to_le_bytes());
    out.extend_from_slice(&(pack.n_trees as u32).to_le_bytes());
    out.extend_from_slice(&(pack.nodes.len() as u32).to_le_bytes());
    for &v in &pack.hot_global_fidx {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &pack.tree_roots {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for node in &pack.nodes {
        out.extend_from_slice(&node.thr.to_le_bytes());
        out.extend_from_slice(&node.leaf.to_le_bytes());
        out.extend_from_slice(&node.left.to_le_bytes());
        out.extend_from_slice(&node.right.to_le_bytes());
        out.extend_from_slice(&node.missing.to_le_bytes());
        out.extend_from_slice(&node.hot_fidx_flags.to_le_bytes());
    }
    fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn load_compiled_hot_prefix_pack(path: &PathBuf) -> Result<HotCompiledPrefixPack> {
    let mmap = mmap_readonly(path)?;
    let buf = &mmap[..];
    if buf.len() < MAGIC_PRFC.len() + 12 {
        bail!("compiled hot prefix pack too short");
    }
    if &buf[0..MAGIC_PRFC.len()] != MAGIC_PRFC {
        bail!("invalid compiled hot prefix pack magic");
    }
    let mut off = MAGIC_PRFC.len();
    let n_hot_features = le_u32(buf, &mut off)? as usize;
    let n_trees = le_u32(buf, &mut off)? as usize;
    let n_nodes = le_u32(buf, &mut off)? as usize;

    let mut hot_global_fidx = Vec::with_capacity(n_hot_features);
    for _ in 0..n_hot_features {
        hot_global_fidx.push(le_u32(buf, &mut off)?);
    }
    let mut tree_roots = Vec::with_capacity(n_trees);
    for _ in 0..n_trees {
        tree_roots.push(le_u16(buf, &mut off)?);
    }
    let mut nodes = Vec::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        nodes.push(HotCompiledPrefixNode {
            thr: le_f32(buf, &mut off)?,
            leaf: le_f32(buf, &mut off)?,
            left: le_u16(buf, &mut off)?,
            right: le_u16(buf, &mut off)?,
            missing: le_u16(buf, &mut off)?,
            hot_fidx_flags: le_u16(buf, &mut off)?,
        });
    }
    Ok(HotCompiledPrefixPack {
        n_hot_features,
        n_trees,
        hot_global_fidx,
        tree_roots,
        nodes,
    })
}

fn save_hot_checkpoint_layout(path: &PathBuf, layout: &HotCheckpointLayout) -> Result<()> {
    let mut out = Vec::with_capacity(
        MAGIC_HCL.len()
            + 24
            + (layout.hot_values.len() * 8)
            + (layout.hot_indices.len() * 8)
            + (layout.late_values.len() * 8)
            + (layout.late_indices.len() * 8)
            + (layout.late_feature_ids.len() * 2)
            + (layout.late_feature_offsets.len() * 4),
    );
    out.extend_from_slice(MAGIC_HCL);
    out.extend_from_slice(&(layout.hot_values.len() as u32).to_le_bytes());
    out.extend_from_slice(&(layout.hot_indices.len() as u32).to_le_bytes());
    out.extend_from_slice(&(layout.late_values.len() as u32).to_le_bytes());
    out.extend_from_slice(&(layout.late_indices.len() as u32).to_le_bytes());
    out.extend_from_slice(&(layout.late_feature_ids.len() as u32).to_le_bytes());
    out.extend_from_slice(&(layout.late_feature_offsets.len() as u32).to_le_bytes());
    for &v in &layout.hot_values {
        out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    for &v in &layout.hot_indices {
        out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    for &v in &layout.late_values {
        out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    for &v in &layout.late_indices {
        out.extend_from_slice(&(v as u32).to_le_bytes());
    }
    for &v in &layout.late_feature_ids {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &layout.late_feature_offsets {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn load_hot_checkpoint_layout(path: &PathBuf) -> Result<HotCheckpointLayout> {
    let mmap = mmap_readonly(path)?;
    let buf = &mmap[..];
    if buf.len() < MAGIC_HCL.len() + 24 {
        bail!("hot checkpoint layout too short");
    }
    if &buf[0..MAGIC_HCL.len()] != MAGIC_HCL {
        bail!("invalid hot checkpoint layout magic");
    }
    let mut off = MAGIC_HCL.len();
    let hot_values_len = le_u32(buf, &mut off)? as usize;
    let hot_indices_len = le_u32(buf, &mut off)? as usize;
    let late_values_len = le_u32(buf, &mut off)? as usize;
    let late_indices_len = le_u32(buf, &mut off)? as usize;
    let late_feature_ids_len = le_u32(buf, &mut off)? as usize;
    let late_feature_offsets_len = le_u32(buf, &mut off)? as usize;

    let mut hot_values = Vec::with_capacity(hot_values_len);
    for _ in 0..hot_values_len {
        hot_values.push(le_u32(buf, &mut off)? as usize);
    }
    let mut hot_indices = Vec::with_capacity(hot_indices_len);
    for _ in 0..hot_indices_len {
        hot_indices.push(le_u32(buf, &mut off)? as usize);
    }
    let mut late_values = Vec::with_capacity(late_values_len);
    for _ in 0..late_values_len {
        late_values.push(le_u32(buf, &mut off)? as usize);
    }
    let mut late_indices = Vec::with_capacity(late_indices_len);
    for _ in 0..late_indices_len {
        late_indices.push(le_u32(buf, &mut off)? as usize);
    }
    let mut late_feature_ids = Vec::with_capacity(late_feature_ids_len);
    for _ in 0..late_feature_ids_len {
        late_feature_ids.push(le_u16(buf, &mut off)?);
    }
    let mut late_feature_offsets = Vec::with_capacity(late_feature_offsets_len);
    for _ in 0..late_feature_offsets_len {
        late_feature_offsets.push(le_u32(buf, &mut off)?);
    }
    Ok(HotCheckpointLayout {
        hot_values,
        hot_indices,
        late_values,
        late_indices,
        late_feature_ids,
        late_feature_offsets,
        late_cut_starts: Vec::new(),
        late_cut_ends: Vec::new(),
    })
}

fn save_prefix_certify_plan(path: &PathBuf, plan: &LoadedPrefixCertifyPlan) -> Result<()> {
    let mut out = Vec::new();
    out.extend_from_slice(b"L2CERTV1");
    out.extend_from_slice(&(plan.tau_edges.len() as u32).to_le_bytes());
    for &v in &plan.tau_edges {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&(plan.checkpoints.len() as u32).to_le_bytes());
    for cp in &plan.checkpoints {
        let table = &cp.table;
        let atlas = &cp.atlas;
        out.extend_from_slice(&(table.checkpoint as u32).to_le_bytes());
        out.extend_from_slice(&(table.gap_edges.len() as u32).to_le_bytes());
        out.extend_from_slice(&(table.global_ref_hi.len() as u32).to_le_bytes());
        out.extend_from_slice(&(table.global_rej_lo.len() as u32).to_le_bytes());
        out.extend_from_slice(&(table.tau_edges.len() as u32).to_le_bytes());
        out.extend_from_slice(&(table.fold_ids.len() as u32).to_le_bytes());
        out.extend_from_slice(&table.fold_lut_base.to_le_bytes());
        out.extend_from_slice(&(table.fold_slot_lut.len() as u32).to_le_bytes());
        out.extend_from_slice(&(table.fold_ref_hi.len() as u32).to_le_bytes());
        out.extend_from_slice(&(table.fold_rej_lo.len() as u32).to_le_bytes());
        out.extend_from_slice(&(table.tau_bin_count as u32).to_le_bytes());
        out.extend_from_slice(&(table.tau_fold_mask.len() as u32).to_le_bytes());
        out.extend_from_slice(&(table.tau_ref_hi.len() as u32).to_le_bytes());
        out.extend_from_slice(&(table.tau_rej_lo.len() as u32).to_le_bytes());
        for &v in &table.gap_edges {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &table.global_ref_hi {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &table.global_rej_lo {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &table.tau_edges {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &table.fold_ids {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &table.fold_slot_lut {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &table.fold_ref_hi {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &table.fold_rej_lo {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&table.tau_fold_mask);
        for &v in &table.tau_ref_hi {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &table.tau_rej_lo {
            out.extend_from_slice(&v.to_le_bytes());
        }

        out.extend_from_slice(&(atlas.checkpoint as u32).to_le_bytes());
        out.push(if atlas.shared_gap_edges { 1 } else { 0 });
        out.extend_from_slice(&(atlas.gap_edges.len() as u32).to_le_bytes());
        out.extend_from_slice(&(atlas.cluster_count as u32).to_le_bytes());
        out.extend_from_slice(&(atlas.tau_bin_count as u32).to_le_bytes());
        for &v in &atlas.feature_mean {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &atlas.feature_inv_std {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &atlas.gap_edges {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for vec in [
            &atlas.centroid_w0,
            &atlas.centroid_w1,
            &atlas.centroid_w2,
            &atlas.centroid_w3,
            &atlas.centroid_bias,
        ] {
            out.extend_from_slice(&(vec.len() as u32).to_le_bytes());
            for &v in vec.iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out.extend_from_slice(&(atlas.decision_grid.len() as u32).to_le_bytes());
        out.extend_from_slice(&atlas.decision_grid);
    }
    fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn load_prefix_certify_plan(path: &PathBuf) -> Result<LoadedPrefixCertifyPlan> {
    let mmap = mmap_readonly(path)?;
    let buf = &mmap[..];
    if buf.len() < 12 {
        bail!("prefix certify plan too short");
    }
    let mut off = 0usize;
    let magic = le_bytes(buf, &mut off, 8)?;
    if magic != b"L2CERTV1" {
        bail!("invalid prefix certify plan magic");
    }
    let tau_edges_len = le_u32(buf, &mut off)? as usize;
    let tau_edges = le_f32_vec(buf, &mut off, tau_edges_len)?;
    let checkpoint_len = le_u32(buf, &mut off)? as usize;
    let mut checkpoints = Vec::with_capacity(checkpoint_len);
    for _ in 0..checkpoint_len {
        let checkpoint = le_u32(buf, &mut off)? as usize;
        let gap_edges_len = le_u32(buf, &mut off)? as usize;
        let global_ref_hi_len = le_u32(buf, &mut off)? as usize;
        let global_rej_lo_len = le_u32(buf, &mut off)? as usize;
        let tau_edges_len = le_u32(buf, &mut off)? as usize;
        let fold_ids_len = le_u32(buf, &mut off)? as usize;
        let fold_lut_base = le_i32(buf, &mut off)?;
        let fold_slot_lut_len = le_u32(buf, &mut off)? as usize;
        let fold_ref_hi_len = le_u32(buf, &mut off)? as usize;
        let fold_rej_lo_len = le_u32(buf, &mut off)? as usize;
        let tau_bin_count = le_u32(buf, &mut off)? as usize;
        let tau_fold_mask_len = le_u32(buf, &mut off)? as usize;
        let tau_ref_hi_len = le_u32(buf, &mut off)? as usize;
        let tau_rej_lo_len = le_u32(buf, &mut off)? as usize;
        let gap_edges = le_f32_vec(buf, &mut off, gap_edges_len)?;
        let global_ref_hi = le_f32_vec(buf, &mut off, global_ref_hi_len)?;
        let global_rej_lo = le_f32_vec(buf, &mut off, global_rej_lo_len)?;
        let tau_edges_cp = le_f32_vec(buf, &mut off, tau_edges_len)?;
        let mut fold_ids = Vec::with_capacity(fold_ids_len);
        for _ in 0..fold_ids_len {
            fold_ids.push(le_i32(buf, &mut off)?);
        }
        let mut fold_slot_lut = Vec::with_capacity(fold_slot_lut_len);
        for _ in 0..fold_slot_lut_len {
            fold_slot_lut.push(le_u16(buf, &mut off)? as i16);
        }
        let fold_ref_hi = le_f32_vec(buf, &mut off, fold_ref_hi_len)?;
        let fold_rej_lo = le_f32_vec(buf, &mut off, fold_rej_lo_len)?;
        let tau_fold_mask = le_bytes(buf, &mut off, tau_fold_mask_len)?.to_vec();
        let tau_ref_hi = le_f32_vec(buf, &mut off, tau_ref_hi_len)?;
        let tau_rej_lo = le_f32_vec(buf, &mut off, tau_rej_lo_len)?;
        let table = LoadedPrefixCheckpoint {
            checkpoint,
            gap_edges,
            global_ref_hi,
            global_rej_lo,
            tau_edges: tau_edges_cp,
            fold_ids,
            fold_lut_base,
            fold_slot_lut,
            fold_ref_hi,
            fold_rej_lo,
            tau_bin_count,
            tau_fold_mask,
            tau_ref_hi,
            tau_rej_lo,
        };

        let atlas_checkpoint = le_u32(buf, &mut off)? as usize;
        let shared_gap_edges = le_bytes(buf, &mut off, 1)?[0] != 0;
        let atlas_gap_edges_len = le_u32(buf, &mut off)? as usize;
        let cluster_count = le_u32(buf, &mut off)? as usize;
        let atlas_tau_bin_count = le_u32(buf, &mut off)? as usize;
        let feature_mean_v = le_f32_vec(buf, &mut off, 4)?;
        let feature_inv_std_v = le_f32_vec(buf, &mut off, 4)?;
        let atlas_gap_edges = le_f32_vec(buf, &mut off, atlas_gap_edges_len)?;
        let centroid_w0_len = le_u32(buf, &mut off)? as usize;
        let centroid_w0 = le_f32_vec(buf, &mut off, centroid_w0_len)?;
        let centroid_w1_len = le_u32(buf, &mut off)? as usize;
        let centroid_w1 = le_f32_vec(buf, &mut off, centroid_w1_len)?;
        let centroid_w2_len = le_u32(buf, &mut off)? as usize;
        let centroid_w2 = le_f32_vec(buf, &mut off, centroid_w2_len)?;
        let centroid_w3_len = le_u32(buf, &mut off)? as usize;
        let centroid_w3 = le_f32_vec(buf, &mut off, centroid_w3_len)?;
        let centroid_bias_len = le_u32(buf, &mut off)? as usize;
        let centroid_bias = le_f32_vec(buf, &mut off, centroid_bias_len)?;
        let centroids16 = if cluster_count == 16
            && centroid_w0.len() == 16
            && centroid_w1.len() == 16
            && centroid_w2.len() == 16
            && centroid_w3.len() == 16
            && centroid_bias.len() == 16
        {
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
        let grid_len = le_u32(buf, &mut off)? as usize;
        let decision_grid = le_bytes(buf, &mut off, grid_len)?.to_vec();
        let atlas = LoadedPrefixAtlasCheckpoint {
            checkpoint: atlas_checkpoint,
            gap_edges: atlas_gap_edges,
            shared_gap_edges,
            feature_mean: [feature_mean_v[0], feature_mean_v[1], feature_mean_v[2], feature_mean_v[3]],
            feature_inv_std: [
                feature_inv_std_v[0],
                feature_inv_std_v[1],
                feature_inv_std_v[2],
                feature_inv_std_v[3],
            ],
            cluster_count,
            tau_bin_count: atlas_tau_bin_count,
            centroid_w0,
            centroid_w1,
            centroid_w2,
            centroid_w3,
            centroid_bias,
            centroids16,
            decision_grid,
        };
        checkpoints.push(LoadedPrefixCertifyCheckpoint { table, atlas });
    }
    Ok(LoadedPrefixCertifyPlan { tau_edges, checkpoints })
}

fn build_late_segment_runtime(
    base_pack_path: &PathBuf,
    pack: &qs_exact::QsPack,
    hot_layout: &HotCheckpointLayout,
    hot_tree_count: usize,
) -> Result<LateSegmentRuntime> {
    if hot_layout.late_values.is_empty() {
        return Ok(LateSegmentRuntime::default());
    }
    let manifest_path = default_late_segment_manifest_path(base_pack_path);
    if manifest_path.exists() {
        let txt = fs::read_to_string(&manifest_path)
            .with_context(|| format!("read {}", manifest_path.display()))?;
        let manifest: LateSegmentManifest = serde_json::from_str(&txt)
            .with_context(|| format!("parse {}", manifest_path.display()))?;
        if manifest.format != "L2LateSegmentsV1" {
            bail!("unsupported late segment manifest format: {}", manifest.format);
        }
        let mut segments = Vec::with_capacity(manifest.segments.len());
        for meta in manifest.segments {
            let pack_path = if PathBuf::from(&meta.pack_file).is_absolute() {
                PathBuf::from(&meta.pack_file)
            } else {
                manifest_path
                    .parent()
                    .map(|p| p.join(&meta.pack_file))
                    .unwrap_or_else(|| PathBuf::from(&meta.pack_file))
            };
            segments.push(LoadedLateSegment {
                checkpoint: meta.checkpoint,
                global_cp_idx: meta.global_cp_idx,
                start_tree_idx: meta.start_tree_idx,
                local_to_global_fid: meta.local_to_global_fid,
                pack: qs_exact::load_qs_pack(&pack_path)?,
            });
        }
        return Ok(LateSegmentRuntime { segments });
    }

    let mut segments = Vec::with_capacity(hot_layout.late_values.len());
    let mut manifest = LateSegmentManifest {
        format: "L2LateSegmentsV1".to_string(),
        hot_tree_count,
        segments: Vec::with_capacity(hot_layout.late_values.len()),
    };
    let mut late_start = hot_tree_count.min(pack.n_trees());
    for (late_pos, checkpoint) in hot_layout.late_values.iter().copied().enumerate() {
        let global_cp_idx = hot_layout.late_indices[late_pos];
        let segment_path = default_late_segment_pack_path(base_pack_path, checkpoint);
        let (segment_pack, local_to_global_fid) =
            qs_exact::extract_tree_range_remapped(pack, late_start, checkpoint)?;
        qs_exact::save_qs_pack_v2(&segment_path, &segment_pack)?;
        let pack_file = segment_path
            .file_name()
            .and_then(|x| x.to_str())
            .map(|x| x.to_string())
            .unwrap_or_else(|| segment_path.to_string_lossy().to_string());
        manifest.segments.push(LateSegmentMeta {
            checkpoint,
            global_cp_idx,
            start_tree_idx: late_start,
            pack_file,
            local_to_global_fid: local_to_global_fid.clone(),
        });
        segments.push(LoadedLateSegment {
            checkpoint,
            global_cp_idx,
            start_tree_idx: late_start,
            local_to_global_fid,
            pack: segment_pack,
        });
        late_start = checkpoint;
    }
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("write {}", manifest_path.display()))?;
    Ok(LateSegmentRuntime { segments })
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
    let qs_pack_path = resolved.qs_pack.clone();
    let pack = qs_exact::load_qs_pack(&qs_pack_path)?;
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
    let compiled_hot_pack = if resolved.direct_kernel == PrefixDirectKernel::Qs {
        None
    } else if let Some(path) = resolved.hot_exact_prefix_pack.as_ref() {
        let compiled_path = default_compiled_hot_pack_path(path);
        if compiled_path.exists() {
            Some(load_compiled_hot_prefix_pack(&compiled_path)?)
        } else {
            match hot_pack.as_ref() {
                Some(pack) => {
                    let compiled = compile_hot_prefix_pack(pack)?;
                    let _ = save_compiled_hot_prefix_pack(&compiled_path, &compiled);
                    Some(compiled)
                }
                None => None,
            }
        }
    } else {
        None
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
    let hot_tree_count = compiled_hot_pack
        .as_ref()
        .map(|x| x.n_trees)
        .or_else(|| hot_pack.as_ref().map(|x| x.n_trees));
    let hot_checkpoint_layout = if resolved.direct_kernel == PrefixDirectKernel::Qs {
        None
    } else if let Some(path) = resolved.hot_exact_prefix_pack.as_ref() {
        let layout_path = default_hot_checkpoint_layout_path(path);
        if layout_path.exists() {
            Some(load_hot_checkpoint_layout(&layout_path)?)
        } else {
            let built =
                build_hot_checkpoint_layout(&calibration, hot_tree_count, &pack, resolved.direct_kernel);
            if let Some(layout) = built.as_ref() {
                let _ = save_hot_checkpoint_layout(&layout_path, layout);
            }
            built
        }
    } else {
        build_hot_checkpoint_layout(&calibration, hot_tree_count, &pack, resolved.direct_kernel)
    };
    let hot_checkpoint_layout = hot_checkpoint_layout.map(|mut layout| {
        populate_hot_checkpoint_quant_spans(&mut layout, &pack);
        layout
    });
    let hot_stage_plan = match (compiled_hot_pack.as_ref(), hot_checkpoint_layout.as_ref()) {
        (Some(compiled), Some(layout)) if !layout.hot_values.is_empty() => {
            Some(build_hot_stage_plan(compiled, &layout.hot_values)?)
        }
        _ => None,
    };
    let certify_plan = if resolved.certifier_kind == PrefixCertifierKind::AtlasV1 {
        match (&atlas_certifier, &resolved.atlas_bin) {
            (Some(atlas), Some(atlas_path)) => {
                let certify_path = default_certify_plan_path(atlas_path);
                if certify_path.exists() {
                    Some(load_prefix_certify_plan(&certify_path)?)
                } else {
                    let plan = LoadedPrefixCertifyPlan {
                        tau_edges: atlas.tau_edges.clone(),
                        checkpoints: calibration
                            .tables
                            .iter()
                            .cloned()
                            .zip(atlas.checkpoints.iter().cloned())
                            .map(|(table, atlas)| LoadedPrefixCertifyCheckpoint { table, atlas })
                            .collect(),
                    };
                    let _ = save_prefix_certify_plan(&certify_path, &plan);
                    Some(plan)
                }
            }
            _ => None,
        }
    } else {
        None
    };
    let late_segment_runtime = if hot_checkpoint_layout
        .as_ref()
        .map(|x| !x.late_values.is_empty())
        .unwrap_or(false)
    {
        Some(build_late_segment_runtime(
            &resolved.qs_pack,
            &pack,
            hot_checkpoint_layout.as_ref().unwrap(),
            compiled_hot_pack
                .as_ref()
                .map(|x| x.n_trees.min(pack.n_trees()))
                .unwrap_or(0),
        )?)
    } else {
        None
    };
    let mut runtime = LoadedPrefixRuntime {
        pack,
        calibration,
        atlas_certifier,
        mlp_certifier,
        hot_pack,
        compiled_hot_pack,
        hot_checkpoint_layout,
        hot_stage_plan,
        certify_plan,
        late_segment_runtime,
        prefix_v2: None,
        plan,
        direct_kernel: resolved.direct_kernel,
        certifier_kind: resolved.certifier_kind,
        hot_exact_prefix_limit: resolved.hot_exact_prefix_limit.unwrap_or(0),
        telemetry_schema_version: resolved.telemetry_schema_version,
        packet_scheduler: None,
        anchor_rescue: None,
    };
    let _ = maybe_prepare_prefix_v2(resolved, &mut runtime);
    Ok(runtime)
}

impl LoadedPrefixRuntime {
    fn trim_for_online(&mut self) {
        if self.compiled_hot_pack.is_some() {
            self.hot_pack = None;
        }
        if let Some(anchor) = self.anchor_rescue.as_mut() {
            anchor.reject_rescue.trim_for_online();
            anchor.refer_rescue.trim_for_online();
        }
    }
}

fn build_hot_checkpoint_layout(
    calibration: &LoadedPrefixCal,
    hot_tree_count: Option<usize>,
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
    let hot_limit = hot_tree_count?.min(pack.n_trees());
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
    if !out.late_values.is_empty() {
        out.late_feature_offsets.push(0);
        let mut seen = vec![false; pack.n_features()];
        let mut late_start = hot_limit;
        for checkpoint in out.late_values.iter().copied() {
            qs_exact::append_tree_range_new_feature_ids(
                pack,
                late_start,
                checkpoint,
                &mut seen,
                &mut out.late_feature_ids,
            );
            out.late_feature_offsets.push(out.late_feature_ids.len() as u32);
            late_start = checkpoint;
        }
    }
    Some(out)
}

fn build_hot_stage_plan(
    compiled: &HotCompiledPrefixPack,
    checkpoints: &[usize],
) -> Result<HotStagePlan> {
    if checkpoints.is_empty() {
        return Ok(HotStagePlan::default());
    }
    let mut hot_seen = vec![false; compiled.n_hot_features];
    let mut hot_feature_ids = Vec::new();
    let mut feature_offsets = Vec::with_capacity(checkpoints.len() + 1);
    feature_offsets.push(0);
    let mut tree_start = 0usize;
    for &checkpoint in checkpoints {
        if checkpoint <= tree_start || checkpoint > compiled.n_trees {
            bail!(
                "invalid hot checkpoint {} for n_trees={}",
                checkpoint,
                compiled.n_trees
            );
        }
        collect_tree_hot_features(compiled, tree_start, checkpoint, &mut hot_seen, &mut hot_feature_ids)?;
        feature_offsets.push(hot_feature_ids.len() as u32);
        tree_start = checkpoint;
    }
    Ok(HotStagePlan {
        checkpoints: checkpoints.to_vec(),
        feature_offsets,
        hot_feature_ids,
    })
}

fn collect_tree_hot_features(
    compiled: &HotCompiledPrefixPack,
    start_tree: usize,
    end_tree: usize,
    hot_seen: &mut [bool],
    out: &mut Vec<u16>,
) -> Result<()> {
    let mut stack = Vec::new();
    let mut node_seen = vec![false; compiled.nodes.len()];
    for tree_idx in start_tree..end_tree {
        let root = *compiled
            .tree_roots
            .get(tree_idx)
            .context("tree root missing")? as usize;
        stack.push(root);
        while let Some(node_idx) = stack.pop() {
            if node_idx >= compiled.nodes.len() || node_seen[node_idx] {
                continue;
            }
            node_seen[node_idx] = true;
            let node = compiled.nodes[node_idx];
            if node.is_leaf() {
                continue;
            }
            let hot_fidx = node.hot_fidx();
            if !hot_seen[hot_fidx] {
                hot_seen[hot_fidx] = true;
                out.push(hot_fidx as u16);
            }
            stack.push(node.left as usize);
            stack.push(node.right as usize);
            stack.push(node.missing as usize);
        }
    }
    Ok(())
}

fn populate_hot_checkpoint_quant_spans(layout: &mut HotCheckpointLayout, pack: &qs_exact::QsPack) {
    if layout.late_cut_starts.len() == layout.late_feature_ids.len()
        && layout.late_cut_ends.len() == layout.late_feature_ids.len()
    {
        return;
    }
    layout.late_cut_starts.clear();
    layout.late_cut_ends.clear();
    layout.late_cut_starts.reserve(layout.late_feature_ids.len());
    layout.late_cut_ends.reserve(layout.late_feature_ids.len());
    for &fid_u16 in &layout.late_feature_ids {
        let fid = fid_u16 as usize;
        layout.late_cut_starts.push(pack.cut_offset(fid) as u32);
        layout.late_cut_ends.push(pack.cut_offset(fid + 1) as u32);
    }
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
        bundle_manifest: Some(bundle_manifest),
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
    if policy.used_tau_ref_dense.len() != policy.used_tau_ref.len() {
        policy.used_tau_ref_dense = policy
            .used_tau_ref
            .iter()
            .map(|v| v.unwrap_or(f32::NAN))
            .collect();
    }
    if policy.used_tau_positive_dense.len() != used_pos {
        policy.used_tau_positive_dense = policy
            .used_tau_positive()
            .iter()
            .map(|v| v.unwrap_or(f32::NAN))
            .collect();
    }
    if policy.checkpoint_exit_counts.len() != policy.used_checkpoints.len() {
        policy.checkpoint_exit_counts = vec![0u64; policy.used_checkpoints.len()];
    }
    Ok(policy)
}

impl ApproxPolicy {
    #[inline(always)]
    pub(crate) fn used_tau_positive(&self) -> &[Option<f32>] {
        if !self.used_tau_reject.is_empty() {
            &self.used_tau_reject
        } else {
            &self.used_tau_pass
        }
    }

    #[inline(always)]
    pub(crate) fn used_tau_ref_dense(&self) -> &[f32] {
        &self.used_tau_ref_dense
    }

    #[inline(always)]
    pub(crate) fn used_tau_positive_dense(&self) -> &[f32] {
        &self.used_tau_positive_dense
    }
}

fn load_rank_pack(path: &PathBuf) -> Result<RankPack> {
    let mmap = mmap_readonly(path)?;
    let buf = &mmap[..];
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
