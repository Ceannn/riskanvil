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

fn parse_base_score_from_any_json(json_path: &PathBuf) -> Result<f32> {
    let txt = fs::read_to_string(json_path)
        .with_context(|| format!("read json failed: {}", json_path.display()))?;
    let v: Value = serde_json::from_str(&txt).context("parse json failed")?;

    fn visit(v: &Value) -> Option<f32> {
        match v {
            Value::Object(map) => {
                if let Some(raw) = map.get("base_score") {
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
                    if let Ok(v) = cleaned.parse::<f32>() {
                        return Some(v);
                    }
                }
                for child in map.values() {
                    if let Some(v) = visit(child) {
                        return Some(v);
                    }
                }
                None
            }
            Value::Array(items) => {
                for child in items {
                    if let Some(v) = visit(child) {
                        return Some(v);
                    }
                }
                None
            }
            _ => None,
        }
    }

    visit(&v).with_context(|| format!("base_score missing in {}", json_path.display()))
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

    #[inline(always)]
    unsafe fn fill_subset_from_global_u32(
        &mut self,
        hot_global_fidx: &[u32],
        feat: &[f32],
        hot_ids: &[u16],
    ) {
        let mut i = 0usize;
        while i < hot_ids.len() {
            let hot_idx = *hot_ids.get_unchecked(i) as usize;
            let g = *hot_global_fidx.get_unchecked(hot_idx) as usize;
            *self.values.get_unchecked_mut(hot_idx) = *feat.get_unchecked(g);
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
unsafe fn hot_exact_prefix_until_compiled_nomiss_staged<F>(
    hot_pack: &HotCompiledPrefixPack,
    hot_buf: &mut HotFeatureBuf,
    feat: &[f32],
    stage_plan: &HotStagePlan,
    init_score: f32,
    mut on_checkpoint: F,
) -> Result<(f32, usize, u64, u64)>
where
    F: FnMut(usize, f32, u64, u64) -> bool,
{
    if stage_plan.checkpoints.is_empty() {
        bail!("compiled hot exact staged checkpoints must not be empty");
    }
    if stage_plan.feature_offsets.len() != stage_plan.checkpoints.len() + 1 {
        bail!("compiled hot exact staged offsets len mismatch");
    }

    let mut score = init_score;
    let mut node_evals = 0u64;
    let resolved_early = 0u64;
    let mut tree_start = 0usize;

    for (stage_idx, &checkpoint) in stage_plan.checkpoints.iter().enumerate() {
        if checkpoint <= tree_start || checkpoint > hot_pack.n_trees {
            bail!(
                "invalid compiled hot exact staged checkpoint {} for n_trees={}",
                checkpoint,
                hot_pack.n_trees
            );
        }
        let feature_off = stage_plan.feature_offsets[stage_idx] as usize;
        let feature_end = stage_plan.feature_offsets[stage_idx + 1] as usize;
        if feature_end > feature_off {
            hot_buf.fill_subset_from_global_u32(
                &hot_pack.hot_global_fidx,
                feat,
                &stage_plan.hot_feature_ids[feature_off..feature_end],
            );
        }
        let hot_feat = hot_buf.values.as_slice();

        for pos in tree_start..checkpoint {
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
        }
        if on_checkpoint(checkpoint, score, node_evals, resolved_early) {
            return Ok((score, checkpoint, node_evals, resolved_early));
        }
        tree_start = checkpoint;
    }
    Ok((score, tree_start, node_evals, resolved_early))
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

const HOT_L1_COMPACT_NODE_LEAF: u16 = 1 << 15;
const HOT_L1_COMPACT_NODE_FIDX_MASK: u16 = HOT_L1_COMPACT_NODE_LEAF - 1;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct HotApproxL1CompactNode {
    thr: f32,
    leaf: f32,
    left: u16,
    right: u16,
    fidx_flags: u16,
    _pad: u16,
}

impl HotApproxL1CompactNode {
    #[inline(always)]
    fn is_leaf(self) -> bool {
        (self.fidx_flags & HOT_L1_COMPACT_NODE_LEAF) != 0
    }

    #[inline(always)]
    fn fidx(self) -> usize {
        (self.fidx_flags & HOT_L1_COMPACT_NODE_FIDX_MASK) as usize
    }
}

#[derive(Debug)]
pub(crate) struct HotApproxL1CompactPack {
    n_trees: usize,
    tree_node_offs: Vec<u32>,
    nodes: Vec<HotApproxL1CompactNode>,
}

pub(crate) fn compile_hot_approx_l1_compact_pack(
    model: &SoaModel,
    plan: &TreeOrder,
    approx_policy: &ApproxPolicy,
) -> Result<Option<HotApproxL1CompactPack>> {
    let k_hot = approx_policy.k_hot.min(plan.n_trees);
    if k_hot == 0 || model.n_features > HOT_L1_COMPACT_NODE_FIDX_MASK as usize {
        return Ok(None);
    }

    let mut tree_node_offs = Vec::with_capacity(k_hot);
    let mut nodes = Vec::new();
    let mut local_idx_buf = vec![u16::MAX; model.is_leaf.len()];
    let mut stack = Vec::new();

    for pos in 0..k_hot {
        let tree_idx = plan.order[pos] as usize;
        let root = model.tree_roots[tree_idx] as usize;
        tree_node_offs.push(nodes.len() as u32);
        stack.clear();
        stack.push(root);

        while let Some(idx) = stack.pop() {
            if local_idx_buf[idx] != u16::MAX {
                continue;
            }
            let local_idx = (nodes.len() - tree_node_offs[pos] as usize) as u16;
            local_idx_buf[idx] = local_idx;
            if model.is_leaf[idx] != 0 {
                nodes.push(HotApproxL1CompactNode {
                    thr: 0.0,
                    leaf: model.leaf[idx],
                    left: 0,
                    right: 0,
                    fidx_flags: HOT_L1_COMPACT_NODE_LEAF,
                    _pad: 0,
                });
                continue;
            }
            nodes.push(HotApproxL1CompactNode {
                thr: model.thr[idx],
                leaf: 0.0,
                left: 0,
                right: 0,
                fidx_flags: model.fidx[idx] as u16,
                _pad: 0,
            });
            stack.push(model.right[idx] as usize);
            stack.push(model.left[idx] as usize);
        }

        stack.clear();
        stack.push(root);
        while let Some(idx) = stack.pop() {
            let local_idx = local_idx_buf[idx] as usize + tree_node_offs[pos] as usize;
            if model.is_leaf[idx] != 0 {
                continue;
            }
            let left_idx = model.left[idx] as usize;
            let right_idx = model.right[idx] as usize;
            nodes[local_idx].left = local_idx_buf[left_idx];
            nodes[local_idx].right = local_idx_buf[right_idx];
            stack.push(right_idx);
            stack.push(left_idx);
        }

        stack.clear();
        stack.push(root);
        while let Some(idx) = stack.pop() {
            let local = std::mem::replace(&mut local_idx_buf[idx], u16::MAX);
            if local == u16::MAX || model.is_leaf[idx] != 0 {
                continue;
            }
            stack.push(model.right[idx] as usize);
            stack.push(model.left[idx] as usize);
        }
    }

    Ok(Some(HotApproxL1CompactPack {
        n_trees: k_hot,
        tree_node_offs,
        nodes,
    }))
}

#[inline(always)]
fn maybe_exact_exit_margin(
    margin: f32,
    suffix_min: f32,
    suffix_max: f32,
    eps: f32,
    bound_guard: f32,
) -> Option<bool> {
    let lower = (margin as f64) + (suffix_min as f64);
    let upper = (margin as f64) + (suffix_max as f64);
    let pass_cut = (eps + bound_guard) as f64;
    let ref_cut = -(eps + bound_guard) as f64;
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
    traverse_approx_float_nomiss_l1_hot_impl(
        None,
        model,
        feat,
        threshold,
        base_score,
        plan,
        approx_policy,
        eps,
        bound_guard,
        bound_check_every,
    )
}

#[inline(always)]
pub(crate) unsafe fn traverse_approx_float_nomiss_l1_hot_compact(
    hot_pack: &HotApproxL1CompactPack,
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
    traverse_approx_float_nomiss_l1_hot_impl(
        Some(hot_pack),
        model,
        feat,
        threshold,
        base_score,
        plan,
        approx_policy,
        eps,
        bound_guard,
        bound_check_every,
    )
}

#[inline(always)]
unsafe fn traverse_approx_float_nomiss_l1_hot_impl(
    hot_pack: Option<&HotApproxL1CompactPack>,
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
    let mut margin = base_score - threshold;
    let mut visited = 0i32;
    let checkpoints = approx_policy.used_checkpoints.as_slice();
    let tau_ref = approx_policy.used_tau_ref_dense();
    let tau_pos = approx_policy.used_tau_positive_dense();
    let mut cp_idx = 0usize;
    let mut next_cp = checkpoints.get(0).copied().unwrap_or(usize::MAX);
    let k_hot = approx_policy.k_hot.min(plan.n_trees);

    if let Some(hot_pack) = hot_pack {
        let hot_tree_offs = hot_pack.tree_node_offs.as_slice();
        let hot_nodes = hot_pack.nodes.as_slice();
        let hot_trees = hot_pack.n_trees.min(k_hot);
        for pos in 0..hot_trees {
            let base = *hot_tree_offs.get_unchecked(pos) as usize;
            let mut local_idx = 0usize;
            loop {
                let node = *hot_nodes.get_unchecked(base + local_idx);
                if node.is_leaf() {
                    margin += node.leaf;
                    break;
                }
                let x = *feat.get_unchecked(node.fidx());
                local_idx = if x < node.thr {
                    node.left as usize
                } else {
                    node.right as usize
                };
            }
            let m_end = pos + 1;
            visited = m_end as i32;
            if m_end == next_cp {
                let checkpoint_slot = cp_idx;
                cp_idx += 1;
                let tau = *tau_ref.get_unchecked(checkpoint_slot);
                if !tau.is_nan() && margin <= tau {
                    return Ok((
                        margin + threshold,
                        false,
                        visited,
                        RowMeta {
                            approx_pass: false,
                            approx_refer: true,
                            approx_checkpoint_idx: checkpoint_slot as i32,
                        },
                    ));
                }
                let tau = *tau_pos.get_unchecked(checkpoint_slot);
                if !tau.is_nan() && margin >= tau {
                    return Ok((
                        margin + threshold,
                        true,
                        visited,
                        RowMeta {
                            approx_pass: true,
                            approx_refer: false,
                            approx_checkpoint_idx: checkpoint_slot as i32,
                        },
                    ));
                }
                next_cp = checkpoints.get(cp_idx).copied().unwrap_or(usize::MAX);
            }
        }
    } else {
        for pos in 0..k_hot {
            let tree_idx = *plan.order.get_unchecked(pos) as usize;
            let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
            loop {
                if *model.is_leaf.get_unchecked(idx) != 0 {
                    margin += *model.leaf.get_unchecked(idx);
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
                let tau = *tau_ref.get_unchecked(checkpoint_slot);
                if !tau.is_nan() && margin <= tau {
                    return Ok((
                        margin + threshold,
                        false,
                        visited,
                        RowMeta {
                            approx_pass: false,
                            approx_refer: true,
                            approx_checkpoint_idx: checkpoint_slot as i32,
                        },
                    ));
                }
                let tau = *tau_pos.get_unchecked(checkpoint_slot);
                if !tau.is_nan() && margin >= tau {
                    return Ok((
                        margin + threshold,
                        true,
                        visited,
                        RowMeta {
                            approx_pass: true,
                            approx_refer: false,
                            approx_checkpoint_idx: checkpoint_slot as i32,
                        },
                    ));
                }
                next_cp = checkpoints.get(cp_idx).copied().unwrap_or(usize::MAX);
            }
        }
    }

    let hot_done = hot_pack.map(|p| p.n_trees.min(k_hot)).unwrap_or(k_hot);
    if bound_check_every == 1 {
        for pos in hot_done..plan.n_trees {
            let tree_idx = *plan.order.get_unchecked(pos) as usize;
            let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
            loop {
                if *model.is_leaf.get_unchecked(idx) != 0 {
                    margin += *model.leaf.get_unchecked(idx);
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
            if let Some(pass) = maybe_exact_exit_margin(
                margin,
                *plan.suffix_min.get_unchecked(m_end),
                *plan.suffix_max.get_unchecked(m_end),
                eps,
                bound_guard,
            ) {
                return Ok((margin + threshold, pass, visited, RowMeta::default()));
            }
        }
    } else {
        for pos in hot_done..plan.n_trees {
            let tree_idx = *plan.order.get_unchecked(pos) as usize;
            let mut idx = *model.tree_roots.get_unchecked(tree_idx) as usize;
            loop {
                if *model.is_leaf.get_unchecked(idx) != 0 {
                    margin += *model.leaf.get_unchecked(idx);
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
                if let Some(pass) = maybe_exact_exit_margin(
                    margin,
                    *plan.suffix_min.get_unchecked(m_end),
                    *plan.suffix_max.get_unchecked(m_end),
                    eps,
                    bound_guard,
                ) {
                    return Ok((margin + threshold, pass, visited, RowMeta::default()));
                }
            }
        }
    }

    Ok((margin + threshold, margin >= 0.0, visited, RowMeta::default()))
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
                    let mut rank_cache = rank_pack.map(|pack| RankCache::new(pack.n_features));
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
            let mut rank_cache = rank_pack.map(|pack| RankCache::new(pack.n_features));
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
