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
        let route_meta = load_route_meta(&resolved.route_meta)?;
        let model = load_soa(&resolved.soa)?;
        let bounds = load_bounds(&resolved.bounds)?;
        let mut runtime = load_prefix_runtime(&resolved, &model, &bounds)?;
        let sample_rows = batch.n_rows.min(route_meta.n_rows);
        maybe_apply_sampled_late_block_reorder(&mut runtime, &batch, &route_meta, sample_rows)?;
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

fn maybe_apply_sampled_late_block_reorder(
    runtime: &mut LoadedPrefixRuntime,
    batch: &FeatureBatch,
    route_meta: &RouteMeta,
    n: usize,
) -> Result<()> {
    if !batch.nan_free {
        return Ok(());
    }
    let Some(compiled_hot_pack) = runtime.compiled_hot_pack.as_ref() else {
        return Ok(());
    };
    let Some(hot_checkpoint_layout) = runtime.hot_checkpoint_layout.as_ref() else {
        return Ok(());
    };
    if hot_checkpoint_layout.late_values.is_empty() {
        return Ok(());
    }

    let hot_limit = compiled_hot_pack.n_trees.min(runtime.pack.n_trees());
    let late_end = *hot_checkpoint_layout.late_values.last().unwrap_or(&hot_limit);
    if hot_limit >= late_end {
        return Ok(());
    }

    let sample_cap = 256usize;
    let mut sample_ranks = Vec::with_capacity(sample_cap);
    let mut missing = vec![0u8; runtime.pack.n_features()];
    let mut active_rows = Vec::new();
    for row_idx in 0..n.min(batch.n_rows).min(route_meta.n_rows) {
        if route_meta.active[row_idx] != 0 {
            active_rows.push(row_idx);
        }
    }
    if active_rows.is_empty() {
        return Ok(());
    }

    let sample_n = sample_cap.min(active_rows.len());
    for sample_idx in 0..sample_n {
        let row_idx = active_rows[sample_idx * active_rows.len() / sample_n];
        let feat = batch.row(row_idx);
        let mut ranks = vec![0u8; runtime.pack.n_features()];
        qs_exact::quantize_into(&runtime.pack, feat, &mut ranks, &mut missing);
        sample_ranks.push(ranks);
    }

    if sample_ranks.is_empty() {
        return Ok(());
    }

    qs_exact::reorder_front_blocks_sampled_nomiss(
        &mut runtime.pack,
        &sample_ranks,
        hot_limit,
        late_end,
        16,
    )
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
