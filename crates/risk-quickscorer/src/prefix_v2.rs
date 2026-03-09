#[derive(Debug, Clone, Serialize, Deserialize)]
struct PrefixV2Manifest {
    format: String,
    direct_kernel: String,
    certifier_kind: String,
    compiled_hot_pack_file: String,
    certify_plan_file: Option<String>,
    late_segment_manifest_file: Option<String>,
    hot_stage_plan_file: String,
}

#[derive(Debug, Clone, Copy)]
struct PrefixV2HotStage {
    checkpoint: usize,
    global_cp_idx: usize,
    feature_start: u32,
    feature_end: u32,
}

#[derive(Debug, Clone)]
struct PrefixV2HotStagePlan {
    hot_feature_ids: Vec<u16>,
    stages: Vec<PrefixV2HotStage>,
}

#[derive(Debug, Clone)]
struct LoadedPrefixV2Runtime {
    hot_stage_plan: PrefixV2HotStagePlan,
}

fn default_prefix_v2_manifest_path(path: &PathBuf) -> PathBuf {
    let mut out = path.clone();
    let file_name = path
        .file_name()
        .and_then(|x| x.to_str())
        .map(|x| format!("{}.prefix_v2.json", x))
        .unwrap_or_else(|| "prefix_v2.json".to_string());
    out.set_file_name(file_name);
    out
}

fn default_prefix_v2_hot_stage_plan_path(path: &PathBuf) -> PathBuf {
    let mut out = path.clone();
    let file_name = path
        .file_name()
        .and_then(|x| x.to_str())
        .map(|x| format!("{}.prefix_v2.hot_stage.bin", x))
        .unwrap_or_else(|| "prefix_v2.hot_stage.bin".to_string());
    out.set_file_name(file_name);
    out
}

fn save_prefix_v2_hot_stage_plan(path: &PathBuf, plan: &PrefixV2HotStagePlan) -> Result<()> {
    let mut out = Vec::with_capacity(
        8 + 8 + (plan.hot_feature_ids.len() * 2) + (plan.stages.len() * 16),
    );
    out.extend_from_slice(b"L2P2STG1");
    out.extend_from_slice(&(plan.hot_feature_ids.len() as u32).to_le_bytes());
    out.extend_from_slice(&(plan.stages.len() as u32).to_le_bytes());
    for &fid in &plan.hot_feature_ids {
        out.extend_from_slice(&fid.to_le_bytes());
    }
    for stage in &plan.stages {
        out.extend_from_slice(&(stage.checkpoint as u32).to_le_bytes());
        out.extend_from_slice(&(stage.global_cp_idx as u32).to_le_bytes());
        out.extend_from_slice(&stage.feature_start.to_le_bytes());
        out.extend_from_slice(&stage.feature_end.to_le_bytes());
    }
    fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn load_prefix_v2_hot_stage_plan(path: &PathBuf) -> Result<PrefixV2HotStagePlan> {
    let mmap = mmap_readonly(path)?;
    let buf = &mmap[..];
    if buf.len() < 16 {
        bail!("prefix_v2 hot stage plan too short");
    }
    let mut off = 0usize;
    let magic = le_bytes(buf, &mut off, 8)?;
    if magic != b"L2P2STG1" {
        bail!("invalid prefix_v2 hot stage plan magic");
    }
    let hot_feature_len = le_u32(buf, &mut off)? as usize;
    let stage_len = le_u32(buf, &mut off)? as usize;
    let mut hot_feature_ids = Vec::with_capacity(hot_feature_len);
    for _ in 0..hot_feature_len {
        hot_feature_ids.push(le_u16(buf, &mut off)?);
    }
    let mut stages = Vec::with_capacity(stage_len);
    for _ in 0..stage_len {
        stages.push(PrefixV2HotStage {
            checkpoint: le_u32(buf, &mut off)? as usize,
            global_cp_idx: le_u32(buf, &mut off)? as usize,
            feature_start: le_u32(buf, &mut off)?,
            feature_end: le_u32(buf, &mut off)?,
        });
    }
    Ok(PrefixV2HotStagePlan {
        hot_feature_ids,
        stages,
    })
}

fn build_prefix_v2_hot_stage_plan(
    stage_plan: &HotStagePlan,
    hot_indices: &[usize],
) -> Result<PrefixV2HotStagePlan> {
    if stage_plan.checkpoints.len() != hot_indices.len() {
        bail!(
            "prefix_v2 stage length mismatch: checkpoints={} hot_indices={}",
            stage_plan.checkpoints.len(),
            hot_indices.len()
        );
    }
    if stage_plan.feature_offsets.len() != stage_plan.checkpoints.len() + 1 {
        bail!(
            "prefix_v2 feature_offsets mismatch: offsets={} checkpoints={}",
            stage_plan.feature_offsets.len(),
            stage_plan.checkpoints.len()
        );
    }
    let stages = stage_plan
        .checkpoints
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, checkpoint)| PrefixV2HotStage {
            checkpoint,
            global_cp_idx: hot_indices[idx],
            feature_start: stage_plan.feature_offsets[idx],
            feature_end: stage_plan.feature_offsets[idx + 1],
        })
        .collect();
    Ok(PrefixV2HotStagePlan {
        hot_feature_ids: stage_plan.hot_feature_ids.clone(),
        stages,
    })
}

fn build_prefix_v2_manifest(
    resolved: &ResolvedPrefixCalBundle,
    runtime: &LoadedPrefixRuntime,
    base_dir: &PathBuf,
) -> Result<PrefixV2Manifest> {
    let hot_pack_path = resolved
        .hot_exact_prefix_pack
        .as_ref()
        .context("prefix_v2 requires hot exact prefix pack")?;
    let compiled_path = default_compiled_hot_pack_path(hot_pack_path);
    let hot_stage_plan_path = default_prefix_v2_hot_stage_plan_path(hot_pack_path);
    let certify_plan_file = resolved.atlas_bin.as_ref().and_then(|atlas_path| {
        let path = default_certify_plan_path(atlas_path);
        path.exists().then(|| {
            path.file_name()
                .and_then(|x| x.to_str())
                .map(|x| x.to_string())
                .unwrap_or_else(|| path.to_string_lossy().to_string())
        })
    });
    let late_segment_manifest_file = {
        let path = default_late_segment_manifest_path(&resolved.qs_pack);
        path.exists().then(|| {
            path.file_name()
                .and_then(|x| x.to_str())
                .map(|x| x.to_string())
                .unwrap_or_else(|| path.to_string_lossy().to_string())
        })
    };
    let compiled_hot_pack_file = compiled_path
        .strip_prefix(base_dir)
        .ok()
        .and_then(|x| x.to_str())
        .map(|x| x.to_string())
        .unwrap_or_else(|| {
            compiled_path
                .file_name()
                .and_then(|x| x.to_str())
                .map(|x| x.to_string())
                .unwrap_or_else(|| compiled_path.to_string_lossy().to_string())
        });
    let hot_stage_plan_file = hot_stage_plan_path
        .strip_prefix(base_dir)
        .ok()
        .and_then(|x| x.to_str())
        .map(|x| x.to_string())
        .unwrap_or_else(|| {
            hot_stage_plan_path
                .file_name()
                .and_then(|x| x.to_str())
                .map(|x| x.to_string())
                .unwrap_or_else(|| hot_stage_plan_path.to_string_lossy().to_string())
        });
    Ok(PrefixV2Manifest {
        format: "L2PrefixV2".to_string(),
        direct_kernel: prefix_direct_kernel_str(runtime.direct_kernel).to_string(),
        certifier_kind: prefix_certifier_kind_str(runtime.certifier_kind).to_string(),
        compiled_hot_pack_file,
        certify_plan_file,
        late_segment_manifest_file,
        hot_stage_plan_file,
    })
}

fn load_prefix_v2_runtime(
    manifest_path: &PathBuf,
    runtime: &LoadedPrefixRuntime,
) -> Result<LoadedPrefixV2Runtime> {
    let txt = fs::read_to_string(manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest: PrefixV2Manifest = serde_json::from_str(&txt)
        .with_context(|| format!("parse {}", manifest_path.display()))?;
    if manifest.format != "L2PrefixV2" {
        bail!("unsupported prefix_v2 format: {}", manifest.format);
    }
    if manifest.direct_kernel != prefix_direct_kernel_str(runtime.direct_kernel) {
        bail!(
            "prefix_v2 direct_kernel mismatch: manifest={} runtime={}",
            manifest.direct_kernel,
            prefix_direct_kernel_str(runtime.direct_kernel)
        );
    }
    if manifest.certifier_kind != prefix_certifier_kind_str(runtime.certifier_kind) {
        bail!(
            "prefix_v2 certifier_kind mismatch: manifest={} runtime={}",
            manifest.certifier_kind,
            prefix_certifier_kind_str(runtime.certifier_kind)
        );
    }
    if runtime.compiled_hot_pack.is_none() {
        bail!("prefix_v2 requires compiled hot pack");
    }
    if runtime.certify_plan.is_none() {
        bail!("prefix_v2 requires certify plan");
    }
    if runtime
        .late_segment_runtime
        .as_ref()
        .map(|x| x.segments.is_empty())
        .unwrap_or(true)
    {
        bail!("prefix_v2 requires late segment runtime");
    }
    let base_dir = manifest_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let hot_stage_plan_path = base_dir.join(&manifest.hot_stage_plan_file);
    let hot_stage_plan = load_prefix_v2_hot_stage_plan(&hot_stage_plan_path)?;
    Ok(LoadedPrefixV2Runtime { hot_stage_plan })
}

fn maybe_prepare_prefix_v2(
    resolved: &ResolvedPrefixCalBundle,
    runtime: &mut LoadedPrefixRuntime,
) -> Result<()> {
    let Some(hot_pack_path) = resolved.hot_exact_prefix_pack.as_ref() else {
        return Ok(());
    };
    let Some(stage_plan) = runtime.hot_stage_plan.as_ref() else {
        return Ok(());
    };
    let Some(hot_layout) = runtime.hot_checkpoint_layout.as_ref() else {
        return Ok(());
    };
    if runtime.certify_plan.is_none() || runtime.compiled_hot_pack.is_none() {
        return Ok(());
    }
    if runtime
        .late_segment_runtime
        .as_ref()
        .map(|x| x.segments.is_empty())
        .unwrap_or(true)
    {
        return Ok(());
    }

    let manifest_path = default_prefix_v2_manifest_path(hot_pack_path);
    let hot_stage_plan_path = default_prefix_v2_hot_stage_plan_path(hot_pack_path);
    if !manifest_path.exists() {
        let plan = build_prefix_v2_hot_stage_plan(stage_plan, &hot_layout.hot_indices)?;
        save_prefix_v2_hot_stage_plan(&hot_stage_plan_path, &plan)?;
        let manifest = build_prefix_v2_manifest(
            resolved,
            runtime,
            &manifest_path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from(".")),
        )?;
        fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
            .with_context(|| format!("write {}", manifest_path.display()))?;
    }
    runtime.prefix_v2 = Some(load_prefix_v2_runtime(&manifest_path, runtime)?);
    Ok(())
}

fn run_prefix_shadow_row_single_route_prefix_v2_nomiss(
    runtime: &LoadedPrefixRuntime,
    feat: &[f32],
    row_tau: f32,
    row_fold: i32,
    hot_buf: &mut HotFeatureBuf,
    ranks: &mut [u8],
) -> Result<Option<PrefixRowShadow>> {
    let Some(prefix_v2) = runtime.prefix_v2.as_ref() else {
        return Ok(None);
    };
    let Some(compiled_hot_pack) = runtime.compiled_hot_pack.as_ref() else {
        return Ok(None);
    };
    let Some(certify_plan) = runtime.certify_plan.as_ref() else {
        return Ok(None);
    };
    let Some(atlas) = runtime.atlas_certifier.as_ref() else {
        return Ok(None);
    };
    let Some(late_runtime) = runtime.late_segment_runtime.as_ref() else {
        return Ok(None);
    };
    if runtime.certifier_kind != PrefixCertifierKind::AtlasV1 {
        return Ok(None);
    }
    if !matches!(
        runtime.direct_kernel,
        PrefixDirectKernel::HotExact96
            | PrefixDirectKernel::HotExact128
            | PrefixDirectKernel::HotExact192
            | PrefixDirectKernel::HotExact256
            | PrefixDirectKernel::HotExact384
    ) {
        return Ok(None);
    }

    let mut out = PrefixRowShadow::default();
    let mut last_checkpoint_score = runtime.pack.base_score;
    let mut score_at_64: Option<f32> = None;
    let atlas_tau_bin = atlas_tau_bin(&atlas.tau_edges, row_tau);

    let mut score = runtime.pack.base_score;
    let mut work_evals = 0u64;
    let mut resolved_early = 0u64;
    let mut trees_used = 0usize;
    let mut direct_decision: Option<(bool, f32, usize)> = None;
    let mut tree_start = 0usize;

    for stage in &prefix_v2.hot_stage_plan.stages {
        let feature_start = stage.feature_start as usize;
        let feature_end = stage.feature_end as usize;
        if feature_end > feature_start {
            unsafe {
                hot_buf.fill_subset_from_global_u32(
                    &compiled_hot_pack.hot_global_fidx,
                    feat,
                    &prefix_v2.hot_stage_plan.hot_feature_ids[feature_start..feature_end],
                );
            }
        }
        let hot_feat = hot_buf.values.as_slice();
        for pos in tree_start..stage.checkpoint {
            let mut idx = compiled_hot_pack.tree_roots[pos] as usize;
            loop {
                let node = compiled_hot_pack.nodes[idx];
                if node.is_leaf() {
                    score += node.leaf;
                    break;
                }
                let x = hot_feat[node.hot_fidx()];
                work_evals += 1;
                idx = if x < node.thr {
                    node.left as usize
                } else {
                    node.right as usize
                };
            }
        }
        let delta_prev = if stage.global_cp_idx == 0 {
            score - runtime.pack.base_score
        } else {
            score - last_checkpoint_score
        };
        let delta_from_64 = if let Some(score64) = score_at_64 {
            score - score64
        } else {
            0.0
        };
        if stage.checkpoint == 64 {
            score_at_64 = Some(score);
        }
        last_checkpoint_score = score;
        if let Some((is_reject, route_score)) = maybe_prefix_certify_plan_fast(
            &certify_plan.checkpoints[stage.global_cp_idx],
            row_fold,
            row_tau,
            atlas_tau_bin,
            score,
            delta_prev,
            delta_from_64,
        ) {
            direct_decision = Some((is_reject, route_score, stage.checkpoint));
            trees_used = stage.checkpoint;
            break;
        }
        trees_used = stage.checkpoint;
        tree_start = stage.checkpoint;
    }

    if direct_decision.is_none() {
        for segment in &late_runtime.segments {
            qs_exact::quantize_feature_subset_nomiss_mapped(
                &segment.pack,
                feat,
                ranks,
                &segment.local_to_global_fid,
            );
            let late_row = qs_exact::score_all_trees_prequantized_nomiss(
                &segment.pack,
                ranks,
                score,
                work_evals,
                resolved_early,
            )?;
            score = late_row.prefix_score;
            work_evals = late_row.block_evals;
            resolved_early = late_row.resolved_early_trees;
            trees_used = segment.checkpoint;

            let delta_prev = score - last_checkpoint_score;
            let delta_from_64 = if let Some(score64) = score_at_64 {
                score - score64
            } else {
                0.0
            };
            last_checkpoint_score = score;
            if let Some((is_reject, route_score)) = maybe_prefix_certify_plan_fast(
                &certify_plan.checkpoints[segment.global_cp_idx],
                row_fold,
                row_tau,
                atlas_tau_bin,
                score,
                delta_prev,
                delta_from_64,
            ) {
                direct_decision = Some((is_reject, route_score, segment.checkpoint));
                break;
            }
        }
    }

    out.prefix_score = score;
    out.trees_used = trees_used;
    out.work_evals = work_evals;
    out.resolved_early_trees = resolved_early;
    if let Some((is_reject, route_score, checkpoint)) = direct_decision {
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
    Ok(Some(out))
}
