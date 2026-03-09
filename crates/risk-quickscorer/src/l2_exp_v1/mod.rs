use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::{
    absolutize_resolved_bundle, atlas_tau_bin, load_bounds, load_compiled_hot_prefix_pack,
    load_hot_checkpoint_layout, load_prefix_runtime, load_soa, maybe_prefix_certify_atlas_fast,
    prefix_fold_slot, qs_exact, resolve_prefix_cal_bundle, save_compiled_hot_prefix_pack,
    save_hot_checkpoint_layout, HotCheckpointLayout, HotCompiledPrefixPack, LoadedPrefixRuntime,
    OnlineL2Output, QsL2PrefixCalArgs,
};

pub(crate) const EXP_V1_FORMAT: &str = "L2ExpV1";

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ExpV1Manifest {
    format: String,
    compiled_hot_pack_file: String,
    hot_layout_file: String,
    hot_stage_plan_file: String,
    final_checkpoint: usize,
    late_segments: Vec<ExpV1LateSegmentMeta>,
    tail: Option<ExpV1TailMeta>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExpV1LateSegmentMeta {
    checkpoint: usize,
    global_cp_idx: usize,
    pack_file: String,
    local_to_global_fid: Vec<u16>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ExpV1TailMeta {
    pack_file: String,
    local_to_global_fid: Vec<u16>,
    tree_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct HotStagePlanFile {
    checkpoints: Vec<usize>,
    feature_offsets: Vec<u32>,
    hot_feature_ids: Vec<u16>,
}

#[derive(Debug)]
struct HotStagePlan {
    checkpoints: Vec<usize>,
    feature_offsets: Vec<u32>,
    hot_feature_ids: Vec<u16>,
}

#[derive(Debug)]
struct ExpV1LateSegment {
    checkpoint: usize,
    global_cp_idx: usize,
    pack: qs_exact::QsPack,
    local_to_global_fid: Vec<u16>,
    all_tree_indices: Vec<u32>,
}

#[derive(Debug)]
struct ExpV1TailRuntime {
    pack: qs_exact::QsPack,
    local_to_global_fid: Vec<u16>,
    all_tree_indices: Vec<u32>,
    tree_count: usize,
}

#[derive(Debug)]
pub(crate) struct L2ExpV1Runtime {
    compiled_hot_pack: HotCompiledPrefixPack,
    hot_checkpoint_layout: HotCheckpointLayout,
    hot_stage_plan: HotStagePlan,
    late_segments: Vec<ExpV1LateSegment>,
    tail: Option<ExpV1TailRuntime>,
    final_checkpoint: usize,
}

impl L2ExpV1Runtime {
    pub(crate) fn hot_feature_count(&self) -> usize {
        self.compiled_hot_pack.n_hot_features
    }
}

#[derive(Debug)]
pub(crate) struct ExpHotStageBuf {
    values: Vec<f32>,
}

impl ExpHotStageBuf {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            values: vec![0.0; len],
        }
    }

    pub(crate) fn ensure_len(&mut self, len: usize) {
        if self.values.len() != len {
            self.values.resize(len, 0.0);
        }
    }

    #[inline(always)]
    fn stage_range(&mut self, compiled: &HotCompiledPrefixPack, feat: &[f32], ids: &[u16]) {
        for &hot_fid_u16 in ids {
            let hot_fid = hot_fid_u16 as usize;
            let global_fid = compiled.hot_global_fidx[hot_fid] as usize;
            self.values[hot_fid] = feat[global_fid];
        }
    }
}

pub(crate) fn repack_from_bundle_manifest(
    bundle_manifest: &PathBuf,
    output_dir: Option<&PathBuf>,
) -> Result<PathBuf> {
    let prefix_args = QsL2PrefixCalArgs {
        bundle_manifest: Some(bundle_manifest.clone()),
        qs_pack: None,
        calibration_json: None,
        variant_key: None,
        feat_bin: None,
        route_meta: None,
        soa: None,
        bounds: None,
        tree_order: None,
        model_json: None,
        certifier_json: None,
        direct_kernel: None,
        certifier_kind: None,
        max_rows: None,
        out_tsv: None,
        stats_json: None,
        threads: 0,
        chunk_rows: 128,
        parallel_min_rows: 4096,
        trace_jsonl: None,
        shadow_only: false,
    };
    let resolved = resolve_prefix_cal_bundle(&prefix_args)?;
    let bundle_root = bundle_manifest
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "unexpected bundle manifest location: {}",
                bundle_manifest.display()
            )
        })?
        .to_path_buf();
    let resolved = absolutize_resolved_bundle(&bundle_root, resolved);
    let model = load_soa(&resolved.soa)?;
    let bounds = load_bounds(&resolved.bounds)?;
    let runtime = load_prefix_runtime(&resolved, &model, &bounds)?;
    let compiled = runtime
        .compiled_hot_pack
        .as_ref()
        .context("l2_exp_v1 requires compiled hot prefix pack")?;
    let hot_layout = runtime
        .hot_checkpoint_layout
        .as_ref()
        .context("l2_exp_v1 requires hot checkpoint layout")?;

    let out_dir = output_dir.cloned().unwrap_or_else(|| {
        resolved
            .qs_pack
            .parent()
            .map(|p| p.join("l2_exp_v1"))
            .unwrap_or_else(|| PathBuf::from("l2_exp_v1"))
    });
    fs::create_dir_all(&out_dir).with_context(|| format!("mkdir {}", out_dir.display()))?;

    let compiled_path = out_dir.join("hot_compiled.bin");
    save_compiled_hot_prefix_pack(&compiled_path, compiled)?;
    let hot_layout_path = out_dir.join("hot_layout.bin");
    save_hot_checkpoint_layout(&hot_layout_path, hot_layout)?;

    let hot_stage_plan = build_hot_stage_plan(compiled, &hot_layout.hot_values)?;
    let hot_stage_plan_path = out_dir.join("hot_stage_plan.json");
    fs::write(
        &hot_stage_plan_path,
        serde_json::to_vec_pretty(&hot_stage_plan)?,
    )
    .with_context(|| format!("write {}", hot_stage_plan_path.display()))?;

    let hot_tree_count = compiled.n_trees.min(runtime.pack.n_trees());
    let mut late_start = hot_tree_count;
    let mut late_segments = Vec::with_capacity(hot_layout.late_values.len());
    for (late_pos, checkpoint) in hot_layout.late_values.iter().copied().enumerate() {
        let global_cp_idx = hot_layout.late_indices[late_pos];
        let segment_path = out_dir.join(format!("late_cp_{}.v2.bin", checkpoint));
        let (segment_pack, local_to_global_fid) =
            qs_exact::extract_tree_range_remapped(&runtime.pack, late_start, checkpoint)?;
        qs_exact::save_qs_pack_v2(&segment_path, &segment_pack)?;
        late_segments.push(ExpV1LateSegmentMeta {
            checkpoint,
            global_cp_idx,
            pack_file: file_name_string(&segment_path),
            local_to_global_fid,
        });
        late_start = checkpoint;
    }

    let final_checkpoint = runtime
        .calibration
        .checkpoints
        .last()
        .copied()
        .unwrap_or(0usize);
    let tail_indices = &runtime.plan.order[final_checkpoint.min(runtime.plan.order.len())..];
    let tail = if tail_indices.is_empty() {
        None
    } else {
        let tail_path = out_dir.join("tail.v2.bin");
        let (tail_pack, local_to_global_fid) =
            qs_exact::extract_tree_indices_remapped(&runtime.pack, tail_indices)?;
        qs_exact::save_qs_pack_v2(&tail_path, &tail_pack)?;
        Some(ExpV1TailMeta {
            pack_file: file_name_string(&tail_path),
            local_to_global_fid,
            tree_count: tail_indices.len(),
        })
    };

    let manifest = ExpV1Manifest {
        format: EXP_V1_FORMAT.to_string(),
        compiled_hot_pack_file: file_name_string(&compiled_path),
        hot_layout_file: file_name_string(&hot_layout_path),
        hot_stage_plan_file: file_name_string(&hot_stage_plan_path),
        final_checkpoint,
        late_segments,
        tail,
    };
    let manifest_path = out_dir.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("write {}", manifest_path.display()))?;
    Ok(manifest_path)
}

pub(crate) fn load_from_manifest_dir(exp_dir: &PathBuf) -> Result<Option<L2ExpV1Runtime>> {
    let manifest_path = exp_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Ok(None);
    }
    let txt = fs::read_to_string(&manifest_path)
        .with_context(|| format!("read {}", manifest_path.display()))?;
    let manifest: ExpV1Manifest =
        serde_json::from_str(&txt).with_context(|| format!("parse {}", manifest_path.display()))?;
    if manifest.format != EXP_V1_FORMAT {
        bail!("unsupported l2_exp_v1 manifest format: {}", manifest.format);
    }

    let compiled_hot_pack =
        load_compiled_hot_prefix_pack(&exp_dir.join(&manifest.compiled_hot_pack_file))?;
    let hot_checkpoint_layout =
        load_hot_checkpoint_layout(&exp_dir.join(&manifest.hot_layout_file))?;
    let hot_stage_plan = load_hot_stage_plan(&exp_dir.join(&manifest.hot_stage_plan_file))?;

    let mut late_segments = Vec::with_capacity(manifest.late_segments.len());
    for seg in manifest.late_segments {
        let pack = qs_exact::load_qs_pack(&exp_dir.join(&seg.pack_file))?;
        let all_tree_indices = (0..pack.n_trees()).map(|i| i as u32).collect();
        late_segments.push(ExpV1LateSegment {
            checkpoint: seg.checkpoint,
            global_cp_idx: seg.global_cp_idx,
            pack,
            local_to_global_fid: seg.local_to_global_fid,
            all_tree_indices,
        });
    }

    let tail = manifest
        .tail
        .map(|tail| {
            let pack = qs_exact::load_qs_pack(&exp_dir.join(&tail.pack_file))?;
            let all_tree_indices = (0..pack.n_trees()).map(|i| i as u32).collect();
            Ok::<_, anyhow::Error>(ExpV1TailRuntime {
                pack,
                local_to_global_fid: tail.local_to_global_fid,
                all_tree_indices,
                tree_count: tail.tree_count,
            })
        })
        .transpose()?;

    Ok(Some(L2ExpV1Runtime {
        compiled_hot_pack,
        hot_checkpoint_layout,
        hot_stage_plan,
        late_segments,
        tail,
        final_checkpoint: manifest.final_checkpoint,
    }))
}

pub(crate) fn load_from_resolved_bundle(
    resolved: &crate::ResolvedPrefixCalBundle,
) -> Result<Option<L2ExpV1Runtime>> {
    let exp_dir = resolved
        .qs_pack
        .parent()
        .map(|p| p.join("l2_exp_v1"))
        .unwrap_or_else(|| PathBuf::from("l2_exp_v1"));
    load_from_manifest_dir(&exp_dir)
}

pub(crate) fn predict_nomiss(
    exp: &L2ExpV1Runtime,
    base_runtime: &LoadedPrefixRuntime,
    feat: &[f32],
    row_tau: f32,
    row_fold: i32,
    hot_buf: &mut ExpHotStageBuf,
    local_ranks: &mut Vec<u8>,
) -> Result<OnlineL2Output> {
    let atlas = base_runtime
        .atlas_certifier
        .as_ref()
        .context("l2_exp_v1 requires atlas certifier")?;
    let calibration = &base_runtime.calibration;
    let mut score = base_runtime.pack.base_score;
    let mut trees_used = 0usize;
    let atlas_tau = atlas_tau_bin(&atlas.tau_edges, row_tau);
    let calibration_tau = if let Some(first) = calibration.tables.first() {
        if first.tau_edges.len() >= 2 {
            crate::prefix_gap_bin(&first.tau_edges, row_tau)
        } else {
            0usize
        }
    } else {
        0usize
    };
    let mut last_checkpoint_score = score;
    let mut score_at_64 = None::<f32>;

    hot_buf.ensure_len(exp.compiled_hot_pack.n_hot_features);
    let mut hot_start = 0usize;
    for (cp_slot, checkpoint) in exp.hot_stage_plan.checkpoints.iter().copied().enumerate() {
        let feat_off = exp.hot_stage_plan.feature_offsets[cp_slot] as usize;
        let feat_end = exp.hot_stage_plan.feature_offsets[cp_slot + 1] as usize;
        if feat_end > feat_off {
            hot_buf.stage_range(
                &exp.compiled_hot_pack,
                feat,
                &exp.hot_stage_plan.hot_feature_ids[feat_off..feat_end],
            );
        }
        score = run_hot_tree_range_nomiss(
            &exp.compiled_hot_pack,
            hot_buf.values.as_slice(),
            hot_start,
            checkpoint,
            score,
        )?;
        trees_used = checkpoint;
        let global_cp_idx = exp.hot_checkpoint_layout.hot_indices[cp_slot];
        let delta_prev = score - last_checkpoint_score;
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
            &calibration.tables[global_cp_idx],
            &atlas.checkpoints[global_cp_idx],
            prefix_fold_slot(&calibration.tables[global_cp_idx], row_fold),
            row_tau,
            atlas_tau,
            calibration_tau,
            score,
            delta_prev,
            delta_from_64,
        ) {
            return Ok(OnlineL2Output {
                score: route_score,
                reject: is_reject,
                used_fallback: false,
                trees_used: checkpoint as i32,
            });
        }
        hot_start = checkpoint;
    }

    for segment in &exp.late_segments {
        ensure_local_ranks(local_ranks, segment.pack.n_features());
        qs_exact::quantize_feature_subset_nomiss_mapped(
            &segment.pack,
            feat,
            local_ranks.as_mut_slice(),
            &segment.local_to_global_fid,
        );
        let (seg_score, _, _) = qs_exact::score_tree_indices_from_quantized_nomiss(
            &segment.pack,
            &segment.all_tree_indices,
            local_ranks.as_slice(),
        )?;
        score += seg_score;
        trees_used = segment.checkpoint;
        let delta_prev = score - last_checkpoint_score;
        let delta_from_64 = if let Some(score64) = score_at_64 {
            score - score64
        } else {
            0.0
        };
        last_checkpoint_score = score;
        if let Some((is_reject, route_score)) = maybe_prefix_certify_atlas_fast(
            &calibration.tables[segment.global_cp_idx],
            &atlas.checkpoints[segment.global_cp_idx],
            prefix_fold_slot(&calibration.tables[segment.global_cp_idx], row_fold),
            row_tau,
            atlas_tau,
            calibration_tau,
            score,
            delta_prev,
            delta_from_64,
        ) {
            return Ok(OnlineL2Output {
                score: route_score,
                reject: is_reject,
                used_fallback: false,
                trees_used: segment.checkpoint as i32,
            });
        }
    }

    if let Some(tail) = exp.tail.as_ref() {
        ensure_local_ranks(local_ranks, tail.pack.n_features());
        qs_exact::quantize_feature_subset_nomiss_mapped(
            &tail.pack,
            feat,
            local_ranks.as_mut_slice(),
            &tail.local_to_global_fid,
        );
        let (tail_score, _, _) = qs_exact::score_tree_indices_from_quantized_nomiss(
            &tail.pack,
            &tail.all_tree_indices,
            local_ranks.as_slice(),
        )?;
        score += tail_score;
        trees_used += tail.tree_count;
    }

    Ok(OnlineL2Output {
        score,
        reject: score >= row_tau,
        used_fallback: true,
        trees_used: trees_used as i32,
    })
}

fn file_name_string(path: &PathBuf) -> String {
    path.file_name()
        .and_then(|x| x.to_str())
        .map(|x| x.to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string())
}

fn load_hot_stage_plan(path: &PathBuf) -> Result<HotStagePlan> {
    let txt = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let file: HotStagePlanFile =
        serde_json::from_str(&txt).with_context(|| format!("parse {}", path.display()))?;
    if file.feature_offsets.len() != file.checkpoints.len() + 1 {
        bail!("invalid hot stage plan offsets len");
    }
    Ok(HotStagePlan {
        checkpoints: file.checkpoints,
        feature_offsets: file.feature_offsets,
        hot_feature_ids: file.hot_feature_ids,
    })
}

fn build_hot_stage_plan(
    compiled: &HotCompiledPrefixPack,
    checkpoints: &[usize],
) -> Result<HotStagePlanFile> {
    if checkpoints.is_empty() {
        bail!("hot stage checkpoints must not be empty");
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
        for tree_idx in tree_start..checkpoint {
            collect_tree_hot_features(compiled, tree_idx, &mut hot_seen, &mut hot_feature_ids)?;
        }
        feature_offsets.push(hot_feature_ids.len() as u32);
        tree_start = checkpoint;
    }
    Ok(HotStagePlanFile {
        checkpoints: checkpoints.to_vec(),
        feature_offsets,
        hot_feature_ids,
    })
}

fn collect_tree_hot_features(
    compiled: &HotCompiledPrefixPack,
    tree_idx: usize,
    hot_seen: &mut [bool],
    out: &mut Vec<u16>,
) -> Result<()> {
    let mut stack = vec![*compiled
        .tree_roots
        .get(tree_idx)
        .context("tree root missing")? as usize];
    let mut node_seen = vec![false; compiled.nodes.len()];
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
    Ok(())
}

#[inline(always)]
fn ensure_local_ranks(buf: &mut Vec<u8>, need: usize) {
    if buf.len() != need {
        buf.resize(need, 0);
    } else {
        buf.fill(0);
    }
}

#[inline(always)]
fn run_hot_tree_range_nomiss(
    compiled: &HotCompiledPrefixPack,
    hot_feat: &[f32],
    start_tree: usize,
    end_tree: usize,
    mut score: f32,
) -> Result<f32> {
    for pos in start_tree..end_tree {
        let mut idx = compiled.tree_roots[pos] as usize;
        loop {
            let node = compiled.nodes[idx];
            if node.is_leaf() {
                score += node.leaf;
                break;
            }
            let x = hot_feat[node.hot_fidx()];
            idx = if x < node.thr {
                node.left as usize
            } else {
                node.right as usize
            };
        }
    }
    Ok(score)
}
