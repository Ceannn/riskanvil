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

include!("standalone_cli.rs");
include!("standalone_types.rs");

include!("standalone_loaders.rs");

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
    let packet_summary_path =
        resolve_relative_compat(&packet_manifest_path, &raw.packet_summary_tsv);
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
        bail!("missing required anchor-rescue field: variant_key or selected_exact_variant");
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
    let rescue_manifest: PrefixAnchorRescueManifestFile = serde_json::from_str(&rescue_txt)
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
    let router: PrefixOrderRouterFile = serde_json::from_str(&txt)
        .with_context(|| format!("parse {}", order_router_path.display()))?;
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
        bail!("missing required V4 prefix-cal field: variant_key or selected_exact_variant");
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
    Ok((
        router.tau_edges,
        routes,
        selected_variant,
        telemetry_schema_version,
    ))
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
    lookup_prefix_bands_dense(
        table,
        prefix_fold_slot(table, fold_id),
        None,
        guard_lo,
        guard_hi,
    )
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
        if table.tau_bin_count == 0 {
            None
        } else {
            Some(tau_bin)
        },
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
fn atlas_cluster_id_16_scalar(centroids: &LoadedPrefixAtlasCentroids16, feat: [f32; 4]) -> usize {
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
    let (ref_hi, rej_lo) =
        lookup_prefix_bands_with_tau_bin_slot(table, row_fold_slot, row_calibration_tau_bin, gap);
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
        checkpoint_work_evals: if trace_enabled {
            vec![0u32; cp_len]
        } else {
            Vec::new()
        },
        checkpoint_deltas: if trace_enabled {
            vec![0.0f32; cp_len]
        } else {
            Vec::new()
        },
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

    let mut handle_checkpoint = |global_cp_idx: usize,
                                 checkpoint: usize,
                                 score: f32,
                                 work_evals: u64,
                                 resolved_early: u64| {
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
            out.checkpoint_work_evals[global_cp_idx] = work_evals.min(u32::MAX as u64) as u32;
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
        checkpoint_work_evals: if trace_enabled {
            vec![0u32; cp_len]
        } else {
            Vec::new()
        },
        checkpoint_deltas: if trace_enabled {
            vec![0.0f32; cp_len]
        } else {
            Vec::new()
        },
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

    let mut handle_checkpoint = |global_cp_idx: usize,
                                 checkpoint: usize,
                                 score: f32,
                                 work_evals: u64,
                                 resolved_early: u64| {
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
            out.checkpoint_work_evals[global_cp_idx] = work_evals.min(u32::MAX as u64) as u32;
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

    let mut handle_checkpoint = |global_cp_idx: usize,
                                 checkpoint: usize,
                                 score: f32,
                                 work_evals: u64,
                                 resolved_early: u64| {
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
        let mut late_pos = 0usize;
        let late_row = if nan_free {
            if let Some(prog) = runtime.late_iftree_program.as_ref() {
                unsafe {
                    late_iftree_prefix_until_nomiss(
                        prog,
                        feat,
                        &hot_checkpoint_layout.late_values,
                        score,
                        work_evals,
                        resolved_early,
                        |checkpoint, prefix_score, qs_work, resolved| {
                            let global_cp_idx = hot_checkpoint_layout.late_indices[late_pos];
                            late_pos += 1;
                            handle_checkpoint(
                                global_cp_idx,
                                checkpoint,
                                prefix_score,
                                qs_work,
                                resolved,
                            )
                        },
                    )
                }?
            } else {
                let hot_limit = compiled_hot_pack.n_trees.min(runtime.pack.n_trees());
                qs_exact::prefix_until_from(
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
                        handle_checkpoint(
                            global_cp_idx,
                            checkpoint,
                            prefix_score,
                            qs_work,
                            resolved,
                        )
                    },
                )?
            }
        } else {
            let hot_limit = compiled_hot_pack.n_trees.min(runtime.pack.n_trees());
            qs_exact::prefix_until_from(
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
            )?
        };
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
        out.fallback_entry_checkpoint = *runtime.calibration.checkpoints.last().unwrap_or(&0usize);
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
    let tau_bin =
        mlp_tau_bin(&router.tau_edges, row_tau).min(router.tau_edges.len().saturating_sub(2));
    let gap = row_tau - prefix_score;
    let gap_bin =
        prefix_gap_bin(&router.gap_edges, gap).min(router.gap_edges.len().saturating_sub(2));
    let shadow_side = if prefix_score >= row_tau { 1 } else { -1 };
    if let Some(action) =
        router
            .actions
            .get(&(anchor_checkpoint, tau_bin, row_fold, shadow_side, gap_bin))
    {
        return *action;
    }
    if let Some(action) = router
        .defaults
        .get(&(anchor_checkpoint, tau_bin, shadow_side))
    {
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

unsafe fn late_iftree_prefix_until_nomiss<F>(
    prog: &LateIfTreeProgram,
    feat: &[f32],
    checkpoints: &[usize],
    init_score: f32,
    init_node_evals: u64,
    init_resolved_early_trees: u64,
    mut on_checkpoint: F,
) -> Result<qs_exact::QsPrefixDecisionRow>
where
    F: FnMut(usize, f32, u64, u64) -> bool,
{
    if checkpoints.is_empty() {
        bail!("late if-tree checkpoints must not be empty");
    }
    debug_assert!(checkpoints[0] > prog.start_tree_idx);
    debug_assert!(checkpoints[checkpoints.len() - 1] <= prog.start_tree_idx + prog.n_trees);

    let mut score = init_score;
    let mut node_evals = init_node_evals;
    let resolved_early = init_resolved_early_trees;
    let mut next_checkpoint_idx = 0usize;

    for pos in 0..prog.n_trees {
        let base = *prog.tree_node_offs.get_unchecked(pos) as usize;
        let mut idx = 0usize;
        loop {
            let node = *prog.nodes.get_unchecked(base + idx);
            if node.is_leaf() {
                score += node.leaf;
                break;
            }
            let x = *feat.get_unchecked(node.fidx());
            node_evals += 1;
            idx = if x < node.thr {
                node.left as usize
            } else {
                node.right as usize
            };
        }
        let completed_trees = prog.start_tree_idx + pos + 1;
        if completed_trees == checkpoints[next_checkpoint_idx] {
            if on_checkpoint(completed_trees, score, node_evals, resolved_early) {
                return Ok(qs_exact::QsPrefixDecisionRow {
                    prefix_score: score,
                    trees_used: completed_trees,
                    block_evals: node_evals,
                    resolved_early_trees: resolved_early,
                });
            }
            next_checkpoint_idx += 1;
            if next_checkpoint_idx >= checkpoints.len() {
                return Ok(qs_exact::QsPrefixDecisionRow {
                    prefix_score: score,
                    trees_used: completed_trees,
                    block_evals: node_evals,
                    resolved_early_trees: resolved_early,
                });
            }
        }
    }

    bail!("failed to reach final late if-tree checkpoint");
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
        checkpoint_work_evals: if trace_enabled {
            vec![0u32; cp_len]
        } else {
            Vec::new()
        },
        checkpoint_deltas: if trace_enabled {
            vec![0.0f32; cp_len]
        } else {
            Vec::new()
        },
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
        let delta_bin = prefix_gap_bin(&scheduler.policy.delta_edges_by_step[step], prev_delta);
        let ranking =
            lookup_packet_policy_ranking(&scheduler.policy, step, tau_bin, gap_bin, delta_bin);
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
            out.checkpoint_work_evals[step] = work_evals.min(u32::MAX as u64) as u32;
            out.checkpoint_deltas[step] = packet_score;
            out.checkpoint_resolved_early[step] = resolved_early_trees.min(u32::MAX as u64) as u32;
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
        let (remaining_score, _, _) =
            qs_exact::score_tree_indices_from_quantized(&runtime.pack, &remaining, ranks, missing)?;
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

fn load_dispatch_slot(
    slot: i32,
    label: String,
    model_dir: &PathBuf,
    threshold_mode: DispatchThresholdMode,
) -> Result<LoadedDispatchSlot> {
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
        InferMode::RouteFast | InferMode::RouteExactReordered | InferMode::L2RouteExactReordered
    )
}

fn is_l2_route_mode(mode: InferMode) -> bool {
    matches!(
        mode,
        InferMode::L2RouteExactReordered | InferMode::L2RouteApprox
    )
}

fn is_approx_mode(mode: InferMode) -> bool {
    matches!(mode, InferMode::RouteApprox | InferMode::L2RouteApprox)
}

fn score_col_name(mode: InferMode) -> &'static str {
    match mode {
        InferMode::MarginExact | InferMode::MarginExactReordered => "l1_score",
        InferMode::RouteFast | InferMode::RouteExactReordered | InferMode::RouteApprox => {
            "prefix_margin"
        }
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

include!("standalone_commands.rs");

include!("standalone_runtime_api.rs");
