use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::path::PathBuf;

use crate::{active_threads, install_in_pool, le_f32, le_u32, le_u64, FeatureBatch};

pub const MAGIC_QS: &[u8] = b"L2QSv1\0";

#[derive(Debug)]
pub struct QsPack {
    pub n_features: usize,
    pub n_trees: usize,
    pub n_blocks: usize,
    pub base_score: f32,
    cut_offsets: Vec<u32>,
    cuts: Vec<f32>,
    tree_hdrs: Vec<QsTreeHdr>,
    block_hdrs: Vec<QsBlockHdr>,
    luts: Vec<u8>,
    masks: Vec<Mask128>,
    leaf_values: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct QsTreeHdr {
    block_off: u32,
    block_cnt: u16,
    leaf_cnt: u16,
    leaf_val_off: u32,
    init_lo: u64,
    init_hi: u64,
}

#[derive(Clone, Copy, Debug)]
struct QsBlockHdr {
    fid: u8,
    bucket_cnt: u8,
    miss_bucket: u8,
    lut_off: u32,
    mask_off: u32,
}

#[derive(Clone, Copy, Debug)]
struct Mask128 {
    lo: u64,
    hi: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QsAgg {
    pub total_block_evals: u64,
    pub resolved_early_trees: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QsIntervalRow {
    pub lower: f32,
    pub upper: f32,
    pub block_evals: u64,
    pub resolved_early_trees: u64,
}

#[derive(Clone, Debug, Default)]
pub struct QsPrefixRow {
    pub checkpoint_scores: Vec<f32>,
    pub checkpoint_block_evals: Vec<u64>,
    pub checkpoint_resolved_early_trees: Vec<u64>,
    pub resolved_early_trees: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct QsPrefixDecisionRow {
    pub prefix_score: f32,
    pub trees_used: usize,
    pub block_evals: u64,
    pub resolved_early_trees: u64,
}

pub fn load_qs_pack(path: &PathBuf) -> Result<QsPack> {
    let file =
        File::open(path).with_context(|| format!("open qs pack failed: {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("mmap qs pack failed: {}", path.display()))?;
    let buf = &mmap[..];
    if buf.len() < MAGIC_QS.len() + 16 + 56 {
        bail!("qs pack too short");
    }
    if &buf[0..MAGIC_QS.len()] != MAGIC_QS {
        bail!("invalid qs pack magic");
    }
    let mut off = MAGIC_QS.len();
    let n_features = le_u32(&buf, &mut off)? as usize;
    let n_trees = le_u32(&buf, &mut off)? as usize;
    let n_blocks = le_u32(&buf, &mut off)? as usize;
    let base_score = le_f32(&buf, &mut off)?;
    let cut_offsets_off = le_u64(&buf, &mut off)? as usize;
    let cuts_off = le_u64(&buf, &mut off)? as usize;
    let tree_hdrs_off = le_u64(&buf, &mut off)? as usize;
    let block_hdrs_off = le_u64(&buf, &mut off)? as usize;
    let luts_off = le_u64(&buf, &mut off)? as usize;
    let masks_off = le_u64(&buf, &mut off)? as usize;
    let leafs_off = le_u64(&buf, &mut off)? as usize;

    if cut_offsets_off != off {
        bail!("qs pack cut_offsets offset mismatch");
    }

    let mut cut_offsets = Vec::with_capacity(n_features + 1);
    let mut pos = cut_offsets_off;
    for _ in 0..(n_features + 1) {
        cut_offsets.push(le_u32(&buf, &mut pos)?);
    }
    if pos != cuts_off {
        bail!("qs pack cuts offset mismatch");
    }
    let n_cuts = *cut_offsets.last().unwrap_or(&0) as usize;
    let mut cuts = Vec::with_capacity(n_cuts);
    for _ in 0..n_cuts {
        cuts.push(le_f32(&buf, &mut pos)?);
    }
    if pos != tree_hdrs_off {
        bail!("qs pack tree header offset mismatch");
    }

    let mut tree_hdrs = Vec::with_capacity(n_trees);
    pos = tree_hdrs_off;
    for _ in 0..n_trees {
        let block_off = le_u32(&buf, &mut pos)?;
        let block_cnt = le_u32(&buf, &mut pos)?; // read 4, split below
        let leaf_val_off = le_u32(&buf, &mut pos)?;
        let init_lo = le_u64(&buf, &mut pos)?;
        let init_hi = le_u64(&buf, &mut pos)?;
        tree_hdrs.push(QsTreeHdr {
            block_off,
            block_cnt: (block_cnt & 0xFFFF) as u16,
            leaf_cnt: (block_cnt >> 16) as u16,
            leaf_val_off,
            init_lo,
            init_hi,
        });
    }
    if pos != block_hdrs_off {
        bail!("qs pack block header offset mismatch");
    }

    let mut block_hdrs = Vec::with_capacity(n_blocks);
    pos = block_hdrs_off;
    for _ in 0..n_blocks {
        if pos + 12 > buf.len() {
            bail!("qs block header truncated");
        }
        let fid = buf[pos];
        let bucket_cnt = buf[pos + 1];
        let miss_bucket = buf[pos + 2];
        pos += 4;
        let lut_off = le_u32(&buf, &mut pos)?;
        let mask_off = le_u32(&buf, &mut pos)?;
        block_hdrs.push(QsBlockHdr {
            fid,
            bucket_cnt,
            miss_bucket,
            lut_off,
            mask_off,
        });
    }
    if pos != luts_off {
        bail!("qs pack lut offset mismatch");
    }

    let max_lut_end = block_hdrs
        .iter()
        .map(|b| b.lut_off as usize + 129usize)
        .max()
        .unwrap_or(0usize);
    let luts_len = masks_off.saturating_sub(luts_off);
    if luts_len < max_lut_end {
        bail!("qs pack lut section too short");
    }
    let luts = buf[luts_off..masks_off].to_vec();

    let max_mask_end = block_hdrs
        .iter()
        .map(|b| b.mask_off as usize + b.bucket_cnt as usize)
        .max()
        .unwrap_or(0usize);
    let mask_count = (leafs_off.saturating_sub(masks_off)) / 16usize;
    if mask_count < max_mask_end {
        bail!("qs pack mask section too short");
    }
    pos = masks_off;
    let mut masks = Vec::with_capacity(mask_count);
    for _ in 0..mask_count {
        let lo = le_u64(&buf, &mut pos)?;
        let hi = le_u64(&buf, &mut pos)?;
        masks.push(Mask128 { lo, hi });
    }
    if pos != leafs_off {
        bail!("qs pack leaf offset mismatch");
    }

    let total_leafs = tree_hdrs
        .iter()
        .map(|t| t.leaf_val_off as usize + t.leaf_cnt as usize)
        .max()
        .unwrap_or(0usize);
    let leaf_bytes = buf.len().saturating_sub(leafs_off);
    if leaf_bytes < total_leafs * 4 {
        bail!("qs pack leaf section too short");
    }
    let mut leaf_values = Vec::with_capacity(total_leafs);
    pos = leafs_off;
    for _ in 0..total_leafs {
        leaf_values.push(le_f32(&buf, &mut pos)?);
    }

    Ok(QsPack {
        n_features,
        n_trees,
        n_blocks,
        base_score,
        cut_offsets,
        cuts,
        tree_hdrs,
        block_hdrs,
        luts,
        masks,
        leaf_values,
    })
}

impl QsPack {
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn n_trees(&self) -> usize {
        self.n_trees
    }

    pub fn n_blocks(&self) -> usize {
        self.n_blocks
    }
}

#[inline(always)]
fn resolved(lo: u64, hi: u64) -> bool {
    (hi == 0 && lo != 0 && lo.is_power_of_two()) || (lo == 0 && hi != 0 && hi.is_power_of_two())
}

#[inline(always)]
fn decode_leaf(lo: u64, hi: u64) -> Result<usize> {
    if hi == 0 && lo != 0 && lo.is_power_of_two() {
        return Ok(lo.trailing_zeros() as usize);
    }
    if lo == 0 && hi != 0 && hi.is_power_of_two() {
        return Ok(64 + hi.trailing_zeros() as usize);
    }
    bail!("candidate leaf set did not resolve to a single bit");
}

#[inline(always)]
fn upper_bound(vals: &[f32], x: f32) -> u8 {
    if vals.len() <= 16 {
        let mut i = 0usize;
        while i < vals.len() {
            if vals[i] > x {
                return i as u8;
            }
            i += 1;
        }
        return vals.len() as u8;
    }
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
    lo as u8
}

#[inline(always)]
fn quantize_row(pack: &QsPack, feat: &[f32], ranks: &mut [u8], missing: &mut [u8]) {
    let mut fid = 0usize;
    while fid < pack.n_features {
        let x = feat[fid];
        if x.is_nan() {
            ranks[fid] = 0;
            missing[fid] = 1;
        } else {
            let st = pack.cut_offsets[fid] as usize;
            let ed = pack.cut_offsets[fid + 1] as usize;
            ranks[fid] = upper_bound(&pack.cuts[st..ed], x);
            missing[fid] = 0;
        }
        fid += 1;
    }
}

#[inline(always)]
pub fn quantize_into(pack: &QsPack, feat: &[f32], ranks: &mut [u8], missing: &mut [u8]) {
    quantize_row(pack, feat, ranks, missing);
}

#[inline(always)]
fn apply_block_nomiss(
    block: QsBlockHdr,
    luts: &[u8],
    masks: &[Mask128],
    ranks: &[u8],
    lo: u64,
    hi: u64,
) -> (u64, u64) {
    let fid = block.fid as usize;
    let rank = unsafe { *ranks.get_unchecked(fid) as usize };
    let bucket = unsafe { *luts.get_unchecked(block.lut_off as usize + rank) as usize };
    let mask = unsafe { *masks.get_unchecked(block.mask_off as usize + bucket) };
    (lo & mask.lo, hi & mask.hi)
}

#[inline(always)]
fn popcnt128(lo: u64, hi: u64) -> u32 {
    lo.count_ones() + hi.count_ones()
}

pub fn reorder_front_blocks_sampled_nomiss(
    pack: &mut QsPack,
    sample_ranks: &[Vec<u8>],
    start_tree_idx: usize,
    end_tree_idx: usize,
    front_blocks: usize,
) -> Result<()> {
    if sample_ranks.is_empty() || front_blocks == 0 || start_tree_idx >= end_tree_idx {
        return Ok(());
    }

    let tree_end = end_tree_idx.min(pack.n_trees);
    for tree_idx in start_tree_idx..tree_end {
        let tree = pack.tree_hdrs[tree_idx];
        let block_start = tree.block_off as usize;
        let block_end = block_start + tree.block_cnt as usize;
        let block_len = block_end - block_start;
        if block_len <= 1 {
            continue;
        }

        let choose = front_blocks.min(block_len);
        let original = pack.block_hdrs[block_start..block_end].to_vec();
        let mut remaining: Vec<usize> = (0..block_len).collect();
        let mut chosen = Vec::with_capacity(choose);
        let mut states = vec![(tree.init_lo, tree.init_hi); sample_ranks.len()];

        for _ in 0..choose {
            let mut best_pos = 0usize;
            let mut best_score = i64::MIN;
            for (cand_pos, &block_pos) in remaining.iter().enumerate() {
                let block = original[block_pos];
                let mut score = 0i64;
                for (sample_idx, ranks) in sample_ranks.iter().enumerate() {
                    let (cur_lo, cur_hi) = states[sample_idx];
                    if resolved(cur_lo, cur_hi) {
                        continue;
                    }
                    let (next_lo, next_hi) =
                        apply_block_nomiss(block, &pack.luts, &pack.masks, ranks, cur_lo, cur_hi);
                    let cur_half = (cur_lo == 0) ^ (cur_hi == 0);
                    let next_half = (next_lo == 0) ^ (next_hi == 0);
                    let cur_pop = popcnt128(cur_lo, cur_hi) as i64;
                    let next_pop = popcnt128(next_lo, next_hi) as i64;
                    if resolved(next_lo, next_hi) {
                        score += 1_000_000;
                    } else if !cur_half && next_half {
                        score += 10_000;
                    }
                    score += cur_pop - next_pop;
                }
                if score > best_score {
                    best_score = score;
                    best_pos = cand_pos;
                }
            }

            let block_pos = remaining.remove(best_pos);
            let block = original[block_pos];
            for (sample_idx, ranks) in sample_ranks.iter().enumerate() {
                let (cur_lo, cur_hi) = states[sample_idx];
                if resolved(cur_lo, cur_hi) {
                    continue;
                }
                states[sample_idx] =
                    apply_block_nomiss(block, &pack.luts, &pack.masks, ranks, cur_lo, cur_hi);
            }
            chosen.push(block);
        }

        let mut out = Vec::with_capacity(block_len);
        out.extend_from_slice(&chosen);
        for block_pos in remaining {
            out.push(original[block_pos]);
        }
        pack.block_hdrs[block_start..block_end].copy_from_slice(&out);
    }

    Ok(())
}

#[inline(always)]
fn score_tree_quantized(
    pack: &QsPack,
    tree: &QsTreeHdr,
    ranks: &[u8],
    missing: &[u8],
) -> Result<(f32, u64, u64)> {
    let mut lo = tree.init_lo;
    let mut hi = tree.init_hi;
    let mut block_evals = 0u64;
    let mut resolved_early = 0u64;
    let block_start = tree.block_off as usize;
    let block_end = block_start + tree.block_cnt as usize;
    for block_idx in block_start..block_end {
        let block = &pack.block_hdrs[block_idx];
        let fid = block.fid as usize;
        let bucket = if missing[fid] != 0 {
            block.miss_bucket as usize
        } else {
            pack.luts[block.lut_off as usize + ranks[fid] as usize] as usize
        };
        let mask = pack.masks[block.mask_off as usize + bucket];
        lo &= mask.lo;
        hi &= mask.hi;
        block_evals += 1;
        if resolved(lo, hi) {
            resolved_early += 1;
            break;
        }
    }
    let leaf_idx = decode_leaf(lo, hi)?;
    if leaf_idx >= tree.leaf_cnt as usize {
        bail!(
            "decoded leaf {} outside tree leaf count {}",
            leaf_idx,
            tree.leaf_cnt
        );
    }
    Ok((
        pack.leaf_values[tree.leaf_val_off as usize + leaf_idx],
        block_evals,
        resolved_early,
    ))
}

#[inline(always)]
pub fn score_tree_indices_from_quantized(
    pack: &QsPack,
    tree_indices: &[u32],
    ranks: &[u8],
    missing: &[u8],
) -> Result<(f32, u64, u64)> {
    let mut score = 0.0f32;
    let mut block_evals = 0u64;
    let mut resolved_early = 0u64;
    for &tree_idx_u32 in tree_indices {
        let tree_idx = tree_idx_u32 as usize;
        if tree_idx >= pack.tree_hdrs.len() {
            bail!(
                "tree index {} outside qs pack tree count {}",
                tree_idx,
                pack.tree_hdrs.len()
            );
        }
        let (tree_score, tree_blocks, tree_resolved) =
            score_tree_quantized(pack, &pack.tree_hdrs[tree_idx], ranks, missing)?;
        score += tree_score;
        block_evals += tree_blocks;
        resolved_early += tree_resolved;
    }
    Ok((score, block_evals, resolved_early))
}

#[inline(always)]
fn score_row(
    pack: &QsPack,
    feat: &[f32],
    ranks: &mut [u8],
    missing: &mut [u8],
) -> Result<(f32, u64, u64)> {
    quantize_row(pack, feat, ranks, missing);
    let mut score = pack.base_score;
    let mut block_evals = 0u64;
    let mut resolved_early = 0u64;
    for tree in &pack.tree_hdrs {
        let mut lo = tree.init_lo;
        let mut hi = tree.init_hi;
        let block_start = tree.block_off as usize;
        let block_end = block_start + tree.block_cnt as usize;
        for block_idx in block_start..block_end {
            let block = &pack.block_hdrs[block_idx];
            let fid = block.fid as usize;
            let bucket = if missing[fid] != 0 {
                block.miss_bucket as usize
            } else {
                pack.luts[block.lut_off as usize + ranks[fid] as usize] as usize
            };
            let mask = pack.masks[block.mask_off as usize + bucket];
            lo &= mask.lo;
            hi &= mask.hi;
            block_evals += 1;
            if resolved(lo, hi) {
                resolved_early += 1;
                break;
            }
        }
        let leaf_idx = decode_leaf(lo, hi)?;
        if leaf_idx >= tree.leaf_cnt as usize {
            bail!(
                "decoded leaf {} outside tree leaf count {}",
                leaf_idx,
                tree.leaf_cnt
            );
        }
        score += pack.leaf_values[tree.leaf_val_off as usize + leaf_idx];
    }
    Ok((score, block_evals, resolved_early))
}

#[inline(always)]
fn mask_bounds(pack: &QsPack, tree: &QsTreeHdr, lo: u64, hi: u64) -> Result<(f32, f32)> {
    let leaf_values = &pack.leaf_values
        [tree.leaf_val_off as usize..tree.leaf_val_off as usize + tree.leaf_cnt as usize];
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut found = false;

    let mut lo_bits = lo;
    while lo_bits != 0 {
        let idx = lo_bits.trailing_zeros() as usize;
        min_val = min_val.min(leaf_values[idx]);
        max_val = max_val.max(leaf_values[idx]);
        found = true;
        lo_bits &= lo_bits - 1;
    }

    let mut hi_bits = hi;
    while hi_bits != 0 {
        let idx = 64 + hi_bits.trailing_zeros() as usize;
        if idx >= tree.leaf_cnt as usize {
            bail!(
                "decoded leaf {} outside tree leaf count {}",
                idx,
                tree.leaf_cnt
            );
        }
        min_val = min_val.min(leaf_values[idx]);
        max_val = max_val.max(leaf_values[idx]);
        found = true;
        hi_bits &= hi_bits - 1;
    }

    if !found {
        bail!("candidate leaf set resolved to empty mask");
    }
    Ok((min_val, max_val))
}

#[inline(always)]
pub fn interval_row(
    pack: &QsPack,
    feat: &[f32],
    ranks: &mut [u8],
    missing: &mut [u8],
) -> Result<QsIntervalRow> {
    quantize_row(pack, feat, ranks, missing);
    let mut lower = pack.base_score;
    let mut upper = pack.base_score;
    let mut block_evals = 0u64;
    let mut resolved_early = 0u64;
    for tree in &pack.tree_hdrs {
        let mut lo = tree.init_lo;
        let mut hi = tree.init_hi;
        let block_start = tree.block_off as usize;
        let block_end = block_start + tree.block_cnt as usize;
        for block_idx in block_start..block_end {
            let block = &pack.block_hdrs[block_idx];
            let fid = block.fid as usize;
            let bucket = if missing[fid] != 0 {
                block.miss_bucket as usize
            } else {
                pack.luts[block.lut_off as usize + ranks[fid] as usize] as usize
            };
            let mask = pack.masks[block.mask_off as usize + bucket];
            lo &= mask.lo;
            hi &= mask.hi;
            block_evals += 1;
            if resolved(lo, hi) {
                resolved_early += 1;
                break;
            }
        }
        let (tree_min, tree_max) = mask_bounds(pack, tree, lo, hi)?;
        lower += tree_min;
        upper += tree_max;
    }
    Ok(QsIntervalRow {
        lower,
        upper,
        block_evals,
        resolved_early_trees: resolved_early,
    })
}

#[inline(always)]
pub fn prefix_checkpoints_row_from(
    pack: &QsPack,
    feat: &[f32],
    ranks: &mut [u8],
    missing: &mut [u8],
    checkpoints: &[usize],
    start_tree_idx: usize,
    init_score: f32,
    init_block_evals: u64,
    init_resolved_early_trees: u64,
) -> Result<QsPrefixRow> {
    if checkpoints.is_empty() {
        bail!("prefix checkpoints must not be empty");
    }
    let mut prev = 0usize;
    for &cp in checkpoints {
        if cp == 0 || cp > pack.n_trees {
            bail!("invalid checkpoint {} for n_trees={}", cp, pack.n_trees);
        }
        if cp <= prev {
            bail!("checkpoints must be strictly increasing");
        }
        prev = cp;
    }

    quantize_row(pack, feat, ranks, missing);
    let max_checkpoint = *checkpoints.last().unwrap_or(&0usize);
    let mut score = init_score;
    let mut block_evals = init_block_evals;
    let mut resolved_early = init_resolved_early_trees;
    let mut checkpoint_scores = Vec::with_capacity(checkpoints.len());
    let mut checkpoint_block_evals = Vec::with_capacity(checkpoints.len());
    let mut checkpoint_resolved_early_trees = Vec::with_capacity(checkpoints.len());
    let mut next_checkpoint_idx = 0usize;

    if start_tree_idx > pack.n_trees {
        bail!(
            "start_tree_idx {} exceeds n_trees={}",
            start_tree_idx,
            pack.n_trees
        );
    }

    for (tree_idx, tree) in pack.tree_hdrs.iter().enumerate().skip(start_tree_idx) {
        if tree_idx >= max_checkpoint {
            break;
        }
        let mut lo = tree.init_lo;
        let mut hi = tree.init_hi;
        let block_start = tree.block_off as usize;
        let block_end = block_start + tree.block_cnt as usize;
        for block_idx in block_start..block_end {
            let block = &pack.block_hdrs[block_idx];
            let fid = block.fid as usize;
            let bucket = if missing[fid] != 0 {
                block.miss_bucket as usize
            } else {
                pack.luts[block.lut_off as usize + ranks[fid] as usize] as usize
            };
            let mask = pack.masks[block.mask_off as usize + bucket];
            lo &= mask.lo;
            hi &= mask.hi;
            block_evals += 1;
            if resolved(lo, hi) {
                resolved_early += 1;
                break;
            }
        }
        let leaf_idx = decode_leaf(lo, hi)?;
        if leaf_idx >= tree.leaf_cnt as usize {
            bail!(
                "decoded leaf {} outside tree leaf count {}",
                leaf_idx,
                tree.leaf_cnt
            );
        }
        score += pack.leaf_values[tree.leaf_val_off as usize + leaf_idx];

        let completed_trees = tree_idx + 1;
        if completed_trees == checkpoints[next_checkpoint_idx] {
            checkpoint_scores.push(score);
            checkpoint_block_evals.push(block_evals);
            checkpoint_resolved_early_trees.push(resolved_early);
            next_checkpoint_idx += 1;
            if next_checkpoint_idx >= checkpoints.len() {
                break;
            }
        }
    }

    if checkpoint_scores.len() != checkpoints.len() {
        bail!(
            "failed to reach all checkpoints: expected={} actual={}",
            checkpoints.len(),
            checkpoint_scores.len()
        );
    }

    Ok(QsPrefixRow {
        checkpoint_scores,
        checkpoint_block_evals,
        checkpoint_resolved_early_trees,
        resolved_early_trees: resolved_early,
    })
}

#[inline(always)]
pub fn prefix_checkpoints_row(
    pack: &QsPack,
    feat: &[f32],
    ranks: &mut [u8],
    missing: &mut [u8],
    checkpoints: &[usize],
) -> Result<QsPrefixRow> {
    prefix_checkpoints_row_from(
        pack,
        feat,
        ranks,
        missing,
        checkpoints,
        0,
        pack.base_score,
        0,
        0,
    )
}

#[inline(always)]
pub fn prefix_until_from<F>(
    pack: &QsPack,
    feat: &[f32],
    ranks: &mut [u8],
    missing: &mut [u8],
    checkpoints: &[usize],
    start_tree_idx: usize,
    init_score: f32,
    init_block_evals: u64,
    init_resolved_early_trees: u64,
    mut on_checkpoint: F,
) -> Result<QsPrefixDecisionRow>
where
    F: FnMut(usize, f32, u64, u64) -> bool,
{
    if checkpoints.is_empty() {
        bail!("prefix checkpoints must not be empty");
    }
    let mut prev = 0usize;
    for &cp in checkpoints {
        if cp == 0 || cp > pack.n_trees {
            bail!("invalid checkpoint {} for n_trees={}", cp, pack.n_trees);
        }
        if cp <= prev {
            bail!("checkpoints must be strictly increasing");
        }
        prev = cp;
    }

    quantize_row(pack, feat, ranks, missing);
    let max_checkpoint = *checkpoints.last().unwrap_or(&0usize);
    let mut score = init_score;
    let mut block_evals = init_block_evals;
    let mut resolved_early = init_resolved_early_trees;
    let mut next_checkpoint_idx = 0usize;

    if start_tree_idx > pack.n_trees {
        bail!(
            "start_tree_idx {} exceeds n_trees={}",
            start_tree_idx,
            pack.n_trees
        );
    }

    for (tree_idx, tree) in pack.tree_hdrs.iter().enumerate().skip(start_tree_idx) {
        if tree_idx >= max_checkpoint {
            break;
        }
        let mut lo = tree.init_lo;
        let mut hi = tree.init_hi;
        let block_start = tree.block_off as usize;
        let block_end = block_start + tree.block_cnt as usize;
        for block_idx in block_start..block_end {
            let block = &pack.block_hdrs[block_idx];
            let fid = block.fid as usize;
            let bucket = if missing[fid] != 0 {
                block.miss_bucket as usize
            } else {
                pack.luts[block.lut_off as usize + ranks[fid] as usize] as usize
            };
            let mask = pack.masks[block.mask_off as usize + bucket];
            lo &= mask.lo;
            hi &= mask.hi;
            block_evals += 1;
            if resolved(lo, hi) {
                resolved_early += 1;
                break;
            }
        }
        let leaf_idx = decode_leaf(lo, hi)?;
        if leaf_idx >= tree.leaf_cnt as usize {
            bail!(
                "decoded leaf {} outside tree leaf count {}",
                leaf_idx,
                tree.leaf_cnt
            );
        }
        score += pack.leaf_values[tree.leaf_val_off as usize + leaf_idx];

        let completed_trees = tree_idx + 1;
        if completed_trees == checkpoints[next_checkpoint_idx] {
            if on_checkpoint(completed_trees, score, block_evals, resolved_early) {
                return Ok(QsPrefixDecisionRow {
                    prefix_score: score,
                    trees_used: completed_trees,
                    block_evals,
                    resolved_early_trees: resolved_early,
                });
            }
            next_checkpoint_idx += 1;
            if next_checkpoint_idx >= checkpoints.len() {
                return Ok(QsPrefixDecisionRow {
                    prefix_score: score,
                    trees_used: completed_trees,
                    block_evals,
                    resolved_early_trees: resolved_early,
                });
            }
        }
    }

    bail!("failed to reach final prefix checkpoint");
}

#[inline(always)]
pub fn prefix_until<F>(
    pack: &QsPack,
    feat: &[f32],
    ranks: &mut [u8],
    missing: &mut [u8],
    checkpoints: &[usize],
    on_checkpoint: F,
) -> Result<QsPrefixDecisionRow>
where
    F: FnMut(usize, f32, u64, u64) -> bool,
{
    prefix_until_from(
        pack,
        feat,
        ranks,
        missing,
        checkpoints,
        0,
        pack.base_score,
        0,
        0,
        on_checkpoint,
    )
}

pub fn run_qs_exact(
    pack: &QsPack,
    batch: &FeatureBatch,
    n: usize,
    threads: usize,
    chunk_rows: usize,
    parallel_min_rows: usize,
    thread_pool: Option<&rayon::ThreadPool>,
) -> Result<(Vec<f32>, bool, usize, QsAgg)> {
    if batch.n_cols != pack.n_features {
        bail!(
            "feature count mismatch: feat_bin={} qs_pack={}",
            batch.n_cols,
            pack.n_features
        );
    }
    let mut scores = vec![0.0f32; n];
    let par_threads = active_threads(threads);
    let parallel_enabled = par_threads > 1 && n >= parallel_min_rows;

    let work = || -> Result<QsAgg> {
        if parallel_enabled {
            scores
                .par_chunks_mut(chunk_rows.max(1))
                .enumerate()
                .map(|(chunk_idx, score_chunk)| -> Result<QsAgg> {
                    let start = chunk_idx * chunk_rows.max(1);
                    let mut ranks = vec![0u8; pack.n_features];
                    let mut missing = vec![0u8; pack.n_features];
                    let mut agg = QsAgg::default();
                    for local_idx in 0..score_chunk.len() {
                        let row_idx = start + local_idx;
                        let feat = batch.row(row_idx);
                        let (score, blocks, resolved) =
                            score_row(pack, feat, &mut ranks, &mut missing)?;
                        score_chunk[local_idx] = score;
                        agg.total_block_evals += blocks;
                        agg.resolved_early_trees += resolved;
                    }
                    Ok(agg)
                })
                .try_reduce(QsAgg::default, |mut a, b| {
                    a.total_block_evals += b.total_block_evals;
                    a.resolved_early_trees += b.resolved_early_trees;
                    Ok(a)
                })
        } else {
            let mut ranks = vec![0u8; pack.n_features];
            let mut missing = vec![0u8; pack.n_features];
            let mut agg = QsAgg::default();
            for row_idx in 0..n {
                let feat = batch.row(row_idx);
                let (score, blocks, resolved) = score_row(pack, feat, &mut ranks, &mut missing)?;
                scores[row_idx] = score;
                agg.total_block_evals += blocks;
                agg.resolved_early_trees += resolved;
            }
            Ok(agg)
        }
    };
    let agg = install_in_pool(thread_pool, work)?;
    Ok((scores, parallel_enabled, par_threads, agg))
}
