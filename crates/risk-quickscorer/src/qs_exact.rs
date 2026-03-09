use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use std::fs::File;
use std::path::PathBuf;

use crate::{active_threads, install_in_pool, le_f32, le_u32, le_u64, FeatureBatch};

pub const MAGIC_QS: &[u8] = b"L2QSv1\0";
pub const MAGIC_QS_V2: &[u8] = b"L2QSv2\0";

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
    tree_block_off: Vec<u32>,
    tree_block_cnt: Vec<u16>,
    tree_leaf_cnt: Vec<u16>,
    tree_leaf_val_off: Vec<u32>,
    tree_init_lo: Vec<u64>,
    tree_init_hi: Vec<u64>,
    block_fid: Vec<u8>,
    block_bucket_cnt: Vec<u8>,
    block_miss_bucket: Vec<u8>,
    block_lut_off: Vec<u32>,
    block_mask_off: Vec<u32>,
    luts: Vec<u8>,
    masks: Vec<Mask128>,
    masks_lo: Vec<u64>,
    masks_hi: Vec<u64>,
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
    let magic = &buf[0..MAGIC_QS.len()];
    if magic != MAGIC_QS && magic != MAGIC_QS_V2 {
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
    let mut tree_block_off = Vec::with_capacity(n_trees);
    let mut tree_block_cnt = Vec::with_capacity(n_trees);
    let mut tree_leaf_cnt = Vec::with_capacity(n_trees);
    let mut tree_leaf_val_off = Vec::with_capacity(n_trees);
    let mut tree_init_lo = Vec::with_capacity(n_trees);
    let mut tree_init_hi = Vec::with_capacity(n_trees);
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
        tree_block_off.push(block_off);
        tree_block_cnt.push((block_cnt & 0xFFFF) as u16);
        tree_leaf_cnt.push((block_cnt >> 16) as u16);
        tree_leaf_val_off.push(leaf_val_off);
        tree_init_lo.push(init_lo);
        tree_init_hi.push(init_hi);
    }
    if pos != block_hdrs_off {
        bail!("qs pack block header offset mismatch");
    }

    let mut block_hdrs = Vec::with_capacity(n_blocks);
    let mut block_fid = Vec::with_capacity(n_blocks);
    let mut block_bucket_cnt = Vec::with_capacity(n_blocks);
    let mut block_miss_bucket = Vec::with_capacity(n_blocks);
    let mut block_lut_off = Vec::with_capacity(n_blocks);
    let mut block_mask_off = Vec::with_capacity(n_blocks);
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
        block_fid.push(fid);
        block_bucket_cnt.push(bucket_cnt);
        block_miss_bucket.push(miss_bucket);
        block_lut_off.push(lut_off);
        block_mask_off.push(mask_off);
    }
    if pos != luts_off {
        bail!("qs pack lut offset mismatch");
    }

    let max_lut_end = block_lut_off
        .iter()
        .map(|&off| off as usize + 129usize)
        .max()
        .unwrap_or(0usize);
    let luts_len = masks_off.saturating_sub(luts_off);
    if luts_len < max_lut_end {
        bail!("qs pack lut section too short");
    }
    let luts = buf[luts_off..masks_off].to_vec();

    let max_mask_end = block_mask_off
        .iter()
        .zip(block_bucket_cnt.iter())
        .map(|(&mask_off, &bucket_cnt)| mask_off as usize + bucket_cnt as usize)
        .max()
        .unwrap_or(0usize);
    let mask_count = (leafs_off.saturating_sub(masks_off)) / 16usize;
    if mask_count < max_mask_end {
        bail!("qs pack mask section too short");
    }
    pos = masks_off;
    let mut masks = Vec::with_capacity(mask_count);
    let mut masks_lo = Vec::with_capacity(mask_count);
    let mut masks_hi = Vec::with_capacity(mask_count);
    for _ in 0..mask_count {
        let lo = le_u64(&buf, &mut pos)?;
        let hi = le_u64(&buf, &mut pos)?;
        masks.push(Mask128 { lo, hi });
        masks_lo.push(lo);
        masks_hi.push(hi);
    }
    if pos != leafs_off {
        bail!("qs pack leaf offset mismatch");
    }

    let total_leafs = tree_leaf_val_off
        .iter()
        .zip(tree_leaf_cnt.iter())
        .map(|(&leaf_val_off, &leaf_cnt)| leaf_val_off as usize + leaf_cnt as usize)
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

    let is_v2 = magic == MAGIC_QS_V2;
    Ok(QsPack {
        n_features,
        n_trees,
        n_blocks,
        base_score,
        cut_offsets,
        cuts,
        tree_hdrs: if is_v2 { Vec::new() } else { tree_hdrs },
        block_hdrs: if is_v2 { Vec::new() } else { block_hdrs },
        tree_block_off,
        tree_block_cnt,
        tree_leaf_cnt,
        tree_leaf_val_off,
        tree_init_lo,
        tree_init_hi,
        block_fid,
        block_bucket_cnt,
        block_miss_bucket,
        block_lut_off,
        block_mask_off,
        luts,
        masks: if is_v2 { Vec::new() } else { masks },
        masks_lo,
        masks_hi,
        leaf_values,
    })
}

pub fn save_qs_pack_v2(path: &PathBuf, pack: &QsPack) -> Result<()> {
    let mut cut_offsets_bytes = Vec::with_capacity((pack.n_features + 1) * 4);
    for &v in &pack.cut_offsets {
        cut_offsets_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut cuts_bytes = Vec::with_capacity(pack.cuts.len() * 4);
    for &v in &pack.cuts {
        cuts_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let mut tree_hdr_bytes = Vec::with_capacity(pack.n_trees * 28);
    for tree_idx in 0..pack.n_trees {
        tree_hdr_bytes.extend_from_slice(&pack.tree_block_off[tree_idx].to_le_bytes());
        let block_leaf =
            (pack.tree_block_cnt[tree_idx] as u32) | ((pack.tree_leaf_cnt[tree_idx] as u32) << 16);
        tree_hdr_bytes.extend_from_slice(&block_leaf.to_le_bytes());
        tree_hdr_bytes.extend_from_slice(&pack.tree_leaf_val_off[tree_idx].to_le_bytes());
        tree_hdr_bytes.extend_from_slice(&pack.tree_init_lo[tree_idx].to_le_bytes());
        tree_hdr_bytes.extend_from_slice(&pack.tree_init_hi[tree_idx].to_le_bytes());
    }
    let mut block_hdr_bytes = Vec::with_capacity(pack.n_blocks * 12);
    for block_idx in 0..pack.n_blocks {
        block_hdr_bytes.push(pack.block_fid[block_idx]);
        block_hdr_bytes.push(pack.block_bucket_cnt[block_idx]);
        block_hdr_bytes.push(pack.block_miss_bucket[block_idx]);
        block_hdr_bytes.push(0);
        block_hdr_bytes.extend_from_slice(&pack.block_lut_off[block_idx].to_le_bytes());
        block_hdr_bytes.extend_from_slice(&pack.block_mask_off[block_idx].to_le_bytes());
    }
    let mut luts_bytes = Vec::with_capacity(pack.luts.len());
    luts_bytes.extend_from_slice(&pack.luts);
    let mut masks_bytes = Vec::with_capacity(pack.masks_lo.len() * 16);
    for idx in 0..pack.masks_lo.len() {
        masks_bytes.extend_from_slice(&pack.masks_lo[idx].to_le_bytes());
        masks_bytes.extend_from_slice(&pack.masks_hi[idx].to_le_bytes());
    }
    let mut leaf_bytes = Vec::with_capacity(pack.leaf_values.len() * 4);
    for &v in &pack.leaf_values {
        leaf_bytes.extend_from_slice(&v.to_le_bytes());
    }

    let header_len = MAGIC_QS_V2.len() + 16 + 56;
    let cut_offsets_off = header_len;
    let cuts_off = cut_offsets_off + cut_offsets_bytes.len();
    let tree_hdrs_off = cuts_off + cuts_bytes.len();
    let block_hdrs_off = tree_hdrs_off + tree_hdr_bytes.len();
    let luts_off = block_hdrs_off + block_hdr_bytes.len();
    let masks_off = luts_off + luts_bytes.len();
    let leafs_off = masks_off + masks_bytes.len();

    let mut out = Vec::with_capacity(leafs_off + leaf_bytes.len());
    out.extend_from_slice(MAGIC_QS_V2);
    out.extend_from_slice(&(pack.n_features as u32).to_le_bytes());
    out.extend_from_slice(&(pack.n_trees as u32).to_le_bytes());
    out.extend_from_slice(&(pack.n_blocks as u32).to_le_bytes());
    out.extend_from_slice(&pack.base_score.to_le_bytes());
    out.extend_from_slice(&(cut_offsets_off as u64).to_le_bytes());
    out.extend_from_slice(&(cuts_off as u64).to_le_bytes());
    out.extend_from_slice(&(tree_hdrs_off as u64).to_le_bytes());
    out.extend_from_slice(&(block_hdrs_off as u64).to_le_bytes());
    out.extend_from_slice(&(luts_off as u64).to_le_bytes());
    out.extend_from_slice(&(masks_off as u64).to_le_bytes());
    out.extend_from_slice(&(leafs_off as u64).to_le_bytes());
    out.extend_from_slice(&cut_offsets_bytes);
    out.extend_from_slice(&cuts_bytes);
    out.extend_from_slice(&tree_hdr_bytes);
    out.extend_from_slice(&block_hdr_bytes);
    out.extend_from_slice(&luts_bytes);
    out.extend_from_slice(&masks_bytes);
    out.extend_from_slice(&leaf_bytes);
    std::fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

pub fn extract_tree_range_remapped(
    pack: &QsPack,
    start_tree_idx: usize,
    end_tree_idx: usize,
) -> Result<(QsPack, Vec<u16>)> {
    let end = end_tree_idx.min(pack.n_trees);
    let start = start_tree_idx.min(end);
    if start == end {
        bail!(
            "empty tree range for segment: start={} end={}",
            start_tree_idx,
            end_tree_idx
        );
    }

    let mut global_to_local = vec![u16::MAX; pack.n_features];
    let mut local_to_global = Vec::new();
    for tree_idx in start..end {
        let (block_start, block_end) = pack.tree_block_range(tree_idx);
        for block_idx in block_start..block_end {
            let global_fid = pack.block_fid[block_idx] as usize;
            if global_to_local[global_fid] == u16::MAX {
                let local_fid = local_to_global.len() as u16;
                global_to_local[global_fid] = local_fid;
                local_to_global.push(global_fid as u16);
            }
        }
    }

    let mut cut_offsets = Vec::with_capacity(local_to_global.len() + 1);
    let mut cuts = Vec::new();
    cut_offsets.push(0);
    for &global_fid_u16 in &local_to_global {
        let global_fid = global_fid_u16 as usize;
        let start_off = pack.cut_offsets[global_fid] as usize;
        let end_off = pack.cut_offsets[global_fid + 1] as usize;
        cuts.extend_from_slice(&pack.cuts[start_off..end_off]);
        cut_offsets.push(cuts.len() as u32);
    }

    let mut tree_block_off = Vec::with_capacity(end - start);
    let mut tree_block_cnt = Vec::with_capacity(end - start);
    let mut tree_leaf_cnt = Vec::with_capacity(end - start);
    let mut tree_leaf_val_off = Vec::with_capacity(end - start);
    let mut tree_init_lo = Vec::with_capacity(end - start);
    let mut tree_init_hi = Vec::with_capacity(end - start);
    let mut block_fid = Vec::new();
    let mut block_bucket_cnt = Vec::new();
    let mut block_miss_bucket = Vec::new();
    let mut block_lut_off = Vec::new();
    let mut block_mask_off = Vec::new();
    let mut luts = Vec::new();
    let mut masks_lo = Vec::new();
    let mut masks_hi = Vec::new();
    let mut leaf_values = Vec::new();

    for tree_idx in start..end {
        tree_block_off.push(block_fid.len() as u32);
        tree_block_cnt.push(pack.tree_block_cnt[tree_idx]);
        tree_leaf_cnt.push(pack.tree_leaf_cnt[tree_idx]);
        tree_leaf_val_off.push(leaf_values.len() as u32);
        tree_init_lo.push(pack.tree_init_lo[tree_idx]);
        tree_init_hi.push(pack.tree_init_hi[tree_idx]);

        let (block_start, block_end) = pack.tree_block_range(tree_idx);
        for block_idx in block_start..block_end {
            let global_fid = pack.block_fid[block_idx] as usize;
            let local_fid = global_to_local[global_fid];
            debug_assert!(local_fid != u16::MAX);
            block_fid.push(local_fid as u8);
            block_bucket_cnt.push(pack.block_bucket_cnt[block_idx]);
            block_miss_bucket.push(pack.block_miss_bucket[block_idx]);

            let lut_base = pack.block_lut_off[block_idx] as usize;
            let lut_off = luts.len() as u32;
            luts.extend_from_slice(&pack.luts[lut_base..lut_base + 129]);
            block_lut_off.push(lut_off);

            let mask_base = pack.block_mask_off[block_idx] as usize;
            let bucket_cnt = pack.block_bucket_cnt[block_idx] as usize;
            let mask_off = masks_lo.len() as u32;
            masks_lo.extend_from_slice(&pack.masks_lo[mask_base..mask_base + bucket_cnt]);
            masks_hi.extend_from_slice(&pack.masks_hi[mask_base..mask_base + bucket_cnt]);
            block_mask_off.push(mask_off);
        }

        let leaf_base = pack.tree_leaf_val_off[tree_idx] as usize;
        let leaf_cnt = pack.tree_leaf_cnt[tree_idx] as usize;
        leaf_values.extend_from_slice(&pack.leaf_values[leaf_base..leaf_base + leaf_cnt]);
    }

    Ok((
        QsPack {
            n_features: local_to_global.len(),
            n_trees: end - start,
            n_blocks: block_fid.len(),
            base_score: pack.base_score,
            cut_offsets,
            cuts,
            tree_hdrs: Vec::new(),
            block_hdrs: Vec::new(),
            tree_block_off,
            tree_block_cnt,
            tree_leaf_cnt,
            tree_leaf_val_off,
            tree_init_lo,
            tree_init_hi,
            block_fid,
            block_bucket_cnt,
            block_miss_bucket,
            block_lut_off,
            block_mask_off,
            luts,
            masks: Vec::new(),
            masks_lo,
            masks_hi,
            leaf_values,
        },
        local_to_global,
    ))
}

pub fn extract_tree_indices_remapped(
    pack: &QsPack,
    tree_indices: &[u32],
) -> Result<(QsPack, Vec<u16>)> {
    if tree_indices.is_empty() {
        bail!("empty tree index list for remapped segment");
    }

    let mut global_to_local = vec![u16::MAX; pack.n_features];
    let mut local_to_global = Vec::new();
    for &tree_idx_u32 in tree_indices {
        let tree_idx = tree_idx_u32 as usize;
        if tree_idx >= pack.n_trees {
            bail!(
                "tree index {} outside qs pack tree count {}",
                tree_idx,
                pack.n_trees
            );
        }
        let (block_start, block_end) = pack.tree_block_range(tree_idx);
        for block_idx in block_start..block_end {
            let global_fid = pack.block_fid[block_idx] as usize;
            if global_to_local[global_fid] == u16::MAX {
                let local_fid = local_to_global.len() as u16;
                global_to_local[global_fid] = local_fid;
                local_to_global.push(global_fid as u16);
            }
        }
    }

    let mut cut_offsets = Vec::with_capacity(local_to_global.len() + 1);
    let mut cuts = Vec::new();
    cut_offsets.push(0);
    for &global_fid_u16 in &local_to_global {
        let global_fid = global_fid_u16 as usize;
        let start_off = pack.cut_offsets[global_fid] as usize;
        let end_off = pack.cut_offsets[global_fid + 1] as usize;
        cuts.extend_from_slice(&pack.cuts[start_off..end_off]);
        cut_offsets.push(cuts.len() as u32);
    }

    let mut tree_block_off = Vec::with_capacity(tree_indices.len());
    let mut tree_block_cnt = Vec::with_capacity(tree_indices.len());
    let mut tree_leaf_cnt = Vec::with_capacity(tree_indices.len());
    let mut tree_leaf_val_off = Vec::with_capacity(tree_indices.len());
    let mut tree_init_lo = Vec::with_capacity(tree_indices.len());
    let mut tree_init_hi = Vec::with_capacity(tree_indices.len());
    let mut block_fid = Vec::new();
    let mut block_bucket_cnt = Vec::new();
    let mut block_miss_bucket = Vec::new();
    let mut block_lut_off = Vec::new();
    let mut block_mask_off = Vec::new();
    let mut luts = Vec::new();
    let mut masks_lo = Vec::new();
    let mut masks_hi = Vec::new();
    let mut leaf_values = Vec::new();

    for &tree_idx_u32 in tree_indices {
        let tree_idx = tree_idx_u32 as usize;
        tree_block_off.push(block_fid.len() as u32);
        tree_block_cnt.push(pack.tree_block_cnt[tree_idx]);
        tree_leaf_cnt.push(pack.tree_leaf_cnt[tree_idx]);
        tree_leaf_val_off.push(leaf_values.len() as u32);
        tree_init_lo.push(pack.tree_init_lo[tree_idx]);
        tree_init_hi.push(pack.tree_init_hi[tree_idx]);

        let (block_start, block_end) = pack.tree_block_range(tree_idx);
        for block_idx in block_start..block_end {
            let global_fid = pack.block_fid[block_idx] as usize;
            let local_fid = global_to_local[global_fid];
            debug_assert!(local_fid != u16::MAX);
            block_fid.push(local_fid as u8);
            block_bucket_cnt.push(pack.block_bucket_cnt[block_idx]);
            block_miss_bucket.push(pack.block_miss_bucket[block_idx]);

            let lut_base = pack.block_lut_off[block_idx] as usize;
            let lut_off = luts.len() as u32;
            luts.extend_from_slice(&pack.luts[lut_base..lut_base + 129]);
            block_lut_off.push(lut_off);

            let mask_base = pack.block_mask_off[block_idx] as usize;
            let bucket_cnt = pack.block_bucket_cnt[block_idx] as usize;
            let mask_off = masks_lo.len() as u32;
            masks_lo.extend_from_slice(&pack.masks_lo[mask_base..mask_base + bucket_cnt]);
            masks_hi.extend_from_slice(&pack.masks_hi[mask_base..mask_base + bucket_cnt]);
            block_mask_off.push(mask_off);
        }

        let leaf_base = pack.tree_leaf_val_off[tree_idx] as usize;
        let leaf_cnt = pack.tree_leaf_cnt[tree_idx] as usize;
        leaf_values.extend_from_slice(&pack.leaf_values[leaf_base..leaf_base + leaf_cnt]);
    }

    Ok((
        QsPack {
            n_features: local_to_global.len(),
            n_trees: tree_indices.len(),
            n_blocks: block_fid.len(),
            base_score: pack.base_score,
            cut_offsets,
            cuts,
            tree_hdrs: Vec::new(),
            block_hdrs: Vec::new(),
            tree_block_off,
            tree_block_cnt,
            tree_leaf_cnt,
            tree_leaf_val_off,
            tree_init_lo,
            tree_init_hi,
            block_fid,
            block_bucket_cnt,
            block_miss_bucket,
            block_lut_off,
            block_mask_off,
            luts,
            masks: Vec::new(),
            masks_lo,
            masks_hi,
            leaf_values,
        },
        local_to_global,
    ))
}

#[inline(always)]
pub fn quantize_feature_subset_nomiss_mapped(
    pack: &QsPack,
    feat: &[f32],
    ranks: &mut [u8],
    local_to_global_fid: &[u16],
) {
    debug_assert!(ranks.len() >= pack.n_features);
    debug_assert_eq!(local_to_global_fid.len(), pack.n_features);
    for (local_fid, &global_fid_u16) in local_to_global_fid.iter().enumerate() {
        let global_fid = global_fid_u16 as usize;
        let value = feat[global_fid];
        let start = pack.cut_offsets[local_fid] as usize;
        let end = pack.cut_offsets[local_fid + 1] as usize;
        let cuts = &pack.cuts[start..end];
        let mut lo = 0usize;
        let mut hi = cuts.len();
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if value <= cuts[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        ranks[local_fid] = lo.min(255) as u8;
    }
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

    #[inline(always)]
    pub fn cut_offset(&self, fid: usize) -> usize {
        self.cut_offsets[fid] as usize
    }

    #[inline(always)]
    pub fn cuts_slice(&self) -> &[f32] {
        &self.cuts
    }

    #[inline(always)]
    fn tree_block_range(&self, tree_idx: usize) -> (usize, usize) {
        let start = self.tree_block_off[tree_idx] as usize;
        let end = start + self.tree_block_cnt[tree_idx] as usize;
        (start, end)
    }

    #[inline(always)]
    fn tree_leaf_count(&self, tree_idx: usize) -> usize {
        self.tree_leaf_cnt[tree_idx] as usize
    }

    #[inline(always)]
    fn tree_leaf_value(&self, tree_idx: usize, leaf_idx: usize) -> f32 {
        self.leaf_values[self.tree_leaf_val_off[tree_idx] as usize + leaf_idx]
    }

    #[inline(always)]
    fn tree_init_mask(&self, tree_idx: usize) -> (u64, u64) {
        (self.tree_init_lo[tree_idx], self.tree_init_hi[tree_idx])
    }

    #[inline(always)]
    fn block_bucket(&self, block_idx: usize, ranks: &[u8], missing: &[u8]) -> usize {
        let fid = self.block_fid[block_idx] as usize;
        if missing[fid] != 0 {
            self.block_miss_bucket[block_idx] as usize
        } else {
            self.luts[self.block_lut_off[block_idx] as usize + ranks[fid] as usize] as usize
        }
    }

    #[inline(always)]
    fn block_bucket_nomiss(&self, block_idx: usize, ranks: &[u8]) -> usize {
        let fid = self.block_fid[block_idx] as usize;
        self.luts[self.block_lut_off[block_idx] as usize + ranks[fid] as usize] as usize
    }

    #[inline(always)]
    fn mask(&self, block_idx: usize, bucket: usize) -> (u64, u64) {
        let mask_idx = self.block_mask_off[block_idx] as usize + bucket;
        (self.masks_lo[mask_idx], self.masks_hi[mask_idx])
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
fn quantize_row_nomiss(pack: &QsPack, feat: &[f32], ranks: &mut [u8]) {
    let mut fid = 0usize;
    while fid < pack.n_features {
        let st = pack.cut_offsets[fid] as usize;
        let ed = pack.cut_offsets[fid + 1] as usize;
        ranks[fid] = upper_bound(&pack.cuts[st..ed], feat[fid]);
        fid += 1;
    }
}

#[inline(always)]
pub fn quantize_into_nomiss(pack: &QsPack, feat: &[f32], ranks: &mut [u8]) {
    quantize_row_nomiss(pack, feat, ranks);
}

#[inline(always)]
pub fn quantize_feature_subset_nomiss(
    pack: &QsPack,
    feat: &[f32],
    ranks: &mut [u8],
    feature_ids: &[u16],
) {
    let mut i = 0usize;
    while i < feature_ids.len() {
        let fid = feature_ids[i] as usize;
        let st = pack.cut_offsets[fid] as usize;
        let ed = pack.cut_offsets[fid + 1] as usize;
        ranks[fid] = upper_bound(&pack.cuts[st..ed], feat[fid]);
        i += 1;
    }
}

#[inline(always)]
pub fn quantize_feature_subset_nomiss_precomputed(
    cuts: &[f32],
    feat: &[f32],
    ranks: &mut [u8],
    feature_ids: &[u16],
    cut_starts: &[u32],
    cut_ends: &[u32],
) {
    let mut i = 0usize;
    while i < feature_ids.len() {
        let fid = feature_ids[i] as usize;
        let st = cut_starts[i] as usize;
        let ed = cut_ends[i] as usize;
        ranks[fid] = upper_bound(&cuts[st..ed], feat[fid]);
        i += 1;
    }
}

#[inline(always)]
fn score_tree_quantized(
    pack: &QsPack,
    tree_idx: usize,
    ranks: &[u8],
    missing: &[u8],
) -> Result<(f32, u64, u64)> {
    let mut lo = pack.tree_init_lo[tree_idx];
    let mut hi = pack.tree_init_hi[tree_idx];
    let mut block_evals = 0u64;
    let mut resolved_early = 0u64;
    let block_start = pack.tree_block_off[tree_idx] as usize;
    let block_end = block_start + pack.tree_block_cnt[tree_idx] as usize;
    for block_idx in block_start..block_end {
        let fid = pack.block_fid[block_idx] as usize;
        let bucket = if missing[fid] != 0 {
            pack.block_miss_bucket[block_idx] as usize
        } else {
            pack.luts[pack.block_lut_off[block_idx] as usize + ranks[fid] as usize] as usize
        };
        let mask_idx = pack.block_mask_off[block_idx] as usize + bucket;
        lo &= pack.masks_lo[mask_idx];
        hi &= pack.masks_hi[mask_idx];
        block_evals += 1;
        if resolved(lo, hi) {
            resolved_early += 1;
            break;
        }
    }
    let leaf_idx = decode_leaf(lo, hi)?;
    if leaf_idx >= pack.tree_leaf_cnt[tree_idx] as usize {
        bail!(
            "decoded leaf {} outside tree leaf count {}",
            leaf_idx,
            pack.tree_leaf_cnt[tree_idx]
        );
    }
    Ok((
        pack.leaf_values[pack.tree_leaf_val_off[tree_idx] as usize + leaf_idx],
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
            if tree_idx >= pack.n_trees {
                bail!(
                    "tree index {} outside qs pack tree count {}",
                    tree_idx,
                    pack.n_trees
                );
            }
        }
        let (tree_score, tree_blocks, tree_resolved) =
            score_tree_quantized(pack, tree_idx, ranks, missing)?;
        score += tree_score;
        block_evals += tree_blocks;
        resolved_early += tree_resolved;
    }
    Ok((score, block_evals, resolved_early))
}

#[inline(always)]
fn score_tree_quantized_nomiss(
    pack: &QsPack,
    tree_idx: usize,
    ranks: &[u8],
) -> Result<(f32, u64, u64)> {
    let mut lo = pack.tree_init_lo[tree_idx];
    let mut hi = pack.tree_init_hi[tree_idx];
    let mut block_evals = 0u64;
    let mut resolved_early = 0u64;
    let block_start = pack.tree_block_off[tree_idx] as usize;
    let block_end = block_start + pack.tree_block_cnt[tree_idx] as usize;
    for block_idx in block_start..block_end {
        let fid = pack.block_fid[block_idx] as usize;
        let bucket =
            pack.luts[pack.block_lut_off[block_idx] as usize + ranks[fid] as usize] as usize;
        let mask_idx = pack.block_mask_off[block_idx] as usize + bucket;
        lo &= pack.masks_lo[mask_idx];
        hi &= pack.masks_hi[mask_idx];
        block_evals += 1;
        if resolved(lo, hi) {
            resolved_early += 1;
            break;
        }
    }
    let leaf_idx = decode_leaf(lo, hi)?;
    if leaf_idx >= pack.tree_leaf_cnt[tree_idx] as usize {
        bail!(
            "decoded leaf {} outside tree leaf count {}",
            leaf_idx,
            pack.tree_leaf_cnt[tree_idx]
        );
    }
    Ok((
        pack.leaf_values[pack.tree_leaf_val_off[tree_idx] as usize + leaf_idx],
        block_evals,
        resolved_early,
    ))
}

#[inline(always)]
pub fn score_tree_indices_from_quantized_nomiss(
    pack: &QsPack,
    tree_indices: &[u32],
    ranks: &[u8],
) -> Result<(f32, u64, u64)> {
    let mut score = 0.0f32;
    let mut block_evals = 0u64;
    let mut resolved_early = 0u64;
    for &tree_idx_u32 in tree_indices {
        let tree_idx = tree_idx_u32 as usize;
        if tree_idx >= pack.n_trees {
            bail!(
                "tree index {} outside qs pack tree count {}",
                tree_idx,
                pack.n_trees
            );
        }
        let (tree_score, tree_blocks, tree_resolved) =
            score_tree_quantized_nomiss(pack, tree_idx, ranks)?;
        score += tree_score;
        block_evals += tree_blocks;
        resolved_early += tree_resolved;
    }
    Ok((score, block_evals, resolved_early))
}

#[inline(always)]
pub fn score_all_trees_prequantized_nomiss(
    pack: &QsPack,
    ranks: &[u8],
    init_score: f32,
    init_block_evals: u64,
    init_resolved_early_trees: u64,
) -> Result<QsPrefixDecisionRow> {
    let mut score = init_score;
    let mut block_evals = init_block_evals;
    let mut resolved_early = init_resolved_early_trees;

    for tree_idx in 0..pack.n_trees {
        let (tree_score, tree_blocks, tree_resolved) =
            score_tree_quantized_nomiss(pack, tree_idx, ranks)?;
        score += tree_score;
        block_evals += tree_blocks;
        resolved_early += tree_resolved;
    }

    Ok(QsPrefixDecisionRow {
        prefix_score: score,
        trees_used: pack.n_trees,
        block_evals,
        resolved_early_trees: resolved_early,
    })
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

    for tree_idx in start_tree_idx..pack.n_trees {
        if tree_idx >= max_checkpoint {
            break;
        }
        let (mut lo, mut hi) = pack.tree_init_mask(tree_idx);
        let (block_start, block_end) = pack.tree_block_range(tree_idx);
        for block_idx in block_start..block_end {
            let bucket = pack.block_bucket(block_idx, ranks, missing);
            let (mask_lo, mask_hi) = pack.mask(block_idx, bucket);
            lo &= mask_lo;
            hi &= mask_hi;
            block_evals += 1;
            if resolved(lo, hi) {
                resolved_early += 1;
                break;
            }
        }
        let leaf_idx = decode_leaf(lo, hi)?;
        if leaf_idx >= pack.tree_leaf_count(tree_idx) {
            bail!(
                "decoded leaf {} outside tree leaf count {}",
                leaf_idx,
                pack.tree_leaf_count(tree_idx)
            );
        }
        score += pack.tree_leaf_value(tree_idx, leaf_idx);

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

    for tree_idx in start_tree_idx..pack.n_trees {
        if tree_idx >= max_checkpoint {
            break;
        }
        let (mut lo, mut hi) = pack.tree_init_mask(tree_idx);
        let (block_start, block_end) = pack.tree_block_range(tree_idx);
        for block_idx in block_start..block_end {
            let bucket = pack.block_bucket(block_idx, ranks, missing);
            let (mask_lo, mask_hi) = pack.mask(block_idx, bucket);
            lo &= mask_lo;
            hi &= mask_hi;
            block_evals += 1;
            if resolved(lo, hi) {
                resolved_early += 1;
                break;
            }
        }
        let leaf_idx = decode_leaf(lo, hi)?;
        if leaf_idx >= pack.tree_leaf_count(tree_idx) {
            bail!(
                "decoded leaf {} outside tree leaf count {}",
                leaf_idx,
                pack.tree_leaf_count(tree_idx)
            );
        }
        score += pack.tree_leaf_value(tree_idx, leaf_idx);

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
pub fn prefix_until_from_nomiss<F>(
    pack: &QsPack,
    feat: &[f32],
    ranks: &mut [u8],
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

    quantize_row_nomiss(pack, feat, ranks);
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

    for tree_idx in start_tree_idx..pack.n_trees {
        if tree_idx >= max_checkpoint {
            break;
        }
        let (mut lo, mut hi) = pack.tree_init_mask(tree_idx);
        let (block_start, block_end) = pack.tree_block_range(tree_idx);
        for block_idx in block_start..block_end {
            let bucket = pack.block_bucket_nomiss(block_idx, ranks);
            let (mask_lo, mask_hi) = pack.mask(block_idx, bucket);
            lo &= mask_lo;
            hi &= mask_hi;
            block_evals += 1;
            if resolved(lo, hi) {
                resolved_early += 1;
                break;
            }
        }
        let leaf_idx = decode_leaf(lo, hi)?;
        if leaf_idx >= pack.tree_leaf_count(tree_idx) {
            bail!(
                "decoded leaf {} outside tree leaf count {}",
                leaf_idx,
                pack.tree_leaf_count(tree_idx)
            );
        }
        score += pack.tree_leaf_value(tree_idx, leaf_idx);

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
pub fn prefix_until_from_prequantized_nomiss<F>(
    pack: &QsPack,
    ranks: &[u8],
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

    for tree_idx in start_tree_idx..pack.n_trees {
        if tree_idx >= max_checkpoint {
            break;
        }
        let (mut lo, mut hi) = pack.tree_init_mask(tree_idx);
        let (block_start, block_end) = pack.tree_block_range(tree_idx);
        for block_idx in block_start..block_end {
            let bucket = pack.block_bucket_nomiss(block_idx, ranks);
            let (mask_lo, mask_hi) = pack.mask(block_idx, bucket);
            lo &= mask_lo;
            hi &= mask_hi;
            block_evals += 1;
            if resolved(lo, hi) {
                resolved_early += 1;
                break;
            }
        }
        let leaf_idx = decode_leaf(lo, hi)?;
        if leaf_idx >= pack.tree_leaf_count(tree_idx) {
            bail!(
                "decoded leaf {} outside tree leaf count {}",
                leaf_idx,
                pack.tree_leaf_count(tree_idx)
            );
        }
        score += pack.tree_leaf_value(tree_idx, leaf_idx);

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

pub fn tree_range_feature_ids(
    pack: &QsPack,
    start_tree_idx: usize,
    end_tree_idx: usize,
) -> Vec<u16> {
    let end = end_tree_idx.min(pack.n_trees);
    let start = start_tree_idx.min(end);
    let mut seen = vec![false; pack.n_features];
    let mut out = Vec::new();
    for tree_idx in start..end {
        let (block_start, block_end) = pack.tree_block_range(tree_idx);
        for block_idx in block_start..block_end {
            let fid = pack.block_fid[block_idx] as usize;
            if !seen[fid] {
                seen[fid] = true;
                out.push(fid as u16);
            }
        }
    }
    out
}

pub fn append_tree_range_new_feature_ids(
    pack: &QsPack,
    start_tree_idx: usize,
    end_tree_idx: usize,
    seen: &mut [bool],
    out: &mut Vec<u16>,
) {
    let end = end_tree_idx.min(pack.n_trees);
    let start = start_tree_idx.min(end);
    for tree_idx in start..end {
        let (block_start, block_end) = pack.tree_block_range(tree_idx);
        for block_idx in block_start..block_end {
            let fid = pack.block_fid[block_idx] as usize;
            if !seen[fid] {
                seen[fid] = true;
                out.push(fid as u16);
            }
        }
    }
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

    let mut work = || -> Result<QsAgg> {
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
