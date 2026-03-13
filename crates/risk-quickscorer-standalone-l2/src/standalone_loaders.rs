
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

fn compile_late_if_tree_program(
    model: &SoaModel,
    plan: &TreeOrder,
    start_tree_idx: usize,
    checkpoints: &[usize],
) -> Result<LateIfTreeProgram> {
    if checkpoints.is_empty() {
        bail!("late if-tree checkpoints must not be empty");
    }
    let end_tree_idx = *checkpoints.last().unwrap_or(&start_tree_idx);
    if start_tree_idx > end_tree_idx || end_tree_idx > plan.n_trees {
        bail!(
            "invalid late if-tree range [{}..{}) for plan.n_trees={}",
            start_tree_idx,
            end_tree_idx,
            plan.n_trees
        );
    }
    if model.n_features > LATE_IF_TREE_NODE_FIDX_MASK as usize {
        bail!(
            "late if-tree program requires n_features <= {} got {}",
            LATE_IF_TREE_NODE_FIDX_MASK,
            model.n_features
        );
    }

    let mut tree_node_offs = Vec::with_capacity(end_tree_idx.saturating_sub(start_tree_idx));
    let mut nodes = Vec::new();
    for pos in start_tree_idx..end_tree_idx {
        let tree_idx = *plan.order.get(pos).unwrap_or(&0) as usize;
        let src_start = *model.tree_roots.get(tree_idx).unwrap_or(&0) as usize;
        let src_end = if tree_idx + 1 < model.n_trees {
            *model
                .tree_roots
                .get(tree_idx + 1)
                .unwrap_or(&(model.node_count as u32)) as usize
        } else {
            model.node_count
        };
        if src_end < src_start || src_end > model.node_count {
            bail!(
                "invalid source node range [{}..{}) for tree_idx={} node_count={}",
                src_start,
                src_end,
                tree_idx,
                model.node_count
            );
        }
        let local_node_count = src_end - src_start;
        if local_node_count > u8::MAX as usize {
            bail!(
                "late if-tree local node count {} exceeds compact limit for tree_idx={}",
                local_node_count,
                tree_idx
            );
        }
        let dst_base = nodes.len() as u32;
        tree_node_offs.push(dst_base);
        for src_idx in src_start..src_end {
            let is_leaf = *model.is_leaf.get(src_idx).unwrap_or(&0) != 0;
            let fidx_flags = if is_leaf {
                LATE_IF_TREE_NODE_LEAF
            } else {
                let fidx = *model.fidx.get(src_idx).unwrap_or(&0);
                if fidx < 0 {
                    bail!("negative fidx in non-leaf late node: {}", fidx);
                }
                fidx as u8
            };
            let left = if is_leaf {
                0
            } else {
                (*model.left.get(src_idx).unwrap_or(&0) as usize - src_start) as u8
            };
            let right = if is_leaf {
                0
            } else {
                (*model.right.get(src_idx).unwrap_or(&0) as usize - src_start) as u8
            };
            nodes.push(LateIfTreeNode {
                thr: *model.thr.get(src_idx).unwrap_or(&0.0),
                leaf: *model.leaf.get(src_idx).unwrap_or(&0.0),
                left,
                right,
                fidx_flags,
                _pad: 0,
            });
        }
    }

    Ok(LateIfTreeProgram {
        start_tree_idx,
        n_trees: end_tree_idx.saturating_sub(start_tree_idx),
        tree_node_offs,
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
    let late_iftree_program = if let (Some(compiled_hot_pack), Some(layout)) =
        (compiled_hot_pack.as_ref(), hot_checkpoint_layout.as_ref())
    {
        if layout.late_values.is_empty() {
            None
        } else {
            let hot_limit = compiled_hot_pack.n_trees.min(plan.n_trees);
            Some(compile_late_if_tree_program(
                model,
                &plan,
                hot_limit,
                &layout.late_values,
            )?)
        }
    } else {
        None
    };
    Ok(LoadedPrefixRuntime {
        pack,
        late_iftree_program,
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
