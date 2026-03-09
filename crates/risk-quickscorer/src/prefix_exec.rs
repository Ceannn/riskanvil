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
fn maybe_prefix_certify_plan_fast(
    checkpoint: &LoadedPrefixCertifyCheckpoint,
    row_fold: i32,
    row_tau: f32,
    atlas_tau_bin: usize,
    prefix_score: f32,
    delta_prev_checkpoint: f32,
    delta_from_64: f32,
) -> Option<(bool, f32)> {
    let row_tau_bin = atlas_tau_bin.min(checkpoint.atlas.tau_bin_count.saturating_sub(1));
    let row_calibration_tau_bin = if checkpoint.table.tau_edges.len() >= 2 {
        prefix_gap_bin(&checkpoint.table.tau_edges, row_tau)
    } else {
        0usize
    };
    maybe_prefix_certify_atlas_fast(
        &checkpoint.table,
        &checkpoint.atlas,
        prefix_fold_slot(&checkpoint.table, row_fold),
        row_tau,
        row_tau_bin,
        row_calibration_tau_bin,
        prefix_score,
        delta_prev_checkpoint,
        delta_from_64,
    )
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
    let (ref_hi, rej_lo) = lookup_prefix_bands_with_tau_bin_slot(
        table,
        row_fold_slot,
        row_calibration_tau_bin,
        gap,
    );
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
        checkpoint_work_evals: if trace_enabled { vec![0u32; cp_len] } else { Vec::new() },
        checkpoint_deltas: if trace_enabled { vec![0.0f32; cp_len] } else { Vec::new() },
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

    let mut handle_checkpoint =
        |global_cp_idx: usize, checkpoint: usize, score: f32, work_evals: u64, resolved_early: u64| {
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
                out.checkpoint_work_evals[global_cp_idx] =
                    work_evals.min(u32::MAX as u64) as u32;
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
        checkpoint_work_evals: if trace_enabled { vec![0u32; cp_len] } else { Vec::new() },
        checkpoint_deltas: if trace_enabled { vec![0.0f32; cp_len] } else { Vec::new() },
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

    let mut handle_checkpoint =
        |global_cp_idx: usize, checkpoint: usize, score: f32, work_evals: u64, resolved_early: u64| {
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
                out.checkpoint_work_evals[global_cp_idx] =
                    work_evals.min(u32::MAX as u64) as u32;
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

    let mut handle_checkpoint =
        |global_cp_idx: usize, checkpoint: usize, score: f32, work_evals: u64, resolved_early: u64| {
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
        let hot_limit = compiled_hot_pack.n_trees.min(runtime.pack.n_trees());
        let mut late_pos = 0usize;
        let late_row = qs_exact::prefix_until_from(
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
        out.fallback_entry_checkpoint = *runtime
            .calibration
            .checkpoints
            .last()
            .unwrap_or(&0usize);
    }
    Ok(out)
}

fn run_prefix_shadow_row_single_route_compiled_nomiss_online(
    runtime: &LoadedPrefixRuntime,
    feat: &[f32],
    row_tau: f32,
    row_fold: i32,
    hot_buf: &mut HotFeatureBuf,
    ranks: &mut [u8],
) -> Result<Option<PrefixRowShadow>> {
    let Some(compiled_hot_pack) = runtime.compiled_hot_pack.as_ref() else {
        return Ok(None);
    };
    let Some(hot_checkpoint_layout) = runtime.hot_checkpoint_layout.as_ref() else {
        return Ok(None);
    };
    let Some(atlas) = runtime.atlas_certifier.as_ref() else {
        return Ok(None);
    };
    let certify_plan = runtime.certify_plan.as_ref();
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
        return Ok(None);
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

    let mut handle_checkpoint =
        |global_cp_idx: usize, checkpoint: usize, score: f32, work_evals: u64, resolved_early: u64| {
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
            let cert = if let Some(plan) = certify_plan {
                plan.checkpoints.get(global_cp_idx).and_then(|cp| {
                    maybe_prefix_certify_plan_fast(
                        cp,
                        row_fold,
                        row_tau,
                        atlas_tau_bin,
                        score,
                        delta_prev,
                        delta_from_64,
                    )
                })
            } else {
                maybe_prefix_certify_atlas_fast(
                    &runtime.calibration.tables[global_cp_idx],
                    &atlas.checkpoints[global_cp_idx],
                    prefix_fold_slot(&runtime.calibration.tables[global_cp_idx], row_fold),
                    row_tau,
                    atlas_tau_bin,
                    calibration_tau_bin,
                    score,
                    delta_prev,
                    delta_from_64,
                )
            };
            if let Some((is_reject, route_score)) = cert {
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
        let mut hot_pos = 0usize;
        let (hot_score, hot_trees_used, hot_work, hot_resolved) = unsafe {
            if let Some(stage_plan) = runtime.hot_stage_plan.as_ref() {
                hot_exact_prefix_until_compiled_nomiss_staged(
                    compiled_hot_pack,
                    hot_buf,
                    feat,
                    stage_plan,
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
            }
        }?;
        score = hot_score;
        work_evals = hot_work;
        resolved_early = hot_resolved;
        trees_used = hot_trees_used;
    }

    if direct_decision.get().is_none()
        && runtime
            .late_segment_runtime
            .as_ref()
            .map(|x| !x.segments.is_empty())
            .unwrap_or(false)
    {
        let late_runtime = runtime.late_segment_runtime.as_ref().unwrap();
        for segment in &late_runtime.segments {
            qs_exact::quantize_feature_subset_nomiss_mapped(
                &segment.pack,
                feat,
                ranks,
                &segment.local_to_global_fid,
            );
            let late_row = qs_exact::prefix_until_from_prequantized_nomiss(
                &segment.pack,
                ranks,
                &[segment.pack.n_trees()],
                0,
                score,
                work_evals,
                resolved_early,
                |_, prefix_score, qs_work, resolved| {
                    handle_checkpoint(
                        segment.global_cp_idx,
                        segment.checkpoint,
                        prefix_score,
                        qs_work,
                        resolved,
                    )
                },
            )?;
            score = late_row.prefix_score;
            work_evals = late_row.block_evals;
            resolved_early = late_row.resolved_early_trees;
            trees_used = segment.checkpoint;
            if direct_decision.get().is_some() {
                break;
            }
        }
    } else if direct_decision.get().is_none() && !hot_checkpoint_layout.late_values.is_empty() {
        let hot_limit = compiled_hot_pack.n_trees.min(runtime.pack.n_trees());
        let mut late_start = hot_limit;
        for (late_pos, checkpoint) in hot_checkpoint_layout.late_values.iter().copied().enumerate() {
            let feature_off = hot_checkpoint_layout.late_feature_offsets[late_pos] as usize;
            let feature_end = hot_checkpoint_layout.late_feature_offsets[late_pos + 1] as usize;
            if feature_end > feature_off {
                qs_exact::quantize_feature_subset_nomiss_precomputed(
                    runtime.pack.cuts_slice(),
                    feat,
                    ranks,
                    &hot_checkpoint_layout.late_feature_ids[feature_off..feature_end],
                    &hot_checkpoint_layout.late_cut_starts[feature_off..feature_end],
                    &hot_checkpoint_layout.late_cut_ends[feature_off..feature_end],
                );
            }
            let global_cp_idx = hot_checkpoint_layout.late_indices[late_pos];
            let late_row = qs_exact::prefix_until_from_prequantized_nomiss(
                &runtime.pack,
                ranks,
                &[checkpoint],
                late_start,
                score,
                work_evals,
                resolved_early,
                |cp, prefix_score, qs_work, resolved| {
                    handle_checkpoint(global_cp_idx, cp, prefix_score, qs_work, resolved)
                },
            )?;
            score = late_row.prefix_score;
            work_evals = late_row.block_evals;
            resolved_early = late_row.resolved_early_trees;
            trees_used = late_row.trees_used;
            if direct_decision.get().is_some() {
                break;
            }
            late_start = checkpoint;
        }
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
        out.fallback_entry_checkpoint = *runtime
            .calibration
            .checkpoints
            .last()
            .unwrap_or(&0usize);
    }

    Ok(Some(out))
}


fn lookup_rescue_action(
    router: &LoadedRescueRouter,
    anchor_checkpoint: usize,
    row_tau: f32,
    row_fold: i32,
    prefix_score: f32,
) -> RescueAction {
    let tau_bin = mlp_tau_bin(&router.tau_edges, row_tau).min(router.tau_edges.len().saturating_sub(2));
    let gap = row_tau - prefix_score;
    let gap_bin = prefix_gap_bin(&router.gap_edges, gap).min(router.gap_edges.len().saturating_sub(2));
    let shadow_side = if prefix_score >= row_tau { 1 } else { -1 };
    if let Some(action) =
        router
            .actions
            .get(&(anchor_checkpoint, tau_bin, row_fold, shadow_side, gap_bin))
    {
        return *action;
    }
    if let Some(action) = router.defaults.get(&(anchor_checkpoint, tau_bin, shadow_side)) {
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
        checkpoint_work_evals: if trace_enabled { vec![0u32; cp_len] } else { Vec::new() },
        checkpoint_deltas: if trace_enabled { vec![0.0f32; cp_len] } else { Vec::new() },
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
        let delta_bin =
            prefix_gap_bin(&scheduler.policy.delta_edges_by_step[step], prev_delta);
        let ranking = lookup_packet_policy_ranking(
            &scheduler.policy,
            step,
            tau_bin,
            gap_bin,
            delta_bin,
        );
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
            out.checkpoint_work_evals[step] =
                work_evals.min(u32::MAX as u64) as u32;
            out.checkpoint_deltas[step] = packet_score;
            out.checkpoint_resolved_early[step] =
                resolved_early_trees.min(u32::MAX as u64) as u32;
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
        let (remaining_score, _, _) = qs_exact::score_tree_indices_from_quantized(
            &runtime.pack,
            &remaining,
            ranks,
            missing,
        )?;
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
fn run_exact_continuation_nomiss(
    runtime: &LoadedPrefixRuntime,
    _model: &SoaModel,
    feat: &[f32],
    row_tau: f32,
    shadow_row: &PrefixRowShadow,
    ranks: &mut [u8],
) -> Result<(f32, bool, i32, RowMeta)> {
    if let Some(anchor_rescue) = runtime.anchor_rescue.as_ref() {
        qs_exact::quantize_into_nomiss(&runtime.pack, feat, ranks);
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
            qs_exact::score_tree_indices_from_quantized_nomiss(&runtime.pack, &remaining, ranks)?;
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
                qs_exact::score_tree_indices_from_quantized_nomiss(
                    &runtime.pack,
                    &packet.tree_indices,
                    ranks,
                )?
            };
            score += packet_score;
            visited += packet.tree_count as i32;
        }
        Ok((score, score >= row_tau, visited, RowMeta::default()))
    } else {
        let fallback_start_tree_idx = shadow_row.trees_used;
        let remaining = &runtime.plan.order[fallback_start_tree_idx.min(runtime.plan.order.len())..];
        let (tail_score, _, _) =
            qs_exact::score_tree_indices_from_quantized_nomiss(&runtime.pack, remaining, ranks)?;
        let score = shadow_row.prefix_score + tail_score;
        let reject = score >= row_tau;
        Ok((score, reject, remaining.len() as i32, RowMeta::default()))
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
