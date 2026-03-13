fn run_qs_l2_exact(args: QsL2ExactArgs) -> Result<()> {
    let pack = qs_exact::load_qs_pack(&args.qs_pack)?;
    let batch = load_features(&args.feat_bin)?;
    let n = args.max_rows.unwrap_or(batch.n_rows).min(batch.n_rows);

    let thread_pool = build_thread_pool(args.threads)?;
    let t0 = Instant::now();
    let (scores, parallel_enabled, used_threads, agg) = qs_exact::run_qs_exact(
        &pack,
        &batch,
        n,
        args.threads,
        args.chunk_rows.max(1),
        args.parallel_min_rows.max(1),
        thread_pool.as_ref(),
    )?;
    let elapsed = t0.elapsed().as_secs_f64();

    if let Some(path) = &args.out_tsv {
        write_qs_output_tsv(path, &batch, n, &scores)?;
    }

    let stats = QsExactStats {
        n_rows: n,
        n_features: pack.n_features,
        n_trees: pack.n_trees,
        n_blocks: pack.n_blocks,
        threads: used_threads,
        parallel_enabled,
        feature_format: batch.format_tag.clone(),
        nan_free: batch.nan_free,
        elapsed_sec: elapsed,
        rows_per_sec: n as f64 / elapsed.max(1e-9),
        rss_peak_mb: peak_rss_mb(),
        avg_blocks_per_row: agg.total_block_evals as f64 / n.max(1) as f64,
        avg_blocks_per_tree: agg.total_block_evals as f64 / (n.max(1) * pack.n_trees).max(1) as f64,
        resolved_early_rate: agg.resolved_early_trees as f64 / (n.max(1) * pack.n_trees).max(1) as f64,
    };

    if let Some(path) = &args.stats_json {
        let txt = serde_json::to_string_pretty(&stats)?;
        fs::write(path, txt).with_context(|| format!("write {}", path.display()))?;
    }

    println!(
        "mode=l2-qs-exact rows={} threads={} parallel={} rows/s={:.1} avg_blocks/row={:.1} resolved_early_rate={:.6} format={}",
        stats.n_rows,
        stats.threads,
        stats.parallel_enabled,
        stats.rows_per_sec,
        stats.avg_blocks_per_row,
        stats.resolved_early_rate,
        stats.feature_format,
    );

    Ok(())
}

fn run_compile_hot_prefix(args: CompileHotPrefixArgs) -> Result<()> {
    let pack = load_prefix_pack(&args.prefix_pack)?;
    let compiled = compile_hot_prefix_pack(&pack)?;
    let out_path = args
        .output
        .unwrap_or_else(|| default_compiled_hot_pack_path(&args.prefix_pack));
    save_compiled_hot_prefix_pack(&out_path, &compiled)?;
    println!(
        "compiled_hot_prefix input={} output={} hot_features={} trees={} nodes={}",
        args.prefix_pack.display(),
        out_path.display(),
        compiled.n_hot_features,
        compiled.n_trees,
        compiled.nodes.len()
    );
    Ok(())
}

fn run_repack_l2_prefix_v2(args: RepackL2PrefixV2Args) -> Result<()> {
    let prefix_args = QsL2PrefixCalArgs {
        bundle_manifest: Some(args.bundle_manifest.clone()),
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
    let model = load_soa(&resolved.soa)?;
    let bounds = load_bounds(&resolved.bounds)?;
    let runtime = load_prefix_runtime(&resolved, &model, &bounds)?;
    let hot_pack_path = resolved
        .hot_exact_prefix_pack
        .as_ref()
        .context("prefix_v2 requires hot exact prefix pack")?;
    let out_dir = args.output_dir.unwrap_or_else(|| {
        hot_pack_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    });
    fs::create_dir_all(&out_dir).with_context(|| format!("mkdir -p {}", out_dir.display()))?;
    let manifest_path = out_dir.join("prefix_v2_manifest.json");
    let stage_plan = runtime
        .hot_stage_plan
        .as_ref()
        .context("prefix_v2 requires hot_stage_plan")?;
    let hot_layout = runtime
        .hot_checkpoint_layout
        .as_ref()
        .context("prefix_v2 requires hot_checkpoint_layout")?;
    let hot_stage_plan = build_prefix_v2_hot_stage_plan(stage_plan, &hot_layout.hot_indices)?;
    let hot_stage_plan_path = out_dir.join("prefix_v2_hot_stage.bin");
    save_prefix_v2_hot_stage_plan(&hot_stage_plan_path, &hot_stage_plan)?;
    let manifest = PrefixV2Manifest {
        format: "L2PrefixV2".to_string(),
        direct_kernel: prefix_direct_kernel_str(runtime.direct_kernel).to_string(),
        certifier_kind: prefix_certifier_kind_str(runtime.certifier_kind).to_string(),
        compiled_hot_pack_file: default_compiled_hot_pack_path(hot_pack_path)
            .to_string_lossy()
            .to_string(),
        certify_plan_file: resolved
            .atlas_bin
            .as_ref()
            .map(default_certify_plan_path)
            .filter(|p| p.exists())
            .map(|p| p.to_string_lossy().to_string()),
        late_segment_manifest_file: Some(
            default_late_segment_manifest_path(&resolved.qs_pack)
                .to_string_lossy()
                .to_string(),
        ),
        hot_stage_plan_file: hot_stage_plan_path.to_string_lossy().to_string(),
    };
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("write {}", manifest_path.display()))?;
    println!(
        "repack_l2_prefix_v2 bundle_manifest={} manifest={} stages={}",
        args.bundle_manifest.display(),
        manifest_path.display(),
        hot_stage_plan.stages.len()
    );
    Ok(())
}

fn run_repack_l2_runtime_v2(args: RepackL2RuntimeV2Args) -> Result<()> {
    let pack = qs_exact::load_qs_pack(&args.qs_pack)?;
    let out_path = args.output.unwrap_or_else(|| {
        let mut raw = args.qs_pack.as_os_str().to_os_string();
        raw.push(".v2.bin");
        PathBuf::from(raw)
    });
    qs_exact::save_qs_pack_v2(&out_path, &pack)?;
    println!(
        "repack_l2_runtime_v2 input={} output={} n_features={} n_trees={} n_blocks={}",
        args.qs_pack.display(),
        out_path.display(),
        pack.n_features,
        pack.n_trees,
        pack.n_blocks,
    );
    Ok(())
}

fn run_repack_l2_runtime_v3(args: RepackL2RuntimeV3Args) -> Result<()> {
    let prefix_args = QsL2PrefixCalArgs {
        bundle_manifest: Some(args.bundle_manifest.clone()),
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
    let model = load_soa(&resolved.soa)?;
    let bounds = load_bounds(&resolved.bounds)?;
    let runtime = load_prefix_runtime(&resolved, &model, &bounds)?;
    println!(
        "repack_l2_runtime_v3 bundle_manifest={} qs_pack={} compiled_hot={} hot_layout={} late_segments={}",
        args.bundle_manifest.display(),
        resolved.qs_pack.display(),
        runtime.compiled_hot_pack.is_some(),
        runtime.hot_checkpoint_layout.is_some(),
        runtime
            .late_segment_runtime
            .as_ref()
            .map(|x| x.segments.len())
            .unwrap_or(0),
    );
    Ok(())
}

fn run_repack_l2_exp_v1(args: RepackL2ExpV1Args) -> Result<()> {
    let manifest_path =
        l2_exp_v1::repack_from_bundle_manifest(&args.bundle_manifest, args.output_dir.as_ref())?;
    println!(
        "repack_l2_exp_v1 bundle_manifest={} manifest={}",
        args.bundle_manifest.display(),
        manifest_path.display(),
    );
    Ok(())
}

fn run_repack_l2_kernel_7945hx_v1(args: RepackL2Kernel7945hxV1Args) -> Result<()> {
    let manifest_path = l2_kernel_7945hx::repack_from_bundle_manifest(
        &args.bundle_manifest,
        args.output_dir.as_ref(),
    )?;
    info!(
        "repack_l2_kernel_7945hx_v1 bundle_manifest={} manifest={}",
        args.bundle_manifest.display(),
        manifest_path.display()
    );
    Ok(())
}

fn run_qs_l2_fast(args: QsL2FastArgs) -> Result<()> {
    let pack = qs_exact::load_qs_pack(&args.qs_pack)?;
    let batch = load_features(&args.feat_bin)?;
    let route_meta = load_route_meta(&args.route_meta)?;
    let model = load_soa(&args.soa)?;
    let bounds = load_bounds(&args.bounds)?;
    let raw_tree_order = load_tree_order(&args.tree_order)?;
    let plan = materialize_tree_plan(&bounds, Some(&raw_tree_order), None)?;
    let base_score = parse_base_score(&args.model_json)?;

    if pack.n_features() != batch.n_cols {
        bail!(
            "feature count mismatch: feat_bin={} qs_pack={}",
            batch.n_cols,
            pack.n_features()
        );
    }
    if route_meta.n_rows != batch.n_rows {
        bail!(
            "route meta row mismatch: route_meta={} feat_bin={}",
            route_meta.n_rows,
            batch.n_rows
        );
    }
    for i in 0..batch.n_rows {
        if batch.id(i) != route_meta.ids[i] {
            bail!(
                "route meta TransactionID mismatch at row {}: feat_bin={} route_meta={}",
                i,
                batch.id(i),
                route_meta.ids[i]
            );
        }
    }

    let n = args.max_rows.unwrap_or(batch.n_rows).min(batch.n_rows);
    let mut lower = vec![f32::NAN; n];
    let mut upper = vec![f32::NAN; n];
    let mut shadow_reject = vec![false; n];
    let mut final_reject = vec![false; n];
    let mut fallback_used = vec![0u8; n];
    let mut route_scores = vec![f32::NAN; n];
    let mut block_evals = vec![0u32; n];
    let mut exact_visits = vec![0i32; n];

    let par_threads = active_threads(args.threads);
    let parallel_enabled = par_threads > 1 && n >= args.parallel_min_rows.max(1);
    let hist_len = plan.n_trees + 1;
    let thread_pool = build_thread_pool(args.threads)?;
    let t0 = Instant::now();

    let mut work = || -> Result<(usize, usize, usize, usize, usize, usize, u64, u64, u64, Vec<u64>)> {
        if parallel_enabled {
            lower
                .par_chunks_mut(args.chunk_rows.max(1))
                .zip(upper.par_chunks_mut(args.chunk_rows.max(1)))
                .zip(shadow_reject.par_chunks_mut(args.chunk_rows.max(1)))
                .zip(final_reject.par_chunks_mut(args.chunk_rows.max(1)))
                .zip(fallback_used.par_chunks_mut(args.chunk_rows.max(1)))
                .zip(route_scores.par_chunks_mut(args.chunk_rows.max(1)))
                .zip(block_evals.par_chunks_mut(args.chunk_rows.max(1)))
                .zip(exact_visits.par_chunks_mut(args.chunk_rows.max(1)))
                .enumerate()
                .map(
                    |(
                        chunk_idx,
                        (((((((lower_chunk, upper_chunk), shadow_chunk), final_chunk), fallback_chunk), score_chunk), block_chunk), visit_chunk),
                    )| -> Result<(usize, usize, usize, usize, usize, usize, u64, u64, u64, Vec<u64>)> {
                        let start = chunk_idx * args.chunk_rows.max(1);
                        let mut ranks = vec![0u8; pack.n_features()];
                        let mut missing = vec![0u8; pack.n_features()];
                        let mut active_cnt = 0usize;
                        let mut shadow_reject_cnt = 0usize;
                        let mut final_reject_cnt = 0usize;
                        let mut direct_reject_cnt = 0usize;
                        let mut direct_refer_cnt = 0usize;
                        let mut fallback_cnt = 0usize;
                        let mut total_block_evals = 0u64;
                        let mut resolved_early_trees = 0u64;
                        let mut exact_visit_sum = 0u64;
                        let mut exact_hist = vec![0u64; hist_len];

                        for local_idx in 0..lower_chunk.len() {
                            let row_idx = start + local_idx;
                            let active = route_meta.active[row_idx] != 0;
                            if !active {
                                shadow_chunk[local_idx] = false;
                                final_chunk[local_idx] = false;
                                fallback_chunk[local_idx] = 0;
                                block_chunk[local_idx] = 0;
                                visit_chunk[local_idx] = 0;
                                score_chunk[local_idx] = f32::NAN;
                                lower_chunk[local_idx] = f32::NAN;
                                upper_chunk[local_idx] = f32::NAN;
                                continue;
                            }

                            active_cnt += 1;
                            let feat = batch.row(row_idx);
                            let interval = qs_exact::interval_row(&pack, feat, &mut ranks, &mut missing)?;
                            let row_tau = route_meta.tau_used[row_idx];
                            let shadow_is_reject = interval.lower >= row_tau;
                            let direct_refer = interval.upper < row_tau;

                            lower_chunk[local_idx] = interval.lower;
                            upper_chunk[local_idx] = interval.upper;
                            block_chunk[local_idx] = interval.block_evals.min(u32::MAX as u64) as u32;
                            total_block_evals += interval.block_evals;
                            resolved_early_trees += interval.resolved_early_trees;
                            shadow_chunk[local_idx] = shadow_is_reject;
                            if shadow_is_reject {
                                shadow_reject_cnt += 1;
                            }

                            if shadow_is_reject {
                                direct_reject_cnt += 1;
                                final_reject_cnt += 1;
                                final_chunk[local_idx] = true;
                                fallback_chunk[local_idx] = 0;
                                score_chunk[local_idx] = interval.lower;
                                visit_chunk[local_idx] = 0;
                            } else if direct_refer {
                                direct_refer_cnt += 1;
                                final_chunk[local_idx] = false;
                                fallback_chunk[local_idx] = 0;
                                score_chunk[local_idx] = interval.upper;
                                visit_chunk[local_idx] = 0;
                            } else {
                                fallback_cnt += 1;
                                fallback_chunk[local_idx] = 1;
                                if args.shadow_only {
                                    final_chunk[local_idx] = false;
                                    score_chunk[local_idx] = interval.upper;
                                    visit_chunk[local_idx] = 0;
                                } else {
                                    let (exact_score, exact_reject_row, visit, _) = if batch.nan_free {
                                        unsafe {
                                            traverse_float_nomiss(
                                                &model,
                                                feat,
                                                row_tau,
                                                base_score,
                                                InferMode::L2RouteExactReordered,
                                                &plan,
                                                None,
                                                0.0,
                                                0.0,
                                                1,
                                            )
                                        }?
                                    } else {
                                        traverse_generic(
                                            &model,
                                            None,
                                            None,
                                            feat,
                                            row_tau,
                                            base_score,
                                            InferMode::L2RouteExactReordered,
                                            &plan,
                                            None,
                                            0.0,
                                            0.0,
                                            1,
                                        )?
                                    };
                                    if exact_reject_row {
                                        final_reject_cnt += 1;
                                    }
                                    let visit_usize = visit.max(0) as usize;
                                    if visit_usize < exact_hist.len() {
                                        exact_hist[visit_usize] += 1;
                                    }
                                    exact_visit_sum += visit_usize as u64;
                                    final_chunk[local_idx] = exact_reject_row;
                                    score_chunk[local_idx] = exact_score;
                                    visit_chunk[local_idx] = visit;
                                }
                            }
                        }

                        Ok((
                            active_cnt,
                            shadow_reject_cnt,
                            final_reject_cnt,
                            direct_reject_cnt,
                            direct_refer_cnt,
                            fallback_cnt,
                            total_block_evals,
                            resolved_early_trees,
                            exact_visit_sum,
                            exact_hist,
                        ))
                    },
                )
                .try_reduce(
                    || (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0u64, 0u64, 0u64, vec![0u64; hist_len]),
                    |a, b| {
                        let mut hist = a.9;
                        for (dst, src) in hist.iter_mut().zip(b.9.iter()) {
                            *dst += *src;
                        }
                        Ok((
                            a.0 + b.0,
                            a.1 + b.1,
                            a.2 + b.2,
                            a.3 + b.3,
                            a.4 + b.4,
                            a.5 + b.5,
                            a.6 + b.6,
                            a.7 + b.7,
                            a.8 + b.8,
                            hist,
                        ))
                    },
                )
        } else {
            let mut ranks = vec![0u8; pack.n_features()];
            let mut missing = vec![0u8; pack.n_features()];
            let mut active_cnt = 0usize;
            let mut shadow_reject_cnt = 0usize;
            let mut final_reject_cnt = 0usize;
            let mut direct_reject_cnt = 0usize;
            let mut direct_refer_cnt = 0usize;
            let mut fallback_cnt = 0usize;
            let mut total_block_evals = 0u64;
            let mut resolved_early_trees = 0u64;
            let mut exact_visit_sum = 0u64;
            let mut exact_hist = vec![0u64; hist_len];

            for row_idx in 0..n {
                let active = route_meta.active[row_idx] != 0;
                if !active {
                    continue;
                }

                active_cnt += 1;
                let feat = batch.row(row_idx);
                let interval = qs_exact::interval_row(&pack, feat, &mut ranks, &mut missing)?;
                let row_tau = route_meta.tau_used[row_idx];
                let shadow_is_reject = interval.lower >= row_tau;
                let direct_refer = interval.upper < row_tau;

                lower[row_idx] = interval.lower;
                upper[row_idx] = interval.upper;
                block_evals[row_idx] = interval.block_evals.min(u32::MAX as u64) as u32;
                total_block_evals += interval.block_evals;
                resolved_early_trees += interval.resolved_early_trees;
                shadow_reject[row_idx] = shadow_is_reject;
                if shadow_is_reject {
                    shadow_reject_cnt += 1;
                }

                if shadow_is_reject {
                    direct_reject_cnt += 1;
                    final_reject_cnt += 1;
                    final_reject[row_idx] = true;
                    route_scores[row_idx] = interval.lower;
                } else if direct_refer {
                    direct_refer_cnt += 1;
                    final_reject[row_idx] = false;
                    route_scores[row_idx] = interval.upper;
                } else {
                    fallback_cnt += 1;
                    fallback_used[row_idx] = 1;
                    if args.shadow_only {
                        final_reject[row_idx] = false;
                        route_scores[row_idx] = interval.upper;
                    } else {
                        let (exact_score, exact_reject_row, visit, _) = if batch.nan_free {
                            unsafe {
                                traverse_float_nomiss(
                                    &model,
                                    feat,
                                    row_tau,
                                    base_score,
                                    InferMode::L2RouteExactReordered,
                                    &plan,
                                    None,
                                    0.0,
                                    0.0,
                                    1,
                                )
                            }?
                        } else {
                            traverse_generic(
                                &model,
                                None,
                                None,
                                feat,
                                row_tau,
                                base_score,
                                InferMode::L2RouteExactReordered,
                                &plan,
                                None,
                                0.0,
                                0.0,
                                1,
                            )?
                        };
                        if exact_reject_row {
                            final_reject_cnt += 1;
                        }
                        let visit_usize = visit.max(0) as usize;
                        if visit_usize < exact_hist.len() {
                            exact_hist[visit_usize] += 1;
                        }
                        exact_visit_sum += visit_usize as u64;
                        final_reject[row_idx] = exact_reject_row;
                        route_scores[row_idx] = exact_score;
                        exact_visits[row_idx] = visit;
                    }
                }
            }

            Ok((
                active_cnt,
                shadow_reject_cnt,
                final_reject_cnt,
                direct_reject_cnt,
                direct_refer_cnt,
                fallback_cnt,
                total_block_evals,
                resolved_early_trees,
                exact_visit_sum,
                exact_hist,
            ))
        }
    };
    let (
        active_cnt,
        shadow_reject_cnt,
        final_reject_cnt,
        direct_reject_cnt,
        direct_refer_cnt,
        fallback_cnt,
        total_block_evals,
        resolved_early_trees,
        exact_visit_sum,
        exact_visit_hist,
    ) = install_in_pool(thread_pool.as_ref(), work)?;
    let elapsed = t0.elapsed().as_secs_f64();

    if let Some(path) = &args.out_tsv {
        write_qs_fast_output_tsv(
            path,
            &batch,
            n,
            &lower,
            &upper,
            &shadow_reject,
            &final_reject,
            &fallback_used,
            &route_scores,
            &block_evals,
            &exact_visits,
        )?;
    }

    let active_f = active_cnt.max(1) as f64;
    let fallback_n = fallback_cnt.max(1);
    let stats = QsFastStats {
        n_rows: n,
        n_features: pack.n_features(),
        n_trees: pack.n_trees(),
        n_blocks: pack.n_blocks(),
        threads: par_threads,
        parallel_enabled,
        feature_format: batch.format_tag.clone(),
        nan_free: batch.nan_free,
        shadow_only: args.shadow_only,
        elapsed_sec: elapsed,
        rows_per_sec: n as f64 / elapsed.max(1e-9),
        rss_peak_mb: peak_rss_mb(),
        route_active_rate: active_cnt as f64 / n.max(1) as f64,
        shadow_reject_rate: shadow_reject_cnt as f64 / active_f,
        final_reject_rate: final_reject_cnt as f64 / active_f,
        direct_reject_rate: direct_reject_cnt as f64 / active_f,
        direct_refer_rate: direct_refer_cnt as f64 / active_f,
        fallback_rate: fallback_cnt as f64 / active_f,
        avg_blocks_per_row: total_block_evals as f64 / active_f,
        avg_blocks_per_tree: total_block_evals as f64 / ((active_cnt.max(1) * pack.n_trees()).max(1) as f64),
        resolved_early_rate: resolved_early_trees as f64
            / ((active_cnt.max(1) * pack.n_trees()).max(1) as f64),
        exact_avg_visited_trees: exact_visit_sum as f64 / fallback_n as f64,
        exact_p99_visited_trees: pct_from_hist(&exact_visit_hist, 0.99, fallback_cnt),
        exact_visit_hist,
    };

    if let Some(path) = &args.stats_json {
        let txt = serde_json::to_string_pretty(&stats)?;
        fs::write(path, txt).with_context(|| format!("write {}", path.display()))?;
    }

    println!(
        "mode=l2-qs-fast rows={} threads={} parallel={} rows/s={:.1} active_rate={:.4} fallback_rate={:.4} avg_blocks/row={:.1}",
        stats.n_rows,
        stats.threads,
        stats.parallel_enabled,
        stats.rows_per_sec,
        stats.route_active_rate,
        stats.fallback_rate,
        stats.avg_blocks_per_row,
    );

    Ok(())
}

fn run_qs_l2_prefix_cal_single_route_hot(
    runtime: &LoadedPrefixRuntime,
    model: &SoaModel,
    batch: &FeatureBatch,
    route_meta: &RouteMeta,
    args: &QsL2PrefixCalArgs,
    n: usize,
    hist_len: usize,
    parallel_enabled: bool,
    thread_pool: Option<&rayon::ThreadPool>,
    prefix_scores: &mut [f32],
    trees_used: &mut [u16],
    shadow_reject: &mut [bool],
    final_reject: &mut [bool],
    fallback_used: &mut [u8],
    direct_checkpoints: &mut [u16],
    fallback_entry_checkpoints: &mut [u16],
    route_scores: &mut [f32],
    block_evals: &mut [u32],
    exact_visits: &mut [i32],
) -> Result<(usize, usize, usize, usize, usize, usize, u64, u64, u64, u64, Vec<u64>)> {
    #[derive(Default)]
    struct L2SingleRouteAgg {
        active_cnt: usize,
        shadow_reject_cnt: usize,
        final_reject_cnt: usize,
        direct_reject_cnt: usize,
        direct_refer_cnt: usize,
        fallback_cnt: usize,
        total_block_evals: u64,
        resolved_early_trees: u64,
        total_prefix_trees: u64,
    }

    impl L2SingleRouteAgg {
        #[inline(always)]
        fn merge(&mut self, other: Self) {
            self.active_cnt += other.active_cnt;
            self.shadow_reject_cnt += other.shadow_reject_cnt;
            self.final_reject_cnt += other.final_reject_cnt;
            self.direct_reject_cnt += other.direct_reject_cnt;
            self.direct_refer_cnt += other.direct_refer_cnt;
            self.fallback_cnt += other.fallback_cnt;
            self.total_block_evals += other.total_block_evals;
            self.resolved_early_trees += other.resolved_early_trees;
            self.total_prefix_trees += other.total_prefix_trees;
        }
    }

    struct L2SingleRouteScratch {
        ranks: Vec<u8>,
        missing: Vec<u8>,
        hot_buf: Option<HotFeatureBuf>,
        hot_cache: Option<LazyHotFeatureCache>,
    }

    impl L2SingleRouteScratch {
        #[inline]
        fn new(runtime: &LoadedPrefixRuntime, n_features: usize) -> Self {
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

    let n_features = runtime.pack.n_features();
    let chunk_rows = args.chunk_rows.max(1);

    let work = || -> Result<L2SingleRouteAgg> {
            if parallel_enabled {
                prefix_scores
                    .par_chunks_mut(chunk_rows)
                    .zip(trees_used.par_chunks_mut(chunk_rows))
                    .zip(shadow_reject.par_chunks_mut(chunk_rows))
                    .zip(final_reject.par_chunks_mut(chunk_rows))
                    .zip(fallback_used.par_chunks_mut(chunk_rows))
                    .zip(direct_checkpoints.par_chunks_mut(chunk_rows))
                    .zip(fallback_entry_checkpoints.par_chunks_mut(chunk_rows))
                    .zip(route_scores.par_chunks_mut(chunk_rows))
                    .zip(block_evals.par_chunks_mut(chunk_rows))
                    .zip(exact_visits.par_chunks_mut(chunk_rows))
                    .enumerate()
                    .map_init(
                        || L2SingleRouteScratch::new(runtime, n_features),
                        |scratch,
                         (
                            chunk_idx,
                            (((((((((prefix_chunk, trees_chunk), shadow_chunk), final_chunk), fallback_chunk), direct_cp_chunk), fallback_cp_chunk), score_chunk), block_chunk), visit_chunk),
                        )| -> Result<L2SingleRouteAgg> {
                            let start = chunk_idx * chunk_rows;
                            let mut agg = L2SingleRouteAgg::default();

                            for local_idx in 0..prefix_chunk.len() {
                                let row_idx = start + local_idx;
                                if route_meta.active[row_idx] == 0 {
                                    continue;
                                }

                                agg.active_cnt += 1;
                                let feat = batch.row(row_idx);
                                let row_tau = route_meta.tau_used[row_idx];
                                let row_fold = route_meta.fold_id[row_idx];
                                let shadow_row = run_prefix_shadow_row_single_route_compiled(
                                    runtime,
                                    feat,
                                    row_tau,
                                    row_fold,
                                    batch.nan_free,
                                    &mut scratch.ranks,
                                    &mut scratch.missing,
                                    scratch.hot_buf.as_mut(),
                                    scratch.hot_cache.as_mut(),
                                )?;

                                prefix_chunk[local_idx] = shadow_row.prefix_score;
                                trees_chunk[local_idx] =
                                    shadow_row.trees_used.min(u16::MAX as usize) as u16;
                                block_chunk[local_idx] =
                                    shadow_row.work_evals.min(u32::MAX as u64) as u32;
                                direct_cp_chunk[local_idx] =
                                    shadow_row.direct_checkpoint.min(u16::MAX as usize) as u16;
                                fallback_cp_chunk[local_idx] = shadow_row
                                    .fallback_entry_checkpoint
                                    .min(u16::MAX as usize) as u16;
                                agg.total_block_evals += shadow_row.work_evals;
                                agg.resolved_early_trees += shadow_row.resolved_early_trees;
                                agg.total_prefix_trees += shadow_row.trees_used as u64;

                                if !shadow_row.fallback_used {
                                    let is_reject = shadow_row.shadow_reject;
                                    shadow_chunk[local_idx] = is_reject;
                                    final_chunk[local_idx] = is_reject;
                                    score_chunk[local_idx] = shadow_row.shadow_route_score;
                                    if is_reject {
                                        agg.shadow_reject_cnt += 1;
                                        agg.final_reject_cnt += 1;
                                        agg.direct_reject_cnt += 1;
                                    } else {
                                        agg.direct_refer_cnt += 1;
                                    }
                                } else {
                                    agg.fallback_cnt += 1;
                                    fallback_chunk[local_idx] = 1;
                                    score_chunk[local_idx] = shadow_row.prefix_score;
                                    if args.shadow_only {
                                        final_chunk[local_idx] = false;
                                    } else {
                                        let (exact_score, exact_reject_row, visit, _) =
                                            run_exact_continuation(
                                                runtime,
                                                model,
                                                feat,
                                                row_tau,
                                                &shadow_row,
                                                batch.nan_free,
                                                &mut scratch.ranks,
                                                &mut scratch.missing,
                                            )?;
                                        if exact_reject_row {
                                            agg.final_reject_cnt += 1;
                                        }
                                        final_chunk[local_idx] = exact_reject_row;
                                        score_chunk[local_idx] = exact_score;
                                        visit_chunk[local_idx] = visit;
                                    }
                                }
                            }

                            Ok(agg)
                        },
                    )
                    .try_reduce(
                        L2SingleRouteAgg::default,
                        |a, b| {
                            let mut acc = a;
                            acc.merge(b);
                            Ok(acc)
                        },
                    )
            } else {
                let mut scratch = L2SingleRouteScratch::new(runtime, n_features);
                let mut agg = L2SingleRouteAgg::default();

                for row_idx in 0..n {
                    if route_meta.active[row_idx] == 0 {
                        continue;
                    }

                    agg.active_cnt += 1;
                    let feat = batch.row(row_idx);
                    let row_tau = route_meta.tau_used[row_idx];
                    let row_fold = route_meta.fold_id[row_idx];
                    let shadow_row = run_prefix_shadow_row_single_route_compiled(
                        runtime,
                        feat,
                        row_tau,
                        row_fold,
                        batch.nan_free,
                        &mut scratch.ranks,
                        &mut scratch.missing,
                        scratch.hot_buf.as_mut(),
                        scratch.hot_cache.as_mut(),
                    )?;

                    prefix_scores[row_idx] = shadow_row.prefix_score;
                    trees_used[row_idx] = shadow_row.trees_used.min(u16::MAX as usize) as u16;
                    block_evals[row_idx] = shadow_row.work_evals.min(u32::MAX as u64) as u32;
                    direct_checkpoints[row_idx] =
                        shadow_row.direct_checkpoint.min(u16::MAX as usize) as u16;
                    fallback_entry_checkpoints[row_idx] =
                        shadow_row.fallback_entry_checkpoint.min(u16::MAX as usize) as u16;
                    agg.total_block_evals += shadow_row.work_evals;
                    agg.resolved_early_trees += shadow_row.resolved_early_trees;
                    agg.total_prefix_trees += shadow_row.trees_used as u64;

                    if !shadow_row.fallback_used {
                        let is_reject = shadow_row.shadow_reject;
                        shadow_reject[row_idx] = is_reject;
                        final_reject[row_idx] = is_reject;
                        route_scores[row_idx] = shadow_row.shadow_route_score;
                        if is_reject {
                            agg.shadow_reject_cnt += 1;
                            agg.final_reject_cnt += 1;
                            agg.direct_reject_cnt += 1;
                        } else {
                            agg.direct_refer_cnt += 1;
                        }
                    } else {
                        agg.fallback_cnt += 1;
                        fallback_used[row_idx] = 1;
                        route_scores[row_idx] = shadow_row.prefix_score;
                        if args.shadow_only {
                            final_reject[row_idx] = false;
                        } else {
                            let (exact_score, exact_reject_row, visit, _) =
                                run_exact_continuation(
                                    runtime,
                                    model,
                                    feat,
                                    row_tau,
                                    &shadow_row,
                                    batch.nan_free,
                                    &mut scratch.ranks,
                                    &mut scratch.missing,
                                )?;
                            if exact_reject_row {
                                agg.final_reject_cnt += 1;
                            }
                            final_reject[row_idx] = exact_reject_row;
                            route_scores[row_idx] = exact_score;
                            exact_visits[row_idx] = visit;
                        }
                    }
                }

                Ok(agg)
            }
        };

    let agg = install_in_pool(thread_pool, work)?;
    let mut exact_visit_sum = 0u64;
    let mut exact_visit_hist = vec![0u64; hist_len];
    if !args.shadow_only {
        for (fallback_flag, visit) in fallback_used.iter().zip(exact_visits.iter()) {
            if *fallback_flag == 0 {
                continue;
            }
            let visit_usize = (*visit).max(0) as usize;
            if visit_usize < exact_visit_hist.len() {
                exact_visit_hist[visit_usize] += 1;
            }
            exact_visit_sum += visit_usize as u64;
        }
    }
    Ok((
        agg.active_cnt,
        agg.shadow_reject_cnt,
        agg.final_reject_cnt,
        agg.direct_reject_cnt,
        agg.direct_refer_cnt,
        agg.fallback_cnt,
        agg.total_block_evals,
        agg.resolved_early_trees,
        agg.total_prefix_trees,
        exact_visit_sum,
        exact_visit_hist,
    ))
}

fn run_qs_l2_prefix_cal(args: QsL2PrefixCalArgs) -> Result<()> {
    let parsed_manifest = if let Some(path) = &args.bundle_manifest {
        let txt = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let parsed: PrefixCalBundleManifest =
            serde_json::from_str(&txt).with_context(|| format!("parse {}", path.display()))?;
        Some(parsed)
    } else {
        None
    };

    let (
        batch,
        route_meta,
        model,
        runtimes,
        tau_router_edges,
        v4_mode,
    ): (
        FeatureBatch,
        RouteMeta,
        SoaModel,
        Vec<LoadedPrefixOrderRoute>,
        Vec<f32>,
        String,
    ) =
        if let (Some(path), Some(parsed)) = (&args.bundle_manifest, parsed_manifest.as_ref()) {
            match parsed.format.as_str() {
                "L2PrefixCalBundleV1" => {
                    let resolved = resolve_prefix_cal_bundle(&args)?;
                    let batch = load_features(&resolved.feat_bin)?;
                    let route_meta = load_route_meta(&resolved.route_meta)?;
                    let model = load_soa(&resolved.soa)?;
                    let bounds = load_bounds(&resolved.bounds)?;
                    let runtime = load_prefix_runtime(&resolved, &model, &bounds)?;
                    (
                        batch,
                        route_meta,
                        model,
                        vec![LoadedPrefixOrderRoute {
                            tau_bin: 0,
                            label: "single".to_string(),
                            runtime,
                        }],
                        vec![-f32::MAX, f32::MAX],
                        "single_order_v1".to_string(),
                    )
                }
                "L2PrefixCalBundleV4" => {
                    let mode = parse_prefix_v4_mode(
                        parsed
                            .v4_mode
                            .as_ref()
                            .context("missing required prefix-cal V4 field: v4_mode")?,
                    )?;
                    let feat_bin = resolve_bundle_path(
                        args.feat_bin.as_ref(),
                        parsed.feat_bin.as_ref(),
                        Some(path),
                        "feat_bin",
                    )?;
                    let route_meta_path = resolve_bundle_path(
                        args.route_meta.as_ref(),
                        parsed.route_meta.as_ref(),
                        Some(path),
                        "route_meta",
                    )?;
                    let soa_path = resolve_bundle_path(
                        args.soa.as_ref(),
                        parsed.soa_bin.as_ref(),
                        Some(path),
                        "soa_bin",
                    )?;
                    let bounds_path = resolve_bundle_path(
                        args.bounds.as_ref(),
                        parsed.bounds_bin.as_ref(),
                        Some(path),
                        "bounds_bin",
                    )?;
                    let batch = load_features(&feat_bin)?;
                    let route_meta = load_route_meta(&route_meta_path)?;
                    let model = load_soa(&soa_path)?;
                    let bounds = load_bounds(&bounds_path)?;
                    match mode {
                        PrefixV4Mode::MultiOrderV1 => {
                            let (tau_edges, routes, _selected_variant, _telemetry_schema_version) =
                                load_v4_order_router(&args, path, parsed, &model, &bounds)?;
                            (
                                batch,
                                route_meta,
                                model,
                                routes,
                                tau_edges,
                                "multi_order_v1".to_string(),
                            )
                        }
                        PrefixV4Mode::PacketSchedulerV1 => {
                            let child_args = QsL2PrefixCalArgs {
                                bundle_manifest: None,
                                qs_pack: resolve_bundle_path_optional(
                                    None,
                                    parsed.pack_path.as_ref(),
                                    Some(path),
                                ),
                                calibration_json: resolve_bundle_path_optional(
                                    None,
                                    parsed.calibration_json.as_ref(),
                                    Some(path),
                                ),
                                variant_key: args
                                    .variant_key
                                    .clone()
                                    .or_else(|| parsed.selected_exact_variant.clone())
                                    .or_else(|| parsed.selected_variant.clone()),
                                feat_bin: Some(feat_bin.clone()),
                                route_meta: Some(route_meta_path.clone()),
                                soa: Some(soa_path.clone()),
                                bounds: Some(bounds_path.clone()),
                                tree_order: resolve_bundle_path_optional(
                                    None,
                                    parsed.tree_order_bin.as_ref(),
                                    Some(path),
                                ),
                                model_json: resolve_bundle_path_optional(
                                    None,
                                    parsed.model_json.as_ref(),
                                    Some(path),
                                ),
                                direct_kernel: parsed.direct_kernel.clone(),
                                certifier_kind: parsed.certifier_kind.clone(),
                                certifier_json: resolve_bundle_path_optional(
                                    None,
                                    parsed.certifier_json.as_ref(),
                                    Some(path),
                                ),
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
                            let mut runtime = load_prefix_runtime(&resolved, &model, &bounds)?;
                            runtime.packet_scheduler = Some(load_v4_packet_scheduler(path, parsed)?);
                            (
                                batch,
                                route_meta,
                                model,
                                vec![LoadedPrefixOrderRoute {
                                    tau_bin: 0,
                                    label: "packet_scheduler".to_string(),
                                    runtime,
                                }],
                                vec![-f32::MAX, f32::MAX],
                                "packet_scheduler_v1".to_string(),
                            )
                        }
                        PrefixV4Mode::AnchorRescueV1 | PrefixV4Mode::AnchorRescueLossyV1 => {
                            let (runtime, _selected_variant) =
                                load_v4_anchor_rescue(&args, path, parsed, &model, &bounds)?;
                            let mode_name = if mode == PrefixV4Mode::AnchorRescueLossyV1 {
                                "anchor_rescue_lossy_v1"
                            } else {
                                "anchor_rescue_v1"
                            };
                            (
                                batch,
                                route_meta,
                                model,
                                vec![LoadedPrefixOrderRoute {
                                    tau_bin: 0,
                                    label: "anchor_rescue".to_string(),
                                    runtime,
                                }],
                                vec![-f32::MAX, f32::MAX],
                                mode_name.to_string(),
                            )
                        }
                        other => bail!(
                            "V4 mode {:?} not yet supported in qs-l2-prefix-cal runtime",
                            other
                        ),
                    }
                }
                other => bail!("unsupported prefix-cal bundle format: {}", other),
            }
        } else {
            let resolved = resolve_prefix_cal_bundle(&args)?;
            let batch = load_features(&resolved.feat_bin)?;
            let route_meta = load_route_meta(&resolved.route_meta)?;
            let model = load_soa(&resolved.soa)?;
            let bounds = load_bounds(&resolved.bounds)?;
            let runtime = load_prefix_runtime(&resolved, &model, &bounds)?;
            (
                batch,
                route_meta,
                model,
                vec![LoadedPrefixOrderRoute {
                    tau_bin: 0,
                    label: "single".to_string(),
                    runtime,
                }],
                vec![-f32::MAX, f32::MAX],
                "single_order_v1".to_string(),
            )
        };

    if runtimes.is_empty() {
        bail!("no prefix runtime routes loaded");
    }
    let first_runtime = &runtimes[0].runtime;
    let checkpoints = first_runtime.calibration.checkpoints.clone();
    let cp_len = checkpoints.len();
    if cp_len == 0 {
        bail!("prefix calibration has no checkpoints");
    }
    let base_n_features = first_runtime.pack.n_features();
    let base_n_trees = first_runtime.pack.n_trees();
    let hist_len = first_runtime.plan.n_trees + 1;
    let telemetry_schema_version = if runtimes.len() > 1 { 4 } else { first_runtime.telemetry_schema_version };
    for route in runtimes.iter() {
        if route.runtime.pack.n_features() != base_n_features {
            bail!("multi-order route feature count mismatch");
        }
        if route.runtime.pack.n_trees() != base_n_trees {
            bail!("multi-order route tree count mismatch");
        }
        if route.runtime.calibration.checkpoints != checkpoints {
            bail!("multi-order route checkpoint ladder mismatch");
        }
        if route.runtime.plan.n_trees + 1 != hist_len {
            bail!("multi-order route exact plan length mismatch");
        }
    }
    if base_n_features != batch.n_cols {
        bail!(
            "feature count mismatch: feat_bin={} qs_pack={}",
            batch.n_cols,
            base_n_features
        );
    }
    if route_meta.n_rows != batch.n_rows {
        bail!(
            "route meta row mismatch: route_meta={} feat_bin={}",
            route_meta.n_rows,
            batch.n_rows
        );
    }
    for i in 0..batch.n_rows {
        if batch.id(i) != route_meta.ids[i] {
            bail!(
                "route meta TransactionID mismatch at row {}: feat_bin={} route_meta={}",
                i,
                batch.id(i),
                route_meta.ids[i]
            );
        }
    }
    if model.n_features != batch.n_cols {
        bail!(
            "soa feature count mismatch: soa={} feat_bin={}",
            model.n_features,
            batch.n_cols
        );
    }
    let router_bin_count = tau_router_edges.len().saturating_sub(1).max(1);
    let mut router_lookup = vec![usize::MAX; router_bin_count];
    for (idx, route) in runtimes.iter().enumerate() {
        if route.tau_bin >= router_lookup.len() {
            bail!(
                "router tau_bin {} exceeds available tau bins {}",
                route.tau_bin,
                router_lookup.len()
            );
        }
        router_lookup[route.tau_bin] = idx;
    }
    for slot in router_lookup.iter_mut() {
        if *slot == usize::MAX {
            if runtimes.len() == 1 {
                *slot = 0;
            } else {
                bail!("missing router bundle assignment for one or more tau bins");
            }
        }
    }

    let n = args.max_rows.unwrap_or(batch.n_rows).min(batch.n_rows);
    let has_anchor_rescue = runtimes
        .iter()
        .any(|route| route.runtime.anchor_rescue.is_some());
    let mut prefix_scores = vec![f32::NAN; n];
    let mut trees_used = vec![0u16; n];
    let mut shadow_reject = vec![false; n];
    let mut final_reject = vec![false; n];
    let mut fallback_used = vec![0u8; n];
    let mut route_scores = vec![f32::NAN; n];
    let mut block_evals = vec![0u32; n];
    let mut exact_visits = vec![0i32; n];
    let mut direct_checkpoints = vec![0u16; n];
    let mut fallback_entry_checkpoints = vec![0u16; n];
    let mut router_choice = vec![0u16; n];
    let mut router_tau_bin = vec![0u16; n];
    let trace_enabled = args.trace_jsonl.is_some();
    let mut anchor_direct_checkpoints = vec![0u16; n];
    let mut rescue_direct_checkpoints = vec![0u16; n];
    let mut rescue_route_choice = vec![0u8; n];
    let mut checkpoint_scores_flat = if trace_enabled {
        vec![f32::NAN; n * cp_len]
    } else {
        Vec::new()
    };
    let mut checkpoint_work_evals_flat = if trace_enabled {
        vec![0u32; n * cp_len]
    } else {
        Vec::new()
    };
    let mut checkpoint_deltas_flat = if trace_enabled {
        vec![0.0f32; n * cp_len]
    } else {
        Vec::new()
    };
    let mut checkpoint_resolved_flat = if trace_enabled {
        vec![0u32; n * cp_len]
    } else {
        Vec::new()
    };

    let par_threads = active_threads(args.threads);
    let parallel_enabled =
        !trace_enabled && par_threads > 1 && n >= args.parallel_min_rows.max(1);
    let thread_pool = build_thread_pool(args.threads)?;
    let use_single_route_hot =
        !trace_enabled
            && runtimes.len() == 1
            && runtimes[0].runtime.anchor_rescue.is_none()
            && runtimes[0].runtime.packet_scheduler.is_none();
    let t0 = Instant::now();

    let mut work =
        || -> Result<(usize, usize, usize, usize, usize, usize, u64, u64, u64, u64, Vec<u64>)> {
            if parallel_enabled {
                prefix_scores
                    .par_chunks_mut(args.chunk_rows.max(1))
                    .zip(trees_used.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(shadow_reject.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(final_reject.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(fallback_used.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(direct_checkpoints.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(anchor_direct_checkpoints.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(rescue_direct_checkpoints.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(rescue_route_choice.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(fallback_entry_checkpoints.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(router_choice.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(router_tau_bin.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(route_scores.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(block_evals.par_chunks_mut(args.chunk_rows.max(1)))
                    .zip(exact_visits.par_chunks_mut(args.chunk_rows.max(1)))
                    .enumerate()
                    .map(
                        |(chunk_idx, chunk)| -> Result<(usize, usize, usize, usize, usize, usize, u64, u64, u64, u64, Vec<u64>)> {
                            let ((((((((((((((prefix_chunk, trees_chunk), shadow_chunk), final_chunk), fallback_chunk), direct_cp_chunk), anchor_cp_chunk), rescue_cp_chunk), rescue_route_chunk), fallback_cp_chunk), router_choice_chunk), router_tau_chunk), score_chunk), block_chunk), visit_chunk) =
                                chunk;
                            let start = chunk_idx * args.chunk_rows.max(1);
                            let mut ranks = vec![0u8; base_n_features];
                            let mut missing = vec![0u8; base_n_features];
                            let mut hot_buf_by_route: Vec<Option<HotFeatureBuf>> = runtimes
                                .iter()
                                .map(|route| {
                                    route.runtime.compiled_hot_pack.as_ref().map(|pack| {
                                        HotFeatureBuf::new(pack.n_hot_features)
                                    })
                                })
                                .collect();
                            let mut hot_cache_by_route: Vec<Option<LazyHotFeatureCache>> = runtimes
                                .iter()
                                .map(|route| {
                                    route.runtime.hot_pack.as_ref().map(|pack| {
                                        LazyHotFeatureCache::new(pack.n_hot_features)
                                    })
                                })
                                .collect();
                            let mut active_cnt = 0usize;
                            let mut shadow_reject_cnt = 0usize;
                            let mut final_reject_cnt = 0usize;
                            let mut direct_reject_cnt = 0usize;
                            let mut direct_refer_cnt = 0usize;
                            let mut fallback_cnt = 0usize;
                            let mut total_block_evals = 0u64;
                            let mut resolved_early_trees = 0u64;
                            let mut total_prefix_trees = 0u64;
                            let mut exact_visit_sum = 0u64;
                            let mut exact_hist = vec![0u64; hist_len];

                            for local_idx in 0..prefix_chunk.len() {
                                let row_idx = start + local_idx;
                                let active = route_meta.active[row_idx] != 0;
                                if !active {
                                    shadow_chunk[local_idx] = false;
                                    final_chunk[local_idx] = false;
                                    fallback_chunk[local_idx] = 0;
                                    direct_cp_chunk[local_idx] = 0;
                                    anchor_cp_chunk[local_idx] = 0;
                                    rescue_cp_chunk[local_idx] = 0;
                                    rescue_route_chunk[local_idx] = 0;
                                    fallback_cp_chunk[local_idx] = 0;
                                    router_choice_chunk[local_idx] = 0;
                                    router_tau_chunk[local_idx] = 0;
                                    block_chunk[local_idx] = 0;
                                    visit_chunk[local_idx] = 0;
                                    score_chunk[local_idx] = f32::NAN;
                                    prefix_chunk[local_idx] = f32::NAN;
                                    trees_chunk[local_idx] = 0;
                                    continue;
                                }

                                active_cnt += 1;
                                let feat = batch.row(row_idx);
                                let row_tau = route_meta.tau_used[row_idx];
                                let row_fold = route_meta.fold_id[row_idx];
                                let tau_bin = mlp_tau_bin(&tau_router_edges, row_tau)
                                    .min(router_lookup.len().saturating_sub(1));
                                let route_idx = router_lookup[tau_bin];
                                let runtime = &runtimes[route_idx].runtime;
                                router_choice_chunk[local_idx] =
                                    route_idx.min(u16::MAX as usize) as u16;
                                router_tau_chunk[local_idx] =
                                    tau_bin.min(u16::MAX as usize) as u16;
                                let shadow_row = if let Some(anchor_rescue) =
                                    runtime.anchor_rescue.as_ref()
                                {
                                    run_anchor_rescue_shadow_row(
                                        runtime,
                                        anchor_rescue,
                                        feat,
                                        row_tau,
                                        row_fold,
                                        &mut ranks,
                                        &mut missing,
                                        hot_cache_by_route[route_idx].as_mut(),
                                        false,
                                    )?
                                } else if let Some(packet_runtime) =
                                    runtime.packet_scheduler.as_ref()
                                {
                                    run_packet_shadow_row(
                                        runtime,
                                        packet_runtime,
                                        feat,
                                        row_tau,
                                        row_fold,
                                        &mut ranks,
                                        &mut missing,
                                        false,
                                    )?
                                } else {
                                    run_prefix_shadow_row_single_route_compiled(
                                        runtime,
                                        feat,
                                        row_tau,
                                        row_fold,
                                        batch.nan_free,
                                        &mut ranks,
                                        &mut missing,
                                        hot_buf_by_route[route_idx].as_mut(),
                                        hot_cache_by_route[route_idx].as_mut(),
                                    )?
                                };

                                prefix_chunk[local_idx] = shadow_row.prefix_score;
                                trees_chunk[local_idx] =
                                    shadow_row.trees_used.min(u16::MAX as usize) as u16;
                                block_chunk[local_idx] =
                                    shadow_row.work_evals.min(u32::MAX as u64) as u32;
                                direct_cp_chunk[local_idx] =
                                    shadow_row.direct_checkpoint.min(u16::MAX as usize) as u16;
                                anchor_cp_chunk[local_idx] = shadow_row
                                    .anchor_direct_checkpoint
                                    .min(u16::MAX as usize) as u16;
                                rescue_cp_chunk[local_idx] = shadow_row
                                    .rescue_direct_checkpoint
                                    .min(u16::MAX as usize) as u16;
                                rescue_route_chunk[local_idx] = shadow_row.rescue_route;
                                fallback_cp_chunk[local_idx] = shadow_row
                                    .fallback_entry_checkpoint
                                    .min(u16::MAX as usize) as u16;
                                total_block_evals += shadow_row.work_evals;
                                resolved_early_trees += shadow_row.resolved_early_trees;
                                total_prefix_trees += shadow_row.trees_used as u64;

                                if !shadow_row.fallback_used {
                                    let is_reject = shadow_row.shadow_reject;
                                    shadow_chunk[local_idx] = is_reject;
                                    final_chunk[local_idx] = is_reject;
                                    score_chunk[local_idx] = shadow_row.shadow_route_score;
                                    if is_reject {
                                        shadow_reject_cnt += 1;
                                        final_reject_cnt += 1;
                                        direct_reject_cnt += 1;
                                    } else {
                                        direct_refer_cnt += 1;
                                    }
                                } else {
                                    fallback_cnt += 1;
                                    fallback_chunk[local_idx] = 1;
                                    shadow_chunk[local_idx] = false;
                                    score_chunk[local_idx] = shadow_row.prefix_score;
                                    if args.shadow_only {
                                        final_chunk[local_idx] = false;
                                    } else {
                                        let (exact_score, exact_reject_row, visit, _) =
                                            run_exact_continuation(
                                                runtime,
                                                &model,
                                                feat,
                                                row_tau,
                                                &shadow_row,
                                                batch.nan_free,
                                                &mut ranks,
                                                &mut missing,
                                            )?;
                                        if exact_reject_row {
                                            final_reject_cnt += 1;
                                        }
                                        let visit_usize = visit.max(0) as usize;
                                        if visit_usize < exact_hist.len() {
                                            exact_hist[visit_usize] += 1;
                                        }
                                        exact_visit_sum += visit_usize as u64;
                                        final_chunk[local_idx] = exact_reject_row;
                                        score_chunk[local_idx] = exact_score;
                                        visit_chunk[local_idx] = visit;
                                    }
                                }
                            }

                            Ok((
                                active_cnt,
                                shadow_reject_cnt,
                                final_reject_cnt,
                                direct_reject_cnt,
                                direct_refer_cnt,
                                fallback_cnt,
                                total_block_evals,
                                resolved_early_trees,
                                total_prefix_trees,
                                exact_visit_sum,
                                exact_hist,
                            ))
                        },
                    )
                    .try_reduce(
                        || {
                            (
                                0usize,
                                0usize,
                                0usize,
                                0usize,
                                0usize,
                                0usize,
                                0u64,
                                0u64,
                                0u64,
                                0u64,
                                vec![0u64; hist_len],
                            )
                        },
                        |a, b| {
                            let mut hist = a.10;
                            for (dst, src) in hist.iter_mut().zip(b.10.iter()) {
                                *dst += *src;
                            }
                            Ok((
                                a.0 + b.0,
                                a.1 + b.1,
                                a.2 + b.2,
                                a.3 + b.3,
                                a.4 + b.4,
                                a.5 + b.5,
                                a.6 + b.6,
                                a.7 + b.7,
                                a.8 + b.8,
                                a.9 + b.9,
                                hist,
                            ))
                        },
                    )
            } else {
                let mut ranks = vec![0u8; base_n_features];
                let mut missing = vec![0u8; base_n_features];
                let mut hot_buf_by_route: Vec<Option<HotFeatureBuf>> = runtimes
                    .iter()
                    .map(|route| {
                        route.runtime
                            .compiled_hot_pack
                            .as_ref()
                            .map(|pack| HotFeatureBuf::new(pack.n_hot_features))
                    })
                    .collect();
                let mut hot_cache_by_route: Vec<Option<LazyHotFeatureCache>> = runtimes
                    .iter()
                    .map(|route| {
                        route.runtime.hot_pack.as_ref().map(|pack| {
                            LazyHotFeatureCache::new(pack.n_hot_features)
                        })
                    })
                    .collect();
                let mut active_cnt = 0usize;
                let mut shadow_reject_cnt = 0usize;
                let mut final_reject_cnt = 0usize;
                let mut direct_reject_cnt = 0usize;
                let mut direct_refer_cnt = 0usize;
                let mut fallback_cnt = 0usize;
                let mut total_block_evals = 0u64;
                let mut resolved_early_trees = 0u64;
                let mut total_prefix_trees = 0u64;
                let mut exact_visit_sum = 0u64;
                let mut exact_hist = vec![0u64; hist_len];

                for row_idx in 0..n {
                    let active = route_meta.active[row_idx] != 0;
                    if !active {
                        continue;
                    }

                    active_cnt += 1;
                    let feat = batch.row(row_idx);
                    let row_tau = route_meta.tau_used[row_idx];
                    let row_fold = route_meta.fold_id[row_idx];
                    let tau_bin = mlp_tau_bin(&tau_router_edges, row_tau)
                        .min(router_lookup.len().saturating_sub(1));
                    let route_idx = router_lookup[tau_bin];
                    let runtime = &runtimes[route_idx].runtime;
                    let shadow_row = if let Some(anchor_rescue) =
                        runtime.anchor_rescue.as_ref()
                    {
                        run_anchor_rescue_shadow_row(
                            runtime,
                            anchor_rescue,
                            feat,
                            row_tau,
                            row_fold,
                            &mut ranks,
                            &mut missing,
                            hot_cache_by_route[route_idx].as_mut(),
                            trace_enabled,
                        )?
                    } else if let Some(packet_runtime) =
                        runtime.packet_scheduler.as_ref()
                    {
                        run_packet_shadow_row(
                            runtime,
                            packet_runtime,
                            feat,
                            row_tau,
                            row_fold,
                            &mut ranks,
                            &mut missing,
                            trace_enabled,
                        )?
                    } else if !trace_enabled {
                        run_prefix_shadow_row_single_route_compiled(
                            runtime,
                            feat,
                            row_tau,
                            row_fold,
                            batch.nan_free,
                            &mut ranks,
                            &mut missing,
                            hot_buf_by_route[route_idx].as_mut(),
                            hot_cache_by_route[route_idx].as_mut(),
                        )?
                    } else {
                        run_prefix_shadow_row(
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
                            &mut ranks,
                            &mut missing,
                            hot_cache_by_route[route_idx].as_mut(),
                            trace_enabled,
                        )?
                    };

                    prefix_scores[row_idx] = shadow_row.prefix_score;
                    trees_used[row_idx] = shadow_row.trees_used.min(u16::MAX as usize) as u16;
                    block_evals[row_idx] = shadow_row.work_evals.min(u32::MAX as u64) as u32;
                    direct_checkpoints[row_idx] =
                        shadow_row.direct_checkpoint.min(u16::MAX as usize) as u16;
                    anchor_direct_checkpoints[row_idx] =
                        shadow_row.anchor_direct_checkpoint.min(u16::MAX as usize) as u16;
                    rescue_direct_checkpoints[row_idx] =
                        shadow_row.rescue_direct_checkpoint.min(u16::MAX as usize) as u16;
                    rescue_route_choice[row_idx] = shadow_row.rescue_route;
                    fallback_entry_checkpoints[row_idx] =
                        shadow_row.fallback_entry_checkpoint.min(u16::MAX as usize) as u16;
                    router_choice[row_idx] = route_idx.min(u16::MAX as usize) as u16;
                    router_tau_bin[row_idx] = tau_bin.min(u16::MAX as usize) as u16;
                    total_block_evals += shadow_row.work_evals;
                    resolved_early_trees += shadow_row.resolved_early_trees;
                    total_prefix_trees += shadow_row.trees_used as u64;
                    if trace_enabled {
                        let st = row_idx * cp_len;
                        let ed = st + cp_len;
                        checkpoint_scores_flat[st..ed]
                            .copy_from_slice(&shadow_row.checkpoint_scores);
                        checkpoint_work_evals_flat[st..ed]
                            .copy_from_slice(&shadow_row.checkpoint_work_evals);
                        checkpoint_deltas_flat[st..ed]
                            .copy_from_slice(&shadow_row.checkpoint_deltas);
                        checkpoint_resolved_flat[st..ed]
                            .copy_from_slice(&shadow_row.checkpoint_resolved_early);
                    }

                    if !shadow_row.fallback_used {
                        let is_reject = shadow_row.shadow_reject;
                        shadow_reject[row_idx] = is_reject;
                        final_reject[row_idx] = is_reject;
                        route_scores[row_idx] = shadow_row.shadow_route_score;
                        if is_reject {
                            shadow_reject_cnt += 1;
                            final_reject_cnt += 1;
                            direct_reject_cnt += 1;
                        } else {
                            direct_refer_cnt += 1;
                        }
                    } else {
                        fallback_cnt += 1;
                        fallback_used[row_idx] = 1;
                        shadow_reject[row_idx] = false;
                        route_scores[row_idx] = shadow_row.prefix_score;
                        if args.shadow_only {
                            final_reject[row_idx] = false;
                        } else {
                            let (exact_score, exact_reject_row, visit, _) =
                                run_exact_continuation(
                                    runtime,
                                    &model,
                                    feat,
                                    row_tau,
                                    &shadow_row,
                                    batch.nan_free,
                                    &mut ranks,
                                    &mut missing,
                                )?;
                            if exact_reject_row {
                                final_reject_cnt += 1;
                            }
                            let visit_usize = visit.max(0) as usize;
                            if visit_usize < exact_hist.len() {
                                exact_hist[visit_usize] += 1;
                            }
                            exact_visit_sum += visit_usize as u64;
                            final_reject[row_idx] = exact_reject_row;
                            route_scores[row_idx] = exact_score;
                            exact_visits[row_idx] = visit;
                        }
                    }
                }

                Ok((
                    active_cnt,
                    shadow_reject_cnt,
                    final_reject_cnt,
                    direct_reject_cnt,
                    direct_refer_cnt,
                    fallback_cnt,
                    total_block_evals,
                    resolved_early_trees,
                    total_prefix_trees,
                    exact_visit_sum,
                    exact_hist,
                ))
            }
        };
    let (
        active_cnt,
        shadow_reject_cnt,
        final_reject_cnt,
        direct_reject_cnt,
        direct_refer_cnt,
        fallback_cnt,
        total_block_evals,
        resolved_early_trees,
        total_prefix_trees,
        exact_visit_sum,
        exact_visit_hist,
    ) = if use_single_route_hot {
        run_qs_l2_prefix_cal_single_route_hot(
            &runtimes[0].runtime,
            &model,
            &batch,
            &route_meta,
            &args,
            n,
            hist_len,
            parallel_enabled,
            thread_pool.as_ref(),
            &mut prefix_scores,
            &mut trees_used,
            &mut shadow_reject,
            &mut final_reject,
            &mut fallback_used,
            &mut direct_checkpoints,
            &mut fallback_entry_checkpoints,
            &mut route_scores,
            &mut block_evals,
            &mut exact_visits,
        )?
    } else {
        install_in_pool(thread_pool.as_ref(), work)?
    };
    let elapsed = t0.elapsed().as_secs_f64();

    let mut checkpoint_exit_counts = vec![0u64; cp_len];
    let mut direct_rows_by_checkpoint = vec![0u64; cp_len];
    let mut shadow_mismatch_by_checkpoint = vec![0u64; cp_len];
    let mut fallback_entry_checkpoint_counts = vec![0u64; cp_len];
    let mut router_choice_counts = vec![0u64; runtimes.len().max(1)];
    let mut router_tau_bin_counts = vec![0u64; router_lookup.len().max(1)];
    let mut anchor_exit_counts = vec![0u64; cp_len];
    let mut rescue_route_counts = vec![0u64; 4];
    let mut rescue_exit_counts = vec![0u64; 2];
    let mut tail_continuation_rows = 0u64;
    let mut shadow_mismatch_by_stage = vec![0u64; 2];
    let mut checkpoint_to_idx = HashMap::new();
    for (idx, checkpoint) in checkpoints.iter().copied().enumerate() {
        checkpoint_to_idx.insert(checkpoint as u16, idx);
    }
    for row_idx in 0..n {
        if route_meta.active[row_idx] != 0 {
            let route_idx = router_choice[row_idx] as usize;
            if route_idx < router_choice_counts.len() {
                router_choice_counts[route_idx] += 1;
            }
            let tau_bin = router_tau_bin[row_idx] as usize;
            if tau_bin < router_tau_bin_counts.len() {
                router_tau_bin_counts[tau_bin] += 1;
            }
        }
        let exact_reject = route_meta.exact_positive[row_idx] != 0;
        if has_anchor_rescue {
            let anchor_cp = anchor_direct_checkpoints[row_idx];
            if anchor_cp != 0 {
                if let Some(idx) = checkpoint_to_idx.get(&anchor_cp) {
                    checkpoint_exit_counts[*idx] += 1;
                    direct_rows_by_checkpoint[*idx] += 1;
                    anchor_exit_counts[*idx] += 1;
                    if shadow_reject[row_idx] != exact_reject {
                        shadow_mismatch_by_checkpoint[*idx] += 1;
                        shadow_mismatch_by_stage[0] += 1;
                    }
                }
            } else {
                let rescue_route = rescue_route_choice[row_idx] as usize;
                if rescue_route < rescue_route_counts.len() {
                    rescue_route_counts[rescue_route] += 1;
                }
                let rescue_cp = rescue_direct_checkpoints[row_idx];
                if rescue_cp != 0 {
                    if rescue_route_choice[row_idx] == 2 {
                        rescue_exit_counts[0] += 1;
                    } else if rescue_route_choice[row_idx] == 3 {
                        rescue_exit_counts[1] += 1;
                    }
                    if shadow_reject[row_idx] != exact_reject {
                        shadow_mismatch_by_stage[1] += 1;
                    }
                } else if fallback_used[row_idx] != 0 {
                    tail_continuation_rows += 1;
                }
                let fallback_cp = fallback_entry_checkpoints[row_idx];
                if let Some(idx) = checkpoint_to_idx.get(&fallback_cp) {
                    checkpoint_exit_counts[*idx] += 1;
                    fallback_entry_checkpoint_counts[*idx] += 1;
                }
            }
        } else {
            let direct_cp = direct_checkpoints[row_idx];
            if direct_cp != 0 {
                if let Some(idx) = checkpoint_to_idx.get(&direct_cp) {
                    checkpoint_exit_counts[*idx] += 1;
                    direct_rows_by_checkpoint[*idx] += 1;
                    if shadow_reject[row_idx] != exact_reject {
                        shadow_mismatch_by_checkpoint[*idx] += 1;
                    }
                }
            } else {
                let fallback_cp = fallback_entry_checkpoints[row_idx];
                if let Some(idx) = checkpoint_to_idx.get(&fallback_cp) {
                    checkpoint_exit_counts[*idx] += 1;
                    fallback_entry_checkpoint_counts[*idx] += 1;
                }
            }
        }
    }

    if let Some(path) = &args.out_tsv {
        write_qs_prefix_output_tsv(
            path,
            &batch,
            n,
            &prefix_scores,
            &trees_used,
            &shadow_reject,
            &final_reject,
            &fallback_used,
            &route_scores,
            &block_evals,
            &exact_visits,
            &direct_checkpoints,
            &fallback_entry_checkpoints,
            &router_choice,
            &router_tau_bin,
        )?;
    }
    if let Some(path) = &args.trace_jsonl {
        write_qs_prefix_trace_jsonl(
            path,
            &batch,
            &route_meta,
            n,
            &checkpoints,
            &prefix_scores,
            &trees_used,
            &final_reject,
            &fallback_used,
            &router_choice,
            &router_tau_bin,
            &checkpoint_scores_flat,
            &checkpoint_work_evals_flat,
            &checkpoint_deltas_flat,
            &checkpoint_resolved_flat,
        )?;
    }

    let active_f = active_cnt.max(1) as f64;
    let fallback_n = fallback_cnt.max(1);
    let direct_decision_cnt = direct_reject_cnt + direct_refer_cnt;
    let stats = QsPrefixCalStats {
        variant_key: first_runtime.calibration.variant_key.clone(),
        ref_q: first_runtime.calibration.ref_q,
        rej_q: first_runtime.calibration.rej_q,
        checkpoints: checkpoints.clone(),
        n_rows: n,
        n_features: base_n_features,
        n_trees: base_n_trees,
        n_blocks: first_runtime.pack.n_blocks(),
        threads: par_threads,
        parallel_enabled,
        feature_format: batch.format_tag.clone(),
        nan_free: batch.nan_free,
        shadow_only: args.shadow_only,
        v4_mode,
        direct_kernel: prefix_direct_kernel_str(first_runtime.direct_kernel).to_string(),
        certifier_kind: prefix_certifier_kind_str(first_runtime.certifier_kind).to_string(),
        hot_exact_prefix_limit: first_runtime.hot_exact_prefix_limit,
        telemetry_schema_version,
        elapsed_sec: elapsed,
        rows_per_sec: n as f64 / elapsed.max(1e-9),
        rss_peak_mb: peak_rss_mb(),
        route_active_rate: active_cnt as f64 / n.max(1) as f64,
        shadow_reject_rate: shadow_reject_cnt as f64 / active_f,
        final_reject_rate: final_reject_cnt as f64 / active_f,
        direct_reject_rate: direct_reject_cnt as f64 / active_f,
        direct_refer_rate: direct_refer_cnt as f64 / active_f,
        direct_decision_rate: direct_decision_cnt as f64 / active_f,
        fallback_rate: fallback_cnt as f64 / active_f,
        avg_blocks_per_row: total_block_evals as f64 / active_f,
        avg_blocks_per_tree: total_block_evals as f64
            / ((active_cnt.max(1) * base_n_trees).max(1) as f64),
        avg_prefix_trees_used: total_prefix_trees as f64 / active_f,
        resolved_early_rate: resolved_early_trees as f64 / total_prefix_trees.max(1) as f64,
        exact_avg_visited_trees: exact_visit_sum as f64 / fallback_n as f64,
        exact_p99_visited_trees: pct_from_hist(&exact_visit_hist, 0.99, fallback_cnt),
        checkpoint_exit_counts,
        direct_rows_by_checkpoint,
        shadow_mismatch_by_checkpoint,
        fallback_entry_checkpoint_counts,
        exact_visit_hist,
        router_choice_counts,
        router_tau_bin_counts,
        anchor_exit_counts,
        rescue_route_counts,
        rescue_exit_counts,
        tail_continuation_rows,
        shadow_mismatch_by_stage,
    };

    if let Some(path) = &args.stats_json {
        let txt = serde_json::to_string_pretty(&stats)?;
        fs::write(path, txt).with_context(|| format!("write {}", path.display()))?;
    }

    println!(
        "mode=l2-qs-prefix-cal variant={} rows={} threads={} parallel={} rows/s={:.1} direct_rate={:.4} fallback_rate={:.4} avg_prefix_trees={:.1}",
        stats.variant_key,
        stats.n_rows,
        stats.threads,
        stats.parallel_enabled,
        stats.rows_per_sec,
        stats.direct_decision_rate,
        stats.fallback_rate,
        stats.avg_prefix_trees_used,
    );

    Ok(())
}

fn write_output_tsv(
    path: &PathBuf,
    batch: &FeatureBatch,
    n: usize,
    mode: InferMode,
    scores: &[f32],
    passes: &[bool],
    visits: &[i32],
) -> Result<()> {
    let mut w = BufWriter::new(
        fs::File::create(path).with_context(|| format!("create {}", path.display()))?,
    );
    let score_col = score_col_name(mode);
    let decision_col = decision_col_name(mode);
    writeln!(
        w,
        "TransactionID\tisFraud\t{}\t{}\tvisited_trees",
        score_col, decision_col
    )?;
    let pos_label = positive_label(mode);
    let neg_label = negative_label(mode);
    for i in 0..n {
        let decision = if passes[i] { pos_label } else { neg_label };
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

fn run_infer(args: InferArgs) -> Result<()> {
    let model = load_soa(&args.soa)?;
    let bounds = load_bounds(&args.bounds)?;
    let batch = load_features(&args.feat_bin)?;
    let rank_pack = if let Some(path) = &args.rank_pack {
        Some(load_rank_pack(path)?)
    } else {
        None
    };
    let raw_tree_order = if let Some(path) = &args.tree_order {
        Some(load_tree_order(path)?)
    } else {
        None
    };
    let prefix16_pack = if let Some(path) = &args.prefix16_pack {
        Some(load_prefix_pack(path)?)
    } else {
        None
    };
    let prefix32_pack = if let Some(path) = &args.prefix32_pack {
        Some(load_prefix_pack(path)?)
    } else {
        None
    };
    let approx_policy = if let Some(path) = &args.approx_policy {
        Some(load_approx_policy(path)?)
    } else {
        None
    };
    let route_meta = if let Some(path) = &args.route_meta {
        Some(load_route_meta(path)?)
    } else {
        None
    };
    let threshold = if matches!(args.mode, InferMode::L2RouteExactReordered | InferMode::L2RouteApprox) {
        args.threshold_override.unwrap_or(0.0)
    } else {
        load_threshold(&args.policy, args.threshold_override)?
    };
    let base_score = if let Some(p) = &args.model_json {
        parse_base_score(p)?
    } else {
        0.0
    };

    if model.n_features != batch.n_cols {
        bail!(
            "feature count mismatch: model={} input={}",
            model.n_features,
            batch.n_cols
        );
    }
    if model.n_trees != bounds.n_trees {
        bail!(
            "tree count mismatch: model={} bounds={}",
            model.n_trees,
            bounds.n_trees
        );
    }
    if let Some(pack) = &rank_pack {
        if pack.n_features != model.n_features {
            bail!(
                "rank pack feature mismatch: rank_pack={} model={}",
                pack.n_features,
                model.n_features
            );
        }
        if pack.node_thr_rank.len() != model.node_count {
            bail!(
                "rank pack node mismatch: rank_pack={} model={}",
                pack.node_thr_rank.len(),
                model.node_count
            );
        }
    }
    if let Some(meta) = &route_meta {
        if meta.n_rows != batch.n_rows {
            bail!(
                "route meta row mismatch: route_meta={} feat_bin={}",
                meta.n_rows,
                batch.n_rows
            );
        }
        if meta.ids.len() != batch.n_rows
            || meta.fold_id.len() != batch.n_rows
            || meta.tau_used.len() != batch.n_rows
            || meta.active.len() != batch.n_rows
            || meta.exact_positive.len() != batch.n_rows
        {
            bail!("route meta internal length mismatch");
        }
        for i in 0..batch.n_rows {
            if batch.id(i) != meta.ids[i] {
                bail!(
                    "route meta TransactionID mismatch at row {}: feat_bin={} route_meta={}",
                    i,
                    batch.id(i),
                    meta.ids[i]
                );
            }
        }
    }
    if matches!(
        args.mode,
        InferMode::RouteExactReordered
            | InferMode::MarginExactReordered
            | InferMode::RouteApprox
            | InferMode::L2RouteExactReordered
            | InferMode::L2RouteApprox
    ) && raw_tree_order.is_none()
    {
        bail!("mode {:?} requires --tree-order", args.mode);
    }
    if matches!(args.mode, InferMode::RouteApprox | InferMode::L2RouteApprox)
        && approx_policy.is_none()
    {
        bail!("mode {:?} requires --approx-policy", args.mode);
    }
    if args.fast_no_stats && args.out_tsv.is_some() {
        bail!("--fast-no-stats cannot be combined with --out-tsv");
    }
    if matches!(args.mode, InferMode::L2RouteExactReordered | InferMode::L2RouteApprox)
        && args.route_meta.is_none()
    {
        bail!("mode {:?} requires --route-meta", args.mode);
    }
    if let Some(policy) = &approx_policy {
        if !matches!(args.mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
            bail!("--approx-policy is only valid with --mode route-approx");
        }
        if policy.fallback_mode != "route-exact-reordered" {
            bail!("unsupported approx fallback_mode: {}", policy.fallback_mode);
        }
        let _ = &policy.calibration_manifest;
    }

    let effective_rank_pack = if matches!(args.mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
        if let Some(policy) = &approx_policy {
            match policy.rank_mode {
                RankMode::RankPack => {
                    if rank_pack.is_none() {
                        bail!("route-approx rank-pack policy requires --rank-pack");
                    }
                    rank_pack.as_ref()
                }
                RankMode::Float => None,
            }
        } else {
            None
        }
    } else {
        rank_pack.as_ref()
    };

    let plan = materialize_tree_plan(&bounds, raw_tree_order.as_ref(), args.max_trees)?;
    if let Some(policy) = &approx_policy {
        let mx = policy
            .k_hot
            .max(policy.used_checkpoints.iter().copied().max().unwrap_or(0));
        if mx > plan.n_trees {
            bail!(
                "approx checkpoint {} exceeds effective tree count {}",
                mx,
                plan.n_trees
            );
        }
    }

    let n = args.max_rows.unwrap_or(batch.n_rows).min(batch.n_rows);
    let thread_pool = build_thread_pool(args.threads)?;
    let prefix_pack_kind = if matches!(args.mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
        if prefix16_pack.is_some()
            && approx_policy
                .as_ref()
                .map(|p| {
                    p.used_checkpoints.first().copied().unwrap_or(usize::MAX) <= 16
                        && matches!(p.rank_mode, RankMode::Float)
                })
                .unwrap_or(false)
        {
            "prefix16"
        } else if prefix32_pack.is_some()
            && approx_policy
                .as_ref()
                .map(|p| {
                    p.used_checkpoints.first().copied().unwrap_or(usize::MAX) <= 32
                        && matches!(p.rank_mode, RankMode::Float)
                })
                .unwrap_or(false)
        {
            "prefix32"
        } else {
            ""
        }
    } else {
        ""
    };

    let stats = if args.fast_no_stats && matches!(args.mode, InferMode::RouteApprox | InferMode::L2RouteApprox) {
        let policy = approx_policy.as_ref().unwrap();
        let t0 = Instant::now();
        let (agg, parallel_enabled, used_threads) = run_kernel_fast_approx(
            &model,
            &plan,
            &batch,
            route_meta.as_ref(),
            policy,
            prefix16_pack.as_ref(),
            prefix32_pack.as_ref(),
            n,
            threshold,
            base_score,
            args.eps,
            args.bound_guard,
            args.bound_check_every.max(1),
            args.threads,
            args.chunk_rows.max(1),
            args.parallel_min_rows.max(1),
            thread_pool.as_ref(),
        )?;
        let elapsed = t0.elapsed().as_secs_f64();
        let approx_cov = (agg.approx_pass_cnt + agg.approx_ref_cnt) as f64 / n as f64;
        InferStats {
            n_rows: n,
            n_trees: plan.n_trees,
            threshold,
            base_score,
            mode: args.mode,
            threads: used_threads,
            parallel_enabled,
            rank_pack_enabled: effective_rank_pack.is_some(),
            tree_order_enabled: raw_tree_order.is_some(),
            approx_policy_enabled: approx_policy.is_some(),
            feature_format: batch.format_tag.clone(),
            nan_free: batch.nan_free,
            fast_no_stats: true,
            bound_check_every: args.bound_check_every.max(1),
            max_trees: plan.n_trees,
            eps: args.eps,
            bound_guard: args.bound_guard,
            route_pass_rate: agg.pass_cnt as f64 / n as f64,
            route_active_rate: agg.active_cnt as f64 / n as f64,
            avg_visited_trees: agg.visit_sum as f64 / n as f64,
            p50_visited_trees: pct_from_hist(&agg.visit_hist, 0.50, n),
            p90_visited_trees: pct_from_hist(&agg.visit_hist, 0.90, n),
            p99_visited_trees: pct_from_hist(&agg.visit_hist, 0.99, n),
            visit_hist: agg.visit_hist.clone(),
            full_eval_rate: agg.full_cnt as f64 / n as f64,
            approx_coverage_rate: approx_cov,
            approx_pass_rate: agg.approx_pass_cnt as f64 / n as f64,
            approx_refer_rate: agg.approx_ref_cnt as f64 / n as f64,
            approx_fallback_rate: 1.0 - approx_cov,
            approx_k_hot: policy.k_hot,
            approx_rank_mode: match policy.rank_mode {
                RankMode::RankPack => "rank-pack",
                RankMode::Float => "float",
            }
            .to_string(),
            approx_checkpoint_exit_counts: agg.checkpoint_counts,
            prefix_pack_kind: prefix_pack_kind.to_string(),
            elapsed_sec: elapsed,
            rows_per_sec: n as f64 / elapsed.max(1e-9),
            rss_peak_mb: peak_rss_mb(),
        }
    } else {
        let t0 = Instant::now();
        let (
            scores,
            passes,
            visits,
            parallel_enabled,
            used_threads,
            approx_pass_cnt,
            approx_ref_cnt,
            checkpoint_counts,
        ) = run_kernel(
            &model,
            &plan,
            &batch,
            route_meta.as_ref(),
            effective_rank_pack,
            approx_policy.as_ref(),
            prefix16_pack.as_ref(),
            prefix32_pack.as_ref(),
            n,
            threshold,
            base_score,
            args.mode,
            args.eps,
            args.bound_guard,
            args.bound_check_every.max(1),
            args.threads,
            args.chunk_rows.max(1),
            args.parallel_min_rows.max(1),
            thread_pool.as_ref(),
        )?;
        let elapsed = t0.elapsed().as_secs_f64();

        if let Some(path) = &args.out_tsv {
            write_output_tsv(path, &batch, n, args.mode, &scores, &passes, &visits)?;
        }

        let pass_cnt = passes.iter().filter(|&&x| x).count();
        let full_cnt = visits
            .iter()
            .filter(|&&v| v as usize == plan.n_trees)
            .count();
        let avg_visit = visits.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let approx_cov = (approx_pass_cnt + approx_ref_cnt) as f64 / n as f64;
        InferStats {
            n_rows: n,
            n_trees: plan.n_trees,
            threshold,
            base_score,
            mode: args.mode,
            threads: used_threads,
            parallel_enabled,
            rank_pack_enabled: effective_rank_pack.is_some(),
            tree_order_enabled: raw_tree_order.is_some(),
            approx_policy_enabled: approx_policy.is_some(),
            feature_format: batch.format_tag.clone(),
            nan_free: batch.nan_free,
            fast_no_stats: false,
            bound_check_every: args.bound_check_every.max(1),
            max_trees: plan.n_trees,
            eps: args.eps,
            bound_guard: args.bound_guard,
            route_pass_rate: pass_cnt as f64 / n as f64,
            route_active_rate: route_meta
                .as_ref()
                .map(|m| m.active[..n].iter().filter(|&&x| x != 0).count() as f64 / n as f64)
                .unwrap_or(1.0),
            avg_visited_trees: avg_visit,
            p50_visited_trees: pct(&visits, 0.50),
            p90_visited_trees: pct(&visits, 0.90),
            p99_visited_trees: pct(&visits, 0.99),
            visit_hist: histogram_from_visits(&visits, plan.n_trees),
            full_eval_rate: full_cnt as f64 / n as f64,
            approx_coverage_rate: approx_cov,
            approx_pass_rate: approx_pass_cnt as f64 / n as f64,
            approx_refer_rate: approx_ref_cnt as f64 / n as f64,
            approx_fallback_rate: 1.0 - approx_cov,
            approx_k_hot: approx_policy.as_ref().map(|p| p.k_hot).unwrap_or(0),
            approx_rank_mode: approx_policy
                .as_ref()
                .map(|p| match p.rank_mode {
                    RankMode::RankPack => "rank-pack",
                    RankMode::Float => "float",
                })
                .unwrap_or("")
                .to_string(),
            approx_checkpoint_exit_counts: checkpoint_counts,
            prefix_pack_kind: prefix_pack_kind.to_string(),
            elapsed_sec: elapsed,
            rows_per_sec: n as f64 / elapsed.max(1e-9),
            rss_peak_mb: peak_rss_mb(),
        }
    };

    if let Some(path) = &args.stats_json {
        let txt = serde_json::to_string_pretty(&stats)?;
        fs::write(path, txt).with_context(|| format!("write {}", path.display()))?;
    }

    println!(
        "mode={:?} rows={} threads={} parallel={} rows/s={:.1} avg_visited={:.1} full_eval_rate={:.6} format={} rank_pack={} fast_no_stats={} prefix_pack={}",
        stats.mode,
        stats.n_rows,
        stats.threads,
        stats.parallel_enabled,
        stats.rows_per_sec,
        stats.avg_visited_trees,
        stats.full_eval_rate,
        stats.feature_format,
        stats.rank_pack_enabled,
        stats.fast_no_stats,
        stats.prefix_pack_kind
    );

    Ok(())
}

pub fn run_cli() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Infer(args) => run_infer(args),
        Command::DispatchL2Exact(args) => run_dispatch_l2_exact(args),
        Command::QsL2Exact(args) => run_qs_l2_exact(args),
        Command::QsL2Fast(args) => run_qs_l2_fast(args),
        Command::QsL2PrefixCal(args) => run_qs_l2_prefix_cal(args),
        Command::CompileHotPrefix(args) => run_compile_hot_prefix(args),
        Command::RepackL2PrefixV2(args) => run_repack_l2_prefix_v2(args),
        Command::RepackL2RuntimeV2(args) => run_repack_l2_runtime_v2(args),
        Command::RepackL2RuntimeV3(args) => run_repack_l2_runtime_v3(args),
        Command::RepackL2ExpV1(args) => run_repack_l2_exp_v1(args),
        Command::RepackL2Kernel7945hxV1(args) => run_repack_l2_kernel_7945hx_v1(args),
    }
}
