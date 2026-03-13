const KERNEL_ZEN4_V1_DIR: &str = "l2_kernel_zen4_v1";

pub(crate) fn repack_from_bundle_manifest(
    bundle_manifest: &std::path::PathBuf,
    output_dir: Option<&std::path::PathBuf>,
) -> anyhow::Result<std::path::PathBuf> {
    let out_dir = output_dir.cloned().or_else(|| {
        let prefix_args = crate::QsL2PrefixCalArgs {
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
            direct_kernel: None,
            certifier_kind: None,
            certifier_json: None,
            threads: 0,
            chunk_rows: 128,
            parallel_min_rows: 4096,
            max_rows: None,
            out_tsv: None,
            stats_json: None,
            trace_jsonl: None,
            shadow_only: false,
        };
        let resolved = crate::resolve_prefix_cal_bundle(&prefix_args).ok()?;
        let bundle_root = bundle_manifest
            .parent()
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())
            .and_then(|p| p.parent())?
            .to_path_buf();
        let resolved = crate::absolutize_resolved_bundle(&bundle_root, resolved);
        resolved
            .qs_pack
            .parent()
            .map(|p| p.join(KERNEL_ZEN4_V1_DIR))
    });
    crate::l2_exp_v1::repack_from_bundle_manifest(bundle_manifest, out_dir.as_ref())
}

pub(crate) fn load_from_resolved_bundle(
    resolved: &crate::ResolvedPrefixCalBundle,
) -> anyhow::Result<Option<crate::l2_exp_v1::L2ExpV1Runtime>> {
    let exp_dir = resolved
        .qs_pack
        .parent()
        .map(|p| p.join(KERNEL_ZEN4_V1_DIR))
        .unwrap_or_else(|| std::path::PathBuf::from(KERNEL_ZEN4_V1_DIR));
    crate::l2_exp_v1::load_from_manifest_dir(&exp_dir)
}

pub(crate) fn predict_nomiss(
    runtime: &crate::l2_exp_v1::L2ExpV1Runtime,
    base_runtime: &crate::LoadedPrefixRuntime,
    feat: &[f32],
    row_tau: f32,
    row_fold: i32,
    hot_buf: &mut crate::l2_exp_v1::ExpHotStageBuf,
    local_ranks: &mut Vec<u8>,
) -> anyhow::Result<crate::OnlineL2Output> {
    crate::l2_exp_v1::predict_nomiss(
        runtime,
        base_runtime,
        feat,
        row_tau,
        row_fold,
        hot_buf,
        local_ranks,
    )
}
