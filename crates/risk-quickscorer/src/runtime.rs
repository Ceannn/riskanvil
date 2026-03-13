use std::cell::RefCell;
use std::path::Path;
use std::sync::OnceLock;
use tracing::{info, trace};
use crate::exp_l2_zen4::ExperimentalL2KernelZen4;
use crate::l2_exp_v1::{ExpHotStageBuf, L2ExpV1Runtime};

thread_local! {
    static ONLINE_TLS_SCRATCH: RefCell<OnlineTlsScratch> = RefCell::new(OnlineTlsScratch::default());
}

fn l2_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("QS_TRACE_L2_STAGES").is_some())
}

#[derive(Default)]
struct OnlineTlsScratch {
    l1_feat: Vec<f32>,
    l2_ranks: Vec<u8>,
    l2_missing: Vec<u8>,
    l2_hot_buf: Option<HotFeatureBuf>,
    l2_hot_cache: Option<LazyHotFeatureCache>,
    l2_exp_hot_buf: Option<ExpHotStageBuf>,
    l2_exp_local_ranks: Vec<u8>,
}

impl OnlineTlsScratch {
    fn decode_l1_bytes<'a>(&'a mut self, bytes: &[u8], dim: usize) -> Result<&'a [f32]> {
        decode_dense_f32le_into(bytes, dim, &mut self.l1_feat)?;
        Ok(self.l1_feat.as_slice())
    }

    fn ensure_l2<'a>(
        &'a mut self,
        runtime: &LoadedPrefixRuntime,
    ) -> (&'a mut [u8], &'a mut [u8], Option<&'a mut HotFeatureBuf>, Option<&'a mut LazyHotFeatureCache>) {
        let n_features = runtime.pack.n_features();
        if self.l2_ranks.len() != n_features {
            self.l2_ranks.resize(n_features, 0);
        } else {
            self.l2_ranks.fill(0);
        }
        if self.l2_missing.len() != n_features {
            self.l2_missing.resize(n_features, 0);
        } else {
            self.l2_missing.fill(0);
        }

        match runtime.compiled_hot_pack.as_ref() {
            Some(pack) => {
                let need = pack.n_hot_features;
                match self.l2_hot_buf.as_mut() {
                    Some(buf) if buf.values.len() == need => {}
                    _ => self.l2_hot_buf = Some(HotFeatureBuf::new(need)),
                }
            }
            None => self.l2_hot_buf = None,
        }

        match runtime.hot_pack.as_ref() {
            Some(pack) => {
                let need = pack.n_hot_features;
                let recreate = match self.l2_hot_cache.as_ref() {
                    Some(cache) => cache.values.len() != need,
                    None => true,
                };
                if recreate {
                    self.l2_hot_cache = Some(LazyHotFeatureCache::new(need));
                } else if let Some(cache) = self.l2_hot_cache.as_mut() {
                    cache.next_row();
                }
            }
            None => self.l2_hot_cache = None,
        }

        (
            self.l2_ranks.as_mut_slice(),
            self.l2_missing.as_mut_slice(),
            self.l2_hot_buf.as_mut(),
            self.l2_hot_cache.as_mut(),
        )
    }

    fn ensure_l2_nomiss_fast<'a>(
        &'a mut self,
        runtime: &LoadedPrefixRuntime,
    ) -> Option<(&'a mut HotFeatureBuf, &'a mut [u8])> {
        let pack = runtime.compiled_hot_pack.as_ref()?;
        let need = pack.n_hot_features;
        match self.l2_hot_buf.as_mut() {
            Some(buf) if buf.values.len() == need => {}
            _ => self.l2_hot_buf = Some(HotFeatureBuf::new(need)),
        }
        let n_features = runtime.pack.n_features();
        if self.l2_ranks.len() != n_features {
            self.l2_ranks.resize(n_features, 0);
        }
        Some((self.l2_hot_buf.as_mut()?, self.l2_ranks.as_mut_slice()))
    }

    fn ensure_l2_exp<'a>(
        &'a mut self,
        exp: &L2ExpV1Runtime,
    ) -> (&'a mut ExpHotStageBuf, &'a mut Vec<u8>) {
        let need = exp.hot_feature_count();
        match self.l2_exp_hot_buf.as_mut() {
            Some(buf) => buf.ensure_len(need),
            None => self.l2_exp_hot_buf = Some(ExpHotStageBuf::new(need)),
        }
        self.l2_exp_local_ranks.fill(0);
        (self.l2_exp_hot_buf.as_mut().unwrap(), &mut self.l2_exp_local_ranks)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OnlineL1Output {
    pub score: f32,
    pub passed: bool,
    pub visited_trees: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct OnlineL2Output {
    pub score: f32,
    pub reject: bool,
    pub used_fallback: bool,
    pub trees_used: i32,
}

#[derive(Debug)]
struct OnlineL1Runtime {
    model: SoaModel,
    plan: TreeOrder,
    approx_policy: ApproxPolicy,
    hot_compact_pack: Option<HotApproxL1CompactPack>,
    threshold: f32,
    base_score: f32,
}

#[derive(Debug)]
struct OnlineL2Runtime {
    model: SoaModel,
    runtime: LoadedPrefixRuntime,
    experimental_zen4: Option<ExperimentalL2KernelZen4>,
    exp_v1: Option<L2ExpV1Runtime>,
    kernel_zen4_v1: Option<L2ExpV1Runtime>,
}

#[derive(Debug)]
pub struct MinpackRuntime {
    l1: OnlineL1Runtime,
    l2: OnlineL2Runtime,
    l1_dim: usize,
    l2_dim: usize,
    bundle_dir: PathBuf,
}

impl MinpackRuntime {
    pub fn load(bundle_dir: &std::path::Path) -> Result<Self> {
        let bundle_dir = bundle_dir.to_path_buf();
        log_load_mem("runtime_load_begin");

        let l1_model_dir =
            require_path(&bundle_dir, "quickscorer_l1l2_single_l512_cascade_20260304/models/quickscorer_l1")?;
        let l2_model_dir =
            require_path(&bundle_dir, "quickscorer_l1l2_single_l512_cascade_20260304/models/quickscorer_l2")?;

        let l1_soa = load_soa(&require_path(
            &bundle_dir,
            "runs/L1_EXECUTOR_BENCH/l1_model_soa.bin",
        )?)?;
        let l1_bounds = load_bounds(&require_path(
            &bundle_dir,
            "runs/L1_EXECUTOR_BENCH/l1_bounds.bin",
        )?)?;
        let l1_order = load_tree_order(&require_path(
            &bundle_dir,
            "runs/L1_RUST_V4/tree_order.bin",
        )?)?;
        let l1_plan = materialize_tree_plan(&l1_bounds, Some(&l1_order), None)?;
        let l1_approx_policy = load_approx_policy(&require_path(
            &bundle_dir,
            "runs/L1_RUST_V5/approx_policy_v3/approx_policy.json",
        )?)?;
        let l1_hot_compact_pack =
            compile_hot_approx_l1_compact_pack(&l1_soa, &l1_plan, &l1_approx_policy)?;
        let l1_threshold = load_threshold(&l1_model_dir.join("policy.json"), None)?;
        let l1_base_score = parse_base_score_from_any_json(&require_path(
            &bundle_dir,
            "runs/L1_RUST_V5/approx_policy_v3/approx_policy.json",
        )?)?;
        let l1_dim = load_feature_names_from_json(&l1_model_dir.join("feature_names.json"))?.len();
        log_load_mem("runtime_load_after_l1");

        let manifest_path = require_path(
            &bundle_dir,
            "runs/L2_Q2_4500_PREFIX_CAL_V5_ATLAS_20260307/atlas/manifest.json",
        )?;
        let l2_args = QsL2PrefixCalArgs {
            bundle_manifest: Some(manifest_path.clone()),
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
        let resolved = resolve_prefix_cal_bundle(&l2_args)?;
        let resolved = absolutize_resolved_bundle(&bundle_dir, resolved);
        let l2_soa = load_soa(&resolved.soa)?;
        let l2_bounds = load_bounds(&resolved.bounds)?;
        let mut l2_runtime = load_prefix_runtime(&resolved, &l2_soa, &l2_bounds)?;
        let l2_exp_v1 = l2_exp_v1::load_from_resolved_bundle(&resolved)?;
        let l2_kernel_zen4_v1 = l2_kernel_zen4::load_from_resolved_bundle(&resolved)?;
        l2_runtime.trim_for_online();
        let l2_dim = load_feature_names_from_json(&l2_model_dir.join("feature_names.json"))?.len();
        log_load_mem("runtime_load_after_l2");

        let out = Self {
            l1: OnlineL1Runtime {
                model: l1_soa,
                plan: l1_plan,
                approx_policy: l1_approx_policy,
                hot_compact_pack: l1_hot_compact_pack,
                threshold: l1_threshold,
                base_score: l1_base_score,
            },
            l2: OnlineL2Runtime {
                model: l2_soa,
                experimental_zen4: ExperimentalL2KernelZen4::new(&l2_runtime),
                exp_v1: l2_exp_v1,
                kernel_zen4_v1: l2_kernel_zen4_v1,
                runtime: l2_runtime,
            },
            l1_dim,
            l2_dim,
            bundle_dir,
        };
        log_load_mem("runtime_load_done");
        Ok(out)
    }

    #[inline]
    pub fn backend_name(&self) -> &'static str {
        "rust_quickscorer_zen4"
    }

    #[inline]
    pub fn bundle_dir(&self) -> &std::path::Path {
        &self.bundle_dir
    }

    #[inline]
    pub fn l1_dim(&self) -> usize {
        self.l1_dim
    }

    #[inline]
    pub fn l2_dim(&self) -> usize {
        self.l2_dim
    }

    #[inline]
    pub fn l1_threshold(&self) -> f32 {
        self.l1.threshold
    }

    pub fn predict_l1_bytes(&self, bytes: &[u8]) -> Result<OnlineL1Output> {
        ONLINE_TLS_SCRATCH.with(|cell| -> Result<OnlineL1Output> {
            let mut scratch = cell.borrow_mut();
            let feat = scratch.decode_l1_bytes(bytes, self.l1_dim)?;
            self.predict_l1_feat(feat)
        })
    }

    pub fn predict_l1_bytes_nomiss(&self, bytes: &[u8]) -> Result<OnlineL1Output> {
        ONLINE_TLS_SCRATCH.with(|cell| -> Result<OnlineL1Output> {
            let mut scratch = cell.borrow_mut();
            let feat = scratch.decode_l1_bytes(bytes, self.l1_dim)?;
            self.predict_l1_feat_nomiss(feat)
        })
    }

    pub fn predict_l1_feat(&self, feat: &[f32]) -> Result<OnlineL1Output> {
        if feat.len() != self.l1_dim {
            bail!("l1 feat len mismatch: got={} expected={}", feat.len(), self.l1_dim);
        }
        let nan_free = !feat.iter().any(|x| x.is_nan());
        let (score, passed, visited, _) = if nan_free {
            unsafe {
                if let Some(hot_pack) = self.l1.hot_compact_pack.as_ref() {
                    traverse_approx_float_nomiss_l1_hot_compact(
                        hot_pack,
                        &self.l1.model,
                        feat,
                        self.l1.threshold,
                        self.l1.base_score,
                        &self.l1.plan,
                        &self.l1.approx_policy,
                        0.0,
                        0.0,
                        1,
                    )?
                } else {
                    traverse_approx_float_nomiss_l1_hot(
                        &self.l1.model,
                        feat,
                        self.l1.threshold,
                        self.l1.base_score,
                        &self.l1.plan,
                        &self.l1.approx_policy,
                        0.0,
                        0.0,
                        1,
                    )?
                }
            }
        } else {
            traverse_approx_generic(
                &self.l1.model,
                None,
                None,
                feat,
                self.l1.threshold,
                self.l1.base_score,
                &self.l1.plan,
                &self.l1.approx_policy,
                0.0,
                0.0,
                1,
            )?
        };
        Ok(OnlineL1Output {
            score,
            passed,
            visited_trees: visited,
        })
    }

    pub fn predict_l1_feat_nomiss(&self, feat: &[f32]) -> Result<OnlineL1Output> {
        if feat.len() != self.l1_dim {
            bail!("l1 feat len mismatch: got={} expected={}", feat.len(), self.l1_dim);
        }
        let (score, passed, visited, _) = unsafe {
            if let Some(hot_pack) = self.l1.hot_compact_pack.as_ref() {
                traverse_approx_float_nomiss_l1_hot_compact(
                    hot_pack,
                    &self.l1.model,
                    feat,
                    self.l1.threshold,
                    self.l1.base_score,
                    &self.l1.plan,
                    &self.l1.approx_policy,
                    0.0,
                    0.0,
                    1,
                )?
            } else {
                traverse_approx_float_nomiss_l1_hot(
                    &self.l1.model,
                    feat,
                    self.l1.threshold,
                    self.l1.base_score,
                    &self.l1.plan,
                    &self.l1.approx_policy,
                    0.0,
                    0.0,
                    1,
                )?
            }
        };
        Ok(OnlineL1Output {
            score,
            passed,
            visited_trees: visited,
        })
    }

    pub fn predict_l2_row(&self, row: &[f32], row_tau: f32, row_fold: i32) -> Result<OnlineL2Output> {
        if row.len() != self.l2_dim {
            bail!("l2 row len mismatch: got={} expected={}", row.len(), self.l2_dim);
        }
        ONLINE_TLS_SCRATCH.with(|cell| -> Result<OnlineL2Output> {
            let mut scratch = cell.borrow_mut();
            let nan_free = !row.iter().any(|x| x.is_nan());
            if nan_free {
                if let Some((hot_buf, ranks)) = scratch.ensure_l2_nomiss_fast(&self.l2.runtime) {
                    if let Some(shadow) = run_prefix_shadow_row_single_route_prefix_v2_nomiss(
                        &self.l2.runtime,
                        row,
                        row_tau,
                        row_fold,
                        hot_buf,
                        ranks,
                    )? {
                        if !shadow.fallback_used {
                            return Ok(OnlineL2Output {
                                score: shadow.shadow_route_score,
                                reject: shadow.shadow_reject,
                                used_fallback: false,
                                trees_used: shadow.trees_used as i32,
                            });
                        }

                        if self.l2.runtime.anchor_rescue.is_none()
                            && self.l2.runtime.packet_scheduler.is_none()
                        {
                            let (score, reject, visited, _) = unsafe {
                                traverse_float_nomiss_from(
                                    &self.l2.model,
                                    row,
                                    row_tau,
                                    shadow.prefix_score,
                                    InferMode::L2RouteExactReordered,
                                    &self.l2.runtime.plan,
                                    None,
                                    0.0,
                                    0.0,
                                    1,
                                    shadow.trees_used,
                                )
                            }?;
                            return Ok(OnlineL2Output {
                                score,
                                reject,
                                used_fallback: true,
                                trees_used: shadow.trees_used as i32 + visited,
                            });
                        }

                        let (score, reject, visited, _) = run_exact_continuation_nomiss(
                            &self.l2.runtime,
                            &self.l2.model,
                            row,
                            row_tau,
                            &shadow,
                            ranks,
                        )?;
                        return Ok(OnlineL2Output {
                            score,
                            reject,
                            used_fallback: true,
                            trees_used: shadow.trees_used as i32 + visited,
                        });
                    }
                    if let Some(shadow) = run_prefix_shadow_row_single_route_compiled_nomiss_online(
                        &self.l2.runtime,
                        row,
                        row_tau,
                        row_fold,
                        hot_buf,
                        ranks,
                    )? {
                        if !shadow.fallback_used {
                            return Ok(OnlineL2Output {
                                score: shadow.shadow_route_score,
                                reject: shadow.shadow_reject,
                                used_fallback: false,
                                trees_used: shadow.trees_used as i32,
                            });
                        }

                        if self.l2.runtime.anchor_rescue.is_none()
                            && self.l2.runtime.packet_scheduler.is_none()
                        {
                            let (score, reject, visited, _) = unsafe {
                                traverse_float_nomiss_from(
                                    &self.l2.model,
                                    row,
                                    row_tau,
                                    shadow.prefix_score,
                                    InferMode::L2RouteExactReordered,
                                    &self.l2.runtime.plan,
                                    None,
                                    0.0,
                                    0.0,
                                    1,
                                    shadow.trees_used,
                                )
                            }?;
                            return Ok(OnlineL2Output {
                                score,
                                reject,
                                used_fallback: true,
                                trees_used: shadow.trees_used as i32 + visited,
                            });
                        }

                        let (score, reject, visited, _) = run_exact_continuation_nomiss(
                            &self.l2.runtime,
                            &self.l2.model,
                            row,
                            row_tau,
                            &shadow,
                            ranks,
                        )?;
                        return Ok(OnlineL2Output {
                            score,
                            reject,
                            used_fallback: true,
                            trees_used: shadow.trees_used as i32 + visited,
                        });
                    }
                }
            }

            let (ranks, missing, hot_buf, hot_cache) = scratch.ensure_l2(&self.l2.runtime);

            let shadow = run_prefix_shadow_row_single_route_compiled(
                &self.l2.runtime,
                row,
                row_tau,
                row_fold,
                nan_free,
                ranks,
                missing,
                hot_buf,
                hot_cache,
            )?;

            if !shadow.fallback_used {
                return Ok(OnlineL2Output {
                    score: shadow.shadow_route_score,
                    reject: shadow.shadow_reject,
                    used_fallback: false,
                    trees_used: shadow.trees_used as i32,
                });
            }

            let (score, reject, visited, _) = run_exact_continuation(
                &self.l2.runtime,
                &self.l2.model,
                row,
                row_tau,
                &shadow,
                nan_free,
                ranks,
                missing,
            )?;
            Ok(OnlineL2Output {
                score,
                reject,
                used_fallback: true,
                trees_used: shadow.trees_used as i32 + visited,
            })
        })
    }

    pub fn predict_l2_row_nomiss(
        &self,
        row: &[f32],
        row_tau: f32,
        row_fold: i32,
    ) -> Result<OnlineL2Output> {
        if row.len() != self.l2_dim {
            bail!("l2 row len mismatch: got={} expected={}", row.len(), self.l2_dim);
        }
        ONLINE_TLS_SCRATCH.with(|cell| -> Result<OnlineL2Output> {
            let mut scratch = cell.borrow_mut();
            let trace_enabled = l2_trace_enabled();
            if let Some((hot_buf, ranks)) = scratch.ensure_l2_nomiss_fast(&self.l2.runtime) {
                let prefix_t0 = if trace_enabled { Some(Instant::now()) } else { None };
                if let Some(shadow) = run_prefix_shadow_row_single_route_prefix_v2_nomiss(
                    &self.l2.runtime,
                    row,
                    row_tau,
                    row_fold,
                    hot_buf,
                    ranks,
                )? {
                    if let Some(t0) = prefix_t0 {
                        trace!(
                            stage = "l2_prefix",
                            fallback_used = shadow.fallback_used,
                            trees_used = shadow.trees_used,
                            us = t0.elapsed().as_micros() as u64
                        );
                    }
                    if !shadow.fallback_used {
                        return Ok(OnlineL2Output {
                            score: shadow.shadow_route_score,
                            reject: shadow.shadow_reject,
                            used_fallback: false,
                            trees_used: shadow.trees_used as i32,
                        });
                    }

                    if self.l2.runtime.anchor_rescue.is_none()
                        && self.l2.runtime.packet_scheduler.is_none()
                    {
                        let exact_t0 = if trace_enabled { Some(Instant::now()) } else { None };
                        let (score, reject, visited, _) = unsafe {
                            traverse_float_nomiss_from(
                                &self.l2.model,
                                row,
                                row_tau,
                                shadow.prefix_score,
                                InferMode::L2RouteExactReordered,
                                &self.l2.runtime.plan,
                                None,
                                0.0,
                                0.0,
                                1,
                                shadow.trees_used,
                            )
                        }?;
                        if let Some(t0) = exact_t0 {
                            trace!(
                                stage = "l2_exact_tail",
                                fast_path = true,
                                visited = visited,
                                us = t0.elapsed().as_micros() as u64
                            );
                        }
                        return Ok(OnlineL2Output {
                            score,
                            reject,
                            used_fallback: true,
                            trees_used: shadow.trees_used as i32 + visited,
                        });
                    }

                    let exact_t0 = if trace_enabled { Some(Instant::now()) } else { None };
                    let (score, reject, visited, _) = run_exact_continuation_nomiss(
                        &self.l2.runtime,
                        &self.l2.model,
                        row,
                        row_tau,
                        &shadow,
                        ranks,
                    )?;
                    if let Some(t0) = exact_t0 {
                        trace!(
                            stage = "l2_exact_continuation",
                            fast_path = false,
                            visited = visited,
                            us = t0.elapsed().as_micros() as u64
                        );
                    }
                    return Ok(OnlineL2Output {
                        score,
                        reject,
                        used_fallback: true,
                        trees_used: shadow.trees_used as i32 + visited,
                    });
                }
                if let Some(shadow) = run_prefix_shadow_row_single_route_compiled_nomiss_online(
                    &self.l2.runtime,
                    row,
                    row_tau,
                    row_fold,
                    hot_buf,
                    ranks,
                )? {
                    if let Some(t0) = prefix_t0 {
                        trace!(
                            stage = "l2_prefix",
                            fallback_used = shadow.fallback_used,
                            trees_used = shadow.trees_used,
                            us = t0.elapsed().as_micros() as u64
                        );
                    }
                    if !shadow.fallback_used {
                        return Ok(OnlineL2Output {
                            score: shadow.shadow_route_score,
                            reject: shadow.shadow_reject,
                            used_fallback: false,
                            trees_used: shadow.trees_used as i32,
                        });
                    }

                    if self.l2.runtime.anchor_rescue.is_none()
                        && self.l2.runtime.packet_scheduler.is_none()
                    {
                        let exact_t0 = if trace_enabled { Some(Instant::now()) } else { None };
                        let (score, reject, visited, _) = unsafe {
                            traverse_float_nomiss_from(
                                &self.l2.model,
                                row,
                                row_tau,
                                shadow.prefix_score,
                                InferMode::L2RouteExactReordered,
                                &self.l2.runtime.plan,
                                None,
                                0.0,
                                0.0,
                                1,
                                shadow.trees_used,
                            )
                        }?;
                        if let Some(t0) = exact_t0 {
                            trace!(
                                stage = "l2_exact_tail",
                                fast_path = true,
                                visited = visited,
                                us = t0.elapsed().as_micros() as u64
                            );
                        }
                        return Ok(OnlineL2Output {
                            score,
                            reject,
                            used_fallback: true,
                            trees_used: shadow.trees_used as i32 + visited,
                        });
                    }

                    let exact_t0 = if trace_enabled { Some(Instant::now()) } else { None };
                    let (score, reject, visited, _) = run_exact_continuation_nomiss(
                        &self.l2.runtime,
                        &self.l2.model,
                        row,
                        row_tau,
                        &shadow,
                        ranks,
                    )?;
                    if let Some(t0) = exact_t0 {
                        trace!(
                            stage = "l2_exact_continuation",
                            fast_path = false,
                            visited = visited,
                            us = t0.elapsed().as_micros() as u64
                        );
                    }
                    return Ok(OnlineL2Output {
                        score,
                        reject,
                        used_fallback: true,
                        trees_used: shadow.trees_used as i32 + visited,
                    });
                }
            }
            let (ranks, missing, hot_buf, hot_cache) = scratch.ensure_l2(&self.l2.runtime);
            let shadow = run_prefix_shadow_row_single_route_compiled(
                &self.l2.runtime,
                row,
                row_tau,
                row_fold,
                true,
                ranks,
                missing,
                hot_buf,
                hot_cache,
            )?;

            if !shadow.fallback_used {
                return Ok(OnlineL2Output {
                    score: shadow.shadow_route_score,
                    reject: shadow.shadow_reject,
                    used_fallback: false,
                    trees_used: shadow.trees_used as i32,
                });
            }

            let (score, reject, visited, _) = run_exact_continuation(
                &self.l2.runtime,
                &self.l2.model,
                row,
                row_tau,
                &shadow,
                true,
                ranks,
                missing,
            )?;
            Ok(OnlineL2Output {
                score,
                reject,
                used_fallback: true,
                trees_used: shadow.trees_used as i32 + visited,
            })
        })
    }

    pub fn predict_l2_row_nomiss_experimental_zen4(
        &self,
        row: &[f32],
        row_tau: f32,
        row_fold: i32,
    ) -> Result<OnlineL2Output> {
        if row.len() != self.l2_dim {
            bail!("l2 row len mismatch: got={} expected={}", row.len(), self.l2_dim);
        }
        if let Some(exp) = self.l2.kernel_zen4_v1.as_ref() {
            return ONLINE_TLS_SCRATCH.with(|cell| -> Result<OnlineL2Output> {
                let mut scratch = cell.borrow_mut();
                let (hot_buf, local_ranks) = scratch.ensure_l2_exp(exp);
                l2_kernel_zen4::predict_nomiss(
                    exp,
                    &self.l2.runtime,
                    row,
                    row_tau,
                    row_fold,
                    hot_buf,
                    local_ranks,
                )
            });
        }
        if let Some(exp) = self.l2.exp_v1.as_ref() {
            return ONLINE_TLS_SCRATCH.with(|cell| -> Result<OnlineL2Output> {
                let mut scratch = cell.borrow_mut();
                let (hot_buf, local_ranks) = scratch.ensure_l2_exp(exp);
                l2_exp_v1::predict_nomiss(
                    exp,
                    &self.l2.runtime,
                    row,
                    row_tau,
                    row_fold,
                    hot_buf,
                    local_ranks,
                )
            });
        }
        let kernel = self
            .l2
            .experimental_zen4
            .ok_or_else(|| anyhow::anyhow!("experimental ZEN4 L2 kernel unavailable"))?;
        ONLINE_TLS_SCRATCH.with(|cell| -> Result<OnlineL2Output> {
            let mut scratch = cell.borrow_mut();
            let (hot_buf, ranks) = scratch
                .ensure_l2_nomiss_fast(&self.l2.runtime)
                .ok_or_else(|| anyhow::anyhow!("experimental ZEN4 L2 hot path unavailable"))?;
            kernel.predict_nomiss(&self.l2.runtime, &self.l2.model, row, row_tau, row_fold, hot_buf, ranks)
        })
    }
}

fn absolutize_resolved_bundle(root: &PathBuf, mut resolved: ResolvedPrefixCalBundle) -> ResolvedPrefixCalBundle {
    resolved.qs_pack = bundle_abspath(root, resolved.qs_pack);
    resolved.calibration_json = bundle_abspath(root, resolved.calibration_json);
    resolved.feat_bin = bundle_abspath(root, resolved.feat_bin);
    resolved.route_meta = bundle_abspath(root, resolved.route_meta);
    resolved.soa = bundle_abspath(root, resolved.soa);
    resolved.bounds = bundle_abspath(root, resolved.bounds);
    resolved.tree_order = bundle_abspath(root, resolved.tree_order);
    resolved.model_json = bundle_abspath(root, resolved.model_json);
    resolved.certifier_json = resolved.certifier_json.map(|p| bundle_abspath(root, p));
    resolved.atlas_bin = resolved.atlas_bin.map(|p| bundle_abspath(root, p));
    resolved.hot_exact_prefix_pack = resolved
        .hot_exact_prefix_pack
        .map(|p| bundle_abspath(root, p));
    resolved
}

fn bundle_abspath(root: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        root.join(path)
    }
}

fn require_path(root: &Path, rel: &str) -> Result<PathBuf> {
    let path = root.join(rel);
    if path.exists() {
        Ok(path)
    } else {
        bail!("required bundle asset missing: {}", path.display())
    }
}

fn load_feature_names_from_json(path: &PathBuf) -> Result<Vec<String>> {
    let txt = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let out: Vec<String> =
        serde_json::from_str(&txt).with_context(|| format!("parse {}", path.display()))?;
    Ok(out)
}

fn decode_dense_f32le(bytes: &[u8], dim: usize) -> Result<Vec<f32>> {
    let need = dim
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("dim too large: {}", dim))?;
    if bytes.len() != need {
        bail!("dense bytes len mismatch: got={} expected={}", bytes.len(), need);
    }
    let mut out = vec![0.0f32; dim];
    for (i, slot) in out.iter_mut().enumerate() {
        let off = i * 4;
        *slot = f32::from_le_bytes([
            bytes[off],
            bytes[off + 1],
            bytes[off + 2],
            bytes[off + 3],
        ]);
    }
    Ok(out)
}

fn decode_dense_f32le_into(bytes: &[u8], dim: usize, out: &mut Vec<f32>) -> Result<()> {
    let need = dim
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("dim too large: {}", dim))?;
    if bytes.len() != need {
        bail!("dense bytes len mismatch: got={} expected={}", bytes.len(), need);
    }
    if out.len() != dim {
        out.resize(dim, 0.0);
    }
    #[cfg(target_endian = "little")]
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, need);
        return Ok(());
    }
    #[allow(unreachable_code)]
    {
        for (i, slot) in out.iter_mut().enumerate() {
            let off = i * 4;
            *slot = f32::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
            ]);
        }
        Ok(())
    }
}

fn current_rss_mb() -> f64 {
    let Ok(txt) = fs::read_to_string("/proc/self/status") else {
        return 0.0;
    };
    for line in txt.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
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

fn log_load_mem(stage: &str) {
    info!(
        stage = stage,
        rss_mb = current_rss_mb(),
        peak_rss_mb = peak_rss_mb(),
        "quickscorer load memory"
    );
}
