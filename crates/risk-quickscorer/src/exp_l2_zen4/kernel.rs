use anyhow::{bail, Result};

use super::super::{
    run_exact_continuation_nomiss, run_prefix_shadow_row_single_route_compiled_nomiss_online,
    HotFeatureBuf, LoadedPrefixRuntime, OnlineL2Output, PrefixCertifierKind, PrefixDirectKernel,
    SoaModel,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct ExperimentalL2KernelZen4;

impl ExperimentalL2KernelZen4 {
    pub(crate) fn new(runtime: &LoadedPrefixRuntime) -> Option<Self> {
        let direct_ok = matches!(
            runtime.direct_kernel,
            PrefixDirectKernel::HotExact96
                | PrefixDirectKernel::HotExact128
                | PrefixDirectKernel::HotExact192
                | PrefixDirectKernel::HotExact256
                | PrefixDirectKernel::HotExact384
        );
        if !direct_ok {
            return None;
        }
        if runtime.certifier_kind != PrefixCertifierKind::AtlasV1 {
            return None;
        }
        if runtime.compiled_hot_pack.is_none() || runtime.hot_checkpoint_layout.is_none() {
            return None;
        }
        Some(Self)
    }

    #[inline(always)]
    pub(crate) fn predict_nomiss(
        &self,
        runtime: &LoadedPrefixRuntime,
        model: &SoaModel,
        row: &[f32],
        row_tau: f32,
        row_fold: i32,
        hot_buf: &mut HotFeatureBuf,
        ranks: &mut [u8],
    ) -> Result<OnlineL2Output> {
        let Some(shadow) = run_prefix_shadow_row_single_route_compiled_nomiss_online(
            runtime, row, row_tau, row_fold, hot_buf, ranks,
        )?
        else {
            bail!("experimental ZEN4 L2 kernel unavailable for current runtime");
        };

        if !shadow.fallback_used {
            return Ok(OnlineL2Output {
                score: shadow.shadow_route_score,
                reject: shadow.shadow_reject,
                used_fallback: false,
                trees_used: shadow.trees_used as i32,
            });
        }

        let (score, reject, visited, _) =
            run_exact_continuation_nomiss(runtime, model, row, row_tau, &shadow, ranks)?;
        Ok(OnlineL2Output {
            score,
            reject,
            used_fallback: true,
            trees_used: shadow.trees_used as i32 + visited,
        })
    }
}
