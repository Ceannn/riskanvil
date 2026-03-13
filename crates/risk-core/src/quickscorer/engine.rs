use crate::quickscorer::decision::{final_decision, l2_is_reject};
use crate::quickscorer::policy::{L2FeatureSource, QuickPolicy};
use crate::quickscorer::types::{
    QuickDebugInfo, QuickL1PredictOutput, QuickPredictOutput, QuickRouteMeta,
};
use crate::schema::Decision;
use anyhow::{anyhow, ensure};
use risk_quickscorer::MinpackRuntime;
use std::cell::RefCell;
use std::path::Path;
use std::time::Instant;

thread_local! {
    static QUICK_TLS_SCRATCH: RefCell<QuickTlsScratch> = RefCell::new(QuickTlsScratch::default());
}

#[derive(Default)]
struct QuickTlsScratch {
    l1_feat: Vec<f32>,
    l2_row: Vec<f32>,
    seg_key_buf: Vec<u8>,
    batch_l1: Vec<f32>,
    batch_l1_scores: Vec<f32>,
    batch_survivors: Vec<usize>,
}

#[derive(Debug)]
pub struct QuickScorerEngine {
    policy: QuickPolicy,
    runtime: MinpackRuntime,
}

impl QuickScorerEngine {
    pub fn load(bundle_dir: &Path) -> anyhow::Result<Self> {
        let policy = QuickPolicy::load_bundle(bundle_dir)?;
        let runtime = MinpackRuntime::load(bundle_dir)?;

        ensure!(
            runtime.l1_dim() == policy.l1_dim,
            "quickscorer l1 dim mismatch: policy={} runtime={}",
            policy.l1_dim,
            runtime.l1_dim()
        );
        if let Some(l2) = policy.l2.as_ref() {
            ensure!(
                runtime.l2_dim() == l2.dim,
                "quickscorer l2 dim mismatch: policy={} runtime={}",
                l2.dim,
                runtime.l2_dim()
            );
        }

        Ok(Self { policy, runtime })
    }

    #[inline]
    pub fn backend_name(&self) -> &'static str {
        self.runtime.backend_name()
    }

    #[inline]
    pub fn l1_dim(&self) -> usize {
        self.policy.l1_dim
    }

    #[inline]
    pub fn l2_dim(&self) -> usize {
        self.policy.l2.as_ref().map(|x| x.dim).unwrap_or(0)
    }

    #[inline]
    pub fn l1_threshold(&self) -> f32 {
        self.policy.l1_threshold
    }

    pub fn debug_info(&self) -> QuickDebugInfo {
        let l2 = self.policy.l2.as_ref();
        QuickDebugInfo {
            backend: self.backend_name(),
            l1_dim: self.policy.l1_dim,
            l2_dim: self.l2_dim(),
            l1_threshold: self.policy.l1_threshold,
            l2_default_fold: l2.map(|x| x.default_fold).unwrap_or(0),
            l2_gb_target: l2.map(|x| x.gb_target.clone()).unwrap_or_default(),
            l2_segmented: l2.map(|x| x.seg_enabled).unwrap_or(false),
            l2_seg_cols: l2.map(|x| x.seg_cols.clone()).unwrap_or_default(),
        }
    }

    pub fn predict_from_l1_bytes(&self, l1_bytes: &[u8]) -> anyhow::Result<QuickPredictOutput> {
        self.predict_from_l1_bytes_with_meta(l1_bytes, None)
    }

    pub fn predict_l1_only_from_bytes(
        &self,
        l1_bytes: &[u8],
    ) -> anyhow::Result<QuickL1PredictOutput> {
        let l1_need = self
            .policy
            .l1_dim
            .checked_mul(4)
            .ok_or_else(|| anyhow!("l1_dim too large"))?;
        ensure!(
            l1_bytes.len() == l1_need,
            "quickscorer input size mismatch: got={}, expect={} (l1_dim={})",
            l1_bytes.len(),
            l1_need,
            self.policy.l1_dim
        );

        let t_router = Instant::now();
        let l1_dim = self.policy.l1_dim;

        QUICK_TLS_SCRATCH.with(|cell| -> anyhow::Result<QuickL1PredictOutput> {
            let mut scratch = cell.borrow_mut();
            decode_dense_f32le_into(l1_bytes, l1_dim, &mut scratch.l1_feat)?;
            let t_l1 = Instant::now();
            let l1_out = self
                .runtime
                .predict_l1_feat_nomiss(scratch.l1_feat.as_slice())?;
            Ok(QuickL1PredictOutput {
                l1_score: l1_out.score,
                passed: l1_out.passed,
                l1_us: elapsed_us(t_l1),
                router_us: elapsed_us(t_router),
            })
        })
    }

    pub fn predict_l1_only_batch128_from_bytes(
        &self,
        rows: &[&[u8]; 128],
    ) -> anyhow::Result<[QuickL1PredictOutput; 128]> {
        let l1_need = self
            .policy
            .l1_dim
            .checked_mul(4)
            .ok_or_else(|| anyhow!("l1_dim too large"))?;
        let t_router = Instant::now();
        let l1_dim = self.policy.l1_dim;

        QUICK_TLS_SCRATCH.with(|cell| -> anyhow::Result<[QuickL1PredictOutput; 128]> {
            let mut scratch = cell.borrow_mut();
            let batch_l1 = &mut scratch.batch_l1;
            let mut outs = std::array::from_fn(|_| QuickL1PredictOutput {
                l1_score: 0.0,
                passed: false,
                l1_us: 0,
                router_us: 0,
            });
            let batch_need = 128usize
                .checked_mul(l1_dim)
                .ok_or_else(|| anyhow!("batch l1 scratch too large"))?;
            if batch_l1.len() != batch_need {
                batch_l1.resize(batch_need, 0.0);
            }
            for i in 0..128 {
                let l1_bytes = rows[i];
                ensure!(
                    l1_bytes.len() == l1_need,
                    "quickscorer input size mismatch: got={}, expect={} (l1_dim={})",
                    l1_bytes.len(),
                    l1_need,
                    l1_dim
                );
                let row_off = i * l1_dim;
                let l1_feat_row = &mut batch_l1[row_off..row_off + l1_dim];
                #[cfg(target_endian = "little")]
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        l1_bytes.as_ptr(),
                        l1_feat_row.as_mut_ptr() as *mut u8,
                        l1_need,
                    );
                }
                #[cfg(not(target_endian = "little"))]
                {
                    for (j, slot) in l1_feat_row.iter_mut().enumerate() {
                        let off = j * 4;
                        *slot = f32::from_le_bytes([
                            l1_bytes[off],
                            l1_bytes[off + 1],
                            l1_bytes[off + 2],
                            l1_bytes[off + 3],
                        ]);
                    }
                }
                let t_l1 = Instant::now();
                let l1_out = self.runtime.predict_l1_feat_nomiss(l1_feat_row)?;
                outs[i] = QuickL1PredictOutput {
                    l1_score: l1_out.score,
                    passed: l1_out.passed,
                    l1_us: elapsed_us(t_l1),
                    router_us: elapsed_us(t_router),
                };
            }
            Ok(outs)
        })
    }

    pub fn predict_batch128_counts_from_l1_bytes_with_meta(
        &self,
        rows: &[&[u8]; 128],
        route_metas: &[Option<QuickRouteMeta>; 128],
    ) -> anyhow::Result<(u32, [u32; 5])> {
        let l1_need = self
            .policy
            .l1_dim
            .checked_mul(4)
            .ok_or_else(|| anyhow!("l1_dim too large"))?;
        let l1_dim = self.policy.l1_dim;
        let Some(l2_policy) = self.policy.l2.as_ref() else {
            let l1_outs = self.predict_l1_only_batch128_from_bytes(rows)?;
            let mut decision_counts = [0u32; 5];
            for out in l1_outs {
                let idx = if out.passed { 0 } else { 2 };
                decision_counts[idx] += 1;
            }
            return Ok((0, decision_counts));
        };

        QUICK_TLS_SCRATCH.with(|cell| -> anyhow::Result<(u32, [u32; 5])> {
            let mut scratch = cell.borrow_mut();
            let QuickTlsScratch {
                l1_feat: _,
                l2_row,
                seg_key_buf,
                batch_l1,
                batch_l1_scores,
                batch_survivors,
            } = &mut *scratch;
            let mut used_l2_count = 0u32;
            let mut decision_counts = [0u32; 5];
            let batch_need = 128usize
                .checked_mul(l1_dim)
                .ok_or_else(|| anyhow!("batch l1 scratch too large"))?;
            if batch_l1.len() != batch_need {
                batch_l1.resize(batch_need, 0.0);
            }
            if batch_l1_scores.len() != 128 {
                batch_l1_scores.resize(128, 0.0);
            }
            batch_survivors.clear();

            for i in 0..128 {
                let l1_bytes = rows[i];
                ensure!(
                    l1_bytes.len() == l1_need,
                    "quickscorer input size mismatch: got={}, expect={} (l1_dim={})",
                    l1_bytes.len(),
                    l1_need,
                    l1_dim
                );
                let row_off = i * l1_dim;
                let l1_feat_row = &mut batch_l1[row_off..row_off + l1_dim];
                #[cfg(target_endian = "little")]
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        l1_bytes.as_ptr(),
                        l1_feat_row.as_mut_ptr() as *mut u8,
                        l1_need,
                    );
                }
                #[cfg(not(target_endian = "little"))]
                {
                    for (j, slot) in l1_feat_row.iter_mut().enumerate() {
                        let off = j * 4;
                        *slot = f32::from_le_bytes([
                            l1_bytes[off],
                            l1_bytes[off + 1],
                            l1_bytes[off + 2],
                            l1_bytes[off + 3],
                        ]);
                    }
                }
                let l1_out = self.runtime.predict_l1_feat_nomiss(l1_feat_row)?;
                let l1_score = l1_out.score;
                batch_l1_scores[i] = l1_score;

                if l1_out.passed {
                    decision_counts[0] += 1;
                } else {
                    batch_survivors.push(i);
                }
            }

            for &i in batch_survivors.iter() {
                let row_off = i * l1_dim;
                let l1_feat_row = &batch_l1[row_off..row_off + l1_dim];
                let l1_score = batch_l1_scores[i];
                materialize_l2_row_into(l1_feat_row, l1_score, l2_policy, l2_row);
                let l2_resolved = l2_policy.resolve_threshold_with_route_meta(
                    l2_row.as_slice(),
                    route_metas[i].as_ref(),
                    seg_key_buf,
                )?;
                let l2_out = self.runtime.predict_l2_row_nomiss(
                    l2_row.as_slice(),
                    l2_resolved.tau,
                    l2_resolved.fold,
                )?;
                used_l2_count += 1;
                let reject = l2_is_reject(l2_out.score, l2_resolved.tau);
                let idx = match final_decision(false, Some(reject)) {
                    Decision::Allow => 0,
                    Decision::Deny => 1,
                    Decision::ManualReview => 2,
                    Decision::DegradeAllow => 3,
                };
                decision_counts[idx] += 1;
            }

            Ok((used_l2_count, decision_counts))
        })
    }

    pub fn predict_from_l1_bytes_with_meta(
        &self,
        l1_bytes: &[u8],
        route_meta: Option<&QuickRouteMeta>,
    ) -> anyhow::Result<QuickPredictOutput> {
        let l1_need = self
            .policy
            .l1_dim
            .checked_mul(4)
            .ok_or_else(|| anyhow!("l1_dim too large"))?;
        ensure!(
            l1_bytes.len() == l1_need,
            "quickscorer input size mismatch: got={}, expect={} (l1_dim={})",
            l1_bytes.len(),
            l1_need,
            self.policy.l1_dim
        );

        let t_router = Instant::now();
        let l1_dim = self.policy.l1_dim;

        let Some(l2_policy) = self.policy.l2.as_ref() else {
            let t_l1 = Instant::now();
            let l1_out = self.runtime.predict_l1_bytes_nomiss(l1_bytes)?;
            let l1_score = l1_out.score;
            return Ok(QuickPredictOutput {
                l1_score,
                l2_score: None,
                final_score: l1_score,
                decision: if l1_out.passed {
                    Decision::Allow
                } else {
                    Decision::ManualReview
                },
                used_l2: false,
                feature_us: 0,
                l1_us: elapsed_us(t_l1),
                l2_us: 0,
                router_us: elapsed_us(t_router),
            });
        };

        QUICK_TLS_SCRATCH.with(|cell| -> anyhow::Result<QuickPredictOutput> {
            let mut scratch = cell.borrow_mut();
            let QuickTlsScratch {
                l1_feat,
                l2_row,
                seg_key_buf,
                ..
            } = &mut *scratch;

            decode_dense_f32le_into(l1_bytes, l1_dim, l1_feat)?;

            let t_l1 = Instant::now();
            let l1_out = self.runtime.predict_l1_feat_nomiss(l1_feat.as_slice())?;
            let l1_score = l1_out.score;
            let l1_us = elapsed_us(t_l1);

            if l1_out.passed {
                return Ok(QuickPredictOutput {
                    l1_score,
                    l2_score: None,
                    final_score: l1_score,
                    decision: Decision::Allow,
                    used_l2: false,
                    feature_us: 0,
                    l1_us,
                    l2_us: 0,
                    router_us: elapsed_us(t_router),
                });
            }

            let t_feature = Instant::now();
            materialize_l2_row_into(l1_feat.as_slice(), l1_score, l2_policy, l2_row);
            let feature_us = elapsed_us(t_feature);
            let l2_resolved = l2_policy.resolve_threshold_with_route_meta(
                l2_row.as_slice(),
                route_meta,
                seg_key_buf,
            )?;

            let t_l2 = Instant::now();
            let l2_out = self.runtime.predict_l2_row_nomiss(
                l2_row.as_slice(),
                l2_resolved.tau,
                l2_resolved.fold,
            )?;
            let l2_us = elapsed_us(t_l2);
            let l2_score = l2_out.score;

            let reject = l2_is_reject(l2_score, l2_resolved.tau);
            let decision = final_decision(false, Some(reject));

            Ok(QuickPredictOutput {
                l1_score,
                l2_score: Some(l2_score),
                final_score: l2_score,
                decision,
                used_l2: true,
                feature_us,
                l1_us,
                l2_us,
                router_us: elapsed_us(t_router),
            })
        })
    }
}

fn materialize_l2_row_into(
    l1_feat: &[f32],
    l1_score: f32,
    l2_policy: &crate::quickscorer::policy::L2Policy,
    out: &mut Vec<f32>,
) {
    if out.len() != l2_policy.dim {
        out.resize(l2_policy.dim, f32::NAN);
    }
    for (i, src) in l2_policy.feature_sources.iter().enumerate() {
        out[i] = match src {
            L2FeatureSource::FromL1(idx) => l1_feat[*idx],
            L2FeatureSource::L1Score => l1_score,
        };
    }
}

fn decode_dense_f32le_into(bytes: &[u8], dim: usize, out: &mut Vec<f32>) -> anyhow::Result<()> {
    let need = dim
        .checked_mul(4)
        .ok_or_else(|| anyhow!("l1_dim too large"))?;
    ensure!(
        bytes.len() == need,
        "quickscorer input size mismatch: got={}, expect={} (l1_dim={})",
        bytes.len(),
        need,
        dim
    );
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
            *slot =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
        Ok(())
    }
}

#[inline]
fn elapsed_us(t0: Instant) -> u64 {
    t0.elapsed().as_micros().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use memmap2::Mmap;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::fs::File;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingAlloc;

    static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);

    #[global_allocator]
    static GLOBAL: CountingAlloc = CountingAlloc;

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            unsafe { System.realloc(ptr, layout, new_size) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    fn reset_alloc_calls() {
        ALLOC_CALLS.store(0, Ordering::Relaxed);
    }

    fn alloc_calls() -> usize {
        ALLOC_CALLS.load(Ordering::Relaxed)
    }

    fn local_bundle_dir() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("QS_TEST_BUNDLE_DIR") {
            let path = PathBuf::from(p);
            if path.exists() {
                return Some(path);
            }
        }
        None
    }

    fn encode_constant_row(dim: usize, value: f32) -> Vec<u8> {
        let mut out = Vec::with_capacity(dim * 4);
        for _ in 0..dim {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn encode_alternating_row(dim: usize, a: f32, b: f32) -> Vec<u8> {
        let mut out = Vec::with_capacity(dim * 4);
        for i in 0..dim {
            let v = if i % 2 == 0 { a } else { b };
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn find_review_sample(engine: &QuickScorerEngine) -> anyhow::Result<Vec<u8>> {
        let dim = engine.l1_dim();
        if let Ok(path) = std::env::var("QS_TEST_REVIEW_SAMPLE_BIN") {
            let row = std::fs::read(&path)?;
            if row.len() == dim * 4 {
                let out = engine.predict_from_l1_bytes(&row)?;
                if out.used_l2 {
                    return Ok(row);
                }
            }
        }
        let candidates = [
            encode_constant_row(dim, 0.0),
            encode_constant_row(dim, -1.0),
            encode_constant_row(dim, 1.0),
            encode_constant_row(dim, -10.0),
            encode_constant_row(dim, 10.0),
            encode_constant_row(dim, f32::NAN),
            encode_alternating_row(dim, 0.0, f32::NAN),
            encode_alternating_row(dim, -1.0, 1.0),
        ];
        for row in candidates {
            let out = engine.predict_from_l1_bytes(&row)?;
            if out.used_l2 {
                return Ok(row);
            }
        }

        let feature_bin = local_bundle_dir()
            .map(|p| p.join("runs/L1_RUST_DEV/l1_features_120k_v2.bin"))
            .filter(|p| p.exists())
            .ok_or_else(|| anyhow::anyhow!("local L1 feature bin unavailable"))?;
        find_review_sample_in_feature_bin(engine, &feature_bin)
    }

    fn find_review_sample_in_feature_bin(
        engine: &QuickScorerEngine,
        path: &std::path::Path,
    ) -> anyhow::Result<Vec<u8>> {
        const MAGIC_FEA_V2: &[u8] = b"L1FEATv2\0";

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let buf = &mmap[..];
        if buf.len() < MAGIC_FEA_V2.len() + 28 || &buf[..MAGIC_FEA_V2.len()] != MAGIC_FEA_V2 {
            return Err(anyhow::anyhow!(
                "unexpected feature bin format: {}",
                path.display()
            ));
        }

        let mut off = MAGIC_FEA_V2.len();
        let n_rows = le_u32(buf, &mut off)? as usize;
        let n_cols = le_u32(buf, &mut off)? as usize;
        let x_offset = le_u64(buf, &mut off)? as usize;
        let _ids_offset = le_u64(buf, &mut off)? as usize;
        let _y_offset = le_u64(buf, &mut off)? as usize;

        if n_cols != engine.l1_dim() {
            return Err(anyhow::anyhow!(
                "feature bin dim mismatch: bin={} engine={}",
                n_cols,
                engine.l1_dim()
            ));
        }

        let row_bytes = n_cols
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("row byte size overflow"))?;
        let limit = n_rows.min(4096);
        for row_idx in 0..limit {
            let start = x_offset + row_idx * row_bytes;
            let end = start + row_bytes;
            if end > buf.len() {
                break;
            }
            let row = &buf[start..end];
            let out = engine.predict_from_l1_bytes(row)?;
            if out.used_l2 {
                return Ok(row.to_vec());
            }
        }

        Err(anyhow::anyhow!(
            "failed to find review->l2 sample in first {} rows of {}",
            limit,
            path.display()
        ))
    }

    fn le_u32(buf: &[u8], off: &mut usize) -> anyhow::Result<u32> {
        let end = *off + 4;
        let bytes: [u8; 4] = buf
            .get(*off..end)
            .ok_or_else(|| anyhow::anyhow!("truncated u32"))?
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid u32 slice"))?;
        *off = end;
        Ok(u32::from_le_bytes(bytes))
    }

    fn le_u64(buf: &[u8], off: &mut usize) -> anyhow::Result<u64> {
        let end = *off + 8;
        let bytes: [u8; 8] = buf
            .get(*off..end)
            .ok_or_else(|| anyhow::anyhow!("truncated u64"))?
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid u64 slice"))?;
        *off = end;
        Ok(u64::from_le_bytes(bytes))
    }

    #[test]
    #[ignore = "requires local quickscorer bundle on the development machine"]
    fn online_allow_path_is_zero_alloc_after_warmup() -> anyhow::Result<()> {
        let Some(bundle_dir) = local_bundle_dir() else {
            return Ok(());
        };
        let engine = QuickScorerEngine::load(&bundle_dir)?;
        let row = encode_constant_row(engine.l1_dim(), 0.0);

        let first = engine.predict_from_l1_bytes(&row)?;
        assert!(
            !first.used_l2,
            "expected warmup sample to stay on L1 allow path"
        );

        reset_alloc_calls();
        let out = engine.predict_from_l1_bytes(&row)?;
        let calls = alloc_calls();
        assert!(!out.used_l2, "expected allow path on second run");
        assert_eq!(
            calls, 0,
            "allow path allocated {} times after warmup",
            calls
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires local quickscorer bundle on the development machine"]
    fn online_review_l2_path_is_zero_alloc_after_warmup() -> anyhow::Result<()> {
        let Some(bundle_dir) = local_bundle_dir() else {
            return Ok(());
        };
        let engine = QuickScorerEngine::load(&bundle_dir)?;
        let row = match find_review_sample(&engine) {
            Ok(row) => row,
            Err(err) => {
                eprintln!("skipping review->l2 zero-alloc gate: {err:#}");
                return Ok(());
            }
        };

        let first = engine.predict_from_l1_bytes(&row)?;
        assert!(first.used_l2, "expected warmup sample to enter L2");

        reset_alloc_calls();
        let out = engine.predict_from_l1_bytes(&row)?;
        let calls = alloc_calls();
        assert!(out.used_l2, "expected review->l2 path on second run");
        assert_eq!(
            calls, 0,
            "review->l2 path allocated {} times after warmup",
            calls
        );
        Ok(())
    }
}
