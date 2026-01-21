use crate::{
    CpuDispatch, ForwardKind, KernelError, LayerWeights, MathBackend, ModelConfig, StateSurgery,
    WeightsView,
};
use crate::{
    exp_approx_scalar, gelu_dispatch, gelu_sigmoid_fast, layer_norm_dispatch, l2norm,
    matmul_packed16_batch_dispatch, matmul_packed16_dispatch, matmul_packed16_dispatch_m1,
    matmul_packed16_tail_dispatch, matmul_packed16_tail_dispatch_m1, matmul_vec_dispatch,
    sigmoid_dispatch, silu_dispatch, softplus_dispatch,
};
#[cfg(target_arch = "x86_64")]
use crate::{
    exp256_ps,
    exp512_ps,
    gelu_sigmoid_fast_avx2,
    gelu_sigmoid_fast_avx512,
    matmul_packed16_batch_avx512_bf16,
    sigmoid_fast_avx2,
    sigmoid_fast_avx512,
};
#[cfg(feature = "bench_instrument")]
use crate::exp_approx_scalar;
use crate::scratch::{scratch_from_bytes, scratch_full_from_bytes, Scratch};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

static AUDIT_ONCE: AtomicU64 = AtomicU64::new(0);
static AUDIT_TRACE: OnceLock<Mutex<HashMap<String, u64>>> = OnceLock::new();
static STAGE_SAMPLE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn stage_sample_rate() -> u64 {
    static RATE: OnceLock<u64> = OnceLock::new();
    *RATE.get_or_init(|| {
        std::env::var("RISK_MAMBA_STAGE_SAMPLE")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1024)
    })
}

const AUDIT_CONV: u64 = 1 << 0;
const AUDIT_MATMUL: u64 = 1 << 1;
const AUDIT_SSM: u64 = 1 << 2;
const AUDIT_MLP: u64 = 1 << 3;
const AUDIT_HEAD: u64 = 1 << 4;
const AUDIT_FALLBACK: u64 = 1 << 5;

fn audit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RISK_MAMBA_AUDIT_KERNEL")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

fn no_fallback() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RISK_MAMBA_NO_FALLBACK")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

fn allow_latency_experiments() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RISK_MAMBA_ALLOW_LATENCY_EXPERIMENTS")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

fn gelu_fast_for_approx() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RISK_MAMBA_GELU_FAST")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

fn softplus_fast_for_approx() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("RISK_MAMBA_SOFTPLUS_FAST")
            .ok()
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

fn use_bf16_matmul(math_backend: MathBackend, dispatch: CpuDispatch) -> bool {
    if dispatch != CpuDispatch::Avx512 {
        return false;
    }
    if !matches!(math_backend, MathBackend::FastBf16 | MathBackend::FastWild) {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx512bf16")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

const SSM_LAYOUT_V2_TILE_AVX512: usize = 16;
const SSM_LAYOUT_V2_TILE_AVX2: usize = 8;
const SSM_LAYOUT_V2_LANE_STRIDE: usize = 16;

fn ssm_layout_v2_tile(dispatch: CpuDispatch) -> usize {
    match dispatch {
        CpuDispatch::Avx512 => SSM_LAYOUT_V2_TILE_AVX512,
        CpuDispatch::Avx2 => SSM_LAYOUT_V2_TILE_AVX2,
        CpuDispatch::Scalar => 0,
    }
}

fn ssm_layout_v2_panel_enabled() -> bool {
    let want = std::env::var("RISK_MAMBA_SSM_LAYOUT").ok();
    matches!(want.as_deref(), Some("v2p") | Some("panel") | Some("2"))
}

fn ssm_layout_v2_lane_stride(dispatch: CpuDispatch, panel: bool) -> usize {
    if !panel {
        return ssm_layout_v2_tile(dispatch);
    }
    match dispatch {
        CpuDispatch::Avx512 => SSM_LAYOUT_V2_LANE_STRIDE,
        CpuDispatch::Avx2 => SSM_LAYOUT_V2_LANE_STRIDE,
        CpuDispatch::Scalar => 0,
    }
}

fn ssm_layout_v2_enabled(
    cfg: &ModelConfig,
    math_backend: MathBackend,
    has_dbg: bool,
    dispatch: CpuDispatch,
) -> bool {
    if cfg.seq_len == 1 && !allow_latency_experiments() {
        return false;
    }
    if has_dbg {
        return false;
    }
    let tile = ssm_layout_v2_tile(dispatch);
    if tile == 0 || cfg.d_inner % tile != 0 {
        return false;
    }
    let fast_math = matches!(
        math_backend,
        MathBackend::Approx | MathBackend::Sleef | MathBackend::FastBf16 | MathBackend::FastWild
    );
    if !fast_math {
        return false;
    }
    let want = std::env::var("RISK_MAMBA_SSM_LAYOUT").ok();
    match want.as_deref() {
        Some("v2") | Some("aosoa") | Some("aoSoA") | Some("1") | Some("v2p") | Some("panel") | Some("2") => true,
        Some("v1") | Some("0") => false,
        Some(_) => false,
        None => math_backend == MathBackend::FastWild,
    }
}

pub fn kernel_path_tag(cfg: &ModelConfig) -> &'static str {
    if cfg.seq_len == 1 {
        "latency"
    } else {
        "throughput"
    }
}

pub fn layout_tag(
    cfg: &ModelConfig,
    math_backend: MathBackend,
    has_dbg: bool,
    dispatch: CpuDispatch,
) -> &'static str {
    if ssm_layout_v2_enabled(cfg, math_backend, has_dbg, dispatch) && ssm_layout_v2_panel_enabled() {
        "v2p"
    } else {
        "v1"
    }
}

pub fn mlp_fused_enabled(seq_len: usize, has_dbg: bool, math_backend: MathBackend) -> bool {
    if has_dbg {
        return false;
    }
    if seq_len == 1 && !allow_latency_experiments() {
        return false;
    }
    if seq_len > 4 {
        return false;
    }
    if !matches!(
        math_backend,
        MathBackend::FastWild | MathBackend::Approx | MathBackend::FastBf16
    ) {
        return false;
    }
    std::env::var("RISK_MAMBA_MLP_FUSED")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(false)
}

fn audit_once(bit: u64, msg: &str) {
    if !audit_enabled() {
        return;
    }
    if AUDIT_ONCE.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
        eprintln!("[kernel-audit] {msg}");
    }
}

fn audit_trace(op_name: &str, chosen_impl: &str, reason: &str) {
    if !audit_enabled() {
        return;
    }
    let map = AUDIT_TRACE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap();
    let key = format!("{op_name}\t{chosen_impl}\t{reason}");
    *guard.entry(key).or_insert(0) += 1;
}

pub fn audit_trace_reset() {
    if let Some(map) = AUDIT_TRACE.get() {
        map.lock().unwrap().clear();
    }
}

pub fn audit_trace_dump_tsv() -> String {
    let mut out = String::from("op_name\tchosen_impl\treason\tcount\n");
    if let Some(map) = AUDIT_TRACE.get() {
        let guard = map.lock().unwrap();
        let mut rows: Vec<_> = guard.iter().collect();
        rows.sort_by(|(a, _), (b, _)| a.cmp(b));
        for (key, count) in rows {
            out.push_str(key);
            out.push('\t');
            out.push_str(&count.to_string());
            out.push('\n');
        }
    }
    out
}

#[track_caller]
#[inline(never)]
fn fallback_or_panic(msg: &str) {
    if no_fallback() {
        panic!("kernel fallback: {msg}");
    }
    audit_once(AUDIT_FALLBACK, msg);
}

pub trait DbgTaps {
    fn tap_f32(&mut self, key: &str, shape: &[usize], data: &[f32]) -> Result<(), KernelError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Ln1,
    InProj,
    Conv1dStep1,
    XProj,
    DtProjSoftplus,
    SsmUpdate,
    OutProj,
    Resid1,
    Ln2,
    MlpFc1,
    Gelu,
    MlpFc2,
    Resid2,
    FinalNorm,
    HeadL2Norm,
    HeadMatmul,
}

#[derive(Debug, Clone, Default)]
pub struct StageStats {
    pub ln1: u64,
    pub in_proj: u64,
    pub conv1d_step1: u64,
    pub x_proj: u64,
    pub dt_proj_softplus: u64,
    pub ssm_update: u64,
    pub out_proj: u64,
    pub resid1: u64,
    pub ln2: u64,
    pub mlp_fc1: u64,
    pub gelu: u64,
    pub mlp_fc2: u64,
    pub resid2: u64,
    pub final_norm: u64,
    pub head_l2norm: u64,
    pub head_matmul: u64,
}

#[derive(Debug, Default, Clone)]
pub struct PerfCounters {
    pub mlp_fc1_calls: u64,
    pub mlp_fc1_layers: u64,
    pub mlp_fc1_impl_bits: u8,
    pub mlp_fc1_m: usize,
    pub mlp_fc1_k: usize,
    pub mlp_fc1_n: usize,
}

impl PerfCounters {
    fn add_fc1(&mut self, m: usize, k: usize, n: usize, impl_bit: u8) {
        self.mlp_fc1_calls += m as u64;
        self.mlp_fc1_layers += 1;
        self.mlp_fc1_impl_bits |= impl_bit;
        self.mlp_fc1_m = m;
        self.mlp_fc1_k = k;
        self.mlp_fc1_n = n;
    }
}

const FC1_IMPL_PACKED: u8 = 1 << 0;
const FC1_IMPL_UNPACKED: u8 = 1 << 1;
const FC1_IMPL_FUSED: u8 = 1 << 2;

impl StageStats {
    pub fn clear(&mut self) {
        *self = StageStats::default();
    }

    pub fn merge(&mut self, other: &StageStats) {
        self.ln1 += other.ln1;
        self.in_proj += other.in_proj;
        self.conv1d_step1 += other.conv1d_step1;
        self.x_proj += other.x_proj;
        self.dt_proj_softplus += other.dt_proj_softplus;
        self.ssm_update += other.ssm_update;
        self.out_proj += other.out_proj;
        self.resid1 += other.resid1;
        self.ln2 += other.ln2;
        self.mlp_fc1 += other.mlp_fc1;
        self.gelu += other.gelu;
        self.mlp_fc2 += other.mlp_fc2;
        self.resid2 += other.resid2;
        self.final_norm += other.final_norm;
        self.head_l2norm += other.head_l2norm;
        self.head_matmul += other.head_matmul;
    }

    pub fn add(&mut self, stage: Stage, cycles: u64) {
        match stage {
            Stage::Ln1 => self.ln1 += cycles,
            Stage::InProj => self.in_proj += cycles,
            Stage::Conv1dStep1 => self.conv1d_step1 += cycles,
            Stage::XProj => self.x_proj += cycles,
            Stage::DtProjSoftplus => self.dt_proj_softplus += cycles,
            Stage::SsmUpdate => self.ssm_update += cycles,
            Stage::OutProj => self.out_proj += cycles,
            Stage::Resid1 => self.resid1 += cycles,
            Stage::Ln2 => self.ln2 += cycles,
            Stage::MlpFc1 => self.mlp_fc1 += cycles,
            Stage::Gelu => self.gelu += cycles,
            Stage::MlpFc2 => self.mlp_fc2 += cycles,
            Stage::Resid2 => self.resid2 += cycles,
            Stage::FinalNorm => self.final_norm += cycles,
            Stage::HeadL2Norm => self.head_l2norm += cycles,
            Stage::HeadMatmul => self.head_matmul += cycles,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FastMathStats {
    pub exp_samples: u64,
    pub exp_max_abs: f32,
    pub exp_max_rel: f32,
    pub exp_min_input: f32,
    pub exp_max_input: f32,
    pub exp_max_samples: u64,
    pub exp_abs_errors: Vec<f32>,
    pub exp_rel_errors_masked: Vec<f32>,
    pub exp_underflow_normal: u64,
    pub exp_underflow_subnormal: u64,
    pub exp_hist_counts: [u64; EXP_HIST_BUCKETS],
}

pub const EXP_HIST_EDGES: [f32; 7] = [-500.0, -200.0, -120.0, -104.0, -90.0, -60.0, 0.0];
const EXP_HIST_BUCKETS: usize = EXP_HIST_EDGES.len() - 1;

impl FastMathStats {
    pub fn new(max_samples: u64) -> Self {
        let cap = max_samples as usize;
        Self {
            exp_samples: 0,
            exp_max_abs: 0.0,
            exp_max_rel: 0.0,
            exp_min_input: f32::INFINITY,
            exp_max_input: f32::NEG_INFINITY,
            exp_max_samples: max_samples,
            exp_abs_errors: Vec::with_capacity(cap),
            exp_rel_errors_masked: Vec::with_capacity(cap),
            exp_underflow_normal: 0,
            exp_underflow_subnormal: 0,
            exp_hist_counts: [0u64; EXP_HIST_BUCKETS],
        }
    }

    pub fn clear(&mut self) {
        let max = self.exp_max_samples;
        *self = Self::new(max);
    }

    pub fn record_exp(&mut self, input: f32, approx: f32, exact: f32) {
        if self.exp_samples >= self.exp_max_samples {
            return;
        }
        self.exp_samples += 1;
        if input < self.exp_min_input {
            self.exp_min_input = input;
        }
        if input > self.exp_max_input {
            self.exp_max_input = input;
        }
        let diff = (approx - exact).abs();
        let rel = if exact == 0.0 { diff } else { diff / exact.abs() };
        if diff > self.exp_max_abs {
            self.exp_max_abs = diff;
        }
        if rel > self.exp_max_rel {
            self.exp_max_rel = rel;
        }
        self.exp_abs_errors.push(diff);
        if exact >= f32::MIN_POSITIVE {
            self.exp_rel_errors_masked.push(rel);
        }
        if exact < f32::MIN_POSITIVE {
            self.exp_underflow_normal += 1;
        }
        if exact < f32::from_bits(1) || exact == 0.0 {
            self.exp_underflow_subnormal += 1;
        }
        let idx = exp_hist_index(input);
        self.exp_hist_counts[idx] += 1;
    }
}

fn exp_hist_index(x: f32) -> usize {
    if x < EXP_HIST_EDGES[0] {
        return 0;
    }
    if x >= EXP_HIST_EDGES[EXP_HIST_EDGES.len() - 1] {
        return EXP_HIST_BUCKETS - 1;
    }
    for i in 0..EXP_HIST_BUCKETS {
        if x >= EXP_HIST_EDGES[i] && x < EXP_HIST_EDGES[i + 1] {
            return i;
        }
    }
    EXP_HIST_BUCKETS - 1
}

#[inline]
fn time_stamp() -> u64 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_lfence, _rdtsc};
            _mm_lfence();
            let t = _rdtsc() as u64;
            _mm_lfence();
            t
        }
        #[cfg(target_arch = "x86")]
        {
            use std::arch::x86::{_mm_lfence, _rdtsc};
            _mm_lfence();
            let t = _rdtsc() as u64;
            _mm_lfence();
            t
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        0
    }
}

fn record_stage<F: FnOnce()>(stats: &mut Option<&mut StageStats>, stage: Stage, f: F) {
    if stats.is_none() {
        f();
        return;
    }
    let rate = stage_sample_rate();
    if rate == 0 {
        f();
        return;
    }
    let idx = STAGE_SAMPLE_COUNTER.fetch_add(1, Ordering::Relaxed);
    if idx % rate != 0 {
        f();
        return;
    }
    let start = time_stamp();
    f();
    let end = time_stamp();
    if let Some(s) = stats.as_deref_mut() {
        s.add(stage, end.saturating_sub(start));
    }
}

fn expect_len(actual: usize, expected: usize, what: &'static str) -> Result<(), KernelError> {
    if actual != expected {
        return Err(KernelError::BadLen(what));
    }
    Ok(())
}

fn expect_weights<'a>(opt: Option<&'a [f32]>, what: &'static str) -> Result<&'a [f32], KernelError> {
    opt.ok_or(KernelError::MissingWeights(what))
}

fn slice_batched<'a>(buf: &'a [f32], _batch: usize, width: usize, b: usize) -> &'a [f32] {
    let start = b * width;
    &buf[start..start + width]
}

fn slice_batched_mut<'a>(buf: &'a mut [f32], _batch: usize, width: usize, b: usize) -> &'a mut [f32] {
    let start = b * width;
    &mut buf[start..start + width]
}

fn slice_seq<'a>(buf: &'a [f32], width: usize, t: usize) -> &'a [f32] {
    let start = t * width;
    &buf[start..start + width]
}

fn slice_seq_mut<'a>(buf: &'a mut [f32], width: usize, t: usize) -> &'a mut [f32] {
    let start = t * width;
    &mut buf[start..start + width]
}

fn layer_state_slice<'a>(cfg: &ModelConfig, batch: usize, layer: usize, state: &'a mut [f32]) -> &'a mut [f32] {
    let layer_stride = batch * cfg.d_inner * cfg.d_state_pad;
    let start = layer * layer_stride;
    &mut state[start..start + layer_stride]
}

fn check_dispatch(dispatch: CpuDispatch) -> Result<(), KernelError> {
    match dispatch {
        CpuDispatch::Scalar => Ok(()),
        CpuDispatch::Avx2 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if !std::is_x86_feature_detected!("avx2")
                    || !std::is_x86_feature_detected!("fma")
                {
                    return Err(KernelError::CpuFeatureMissing("avx2+fma"));
                }
                Ok(())
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                Err(KernelError::CpuFeatureMissing("avx2"))
            }
        }
        CpuDispatch::Avx512 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if !std::is_x86_feature_detected!("avx512f")
                    || !std::is_x86_feature_detected!("fma")
                {
                    return Err(KernelError::CpuFeatureMissing("avx512f+fma"));
                }
                Ok(())
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                Err(KernelError::CpuFeatureMissing("avx512f"))
            }
        }
    }
}

fn matmul_proj(
    weight: &[f32],
    weight_packed: Option<&[f32]>,
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: Option<&[f32]>,
    bias_zero: Option<&[f32]>,
    output: &mut [f32],
    dispatch: CpuDispatch,
) {
    if dispatch != CpuDispatch::Scalar {
        if let Some(packed) = weight_packed {
            if let Some(bias_slice) = bias.or(bias_zero) {
                let packed_out = packed16_out_dim_f32(packed.len(), in_dim);
                audit_trace(
                    "matmul_proj",
                    match dispatch {
                        CpuDispatch::Avx2 => "avx2",
                        CpuDispatch::Avx512 => "avx512",
                        CpuDispatch::Scalar => "scalar",
                    },
                    if packed_out == out_dim { "packed16" } else { "packed16_tail" },
                );
                audit_once(AUDIT_MATMUL, "matmul_proj: packed16");
                if packed_out >= out_dim && packed_out > 0 {
                    let block = 16usize;
                    let full_out = (out_dim / block) * block;
                    if full_out > 0 {
                        matmul_packed16_dispatch(
                            packed,
                            full_out,
                            in_dim,
                            input,
                            bias_slice,
                            &mut output[..full_out],
                            dispatch,
                        );
                    }
                    if full_out < out_dim {
                        if packed_out < full_out + block {
                            let msg = format!(
                                "matmul_proj: packed tail missing (packed_out={}, out_dim={}, in_dim={})",
                                packed_out, out_dim, in_dim
                            );
                            fallback_or_panic(&format!("E2003 {msg}"));
                        } else {
                            let tail_out = out_dim - full_out;
                            let tail_base = full_out * in_dim;
                            let tail_packed = &packed[tail_base..tail_base + in_dim * block];
                            let tail_bias = &bias_slice[full_out..];
                            matmul_packed16_tail_dispatch(
                                tail_packed,
                                in_dim,
                                input,
                                tail_bias,
                                tail_out,
                                &mut output[full_out..],
                                dispatch,
                            );
                            return;
                        }
                    } else {
                        return;
                    }
                }
                if packed_out > 0 && packed_out < out_dim {
                    let msg = format!(
                        "matmul_proj: packed path incomplete (packed_out={}, out_dim={}, in_dim={})",
                        packed_out, out_dim, in_dim
                    );
                    fallback_or_panic(&format!("E2003 {msg}"));
                }
            }
        } else if out_dim % 16 != 0 {
            audit_trace(
                "matmul_proj",
                match dispatch {
                    CpuDispatch::Avx2 => "avx2",
                    CpuDispatch::Avx512 => "avx512",
                    CpuDispatch::Scalar => "scalar",
                },
                "unpacked_outdim",
            );
            audit_once(AUDIT_MATMUL, "matmul_proj: unpacked (out_dim not multiple of 16)");
            let msg = format!(
                "matmul_proj: packed path unavailable (out_dim={}, in_dim={})",
                out_dim, in_dim
            );
            fallback_or_panic(&format!("E2001 {msg}"));
        } else {
            let msg = format!(
                "matmul_proj: packed path unavailable (out_dim={}, in_dim={})",
                out_dim, in_dim
            );
            fallback_or_panic(&format!("E2001 {msg}"));
        }
    }
    audit_trace(
        "matmul_proj",
        match dispatch {
            CpuDispatch::Avx2 => "avx2",
            CpuDispatch::Avx512 => "avx512",
            CpuDispatch::Scalar => "scalar",
        },
        "unpacked",
    );
    matmul_vec_dispatch(weight, out_dim, in_dim, input, bias, output, dispatch);
}

fn matmul_proj_batch(
    weight: &[f32],
    weight_packed: Option<&[f32]>,
    weight_packed_bf16: Option<&[u16]>,
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: Option<&[f32]>,
    bias_zero: Option<&[f32]>,
    seq_len: usize,
    output: &mut [f32],
    dispatch: CpuDispatch,
    math_backend: MathBackend,
) {
    if seq_len == 1 {
        matmul_proj(
            weight,
            weight_packed,
            out_dim,
            in_dim,
            input,
            bias,
            bias_zero,
            output,
            dispatch,
        );
        return;
    }

    if dispatch != CpuDispatch::Scalar && seq_len <= 4 {
        if let Some(packed) = weight_packed {
            if let Some(bias_slice) = bias.or(bias_zero) {
                let packed_out = packed16_out_dim_f32(packed.len(), in_dim);
                audit_trace(
                    "matmul_proj_batch",
                    match dispatch {
                        CpuDispatch::Avx2 => "avx2",
                        CpuDispatch::Avx512 => "avx512",
                        CpuDispatch::Scalar => "scalar",
                    },
                    if packed_out == out_dim { "m1_seq" } else { "m1_seq_tail" },
                );
                if packed_out >= out_dim && packed_out > 0 {
                    let block = 16usize;
                    let full_out = (out_dim / block) * block;
                    if full_out < out_dim && packed_out < full_out + block {
                        let msg = format!(
                            "matmul_proj_batch: packed tail missing (packed_out={}, out_dim={}, in_dim={})",
                            packed_out, out_dim, in_dim
                        );
                        fallback_or_panic(&format!("E2004 {msg}"));
                    } else {
                        let tail_out = out_dim - full_out;
                        let tail_base = full_out * in_dim;
                        let tail_packed = if tail_out > 0 {
                            Some(&packed[tail_base..tail_base + in_dim * block])
                        } else {
                            None
                        };
                        let tail_bias = &bias_slice[full_out..];
                        for t in 0..seq_len {
                            let in_t = &input[t * in_dim..][..in_dim];
                            let out_t = &mut output[t * out_dim..][..out_dim];
                            if full_out > 0 {
                                matmul_packed16_dispatch(
                                    packed,
                                    full_out,
                                    in_dim,
                                    in_t,
                                    bias_slice,
                                    &mut out_t[..full_out],
                                    dispatch,
                                );
                            }
                            if let Some(tail_packed) = tail_packed {
                                matmul_packed16_tail_dispatch(
                                    tail_packed,
                                    in_dim,
                                    in_t,
                                    tail_bias,
                                    tail_out,
                                    &mut out_t[full_out..],
                                    dispatch,
                                );
                            }
                        }
                        return;
                    }
                }
                if packed_out > 0 && packed_out < out_dim {
                    let msg = format!(
                        "matmul_proj_batch: packed path incomplete (packed_out={}, out_dim={}, in_dim={})",
                        packed_out, out_dim, in_dim
                    );
                    fallback_or_panic(&format!("E2004 {msg}"));
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    if dispatch == CpuDispatch::Avx512 && matches!(math_backend, MathBackend::FastBf16) {
        if let Some(packed_bf16) = weight_packed_bf16 {
            if let Some(bias_slice) = bias.or(bias_zero) {
                if std::is_x86_feature_detected!("avx512bf16") {
                    let packed_out = packed16_out_dim_bf16(packed_bf16.len(), in_dim);
                    if packed_out != out_dim {
                        // fall through to f32 packed path
                    } else {
                    unsafe {
                        audit_trace("matmul_proj_batch", "avx512", "packed_bf16");
                        audit_once(AUDIT_MATMUL, "matmul_proj_batch: avx512 bf16 packed16");
                        matmul_packed16_batch_avx512_bf16(
                            packed_bf16,
                            out_dim,
                            in_dim,
                            input,
                            seq_len,
                            bias_slice,
                            output,
                        );
                    }
                    return;
                    }
                }
            }
        }
    }

    if dispatch != CpuDispatch::Scalar {
        if let Some(packed) = weight_packed {
            if let Some(bias_slice) = bias.or(bias_zero) {
                let packed_out = packed16_out_dim_f32(packed.len(), in_dim);
                audit_trace(
                    "matmul_proj_batch",
                    match dispatch {
                        CpuDispatch::Avx2 => "avx2",
                        CpuDispatch::Avx512 => "avx512",
                        CpuDispatch::Scalar => "scalar",
                    },
                    if packed_out == out_dim { "packed16" } else { "packed16_tail" },
                );
                audit_once(AUDIT_MATMUL, "matmul_proj_batch: packed16");
                if packed_out >= out_dim && packed_out > 0 {
                    let block = 16usize;
                    let full_out = (out_dim / block) * block;
                    if full_out < out_dim && packed_out < full_out + block {
                        let msg = format!(
                            "matmul_proj_batch: packed tail missing (packed_out={}, out_dim={}, in_dim={})",
                            packed_out, out_dim, in_dim
                        );
                        fallback_or_panic(&format!("E2005 {msg}"));
                    } else {
                        if full_out > 0 {
                            matmul_packed16_batch_dispatch(
                                packed,
                                full_out,
                                in_dim,
                                input,
                                seq_len,
                                bias_slice,
                                output,
                                dispatch,
                            );
                        }
                        if full_out < out_dim {
                            let tail_out = out_dim - full_out;
                            let tail_base = full_out * in_dim;
                            let tail_packed = &packed[tail_base..tail_base + in_dim * block];
                            let tail_bias = &bias_slice[full_out..];
                            for t in 0..seq_len {
                                let in_t = &input[t * in_dim..][..in_dim];
                                let out_t = &mut output[t * out_dim..][..out_dim];
                                matmul_packed16_tail_dispatch(
                                    tail_packed,
                                    in_dim,
                                    in_t,
                                    tail_bias,
                                    tail_out,
                                    &mut out_t[full_out..],
                                    dispatch,
                                );
                            }
                        }
                        return;
                    }
                }
                if packed_out > 0 && packed_out < out_dim {
                    let msg = format!(
                        "matmul_proj_batch: packed path incomplete (packed_out={}, out_dim={}, in_dim={})",
                        packed_out, out_dim, in_dim
                    );
                    fallback_or_panic(&format!("E2005 {msg}"));
                }
            }
        } else if out_dim % 16 != 0 {
            audit_trace(
                "matmul_proj_batch",
                match dispatch {
                    CpuDispatch::Avx2 => "avx2",
                    CpuDispatch::Avx512 => "avx512",
                    CpuDispatch::Scalar => "scalar",
                },
                "unpacked_outdim",
            );
            audit_once(AUDIT_MATMUL, "matmul_proj_batch: unpacked (out_dim not multiple of 16)");
            let msg = format!(
                "matmul_proj_batch: packed path unavailable (out_dim={}, in_dim={}, seq_len={})",
                out_dim, in_dim, seq_len
            );
            fallback_or_panic(&format!("E2002 {msg}"));
        } else {
            let msg = format!(
                "matmul_proj_batch: packed path unavailable (out_dim={}, in_dim={}, seq_len={})",
                out_dim, in_dim, seq_len
            );
            fallback_or_panic(&format!("E2002 {msg}"));
        }
    }

    audit_trace(
        "matmul_proj_batch",
        match dispatch {
            CpuDispatch::Avx2 => "avx2",
            CpuDispatch::Avx512 => "avx512",
            CpuDispatch::Scalar => "scalar",
        },
        "unpacked",
    );
    for t in 0..seq_len {
        let in_t = &input[t * in_dim..][..in_dim];
        let out_t = &mut output[t * out_dim..][..out_dim];
        matmul_vec_dispatch(weight, out_dim, in_dim, in_t, bias, out_t, dispatch);
    }
}

fn fill_a_pre(a_log: &[f32], a_pre: &mut [f32]) {
    for (dst, src) in a_pre.iter_mut().zip(a_log.iter()) {
        *dst = -libm::expf(*src);
    }
}

fn fill_a_pre_v2(cfg: &ModelConfig, tile: usize, lane_stride: usize, a_log: &[f32], a_pre: &mut [f32]) {
    let tiles = cfg.d_inner / tile;
    let d_state_pad = cfg.d_state_pad;
    for ti in 0..tiles {
        let base_i = ti * tile;
        for j in 0..d_state_pad {
            let dst_base = (ti * d_state_pad + j) * lane_stride;
            let src_base = (base_i * d_state_pad) + j;
            for lane in 0..tile {
                let src = a_log[src_base + lane * d_state_pad];
                a_pre[dst_base + lane] = -libm::expf(src);
            }
            for lane in tile..lane_stride {
                a_pre[dst_base + lane] = 0.0;
            }
        }
    }
}

fn reorder_a_pre_v2_from_v1(
    cfg: &ModelConfig,
    tile: usize,
    lane_stride: usize,
    src: &[f32],
    dst: &mut [f32],
) {
    let tiles = cfg.d_inner / tile;
    let d_state_pad = cfg.d_state_pad;
    for ti in 0..tiles {
        let base_i = ti * tile;
        for j in 0..d_state_pad {
            let dst_base = (ti * d_state_pad + j) * lane_stride;
            let src_base = (base_i * d_state_pad) + j;
            for lane in 0..tile {
                dst[dst_base + lane] = src[src_base + lane * d_state_pad];
            }
            for lane in tile..lane_stride {
                dst[dst_base + lane] = 0.0;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn ssm_update_avx512_full(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    conv_out: &[f32],
    z: &[f32],
    delta: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    x_proj_out: &[f32],
    x_proj_dim: usize,
    dt_rank: usize,
    use_xproj_bc: bool,
    a_pre: &[f32],
    scan_out: &mut [f32],
    layer_state: &mut [f32],
    math_backend: MathBackend,
) {
    use std::arch::x86_64::*;
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let seq_len = cfg.seq_len;
    let prefetch_dist = if d_state >= 64 { 32 } else { 0 };
    for i in 0..d_inner {
        let state_row = &mut layer_state[i * d_state_pad..(i + 1) * d_state_pad];
        let a_row = &a_pre[i * d_state_pad..(i + 1) * d_state_pad];
        for t in 0..seq_len {
            let x_i = conv_out[t * d_inner + i];
            let z_i = z[t * d_inner + i];
            let delta_i = delta[t * d_inner + i];
            let (b_vec_t, c_vec_t) = if use_xproj_bc {
                let x_proj_t = &x_proj_out[t * x_proj_dim..][..x_proj_dim];
                (
                    &x_proj_t[dt_rank..dt_rank + d_state_pad],
                    &x_proj_t[dt_rank + d_state_pad..dt_rank + 2 * d_state_pad],
                )
            } else {
                (
                    &b_vec[t * d_state_pad..(t + 1) * d_state_pad],
                    &c_vec[t * d_state_pad..(t + 1) * d_state_pad],
                )
            };
            let scan_t = &mut scan_out[t * d_inner..(t + 1) * d_inner];

            let x_v = _mm512_set1_ps(x_i);
            let delta_v = _mm512_set1_ps(delta_i);
            let mut acc_v = _mm512_setzero_ps();
            let mut j = 0usize;
            while j + 32 <= d_state {
                if prefetch_dist != 0 && j + prefetch_dist < d_state {
                    _mm_prefetch(state_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(b_vec_t.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(c_vec_t.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                }

                let a0 = _mm512_loadu_ps(a_row.as_ptr().add(j));
                let x0 = _mm512_mul_ps(delta_v, a0);
                let a_bar0 = exp512_ps(x0);
                let h0 = _mm512_loadu_ps(state_row.as_ptr().add(j));
                let b0 = _mm512_loadu_ps(b_vec_t.as_ptr().add(j));
                let c0 = _mm512_loadu_ps(c_vec_t.as_ptr().add(j));
                let b_bar0 = _mm512_mul_ps(b0, delta_v);
                let h_new0 = _mm512_fmadd_ps(a_bar0, h0, _mm512_mul_ps(b_bar0, x_v));
                _mm512_storeu_ps(state_row.as_mut_ptr().add(j), h_new0);
                acc_v = _mm512_fmadd_ps(c0, h_new0, acc_v);

                let j1 = j + 16;
                let a1 = _mm512_loadu_ps(a_row.as_ptr().add(j1));
                let x1 = _mm512_mul_ps(delta_v, a1);
                let a_bar1 = exp512_ps(x1);
                let h1 = _mm512_loadu_ps(state_row.as_ptr().add(j1));
                let b1 = _mm512_loadu_ps(b_vec_t.as_ptr().add(j1));
                let c1 = _mm512_loadu_ps(c_vec_t.as_ptr().add(j1));
                let b_bar1 = _mm512_mul_ps(b1, delta_v);
                let h_new1 = _mm512_fmadd_ps(a_bar1, h1, _mm512_mul_ps(b_bar1, x_v));
                _mm512_storeu_ps(state_row.as_mut_ptr().add(j1), h_new1);
                acc_v = _mm512_fmadd_ps(c1, h_new1, acc_v);

                j += 32;
            }
            while j + 16 <= d_state {
                if prefetch_dist != 0 && j + prefetch_dist < d_state {
                    _mm_prefetch(state_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                }
                let a = _mm512_loadu_ps(a_row.as_ptr().add(j));
                let x = _mm512_mul_ps(delta_v, a);
                let a_bar = exp512_ps(x);
                let h = _mm512_loadu_ps(state_row.as_ptr().add(j));
                let b_val = _mm512_loadu_ps(b_vec_t.as_ptr().add(j));
                let c_val = _mm512_loadu_ps(c_vec_t.as_ptr().add(j));
                let b_bar = _mm512_mul_ps(b_val, delta_v);
                let h_new = _mm512_fmadd_ps(a_bar, h, _mm512_mul_ps(b_bar, x_v));
                _mm512_storeu_ps(state_row.as_mut_ptr().add(j), h_new);
                acc_v = _mm512_fmadd_ps(c_val, h_new, acc_v);
                j += 16;
            }
            let mut acc = hsum512_ps(acc_v);
            while j < d_state {
                let a = a_row[j];
                let a_bar = exp_approx_scalar(delta_i * a);
                let b_bar = b_vec_t[j] * delta_i;
                let h = state_row[j];
                let h_new = a_bar * h + b_bar * x_i;
                state_row[j] = h_new;
                acc += c_vec_t[j] * h_new;
                j += 1;
            }
            let mut y = acc + lw.d[i] * x_i;
            y *= silu_dispatch(z_i, math_backend);
            scan_t[i] = y;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn ssm_update_avx2_full(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    conv_out: &[f32],
    z: &[f32],
    delta: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    x_proj_out: &[f32],
    x_proj_dim: usize,
    dt_rank: usize,
    use_xproj_bc: bool,
    a_pre: &[f32],
    scan_out: &mut [f32],
    layer_state: &mut [f32],
    math_backend: MathBackend,
) {
    use std::arch::x86_64::*;
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let seq_len = cfg.seq_len;
    let prefetch_dist = if d_state >= 32 { 16 } else { 0 };
    for i in 0..d_inner {
        let state_row = &mut layer_state[i * d_state_pad..(i + 1) * d_state_pad];
        let a_row = &a_pre[i * d_state_pad..(i + 1) * d_state_pad];
        for t in 0..seq_len {
            let x_i = conv_out[t * d_inner + i];
            let z_i = z[t * d_inner + i];
            let delta_i = delta[t * d_inner + i];
            let (b_vec_t, c_vec_t) = if use_xproj_bc {
                let x_proj_t = &x_proj_out[t * x_proj_dim..][..x_proj_dim];
                (
                    &x_proj_t[dt_rank..dt_rank + d_state_pad],
                    &x_proj_t[dt_rank + d_state_pad..dt_rank + 2 * d_state_pad],
                )
            } else {
                (
                    &b_vec[t * d_state_pad..(t + 1) * d_state_pad],
                    &c_vec[t * d_state_pad..(t + 1) * d_state_pad],
                )
            };
            let scan_t = &mut scan_out[t * d_inner..(t + 1) * d_inner];

            let x_v = _mm256_set1_ps(x_i);
            let delta_v = _mm256_set1_ps(delta_i);
            let mut acc_v = _mm256_setzero_ps();
            let mut j = 0usize;
            while j + 16 <= d_state {
                if prefetch_dist != 0 && j + prefetch_dist < d_state {
                    _mm_prefetch(state_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(b_vec_t.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(c_vec_t.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                }

                let a0 = _mm256_loadu_ps(a_row.as_ptr().add(j));
                let x0 = _mm256_mul_ps(delta_v, a0);
                let a_bar0 = exp256_ps(x0);
                let h0 = _mm256_loadu_ps(state_row.as_ptr().add(j));
                let b0 = _mm256_loadu_ps(b_vec_t.as_ptr().add(j));
                let c0 = _mm256_loadu_ps(c_vec_t.as_ptr().add(j));
                let b_bar0 = _mm256_mul_ps(b0, delta_v);
                let h_new0 = _mm256_fmadd_ps(a_bar0, h0, _mm256_mul_ps(b_bar0, x_v));
                _mm256_storeu_ps(state_row.as_mut_ptr().add(j), h_new0);
                acc_v = _mm256_fmadd_ps(c0, h_new0, acc_v);

                let j1 = j + 8;
                let a1 = _mm256_loadu_ps(a_row.as_ptr().add(j1));
                let x1 = _mm256_mul_ps(delta_v, a1);
                let a_bar1 = exp256_ps(x1);
                let h1 = _mm256_loadu_ps(state_row.as_ptr().add(j1));
                let b1 = _mm256_loadu_ps(b_vec_t.as_ptr().add(j1));
                let c1 = _mm256_loadu_ps(c_vec_t.as_ptr().add(j1));
                let b_bar1 = _mm256_mul_ps(b1, delta_v);
                let h_new1 = _mm256_fmadd_ps(a_bar1, h1, _mm256_mul_ps(b_bar1, x_v));
                _mm256_storeu_ps(state_row.as_mut_ptr().add(j1), h_new1);
                acc_v = _mm256_fmadd_ps(c1, h_new1, acc_v);

                j += 16;
            }
            while j + 8 <= d_state {
                if prefetch_dist != 0 && j + prefetch_dist < d_state {
                    _mm_prefetch(state_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                }
                let a = _mm256_loadu_ps(a_row.as_ptr().add(j));
                let x = _mm256_mul_ps(delta_v, a);
                let a_bar = exp256_ps(x);
                let h = _mm256_loadu_ps(state_row.as_ptr().add(j));
                let b_val = _mm256_loadu_ps(b_vec_t.as_ptr().add(j));
                let c_val = _mm256_loadu_ps(c_vec_t.as_ptr().add(j));
                let b_bar = _mm256_mul_ps(b_val, delta_v);
                let h_new = _mm256_fmadd_ps(a_bar, h, _mm256_mul_ps(b_bar, x_v));
                _mm256_storeu_ps(state_row.as_mut_ptr().add(j), h_new);
                acc_v = _mm256_fmadd_ps(c_val, h_new, acc_v);
                j += 8;
            }
            let mut acc = hsum256_ps(acc_v);
            while j < d_state {
                let a = a_row[j];
                let a_bar = exp_approx_scalar(delta_i * a);
                let b_bar = b_vec_t[j] * delta_i;
                let h = state_row[j];
                let h_new = a_bar * h + b_bar * x_i;
                state_row[j] = h_new;
                acc += c_vec_t[j] * h_new;
                j += 1;
            }
            let mut y = acc + lw.d[i] * x_i;
            y *= silu_dispatch(z_i, math_backend);
            scan_t[i] = y;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn ssm_update_avx512_full_v2(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    conv_out: &[f32],
    z: &[f32],
    delta: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    x_proj_out: &[f32],
    x_proj_dim: usize,
    dt_rank: usize,
    use_xproj_bc: bool,
    a_pre: &[f32],
    lane_stride: usize,
    scan_out: &mut [f32],
    layer_state: &mut [f32],
) {
    use std::arch::x86_64::*;
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let seq_len = cfg.seq_len;
    let tile = SSM_LAYOUT_V2_TILE_AVX512;
    let tiles = d_inner / tile;
    let panel = lane_stride > tile;
    let prefetch_dist = if d_state >= 32 {
        if panel { 16 } else { 8 }
    } else {
        0
    };

    for t in 0..seq_len {
        let (b_vec_t, c_vec_t) = if use_xproj_bc {
            let x_proj_t = &x_proj_out[t * x_proj_dim..][..x_proj_dim];
            (
                &x_proj_t[dt_rank..dt_rank + d_state_pad],
                &x_proj_t[dt_rank + d_state_pad..dt_rank + 2 * d_state_pad],
            )
        } else {
            (
                &b_vec[t * d_state_pad..(t + 1) * d_state_pad],
                &c_vec[t * d_state_pad..(t + 1) * d_state_pad],
            )
        };
        let scan_t = &mut scan_out[t * d_inner..(t + 1) * d_inner];
        for ti in 0..tiles {
            let base_i = ti * tile;
            let x_ptr = conv_out.as_ptr().add(t * d_inner + base_i);
            let z_ptr = z.as_ptr().add(t * d_inner + base_i);
            let delta_ptr = delta.as_ptr().add(t * d_inner + base_i);
            let x_v = _mm512_loadu_ps(x_ptr);
            let z_v = _mm512_loadu_ps(z_ptr);
            let delta_v = _mm512_loadu_ps(delta_ptr);
            let mut acc_v = _mm512_setzero_ps();
            let base_off = ti * d_state_pad * lane_stride;
            let a_base = a_pre.as_ptr().add(base_off);
            let h_base = layer_state.as_ptr().add(base_off);
            let next_base = if panel && ti + 1 < tiles {
                Some((ti + 1) * d_state_pad * lane_stride)
            } else {
                None
            };
            let mut a_ptr = a_base;
            let mut h_ptr = h_base;
            for j in 0..d_state {
                if prefetch_dist != 0 && j + prefetch_dist < d_state {
                    let pf = (j + prefetch_dist) * lane_stride;
                    _mm_prefetch(h_base.add(pf) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_base.add(pf) as *const i8, _MM_HINT_T0);
                    if let Some(next) = next_base {
                        let pf_next = next + (j + prefetch_dist) * lane_stride;
                        _mm_prefetch(layer_state.as_ptr().add(pf_next) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(a_pre.as_ptr().add(pf_next) as *const i8, _MM_HINT_T0);
                    }
                }
                debug_assert_eq!((a_ptr as usize) & 63, 0);
                debug_assert_eq!((h_ptr as usize) & 63, 0);
                let a_v = _mm512_load_ps(a_ptr);
                let h_v = _mm512_load_ps(h_ptr);
                let x = _mm512_mul_ps(delta_v, a_v);
                let a_bar = exp512_ps(x);
                let b = _mm512_set1_ps(b_vec_t[j]);
                let b_bar = _mm512_mul_ps(b, delta_v);
                let h_new = _mm512_fmadd_ps(a_bar, h_v, _mm512_mul_ps(b_bar, x_v));
                _mm512_store_ps(layer_state.as_mut_ptr().add(base_off + j * lane_stride), h_new);
                let c = _mm512_set1_ps(c_vec_t[j]);
                acc_v = _mm512_fmadd_ps(c, h_new, acc_v);
                a_ptr = a_ptr.add(lane_stride);
                h_ptr = h_ptr.add(lane_stride);
            }
            let d_v = _mm512_loadu_ps(lw.d.as_ptr().add(base_i));
            let mut y_v = _mm512_fmadd_ps(d_v, x_v, acc_v);
            let sig = sigmoid_fast_avx512(z_v);
            let silu = _mm512_mul_ps(z_v, sig);
            y_v = _mm512_mul_ps(y_v, silu);
            _mm512_storeu_ps(scan_t.as_mut_ptr().add(base_i), y_v);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn ssm_update_avx512_full_v2_fused_dt(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    conv_out: &[f32],
    z: &[f32],
    dt_in: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    x_proj_out: &[f32],
    x_proj_dim: usize,
    dt_rank: usize,
    use_xproj_bc: bool,
    a_pre: &[f32],
    lane_stride: usize,
    fast_dt: bool,
    scan_out: &mut [f32],
    layer_state: &mut [f32],
) {
    use std::arch::x86_64::*;
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let seq_len = cfg.seq_len;
    let tile = SSM_LAYOUT_V2_TILE_AVX512;
    let tiles = d_inner / tile;
    let panel = lane_stride > tile;
    let prefetch_dist = if d_state >= 32 {
        if panel { 16 } else { 8 }
    } else {
        0
    };
    let mut delta_buf = [0.0f32; SSM_LAYOUT_V2_TILE_AVX512];
    let use_fast_dt = fast_dt && dt_rank == 16 && lw.dt_proj_w_packed.is_some();

    for t in 0..seq_len {
        let dt_in_t = &dt_in[t * dt_rank..][..dt_rank];
        let (b_vec_t, c_vec_t) = if use_xproj_bc {
            let x_proj_t = &x_proj_out[t * x_proj_dim..][..x_proj_dim];
            (
                &x_proj_t[dt_rank..dt_rank + d_state_pad],
                &x_proj_t[dt_rank + d_state_pad..dt_rank + 2 * d_state_pad],
            )
        } else {
            (
                &b_vec[t * d_state_pad..(t + 1) * d_state_pad],
                &c_vec[t * d_state_pad..(t + 1) * d_state_pad],
            )
        };
        let scan_t = &mut scan_out[t * d_inner..(t + 1) * d_inner];
        for ti in 0..tiles {
            let base_i = ti * tile;
            let delta_v = if use_fast_dt {
                let packed = lw.dt_proj_w_packed.as_deref().unwrap();
                unsafe {
                    matmul_packed16_block_avx512(
                        packed,
                        dt_rank,
                        dt_in_t,
                        ti,
                        lw.dt_proj_b,
                        &mut delta_buf,
                    );
                }
                let x = _mm512_loadu_ps(delta_buf.as_ptr());
                let beta = cfg.softplus_beta;
                let inv_beta = if beta != 0.0 { 1.0 / beta } else { 1.0 };
                let threshold = cfg.softplus_threshold;
                let zero = _mm512_setzero_ps();
                let sign_mask = _mm512_set1_ps(-0.0f32);
                let beta_v = _mm512_set1_ps(beta);
                let inv_beta_v = _mm512_set1_ps(inv_beta);
                let thresh_v = _mm512_set1_ps(threshold);
                let bx = _mm512_mul_ps(x, beta_v);
                let abs = _mm512_andnot_ps(sign_mask, bx);
                let neg_abs = _mm512_sub_ps(zero, abs);
                let exp = exp512_ps(neg_abs);
                let base = _mm512_max_ps(bx, zero);
                let approx = _mm512_add_ps(base, exp);
                let mut res = _mm512_mul_ps(approx, inv_beta_v);
                let mask = _mm512_cmp_ps_mask(bx, thresh_v, _CMP_GT_OQ);
                res = _mm512_mask_blend_ps(mask, res, x);
                res
            } else {
                for lane in 0..tile {
                    let i = base_i + lane;
                    let dt_w_row = &lw.dt_proj_w[i * dt_rank..][..dt_rank];
                    let mut dot = 0.0f32;
                    for r in 0..dt_rank {
                        dot += dt_w_row[r] * dt_in_t[r];
                    }
                    dot += lw.dt_proj_b[i];
                    delta_buf[lane] = softplus_dispatch(
                        dot,
                        cfg.softplus_kind,
                        cfg.softplus_beta,
                        cfg.softplus_threshold,
                        MathBackend::Exact,
                    );
                }
                _mm512_loadu_ps(delta_buf.as_ptr())
            };
            let x_ptr = conv_out.as_ptr().add(t * d_inner + base_i);
            let z_ptr = z.as_ptr().add(t * d_inner + base_i);
            let x_v = _mm512_loadu_ps(x_ptr);
            let z_v = _mm512_loadu_ps(z_ptr);
            let mut acc_v = _mm512_setzero_ps();
            let base_off = ti * d_state_pad * lane_stride;
            let a_base = a_pre.as_ptr().add(base_off);
            let h_base = layer_state.as_ptr().add(base_off);
            let next_base = if panel && ti + 1 < tiles {
                Some((ti + 1) * d_state_pad * lane_stride)
            } else {
                None
            };
            let mut a_ptr = a_base;
            let mut h_ptr = h_base;
            for j in 0..d_state {
                if prefetch_dist != 0 && j + prefetch_dist < d_state {
                    let pf = (j + prefetch_dist) * lane_stride;
                    _mm_prefetch(h_base.add(pf) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_base.add(pf) as *const i8, _MM_HINT_T0);
                    if let Some(next) = next_base {
                        let pf_next = next + (j + prefetch_dist) * lane_stride;
                        _mm_prefetch(layer_state.as_ptr().add(pf_next) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(a_pre.as_ptr().add(pf_next) as *const i8, _MM_HINT_T0);
                    }
                }
                debug_assert_eq!((a_ptr as usize) & 63, 0);
                debug_assert_eq!((h_ptr as usize) & 63, 0);
                let a_v = _mm512_load_ps(a_ptr);
                let h_v = _mm512_load_ps(h_ptr);
                let x = _mm512_mul_ps(delta_v, a_v);
                let a_bar = exp512_ps(x);
                let b = _mm512_set1_ps(b_vec_t[j]);
                let b_bar = _mm512_mul_ps(b, delta_v);
                let h_new = _mm512_fmadd_ps(a_bar, h_v, _mm512_mul_ps(b_bar, x_v));
                _mm512_store_ps(layer_state.as_mut_ptr().add(base_off + j * lane_stride), h_new);
                let c = _mm512_set1_ps(c_vec_t[j]);
                acc_v = _mm512_fmadd_ps(c, h_new, acc_v);
                a_ptr = a_ptr.add(lane_stride);
                h_ptr = h_ptr.add(lane_stride);
            }
            let d_v = _mm512_loadu_ps(lw.d.as_ptr().add(base_i));
            let mut y_v = _mm512_fmadd_ps(d_v, x_v, acc_v);
            let sig = sigmoid_fast_avx512(z_v);
            let silu = _mm512_mul_ps(z_v, sig);
            y_v = _mm512_mul_ps(y_v, silu);
            _mm512_storeu_ps(scan_t.as_mut_ptr().add(base_i), y_v);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn ssm_update_avx2_full_v2(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    conv_out: &[f32],
    z: &[f32],
    delta: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    x_proj_out: &[f32],
    x_proj_dim: usize,
    dt_rank: usize,
    use_xproj_bc: bool,
    a_pre: &[f32],
    lane_stride: usize,
    scan_out: &mut [f32],
    layer_state: &mut [f32],
) {
    use std::arch::x86_64::*;
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let seq_len = cfg.seq_len;
    let tile = SSM_LAYOUT_V2_TILE_AVX2;
    let tiles = d_inner / tile;
    let panel = lane_stride > tile;
    let prefetch_dist = if d_state >= 32 {
        if panel { 8 } else { 4 }
    } else if d_state >= 16 {
        if panel { 8 } else { 4 }
    } else {
        0
    };

    for t in 0..seq_len {
        let (b_vec_t, c_vec_t) = if use_xproj_bc {
            let x_proj_t = &x_proj_out[t * x_proj_dim..][..x_proj_dim];
            (
                &x_proj_t[dt_rank..dt_rank + d_state_pad],
                &x_proj_t[dt_rank + d_state_pad..dt_rank + 2 * d_state_pad],
            )
        } else {
            (
                &b_vec[t * d_state_pad..(t + 1) * d_state_pad],
                &c_vec[t * d_state_pad..(t + 1) * d_state_pad],
            )
        };
        let scan_t = &mut scan_out[t * d_inner..(t + 1) * d_inner];
        for ti in 0..tiles {
            let base_i = ti * tile;
            let x_ptr = conv_out.as_ptr().add(t * d_inner + base_i);
            let z_ptr = z.as_ptr().add(t * d_inner + base_i);
            let delta_ptr = delta.as_ptr().add(t * d_inner + base_i);
            let x0 = _mm256_loadu_ps(x_ptr);
            let z0 = _mm256_loadu_ps(z_ptr);
            let delta0 = _mm256_loadu_ps(delta_ptr);
            let mut acc0 = _mm256_setzero_ps();
            let base_off = ti * d_state_pad * lane_stride;
            let a_base = a_pre.as_ptr().add(base_off);
            let h_base = layer_state.as_ptr().add(base_off);
            let next_base = if panel && ti + 1 < tiles {
                Some((ti + 1) * d_state_pad * lane_stride)
            } else {
                None
            };
            let mut a_ptr = a_base;
            let mut h_ptr = h_base;
            for j in 0..d_state {
                if prefetch_dist != 0 && j + prefetch_dist < d_state {
                    let pf = (j + prefetch_dist) * lane_stride;
                    _mm_prefetch(h_base.add(pf) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_base.add(pf) as *const i8, _MM_HINT_T0);
                    if let Some(next) = next_base {
                        let pf_next = next + (j + prefetch_dist) * lane_stride;
                        _mm_prefetch(layer_state.as_ptr().add(pf_next) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(a_pre.as_ptr().add(pf_next) as *const i8, _MM_HINT_T0);
                    }
                }
                debug_assert_eq!((a_ptr as usize) & 31, 0);
                debug_assert_eq!((h_ptr as usize) & 31, 0);
                let a0 = _mm256_load_ps(a_ptr);
                let h0 = _mm256_load_ps(h_ptr);
                let x0v = _mm256_mul_ps(delta0, a0);
                let a_bar0 = exp256_ps(x0v);
                let b = _mm256_set1_ps(b_vec_t[j]);
                let b_bar0 = _mm256_mul_ps(b, delta0);
                let h_new0 = _mm256_fmadd_ps(a_bar0, h0, _mm256_mul_ps(b_bar0, x0));
                _mm256_store_ps(layer_state.as_mut_ptr().add(base_off + j * lane_stride), h_new0);
                let c = _mm256_set1_ps(c_vec_t[j]);
                acc0 = _mm256_fmadd_ps(c, h_new0, acc0);
                a_ptr = a_ptr.add(lane_stride);
                h_ptr = h_ptr.add(lane_stride);
            }
            let d0 = _mm256_loadu_ps(lw.d.as_ptr().add(base_i));
            let mut y0 = _mm256_fmadd_ps(d0, x0, acc0);
            let sig0 = sigmoid_fast_avx2(z0);
            let silu0 = _mm256_mul_ps(z0, sig0);
            y0 = _mm256_mul_ps(y0, silu0);
            _mm256_storeu_ps(scan_t.as_mut_ptr().add(base_i), y0);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn conv1d_depthwise_avx2(
    cfg: &ModelConfig,
    conv_w_packed: &[f32],
    conv_b: &[f32],
    x: &[f32],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    let d_inner = cfg.d_inner;
    let k = cfg.conv_kernel;
    let pad = k - 1;
    let seq_len = cfg.seq_len;
    for t in 0..seq_len {
        let out_t = &mut out[t * d_inner..(t + 1) * d_inner];
        let mut i = 0usize;
        while i + 8 <= d_inner {
            let mut acc = _mm256_loadu_ps(conv_b.as_ptr().add(i));
            for kk in 0..k {
                let idx = t as isize + kk as isize - pad as isize;
                if idx >= 0 {
                    let x_ptr = x.as_ptr().add(idx as usize * d_inner + i);
                    let w_ptr = conv_w_packed.as_ptr().add(kk * d_inner + i);
                    let xv = _mm256_loadu_ps(x_ptr);
                    let wv = _mm256_loadu_ps(w_ptr);
                    acc = _mm256_fmadd_ps(wv, xv, acc);
                }
            }
            let sig = sigmoid_fast_avx2(acc);
            let y = _mm256_mul_ps(acc, sig);
            _mm256_storeu_ps(out_t.as_mut_ptr().add(i), y);
            i += 8;
        }
        while i < d_inner {
            let mut acc = conv_b[i];
            for kk in 0..k {
                let idx = t as isize + kk as isize - pad as isize;
                if idx >= 0 {
                    acc += conv_w_packed[kk * d_inner + i] * x[idx as usize * d_inner + i];
                }
            }
            out_t[i] = silu_dispatch(acc, MathBackend::Approx);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn conv1d_depthwise_avx512(
    cfg: &ModelConfig,
    conv_w_packed: &[f32],
    conv_b: &[f32],
    x: &[f32],
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    let d_inner = cfg.d_inner;
    let k = cfg.conv_kernel;
    let pad = k - 1;
    let seq_len = cfg.seq_len;
    for t in 0..seq_len {
        let out_t = &mut out[t * d_inner..(t + 1) * d_inner];
        let mut i = 0usize;
        while i + 16 <= d_inner {
            let mut acc = _mm512_loadu_ps(conv_b.as_ptr().add(i));
            for kk in 0..k {
                let idx = t as isize + kk as isize - pad as isize;
                if idx >= 0 {
                    let x_ptr = x.as_ptr().add(idx as usize * d_inner + i);
                    let w_ptr = conv_w_packed.as_ptr().add(kk * d_inner + i);
                    let xv = _mm512_loadu_ps(x_ptr);
                    let wv = _mm512_loadu_ps(w_ptr);
                    acc = _mm512_fmadd_ps(wv, xv, acc);
                }
            }
            let sig = sigmoid_fast_avx512(acc);
            let y = _mm512_mul_ps(acc, sig);
            _mm512_storeu_ps(out_t.as_mut_ptr().add(i), y);
            i += 16;
        }
        while i < d_inner {
            let mut acc = conv_b[i];
            for kk in 0..k {
                let idx = t as isize + kk as isize - pad as isize;
                if idx >= 0 {
                    acc += conv_w_packed[kk * d_inner + i] * x[idx as usize * d_inner + i];
                }
            }
            out_t[i] = silu_dispatch(acc, MathBackend::Approx);
            i += 1;
        }
    }
}

fn ssm_update_full_scalar(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    conv_out: &[f32],
    z: &[f32],
    delta: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    x_proj_out: &[f32],
    x_proj_dim: usize,
    dt_rank: usize,
    use_xproj_bc: bool,
    a_pre: &[f32],
    scan_out: &mut [f32],
    layer_state: &mut [f32],
    math_backend: MathBackend,
    fast_math: bool,
    fast_stats: &mut Option<&mut FastMathStats>,
) {
    #[cfg(not(feature = "bench_instrument"))]
    let _ = fast_stats;
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let seq_len = cfg.seq_len;
    for i in 0..d_inner {
        let state_row = &mut layer_state[i * d_state_pad..(i + 1) * d_state_pad];
        let a_row = &a_pre[i * d_state_pad..(i + 1) * d_state_pad];
        for t in 0..seq_len {
            let x_i = conv_out[t * d_inner + i];
            let z_i = z[t * d_inner + i];
            let delta_i = delta[t * d_inner + i];
            let (b_vec_t, c_vec_t) = if use_xproj_bc {
                let x_proj_t = &x_proj_out[t * x_proj_dim..][..x_proj_dim];
                (
                    &x_proj_t[dt_rank..dt_rank + d_state_pad],
                    &x_proj_t[dt_rank + d_state_pad..dt_rank + 2 * d_state_pad],
                )
            } else {
                (
                    &b_vec[t * d_state_pad..(t + 1) * d_state_pad],
                    &c_vec[t * d_state_pad..(t + 1) * d_state_pad],
                )
            };
            let scan_t = &mut scan_out[t * d_inner..(t + 1) * d_inner];

            #[cfg(feature = "bench_instrument")]
            if t == 0 {
                sample_exp_stats(fast_math, fast_stats, delta_i, a_row, i, d_state);
            }

            let mut acc = 0.0f32;
            for j in 0..d_state {
                let a = a_row[j];
                let a_bar = if fast_math {
                    exp_approx_scalar(delta_i * a)
                } else {
                    libm::expf(delta_i * a)
                };
                let b_bar = b_vec_t[j] * delta_i;
                let h = state_row[j];
                let h_new = a_bar * h + b_bar * x_i;
                state_row[j] = h_new;
                acc += c_vec_t[j] * h_new;
            }

            let mut y = acc + lw.d[i] * x_i;
            y *= silu_dispatch(z_i, math_backend);
            scan_t[i] = y;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn ssm_update_avx512_full_fused_dt(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    conv_out: &[f32],
    z: &[f32],
    dt_in: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    x_proj_out: &[f32],
    x_proj_dim: usize,
    dt_rank: usize,
    use_xproj_bc: bool,
    a_pre: &[f32],
    scan_out: &mut [f32],
    layer_state: &mut [f32],
    math_backend: MathBackend,
) {
    use std::arch::x86_64::*;
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let seq_len = cfg.seq_len;
    let prefetch_dist = if d_state >= 32 { 16 } else { 0 };
    for i in 0..d_inner {
        let state_row = &mut layer_state[i * d_state_pad..(i + 1) * d_state_pad];
        let a_row = &a_pre[i * d_state_pad..(i + 1) * d_state_pad];
        let dt_w_row = &lw.dt_proj_w[i * dt_rank..][..dt_rank];
        let dt_b = lw.dt_proj_b[i];
        for t in 0..seq_len {
            let dt_in_t = &dt_in[t * dt_rank..][..dt_rank];
            let mut dot = if dt_rank == 16 {
                dot16_avx512(dt_w_row.as_ptr(), dt_in_t.as_ptr())
            } else {
                let mut acc = 0.0f32;
                for r in 0..dt_rank {
                    acc += dt_w_row[r] * dt_in_t[r];
                }
                acc
            };
            dot += dt_b;
            let delta_i = softplus_dispatch(
                dot,
                cfg.softplus_kind,
                cfg.softplus_beta,
                cfg.softplus_threshold,
                math_backend,
            );

            let x_i = conv_out[t * d_inner + i];
            let z_i = z[t * d_inner + i];
            let (b_vec_t, c_vec_t) = if use_xproj_bc {
                let x_proj_t = &x_proj_out[t * x_proj_dim..][..x_proj_dim];
                (
                    &x_proj_t[dt_rank..dt_rank + d_state_pad],
                    &x_proj_t[dt_rank + d_state_pad..dt_rank + 2 * d_state_pad],
                )
            } else {
                (
                    &b_vec[t * d_state_pad..(t + 1) * d_state_pad],
                    &c_vec[t * d_state_pad..(t + 1) * d_state_pad],
                )
            };
            let scan_t = &mut scan_out[t * d_inner..(t + 1) * d_inner];

            let x_v = _mm512_set1_ps(x_i);
            let delta_v = _mm512_set1_ps(delta_i);
            let mut acc_v = _mm512_setzero_ps();
            let mut j = 0usize;
            while j + 32 <= d_state {
                if prefetch_dist != 0 && j + prefetch_dist < d_state {
                    _mm_prefetch(state_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(b_vec_t.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(c_vec_t.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                }

                let a0 = _mm512_loadu_ps(a_row.as_ptr().add(j));
                let x0 = _mm512_mul_ps(delta_v, a0);
                let a_bar0 = exp512_ps(x0);
                let h0 = _mm512_loadu_ps(state_row.as_ptr().add(j));
                let b0 = _mm512_loadu_ps(b_vec_t.as_ptr().add(j));
                let c0 = _mm512_loadu_ps(c_vec_t.as_ptr().add(j));
                let b_bar0 = _mm512_mul_ps(b0, delta_v);
                let h_new0 = _mm512_fmadd_ps(a_bar0, h0, _mm512_mul_ps(b_bar0, x_v));
                _mm512_storeu_ps(state_row.as_mut_ptr().add(j), h_new0);
                acc_v = _mm512_fmadd_ps(c0, h_new0, acc_v);

                let j1 = j + 16;
                let a1 = _mm512_loadu_ps(a_row.as_ptr().add(j1));
                let x1 = _mm512_mul_ps(delta_v, a1);
                let a_bar1 = exp512_ps(x1);
                let h1 = _mm512_loadu_ps(state_row.as_ptr().add(j1));
                let b1 = _mm512_loadu_ps(b_vec_t.as_ptr().add(j1));
                let c1 = _mm512_loadu_ps(c_vec_t.as_ptr().add(j1));
                let b_bar1 = _mm512_mul_ps(b1, delta_v);
                let h_new1 = _mm512_fmadd_ps(a_bar1, h1, _mm512_mul_ps(b_bar1, x_v));
                _mm512_storeu_ps(state_row.as_mut_ptr().add(j1), h_new1);
                acc_v = _mm512_fmadd_ps(c1, h_new1, acc_v);
                j += 32;
            }
            while j + 16 <= d_state {
                if prefetch_dist != 0 && j + prefetch_dist < d_state {
                    _mm_prefetch(state_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                }
                let a = _mm512_loadu_ps(a_row.as_ptr().add(j));
                let x = _mm512_mul_ps(delta_v, a);
                let a_bar = exp512_ps(x);
                let h = _mm512_loadu_ps(state_row.as_ptr().add(j));
                let b_val = _mm512_loadu_ps(b_vec_t.as_ptr().add(j));
                let c_val = _mm512_loadu_ps(c_vec_t.as_ptr().add(j));
                let b_bar = _mm512_mul_ps(b_val, delta_v);
                let h_new = _mm512_fmadd_ps(a_bar, h, _mm512_mul_ps(b_bar, x_v));
                _mm512_storeu_ps(state_row.as_mut_ptr().add(j), h_new);
                acc_v = _mm512_fmadd_ps(c_val, h_new, acc_v);
                j += 16;
            }
            let mut acc = hsum512_ps(acc_v);
            while j < d_state {
                let a = a_row[j];
                let a_bar = exp_approx_scalar(delta_i * a);
                let b_bar = b_vec_t[j] * delta_i;
                let h = state_row[j];
                let h_new = a_bar * h + b_bar * x_i;
                state_row[j] = h_new;
                acc += c_vec_t[j] * h_new;
                j += 1;
            }
            let mut y = acc + lw.d[i] * x_i;
            y *= silu_dispatch(z_i, math_backend);
            scan_t[i] = y;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn ssm_update_avx2_full_fused_dt(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    conv_out: &[f32],
    z: &[f32],
    dt_in: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    x_proj_out: &[f32],
    x_proj_dim: usize,
    dt_rank: usize,
    use_xproj_bc: bool,
    a_pre: &[f32],
    scan_out: &mut [f32],
    layer_state: &mut [f32],
    math_backend: MathBackend,
) {
    use std::arch::x86_64::*;
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let seq_len = cfg.seq_len;
    let prefetch_dist = if d_state >= 32 { 16 } else { 0 };
    for i in 0..d_inner {
        let state_row = &mut layer_state[i * d_state_pad..(i + 1) * d_state_pad];
        let a_row = &a_pre[i * d_state_pad..(i + 1) * d_state_pad];
        let dt_w_row = &lw.dt_proj_w[i * dt_rank..][..dt_rank];
        let dt_b = lw.dt_proj_b[i];
        for t in 0..seq_len {
            let dt_in_t = &dt_in[t * dt_rank..][..dt_rank];
            let mut dot = if dt_rank == 16 {
                dot16_avx2(dt_w_row.as_ptr(), dt_in_t.as_ptr())
            } else {
                let mut acc = 0.0f32;
                for r in 0..dt_rank {
                    acc += dt_w_row[r] * dt_in_t[r];
                }
                acc
            };
            dot += dt_b;
            let delta_i = softplus_dispatch(
                dot,
                cfg.softplus_kind,
                cfg.softplus_beta,
                cfg.softplus_threshold,
                math_backend,
            );

            let x_i = conv_out[t * d_inner + i];
            let z_i = z[t * d_inner + i];
            let (b_vec_t, c_vec_t) = if use_xproj_bc {
                let x_proj_t = &x_proj_out[t * x_proj_dim..][..x_proj_dim];
                (
                    &x_proj_t[dt_rank..dt_rank + d_state_pad],
                    &x_proj_t[dt_rank + d_state_pad..dt_rank + 2 * d_state_pad],
                )
            } else {
                (
                    &b_vec[t * d_state_pad..(t + 1) * d_state_pad],
                    &c_vec[t * d_state_pad..(t + 1) * d_state_pad],
                )
            };
            let scan_t = &mut scan_out[t * d_inner..(t + 1) * d_inner];

            let x_v = _mm256_set1_ps(x_i);
            let delta_v = _mm256_set1_ps(delta_i);
            let mut acc_v = _mm256_setzero_ps();
            let mut j = 0usize;
            while j + 16 <= d_state {
                if prefetch_dist != 0 && j + prefetch_dist < d_state {
                    _mm_prefetch(state_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(b_vec_t.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(c_vec_t.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                }

                let a0 = _mm256_loadu_ps(a_row.as_ptr().add(j));
                let x0 = _mm256_mul_ps(delta_v, a0);
                let a_bar0 = exp256_ps(x0);
                let h0 = _mm256_loadu_ps(state_row.as_ptr().add(j));
                let b0 = _mm256_loadu_ps(b_vec_t.as_ptr().add(j));
                let c0 = _mm256_loadu_ps(c_vec_t.as_ptr().add(j));
                let b_bar0 = _mm256_mul_ps(b0, delta_v);
                let h_new0 = _mm256_fmadd_ps(a_bar0, h0, _mm256_mul_ps(b_bar0, x_v));
                _mm256_storeu_ps(state_row.as_mut_ptr().add(j), h_new0);
                acc_v = _mm256_fmadd_ps(c0, h_new0, acc_v);

                let j1 = j + 8;
                let a1 = _mm256_loadu_ps(a_row.as_ptr().add(j1));
                let x1 = _mm256_mul_ps(delta_v, a1);
                let a_bar1 = exp256_ps(x1);
                let h1 = _mm256_loadu_ps(state_row.as_ptr().add(j1));
                let b1 = _mm256_loadu_ps(b_vec_t.as_ptr().add(j1));
                let c1 = _mm256_loadu_ps(c_vec_t.as_ptr().add(j1));
                let b_bar1 = _mm256_mul_ps(b1, delta_v);
                let h_new1 = _mm256_fmadd_ps(a_bar1, h1, _mm256_mul_ps(b_bar1, x_v));
                _mm256_storeu_ps(state_row.as_mut_ptr().add(j1), h_new1);
                acc_v = _mm256_fmadd_ps(c1, h_new1, acc_v);
                j += 16;
            }
            while j + 8 <= d_state {
                if prefetch_dist != 0 && j + prefetch_dist < d_state {
                    _mm_prefetch(state_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_row.as_ptr().add(j + prefetch_dist) as *const i8, _MM_HINT_T0);
                }
                let a = _mm256_loadu_ps(a_row.as_ptr().add(j));
                let x = _mm256_mul_ps(delta_v, a);
                let a_bar = exp256_ps(x);
                let h = _mm256_loadu_ps(state_row.as_ptr().add(j));
                let b_val = _mm256_loadu_ps(b_vec_t.as_ptr().add(j));
                let c_val = _mm256_loadu_ps(c_vec_t.as_ptr().add(j));
                let b_bar = _mm256_mul_ps(b_val, delta_v);
                let h_new = _mm256_fmadd_ps(a_bar, h, _mm256_mul_ps(b_bar, x_v));
                _mm256_storeu_ps(state_row.as_mut_ptr().add(j), h_new);
                acc_v = _mm256_fmadd_ps(c_val, h_new, acc_v);
                j += 8;
            }
            let mut acc = hsum256_ps(acc_v);
            while j < d_state {
                let a = a_row[j];
                let a_bar = exp_approx_scalar(delta_i * a);
                let b_bar = b_vec_t[j] * delta_i;
                let h = state_row[j];
                let h_new = a_bar * h + b_bar * x_i;
                state_row[j] = h_new;
                acc += c_vec_t[j] * h_new;
                j += 1;
            }
            let mut y = acc + lw.d[i] * x_i;
            y *= silu_dispatch(z_i, math_backend);
            scan_t[i] = y;
        }
    }
}

fn ssm_update_full_scalar_fused_dt(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    conv_out: &[f32],
    z: &[f32],
    dt_in: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    x_proj_out: &[f32],
    x_proj_dim: usize,
    dt_rank: usize,
    use_xproj_bc: bool,
    a_pre: &[f32],
    scan_out: &mut [f32],
    layer_state: &mut [f32],
    math_backend: MathBackend,
    fast_stats: &mut Option<&mut FastMathStats>,
) {
    #[cfg(not(feature = "bench_instrument"))]
    let _ = fast_stats;
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let seq_len = cfg.seq_len;
    for i in 0..d_inner {
        let state_row = &mut layer_state[i * d_state_pad..(i + 1) * d_state_pad];
        let a_row = &a_pre[i * d_state_pad..(i + 1) * d_state_pad];
        let dt_w_row = &lw.dt_proj_w[i * dt_rank..][..dt_rank];
        let dt_b = lw.dt_proj_b[i];
        for t in 0..seq_len {
            let dt_in_t = &dt_in[t * dt_rank..][..dt_rank];
            let mut dot = dt_b;
            for r in 0..dt_rank {
                dot += dt_w_row[r] * dt_in_t[r];
            }
            let delta_i = softplus_dispatch(
                dot,
                cfg.softplus_kind,
                cfg.softplus_beta,
                cfg.softplus_threshold,
                math_backend,
            );

            let x_i = conv_out[t * d_inner + i];
            let z_i = z[t * d_inner + i];
            let (b_vec_t, c_vec_t) = if use_xproj_bc {
                let x_proj_t = &x_proj_out[t * x_proj_dim..][..x_proj_dim];
                (
                    &x_proj_t[dt_rank..dt_rank + d_state_pad],
                    &x_proj_t[dt_rank + d_state_pad..dt_rank + 2 * d_state_pad],
                )
            } else {
                (
                    &b_vec[t * d_state_pad..(t + 1) * d_state_pad],
                    &c_vec[t * d_state_pad..(t + 1) * d_state_pad],
                )
            };
            let scan_t = &mut scan_out[t * d_inner..(t + 1) * d_inner];

            #[cfg(feature = "bench_instrument")]
            if t == 0 {
                sample_exp_stats(true, fast_stats, delta_i, a_row, i, d_state);
            }

            let mut acc = 0.0f32;
            for j in 0..d_state {
                let a = a_row[j];
                let a_bar = exp_approx_scalar(delta_i * a);
                let b_bar = b_vec_t[j] * delta_i;
                let h = state_row[j];
                let h_new = a_bar * h + b_bar * x_i;
                state_row[j] = h_new;
                acc += c_vec_t[j] * h_new;
            }

            let mut y = acc + lw.d[i] * x_i;
            y *= silu_dispatch(z_i, math_backend);
            scan_t[i] = y;
        }
    }
}

#[cfg(feature = "bench_instrument")]
#[inline]
fn sample_exp_stats(
    fast_math: bool,
    stats: &mut Option<&mut FastMathStats>,
    delta_i: f32,
    a_row: &[f32],
    i: usize,
    d_state_pad: usize,
) {
    if !fast_math {
        return;
    }
    let stats = match stats.as_deref_mut() {
        Some(s) => s,
        None => return,
    };
    if stats.exp_samples >= stats.exp_max_samples || d_state_pad == 0 {
        return;
    }
    let j = (i.wrapping_mul(13).wrapping_add(7)) % d_state_pad;
    let x = delta_i * a_row[j];
    let approx = exp_approx_scalar(x);
    let exact = libm::expf(x);
    stats.record_exp(x, approx, exact);
}

fn ssm_update_scalar(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    batch: usize,
    conv_out: &[f32],
    z: &[f32],
    delta: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    a_pre: &[f32],
    math_backend: MathBackend,
    scan_out: &mut [f32],
    layer_state: &mut [f32],
    fast_stats: &mut Option<&mut FastMathStats>,
) {
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let layer_state_stride = d_inner * d_state_pad;
    #[cfg(feature = "bench_instrument")]
    let fast_math = matches!(
        math_backend,
        MathBackend::Approx | MathBackend::Sleef | MathBackend::FastBf16 | MathBackend::FastWild
    );
    #[cfg(not(feature = "bench_instrument"))]
    let _ = fast_stats;

    for b in 0..batch {
        let conv_b = slice_batched(conv_out, batch, d_inner, b);
        let z_b = slice_batched(z, batch, d_inner, b);
        let delta_b = slice_batched(delta, batch, d_inner, b);
        let b_vec_b = slice_batched(b_vec, batch, d_state_pad, b);
        let c_vec_b = slice_batched(c_vec, batch, d_state_pad, b);
        let scan_out_b = slice_batched_mut(scan_out, batch, d_inner, b);
        let state_b = &mut layer_state[b * layer_state_stride..(b + 1) * layer_state_stride];

        for i in 0..d_inner {
            let x_i = conv_b[i];
            let z_i = z_b[i];
            let delta_i = delta_b[i];
            let mut acc = 0.0f32;
            let state_row = &mut state_b[i * d_state_pad..(i + 1) * d_state_pad];
            let a_row = &a_pre[i * d_state_pad..(i + 1) * d_state_pad];
            #[cfg(feature = "bench_instrument")]
            if b == 0 {
                sample_exp_stats(fast_math, fast_stats, delta_i, a_row, i, d_state);
            }

            for j in 0..d_state {
                let a = a_row[j];
                let a_bar = libm::expf(delta_i * a);
                let b_bar = b_vec_b[j] * delta_i;
                let h = state_row[j];
                let h_new = a_bar * h + b_bar * x_i;
                state_row[j] = h_new;
                acc += c_vec_b[j] * h_new;
            }

            let mut y = acc + lw.d[i] * x_i;
            y *= silu_dispatch(z_i, math_backend);
            scan_out_b[i] = y;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    let mut tmp = [0.0f32; 8];
    std::arch::x86_64::_mm256_storeu_ps(tmp.as_mut_ptr(), v);
    tmp.iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum512_ps(v: std::arch::x86_64::__m512) -> f32 {
    let mut tmp = [0.0f32; 16];
    std::arch::x86_64::_mm512_storeu_ps(tmp.as_mut_ptr(), v);
    tmp.iter().sum()
}

fn layer_norm_stats_scalar(input: &[f32], eps: f32) -> (f32, f32) {
    let n = input.len();
    let mut sum = 0.0f32;
    for &v in input {
        sum += v;
    }
    let mean = sum / n as f32;
    let mut var = 0.0f32;
    for &v in input {
        let d = v - mean;
        var += d * d;
    }
    var /= n as f32;
    let inv = 1.0f32 / (var + eps).sqrt();
    (mean, inv)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn layer_norm_stats_avx2(input: &[f32], eps: f32) -> (f32, f32) {
    use std::arch::x86_64::*;
    let n = input.len();
    let mut sum_v = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        sum_v = _mm256_add_ps(sum_v, x);
        i += 8;
    }
    let mut sum = hsum256_ps(sum_v);
    while i < n {
        sum += *input.get_unchecked(i);
        i += 1;
    }
    let mean = sum / n as f32;
    let mean_v = _mm256_set1_ps(mean);
    let mut var_v = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= n {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let d = _mm256_sub_ps(x, mean_v);
        var_v = _mm256_fmadd_ps(d, d, var_v);
        i += 8;
    }
    let mut var = hsum256_ps(var_v);
    while i < n {
        let d = *input.get_unchecked(i) - mean;
        var += d * d;
        i += 1;
    }
    var /= n as f32;
    let inv = 1.0f32 / (var + eps).sqrt();
    (mean, inv)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn layer_norm_stats_avx512(input: &[f32], eps: f32) -> (f32, f32) {
    use std::arch::x86_64::*;
    let n = input.len();
    let mut sum_v = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let x = _mm512_loadu_ps(input.as_ptr().add(i));
        sum_v = _mm512_add_ps(sum_v, x);
        i += 16;
    }
    let mut sum = hsum512_ps(sum_v);
    while i < n {
        sum += *input.get_unchecked(i);
        i += 1;
    }
    let mean = sum / n as f32;
    let mean_v = _mm512_set1_ps(mean);
    let mut var_v = _mm512_setzero_ps();
    i = 0;
    while i + 16 <= n {
        let x = _mm512_loadu_ps(input.as_ptr().add(i));
        let d = _mm512_sub_ps(x, mean_v);
        var_v = _mm512_fmadd_ps(d, d, var_v);
        i += 16;
    }
    let mut var = hsum512_ps(var_v);
    while i < n {
        let d = *input.get_unchecked(i) - mean;
        var += d * d;
        i += 1;
    }
    var /= n as f32;
    let inv = 1.0f32 / (var + eps).sqrt();
    (mean, inv)
}

fn layer_norm_stats_dispatch(input: &[f32], eps: f32, dispatch: CpuDispatch) -> (f32, f32) {
    match dispatch {
        CpuDispatch::Scalar => layer_norm_stats_scalar(input, eps),
        CpuDispatch::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                layer_norm_stats_avx2(input, eps)
            }
            #[cfg(not(target_arch = "x86_64"))]
            layer_norm_stats_scalar(input, eps)
        }
        CpuDispatch::Avx512 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                layer_norm_stats_avx512(input, eps)
            }
            #[cfg(not(target_arch = "x86_64"))]
            layer_norm_stats_scalar(input, eps)
        }
    }
}

fn packed16_out_dim_f32(packed_len: usize, in_dim: usize) -> usize {
    let block = 16usize;
    if in_dim == 0 {
        return 0;
    }
    let stride = in_dim * block;
    if stride == 0 {
        return 0;
    }
    (packed_len / stride) * block
}

fn packed16_out_dim_bf16(packed_len: usize, in_dim: usize) -> usize {
    let block = 16usize;
    if in_dim == 0 {
        return 0;
    }
    let pairs = (in_dim + 1) / 2;
    let stride = pairs * block * 2;
    if stride == 0 {
        return 0;
    }
    (packed_len / stride) * block
}

fn matmul_step1_packed(
    weight: &[f32],
    weight_packed: Option<&[f32]>,
    weight_packed_bf16: Option<&[u16]>,
    out_dim: usize,
    in_dim: usize,
    input: &[f32],
    bias: &[f32],
    output: &mut [f32],
    dispatch: CpuDispatch,
    math_backend: MathBackend,
) {
    let use_bf16 = use_bf16_matmul(math_backend, dispatch);
    if use_bf16 {
        if let Some(packed_bf16) = weight_packed_bf16 {
            let packed_out = packed16_out_dim_bf16(packed_bf16.len(), in_dim);
            if packed_out == out_dim {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    audit_trace("matmul_step1", "avx512", "packed_bf16");
                    matmul_packed16_batch_avx512_bf16(
                        packed_bf16,
                        out_dim,
                        in_dim,
                        input,
                        1,
                        bias,
                        output,
                    );
                    return;
                }
            }
        } else {
            audit_once(AUDIT_FALLBACK, "matmul_step1: missing bf16 packed");
        }
    }
    if let Some(packed) = weight_packed {
        let packed_out = packed16_out_dim_f32(packed.len(), in_dim);
        let reason = if packed_out == out_dim {
            "gemv_m1_kernel_v1"
        } else {
            "gemv_m1_tail_v1"
        };
        audit_trace(
            "matmul_step1",
            match dispatch {
                CpuDispatch::Avx2 => "avx2",
                CpuDispatch::Avx512 => "avx512",
                CpuDispatch::Scalar => "scalar",
            },
            reason,
        );
        if packed_out >= out_dim && packed_out > 0 {
            let block = 16usize;
            let full_out = (out_dim / block) * block;
            if full_out > 0 {
                matmul_packed16_dispatch_m1(packed, full_out, in_dim, input, bias, &mut output[..full_out], dispatch);
            }
            if full_out < out_dim {
                if packed_out < full_out + block {
                    let msg = format!(
                        "matmul_step1: packed tail missing (packed_out={}, out_dim={}, in_dim={})",
                        packed_out, out_dim, in_dim
                    );
                    fallback_or_panic(&format!("E2L02 {msg}"));
                } else {
                    let tail_out = out_dim - full_out;
                    let tail_base = full_out * in_dim;
                    let tail_packed = &packed[tail_base..tail_base + in_dim * block];
                    let tail_bias = &bias[full_out..];
                    let tail_output = &mut output[full_out..];
                    matmul_packed16_tail_dispatch_m1(
                        tail_packed,
                        in_dim,
                        input,
                        tail_bias,
                        tail_out,
                        tail_output,
                        dispatch,
                    );
                    return;
                }
            } else {
                return;
            }
        }
        if packed_out > 0 && packed_out < out_dim {
            let msg = format!(
                "matmul_step1: packed path incomplete (packed_out={}, out_dim={}, in_dim={})",
                packed_out, out_dim, in_dim
            );
            fallback_or_panic(&format!("E2L02 {msg}"));
        }
    }
    if dispatch != CpuDispatch::Scalar && out_dim > 0 {
        let msg = format!(
            "matmul_step1: packed path unavailable (out_dim={}, in_dim={})",
            out_dim, in_dim
        );
        fallback_or_panic(&format!("E2L01 {msg}"));
    }
    audit_trace(
        "matmul_step1",
        match dispatch {
            CpuDispatch::Avx2 => "avx2",
            CpuDispatch::Avx512 => "avx512",
            CpuDispatch::Scalar => "scalar",
        },
        "unpacked",
    );
    matmul_vec_dispatch(weight, out_dim, in_dim, input, Some(bias), output, dispatch);
}

fn ln1_in_proj_fused_step1(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    dispatch: CpuDispatch,
    math_backend: MathBackend,
    x_in: &[f32],
    scaled: &mut [f32],
    out: &mut [f32],
) -> Result<(), KernelError> {
    let (mean, inv) = layer_norm_stats_dispatch(x_in, cfg.ln_eps1, dispatch);
    let d_model = cfg.d_model;
    let out_dim = cfg.d_inner * 2;
    if scaled.len() < d_model || out.len() < out_dim {
        return Err(KernelError::BadLen("ln1/in_proj fused"));
    }
    for i in 0..d_model {
        unsafe {
            *scaled.get_unchecked_mut(i) = *x_in.get_unchecked(i) * *lw.ln1_w.get_unchecked(i);
        }
    }
    let bias = lw
        .in_proj_b
        .or_else(|| lw.in_proj_b_zero.as_deref())
        .ok_or(KernelError::MissingWeights("in_proj.bias"))?;
    matmul_step1_packed(
        lw.in_proj_w,
        lw.in_proj_w_packed.as_deref(),
        lw.in_proj_w_packed_bf16.as_deref(),
        out_dim,
        d_model,
        scaled,
        bias,
        out,
        dispatch,
        math_backend,
    );
    let gamma_sum = lw
        .in_proj_w_gamma_sum
        .as_deref()
        .ok_or(KernelError::MissingWeights("in_proj.gamma_sum"))?;
    let beta_sum = lw
        .in_proj_w_beta_sum
        .as_deref()
        .ok_or(KernelError::MissingWeights("in_proj.beta_sum"))?;
    let inv_mean = inv * mean;
    for o in 0..out_dim {
        unsafe {
            let bias_v = *bias.get_unchecked(o);
            let dot = *out.get_unchecked(o) - bias_v;
            let adj = inv * dot + *beta_sum.get_unchecked(o) - inv_mean * *gamma_sum.get_unchecked(o);
            *out.get_unchecked_mut(o) = bias_v + adj;
        }
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn dot16_avx2(w: *const f32, x: *const f32) -> f32 {
    use std::arch::x86_64::*;
    let w0 = _mm256_loadu_ps(w);
    let x0 = _mm256_loadu_ps(x);
    let w1 = _mm256_loadu_ps(w.add(8));
    let x1 = _mm256_loadu_ps(x.add(8));
    let m0 = _mm256_mul_ps(w0, x0);
    let m1 = _mm256_mul_ps(w1, x1);
    hsum256_ps(m0) + hsum256_ps(m1)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn dot16_avx512(w: *const f32, x: *const f32) -> f32 {
    use std::arch::x86_64::*;
    let wv = _mm512_loadu_ps(w);
    let xv = _mm512_loadu_ps(x);
    let mv = _mm512_mul_ps(wv, xv);
    hsum512_ps(mv)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn matmul_packed16_block_avx2(
    weight_packed: &[f32],
    in_dim: usize,
    input: &[f32],
    block_idx: usize,
    bias: &[f32],
    out_block: &mut [f32; 16],
) {
    use std::arch::x86_64::*;
    let block = 16usize;
    let base = block_idx * in_dim * block;
    let bias_ptr = bias.as_ptr().add(block_idx * block);
    let mut acc_lo = _mm256_loadu_ps(bias_ptr);
    let mut acc_hi = _mm256_loadu_ps(bias_ptr.add(8));
    for k in 0..in_dim {
        let w_lo = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block));
        let w_hi = _mm256_loadu_ps(weight_packed.as_ptr().add(base + k * block + 8));
        let x = _mm256_set1_ps(*input.get_unchecked(k));
        acc_lo = _mm256_fmadd_ps(w_lo, x, acc_lo);
        acc_hi = _mm256_fmadd_ps(w_hi, x, acc_hi);
    }
    _mm256_storeu_ps(out_block.as_mut_ptr(), acc_lo);
    _mm256_storeu_ps(out_block.as_mut_ptr().add(8), acc_hi);
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn matmul_packed16_block_avx512(
    weight_packed: &[f32],
    in_dim: usize,
    input: &[f32],
    block_idx: usize,
    bias: &[f32],
    out_block: &mut [f32; 16],
) {
    use std::arch::x86_64::*;
    let block = 16usize;
    let base = block_idx * in_dim * block;
    let bias_ptr = bias.as_ptr().add(block_idx * block);
    let mut acc = _mm512_loadu_ps(bias_ptr);
    for k in 0..in_dim {
        let w = _mm512_loadu_ps(weight_packed.as_ptr().add(base + k * block));
        let x = _mm512_set1_ps(*input.get_unchecked(k));
        acc = _mm512_fmadd_ps(w, x, acc);
    }
    _mm512_storeu_ps(out_block.as_mut_ptr(), acc);
}

fn matmul_packed16_block_scalar(
    weight_packed: &[f32],
    in_dim: usize,
    input: &[f32],
    block_idx: usize,
    bias: &[f32],
    out_block: &mut [f32; 16],
) {
    let block = 16usize;
    let base = block_idx * in_dim * block;
    let bias_slice = &bias[block_idx * block..][..block];
    for i in 0..block {
        let mut acc = bias_slice[i];
        for k in 0..in_dim {
            acc += weight_packed[base + k * block + i] * input[k];
        }
        out_block[i] = acc;
    }
}

fn mlp_fused_streamed(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    dispatch: CpuDispatch,
    math_backend: MathBackend,
    ln2_out: &[f32],
    out: &mut [f32],
) {
    let block = 16usize;
    if cfg.d_mlp % block != 0 {
        fallback_or_panic("E2101 mlp_fused_streamed: d_mlp not multiple of 16");
        return;
    }
    let fc1_packed = match lw.fc1_w_packed.as_deref() {
        Some(p) => p,
        None => {
            fallback_or_panic("E2102 mlp_fused_streamed: missing fc1 packed");
            return;
        }
    };
    let fc2_packed = match lw.fc2_w_packed.as_deref() {
        Some(p) => p,
        None => {
            fallback_or_panic("E2103 mlp_fused_streamed: missing fc2 packed");
            return;
        }
    };
    match dispatch {
        CpuDispatch::Avx512 => audit_once(AUDIT_MLP, "mlp_fused_streamed: avx512"),
        CpuDispatch::Avx2 => audit_once(AUDIT_MLP, "mlp_fused_streamed: avx2"),
        CpuDispatch::Scalar => audit_once(AUDIT_MLP, "mlp_fused_streamed: scalar"),
    }
    let fc1_b = lw.fc1_b;
    let fc2_b = lw.fc2_b;
    let blocks = cfg.d_mlp / block;
    let out_blocks = cfg.d_model / block;
    if cfg.d_model % block != 0 {
        return;
    }
    for t in 0..cfg.seq_len {
        let x_t = slice_seq(ln2_out, cfg.d_model, t);
        let y_t = slice_seq_mut(out, cfg.d_model, t);
        y_t.copy_from_slice(fc2_b);
        for blk in 0..blocks {
            let mut fc1_block = [0.0f32; 16];
            match dispatch {
                CpuDispatch::Avx512 => {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        matmul_packed16_block_avx512(fc1_packed, cfg.d_model, x_t, blk, fc1_b, &mut fc1_block);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    matmul_packed16_block_scalar(fc1_packed, cfg.d_model, x_t, blk, fc1_b, &mut fc1_block);
                }
                CpuDispatch::Avx2 => {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        matmul_packed16_block_avx2(fc1_packed, cfg.d_model, x_t, blk, fc1_b, &mut fc1_block);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    matmul_packed16_block_scalar(fc1_packed, cfg.d_model, x_t, blk, fc1_b, &mut fc1_block);
                }
                CpuDispatch::Scalar => {
                    matmul_packed16_block_scalar(fc1_packed, cfg.d_model, x_t, blk, fc1_b, &mut fc1_block);
                }
            }
            for v in fc1_block.iter_mut() {
                *v = gelu_dispatch(*v, cfg.gelu_kind, math_backend);
            }
            match dispatch {
                CpuDispatch::Avx512 => {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        use std::arch::x86_64::*;
                        for out_blk in 0..out_blocks {
                            let base = out_blk * cfg.d_mlp * block + blk * block * block;
                            let mut acc = _mm512_loadu_ps(y_t.as_ptr().add(out_blk * block));
                            for k in 0..block {
                                let w = _mm512_loadu_ps(fc2_packed.as_ptr().add(base + k * block));
                                let x = _mm512_set1_ps(fc1_block[k]);
                                acc = _mm512_fmadd_ps(w, x, acc);
                            }
                            _mm512_storeu_ps(y_t.as_mut_ptr().add(out_blk * block), acc);
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        for out_blk in 0..out_blocks {
                            let base = out_blk * cfg.d_mlp * block + blk * block * block;
                            let y_block = &mut y_t[out_blk * block..][..block];
                            for k in 0..block {
                                let w_row = &fc2_packed[base + k * block..][..block];
                                let x = fc1_block[k];
                                for j in 0..block {
                                    y_block[j] += w_row[j] * x;
                                }
                            }
                        }
                    }
                }
                CpuDispatch::Avx2 => {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        use std::arch::x86_64::*;
                        for out_blk in 0..out_blocks {
                            let base = out_blk * cfg.d_mlp * block + blk * block * block;
                            let mut acc_lo = _mm256_loadu_ps(y_t.as_ptr().add(out_blk * block));
                            let mut acc_hi =
                                _mm256_loadu_ps(y_t.as_ptr().add(out_blk * block + 8));
                            for k in 0..block {
                                let w_lo =
                                    _mm256_loadu_ps(fc2_packed.as_ptr().add(base + k * block));
                                let w_hi = _mm256_loadu_ps(
                                    fc2_packed.as_ptr().add(base + k * block + 8),
                                );
                                let x = _mm256_set1_ps(fc1_block[k]);
                                acc_lo = _mm256_fmadd_ps(w_lo, x, acc_lo);
                                acc_hi = _mm256_fmadd_ps(w_hi, x, acc_hi);
                            }
                            _mm256_storeu_ps(y_t.as_mut_ptr().add(out_blk * block), acc_lo);
                            _mm256_storeu_ps(
                                y_t.as_mut_ptr().add(out_blk * block + 8),
                                acc_hi,
                            );
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        for out_blk in 0..out_blocks {
                            let base = out_blk * cfg.d_mlp * block + blk * block * block;
                            let y_block = &mut y_t[out_blk * block..][..block];
                            for k in 0..block {
                                let w_row = &fc2_packed[base + k * block..][..block];
                                let x = fc1_block[k];
                                for j in 0..block {
                                    y_block[j] += w_row[j] * x;
                                }
                            }
                        }
                    }
                }
                CpuDispatch::Scalar => {
                    for out_blk in 0..out_blocks {
                        let base = out_blk * cfg.d_mlp * block + blk * block * block;
                        let y_block = &mut y_t[out_blk * block..][..block];
                        for k in 0..block {
                            let w_row = &fc2_packed[base + k * block..][..block];
                            let x = fc1_block[k];
                            for j in 0..block {
                                y_block[j] += w_row[j] * x;
                            }
                        }
                    }
                }
            }
        }
    }
}

fn head_matmul_two_class(
    weight: &[f32],
    bias: &[f32],
    input: &[f32],
    output: &mut [f32],
    dispatch: CpuDispatch,
) {
    match dispatch {
        CpuDispatch::Scalar => {
            let d = input.len();
            let (w0, w1) = weight.split_at(d);
            let mut acc0 = bias[0];
            let mut acc1 = bias[1];
            for i in 0..d {
                let x = input[i];
                acc0 += w0[i] * x;
                acc1 += w1[i] * x;
            }
            output[0] = acc0;
            output[1] = acc1;
        }
        CpuDispatch::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                head_matmul_two_class_avx2(weight, bias, input, output);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let d = input.len();
                let (w0, w1) = weight.split_at(d);
                let mut acc0 = bias[0];
                let mut acc1 = bias[1];
                for i in 0..d {
                    let x = input[i];
                    acc0 += w0[i] * x;
                    acc1 += w1[i] * x;
                }
                output[0] = acc0;
                output[1] = acc1;
            }
        }
        CpuDispatch::Avx512 => {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                head_matmul_two_class_avx512(weight, bias, input, output);
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                let d = input.len();
                let (w0, w1) = weight.split_at(d);
                let mut acc0 = bias[0];
                let mut acc1 = bias[1];
                for i in 0..d {
                    let x = input[i];
                    acc0 += w0[i] * x;
                    acc1 += w1[i] * x;
                }
                output[0] = acc0;
                output[1] = acc1;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn head_matmul_two_class_avx2(
    weight: &[f32],
    bias: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let d = input.len();
    let (w0, w1) = weight.split_at(d);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= d {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let w0v = _mm256_loadu_ps(w0.as_ptr().add(i));
        let w1v = _mm256_loadu_ps(w1.as_ptr().add(i));
        acc0 = _mm256_fmadd_ps(w0v, x, acc0);
        acc1 = _mm256_fmadd_ps(w1v, x, acc1);
        i += 8;
    }
    let mut sum0 = hsum256_ps(acc0) + bias[0];
    let mut sum1 = hsum256_ps(acc1) + bias[1];
    while i < d {
        let x = input[i];
        sum0 += w0[i] * x;
        sum1 += w1[i] * x;
        i += 1;
    }
    output[0] = sum0;
    output[1] = sum1;
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn head_matmul_two_class_avx512(
    weight: &[f32],
    bias: &[f32],
    input: &[f32],
    output: &mut [f32],
) {
    use std::arch::x86_64::*;
    let d = input.len();
    let (w0, w1) = weight.split_at(d);
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= d {
        let x = _mm512_loadu_ps(input.as_ptr().add(i));
        let w0v = _mm512_loadu_ps(w0.as_ptr().add(i));
        let w1v = _mm512_loadu_ps(w1.as_ptr().add(i));
        acc0 = _mm512_fmadd_ps(w0v, x, acc0);
        acc1 = _mm512_fmadd_ps(w1v, x, acc1);
        i += 16;
    }
    let mut sum0 = hsum512_ps(acc0) + bias[0];
    let mut sum1 = hsum512_ps(acc1) + bias[1];
    while i < d {
        let x = input[i];
        sum0 += w0[i] * x;
        sum1 += w1[i] * x;
        i += 1;
    }
    output[0] = sum0;
    output[1] = sum1;
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn ssm_update_avx2(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    batch: usize,
    conv_out: &[f32],
    z: &[f32],
    delta: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    a_pre: &[f32],
    math_backend: MathBackend,
    scan_out: &mut [f32],
    layer_state: &mut [f32],
    fast_stats: &mut Option<&mut FastMathStats>,
) {
    use std::arch::x86_64::*;
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let layer_state_stride = d_inner * d_state_pad;
    #[cfg(feature = "bench_instrument")]
    let fast_math = matches!(
        math_backend,
        MathBackend::Approx | MathBackend::Sleef | MathBackend::FastBf16 | MathBackend::FastWild
    );
    #[cfg(not(feature = "bench_instrument"))]
    let _ = fast_stats;

    for b in 0..batch {
        let conv_b = slice_batched(conv_out, batch, d_inner, b);
        let z_b = slice_batched(z, batch, d_inner, b);
        let delta_b = slice_batched(delta, batch, d_inner, b);
        let b_vec_b = slice_batched(b_vec, batch, d_state_pad, b);
        let c_vec_b = slice_batched(c_vec, batch, d_state_pad, b);
        let scan_out_b = slice_batched_mut(scan_out, batch, d_inner, b);
        let state_b = &mut layer_state[b * layer_state_stride..(b + 1) * layer_state_stride];

        for i in 0..d_inner {
            let x_i = conv_b[i];
            let z_i = z_b[i];
            let delta_i = delta_b[i];
            let x_v = _mm256_set1_ps(x_i);
            let delta_v = _mm256_set1_ps(delta_i);
            let mut acc_v = _mm256_setzero_ps();
            let state_row = &mut state_b[i * d_state_pad..(i + 1) * d_state_pad];
            let a_row = &a_pre[i * d_state_pad..(i + 1) * d_state_pad];
            #[cfg(feature = "bench_instrument")]
            if b == 0 {
                sample_exp_stats(fast_math, fast_stats, delta_i, a_row, i, d_state);
            }

            let mut j = 0usize;
            while j + 8 <= d_state {
                let a = _mm256_loadu_ps(a_row.as_ptr().add(j));
                let x = _mm256_mul_ps(delta_v, a);
                let a_bar = exp256_ps(x);
                let h = _mm256_loadu_ps(state_row.as_ptr().add(j));
                let b_val = _mm256_loadu_ps(b_vec_b.as_ptr().add(j));
                let c_val = _mm256_loadu_ps(c_vec_b.as_ptr().add(j));
                let b_bar = _mm256_mul_ps(b_val, delta_v);
                let h_new = _mm256_fmadd_ps(a_bar, h, _mm256_mul_ps(b_bar, x_v));
                _mm256_storeu_ps(state_row.as_mut_ptr().add(j), h_new);
                acc_v = _mm256_fmadd_ps(c_val, h_new, acc_v);
                j += 8;
            }

            let mut acc = hsum256_ps(acc_v);
            while j < d_state {
                let a = a_row[j];
                let a_bar = libm::expf(delta_i * a);
                let b_bar = b_vec_b[j] * delta_i;
                let h = state_row[j];
                let h_new = a_bar * h + b_bar * x_i;
                state_row[j] = h_new;
                acc += c_vec_b[j] * h_new;
                j += 1;
            }

            let mut y = acc + lw.d[i] * x_i;
            y *= silu_dispatch(z_i, math_backend);
            scan_out_b[i] = y;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
#[inline(never)]
unsafe fn ssm_update_avx512(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    batch: usize,
    conv_out: &[f32],
    z: &[f32],
    delta: &[f32],
    b_vec: &[f32],
    c_vec: &[f32],
    a_pre: &[f32],
    math_backend: MathBackend,
    scan_out: &mut [f32],
    layer_state: &mut [f32],
    fast_stats: &mut Option<&mut FastMathStats>,
) {
    use std::arch::x86_64::*;
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let layer_state_stride = d_inner * d_state_pad;
    #[cfg(feature = "bench_instrument")]
    let fast_math = matches!(
        math_backend,
        MathBackend::Approx | MathBackend::Sleef | MathBackend::FastBf16 | MathBackend::FastWild
    );
    #[cfg(not(feature = "bench_instrument"))]
    let _ = fast_stats;

    for b in 0..batch {
        let conv_b = slice_batched(conv_out, batch, d_inner, b);
        let z_b = slice_batched(z, batch, d_inner, b);
        let delta_b = slice_batched(delta, batch, d_inner, b);
        let b_vec_b = slice_batched(b_vec, batch, d_state_pad, b);
        let c_vec_b = slice_batched(c_vec, batch, d_state_pad, b);
        let scan_out_b = slice_batched_mut(scan_out, batch, d_inner, b);
        let state_b = &mut layer_state[b * layer_state_stride..(b + 1) * layer_state_stride];

        for i in 0..d_inner {
            let x_i = conv_b[i];
            let z_i = z_b[i];
            let delta_i = delta_b[i];
            let x_v = _mm512_set1_ps(x_i);
            let delta_v = _mm512_set1_ps(delta_i);
            let mut acc_v = _mm512_setzero_ps();
            let state_row = &mut state_b[i * d_state_pad..(i + 1) * d_state_pad];
            let a_row = &a_pre[i * d_state_pad..(i + 1) * d_state_pad];
            #[cfg(feature = "bench_instrument")]
            if b == 0 {
                sample_exp_stats(fast_math, fast_stats, delta_i, a_row, i, d_state);
            }

            let mut j = 0usize;
            while j + 16 <= d_state {
                let a = _mm512_loadu_ps(a_row.as_ptr().add(j));
                let x = _mm512_mul_ps(delta_v, a);
                let a_bar = exp512_ps(x);
                let h = _mm512_loadu_ps(state_row.as_ptr().add(j));
                let b_val = _mm512_loadu_ps(b_vec_b.as_ptr().add(j));
                let c_val = _mm512_loadu_ps(c_vec_b.as_ptr().add(j));
                let b_bar = _mm512_mul_ps(b_val, delta_v);
                let h_new = _mm512_fmadd_ps(a_bar, h, _mm512_mul_ps(b_bar, x_v));
                _mm512_storeu_ps(state_row.as_mut_ptr().add(j), h_new);
                acc_v = _mm512_fmadd_ps(c_val, h_new, acc_v);
                j += 16;
            }

            let mut acc = hsum512_ps(acc_v);
            while j < d_state {
                let a = a_row[j];
                let a_bar = libm::expf(delta_i * a);
                let b_bar = b_vec_b[j] * delta_i;
                let h = state_row[j];
                let h_new = a_bar * h + b_bar * x_i;
                state_row[j] = h_new;
                acc += c_vec_b[j] * h_new;
                j += 1;
            }

            let mut y = acc + lw.d[i] * x_i;
            y *= silu_dispatch(z_i, math_backend);
            scan_out_b[i] = y;
        }
    }
}

fn ssm_update_step1_fused_scalar(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    batch: usize,
    conv_out: &[f32],
    z: &[f32],
    x_proj_out: &[f32],
    a_pre: &[f32],
    math_backend: MathBackend,
    scan_out: &mut [f32],
    layer_state: &mut [f32],
    fast_stats: &mut Option<&mut FastMathStats>,
) {
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let dt_rank = cfg.dt_rank;
    let x_proj_dim = dt_rank + 2 * d_state_pad;
    let layer_state_stride = d_inner * d_state_pad;
    #[cfg(feature = "bench_instrument")]
    let fast_math = matches!(
        math_backend,
        MathBackend::Approx | MathBackend::Sleef | MathBackend::FastBf16 | MathBackend::FastWild
    );
    #[cfg(not(feature = "bench_instrument"))]
    let _ = fast_stats;

    for b in 0..batch {
        let conv_b = slice_batched(conv_out, batch, d_inner, b);
        let z_b = slice_batched(z, batch, d_inner, b);
        let x_proj_b = slice_batched(x_proj_out, batch, x_proj_dim, b);
        let dt_in = &x_proj_b[..dt_rank];
        let b_vec_b = &x_proj_b[dt_rank..dt_rank + d_state_pad];
        let c_vec_b = &x_proj_b[dt_rank + d_state_pad..dt_rank + 2 * d_state_pad];
        let scan_out_b = slice_batched_mut(scan_out, batch, d_inner, b);
        let state_b = &mut layer_state[b * layer_state_stride..(b + 1) * layer_state_stride];

        for i in 0..d_inner {
            let mut acc_dt = lw.dt_proj_b[i];
            let w_row = &lw.dt_proj_w[i * dt_rank..][..dt_rank];
            for k in 0..dt_rank {
                acc_dt += w_row[k] * dt_in[k];
            }
            let delta_i = softplus_dispatch(
                acc_dt,
                cfg.softplus_kind,
                cfg.softplus_beta,
                cfg.softplus_threshold,
                math_backend,
            );
            let x_i = conv_b[i];
            let z_i = z_b[i];
            let mut acc = 0.0f32;
            let state_row = &mut state_b[i * d_state_pad..(i + 1) * d_state_pad];
            let a_row = &a_pre[i * d_state_pad..(i + 1) * d_state_pad];
            #[cfg(feature = "bench_instrument")]
            if b == 0 {
                sample_exp_stats(fast_math, fast_stats, delta_i, a_row, i, d_state);
            }

            for j in 0..d_state {
                let a = a_row[j];
                let a_bar = libm::expf(delta_i * a);
                let b_bar = b_vec_b[j] * delta_i;
                let h = state_row[j];
                let h_new = a_bar * h + b_bar * x_i;
                state_row[j] = h_new;
                acc += c_vec_b[j] * h_new;
            }

            let mut y = acc + lw.d[i] * x_i;
            y *= silu_dispatch(z_i, math_backend);
            scan_out_b[i] = y;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ssm_update_step1_fused_avx2(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    batch: usize,
    conv_out: &[f32],
    z: &[f32],
    x_proj_out: &[f32],
    a_pre: &[f32],
    math_backend: MathBackend,
    scan_out: &mut [f32],
    layer_state: &mut [f32],
    fast_stats: &mut Option<&mut FastMathStats>,
) {
    use std::arch::x86_64::*;
    if cfg.dt_rank != 16 {
        ssm_update_step1_fused_scalar(
            cfg,
            lw,
            batch,
            conv_out,
            z,
            x_proj_out,
            a_pre,
            math_backend,
            scan_out,
            layer_state,
            fast_stats,
        );
        return;
    }
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let dt_rank = cfg.dt_rank;
    let x_proj_dim = dt_rank + 2 * d_state_pad;
    let layer_state_stride = d_inner * d_state_pad;
    #[cfg(feature = "bench_instrument")]
    let fast_math = matches!(
        math_backend,
        MathBackend::Approx | MathBackend::Sleef | MathBackend::FastBf16 | MathBackend::FastWild
    );
    #[cfg(not(feature = "bench_instrument"))]
    let _ = fast_stats;

    for b in 0..batch {
        let conv_b = slice_batched(conv_out, batch, d_inner, b);
        let z_b = slice_batched(z, batch, d_inner, b);
        let x_proj_b = slice_batched(x_proj_out, batch, x_proj_dim, b);
        let dt_in = x_proj_b.as_ptr();
        let b_vec_b = x_proj_b.as_ptr().add(dt_rank);
        let c_vec_b = x_proj_b.as_ptr().add(dt_rank + d_state_pad);
        let scan_out_b = slice_batched_mut(scan_out, batch, d_inner, b);
        let state_b = &mut layer_state[b * layer_state_stride..(b + 1) * layer_state_stride];

        for i in 0..d_inner {
            let w_row = lw.dt_proj_w.as_ptr().add(i * dt_rank);
            let dot = dot16_avx2(w_row, dt_in);
            let delta_i = softplus_dispatch(
                dot + *lw.dt_proj_b.get_unchecked(i),
                cfg.softplus_kind,
                cfg.softplus_beta,
                cfg.softplus_threshold,
                math_backend,
            );
            let x_i = *conv_b.get_unchecked(i);
            let z_i = *z_b.get_unchecked(i);
            let x_v = _mm256_set1_ps(x_i);
            let delta_v = _mm256_set1_ps(delta_i);
            let mut acc_v = _mm256_setzero_ps();
            let state_row = &mut state_b[i * d_state_pad..(i + 1) * d_state_pad];
            let a_row = &a_pre[i * d_state_pad..(i + 1) * d_state_pad];
            #[cfg(feature = "bench_instrument")]
            if b == 0 {
                sample_exp_stats(fast_math, fast_stats, delta_i, a_row, i, d_state);
            }

            let mut j = 0usize;
            while j + 8 <= d_state {
                let a = _mm256_loadu_ps(a_row.as_ptr().add(j));
                let x = _mm256_mul_ps(delta_v, a);
                let a_bar = exp256_ps(x);
                let h = _mm256_loadu_ps(state_row.as_ptr().add(j));
                let b_val = _mm256_loadu_ps(b_vec_b.add(j));
                let c_val = _mm256_loadu_ps(c_vec_b.add(j));
                let b_bar = _mm256_mul_ps(b_val, delta_v);
                let h_new = _mm256_fmadd_ps(a_bar, h, _mm256_mul_ps(b_bar, x_v));
                _mm256_storeu_ps(state_row.as_mut_ptr().add(j), h_new);
                acc_v = _mm256_fmadd_ps(c_val, h_new, acc_v);
                j += 8;
            }

            let mut acc = hsum256_ps(acc_v);
            while j < d_state {
                let a = *a_row.get_unchecked(j);
                let a_bar = libm::expf(delta_i * a);
                let b_bar = *b_vec_b.add(j) * delta_i;
                let h = *state_row.get_unchecked(j);
                let h_new = a_bar * h + b_bar * x_i;
                *state_row.get_unchecked_mut(j) = h_new;
                acc += *c_vec_b.add(j) * h_new;
                j += 1;
            }

            let mut y = acc + *lw.d.get_unchecked(i) * x_i;
            y *= silu_dispatch(z_i, math_backend);
            *scan_out_b.get_unchecked_mut(i) = y;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn ssm_update_step1_fused_avx512(
    cfg: &ModelConfig,
    lw: &LayerWeights,
    batch: usize,
    conv_out: &[f32],
    z: &[f32],
    x_proj_out: &[f32],
    a_pre: &[f32],
    math_backend: MathBackend,
    scan_out: &mut [f32],
    layer_state: &mut [f32],
    fast_stats: &mut Option<&mut FastMathStats>,
) {
    use std::arch::x86_64::*;
    if cfg.dt_rank != 16 {
        ssm_update_step1_fused_scalar(
            cfg,
            lw,
            batch,
            conv_out,
            z,
            x_proj_out,
            a_pre,
            math_backend,
            scan_out,
            layer_state,
            fast_stats,
        );
        return;
    }
    let d_inner = cfg.d_inner;
    let d_state = cfg.d_state;
    let d_state_pad = cfg.d_state_pad;
    let dt_rank = cfg.dt_rank;
    let x_proj_dim = dt_rank + 2 * d_state_pad;
    let layer_state_stride = d_inner * d_state_pad;
    #[cfg(feature = "bench_instrument")]
    let fast_math = matches!(
        math_backend,
        MathBackend::Approx | MathBackend::Sleef | MathBackend::FastBf16 | MathBackend::FastWild
    );
    #[cfg(not(feature = "bench_instrument"))]
    let _ = fast_stats;

    for b in 0..batch {
        let conv_b = slice_batched(conv_out, batch, d_inner, b);
        let z_b = slice_batched(z, batch, d_inner, b);
        let x_proj_b = slice_batched(x_proj_out, batch, x_proj_dim, b);
        let dt_in = x_proj_b.as_ptr();
        let b_vec_b = x_proj_b.as_ptr().add(dt_rank);
        let c_vec_b = x_proj_b.as_ptr().add(dt_rank + d_state_pad);
        let scan_out_b = slice_batched_mut(scan_out, batch, d_inner, b);
        let state_b = &mut layer_state[b * layer_state_stride..(b + 1) * layer_state_stride];

        for i in 0..d_inner {
            let w_row = lw.dt_proj_w.as_ptr().add(i * dt_rank);
            let dot = dot16_avx512(w_row, dt_in);
            let delta_i = softplus_dispatch(
                dot + *lw.dt_proj_b.get_unchecked(i),
                cfg.softplus_kind,
                cfg.softplus_beta,
                cfg.softplus_threshold,
                math_backend,
            );
            let x_i = *conv_b.get_unchecked(i);
            let z_i = *z_b.get_unchecked(i);
            let x_v = _mm512_set1_ps(x_i);
            let delta_v = _mm512_set1_ps(delta_i);
            let mut acc_v = _mm512_setzero_ps();
            let state_row = &mut state_b[i * d_state_pad..(i + 1) * d_state_pad];
            let a_row = &a_pre[i * d_state_pad..(i + 1) * d_state_pad];
            #[cfg(feature = "bench_instrument")]
            if b == 0 {
                sample_exp_stats(fast_math, fast_stats, delta_i, a_row, i, d_state);
            }

            let mut j = 0usize;
            while j + 16 <= d_state {
                let a = _mm512_loadu_ps(a_row.as_ptr().add(j));
                let x = _mm512_mul_ps(delta_v, a);
                let a_bar = exp512_ps(x);
                let h = _mm512_loadu_ps(state_row.as_ptr().add(j));
                let b_val = _mm512_loadu_ps(b_vec_b.add(j));
                let c_val = _mm512_loadu_ps(c_vec_b.add(j));
                let b_bar = _mm512_mul_ps(b_val, delta_v);
                let h_new = _mm512_fmadd_ps(a_bar, h, _mm512_mul_ps(b_bar, x_v));
                _mm512_storeu_ps(state_row.as_mut_ptr().add(j), h_new);
                acc_v = _mm512_fmadd_ps(c_val, h_new, acc_v);
                j += 16;
            }

            let mut acc = hsum512_ps(acc_v);
            while j < d_state {
                let a = *a_row.get_unchecked(j);
                let a_bar = libm::expf(delta_i * a);
                let b_bar = *b_vec_b.add(j) * delta_i;
                let h = *state_row.get_unchecked(j);
                let h_new = a_bar * h + b_bar * x_i;
                *state_row.get_unchecked_mut(j) = h_new;
                acc += *c_vec_b.add(j) * h_new;
                j += 1;
            }

            let mut y = acc + *lw.d.get_unchecked(i) * x_i;
            y *= silu_dispatch(z_i, math_backend);
            *scan_out_b.get_unchecked_mut(i) = y;
        }
    }
}

pub fn step1_block_latency(
    cfg: &ModelConfig,
    w: &WeightsView,
    dispatch: CpuDispatch,
    math_backend: MathBackend,
    layer: usize,
    x_in: &[f32],
    state: &mut [f32],
    scratch: &mut Scratch,
    x_out: &mut [f32],
    stats: Option<&mut StageStats>,
    fast_stats: Option<&mut FastMathStats>,
) -> Result<(), KernelError> {
    if layer >= cfg.n_layers {
        return Err(KernelError::BadLen("layer index"));
    }
    let x_len = cfg.d_model;
    expect_len(x_in.len(), x_len, "x_in")?;
    expect_len(x_out.len(), x_len, "x_out")?;
    let state_len = cfg.n_layers * cfg.d_inner * cfg.d_state_pad;
    expect_len(state.len(), state_len, "state")?;
    #[cfg(not(feature = "bench_instrument"))]
    let _ = fast_stats;

    let lw = &w.layers[layer];
    let mut stats = stats;

    let mut ln_res = Ok(());
    record_stage(&mut stats, Stage::InProj, || {
        ln_res = ln1_in_proj_fused_step1(
            cfg,
            lw,
            dispatch,
            math_backend,
            x_in,
            scratch.ln1_out,
            scratch.in_proj_out,
        );
    });
    ln_res?;

    // split x/z
    for i in 0..cfg.d_inner {
        unsafe {
            *scratch.x.get_unchecked_mut(i) = *scratch.in_proj_out.get_unchecked(i);
            *scratch.z.get_unchecked_mut(i) = *scratch.in_proj_out.get_unchecked(cfg.d_inner + i);
        }
    }

    // conv + silu (depthwise, seq_len=1, last tap)
    record_stage(&mut stats, Stage::Conv1dStep1, || {
        let tap_idx = cfg.conv_kernel - 1;
        for i in 0..cfg.d_inner {
            unsafe {
                let w_idx = i * cfg.conv_kernel + tap_idx;
                let v = *lw.conv_b.get_unchecked(i) + *lw.conv_w.get_unchecked(w_idx) * *scratch.x.get_unchecked(i);
                *scratch.conv_out.get_unchecked_mut(i) = silu_dispatch(v, math_backend);
            }
        }
    });

    // x_proj
    record_stage(&mut stats, Stage::XProj, || {
        let bias = lw
            .x_proj_b_zero
            .as_deref()
            .expect("x_proj bias zero");
        matmul_step1_packed(
            lw.x_proj_w,
            lw.x_proj_w_packed.as_deref(),
            lw.x_proj_w_packed_bf16.as_deref(),
            cfg.dt_rank + 2 * cfg.d_state_pad,
            cfg.d_inner,
            scratch.conv_out,
            bias,
            scratch.x_proj_out,
            dispatch,
            math_backend,
        );
    });

    // scan (dt proj + softplus fused)
    record_stage(&mut stats, Stage::SsmUpdate, || {
        let a_pre: &[f32] = if let Some(pre) = lw.a_pre.as_deref() {
            pre
        } else {
            let tmp = &mut scratch.a_pre[..cfg.d_inner * cfg.d_state_pad];
            fill_a_pre(lw.a_log, tmp);
            tmp
        };
        let mut fast_stats = fast_stats;
        let layer_state = layer_state_slice(cfg, 1, layer, state);
        match dispatch {
            CpuDispatch::Scalar => ssm_update_step1_fused_scalar(
                cfg,
                lw,
                1,
                scratch.conv_out,
                scratch.z,
                scratch.x_proj_out,
                a_pre,
                math_backend,
                scratch.scan_out,
                layer_state,
                &mut fast_stats,
            ),
            CpuDispatch::Avx2 => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if std::is_x86_feature_detected!("avx2") {
                        ssm_update_step1_fused_avx2(
                            cfg,
                            lw,
                            1,
                            scratch.conv_out,
                            scratch.z,
                            scratch.x_proj_out,
                            a_pre,
                            math_backend,
                            scratch.scan_out,
                            layer_state,
                            &mut fast_stats,
                        );
                    } else {
                        ssm_update_step1_fused_scalar(
                            cfg,
                            lw,
                            1,
                            scratch.conv_out,
                            scratch.z,
                            scratch.x_proj_out,
                            a_pre,
                            math_backend,
                            scratch.scan_out,
                            layer_state,
                            &mut fast_stats,
                        );
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                ssm_update_step1_fused_scalar(
                    cfg,
                    lw,
                    1,
                    scratch.conv_out,
                    scratch.z,
                    scratch.x_proj_out,
                    a_pre,
                    math_backend,
                    scratch.scan_out,
                    layer_state,
                    &mut fast_stats,
                );
            }
            CpuDispatch::Avx512 => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if std::is_x86_feature_detected!("avx512f") {
                        ssm_update_step1_fused_avx512(
                            cfg,
                            lw,
                            1,
                            scratch.conv_out,
                            scratch.z,
                            scratch.x_proj_out,
                            a_pre,
                            math_backend,
                            scratch.scan_out,
                            layer_state,
                            &mut fast_stats,
                        );
                    } else {
                        ssm_update_step1_fused_scalar(
                            cfg,
                            lw,
                            1,
                            scratch.conv_out,
                            scratch.z,
                            scratch.x_proj_out,
                            a_pre,
                            math_backend,
                            scratch.scan_out,
                            layer_state,
                            &mut fast_stats,
                        );
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                ssm_update_step1_fused_scalar(
                    cfg,
                    lw,
                    1,
                    scratch.conv_out,
                    scratch.z,
                    scratch.x_proj_out,
                    a_pre,
                    math_backend,
                    scratch.scan_out,
                    layer_state,
                    &mut fast_stats,
                );
            }
        }
    });

    // out_proj
    record_stage(&mut stats, Stage::OutProj, || {
        let bias = lw
            .out_proj_b
            .or_else(|| lw.out_proj_b_zero.as_deref())
            .expect("out_proj bias");
        matmul_step1_packed(
            lw.out_proj_w,
            lw.out_proj_w_packed.as_deref(),
            lw.out_proj_w_packed_bf16.as_deref(),
            cfg.d_model,
            cfg.d_inner,
            scratch.scan_out,
            bias,
            scratch.out_proj_out,
            dispatch,
            math_backend,
        );
    });

    // residual1
    record_stage(&mut stats, Stage::Resid1, || {
        for i in 0..cfg.d_model {
            unsafe {
                *scratch.resid1_out.get_unchecked_mut(i) =
                    *x_in.get_unchecked(i) + *scratch.out_proj_out.get_unchecked(i);
            }
        }
    });

    // ln2
    record_stage(&mut stats, Stage::Ln2, || {
        layer_norm_dispatch(
            scratch.resid1_out,
            lw.ln2_w,
            lw.ln2_b,
            cfg.ln_eps2,
            scratch.ln2_out,
            dispatch,
        );
    });

    // MLP fused (seq_len=1)
    let use_mlp_fused = mlp_fused_enabled(1, false, math_backend);
    if use_mlp_fused {
        record_stage(&mut stats, Stage::MlpFc1, || {
            mlp_fused_streamed(cfg, lw, dispatch, math_backend, scratch.ln2_out, scratch.mlp_fc2);
        });
    } else {
        record_stage(&mut stats, Stage::MlpFc1, || {
            matmul_step1_packed(
                lw.fc1_w,
                lw.fc1_w_packed.as_deref(),
                lw.fc1_w_packed_bf16.as_deref(),
                cfg.d_mlp,
                cfg.d_model,
                scratch.ln2_out,
                lw.fc1_b,
                scratch.mlp_fc1,
                dispatch,
                math_backend,
            );
        });
        record_stage(&mut stats, Stage::Gelu, || {
            for i in 0..cfg.d_mlp {
                unsafe {
                    *scratch.mlp_gelu.get_unchecked_mut(i) = gelu_dispatch(
                        *scratch.mlp_fc1.get_unchecked(i),
                        cfg.gelu_kind,
                        math_backend,
                    );
                }
            }
        });
        record_stage(&mut stats, Stage::MlpFc2, || {
            matmul_step1_packed(
                lw.fc2_w,
                lw.fc2_w_packed.as_deref(),
                lw.fc2_w_packed_bf16.as_deref(),
                cfg.d_model,
                cfg.d_mlp,
                scratch.mlp_gelu,
                lw.fc2_b,
                scratch.mlp_fc2,
                dispatch,
                math_backend,
            );
        });
    }

    // residual2
    record_stage(&mut stats, Stage::Resid2, || {
        for i in 0..cfg.d_model {
            unsafe {
                *scratch.resid2_out.get_unchecked_mut(i) =
                    *scratch.resid1_out.get_unchecked(i) + *scratch.mlp_fc2.get_unchecked(i);
            }
        }
    });

    x_out.copy_from_slice(scratch.resid2_out);
    Ok(())
}

pub fn step1_model_latency(
    cfg: &ModelConfig,
    w: &WeightsView,
    dispatch: CpuDispatch,
    math_backend: MathBackend,
    batch: usize,
    x_in: &[f32],
    state: &mut [f32],
    scratch_bytes: &mut [u8],
    logits_out: &mut [f32],
    margin_out: &mut [f32],
    stats: Option<&mut StageStats>,
    fast_stats: Option<&mut FastMathStats>,
) -> Result<(), KernelError> {
    check_dispatch(dispatch)?;
    if batch != 1 {
        return Err(KernelError::InvalidConfig("step1 latency batch != 1"));
    }
    let mut scratch = scratch_from_bytes(cfg, batch, scratch_bytes)?;
    let x_len = cfg.d_model;
    expect_len(x_in.len(), x_len, "x_in")?;
    expect_len(logits_out.len(), cfg.n_class, "logits_out")?;
    expect_len(margin_out.len(), batch, "margin_out")?;
    let state_len = cfg.n_layers * cfg.d_inner * cfg.d_state_pad;
    expect_len(state.len(), state_len, "state")?;

    scratch.x_cur[..x_len].copy_from_slice(x_in);
    let mut stats = stats;
    let mut fast_stats = fast_stats;

    for layer in 0..cfg.n_layers {
        let x_in_ptr = scratch.x_cur.as_ptr();
        let x_out_ptr = scratch.x_cur.as_mut_ptr();
        let x_in_slice = unsafe { std::slice::from_raw_parts(x_in_ptr, x_len) };
        let x_out_slice = unsafe { std::slice::from_raw_parts_mut(x_out_ptr, x_len) };
        step1_block_latency(
            cfg,
            w,
            dispatch,
            math_backend,
            layer,
            x_in_slice,
            state,
            &mut scratch,
            x_out_slice,
            stats.as_deref_mut(),
            fast_stats.as_deref_mut(),
        )?;
    }

    // final norm
    record_stage(&mut stats, Stage::FinalNorm, || {
        layer_norm_dispatch(
            scratch.x_cur,
            w.norm_w,
            w.norm_b,
            cfg.ln_eps2,
            scratch.final_norm_out,
            dispatch,
        );
    });

    // head_in: l2norm (no epsilon)
    record_stage(&mut stats, Stage::HeadL2Norm, || {
        let norm = l2norm(scratch.final_norm_out);
        if norm == 0.0 {
            for v in scratch.head_in.iter_mut() {
                *v = 0.0;
            }
        } else {
            for i in 0..cfg.d_model {
                scratch.head_in[i] = scratch.final_norm_out[i] / norm;
            }
        }
    });

    // head
    let head_w = w.head_w;
    let head_b = w.head_b;
    record_stage(&mut stats, Stage::HeadMatmul, || {
        if cfg.n_class == 2 {
            head_matmul_two_class(head_w, head_b, scratch.head_in, logits_out, dispatch);
        } else {
            matmul_proj(
                head_w,
                w.head_w_packed.as_deref(),
                cfg.n_class,
                cfg.d_model,
                scratch.head_in,
                Some(head_b),
                None,
                logits_out,
                dispatch,
            );
        }
    });

    margin_out[0] = logits_out[1];
    Ok(())
}

pub fn step1_block_fused(
    cfg: &ModelConfig,
    w: &WeightsView,
    dispatch: CpuDispatch,
    math_backend: MathBackend,
    batch: usize,
    layer: usize,
    x_in: &[f32],
    state: &mut [f32],
    scratch: &mut Scratch,
    x_out: &mut [f32],
    mut dbg: Option<&mut dyn DbgTaps>,
    stats: Option<&mut StageStats>,
    fast_stats: Option<&mut FastMathStats>,
) -> Result<(), KernelError> {
    check_dispatch(dispatch)?;
    if layer >= cfg.n_layers {
        return Err(KernelError::BadLen("layer index"));
    }
    #[cfg(not(feature = "bench_instrument"))]
    let _ = fast_stats;
    let x_len = batch * cfg.d_model;
    expect_len(x_in.len(), x_len, "x_in")?;
    expect_len(x_out.len(), x_len, "x_out")?;
    let state_len = batch * cfg.n_layers * cfg.d_inner * cfg.d_state_pad;
    expect_len(state.len(), state_len, "state")?;

    let lw = &w.layers[layer];
    let mut stats = stats;
    let fast_math = matches!(
        math_backend,
        MathBackend::Approx | MathBackend::Sleef | MathBackend::FastBf16 | MathBackend::FastWild
    );

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(&format!("layer{}/x_in", layer), &[batch, cfg.d_model], x_in)?;
    }

    let layer_state = layer_state_slice(cfg, batch, layer, state);
    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/state_in", layer),
            &[batch, cfg.d_inner, cfg.d_state_pad],
            layer_state,
        )?;
    }

    record_stage(&mut stats, Stage::Ln1, || {
        for b in 0..batch {
            let x_in_b = slice_batched(x_in, batch, cfg.d_model, b);
            let ln1_out_b = slice_batched_mut(scratch.ln1_out, batch, cfg.d_model, b);
            layer_norm_dispatch(x_in_b, lw.ln1_w, lw.ln1_b, cfg.ln_eps1, ln1_out_b, dispatch);
        }
    });

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/ln1_out", layer),
            &[batch, cfg.d_model],
            scratch.ln1_out,
        )?;
    }

    // in_proj
    record_stage(&mut stats, Stage::InProj, || {
        for b in 0..batch {
            let ln1_out_b = slice_batched(scratch.ln1_out, batch, cfg.d_model, b);
            let in_proj_out_b = slice_batched_mut(scratch.in_proj_out, batch, 2 * cfg.d_inner, b);
            matmul_proj(
                lw.in_proj_w,
                lw.in_proj_w_packed.as_deref(),
                2 * cfg.d_inner,
                cfg.d_model,
                ln1_out_b,
                lw.in_proj_b,
                lw.in_proj_b_zero.as_deref(),
                in_proj_out_b,
                dispatch,
            );
        }
    });

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/mamba/in_proj_out", layer),
            &[batch, 2 * cfg.d_inner],
            scratch.in_proj_out,
        )?;
    }

    // split x/z
    for b in 0..batch {
        let in_proj_out_b = slice_batched(scratch.in_proj_out, batch, 2 * cfg.d_inner, b);
        let x_b = slice_batched_mut(scratch.x, batch, cfg.d_inner, b);
        let z_b = slice_batched_mut(scratch.z, batch, cfg.d_inner, b);
        x_b.copy_from_slice(&in_proj_out_b[..cfg.d_inner]);
        z_b.copy_from_slice(&in_proj_out_b[cfg.d_inner..]);
    }

    // conv + silu (depthwise, seq_len=1, last tap)
    record_stage(&mut stats, Stage::Conv1dStep1, || {
        let tap_idx = cfg.conv_kernel - 1;
        for b in 0..batch {
            let x_b = slice_batched(scratch.x, batch, cfg.d_inner, b);
            let conv_out_b = slice_batched_mut(scratch.conv_out, batch, cfg.d_inner, b);
            for i in 0..cfg.d_inner {
                let w_idx = i * cfg.conv_kernel + tap_idx;
                let v = lw.conv_b[i] + lw.conv_w[w_idx] * x_b[i];
                conv_out_b[i] = silu_dispatch(v, math_backend);
            }
        }
    });

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/mamba/conv_out", layer),
            &[batch, cfg.d_inner],
            scratch.conv_out,
        )?;
    }

    // x_proj
    let x_proj_dim = cfg.dt_rank + 2 * cfg.d_state_pad;
    record_stage(&mut stats, Stage::XProj, || {
        for b in 0..batch {
            let conv_b = slice_batched(scratch.conv_out, batch, cfg.d_inner, b);
            let x_proj_out_b = slice_batched_mut(scratch.x_proj_out, batch, x_proj_dim, b);
            matmul_proj(
                lw.x_proj_w,
                lw.x_proj_w_packed.as_deref(),
                x_proj_dim,
                cfg.d_inner,
                conv_b,
                None,
                lw.x_proj_b_zero.as_deref(),
                x_proj_out_b,
                dispatch,
            );
        }
    });

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/mamba/x_proj_out", layer),
            &[batch, x_proj_dim],
            scratch.x_proj_out,
        )?;
    }

    // dt_in, B, C
    for b in 0..batch {
        let x_proj_out_b = slice_batched(scratch.x_proj_out, batch, x_proj_dim, b);
        let dt_in_b = slice_batched_mut(scratch.dt_in, batch, cfg.dt_rank, b);
        let b_vec_b = slice_batched_mut(scratch.b_vec, batch, cfg.d_state_pad, b);
        let c_vec_b = slice_batched_mut(scratch.c_vec, batch, cfg.d_state_pad, b);

        dt_in_b.copy_from_slice(&x_proj_out_b[..cfg.dt_rank]);
        b_vec_b.copy_from_slice(&x_proj_out_b[cfg.dt_rank..cfg.dt_rank + cfg.d_state_pad]);
        c_vec_b.copy_from_slice(
            &x_proj_out_b[cfg.dt_rank + cfg.d_state_pad..cfg.dt_rank + 2 * cfg.d_state_pad],
        );
    }

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/mamba/dt_in", layer),
            &[batch, cfg.dt_rank],
            scratch.dt_in,
        )?;
        tap.tap_f32(
            &format!("layer{}/mamba/B", layer),
            &[batch, cfg.d_state_pad],
            scratch.b_vec,
        )?;
        tap.tap_f32(
            &format!("layer{}/mamba/C", layer),
            &[batch, cfg.d_state_pad],
            scratch.c_vec,
        )?;
    }

    // delta
    record_stage(&mut stats, Stage::DtProjSoftplus, || {
        for b in 0..batch {
            let dt_in_b = slice_batched(scratch.dt_in, batch, cfg.dt_rank, b);
            let delta_b = slice_batched_mut(scratch.delta, batch, cfg.d_inner, b);
            matmul_proj(
                lw.dt_proj_w,
                lw.dt_proj_w_packed.as_deref(),
                cfg.d_inner,
                cfg.dt_rank,
                dt_in_b,
                Some(lw.dt_proj_b),
                None,
                delta_b,
                dispatch,
            );
            for i in 0..cfg.d_inner {
                delta_b[i] = softplus_dispatch(
                    delta_b[i],
                    cfg.softplus_kind,
                    cfg.softplus_beta,
                    cfg.softplus_threshold,
                    math_backend,
                );
            }
        }
    });

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/mamba/delta", layer),
            &[batch, cfg.d_inner],
            scratch.delta,
        )?;
    }

    // scan
    record_stage(&mut stats, Stage::SsmUpdate, || {
        let a_pre = if let Some(pre) = lw.a_pre.as_deref() {
            pre
        } else {
            let tmp = &mut scratch.a_pre[..cfg.d_inner * cfg.d_state_pad];
            fill_a_pre(lw.a_log, tmp);
            tmp
        };

        let use_fast = fast_math;
        #[cfg(feature = "bench_instrument")]
        let mut fast_stats = if fast_math { fast_stats } else { None };
        #[cfg(not(feature = "bench_instrument"))]
        let mut fast_stats: Option<&mut FastMathStats> = None;
        match dispatch {
            CpuDispatch::Scalar => ssm_update_scalar(
                cfg,
                lw,
                batch,
                scratch.conv_out,
                scratch.z,
                scratch.delta,
                scratch.b_vec,
                scratch.c_vec,
                a_pre,
                math_backend,
                scratch.scan_out,
                layer_state,
                &mut fast_stats,
            ),
            CpuDispatch::Avx2 => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if use_fast {
                        ssm_update_avx2(
                            cfg,
                            lw,
                            batch,
                            scratch.conv_out,
                            scratch.z,
                            scratch.delta,
                            scratch.b_vec,
                            scratch.c_vec,
                            a_pre,
                            math_backend,
                            scratch.scan_out,
                            layer_state,
                            &mut fast_stats,
                        );
                    } else {
                        ssm_update_scalar(
                            cfg,
                            lw,
                            batch,
                            scratch.conv_out,
                            scratch.z,
                            scratch.delta,
                            scratch.b_vec,
                            scratch.c_vec,
                            a_pre,
                            math_backend,
                            scratch.scan_out,
                            layer_state,
                            &mut fast_stats,
                        );
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                ssm_update_scalar(
                    cfg,
                    lw,
                    batch,
                    scratch.conv_out,
                    scratch.z,
                    scratch.delta,
                    scratch.b_vec,
                    scratch.c_vec,
                    a_pre,
                    math_backend,
                    scratch.scan_out,
                    layer_state,
                    &mut fast_stats,
                );
            }
            CpuDispatch::Avx512 => {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if use_fast {
                        ssm_update_avx512(
                            cfg,
                            lw,
                            batch,
                            scratch.conv_out,
                            scratch.z,
                            scratch.delta,
                            scratch.b_vec,
                            scratch.c_vec,
                            a_pre,
                            math_backend,
                            scratch.scan_out,
                            layer_state,
                            &mut fast_stats,
                        );
                    } else {
                        ssm_update_scalar(
                            cfg,
                            lw,
                            batch,
                            scratch.conv_out,
                            scratch.z,
                            scratch.delta,
                            scratch.b_vec,
                            scratch.c_vec,
                            a_pre,
                            math_backend,
                            scratch.scan_out,
                            layer_state,
                            &mut fast_stats,
                        );
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                ssm_update_scalar(
                    cfg,
                    lw,
                    batch,
                    scratch.conv_out,
                    scratch.z,
                    scratch.delta,
                    scratch.b_vec,
                    scratch.c_vec,
                    a_pre,
                    math_backend,
                    scratch.scan_out,
                    layer_state,
                    &mut fast_stats,
                );
            }
        }
    });

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/mamba/scan_out", layer),
            &[batch, cfg.d_inner],
            scratch.scan_out,
        )?;
    }

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/state_out", layer),
            &[batch, cfg.d_inner, cfg.d_state_pad],
            layer_state,
        )?;
    }

    // out_proj
    record_stage(&mut stats, Stage::OutProj, || {
        for b in 0..batch {
            let scan_out_b = slice_batched(scratch.scan_out, batch, cfg.d_inner, b);
            let out_proj_b = slice_batched_mut(scratch.out_proj_out, batch, cfg.d_model, b);
            matmul_proj(
                lw.out_proj_w,
                lw.out_proj_w_packed.as_deref(),
                cfg.d_model,
                cfg.d_inner,
                scan_out_b,
                lw.out_proj_b,
                lw.out_proj_b_zero.as_deref(),
                out_proj_b,
                dispatch,
            );
        }
    });

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/mamba/out_proj_out", layer),
            &[batch, cfg.d_model],
            scratch.out_proj_out,
        )?;
    }

    // residual1
    record_stage(&mut stats, Stage::Resid1, || {
        for b in 0..batch {
            let x_in_b = slice_batched(x_in, batch, cfg.d_model, b);
            let out_proj_b = slice_batched(scratch.out_proj_out, batch, cfg.d_model, b);
            let resid1_b = slice_batched_mut(scratch.resid1_out, batch, cfg.d_model, b);
            for i in 0..cfg.d_model {
                resid1_b[i] = x_in_b[i] + out_proj_b[i];
            }
        }
    });

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/residual1_out", layer),
            &[batch, cfg.d_model],
            scratch.resid1_out,
        )?;
    }

    // ln2
    record_stage(&mut stats, Stage::Ln2, || {
        for b in 0..batch {
            let resid1_b = slice_batched(scratch.resid1_out, batch, cfg.d_model, b);
            let ln2_b = slice_batched_mut(scratch.ln2_out, batch, cfg.d_model, b);
            layer_norm_dispatch(resid1_b, lw.ln2_w, lw.ln2_b, cfg.ln_eps2, ln2_b, dispatch);
        }
    });

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/ln2_out", layer),
            &[batch, cfg.d_model],
            scratch.ln2_out,
        )?;
    }

    // MLP fc1
    record_stage(&mut stats, Stage::MlpFc1, || {
        for b in 0..batch {
            let ln2_b = slice_batched(scratch.ln2_out, batch, cfg.d_model, b);
            let fc1_b = slice_batched_mut(scratch.mlp_fc1, batch, cfg.d_mlp, b);
            matmul_proj(
                lw.fc1_w,
                lw.fc1_w_packed.as_deref(),
                cfg.d_mlp,
                cfg.d_model,
                ln2_b,
                Some(lw.fc1_b),
                None,
                fc1_b,
                dispatch,
            );
        }
    });

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/mlp/fc1_out", layer),
            &[batch, cfg.d_mlp],
            scratch.mlp_fc1,
        )?;
    }

    // GELU
    record_stage(&mut stats, Stage::Gelu, || {
        for b in 0..batch {
            let fc1_b = slice_batched(scratch.mlp_fc1, batch, cfg.d_mlp, b);
            let gelu_b = slice_batched_mut(scratch.mlp_gelu, batch, cfg.d_mlp, b);
            for i in 0..cfg.d_mlp {
                gelu_b[i] = gelu_dispatch(fc1_b[i], cfg.gelu_kind, math_backend);
            }
        }
    });

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/mlp/gelu_out", layer),
            &[batch, cfg.d_mlp],
            scratch.mlp_gelu,
        )?;
    }

    // fc2
    record_stage(&mut stats, Stage::MlpFc2, || {
        for b in 0..batch {
            let gelu_b = slice_batched(scratch.mlp_gelu, batch, cfg.d_mlp, b);
            let fc2_b = slice_batched_mut(scratch.mlp_fc2, batch, cfg.d_model, b);
            matmul_proj(
                lw.fc2_w,
                lw.fc2_w_packed.as_deref(),
                cfg.d_model,
                cfg.d_mlp,
                gelu_b,
                Some(lw.fc2_b),
                None,
                fc2_b,
                dispatch,
            );
        }
    });

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/mlp/fc2_out", layer),
            &[batch, cfg.d_model],
            scratch.mlp_fc2,
        )?;
    }

    // residual2
    record_stage(&mut stats, Stage::Resid2, || {
        for b in 0..batch {
            let resid1_b = slice_batched(scratch.resid1_out, batch, cfg.d_model, b);
            let fc2_b = slice_batched(scratch.mlp_fc2, batch, cfg.d_model, b);
            let resid2_b = slice_batched_mut(scratch.resid2_out, batch, cfg.d_model, b);
            for i in 0..cfg.d_model {
                resid2_b[i] = resid1_b[i] + fc2_b[i];
            }
        }
    });

    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32(
            &format!("layer{}/residual2_out", layer),
            &[batch, cfg.d_model],
            scratch.resid2_out,
        )?;
    }

    // output
    x_out.copy_from_slice(scratch.resid2_out);

    Ok(())
}

pub fn step1_model(
    cfg: &ModelConfig,
    w: &WeightsView,
    dispatch: CpuDispatch,
    math_backend: MathBackend,
    batch: usize,
    x_in: &[f32],
    state: &mut [f32],
    scratch_bytes: &mut [u8],
    logits_out: &mut [f32],
    margin_out: &mut [f32],
    dbg: Option<&mut dyn DbgTaps>,
    stats: Option<&mut StageStats>,
    fast_stats: Option<&mut FastMathStats>,
) -> Result<(), KernelError> {
    let mut scratch = scratch_from_bytes(cfg, batch, scratch_bytes)?;
    let x_len = batch * cfg.d_model;
    expect_len(x_in.len(), x_len, "x_in")?;
    expect_len(logits_out.len(), batch * cfg.n_class, "logits_out")?;
    expect_len(margin_out.len(), batch, "margin_out")?;

    // working buffer uses x_cur
    scratch.x_cur[..x_len].copy_from_slice(x_in);

    let _has_dbg = dbg.is_some();
    let dbg_ptr = dbg.map(|tap| tap as *mut dyn DbgTaps);
    let mut stats = stats;
    let stats_ptr = stats.as_deref_mut().map(|s| s as *mut StageStats);
    let mut fast_stats = fast_stats;
    let fast_stats_ptr = fast_stats.as_deref_mut().map(|s| s as *mut FastMathStats);
    if let Some(ptr) = dbg_ptr {
        unsafe {
            (&mut *ptr).tap_f32("x_in", &[batch, cfg.d_model], x_in)?;
        }
    }

    for layer in 0..cfg.n_layers {
        let x_in_ptr = scratch.x_cur.as_ptr();
        let x_out_ptr = scratch.x_cur.as_mut_ptr();
        let x_in_slice = unsafe { std::slice::from_raw_parts(x_in_ptr, x_len) };
        let x_out_slice = unsafe { std::slice::from_raw_parts_mut(x_out_ptr, x_len) };
        let layer_dbg = dbg_ptr.map(|ptr| unsafe { &mut *ptr });
        let layer_stats = stats_ptr.map(|ptr| unsafe { &mut *ptr });
        let mut layer_fast_stats = fast_stats_ptr.map(|ptr| unsafe { &mut *ptr });
        #[cfg(not(feature = "bench_instrument"))]
        let _ = &mut layer_fast_stats;

        step1_block_fused(
            cfg,
            w,
            dispatch,
            math_backend,
            batch,
            layer,
            x_in_slice,
            state,
            &mut scratch,
            x_out_slice,
            layer_dbg,
            layer_stats,
            layer_fast_stats,
        )?;

    }

    // final norm
    record_stage(&mut stats, Stage::FinalNorm, || {
        for b in 0..batch {
            let resid2_b = slice_batched(scratch.x_cur, batch, cfg.d_model, b);
            let final_norm_b = slice_batched_mut(scratch.final_norm_out, batch, cfg.d_model, b);
            layer_norm_dispatch(resid2_b, w.norm_w, w.norm_b, cfg.ln_eps2, final_norm_b, dispatch);
        }
    });

    if let Some(ptr) = dbg_ptr {
        unsafe {
            (&mut *ptr).tap_f32("final_norm_out", &[batch, cfg.d_model], scratch.final_norm_out)?;
        }
    }

    // head_in: l2norm (no epsilon)
    record_stage(&mut stats, Stage::HeadL2Norm, || {
        for b in 0..batch {
            let final_norm_b = slice_batched(scratch.final_norm_out, batch, cfg.d_model, b);
            let head_in_b = slice_batched_mut(scratch.head_in, batch, cfg.d_model, b);
            let norm = l2norm(final_norm_b);
            if norm == 0.0 {
                for v in head_in_b.iter_mut() {
                    *v = 0.0;
                }
            } else {
                for i in 0..cfg.d_model {
                    head_in_b[i] = final_norm_b[i] / norm;
                }
            }
        }
    });

    if let Some(ptr) = dbg_ptr {
        unsafe {
            (&mut *ptr).tap_f32("head_in", &[batch, cfg.d_model], scratch.head_in)?;
        }
    }

    // head
    let head_w = w.head_w;
    let head_b = w.head_b;
    record_stage(&mut stats, Stage::HeadMatmul, || {
        for b in 0..batch {
            let head_in_b = slice_batched(scratch.head_in, batch, cfg.d_model, b);
            let logits_b = slice_batched_mut(logits_out, batch, cfg.n_class, b);
            if cfg.n_class == 2 {
                head_matmul_two_class(head_w, head_b, head_in_b, logits_b, dispatch);
            } else {
                matmul_proj(
                    head_w,
                    w.head_w_packed.as_deref(),
                    cfg.n_class,
                    cfg.d_model,
                    head_in_b,
                    Some(head_b),
                    None,
                    logits_b,
                    dispatch,
                );
            }
        }
    });

    for b in 0..batch {
        let logits_b = slice_batched(logits_out, batch, cfg.n_class, b);
        margin_out[b] = logits_b[1];
    }

    Ok(())
}

fn embed_sum_ids(
    emb_w: &[f32],
    vocab_size: usize,
    d_model: usize,
    seq_len: usize,
    feature_dim: usize,
    input_ids: &[i64],
    out: &mut [f32],
) -> Result<(), KernelError> {
    expect_len(input_ids.len(), seq_len * feature_dim, "input_ids")?;
    expect_len(out.len(), seq_len * d_model, "x_cur")?;
    let vocab = if vocab_size == 0 {
        emb_w.len() / d_model
    } else {
        vocab_size
    };
    for t in 0..seq_len {
        let out_t = slice_seq_mut(out, d_model, t);
        for v in out_t.iter_mut() {
            *v = 0.0;
        }
        let base = t * feature_dim;
        for f in 0..feature_dim {
            let id = input_ids[base + f];
            let idx = if id >= 0 && (id as usize) < vocab {
                id as usize
            } else {
                0
            };
            let emb = &emb_w[idx * d_model..][..d_model];
            for i in 0..d_model {
                out_t[i] += emb[i];
            }
        }
    }
    Ok(())
}

pub fn full_stateless_forward(
    cfg: &ModelConfig,
    w: &WeightsView,
    dispatch: CpuDispatch,
    math_backend: MathBackend,
    input_ids: &[i64],
    static_total: &[f32],
    lengths: Option<&[i64]>,
    scratch_bytes: &mut [u8],
    logits_out: &mut [f32],
    margin_out: &mut [f32],
    mut dbg: Option<&mut dyn DbgTaps>,
    stats: Option<&mut StageStats>,
    mut perf: Option<&mut PerfCounters>,
    fast_stats: Option<&mut FastMathStats>,
) -> Result<(), KernelError> {
    if cfg.forward_kind != ForwardKind::FullStateless {
        return Err(KernelError::InvalidConfig("forward_kind != full_stateless"));
    }
    let seq_len = cfg.seq_len;
    if seq_len == 0 {
        return Err(KernelError::InvalidConfig("seq_len == 0"));
    }

    let emb_w = expect_weights(w.emb_w, "emb.weight")?;
    let pos_w = expect_weights(w.pos_w, "pos.weight")?;
    let static_proj_w = expect_weights(w.static_proj_w, "static_proj.weight")?;
    let static_proj_b = w.static_proj_b;

    expect_len(pos_w.len(), seq_len * cfg.d_model, "pos.weight")?;
    expect_len(static_total.len(), cfg.static_dim_total, "static_total")?;
    expect_len(logits_out.len(), cfg.n_class, "logits_out")?;
    expect_len(margin_out.len(), 1, "margin_out")?;

    let scratch = scratch_full_from_bytes(cfg, scratch_bytes)?;
    embed_sum_ids(
        emb_w,
        cfg.vocab_size,
        cfg.d_model,
        seq_len,
        cfg.feature_dim.max(1),
        input_ids,
        scratch.x_cur,
    )?;

    // add position embedding
    for t in 0..seq_len {
        let x_t = slice_seq_mut(scratch.x_cur, cfg.d_model, t);
        let pos_t = slice_seq(pos_w, cfg.d_model, t);
        for i in 0..cfg.d_model {
            x_t[i] += pos_t[i];
        }
    }

    // static prior
    matmul_vec_dispatch(
        static_proj_w,
        cfg.d_model,
        cfg.static_dim_total,
        static_total,
        static_proj_b,
        scratch.head_in,
        dispatch,
    );
    if cfg.prior_alpha != 1.0 {
        for v in scratch.head_in.iter_mut() {
            *v *= cfg.prior_alpha;
        }
    }

    let prior = &scratch.head_in[..cfg.d_model];
    match cfg.state_surgery {
        StateSurgery::None => {}
        StateSurgery::AddAll => {
            for t in 0..seq_len {
                let x_t = slice_seq_mut(scratch.x_cur, cfg.d_model, t);
                for i in 0..cfg.d_model {
                    x_t[i] += prior[i];
                }
            }
        }
        StateSurgery::AddFirst => {
            let start = if let Some(lens) = lengths {
                if lens.is_empty() {
                    0usize
                } else {
                    let len = lens[0].max(1) as usize;
                    seq_len.saturating_sub(len).min(seq_len.saturating_sub(1))
                }
            } else {
                0usize
            };
            let x_t = slice_seq_mut(scratch.x_cur, cfg.d_model, start);
            for i in 0..cfg.d_model {
                x_t[i] += prior[i];
            }
        }
        StateSurgery::GateAll => {
            let prior_gate_w = expect_weights(w.prior_gate_w, "prior_gate.weight")?;
            let prior_gate_b = w.prior_gate_b;
            let gate = &mut scratch.ln1_out[..cfg.d_model];
            matmul_vec_dispatch(
                prior_gate_w,
                cfg.d_model,
                cfg.d_model,
                prior,
                prior_gate_b,
                gate,
                dispatch,
            );
            for g in gate.iter_mut() {
                *g = sigmoid_dispatch(*g, math_backend);
            }
            for t in 0..seq_len {
                let x_t = slice_seq_mut(scratch.x_cur, cfg.d_model, t);
                for i in 0..cfg.d_model {
                    x_t[i] += gate[i] * prior[i];
                }
            }
        }
    }

    let batch = 1usize;
    let has_dbg = dbg.is_some();
    if let Some(tap) = dbg.as_deref_mut() {
        tap.tap_f32("x_in", &[batch, seq_len, cfg.d_model], scratch.x_cur)?;
    }

    let dbg_ptr = dbg.map(|tap| tap as *mut dyn DbgTaps);
    let mut stats = stats;
    let mut fast_stats = fast_stats;
    let fast_stats_ptr = fast_stats.as_deref_mut().map(|s| s as *mut FastMathStats);

    let x_proj_dim = cfg.dt_rank + 2 * cfg.d_state_pad;

    for layer in 0..cfg.n_layers {
        let lw = &w.layers[layer];
        let mut layer_dbg = dbg_ptr.map(|ptr| unsafe { &mut *ptr });
        let mut layer_fast_stats = fast_stats_ptr.map(|ptr| unsafe { &mut *ptr });
        #[cfg(not(feature = "bench_instrument"))]
        let _ = &mut layer_fast_stats;

        if let Some(tap) = layer_dbg.as_deref_mut() {
            tap.tap_f32(
                &format!("layer{}/residual_in", layer),
                &[batch, seq_len, cfg.d_model],
                scratch.x_cur,
            )?;
        }

        // LN1
        record_stage(&mut stats, Stage::Ln1, || {
            audit_trace(
                "ln",
                match dispatch {
                    CpuDispatch::Avx2 => "avx2",
                    CpuDispatch::Avx512 => "avx512",
                    CpuDispatch::Scalar => "scalar",
                },
                "ln1",
            );
            for t in 0..seq_len {
                let x_t = slice_seq(scratch.x_cur, cfg.d_model, t);
                let ln1_t = slice_seq_mut(scratch.ln1_out, cfg.d_model, t);
                layer_norm_dispatch(x_t, lw.ln1_w, lw.ln1_b, cfg.ln_eps1, ln1_t, dispatch);
            }
        });

        // in_proj
        record_stage(&mut stats, Stage::InProj, || {
            audit_trace(
                "fc1",
                match dispatch {
                    CpuDispatch::Avx2 => "avx2",
                    CpuDispatch::Avx512 => "avx512",
                    CpuDispatch::Scalar => "scalar",
                },
                "in_proj",
            );
            matmul_proj_batch(
                lw.in_proj_w,
                lw.in_proj_w_packed.as_deref(),
                lw.in_proj_w_packed_bf16.as_deref(),
                2 * cfg.d_inner,
                cfg.d_model,
                scratch.ln1_out,
                lw.in_proj_b,
                lw.in_proj_b_zero.as_deref(),
                seq_len,
                scratch.in_proj_out,
                dispatch,
                math_backend,
            );
        });

        // split x/z
        for t in 0..seq_len {
            let in_proj_t = slice_seq(scratch.in_proj_out, 2 * cfg.d_inner, t);
            let x_t = slice_seq_mut(scratch.x, cfg.d_inner, t);
            let z_t = slice_seq_mut(scratch.z, cfg.d_inner, t);
            x_t.copy_from_slice(&in_proj_t[..cfg.d_inner]);
            z_t.copy_from_slice(&in_proj_t[cfg.d_inner..]);
        }

        // conv1d + silu
        record_stage(&mut stats, Stage::Conv1dStep1, || {
            let fast_math = matches!(
                math_backend,
                MathBackend::Approx | MathBackend::Sleef | MathBackend::FastBf16 | MathBackend::FastWild
            );
            if fast_math {
                if let Some(packed) = lw.conv_w_packed.as_deref() {
                    match dispatch {
                        CpuDispatch::Avx512 => {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                if std::is_x86_feature_detected!("avx512f") {
                                    audit_trace("conv1d", "avx512", "packed");
                                    audit_once(AUDIT_CONV, "conv1d: avx512 packed");
                                    conv1d_depthwise_avx512(cfg, packed, lw.conv_b, scratch.x, scratch.conv_out);
                                    return;
                                }
                            }
                        }
                        CpuDispatch::Avx2 => {
                            #[cfg(target_arch = "x86_64")]
                            unsafe {
                                if std::is_x86_feature_detected!("avx2") {
                                    audit_trace("conv1d", "avx2", "packed");
                                    audit_once(AUDIT_CONV, "conv1d: avx2 packed");
                                    conv1d_depthwise_avx2(cfg, packed, lw.conv_b, scratch.x, scratch.conv_out);
                                    return;
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            if fast_math && dispatch != CpuDispatch::Scalar {
                fallback_or_panic("E2201 conv1d: packed SIMD path unavailable");
            }
            audit_trace(
                "conv1d",
                match dispatch {
                    CpuDispatch::Avx2 => "avx2",
                    CpuDispatch::Avx512 => "avx512",
                    CpuDispatch::Scalar => "scalar",
                },
                "scalar",
            );

            let pad = cfg.conv_kernel - 1;
            for t in 0..seq_len {
                let conv_t = slice_seq_mut(scratch.conv_out, cfg.d_inner, t);
                for i in 0..cfg.d_inner {
                    let mut acc = lw.conv_b[i];
                    for k in 0..cfg.conv_kernel {
                        let idx = t as isize + k as isize - pad as isize;
                        if idx >= 0 {
                            let x_tk = scratch.x[idx as usize * cfg.d_inner + i];
                            let w_idx = i * cfg.conv_kernel + k;
                            acc += lw.conv_w[w_idx] * x_tk;
                        }
                    }
                    conv_t[i] = silu_dispatch(acc, math_backend);
                }
            }
        });

        // x_proj
        record_stage(&mut stats, Stage::XProj, || {
            audit_trace(
                "fc1",
                match dispatch {
                    CpuDispatch::Avx2 => "avx2",
                    CpuDispatch::Avx512 => "avx512",
                    CpuDispatch::Scalar => "scalar",
                },
                "x_proj",
            );
            matmul_proj_batch(
                lw.x_proj_w,
                lw.x_proj_w_packed.as_deref(),
                lw.x_proj_w_packed_bf16.as_deref(),
                x_proj_dim,
                cfg.d_inner,
                scratch.conv_out,
                None,
                lw.x_proj_b_zero.as_deref(),
                seq_len,
                scratch.x_proj_out,
                dispatch,
                math_backend,
            );
        });

        // dt_in, B, C
        let fast_math = matches!(
            math_backend,
            MathBackend::Approx | MathBackend::Sleef | MathBackend::FastBf16 | MathBackend::FastWild
        );
        let ssm_layout_v2 = ssm_layout_v2_enabled(cfg, math_backend, has_dbg, dispatch);
        let ssm_layout_v2_panel = ssm_layout_v2 && ssm_layout_v2_panel_enabled();
        let fuse_dt_v2 = if ssm_layout_v2
            && dispatch == CpuDispatch::Avx512
            && fast_math
            && !has_dbg
        {
            std::env::var("RISK_MAMBA_FUSE_DT")
                .ok()
                .map(|v| v != "0")
                .unwrap_or(matches!(math_backend, MathBackend::FastWild))
        } else {
            false
        };
        let fuse_dt_v2_fast = fuse_dt_v2
            && std::env::var("RISK_MAMBA_FUSE_DT_FAST")
                .ok()
                .map(|v| v != "0")
                .unwrap_or(false);
        // Enable fused dt for fast math when dbg taps are off, with env override.
        let fuse_dt = if fast_math && !has_dbg && cfg.dt_rank == 16 && !ssm_layout_v2 {
            std::env::var("RISK_MAMBA_FUSE_DT")
                .ok()
                .map(|v| v != "0")
                .unwrap_or(matches!(math_backend, MathBackend::FastWild))
        } else {
            false
        };
        let use_xproj_bc = fast_math && !has_dbg;
        for t in 0..seq_len {
            let x_proj_t = slice_seq(scratch.x_proj_out, x_proj_dim, t);
            let dt_in_t = slice_seq_mut(scratch.dt_in, cfg.dt_rank, t);
            dt_in_t.copy_from_slice(&x_proj_t[..cfg.dt_rank]);
            if !use_xproj_bc {
                let b_vec_t = slice_seq_mut(scratch.b_vec, cfg.d_state_pad, t);
                let c_vec_t = slice_seq_mut(scratch.c_vec, cfg.d_state_pad, t);
                b_vec_t.copy_from_slice(&x_proj_t[cfg.dt_rank..cfg.dt_rank + cfg.d_state_pad]);
                c_vec_t.copy_from_slice(
                    &x_proj_t[cfg.dt_rank + cfg.d_state_pad..cfg.dt_rank + 2 * cfg.d_state_pad],
                );
            }
        }

        // delta
        if !fuse_dt && !fuse_dt_v2 {
            record_stage(&mut stats, Stage::DtProjSoftplus, || {
                matmul_proj_batch(
                    lw.dt_proj_w,
                    lw.dt_proj_w_packed.as_deref(),
                    lw.dt_proj_w_packed_bf16.as_deref(),
                    cfg.d_inner,
                    cfg.dt_rank,
                    scratch.dt_in,
                    Some(lw.dt_proj_b),
                    None,
                    seq_len,
                    scratch.delta,
                    dispatch,
                    math_backend,
                );
                let fast_softplus = math_backend == MathBackend::FastWild
                    || (math_backend == MathBackend::Approx && softplus_fast_for_approx());
                audit_trace(
                    "softplus",
                    match dispatch {
                        CpuDispatch::Avx2 => "avx2",
                        CpuDispatch::Avx512 => "avx512",
                        CpuDispatch::Scalar => "scalar",
                    },
                    if fast_softplus { "fast" } else { "exact" },
                );
                for t in 0..seq_len {
                    let delta_t = slice_seq_mut(scratch.delta, cfg.d_inner, t);
                    if fast_softplus {
                        let beta = cfg.softplus_beta;
                        let inv_beta = if beta != 0.0 { 1.0 / beta } else { 1.0 };
                        let threshold = cfg.softplus_threshold;
                        match dispatch {
                            CpuDispatch::Avx512 => {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    if std::is_x86_feature_detected!("avx512f") {
                                        use std::arch::x86_64::*;
                                        let zero = _mm512_setzero_ps();
                                        let sign_mask = _mm512_set1_ps(-0.0f32);
                                        let beta_v = _mm512_set1_ps(beta);
                                        let inv_beta_v = _mm512_set1_ps(inv_beta);
                                        let thresh_v = _mm512_set1_ps(threshold);
                                        let mut i = 0usize;
                                        while i + 16 <= cfg.d_inner {
                                            let x = _mm512_loadu_ps(delta_t.as_ptr().add(i));
                                            let bx = _mm512_mul_ps(x, beta_v);
                                            let abs = _mm512_andnot_ps(sign_mask, bx);
                                            let neg_abs = _mm512_sub_ps(zero, abs);
                                            let exp = exp512_ps(neg_abs);
                                            let base = _mm512_max_ps(bx, zero);
                                            let approx = _mm512_add_ps(base, exp);
                                            let mut res = _mm512_mul_ps(approx, inv_beta_v);
                                            let mask = _mm512_cmp_ps_mask(bx, thresh_v, _CMP_GT_OQ);
                                            res = _mm512_mask_blend_ps(mask, res, x);
                                            _mm512_storeu_ps(delta_t.as_mut_ptr().add(i), res);
                                            i += 16;
                                        }
                                        while i < cfg.d_inner {
                                            let x = delta_t[i];
                                            let bx = beta * x;
                                            if bx > threshold {
                                                delta_t[i] = x;
                                            } else {
                                                let abs = bx.abs();
                                                let base = if bx > 0.0 { bx } else { 0.0 };
                                                delta_t[i] = (base + exp_approx_scalar(-abs)) * inv_beta;
                                            }
                                            i += 1;
                                        }
                                        continue;
                                    }
                                }
                            }
                            CpuDispatch::Avx2 => {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    if std::is_x86_feature_detected!("avx2") {
                                        use std::arch::x86_64::*;
                                        let zero = _mm256_setzero_ps();
                                        let sign_mask = _mm256_set1_ps(-0.0f32);
                                        let beta_v = _mm256_set1_ps(beta);
                                        let inv_beta_v = _mm256_set1_ps(inv_beta);
                                        let thresh_v = _mm256_set1_ps(threshold);
                                        let mut i = 0usize;
                                        while i + 8 <= cfg.d_inner {
                                            let x = _mm256_loadu_ps(delta_t.as_ptr().add(i));
                                            let bx = _mm256_mul_ps(x, beta_v);
                                            let abs = _mm256_andnot_ps(sign_mask, bx);
                                            let neg_abs = _mm256_sub_ps(zero, abs);
                                            let exp = exp256_ps(neg_abs);
                                            let base = _mm256_max_ps(bx, zero);
                                            let approx = _mm256_add_ps(base, exp);
                                            let mut res = _mm256_mul_ps(approx, inv_beta_v);
                                            let mask = _mm256_cmp_ps(bx, thresh_v, _CMP_GT_OQ);
                                            res = _mm256_blendv_ps(res, x, mask);
                                            _mm256_storeu_ps(delta_t.as_mut_ptr().add(i), res);
                                            i += 8;
                                        }
                                        while i < cfg.d_inner {
                                            let x = delta_t[i];
                                            let bx = beta * x;
                                            if bx > threshold {
                                                delta_t[i] = x;
                                            } else {
                                                let abs = bx.abs();
                                                let base = if bx > 0.0 { bx } else { 0.0 };
                                                delta_t[i] = (base + exp_approx_scalar(-abs)) * inv_beta;
                                            }
                                            i += 1;
                                        }
                                        continue;
                                    }
                                }
                            }
                            CpuDispatch::Scalar => {}
                        }
                        for i in 0..cfg.d_inner {
                            let x = delta_t[i];
                            let bx = beta * x;
                            if bx > threshold {
                                delta_t[i] = x;
                            } else {
                                let abs = bx.abs();
                                let base = if bx > 0.0 { bx } else { 0.0 };
                                delta_t[i] = (base + exp_approx_scalar(-abs)) * inv_beta;
                            }
                        }
                    } else {
                        for i in 0..cfg.d_inner {
                            delta_t[i] = softplus_dispatch(
                                delta_t[i],
                                cfg.softplus_kind,
                                cfg.softplus_beta,
                                cfg.softplus_threshold,
                                math_backend,
                            );
                        }
                    }
                }
            });
        }

        // scan (stateless: layer_state zeroed)
        record_stage(&mut stats, Stage::SsmUpdate, || {
            let a_pre: &[f32] = if ssm_layout_v2 {
                let tile = ssm_layout_v2_tile(dispatch);
                let lane_stride = ssm_layout_v2_lane_stride(dispatch, ssm_layout_v2_panel);
                if ssm_layout_v2_panel {
                    let panel_pre = match dispatch {
                        CpuDispatch::Avx512 => lw.a_pre_v2p_16.as_deref(),
                        CpuDispatch::Avx2 => lw.a_pre_v2p_8.as_deref(),
                        CpuDispatch::Scalar => None,
                    };
                    if let Some(pre) = panel_pre {
                        pre
                    } else {
                        let tiles = cfg.d_inner / tile;
                        let needed = tiles * cfg.d_state_pad * lane_stride;
                        let tmp = &mut scratch.a_pre[..needed];
                        if let Some(pre) = lw.a_pre.as_deref() {
                            reorder_a_pre_v2_from_v1(cfg, tile, lane_stride, pre, tmp);
                        } else {
                            fill_a_pre_v2(cfg, tile, lane_stride, lw.a_log, tmp);
                        }
                        tmp
                    }
                } else {
                    let tiles = cfg.d_inner / tile;
                    let needed = tiles * cfg.d_state_pad * lane_stride;
                    let tmp = &mut scratch.a_pre[..needed];
                    if let Some(pre) = lw.a_pre.as_deref() {
                        reorder_a_pre_v2_from_v1(cfg, tile, lane_stride, pre, tmp);
                    } else {
                        fill_a_pre_v2(cfg, tile, lane_stride, lw.a_log, tmp);
                    }
                    tmp
                }
            } else if let Some(pre) = lw.a_pre.as_deref() {
                pre
            } else {
                let tmp = &mut scratch.a_pre[..cfg.d_inner * cfg.d_state_pad];
                fill_a_pre(lw.a_log, tmp);
                tmp
            };
            let layer_state_len = if ssm_layout_v2 {
                let tile = ssm_layout_v2_tile(dispatch);
                let lane_stride = ssm_layout_v2_lane_stride(dispatch, ssm_layout_v2_panel);
                let tiles = cfg.d_inner / tile;
                tiles * cfg.d_state_pad * lane_stride
            } else {
                scratch.layer_state.len()
            };
            for v in scratch.layer_state[..layer_state_len].iter_mut() {
                *v = 0.0;
            }

            let fast_math = matches!(
                math_backend,
                MathBackend::Approx | MathBackend::Sleef | MathBackend::FastBf16 | MathBackend::FastWild
            );
            let lane_stride = if ssm_layout_v2 {
                ssm_layout_v2_lane_stride(dispatch, ssm_layout_v2_panel)
            } else {
                0
            };
            #[cfg(target_arch = "x86_64")]
            unsafe {
                if fuse_dt_v2
                    && fast_math
                    && dispatch == CpuDispatch::Avx512
                    && std::is_x86_feature_detected!("avx512f")
                {
                    audit_trace("ssm_update", "avx512", "v2_fused_dt");
                    audit_once(AUDIT_SSM, "ssm_update: avx512 v2 fused_dt");
                    ssm_update_avx512_full_v2_fused_dt(
                        cfg,
                        lw,
                        scratch.conv_out,
                        scratch.z,
                        scratch.dt_in,
                        scratch.b_vec,
                        scratch.c_vec,
                        scratch.x_proj_out,
                        x_proj_dim,
                        cfg.dt_rank,
                        use_xproj_bc,
                        a_pre,
                        lane_stride,
                        fuse_dt_v2_fast,
                        scratch.scan_out,
                        scratch.layer_state,
                    );
                    return;
                }
                let ssm_v2_tag = if ssm_layout_v2_panel { "v2p" } else { "v2" };
                if ssm_layout_v2 && fast_math && dispatch == CpuDispatch::Avx512 && std::is_x86_feature_detected!("avx512f") {
                    audit_trace("ssm_update", "avx512", ssm_v2_tag);
                    audit_once(AUDIT_SSM, "ssm_update: avx512 v2");
                    ssm_update_avx512_full_v2(
                        cfg,
                        lw,
                        scratch.conv_out,
                        scratch.z,
                        scratch.delta,
                        scratch.b_vec,
                        scratch.c_vec,
                        scratch.x_proj_out,
                        x_proj_dim,
                        cfg.dt_rank,
                        use_xproj_bc,
                        a_pre,
                        lane_stride,
                        scratch.scan_out,
                        scratch.layer_state,
                    );
                    return;
                }
                if ssm_layout_v2 && fast_math && dispatch == CpuDispatch::Avx2 && std::is_x86_feature_detected!("avx2") {
                    audit_trace("ssm_update", "avx2", ssm_v2_tag);
                    audit_once(AUDIT_SSM, "ssm_update: avx2 v2");
                    ssm_update_avx2_full_v2(
                        cfg,
                        lw,
                        scratch.conv_out,
                        scratch.z,
                        scratch.delta,
                        scratch.b_vec,
                        scratch.c_vec,
                        scratch.x_proj_out,
                        x_proj_dim,
                        cfg.dt_rank,
                        use_xproj_bc,
                        a_pre,
                        lane_stride,
                        scratch.scan_out,
                        scratch.layer_state,
                    );
                    return;
                }
                if fuse_dt && fast_math && dispatch == CpuDispatch::Avx512 && std::is_x86_feature_detected!("avx512f") {
                    audit_trace("ssm_update", "avx512", "fused_dt");
                    audit_once(AUDIT_SSM, "ssm_update: avx512 fused_dt");
                    ssm_update_avx512_full_fused_dt(
                        cfg,
                        lw,
                        scratch.conv_out,
                        scratch.z,
                        scratch.dt_in,
                        scratch.b_vec,
                        scratch.c_vec,
                        scratch.x_proj_out,
                        x_proj_dim,
                        cfg.dt_rank,
                        use_xproj_bc,
                        a_pre,
                        scratch.scan_out,
                        scratch.layer_state,
                        math_backend,
                    );
                    return;
                }
                if fuse_dt && fast_math && dispatch == CpuDispatch::Avx2 && std::is_x86_feature_detected!("avx2") {
                    audit_trace("ssm_update", "avx2", "fused_dt");
                    audit_once(AUDIT_SSM, "ssm_update: avx2 fused_dt");
                    ssm_update_avx2_full_fused_dt(
                        cfg,
                        lw,
                        scratch.conv_out,
                        scratch.z,
                        scratch.dt_in,
                        scratch.b_vec,
                        scratch.c_vec,
                        scratch.x_proj_out,
                        x_proj_dim,
                        cfg.dt_rank,
                        use_xproj_bc,
                        a_pre,
                        scratch.scan_out,
                        scratch.layer_state,
                        math_backend,
                    );
                    return;
                }
                if fast_math && dispatch == CpuDispatch::Avx512 && std::is_x86_feature_detected!("avx512f") {
                    audit_trace("ssm_update", "avx512", "v1");
                    audit_once(AUDIT_SSM, "ssm_update: avx512");
                    ssm_update_avx512_full(
                        cfg,
                        lw,
                        scratch.conv_out,
                        scratch.z,
                        scratch.delta,
                        scratch.b_vec,
                        scratch.c_vec,
                        scratch.x_proj_out,
                        x_proj_dim,
                        cfg.dt_rank,
                        use_xproj_bc,
                        a_pre,
                        scratch.scan_out,
                        scratch.layer_state,
                        math_backend,
                    );
                    return;
                }
                if fast_math && dispatch == CpuDispatch::Avx2 && std::is_x86_feature_detected!("avx2") {
                    audit_trace("ssm_update", "avx2", "v1");
                    audit_once(AUDIT_SSM, "ssm_update: avx2");
                    ssm_update_avx2_full(
                        cfg,
                        lw,
                        scratch.conv_out,
                        scratch.z,
                        scratch.delta,
                        scratch.b_vec,
                        scratch.c_vec,
                        scratch.x_proj_out,
                        x_proj_dim,
                        cfg.dt_rank,
                        use_xproj_bc,
                        a_pre,
                        scratch.scan_out,
                        scratch.layer_state,
                        math_backend,
                    );
                    return;
                }
            }
            if ssm_layout_v2 {
                fallback_or_panic("E2301 ssm_update: v2 requires SIMD");
            }
            if fast_math && dispatch != CpuDispatch::Scalar {
                fallback_or_panic("E2302 ssm_update: SIMD path unavailable");
            }
            audit_trace("ssm_update", "scalar", if fuse_dt { "fused_dt" } else { "v1" });
            if fuse_dt {
                ssm_update_full_scalar_fused_dt(
                    cfg,
                    lw,
                    scratch.conv_out,
                    scratch.z,
                    scratch.dt_in,
                    scratch.b_vec,
                    scratch.c_vec,
                    scratch.x_proj_out,
                    x_proj_dim,
                    cfg.dt_rank,
                    use_xproj_bc,
                    a_pre,
                    scratch.scan_out,
                    scratch.layer_state,
                    math_backend,
                    &mut layer_fast_stats,
                );
            } else {
                ssm_update_full_scalar(
                    cfg,
                    lw,
                    scratch.conv_out,
                    scratch.z,
                    scratch.delta,
                    scratch.b_vec,
                    scratch.c_vec,
                    scratch.x_proj_out,
                    x_proj_dim,
                    cfg.dt_rank,
                    use_xproj_bc,
                    a_pre,
                    scratch.scan_out,
                    scratch.layer_state,
                    math_backend,
                    fast_math,
                    &mut layer_fast_stats,
                );
            }
        });

        // out_proj
        record_stage(&mut stats, Stage::OutProj, || {
            audit_trace(
                "fc2",
                match dispatch {
                    CpuDispatch::Avx2 => "avx2",
                    CpuDispatch::Avx512 => "avx512",
                    CpuDispatch::Scalar => "scalar",
                },
                "out_proj",
            );
            matmul_proj_batch(
                lw.out_proj_w,
                lw.out_proj_w_packed.as_deref(),
                lw.out_proj_w_packed_bf16.as_deref(),
                cfg.d_model,
                cfg.d_inner,
                scratch.scan_out,
                lw.out_proj_b,
                lw.out_proj_b_zero.as_deref(),
                seq_len,
                scratch.out_proj_out,
                dispatch,
                math_backend,
            );
        });

        if let Some(tap) = layer_dbg.as_deref_mut() {
            tap.tap_f32(
                &format!("layer{}/mamba_out", layer),
                &[batch, seq_len, cfg.d_model],
                scratch.out_proj_out,
            )?;
        }

        // resid1
        record_stage(&mut stats, Stage::Resid1, || {
            for t in 0..seq_len {
                let x_t = slice_seq(scratch.x_cur, cfg.d_model, t);
                let out_proj_t = slice_seq(scratch.out_proj_out, cfg.d_model, t);
                let resid1_t = slice_seq_mut(scratch.resid1_out, cfg.d_model, t);
                for i in 0..cfg.d_model {
                    resid1_t[i] = x_t[i] + out_proj_t[i];
                }
            }
        });

        // ln2
        record_stage(&mut stats, Stage::Ln2, || {
            audit_trace(
                "ln",
                match dispatch {
                    CpuDispatch::Avx2 => "avx2",
                    CpuDispatch::Avx512 => "avx512",
                    CpuDispatch::Scalar => "scalar",
                },
                "ln2",
            );
            for t in 0..seq_len {
                let resid1_t = slice_seq(scratch.resid1_out, cfg.d_model, t);
                let ln2_t = slice_seq_mut(scratch.ln2_out, cfg.d_model, t);
                layer_norm_dispatch(resid1_t, lw.ln2_w, lw.ln2_b, cfg.ln_eps2, ln2_t, dispatch);
            }
        });

        // fc1
        let use_mlp_fused = mlp_fused_enabled(seq_len, has_dbg, math_backend);
        if use_mlp_fused {
            if let Some(perf) = perf.as_deref_mut() {
                perf.add_fc1(seq_len, cfg.d_model, cfg.d_mlp, FC1_IMPL_FUSED);
            }
            record_stage(&mut stats, Stage::MlpFc1, || {
                audit_trace(
                    "fc1",
                    match dispatch {
                        CpuDispatch::Avx2 => "avx2",
                        CpuDispatch::Avx512 => "avx512",
                        CpuDispatch::Scalar => "scalar",
                    },
                    "mlp_fused",
                );
                audit_once(AUDIT_MLP, "mlp path: fused_streamed");
                mlp_fused_streamed(
                    cfg,
                    lw,
                    dispatch,
                    math_backend,
                    scratch.ln2_out,
                    scratch.mlp_fc2,
                );
            });
        } else {
            if let Some(perf) = perf.as_deref_mut() {
                let impl_bit = if lw.fc1_w_packed.is_some() && cfg.d_mlp % 16 == 0 {
                    FC1_IMPL_PACKED
                } else {
                    FC1_IMPL_UNPACKED
                };
                perf.add_fc1(seq_len, cfg.d_model, cfg.d_mlp, impl_bit);
            }
            record_stage(&mut stats, Stage::MlpFc1, || {
                audit_trace(
                    "fc1",
                    match dispatch {
                        CpuDispatch::Avx2 => "avx2",
                        CpuDispatch::Avx512 => "avx512",
                        CpuDispatch::Scalar => "scalar",
                    },
                    "mlp_fc1",
                );
                matmul_proj_batch(
                    lw.fc1_w,
                    lw.fc1_w_packed.as_deref(),
                    lw.fc1_w_packed_bf16.as_deref(),
                    cfg.d_mlp,
                    cfg.d_model,
                    scratch.ln2_out,
                    Some(lw.fc1_b),
                    None,
                    seq_len,
                    scratch.mlp_fc1,
                    dispatch,
                    math_backend,
                );
            });

        // gelu
        let mlp_inplace_gelu = !has_dbg
            && matches!(
                math_backend,
                MathBackend::Approx | MathBackend::Sleef | MathBackend::FastBf16 | MathBackend::FastWild
            )
            && std::env::var("RISK_MAMBA_MLP_INPLACE")
                .ok()
                .map(|v| v != "0")
                .unwrap_or(true);
        record_stage(&mut stats, Stage::Gelu, || {
            let fast_gelu =
                math_backend == MathBackend::FastWild
                    || (math_backend == MathBackend::Approx && gelu_fast_for_approx());
            audit_trace(
                "gelu",
                match dispatch {
                    CpuDispatch::Avx2 => "avx2",
                    CpuDispatch::Avx512 => "avx512",
                    CpuDispatch::Scalar => "scalar",
                },
                if fast_gelu { "fast" } else { "exact" },
            );
            for t in 0..seq_len {
                if mlp_inplace_gelu {
                    let fc1_t = slice_seq_mut(scratch.mlp_fc1, cfg.d_mlp, t);
                    let fc1_ptr = fc1_t.as_ptr();
                    let gelu_ptr = fc1_t.as_mut_ptr();
                    if fast_gelu {
                        match dispatch {
                            CpuDispatch::Avx512 => {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    if std::is_x86_feature_detected!("avx512f") {
                                        let mut i = 0usize;
                                        while i + 16 <= cfg.d_mlp {
                                            let x = std::arch::x86_64::_mm512_loadu_ps(fc1_ptr.add(i));
                                            let y = gelu_sigmoid_fast_avx512(x);
                                            std::arch::x86_64::_mm512_storeu_ps(gelu_ptr.add(i), y);
                                            i += 16;
                                        }
                                        while i < cfg.d_mlp {
                                            unsafe {
                                                *gelu_ptr.add(i) = gelu_sigmoid_fast(*fc1_ptr.add(i));
                                            }
                                            i += 1;
                                        }
                                        continue;
                                    }
                                }
                            }
                            CpuDispatch::Avx2 => {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    if std::is_x86_feature_detected!("avx2") {
                                        let mut i = 0usize;
                                        while i + 8 <= cfg.d_mlp {
                                            let x = std::arch::x86_64::_mm256_loadu_ps(fc1_ptr.add(i));
                                            let y = gelu_sigmoid_fast_avx2(x);
                                            std::arch::x86_64::_mm256_storeu_ps(gelu_ptr.add(i), y);
                                            i += 8;
                                        }
                                        while i < cfg.d_mlp {
                                            unsafe {
                                                *gelu_ptr.add(i) = gelu_sigmoid_fast(*fc1_ptr.add(i));
                                            }
                                            i += 1;
                                        }
                                        continue;
                                    }
                                }
                            }
                            CpuDispatch::Scalar => {}
                        }
                        for i in 0..cfg.d_mlp {
                            unsafe {
                                *gelu_ptr.add(i) = gelu_sigmoid_fast(*fc1_ptr.add(i));
                            }
                        }
                    } else {
                        for i in 0..cfg.d_mlp {
                            unsafe {
                                *gelu_ptr.add(i) = gelu_dispatch(
                                    *fc1_ptr.add(i),
                                    cfg.gelu_kind,
                                    math_backend,
                                );
                            }
                        }
                    }
                } else {
                    let fc1_t = slice_seq(scratch.mlp_fc1, cfg.d_mlp, t);
                    let gelu_t = slice_seq_mut(scratch.mlp_gelu, cfg.d_mlp, t);
                    if fast_gelu {
                        match dispatch {
                            CpuDispatch::Avx512 => {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    if std::is_x86_feature_detected!("avx512f") {
                                        let mut i = 0usize;
                                        while i + 16 <= cfg.d_mlp {
                                            let x = std::arch::x86_64::_mm512_loadu_ps(
                                                fc1_t.as_ptr().add(i),
                                            );
                                            let y = gelu_sigmoid_fast_avx512(x);
                                            std::arch::x86_64::_mm512_storeu_ps(
                                                gelu_t.as_mut_ptr().add(i),
                                                y,
                                            );
                                            i += 16;
                                        }
                                        while i < cfg.d_mlp {
                                            gelu_t[i] = gelu_sigmoid_fast(fc1_t[i]);
                                            i += 1;
                                        }
                                        continue;
                                    }
                                }
                            }
                            CpuDispatch::Avx2 => {
                                #[cfg(target_arch = "x86_64")]
                                unsafe {
                                    if std::is_x86_feature_detected!("avx2") {
                                        let mut i = 0usize;
                                        while i + 8 <= cfg.d_mlp {
                                            let x = std::arch::x86_64::_mm256_loadu_ps(
                                                fc1_t.as_ptr().add(i),
                                            );
                                            let y = gelu_sigmoid_fast_avx2(x);
                                            std::arch::x86_64::_mm256_storeu_ps(
                                                gelu_t.as_mut_ptr().add(i),
                                                y,
                                            );
                                            i += 8;
                                        }
                                        while i < cfg.d_mlp {
                                            gelu_t[i] = gelu_sigmoid_fast(fc1_t[i]);
                                            i += 1;
                                        }
                                        continue;
                                    }
                                }
                            }
                            CpuDispatch::Scalar => {}
                        }
                        for i in 0..cfg.d_mlp {
                            gelu_t[i] = gelu_sigmoid_fast(fc1_t[i]);
                        }
                    } else {
                        for i in 0..cfg.d_mlp {
                            gelu_t[i] = gelu_dispatch(fc1_t[i], cfg.gelu_kind, math_backend);
                        }
                    }
                }
            }
        });

        // fc2
        record_stage(&mut stats, Stage::MlpFc2, || {
            audit_trace(
                "fc2",
                match dispatch {
                    CpuDispatch::Avx2 => "avx2",
                    CpuDispatch::Avx512 => "avx512",
                    CpuDispatch::Scalar => "scalar",
                },
                if mlp_inplace_gelu { "mlp_fc2_inplace" } else { "mlp_fc2" },
            );
            let gelu_src = if mlp_inplace_gelu {
                unsafe { std::slice::from_raw_parts(scratch.mlp_fc1.as_ptr(), scratch.mlp_fc1.len()) }
            } else {
                unsafe { std::slice::from_raw_parts(scratch.mlp_gelu.as_ptr(), scratch.mlp_gelu.len()) }
            };
            matmul_proj_batch(
                lw.fc2_w,
                lw.fc2_w_packed.as_deref(),
                lw.fc2_w_packed_bf16.as_deref(),
                cfg.d_model,
                cfg.d_mlp,
                gelu_src,
                Some(lw.fc2_b),
                None,
                seq_len,
                scratch.mlp_fc2,
                dispatch,
                math_backend,
            );
        });
        }

        // resid2
        record_stage(&mut stats, Stage::Resid2, || {
            for t in 0..seq_len {
                let resid1_t = slice_seq(scratch.resid1_out, cfg.d_model, t);
                let fc2_t = slice_seq(scratch.mlp_fc2, cfg.d_model, t);
                let resid2_t = slice_seq_mut(scratch.resid2_out, cfg.d_model, t);
                for i in 0..cfg.d_model {
                    resid2_t[i] = resid1_t[i] + fc2_t[i];
                }
            }
        });

        if let Some(tap) = layer_dbg.as_deref_mut() {
            tap.tap_f32(
                &format!("layer{}/mlp_out", layer),
                &[batch, seq_len, cfg.d_model],
                scratch.mlp_fc2,
            )?;
            tap.tap_f32(
                &format!("layer{}/residual_out", layer),
                &[batch, seq_len, cfg.d_model],
                scratch.resid2_out,
            )?;
        }

        scratch.x_cur.copy_from_slice(scratch.resid2_out);
    }

    // final norm
    record_stage(&mut stats, Stage::FinalNorm, || {
        for t in 0..seq_len {
            let x_t = slice_seq(scratch.x_cur, cfg.d_model, t);
            let norm_t = slice_seq_mut(scratch.final_norm_out, cfg.d_model, t);
            layer_norm_dispatch(x_t, w.norm_w, w.norm_b, cfg.ln_eps2, norm_t, dispatch);
        }
    });

    if let Some(tap) = dbg_ptr {
        let last = seq_len - 1;
        let last_norm = slice_seq(scratch.final_norm_out, cfg.d_model, last);
        unsafe {
            (&mut *tap).tap_f32("final_norm_out", &[batch, cfg.d_model], last_norm)?;
        }
    }

    // head_in: use last token
    let last = seq_len - 1;
    let last_norm = slice_seq(scratch.final_norm_out, cfg.d_model, last);
    let head_in = &mut scratch.head_in[..cfg.d_model];
    let norm = l2norm(last_norm);
    if norm == 0.0 {
        for v in head_in.iter_mut() {
            *v = 0.0;
        }
    } else {
        for i in 0..cfg.d_model {
            head_in[i] = last_norm[i] / norm;
        }
    }

    if let Some(tap) = dbg_ptr {
        unsafe {
            (&mut *tap).tap_f32("head_in", &[batch, cfg.d_model], head_in)?;
        }
    }

    // head
    record_stage(&mut stats, Stage::HeadMatmul, || {
        if cfg.n_class == 2 {
            head_matmul_two_class(w.head_w, w.head_b, head_in, logits_out, dispatch);
        } else {
            matmul_proj(
                w.head_w,
                w.head_w_packed.as_deref(),
                cfg.n_class,
                cfg.d_model,
                head_in,
                Some(w.head_b),
                None,
                logits_out,
                dispatch,
            );
        }
    });

    margin_out[0] = if cfg.n_class >= 2 {
        logits_out[1] - logits_out[0]
    } else {
        logits_out[0]
    };

    Ok(())
}
