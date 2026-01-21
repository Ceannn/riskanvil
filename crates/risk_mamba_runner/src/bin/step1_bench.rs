use clap::Parser;
use rand::{rngs::StdRng, Rng, SeedableRng};
use risk_mamba_kernel::{
    audit_trace_dump_tsv, audit_trace_reset, exp_approx_scalar, kernel_path_tag, layout_tag,
    mlp_fused_enabled, scratch_bytes, step1_model, step1_model_latency, CpuDispatch, EmbSumMode,
    FastMathStats, ForwardKind, GeluKind, HeadInputKind, LayerWeights, LogitsKind, MathBackend,
    ModelConfig, PriorSource, SoftplusKind, StageStats, StateSurgery, WeightsView, SLEEF_AVAILABLE,
};
use serde::Deserialize;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Step1 kernel benchmark (no dbg/metrics)")]
struct Args {
    #[arg(long, alias = "bundle", default_value = "mamba/static_bundle_step1_v2_1")]
    bundle_dir: String,
    #[arg(long, default_value_t = 16)]
    batch: usize,
    #[arg(long, default_value_t = 200)]
    iters: usize,
    #[arg(long, default_value_t = 50)]
    warmup: usize,
    #[arg(long, default_value_t = 5)]
    repeat: usize,
    #[arg(long, default_value = "auto", value_parser = ["auto", "scalar", "avx2", "avx512"])]
    dispatch: String,
    #[arg(long, default_value = "auto", value_parser = ["auto", "fast", "exact", "sleef", "fast3_bf16", "fast_wild"])]
    math: String,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value = "outputs-7945hx")]
    out_dir: String,
    #[arg(long)]
    audit_kernel: bool,
}

#[derive(Deserialize)]
struct Manifest {
    format: String,
    schema_version: u32,
    weights_blob_sha256: String,
    weights_endianness: String,
    weights_file: String,
    alignment_bytes: usize,
    model_config: ManifestConfig,
    tensors: Vec<TensorSpec>,
}

#[derive(Deserialize)]
struct ManifestConfig {
    n_layers: usize,
    seq_len: usize,
    d_model: usize,
    d_inner: usize,
    d_mlp: usize,
    d_state: usize,
    d_state_pad: usize,
    dt_rank: usize,
    conv_kernel: usize,
    conv_groups: usize,
    conv_causal: bool,
    conv_current_tap: String,
    ln_eps1: f32,
    ln_eps2: f32,
    gelu_kind: String,
    softplus_beta: f32,
    softplus_threshold: f32,
    pad_policy: PadPolicy,
    logits_kind: String,
    n_class: usize,
    head_input_kind: String,
}

#[derive(Deserialize)]
struct PadPolicy {
    #[serde(rename = "A_log_pad_value")]
    a_log_pad_value: f32,
    #[serde(rename = "BC_pad_value")]
    bc_pad_value: f32,
    #[serde(rename = "state_pad_value")]
    state_pad_value: f32,
}

#[derive(Deserialize, Clone)]
struct TensorSpec {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    offset: usize,
    nbytes: usize,
    sha256: String,
}

#[derive(Serialize)]
struct StageStatsOutput {
    ln1: u64,
    in_proj: u64,
    conv1d_step1: u64,
    x_proj: u64,
    dt_proj_softplus: u64,
    ssm_update: u64,
    out_proj: u64,
    resid1: u64,
    ln2: u64,
    mlp_fc1: u64,
    gelu: u64,
    mlp_fc2: u64,
    resid2: u64,
    final_norm: u64,
    head_l2norm: u64,
    head_matmul: u64,
}

#[derive(Serialize)]
struct StageStatsPct {
    ln1: f64,
    in_proj: f64,
    conv1d_step1: f64,
    x_proj: f64,
    dt_proj_softplus: f64,
    ssm_update: f64,
    out_proj: f64,
    resid1: f64,
    ln2: f64,
    mlp_fc1: f64,
    gelu: f64,
    mlp_fc2: f64,
    resid2: f64,
    final_norm: f64,
    head_l2norm: f64,
    head_matmul: f64,
}

#[derive(Serialize)]
struct FastMathErrorEstimate {
    function: String,
    samples: usize,
    range: [f32; 2],
    max_abs: f32,
    max_rel: f32,
}

#[derive(Serialize)]
struct FastMathRuntimeError {
    samples: u64,
    input_min: f32,
    input_max: f32,
    max_abs: f32,
    max_rel: f32,
    abs_err_p99: f32,
    rel_err_max_masked: f32,
    rel_err_p99_masked: f32,
    masked_samples: u64,
    underflow_rate_normal: f64,
    underflow_rate_subnormal_or_zero: f64,
    hist_edges: Vec<f32>,
    hist_counts: Vec<u64>,
}

#[derive(Serialize)]
struct BenchOutput {
    cmd: String,
    bundle: String,
    dispatch_requested: String,
    math_requested: String,
    dispatch: String,
    math_backend: String,
    kernel_path: String,
    layout: String,
    mlp_impl: String,
    packed_layout: Option<String>,
    fast_math_error: Option<FastMathErrorEstimate>,
    fast_math_runtime_error: Option<FastMathRuntimeError>,
    batch: usize,
    iters: usize,
    warmup: usize,
    repeat: usize,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    min_ms: f64,
    max_ms: f64,
    runs_ms: Vec<f64>,
    outlier_count: usize,
    outlier_indices: Vec<usize>,
    between_repeat_cv: f64,
    stage_timing_enabled: bool,
    stage_stats_unit: Option<String>,
    stage_stats_total: Option<StageStatsOutput>,
    stage_stats_per_iter: Option<StageStatsOutput>,
    stage_stats_pct: Option<StageStatsPct>,
    cycles_per_iter_total: Option<u64>,
}

struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    layout: std::alloc::Layout,
}

impl AlignedBuf {
    fn new(len: usize, align: usize) -> Result<Self, String> {
        let layout = std::alloc::Layout::from_size_align(len, align).map_err(|e| e.to_string())?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err("alloc failed".to_string());
        }
        Ok(Self { ptr, len, layout })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.ptr, self.layout);
        }
    }
}

fn load_manifest(path: &Path) -> Result<Manifest, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut buf = String::new();
    file.read_to_string(&mut buf).map_err(|e| e.to_string())?;
    serde_json::from_str(&buf).map_err(|e| e.to_string())
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let digest = hasher.finalize();
    format!("{:x}", digest)
}

fn slice_tensor<'a>(blob: &'a [u8], spec: &TensorSpec) -> Result<&'a [f32], String> {
    if spec.dtype != "f32" {
        return Err(format!("tensor {} dtype {}", spec.name, spec.dtype));
    }
    if spec.nbytes % 4 != 0 {
        return Err(format!("tensor {} nbytes not multiple of 4", spec.name));
    }
    if spec.offset + spec.nbytes > blob.len() {
        return Err(format!("tensor {} out of range", spec.name));
    }
    let ptr = unsafe { blob.as_ptr().add(spec.offset) } as *const f32;
    if (ptr as usize) % 4 != 0 {
        return Err(format!("tensor {} misaligned", spec.name));
    }
    let len = spec.nbytes / 4;
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(slice)
}

fn check_tensor_sha(blob: &[u8], spec: &TensorSpec) -> Result<(), String> {
    let data = &blob[spec.offset..spec.offset + spec.nbytes];
    let actual = sha256_hex(data);
    if actual != spec.sha256 {
        return Err(format!("tensor {} sha256 mismatch", spec.name));
    }
    Ok(())
}

fn validate_manifest_shapes(manifest: &Manifest) -> Result<(), String> {
    let cfg = &manifest.model_config;
    let mut map = HashMap::new();
    for spec in &manifest.tensors {
        map.insert(spec.name.clone(), spec.clone());
    }
    let check = |name: &str, shape: &[usize]| -> Result<(), String> {
        let spec = map.get(name).ok_or_else(|| format!("missing tensor {}", name))?;
        if spec.shape != shape {
            return Err(format!("shape mismatch {} {:?} != {:?}", name, spec.shape, shape));
        }
        let expect_bytes: usize = shape.iter().product::<usize>() * 4;
        if spec.nbytes != expect_bytes {
            return Err(format!("nbytes mismatch {} {} != {}", name, spec.nbytes, expect_bytes));
        }
        Ok(())
    };

    check("norm.weight", &[cfg.d_model])?;
    check("norm.bias", &[cfg.d_model])?;
    check("head.weight", &[cfg.n_class, cfg.d_model])?;
    check("head.bias", &[cfg.n_class])?;

    for i in 0..cfg.n_layers {
        let prefix = format!("layers.{}", i);
        check(&format!("{}.norm1.weight", prefix), &[cfg.d_model])?;
        check(&format!("{}.norm1.bias", prefix), &[cfg.d_model])?;
        check(&format!("{}.norm2.weight", prefix), &[cfg.d_model])?;
        check(&format!("{}.norm2.bias", prefix), &[cfg.d_model])?;
        check(&format!("{}.mamba.in_proj.weight", prefix), &[2 * cfg.d_inner, cfg.d_model])?;
        check(&format!("{}.mamba.conv1d.weight", prefix), &[cfg.d_inner, 1, cfg.conv_kernel])?;
        check(&format!("{}.mamba.conv1d.bias", prefix), &[cfg.d_inner])?;
        check(
            &format!("{}.mamba.x_proj.weight", prefix),
            &[cfg.dt_rank + 2 * cfg.d_state_pad, cfg.d_inner],
        )?;
        check(&format!("{}.mamba.dt_proj.weight", prefix), &[cfg.d_inner, cfg.dt_rank])?;
        check(&format!("{}.mamba.dt_proj.bias", prefix), &[cfg.d_inner])?;
        check(&format!("{}.mamba.A_log", prefix), &[cfg.d_inner, cfg.d_state_pad])?;
        check(&format!("{}.mamba.D", prefix), &[cfg.d_inner])?;
        check(&format!("{}.mamba.out_proj.weight", prefix), &[cfg.d_model, cfg.d_inner])?;
        check(&format!("{}.mlp.fc1.weight", prefix), &[cfg.d_mlp, cfg.d_model])?;
        check(&format!("{}.mlp.fc1.bias", prefix), &[cfg.d_mlp])?;
        check(&format!("{}.mlp.fc2.weight", prefix), &[cfg.d_model, cfg.d_mlp])?;
        check(&format!("{}.mlp.fc2.bias", prefix), &[cfg.d_model])?;
    }

    Ok(())
}

fn build_weights<'a>(manifest: &Manifest, blob: &'a [u8]) -> Result<WeightsView<'a>, String> {
    let cfg = ModelConfig {
        forward_kind: ForwardKind::Step1,
        n_layers: manifest.model_config.n_layers,
        seq_len: manifest.model_config.seq_len,
        feature_dim: 1,
        vocab_size: 0,
        static_dim_total: 0,
        micro_state_dim: 0,
        d_model: manifest.model_config.d_model,
        d_inner: manifest.model_config.d_inner,
        d_state: manifest.model_config.d_state,
        d_state_pad: manifest.model_config.d_state_pad,
        d_mlp: manifest.model_config.d_mlp,
        conv_kernel: manifest.model_config.conv_kernel,
        dt_rank: manifest.model_config.dt_rank,
        ln_eps1: manifest.model_config.ln_eps1,
        ln_eps2: manifest.model_config.ln_eps2,
        gelu_kind: GeluKind::Erf,
        softplus_kind: SoftplusKind::Exact,
        softplus_beta: manifest.model_config.softplus_beta,
        softplus_threshold: manifest.model_config.softplus_threshold,
        logits_kind: LogitsKind::NativeTwoClass,
        head_input_kind: HeadInputKind::HeadIn,
        n_class: manifest.model_config.n_class,
        state_surgery: StateSurgery::None,
        prior_source: PriorSource::None,
        emb_sum_mode: EmbSumMode::Fast,
        prior_alpha: 0.0,
    };

    let mut map = HashMap::new();
    for spec in &manifest.tensors {
        map.insert(spec.name.clone(), spec.clone());
    }

    let get = |name: &str| -> Result<&TensorSpec, String> {
        map.get(name).ok_or_else(|| format!("missing tensor {}", name))
    };

    let norm_w = {
        let spec = get("norm.weight")?;
        check_tensor_sha(blob, spec)?;
        slice_tensor(blob, spec)?
    };
    let norm_b = {
        let spec = get("norm.bias")?;
        check_tensor_sha(blob, spec)?;
        slice_tensor(blob, spec)?
    };
    let head_w = {
        let spec = get("head.weight")?;
        check_tensor_sha(blob, spec)?;
        slice_tensor(blob, spec)?
    };
    let head_b = {
        let spec = get("head.bias")?;
        check_tensor_sha(blob, spec)?;
        slice_tensor(blob, spec)?
    };

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let prefix = format!("layers.{}", i);
        let ln1_w = {
            let spec = get(&format!("{}.norm1.weight", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let ln1_b = {
            let spec = get(&format!("{}.norm1.bias", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let ln2_w = {
            let spec = get(&format!("{}.norm2.weight", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let ln2_b = {
            let spec = get(&format!("{}.norm2.bias", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let in_proj_w = {
            let spec = get(&format!("{}.mamba.in_proj.weight", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let conv_w = {
            let spec = get(&format!("{}.mamba.conv1d.weight", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let conv_b = {
            let spec = get(&format!("{}.mamba.conv1d.bias", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let x_proj_w = {
            let spec = get(&format!("{}.mamba.x_proj.weight", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let dt_proj_w = {
            let spec = get(&format!("{}.mamba.dt_proj.weight", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let dt_proj_b = {
            let spec = get(&format!("{}.mamba.dt_proj.bias", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let a_log = {
            let spec = get(&format!("{}.mamba.A_log", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let d = {
            let spec = get(&format!("{}.mamba.D", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let out_proj_w = {
            let spec = get(&format!("{}.mamba.out_proj.weight", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let fc1_w = {
            let spec = get(&format!("{}.mlp.fc1.weight", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let fc1_b = {
            let spec = get(&format!("{}.mlp.fc1.bias", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let fc2_w = {
            let spec = get(&format!("{}.mlp.fc2.weight", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };
        let fc2_b = {
            let spec = get(&format!("{}.mlp.fc2.bias", prefix))?;
            check_tensor_sha(blob, spec)?;
            slice_tensor(blob, spec)?
        };

        layers.push(LayerWeights {
            ln1_w,
            ln1_b,
            ln2_w,
            ln2_b,
            in_proj_w,
            in_proj_w_packed: None,
            in_proj_w_packed_bf16: None,
            in_proj_w_gamma_sum: None,
            in_proj_w_beta_sum: None,
            in_proj_b: None,
            in_proj_b_zero: None,
            conv_w,
            conv_b,
            conv_w_packed: None,
            x_proj_w,
            x_proj_w_packed: None,
            x_proj_w_packed_bf16: None,
            x_proj_b_zero: None,
            dt_proj_w,
            dt_proj_w_packed: None,
            dt_proj_w_packed_bf16: None,
            dt_proj_b,
            a_log,
            a_pre: None,
            a_pre_v2p_16: None,
            a_pre_v2p_8: None,
            d,
            out_proj_w,
            out_proj_w_packed: None,
            out_proj_w_packed_bf16: None,
            out_proj_b: None,
            out_proj_b_zero: None,
            fc1_w,
            fc1_w_packed: None,
            fc1_w_packed_bf16: None,
            fc1_b,
            fc2_w,
            fc2_w_packed: None,
            fc2_w_packed_bf16: None,
            fc2_b,
        });
    }

    let mut weights =
        WeightsView::new(cfg, norm_w, norm_b, head_w, head_b, layers).map_err(|e| e.to_string())?;
    weights.precompute_a_pre();
    weights.precompute_packed(16);
    Ok(weights)
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len() as f64;
    let rank = (p / 100.0) * (n - 1.0);
    let idx = rank.round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(feature = "bench_instrument")]
fn percentile_f32(values: &[f32], p: f64) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len() as f64;
    let rank = (p / 100.0) * (n - 1.0);
    let idx = rank.round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64], mean_val: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|v| {
            let d = v - mean_val;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    var.sqrt()
}

fn stage_stats_output(stats: &StageStats) -> StageStatsOutput {
    StageStatsOutput {
        ln1: stats.ln1,
        in_proj: stats.in_proj,
        conv1d_step1: stats.conv1d_step1,
        x_proj: stats.x_proj,
        dt_proj_softplus: stats.dt_proj_softplus,
        ssm_update: stats.ssm_update,
        out_proj: stats.out_proj,
        resid1: stats.resid1,
        ln2: stats.ln2,
        mlp_fc1: stats.mlp_fc1,
        gelu: stats.gelu,
        mlp_fc2: stats.mlp_fc2,
        resid2: stats.resid2,
        final_norm: stats.final_norm,
        head_l2norm: stats.head_l2norm,
        head_matmul: stats.head_matmul,
    }
}

fn stage_stats_total(stats: &StageStatsOutput) -> u64 {
    stats.ln1
        + stats.in_proj
        + stats.conv1d_step1
        + stats.x_proj
        + stats.dt_proj_softplus
        + stats.ssm_update
        + stats.out_proj
        + stats.resid1
        + stats.ln2
        + stats.mlp_fc1
        + stats.gelu
        + stats.mlp_fc2
        + stats.resid2
        + stats.final_norm
        + stats.head_l2norm
        + stats.head_matmul
}

fn stage_stats_pct(stats: &StageStatsOutput, total: u64) -> StageStatsPct {
    let denom = if total == 0 { 1.0 } else { total as f64 };
    let pct = |v: u64| v as f64 * 100.0 / denom;
    StageStatsPct {
        ln1: pct(stats.ln1),
        in_proj: pct(stats.in_proj),
        conv1d_step1: pct(stats.conv1d_step1),
        x_proj: pct(stats.x_proj),
        dt_proj_softplus: pct(stats.dt_proj_softplus),
        ssm_update: pct(stats.ssm_update),
        out_proj: pct(stats.out_proj),
        resid1: pct(stats.resid1),
        ln2: pct(stats.ln2),
        mlp_fc1: pct(stats.mlp_fc1),
        gelu: pct(stats.gelu),
        mlp_fc2: pct(stats.mlp_fc2),
        resid2: pct(stats.resid2),
        final_norm: pct(stats.final_norm),
        head_l2norm: pct(stats.head_l2norm),
        head_matmul: pct(stats.head_matmul),
    }
}

fn estimate_fast_math_error(samples: usize, seed: u64, range: [f32; 2]) -> FastMathErrorEstimate {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for _ in 0..samples {
        let x = rng.gen_range(range[0]..range[1]);
        let approx = exp_approx_scalar(x);
        let exact = x.exp();
        let diff = (approx - exact).abs();
        let rel = if exact == 0.0 { diff } else { diff / exact.abs() };
        if diff > max_abs {
            max_abs = diff;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    FastMathErrorEstimate {
        function: "exp_approx".to_string(),
        samples,
        range,
        max_abs,
        max_rel,
    }
}

#[cfg(feature = "bench_instrument")]
fn runtime_error_output(stats: &FastMathStats) -> FastMathRuntimeError {
    let abs_err_p99 = percentile_f32(&stats.exp_abs_errors, 99.0);
    let rel_err_max_masked = stats
        .exp_rel_errors_masked
        .iter()
        .fold(0.0f32, |acc, v| if *v > acc { *v } else { acc });
    let rel_err_p99_masked = percentile_f32(&stats.exp_rel_errors_masked, 99.0);
    let masked_samples = stats.exp_rel_errors_masked.len() as u64;
    let samples = stats.exp_samples.max(1);
    let underflow_rate_normal = stats.exp_underflow_normal as f64 / samples as f64;
    let underflow_rate_subnormal_or_zero = stats.exp_underflow_subnormal as f64 / samples as f64;
    FastMathRuntimeError {
        samples: stats.exp_samples,
        input_min: stats.exp_min_input,
        input_max: stats.exp_max_input,
        max_abs: stats.exp_max_abs,
        max_rel: stats.exp_max_rel,
        abs_err_p99,
        rel_err_max_masked,
        rel_err_p99_masked,
        masked_samples,
        underflow_rate_normal,
        underflow_rate_subnormal_or_zero,
        hist_edges: EXP_HIST_EDGES.to_vec(),
        hist_counts: stats.exp_hist_counts.to_vec(),
    }
}

fn packed_layout(weights: &WeightsView, block: usize) -> Option<String> {
    let mut packed_any = false;
    for layer in &weights.layers {
        packed_any |= layer.in_proj_w_packed.is_some();
        packed_any |= layer.x_proj_w_packed.is_some();
        packed_any |= layer.dt_proj_w_packed.is_some();
        packed_any |= layer.out_proj_w_packed.is_some();
        packed_any |= layer.fc1_w_packed.is_some();
        packed_any |= layer.fc2_w_packed.is_some();
    }
    packed_any |= weights.head_w_packed.is_some();
    if packed_any {
        Some(format!("block{}", block))
    } else {
        None
    }
}

fn resolve_dispatch(name: &str) -> Result<CpuDispatch, String> {
    match name {
        "scalar" => Ok(CpuDispatch::Scalar),
        "avx2" => Ok(CpuDispatch::Avx2),
        "avx512" => Ok(CpuDispatch::Avx512),
        "auto" => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("fma")
                {
                    return Ok(CpuDispatch::Avx512);
                }
                if std::is_x86_feature_detected!("avx2")
                    && std::is_x86_feature_detected!("fma")
                {
                    return Ok(CpuDispatch::Avx2);
                }
            }
            Ok(CpuDispatch::Scalar)
        }
        other => Err(format!("invalid dispatch {}", other)),
    }
}

fn dispatch_label(dispatch: CpuDispatch) -> &'static str {
    match dispatch {
        CpuDispatch::Scalar => "scalar",
        CpuDispatch::Avx2 => "avx2",
        CpuDispatch::Avx512 => "avx512",
    }
}

fn main() -> Result<(), String> {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        const FTZ: u32 = 1 << 15;
        const DAZ: u32 = 1 << 6;
        unsafe {
            let csr = _mm_getcsr();
            _mm_setcsr(csr | FTZ | DAZ);
        }
    }
    let args = Args::parse();
    if args.audit_kernel {
        std::env::set_var("RISK_MAMBA_AUDIT_KERNEL", "1");
        audit_trace_reset();
    }
    if args.iters == 0 || args.repeat == 0 {
        return Err("iters and repeat must be > 0".to_string());
    }
    let bundle_dir = PathBuf::from(&args.bundle_dir);
    let manifest_path = bundle_dir.join("weights_manifest.json");
    let manifest = load_manifest(&manifest_path)?;

    if manifest.schema_version < 2 {
        return Err("schema_version < 2".to_string());
    }
    if manifest.weights_endianness != "little" {
        return Err("weights_endianness must be little".to_string());
    }
    if manifest.model_config.gelu_kind != "erf" {
        return Err("gelu_kind must be erf for correctness gate".to_string());
    }
    if manifest.model_config.softplus_beta != 1.0 {
        return Err("softplus_beta must be 1.0".to_string());
    }
    if manifest.model_config.softplus_threshold != 20.0 {
        return Err("softplus_threshold must be 20.0".to_string());
    }
    if manifest.model_config.logits_kind != "native_two_class" {
        return Err("logits_kind must be native_two_class".to_string());
    }
    if manifest.model_config.head_input_kind != "head_in" {
        return Err("head_input_kind must be head_in".to_string());
    }
    if manifest.model_config.n_class != 2 {
        return Err("n_class must be 2".to_string());
    }
    if manifest.model_config.conv_kernel == 0 {
        return Err("conv_kernel must be > 0".to_string());
    }
    if manifest.model_config.conv_groups != manifest.model_config.d_inner {
        return Err("conv_groups must equal d_inner".to_string());
    }
    if !manifest.model_config.conv_causal {
        return Err("conv_causal must be true".to_string());
    }
    if manifest.model_config.conv_current_tap != "last" {
        return Err("conv_current_tap must be last".to_string());
    }
    if manifest.model_config.seq_len != 1 {
        return Err("seq_len must be 1 for step1".to_string());
    }
    if manifest.model_config.pad_policy.a_log_pad_value != -30.0 {
        return Err("A_log_pad_value must be -30.0".to_string());
    }
    if manifest.model_config.pad_policy.bc_pad_value != 0.0 {
        return Err("BC_pad_value must be 0.0".to_string());
    }
    if manifest.model_config.pad_policy.state_pad_value != 0.0 {
        return Err("state_pad_value must be 0.0".to_string());
    }

    validate_manifest_shapes(&manifest)?;

    let weights_path = bundle_dir.join(&manifest.weights_file);
    let mut blob = Vec::new();
    File::open(&weights_path)
        .map_err(|e| e.to_string())?
        .read_to_end(&mut blob)
        .map_err(|e| e.to_string())?;

    let blob_sha = sha256_hex(&blob);
    if blob_sha != manifest.weights_blob_sha256 {
        return Err("weights_blob_sha256 mismatch".to_string());
    }

    let mut weights = build_weights(&manifest, &blob)?;
    let cfg = weights.cfg.clone();
    let packed_layout = packed_layout(&weights, 16);

    let mut dispatch_name = args.dispatch.clone();
    if dispatch_name == "auto" {
        if let Ok(env) = std::env::var("RISK_MAMBA_DISPATCH") {
            dispatch_name = env;
        }
    }
    let dispatch = resolve_dispatch(&dispatch_name)?;

    let mut math_name = args.math.clone();
    if math_name == "auto" {
        if let Ok(env) = std::env::var("RISK_MAMBA_MATH") {
            math_name = env;
        } else {
            math_name = "fast".to_string();
        }
    }
    let math_backend = match math_name.as_str() {
        "fast" => MathBackend::Approx,
        "exact" => MathBackend::Exact,
        "fast3_bf16" => MathBackend::FastBf16,
        "fast_wild" => MathBackend::FastWild,
        "sleef" => {
            if !SLEEF_AVAILABLE {
                return Err(
                    "sleef backend not enabled (build with -p risk_mamba_runner --features sleef)"
                        .to_string(),
                );
            }
            MathBackend::Sleef
        }
        other => return Err(format!("invalid math {}", other)),
    };
    if matches!(math_backend, MathBackend::FastBf16 | MathBackend::FastWild) {
        weights.precompute_packed_bf16(16);
    }
    let fast_math = matches!(
        math_backend,
        MathBackend::Approx | MathBackend::Sleef | MathBackend::FastBf16 | MathBackend::FastWild
    );
    #[cfg(not(feature = "bench_instrument"))]
    let _ = fast_math;
    let fast_math_error = if matches!(math_backend, MathBackend::Approx | MathBackend::FastBf16 | MathBackend::FastWild)
    {
        Some(estimate_fast_math_error(
            4096,
            args.seed ^ 0x9e3779b97f4a7c15,
            [-30.0, 0.0],
        ))
    } else {
        None
    };
    #[cfg(feature = "bench_instrument")]
    let mut fast_stats = if fast_math {
        Some(FastMathStats::new(512))
    } else {
        None
    };
    #[cfg(not(feature = "bench_instrument"))]
    let mut fast_stats: Option<FastMathStats> = None;

    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut x_in = vec![0.0f32; args.batch * cfg.d_model];
    for v in x_in.iter_mut() {
        *v = rng.gen_range(-0.1f32..0.1f32);
    }

    let mut state_init = vec![0.0f32; args.batch * cfg.n_layers * cfg.d_inner * cfg.d_state_pad];
    let inner_stride = cfg.d_state_pad;
    let layer_stride = args.batch * cfg.d_inner * cfg.d_state_pad;
    for layer in 0..cfg.n_layers {
        for b in 0..args.batch {
            let base = layer * layer_stride + b * cfg.d_inner * cfg.d_state_pad;
            for i in 0..cfg.d_inner {
                let row = &mut state_init[base + i * inner_stride..base + (i + 1) * inner_stride];
                for j in 0..cfg.d_state {
                    row[j] = rng.gen_range(-0.01f32..0.01f32);
                }
                for j in cfg.d_state..cfg.d_state_pad {
                    row[j] = 0.0;
                }
            }
        }
    }

    let scratch_len = scratch_bytes(&cfg, args.batch);
    let mut scratch = AlignedBuf::new(scratch_len, 64)?;
    let mut state = vec![0.0f32; state_init.len()];
    let mut logits = vec![0.0f32; args.batch * cfg.n_class];
    let mut margin = vec![0.0f32; args.batch];
    let mut stage_stats = StageStats::default();
    let use_latency = args.batch == 1;

    // warmup
    state.copy_from_slice(&state_init);
    for _ in 0..args.warmup {
        if use_latency {
            step1_model_latency(
                &cfg,
                &weights,
                dispatch,
                math_backend,
                args.batch,
                &x_in,
                &mut state,
                scratch.as_mut_slice(),
                &mut logits,
                &mut margin,
                Some(&mut stage_stats),
                None,
            )
        } else {
            step1_model(
                &cfg,
                &weights,
                dispatch,
                math_backend,
                args.batch,
                &x_in,
                &mut state,
                scratch.as_mut_slice(),
                &mut logits,
                &mut margin,
                None,
                Some(&mut stage_stats),
                None,
            )
        }
        .map_err(|e| e.to_string())?;
    }

    stage_stats.clear();
    #[cfg(feature = "bench_instrument")]
    if let Some(stats) = fast_stats.as_mut() {
        stats.clear();
    }

    let mut runs_ms = Vec::with_capacity(args.repeat);
    for _ in 0..args.repeat {
        state.copy_from_slice(&state_init);
        let start = Instant::now();
        for _ in 0..args.iters {
            if use_latency {
                step1_model_latency(
                    &cfg,
                    &weights,
                    dispatch,
                    math_backend,
                    args.batch,
                    &x_in,
                    &mut state,
                    scratch.as_mut_slice(),
                    &mut logits,
                    &mut margin,
                    Some(&mut stage_stats),
                    fast_stats.as_mut(),
                )
            } else {
                step1_model(
                    &cfg,
                    &weights,
                    dispatch,
                    math_backend,
                    args.batch,
                    &x_in,
                    &mut state,
                    scratch.as_mut_slice(),
                    &mut logits,
                    &mut margin,
                    None,
                    Some(&mut stage_stats),
                    fast_stats.as_mut(),
                )
            }
            .map_err(|e| e.to_string())?;
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        runs_ms.push(elapsed / args.iters as f64);
    }

    let mut sorted = runs_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean_ms = mean(&runs_ms);
    let std_ms = stddev(&runs_ms, mean_ms);
    let p50_ms = percentile(&sorted, 50.0);
    let p95_ms = percentile(&sorted, 95.0);
    let p99_ms = percentile(&sorted, 99.0);
    let min_ms = *sorted.first().unwrap_or(&0.0);
    let max_ms = *sorted.last().unwrap_or(&0.0);
    let outlier_thresh = mean_ms + 3.0 * std_ms;
    let outlier_indices: Vec<usize> = runs_ms
        .iter()
        .enumerate()
        .filter(|(_, v)| **v > outlier_thresh)
        .map(|(i, _)| i)
        .collect();
    let between_repeat_cv = if mean_ms > 0.0 { std_ms / mean_ms } else { 0.0 };

    let total_iters = args.iters * args.repeat;
    let stage_total = stage_stats_output(&stage_stats);
    let stage_per_iter = StageStatsOutput {
        ln1: stage_stats.ln1 / total_iters as u64,
        in_proj: stage_stats.in_proj / total_iters as u64,
        conv1d_step1: stage_stats.conv1d_step1 / total_iters as u64,
        x_proj: stage_stats.x_proj / total_iters as u64,
        dt_proj_softplus: stage_stats.dt_proj_softplus / total_iters as u64,
        ssm_update: stage_stats.ssm_update / total_iters as u64,
        out_proj: stage_stats.out_proj / total_iters as u64,
        resid1: stage_stats.resid1 / total_iters as u64,
        ln2: stage_stats.ln2 / total_iters as u64,
        mlp_fc1: stage_stats.mlp_fc1 / total_iters as u64,
        gelu: stage_stats.gelu / total_iters as u64,
        mlp_fc2: stage_stats.mlp_fc2 / total_iters as u64,
        resid2: stage_stats.resid2 / total_iters as u64,
        final_norm: stage_stats.final_norm / total_iters as u64,
        head_l2norm: stage_stats.head_l2norm / total_iters as u64,
        head_matmul: stage_stats.head_matmul / total_iters as u64,
    };
    let cycles_per_iter_total = stage_stats_total(&stage_per_iter);
    let stage_pct = stage_stats_pct(&stage_per_iter, cycles_per_iter_total);
    #[cfg(feature = "bench_instrument")]
    let fast_math_runtime_error = fast_stats.as_ref().map(runtime_error_output);
    #[cfg(not(feature = "bench_instrument"))]
    let fast_math_runtime_error = None;

    let stage_timing_enabled = cfg!(feature = "stage_timing");
    let kernel_path = kernel_path_tag(&cfg).to_string();
    let layout = layout_tag(&cfg, math_backend, false, dispatch).to_string();
    let mlp_impl = if mlp_fused_enabled(cfg.seq_len, false, math_backend) {
        "fused_streamed"
    } else if cfg.seq_len == 1 {
        "unfused_m1"
    } else {
        "unfused"
    }
    .to_string();
    let output = BenchOutput {
        cmd: std::env::args().collect::<Vec<_>>().join(" "),
        bundle: args.bundle_dir.clone(),
        dispatch_requested: args.dispatch.clone(),
        math_requested: args.math.clone(),
        dispatch: dispatch_label(dispatch).to_string(),
        math_backend: math_name,
        kernel_path,
        layout,
        mlp_impl,
        packed_layout,
        fast_math_error,
        fast_math_runtime_error,
        batch: args.batch,
        iters: args.iters,
        warmup: args.warmup,
        repeat: args.repeat,
        mean_ms,
        p50_ms,
        p95_ms,
        p99_ms,
        min_ms,
        max_ms,
        runs_ms,
        outlier_count: outlier_indices.len(),
        outlier_indices,
        between_repeat_cv,
        stage_timing_enabled,
        stage_stats_unit: if stage_timing_enabled { Some("cycles".to_string()) } else { None },
        stage_stats_total: if stage_timing_enabled { Some(stage_total) } else { None },
        stage_stats_per_iter: if stage_timing_enabled { Some(stage_per_iter) } else { None },
        stage_stats_pct: if stage_timing_enabled { Some(stage_pct) } else { None },
        cycles_per_iter_total: if stage_timing_enabled { Some(cycles_per_iter_total) } else { None },
    };

    let bundle_name = bundle_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("bundle");
    let out_dir = PathBuf::from(args.out_dir);
    create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    let out_path = out_dir.join(format!(
        "rust_perf_step1_{}_{}_{}_b{}.json",
        bundle_name, output.dispatch, output.math_backend, output.batch
    ));

    let mut out_file = File::create(&out_path).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(&output).map_err(|e| e.to_string())?;
    out_file.write_all(json.as_bytes()).map_err(|e| e.to_string())?;

    if args.audit_kernel {
        let audit_path = out_dir.join(format!(
            "rust_audit_step1_{}_{}_{}_b{}.tsv",
            bundle_name, output.dispatch, output.math_backend, output.batch
        ));
        let mut audit_file = File::create(&audit_path).map_err(|e| e.to_string())?;
        let audit_tsv = audit_trace_dump_tsv();
        audit_file
            .write_all(audit_tsv.as_bytes())
            .map_err(|e| e.to_string())?;
        println!("wrote {}", audit_path.display());
    }

    println!("wrote {}", out_path.display());
    Ok(())
}
