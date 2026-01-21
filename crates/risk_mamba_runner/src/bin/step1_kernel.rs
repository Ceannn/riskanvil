use clap::Parser;
use rand::{rngs::StdRng, SeedableRng};
use rand::seq::index;
use risk_mamba_kernel::{
    scratch_bytes, step1_model, step1_model_latency, CpuDispatch, DbgTaps, EmbSumMode, ForwardKind,
    GeluKind, HeadInputKind, KernelError, LayerWeights, LogitsKind, MathBackend, ModelConfig,
    PriorSource, SoftplusKind, StateSurgery, WeightsView, SLEEF_AVAILABLE,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(about = "Step1 kernel correctness gate (min/dbg)")]
struct Args {
    #[arg(long, alias = "bundle", default_value = "mamba/static_bundle_step1_v2_1")]
    bundle_dir: String,
    #[arg(long)]
    golden: Option<String>,
    #[arg(long, default_value = "min", value_parser = ["min", "dbg"])]
    mode: String,
    #[arg(long, default_value_t = 1e-5)]
    atol: f32,
    #[arg(long, default_value_t = 1e-4)]
    rtol: f32,
    #[arg(long)]
    state_atol: Option<f32>,
    #[arg(long)]
    state_rtol: Option<f32>,
    #[arg(long)]
    fail_fast: bool,
    #[arg(long)]
    dump_first_mismatch: bool,
    #[arg(long, default_value_t = 0)]
    topk: usize,
    #[arg(long)]
    sample: Option<usize>,
    #[arg(long)]
    dbg_sample: Option<usize>,
    #[arg(long, default_value_t = 0)]
    dbg_seed: u64,
    #[arg(long)]
    stop_after: Option<String>,
    #[arg(long, default_value = "scalar", value_parser = ["scalar", "avx2", "avx512"])]
    dispatch: String,
    #[arg(long, default_value = "exact", value_parser = ["exact", "fast", "sleef", "fast3_bf16", "fast_wild"])]
    math: String,
    #[arg(long)]
    latency: bool,
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

#[derive(Debug)]
struct NpyArray {
    descr: String,
    shape: Vec<usize>,
    fortran_order: bool,
    data: Vec<u8>,
}

impl NpyArray {
    fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    fn as_f32(&self) -> Result<Vec<f32>, String> {
        if self.descr != "<f4" {
            return Err(format!("expected <f4, got {}", self.descr));
        }
        let numel = self.numel();
        if self.data.len() != numel * 4 {
            return Err(format!("nbytes mismatch for f32: {}", self.data.len()));
        }
        let mut out = Vec::with_capacity(numel);
        for chunk in self.data.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    fn as_i64(&self) -> Result<Vec<i64>, String> {
        if self.descr != "<i8" {
            return Err(format!("expected <i8, got {}", self.descr));
        }
        let numel = self.numel();
        if self.data.len() != numel * 8 {
            return Err(format!("nbytes mismatch for i64: {}", self.data.len()));
        }
        let mut out = Vec::with_capacity(numel);
        for chunk in self.data.chunks_exact(8) {
            out.push(i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]));
        }
        Ok(out)
    }
}

#[derive(Clone, Copy)]
struct DiffOptions {
    atol: f32,
    rtol: f32,
    fail_fast: bool,
    dump_first: bool,
    topk: usize,
}

fn parse_npy(bytes: &[u8]) -> Result<NpyArray, String> {
    if bytes.len() < 10 {
        return Err("npy too short".to_string());
    }
    if &bytes[..6] != b"\x93NUMPY" {
        return Err("bad npy magic".to_string());
    }
    let major = bytes[6];
    let minor = bytes[7];
    let (header_len, header_start) = match (major, minor) {
        (1, 0) => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10)
        }
        (2, 0) => {
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12)
        }
        _ => return Err(format!("unsupported npy version {}.{}", major, minor)),
    };

    let header_end = header_start + header_len;
    if bytes.len() < header_end {
        return Err("npy header truncated".to_string());
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])
        .map_err(|_| "npy header utf8 error".to_string())?;

    let descr = extract_str(header, "descr")?;
    let fortran_order = extract_bool(header, "fortran_order")?;
    let shape = extract_shape(header)?;
    if fortran_order {
        return Err("fortran_order must be false".to_string());
    }

    let elem_size = match descr.as_str() {
        "<f4" => 4,
        "<i8" => 8,
        _ => return Err(format!("unsupported descr {}", descr)),
    };
    let numel: usize = shape.iter().product();
    let data_bytes = numel * elem_size;
    if bytes.len() < header_end + data_bytes {
        return Err("npy data truncated".to_string());
    }
    let data = bytes[header_end..header_end + data_bytes].to_vec();

    Ok(NpyArray {
        descr,
        shape,
        fortran_order,
        data,
    })
}

fn extract_str(header: &str, key: &str) -> Result<String, String> {
    let key1 = format!("'{}'", key);
    let key2 = format!("\"{}\"", key);
    let pos = header.find(&key1).or_else(|| header.find(&key2)).ok_or_else(|| {
        format!("missing key {} in npy header", key)
    })?;
    let rest = &header[pos..];
    let colon = rest.find(':').ok_or_else(|| "npy header missing ':'".to_string())?;
    let rest = rest[colon + 1..].trim_start();
    let quote = rest.chars().next().ok_or_else(|| "npy header empty".to_string())?;
    if quote != '\'' && quote != '"' {
        return Err("npy header missing quote".to_string());
    }
    let end = rest[1..]
        .find(quote)
        .ok_or_else(|| "npy header unterminated string".to_string())?;
    Ok(rest[1..1 + end].to_string())
}

fn extract_bool(header: &str, key: &str) -> Result<bool, String> {
    let key1 = format!("'{}'", key);
    let key2 = format!("\"{}\"", key);
    let pos = header.find(&key1).or_else(|| header.find(&key2)).ok_or_else(|| {
        format!("missing key {} in npy header", key)
    })?;
    let rest = &header[pos..];
    let colon = rest.find(':').ok_or_else(|| "npy header missing ':'".to_string())?;
    let rest = rest[colon + 1..].trim_start();
    if rest.starts_with("True") {
        Ok(true)
    } else if rest.starts_with("False") {
        Ok(false)
    } else {
        Err("npy header invalid bool".to_string())
    }
}

fn extract_shape(header: &str) -> Result<Vec<usize>, String> {
    let key1 = "'shape'";
    let key2 = "\"shape\"";
    let pos = header
        .find(key1)
        .or_else(|| header.find(key2))
        .ok_or_else(|| "missing shape in npy header".to_string())?;
    let rest = &header[pos..];
    let colon = rest.find(':').ok_or_else(|| "npy header missing ':'".to_string())?;
    let rest = rest[colon + 1..].trim_start();
    let lpar = rest.find('(').ok_or_else(|| "npy header missing '('".to_string())?;
    let rpar = rest.find(')').ok_or_else(|| "npy header missing ')'".to_string())?;
    let inner = &rest[lpar + 1..rpar];
    let mut shape = Vec::new();
    for part in inner.split(',') {
        let s = part.trim();
        if s.is_empty() {
            continue;
        }
        shape.push(s.parse::<usize>().map_err(|_| "invalid shape".to_string())?);
    }
    if shape.is_empty() {
        return Err("empty shape".to_string());
    }
    Ok(shape)
}

fn load_npz(path: &Path) -> Result<HashMap<String, NpyArray>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).map_err(|e| e.to_string())?;
        let name = entry.name().to_string();
        if !name.ends_with(".npy") {
            continue;
        }
        let mut data = Vec::new();
        entry.read_to_end(&mut data).map_err(|e| e.to_string())?;
        let array = parse_npy(&data)?;
        let key = name.trim_end_matches(".npy").to_string();
        out.insert(key, array);
    }
    Ok(out)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    format!("{:x}", digest)
}

fn load_manifest(path: &Path) -> Result<Manifest, String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&data).map_err(|e| e.to_string())
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
    if (ptr as usize) % std::mem::align_of::<f32>() != 0 {
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

#[derive(Clone, Copy)]
struct Mismatch {
    idx: usize,
    actual: f32,
    expected: f32,
    diff: f32,
    rel: f32,
}

fn format_index(idx: usize, shape: &[usize]) -> String {
    if shape.is_empty() {
        return idx.to_string();
    }
    let mut rem = idx;
    let mut coords = Vec::with_capacity(shape.len());
    for &dim in shape.iter().rev() {
        let v = if dim == 0 { 0 } else { rem % dim };
        coords.push(v);
        if dim > 0 {
            rem /= dim;
        }
    }
    coords.reverse();
    let mut out = String::from("[");
    for (i, v) in coords.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&v.to_string());
    }
    out.push(']');
    out
}

fn push_topk(vec: &mut Vec<Mismatch>, m: Mismatch, k: usize) {
    if k == 0 {
        return;
    }
    vec.push(m);
    vec.sort_by(|a, b| b.diff.partial_cmp(&a.diff).unwrap_or(std::cmp::Ordering::Equal));
    if vec.len() > k {
        vec.truncate(k);
    }
}

fn compare_f32(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    shape: &[usize],
    opts: DiffOptions,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("{} length mismatch", name));
    }
    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;
    let mut first: Option<Mismatch> = None;
    let mut topk: Vec<Mismatch> = Vec::new();

    for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
        if !a.is_finite() || !b.is_finite() {
            return Err(format!(
                "{} non-finite at idx={} actual={} expected={}",
                name,
                format_index(i, shape),
                a,
                b
            ));
        }
        let diff = (a - b).abs();
        let tol = opts.atol + opts.rtol * b.abs();
        if diff > tol {
            let rel = if b == 0.0 { diff } else { diff / b.abs() };
            let m = Mismatch {
                idx: i,
                actual: a,
                expected: b,
                diff,
                rel,
            };
            if first.is_none() {
                first = Some(m);
            }
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
            }
            if opts.topk > 0 {
                push_topk(&mut topk, m, opts.topk);
            }
            if opts.fail_fast {
                let idx_str = format_index(i, shape);
                return Err(format!(
                    "{} mismatch idx={} diff={} rel={} actual={} expected={}",
                    name, idx_str, diff, rel, a, b
                ));
            }
        }
    }

    if max_diff > 0.0 {
        if opts.dump_first {
            if let Some(m) = first {
                println!(
                    "{} first_mismatch idx={} diff={} rel={} actual={} expected={}",
                    name,
                    format_index(m.idx, shape),
                    m.diff,
                    m.rel,
                    m.actual,
                    m.expected
                );
            }
        }
        if opts.topk > 0 && !topk.is_empty() {
            println!("{} topk_mismatch:", name);
            for m in topk.iter() {
                println!(
                    "  idx={} diff={} rel={} actual={} expected={}",
                    format_index(m.idx, shape),
                    m.diff,
                    m.rel,
                    m.actual,
                    m.expected
                );
            }
        }
        return Err(format!(
            "{} mismatch max_diff={} idx={} shape={:?}",
            name,
            max_diff,
            format_index(max_idx, shape),
            shape
        ));
    }
    Ok(())
}

fn margin_matches_logits(logits: &[f32], margin: &[f32], batch: usize, n_class: usize) -> bool {
    if n_class < 2 || margin.len() != batch || logits.len() != batch * n_class {
        return false;
    }
    for b in 0..batch {
        let idx = b * n_class + 1;
        if (margin[b] - logits[idx]).abs() > 0.0 {
            return false;
        }
    }
    true
}

fn slice_sample_f32(buf: &[f32], shape: &[usize], sample: usize) -> Result<Vec<f32>, String> {
    if shape.is_empty() {
        return Err("cannot sample scalar".to_string());
    }
    let batch = shape[0];
    if sample >= batch {
        return Err(format!("sample {} out of range {}", sample, batch));
    }
    let stride: usize = shape[1..].iter().product();
    let start = sample * stride;
    let end = start + stride;
    if end > buf.len() {
        return Err("sample slice out of range".to_string());
    }
    Ok(buf[start..end].to_vec())
}

fn slice_state_layer_major(
    data: &[f32],
    batch: usize,
    n_layers: usize,
    d_inner: usize,
    d_state_pad: usize,
    sample: usize,
) -> Result<Vec<f32>, String> {
    if sample >= batch {
        return Err(format!("sample {} out of range {}", sample, batch));
    }
    let layer_stride = d_inner * d_state_pad;
    let mut out = vec![0.0f32; n_layers * layer_stride];
    for layer in 0..n_layers {
        let src = (layer * batch + sample) * layer_stride;
        let dst = layer * layer_stride;
        out[dst..dst + layer_stride].copy_from_slice(&data[src..src + layer_stride]);
    }
    Ok(out)
}

fn check_state_pad_zero(state: &[f32], d_state: usize, d_state_pad: usize) -> Result<(), String> {
    if d_state_pad == d_state {
        return Ok(());
    }
    if d_state_pad < d_state {
        return Err("d_state_pad < d_state".to_string());
    }
    let lanes = state.len() / d_state_pad;
    for lane in 0..lanes {
        let base = lane * d_state_pad;
        for j in d_state..d_state_pad {
            if state[base + j] != 0.0 {
                return Err("state pad lane must be 0".to_string());
            }
        }
    }
    Ok(())
}

fn check_a_log_pad(weights: &WeightsView, pad_value: f32) -> Result<(), String> {
    let cfg = &weights.cfg;
    let pad_start = cfg.d_state;
    let pad_end = cfg.d_state_pad;
    if pad_end <= pad_start {
        return Ok(());
    }
    for (layer_idx, layer) in weights.layers.iter().enumerate() {
        for i in 0..cfg.d_inner {
            let row = &layer.a_log[i * cfg.d_state_pad..][..cfg.d_state_pad];
            for &v in &row[pad_start..pad_end] {
                if v != pad_value {
                    return Err(format!("A_log pad mismatch at layer {}", layer_idx));
                }
            }
        }
    }
    Ok(())
}

struct GoldenDbg {
    arrays: HashMap<String, NpyArray>,
    diff_opts: DiffOptions,
    stop_after: Option<String>,
    sample: Option<usize>,
    sample_indices: Option<Vec<usize>>,
    orig_batch: usize,
}

impl GoldenDbg {
    fn new(
        arrays: HashMap<String, NpyArray>,
        diff_opts: DiffOptions,
        stop_after: Option<String>,
        sample: Option<usize>,
        sample_indices: Option<Vec<usize>>,
        orig_batch: usize,
    ) -> Self {
        Self {
            arrays,
            diff_opts,
            stop_after,
            sample,
            sample_indices,
            orig_batch,
        }
    }
}

impl DbgTaps for GoldenDbg {
    fn tap_f32(&mut self, key: &str, shape: &[usize], data: &[f32]) -> Result<(), KernelError> {
        let arr = self
            .arrays
            .get(key)
            .ok_or_else(|| KernelError::DebugTapError(format!("missing key {}", key)))?;
        let expected = arr
            .as_f32()
            .map_err(|e| KernelError::DebugTapError(format!("{}: {}", key, e)))?;
        let sample_indices = if self.sample.is_none() {
            self.sample_indices.as_deref()
        } else {
            None
        };
        if arr.shape == shape {
            if let Some(indices) = sample_indices {
                if shape.is_empty() || shape[0] != self.orig_batch {
                    return Err(KernelError::DebugTapError(format!(
                        "shape mismatch {} expect {:?} got {:?}",
                        key, arr.shape, shape
                    )));
                }
                let stride = expected.len() / self.orig_batch;
                let mut sample_shape = shape.to_vec();
                sample_shape[0] = 1;
                for &idx in indices {
                    let start = idx * stride;
                    let end = start + stride;
                    let expected_sample = &expected[start..end];
                    let actual_sample = &data[start..end];
                    let sample_key = format!("{}@sample{}", key, idx);
                    compare_f32(&sample_key, actual_sample, expected_sample, &sample_shape, self.diff_opts)
                        .map_err(|e| KernelError::DebugTapError(e))?;
                }
            } else {
                compare_f32(key, data, &expected, shape, self.diff_opts)
                    .map_err(|e| KernelError::DebugTapError(e))?;
            }
        } else if let Some(sample) = self.sample {
            if arr.shape.len() == shape.len()
                && !arr.shape.is_empty()
                && !shape.is_empty()
                && arr.shape[0] == self.orig_batch
                && shape[0] == 1
                && arr.shape[1..] == shape[1..]
            {
                let stride = expected.len() / self.orig_batch;
                let start = sample * stride;
                let end = start + stride;
                let expected_sample = &expected[start..end];
                compare_f32(key, data, expected_sample, shape, self.diff_opts)
                    .map_err(|e| KernelError::DebugTapError(e))?;
            } else {
                return Err(KernelError::DebugTapError(format!(
                    "shape mismatch {} expect {:?} got {:?}",
                    key, arr.shape, shape
                )));
            }
        } else {
            return Err(KernelError::DebugTapError(format!(
                "shape mismatch {} expect {:?} got {:?}",
                key, arr.shape, shape
            )));
        }
        if let Some(stop_key) = &self.stop_after {
            if stop_key == key {
                return Err(KernelError::DebugTapError(format!("stop-after {}", key)));
            }
        }
        Ok(())
    }
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
    let bundle_dir = PathBuf::from(args.bundle_dir);
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

    let dispatch = match args.dispatch.as_str() {
        "scalar" => CpuDispatch::Scalar,
        "avx2" => CpuDispatch::Avx2,
        "avx512" => CpuDispatch::Avx512,
        _ => return Err("invalid dispatch".to_string()),
    };
    let math_backend = match args.math.as_str() {
        "exact" => MathBackend::Exact,
        "fast" => MathBackend::Approx,
        "fast3_bf16" => MathBackend::FastBf16,
        "fast_wild" => MathBackend::FastWild,
        "sleef" => {
            if !SLEEF_AVAILABLE {
                return Err("sleef math backend not available".to_string());
            }
            MathBackend::Sleef
        }
        _ => return Err("invalid math backend".to_string()),
    };

    let mut weights = build_weights(&manifest, &blob)?;
    if matches!(math_backend, MathBackend::FastBf16 | MathBackend::FastWild) {
        weights.precompute_packed_bf16(16);
    }
    let cfg = weights.cfg.clone();

    let mode_dbg = args.mode == "dbg";
    if args.stop_after.is_some() && !mode_dbg {
        return Err("stop_after only valid in dbg mode".to_string());
    }
    if args.dbg_sample.is_some() && !mode_dbg {
        return Err("dbg_sample only valid in dbg mode".to_string());
    }
    if args.sample.is_some() && args.dbg_sample.is_some() {
        return Err("sample and dbg_sample are mutually exclusive".to_string());
    }

    let diff_opts = DiffOptions {
        atol: args.atol,
        rtol: args.rtol,
        fail_fast: args.fail_fast,
        dump_first: args.dump_first_mismatch,
        topk: args.topk,
    };
    let mut state_opts = diff_opts;
    state_opts.atol = args.state_atol.unwrap_or(args.atol);
    state_opts.rtol = args.state_rtol.unwrap_or(args.rtol);

    let min_npz = if let Some(path) = args.golden.as_ref() {
        let p = PathBuf::from(path);
        if p.is_relative() {
            bundle_dir.join(p)
        } else {
            p
        }
    } else {
        bundle_dir.join("golden_step1_min_v2_1.npz")
    };
    let dbg_npz = bundle_dir.join("golden_step1_dbg_v2_1.npz");

    let min = load_npz(&min_npz)?;
    let dbg = if mode_dbg { Some(load_npz(&dbg_npz)?) } else { None };

    let data_npz = if mode_dbg {
        dbg.as_ref().ok_or_else(|| "missing dbg npz".to_string())?
    } else {
        &min
    };

    let x_in_full = data_npz
        .get("x_in")
        .ok_or_else(|| "missing x_in".to_string())?
        .as_f32()?;
    let logits_exp_full = data_npz
        .get("logits")
        .ok_or_else(|| "missing logits".to_string())?
        .as_f32()?;
    let margin_exp_full = data_npz
        .get("margin")
        .ok_or_else(|| "missing margin".to_string())?
        .as_f32()?;

    let orig_batch = data_npz
        .get("x_in")
        .ok_or_else(|| "missing x_in".to_string())?
        .shape[0];
    let dbg_sample_indices = if let Some(count) = args.dbg_sample {
        if count == 0 {
            None
        } else {
            if count > orig_batch {
                return Err(format!("dbg_sample {} out of range {}", count, orig_batch));
            }
            let mut rng = StdRng::seed_from_u64(args.dbg_seed);
            let mut indices: Vec<usize> = index::sample(&mut rng, orig_batch, count).into_vec();
            indices.sort_unstable();
            Some(indices)
        }
    } else {
        None
    };

    let x_shape = data_npz.get("x_in").unwrap().shape.clone();
    if x_shape != vec![orig_batch, cfg.d_model] {
        return Err(format!("x_in shape mismatch {:?}", x_shape));
    }
    let logits_shape = data_npz.get("logits").unwrap().shape.clone();
    if logits_shape != vec![orig_batch, cfg.n_class] {
        return Err(format!("logits shape mismatch {:?}", logits_shape));
    }
    let margin_shape = data_npz.get("margin").unwrap().shape.clone();
    if margin_shape != vec![orig_batch] {
        return Err(format!("margin shape mismatch {:?}", margin_shape));
    }

    if !margin_matches_logits(&logits_exp_full, &margin_exp_full, orig_batch, cfg.n_class) {
        return Err("margin does not match logits[:,1]".to_string());
    }

    check_a_log_pad(&weights, manifest.model_config.pad_policy.a_log_pad_value)?;

    let mut state_in_full = if mode_dbg {
        let dbg_arrays = dbg.as_ref().ok_or_else(|| "missing dbg npz".to_string())?;
        let mut buf = vec![0.0f32; orig_batch * cfg.n_layers * cfg.d_inner * cfg.d_state_pad];
        let layer_stride = cfg.d_inner * cfg.d_state_pad;
        for layer in 0..cfg.n_layers {
            let key = format!("layer{}/state_in", layer);
            let arr = dbg_arrays
                .get(&key)
                .ok_or_else(|| format!("missing {}", key))?;
            let expect = vec![orig_batch, cfg.d_inner, cfg.d_state_pad];
            if arr.shape != expect {
                return Err(format!("{} shape mismatch {:?}", key, arr.shape));
            }
            let data = arr.as_f32()?;
            for b in 0..orig_batch {
                let src = b * layer_stride;
                let dst = (layer * orig_batch + b) * layer_stride;
                buf[dst..dst + layer_stride].copy_from_slice(&data[src..src + layer_stride]);
            }
        }
        buf
    } else {
        min.get("state_in")
            .ok_or_else(|| "missing state_in".to_string())?
            .as_f32()?
    };

    let mut state_out_full = if mode_dbg {
        Vec::new()
    } else {
        min.get("state_out")
            .ok_or_else(|| "missing state_out".to_string())?
            .as_f32()?
    };

    let expect_state = vec![orig_batch, cfg.n_layers, cfg.d_inner, cfg.d_state_pad];
    let state_shape = if mode_dbg {
        expect_state.clone()
    } else {
        min.get("state_in").unwrap().shape.clone()
    };
    if state_shape != expect_state {
        return Err(format!("state_in shape mismatch {:?}", state_shape));
    }

    check_state_pad_zero(&state_in_full, cfg.d_state, cfg.d_state_pad)?;
    if !mode_dbg {
        check_state_pad_zero(&state_out_full, cfg.d_state, cfg.d_state_pad)?;
    }

    let sample = args.sample;
    let batch = if sample.is_some() { 1 } else { orig_batch };

    let x_in = if let Some(s) = sample {
        slice_sample_f32(&x_in_full, &x_shape, s)?
    } else {
        x_in_full
    };
    let state_in = if let Some(s) = sample {
        if mode_dbg {
            slice_state_layer_major(
                &state_in_full,
                orig_batch,
                cfg.n_layers,
                cfg.d_inner,
                cfg.d_state_pad,
                s,
            )?
        } else {
            slice_sample_f32(&state_in_full, &state_shape, s)?
        }
    } else {
        state_in_full
    };
    let state_out = if mode_dbg {
        Vec::new()
    } else if let Some(s) = sample {
        slice_sample_f32(&state_out_full, &state_shape, s)?
    } else {
        state_out_full
    };
    let logits_exp = if let Some(s) = sample {
        slice_sample_f32(&logits_exp_full, &logits_shape, s)?
    } else {
        logits_exp_full
    };
    let margin_exp = if let Some(s) = sample {
        slice_sample_f32(&margin_exp_full, &margin_shape, s)?
    } else {
        margin_exp_full
    };

    let scratch_len = scratch_bytes(&cfg, batch);
    let mut scratch = AlignedBuf::new(scratch_len, 64)?;

    let mut state = state_in.clone();
    let mut logits = vec![0.0f32; batch * cfg.n_class];
    let mut margin = vec![0.0f32; batch];

    let mut dbg_sink = dbg.map(|m| {
        GoldenDbg::new(
            m,
            diff_opts,
            args.stop_after.clone(),
            sample,
            dbg_sample_indices.clone(),
            orig_batch,
        )
    });

    let dbg_ref: Option<&mut dyn DbgTaps> = dbg_sink.as_mut().map(|s| s as &mut dyn DbgTaps);

    let want_latency =
        args.latency || dispatch != CpuDispatch::Scalar || math_backend != MathBackend::Exact;
    let use_latency = want_latency && dbg_ref.is_none() && batch == 1;
    let run = if use_latency {
        step1_model_latency(
            &cfg,
            &weights,
            dispatch,
            math_backend,
            batch,
            &x_in,
            &mut state,
            scratch.as_mut_slice(),
            &mut logits,
            &mut margin,
            None,
            None,
        )
    } else {
        step1_model(
            &cfg,
            &weights,
            dispatch,
            math_backend,
            batch,
            &x_in,
            &mut state,
            scratch.as_mut_slice(),
            &mut logits,
            &mut margin,
            dbg_ref,
            None,
            None,
        )
    };
    if let Err(e) = run {
        let msg = e.to_string();
        if msg.starts_with("debug tap error: stop-after ") {
            println!("OK: {}", msg);
            return Ok(());
        }
        return Err(msg);
    }

    compare_f32("logits", &logits, &logits_exp, &[batch, cfg.n_class], diff_opts)?;
    compare_f32("margin", &margin, &margin_exp, &[batch], diff_opts)?;
    if !mode_dbg {
        compare_f32(
            "state_out",
            &state,
            &state_out,
            &[batch, cfg.n_layers, cfg.d_inner, cfg.d_state_pad],
            state_opts,
        )?;
    }

    println!("OK: golden_{}", args.mode);
    Ok(())
}
