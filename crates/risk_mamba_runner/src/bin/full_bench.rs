use clap::Parser;
use risk_mamba_kernel::{
    audit_trace_dump_tsv,
    audit_trace_reset,
    full_stateless_forward,
    kernel_path_tag,
    layout_tag,
    mlp_fused_enabled,
    scratch_bytes_full,
    weights_pack_stats,
    CpuDispatch,
    EmbSumMode,
    ForwardKind,
    GeluKind,
    HeadInputKind,
    LayerWeights,
    LogitsKind,
    MathBackend,
    ModelConfig,
    PerfCounters,
    PriorSource,
    SoftplusKind,
    StageStats,
    StateSurgery,
    WeightsView,
    SLEEF_AVAILABLE,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use libc::{mlockall, MCL_CURRENT, MCL_FUTURE};

#[cfg(feature = "v1_1_experiment")]
const EXPECT_STATIC_DIM_TOTAL: usize = 418;
#[cfg(feature = "v1_1_experiment")]
const EXPECT_MICRO_STATE_DIM: usize = 32;
#[cfg(not(feature = "v1_1_experiment"))]
const EXPECT_STATIC_DIM_TOTAL: usize = 386;
#[cfg(not(feature = "v1_1_experiment"))]
const EXPECT_MICRO_STATE_DIM: usize = 0;

#[derive(Parser, Debug)]
#[command(about = "Full stateless kernel benchmark (B=1)")]
struct Args {
    #[arg(long, alias = "bundle", default_value = "mamba/static_bundle_full_v1_mainline")]
    bundle_dir: String,
    #[arg(long)]
    golden: Option<String>,
    #[arg(long, default_value = "custom", value_parser = ["custom", "exact_baseline", "fast_perf"])]
    preset: String,
    #[arg(long, default_value = "hot", value_parser = ["cold", "warm", "hot"])]
    state: String,
    #[arg(long, default_value_t = 1)]
    batch: usize,
    #[arg(long, default_value_t = 200)]
    iters: usize,
    #[arg(long, default_value_t = 50)]
    warmup: usize,
    #[arg(long, default_value_t = 5)]
    repeat: usize,
    #[arg(long, default_value_t = 20)]
    auto_dispatch_iters: usize,
    #[arg(long, default_value_t = 5)]
    auto_dispatch_warmup: usize,
    #[arg(long, default_value = "auto", value_parser = ["auto", "scalar", "avx2", "avx512"])]
    dispatch: String,
    #[arg(long, default_value = "auto", value_parser = ["auto", "fast", "exact", "sleef", "fast3_bf16", "fast_wild"])]
    math: String,
    #[arg(long, default_value = "outputs-7945hx")]
    out_dir: String,
    #[arg(long)]
    audit_kernel: bool,
    #[arg(long)]
    no_fallback: bool,
}

#[derive(Deserialize)]
struct Manifest {
    format: String,
    schema_version: u32,
    schema_hash: Option<String>,
    feature_schema_sha256: Option<String>,
    micro_state_schema_sha256: Option<String>,
    weights_sha256: String,
    weights_endianness: String,
    weights_file: String,
    alignment_bytes: usize,
    model_config: ManifestConfig,
    tensors: Vec<TensorSpec>,
}

#[derive(Deserialize)]
struct ManifestConfig {
    forward_kind: String,
    n_layers: usize,
    seq_len: usize,
    input_ids_len: usize,
    feature_dim: usize,
    vocab_size: usize,
    pad_id: i64,
    missing_id: i64,
    static_dim_base: usize,
    static_dim_total: usize,
    micro_state_dim: Option<usize>,
    d_model: usize,
    d_inner: usize,
    d_mlp: usize,
    d_state: usize,
    conv_kernel: usize,
    conv_groups: usize,
    ln_eps: f32,
    gelu_kind: String,
    softplus_kind: String,
    softplus_beta: f32,
    softplus_threshold: f32,
    pad_policy: Option<PadPolicy>,
    logits_kind: String,
    n_class: usize,
    head_input_kind: String,
    state_surgery: String,
    prior_source: String,
    residual_alpha: f32,
    near_band: f32,
    thr01_fixed_fpr: Option<f32>,
    thr_fixedfpr_gate: Option<f32>,
    thr_anchor: Option<f32>,
    teacher_static_col: usize,
    use_dt: Option<bool>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PadPolicy {
    None(String),
    Values {
        #[serde(rename = "A_log_pad_value")]
        a_log_pad_value: f32,
        #[serde(rename = "BC_pad_value")]
        bc_pad_value: f32,
        #[serde(rename = "state_pad_value")]
        state_pad_value: f32,
    },
}

#[derive(Deserialize)]
struct FeatureSchema {
    input_ids_len: usize,
    static_base_dim: usize,
    micro_state_dim: usize,
    static_total_dim: usize,
    missing_id: Option<i64>,
    micro_state_schema_sha256: Option<String>,
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

#[derive(Clone, Copy)]
struct DerivedDims {
    dt_rank: usize,
    d_state_pad: usize,
    emb_vocab: usize,
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
        match self.descr.as_str() {
            "<i8" => {
                let numel = self.numel();
                if self.data.len() != numel * 8 {
                    return Err(format!("nbytes mismatch for i64: {}", self.data.len()));
                }
                let mut out = Vec::with_capacity(numel);
                for chunk in self.data.chunks_exact(8) {
                    out.push(i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]));
                }
                Ok(out)
            }
            "<i4" => {
                let numel = self.numel();
                if self.data.len() != numel * 4 {
                    return Err(format!("nbytes mismatch for i32: {}", self.data.len()));
                }
                let mut out = Vec::with_capacity(numel);
                for chunk in self.data.chunks_exact(4) {
                    let v = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    out.push(v as i64);
                }
                Ok(out)
            }
            _ => Err(format!("expected <i8 or <i4, got {}", self.descr)),
        }
    }
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
        "<i4" => 4,
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
    if !rest.starts_with('(') {
        return Err("npy header invalid shape".to_string());
    }
    let end = rest.find(')').ok_or_else(|| "npy header missing ')'".to_string())?;
    let inner = &rest[1..end];
    let mut shape = Vec::new();
    for part in inner.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let val = part
            .parse::<usize>()
            .map_err(|_| "npy header invalid shape number".to_string())?;
        shape.push(val);
    }
    Ok(shape)
}

fn load_npz(path: &Path) -> Result<HashMap<String, NpyArray>, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    let mut out = HashMap::new();
    for i in 0..zip.len() {
        let mut file = zip.by_index(i).map_err(|e| e.to_string())?;
        let name = file.name().to_string();
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        let array = parse_npy(&bytes)?;
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
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

fn load_feature_schema(path: &Path) -> Result<FeatureSchema, String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&data).map_err(|e| e.to_string())
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    Ok(sha256_hex(&buf))
}

fn derive_dims(manifest: &Manifest) -> Result<DerivedDims, String> {
    let mut map = HashMap::new();
    for spec in &manifest.tensors {
        map.insert(spec.name.clone(), spec.clone());
    }
    let get = |name: &str| -> Result<&TensorSpec, String> {
        map.get(name).ok_or_else(|| format!("missing tensor {}", name))
    };
    let emb = get("emb.weight")?;
    if emb.shape.len() != 2 {
        return Err("emb.weight shape rank != 2".to_string());
    }
    if emb.shape[1] != manifest.model_config.d_model {
        return Err("emb.weight d_model mismatch".to_string());
    }
    let emb_vocab = emb.shape[0];
    let vocab = manifest.model_config.vocab_size;
    if emb_vocab != vocab && emb_vocab != vocab + 1 {
        return Err(format!(
            "emb.weight rows {} != vocab_size {} (+1 allowed)",
            emb_vocab, vocab
        ));
    }

    let dt_proj = get("layers.0.mamba.dt_proj.weight")?;
    if dt_proj.shape.len() != 2 {
        return Err("dt_proj.weight shape rank != 2".to_string());
    }
    let dt_rank = dt_proj.shape[1];

    let a_log = get("layers.0.mamba.A_log")?;
    if a_log.shape.len() != 2 {
        return Err("A_log shape rank != 2".to_string());
    }
    let d_state_pad = a_log.shape[1];
    if d_state_pad < manifest.model_config.d_state {
        return Err("d_state_pad < d_state".to_string());
    }

    Ok(DerivedDims {
        dt_rank,
        d_state_pad,
        emb_vocab,
    })
}

fn sanitize_input_ids(input_ids: &mut [i64], vocab_size: usize, missing_id: i64) -> usize {
    let mut remapped = 0usize;
    for v in input_ids.iter_mut() {
        if *v < 0 || (*v as usize) >= vocab_size {
            *v = missing_id;
            remapped += 1;
        }
    }
    remapped
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

fn validate_manifest_shapes(manifest: &Manifest, dims: DerivedDims) -> Result<(), String> {
    let cfg = &manifest.model_config;
    let mut map = HashMap::new();
    for spec in &manifest.tensors {
        map.insert(spec.name.clone(), spec.clone());
    }
    let check = |name: &str, shape: &[usize]| -> Result<(), String> {
        let spec = map.get(name).ok_or_else(|| format!("missing tensor {}", name))?;
        if spec.offset % manifest.alignment_bytes != 0 {
            return Err(format!(
                "tensor {} offset {} not aligned to {}",
                name, spec.offset, manifest.alignment_bytes
            ));
        }
        if spec.shape != shape {
            return Err(format!("shape mismatch {} {:?} != {:?}", name, spec.shape, shape));
        }
        let expect_bytes: usize = shape.iter().product::<usize>() * 4;
        if spec.nbytes != expect_bytes {
            return Err(format!("nbytes mismatch {} {} != {}", name, spec.nbytes, expect_bytes));
        }
        Ok(())
    };

    check("emb.weight", &[dims.emb_vocab, cfg.d_model])?;
    check("pos.weight", &[cfg.seq_len, cfg.d_model])?;
    check("static_proj.weight", &[cfg.d_model, cfg.static_dim_total])?;
    check("static_proj.bias", &[cfg.d_model])?;
    if cfg.state_surgery == "gate_all" {
        check("prior_gate.weight", &[cfg.d_model, cfg.d_model])?;
        check("prior_gate.bias", &[cfg.d_model])?;
    }
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
        if map.contains_key(&format!("{}.mamba.in_proj.bias", prefix)) {
            check(&format!("{}.mamba.in_proj.bias", prefix), &[2 * cfg.d_inner])?;
        }
        check(&format!("{}.mamba.conv1d.weight", prefix), &[cfg.d_inner, 1, cfg.conv_kernel])?;
        check(&format!("{}.mamba.conv1d.bias", prefix), &[cfg.d_inner])?;
        check(
            &format!("{}.mamba.x_proj.weight", prefix),
            &[dims.dt_rank + 2 * dims.d_state_pad, cfg.d_inner],
        )?;
        check(&format!("{}.mamba.dt_proj.weight", prefix), &[cfg.d_inner, dims.dt_rank])?;
        check(&format!("{}.mamba.dt_proj.bias", prefix), &[cfg.d_inner])?;
        check(&format!("{}.mamba.A_log", prefix), &[cfg.d_inner, dims.d_state_pad])?;
        check(&format!("{}.mamba.D", prefix), &[cfg.d_inner])?;
        check(&format!("{}.mamba.out_proj.weight", prefix), &[cfg.d_model, cfg.d_inner])?;
        if map.contains_key(&format!("{}.mamba.out_proj.bias", prefix)) {
            check(&format!("{}.mamba.out_proj.bias", prefix), &[cfg.d_model])?;
        }
        check(&format!("{}.mlp.fc1.weight", prefix), &[cfg.d_mlp, cfg.d_model])?;
        check(&format!("{}.mlp.fc1.bias", prefix), &[cfg.d_mlp])?;
        check(&format!("{}.mlp.fc2.weight", prefix), &[cfg.d_model, cfg.d_mlp])?;
        check(&format!("{}.mlp.fc2.bias", prefix), &[cfg.d_model])?;
    }

    Ok(())
}

fn parse_state_surgery(value: &str) -> Result<StateSurgery, String> {
    match value {
        "none" => Ok(StateSurgery::None),
        "add_first" => Ok(StateSurgery::AddFirst),
        "add_all" => Ok(StateSurgery::AddAll),
        "gate_all" => Ok(StateSurgery::GateAll),
        _ => Err(format!("unsupported state_surgery {}", value)),
    }
}

fn parse_prior_source(value: &str) -> Result<PriorSource, String> {
    match value {
        "none" => Ok(PriorSource::None),
        "static" => Ok(PriorSource::Static),
        "uid" => Ok(PriorSource::Uid),
        "auto" => Ok(PriorSource::Auto),
        _ => Err(format!("unsupported prior_source {}", value)),
    }
}

fn build_weights<'a>(
    manifest: &Manifest,
    dims: DerivedDims,
    blob: &'a [u8],
) -> Result<WeightsView<'a>, String> {
    let cfg = ModelConfig {
        forward_kind: ForwardKind::FullStateless,
        n_layers: manifest.model_config.n_layers,
        seq_len: manifest.model_config.seq_len,
        feature_dim: manifest.model_config.feature_dim,
        vocab_size: manifest.model_config.vocab_size,
        static_dim_total: manifest.model_config.static_dim_total,
        micro_state_dim: manifest.model_config.micro_state_dim.unwrap_or(0),
        d_model: manifest.model_config.d_model,
        d_inner: manifest.model_config.d_inner,
        d_state: manifest.model_config.d_state,
        d_state_pad: dims.d_state_pad,
        d_mlp: manifest.model_config.d_mlp,
        conv_kernel: manifest.model_config.conv_kernel,
        dt_rank: dims.dt_rank,
        ln_eps1: manifest.model_config.ln_eps,
        ln_eps2: manifest.model_config.ln_eps,
        gelu_kind: GeluKind::Erf,
        softplus_kind: SoftplusKind::Exact,
        softplus_beta: manifest.model_config.softplus_beta,
        softplus_threshold: manifest.model_config.softplus_threshold,
        logits_kind: LogitsKind::NativeTwoClass,
        head_input_kind: HeadInputKind::HeadIn,
        n_class: manifest.model_config.n_class,
        state_surgery: parse_state_surgery(&manifest.model_config.state_surgery)?,
        prior_source: parse_prior_source(&manifest.model_config.prior_source)?,
        emb_sum_mode: EmbSumMode::Fast,
        prior_alpha: 1.0,
    };

    let mut map = HashMap::new();
    for spec in &manifest.tensors {
        map.insert(spec.name.clone(), spec.clone());
    }

    let get = |name: &str| -> Result<&TensorSpec, String> {
        map.get(name).ok_or_else(|| format!("missing tensor {}", name))
    };

    let emb_w = {
        let spec = get("emb.weight")?;
        check_tensor_sha(blob, spec)?;
        slice_tensor(blob, spec)?
    };
    if dims.emb_vocab == manifest.model_config.vocab_size + 1 {
        let d_model = manifest.model_config.d_model;
        let start = (dims.emb_vocab - 1) * d_model;
        let end = start + d_model;
        let row = &emb_w[start..end];
        let mut max_abs = 0.0f32;
        for &v in row {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        if max_abs > 1e-6 {
            return Err(format!(
                "emb.weight extra row not zero (max_abs={})",
                max_abs
            ));
        }
    }
    let pos_w = {
        let spec = get("pos.weight")?;
        check_tensor_sha(blob, spec)?;
        slice_tensor(blob, spec)?
    };
    let static_proj_w = {
        let spec = get("static_proj.weight")?;
        check_tensor_sha(blob, spec)?;
        slice_tensor(blob, spec)?
    };
    let static_proj_b = {
        let spec = get("static_proj.bias")?;
        check_tensor_sha(blob, spec)?;
        slice_tensor(blob, spec)?
    };

    let prior_gate_w = if manifest.model_config.state_surgery == "gate_all" {
        let spec = get("prior_gate.weight")?;
        check_tensor_sha(blob, spec)?;
        Some(slice_tensor(blob, spec)?)
    } else {
        None
    };
    let prior_gate_b = if manifest.model_config.state_surgery == "gate_all" {
        let spec = get("prior_gate.bias")?;
        check_tensor_sha(blob, spec)?;
        Some(slice_tensor(blob, spec)?)
    } else {
        None
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
        let in_proj_b = map
            .get(&format!("{}.mamba.in_proj.bias", prefix))
            .map(|spec| {
                check_tensor_sha(blob, spec)?;
                slice_tensor(blob, spec)
            })
            .transpose()?;
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
        let out_proj_b = map
            .get(&format!("{}.mamba.out_proj.bias", prefix))
            .map(|spec| {
                check_tensor_sha(blob, spec)?;
                slice_tensor(blob, spec)
            })
            .transpose()?;
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
            in_proj_b,
            in_proj_b_zero: if in_proj_b.is_some() {
                None
            } else {
                Some(vec![0.0f32; cfg.d_inner * 2])
            },
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
            out_proj_b,
            out_proj_b_zero: if out_proj_b.is_some() {
                None
            } else {
                Some(vec![0.0f32; cfg.d_model])
            },
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
    weights.emb_w = Some(emb_w);
    weights.pos_w = Some(pos_w);
    weights.static_proj_w = Some(static_proj_w);
    weights.static_proj_b = Some(static_proj_b);
    weights.prior_gate_w = prior_gate_w;
    weights.prior_gate_b = prior_gate_b;
    weights.precompute_a_pre();
    weights.precompute_packed(16);
    Ok(weights)
}

#[derive(Debug, Clone, Default, Serialize)]
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

#[derive(Serialize, Clone)]
struct DispatchAutoDecision {
    selected: String,
    avx2_ms: Option<f64>,
    avx512_ms: Option<f64>,
    iters: usize,
    warmup: usize,
    chosen_reason: String,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0) * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = rank - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / (values.len() as f64);
    var.sqrt()
}

#[derive(Serialize)]
struct LinearShape {
    m: usize,
    k: usize,
    n: usize,
}

#[derive(Serialize)]
struct BenchOutput {
    bundle: String,
    forward_kind: String,
    schema_hash: String,
    seq_len: usize,
    static_dim_total: usize,
    micro_state_dim: usize,
    dispatch: String,
    math_backend: String,
    kernel_path: String,
    layout: String,
    mlp_impl: String,
    preset: String,
    state: String,
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
    cv: f64,
    outlier_indices: Vec<usize>,
    stage_total: StageStatsOutput,
    stage_per_iter: StageStatsOutput,
    mlp_fc1_call_count: u64,
    mlp_fc1_layer_count: u64,
    mlp_fc1_shape: LinearShape,
    mlp_fc1_impl: String,
    dispatch_auto: Option<DispatchAutoDecision>,
    weights_pack_once: u64,
    weights_pack_repeated: u64,
    linear_shapes: HashMap<String, LinearShape>,
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

fn resolve_dispatch(name: &str) -> Result<CpuDispatch, String> {
    match name {
        "scalar" => Ok(CpuDispatch::Scalar),
        "avx2" => Ok(CpuDispatch::Avx2),
        "avx512" => Ok(CpuDispatch::Avx512),
        "auto" => {
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx512f") {
                    return Ok(CpuDispatch::Avx512);
                }
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                    return Ok(CpuDispatch::Avx2);
                }
            }
            Ok(CpuDispatch::Scalar)
        }
        other => Err(format!("invalid dispatch {}", other)),
    }
}

fn dispatch_candidates() -> Vec<CpuDispatch> {
    let mut out = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            out.push(CpuDispatch::Avx2);
        }
        if std::is_x86_feature_detected!("avx512f") {
            out.push(CpuDispatch::Avx512);
        }
    }
    if out.is_empty() {
        out.push(CpuDispatch::Scalar);
    }
    out
}

fn microbench_dispatch(
    cfg: &ModelConfig,
    w: &WeightsView,
    math_backend: MathBackend,
    input_ids: &[i64],
    static_total: &[f32],
    scratch_bytes: &mut [u8],
    logits: &mut [f32],
    margin: &mut [f32],
    iters: usize,
    warmup: usize,
) -> DispatchAutoDecision {
    let mut best = CpuDispatch::Scalar;
    let mut best_ms = f64::INFINITY;
    let mut avx2_ms = None;
    let mut avx512_ms = None;
    let mut chosen_reason = "best_ms".to_string();

    for &dispatch in &dispatch_candidates() {
        for _ in 0..warmup {
            let _ = full_stateless_forward(
                cfg,
                w,
                dispatch,
                math_backend,
                input_ids,
                static_total,
                None,
                scratch_bytes,
                logits,
                margin,
                None,
                None,
                None,
                None,
            );
        }
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = full_stateless_forward(
                cfg,
                w,
                dispatch,
                math_backend,
                input_ids,
                static_total,
                None,
                scratch_bytes,
                logits,
                margin,
                None,
                None,
                None,
                None,
            );
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        match dispatch {
            CpuDispatch::Avx2 => avx2_ms = Some(ms),
            CpuDispatch::Avx512 => avx512_ms = Some(ms),
            CpuDispatch::Scalar => {}
        }
        if ms < best_ms {
            best_ms = ms;
            best = dispatch;
        }
    }

    if let (Some(a2), Some(a512)) = (avx2_ms, avx512_ms) {
        let margin = std::env::var("RISK_MAMBA_AVX512_MARGIN")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.98);
        if a512 <= a2 * margin {
            best = CpuDispatch::Avx512;
            chosen_reason = format!("avx512_faster_margin={:.3}", margin);
        } else {
            best = CpuDispatch::Avx2;
            chosen_reason = format!("avx2_within_margin={:.3}", margin);
        }
    } else if best == CpuDispatch::Scalar {
        chosen_reason = "scalar_only".to_string();
    }

    DispatchAutoDecision {
        selected: dispatch_label(best).to_string(),
        avx2_ms,
        avx512_ms,
        iters,
        warmup,
        chosen_reason,
    }
}

fn dispatch_label(dispatch: CpuDispatch) -> &'static str {
    match dispatch {
        CpuDispatch::Scalar => "scalar",
        CpuDispatch::Avx2 => "avx2",
        CpuDispatch::Avx512 => "avx512",
    }
}

fn apply_preset(args: &mut Args) -> Result<(), String> {
    match args.preset.as_str() {
        "custom" => Ok(()),
        "exact_baseline" => {
            args.warmup = 10;
            args.iters = 50;
            args.repeat = 2;
            Ok(())
        }
        "fast_perf" => {
            args.warmup = 100;
            args.iters = 1000;
            args.repeat = 5;
            Ok(())
        }
        other => Err(format!("invalid preset {}", other)),
    }
}

fn maybe_mlockall() {
    let enabled = std::env::var("RISK_MAMBA_MLOCK")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(true);
    if !enabled {
        return;
    }
    unsafe {
        if mlockall(MCL_CURRENT | MCL_FUTURE) != 0 {
            eprintln!("[mlock] warning: mlockall failed");
        } else {
            eprintln!("[mlock] mlockall ok");
        }
    }
}

fn pretouch_read(buf: &[u8]) {
    let enabled = std::env::var("RISK_MAMBA_PRETOUCH")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(true);
    if !enabled {
        return;
    }
    let page = 4096usize;
    let mut i = 0usize;
    while i < buf.len() {
        unsafe {
            std::ptr::read_volatile(buf.as_ptr().add(i));
        }
        i += page;
    }
}

fn pretouch_write(buf: &mut [u8]) {
    let enabled = std::env::var("RISK_MAMBA_PRETOUCH")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(true);
    if !enabled {
        return;
    }
    let page = 4096usize;
    let mut i = 0usize;
    while i < buf.len() {
        unsafe {
            std::ptr::write_volatile(buf.as_mut_ptr().add(i), 0);
        }
        i += page;
    }
}

fn main() -> Result<(), String> {
    let mut args = Args::parse();
    apply_preset(&mut args)?;
    if args.audit_kernel {
        std::env::set_var("RISK_MAMBA_AUDIT_KERNEL", "1");
        audit_trace_reset();
    }
    std::env::set_var("RISK_MAMBA_STAGE_SAMPLE", "1");
    if args.no_fallback {
        std::env::set_var("RISK_MAMBA_NO_FALLBACK", "1");
    }
    if args.iters == 0 || args.repeat == 0 {
        return Err("iters and repeat must be > 0".to_string());
    }
    if args.batch != 1 {
        return Err("full_bench only supports batch=1".to_string());
    }
    let bundle_dir = PathBuf::from(&args.bundle_dir);
    let manifest_path = bundle_dir.join("weights_manifest.json");
    let manifest = load_manifest(&manifest_path)?;
    let feature_schema_path = bundle_dir.join("feature_schema.json");
    let feature_schema = load_feature_schema(&feature_schema_path)?;

    #[cfg(feature = "v1_1_experiment")]
    if manifest.format != "fraud_full_stateless_bundle_v1_1" {
        return Err("format must be fraud_full_stateless_bundle_v1_1".to_string());
    }
    #[cfg(not(feature = "v1_1_experiment"))]
    if manifest.format != "fraud_full_stateless_bundle_v1_mainline" {
        return Err("format must be fraud_full_stateless_bundle_v1_mainline".to_string());
    }
    if manifest.model_config.forward_kind != "full_stateless" {
        return Err("forward_kind must be full_stateless".to_string());
    }
    if manifest.schema_version != 2 {
        return Err("schema_version must be 2".to_string());
    }
    if manifest.weights_endianness != "little" {
        return Err("weights_endianness must be little".to_string());
    }
    if manifest.model_config.gelu_kind != "erf" {
        return Err("gelu_kind must be erf for correctness gate".to_string());
    }
    if manifest.model_config.softplus_kind != "beta_threshold" {
        return Err("softplus_kind must be beta_threshold".to_string());
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
    if manifest.model_config.seq_len != 128 {
        return Err("seq_len must be 128".to_string());
    }
    if manifest.model_config.input_ids_len != manifest.model_config.seq_len {
        return Err("input_ids_len must equal seq_len".to_string());
    }
    let micro_state_dim = manifest.model_config.micro_state_dim.unwrap_or(0);
    if manifest.model_config.static_dim_total != EXPECT_STATIC_DIM_TOTAL {
        #[cfg(not(feature = "v1_1_experiment"))]
        if manifest.model_config.static_dim_total == 418 {
            return Err("static_dim_total=418 is v1_1 experiment; rebuild with --features v1_1_experiment".to_string());
        }
        return Err(format!(
            "static_dim_total must be {}",
            EXPECT_STATIC_DIM_TOTAL
        ));
    }
    if micro_state_dim != EXPECT_MICRO_STATE_DIM {
        #[cfg(not(feature = "v1_1_experiment"))]
        if micro_state_dim == 32 {
            return Err("micro_state_dim=32 is v1_1 experiment; rebuild with --features v1_1_experiment".to_string());
        }
        return Err(format!(
            "micro_state_dim must be {}",
            EXPECT_MICRO_STATE_DIM
        ));
    }
    if manifest.model_config.vocab_size != 4097 {
        return Err("vocab_size must be 4097".to_string());
    }
    if manifest.model_config.feature_dim != 1 {
        return Err("feature_dim must be 1".to_string());
    }
    if manifest.model_config.teacher_static_col != 1 {
        return Err("teacher_static_col must be 1".to_string());
    }
    {
        let feature_hash = file_sha256(&feature_schema_path)?;
        let schema_hash = manifest
            .schema_hash
            .as_deref()
            .ok_or_else(|| "missing schema_hash".to_string())?;
        let feature_sha = manifest
            .feature_schema_sha256
            .as_deref()
            .ok_or_else(|| "missing feature_schema_sha256".to_string())?;
        if feature_schema.static_base_dim != manifest.model_config.static_dim_base {
            return Err("static_base_dim mismatch with feature_schema".to_string());
        }
        if feature_schema.micro_state_dim != micro_state_dim {
            return Err("micro_state_dim mismatch with feature_schema".to_string());
        }
        if feature_schema.static_total_dim != manifest.model_config.static_dim_total {
            return Err("static_total_dim mismatch with feature_schema".to_string());
        }
        if feature_schema.input_ids_len != manifest.model_config.seq_len {
            return Err("input_ids_len mismatch with feature_schema".to_string());
        }
        if let Some(fs_missing) = feature_schema.missing_id {
            if fs_missing != manifest.model_config.missing_id {
                return Err("missing_id mismatch with feature_schema".to_string());
            }
        }
        if feature_hash != feature_sha {
            return Err("feature_schema_sha256 mismatch".to_string());
        }
        if feature_hash != schema_hash {
            return Err("schema_hash mismatch".to_string());
        }
        if let Some(expect_sha) = manifest.micro_state_schema_sha256.as_deref() {
            let schema_path = bundle_dir.join("micro_state_schema.json");
            let actual = file_sha256(&schema_path)?;
            if actual != expect_sha {
                return Err("micro_state_schema_sha256 mismatch".to_string());
            }
        }
    }
    #[cfg(not(feature = "v1_1_experiment"))]
    if micro_state_dim != 0 {
        return Err("micro_state_dim must be 0 in mainline".to_string());
    }
    if let Some(policy) = manifest.model_config.pad_policy.as_ref() {
        match policy {
            PadPolicy::None(name) => {
                if name != "none" {
                    return Err("pad_policy must be none".to_string());
                }
            }
            PadPolicy::Values {
                a_log_pad_value,
                bc_pad_value,
                state_pad_value,
            } => {
                if *a_log_pad_value != -30.0 {
                    return Err("A_log_pad_value must be -30.0".to_string());
                }
                if *bc_pad_value != 0.0 {
                    return Err("BC_pad_value must be 0.0".to_string());
                }
                if *state_pad_value != 0.0 {
                    return Err("state_pad_value must be 0.0".to_string());
                }
            }
        }
    }

    let dims = derive_dims(&manifest)?;
    let vocab = manifest.model_config.vocab_size;
    if dims.emb_vocab != vocab && dims.emb_vocab != vocab + 1 {
        return Err("emb.weight vocab size mismatch".to_string());
    }
    if manifest.model_config.missing_id < 0
        || (manifest.model_config.missing_id as usize) >= vocab
    {
        return Err("missing_id must be within vocab_size".to_string());
    }
    if manifest.model_config.pad_id < 0 || (manifest.model_config.pad_id as usize) >= vocab {
        return Err("pad_id must be within vocab_size".to_string());
    }
    validate_manifest_shapes(&manifest, dims)?;

    let weights_path = bundle_dir.join(&manifest.weights_file);
    let mut blob = Vec::new();
    File::open(&weights_path)
        .map_err(|e| e.to_string())?
        .read_to_end(&mut blob)
        .map_err(|e| e.to_string())?;
    pretouch_read(&blob);
    maybe_mlockall();

    let blob_sha = sha256_hex(&blob);
    if blob_sha != manifest.weights_sha256 {
        return Err("weights_sha256 mismatch".to_string());
    }

    let mut weights = build_weights(&manifest, dims, &blob)?;
    let cfg = weights.cfg.clone();

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
    if matches!(math_backend, MathBackend::FastBf16) {
        weights.precompute_packed_bf16(16);
    }

    let schema_hash = manifest
        .schema_hash
        .clone()
        .unwrap_or_else(|| "none".to_string());

    let golden_npz = if let Some(path) = args.golden.as_ref() {
        let p = PathBuf::from(path);
        if p.is_relative() {
            bundle_dir.join(p)
        } else {
            p
        }
    } else {
        bundle_dir.join("golden_full_min_v1_mainline.npz")
    };
    let data = load_npz(&golden_npz)?;

    let input_ids_arr = data
        .get("input_ids")
        .ok_or_else(|| "missing input_ids".to_string())?;
    let mut input_ids = input_ids_arr.as_i64()?;
    let input_shape = input_ids_arr.shape.clone();
    let (batch_from_shape, seq_len, feat_dim) = match input_shape.as_slice() {
        [b, l, f] => (*b, *l, *f),
        [b, l] => (*b, *l, 1),
        [l] => (1, *l, 1),
        _ => return Err(format!("input_ids shape unsupported {:?}", input_shape)),
    };
    if batch_from_shape != 1 {
        if args.batch != 1 {
            return Err("bench expects batch=1 in golden".to_string());
        }
        let stride = input_ids.len() / batch_from_shape;
        input_ids.truncate(stride);
    }
    if seq_len != cfg.seq_len || feat_dim != cfg.feature_dim {
        return Err("input_ids shape mismatch".to_string());
    }
    let remapped = sanitize_input_ids(&mut input_ids, cfg.vocab_size, manifest.model_config.missing_id);
    if remapped > 0 {
        println!("note: remapped {} input_ids to missing_id", remapped);
    }

    let mut static_arr = if let Some(arr) = data.get("static_total") {
        arr.as_f32()?
    } else if let Some(arr) = data.get("static") {
        arr.as_f32()?
    } else {
        #[cfg(feature = "v1_1_experiment")]
        {
            if let (Some(base), Some(micro)) = (data.get("static_base"), data.get("micro_state")) {
                let base = base.as_f32()?;
                let micro = micro.as_f32()?;
                let mut out = Vec::with_capacity(base.len() + micro.len());
                out.extend_from_slice(&base);
                out.extend_from_slice(&micro);
                out
            } else {
                return Err("missing static_total/static or static_base+micro_state".to_string());
            }
        }
        #[cfg(not(feature = "v1_1_experiment"))]
        {
            return Err("mainline requires static_total/static input".to_string());
        }
    };
    if static_arr.len() == cfg.static_dim_total * batch_from_shape && batch_from_shape != 1 {
        static_arr.truncate(cfg.static_dim_total);
    }
    if static_arr.len() != cfg.static_dim_total {
        return Err("static_total dim mismatch".to_string());
    }
    if static_arr.len() != cfg.static_dim_total {
        return Err("static_total length mismatch".to_string());
    }

    let mut dispatch_name = args.dispatch.clone();
    let dispatch_auto = if dispatch_name == "auto" {
        if let Ok(env) = std::env::var("RISK_MAMBA_DISPATCH") {
            dispatch_name = env;
            None
        } else {
            let mut scratch = AlignedBuf::new(scratch_bytes_full(&cfg), 64)?;
            let mut logits = vec![0.0f32; cfg.n_class];
            let mut margin = vec![0.0f32; 1];
            Some(microbench_dispatch(
                &cfg,
                &weights,
                math_backend,
                &input_ids,
                &static_arr,
                scratch.as_mut_slice(),
                &mut logits,
                &mut margin,
                args.auto_dispatch_iters.max(1),
                args.auto_dispatch_warmup,
            ))
        }
    } else {
        None
    };
    if let Some(decision) = dispatch_auto.as_ref() {
        dispatch_name = decision.selected.clone();
    }
    let dispatch = resolve_dispatch(&dispatch_name)?;
    println!(
        "bundle schema_hash={} static_dim_total={} seq_len={} vocab_size_effective={} embedding_rows={} dispatch={} math={}",
        schema_hash,
        cfg.static_dim_total,
        cfg.seq_len,
        cfg.vocab_size,
        dims.emb_vocab,
        dispatch_label(dispatch),
        math_name
    );

    let lengths = data.get("lengths").map(|arr| arr.as_i64()).transpose()?;
    let lengths_ref = lengths.as_deref();

    let scratch_len = scratch_bytes_full(&cfg);
    let mut scratch = AlignedBuf::new(scratch_len, 64)?;
    pretouch_write(scratch.as_mut_slice());
    let mut logits = vec![0.0f32; cfg.n_class];
    let mut margin = vec![0.0f32; 1];
    let mut stage_stats = StageStats::default();
    let mut perf_counters = PerfCounters::default();

    // warmup
    for _ in 0..args.warmup {
        full_stateless_forward(
            &cfg,
            &weights,
            dispatch,
            math_backend,
            &input_ids,
            &static_arr,
            lengths_ref,
            scratch.as_mut_slice(),
            &mut logits,
            &mut margin,
            None,
            Some(&mut stage_stats),
            Some(&mut perf_counters),
            None,
        )
        .map_err(|e| e.to_string())?;
    }

    stage_stats.clear();

    let mut runs_ms = Vec::with_capacity(args.repeat);
    for _ in 0..args.repeat {
        let start = Instant::now();
        for _ in 0..args.iters {
            full_stateless_forward(
                &cfg,
                &weights,
                dispatch,
                math_backend,
                &input_ids,
                &static_arr,
                lengths_ref,
                scratch.as_mut_slice(),
                &mut logits,
            &mut margin,
            None,
            Some(&mut stage_stats),
            Some(&mut perf_counters),
            None,
        )
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

    let bundle_name = bundle_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("bundle");

    let (weights_pack_once, weights_pack_repeated) = weights_pack_stats();
    let mut linear_shapes = HashMap::new();
    let seq_len = cfg.seq_len;
    linear_shapes.insert(
        "in_proj".to_string(),
        LinearShape {
            m: seq_len,
            k: cfg.d_model,
            n: 2 * cfg.d_inner,
        },
    );
    linear_shapes.insert(
        "x_proj".to_string(),
        LinearShape {
            m: seq_len,
            k: cfg.d_inner,
            n: cfg.dt_rank + 2 * cfg.d_state_pad,
        },
    );
    linear_shapes.insert(
        "dt_proj".to_string(),
        LinearShape {
            m: seq_len,
            k: cfg.dt_rank,
            n: cfg.d_inner,
        },
    );
    linear_shapes.insert(
        "out_proj".to_string(),
        LinearShape {
            m: seq_len,
            k: cfg.d_inner,
            n: cfg.d_model,
        },
    );
    linear_shapes.insert(
        "mlp_fc1".to_string(),
        LinearShape {
            m: seq_len,
            k: cfg.d_model,
            n: cfg.d_mlp,
        },
    );
    linear_shapes.insert(
        "mlp_fc2".to_string(),
        LinearShape {
            m: seq_len,
            k: cfg.d_mlp,
            n: cfg.d_model,
        },
    );
    linear_shapes.insert(
        "head".to_string(),
        LinearShape {
            m: 1,
            k: cfg.d_model,
            n: cfg.n_class,
        },
    );
    let mlp_fc1_impl = {
        let bits = perf_counters.mlp_fc1_impl_bits;
        if bits == 0 {
            "unknown".to_string()
        } else {
            let mut parts = Vec::new();
            if bits & 1 != 0 {
                parts.push("packed16");
            }
            if bits & 2 != 0 {
                parts.push("unpacked");
            }
            if bits & 4 != 0 {
                parts.push("fused");
            }
            parts.join("|")
        }
    };
    let kernel_path = kernel_path_tag(&cfg).to_string();
    let layout = layout_tag(&cfg, math_backend, false, dispatch).to_string();
    let mlp_impl = if mlp_fused_enabled(cfg.seq_len, false, math_backend) {
        "fused_streamed".to_string()
    } else if perf_counters.mlp_fc1_impl_bits & 1 != 0 {
        "unfused_packed".to_string()
    } else if perf_counters.mlp_fc1_impl_bits & 2 != 0 {
        "unfused_unpacked".to_string()
    } else {
        "unfused".to_string()
    };
    let output = BenchOutput {
        bundle: args.bundle_dir.clone(),
        forward_kind: manifest.model_config.forward_kind.clone(),
        schema_hash: manifest
            .schema_hash
            .clone()
            .unwrap_or_else(|| "none".to_string()),
        seq_len: cfg.seq_len,
        static_dim_total: cfg.static_dim_total,
        micro_state_dim: cfg.micro_state_dim,
        dispatch: dispatch_label(dispatch).to_string(),
        math_backend: math_name,
        kernel_path,
        layout,
        mlp_impl,
        preset: args.preset.clone(),
        state: args.state.clone(),
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
        cv: between_repeat_cv,
        outlier_indices,
        stage_total,
        stage_per_iter,
        mlp_fc1_call_count: perf_counters.mlp_fc1_calls,
        mlp_fc1_layer_count: perf_counters.mlp_fc1_layers,
        mlp_fc1_shape: LinearShape {
            m: perf_counters.mlp_fc1_m,
            k: perf_counters.mlp_fc1_k,
            n: perf_counters.mlp_fc1_n,
        },
        mlp_fc1_impl,
        dispatch_auto,
        weights_pack_once,
        weights_pack_repeated,
        linear_shapes,
    };

    create_dir_all(&args.out_dir).map_err(|e| e.to_string())?;
    let out_path = PathBuf::from(&args.out_dir).join(format!(
        "rust_perf_full_{}_{}_{}_{}_{}_b{}.json",
        bundle_name,
        output.dispatch,
        output.math_backend,
        output.state,
        output.preset,
        output.batch
    ));
    let mut f = File::create(&out_path).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(&output).map_err(|e| e.to_string())?;
    f.write_all(json.as_bytes()).map_err(|e| e.to_string())?;

    if args.audit_kernel {
        let audit_dir = PathBuf::from("nightly_takeoff_report_v6");
        create_dir_all(&audit_dir).map_err(|e| e.to_string())?;
        let audit_path = audit_dir.join("avx2_audit_trace.tsv");
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
