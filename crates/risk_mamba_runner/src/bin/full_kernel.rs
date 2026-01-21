use clap::Parser;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand::seq::index;
use risk_mamba_kernel::{
    full_stateless_forward, scratch_bytes_full, CpuDispatch, DbgTaps, EmbSumMode, ForwardKind,
    GeluKind, HeadInputKind, KernelError, LayerWeights, LogitsKind, MathBackend, ModelConfig,
    PriorSource, SoftplusKind, StateSurgery, WeightsView,
};
use serde::{Deserialize, Serialize};
use regex::Regex;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
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
#[command(about = "Full stateless kernel correctness gate (min/dbg)")]
struct Args {
    #[arg(long, alias = "bundle", default_value = "mamba/static_bundle_full_v1_mainline")]
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
    #[arg(long)]
    audit_kernel: bool,
    #[arg(long)]
    no_fallback: bool,

    /// Fuzz mode (random inputs), skips golden checks
    #[arg(long)]
    fuzz: bool,

    /// Fuzz iterations
    #[arg(long, default_value_t = 1000)]
    fuzz_iters: usize,

    /// Fuzz RNG seed
    #[arg(long, default_value_t = 0)]
    fuzz_seed: u64,

    /// Optional fuzz report path (JSON)
    #[arg(long)]
    fuzz_report: Option<String>,

    /// Inject NaN/Inf into static_total during fuzz
    #[arg(long, default_value_t = false)]
    fuzz_inject_nan: bool,

    /// Replay JSONL path (input_ids/static_total + expected margin/logits)
    #[arg(long)]
    replay: Option<String>,

    /// Optional replay report path (JSON)
    #[arg(long)]
    replay_report: Option<String>,

    /// Gate policy file for dbg per-key tolerances
    #[arg(long, default_value = "gate_policy_v1_mainline.json")]
    gate_policy: String,
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

#[derive(Debug, Clone)]
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

#[derive(Debug, Deserialize)]
struct ReplaySample {
    input_ids: Vec<i64>,
    static_total: Vec<f32>,
    margin: Option<f32>,
    logits: Option<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct FuzzReport {
    iters: usize,
    errors: usize,
    nan_outputs: usize,
    inf_outputs: usize,
    out_of_range_ids: usize,
}

#[derive(Debug, Serialize)]
struct ReplayReport {
    samples: usize,
    mismatches: usize,
    max_abs: f32,
    max_rel: f32,
}

#[derive(Clone, Copy)]
struct DiffOptions {
    atol: f32,
    rtol: f32,
    fail_fast: bool,
    dump_first: bool,
    topk: usize,
}

#[derive(Deserialize)]
struct GatePolicyFile {
    policy_version: String,
    tiers: HashMap<String, GatePolicyTier>,
    patterns: Vec<GatePolicyPattern>,
}

#[derive(Deserialize)]
struct GatePolicyTier {
    atol: f32,
    rtol: f32,
}

#[derive(Deserialize)]
struct GatePolicyPattern {
    tier: String,
    regex: String,
}

#[derive(Clone)]
struct GatePolicyRuntime {
    policy_version: String,
    policy_hash: String,
    tiers: HashMap<String, DiffOptions>,
    patterns: Vec<(Regex, String)>,
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

fn resolve_thr(cfg: &ManifestConfig) -> Result<f32, String> {
    if let Some(v) = cfg.thr_fixedfpr_gate {
        return Ok(v);
    }
    if let Some(v) = cfg.thr01_fixed_fpr {
        return Ok(v);
    }
    if let Some(v) = cfg.thr_anchor {
        return Ok(v);
    }
    Err("missing threshold (thr_fixedfpr_gate/thr01_fixed_fpr/thr_anchor)".to_string())
}

fn pad_policy_values(policy: Option<&PadPolicy>) -> Option<(f32, f32, f32)> {
    match policy {
        None => None,
        Some(PadPolicy::None(_)) => None,
        Some(PadPolicy::Values {
            a_log_pad_value,
            bc_pad_value,
            state_pad_value,
        }) => Some((*a_log_pad_value, *bc_pad_value, *state_pad_value)),
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

fn load_gate_policy(path: &Path, base: DiffOptions) -> Result<GatePolicyRuntime, String> {
    let contents = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let policy: GatePolicyFile = serde_json::from_str(&contents).map_err(|e| e.to_string())?;
    let policy_hash = file_sha256(path)?;
    let mut tiers = HashMap::new();
    for (name, spec) in policy.tiers {
        tiers.insert(
            name,
            DiffOptions {
                atol: spec.atol,
                rtol: spec.rtol,
                fail_fast: base.fail_fast,
                dump_first: base.dump_first,
                topk: base.topk,
            },
        );
    }
    let mut patterns = Vec::new();
    for pat in policy.patterns {
        let re = Regex::new(&pat.regex).map_err(|e| e.to_string())?;
        patterns.push((re, pat.tier));
    }
    Ok(GatePolicyRuntime {
        policy_version: policy.policy_version,
        policy_hash,
        tiers,
        patterns,
    })
}

fn diff_opts_for_key(
    key: &str,
    policy: &GatePolicyRuntime,
    default_tier: &str,
    base: DiffOptions,
) -> (DiffOptions, String) {
    let mut tier = default_tier;
    for (re, name) in &policy.patterns {
        if re.is_match(key) {
            tier = name;
            break;
        }
    }
    let opts = policy
        .tiers
        .get(tier)
        .copied()
        .unwrap_or(base);
    (opts, tier.to_string())
}

fn run_fuzz(
    args: &Args,
    cfg: &ModelConfig,
    weights: &WeightsView,
) -> Result<(), String> {
    let mut rng = StdRng::seed_from_u64(args.fuzz_seed);
    let mut scratch = AlignedBuf::new(scratch_bytes_full(cfg), 64)?;
    pretouch_write(scratch.as_mut_slice());
    let mut errors = 0usize;
    let mut nan_outputs = 0usize;
    let mut inf_outputs = 0usize;
    let mut out_of_range_ids = 0usize;
    for _ in 0..args.fuzz_iters {
        let mut input_ids = vec![0i64; cfg.seq_len * cfg.feature_dim];
        for v in &mut input_ids {
            let id = rng.gen_range(-2i64..(cfg.vocab_size as i64 + 2));
            if id < 0 || id >= cfg.vocab_size as i64 {
                out_of_range_ids += 1;
            }
            *v = id;
        }
        let mut static_total = vec![0.0f32; cfg.static_dim_total];
        for v in &mut static_total {
            *v = rng.gen_range(-5.0f32..5.0f32);
        }
        if args.fuzz_inject_nan && rng.gen_bool(0.001) {
            static_total[0] = f32::NAN;
        }
        let mut logits = vec![0.0f32; cfg.n_class];
        let mut margin = vec![0.0f32; 1];
        let res = full_stateless_forward(
            cfg,
            weights,
            CpuDispatch::Scalar,
            MathBackend::Exact,
            &input_ids,
            &static_total,
            None,
            scratch.as_mut_slice(),
            &mut logits,
            &mut margin,
            None,
            None,
            None,
            None,
        );
        if res.is_err() {
            errors += 1;
            continue;
        }
        for v in logits.iter().chain(margin.iter()) {
            if v.is_nan() {
                nan_outputs += 1;
            } else if !v.is_finite() {
                inf_outputs += 1;
            }
        }
    }
    let report = FuzzReport {
        iters: args.fuzz_iters,
        errors,
        nan_outputs,
        inf_outputs,
        out_of_range_ids,
    };
    println!(
        "fuzz iters={} errors={} nan_outputs={} inf_outputs={} out_of_range_ids={}",
        report.iters, report.errors, report.nan_outputs, report.inf_outputs, report.out_of_range_ids
    );
    if let Some(path) = args.fuzz_report.as_ref() {
        let data = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        std::fs::write(path, data).map_err(|e| e.to_string())?;
    }
    if report.errors > 0 || report.nan_outputs > 0 || report.inf_outputs > 0 {
        return Err("fuzz detected errors or non-finite outputs".to_string());
    }
    Ok(())
}

fn run_replay(
    args: &Args,
    cfg: &ModelConfig,
    weights: &WeightsView,
    path: &Path,
) -> Result<(), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let mut scratch = AlignedBuf::new(scratch_bytes_full(cfg), 64)?;
    pretouch_write(scratch.as_mut_slice());
    let mut samples = 0usize;
    let mut mismatches = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let sample: ReplaySample = serde_json::from_str(&line).map_err(|e| e.to_string())?;
        if sample.input_ids.len() != cfg.seq_len * cfg.feature_dim {
            return Err("replay input_ids length mismatch".to_string());
        }
        if sample.static_total.len() != cfg.static_dim_total {
            return Err("replay static_total length mismatch".to_string());
        }
        let mut logits = vec![0.0f32; cfg.n_class];
        let mut margin = vec![0.0f32; 1];
        full_stateless_forward(
            cfg,
            weights,
            CpuDispatch::Scalar,
            MathBackend::Exact,
            &sample.input_ids,
            &sample.static_total,
            None,
            scratch.as_mut_slice(),
            &mut logits,
            &mut margin,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| format!("replay forward error: {:?}", e))?;

        if let Some(exp) = sample.margin {
            let diff = (margin[0] - exp).abs();
            let rel = if exp == 0.0 { diff } else { diff / exp.abs() };
            let tol = args.atol + args.rtol * exp.abs();
            if diff > tol {
                mismatches += 1;
            }
            if diff > max_abs {
                max_abs = diff;
            }
            if rel > max_rel {
                max_rel = rel;
            }
        }
        if let Some(exp_logits) = sample.logits.as_ref() {
            if exp_logits.len() != cfg.n_class {
                return Err("replay logits length mismatch".to_string());
            }
            for (a, b) in logits.iter().zip(exp_logits.iter()) {
                let diff = (*a - *b).abs();
                let rel = if *b == 0.0 { diff } else { diff / b.abs() };
                let tol = args.atol + args.rtol * b.abs();
                if diff > tol {
                    mismatches += 1;
                }
                if diff > max_abs {
                    max_abs = diff;
                }
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
        samples += 1;
    }
    let report = ReplayReport {
        samples,
        mismatches,
        max_abs,
        max_rel,
    };
    println!(
        "replay samples={} mismatches={} max_abs={} max_rel={}",
        report.samples, report.mismatches, report.max_abs, report.max_rel
    );
    if let Some(path) = args.replay_report.as_ref() {
        let data = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        std::fs::write(path, data).map_err(|e| e.to_string())?;
    }
    if report.mismatches > 0 {
        return Err("replay mismatches detected".to_string());
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

fn slice_sample_i64(buf: &[i64], shape: &[usize], sample: usize) -> Result<Vec<i64>, String> {
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

struct RankingExpect {
    k: usize,
    jaccard: f64,
    swaps: usize,
}

fn load_ranking_report(path: &Path) -> Result<RankingExpect, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    for (idx, line) in text.lines().enumerate() {
        if idx == 0 {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 7 {
            continue;
        }
        let subset = cols[1];
        if subset != "near_thr" {
            continue;
        }
        let k = cols[3].parse::<usize>().map_err(|_| "invalid k".to_string())?;
        let jaccard = cols[4]
            .parse::<f64>()
            .map_err(|_| "invalid jaccard".to_string())?;
        let swaps = cols[5]
            .parse::<usize>()
            .map_err(|_| "invalid swaps".to_string())?;
        return Ok(RankingExpect { k, jaccard, swaps });
    }
    Err("missing near_thr in ranking report".to_string())
}

fn topk_indices(values: &[f32], indices: &[usize], k: usize) -> Vec<usize> {
    let mut idxs = indices.to_vec();
    idxs.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap());
    idxs.truncate(k);
    idxs
}

fn jaccard(a: &[usize], b: &[usize]) -> f64 {
    let mut a_set = a.to_vec();
    let mut b_set = b.to_vec();
    a_set.sort_unstable();
    b_set.sort_unstable();
    let mut i = 0usize;
    let mut j = 0usize;
    let mut inter = 0usize;
    while i < a_set.len() && j < b_set.len() {
        if a_set[i] == b_set[j] {
            inter += 1;
            i += 1;
            j += 1;
        } else if a_set[i] < b_set[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    if a.is_empty() {
        1.0
    } else {
        inter as f64 / a.len() as f64
    }
}

fn inversion_count(order: &[usize], reference_rank: &HashMap<usize, usize>) -> usize {
    let mut count = 0usize;
    for i in 0..order.len() {
        let ri = reference_rank[&order[i]];
        for j in (i + 1)..order.len() {
            let rj = reference_rank[&order[j]];
            if ri > rj {
                count += 1;
            }
        }
    }
    count
}

struct GoldenDbg {
    arrays: HashMap<String, NpyArray>,
    diff_opts: DiffOptions,
    stop_after: Option<String>,
    sample: Option<usize>,
    sample_indices: Option<Vec<usize>>,
    orig_batch: usize,
    policy: Option<GatePolicyRuntime>,
    default_tier: String,
}

impl GoldenDbg {
    fn new(
        arrays: HashMap<String, NpyArray>,
        diff_opts: DiffOptions,
        stop_after: Option<String>,
        sample: Option<usize>,
        sample_indices: Option<Vec<usize>>,
        orig_batch: usize,
        policy: Option<GatePolicyRuntime>,
        default_tier: String,
    ) -> Self {
        Self {
            arrays,
            diff_opts,
            stop_after,
            sample,
            sample_indices,
            orig_batch,
            policy,
            default_tier,
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
        let (opts, tier) = if let Some(policy) = self.policy.as_ref() {
            diff_opts_for_key(key, policy, &self.default_tier, self.diff_opts)
        } else {
            (self.diff_opts, "default".to_string())
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
                    compare_f32(&sample_key, actual_sample, expected_sample, &sample_shape, opts)
                        .map_err(|e| KernelError::DebugTapError(format!("tier={} {}", tier, e)))?;
                }
            } else {
                compare_f32(key, data, &expected, shape, opts)
                    .map_err(|e| KernelError::DebugTapError(format!("tier={} {}", tier, e)))?;
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
                compare_f32(key, data, expected_sample, shape, opts)
                    .map_err(|e| KernelError::DebugTapError(format!("tier={} {}", tier, e)))?;
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
    let args = Args::parse();
    if args.audit_kernel {
        std::env::set_var("RISK_MAMBA_AUDIT_KERNEL", "1");
    }
    if args.no_fallback {
        std::env::set_var("RISK_MAMBA_NO_FALLBACK", "1");
    }
    let bundle_dir = PathBuf::from(args.bundle_dir.as_str());
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

    let weights = build_weights(&manifest, dims, &blob)?;
    let cfg = weights.cfg.clone();

    let schema_hash = manifest
        .schema_hash
        .clone()
        .unwrap_or_else(|| "none".to_string());
    println!(
        "bundle schema_hash={} static_dim_total={} seq_len={} vocab_size_effective={} embedding_rows={} dispatch=scalar_exact",
        schema_hash, cfg.static_dim_total, cfg.seq_len, cfg.vocab_size, dims.emb_vocab
    );

    if args.fuzz {
        return run_fuzz(&args, &cfg, &weights);
    }
    if let Some(path) = args.replay.as_ref() {
        let replay_path = PathBuf::from(path);
        return run_replay(&args, &cfg, &weights, &replay_path);
    }

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

    let gate_policy = if mode_dbg {
        let path = PathBuf::from(&args.gate_policy);
        match load_gate_policy(&path, diff_opts) {
            Ok(policy) => {
                println!(
                    "gate_policy version={} hash={}",
                    policy.policy_version, policy.policy_hash
                );
                println!("gate_policy patterns:");
                for (re, tier) in &policy.patterns {
                    println!("  {} -> {}", re.as_str(), tier);
                }
                Some(policy)
            }
            Err(e) => {
                println!("warning: failed to load gate policy ({}), using base tolerances", e);
                None
            }
        }
    } else {
        None
    };

    let min_npz = if let Some(path) = args.golden.as_ref() {
        let p = PathBuf::from(path);
        if p.is_relative() {
            bundle_dir.join(p)
        } else {
            p
        }
    } else {
        bundle_dir.join("golden_full_min_v1_mainline.npz")
    };
    let dbg_npz = bundle_dir.join("golden_full_dbg_v1_mainline.npz");

    let min = load_npz(&min_npz)?;
    let dbg = if mode_dbg { Some(load_npz(&dbg_npz)?) } else { None };

    let data_npz = if mode_dbg {
        dbg.as_ref().ok_or_else(|| "missing dbg npz".to_string())?
    } else {
        &min
    };

    let input_ids_arr = data_npz
        .get("input_ids")
        .ok_or_else(|| "missing input_ids".to_string())?;
    let input_ids_full = input_ids_arr.as_i64()?;
    let input_shape = input_ids_arr.shape.clone();
    let (orig_batch, seq_len, feat_dim) = match input_shape.as_slice() {
        [b, l, f] => (*b, *l, *f),
        [b, l] => (*b, *l, 1),
        [l] => (1, *l, 1),
        _ => return Err(format!("input_ids shape unsupported {:?}", input_shape)),
    };
    if seq_len != cfg.seq_len {
        return Err(format!("input_ids seq_len mismatch {}", seq_len));
    }
    if feat_dim != cfg.feature_dim {
        return Err(format!("input_ids feature_dim mismatch {}", feat_dim));
    }

    let static_total_arr = data_npz.get("static_total").or_else(|| data_npz.get("static"));
    let static_base_arr = data_npz.get("static_base");
    let micro_state_arr = data_npz.get("micro_state");

    #[cfg(not(feature = "v1_1_experiment"))]
    if static_total_arr.is_none() {
        return Err("mainline requires static_total/static input".to_string());
    }
    #[cfg(feature = "v1_1_experiment")]
    if static_total_arr.is_none() && (static_base_arr.is_none() || micro_state_arr.is_none()) {
        return Err("missing static_total/static or static_base+micro_state".to_string());
    }

    let static_total_full = static_total_arr.map(|arr| arr.as_f32()).transpose()?;
    let static_total_shape = static_total_arr.map(|arr| arr.shape.clone());
    let static_base_full = static_base_arr.map(|arr| arr.as_f32()).transpose()?;
    let static_base_shape = static_base_arr.map(|arr| arr.shape.clone());
    let micro_state_full = micro_state_arr.map(|arr| arr.as_f32()).transpose()?;
    let micro_state_shape = micro_state_arr.map(|arr| arr.shape.clone());

    if let Some(shape) = static_total_shape.as_ref() {
        if shape.len() == 2 && shape[0] != orig_batch {
            return Err(format!("static_total batch mismatch {:?}", shape));
        }
        if shape.len() == 2 && shape[1] != cfg.static_dim_total {
            return Err(format!("static_total dim mismatch {:?}", shape));
        }
        if shape.len() == 1 && shape[0] != cfg.static_dim_total {
            return Err(format!("static_total dim mismatch {:?}", shape));
        }
    }
    if let Some(shape) = static_base_shape.as_ref() {
        if shape.len() == 2 && shape[0] != orig_batch {
            return Err(format!("static_base batch mismatch {:?}", shape));
        }
        if shape.len() == 2 && shape[1] != cfg.static_dim_total - cfg.micro_state_dim {
            return Err(format!("static_base dim mismatch {:?}", shape));
        }
        if shape.len() == 1 && shape[0] != cfg.static_dim_total - cfg.micro_state_dim {
            return Err(format!("static_base dim mismatch {:?}", shape));
        }
    }
    if let Some(shape) = micro_state_shape.as_ref() {
        if shape.len() == 2 && shape[0] != orig_batch {
            return Err(format!("micro_state batch mismatch {:?}", shape));
        }
        if shape.len() == 2 && shape[1] != cfg.micro_state_dim {
            return Err(format!("micro_state dim mismatch {:?}", shape));
        }
        if shape.len() == 1 && shape[0] != cfg.micro_state_dim {
            return Err(format!("micro_state dim mismatch {:?}", shape));
        }
    }

    let logits_exp_full = data_npz.get("logits").map(|arr| arr.as_f32()).transpose()?;
    let margin_key = if data_npz.contains_key("margin") {
        Some("margin")
    } else if data_npz.contains_key("delta_margin") {
        Some("delta_margin")
    } else {
        None
    };
    let margin_exp_full = margin_key
        .and_then(|key| data_npz.get(key))
        .map(|arr| arr.as_f32())
        .transpose()?;

    if logits_exp_full.is_none() && margin_exp_full.is_none() {
        return Err("missing logits or delta_margin/margin".to_string());
    }
    if margin_key == Some("margin") {
        if let (Some(logits), Some(margin)) = (logits_exp_full.as_ref(), margin_exp_full.as_ref()) {
            if !margin_matches_logits(logits, margin, orig_batch, cfg.n_class) {
                return Err("margin does not match logits[:,1]".to_string());
            }
        }
    }

    if let Some((a_log_pad, _bc_pad, _state_pad)) =
        pad_policy_values(manifest.model_config.pad_policy.as_ref())
    {
        check_a_log_pad(&weights, a_log_pad)?;
    }

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

    let sample_indices = if let Some(s) = args.sample {
        if s >= orig_batch {
            return Err(format!("sample {} out of range {}", s, orig_batch));
        }
        vec![s]
    } else if let Some(indices) = dbg_sample_indices.clone() {
        indices
    } else {
        (0..orig_batch).collect()
    };

    let scratch_len = scratch_bytes_full(&cfg);
    let mut scratch = AlignedBuf::new(scratch_len, 64)?;
    pretouch_write(scratch.as_mut_slice());
    let full_run = sample_indices.len() == orig_batch;
    let mut pred_margin_full = if full_run {
        vec![0.0f32; orig_batch]
    } else {
        Vec::new()
    };

    for sample in sample_indices {
        let mut input_ids = if orig_batch > 1 {
            slice_sample_i64(&input_ids_full, &input_shape, sample)?
        } else {
            input_ids_full.clone()
        };
        let remapped = sanitize_input_ids(
            &mut input_ids,
            cfg.vocab_size,
            manifest.model_config.missing_id,
        );
        if remapped > 0 {
            println!("note: remapped {} input_ids to missing_id", remapped);
        }

        let static_total = if let Some(static_total) = static_total_full.as_ref() {
            let has_batch = static_total_shape
                .as_ref()
                .map(|s| s.len() > 1)
                .unwrap_or(false);
            if orig_batch > 1 && has_batch {
                slice_sample_f32(static_total, static_total_shape.as_ref().unwrap(), sample)?
            } else {
                static_total.clone()
            }
        } else {
            let base = static_base_full
                .as_ref()
                .ok_or_else(|| "missing static_base".to_string())?;
            let micro = micro_state_full
                .as_ref()
                .ok_or_else(|| "missing micro_state".to_string())?;
            let base_has_batch = static_base_shape
                .as_ref()
                .map(|s| s.len() > 1)
                .unwrap_or(false);
            let base_slice = if orig_batch > 1 && base_has_batch {
                slice_sample_f32(base, static_base_shape.as_ref().unwrap(), sample)?
            } else {
                base.clone()
            };
            let micro_has_batch = micro_state_shape
                .as_ref()
                .map(|s| s.len() > 1)
                .unwrap_or(false);
            let micro_slice = if orig_batch > 1 && micro_has_batch {
                slice_sample_f32(micro, micro_state_shape.as_ref().unwrap(), sample)?
            } else {
                micro.clone()
            };
            let mut out = Vec::with_capacity(base_slice.len() + micro_slice.len());
            out.extend_from_slice(&base_slice);
            out.extend_from_slice(&micro_slice);
            out
        };
        if static_total.len() != cfg.static_dim_total {
            return Err(format!(
                "static_total length mismatch {} != {}",
                static_total.len(),
                cfg.static_dim_total
            ));
        }

        let lengths_full = data_npz
            .get("lengths")
            .map(|arr| arr.as_i64())
            .transpose()?;
        let lengths_sample = if let Some(l) = lengths_full.as_ref() {
            if l.len() > 1 {
                slice_sample_i64(l, &[orig_batch], sample)?
            } else {
                l.clone()
            }
        } else {
            Vec::new()
        };
        let lengths_ref = if lengths_sample.is_empty() {
            None
        } else {
            Some(lengths_sample.as_slice())
        };

        let mut logits_out = vec![0.0f32; cfg.n_class];
        let mut margin_out = vec![0.0f32; 1];

        let mut dbg_sink = if mode_dbg {
            Some(GoldenDbg::new(
                dbg.clone().unwrap(),
                diff_opts,
                args.stop_after.clone(),
                Some(sample),
                None,
                orig_batch,
                gate_policy.clone(),
                "dbg-critical".to_string(),
            ))
        } else {
            None
        };
        let dbg_ref: Option<&mut dyn DbgTaps> = dbg_sink.as_mut().map(|s| s as &mut dyn DbgTaps);

        let run = full_stateless_forward(
            &cfg,
            &weights,
            CpuDispatch::Scalar,
            MathBackend::Exact,
            &input_ids,
            &static_total,
            lengths_ref,
            scratch.as_mut_slice(),
            &mut logits_out,
            &mut margin_out,
            dbg_ref,
            None,
            None,
            None,
        );
        if let Err(e) = run {
            let msg = e.to_string();
            if msg.starts_with("debug tap error: stop-after ") {
                println!("OK: {}", msg);
                return Ok(());
            }
            return Err(msg);
        }

        if let Some(logits) = logits_exp_full.as_ref() {
            let logits_exp = if orig_batch > 1 {
                slice_sample_f32(logits, &[orig_batch, cfg.n_class], sample)?
            } else {
                logits.clone()
            };
            compare_f32("logits", &logits_out, &logits_exp, &[cfg.n_class], diff_opts)?;
        }
        if let Some(margin) = margin_exp_full.as_ref() {
            let margin_exp = if orig_batch > 1 {
                slice_sample_f32(margin, &[orig_batch], sample)?
            } else {
                margin.clone()
            };
            compare_f32("delta_margin", &margin_out, &margin_exp, &[1], diff_opts)?;
        }

        if full_run {
            pred_margin_full[sample] = margin_out[0];
        }
    }

    if full_run {
        let expect = load_ranking_report(&bundle_dir.join("golden_ranking_report.tsv"))?;
        let margin_exp = margin_exp_full
            .as_ref()
            .ok_or_else(|| "missing delta_margin for near_thr".to_string())?;
        if margin_exp.len() != orig_batch {
            return Err("delta_margin batch mismatch for near_thr".to_string());
        }

        let static_base_dim = manifest.model_config.static_dim_base;
        let teacher_col = manifest.model_config.teacher_static_col;
        if teacher_col >= static_base_dim {
            return Err("teacher_static_col out of range".to_string());
        }
        let near_band = manifest.model_config.near_band;
        let thr = resolve_thr(&manifest.model_config)?;

        let mut near_idx = Vec::new();
        for b in 0..orig_batch {
            let l2_margin = if let Some(static_total) = static_total_full.as_ref() {
                let has_batch = static_total_shape
                    .as_ref()
                    .map(|s| s.len() > 1)
                    .unwrap_or(false);
                let base = if has_batch {
                    b * cfg.static_dim_total
                } else {
                    0
                };
                static_total[base + teacher_col]
            } else if let Some(static_base) = static_base_full.as_ref() {
                let has_batch = static_base_shape
                    .as_ref()
                    .map(|s| s.len() > 1)
                    .unwrap_or(false);
                let base = if has_batch { b * static_base_dim } else { 0 };
                static_base[base + teacher_col]
            } else {
                return Err("missing static_total/static_base for near_thr".to_string());
            };
            if (l2_margin - thr).abs() <= near_band {
                near_idx.push(b);
            }
        }

        if !near_idx.is_empty() {
            let k = expect.k.min(near_idx.len());
            let top_pred = topk_indices(&pred_margin_full, &near_idx, k);
            let top_exp = topk_indices(margin_exp, &near_idx, k);
            let jacc = jaccard(&top_pred, &top_exp);
            if (jacc - expect.jaccard).abs() > 1e-6 {
                return Err(format!(
                    "near_thr jaccard mismatch {} != {}",
                    jacc, expect.jaccard
                ));
            }

            let mut ref_rank = HashMap::new();
            for (rank, idx) in top_exp.iter().enumerate() {
                ref_rank.insert(*idx, rank);
            }
            let swaps = inversion_count(&top_pred, &ref_rank);
            if swaps != expect.swaps {
                return Err(format!(
                    "near_thr swaps mismatch {} != {}",
                    swaps, expect.swaps
                ));
            }
        }
    }

    println!("OK: golden_{}", args.mode);
    Ok(())
}
