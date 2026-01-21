use risk_mamba_kernel::{
    full_stateless_forward, kernel_path_tag, layout_tag, mlp_fused_enabled, scratch_bytes_full,
    CpuDispatch, EmbSumMode, ForwardKind, GeluKind, HeadInputKind, LayerWeights, LogitsKind,
    MathBackend, ModelConfig, PriorSource, SoftplusKind, StateSurgery, WeightsView,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const ALIGN_BYTES: usize = 64;
const PAGE_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HugepagePolicy {
    Never,
    Madvise,
}

#[derive(Clone, Copy, Debug)]
pub struct MemoryPolicy {
    pub pretouch: bool,
    pub hugepage: HugepagePolicy,
    pub willneed: bool,
}

impl Default for MemoryPolicy {
    fn default() -> Self {
        Self {
            pretouch: false,
            hugepage: HugepagePolicy::Never,
            willneed: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MambaOutcome {
    pub logits: [f32; 2],
    pub margin: f32,
    pub dispatch: String,
    pub math: String,
    pub fallback_reason: Option<String>,
}

#[derive(Clone, Copy, Debug)]
pub enum FallbackMode {
    Exact,
    DefaultZero,
}

#[derive(Clone, Debug)]
pub struct CircuitBreakerConfig {
    pub max_errors: u64,
    pub error_rate: f64,
    pub window_ms: u64,
    pub trip_ms: u64,
}

pub struct MambaRuntime {
    cfg: ModelConfig,
    weights: WeightsView<'static>,
    dispatch: CpuDispatch,
    math_backend: MathBackend,
    dispatch_label: String,
    math_label: String,
    schema_hash: String,
    scratch_pool: Mutex<Vec<AlignedBuf>>,
    cb: CircuitBreaker,
    fallback: FallbackMode,
}

// Safety: MambaRuntime owns immutable weights (post-load) and uses a Mutex-protected
// scratch pool for mutable buffers. It is safe to share across threads.
unsafe impl Send for MambaRuntime {}
unsafe impl Sync for MambaRuntime {}

impl MambaRuntime {
    pub fn load(
        bundle_dir: &Path,
        dispatch_name: &str,
        math_name: &str,
        pool_size: usize,
        auto_iters: usize,
        auto_warmup: usize,
        mem_policy: MemoryPolicy,
        cb_cfg: CircuitBreakerConfig,
        fallback: FallbackMode,
    ) -> Result<Self, String> {
        let manifest_path = bundle_dir.join("weights_manifest.json");
        let feature_schema_path = bundle_dir.join("feature_schema.json");
        let manifest = load_manifest(&manifest_path)?;
        let feature_schema = load_feature_schema(&feature_schema_path)?;

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
            return Err("gelu_kind must be erf".to_string());
        }
        if manifest.model_config.softplus_kind != "beta_threshold" {
            return Err("softplus_kind must be beta_threshold".to_string());
        }
        if manifest.model_config.static_dim_total != 386 {
            return Err("static_dim_total must be 386".to_string());
        }
        if manifest.model_config.micro_state_dim.unwrap_or(0) != 0 {
            return Err("micro_state_dim must be 0".to_string());
        }
        if manifest.model_config.seq_len != 128 {
            return Err("seq_len must be 128".to_string());
        }
        if manifest.model_config.vocab_size != 4097 {
            return Err("vocab_size must be 4097".to_string());
        }
        if manifest.model_config.feature_dim != 1 {
            return Err("feature_dim must be 1".to_string());
        }
        if feature_schema.static_total_dim != manifest.model_config.static_dim_total {
            return Err("feature_schema static_total_dim mismatch".to_string());
        }
        if feature_schema.input_ids_len != manifest.model_config.seq_len {
            return Err("feature_schema input_ids_len mismatch".to_string());
        }
        if let Some(fs_missing) = feature_schema.missing_id {
            if fs_missing != manifest.model_config.missing_id {
                return Err("feature_schema missing_id mismatch".to_string());
            }
        }
        if manifest.model_config.missing_id < 0
            || (manifest.model_config.missing_id as usize) >= manifest.model_config.vocab_size
        {
            return Err("missing_id must be within vocab_size".to_string());
        }
        if manifest.model_config.pad_id < 0
            || (manifest.model_config.pad_id as usize) >= manifest.model_config.vocab_size
        {
            return Err("pad_id must be within vocab_size".to_string());
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

        let feature_hash = file_sha256(&feature_schema_path)?;
        let schema_hash = manifest
            .schema_hash
            .as_deref()
            .ok_or_else(|| "missing schema_hash".to_string())?;
        let feature_sha = manifest
            .feature_schema_sha256
            .as_deref()
            .ok_or_else(|| "missing feature_schema_sha256".to_string())?;
        if feature_hash != feature_sha {
            return Err("feature_schema_sha256 mismatch".to_string());
        }
        if feature_hash != schema_hash {
            return Err("schema_hash mismatch".to_string());
        }

        let dims = derive_dims(&manifest)?;
        validate_manifest_shapes(&manifest, dims)?;

        let weights_path = bundle_dir.join(&manifest.weights_file);
        let mut blob = Vec::new();
        File::open(&weights_path)
            .map_err(|e| e.to_string())?
            .read_to_end(&mut blob)
            .map_err(|e| e.to_string())?;

        let blob_sha = sha256_hex(&blob);
        if blob_sha != manifest.weights_sha256 {
            return Err("weights_sha256 mismatch".to_string());
        }

        prefetch_blob(&blob);

        if dims.emb_vocab != manifest.model_config.vocab_size {
            println!(
                "note: emb.weight rows {} != vocab_size {}, using effective vocab_size and validating extra row",
                dims.emb_vocab, manifest.model_config.vocab_size
            );
        }
        if dims.emb_vocab == manifest.model_config.vocab_size + 1 {
            let spec = manifest
                .tensors
                .iter()
                .find(|t| t.name == "emb.weight")
                .ok_or_else(|| "missing emb.weight".to_string())?;
            let emb_w = slice_tensor(&blob, spec)?;
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
                if std::env::var("RISK_MAMBA_STRICT_EMB_ROW").is_ok() {
                    return Err(format!(
                        "emb.weight extra row not zero (max_abs={})",
                        max_abs
                    ));
                }
                println!(
                    "warning: emb.weight extra row not zero (max_abs={}); continuing (set RISK_MAMBA_STRICT_EMB_ROW=1 to fail)",
                    max_abs
                );
            }
        }

        // Leak blob for 'static slices (server lifetime == process lifetime).
        let blob_static: &'static [u8] = Box::leak(blob.into_boxed_slice());
        let math_backend = resolve_math(math_name)?;
        let mut weights = build_weights(&manifest, dims, blob_static)?;
        weights.precompute_a_pre();
        weights.precompute_packed(16);
        if matches!(math_backend, MathBackend::FastBf16) {
            weights.precompute_packed_bf16(16);
        }

        let cfg = weights.cfg.clone();
        let (dispatch, dispatch_label) = resolve_dispatch_with_microbench(
            dispatch_name,
            &cfg,
            &weights,
            math_backend,
            auto_iters.max(1),
            auto_warmup,
        )?;

        let scratch_len = scratch_bytes_full(&cfg);
        let mut scratch_pool = Vec::with_capacity(pool_size.max(1));
        for _ in 0..pool_size.max(1) {
            let mut buf = AlignedBuf::new(scratch_len, ALIGN_BYTES)?;
            apply_madvise(&mut buf, mem_policy)?;
            if mem_policy.pretouch {
                pretouch_buf(&mut buf);
            }
            scratch_pool.push(buf);
        }
        apply_blob_madvise(blob_static, mem_policy)?;
        if mem_policy.pretouch {
            pretouch_blob(blob_static);
        }

        println!(
            "mamba mem_policy pretouch={} hugepage={:?} willneed={}",
            mem_policy.pretouch, mem_policy.hugepage, mem_policy.willneed
        );
        Ok(Self {
            cfg,
            weights,
            dispatch,
            math_backend,
            dispatch_label,
            math_label: math_name.to_string(),
            schema_hash: manifest
                .schema_hash
                .clone()
                .unwrap_or_else(|| "none".to_string()),
            scratch_pool: Mutex::new(scratch_pool),
            cb: CircuitBreaker::new(cb_cfg),
            fallback,
        })
    }

    pub fn schema_hash(&self) -> &str {
        &self.schema_hash
    }

    pub fn dispatch_label(&self) -> &str {
        &self.dispatch_label
    }

    pub fn kernel_path_label(&self) -> &'static str {
        kernel_path_tag(&self.cfg)
    }

    pub fn layout_label(&self) -> &'static str {
        layout_tag(&self.cfg, self.math_backend, false, self.dispatch)
    }

    pub fn mlp_impl_label(&self) -> &'static str {
        if mlp_fused_enabled(self.cfg.seq_len, false, self.math_backend) {
            "fused_streamed"
        } else if self.cfg.seq_len == 1 {
            "unfused_m1"
        } else if self
            .weights
            .layers
            .first()
            .map(|layer| layer.fc1_w_packed.is_some() && self.cfg.d_mlp % 16 == 0)
            .unwrap_or(false)
        {
            "unfused_packed"
        } else {
            "unfused_unpacked"
        }
    }

    pub fn math_label(&self) -> &str {
        &self.math_label
    }

    pub fn input_ids_len(&self) -> usize {
        self.cfg.seq_len * self.cfg.feature_dim
    }

    pub fn static_dim_total(&self) -> usize {
        self.cfg.static_dim_total
    }

    pub fn score(
        &self,
        input_ids: &[i64],
        static_total: &[f32],
    ) -> Result<MambaOutcome, String> {
        if input_ids.len() != self.cfg.seq_len * self.cfg.feature_dim {
            return Err("input_ids length mismatch".to_string());
        }
        if static_total.len() != self.cfg.static_dim_total {
            return Err("static_total length mismatch".to_string());
        }

        if self.cb.is_tripped() {
            let out = self.run_with_fallback(input_ids, static_total, "cb_trip")?;
            return Ok(out);
        }

        match self.run_kernel(self.dispatch, self.math_backend, input_ids, static_total) {
            Ok(out) => {
                self.cb.record(true);
                Ok(out)
            }
            Err(e) => {
                self.cb.record(false);
                let out = self.run_with_fallback(input_ids, static_total, &e)?;
                Ok(out)
            }
        }
    }

    fn run_with_fallback(
        &self,
        input_ids: &[i64],
        static_total: &[f32],
        reason: &str,
    ) -> Result<MambaOutcome, String> {
        match self.fallback {
            FallbackMode::Exact => {
                let mut out =
                    self.run_kernel(CpuDispatch::Scalar, MathBackend::Exact, input_ids, static_total)?;
                out.fallback_reason = Some(format!("fallback_exact:{}", reason));
                Ok(out)
            }
            FallbackMode::DefaultZero => Ok(MambaOutcome {
                logits: [0.0, 0.0],
                margin: 0.0,
                dispatch: "scalar".to_string(),
                math: "exact".to_string(),
                fallback_reason: Some(format!("fallback_zero:{}", reason)),
            }),
        }
    }

    fn run_kernel(
        &self,
        dispatch: CpuDispatch,
        math_backend: MathBackend,
        input_ids: &[i64],
        static_total: &[f32],
    ) -> Result<MambaOutcome, String> {
        let mut scratch = ScratchGuard::new(&self.scratch_pool)?;
        let mut logits = [0.0f32; 2];
        let mut margin = [0.0f32; 1];

        full_stateless_forward(
            &self.cfg,
            &self.weights,
            dispatch,
            math_backend,
            input_ids,
            static_total,
            None,
            scratch.as_mut_slice(),
            &mut logits,
            &mut margin,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| e.to_string())?;

        if !margin[0].is_finite() || !logits[0].is_finite() || !logits[1].is_finite() {
            return Err("nan_or_inf".to_string());
        }
        Ok(MambaOutcome {
            logits,
            margin: margin[0],
            dispatch: dispatch_label(dispatch).to_string(),
            math: math_label(math_backend).to_string(),
            fallback_reason: None,
        })
    }
}

struct ScratchGuard<'a> {
    buf: Option<AlignedBuf>,
    pool: &'a Mutex<Vec<AlignedBuf>>,
}

impl<'a> ScratchGuard<'a> {
    fn new(pool: &'a Mutex<Vec<AlignedBuf>>) -> Result<Self, String> {
        let mut guard = pool.lock().map_err(|_| "scratch_pool poisoned".to_string())?;
        let buf = guard.pop().ok_or_else(|| "scratch_pool empty".to_string())?;
        Ok(Self {
            buf: Some(buf),
            pool,
        })
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buf.as_mut().unwrap().as_mut_slice()
    }
}

impl Drop for ScratchGuard<'_> {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            if let Ok(mut guard) = self.pool.lock() {
                guard.push(buf);
            }
        }
    }
}

struct CircuitBreaker {
    cfg: CircuitBreakerConfig,
    state: Mutex<CircuitState>,
}

struct CircuitState {
    window_start: Instant,
    total: u64,
    errors: u64,
    consecutive_errors: u64,
    tripped_until: Option<Instant>,
}

impl CircuitBreaker {
    fn new(cfg: CircuitBreakerConfig) -> Self {
        Self {
            cfg,
            state: Mutex::new(CircuitState {
                window_start: Instant::now(),
                total: 0,
                errors: 0,
                consecutive_errors: 0,
                tripped_until: None,
            }),
        }
    }

    fn is_tripped(&self) -> bool {
        let mut st = match self.state.lock() {
            Ok(v) => v,
            Err(e) => e.into_inner(),
        };
        if let Some(until) = st.tripped_until {
            if Instant::now() < until {
                return true;
            }
            st.tripped_until = None;
        }
        false
    }

    fn record(&self, ok: bool) {
        let mut st = match self.state.lock() {
            Ok(v) => v,
            Err(e) => e.into_inner(),
        };
        let now = Instant::now();
        if now.duration_since(st.window_start).as_millis() as u64 > self.cfg.window_ms {
            st.window_start = now;
            st.total = 0;
            st.errors = 0;
            st.consecutive_errors = 0;
        }
        st.total += 1;
        if ok {
            st.consecutive_errors = 0;
        } else {
            st.errors += 1;
            st.consecutive_errors += 1;
        }
        let rate = if st.total == 0 {
            0.0
        } else {
            st.errors as f64 / st.total as f64
        };
        if st.consecutive_errors >= self.cfg.max_errors || rate >= self.cfg.error_rate {
            st.tripped_until = Some(now + Duration::from_millis(self.cfg.trip_ms));
            st.consecutive_errors = 0;
            st.total = 0;
            st.errors = 0;
        }
    }
}

struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    layout: std::alloc::Layout,
}

// Safety: AlignedBuf is a raw allocation owned by this struct. We only hand out
// mutable slices under exclusive access (scratch pool + guard), so cross-thread
// moves are safe as long as access is serialized.
unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

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

#[derive(Deserialize)]
struct Manifest {
    format: String,
    schema_version: u32,
    schema_hash: Option<String>,
    feature_schema_sha256: Option<String>,
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

#[derive(Deserialize)]
struct FeatureSchema {
    input_ids_len: usize,
    static_base_dim: usize,
    micro_state_dim: usize,
    static_total_dim: usize,
    missing_id: Option<i64>,
}

fn resolve_math(name: &str) -> Result<MathBackend, String> {
    match name {
        "fast" => Ok(MathBackend::Approx),
        "fast_wild" => Ok(MathBackend::FastWild),
        "exact" => Ok(MathBackend::Exact),
        "fast3_bf16" => Ok(MathBackend::FastBf16),
        other => Err(format!("invalid math {}", other)),
    }
}

fn resolve_dispatch_with_microbench(
    name: &str,
    cfg: &ModelConfig,
    weights: &WeightsView,
    math_backend: MathBackend,
    iters: usize,
    warmup: usize,
) -> Result<(CpuDispatch, String), String> {
    if name != "auto" {
        let dispatch = resolve_dispatch(name)?;
        return Ok((dispatch, dispatch_label(dispatch).to_string()));
    }

    let mut best = CpuDispatch::Scalar;
    let mut best_ms = f64::INFINITY;
    let candidates = dispatch_candidates();
    let mut scratch = vec![0u8; scratch_bytes_full(cfg)];
    let input_ids = vec![0i64; cfg.seq_len * cfg.feature_dim];
    let static_total = vec![0.0f32; cfg.static_dim_total];
    let mut logits = [0.0f32; 2];
    let mut margin = [0.0f32; 1];

    for &dispatch in &candidates {
        for _ in 0..warmup {
            let _ = full_stateless_forward(
                cfg,
                weights,
                dispatch,
                math_backend,
                &input_ids,
                &static_total,
                None,
                &mut scratch,
                &mut logits,
                &mut margin,
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
                weights,
                dispatch,
                math_backend,
                &input_ids,
                &static_total,
                None,
                &mut scratch,
                &mut logits,
                &mut margin,
                None,
                None,
                None,
                None,
            );
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        if ms < best_ms {
            best_ms = ms;
            best = dispatch;
        }
    }
    Ok((best, dispatch_label(best).to_string()))
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

fn resolve_dispatch(name: &str) -> Result<CpuDispatch, String> {
    match name {
        "scalar" => Ok(CpuDispatch::Scalar),
        "avx2" => Ok(CpuDispatch::Avx2),
        "avx512" => Ok(CpuDispatch::Avx512),
        "auto" => Ok(CpuDispatch::Scalar),
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

fn math_label(backend: MathBackend) -> &'static str {
    match backend {
        MathBackend::Approx => "fast",
        MathBackend::FastWild => "fast_wild",
        MathBackend::Exact => "exact",
        MathBackend::Sleef => "sleef",
        MathBackend::FastBf16 => "fast3_bf16",
    }
}

fn load_manifest(path: &Path) -> Result<Manifest, String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&data).map_err(|e| e.to_string())
}

fn load_feature_schema(path: &Path) -> Result<FeatureSchema, String> {
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&data).map_err(|e| e.to_string())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    format!("{:x}", digest)
}

fn prefetch_blob(blob: &[u8]) {
    let mut sum: u64 = 0;
    let step = 4096;
    for chunk in blob.chunks(step) {
        if let Some(&b) = chunk.first() {
            sum = sum.wrapping_add(b as u64);
        }
    }
    if sum == 0 {
        // Prevent optimizing away
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }
}

pub fn mlockall() -> Result<(), String> {
    #[cfg(target_os = "linux")]
    unsafe {
        let rc = libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE);
        if rc != 0 {
            return Err(format!(
                "mlockall failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        return Ok(());
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err("mlockall is only supported on linux".to_string())
    }
}

fn apply_blob_madvise(blob: &[u8], policy: MemoryPolicy) -> Result<(), String> {
    if policy.hugepage == HugepagePolicy::Never && !policy.willneed {
        return Ok(());
    }
    let ptr = blob.as_ptr() as *mut u8;
    let len = blob.len();
    if policy.willneed {
        madvise_range(ptr, len, libc::MADV_WILLNEED)?;
    }
    if policy.hugepage == HugepagePolicy::Madvise {
        madvise_range(ptr, len, libc::MADV_HUGEPAGE)?;
    }
    Ok(())
}

fn apply_madvise(buf: &mut AlignedBuf, policy: MemoryPolicy) -> Result<(), String> {
    if policy.hugepage == HugepagePolicy::Never && !policy.willneed {
        return Ok(());
    }
    let ptr = buf.ptr;
    let len = buf.len;
    if policy.willneed {
        madvise_range(ptr, len, libc::MADV_WILLNEED)?;
    }
    if policy.hugepage == HugepagePolicy::Madvise {
        madvise_range(ptr, len, libc::MADV_HUGEPAGE)?;
    }
    Ok(())
}

fn madvise_range(ptr: *mut u8, len: usize, advice: i32) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    unsafe {
        let rc = libc::madvise(ptr as *mut libc::c_void, len, advice);
        if rc != 0 {
            return Err(format!(
                "madvise {:?} failed: {}",
                advice,
                std::io::Error::last_os_error()
            ));
        }
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, len, advice);
        Ok(())
    }
}

fn pretouch_blob(blob: &[u8]) {
    let ptr = blob.as_ptr();
    pretouch_range(ptr, blob.len());
}

fn pretouch_buf(buf: &mut AlignedBuf) {
    let ptr = buf.ptr as *const u8;
    pretouch_range(ptr, buf.len);
}

fn pretouch_range(ptr: *const u8, len: usize) {
    let mut sum: u64 = 0;
    for off in (0..len).step_by(PAGE_BYTES) {
        let v = unsafe { std::ptr::read_volatile(ptr.add(off)) };
        sum = sum.wrapping_add(v as u64);
    }
    if sum == 0 {
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    Ok(sha256_hex(&buf))
}

fn check_tensor_sha(blob: &[u8], spec: &TensorSpec) -> Result<(), String> {
    let data = &blob[spec.offset..spec.offset + spec.nbytes];
    let actual = sha256_hex(data);
    if actual != spec.sha256 {
        return Err(format!("tensor {} sha256 mismatch", spec.name));
    }
    Ok(())
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
    Ok(weights)
}
