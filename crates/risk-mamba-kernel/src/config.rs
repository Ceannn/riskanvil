#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeluKind {
    Erf,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SoftplusKind {
    Exact,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogitsKind {
    NativeTwoClass,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadInputKind {
    HeadIn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardKind {
    Step1,
    FullStateless,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSurgery {
    None,
    AddFirst,
    AddAll,
    GateAll,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorSource {
    None,
    Static,
    Uid,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbSumMode {
    Fast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuDispatch {
    Scalar,
    Avx2,
    Avx512,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathBackend {
    Exact,
    Approx,
    Sleef,
    FastBf16,
    FastWild,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub forward_kind: ForwardKind,
    pub n_layers: usize,
    pub seq_len: usize,
    pub feature_dim: usize,
    pub vocab_size: usize,
    pub static_dim_total: usize,
    pub micro_state_dim: usize,
    pub d_model: usize,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_state_pad: usize,
    pub d_mlp: usize,
    pub conv_kernel: usize,
    pub dt_rank: usize,
    pub ln_eps1: f32,
    pub ln_eps2: f32,
    pub gelu_kind: GeluKind,
    pub softplus_kind: SoftplusKind,
    pub softplus_beta: f32,
    pub softplus_threshold: f32,
    pub logits_kind: LogitsKind,
    pub head_input_kind: HeadInputKind,
    pub n_class: usize,
    pub state_surgery: StateSurgery,
    pub prior_source: PriorSource,
    pub emb_sum_mode: EmbSumMode,
    pub prior_alpha: f32,
}
