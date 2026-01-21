#[derive(Debug)]
pub enum KernelError {
    BadLen(&'static str),
    BadAlign(&'static str),
    CpuFeatureMissing(&'static str),
    UnsupportedDispatch(&'static str),
    InvalidConfig(&'static str),
    MissingWeights(&'static str),
    DebugTapError(String),
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelError::BadLen(msg) => write!(f, "bad length: {}", msg),
            KernelError::BadAlign(msg) => write!(f, "bad alignment: {}", msg),
            KernelError::CpuFeatureMissing(msg) => write!(f, "cpu feature missing: {}", msg),
            KernelError::UnsupportedDispatch(msg) => write!(f, "unsupported dispatch: {}", msg),
            KernelError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
            KernelError::MissingWeights(msg) => write!(f, "missing weights: {}", msg),
            KernelError::DebugTapError(msg) => write!(f, "debug tap error: {}", msg),
        }
    }
}

impl std::error::Error for KernelError {}
