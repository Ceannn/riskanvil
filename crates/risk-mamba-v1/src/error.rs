use std::fmt;

#[derive(Debug)]
pub enum MicroStateError {
    BadLen(&'static str),
    BadSchema(String),
    MissingKey,
    Io(String),
}

impl fmt::Display for MicroStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MicroStateError::BadLen(what) => write!(f, "bad length: {}", what),
            MicroStateError::BadSchema(msg) => write!(f, "bad schema: {}", msg),
            MicroStateError::MissingKey => write!(f, "missing entity key"),
            MicroStateError::Io(msg) => write!(f, "io error: {}", msg),
        }
    }
}

impl std::error::Error for MicroStateError {}

impl From<std::io::Error> for MicroStateError {
    fn from(err: std::io::Error) -> Self {
        MicroStateError::Io(err.to_string())
    }
}

impl From<serde_json::Error> for MicroStateError {
    fn from(err: serde_json::Error) -> Self {
        MicroStateError::Io(err.to_string())
    }
}
