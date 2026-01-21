mod config;
mod error;
mod kernel;
mod math;
mod scratch;
mod weights;

pub use config::*;
pub use error::*;
pub use kernel::*;
pub use math::*;
pub use scratch::*;
pub use weights::*;

pub const SLEEF_AVAILABLE: bool = cfg!(feature = "sleef");
