//! Core engine for sadda. Hosts the cross-cutting types (`Audio`, `Project`,
//! `EngineError`) and the analyses (currently autocorrelation f0) consumed by
//! the Python, UniFFI, and desktop-app crates.
#![warn(missing_docs)]

pub mod audio;
pub mod corpus;
pub mod error;
pub mod pitch;

pub use audio::Audio;
pub use corpus::{Bundle, Project, SCHEMA_VERSION};
pub use error::{EngineError, Result};
pub use pitch::{PitchConfig, PitchFrame, autocorrelation};

/// Returns the engine crate's semver string, taken from `Cargo.toml` at build
/// time. Useful as a sanity check at the language-binding boundaries.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_nonempty() {
        assert!(!version().is_empty());
    }
}
