//! Core engine for sadda. Placeholder skeleton; DSP, corpus, and I/O land in Phase 0–1.

pub mod audio;
pub mod corpus;
pub mod error;
pub mod pitch;

pub use audio::Audio;
pub use corpus::{Bundle, Project, SCHEMA_VERSION};
pub use error::{EngineError, Result};
pub use pitch::{PitchConfig, PitchFrame, autocorrelation};

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
