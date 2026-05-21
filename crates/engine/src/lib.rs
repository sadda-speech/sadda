//! Core engine for sadda. Placeholder skeleton; DSP, corpus, and I/O land in Phase 0–1.

pub mod audio;
pub mod error;

pub use audio::Audio;
pub use error::{EngineError, Result};

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
