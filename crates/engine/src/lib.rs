//! Core engine for sadda. Hosts the cross-cutting types (`Audio`, `Project`,
//! `EngineError`) and the analyses (currently autocorrelation f0) consumed by
//! the Python, UniFFI, and desktop-app crates.
#![warn(missing_docs)]

pub mod audio;
pub mod corpus;
pub mod dsp;
pub mod error;
pub mod io;
pub mod live;
pub mod pitch;
pub mod storage;

pub use audio::Audio;
pub use corpus::{
    Bundle, BundleSpec, DerivedSignal, Interval, IntervalSpec, Point, PointSpec, Project,
    Reference, ReferenceSpec, Session, SessionSpec, Speaker, SpeakerSpec, Tier, TierRows, TierSpec,
    TierType,
};
pub use error::{EngineError, Result};
pub use live::{
    LiveConfig, LiveFormantsFrame, LiveIntensityFrame, LivePitchFrame, LiveResults, LiveSession,
    MeterFrame, StoppedSession,
};
pub use pitch::{PitchConfig, PitchFrame, autocorrelation};

/// Returns the engine crate's semver string, taken from `Cargo.toml` at build
/// time. Useful as a sanity check at the language-binding boundaries.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Returns the highest corpus-database schema version this build of the
/// engine knows how to apply. Bumped whenever a new migration is added under
/// `crates/engine/migrations/`.
pub fn schema_version() -> i64 {
    corpus::migrations::engine_max_version()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_nonempty() {
        assert!(!version().is_empty());
    }
}
