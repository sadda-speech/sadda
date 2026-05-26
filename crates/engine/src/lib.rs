//! Core engine for sadda. Hosts the cross-cutting types (`Audio`, `Project`,
//! `EngineError`) and the analyses (currently autocorrelation f0) consumed by
//! the Python, UniFFI, and desktop-app crates.
#![warn(missing_docs)]

pub mod audio;
pub mod citation;
pub mod clinical;
pub mod corpus;
pub mod dsp;
pub mod error;
pub mod io;
pub mod live;
pub mod pitch;
pub mod storage;
pub mod units;

pub use audio::Audio;
pub use citation::{Citation, citation_for};
pub use clinical::{
    CppsConfig, GneConfig, H1H2Config, HnrConfig, PerturbationConfig, PerturbationReport, avqi,
    cpps, gne, h1_h2, hnr, perturbation,
};
pub use corpus::{
    Bundle, BundleSpec, Calibration, DerivedSignal, Instrument, InstrumentSpec, Interval,
    IntervalSpec, Point, PointSpec, ProcessingRunKind, ProcessingRunRow, ProcessingRunSpec,
    ProcessingRunStatus, Project, RecipeRun, Reference, ReferenceSpec, Session, SessionSpec,
    Speaker, SpeakerSpec, Tier, TierRows, TierSpec, TierType,
};
pub use error::{EngineError, Result};
pub use live::{
    LiveConfig, LiveFormantsFrame, LiveIntensityFrame, LivePitchFrame, LiveResults, LiveSession,
    MeterFrame, StoppedSession,
};
pub use pitch::{PitchConfig, PitchFrame, autocorrelation};
pub use units::{Decibels, Hertz, Ratio, Seconds};

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
