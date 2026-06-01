//! Core engine for sadda. Hosts the cross-cutting types (`Audio`, `Project`,
//! `EngineError`) and the analyses (currently autocorrelation f0) consumed by
//! the Python, UniFFI, and desktop-app crates.
#![warn(missing_docs)]

pub mod audio;
pub mod citation;
pub mod clinical;
pub mod corpus;
pub mod criteria;
pub mod dsp;
pub mod error;
pub mod io;
pub mod live;
/// E11 ML inference (ONNX Runtime). Behind the `ml` feature; ONNX Runtime
/// is loaded at runtime (`load-dynamic`), not linked at build time.
#[cfg(feature = "ml")]
pub mod ml;
/// E11 model registry (consume side) — resolve + run ONNX models by id.
/// Behind the `ml` feature.
#[cfg(feature = "ml")]
pub mod models;
pub mod pitch;
pub mod refdist;
pub mod storage;
pub mod units;

pub use audio::Audio;
pub use citation::{Citation, citation_for};
pub use clinical::{
    CppsConfig, GneConfig, H1H2Config, HnrConfig, HnrDConfig, PerturbationConfig,
    PerturbationReport, abi, avqi, cpps, gne, h1_h2, hfno, hnr, hnr_d, perturbation,
};
pub use corpus::{
    ASSIGNMENT_ROLES, ASSIGNMENT_STATUSES, Assignment, AssignmentSpec, Bundle, BundleSpec,
    Calibration, Criterion, DerivedSignal, ExportSummary, ImportSummary, Instrument, InstrumentSpec,
    Interval, IntervalSpec, LabelCheck, Point, PointSpec, ProcessingRunKind, ProcessingRunRow,
    ProcessingRunSpec, ProcessingRunStatus, Project, RecipeRun, Reference, ReferenceSpec, Rubric,
    RubricTier, Session, SessionSpec, Speaker, SpeakerSpec, StatusDef, TARGET_STATUSES, Target,
    TargetSpec, Tier, TierRows, TierSpec, TierType, VocabEntry,
};
pub use criteria::expr::{Expr, SampledSignal, SignalSet};
pub use criteria::{CriterionRule, Emit, EvalInterval, Proposal, Selector};
pub use error::{EngineError, Result};
pub use live::{
    LiveConfig, LiveFormantsFrame, LiveIntensityFrame, LivePitchFrame, LiveResults, LiveSession,
    MeterFrame, StoppedSession,
};
#[cfg(feature = "ml")]
pub use ml::{SpeechSegment, VadFrame, probe_ort_dylib, speech_segments, vad};
#[cfg(feature = "ml")]
pub use models::{
    Model, ModelManifest, ModelRegistryEntry, ModelRegistryIndex, ModelStore, load_model,
    parse_model_index, vad_bundled, verify_checksum,
};
pub use pitch::{PitchConfig, PitchFrame, autocorrelation};
pub use refdist::{
    Citation as RefdistCitation, Histogram, Measure, MeasureKind, Population, Privacy, QuerySpec,
    RefDist, RefdistManifest, RefdistStore, RegistryEntry, RegistryIndex, Schema as RefdistSchema,
    Summary, scaffold,
};
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
