//! Error and `Result` types for the engine. Will split into per-domain error
//! enums (`CorpusError`, `AnalysisError`, …) as those surfaces solidify.

use thiserror::Error;

/// All errors the engine returns from public APIs.
///
/// This will grow into the `SatError` hierarchy sketched in the 2026-05-18
/// API surface entry — split into `CorpusError`, `AnalysisError`, `ModelError`,
/// etc. — as the surfaces solidify. At Phase 0 it's a single flat enum.
#[derive(Debug, Error)]
pub enum EngineError {
    /// An underlying I/O error (file not found, permission denied, …).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The audio file was found but couldn't be decoded as WAV.
    #[error("WAV decoding error: {0}")]
    WavDecode(#[from] hound::Error),

    /// The audio file's sample format / bit depth combination isn't supported
    /// by the WAV loader yet (e.g. 8-bit μ-law).
    #[error("unsupported audio format: {0}")]
    UnsupportedFormat(String),

    /// A corpus-database (SQLite) operation failed.
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// A corpus-level invariant was violated (missing directory, duplicate
    /// bundle, project not found, …).
    #[error("corpus error: {0}")]
    Corpus(String),

    /// A reference-distribution manifest failed to parse or a store
    /// operation failed (bad `refdist.toml`, missing data file, …).
    #[error("reference-distribution error: {0}")]
    RefDist(String),

    /// A DSP preset failed to parse, or a preset-registry store operation
    /// failed (malformed `*.toml`, unknown preset id, unwritable store
    /// directory, …). Surfaced by the on-disk MFCC preset registry.
    #[error("preset error: {0}")]
    Preset(String),

    /// A forced-alignment operation failed (A-series): the `espeak-ng` G2P
    /// binary is missing or errored, or a phonemized phone isn't in the acoustic
    /// model's vocabulary. Distinct from [`EngineError::Ml`] (the ONNX side).
    #[error("align error: {0}")]
    Align(String),

    /// An ML-inference operation failed (E11): ONNX Runtime not loadable
    /// (`ORT_DYLIB_PATH` unset / wrong version), a model failed to load,
    /// or inference errored. Surfaced only with the `ml` feature.
    #[error("ml error: {0}")]
    Ml(String),

    /// The corpus database is at a higher schema version than this engine
    /// knows how to read. Forward-compat clamp: the engine refuses to open
    /// rather than risk operating on tables it doesn't understand. Resolution
    /// is to upgrade the engine (or restore an older `corpus.db.bak.<n>`).
    #[error(
        "corpus database schema (v{db_version}) is newer than this engine (max v{engine_max}); upgrade sadda or restore an older backup"
    )]
    SchemaTooNew {
        /// Highest version found in the database's `schema_migrations` table.
        db_version: i64,
        /// Highest version known to this build of the engine.
        engine_max: i64,
    },

    /// A parent-child cardinality rule was violated at annotation insert
    /// time (missing required `parent_annotation_id`, non-existent parent,
    /// `one_to_one` violation, …). Surfaced by `Project::add_interval` /
    /// `add_point` / `add_reference`.
    #[error("cardinality violation: {0}")]
    Cardinality(String),

    /// `Project::open` / `Project::create` found a live
    /// `.sadda-lock` file in the project root. The slice-F10
    /// single-writer guarantee refuses to open a project a
    /// different process is already writing to.
    #[error(
        "project at {lockfile_path} is locked by PID {holder_pid} on {hostname}; \
         close the other sadda instance or remove the .sadda-lock file"
    )]
    ProjectLocked {
        /// PID recorded in the lockfile.
        holder_pid: u32,
        /// Hostname recorded in the lockfile.
        hostname: String,
        /// Filesystem path to the `.sadda-lock` file.
        lockfile_path: std::path::PathBuf,
    },

    /// A measurement could not be computed reliably on the given input
    /// (insufficient signal, out-of-range parameters, no voiced frames,
    /// …). Returned *instead of* a guessed number — the no-silent-
    /// fallback discipline for clinical-path measures from the
    /// 2026-05-18 clinical-regulatory entry. Clinical measures return
    /// this rather than a fabricated value when their preconditions
    /// aren't met.
    #[error("measure '{measure}' could not be computed reliably: {reason}")]
    Unreliable {
        /// The measure that couldn't be computed (e.g. `"cpps"`).
        measure: String,
        /// Why it couldn't be computed reliably.
        reason: String,
    },
}

impl EngineError {
    /// Builds an [`EngineError::Unreliable`] for a clinical-path measure
    /// whose preconditions weren't met. Use this instead of returning a
    /// guessed value.
    pub fn unreliable(measure: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Unreliable {
            measure: measure.into(),
            reason: reason.into(),
        }
    }
}

/// Convenience alias for `Result<T, EngineError>`, mirroring the std lib's
/// `io::Result` convention.
pub type Result<T> = std::result::Result<T, EngineError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unreliable_names_the_measure_and_reason() {
        let e = EngineError::unreliable("cpps", "no voiced frames in selection");
        assert!(matches!(e, EngineError::Unreliable { .. }));
        let msg = e.to_string();
        assert!(msg.contains("cpps"));
        assert!(msg.contains("no voiced frames"));
    }
}
