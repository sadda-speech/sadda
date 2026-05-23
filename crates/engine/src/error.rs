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
}

/// Convenience alias for `Result<T, EngineError>`, mirroring the std lib's
/// `io::Result` convention.
pub type Result<T> = std::result::Result<T, EngineError>;
