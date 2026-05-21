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
}

/// Convenience alias for `Result<T, EngineError>`, mirroring the std lib's
/// `io::Result` convention.
pub type Result<T> = std::result::Result<T, EngineError>;
