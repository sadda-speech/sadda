use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("WAV decoding error: {0}")]
    WavDecode(#[from] hound::Error),

    #[error("unsupported audio format: {0}")]
    UnsupportedFormat(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;
