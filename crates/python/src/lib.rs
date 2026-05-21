//! PyO3 bindings for sadda — builds into the `sadda` Python module via maturin.
//!
//! Doc comments on `#[pyfunction]`, `#[pyclass]`, and `#[pymethods]` items here
//! become the corresponding Python `__doc__` strings, so they serve both the
//! Rust API reference and `help(sadda.X)` in a REPL.
#![warn(missing_docs)]

use std::path::PathBuf;

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

fn engine_err_to_py(e: sadda_engine::EngineError) -> PyErr {
    match e {
        sadda_engine::EngineError::Io(err) => PyIOError::new_err(err.to_string()),
        sadda_engine::EngineError::WavDecode(err) => {
            PyValueError::new_err(format!("WAV decode error: {err}"))
        }
        sadda_engine::EngineError::UnsupportedFormat(msg) => {
            PyValueError::new_err(format!("unsupported audio format: {msg}"))
        }
        sadda_engine::EngineError::Sqlite(err) => {
            PyRuntimeError::new_err(format!("corpus database error: {err}"))
        }
        sadda_engine::EngineError::Corpus(msg) => {
            PyRuntimeError::new_err(format!("corpus error: {msg}"))
        }
        sadda_engine::EngineError::SchemaTooNew {
            db_version,
            engine_max,
        } => PyRuntimeError::new_err(format!(
            "corpus database schema (v{db_version}) is newer than this engine (max v{engine_max}); upgrade sadda or restore an older backup"
        )),
    }
}

/// Audio data loaded from disk. Samples are interleaved float32 in `[-1.0, 1.0]`;
/// for stereo the layout is `[L0, R0, L1, R1, ...]`. Construct via
/// `sadda.load_wav(path)`.
#[pyclass(name = "Audio")]
struct PyAudio {
    inner: sadda_engine::Audio,
}

#[pymethods]
impl PyAudio {
    /// Sample rate in Hz.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }

    /// Number of audio channels (1 = mono, 2 = stereo, …).
    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels
    }

    /// Number of audio frames (samples per channel).
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.frame_count()
    }

    /// Audio duration in seconds.
    #[getter]
    fn duration_seconds(&self) -> f64 {
        self.inner.duration_seconds()
    }

    /// Interleaved samples as a 1-D float32 NumPy array.
    /// For stereo the layout is `[L0, R0, L1, R1, ...]`.
    #[getter]
    fn samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.samples.clone().into_pyarray(py)
    }

    /// Mono mixdown of the audio as a 1-D float32 NumPy array.
    fn mono<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mono: Vec<f32> = self.inner.mono_samples().collect();
        mono.into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Audio(sample_rate={}, channels={}, n_frames={}, duration_seconds={:.4})",
            self.inner.sample_rate,
            self.inner.channels,
            self.inner.frame_count(),
            self.inner.duration_seconds()
        )
    }
}

/// Returns the underlying engine version string.
#[pyfunction]
fn version() -> &'static str {
    sadda_engine::version()
}

/// Loads a WAV file from disk. Returns a sadda.Audio.
#[pyfunction]
fn load_wav(path: PathBuf) -> PyResult<PyAudio> {
    let audio = sadda_engine::Audio::from_wav_path(&path).map_err(engine_err_to_py)?;
    Ok(PyAudio { inner: audio })
}

/// Estimates f0 over an Audio via time-domain autocorrelation.
///
/// Returns `(times, frequencies)` as a 2-tuple of NumPy arrays:
/// `times` is float64 in seconds, `frequencies` is float32 in Hz.
#[pyfunction]
#[pyo3(signature = (audio, *, frame_size_seconds=0.030, hop_size_seconds=0.010, min_freq_hz=75.0, max_freq_hz=500.0))]
fn f0<'py>(
    py: Python<'py>,
    audio: &PyAudio,
    frame_size_seconds: f32,
    hop_size_seconds: f32,
    min_freq_hz: f32,
    max_freq_hz: f32,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f32>>) {
    let config = sadda_engine::PitchConfig {
        frame_size_seconds,
        hop_size_seconds,
        min_freq_hz,
        max_freq_hz,
    };
    let frames = sadda_engine::autocorrelation(&audio.inner, &config);
    let times: Vec<f64> = frames.iter().map(|f| f.time_seconds).collect();
    let freqs: Vec<f32> = frames.iter().map(|f| f.frequency_hz).collect();
    (times.into_pyarray(py), freqs.into_pyarray(py))
}

/// sadda — open-source toolkit for phonetics and speech science research.
///
/// Phase 0 surface: `version()`, `load_wav(path)`, `f0(audio, ...)` and the
/// `Audio` class. Many more analyses land in Phase 1.
#[pymodule]
fn sadda(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(load_wav, m)?)?;
    m.add_function(wrap_pyfunction!(f0, m)?)?;
    m.add_class::<PyAudio>()?;
    Ok(())
}
