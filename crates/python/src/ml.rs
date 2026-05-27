//! PyO3 bindings for E11 ML inference — `sadda.ml.*`. Thin wrappers over
//! `sadda_engine::ml` (the `ml` feature is on by default for the wheel).
//! ONNX Runtime is loaded at runtime (`load-dynamic`); these functions
//! raise a clean error (not a crash) if it isn't present.

use std::path::Path;

use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use crate::PyAudio;
use crate::engine_err_to_py;

fn run_vad(audio: &PyAudio, model_path: Option<String>) -> PyResult<Vec<sadda_engine::VadFrame>> {
    match model_path {
        Some(p) => sadda_engine::vad(&audio.inner, Path::new(&p)),
        None => sadda_engine::vad_bundled(&audio.inner),
    }
    .map_err(engine_err_to_py)
}

/// Runs Silero VAD over `audio`, returning `(times, speech_probs)` as
/// NumPy arrays — one window (~32 ms at 16 kHz) per element. Uses the
/// bundled model unless `model_path` is given. Raises if ONNX Runtime
/// isn't available (set `ORT_DYLIB_PATH`).
#[pyfunction]
#[pyo3(signature = (audio, *, model_path=None))]
#[allow(clippy::type_complexity)]
pub(crate) fn vad<'py>(
    py: Python<'py>,
    audio: &PyAudio,
    model_path: Option<String>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f32>>)> {
    let frames = run_vad(audio, model_path)?;
    let times: Vec<f64> = frames.iter().map(|f| f.time_seconds).collect();
    let probs: Vec<f32> = frames.iter().map(|f| f.speech_prob).collect();
    Ok((times.into_pyarray(py), probs.into_pyarray(py)))
}

/// Speech regions in `audio` as `(start_seconds, end_seconds)` tuples —
/// runs VAD, then merges windows with probability `>= threshold`. Uses
/// the bundled model unless `model_path` is given.
#[pyfunction]
#[pyo3(signature = (audio, *, threshold=0.5, model_path=None))]
pub(crate) fn speech_segments(
    audio: &PyAudio,
    threshold: f32,
    model_path: Option<String>,
) -> PyResult<Vec<(f64, f64)>> {
    let frames = run_vad(audio, model_path)?;
    Ok(sadda_engine::speech_segments(&frames, threshold)
        .iter()
        .map(|s| (s.start_seconds, s.end_seconds))
        .collect())
}
