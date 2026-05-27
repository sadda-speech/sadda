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

/// A model resolved from the registry by [`load_model`].
#[pyclass(name = "Model")]
pub(crate) struct PyModel {
    inner: sadda_engine::Model,
}

#[pymethods]
impl PyModel {
    /// Resolvable id (e.g. `"sadda/silero-vad"`).
    #[getter]
    fn id(&self) -> &str {
        self.inner.id()
    }
    /// Version.
    #[getter]
    fn version(&self) -> &str {
        self.inner.version()
    }
    /// Model kind (`vad`, `embedding`, …).
    #[getter]
    fn kind(&self) -> &str {
        self.inner.kind()
    }
    /// Human-readable title.
    #[getter]
    fn title(&self) -> &str {
        &self.inner.manifest.title
    }
    /// SPDX license id, if declared.
    #[getter]
    fn license(&self) -> Option<String> {
        self.inner.manifest.license.clone()
    }
    /// Weights checksum (`sha256:…`), if declared.
    #[getter]
    fn weights_checksum(&self) -> Option<String> {
        self.inner.weights_checksum().map(str::to_string)
    }

    /// Runs this model as a VAD over `audio` → `(times, speech_probs)`.
    /// Errors unless it's a `vad` model and ONNX Runtime is available.
    #[allow(clippy::type_complexity)]
    fn vad<'py>(
        &self,
        py: Python<'py>,
        audio: &PyAudio,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f32>>)> {
        let frames = self.inner.vad(&audio.inner).map_err(engine_err_to_py)?;
        let times: Vec<f64> = frames.iter().map(|f| f.time_seconds).collect();
        let probs: Vec<f32> = frames.iter().map(|f| f.speech_prob).collect();
        Ok((times.into_pyarray(py), probs.into_pyarray(py)))
    }

    fn __repr__(&self) -> String {
        format!(
            "Model(id={:?}, version={:?}, kind={:?})",
            self.inner.id(),
            self.inner.version(),
            self.inner.kind()
        )
    }
}

/// Resolves a model by id: `sadda/<name>[@version]` (curated / bundled),
/// `local://<path>` (a model dir or bare file), or `hf://<repo>` (arrives
/// in E12). Returns a [`PyModel`].
#[pyfunction]
pub(crate) fn load_model(id: &str) -> PyResult<PyModel> {
    Ok(PyModel {
        inner: sadda_engine::load_model(id).map_err(engine_err_to_py)?,
    })
}
