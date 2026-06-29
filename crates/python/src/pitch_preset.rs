//! PyO3 bindings for the on-disk pitch preset registry (roadmap item 6).
//! Mirrors `mfcc_preset.rs`: build a [`PitchParams`] (method + config) from a
//! reference preset, override individual parameters, browse/save user presets,
//! and run a tracker from params. Re-exported under `sadda.dsp.*`
//! (PROVISIONAL). Unstubbed, per the refdist/ml/mfcc_preset convention.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sadda_engine::pitch::{PitchConfig, PitchMethod, PitchParams, pitch_with_params};
use sadda_engine::pitch_preset::{PitchPreset, PitchPresetStore, pitch_builtin_presets};

use crate::{PyAudio, engine_err_to_py};

fn method_str(m: PitchMethod) -> &'static str {
    match m {
        PitchMethod::Autocorrelation => "autocorrelation",
        PitchMethod::WindowedAutocorrelation => "windowed_autocorrelation",
        PitchMethod::Boersma => "boersma",
        PitchMethod::Yin => "yin",
        PitchMethod::PYin => "pyin",
        PitchMethod::Swipe => "swipe",
    }
}
fn parse_method(s: &str) -> PyResult<PitchMethod> {
    match s {
        "autocorrelation" => Ok(PitchMethod::Autocorrelation),
        "windowed_autocorrelation" => Ok(PitchMethod::WindowedAutocorrelation),
        "boersma" => Ok(PitchMethod::Boersma),
        "yin" => Ok(PitchMethod::Yin),
        "pyin" => Ok(PitchMethod::PYin),
        "swipe" => Ok(PitchMethod::Swipe),
        other => Err(PyValueError::new_err(format!(
            "unknown pitch method {other:?}; expected autocorrelation | \
             windowed_autocorrelation | boersma | yin | pyin | swipe"
        ))),
    }
}

// ---- PitchParams -----------------------------------------------------------

/// A full pitch-tracking spec: a method plus its config. Build one from a
/// preset or `PitchParams.for_method(...)`, override fields with `.replace(...)`,
/// and pass it to `sadda.dsp.voiced_pitch(audio, params=...)`.
#[pyclass(
    module = "sadda._native.pitch_preset",
    name = "PitchParams",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyPitchParams {
    pub(crate) inner: PitchParams,
}

#[pymethods]
impl PyPitchParams {
    /// Params for a named tracking method at its default (Praat/paper)
    /// configuration. `method` is autocorrelation | windowed_autocorrelation |
    /// boersma | yin | pyin | swipe.
    #[staticmethod]
    #[pyo3(signature = (method="boersma"))]
    fn for_method(method: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PitchParams {
                method: parse_method(method)?,
                config: PitchConfig::default(),
            },
        })
    }

    /// Returns a copy with the given fields overridden. `method` takes its
    /// string name; the rest are the `PitchConfig` knobs.
    #[pyo3(signature = (*, method=None, frame_size_seconds=None, hop_size_seconds=None,
        min_freq_hz=None, max_freq_hz=None, voicing_threshold=None,
        boersma_max_candidates=None, boersma_silence_threshold=None, boersma_octave_cost=None,
        boersma_octave_jump_cost=None, boersma_voiced_unvoiced_cost=None,
        yin_threshold=None, pyin_n_thresholds=None, pyin_transition_semitone_cost=None,
        pyin_voiced_unvoiced_cost=None, pyin_bins_per_semitone=None))]
    #[allow(clippy::too_many_arguments)]
    fn replace(
        &self,
        method: Option<String>,
        frame_size_seconds: Option<f32>,
        hop_size_seconds: Option<f32>,
        min_freq_hz: Option<f32>,
        max_freq_hz: Option<f32>,
        voicing_threshold: Option<f32>,
        boersma_max_candidates: Option<usize>,
        boersma_silence_threshold: Option<f32>,
        boersma_octave_cost: Option<f32>,
        boersma_octave_jump_cost: Option<f32>,
        boersma_voiced_unvoiced_cost: Option<f32>,
        yin_threshold: Option<f32>,
        pyin_n_thresholds: Option<usize>,
        pyin_transition_semitone_cost: Option<f32>,
        pyin_voiced_unvoiced_cost: Option<f32>,
        pyin_bins_per_semitone: Option<usize>,
    ) -> PyResult<Self> {
        let mut p = self.inner;
        if let Some(s) = method {
            p.method = parse_method(&s)?;
        }
        let c = &mut p.config;
        if let Some(v) = frame_size_seconds {
            c.frame_size_seconds = v;
        }
        if let Some(v) = hop_size_seconds {
            c.hop_size_seconds = v;
        }
        if let Some(v) = min_freq_hz {
            c.min_freq_hz = v;
        }
        if let Some(v) = max_freq_hz {
            c.max_freq_hz = v;
        }
        if let Some(v) = voicing_threshold {
            c.voicing_threshold = v;
        }
        if let Some(v) = boersma_max_candidates {
            c.boersma_max_candidates = v;
        }
        if let Some(v) = boersma_silence_threshold {
            c.boersma_silence_threshold = v;
        }
        if let Some(v) = boersma_octave_cost {
            c.boersma_octave_cost = v;
        }
        if let Some(v) = boersma_octave_jump_cost {
            c.boersma_octave_jump_cost = v;
        }
        if let Some(v) = boersma_voiced_unvoiced_cost {
            c.boersma_voiced_unvoiced_cost = v;
        }
        if let Some(v) = yin_threshold {
            c.yin_threshold = v;
        }
        if let Some(v) = pyin_n_thresholds {
            c.pyin_n_thresholds = v;
        }
        if let Some(v) = pyin_transition_semitone_cost {
            c.pyin_transition_semitone_cost = v;
        }
        if let Some(v) = pyin_voiced_unvoiced_cost {
            c.pyin_voiced_unvoiced_cost = v;
        }
        if let Some(v) = pyin_bins_per_semitone {
            c.pyin_bins_per_semitone = v;
        }
        Ok(Self { inner: p })
    }

    #[getter]
    fn method(&self) -> &'static str {
        method_str(self.inner.method)
    }
    #[getter]
    fn frame_size_seconds(&self) -> f32 {
        self.inner.config.frame_size_seconds
    }
    #[getter]
    fn hop_size_seconds(&self) -> f32 {
        self.inner.config.hop_size_seconds
    }
    #[getter]
    fn min_freq_hz(&self) -> f32 {
        self.inner.config.min_freq_hz
    }
    #[getter]
    fn max_freq_hz(&self) -> f32 {
        self.inner.config.max_freq_hz
    }
    #[getter]
    fn voicing_threshold(&self) -> f32 {
        self.inner.config.voicing_threshold
    }
    #[getter]
    fn boersma_max_candidates(&self) -> usize {
        self.inner.config.boersma_max_candidates
    }
    #[getter]
    fn boersma_silence_threshold(&self) -> f32 {
        self.inner.config.boersma_silence_threshold
    }
    #[getter]
    fn boersma_octave_cost(&self) -> f32 {
        self.inner.config.boersma_octave_cost
    }
    #[getter]
    fn boersma_octave_jump_cost(&self) -> f32 {
        self.inner.config.boersma_octave_jump_cost
    }
    #[getter]
    fn boersma_voiced_unvoiced_cost(&self) -> f32 {
        self.inner.config.boersma_voiced_unvoiced_cost
    }
    #[getter]
    fn yin_threshold(&self) -> f32 {
        self.inner.config.yin_threshold
    }
    #[getter]
    fn pyin_n_thresholds(&self) -> usize {
        self.inner.config.pyin_n_thresholds
    }
    #[getter]
    fn pyin_transition_semitone_cost(&self) -> f32 {
        self.inner.config.pyin_transition_semitone_cost
    }
    #[getter]
    fn pyin_voiced_unvoiced_cost(&self) -> f32 {
        self.inner.config.pyin_voiced_unvoiced_cost
    }
    #[getter]
    fn pyin_bins_per_semitone(&self) -> usize {
        self.inner.config.pyin_bins_per_semitone
    }

    /// The full spec as a TOML string (the on-disk preset `[params]` body).
    fn to_toml(&self) -> PyResult<String> {
        toml::to_string_pretty(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("cannot serialize params: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "PitchParams(method={:?}, min_freq_hz={}, max_freq_hz={})",
            method_str(self.inner.method),
            self.inner.config.min_freq_hz,
            self.inner.config.max_freq_hz,
        )
    }
}

// ---- PitchPreset -----------------------------------------------------------

/// A named pitch preset: parameters plus provenance.
#[pyclass(module = "sadda._native.pitch_preset", name = "PitchPreset")]
pub(crate) struct PyPitchPreset {
    inner: PitchPreset,
}

#[pymethods]
impl PyPitchPreset {
    /// Builds a preset from params + metadata (for saving a user preset).
    /// `based_on` is a free-text lineage label (e.g. "praat", "custom").
    #[new]
    #[pyo3(signature = (id, params, *, version="1.0.0".to_string(), title="".to_string(),
        description="".to_string(), based_on="custom".to_string(), faithful=false, reference=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: String,
        params: PyPitchParams,
        version: String,
        title: String,
        description: String,
        based_on: String,
        faithful: bool,
        reference: Option<String>,
    ) -> Self {
        Self {
            inner: PitchPreset {
                id,
                version,
                title,
                description,
                based_on,
                faithful,
                reference,
                params: params.inner,
            },
        }
    }

    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }
    #[getter]
    fn version(&self) -> &str {
        &self.inner.version
    }
    #[getter]
    fn title(&self) -> &str {
        &self.inner.title
    }
    #[getter]
    fn description(&self) -> &str {
        &self.inner.description
    }
    #[getter]
    fn based_on(&self) -> &str {
        &self.inner.based_on
    }
    #[getter]
    fn faithful(&self) -> bool {
        self.inner.faithful
    }
    #[getter]
    fn reference(&self) -> Option<&str> {
        self.inner.reference.as_deref()
    }
    #[getter]
    fn params(&self) -> PyPitchParams {
        PyPitchParams {
            inner: self.inner.params,
        }
    }

    fn to_toml(&self) -> PyResult<String> {
        self.inner.to_toml().map_err(engine_err_to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "PitchPreset(id={:?}, based_on={:?}, faithful={})",
            self.inner.id, self.inner.based_on, self.inner.faithful
        )
    }
}

// ---- store functions -------------------------------------------------------

fn store_for(root: Option<String>) -> PyResult<PitchPresetStore> {
    match root {
        Some(r) => Ok(PitchPresetStore::new(r)),
        None => PitchPresetStore::user_default().map_err(engine_err_to_py),
    }
}

/// Filesystem path of the active pitch preset store (default:
/// `~/.local/share/sadda/presets/pitch/` or the platform equivalent).
#[pyfunction]
#[pyo3(signature = (*, root=None))]
pub(crate) fn store_root(root: Option<String>) -> PyResult<String> {
    Ok(store_for(root)?.root().to_string_lossy().into_owned())
}

/// The built-in authoritative pitch presets (praat-ac / yin / pyin / swipe).
#[pyfunction]
pub(crate) fn builtin() -> Vec<PyPitchPreset> {
    pitch_builtin_presets()
        .into_iter()
        .map(|inner| PyPitchPreset { inner })
        .collect()
}

/// All presets: the built-ins followed by the user's on-disk presets.
#[pyfunction]
#[pyo3(signature = (*, root=None))]
pub(crate) fn list_all(root: Option<String>) -> PyResult<Vec<PyPitchPreset>> {
    Ok(store_for(root)?
        .list()
        .into_iter()
        .map(|inner| PyPitchPreset { inner })
        .collect())
}

/// The user's on-disk presets only (no built-ins).
#[pyfunction]
#[pyo3(signature = (*, root=None))]
pub(crate) fn list_user(root: Option<String>) -> PyResult<Vec<PyPitchPreset>> {
    Ok(store_for(root)?
        .list_user()
        .into_iter()
        .map(|inner| PyPitchPreset { inner })
        .collect())
}

/// The preset with this `id` (built-in or on-disk), or `None`.
#[pyfunction]
#[pyo3(signature = (id, *, root=None))]
pub(crate) fn get(id: &str, root: Option<String>) -> PyResult<Option<PyPitchPreset>> {
    Ok(store_for(root)?
        .get(id)
        .map(|inner| PyPitchPreset { inner }))
}

/// Saves a user preset; returns the file path. Errors if the id is invalid or
/// collides with a built-in.
#[pyfunction]
#[pyo3(signature = (preset, *, root=None))]
pub(crate) fn save(preset: &PyPitchPreset, root: Option<String>) -> PyResult<String> {
    let path = store_for(root)?
        .save(&preset.inner)
        .map_err(engine_err_to_py)?;
    Ok(path.to_string_lossy().into_owned())
}

/// Deletes the user preset with this `id`. Returns `True` if a file was removed.
#[pyfunction]
#[pyo3(signature = (id, *, root=None))]
pub(crate) fn delete(id: &str, root: Option<String>) -> PyResult<bool> {
    store_for(root)?.delete(id).map_err(engine_err_to_py)
}

/// Runs the tracker named by `params` and returns `(times, frequencies,
/// voicing)` — the same three NumPy arrays as `sadda.dsp.voiced_pitch`.
#[pyfunction]
#[allow(clippy::type_complexity)] // (times, freqs, voicing) — matches voiced_pitch
pub(crate) fn compute<'py>(
    py: Python<'py>,
    audio: &PyAudio,
    params: &PyPitchParams,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let frames = pitch_with_params(&audio.inner, &params.inner);
    let times: Vec<f64> = frames.iter().map(|f| f.time_seconds).collect();
    let freqs: Vec<f32> = frames.iter().map(|f| f.frequency_hz.value()).collect();
    let voicing: Vec<f32> = frames.iter().map(|f| f.voicing).collect();
    Ok((
        times.into_pyarray(py),
        freqs.into_pyarray(py),
        voicing.into_pyarray(py),
    ))
}
