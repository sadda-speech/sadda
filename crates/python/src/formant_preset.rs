//! PyO3 bindings for the on-disk formant preset registry (roadmap item 6).
//! Mirrors `mfcc_preset.rs` / `pitch_preset.rs`. `FormantsConfig` already
//! bundles the LPC method, so it *is* the preset payload (no wrapper). Re-
//! exported under `sadda.dsp.*` (PROVISIONAL). Unstubbed, per convention.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sadda_engine::dsp::formant_preset::{
    FormantPreset, FormantPresetStore, formant_builtin_presets,
};
use sadda_engine::dsp::lpc::LpcMethod;
use sadda_engine::dsp::{FormantsConfig, formants};

use crate::{PyAudio, PyFormantFrame, engine_err_to_py};

fn method_str(m: LpcMethod) -> &'static str {
    match m {
        LpcMethod::Autocorrelation => "autocorrelation",
        LpcMethod::Burg => "burg",
    }
}
fn parse_method(s: &str) -> PyResult<LpcMethod> {
    match s {
        "autocorrelation" => Ok(LpcMethod::Autocorrelation),
        "burg" => Ok(LpcMethod::Burg),
        other => Err(PyValueError::new_err(format!(
            "unknown LPC method {other:?}; expected autocorrelation | burg"
        ))),
    }
}

// ---- FormantsParams (a FormantsConfig) -------------------------------------

/// A full formant-tracking spec (LPC method + analysis config). Build one from
/// a preset or `FormantsParams.for_method(...)`, override fields with
/// `.replace(...)`, and pass it to `sadda.dsp.formants(audio, params=...)`.
#[pyclass(
    module = "sadda._native.formant_preset",
    name = "FormantsParams",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyFormantsParams {
    pub(crate) inner: FormantsConfig,
}

/// Builds a `FormantsConfig` for `lpc_method` with the common analysis
/// settings (`lpc_order` left auto). Backs the named constructors below.
#[allow(clippy::too_many_arguments)]
fn formants_config_with(
    lpc_method: LpcMethod,
    frame_size_seconds: f32,
    hop_seconds: f32,
    n_formants: usize,
    pre_emphasis: f32,
    max_bandwidth_hz: f32,
    min_frequency_hz: f32,
) -> PyFormantsParams {
    PyFormantsParams {
        inner: FormantsConfig {
            frame_size_seconds,
            hop_seconds,
            n_formants,
            pre_emphasis,
            lpc_order: None,
            lpc_method,
            max_bandwidth_hz,
            min_frequency_hz,
        },
    }
}

#[pymethods]
impl PyFormantsParams {
    /// Params for a named LPC method at the default config. `method` is
    /// `burg` (default) or `autocorrelation`.
    #[staticmethod]
    #[pyo3(signature = (method="burg"))]
    fn for_method(method: &str) -> PyResult<Self> {
        Ok(Self {
            inner: FormantsConfig {
                lpc_method: parse_method(method)?,
                ..FormantsConfig::default()
            },
        })
    }

    /// Praat `Sound.to_formant_burg` — Burg LPC, the default. (Named
    /// constructor parallel to `MfccParams.librosa`; `lpc_order` stays auto.)
    #[staticmethod]
    #[pyo3(signature = (frame_size_seconds=0.025, hop_seconds=0.010, n_formants=5, pre_emphasis=0.97, max_bandwidth_hz=1000.0, min_frequency_hz=50.0))]
    fn burg(
        frame_size_seconds: f32,
        hop_seconds: f32,
        n_formants: usize,
        pre_emphasis: f32,
        max_bandwidth_hz: f32,
        min_frequency_hz: f32,
    ) -> Self {
        formants_config_with(
            LpcMethod::Burg,
            frame_size_seconds,
            hop_seconds,
            n_formants,
            pre_emphasis,
            max_bandwidth_hz,
            min_frequency_hz,
        )
    }

    /// Autocorrelation-method LPC (Levinson–Durbin).
    #[staticmethod]
    #[pyo3(signature = (frame_size_seconds=0.025, hop_seconds=0.010, n_formants=5, pre_emphasis=0.97, max_bandwidth_hz=1000.0, min_frequency_hz=50.0))]
    fn autocorrelation(
        frame_size_seconds: f32,
        hop_seconds: f32,
        n_formants: usize,
        pre_emphasis: f32,
        max_bandwidth_hz: f32,
        min_frequency_hz: f32,
    ) -> Self {
        formants_config_with(
            LpcMethod::Autocorrelation,
            frame_size_seconds,
            hop_seconds,
            n_formants,
            pre_emphasis,
            max_bandwidth_hz,
            min_frequency_hz,
        )
    }

    /// Returns a copy with the given fields overridden. `method` takes its
    /// string name. (`lpc_order` keeps its default — auto = `2·n_formants + 2`
    /// when unset; it's a rare advanced knob and not exposed here.)
    #[pyo3(signature = (*, method=None, frame_size_seconds=None, hop_seconds=None,
        n_formants=None, pre_emphasis=None, max_bandwidth_hz=None, min_frequency_hz=None))]
    #[allow(clippy::too_many_arguments)]
    fn replace(
        &self,
        method: Option<String>,
        frame_size_seconds: Option<f32>,
        hop_seconds: Option<f32>,
        n_formants: Option<usize>,
        pre_emphasis: Option<f32>,
        max_bandwidth_hz: Option<f32>,
        min_frequency_hz: Option<f32>,
    ) -> PyResult<Self> {
        let mut c = self.inner;
        if let Some(s) = method {
            c.lpc_method = parse_method(&s)?;
        }
        if let Some(v) = frame_size_seconds {
            c.frame_size_seconds = v;
        }
        if let Some(v) = hop_seconds {
            c.hop_seconds = v;
        }
        if let Some(v) = n_formants {
            c.n_formants = v;
        }
        if let Some(v) = pre_emphasis {
            c.pre_emphasis = v;
        }
        if let Some(v) = max_bandwidth_hz {
            c.max_bandwidth_hz = v;
        }
        if let Some(v) = min_frequency_hz {
            c.min_frequency_hz = v;
        }
        Ok(Self { inner: c })
    }

    #[getter]
    fn method(&self) -> &'static str {
        method_str(self.inner.lpc_method)
    }
    #[getter]
    fn frame_size_seconds(&self) -> f32 {
        self.inner.frame_size_seconds
    }
    #[getter]
    fn hop_seconds(&self) -> f32 {
        self.inner.hop_seconds
    }
    #[getter]
    fn n_formants(&self) -> usize {
        self.inner.n_formants
    }
    #[getter]
    fn pre_emphasis(&self) -> f32 {
        self.inner.pre_emphasis
    }
    #[getter]
    fn lpc_order(&self) -> Option<usize> {
        self.inner.lpc_order
    }
    #[getter]
    fn max_bandwidth_hz(&self) -> f32 {
        self.inner.max_bandwidth_hz
    }
    #[getter]
    fn min_frequency_hz(&self) -> f32 {
        self.inner.min_frequency_hz
    }

    fn to_toml(&self) -> PyResult<String> {
        toml::to_string_pretty(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("cannot serialize params: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "FormantsParams(method={:?}, n_formants={})",
            method_str(self.inner.lpc_method),
            self.inner.n_formants,
        )
    }
}

// ---- FormantPreset ---------------------------------------------------------

/// A named formant preset: parameters plus provenance.
#[pyclass(module = "sadda._native.formant_preset", name = "FormantPreset")]
pub(crate) struct PyFormantPreset {
    inner: FormantPreset,
}

#[pymethods]
impl PyFormantPreset {
    /// Builds a preset from params + metadata. `based_on` is a free-text
    /// lineage label (e.g. "praat", "custom").
    #[new]
    #[pyo3(signature = (id, params, *, version="1.0.0".to_string(), title="".to_string(),
        description="".to_string(), based_on="custom".to_string(), faithful=false, reference=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: String,
        params: PyFormantsParams,
        version: String,
        title: String,
        description: String,
        based_on: String,
        faithful: bool,
        reference: Option<String>,
    ) -> Self {
        Self {
            inner: FormantPreset {
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
    fn params(&self) -> PyFormantsParams {
        PyFormantsParams {
            inner: self.inner.params,
        }
    }

    fn to_toml(&self) -> PyResult<String> {
        self.inner.to_toml().map_err(engine_err_to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "FormantPreset(id={:?}, based_on={:?}, faithful={})",
            self.inner.id, self.inner.based_on, self.inner.faithful
        )
    }
}

// ---- store functions -------------------------------------------------------

fn store_for(root: Option<String>) -> PyResult<FormantPresetStore> {
    match root {
        Some(r) => Ok(FormantPresetStore::new(r)),
        None => FormantPresetStore::user_default().map_err(engine_err_to_py),
    }
}

/// Filesystem path of the active formant preset store (default:
/// `~/.local/share/sadda/presets/formant/` or the platform equivalent).
#[pyfunction]
#[pyo3(signature = (*, root=None))]
pub(crate) fn store_root(root: Option<String>) -> PyResult<String> {
    Ok(store_for(root)?.root().to_string_lossy().into_owned())
}

/// The built-in authoritative formant presets (praat-burg / autocorrelation).
#[pyfunction]
pub(crate) fn builtin() -> Vec<PyFormantPreset> {
    formant_builtin_presets()
        .into_iter()
        .map(|inner| PyFormantPreset { inner })
        .collect()
}

/// All presets: the built-ins followed by the user's on-disk presets.
#[pyfunction]
#[pyo3(signature = (*, root=None))]
pub(crate) fn list_all(root: Option<String>) -> PyResult<Vec<PyFormantPreset>> {
    Ok(store_for(root)?
        .list()
        .into_iter()
        .map(|inner| PyFormantPreset { inner })
        .collect())
}

/// The user's on-disk presets only (no built-ins).
#[pyfunction]
#[pyo3(signature = (*, root=None))]
pub(crate) fn list_user(root: Option<String>) -> PyResult<Vec<PyFormantPreset>> {
    Ok(store_for(root)?
        .list_user()
        .into_iter()
        .map(|inner| PyFormantPreset { inner })
        .collect())
}

/// The preset with this `id` (built-in or on-disk), or `None`.
#[pyfunction]
#[pyo3(signature = (id, *, root=None))]
pub(crate) fn get(id: &str, root: Option<String>) -> PyResult<Option<PyFormantPreset>> {
    Ok(store_for(root)?
        .get(id)
        .map(|inner| PyFormantPreset { inner }))
}

/// Saves a user preset; returns the file path. Errors if the id is invalid or
/// collides with a built-in.
#[pyfunction]
#[pyo3(signature = (preset, *, root=None))]
pub(crate) fn save(preset: &PyFormantPreset, root: Option<String>) -> PyResult<String> {
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

/// Runs the formant tracker from `params`; returns a list of `FormantFrame`
/// — the same as `sadda.dsp.formants`.
#[pyfunction]
pub(crate) fn compute(audio: &PyAudio, params: &PyFormantsParams) -> Vec<PyFormantFrame> {
    let mono: Vec<f32> = audio.inner.mono_samples().collect();
    formants(&mono, audio.inner.sample_rate, &params.inner)
        .into_iter()
        .map(|inner| PyFormantFrame { inner })
        .collect()
}
