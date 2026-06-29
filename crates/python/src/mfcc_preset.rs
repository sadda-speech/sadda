//! PyO3 bindings for the on-disk MFCC preset registry (roadmap item 3/4).
//! Thin wrappers over `sadda_engine::dsp::preset`: build [`MfccParams`] from a
//! reference preset, override individual parameters, browse the built-in and
//! user presets, and save/delete user presets. The `sadda/dsp/__init__.py`
//! re-exports these under `sadda.dsp.*` (the preset surface is PROVISIONAL).
//!
//! Conventions follow the sibling registries (`refdist.rs` / `ml.rs`): a
//! `store_for(root)` helper, free functions taking `*, root=None`, and result
//! pyclasses. Like those submodules it is intentionally left out of the
//! generated `.pyi` (no `gen_stub` attrs).

use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sadda_engine::dsp::mfcc::{
    MelScaleKind, MfccDct, MfccFft, MfccFilterNorm, MfccFilters, MfccFraming, MfccParams,
    MfccPowerNorm, MfccWindow, mfcc_with_params,
};
use sadda_engine::dsp::preset::{MfccPreset, MfccPresetStore, PresetLineage, builtin_presets};

use crate::{PyAudio, engine_err_to_py};

// ---- enum <-> string helpers (the snake_case TOML names) -------------------

fn window_str(w: MfccWindow) -> &'static str {
    match w {
        MfccWindow::PeriodicHann => "periodic_hann",
        MfccWindow::Povey => "povey",
        MfccWindow::PraatGaussian => "praat_gaussian",
        MfccWindow::Hamming => "hamming",
    }
}
fn parse_window(s: &str) -> PyResult<MfccWindow> {
    match s {
        "periodic_hann" => Ok(MfccWindow::PeriodicHann),
        "povey" => Ok(MfccWindow::Povey),
        "praat_gaussian" => Ok(MfccWindow::PraatGaussian),
        "hamming" => Ok(MfccWindow::Hamming),
        other => Err(PyValueError::new_err(format!(
            "unknown window {other:?}; expected periodic_hann | povey | praat_gaussian | hamming"
        ))),
    }
}

fn framing_str(f: MfccFraming) -> &'static str {
    match f {
        MfccFraming::Centered => "centered",
        MfccFraming::SnipEdges => "snip_edges",
    }
}
fn parse_framing(s: &str) -> PyResult<MfccFraming> {
    match s {
        "centered" => Ok(MfccFraming::Centered),
        "snip_edges" => Ok(MfccFraming::SnipEdges),
        other => Err(PyValueError::new_err(format!(
            "unknown framing {other:?}; expected centered | snip_edges"
        ))),
    }
}

fn fft_str(f: MfccFft) -> &'static str {
    match f {
        MfccFft::WindowLength => "window_length",
        MfccFft::NextPow2 => "next_pow2",
    }
}
fn parse_fft(s: &str) -> PyResult<MfccFft> {
    match s {
        "window_length" => Ok(MfccFft::WindowLength),
        "next_pow2" => Ok(MfccFft::NextPow2),
        other => Err(PyValueError::new_err(format!(
            "unknown fft rule {other:?}; expected window_length | next_pow2"
        ))),
    }
}

fn mel_scale_str(m: MelScaleKind) -> &'static str {
    match m {
        MelScaleKind::Slaney => "slaney",
        MelScaleKind::Htk => "htk",
    }
}
fn parse_mel_scale(s: &str) -> PyResult<MelScaleKind> {
    match s {
        "slaney" => Ok(MelScaleKind::Slaney),
        "htk" => Ok(MelScaleKind::Htk),
        other => Err(PyValueError::new_err(format!(
            "unknown mel scale {other:?}; expected slaney | htk"
        ))),
    }
}

fn filter_norm_str(f: MfccFilterNorm) -> &'static str {
    match f {
        MfccFilterNorm::AreaSlaney => "area_slaney",
        MfccFilterNorm::UnitPeak => "unit_peak",
    }
}
fn parse_filter_norm(s: &str) -> PyResult<MfccFilterNorm> {
    match s {
        "area_slaney" => Ok(MfccFilterNorm::AreaSlaney),
        "unit_peak" => Ok(MfccFilterNorm::UnitPeak),
        other => Err(PyValueError::new_err(format!(
            "unknown filter norm {other:?}; expected area_slaney | unit_peak"
        ))),
    }
}

fn dct_str(d: MfccDct) -> &'static str {
    match d {
        MfccDct::Ortho => "ortho",
        MfccDct::Unnormalized => "unnormalized",
    }
}
fn parse_dct(s: &str) -> PyResult<MfccDct> {
    match s {
        "ortho" => Ok(MfccDct::Ortho),
        "unnormalized" => Ok(MfccDct::Unnormalized),
        other => Err(PyValueError::new_err(format!(
            "unknown dct norm {other:?}; expected ortho | unnormalized"
        ))),
    }
}

fn power_norm_str(p: MfccPowerNorm) -> &'static str {
    match p {
        MfccPowerNorm::Raw => "raw",
        MfccPowerNorm::PraatDuration => "praat_duration",
    }
}
fn parse_power_norm(s: &str) -> PyResult<MfccPowerNorm> {
    match s {
        "raw" => Ok(MfccPowerNorm::Raw),
        "praat_duration" => Ok(MfccPowerNorm::PraatDuration),
        other => Err(PyValueError::new_err(format!(
            "unknown power norm {other:?}; expected raw | praat_duration"
        ))),
    }
}

fn lineage_str(l: PresetLineage) -> &'static str {
    match l {
        PresetLineage::Librosa => "librosa",
        PresetLineage::Kaldi => "kaldi",
        PresetLineage::Praat => "praat",
        PresetLineage::Htk => "htk",
        PresetLineage::Custom => "custom",
    }
}
fn parse_lineage(s: &str) -> PyResult<PresetLineage> {
    match s {
        "librosa" => Ok(PresetLineage::Librosa),
        "kaldi" => Ok(PresetLineage::Kaldi),
        "praat" => Ok(PresetLineage::Praat),
        "htk" => Ok(PresetLineage::Htk),
        "custom" => Ok(PresetLineage::Custom),
        other => Err(PyValueError::new_err(format!(
            "unknown lineage {other:?}; expected librosa | kaldi | praat | htk | custom"
        ))),
    }
}

// ---- MfccParams ------------------------------------------------------------

/// A full MFCC parameter set. Build one from a reference preset
/// (`MfccParams.librosa(...)` / `.kaldi(...)` / `.praat(...)`) and override
/// individual fields with `.replace(...)`, or load one from a stored preset.
/// Pass it to `sadda.dsp.mfcc(audio, params=...)`.
#[pyclass(
    module = "sadda._native.mfcc_preset",
    name = "MfccParams",
    frozen,
    from_py_object
)]
#[derive(Clone)]
pub(crate) struct PyMfccParams {
    pub(crate) inner: MfccParams,
}

#[pymethods]
impl PyMfccParams {
    /// Faithful `librosa.feature.mfcc` (0.11) parameters.
    #[staticmethod]
    #[pyo3(signature = (frame_size_seconds=0.025, hop_seconds=0.010, n_mels=40, n_mfcc=13, f_min=0.0, f_max=8000.0))]
    #[allow(clippy::too_many_arguments)]
    fn librosa(
        frame_size_seconds: f32,
        hop_seconds: f32,
        n_mels: usize,
        n_mfcc: usize,
        f_min: f32,
        f_max: f32,
    ) -> Self {
        Self {
            inner: MfccParams::librosa(
                frame_size_seconds,
                hop_seconds,
                n_mels,
                n_mfcc,
                f_min,
                f_max,
            ),
        }
    }

    /// Faithful Kaldi `compute-mfcc-feats` parameters.
    #[staticmethod]
    #[pyo3(signature = (frame_size_seconds=0.025, hop_seconds=0.010, n_mels=23, n_mfcc=13, f_min=20.0, f_max=8000.0))]
    #[allow(clippy::too_many_arguments)]
    fn kaldi(
        frame_size_seconds: f32,
        hop_seconds: f32,
        n_mels: usize,
        n_mfcc: usize,
        f_min: f32,
        f_max: f32,
    ) -> Self {
        Self {
            inner: MfccParams::kaldi(
                frame_size_seconds,
                hop_seconds,
                n_mels,
                n_mfcc,
                f_min,
                f_max,
            ),
        }
    }

    /// Praat `Sound: To MFCC…` parameters. Through the shared pipeline this is
    /// an approximation — for faithful Praat output use
    /// `sadda.dsp.mfcc(audio, method="praat")`.
    #[staticmethod]
    #[pyo3(signature = (frame_size_seconds=0.025, hop_seconds=0.010, n_mfcc=13, f_max=8000.0))]
    fn praat(frame_size_seconds: f32, hop_seconds: f32, n_mfcc: usize, f_max: f32) -> Self {
        Self {
            inner: MfccParams::praat(frame_size_seconds, hop_seconds, n_mfcc, f_max),
        }
    }

    /// Returns a copy with the given fields overridden (the "pick a preset,
    /// then modify individual parameters" path). Enum fields take their
    /// snake_case names (e.g. `window="hamming"`, `mel_scale="htk"`).
    /// `n_mels` applies only when the preset uses the n-mels filterbank.
    #[pyo3(signature = (*, frame_size_seconds=None, hop_seconds=None, n_mfcc=None, n_mels=None,
        f_min=None, f_max=None, window_duration_factor=None, pre_emphasis=None, lifter=None,
        remove_dc=None, triangle_in_mel=None, exclude_nyquist_bin=None,
        window=None, framing=None, fft=None, mel_scale=None, filter_norm=None, dct=None, power_norm=None))]
    #[allow(clippy::too_many_arguments)]
    fn replace(
        &self,
        frame_size_seconds: Option<f32>,
        hop_seconds: Option<f32>,
        n_mfcc: Option<usize>,
        n_mels: Option<usize>,
        f_min: Option<f32>,
        f_max: Option<f32>,
        window_duration_factor: Option<f32>,
        pre_emphasis: Option<f32>,
        lifter: Option<f32>,
        remove_dc: Option<bool>,
        triangle_in_mel: Option<bool>,
        exclude_nyquist_bin: Option<bool>,
        window: Option<String>,
        framing: Option<String>,
        fft: Option<String>,
        mel_scale: Option<String>,
        filter_norm: Option<String>,
        dct: Option<String>,
        power_norm: Option<String>,
    ) -> PyResult<Self> {
        let mut p = self.inner.clone();
        if let Some(v) = frame_size_seconds {
            p.frame_size_seconds = v;
        }
        if let Some(v) = hop_seconds {
            p.hop_seconds = v;
        }
        if let Some(v) = n_mfcc {
            p.n_mfcc = v;
        }
        if let Some(v) = f_min {
            p.f_min = v;
        }
        if let Some(v) = f_max {
            p.f_max = v;
        }
        if let Some(v) = window_duration_factor {
            p.window_duration_factor = v;
        }
        if let Some(v) = pre_emphasis {
            p.pre_emphasis = v;
        }
        if let Some(v) = lifter {
            p.lifter = v;
        }
        if let Some(v) = remove_dc {
            p.remove_dc = v;
        }
        if let Some(v) = triangle_in_mel {
            p.triangle_in_mel = v;
        }
        if let Some(v) = exclude_nyquist_bin {
            p.exclude_nyquist_bin = v;
        }
        if let Some(s) = window {
            p.window = parse_window(&s)?;
        }
        if let Some(s) = framing {
            p.framing = parse_framing(&s)?;
        }
        if let Some(s) = fft {
            p.fft = parse_fft(&s)?;
        }
        if let Some(s) = mel_scale {
            p.mel_scale = parse_mel_scale(&s)?;
        }
        if let Some(s) = filter_norm {
            p.filter_norm = parse_filter_norm(&s)?;
        }
        if let Some(s) = dct {
            p.dct = parse_dct(&s)?;
        }
        if let Some(s) = power_norm {
            p.power_norm = parse_power_norm(&s)?;
        }
        if let Some(n) = n_mels {
            match &mut p.filters {
                MfccFilters::NMels { n_mels } => *n_mels = n,
                MfccFilters::MelSpacing { .. } => {
                    return Err(PyValueError::new_err(
                        "n_mels override applies only to presets using the n-mels filterbank \
                         (this preset uses mel-spacing)",
                    ));
                }
            }
        }
        Ok(Self { inner: p })
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
    fn n_mfcc(&self) -> usize {
        self.inner.n_mfcc
    }
    /// Number of mel filters, or `None` if the filterbank is mel-spacing-based.
    #[getter]
    fn n_mels(&self) -> Option<usize> {
        match self.inner.filters {
            MfccFilters::NMels { n_mels } => Some(n_mels),
            MfccFilters::MelSpacing { .. } => None,
        }
    }
    #[getter]
    fn f_min(&self) -> f32 {
        self.inner.f_min
    }
    #[getter]
    fn f_max(&self) -> f32 {
        self.inner.f_max
    }
    #[getter]
    fn window_duration_factor(&self) -> f32 {
        self.inner.window_duration_factor
    }
    #[getter]
    fn pre_emphasis(&self) -> f32 {
        self.inner.pre_emphasis
    }
    #[getter]
    fn lifter(&self) -> f32 {
        self.inner.lifter
    }
    #[getter]
    fn remove_dc(&self) -> bool {
        self.inner.remove_dc
    }
    #[getter]
    fn triangle_in_mel(&self) -> bool {
        self.inner.triangle_in_mel
    }
    #[getter]
    fn exclude_nyquist_bin(&self) -> bool {
        self.inner.exclude_nyquist_bin
    }
    #[getter]
    fn window(&self) -> &'static str {
        window_str(self.inner.window)
    }
    #[getter]
    fn framing(&self) -> &'static str {
        framing_str(self.inner.framing)
    }
    #[getter]
    fn fft(&self) -> &'static str {
        fft_str(self.inner.fft)
    }
    #[getter]
    fn mel_scale(&self) -> &'static str {
        mel_scale_str(self.inner.mel_scale)
    }
    #[getter]
    fn filter_norm(&self) -> &'static str {
        filter_norm_str(self.inner.filter_norm)
    }
    #[getter]
    fn dct(&self) -> &'static str {
        dct_str(self.inner.dct)
    }
    #[getter]
    fn power_norm(&self) -> &'static str {
        power_norm_str(self.inner.power_norm)
    }

    /// The full parameter set as a TOML string (the on-disk preset `[params]`
    /// table body) — the escape hatch for fields without a dedicated getter.
    fn to_toml(&self) -> PyResult<String> {
        toml::to_string_pretty(&self.inner)
            .map_err(|e| PyValueError::new_err(format!("cannot serialize params: {e}")))
    }

    fn __repr__(&self) -> String {
        format!(
            "MfccParams(n_mfcc={}, window={:?}, mel_scale={:?}, dct={:?})",
            self.inner.n_mfcc,
            window_str(self.inner.window),
            mel_scale_str(self.inner.mel_scale),
            dct_str(self.inner.dct),
        )
    }
}

// ---- MfccPreset ------------------------------------------------------------

/// A named MFCC preset: parameters plus provenance (which reference it derives
/// from, whether it's a faithful reproduction, a citation).
#[pyclass(module = "sadda._native.mfcc_preset", name = "MfccPreset")]
pub(crate) struct PyMfccPreset {
    inner: MfccPreset,
}

#[pymethods]
impl PyMfccPreset {
    /// Builds a preset from params + metadata (for saving a user preset).
    /// `based_on` is one of librosa | kaldi | praat | htk | custom.
    #[new]
    #[pyo3(signature = (id, params, *, version="1.0.0".to_string(), title="".to_string(),
        description="".to_string(), based_on="custom".to_string(), faithful=false, reference=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: String,
        params: PyMfccParams,
        version: String,
        title: String,
        description: String,
        based_on: String,
        faithful: bool,
        reference: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: MfccPreset {
                id,
                version,
                title,
                description,
                based_on: parse_lineage(&based_on)?,
                faithful,
                reference,
                params: params.inner,
            },
        })
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
    /// The reference this preset derives from: librosa | kaldi | praat | htk |
    /// custom.
    #[getter]
    fn based_on(&self) -> &'static str {
        lineage_str(self.inner.based_on)
    }
    /// Whether running this preset through the pipeline reproduces its
    /// reference to tolerance. (The Praat built-in is `False` — its pipeline
    /// path is f32-approximate; use `mfcc(method="praat")` for faithful Praat.)
    #[getter]
    fn faithful(&self) -> bool {
        self.inner.faithful
    }
    #[getter]
    fn reference(&self) -> Option<&str> {
        self.inner.reference.as_deref()
    }
    /// The parameter set.
    #[getter]
    fn params(&self) -> PyMfccParams {
        PyMfccParams {
            inner: self.inner.params.clone(),
        }
    }

    /// The whole preset as a TOML string (the on-disk `<id>.toml` form).
    fn to_toml(&self) -> PyResult<String> {
        self.inner.to_toml().map_err(engine_err_to_py)
    }

    fn __repr__(&self) -> String {
        format!(
            "MfccPreset(id={:?}, based_on={:?}, faithful={})",
            self.inner.id,
            lineage_str(self.inner.based_on),
            self.inner.faithful
        )
    }
}

// ---- store functions -------------------------------------------------------

fn store_for(root: Option<String>) -> PyResult<MfccPresetStore> {
    match root {
        Some(r) => Ok(MfccPresetStore::new(r)),
        None => MfccPresetStore::user_default().map_err(engine_err_to_py),
    }
}

/// Filesystem path of the active preset store (default:
/// `~/.local/share/sadda/presets/mfcc/` or the platform equivalent).
#[pyfunction]
#[pyo3(signature = (*, root=None))]
pub(crate) fn store_root(root: Option<String>) -> PyResult<String> {
    Ok(store_for(root)?.root().to_string_lossy().into_owned())
}

/// The built-in authoritative presets (librosa / kaldi / praat).
#[pyfunction]
pub(crate) fn builtin() -> Vec<PyMfccPreset> {
    builtin_presets()
        .into_iter()
        .map(|inner| PyMfccPreset { inner })
        .collect()
}

/// All presets: the built-ins followed by the user's on-disk presets.
#[pyfunction]
#[pyo3(signature = (*, root=None))]
pub(crate) fn list_all(root: Option<String>) -> PyResult<Vec<PyMfccPreset>> {
    Ok(store_for(root)?
        .list()
        .into_iter()
        .map(|inner| PyMfccPreset { inner })
        .collect())
}

/// The user's on-disk presets only (no built-ins).
#[pyfunction]
#[pyo3(signature = (*, root=None))]
pub(crate) fn list_user(root: Option<String>) -> PyResult<Vec<PyMfccPreset>> {
    Ok(store_for(root)?
        .list_user()
        .into_iter()
        .map(|inner| PyMfccPreset { inner })
        .collect())
}

/// The preset with this `id` (built-in or on-disk), or `None`.
#[pyfunction]
#[pyo3(signature = (id, *, root=None))]
pub(crate) fn get(id: &str, root: Option<String>) -> PyResult<Option<PyMfccPreset>> {
    Ok(store_for(root)?.get(id).map(|inner| PyMfccPreset { inner }))
}

/// Saves a user preset to the store; returns the file path. Errors if the id
/// is invalid or collides with a built-in.
#[pyfunction]
#[pyo3(signature = (preset, *, root=None))]
pub(crate) fn save(preset: &PyMfccPreset, root: Option<String>) -> PyResult<String> {
    let path = store_for(root)?
        .save(&preset.inner)
        .map_err(engine_err_to_py)?;
    Ok(path.to_string_lossy().into_owned())
}

/// Deletes the user preset with this `id`. Returns `True` if a file was
/// removed.
#[pyfunction]
#[pyo3(signature = (id, *, root=None))]
pub(crate) fn delete(id: &str, root: Option<String>) -> PyResult<bool> {
    store_for(root)?.delete(id).map_err(engine_err_to_py)
}

/// Computes MFCCs from explicit `params`, shape `(n_frames, n_mfcc)`. The
/// engine `mfcc_with_params` pipeline behind `sadda.dsp.mfcc(audio, params=…)`.
#[pyfunction]
pub(crate) fn compute<'py>(
    py: Python<'py>,
    audio: &PyAudio,
    params: &PyMfccParams,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mono: Vec<f32> = audio.inner.mono_samples().collect();
    let arr = mfcc_with_params(&mono, audio.inner.sample_rate, &params.inner);
    Ok(arr.into_pyarray(py))
}
