//! PyO3 bindings for sadda — built by maturin into `sadda._native`, the Rust
//! submodule of the [`sadda`] Python package. User-facing imports happen via
//! `python/sadda/__init__.py`, which re-exports these symbols with stability
//! decorators applied.
//!
//! Doc comments on `#[pyfunction]`, `#[pyclass]`, and `#[pymethods]` items
//! here become the corresponding Python `__doc__` strings, so they serve both
//! the Rust API reference and `help(sadda.X)` in a REPL.
#![warn(missing_docs)]

use std::path::PathBuf;

use ndarray::Array2;
use numpy::{Complex32, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

mod align;
mod formant_preset;
mod live;
mod mfcc_preset;
mod ml;
mod pitch_preset;
mod recipe;
mod refdist;

pub(crate) fn engine_err_to_py(e: sadda_engine::EngineError) -> PyErr {
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
        sadda_engine::EngineError::RefDist(msg) => {
            PyValueError::new_err(format!("reference-distribution error: {msg}"))
        }
        sadda_engine::EngineError::SchemaTooNew {
            db_version,
            engine_max,
        } => PyRuntimeError::new_err(format!(
            "corpus database schema (v{db_version}) is newer than this engine (max v{engine_max}); upgrade sadda or restore an older backup"
        )),
        sadda_engine::EngineError::Cardinality(msg) => {
            PyValueError::new_err(format!("cardinality violation: {msg}"))
        }
        sadda_engine::EngineError::ProjectLocked {
            holder_pid,
            hostname,
            lockfile_path,
        } => PyRuntimeError::new_err(format!(
            "project locked by PID {holder_pid} on {hostname}; lockfile at {}",
            lockfile_path.display()
        )),
        sadda_engine::EngineError::Unreliable { measure, reason } => {
            PyValueError::new_err(format!("measure '{measure}' unreliable: {reason}"))
        }
        sadda_engine::EngineError::Ml(msg) => PyRuntimeError::new_err(format!("ml error: {msg}")),
        sadda_engine::EngineError::Preset(msg) => {
            PyValueError::new_err(format!("preset error: {msg}"))
        }
        sadda_engine::EngineError::Align(msg) => {
            PyValueError::new_err(format!("alignment error: {msg}"))
        }
    }
}

/// Audio data loaded from disk. Samples are interleaved float32 in `[-1.0, 1.0]`;
/// for stereo the layout is `[L0, R0, L1, R1, ...]`. Construct via
/// `sadda.load_wav(path)`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Audio")]
pub(crate) struct PyAudio {
    pub(crate) inner: sadda_engine::Audio,
}

#[gen_stub_pymethods]
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

    /// Mono downmix as a new single-channel `Audio` (multi-channel frames are
    /// averaged). Returns an `Audio` — not a raw array — so it can be passed
    /// straight back into `dsp.*` functions; reach the samples via
    /// `audio.mono().samples`. Note most `dsp.*` and `clinical.*` functions
    /// already mono-mix internally, so you rarely need to call this first.
    fn mono(&self) -> PyAudio {
        PyAudio {
            inner: self.inner.to_mono(),
        }
    }

    /// Construct an `Audio` from a 1-D float32 NumPy array of interleaved samples
    /// (values in `[-1.0, 1.0]`). For stereo the layout is `[L0, R0, L1, R1, ...]`.
    #[staticmethod]
    #[pyo3(signature = (samples, sample_rate, *, channels=1))]
    fn from_samples(
        samples: PyReadonlyArray1<'_, f32>,
        sample_rate: u32,
        channels: u16,
    ) -> PyAudio {
        PyAudio {
            inner: sadda_engine::Audio::from_samples(
                samples.as_array().to_vec(),
                sample_rate,
                channels,
            ),
        }
    }

    /// Return a new `Audio` resampled to `target_hz` (Hz), preserving the
    /// channel count. Uses the engine's FFT-domain resampler — the same one the
    /// VAD path uses to reach a model's required rate — so any-rate audio can be
    /// fed to a fixed-rate model (e.g. forced alignment's 16 kHz net). A no-op
    /// copy when the rate already matches.
    fn resample(&self, target_hz: u32) -> PyAudio {
        PyAudio {
            inner: self.inner.resample_to(target_hz),
        }
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

/// One recording inside a [`Project`]: audio header plus optional Session +
/// Speaker FKs and a freeform JSON `extra` payload. Read-only view; mutate
/// via `Project.add_bundle(...)`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Bundle", frozen)]
struct PyBundle {
    inner: sadda_engine::Bundle,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyBundle {
    /// Bundle id (primary key).
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// Human-readable bundle name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    /// Audio file path relative to the project root.
    #[getter]
    fn audio_relative_path(&self) -> String {
        self.inner.audio_relative_path.clone()
    }
    /// Audio sample rate in Hz.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }
    /// Number of audio channels.
    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels
    }
    /// Number of audio frames (samples per channel).
    #[getter]
    fn n_frames(&self) -> usize {
        self.inner.n_frames
    }
    /// Optional Session id this bundle belongs to.
    #[getter]
    fn session_id(&self) -> Option<i64> {
        self.inner.session_id
    }
    /// Optional Speaker id this bundle recorded.
    #[getter]
    fn speaker_id(&self) -> Option<i64> {
        self.inner.speaker_id
    }
    /// Freeform JSON payload (stored as text).
    #[getter]
    fn extra(&self) -> Option<String> {
        self.inner.extra.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Bundle(id={}, name={:?}, sample_rate={}, channels={}, n_frames={})",
            self.inner.id,
            self.inner.name,
            self.inner.sample_rate,
            self.inner.channels,
            self.inner.n_frames,
        )
    }
}

/// A person who produced speech in the project (participant, patient, case
/// subject, …). Read-only view; create via `Project.add_speaker(...)`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Speaker", frozen)]
struct PySpeaker {
    inner: sadda_engine::Speaker,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySpeaker {
    /// Speaker id (primary key).
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// Human-readable name or pseudonymous identifier.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    /// Sex / gender label (free text).
    #[getter]
    fn sex(&self) -> Option<String> {
        self.inner.sex.clone()
    }
    /// Birth year (full integer year).
    #[getter]
    fn birth_year(&self) -> Option<i32> {
        self.inner.birth_year
    }
    /// Freeform notes.
    #[getter]
    fn notes(&self) -> Option<String> {
        self.inner.notes.clone()
    }
    /// Freeform JSON payload.
    #[getter]
    fn extra(&self) -> Option<String> {
        self.inner.extra.clone()
    }
    /// ISO 8601 UTC creation timestamp.
    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at.clone()
    }

    fn __repr__(&self) -> String {
        format!("Speaker(id={}, name={:?})", self.inner.id, self.inner.name)
    }
}

/// A recording session — a time-bounded block during which one or more
/// bundles were captured. Read-only view; create via `Project.add_session(...)`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Session", frozen)]
struct PySession {
    inner: sadda_engine::Session,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySession {
    /// Session id (primary key).
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// Human-readable session name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    /// ISO 8601 UTC start timestamp.
    #[getter]
    fn started_at(&self) -> Option<String> {
        self.inner.started_at.clone()
    }
    /// ISO 8601 UTC end timestamp.
    #[getter]
    fn ended_at(&self) -> Option<String> {
        self.inner.ended_at.clone()
    }
    /// Free-form location label.
    #[getter]
    fn location(&self) -> Option<String> {
        self.inner.location.clone()
    }
    /// FK into the instrument table.
    #[getter]
    fn instrument_id(&self) -> Option<i64> {
        self.inner.instrument_id
    }
    /// FK into the protocol table.
    #[getter]
    fn protocol_id(&self) -> Option<i64> {
        self.inner.protocol_id
    }
    /// Freeform notes.
    #[getter]
    fn notes(&self) -> Option<String> {
        self.inner.notes.clone()
    }
    /// Freeform JSON payload.
    #[getter]
    fn extra(&self) -> Option<String> {
        self.inner.extra.clone()
    }
    /// ISO 8601 UTC creation timestamp.
    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at.clone()
    }

    fn __repr__(&self) -> String {
        format!("Session(id={}, name={:?})", self.inner.id, self.inner.name)
    }
}

/// One annotation tier: the header row in `tier`. Annotation rows
/// belonging to it live in `annotation_interval` / `annotation_point` /
/// `annotation_reference` (for the three sparse types) or a Parquet sidecar
/// (the three dense types, landing in B3). Read-only view; create via
/// `Project.add_tier(...)`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Tier", frozen)]
struct PyTier {
    inner: sadda_engine::Tier,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTier {
    /// Tier id (primary key).
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// Bundle this tier belongs to.
    #[getter]
    fn bundle_id(&self) -> i64 {
        self.inner.bundle_id
    }
    /// Human-readable tier name (unique within a bundle).
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    /// Tier type: one of `interval`, `point`, `reference`,
    /// `continuous_numeric`, `continuous_vector`, `categorical_sampled`.
    #[getter]
    fn r#type(&self) -> &'static str {
        self.inner.r#type.as_str()
    }
    /// Optional parent tier id.
    #[getter]
    fn parent_id(&self) -> Option<i64> {
        self.inner.parent_id
    }
    /// Parent-child cardinality (`one_to_one` | `one_to_many` | `many_to_one`
    /// | `none`).
    #[getter]
    fn cardinality(&self) -> Option<String> {
        self.inner.cardinality.clone()
    }
    /// JSON `schema` payload.
    #[getter]
    fn schema(&self) -> Option<String> {
        self.inner.schema.clone()
    }
    /// JSON `extra` payload.
    #[getter]
    fn extra(&self) -> Option<String> {
        self.inner.extra.clone()
    }
    /// ISO 8601 UTC creation timestamp.
    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Tier(id={}, name={:?}, type={:?}, bundle_id={})",
            self.inner.id,
            self.inner.name,
            self.inner.r#type.as_str(),
            self.inner.bundle_id,
        )
    }
}

/// One interval annotation. Read-only view; create via
/// `Project.add_interval(...)`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Interval", frozen)]
struct PyInterval {
    inner: sadda_engine::Interval,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInterval {
    /// Annotation id.
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// Tier this interval belongs to.
    #[getter]
    fn tier_id(&self) -> i64 {
        self.inner.tier_id
    }
    /// Start time in seconds.
    #[getter]
    fn start_seconds(&self) -> f64 {
        self.inner.start_seconds
    }
    /// End time in seconds.
    #[getter]
    fn end_seconds(&self) -> f64 {
        self.inner.end_seconds
    }
    /// Duration in seconds (`end_seconds - start_seconds`).
    #[getter]
    fn duration_seconds(&self) -> f64 {
        self.inner.end_seconds - self.inner.start_seconds
    }
    /// Label string.
    #[getter]
    fn label(&self) -> Option<String> {
        self.inner.label.clone()
    }
    /// Parent annotation id in the parent tier.
    #[getter]
    fn parent_annotation_id(&self) -> Option<i64> {
        self.inner.parent_annotation_id
    }
    /// Annotation status (a rubric-defined status string), or `None`.
    #[getter]
    fn status(&self) -> Option<String> {
        self.inner.status.clone()
    }
    /// Free-text note, or `None`.
    #[getter]
    fn note(&self) -> Option<String> {
        self.inner.note.clone()
    }
    /// Provenance link to the producing `ProcessingRun` (e.g. a criterion
    /// run), or `None` for a hand-made annotation.
    #[getter]
    fn processing_run_id(&self) -> Option<i64> {
        self.inner.processing_run_id
    }
    /// JSON `extra` payload.
    #[getter]
    fn extra(&self) -> Option<String> {
        self.inner.extra.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Interval(id={}, tier_id={}, start={}, end={}, label={:?})",
            self.inner.id,
            self.inner.tier_id,
            self.inner.start_seconds,
            self.inner.end_seconds,
            self.inner.label,
        )
    }
}

/// One point annotation. Read-only view; create via
/// `Project.add_point(...)`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Point", frozen)]
struct PyPoint {
    inner: sadda_engine::Point,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPoint {
    /// Annotation id.
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// Tier this point belongs to.
    #[getter]
    fn tier_id(&self) -> i64 {
        self.inner.tier_id
    }
    /// Time in seconds.
    #[getter]
    fn time_seconds(&self) -> f64 {
        self.inner.time_seconds
    }
    /// Label string.
    #[getter]
    fn label(&self) -> Option<String> {
        self.inner.label.clone()
    }
    /// Parent annotation id.
    #[getter]
    fn parent_annotation_id(&self) -> Option<i64> {
        self.inner.parent_annotation_id
    }
    /// Annotation status (a rubric-defined status string), or `None`.
    #[getter]
    fn status(&self) -> Option<String> {
        self.inner.status.clone()
    }
    /// Free-text note, or `None`.
    #[getter]
    fn note(&self) -> Option<String> {
        self.inner.note.clone()
    }
    /// Provenance link to the producing `ProcessingRun` (e.g. a criterion
    /// run), or `None` for a hand-made annotation.
    #[getter]
    fn processing_run_id(&self) -> Option<i64> {
        self.inner.processing_run_id
    }
    /// JSON `extra` payload.
    #[getter]
    fn extra(&self) -> Option<String> {
        self.inner.extra.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Point(id={}, tier_id={}, time={}, label={:?})",
            self.inner.id, self.inner.tier_id, self.inner.time_seconds, self.inner.label,
        )
    }
}

/// One reference annotation. Read-only view; create via
/// `Project.add_reference(...)`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Reference", frozen)]
struct PyReference {
    inner: sadda_engine::Reference,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyReference {
    /// Annotation id.
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// Tier this reference belongs to.
    #[getter]
    fn tier_id(&self) -> i64 {
        self.inner.tier_id
    }
    /// Target kind: `bundle` | `session` | `speaker` | `tier` | `annotation`.
    #[getter]
    fn target_kind(&self) -> String {
        self.inner.target_kind.clone()
    }
    /// Target row id.
    #[getter]
    fn target_id(&self) -> i64 {
        self.inner.target_id
    }
    /// Label string.
    #[getter]
    fn label(&self) -> Option<String> {
        self.inner.label.clone()
    }
    /// Parent annotation id.
    #[getter]
    fn parent_annotation_id(&self) -> Option<i64> {
        self.inner.parent_annotation_id
    }
    /// JSON `extra` payload.
    #[getter]
    fn extra(&self) -> Option<String> {
        self.inner.extra.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Reference(id={}, tier_id={}, target_kind={:?}, target_id={})",
            self.inner.id, self.inner.tier_id, self.inner.target_kind, self.inner.target_id,
        )
    }
}

/// The project's annotation rubric (one per project): guidelines + the
/// allowed status vocabulary + per-tier controlled vocabularies. Read-only
/// view; create/update via `Project.set_rubric(...)`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Rubric", frozen)]
struct PyRubric {
    inner: sadda_engine::Rubric,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRubric {
    /// Always 1 (the rubric is a per-project singleton).
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// Human-readable rubric name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    /// Monotonic version integer.
    #[getter]
    fn version(&self) -> i64 {
        self.inner.version
    }
    /// Free-text annotation guidelines, or `None`.
    #[getter]
    fn guidelines(&self) -> Option<String> {
        self.inner.guidelines.clone()
    }
    /// ISO 8601 UTC creation timestamp.
    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at.clone()
    }
    /// ISO 8601 UTC timestamp of the last update.
    #[getter]
    fn updated_at(&self) -> String {
        self.inner.updated_at.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Rubric(name={:?}, version={})",
            self.inner.name, self.inner.version
        )
    }
}

/// One allowed annotation-status value defined by the rubric. Read-only view.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "StatusDef", frozen)]
struct PyStatusDef {
    inner: sadda_engine::StatusDef,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStatusDef {
    /// The status string stored on annotations.
    #[getter]
    fn value(&self) -> String {
        self.inner.value.clone()
    }
    /// Optional description of what the status means.
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }
    /// Display ordering (ascending).
    #[getter]
    fn sort_order(&self) -> i64 {
        self.inner.sort_order
    }

    fn __repr__(&self) -> String {
        format!("StatusDef(value={:?})", self.inner.value)
    }
}

/// The rubric's configuration for a tier name: guidelines + open/closed
/// controlled vocabulary. Read-only view; set via `Project.set_rubric_tier`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "RubricTier", frozen)]
struct PyRubricTier {
    inner: sadda_engine::RubricTier,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRubricTier {
    /// Tier name this configuration applies to.
    #[getter]
    fn tier_name(&self) -> String {
        self.inner.tier_name.clone()
    }
    /// Optional per-tier annotation guidance.
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }
    /// Whether the controlled vocabulary is closed (rejects out-of-vocab).
    #[getter]
    fn closed_vocabulary(&self) -> bool {
        self.inner.closed_vocabulary
    }

    fn __repr__(&self) -> String {
        format!(
            "RubricTier(tier_name={:?}, closed_vocabulary={})",
            self.inner.tier_name, self.inner.closed_vocabulary
        )
    }
}

/// One controlled-vocabulary entry (an allowed label) for a tier. Read-only.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "VocabEntry", frozen)]
struct PyVocabEntry {
    inner: sadda_engine::VocabEntry,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyVocabEntry {
    /// The allowed label value.
    #[getter]
    fn value(&self) -> String {
        self.inner.value.clone()
    }
    /// Optional gloss / description of the label.
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }
    /// Display ordering (ascending).
    #[getter]
    fn sort_order(&self) -> i64 {
        self.inner.sort_order
    }

    fn __repr__(&self) -> String {
        format!("VocabEntry(value={:?})", self.inner.value)
    }
}

/// Result of checking a label against a tier's controlled vocabulary.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "LabelCheck", frozen)]
struct PyLabelCheck {
    inner: sadda_engine::LabelCheck,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLabelCheck {
    /// Whether the tier has any controlled vocabulary defined.
    #[getter]
    fn has_vocabulary(&self) -> bool {
        self.inner.has_vocabulary
    }
    /// Whether the tier's vocabulary is closed.
    #[getter]
    fn closed(&self) -> bool {
        self.inner.closed
    }
    /// Whether the label is in the vocabulary (true for empty/no label, and
    /// when no vocabulary is defined).
    #[getter]
    fn in_vocabulary(&self) -> bool {
        self.inner.in_vocabulary
    }

    fn __repr__(&self) -> String {
        format!(
            "LabelCheck(has_vocabulary={}, closed={}, in_vocabulary={})",
            self.inner.has_vocabulary, self.inner.closed, self.inner.in_vocabulary
        )
    }
}

/// A criteria-engine rule (S2). Read-only view; create via
/// `Project.set_criterion(...)`. `kind` is `"structured"` (a JSON rule the
/// engine evaluates) or `"python"` (a function body run by
/// `sadda.criteria.run_criterion`).
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Criterion", frozen)]
struct PyCriterion {
    inner: sadda_engine::Criterion,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCriterion {
    /// Criterion id.
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// Unique name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    /// Optional description.
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }
    /// `"structured"` or `"python"`.
    #[getter]
    fn kind(&self) -> String {
        self.inner.kind.clone()
    }
    /// JSON rule (structured) or Python source (python).
    #[getter]
    fn body(&self) -> String {
        self.inner.body.clone()
    }
    /// Name of the tier accepted proposals promote to.
    #[getter]
    fn target_tier(&self) -> String {
        self.inner.target_tier.clone()
    }
    /// ISO 8601 UTC creation timestamp.
    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at.clone()
    }
    /// ISO 8601 UTC timestamp of the last update.
    #[getter]
    fn updated_at(&self) -> String {
        self.inner.updated_at.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Criterion(name={:?}, kind={:?}, target_tier={:?})",
            self.inner.name, self.inner.kind, self.inner.target_tier
        )
    }
}

/// A campaign **target** (slice S4a): the first-class unit of annotation work —
/// a region of interest on a bundle that needs a kind of annotation, carrying a
/// `status` through the campaign lifecycle. Generated from a criterion's RoI
/// selection (`source = "criterion"`) or hand-marked (`source = "manual"`).
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Target", frozen)]
struct PyTarget {
    inner: sadda_engine::Target,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTarget {
    /// Target id (primary key).
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// Bundle (file) the RoI lives on.
    #[getter]
    fn bundle_id(&self) -> i64 {
        self.inner.bundle_id
    }
    /// RoI start, in seconds.
    #[getter]
    fn start_seconds(&self) -> f64 {
        self.inner.start_seconds
    }
    /// RoI end, in seconds.
    #[getter]
    fn end_seconds(&self) -> f64 {
        self.inner.end_seconds
    }
    /// What kind of annotation work the RoI needs (usually a tier name).
    #[getter]
    fn target_type(&self) -> String {
        self.inner.target_type.clone()
    }
    /// Lifecycle: `unassigned` / `assigned` / `in_progress` / `done` / `flagged`.
    #[getter]
    fn status(&self) -> String {
        self.inner.status.clone()
    }
    /// Origin: `manual` or `criterion`.
    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }
    /// Generating criterion id when `source == "criterion"`, else `None`.
    #[getter]
    fn criterion_id(&self) -> Option<i64> {
        self.inner.criterion_id
    }
    /// Optional free-text note.
    #[getter]
    fn note(&self) -> Option<String> {
        self.inner.note.clone()
    }
    /// ISO 8601 UTC creation timestamp.
    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at.clone()
    }
    /// ISO 8601 UTC timestamp of the last update.
    #[getter]
    fn updated_at(&self) -> String {
        self.inner.updated_at.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Target(id={}, bundle_id={}, roi=[{}, {}], type={:?}, status={:?}, source={:?})",
            self.inner.id,
            self.inner.bundle_id,
            self.inner.start_seconds,
            self.inner.end_seconds,
            self.inner.target_type,
            self.inner.status,
            self.inner.source,
        )
    }
}

/// An **assignment** (slice S4b): distributes a `Target` to an annotator. A
/// dedicated object; a target may carry several (overlap → agreement). Created
/// by hand (`Project.add_assignment`) or in bulk (`assign_targets_randomly`).
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Assignment", frozen)]
struct PyAssignment {
    inner: sadda_engine::Assignment,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAssignment {
    /// Assignment id (primary key).
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// The target this assignment distributes.
    #[getter]
    fn target_id(&self) -> i64 {
        self.inner.target_id
    }
    /// The annotator (free-text identifier).
    #[getter]
    fn annotator(&self) -> String {
        self.inner.annotator.clone()
    }
    /// `"primary"` or `"secondary"`.
    #[getter]
    fn role(&self) -> String {
        self.inner.role.clone()
    }
    /// Per-annotator progress: `assigned` / `in_progress` / `done`.
    #[getter]
    fn status(&self) -> String {
        self.inner.status.clone()
    }
    /// The `assign_targets_randomly` seed when batch-assigned, else `None`.
    #[getter]
    fn seed(&self) -> Option<i64> {
        self.inner.seed
    }
    /// ISO 8601 UTC creation timestamp.
    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at.clone()
    }
    /// ISO 8601 UTC timestamp of the last update.
    #[getter]
    fn updated_at(&self) -> String {
        self.inner.updated_at.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Assignment(id={}, target_id={}, annotator={:?}, role={:?}, status={:?})",
            self.inner.id,
            self.inner.target_id,
            self.inner.annotator,
            self.inner.role,
            self.inner.status,
        )
    }
}

/// Result of `Project.export_annotator_package` (slice S4c).
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "ExportSummary", frozen)]
struct PyExportSummary {
    inner: sadda_engine::ExportSummary,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyExportSummary {
    /// The annotator the package was built for.
    #[getter]
    fn annotator(&self) -> String {
        self.inner.annotator.clone()
    }
    /// The package directory (a self-contained sadda sub-project).
    #[getter]
    fn path(&self) -> String {
        self.inner.path.to_string_lossy().into_owned()
    }
    /// Number of bundles included.
    #[getter]
    fn bundles(&self) -> usize {
        self.inner.bundles
    }
    /// Number of targets included.
    #[getter]
    fn targets(&self) -> usize {
        self.inner.targets
    }
    /// Number of assignments included.
    #[getter]
    fn assignments(&self) -> usize {
        self.inner.assignments
    }

    fn __repr__(&self) -> String {
        format!(
            "ExportSummary(annotator={:?}, bundles={}, targets={}, assignments={})",
            self.inner.annotator, self.inner.bundles, self.inner.targets, self.inner.assignments
        )
    }
}

/// Result of `Project.import_annotator_package` (slice S4c).
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "ImportSummary", frozen)]
struct PyImportSummary {
    inner: sadda_engine::ImportSummary,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyImportSummary {
    /// The annotator whose work was merged in.
    #[getter]
    fn annotator(&self) -> String {
        self.inner.annotator.clone()
    }
    /// Package bundles matched (by name) to a bundle in this project.
    #[getter]
    fn bundles_matched(&self) -> usize {
        self.inner.bundles_matched
    }
    /// Per-annotator tiers created or refilled.
    #[getter]
    fn tiers_imported(&self) -> usize {
        self.inner.tiers_imported
    }
    /// Annotations copied onto those tiers.
    #[getter]
    fn annotations_imported(&self) -> usize {
        self.inner.annotations_imported
    }
    /// Assignments advanced to `done`.
    #[getter]
    fn assignments_marked_done(&self) -> usize {
        self.inner.assignments_marked_done
    }

    fn __repr__(&self) -> String {
        format!(
            "ImportSummary(annotator={:?}, bundles_matched={}, tiers_imported={}, \
             annotations_imported={}, assignments_marked_done={})",
            self.inner.annotator,
            self.inner.bundles_matched,
            self.inner.tiers_imported,
            self.inner.annotations_imported,
            self.inner.assignments_marked_done,
        )
    }
}

/// Result of `Project.build_concordance` (P3 aggregate view). Describes the
/// derived concatenated-timeline bundle that was created.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "ConcordanceSummary", frozen)]
struct PyConcordanceSummary {
    inner: sadda_engine::ConcordanceSummary,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyConcordanceSummary {
    /// Id of the new derived bundle holding the concatenated tokens.
    #[getter]
    fn bundle_id(&self) -> i64 {
        self.inner.bundle_id
    }
    /// Number of tokens (matching intervals) concatenated.
    #[getter]
    fn n_tokens(&self) -> usize {
        self.inner.n_tokens
    }
    /// Total duration of the concatenated timeline, in seconds.
    #[getter]
    fn duration_seconds(&self) -> f64 {
        self.inner.duration_seconds
    }
    /// Surrounding annotations clipped + remapped onto the timeline.
    #[getter]
    fn n_context_annotations(&self) -> usize {
        self.inner.n_context_annotations
    }

    fn __repr__(&self) -> String {
        format!(
            "ConcordanceSummary(bundle_id={}, n_tokens={}, duration_seconds={:.3}, \
             n_context_annotations={})",
            self.inner.bundle_id,
            self.inner.n_tokens,
            self.inner.duration_seconds,
            self.inner.n_context_annotations,
        )
    }
}

/// Inter-annotation agreement between two tiers (slice S5) — the result of
/// `Project.compare_tiers`. Percentages are fractions in [0, 1]; κ is in
/// (-inf, 1]. Frame fields are 0.0 for point comparisons.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "AgreementReport", frozen)]
struct PyAgreementReport {
    inner: sadda_engine::AgreementReport,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAgreementReport {
    /// `"interval"` or `"point"`.
    #[getter]
    fn tier_type(&self) -> String {
        self.inner.tier_type.clone()
    }
    /// Unit count on side A.
    #[getter]
    fn n_a(&self) -> usize {
        self.inner.n_a
    }
    /// Unit count on side B.
    #[getter]
    fn n_b(&self) -> usize {
        self.inner.n_b
    }
    /// Units matched 1:1.
    #[getter]
    fn n_matched(&self) -> usize {
        self.inner.n_matched
    }
    /// Units only in A (deletions).
    #[getter]
    fn n_only_a(&self) -> usize {
        self.inner.n_only_a
    }
    /// Units only in B (insertions).
    #[getter]
    fn n_only_b(&self) -> usize {
        self.inner.n_only_b
    }
    /// Fraction of matched pairs whose labels agree.
    #[getter]
    fn percent_label_agreement(&self) -> f64 {
        self.inner.percent_label_agreement
    }
    /// Cohen's κ over matched labels.
    #[getter]
    fn cohen_kappa(&self) -> f64 {
        self.inner.cohen_kappa
    }
    /// Mean absolute boundary/time deviation (seconds) over matched pairs.
    #[getter]
    fn mean_abs_boundary_diff(&self) -> f64 {
        self.inner.mean_abs_boundary_diff
    }
    /// Fraction of matched boundaries/instants within the tolerance.
    #[getter]
    fn boundary_within_tolerance(&self) -> f64 {
        self.inner.boundary_within_tolerance
    }
    /// The boundary tolerance used (seconds).
    #[getter]
    fn boundary_tolerance_seconds(&self) -> f64 {
        self.inner.boundary_tolerance_seconds
    }
    /// Frame-based fraction of agreeing frames (intervals only).
    #[getter]
    fn frame_percent_agreement(&self) -> f64 {
        self.inner.frame_percent_agreement
    }
    /// Frame-based Cohen's κ (intervals only).
    #[getter]
    fn frame_kappa(&self) -> f64 {
        self.inner.frame_kappa
    }
    /// The frame step used (seconds).
    #[getter]
    fn frame_step_seconds(&self) -> f64 {
        self.inner.frame_step_seconds
    }

    fn __repr__(&self) -> String {
        format!(
            "AgreementReport(type={:?}, matched={}/{}+{}, label_agree={:.3}, kappa={:.3}, frame_kappa={:.3})",
            self.inner.tier_type,
            self.inner.n_matched,
            self.inner.n_only_a,
            self.inner.n_only_b,
            self.inner.percent_label_agreement,
            self.inner.cohen_kappa,
            self.inner.frame_kappa,
        )
    }
}

/// A bundle's target counts by status (slice S5) — the campaign progress
/// readout from `Project.target_progress`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "ProgressCounts", frozen)]
struct PyProgressCounts {
    inner: sadda_engine::ProgressCounts,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProgressCounts {
    /// All targets on the bundle.
    #[getter]
    fn total(&self) -> usize {
        self.inner.total
    }
    /// `unassigned` targets.
    #[getter]
    fn unassigned(&self) -> usize {
        self.inner.unassigned
    }
    /// `assigned` targets.
    #[getter]
    fn assigned(&self) -> usize {
        self.inner.assigned
    }
    /// `in_progress` targets.
    #[getter]
    fn in_progress(&self) -> usize {
        self.inner.in_progress
    }
    /// `done` targets.
    #[getter]
    fn done(&self) -> usize {
        self.inner.done
    }
    /// `flagged` targets.
    #[getter]
    fn flagged(&self) -> usize {
        self.inner.flagged
    }

    fn __repr__(&self) -> String {
        format!(
            "ProgressCounts(total={}, done={}, in_progress={}, assigned={}, unassigned={}, flagged={})",
            self.inner.total,
            self.inner.done,
            self.inner.in_progress,
            self.inner.assigned,
            self.inner.unassigned,
            self.inner.flagged,
        )
    }
}

/// One annotator's assignment counts across the project (slice S6) — from
/// `Project.assignment_progress`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "AnnotatorProgress", frozen)]
struct PyAnnotatorProgress {
    inner: sadda_engine::AnnotatorProgress,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAnnotatorProgress {
    /// The annotator.
    #[getter]
    fn annotator(&self) -> String {
        self.inner.annotator.clone()
    }
    /// Assignments not yet started.
    #[getter]
    fn assigned(&self) -> usize {
        self.inner.assigned
    }
    /// Assignments in progress.
    #[getter]
    fn in_progress(&self) -> usize {
        self.inner.in_progress
    }
    /// Assignments done.
    #[getter]
    fn done(&self) -> usize {
        self.inner.done
    }

    fn __repr__(&self) -> String {
        format!(
            "AnnotatorProgress(annotator={:?}, done={}, in_progress={}, assigned={})",
            self.inner.annotator, self.inner.done, self.inner.in_progress, self.inner.assigned
        )
    }
}

/// QA findings for one tier (slice S6) — from `Project.tier_qa`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "QaReport", frozen)]
struct PyQaReport {
    inner: sadda_engine::QaReport,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyQaReport {
    /// The tier inspected.
    #[getter]
    fn tier_id(&self) -> i64 {
        self.inner.tier_id
    }
    /// Total annotations on the tier.
    #[getter]
    fn n_annotations(&self) -> usize {
        self.inner.n_annotations
    }
    /// Annotations whose label is out of the controlled vocabulary.
    #[getter]
    fn out_of_vocab(&self) -> usize {
        self.inner.out_of_vocab
    }
    /// Annotations with an empty / missing label.
    #[getter]
    fn missing_label(&self) -> usize {
        self.inner.missing_label
    }
    /// Overlapping interval pairs (0 for point tiers).
    #[getter]
    fn overlaps(&self) -> usize {
        self.inner.overlaps
    }

    fn __repr__(&self) -> String {
        format!(
            "QaReport(tier_id={}, n={}, out_of_vocab={}, missing={}, overlaps={})",
            self.inner.tier_id,
            self.inner.n_annotations,
            self.inner.out_of_vocab,
            self.inner.missing_label,
            self.inner.overlaps,
        )
    }
}

/// Pairwise inter-annotator agreement for one tier (slice S6) — an element of
/// `Project.agreement_summary`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "PairAgreement", frozen)]
struct PyPairAgreement {
    inner: sadda_engine::PairAgreement,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPairAgreement {
    /// First annotator.
    #[getter]
    fn annotator_a(&self) -> String {
        self.inner.annotator_a.clone()
    }
    /// Second annotator.
    #[getter]
    fn annotator_b(&self) -> String {
        self.inner.annotator_b.clone()
    }
    /// Their agreement.
    #[getter]
    fn report(&self) -> PyAgreementReport {
        PyAgreementReport {
            inner: self.inner.report.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PairAgreement({:?} vs {:?}, kappa={:.3})",
            self.inner.annotator_a, self.inner.annotator_b, self.inner.report.cohen_kappa
        )
    }
}

/// One tier's config within a published rubric snapshot (slice S6b).
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "RubricTierSnapshot", frozen)]
struct PyRubricTierSnapshot {
    inner: sadda_engine::RubricTierSnapshot,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRubricTierSnapshot {
    /// Tier name.
    #[getter]
    fn tier_name(&self) -> String {
        self.inner.tier_name.clone()
    }
    /// Per-tier guidance.
    #[getter]
    fn description(&self) -> Option<String> {
        self.inner.description.clone()
    }
    /// Whether the controlled vocabulary is closed.
    #[getter]
    fn closed_vocabulary(&self) -> bool {
        self.inner.closed_vocabulary
    }
    /// The controlled vocabulary at snapshot time.
    #[getter]
    fn vocab(&self) -> Vec<PyVocabEntry> {
        self.inner
            .vocab
            .iter()
            .map(|v| PyVocabEntry { inner: v.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "RubricTierSnapshot(tier_name={:?}, closed={}, vocab={})",
            self.inner.tier_name,
            self.inner.closed_vocabulary,
            self.inner.vocab.len()
        )
    }
}

/// A published rubric version snapshot (slice S6b) — from
/// `Project.publish_rubric_version` / `rubric_versions` / `get_rubric_version`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "RubricVersion", frozen)]
struct PyRubricVersion {
    inner: sadda_engine::RubricVersion,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyRubricVersion {
    /// The rubric version captured.
    #[getter]
    fn version(&self) -> i64 {
        self.inner.version
    }
    /// Rubric name at snapshot time.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    /// Guidelines prose at snapshot time.
    #[getter]
    fn guidelines(&self) -> Option<String> {
        self.inner.guidelines.clone()
    }
    /// Note recorded at publish time.
    #[getter]
    fn note(&self) -> Option<String> {
        self.inner.note.clone()
    }
    /// ISO 8601 UTC publish timestamp.
    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at.clone()
    }
    /// Status vocabulary at snapshot time.
    #[getter]
    fn statuses(&self) -> Vec<PyStatusDef> {
        self.inner
            .statuses
            .iter()
            .map(|s| PyStatusDef { inner: s.clone() })
            .collect()
    }
    /// Per-tier config + controlled vocabularies at snapshot time.
    #[getter]
    fn tiers(&self) -> Vec<PyRubricTierSnapshot> {
        self.inner
            .tiers
            .iter()
            .map(|t| PyRubricTierSnapshot { inner: t.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "RubricVersion(version={}, name={:?}, tiers={})",
            self.inner.version,
            self.inner.name,
            self.inner.tiers.len()
        )
    }
}

/// How a rubric change affects one tier (slice S6b) — an element of
/// `Project.rubric_impact`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "TierImpact", frozen)]
struct PyTierImpact {
    inner: sadda_engine::TierImpact,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTierImpact {
    /// Tier name.
    #[getter]
    fn tier_name(&self) -> String {
        self.inner.tier_name.clone()
    }
    /// Vocabulary values added since the compared version.
    #[getter]
    fn vocab_added(&self) -> Vec<String> {
        self.inner.vocab_added.clone()
    }
    /// Vocabulary values removed since the compared version.
    #[getter]
    fn vocab_removed(&self) -> Vec<String> {
        self.inner.vocab_removed.clone()
    }
    /// Current annotations now out of the current vocabulary.
    #[getter]
    fn affected_annotations(&self) -> usize {
        self.inner.affected_annotations
    }

    fn __repr__(&self) -> String {
        format!(
            "TierImpact(tier_name={:?}, added={:?}, removed={:?}, affected={})",
            self.inner.tier_name,
            self.inner.vocab_added,
            self.inner.vocab_removed,
            self.inner.affected_annotations,
        )
    }
}

/// A PI lab-notebook entry (slice S7): an observation / measurement / decision
/// captured while exploring, promotable into a criterion or rubric guidance.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "NotebookEntry", frozen)]
struct PyNotebookEntry {
    inner: sadda_engine::NotebookEntry,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyNotebookEntry {
    /// Entry id.
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// What the note is about (usually a tier name).
    #[getter]
    fn target_type(&self) -> String {
        self.inner.target_type.clone()
    }
    /// `observation` / `measurement` / `decision`.
    #[getter]
    fn kind(&self) -> String {
        self.inner.kind.clone()
    }
    /// The note prose.
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }
    /// Optional recorded measurement.
    #[getter]
    fn measurement(&self) -> Option<String> {
        self.inner.measurement.clone()
    }
    /// Optional context bundle.
    #[getter]
    fn bundle_id(&self) -> Option<i64> {
        self.inner.bundle_id
    }
    /// `criterion` / `rubric_guidance` once promoted, else `None`.
    #[getter]
    fn promoted_kind(&self) -> Option<String> {
        self.inner.promoted_kind.clone()
    }
    /// Reference (name) of the produced artifact once promoted.
    #[getter]
    fn promoted_ref(&self) -> Option<String> {
        self.inner.promoted_ref.clone()
    }
    /// ISO 8601 UTC creation timestamp.
    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at.clone()
    }
    /// ISO 8601 UTC timestamp of the last update.
    #[getter]
    fn updated_at(&self) -> String {
        self.inner.updated_at.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "NotebookEntry(id={}, target_type={:?}, kind={:?}, promoted={:?})",
            self.inner.id, self.inner.target_type, self.inner.kind, self.inner.promoted_kind
        )
    }
}

/// Registration row for a Parquet sidecar holding a dense tier's data.
/// Created automatically by the `Project.write_continuous_numeric` /
/// `write_continuous_vector` / `write_categorical_sampled` methods.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "DerivedSignal", frozen)]
struct PyDerivedSignal {
    inner: sadda_engine::DerivedSignal,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyDerivedSignal {
    /// DerivedSignal id (primary key).
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// Tier this sidecar belongs to.
    #[getter]
    fn tier_id(&self) -> i64 {
        self.inner.tier_id
    }
    /// Path to the Parquet sidecar, relative to the project root.
    #[getter]
    fn relative_path(&self) -> String {
        self.inner.relative_path.clone()
    }
    /// Number of frames in the sidecar.
    #[getter]
    fn n_frames(&self) -> i64 {
        self.inner.n_frames
    }
    /// Number of dimensions per frame.
    #[getter]
    fn n_dims(&self) -> i64 {
        self.inner.n_dims
    }
    /// Sample rate in Hz; `None` for non-sampled / variable-rate signals.
    #[getter]
    fn sample_rate_hz(&self) -> Option<f64> {
        self.inner.sample_rate_hz
    }
    /// Dtype label: `f64`, `f32`, `utf8`.
    #[getter]
    fn dtype(&self) -> String {
        self.inner.dtype.clone()
    }
    /// Freeform JSON payload.
    #[getter]
    fn extra(&self) -> Option<String> {
        self.inner.extra.clone()
    }
    /// ISO 8601 UTC creation timestamp.
    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DerivedSignal(id={}, tier_id={}, relative_path={:?}, n_frames={}, n_dims={}, dtype={:?})",
            self.inner.id,
            self.inner.tier_id,
            self.inner.relative_path,
            self.inner.n_frames,
            self.inner.n_dims,
            self.inner.dtype,
        )
    }
}

/// One row of a bundle's provenance timeline — an analysis that ran on
/// the bundle. Returned by `Project.processing_runs(bundle_id)`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "ProcessingRun", frozen)]
struct PyProcessingRun {
    inner: sadda_engine::ProcessingRunRow,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProcessingRun {
    /// Processing-run id (primary key).
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// Bundle the run targeted.
    #[getter]
    fn bundle_id(&self) -> i64 {
        self.inner.bundle_id
    }
    /// `dsp_algorithm` | `ml_model` | `clinical_measure` | `plugin` | `live_recording`.
    #[getter]
    fn kind(&self) -> String {
        self.inner.kind.clone()
    }
    /// Reverse-DNS processor id, e.g. `sadda.dsp.pitch.autocorrelation`.
    #[getter]
    fn processor_id(&self) -> String {
        self.inner.processor_id.clone()
    }
    /// Sadda version at run time.
    #[getter]
    fn processor_version(&self) -> String {
        self.inner.processor_version.clone()
    }
    /// JSON parameters (processor-specific shape), if any.
    #[getter]
    fn parameters(&self) -> Option<String> {
        self.inner.parameters.clone()
    }
    /// JSON array of produced tier ids, if any.
    #[getter]
    fn output_tier_ids(&self) -> Option<String> {
        self.inner.output_tier_ids.clone()
    }
    /// ISO 8601 UTC start timestamp.
    #[getter]
    fn started_at(&self) -> String {
        self.inner.started_at.clone()
    }
    /// ISO 8601 UTC finish timestamp, if recorded.
    #[getter]
    fn finished_at(&self) -> Option<String> {
        self.inner.finished_at.clone()
    }
    /// `ok` | `error` | `partial`.
    #[getter]
    fn status(&self) -> String {
        self.inner.status.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ProcessingRun(id={}, kind={:?}, processor_id={:?}, status={:?})",
            self.inner.id, self.inner.kind, self.inner.processor_id, self.inner.status,
        )
    }
}

/// A literature reference for an analysis a bundle used. Returned by
/// `Project.citations(bundle_id)`, suitable for a paper's reference list.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Citation", frozen)]
struct PyCitation {
    inner: sadda_engine::Citation,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCitation {
    /// The processor this cites (matches `ProcessingRun.processor_id`).
    #[getter]
    fn processor_id(&self) -> String {
        self.inner.processor_id.clone()
    }
    /// Human-readable reference string.
    #[getter]
    fn reference(&self) -> String {
        self.inner.reference.clone()
    }
    /// Bare DOI, if one exists.
    #[getter]
    fn doi(&self) -> Option<String> {
        self.inner.doi.clone()
    }
    /// A resolvable weblink: the `https://doi.org/<doi>` URL when there's a DOI,
    /// otherwise an explicit canonical URL. `None` only if neither is known.
    #[getter]
    fn weblink(&self) -> Option<String> {
        self.inner.weblink()
    }

    fn __repr__(&self) -> String {
        format!(
            "Citation(processor_id={:?}, doi={:?}, weblink={:?})",
            self.inner.processor_id,
            self.inner.doi,
            self.inner.weblink()
        )
    }
}

/// Microphone / signal-chain calibration mapping dB-FS to dB-SPL.
/// Construct with `Calibration(reference_spl_db=…, reference_db_fs=…)`.
#[gen_stub_pyclass]
// `from_py_object` opts into the FromPyObject derive (pyo3 0.28 makes it
// explicit for Clone pyclasses) so `Calibration` can be passed by value
// to `add_instrument`.
#[pyclass(module = "sadda._native", name = "Calibration", frozen, from_py_object)]
#[derive(Clone)]
struct PyCalibration {
    inner: sadda_engine::Calibration,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCalibration {
    /// Builds a calibration from a reference tone: its known SPL and the
    /// dB-FS the engine measured for it.
    #[new]
    fn new(reference_spl_db: f64, reference_db_fs: f64) -> Self {
        Self {
            inner: sadda_engine::Calibration {
                reference_spl_db,
                reference_db_fs,
            },
        }
    }
    /// SPL of the calibration tone (dB-SPL).
    #[getter]
    fn reference_spl_db(&self) -> f64 {
        self.inner.reference_spl_db
    }
    /// dB-FS measured for that tone.
    #[getter]
    fn reference_db_fs(&self) -> f64 {
        self.inner.reference_db_fs
    }
    /// dB added to a dB-FS reading to obtain dB-SPL.
    fn spl_offset_db(&self) -> f64 {
        self.inner.spl_offset_db()
    }
    /// Converts a relative dB-FS value to calibrated dB-SPL.
    fn to_spl(&self, db_fs: f32) -> f32 {
        self.inner
            .to_spl(sadda_engine::Decibels::new(db_fs))
            .value()
    }

    fn __repr__(&self) -> String {
        format!(
            "Calibration(reference_spl_db={}, reference_db_fs={})",
            self.inner.reference_spl_db, self.inner.reference_db_fs
        )
    }
}

/// A capture instrument (microphone, interface) and its optional
/// calibration. Returned by `Project.instruments()` / `get_instrument()`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Instrument", frozen)]
struct PyInstrument {
    inner: sadda_engine::Instrument,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyInstrument {
    /// Instrument id (primary key).
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    /// Human-readable name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }
    /// Kind label (e.g. `"microphone"`).
    #[getter]
    fn kind(&self) -> Option<String> {
        self.inner.kind.clone()
    }
    /// Serial number.
    #[getter]
    fn serial(&self) -> Option<String> {
        self.inner.serial.clone()
    }
    /// Calibration, if the instrument has been calibrated.
    #[getter]
    fn calibration(&self) -> Option<PyCalibration> {
        self.inner.calibration.map(|inner| PyCalibration { inner })
    }
    /// Freeform JSON payload.
    #[getter]
    fn extra(&self) -> Option<String> {
        self.inner.extra.clone()
    }
    /// ISO 8601 UTC creation timestamp.
    #[getter]
    fn created_at(&self) -> String {
        self.inner.created_at.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Instrument(id={}, name={:?}, calibrated={})",
            self.inner.id,
            self.inner.name,
            self.inner.calibration.is_some()
        )
    }
}

/// Parses a `processing_run.kind` string into the engine enum, raising
/// `ValueError` on an unknown value.
fn parse_run_kind(s: &str) -> PyResult<sadda_engine::ProcessingRunKind> {
    use sadda_engine::ProcessingRunKind as K;
    Ok(match s {
        "dsp_algorithm" => K::DspAlgorithm,
        "ml_model" => K::MlModel,
        "clinical_measure" => K::ClinicalMeasure,
        "plugin" => K::Plugin,
        "live_recording" => K::LiveRecording,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "invalid processing-run kind {other:?}; expected one of \
                 dsp_algorithm, ml_model, clinical_measure, plugin, live_recording"
            )));
        }
    })
}

/// A sadda project: a directory holding audio, derived signals, attachments,
/// and a SQLite-backed corpus database. Construct via `sadda.new_project(...)`
/// or `sadda.open_project(...)`.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Project", unsendable)]
pub(crate) struct PyProject {
    // `unsendable`: rusqlite::Connection is Send but not Sync; the GIL
    // serializes Python-side access, but PyO3 requires Send+Sync by default.
    // `unsendable` tells PyO3 to check at runtime that this object isn't
    // shared across threads.
    pub(crate) inner: sadda_engine::Project,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProject {
    /// Project's filesystem root.
    #[getter]
    fn root(&self) -> String {
        self.inner.root().to_string_lossy().into_owned()
    }

    /// Project's human-readable name (from the singleton `project` row).
    #[getter]
    fn name(&self) -> PyResult<String> {
        self.inner.name().map_err(engine_err_to_py)
    }

    /// Registers a bundle by copying `source_audio_path` into the project's
    /// `signals/original/` directory and recording its metadata in the corpus
    /// database. Returns the new bundle's id. Optional kwargs attach the
    /// bundle to a Session / Speaker and set a JSON `extra` payload.
    #[pyo3(signature = (name, source_audio_path, *, session_id=None, speaker_id=None, extra=None))]
    fn add_bundle(
        &self,
        name: &str,
        source_audio_path: PathBuf,
        session_id: Option<i64>,
        speaker_id: Option<i64>,
        extra: Option<String>,
    ) -> PyResult<i64> {
        let spec = sadda_engine::BundleSpec {
            name: name.into(),
            session_id,
            speaker_id,
            extra,
        };
        self.inner
            .add_bundle_with(&spec, source_audio_path)
            .map_err(engine_err_to_py)
    }

    /// Splits a (typically very long) WAV into contiguous chunks of about
    /// `chunk_seconds` each, writing every chunk into the project as its own
    /// bundle named `"<name_prefix>_NNN"`. The source is streamed, so memory
    /// stays flat regardless of length — this is how a file too large to load
    /// whole still gets in. Chunk audio preserves the source format; the final
    /// chunk holds the remainder. Returns the new bundle ids in order.
    fn add_bundle_split(
        &self,
        name_prefix: &str,
        source_audio_path: PathBuf,
        chunk_seconds: f64,
    ) -> PyResult<Vec<i64>> {
        self.inner
            .add_bundle_split(name_prefix, source_audio_path, chunk_seconds)
            .map_err(engine_err_to_py)
    }

    /// Lists all bundles in id order.
    fn bundles(&self) -> PyResult<Vec<PyBundle>> {
        self.inner
            .bundles()
            .map(|bs| bs.into_iter().map(|inner| PyBundle { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Permanently deletes a bundle and all its tiers, annotations,
    /// derived signals, and processing-run audit rows. Best-effort
    /// removes the underlying WAV from disk. No-op if `bundle_id`
    /// does not exist.
    fn delete_bundle(&self, bundle_id: i64) -> PyResult<()> {
        self.inner
            .delete_bundle(bundle_id)
            .map_err(engine_err_to_py)
    }

    /// Renames a bundle's display name. The underlying WAV file is
    /// left untouched. Raises if `bundle_id` does not exist or the
    /// new name is empty / whitespace-only.
    fn rename_bundle(&self, bundle_id: i64, new_name: &str) -> PyResult<()> {
        self.inner
            .rename_bundle(bundle_id, new_name)
            .map_err(engine_err_to_py)
    }

    /// Records a completed processing run for audit provenance and
    /// returns its id. The engine fills in the sadda version, timestamps,
    /// and active recipe id. `kind` is one of `dsp_algorithm`,
    /// `ml_model`, `clinical_measure`, `plugin`, `live_recording`.
    #[pyo3(signature = (bundle_id, kind, processor_id, *, parameters=None,
        input_tier_ids=None, output_tier_ids=None, output_signal_ids=None,
        weights_checksum=None))]
    #[allow(clippy::too_many_arguments)]
    fn record_processing_run(
        &self,
        bundle_id: i64,
        kind: &str,
        processor_id: &str,
        parameters: Option<String>,
        input_tier_ids: Option<Vec<i64>>,
        output_tier_ids: Option<Vec<i64>>,
        output_signal_ids: Option<Vec<i64>>,
        weights_checksum: Option<String>,
    ) -> PyResult<i64> {
        let mut spec =
            sadda_engine::ProcessingRunSpec::new(bundle_id, parse_run_kind(kind)?, processor_id);
        spec.parameters = parameters;
        spec.input_tier_ids = input_tier_ids.unwrap_or_default();
        spec.output_tier_ids = output_tier_ids.unwrap_or_default();
        spec.output_signal_ids = output_signal_ids.unwrap_or_default();
        spec.weights_checksum = weights_checksum;
        self.inner
            .record_processing_run(&spec)
            .map_err(engine_err_to_py)
    }

    /// Returns a bundle's processing-run timeline (provenance), oldest
    /// first.
    fn processing_runs(&self, bundle_id: i64) -> PyResult<Vec<PyProcessingRun>> {
        self.inner
            .processing_runs(bundle_id)
            .map(|rows| {
                rows.into_iter()
                    .map(|inner| PyProcessingRun { inner })
                    .collect()
            })
            .map_err(engine_err_to_py)
    }

    /// Reads a single `ProcessingRun` by id, or `None`. Resolves an
    /// annotation's `processing_run_id` to the run that produced it.
    fn get_processing_run(&self, id: i64) -> PyResult<Option<PyProcessingRun>> {
        self.inner
            .get_processing_run(id)
            .map(|opt| opt.map(|inner| PyProcessingRun { inner }))
            .map_err(engine_err_to_py)
    }

    /// Returns the literature citations for the analyses a bundle used,
    /// deduplicated by processor and ordered by first use. Uncited
    /// processors (imports, recording) are omitted.
    fn citations(&self, bundle_id: i64) -> PyResult<Vec<PyCitation>> {
        self.inner
            .citations(bundle_id)
            .map(|cs| cs.into_iter().map(|inner| PyCitation { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Inserts an instrument (microphone / interface), optionally with a
    /// `Calibration`. Returns the new instrument's id.
    #[pyo3(signature = (name, *, kind=None, serial=None, calibration=None, extra=None))]
    fn add_instrument(
        &self,
        name: &str,
        kind: Option<String>,
        serial: Option<String>,
        calibration: Option<PyCalibration>,
        extra: Option<String>,
    ) -> PyResult<i64> {
        let spec = sadda_engine::InstrumentSpec {
            name: name.into(),
            kind,
            serial,
            calibration: calibration.map(|c| c.inner),
            extra,
        };
        self.inner.add_instrument(&spec).map_err(engine_err_to_py)
    }

    /// Lists all instruments in id order.
    fn instruments(&self) -> PyResult<Vec<PyInstrument>> {
        self.inner
            .instruments()
            .map(|xs| xs.into_iter().map(|inner| PyInstrument { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Fetches a single instrument by id.
    fn get_instrument(&self, instrument_id: i64) -> PyResult<PyInstrument> {
        self.inner
            .get_instrument(instrument_id)
            .map(|inner| PyInstrument { inner })
            .map_err(engine_err_to_py)
    }

    /// Resolves a bundle's calibration via bundle → session →
    /// instrument. `None` means levels for that bundle are dB-FS only.
    fn bundle_calibration(&self, bundle_id: i64) -> PyResult<Option<PyCalibration>> {
        self.inner
            .bundle_calibration(bundle_id)
            .map(|o| o.map(|inner| PyCalibration { inner }))
            .map_err(engine_err_to_py)
    }

    /// Loads the audio file for a bundle.
    fn load_audio(&self, bundle_id: i64) -> PyResult<PyAudio> {
        self.inner
            .load_audio(bundle_id)
            .map(|inner| PyAudio { inner })
            .map_err(engine_err_to_py)
    }

    /// Inserts a Speaker row. Returns the new speaker's id.
    #[pyo3(signature = (name, *, sex=None, birth_year=None, notes=None, extra=None))]
    fn add_speaker(
        &self,
        name: &str,
        sex: Option<String>,
        birth_year: Option<i32>,
        notes: Option<String>,
        extra: Option<String>,
    ) -> PyResult<i64> {
        let spec = sadda_engine::SpeakerSpec {
            name: name.into(),
            sex,
            birth_year,
            notes,
            extra,
        };
        self.inner.add_speaker(&spec).map_err(engine_err_to_py)
    }

    /// Lists all speakers in id order.
    fn speakers(&self) -> PyResult<Vec<PySpeaker>> {
        self.inner
            .speakers()
            .map(|ss| ss.into_iter().map(|inner| PySpeaker { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Fetches a single speaker by id.
    fn get_speaker(&self, id: i64) -> PyResult<PySpeaker> {
        self.inner
            .get_speaker(id)
            .map(|inner| PySpeaker { inner })
            .map_err(engine_err_to_py)
    }

    /// Inserts a Session row. Returns the new session's id.
    #[pyo3(signature = (
        name, *,
        started_at=None, ended_at=None, location=None,
        instrument_id=None, protocol_id=None,
        notes=None, extra=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_session(
        &self,
        name: &str,
        started_at: Option<String>,
        ended_at: Option<String>,
        location: Option<String>,
        instrument_id: Option<i64>,
        protocol_id: Option<i64>,
        notes: Option<String>,
        extra: Option<String>,
    ) -> PyResult<i64> {
        let spec = sadda_engine::SessionSpec {
            name: name.into(),
            started_at,
            ended_at,
            location,
            instrument_id,
            protocol_id,
            notes,
            extra,
        };
        self.inner.add_session(&spec).map_err(engine_err_to_py)
    }

    /// Lists all sessions in id order.
    fn sessions(&self) -> PyResult<Vec<PySession>> {
        self.inner
            .sessions()
            .map(|ss| ss.into_iter().map(|inner| PySession { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Fetches a single session by id.
    fn get_session(&self, id: i64) -> PyResult<PySession> {
        self.inner
            .get_session(id)
            .map(|inner| PySession { inner })
            .map_err(engine_err_to_py)
    }

    /// User string written into `audit_log.user` for mutations on this
    /// connection. Defaults to `"local"`.
    #[getter]
    fn audit_user(&self) -> PyResult<String> {
        self.inner.audit_user().map_err(engine_err_to_py)
    }

    /// Sets the user string written into `audit_log.user` for subsequent
    /// mutations on this connection.
    fn set_audit_user(&self, user: &str) -> PyResult<()> {
        self.inner.set_audit_user(user).map_err(engine_err_to_py)
    }

    /// Pins a reference distribution `id` to a specific `version` in
    /// `project.toml`, so the project reopens against the same data for
    /// reproducibility (C7). Overwrites any existing pin for that id.
    fn pin_refdist(&self, id: &str, version: &str) -> PyResult<()> {
        self.inner
            .pin_refdist(id, version)
            .map_err(engine_err_to_py)
    }

    /// The reference distributions this project has pinned, as a
    /// `{id: version}` dict.
    fn refdist_pins(&self) -> PyResult<std::collections::HashMap<String, String>> {
        Ok(self
            .inner
            .refdist_pins()
            .map_err(engine_err_to_py)?
            .into_iter()
            .collect())
    }

    /// Removes a reference-distribution pin; returns whether one existed.
    fn remove_refdist_pin(&self, id: &str) -> PyResult<bool> {
        self.inner.remove_refdist_pin(id).map_err(engine_err_to_py)
    }

    /// E12: resolve `model_id` (`sadda/…` / `local://…` / `hf://…`), run it
    /// as an embedding extractor over `bundle_id`'s audio, and store the
    /// result as a new `continuous_vector` tier `tier_name`, recording an
    /// `ml_model` processing run. Returns the new tier id. (Provisional.)
    fn extract_embeddings(&self, bundle_id: i64, model_id: &str, tier_name: &str) -> PyResult<i64> {
        let model = sadda_engine::load_model(model_id).map_err(engine_err_to_py)?;
        self.inner
            .extract_embeddings(bundle_id, &model, tier_name)
            .map_err(engine_err_to_py)
    }

    /// Inserts a Tier row. `type` is one of `interval`, `point`, `reference`,
    /// `continuous_numeric`, `continuous_vector`, `categorical_sampled`.
    /// Returns the new tier's id.
    #[pyo3(signature = (
        bundle_id, name, r#type, *,
        parent_id=None, cardinality=None, schema=None, extra=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_tier(
        &self,
        bundle_id: i64,
        name: &str,
        r#type: &str,
        parent_id: Option<i64>,
        cardinality: Option<String>,
        schema: Option<String>,
        extra: Option<String>,
    ) -> PyResult<i64> {
        let tier_type = r#type
            .parse::<sadda_engine::TierType>()
            .map_err(engine_err_to_py)?;
        let spec = sadda_engine::TierSpec {
            bundle_id,
            name: name.into(),
            r#type: Some(tier_type),
            parent_id,
            cardinality,
            schema,
            extra,
        };
        self.inner.add_tier(&spec).map_err(engine_err_to_py)
    }

    /// Lists tiers, optionally restricted to a single bundle.
    #[pyo3(signature = (bundle_id=None))]
    fn tiers(&self, bundle_id: Option<i64>) -> PyResult<Vec<PyTier>> {
        self.inner
            .tiers(bundle_id)
            .map(|ts| ts.into_iter().map(|inner| PyTier { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Fetches a single tier by id.
    fn get_tier(&self, id: i64) -> PyResult<PyTier> {
        self.inner
            .get_tier(id)
            .map(|inner| PyTier { inner })
            .map_err(engine_err_to_py)
    }

    /// Renames a tier's display name. Raises if `tier_id` does not exist
    /// or the new name is empty / whitespace-only.
    fn rename_tier(&self, tier_id: i64, new_name: &str) -> PyResult<()> {
        self.inner
            .rename_tier(tier_id, new_name)
            .map_err(engine_err_to_py)
    }

    /// Deletes a tier and all of its annotations (intervals / points /
    /// references) and any dense derived-signal sidecar. Raises if the
    /// tier has child tiers (delete those first) or if `tier_id` does
    /// not exist.
    fn delete_tier(&self, tier_id: i64) -> PyResult<()> {
        self.inner.delete_tier(tier_id).map_err(engine_err_to_py)
    }

    /// Inserts an interval annotation. Enforces parent-child cardinality at
    /// insert time; raises `ValueError` on cardinality violation.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        tier_id, start_seconds, end_seconds, *,
        label=None, parent_annotation_id=None, status=None, note=None, extra=None,
    ))]
    fn add_interval(
        &self,
        tier_id: i64,
        start_seconds: f64,
        end_seconds: f64,
        label: Option<String>,
        parent_annotation_id: Option<i64>,
        status: Option<String>,
        note: Option<String>,
        extra: Option<String>,
    ) -> PyResult<i64> {
        let spec = sadda_engine::IntervalSpec {
            tier_id,
            start_seconds,
            end_seconds,
            label,
            parent_annotation_id,
            status,
            note,
            extra,
            ..Default::default()
        };
        self.inner.add_interval(&spec).map_err(engine_err_to_py)
    }

    /// Lists intervals for a tier in (start_seconds, id) order.
    fn intervals(&self, tier_id: i64) -> PyResult<Vec<PyInterval>> {
        self.inner
            .intervals(tier_id)
            .map(|rs| rs.into_iter().map(|inner| PyInterval { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Inserts a point annotation. Enforces parent-child cardinality.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        tier_id, time_seconds, *,
        label=None, parent_annotation_id=None, status=None, note=None, extra=None,
    ))]
    fn add_point(
        &self,
        tier_id: i64,
        time_seconds: f64,
        label: Option<String>,
        parent_annotation_id: Option<i64>,
        status: Option<String>,
        note: Option<String>,
        extra: Option<String>,
    ) -> PyResult<i64> {
        let spec = sadda_engine::PointSpec {
            tier_id,
            time_seconds,
            label,
            parent_annotation_id,
            status,
            note,
            extra,
            ..Default::default()
        };
        self.inner.add_point(&spec).map_err(engine_err_to_py)
    }

    /// Lists points for a tier in (time_seconds, id) order.
    fn points(&self, tier_id: i64) -> PyResult<Vec<PyPoint>> {
        self.inner
            .points(tier_id)
            .map(|rs| rs.into_iter().map(|inner| PyPoint { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Inserts a reference annotation pointing at another row via
    /// `(target_kind, target_id)`.
    #[pyo3(signature = (
        tier_id, target_kind, target_id, *,
        label=None, parent_annotation_id=None, extra=None,
    ))]
    fn add_reference(
        &self,
        tier_id: i64,
        target_kind: &str,
        target_id: i64,
        label: Option<String>,
        parent_annotation_id: Option<i64>,
        extra: Option<String>,
    ) -> PyResult<i64> {
        let spec = sadda_engine::ReferenceSpec {
            tier_id,
            target_kind: target_kind.into(),
            target_id,
            label,
            parent_annotation_id,
            extra,
        };
        self.inner.add_reference(&spec).map_err(engine_err_to_py)
    }

    /// Lists references for a tier in id order. Named `references_for`
    /// rather than `references` (which collides with Rust's `ref` family
    /// of grep targets).
    fn references_for(&self, tier_id: i64) -> PyResult<Vec<PyReference>> {
        self.inner
            .references_for(tier_id)
            .map(|rs| rs.into_iter().map(|inner| PyReference { inner }).collect())
            .map_err(engine_err_to_py)
    }

    // -- Annotation rubric (slice S1) ------------------------------------

    /// Creates or updates the project's singleton annotation rubric.
    #[pyo3(signature = (name, version=1, guidelines=None))]
    fn set_rubric(&self, name: &str, version: i64, guidelines: Option<&str>) -> PyResult<PyRubric> {
        self.inner
            .set_rubric(name, version, guidelines)
            .map(|inner| PyRubric { inner })
            .map_err(engine_err_to_py)
    }

    /// Reads the project's rubric, or `None` if none has been defined.
    fn rubric(&self) -> PyResult<Option<PyRubric>> {
        self.inner
            .rubric()
            .map(|opt| opt.map(|inner| PyRubric { inner }))
            .map_err(engine_err_to_py)
    }

    /// Replaces the rubric's status vocabulary. `statuses` is a list of
    /// `(value, description, sort_order)` tuples. Requires a rubric.
    fn set_rubric_statuses(&self, statuses: Vec<(String, Option<String>, i64)>) -> PyResult<()> {
        let defs: Vec<sadda_engine::StatusDef> = statuses
            .into_iter()
            .map(|(value, description, sort_order)| sadda_engine::StatusDef {
                value,
                description,
                sort_order,
            })
            .collect();
        self.inner
            .set_rubric_statuses(&defs)
            .map_err(engine_err_to_py)
    }

    /// Reads the rubric's status vocabulary in (sort_order, value) order.
    fn rubric_statuses(&self) -> PyResult<Vec<PyStatusDef>> {
        self.inner
            .rubric_statuses()
            .map(|rs| rs.into_iter().map(|inner| PyStatusDef { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Creates or updates the rubric configuration for a tier name
    /// (guidelines + open/closed vocabulary). Requires a rubric.
    #[pyo3(signature = (tier_name, description=None, closed=false))]
    fn set_rubric_tier(
        &self,
        tier_name: &str,
        description: Option<&str>,
        closed: bool,
    ) -> PyResult<PyRubricTier> {
        self.inner
            .set_rubric_tier(tier_name, description, closed)
            .map(|inner| PyRubricTier { inner })
            .map_err(engine_err_to_py)
    }

    /// Reads the rubric configuration for a tier name, or `None`.
    fn rubric_tier(&self, tier_name: &str) -> PyResult<Option<PyRubricTier>> {
        self.inner
            .rubric_tier(tier_name)
            .map(|opt| opt.map(|inner| PyRubricTier { inner }))
            .map_err(engine_err_to_py)
    }

    /// Replaces the controlled vocabulary for a tier name. `entries` is a
    /// list of `(value, description, sort_order)` tuples. Auto-creates an
    /// open rubric-tier row if needed. Requires a rubric.
    fn set_controlled_vocabulary(
        &self,
        tier_name: &str,
        entries: Vec<(String, Option<String>, i64)>,
    ) -> PyResult<()> {
        let defs: Vec<sadda_engine::VocabEntry> = entries
            .into_iter()
            .map(
                |(value, description, sort_order)| sadda_engine::VocabEntry {
                    value,
                    description,
                    sort_order,
                },
            )
            .collect();
        self.inner
            .set_controlled_vocabulary(tier_name, &defs)
            .map_err(engine_err_to_py)
    }

    /// Reads the controlled vocabulary for a tier name.
    fn controlled_vocabulary(&self, tier_name: &str) -> PyResult<Vec<PyVocabEntry>> {
        self.inner
            .controlled_vocabulary(tier_name)
            .map(|rs| rs.into_iter().map(|inner| PyVocabEntry { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Checks a label against a tier's controlled vocabulary.
    #[pyo3(signature = (tier_name, label=None))]
    fn label_check(&self, tier_name: &str, label: Option<&str>) -> PyResult<PyLabelCheck> {
        self.inner
            .label_check(tier_name, label)
            .map(|inner| PyLabelCheck { inner })
            .map_err(engine_err_to_py)
    }

    /// Sets the status + note on an interval annotation, validating the
    /// status against the rubric. Either may be `None` to clear it.
    #[pyo3(signature = (id, status=None, note=None))]
    fn set_interval_status(
        &self,
        id: i64,
        status: Option<&str>,
        note: Option<&str>,
    ) -> PyResult<()> {
        self.inner
            .set_interval_status(id, status, note)
            .map_err(engine_err_to_py)
    }

    /// Sets the status + note on a point annotation, validating the status
    /// against the rubric. Either may be `None` to clear it.
    #[pyo3(signature = (id, status=None, note=None))]
    fn set_point_status(&self, id: i64, status: Option<&str>, note: Option<&str>) -> PyResult<()> {
        self.inner
            .set_point_status(id, status, note)
            .map_err(engine_err_to_py)
    }

    // -- Criteria engine (slice S2) --------------------------------------

    /// Creates or updates a criterion (upsert by name). `kind` is
    /// `"structured"` (a JSON rule) or `"python"` (a function body).
    #[pyo3(signature = (name, kind, body, target_tier, description=None))]
    fn set_criterion(
        &self,
        name: &str,
        kind: &str,
        body: &str,
        target_tier: &str,
        description: Option<&str>,
    ) -> PyResult<PyCriterion> {
        self.inner
            .set_criterion(name, description, kind, body, target_tier)
            .map(|inner| PyCriterion { inner })
            .map_err(engine_err_to_py)
    }

    /// Lists all criteria in name order.
    fn criteria(&self) -> PyResult<Vec<PyCriterion>> {
        self.inner
            .criteria()
            .map(|cs| cs.into_iter().map(|inner| PyCriterion { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Reads a criterion by id, or `None`.
    fn get_criterion(&self, id: i64) -> PyResult<Option<PyCriterion>> {
        self.inner
            .get_criterion(id)
            .map(|opt| opt.map(|inner| PyCriterion { inner }))
            .map_err(engine_err_to_py)
    }

    /// Deletes a criterion by id (idempotent).
    fn delete_criterion(&self, id: i64) -> PyResult<()> {
        self.inner.delete_criterion(id).map_err(engine_err_to_py)
    }

    /// Adds a campaign target (a region of work) on `bundle_id` over
    /// `[start_seconds, end_seconds)` for `target_type`. `status` defaults to
    /// `"unassigned"`; `source` to `"manual"`. Returns the new target id.
    #[pyo3(signature = (
        bundle_id, start_seconds, end_seconds, target_type, *,
        status=None, source=None, criterion_id=None, note=None, extra=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_target(
        &self,
        bundle_id: i64,
        start_seconds: f64,
        end_seconds: f64,
        target_type: &str,
        status: Option<String>,
        source: Option<String>,
        criterion_id: Option<i64>,
        note: Option<String>,
        extra: Option<String>,
    ) -> PyResult<i64> {
        let spec = sadda_engine::TargetSpec {
            bundle_id,
            start_seconds,
            end_seconds,
            target_type: target_type.to_owned(),
            status,
            source,
            criterion_id,
            note,
            extra,
        };
        self.inner.add_target(&spec).map_err(engine_err_to_py)
    }

    /// Lists a bundle's targets in time order.
    fn targets(&self, bundle_id: i64) -> PyResult<Vec<PyTarget>> {
        self.inner
            .targets(bundle_id)
            .map(|ts| ts.into_iter().map(|inner| PyTarget { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Reads a target by id, or `None`.
    fn get_target(&self, id: i64) -> PyResult<Option<PyTarget>> {
        self.inner
            .get_target(id)
            .map(|opt| opt.map(|inner| PyTarget { inner }))
            .map_err(engine_err_to_py)
    }

    /// Sets a target's lifecycle `status` (one of `unassigned` / `assigned` /
    /// `in_progress` / `done` / `flagged`). Raises if the target is missing or
    /// the status is invalid.
    fn update_target_status(&self, id: i64, status: &str) -> PyResult<()> {
        self.inner
            .update_target_status(id, status)
            .map_err(engine_err_to_py)
    }

    /// Sets (or clears, with `None`) a target's note.
    #[pyo3(signature = (id, note=None))]
    fn set_target_note(&self, id: i64, note: Option<&str>) -> PyResult<()> {
        self.inner
            .set_target_note(id, note)
            .map_err(engine_err_to_py)
    }

    /// Deletes a target by id (idempotent).
    fn delete_target(&self, id: i64) -> PyResult<()> {
        self.inner.delete_target(id).map_err(engine_err_to_py)
    }

    /// Generates targets from a `structured` criterion's RoI selection on
    /// `bundle_id` — one target per surviving select interval, typed by the
    /// criterion's target tier and back-linked via `criterion_id`. Replaces
    /// this criterion's prior targets on the bundle. Returns the count.
    /// `python` criteria are rejected (run them in `sadda.criteria`).
    fn generate_targets_from_criterion(
        &self,
        criterion_id: i64,
        bundle_id: i64,
    ) -> PyResult<usize> {
        self.inner
            .generate_targets_from_criterion(criterion_id, bundle_id)
            .map_err(engine_err_to_py)
    }

    /// Assigns a target to an annotator. `role` defaults to `"primary"`,
    /// `status` to `"assigned"`. Advances the target `unassigned` → `assigned`.
    /// Raises if the target is missing or it is already assigned to `annotator`.
    /// Returns the new assignment id.
    #[pyo3(signature = (target_id, annotator, *, role=None, status=None, seed=None, extra=None))]
    fn add_assignment(
        &self,
        target_id: i64,
        annotator: &str,
        role: Option<String>,
        status: Option<String>,
        seed: Option<i64>,
        extra: Option<String>,
    ) -> PyResult<i64> {
        let spec = sadda_engine::AssignmentSpec {
            target_id,
            annotator: annotator.to_owned(),
            role,
            status,
            seed,
            extra,
        };
        self.inner.add_assignment(&spec).map_err(engine_err_to_py)
    }

    /// Lists a bundle's assignments (across all its targets).
    fn assignments(&self, bundle_id: i64) -> PyResult<Vec<PyAssignment>> {
        self.inner
            .assignments(bundle_id)
            .map(|xs| xs.into_iter().map(|inner| PyAssignment { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Lists the assignments on a single target.
    fn assignments_for_target(&self, target_id: i64) -> PyResult<Vec<PyAssignment>> {
        self.inner
            .assignments_for_target(target_id)
            .map(|xs| xs.into_iter().map(|inner| PyAssignment { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Sets an assignment's per-annotator `status` (`assigned` / `in_progress`
    /// / `done`).
    fn update_assignment_status(&self, id: i64, status: &str) -> PyResult<()> {
        self.inner
            .update_assignment_status(id, status)
            .map_err(engine_err_to_py)
    }

    /// Reassigns an assignment to a different annotator.
    fn set_assignment_annotator(&self, id: i64, annotator: &str) -> PyResult<()> {
        self.inner
            .set_assignment_annotator(id, annotator)
            .map_err(engine_err_to_py)
    }

    /// Deletes an assignment (idempotent). Reverts the target to `unassigned`
    /// when its last assignment is removed and it was merely `assigned`.
    fn delete_assignment(&self, id: i64) -> PyResult<()> {
        self.inner.delete_assignment(id).map_err(engine_err_to_py)
    }

    /// Distributes a bundle's currently-`unassigned` targets across `annotators`
    /// with a deterministic, seed-driven shuffle (reproducible — same seed +
    /// roster + targets → same assignment). Already-assigned targets are left
    /// alone, so re-running after a roster change re-randomizes the remainder.
    /// `role` defaults to `"primary"`. Returns the number assigned.
    #[pyo3(signature = (bundle_id, annotators, seed, *, role=None))]
    fn assign_targets_randomly(
        &self,
        bundle_id: i64,
        annotators: Vec<String>,
        seed: i64,
        role: Option<&str>,
    ) -> PyResult<usize> {
        self.inner
            .assign_targets_randomly(bundle_id, &annotators, seed, role)
            .map_err(engine_err_to_py)
    }

    /// Exports a self-contained sub-project for `annotator` at `dest_dir` (their
    /// assigned bundles + audio + sparse tiers/annotations + targets/assignments
    /// + the rubric + a manifest). The annotator opens it as a normal sadda
    /// project and works offline. Returns an `ExportSummary`.
    fn export_annotator_package(
        &self,
        annotator: &str,
        dest_dir: std::path::PathBuf,
    ) -> PyResult<PyExportSummary> {
        self.inner
            .export_annotator_package(annotator, &dest_dir)
            .map(|inner| PyExportSummary { inner })
            .map_err(engine_err_to_py)
    }

    /// Imports a returned annotator package at `package_dir`, landing the
    /// annotator's work on per-annotator tiers `"<tier> [annotator]"` (use
    /// `merge_tiers` to reconcile) and marking their assignments `done`.
    /// Returns an `ImportSummary`.
    fn import_annotator_package(
        &self,
        package_dir: std::path::PathBuf,
    ) -> PyResult<PyImportSummary> {
        self.inner
            .import_annotator_package(&package_dir)
            .map(|inner| PyImportSummary { inner })
            .map_err(engine_err_to_py)
    }

    /// Builds an aggregate "concordance" bundle (P3): every interval on tier
    /// `tier_name` across the corpus whose label is in `labels` (empty = any)
    /// becomes a token; the tokens' mono audio is concatenated in sequence
    /// (with `gap_seconds` of silence between) into a new bundle `dest_name`.
    /// A `"⟨source⟩"` divider tier marks each token's origin, and each token's
    /// surrounding annotations are clipped + remapped onto the timeline. The
    /// matched bundles must share one sample rate. Returns a
    /// `ConcordanceSummary`.
    #[pyo3(signature = (tier_name, labels, dest_name, gap_seconds=0.25))]
    fn build_concordance(
        &self,
        tier_name: &str,
        labels: Vec<String>,
        dest_name: &str,
        gap_seconds: f64,
    ) -> PyResult<PyConcordanceSummary> {
        self.inner
            .build_concordance(tier_name, &labels, dest_name, gap_seconds)
            .map(|inner| PyConcordanceSummary { inner })
            .map_err(engine_err_to_py)
    }

    /// Unions the annotations of `source_tier_names` into `dest_tier_name` on
    /// `bundle_id` (time-ordered), creating the destination and replacing its
    /// contents. All sources must share one type (interval/point). Returns the
    /// number of annotations written.
    fn merge_tiers(
        &self,
        bundle_id: i64,
        source_tier_names: Vec<String>,
        dest_tier_name: &str,
    ) -> PyResult<usize> {
        self.inner
            .merge_tiers(bundle_id, &source_tier_names, dest_tier_name)
            .map_err(engine_err_to_py)
    }

    /// Compares two tiers on `bundle_id` for agreement (slice S5): unit-based
    /// label κ, boundary deviation/tolerance, insertions/deletions, and — for
    /// interval tiers — a frame-based label κ/agreement. Returns an
    /// `AgreementReport`. Powers inter-annotator agreement, auto-vs-gold, and
    /// rubric-version impact alike.
    #[pyo3(signature = (
        bundle_id, tier_a_id, tier_b_id, *,
        boundary_tolerance_seconds=0.020, frame_step_seconds=0.010,
    ))]
    fn compare_tiers(
        &self,
        bundle_id: i64,
        tier_a_id: i64,
        tier_b_id: i64,
        boundary_tolerance_seconds: f64,
        frame_step_seconds: f64,
    ) -> PyResult<PyAgreementReport> {
        let opts = sadda_engine::AgreementOptions {
            boundary_tolerance_seconds,
            frame_step_seconds,
        };
        self.inner
            .compare_tiers(bundle_id, tier_a_id, tier_b_id, Some(opts))
            .map(|inner| PyAgreementReport { inner })
            .map_err(engine_err_to_py)
    }

    /// Counts a bundle's targets by status — the campaign progress readout.
    fn target_progress(&self, bundle_id: i64) -> PyResult<PyProgressCounts> {
        self.inner
            .target_progress(bundle_id)
            .map(|inner| PyProgressCounts { inner })
            .map_err(engine_err_to_py)
    }

    /// The next target on `bundle_id` whose status is in `statuses` (time
    /// order) — the work-queue navigator. `None` when none match.
    fn next_target(&self, bundle_id: i64, statuses: Vec<String>) -> PyResult<Option<PyTarget>> {
        self.inner
            .next_target(bundle_id, &statuses)
            .map(|opt| opt.map(|inner| PyTarget { inner }))
            .map_err(engine_err_to_py)
    }

    /// Project-wide target counts by status — the QA dashboard completeness
    /// headline (slice S6).
    fn project_target_progress(&self) -> PyResult<PyProgressCounts> {
        self.inner
            .project_target_progress()
            .map(|inner| PyProgressCounts { inner })
            .map_err(engine_err_to_py)
    }

    /// Per-annotator assignment counts across the project, in annotator order.
    fn assignment_progress(&self) -> PyResult<Vec<PyAnnotatorProgress>> {
        self.inner
            .assignment_progress()
            .map(|xs| {
                xs.into_iter()
                    .map(|inner| PyAnnotatorProgress { inner })
                    .collect()
            })
            .map_err(engine_err_to_py)
    }

    /// QA findings for a tier: out-of-vocabulary / missing labels and (interval
    /// tiers) overlapping pairs.
    fn tier_qa(&self, tier_id: i64) -> PyResult<PyQaReport> {
        self.inner
            .tier_qa(tier_id)
            .map(|inner| PyQaReport { inner })
            .map_err(engine_err_to_py)
    }

    /// Pairwise inter-annotator agreement over every `"<base> [annotator]"`
    /// tier on `bundle_id`.
    fn agreement_summary(
        &self,
        bundle_id: i64,
        base_tier_name: &str,
    ) -> PyResult<Vec<PyPairAgreement>> {
        self.inner
            .agreement_summary(bundle_id, base_tier_name)
            .map(|xs| {
                xs.into_iter()
                    .map(|inner| PyPairAgreement { inner })
                    .collect()
            })
            .map_err(engine_err_to_py)
    }

    /// Snapshots the current rubric under its current version (slice S6b),
    /// recording `note`. Re-publishing the same version updates that snapshot;
    /// bump `set_rubric(version+1)` to start a new one. Returns the snapshot.
    #[pyo3(signature = (note=None))]
    fn publish_rubric_version(&self, note: Option<&str>) -> PyResult<PyRubricVersion> {
        self.inner
            .publish_rubric_version(note)
            .map(|inner| PyRubricVersion { inner })
            .map_err(engine_err_to_py)
    }

    /// Lists published rubric versions in version order.
    fn rubric_versions(&self) -> PyResult<Vec<PyRubricVersion>> {
        self.inner
            .rubric_versions()
            .map(|xs| {
                xs.into_iter()
                    .map(|inner| PyRubricVersion { inner })
                    .collect()
            })
            .map_err(engine_err_to_py)
    }

    /// Recalls a published rubric version's full snapshot, or `None`.
    fn get_rubric_version(&self, version: i64) -> PyResult<Option<PyRubricVersion>> {
        self.inner
            .get_rubric_version(version)
            .map(|opt| opt.map(|inner| PyRubricVersion { inner }))
            .map_err(engine_err_to_py)
    }

    /// Reports how the current rubric differs from a published `version`: per
    /// tier, vocabulary added/removed and current annotations now out of
    /// vocabulary (needing revisiting). Raises if `version` was never published.
    fn rubric_impact(&self, version: i64) -> PyResult<Vec<PyTierImpact>> {
        self.inner
            .rubric_impact(version)
            .map(|xs| xs.into_iter().map(|inner| PyTierImpact { inner }).collect())
            .map_err(engine_err_to_py)
    }

    /// Records a PI lab-notebook entry about `target_type` (slice S7). `kind`
    /// defaults to `"observation"`. Returns the new entry id.
    #[pyo3(signature = (target_type, text, *, kind=None, measurement=None, bundle_id=None))]
    fn add_notebook_entry(
        &self,
        target_type: &str,
        text: &str,
        kind: Option<String>,
        measurement: Option<String>,
        bundle_id: Option<i64>,
    ) -> PyResult<i64> {
        let spec = sadda_engine::NotebookEntrySpec {
            target_type: target_type.to_owned(),
            kind,
            text: text.to_owned(),
            measurement,
            bundle_id,
        };
        self.inner
            .add_notebook_entry(&spec)
            .map_err(engine_err_to_py)
    }

    /// Lists notebook entries (newest first), optionally restricted to a
    /// `target_type`.
    #[pyo3(signature = (target_type=None))]
    fn notebook_entries(&self, target_type: Option<&str>) -> PyResult<Vec<PyNotebookEntry>> {
        self.inner
            .notebook_entries(target_type)
            .map(|xs| {
                xs.into_iter()
                    .map(|inner| PyNotebookEntry { inner })
                    .collect()
            })
            .map_err(engine_err_to_py)
    }

    /// Reads a notebook entry by id, or `None`.
    fn get_notebook_entry(&self, id: i64) -> PyResult<Option<PyNotebookEntry>> {
        self.inner
            .get_notebook_entry(id)
            .map(|opt| opt.map(|inner| PyNotebookEntry { inner }))
            .map_err(engine_err_to_py)
    }

    /// Edits a notebook entry's text and measurement.
    #[pyo3(signature = (id, text, measurement=None))]
    fn update_notebook_entry(
        &self,
        id: i64,
        text: &str,
        measurement: Option<&str>,
    ) -> PyResult<()> {
        self.inner
            .update_notebook_entry(id, text, measurement)
            .map_err(engine_err_to_py)
    }

    /// Deletes a notebook entry by id (idempotent).
    fn delete_notebook_entry(&self, id: i64) -> PyResult<()> {
        self.inner
            .delete_notebook_entry(id)
            .map_err(engine_err_to_py)
    }

    /// Promotes a notebook entry into a criterion (creates it + links the
    /// entry). Returns the criterion.
    fn promote_entry_to_criterion(
        &self,
        entry_id: i64,
        name: &str,
        kind: &str,
        body: &str,
        target_tier: &str,
    ) -> PyResult<PyCriterion> {
        self.inner
            .promote_entry_to_criterion(entry_id, name, kind, body, target_tier)
            .map(|inner| PyCriterion { inner })
            .map_err(engine_err_to_py)
    }

    /// Promotes a notebook entry into rubric-tier guidance: appends its text to
    /// the `target_type` tier's rubric description and links the entry.
    fn promote_entry_to_rubric_guidance(&self, entry_id: i64) -> PyResult<()> {
        self.inner
            .promote_entry_to_rubric_guidance(entry_id)
            .map_err(engine_err_to_py)
    }

    /// Runs a `structured` criterion against a bundle, (re)writing its
    /// proposals onto the preview tier. Returns the proposal count.
    /// `python` criteria are run via `sadda.criteria.run_criterion`.
    fn run_criterion(&self, id: i64, bundle_id: i64) -> PyResult<usize> {
        self.inner
            .run_criterion(id, bundle_id)
            .map_err(engine_err_to_py)
    }

    /// (Re)writes proposals onto the preview tier `"<target> (auto)"`,
    /// replacing prior ones. `proposals` is a list of
    /// `(start, end_or_None, label_or_None)` tuples — `end=None` for a point.
    /// `processing_run_id` stamps each proposal with its provenance link (the
    /// row returned by `record_criterion_run`); pass `None` for unattributed
    /// proposals. Used by the python-escape criterion executor.
    #[pyo3(signature = (bundle_id, target_tier, proposals, processing_run_id=None))]
    fn set_proposals(
        &self,
        bundle_id: i64,
        target_tier: &str,
        proposals: Vec<(f64, Option<f64>, Option<String>)>,
        processing_run_id: Option<i64>,
    ) -> PyResult<usize> {
        let props: Vec<sadda_engine::Proposal> = proposals
            .into_iter()
            .map(|(start, end, label)| sadda_engine::Proposal { start, end, label })
            .collect();
        self.inner
            .set_proposals(bundle_id, target_tier, &props, processing_run_id)
            .map_err(engine_err_to_py)
    }

    /// Records a `processing_run` of kind `criterion_run` for an execution of
    /// criterion `criterion_id` against `bundle_id`, returning its id. The
    /// python-escape executor calls this before `set_proposals` so a python
    /// criterion's run is traced exactly like a structured one.
    fn record_criterion_run(&self, criterion_id: i64, bundle_id: i64) -> PyResult<i64> {
        let crit = self
            .inner
            .get_criterion(criterion_id)
            .map_err(engine_err_to_py)?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "no criterion with id {criterion_id}"
                ))
            })?;
        self.inner
            .record_criterion_run(&crit, bundle_id)
            .map_err(engine_err_to_py)
    }

    /// Promotes all proposals on `"<target> (auto)"` to the target tier
    /// (validated against its rubric), then clears the preview tier. Returns
    /// the number promoted.
    fn accept_proposals(&self, bundle_id: i64, target_tier: &str) -> PyResult<usize> {
        self.inner
            .accept_proposals(bundle_id, target_tier)
            .map_err(engine_err_to_py)
    }

    /// Discards all proposals on `"<target> (auto)"`. Returns the count.
    fn clear_proposals(&self, bundle_id: i64, target_tier: &str) -> PyResult<usize> {
        self.inner
            .clear_proposals(bundle_id, target_tier)
            .map_err(engine_err_to_py)
    }

    /// Writes a `continuous_numeric` Parquet sidecar from a 1-D float64
    /// NumPy array and inserts the matching `DerivedSignal` row. Returns
    /// the new DerivedSignal id. Errors with TypeError-style messages if
    /// the tier isn't `continuous_numeric` or already has a sidecar.
    fn write_continuous_numeric(
        &self,
        tier_id: i64,
        samples: PyReadonlyArray1<'_, f64>,
        sample_rate_hz: f64,
    ) -> PyResult<i64> {
        let slice = samples
            .as_slice()
            .map_err(|e| PyValueError::new_err(format!("samples must be contiguous: {e}")))?;
        self.inner
            .write_continuous_numeric(tier_id, slice, sample_rate_hz)
            .map_err(engine_err_to_py)
    }

    /// Reads a `continuous_numeric` sidecar back into a 1-D float64 NumPy
    /// array.
    fn read_continuous_numeric<'py>(
        &self,
        py: Python<'py>,
        tier_id: i64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let vec = self
            .inner
            .read_continuous_numeric(tier_id)
            .map_err(engine_err_to_py)?;
        Ok(vec.into_pyarray(py))
    }

    /// Writes a `continuous_vector` Parquet sidecar from a 2-D float64
    /// NumPy array of shape `[n_frames, n_dims]`.
    fn write_continuous_vector(
        &self,
        tier_id: i64,
        frames: PyReadonlyArray2<'_, f64>,
        sample_rate_hz: f64,
    ) -> PyResult<i64> {
        let view = frames.as_array();
        self.inner
            .write_continuous_vector(tier_id, view, sample_rate_hz)
            .map_err(engine_err_to_py)
    }

    /// Reads a `continuous_vector` sidecar back into a 2-D float64 NumPy
    /// array.
    fn read_continuous_vector<'py>(
        &self,
        py: Python<'py>,
        tier_id: i64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr: Array2<f64> = self
            .inner
            .read_continuous_vector(tier_id)
            .map_err(engine_err_to_py)?;
        Ok(arr.into_pyarray(py))
    }

    /// Writes a `categorical_sampled` Parquet sidecar from a list of
    /// strings.
    fn write_categorical_sampled(
        &self,
        tier_id: i64,
        labels: Vec<String>,
        sample_rate_hz: f64,
    ) -> PyResult<i64> {
        self.inner
            .write_categorical_sampled(tier_id, &labels, sample_rate_hz)
            .map_err(engine_err_to_py)
    }

    /// Reads a `categorical_sampled` sidecar back into a list of strings.
    fn read_categorical_sampled(&self, tier_id: i64) -> PyResult<Vec<String>> {
        self.inner
            .read_categorical_sampled(tier_id)
            .map_err(engine_err_to_py)
    }

    /// Returns the DerivedSignal row for a tier, or None if no sidecar
    /// has been written yet.
    fn derived_signal(&self, tier_id: i64) -> PyResult<Option<PyDerivedSignal>> {
        self.inner
            .derived_signal(tier_id)
            .map(|opt| opt.map(|inner| PyDerivedSignal { inner }))
            .map_err(engine_err_to_py)
    }

    /// Returns the absolute filesystem path of a dense tier's Parquet
    /// sidecar (as a string), or None if no sidecar has been written yet.
    /// Use with `polars.scan_parquet(path)` for zero-engine-API reads.
    fn dense_path(&self, tier_id: i64) -> PyResult<Option<String>> {
        let opt = self.inner.dense_path(tier_id).map_err(engine_err_to_py)?;
        Ok(opt.map(|p| p.to_string_lossy().into_owned()))
    }

    /// Imports a Praat TextGrid into `bundle_id`. Each Praat tier becomes a
    /// new Tier row (interval or point); each annotation becomes an
    /// `annotation_interval` / `annotation_point` row. JSON sentinels in
    /// labels are decoded back into the `extra` field. Returns the list of
    /// new tier IDs in import order. Records a `processing_run` row for
    /// audit provenance.
    fn import_textgrid(&self, path: PathBuf, bundle_id: i64) -> PyResult<Vec<i64>> {
        self.inner
            .import_textgrid(path, bundle_id)
            .map_err(engine_err_to_py)
    }

    /// Writes a Praat TextGrid for `bundle_id`'s sparse tiers to `path`.
    /// If `tier_ids` is given, only those tiers are exported. Dense tiers
    /// (continuous_numeric / vector / categorical_sampled) are skipped.
    /// Reference tiers are exported as IntervalTiers with a degenerate
    /// `[0.0, 0.001]` time span plus a JSON sentinel carrying their
    /// `(target_kind, target_id)`.
    #[pyo3(signature = (bundle_id, path, *, tier_ids=None))]
    fn export_textgrid(
        &self,
        bundle_id: i64,
        path: PathBuf,
        tier_ids: Option<Vec<i64>>,
    ) -> PyResult<()> {
        self.inner
            .export_textgrid(bundle_id, path, tier_ids.as_deref())
            .map_err(engine_err_to_py)
    }

    /// Imports an ELAN .eaf into `bundle_id`. Tier hierarchy is preserved
    /// via EAF's `PARENT_REF` ↔ `tier.parent_id`. Point tiers are
    /// recovered from degenerate `[t, t+1ms]` alignable annotations via
    /// a ≤2ms heuristic. Reference tiers (`Symbolic_Association`
    /// linguistic type) come back as `reference` tiers. Returns the new
    /// tier IDs in import order (parents first per topological sort).
    /// Records a `processing_run` row for audit provenance.
    fn import_eaf(&self, path: PathBuf, bundle_id: i64) -> PyResult<Vec<i64>> {
        self.inner
            .import_eaf(path, bundle_id)
            .map_err(engine_err_to_py)
    }

    /// Writes an ELAN .eaf (EAF 2.8) for `bundle_id`'s sparse tiers to
    /// `path`. If `tier_ids` is given, only those tiers are exported.
    /// Dense tiers (continuous_numeric / vector / categorical_sampled)
    /// are skipped. Interval tiers with parents use the `Included_In`
    /// stereotype; reference tiers become `REF_ANNOTATION` tiers with
    /// the `Symbolic_Association` stereotype + a JSON sentinel encoding
    /// `(target_kind, target_id)`.
    #[pyo3(signature = (bundle_id, path, *, tier_ids=None))]
    fn export_eaf(
        &self,
        bundle_id: i64,
        path: PathBuf,
        tier_ids: Option<Vec<i64>>,
    ) -> PyResult<()> {
        self.inner
            .export_eaf(bundle_id, path, tier_ids.as_deref())
            .map_err(engine_err_to_py)
    }

    /// Writes a flat CSV of `bundle_id`'s sparse annotations to `path`: one
    /// tidy row per annotation across all interval / point / reference tiers
    /// (the shape pandas / polars / R expect). If `tier_ids` is given, only
    /// those tiers are exported. Dense tiers (continuous_* / categorical_*)
    /// are skipped — their samples live in Parquet sidecars (see `query`).
    #[pyo3(signature = (bundle_id, path, *, tier_ids=None))]
    fn export_csv(
        &self,
        bundle_id: i64,
        path: PathBuf,
        tier_ids: Option<Vec<i64>>,
    ) -> PyResult<()> {
        self.inner
            .export_csv(bundle_id, path, tier_ids.as_deref())
            .map_err(engine_err_to_py)
    }

    /// Writes a structured JSON document of `bundle_id`'s sparse annotations
    /// to `path`: bundle metadata plus a `tiers` array, each tier carrying
    /// its native rows (faithful, unlike the flattened CSV). If `tier_ids`
    /// is given, only those tiers are exported. Dense tiers are skipped.
    #[pyo3(signature = (bundle_id, path, *, tier_ids=None))]
    fn export_json(
        &self,
        bundle_id: i64,
        path: PathBuf,
        tier_ids: Option<Vec<i64>>,
    ) -> PyResult<()> {
        self.inner
            .export_json(bundle_id, path, tier_ids.as_deref())
            .map_err(engine_err_to_py)
    }

    /// Imports a flat CSV (as written by `export_csv`) into `bundle_id`. Rows
    /// are grouped into new tiers by `(tier_name, tier_type)`; interval /
    /// point rows become annotations. Returns the new tier IDs and records a
    /// `processing_run`. v1 imports only interval + point tiers; `status`,
    /// `parent_annotation_id`, and `processing_run_id` are dropped (times,
    /// label, note, and extra are honoured).
    fn import_csv(&self, path: PathBuf, bundle_id: i64) -> PyResult<Vec<i64>> {
        self.inner
            .import_csv(path, bundle_id)
            .map_err(engine_err_to_py)
    }

    /// Imports a structured JSON document (as written by `export_json`) into
    /// `bundle_id`. Each `tiers[]` entry becomes a new tier; its rows become
    /// annotations. Returns the new tier IDs and records a `processing_run`.
    /// Same v1 limits as `import_csv`.
    fn import_json(&self, path: PathBuf, bundle_id: i64) -> PyResult<Vec<i64>> {
        self.inner
            .import_json(path, bundle_id)
            .map_err(engine_err_to_py)
    }

    fn __repr__(&self) -> String {
        format!("Project(root={:?})", self.root())
    }
}

/// Returns the underlying engine version string.
#[gen_stub_pyfunction]
#[pyfunction]
fn version() -> &'static str {
    sadda_engine::version()
}

/// Returns the highest corpus-database schema version this engine supports.
#[gen_stub_pyfunction]
#[pyfunction]
fn schema_version() -> i64 {
    sadda_engine::schema_version()
}

/// Loads a WAV file from disk. Returns a sadda.Audio.
#[gen_stub_pyfunction]
#[pyfunction]
fn load_wav(path: PathBuf) -> PyResult<PyAudio> {
    let audio = sadda_engine::Audio::from_wav_path(&path).map_err(engine_err_to_py)?;
    Ok(PyAudio { inner: audio })
}

/// Reads only a WAV file's header (no samples decoded) to learn its size.
/// Returns a sadda.AudioProbe — cheap regardless of file length. Use it to
/// decide whether a file is large enough to warrant splitting before loading.
#[gen_stub_pyfunction]
#[pyfunction]
fn probe_wav(path: PathBuf) -> PyResult<PyAudioProbe> {
    let inner = sadda_engine::Audio::probe(&path).map_err(engine_err_to_py)?;
    Ok(PyAudioProbe { inner })
}

/// One interval row: `(start_seconds, end_seconds, label)`.
type IntervalRow = (f64, f64, String);
/// A named interval tier and its rows: `(tier_name, [row, ...])`.
type NamedIntervalTier = (String, Vec<IntervalRow>);

/// Parse a Praat TextGrid file into its interval tiers, as a list of
/// `(tier_name, [(start_seconds, end_seconds, label), ...])`. Point tiers are
/// skipped (forced-alignment output is interval-only). Labels are preserved
/// verbatim — the empty string and modeled-silence marks (`sil`/`sp`) included.
/// Reuses the engine's TextGrid reader; used to turn an external aligner's
/// (e.g. MFA) TextGrid output into a `sadda.align.Alignment`.
#[gen_stub_pyfunction]
#[pyfunction]
fn parse_textgrid_intervals(path: PathBuf) -> PyResult<Vec<NamedIntervalTier>> {
    let file = sadda_engine::io::textgrid::read(&path).map_err(engine_err_to_py)?;
    let mut out = Vec::new();
    for tier in file.tiers {
        if let sadda_engine::io::textgrid::TextGridTier::Interval(t) = tier {
            let rows = t
                .intervals
                .into_iter()
                .map(|e| (e.xmin, e.xmax, e.text))
                .collect();
            out.push((t.name, rows));
        }
    }
    Ok(out)
}

/// Syllabify a sequence of IPA phone labels into `[start, end)` phone-index
/// ranges, one per syllable (Sonority Sequencing + Maximal Onset; see
/// `sadda_engine::syllable`). Pass one word's phones — syllabification is
/// word-internal.
#[gen_stub_pyfunction]
#[pyfunction]
fn syllabify(phones: Vec<String>) -> Vec<(usize, usize)> {
    sadda_engine::syllable::syllabify(&phones)
}

/// Header-only summary of a WAV file (see `sadda.probe_wav`): its size without
/// the cost of decoding. Lets a caller gauge a file's in-memory footprint
/// before loading it.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "AudioProbe", frozen)]
struct PyAudioProbe {
    inner: sadda_engine::AudioProbe,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAudioProbe {
    /// Sample rate in Hz.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }
    /// Number of channels.
    #[getter]
    fn channels(&self) -> u16 {
        self.inner.channels
    }
    /// Number of frames (samples per channel).
    #[getter]
    fn n_frames(&self) -> u64 {
        self.inner.n_frames
    }
    /// Duration in seconds.
    #[getter]
    fn duration_seconds(&self) -> f64 {
        self.inner.duration_seconds
    }
    /// Bytes a full decode would occupy (interleaved f32): the RAM cost of
    /// loading this file whole.
    #[getter]
    fn decoded_bytes(&self) -> u64 {
        self.inner.decoded_bytes
    }

    fn __repr__(&self) -> String {
        format!(
            "AudioProbe(sample_rate={}, channels={}, n_frames={}, duration_seconds={:.4}, \
             decoded_bytes={})",
            self.inner.sample_rate,
            self.inner.channels,
            self.inner.n_frames,
            self.inner.duration_seconds,
            self.inner.decoded_bytes,
        )
    }
}

/// Creates a new sadda project at `path` (which must not already exist).
/// Returns a sadda.Project handle ready for `.add_speaker(...)` /
/// `.add_session(...)` / `.add_bundle(...)` calls.
#[gen_stub_pyfunction]
#[pyfunction]
fn new_project(path: PathBuf, name: &str) -> PyResult<PyProject> {
    sadda_engine::Project::create(&path, name)
        .map(|inner| PyProject { inner })
        .map_err(engine_err_to_py)
}

/// Opens an existing sadda project at `path`. Applies any pending schema
/// migrations first, writing a `corpus.db.bak.<old_version>` backup.
#[gen_stub_pyfunction]
#[pyfunction]
fn open_project(path: PathBuf) -> PyResult<PyProject> {
    sadda_engine::Project::open(&path)
        .map(|inner| PyProject { inner })
        .map_err(engine_err_to_py)
}

/// Estimates f0 over an Audio via time-domain autocorrelation.
///
/// Returns `(times, frequencies)` as a 2-tuple of NumPy arrays:
/// `times` is float64 in seconds, `frequencies` is float32 in Hz.
#[gen_stub_pyfunction]
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
        ..Default::default()
    };
    let frames = sadda_engine::autocorrelation(&audio.inner, &config);
    let times: Vec<f64> = frames.iter().map(|f| f.time_seconds).collect();
    let freqs: Vec<f32> = frames.iter().map(|f| f.frequency_hz.value()).collect();
    (times.into_pyarray(py), freqs.into_pyarray(py))
}

/// Hann window: `0.5 * (1 - cos(2π n / (N-1)))`.
#[gen_stub_pyfunction]
#[pyfunction]
fn hann<'py>(py: Python<'py>, n: usize) -> Bound<'py, PyArray1<f32>> {
    sadda_engine::dsp::hann(n).into_pyarray(py)
}

/// Hamming window: `0.54 - 0.46 * cos(2π n / (N-1))`.
#[gen_stub_pyfunction]
#[pyfunction]
fn hamming<'py>(py: Python<'py>, n: usize) -> Bound<'py, PyArray1<f32>> {
    sadda_engine::dsp::hamming(n).into_pyarray(py)
}

/// Blackman window:
/// `0.42 - 0.5*cos(2π n / (N-1)) + 0.08*cos(4π n / (N-1))`.
#[gen_stub_pyfunction]
#[pyfunction]
fn blackman<'py>(py: Python<'py>, n: usize) -> Bound<'py, PyArray1<f32>> {
    sadda_engine::dsp::blackman(n).into_pyarray(py)
}

/// Gaussian window of length `n` with standard deviation `sigma` (in samples).
#[gen_stub_pyfunction]
#[pyfunction]
fn gaussian<'py>(py: Python<'py>, n: usize, sigma: f32) -> Bound<'py, PyArray1<f32>> {
    sadda_engine::dsp::gaussian(n, sigma).into_pyarray(py)
}

/// Kaiser window of length `n` with shape parameter `beta`.
#[gen_stub_pyfunction]
#[pyfunction]
fn kaiser<'py>(py: Python<'py>, n: usize, beta: f32) -> Bound<'py, PyArray1<f32>> {
    sadda_engine::dsp::kaiser(n, beta).into_pyarray(py)
}

/// Short-time Fourier transform of a real-valued 1-D float32 signal.
///
/// Returns the complex spectrogram with shape `(n_frames, n_freq_bins)` where
/// `n_freq_bins = frame_size / 2 + 1` (the unique half of the spectrum for
/// real input). If `window` is omitted, a Hann window of length `frame_size`
/// is used (matches `scipy.signal.stft`'s default).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (samples, frame_size, hop_size, *, window=None))]
fn stft<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'_, f32>,
    frame_size: usize,
    hop_size: usize,
    window: Option<PyReadonlyArray1<'_, f32>>,
) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
    if frame_size == 0 {
        return Err(PyValueError::new_err("frame_size must be > 0"));
    }
    if hop_size == 0 {
        return Err(PyValueError::new_err("hop_size must be > 0"));
    }
    let samples_slice = samples
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("samples must be contiguous: {e}")))?;
    let owned_window: Vec<f32>;
    let window_slice: &[f32] = match window {
        Some(w) => {
            let w_slice = w
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("window must be contiguous: {e}")))?;
            if w_slice.len() != frame_size {
                return Err(PyValueError::new_err(format!(
                    "window length {} does not match frame_size {frame_size}",
                    w_slice.len()
                )));
            }
            // Borrowed copy — the readonly array's slice is short-lived.
            owned_window = w_slice.to_vec();
            owned_window.as_slice()
        }
        None => {
            owned_window = sadda_engine::dsp::hann(frame_size);
            owned_window.as_slice()
        }
    };
    let (data, shape) = sadda_engine::dsp::stft(samples_slice, window_slice, hop_size);
    // `data` is row-major (n_frames, n_freq_bins).
    let arr = Array2::from_shape_vec((shape.n_frames, shape.n_freq_bins), data)
        .map_err(|e| PyRuntimeError::new_err(format!("STFT reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Power spectrogram of a real-valued signal: `|X|²` of the STFT, in shape
/// `(n_freq_bins, n_frames)`. If `window` is omitted, a Hann window of
/// length `frame_size` is used.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (samples, frame_size, hop_size, *, window=None))]
fn spectrogram<'py>(
    py: Python<'py>,
    samples: PyReadonlyArray1<'_, f32>,
    frame_size: usize,
    hop_size: usize,
    window: Option<PyReadonlyArray1<'_, f32>>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if frame_size == 0 {
        return Err(PyValueError::new_err("frame_size must be > 0"));
    }
    if hop_size == 0 {
        return Err(PyValueError::new_err("hop_size must be > 0"));
    }
    let samples_slice = samples
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("samples must be contiguous: {e}")))?;
    let owned_window: Vec<f32>;
    let window_slice: &[f32] = match window {
        Some(w) => {
            let w_slice = w
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("window must be contiguous: {e}")))?;
            if w_slice.len() != frame_size {
                return Err(PyValueError::new_err(format!(
                    "window length {} does not match frame_size {frame_size}",
                    w_slice.len()
                )));
            }
            owned_window = w_slice.to_vec();
            owned_window.as_slice()
        }
        None => {
            owned_window = sadda_engine::dsp::hann(frame_size);
            owned_window.as_slice()
        }
    };
    let (data, shape) = sadda_engine::dsp::stft(samples_slice, window_slice, hop_size);
    let p = sadda_engine::dsp::power_spectrogram(&data, shape);
    let arr = Array2::from_shape_vec((shape.n_freq_bins, shape.n_frames), p)
        .map_err(|e| PyRuntimeError::new_err(format!("spectrogram reshape failed: {e}")))?;
    Ok(arr.into_pyarray(py))
}

/// Per-frame intensity over an [`Audio`]: returns `(times, rms, db_fs)` as
/// three NumPy arrays. `times` is float64 seconds at frame centres; `rms` is
/// float32 linear amplitude; `db_fs` is float32 dB relative to digital
/// full-scale (clamped to -200 dB on silence). dB-SPL (Praat convention)
/// arrives in a later slice once microphone calibration is wired through.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (audio, *, frame_size_seconds=0.030, hop_seconds=0.010))]
#[allow(clippy::type_complexity)]
fn intensity<'py>(
    py: Python<'py>,
    audio: &PyAudio,
    frame_size_seconds: f32,
    hop_seconds: f32,
) -> (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
) {
    let mono: Vec<f32> = audio.inner.mono_samples().collect();
    let frames = sadda_engine::dsp::intensity(
        &mono,
        audio.inner.sample_rate,
        frame_size_seconds,
        hop_seconds,
    );
    let times: Vec<f64> = frames.iter().map(|f| f.time_seconds).collect();
    let rms: Vec<f32> = frames.iter().map(|f| f.rms).collect();
    let db_fs: Vec<f32> = frames.iter().map(|f| f.db_fs.value()).collect();
    (
        times.into_pyarray(py),
        rms.into_pyarray(py),
        db_fs.into_pyarray(py),
    )
}

/// One frame of formant output. Variable-length `frequencies` /
/// `bandwidths` per frame — frames where the LPC root-finder didn't return
/// enough valid roots in the formant range are honestly empty rather than
/// NaN-padded.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "FormantFrame", frozen)]
pub(crate) struct PyFormantFrame {
    pub(crate) inner: sadda_engine::dsp::FormantFrame,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyFormantFrame {
    /// Time at the centre of the analysis frame, in seconds.
    #[getter]
    fn time_seconds(&self) -> f64 {
        self.inner.time_seconds
    }
    /// Formant frequencies in Hz, ascending.
    #[getter]
    fn frequencies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let hz: Vec<f32> = self.inner.frequencies.iter().map(|h| h.value()).collect();
        hz.into_pyarray(py)
    }
    /// Bandwidths in Hz, co-indexed with `frequencies`.
    #[getter]
    fn bandwidths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let hz: Vec<f32> = self.inner.bandwidths.iter().map(|h| h.value()).collect();
        hz.into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "FormantFrame(t={:.3}s, n_formants={}, frequencies={:?})",
            self.inner.time_seconds,
            self.inner.frequencies.len(),
            self.inner.frequencies,
        )
    }
}

fn parse_pitch_method(s: &str) -> PyResult<sadda_engine::pitch::PitchMethod> {
    match s {
        "autocorrelation" => Ok(sadda_engine::pitch::PitchMethod::Autocorrelation),
        "windowed_autocorrelation" => Ok(sadda_engine::pitch::PitchMethod::WindowedAutocorrelation),
        "boersma" => Ok(sadda_engine::pitch::PitchMethod::Boersma),
        "yin" => Ok(sadda_engine::pitch::PitchMethod::Yin),
        "pyin" => Ok(sadda_engine::pitch::PitchMethod::PYin),
        "swipe" => Ok(sadda_engine::pitch::PitchMethod::Swipe),
        other => Err(PyValueError::new_err(format!(
            "unknown pitch method {other:?}; expected 'autocorrelation', 'windowed_autocorrelation', 'boersma', 'yin', 'pyin', or 'swipe'"
        ))),
    }
}

fn parse_lpc_method(s: &str) -> PyResult<sadda_engine::dsp::LpcMethod> {
    match s {
        "autocorrelation" => Ok(sadda_engine::dsp::LpcMethod::Autocorrelation),
        "burg" => Ok(sadda_engine::dsp::LpcMethod::Burg),
        other => Err(PyValueError::new_err(format!(
            "unknown LPC method {other:?}; expected 'autocorrelation' or 'burg'"
        ))),
    }
}

fn parse_mfcc_method(s: &str) -> PyResult<sadda_engine::dsp::MfccMethod> {
    match s {
        "librosa" => Ok(sadda_engine::dsp::MfccMethod::Librosa),
        "kaldi" => Ok(sadda_engine::dsp::MfccMethod::Kaldi),
        "praat" => Ok(sadda_engine::dsp::MfccMethod::Praat),
        other => Err(PyValueError::new_err(format!(
            "unknown MFCC method {other:?}; expected 'librosa', 'kaldi', or 'praat'"
        ))),
    }
}

/// Estimates f0 with a voicing decision and returns `(times, frequencies,
/// voicing)` as three NumPy arrays. `times` is float64 seconds at frame
/// centres; `frequencies` is float32 Hz; `voicing` is float32 in `[0, 1]`.
///
/// `method` selects the pitch tracker. Two algorithmic families —
/// autocorrelation and cumulative-mean-normalized-difference — covering
/// both Praat-faithful and librosa-faithful expectations:
///
/// **Autocorrelation family:**
/// - `"boersma"` (**default**) — **faithful Boersma 1993 / Praat `Sound:
///   To Pitch (ac)…`** with `very_accurate = false`. Multi-candidate
///   per-frame detection + Viterbi path-finder with octave-cost /
///   octave-jump-cost / voiced-unvoiced-cost terms. Robust to halving /
///   doubling / transient errors; Praat-validated. The default because it
///   does not latch onto subharmonics of clean tones the way the simpler
///   trackers below do (e.g. 150→75, 250→83.3).
/// - `"windowed_autocorrelation"` — adopts Boersma 1993's window-correction
///   idea (divides windowed-signal autocorrelation by window
///   autocorrelation); fast single-peak tracker, but **prone to
///   subharmonic / octave-down errors** (no octave cost or path-finding).
/// - `"autocorrelation"` — naive time-domain autocorrelation (Phase-0
///   tracker; what `sadda.dsp.f0(...)` calls).
///
/// **Cumulative-mean-normalized-difference family:**
/// - `"yin"` — de Cheveigné & Kawahara 2002. Difference function +
///   CMNDF + absolute threshold. Simple and fast; independent
///   algorithmic family from autocorrelation, useful for
///   cross-validation against `"boersma"`.
/// - `"pyin"` — Mauch & Dixon 2014, librosa's default. Probabilistic
///   YIN with a beta-prior distribution over thresholds plus an HMM
///   smoothing pass. librosa-validated.
/// - `"swipe"` — Camacho & Harris 2008 SWIPE' (prime variant). Spectral
///   method (a third algorithmic family): matches the `sqrt`-loudness
///   ERB-scale spectrum against prime-harmonic cosine kernels. Validated
///   against the author's own MATLAB run under Octave.
///
/// `voicing_threshold` is informational here: the function returns voicing
/// values for every frame so callers can apply their own threshold.
///
/// `range_mode` selects how the analysis floor/ceiling are chosen:
/// - `"manual"` (default) — use `min_freq_hz` / `max_freq_hz` as given.
/// - `"two_pass"` — adapt them to the recording via De Looze & Hirst (2008):
///   analyse over a wide range, then set `floor = 0.75·q25`, `ceiling = 1.5·q75`
///   from the voiced-f0 quartiles, and re-track. `min_freq_hz` / `max_freq_hz`
///   are then ignored. See `estimate_pitch_range`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    audio, *,
    frame_size_seconds=0.030, hop_size_seconds=0.010,
    min_freq_hz=75.0, max_freq_hz=500.0,
    method="boersma",
    voicing_threshold=0.45,
    range_mode="manual",
))]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn voiced_pitch<'py>(
    py: Python<'py>,
    audio: &PyAudio,
    frame_size_seconds: f32,
    hop_size_seconds: f32,
    min_freq_hz: f32,
    max_freq_hz: f32,
    method: &str,
    voicing_threshold: f32,
    range_mode: &str,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
)> {
    let pitch_method = parse_pitch_method(method)?;
    let config = sadda_engine::pitch::PitchConfig {
        frame_size_seconds,
        hop_size_seconds,
        min_freq_hz,
        max_freq_hz,
        voicing_threshold,
        ..sadda_engine::pitch::PitchConfig::default()
    };
    let frames = match range_mode {
        "manual" => sadda_engine::pitch::pitch(&audio.inner, &config, pitch_method),
        "two_pass" => sadda_engine::pitch::two_pass_pitch(&audio.inner, &config, pitch_method),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown range_mode {other:?}; expected \"manual\" or \"two_pass\""
            )));
        }
    };
    let times: Vec<f64> = frames.iter().map(|f| f.time_seconds).collect();
    let freqs: Vec<f32> = frames.iter().map(|f| f.frequency_hz.value()).collect();
    let voicing: Vec<f32> = frames.iter().map(|f| f.voicing).collect();
    Ok((
        times.into_pyarray(py),
        freqs.into_pyarray(py),
        voicing.into_pyarray(py),
    ))
}

/// Estimates a speaker-appropriate `(floor_hz, ceiling_hz)` from a recording via
/// the De Looze & Hirst (2008) two-pass rule: analyse f0 over a wide range, then
/// `floor = 0.75·q25`, `ceiling = 1.5·q75` from the first/third quartiles of the
/// voiced f0. Returns `None` when the recording has too few voiced frames.
///
/// This is the range `voiced_pitch(..., range_mode="two_pass")` derives before
/// its second pass; call it directly to inspect or reuse the range.
///
/// Reference: De Looze & Hirst (2008), "Detecting changes in key and range for
/// the automatic modelling and coding of intonation," Speech Prosody 2008,
/// <https://doi.org/10.21437/SpeechProsody.2008-32>.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (audio, *, method="boersma", voicing_threshold=0.45))]
fn estimate_pitch_range(
    audio: &PyAudio,
    method: &str,
    voicing_threshold: f32,
) -> PyResult<Option<(f32, f32)>> {
    let pitch_method = parse_pitch_method(method)?;
    let config = sadda_engine::pitch::PitchConfig {
        voicing_threshold,
        ..sadda_engine::pitch::PitchConfig::default()
    };
    Ok(sadda_engine::pitch::estimate_pitch_range(
        &audio.inner,
        &config,
        pitch_method,
    ))
}

/// Complete-pooling speaker pitch range: pool the voiced f0 of all of a
/// speaker's recordings into one distribution, then apply the De Looze & Hirst
/// (2008) rule once (`floor = 0.75·q25`, `ceiling = 1.5·q75`). Returns one
/// `(floor_hz, ceiling_hz)` for the speaker, or `None` if too few voiced frames
/// pooled across the recordings.
///
/// Each recording is analysed over a wide range first so the pooled quartiles
/// aren't clipped. This is the "complete pooling" baseline; its partial-pooling
/// companion is `speaker_pitch_ranges_shrunk` (empirical Bayes).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (audios, *, method="boersma", voicing_threshold=0.45))]
fn speaker_pitch_range_pooled(
    py: Python<'_>,
    audios: Vec<Py<PyAudio>>,
    method: &str,
    voicing_threshold: f32,
) -> PyResult<Option<(f32, f32)>> {
    let pitch_method = parse_pitch_method(method)?;
    let wide = sadda_engine::pitch::PitchConfig {
        min_freq_hz: sadda_engine::pitch::TWO_PASS_FLOOR_HZ,
        max_freq_hz: sadda_engine::pitch::TWO_PASS_CEILING_HZ,
        voicing_threshold,
        ..sadda_engine::pitch::PitchConfig::default()
    };
    let recordings: Vec<Vec<sadda_engine::pitch::PitchFrame>> = audios
        .iter()
        .map(|a| sadda_engine::pitch::pitch(&a.borrow(py).inner, &wide, pitch_method))
        .collect();
    Ok(sadda_engine::pitch::pooled_pitch_range(
        &recordings,
        voicing_threshold,
    ))
}

/// Computes per-frame formants over an [`Audio`] via LPC + polynomial
/// root-finding. Returns a list of `FormantFrame`s; each frame has
/// variable-length `frequencies` / `bandwidths` (honestly empty for frames
/// where the root-finder didn't return enough valid roots).
///
/// `method` selects the LPC estimator: `"burg"` (default; Praat
/// convention) or `"autocorrelation"`. `n_formants` is the maximum kept per
/// frame after filtering; `lpc_order = 2 · n_formants + 2` by default.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    audio, *,
    frame_size_seconds=0.025, hop_seconds=0.010,
    n_formants=5, pre_emphasis=0.97, lpc_order=None,
    method="burg",
    max_bandwidth_hz=1000.0, min_frequency_hz=50.0,
))]
#[allow(clippy::too_many_arguments)]
fn formants(
    audio: &PyAudio,
    frame_size_seconds: f32,
    hop_seconds: f32,
    n_formants: usize,
    pre_emphasis: f32,
    lpc_order: Option<usize>,
    method: &str,
    max_bandwidth_hz: f32,
    min_frequency_hz: f32,
) -> PyResult<Vec<PyFormantFrame>> {
    let lpc_method = parse_lpc_method(method)?;
    let config = sadda_engine::dsp::FormantsConfig {
        frame_size_seconds,
        hop_seconds,
        n_formants,
        pre_emphasis,
        lpc_order,
        lpc_method,
        max_bandwidth_hz,
        min_frequency_hz,
    };
    let mono: Vec<f32> = audio.inner.mono_samples().collect();
    let frames = sadda_engine::dsp::formants(&mono, audio.inner.sample_rate, &config);
    Ok(frames
        .into_iter()
        .map(|inner| PyFormantFrame { inner })
        .collect())
}

/// Computes Mel-Frequency Cepstral Coefficients over an [`Audio`]. Returns
/// a 2-D float32 NumPy array of shape `(n_frames, n_mfcc)`, frames-first.
///
/// "MFCC" is a family, not one algorithm — toolkits differ on log base,
/// windowing, mel scale, framing, etc. `method` picks the reference
/// implementation to reproduce faithfully:
///
/// - `"librosa"` (**default**) — faithful `librosa.feature.mfcc` (0.11):
///   Slaney mel scale + area norm, power spectrum, `10·log10` power-to-dB
///   with an 80 dB global floor, periodic Hann, `center=True` framing
///   (so `n_frames = 1 + n/hop`), orthonormal DCT-II.
/// - `"kaldi"` — faithful Kaldi `compute-mfcc-feats`: DC removal,
///   pre-emphasis 0.97, Povey window, power-of-two FFT, HTK mel scale with
///   unit-peak filters, natural-log energies, DCT-II, cepstral lifter (L=22),
///   `snip_edges` framing. Validated against torchaudio's kaldi-compliance.
/// - `"praat"` — Praat `Sound: To MFCC…` (Gaussian window, HTK mel,
///   unit-peak filters, un-normalised DCT, c0 in column 0). **Approximate**:
///   structurally faithful but not yet byte-exact (see `MfccMethod::Praat`).
///
/// Other params: `n_mels=40`, `n_mfcc=13`, `f_min=0`, `f_max=sr/2`, 25 ms
/// frame, 10 ms hop.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    audio, *,
    frame_size_seconds=0.025, hop_seconds=0.010,
    n_mels=40, n_mfcc=13,
    f_min=0.0, f_max=None,
    method="librosa",
))]
#[allow(clippy::too_many_arguments)]
fn mfcc<'py>(
    py: Python<'py>,
    audio: &PyAudio,
    frame_size_seconds: f32,
    hop_seconds: f32,
    n_mels: usize,
    n_mfcc: usize,
    f_min: f32,
    f_max: Option<f32>,
    method: &str,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mono: Vec<f32> = audio.inner.mono_samples().collect();
    let f_max = f_max.unwrap_or(audio.inner.sample_rate as f32 / 2.0);
    let mfcc_method = parse_mfcc_method(method)?;
    let arr = sadda_engine::dsp::mfcc(
        &mono,
        audio.inner.sample_rate,
        frame_size_seconds,
        hop_seconds,
        n_mels,
        n_mfcc,
        f_min,
        f_max,
        mfcc_method,
    );
    Ok(arr.into_pyarray(py))
}

/// Whisper-exact log-mel spectrogram, shape `(n_frames, n_mels)`.
///
/// Byte-faithful to OpenAI Whisper's encoder front end (Slaney mel,
/// power STFT with a periodic Hann window, `log10` + clamp, global
/// dynamic-range floor, `(+4)/4` normalisation). Expects **16 kHz mono**
/// for Whisper fidelity. `target_frames` pads/trims the audio so the
/// result has exactly that many frames (Whisper uses 3000 for 30 s);
/// `None` keeps the natural length.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (audio, *, n_fft=400, hop_length=160, n_mels=80, target_frames=None))]
fn log_mel_whisper<'py>(
    py: Python<'py>,
    audio: &PyAudio,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    target_frames: Option<usize>,
) -> Bound<'py, PyArray2<f32>> {
    let mono: Vec<f32> = audio.inner.mono_samples().collect();
    let arr = sadda_engine::dsp::log_mel_whisper(
        &mono,
        audio.inner.sample_rate,
        n_fft,
        hop_length,
        n_mels,
        target_frames,
    );
    arr.into_pyarray(py)
}

/// A long-term average spectrum: mean power per `bin_hz`-wide band, in
/// dB. Returned by `sadda.dsp.ltas`. Level *differences* (slope, tilt,
/// alpha ratio) are the meaningful quantities.
///
/// Convention: the stored spectrum data are **attributes** (`levels_db`,
/// `bin_hz`, `sample_rate` — accessed without parentheses), while the derived
/// scalar measures are **methods** (`slope(...)`, `tilt(...)`, `alpha_ratio()`
/// — called with parentheses). The measures are methods because they compute
/// over frequency bands; `slope`/`tilt` take the band edges as arguments and
/// `alpha_ratio` uses a fixed 1 kHz split.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "Ltas", frozen)]
struct PyLtas {
    inner: sadda_engine::dsp::Ltas,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyLtas {
    /// Band width in Hz.
    #[getter]
    fn bin_hz(&self) -> f32 {
        self.inner.bin_hz
    }
    /// Source sample rate in Hz.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate
    }
    /// dB level per band (band `i` spans `[i·bin_hz, (i+1)·bin_hz)`).
    #[getter]
    fn levels_db<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.levels_db.clone().into_pyarray(py)
    }
    /// Spectral slope (dB): high-band / low-band energy ratio.
    fn slope(&self, low_lo: f32, low_hi: f32, high_lo: f32, high_hi: f32) -> f32 {
        self.inner
            .slope((low_lo, low_hi), (high_lo, high_hi))
            .value()
    }
    /// Spectral tilt: regression slope of band dB over `[f_lo, f_hi)`,
    /// in dB per kHz.
    fn tilt(&self, f_lo: f32, f_hi: f32) -> f32 {
        self.inner.tilt(f_lo, f_hi)
    }
    /// Alpha ratio (dB): energy above vs below 1 kHz.
    fn alpha_ratio(&self) -> f32 {
        self.inner.alpha_ratio().value()
    }

    fn __repr__(&self) -> String {
        format!(
            "Ltas(bin_hz={}, n_bands={})",
            self.inner.bin_hz,
            self.inner.levels_db.len()
        )
    }
}

/// Computes the long-term average spectrum of `audio` with `bin_hz`-wide
/// bands (Welch-averaged power).
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (audio, *, bin_hz=100.0))]
fn ltas(audio: &PyAudio, bin_hz: f32) -> PyLtas {
    let mono: Vec<f32> = audio.inner.mono_samples().collect();
    PyLtas {
        inner: sadda_engine::dsp::ltas(&mono, audio.inner.sample_rate, bin_hz),
    }
}

/// Jitter + shimmer over a sustained phonation. Fields are floats:
/// jitter / relative shimmers are fractions (0.01 = 1%);
/// `shimmer_local_db` is in dB.
#[gen_stub_pyclass]
#[pyclass(module = "sadda._native", name = "PerturbationReport", frozen)]
struct PyPerturbationReport {
    inner: sadda_engine::PerturbationReport,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyPerturbationReport {
    /// Glottal periods the measures were computed over.
    #[getter]
    fn n_periods(&self) -> usize {
        self.inner.n_periods
    }
    /// Jitter (local) — fraction.
    #[getter]
    fn jitter_local(&self) -> f32 {
        self.inner.jitter_local.value()
    }
    /// Jitter (rap) — fraction.
    #[getter]
    fn jitter_rap(&self) -> f32 {
        self.inner.jitter_rap.value()
    }
    /// Jitter (ppq5) — fraction.
    #[getter]
    fn jitter_ppq5(&self) -> f32 {
        self.inner.jitter_ppq5.value()
    }
    /// Shimmer (local) — fraction.
    #[getter]
    fn shimmer_local(&self) -> f32 {
        self.inner.shimmer_local.value()
    }
    /// Shimmer (local) — dB.
    #[getter]
    fn shimmer_local_db(&self) -> f32 {
        self.inner.shimmer_local_db.value()
    }
    /// Shimmer (apq3) — fraction.
    #[getter]
    fn shimmer_apq3(&self) -> f32 {
        self.inner.shimmer_apq3.value()
    }
    /// Shimmer (apq5) — fraction.
    #[getter]
    fn shimmer_apq5(&self) -> f32 {
        self.inner.shimmer_apq5.value()
    }
    /// Period standard deviation (PSD) — seconds. An ABI component.
    #[getter]
    fn period_std_s(&self) -> f64 {
        self.inner.period_std_s.value()
    }

    fn __repr__(&self) -> String {
        format!(
            "PerturbationReport(n_periods={}, jitter_local={:.4}, shimmer_local={:.4})",
            self.inner.n_periods,
            self.inner.jitter_local.value(),
            self.inner.shimmer_local.value()
        )
    }
}

/// Computes jitter + shimmer for a sustained phonation. Raises
/// `ValueError` if no voiced f0 is found or too few periods are
/// detected (no-silent-fallback). Praat is the validation reference.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (audio, *, pitch_floor_hz=75.0, pitch_ceiling_hz=600.0))]
fn perturbation(
    audio: &PyAudio,
    pitch_floor_hz: f32,
    pitch_ceiling_hz: f32,
) -> PyResult<PyPerturbationReport> {
    let cfg = sadda_engine::PerturbationConfig {
        pitch_floor_hz,
        pitch_ceiling_hz,
    };
    sadda_engine::perturbation(&audio.inner, &cfg)
        .map(|inner| PyPerturbationReport { inner })
        .map_err(engine_err_to_py)
}

/// Mean harmonics-to-noise ratio (dB) of a sustained phonation, via the
/// Boersma-1993 cross-correlation method (Praat's `To Harmonicity
/// (cc)`). Raises `ValueError` if the signal is too short or silent.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (audio, *, pitch_floor_hz=75.0, pitch_ceiling_hz=600.0, hop_seconds=0.01))]
fn hnr(
    audio: &PyAudio,
    pitch_floor_hz: f32,
    pitch_ceiling_hz: f32,
    hop_seconds: f32,
) -> PyResult<f32> {
    let cfg = sadda_engine::HnrConfig {
        pitch_floor_hz,
        pitch_ceiling_hz,
        hop_seconds,
    };
    sadda_engine::hnr(&audio.inner, &cfg)
        .map(|d| d.value())
        .map_err(engine_err_to_py)
}

/// Smoothed cepstral peak prominence (dB) of a sustained phonation
/// (Praat's `PowerCepstrogram` → `Get CPPS`). Raises `ValueError` if the
/// signal is too short. Intended for sustained vowels.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (audio, *, pitch_floor_hz=60.0, pitch_ceiling_hz=330.0))]
fn cpps(audio: &PyAudio, pitch_floor_hz: f32, pitch_ceiling_hz: f32) -> PyResult<f32> {
    let cfg = sadda_engine::CppsConfig {
        pitch_floor_hz,
        pitch_ceiling_hz,
        ..Default::default()
    };
    sadda_engine::cpps(&audio.inner, &cfg)
        .map(|d| d.value())
        .map_err(engine_err_to_py)
}

/// H1–H2 (dB): level of the first harmonic minus the second, a glottal-source
/// / open-quotient correlate and an ABI component. Uncorrected (no formant
/// correction). Raises `ValueError` if no voiced f0 is found or the signal is
/// too short. Intended for sustained vowels.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (audio, *, pitch_floor_hz=75.0, pitch_ceiling_hz=600.0))]
fn h1_h2(audio: &PyAudio, pitch_floor_hz: f32, pitch_ceiling_hz: f32) -> PyResult<f32> {
    let cfg = sadda_engine::H1H2Config {
        pitch_floor_hz,
        pitch_ceiling_hz,
        ..Default::default()
    };
    sadda_engine::h1_h2(&audio.inner, &cfg)
        .map(|d| d.value())
        .map_err(engine_err_to_py)
}

/// Glottal-to-Noise Excitation ratio (Michaelis et al. 1997), in [0, 1] — a
/// breathiness / turbulent-noise correlate and an ABI component. ~1 for
/// pulsatile (glottal) excitation, toward 0 for turbulent noise. Raises
/// `ValueError` if the signal is too short. Intended for sustained vowels.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (audio, *, downsample_hz=10_000, lpc_order=13, bandwidth_hz=1000.0, fshift_hz=300.0))]
fn gne(
    audio: &PyAudio,
    downsample_hz: u32,
    lpc_order: usize,
    bandwidth_hz: f32,
    fshift_hz: f32,
) -> PyResult<f32> {
    let cfg = sadda_engine::GneConfig {
        downsample_hz,
        lpc_order,
        bandwidth_hz,
        fshift_hz,
    };
    sadda_engine::gne(&audio.inner, &cfg)
        .map(|r| r.value())
        .map_err(engine_err_to_py)
}

/// High-frequency noise level Hfno-6000 (dB): the LTAS level difference
/// between the 0–6 kHz and 6–10 kHz bands. Larger ⇒ less high-frequency
/// noise ⇒ less breathy. An ABI component. Requires a sample rate ≥ 20 kHz;
/// raises `ValueError` otherwise.
#[gen_stub_pyfunction]
#[pyfunction]
fn hfno(audio: &PyAudio) -> PyResult<f32> {
    sadda_engine::hfno(&audio.inner)
        .map(|d| d.value())
        .map_err(engine_err_to_py)
}

/// HNR-D (dB): the Dejonckere–Lebacq harmonic-to-noise ratio in the
/// 500–1500 Hz formant zone — an ABI component. CLEAN-ROOM / PROVISIONAL:
/// reconstructed from the ABI papers' prose, not the authors' exact
/// procedure. Raises `ValueError` if no voiced f0 or too few in-band
/// harmonics. Intended for sustained vowels.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (audio, *, pitch_floor_hz=75.0, pitch_ceiling_hz=600.0, band_lo_hz=500.0, band_hi_hz=1500.0, frame_size=8192))]
fn hnr_d(
    audio: &PyAudio,
    pitch_floor_hz: f32,
    pitch_ceiling_hz: f32,
    band_lo_hz: f32,
    band_hi_hz: f32,
    frame_size: usize,
) -> PyResult<f32> {
    let cfg = sadda_engine::HnrDConfig {
        pitch_floor_hz,
        pitch_ceiling_hz,
        band_lo_hz,
        band_hi_hz,
        frame_size,
    };
    sadda_engine::hnr_d(&audio.inner, &cfg)
        .map(|d| d.value())
        .map_err(engine_err_to_py)
}

/// Acoustic Breathiness Index v01 (Barsties von Latoszek et al. 2017): a
/// 0–10 breathiness score from its nine components. Clean-room from the
/// published formula; **PROVISIONAL** — the HNR-D/Hfno definitions and the
/// component unit conventions are not yet confirmed against the authors'
/// artifact (so `abi_from_audio` is intentionally not provided). Units:
/// CPPS / Hfno / HNR-D / H1−H2 / shimmer-dB in dB; GNE a ratio in [0,1];
/// jitter and shimmer-local as percents; PSD in seconds.
#[gen_stub_pyfunction]
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn abi(
    cpps: f32,
    jitter_pct: f32,
    gne: f32,
    hfno: f32,
    hnr_d: f32,
    h1_h2: f32,
    shimmer_db: f32,
    shimmer_pct: f32,
    psd_s: f32,
) -> f32 {
    sadda_engine::abi(
        cpps,
        jitter_pct,
        gne,
        hfno,
        hnr_d,
        h1_h2,
        shimmer_db,
        shimmer_pct,
        psd_s,
    )
}

/// Acoustic Voice Quality Index v03.01 from its six components. Clean-room
/// from the publications; **not yet confirmed against the reference Praat
/// script** (exposed as PROVISIONAL). Units: CPPS / HNR / shimmer-dB /
/// slope / tilt in dB, shimmer-local as a percent.
#[gen_stub_pyfunction]
#[pyfunction]
fn avqi(
    cpps: f32,
    hnr: f32,
    shimmer_local_pct: f32,
    shimmer_local_db: f32,
    slope: f32,
    tilt: f32,
) -> f32 {
    sadda_engine::avqi(cpps, hnr, shimmer_local_pct, shimmer_local_db, slope, tilt)
}

/// Timeline navigation state: a cursor (playhead), a visible view window, and an
/// optional `(start, end)` selection over a recording of `duration` seconds.
///
/// Every action comes as a **move-to** (absolute) / **move-by** (relative) pair
/// — e.g. `set_cursor(t)` vs `move_cursor_by(dt)` — so scripts can drive the
/// same navigation the desktop app's keyboard does. Times are in seconds; the
/// cursor and selection clamp to `[0, duration]`, and the view always stays
/// within the recording.
///
/// Construct with `Timeline(duration_seconds)`.
#[gen_stub_pyclass]
// `skip_from_py_object`: this is a mutable value type that's never passed by
// value into other binding methods, so it doesn't need the FromPyObject derive
// (pyo3 0.28 makes that opt-in for Clone pyclasses).
#[pyclass(module = "sadda._native", name = "Timeline", skip_from_py_object)]
#[derive(Clone)]
struct PyTimeline {
    inner: sadda_engine::Timeline,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyTimeline {
    /// Builds a timeline for a recording of `duration_seconds`, with the view
    /// spanning the whole recording and the cursor at the start.
    #[new]
    fn new(duration_seconds: f64) -> Self {
        Self {
            inner: sadda_engine::Timeline::new(duration_seconds),
        }
    }

    /// Cursor (playhead) position, in seconds.
    #[getter]
    fn cursor(&self) -> f64 {
        self.inner.cursor
    }
    /// Left edge of the visible view window, in seconds.
    #[getter]
    fn view_start(&self) -> f64 {
        self.inner.view_start
    }
    /// Right edge of the visible view window, in seconds (exclusive).
    #[getter]
    fn view_end(&self) -> f64 {
        self.inner.view_end
    }
    /// Recording duration, in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        self.inner.duration
    }
    /// Width of the visible window (`view_end - view_start`), in seconds.
    #[getter]
    fn view_range(&self) -> f64 {
        self.inner.view_range()
    }
    /// The current selection as `(start, end)` seconds, or `None`.
    #[getter]
    fn selection(&self) -> Option<(f64, f64)> {
        self.inner.selection
    }

    /// Re-initialises for a freshly-loaded recording: view spans the whole
    /// recording, cursor at 0, no selection.
    fn reset_for_bundle(&mut self, duration_seconds: f64) {
        self.inner.reset_for_bundle(duration_seconds);
    }

    // ----- cursor -----

    /// Moves the cursor **to** `t` seconds (clamped to the recording).
    fn set_cursor(&mut self, t: f64) {
        self.inner.set_cursor(t);
    }
    /// Moves the cursor **by** `delta_seconds` (negative = left).
    fn move_cursor_by(&mut self, delta_seconds: f64) {
        self.inner.move_cursor_by(delta_seconds);
    }

    // ----- selection -----

    /// Sets the selection's **start** edge **to** `t`, seeding a selection at
    /// the cursor when none exists and clamping so `start <= end`.
    fn set_selection_start(&mut self, t: f64) {
        self.inner.set_selection_start(t);
    }
    /// Moves the selection's **start** edge **by** `delta_seconds`.
    fn move_selection_start_by(&mut self, delta_seconds: f64) {
        self.inner.move_selection_start_by(delta_seconds);
    }
    /// Sets the selection's **end** edge **to** `t`, seeding a selection at the
    /// cursor when none exists and clamping so `end >= start`.
    fn set_selection_end(&mut self, t: f64) {
        self.inner.set_selection_end(t);
    }
    /// Moves the selection's **end** edge **by** `delta_seconds`.
    fn move_selection_end_by(&mut self, delta_seconds: f64) {
        self.inner.move_selection_end_by(delta_seconds);
    }
    /// Sets the selection to exactly `[start, end]` seconds in one call (sorted
    /// and clamped). The selection analogue of `set_view_range`.
    fn set_selection_range(&mut self, start: f64, end: f64) {
        self.inner.set_selection_range(start, end);
    }
    /// Places a zero-width selection point at `t`.
    fn set_point_selection(&mut self, t: f64) {
        self.inner.set_point_selection(t);
    }
    /// Clears any selection.
    fn clear_selection(&mut self) {
        self.inner.clear_selection();
    }

    // ----- view: scroll & zoom -----

    /// Pans the view so it starts **at** `t` seconds, preserving the range.
    fn set_view_start(&mut self, t: f64) {
        self.inner.set_view_start(t);
    }
    /// Pans the view **by** `delta_seconds`, preserving the range.
    fn scroll_by(&mut self, delta_seconds: f64) {
        self.inner.scroll_by(delta_seconds);
    }
    /// Frames the view to exactly `[start, end]` seconds (clamped). Use for
    /// "fit whole recording" (`0, duration`) or "zoom to selection".
    fn set_view_range(&mut self, start: f64, end: f64) {
        self.inner.set_view_range(start, end);
    }
    /// Zooms around `time_seconds` by `factor` (< 1 zooms in, > 1 zooms out).
    fn zoom_at(&mut self, time_seconds: f64, factor: f64) {
        self.inner.zoom_at(time_seconds, factor);
    }
    /// Pans the minimum amount to bring `t` into the visible window.
    fn scroll_into_view(&mut self, t: f64) {
        self.inner.scroll_into_view(t);
    }
    /// Shifts the view (if needed) so the cursor sits a quarter of the way in
    /// from the left edge — the playback-follow convention.
    fn ensure_cursor_visible(&mut self) {
        self.inner.ensure_cursor_visible();
    }
    /// Maps a pixel-x within `[0, plot_width_px)` to seconds in the view range.
    fn pixel_to_time(&self, pixel_x: f64, plot_width_px: f64) -> f64 {
        self.inner.pixel_to_time(pixel_x, plot_width_px)
    }

    fn __repr__(&self) -> String {
        format!(
            "Timeline(cursor={:.3}, view=[{:.3}, {:.3}], duration={:.3}, selection={:?})",
            self.inner.cursor,
            self.inner.view_start,
            self.inner.view_end,
            self.inner.duration,
            self.inner.selection,
        )
    }
}

/// sadda._native — Rust extension submodule. End users should `import sadda`
/// and use the decorated re-exports in `sadda.__init__` rather than reaching
/// in here directly.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(load_wav, m)?)?;
    m.add_function(wrap_pyfunction!(probe_wav, m)?)?;
    m.add_function(wrap_pyfunction!(parse_textgrid_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(syllabify, m)?)?;
    m.add_function(wrap_pyfunction!(f0, m)?)?;
    m.add_function(wrap_pyfunction!(hann, m)?)?;
    m.add_function(wrap_pyfunction!(hamming, m)?)?;
    m.add_function(wrap_pyfunction!(blackman, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian, m)?)?;
    m.add_function(wrap_pyfunction!(kaiser, m)?)?;
    m.add_function(wrap_pyfunction!(stft, m)?)?;
    m.add_function(wrap_pyfunction!(spectrogram, m)?)?;
    m.add_function(wrap_pyfunction!(intensity, m)?)?;
    m.add_function(wrap_pyfunction!(voiced_pitch, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_pitch_range, m)?)?;
    m.add_function(wrap_pyfunction!(speaker_pitch_range_pooled, m)?)?;
    m.add_function(wrap_pyfunction!(formants, m)?)?;
    m.add_function(wrap_pyfunction!(mfcc, m)?)?;
    m.add_function(wrap_pyfunction!(log_mel_whisper, m)?)?;
    m.add_function(wrap_pyfunction!(align::forced_align, m)?)?;
    m.add_function(wrap_pyfunction!(ltas, m)?)?;
    m.add_function(wrap_pyfunction!(perturbation, m)?)?;
    m.add_function(wrap_pyfunction!(hnr, m)?)?;
    m.add_function(wrap_pyfunction!(cpps, m)?)?;
    m.add_function(wrap_pyfunction!(h1_h2, m)?)?;
    m.add_function(wrap_pyfunction!(gne, m)?)?;
    m.add_function(wrap_pyfunction!(hfno, m)?)?;
    m.add_function(wrap_pyfunction!(hnr_d, m)?)?;
    m.add_function(wrap_pyfunction!(abi, m)?)?;
    m.add_function(wrap_pyfunction!(avqi, m)?)?;
    m.add_function(wrap_pyfunction!(new_project, m)?)?;
    m.add_function(wrap_pyfunction!(open_project, m)?)?;
    m.add_class::<PyAudio>()?;
    m.add_class::<PyAudioProbe>()?;
    m.add_class::<PyBundle>()?;
    m.add_class::<PySpeaker>()?;
    m.add_class::<PySession>()?;
    m.add_class::<PyTier>()?;
    m.add_class::<PyInterval>()?;
    m.add_class::<PyPoint>()?;
    m.add_class::<PyReference>()?;
    m.add_class::<PyRubric>()?;
    m.add_class::<PyStatusDef>()?;
    m.add_class::<PyRubricTier>()?;
    m.add_class::<PyVocabEntry>()?;
    m.add_class::<PyLabelCheck>()?;
    m.add_class::<PyCriterion>()?;
    m.add_class::<PyTarget>()?;
    m.add_class::<PyAssignment>()?;
    m.add_class::<PyExportSummary>()?;
    m.add_class::<PyImportSummary>()?;
    m.add_class::<PyConcordanceSummary>()?;
    m.add_class::<PyAgreementReport>()?;
    m.add_class::<PyProgressCounts>()?;
    m.add_class::<PyAnnotatorProgress>()?;
    m.add_class::<PyQaReport>()?;
    m.add_class::<PyPairAgreement>()?;
    m.add_class::<PyRubricVersion>()?;
    m.add_class::<PyRubricTierSnapshot>()?;
    m.add_class::<PyTierImpact>()?;
    m.add_class::<PyNotebookEntry>()?;
    m.add_class::<PyDerivedSignal>()?;
    m.add_class::<PyFormantFrame>()?;
    m.add_class::<PyLtas>()?;
    m.add_class::<PyPerturbationReport>()?;
    m.add_class::<PyProcessingRun>()?;
    m.add_class::<PyCitation>()?;
    m.add_class::<PyCalibration>()?;
    m.add_class::<PyTimeline>()?;
    m.add_class::<PyInstrument>()?;
    m.add_class::<PyProject>()?;

    // Live recording: sadda.live.* surface. We register it as a Python
    // submodule (`sadda._native.live`) and the Python-side
    // `sadda/live/__init__.py` re-exports the symbols under the
    // user-facing `sadda.live.*` path.
    // Submodules are created with their FULL dotted name so the
    // functions added to them report `__module__ == "sadda._native.live"`
    // (etc.) rather than a bare `"live"`. Bare names leave the symbols
    // unresolvable to documentation tooling that inspects the runtime
    // (griffe → mkdocstrings), and are wrong for pickling / `repr`. We
    // then attach each under its SHORT attribute name via `m.add(...)`
    // (so `sadda._native.live` works) and register it in `sys.modules`
    // so `import sadda._native.live` resolves too — the documented PyO3
    // submodule idiom.
    let sys_modules = m.py().import("sys")?.getattr("modules")?;

    let live_mod = PyModule::new(m.py(), "sadda._native.live")?;
    live_mod.add_function(wrap_pyfunction!(live::start_session, &live_mod)?)?;
    live_mod.add_function(wrap_pyfunction!(live::list_input_devices, &live_mod)?)?;
    live_mod.add_function(wrap_pyfunction!(live::default_input_device, &live_mod)?)?;
    live_mod.add_class::<live::PyLiveSession>()?;
    m.add("live", &live_mod)?;
    sys_modules.set_item("sadda._native.live", &live_mod)?;

    // Recipes: sadda.recipe.* surface (F1). Mirrors the live module
    // pattern — registered as `sadda._native.recipe`; the Python
    // `sadda/recipe/__init__.py` wraps `start` / `end` /
    // `generate_script` inside a context-manager class and decorates
    // the public surface as `@provisional`.
    let recipe_mod = PyModule::new(m.py(), "sadda._native.recipe")?;
    recipe_mod.add_function(wrap_pyfunction!(recipe::start, &recipe_mod)?)?;
    recipe_mod.add_function(wrap_pyfunction!(recipe::end, &recipe_mod)?)?;
    recipe_mod.add_function(wrap_pyfunction!(recipe::list_recipes, &recipe_mod)?)?;
    recipe_mod.add_function(wrap_pyfunction!(recipe::get, &recipe_mod)?)?;
    recipe_mod.add_function(wrap_pyfunction!(recipe::generate_script, &recipe_mod)?)?;
    recipe_mod.add_function(wrap_pyfunction!(recipe::script_path, &recipe_mod)?)?;
    recipe_mod.add_class::<recipe::PyRecipe>()?;
    m.add("recipe", &recipe_mod)?;
    sys_modules.set_item("sadda._native.recipe", &recipe_mod)?;

    // Reference distributions: sadda.refdist.* surface (C7, consumption
    // side). Registered as `sadda._native.refdist`; the Python
    // `sadda/refdist/__init__.py` re-exports the symbols and adds a Polars
    // `.data()` helper on RefDist.
    let refdist_mod = PyModule::new(m.py(), "sadda._native.refdist")?;
    refdist_mod.add_function(wrap_pyfunction!(refdist::query, &refdist_mod)?)?;
    refdist_mod.add_function(wrap_pyfunction!(refdist::get, &refdist_mod)?)?;
    refdist_mod.add_function(wrap_pyfunction!(refdist::list_all, &refdist_mod)?)?;
    refdist_mod.add_function(wrap_pyfunction!(refdist::install, &refdist_mod)?)?;
    refdist_mod.add_function(wrap_pyfunction!(refdist::scaffold, &refdist_mod)?)?;
    refdist_mod.add_function(wrap_pyfunction!(refdist::store_root, &refdist_mod)?)?;
    refdist_mod.add_class::<refdist::PyRefDist>()?;
    refdist_mod.add_class::<refdist::PySummary>()?;
    refdist_mod.add_class::<refdist::PyHistogram>()?;
    m.add("refdist", &refdist_mod)?;
    sys_modules.set_item("sadda._native.refdist", &refdist_mod)?;

    // E11 ML inference: sadda.ml.* (registered as `sadda._native.ml`;
    // the Python `sadda/ml/__init__.py` re-exports with stability tiers).
    let ml_mod = PyModule::new(m.py(), "sadda._native.ml")?;
    ml_mod.add_function(wrap_pyfunction!(ml::vad, &ml_mod)?)?;
    ml_mod.add_function(wrap_pyfunction!(ml::speech_segments, &ml_mod)?)?;
    ml_mod.add_function(wrap_pyfunction!(ml::load_model, &ml_mod)?)?;
    ml_mod.add_function(wrap_pyfunction!(ml::install_model, &ml_mod)?)?;
    ml_mod.add_function(wrap_pyfunction!(ml::get_model, &ml_mod)?)?;
    ml_mod.add_class::<ml::PyModel>()?;
    m.add("ml", &ml_mod)?;
    sys_modules.set_item("sadda._native.ml", &ml_mod)?;

    // MFCC preset registry: sadda.dsp preset surface (roadmap item 3/4).
    // Registered as `sadda._native.mfcc_preset`; the Python
    // `sadda/dsp/__init__.py` re-exports the params/preset types and the
    // store functions, and dispatches `mfcc(audio, params=…)` to `compute`.
    let mfcc_preset_mod = PyModule::new(m.py(), "sadda._native.mfcc_preset")?;
    mfcc_preset_mod.add_function(wrap_pyfunction!(mfcc_preset::store_root, &mfcc_preset_mod)?)?;
    mfcc_preset_mod.add_function(wrap_pyfunction!(mfcc_preset::builtin, &mfcc_preset_mod)?)?;
    mfcc_preset_mod.add_function(wrap_pyfunction!(mfcc_preset::list_all, &mfcc_preset_mod)?)?;
    mfcc_preset_mod.add_function(wrap_pyfunction!(mfcc_preset::list_user, &mfcc_preset_mod)?)?;
    mfcc_preset_mod.add_function(wrap_pyfunction!(mfcc_preset::get, &mfcc_preset_mod)?)?;
    mfcc_preset_mod.add_function(wrap_pyfunction!(mfcc_preset::save, &mfcc_preset_mod)?)?;
    mfcc_preset_mod.add_function(wrap_pyfunction!(mfcc_preset::delete, &mfcc_preset_mod)?)?;
    mfcc_preset_mod.add_function(wrap_pyfunction!(mfcc_preset::compute, &mfcc_preset_mod)?)?;
    mfcc_preset_mod.add_class::<mfcc_preset::PyMfccParams>()?;
    mfcc_preset_mod.add_class::<mfcc_preset::PyMfccPreset>()?;
    m.add("mfcc_preset", &mfcc_preset_mod)?;
    sys_modules.set_item("sadda._native.mfcc_preset", &mfcc_preset_mod)?;

    // Pitch preset registry: sadda.dsp preset surface (roadmap item 6).
    // Registered as `sadda._native.pitch_preset`; the Python
    // `sadda/dsp/__init__.py` re-exports the params/preset types and the
    // store functions, and dispatches `voiced_pitch(audio, params=…)`.
    let pitch_preset_mod = PyModule::new(m.py(), "sadda._native.pitch_preset")?;
    pitch_preset_mod.add_function(wrap_pyfunction!(
        pitch_preset::store_root,
        &pitch_preset_mod
    )?)?;
    pitch_preset_mod.add_function(wrap_pyfunction!(pitch_preset::builtin, &pitch_preset_mod)?)?;
    pitch_preset_mod.add_function(wrap_pyfunction!(pitch_preset::list_all, &pitch_preset_mod)?)?;
    pitch_preset_mod.add_function(wrap_pyfunction!(
        pitch_preset::list_user,
        &pitch_preset_mod
    )?)?;
    pitch_preset_mod.add_function(wrap_pyfunction!(pitch_preset::get, &pitch_preset_mod)?)?;
    pitch_preset_mod.add_function(wrap_pyfunction!(pitch_preset::save, &pitch_preset_mod)?)?;
    pitch_preset_mod.add_function(wrap_pyfunction!(pitch_preset::delete, &pitch_preset_mod)?)?;
    pitch_preset_mod.add_function(wrap_pyfunction!(pitch_preset::compute, &pitch_preset_mod)?)?;
    pitch_preset_mod.add_class::<pitch_preset::PyPitchParams>()?;
    pitch_preset_mod.add_class::<pitch_preset::PyPitchPreset>()?;
    m.add("pitch_preset", &pitch_preset_mod)?;
    sys_modules.set_item("sadda._native.pitch_preset", &pitch_preset_mod)?;

    // Formant preset registry: sadda.dsp preset surface (roadmap item 6).
    let formant_preset_mod = PyModule::new(m.py(), "sadda._native.formant_preset")?;
    formant_preset_mod.add_function(wrap_pyfunction!(
        formant_preset::store_root,
        &formant_preset_mod
    )?)?;
    formant_preset_mod.add_function(wrap_pyfunction!(
        formant_preset::builtin,
        &formant_preset_mod
    )?)?;
    formant_preset_mod.add_function(wrap_pyfunction!(
        formant_preset::list_all,
        &formant_preset_mod
    )?)?;
    formant_preset_mod.add_function(wrap_pyfunction!(
        formant_preset::list_user,
        &formant_preset_mod
    )?)?;
    formant_preset_mod.add_function(wrap_pyfunction!(formant_preset::get, &formant_preset_mod)?)?;
    formant_preset_mod
        .add_function(wrap_pyfunction!(formant_preset::save, &formant_preset_mod)?)?;
    formant_preset_mod.add_function(wrap_pyfunction!(
        formant_preset::delete,
        &formant_preset_mod
    )?)?;
    formant_preset_mod.add_function(wrap_pyfunction!(
        formant_preset::compute,
        &formant_preset_mod
    )?)?;
    formant_preset_mod.add_class::<formant_preset::PyFormantsParams>()?;
    formant_preset_mod.add_class::<formant_preset::PyFormantPreset>()?;
    m.add("formant_preset", &formant_preset_mod)?;
    sys_modules.set_item("sadda._native.formant_preset", &formant_preset_mod)?;
    Ok(())
}

/// Stub-info gatherer for `pyo3-stub-gen`. The library-provided
/// `define_stub_info_gatherer!` macro hardcodes the pyproject.toml location to
/// `CARGO_MANIFEST_DIR/pyproject.toml`; ours lives at the workspace root
/// (two directories up from `crates/python/`), so we point at it manually.
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let manifest_dir: &std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/python/ should sit two directories below the workspace root");
    pyo3_stub_gen::StubInfo::from_pyproject_toml(workspace_root.join("pyproject.toml"))
}
