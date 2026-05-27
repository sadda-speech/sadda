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

mod live;
mod ml;
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
    }
}

/// Audio data loaded from disk. Samples are interleaved float32 in `[-1.0, 1.0]`;
/// for stereo the layout is `[L0, R0, L1, R1, ...]`. Construct via
/// `sadda.load_wav(path)`.
#[gen_stub_pyclass]
#[pyclass(name = "Audio")]
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

/// One recording inside a [`Project`]: audio header plus optional Session +
/// Speaker FKs and a freeform JSON `extra` payload. Read-only view; mutate
/// via `Project.add_bundle(...)`.
#[gen_stub_pyclass]
#[pyclass(name = "Bundle", frozen)]
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
#[pyclass(name = "Speaker", frozen)]
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
#[pyclass(name = "Session", frozen)]
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
#[pyclass(name = "Tier", frozen)]
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
#[pyclass(name = "Interval", frozen)]
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
#[pyclass(name = "Point", frozen)]
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
#[pyclass(name = "Reference", frozen)]
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

/// Registration row for a Parquet sidecar holding a dense tier's data.
/// Created automatically by the `Project.write_continuous_numeric` /
/// `write_continuous_vector` / `write_categorical_sampled` methods.
#[gen_stub_pyclass]
#[pyclass(name = "DerivedSignal", frozen)]
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
#[pyclass(name = "ProcessingRun", frozen)]
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
#[pyclass(name = "Citation", frozen)]
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

    fn __repr__(&self) -> String {
        format!(
            "Citation(processor_id={:?}, doi={:?})",
            self.inner.processor_id, self.inner.doi
        )
    }
}

/// Microphone / signal-chain calibration mapping dB-FS to dB-SPL.
/// Construct with `Calibration(reference_spl_db=…, reference_db_fs=…)`.
#[gen_stub_pyclass]
// `from_py_object` opts into the FromPyObject derive (pyo3 0.28 makes it
// explicit for Clone pyclasses) so `Calibration` can be passed by value
// to `add_instrument`.
#[pyclass(name = "Calibration", frozen, from_py_object)]
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
#[pyclass(name = "Instrument", frozen)]
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
#[pyclass(name = "Project", unsendable)]
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

    /// Inserts an interval annotation. Enforces parent-child cardinality at
    /// insert time; raises `ValueError` on cardinality violation.
    #[pyo3(signature = (
        tier_id, start_seconds, end_seconds, *,
        label=None, parent_annotation_id=None, extra=None,
    ))]
    fn add_interval(
        &self,
        tier_id: i64,
        start_seconds: f64,
        end_seconds: f64,
        label: Option<String>,
        parent_annotation_id: Option<i64>,
        extra: Option<String>,
    ) -> PyResult<i64> {
        let spec = sadda_engine::IntervalSpec {
            tier_id,
            start_seconds,
            end_seconds,
            label,
            parent_annotation_id,
            extra,
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
    #[pyo3(signature = (
        tier_id, time_seconds, *,
        label=None, parent_annotation_id=None, extra=None,
    ))]
    fn add_point(
        &self,
        tier_id: i64,
        time_seconds: f64,
        label: Option<String>,
        parent_annotation_id: Option<i64>,
        extra: Option<String>,
    ) -> PyResult<i64> {
        let spec = sadda_engine::PointSpec {
            tier_id,
            time_seconds,
            label,
            parent_annotation_id,
            extra,
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
#[pyclass(name = "FormantFrame", frozen)]
struct PyFormantFrame {
    inner: sadda_engine::dsp::FormantFrame,
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
        "windowed_autocorrelation" | "boersma" => {
            Ok(sadda_engine::pitch::PitchMethod::WindowedAutocorrelation)
        }
        other => Err(PyValueError::new_err(format!(
            "unknown pitch method {other:?}; expected 'autocorrelation' or 'windowed_autocorrelation'"
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

/// Estimates f0 with a voicing decision and returns `(times, frequencies,
/// voicing)` as three NumPy arrays. `times` is float64 seconds at frame
/// centres; `frequencies` is float32 Hz; `voicing` is float32 in `[0, 1]`.
///
/// `method` selects the pitch tracker:
/// - `"windowed_autocorrelation"` (default) — adopts Boersma 1993's
///   window-correction idea (divides windowed-signal autocorrelation by
///   window autocorrelation); not a full Boersma implementation. Strict
///   improvement on `"autocorrelation"`.
/// - `"autocorrelation"` — naive time-domain autocorrelation (Phase-0
///   tracker; what `sadda.dsp.f0(...)` calls).
///
/// `voicing_threshold` is informational here: the function returns voicing
/// values for every frame so callers can apply their own threshold.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    audio, *,
    frame_size_seconds=0.030, hop_size_seconds=0.010,
    min_freq_hz=75.0, max_freq_hz=500.0,
    method="windowed_autocorrelation",
    voicing_threshold=0.45,
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
    };
    let frames = sadda_engine::pitch::pitch(&audio.inner, &config, pitch_method);
    let times: Vec<f64> = frames.iter().map(|f| f.time_seconds).collect();
    let freqs: Vec<f32> = frames.iter().map(|f| f.frequency_hz.value()).collect();
    let voicing: Vec<f32> = frames.iter().map(|f| f.voicing).collect();
    Ok((
        times.into_pyarray(py),
        freqs.into_pyarray(py),
        voicing.into_pyarray(py),
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
/// Defaults match `librosa.feature.mfcc`: Slaney mel scale, `n_mels=40`,
/// `n_mfcc=13`, `f_min=0`, `f_max=sr/2`, 25 ms frame, 10 ms hop.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    audio, *,
    frame_size_seconds=0.025, hop_seconds=0.010,
    n_mels=40, n_mfcc=13,
    f_min=0.0, f_max=None,
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
) -> Bound<'py, PyArray2<f32>> {
    let mono: Vec<f32> = audio.inner.mono_samples().collect();
    let f_max = f_max.unwrap_or(audio.inner.sample_rate as f32 / 2.0);
    let arr = sadda_engine::dsp::mfcc(
        &mono,
        audio.inner.sample_rate,
        frame_size_seconds,
        hop_seconds,
        n_mels,
        n_mfcc,
        f_min,
        f_max,
    );
    arr.into_pyarray(py)
}

/// A long-term average spectrum: mean power per `bin_hz`-wide band, in
/// dB. Returned by `sadda.dsp.ltas`. Level *differences* (slope, tilt,
/// alpha ratio) are the meaningful quantities.
#[gen_stub_pyclass]
#[pyclass(name = "Ltas", frozen)]
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
#[pyclass(name = "PerturbationReport", frozen)]
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

/// sadda._native — Rust extension submodule. End users should `import sadda`
/// and use the decorated re-exports in `sadda.__init__` rather than reaching
/// in here directly.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(load_wav, m)?)?;
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
    m.add_function(wrap_pyfunction!(formants, m)?)?;
    m.add_function(wrap_pyfunction!(mfcc, m)?)?;
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
    m.add_class::<PyBundle>()?;
    m.add_class::<PySpeaker>()?;
    m.add_class::<PySession>()?;
    m.add_class::<PyTier>()?;
    m.add_class::<PyInterval>()?;
    m.add_class::<PyPoint>()?;
    m.add_class::<PyReference>()?;
    m.add_class::<PyDerivedSignal>()?;
    m.add_class::<PyFormantFrame>()?;
    m.add_class::<PyLtas>()?;
    m.add_class::<PyPerturbationReport>()?;
    m.add_class::<PyProcessingRun>()?;
    m.add_class::<PyCitation>()?;
    m.add_class::<PyCalibration>()?;
    m.add_class::<PyInstrument>()?;
    m.add_class::<PyProject>()?;

    // Live recording: sadda.live.* surface. We register it as a Python
    // submodule (`sadda._native.live`) and the Python-side
    // `sadda/live/__init__.py` re-exports the symbols under the
    // user-facing `sadda.live.*` path.
    let live_mod = PyModule::new(m.py(), "live")?;
    live_mod.add_function(wrap_pyfunction!(live::start_session, &live_mod)?)?;
    live_mod.add_function(wrap_pyfunction!(live::list_input_devices, &live_mod)?)?;
    live_mod.add_function(wrap_pyfunction!(live::default_input_device, &live_mod)?)?;
    live_mod.add_class::<live::PyLiveSession>()?;
    m.add_submodule(&live_mod)?;

    // Recipes: sadda.recipe.* surface (F1). Mirrors the live module
    // pattern — registered as `sadda._native.recipe`; the Python
    // `sadda/recipe/__init__.py` wraps `start` / `end` /
    // `generate_script` inside a context-manager class and decorates
    // the public surface as `@provisional`.
    let recipe_mod = PyModule::new(m.py(), "recipe")?;
    recipe_mod.add_function(wrap_pyfunction!(recipe::start, &recipe_mod)?)?;
    recipe_mod.add_function(wrap_pyfunction!(recipe::end, &recipe_mod)?)?;
    recipe_mod.add_function(wrap_pyfunction!(recipe::list_recipes, &recipe_mod)?)?;
    recipe_mod.add_function(wrap_pyfunction!(recipe::get, &recipe_mod)?)?;
    recipe_mod.add_function(wrap_pyfunction!(recipe::generate_script, &recipe_mod)?)?;
    recipe_mod.add_function(wrap_pyfunction!(recipe::script_path, &recipe_mod)?)?;
    recipe_mod.add_class::<recipe::PyRecipe>()?;
    m.add_submodule(&recipe_mod)?;

    // Reference distributions: sadda.refdist.* surface (C7, consumption
    // side). Registered as `sadda._native.refdist`; the Python
    // `sadda/refdist/__init__.py` re-exports the symbols and adds a Polars
    // `.data()` helper on RefDist.
    let refdist_mod = PyModule::new(m.py(), "refdist")?;
    refdist_mod.add_function(wrap_pyfunction!(refdist::query, &refdist_mod)?)?;
    refdist_mod.add_function(wrap_pyfunction!(refdist::get, &refdist_mod)?)?;
    refdist_mod.add_function(wrap_pyfunction!(refdist::list_all, &refdist_mod)?)?;
    refdist_mod.add_function(wrap_pyfunction!(refdist::install, &refdist_mod)?)?;
    refdist_mod.add_function(wrap_pyfunction!(refdist::scaffold, &refdist_mod)?)?;
    refdist_mod.add_function(wrap_pyfunction!(refdist::store_root, &refdist_mod)?)?;
    refdist_mod.add_class::<refdist::PyRefDist>()?;
    refdist_mod.add_class::<refdist::PySummary>()?;
    refdist_mod.add_class::<refdist::PyHistogram>()?;
    m.add_submodule(&refdist_mod)?;

    // E11 ML inference: sadda.ml.* (registered as `sadda._native.ml`;
    // the Python `sadda/ml/__init__.py` re-exports with stability tiers).
    let ml_mod = PyModule::new(m.py(), "ml")?;
    ml_mod.add_function(wrap_pyfunction!(ml::vad, &ml_mod)?)?;
    ml_mod.add_function(wrap_pyfunction!(ml::speech_segments, &ml_mod)?)?;
    ml_mod.add_function(wrap_pyfunction!(ml::load_model, &ml_mod)?)?;
    ml_mod.add_function(wrap_pyfunction!(ml::install_model, &ml_mod)?)?;
    ml_mod.add_function(wrap_pyfunction!(ml::get_model, &ml_mod)?)?;
    ml_mod.add_class::<ml::PyModel>()?;
    m.add_submodule(&ml_mod)?;
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
