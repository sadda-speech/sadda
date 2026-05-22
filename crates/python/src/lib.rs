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
        sadda_engine::EngineError::Cardinality(msg) => {
            PyValueError::new_err(format!("cardinality violation: {msg}"))
        }
    }
}

/// Audio data loaded from disk. Samples are interleaved float32 in `[-1.0, 1.0]`;
/// for stereo the layout is `[L0, R0, L1, R1, ...]`. Construct via
/// `sadda.load_wav(path)`.
#[gen_stub_pyclass]
#[pyclass(name = "Audio")]
struct PyAudio {
    inner: sadda_engine::Audio,
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

/// A sadda project: a directory holding audio, derived signals, attachments,
/// and a SQLite-backed corpus database. Construct via `sadda.new_project(...)`
/// or `sadda.open_project(...)`.
#[gen_stub_pyclass]
#[pyclass(name = "Project", unsendable)]
struct PyProject {
    // `unsendable`: rusqlite::Connection is Send but not Sync; the GIL
    // serializes Python-side access, but PyO3 requires Send+Sync by default.
    // `unsendable` tells PyO3 to check at runtime that this object isn't
    // shared across threads.
    inner: sadda_engine::Project,
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
    let freqs: Vec<f32> = frames.iter().map(|f| f.frequency_hz).collect();
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
    let db_fs: Vec<f32> = frames.iter().map(|f| f.db_fs).collect();
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
        self.inner.frequencies.clone().into_pyarray(py)
    }
    /// Bandwidths in Hz, co-indexed with `frequencies`.
    #[getter]
    fn bandwidths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.bandwidths.clone().into_pyarray(py)
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
    let freqs: Vec<f32> = frames.iter().map(|f| f.frequency_hz).collect();
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
    m.add_class::<PyProject>()?;
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
