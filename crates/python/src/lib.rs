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

use numpy::{IntoPyArray, PyArray1};
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
    };
    let frames = sadda_engine::autocorrelation(&audio.inner, &config);
    let times: Vec<f64> = frames.iter().map(|f| f.time_seconds).collect();
    let freqs: Vec<f32> = frames.iter().map(|f| f.frequency_hz).collect();
    (times.into_pyarray(py), freqs.into_pyarray(py))
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
    m.add_function(wrap_pyfunction!(new_project, m)?)?;
    m.add_function(wrap_pyfunction!(open_project, m)?)?;
    m.add_class::<PyAudio>()?;
    m.add_class::<PyBundle>()?;
    m.add_class::<PySpeaker>()?;
    m.add_class::<PySession>()?;
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
