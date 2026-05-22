//! PyO3 bindings for live audio recording. Wraps `sadda_engine::live`
//! and adds the cpal-stream lifecycle plus a Python dispatch thread
//! that pops from the engine's result rtrbs with the GIL held and
//! invokes registered Python callables.
//!
//! Module layout (see the 2026-05-22 E1 DEVLOG entry):
//!
//! - [`PyLiveSession`] — the `sadda.live.LiveSession` Python class.
//! - [`start_session`] — module-level constructor.
//! - [`list_input_devices`], [`default_input_device`] — discovery helpers.
//!
//! ## Test seam
//!
//! `start_session` does not require an audio device to construct — the
//! cpal stream is built lazily by [`PyLiveSession::start`]. A separate
//! `push_samples_for_tests` method on the session lets Python tests
//! push synthetic samples without touching cpal. CI has no audio
//! hardware, so all Python tests for the live surface use that path.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use sadda_engine::{
    LiveConfig, LiveSession as EngineLiveSession, StoppedSession,
};

use crate::engine_err_to_py;

/// Stored Python callables. Set via the `on_*` decorator methods; read
/// by the dispatch thread on each iteration. The dispatch thread
/// re-acquires the GIL on every poll, so the `Py<PyAny>` handles stay
/// inert in between.
#[derive(Default)]
struct Callbacks {
    meter: Option<Py<PyAny>>,
    pitch: Option<Py<PyAny>>,
    intensity: Option<Py<PyAny>>,
    formants: Option<Py<PyAny>>,
}

/// State that lives on the dispatch thread. Owns the result-rtrb
/// consumer handles for the lifetime of the recording.
struct DispatchState {
    results: sadda_engine::LiveResults,
    callbacks: Arc<Mutex<Callbacks>>,
    stop_flag: Arc<AtomicBool>,
}

impl DispatchState {
    fn run(mut self) {
        loop {
            let mut did_work = false;
            Python::attach(|py| {
                let cbs = self.callbacks.lock().unwrap();
                while let Ok(m) = self.results.meters.pop() {
                    did_work = true;
                    if let Some(cb) = cbs.meter.as_ref() {
                        let _ = cb.call1(py, (m.peak, m.rms, m.rms_db, m.time_seconds));
                    }
                }
                while let Ok(p) = self.results.pitches.pop() {
                    did_work = true;
                    if let Some(cb) = cbs.pitch.as_ref() {
                        let voiced = p.voicing >= 0.45;
                        let _ = cb.call1(py, (p.frequency_hz, voiced, p.time_seconds));
                    }
                }
                while let Ok(i) = self.results.intensities.pop() {
                    did_work = true;
                    if let Some(cb) = cbs.intensity.as_ref() {
                        let _ = cb.call1(py, (i.db_fs, i.time_seconds));
                    }
                }
                while let Ok(f) = self.results.formants.pop() {
                    did_work = true;
                    if let Some(cb) = cbs.formants.as_ref() {
                        let _ = cb.call1(
                            py,
                            (f.frequencies.clone(), f.bandwidths.clone(), f.time_seconds),
                        );
                    }
                }
            });
            if self.stop_flag.load(Ordering::Acquire) && !did_work {
                break;
            }
            if !did_work {
                thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

/// cpal::Stream is `!Send` on Linux ALSA, so we keep it alive on a
/// dedicated thread that owns it until told to drop. The handle
/// communicates with that thread via a stop sender + JoinHandle.
struct CpalStreamHandle {
    stop_tx: std::sync::mpsc::Sender<()>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for CpalStreamHandle {
    fn drop(&mut self) {
        let _ = self.stop_tx.send(());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// Recording state during capture; `None` after `stop()`.
struct ActiveRecording {
    engine: EngineLiveSession,
    cpal_stream: Option<CpalStreamHandle>,
    dispatch_handle: JoinHandle<()>,
    dispatch_stop: Arc<AtomicBool>,
}

/// Live recording session. Construct via [`start_session`]; register
/// callbacks via `on_meter` / `on_pitch` / `on_intensity` / `on_formants`;
/// then `start()` → `stop()` → `commit()` or `discard()`.
#[pyclass(name = "LiveSession", unsendable)]
pub(crate) struct PyLiveSession {
    active: Option<ActiveRecording>,
    stopped: Option<StoppedSession>,
    callbacks: Arc<Mutex<Callbacks>>,
    config: LiveConfig,
    device_label: String,
    name: String,
    in_progress_dir: PathBuf,
}

#[pymethods]
impl PyLiveSession {
    /// Registers a callable invoked once per drained chunk with
    /// `(peak, rms, rms_db, time_seconds)`. Returns the callable
    /// unchanged so the method can be used as a decorator.
    fn on_meter<'py>(&self, callback: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        self.callbacks.lock().unwrap().meter = Some(callback.clone().unbind());
        callback
    }

    /// Registers a callable invoked once per analysis hop with
    /// `(frequency_hz, voiced: bool, time_seconds)`. `voiced` is
    /// `voicing >= 0.45` (Boersma's recommended threshold).
    fn on_pitch<'py>(&self, callback: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        self.callbacks.lock().unwrap().pitch = Some(callback.clone().unbind());
        callback
    }

    /// Registers a callable invoked once per analysis hop with
    /// `(db_fs, time_seconds)`.
    fn on_intensity<'py>(&self, callback: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        self.callbacks.lock().unwrap().intensity = Some(callback.clone().unbind());
        callback
    }

    /// Registers a callable invoked once per analysis hop with
    /// `(frequencies: list[float], bandwidths: list[float],
    /// time_seconds)`. Both lists are co-indexed and the same length
    /// (which varies per frame).
    fn on_formants<'py>(&self, callback: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        self.callbacks.lock().unwrap().formants = Some(callback.clone().unbind());
        callback
    }

    /// Begins capture. Builds the cpal input stream against the
    /// requested device and starts the audio thread. Idempotent: a
    /// second call returns `Ok` without rebuilding the stream.
    fn start(&mut self) -> PyResult<()> {
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("session already stopped"))?;
        if active.cpal_stream.is_some() {
            return Ok(());
        }
        let mut producer = active.engine.take_producer().ok_or_else(|| {
            PyRuntimeError::new_err("engine session has no producer (already started)")
        })?;

        let device = pick_input_device(&self.device_label)?;
        let stream_cfg = build_stream_config(&device, &self.config)?;
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let stream_thread = thread::Builder::new()
            .name("sadda-live-cpal".into())
            .spawn(move || {
                let stream = match device.build_input_stream::<f32, _, _>(
                    &stream_cfg,
                    move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                        for &s in data {
                            let _ = producer.push(s);
                        }
                    },
                    move |err| {
                        eprintln!("sadda.live cpal error: {err}");
                    },
                    None,
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("sadda.live: build_input_stream failed: {e}");
                        return;
                    }
                };
                if let Err(e) = stream.play() {
                    eprintln!("sadda.live: stream.play() failed: {e}");
                    return;
                }
                let _ = stop_rx.recv();
                drop(stream);
            })
            .map_err(|e| PyRuntimeError::new_err(format!("spawn cpal thread: {e}")))?;
        active.cpal_stream = Some(CpalStreamHandle {
            stop_tx,
            thread: Some(stream_thread),
        });
        Ok(())
    }

    /// Stops the capture, joins the cpal-stream and dispatch threads,
    /// and flushes the WAV writer. After `stop()` the session waits
    /// for `commit()` or `discard()`. Idempotent.
    fn stop(&mut self, py: Python<'_>) -> PyResult<()> {
        let active = match self.active.take() {
            Some(a) => a,
            None => return Ok(()),
        };
        let ActiveRecording {
            engine,
            cpal_stream,
            dispatch_handle,
            dispatch_stop,
        } = active;
        // 1. Drop the cpal stream (stops the audio callback firing).
        drop(cpal_stream);
        // 2. Stop the engine session (drops the raw-samples producer
        //    so the consumer thread drains and exits).
        let stopped = engine.stop().map_err(engine_err_to_py)?;
        // 3. Signal the dispatch thread to wind down once its rings
        //    are empty, then join it. Release the GIL while joining
        //    so the dispatch thread can keep acquiring it.
        dispatch_stop.store(true, Ordering::Release);
        py.detach(|| {
            let _ = dispatch_handle.join();
        });
        self.stopped = Some(stopped);
        Ok(())
    }

    /// Atomically commits the recording into the parent project.
    /// Returns the new bundle id. Errors if the session has not been
    /// stopped, or if a bundle named `name` already exists.
    fn commit(&mut self, project: &Bound<'_, crate::PyProject>) -> PyResult<i64> {
        let stopped = self
            .stopped
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("call stop() before commit()"))?;
        let params_json = self.params_json(&stopped);
        let pyproj = project.borrow();
        pyproj
            .inner
            .commit_recording(stopped, &self.name, &params_json)
            .map_err(engine_err_to_py)
    }

    /// Discards the in-progress recording (deletes the directory).
    /// Errors if the session has not been stopped.
    fn discard(&mut self) -> PyResult<()> {
        let stopped = self
            .stopped
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("call stop() before discard()"))?;
        stopped.discard().map_err(engine_err_to_py)
    }

    /// Number of samples (post-downmix mono frames) written to disk.
    #[getter]
    fn frames_written(&self) -> usize {
        self.stopped.as_ref().map(|s| s.frames_written).unwrap_or(0)
    }

    /// Number of samples dropped due to consumer back-pressure.
    /// Non-zero means the recording is imperfect.
    #[getter]
    fn dropped_samples(&self) -> usize {
        self.stopped
            .as_ref()
            .map(|s| s.dropped_samples)
            .unwrap_or(0)
    }

    /// Duration of the recording in seconds. `0.0` while still
    /// capturing.
    #[getter]
    fn duration_seconds(&self) -> f64 {
        self.stopped
            .as_ref()
            .map(|s| s.duration_seconds())
            .unwrap_or(0.0)
    }

    /// Path to the `.in_progress/<uuid>/` directory as a string.
    #[getter]
    fn in_progress_dir(&self) -> String {
        self.in_progress_dir.to_string_lossy().into_owned()
    }

    /// **Test-only.** Pushes synthetic samples directly into the
    /// engine's raw-samples ring, bypassing cpal. Used by the Python
    /// test suite since CI has no audio hardware. Returns the count
    /// of dropped samples.
    fn push_samples_for_tests(&mut self, samples: Vec<f32>) -> PyResult<usize> {
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("session already stopped"))?;
        if active.cpal_stream.is_some() {
            return Err(PyRuntimeError::new_err(
                "push_samples_for_tests cannot be used after start()",
            ));
        }
        Ok(active.engine.push_samples(&samples))
    }
}

impl PyLiveSession {
    fn params_json(&self, stopped: &StoppedSession) -> String {
        format!(
            "{{\"device\":{:?},\"sample_rate\":{},\"channels\":{},\
             \"duration_s\":{},\"analysis_window_ms\":{},\
             \"analysis_hop_ms\":{},\"num_formants\":{},\
             \"dropped_samples\":{}}}",
            self.device_label,
            self.config.sample_rate,
            self.config.channels,
            stopped.duration_seconds(),
            self.config.analysis_window_ms,
            self.config.analysis_hop_ms,
            self.config.num_formants,
            stopped.dropped_samples,
        )
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Constructs a new live-recording session. `device` is a device name
/// from [`list_input_devices`] or `"default"`. The cpal stream is
/// *not* built here — call `session.start()` to actually begin capture.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    project, *,
    device = "default".to_string(),
    sample_rate = 44_100,
    channels = 1,
    name = None,
    analysis_window_ms = 30.0,
    analysis_hop_ms = 10.0,
    num_formants = 4,
    min_pitch_hz = 75.0,
    max_pitch_hz = 500.0,
))]
pub(crate) fn start_session(
    project: &Bound<'_, crate::PyProject>,
    device: String,
    sample_rate: u32,
    channels: u16,
    name: Option<String>,
    analysis_window_ms: f32,
    analysis_hop_ms: f32,
    num_formants: usize,
    min_pitch_hz: f32,
    max_pitch_hz: f32,
) -> PyResult<PyLiveSession> {
    let cfg = LiveConfig {
        sample_rate,
        channels,
        analysis_window_ms,
        analysis_hop_ms,
        num_formants,
        min_pitch_hz,
        max_pitch_hz,
    };
    let root: PathBuf = {
        let pyproj = project.borrow();
        pyproj.inner.root().to_path_buf()
    };
    let name = name.unwrap_or_else(default_session_name);

    let (engine, results) =
        EngineLiveSession::start(&root, cfg.clone()).map_err(engine_err_to_py)?;
    let in_progress_dir = engine.in_progress_dir().to_path_buf();

    let callbacks = Arc::new(Mutex::new(Callbacks::default()));
    let dispatch_stop = Arc::new(AtomicBool::new(false));
    let dispatch_state = DispatchState {
        results,
        callbacks: Arc::clone(&callbacks),
        stop_flag: Arc::clone(&dispatch_stop),
    };
    let dispatch_handle = thread::Builder::new()
        .name("sadda-live-dispatch".into())
        .spawn(move || dispatch_state.run())
        .map_err(|e| PyRuntimeError::new_err(format!("spawn dispatch thread: {e}")))?;

    Ok(PyLiveSession {
        active: Some(ActiveRecording {
            engine,
            cpal_stream: None,
            dispatch_handle,
            dispatch_stop,
        }),
        stopped: None,
        callbacks,
        config: cfg,
        device_label: device,
        name,
        in_progress_dir,
    })
}

/// Returns the names of available cpal input devices on the default
/// host. Names may not be unique across hosts; for stable identifiers
/// across reboots, hold onto the string returned by
/// [`default_input_device`] and pass it back to [`start_session`].
#[pyfunction]
pub(crate) fn list_input_devices() -> PyResult<Vec<String>> {
    let host = cpal::default_host();
    let devices = host
        .input_devices()
        .map_err(|e| PyRuntimeError::new_err(format!("cpal input_devices: {e}")))?;
    let mut names = Vec::new();
    for d in devices {
        if let Ok(n) = device_label(&d) {
            names.push(n);
        }
    }
    Ok(names)
}

/// Returns the default input device's name, or `None` if there isn't
/// one.
#[pyfunction]
pub(crate) fn default_input_device() -> Option<String> {
    let host = cpal::default_host();
    host.default_input_device()
        .and_then(|d| device_label(&d).ok())
}

fn pick_input_device(label: &str) -> PyResult<cpal::Device> {
    let host = cpal::default_host();
    if label == "default" {
        return host
            .default_input_device()
            .ok_or_else(|| PyRuntimeError::new_err("no default input device available"));
    }
    let devices = host
        .input_devices()
        .map_err(|e| PyRuntimeError::new_err(format!("cpal input_devices: {e}")))?;
    for d in devices {
        if let Ok(n) = device_label(&d) {
            if n == label {
                return Ok(d);
            }
        }
    }
    Err(PyValueError::new_err(format!(
        "no input device named {label:?}"
    )))
}

#[allow(deprecated)]
fn device_label(d: &cpal::Device) -> Result<String, cpal::DeviceNameError> {
    // cpal 0.17 deprecated `name()` in favour of `description()` /
    // `id()`. For now we keep the simpler one-word `name()` surface;
    // upgrading to `description()` is tracked as a 0.1.x polish item.
    d.name()
}

fn build_stream_config(
    device: &cpal::Device,
    cfg: &LiveConfig,
) -> PyResult<cpal::StreamConfig> {
    let default = device
        .default_input_config()
        .map_err(|e| PyRuntimeError::new_err(format!("default_input_config: {e}")))?;
    let mut stream_cfg: cpal::StreamConfig = default.config();
    stream_cfg.sample_rate = cfg.sample_rate;
    stream_cfg.channels = cfg.channels;
    Ok(stream_cfg)
}

fn default_session_name() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("recording_{secs}")
}
