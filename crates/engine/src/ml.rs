//! ML inference (Phase 3 cluster E / E11) — ONNX models via ONNX Runtime.
//!
//! Behind the `ml` cargo feature (default off). ONNX Runtime is **not**
//! linked at build time (`ort`'s `load-dynamic`); it is loaded at runtime
//! from `ORT_DYLIB_PATH` (or the system library search path) — point it
//! at a `libonnxruntime.so` / `.dylib` / `onnxruntime.dll`. This keeps the
//! ~23 MB runtime out of every build artifact (it's an optional sidecar);
//! see the 2026-05-26 "ort-bundling spike" DEVLOG entry.
//!
//! The first model is the bundled **Silero VAD** (`models-bundled/`), via
//! [`vad`] / [`vad_bundled`]. Richer surfaces (Python `sadda.ml`, a GUI
//! VAD tier) and the on-demand model registry are later E11/E12 slices.

use std::path::Path;

use ort::value::Tensor;

use crate::audio::Audio;
use crate::error::{EngineError, Result};

/// Sample rate the Silero VAD model operates at.
pub const VAD_SAMPLE_RATE: u32 = 16_000;
/// Samples per Silero VAD analysis window at 16 kHz (the model's fixed
/// frame — 32 ms).
pub const VAD_WINDOW: usize = 512;
/// Samples of previous-window context the Silero 2024 model prepends to each
/// window (its internal `_context`). The model input is therefore
/// `VAD_CONTEXT + VAD_WINDOW = 576` samples; feeding a bare `VAD_WINDOW` makes
/// the model return ~0 for every window (it never sees the lookahead it was
/// trained with).
const VAD_CONTEXT: usize = 64;
/// Length of the Silero recurrent state tensor (`[2, 1, 128]` flattened).
const VAD_STATE_LEN: usize = 2 * 128;

fn ml_err(e: impl std::fmt::Display) -> EngineError {
    EngineError::Ml(e.to_string())
}

/// Platform default ONNX Runtime library name, matching `ort`'s
/// `load-dynamic` resolution when `ORT_DYLIB_PATH` is unset.
fn default_ort_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "onnxruntime.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "libonnxruntime.dylib"
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        "libonnxruntime.so"
    }
}

/// Probes that the ONNX Runtime shared library at `path` is both
/// loadable *and* actually the runtime, *before* invoking `ort` — which
/// otherwise `panic!`s in its lazy loader on a bad runtime (a library
/// must not abort the process for an absent/misconfigured optional
/// dependency). Two failure modes, two distinct errors:
///
/// * the library can't be `dlopen`ed at all (missing, wrong arch); or
/// * it loads but doesn't export the ORT C API entry point
///   `OrtGetApiBase`, so it is a valid shared object but *not* the
///   runtime. The classic trip-up is pointing `ORT_DYLIB_PATH` at
///   `libonnxruntime_providers_shared.so` (the small provider shim),
///   which opens cleanly yet has none of the ORT symbols — a bare
///   `dlopen` check would wave it through and `ort` would then fail
///   opaquely downstream.
///
/// Public so the app shell can validate a sidecar dylib before setting
/// `ORT_DYLIB_PATH` from it at startup (avoiding a startup-time discovery
/// that points the runtime at a bogus file).
pub fn probe_ort_dylib(path: &str) -> Result<()> {
    // SAFETY: opening a shared library for a load check; the handle is
    // dropped at function end (dlopen is ref-counted, so `ort`'s later
    // load is unaffected). No symbols from it are ever called.
    let lib = unsafe { libloading::Library::new(path) }.map_err(|e| {
        EngineError::Ml(format!(
            "ONNX Runtime not loadable ({path}): {e}. Set ORT_DYLIB_PATH to a \
             libonnxruntime shared library (see the 2026-05-26 E11 DEVLOG entry)."
        ))
    })?;
    // SAFETY: resolving a symbol's address only; it is never called.
    if unsafe { lib.get::<unsafe extern "C" fn() -> *const std::ffi::c_void>(b"OrtGetApiBase\0") }
        .is_err()
    {
        return Err(EngineError::Ml(format!(
            "the shared library at {path} loads but does not export `OrtGetApiBase`, \
             so it is not the ONNX Runtime (the usual cause is pointing ORT_DYLIB_PATH \
             at `libonnxruntime_providers_shared.so`, the provider shim). Set \
             ORT_DYLIB_PATH to the runtime itself — `libonnxruntime.so` / `.dylib` / \
             `onnxruntime.dll`; a versioned filename such as `libonnxruntime.so.1.26.0` \
             is fine."
        )));
    }
    Ok(())
}

/// Resolves the ONNX Runtime path the same way `ort` will — the
/// `ORT_DYLIB_PATH` env var, else the platform default name on the system
/// search path — and [`probe_ort_dylib`]s it before any `ort` call.
pub(crate) fn ensure_ort_available() -> Result<()> {
    let path = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| default_ort_name().to_string());
    probe_ort_dylib(&path)
}

/// One Silero-VAD window: its centre time (seconds) and the model's
/// speech probability for that window, in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadFrame {
    /// Centre time of the window, in seconds from the start of the audio.
    pub time_seconds: f64,
    /// Speech probability for the window, `[0, 1]`.
    pub speech_prob: f32,
}

/// A contiguous speech region, derived from [`VadFrame`]s above a
/// threshold by [`speech_segments`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeechSegment {
    /// Start time, seconds.
    pub start_seconds: f64,
    /// End time, seconds.
    pub end_seconds: f64,
}

/// Assembles one Silero VAD model input: `context` (the tail of the previous
/// window) followed by the current `window`. The Silero 2024 model needs this
/// 64-sample lookback prepended — feeding a bare window returns ~0 per frame.
fn vad_model_input(context: &[f32], window: &[f32]) -> Vec<f32> {
    let mut buf = Vec::with_capacity(context.len() + window.len());
    buf.extend_from_slice(context);
    buf.extend_from_slice(window);
    buf
}

/// Runs Silero VAD over `audio` using the ONNX model at `model_path`,
/// returning a speech probability for each 512-sample window at 16 kHz.
///
/// The audio is mono-mixed and resampled to [`VAD_SAMPLE_RATE`]; the
/// model's recurrent state is threaded window-to-window. Returns
/// [`EngineError::Ml`] if ONNX Runtime can't be loaded (e.g.
/// `ORT_DYLIB_PATH` unset), the model fails to load, the audio is shorter
/// than one window, or inference fails.
pub fn vad(audio: &Audio, model_path: &Path) -> Result<Vec<VadFrame>> {
    let mono: Vec<f32> = audio.mono_samples().collect();
    let samples = crate::dsp::resample_to_hz(&mono, audio.sample_rate, VAD_SAMPLE_RATE);
    if samples.len() < VAD_WINDOW {
        return Err(EngineError::Ml(format!(
            "audio too short for VAD: {} samples at {VAD_SAMPLE_RATE} Hz (need ≥ {VAD_WINDOW})",
            samples.len()
        )));
    }

    // Convert "no ONNX Runtime" into a clean error rather than letting
    // `ort` panic in its lazy loader.
    ensure_ort_available()?;
    let mut session = ort::session::Session::builder()
        .map_err(ml_err)?
        .commit_from_file(model_path)
        .map_err(ml_err)?;

    let mut state = vec![0.0f32; VAD_STATE_LEN];
    // The model input is `[context(64) ++ window(512)]`; the context is the tail
    // of the previous window, zero for the first. Without it the model returns
    // ~0 for every window (Silero 2024's `_context` mechanism).
    let mut context = vec![0.0f32; VAD_CONTEXT];
    let n_windows = samples.len() / VAD_WINDOW;
    let mut frames = Vec::with_capacity(n_windows);

    for w in 0..n_windows {
        let chunk = &samples[w * VAD_WINDOW..(w + 1) * VAD_WINDOW];
        let input_buf = vad_model_input(&context, chunk);
        let input =
            Tensor::from_array(([1usize, VAD_CONTEXT + VAD_WINDOW], input_buf)).map_err(ml_err)?;
        let state_t = Tensor::from_array(([2usize, 1, 128], state.clone())).map_err(ml_err)?;
        // sr as a 1-element i64 tensor — accepted by the model's scalar
        // `sr` input (verified in the spike).
        let sr = Tensor::from_array(([1usize], vec![VAD_SAMPLE_RATE as i64])).map_err(ml_err)?;

        let outputs = session
            .run(ort::inputs!["input" => input, "state" => state_t, "sr" => sr])
            .map_err(ml_err)?;

        let speech_prob = {
            let (_shape, prob) = outputs["output"]
                .try_extract_tensor::<f32>()
                .map_err(ml_err)?;
            *prob
                .first()
                .ok_or_else(|| EngineError::Ml("empty VAD output".into()))?
        };
        {
            let (_shape, next) = outputs["stateN"]
                .try_extract_tensor::<f32>()
                .map_err(ml_err)?;
            if next.len() == VAD_STATE_LEN {
                state.copy_from_slice(next);
            }
        }

        // Carry the tail of this window as the next window's context.
        context.copy_from_slice(&chunk[VAD_WINDOW - VAD_CONTEXT..]);

        let centre_sample = w * VAD_WINDOW + VAD_WINDOW / 2;
        frames.push(VadFrame {
            time_seconds: centre_sample as f64 / VAD_SAMPLE_RATE as f64,
            speech_prob,
        });
    }
    Ok(frames)
}

/// Merges consecutive [`VadFrame`]s whose probability is `>= threshold`
/// into [`SpeechSegment`]s. Each window spans [`VAD_WINDOW`] samples, so a
/// segment runs from the first qualifying window's start to the last's
/// end. Pure — no ONNX Runtime needed (independently testable).
pub fn speech_segments(frames: &[VadFrame], threshold: f32) -> Vec<SpeechSegment> {
    let half = (VAD_WINDOW as f64 / VAD_SAMPLE_RATE as f64) / 2.0;
    let mut segments = Vec::new();
    let mut start: Option<f64> = None;
    let mut last_end = 0.0;
    for f in frames {
        if f.speech_prob >= threshold {
            let w_start = (f.time_seconds - half).max(0.0);
            let w_end = f.time_seconds + half;
            if start.is_none() {
                start = Some(w_start);
            }
            last_end = w_end;
        } else if let Some(s) = start.take() {
            segments.push(SpeechSegment {
                start_seconds: s,
                end_seconds: last_end,
            });
        }
    }
    if let Some(s) = start {
        segments.push(SpeechSegment {
            start_seconds: s,
            end_seconds: last_end,
        });
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vad_input_prepends_context_to_window() {
        // Guards the Silero 2024 calling convention: the model input is
        // context(64) ++ window(512) = 576, NOT a bare 512 (which returns ~0).
        let ctx = vec![9.0_f32; VAD_CONTEXT];
        let win = vec![1.0_f32; VAD_WINDOW];
        let inp = vad_model_input(&ctx, &win);
        assert_eq!(inp.len(), VAD_CONTEXT + VAD_WINDOW);
        assert_eq!(VAD_CONTEXT + VAD_WINDOW, 576);
        assert_eq!(inp[0], 9.0, "context comes first");
        assert_eq!(inp[VAD_CONTEXT], 1.0, "then the window");
    }

    #[test]
    fn speech_segments_merges_runs_above_threshold() {
        // Windows at ~0.016, 0.048, 0.080, 0.112 s (centres). A run of
        // two speech windows, a gap, then one more speech window.
        let dt = VAD_WINDOW as f64 / VAD_SAMPLE_RATE as f64;
        let half = dt / 2.0;
        let frames: Vec<VadFrame> = [0.9, 0.8, 0.1, 0.7]
            .iter()
            .enumerate()
            .map(|(i, &p)| VadFrame {
                time_seconds: (i * VAD_WINDOW + VAD_WINDOW / 2) as f64 / VAD_SAMPLE_RATE as f64,
                speech_prob: p,
            })
            .collect();
        let segs = speech_segments(&frames, 0.5);
        assert_eq!(segs.len(), 2);
        // First segment spans windows 0..1.
        assert!((segs[0].start_seconds - (half - half).max(0.0)).abs() < 1e-9);
        assert!(segs[0].end_seconds > segs[0].start_seconds);
        // Second segment is the lone window 3.
        assert!(segs[1].start_seconds > segs[0].end_seconds);
        let _ = dt;
    }

    #[test]
    fn speech_segments_empty_when_all_below() {
        let frames = vec![VadFrame {
            time_seconds: 0.0,
            speech_prob: 0.1,
        }];
        assert!(speech_segments(&frames, 0.5).is_empty());
    }

    #[test]
    fn probe_ort_rejects_missing_library() {
        let err = probe_ort_dylib("/nonexistent/libonnxruntime.so.999")
            .expect_err("a missing path must not pass the probe");
        let EngineError::Ml(msg) = err else {
            panic!("expected EngineError::Ml, got {err:?}");
        };
        assert!(msg.contains("not loadable"), "unexpected message: {msg}");
    }

    #[test]
    fn probe_ort_rejects_non_ort_library() {
        // A guaranteed-present system C library is a valid, loadable shared
        // object that does not export `OrtGetApiBase` — a portable stand-in
        // for a "wrong .so" (e.g. the provider shim), needing no ONNX
        // Runtime. (Can't use the test executable: Linux refuses to dlopen
        // a position-independent executable.)
        #[cfg(target_os = "linux")]
        let sys_lib = "libc.so.6";
        #[cfg(target_os = "macos")]
        let sys_lib = "libSystem.B.dylib";
        #[cfg(target_os = "windows")]
        let sys_lib = "kernel32.dll";

        match probe_ort_dylib(sys_lib) {
            Err(EngineError::Ml(msg)) if msg.contains("OrtGetApiBase") => {}
            // An unusual libc layout (e.g. musl) may not load under this
            // name; then the symbol branch isn't reached here, but the
            // missing-library branch is covered by the test above.
            Err(EngineError::Ml(msg)) if msg.contains("not loadable") => {
                eprintln!("skipping: {sys_lib} not dlopen-able here: {msg}");
            }
            other => panic!("expected the OrtGetApiBase rejection, got {other:?}"),
        }
    }
}
