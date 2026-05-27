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

/// Probes that the ONNX Runtime shared library is loadable *before*
/// invoking `ort` — which otherwise `panic!`s in its lazy loader on a
/// missing runtime (a library must not abort the process for an absent
/// optional dependency). Resolves the same path `ort` will: the
/// `ORT_DYLIB_PATH` env var, else the platform default name on the
/// system search path.
pub(crate) fn ensure_ort_available() -> Result<()> {
    let path = std::env::var("ORT_DYLIB_PATH").unwrap_or_else(|_| default_ort_name().to_string());
    // SAFETY: opening a shared library for a load check; the handle is
    // dropped immediately (dlopen is ref-counted, so `ort`'s later load
    // is unaffected). No symbols from it are called here.
    match unsafe { libloading::Library::new(&path) } {
        Ok(_handle) => Ok(()),
        Err(e) => Err(EngineError::Ml(format!(
            "ONNX Runtime not loadable ({path}): {e}. Set ORT_DYLIB_PATH to a \
             libonnxruntime shared library (see the 2026-05-26 E11 DEVLOG entry)."
        ))),
    }
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
    let n_windows = samples.len() / VAD_WINDOW;
    let mut frames = Vec::with_capacity(n_windows);

    for w in 0..n_windows {
        let chunk = samples[w * VAD_WINDOW..(w + 1) * VAD_WINDOW].to_vec();
        let input = Tensor::from_array(([1usize, VAD_WINDOW], chunk)).map_err(ml_err)?;
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
}
