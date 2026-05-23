//! C5 audio playback. Builds a cpal output stream that reads from a
//! pre-loaded mono mixdown (possibly linear-resampled to the device
//! sample rate) and advances an atomic cursor the GUI polls each
//! frame.
//!
//! Threading model: cpal's output callback runs on a real-time audio
//! thread. The callback reads samples (immutable shared `Arc<Vec<f32>>`)
//! and advances an `AtomicUsize`; both lock-free. End-of-audio
//! signals via an `AtomicBool` the main thread polls. No `rtrb`
//! needed because the entire sample buffer is in memory — the
//! callback is "play a static buffer with a moving read head," not
//! "stream samples through a ring" the way E1 input does.
//!
//! cpal::Stream is `!Send` on Linux ALSA. The `Playback` struct
//! stays on the main thread (where `SaddaApp` lives); we don't try
//! to move it across threads.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// Live playback handle. Drop it to stop. cpal's `Stream` Drop impl
/// stops the audio callback cleanly.
pub struct Playback {
    /// Kept around so the audio callback keeps firing until the
    /// `Playback` is dropped. The leading underscore silences the
    /// "field never read" warning; the field's lifetime *is* the
    /// behaviour.
    _stream: cpal::Stream,
    state: Arc<PlaybackState>,
    /// Sample rate of the audio actually being sent to the device
    /// (may differ from the bundle's source rate if we resampled).
    output_sample_rate: u32,
    /// Original bundle sample rate; used to map the atomic cursor
    /// back to bundle time when source ≠ output rate.
    bundle_sample_rate: u32,
}

struct PlaybackState {
    /// Mono samples to play. Possibly linear-resampled from the
    /// bundle's source rate to the device's preferred rate.
    samples: Vec<f32>,
    /// Current read position in samples within `samples`. The audio
    /// callback advances; the GUI reads (Relaxed is fine — we tolerate
    /// a one-frame stale read).
    cursor: AtomicUsize,
    /// Set to `true` once the callback writes the last sample.
    finished: AtomicBool,
}

impl Playback {
    /// Starts playback of the given mono mixdown from a cursor
    /// `start_seconds` into the bundle. Picks the system default
    /// output device + its preferred sample rate; if the device
    /// rate differs from `bundle_sample_rate`, the samples are
    /// linear-resampled once at start.
    ///
    /// On error returns a human-readable string suitable for the
    /// app's error banner. cpal failures (no device, format
    /// mismatch the engine couldn't reconcile) are the main hazards.
    pub fn start(
        bundle_mono: &[f32],
        bundle_sample_rate: u32,
        start_seconds: f64,
    ) -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| "No default audio output device available".to_string())?;
        let supported = device
            .default_output_config()
            .map_err(|e| format!("default_output_config failed: {e}"))?;

        // cpal 0.17 made `SampleRate` a plain `u32` type alias (no
        // `.0` accessor, no tuple-struct constructor).
        let output_sample_rate: u32 = supported.sample_rate();
        let output_channels = supported.channels();
        let sample_format = supported.sample_format();

        // Resample the bundle's mono mixdown to the device rate
        // once at start. Linear interp is fine for monitoring.
        let resampled = if output_sample_rate == bundle_sample_rate {
            bundle_mono.to_vec()
        } else {
            linear_resample(bundle_mono, bundle_sample_rate, output_sample_rate)
        };

        // Convert the bundle-relative start time to an output-rate
        // sample index.
        let start_sample = ((start_seconds.max(0.0)) * output_sample_rate as f64).round() as usize;
        let start_sample = start_sample.min(resampled.len());

        let state = Arc::new(PlaybackState {
            samples: resampled,
            cursor: AtomicUsize::new(start_sample),
            finished: AtomicBool::new(false),
        });

        // Build a stream config matching the device's preferred
        // sample rate but our (mono) channel count fanned out to
        // however many channels the device wants. Most output
        // devices are stereo; we duplicate the mono sample across
        // all channels.
        let stream_config = cpal::StreamConfig {
            channels: output_channels,
            sample_rate: output_sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        let stream = build_stream(
            &device,
            &stream_config,
            sample_format,
            output_channels as usize,
            Arc::clone(&state),
        )?;

        stream
            .play()
            .map_err(|e| format!("Stream.play failed: {e}"))?;

        Ok(Self {
            _stream: stream,
            state,
            output_sample_rate,
            bundle_sample_rate,
        })
    }

    /// Returns the current cursor position in bundle-time seconds.
    /// Maps the output-rate atomic back to the bundle's clock so
    /// the GUI's `timeline.cursor` stays in bundle-time regardless
    /// of any resampling.
    pub fn cursor_seconds(&self) -> f64 {
        let cursor = self.state.cursor.load(Ordering::Relaxed);
        let secs_in_output = cursor as f64 / self.output_sample_rate as f64;
        // bundle_time = output_time exactly; resampling preserves
        // wall-clock duration. Kept as a separate calculation in case
        // we ever drift (e.g. drop-frame or jitter-correcting
        // resamplers).
        let _ = self.bundle_sample_rate;
        secs_in_output
    }

    /// Whether the cursor has reached the end of the buffer.
    /// Callers should drop `self` when this returns `true`.
    pub fn is_finished(&self) -> bool {
        self.state.finished.load(Ordering::Relaxed)
    }
}

fn build_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    channels: usize,
    state: Arc<PlaybackState>,
) -> Result<cpal::Stream, String> {
    let err_fn = |err| eprintln!("sadda playback cpal error: {err}");

    match sample_format {
        cpal::SampleFormat::F32 => device
            .build_output_stream(
                config,
                move |data: &mut [f32], _info| fill_buffer_f32(data, channels, &state),
                err_fn,
                None,
            )
            .map_err(|e| format!("build_output_stream(F32) failed: {e}")),
        cpal::SampleFormat::I16 => device
            .build_output_stream(
                config,
                move |data: &mut [i16], _info| fill_buffer_i16(data, channels, &state),
                err_fn,
                None,
            )
            .map_err(|e| format!("build_output_stream(I16) failed: {e}")),
        cpal::SampleFormat::U16 => device
            .build_output_stream(
                config,
                move |data: &mut [u16], _info| fill_buffer_u16(data, channels, &state),
                err_fn,
                None,
            )
            .map_err(|e| format!("build_output_stream(U16) failed: {e}")),
        other => Err(format!("Unsupported sample format: {other:?}")),
    }
}

/// Audio-thread callback for F32 output devices. Writes the next
/// `data.len() / channels` mono samples, fanned out across all
/// channels. RT-safe: no allocations, no locks.
fn fill_buffer_f32(data: &mut [f32], channels: usize, state: &PlaybackState) {
    let frames = data.len() / channels;
    let start = state.cursor.load(Ordering::Relaxed);
    let n_total = state.samples.len();
    for f in 0..frames {
        let i = start + f;
        let sample = if i < n_total { state.samples[i] } else { 0.0 };
        for c in 0..channels {
            data[f * channels + c] = sample;
        }
    }
    let new_cursor = (start + frames).min(n_total);
    state.cursor.store(new_cursor, Ordering::Relaxed);
    if new_cursor >= n_total {
        state.finished.store(true, Ordering::Relaxed);
    }
}

fn fill_buffer_i16(data: &mut [i16], channels: usize, state: &PlaybackState) {
    let frames = data.len() / channels;
    let start = state.cursor.load(Ordering::Relaxed);
    let n_total = state.samples.len();
    for f in 0..frames {
        let i = start + f;
        let sample = if i < n_total { state.samples[i] } else { 0.0 };
        let s_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        for c in 0..channels {
            data[f * channels + c] = s_i16;
        }
    }
    let new_cursor = (start + frames).min(n_total);
    state.cursor.store(new_cursor, Ordering::Relaxed);
    if new_cursor >= n_total {
        state.finished.store(true, Ordering::Relaxed);
    }
}

fn fill_buffer_u16(data: &mut [u16], channels: usize, state: &PlaybackState) {
    let frames = data.len() / channels;
    let start = state.cursor.load(Ordering::Relaxed);
    let n_total = state.samples.len();
    for f in 0..frames {
        let i = start + f;
        let sample = if i < n_total { state.samples[i] } else { 0.0 };
        // U16 mid-scale is 32768; map [-1, 1] → [0, 65535].
        let s_u16 = ((sample.clamp(-1.0, 1.0) * 0.5 + 0.5) * u16::MAX as f32) as u16;
        for c in 0..channels {
            data[f * channels + c] = s_u16;
        }
    }
    let new_cursor = (start + frames).min(n_total);
    state.cursor.store(new_cursor, Ordering::Relaxed);
    if new_cursor >= n_total {
        state.finished.store(true, Ordering::Relaxed);
    }
}

/// Linear-interpolation resampler. Input rate → output rate, mono
/// f32 in / out. Pure-data; testable without an audio device.
pub fn linear_resample(input: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if input.is_empty() || input_rate == 0 || output_rate == 0 {
        return Vec::new();
    }
    if input_rate == output_rate {
        return input.to_vec();
    }
    let ratio = input_rate as f64 / output_rate as f64;
    let out_len = ((input.len() as f64) / ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let i0 = src_pos.floor() as usize;
        let i1 = (i0 + 1).min(input.len() - 1);
        let frac = (src_pos - i0 as f64) as f32;
        let s0 = input[i0];
        let s1 = input[i1];
        out.push(s0 + (s1 - s0) * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_resample_identity_when_rates_match() {
        let s = vec![0.1, 0.2, 0.3, 0.4];
        let out = linear_resample(&s, 16_000, 16_000);
        assert_eq!(out, s);
    }

    #[test]
    fn linear_resample_handles_empty() {
        assert!(linear_resample(&[], 16_000, 44_100).is_empty());
        assert!(linear_resample(&[0.1, 0.2], 0, 44_100).is_empty());
        assert!(linear_resample(&[0.1, 0.2], 16_000, 0).is_empty());
    }

    #[test]
    fn linear_resample_double_rate_doubles_length() {
        let s = vec![0.0_f32, 1.0, 0.0, 1.0];
        let out = linear_resample(&s, 8_000, 16_000);
        assert_eq!(out.len(), 8);
        // First sample matches exactly.
        assert!((out[0] - 0.0).abs() < 1e-6);
        // Sample at i=2 (input index 1) matches s[1] exactly.
        assert!((out[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn linear_resample_half_rate_halves_length() {
        let s = vec![0.0_f32, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5];
        let out = linear_resample(&s, 16_000, 8_000);
        assert_eq!(out.len(), 4);
        // out[0] = s[0]; out[1] = s[2]; out[2] = s[4]; out[3] = s[6].
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
        assert!((out[2] - 0.0).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
    }
}
