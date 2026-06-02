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
//! A run plays a **span** `[start_sample, end_sample)` rather than the
//! whole buffer, and can **loop** (with a silent inter-repetition gap)
//! and **pause** (hold the read head, emit silence). All of that lives
//! in [`next_mono_sample`], a pure state-machine step the GUI keymap
//! drives — and which the tests exercise without an audio device.
//!
//! cpal::Stream is `!Send` on Linux ALSA. The `Playback` struct
//! stays on the main thread (where `SaddaApp` lives); we don't try
//! to move it across threads.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

/// How a span should play: once through, or repeat with a fixed silent gap
/// between repetitions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoopMode {
    /// Play the span once, then finish.
    Once,
    /// Loop the span indefinitely, inserting `pause_seconds` of silence between
    /// each repetition. Stop by dropping the `Playback`.
    Loop { pause_seconds: f64 },
}

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
    /// First sample of the span (output-rate index). The cursor resets
    /// here on each loop repetition.
    start_sample: usize,
    /// One-past-the-last sample of the span (output-rate index). Playback
    /// stops (or loops) when the cursor reaches it.
    end_sample: usize,
    /// Current read position in samples within `samples`. The audio
    /// callback advances; the GUI reads (Relaxed is fine — we tolerate
    /// a one-frame stale read).
    cursor: AtomicUsize,
    /// Set to `true` once a non-looping span reaches its end.
    finished: AtomicBool,
    /// While `true`, the callback emits silence and does not advance the
    /// cursor (pause). Toggled from the GUI thread.
    paused: AtomicBool,
    /// Whether to restart at `start_sample` instead of finishing at the end.
    looping: bool,
    /// Silent frames inserted between loop repetitions (output-rate). Zero for
    /// `Once` or a zero-pause loop.
    loop_pause_samples: usize,
    /// Countdown of remaining inter-repetition silent frames. When > 0 the
    /// callback emits silence and holds the cursor at `start_sample`.
    loop_pause_remaining: AtomicUsize,
}

impl Playback {
    /// Starts playback of the span `[start_seconds, end_seconds)` of the given
    /// mono mixdown. `end_seconds == None` plays to the end of the buffer.
    /// `loop_mode` selects once-through vs looping (with a silent gap). Picks
    /// the system default output device + its preferred sample rate; if the
    /// device rate differs from `bundle_sample_rate`, the samples are
    /// linear-resampled once at start.
    ///
    /// On error returns a human-readable string suitable for the app's error
    /// banner. cpal failures (no device, format mismatch) are the main hazards.
    pub fn start_span(
        bundle_mono: &[f32],
        bundle_sample_rate: u32,
        start_seconds: f64,
        end_seconds: Option<f64>,
        loop_mode: LoopMode,
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

        let to_sample = |secs: f64| (secs.max(0.0) * output_sample_rate as f64).round() as usize;
        let start_sample = to_sample(start_seconds).min(resampled.len());
        let end_sample = end_seconds
            .map(|e| to_sample(e).min(resampled.len()))
            .unwrap_or(resampled.len())
            .max(start_sample);

        let (looping, loop_pause_samples) = match loop_mode {
            LoopMode::Once => (false, 0),
            LoopMode::Loop { pause_seconds } => (true, to_sample(pause_seconds)),
        };

        let state = Arc::new(PlaybackState {
            samples: resampled,
            start_sample,
            end_sample,
            cursor: AtomicUsize::new(start_sample),
            finished: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            looping,
            loop_pause_samples,
            loop_pause_remaining: AtomicUsize::new(0),
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

    /// Whether a non-looping span has reached its end. Callers should drop
    /// `self` when this returns `true`. Always `false` for a looping span.
    pub fn is_finished(&self) -> bool {
        self.state.finished.load(Ordering::Relaxed)
    }

    /// Pauses (holds the read head, emits silence) or resumes playback.
    pub fn set_paused(&self, paused: bool) {
        self.state.paused.store(paused, Ordering::Relaxed);
    }

    /// Whether playback is currently paused.
    pub fn is_paused(&self) -> bool {
        self.state.paused.load(Ordering::Relaxed)
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

/// One step of the playback state machine: returns the next mono sample to emit
/// and advances the cursor / loop-pause / finished state accordingly. RT-safe
/// (atomics only, no allocation, no locks) and pure enough to unit-test by
/// hand-constructing a [`PlaybackState`].
///
/// Order of precedence: paused → inter-loop pause → span body → end (loop or
/// finish). On a loop wrap the cursor resets to `start_sample` and a single
/// frame of silence is emitted (the gap, or one sample if the gap is zero).
fn next_mono_sample(state: &PlaybackState) -> f32 {
    if state.paused.load(Ordering::Relaxed) {
        return 0.0;
    }
    let pause_left = state.loop_pause_remaining.load(Ordering::Relaxed);
    if pause_left > 0 {
        state
            .loop_pause_remaining
            .store(pause_left - 1, Ordering::Relaxed);
        return 0.0;
    }
    let cursor = state.cursor.load(Ordering::Relaxed);
    if cursor >= state.end_sample {
        if state.looping {
            if state.loop_pause_samples > 0 {
                state
                    .loop_pause_remaining
                    .store(state.loop_pause_samples, Ordering::Relaxed);
            }
            state.cursor.store(state.start_sample, Ordering::Relaxed);
        } else {
            state.finished.store(true, Ordering::Relaxed);
        }
        return 0.0;
    }
    let sample = state.samples.get(cursor).copied().unwrap_or(0.0);
    state.cursor.store(cursor + 1, Ordering::Relaxed);
    sample
}

/// Audio-thread callback for F32 output devices. Writes the next
/// `data.len() / channels` mono samples (via [`next_mono_sample`]), fanned out
/// across all channels. RT-safe: no allocations, no locks.
fn fill_buffer_f32(data: &mut [f32], channels: usize, state: &PlaybackState) {
    let frames = data.len() / channels;
    for f in 0..frames {
        let sample = next_mono_sample(state);
        for c in 0..channels {
            data[f * channels + c] = sample;
        }
    }
}

fn fill_buffer_i16(data: &mut [i16], channels: usize, state: &PlaybackState) {
    let frames = data.len() / channels;
    for f in 0..frames {
        let sample = next_mono_sample(state);
        let s_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        for c in 0..channels {
            data[f * channels + c] = s_i16;
        }
    }
}

fn fill_buffer_u16(data: &mut [u16], channels: usize, state: &PlaybackState) {
    let frames = data.len() / channels;
    for f in 0..frames {
        let sample = next_mono_sample(state);
        // U16 mid-scale is 32768; map [-1, 1] → [0, 65535].
        let s_u16 = ((sample.clamp(-1.0, 1.0) * 0.5 + 0.5) * u16::MAX as f32) as u16;
        for c in 0..channels {
            data[f * channels + c] = s_u16;
        }
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

    /// Builds a `PlaybackState` over `[start, end)` of `samples` for driving
    /// `next_mono_sample` directly (no audio device).
    fn state(
        samples: Vec<f32>,
        start: usize,
        end: usize,
        looping: bool,
        loop_pause_samples: usize,
    ) -> PlaybackState {
        PlaybackState {
            samples,
            start_sample: start,
            end_sample: end,
            cursor: AtomicUsize::new(start),
            finished: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            looping,
            loop_pause_samples,
            loop_pause_remaining: AtomicUsize::new(0),
        }
    }

    fn drain(st: &PlaybackState, n: usize) -> Vec<f32> {
        (0..n).map(|_| next_mono_sample(st)).collect()
    }

    #[test]
    fn once_plays_span_then_finishes_with_silence() {
        let st = state(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 4, false, 0);
        // Span is samples[1..4] = [2,3,4]; then a silence frame flips finished.
        let out = drain(&st, 5);
        assert_eq!(&out[..3], &[2.0, 3.0, 4.0]);
        assert_eq!(out[3], 0.0);
        assert!(st.finished.load(Ordering::Relaxed));
        assert_eq!(out[4], 0.0);
    }

    #[test]
    fn loop_repeats_span_and_never_finishes() {
        let st = state(vec![1.0, 2.0, 3.0], 0, 3, true, 0);
        let out = drain(&st, 8);
        // [1,2,3] then a one-frame wrap gap, then [1,2,3], wrap, [1] ...
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 3.0);
        assert_eq!(out[3], 0.0); // wrap frame
        assert_eq!(out[4], 1.0);
        assert_eq!(out[5], 2.0);
        assert!(!st.finished.load(Ordering::Relaxed));
    }

    #[test]
    fn loop_inserts_the_silent_pause_between_repetitions() {
        let st = state(vec![1.0, 2.0], 0, 2, true, 3);
        let out = drain(&st, 9);
        // [1,2], wrap(0) + 3 pause frames? The wrap frame schedules the pause,
        // so: 1,2, then wrap returns 0 and sets pause=3, then 3 pause frames,
        // then 1,2 again.
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0);
        assert_eq!(out[2], 0.0); // wrap (schedules pause)
        assert_eq!(out[3], 0.0); // pause 1
        assert_eq!(out[4], 0.0); // pause 2
        assert_eq!(out[5], 0.0); // pause 3
        assert_eq!(out[6], 1.0); // back to span start
        assert_eq!(out[7], 2.0);
    }

    #[test]
    fn paused_holds_the_cursor_and_emits_silence() {
        let st = state(vec![1.0, 2.0, 3.0], 0, 3, false, 0);
        assert_eq!(next_mono_sample(&st), 1.0);
        st.paused.store(true, Ordering::Relaxed);
        assert_eq!(drain(&st, 4), vec![0.0, 0.0, 0.0, 0.0]);
        st.paused.store(false, Ordering::Relaxed);
        // Resumes exactly where it paused.
        assert_eq!(next_mono_sample(&st), 2.0);
        assert_eq!(next_mono_sample(&st), 3.0);
    }

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
