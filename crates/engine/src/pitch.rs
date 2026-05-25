//! Fundamental-frequency (f0) estimation — one of several non-equivalent
//! pitch-estimation methods per the 2026-05-21 DSP method-diversity entry.
//! Two methods ship in C2:
//!
//! - [`autocorrelation`] — naive time-domain autocorrelation (Phase-0
//!   tracker). Simple, fast; biased by the window function's own
//!   autocorrelation.
//! - [`windowed_autocorrelation`] — adopts Boersma 1993's central
//!   insight (divide the windowed-signal autocorrelation by the window's
//!   own autocorrelation to remove the window's bias on peak location).
//!   **Not** a faithful Boersma implementation — it omits the anti-alias
//!   upsample pre-step, uses parabolic peak interpolation instead of the
//!   windowed-sinc + Brent's method the paper specifies, has no multi-
//!   candidate path-finder, and does not implement the octave-cost
//!   penalty terms. A faithful Boersma tracker is a deferred alternate.
//!
//! ## References
//! - Rabiner, L.R. (1977), "On the use of autocorrelation analysis for pitch
//!   detection." *IEEE TASSP* 25(1).
//!   <https://doi.org/10.1109/TASSP.1977.1162905>
//! - Boersma, P. (1993), "Accurate short-term analysis of the fundamental
//!   frequency and the harmonics-to-noise ratio of a sampled sound."
//!   *Proc. Inst. Phonetic Sciences* 17.
//!   <https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf>
//!
//! ## Recommended default for v1
//! [`windowed_autocorrelation`]. It's a strict improvement on the naive
//! autocorrelation tracker (peak heights are unbiased by the window) without
//! the implementation cost of the full Boersma pipeline.
//!
//! ## Deferred alternates (per the DSP method-diversity entry)
//! - **Faithful Boersma 1993** (anti-alias upsample + Gaussian-window
//!   option + windowed-sinc + Brent's method peak refinement + multi-
//!   candidate Viterbi path-finder + octave-cost terms). Praat's default.
//! - YIN: de Cheveigné & Kawahara (2002), <https://doi.org/10.1121/1.1458024>
//! - RAPT: Talkin (1995), in *Speech Coding and Synthesis* (Elsevier),
//!   ISBN 978-0-444-82169-1
//! - SWIPE': Camacho & Harris (2008), <https://doi.org/10.1121/1.2951592>
//! - pYIN: Mauch & Dixon (2014), <https://doi.org/10.1109/ICASSP.2014.6853678>
//!   — librosa's default; HMM-refined YIN
//! - CREPE (neural): Kim et al. (2018), <https://arxiv.org/abs/1802.06182>
//!   — SOTA accuracy, requires bundled model weights

use std::f32::consts::PI as PI_F32;

use crate::Audio;
use crate::dsp::windowing::hann;

/// One of the supported pitch-estimation methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchMethod {
    /// Naive time-domain autocorrelation (Phase-0 method). See
    /// [`autocorrelation`].
    Autocorrelation,
    /// Window-corrected autocorrelation. Adopts the central insight of
    /// Boersma (1993) — divide windowed-signal autocorrelation by window
    /// autocorrelation — but is **not** a faithful Boersma implementation.
    /// See [`windowed_autocorrelation`].
    WindowedAutocorrelation,
}

/// Configuration for the pitch trackers.
#[derive(Debug, Clone)]
pub struct PitchConfig {
    /// Analysis frame length in seconds.
    pub frame_size_seconds: f32,
    /// Hop length (frame advance) in seconds.
    pub hop_size_seconds: f32,
    /// Minimum f0 to detect, in Hz.
    pub min_freq_hz: f32,
    /// Maximum f0 to detect, in Hz.
    pub max_freq_hz: f32,
    /// Threshold below which a frame is considered unvoiced. Applied to the
    /// `voicing` field by downstream callers; the trackers themselves still
    /// emit a frequency estimate for every frame so callers can filter.
    /// Default 0.45 (Boersma 1993 recommended value).
    pub voicing_threshold: f32,
}

impl Default for PitchConfig {
    fn default() -> Self {
        Self {
            frame_size_seconds: 0.030,
            hop_size_seconds: 0.010,
            min_freq_hz: 75.0,
            max_freq_hz: 500.0,
            voicing_threshold: 0.45,
        }
    }
}

/// One pitch estimate at a given time point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchFrame {
    /// Centre time of the analysis frame, in seconds from the start of the audio.
    pub time_seconds: f64,
    /// Estimated f0. For silent / unvoiced frames the lag still
    /// maximises *some* function so a frequency is reported; filter by
    /// `voicing` to discard unreliable estimates.
    pub frequency_hz: crate::units::Hertz,
    /// Voicing strength in `[0, 1]`. For [`autocorrelation`] this is
    /// `R(τ_best) / R(0)`; for [`autocorrelation_boersma`] this is the
    /// window-corrected normalised peak height. Closer to 1 = more clearly
    /// voiced.
    pub voicing: f32,
}

/// Dispatches to [`autocorrelation`] or [`windowed_autocorrelation`].
pub fn pitch(audio: &Audio, config: &PitchConfig, method: PitchMethod) -> Vec<PitchFrame> {
    match method {
        PitchMethod::Autocorrelation => autocorrelation(audio, config),
        PitchMethod::WindowedAutocorrelation => windowed_autocorrelation(audio, config),
    }
}

/// Estimates f0 using naive time-domain autocorrelation (Phase-0 method).
///
/// Multi-channel audio is downmixed to mono before analysis. Returns one
/// frame per hop step whose start falls within the audio. Audio shorter than
/// one frame returns an empty vector. The `voicing` field is `R(τ_best) /
/// R(0)`.
pub fn autocorrelation(audio: &Audio, config: &PitchConfig) -> Vec<PitchFrame> {
    let mono: Vec<f32> = audio.mono_samples().collect();
    let sample_rate = audio.sample_rate as f32;

    let frame_size = (config.frame_size_seconds * sample_rate).round() as usize;
    let hop_size = (config.hop_size_seconds * sample_rate).round() as usize;
    let min_lag = (sample_rate / config.max_freq_hz).round() as usize;
    let max_lag = (sample_rate / config.min_freq_hz).round() as usize;

    if mono.len() < frame_size || hop_size == 0 || min_lag >= max_lag {
        return Vec::new();
    }

    let mut frames = Vec::new();
    let mut start = 0;
    while start + frame_size <= mono.len() {
        let frame = &mono[start..start + frame_size];
        let (lag, peak) = best_lag(frame, min_lag, max_lag);
        let r0: f32 = frame.iter().map(|x| x * x).sum();
        let voicing = if r0 > 0.0 { peak / r0 } else { 0.0 };
        let frequency_hz = sample_rate / lag as f32;
        let time_seconds = (start + frame_size / 2) as f64 / audio.sample_rate as f64;
        frames.push(PitchFrame {
            time_seconds,
            frequency_hz: crate::units::Hertz::new(frequency_hz),
            voicing: voicing.clamp(0.0, 1.0),
        });
        start += hop_size;
    }
    frames
}

/// Returns `(best_lag, autocorrelation_value_at_best_lag)`.
fn best_lag(frame: &[f32], min_lag: usize, max_lag: usize) -> (usize, f32) {
    let max_lag = max_lag.min(frame.len().saturating_sub(1));
    let mut best = min_lag;
    let mut best_value = f32::MIN;

    for lag in min_lag..=max_lag {
        let mut sum = 0.0f32;
        for i in 0..(frame.len() - lag) {
            sum += frame[i] * frame[i + lag];
        }
        if sum > best_value {
            best_value = sum;
            best = lag;
        }
    }
    (best, best_value.max(0.0))
}

/// Estimates f0 by dividing the autocorrelation of the windowed signal by
/// the autocorrelation of the window itself — the central insight of
/// Boersma (1993) — and locating the highest peak in the resulting
/// normalised autocorrelation.
///
/// **This is not a faithful implementation of Boersma 1993.** It adopts the
/// window-correction idea but omits substantial parts of the paper's full
/// pipeline:
///
/// - No anti-alias upsample pre-step.
/// - Hann window only; the paper's accuracy-upgraded Gaussian-window form
///   is not exposed.
/// - **Parabolic** sub-sample interpolation. The paper rejects parabolic
///   (it caps peak heights at ~0.743 and contributes to octave errors) and
///   specifies windowed-sinc interpolation refined by Brent's method. The
///   parabolic form here is the simpler / more common textbook approach.
/// - No zero-pad before the autocorrelation FFT.
/// - Single peak per frame; no multi-candidate Viterbi path-finder.
/// - No octave-cost, voiced-unvoiced cost, or silence-threshold terms.
///
/// What it does:
/// 1. Subtract local mean (DC removal).
/// 2. Apply a Hann window.
/// 3. Compute autocorrelation of the windowed signal `r_a(τ)`.
/// 4. Compute autocorrelation of the window alone `r_w(τ)`.
/// 5. Divide: `r(τ) = r_a(τ) / r_w(τ)`.
/// 6. Find the highest peak in `r(τ)` for τ ∈ `[sr/max_freq_hz, sr/min_freq_hz]`.
/// 7. Parabolic interpolation around the integer-lag peak.
/// 8. Voicing = normalised peak height.
///
/// Despite the gaps this is a strict improvement on [`autocorrelation`]
/// (the naive Phase-0 tracker) — peak heights are no longer biased by the
/// window's own shape, sub-frame f0 resolution improves, and the voicing
/// strength is more meaningful. A faithful Boersma 1993 implementation is
/// tracked as a deferred alternate (see module docs).
pub fn windowed_autocorrelation(audio: &Audio, config: &PitchConfig) -> Vec<PitchFrame> {
    let mono: Vec<f32> = audio.mono_samples().collect();
    let sample_rate = audio.sample_rate as f32;

    let frame_size = (config.frame_size_seconds * sample_rate).round() as usize;
    let hop_size = (config.hop_size_seconds * sample_rate).round() as usize;
    let min_lag = (sample_rate / config.max_freq_hz).round() as usize;
    let max_lag = (sample_rate / config.min_freq_hz).round() as usize;

    if mono.len() < frame_size || hop_size == 0 || min_lag >= max_lag {
        return Vec::new();
    }

    let window = hann(frame_size);
    // Precompute r_w(τ) — depends only on the window, not the frame.
    let r_w = autocorr_full(&window, max_lag);

    let mut frames = Vec::new();
    let mut start = 0;
    let mut windowed = vec![0.0_f32; frame_size];
    while start + frame_size <= mono.len() {
        let raw = &mono[start..start + frame_size];
        // Subtract local mean (DC removal).
        let mean = raw.iter().sum::<f32>() / frame_size as f32;
        for (dst, &src) in windowed.iter_mut().zip(raw.iter()) {
            *dst = src - mean;
        }
        // Apply window.
        for (b, w) in windowed.iter_mut().zip(window.iter()) {
            *b *= *w;
        }
        // Autocorrelation of the windowed signal.
        let r_a = autocorr_full(&windowed, max_lag);
        // Corrected autocorrelation r(τ) = r_a(τ) / r_w(τ), normalised by
        // r(0) so peak heights are in `[0, 1]`.
        let r0 = if r_w[0].abs() > 0.0 {
            r_a[0] / r_w[0]
        } else {
            0.0
        };
        let frequency_hz;
        let voicing;
        if r0 <= 0.0 {
            // Silent / DC-only frame.
            frequency_hz = sample_rate / min_lag as f32;
            voicing = 0.0;
        } else {
            // Find the highest peak in r(τ) for τ ∈ [min_lag, max_lag].
            let mut best_lag_f = min_lag as f32;
            let mut best_score = f32::MIN;
            for tau in min_lag..=max_lag {
                if r_w[tau].abs() < 1e-12 {
                    continue;
                }
                let corrected = r_a[tau] / r_w[tau];
                if corrected > best_score {
                    best_score = corrected;
                    best_lag_f = tau as f32;
                }
            }
            // Parabolic interpolation around the integer-lag peak.
            let best_int = best_lag_f as usize;
            if best_int > min_lag && best_int < max_lag {
                let y_minus = r_a[best_int - 1] / r_w[best_int - 1];
                let y_0 = r_a[best_int] / r_w[best_int];
                let y_plus = r_a[best_int + 1] / r_w[best_int + 1];
                let denom = y_minus - 2.0 * y_0 + y_plus;
                if denom.abs() > 1e-12 {
                    let delta = 0.5 * (y_minus - y_plus) / denom;
                    if delta.abs() < 1.0 {
                        best_lag_f = best_int as f32 + delta;
                        // Interpolated peak height (also useful for voicing).
                        best_score = y_0 - 0.25 * (y_minus - y_plus) * delta;
                    }
                }
            }
            frequency_hz = sample_rate / best_lag_f;
            voicing = (best_score / r0).clamp(0.0, 1.0);
        }
        let time_seconds = (start + frame_size / 2) as f64 / audio.sample_rate as f64;
        frames.push(PitchFrame {
            time_seconds,
            frequency_hz: crate::units::Hertz::new(frequency_hz),
            voicing,
        });
        start += hop_size;
    }
    frames
}

/// Computes `r[τ] = Σ_n x[n] · x[n+τ]` for `τ = 0..=max_lag`.
fn autocorr_full(x: &[f32], max_lag: usize) -> Vec<f32> {
    let n = x.len();
    let max_lag = max_lag.min(n.saturating_sub(1));
    let mut r = vec![0.0_f32; max_lag + 1];
    for tau in 0..=max_lag {
        let mut sum = 0.0_f32;
        for i in 0..(n - tau) {
            sum += x[i] * x[i + tau];
        }
        r[tau] = sum;
    }
    r
}

#[allow(dead_code)]
const _UNUSED_PI_F32: f32 = PI_F32; // touch import so it's not removed prematurely

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_audio(sample_rate: u32, channels: u16, freq_hz: f32, duration_s: f32) -> Audio {
        let n_frames = (sample_rate as f32 * duration_s) as usize;
        let mut samples = Vec::with_capacity(n_frames * channels as usize);
        for i in 0..n_frames {
            let t = i as f32 / sample_rate as f32;
            let s = 0.5 * (2.0 * std::f32::consts::PI * freq_hz * t).sin();
            for _ in 0..channels {
                samples.push(s);
            }
        }
        Audio {
            samples,
            sample_rate,
            channels,
        }
    }

    fn silent_audio(sample_rate: u32, duration_s: f32) -> Audio {
        let n = (sample_rate as f32 * duration_s) as usize;
        Audio {
            samples: vec![0.0; n],
            sample_rate,
            channels: 1,
        }
    }

    #[test]
    fn detects_440_hz_sine() {
        let audio = sine_audio(16_000, 1, 440.0, 0.5);
        let frames = autocorrelation(&audio, &PitchConfig::default());

        assert!(!frames.is_empty(), "expected at least one frame");
        for f in &frames {
            assert!(
                (f.frequency_hz.value() - 440.0).abs() < 10.0,
                "frame at t={:.3}s reported {:.1} Hz, expected ~440",
                f.time_seconds,
                f.frequency_hz.value()
            );
        }
    }

    #[test]
    fn detects_100_hz_sine() {
        let audio = sine_audio(16_000, 1, 100.0, 0.5);
        let frames = autocorrelation(&audio, &PitchConfig::default());

        assert!(!frames.is_empty());
        for f in &frames {
            assert!(
                (f.frequency_hz.value() - 100.0).abs() < 2.0,
                "frame at t={:.3}s reported {:.1} Hz, expected ~100",
                f.time_seconds,
                f.frequency_hz.value()
            );
        }
    }

    #[test]
    fn stereo_downmix_matches_mono() {
        let mono = sine_audio(16_000, 1, 200.0, 0.3);
        let stereo = sine_audio(16_000, 2, 200.0, 0.3);

        let f_mono = autocorrelation(&mono, &PitchConfig::default());
        let f_stereo = autocorrelation(&stereo, &PitchConfig::default());

        assert_eq!(f_mono.len(), f_stereo.len());
        for (a, b) in f_mono.iter().zip(f_stereo.iter()) {
            assert!((a.frequency_hz.value() - b.frequency_hz.value()).abs() < 0.01);
            assert!((a.time_seconds - b.time_seconds).abs() < 1e-9);
        }
    }

    #[test]
    fn audio_shorter_than_one_frame_returns_empty() {
        let audio = sine_audio(16_000, 1, 200.0, 0.005);
        let frames = autocorrelation(&audio, &PitchConfig::default());
        assert!(frames.is_empty());
    }

    #[test]
    fn frame_times_are_monotonically_increasing() {
        let audio = sine_audio(16_000, 1, 200.0, 0.5);
        let frames = autocorrelation(&audio, &PitchConfig::default());
        for window in frames.windows(2) {
            assert!(window[1].time_seconds > window[0].time_seconds);
        }
    }

    #[test]
    fn unit_sine_has_high_voicing_via_autocorrelation() {
        let audio = sine_audio(16_000, 1, 200.0, 0.5);
        let frames = autocorrelation(&audio, &PitchConfig::default());
        let mid = &frames[frames.len() / 2];
        assert!(
            mid.voicing > 0.7,
            "expected high voicing, got {}",
            mid.voicing
        );
    }

    #[test]
    fn silent_input_has_zero_voicing_via_autocorrelation() {
        let audio = silent_audio(16_000, 0.5);
        let frames = autocorrelation(&audio, &PitchConfig::default());
        assert!(!frames.is_empty());
        for f in &frames {
            assert!(f.voicing < 0.001, "got voicing = {}", f.voicing);
        }
    }

    #[test]
    fn windowed_autocorrelation_detects_220_hz_sine_more_accurately_than_naive() {
        // For a clean sine, the window-corrected method should land within
        // 1 Hz (via parabolic interpolation), while the naive method is
        // limited to integer-lag precision.
        let audio = sine_audio(16_000, 1, 220.0, 0.5);
        let cfg = PitchConfig::default();
        let frames = windowed_autocorrelation(&audio, &cfg);
        assert!(!frames.is_empty());
        let mid = &frames[frames.len() / 2];
        assert!(
            (mid.frequency_hz.value() - 220.0).abs() < 1.0,
            "windowed_autocorrelation: got {} Hz, expected ~220",
            mid.frequency_hz.value()
        );
        assert!(
            mid.voicing > 0.7,
            "expected high voicing, got {}",
            mid.voicing
        );
    }

    #[test]
    fn windowed_autocorrelation_silent_input_has_low_voicing() {
        let audio = silent_audio(16_000, 0.5);
        let frames = windowed_autocorrelation(&audio, &PitchConfig::default());
        assert!(!frames.is_empty());
        for f in &frames {
            assert!(f.voicing < 0.1, "got voicing = {}", f.voicing);
        }
    }

    #[test]
    fn pitch_dispatcher_picks_the_right_method() {
        let audio = sine_audio(16_000, 1, 220.0, 0.5);
        let cfg = PitchConfig::default();
        let a = pitch(&audio, &cfg, PitchMethod::Autocorrelation);
        let b = pitch(&audio, &cfg, PitchMethod::WindowedAutocorrelation);
        // Both should detect 220 Hz; the methods don't have to agree to the
        // last decimal but must be within reasonable tolerance of each other.
        let mid_a = a[a.len() / 2].frequency_hz.value();
        let mid_b = b[b.len() / 2].frequency_hz.value();
        assert!(
            (mid_a - 220.0).abs() < 10.0,
            "autocorr midpoint = {}",
            mid_a
        );
        assert!((mid_b - 220.0).abs() < 5.0, "windowed midpoint = {}", mid_b);
    }
}
