//! Fundamental-frequency (f0) estimation via time-domain autocorrelation.
//! Phase 0 implementation; sub-sample lag interpolation and a voicing
//! decision are deferred.

use crate::Audio;

/// Configuration for the autocorrelation pitch tracker.
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
}

impl Default for PitchConfig {
    fn default() -> Self {
        Self {
            frame_size_seconds: 0.030,
            hop_size_seconds: 0.010,
            min_freq_hz: 75.0,
            max_freq_hz: 500.0,
        }
    }
}

/// One pitch estimate at a given time point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchFrame {
    /// Centre time of the analysis frame, in seconds from the start of the audio.
    pub time_seconds: f64,
    /// Estimated f0 in Hz. No voicing decision is applied yet — silent frames
    /// still get whichever lag happened to maximise the autocorrelation.
    pub frequency_hz: f32,
}

/// Estimates f0 using time-domain autocorrelation.
///
/// Multi-channel audio is downmixed to mono before analysis. Returns one frame
/// per hop step whose centre time falls within the audio. Audio shorter than
/// one frame returns an empty vector. A voicing decision is *not* applied at
/// this stage — silent or unvoiced frames will still report whichever lag
/// happens to maximise the (zero-valued) autocorrelation.
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
        let lag = best_lag(frame, min_lag, max_lag);
        let frequency_hz = sample_rate / lag as f32;
        let time_seconds = (start + frame_size / 2) as f64 / audio.sample_rate as f64;
        frames.push(PitchFrame {
            time_seconds,
            frequency_hz,
        });
        start += hop_size;
    }
    frames
}

fn best_lag(frame: &[f32], min_lag: usize, max_lag: usize) -> usize {
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
    best
}

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

    #[test]
    fn detects_440_hz_sine() {
        let audio = sine_audio(16_000, 1, 440.0, 0.5);
        let frames = autocorrelation(&audio, &PitchConfig::default());

        assert!(!frames.is_empty(), "expected at least one frame");
        for f in &frames {
            assert!(
                (f.frequency_hz - 440.0).abs() < 10.0,
                "frame at t={:.3}s reported {:.1} Hz, expected ~440",
                f.time_seconds,
                f.frequency_hz
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
                (f.frequency_hz - 100.0).abs() < 2.0,
                "frame at t={:.3}s reported {:.1} Hz, expected ~100",
                f.time_seconds,
                f.frequency_hz
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
            assert!((a.frequency_hz - b.frequency_hz).abs() < 0.01);
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
}
