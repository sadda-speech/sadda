//! Clinical perturbation measures: jitter + shimmer (Phase 3 B4).
//!
//! Period-to-period frequency perturbation (**jitter**) and amplitude
//! perturbation (**shimmer**) over a sustained phonation, in the standard
//! family Praat reports. Praat is the validation reference (see the
//! `crates/engine/tests/clinical` fixtures and the 2026-05-25
//! clinical-validation-references DEVLOG entry).
//!
//! Pipeline: estimate a nominal f0 (autocorrelation pitch) → detect one
//! glottal pulse per period by pitch-synchronous peak-picking → compute
//! the perturbation quotients over the realized period / peak-amplitude
//! sequences.
//!
//! Per the A2 no-silent-fallback discipline, too few detected periods
//! returns [`EngineError::Unreliable`] rather than a fabricated number.

use crate::Audio;
use crate::error::{EngineError, Result};
use crate::pitch::{self, PitchConfig};
use crate::units::{Decibels, Ratio};

/// Minimum periods required (the 5-point measures ppq5 / apq5 need a
/// 5-period window, so a handful more than that to be meaningful).
const MIN_PERIODS: usize = 6;

/// Configuration for [`perturbation`].
#[derive(Debug, Clone)]
pub struct PerturbationConfig {
    /// Lowest f0 to consider, in Hz.
    pub pitch_floor_hz: f32,
    /// Highest f0 to consider, in Hz.
    pub pitch_ceiling_hz: f32,
}

impl Default for PerturbationConfig {
    fn default() -> Self {
        Self {
            pitch_floor_hz: 75.0,
            pitch_ceiling_hz: 600.0,
        }
    }
}

/// Jitter + shimmer over a sustained phonation. Jitter and the relative
/// shimmers are dimensionless ratios (`0.01` = 1%); `shimmer_local_db`
/// is in decibels. Definitions follow Praat's Voice report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerturbationReport {
    /// Number of glottal periods the measures were computed over.
    pub n_periods: usize,
    /// Mean absolute difference of consecutive periods / mean period.
    pub jitter_local: Ratio,
    /// 3-point relative average perturbation of the period sequence.
    pub jitter_rap: Ratio,
    /// 5-point period perturbation quotient.
    pub jitter_ppq5: Ratio,
    /// Mean absolute difference of consecutive peak amplitudes / mean.
    pub shimmer_local: Ratio,
    /// Mean absolute consecutive peak-amplitude ratio, in dB.
    pub shimmer_local_db: Decibels,
    /// 3-point amplitude perturbation quotient.
    pub shimmer_apq3: Ratio,
    /// 5-point amplitude perturbation quotient.
    pub shimmer_apq5: Ratio,
}

/// Computes jitter + shimmer for a sustained phonation. Multi-channel
/// audio is downmixed to mono. Errors ([`EngineError::Unreliable`]) when
/// no voiced f0 is found or too few periods are detected.
pub fn perturbation(audio: &Audio, config: &PerturbationConfig) -> Result<PerturbationReport> {
    let sr = audio.sample_rate;
    let mono: Vec<f32> = audio.mono_samples().collect();

    let f0 = estimate_f0(audio, config)
        .ok_or_else(|| EngineError::unreliable("perturbation", "no voiced f0 detected"))?;

    let pulses = detect_pulses(&mono, sr, f0);
    if pulses.len() < MIN_PERIODS + 1 {
        return Err(EngineError::unreliable(
            "perturbation",
            format!(
                "only {} glottal pulses detected (need ≥ {})",
                pulses.len(),
                MIN_PERIODS + 1
            ),
        ));
    }

    let periods: Vec<f64> = pulses.windows(2).map(|w| w[1].0 - w[0].0).collect();
    let amps: Vec<f64> = pulses.iter().map(|p| p.1 as f64).collect();
    let mean_p = mean(&periods);
    let mean_a = mean(&amps);

    Ok(PerturbationReport {
        n_periods: periods.len(),
        jitter_local: Ratio::new((local_perturbation(&periods) / mean_p) as f32),
        jitter_rap: Ratio::new((ppq(&periods, 3) / mean_p) as f32),
        jitter_ppq5: Ratio::new((ppq(&periods, 5) / mean_p) as f32),
        shimmer_local: Ratio::new((local_perturbation(&amps) / mean_a) as f32),
        shimmer_local_db: Decibels::new(shimmer_db(&amps) as f32),
        shimmer_apq3: Ratio::new((ppq(&amps, 3) / mean_a) as f32),
        shimmer_apq5: Ratio::new((ppq(&amps, 5) / mean_a) as f32),
    })
}

/// Median voiced f0 over the signal (a sustained phonation has a stable
/// f0; the median is robust to a few unvoiced/edge frames).
fn estimate_f0(audio: &Audio, config: &PerturbationConfig) -> Option<f32> {
    let cfg = PitchConfig {
        min_freq_hz: config.pitch_floor_hz,
        max_freq_hz: config.pitch_ceiling_hz,
        ..Default::default()
    };
    let mut voiced: Vec<f32> = pitch::autocorrelation(audio, &cfg)
        .into_iter()
        .filter(|f| f.voicing >= 0.45)
        .map(|f| f.frequency_hz.value())
        .filter(|hz| *hz > 0.0)
        .collect();
    if voiced.is_empty() {
        return None;
    }
    voiced.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(voiced[voiced.len() / 2])
}

/// Pitch-synchronous peak-picking: one positive peak per ~period.
/// Returns `(time_seconds, peak_amplitude)` per pulse. A one-period
/// search window starting ~0.7 period past the previous peak tolerates
/// the period jitter we're trying to measure.
fn detect_pulses(samples: &[f32], sr: u32, f0: f32) -> Vec<(f64, f32)> {
    let n = samples.len();
    let win = (sr as f32 / f0).round() as usize;
    if win < 2 || n < win {
        return Vec::new();
    }
    let global_max = samples.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    if global_max <= 0.0 {
        return Vec::new();
    }
    let threshold = 0.2 * global_max;

    let mut pulses = Vec::new();
    let mut idx = 0usize;
    while idx + win <= n {
        let mut best = idx;
        let mut best_v = samples[idx];
        for (off, &v) in samples[idx..idx + win].iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = idx + off;
            }
        }
        if best_v > threshold {
            pulses.push((best as f64 / sr as f64, best_v));
            idx = best + (0.7 * win as f32) as usize;
        } else {
            idx += (win / 2).max(1);
        }
    }
    pulses
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Mean absolute difference of consecutive values (the "local" measure,
/// before normalization by the mean).
fn local_perturbation(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let s: f64 = xs.windows(2).map(|w| (w[1] - w[0]).abs()).sum();
    s / (xs.len() - 1) as f64
}

/// `m`-point perturbation quotient (m odd; m=3 → rap/apq3, m=5 →
/// ppq5/apq5): mean over centered windows of `|x_i − mean(window)|`,
/// before normalization by the mean.
fn ppq(xs: &[f64], m: usize) -> f64 {
    let h = m / 2;
    if xs.len() <= 2 * h {
        return 0.0;
    }
    let mut acc = 0.0;
    let mut count = 0usize;
    for i in h..xs.len() - h {
        let window_mean = xs[i - h..=i + h].iter().sum::<f64>() / m as f64;
        acc += (xs[i] - window_mean).abs();
        count += 1;
    }
    if count == 0 { 0.0 } else { acc / count as f64 }
}

/// Mean absolute consecutive amplitude ratio in dB (`shimmer local dB`).
fn shimmer_db(amps: &[f64]) -> f64 {
    if amps.len() < 2 {
        return 0.0;
    }
    let s: f64 = amps
        .windows(2)
        .filter(|w| w[0] > 0.0 && w[1] > 0.0)
        .map(|w| (20.0 * (w[1] / w[0]).log10()).abs())
        .sum();
    s / (amps.len() - 1) as f64
}

/// Configuration for [`hnr`].
#[derive(Debug, Clone)]
pub struct HnrConfig {
    /// Lowest f0 to consider, in Hz (sets the longest lag / window).
    pub pitch_floor_hz: f32,
    /// Highest f0 to consider, in Hz (sets the shortest lag).
    pub pitch_ceiling_hz: f32,
    /// Frame advance, in seconds.
    pub hop_seconds: f32,
}

impl Default for HnrConfig {
    fn default() -> Self {
        Self {
            pitch_floor_hz: 75.0,
            pitch_ceiling_hz: 600.0,
            hop_seconds: 0.01,
        }
    }
}

/// Mean harmonics-to-noise ratio (dB) of a sustained phonation, via the
/// Boersma-1993 **cross-correlation** method (Praat's `To Harmonicity
/// (cc)`).
///
/// Per frame, the maximum normalized cross-correlation `r` over the
/// pitch lag range gives `HNR = 10·log10(r / (1 − r))`; the mean is
/// taken over non-silent frames. The geometric-mean energy
/// normalization `r(τ) = Σ x_i x_{i+τ} / √(Σ x_i² · Σ x_{i+τ}²)` is what
/// makes this track Praat near `r → 1`, where the pitch tracker's
/// window-corrected voicing over-reads badly. [`EngineError::Unreliable`]
/// if the signal is too short or wholly silent.
pub fn hnr(audio: &Audio, config: &HnrConfig) -> Result<Decibels> {
    let sr = audio.sample_rate as f32;
    let x: Vec<f64> = audio.mono_samples().map(|s| s as f64).collect();
    let n = x.len();
    let min_lag = (sr / config.pitch_ceiling_hz).round() as usize;
    let max_lag = (sr / config.pitch_floor_hz).round() as usize;
    let win = max_lag.max(1); // comparison window ≥ one floor-period
    let hop = ((config.hop_seconds * sr) as usize).max(1);
    if min_lag < 1 || max_lag <= min_lag || n < win + max_lag {
        return Err(EngineError::unreliable(
            "hnr",
            "signal too short for the requested pitch range",
        ));
    }

    let frame_energy = |s: usize| -> f64 { x[s..s + win].iter().map(|v| v * v).sum() };

    // Silence gate at 1% of the strongest frame's energy.
    let mut max_e = 0.0_f64;
    let mut s = 0;
    while s + win + max_lag <= n {
        max_e = max_e.max(frame_energy(s));
        s += hop;
    }
    let silence = 0.01 * max_e;

    let mut sum = 0.0_f64;
    let mut count = 0usize;
    s = 0;
    while s + win + max_lag <= n {
        let e0 = frame_energy(s);
        if e0 <= silence {
            s += hop;
            continue;
        }
        let mut best_r = 0.0_f64;
        for tau in min_lag..=max_lag {
            let mut cc = 0.0_f64;
            let mut e1 = 0.0_f64;
            for i in 0..win {
                let b = x[s + i + tau];
                cc += x[s + i] * b;
                e1 += b * b;
            }
            if e1 > 0.0 {
                let r = cc / (e0 * e1).sqrt();
                if r > best_r {
                    best_r = r;
                }
            }
        }
        let r = best_r.clamp(1e-6, 1.0 - 1e-6);
        sum += 10.0 * (r / (1.0 - r)).log10();
        count += 1;
        s += hop;
    }
    if count == 0 {
        return Err(EngineError::unreliable("hnr", "no non-silent frames"));
    }
    Ok(Decibels::new((sum / count as f64) as f32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppq_window_math() {
        // Constant sequence → zero perturbation.
        assert_eq!(ppq(&[1.0; 10], 3), 0.0);
        assert_eq!(local_perturbation(&[2.0; 5]), 0.0);
    }

    #[test]
    fn silence_is_unreliable_not_a_guess() {
        // No voiced f0 → an explicit error, never a fabricated number.
        let audio = Audio {
            samples: vec![0.0; 16_000],
            sample_rate: 16_000,
            channels: 1,
        };
        let err = perturbation(&audio, &PerturbationConfig::default()).unwrap_err();
        assert!(matches!(err, EngineError::Unreliable { .. }));
    }
}
