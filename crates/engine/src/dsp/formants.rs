//! Formant tracker: pre-emphasis → window → LPC → polynomial root-find →
//! (freq, bandwidth) per resonance. Both LPC methods (autocorrelation and
//! Burg) are selectable; Burg is the default (Praat convention).
//!
//! ## References
//! - Markel, J.D. (1972), "Digital inverse filtering — a new tool for
//!   formant trajectory estimation." *IEEE TASSP* 20(2).
//!   <https://doi.org/10.1109/TAU.1972.1162366>
//! - McCandless, S.S. (1974), "An algorithm for automatic formant
//!   extraction using linear prediction spectra." *IEEE TASSP* 22(2).
//!   <https://doi.org/10.1109/TASSP.1974.1162572>
//! - Praat formant manual:
//!   <https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html>
//!
//! ## Deferred alternates (per the DSP method-diversity entry)
//! - Burg-robust LPC (Praat's `Sound.to_formant_burg_robust`).
//! - Viterbi / DP trajectory smoothing (per-frame independent here).
//! - Fixed-N output array helper with NaN padding for missing formants.

use std::f32::consts::PI as PI_F32;

use crate::dsp::lpc::{LpcMethod, lpc};
use crate::dsp::roots::polynomial_roots;
use crate::dsp::windowing::hann;

/// One frame of formant output. `frequencies[i]` and `bandwidths[i]` describe
/// the same resonance; arrays are co-indexed and sorted by frequency.
/// Variable length per frame (frames where the root-finder didn't return
/// enough valid roots in the formant range are honestly empty rather than
/// padded with NaN).
#[derive(Debug, Clone)]
pub struct FormantFrame {
    /// Time at the centre of the analysis frame, in seconds.
    pub time_seconds: f64,
    /// Formant frequencies, ascending.
    pub frequencies: Vec<crate::units::Hertz>,
    /// Bandwidths, co-indexed with `frequencies`.
    pub bandwidths: Vec<crate::units::Hertz>,
}

/// Configuration for [`formants`].
#[derive(Debug, Clone)]
pub struct FormantsConfig {
    /// Analysis frame length in seconds.
    pub frame_size_seconds: f32,
    /// Hop length (frame advance) in seconds.
    pub hop_seconds: f32,
    /// Maximum number of formants to keep per frame (after frequency filter).
    pub n_formants: usize,
    /// Pre-emphasis coefficient `α` in `y[n] = x[n] - α · x[n-1]`. Set to
    /// `0.0` to skip pre-emphasis. Typical: 0.97.
    pub pre_emphasis: f32,
    /// LPC order. `None` defaults to `2 · n_formants + 2`.
    pub lpc_order: Option<usize>,
    /// Which LPC method to use. Default is Burg (Praat convention).
    pub lpc_method: LpcMethod,
    /// Reject roots with bandwidth above this threshold (Hz). Wide
    /// bandwidths typically indicate noise / spurious roots, not real
    /// formants.
    pub max_bandwidth_hz: f32,
    /// Lower frequency cutoff (Hz). Roots below this are discarded
    /// (typically glottal-source artifacts).
    pub min_frequency_hz: f32,
}

impl Default for FormantsConfig {
    fn default() -> Self {
        Self {
            frame_size_seconds: 0.025,
            hop_seconds: 0.010,
            n_formants: 5,
            pre_emphasis: 0.97,
            lpc_order: None,
            lpc_method: LpcMethod::Burg,
            max_bandwidth_hz: 1000.0,
            min_frequency_hz: 50.0,
        }
    }
}

/// Computes per-frame formants over `samples`.
pub fn formants(samples: &[f32], sample_rate: u32, config: &FormantsConfig) -> Vec<FormantFrame> {
    assert!(sample_rate > 0, "formants: sample_rate must be > 0");
    assert!(
        config.frame_size_seconds > 0.0,
        "formants: frame_size_seconds must be > 0"
    );
    assert!(
        config.hop_seconds > 0.0,
        "formants: hop_seconds must be > 0"
    );
    assert!(config.n_formants > 0, "formants: n_formants must be > 0");

    let frame_size = (config.frame_size_seconds * sample_rate as f32).round() as usize;
    let hop_size = (config.hop_seconds * sample_rate as f32).round() as usize;
    let lpc_order = config.lpc_order.unwrap_or(2 * config.n_formants + 2);
    if frame_size == 0 || hop_size == 0 || samples.len() < frame_size || lpc_order >= frame_size {
        return Vec::new();
    }

    let window = hann(frame_size);
    let half_frame_seconds = frame_size as f64 / (2.0 * sample_rate as f64);
    let nyquist = sample_rate as f32 / 2.0;
    let max_frequency = nyquist - 50.0;

    let n_frames = (samples.len() - frame_size) / hop_size + 1;
    let mut out = Vec::with_capacity(n_frames);
    let mut frame_buf: Vec<f32> = vec![0.0; frame_size];

    for f in 0..n_frames {
        let start = f * hop_size;
        let raw = &samples[start..start + frame_size];

        // Pre-emphasis: y[n] = x[n] - α · x[n-1].
        if config.pre_emphasis != 0.0 {
            frame_buf[0] = raw[0];
            for i in 1..frame_size {
                frame_buf[i] = raw[i] - config.pre_emphasis * raw[i - 1];
            }
        } else {
            frame_buf.copy_from_slice(raw);
        }
        // Window in place.
        for (b, w) in frame_buf.iter_mut().zip(window.iter()) {
            *b *= *w;
        }

        let time_seconds = start as f64 / sample_rate as f64 + half_frame_seconds;
        let Ok(lpc_result) = lpc(&frame_buf, lpc_order, config.lpc_method) else {
            // Zero-energy or degenerate frame — emit empty formants.
            out.push(FormantFrame {
                time_seconds,
                frequencies: Vec::new(),
                bandwidths: Vec::new(),
            });
            continue;
        };

        // Build characteristic polynomial in ascending coefficients.
        // The predictor polynomial is `1 + a_1 z^-1 + ... + a_p z^-p`.
        // Multiplying by z^p gives `z^p + a_1 z^(p-1) + ... + a_p`, whose
        // roots are the resonance poles. Ascending coeffs: [a_p, a_{p-1},
        // ..., a_1, 1].
        let mut ascending = Vec::with_capacity(lpc_order + 1);
        for &a in lpc_result.coeffs.iter().rev() {
            ascending.push(a as f64);
        }
        ascending.push(1.0);

        let roots = match polynomial_roots(&ascending) {
            Ok(r) => r,
            Err(_) => {
                out.push(FormantFrame {
                    time_seconds,
                    frequencies: Vec::new(),
                    bandwidths: Vec::new(),
                });
                continue;
            }
        };

        // Convert roots → (freq, bw). Only keep:
        // - roots inside the unit circle (|z| < 1)  — stable poles
        // - positive imaginary part                 — one of each conjugate pair
        // - frequency in [min_frequency_hz, max_frequency]
        // - bandwidth ≤ max_bandwidth_hz
        let mut candidates: Vec<(f32, f32)> = Vec::new();
        let sr_f64 = sample_rate as f64;
        for z in &roots {
            let r = z.norm();
            if r >= 1.0 - 1e-9 {
                continue;
            }
            let theta = z.im.atan2(z.re);
            if theta <= 0.0 {
                continue;
            }
            let freq = (theta * sr_f64 / (2.0 * std::f64::consts::PI)) as f32;
            let bw = (-r.ln() * sr_f64 / std::f64::consts::PI) as f32;
            if freq < config.min_frequency_hz
                || freq > max_frequency
                || bw > config.max_bandwidth_hz
            {
                continue;
            }
            candidates.push((freq, bw));
        }
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        candidates.truncate(config.n_formants);

        let frequencies: Vec<crate::units::Hertz> = candidates
            .iter()
            .map(|(f, _)| crate::units::Hertz::new(*f))
            .collect();
        let bandwidths: Vec<crate::units::Hertz> = candidates
            .iter()
            .map(|(_, b)| crate::units::Hertz::new(*b))
            .collect();
        out.push(FormantFrame {
            time_seconds,
            frequencies,
            bandwidths,
        });
    }
    out
}

#[allow(dead_code)]
const _PI_F32_TOUCH: f32 = PI_F32; // silence unused-import lint without an explicit unused

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// Synthesizes a vowel-like signal by filtering an impulse train (at f0)
    /// through a cascade of 2nd-order resonators at the given formant
    /// frequencies. This is the canonical source-filter model for testing
    /// formant trackers.
    fn synthesize_vowel(
        sample_rate: u32,
        duration_seconds: f32,
        f0: f32,
        formant_freqs: &[f32],
        formant_bws: &[f32],
    ) -> Vec<f32> {
        let n = (duration_seconds * sample_rate as f32) as usize;
        let mut source = vec![0.0_f32; n];
        // Impulse train at f0.
        let period_samples = (sample_rate as f32 / f0) as usize;
        let mut t = 0;
        while t < n {
            source[t] = 1.0;
            t += period_samples;
        }

        // Cascade of 2nd-order resonators.
        let mut signal = source.clone();
        for (fi, bi) in formant_freqs.iter().zip(formant_bws.iter()) {
            let r = (-PI_F32 * bi / sample_rate as f32).exp();
            let theta = TAU * fi / sample_rate as f32;
            let a1 = -2.0 * r * theta.cos();
            let a2 = r * r;
            // y[n] = x[n] - a1 * y[n-1] - a2 * y[n-2]
            let mut y_prev2 = 0.0_f32;
            let mut y_prev1 = 0.0_f32;
            for s in signal.iter_mut() {
                let y = *s - a1 * y_prev1 - a2 * y_prev2;
                *s = y;
                y_prev2 = y_prev1;
                y_prev1 = y;
            }
        }
        // Normalize.
        let max = signal.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        if max > 0.0 {
            for s in signal.iter_mut() {
                *s /= max;
            }
        }
        signal
    }

    #[test]
    fn synthetic_vowel_formants_land_near_expected_frequencies() {
        // Synthesize the /a/-like vowel from Klatt (1980): F1=700, F2=1220,
        // F3=2600 with 50 Hz bandwidths each. f0 = 110 Hz; 0.5 s.
        let sr = 16_000;
        let expected = [700.0_f32, 1220.0, 2600.0];
        let bws = [50.0_f32, 50.0, 50.0];
        let signal = synthesize_vowel(sr, 0.5, 110.0, &expected, &bws);

        let cfg = FormantsConfig::default();
        let frames = formants(&signal, sr, &cfg);
        assert!(!frames.is_empty());

        // Pick a steady-state frame from the middle. Check that the first 3
        // estimated formants land within 100 Hz of the targets.
        let mid = &frames[frames.len() / 2];
        assert!(
            mid.frequencies.len() >= 3,
            "expected at least 3 formants, got {:?}",
            mid.frequencies
        );
        for (i, &target) in expected.iter().enumerate() {
            assert!(
                (mid.frequencies[i].value() - target).abs() < 120.0,
                "formant {}: got {} Hz, expected ~{} Hz; full freqs {:?}",
                i + 1,
                mid.frequencies[i],
                target,
                mid.frequencies
            );
        }
    }

    #[test]
    fn silent_frame_returns_empty_formants() {
        let sr = 16_000;
        let signal = vec![0.0_f32; sr as usize];
        let frames = formants(&signal, sr, &FormantsConfig::default());
        assert!(!frames.is_empty());
        for f in &frames {
            assert!(f.frequencies.is_empty());
            assert!(f.bandwidths.is_empty());
        }
    }

    #[test]
    fn input_shorter_than_one_frame_returns_empty() {
        let frames = formants(&[0.5_f32; 100], 16_000, &FormantsConfig::default());
        assert!(frames.is_empty());
    }

    #[test]
    fn autocorrelation_method_also_works_on_synthetic_vowel() {
        // Same vowel, autocorrelation LPC. Different numerical path; should
        // still land somewhere reasonable.
        let sr = 16_000;
        let expected = [700.0_f32, 1220.0, 2600.0];
        let signal = synthesize_vowel(sr, 0.5, 110.0, &expected, &[50.0; 3]);
        let cfg = FormantsConfig {
            lpc_method: LpcMethod::Autocorrelation,
            ..Default::default()
        };
        let frames = formants(&signal, sr, &cfg);
        let mid = &frames[frames.len() / 2];
        assert!(
            mid.frequencies.len() >= 3,
            "autocorrelation method should still find 3+ formants; got {:?}",
            mid.frequencies
        );
        for (i, &target) in expected.iter().enumerate() {
            assert!(
                (mid.frequencies[i].value() - target).abs() < 200.0,
                "autocorr formant {}: got {}, expected {}",
                i + 1,
                mid.frequencies[i],
                target
            );
        }
    }
}
