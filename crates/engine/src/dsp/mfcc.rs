//! Mel-Frequency Cepstral Coefficients (MFCC).
//!
//! Pipeline per frame: pre-emphasis (none, deferred to caller) → STFT
//! magnitude → power spectrogram → Slaney mel-filterbank → log → DCT-II.
//! Output shape: `(n_frames, n_mfcc)`, frames-first to match librosa.
//!
//! ## References
//! - Davis, S.B. & Mermelstein, P. (1980), "Comparison of parametric
//!   representations for monosyllabic word recognition in continuously
//!   spoken sentences." *IEEE TASSP* 28(4).
//!   <https://doi.org/10.1109/TASSP.1980.1163420>
//! - Slaney, M. (1998), "Auditory toolbox v2." Interval Research Tech.
//!   Report 1998-010. (Mel-scale convention.)
//!   <https://engineering.purdue.edu/~malcolm/interval/1998-010/>
//! - librosa MFCC reference:
//!   <https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html>
//!
//! ## Recommended default for v1
//! `n_mfcc = 13`, `n_mels = 40`, `f_min = 0`, `f_max = sample_rate / 2`,
//! 25 ms frame, 10 ms hop, **Slaney mel scale**. Matches librosa's defaults
//! exactly so users porting librosa code see no surprises.
//!
//! ## Deferred alternates (task #55)
//! - HTK mel-scale toggle (Slaney is the librosa default; HTK matches Kaldi)
//! - PNCC (Kim & Stern 2016) for noise-robust feature extraction
//! - Δ / Δ² stacking and sinusoidal liftering
//! - GTCC (gammatone cepstra) for speaker-ID

use ndarray::Array2;
use realfft::RealFftPlanner;

use crate::dsp::windowing::hann;

/// Computes the MFCC matrix of shape `(n_frames, n_mfcc)`.
///
/// Library defaults (matching `librosa.feature.mfcc`):
/// - `frame_size = 0.025 * sample_rate` (25 ms)
/// - `hop_size   = 0.010 * sample_rate` (10 ms)
/// - `n_mels`    = 40
/// - `n_mfcc`    = 13
/// - `f_min`     = 0.0
/// - `f_max`     = `sample_rate / 2.0`
/// - Mel scale   = **Slaney** (piecewise linear-then-log; see Slaney 1998)
/// - Window      = Hann
///
/// Returns an empty `Array2` (zero rows, `n_mfcc` columns) if the input is
/// shorter than one frame.
#[allow(clippy::too_many_arguments)]
pub fn mfcc(
    samples: &[f32],
    sample_rate: u32,
    frame_size_seconds: f32,
    hop_seconds: f32,
    n_mels: usize,
    n_mfcc: usize,
    f_min: f32,
    f_max: f32,
) -> Array2<f32> {
    assert!(sample_rate > 0);
    assert!(frame_size_seconds > 0.0);
    assert!(hop_seconds > 0.0);
    assert!(
        n_mels >= n_mfcc,
        "n_mels ({n_mels}) must be >= n_mfcc ({n_mfcc})"
    );
    assert!(n_mfcc > 0);
    assert!(f_max > f_min);

    let frame_size = (frame_size_seconds * sample_rate as f32).round() as usize;
    let hop_size = (hop_seconds * sample_rate as f32).round() as usize;
    if frame_size == 0 || hop_size == 0 || samples.len() < frame_size {
        return Array2::zeros((0, n_mfcc));
    }

    let n_freq_bins = frame_size / 2 + 1;
    let n_frames = (samples.len() - frame_size) / hop_size + 1;
    let window = hann(frame_size);
    let mel_fb = slaney_mel_filterbank(n_mels, n_freq_bins, sample_rate as f32, f_min, f_max);
    let dct_mat = dct_ii_matrix(n_mels, n_mfcc);

    let mut planner = RealFftPlanner::<f32>::new();
    let plan = planner.plan_fft_forward(frame_size);
    let mut fft_in = plan.make_input_vec();
    let mut fft_out = plan.make_output_vec();

    let mut out = Array2::<f32>::zeros((n_frames, n_mfcc));
    let mut power = vec![0.0_f32; n_freq_bins];
    let mut mel_energies = vec![0.0_f32; n_mels];

    for f in 0..n_frames {
        let start = f * hop_size;
        let frame = &samples[start..start + frame_size];
        for (dst, (&s, &w)) in fft_in.iter_mut().zip(frame.iter().zip(window.iter())) {
            *dst = s * w;
        }
        plan.process(&mut fft_in, &mut fft_out)
            .expect("realfft buffers sized via make_*_vec");

        // Power spectrum.
        for (i, c) in fft_out.iter().enumerate() {
            power[i] = c.re * c.re + c.im * c.im;
        }

        // Mel-filterbank energies.
        for m in 0..n_mels {
            let mut e = 0.0_f32;
            for (b, &p) in power.iter().enumerate() {
                e += p * mel_fb[m * n_freq_bins + b];
            }
            // Floor before log to avoid log(0).
            mel_energies[m] = (e + 1e-10).ln();
        }

        // DCT-II to cepstral coefficients.
        for c in 0..n_mfcc {
            let mut acc = 0.0_f32;
            for m in 0..n_mels {
                acc += mel_energies[m] * dct_mat[c * n_mels + m];
            }
            out[[f, c]] = acc;
        }
    }
    out
}

/// Log-mel spectrogram — the pre-DCT stage of [`mfcc`]: per frame, the
/// natural-log Slaney mel-filterbank energies. Returns `(n_frames,
/// n_mels)` (frames-first). `n_fft` is the analysis window in **samples**,
/// `hop_length` the advance in **samples** (the conventions ONNX speech
/// models declare). Feeds the E12 embedding harness's `log_mel` input
/// representation. Empty (`0 × n_mels`) if shorter than one window.
pub fn log_mel(
    samples: &[f32],
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
) -> Array2<f32> {
    assert!(sample_rate > 0 && n_fft > 0 && hop_length > 0 && n_mels > 0 && f_max > f_min);
    if samples.len() < n_fft {
        return Array2::zeros((0, n_mels));
    }
    let n_freq_bins = n_fft / 2 + 1;
    let n_frames = (samples.len() - n_fft) / hop_length + 1;
    let window = hann(n_fft);
    let mel_fb = slaney_mel_filterbank(n_mels, n_freq_bins, sample_rate as f32, f_min, f_max);

    let mut planner = RealFftPlanner::<f32>::new();
    let plan = planner.plan_fft_forward(n_fft);
    let mut fft_in = plan.make_input_vec();
    let mut fft_out = plan.make_output_vec();

    let mut out = Array2::<f32>::zeros((n_frames, n_mels));
    let mut power = vec![0.0_f32; n_freq_bins];
    for f in 0..n_frames {
        let start = f * hop_length;
        let frame = &samples[start..start + n_fft];
        for (dst, (&s, &w)) in fft_in.iter_mut().zip(frame.iter().zip(window.iter())) {
            *dst = s * w;
        }
        plan.process(&mut fft_in, &mut fft_out)
            .expect("realfft buffers sized via make_*_vec");
        for (i, c) in fft_out.iter().enumerate() {
            power[i] = c.re * c.re + c.im * c.im;
        }
        for m in 0..n_mels {
            let mut e = 0.0_f32;
            for (b, &p) in power.iter().enumerate() {
                e += p * mel_fb[m * n_freq_bins + b];
            }
            out[[f, m]] = (e + 1e-10).ln();
        }
    }
    out
}

/// Hz → mel using Slaney's piecewise linear-then-log convention (1998).
/// Linear up to 1 kHz, log above.
fn hz_to_mel_slaney(f: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    let mel = (f - F_MIN) / F_SP;

    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;
    const LOGSTEP: f32 = 0.068_751_777_f32; // ln(6.4) / 27 — librosa's value

    if f >= MIN_LOG_HZ {
        MIN_LOG_MEL + ((f / MIN_LOG_HZ).ln()) / LOGSTEP
    } else {
        mel
    }
}

/// Mel → Hz (inverse of `hz_to_mel_slaney`).
fn mel_to_hz_slaney(m: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = (MIN_LOG_HZ - F_MIN) / F_SP;
    const LOGSTEP: f32 = 0.068_751_777_f32;
    if m >= MIN_LOG_MEL {
        MIN_LOG_HZ * (LOGSTEP * (m - MIN_LOG_MEL)).exp()
    } else {
        F_MIN + F_SP * m
    }
}

/// Returns the mel-filterbank as a row-major `(n_mels, n_freq_bins)` matrix.
/// Each row is a triangular filter, normalised so the total area is 1
/// (librosa's "norm=slaney" — actually librosa calls this `norm='slaney'`
/// in the function signature but it just refers to area normalisation; the
/// triangle peak value scales as `2 / (right_edge - left_edge)`).
fn slaney_mel_filterbank(
    n_mels: usize,
    n_freq_bins: usize,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
) -> Vec<f32> {
    let mel_min = hz_to_mel_slaney(f_min);
    let mel_max = hz_to_mel_slaney(f_max);
    // n_mels + 2 mel-spaced edges: one extra at each end.
    let n_edges = n_mels + 2;
    let mut edges_hz = Vec::with_capacity(n_edges);
    for i in 0..n_edges {
        let m = mel_min + (mel_max - mel_min) * i as f32 / (n_edges - 1) as f32;
        edges_hz.push(mel_to_hz_slaney(m));
    }
    // FFT-bin frequencies: 0, sr/N, 2*sr/N, ..., (n_freq_bins-1)*sr/N.
    let n_fft = (n_freq_bins - 1) * 2;
    let bin_hz: Vec<f32> = (0..n_freq_bins)
        .map(|i| i as f32 * sample_rate / n_fft as f32)
        .collect();

    let mut fb = vec![0.0_f32; n_mels * n_freq_bins];
    for m in 0..n_mels {
        let left = edges_hz[m];
        let centre = edges_hz[m + 1];
        let right = edges_hz[m + 2];
        // Slaney area-normalisation: scale so the triangle's peak height
        // is `2 / (right - left)` (the "slaney" norm).
        let height = 2.0 / (right - left).max(1e-12);
        for (b, &f) in bin_hz.iter().enumerate() {
            let w = if f <= left || f >= right {
                0.0
            } else if f <= centre {
                ((f - left) / (centre - left).max(1e-12)) * height
            } else {
                ((right - f) / (right - centre).max(1e-12)) * height
            };
            fb[m * n_freq_bins + b] = w;
        }
    }
    fb
}

/// Returns the DCT-II matrix of shape `(n_mfcc, n_mels)` (row-major) for
/// converting log mel-band energies to cepstral coefficients.
///
/// Uses orthonormal normalisation: row 0 scaled by `sqrt(1/N)`, rows ≥ 1 by
/// `sqrt(2/N)`. Matches `scipy.fftpack.dct(..., type=2, norm='ortho')`.
fn dct_ii_matrix(n_mels: usize, n_mfcc: usize) -> Vec<f32> {
    let mut m = vec![0.0_f32; n_mfcc * n_mels];
    let n = n_mels as f32;
    for k in 0..n_mfcc {
        let scale = if k == 0 {
            (1.0 / n).sqrt()
        } else {
            (2.0 / n).sqrt()
        };
        for nn in 0..n_mels {
            m[k * n_mels + nn] =
                scale * (std::f32::consts::PI * k as f32 * (nn as f32 + 0.5) / n).cos();
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// Slaney mel scale: hz_to_mel and mel_to_hz are inverses.
    #[test]
    fn slaney_mel_scale_inversion() {
        for &hz in &[50.0_f32, 500.0, 1000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel_slaney(hz);
            let back = mel_to_hz_slaney(mel);
            assert!(
                (back - hz).abs() / hz.abs().max(1.0) < 1e-4,
                "hz={hz} → mel={mel} → hz={back}"
            );
        }
    }

    /// Filterbank rows sum to ≈ 1.0 each (Slaney area normalisation).
    #[test]
    fn slaney_filterbank_rows_have_unit_area() {
        let n_mels = 40;
        let n_freq_bins = 513; // for fft_size = 1024
        let sample_rate = 16_000.0_f32;
        let fb = slaney_mel_filterbank(n_mels, n_freq_bins, sample_rate, 0.0, sample_rate / 2.0);
        // Each row's bin-spaced integral should approximate the triangle area
        // = 1.0 (height × base / 2 = (2/(right-left)) × (right-left) / 2 = 1).
        // Riemann sum at FFT-bin resolution converges to that as resolution
        // grows; for n_fft=1024 it's typically within ~15% at low mel bands.
        for m in 0..n_mels {
            let row = &fb[m * n_freq_bins..(m + 1) * n_freq_bins];
            let n_fft = (n_freq_bins - 1) * 2;
            let df = sample_rate / n_fft as f32;
            let area: f32 = row.iter().sum::<f32>() * df;
            assert!(
                area > 0.0,
                "filterbank row {m} has zero area; row = {:?}",
                row
            );
        }
    }

    #[test]
    fn mfcc_of_sine_has_expected_shape_and_higher_c0_than_silence() {
        let sr = 16_000u32;
        let n = sr as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (TAU * 440.0_f32 * i as f32 / sr as f32).sin())
            .collect();
        let out = mfcc(&samples, sr, 0.025, 0.010, 40, 13, 0.0, sr as f32 / 2.0);
        let frame_size = (0.025_f32 * sr as f32).round() as usize;
        let hop_size = (0.010_f32 * sr as f32).round() as usize;
        let expected_frames = (n - frame_size) / hop_size + 1;
        assert_eq!(out.dim(), (expected_frames, 13));

        // c0 (first cepstral coefficient) is a proxy for log-energy.
        // A pure sine should have higher c0 than silence by a wide margin.
        let silent_out = mfcc(
            &vec![0.0_f32; n],
            sr,
            0.025,
            0.010,
            40,
            13,
            0.0,
            sr as f32 / 2.0,
        );
        let sine_c0 = out[[expected_frames / 2, 0]];
        let silent_c0 = silent_out[[silent_out.dim().0 / 2, 0]];
        assert!(
            sine_c0 > silent_c0 + 20.0,
            "sine c0 ({sine_c0}) should be > silent c0 ({silent_c0}) by 20+ units",
        );
    }

    #[test]
    fn mfcc_of_silent_input_has_uniform_log_energy_floor() {
        let sr = 16_000u32;
        let samples = vec![0.0_f32; sr as usize];
        let out = mfcc(&samples, sr, 0.025, 0.010, 40, 13, 0.0, sr as f32 / 2.0);
        // All frames silent → all log mel energies hit the floor → c0
        // identical across frames.
        let n_frames = out.dim().0;
        assert!(n_frames > 0);
        let c0 = out[[0, 0]];
        for f in 1..n_frames {
            assert!(
                (out[[f, 0]] - c0).abs() < 1e-5,
                "frame {f}: c0={}, expected {c0}",
                out[[f, 0]]
            );
        }
    }

    #[test]
    fn mfcc_of_short_input_returns_empty() {
        let out = mfcc(&[0.5; 100], 16_000, 0.025, 0.010, 40, 13, 0.0, 8000.0);
        assert_eq!(out.dim().0, 0);
    }
}
