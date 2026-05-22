//! Power spectrogram: magnitude-squared of an STFT, transposed to the
//! canonical `(n_freq_bins, n_frames)` shape from the 2026-05-18 API
//! surface entry.
//!
//! ## References
//! - Oppenheim, A.V. & Schafer, R.W. (2010), *Discrete-Time Signal
//!   Processing*, 3rd ed., §10.3.4 (Spectrogram).
//!   ISBN 978-0-13-198842-2
//! - `scipy.signal.spectrogram`:
//!   <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html>
//!
//! C1 ships **power** (`|X|²`) only; magnitude (`|X|`) and log-power
//! (`10·log10(|X|²)`) are deferred alternates per the DSP method-diversity
//! entry.

use rustfft::num_complex::Complex;

use crate::dsp::stft::Shape;

/// Returns the power spectrogram of an STFT output, row-major in shape
/// `(n_freq_bins, n_frames)`. Element `[bin, frame] = |X[frame, bin]|²`.
///
/// `stft_out` is the flattened STFT in `(n_frames, n_freq_bins)` layout (the
/// shape [`stft`] returns); `shape` is its companion descriptor.
pub fn power_spectrogram(stft_out: &[Complex<f32>], shape: Shape) -> Vec<f32> {
    let Shape {
        n_frames,
        n_freq_bins,
    } = shape;
    assert_eq!(
        stft_out.len(),
        n_frames * n_freq_bins,
        "STFT length mismatch"
    );
    let mut out = vec![0.0_f32; n_freq_bins * n_frames];
    for f in 0..n_frames {
        for b in 0..n_freq_bins {
            let c = stft_out[f * n_freq_bins + b];
            // |c|² = re² + im²; cheaper than .norm() (which does a sqrt).
            out[b * n_frames + f] = c.re * c.re + c.im * c.im;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::stft::stft;
    use crate::dsp::windowing::hann;
    use std::f32::consts::TAU;

    #[test]
    fn power_spectrogram_shape_is_freq_first() {
        let frame_size = 64;
        let hop = 16;
        let samples = vec![0.1_f32; 256];
        let window = hann(frame_size);
        let (st, shape) = stft(&samples, &window, hop);
        let p = power_spectrogram(&st, shape);
        assert_eq!(p.len(), shape.n_freq_bins * shape.n_frames);
    }

    #[test]
    fn power_spectrogram_of_pure_sine_peaks_at_expected_bin_per_frame() {
        let sample_rate = 16_000u32;
        let freq = 1_000.0f32;
        let n_samples = sample_rate as usize / 2; // 0.5 s
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (TAU * freq * (i as f32 / sample_rate as f32)).sin())
            .collect();
        let frame_size = 1024;
        let hop = 256;
        let window = hann(frame_size);
        let (st, shape) = stft(&samples, &window, hop);
        let p = power_spectrogram(&st, shape);

        let bin_hz = sample_rate as f32 / frame_size as f32;
        let expected_bin = (freq / bin_hz).round() as usize;

        // For each frame (column f), the row with the largest power should be
        // near expected_bin.
        for f in 0..shape.n_frames {
            let mut peak_bin = 0usize;
            let mut peak_val = 0.0_f32;
            for b in 0..shape.n_freq_bins {
                let v = p[b * shape.n_frames + f];
                if v > peak_val {
                    peak_val = v;
                    peak_bin = b;
                }
            }
            assert!(
                peak_bin.abs_diff(expected_bin) <= 1,
                "frame {f}: peak at bin {peak_bin}, expected near {expected_bin}",
            );
        }
    }
}
