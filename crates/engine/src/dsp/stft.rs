//! Short-time Fourier transform over real-valued audio. Real-input
//! optimized via `realfft`; output is the unique half of the spectrum
//! (`n_freq_bins = window.len() / 2 + 1`).

use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;

/// Output shape of [`stft`]. `n_frames` is the number of analysis windows
/// that fit; `n_freq_bins` is `window.len() / 2 + 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// Number of analysis frames.
    pub n_frames: usize,
    /// Number of unique frequency bins per frame (`window.len() / 2 + 1`).
    pub n_freq_bins: usize,
}

/// Returns the row-major STFT of `samples`, window-by-window. Layout is
/// `(n_frames, n_freq_bins)`; row `f` is the spectrum of frame `f`.
///
/// `samples`: real-valued input.
/// `window`: precomputed window function values; its length is the frame size.
/// `hop_size`: stride between successive frame starts, in samples.
///
/// Returns an empty `Vec` (with `n_frames = 0`) if `samples.len() < window.len()`.
pub fn stft(samples: &[f32], window: &[f32], hop_size: usize) -> (Vec<Complex<f32>>, Shape) {
    let frame_size = window.len();
    assert!(frame_size > 0, "STFT requires a non-empty window");
    assert!(hop_size > 0, "STFT requires hop_size > 0");

    let n_freq_bins = frame_size / 2 + 1;
    if samples.len() < frame_size {
        return (
            Vec::new(),
            Shape {
                n_frames: 0,
                n_freq_bins,
            },
        );
    }

    let n_frames = (samples.len() - frame_size) / hop_size + 1;
    let mut planner = RealFftPlanner::<f32>::new();
    let plan = planner.plan_fft_forward(frame_size);
    let mut input = plan.make_input_vec();
    let mut output = plan.make_output_vec();
    let mut all = Vec::with_capacity(n_frames * n_freq_bins);

    for f in 0..n_frames {
        let start = f * hop_size;
        let frame = &samples[start..start + frame_size];
        for (dst, (&s, &w)) in input.iter_mut().zip(frame.iter().zip(window.iter())) {
            *dst = s * w;
        }
        // realfft writes the unique half of the spectrum into `output`.
        plan.process(&mut input, &mut output)
            .expect("realfft process: input/output sized via make_*_vec");
        all.extend_from_slice(&output);
    }

    (
        all,
        Shape {
            n_frames,
            n_freq_bins,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::windowing::hann;
    use std::f32::consts::TAU;

    /// A pure 440 Hz sine sampled at 16 kHz should peak at the bin closest
    /// to 440 Hz in every frame.
    #[test]
    fn stft_of_pure_sine_peaks_at_expected_bin() {
        let sample_rate = 16_000u32;
        let freq = 440.0f32;
        let duration_secs = 0.5;
        let n_samples = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (TAU * freq * (i as f32 / sample_rate as f32)).sin())
            .collect();

        let frame_size = 1024;
        let hop = 256;
        let window = hann(frame_size);
        let (data, shape) = stft(&samples, &window, hop);
        assert!(shape.n_frames > 0);
        assert_eq!(shape.n_freq_bins, frame_size / 2 + 1);

        let bin_hz = sample_rate as f32 / frame_size as f32;
        let expected_bin = (freq / bin_hz).round() as usize;

        // For every frame, the magnitude must peak near expected_bin.
        for f in 0..shape.n_frames {
            let row = &data[f * shape.n_freq_bins..(f + 1) * shape.n_freq_bins];
            let (peak_bin, _peak_mag) = row
                .iter()
                .map(|c| c.norm())
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            assert!(
                peak_bin.abs_diff(expected_bin) <= 1,
                "frame {f}: peak at bin {peak_bin}, expected near {expected_bin}",
            );
        }
    }

    #[test]
    fn stft_with_short_input_returns_zero_frames() {
        let window = hann(1024);
        let samples = vec![0.5_f32; 100];
        let (data, shape) = stft(&samples, &window, 256);
        assert_eq!(shape.n_frames, 0);
        assert!(data.is_empty());
    }
}
