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

use crate::dsp::colormap::ColormapKind;
use crate::dsp::stft::Shape;

/// Floor for converting linear power to dB-FS without blowing up on silent
/// frames. Matches the floor [`power_to_db_normalized`] applies when
/// `power == 0`.
const POWER_DB_FLOOR: f32 = -200.0;

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

/// Converts linear power values into `[0, 1]` normalised dB-FS, suitable for
/// direct colormap indexing.
///
/// Pipeline per cell:
/// 1. `db = 10 · log10(power)` (or [`POWER_DB_FLOOR`] for silent cells).
/// 2. Find the global max across the buffer.
/// 3. Re-reference: `db_rel = db - max_db`.
/// 4. Clamp to `[-dynamic_range_db, 0]`.
/// 5. Normalise to `[0, 1]`: `(db_rel + dynamic_range_db) / dynamic_range_db`.
///
/// Returns an empty vector for empty input. `dynamic_range_db` must be `> 0`;
/// values `<= 0` are treated as `1.0` to avoid div-by-zero.
///
/// The layout is preserved element-for-element, so a freq-major power matrix
/// stays freq-major.
pub fn power_to_db_normalized(power: &[f32], dynamic_range_db: f32) -> Vec<f32> {
    if power.is_empty() {
        return Vec::new();
    }
    let dr = if dynamic_range_db > 0.0 {
        dynamic_range_db
    } else {
        1.0
    };
    // 1. power → dB (with floor for zeros).
    let mut db: Vec<f32> = power
        .iter()
        .map(|&p| {
            if p > 0.0 {
                10.0 * p.log10()
            } else {
                POWER_DB_FLOOR
            }
        })
        .collect();
    // 2. global max.
    let max_db = db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // 3–5. re-reference, clamp, normalise. In-place to save the alloc.
    for d in db.iter_mut() {
        let rel = (*d - max_db).clamp(-dr, 0.0);
        *d = (rel + dr) / dr;
    }
    db
}

/// Bakes a normalised `[0, 1]` freq-major buffer into a row-major `RGBA8`
/// image (4 bytes per pixel, opaque alpha).
///
/// `power` is laid out as `height` rows of `width` columns (a freq-major
/// spectrogram, `[freq_bin * width + frame]`). The output flips the y-axis so
/// frequency increases bottom→top in image coordinates — the convention every
/// spectrogram viewer uses. The result drops straight into
/// `egui::ColorImage::from_rgba_unmultiplied` (GUI) or an image encoder
/// (headless figure export).
pub fn colormap_bake(
    power: &[f32],
    width: usize,
    height: usize,
    colormap: ColormapKind,
) -> Vec<u8> {
    debug_assert_eq!(power.len(), width * height, "colormap_bake: shape mismatch");
    let mut out = vec![0u8; width * height * 4];
    for y in 0..height {
        // Flip: image row 0 is at the top, which should show the highest
        // frequency bin (`height - 1`).
        let bin = height - 1 - y;
        for x in 0..width {
            let v = power[bin * width + x].clamp(0.0, 1.0);
            let (r, g, b) = colormap.sample(v);
            let i = (y * width + x) * 4;
            out[i] = r;
            out[i + 1] = g;
            out[i + 2] = b;
            out[i + 3] = 255;
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
    fn power_to_db_normalized_empty_returns_empty() {
        assert!(power_to_db_normalized(&[], 70.0).is_empty());
    }

    #[test]
    fn power_to_db_normalized_constant_input_returns_ones() {
        // All cells equal → all re-referenced to 0 dB → normalised to 1.
        let out = power_to_db_normalized(&[1.0, 1.0, 1.0, 1.0], 70.0);
        for v in out {
            assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn power_to_db_normalized_clamps_below_dynamic_range() {
        // Max is 1.0 (0 dB); silent cell is at POWER_DB_FLOOR (very negative),
        // should clamp to the bottom of the range (= 0.0).
        let out = power_to_db_normalized(&[1.0, 0.0, 0.0, 0.0], 70.0);
        assert!((out[0] - 1.0).abs() < 1e-6);
        for &v in &out[1..] {
            assert!(v.abs() < 1e-6, "silent cell should clamp to 0.0, got {v}");
        }
    }

    #[test]
    fn power_to_db_normalized_midpoint_is_half_at_floor_minus_half() {
        // Cell at -35 dB relative to max with 70 dB range → 0.5 after
        // normalisation. power that gives -35 dB relative: 10^(-3.5) ≈ 0.000316.
        let out = power_to_db_normalized(&[1.0, 10f32.powf(-3.5)], 70.0);
        assert!((out[1] - 0.5).abs() < 1e-3, "got {}", out[1]);
    }

    #[test]
    fn colormap_bake_shape_is_rgba_height_times_width() {
        let power = vec![0.5_f32; 6]; // 2 freq bins × 3 frames
        let rgba = colormap_bake(&power, 3, 2, ColormapKind::Greyscale);
        assert_eq!(rgba.len(), 3 * 2 * 4);
        // Greyscale @ 0.5 ≈ (128, 128, 128, 255).
        for chunk in rgba.chunks_exact(4) {
            assert!((chunk[0] as i32 - 128).abs() < 2);
            assert_eq!(chunk[0], chunk[1]);
            assert_eq!(chunk[0], chunk[2]);
            assert_eq!(chunk[3], 255);
        }
    }

    #[test]
    fn colormap_bake_flips_y_axis_so_highest_freq_is_at_top() {
        // 2 freq bins × 1 frame. bin 0 (low) = 0.0, bin 1 (high) = 1.0.
        let power = vec![0.0_f32, 1.0_f32];
        let rgba = colormap_bake(&power, 1, 2, ColormapKind::Greyscale);
        // Image row 0 (top) should reflect the high freq (1.0 → 255).
        assert_eq!(rgba[0], 255, "image top row should be the high-freq cell");
        // Image row 1 (bottom) should reflect the low freq (0.0 → 0).
        assert_eq!(rgba[4], 0, "image bottom row should be the low-freq cell");
    }

    /// G0 bake-parity: the full engine pipeline (STFT power → dB-normalise →
    /// colormap bake) produces a well-formed opaque RGBA raster of exactly the
    /// spectrogram's `(n_frames × n_freq_bins)` shape, from real audio — the
    /// path the headless figure exporter drives.
    #[test]
    fn bake_pipeline_end_to_end_shape_and_opacity() {
        let sample_rate = 16_000u32;
        let freq = 1_000.0f32;
        let n_samples = sample_rate as usize / 2; // 0.5 s
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (TAU * freq * (i as f32 / sample_rate as f32)).sin())
            .collect();
        let window = hann(512);
        let (st, shape) = stft(&samples, &window, 128);
        let power = power_spectrogram(&st, shape);
        let normalized = power_to_db_normalized(&power, 70.0);
        assert_eq!(normalized.len(), shape.n_freq_bins * shape.n_frames);
        // Normalisation must keep every cell inside [0, 1].
        assert!(normalized.iter().all(|&v| (0.0..=1.0).contains(&v)));
        let rgba = colormap_bake(
            &normalized,
            shape.n_frames,
            shape.n_freq_bins,
            ColormapKind::Viridis,
        );
        assert_eq!(rgba.len(), shape.n_frames * shape.n_freq_bins * 4);
        // Every pixel is opaque.
        assert!(rgba.chunks_exact(4).all(|px| px[3] == 255));
    }

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
