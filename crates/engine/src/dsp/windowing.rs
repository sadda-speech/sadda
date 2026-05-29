//! Window functions: Hann, Hamming, Blackman, Gaussian, Kaiser.
//!
//! All five return a `Vec<f32>` of the requested length, computed against
//! the standard `scipy.signal.windows` symmetric formulas (so values match
//! scipy reference to within float precision).
//!
//! ## References
//! - Harris, F.J. (1978), "On the use of windows for harmonic analysis with
//!   the discrete Fourier transform." *Proc. IEEE* 66(1).
//!   <https://doi.org/10.1109/PROC.1978.10837>
//! - Kaiser, J.F. (1980), "Some useful properties of Teager's energy
//!   operators." (companion treatment of Kaiser window.)
//!   <https://doi.org/10.1109/ICASSP.1980.1170960>
//! - `scipy.signal.windows` source:
//!   <https://docs.scipy.org/doc/scipy/reference/signal.windows.html>
//!
//! Periodic-window variant (the `sym=False` librosa convention used by STFTs
//! for perfect-reconstruction overlap-add) is a deferred alternate per the
//! 2026-05-21 DSP method diversity entry.

use std::f32::consts::PI;

/// Hann window: `0.5 * (1 - cos(2π n / (N-1)))`.
/// Length-0 / length-1 inputs return an empty / single-element vec.
pub fn hann(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f32;
    (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / denom).cos()))
        .collect()
}

/// Periodic ("DFT-even") Hann window: `0.5 * (1 - cos(2π n / N))` — the
/// `sym=False` / `fftbins=True` convention used by STFT front ends
/// (`torch.hann_window` default, `librosa.stft`, OpenAI Whisper). Differs
/// from [`hann`] only in the denominator (`N` vs `N-1`); use this where an
/// analysis must match those libraries bin-for-bin, [`hann`] for the
/// scipy-symmetric default. Length-0 / length-1 return empty / `[1.0]`.
pub fn hann_periodic(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = n as f32;
    (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / denom).cos()))
        .collect()
}

/// Hamming window: `0.54 - 0.46 * cos(2π n / (N-1))`.
pub fn hamming(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f32;
    (0..n)
        .map(|k| 0.54 - 0.46 * (2.0 * PI * k as f32 / denom).cos())
        .collect()
}

/// Blackman window: `0.42 - 0.5*cos(2π n / (N-1)) + 0.08*cos(4π n / (N-1))`.
pub fn blackman(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let denom = (n - 1) as f32;
    (0..n)
        .map(|k| {
            let t = 2.0 * PI * k as f32 / denom;
            0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
        })
        .collect()
}

/// Gaussian window: `exp(-0.5 * ((k - (N-1)/2) / sigma)^2)`.
///
/// `sigma` controls the width in samples. `sigma > 0`; smaller sigma → tighter
/// taper. Common choice for analysis frames is `sigma = (n - 1) / 6`
/// (~99.7% energy inside the frame), but C1 doesn't bake that in — callers
/// pass `sigma` explicitly.
pub fn gaussian(n: usize, sigma: f32) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let center = (n - 1) as f32 / 2.0;
    (0..n)
        .map(|k| {
            let x = (k as f32 - center) / sigma;
            (-0.5 * x * x).exp()
        })
        .collect()
}

/// Kaiser window: `I0(β * sqrt(1 - ((k - (N-1)/2) / ((N-1)/2))^2)) / I0(β)`.
///
/// `beta` controls the trade-off between main-lobe width and side-lobe
/// attenuation; common values are 5 (moderate roll-off) and 8.6 (Praat's
/// default for Hann-like analysis).
pub fn kaiser(n: usize, beta: f32) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    let half = (n - 1) as f32 / 2.0;
    let i0_beta = bessel_i0(beta);
    (0..n)
        .map(|k| {
            let r = (k as f32 - half) / half;
            let arg = beta * (1.0 - r * r).max(0.0).sqrt();
            bessel_i0(arg) / i0_beta
        })
        .collect()
}

/// Modified Bessel function of the first kind, order 0.
/// Power-series approximation via `(x/2)^(2k) / (k!)^2` — converges fast
/// for the `|x| < 50` range Kaiser windows live in.
fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0_f32;
    let mut term = 1.0_f32;
    let half_x_sq = (x * 0.5) * (x * 0.5);
    for k in 1..50 {
        term *= half_x_sq / (k as f32 * k as f32);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hann at k=0 and k=N-1 is exactly 0; at k=(N-1)/2 it's exactly 1.
    #[test]
    fn hann_endpoints_are_zero_and_peak_is_one() {
        let w = hann(9);
        assert!((w[0]).abs() < 1e-6);
        assert!((w[8]).abs() < 1e-6);
        assert!((w[4] - 1.0).abs() < 1e-6);
    }

    /// Hann is symmetric.
    #[test]
    fn hann_is_symmetric() {
        let w = hann(32);
        for i in 0..16 {
            assert!((w[i] - w[31 - i]).abs() < 1e-6, "i={i}");
        }
    }

    /// Hamming at k=0 and k=N-1 is exactly 0.08 (since 0.54 - 0.46).
    #[test]
    fn hamming_endpoints_are_zero_point_zero_eight() {
        let w = hamming(11);
        assert!((w[0] - 0.08).abs() < 1e-6);
        assert!((w[10] - 0.08).abs() < 1e-6);
    }

    /// Blackman endpoints: 0.42 - 0.5 + 0.08 = 0.0 exactly.
    #[test]
    fn blackman_endpoints_are_zero() {
        let w = blackman(13);
        assert!(w[0].abs() < 1e-6);
        assert!(w[12].abs() < 1e-6);
    }

    /// Gaussian: max at center, equals 1.0 there.
    #[test]
    fn gaussian_peak_is_at_center_and_equals_one() {
        let w = gaussian(11, 2.0);
        assert!((w[5] - 1.0).abs() < 1e-6);
        for i in 0..5 {
            assert!(w[i] < w[i + 1], "w[{i}]={} >= w[{}]", w[i], w[i + 1]);
        }
    }

    /// Kaiser at beta=0 is the rectangular window (all 1s).
    #[test]
    fn kaiser_beta_zero_is_rectangular() {
        let w = kaiser(7, 0.0);
        for &v in &w {
            assert!((v - 1.0).abs() < 1e-5, "got {v}");
        }
    }

    /// Kaiser endpoints with beta>0 are small; center is 1.0.
    #[test]
    fn kaiser_with_beta_decays_to_endpoints() {
        let w = kaiser(11, 8.6);
        assert!((w[5] - 1.0).abs() < 1e-5);
        assert!(w[0] < 0.05, "expected small endpoint, got {}", w[0]);
        assert!(w[10] < 0.05, "expected small endpoint, got {}", w[10]);
    }

    #[test]
    fn zero_and_one_length_edge_cases() {
        assert!(hann(0).is_empty());
        assert_eq!(hann(1), vec![1.0]);
        assert_eq!(gaussian(1, 1.0), vec![1.0]);
        assert_eq!(kaiser(1, 5.0), vec![1.0]);
    }

    /// Modified Bessel I0 sanity: I0(0) = 1; I0 is monotone in |x| for x>=0.
    #[test]
    fn bessel_i0_baseline_values() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-6);
        // Known reference: I0(1) ≈ 1.2660658732, I0(2) ≈ 2.2795853024.
        assert!((bessel_i0(1.0) - 1.266_066).abs() < 1e-4);
        assert!((bessel_i0(2.0) - 2.279_585).abs() < 1e-3);
    }
}
