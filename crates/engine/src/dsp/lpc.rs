//! Linear Predictive Coding (LPC) — predictor coefficients for an AR(p)
//! model of an audio frame. Two non-equivalent methods are exposed here per
//! the 2026-05-21 DSP method-diversity entry:
//!
//! - [`autocorr_lpc`] — autocorrelation method via the Levinson-Durbin
//!   recursion. Always produces a stable predictor (reflection coefficients
//!   |k_i| < 1) but tapers signal energy at frame edges (effectively assumes
//!   zero-extension outside the frame), which biases formant estimates on
//!   short frames.
//! - [`burg_lpc`] — Burg's method (added in C2). Estimates reflection
//!   coefficients directly from forward/backward prediction errors without
//!   the windowing implied by the autocorrelation method; tends to be more
//!   accurate on short frames. Praat's `Sound.to_formant_burg` uses this.
//!
//! ## References
//! - Makhoul, J. (1975), "Linear prediction: A tutorial review." *Proc. IEEE*
//!   63(4). <https://doi.org/10.1109/PROC.1975.9792>
//! - Markel, J.D. & Gray, A.H. (1976), *Linear Prediction of Speech*.
//!   Springer. ISBN 978-3-642-66288-1
//! - Levinson, N. (1947), "The Wiener RMS error criterion in filter design
//!   and prediction." *J. Math. & Phys.* 25(1).
//!   <https://doi.org/10.1002/sapm1946251261>
//! - Durbin, J. (1960), "The fitting of time-series models." *Rev. Int. Stat.
//!   Inst.* 28(3). <https://doi.org/10.2307/1401322>
//! - Burg, J.P. (1975), *Maximum Entropy Spectral Analysis*. PhD thesis,
//!   Stanford. <https://sepwww.stanford.edu/data/media/public/oldreports/sep06/>
//! - Praat formant-via-Burg manual:
//!   <https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html>
//!
//! Given a signal frame, computes the predictor polynomial coefficients
//! `a_1, ..., a_p` that minimize the residual energy when predicting
//! `x[n]` from a linear combination of `x[n-1], ..., x[n-p]`.

use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};

/// One of the two LPC estimation methods exposed here. See module-level
/// docs for the trade-offs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LpcMethod {
    /// Autocorrelation method via Levinson-Durbin recursion. Always produces
    /// a stable predictor.
    Autocorrelation,
    /// Burg's method. Estimates reflection coefficients directly from
    /// forward/backward prediction errors; more accurate on short frames.
    /// Praat's `Sound.to_formant_burg` default.
    Burg,
}

/// Output of [`autocorr_lpc`] / [`burg_lpc`] / [`lpc`].
#[derive(Debug, Clone)]
pub struct LpcResult {
    /// Predictor coefficients `[a_1, a_2, ..., a_p]` (length `order`).
    /// The full LPC polynomial is `1 + a_1 z^-1 + ... + a_p z^-p`.
    pub coeffs: Vec<f32>,
    /// Reflection (PARCOR) coefficients `[k_1, k_2, ..., k_p]`.
    pub reflection: Vec<f32>,
    /// Final prediction error (residual variance after p-th iteration).
    pub gain: f32,
    /// Which method produced this result.
    pub method: LpcMethod,
}

/// Dispatches to [`autocorr_lpc`] or [`burg_lpc`] based on `method`.
pub fn lpc(samples: &[f32], order: usize, method: LpcMethod) -> Result<LpcResult> {
    match method {
        LpcMethod::Autocorrelation => autocorr_lpc(samples, order),
        LpcMethod::Burg => burg_lpc(samples, order),
    }
}

/// Computes LPC coefficients of `order` for `samples` via the autocorrelation
/// method (Levinson-Durbin recursion).
///
/// `order` must satisfy `1 <= order < samples.len()`. Returns an error if the
/// autocorrelation matrix is degenerate (e.g. the input is all zeros).
pub fn autocorr_lpc(samples: &[f32], order: usize) -> Result<LpcResult> {
    if order == 0 {
        return Err(EngineError::Corpus("LPC order must be >= 1".into()));
    }
    if samples.len() <= order {
        return Err(EngineError::Corpus(format!(
            "LPC: samples.len() = {} must exceed order = {order}",
            samples.len()
        )));
    }

    // Autocorrelation R[k] = Σ x[n] * x[n+k] for k = 0, ..., order.
    let mut r = vec![0.0_f32; order + 1];
    for k in 0..=order {
        let mut sum = 0.0_f32;
        for n in 0..samples.len() - k {
            sum += samples[n] * samples[n + k];
        }
        r[k] = sum;
    }

    if r[0] <= f32::EPSILON {
        return Err(EngineError::Corpus(
            "LPC: zero-energy frame, autocorrelation R[0] is ~0".into(),
        ));
    }

    // Levinson-Durbin recursion.
    let mut a = vec![0.0_f32; order + 1];
    let mut reflection = vec![0.0_f32; order];
    a[0] = 1.0;
    let mut error = r[0];

    for i in 1..=order {
        // k_i = -(R[i] + Σ_{j=1}^{i-1} a_j * R[i-j]) / error
        let mut acc = r[i];
        for j in 1..i {
            acc += a[j] * r[i - j];
        }
        let k = -acc / error;
        reflection[i - 1] = k;

        // Update a in place: a_j^(i) = a_j^(i-1) + k * a_{i-j}^(i-1).
        let mut new_a = a.clone();
        new_a[i] = k;
        for j in 1..i {
            new_a[j] = a[j] + k * a[i - j];
        }
        a = new_a;
        error *= 1.0 - k * k;
        if error <= 0.0 {
            // Numerical breakdown — autocorrelation matrix not positive definite.
            return Err(EngineError::Corpus(format!(
                "LPC: Levinson-Durbin breakdown at order {i} (gain became non-positive)"
            )));
        }
    }

    // Strip the leading 1.0 — callers expect coeffs[0] == a_1.
    let coeffs: Vec<f32> = a[1..].to_vec();
    Ok(LpcResult {
        coeffs,
        reflection,
        gain: error,
        method: LpcMethod::Autocorrelation,
    })
}

/// Burg's method (1975) for LPC coefficient estimation. Computes the
/// reflection coefficients directly from forward and backward prediction
/// errors by minimising their combined power at each lattice stage. Avoids
/// the implicit zero-extension windowing of the autocorrelation method,
/// giving more accurate estimates on short frames — at the cost of slightly
/// more arithmetic per stage.
///
/// Sign convention matches [`autocorr_lpc`]: the returned predictor
/// polynomial is `1 + a_1 z⁻¹ + ... + a_p z⁻ᵖ`.
pub fn burg_lpc(samples: &[f32], order: usize) -> Result<LpcResult> {
    if order == 0 {
        return Err(EngineError::Corpus("LPC order must be >= 1".into()));
    }
    if samples.len() <= order {
        return Err(EngineError::Corpus(format!(
            "Burg LPC: samples.len() = {} must exceed order = {order}",
            samples.len()
        )));
    }
    let n = samples.len();

    let mut ef: Vec<f32> = samples.to_vec();
    let mut eb: Vec<f32> = samples.to_vec();

    let mut a: Vec<f32> = vec![0.0; order + 1];
    a[0] = 1.0;
    let mut reflection = vec![0.0_f32; order];

    let mut gain: f32 = samples.iter().map(|&x| x * x).sum::<f32>() / n as f32;
    if gain <= f32::EPSILON {
        return Err(EngineError::Corpus(
            "Burg LPC: zero-energy frame (signal is all zeros)".into(),
        ));
    }

    for m in 1..=order {
        // k_m = -2 · Σ ef·eb_prev / Σ (ef² + eb_prev²)
        let mut num = 0.0_f32;
        let mut den = 0.0_f32;
        for j in m..n {
            num += ef[j] * eb[j - 1];
            den += ef[j] * ef[j] + eb[j - 1] * eb[j - 1];
        }
        if den <= f32::EPSILON {
            return Err(EngineError::Corpus(format!(
                "Burg LPC: zero denominator at order {m} (signal too short or degenerate)"
            )));
        }
        let k = -2.0 * num / den;
        reflection[m - 1] = k;

        // LPC coefficient update (Levinson-style; matches autocorr_lpc).
        let mut new_a = a.clone();
        new_a[m] = k;
        for i in 1..m {
            new_a[i] = a[i] + k * a[m - i];
        }
        a = new_a;

        // Forward and backward error update; iterate downward so eb[j-1]
        // and ef[j] both still hold their pre-update values.
        for j in (m..n).rev() {
            let old_ef = ef[j];
            let old_eb = eb[j - 1];
            ef[j] = old_ef + k * old_eb;
            eb[j] = old_eb + k * old_ef;
        }

        gain *= 1.0 - k * k;
        if gain <= 0.0 {
            return Err(EngineError::Corpus(format!(
                "Burg LPC: gain became non-positive at order {m}"
            )));
        }
    }

    Ok(LpcResult {
        coeffs: a[1..].to_vec(),
        reflection,
        gain,
        method: LpcMethod::Burg,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    #[test]
    fn lpc_rejects_zero_order_and_short_input() {
        assert!(autocorr_lpc(&[1.0, 2.0, 3.0], 0).is_err());
        assert!(autocorr_lpc(&[1.0, 2.0], 5).is_err());
    }

    #[test]
    fn lpc_zero_input_errors_cleanly() {
        assert!(autocorr_lpc(&[0.0_f32; 100], 8).is_err());
    }

    #[test]
    fn lpc_of_pure_sine_picks_up_the_resonance() {
        // A sine wave at f0 is well-modelled by a 2nd-order predictor with
        // coefficients ~(-2 cos(2π f0 / sr), 1).
        let sr = 16_000.0_f32;
        let f0 = 220.0_f32;
        let n = 1024;
        let samples: Vec<f32> = (0..n).map(|i| (TAU * f0 * i as f32 / sr).sin()).collect();
        let result = autocorr_lpc(&samples, 2).unwrap();
        let expected_a1 = -2.0 * (TAU * f0 / sr).cos();
        // The 2nd-order LPC of a windowless finite sine won't be exact (the
        // signal isn't a perfect AR(2) process — endpoints leak energy across
        // the autocorrelation), so we allow a generous tolerance. The point
        // of the test is that the predictor lands in the right *direction*.
        assert!(
            (result.coeffs[0] - expected_a1).abs() < 0.1,
            "got a_1 = {}, expected ~{}",
            result.coeffs[0],
            expected_a1
        );
        // a_2 should be near +1 for an undamped sine.
        assert!(
            result.coeffs[1] > 0.7 && result.coeffs[1] < 1.1,
            "got a_2 = {}, expected ~1.0",
            result.coeffs[1]
        );
    }

    #[test]
    fn lpc_reflection_coeffs_are_within_unit_interval_for_stable_signals() {
        let sr = 16_000.0_f32;
        let f0 = 440.0_f32;
        let n = 1024;
        let samples: Vec<f32> = (0..n).map(|i| (TAU * f0 * i as f32 / sr).sin()).collect();
        let result = autocorr_lpc(&samples, 8).unwrap();
        // Lattice filter is stable iff |k_i| < 1 for all i.
        for (i, &k) in result.reflection.iter().enumerate() {
            assert!(k.abs() < 1.0, "reflection[{i}] = {k} out of (-1, 1)");
        }
    }

    #[test]
    fn burg_lpc_rejects_zero_order_and_short_input() {
        assert!(burg_lpc(&[1.0, 2.0, 3.0], 0).is_err());
        assert!(burg_lpc(&[1.0, 2.0], 5).is_err());
    }

    #[test]
    fn burg_lpc_zero_input_errors_cleanly() {
        assert!(burg_lpc(&[0.0_f32; 100], 8).is_err());
    }

    #[test]
    fn burg_lpc_of_pure_sine_picks_up_the_resonance() {
        let sr = 16_000.0_f32;
        let f0 = 220.0_f32;
        let n = 1024;
        let samples: Vec<f32> = (0..n).map(|i| (TAU * f0 * i as f32 / sr).sin()).collect();
        let result = burg_lpc(&samples, 2).unwrap();
        let expected_a1 = -2.0 * (TAU * f0 / sr).cos();
        assert!(
            (result.coeffs[0] - expected_a1).abs() < 0.05,
            "Burg a_1 = {}, expected ~{}",
            result.coeffs[0],
            expected_a1
        );
        assert!(
            result.coeffs[1] > 0.85 && result.coeffs[1] < 1.05,
            "Burg a_2 = {}, expected ~1.0",
            result.coeffs[1]
        );
    }

    #[test]
    fn burg_reflection_coeffs_are_within_unit_interval_for_stable_signals() {
        let sr = 16_000.0_f32;
        let f0 = 440.0_f32;
        let n = 1024;
        let samples: Vec<f32> = (0..n).map(|i| (TAU * f0 * i as f32 / sr).sin()).collect();
        let result = burg_lpc(&samples, 8).unwrap();
        for (i, &k) in result.reflection.iter().enumerate() {
            assert!(k.abs() < 1.0, "reflection[{i}] = {k} out of (-1, 1)");
        }
    }

    #[test]
    fn burg_gain_decreases_monotonically_with_order() {
        // As order increases the residual power can only stay equal or
        // drop — that's a sanity check independent of the chosen method.
        let sr = 16_000.0_f32;
        let f0 = 440.0_f32;
        let n = 4096;
        let samples: Vec<f32> = (0..n).map(|i| (TAU * f0 * i as f32 / sr).sin()).collect();
        let mut prev_gain = f32::INFINITY;
        for order in 1..=6 {
            let r = burg_lpc(&samples, order).unwrap();
            assert!(
                r.gain <= prev_gain + 1e-6,
                "order {order}: gain {} > prev {prev_gain}",
                r.gain
            );
            prev_gain = r.gain;
        }
    }

    #[test]
    fn lpc_dispatcher_picks_the_right_method() {
        let sr = 16_000.0_f32;
        let f0 = 220.0_f32;
        let n = 1024;
        let samples: Vec<f32> = (0..n).map(|i| (TAU * f0 * i as f32 / sr).sin()).collect();
        let auto = lpc(&samples, 4, LpcMethod::Autocorrelation).unwrap();
        let burg = lpc(&samples, 4, LpcMethod::Burg).unwrap();
        assert_eq!(auto.method, LpcMethod::Autocorrelation);
        assert_eq!(burg.method, LpcMethod::Burg);
    }
}
