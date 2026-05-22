//! Polynomial root-finding via the Aberth-Ehrlich (parallel-Newton with
//! deflation correction) method. Used by the formant tracker to find the
//! roots of the LPC predictor polynomial.
//!
//! ## References
//! - Aberth, O. (1973), "Iteration methods for finding all zeros of a
//!   polynomial simultaneously." *Math. Comp.* 27(122).
//!   <https://doi.org/10.1090/S0025-5718-1973-0329236-7>
//! - Bini, D.A. (1996), "Numerical computation of polynomial zeros by
//!   means of Aberth's method." *Numerical Algorithms* 13.
//!   <https://doi.org/10.1007/BF02207694>
//! - Pan, V.Y. (1997), "Solving a polynomial equation: Some history and
//!   recent progress." *SIAM Review* 39(2). (Survey context.)
//!   <https://doi.org/10.1137/S0036144595288804>

use std::f64::consts::PI;

use rustfft::num_complex::Complex;

use crate::error::{EngineError, Result};

/// Finds all complex roots of the polynomial
/// `coeffs[0] + coeffs[1]·z + ... + coeffs[n]·z^n` (ascending order).
/// Returns a `Vec<Complex<f64>>` of length `n` (the polynomial degree).
///
/// Uses the Aberth-Ehrlich iteration:
///
/// ```text
/// z_j ← z_j - p(z_j) / (p'(z_j) - p(z_j) · Σ_{k≠j} 1 / (z_j - z_k))
/// ```
///
/// Initial guesses are spread around a circle of radius `(max|c_i / c_n|)^(1/n)`
/// (a variant of the Cauchy bound). Converges quadratically for
/// well-conditioned polynomials; LPC predictor polynomials from speech are
/// typically well-conditioned.
pub fn polynomial_roots(coeffs: &[f64]) -> Result<Vec<Complex<f64>>> {
    if coeffs.len() < 2 {
        return Err(EngineError::Corpus(
            "polynomial_roots: degree must be >= 1".into(),
        ));
    }
    let n = coeffs.len() - 1; // polynomial degree
    let leading = coeffs[n];
    if leading.abs() < 1e-30 {
        return Err(EngineError::Corpus(
            "polynomial_roots: leading coefficient is zero".into(),
        ));
    }

    // Cauchy-bound-like initial radius. Floor at 0.1 so all-tiny polynomials
    // (LPC of near-silent frames) don't collapse to the origin.
    let max_ratio = coeffs[..n]
        .iter()
        .map(|c| (c / leading).abs())
        .fold(0.0_f64, f64::max);
    let radius = max_ratio.powf(1.0 / n as f64).max(0.1);

    // Initial roots spread on a circle, offset by π/(2n) so points aren't on
    // the real axis (which would coincide with potential real roots).
    let mut z: Vec<Complex<f64>> = (0..n)
        .map(|j| {
            let theta = 2.0 * PI * (j as f64 + 0.5) / n as f64;
            Complex::new(radius * theta.cos(), radius * theta.sin())
        })
        .collect();

    const MAX_ITER: usize = 200;
    const TOL: f64 = 1e-10;

    for _iter in 0..MAX_ITER {
        // Snapshot so all corrections in this iteration use the same z_k
        // (true "parallel-Newton" semantics; Bini 1996 shows convergence
        // is robust either way, but parallel is more predictable).
        let z_prev = z.clone();
        let mut max_correction = 0.0_f64;
        for j in 0..n {
            let zj = z_prev[j];
            let (p, dp) = horner_with_deriv(coeffs, zj);
            let p_norm = p.norm();
            if p_norm < TOL {
                continue; // root already at machine precision
            }
            if dp.norm() < f64::EPSILON {
                // Derivative vanished — happens at multiple roots or
                // numerical breakdown. Perturb slightly and try again next
                // iteration.
                z[j] += Complex::new(1e-6, 1e-6);
                continue;
            }
            let mut sum = Complex::new(0.0, 0.0);
            for (k, &zk) in z_prev.iter().enumerate() {
                if k != j {
                    let diff = zj - zk;
                    if diff.norm() > f64::EPSILON {
                        sum += Complex::new(1.0, 0.0) / diff;
                    }
                }
            }
            let denom = dp - p * sum;
            if denom.norm() < f64::EPSILON {
                continue;
            }
            let correction = p / denom;
            z[j] = zj - correction;
            let mag = correction.norm();
            if mag > max_correction {
                max_correction = mag;
            }
        }
        if max_correction < TOL {
            break;
        }
    }
    Ok(z)
}

/// Evaluates `p(z)` and `p'(z)` simultaneously via Horner's rule.
/// `coeffs` is ascending: `p(z) = Σ coeffs[i] · z^i`.
fn horner_with_deriv(coeffs: &[f64], z: Complex<f64>) -> (Complex<f64>, Complex<f64>) {
    let n = coeffs.len();
    let mut p = Complex::new(coeffs[n - 1], 0.0);
    let mut dp = Complex::new(0.0, 0.0);
    for i in (0..n - 1).rev() {
        dp = dp * z + p;
        p = p * z + Complex::new(coeffs[i], 0.0);
    }
    (p, dp)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `match_roots(found, expected, tol)`: every `expected` root has at
    /// least one `found` root within `tol`. Order-independent.
    fn match_roots(found: &[Complex<f64>], expected: &[Complex<f64>], tol: f64) {
        for e in expected {
            let best = found
                .iter()
                .map(|f| (f - e).norm())
                .fold(f64::INFINITY, f64::min);
            assert!(
                best < tol,
                "expected root {e} not matched within {tol}; best distance {best}"
            );
        }
    }

    #[test]
    fn quadratic_with_real_roots() {
        // x² - 1 = (x - 1)(x + 1)
        let roots = polynomial_roots(&[-1.0, 0.0, 1.0]).unwrap();
        assert_eq!(roots.len(), 2);
        match_roots(
            &roots,
            &[Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)],
            1e-6,
        );
    }

    #[test]
    fn quadratic_with_complex_roots() {
        // x² + 1 = (x - i)(x + i)
        let roots = polynomial_roots(&[1.0, 0.0, 1.0]).unwrap();
        match_roots(
            &roots,
            &[Complex::new(0.0, 1.0), Complex::new(0.0, -1.0)],
            1e-6,
        );
    }

    #[test]
    fn cubic_with_one_real_one_complex_pair() {
        // x³ - 1 = (x - 1)(x² + x + 1)
        // Roots: 1, -1/2 + i√3/2, -1/2 - i√3/2
        let roots = polynomial_roots(&[-1.0, 0.0, 0.0, 1.0]).unwrap();
        let s = 0.5_f64 * 3.0_f64.sqrt();
        match_roots(
            &roots,
            &[
                Complex::new(1.0, 0.0),
                Complex::new(-0.5, s),
                Complex::new(-0.5, -s),
            ],
            1e-6,
        );
    }

    #[test]
    fn linear_polynomial_works() {
        // 2x + 4 = 0 → x = -2
        let roots = polynomial_roots(&[4.0, 2.0]).unwrap();
        assert_eq!(roots.len(), 1);
        match_roots(&roots, &[Complex::new(-2.0, 0.0)], 1e-9);
    }

    #[test]
    fn rejects_degenerate_input() {
        assert!(polynomial_roots(&[]).is_err());
        assert!(polynomial_roots(&[1.0]).is_err());
        // Leading coefficient zero.
        assert!(polynomial_roots(&[1.0, 2.0, 0.0]).is_err());
    }

    #[test]
    fn quartic_finds_four_roots() {
        // (x² + 1)(x² + 4) = x⁴ + 5x² + 4
        // Roots: ±i, ±2i
        let roots = polynomial_roots(&[4.0, 0.0, 5.0, 0.0, 1.0]).unwrap();
        assert_eq!(roots.len(), 4);
        match_roots(
            &roots,
            &[
                Complex::new(0.0, 1.0),
                Complex::new(0.0, -1.0),
                Complex::new(0.0, 2.0),
                Complex::new(0.0, -2.0),
            ],
            1e-5,
        );
    }

    #[test]
    fn roots_of_lpc_polynomial_lie_inside_unit_circle_for_stable_predictor() {
        // A representative LPC polynomial of order 4 with all reflection
        // coefficients < 1 should have all roots strictly inside the unit
        // circle. Construct one by hand: (z - 0.9 e^iπ/4)(z - 0.9 e^-iπ/4)
        //                              · (z - 0.7 e^iπ/2)(z - 0.7 e^-iπ/2).
        // Expand to ascending coefficients.
        let r1 = 0.9_f64;
        let t1 = std::f64::consts::PI / 4.0;
        let r2 = 0.7_f64;
        let t2 = std::f64::consts::PI / 2.0;
        // (z² - 2 r1 cos(t1) z + r1²) · (z² - 2 r2 cos(t2) z + r2²)
        let p1 = [r1 * r1, -2.0 * r1 * t1.cos(), 1.0]; // ascending in z
        let p2 = [r2 * r2, -2.0 * r2 * t2.cos(), 1.0];
        // Multiply ascending polynomials.
        let mut prod = vec![0.0_f64; p1.len() + p2.len() - 1];
        for (i, &a) in p1.iter().enumerate() {
            for (j, &b) in p2.iter().enumerate() {
                prod[i + j] += a * b;
            }
        }
        let roots = polynomial_roots(&prod).unwrap();
        assert_eq!(roots.len(), 4);
        for r in &roots {
            assert!(r.norm() < 1.0, "root {r} should lie inside unit circle");
        }
    }
}
