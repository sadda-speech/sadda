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
use crate::units::{Decibels, Ratio, Seconds};

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
    /// Period standard deviation (PSD), in seconds — sample SD of the
    /// period sequence (Praat's PointProcess period SD). An ABI component.
    pub period_std_s: Seconds,
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
        period_std_s: Seconds::new(sample_std(&periods, mean_p)),
    })
}

/// Sample standard deviation (n−1 denominator, matching Praat) of a
/// sequence about a precomputed mean. Returns 0 for fewer than 2 values.
fn sample_std(xs: &[f64], mean: f64) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let ss: f64 = xs.iter().map(|&x| (x - mean) * (x - mean)).sum();
    (ss / (xs.len() - 1) as f64).sqrt()
}

/// Median voiced f0 over the signal (a sustained phonation has a stable
/// f0; the median is robust to a few unvoiced/edge frames).
fn estimate_f0(audio: &Audio, config: &PerturbationConfig) -> Option<f32> {
    median_voiced_f0(audio, config.pitch_floor_hz, config.pitch_ceiling_hz)
}

/// Median f0 (Hz) over voiced frames within `[floor, ceiling]`, or
/// `None` if no frame is voiced.
fn median_voiced_f0(audio: &Audio, floor_hz: f32, ceiling_hz: f32) -> Option<f32> {
    let cfg = PitchConfig {
        min_freq_hz: floor_hz,
        max_freq_hz: ceiling_hz,
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

/// Configuration for [`cpps`].
#[derive(Debug, Clone)]
pub struct CppsConfig {
    /// Cepstral-peak search lower bound, in Hz (quefrency `1/f`).
    pub pitch_floor_hz: f32,
    /// Cepstral-peak search upper bound, in Hz.
    pub pitch_ceiling_hz: f32,
    /// FFT / analysis-window length in samples (power of two).
    pub frame_size: usize,
    /// Frame advance, in seconds.
    pub hop_seconds: f32,
    /// Tilt-regression quefrency range, in seconds.
    pub tilt_quefrency_min_s: f32,
    /// Tilt-regression quefrency upper bound, in seconds.
    pub tilt_quefrency_max_s: f32,
    /// Quefrency smoothing window, in seconds (Praat's CPPS quefrency
    /// averaging — lowers the sharp peak toward the smoothed prominence).
    pub quefrency_smooth_s: f32,
}

impl Default for CppsConfig {
    fn default() -> Self {
        Self {
            pitch_floor_hz: 60.0,
            pitch_ceiling_hz: 330.0,
            frame_size: 4096,
            hop_seconds: 0.005,
            tilt_quefrency_min_s: 0.001,
            tilt_quefrency_max_s: 0.05,
            quefrency_smooth_s: 0.00015,
        }
    }
}

/// Smoothed cepstral peak prominence (dB) of a sustained phonation —
/// the prominence of the cepstral peak (at the f0 quefrency) above the
/// cepstrum's regression tilt line, averaged over frames. Praat's
/// `PowerCepstrogram` → `Get CPPS`.
///
/// The prominence (peak − tilt line) is invariant to the cepstrum's
/// overall scaling, so it's robust to FFT-normalization / log-base /
/// power-vs-magnitude conventions — only the cepstrum *shape* and the
/// regression matter. A robust (outlier-downweighted) tilt fit keeps the
/// peak itself from dragging the line up.
pub fn cpps(audio: &Audio, config: &CppsConfig) -> Result<Decibels> {
    use realfft::RealFftPlanner;
    use rustfft::num_complex::Complex;

    let sr = audio.sample_rate as f32;
    let x: Vec<f32> = audio.mono_samples().collect();
    let n = config.frame_size;
    if x.len() < n {
        return Err(EngineError::unreliable(
            "cpps",
            "signal shorter than one analysis frame",
        ));
    }
    let window = crate::dsp::windowing::hann(n);
    let mut planner = RealFftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);
    let mut frame = fwd.make_input_vec();
    let mut spec = fwd.make_output_vec();
    let mut logspec = inv.make_input_vec();
    let mut ceps = inv.make_output_vec();

    let q_peak_lo = (sr / config.pitch_ceiling_hz) as usize;
    let q_peak_hi = ((sr / config.pitch_floor_hz) as usize).min(n - 1);
    let q_tilt_lo = ((config.tilt_quefrency_min_s * sr) as usize).max(1);
    let q_tilt_hi = ((config.tilt_quefrency_max_s * sr) as usize).min(n / 2);
    let hop = ((config.hop_seconds * sr) as usize).max(1);
    if q_peak_lo >= q_peak_hi || q_tilt_lo >= q_tilt_hi {
        return Err(EngineError::unreliable(
            "cpps",
            "degenerate quefrency ranges",
        ));
    }

    let mut cpp_sum = 0.0_f64;
    let mut count = 0usize;
    let mut s = 0;
    while s + n <= x.len() {
        for i in 0..n {
            frame[i] = x[s + i] * window[i];
        }
        fwd.process(&mut frame, &mut spec)
            .expect("fft sized via make_*_vec");
        for (dst, c) in logspec.iter_mut().zip(spec.iter()) {
            *dst = Complex::new((c.norm_sqr() + 1e-12).ln(), 0.0);
        }
        inv.process(&mut logspec, &mut ceps)
            .expect("ifft sized via make_*_vec");

        // Quefrency-smooth the *power* cepstrum (Praat stores power, and
        // power-domain averaging lowers the peak ~10·log10 rather than the
        // harsher 20·log10 of magnitude smoothing), then dB. The dB offset
        // still cancels in peak − tilt.
        let qsmooth = ((config.quefrency_smooth_s * sr) as usize).max(1);
        let power: Vec<f64> = ceps.iter().map(|&c| (c as f64) * (c as f64)).collect();
        let smoothed = moving_average(&power, qsmooth);
        let ceps_db: Vec<f64> = smoothed
            .iter()
            .map(|&p| 10.0 * (p + 1e-12).log10())
            .collect();

        // Peak in the f0 quefrency band.
        let mut peak_q = q_peak_lo;
        let mut peak_v = ceps_db[q_peak_lo];
        for (q, &v) in ceps_db
            .iter()
            .enumerate()
            .take(q_peak_hi + 1)
            .skip(q_peak_lo)
        {
            if v > peak_v {
                peak_v = v;
                peak_q = q;
            }
        }
        // Robust (IRLS) straight-line tilt over the regression band.
        let (a, b) = robust_line(&ceps_db, q_tilt_lo, q_tilt_hi);
        let line_at_peak = a + b * peak_q as f64;
        cpp_sum += peak_v - line_at_peak;
        count += 1;
        s += hop;
    }
    if count == 0 {
        return Err(EngineError::unreliable("cpps", "no frames"));
    }
    Ok(Decibels::new((cpp_sum / count as f64) as f32))
}

/// Centered moving-average smoothing with a window of `width` samples.
fn moving_average(x: &[f64], width: usize) -> Vec<f64> {
    if width <= 1 {
        return x.to_vec();
    }
    let h = width / 2;
    let n = x.len();
    let mut out = vec![0.0; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let lo = i.saturating_sub(h);
        let hi = (i + h + 1).min(n);
        let win = &x[lo..hi];
        *slot = win.iter().sum::<f64>() / win.len() as f64;
    }
    out
}

/// Iteratively-reweighted least-squares straight-line fit of `y[lo..=hi]`
/// against the index, returning `(intercept, slope)`. Bisquare weights
/// downweight outliers (e.g. the cepstral peak) so the line tracks the
/// baseline tilt — matching Praat's "Robust" fit.
fn robust_line(y: &[f64], lo: usize, hi: usize) -> (f64, f64) {
    let xs: Vec<f64> = (lo..=hi).map(|q| q as f64).collect();
    let ys: &[f64] = &y[lo..=hi];
    let mut w = vec![1.0_f64; xs.len()];
    let (mut a, mut b) = (0.0, 0.0);
    for _ in 0..5 {
        let sw: f64 = w.iter().sum();
        let swx: f64 = w.iter().zip(&xs).map(|(w, x)| w * x).sum();
        let swy: f64 = w.iter().zip(ys).map(|(w, y)| w * y).sum();
        let swxx: f64 = w.iter().zip(&xs).map(|(w, x)| w * x * x).sum();
        let swxy: f64 = w
            .iter()
            .zip(xs.iter().zip(ys))
            .map(|(w, (x, y))| w * x * y)
            .sum();
        let denom = sw * swxx - swx * swx;
        if denom.abs() < 1e-12 {
            break;
        }
        b = (sw * swxy - swx * swy) / denom;
        a = (swy - b * swx) / sw;
        // Update bisquare weights from residuals.
        let res: Vec<f64> = xs.iter().zip(ys).map(|(x, y)| y - (a + b * x)).collect();
        let mut absr: Vec<f64> = res.iter().map(|r| r.abs()).collect();
        absr.sort_by(|p, q| p.partial_cmp(q).unwrap());
        let mad = absr[absr.len() / 2].max(1e-9);
        let c = 4.685 * 1.4826 * mad;
        for (wi, r) in w.iter_mut().zip(&res) {
            let u = r / c;
            *wi = if u.abs() < 1.0 {
                (1.0 - u * u).powi(2)
            } else {
                0.0
            };
        }
    }
    (a, b)
}

/// Acoustic Voice Quality Index (AVQI) **v03.01** — a single 0–10
/// dysphonia-severity score from a weighted combination of six measures
/// (Maryn et al. 2010; Barsties von Latoszek et al., v03.01).
///
/// **Clean-room** from the publications: the Phonanium AVQI Praat plugin
/// is proprietary and is *not* used as a model — only (later) as a
/// black-box oracle to confirm output. The coefficients here are the
/// published v03.01 form.
///
/// Inputs are unit-specific (the coefficients were fit to these units):
/// - `cpps` — smoothed cepstral peak prominence, dB
/// - `hnr` — harmonics-to-noise ratio, dB
/// - `shimmer_local_pct` — shimmer local as a **percent** (e.g. `2.77`)
/// - `shimmer_local_db` — shimmer local, dB
/// - `slope` — LTAS slope, dB
/// - `tilt` — LTAS trendline tilt, dB
///
/// Result is clamped to the published `[0, 10]` AVQI range.
///
/// ## Not yet reference-confirmed (pending the authors / Praat-script oracle)
/// - The v03.01 coefficients could not be byte-verified against a
///   *v03.01* worked example. The accessible worked examples (Maryn &
///   Weenink 2015) are script v02.02 and give different absolute values
///   for the same component vectors (their normal-voice example reads
///   2.76 there vs ~1.13 here) — a version-scaling difference, flagged.
/// - `slope` / `tilt` must be measured per AVQI's exact LTAS definitions
///   (tilt is a dB trendline value, not the dB/kHz `Ltas::tilt`); the
///   audio→AVQI wiring is deferred until those are confirmed.
pub fn avqi(
    cpps: f32,
    hnr: f32,
    shimmer_local_pct: f32,
    shimmer_local_db: f32,
    slope: f32,
    tilt: f32,
) -> f32 {
    let inner = 3.295 - 0.111 * cpps - 0.073 * hnr - 0.213 * shimmer_local_pct
        + 2.789 * shimmer_local_db
        - 0.032 * slope
        + 0.077 * tilt;
    (2.571 * inner).clamp(0.0, 10.0)
}

/// Configuration for [`h1_h2`].
#[derive(Debug, Clone)]
pub struct H1H2Config {
    /// Lowest f0 to consider, in Hz.
    pub pitch_floor_hz: f32,
    /// Highest f0 to consider, in Hz.
    pub pitch_ceiling_hz: f32,
    /// FFT / analysis-window length (power of two).
    pub frame_size: usize,
}

impl Default for H1H2Config {
    fn default() -> Self {
        Self {
            pitch_floor_hz: 75.0,
            pitch_ceiling_hz: 600.0,
            frame_size: 4096,
        }
    }
}

/// H1–H2: the level of the first harmonic (at f0) minus the second
/// (at 2·f0), in dB — a glottal-source / open-quotient correlate and an
/// ABI component. Per frame, the magnitude-spectrum peak near f0 and
/// near 2·f0 are located and `20·log10(A1/A2)` is averaged over frames.
/// **Uncorrected** (no formant correction). [`EngineError::Unreliable`]
/// if no voiced f0 is found or the signal is shorter than one frame.
pub fn h1_h2(audio: &Audio, config: &H1H2Config) -> Result<Decibels> {
    use realfft::RealFftPlanner;
    use rustfft::num_complex::Complex;

    let sr = audio.sample_rate as f32;
    let x: Vec<f32> = audio.mono_samples().collect();
    let n = config.frame_size;
    if x.len() < n {
        return Err(EngineError::unreliable(
            "h1_h2",
            "signal shorter than one analysis frame",
        ));
    }
    let f0 = median_voiced_f0(audio, config.pitch_floor_hz, config.pitch_ceiling_hz)
        .ok_or_else(|| EngineError::unreliable("h1_h2", "no voiced f0 detected"))?;

    let window = crate::dsp::windowing::hann(n);
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut input = fft.make_input_vec();
    let mut spec = fft.make_output_vec();

    // Peak magnitude within ±15% of a target harmonic frequency.
    let peak = |spec: &[Complex<f32>], f: f32| -> f32 {
        let k0 = ((f * 0.85) * n as f32 / sr).floor() as usize;
        let k1 = (((f * 1.15) * n as f32 / sr).ceil() as usize).min(spec.len() - 1);
        spec[k0..=k1.max(k0)]
            .iter()
            .fold(0.0_f32, |m, c| m.max(c.norm()))
    };

    let mut sum = 0.0_f64;
    let mut count = 0usize;
    let hop = n / 2;
    let mut s = 0;
    while s + n <= x.len() {
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = x[s + i] * window[i];
        }
        fft.process(&mut input, &mut spec)
            .expect("fft sized via make_*_vec");
        let a1 = peak(&spec, f0);
        let a2 = peak(&spec, 2.0 * f0);
        if a1 > 0.0 && a2 > 0.0 {
            sum += 20.0 * (a1 as f64 / a2 as f64).log10();
            count += 1;
        }
        s += hop;
    }
    if count == 0 {
        return Err(EngineError::unreliable("h1_h2", "no analyzable frames"));
    }
    Ok(Decibels::new((sum / count as f64) as f32))
}

/// Configuration for [`gne`].
#[derive(Debug, Clone)]
pub struct GneConfig {
    /// Analysis sample rate; the signal is band-limited and resampled to
    /// this rate before analysis (Michaelis et al.: 10 kHz).
    pub downsample_hz: u32,
    /// LPC order of the inverse (whitening) filter, at `downsample_hz`.
    pub lpc_order: usize,
    /// Bandpass bandwidth, Hz (Michaelis et al. optimal: 1000).
    pub bandwidth_hz: f32,
    /// Center-frequency step between adjacent bands, Hz (Michaelis et al.
    /// optimal: 300).
    pub fshift_hz: f32,
}

impl Default for GneConfig {
    fn default() -> Self {
        Self {
            downsample_hz: 10_000,
            lpc_order: 13,
            bandwidth_hz: 1000.0,
            fshift_hz: 300.0,
        }
    }
}

/// Glottal-to-Noise Excitation ratio (Michaelis, Gramß & Strube 1997): a
/// breathiness / turbulent-noise correlate in `[0, 1]` and an ABI
/// component. Pulsatile (glottal) excitation drives all frequency bands
/// synchronously → highly correlated band envelopes → GNE near 1;
/// turbulent noise excites bands independently → low correlation → GNE
/// toward 0.
///
/// Algorithm: band-limit + resample to `downsample_hz`, inverse-filter
/// (LPC whitening) to the excitation residual, take the Hilbert envelope
/// of each `bandwidth_hz` band stepped by `fshift_hz`, and return the
/// maximum correlation over band pairs whose centers differ by at least
/// half a bandwidth (non-overlapping passbands). Because the FFT bandpass
/// is zero-phase, synchronous excitation shows up at lag 0, so the
/// per-pair correlation is evaluated there.
///
/// [`EngineError::Unreliable`] if the signal is too short to form ≥2
/// bands.
pub fn gne(audio: &Audio, config: &GneConfig) -> Result<Ratio> {
    use rustfft::{FftPlanner, num_complex::Complex};

    let mono: Vec<f32> = audio.mono_samples().collect();
    let x = resample_to_hz(&mono, audio.sample_rate, config.downsample_hz);
    let fs = config.downsample_hz as f32;
    if x.len() <= config.lpc_order * 4 || x.len() < 256 {
        return Err(EngineError::unreliable("gne", "signal too short"));
    }

    // Inverse-filter (LPC whitening) → excitation residual e[n].
    let lpc = crate::dsp::lpc::lpc(
        &x,
        config.lpc_order,
        crate::dsp::lpc::LpcMethod::Autocorrelation,
    )
    .map_err(|_| EngineError::unreliable("gne", "LPC inverse filter failed"))?;
    let residual = inverse_filter(&x, &lpc.coeffs);

    // FFT the residual once; each band reuses it.
    let l = residual.len();
    let mut planner = FftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(l);
    let inv = planner.plan_fft_inverse(l);
    let mut spec: Vec<Complex<f32>> = residual.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fwd.process(&mut spec);

    // Band center frequencies, and their Hilbert envelopes.
    let half = config.bandwidth_hz / 2.0;
    let mut centers = Vec::new();
    let mut fc = half;
    while fc <= fs / 2.0 - half {
        centers.push(fc);
        fc += config.fshift_hz;
    }
    if centers.len() < 2 {
        return Err(EngineError::unreliable("gne", "too few frequency bands"));
    }
    let norm = 1.0 / l as f32;
    let envelopes: Vec<Vec<f32>> = centers
        .iter()
        .map(|&fc| {
            let klo = (((fc - half) * l as f32 / fs).round() as usize).max(1);
            let khi = (((fc + half) * l as f32 / fs).round() as usize).min(l / 2);
            // Analytic signal: keep this band's positive-freq bins (×2),
            // zero everything else; |IFFT| is the Hilbert envelope.
            let mut z = vec![Complex::<f32>::new(0.0, 0.0); l];
            for k in klo..=khi {
                z[k] = spec[k] * 2.0;
            }
            inv.process(&mut z);
            z.iter().map(|c| c.norm() * norm).collect()
        })
        .collect();

    // GNE = max correlation over non-overlapping band pairs.
    let mut gne = 0.0_f32;
    for i in 0..centers.len() {
        for j in (i + 1)..centers.len() {
            if (centers[i] - centers[j]).abs() >= half {
                gne = gne.max(zero_lag_corr(&envelopes[i], &envelopes[j]));
            }
        }
    }
    Ok(Ratio::new(gne))
}

/// Linear-prediction inverse filter: `e[n] = x[n] + Σ_k a_k·x[n−k]`,
/// where `coeffs = [a_1, …, a_p]`. Flattens (whitens) the spectrum,
/// recovering the source excitation.
fn inverse_filter(x: &[f32], coeffs: &[f32]) -> Vec<f32> {
    let mut e = vec![0.0_f32; x.len()];
    for n in 0..x.len() {
        let mut acc = x[n];
        for (k, &a) in coeffs.iter().enumerate() {
            if n > k {
                acc += a * x[n - k - 1];
            }
        }
        e[n] = acc;
    }
    e
}

/// Zero-lag normalized (Pearson) correlation of two equal-length signals.
fn zero_lag_corr(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let ma = a[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mb = b[..n].iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let (mut num, mut da, mut db) = (0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..n {
        let (xa, xb) = (a[i] as f64 - ma, b[i] as f64 - mb);
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    if da <= 0.0 || db <= 0.0 {
        return 0.0;
    }
    (num / (da.sqrt() * db.sqrt())) as f32
}

/// Band-limited resample of `signal` from `fs_in` to `fs_out` via the
/// frequency domain (à la `scipy.signal.resample`): forward FFT, copy the
/// in-band bins into a spectrum of the new length, inverse FFT. For
/// downsampling this folds in the anti-alias low-pass. Approximate at the
/// Nyquist bin — fine here since GNE uses bands strictly below it.
fn resample_to_hz(signal: &[f32], fs_in: u32, fs_out: u32) -> Vec<f32> {
    use rustfft::{FftPlanner, num_complex::Complex};

    if fs_in == fs_out || signal.len() < 2 {
        return signal.to_vec();
    }
    let n = signal.len();
    let m = ((n as u64 * fs_out as u64) / fs_in as u64) as usize;
    if m < 2 {
        return signal.to_vec();
    }
    let mut planner = FftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(m);
    let mut buf: Vec<Complex<f32>> = signal.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fwd.process(&mut buf);

    let mut out = vec![Complex::<f32>::new(0.0, 0.0); m];
    let half = n.min(m) / 2;
    out[..=half].copy_from_slice(&buf[..=half]); // positive freqs incl. DC
    for k in 1..=half {
        out[m - k] = buf[n - k]; // mirror negatives
    }
    inv.process(&mut out);
    let scale = 1.0 / n as f32;
    out.iter().map(|c| c.re * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avqi_formula_orders_normal_below_dysphonic() {
        // Component vectors from Maryn & Weenink 2015 Figs 1 (normal) &
        // 2 (dysphonic). The v03.01 formula gives different absolute
        // values than that paper's v02.02 figures (version difference),
        // but must order normal < dysphonic and stay in [0, 10]. The
        // ~1.13 / ~4.82 are this formula's own values (arithmetic
        // regression check), not the v02.02 figures' 2.76 / 5.92.
        let normal = avqi(14.50, 21.96, 2.77, 0.35, -24.73, -10.66);
        let dysphonic = avqi(8.57, 16.31, 7.80, 0.75, -31.51, -9.31);
        assert!((normal - 1.129).abs() < 0.02, "normal AVQI {normal}");
        assert!(
            (dysphonic - 4.821).abs() < 0.02,
            "dysphonic AVQI {dysphonic}"
        );
        assert!(normal < dysphonic);
        assert!((0.0..=10.0).contains(&normal));
    }

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
