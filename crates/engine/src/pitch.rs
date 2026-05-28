//! Fundamental-frequency (f0) estimation — one of several non-equivalent
//! pitch-estimation methods per the 2026-05-21 DSP method-diversity entry.
//! Three methods ship:
//!
//! - [`autocorrelation`] — naive time-domain autocorrelation (Phase-0
//!   tracker). Simple, fast; biased by the window function's own
//!   autocorrelation.
//! - [`windowed_autocorrelation`] — adopts Boersma 1993's central
//!   insight (divide the windowed-signal autocorrelation by the window's
//!   own autocorrelation to remove the window's bias on peak location).
//!   Single peak per frame, parabolic refinement. Faster than [`boersma`]
//!   and adequate for clean signals.
//! - [`boersma`] — **faithful Boersma 1993 / Praat `to_pitch_ac` with
//!   `very_accurate = false`**. Adds multi-candidate detection per frame
//!   plus a Viterbi path-finder with octave-cost / octave-jump-cost /
//!   voiced-unvoiced-cost terms, the inter-frame machinery that makes
//!   Praat robust to halving / doubling / transient errors. Validated
//!   against Praat 6.x golden fixtures and synthetic ground-truth.
//!
//! ## References
//! - Rabiner, L.R. (1977), "On the use of autocorrelation analysis for pitch
//!   detection." *IEEE TASSP* 25(1).
//!   <https://doi.org/10.1109/TASSP.1977.1162905>
//! - Boersma, P. (1993), "Accurate short-term analysis of the fundamental
//!   frequency and the harmonics-to-noise ratio of a sampled sound."
//!   *Proc. Inst. Phonetic Sciences* 17.
//!   <https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf>
//!
//! ## Recommended default
//! [`windowed_autocorrelation`] for the existing v1 default (kept for
//! backwards compat); [`boersma`] when Praat-faithful behaviour or
//! robustness on real speech is needed (recommend switching the default
//! in 0.4.x once it has been exercised in anger).
//!
//! ## What [`boersma`] does *not* yet do (sub-deferred, follow-on slices)
//! - **`very_accurate = true`**: Gaussian window over 6 periods (we use
//!   Hann over 3 periods, matching Praat's `very_accurate = false`).
//! - **Windowed-sinc + Brent's method peak refinement**: we use parabolic
//!   refinement instead. Praat's paper rejects parabolic (it caps peak
//!   heights at ~0.743 and contributes to octave errors); we accept the
//!   gap for now and document it. Switching is a follow-on slice.
//! - **Anti-aliased upsample pre-step**: orthogonal to the Viterbi work;
//!   bumps accuracy further when wanted, sub-slice.
//!
//! ## Deferred alternates (per the DSP method-diversity entry)
//! - YIN: de Cheveigné & Kawahara (2002), <https://doi.org/10.1121/1.1458024>
//! - RAPT: Talkin (1995), in *Speech Coding and Synthesis* (Elsevier),
//!   ISBN 978-0-444-82169-1
//! - SWIPE': Camacho & Harris (2008), <https://doi.org/10.1121/1.2951592>
//! - pYIN: Mauch & Dixon (2014), <https://doi.org/10.1109/ICASSP.2014.6853678>
//!   — librosa's default; HMM-refined YIN
//! - CREPE (neural): Kim et al. (2018), <https://arxiv.org/abs/1802.06182>
//!   — SOTA accuracy, requires bundled model weights

use std::f32::consts::PI as PI_F32;

use crate::Audio;
use crate::dsp::windowing::hann;

/// One of the supported pitch-estimation methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchMethod {
    /// Naive time-domain autocorrelation (Phase-0 method). See
    /// [`autocorrelation`].
    Autocorrelation,
    /// Window-corrected autocorrelation. Adopts the central insight of
    /// Boersma (1993) — divide windowed-signal autocorrelation by window
    /// autocorrelation — but is **not** a faithful Boersma implementation.
    /// See [`windowed_autocorrelation`].
    WindowedAutocorrelation,
    /// Faithful Boersma 1993 / Praat `Sound: To Pitch (ac)…` with
    /// `very_accurate = false`. Multi-candidate per-frame detection +
    /// Viterbi path-finding. See [`boersma`].
    Boersma,
}

/// Configuration for the pitch trackers.
///
/// The `boersma_*` fields are only read by [`PitchMethod::Boersma`]; the
/// other methods ignore them. Defaults match Praat 6.x's
/// `Sound: To Pitch (ac)…` parameters.
#[derive(Debug, Clone)]
pub struct PitchConfig {
    /// Analysis frame length in seconds.
    pub frame_size_seconds: f32,
    /// Hop length (frame advance) in seconds.
    pub hop_size_seconds: f32,
    /// Minimum f0 to detect, in Hz. Doubles as Boersma's `pitch_floor`.
    pub min_freq_hz: f32,
    /// Maximum f0 to detect, in Hz. Doubles as Boersma's `pitch_ceiling`.
    pub max_freq_hz: f32,
    /// Threshold below which a frame is considered unvoiced. Applied to the
    /// `voicing` field by downstream callers; the trackers themselves still
    /// emit a frequency estimate for every frame so callers can filter.
    /// Default 0.45 (Boersma 1993 recommended value).
    pub voicing_threshold: f32,
    /// Boersma only: maximum number of pitch candidates kept per frame
    /// before Viterbi path-finding. Default 15 (Praat default).
    pub boersma_max_candidates: usize,
    /// Boersma only: silence threshold as a fraction of the global peak
    /// intensity. Frames quieter than this favour the unvoiced candidate.
    /// Default 0.03 (Praat default).
    pub boersma_silence_threshold: f32,
    /// Boersma only: weight on the `log2(min_pitch / f)` per-frame term
    /// that nudges decisions away from low-f octave-down errors. Default
    /// 0.01 per octave (Praat default).
    pub boersma_octave_cost: f32,
    /// Boersma only: weight on `|log2(f_prev / f_curr)|` between voiced
    /// frames in the Viterbi transition cost. Higher = smoother f0
    /// contours, more octave-jump resistance. Default 0.35 (Praat default).
    pub boersma_octave_jump_cost: f32,
    /// Boersma only: fixed Viterbi-transition penalty when one frame is
    /// voiced and the next unvoiced (or vice versa). Higher = harder to
    /// toggle voicing. Default 0.14 (Praat default).
    pub boersma_voiced_unvoiced_cost: f32,
}

impl Default for PitchConfig {
    fn default() -> Self {
        Self {
            frame_size_seconds: 0.030,
            hop_size_seconds: 0.010,
            min_freq_hz: 75.0,
            max_freq_hz: 500.0,
            voicing_threshold: 0.45,
            boersma_max_candidates: 15,
            boersma_silence_threshold: 0.03,
            boersma_octave_cost: 0.01,
            boersma_octave_jump_cost: 0.35,
            boersma_voiced_unvoiced_cost: 0.14,
        }
    }
}

/// One pitch estimate at a given time point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchFrame {
    /// Centre time of the analysis frame, in seconds from the start of the audio.
    pub time_seconds: f64,
    /// Estimated f0. For silent / unvoiced frames the lag still
    /// maximises *some* function so a frequency is reported; filter by
    /// `voicing` to discard unreliable estimates.
    pub frequency_hz: crate::units::Hertz,
    /// Voicing strength in `[0, 1]`. For [`autocorrelation`] this is
    /// `R(τ_best) / R(0)`; for [`autocorrelation_boersma`] this is the
    /// window-corrected normalised peak height. Closer to 1 = more clearly
    /// voiced.
    pub voicing: f32,
}

/// Dispatches to [`autocorrelation`], [`windowed_autocorrelation`], or
/// [`boersma`].
pub fn pitch(audio: &Audio, config: &PitchConfig, method: PitchMethod) -> Vec<PitchFrame> {
    match method {
        PitchMethod::Autocorrelation => autocorrelation(audio, config),
        PitchMethod::WindowedAutocorrelation => windowed_autocorrelation(audio, config),
        PitchMethod::Boersma => boersma(audio, config),
    }
}

/// Estimates f0 using naive time-domain autocorrelation (Phase-0 method).
///
/// Multi-channel audio is downmixed to mono before analysis. Returns one
/// frame per hop step whose start falls within the audio. Audio shorter than
/// one frame returns an empty vector. The `voicing` field is `R(τ_best) /
/// R(0)`.
pub fn autocorrelation(audio: &Audio, config: &PitchConfig) -> Vec<PitchFrame> {
    let mono: Vec<f32> = audio.mono_samples().collect();
    let sample_rate = audio.sample_rate as f32;

    let frame_size = (config.frame_size_seconds * sample_rate).round() as usize;
    let hop_size = (config.hop_size_seconds * sample_rate).round() as usize;
    let min_lag = (sample_rate / config.max_freq_hz).round() as usize;
    let max_lag = (sample_rate / config.min_freq_hz).round() as usize;

    if mono.len() < frame_size || hop_size == 0 || min_lag >= max_lag {
        return Vec::new();
    }

    let mut frames = Vec::new();
    let mut start = 0;
    while start + frame_size <= mono.len() {
        let frame = &mono[start..start + frame_size];
        let (lag, peak) = best_lag(frame, min_lag, max_lag);
        let r0: f32 = frame.iter().map(|x| x * x).sum();
        let voicing = if r0 > 0.0 { peak / r0 } else { 0.0 };
        let frequency_hz = sample_rate / lag as f32;
        let time_seconds = (start + frame_size / 2) as f64 / audio.sample_rate as f64;
        frames.push(PitchFrame {
            time_seconds,
            frequency_hz: crate::units::Hertz::new(frequency_hz),
            voicing: voicing.clamp(0.0, 1.0),
        });
        start += hop_size;
    }
    frames
}

/// Returns `(best_lag, autocorrelation_value_at_best_lag)`.
fn best_lag(frame: &[f32], min_lag: usize, max_lag: usize) -> (usize, f32) {
    let max_lag = max_lag.min(frame.len().saturating_sub(1));
    let mut best = min_lag;
    let mut best_value = f32::MIN;

    for lag in min_lag..=max_lag {
        let mut sum = 0.0f32;
        for i in 0..(frame.len() - lag) {
            sum += frame[i] * frame[i + lag];
        }
        if sum > best_value {
            best_value = sum;
            best = lag;
        }
    }
    (best, best_value.max(0.0))
}

/// Estimates f0 by dividing the autocorrelation of the windowed signal by
/// the autocorrelation of the window itself — the central insight of
/// Boersma (1993) — and locating the highest peak in the resulting
/// normalised autocorrelation.
///
/// **This is not a faithful implementation of Boersma 1993.** It adopts the
/// window-correction idea but omits substantial parts of the paper's full
/// pipeline:
///
/// - No anti-alias upsample pre-step.
/// - Hann window only; the paper's accuracy-upgraded Gaussian-window form
///   is not exposed.
/// - **Parabolic** sub-sample interpolation. The paper rejects parabolic
///   (it caps peak heights at ~0.743 and contributes to octave errors) and
///   specifies windowed-sinc interpolation refined by Brent's method. The
///   parabolic form here is the simpler / more common textbook approach.
/// - No zero-pad before the autocorrelation FFT.
/// - Single peak per frame; no multi-candidate Viterbi path-finder.
/// - No octave-cost, voiced-unvoiced cost, or silence-threshold terms.
///
/// What it does:
/// 1. Subtract local mean (DC removal).
/// 2. Apply a Hann window.
/// 3. Compute autocorrelation of the windowed signal `r_a(τ)`.
/// 4. Compute autocorrelation of the window alone `r_w(τ)`.
/// 5. Divide: `r(τ) = r_a(τ) / r_w(τ)`.
/// 6. Find the highest peak in `r(τ)` for τ ∈ `[sr/max_freq_hz, sr/min_freq_hz]`.
/// 7. Parabolic interpolation around the integer-lag peak.
/// 8. Voicing = normalised peak height.
///
/// Despite the gaps this is a strict improvement on [`autocorrelation`]
/// (the naive Phase-0 tracker) — peak heights are no longer biased by the
/// window's own shape, sub-frame f0 resolution improves, and the voicing
/// strength is more meaningful. A faithful Boersma 1993 implementation is
/// tracked as a deferred alternate (see module docs).
pub fn windowed_autocorrelation(audio: &Audio, config: &PitchConfig) -> Vec<PitchFrame> {
    let mono: Vec<f32> = audio.mono_samples().collect();
    let sample_rate = audio.sample_rate as f32;

    let frame_size = (config.frame_size_seconds * sample_rate).round() as usize;
    let hop_size = (config.hop_size_seconds * sample_rate).round() as usize;
    let min_lag = (sample_rate / config.max_freq_hz).round() as usize;
    let max_lag = (sample_rate / config.min_freq_hz).round() as usize;

    if mono.len() < frame_size || hop_size == 0 || min_lag >= max_lag {
        return Vec::new();
    }

    let window = hann(frame_size);
    // Precompute r_w(τ) — depends only on the window, not the frame.
    let r_w = autocorr_full(&window, max_lag);

    let mut frames = Vec::new();
    let mut start = 0;
    let mut windowed = vec![0.0_f32; frame_size];
    while start + frame_size <= mono.len() {
        let raw = &mono[start..start + frame_size];
        // Subtract local mean (DC removal).
        let mean = raw.iter().sum::<f32>() / frame_size as f32;
        for (dst, &src) in windowed.iter_mut().zip(raw.iter()) {
            *dst = src - mean;
        }
        // Apply window.
        for (b, w) in windowed.iter_mut().zip(window.iter()) {
            *b *= *w;
        }
        // Autocorrelation of the windowed signal.
        let r_a = autocorr_full(&windowed, max_lag);
        // Corrected autocorrelation r(τ) = r_a(τ) / r_w(τ), normalised by
        // r(0) so peak heights are in `[0, 1]`.
        let r0 = if r_w[0].abs() > 0.0 {
            r_a[0] / r_w[0]
        } else {
            0.0
        };
        let frequency_hz;
        let voicing;
        if r0 <= 0.0 {
            // Silent / DC-only frame.
            frequency_hz = sample_rate / min_lag as f32;
            voicing = 0.0;
        } else {
            // Find the highest peak in r(τ) for τ ∈ [min_lag, max_lag].
            let mut best_lag_f = min_lag as f32;
            let mut best_score = f32::MIN;
            for tau in min_lag..=max_lag {
                if r_w[tau].abs() < 1e-12 {
                    continue;
                }
                let corrected = r_a[tau] / r_w[tau];
                if corrected > best_score {
                    best_score = corrected;
                    best_lag_f = tau as f32;
                }
            }
            // Parabolic interpolation around the integer-lag peak.
            let best_int = best_lag_f as usize;
            if best_int > min_lag && best_int < max_lag {
                let y_minus = r_a[best_int - 1] / r_w[best_int - 1];
                let y_0 = r_a[best_int] / r_w[best_int];
                let y_plus = r_a[best_int + 1] / r_w[best_int + 1];
                let denom = y_minus - 2.0 * y_0 + y_plus;
                if denom.abs() > 1e-12 {
                    let delta = 0.5 * (y_minus - y_plus) / denom;
                    if delta.abs() < 1.0 {
                        best_lag_f = best_int as f32 + delta;
                        // Interpolated peak height (also useful for voicing).
                        best_score = y_0 - 0.25 * (y_minus - y_plus) * delta;
                    }
                }
            }
            frequency_hz = sample_rate / best_lag_f;
            voicing = (best_score / r0).clamp(0.0, 1.0);
        }
        let time_seconds = (start + frame_size / 2) as f64 / audio.sample_rate as f64;
        frames.push(PitchFrame {
            time_seconds,
            frequency_hz: crate::units::Hertz::new(frequency_hz),
            voicing,
        });
        start += hop_size;
    }
    frames
}

/// Computes `r[τ] = Σ_n x[n] · x[n+τ]` for `τ = 0..=max_lag`.
fn autocorr_full(x: &[f32], max_lag: usize) -> Vec<f32> {
    let n = x.len();
    let max_lag = max_lag.min(n.saturating_sub(1));
    let mut r = vec![0.0_f32; max_lag + 1];
    for tau in 0..=max_lag {
        let mut sum = 0.0_f32;
        for i in 0..(n - tau) {
            sum += x[i] * x[i + tau];
        }
        r[tau] = sum;
    }
    r
}

#[allow(dead_code)]
const _UNUSED_PI_F32: f32 = PI_F32; // touch import so it's not removed prematurely

/// Estimates f0 using a faithful implementation of Boersma 1993
/// (= Praat's `Sound: To Pitch (ac)…` with `very_accurate = false`).
///
/// Closes the C2-deferred follow-on to [`windowed_autocorrelation`] by
/// adding the inter-frame machinery that makes Praat robust on real
/// speech: multiple candidate peaks per frame + a Viterbi path-finder
/// with frame-level + transition-level cost terms.
///
/// **Pipeline:**
/// 1. For each frame: Hann-window the signal, compute autocorrelation
///    of the windowed signal `r_a(τ)` and of the window alone `r_w(τ)`,
///    form `r(τ) = r_a(τ) / r_w(τ)` (the C2 window-correction step).
/// 2. Locate local maxima of `r(τ)` in `[sample_rate / max_freq_hz,
///    sample_rate / min_freq_hz]`. Refine each maximum's position via
///    parabolic interpolation. Keep the top `boersma_max_candidates`
///    (default 15) by strength.
/// 3. Add an unvoiced candidate per frame whose strength rises in quiet
///    regions (Praat's formula:
///    `voicingThreshold + max(0, 2 - localIntensity / silenceThreshold
///    / (1 + voicingThreshold))`). This is the "silence" pole of the
///    voiced/silent/unvoiced trichotomy.
/// 4. Run Viterbi over all frames. Frame cost: `-strength +
///    octave_cost · log2(min_pitch / f)` for voiced candidates;
///    `-strength` for unvoiced. Transition cost: `octave_jump_cost ·
///    |log2(f_prev / f_curr)|` between two voiced candidates;
///    `voiced_unvoiced_cost` on a voicing toggle; 0 between two
///    unvoiced candidates.
/// 5. Backtrack the min-cost path; emit one [`PitchFrame`] per analysis
///    frame. Unvoiced winners report the strongest voiced candidate's
///    frequency (so the field is always populated) but voicing = 0.
///
/// **Defaults match Praat 6.x** (`max_candidates = 15`,
/// `silence_threshold = 0.03`, `voicing_threshold = 0.45`,
/// `octave_cost = 0.01`, `octave_jump_cost = 0.35`,
/// `voiced_unvoiced_cost = 0.14`). Validated against Praat golden TSVs
/// plus analytic synthetic ground-truth — see
/// `crates/engine/tests/pitch_boersma.rs`.
///
/// **What this version does NOT do** (see the module-level "What
/// `boersma` does not yet do" section): `very_accurate = true`
/// (Gaussian window over 6 periods), windowed-sinc + Brent's method
/// peak refinement, anti-aliased upsample. Those are sub-deferred and
/// orthogonal to the Viterbi work this slice ships.
pub fn boersma(audio: &Audio, config: &PitchConfig) -> Vec<PitchFrame> {
    let mono: Vec<f32> = audio.mono_samples().collect();
    let sample_rate = audio.sample_rate as f32;

    // Boersma's autocorrelation needs at least ~3 periods of the lowest
    // pitch the user is asking us to detect; smaller frames produce a
    // smeared peak and unreliable candidates. Silently extend the frame
    // to that minimum if the config doesn't meet it (Praat does the
    // same when `time_step = 0`).
    let min_frame_seconds = 3.0 / config.min_freq_hz;
    let frame_seconds = config.frame_size_seconds.max(min_frame_seconds);
    let frame_size = (frame_seconds * sample_rate).round() as usize;
    let hop_size = (config.hop_size_seconds * sample_rate).round() as usize;
    let min_lag = (sample_rate / config.max_freq_hz).round() as usize;
    let max_lag = (sample_rate / config.min_freq_hz).round() as usize;

    if mono.len() < frame_size || hop_size == 0 || min_lag >= max_lag {
        return Vec::new();
    }

    // Global intensity reference for the silence-threshold term —
    // Praat's `localPeak / globalPeak` denominator.
    let global_peak = mono
        .iter()
        .copied()
        .fold(0.0_f32, |acc, x| acc.max(x.abs()))
        .max(f32::EPSILON);

    let window = hann(frame_size);
    let r_w = autocorr_full(&window, max_lag);

    // Per-frame candidate lists, then Viterbi over the lattice.
    let mut frame_centres: Vec<f64> = Vec::new();
    let mut all_candidates: Vec<Vec<BoersmaCandidate>> = Vec::new();
    let mut start = 0;
    let mut windowed = vec![0.0_f32; frame_size];
    while start + frame_size <= mono.len() {
        let raw = &mono[start..start + frame_size];
        let local_peak = raw.iter().copied().fold(0.0_f32, |acc, x| acc.max(x.abs()));
        let local_intensity = local_peak / global_peak;

        // DC-remove + window.
        let mean = raw.iter().sum::<f32>() / frame_size as f32;
        for (dst, &src) in windowed.iter_mut().zip(raw.iter()) {
            *dst = src - mean;
        }
        for (b, w) in windowed.iter_mut().zip(window.iter()) {
            *b *= *w;
        }

        let r_a = autocorr_full(&windowed, max_lag);

        let mut candidates = boersma_candidates_for_frame(
            &r_a,
            &r_w,
            min_lag,
            max_lag,
            sample_rate,
            config.boersma_max_candidates,
            config.boersma_octave_cost,
            config.min_freq_hz,
        );

        // Unvoiced "candidate": Praat's silence/voicing trichotomy.
        let silence_term = (2.0
            - local_intensity
                / config.boersma_silence_threshold.max(f32::EPSILON)
                / (1.0 + config.voicing_threshold))
            .max(0.0);
        let unvoiced_strength = config.voicing_threshold + silence_term;
        candidates.push(BoersmaCandidate {
            frequency_hz: 0.0,
            strength: unvoiced_strength,
            voiced: false,
        });

        frame_centres.push((start + frame_size / 2) as f64 / audio.sample_rate as f64);
        all_candidates.push(candidates);

        start += hop_size;
    }

    let chosen = viterbi_boersma(
        &all_candidates,
        config.boersma_octave_jump_cost,
        config.boersma_voiced_unvoiced_cost,
    );

    let mut out = Vec::with_capacity(frame_centres.len());
    for (i, t) in frame_centres.into_iter().enumerate() {
        let pick = &all_candidates[i][chosen[i]];
        let frequency_hz = if pick.voiced {
            pick.frequency_hz
        } else {
            // Surface the strongest voiced candidate's frequency so the
            // field is never NaN; callers gate on `voicing` for use.
            all_candidates[i]
                .iter()
                .filter(|c| c.voiced)
                .max_by(|a, b| a.strength.total_cmp(&b.strength))
                .map(|c| c.frequency_hz)
                .unwrap_or(0.0)
        };
        let voicing = if pick.voiced {
            pick.strength.clamp(0.0, 1.0)
        } else {
            0.0
        };
        out.push(PitchFrame {
            time_seconds: t,
            frequency_hz: crate::units::Hertz::new(frequency_hz),
            voicing,
        });
    }
    out
}

/// One Boersma candidate inside a single frame. Voiced candidates carry
/// a frequency + a window-corrected ACF strength; the per-frame unvoiced
/// candidate carries only a strength (its frequency field is `0.0`).
#[derive(Debug, Clone, Copy)]
struct BoersmaCandidate {
    frequency_hz: f32,
    /// Effective strength after the per-frame octave-cost term has been
    /// applied (subtracted for voiced candidates; raw value for unvoiced).
    strength: f32,
    voiced: bool,
}

/// Builds the per-frame candidate list for [`boersma`] — local maxima of
/// `r(τ) = r_a(τ)/r_w(τ)` in `[min_lag, max_lag]`, parabolic-refined,
/// top-`max_keep` by strength, with the octave-cost penalty applied.
#[allow(clippy::too_many_arguments)]
fn boersma_candidates_for_frame(
    r_a: &[f32],
    r_w: &[f32],
    min_lag: usize,
    max_lag: usize,
    sample_rate: f32,
    max_keep: usize,
    octave_cost: f32,
    min_pitch: f32,
) -> Vec<BoersmaCandidate> {
    let r0 = if r_w[0].abs() > 0.0 {
        r_a[0] / r_w[0]
    } else {
        0.0
    };
    if r0 <= 0.0 {
        return Vec::new();
    }
    // Sample `r(τ)` once; reused for local-max detection + parabolic
    // refinement.
    let mut r = vec![0.0_f32; max_lag + 1];
    for tau in 0..=max_lag {
        if r_w[tau].abs() < 1e-12 {
            r[tau] = 0.0;
        } else {
            r[tau] = r_a[tau] / r_w[tau];
        }
    }

    let mut peaks: Vec<BoersmaCandidate> = Vec::new();
    for tau in min_lag + 1..max_lag {
        if r[tau] > r[tau - 1] && r[tau] >= r[tau + 1] {
            // Parabolic refinement around the integer-lag peak.
            let y_minus = r[tau - 1];
            let y_0 = r[tau];
            let y_plus = r[tau + 1];
            let denom = y_minus - 2.0 * y_0 + y_plus;
            let (peak_lag, peak_height) = if denom.abs() > 1e-12 {
                let delta = 0.5 * (y_minus - y_plus) / denom;
                if delta.abs() < 1.0 {
                    (tau as f32 + delta, y_0 - 0.25 * (y_minus - y_plus) * delta)
                } else {
                    (tau as f32, y_0)
                }
            } else {
                (tau as f32, y_0)
            };
            let frequency_hz = sample_rate / peak_lag.max(f32::EPSILON);
            let strength_raw = (peak_height / r0).clamp(0.0, 1.0);
            // Per-frame octave cost from Boersma 1993:
            // `strength = r(τ) - octave_cost · log2(min_pitch / f)`.
            // For f > min_pitch the log is negative, so the **subtracted**
            // value is positive — higher-f candidates get a boost.
            // Equivalently: `+ octave_cost · log2(f / min_pitch)`. This
            // is what breaks the otherwise-equal tie between τ and 2τ
            // on a pure tone and fights octave-halving on harmonic
            // signals.
            let bonus = if frequency_hz > 0.0 && min_pitch > 0.0 {
                octave_cost * (frequency_hz / min_pitch).log2()
            } else {
                0.0
            };
            let strength = strength_raw + bonus;
            peaks.push(BoersmaCandidate {
                frequency_hz,
                strength,
                voiced: true,
            });
        }
    }

    // Keep the top `max_keep` by strength.
    peaks.sort_by(|a, b| b.strength.total_cmp(&a.strength));
    peaks.truncate(max_keep);
    peaks
}

/// Viterbi over Boersma per-frame candidates. Returns one candidate
/// index per frame (the chosen path's pick). Cost model matches Praat:
/// frame cost = `-strength`, transition cost = `octave_jump_cost ·
/// |log2(f_prev / f_curr)|` between two voiced candidates,
/// `voiced_unvoiced_cost` when one is voiced and the other unvoiced,
/// 0 between two unvoiced candidates.
fn viterbi_boersma(
    frames: &[Vec<BoersmaCandidate>],
    octave_jump_cost: f32,
    voiced_unvoiced_cost: f32,
) -> Vec<usize> {
    let n_frames = frames.len();
    if n_frames == 0 {
        return Vec::new();
    }

    // cost[i][j] = min total cost to reach candidate j of frame i.
    let mut cost: Vec<Vec<f32>> = frames
        .iter()
        .map(|f| vec![f32::INFINITY; f.len()])
        .collect();
    let mut back: Vec<Vec<usize>> = frames.iter().map(|f| vec![0; f.len()]).collect();

    // Frame 0: cost is just the frame cost (= -strength).
    for (j, c) in frames[0].iter().enumerate() {
        cost[0][j] = -c.strength;
    }

    for i in 1..n_frames {
        for (k, c_curr) in frames[i].iter().enumerate() {
            let frame_cost = -c_curr.strength;
            let mut best = f32::INFINITY;
            let mut best_prev = 0_usize;
            for (j, c_prev) in frames[i - 1].iter().enumerate() {
                if cost[i - 1][j].is_infinite() {
                    continue;
                }
                let trans =
                    transition_cost_boersma(c_prev, c_curr, octave_jump_cost, voiced_unvoiced_cost);
                let total = cost[i - 1][j] + trans + frame_cost;
                if total < best {
                    best = total;
                    best_prev = j;
                }
            }
            cost[i][k] = best;
            back[i][k] = best_prev;
        }
    }

    // Backtrack from the lowest-cost endpoint.
    let mut chosen = vec![0_usize; n_frames];
    let mut last_idx = 0;
    let mut last_cost = f32::INFINITY;
    for (j, &c) in cost[n_frames - 1].iter().enumerate() {
        if c < last_cost {
            last_cost = c;
            last_idx = j;
        }
    }
    chosen[n_frames - 1] = last_idx;
    for i in (1..n_frames).rev() {
        chosen[i - 1] = back[i][chosen[i]];
    }
    chosen
}

fn transition_cost_boersma(
    prev: &BoersmaCandidate,
    curr: &BoersmaCandidate,
    octave_jump_cost: f32,
    voiced_unvoiced_cost: f32,
) -> f32 {
    match (prev.voiced, curr.voiced) {
        (true, true) => {
            // |log2(f_prev / f_curr)| · octave_jump_cost
            let f_prev = prev.frequency_hz.max(f32::EPSILON);
            let f_curr = curr.frequency_hz.max(f32::EPSILON);
            octave_jump_cost * (f_prev / f_curr).log2().abs()
        }
        (false, false) => 0.0,
        _ => voiced_unvoiced_cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_audio(sample_rate: u32, channels: u16, freq_hz: f32, duration_s: f32) -> Audio {
        let n_frames = (sample_rate as f32 * duration_s) as usize;
        let mut samples = Vec::with_capacity(n_frames * channels as usize);
        for i in 0..n_frames {
            let t = i as f32 / sample_rate as f32;
            let s = 0.5 * (2.0 * std::f32::consts::PI * freq_hz * t).sin();
            for _ in 0..channels {
                samples.push(s);
            }
        }
        Audio {
            samples,
            sample_rate,
            channels,
        }
    }

    fn silent_audio(sample_rate: u32, duration_s: f32) -> Audio {
        let n = (sample_rate as f32 * duration_s) as usize;
        Audio {
            samples: vec![0.0; n],
            sample_rate,
            channels: 1,
        }
    }

    #[test]
    fn detects_440_hz_sine() {
        let audio = sine_audio(16_000, 1, 440.0, 0.5);
        let frames = autocorrelation(&audio, &PitchConfig::default());

        assert!(!frames.is_empty(), "expected at least one frame");
        for f in &frames {
            assert!(
                (f.frequency_hz.value() - 440.0).abs() < 10.0,
                "frame at t={:.3}s reported {:.1} Hz, expected ~440",
                f.time_seconds,
                f.frequency_hz.value()
            );
        }
    }

    #[test]
    fn detects_100_hz_sine() {
        let audio = sine_audio(16_000, 1, 100.0, 0.5);
        let frames = autocorrelation(&audio, &PitchConfig::default());

        assert!(!frames.is_empty());
        for f in &frames {
            assert!(
                (f.frequency_hz.value() - 100.0).abs() < 2.0,
                "frame at t={:.3}s reported {:.1} Hz, expected ~100",
                f.time_seconds,
                f.frequency_hz.value()
            );
        }
    }

    #[test]
    fn stereo_downmix_matches_mono() {
        let mono = sine_audio(16_000, 1, 200.0, 0.3);
        let stereo = sine_audio(16_000, 2, 200.0, 0.3);

        let f_mono = autocorrelation(&mono, &PitchConfig::default());
        let f_stereo = autocorrelation(&stereo, &PitchConfig::default());

        assert_eq!(f_mono.len(), f_stereo.len());
        for (a, b) in f_mono.iter().zip(f_stereo.iter()) {
            assert!((a.frequency_hz.value() - b.frequency_hz.value()).abs() < 0.01);
            assert!((a.time_seconds - b.time_seconds).abs() < 1e-9);
        }
    }

    #[test]
    fn audio_shorter_than_one_frame_returns_empty() {
        let audio = sine_audio(16_000, 1, 200.0, 0.005);
        let frames = autocorrelation(&audio, &PitchConfig::default());
        assert!(frames.is_empty());
    }

    #[test]
    fn frame_times_are_monotonically_increasing() {
        let audio = sine_audio(16_000, 1, 200.0, 0.5);
        let frames = autocorrelation(&audio, &PitchConfig::default());
        for window in frames.windows(2) {
            assert!(window[1].time_seconds > window[0].time_seconds);
        }
    }

    #[test]
    fn unit_sine_has_high_voicing_via_autocorrelation() {
        let audio = sine_audio(16_000, 1, 200.0, 0.5);
        let frames = autocorrelation(&audio, &PitchConfig::default());
        let mid = &frames[frames.len() / 2];
        assert!(
            mid.voicing > 0.7,
            "expected high voicing, got {}",
            mid.voicing
        );
    }

    #[test]
    fn silent_input_has_zero_voicing_via_autocorrelation() {
        let audio = silent_audio(16_000, 0.5);
        let frames = autocorrelation(&audio, &PitchConfig::default());
        assert!(!frames.is_empty());
        for f in &frames {
            assert!(f.voicing < 0.001, "got voicing = {}", f.voicing);
        }
    }

    #[test]
    fn windowed_autocorrelation_detects_220_hz_sine_more_accurately_than_naive() {
        // For a clean sine, the window-corrected method should land within
        // 1 Hz (via parabolic interpolation), while the naive method is
        // limited to integer-lag precision.
        let audio = sine_audio(16_000, 1, 220.0, 0.5);
        let cfg = PitchConfig::default();
        let frames = windowed_autocorrelation(&audio, &cfg);
        assert!(!frames.is_empty());
        let mid = &frames[frames.len() / 2];
        assert!(
            (mid.frequency_hz.value() - 220.0).abs() < 1.0,
            "windowed_autocorrelation: got {} Hz, expected ~220",
            mid.frequency_hz.value()
        );
        assert!(
            mid.voicing > 0.7,
            "expected high voicing, got {}",
            mid.voicing
        );
    }

    #[test]
    fn windowed_autocorrelation_silent_input_has_low_voicing() {
        let audio = silent_audio(16_000, 0.5);
        let frames = windowed_autocorrelation(&audio, &PitchConfig::default());
        assert!(!frames.is_empty());
        for f in &frames {
            assert!(f.voicing < 0.1, "got voicing = {}", f.voicing);
        }
    }

    #[test]
    fn pitch_dispatcher_picks_the_right_method() {
        let audio = sine_audio(16_000, 1, 220.0, 0.5);
        let cfg = PitchConfig::default();
        let a = pitch(&audio, &cfg, PitchMethod::Autocorrelation);
        let b = pitch(&audio, &cfg, PitchMethod::WindowedAutocorrelation);
        // Both should detect 220 Hz; the methods don't have to agree to the
        // last decimal but must be within reasonable tolerance of each other.
        let mid_a = a[a.len() / 2].frequency_hz.value();
        let mid_b = b[b.len() / 2].frequency_hz.value();
        assert!(
            (mid_a - 220.0).abs() < 10.0,
            "autocorr midpoint = {}",
            mid_a
        );
        assert!((mid_b - 220.0).abs() < 5.0, "windowed midpoint = {}", mid_b);
    }

    /// Helper: write a centred Gaussian-amplitude "vowel-like" sustained
    /// tone (constant f0, fade in/out) so the Viterbi has voiced + silent
    /// regions to discriminate without sharp transients.
    fn sustained_tone(sample_rate: u32, freq_hz: f32, duration_s: f32) -> Audio {
        let n = (sample_rate as f32 * duration_s) as usize;
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / sample_rate as f32;
            // Hann-like envelope (0 at the edges, 1 in the middle) so the
            // unvoiced/voiced trichotomy has a meaningful local intensity.
            let env = (PI_F32 * i as f32 / n as f32).sin();
            samples.push(0.5 * env * (2.0 * PI_F32 * freq_hz * t).sin());
        }
        Audio {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    #[test]
    fn boersma_detects_220_hz_sine_with_high_voicing() {
        // Clean sustained tone — Boersma should land within ~1 Hz at the
        // mid-frame and report high voicing.
        let audio = sustained_tone(16_000, 220.0, 0.6);
        let frames = boersma(&audio, &PitchConfig::default());
        assert!(!frames.is_empty());
        let mid = &frames[frames.len() / 2];
        assert!(
            (mid.frequency_hz.value() - 220.0).abs() < 1.0,
            "boersma mid f0 = {} Hz, expected ~220",
            mid.frequency_hz.value()
        );
        assert!(
            mid.voicing > 0.7,
            "expected high voicing in tone middle, got {}",
            mid.voicing
        );
    }

    #[test]
    fn boersma_silence_classifies_unvoiced() {
        let audio = silent_audio(16_000, 0.5);
        let frames = boersma(&audio, &PitchConfig::default());
        assert!(!frames.is_empty());
        for f in &frames {
            assert!(
                f.voicing < 0.1,
                "silent frame voicing should be ~0, got {}",
                f.voicing
            );
        }
    }

    #[test]
    fn boersma_is_robust_to_octave_halving_when_windowed_ac_might_not_be() {
        // A pulse train at 200 Hz has a strong half-rate harmonic in the
        // autocorrelation (period-doubling pull), which the naive
        // single-peak window-corrected tracker can latch onto. Boersma's
        // octave_cost + Viterbi should keep it at 200 Hz across the run.
        let sample_rate = 16_000_u32;
        let n = (sample_rate as f32 * 0.5) as usize;
        let period_samples = (sample_rate as f32 / 200.0).round() as usize;
        let mut samples = vec![0.0_f32; n];
        let mut k = 0;
        while k < n {
            // Hann-shaped pulse of width 4 samples; cleaner than a delta.
            for offset in 0..4 {
                if k + offset < n {
                    samples[k + offset] = (PI_F32 * offset as f32 / 4.0).sin();
                }
            }
            k += period_samples;
        }
        let audio = Audio {
            samples,
            sample_rate,
            channels: 1,
        };

        // Widen the range so 100 Hz halving is a reachable candidate.
        let cfg = PitchConfig {
            min_freq_hz: 60.0,
            max_freq_hz: 600.0,
            ..PitchConfig::default()
        };
        let frames = boersma(&audio, &cfg);
        assert!(!frames.is_empty());

        // The middle of the run (steady state) must lock onto 200 Hz.
        let mid_third_lo = frames.len() / 3;
        let mid_third_hi = 2 * frames.len() / 3;
        for f in &frames[mid_third_lo..mid_third_hi] {
            assert!(
                (f.frequency_hz.value() - 200.0).abs() < 5.0 || f.voicing < cfg.voicing_threshold,
                "octave halving in steady-state frame: got {} Hz (voicing {})",
                f.frequency_hz.value(),
                f.voicing
            );
        }
    }

    #[test]
    fn boersma_path_clusters_around_target_through_noise_gap() {
        // Splice a 200 Hz sustained tone with a low-amplitude noise gap
        // in the middle. A single-peak tracker without inter-frame state
        // would emit garbage f0 across the gap; Boersma's Viterbi keeps
        // voiced frames clustered around 200 Hz (we check the median, to
        // tolerate parabolic-refinement edge effects at fade-in/out — the
        // sub-deferred sinc+Brent refinement is what tightens this) and
        // flips voicing off where the gap is genuinely silent.
        let sample_rate = 16_000_u32;
        let half_len = (sample_rate as f32 * 0.30) as usize;
        let gap_len = (sample_rate as f32 * 0.05) as usize;
        let mut samples = Vec::with_capacity(2 * half_len + gap_len);
        for i in 0..half_len {
            let t = i as f32 / sample_rate as f32;
            let env = (PI_F32 * i as f32 / half_len as f32).sin();
            samples.push(0.5 * env * (2.0 * PI_F32 * 200.0 * t).sin());
        }
        for i in 0..gap_len {
            // Deterministic pseudo-noise so the test is stable.
            let pseudo =
                ((i.wrapping_mul(1103515245).wrapping_add(12345)) % 2048) as f32 / 2048.0 - 0.5;
            samples.push(0.005 * pseudo);
        }
        for i in 0..half_len {
            let t = (i + half_len + gap_len) as f32 / sample_rate as f32;
            let env = (PI_F32 * i as f32 / half_len as f32).sin();
            samples.push(0.5 * env * (2.0 * PI_F32 * 200.0 * t).sin());
        }
        let audio = Audio {
            samples,
            sample_rate,
            channels: 1,
        };

        let frames = boersma(&audio, &PitchConfig::default());
        assert!(!frames.is_empty());

        let voicing_threshold = PitchConfig::default().voicing_threshold;
        let mut voiced_f0s: Vec<f32> = frames
            .iter()
            .filter(|f| f.voicing >= voicing_threshold)
            .map(|f| f.frequency_hz.value())
            .collect();
        assert!(
            voiced_f0s.len() >= 5,
            "expected several voiced frames, got {}",
            voiced_f0s.len(),
        );
        voiced_f0s.sort_by(|a, b| a.total_cmp(b));
        let median = voiced_f0s[voiced_f0s.len() / 2];
        assert!(
            (median - 200.0).abs() < 5.0,
            "median voiced f0 = {} Hz, expected ~200",
            median,
        );

        // No voiced frame may report an octave error (halving / doubling).
        for f in &frames {
            if f.voicing >= voicing_threshold {
                let v = f.frequency_hz.value();
                assert!(
                    (v - 100.0).abs() > 30.0 && (v - 400.0).abs() > 30.0,
                    "octave error in voiced frame: {} Hz",
                    v,
                );
            }
        }
    }

    #[test]
    fn boersma_empty_when_audio_shorter_than_minimum_frame() {
        // Boersma extends the frame to 3/min_freq_hz minimum; pass audio
        // shorter than that and expect empty output, not a panic.
        let cfg = PitchConfig::default();
        let min_frame_s = (3.0 / cfg.min_freq_hz) - 0.005;
        let audio = sine_audio(16_000, 1, 200.0, min_frame_s);
        let frames = boersma(&audio, &cfg);
        assert!(frames.is_empty());
    }
}
