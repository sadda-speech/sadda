//! Fundamental-frequency (f0) estimation — five non-equivalent
//! pitch-estimation methods per the 2026-05-21 DSP method-diversity entry.
//! Two algorithmic families are covered (autocorrelation and
//! cumulative-mean-normalized-difference); having two independent
//! estimators is the cross-validation story for downstream work.
//!
//! **Autocorrelation family:**
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
//! **Cumulative-mean-normalized-difference family:**
//! - [`yin`] — de Cheveigné & Kawahara 2002. Difference function +
//!   CMNDF + absolute threshold. Simple baseline; the canonical
//!   non-autocorrelation tracker.
//! - [`pyin`] — Mauch & Dixon 2014, librosa's default. Probabilistic
//!   YIN with a beta-prior distribution over thresholds plus an HMM
//!   smoothing pass (semitone-distance transition + voicing-toggle
//!   cost). The modern Python-DSP audience expectation. Validated
//!   against librosa golden fixtures and synthetic ground-truth.
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
    /// de Cheveigné & Kawahara 2002 YIN — cumulative-mean-normalized-
    /// difference-function pitch tracker. Different algorithmic family
    /// from autocorrelation; provides an independent f0 estimate for
    /// cross-validation. See [`yin`].
    Yin,
    /// Mauch & Dixon 2014 pYIN — probabilistic YIN with a beta-prior
    /// distribution over thresholds plus an HMM smoothing pass over
    /// per-frame candidates. librosa's default; closest to the
    /// modern Python-DSP audience expectation. See [`pyin`].
    PYin,
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
    /// YIN / pYIN: absolute threshold on the cumulative mean normalized
    /// difference function (CMNDF). The chosen lag is the first dip
    /// below this threshold (or the global minimum if none drop low
    /// enough). Default 0.1 — the YIN paper's recommended value;
    /// matches librosa's `yin(trough_threshold=0.1)` default.
    pub yin_threshold: f32,
    /// pYIN: number of discrete thresholds drawn from the beta prior.
    /// Each threshold yields one candidate τ per frame; the prior
    /// weights are summed per unique τ to produce the per-frame
    /// probability distribution. Default 100 (matches librosa's pyin).
    pub pyin_n_thresholds: usize,
    /// pYIN: HMM transition penalty (cost per semitone) between two
    /// voiced bins. Higher = smoother f0 contour, more octave-jump
    /// resistance. Default 0.5 — chosen so a one-semitone step costs
    /// the same as the Boersma `octave_jump_cost` at one semitone.
    pub pyin_transition_semitone_cost: f32,
    /// pYIN: HMM cost of toggling between the voiced and unvoiced
    /// states between consecutive frames. Default 0.05 — librosa's
    /// pYIN keeps `switch_prob` small so the smoother prefers staying
    /// in the same voicing state.
    pub pyin_voiced_unvoiced_cost: f32,
    /// pYIN: number of semitone bins between [`PitchConfig::min_freq_hz`]
    /// and [`PitchConfig::max_freq_hz`] for the HMM state space.
    /// Default 20 bins per semitone (librosa's pyin default = `bins_per_semitone=12`
    /// times a 12-semitone band). A finer grid gives sharper Viterbi
    /// decoding at the cost of HMM runtime; the default is a moderate
    /// trade-off.
    pub pyin_bins_per_semitone: usize,
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
            yin_threshold: 0.1,
            pyin_n_thresholds: 100,
            pyin_transition_semitone_cost: 0.5,
            pyin_voiced_unvoiced_cost: 0.05,
            pyin_bins_per_semitone: 20,
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

/// Dispatches to one of the five pitch trackers.
pub fn pitch(audio: &Audio, config: &PitchConfig, method: PitchMethod) -> Vec<PitchFrame> {
    match method {
        PitchMethod::Autocorrelation => autocorrelation(audio, config),
        PitchMethod::WindowedAutocorrelation => windowed_autocorrelation(audio, config),
        PitchMethod::Boersma => boersma(audio, config),
        PitchMethod::Yin => yin(audio, config),
        PitchMethod::PYin => pyin(audio, config),
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

/// Estimates f0 using de Cheveigné & Kawahara 2002 YIN — the canonical
/// cumulative-mean-normalized-difference-function pitch tracker.
///
/// Different algorithmic family from the autocorrelation methods. For
/// downstream work, having an independent estimator means disagreement
/// between this and [`boersma`] is a confidence signal you can't get
/// from a single tracker alone.
///
/// **Pipeline** (per the paper's six steps):
/// 1. Difference function: `d[τ] = Σ_{i} (x[i] - x[i+τ])²` over the
///    frame, for `τ ∈ [1, max_lag]`.
/// 2. Cumulative mean normalized difference (CMNDF):
///    `d'[0] = 1`; `d'[τ] = d[τ] · τ / Σ_{j=1..τ} d[j]` for `τ ≥ 1`.
/// 3. Absolute threshold: find the smallest `τ ≥ min_lag` such that
///    `d'[τ] < yin_threshold` (default 0.1) **and** `d'[τ]` is a local
///    minimum (`d'[τ-1] > d'[τ] < d'[τ+1]`). If no such τ exists,
///    take the global argmin of `d'` in `[min_lag, max_lag]`.
/// 4. Parabolic refinement around the chosen τ.
/// 5. Best local estimate: the paper's step 5 (local refinement) folds
///    into the parabolic refinement above for typical signals; we omit
///    the explicit window-search since the parabolic step already
///    handles sub-sample peak placement to within a fraction of the
///    sample period.
/// 6. Voicing strength = `1 - d'[τ_chosen]`, clamped to `[0, 1]`.
///
/// `frame_size_seconds` is silently extended to `2 · max_lag / sr` so
/// the difference function always has at least one full period of the
/// lowest pitch to work with.
pub fn yin(audio: &Audio, config: &PitchConfig) -> Vec<PitchFrame> {
    let mono: Vec<f32> = audio.mono_samples().collect();
    let sample_rate = audio.sample_rate as f32;

    let min_lag = (sample_rate / config.max_freq_hz).round() as usize;
    let max_lag = (sample_rate / config.min_freq_hz).round() as usize;
    // YIN needs the frame to be at least 2 · max_lag (so the difference
    // function has a full second period to compare against).
    let min_frame_size = 2 * max_lag;
    let frame_size =
        ((config.frame_size_seconds * sample_rate).round() as usize).max(min_frame_size);
    let hop_size = (config.hop_size_seconds * sample_rate).round() as usize;

    if mono.len() < frame_size || hop_size == 0 || min_lag >= max_lag {
        return Vec::new();
    }
    let threshold = config.yin_threshold;

    let mut frames = Vec::new();
    let mut start = 0;
    while start + frame_size <= mono.len() {
        let frame = &mono[start..start + frame_size];
        let cmndf = yin_cmndf(frame, max_lag);
        let (tau_chosen, dprime_at_tau) = yin_pick_lag(&cmndf, min_lag, max_lag, threshold);

        let frequency_hz = sample_rate / tau_chosen.max(f32::EPSILON);
        let voicing = (1.0 - dprime_at_tau).clamp(0.0, 1.0);
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

/// Computes the YIN cumulative mean normalized difference function over
/// a frame, for lags `τ ∈ [0, max_lag]`. `cmndf[0] = 1` (the paper's
/// special case to avoid the division by zero at τ=0).
fn yin_cmndf(frame: &[f32], max_lag: usize) -> Vec<f32> {
    let n = frame.len();
    let max_lag = max_lag.min(n.saturating_sub(1));
    let mut diff = vec![0.0_f32; max_lag + 1];
    for tau in 1..=max_lag {
        let mut sum = 0.0_f32;
        for i in 0..(n - tau) {
            let d = frame[i] - frame[i + tau];
            sum += d * d;
        }
        diff[tau] = sum;
    }

    // CMNDF: d'[τ] = d[τ] · τ / Σ_{j=1..τ} d[j].
    let mut cmndf = vec![0.0_f32; max_lag + 1];
    cmndf[0] = 1.0;
    let mut running = 0.0_f32;
    for tau in 1..=max_lag {
        running += diff[tau];
        if running > 0.0 {
            cmndf[tau] = diff[tau] * tau as f32 / running;
        } else {
            cmndf[tau] = 1.0;
        }
    }
    cmndf
}

/// Picks the YIN lag from a CMNDF: smallest τ ≥ `min_lag` such that
/// `cmndf[τ] < threshold` AND `cmndf[τ]` is a local minimum. Falls back
/// to the global argmin if no candidate clears the threshold. Returns
/// `(refined_lag, refined_cmndf_value)`. The refined lag is parabolic-
/// interpolated around the integer-lag pick.
fn yin_pick_lag(cmndf: &[f32], min_lag: usize, max_lag: usize, threshold: f32) -> (f32, f32) {
    let upper = max_lag.min(cmndf.len().saturating_sub(2));
    let mut chosen: Option<usize> = None;
    for tau in min_lag.max(1)..upper {
        if cmndf[tau] < threshold && cmndf[tau] < cmndf[tau - 1] && cmndf[tau] <= cmndf[tau + 1] {
            chosen = Some(tau);
            break;
        }
    }
    let tau_int = chosen.unwrap_or_else(|| {
        // Fallback: global argmin of cmndf in [min_lag, max_lag].
        let lo = min_lag.max(1);
        let mut best = lo;
        let mut best_val = cmndf[lo];
        for (tau, &v) in cmndf.iter().enumerate().take(upper + 1).skip(lo + 1) {
            if v < best_val {
                best_val = v;
                best = tau;
            }
        }
        best
    });

    // Parabolic refinement around tau_int. Avoid the array edges.
    if tau_int <= min_lag.max(1) || tau_int + 1 > upper {
        return (tau_int as f32, cmndf[tau_int]);
    }
    let y_minus = cmndf[tau_int - 1];
    let y_0 = cmndf[tau_int];
    let y_plus = cmndf[tau_int + 1];
    let denom = y_minus - 2.0 * y_0 + y_plus;
    if denom.abs() < 1e-12 {
        return (tau_int as f32, y_0);
    }
    let delta = 0.5 * (y_minus - y_plus) / denom;
    if delta.abs() >= 1.0 {
        return (tau_int as f32, y_0);
    }
    let refined_lag = tau_int as f32 + delta;
    let refined_val = y_0 - 0.25 * (y_minus - y_plus) * delta;
    (refined_lag, refined_val)
}

/// Estimates f0 using Mauch & Dixon 2014 pYIN — probabilistic YIN with
/// HMM smoothing. librosa's default; the modern Python-DSP audience
/// expectation.
///
/// **Pipeline:**
/// 1. Per frame: compute CMNDF (same as YIN).
/// 2. **Threshold distribution**: instead of YIN's single absolute
///    threshold, integrate the threshold rule against a beta(α=2, β=18)
///    prior discretized to `pyin_n_thresholds` (default 100) values.
///    For each threshold draw, pick the YIN lag; weight that pick by
///    the beta prior PDF at the threshold. Sum the weights per unique
///    lag → per-frame distribution `{(τ_k, p_k)}`.
/// 3. **HMM state space**: 2 × `n_bins` states — `n_bins` log-spaced
///    semitone-grid voiced bins between [`PitchConfig::min_freq_hz`]
///    and [`PitchConfig::max_freq_hz`], paired with `n_bins` parallel
///    unvoiced states. `n_bins` ≈ `12 · log2(max/min) · pyin_bins_per_semitone`.
/// 4. **Emission**: each frame's voiced probabilities go to the bins
///    their τ_k maps to; the residual `1 - Σp_k` mass spreads uniformly
///    across the unvoiced states.
/// 5. **Transition matrix**: factorized into independent voicing- and
///    frequency-transition components. Voicing toggle costs
///    [`PitchConfig::pyin_voiced_unvoiced_cost`] (read as a switch
///    probability, not a Boersma-style additive cost). Frequency
///    transitions are Gaussian on semitone distance, σ controlled by
///    [`PitchConfig::pyin_transition_semitone_cost`] (read as a
///    semitone σ for the Gaussian — smaller σ = smoother f0 contour).
/// 6. **Viterbi decode** (in log-space to avoid underflow) → MAP path
///    of (voicing, bin) per frame. Voicing strength reported = the
///    summed voiced probability at the chosen bin in the emission row.
///
/// Defaults match librosa.pyin's recommended values:
/// `n_thresholds = 100`, `pyin_voiced_unvoiced_cost = 0.05` (switch
/// probability), `pyin_transition_semitone_cost = 0.5` (Gaussian σ in
/// semitones), `pyin_bins_per_semitone = 20`.
pub fn pyin(audio: &Audio, config: &PitchConfig) -> Vec<PitchFrame> {
    let mono: Vec<f32> = audio.mono_samples().collect();
    let sample_rate = audio.sample_rate as f32;

    let min_lag = (sample_rate / config.max_freq_hz).round() as usize;
    let max_lag = (sample_rate / config.min_freq_hz).round() as usize;
    let min_frame_size = 2 * max_lag;
    let frame_size =
        ((config.frame_size_seconds * sample_rate).round() as usize).max(min_frame_size);
    let hop_size = (config.hop_size_seconds * sample_rate).round() as usize;

    if mono.len() < frame_size || hop_size == 0 || min_lag >= max_lag {
        return Vec::new();
    }

    // ---------- Threshold distribution (beta prior) ----------
    let n_thresholds = config.pyin_n_thresholds.max(1);
    let mut thresholds = Vec::with_capacity(n_thresholds);
    let mut weights = Vec::with_capacity(n_thresholds);
    let mut weight_sum = 0.0_f32;
    // Sample thresholds at midpoints of equal-width bins on (0, 1).
    for i in 0..n_thresholds {
        let x = (i as f32 + 0.5) / n_thresholds as f32;
        let w = beta_pdf(x, 2.0, 18.0);
        thresholds.push(x);
        weights.push(w);
        weight_sum += w;
    }
    if weight_sum > 0.0 {
        for w in &mut weights {
            *w /= weight_sum;
        }
    }

    // ---------- HMM state-space grid ----------
    let f_lo = config.min_freq_hz;
    let f_hi = config.max_freq_hz;
    let log_lo = f_lo.log2();
    let log_hi = f_hi.log2();
    let n_semitones = ((log_hi - log_lo) * 12.0).round() as usize;
    let bins_per_semitone = config.pyin_bins_per_semitone.max(1);
    let n_bins = (n_semitones * bins_per_semitone).max(2);
    // bin_idx → semitone-grid frequency.
    let bin_to_hz = |b: usize| -> f32 {
        let log_f = log_lo + (log_hi - log_lo) * (b as f32 / (n_bins - 1) as f32);
        2.0_f32.powf(log_f)
    };
    let hz_to_bin = |f: f32| -> Option<usize> {
        if !f.is_finite() || f <= 0.0 {
            return None;
        }
        let lf = f.log2();
        if lf < log_lo || lf > log_hi {
            return None;
        }
        let rel = (lf - log_lo) / (log_hi - log_lo);
        Some(((rel * (n_bins - 1) as f32).round() as usize).min(n_bins - 1))
    };

    // ---------- Per-frame emissions ----------
    let mut emissions: Vec<Vec<f32>> = Vec::new();
    let mut frame_centres: Vec<f64> = Vec::new();
    let mut start = 0;
    while start + frame_size <= mono.len() {
        let frame = &mono[start..start + frame_size];
        let cmndf = yin_cmndf(frame, max_lag);

        // For each threshold draw, pick the YIN lag. If CMNDF actually
        // dipped below the threshold, the weight goes to the
        // corresponding voiced bin; if `yin_pick_lag` fell back to the
        // global argmin (no real dip), the weight flows to the
        // unvoiced pool — this is what makes silent or noisy frames
        // correctly classify unvoiced. (Mauch & Dixon 2014 §3.1.)
        let mut bin_probs = vec![0.0_f32; n_bins];
        let mut unvoiced_pool = 0.0_f32;
        for (i, &th) in thresholds.iter().enumerate() {
            let (tau, val) = yin_pick_lag(&cmndf, min_lag, max_lag, th);
            if val < th {
                let f = sample_rate / tau.max(f32::EPSILON);
                if let Some(b) = hz_to_bin(f) {
                    bin_probs[b] += weights[i];
                } else {
                    unvoiced_pool += weights[i];
                }
            } else {
                unvoiced_pool += weights[i];
            }
        }
        let voiced_total: f32 = bin_probs.iter().sum();
        let unvoiced_total =
            (unvoiced_pool + (1.0 - voiced_total - unvoiced_pool).max(0.0)).min(1.0);

        // Voiced emissions first, then unvoiced. Distribute unvoiced
        // mass uniformly across the unvoiced layer.
        let mut emission = Vec::with_capacity(2 * n_bins);
        emission.extend_from_slice(&bin_probs);
        let unvoiced_each = unvoiced_total / n_bins as f32;
        emission.extend(std::iter::repeat_n(unvoiced_each, n_bins));
        emissions.push(emission);
        frame_centres.push((start + frame_size / 2) as f64 / audio.sample_rate as f64);

        start += hop_size;
    }

    if emissions.is_empty() {
        return Vec::new();
    }

    // ---------- Viterbi (log-space) ----------
    let n_states = 2 * n_bins;
    let switch_prob = config.pyin_voiced_unvoiced_cost.clamp(1e-6, 0.5);
    let sigma_bins = config.pyin_transition_semitone_cost * bins_per_semitone as f32;
    // Precompute the per-bin Gaussian "frequency-transition kernel"
    // (centred at bin 0); slide it for each source bin. Normalize so
    // each row sums to 1 over the [0, n_bins) range.
    let mut kernel = vec![0.0_f32; n_bins];
    let two_sigma_sq = 2.0 * sigma_bins * sigma_bins;
    let mut kernel_sum = 0.0_f32;
    for (k, slot) in kernel.iter_mut().enumerate() {
        let d = k as f32;
        let v = (-d * d / two_sigma_sq).exp();
        *slot = v;
        kernel_sum += v;
    }
    // The kernel is symmetric (slide-against-source); per-row sums of
    // the resulting truncated Gaussian aren't all the same, but using
    // the centred sum keeps the math fast and the bias small at the
    // grid interior. Edge bins lose a little mass to truncation — a
    // known approximation.
    let kernel_norm = if kernel_sum > 0.0 { kernel_sum } else { 1.0 };

    let ln_eps = (1e-30_f32).ln();
    let ln_voiced_stay = (1.0 - switch_prob).ln();
    let ln_voicing_switch = switch_prob.ln();

    // cost[t][s] = max-log-prob to reach state s at frame t.
    let mut cost: Vec<Vec<f32>> = (0..emissions.len())
        .map(|_| vec![f32::NEG_INFINITY; n_states])
        .collect();
    let mut back: Vec<Vec<usize>> = (0..emissions.len())
        .map(|_| vec![0_usize; n_states])
        .collect();

    // Initial frame: log emission + uniform initial prior.
    let init_log = -(n_states as f32).ln();
    for s in 0..n_states {
        let e = emissions[0][s];
        let ln_e = if e > 0.0 { e.ln() } else { ln_eps };
        cost[0][s] = init_log + ln_e;
    }

    for t in 1..emissions.len() {
        // Cache per-source-bin: max(cost[t-1][voiced_b] + ln_kernel(b, b'))
        // over each b' destination. Rather than the full O(n_bins²) inner
        // loop, we compute the best source bin for each destination
        // bin by sliding the Gaussian kernel.
        // For correctness in this first cut we do the straightforward
        // O(n_bins²) — n_bins ≈ 660 by default; an 1.3M ops per frame
        // × ~100 frames ≈ 130M ops, ~50ms on a modern core. Fine.
        for b_to in 0..n_bins {
            let mut best_voiced = f32::NEG_INFINITY;
            let mut best_voiced_idx = 0_usize;
            let mut best_unvoiced = f32::NEG_INFINITY;
            let mut best_unvoiced_idx = 0_usize;
            for b_from in 0..n_bins {
                let d = (b_to as i32 - b_from as i32).unsigned_abs() as usize;
                let kvec = if d < kernel.len() { kernel[d] } else { 0.0 };
                let ln_freq = if kvec > 0.0 {
                    (kvec / kernel_norm).ln()
                } else {
                    ln_eps
                };
                // Source voiced.
                let c_v = cost[t - 1][b_from] + ln_voiced_stay + ln_freq;
                if c_v > best_voiced {
                    best_voiced = c_v;
                    best_voiced_idx = b_from;
                }
                // Source unvoiced (state = n_bins + b_from).
                let c_u = cost[t - 1][n_bins + b_from] + ln_voicing_switch + ln_freq;
                if c_u > best_voiced {
                    best_voiced = c_u;
                    best_voiced_idx = n_bins + b_from;
                }
                // Same for destination unvoiced (b_to + n_bins). Per
                // factorized model: P(U | U) = 1 - switch; P(U | V) =
                // switch.
                let cu_from_v = cost[t - 1][b_from] + ln_voicing_switch + ln_freq;
                let cu_from_u = cost[t - 1][n_bins + b_from] + ln_voiced_stay + ln_freq;
                if cu_from_v > best_unvoiced {
                    best_unvoiced = cu_from_v;
                    best_unvoiced_idx = b_from;
                }
                if cu_from_u > best_unvoiced {
                    best_unvoiced = cu_from_u;
                    best_unvoiced_idx = n_bins + b_from;
                }
            }
            let e_v = emissions[t][b_to];
            let ln_ev = if e_v > 0.0 { e_v.ln() } else { ln_eps };
            cost[t][b_to] = best_voiced + ln_ev;
            back[t][b_to] = best_voiced_idx;

            let e_u = emissions[t][n_bins + b_to];
            let ln_eu = if e_u > 0.0 { e_u.ln() } else { ln_eps };
            cost[t][n_bins + b_to] = best_unvoiced + ln_eu;
            back[t][n_bins + b_to] = best_unvoiced_idx;
        }
    }

    // Backtrack.
    let last_t = emissions.len() - 1;
    let mut path = vec![0_usize; emissions.len()];
    let mut best_state = 0_usize;
    let mut best_cost = f32::NEG_INFINITY;
    for (s, &c) in cost[last_t].iter().enumerate() {
        if c > best_cost {
            best_cost = c;
            best_state = s;
        }
    }
    path[last_t] = best_state;
    for t in (1..emissions.len()).rev() {
        path[t - 1] = back[t][path[t]];
    }

    // Convert path → PitchFrames.
    let mut out = Vec::with_capacity(emissions.len());
    for (t, &state) in path.iter().enumerate() {
        let (bin, voiced) = if state < n_bins {
            (state, true)
        } else {
            (state - n_bins, false)
        };
        let f = bin_to_hz(bin);
        // Voicing strength = voiced emission probability at the chosen
        // bin (the per-frame "how strongly does the threshold
        // distribution support this bin as a voiced pick"). The HMM
        // path may have selected an unvoiced state; in that case
        // report the voiced-prob-at-this-bin so callers have a usable
        // number, but voicing < threshold will mark it unvoiced in
        // downstream gates.
        let v_strength = emissions[t][bin].clamp(0.0, 1.0);
        let voicing = if voiced { v_strength } else { 0.0 };
        out.push(PitchFrame {
            time_seconds: frame_centres[t],
            frequency_hz: crate::units::Hertz::new(f),
            voicing,
        });
    }
    out
}

/// Beta(α, β) probability density at `x ∈ (0, 1)`. The Beta function
/// `B(α, β)` is computed via Stirling-corrected log-gammas; we only
/// need the ratio so any constant scaling factor cancels at
/// normalization time. For pYIN α=2 / β=18 specifically the formula
/// reduces to `x · (1 - x)^17 · 19 · 18` (exact analytic form), which
/// we use to avoid the lgamma dependency.
fn beta_pdf(x: f32, alpha: f32, beta: f32) -> f32 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    // Specialized for the pYIN-canonical α=2, β=18.
    if (alpha - 2.0).abs() < 1e-6 && (beta - 18.0).abs() < 1e-6 {
        // B(2, 18) = 1! · 17! / 19! = 1 / (19 · 18) = 1/342.
        // pdf = x · (1-x)^17 / B(2, 18) = 342 · x · (1-x)^17.
        return 342.0 * x * (1.0 - x).powi(17);
    }
    // Generic form via the log-gamma function.
    let ln_beta = ln_gamma(alpha) + ln_gamma(beta) - ln_gamma(alpha + beta);
    ((alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln() - ln_beta).exp()
}

/// Stirling approximation to ln Γ(x) for x > 0. Accurate to ~1e-4 for
/// `x ≥ 1`. Only used by [`beta_pdf`]'s generic branch when α or β
/// don't match the pYIN-canonical pair; not on the hot path.
fn ln_gamma(x: f32) -> f32 {
    // Lanczos with g=7, n=9 coefficients.
    #[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)]
    let coefs: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x as f64;
    let mut a = coefs[0];
    let t = x + 7.0 - 0.5;
    for (i, &c) in coefs.iter().enumerate().skip(1) {
        a += c / (x + i as f64 - 1.0);
    }
    let ln = 0.5 * (2.0_f64 * std::f64::consts::PI).ln() + (x - 0.5) * t.ln() - t + a.ln();
    ln as f32
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

    // ----- YIN -----

    #[test]
    fn yin_detects_220_hz_sine_with_high_voicing() {
        let audio = sustained_tone(16_000, 220.0, 0.6);
        let frames = yin(&audio, &PitchConfig::default());
        assert!(!frames.is_empty());
        let mid = &frames[frames.len() / 2];
        assert!(
            (mid.frequency_hz.value() - 220.0).abs() < 1.0,
            "yin mid f0 = {} Hz, expected ~220",
            mid.frequency_hz.value()
        );
        assert!(
            mid.voicing > 0.8,
            "expected high voicing (CMNDF dip should be near 0), got {}",
            mid.voicing,
        );
    }

    #[test]
    fn yin_silent_input_has_low_voicing() {
        let audio = silent_audio(16_000, 0.5);
        let frames = yin(&audio, &PitchConfig::default());
        assert!(!frames.is_empty());
        for f in &frames {
            assert!(
                f.voicing < 0.1,
                "silent frame voicing should be ~0, got {}",
                f.voicing,
            );
        }
    }

    #[test]
    fn yin_and_boersma_agree_on_clean_tone() {
        // Two independent algorithmic families on the same clean signal
        // should land at the same f0 within sub-Hz tolerance — this is
        // the cross-validation story (disagreement = real low confidence).
        let audio = sustained_tone(16_000, 175.0, 0.6);
        let cfg = PitchConfig::default();
        let yin_frames = yin(&audio, &cfg);
        let boersma_frames = boersma(&audio, &cfg);
        let mid_yin = yin_frames[yin_frames.len() / 2].frequency_hz.value();
        let mid_boersma = boersma_frames[boersma_frames.len() / 2]
            .frequency_hz
            .value();
        assert!(
            (mid_yin - mid_boersma).abs() < 1.0,
            "two independent trackers disagree on clean 175 Hz tone: \
             yin = {} Hz, boersma = {} Hz",
            mid_yin,
            mid_boersma,
        );
    }

    #[test]
    fn yin_cmndf_first_dip_below_threshold_picks_lag() {
        // Constructed CMNDF: 1.0, 0.9, 0.05, 0.5, 0.02, 0.6
        // min_lag=1, max_lag=5, threshold=0.1
        // First τ where cmndf[τ] < 0.1 AND local min: τ=2 (0.05 < 0.1,
        // and 0.05 < 0.9 + 0.05 < 0.5).
        let cmndf = vec![1.0_f32, 0.9, 0.05, 0.5, 0.02, 0.6];
        let (lag, val) = yin_pick_lag(&cmndf, 1, 5, 0.1);
        // Parabolic refinement may shift slightly but rounded lag = 2.
        assert!((lag - 2.0).abs() < 0.5, "lag = {lag}");
        assert!(val < 0.1, "picked val should be near the dip, got {val}");
    }

    #[test]
    fn yin_cmndf_falls_back_to_argmin_when_no_dip_clears_threshold() {
        // CMNDF: 1.0, 0.9, 0.4, 0.2, 0.3, 0.5
        // threshold=0.1; no value clears. Argmin in [1, 5] is τ=3 (0.2).
        let cmndf = vec![1.0_f32, 0.9, 0.4, 0.2, 0.3, 0.5];
        let (lag, _val) = yin_pick_lag(&cmndf, 1, 5, 0.1);
        assert!(
            (lag - 3.0).abs() < 0.5,
            "fallback argmin should be τ=3, got {lag}"
        );
    }

    #[test]
    fn yin_empty_when_audio_shorter_than_two_periods() {
        let cfg = PitchConfig::default();
        // 2 · max_lag at 16 kHz with min_freq_hz=75 is ≈ 0.0267s.
        let audio = sine_audio(16_000, 1, 200.0, 0.015);
        let frames = yin(&audio, &cfg);
        assert!(frames.is_empty());
    }

    // ----- pYIN -----

    #[test]
    fn pyin_detects_220_hz_sine_with_high_voicing() {
        // pYIN is grid-quantized to log-spaced bins, so a "perfect"
        // 220 Hz tone may land on a bin slightly off the true f0. Allow
        // a couple of Hz for that grid resolution (default 20
        // bins/semitone ≈ 0.6 Hz at 220 Hz — well within 2 Hz).
        let audio = sustained_tone(16_000, 220.0, 0.6);
        let frames = pyin(&audio, &PitchConfig::default());
        assert!(!frames.is_empty());
        let mid = &frames[frames.len() / 2];
        assert!(
            (mid.frequency_hz.value() - 220.0).abs() < 2.0,
            "pyin mid f0 = {} Hz, expected ~220",
            mid.frequency_hz.value(),
        );
        assert!(
            mid.voicing > 0.5,
            "pyin mid voicing = {}, expected > 0.5",
            mid.voicing,
        );
    }

    #[test]
    fn pyin_silent_input_emits_unvoiced() {
        let audio = silent_audio(16_000, 0.5);
        let frames = pyin(&audio, &PitchConfig::default());
        assert!(!frames.is_empty());
        for f in &frames {
            assert!(
                f.voicing < 0.5,
                "silent frame voicing should be low, got {}",
                f.voicing,
            );
        }
    }

    #[test]
    fn pyin_agrees_with_boersma_on_clean_tone() {
        // Cross-validation: two independent algorithmic families on a
        // clean 175 Hz tone should land within 2 Hz. Wider tolerance
        // than YIN's because pYIN snaps to its log-bin grid.
        let audio = sustained_tone(16_000, 175.0, 0.6);
        let cfg = PitchConfig::default();
        let pyin_frames = pyin(&audio, &cfg);
        let boersma_frames = boersma(&audio, &cfg);
        let mid_pyin = pyin_frames[pyin_frames.len() / 2].frequency_hz.value();
        let mid_boersma = boersma_frames[boersma_frames.len() / 2]
            .frequency_hz
            .value();
        assert!(
            (mid_pyin - mid_boersma).abs() < 2.0,
            "pyin / boersma disagree on 175 Hz tone: pyin = {}, boersma = {}",
            mid_pyin,
            mid_boersma,
        );
    }

    #[test]
    fn pyin_hmm_keeps_path_through_noise_gap() {
        // Same spliced-tone fixture as the Boersma transient-smoothing
        // test: the HMM transition cost should keep voiced frames near
        // 200 Hz across the run rather than letting them ramble.
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

        let cfg = PitchConfig::default();
        let frames = pyin(&audio, &cfg);
        let mut voiced_f0s: Vec<f32> = frames
            .iter()
            .filter(|f| f.voicing >= cfg.voicing_threshold)
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
            "pyin HMM median voiced f0 = {} Hz, expected ~200",
            median,
        );
        // No octave-error voiced frames.
        for f in &frames {
            if f.voicing >= cfg.voicing_threshold {
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
    fn beta_pdf_analytic_alpha_2_beta_18_matches_closed_form() {
        // For α=2, β=18: pdf(x) = 342 · x · (1-x)^17. Verify the
        // dispatch returns this for the canonical pair.
        for &x in &[0.05_f32, 0.1, 0.2, 0.5] {
            let expected = 342.0 * x * (1.0 - x).powi(17);
            let got = beta_pdf(x, 2.0, 18.0);
            assert!(
                (got - expected).abs() < 1e-3,
                "beta_pdf(α=2, β=18) at x={x}: got {got}, expected {expected}",
            );
        }
    }

    #[test]
    fn beta_pdf_edges_are_zero() {
        assert_eq!(beta_pdf(0.0, 2.0, 18.0), 0.0);
        assert_eq!(beta_pdf(1.0, 2.0, 18.0), 0.0);
    }
}
