//! Fundamental-frequency (f0) estimation — six non-equivalent
//! pitch-estimation methods per the 2026-05-21 DSP method-diversity entry.
//! Three algorithmic families are covered (autocorrelation,
//! cumulative-mean-normalized-difference, and spectral); having
//! independent estimators from different families is the cross-validation
//! story for downstream work.
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
//! **Spectral family:**
//! - [`swipe`] — Camacho & Harris 2008 SWIPE' (prime variant). Matches the
//!   `sqrt`-loudness spectrum (on an ERB-rate scale) against prime-harmonic
//!   cosine kernels at the optimal window size per candidate. A faithful
//!   port of Camacho's dissertation MATLAB, validated against the author's
//!   *own* code run under Octave (the cross-check caught a 1-based-index
//!   bug during development).
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
//! - RAPT: Talkin (1995), in *Speech Coding and Synthesis* (Elsevier),
//!   ISBN 978-0-444-82169-1
//! - CREPE (neural): Kim et al. (2018), <https://arxiv.org/abs/1802.06182>
//!   — SOTA accuracy, requires bundled model weights

use std::f32::consts::PI as PI_F32;

use serde::{Deserialize, Serialize};

use crate::Audio;
use crate::dsp::windowing::hann;

/// One of the supported pitch-estimation methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PitchMethod {
    /// Naive time-domain autocorrelation (Phase-0 method). See
    /// [`autocorrelation`].
    Autocorrelation,
    /// Window-corrected autocorrelation. Adopts the central insight of
    /// Boersma (1993) — divide windowed-signal autocorrelation by window
    /// autocorrelation — but is **not** a faithful Boersma implementation.
    /// **Prone to subharmonic / octave-down errors** on clean tones (it
    /// picks the global max of `r_a/r_w`, which over-inflates long-lag
    /// subharmonic peaks: 150→75, 250→83.3): it has no octave cost or
    /// path-finding. Prefer [`PitchMethod::Boersma`] (the default).
    /// See [`windowed_autocorrelation`].
    WindowedAutocorrelation,
    /// Faithful Boersma 1993 / Praat `Sound: To Pitch (ac)…` with
    /// `very_accurate = false`. Multi-candidate per-frame detection +
    /// Viterbi path-finding. See [`boersma`]. The default tracker.
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
    // `pyin`, not the snake_case default `p_yin`, to match the method-string
    // vocabulary used by `sadda.dsp.voiced_pitch(method=…)`.
    #[serde(rename = "pyin")]
    PYin,
    /// Camacho & Harris 2008 SWIPE' — Sawtooth Waveform Inspired Pitch
    /// Estimator (prime variant). Spectral method: matches the
    /// `sqrt`-loudness ERB-scale spectrum against prime-harmonic cosine
    /// kernels across multiple optimal window sizes. A third algorithmic
    /// family (neither autocorrelation nor CMNDF). See [`swipe`].
    Swipe,
}

/// Configuration for the pitch trackers.
///
/// The `boersma_*` fields are only read by [`PitchMethod::Boersma`]; the
/// other methods ignore them. Defaults match Praat 6.x's
/// `Sound: To Pitch (ac)…` parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

impl Default for PitchMethod {
    /// The canonical default tracker is [`PitchMethod::Boersma`] — the faithful
    /// Praat-equivalent with octave-cost + Viterbi path-finding. It is robust to
    /// the subharmonic (octave-down / ⅓-rate) errors that the simpler
    /// [`windowed_autocorrelation`] latches onto on clean tones (e.g. 150→75,
    /// 250→83.3). App / Python / criteria all default to this.
    fn default() -> Self {
        PitchMethod::Boersma
    }
}

/// Dispatches to one of the six pitch trackers.
pub fn pitch(audio: &Audio, config: &PitchConfig, method: PitchMethod) -> Vec<PitchFrame> {
    match method {
        PitchMethod::Autocorrelation => autocorrelation(audio, config),
        PitchMethod::WindowedAutocorrelation => windowed_autocorrelation(audio, config),
        PitchMethod::Boersma => boersma(audio, config),
        PitchMethod::Yin => yin(audio, config),
        PitchMethod::PYin => pyin(audio, config),
        PitchMethod::Swipe => swipe(audio, config),
    }
}

/// A complete, serializable pitch-tracking specification: a [`PitchMethod`]
/// plus its [`PitchConfig`]. The pitch analogue of `MfccParams` — the unit a
/// preset stores, so a named pitch preset carries both "which algorithm" and
/// "with what knobs". Run via [`pitch_with_params`].
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct PitchParams {
    /// The tracking algorithm.
    #[serde(default)]
    pub method: PitchMethod,
    /// The tracker configuration (analysis + method-specific knobs).
    pub config: PitchConfig,
}

/// Runs the pitch tracker named by a [`PitchParams`] — `pitch(audio,
/// &p.config, p.method)`. The symmetric counterpart of `mfcc_with_params`.
pub fn pitch_with_params(audio: &Audio, params: &PitchParams) -> Vec<PitchFrame> {
    pitch(audio, &params.config, params.method)
}

/// First-pass floor for the two-pass adaptive range estimator (De Looze &
/// Hirst 2008): the low end of a deliberately wide bracket so the initial
/// analysis captures the speaker's true range before it is refined. The paper's
/// exact first-pass value.
pub const TWO_PASS_FLOOR_HZ: f32 = 60.0;
/// First-pass ceiling for the two-pass adaptive range estimator (De Looze &
/// Hirst 2008): the high end of the wide first-pass bracket. The paper's exact
/// first-pass value.
pub const TWO_PASS_CEILING_HZ: f32 = 750.0;

/// Minimum voiced frames before an adaptive range estimate is trusted; below
/// this the f0 distribution is too sparse for stable quartiles.
pub const ADAPTIVE_MIN_VOICED: usize = 10;

/// The usable voiced f0 values (Hz) in `frames`: `voicing >= voicing_threshold`,
/// finite and positive.
fn voiced_f0_values(
    frames: &[PitchFrame],
    voicing_threshold: f32,
) -> impl Iterator<Item = f64> + '_ {
    frames
        .iter()
        .filter(move |f| f.voicing >= voicing_threshold)
        .map(|f| f.frequency_hz.value() as f64)
        .filter(|&hz| hz.is_finite() && hz > 0.0)
}

/// Quantiles of a set of values by type-7 linear interpolation (NumPy's default
/// `quantile`). `qs` are cut points in `[0, 1]`. Returns `None` when fewer than
/// [`ADAPTIVE_MIN_VOICED`] values are present — too few for a stable estimate.
fn quantiles_of(mut vs: Vec<f64>, qs: &[f64]) -> Option<Vec<f32>> {
    if vs.len() < ADAPTIVE_MIN_VOICED {
        return None;
    }
    vs.sort_by(|a, b| a.total_cmp(b));
    let n = vs.len();
    Some(
        qs.iter()
            .map(|&q| {
                let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
                let lo = pos.floor() as usize;
                let hi = pos.ceil() as usize;
                let frac = pos - lo as f64;
                (vs[lo] + (vs[hi] - vs[lo]) * frac) as f32
            })
            .collect(),
    )
}

/// Quantiles of the voiced f0 values in `frames` (those with
/// `voicing >= voicing_threshold`).
///
/// `qs` are cut points in `[0, 1]`; each is evaluated by linear interpolation
/// on the sorted voiced-f0 values (the "linear" / type-7 convention, matching
/// NumPy's default `quantile`). Non-finite or non-positive f0 values are
/// dropped. Returns `None` when fewer than [`ADAPTIVE_MIN_VOICED`] usable voiced
/// frames remain — too few for a stable estimate.
pub fn voiced_f0_quantiles(
    frames: &[PitchFrame],
    voicing_threshold: f32,
    qs: &[f64],
) -> Option<Vec<f32>> {
    quantiles_of(voiced_f0_values(frames, voicing_threshold).collect(), qs)
}

/// Estimates a speaker-appropriate `(floor_hz, ceiling_hz)` from a single
/// recording via the De Looze & Hirst (2008) two-pass rule: analyse f0 over a
/// wide range ([`TWO_PASS_FLOOR_HZ`]–[`TWO_PASS_CEILING_HZ`]), then set
/// `floor = 0.75·q25` and `ceiling = 1.5·q75` from the first and third quartiles
/// of the voiced f0. Returns `None` when the recording has too few voiced frames
/// to estimate a range (see [`voiced_f0_quantiles`]).
///
/// Reference: De Looze, C. & Hirst, D.J. (2008). "Detecting changes in key and
/// range for the automatic modelling and coding of intonation." Speech Prosody
/// 2008, 135–138. <https://doi.org/10.21437/SpeechProsody.2008-32>
pub fn estimate_pitch_range(
    audio: &Audio,
    config: &PitchConfig,
    method: PitchMethod,
) -> Option<(f32, f32)> {
    let mut wide = *config;
    wide.min_freq_hz = TWO_PASS_FLOOR_HZ;
    wide.max_freq_hz = TWO_PASS_CEILING_HZ;
    let frames = pitch(audio, &wide, method);
    let q = voiced_f0_quantiles(&frames, config.voicing_threshold, &[0.25, 0.75])?;
    Some(pitch_range_from_quartiles(q[0], q[1]))
}

/// The De Looze & Hirst (2008) floor/ceiling formula from a recording's first
/// (`q25`) and third (`q75`) voiced-f0 quartiles: `floor = 0.75·q25`,
/// `ceiling = 1.5·q75`, clamped to stay positive and ordered. Shared by the
/// single-recording ([`estimate_pitch_range`]) and speaker-level estimators.
pub fn pitch_range_from_quartiles(q25: f32, q75: f32) -> (f32, f32) {
    let floor = (0.75 * q25).max(1.0);
    let ceiling = (1.5 * q75).max(floor + 1.0);
    (floor, ceiling)
}

/// Two-pass adaptive pitch tracking (De Looze & Hirst 2008): estimate a
/// speaker-appropriate floor/ceiling from the signal with
/// [`estimate_pitch_range`], then track with that refined range. Falls back to
/// the `config`'s own range when the estimate is unavailable (too few voiced
/// frames), degrading to a single pass rather than failing.
pub fn two_pass_pitch(audio: &Audio, config: &PitchConfig, method: PitchMethod) -> Vec<PitchFrame> {
    let mut cfg = *config;
    if let Some((floor, ceiling)) = estimate_pitch_range(audio, config, method) {
        cfg.min_freq_hz = floor;
        cfg.max_freq_hz = ceiling;
    }
    pitch(audio, &cfg, method)
}

/// Complete-pooling speaker pitch range: pool the voiced f0 of *all* a speaker's
/// recordings into one distribution, then apply the De Looze & Hirst formula
/// once (`floor = 0.75·q25`, `ceiling = 1.5·q75`). One range shared by the
/// speaker — the "complete pooling" end of the partial-pooling spectrum.
///
/// This is sadda's own speaker-adaptive baseline (the underlying quartile rule
/// is De Looze & Hirst 2008); pair it with [`empirical_bayes_pitch_ranges`],
/// which partially pools instead. Each recording's `frames` should come from a
/// wide first pass (see [`TWO_PASS_FLOOR_HZ`]/[`TWO_PASS_CEILING_HZ`]) so the
/// pooled distribution isn't clipped. Returns `None` if the speaker has too few
/// voiced frames pooled across all recordings.
pub fn pooled_pitch_range(
    recordings: &[Vec<PitchFrame>],
    voicing_threshold: f32,
) -> Option<(f32, f32)> {
    let pooled: Vec<f64> = recordings
        .iter()
        .flat_map(|frames| voiced_f0_values(frames, voicing_threshold))
        .collect();
    let q = quantiles_of(pooled, &[0.25, 0.75])?;
    Some(pitch_range_from_quartiles(q[0], q[1]))
}

// [docs-impl:sadda.dsp.f0]  — engine algorithm behind the `sadda.dsp.f0`
// PyO3 shim; the source-link scanner renders this as the "impl" link.
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

/// **Spike scaffolding (throwaway — `spike` feature only).** A streaming/boxed
/// compositional mirror of [`autocorrelation`], to check whether "streaming
/// composition is free" holds for an autocorrelation inner-loop profile
/// (memory-access-bound `O(frame·lag)`, no FFT, no root-solving). Output-equality
/// with production is pinned by the test below. See the compositional-DSP thread
/// in DEVLOG.
#[cfg(feature = "spike")]
#[doc(hidden)]
pub mod spike {
    use super::*;

    /// Tagged intermediate carrier (samples → autocorrelation curve → detection).
    enum FVal {
        Samples(Vec<f32>),
        Curve { vals: Vec<f32>, r0: f32 },
        Detected { lag: usize, peak: f32, r0: f32 },
    }
    trait Stage {
        fn run(&mut self, v: FVal) -> FVal;
    }

    /// Materialise the autocorrelation curve `r(τ)` over `[min_lag, max_lag]`
    /// (production tracks the max inline; the compositional split produces the
    /// full curve as an intermediate, then picks).
    struct Autocorrelate {
        min_lag: usize,
        max_lag: usize,
    }
    impl Stage for Autocorrelate {
        fn run(&mut self, v: FVal) -> FVal {
            let FVal::Samples(frame) = v else {
                return v;
            };
            let max_lag = self.max_lag.min(frame.len().saturating_sub(1));
            let r0: f32 = frame.iter().map(|x| x * x).sum();
            let mut vals = Vec::with_capacity((max_lag + 1).saturating_sub(self.min_lag));
            for lag in self.min_lag..=max_lag {
                let mut sum = 0.0f32;
                for i in 0..(frame.len() - lag) {
                    sum += frame[i] * frame[i + lag];
                }
                vals.push(sum);
            }
            FVal::Curve { vals, r0 }
        }
    }
    struct PeakPick {
        min_lag: usize,
    }
    impl Stage for PeakPick {
        fn run(&mut self, v: FVal) -> FVal {
            let FVal::Curve { vals, r0 } = v else {
                return v;
            };
            // First-wins on ties, mirroring `best_lag`'s `sum > best_value`.
            let mut best = self.min_lag;
            let mut best_value = f32::MIN;
            for (k, &sum) in vals.iter().enumerate() {
                if sum > best_value {
                    best_value = sum;
                    best = self.min_lag + k;
                }
            }
            FVal::Detected {
                lag: best,
                peak: best_value.max(0.0),
                r0,
            }
        }
    }

    /// **Streaming composition** of the naive autocorrelation pitch tracker:
    /// one frame at a time through boxed `dyn Stage`s (autocorrelate → peak-pick),
    /// with the cheap frequency/voicing conversion inline.
    pub fn streaming_compositional(
        samples: &[f32],
        sample_rate: u32,
        config: &PitchConfig,
    ) -> Vec<PitchFrame> {
        let sr = sample_rate as f32;
        let frame_size = (config.frame_size_seconds * sr).round() as usize;
        let hop_size = (config.hop_size_seconds * sr).round() as usize;
        let min_lag = (sr / config.max_freq_hz).round() as usize;
        let max_lag = (sr / config.min_freq_hz).round() as usize;
        if samples.len() < frame_size || hop_size == 0 || min_lag >= max_lag {
            return Vec::new();
        }
        let mut stages: Vec<Box<dyn Stage>> = vec![
            Box::new(Autocorrelate { min_lag, max_lag }),
            Box::new(PeakPick { min_lag }),
        ];
        let mut out = Vec::new();
        let mut start = 0;
        while start + frame_size <= samples.len() {
            let mut v = FVal::Samples(samples[start..start + frame_size].to_vec());
            for s in stages.iter_mut() {
                v = s.run(v);
            }
            let FVal::Detected { lag, peak, r0 } = v else {
                unreachable!("pipeline ends in Detected")
            };
            let voicing = if r0 > 0.0 { peak / r0 } else { 0.0 };
            let frequency_hz = sr / lag as f32;
            let time_seconds = (start + frame_size / 2) as f64 / sample_rate as f64;
            out.push(PitchFrame {
                time_seconds,
                frequency_hz: crate::units::Hertz::new(frequency_hz),
                voicing: voicing.clamp(0.0, 1.0),
            });
            start += hop_size;
        }
        out
    }

    /// A coarser boxed stage: the *whole* per-frame detection (autocorrelation +
    /// peak-pick, the intact hot loop) behind a single `dyn` call, fed a frame
    /// slice with no per-frame copy and no curve materialisation. Isolates *pure
    /// dispatch* cost from the decomposition overhead of [`streaming_compositional`].
    trait FramePitchStage {
        fn detect(&self, frame: &[f32]) -> (usize, f32, f32);
    }
    struct AutocorrPeak {
        min_lag: usize,
        max_lag: usize,
    }
    impl FramePitchStage for AutocorrPeak {
        fn detect(&self, frame: &[f32]) -> (usize, f32, f32) {
            // Byte-for-byte the production `best_lag` + r0, kept fused.
            let max_lag = self.max_lag.min(frame.len().saturating_sub(1));
            let r0: f32 = frame.iter().map(|x| x * x).sum();
            let mut best = self.min_lag;
            let mut best_value = f32::MIN;
            for lag in self.min_lag..=max_lag {
                let mut sum = 0.0f32;
                for i in 0..(frame.len() - lag) {
                    sum += frame[i] * frame[i + lag];
                }
                if sum > best_value {
                    best_value = sum;
                    best = lag;
                }
            }
            (best, best_value.max(0.0), r0)
        }
    }

    /// **Fused streaming composition** — one `dyn` call per frame around the
    /// intact detection loop, no per-frame allocation, no intermediate
    /// materialisation. The fair "composition done carefully" comparison.
    pub fn streaming_fused(
        samples: &[f32],
        sample_rate: u32,
        config: &PitchConfig,
    ) -> Vec<PitchFrame> {
        let sr = sample_rate as f32;
        let frame_size = (config.frame_size_seconds * sr).round() as usize;
        let hop_size = (config.hop_size_seconds * sr).round() as usize;
        let min_lag = (sr / config.max_freq_hz).round() as usize;
        let max_lag = (sr / config.min_freq_hz).round() as usize;
        if samples.len() < frame_size || hop_size == 0 || min_lag >= max_lag {
            return Vec::new();
        }
        let stage: Box<dyn FramePitchStage> = Box::new(AutocorrPeak { min_lag, max_lag });
        let mut out = Vec::new();
        let mut start = 0;
        while start + frame_size <= samples.len() {
            let (lag, peak, r0) = stage.detect(&samples[start..start + frame_size]);
            let voicing = if r0 > 0.0 { peak / r0 } else { 0.0 };
            let frequency_hz = sr / lag as f32;
            let time_seconds = (start + frame_size / 2) as f64 / sample_rate as f64;
            out.push(PitchFrame {
                time_seconds,
                frequency_hz: crate::units::Hertz::new(frequency_hz),
                voicing: voicing.clamp(0.0, 1.0),
            });
            start += hop_size;
        }
        out
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::audio::Audio;
        use crate::dsp::mfcc::spike::synth_signal;

        #[test]
        fn streaming_matches_production() {
            let config = PitchConfig::default();
            let sig = synth_signal(16_000, 16_000); // 1 s @ 16 kHz
            let audio = Audio {
                samples: sig.clone(),
                sample_rate: 16_000,
                channels: 1,
            };
            let prod = autocorrelation(&audio, &config);
            let stream = streaming_compositional(&sig, 16_000, &config);
            let fused = streaming_fused(&sig, 16_000, &config);
            assert_eq!(prod.len(), stream.len(), "frame count differs");
            assert_eq!(prod.len(), fused.len(), "fused frame count differs");
            for ((a, b), c) in prod.iter().zip(&stream).zip(&fused) {
                assert!((a.time_seconds - b.time_seconds).abs() < 1e-9);
                assert!(
                    (a.frequency_hz.value() - b.frequency_hz.value()).abs() < 1e-3,
                    "f0 {:?} vs {:?}",
                    a.frequency_hz,
                    b.frequency_hz
                );
                assert!((a.voicing - b.voicing).abs() < 1e-6, "voicing");
                // fused must match production exactly (same fused math).
                assert!((a.frequency_hz.value() - c.frequency_hz.value()).abs() < 1e-3);
                assert!((a.voicing - c.voicing).abs() < 1e-6);
            }
        }
    }
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

/// Computes `r[τ] = Σ_n x[n] · x[n+τ]` for `τ = 0..=max_lag` via the FFT
/// (`autocorr = IFFT(|FFT(x)|²)`), in `O(N log N)` instead of the naive
/// `O(N · max_lag)`. Returns the same (un-normalised) values as the time-domain
/// sum — the signal is zero-padded to `≥ N + max_lag` so the result is the
/// *linear* (not circular) autocorrelation over the requested lags. The FFT
/// plans are cached per thread, so repeated calls (one per analysis frame)
/// reuse them.
fn autocorr_full(x: &[f32], max_lag: usize) -> Vec<f32> {
    use realfft::RealFftPlanner;
    use rustfft::num_complex::Complex;
    use std::cell::RefCell;

    thread_local! {
        static PLANNER: RefCell<RealFftPlanner<f32>> =
            RefCell::new(RealFftPlanner::<f32>::new());
    }

    let n = x.len();
    let max_lag = max_lag.min(n.saturating_sub(1));
    if n == 0 {
        return vec![0.0; max_lag + 1];
    }
    // Zero-pad to avoid circular wraparound for lags 0..=max_lag.
    let fft_len = (n + max_lag + 1).next_power_of_two().max(2);

    PLANNER.with(|p| {
        let mut planner = p.borrow_mut();
        let fwd = planner.plan_fft_forward(fft_len);
        let inv = planner.plan_fft_inverse(fft_len);

        let mut input = fwd.make_input_vec();
        input[..n].copy_from_slice(x);
        for v in input[n..].iter_mut() {
            *v = 0.0;
        }
        let mut spectrum = fwd.make_output_vec();
        fwd.process(&mut input, &mut spectrum)
            .expect("rfft autocorrelation");

        // Power spectrum (real); the inverse real-FFT of |X|² is the
        // autocorrelation. `realfft`'s inverse is un-normalised, so we divide
        // by `fft_len` to recover the time-domain sums exactly.
        for c in spectrum.iter_mut() {
            *c = Complex::new(c.re * c.re + c.im * c.im, 0.0);
        }
        let mut out = inv.make_output_vec();
        inv.process(&mut spectrum, &mut out)
            .expect("irfft autocorrelation");

        let scale = 1.0 / fft_len as f32;
        (0..=max_lag).map(|tau| out[tau] * scale).collect()
    })
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

/// Hz → ERB-rate (Camacho's `hz2erbs`: `21.4·log10(1 + hz/229)`).
fn hz2erbs(hz: f64) -> f64 {
    21.4 * (1.0 + hz / 229.0).log10()
}

/// ERB-rate → Hz (inverse of [`hz2erbs`]).
fn erbs2hz(erbs: f64) -> f64 {
    (10f64.powf(erbs / 21.4) - 1.0) * 229.0
}

/// Primes ≤ `n` (sieve). SWIPE' sums the 1st + prime harmonics only.
fn primes_upto(n: i64) -> Vec<usize> {
    if n < 2 {
        return Vec::new();
    }
    let n = n as usize;
    let mut sieve = vec![true; n + 1];
    sieve[0] = false;
    sieve[1] = false;
    let mut i = 2;
    while i * i <= n {
        if sieve[i] {
            let mut m = i * i;
            while m <= n {
                sieve[m] = false;
                m += i;
            }
        }
        i += 1;
    }
    (2..=n).filter(|&k| sieve[k]).collect()
}

/// Quadratic `(c2, c1, c0)` of `c2·x² + c1·x + c0` through three points
/// (exact Lagrange fit, the 3-point case of Camacho's `polyfit(…, 2)`).
fn quad_through_3(x: [f64; 3], y: [f64; 3]) -> (f64, f64, f64) {
    let (x0, x1, x2) = (x[0], x[1], x[2]);
    let (y0, y1, y2) = (y[0], y[1], y[2]);
    let d0 = (x0 - x1) * (x0 - x2);
    let d1 = (x1 - x0) * (x1 - x2);
    let d2 = (x2 - x0) * (x2 - x1);
    let c2 = y0 / d0 + y1 / d1 + y2 / d2;
    let c1 = -(y0 * (x1 + x2) / d0 + y1 * (x0 + x2) / d1 + y2 * (x0 + x1) / d2);
    let c0 = y0 * x1 * x2 / d0 + y1 * x0 * x2 / d1 + y2 * x0 * x1 / d2;
    (c2, c1, c0)
}

/// Not-a-knot cubic-spline second derivatives ("moments") for **uniformly
/// spaced** samples `y` (spacing `h`) — MATLAB's `interp1(…, 'spline')` /
/// scipy `CubicSpline` default. On a uniform grid the not-a-knot end
/// conditions (`M₀ = 2M₁ − M₂`, `M_{n-1} = 2M_{n-2} − M_{n-3}`) decouple
/// `M₁` and `M_{n-2}` (`= rᵢ/6`), leaving a `1,4,1` tridiagonal system for
/// the interior moments (Thomas algorithm). Validated against scipy.
fn spline_moments_uniform(y: &[f64], h: f64) -> Vec<f64> {
    let n = y.len();
    let mut m = vec![0.0f64; n];
    if n < 3 {
        return m;
    }
    let r = |i: usize| 6.0 / (h * h) * (y[i + 1] - 2.0 * y[i] + y[i - 1]);
    m[1] = r(1) / 6.0;
    if n == 3 {
        m[0] = m[1];
        m[2] = m[1];
        return m;
    }
    m[n - 2] = r(n - 2) / 6.0;
    let cnt = n - 4; // interior unknowns M[2..=n-3]
    if cnt > 0 {
        let mut b = vec![4.0f64; cnt];
        let mut d: Vec<f64> = (0..cnt).map(|k| r(2 + k)).collect();
        d[0] -= m[1];
        d[cnt - 1] -= m[n - 2];
        for k in 1..cnt {
            let w = 1.0 / b[k - 1];
            b[k] -= w;
            d[k] -= w * d[k - 1];
        }
        let mut sol = vec![0.0f64; cnt];
        sol[cnt - 1] = d[cnt - 1] / b[cnt - 1];
        for k in (0..cnt - 1).rev() {
            sol[k] = (d[k] - sol[k + 1]) / b[k];
        }
        m[2..(cnt + 2)].copy_from_slice(&sol);
    }
    m[0] = 2.0 * m[1] - m[2];
    m[n - 1] = 2.0 * m[n - 2] - m[n - 3];
    m
}

/// Evaluates the uniform not-a-knot cubic spline (samples `y` at `0, h,
/// 2h, …`, moments `m` from [`spline_moments_uniform`]) at `xq`. Clamps the
/// segment to the data range; callers zero out points outside it.
fn spline_eval_uniform(y: &[f64], m: &[f64], h: f64, xq: f64) -> f64 {
    let n = y.len();
    let i = ((xq / h).floor() as isize).clamp(0, n as isize - 2) as usize;
    let t = xq - i as f64 * h;
    let (a, b) = (m[i], m[i + 1]);
    y[i] + ((y[i + 1] - y[i]) / h - h * (2.0 * a + b) / 6.0) * t
        + a / 2.0 * t * t
        + (b - a) / (6.0 * h) * t * t * t
}

/// SWIPE' pitch strength of candidate `pc` over the per-frame-L2-normalised
/// ERB-scale loudness `l` (`fERBs × n_frames`): builds the prime-harmonic
/// cosine kernel (peaks `|q−h|<.25 → cos(2πq)`, valleys `.25<|q−h|<.75 →
/// +cos(2πq)/2`), applies the `√(1/f)` envelope, normalises by `‖k₊‖`, and
/// returns `kernel · l` per frame.
fn swipe_strength_one(ferbs: &[f64], l: &[Vec<f64>], pc: f64, n_frames: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    let n_harm = (ferbs[ferbs.len() - 1] / pc - 0.75).floor() as i64;
    let mut harms = vec![1i64];
    harms.extend(primes_upto(n_harm).into_iter().map(|p| p as i64));
    let mut kernel = vec![0.0f64; ferbs.len()];
    for &h in &harms {
        for (e, &fe) in ferbs.iter().enumerate() {
            let q = fe / pc;
            let a = (q - h as f64).abs();
            if a < 0.25 {
                kernel[e] = (2.0 * PI * q).cos();
            } else if a < 0.75 {
                kernel[e] += (2.0 * PI * q).cos() / 2.0;
            }
        }
    }
    let mut nrm = 0.0;
    for (e, &fe) in ferbs.iter().enumerate() {
        kernel[e] *= (1.0 / fe).sqrt();
        if kernel[e] > 0.0 {
            nrm += kernel[e] * kernel[e];
        }
    }
    nrm = nrm.sqrt();
    let mut s = vec![0.0f64; n_frames];
    if nrm == 0.0 {
        return s;
    }
    for (e, k) in kernel.iter().enumerate() {
        let ke = k / nrm;
        if ke == 0.0 {
            continue;
        }
        for (m, sm) in s.iter_mut().enumerate() {
            *sm += ke * l[e][m];
        }
    }
    s
}

/// **SWIPE'** — Camacho & Harris 2008 "Sawtooth Waveform Inspired Pitch
/// Estimator" (prime variant), a faithful port of Camacho's own
/// dissertation-appendix MATLAB `swipep`. A third algorithmic family
/// alongside the autocorrelation and CMNDF trackers: it matches the
/// `sqrt`-loudness spectrum (resampled onto an ERB-rate scale) against
/// prime-harmonic cosine kernels, computed at the optimal power-of-2
/// window size for each candidate pitch and blended across the two
/// bracketing window sizes.
///
/// Reads `min_freq_hz` / `max_freq_hz` as the search range and
/// `hop_size_seconds` as the output time step; `frame_size_seconds` is
/// unused (SWIPE' derives its own per-candidate window sizes). Emits one
/// [`PitchFrame`] per step with `voicing` = the winning candidate's SWIPE'
/// strength, so callers threshold `voicing` to drop unvoiced frames
/// (Camacho's default `sTHR` is 0.30), as with the other trackers.
///
/// ## Fidelity to the published algorithm
/// Step-for-step with Camacho's dissertation MATLAB, with **no algorithmic
/// deviation**: ERB scale, `4·K = 8` window sizing, 50 % hop, the
/// prime-harmonic kernel + `√(1/f)` envelope + `‖k₊‖` normalisation,
/// per-frame L2-normalised loudness, the **not-a-knot cubic-spline**
/// interpolation onto the ERB grid (`interp1(…, 'spline', 0)`),
/// `mu = 1 − |d − i|` cross-window blending, and parabolic refinement on a
/// 1/768-octave grid. Validated against the author's own code: a golden
/// produced by running Camacho's verbatim `swipep` under Octave
/// (`tests/dsp/swipe/`).
///
/// One clarifying note (a difference from *Gorman's C port*, not from the
/// published algorithm): we use Camacho's `hanning(N)` window (denominator
/// `N + 1`); Gorman's widely-used C uses a periodic Hann (denominator `N`).
/// We match the original author's MATLAB. The difference is negligible for
/// SWIPE's large windows regardless.
///
/// Reference: Camacho & Harris (2008), JASA 124(3).
/// <https://doi.org/10.1121/1.2951592>
pub fn swipe(audio: &Audio, config: &PitchConfig) -> Vec<PitchFrame> {
    use std::f64::consts::PI;
    let fs = audio.sample_rate as f64;
    let x: Vec<f64> = audio.mono_samples().map(|s| s as f64).collect();
    let p_min = config.min_freq_hz as f64;
    let p_max = config.max_freq_hz as f64;
    let dt = config.hop_size_seconds as f64;
    if x.is_empty() || p_min <= 0.0 || p_max <= p_min || dt <= 0.0 {
        return Vec::new();
    }

    let dlog2p = 1.0 / 96.0;
    let derbs = 0.1;
    let four_k = 8.0; // 4·K, K = 2

    let n_t = ((x.len() as f64 / fs) / dt).floor() as usize + 1;

    // Pitch candidates, log2-spaced.
    let log2_min = p_min.log2();
    let log2_max = p_max.log2();
    let n_pc = ((log2_max - log2_min) / dlog2p).floor() as usize + 1;
    let log2pc: Vec<f64> = (0..n_pc).map(|i| log2_min + i as f64 * dlog2p).collect();
    let pc: Vec<f64> = log2pc.iter().map(|&l| 2f64.powf(l)).collect();

    // Power-of-2 window sizes (largest for p_min … smallest for p_max).
    let logws_hi = (four_k * fs / p_min).log2().round() as i64;
    let logws_lo = (four_k * fs / p_max).log2().round() as i64;
    if logws_lo < 1 || logws_hi < logws_lo || logws_hi > 30 {
        return Vec::new();
    }
    let ws: Vec<usize> = (logws_lo..=logws_hi).rev().map(|e| 1usize << e).collect();
    let p_opt: Vec<f64> = ws.iter().map(|&w| four_k * fs / w as f64).collect();
    let d: Vec<f64> = log2pc
        .iter()
        .map(|&l| 1.0 + l - (four_k * fs / ws[0] as f64).log2())
        .collect();

    // ERB-spaced analysis frequencies.
    let erb_lo = hz2erbs(pc[0] / 4.0);
    let erb_hi = hz2erbs(fs / 2.0);
    let n_erb = ((erb_hi - erb_lo) / derbs).floor() as usize + 1;
    let ferbs: Vec<f64> = (0..n_erb)
        .map(|i| erbs2hz(erb_lo + i as f64 * derbs))
        .collect();

    let mut s_mat = vec![vec![0.0f64; n_t]; n_pc];
    let mut planner = realfft::RealFftPlanner::<f64>::new();

    for (i, &w) in ws.iter().enumerate() {
        let dn = (4.0 * fs / p_opt[i]).round() as usize; // dc = 4 → hop = w/2
        if dn == 0 {
            continue;
        }
        let pad_l = w / 2;
        let mut xzp = vec![0.0f64; pad_l + x.len() + dn + w / 2];
        xzp[pad_l..pad_l + x.len()].copy_from_slice(&x);
        if xzp.len() < w {
            continue;
        }
        // Camacho's hanning(w): denominator N+1.
        let win: Vec<f64> = (0..w)
            .map(|k| 0.5 * (1.0 - (2.0 * PI * (k as f64 + 1.0) / (w as f64 + 1.0)).cos()))
            .collect();
        let step = dn;
        let ncol = 1 + (xzp.len() - w) / step;
        let n_freq = w / 2 + 1;
        let bin_hz = fs / w as f64;
        let ti: Vec<f64> = (0..ncol)
            .map(|m| ((m * step) as f64 + w as f64 / 2.0) / fs)
            .collect();

        let plan = planner.plan_fft_forward(w);
        let mut fin = plan.make_input_vec();
        let mut fout = plan.make_output_vec();
        let mut mag = vec![vec![0.0f64; ncol]; n_freq];
        for m in 0..ncol {
            for (dst, (&xs, &wv)) in fin
                .iter_mut()
                .zip(xzp[m * step..m * step + w].iter().zip(win.iter()))
            {
                *dst = xs * wv;
            }
            plan.process(&mut fin, &mut fout).expect("realfft sized");
            for (b, c) in fout.iter().enumerate() {
                mag[b][m] = (c.re * c.re + c.im * c.im).sqrt();
            }
        }
        // Loudness L[erb][frame] = sqrt(not-a-knot cubic-spline interpolation
        // of the magnitude onto the ERB grid), 0 outside the spectrum range —
        // matching Camacho's `interp1(…, 'spline', 0)` exactly. The spline is
        // built per frame from that frame's magnitude column.
        let f_hi = (n_freq - 1) as f64 * bin_hz;
        let mut l_mat = vec![vec![0.0f64; ncol]; ferbs.len()];
        let mut col = vec![0.0f64; n_freq];
        for m in 0..ncol {
            for (b, c) in col.iter_mut().enumerate() {
                *c = mag[b][m];
            }
            let moments = spline_moments_uniform(&col, bin_hz);
            for (e, &fe) in ferbs.iter().enumerate() {
                if fe < 0.0 || fe > f_hi {
                    continue;
                }
                let v = spline_eval_uniform(&col, &moments, bin_hz, fe).max(0.0);
                l_mat[e][m] = v.sqrt();
            }
        }
        // Per-frame L2 normalisation over the ERB axis.
        for m in 0..ncol {
            let mut nrm = 0.0;
            for row in &l_mat {
                nrm += row[m] * row[m];
            }
            if nrm > 0.0 {
                let nrm = nrm.sqrt();
                for row in l_mat.iter_mut() {
                    row[m] /= nrm;
                }
            }
        }
        // Candidates using this window + the blend subset `k` (indices into j).
        // `d` is calibrated to Camacho's 1-based window index (d ≈ 1 for the
        // first window), so compare against `i + 1`, not the 0-based `i`.
        let di = (i + 1) as f64;
        let (j_idx, k_in_j): (Vec<usize>, Vec<usize>) = if i == ws.len() - 1 {
            let j: Vec<usize> = (0..n_pc).filter(|&c| d[c] - di > -1.0).collect();
            let k: Vec<usize> = (0..j.len()).filter(|&jj| d[j[jj]] - di < 0.0).collect();
            (j, k)
        } else if i == 0 {
            let j: Vec<usize> = (0..n_pc).filter(|&c| d[c] - di < 1.0).collect();
            let k: Vec<usize> = (0..j.len()).filter(|&jj| d[j[jj]] - di > 0.0).collect();
            (j, k)
        } else {
            let j: Vec<usize> = (0..n_pc).filter(|&c| (d[c] - di).abs() < 1.0).collect();
            let k = (0..j.len()).collect();
            (j, k)
        };
        let mut mu = vec![1.0f64; j_idx.len()];
        for &kj in &k_in_j {
            mu[kj] = 1.0 - (d[j_idx[kj]] - di).abs();
        }
        for (jj, &c) in j_idx.iter().enumerate() {
            let si = swipe_strength_one(&ferbs, &l_mat, pc[c], ncol);
            for (tk, s_row) in s_mat[c].iter_mut().enumerate() {
                let tt = tk as f64 * dt;
                let val = if ncol < 2 || tt < ti[0] || tt > ti[ncol - 1] {
                    f64::NAN
                } else {
                    let mut seg = 0usize;
                    while seg + 1 < ncol && ti[seg + 1] < tt {
                        seg += 1;
                    }
                    let f = (tt - ti[seg]) / (ti[seg + 1] - ti[seg]);
                    si[seg] * (1.0 - f) + si[seg + 1] * f
                };
                *s_row += mu[jj] * val;
            }
        }
    }

    // Parabolic pitch pick per time column.
    let poly_step = 1.0 / 12.0 / 64.0; // 1/768 octave
    let mut out = Vec::with_capacity(n_t);
    for tk in 0..n_t {
        let mut best = f64::NEG_INFINITY;
        let mut imax = usize::MAX;
        for (c, row) in s_mat.iter().enumerate() {
            if row[tk].is_finite() && row[tk] > best {
                best = row[tk];
                imax = c;
            }
        }
        let time_seconds = tk as f64 * dt;
        if imax == usize::MAX {
            out.push(PitchFrame {
                time_seconds,
                frequency_hz: crate::units::Hertz::new(0.0),
                voicing: 0.0,
            });
            continue;
        }
        let voicing = best.clamp(0.0, 1.0) as f32;
        let freq = if imax == 0 || imax == n_pc - 1 {
            pc[0] // Camacho's edge-candidate behaviour
        } else {
            let idx = [imax - 1, imax, imax + 1];
            let tc = [1.0 / pc[idx[0]], 1.0 / pc[idx[1]], 1.0 / pc[idx[2]]];
            let ntc = [
                (tc[0] / tc[1] - 1.0) * 2.0 * PI,
                0.0,
                (tc[2] / tc[1] - 1.0) * 2.0 * PI,
            ];
            let sy = [s_mat[idx[0]][tk], s_mat[idx[1]][tk], s_mat[idx[2]][tk]];
            let (c2, c1, c0) = quad_through_3(ntc, sy);
            let lo = pc[idx[0]].log2();
            let hi = pc[idx[2]].log2();
            let n_fine = ((hi - lo) / poly_step).floor() as usize + 1;
            let mut bestv = f64::NEG_INFINITY;
            let mut bestk = 0usize;
            for kk in 0..n_fine {
                let nf = (1.0 / 2f64.powf(lo + kk as f64 * poly_step) / tc[1] - 1.0) * 2.0 * PI;
                let val = c2 * nf * nf + c1 * nf + c0;
                if val > bestv {
                    bestv = val;
                    bestk = kk;
                }
            }
            2f64.powf(lo + bestk as f64 * poly_step)
        };
        out.push(PitchFrame {
            time_seconds,
            frequency_hz: crate::units::Hertz::new(freq as f32),
            voicing,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The FFT-based `autocorr_full` must return the same values as the naive
    /// time-domain sum it replaced (so the pitch trackers' behaviour is
    /// unchanged — only faster). Checks an arbitrary signal across all lags.
    #[test]
    fn fft_autocorrelation_matches_naive_sum() {
        let x: Vec<f32> = (0..1500)
            .map(|i| (i as f32 * 0.047).sin() + 0.3 * (i as f32 * 0.131).sin() - 0.2)
            .collect();
        let max_lag = 600;
        let fft = autocorr_full(&x, max_lag);
        for tau in 0..=max_lag {
            let naive: f32 = (0..(x.len() - tau)).map(|i| x[i] * x[i + tau]).sum();
            let tol = 1e-3 * (1.0 + naive.abs());
            assert!(
                (fft[tau] - naive).abs() <= tol,
                "lag {tau}: fft {} vs naive {naive}",
                fft[tau]
            );
        }
    }

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
    fn default_pitch_method_is_boersma() {
        // The app, Python `voiced_pitch`, and the criteria `f0` signal all rely
        // on this being the octave-robust tracker.
        assert!(matches!(PitchMethod::default(), PitchMethod::Boersma));
    }

    #[test]
    fn boersma_tracks_pure_sines_without_subharmonic_errors() {
        // Regression guard for the f0 octave-down bug: the simpler
        // `windowed_autocorrelation` latches onto subharmonics of clean tones
        // (150→75, 250→83.3, 300→100 under PitchConfig::default()), so the
        // default tracker was switched to Boersma. Boersma must report the true
        // f0 (not ½/⅓ of it) across the band.
        let sr = 16_000_u32;
        let cfg = PitchConfig::default();
        for &f in &[150.0_f32, 200.0, 250.0, 300.0, 400.0] {
            let audio = sine_audio(sr, 1, f, 0.5);
            let frames = boersma(&audio, &cfg);
            assert!(!frames.is_empty());
            let mid = frames[frames.len() / 2].frequency_hz.value();
            assert!(
                (mid - f).abs() < 3.0,
                "boersma octave/subharmonic error at {f} Hz: got {mid} Hz",
            );
        }
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

    // --- SWIPE' ---

    fn harmonic_audio(sample_rate: u32, f0: f32, duration_s: f32, n_harm: usize) -> Audio {
        let n = (sample_rate as f32 * duration_s) as usize;
        let mut samples = vec![0.0f32; n];
        for (i, s) in samples.iter_mut().enumerate() {
            let t = i as f32 / sample_rate as f32;
            let mut v = 0.0;
            for h in 1..=n_harm {
                v += (2.0 * std::f32::consts::PI * h as f32 * f0 * t).sin() / h as f32;
            }
            *s = v;
        }
        let peak = samples
            .iter()
            .fold(0.0f32, |a, &x| a.max(x.abs()))
            .max(1e-9);
        for s in samples.iter_mut() {
            *s /= peak;
        }
        Audio {
            samples,
            sample_rate,
            channels: 1,
        }
    }

    fn median_voiced(frames: &[PitchFrame]) -> f64 {
        let mut v: Vec<f64> = frames
            .iter()
            .filter(|f| f.voicing >= 0.30)
            .map(|f| f.frequency_hz.value() as f64)
            .collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(!v.is_empty(), "no voiced frames");
        v[v.len() / 2]
    }

    #[test]
    fn spline_uniform_matches_scipy() {
        // Reference values from scipy.interpolate.CubicSpline (default
        // not-a-knot) on x=0..7, y below, queried at the half-points.
        let y = [0.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0, 4.0];
        let m = spline_moments_uniform(&y, 1.0);
        let xq = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        let scipy = [
            1.318481, 0.181519, 1.080442, 1.496711, 1.932723, 2.772425, 1.977575,
        ];
        for (q, &want) in xq.iter().zip(scipy.iter()) {
            let got = spline_eval_uniform(&y, &m, 1.0, *q);
            assert!((got - want).abs() < 1e-5, "x={q}: got {got}, want {want}");
        }
    }

    #[test]
    fn swipe_recovers_harmonic_tone() {
        let audio = harmonic_audio(16_000, 200.0, 0.5, 10);
        let cfg = PitchConfig {
            min_freq_hz: 100.0,
            max_freq_hz: 600.0,
            hop_size_seconds: 0.01,
            ..PitchConfig::default()
        };
        let med = median_voiced(&swipe(&audio, &cfg));
        assert!(
            (med - 200.0).abs() < 2.0,
            "swipe median {med} Hz, expected ~200"
        );
    }

    #[test]
    fn swipe_agrees_with_yin_and_boersma() {
        // Independent-family cross-validation: SWIPE' (spectral), YIN (CMNDF),
        // Boersma (autocorrelation) should agree on a clean harmonic tone.
        let audio = harmonic_audio(16_000, 180.0, 0.5, 12);
        let cfg = PitchConfig {
            min_freq_hz: 100.0,
            max_freq_hz: 600.0,
            hop_size_seconds: 0.01,
            ..PitchConfig::default()
        };
        let s = median_voiced(&swipe(&audio, &cfg));
        let y = median_voiced(&yin(&audio, &cfg));
        let b = median_voiced(&boersma(&audio, &cfg));
        assert!((s - y).abs() < 5.0, "swipe {s} vs yin {y}");
        assert!((s - b).abs() < 5.0, "swipe {s} vs boersma {b}");
    }

    #[test]
    fn swipe_silence_has_no_voiced_frames() {
        let audio = silent_audio(16_000, 0.3);
        let cfg = PitchConfig::default();
        let frames = swipe(&audio, &cfg);
        assert!(
            frames.iter().all(|f| f.voicing < 0.30),
            "silence should be unvoiced"
        );
    }
}
