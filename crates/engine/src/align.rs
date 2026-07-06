//! Forced alignment: align a *known* phone sequence to audio, given per-frame
//! CTC emission posteriors from an acoustic model. This is **stage 2** of the
//! alignment pipeline described in the 2026-07-05 DEVLOG design entry — a pure
//! dynamic program over emissions, with no acoustic model here (the model, e.g.
//! the espeak-IPA wav2vec2 CTC net, produces the emissions in `sadda.ml`; the
//! G2P produces the target phone sequence via espeak-ng).
//!
//! The algorithm is the CTC alignment lattice constrained to a fixed target: a
//! blank-staggered trellis with stay / advance / skip-blank transitions, solved
//! for the max-probability (Viterbi) path. This is the forced-alignment
//! (fixed-transcript) instance of Graves et al.'s CTC, and the same formulation
//! `torchaudio.functional.forced_align` implements; it was cross-checked against
//! that reference. Pure functions over plain slices — no
//! [`crate::corpus::Project`] coupling — so they're unit-testable with synthetic
//! emissions.
//!
//! # References
//!
//! - Graves, A., Fernández, S., Gomez, F. & Schmidhuber, J. (2006). Connectionist
//!   temporal classification: labelling unsegmented sequence data with recurrent
//!   neural networks. *ICML 2006*, 369–376. doi:10.1145/1143844.1143891. — the
//!   CTC alignment lattice this is a forced (fixed-target) instance of; the
//!   citation the [`crate::citation`] registry returns for `sadda.align.forced_align`.
//! - Viterbi, A. J. (1967). Error bounds for convolutional codes and an
//!   asymptotically optimum decoding algorithm. *IEEE Trans. Inf. Theory* 13(2):
//!   260–269. doi:10.1109/TIT.1967.1054010. — the max-probability-path dynamic
//!   program.
//! - Kürzinger, L., Winkelbauer, D., Li, L., Watzel, T. & Rigoll, G. (2020).
//!   CTC-Segmentation of Large Corpora for German End-to-End Speech Recognition.
//!   *SPECOM 2020*, LNCS 12335. doi:10.1007/978-3-030-60276-5_27. — using CTC
//!   posteriors as a forced aligner for text-to-audio segmentation.
//!
//! Implementation reference (not a paper): `torchaudio.functional.forced_align`
//! — same blank-staggered trellis and transition rules.
//!
//! Frames on the CTC *blank* are attributed to the preceding phone
//! (carry-forward), so phones stay contiguous — **except** where silence is
//! carved out (long blank runs via `min_silence_frames`, or an external mask),
//! which becomes its own empty-labeled interval. The result is always a
//! contiguous partition of `0..T` (a full interval tier); silence is *empty*,
//! not a gap.

/// One interval of the aligned tier: either a phone or a stretch of silence.
///
/// The returned spans are **contiguous** and cover `0..T` — a full partition, as
/// an interval tier should be. Silence intervals (`is_silence`) carry no phone
/// (`token == usize::MAX`, `label == blank`); a consumer renders them as
/// empty-labeled intervals.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenSpan {
    /// Index of this phone within `targets`, or `usize::MAX` for a silence span.
    pub token: usize,
    /// The emission class id (`targets[token]` for a phone; `blank` for silence).
    pub label: usize,
    /// First frame of the span (inclusive).
    pub start_frame: usize,
    /// Last frame of the span (exclusive).
    pub end_frame: usize,
    /// Mean emission log-probability over the span (an alignment-confidence proxy).
    pub score: f32,
    /// Whether this span is detected silence (an empty interval) rather than a phone.
    pub is_silence: bool,
}

/// Why a forced alignment could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlignError {
    /// The target phone sequence was empty.
    EmptyTargets,
    /// No emission frames were provided.
    NoFrames,
    /// A target id (or `blank`) is outside the emission's class range.
    LabelOutOfRange,
    /// Too few frames to place every phone (needs at least one frame per phone,
    /// plus a separating blank between adjacent identical phones).
    TooFewFrames {
        /// Minimum frames the target sequence requires.
        needed: usize,
        /// Frames actually supplied.
        have: usize,
    },
    /// The external `silence_mask` length did not match the number of frames.
    SilenceMaskLength,
}

const NEG_INF: f32 = f32::NEG_INFINITY;

/// CTC forced alignment, optionally carving out silence.
///
/// - `emissions`: `T` frames, each a slice of `C` **log**-probabilities
///   (log-softmax over classes, blank included). All rows must have width `C`.
/// - `targets`: `L` class ids to align in order (none may equal `blank`).
/// - `blank`: the CTC blank class id.
/// - `min_silence_frames`: if `> 0`, contiguous runs of the CTC blank at least
///   this long become silence intervals (0 disables blank-run silence).
/// - `silence_mask`: an optional external per-frame silence mask (e.g. from VAD),
///   length `T`; frames marked `true` become silence. Combined (OR) with the
///   blank-run silence.
///
/// Returns a **contiguous** partition of `0..T`: one [`TokenSpan`] per target
/// phone in order, interleaved with `is_silence` spans where silence is detected.
/// With `min_silence_frames == 0` and no mask, the result is fully phone-labeled
/// (one span per target), as before.
pub fn forced_align(
    emissions: &[&[f32]],
    targets: &[usize],
    blank: usize,
    min_silence_frames: usize,
    silence_mask: Option<&[bool]>,
) -> Result<Vec<TokenSpan>, AlignError> {
    let t_len = emissions.len();
    let l_len = targets.len();
    if l_len == 0 {
        return Err(AlignError::EmptyTargets);
    }
    if t_len == 0 {
        return Err(AlignError::NoFrames);
    }
    let n_classes = emissions[0].len();
    if blank >= n_classes || targets.iter().any(|&c| c >= n_classes) {
        return Err(AlignError::LabelOutOfRange);
    }
    if silence_mask.is_some_and(|m| m.len() != t_len) {
        return Err(AlignError::SilenceMaskLength);
    }

    // Shortest feasible path = one frame per phone, plus a mandatory blank
    // between adjacent *identical* phones (CTC can't emit the same label twice
    // without a blank between).
    let repeats = targets.windows(2).filter(|w| w[0] == w[1]).count();
    let needed = l_len + repeats;
    if t_len < needed {
        return Err(AlignError::TooFewFrames {
            needed,
            have: t_len,
        });
    }

    // Blank-staggered extended sequence: [blank, t0, blank, t1, ..., blank].
    let s_len = 2 * l_len + 1;
    let ext = |s: usize| -> usize {
        if s % 2 == 0 {
            blank
        } else {
            targets[(s - 1) / 2]
        }
    };

    // Viterbi trellis in the log domain with backpointers.
    // alpha[t][s] = best log-prob aligning frames 0..=t ending at ext position s.
    let mut alpha = vec![vec![NEG_INF; s_len]; t_len];
    // back[t][s] = the s' at t-1 we came from.
    let mut back = vec![vec![usize::MAX; s_len]; t_len];

    // t = 0: may start on the leading blank (s=0) or the first phone (s=1).
    alpha[0][0] = emissions[0][blank];
    if s_len > 1 {
        alpha[0][1] = emissions[0][ext(1)];
    }

    for t in 1..t_len {
        for s in 0..s_len {
            // Candidate predecessors: stay (s), advance (s-1), skip-blank (s-2).
            let mut best = alpha[t - 1][s];
            let mut from = s;
            if s >= 1 && alpha[t - 1][s - 1] > best {
                best = alpha[t - 1][s - 1];
                from = s - 1;
            }
            // The s-2 skip is only legal past a blank whose neighbours differ
            // (else two identical labels would merge without a separating blank).
            if s >= 2 && ext(s) != blank && ext(s) != ext(s - 2) && alpha[t - 1][s - 2] > best {
                best = alpha[t - 1][s - 2];
                from = s - 2;
            }
            if best > NEG_INF {
                alpha[t][s] = best + emissions[t][ext(s)];
                back[t][s] = from;
            }
        }
    }

    // Terminate on the last phone (s_len-2) or the trailing blank (s_len-1).
    let last_phone = s_len - 2;
    let last_blank = s_len - 1;
    let mut s = if alpha[t_len - 1][last_blank] >= alpha[t_len - 1][last_phone] {
        last_blank
    } else {
        last_phone
    };
    if alpha[t_len - 1][s] == NEG_INF {
        // Unreachable in practice (feasibility checked above), but stay safe.
        return Err(AlignError::TooFewFrames {
            needed,
            have: t_len,
        });
    }

    // Backtrack to a per-frame extended-position path.
    let mut path = vec![0usize; t_len];
    for t in (0..t_len).rev() {
        path[t] = s;
        if t > 0 {
            s = back[t][s];
        }
    }

    // Map each frame to a target-token index, carrying blanks forward onto the
    // preceding phone so spans are contiguous. Leading blanks (before any phone)
    // attach to token 0.
    let mut frame_token = vec![0usize; t_len];
    let mut current: Option<usize> = None;
    for t in 0..t_len {
        let s = path[t];
        if s % 2 == 1 {
            current = Some((s - 1) / 2);
        }
        frame_token[t] = current.unwrap_or(0);
    }

    // Collapse runs of the same token into spans, scoring each by the mean
    // emission log-prob of its phone over the span.
    let mut spans: Vec<TokenSpan> = Vec::with_capacity(l_len);
    let mut start = 0usize;
    for t in 1..=t_len {
        if t == t_len || frame_token[t] != frame_token[start] {
            let token = frame_token[start];
            let label = targets[token];
            let sum: f32 = (start..t).map(|f| emissions[f][label]).sum();
            spans.push(TokenSpan {
                token,
                label,
                start_frame: start,
                end_frame: t,
                score: sum / (t - start) as f32,
                is_silence: false,
            });
            start = t;
        }
    }

    // A phone with zero aligned frames (possible only in pathological ties) is
    // dropped by the run-collapse above; guarantee one span per target by
    // splicing zero-width spans at the correct boundary if any are missing.
    if spans.len() != l_len {
        spans = splice_missing(spans, targets, t_len);
    }

    // Carve detected silence out of the contiguous phone spans → a contiguous
    // tier with empty-labeled silence intervals. Silence = an external per-frame
    // mask (e.g. VAD) OR runs of the CTC blank at least `min_silence_frames` long.
    if min_silence_frames > 0 || silence_mask.is_some() {
        let sil = silence_frames(&path, min_silence_frames, silence_mask, t_len);
        if sil.iter().any(|&b| b) {
            spans = carve_silence(spans, &sil, emissions, blank);
        }
    }

    Ok(spans)
}

/// Per-frame silence: an external `mask` OR runs of the CTC blank (even extended
/// positions in `path`) at least `min_silence_frames` long. Blank runs shorter
/// than the threshold stay attached to a neighbouring phone (coarticulation).
fn silence_frames(
    path: &[usize],
    min_silence_frames: usize,
    mask: Option<&[bool]>,
    t_len: usize,
) -> Vec<bool> {
    let mut sil = vec![false; t_len];
    if let Some(m) = mask {
        sil.copy_from_slice(m);
    }
    if min_silence_frames > 0 {
        let mut i = 0;
        while i < t_len {
            if path[i] % 2 == 0 {
                let start = i;
                while i < t_len && path[i] % 2 == 0 {
                    i += 1;
                }
                if i - start >= min_silence_frames {
                    sil[start..i].fill(true);
                }
            } else {
                i += 1;
            }
        }
    }
    sil
}

/// Split contiguous phone spans into a contiguous tier where silence frames
/// become empty-labeled intervals (`is_silence`, `token == usize::MAX`).
fn carve_silence(
    phone_spans: Vec<TokenSpan>,
    sil: &[bool],
    emissions: &[&[f32]],
    blank: usize,
) -> Vec<TokenSpan> {
    let t_len = sil.len();
    // frame -> index into phone_spans (which are contiguous and cover 0..t_len).
    let mut frame_span = vec![0usize; t_len];
    for (i, sp) in phone_spans.iter().enumerate() {
        frame_span[sp.start_frame..sp.end_frame].fill(i);
    }
    let key = |t: usize| -> Option<usize> { if sil[t] { None } else { Some(frame_span[t]) } };
    let mut out: Vec<TokenSpan> = Vec::with_capacity(phone_spans.len() + 4);
    let mut start = 0usize;
    for t in 1..=t_len {
        if t == t_len || key(t) != key(start) {
            let (start_frame, end_frame) = (start, t);
            match key(start) {
                Some(idx) => {
                    let sp = &phone_spans[idx];
                    let sum: f32 = (start..t).map(|f| emissions[f][sp.label]).sum();
                    out.push(TokenSpan {
                        token: sp.token,
                        label: sp.label,
                        start_frame,
                        end_frame,
                        score: sum / (t - start) as f32,
                        is_silence: false,
                    });
                }
                None => {
                    let sum: f32 = (start..t).map(|f| emissions[f][blank]).sum();
                    out.push(TokenSpan {
                        token: usize::MAX,
                        label: blank,
                        start_frame,
                        end_frame,
                        score: sum / (t - start) as f32,
                        is_silence: true,
                    });
                }
            }
            start = t;
        }
    }
    out
}

/// Ensure exactly one span per target, inserting zero-width spans (at the prior
/// span's end) for any phone the trellis collapsed away. Rare; keeps the phone
/// tier 1:1 with the transcript.
fn splice_missing(partial: Vec<TokenSpan>, targets: &[usize], t_len: usize) -> Vec<TokenSpan> {
    let mut out: Vec<TokenSpan> = Vec::with_capacity(targets.len());
    let mut it = partial.into_iter().peekable();
    for (token, &label) in targets.iter().enumerate() {
        if it.peek().map(|s| s.token) == Some(token) {
            out.push(it.next().unwrap());
        } else {
            let at = out.last().map(|s| s.end_frame).unwrap_or(0).min(t_len);
            out.push(TokenSpan {
                token,
                label,
                start_frame: at,
                end_frame: at,
                score: NEG_INF,
                is_silence: false,
            });
        }
    }
    out
}

/// Convert a frame index to seconds. `frame_rate` is the emission frames per
/// second (for a wav2vec2 CTC head with a 20 ms stride, 50.0).
#[inline]
pub fn frame_to_seconds(frame: usize, frame_rate: f64) -> f64 {
    frame as f64 / frame_rate
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a T×C log-emission matrix from per-frame "hot" classes: the hot
    // class gets log(0.8), the rest share the remainder. Blank is class 0.
    fn emissions(rows: &[usize], n_classes: usize) -> Vec<Vec<f32>> {
        rows.iter()
            .map(|&hot| {
                let hi = 0.8f32.ln();
                let lo = (0.2f32 / (n_classes - 1) as f32).ln();
                (0..n_classes)
                    .map(|c| if c == hot { hi } else { lo })
                    .collect()
            })
            .collect()
    }

    fn as_slices(m: &[Vec<f32>]) -> Vec<&[f32]> {
        m.iter().map(|r| r.as_slice()).collect()
    }

    #[test]
    fn aligns_two_phones_at_the_emission_boundary() {
        // classes: 0=blank, 1=phone A, 2=phone B. Frames 0-1 favour A, 2-3 B.
        let m = emissions(&[1, 1, 2, 2], 3);
        let spans = forced_align(&as_slices(&m), &[1, 2], 0, 0, None).unwrap();
        assert_eq!(spans.len(), 2);
        assert_eq!(
            (spans[0].label, spans[0].start_frame, spans[0].end_frame),
            (1, 0, 2)
        );
        assert_eq!(
            (spans[1].label, spans[1].start_frame, spans[1].end_frame),
            (2, 2, 4)
        );
    }

    #[test]
    fn spans_are_contiguous_covering_all_frames() {
        // A leading blank frame + trailing blank frame must still be covered.
        let m = emissions(&[0, 1, 2, 0], 3);
        let spans = forced_align(&as_slices(&m), &[1, 2], 0, 0, None).unwrap();
        assert_eq!(
            spans[0].start_frame, 0,
            "leading blank attaches to first phone"
        );
        assert_eq!(
            spans.last().unwrap().end_frame,
            4,
            "trailing blank attaches to last phone"
        );
        // no gaps
        for w in spans.windows(2) {
            assert_eq!(w[0].end_frame, w[1].start_frame);
        }
    }

    #[test]
    fn repeated_phone_needs_a_separating_blank() {
        // targets [A, A] require a blank between → at least 3 frames.
        let m = emissions(&[1, 0, 1], 3);
        let spans = forced_align(&as_slices(&m), &[1, 1], 0, 0, None).unwrap();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].label, 1);
        assert_eq!(spans[1].label, 1);
        assert_eq!(spans[1].end_frame, 3);
        // too few frames for two identical phones (need 3: A, blank, A) fails cleanly.
        let short = emissions(&[1], 3);
        assert_eq!(
            forced_align(&as_slices(&short), &[1, 1], 0, 0, None),
            Err(AlignError::TooFewFrames { needed: 3, have: 1 })
        );
    }

    #[test]
    fn rejects_bad_input() {
        let m = emissions(&[1, 2], 3);
        assert_eq!(
            forced_align(&as_slices(&m), &[], 0, 0, None),
            Err(AlignError::EmptyTargets)
        );
        assert_eq!(
            forced_align(&[], &[1], 0, 0, None),
            Err(AlignError::NoFrames)
        );
        assert_eq!(
            forced_align(&as_slices(&m), &[9], 0, 0, None),
            Err(AlignError::LabelOutOfRange)
        );
    }

    #[test]
    fn score_reflects_emission_confidence() {
        let m = emissions(&[1, 1, 2, 2], 3);
        let spans = forced_align(&as_slices(&m), &[1, 2], 0, 0, None).unwrap();
        // both phones sat on their hot class → score ~ log(0.8)
        for s in &spans {
            assert!((s.score - 0.8f32.ln()).abs() < 1e-5);
        }
    }

    #[test]
    fn frame_to_seconds_uses_frame_rate() {
        assert!((frame_to_seconds(50, 50.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn long_blank_run_becomes_an_empty_silence_interval() {
        // A, then a 3-frame blank run, then B. classes: 0=blank, 1=A, 2=B.
        let m = emissions(&[1, 0, 0, 0, 2], 3);
        let slices = as_slices(&m);

        // Without silence: the blank run is carried onto A → contiguous phones.
        let plain = forced_align(&slices, &[1, 2], 0, 0, None).unwrap();
        assert_eq!(plain.len(), 2);
        assert!(plain.iter().all(|s| !s.is_silence));
        assert_eq!(plain[0].end_frame, 4); // A absorbs the blanks

        // With min_silence_frames=2: the 3-frame blank run → an empty interval.
        let sil = forced_align(&slices, &[1, 2], 0, 2, None).unwrap();
        assert_eq!(sil.len(), 3, "A, <silence>, B");
        assert_eq!(
            (sil[0].label, sil[0].start_frame, sil[0].end_frame),
            (1, 0, 1)
        );
        assert!(sil[1].is_silence);
        assert_eq!((sil[1].start_frame, sil[1].end_frame), (1, 4));
        assert_eq!(sil[1].token, usize::MAX);
        assert_eq!(
            (sil[2].label, sil[2].start_frame, sil[2].end_frame),
            (2, 4, 5)
        );
        // still a contiguous partition of 0..T
        assert_eq!(sil[0].start_frame, 0);
        assert_eq!(sil.last().unwrap().end_frame, 5);
        for w in sil.windows(2) {
            assert_eq!(w[0].end_frame, w[1].start_frame);
        }
    }

    #[test]
    fn external_mask_carves_leading_silence() {
        // blank, blank, A, B — mask the two leading frames as silence (e.g. VAD).
        let m = emissions(&[0, 0, 1, 2], 3);
        let mask = [true, true, false, false];
        let spans = forced_align(&as_slices(&m), &[1, 2], 0, 0, Some(&mask)).unwrap();
        assert!(spans[0].is_silence);
        assert_eq!((spans[0].start_frame, spans[0].end_frame), (0, 2));
        assert_eq!((spans[1].label, spans[1].start_frame), (1, 2)); // A no longer absorbs the lead
        assert_eq!(spans.last().unwrap().end_frame, 4);
    }

    #[test]
    fn silence_mask_wrong_length_errors() {
        let m = emissions(&[1, 2], 3);
        let mask = [false, false, false];
        assert_eq!(
            forced_align(&as_slices(&m), &[1, 2], 0, 0, Some(&mask)),
            Err(AlignError::SilenceMaskLength)
        );
    }
}
