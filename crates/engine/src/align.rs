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
//!   neural networks. *ICML 2006*, 369–376. <https://doi.org/10.1145/1143844.1143891>
//!   — the CTC alignment lattice this is a forced (fixed-target) instance of; the
//!   citation the [`crate::citation`] registry returns for `sadda.align.forced_align`.
//! - Viterbi, A. J. (1967). Error bounds for convolutional codes and an
//!   asymptotically optimum decoding algorithm. *IEEE Trans. Inf. Theory* 13(2):
//!   260–269. <https://doi.org/10.1109/TIT.1967.1054010> — the max-probability-path
//!   dynamic program.
//! - Kürzinger, L., Winkelbauer, D., Li, L., Watzel, T. & Rigoll, G. (2020).
//!   CTC-Segmentation of Large Corpora for German End-to-End Speech Recognition.
//!   *SPECOM 2020*, LNCS 12335. <https://doi.org/10.1007/978-3-030-60276-5_27> —
//!   using CTC posteriors as a forced aligner for text-to-audio segmentation.
//!
//! Implementation reference (not a paper): `torchaudio.functional.forced_align`
//! — same blank-staggered trellis and transition rules.
//!
//! Frames on the CTC *blank* are attributed to the preceding phone
//! (carry-forward), so the returned phone spans are **contiguous** — which is
//! what phoneticians expect of a phone tier — rather than leaving inter-phone
//! gaps. Leading blanks (before the first phone) attach to the first phone.

/// One aligned phone: the position in the target sequence, the class id
/// (phone) it carries, and its frame span into the emission matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenSpan {
    /// Index of this token within the input `targets` sequence.
    pub token: usize,
    /// The emission class id (phone) aligned — `targets[token]`.
    pub label: usize,
    /// First frame of the span (inclusive).
    pub start_frame: usize,
    /// Last frame of the span (exclusive).
    pub end_frame: usize,
    /// Mean emission log-probability over the span (an alignment-confidence proxy).
    pub score: f32,
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
}

const NEG_INF: f32 = f32::NEG_INFINITY;

/// CTC forced alignment.
///
/// - `emissions`: `T` frames, each a slice of `C` **log**-probabilities
///   (log-softmax over classes, blank included). All rows must have width `C`.
/// - `targets`: `L` class ids to align in order (none may equal `blank`).
/// - `blank`: the CTC blank class id.
///
/// Returns one [`TokenSpan`] per target phone, in order, with contiguous frame
/// spans covering `0..T`.
pub fn forced_align(
    emissions: &[&[f32]],
    targets: &[usize],
    blank: usize,
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

    Ok(spans)
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
        let spans = forced_align(&as_slices(&m), &[1, 2], 0).unwrap();
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
        let spans = forced_align(&as_slices(&m), &[1, 2], 0).unwrap();
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
        let spans = forced_align(&as_slices(&m), &[1, 1], 0).unwrap();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].label, 1);
        assert_eq!(spans[1].label, 1);
        assert_eq!(spans[1].end_frame, 3);
        // too few frames for two identical phones (need 3: A, blank, A) fails cleanly.
        let short = emissions(&[1], 3);
        assert_eq!(
            forced_align(&as_slices(&short), &[1, 1], 0),
            Err(AlignError::TooFewFrames { needed: 3, have: 1 })
        );
    }

    #[test]
    fn rejects_bad_input() {
        let m = emissions(&[1, 2], 3);
        assert_eq!(
            forced_align(&as_slices(&m), &[], 0),
            Err(AlignError::EmptyTargets)
        );
        assert_eq!(forced_align(&[], &[1], 0), Err(AlignError::NoFrames));
        assert_eq!(
            forced_align(&as_slices(&m), &[9], 0),
            Err(AlignError::LabelOutOfRange)
        );
    }

    #[test]
    fn score_reflects_emission_confidence() {
        let m = emissions(&[1, 1, 2, 2], 3);
        let spans = forced_align(&as_slices(&m), &[1, 2], 0).unwrap();
        // both phones sat on their hot class → score ~ log(0.8)
        for s in &spans {
            assert!((s.score - 0.8f32.ln()).abs() < 1e-5);
        }
    }

    #[test]
    fn frame_to_seconds_uses_frame_rate() {
        assert!((frame_to_seconds(50, 50.0) - 1.0).abs() < 1e-12);
    }
}
