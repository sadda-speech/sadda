//! Inter-annotation agreement: compare two annotation sets over the same audio
//! and report how well they agree. This is the engine the design calls "one
//! comparison engine, three uses" — inter-annotator agreement (e.g. the
//! `"phones [alice]"` vs `"phones [bob]"` tiers the S4c import produces),
//! auto-criteria-vs-human-gold (a preview `(auto)` tier vs a manual tier), and
//! rubric-version impact (the same tier across rubric versions, S6) are all
//! "compare two label sequences over the same time base".
//!
//! Pure functions over plain `Segment` / `Mark` lists — no
//! [`crate::corpus::Project`] coupling — so they're unit-testable; the
//! `Project::compare_tiers` wrapper adapts stored tiers into these.
//!
//! Two complementary paradigms are reported, because they answer different
//! questions and disagree on purpose (DSP-method-diversity principle):
//!
//! * **Unit-based** (phonetics / forced-alignment tradition): match units
//!   across the two annotations by maximal temporal overlap, then score label
//!   agreement on the matched pairs and the time deviation of their boundaries,
//!   accounting for unmatched units as insertions / deletions. Boundary
//!   tolerance (e.g. 20 ms) mirrors how forced aligners are evaluated.
//! * **Frame-based** (diarization / VAD tradition): sample a fixed time grid and
//!   compare the label each side assigns at each frame. No matching step; it
//!   folds boundary and label error into one number but is robust to wildly
//!   different segmentations.
//!
//! Label agreement is chance-corrected with **Cohen's κ** (Cohen, J. 1960, "A
//! coefficient of agreement for nominal scales", *Educational and Psychological
//! Measurement* 20(1):37–46, <https://doi.org/10.1177/001316446002000104>), the
//! standard two-rater nominal-agreement coefficient; raw percent agreement is
//! reported alongside it.

use std::collections::BTreeMap;

/// One interval-tier unit: a labelled `[start, end)` span (seconds).
#[derive(Debug, Clone)]
pub struct Segment {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds (`> start`).
    pub end: f64,
    /// Label, or `None` for an unlabelled unit.
    pub label: Option<String>,
}

/// One point-tier unit: a labelled instant (seconds).
#[derive(Debug, Clone)]
pub struct Mark {
    /// Time in seconds.
    pub time: f64,
    /// Label, or `None`.
    pub label: Option<String>,
}

/// Knobs for [`compare_intervals`] / [`compare_points`].
#[derive(Debug, Clone, Copy)]
pub struct AgreementOptions {
    /// A boundary (interval) or instant (point) counts as "agreeing" when the
    /// two sides are within this many seconds. Default 20 ms, the common
    /// forced-alignment reporting tolerance.
    pub boundary_tolerance_seconds: f64,
    /// Grid step for the frame-based metric (intervals only). Default 10 ms.
    pub frame_step_seconds: f64,
}

impl Default for AgreementOptions {
    fn default() -> Self {
        Self {
            boundary_tolerance_seconds: 0.020,
            frame_step_seconds: 0.010,
        }
    }
}

/// The result of comparing two annotation sets. Percentages are fractions in
/// `[0, 1]`; κ is in `(-∞, 1]` (1 = perfect, 0 = chance, <0 = worse than
/// chance). Frame fields are `0.0` for point comparisons (frame-based agreement
/// applies to intervals only).
#[derive(Debug, Clone)]
pub struct AgreementReport {
    /// `"interval"` or `"point"`.
    pub tier_type: String,
    /// Unit counts on each side.
    pub n_a: usize,
    /// Unit counts on each side.
    pub n_b: usize,
    /// Units matched 1:1 across the two sides.
    pub n_matched: usize,
    /// Units in A with no match in B (deletions, from B's perspective).
    pub n_only_a: usize,
    /// Units in B with no match in A (insertions).
    pub n_only_b: usize,
    /// Fraction of matched pairs whose labels are equal.
    pub percent_label_agreement: f64,
    /// Cohen's κ over the matched pairs' labels.
    pub cohen_kappa: f64,
    /// Mean absolute boundary/time deviation over matched pairs (seconds): for
    /// intervals the mean over both endpoints, for points the mean `|Δt|`.
    pub mean_abs_boundary_diff: f64,
    /// Fraction of matched boundaries/instants within `boundary_tolerance`.
    pub boundary_within_tolerance: f64,
    /// The tolerance used (seconds), echoed for reporting.
    pub boundary_tolerance_seconds: f64,
    /// Frame-based fraction of agreeing frames (intervals only).
    pub frame_percent_agreement: f64,
    /// Frame-based Cohen's κ (intervals only).
    pub frame_kappa: f64,
    /// The frame step used (seconds), echoed for reporting.
    pub frame_step_seconds: f64,
}

/// Gap / silence marker used as its own label category in the frame-based
/// metric (a frame covered by no interval).
const GAP: &str = "\u{2205}"; // ∅

/// Cohen's κ over a list of `(label_a, label_b)` pairs, treating each distinct
/// label (including a sentinel for `None`) as a nominal category.
///
/// Conventions for degenerate cases: no pairs → `0.0`; when chance agreement
/// `pe` is 1 (a single category on at least one side), κ is `1.0` iff observed
/// agreement is perfect, else `0.0` (the usual "κ undefined → collapse" choice).
fn cohen_kappa(pairs: &[(String, String)]) -> f64 {
    let n = pairs.len();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let agree = pairs.iter().filter(|(a, b)| a == b).count() as f64;
    let po = agree / nf;
    // Marginal label distributions.
    let mut count_a: BTreeMap<&str, f64> = BTreeMap::new();
    let mut count_b: BTreeMap<&str, f64> = BTreeMap::new();
    for (a, b) in pairs {
        *count_a.entry(a.as_str()).or_insert(0.0) += 1.0;
        *count_b.entry(b.as_str()).or_insert(0.0) += 1.0;
    }
    let mut pe = 0.0;
    for (label, &ca) in &count_a {
        if let Some(&cb) = count_b.get(label) {
            pe += (ca / nf) * (cb / nf);
        }
    }
    if (1.0 - pe).abs() < 1e-12 {
        return if (po - 1.0).abs() < 1e-12 { 1.0 } else { 0.0 };
    }
    (po - pe) / (1.0 - pe)
}

fn label_key(label: &Option<String>) -> String {
    // A distinct sentinel for "no label", kept out of the user's label space.
    label.clone().unwrap_or_else(|| "\u{2400}none".to_string())
}

/// Compares two interval annotations. Matching is greedy by maximal temporal
/// overlap (one-to-one); unmatched units are counted as only-in-A / only-in-B.
/// Reports both the unit-based metrics (label κ, boundary deviation/tolerance)
/// and the frame-based metrics (grid-sampled label κ / agreement).
pub fn compare_intervals(a: &[Segment], b: &[Segment], opts: &AgreementOptions) -> AgreementReport {
    // ---- Unit matching: greedy by descending overlap, deterministic ties. ----
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    for (i, sa) in a.iter().enumerate() {
        for (j, sb) in b.iter().enumerate() {
            let ov = (sa.end.min(sb.end) - sa.start.max(sb.start)).max(0.0);
            if ov > 0.0 {
                candidates.push((ov, i, j));
            }
        }
    }
    // Larger overlap first; then by indices for a stable, reproducible result.
    candidates.sort_by(|x, y| {
        y.0.partial_cmp(&x.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.1.cmp(&y.1))
            .then(x.2.cmp(&y.2))
    });
    let mut used_a = vec![false; a.len()];
    let mut used_b = vec![false; b.len()];
    let mut matched: Vec<(usize, usize)> = Vec::new();
    for (_, i, j) in candidates {
        if !used_a[i] && !used_b[j] {
            used_a[i] = true;
            used_b[j] = true;
            matched.push((i, j));
        }
    }

    let label_pairs: Vec<(String, String)> = matched
        .iter()
        .map(|&(i, j)| (label_key(&a[i].label), label_key(&b[j].label)))
        .collect();
    let n_matched = matched.len();
    let percent_label_agreement = if n_matched == 0 {
        0.0
    } else {
        label_pairs.iter().filter(|(x, y)| x == y).count() as f64 / n_matched as f64
    };
    let cohen = cohen_kappa(&label_pairs);

    // Boundary deviations: both endpoints of each matched pair.
    let mut devs: Vec<f64> = Vec::with_capacity(n_matched * 2);
    for &(i, j) in &matched {
        devs.push((a[i].start - b[j].start).abs());
        devs.push((a[i].end - b[j].end).abs());
    }
    let (mean_abs_boundary_diff, boundary_within_tolerance) =
        deviation_stats(&devs, opts.boundary_tolerance_seconds);

    // ---- Frame-based: sample [0, max-end) and compare per-frame labels. ----
    let extent = a
        .iter()
        .map(|s| s.end)
        .chain(b.iter().map(|s| s.end))
        .fold(0.0_f64, f64::max);
    let (frame_percent_agreement, frame_kappa) =
        frame_metrics(a, b, extent, opts.frame_step_seconds);

    AgreementReport {
        tier_type: "interval".to_string(),
        n_a: a.len(),
        n_b: b.len(),
        n_matched,
        n_only_a: a.len() - n_matched,
        n_only_b: b.len() - n_matched,
        percent_label_agreement,
        cohen_kappa: cohen,
        mean_abs_boundary_diff,
        boundary_within_tolerance,
        boundary_tolerance_seconds: opts.boundary_tolerance_seconds,
        frame_percent_agreement,
        frame_kappa,
        frame_step_seconds: opts.frame_step_seconds,
    }
}

/// Compares two point annotations. Matching is greedy by ascending time
/// distance (one-to-one, nearest first); leftover points are only-in-A /
/// only-in-B. Reports label κ / agreement and the time deviation of matched
/// pairs. Frame-based metrics do not apply to points and are `0.0`.
pub fn compare_points(a: &[Mark], b: &[Mark], opts: &AgreementOptions) -> AgreementReport {
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    for (i, ma) in a.iter().enumerate() {
        for (j, mb) in b.iter().enumerate() {
            candidates.push(((ma.time - mb.time).abs(), i, j));
        }
    }
    candidates.sort_by(|x, y| {
        x.0.partial_cmp(&y.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.1.cmp(&y.1))
            .then(x.2.cmp(&y.2))
    });
    let mut used_a = vec![false; a.len()];
    let mut used_b = vec![false; b.len()];
    let mut matched: Vec<(usize, usize)> = Vec::new();
    for (_, i, j) in candidates {
        if !used_a[i] && !used_b[j] {
            used_a[i] = true;
            used_b[j] = true;
            matched.push((i, j));
        }
    }

    let label_pairs: Vec<(String, String)> = matched
        .iter()
        .map(|&(i, j)| (label_key(&a[i].label), label_key(&b[j].label)))
        .collect();
    let n_matched = matched.len();
    let percent_label_agreement = if n_matched == 0 {
        0.0
    } else {
        label_pairs.iter().filter(|(x, y)| x == y).count() as f64 / n_matched as f64
    };
    let cohen = cohen_kappa(&label_pairs);

    let devs: Vec<f64> = matched
        .iter()
        .map(|&(i, j)| (a[i].time - b[j].time).abs())
        .collect();
    let (mean_abs_boundary_diff, boundary_within_tolerance) =
        deviation_stats(&devs, opts.boundary_tolerance_seconds);

    AgreementReport {
        tier_type: "point".to_string(),
        n_a: a.len(),
        n_b: b.len(),
        n_matched,
        n_only_a: a.len() - n_matched,
        n_only_b: b.len() - n_matched,
        percent_label_agreement,
        cohen_kappa: cohen,
        mean_abs_boundary_diff,
        boundary_within_tolerance,
        boundary_tolerance_seconds: opts.boundary_tolerance_seconds,
        frame_percent_agreement: 0.0,
        frame_kappa: 0.0,
        frame_step_seconds: opts.frame_step_seconds,
    }
}

fn deviation_stats(devs: &[f64], tolerance: f64) -> (f64, f64) {
    if devs.is_empty() {
        return (0.0, 0.0);
    }
    let mean = devs.iter().sum::<f64>() / devs.len() as f64;
    let within = devs.iter().filter(|&&d| d <= tolerance).count() as f64 / devs.len() as f64;
    (mean, within)
}

/// Label of the interval covering `t` (`start <= t < end`), or the `GAP`
/// sentinel if no interval covers it. First covering interval wins.
fn label_at(segments: &[Segment], t: f64) -> String {
    for s in segments {
        if t >= s.start && t < s.end {
            return s.label.clone().unwrap_or_else(|| GAP.to_string());
        }
    }
    GAP.to_string()
}

fn frame_metrics(a: &[Segment], b: &[Segment], extent: f64, step: f64) -> (f64, f64) {
    if extent <= 0.0 || step <= 0.0 {
        return (0.0, 0.0);
    }
    let n_frames = (extent / step).ceil() as usize;
    let mut pairs: Vec<(String, String)> = Vec::with_capacity(n_frames);
    for k in 0..n_frames {
        let t = (k as f64 + 0.5) * step; // frame centre
        pairs.push((label_at(a, t), label_at(b, t)));
    }
    if pairs.is_empty() {
        return (0.0, 0.0);
    }
    let po = pairs.iter().filter(|(x, y)| x == y).count() as f64 / pairs.len() as f64;
    (po, cohen_kappa(&pairs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f64, end: f64, label: &str) -> Segment {
        Segment {
            start,
            end,
            label: Some(label.into()),
        }
    }

    #[test]
    fn identical_interval_tiers_agree_perfectly() {
        let a = vec![seg(0.0, 0.1, "a"), seg(0.1, 0.2, "b"), seg(0.2, 0.3, "a")];
        let r = compare_intervals(&a, &a.clone(), &AgreementOptions::default());
        assert_eq!((r.n_a, r.n_b, r.n_matched), (3, 3, 3));
        assert_eq!(r.n_only_a, 0);
        assert!((r.percent_label_agreement - 1.0).abs() < 1e-9);
        assert!((r.cohen_kappa - 1.0).abs() < 1e-9);
        assert!(r.mean_abs_boundary_diff < 1e-9);
        assert!((r.boundary_within_tolerance - 1.0).abs() < 1e-9);
        assert!((r.frame_percent_agreement - 1.0).abs() < 1e-9);
        assert!((r.frame_kappa - 1.0).abs() < 1e-9);
    }

    #[test]
    fn label_disagreement_drops_kappa_but_matches_units() {
        // Same spans, one label flipped → units all match, label agreement 2/3.
        let a = vec![seg(0.0, 0.1, "a"), seg(0.1, 0.2, "b"), seg(0.2, 0.3, "a")];
        let b = vec![seg(0.0, 0.1, "a"), seg(0.1, 0.2, "b"), seg(0.2, 0.3, "c")];
        let r = compare_intervals(&a, &b, &AgreementOptions::default());
        assert_eq!(r.n_matched, 3);
        assert!((r.percent_label_agreement - 2.0 / 3.0).abs() < 1e-9);
        assert!(r.cohen_kappa < 1.0);
        // Boundaries identical → zero deviation.
        assert!(r.mean_abs_boundary_diff < 1e-9);
    }

    #[test]
    fn boundary_shift_within_and_beyond_tolerance() {
        let a = vec![seg(0.0, 0.10, "a")];
        let b = vec![seg(0.0, 0.13, "a")]; // end shifted 30 ms
        let opts = AgreementOptions {
            boundary_tolerance_seconds: 0.020,
            frame_step_seconds: 0.010,
        };
        let r = compare_intervals(&a, &b, &opts);
        assert_eq!(r.n_matched, 1);
        // Deviations: start 0, end 0.03 → mean 0.015; one of two within 20 ms.
        assert!((r.mean_abs_boundary_diff - 0.015).abs() < 1e-9);
        assert!((r.boundary_within_tolerance - 0.5).abs() < 1e-9);
    }

    #[test]
    fn insertions_and_deletions_are_counted() {
        let a = vec![seg(0.0, 0.1, "a"), seg(0.1, 0.2, "b")];
        let b = vec![seg(0.0, 0.1, "a")]; // B missing the second unit
        let r = compare_intervals(&a, &b, &AgreementOptions::default());
        assert_eq!((r.n_matched, r.n_only_a, r.n_only_b), (1, 1, 0));
    }

    #[test]
    fn frame_metric_reflects_partial_overlap() {
        // A: one "a" over [0,0.1). B: "a" over [0,0.05), gap after.
        let a = vec![seg(0.0, 0.10, "a")];
        let b = vec![seg(0.0, 0.05, "a")];
        let opts = AgreementOptions {
            boundary_tolerance_seconds: 0.02,
            frame_step_seconds: 0.01,
        };
        let r = compare_intervals(&a, &b, &opts);
        // 10 frames; first 5 agree ("a"), last 5 disagree ("a" vs ∅) → 0.5.
        assert!((r.frame_percent_agreement - 0.5).abs() < 1e-9);
    }

    #[test]
    fn points_match_nearest_and_measure_time_deviation() {
        let a = vec![
            Mark {
                time: 0.10,
                label: Some("x".into()),
            },
            Mark {
                time: 0.50,
                label: Some("y".into()),
            },
        ];
        let b = vec![
            Mark {
                time: 0.12,
                label: Some("x".into()),
            },
            Mark {
                time: 0.55,
                label: Some("z".into()),
            },
        ];
        let r = compare_points(&a, &b, &AgreementOptions::default());
        assert_eq!((r.n_a, r.n_b, r.n_matched), (2, 2, 2));
        assert!((r.percent_label_agreement - 0.5).abs() < 1e-9); // x==x, y!=z
        // |0.10-0.12| + |0.50-0.55| = 0.02 + 0.05 → mean 0.035.
        assert!((r.mean_abs_boundary_diff - 0.035).abs() < 1e-9);
        assert_eq!(r.frame_percent_agreement, 0.0); // N/A for points
    }

    #[test]
    fn kappa_corrects_for_chance_on_skewed_labels() {
        // 9/10 "a", 1 disagreement → high raw agreement but κ < raw.
        let a: Vec<Segment> = (0..10)
            .map(|i| seg(i as f64, i as f64 + 1.0, "a"))
            .collect();
        let mut b = a.clone();
        b[0].label = Some("b".into());
        let r = compare_intervals(&a, &b, &AgreementOptions::default());
        assert!((r.percent_label_agreement - 0.9).abs() < 1e-9);
        assert!(r.cohen_kappa < r.percent_label_agreement);
    }

    #[test]
    fn empty_inputs_are_safe() {
        let r = compare_intervals(&[], &[], &AgreementOptions::default());
        assert_eq!((r.n_a, r.n_b, r.n_matched), (0, 0, 0));
        assert_eq!(r.cohen_kappa, 0.0);
        assert_eq!(r.frame_percent_agreement, 0.0);
    }
}
