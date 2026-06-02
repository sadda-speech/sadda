//! S2 criteria engine: re-runnable rules that select regions of interest and
//! emit them as *proposals* for review.
//!
//! This module holds the rule **model** — serialized to/from a criterion's
//! JSON `body` — and the **pure evaluator** over already-fetched intervals.
//! Persistence, the tier lookups, and materialization onto a preview ("auto")
//! tier live on [`crate::corpus::Project`]. Design: the 2026-05-30 DEVLOG
//! annotation-workflow entries and the S2 decisions (structured rules +
//! Python escape; cross-tier predicates + within-interval anchors/spans;
//! proposals on an auto tier that promote on accept).

use serde::{Deserialize, Serialize};

pub mod expr;

use expr::{Expr, SignalSet, Value};

/// Float tolerance for containment/overlap comparisons (seconds).
const EPS: f64 = 1e-9;

/// One interval handed to the evaluator (a slimmed [`crate::corpus::Interval`]).
#[derive(Debug, Clone, PartialEq)]
pub struct EvalInterval {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Label, if any.
    pub label: Option<String>,
}

/// Selects intervals on a named tier, optionally filtered by label. With
/// neither `label_any` nor `label_regex`, every interval on the tier matches.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Selector {
    /// Tier name to select from.
    pub tier: String,
    /// If set, the label must be exactly one of these values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_any: Option<Vec<String>>,
    /// If set, the label must match this regular expression.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_regex: Option<String>,
}

impl Selector {
    /// Compiles a reusable label predicate; errors on an invalid regex.
    fn matcher(&self) -> Result<LabelMatcher, String> {
        let re = match &self.label_regex {
            Some(p) => {
                Some(regex::Regex::new(p).map_err(|e| format!("invalid label_regex {p:?}: {e}"))?)
            }
            None => None,
        };
        Ok(LabelMatcher {
            any: self.label_any.clone(),
            re,
        })
    }
}

struct LabelMatcher {
    any: Option<Vec<String>>,
    re: Option<regex::Regex>,
}

impl LabelMatcher {
    fn matches(&self, label: Option<&str>) -> bool {
        let l = label.unwrap_or("");
        if let Some(any) = &self.any {
            if !any.iter().any(|v| v == l) {
                return false;
            }
        }
        if let Some(re) = &self.re {
            if !re.is_match(l) {
                return false;
            }
        }
        true
    }
}

fn default_to() -> f64 {
    1.0
}
fn default_at() -> f64 {
    0.5
}

/// What to emit for each matched interval. The proportion-based `span`/`point`
/// (S2) place an anchor at a fraction (0..1) of the match; the expression-based
/// `point_expr`/`span_expr` (S3) place it at a signal-function [`Expr`] — e.g.
/// `argmax(intensity)` or `start + 30%` — evaluated over the match.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Emit {
    /// A span from `from` to `to` (proportions of the matched interval).
    /// Defaults to the whole interval (`from=0.0`, `to=1.0`).
    Span {
        /// Start of the sub-span as a proportion (0..1) of the match.
        #[serde(default)]
        from: f64,
        /// End of the sub-span as a proportion (0..1) of the match.
        #[serde(default = "default_to")]
        to: f64,
    },
    /// A point at proportion `at` (default `0.5`, the midpoint).
    Point {
        /// Anchor position as a proportion (0..1) of the match.
        #[serde(default = "default_at")]
        at: f64,
    },
    /// A point at a Time [`Expr`] over the match (S3), e.g. `argmax(intensity)`
    /// or `start + 20ms`. The expression's value is read as seconds.
    PointExpr {
        /// Expression yielding the anchor time, in seconds.
        at: String,
    },
    /// A span between two Time [`Expr`]s over the match (S3).
    SpanExpr {
        /// Expression yielding the span start, in seconds.
        from: String,
        /// Expression yielding the span end, in seconds.
        to: String,
    },
}

impl Emit {
    /// Whether this rule emits points (vs. spans) — drives the preview tier
    /// type.
    pub fn is_point(&self) -> bool {
        matches!(self, Emit::Point { .. } | Emit::PointExpr { .. })
    }
}

/// A structured criterion rule (the parsed form of a `structured` criterion's
/// JSON `body`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CriterionRule {
    /// The intervals to consider.
    pub select: Selector,
    /// Keep only matches fully contained within an interval selected here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within: Option<Selector>,
    /// Keep only matches overlapping an interval selected here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overlaps: Option<Selector>,
    /// Signal-function filter (S3): keep only matches for which this boolean
    /// [`Expr`] is true — e.g. `mean(f0) > 1.2 * mean(f0, file)`. A match where
    /// the expression is undefined (an empty reduction, e.g. f0 over a fully
    /// unvoiced interval) is dropped, same as `false`.
    #[serde(default, rename = "where", skip_serializing_if = "Option::is_none")]
    pub where_expr: Option<String>,
    /// What to emit per surviving match.
    pub emit: Emit,
    /// Fixed label for the proposals; when absent the matched interval's label
    /// is carried over.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

impl CriterionRule {
    /// Parses a rule from its JSON body.
    pub fn from_json(body: &str) -> Result<Self, String> {
        serde_json::from_str(body).map_err(|e| format!("invalid criterion rule: {e}"))
    }

    /// Serializes the rule to a JSON body.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("failed to serialize rule: {e}"))
    }
}

/// One emitted proposal. `end` is `None` for a point.
#[derive(Debug, Clone, PartialEq)]
pub struct Proposal {
    /// Start (span) or time (point), in seconds.
    pub start: f64,
    /// End in seconds for a span; `None` for a point.
    pub end: Option<f64>,
    /// Proposed label.
    pub label: Option<String>,
}

/// The signal-function expressions a [`CriterionRule`] references (S3): an
/// optional `where` filter and the emit anchor expressions, parsed once.
struct CompiledExprs {
    filter: Option<Expr>,
    point_at: Option<Expr>,
    span: Option<(Expr, Expr)>,
}

impl CriterionRule {
    /// Parses the rule's `where` and expression-based emit anchors. Errors on a
    /// malformed expression (an authoring error).
    fn compile_exprs(&self) -> Result<CompiledExprs, String> {
        let filter = self
            .where_expr
            .as_deref()
            .map(Expr::parse)
            .transpose()
            .map_err(|e| format!("invalid `where` expression: {e}"))?;
        let (point_at, span) = match &self.emit {
            Emit::PointExpr { at } => (
                Some(Expr::parse(at).map_err(|e| format!("invalid `at` expression: {e}"))?),
                None,
            ),
            Emit::SpanExpr { from, to } => (
                None,
                Some((
                    Expr::parse(from).map_err(|e| format!("invalid `from` expression: {e}"))?,
                    Expr::parse(to).map_err(|e| format!("invalid `to` expression: {e}"))?,
                )),
            ),
            _ => (None, None),
        };
        Ok(CompiledExprs {
            filter,
            point_at,
            span,
        })
    }

    /// The distinct signal names this rule references across its `where` and
    /// emit expressions — what the caller must pre-compute into the
    /// [`SignalSet`]. Empty for a pure S2 (proportion-only) rule.
    pub fn referenced_signals(&self) -> Result<Vec<String>, String> {
        let c = self.compile_exprs()?;
        let mut out = Vec::new();
        let mut add = |e: &Expr| {
            for s in e.signals() {
                if !out.contains(&s) {
                    out.push(s);
                }
            }
        };
        if let Some(e) = &c.filter {
            add(e);
        }
        if let Some(e) = &c.point_at {
            add(e);
        }
        if let Some((a, b)) = &c.span {
            add(a);
            add(b);
        }
        Ok(out)
    }
}

/// Evaluates an anchor [`Expr`] to a time in seconds. `Ok(None)` means the
/// expression was undefined over the match (skip it); a boolean result is an
/// authoring error.
fn eval_time(e: &Expr, ctx: &expr::EvalCtx) -> Result<Option<f64>, String> {
    match e.eval(ctx)? {
        Some(Value::Num(t)) => Ok(Some(t)),
        Some(Value::Bool(_)) => Err("anchor expression must be a number, not a boolean".into()),
        None => Ok(None),
    }
}

/// Selects the regions of interest `rule` matches: the `select`-tier intervals
/// that pass the label predicate, the `within`/`overlaps` relations, and the
/// signal-function `where` filter (false or undefined → dropped). Returned in
/// select order. This is the "RoI query" half of a criterion, shared by
/// [`evaluate`] (which then emits an anchor/span per RoI) and the campaign
/// layer's target generation (which turns each RoI into a unit of work).
pub fn select_rois(
    rule: &CriterionRule,
    select_ivs: &[EvalInterval],
    within_ivs: &[EvalInterval],
    overlaps_ivs: &[EvalInterval],
    signals: &SignalSet,
) -> Result<Vec<EvalInterval>, String> {
    let select_m = rule.select.matcher()?;
    let within_m = rule.within.as_ref().map(Selector::matcher).transpose()?;
    let overlaps_m = rule.overlaps.as_ref().map(Selector::matcher).transpose()?;
    let filter = rule.compile_exprs()?.filter;

    let mut out = Vec::new();
    for iv in select_ivs {
        if !select_m.matches(iv.label.as_deref()) {
            continue;
        }
        if let Some(m) = &within_m {
            let contained = within_ivs.iter().any(|r| {
                m.matches(r.label.as_deref()) && r.start <= iv.start + EPS && r.end + EPS >= iv.end
            });
            if !contained {
                continue;
            }
        }
        if let Some(m) = &overlaps_m {
            let overlapping = overlaps_ivs
                .iter()
                .any(|r| m.matches(r.label.as_deref()) && r.start < iv.end && iv.start < r.end);
            if !overlapping {
                continue;
            }
        }
        // Signal-function `where` filter: drop the match on false or undefined.
        if let Some(f) = &filter {
            let ctx = expr::EvalCtx {
                start: iv.start,
                end: iv.end,
                signals,
            };
            match f.eval(&ctx)? {
                Some(Value::Bool(true)) => {}
                Some(Value::Bool(false)) | None => continue,
                Some(Value::Num(_)) => {
                    return Err("`where` expression must be a boolean, not a number".into());
                }
            }
        }
        out.push(iv.clone());
    }
    Ok(out)
}

/// Evaluates `rule` against pre-fetched intervals. `select_ivs` are the
/// select-tier intervals; `within_ivs` / `overlaps_ivs` are the relation
/// tiers' intervals (the caller fetches whichever the rule references and
/// passes `&[]` for unused ones). `signals` holds the pre-computed series the
/// rule's `where`/emit expressions reference (empty for a pure S2 rule).
/// Returns the proposals in select order.
pub fn evaluate(
    rule: &CriterionRule,
    select_ivs: &[EvalInterval],
    within_ivs: &[EvalInterval],
    overlaps_ivs: &[EvalInterval],
    signals: &SignalSet,
) -> Result<Vec<Proposal>, String> {
    let exprs = rule.compile_exprs()?;

    let mut out = Vec::new();
    for iv in select_rois(rule, select_ivs, within_ivs, overlaps_ivs, signals)? {
        let ctx = expr::EvalCtx {
            start: iv.start,
            end: iv.end,
            signals,
        };
        let dur = iv.end - iv.start;
        let label = rule.label.clone().or_else(|| iv.label.clone());
        match &rule.emit {
            Emit::Span { from, to } => {
                let s = iv.start + from.clamp(0.0, 1.0) * dur;
                let e = iv.start + to.clamp(0.0, 1.0) * dur;
                if e > s + EPS {
                    out.push(Proposal {
                        start: s,
                        end: Some(e),
                        label,
                    });
                }
            }
            Emit::Point { at } => {
                out.push(Proposal {
                    start: iv.start + at.clamp(0.0, 1.0) * dur,
                    end: None,
                    label,
                });
            }
            Emit::PointExpr { .. } => {
                let e = exprs.point_at.as_ref().expect("compiled point_expr");
                if let Some(t) = eval_time(e, &ctx)? {
                    out.push(Proposal {
                        start: t,
                        end: None,
                        label,
                    });
                }
            }
            Emit::SpanExpr { .. } => {
                let (fe, te) = exprs.span.as_ref().expect("compiled span_expr");
                if let (Some(s), Some(e)) = (eval_time(fe, &ctx)?, eval_time(te, &ctx)?) {
                    if e > s + EPS {
                        out.push(Proposal {
                            start: s,
                            end: Some(e),
                            label,
                        });
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iv(start: f64, end: f64, label: &str) -> EvalInterval {
        EvalInterval {
            start,
            end,
            label: Some(label.into()),
        }
    }

    /// No signals — for the pure S2 (proportion-only) evaluator tests.
    fn no_sig() -> SignalSet {
        SignalSet::new()
    }

    #[test]
    fn rule_json_round_trips() {
        let rule = CriterionRule {
            select: Selector {
                tier: "phones".into(),
                label_any: Some(vec!["a".into(), "i".into()]),
                label_regex: None,
            },
            within: Some(Selector {
                tier: "words".into(),
                label_any: None,
                label_regex: Some("stress".into()),
            }),
            overlaps: None,
            where_expr: None,
            emit: Emit::Point { at: 0.5 },
            label: None,
        };
        let json = rule.to_json().unwrap();
        assert_eq!(CriterionRule::from_json(&json).unwrap(), rule);
    }

    #[test]
    fn emit_defaults_fill_in() {
        // A minimal rule: select all, emit the whole span.
        let rule =
            CriterionRule::from_json(r#"{"select":{"tier":"t"},"emit":{"kind":"span"}}"#).unwrap();
        assert_eq!(rule.emit, Emit::Span { from: 0.0, to: 1.0 });
        let rule =
            CriterionRule::from_json(r#"{"select":{"tier":"t"},"emit":{"kind":"point"}}"#).unwrap();
        assert_eq!(rule.emit, Emit::Point { at: 0.5 });
    }

    #[test]
    fn selects_by_label_set_and_regex() {
        let rows = [iv(0.0, 0.1, "a"), iv(0.1, 0.2, "b"), iv(0.2, 0.3, "i")];
        let rule = CriterionRule {
            select: Selector {
                tier: "phones".into(),
                label_any: None,
                label_regex: Some("^[aeiou]$".into()),
            },
            within: None,
            overlaps: None,
            where_expr: None,
            emit: Emit::Span { from: 0.0, to: 1.0 },
            label: None,
        };
        let out = evaluate(&rule, &rows, &[], &[], &no_sig()).unwrap();
        assert_eq!(out.len(), 2); // a, i (not b)
        assert_eq!(out[0].label.as_deref(), Some("a"));
        assert_eq!(out[1].label.as_deref(), Some("i"));
    }

    #[test]
    fn within_relation_filters_by_containment() {
        let phones = [iv(0.0, 0.1, "a"), iv(0.5, 0.6, "a")];
        let words = [iv(0.0, 0.2, "stressed")]; // only contains the first phone
        let rule = CriterionRule {
            select: Selector {
                tier: "phones".into(),
                label_any: Some(vec!["a".into()]),
                label_regex: None,
            },
            within: Some(Selector {
                tier: "words".into(),
                label_any: Some(vec!["stressed".into()]),
                label_regex: None,
            }),
            overlaps: None,
            where_expr: None,
            emit: Emit::Span { from: 0.0, to: 1.0 },
            label: None,
        };
        let out = evaluate(&rule, &phones, &words, &[], &no_sig()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].start, 0.0);
    }

    #[test]
    fn emits_subspan_and_anchor_point_by_proportion() {
        let rows = [iv(1.0, 2.0, "v")]; // 1s long
        let span = CriterionRule {
            select: Selector {
                tier: "t".into(),
                label_any: None,
                label_regex: None,
            },
            within: None,
            overlaps: None,
            where_expr: None,
            emit: Emit::Span { from: 0.2, to: 0.8 },
            label: Some("mid".into()),
        };
        let out = evaluate(&span, &rows, &[], &[], &no_sig()).unwrap();
        assert_eq!(out[0].start, 1.2);
        assert_eq!(out[0].end, Some(1.8));
        assert_eq!(out[0].label.as_deref(), Some("mid")); // fixed label overrides

        let point = CriterionRule {
            emit: Emit::Point { at: 0.5 },
            label: None,
            ..span
        };
        let out = evaluate(&point, &rows, &[], &[], &no_sig()).unwrap();
        assert_eq!(out[0].start, 1.5);
        assert_eq!(out[0].end, None);
        assert_eq!(out[0].label.as_deref(), Some("v")); // carried over
    }

    fn sampled(times: &[f64], values: &[f64]) -> expr::SampledSignal {
        expr::SampledSignal {
            times: times.to_vec(),
            values: values.to_vec(),
        }
    }

    #[test]
    fn signal_where_filter_and_expr_point_anchor() {
        let rows = [iv(0.0, 1.0, "a"), iv(1.0, 2.0, "a")];
        let mut signals = SignalSet::new();
        // intensity: quiet over the first interval, loud over the second.
        signals.insert("intensity".into(), sampled(&[0.5, 1.5], &[-30.0, -10.0]));
        let rule = CriterionRule::from_json(
            r#"{"select": {"tier": "phones", "label_any": ["a"]},
                "where": "mean(intensity) > -20",
                "emit": {"kind": "point_expr", "at": "argmax(intensity)"}}"#,
        )
        .unwrap();
        let out = evaluate(&rule, &rows, &[], &[], &signals).unwrap();
        // Only the loud (2nd) interval survives `where`; the point lands at the
        // intensity argmax inside it.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].start, 1.5);
        assert_eq!(out[0].end, None);
    }

    #[test]
    fn span_expr_with_crossing_endpoint() {
        let rows = [iv(0.0, 2.0, "v")];
        let mut signals = SignalSet::new();
        // f0 rises through 150 at t=0.5 (linear between (0,100) and (1,200)).
        signals.insert("f0".into(), sampled(&[0.0, 1.0], &[100.0, 200.0]));
        let rule = CriterionRule::from_json(
            r#"{"select": {"tier": "t"},
                "emit": {"kind": "span_expr",
                         "from": "start", "to": "first_crossing(f0, 150, rising)"}}"#,
        )
        .unwrap();
        let out = evaluate(&rule, &rows, &[], &[], &signals).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].start, 0.0);
        assert_eq!(out[0].end, Some(0.5));
    }

    #[test]
    fn undefined_anchor_skips_the_match() {
        // The match interval [5,6] has no f0 samples → argmax undefined → skip.
        let rows = [iv(5.0, 6.0, "v")];
        let mut signals = SignalSet::new();
        signals.insert("f0".into(), sampled(&[0.0, 1.0], &[100.0, 110.0]));
        let rule = CriterionRule::from_json(
            r#"{"select": {"tier": "t"},
                "emit": {"kind": "point_expr", "at": "argmax(f0)"}}"#,
        )
        .unwrap();
        assert!(
            evaluate(&rule, &rows, &[], &[], &signals)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn referenced_signals_lists_where_and_emit_signals() {
        let rule = CriterionRule::from_json(
            r#"{"select": {"tier": "t"}, "where": "mean(f0) > 100",
                "emit": {"kind": "point_expr", "at": "argmax(intensity)"}}"#,
        )
        .unwrap();
        assert_eq!(
            rule.referenced_signals().unwrap(),
            vec!["f0".to_string(), "intensity".to_string()]
        );
        // A pure S2 (proportion) rule references no signals.
        let s2 =
            CriterionRule::from_json(r#"{"select":{"tier":"t"},"emit":{"kind":"point"}}"#).unwrap();
        assert!(s2.referenced_signals().unwrap().is_empty());
    }

    #[test]
    fn snake_case_emit_tags_round_trip() {
        let rule = CriterionRule::from_json(
            r#"{"select":{"tier":"t"},"emit":{"kind":"span_expr","from":"start","to":"end"}}"#,
        )
        .unwrap();
        assert!(matches!(rule.emit, Emit::SpanExpr { .. }));
        // Re-serialize and ensure the tag stays snake_case.
        assert!(rule.to_json().unwrap().contains("span_expr"));
    }
}
