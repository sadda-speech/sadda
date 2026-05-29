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

/// What to emit for each matched interval, positioned by proportion (0..1)
/// within the match: a (sub-)span or an anchor point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
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
}

impl Emit {
    /// Whether this rule emits points (vs. spans) — drives the preview tier
    /// type.
    pub fn is_point(&self) -> bool {
        matches!(self, Emit::Point { .. })
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

/// Evaluates `rule` against pre-fetched intervals. `select_ivs` are the
/// select-tier intervals; `within_ivs` / `overlaps_ivs` are the relation
/// tiers' intervals (the caller fetches whichever the rule references and
/// passes `&[]` for unused ones). Returns the proposals in select order.
pub fn evaluate(
    rule: &CriterionRule,
    select_ivs: &[EvalInterval],
    within_ivs: &[EvalInterval],
    overlaps_ivs: &[EvalInterval],
) -> Result<Vec<Proposal>, String> {
    let select_m = rule.select.matcher()?;
    let within_m = rule.within.as_ref().map(Selector::matcher).transpose()?;
    let overlaps_m = rule.overlaps.as_ref().map(Selector::matcher).transpose()?;

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
            emit: Emit::Span { from: 0.0, to: 1.0 },
            label: None,
        };
        let out = evaluate(&rule, &rows, &[], &[]).unwrap();
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
            emit: Emit::Span { from: 0.0, to: 1.0 },
            label: None,
        };
        let out = evaluate(&rule, &phones, &words, &[]).unwrap();
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
            emit: Emit::Span { from: 0.2, to: 0.8 },
            label: Some("mid".into()),
        };
        let out = evaluate(&span, &rows, &[], &[]).unwrap();
        assert_eq!(out[0].start, 1.2);
        assert_eq!(out[0].end, Some(1.8));
        assert_eq!(out[0].label.as_deref(), Some("mid")); // fixed label overrides

        let point = CriterionRule {
            emit: Emit::Point { at: 0.5 },
            label: None,
            ..span
        };
        let out = evaluate(&point, &rows, &[], &[]).unwrap();
        assert_eq!(out[0].start, 1.5);
        assert_eq!(out[0].end, None);
        assert_eq!(out[0].label.as_deref(), Some("v")); // carried over
    }
}
