//! Flat **CSV** and structured **JSON** export of a bundle's sparse
//! annotations (intervals, points, references).
//!
//! Two shapes for two audiences:
//! - [`to_csv`] emits one *tidy* long table — one row per annotation across
//!   every sparse tier — the shape pandas / polars / R expect for stats. The
//!   column set is the union over the three sparse tier kinds; cells that
//!   don't apply to a row's kind are left empty (e.g. `time_seconds` on an
//!   interval, `start_seconds` on a reference).
//! - [`to_json`] emits a *faithful* nested document — bundle metadata plus a
//!   `tiers` array, each tier carrying only its native fields — so a consumer
//!   can reconstruct the per-tier structure that CSV flattens away.
//!
//! Dense tiers (`continuous_*`, `categorical_sampled`) are not handled here:
//! the caller skips them, matching the TextGrid / EAF exporters (their data
//! lives in Parquet sidecars, queryable via `Project.query`).

use crate::corpus::{Interval, Point, Reference};
use serde_json::{Map, Value, json};
use std::borrow::Cow;

/// Bundle identity + timing context stamped into both export formats.
pub struct ExportBundle {
    /// Bundle id.
    pub id: i64,
    /// Bundle name.
    pub name: String,
    /// Audio sample rate (Hz), used to derive `duration_seconds`.
    pub sample_rate: u32,
    /// Audio length in frames.
    pub n_frames: i64,
}

impl ExportBundle {
    /// Audio duration in seconds (`0.0` if the sample rate is unknown).
    pub fn duration_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.n_frames as f64 / self.sample_rate as f64
        }
    }
}

/// One sparse tier's rows, tagged by kind. Dense tiers never reach here.
pub struct ExportTier {
    /// Tier id.
    pub id: i64,
    /// Tier name.
    pub name: String,
    /// The tier's annotation rows.
    pub rows: TierRows,
}

/// The annotation rows of a sparse tier, one variant per sparse tier type.
pub enum TierRows {
    /// Interval-tier rows.
    Intervals(Vec<Interval>),
    /// Point-tier rows.
    Points(Vec<Point>),
    /// Reference-tier rows.
    References(Vec<Reference>),
}

impl TierRows {
    /// The `tier.type` string this variant corresponds to (matches
    /// [`crate::corpus::TierType::as_str`]).
    fn type_str(&self) -> &'static str {
        match self {
            TierRows::Intervals(_) => "interval",
            TierRows::Points(_) => "point",
            TierRows::References(_) => "reference",
        }
    }
}

/// The CSV header — also the column order. Kept in one place so the row
/// writers and the test can't drift from it.
pub const CSV_COLUMNS: &[&str] = &[
    "bundle_id",
    "bundle_name",
    "tier_id",
    "tier_name",
    "tier_type",
    "annotation_id",
    "start_seconds",
    "end_seconds",
    "time_seconds",
    "target_kind",
    "target_id",
    "label",
    "parent_annotation_id",
    "status",
    "note",
    "processing_run_id",
    "extra",
];

/// Serializes a bundle's sparse tiers to a tidy CSV string (RFC 4180,
/// `\r\n` line endings, a header row). One row per annotation; tiers are
/// emitted in the given order, rows in the order the caller supplies.
pub fn to_csv(bundle: &ExportBundle, tiers: &[ExportTier]) -> String {
    let mut out = String::new();
    write_csv_record(&mut out, CSV_COLUMNS.iter().map(|c| Cow::Borrowed(*c)));

    let bid = bundle.id.to_string();
    let bname = &bundle.name;
    for tier in tiers {
        let tid = tier.id.to_string();
        let ttype = tier.rows.type_str();
        match &tier.rows {
            TierRows::Intervals(rows) => {
                for r in rows {
                    write_csv_record(
                        &mut out,
                        [
                            Cow::Borrowed(bid.as_str()),
                            Cow::Borrowed(bname.as_str()),
                            Cow::Borrowed(tid.as_str()),
                            Cow::Borrowed(tier.name.as_str()),
                            Cow::Borrowed(ttype),
                            Cow::Owned(r.id.to_string()),
                            Cow::Owned(fmt_f64(r.start_seconds)),
                            Cow::Owned(fmt_f64(r.end_seconds)),
                            Cow::Borrowed(""), // time_seconds
                            Cow::Borrowed(""), // target_kind
                            Cow::Borrowed(""), // target_id
                            opt_str(&r.label),
                            opt_i64(r.parent_annotation_id),
                            opt_str(&r.status),
                            opt_str(&r.note),
                            opt_i64(r.processing_run_id),
                            opt_str(&r.extra),
                        ]
                        .into_iter(),
                    );
                }
            }
            TierRows::Points(rows) => {
                for r in rows {
                    write_csv_record(
                        &mut out,
                        [
                            Cow::Borrowed(bid.as_str()),
                            Cow::Borrowed(bname.as_str()),
                            Cow::Borrowed(tid.as_str()),
                            Cow::Borrowed(tier.name.as_str()),
                            Cow::Borrowed(ttype),
                            Cow::Owned(r.id.to_string()),
                            Cow::Borrowed(""), // start_seconds
                            Cow::Borrowed(""), // end_seconds
                            Cow::Owned(fmt_f64(r.time_seconds)),
                            Cow::Borrowed(""), // target_kind
                            Cow::Borrowed(""), // target_id
                            opt_str(&r.label),
                            opt_i64(r.parent_annotation_id),
                            opt_str(&r.status),
                            opt_str(&r.note),
                            opt_i64(r.processing_run_id),
                            opt_str(&r.extra),
                        ]
                        .into_iter(),
                    );
                }
            }
            TierRows::References(rows) => {
                for r in rows {
                    write_csv_record(
                        &mut out,
                        [
                            Cow::Borrowed(bid.as_str()),
                            Cow::Borrowed(bname.as_str()),
                            Cow::Borrowed(tid.as_str()),
                            Cow::Borrowed(tier.name.as_str()),
                            Cow::Borrowed(ttype),
                            Cow::Owned(r.id.to_string()),
                            Cow::Borrowed(""), // start_seconds
                            Cow::Borrowed(""), // end_seconds
                            Cow::Borrowed(""), // time_seconds
                            Cow::Borrowed(r.target_kind.as_str()),
                            Cow::Owned(r.target_id.to_string()),
                            opt_str(&r.label),
                            opt_i64(r.parent_annotation_id),
                            Cow::Borrowed(""), // status (references have none)
                            Cow::Borrowed(""), // note
                            Cow::Borrowed(""), // processing_run_id
                            opt_str(&r.extra),
                        ]
                        .into_iter(),
                    );
                }
            }
        }
    }
    out
}

/// Serializes a bundle's sparse tiers to a structured `serde_json::Value`
/// (`{ "bundle": {...}, "tiers": [...] }`). Each tier object carries its
/// type plus a kind-named array (`intervals` / `points` / `references`).
/// `extra` payloads are embedded as parsed JSON where they parse, else as
/// the raw string.
pub fn to_json(bundle: &ExportBundle, tiers: &[ExportTier]) -> Value {
    let tier_values: Vec<Value> = tiers
        .iter()
        .map(|tier| {
            let mut obj = Map::new();
            obj.insert("id".into(), json!(tier.id));
            obj.insert("name".into(), json!(tier.name));
            obj.insert("type".into(), json!(tier.rows.type_str()));
            match &tier.rows {
                TierRows::Intervals(rows) => {
                    obj.insert(
                        "intervals".into(),
                        Value::Array(rows.iter().map(interval_json).collect()),
                    );
                }
                TierRows::Points(rows) => {
                    obj.insert(
                        "points".into(),
                        Value::Array(rows.iter().map(point_json).collect()),
                    );
                }
                TierRows::References(rows) => {
                    obj.insert(
                        "references".into(),
                        Value::Array(rows.iter().map(reference_json).collect()),
                    );
                }
            }
            Value::Object(obj)
        })
        .collect();

    json!({
        "bundle": {
            "id": bundle.id,
            "name": bundle.name,
            "sample_rate": bundle.sample_rate,
            "n_frames": bundle.n_frames,
            "duration_seconds": bundle.duration_seconds(),
        },
        "tiers": tier_values,
    })
}

fn interval_json(r: &Interval) -> Value {
    json!({
        "id": r.id,
        "start_seconds": r.start_seconds,
        "end_seconds": r.end_seconds,
        "label": r.label,
        "parent_annotation_id": r.parent_annotation_id,
        "status": r.status,
        "note": r.note,
        "processing_run_id": r.processing_run_id,
        "extra": embed_extra(&r.extra),
    })
}

fn point_json(r: &Point) -> Value {
    json!({
        "id": r.id,
        "time_seconds": r.time_seconds,
        "label": r.label,
        "parent_annotation_id": r.parent_annotation_id,
        "status": r.status,
        "note": r.note,
        "processing_run_id": r.processing_run_id,
        "extra": embed_extra(&r.extra),
    })
}

fn reference_json(r: &Reference) -> Value {
    json!({
        "id": r.id,
        "target_kind": r.target_kind,
        "target_id": r.target_id,
        "label": r.label,
        "parent_annotation_id": r.parent_annotation_id,
        "extra": embed_extra(&r.extra),
    })
}

/// Embeds the DB's `extra` TEXT (itself JSON) as a parsed JSON value, so the
/// output nests cleanly instead of carrying an escaped string. Falls back to
/// the raw string if it isn't valid JSON, and to `null` if absent.
fn embed_extra(extra: &Option<String>) -> Value {
    match extra {
        None => Value::Null,
        Some(s) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone())),
    }
}

/// Writes one RFC 4180 record (fields joined by `,`, terminated by `\r\n`),
/// quoting each field as needed.
fn write_csv_record<'a>(out: &mut String, fields: impl Iterator<Item = Cow<'a, str>>) {
    let mut first = true;
    for f in fields {
        if !first {
            out.push(',');
        }
        first = false;
        push_csv_field(out, &f);
    }
    out.push_str("\r\n");
}

/// Appends a single CSV field, quoting per RFC 4180 when it contains a
/// comma, double-quote, CR, or LF (internal quotes are doubled).
fn push_csv_field(out: &mut String, field: &str) {
    let needs_quote = field
        .bytes()
        .any(|b| b == b',' || b == b'"' || b == b'\n' || b == b'\r');
    if !needs_quote {
        out.push_str(field);
        return;
    }
    out.push('"');
    for ch in field.chars() {
        if ch == '"' {
            out.push('"');
        }
        out.push(ch);
    }
    out.push('"');
}

/// Formats an `f64` for CSV. Rust's `f64` `Display` already yields the
/// shortest string that round-trips (e.g. `0.5`, `0.1`), which is what we
/// want for time columns.
fn fmt_f64(v: f64) -> String {
    v.to_string()
}

fn opt_str(v: &Option<String>) -> Cow<'_, str> {
    match v {
        Some(s) => Cow::Borrowed(s.as_str()),
        None => Cow::Borrowed(""),
    }
}

fn opt_i64<'a>(v: Option<i64>) -> Cow<'a, str> {
    match v {
        Some(n) => Cow::Owned(n.to_string()),
        None => Cow::Borrowed(""),
    }
}

// ───────────────────────── Import (parse) ──────────────────────────────
//
// The inverse direction. Both parsers yield the same [`ImportTier`] model,
// from which the caller creates fresh tiers + annotations. v1 imports only
// **interval** and **point** tiers; reference and dense tiers are skipped —
// references carry project-local `(target_kind, target_id)` pointers that
// don't port across projects, and dense data isn't sparse-annotation data.
// Identity / linkage columns that reference the *source* project
// (`annotation_id`, `parent_annotation_id`, `processing_run_id`) and the
// rubric-coupled `status` are intentionally dropped; the honoured fields are
// times, `label`, `note`, and `extra`.

/// A tier parsed from a CSV / JSON import: a name plus its rows.
pub struct ImportTier {
    /// Tier name (a new tier of this name is created in the target bundle).
    pub name: String,
    /// The tier's rows (also fixes its interval-vs-point type).
    pub rows: ImportRows,
}

/// Parsed rows for an imported tier — interval or point only.
pub enum ImportRows {
    /// Interval rows.
    Intervals(Vec<ImportInterval>),
    /// Point rows.
    Points(Vec<ImportPoint>),
}

/// One interval to create on import.
pub struct ImportInterval {
    /// Start time (seconds).
    pub start_seconds: f64,
    /// End time (seconds).
    pub end_seconds: f64,
    /// Label, or `None` for an empty cell.
    pub label: Option<String>,
    /// Free-text note, or `None`.
    pub note: Option<String>,
    /// JSON `extra` payload (as a string), or `None`.
    pub extra: Option<String>,
}

/// One point to create on import.
pub struct ImportPoint {
    /// Time (seconds).
    pub time_seconds: f64,
    /// Label, or `None` for an empty cell.
    pub label: Option<String>,
    /// Free-text note, or `None`.
    pub note: Option<String>,
    /// JSON `extra` payload (as a string), or `None`.
    pub extra: Option<String>,
}

/// Parses a flat CSV (as written by [`to_csv`]) back into tiers. Columns are
/// matched by header name, so reordered or extra columns are tolerated;
/// rows are grouped into tiers by `(tier_name, tier_type)` in first-seen
/// order. Reference / dense rows are skipped. Returns an error on a missing
/// required column or an unparseable time.
pub fn parse_csv(text: &str) -> Result<Vec<ImportTier>, String> {
    let records = parse_csv_records(text);
    let header = records
        .first()
        .ok_or_else(|| "CSV is empty (no header row)".to_string())?;
    let col = |name: &str| -> Result<usize, String> {
        header
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("CSV missing required column {name:?}"))
    };
    let (c_tname, c_ttype) = (col("tier_name")?, col("tier_type")?);
    let (c_start, c_end, c_time) = (
        col("start_seconds")?,
        col("end_seconds")?,
        col("time_seconds")?,
    );
    let (c_label, c_note, c_extra) = (col("label")?, col("note")?, col("extra")?);

    let get = |rec: &[String], i: usize| -> String { rec.get(i).cloned().unwrap_or_default() };
    let opt = |s: String| -> Option<String> { if s.is_empty() { None } else { Some(s) } };

    let mut builder = TierBuilder::default();
    for rec in records.iter().skip(1) {
        let tier_type = get(rec, c_ttype);
        let tier_name = get(rec, c_tname);
        match tier_type.as_str() {
            "interval" => {
                let start = parse_seconds(&get(rec, c_start), "start_seconds")?;
                let end = parse_seconds(&get(rec, c_end), "end_seconds")?;
                builder.interval(
                    &tier_name,
                    ImportInterval {
                        start_seconds: start,
                        end_seconds: end,
                        label: opt(get(rec, c_label)),
                        note: opt(get(rec, c_note)),
                        extra: opt(get(rec, c_extra)),
                    },
                )?;
            }
            "point" => {
                let time = parse_seconds(&get(rec, c_time), "time_seconds")?;
                builder.point(
                    &tier_name,
                    ImportPoint {
                        time_seconds: time,
                        label: opt(get(rec, c_label)),
                        note: opt(get(rec, c_note)),
                        extra: opt(get(rec, c_extra)),
                    },
                )?;
            }
            // reference / continuous_* / categorical_* / unknown: skipped.
            _ => {}
        }
    }
    Ok(builder.finish())
}

/// Parses a structured JSON document (as written by [`to_json`]) back into
/// tiers. Walks `tiers[]`, taking each tier's `type` and its kind-named
/// array; reference / dense tiers are skipped. An `extra` value that is a
/// JSON object/array is re-serialized to a string for the DB column.
pub fn parse_json(value: &Value) -> Result<Vec<ImportTier>, String> {
    let tiers = value
        .get("tiers")
        .and_then(Value::as_array)
        .ok_or_else(|| "JSON missing a `tiers` array".to_string())?;

    let mut out = Vec::new();
    for tier in tiers {
        let name = tier
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        match tier.get("type").and_then(Value::as_str) {
            Some("interval") => {
                let rows = tier.get("intervals").and_then(Value::as_array);
                let mut ivs = Vec::new();
                for r in rows.into_iter().flatten() {
                    ivs.push(ImportInterval {
                        start_seconds: json_seconds(r, "start_seconds")?,
                        end_seconds: json_seconds(r, "end_seconds")?,
                        label: json_opt_str(r, "label"),
                        note: json_opt_str(r, "note"),
                        extra: json_extra(r),
                    });
                }
                out.push(ImportTier {
                    name,
                    rows: ImportRows::Intervals(ivs),
                });
            }
            Some("point") => {
                let rows = tier.get("points").and_then(Value::as_array);
                let mut pts = Vec::new();
                for r in rows.into_iter().flatten() {
                    pts.push(ImportPoint {
                        time_seconds: json_seconds(r, "time_seconds")?,
                        label: json_opt_str(r, "label"),
                        note: json_opt_str(r, "note"),
                        extra: json_extra(r),
                    });
                }
                out.push(ImportTier {
                    name,
                    rows: ImportRows::Points(pts),
                });
            }
            // reference / dense / unknown: skipped.
            _ => {}
        }
    }
    Ok(out)
}

/// Accumulates rows into tiers keyed by `(name, kind)`, preserving the order
/// tiers are first seen. A name reused with a different kind makes a separate
/// tier (interval `f` and point `f` are distinct).
#[derive(Default)]
struct TierBuilder {
    order: Vec<(String, bool)>, // (name, is_interval)
    intervals: std::collections::HashMap<String, Vec<ImportInterval>>,
    points: std::collections::HashMap<String, Vec<ImportPoint>>,
}

impl TierBuilder {
    fn interval(&mut self, name: &str, row: ImportInterval) -> Result<(), String> {
        let key = name.to_string();
        if !self.intervals.contains_key(&key) {
            self.order.push((key.clone(), true));
        }
        self.intervals.entry(key).or_default().push(row);
        Ok(())
    }

    fn point(&mut self, name: &str, row: ImportPoint) -> Result<(), String> {
        let key = name.to_string();
        if !self.points.contains_key(&key) {
            self.order.push((key.clone(), false));
        }
        self.points.entry(key).or_default().push(row);
        Ok(())
    }

    fn finish(mut self) -> Vec<ImportTier> {
        let mut out = Vec::with_capacity(self.order.len());
        for (name, is_interval) in &self.order {
            if *is_interval {
                if let Some(rows) = self.intervals.remove(name) {
                    out.push(ImportTier {
                        name: name.clone(),
                        rows: ImportRows::Intervals(rows),
                    });
                }
            } else if let Some(rows) = self.points.remove(name) {
                out.push(ImportTier {
                    name: name.clone(),
                    rows: ImportRows::Points(rows),
                });
            }
        }
        out
    }
}

fn parse_seconds(s: &str, field: &str) -> Result<f64, String> {
    s.parse::<f64>()
        .map_err(|_| format!("CSV {field} is not a number: {s:?}"))
}

fn json_seconds(row: &Value, field: &str) -> Result<f64, String> {
    row.get(field)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("JSON annotation missing numeric {field:?}"))
}

fn json_opt_str(row: &Value, field: &str) -> Option<String> {
    match row.get(field) {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// Recovers the DB `extra` TEXT from a JSON row: a string passes through; an
/// object/array is re-serialized; `null`/absent → `None`.
fn json_extra(row: &Value) -> Option<String> {
    match row.get("extra") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if s.is_empty() => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(v) => serde_json::to_string(v).ok(),
    }
}

/// Splits CSV text into records of fields per RFC 4180: handles quoted
/// fields containing commas, doubled quotes, and embedded CR/LF; accepts
/// `\n` or `\r\n` line endings; skips fully-blank lines.
fn parse_csv_records(text: &str) -> Vec<Vec<String>> {
    let mut records = Vec::new();
    let mut record: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = text.chars().peekable();

    let flush_record =
        |record: &mut Vec<String>, field: &mut String, records: &mut Vec<Vec<String>>| {
            record.push(std::mem::take(field));
            // Skip a blank line (a lone empty field).
            if !(record.len() == 1 && record[0].is_empty()) {
                records.push(std::mem::take(record));
            } else {
                record.clear();
            }
        };

    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                field.push(c);
            }
        } else {
            match c {
                '"' => in_quotes = true,
                ',' => record.push(std::mem::take(&mut field)),
                '\r' => {
                    if chars.peek() == Some(&'\n') {
                        chars.next();
                    }
                    flush_record(&mut record, &mut field, &mut records);
                }
                '\n' => flush_record(&mut record, &mut field, &mut records),
                _ => field.push(c),
            }
        }
    }
    // Flush a trailing record with no final newline.
    if !field.is_empty() || !record.is_empty() {
        flush_record(&mut record, &mut field, &mut records);
    }
    records
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(id: i64, start: f64, end: f64, label: Option<&str>) -> Interval {
        Interval {
            id,
            tier_id: 10,
            start_seconds: start,
            end_seconds: end,
            label: label.map(String::from),
            parent_annotation_id: None,
            status: None,
            note: None,
            processing_run_id: None,
            extra: None,
        }
    }

    fn point(id: i64, t: f64, label: Option<&str>) -> Point {
        Point {
            id,
            tier_id: 11,
            time_seconds: t,
            label: label.map(String::from),
            parent_annotation_id: None,
            status: None,
            note: None,
            processing_run_id: None,
            extra: None,
        }
    }

    fn bundle() -> ExportBundle {
        ExportBundle {
            id: 1,
            name: "rec01".into(),
            sample_rate: 16_000,
            n_frames: 32_000,
        }
    }

    #[test]
    fn csv_header_matches_column_count() {
        let csv = to_csv(&bundle(), &[]);
        let header = csv.lines().next().unwrap();
        assert_eq!(header, CSV_COLUMNS.join(","));
        assert_eq!(header.split(',').count(), CSV_COLUMNS.len());
        // Empty export is header-only.
        assert_eq!(csv.lines().count(), 1);
    }

    #[test]
    fn csv_interval_and_point_rows_fill_the_right_time_columns() {
        let tiers = vec![
            ExportTier {
                id: 10,
                name: "words".into(),
                rows: TierRows::Intervals(vec![interval(1, 0.0, 0.5, Some("hi"))]),
            },
            ExportTier {
                id: 11,
                name: "pulses".into(),
                rows: TierRows::Points(vec![point(2, 0.25, Some("p"))]),
            },
        ];
        let csv = to_csv(&bundle(), &tiers);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 rows

        // interval: start/end set, time_seconds empty (col index 8).
        let iv: Vec<&str> = lines[1].split(',').collect();
        assert_eq!(iv[4], "interval");
        assert_eq!(iv[6], "0"); // start_seconds
        assert_eq!(iv[7], "0.5"); // end_seconds
        assert_eq!(iv[8], ""); // time_seconds
        assert_eq!(iv[11], "hi"); // label

        // point: time set, start/end empty.
        let pt: Vec<&str> = lines[2].split(',').collect();
        assert_eq!(pt[4], "point");
        assert_eq!(pt[6], ""); // start_seconds
        assert_eq!(pt[7], ""); // end_seconds
        assert_eq!(pt[8], "0.25"); // time_seconds
    }

    #[test]
    fn csv_quotes_fields_with_commas_quotes_and_newlines() {
        let mut iv = interval(1, 0.0, 1.0, Some("a, b \"c\"\nd"));
        iv.note = Some("plain".into());
        let tiers = vec![ExportTier {
            id: 10,
            name: "t".into(),
            rows: TierRows::Intervals(vec![iv]),
        }];
        let csv = to_csv(&bundle(), &tiers);
        // The label cell is quoted and its inner quotes doubled; the embedded
        // newline stays inside the quoted field.
        assert!(csv.contains("\"a, b \"\"c\"\"\nd\""));
        // A field with no special chars is left bare.
        assert!(csv.contains(",plain,"));
    }

    #[test]
    fn json_groups_rows_under_typed_tiers() {
        let refs = vec![Reference {
            id: 7,
            tier_id: 12,
            target_kind: "annotation".into(),
            target_id: 99,
            label: Some("link".into()),
            parent_annotation_id: None,
            extra: None,
        }];
        let tiers = vec![
            ExportTier {
                id: 10,
                name: "words".into(),
                rows: TierRows::Intervals(vec![interval(1, 0.0, 0.5, Some("hi"))]),
            },
            ExportTier {
                id: 12,
                name: "links".into(),
                rows: TierRows::References(refs),
            },
        ];
        let v = to_json(&bundle(), &tiers);
        assert_eq!(v["bundle"]["name"], "rec01");
        assert_eq!(v["bundle"]["duration_seconds"], 2.0);
        assert_eq!(v["tiers"][0]["type"], "interval");
        assert_eq!(v["tiers"][0]["intervals"][0]["label"], "hi");
        assert_eq!(v["tiers"][1]["type"], "reference");
        assert_eq!(v["tiers"][1]["references"][0]["target_id"], 99);
    }

    #[test]
    fn json_embeds_extra_as_parsed_json() {
        let mut iv = interval(1, 0.0, 0.5, Some("hi"));
        iv.extra = Some(r#"{"f0":120.5}"#.into());
        let tiers = vec![ExportTier {
            id: 10,
            name: "words".into(),
            rows: TierRows::Intervals(vec![iv]),
        }];
        let v = to_json(&bundle(), &tiers);
        // Embedded as an object, not an escaped string.
        assert_eq!(v["tiers"][0]["intervals"][0]["extra"]["f0"], 120.5);
    }

    #[test]
    fn csv_record_parser_handles_quotes_and_embedded_newlines() {
        let text = "a,b,c\r\n1,\"x, y\",\"line1\nline2\"\r\n\"he said \"\"hi\"\"\",2,3\r\n";
        let recs = parse_csv_records(text);
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[1], vec!["1", "x, y", "line1\nline2"]);
        assert_eq!(recs[2], vec!["he said \"hi\"", "2", "3"]);
    }

    #[test]
    fn parse_csv_matches_columns_by_name_and_groups_tiers() {
        // Deliberately reordered + extra column to prove name-based matching.
        let text = "tier_type,tier_name,label,start_seconds,end_seconds,time_seconds,note,extra,junk\n\
                    interval,words,hi,0,0.5,,n1,,z\n\
                    point,pulses,p,,,0.25,,,z\n\
                    reference,links,r,,,,,,z\n";
        let tiers = parse_csv(text).unwrap();
        // reference tier dropped → 2 tiers.
        assert_eq!(tiers.len(), 2);
        assert_eq!(tiers[0].name, "words");
        match &tiers[0].rows {
            ImportRows::Intervals(rows) => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].start_seconds, 0.0);
                assert_eq!(rows[0].end_seconds, 0.5);
                assert_eq!(rows[0].label.as_deref(), Some("hi"));
                assert_eq!(rows[0].note.as_deref(), Some("n1"));
            }
            _ => panic!("expected interval tier"),
        }
        match &tiers[1].rows {
            ImportRows::Points(rows) => assert_eq!(rows[0].time_seconds, 0.25),
            _ => panic!("expected point tier"),
        }
    }

    #[test]
    fn csv_round_trips_through_export_then_parse() {
        let mut iv = interval(1, 0.0, 0.5, Some("a, b\n\"c\""));
        iv.note = Some("note".into());
        iv.extra = Some(r#"{"k":1}"#.into());
        let tiers = vec![
            ExportTier {
                id: 10,
                name: "words".into(),
                rows: TierRows::Intervals(vec![iv]),
            },
            ExportTier {
                id: 11,
                name: "pulses".into(),
                rows: TierRows::Points(vec![point(2, 0.25, Some("p"))]),
            },
        ];
        let csv = to_csv(&bundle(), &tiers);
        let parsed = parse_csv(&csv).unwrap();
        assert_eq!(parsed.len(), 2);
        match &parsed[0].rows {
            ImportRows::Intervals(rows) => {
                // The tricky label survives the quote round-trip intact.
                assert_eq!(rows[0].label.as_deref(), Some("a, b\n\"c\""));
                assert_eq!(rows[0].extra.as_deref(), Some(r#"{"k":1}"#));
            }
            _ => panic!("expected interval tier"),
        }
    }

    #[test]
    fn parse_json_round_trips_and_skips_references() {
        let refs = vec![Reference {
            id: 7,
            tier_id: 12,
            target_kind: "annotation".into(),
            target_id: 99,
            label: Some("link".into()),
            parent_annotation_id: None,
            extra: None,
        }];
        let mut iv = interval(1, 0.0, 0.5, Some("hi"));
        iv.extra = Some(r#"{"f0":120.5}"#.into());
        let tiers = vec![
            ExportTier {
                id: 10,
                name: "words".into(),
                rows: TierRows::Intervals(vec![iv]),
            },
            ExportTier {
                id: 12,
                name: "links".into(),
                rows: TierRows::References(refs),
            },
        ];
        let v = to_json(&bundle(), &tiers);
        let parsed = parse_json(&v).unwrap();
        // reference tier skipped on import.
        assert_eq!(parsed.len(), 1);
        match &parsed[0].rows {
            ImportRows::Intervals(rows) => {
                assert_eq!(rows[0].label.as_deref(), Some("hi"));
                // extra recovered from the embedded object back to a string.
                assert_eq!(rows[0].extra.as_deref(), Some(r#"{"f0":120.5}"#));
            }
            _ => panic!("expected interval tier"),
        }
    }
}
