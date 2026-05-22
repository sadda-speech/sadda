//! Praat TextGrid round-trip: read both long-text and short-text formats,
//! write long-text. UTF-8 only at v1.
//!
//! ## References
//! - Praat TextGrid file format manual:
//!   <https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html>
//! - praatio (Python TextGrid library, API-shape precedent):
//!   <https://github.com/timmahrt/praatIO>
//!
//! ## JSON sentinel
//!
//! Our annotation rows carry richer metadata than Praat's TextGrid stores
//! (a freeform `extra` JSON payload; reference-tier `(target_kind,
//! target_id)`). To round-trip the metadata through a Praat-edited
//! TextGrid we append a JSON sentinel to the label text:
//!
//! ```text
//! <label> {json:<inline-json>}
//! ```
//!
//! Plain labels with no sentinel round-trip cleanly. The sentinel is
//! recovered via a non-greedy split on ` {json:` at the end of the label.
//!
//! ## Deferred alternates / variants
//! - Binary TextGrid format (task TBD; rare in research workflows).
//! - UTF-16 encoding (some old Praat files; UTF-8 is the modern default).
//! - Lossy report on export (count of skipped dense tiers, lost hierarchy).

use std::fs;
use std::io::Write;
use std::path::Path;

use crate::error::{EngineError, Result};

/// In-memory representation of a parsed TextGrid file.
#[derive(Debug, Clone)]
pub struct TextGridFile {
    /// File-level start time in seconds.
    pub xmin: f64,
    /// File-level end time in seconds.
    pub xmax: f64,
    /// One entry per Praat tier, in file order.
    pub tiers: Vec<TextGridTier>,
}

/// One Praat tier — either an `IntervalTier` or a `TextTier` (point tier).
#[derive(Debug, Clone)]
pub enum TextGridTier {
    /// `IntervalTier` — contiguous (start, end, label) entries.
    Interval(IntervalTier),
    /// `TextTier` — sparse (time, label) entries.
    Point(PointTier),
}

impl TextGridTier {
    /// Name shared by both variants.
    pub fn name(&self) -> &str {
        match self {
            TextGridTier::Interval(t) => &t.name,
            TextGridTier::Point(t) => &t.name,
        }
    }
}

/// An `IntervalTier`.
#[derive(Debug, Clone)]
pub struct IntervalTier {
    /// Tier name.
    pub name: String,
    /// Tier-level start time.
    pub xmin: f64,
    /// Tier-level end time.
    pub xmax: f64,
    /// Contiguous intervals, sorted by `xmin` (Praat enforces; we don't
    /// re-validate on read but the writer guarantees it on export).
    pub intervals: Vec<IntervalEntry>,
}

/// One interval row.
#[derive(Debug, Clone)]
pub struct IntervalEntry {
    /// Interval start time in seconds.
    pub xmin: f64,
    /// Interval end time in seconds.
    pub xmax: f64,
    /// Label text (Praat allows the empty string `""` to mean silence /
    /// gap; we preserve it verbatim).
    pub text: String,
}

/// A `TextTier` (Praat's name for what we call a point tier).
#[derive(Debug, Clone)]
pub struct PointTier {
    /// Tier name.
    pub name: String,
    /// Tier-level start time.
    pub xmin: f64,
    /// Tier-level end time.
    pub xmax: f64,
    /// Per-event points, sorted by `time`.
    pub points: Vec<PointEntry>,
}

/// One point row.
#[derive(Debug, Clone)]
pub struct PointEntry {
    /// Event time in seconds.
    pub time: f64,
    /// Point label (`mark` in Praat's terminology).
    pub mark: String,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Reads a TextGrid file from disk. Auto-detects long-text vs short-text.
pub fn read(path: &Path) -> Result<TextGridFile> {
    let raw = fs::read_to_string(path).map_err(EngineError::Io)?;
    parse(&raw)
}

/// Parses an in-memory TextGrid string. Auto-detects long vs short text.
pub fn parse(input: &str) -> Result<TextGridFile> {
    let tokens = tokenize(input)?;
    let mut walker = TokenWalker::new(tokens);

    // Header: two strings "ooTextFile" and "TextGrid".
    let file_type = walker.next_string()?;
    if file_type != "ooTextFile" {
        return Err(EngineError::Corpus(format!(
            "TextGrid: expected File type=\"ooTextFile\", got {file_type:?}"
        )));
    }
    let object_class = walker.next_string()?;
    if object_class != "TextGrid" {
        return Err(EngineError::Corpus(format!(
            "TextGrid: expected Object class=\"TextGrid\", got {object_class:?}"
        )));
    }

    let xmin = walker.next_number()?;
    let xmax = walker.next_number()?;

    // Optional <exists> / <absent> marker.
    let has_tiers = match walker.peek() {
        Some(Token::Exists) => {
            walker.advance();
            true
        }
        Some(Token::Absent) => {
            walker.advance();
            return Ok(TextGridFile {
                xmin,
                xmax,
                tiers: Vec::new(),
            });
        }
        _ => true, // some writers omit the marker
    };
    if !has_tiers {
        return Ok(TextGridFile {
            xmin,
            xmax,
            tiers: Vec::new(),
        });
    }

    let n_tiers = walker.next_number()? as usize;
    let mut tiers = Vec::with_capacity(n_tiers);
    for _ in 0..n_tiers {
        let class = walker.next_string()?;
        let name = walker.next_string()?;
        let tier_xmin = walker.next_number()?;
        let tier_xmax = walker.next_number()?;
        let n_items = walker.next_number()? as usize;
        match class.as_str() {
            "IntervalTier" => {
                let mut intervals = Vec::with_capacity(n_items);
                for _ in 0..n_items {
                    let i_xmin = walker.next_number()?;
                    let i_xmax = walker.next_number()?;
                    let text = walker.next_string()?;
                    intervals.push(IntervalEntry {
                        xmin: i_xmin,
                        xmax: i_xmax,
                        text,
                    });
                }
                tiers.push(TextGridTier::Interval(IntervalTier {
                    name,
                    xmin: tier_xmin,
                    xmax: tier_xmax,
                    intervals,
                }));
            }
            "TextTier" => {
                let mut points = Vec::with_capacity(n_items);
                for _ in 0..n_items {
                    let time = walker.next_number()?;
                    let mark = walker.next_string()?;
                    points.push(PointEntry { time, mark });
                }
                tiers.push(TextGridTier::Point(PointTier {
                    name,
                    xmin: tier_xmin,
                    xmax: tier_xmax,
                    points,
                }));
            }
            other => {
                return Err(EngineError::Corpus(format!(
                    "TextGrid: unknown tier class {other:?}"
                )));
            }
        }
    }

    Ok(TextGridFile { xmin, xmax, tiers })
}

#[derive(Debug, Clone)]
enum Token {
    Number(f64),
    String(String),
    Exists,
    Absent,
}

struct TokenWalker {
    tokens: Vec<Token>,
    pos: usize,
}

impl TokenWalker {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }
    fn advance(&mut self) {
        self.pos += 1;
    }
    fn next(&mut self) -> Result<Token> {
        let t = self
            .tokens
            .get(self.pos)
            .cloned()
            .ok_or_else(|| EngineError::Corpus("TextGrid: unexpected end of input".into()))?;
        self.pos += 1;
        Ok(t)
    }
    fn next_string(&mut self) -> Result<String> {
        match self.next()? {
            Token::String(s) => Ok(s),
            other => Err(EngineError::Corpus(format!(
                "TextGrid: expected string, got {other:?}"
            ))),
        }
    }
    fn next_number(&mut self) -> Result<f64> {
        match self.next()? {
            Token::Number(n) => Ok(n),
            other => Err(EngineError::Corpus(format!(
                "TextGrid: expected number, got {other:?}"
            ))),
        }
    }
}

/// Tokenises a TextGrid source: extracts quoted strings, numbers, and the
/// `<exists>` / `<absent>` markers; ignores every keyword (`xmin`, `=`,
/// `item`, etc.) — they're noise for our parser because both long and short
/// formats share the same value sequence.
fn tokenize(input: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            // Skip whitespace.
            b' ' | b'\t' | b'\r' | b'\n' => {
                i += 1;
            }
            // Quoted string with Praat-style doubled-quote escaping.
            b'"' => {
                i += 1;
                let mut s = String::new();
                while i < bytes.len() {
                    if bytes[i] == b'"' {
                        // Lookahead for doubled quote.
                        if i + 1 < bytes.len() && bytes[i + 1] == b'"' {
                            s.push('"');
                            i += 2;
                            continue;
                        }
                        i += 1;
                        break;
                    }
                    s.push(bytes[i] as char);
                    i += 1;
                }
                tokens.push(Token::String(s));
            }
            // Praat positional-index `[<digits>]:` — skip as a unit so the
            // index integer isn't tokenized as a number.
            b'[' => {
                let mut end = i + 1;
                while end < bytes.len() && bytes[end].is_ascii_digit() {
                    end += 1;
                }
                if end > i + 1 && end < bytes.len() && bytes[end] == b']' {
                    i = end + 1;
                } else {
                    i += 1; // bare `[` — treat as noise
                }
            }
            // <exists> or <absent> marker.
            b'<' => {
                let rest = &input[i..];
                if rest.starts_with("<exists>") {
                    tokens.push(Token::Exists);
                    i += "<exists>".len();
                } else if rest.starts_with("<absent>") {
                    tokens.push(Token::Absent);
                    i += "<absent>".len();
                } else {
                    // Unknown marker — skip to the next whitespace.
                    while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                        i += 1;
                    }
                }
            }
            // Number (signed or unsigned; may have decimal point or exponent).
            b'-' | b'+' | b'0'..=b'9' | b'.' => {
                let start = i;
                if bytes[i] == b'-' || bytes[i] == b'+' {
                    i += 1;
                }
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                    i += 1;
                }
                if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                    i += 1;
                    if i < bytes.len() && (bytes[i] == b'-' || bytes[i] == b'+') {
                        i += 1;
                    }
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let raw = &input[start..i];
                // Only treat as a number if it parses; otherwise it's a key
                // like `123_invalid` — fall through.
                if let Ok(n) = raw.parse::<f64>() {
                    tokens.push(Token::Number(n));
                } else if raw == "-" || raw == "+" || raw == "." {
                    // Pathological single-character sign / dot — skip.
                }
                // If parse failed for some other reason, it's a noise key —
                // already consumed, move on.
            }
            // Anything else (keywords, equals signs, brackets, colons,
            // question marks) is noise.
            _ => {
                while i < bytes.len() {
                    let b = bytes[i];
                    if b == b'"'
                        || b == b'<'
                        || b.is_ascii_whitespace()
                        || b == b'-'
                        || b == b'+'
                        || b.is_ascii_digit()
                        || b == b'.'
                    {
                        break;
                    }
                    i += 1;
                }
            }
        }
    }
    Ok(tokens)
}

// ---------------------------------------------------------------------------
// Writer (long text only)
// ---------------------------------------------------------------------------

/// Writes a TextGrid to disk in long-text format (Praat's default).
pub fn write(file: &TextGridFile, path: &Path) -> Result<()> {
    let mut f = std::fs::File::create(path)?;
    write_to(file, &mut f)
}

/// Writes a TextGrid to any `Write` sink in long-text format.
pub fn write_to<W: Write>(file: &TextGridFile, sink: &mut W) -> Result<()> {
    writeln!(sink, "File type = \"ooTextFile\"")?;
    writeln!(sink, "Object class = \"TextGrid\"")?;
    writeln!(sink)?;
    writeln!(sink, "xmin = {}", format_number(file.xmin))?;
    writeln!(sink, "xmax = {}", format_number(file.xmax))?;
    if file.tiers.is_empty() {
        writeln!(sink, "tiers? <absent>")?;
        return Ok(());
    }
    writeln!(sink, "tiers? <exists>")?;
    writeln!(sink, "size = {}", file.tiers.len())?;
    writeln!(sink, "item []:")?;
    for (i, tier) in file.tiers.iter().enumerate() {
        let idx = i + 1; // Praat is 1-indexed
        writeln!(sink, "    item [{idx}]:")?;
        match tier {
            TextGridTier::Interval(t) => write_interval_tier(sink, t)?,
            TextGridTier::Point(t) => write_point_tier(sink, t)?,
        }
    }
    Ok(())
}

fn write_interval_tier<W: Write>(sink: &mut W, t: &IntervalTier) -> Result<()> {
    writeln!(sink, "        class = \"IntervalTier\"")?;
    writeln!(sink, "        name = \"{}\"", escape_string(&t.name))?;
    writeln!(sink, "        xmin = {}", format_number(t.xmin))?;
    writeln!(sink, "        xmax = {}", format_number(t.xmax))?;
    writeln!(sink, "        intervals: size = {}", t.intervals.len())?;
    for (i, iv) in t.intervals.iter().enumerate() {
        let idx = i + 1;
        writeln!(sink, "        intervals [{idx}]:")?;
        writeln!(sink, "            xmin = {}", format_number(iv.xmin))?;
        writeln!(sink, "            xmax = {}", format_number(iv.xmax))?;
        writeln!(sink, "            text = \"{}\"", escape_string(&iv.text))?;
    }
    Ok(())
}

fn write_point_tier<W: Write>(sink: &mut W, t: &PointTier) -> Result<()> {
    writeln!(sink, "        class = \"TextTier\"")?;
    writeln!(sink, "        name = \"{}\"", escape_string(&t.name))?;
    writeln!(sink, "        xmin = {}", format_number(t.xmin))?;
    writeln!(sink, "        xmax = {}", format_number(t.xmax))?;
    writeln!(sink, "        points: size = {}", t.points.len())?;
    for (i, p) in t.points.iter().enumerate() {
        let idx = i + 1;
        writeln!(sink, "        points [{idx}]:")?;
        writeln!(sink, "            number = {}", format_number(p.time))?;
        writeln!(sink, "            mark = \"{}\"", escape_string(&p.mark))?;
    }
    Ok(())
}

fn format_number(n: f64) -> String {
    // Praat-style: integers render as integers, fractions render with a
    // reasonable number of significant digits and no trailing zeros.
    if n == n.trunc() && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        // Format with up to 17 significant digits (f64's round-trip width),
        // then strip trailing zeros after a decimal point.
        let s = format!("{n:.17}");
        let trimmed = s.trim_end_matches('0').trim_end_matches('.').to_string();
        if trimmed.is_empty() || trimmed == "-" {
            "0".to_string()
        } else {
            trimmed
        }
    }
}

/// Escapes a label for writing inside double quotes: doubles up any `"`
/// (Praat's convention).
fn escape_string(s: &str) -> String {
    s.replace('"', "\"\"")
}

// ---------------------------------------------------------------------------
// JSON-sentinel codec
// ---------------------------------------------------------------------------

const SENTINEL_PREFIX: &str = " {json:";
const SENTINEL_SUFFIX: &str = "}";

/// Appends a JSON sentinel to a label. `<label> {json:<inline-json>}`.
/// Returns `label` unchanged if `extra_json` is `None` or empty.
pub fn encode_label(label: &str, extra_json: Option<&str>) -> String {
    match extra_json {
        Some(json) if !json.is_empty() => {
            format!("{label}{SENTINEL_PREFIX}{json}{SENTINEL_SUFFIX}")
        }
        _ => label.to_string(),
    }
}

/// Strips the trailing JSON sentinel from a label, returning
/// `(plain_label, extra_json)`. If no sentinel is present, returns
/// `(label, None)`.
pub fn decode_label(text: &str) -> (String, Option<String>) {
    if let Some(idx) = text.rfind(SENTINEL_PREFIX) {
        if text.ends_with(SENTINEL_SUFFIX) {
            let plain = &text[..idx];
            let json_start = idx + SENTINEL_PREFIX.len();
            let json_end = text.len() - SENTINEL_SUFFIX.len();
            if json_end >= json_start {
                let json = &text[json_start..json_end];
                return (plain.to_string(), Some(json.to_string()));
            }
        }
    }
    (text.to_string(), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG_FIXTURE: &str = r#"File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2.5
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 2.5
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 1.0
            text = "h"
        intervals [2]:
            xmin = 1.0
            xmax = 2.0
            text = "e"
        intervals [3]:
            xmin = 2.0
            xmax = 2.5
            text = "l"
    item [2]:
        class = "TextTier"
        name = "events"
        xmin = 0
        xmax = 2.5
        points: size = 2
        points [1]:
            number = 0.5
            mark = "click"
        points [2]:
            number = 1.5
            mark = "release"
"#;

    const SHORT_FIXTURE: &str = r#"File type = "ooTextFile"
Object class = "TextGrid"

0
2.5
<exists>
2
"IntervalTier"
"phones"
0
2.5
3
0
1.0
"h"
1.0
2.0
"e"
2.0
2.5
"l"
"TextTier"
"events"
0
2.5
2
0.5
"click"
1.5
"release"
"#;

    #[test]
    fn parse_long_format_recovers_structure() {
        let tg = parse(LONG_FIXTURE).unwrap();
        assert_eq!(tg.xmin, 0.0);
        assert_eq!(tg.xmax, 2.5);
        assert_eq!(tg.tiers.len(), 2);
        match &tg.tiers[0] {
            TextGridTier::Interval(t) => {
                assert_eq!(t.name, "phones");
                assert_eq!(t.intervals.len(), 3);
                assert_eq!(t.intervals[0].text, "h");
                assert_eq!(t.intervals[1].xmin, 1.0);
                assert_eq!(t.intervals[1].xmax, 2.0);
            }
            _ => panic!("tier 0 should be IntervalTier"),
        }
        match &tg.tiers[1] {
            TextGridTier::Point(t) => {
                assert_eq!(t.name, "events");
                assert_eq!(t.points.len(), 2);
                assert_eq!(t.points[0].time, 0.5);
                assert_eq!(t.points[1].mark, "release");
            }
            _ => panic!("tier 1 should be PointTier"),
        }
    }

    #[test]
    fn parse_short_format_recovers_same_structure() {
        let long_tg = parse(LONG_FIXTURE).unwrap();
        let short_tg = parse(SHORT_FIXTURE).unwrap();
        assert_eq!(long_tg.tiers.len(), short_tg.tiers.len());
        // Spot-check equivalence on tier 0.
        if let (TextGridTier::Interval(a), TextGridTier::Interval(b)) =
            (&long_tg.tiers[0], &short_tg.tiers[0])
        {
            assert_eq!(a.name, b.name);
            assert_eq!(a.intervals.len(), b.intervals.len());
            for (ai, bi) in a.intervals.iter().zip(b.intervals.iter()) {
                assert_eq!(ai.xmin, bi.xmin);
                assert_eq!(ai.xmax, bi.xmax);
                assert_eq!(ai.text, bi.text);
            }
        } else {
            panic!("tier 0 type mismatch");
        }
    }

    #[test]
    fn round_trip_long_format_is_stable() {
        let original = parse(LONG_FIXTURE).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        write_to(&original, &mut buf).unwrap();
        let rendered = String::from_utf8(buf).unwrap();
        let round_tripped = parse(&rendered).unwrap();
        assert_eq!(original.xmin, round_tripped.xmin);
        assert_eq!(original.xmax, round_tripped.xmax);
        assert_eq!(original.tiers.len(), round_tripped.tiers.len());
        if let (TextGridTier::Interval(a), TextGridTier::Interval(b)) =
            (&original.tiers[0], &round_tripped.tiers[0])
        {
            for (ai, bi) in a.intervals.iter().zip(b.intervals.iter()) {
                assert_eq!(ai.text, bi.text);
                assert_eq!(ai.xmin, bi.xmin);
                assert_eq!(ai.xmax, bi.xmax);
            }
        }
    }

    #[test]
    fn empty_label_round_trips_as_empty_string() {
        let tg = TextGridFile {
            xmin: 0.0,
            xmax: 1.0,
            tiers: vec![TextGridTier::Interval(IntervalTier {
                name: "t".into(),
                xmin: 0.0,
                xmax: 1.0,
                intervals: vec![IntervalEntry {
                    xmin: 0.0,
                    xmax: 1.0,
                    text: "".into(),
                }],
            })],
        };
        let mut buf: Vec<u8> = Vec::new();
        write_to(&tg, &mut buf).unwrap();
        let parsed = parse(&String::from_utf8(buf).unwrap()).unwrap();
        if let TextGridTier::Interval(t) = &parsed.tiers[0] {
            assert_eq!(t.intervals[0].text, "");
        } else {
            panic!();
        }
    }

    #[test]
    fn doubled_quotes_escape_correctly() {
        let tg = TextGridFile {
            xmin: 0.0,
            xmax: 1.0,
            tiers: vec![TextGridTier::Interval(IntervalTier {
                name: "t".into(),
                xmin: 0.0,
                xmax: 1.0,
                intervals: vec![IntervalEntry {
                    xmin: 0.0,
                    xmax: 1.0,
                    text: r#"she said "hi""#.into(),
                }],
            })],
        };
        let mut buf: Vec<u8> = Vec::new();
        write_to(&tg, &mut buf).unwrap();
        let rendered = String::from_utf8(buf).unwrap();
        // Each internal `"` becomes `""` per Praat's convention; the outer
        // `text = "..."` keeps its own pair of quotes. The simplest check
        // is that the rendered string contains the escaped form.
        assert!(
            rendered.contains(r#"""hi""""#),
            "rendered output missing doubled quotes: {rendered}",
        );
        let parsed = parse(&rendered).unwrap();
        if let TextGridTier::Interval(t) = &parsed.tiers[0] {
            assert_eq!(t.intervals[0].text, r#"she said "hi""#);
        }
    }

    #[test]
    fn empty_textgrid_uses_absent_marker() {
        let tg = TextGridFile {
            xmin: 0.0,
            xmax: 1.0,
            tiers: Vec::new(),
        };
        let mut buf: Vec<u8> = Vec::new();
        write_to(&tg, &mut buf).unwrap();
        let rendered = String::from_utf8(buf).unwrap();
        assert!(rendered.contains("tiers? <absent>"));
        let parsed = parse(&rendered).unwrap();
        assert!(parsed.tiers.is_empty());
    }

    #[test]
    fn encode_and_decode_json_sentinel_round_trip() {
        let plain = encode_label("hello", None);
        assert_eq!(plain, "hello");

        let with_json = encode_label("hello", Some(r#"{"foo":1}"#));
        assert_eq!(with_json, r#"hello {json:{"foo":1}}"#);

        let (label, json) = decode_label(&with_json);
        assert_eq!(label, "hello");
        assert_eq!(json.as_deref(), Some(r#"{"foo":1}"#));

        let (label, json) = decode_label("plain");
        assert_eq!(label, "plain");
        assert!(json.is_none());

        // Empty plain label with extra:
        let only_json = encode_label("", Some(r#"{"k":"v"}"#));
        assert_eq!(only_json, r#" {json:{"k":"v"}}"#);
        let (label, json) = decode_label(&only_json);
        assert_eq!(label, "");
        assert_eq!(json.as_deref(), Some(r#"{"k":"v"}"#));
    }

    #[test]
    fn decode_label_does_not_mis_detect_braces_in_plain_text() {
        // A label that ends with } but doesn't have the sentinel prefix.
        let (label, json) = decode_label("hello}");
        assert_eq!(label, "hello}");
        assert!(json.is_none());

        // A label that contains " {json:" earlier but doesn't end with }.
        let (label, json) = decode_label("she said {json:foo} earlier");
        assert_eq!(label, "she said {json:foo} earlier");
        assert!(json.is_none());
    }

    #[test]
    fn number_formatting_keeps_integers_as_integers() {
        assert_eq!(format_number(0.0), "0");
        assert_eq!(format_number(1.0), "1");
        assert_eq!(format_number(-5.0), "-5");
        assert_eq!(format_number(0.5), "0.5");
        assert_eq!(format_number(1.25), "1.25");
    }
}
