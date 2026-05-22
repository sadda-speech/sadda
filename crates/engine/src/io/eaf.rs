//! ELAN .eaf round-trip via `quick-xml`. Parser is permissive (accepts
//! EAF 2.7 / 2.8 / 3.0); writer emits EAF 2.8 (widely supported, no 3.0-only
//! features needed).
//!
//! ## References
//! - ELAN EAF format documentation (3.0 spec):
//!   <https://www.mpi.nl/tools/elan/EAF_Annotation_Format_3.0_and_ELAN.pdf>
//! - ELAN tier-stereotype reference:
//!   <https://www.mpi.nl/corpus/html/elan/ch02s02s05.html>
//! - pympi (Python EAF library, API-shape precedent):
//!   <https://github.com/dopefishh/pympi>
//!
//! ## What's preserved on round-trip
//! - Tier hierarchy via `PARENT_REF` ↔ `tier.parent_id`
//! - Annotation values (with JSON sentinel for our `extra` payload)
//! - Time alignment (ms precision)
//! - Reference annotations (`SYMBOLIC_ASSOCIATION` ↔ our `reference` tier)
//!
//! ## What's lost on round-trip
//! - `CONTROLLED_VOCABULARY`, `LANGUAGE`, `LICENSE`, `EXTERNAL_REF`,
//!   `LEXICON_REF`, `REF_LINK_SET`, `LOCALES` — not in our model.
//! - Stereotypes beyond the named three (Time_Subdivision is recognised but
//!   the full Symbolic_Subdivision + Included_In semantics are flattened).
//! - Original annotation IDs (fresh `aN` IDs minted on export).
//! - Media file path metadata in HEADER (we emit a placeholder).
//!
//! ## Point-tier convention
//! ELAN has no native point type. On export, our `point` tier becomes a
//! `[t, t + 1ms]` degenerate alignable annotation. On import, a tier where
//! every annotation has `end - start <= 2ms` is recovered as a `point`
//! tier (the threshold is robust because no real annotator makes
//! sub-millisecond intervals).

use std::fs;
use std::io::Cursor;
use std::path::Path;

use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::reader::Reader;
use quick_xml::writer::Writer;

use crate::error::{EngineError, Result};

/// In-memory representation of a parsed `.eaf` file.
#[derive(Debug, Clone)]
pub struct EafFile {
    /// File-level FORMAT attribute (e.g. `"2.8"`).
    pub format: String,
    /// Optional media-file URL from `<MEDIA_DESCRIPTOR MEDIA_URL=...>`.
    pub media_url: Option<String>,
    /// Tiers in document order.
    pub tiers: Vec<EafTier>,
}

/// One EAF tier.
#[derive(Debug, Clone)]
pub struct EafTier {
    /// Tier ID (must be unique within the file).
    pub tier_id: String,
    /// Linguistic-type reference; resolved to a stereotype during parse.
    pub linguistic_type_ref: String,
    /// Stereotype (if known): `Time_Subdivision`, `Included_In`,
    /// `Symbolic_Subdivision`, `Symbolic_Association`. `None` for top-level
    /// alignable tiers.
    pub stereotype: Option<String>,
    /// Parent tier ID, if any. EAF's `PARENT_REF`.
    pub parent_ref: Option<String>,
    /// Annotations in document order.
    pub annotations: Vec<EafAnnotation>,
}

/// One EAF annotation.
#[derive(Debug, Clone)]
pub enum EafAnnotation {
    /// `ALIGNABLE_ANNOTATION` — independent time alignment.
    Alignable {
        /// `ANNOTATION_ID` attribute.
        id: String,
        /// Start time in milliseconds (resolved via TIME_ORDER).
        start_ms: i64,
        /// End time in milliseconds.
        end_ms: i64,
        /// `<ANNOTATION_VALUE>` body text.
        value: String,
    },
    /// `REF_ANNOTATION` — symbolic / ref-based annotation. Inherits time
    /// from the referenced parent annotation.
    Ref {
        /// `ANNOTATION_ID` attribute.
        id: String,
        /// Parent annotation ID via `ANNOTATION_REF`.
        annotation_ref: String,
        /// Body text.
        value: String,
    },
}

impl EafAnnotation {
    /// `ANNOTATION_ID` accessor common to both variants.
    pub fn id(&self) -> &str {
        match self {
            EafAnnotation::Alignable { id, .. } => id,
            EafAnnotation::Ref { id, .. } => id,
        }
    }

    /// `<ANNOTATION_VALUE>` body accessor common to both variants.
    pub fn value(&self) -> &str {
        match self {
            EafAnnotation::Alignable { value, .. } => value,
            EafAnnotation::Ref { value, .. } => value,
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Reads an EAF file from disk.
pub fn read(path: &Path) -> Result<EafFile> {
    let raw = fs::read_to_string(path).map_err(EngineError::Io)?;
    parse(&raw)
}

fn map_xml_err(e: quick_xml::Error) -> EngineError {
    EngineError::Corpus(format!("EAF XML error: {e}"))
}

fn map_attr_err(e: quick_xml::events::attributes::AttrError) -> EngineError {
    EngineError::Corpus(format!("EAF XML attribute error: {e}"))
}

/// Parses an EAF file from a string.
pub fn parse(input: &str) -> Result<EafFile> {
    let mut reader = Reader::from_str(input);
    let config = reader.config_mut();
    config.trim_text(true);

    let mut format = String::from("2.8");
    let mut media_url: Option<String> = None;

    // TIME_ORDER: slot ID → ms value.
    let mut time_slots: std::collections::HashMap<String, i64> =
        std::collections::HashMap::new();

    // Parsed tiers; finalised after annotations are resolved.
    let mut tiers: Vec<RawTier> = Vec::new();

    // Linguistic-type → stereotype (CONSTRAINTS attribute).
    let mut lt_to_stereotype: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    let mut buf = Vec::new();
    // Current parser context.
    let mut current_tier: Option<RawTier> = None;
    let mut current_annotation_open: Option<PartialAnnotation> = None;
    let mut current_text_target: Option<TextTarget> = None;

    loop {
        match reader.read_event_into(&mut buf).map_err(map_xml_err)? {
            Event::Decl(_) | Event::Comment(_) | Event::CData(_) | Event::PI(_) => {}
            Event::Eof => break,
            Event::Start(e) | Event::Empty(e) => {
                let local_name_owned = e.local_name().as_ref().to_vec();
                let local_name = std::str::from_utf8(&local_name_owned).unwrap_or("");

                match local_name {
                    "ANNOTATION_DOCUMENT" => {
                        for attr in e.attributes() {
                            let attr = attr.map_err(map_attr_err)?;
                            if attr.key.local_name().as_ref() == b"FORMAT" {
                                format = String::from_utf8_lossy(&attr.value).into_owned();
                            }
                        }
                    }
                    "MEDIA_DESCRIPTOR" => {
                        for attr in e.attributes() {
                            let attr = attr.map_err(map_attr_err)?;
                            if attr.key.local_name().as_ref() == b"MEDIA_URL" {
                                media_url =
                                    Some(String::from_utf8_lossy(&attr.value).into_owned());
                            }
                        }
                    }
                    "TIME_SLOT" => {
                        let mut id = String::new();
                        let mut value: Option<i64> = None;
                        for attr in e.attributes() {
                            let attr = attr.map_err(map_attr_err)?;
                            match attr.key.local_name().as_ref() {
                                b"TIME_SLOT_ID" => {
                                    id = String::from_utf8_lossy(&attr.value).into_owned();
                                }
                                b"TIME_VALUE" => {
                                    value = Some(
                                        String::from_utf8_lossy(&attr.value)
                                            .parse::<i64>()
                                            .map_err(|e| {
                                                EngineError::Corpus(format!(
                                                    "EAF: bad TIME_VALUE: {e}"
                                                ))
                                            })?,
                                    );
                                }
                                _ => {}
                            }
                        }
                        if !id.is_empty() {
                            // Slots without a value are valid in EAF (the
                            // slot is "unaligned"); we record 0 as a fallback.
                            time_slots.insert(id, value.unwrap_or(0));
                        }
                    }
                    "TIER" => {
                        let mut tier = RawTier::default();
                        for attr in e.attributes() {
                            let attr = attr.map_err(map_attr_err)?;
                            let v = String::from_utf8_lossy(&attr.value).into_owned();
                            match attr.key.local_name().as_ref() {
                                b"TIER_ID" => tier.tier_id = v,
                                b"LINGUISTIC_TYPE_REF" => tier.linguistic_type_ref = v,
                                b"PARENT_REF" => tier.parent_ref = Some(v),
                                _ => {}
                            }
                        }
                        current_tier = Some(tier);
                    }
                    "ALIGNABLE_ANNOTATION" => {
                        let mut id = String::new();
                        let mut ref1 = String::new();
                        let mut ref2 = String::new();
                        for attr in e.attributes() {
                            let attr = attr.map_err(map_attr_err)?;
                            let v = String::from_utf8_lossy(&attr.value).into_owned();
                            match attr.key.local_name().as_ref() {
                                b"ANNOTATION_ID" => id = v,
                                b"TIME_SLOT_REF1" => ref1 = v,
                                b"TIME_SLOT_REF2" => ref2 = v,
                                _ => {}
                            }
                        }
                        current_annotation_open = Some(PartialAnnotation::Alignable {
                            id,
                            ref1,
                            ref2,
                            value: String::new(),
                        });
                    }
                    "REF_ANNOTATION" => {
                        let mut id = String::new();
                        let mut annotation_ref = String::new();
                        for attr in e.attributes() {
                            let attr = attr.map_err(map_attr_err)?;
                            let v = String::from_utf8_lossy(&attr.value).into_owned();
                            match attr.key.local_name().as_ref() {
                                b"ANNOTATION_ID" => id = v,
                                b"ANNOTATION_REF" => annotation_ref = v,
                                _ => {}
                            }
                        }
                        current_annotation_open = Some(PartialAnnotation::Ref {
                            id,
                            annotation_ref,
                            value: String::new(),
                        });
                    }
                    "ANNOTATION_VALUE" => {
                        current_text_target = Some(TextTarget::AnnotationValue);
                    }
                    "LINGUISTIC_TYPE" => {
                        let mut id = String::new();
                        let mut stereotype: Option<String> = None;
                        for attr in e.attributes() {
                            let attr = attr.map_err(map_attr_err)?;
                            let v = String::from_utf8_lossy(&attr.value).into_owned();
                            match attr.key.local_name().as_ref() {
                                b"LINGUISTIC_TYPE_ID" => id = v,
                                b"CONSTRAINTS" => stereotype = Some(v),
                                _ => {}
                            }
                        }
                        if !id.is_empty() {
                            if let Some(s) = stereotype {
                                lt_to_stereotype.insert(id, s);
                            }
                        }
                    }
                    _ => {} // Ignore unknown elements (CV, LANGUAGE, etc.).
                }
            }
            Event::End(e) => {
                let local_name_owned = e.local_name().as_ref().to_vec();
                let local_name = std::str::from_utf8(&local_name_owned).unwrap_or("");
                match local_name {
                    "TIER" => {
                        if let Some(tier) = current_tier.take() {
                            tiers.push(tier);
                        }
                    }
                    "ALIGNABLE_ANNOTATION" | "REF_ANNOTATION" => {
                        if let (Some(partial), Some(tier)) =
                            (current_annotation_open.take(), current_tier.as_mut())
                        {
                            let resolved = match partial {
                                PartialAnnotation::Alignable { id, ref1, ref2, value } => {
                                    let start = *time_slots.get(&ref1).ok_or_else(|| {
                                        EngineError::Corpus(format!(
                                            "EAF: unresolved TIME_SLOT_REF1 {ref1:?}"
                                        ))
                                    })?;
                                    let end = *time_slots.get(&ref2).ok_or_else(|| {
                                        EngineError::Corpus(format!(
                                            "EAF: unresolved TIME_SLOT_REF2 {ref2:?}"
                                        ))
                                    })?;
                                    EafAnnotation::Alignable {
                                        id,
                                        start_ms: start,
                                        end_ms: end,
                                        value,
                                    }
                                }
                                PartialAnnotation::Ref { id, annotation_ref, value } => {
                                    EafAnnotation::Ref {
                                        id,
                                        annotation_ref,
                                        value,
                                    }
                                }
                            };
                            tier.annotations.push(resolved);
                        }
                    }
                    "ANNOTATION_VALUE" => {
                        current_text_target = None;
                    }
                    _ => {}
                }
            }
            Event::Text(t) => {
                if let Some(TextTarget::AnnotationValue) = current_text_target {
                    let s = t
                        .decode()
                        .map_err(|e| EngineError::Corpus(format!("EAF text decode: {e}")))?;
                    push_value(&mut current_annotation_open, &s);
                }
            }
            // Predefined XML entities (`&quot;`, `&amp;`, `&lt;`, `&gt;`,
            // `&apos;`) and numeric character references (`&#N;`,
            // `&#xN;`) arrive as their own events between text chunks.
            // Resolve and stitch them back into the annotation value.
            Event::GeneralRef(r) => {
                if let Some(TextTarget::AnnotationValue) = current_text_target {
                    let name = r
                        .decode()
                        .map_err(|e| EngineError::Corpus(format!("EAF entity-ref decode: {e}")))?
                        .into_owned();
                    let resolved = resolve_xml_entity_ref(&r, &name)?;
                    push_value(&mut current_annotation_open, &resolved);
                }
            }
            _ => {}
        }
        buf.clear();
    }

    // Resolve stereotype per tier.
    let tiers: Vec<EafTier> = tiers
        .into_iter()
        .map(|t| EafTier {
            stereotype: lt_to_stereotype.get(&t.linguistic_type_ref).cloned(),
            tier_id: t.tier_id,
            linguistic_type_ref: t.linguistic_type_ref,
            parent_ref: t.parent_ref,
            annotations: t.annotations,
        })
        .collect();

    Ok(EafFile {
        format,
        media_url,
        tiers,
    })
}

/// Appends `text` to the current annotation's `value` if one is open.
fn push_value(open: &mut Option<PartialAnnotation>, text: &str) {
    if let Some(p) = open.as_mut() {
        match p {
            PartialAnnotation::Alignable { value, .. } => value.push_str(text),
            PartialAnnotation::Ref { value, .. } => value.push_str(text),
        }
    }
}

/// Resolves a predefined XML entity (`quot`, `amp`, `lt`, `gt`, `apos`) or
/// a numeric character reference (`#N`, `#xN`) to its string form.
fn resolve_xml_entity_ref(r: &quick_xml::events::BytesRef<'_>, name: &str) -> Result<String> {
    match name {
        "quot" => Ok("\"".into()),
        "amp" => Ok("&".into()),
        "lt" => Ok("<".into()),
        "gt" => Ok(">".into()),
        "apos" => Ok("'".into()),
        _ => {
            if let Some(ch) = r
                .resolve_char_ref()
                .map_err(|e| EngineError::Corpus(format!("EAF entity ref: {e}")))?
            {
                Ok(ch.to_string())
            } else {
                // Unknown named entities are dropped; the caller will see
                // text chunks stitched without them. This matches Praat's
                // tolerant attitude for now (we can tighten later).
                Ok(String::new())
            }
        }
    }
}

#[derive(Debug, Default)]
struct RawTier {
    tier_id: String,
    linguistic_type_ref: String,
    parent_ref: Option<String>,
    annotations: Vec<EafAnnotation>,
}

#[derive(Debug)]
enum PartialAnnotation {
    Alignable {
        id: String,
        ref1: String,
        ref2: String,
        value: String,
    },
    Ref {
        id: String,
        annotation_ref: String,
        value: String,
    },
}

#[derive(Debug)]
enum TextTarget {
    AnnotationValue,
}

// ---------------------------------------------------------------------------
// JSON-sentinel label helpers
// ---------------------------------------------------------------------------

const SENTINEL_PREFIX: &str = " {json:";
const SENTINEL_SUFFIX: &str = "}";

/// Appends a JSON sentinel to a label: `<label> {json:<inline-json>}`.
/// Returns `label` unchanged if `extra_json` is `None` or empty. Identical
/// scheme to [`crate::io::textgrid::encode_label`] so the same `extra`
/// payloads round-trip through either format.
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

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Writes an EAF file to disk in EAF 2.8 format.
pub fn write(file: &EafFile, path: &Path) -> Result<()> {
    let bytes = write_to_bytes(file)?;
    fs::write(path, bytes).map_err(EngineError::Io)?;
    Ok(())
}

/// Renders an EAF file as bytes in EAF 2.8 format.
pub fn write_to_bytes(file: &EafFile) -> Result<Vec<u8>> {
    let mut buf = Cursor::new(Vec::new());
    let mut writer = Writer::new_with_indent(&mut buf, b' ', 4);

    writer
        .write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))
        ?;

    // Collect all unique time-ms values used by alignable annotations, then
    // assign synthetic slot IDs in sorted order. This keeps the TIME_ORDER
    // section deterministic and minimal.
    let mut times_set: std::collections::BTreeSet<i64> = std::collections::BTreeSet::new();
    for tier in &file.tiers {
        for ann in &tier.annotations {
            if let EafAnnotation::Alignable {
                start_ms, end_ms, ..
            } = ann
            {
                times_set.insert(*start_ms);
                times_set.insert(*end_ms);
            }
        }
    }
    let times: Vec<i64> = times_set.into_iter().collect();
    let time_to_slot: std::collections::HashMap<i64, String> = times
        .iter()
        .enumerate()
        .map(|(i, &t)| (t, format!("ts{}", i + 1)))
        .collect();

    // <ANNOTATION_DOCUMENT ...>
    let now = "2026-05-22T12:00:00Z"; // deterministic placeholder
    let mut doc = BytesStart::new("ANNOTATION_DOCUMENT");
    doc.push_attribute(("AUTHOR", ""));
    doc.push_attribute(("DATE", now));
    doc.push_attribute(("FORMAT", file.format.as_str()));
    doc.push_attribute(("VERSION", file.format.as_str()));
    doc.push_attribute(("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance"));
    doc.push_attribute((
        "xsi:noNamespaceSchemaLocation",
        format!("http://www.mpi.nl/tools/elan/EAFv{}.xsd", file.format).as_str(),
    ));
    writer.write_event(Event::Start(doc))?;

    // <HEADER ... > <MEDIA_DESCRIPTOR .../> </HEADER>
    let mut header = BytesStart::new("HEADER");
    header.push_attribute(("MEDIA_FILE", ""));
    header.push_attribute(("TIME_UNITS", "milliseconds"));
    writer.write_event(Event::Start(header))?;
    if let Some(url) = &file.media_url {
        let mut md = BytesStart::new("MEDIA_DESCRIPTOR");
        md.push_attribute(("MEDIA_URL", url.as_str()));
        md.push_attribute(("MIME_TYPE", "audio/x-wav"));
        writer.write_event(Event::Empty(md))?;
    }
    writer
        .write_event(Event::End(BytesEnd::new("HEADER")))
        ?;

    // <TIME_ORDER>
    writer
        .write_event(Event::Start(BytesStart::new("TIME_ORDER")))
        ?;
    for t in &times {
        let mut slot = BytesStart::new("TIME_SLOT");
        let id = &time_to_slot[t];
        slot.push_attribute(("TIME_SLOT_ID", id.as_str()));
        slot.push_attribute(("TIME_VALUE", t.to_string().as_str()));
        writer.write_event(Event::Empty(slot))?;
    }
    writer
        .write_event(Event::End(BytesEnd::new("TIME_ORDER")))
        ?;

    // <TIER>...
    for tier in &file.tiers {
        let mut t_el = BytesStart::new("TIER");
        t_el.push_attribute(("LINGUISTIC_TYPE_REF", tier.linguistic_type_ref.as_str()));
        t_el.push_attribute(("TIER_ID", tier.tier_id.as_str()));
        if let Some(parent) = &tier.parent_ref {
            t_el.push_attribute(("PARENT_REF", parent.as_str()));
        }
        writer.write_event(Event::Start(t_el))?;
        for ann in &tier.annotations {
            writer
                .write_event(Event::Start(BytesStart::new("ANNOTATION")))
                ?;
            match ann {
                EafAnnotation::Alignable {
                    id,
                    start_ms,
                    end_ms,
                    value,
                } => {
                    let mut a = BytesStart::new("ALIGNABLE_ANNOTATION");
                    a.push_attribute(("ANNOTATION_ID", id.as_str()));
                    a.push_attribute((
                        "TIME_SLOT_REF1",
                        time_to_slot[start_ms].as_str(),
                    ));
                    a.push_attribute((
                        "TIME_SLOT_REF2",
                        time_to_slot[end_ms].as_str(),
                    ));
                    writer.write_event(Event::Start(a))?;
                    writer
                        .write_event(Event::Start(BytesStart::new("ANNOTATION_VALUE")))
                        ?;
                    writer
                        .write_event(Event::Text(BytesText::new(value)))
                        ?;
                    writer
                        .write_event(Event::End(BytesEnd::new("ANNOTATION_VALUE")))
                        ?;
                    writer
                        .write_event(Event::End(BytesEnd::new("ALIGNABLE_ANNOTATION")))
                        ?;
                }
                EafAnnotation::Ref {
                    id,
                    annotation_ref,
                    value,
                } => {
                    let mut a = BytesStart::new("REF_ANNOTATION");
                    a.push_attribute(("ANNOTATION_ID", id.as_str()));
                    a.push_attribute(("ANNOTATION_REF", annotation_ref.as_str()));
                    writer.write_event(Event::Start(a))?;
                    writer
                        .write_event(Event::Start(BytesStart::new("ANNOTATION_VALUE")))
                        ?;
                    writer
                        .write_event(Event::Text(BytesText::new(value)))
                        ?;
                    writer
                        .write_event(Event::End(BytesEnd::new("ANNOTATION_VALUE")))
                        ?;
                    writer
                        .write_event(Event::End(BytesEnd::new("REF_ANNOTATION")))
                        ?;
                }
            }
            writer
                .write_event(Event::End(BytesEnd::new("ANNOTATION")))
                ?;
        }
        writer
            .write_event(Event::End(BytesEnd::new("TIER")))
            ?;
    }

    // Emit linguistic types used by the tiers. We always emit
    // `sadda_alignable` (no constraints) and `sadda_symbolic` (Symbolic
    // Association) since exporter uses those names.
    let mut lts_emitted: std::collections::HashSet<String> = std::collections::HashSet::new();
    for tier in &file.tiers {
        if !lts_emitted.insert(tier.linguistic_type_ref.clone()) {
            continue;
        }
        let mut lt = BytesStart::new("LINGUISTIC_TYPE");
        lt.push_attribute(("LINGUISTIC_TYPE_ID", tier.linguistic_type_ref.as_str()));
        if let Some(stereotype) = &tier.stereotype {
            lt.push_attribute(("CONSTRAINTS", stereotype.as_str()));
            lt.push_attribute(("TIME_ALIGNABLE", "false"));
        } else {
            lt.push_attribute(("TIME_ALIGNABLE", "true"));
        }
        writer.write_event(Event::Empty(lt))?;
    }

    // Emit the standard EAF constraint definitions so the file is
    // self-contained for ELAN.
    for (stereotype, desc) in &[
        (
            "Time_Subdivision",
            "Time subdivision of parent annotation's time interval, no time gaps allowed within this interval",
        ),
        (
            "Symbolic_Subdivision",
            "Symbolic subdivision of a parent annotation. Annotations refering to the same parent are ordered",
        ),
        (
            "Symbolic_Association",
            "1-1 association with a parent annotation",
        ),
        ("Included_In", "Time alignable annotations within the parent annotation's time interval, gaps are allowed"),
    ] {
        let mut c = BytesStart::new("CONSTRAINT");
        c.push_attribute(("DESCRIPTION", *desc));
        c.push_attribute(("STEREOTYPE", *stereotype));
        writer.write_event(Event::Empty(c))?;
    }

    writer
        .write_event(Event::End(BytesEnd::new("ANNOTATION_DOCUMENT")))
        ?;

    Ok(buf.into_inner())
}

// ---------------------------------------------------------------------------
// Linguistic-type names emitted by the writer
// ---------------------------------------------------------------------------

/// Name we use for "interval tier, no parent / free time-alignment".
pub const LT_ALIGNABLE: &str = "sadda_alignable";

/// Name we use for "REF-annotation tier, Symbolic_Association stereotype".
pub const LT_SYMBOLIC: &str = "sadda_symbolic";

/// Name we use for "interval tier with a parent (Included_In stereotype)".
pub const LT_INCLUDED: &str = "sadda_included";

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_EAF: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="" DATE="2025-01-01T00:00:00Z" FORMAT="2.8" VERSION="2.8"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="http://www.mpi.nl/tools/elan/EAFv2.8.xsd">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds">
        <MEDIA_DESCRIPTOR MEDIA_URL="file:///audio.wav" MIME_TYPE="audio/x-wav"/>
    </HEADER>
    <TIME_ORDER>
        <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>
        <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="500"/>
        <TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="1000"/>
    </TIME_ORDER>
    <TIER LINGUISTIC_TYPE_REF="words" TIER_ID="words">
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
                <ANNOTATION_VALUE>hello</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a2" TIME_SLOT_REF1="ts2" TIME_SLOT_REF2="ts3">
                <ANNOTATION_VALUE>world</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
    </TIER>
    <TIER LINGUISTIC_TYPE_REF="symbolic" PARENT_REF="words" TIER_ID="notes">
        <ANNOTATION>
            <REF_ANNOTATION ANNOTATION_ID="a3" ANNOTATION_REF="a1">
                <ANNOTATION_VALUE>greeting</ANNOTATION_VALUE>
            </REF_ANNOTATION>
        </ANNOTATION>
    </TIER>
    <LINGUISTIC_TYPE LINGUISTIC_TYPE_ID="words" TIME_ALIGNABLE="true"/>
    <LINGUISTIC_TYPE CONSTRAINTS="Symbolic_Association" LINGUISTIC_TYPE_ID="symbolic" TIME_ALIGNABLE="false"/>
</ANNOTATION_DOCUMENT>
"#;

    #[test]
    fn parse_recovers_tier_hierarchy_and_annotations() {
        let eaf = parse(MINIMAL_EAF).unwrap();
        assert_eq!(eaf.format, "2.8");
        assert_eq!(eaf.media_url.as_deref(), Some("file:///audio.wav"));
        assert_eq!(eaf.tiers.len(), 2);
        let words = &eaf.tiers[0];
        assert_eq!(words.tier_id, "words");
        assert!(words.parent_ref.is_none());
        assert_eq!(words.annotations.len(), 2);
        match &words.annotations[0] {
            EafAnnotation::Alignable {
                id,
                start_ms,
                end_ms,
                value,
            } => {
                assert_eq!(id, "a1");
                assert_eq!(*start_ms, 0);
                assert_eq!(*end_ms, 500);
                assert_eq!(value, "hello");
            }
            other => panic!("expected Alignable, got {other:?}"),
        }
        let notes = &eaf.tiers[1];
        assert_eq!(notes.parent_ref.as_deref(), Some("words"));
        assert_eq!(notes.stereotype.as_deref(), Some("Symbolic_Association"));
        match &notes.annotations[0] {
            EafAnnotation::Ref {
                id,
                annotation_ref,
                value,
            } => {
                assert_eq!(id, "a3");
                assert_eq!(annotation_ref, "a1");
                assert_eq!(value, "greeting");
            }
            other => panic!("expected Ref, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_preserves_structure() {
        let original = parse(MINIMAL_EAF).unwrap();
        let bytes = write_to_bytes(&original).unwrap();
        let rendered = String::from_utf8(bytes).unwrap();
        let reparsed = parse(&rendered).unwrap();
        assert_eq!(reparsed.tiers.len(), original.tiers.len());
        assert_eq!(reparsed.tiers[0].tier_id, "words");
        assert_eq!(reparsed.tiers[1].parent_ref.as_deref(), Some("words"));
        // Annotation content survives.
        if let (
            EafAnnotation::Alignable { value: orig, .. },
            EafAnnotation::Alignable { value: round, .. },
        ) = (
            &original.tiers[0].annotations[0],
            &reparsed.tiers[0].annotations[0],
        ) {
            assert_eq!(orig, round);
        }
    }

    #[test]
    fn writer_emits_well_formed_xml() {
        let eaf = parse(MINIMAL_EAF).unwrap();
        let bytes = write_to_bytes(&eaf).unwrap();
        let rendered = String::from_utf8(bytes).unwrap();
        assert!(rendered.starts_with("<?xml"));
        assert!(rendered.contains("<ANNOTATION_DOCUMENT"));
        assert!(rendered.contains("</ANNOTATION_DOCUMENT>"));
        assert!(rendered.contains("FORMAT=\"2.8\""));
        assert!(rendered.contains("TIME_SLOT_ID="));
        assert!(rendered.contains("TIER_ID=\"words\""));
        assert!(rendered.contains("PARENT_REF=\"words\""));
    }

    #[test]
    fn writer_emits_constraint_definitions() {
        let eaf = parse(MINIMAL_EAF).unwrap();
        let bytes = write_to_bytes(&eaf).unwrap();
        let rendered = String::from_utf8(bytes).unwrap();
        for stereotype in &[
            "Time_Subdivision",
            "Symbolic_Subdivision",
            "Symbolic_Association",
            "Included_In",
        ] {
            assert!(
                rendered.contains(&format!("STEREOTYPE=\"{stereotype}\"")),
                "missing stereotype {stereotype} in:\n{rendered}",
            );
        }
    }

    #[test]
    fn unknown_xml_elements_are_ignored() {
        // Add a CONTROLLED_VOCABULARY block + LICENSE; parser must ignore.
        let augmented = MINIMAL_EAF.replace(
            "</ANNOTATION_DOCUMENT>",
            r#"<CONTROLLED_VOCABULARY CV_ID="x">
        <DESCRIPTION LANG_REF="und">a cv</DESCRIPTION>
    </CONTROLLED_VOCABULARY>
    <LICENSE LICENSE_URL="https://example.com/"/>
</ANNOTATION_DOCUMENT>"#,
        );
        let eaf = parse(&augmented).unwrap();
        // Same tier structure as the un-augmented fixture.
        assert_eq!(eaf.tiers.len(), 2);
    }

    #[test]
    fn json_sentinel_round_trips_through_xml_escape() {
        // Reproduces the integration-test scenario: a label whose JSON
        // sentinel contains `"` characters must survive a write→read.
        let original = EafFile {
            format: "2.8".into(),
            media_url: None,
            tiers: vec![EafTier {
                tier_id: "t".into(),
                linguistic_type_ref: LT_ALIGNABLE.into(),
                stereotype: None,
                parent_ref: None,
                annotations: vec![EafAnnotation::Alignable {
                    id: "a1".into(),
                    start_ms: 0,
                    end_ms: 500,
                    value: encode_label("h", Some(r#"{"v":1}"#)),
                }],
            }],
        };
        let bytes = write_to_bytes(&original).unwrap();
        let rendered = String::from_utf8(bytes).unwrap();
        // The raw XML must escape the JSON quotes.
        assert!(
            rendered.contains("&quot;v&quot;"),
            "writer should escape inner quotes:\n{rendered}",
        );
        let reparsed = parse(&rendered).unwrap();
        let value = match &reparsed.tiers[0].annotations[0] {
            EafAnnotation::Alignable { value, .. } => value.as_str(),
            _ => panic!("expected alignable"),
        };
        let (plain, extra) = decode_label(value);
        assert_eq!(plain, "h");
        assert_eq!(extra.as_deref(), Some(r#"{"v":1}"#));
    }

    #[test]
    fn writer_emits_one_time_slot_per_unique_time() {
        let eaf = parse(MINIMAL_EAF).unwrap();
        let bytes = write_to_bytes(&eaf).unwrap();
        let rendered = String::from_utf8(bytes).unwrap();
        // Three unique times (0, 500, 1000) → three TIME_SLOT entries.
        let slots = rendered.matches("<TIME_SLOT").count();
        assert_eq!(slots, 3, "expected 3 TIME_SLOT entries; rendered: {rendered}");
    }
}
