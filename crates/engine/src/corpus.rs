//! Project directory + SQLite-backed corpus database. Schema is owned by the
//! [`migrations`] module; this file holds the user-facing types
//! (`Project`, `Bundle`, `Speaker`, `Session`) and the project-directory
//! layout.
//!
//! The B1 slice landed Speaker / Session / Bundle-extension as Rust types;
//! the schema also includes Instrument, Protocol, Entity, EntityRef, Tier
//! header, and ProcessingRun tables, but those don't have public Rust types
//! yet — they're populated by future slices when each grows a first
//! concrete user (B2 for Tier + EntityRef; F1 for ProcessingRun; …).

pub mod migrations;

use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension};
use sha2::{Digest, Sha256};

use crate::Audio;
use crate::error::{EngineError, Result};

/// A sadda project: a directory holding audio, derived signals, attachments,
/// and a SQLite-backed corpus database.
///
/// At Phase 0 the corpus schema is intentionally tiny — just `project` and
/// `bundle` tables — enough to exercise the create / register / list /
/// load-audio flow end-to-end. The full v1 entity model lands later.
#[derive(Debug)]
pub struct Project {
    root: PathBuf,
    conn: Connection,
    /// Currently-active F1 recipe-run id, if any. Single-threaded
    /// interior-mutable cell — `Project` is single-threaded by design
    /// (the embedded `rusqlite::Connection` isn't `Sync`).
    recipe_run_id: std::cell::Cell<Option<i64>>,
    /// True if this `Project` instance owns the `.sadda-lock` file
    /// and should delete it on `Drop`. False for the (currently-
    /// unused) read-only-equivalent open path, if we ever add one.
    holds_lock: bool,
}

/// Metadata for one recording inside a [`Project`].
///
/// Beyond the audio header, a bundle carries optional foreign keys to
/// [`Session`] and [`Speaker`] plus a freeform JSON `extra` payload.
/// Tiers, derived signals, and processing-run rows attached to a bundle
/// land in later slices.
#[derive(Debug, Clone)]
pub struct Bundle {
    /// Bundle id (primary key in the corpus database).
    pub id: i64,
    /// Human-readable bundle name (set at registration time).
    pub name: String,
    /// Audio file path relative to the project root.
    pub audio_relative_path: String,
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo, …).
    pub channels: u16,
    /// Number of audio frames (samples per channel).
    pub n_frames: usize,
    /// Optional [`Session`] the bundle belongs to.
    pub session_id: Option<i64>,
    /// Optional [`Speaker`] the bundle recorded.
    pub speaker_id: Option<i64>,
    /// Freeform JSON payload (stored as text); shape governed by the active
    /// profile's schema validators.
    pub extra: Option<String>,
}

/// A person who produced speech in this project (participant, patient, case
/// subject, …). The B1 surface mirrors the `speaker` table directly; profile
/// schemas validate the JSON `extra` payload.
#[derive(Debug, Clone)]
pub struct Speaker {
    /// Speaker id (primary key).
    pub id: i64,
    /// Human-readable name or pseudonymous identifier.
    pub name: String,
    /// Sex / gender label (free text; profiles may constrain).
    pub sex: Option<String>,
    /// Birth year (full integer year, e.g. 1987).
    pub birth_year: Option<i32>,
    /// Freeform notes set by the user.
    pub notes: Option<String>,
    /// Freeform JSON payload.
    pub extra: Option<String>,
    /// ISO 8601 UTC timestamp set at insert time.
    pub created_at: String,
}

/// Optional fields for creating a [`Speaker`]. Use [`SpeakerSpec::new`] for
/// the common `name`-only case.
#[derive(Debug, Clone, Default)]
pub struct SpeakerSpec {
    /// Name (required).
    pub name: String,
    /// Sex / gender label.
    pub sex: Option<String>,
    /// Birth year.
    pub birth_year: Option<i32>,
    /// Notes.
    pub notes: Option<String>,
    /// JSON `extra`.
    pub extra: Option<String>,
}

impl SpeakerSpec {
    /// Builds a spec with only the required name field populated.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }
}

/// A recording session — a time-bounded block during which one or more
/// bundles were captured. The B1 surface mirrors the `session` table.
#[derive(Debug, Clone)]
pub struct Session {
    /// Session id (primary key).
    pub id: i64,
    /// Human-readable session name.
    pub name: String,
    /// ISO 8601 UTC start timestamp (recording start).
    pub started_at: Option<String>,
    /// ISO 8601 UTC end timestamp.
    pub ended_at: Option<String>,
    /// Free-form location label (room, lab, field site).
    pub location: Option<String>,
    /// FK into the `instrument` table.
    pub instrument_id: Option<i64>,
    /// FK into the `protocol` table.
    pub protocol_id: Option<i64>,
    /// Freeform notes.
    pub notes: Option<String>,
    /// Freeform JSON payload.
    pub extra: Option<String>,
    /// ISO 8601 UTC timestamp set at insert time.
    pub created_at: String,
}

/// Optional fields for creating a [`Session`]. Use [`SessionSpec::new`] for
/// the common `name`-only case.
#[derive(Debug, Clone, Default)]
pub struct SessionSpec {
    /// Name (required).
    pub name: String,
    /// Start timestamp (ISO 8601).
    pub started_at: Option<String>,
    /// End timestamp (ISO 8601).
    pub ended_at: Option<String>,
    /// Location label.
    pub location: Option<String>,
    /// FK into `instrument` (None until B1's follow-up exposes instruments).
    pub instrument_id: Option<i64>,
    /// FK into `protocol`.
    pub protocol_id: Option<i64>,
    /// Notes.
    pub notes: Option<String>,
    /// JSON `extra`.
    pub extra: Option<String>,
}

impl SessionSpec {
    /// Builds a spec with only the required name field populated.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }
}

/// Microphone / signal-chain calibration that maps the engine's
/// relative dB-FS readings to absolute dB-SPL (re 20 µPa).
///
/// A flat single-offset model (A3): a calibration tone of known SPL is
/// recorded, and the dB-FS the engine measures for it pins the offset —
/// `offset = reference_spl_db − reference_db_fs`, added to any dB-FS
/// reading to get dB-SPL. The reference *pair* is stored (not just the
/// derived offset) so the calibration is auditable. Frequency-response
/// curves are a later refinement. Serialized as JSON in
/// `instrument.calibration`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Calibration {
    /// SPL of the calibration tone, in dB-SPL (e.g. `94.0` from a
    /// class-1 acoustic calibrator).
    pub reference_spl_db: f64,
    /// dB-FS the engine measured for that same tone.
    pub reference_db_fs: f64,
}

impl Calibration {
    /// The offset added to a dB-FS reading to obtain dB-SPL.
    pub fn spl_offset_db(&self) -> f64 {
        self.reference_spl_db - self.reference_db_fs
    }

    /// Converts a relative dB-FS intensity to calibrated dB-SPL.
    pub fn to_spl(&self, db_fs: crate::units::Decibels) -> crate::units::Decibels {
        crate::units::Decibels::new(db_fs.value() + self.spl_offset_db() as f32)
    }
}

/// A capture instrument (microphone, preamp, audio interface) with its
/// optional calibration. Mirrors the `instrument` table; the schema's
/// generic `calibration TEXT` column holds [`Calibration`] as JSON.
#[derive(Debug, Clone)]
pub struct Instrument {
    /// Instrument id (primary key).
    pub id: i64,
    /// Human-readable name.
    pub name: String,
    /// Free-form kind label (e.g. `"microphone"`, `"interface"`).
    pub kind: Option<String>,
    /// Serial number.
    pub serial: Option<String>,
    /// Calibration, if the instrument has been calibrated. `None` when
    /// the column is null or holds an unparseable legacy value.
    pub calibration: Option<Calibration>,
    /// Freeform JSON payload.
    pub extra: Option<String>,
    /// ISO 8601 UTC creation timestamp.
    pub created_at: String,
}

/// Fields for creating an [`Instrument`]. Use [`InstrumentSpec::new`]
/// for the common name-only case.
#[derive(Debug, Clone, Default)]
pub struct InstrumentSpec {
    /// Name (required).
    pub name: String,
    /// Kind label.
    pub kind: Option<String>,
    /// Serial number.
    pub serial: Option<String>,
    /// Calibration, if known at creation time.
    pub calibration: Option<Calibration>,
    /// JSON `extra`.
    pub extra: Option<String>,
}

impl InstrumentSpec {
    /// Builds a spec with only the required name field populated.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }
}

/// Optional fields for creating a [`Bundle`]. Use [`BundleSpec::new`] for
/// the common `name`-only case.
#[derive(Debug, Clone, Default)]
pub struct BundleSpec {
    /// Name (required).
    pub name: String,
    /// Optional [`Session`] this bundle belongs to.
    pub session_id: Option<i64>,
    /// Optional [`Speaker`] this bundle records.
    pub speaker_id: Option<i64>,
    /// JSON `extra`.
    pub extra: Option<String>,
}

impl BundleSpec {
    /// Builds a spec with only the required name field populated.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }
}

/// One of the six sparse / dense tier types. The three sparse variants
/// (`interval`, `point`, `reference`) are exposed in B2; the dense ones
/// (`continuous_numeric`, `continuous_vector`, `categorical_sampled`) ship
/// with Parquet sidecars in B3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierType {
    /// Time-interval annotations: phones, words, segments.
    Interval,
    /// Point annotations: event markers, clicks, glottal pulses.
    Point,
    /// Reference annotations: lexicon links, trial links, speaker turns.
    Reference,
    /// Dense single-channel numeric track (F0, intensity); B3.
    ContinuousNumeric,
    /// Dense multi-channel numeric track (embeddings, MFCC); B3.
    ContinuousVector,
    /// Dense categorical track (VAD, voicing on/off); B3.
    CategoricalSampled,
}

impl TierType {
    /// Returns the lowercase enum string stored in `tier.type` (matches the
    /// SQL CHECK constraint values).
    pub fn as_str(self) -> &'static str {
        match self {
            TierType::Interval => "interval",
            TierType::Point => "point",
            TierType::Reference => "reference",
            TierType::ContinuousNumeric => "continuous_numeric",
            TierType::ContinuousVector => "continuous_vector",
            TierType::CategoricalSampled => "categorical_sampled",
        }
    }
}

impl std::str::FromStr for TierType {
    type Err = EngineError;

    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "interval" => TierType::Interval,
            "point" => TierType::Point,
            "reference" => TierType::Reference,
            "continuous_numeric" => TierType::ContinuousNumeric,
            "continuous_vector" => TierType::ContinuousVector,
            "categorical_sampled" => TierType::CategoricalSampled,
            other => return Err(EngineError::Corpus(format!("unknown tier type: {other}"))),
        })
    }
}

/// One annotation tier (the header row in `tier`). Annotation rows belonging
/// to this tier live in the appropriate sparse table (B2) or a Parquet
/// sidecar (B3).
#[derive(Debug, Clone)]
pub struct Tier {
    /// Tier id (primary key).
    pub id: i64,
    /// Bundle this tier belongs to.
    pub bundle_id: i64,
    /// Human-readable tier name (unique within a bundle).
    pub name: String,
    /// One of the six tier types.
    pub r#type: TierType,
    /// Optional parent tier id (for hierarchical relations like word→phone).
    pub parent_id: Option<i64>,
    /// Parent-child cardinality. `None` is treated as `"none"`.
    pub cardinality: Option<String>,
    /// JSON `schema` payload describing type-specific config.
    pub schema: Option<String>,
    /// Freeform JSON payload.
    pub extra: Option<String>,
    /// ISO 8601 UTC timestamp set at insert time.
    pub created_at: String,
}

/// Optional fields for creating a [`Tier`]. Use [`TierSpec::new`] for the
/// minimum (bundle_id, name, type).
#[derive(Debug, Clone, Default)]
pub struct TierSpec {
    /// Bundle the tier attaches to.
    pub bundle_id: i64,
    /// Tier name (must be unique within the bundle).
    pub name: String,
    /// Tier type.
    pub r#type: Option<TierType>,
    /// Optional parent tier.
    pub parent_id: Option<i64>,
    /// Parent-child cardinality (`one_to_one` | `one_to_many` | `many_to_one`
    /// | `none`). Required if `parent_id` is set; ignored otherwise.
    pub cardinality: Option<String>,
    /// JSON `schema` payload.
    pub schema: Option<String>,
    /// JSON `extra` payload.
    pub extra: Option<String>,
}

impl TierSpec {
    /// Builds a spec from the minimum required fields.
    pub fn new(bundle_id: i64, name: impl Into<String>, r#type: TierType) -> Self {
        Self {
            bundle_id,
            name: name.into(),
            r#type: Some(r#type),
            ..Default::default()
        }
    }
}

/// One interval annotation. Stored in `annotation_interval`.
#[derive(Debug, Clone)]
pub struct Interval {
    /// Annotation id.
    pub id: i64,
    /// Tier this interval belongs to.
    pub tier_id: i64,
    /// Start time in seconds (must be < `end_seconds`).
    pub start_seconds: f64,
    /// End time in seconds.
    pub end_seconds: f64,
    /// Label string (e.g. phone or word).
    pub label: Option<String>,
    /// Parent annotation id in the parent tier (required if the tier has a
    /// parent; null otherwise).
    pub parent_annotation_id: Option<i64>,
    /// Annotation status — one of the rubric-defined status strings, or
    /// `None` for an untouched annotation. See [`Project::set_interval_status`].
    pub status: Option<String>,
    /// Free-text note, settable on an annotation of any status (incl. none).
    pub note: Option<String>,
    /// Provenance link to the [`processing_run`](ProcessingRunRow) that
    /// produced this annotation (e.g. a criterion run), or `None` for a
    /// hand-made annotation. Write-once at insert; see [`Project::set_proposals`].
    pub processing_run_id: Option<i64>,
    /// Freeform JSON payload.
    pub extra: Option<String>,
}

/// Optional fields for creating an [`Interval`].
#[derive(Debug, Clone, Default)]
pub struct IntervalSpec {
    /// Tier id (required).
    pub tier_id: i64,
    /// Start time in seconds.
    pub start_seconds: f64,
    /// End time in seconds (must exceed `start_seconds`).
    pub end_seconds: f64,
    /// Label string.
    pub label: Option<String>,
    /// Parent annotation id (required if tier has a parent; rejected otherwise).
    pub parent_annotation_id: Option<i64>,
    /// Annotation status (one of the rubric-defined statuses, or `None`).
    /// Validated against the rubric's status set on insert/update.
    pub status: Option<String>,
    /// Free-text note.
    pub note: Option<String>,
    /// Provenance link to the producing [`processing_run`](ProcessingRunRow),
    /// or `None`. Honoured by [`Project::add_interval`]; left untouched by
    /// [`Project::update_interval`] (an annotation's origin doesn't change
    /// when its label/status is edited).
    pub processing_run_id: Option<i64>,
    /// JSON `extra` payload.
    pub extra: Option<String>,
}

/// One point annotation. Stored in `annotation_point`.
#[derive(Debug, Clone)]
pub struct Point {
    /// Annotation id.
    pub id: i64,
    /// Tier this point belongs to.
    pub tier_id: i64,
    /// Time in seconds.
    pub time_seconds: f64,
    /// Label string.
    pub label: Option<String>,
    /// Parent annotation id.
    pub parent_annotation_id: Option<i64>,
    /// Annotation status — one of the rubric-defined status strings, or
    /// `None`. See [`Project::set_point_status`].
    pub status: Option<String>,
    /// Free-text note, settable on an annotation of any status (incl. none).
    pub note: Option<String>,
    /// Provenance link to the [`processing_run`](ProcessingRunRow) that
    /// produced this annotation, or `None` for a hand-made one.
    pub processing_run_id: Option<i64>,
    /// JSON `extra` payload.
    pub extra: Option<String>,
}

/// Optional fields for creating a [`Point`].
#[derive(Debug, Clone, Default)]
pub struct PointSpec {
    /// Tier id (required).
    pub tier_id: i64,
    /// Time in seconds.
    pub time_seconds: f64,
    /// Label string.
    pub label: Option<String>,
    /// Parent annotation id (required if tier has a parent).
    pub parent_annotation_id: Option<i64>,
    /// Annotation status (one of the rubric-defined statuses, or `None`).
    pub status: Option<String>,
    /// Free-text note.
    pub note: Option<String>,
    /// Provenance link to the producing [`processing_run`](ProcessingRunRow),
    /// or `None`. Honoured by [`Project::add_point`]; left untouched by
    /// [`Project::update_point`].
    pub processing_run_id: Option<i64>,
    /// JSON `extra` payload.
    pub extra: Option<String>,
}

/// One reference annotation. Stored in `annotation_reference`. Points at
/// another entity / bundle / session / tier / annotation via
/// `(target_kind, target_id)`.
#[derive(Debug, Clone)]
pub struct Reference {
    /// Annotation id.
    pub id: i64,
    /// Tier this reference belongs to.
    pub tier_id: i64,
    /// Target kind: `bundle` | `session` | `speaker` | `tier` | `annotation`.
    pub target_kind: String,
    /// Target row id within the appropriate target table.
    pub target_id: i64,
    /// Label string.
    pub label: Option<String>,
    /// Parent annotation id.
    pub parent_annotation_id: Option<i64>,
    /// JSON `extra` payload.
    pub extra: Option<String>,
}

/// The project's annotation rubric (one per project): the scheme that
/// frames manual annotation — free-text guidelines, the allowed status
/// vocabulary, and per-tier-name controlled vocabularies. Stored as the
/// singleton `rubric` row (+ `rubric_status` / `rubric_tier` /
/// `controlled_vocabulary`). See [`Project::set_rubric`].
#[derive(Debug, Clone)]
pub struct Rubric {
    /// Always 1 (the rubric is a per-project singleton).
    pub id: i64,
    /// Human-readable rubric name.
    pub name: String,
    /// Monotonic version integer (bumped by callers when the scheme changes;
    /// full version history is a later slice).
    pub version: i64,
    /// Free-text annotation guidelines (the prose that used to live in a
    /// separate document).
    pub guidelines: Option<String>,
    /// ISO 8601 UTC timestamp set when the rubric was first created.
    pub created_at: String,
    /// ISO 8601 UTC timestamp updated on each [`Project::set_rubric`].
    pub updated_at: String,
}

/// One allowed annotation-status value, defined by the rubric. The status
/// set is arbitrary and user-defined (e.g. `draft` / `done` / `flagged`).
#[derive(Debug, Clone, Default)]
pub struct StatusDef {
    /// The status string stored on annotations.
    pub value: String,
    /// Optional human-readable description of what the status means.
    pub description: Option<String>,
    /// Display ordering (ascending).
    pub sort_order: i64,
}

/// The rubric's configuration for a named tier role: guidelines plus whether
/// its controlled vocabulary is closed (rejects out-of-vocab labels at
/// entry) or open (accepts them, to be soft-flagged). Keyed by tier name.
#[derive(Debug, Clone)]
pub struct RubricTier {
    /// `rubric_tier` row id.
    pub id: i64,
    /// Tier name this configuration applies to (matches [`Tier::name`]).
    pub tier_name: String,
    /// Optional per-tier annotation guidance.
    pub description: Option<String>,
    /// When true the controlled vocabulary is closed: a non-empty label not
    /// in the vocabulary is rejected on insert/update.
    pub closed_vocabulary: bool,
}

/// One controlled-vocabulary entry (an allowed label) for a tier.
#[derive(Debug, Clone, Default)]
pub struct VocabEntry {
    /// The allowed label value.
    pub value: String,
    /// Optional gloss / description of the label.
    pub description: Option<String>,
    /// Display ordering (ascending).
    pub sort_order: i64,
}

/// Result of checking a label against a tier's controlled vocabulary, for
/// UIs that want to autocomplete and soft-flag out-of-vocab labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelCheck {
    /// Whether the tier has any controlled vocabulary defined at all.
    pub has_vocabulary: bool,
    /// Whether the tier's vocabulary is closed.
    pub closed: bool,
    /// Whether the label is in the vocabulary (true for an empty label, and
    /// true when no vocabulary is defined — nothing to be out of).
    pub in_vocabulary: bool,
}

/// A criteria-engine rule (slice S2): a re-runnable rule that emits proposed
/// annotations onto a preview tier for review. `kind` is `"structured"` (a
/// JSON [`crate::criteria::CriterionRule`] body, evaluated by the engine) or
/// `"python"` (a Python-function body, evaluated in the python/app layer).
#[derive(Debug, Clone)]
pub struct Criterion {
    /// Criterion id (primary key).
    pub id: i64,
    /// Unique human-readable name.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// `"structured"` or `"python"`.
    pub kind: String,
    /// JSON rule (structured) or Python source (python).
    pub body: String,
    /// Name of the tier accepted proposals promote to. The preview tier is
    /// `"<target_tier> (auto)"`.
    pub target_tier: String,
    /// ISO 8601 UTC creation timestamp.
    pub created_at: String,
    /// ISO 8601 UTC timestamp of the last update.
    pub updated_at: String,
}

/// A campaign **target** (slice S4a): the first-class unit of annotation work —
/// a region of interest on a bundle that needs a particular kind of annotation,
/// carrying a status through the campaign lifecycle. Generated from a
/// criterion's RoI selection (`source = "criterion"`) or hand-marked
/// (`source = "manual"`). The S4b assignment layer distributes targets; the QA
/// dashboard reads completeness off `status`.
#[derive(Debug, Clone)]
pub struct Target {
    /// Target id (primary key).
    pub id: i64,
    /// FK into [`crate::Bundle`] — the file this RoI lives on.
    pub bundle_id: i64,
    /// RoI start, in seconds.
    pub start_seconds: f64,
    /// RoI end, in seconds (`> start_seconds`).
    pub end_seconds: f64,
    /// What kind of annotation work the RoI needs (usually a tier name).
    pub target_type: String,
    /// Lifecycle: `unassigned` / `assigned` / `in_progress` / `done` / `flagged`.
    pub status: String,
    /// How the target came to exist: `manual` or `criterion`.
    pub source: String,
    /// The generating criterion when `source == "criterion"`; else `None`.
    pub criterion_id: Option<i64>,
    /// Optional free-text note (e.g. why a target was flagged).
    pub note: Option<String>,
    /// ISO 8601 UTC creation timestamp.
    pub created_at: String,
    /// ISO 8601 UTC timestamp of the last update.
    pub updated_at: String,
}

/// Insert parameters for a [`Target`]. `status` defaults to `"unassigned"` and
/// `source` to `"manual"` when `None`.
#[derive(Debug, Clone, Default)]
pub struct TargetSpec {
    /// FK into [`crate::Bundle`].
    pub bundle_id: i64,
    /// RoI start, in seconds.
    pub start_seconds: f64,
    /// RoI end, in seconds (must be `> start_seconds`).
    pub end_seconds: f64,
    /// What kind of annotation work the RoI needs.
    pub target_type: String,
    /// Lifecycle status; `None` → `"unassigned"`.
    pub status: Option<String>,
    /// Origin; `None` → `"manual"`.
    pub source: Option<String>,
    /// Generating criterion (for `source = "criterion"`).
    pub criterion_id: Option<i64>,
    /// Optional note.
    pub note: Option<String>,
    /// Optional opaque JSON blob.
    pub extra: Option<String>,
}

impl TargetSpec {
    /// A manual target over `[start, end)` on `bundle_id` for `target_type`.
    pub fn new(bundle_id: i64, start_seconds: f64, end_seconds: f64, target_type: &str) -> Self {
        Self {
            bundle_id,
            start_seconds,
            end_seconds,
            target_type: target_type.to_owned(),
            ..Default::default()
        }
    }
}

/// The valid [`Target::status`] values, in lifecycle order.
pub const TARGET_STATUSES: [&str; 5] =
    ["unassigned", "assigned", "in_progress", "done", "flagged"];

/// An **assignment** (slice S4b): distributes a [`Target`] to an annotator. A
/// dedicated object, separate from annotation data and the rubric. A target may
/// carry several (overlap → agreement); each has a `role` (primary / secondary)
/// and its own per-annotator progress `status`. Created by hand or in bulk via
/// [`Project::assign_targets_randomly`].
#[derive(Debug, Clone)]
pub struct Assignment {
    /// Assignment id (primary key).
    pub id: i64,
    /// FK into [`Target`].
    pub target_id: i64,
    /// The annotator this target is assigned to (a free-text identifier).
    pub annotator: String,
    /// `"primary"` or `"secondary"`.
    pub role: String,
    /// Per-annotator progress: `assigned` / `in_progress` / `done`.
    pub status: String,
    /// The [`Project::assign_targets_randomly`] seed when batch-assigned; else
    /// `None` (hand assignment).
    pub seed: Option<i64>,
    /// ISO 8601 UTC creation timestamp.
    pub created_at: String,
    /// ISO 8601 UTC timestamp of the last update.
    pub updated_at: String,
}

/// Insert parameters for an [`Assignment`]. `role` defaults to `"primary"` and
/// `status` to `"assigned"` when `None`.
#[derive(Debug, Clone, Default)]
pub struct AssignmentSpec {
    /// FK into [`Target`].
    pub target_id: i64,
    /// The annotator (free-text identifier).
    pub annotator: String,
    /// `"primary"` / `"secondary"`; `None` → `"primary"`.
    pub role: Option<String>,
    /// `assigned` / `in_progress` / `done`; `None` → `"assigned"`.
    pub status: Option<String>,
    /// Random-assignment seed; `None` for hand assignment.
    pub seed: Option<i64>,
    /// Optional opaque JSON blob.
    pub extra: Option<String>,
}

impl AssignmentSpec {
    /// A primary assignment of `target_id` to `annotator`.
    pub fn new(target_id: i64, annotator: &str) -> Self {
        Self {
            target_id,
            annotator: annotator.to_owned(),
            ..Default::default()
        }
    }
}

/// The valid [`Assignment::status`] values, in lifecycle order.
pub const ASSIGNMENT_STATUSES: [&str; 3] = ["assigned", "in_progress", "done"];

/// The valid [`Assignment::role`] values.
pub const ASSIGNMENT_ROLES: [&str; 2] = ["primary", "secondary"];

/// Result of [`Project::export_annotator_package`].
#[derive(Debug, Clone)]
pub struct ExportSummary {
    /// The annotator the package was built for.
    pub annotator: String,
    /// The package's root directory (a self-contained sadda sub-project).
    pub path: PathBuf,
    /// Bundles included (those with a target assigned to the annotator).
    pub bundles: usize,
    /// Targets included (the annotator's).
    pub targets: usize,
    /// Assignments included.
    pub assignments: usize,
}

/// Result of [`Project::import_annotator_package`].
#[derive(Debug, Clone)]
pub struct ImportSummary {
    /// The annotator whose work was merged in (from the package manifest).
    pub annotator: String,
    /// Package bundles matched (by name) to a bundle in this project.
    pub bundles_matched: usize,
    /// Per-annotator tiers (`"<tier> [annotator]"`) created or refilled.
    pub tiers_imported: usize,
    /// Annotations copied onto those per-annotator tiers.
    pub annotations_imported: usize,
    /// Assignments advanced to `done`.
    pub assignments_marked_done: usize,
}

/// A bundle's target counts by status — the campaign progress readout
/// (slice S5). `total` equals the sum of the per-status fields.
#[derive(Debug, Clone, Default)]
pub struct ProgressCounts {
    /// All targets on the bundle.
    pub total: usize,
    /// `status = 'unassigned'`.
    pub unassigned: usize,
    /// `status = 'assigned'`.
    pub assigned: usize,
    /// `status = 'in_progress'`.
    pub in_progress: usize,
    /// `status = 'done'`.
    pub done: usize,
    /// `status = 'flagged'`.
    pub flagged: usize,
}

/// Registration row for a Parquet sidecar holding a dense tier's data.
/// Created automatically by `Project::write_continuous_numeric` /
/// `write_continuous_vector` / `write_categorical_sampled`.
#[derive(Debug, Clone)]
pub struct DerivedSignal {
    /// DerivedSignal id (primary key).
    pub id: i64,
    /// FK into [`Tier`] — unique (one sidecar per tier in v1).
    pub tier_id: i64,
    /// Path to the Parquet sidecar, relative to the project root.
    pub relative_path: String,
    /// Number of frames in the sidecar.
    pub n_frames: i64,
    /// Number of dimensions per frame (1 for `continuous_numeric` and
    /// `categorical_sampled`; >= 1 for `continuous_vector`).
    pub n_dims: i64,
    /// Sample rate in Hz. `None` for non-sampled or variable-rate signals.
    pub sample_rate_hz: Option<f64>,
    /// Dtype label: `f64`, `f32`, `utf8`, … (v1 uses `f64` and `utf8` only).
    pub dtype: String,
    /// Freeform JSON payload.
    pub extra: Option<String>,
    /// ISO 8601 UTC creation timestamp.
    pub created_at: String,
}

/// Optional fields for creating a [`Reference`].
#[derive(Debug, Clone, Default)]
pub struct ReferenceSpec {
    /// Tier id (required).
    pub tier_id: i64,
    /// Target kind (required; see [`Reference::target_kind`]).
    pub target_kind: String,
    /// Target row id (required).
    pub target_id: i64,
    /// Label string.
    pub label: Option<String>,
    /// Parent annotation id (required if tier has a parent).
    pub parent_annotation_id: Option<i64>,
    /// JSON `extra` payload.
    pub extra: Option<String>,
}

/// A row of the `recipe_run` table (F1). Persistent record of a
/// recipe block: name, parameters, and the audit timestamps.
#[derive(Debug, Clone)]
pub struct RecipeRun {
    /// Recipe id (primary key).
    pub id: i64,
    /// Human-readable name (UNIQUE per project).
    pub name: String,
    /// Sadda version recorded at `start_recipe` time.
    pub sadda_version: String,
    /// Opaque JSON parameters supplied by the caller. `None` if absent.
    pub parameters: Option<String>,
    /// ISO 8601 UTC timestamp set at insert time.
    pub started_at: String,
    /// ISO 8601 UTC timestamp set by `end_recipe`. `None` if the
    /// recipe never completed cleanly (process crash / panic).
    pub completed_at: Option<String>,
    /// `'in_progress'` | `'ok'` | `'error'`.
    pub status: String,
    /// On `status = 'error'`, the exception text. `None` otherwise.
    pub error_message: Option<String>,
}

/// A row of the `processing_run` table — the audit shape every
/// engine-side mutation that produces tiers / signals / bundles
/// emits. The script generator walks these for a recipe to emit the
/// captured calls.
#[derive(Debug, Clone)]
pub struct ProcessingRunRow {
    /// Processing-run id (primary key).
    pub id: i64,
    /// Bundle the run targeted.
    pub bundle_id: i64,
    /// `'dsp_algorithm'` | `'ml_model'` | `'clinical_measure'` | `'plugin'` | `'live_recording'` | `'criterion_run'`.
    pub kind: String,
    /// Reverse-DNS identifier of the processor (e.g. `sadda.io.eaf.import`).
    pub processor_id: String,
    /// Sadda version at run time.
    pub processor_version: String,
    /// JSON parameters; processor-specific shape.
    pub parameters: Option<String>,
    /// JSON-encoded array of new tier ids.
    pub output_tier_ids: Option<String>,
    /// ISO 8601 UTC start.
    pub started_at: String,
    /// ISO 8601 UTC finish.
    pub finished_at: Option<String>,
    /// `'ok'` | `'error'` | `'partial'`.
    pub status: String,
}

/// The `processing_run.kind` discriminator. Tells provenance queries
/// (and citation export) what produced a tier / signal: a built-in DSP
/// analysis, an ML model, a composite clinical measure, a plugin, or a
/// live capture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingRunKind {
    /// Built-in DSP (`sadda.dsp.*`) — pitch, formants, MFCC, …
    DspAlgorithm,
    /// Registry-resolved ML model inference.
    MlModel,
    /// Composite clinical measure (`sadda.clinical.*`) — AVQI, CPP, …
    ClinicalMeasure,
    /// Plugin-supplied analyzer.
    Plugin,
    /// Live microphone capture committed to a bundle.
    LiveRecording,
    /// A criterion run — the execution of an annotation criterion that
    /// materializes proposals onto a preview tier. See
    /// [`Project::run_criterion`] / [`Project::record_criterion_run`].
    CriterionRun,
}

impl ProcessingRunKind {
    /// The on-disk string stored in `processing_run.kind` (and checked
    /// by the table's `CHECK` constraint).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DspAlgorithm => "dsp_algorithm",
            Self::MlModel => "ml_model",
            Self::ClinicalMeasure => "clinical_measure",
            Self::Plugin => "plugin",
            Self::LiveRecording => "live_recording",
            Self::CriterionRun => "criterion_run",
        }
    }
}

/// Terminal status of a recorded run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingRunStatus {
    /// Completed successfully.
    Ok,
    /// Failed; `error_message` carries the detail.
    Error,
    /// Partial result (e.g. some frames dropped).
    Partial,
}

impl ProcessingRunStatus {
    /// The on-disk string stored in `processing_run.status`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error => "error",
            Self::Partial => "partial",
        }
    }
}

/// Provenance facts a caller supplies to [`Project::record_processing_run`].
/// The engine fills in `processor_version` (the sadda version), the
/// timestamps, and the active recipe id; the caller supplies the rest.
///
/// Build with [`ProcessingRunSpec::new`] and set only the fields that
/// apply (most runs leave the tier/signal id lists or `weights_checksum`
/// at their defaults).
#[derive(Debug, Clone)]
pub struct ProcessingRunSpec {
    /// Bundle the run targeted.
    pub bundle_id: i64,
    /// What kind of processor produced the result.
    pub kind: ProcessingRunKind,
    /// Reverse-DNS processor id, e.g. `sadda.dsp.pitch.autocorrelation`.
    /// Matched against the [`crate::citation`] registry for export.
    pub processor_id: String,
    /// JSON parameters (processor-specific shape), if any.
    pub parameters: Option<String>,
    /// Tier ids consumed as input.
    pub input_tier_ids: Vec<i64>,
    /// Tier ids produced.
    pub output_tier_ids: Vec<i64>,
    /// DerivedSignal ids produced (Parquet sidecars).
    pub output_signal_ids: Vec<i64>,
    /// Model weights checksum, for `ml_model` runs.
    pub weights_checksum: Option<String>,
    /// Terminal status (defaults to `Ok`).
    pub status: ProcessingRunStatus,
    /// Failure detail when `status` is `Error` / `Partial`.
    pub error_message: Option<String>,
}

impl ProcessingRunSpec {
    /// A successful run with no inputs/outputs recorded yet. Set the
    /// remaining fields directly on the returned struct.
    pub fn new(bundle_id: i64, kind: ProcessingRunKind, processor_id: impl Into<String>) -> Self {
        Self {
            bundle_id,
            kind,
            processor_id: processor_id.into(),
            parameters: None,
            input_tier_ids: Vec::new(),
            output_tier_ids: Vec::new(),
            output_signal_ids: Vec::new(),
            weights_checksum: None,
            status: ProcessingRunStatus::Ok,
            error_message: None,
        }
    }
}

impl Project {
    /// Creates a new project at `path`. The path must not exist yet. Lays
    /// down the directory tree, opens a fresh `corpus.db`, runs the full
    /// migration chain, and writes the `project.toml` marker.
    pub fn create(path: impl AsRef<Path>, name: &str) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        if root.exists() {
            // Accept an existing *empty* directory — typical for the
            // GUI workflow where the user picks an already-created
            // folder via a file dialog. Reject anything with content,
            // both to avoid scattering project files into an arbitrary
            // tree and to surface an error if a user tries to "create"
            // on top of an existing sadda project.
            let is_empty = match std::fs::read_dir(&root) {
                Ok(mut entries) => entries.next().is_none(),
                Err(_) => false,
            };
            if !is_empty {
                return Err(EngineError::Corpus(format!(
                    "project path already exists and is not empty: {}",
                    root.display()
                )));
            }
        }

        std::fs::create_dir_all(root.join("signals").join("original"))?;
        std::fs::create_dir_all(root.join("signals").join("derived"))?;
        std::fs::create_dir_all(root.join("attachments"))?;
        std::fs::create_dir_all(root.join("exports"))?;

        let mut conn = Connection::open(root.join("corpus.db"))?;
        migrations::run(&mut conn)?;
        conn.execute("INSERT INTO project (id, name) VALUES (1, ?1)", [name])?;

        let toml = format!(
            "name = \"{name}\"\nschema_version = {}\nprofile = \"phonetician\"\n",
            migrations::engine_max_version()
        );
        std::fs::write(root.join("project.toml"), toml)?;

        acquire_lock(&root)?;
        Ok(Project {
            root,
            conn,
            recipe_run_id: std::cell::Cell::new(None),
            holds_lock: true,
        })
    }

    /// Opens an existing project at `path`. Applies any pending migrations
    /// first, writing a `corpus.db.bak.<old_version>` backup beforehand.
    /// Refuses to open a database whose schema version exceeds this engine's
    /// (returning [`EngineError::SchemaTooNew`]).
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        let db_path = root.join("corpus.db");
        if !db_path.exists() {
            return Err(EngineError::Corpus(format!(
                "not a sadda project (no corpus.db): {}",
                root.display()
            )));
        }
        let mut conn = Connection::open(&db_path)?;

        let db_version = migrations::current_db_version(&conn)?;
        let engine_max = migrations::engine_max_version();
        if db_version > engine_max {
            return Err(EngineError::SchemaTooNew {
                db_version,
                engine_max,
            });
        }
        if db_version < engine_max {
            backup_corpus_db(&conn, &db_path, db_version)?;
            migrations::run(&mut conn)?;
        }

        acquire_lock(&root)?;
        Ok(Project {
            root,
            conn,
            recipe_run_id: std::cell::Cell::new(None),
            holds_lock: true,
        })
    }

    /// Returns the project's filesystem root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Lightweight check: returns `true` if `path` looks like a sadda
    /// project root (`project.toml` + `corpus.db` both exist). Does
    /// **not** open the database — use when the cost of a real
    /// [`Project::open`] is wasteful, e.g. greying out recent-projects
    /// rows that have been moved or deleted on disk.
    pub fn is_project_root(path: impl AsRef<Path>) -> bool {
        let p = path.as_ref();
        p.join("project.toml").is_file() && p.join("corpus.db").is_file()
    }

    /// Returns the project's human-readable name (from the singleton row in
    /// the `project` table).
    pub fn name(&self) -> Result<String> {
        let name: String =
            self.conn
                .query_row("SELECT name FROM project WHERE id = 1", [], |row| {
                    row.get(0)
                })?;
        Ok(name)
    }

    /// Path to the project's `project.toml` marker/config file.
    fn project_toml_path(&self) -> PathBuf {
        self.root.join("project.toml")
    }

    /// Parses `project.toml` into a TOML table (empty table if absent).
    fn read_project_toml(&self) -> Result<toml::Table> {
        let path = self.project_toml_path();
        if !path.exists() {
            return Ok(toml::Table::new());
        }
        let text = std::fs::read_to_string(&path)?;
        toml::from_str(&text)
            .map_err(|e| EngineError::RefDist(format!("invalid project.toml: {e}")))
    }

    /// Pins a reference distribution `id` to a specific `version` for this
    /// project, recorded under `[refdist]` in `project.toml` so the choice
    /// travels with the project and reopens reproducibly (C7). Overwrites
    /// any existing pin for the same id.
    pub fn pin_refdist(&self, id: &str, version: &str) -> Result<()> {
        let mut doc = self.read_project_toml()?;
        let table = doc
            .entry("refdist".to_string())
            .or_insert_with(|| toml::Value::Table(toml::Table::new()));
        let toml::Value::Table(pins) = table else {
            return Err(EngineError::RefDist(
                "project.toml [refdist] is not a table".into(),
            ));
        };
        pins.insert(id.to_string(), toml::Value::String(version.to_string()));
        std::fs::write(self.project_toml_path(), toml::to_string(&doc).unwrap())?;
        Ok(())
    }

    /// The reference distributions this project has pinned, as
    /// `(id, version)` pairs sorted by id.
    pub fn refdist_pins(&self) -> Result<Vec<(String, String)>> {
        let doc = self.read_project_toml()?;
        let mut out = Vec::new();
        if let Some(toml::Value::Table(pins)) = doc.get("refdist") {
            for (id, v) in pins {
                if let Some(version) = v.as_str() {
                    out.push((id.clone(), version.to_string()));
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// Removes a reference-distribution pin. Returns `true` if a pin for
    /// `id` was present.
    pub fn remove_refdist_pin(&self, id: &str) -> Result<bool> {
        let mut doc = self.read_project_toml()?;
        let removed = match doc.get_mut("refdist") {
            Some(toml::Value::Table(pins)) => pins.remove(id).is_some(),
            _ => false,
        };
        if removed {
            std::fs::write(self.project_toml_path(), toml::to_string(&doc).unwrap())?;
        }
        Ok(removed)
    }

    /// Registers a new bundle by copying `source_audio_path` into the project's
    /// `signals/original/` directory and recording its metadata in the corpus
    /// database. Returns the new bundle's id. Convenience wrapper around
    /// [`Project::add_bundle_with`] for the common name-only case.
    pub fn add_bundle(&self, name: &str, source_audio_path: impl AsRef<Path>) -> Result<i64> {
        self.add_bundle_with(&BundleSpec::new(name), source_audio_path)
    }

    /// Registers a new bundle from a [`BundleSpec`], optionally attaching it
    /// to a `Session` or `Speaker` and carrying a JSON `extra` payload.
    pub fn add_bundle_with(
        &self,
        spec: &BundleSpec,
        source_audio_path: impl AsRef<Path>,
    ) -> Result<i64> {
        let source = source_audio_path.as_ref();
        let audio = Audio::from_wav_path(source)?;

        let filename = source
            .file_name()
            .ok_or_else(|| EngineError::Corpus("source path has no filename".into()))?;
        let dest_rel = Path::new("signals").join("original").join(filename);
        let dest_abs = self.root.join(&dest_rel);
        if dest_abs.exists() {
            return Err(EngineError::Corpus(format!(
                "destination already exists: {}",
                dest_abs.display()
            )));
        }
        std::fs::copy(source, &dest_abs)?;

        let id: i64 = self.conn.query_row(
            "INSERT INTO bundle \
                (name, audio_relative_path, sample_rate, channels, n_frames,
                 session_id, speaker_id, extra) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) RETURNING id",
            rusqlite::params![
                spec.name,
                dest_rel.to_string_lossy().as_ref(),
                audio.sample_rate as i64,
                audio.channels as i64,
                audio.frame_count() as i64,
                spec.session_id,
                spec.speaker_id,
                spec.extra,
            ],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Lists all bundles in id order.
    pub fn bundles(&self) -> Result<Vec<Bundle>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, audio_relative_path, sample_rate, channels, n_frames, \
                    session_id, speaker_id, extra \
             FROM bundle ORDER BY id",
        )?;
        let bundles = stmt
            .query_map([], |row| {
                Ok(Bundle {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    audio_relative_path: row.get(2)?,
                    sample_rate: row.get::<_, i64>(3)? as u32,
                    channels: row.get::<_, i64>(4)? as u16,
                    n_frames: row.get::<_, i64>(5)? as usize,
                    session_id: row.get(6)?,
                    speaker_id: row.get(7)?,
                    extra: row.get(8)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(bundles)
    }

    /// Loads the audio for a bundle.
    pub fn load_audio(&self, bundle_id: i64) -> Result<Audio> {
        let rel_path: String = self.conn.query_row(
            "SELECT audio_relative_path FROM bundle WHERE id = ?1",
            [bundle_id],
            |row| row.get(0),
        )?;
        Audio::from_wav_path(self.root.join(rel_path))
    }

    /// Deletes a bundle along with everything that hangs off it:
    /// tiers, annotation rows on those tiers, `derived_signal`
    /// sidecar registrations, and `processing_run` audit rows.
    /// The bundle's WAV under `signals/original/<name>.wav` is
    /// best-effort removed too. All DB writes happen in one
    /// transaction; if the SQL fails, the WAV is untouched. If the
    /// WAV remove fails, the DB still committed (the audit trail
    /// remains consistent; orphan WAV is recoverable).
    ///
    /// Idempotent on missing bundle id — returns `Ok(())`.
    pub fn delete_bundle(&self, bundle_id: i64) -> Result<()> {
        // Fetch audio path before deletion so we can clean up the
        // file after the transaction commits. Missing bundle → no-op.
        let audio_rel: Option<String> = self
            .conn
            .query_row(
                "SELECT audio_relative_path FROM bundle WHERE id = ?1",
                [bundle_id],
                |row| row.get(0),
            )
            .optional()?;
        let Some(audio_rel) = audio_rel else {
            return Ok(());
        };

        let tx = self.conn.unchecked_transaction()?;
        // Topological cascade: deepest tables first.
        tx.execute(
            "DELETE FROM derived_signal \
              WHERE tier_id IN (SELECT id FROM tier WHERE bundle_id = ?1)",
            [bundle_id],
        )?;
        for child in [
            "annotation_interval",
            "annotation_point",
            "annotation_reference",
        ] {
            tx.execute(
                &format!(
                    "DELETE FROM {child} \
                      WHERE tier_id IN (SELECT id FROM tier WHERE bundle_id = ?1)"
                ),
                [bundle_id],
            )?;
        }
        tx.execute("DELETE FROM tier WHERE bundle_id = ?1", [bundle_id])?;
        tx.execute(
            "DELETE FROM processing_run WHERE bundle_id = ?1",
            [bundle_id],
        )?;
        tx.execute("DELETE FROM bundle WHERE id = ?1", [bundle_id])?;
        tx.commit()?;

        // Best-effort WAV removal; an orphan file is recoverable
        // (the user can re-import it as a new bundle).
        let _ = std::fs::remove_file(self.root.join(audio_rel));
        Ok(())
    }

    /// Renames a bundle's display name. Updates only the `name`
    /// column — the on-disk WAV (`audio_relative_path`) is keyed
    /// independently of the display name and is left untouched.
    ///
    /// The new name is trimmed; an empty / whitespace-only name is
    /// rejected. Errors if `bundle_id` does not exist. Bundle names
    /// are not unique-constrained, so duplicates are permitted. The
    /// `UPDATE` fires the bundle audit trigger, recording the change
    /// in `audit_log`.
    pub fn rename_bundle(&self, bundle_id: i64, new_name: &str) -> Result<()> {
        let trimmed = new_name.trim();
        if trimmed.is_empty() {
            return Err(EngineError::Corpus("bundle name must not be empty".into()));
        }
        let affected = self.conn.execute(
            "UPDATE bundle SET name = ?1 WHERE id = ?2",
            rusqlite::params![trimmed, bundle_id],
        )?;
        if affected == 0 {
            return Err(EngineError::Corpus(format!(
                "no bundle with id {bundle_id}"
            )));
        }
        Ok(())
    }

    /// Inserts a [`Speaker`] row. Returns the new speaker's id.
    pub fn add_speaker(&self, spec: &SpeakerSpec) -> Result<i64> {
        let id: i64 = self.conn.query_row(
            "INSERT INTO speaker (name, sex, birth_year, notes, extra) \
             VALUES (?1, ?2, ?3, ?4, ?5) RETURNING id",
            rusqlite::params![spec.name, spec.sex, spec.birth_year, spec.notes, spec.extra],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Lists all speakers in id order.
    pub fn speakers(&self) -> Result<Vec<Speaker>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, sex, birth_year, notes, extra, created_at \
             FROM speaker ORDER BY id",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(Speaker {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    sex: row.get(2)?,
                    birth_year: row.get(3)?,
                    notes: row.get(4)?,
                    extra: row.get(5)?,
                    created_at: row.get(6)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Fetches a single speaker by id. Returns
    /// [`EngineError::Sqlite`] (`QueryReturnedNoRows`) if no row matches.
    pub fn get_speaker(&self, id: i64) -> Result<Speaker> {
        let speaker = self.conn.query_row(
            "SELECT id, name, sex, birth_year, notes, extra, created_at \
             FROM speaker WHERE id = ?1",
            [id],
            |row| {
                Ok(Speaker {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    sex: row.get(2)?,
                    birth_year: row.get(3)?,
                    notes: row.get(4)?,
                    extra: row.get(5)?,
                    created_at: row.get(6)?,
                })
            },
        )?;
        Ok(speaker)
    }

    /// Inserts a [`Session`] row. Returns the new session's id.
    pub fn add_session(&self, spec: &SessionSpec) -> Result<i64> {
        let id: i64 = self.conn.query_row(
            "INSERT INTO session \
                (name, started_at, ended_at, location, instrument_id,
                 protocol_id, notes, extra) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) RETURNING id",
            rusqlite::params![
                spec.name,
                spec.started_at,
                spec.ended_at,
                spec.location,
                spec.instrument_id,
                spec.protocol_id,
                spec.notes,
                spec.extra,
            ],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Lists all sessions in id order.
    pub fn sessions(&self) -> Result<Vec<Session>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, started_at, ended_at, location, \
                    instrument_id, protocol_id, notes, extra, created_at \
             FROM session ORDER BY id",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(Session {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    started_at: row.get(2)?,
                    ended_at: row.get(3)?,
                    location: row.get(4)?,
                    instrument_id: row.get(5)?,
                    protocol_id: row.get(6)?,
                    notes: row.get(7)?,
                    extra: row.get(8)?,
                    created_at: row.get(9)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Fetches a single session by id.
    pub fn get_session(&self, id: i64) -> Result<Session> {
        let session = self.conn.query_row(
            "SELECT id, name, started_at, ended_at, location, \
                    instrument_id, protocol_id, notes, extra, created_at \
             FROM session WHERE id = ?1",
            [id],
            |row| {
                Ok(Session {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    started_at: row.get(2)?,
                    ended_at: row.get(3)?,
                    location: row.get(4)?,
                    instrument_id: row.get(5)?,
                    protocol_id: row.get(6)?,
                    notes: row.get(7)?,
                    extra: row.get(8)?,
                    created_at: row.get(9)?,
                })
            },
        )?;
        Ok(session)
    }

    /// Inserts an [`Instrument`] row. Returns the new instrument's id.
    /// The [`Calibration`], if any, is serialized to JSON in the
    /// `calibration` column.
    pub fn add_instrument(&self, spec: &InstrumentSpec) -> Result<i64> {
        let cal_json = match &spec.calibration {
            Some(c) => Some(
                serde_json::to_string(c)
                    .map_err(|e| EngineError::Corpus(format!("calibration serialize: {e}")))?,
            ),
            None => None,
        };
        let id: i64 = self.conn.query_row(
            "INSERT INTO instrument (name, kind, serial, calibration, extra) \
             VALUES (?1, ?2, ?3, ?4, ?5) RETURNING id",
            rusqlite::params![spec.name, spec.kind, spec.serial, cal_json, spec.extra],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Lists all instruments in id order.
    pub fn instruments(&self) -> Result<Vec<Instrument>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, kind, serial, calibration, extra, created_at \
             FROM instrument ORDER BY id",
        )?;
        let rows = stmt
            .query_map([], Self::row_to_instrument)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Fetches a single instrument by id.
    pub fn get_instrument(&self, id: i64) -> Result<Instrument> {
        let instrument = self.conn.query_row(
            "SELECT id, name, kind, serial, calibration, extra, created_at \
             FROM instrument WHERE id = ?1",
            [id],
            Self::row_to_instrument,
        )?;
        Ok(instrument)
    }

    /// Maps a `instrument` row to [`Instrument`], leniently parsing the
    /// calibration JSON (an unparseable/legacy value reads as `None`
    /// rather than failing the whole query).
    fn row_to_instrument(row: &rusqlite::Row<'_>) -> rusqlite::Result<Instrument> {
        let cal_text: Option<String> = row.get(4)?;
        Ok(Instrument {
            id: row.get(0)?,
            name: row.get(1)?,
            kind: row.get(2)?,
            serial: row.get(3)?,
            calibration: cal_text.and_then(|s| serde_json::from_str(&s).ok()),
            extra: row.get(5)?,
            created_at: row.get(6)?,
        })
    }

    /// Resolves a bundle's calibration by walking
    /// bundle → session → instrument. Returns `None` if the bundle has
    /// no session, the session no instrument, or the instrument no
    /// calibration — i.e. levels for that bundle are dB-FS only.
    pub fn bundle_calibration(&self, bundle_id: i64) -> Result<Option<Calibration>> {
        let cal_text: Option<String> = self
            .conn
            .query_row(
                "SELECT i.calibration FROM bundle b \
                 JOIN session s ON b.session_id = s.id \
                 JOIN instrument i ON s.instrument_id = i.id \
                 WHERE b.id = ?1",
                [bundle_id],
                |row| row.get(0),
            )
            .optional()?
            .flatten();
        Ok(cal_text.and_then(|s| serde_json::from_str(&s).ok()))
    }

    /// Returns the user string written into `audit_log.user` for any mutation
    /// happening on this connection. Defaults to `"local"`.
    pub fn audit_user(&self) -> Result<String> {
        let user: String =
            self.conn
                .query_row("SELECT user FROM _audit_context WHERE id = 1", [], |row| {
                    row.get(0)
                })?;
        Ok(user)
    }

    /// Sets the user string written into `audit_log.user` for subsequent
    /// mutations on this connection. Persists in the `_audit_context`
    /// singleton; subsequent `Project::open` calls read whatever was last
    /// written (so a stale value can survive across processes — callers
    /// that care should set the user explicitly on every connection).
    pub fn set_audit_user(&self, user: &str) -> Result<()> {
        self.conn
            .execute("UPDATE _audit_context SET user = ?1 WHERE id = 1", [user])?;
        Ok(())
    }

    /// Inserts a [`Tier`] row.
    pub fn add_tier(&self, spec: &TierSpec) -> Result<i64> {
        let tier_type = spec
            .r#type
            .ok_or_else(|| EngineError::Corpus("TierSpec.type is required".into()))?;
        let id: i64 = self.conn.query_row(
            "INSERT INTO tier (bundle_id, name, type, parent_id, cardinality, schema, extra) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) RETURNING id",
            rusqlite::params![
                spec.bundle_id,
                spec.name,
                tier_type.as_str(),
                spec.parent_id,
                spec.cardinality,
                spec.schema,
                spec.extra,
            ],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Renames a tier's display name. Trims; rejects empty / whitespace-only;
    /// errors if `tier_id` does not exist. Tier names are not unique-
    /// constrained. The `UPDATE` fires the tier audit trigger. Mirrors
    /// [`rename_bundle`](Self::rename_bundle).
    pub fn rename_tier(&self, tier_id: i64, new_name: &str) -> Result<()> {
        let trimmed = new_name.trim();
        if trimmed.is_empty() {
            return Err(EngineError::Corpus("tier name must not be empty".into()));
        }
        let affected = self.conn.execute(
            "UPDATE tier SET name = ?1 WHERE id = ?2",
            rusqlite::params![trimmed, tier_id],
        )?;
        if affected == 0 {
            return Err(EngineError::Corpus(format!("no tier with id {tier_id}")));
        }
        Ok(())
    }

    /// Deletes a tier and all of its annotations (`annotation_interval` /
    /// `_point` / `_reference`) and any dense `derived_signal` row +
    /// Parquet sidecar. Refuses if the tier has child tiers (`parent_id`
    /// points at it) — delete those first — to avoid dangling parents.
    /// Errors if `tier_id` does not exist. Mirrors the cascade in
    /// [`delete_bundle`](Self::delete_bundle), scoped to one tier.
    pub fn delete_tier(&self, tier_id: i64) -> Result<()> {
        let child_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM tier WHERE parent_id = ?1",
            [tier_id],
            |row| row.get(0),
        )?;
        if child_count > 0 {
            return Err(EngineError::Corpus(format!(
                "cannot delete tier {tier_id}: it has {child_count} child tier(s); delete those first"
            )));
        }
        // Dense sidecar path (if any) for post-commit file cleanup.
        let dense_rel: Option<String> = self
            .conn
            .query_row(
                "SELECT relative_path FROM derived_signal WHERE tier_id = ?1",
                [tier_id],
                |row| row.get(0),
            )
            .optional()?;

        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM derived_signal WHERE tier_id = ?1", [tier_id])?;
        for child in [
            "annotation_interval",
            "annotation_point",
            "annotation_reference",
        ] {
            tx.execute(
                &format!("DELETE FROM {child} WHERE tier_id = ?1"),
                [tier_id],
            )?;
        }
        let affected = tx.execute("DELETE FROM tier WHERE id = ?1", [tier_id])?;
        if affected == 0 {
            return Err(EngineError::Corpus(format!("no tier with id {tier_id}")));
        }
        tx.commit()?;

        // Best-effort Parquet removal; an orphan sidecar is harmless.
        if let Some(rel) = dense_rel {
            let _ = std::fs::remove_file(self.root.join(rel));
        }
        Ok(())
    }

    /// Lists tiers for a given bundle (or every tier in the project when
    /// `bundle_id` is `None`), in id order.
    pub fn tiers(&self, bundle_id: Option<i64>) -> Result<Vec<Tier>> {
        let (sql, params): (&str, Vec<i64>) = match bundle_id {
            Some(b) => (
                "SELECT id, bundle_id, name, type, parent_id, cardinality, schema, extra, created_at \
                 FROM tier WHERE bundle_id = ?1 ORDER BY id",
                vec![b],
            ),
            None => (
                "SELECT id, bundle_id, name, type, parent_id, cardinality, schema, extra, created_at \
                 FROM tier ORDER BY id",
                vec![],
            ),
        };
        let mut stmt = self.conn.prepare(sql)?;
        let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<Tier> {
            let type_str: String = row.get(3)?;
            Ok(Tier {
                id: row.get(0)?,
                bundle_id: row.get(1)?,
                name: row.get(2)?,
                r#type: type_str
                    .parse::<TierType>()
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?,
                parent_id: row.get(4)?,
                cardinality: row.get(5)?,
                schema: row.get(6)?,
                extra: row.get(7)?,
                created_at: row.get(8)?,
            })
        };
        let rows = if params.is_empty() {
            stmt.query_map([], mapper)?
                .collect::<rusqlite::Result<Vec<_>>>()?
        } else {
            stmt.query_map(rusqlite::params_from_iter(params), mapper)?
                .collect::<rusqlite::Result<Vec<_>>>()?
        };
        Ok(rows)
    }

    /// Fetches a single tier by id.
    pub fn get_tier(&self, id: i64) -> Result<Tier> {
        let tier = self.conn.query_row(
            "SELECT id, bundle_id, name, type, parent_id, cardinality, schema, extra, created_at \
             FROM tier WHERE id = ?1",
            [id],
            |row| {
                let type_str: String = row.get(3)?;
                Ok(Tier {
                    id: row.get(0)?,
                    bundle_id: row.get(1)?,
                    name: row.get(2)?,
                    r#type: type_str
                        .parse::<TierType>()
                        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?,
                    parent_id: row.get(4)?,
                    cardinality: row.get(5)?,
                    schema: row.get(6)?,
                    extra: row.get(7)?,
                    created_at: row.get(8)?,
                })
            },
        )?;
        Ok(tier)
    }

    /// Inserts an interval annotation. Enforces parent-child cardinality at
    /// insert time (see the 2026-05-21 B2 DEVLOG entry).
    pub fn add_interval(&self, spec: &IntervalSpec) -> Result<i64> {
        let tier = self.get_tier(spec.tier_id)?;
        if tier.r#type != TierType::Interval {
            return Err(EngineError::Corpus(format!(
                "tier {} is type {:?}; expected Interval",
                tier.id, tier.r#type
            )));
        }
        self.enforce_cardinality(&tier, "annotation_interval", spec.parent_annotation_id)?;
        self.validate_annotation(&tier.name, spec.label.as_deref(), spec.status.as_deref())?;
        let id: i64 = self.conn.query_row(
            "INSERT INTO annotation_interval \
                (tier_id, start_seconds, end_seconds, label, parent_annotation_id, \
                 status, note, processing_run_id, extra) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9) RETURNING id",
            rusqlite::params![
                spec.tier_id,
                spec.start_seconds,
                spec.end_seconds,
                spec.label,
                spec.parent_annotation_id,
                spec.status,
                spec.note,
                spec.processing_run_id,
                spec.extra,
            ],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Replace-all update of an interval annotation.
    ///
    /// Updates `start_seconds`, `end_seconds`, `label`, `extra`, and
    /// `parent_annotation_id` to the values in `spec`. The `tier_id`
    /// field of `spec` must match the existing row's `tier_id`;
    /// moving an annotation between tiers is a different operation
    /// and isn't supported here.
    ///
    /// Cardinality is re-validated against the (possibly changed)
    /// `parent_annotation_id`. The V3 audit trigger captures the
    /// before/after JSON automatically.
    pub fn update_interval(&self, id: i64, spec: &IntervalSpec) -> Result<()> {
        let existing_tier_id: i64 = self.conn.query_row(
            "SELECT tier_id FROM annotation_interval WHERE id = ?1",
            [id],
            |row| row.get(0),
        )?;
        if existing_tier_id != spec.tier_id {
            return Err(EngineError::Corpus(format!(
                "update_interval: cannot move annotation {id} between tiers \
                 (was on tier {existing_tier_id}, asked for {})",
                spec.tier_id
            )));
        }
        let tier = self.get_tier(spec.tier_id)?;
        if tier.r#type != TierType::Interval {
            return Err(EngineError::Corpus(format!(
                "tier {} is type {:?}; expected Interval",
                tier.id, tier.r#type
            )));
        }
        self.enforce_cardinality(&tier, "annotation_interval", spec.parent_annotation_id)?;
        self.validate_annotation(&tier.name, spec.label.as_deref(), spec.status.as_deref())?;
        self.conn.execute(
            "UPDATE annotation_interval \
                SET start_seconds = ?2, end_seconds = ?3, label = ?4, \
                    parent_annotation_id = ?5, status = ?6, note = ?7, extra = ?8 \
              WHERE id = ?1",
            rusqlite::params![
                id,
                spec.start_seconds,
                spec.end_seconds,
                spec.label,
                spec.parent_annotation_id,
                spec.status,
                spec.note,
                spec.extra,
            ],
        )?;
        Ok(())
    }

    /// Removes an interval annotation by id. Idempotent — returns
    /// `Ok(())` even when no row matched, matching what users expect
    /// from a "remove" action. The V3 audit trigger captures the
    /// before row automatically.
    pub fn delete_interval(&self, id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM annotation_interval WHERE id = ?1", [id])?;
        Ok(())
    }

    /// Lists intervals for a tier in (start_seconds, id) order.
    pub fn intervals(&self, tier_id: i64) -> Result<Vec<Interval>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, tier_id, start_seconds, end_seconds, label, parent_annotation_id, \
                    status, note, processing_run_id, extra \
             FROM annotation_interval WHERE tier_id = ?1 ORDER BY start_seconds, id",
        )?;
        let rows = stmt
            .query_map([tier_id], |row| {
                Ok(Interval {
                    id: row.get(0)?,
                    tier_id: row.get(1)?,
                    start_seconds: row.get(2)?,
                    end_seconds: row.get(3)?,
                    label: row.get(4)?,
                    parent_annotation_id: row.get(5)?,
                    status: row.get(6)?,
                    note: row.get(7)?,
                    processing_run_id: row.get(8)?,
                    extra: row.get(9)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Inserts a point annotation.
    pub fn add_point(&self, spec: &PointSpec) -> Result<i64> {
        let tier = self.get_tier(spec.tier_id)?;
        if tier.r#type != TierType::Point {
            return Err(EngineError::Corpus(format!(
                "tier {} is type {:?}; expected Point",
                tier.id, tier.r#type
            )));
        }
        self.enforce_cardinality(&tier, "annotation_point", spec.parent_annotation_id)?;
        self.validate_annotation(&tier.name, spec.label.as_deref(), spec.status.as_deref())?;
        let id: i64 = self.conn.query_row(
            "INSERT INTO annotation_point \
                (tier_id, time_seconds, label, parent_annotation_id, status, note, \
                 processing_run_id, extra) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) RETURNING id",
            rusqlite::params![
                spec.tier_id,
                spec.time_seconds,
                spec.label,
                spec.parent_annotation_id,
                spec.status,
                spec.note,
                spec.processing_run_id,
                spec.extra,
            ],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Replace-all update of a point annotation. Same semantics as
    /// [`Self::update_interval`]: replaces `(time_seconds, label,
    /// extra, parent_annotation_id)`; rejects `tier_id` changes;
    /// re-validates cardinality.
    pub fn update_point(&self, id: i64, spec: &PointSpec) -> Result<()> {
        let existing_tier_id: i64 = self.conn.query_row(
            "SELECT tier_id FROM annotation_point WHERE id = ?1",
            [id],
            |row| row.get(0),
        )?;
        if existing_tier_id != spec.tier_id {
            return Err(EngineError::Corpus(format!(
                "update_point: cannot move annotation {id} between tiers \
                 (was on tier {existing_tier_id}, asked for {})",
                spec.tier_id
            )));
        }
        let tier = self.get_tier(spec.tier_id)?;
        if tier.r#type != TierType::Point {
            return Err(EngineError::Corpus(format!(
                "tier {} is type {:?}; expected Point",
                tier.id, tier.r#type
            )));
        }
        self.enforce_cardinality(&tier, "annotation_point", spec.parent_annotation_id)?;
        self.validate_annotation(&tier.name, spec.label.as_deref(), spec.status.as_deref())?;
        self.conn.execute(
            "UPDATE annotation_point \
                SET time_seconds = ?2, label = ?3, parent_annotation_id = ?4, \
                    status = ?5, note = ?6, extra = ?7 \
              WHERE id = ?1",
            rusqlite::params![
                id,
                spec.time_seconds,
                spec.label,
                spec.parent_annotation_id,
                spec.status,
                spec.note,
                spec.extra,
            ],
        )?;
        Ok(())
    }

    /// Removes a point annotation by id. Idempotent — returns
    /// `Ok(())` even when no row matched. The V3 audit trigger
    /// captures the before row automatically.
    pub fn delete_point(&self, id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM annotation_point WHERE id = ?1", [id])?;
        Ok(())
    }

    /// Lists points for a tier in (time_seconds, id) order.
    pub fn points(&self, tier_id: i64) -> Result<Vec<Point>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, tier_id, time_seconds, label, parent_annotation_id, status, note, \
                    processing_run_id, extra \
             FROM annotation_point WHERE tier_id = ?1 ORDER BY time_seconds, id",
        )?;
        let rows = stmt
            .query_map([tier_id], |row| {
                Ok(Point {
                    id: row.get(0)?,
                    tier_id: row.get(1)?,
                    time_seconds: row.get(2)?,
                    label: row.get(3)?,
                    parent_annotation_id: row.get(4)?,
                    status: row.get(5)?,
                    note: row.get(6)?,
                    processing_run_id: row.get(7)?,
                    extra: row.get(8)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    // ====================================================================
    // Annotation rubric (slice S1): the per-project scheme — guidelines,
    // the allowed status vocabulary, and per-tier controlled vocabularies.
    // ====================================================================

    /// Creates or updates the project's singleton rubric (guidelines +
    /// version). Returns the stored [`Rubric`]. `created_at` is preserved
    /// across updates; `updated_at` is refreshed.
    pub fn set_rubric(&self, name: &str, version: i64, guidelines: Option<&str>) -> Result<Rubric> {
        self.conn.execute(
            "INSERT INTO rubric (id, name, version, guidelines, updated_at) \
             VALUES (1, ?1, ?2, ?3, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) \
             ON CONFLICT(id) DO UPDATE SET \
                 name = excluded.name, version = excluded.version, \
                 guidelines = excluded.guidelines, \
                 updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
            rusqlite::params![name, version, guidelines],
        )?;
        self.rubric()?
            .ok_or_else(|| EngineError::Corpus("rubric missing after upsert".into()))
    }

    /// Reads the project's rubric, or `None` if none has been defined.
    pub fn rubric(&self) -> Result<Option<Rubric>> {
        self.conn
            .query_row(
                "SELECT id, name, version, guidelines, created_at, updated_at \
                 FROM rubric WHERE id = 1",
                [],
                |row| {
                    Ok(Rubric {
                        id: row.get(0)?,
                        name: row.get(1)?,
                        version: row.get(2)?,
                        guidelines: row.get(3)?,
                        created_at: row.get(4)?,
                        updated_at: row.get(5)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    /// Errors unless a rubric exists; returns the singleton id (always 1).
    fn require_rubric_id(&self) -> Result<i64> {
        if self.rubric()?.is_some() {
            Ok(1)
        } else {
            Err(EngineError::Corpus(
                "no rubric defined; call set_rubric first".into(),
            ))
        }
    }

    /// Replaces the rubric's status vocabulary with `statuses` (the allowed
    /// annotation-status strings). Requires a rubric to exist.
    pub fn set_rubric_statuses(&self, statuses: &[StatusDef]) -> Result<()> {
        let rid = self.require_rubric_id()?;
        self.conn
            .execute("DELETE FROM rubric_status WHERE rubric_id = ?1", [rid])?;
        let mut stmt = self.conn.prepare(
            "INSERT INTO rubric_status (rubric_id, value, description, sort_order) \
             VALUES (?1, ?2, ?3, ?4)",
        )?;
        for s in statuses {
            stmt.execute(rusqlite::params![rid, s.value, s.description, s.sort_order])?;
        }
        Ok(())
    }

    /// Reads the rubric's status vocabulary in (sort_order, value) order.
    pub fn rubric_statuses(&self) -> Result<Vec<StatusDef>> {
        let mut stmt = self.conn.prepare(
            "SELECT value, description, sort_order FROM rubric_status \
             WHERE rubric_id = 1 ORDER BY sort_order, value",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(StatusDef {
                    value: row.get(0)?,
                    description: row.get(1)?,
                    sort_order: row.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Creates or updates the rubric configuration for a tier name
    /// (guidelines + open/closed vocabulary). Requires a rubric to exist.
    pub fn set_rubric_tier(
        &self,
        tier_name: &str,
        description: Option<&str>,
        closed: bool,
    ) -> Result<RubricTier> {
        let rid = self.require_rubric_id()?;
        self.conn.execute(
            "INSERT INTO rubric_tier (rubric_id, tier_name, description, closed_vocabulary) \
             VALUES (?1, ?2, ?3, ?4) \
             ON CONFLICT(rubric_id, tier_name) DO UPDATE SET \
                 description = excluded.description, \
                 closed_vocabulary = excluded.closed_vocabulary",
            rusqlite::params![rid, tier_name, description, closed as i64],
        )?;
        self.rubric_tier(tier_name)?
            .ok_or_else(|| EngineError::Corpus("rubric_tier missing after upsert".into()))
    }

    /// Reads the rubric configuration for a tier name, or `None`.
    pub fn rubric_tier(&self, tier_name: &str) -> Result<Option<RubricTier>> {
        self.conn
            .query_row(
                "SELECT id, tier_name, description, closed_vocabulary \
                 FROM rubric_tier WHERE rubric_id = 1 AND tier_name = ?1",
                [tier_name],
                |row| {
                    Ok(RubricTier {
                        id: row.get(0)?,
                        tier_name: row.get(1)?,
                        description: row.get(2)?,
                        closed_vocabulary: row.get::<_, i64>(3)? != 0,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    /// `rubric_tier` row id for a tier name, if defined.
    fn rubric_tier_id(&self, tier_name: &str) -> Result<Option<i64>> {
        self.conn
            .query_row(
                "SELECT id FROM rubric_tier WHERE rubric_id = 1 AND tier_name = ?1",
                [tier_name],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    /// Replaces the controlled vocabulary (allowed labels) for a tier name.
    /// Auto-creates an open `rubric_tier` row if none exists yet. Requires a
    /// rubric to exist.
    pub fn set_controlled_vocabulary(&self, tier_name: &str, entries: &[VocabEntry]) -> Result<()> {
        let rid = self.require_rubric_id()?;
        let rt_id = match self.rubric_tier_id(tier_name)? {
            Some(id) => id,
            None => {
                self.conn.execute(
                    "INSERT INTO rubric_tier (rubric_id, tier_name, closed_vocabulary) \
                     VALUES (?1, ?2, 0)",
                    rusqlite::params![rid, tier_name],
                )?;
                self.conn.last_insert_rowid()
            }
        };
        self.conn.execute(
            "DELETE FROM controlled_vocabulary WHERE rubric_tier_id = ?1",
            [rt_id],
        )?;
        let mut stmt = self.conn.prepare(
            "INSERT INTO controlled_vocabulary (rubric_tier_id, value, description, sort_order) \
             VALUES (?1, ?2, ?3, ?4)",
        )?;
        for e in entries {
            stmt.execute(rusqlite::params![
                rt_id,
                e.value,
                e.description,
                e.sort_order
            ])?;
        }
        Ok(())
    }

    /// Reads the controlled vocabulary for a tier name in (sort_order, value)
    /// order. Empty when the tier has no rubric configuration.
    pub fn controlled_vocabulary(&self, tier_name: &str) -> Result<Vec<VocabEntry>> {
        let Some(rt_id) = self.rubric_tier_id(tier_name)? else {
            return Ok(Vec::new());
        };
        let mut stmt = self.conn.prepare(
            "SELECT value, description, sort_order FROM controlled_vocabulary \
             WHERE rubric_tier_id = ?1 ORDER BY sort_order, value",
        )?;
        let rows = stmt
            .query_map([rt_id], |row| {
                Ok(VocabEntry {
                    value: row.get(0)?,
                    description: row.get(1)?,
                    sort_order: row.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Checks a label against a tier's controlled vocabulary (for UIs that
    /// autocomplete and soft-flag out-of-vocab labels). An empty/`None`
    /// label, or a tier with no vocabulary, is always "in vocabulary".
    pub fn label_check(&self, tier_name: &str, label: Option<&str>) -> Result<LabelCheck> {
        let Some(rt) = self.rubric_tier(tier_name)? else {
            return Ok(LabelCheck {
                has_vocabulary: false,
                closed: false,
                in_vocabulary: true,
            });
        };
        let vocab = self.controlled_vocabulary(tier_name)?;
        let has_vocabulary = !vocab.is_empty();
        let in_vocabulary = if !has_vocabulary {
            true
        } else {
            match label {
                None | Some("") => true,
                Some(l) => vocab.iter().any(|v| v.value == l),
            }
        };
        Ok(LabelCheck {
            has_vocabulary,
            closed: rt.closed_vocabulary,
            in_vocabulary,
        })
    }

    /// Validates a status string against the rubric's status vocabulary.
    fn validate_status(&self, status: &str) -> Result<()> {
        let statuses = self.rubric_statuses()?;
        if statuses.iter().any(|s| s.value == status) {
            Ok(())
        } else {
            let defined: Vec<&str> = statuses.iter().map(|s| s.value.as_str()).collect();
            Err(EngineError::Corpus(format!(
                "annotation status {status:?} is not defined in the rubric (defined: {defined:?})"
            )))
        }
    }

    /// Validates a label + status pair for an annotation on `tier_name`:
    /// status must be a rubric-defined value (when set), and a closed tier
    /// rejects a non-empty out-of-vocabulary label.
    fn validate_annotation(
        &self,
        tier_name: &str,
        label: Option<&str>,
        status: Option<&str>,
    ) -> Result<()> {
        if let Some(s) = status {
            self.validate_status(s)?;
        }
        if let Some(l) = label {
            if !l.is_empty() {
                let check = self.label_check(tier_name, Some(l))?;
                if check.closed && !check.in_vocabulary {
                    return Err(EngineError::Corpus(format!(
                        "label {l:?} is not in the closed controlled vocabulary \
                         for tier {tier_name:?}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Sets the status + note on an interval annotation, validating the
    /// status against the rubric. Either may be `None` to clear it.
    pub fn set_interval_status(
        &self,
        id: i64,
        status: Option<&str>,
        note: Option<&str>,
    ) -> Result<()> {
        if let Some(s) = status {
            self.validate_status(s)?;
        }
        let n = self.conn.execute(
            "UPDATE annotation_interval SET status = ?2, note = ?3 WHERE id = ?1",
            rusqlite::params![id, status, note],
        )?;
        if n == 0 {
            return Err(EngineError::Corpus(format!(
                "no interval annotation with id {id}"
            )));
        }
        Ok(())
    }

    /// Sets the status + note on a point annotation, validating the status
    /// against the rubric. Either may be `None` to clear it.
    pub fn set_point_status(
        &self,
        id: i64,
        status: Option<&str>,
        note: Option<&str>,
    ) -> Result<()> {
        if let Some(s) = status {
            self.validate_status(s)?;
        }
        let n = self.conn.execute(
            "UPDATE annotation_point SET status = ?2, note = ?3 WHERE id = ?1",
            rusqlite::params![id, status, note],
        )?;
        if n == 0 {
            return Err(EngineError::Corpus(format!(
                "no point annotation with id {id}"
            )));
        }
        Ok(())
    }

    // ====================================================================
    // Criteria engine (slice S2): re-runnable rules that emit proposed
    // annotations onto a preview ("auto") tier; accept promotes them.
    // ====================================================================

    /// Creates or updates a criterion (upsert by name). Validates that a
    /// `structured` body parses as a rule.
    pub fn set_criterion(
        &self,
        name: &str,
        description: Option<&str>,
        kind: &str,
        body: &str,
        target_tier: &str,
    ) -> Result<Criterion> {
        if !matches!(kind, "structured" | "python") {
            return Err(EngineError::Corpus(format!(
                "criterion kind must be 'structured' or 'python', got {kind:?}"
            )));
        }
        if kind == "structured" {
            crate::criteria::CriterionRule::from_json(body).map_err(EngineError::Corpus)?;
        }
        self.conn.execute(
            "INSERT INTO criterion (name, description, kind, body, target_tier, updated_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) \
             ON CONFLICT(name) DO UPDATE SET \
                 description = excluded.description, kind = excluded.kind, \
                 body = excluded.body, target_tier = excluded.target_tier, \
                 updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
            rusqlite::params![name, description, kind, body, target_tier],
        )?;
        self.criterion_by_name(name)?
            .ok_or_else(|| EngineError::Corpus("criterion missing after upsert".into()))
    }

    /// Lists all criteria in name order.
    pub fn criteria(&self) -> Result<Vec<Criterion>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, description, kind, body, target_tier, created_at, updated_at \
             FROM criterion ORDER BY name",
        )?;
        let rows = stmt
            .query_map([], Self::map_criterion_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Reads a criterion by id.
    pub fn get_criterion(&self, id: i64) -> Result<Option<Criterion>> {
        self.conn
            .query_row(
                "SELECT id, name, description, kind, body, target_tier, created_at, updated_at \
                 FROM criterion WHERE id = ?1",
                [id],
                Self::map_criterion_row,
            )
            .optional()
            .map_err(Into::into)
    }

    fn criterion_by_name(&self, name: &str) -> Result<Option<Criterion>> {
        self.conn
            .query_row(
                "SELECT id, name, description, kind, body, target_tier, created_at, updated_at \
                 FROM criterion WHERE name = ?1",
                [name],
                Self::map_criterion_row,
            )
            .optional()
            .map_err(Into::into)
    }

    fn map_criterion_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Criterion> {
        Ok(Criterion {
            id: row.get(0)?,
            name: row.get(1)?,
            description: row.get(2)?,
            kind: row.get(3)?,
            body: row.get(4)?,
            target_tier: row.get(5)?,
            created_at: row.get(6)?,
            updated_at: row.get(7)?,
        })
    }

    /// Deletes a criterion by id. Idempotent.
    pub fn delete_criterion(&self, id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM criterion WHERE id = ?1", [id])?;
        Ok(())
    }

    // ====================================================================
    // Targets (slice S4a). A `target` is the first-class unit of annotation
    // work: a region of interest on a bundle with a lifecycle status, either
    // generated from a criterion's RoI selection or hand-marked.
    // ====================================================================

    /// Inserts a target. Validates the RoI (`end > start`) and the `status` /
    /// `source` enums; defaults `status` to `"unassigned"` and `source` to
    /// `"manual"`. Returns the new id.
    pub fn add_target(&self, spec: &TargetSpec) -> Result<i64> {
        if spec.end_seconds <= spec.start_seconds {
            return Err(EngineError::Corpus(format!(
                "target RoI must have end > start, got [{}, {}]",
                spec.start_seconds, spec.end_seconds
            )));
        }
        let status = spec.status.as_deref().unwrap_or("unassigned");
        Self::validate_target_status(status)?;
        let source = spec.source.as_deref().unwrap_or("manual");
        if !matches!(source, "manual" | "criterion") {
            return Err(EngineError::Corpus(format!(
                "target source must be 'manual' or 'criterion', got {source:?}"
            )));
        }
        let id: i64 = self.conn.query_row(
            "INSERT INTO target \
                (bundle_id, start_seconds, end_seconds, target_type, status, source, \
                 criterion_id, note, extra) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9) RETURNING id",
            rusqlite::params![
                spec.bundle_id,
                spec.start_seconds,
                spec.end_seconds,
                spec.target_type,
                status,
                source,
                spec.criterion_id,
                spec.note,
                spec.extra,
            ],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Lists a bundle's targets in time order (then id).
    pub fn targets(&self, bundle_id: i64) -> Result<Vec<Target>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, bundle_id, start_seconds, end_seconds, target_type, status, source, \
                    criterion_id, note, created_at, updated_at \
             FROM target WHERE bundle_id = ?1 ORDER BY start_seconds, id",
        )?;
        let rows = stmt
            .query_map([bundle_id], Self::map_target_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Reads a target by id.
    pub fn get_target(&self, id: i64) -> Result<Option<Target>> {
        self.conn
            .query_row(
                "SELECT id, bundle_id, start_seconds, end_seconds, target_type, status, source, \
                        criterion_id, note, created_at, updated_at \
                 FROM target WHERE id = ?1",
                [id],
                Self::map_target_row,
            )
            .optional()
            .map_err(Into::into)
    }

    /// Moves a target through its lifecycle: sets `status` (validated) and
    /// bumps `updated_at`. Errors if the target does not exist.
    pub fn update_target_status(&self, id: i64, status: &str) -> Result<()> {
        Self::validate_target_status(status)?;
        let n = self.conn.execute(
            "UPDATE target SET status = ?1, \
                 updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
             WHERE id = ?2",
            rusqlite::params![status, id],
        )?;
        if n == 0 {
            return Err(EngineError::Corpus(format!("no target with id {id}")));
        }
        Ok(())
    }

    /// Sets (or clears, with `None`) a target's note and bumps `updated_at`.
    /// Errors if the target does not exist.
    pub fn set_target_note(&self, id: i64, note: Option<&str>) -> Result<()> {
        let n = self.conn.execute(
            "UPDATE target SET note = ?1, \
                 updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
             WHERE id = ?2",
            rusqlite::params![note, id],
        )?;
        if n == 0 {
            return Err(EngineError::Corpus(format!("no target with id {id}")));
        }
        Ok(())
    }

    /// Deletes a target by id, along with any assignments on it. Idempotent.
    pub fn delete_target(&self, id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM assignment WHERE target_id = ?1", [id])?;
        self.conn.execute("DELETE FROM target WHERE id = ?1", [id])?;
        Ok(())
    }

    /// Generates targets from a `structured` criterion's RoI selection on
    /// `bundle_id`: each surviving select interval (after label / within /
    /// overlaps / `where` filtering) becomes one target whose RoI is that
    /// interval, `target_type` is the criterion's target tier, `source` is
    /// `"criterion"`, and `criterion_id` links back. Re-running REPLACES this
    /// criterion's prior targets on this bundle (mirrors `run_criterion`'s
    /// replace-proposals semantics). Returns the target count.
    ///
    /// `python` criteria are rejected here (they run in the python/app layer),
    /// exactly as [`run_criterion`](Self::run_criterion).
    pub fn generate_targets_from_criterion(
        &self,
        criterion_id: i64,
        bundle_id: i64,
    ) -> Result<usize> {
        let crit = self
            .get_criterion(criterion_id)?
            .ok_or_else(|| EngineError::Corpus(format!("no criterion with id {criterion_id}")))?;
        if crit.kind != "structured" {
            return Err(EngineError::Corpus(format!(
                "criterion {criterion_id} is kind '{}'; only 'structured' criteria generate \
                 targets in the engine (python criteria run in the sadda.app layer)",
                crit.kind
            )));
        }
        let rule =
            crate::criteria::CriterionRule::from_json(&crit.body).map_err(EngineError::Corpus)?;
        let select_ivs = self
            .eval_intervals(bundle_id, &rule.select.tier)?
            .ok_or_else(|| {
                EngineError::Corpus(format!(
                    "select tier {:?} not found on bundle {bundle_id}",
                    rule.select.tier
                ))
            })?;
        let within_ivs = match &rule.within {
            Some(s) => self.eval_intervals(bundle_id, &s.tier)?.unwrap_or_default(),
            None => Vec::new(),
        };
        let overlaps_ivs = match &rule.overlaps {
            Some(s) => self.eval_intervals(bundle_id, &s.tier)?.unwrap_or_default(),
            None => Vec::new(),
        };
        let signal_names = rule.referenced_signals().map_err(EngineError::Corpus)?;
        let signals = self.signal_set(bundle_id, &signal_names)?;
        let rois =
            crate::criteria::select_rois(&rule, &select_ivs, &within_ivs, &overlaps_ivs, &signals)
                .map_err(EngineError::Corpus)?;

        // Replace this criterion's prior targets on this bundle. (S4b will make
        // regeneration assignment-aware so progressed work isn't discarded.)
        self.conn.execute(
            "DELETE FROM target WHERE bundle_id = ?1 AND criterion_id = ?2",
            rusqlite::params![bundle_id, criterion_id],
        )?;
        for roi in &rois {
            self.add_target(&TargetSpec {
                bundle_id,
                start_seconds: roi.start,
                end_seconds: roi.end,
                target_type: crit.target_tier.clone(),
                source: Some("criterion".into()),
                criterion_id: Some(criterion_id),
                ..Default::default()
            })?;
        }
        Ok(rois.len())
    }

    fn validate_target_status(status: &str) -> Result<()> {
        if !TARGET_STATUSES.contains(&status) {
            return Err(EngineError::Corpus(format!(
                "invalid target status {status:?}; must be one of {TARGET_STATUSES:?}"
            )));
        }
        Ok(())
    }

    fn map_target_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Target> {
        Ok(Target {
            id: row.get(0)?,
            bundle_id: row.get(1)?,
            start_seconds: row.get(2)?,
            end_seconds: row.get(3)?,
            target_type: row.get(4)?,
            status: row.get(5)?,
            source: row.get(6)?,
            criterion_id: row.get(7)?,
            note: row.get(8)?,
            created_at: row.get(9)?,
            updated_at: row.get(10)?,
        })
    }

    // ====================================================================
    // Assignments (slice S4b). An `assignment` distributes a target to an
    // annotator; targets can carry several. Creating one advances the target
    // from `unassigned` to `assigned`; removing the last one reverts it.
    // ====================================================================

    /// Assigns a target to an annotator. Validates the `role` / `status` enums
    /// and that the target exists; advances the target's status from
    /// `unassigned` to `assigned`. The `(target, annotator)` pair is unique —
    /// re-assigning the same annotator errors. Returns the new assignment id.
    pub fn add_assignment(&self, spec: &AssignmentSpec) -> Result<i64> {
        let role = spec.role.as_deref().unwrap_or("primary");
        Self::validate_assignment_role(role)?;
        let status = spec.status.as_deref().unwrap_or("assigned");
        Self::validate_assignment_status(status)?;
        if spec.annotator.trim().is_empty() {
            return Err(EngineError::Corpus("assignment annotator is empty".into()));
        }
        let target = self.get_target(spec.target_id)?.ok_or_else(|| {
            EngineError::Corpus(format!("no target with id {}", spec.target_id))
        })?;
        let id: i64 = self.conn.query_row(
            "INSERT INTO assignment (target_id, annotator, role, status, seed) \
             VALUES (?1, ?2, ?3, ?4, ?5) RETURNING id",
            rusqlite::params![spec.target_id, spec.annotator, role, status, spec.seed],
            |row| row.get(0),
        )?;
        if target.status == "unassigned" {
            self.update_target_status(spec.target_id, "assigned")?;
        }
        Ok(id)
    }

    /// Lists a bundle's assignments (joined through their targets), ordered by
    /// target then assignment id.
    pub fn assignments(&self, bundle_id: i64) -> Result<Vec<Assignment>> {
        let mut stmt = self.conn.prepare(
            "SELECT a.id, a.target_id, a.annotator, a.role, a.status, a.seed, \
                    a.created_at, a.updated_at \
             FROM assignment a JOIN target t ON a.target_id = t.id \
             WHERE t.bundle_id = ?1 ORDER BY a.target_id, a.id",
        )?;
        let rows = stmt
            .query_map([bundle_id], Self::map_assignment_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Lists the assignments on a single target, ordered by id.
    pub fn assignments_for_target(&self, target_id: i64) -> Result<Vec<Assignment>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, target_id, annotator, role, status, seed, created_at, updated_at \
             FROM assignment WHERE target_id = ?1 ORDER BY id",
        )?;
        let rows = stmt
            .query_map([target_id], Self::map_assignment_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Sets an assignment's per-annotator `status` (one of
    /// `assigned` / `in_progress` / `done`). Errors if it does not exist.
    pub fn update_assignment_status(&self, id: i64, status: &str) -> Result<()> {
        Self::validate_assignment_status(status)?;
        let n = self.conn.execute(
            "UPDATE assignment SET status = ?1, \
                 updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
             WHERE id = ?2",
            rusqlite::params![status, id],
        )?;
        if n == 0 {
            return Err(EngineError::Corpus(format!("no assignment with id {id}")));
        }
        Ok(())
    }

    /// Reassigns an assignment to a different annotator (editable throughout).
    /// Errors if the assignment is missing, the annotator is empty, or the
    /// target is already assigned to that annotator.
    pub fn set_assignment_annotator(&self, id: i64, annotator: &str) -> Result<()> {
        if annotator.trim().is_empty() {
            return Err(EngineError::Corpus("assignment annotator is empty".into()));
        }
        let n = self.conn.execute(
            "UPDATE assignment SET annotator = ?1, \
                 updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
             WHERE id = ?2",
            rusqlite::params![annotator, id],
        )?;
        if n == 0 {
            return Err(EngineError::Corpus(format!("no assignment with id {id}")));
        }
        Ok(())
    }

    /// Deletes an assignment. If it was the target's last assignment and the
    /// target was merely `assigned` (not manually advanced), the target reverts
    /// to `unassigned`. Idempotent.
    pub fn delete_assignment(&self, id: i64) -> Result<()> {
        let target_id: Option<i64> = self
            .conn
            .query_row(
                "SELECT target_id FROM assignment WHERE id = ?1",
                [id],
                |row| row.get(0),
            )
            .optional()?;
        let Some(target_id) = target_id else {
            return Ok(()); // already gone
        };
        self.conn
            .execute("DELETE FROM assignment WHERE id = ?1", [id])?;
        let remaining: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM assignment WHERE target_id = ?1",
            [target_id],
            |row| row.get(0),
        )?;
        if remaining == 0 {
            if let Some(t) = self.get_target(target_id)? {
                if t.status == "assigned" {
                    self.update_target_status(target_id, "unassigned")?;
                }
            }
        }
        Ok(())
    }

    /// Distributes a bundle's currently-`unassigned` targets across `annotators`
    /// using a deterministic, seed-driven shuffle (Fisher–Yates over a
    /// splitmix64 stream — no `rand` dep, reproducible per the no-`Math.random`
    /// ethos), round-robining the shuffled order so counts differ by at most
    /// one. Each new assignment records `seed`. Already-assigned targets are
    /// left alone, so calling this again after a roster change re-randomizes
    /// only the remainder. Returns the number of targets assigned.
    pub fn assign_targets_randomly(
        &self,
        bundle_id: i64,
        annotators: &[String],
        seed: i64,
        role: Option<&str>,
    ) -> Result<usize> {
        if annotators.is_empty() || annotators.iter().all(|a| a.trim().is_empty()) {
            return Err(EngineError::Corpus(
                "assign_targets_randomly: empty roster".into(),
            ));
        }
        let role = role.unwrap_or("primary");
        Self::validate_assignment_role(role)?;
        let mut target_ids: Vec<i64> = self
            .targets(bundle_id)?
            .into_iter()
            .filter(|t| t.status == "unassigned")
            .map(|t| t.id)
            .collect();
        deterministic_shuffle(&mut target_ids, seed as u64);
        for (i, tid) in target_ids.iter().enumerate() {
            self.add_assignment(&AssignmentSpec {
                target_id: *tid,
                annotator: annotators[i % annotators.len()].clone(),
                role: Some(role.to_owned()),
                seed: Some(seed),
                ..Default::default()
            })?;
        }
        Ok(target_ids.len())
    }

    fn validate_assignment_status(status: &str) -> Result<()> {
        if !ASSIGNMENT_STATUSES.contains(&status) {
            return Err(EngineError::Corpus(format!(
                "invalid assignment status {status:?}; must be one of {ASSIGNMENT_STATUSES:?}"
            )));
        }
        Ok(())
    }

    fn validate_assignment_role(role: &str) -> Result<()> {
        if !ASSIGNMENT_ROLES.contains(&role) {
            return Err(EngineError::Corpus(format!(
                "invalid assignment role {role:?}; must be one of {ASSIGNMENT_ROLES:?}"
            )));
        }
        Ok(())
    }

    fn map_assignment_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Assignment> {
        Ok(Assignment {
            id: row.get(0)?,
            target_id: row.get(1)?,
            annotator: row.get(2)?,
            role: row.get(3)?,
            status: row.get(4)?,
            seed: row.get(5)?,
            created_at: row.get(6)?,
            updated_at: row.get(7)?,
        })
    }

    // ====================================================================
    // Campaign packages (slice S4c). Local-first distribution: export each
    // annotator a self-contained sub-project (a real sadda project dir), they
    // work offline, the PI imports it back — landing the annotator's work on
    // per-annotator tiers `"<tier> [annotator]"` (never silently merged), with
    // `merge_tiers` as the explicit PI-driven union.
    // ====================================================================

    /// Exports a self-contained sub-project for `annotator` at `dest_dir`: the
    /// bundles with a target assigned to them (audio + sparse interval/point
    /// tiers and annotations), the annotator's targets + assignments, the
    /// (frozen) rubric, and a `sadda_export.json` manifest. The result is a
    /// normal sadda project the annotator opens and works in offline.
    ///
    /// v1 scope: dense (measure-track / vector) tiers and reference tiers are
    /// NOT copied; rubric *versioning* is S6 (the current rubric is copied as-is).
    pub fn export_annotator_package(&self, annotator: &str, dest_dir: &Path) -> Result<ExportSummary> {
        if annotator.trim().is_empty() {
            return Err(EngineError::Corpus("export: annotator is empty".into()));
        }
        // Bundles with at least one target assigned to this annotator.
        let mut assigned: Vec<(Bundle, Vec<Target>, Vec<Assignment>)> = Vec::new();
        for b in self.bundles()? {
            let assigns: Vec<Assignment> = self
                .assignments(b.id)?
                .into_iter()
                .filter(|a| a.annotator == annotator)
                .collect();
            if assigns.is_empty() {
                continue;
            }
            let tids: std::collections::HashSet<i64> = assigns.iter().map(|a| a.target_id).collect();
            let targets: Vec<Target> = self
                .targets(b.id)?
                .into_iter()
                .filter(|t| tids.contains(&t.id))
                .collect();
            assigned.push((b, targets, assigns));
        }
        if assigned.is_empty() {
            return Err(EngineError::Corpus(format!(
                "export: no assignments for annotator {annotator:?}"
            )));
        }

        let pkg = Project::create(dest_dir, &format!("{} [{annotator}]", self.name()?))?;
        let (mut n_targets, mut n_assignments) = (0usize, 0usize);
        for (b, targets, assigns) in &assigned {
            let src_audio = self.root.join(&b.audio_relative_path);
            let new_bundle = pkg.add_bundle(&b.name, &src_audio)?;
            self.copy_bundle_sparse_tiers(b.id, &pkg, new_bundle)?;
            let mut tmap: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
            for t in targets {
                let new_t = pkg.add_target(&TargetSpec {
                    bundle_id: new_bundle,
                    start_seconds: t.start_seconds,
                    end_seconds: t.end_seconds,
                    target_type: t.target_type.clone(),
                    status: Some(t.status.clone()),
                    source: Some(t.source.clone()),
                    criterion_id: None, // the criterion itself is not exported in v1
                    note: t.note.clone(),
                    extra: None,
                })?;
                tmap.insert(t.id, new_t);
                n_targets += 1;
            }
            for a in assigns {
                if let Some(&new_t) = tmap.get(&a.target_id) {
                    pkg.add_assignment(&AssignmentSpec {
                        target_id: new_t,
                        annotator: a.annotator.clone(),
                        role: Some(a.role.clone()),
                        status: Some(a.status.clone()),
                        seed: a.seed,
                        extra: None,
                    })?;
                    n_assignments += 1;
                }
            }
        }
        self.copy_rubric_into(&pkg)?;
        write_export_manifest(dest_dir, annotator, &self.name()?)?;
        Ok(ExportSummary {
            annotator: annotator.to_owned(),
            path: dest_dir.to_path_buf(),
            bundles: assigned.len(),
            targets: n_targets,
            assignments: n_assignments,
        })
    }

    /// Imports a returned annotator package (written by
    /// [`export_annotator_package`](Self::export_annotator_package)) at
    /// `package_dir`, merging the annotator's work back. For each package bundle
    /// matched by **name** to a bundle here, each assigned target-type tier is
    /// landed on a per-annotator tier `"<tier> [annotator]"` (created or
    /// refilled — never merged into the canonical tier; use
    /// [`merge_tiers`](Self::merge_tiers) for that), and the annotator's
    /// assignments on the matched bundles are advanced to `done`.
    pub fn import_annotator_package(&self, package_dir: &Path) -> Result<ImportSummary> {
        let manifest = read_export_manifest(package_dir)?;
        let annotator = manifest.annotator;
        let pkg = Project::open(package_dir)?;
        let parent_bundles = self.bundles()?;
        let mut summary = ImportSummary {
            annotator: annotator.clone(),
            bundles_matched: 0,
            tiers_imported: 0,
            annotations_imported: 0,
            assignments_marked_done: 0,
        };
        for pb in pkg.bundles()? {
            let Some(parent_b) = parent_bundles.iter().find(|b| b.name == pb.name) else {
                continue;
            };
            summary.bundles_matched += 1;

            // The distinct target types the annotator was assigned on this bundle.
            let pkg_assigns: Vec<Assignment> = pkg
                .assignments(pb.id)?
                .into_iter()
                .filter(|a| a.annotator == annotator)
                .collect();
            let by_id: std::collections::HashMap<i64, Target> =
                pkg.targets(pb.id)?.into_iter().map(|t| (t.id, t)).collect();
            let mut types: Vec<String> = pkg_assigns
                .iter()
                .filter_map(|a| by_id.get(&a.target_id))
                .map(|t| t.target_type.clone())
                .collect();
            types.sort();
            types.dedup();

            for ttype in &types {
                let Some(src_tier) = pkg.tier_by_name(pb.id, ttype)? else {
                    continue;
                };
                if !matches!(src_tier.r#type, TierType::Interval | TierType::Point) {
                    continue;
                }
                let dest_name = format!("{ttype} [{annotator}]");
                let dest_tid = self.ensure_tier(parent_b.id, &dest_name, src_tier.r#type)?;
                self.clear_tier_annotations(dest_tid, src_tier.r#type)?;
                let mut n = 0usize;
                match src_tier.r#type {
                    TierType::Interval => {
                        for iv in pkg.intervals(src_tier.id)? {
                            self.add_interval(&IntervalSpec {
                                tier_id: dest_tid,
                                start_seconds: iv.start_seconds,
                                end_seconds: iv.end_seconds,
                                label: iv.label,
                                status: iv.status,
                                note: iv.note,
                                ..Default::default()
                            })?;
                            n += 1;
                        }
                    }
                    TierType::Point => {
                        for p in pkg.points(src_tier.id)? {
                            self.add_point(&PointSpec {
                                tier_id: dest_tid,
                                time_seconds: p.time_seconds,
                                label: p.label,
                                status: p.status,
                                note: p.note,
                                ..Default::default()
                            })?;
                            n += 1;
                        }
                    }
                    _ => {}
                }
                summary.tiers_imported += 1;
                summary.annotations_imported += n;
            }

            // Importing a returned package means the annotator finished their
            // work on these bundles: mark their assignments here `done`.
            for pa in self
                .assignments(parent_b.id)?
                .into_iter()
                .filter(|a| a.annotator == annotator && a.status != "done")
            {
                self.update_assignment_status(pa.id, "done")?;
                summary.assignments_marked_done += 1;
            }
        }
        Ok(summary)
    }

    /// Unions the annotations of `source_tier_names` into `dest_tier_name` on
    /// `bundle_id` (in time order), creating the destination tier if absent and
    /// replacing its contents. All sources must share one type (interval or
    /// point). This is the explicit, PI-driven merge — e.g. combining
    /// `"phones [alice]"` and `"phones [bob]"` into a reconciled `"phones"`.
    /// Returns the number of annotations written.
    pub fn merge_tiers(
        &self,
        bundle_id: i64,
        source_tier_names: &[String],
        dest_tier_name: &str,
    ) -> Result<usize> {
        if source_tier_names.is_empty() {
            return Err(EngineError::Corpus("merge_tiers: no source tiers".into()));
        }
        let mut tiers = Vec::new();
        for name in source_tier_names {
            let t = self.tier_by_name(bundle_id, name)?.ok_or_else(|| {
                EngineError::Corpus(format!("merge_tiers: no tier {name:?} on bundle {bundle_id}"))
            })?;
            tiers.push(t);
        }
        let ttype = tiers[0].r#type;
        if !matches!(ttype, TierType::Interval | TierType::Point) {
            return Err(EngineError::Corpus(
                "merge_tiers: only interval/point tiers can be merged".into(),
            ));
        }
        if tiers.iter().any(|t| t.r#type != ttype) {
            return Err(EngineError::Corpus(
                "merge_tiers: all source tiers must share one type".into(),
            ));
        }
        // Read all source annotations BEFORE clearing the destination, so a
        // destination that is also a source isn't wiped before it's read.
        let cmp = |a: f64, b: f64| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal);
        let dest_tid = self.ensure_tier(bundle_id, dest_tier_name, ttype)?;
        let mut count = 0;
        match ttype {
            TierType::Interval => {
                let mut all: Vec<Interval> = Vec::new();
                for t in &tiers {
                    all.extend(self.intervals(t.id)?);
                }
                all.sort_by(|a, b| {
                    cmp(a.start_seconds, b.start_seconds).then(cmp(a.end_seconds, b.end_seconds))
                });
                self.clear_tier_annotations(dest_tid, ttype)?;
                for iv in all {
                    self.add_interval(&IntervalSpec {
                        tier_id: dest_tid,
                        start_seconds: iv.start_seconds,
                        end_seconds: iv.end_seconds,
                        label: iv.label,
                        status: iv.status,
                        note: iv.note,
                        ..Default::default()
                    })?;
                    count += 1;
                }
            }
            TierType::Point => {
                let mut all: Vec<Point> = Vec::new();
                for t in &tiers {
                    all.extend(self.points(t.id)?);
                }
                all.sort_by(|a, b| cmp(a.time_seconds, b.time_seconds));
                self.clear_tier_annotations(dest_tid, ttype)?;
                for p in all {
                    self.add_point(&PointSpec {
                        tier_id: dest_tid,
                        time_seconds: p.time_seconds,
                        label: p.label,
                        status: p.status,
                        note: p.note,
                        ..Default::default()
                    })?;
                    count += 1;
                }
            }
            _ => {}
        }
        Ok(count)
    }

    /// Copies a bundle's sparse (interval / point) tiers and their annotations
    /// into `dest`'s `dest_bundle_id`, preserving tier hierarchy and annotation
    /// parent links via id remapping (tiers placed parent-first). Reference and
    /// dense tiers are skipped in v1.
    fn copy_bundle_sparse_tiers(
        &self,
        src_bundle_id: i64,
        dest: &Project,
        dest_bundle_id: i64,
    ) -> Result<()> {
        let tiers = parent_first_order(self.tiers(Some(src_bundle_id))?);
        let mut tier_map: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
        let mut anno_map: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
        for t in tiers {
            if !matches!(t.r#type, TierType::Interval | TierType::Point) {
                continue;
            }
            let new_tid = dest.add_tier(&TierSpec {
                bundle_id: dest_bundle_id,
                name: t.name.clone(),
                r#type: Some(t.r#type),
                parent_id: t.parent_id.and_then(|p| tier_map.get(&p).copied()),
                cardinality: t.cardinality.clone(),
                schema: t.schema.clone(),
                extra: t.extra.clone(),
            })?;
            tier_map.insert(t.id, new_tid);
            match t.r#type {
                TierType::Interval => {
                    for iv in self.intervals(t.id)? {
                        let new_id = dest.add_interval(&IntervalSpec {
                            tier_id: new_tid,
                            start_seconds: iv.start_seconds,
                            end_seconds: iv.end_seconds,
                            label: iv.label.clone(),
                            parent_annotation_id: iv
                                .parent_annotation_id
                                .and_then(|p| anno_map.get(&p).copied()),
                            status: iv.status.clone(),
                            note: iv.note.clone(),
                            extra: iv.extra.clone(),
                            ..Default::default()
                        })?;
                        anno_map.insert(iv.id, new_id);
                    }
                }
                TierType::Point => {
                    for p in self.points(t.id)? {
                        let new_id = dest.add_point(&PointSpec {
                            tier_id: new_tid,
                            time_seconds: p.time_seconds,
                            label: p.label.clone(),
                            parent_annotation_id: p
                                .parent_annotation_id
                                .and_then(|x| anno_map.get(&x).copied()),
                            status: p.status.clone(),
                            note: p.note.clone(),
                            extra: p.extra.clone(),
                            ..Default::default()
                        })?;
                        anno_map.insert(p.id, new_id);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Copies this project's rubric (name, version, guidelines, status
    /// vocabulary, and per-tier config + controlled vocabularies for the tiers
    /// already present in `dest`) into `dest`.
    fn copy_rubric_into(&self, dest: &Project) -> Result<()> {
        let Some(r) = self.rubric()? else {
            return Ok(());
        };
        dest.set_rubric(&r.name, r.version, r.guidelines.as_deref())?;
        let statuses = self.rubric_statuses()?;
        if !statuses.is_empty() {
            dest.set_rubric_statuses(&statuses)?;
        }
        for t in dest.tiers(None)? {
            if let Some(rt) = self.rubric_tier(&t.name)? {
                dest.set_rubric_tier(&t.name, rt.description.as_deref(), rt.closed_vocabulary)?;
            }
            let vocab = self.controlled_vocabulary(&t.name)?;
            if !vocab.is_empty() {
                dest.set_controlled_vocabulary(&t.name, &vocab)?;
            }
        }
        Ok(())
    }

    // ====================================================================
    // Agreement + work queue (slice S5). One comparison engine, three uses
    // (inter-annotator, auto-vs-gold, rubric-version impact); plus the
    // campaign progress readout and a next-target navigator.
    // ====================================================================

    /// Compares two tiers on `bundle_id` and reports their agreement (see
    /// [`crate::agreement`]): unit-based label κ, boundary deviation/tolerance,
    /// insertions/deletions, and — for interval tiers — a frame-based label
    /// κ/agreement. The tiers must belong to `bundle_id`, share a type, and be
    /// interval or point tiers. `opts = None` uses the defaults (20 ms boundary
    /// tolerance, 10 ms frame step).
    pub fn compare_tiers(
        &self,
        bundle_id: i64,
        tier_a_id: i64,
        tier_b_id: i64,
        opts: Option<crate::agreement::AgreementOptions>,
    ) -> Result<crate::agreement::AgreementReport> {
        let ta = self.get_tier(tier_a_id)?;
        let tb = self.get_tier(tier_b_id)?;
        for t in [&ta, &tb] {
            if t.bundle_id != bundle_id {
                return Err(EngineError::Corpus(format!(
                    "compare_tiers: tier {} is not on bundle {bundle_id}",
                    t.id
                )));
            }
        }
        if ta.r#type != tb.r#type {
            return Err(EngineError::Corpus(format!(
                "compare_tiers: tier types differ ({:?} vs {:?})",
                ta.r#type, tb.r#type
            )));
        }
        let opts = opts.unwrap_or_default();
        let report = match ta.r#type {
            TierType::Interval => {
                let to_seg = |ivs: Vec<Interval>| -> Vec<crate::agreement::Segment> {
                    ivs.into_iter()
                        .map(|i| crate::agreement::Segment {
                            start: i.start_seconds,
                            end: i.end_seconds,
                            label: i.label,
                        })
                        .collect()
                };
                let a = to_seg(self.intervals(tier_a_id)?);
                let b = to_seg(self.intervals(tier_b_id)?);
                crate::agreement::compare_intervals(&a, &b, &opts)
            }
            TierType::Point => {
                let to_mark = |pts: Vec<Point>| -> Vec<crate::agreement::Mark> {
                    pts.into_iter()
                        .map(|p| crate::agreement::Mark {
                            time: p.time_seconds,
                            label: p.label,
                        })
                        .collect()
                };
                let a = to_mark(self.points(tier_a_id)?);
                let b = to_mark(self.points(tier_b_id)?);
                crate::agreement::compare_points(&a, &b, &opts)
            }
            other => {
                return Err(EngineError::Corpus(format!(
                    "compare_tiers: unsupported tier type {other:?} (interval/point only)"
                )));
            }
        };
        Ok(report)
    }

    /// Counts a bundle's targets by status — the campaign progress readout.
    pub fn target_progress(&self, bundle_id: i64) -> Result<ProgressCounts> {
        let mut pc = ProgressCounts::default();
        for t in self.targets(bundle_id)? {
            pc.total += 1;
            match t.status.as_str() {
                "unassigned" => pc.unassigned += 1,
                "assigned" => pc.assigned += 1,
                "in_progress" => pc.in_progress += 1,
                "done" => pc.done += 1,
                "flagged" => pc.flagged += 1,
                _ => {}
            }
        }
        Ok(pc)
    }

    /// The next target on `bundle_id` whose status is in `statuses`, in time
    /// order (the work-queue navigator: e.g. `["unassigned","assigned"]` for
    /// "next to do", `["flagged"]` for "next flagged"). `None` when none match.
    pub fn next_target(&self, bundle_id: i64, statuses: &[String]) -> Result<Option<Target>> {
        Ok(self
            .targets(bundle_id)?
            .into_iter()
            .find(|t| statuses.iter().any(|s| s == &t.status)))
    }

    /// Finds a tier by name on a bundle, if present.
    fn tier_by_name(&self, bundle_id: i64, name: &str) -> Result<Option<Tier>> {
        Ok(self
            .tiers(Some(bundle_id))?
            .into_iter()
            .find(|t| t.name == name))
    }

    /// Finds a tier by name, creating it with `ttype` if absent. Errors if a
    /// tier of that name exists with a different type.
    fn ensure_tier(&self, bundle_id: i64, name: &str, ttype: TierType) -> Result<i64> {
        match self.tier_by_name(bundle_id, name)? {
            Some(t) if t.r#type == ttype => Ok(t.id),
            Some(t) => Err(EngineError::Corpus(format!(
                "tier {name:?} exists with type {:?}, expected {ttype:?}",
                t.r#type
            ))),
            None => self.add_tier(&TierSpec::new(bundle_id, name, ttype)),
        }
    }

    /// Deletes all interval/point annotations on a tier.
    fn clear_tier_annotations(&self, tier_id: i64, ttype: TierType) -> Result<()> {
        match ttype {
            TierType::Interval => {
                self.conn.execute(
                    "DELETE FROM annotation_interval WHERE tier_id = ?1",
                    [tier_id],
                )?;
            }
            TierType::Point => {
                self.conn
                    .execute("DELETE FROM annotation_point WHERE tier_id = ?1", [tier_id])?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Fetches an interval tier's rows as evaluator intervals; `None` if no
    /// tier of that name exists. Errors if the named tier isn't an interval
    /// tier (only interval tiers are selectable in v1).
    fn eval_intervals(
        &self,
        bundle_id: i64,
        tier_name: &str,
    ) -> Result<Option<Vec<crate::criteria::EvalInterval>>> {
        let Some(tier) = self.tier_by_name(bundle_id, tier_name)? else {
            return Ok(None);
        };
        if tier.r#type != TierType::Interval {
            return Err(EngineError::Corpus(format!(
                "criterion tier {tier_name:?} must be an interval tier (is {:?})",
                tier.r#type
            )));
        }
        let ivs = self
            .intervals(tier.id)?
            .into_iter()
            .map(|i| crate::criteria::EvalInterval {
                start: i.start_seconds,
                end: i.end_seconds,
                label: i.label,
            })
            .collect();
        Ok(Some(ivs))
    }

    /// The preview ("auto") tier name for a criterion's target tier.
    fn preview_tier_name(target_tier: &str) -> String {
        format!("{target_tier} (auto)")
    }

    /// Computes the `names` signals for a bundle into a
    /// [`SignalSet`](crate::criteria::SignalSet) for the criteria expression
    /// evaluator. The built-in names `f0` (voiced frames only) and `intensity`
    /// (dB-FS) are computed from the bundle audio; any other name resolves to a
    /// `continuous_numeric` measure-track tier of that name (sample times
    /// reconstructed from its stored sample rate). An unknown name errors. This
    /// is the open "signal registry" — built-ins plus every stored track.
    fn signal_set(
        &self,
        bundle_id: i64,
        names: &[String],
    ) -> Result<crate::SignalSet> {
        use crate::SampledSignal;
        let mut set = crate::SignalSet::new();
        let needs_audio = names.iter().any(|n| n == "f0" || n == "intensity");
        let audio = if needs_audio {
            Some(self.load_audio(bundle_id)?)
        } else {
            None
        };
        for name in names {
            if set.contains_key(name) {
                continue;
            }
            let sig = match name.as_str() {
                "f0" => {
                    let a = audio.as_ref().expect("audio loaded for f0");
                    let config = crate::pitch::PitchConfig::default();
                    let frames = crate::pitch::pitch(
                        a,
                        &config,
                        crate::pitch::PitchMethod::WindowedAutocorrelation,
                    );
                    let mut times = Vec::new();
                    let mut values = Vec::new();
                    // Voiced frames only — an unvoiced f0 estimate is unreliable,
                    // and excluding them makes "f0 over a fully unvoiced interval"
                    // an empty (undefined) reduction, which skips the match.
                    for f in frames {
                        if f.voicing >= config.voicing_threshold {
                            times.push(f.time_seconds);
                            values.push(f.frequency_hz.value() as f64);
                        }
                    }
                    SampledSignal { times, values }
                }
                "intensity" => {
                    let a = audio.as_ref().expect("audio loaded for intensity");
                    let mono: Vec<f32> = a.mono_samples().collect();
                    let frames = crate::dsp::intensity(&mono, a.sample_rate, 0.025, 0.010);
                    let times = frames.iter().map(|f| f.time_seconds).collect();
                    // dB-FS from the public `rms` field (mirrors the engine's own
                    // db_fs derivation) — avoids the private Decibels inner field.
                    let values = frames
                        .iter()
                        .map(|f| {
                            if f.rms > 0.0 {
                                20.0 * (f.rms as f64).log10()
                            } else {
                                -200.0
                            }
                        })
                        .collect();
                    SampledSignal { times, values }
                }
                other => {
                    let tier = self.tier_by_name(bundle_id, other)?.ok_or_else(|| {
                        EngineError::Corpus(format!(
                            "criterion references unknown signal {other:?} \
                             (not a built-in f0/intensity, nor a continuous_numeric tier)"
                        ))
                    })?;
                    let ds = self.derived_signal(tier.id)?.ok_or_else(|| {
                        EngineError::Corpus(format!(
                            "signal {other:?} is a tier but has no derived-signal data"
                        ))
                    })?;
                    let sr = ds.sample_rate_hz.ok_or_else(|| {
                        EngineError::Corpus(format!(
                            "signal tier {other:?} has no sample rate (needed to place samples in time)"
                        ))
                    })?;
                    let values = self.read_continuous_numeric(tier.id)?;
                    let times = (0..values.len()).map(|i| i as f64 / sr).collect();
                    SampledSignal { times, values }
                }
            };
            set.insert(name.clone(), sig);
        }
        Ok(set)
    }

    /// Runs a `structured` criterion against a bundle: evaluates the rule and
    /// (re)writes its proposals onto the preview tier `"<target> (auto)"`,
    /// replacing any prior proposals. Returns the proposal count. `python`
    /// criteria are rejected here — they run in the python/app layer.
    pub fn run_criterion(&self, id: i64, bundle_id: i64) -> Result<usize> {
        let crit = self
            .get_criterion(id)?
            .ok_or_else(|| EngineError::Corpus(format!("no criterion with id {id}")))?;
        if crit.kind != "structured" {
            return Err(EngineError::Corpus(format!(
                "criterion {id} is kind '{}'; only 'structured' criteria run in the engine \
                 (python criteria run in the sadda.app layer)",
                crit.kind
            )));
        }
        let rule =
            crate::criteria::CriterionRule::from_json(&crit.body).map_err(EngineError::Corpus)?;
        let select_ivs = self
            .eval_intervals(bundle_id, &rule.select.tier)?
            .ok_or_else(|| {
                EngineError::Corpus(format!(
                    "select tier {:?} not found on bundle {bundle_id}",
                    rule.select.tier
                ))
            })?;
        let within_ivs = match &rule.within {
            Some(s) => self.eval_intervals(bundle_id, &s.tier)?.unwrap_or_default(),
            None => Vec::new(),
        };
        let overlaps_ivs = match &rule.overlaps {
            Some(s) => self.eval_intervals(bundle_id, &s.tier)?.unwrap_or_default(),
            None => Vec::new(),
        };
        // S3: compute the signals the rule's where/emit expressions reference.
        let signal_names = rule.referenced_signals().map_err(EngineError::Corpus)?;
        let signals = self.signal_set(bundle_id, &signal_names)?;
        let proposals =
            crate::criteria::evaluate(&rule, &select_ivs, &within_ivs, &overlaps_ivs, &signals)
                .map_err(EngineError::Corpus)?;
        let run_id = self.record_criterion_run(&crit, bundle_id)?;
        self.set_proposals(bundle_id, &crit.target_tier, &proposals, Some(run_id))
    }

    /// (Re)writes `proposals` onto the preview tier `"<target> (auto)"`,
    /// replacing any prior proposals, and returns the count. The tier type is
    /// inferred from the proposals (all points → a point tier, all spans → an
    /// interval tier; a mix is an error). With no proposals, an existing
    /// preview tier is just cleared. This is the shared materialization step
    /// for both the engine's structured `run_criterion` and the python-escape
    /// executor in the python/app layer.
    ///
    /// `processing_run_id` stamps each written proposal with its provenance
    /// link (the [`record_criterion_run`](Self::record_criterion_run) row);
    /// pass `None` for unattributed proposals.
    pub fn set_proposals(
        &self,
        bundle_id: i64,
        target_tier: &str,
        proposals: &[crate::criteria::Proposal],
        processing_run_id: Option<i64>,
    ) -> Result<usize> {
        let preview_name = Self::preview_tier_name(target_tier);
        if proposals.is_empty() {
            if let Some(t) = self.tier_by_name(bundle_id, &preview_name)? {
                self.clear_tier_annotations(t.id, t.r#type)?;
            }
            return Ok(0);
        }
        let is_point = proposals[0].end.is_none();
        if proposals.iter().any(|p| p.end.is_none() != is_point) {
            return Err(EngineError::Corpus(
                "proposals must be all points or all spans".into(),
            ));
        }
        let preview_type = if is_point {
            TierType::Point
        } else {
            TierType::Interval
        };
        let preview_id = self.ensure_tier(bundle_id, &preview_name, preview_type)?;
        self.clear_tier_annotations(preview_id, preview_type)?;
        for p in proposals {
            if is_point {
                self.add_point(&PointSpec {
                    tier_id: preview_id,
                    time_seconds: p.start,
                    label: p.label.clone(),
                    processing_run_id,
                    ..Default::default()
                })?;
            } else {
                self.add_interval(&IntervalSpec {
                    tier_id: preview_id,
                    start_seconds: p.start,
                    end_seconds: p.end.unwrap_or(p.start),
                    label: p.label.clone(),
                    processing_run_id,
                    ..Default::default()
                })?;
            }
        }
        Ok(proposals.len())
    }

    /// Records a `processing_run` of kind `criterion_run` for an execution of
    /// `criterion` against `bundle_id`, and returns its id. The run's
    /// `parameters` JSON captures the criterion id, name, kind, a SHA-256 of
    /// its body (so "which *version* of the criterion ran" is recoverable),
    /// and the singleton rubric id when a rubric exists (rubric *versioning*
    /// lands in S6 — see the V10 migration note). Both the structured
    /// `run_criterion` and the python-escape executor go through this so a
    /// criterion run is a first-class, queryable fact for either kind.
    pub fn record_criterion_run(&self, criterion: &Criterion, bundle_id: i64) -> Result<i64> {
        let body_sha256 = {
            let mut hasher = Sha256::new();
            hasher.update(criterion.body.as_bytes());
            format!("{:x}", hasher.finalize())
        };
        let rubric_id: Option<i64> = self
            .conn
            .query_row("SELECT id FROM rubric WHERE id = 1", [], |row| row.get(0))
            .optional()?;
        let parameters = serde_json::json!({
            "criterion_id": criterion.id,
            "criterion_name": criterion.name,
            "criterion_kind": criterion.kind,
            "body_sha256": body_sha256,
            "rubric_id": rubric_id,
        })
        .to_string();
        let mut spec = ProcessingRunSpec::new(
            bundle_id,
            ProcessingRunKind::CriterionRun,
            format!("sadda.criteria.{}", criterion.name),
        );
        spec.parameters = Some(parameters);
        self.record_processing_run(&spec)
    }

    /// Promotes all proposals on `"<target> (auto)"` to the target tier
    /// (created if needed), then clears the preview tier. Each promoted row
    /// is validated against the target tier's rubric (e.g. a closed
    /// vocabulary), so an out-of-vocab proposal label surfaces an error.
    /// The proposal's `processing_run_id` provenance link is carried onto the
    /// promoted row, so accepting a proposal doesn't lose the trace of which
    /// criterion produced it. Returns the number promoted.
    pub fn accept_proposals(&self, bundle_id: i64, target_tier: &str) -> Result<usize> {
        let preview_name = Self::preview_tier_name(target_tier);
        let Some(preview) = self.tier_by_name(bundle_id, &preview_name)? else {
            return Ok(0);
        };
        let target_id = self.ensure_tier(bundle_id, target_tier, preview.r#type)?;
        let mut promoted = 0;
        match preview.r#type {
            TierType::Point => {
                for p in self.points(preview.id)? {
                    self.add_point(&PointSpec {
                        tier_id: target_id,
                        time_seconds: p.time_seconds,
                        label: p.label,
                        processing_run_id: p.processing_run_id,
                        ..Default::default()
                    })?;
                    promoted += 1;
                }
            }
            _ => {
                for iv in self.intervals(preview.id)? {
                    self.add_interval(&IntervalSpec {
                        tier_id: target_id,
                        start_seconds: iv.start_seconds,
                        end_seconds: iv.end_seconds,
                        label: iv.label,
                        processing_run_id: iv.processing_run_id,
                        ..Default::default()
                    })?;
                    promoted += 1;
                }
            }
        }
        self.clear_tier_annotations(preview.id, preview.r#type)?;
        Ok(promoted)
    }

    /// Discards all proposals on `"<target> (auto)"`. Returns the count
    /// cleared.
    pub fn clear_proposals(&self, bundle_id: i64, target_tier: &str) -> Result<usize> {
        let preview_name = Self::preview_tier_name(target_tier);
        let Some(preview) = self.tier_by_name(bundle_id, &preview_name)? else {
            return Ok(0);
        };
        let n = match preview.r#type {
            TierType::Point => self.points(preview.id)?.len(),
            _ => self.intervals(preview.id)?.len(),
        };
        self.clear_tier_annotations(preview.id, preview.r#type)?;
        Ok(n)
    }

    /// Inserts a reference annotation.
    pub fn add_reference(&self, spec: &ReferenceSpec) -> Result<i64> {
        let tier = self.get_tier(spec.tier_id)?;
        if tier.r#type != TierType::Reference {
            return Err(EngineError::Corpus(format!(
                "tier {} is type {:?}; expected Reference",
                tier.id, tier.r#type
            )));
        }
        self.enforce_cardinality(&tier, "annotation_reference", spec.parent_annotation_id)?;
        let id: i64 = self.conn.query_row(
            "INSERT INTO annotation_reference \
                (tier_id, target_kind, target_id, label, parent_annotation_id, extra) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) RETURNING id",
            rusqlite::params![
                spec.tier_id,
                spec.target_kind,
                spec.target_id,
                spec.label,
                spec.parent_annotation_id,
                spec.extra,
            ],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Lists references for a tier in id order. Named `references_for` rather
    /// than `references` because the latter shadows a Rust language keyword
    /// in common usage and the latter is more grep-able on the API surface.
    pub fn references_for(&self, tier_id: i64) -> Result<Vec<Reference>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, tier_id, target_kind, target_id, label, parent_annotation_id, extra \
             FROM annotation_reference WHERE tier_id = ?1 ORDER BY id",
        )?;
        let rows = stmt
            .query_map([tier_id], |row| {
                Ok(Reference {
                    id: row.get(0)?,
                    tier_id: row.get(1)?,
                    target_kind: row.get(2)?,
                    target_id: row.get(3)?,
                    label: row.get(4)?,
                    parent_annotation_id: row.get(5)?,
                    extra: row.get(6)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Returns the rows of a sparse tier as a Vec of [`AnnotationRow`] —
    /// the raw shape the Python layer wraps in a `polars.DataFrame`.
    pub fn tier_rows(&self, tier_id: i64) -> Result<TierRows> {
        let tier = self.get_tier(tier_id)?;
        match tier.r#type {
            TierType::Interval => Ok(TierRows::Interval(self.intervals(tier_id)?)),
            TierType::Point => Ok(TierRows::Point(self.points(tier_id)?)),
            TierType::Reference => Ok(TierRows::Reference(self.references_for(tier_id)?)),
            TierType::ContinuousNumeric
            | TierType::ContinuousVector
            | TierType::CategoricalSampled => Err(EngineError::Corpus(format!(
                "tier {} is dense ({:?}); use the Parquet sidecar APIs (B3)",
                tier.id, tier.r#type
            ))),
        }
    }

    /// Writes a `continuous_numeric` Parquet sidecar for the tier and records
    /// it in `derived_signal`. Returns the new `derived_signal.id`.
    /// Fails if the tier isn't `continuous_numeric` or already has a sidecar.
    pub fn write_continuous_numeric(
        &self,
        tier_id: i64,
        samples: &[f64],
        sample_rate_hz: f64,
    ) -> Result<i64> {
        let tier = self.get_tier(tier_id)?;
        if tier.r#type != TierType::ContinuousNumeric {
            return Err(EngineError::Corpus(format!(
                "tier {} is type {:?}; expected ContinuousNumeric",
                tier.id, tier.r#type
            )));
        }
        self.guard_no_existing_derived_signal(&tier)?;
        let (abs, rel) = self.dense_paths_for(&tier)?;
        crate::storage::dense::write_continuous_numeric(&abs, samples)?;
        self.insert_derived_signal_row(&tier, &rel, samples.len() as i64, 1, sample_rate_hz, "f64")
    }

    /// Reads a `continuous_numeric` sidecar back into a `Vec<f64>`.
    pub fn read_continuous_numeric(&self, tier_id: i64) -> Result<Vec<f64>> {
        let ds = self.expect_derived_signal(tier_id, TierType::ContinuousNumeric)?;
        crate::storage::dense::read_continuous_numeric(&self.root.join(&ds.relative_path))
    }

    /// Writes a `continuous_vector` Parquet sidecar.
    pub fn write_continuous_vector(
        &self,
        tier_id: i64,
        frames: ndarray::ArrayView2<'_, f64>,
        sample_rate_hz: f64,
    ) -> Result<i64> {
        let tier = self.get_tier(tier_id)?;
        if tier.r#type != TierType::ContinuousVector {
            return Err(EngineError::Corpus(format!(
                "tier {} is type {:?}; expected ContinuousVector",
                tier.id, tier.r#type
            )));
        }
        self.guard_no_existing_derived_signal(&tier)?;
        let (n_frames, n_dims) = frames.dim();
        let (abs, rel) = self.dense_paths_for(&tier)?;
        crate::storage::dense::write_continuous_vector(&abs, frames)?;
        self.insert_derived_signal_row(
            &tier,
            &rel,
            n_frames as i64,
            n_dims as i64,
            sample_rate_hz,
            "f64",
        )
    }

    /// Reads a `continuous_vector` sidecar back into an
    /// `ndarray::Array2<f64>` of shape `[n_frames, n_dims]`.
    pub fn read_continuous_vector(&self, tier_id: i64) -> Result<ndarray::Array2<f64>> {
        let ds = self.expect_derived_signal(tier_id, TierType::ContinuousVector)?;
        crate::storage::dense::read_continuous_vector(&self.root.join(&ds.relative_path))
    }

    /// E12: runs `model` as an embedding extractor over `bundle_id`'s audio
    /// and stores the `(frames × dims)` result as a new `continuous_vector`
    /// tier named `tier_name`. Records an `ml_model` [`ProcessingRun`]
    /// (processor = the model's id, with its weights checksum) so the tier's
    /// provenance is queryable. Returns the new tier id; the tier's frame
    /// rate is the actual `frames / duration`.
    #[cfg(feature = "ml")]
    pub fn extract_embeddings(
        &self,
        bundle_id: i64,
        model: &crate::models::Model,
        tier_name: &str,
    ) -> Result<i64> {
        let audio = self.load_audio(bundle_id)?;
        let emb = model.embeddings(&audio)?;
        let dur = audio.duration_seconds();
        let sr_out = if dur > 0.0 {
            emb.nrows() as f64 / dur
        } else {
            0.0
        };
        let tier_id = self.add_tier(&TierSpec::new(
            bundle_id,
            tier_name,
            TierType::ContinuousVector,
        ))?;
        let signal_id = self.write_continuous_vector(tier_id, emb.view(), sr_out)?;

        let mut run = ProcessingRunSpec::new(bundle_id, ProcessingRunKind::MlModel, model.id());
        run.parameters = Some(serde_json::json!({ "model_version": model.version() }).to_string());
        run.output_tier_ids = vec![tier_id];
        run.output_signal_ids = vec![signal_id];
        run.weights_checksum = model.weights_checksum().map(str::to_string);
        self.record_processing_run(&run)?;
        Ok(tier_id)
    }

    /// Writes a `categorical_sampled` Parquet sidecar.
    pub fn write_categorical_sampled(
        &self,
        tier_id: i64,
        labels: &[String],
        sample_rate_hz: f64,
    ) -> Result<i64> {
        let tier = self.get_tier(tier_id)?;
        if tier.r#type != TierType::CategoricalSampled {
            return Err(EngineError::Corpus(format!(
                "tier {} is type {:?}; expected CategoricalSampled",
                tier.id, tier.r#type
            )));
        }
        self.guard_no_existing_derived_signal(&tier)?;
        let (abs, rel) = self.dense_paths_for(&tier)?;
        crate::storage::dense::write_categorical_sampled(&abs, labels)?;
        self.insert_derived_signal_row(&tier, &rel, labels.len() as i64, 1, sample_rate_hz, "utf8")
    }

    /// Reads a `categorical_sampled` sidecar back into `Vec<String>`.
    pub fn read_categorical_sampled(&self, tier_id: i64) -> Result<Vec<String>> {
        let ds = self.expect_derived_signal(tier_id, TierType::CategoricalSampled)?;
        crate::storage::dense::read_categorical_sampled(&self.root.join(&ds.relative_path))
    }

    /// Returns the [`DerivedSignal`] row for a tier, or `None` if no sidecar
    /// has been written yet.
    pub fn derived_signal(&self, tier_id: i64) -> Result<Option<DerivedSignal>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, tier_id, relative_path, n_frames, n_dims, \
                    sample_rate_hz, dtype, extra, created_at \
             FROM derived_signal WHERE tier_id = ?1",
        )?;
        let row = stmt
            .query_row([tier_id], |row| {
                Ok(DerivedSignal {
                    id: row.get(0)?,
                    tier_id: row.get(1)?,
                    relative_path: row.get(2)?,
                    n_frames: row.get(3)?,
                    n_dims: row.get(4)?,
                    sample_rate_hz: row.get(5)?,
                    dtype: row.get(6)?,
                    extra: row.get(7)?,
                    created_at: row.get(8)?,
                })
            })
            .optional()?;
        Ok(row)
    }

    /// Returns the absolute filesystem path of a dense tier's Parquet
    /// sidecar, or `None` if no sidecar has been written yet. Intended for
    /// external readers (e.g. `polars.scan_parquet(path)` directly).
    pub fn dense_path(&self, tier_id: i64) -> Result<Option<PathBuf>> {
        Ok(self
            .derived_signal(tier_id)?
            .map(|ds| self.root.join(ds.relative_path)))
    }

    fn dense_paths_for(&self, tier: &Tier) -> Result<(PathBuf, String)> {
        let sanitized = sanitize_filename(&tier.name);
        let rel_dir = Path::new("signals")
            .join("derived")
            .join(format!("bundle_{}", tier.bundle_id));
        let rel = rel_dir.join(format!("{sanitized}.parquet"));
        let abs = self.root.join(&rel);
        if let Some(parent) = abs.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok((abs, rel.to_string_lossy().into_owned()))
    }

    fn guard_no_existing_derived_signal(&self, tier: &Tier) -> Result<()> {
        if self.derived_signal(tier.id)?.is_some() {
            return Err(EngineError::Corpus(format!(
                "tier {} already has a derived_signal sidecar; rewrites land in a follow-up slice",
                tier.id
            )));
        }
        Ok(())
    }

    fn insert_derived_signal_row(
        &self,
        tier: &Tier,
        relative_path: &str,
        n_frames: i64,
        n_dims: i64,
        sample_rate_hz: f64,
        dtype: &str,
    ) -> Result<i64> {
        let id: i64 = self.conn.query_row(
            "INSERT INTO derived_signal \
                (tier_id, relative_path, n_frames, n_dims, sample_rate_hz, dtype) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) RETURNING id",
            rusqlite::params![
                tier.id,
                relative_path,
                n_frames,
                n_dims,
                sample_rate_hz,
                dtype
            ],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    fn expect_derived_signal(&self, tier_id: i64, expected: TierType) -> Result<DerivedSignal> {
        let tier = self.get_tier(tier_id)?;
        if tier.r#type != expected {
            return Err(EngineError::Corpus(format!(
                "tier {} is type {:?}; expected {expected:?}",
                tier.id, tier.r#type
            )));
        }
        self.derived_signal(tier_id)?.ok_or_else(|| {
            EngineError::Corpus(format!(
                "no derived_signal sidecar for tier {tier_id}; call write_* first"
            ))
        })
    }

    fn enforce_cardinality(
        &self,
        child_tier: &Tier,
        child_table: &str,
        parent_annotation_id: Option<i64>,
    ) -> Result<()> {
        let parent_tier_id = match child_tier.parent_id {
            None => {
                if parent_annotation_id.is_some() {
                    return Err(EngineError::Cardinality(format!(
                        "tier {} has no parent_tier but parent_annotation_id was provided",
                        child_tier.id
                    )));
                }
                return Ok(());
            }
            Some(p) => p,
        };
        // `cardinality = "none"` (or `None`) means tier-level hierarchy
        // is recorded but no annotation-level link is required. Importers
        // that recover only tier hierarchy (e.g. EAF) rely on this.
        let parent_annotation_id = match parent_annotation_id {
            Some(id) => id,
            None => {
                return match child_tier.cardinality.as_deref() {
                    Some("none") | None => Ok(()),
                    _ => Err(EngineError::Cardinality(format!(
                        "tier {} has parent tier {}; parent_annotation_id is required",
                        child_tier.id, parent_tier_id
                    ))),
                };
            }
        };

        // Verify the parent annotation exists in the right table for the
        // parent tier's type.
        let parent_tier = self.get_tier(parent_tier_id)?;
        let parent_table = match parent_tier.r#type {
            TierType::Interval => "annotation_interval",
            TierType::Point => "annotation_point",
            TierType::Reference => "annotation_reference",
            _ => {
                return Err(EngineError::Cardinality(format!(
                    "parent tier {} is dense ({:?}); only sparse tiers can be parents in B2",
                    parent_tier.id, parent_tier.r#type
                )));
            }
        };
        let parent_exists: i64 = self.conn.query_row(
            &format!("SELECT COUNT(*) FROM {parent_table} WHERE id = ?1 AND tier_id = ?2"),
            rusqlite::params![parent_annotation_id, parent_tier_id],
            |row| row.get(0),
        )?;
        if parent_exists == 0 {
            return Err(EngineError::Cardinality(format!(
                "parent annotation {parent_annotation_id} not found in tier {parent_tier_id} \
                 (table {parent_table})"
            )));
        }

        match child_tier.cardinality.as_deref() {
            Some("one_to_one") => {
                let already: i64 = self.conn.query_row(
                    &format!(
                        "SELECT COUNT(*) FROM {child_table} \
                         WHERE tier_id = ?1 AND parent_annotation_id = ?2"
                    ),
                    rusqlite::params![child_tier.id, parent_annotation_id],
                    |row| row.get(0),
                )?;
                if already > 0 {
                    return Err(EngineError::Cardinality(format!(
                        "one_to_one violation: tier {} already has a child for parent \
                         annotation {parent_annotation_id}",
                        child_tier.id
                    )));
                }
                Ok(())
            }
            Some("one_to_many") | Some("none") | None => Ok(()),
            Some("many_to_one") => Err(EngineError::Cardinality(
                "many_to_one cardinality is not supported until B-cluster follow-up".into(),
            )),
            Some(other) => Err(EngineError::Cardinality(format!(
                "unknown cardinality {other:?}"
            ))),
        }
    }

    /// Imports a Praat TextGrid into `bundle_id`. Each Praat tier becomes a
    /// new [`Tier`] row (interval or point); each annotation becomes an
    /// `annotation_interval` / `annotation_point` row with any JSON
    /// sentinel decoded back into the `extra` field. Returns the new tier
    /// IDs in import order.
    ///
    /// Records a [`ProcessingRun`]-style row in `processing_run` for audit
    /// provenance (`processor_id = "sadda.io.textgrid.import"`).
    ///
    /// Lossiness: tier hierarchy, parent_annotation_id links, and
    /// tier-level `schema` JSON are not in TextGrid and aren't recovered.
    /// Reference tiers exported via the round-trip mechanism come back as
    /// interval tiers (recovering them as reference tiers is a future
    /// enhancement).
    pub fn import_textgrid(&self, path: impl AsRef<Path>, bundle_id: i64) -> Result<Vec<i64>> {
        let path = path.as_ref();
        let textgrid = crate::io::textgrid::read(path)?;
        let mut new_tier_ids = Vec::with_capacity(textgrid.tiers.len());
        for tier in &textgrid.tiers {
            match tier {
                crate::io::textgrid::TextGridTier::Interval(it) => {
                    let tier_id =
                        self.add_tier(&TierSpec::new(bundle_id, &it.name, TierType::Interval))?;
                    for entry in &it.intervals {
                        let (plain, extra) = crate::io::textgrid::decode_label(&entry.text);
                        self.add_interval(&IntervalSpec {
                            tier_id,
                            start_seconds: entry.xmin,
                            end_seconds: entry.xmax,
                            label: Some(plain),
                            extra,
                            ..Default::default()
                        })?;
                    }
                    new_tier_ids.push(tier_id);
                }
                crate::io::textgrid::TextGridTier::Point(pt) => {
                    let tier_id =
                        self.add_tier(&TierSpec::new(bundle_id, &pt.name, TierType::Point))?;
                    for entry in &pt.points {
                        let (plain, extra) = crate::io::textgrid::decode_label(&entry.mark);
                        self.add_point(&PointSpec {
                            tier_id,
                            time_seconds: entry.time,
                            label: Some(plain),
                            extra,
                            ..Default::default()
                        })?;
                    }
                    new_tier_ids.push(tier_id);
                }
            }
        }
        // Record the import as a processing_run for audit provenance.
        let display = path.display().to_string();
        let params = format!(
            "{{\"path\":{:?},\"n_tiers\":{}}}",
            display,
            textgrid.tiers.len()
        );
        let mut spec = ProcessingRunSpec::new(
            bundle_id,
            ProcessingRunKind::DspAlgorithm,
            "sadda.io.textgrid.import",
        );
        spec.parameters = Some(params);
        spec.output_tier_ids = new_tier_ids.clone();
        self.record_processing_run(&spec)?;
        Ok(new_tier_ids)
    }

    /// Writes a Praat TextGrid for `bundle_id`'s sparse tiers to `path`.
    /// If `tier_ids` is `Some`, only those tiers are exported (filtered to
    /// the named bundle).
    ///
    /// Dense tiers (continuous_numeric / vector / categorical_sampled) are
    /// silently skipped. Reference tiers are exported as IntervalTiers with
    /// a degenerate `[0.0, 0.001]` time span and the JSON sentinel carrying
    /// the original `(target_kind, target_id)` payload, so they round-trip
    /// losslessly through Praat at the file level (re-import recovers the
    /// data as interval annotations; reconstituting them as reference
    /// annotations is a future enhancement).
    ///
    /// IntervalTiers must be contiguous in Praat; gaps are padded with
    /// empty `text = ""` intervals. Overlapping intervals are an error.
    pub fn export_textgrid(
        &self,
        bundle_id: i64,
        path: impl AsRef<Path>,
        tier_ids: Option<&[i64]>,
    ) -> Result<()> {
        let path = path.as_ref();
        let bundle = self.bundle(bundle_id)?;
        let audio_duration_seconds = bundle.n_frames as f64 / bundle.sample_rate as f64;

        let all_tiers = self.tiers(Some(bundle_id))?;
        let selected: Vec<&Tier> = match tier_ids {
            Some(ids) => all_tiers.iter().filter(|t| ids.contains(&t.id)).collect(),
            None => all_tiers.iter().collect(),
        };

        let mut tg_tiers: Vec<crate::io::textgrid::TextGridTier> = Vec::new();
        let mut file_xmax = audio_duration_seconds;

        for tier in selected {
            match tier.r#type {
                TierType::Interval => {
                    let rows = self.intervals(tier.id)?;
                    let entries = build_interval_entries(&rows, audio_duration_seconds)?;
                    let tier_xmax = entries
                        .last()
                        .map(|e| e.xmax)
                        .unwrap_or(audio_duration_seconds);
                    file_xmax = file_xmax.max(tier_xmax);
                    tg_tiers.push(crate::io::textgrid::TextGridTier::Interval(
                        crate::io::textgrid::IntervalTier {
                            name: tier.name.clone(),
                            xmin: 0.0,
                            xmax: tier_xmax,
                            intervals: entries,
                        },
                    ));
                }
                TierType::Point => {
                    let rows = self.points(tier.id)?;
                    let points: Vec<crate::io::textgrid::PointEntry> = rows
                        .iter()
                        .map(|p| crate::io::textgrid::PointEntry {
                            time: p.time_seconds,
                            mark: crate::io::textgrid::encode_label(
                                p.label.as_deref().unwrap_or(""),
                                p.extra.as_deref(),
                            ),
                        })
                        .collect();
                    let tier_xmax = points
                        .iter()
                        .map(|p| p.time)
                        .fold(audio_duration_seconds, f64::max);
                    file_xmax = file_xmax.max(tier_xmax);
                    tg_tiers.push(crate::io::textgrid::TextGridTier::Point(
                        crate::io::textgrid::PointTier {
                            name: tier.name.clone(),
                            xmin: 0.0,
                            xmax: tier_xmax,
                            points,
                        },
                    ));
                }
                TierType::Reference => {
                    let rows = self.references_for(tier.id)?;
                    let entries: Vec<crate::io::textgrid::IntervalEntry> = rows
                        .iter()
                        .enumerate()
                        .map(|(i, r)| {
                            let start = i as f64 * 0.001;
                            let end = start + 0.001;
                            let payload = format!(
                                "{{\"target_kind\":{:?},\"target_id\":{}{}}}",
                                r.target_kind,
                                r.target_id,
                                r.extra
                                    .as_deref()
                                    .map(|e| format!(",\"extra\":{e}"))
                                    .unwrap_or_default(),
                            );
                            let label_text = crate::io::textgrid::encode_label(
                                r.label.as_deref().unwrap_or(""),
                                Some(&payload),
                            );
                            crate::io::textgrid::IntervalEntry {
                                xmin: start,
                                xmax: end,
                                text: label_text,
                            }
                        })
                        .collect();
                    let tier_xmax = entries.last().map(|e| e.xmax).unwrap_or(0.001);
                    file_xmax = file_xmax.max(tier_xmax);
                    tg_tiers.push(crate::io::textgrid::TextGridTier::Interval(
                        crate::io::textgrid::IntervalTier {
                            name: tier.name.clone(),
                            xmin: 0.0,
                            xmax: tier_xmax,
                            intervals: entries,
                        },
                    ));
                }
                TierType::ContinuousNumeric
                | TierType::ContinuousVector
                | TierType::CategoricalSampled => {
                    // Skip dense tiers silently. A future API may return a
                    // report of how many were skipped.
                }
            }
        }

        // Normalise every tier's xmax up to the file_xmax (Praat requires
        // tier ranges to match the file range for IntervalTiers).
        for tier in tg_tiers.iter_mut() {
            match tier {
                crate::io::textgrid::TextGridTier::Interval(it) => {
                    if let Some(last) = it.intervals.last() {
                        if last.xmax < file_xmax {
                            it.intervals.push(crate::io::textgrid::IntervalEntry {
                                xmin: last.xmax,
                                xmax: file_xmax,
                                text: String::new(),
                            });
                        }
                    } else {
                        it.intervals.push(crate::io::textgrid::IntervalEntry {
                            xmin: 0.0,
                            xmax: file_xmax,
                            text: String::new(),
                        });
                    }
                    it.xmax = file_xmax;
                }
                crate::io::textgrid::TextGridTier::Point(pt) => {
                    pt.xmax = file_xmax;
                }
            }
        }

        let file = crate::io::textgrid::TextGridFile {
            xmin: 0.0,
            xmax: file_xmax,
            tiers: tg_tiers,
        };
        crate::io::textgrid::write(&file, path)?;
        Ok(())
    }

    /// Imports an ELAN `.eaf` into `bundle_id`. Each EAF tier becomes a
    /// new [`Tier`] (interval / point / reference based on stereotype and
    /// annotation shape); each annotation becomes an `annotation_*` row
    /// with any JSON sentinel decoded back into the `extra` field.
    ///
    /// Tier hierarchy survives the round-trip via EAF's `PARENT_REF`
    /// (resolved to `tier.parent_id`). Parent tiers are imported before
    /// their children regardless of the order they appear in the file.
    ///
    /// Annotation-to-annotation parentage (`ANNOTATION_REF` on
    /// `REF_ANNOTATION` / `parent_annotation_id` on alignable annotations
    /// of a child tier) is not reconstructed in v1 — recovering it
    /// requires a second pass that resolves EAF annotation IDs to our
    /// freshly-minted row IDs, which the schema's `parent_annotation_id`
    /// constraint can't enforce inline.
    ///
    /// Point-tier recovery heuristic: a tier whose every annotation has
    /// `end_ms - start_ms <= 2` is recovered as a `point` tier.
    ///
    /// Records a `processing_run` row with
    /// `processor_id = "sadda.io.eaf.import"` for audit provenance.
    pub fn import_eaf(&self, path: impl AsRef<Path>, bundle_id: i64) -> Result<Vec<i64>> {
        let path = path.as_ref();
        let eaf = crate::io::eaf::read(path)?;

        // Topo-sort tiers so parents come before children. EAF allows any
        // order; SQLite's `tier.parent_id` FK requires the parent row to
        // exist already.
        let order = topo_sort_eaf_tiers(&eaf.tiers)?;

        // tier_id_str → row id (for resolving PARENT_REF when inserting
        // child tiers).
        let mut id_map: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        let mut new_tier_ids: Vec<i64> = Vec::with_capacity(eaf.tiers.len());

        for idx in order {
            let tier = &eaf.tiers[idx];
            let parent_row_id = match &tier.parent_ref {
                Some(pref) => Some(*id_map.get(pref).ok_or_else(|| {
                    EngineError::Corpus(format!(
                        "EAF import: tier {:?} references unknown PARENT_REF {:?}",
                        tier.tier_id, pref,
                    ))
                })?),
                None => None,
            };

            let tier_type = classify_eaf_tier(tier);
            let mut spec = TierSpec::new(bundle_id, &tier.tier_id, tier_type);
            spec.parent_id = parent_row_id;
            if parent_row_id.is_some() {
                // Annotation-level parentage isn't reconstructed in v1 (see
                // the doc comment on `import_eaf`), so child annotations
                // are inserted without `parent_annotation_id`. The tier
                // cardinality must allow that — `"none"` does.
                spec.cardinality = Some("none".into());
            }
            let row_id = self.add_tier(&spec)?;
            id_map.insert(tier.tier_id.clone(), row_id);
            new_tier_ids.push(row_id);

            match tier_type {
                TierType::Point => {
                    for ann in &tier.annotations {
                        if let crate::io::eaf::EafAnnotation::Alignable {
                            start_ms, value, ..
                        } = ann
                        {
                            let (plain, extra) = crate::io::eaf::decode_label(value);
                            self.add_point(&PointSpec {
                                tier_id: row_id,
                                time_seconds: *start_ms as f64 / 1000.0,
                                label: Some(plain),
                                extra,
                                ..Default::default()
                            })?;
                        }
                    }
                }
                TierType::Reference => {
                    for ann in &tier.annotations {
                        // REF_ANNOTATION values may carry the sentinel
                        // payload that encodes (target_kind, target_id);
                        // alignable values shouldn't appear under a
                        // reference tier but we handle them gracefully.
                        let value = ann.value();
                        let (plain, extra) = crate::io::eaf::decode_label(value);
                        let (target_kind, target_id, extra_rest) =
                            parse_reference_sentinel(extra.as_deref());
                        self.add_reference(&ReferenceSpec {
                            tier_id: row_id,
                            target_kind,
                            target_id,
                            label: Some(plain),
                            extra: extra_rest,
                            ..Default::default()
                        })?;
                    }
                }
                _ => {
                    // TierType::Interval — the default for everything else
                    // produced by `classify_eaf_tier`.
                    for ann in &tier.annotations {
                        if let crate::io::eaf::EafAnnotation::Alignable {
                            start_ms,
                            end_ms,
                            value,
                            ..
                        } = ann
                        {
                            let (plain, extra) = crate::io::eaf::decode_label(value);
                            self.add_interval(&IntervalSpec {
                                tier_id: row_id,
                                start_seconds: *start_ms as f64 / 1000.0,
                                end_seconds: *end_ms as f64 / 1000.0,
                                label: Some(plain),
                                extra,
                                ..Default::default()
                            })?;
                        }
                        // REF_ANNOTATION under an interval tier: skipped
                        // in v1 (the data model can't represent symbolic
                        // children of an interval without a typed child
                        // tier of its own).
                    }
                }
            }
        }

        let display = path.display().to_string();
        let params = format!("{{\"path\":{:?},\"n_tiers\":{}}}", display, eaf.tiers.len());
        let mut spec = ProcessingRunSpec::new(
            bundle_id,
            ProcessingRunKind::DspAlgorithm,
            "sadda.io.eaf.import",
        );
        spec.parameters = Some(params);
        spec.output_tier_ids = new_tier_ids.clone();
        self.record_processing_run(&spec)?;
        Ok(new_tier_ids)
    }

    /// Returns the currently-active F1 recipe run id, if any.
    ///
    /// Set by [`Self::start_recipe`] and cleared by [`Self::end_recipe`].
    /// Used by the existing `processing_run` writers (`import_textgrid`,
    /// `import_eaf`, `commit_recording`) to populate
    /// `processing_run.recipe_run_id` whenever they INSERT.
    pub fn current_recipe_id(&self) -> Option<i64> {
        self.recipe_run_id.get()
    }

    /// Creates a new `recipe_run` row with `name` and `parameters` (an
    /// opaque JSON string the caller supplies) and sets it as this
    /// project's active recipe. Returns the new recipe id.
    ///
    /// Refuses to start a second recipe while another is already
    /// active (nesting is out of scope for v1). The `name` must be
    /// unique within the project — UNIQUE constraint on `recipe_run.name`.
    pub fn start_recipe(&self, name: &str, parameters_json: Option<&str>) -> Result<i64> {
        if self.recipe_run_id.get().is_some() {
            return Err(EngineError::Corpus(
                "start_recipe: another recipe is already active for this project".into(),
            ));
        }
        let id: i64 = self.conn.query_row(
            "INSERT INTO recipe_run (name, sadda_version, parameters, status) \
             VALUES (?1, ?2, ?3, 'in_progress') RETURNING id",
            rusqlite::params![name, crate::version(), parameters_json],
            |row| row.get(0),
        )?;
        self.recipe_run_id.set(Some(id));
        Ok(id)
    }

    /// Closes the active recipe: marks the row `'ok'` or `'error'` and
    /// stamps `completed_at`. Clears the project's active recipe id.
    /// Returns silently if the id doesn't match the active one
    /// (defensive: a panicking `__exit__` may double-call this).
    pub fn end_recipe(
        &self,
        recipe_run_id: i64,
        status: &str,
        error_message: Option<&str>,
    ) -> Result<()> {
        // Defensive: only clear the cell if we own the active recipe.
        if self.recipe_run_id.get() == Some(recipe_run_id) {
            self.recipe_run_id.set(None);
        }
        self.conn.execute(
            "UPDATE recipe_run \
                SET status = ?2, \
                    completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), \
                    error_message = ?3 \
              WHERE id = ?1",
            rusqlite::params![recipe_run_id, status, error_message],
        )?;
        Ok(())
    }

    /// Lists recipes in id order. Each row carries the metadata needed
    /// by the script generator (name, parameters, status).
    pub fn recipes(&self) -> Result<Vec<RecipeRun>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, sadda_version, parameters, started_at, completed_at, \
                    status, error_message \
             FROM recipe_run ORDER BY id",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(RecipeRun {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    sadda_version: row.get(2)?,
                    parameters: row.get(3)?,
                    started_at: row.get(4)?,
                    completed_at: row.get(5)?,
                    status: row.get(6)?,
                    error_message: row.get(7)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Fetches a single recipe by name. Errors if not found.
    pub fn recipe_by_name(&self, name: &str) -> Result<RecipeRun> {
        let r = self.conn.query_row(
            "SELECT id, name, sadda_version, parameters, started_at, completed_at, \
                    status, error_message \
             FROM recipe_run WHERE name = ?1",
            [name],
            |row| {
                Ok(RecipeRun {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    sadda_version: row.get(2)?,
                    parameters: row.get(3)?,
                    started_at: row.get(4)?,
                    completed_at: row.get(5)?,
                    status: row.get(6)?,
                    error_message: row.get(7)?,
                })
            },
        )?;
        Ok(r)
    }

    /// Records a completed [`processing_run`](ProcessingRunRow) for
    /// audit provenance and returns its id. Fills in the sadda version,
    /// the start/finish timestamps, and the active recipe id (if a
    /// recipe is running); the caller supplies the [`ProcessingRunSpec`].
    ///
    /// This is the single insert path every analysis that produces a
    /// tier or signal should go through, so a bundle's provenance
    /// timeline ([`Self::processing_runs`]) and citation export
    /// ([`Self::citations`]) stay complete.
    pub fn record_processing_run(&self, spec: &ProcessingRunSpec) -> Result<i64> {
        // Empty id lists serialize to NULL, not "[]", so an unfiltered
        // query distinguishes "no outputs" from "[]".
        fn ids_json(ids: &[i64]) -> Option<String> {
            if ids.is_empty() {
                return None;
            }
            Some(format!(
                "[{}]",
                ids.iter().map(i64::to_string).collect::<Vec<_>>().join(",")
            ))
        }
        let id: i64 = self.conn.query_row(
            "INSERT INTO processing_run \
                (bundle_id, kind, processor_id, processor_version, weights_checksum, \
                 parameters, input_tier_ids, output_tier_ids, output_signal_ids, \
                 finished_at, status, error_message, recipe_run_id) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, \
                     strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?10, ?11, ?12) RETURNING id",
            rusqlite::params![
                spec.bundle_id,
                spec.kind.as_str(),
                spec.processor_id,
                crate::version(),
                spec.weights_checksum,
                spec.parameters,
                ids_json(&spec.input_tier_ids),
                ids_json(&spec.output_tier_ids),
                ids_json(&spec.output_signal_ids),
                spec.status.as_str(),
                spec.error_message,
                self.recipe_run_id.get(),
            ],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Returns a bundle's `processing_run` rows in insertion order — the
    /// provenance timeline answering "where did every tier / signal on
    /// this bundle come from?"
    pub fn processing_runs(&self, bundle_id: i64) -> Result<Vec<ProcessingRunRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, bundle_id, kind, processor_id, processor_version, \
                    parameters, output_tier_ids, started_at, finished_at, status \
             FROM processing_run WHERE bundle_id = ?1 ORDER BY id",
        )?;
        let rows = stmt
            .query_map([bundle_id], |row| {
                Ok(ProcessingRunRow {
                    id: row.get(0)?,
                    bundle_id: row.get(1)?,
                    kind: row.get(2)?,
                    processor_id: row.get(3)?,
                    processor_version: row.get(4)?,
                    parameters: row.get(5)?,
                    output_tier_ids: row.get(6)?,
                    started_at: row.get(7)?,
                    finished_at: row.get(8)?,
                    status: row.get(9)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Reads a single `processing_run` by id, if present. Drives provenance
    /// display — e.g. resolving an annotation's `processing_run_id` to the
    /// criterion run that produced it.
    pub fn get_processing_run(&self, id: i64) -> Result<Option<ProcessingRunRow>> {
        self.conn
            .query_row(
                "SELECT id, bundle_id, kind, processor_id, processor_version, \
                        parameters, output_tier_ids, started_at, finished_at, status \
                 FROM processing_run WHERE id = ?1",
                [id],
                |row| {
                    Ok(ProcessingRunRow {
                        id: row.get(0)?,
                        bundle_id: row.get(1)?,
                        kind: row.get(2)?,
                        processor_id: row.get(3)?,
                        processor_version: row.get(4)?,
                        parameters: row.get(5)?,
                        output_tier_ids: row.get(6)?,
                        started_at: row.get(7)?,
                        finished_at: row.get(8)?,
                        status: row.get(9)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    /// Returns the literature citations for every cited processor that
    /// ran on `bundle_id`, deduplicated by `processor_id` and ordered by
    /// first use. Processors with no academic source (imports, live
    /// recording, …) are omitted. Drives "export the references for the
    /// analyses in this project."
    pub fn citations(&self, bundle_id: i64) -> Result<Vec<crate::citation::Citation>> {
        let mut seen = std::collections::HashSet::new();
        let mut out = Vec::new();
        for run in self.processing_runs(bundle_id)? {
            if !seen.insert(run.processor_id.clone()) {
                continue;
            }
            if let Some(c) = crate::citation::citation_for(&run.processor_id) {
                out.push(c);
            }
        }
        Ok(out)
    }

    /// Returns the `processing_run` rows linked to a recipe, ordered
    /// by `id` (i.e., insertion order). The script generator walks
    /// this list to emit the captured calls.
    pub fn processing_runs_for_recipe(&self, recipe_run_id: i64) -> Result<Vec<ProcessingRunRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, bundle_id, kind, processor_id, processor_version, \
                    parameters, output_tier_ids, started_at, finished_at, status \
             FROM processing_run WHERE recipe_run_id = ?1 ORDER BY id",
        )?;
        let rows = stmt
            .query_map([recipe_run_id], |row| {
                Ok(ProcessingRunRow {
                    id: row.get(0)?,
                    bundle_id: row.get(1)?,
                    kind: row.get(2)?,
                    processor_id: row.get(3)?,
                    processor_version: row.get(4)?,
                    parameters: row.get(5)?,
                    output_tier_ids: row.get(6)?,
                    started_at: row.get(7)?,
                    finished_at: row.get(8)?,
                    status: row.get(9)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Atomically commits a stopped [`crate::live::StoppedSession`] into the
    /// project: renames `.in_progress/<uuid>/audio.wav` into
    /// `signals/original/<sanitized_name>.wav`, inserts a `bundle` row, and
    /// records a `processing_run` row of kind `'live_recording'` with the
    /// capture parameters. Returns the new bundle id.
    ///
    /// `params_json` is recorded verbatim in `processing_run.parameters`.
    /// Recommended shape: `{"device": "...", "sample_rate": 44100,
    /// "channels": 1, "duration_s": 4.2, "analysis_window_ms": 30,
    /// "analysis_hop_ms": 10, "dropped_samples": 0}` — the PyO3 wrapper
    /// builds this from `LiveConfig` + `StoppedSession` for free.
    ///
    /// On any failure between the rename and the SQL inserts, the file is
    /// renamed back into the `.in_progress/` directory if possible; if the
    /// rename-back fails too, the error reports both paths so the operator
    /// can recover manually.
    pub fn commit_recording(
        &self,
        stopped: crate::live::StoppedSession,
        name: &str,
        params_json: &str,
    ) -> Result<i64> {
        if stopped.frames_written == 0 {
            return Err(EngineError::Corpus(
                "commit_recording: no audio captured (frames_written = 0)".into(),
            ));
        }
        let src_wav = stopped.in_progress_dir.join("audio.wav");
        if !src_wav.exists() {
            return Err(EngineError::Corpus(format!(
                "commit_recording: source WAV not found: {}",
                src_wav.display()
            )));
        }
        // Validate by reading; this also gives us sample_rate / channels /
        // n_frames straight from the file rather than trusting the caller.
        let audio = Audio::from_wav_path(&src_wav)?;

        let sanitized = sanitize_filename(name);
        let dest_rel = Path::new("signals")
            .join("original")
            .join(format!("{sanitized}.wav"));
        let dest_abs = self.root.join(&dest_rel);
        if dest_abs.exists() {
            return Err(EngineError::Corpus(format!(
                "commit_recording: destination already exists: {}",
                dest_abs.display()
            )));
        }

        // Atomic move first; on the same filesystem this is a single
        // rename(2) and either fully succeeds or leaves the source intact.
        std::fs::rename(&src_wav, &dest_abs)?;

        // Insert bundle + processing_run. If either fails, rename back.
        let result = (|| -> Result<i64> {
            let bundle_id: i64 = self.conn.query_row(
                "INSERT INTO bundle \
                    (name, audio_relative_path, sample_rate, channels, n_frames, \
                     session_id, speaker_id, extra) \
                 VALUES (?1, ?2, ?3, ?4, ?5, NULL, NULL, NULL) RETURNING id",
                rusqlite::params![
                    name,
                    dest_rel.to_string_lossy().as_ref(),
                    audio.sample_rate as i64,
                    audio.channels as i64,
                    audio.frame_count() as i64,
                ],
                |row| row.get(0),
            )?;
            self.conn.execute(
                "INSERT INTO processing_run \
                    (bundle_id, kind, processor_id, processor_version, \
                     parameters, finished_at, status, recipe_run_id) \
                 VALUES (?1, 'live_recording', 'sadda.live', ?2, ?3, \
                         strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 'ok', ?4)",
                rusqlite::params![
                    bundle_id,
                    crate::version(),
                    params_json,
                    self.recipe_run_id.get(),
                ],
            )?;
            Ok(bundle_id)
        })();

        match result {
            Ok(bundle_id) => {
                // Best-effort cleanup of the (now-empty) .in_progress dir.
                let _ = std::fs::remove_dir_all(&stopped.in_progress_dir);
                Ok(bundle_id)
            }
            Err(e) => {
                // Roll back the rename so the user can retry / inspect.
                if let Err(rename_err) = std::fs::rename(&dest_abs, &src_wav) {
                    return Err(EngineError::Corpus(format!(
                        "commit_recording failed: {e}; \
                         additionally, rolling back the rename failed: {rename_err}. \
                         Recording is at {} (db not updated)",
                        dest_abs.display()
                    )));
                }
                Err(e)
            }
        }
    }

    /// Writes an ELAN `.eaf` file for `bundle_id`'s sparse tiers to `path`
    /// in EAF 2.8 format. If `tier_ids` is `Some`, only those tiers are
    /// exported (filtered to the named bundle).
    ///
    /// - Interval tiers → `ALIGNABLE_ANNOTATION`. Top-level tiers use the
    ///   `sadda_alignable` linguistic type (no stereotype); tiers with a
    ///   `parent_id` use `sadda_included` with the `Included_In`
    ///   stereotype.
    /// - Point tiers → `ALIGNABLE_ANNOTATION` with a degenerate
    ///   `[t, t + 1ms]` interval. Re-import recovers them as points via
    ///   the ≤2ms heuristic.
    /// - Reference tiers → `REF_ANNOTATION` under the `sadda_symbolic`
    ///   linguistic type (`Symbolic_Association`). The sentinel encodes
    ///   `(target_kind, target_id)` so re-import is lossless.
    /// - Dense tiers are silently skipped.
    ///
    /// Annotation IDs and PARENT_REF resolution: each exported annotation
    /// gets a fresh `aN` ID; reference annotations under a parent tier
    /// link to the *parent tier's first annotation* as a placeholder
    /// (annotation-level parentage isn't tracked in v1 — re-import
    /// recovers tier hierarchy but not annotation-level links).
    pub fn export_eaf(
        &self,
        bundle_id: i64,
        path: impl AsRef<Path>,
        tier_ids: Option<&[i64]>,
    ) -> Result<()> {
        let path = path.as_ref();
        let bundle = self.bundle(bundle_id)?;
        let _ = bundle; // currently used only to validate bundle_id

        let all_tiers = self.tiers(Some(bundle_id))?;
        let selected: Vec<&Tier> = match tier_ids {
            Some(ids) => all_tiers.iter().filter(|t| ids.contains(&t.id)).collect(),
            None => all_tiers.iter().collect(),
        };

        // tier_row_id → exported tier_id_str (for PARENT_REF resolution).
        let mut tier_name_by_id: std::collections::HashMap<i64, String> =
            std::collections::HashMap::new();
        for t in &selected {
            tier_name_by_id.insert(t.id, t.name.clone());
        }

        let mut next_ann_id: u64 = 1;
        let mut mint_id = || {
            let s = format!("a{next_ann_id}");
            next_ann_id += 1;
            s
        };
        // For REF_ANNOTATION export we need a "parent annotation ID" to
        // point at. We track the first annotation ID minted per tier and
        // reuse it as the placeholder target for all symbolic children.
        let mut first_ann_id_by_tier: std::collections::HashMap<i64, String> =
            std::collections::HashMap::new();

        let mut eaf_tiers: Vec<crate::io::eaf::EafTier> = Vec::new();

        for tier in &selected {
            let parent_ref = tier
                .parent_id
                .and_then(|pid| tier_name_by_id.get(&pid).cloned());

            match tier.r#type {
                TierType::Interval => {
                    let rows = self.intervals(tier.id)?;
                    let mut annotations = Vec::with_capacity(rows.len());
                    for iv in &rows {
                        let id = mint_id();
                        first_ann_id_by_tier
                            .entry(tier.id)
                            .or_insert_with(|| id.clone());
                        annotations.push(crate::io::eaf::EafAnnotation::Alignable {
                            id,
                            start_ms: (iv.start_seconds * 1000.0).round() as i64,
                            end_ms: (iv.end_seconds * 1000.0).round() as i64,
                            value: crate::io::eaf::encode_label(
                                iv.label.as_deref().unwrap_or(""),
                                iv.extra.as_deref(),
                            ),
                        });
                    }
                    let (lt, stereotype) = if parent_ref.is_some() {
                        (
                            crate::io::eaf::LT_INCLUDED.to_string(),
                            Some("Included_In".to_string()),
                        )
                    } else {
                        (crate::io::eaf::LT_ALIGNABLE.to_string(), None)
                    };
                    eaf_tiers.push(crate::io::eaf::EafTier {
                        tier_id: tier.name.clone(),
                        linguistic_type_ref: lt,
                        stereotype,
                        parent_ref,
                        annotations,
                    });
                }
                TierType::Point => {
                    let rows = self.points(tier.id)?;
                    let mut annotations = Vec::with_capacity(rows.len());
                    for p in &rows {
                        let id = mint_id();
                        first_ann_id_by_tier
                            .entry(tier.id)
                            .or_insert_with(|| id.clone());
                        let t_ms = (p.time_seconds * 1000.0).round() as i64;
                        annotations.push(crate::io::eaf::EafAnnotation::Alignable {
                            id,
                            start_ms: t_ms,
                            end_ms: t_ms + 1,
                            value: crate::io::eaf::encode_label(
                                p.label.as_deref().unwrap_or(""),
                                p.extra.as_deref(),
                            ),
                        });
                    }
                    let (lt, stereotype) = if parent_ref.is_some() {
                        (
                            crate::io::eaf::LT_INCLUDED.to_string(),
                            Some("Included_In".to_string()),
                        )
                    } else {
                        (crate::io::eaf::LT_ALIGNABLE.to_string(), None)
                    };
                    eaf_tiers.push(crate::io::eaf::EafTier {
                        tier_id: tier.name.clone(),
                        linguistic_type_ref: lt,
                        stereotype,
                        parent_ref,
                        annotations,
                    });
                }
                TierType::Reference => {
                    let rows = self.references_for(tier.id)?;
                    let placeholder_parent = tier
                        .parent_id
                        .and_then(|pid| first_ann_id_by_tier.get(&pid).cloned());
                    let mut annotations = Vec::with_capacity(rows.len());
                    for r in &rows {
                        let id = mint_id();
                        let payload = format!(
                            "{{\"target_kind\":{:?},\"target_id\":{}{}}}",
                            r.target_kind,
                            r.target_id,
                            r.extra
                                .as_deref()
                                .map(|e| format!(",\"extra\":{e}"))
                                .unwrap_or_default(),
                        );
                        let value = crate::io::eaf::encode_label(
                            r.label.as_deref().unwrap_or(""),
                            Some(&payload),
                        );
                        let ann_ref = placeholder_parent.clone().unwrap_or_else(|| "".into());
                        annotations.push(crate::io::eaf::EafAnnotation::Ref {
                            id,
                            annotation_ref: ann_ref,
                            value,
                        });
                    }
                    eaf_tiers.push(crate::io::eaf::EafTier {
                        tier_id: tier.name.clone(),
                        linguistic_type_ref: crate::io::eaf::LT_SYMBOLIC.to_string(),
                        stereotype: Some("Symbolic_Association".into()),
                        parent_ref,
                        annotations,
                    });
                }
                TierType::ContinuousNumeric
                | TierType::ContinuousVector
                | TierType::CategoricalSampled => {
                    // Skip dense tiers silently (no EAF analogue).
                }
            }
        }

        let file = crate::io::eaf::EafFile {
            format: "2.8".to_string(),
            media_url: Some(format!("file:{}", bundle.audio_relative_path)),
            tiers: eaf_tiers,
        };
        crate::io::eaf::write(&file, path)?;
        Ok(())
    }

    /// Internal: fetches a single bundle row.
    fn bundle(&self, bundle_id: i64) -> Result<Bundle> {
        let bundle = self.conn.query_row(
            "SELECT id, name, audio_relative_path, sample_rate, channels, n_frames, \
                    session_id, speaker_id, extra \
             FROM bundle WHERE id = ?1",
            [bundle_id],
            |row| {
                Ok(Bundle {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    audio_relative_path: row.get(2)?,
                    sample_rate: row.get::<_, i64>(3)? as u32,
                    channels: row.get::<_, i64>(4)? as u16,
                    n_frames: row.get::<_, i64>(5)? as usize,
                    session_id: row.get(6)?,
                    speaker_id: row.get(7)?,
                    extra: row.get(8)?,
                })
            },
        )?;
        Ok(bundle)
    }
}

/// Convert a list of `Interval` rows into contiguous Praat-style entries.
/// Sorts by start time, pads gaps with empty `text = ""` intervals, and
/// errors on overlap. Always starts at `0.0`; the caller pads the trailing
/// gap up to the file_xmax separately.
fn build_interval_entries(
    rows: &[Interval],
    bundle_duration_seconds: f64,
) -> Result<Vec<crate::io::textgrid::IntervalEntry>> {
    let mut sorted: Vec<&Interval> = rows.iter().collect();
    sorted.sort_by(|a, b| a.start_seconds.partial_cmp(&b.start_seconds).unwrap());

    let mut entries = Vec::new();
    let mut cursor = 0.0_f64;

    for iv in sorted {
        if iv.start_seconds < cursor - 1e-9 {
            return Err(EngineError::Corpus(format!(
                "TextGrid export: overlapping intervals at tier annotation id {}; \
                 interval [{}, {}] starts before previous ended at {}",
                iv.id, iv.start_seconds, iv.end_seconds, cursor
            )));
        }
        // Gap padding.
        if iv.start_seconds > cursor + 1e-9 {
            entries.push(crate::io::textgrid::IntervalEntry {
                xmin: cursor,
                xmax: iv.start_seconds,
                text: String::new(),
            });
        }
        let text = crate::io::textgrid::encode_label(
            iv.label.as_deref().unwrap_or(""),
            iv.extra.as_deref(),
        );
        entries.push(crate::io::textgrid::IntervalEntry {
            xmin: iv.start_seconds,
            xmax: iv.end_seconds,
            text,
        });
        cursor = iv.end_seconds;
    }

    // If the user's intervals don't reach the bundle duration we leave the
    // final padding to the caller (it does the file-level xmax adjustment).
    let _ = bundle_duration_seconds;

    Ok(entries)
}

/// EAF-import helper: classify a parsed EAF tier as one of our tier types.
/// Default is `Interval`. Heuristics:
///
/// - `Symbolic_Association` stereotype → `Reference`.
/// - Every alignable annotation has `end_ms - start_ms <= 2` (and the tier
///   has at least one annotation) → `Point`.
/// - Otherwise → `Interval`.
fn classify_eaf_tier(tier: &crate::io::eaf::EafTier) -> TierType {
    if matches!(tier.stereotype.as_deref(), Some("Symbolic_Association")) {
        return TierType::Reference;
    }
    let mut saw_alignable = false;
    let mut all_short = true;
    for ann in &tier.annotations {
        if let crate::io::eaf::EafAnnotation::Alignable {
            start_ms, end_ms, ..
        } = ann
        {
            saw_alignable = true;
            if end_ms - start_ms > 2 {
                all_short = false;
                break;
            }
        }
    }
    if saw_alignable && all_short {
        TierType::Point
    } else {
        TierType::Interval
    }
}

/// EAF-import helper: topological sort of tier indices so that any tier
/// with a `PARENT_REF` appears after its parent. Returns indices into
/// `tiers`. Errors on cycles or unresolved parent references.
fn topo_sort_eaf_tiers(tiers: &[crate::io::eaf::EafTier]) -> Result<Vec<usize>> {
    use std::collections::HashMap;
    let index_by_id: HashMap<&str, usize> = tiers
        .iter()
        .enumerate()
        .map(|(i, t)| (t.tier_id.as_str(), i))
        .collect();
    let mut order = Vec::with_capacity(tiers.len());
    let mut visited = vec![false; tiers.len()];
    let mut on_stack = vec![false; tiers.len()];

    fn visit(
        i: usize,
        tiers: &[crate::io::eaf::EafTier],
        index_by_id: &std::collections::HashMap<&str, usize>,
        visited: &mut [bool],
        on_stack: &mut [bool],
        order: &mut Vec<usize>,
    ) -> Result<()> {
        if visited[i] {
            return Ok(());
        }
        if on_stack[i] {
            return Err(EngineError::Corpus(format!(
                "EAF import: cycle in tier PARENT_REFs at tier {:?}",
                tiers[i].tier_id,
            )));
        }
        on_stack[i] = true;
        if let Some(pref) = &tiers[i].parent_ref {
            let parent_idx = *index_by_id.get(pref.as_str()).ok_or_else(|| {
                EngineError::Corpus(format!(
                    "EAF import: tier {:?} references unknown PARENT_REF {:?}",
                    tiers[i].tier_id, pref,
                ))
            })?;
            visit(parent_idx, tiers, index_by_id, visited, on_stack, order)?;
        }
        on_stack[i] = false;
        visited[i] = true;
        order.push(i);
        Ok(())
    }

    for i in 0..tiers.len() {
        visit(
            i,
            tiers,
            &index_by_id,
            &mut visited,
            &mut on_stack,
            &mut order,
        )?;
    }
    Ok(order)
}

/// EAF-import helper: extract `(target_kind, target_id, extra_json)` from
/// the JSON sentinel payload attached to a reference annotation. Falls
/// back to `("annotation", 0, None)` if the sentinel is absent or
/// malformed; the synthetic target lets the import succeed but signals
/// to the caller that re-importing externally-authored EAFs as reference
/// tiers is a best-effort affair.
fn parse_reference_sentinel(json: Option<&str>) -> (String, i64, Option<String>) {
    let Some(raw) = json else {
        return ("annotation".to_string(), 0, None);
    };
    // Minimal hand-rolled extractor avoiding a serde_json dependency on
    // this hot path. Looks for the literal substrings emitted by
    // `export_eaf` above.
    let target_kind =
        extract_json_string(raw, "\"target_kind\":").unwrap_or_else(|| "annotation".to_string());
    let target_id = extract_json_i64(raw, "\"target_id\":").unwrap_or(0);
    let extra_rest = extract_json_object(raw, "\"extra\":");
    (target_kind, target_id, extra_rest)
}

/// Finds `key` (e.g. `"target_kind":`) and returns the following
/// double-quoted string value, if present.
fn extract_json_string(raw: &str, key: &str) -> Option<String> {
    let i = raw.find(key)?;
    let rest = &raw[i + key.len()..];
    let rest = rest.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Finds `key` (e.g. `"target_id":`) and returns the following integer
/// value, if present.
fn extract_json_i64(raw: &str, key: &str) -> Option<i64> {
    let i = raw.find(key)?;
    let rest = &raw[i + key.len()..];
    let rest = rest.trim_start();
    let end = rest
        .find(|c: char| c != '-' && !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Finds `key` (e.g. `"extra":`) and returns the following raw JSON
/// object value (`{...}`) as a string, balancing nested braces. Returns
/// `None` if the key isn't present or the value isn't an object.
fn extract_json_object(raw: &str, key: &str) -> Option<String> {
    let i = raw.find(key)?;
    let rest = &raw[i + key.len()..];
    let rest = rest.trim_start();
    if !rest.starts_with('{') {
        return None;
    }
    let mut depth = 0i32;
    let bytes = rest.as_bytes();
    for (idx, &b) in bytes.iter().enumerate() {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(rest[..=idx].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Result of [`Project::tier_rows`]: a typed enum of the three sparse-tier
/// row shapes. The Python layer dispatches on the variant to build a
/// tier-type-specific `polars.DataFrame`.
#[derive(Debug, Clone)]
pub enum TierRows {
    /// Rows from an `interval` tier.
    Interval(Vec<Interval>),
    /// Rows from a `point` tier.
    Point(Vec<Point>),
    /// Rows from a `reference` tier.
    Reference(Vec<Reference>),
}

// ---------------------------------------------------------------------------
// Single-writer lock (F10)
// ---------------------------------------------------------------------------

/// Lockfile name inside the project root. Hidden via the leading
/// dot on UNIX; not hidden on Windows but invisible in most file
/// browsers there too.
const LOCKFILE_NAME: &str = ".sadda-lock";

/// Records the lock holder. TOML-serialised so a human inspecting
/// the file gets a readable explanation.
#[derive(Debug)]
struct LockInfo {
    pid: u32,
    hostname: String,
    acquired_at: String,
}

impl LockInfo {
    fn render(&self) -> String {
        format!(
            "pid = {}\nhostname = {:?}\nacquired_at = {:?}\n",
            self.pid, self.hostname, self.acquired_at,
        )
    }

    fn parse(raw: &str) -> Option<Self> {
        let mut pid: Option<u32> = None;
        let mut hostname: Option<String> = None;
        let mut acquired_at: Option<String> = None;
        for line in raw.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("pid =") {
                pid = rest.trim().parse().ok();
            } else if let Some(rest) = line.strip_prefix("hostname =") {
                hostname = Some(rest.trim().trim_matches('"').to_string());
            } else if let Some(rest) = line.strip_prefix("acquired_at =") {
                acquired_at = Some(rest.trim().trim_matches('"').to_string());
            }
        }
        Some(Self {
            pid: pid?,
            hostname: hostname?,
            acquired_at: acquired_at?,
        })
    }
}

/// Acquires the project's `.sadda-lock` file. Errors with
/// [`EngineError::ProjectLocked`] if a live process on this host
/// already holds it.
fn acquire_lock(root: &Path) -> Result<()> {
    let lockfile = root.join(LOCKFILE_NAME);
    let our_pid = std::process::id();
    let our_host = hostname();

    if lockfile.exists()
        && let Ok(raw) = std::fs::read_to_string(&lockfile)
        && let Some(existing) = LockInfo::parse(&raw)
    {
        // Same PID = our own process re-opening (the previous open
        // leaked); take over silently.
        if existing.pid == our_pid {
            // fall through to overwrite
        } else if existing.hostname == our_host && pid_is_live(existing.pid) {
            return Err(EngineError::ProjectLocked {
                holder_pid: existing.pid,
                hostname: existing.hostname,
                lockfile_path: lockfile,
            });
        } else {
            // Stale (dead PID, or different hostname we can't verify).
            eprintln!(
                "sadda: clearing stale lockfile at {} (was PID {} on {})",
                lockfile.display(),
                existing.pid,
                existing.hostname,
            );
        }
    }

    let info = LockInfo {
        pid: our_pid,
        hostname: our_host,
        acquired_at: now_iso8601(),
    };
    std::fs::write(&lockfile, info.render())?;
    Ok(())
}

/// Best-effort lockfile delete. Ignores I/O errors — a stale
/// file is detected and cleared by the next `acquire_lock`.
fn release_lock(root: &Path) {
    let _ = std::fs::remove_file(root.join(LOCKFILE_NAME));
}

/// Returns the running process's hostname, falling back to
/// `"unknown"` if the OS can't tell us.
fn hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("COMPUTERNAME").ok())
        .or_else(|| {
            // POSIX `uname -n` via libc. We use a simpler fallback:
            // read /proc/sys/kernel/hostname on Linux. On other
            // UNIXes both env vars usually work; on Windows
            // COMPUTERNAME is always set.
            std::fs::read_to_string("/proc/sys/kernel/hostname")
                .ok()
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Crude RFC-3339-ish formatter: seconds since the epoch.
    // The lockfile's audit value is informational; users who need
    // a real timestamp can `stat .sadda-lock`.
    format!("epoch+{secs}s")
}

/// True if a process with `pid` exists on this host. UNIX uses
/// `kill(pid, 0)`; Windows uses `OpenProcess` + `CloseHandle`.
#[cfg(unix)]
fn pid_is_live(pid: u32) -> bool {
    // 0 and negative-on-cast values are special to kill(2): 0 means
    // "every process in the caller's process group", -1 means
    // "every process the caller can signal." Neither is a real
    // liveness check; reject up front.
    if pid == 0 || pid > i32::MAX as u32 {
        return false;
    }
    // SAFETY: kill(pid, 0) is a probe; doesn't change process state.
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if rc == 0 {
        // Successfully would have signalled — process exists.
        return true;
    }
    // EPERM = process exists but we can't signal it (different
    // uid). ESRCH = no such process.
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(windows)]
fn pid_is_live(pid: u32) -> bool {
    // SAFETY: OpenProcess + CloseHandle round-trip; no state change.
    use std::ffi::c_void;
    unsafe extern "system" {
        fn OpenProcess(access: u32, inherit_handle: i32, process_id: u32) -> *mut c_void;
        fn CloseHandle(handle: *mut c_void) -> i32;
    }
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if handle.is_null() {
        return false;
    }
    unsafe {
        CloseHandle(handle);
    }
    true
}

#[cfg(not(any(unix, windows)))]
fn pid_is_live(_pid: u32) -> bool {
    // Unknown platform: trust the lockfile and treat the lock as
    // held. Forces the user to clear manually if they're sure.
    true
}

impl Drop for Project {
    fn drop(&mut self) {
        if self.holds_lock {
            release_lock(&self.root);
        }
    }
}

fn backup_corpus_db(conn: &Connection, db_path: &Path, from_version: i64) -> Result<()> {
    // Flush any WAL state into the main file so the copy is self-contained.
    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
    let backup_path = db_path.with_file_name(format!("corpus.db.bak.{from_version}"));
    std::fs::copy(db_path, &backup_path)?;
    Ok(())
}

/// Replaces any character that isn't ASCII alphanumeric / `_` / `.` / `-`
/// with `_`. Keeps a tier's name recognizable on disk while preventing path
/// traversal or filesystem-unfriendly characters. Tier-name uniqueness within
/// a bundle is enforced by V3's `UNIQUE (bundle_id, name)`; the V5
/// `derived_signal.tier_id UNIQUE` constraint additionally rules out
/// sidecar-path collisions even if two sanitized names somehow coincide.
fn sanitize_filename(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '.' || ch == '-' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push('_');
    }
    out
}

/// In-place Fisher–Yates shuffle driven by a splitmix64 stream seeded with
/// `seed`. Deterministic — the same `(slice contents, seed)` always yields the
/// same permutation — so seeded random assignment is reproducible without a
/// `rand` dependency. splitmix64 is the standard seed mixer (Steele et al.,
/// "Fast Splittable Pseudorandom Number Generators", OOPSLA 2014).
fn deterministic_shuffle<T>(items: &mut [T], seed: u64) {
    let mut state = seed;
    let mut next = || {
        // splitmix64
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    // Fisher–Yates: for i from len-1 down to 1, swap i with a uniform j in 0..=i.
    let len = items.len();
    for i in (1..len).rev() {
        // Lemire's unbiased bounded reduction into 0..=i.
        let bound = (i as u64) + 1;
        let j = ((next() as u128 * bound as u128) >> 64) as usize;
        items.swap(i, j);
    }
}

/// Orders tiers so each tier precedes its children (parent-first), so a copy
/// can remap `parent_id` through an already-populated id map. A tier with no
/// parent, or whose parent isn't in the set, is ready immediately; the rest
/// follow once their parent is placed. A dangling cycle (shouldn't happen) is
/// appended as-is rather than looping forever.
fn parent_first_order(tiers: Vec<Tier>) -> Vec<Tier> {
    let present: std::collections::HashSet<i64> = tiers.iter().map(|t| t.id).collect();
    let mut placed: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut out: Vec<Tier> = Vec::with_capacity(tiers.len());
    while out.len() < tiers.len() {
        let before = out.len();
        for t in &tiers {
            if placed.contains(&t.id) {
                continue;
            }
            let ready = match t.parent_id {
                None => true,
                Some(p) => placed.contains(&p) || !present.contains(&p),
            };
            if ready {
                out.push(t.clone());
                placed.insert(t.id);
            }
        }
        if out.len() == before {
            for t in &tiers {
                if placed.insert(t.id) {
                    out.push(t.clone());
                }
            }
        }
    }
    out
}

/// The `sadda_export.json` manifest written into an annotator package and read
/// back on import — identifies the package and whose work it carries.
#[derive(serde::Serialize, serde::Deserialize)]
struct ExportManifest {
    /// Package format tag, for forward compatibility.
    format: String,
    /// The annotator the package was built for.
    annotator: String,
    /// The source project's name (informational).
    source_project: String,
    /// The engine schema version the package was written at.
    schema_version: i64,
}

const EXPORT_MANIFEST_NAME: &str = "sadda_export.json";

fn write_export_manifest(dir: &Path, annotator: &str, source_project: &str) -> Result<()> {
    let m = ExportManifest {
        format: "sadda-annotator-package/1".into(),
        annotator: annotator.to_owned(),
        source_project: source_project.to_owned(),
        schema_version: crate::corpus::migrations::engine_max_version(),
    };
    let json = serde_json::to_string_pretty(&m)
        .map_err(|e| EngineError::Corpus(format!("manifest serialize: {e}")))?;
    std::fs::write(dir.join(EXPORT_MANIFEST_NAME), json)?;
    Ok(())
}

fn read_export_manifest(dir: &Path) -> Result<ExportManifest> {
    let path = dir.join(EXPORT_MANIFEST_NAME);
    let text = std::fs::read_to_string(&path).map_err(|e| {
        EngineError::Corpus(format!(
            "not a sadda annotator package (missing {EXPORT_MANIFEST_NAME}): {e}"
        ))
    })?;
    serde_json::from_str(&text).map_err(|e| EngineError::Corpus(format!("bad manifest: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound;

    fn unique_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "sadda_engine_corpus_test_{}_{}",
            std::process::id(),
            name
        ));
        p
    }

    fn write_short_wav(path: &Path, sample_rate: u32) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        for i in 0..sample_rate / 4 {
            let t = i as f32 / sample_rate as f32;
            let s = (0.5 * i16::MAX as f32 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()) as i16;
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
    }

    #[test]
    fn create_lays_down_directory_structure() {
        let root = unique_dir("create_dirs");
        let _ = std::fs::remove_dir_all(&root);

        let project = Project::create(&root, "test_project").unwrap();

        assert!(root.join("project.toml").exists());
        assert!(root.join("corpus.db").exists());
        assert!(root.join("signals").join("original").is_dir());
        assert!(root.join("signals").join("derived").is_dir());
        assert!(root.join("attachments").is_dir());
        assert!(root.join("exports").is_dir());
        assert_eq!(project.name().unwrap(), "test_project");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn refdist_pins_round_trip_in_project_toml() {
        let root = unique_dir("refdist_pins");
        let _ = std::fs::remove_dir_all(&root);

        let project = Project::create(&root, "p_pins").unwrap();
        assert!(project.refdist_pins().unwrap().is_empty());

        project
            .pin_refdist("hillenbrand-1995-amE-vowels", "1.0.0")
            .unwrap();
        project
            .pin_refdist("clinical-jitter-norms", "0.2.0")
            .unwrap();
        // Overwriting an existing pin replaces the version.
        project
            .pin_refdist("clinical-jitter-norms", "0.3.0")
            .unwrap();

        let pins = project.refdist_pins().unwrap();
        assert_eq!(
            pins,
            vec![
                ("clinical-jitter-norms".to_string(), "0.3.0".to_string()),
                (
                    "hillenbrand-1995-amE-vowels".to_string(),
                    "1.0.0".to_string()
                ),
            ]
        );

        // The original project.toml keys survive the rewrite.
        let reopened = Project::open(&root).unwrap();
        assert_eq!(reopened.name().unwrap(), "p_pins");
        assert_eq!(reopened.refdist_pins().unwrap().len(), 2);

        assert!(
            reopened
                .remove_refdist_pin("clinical-jitter-norms")
                .unwrap()
        );
        assert!(
            !reopened
                .remove_refdist_pin("clinical-jitter-norms")
                .unwrap()
        );
        assert_eq!(reopened.refdist_pins().unwrap().len(), 1);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn open_existing_project_works() {
        let root = unique_dir("open_existing");
        let _ = std::fs::remove_dir_all(&root);

        Project::create(&root, "p1").unwrap();
        let reopened = Project::open(&root).unwrap();
        assert_eq!(reopened.name().unwrap(), "p1");

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn open_on_non_project_directory_errors() {
        let root = unique_dir("not_a_project");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let err = Project::open(&root).unwrap_err();
        assert!(matches!(err, EngineError::Corpus(_)));

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn add_bundle_copies_audio_and_records_metadata() {
        let root = unique_dir("add_bundle");
        let _ = std::fs::remove_dir_all(&root);

        let source_wav = std::env::temp_dir().join(format!(
            "sadda_engine_corpus_source_{}.wav",
            std::process::id()
        ));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("greeting", &source_wav).unwrap();

        let copied = root
            .join("signals")
            .join("original")
            .join(source_wav.file_name().unwrap());
        assert!(copied.exists(), "audio file should be copied into project");

        let bundles = project.bundles().unwrap();
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].id, bundle_id);
        assert_eq!(bundles[0].name, "greeting");
        assert_eq!(bundles[0].sample_rate, 16_000);
        assert_eq!(bundles[0].channels, 1);
        assert_eq!(bundles[0].n_frames, 4_000);

        let audio = project.load_audio(bundle_id).unwrap();
        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.frame_count(), 4_000);

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rename_bundle_updates_name_trims_and_validates() {
        let root = unique_dir("rename_bundle");
        let _ = std::fs::remove_dir_all(&root);

        let source_wav = std::env::temp_dir().join(format!(
            "sadda_engine_rename_source_{}.wav",
            std::process::id()
        ));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("greeting", &source_wav).unwrap();
        let original_audio_path = project.bundles().unwrap()[0].audio_relative_path.clone();

        // Rename trims surrounding whitespace and leaves the WAV path alone.
        project.rename_bundle(bundle_id, "  farewell  ").unwrap();
        let bundles = project.bundles().unwrap();
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].name, "farewell");
        assert_eq!(bundles[0].audio_relative_path, original_audio_path);

        // Empty / whitespace-only names are rejected.
        let err = project.rename_bundle(bundle_id, "   ").unwrap_err();
        assert!(matches!(err, EngineError::Corpus(_)));
        assert_eq!(project.bundles().unwrap()[0].name, "farewell");

        // Unknown id errors rather than silently no-op'ing.
        let err = project.rename_bundle(9_999, "x").unwrap_err();
        assert!(matches!(err, EngineError::Corpus(_)));

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rename_and_delete_tier_lifecycle() {
        let root = unique_dir("tier_lifecycle");
        let _ = std::fs::remove_dir_all(&root);
        let source_wav =
            std::env::temp_dir().join(format!("sadda_tier_life_{}.wav", std::process::id()));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("b", &source_wav).unwrap();
        let tier_id = project
            .add_tier(&TierSpec::new(bundle_id, "phones", TierType::Interval))
            .unwrap();
        for (s, e, lbl) in [(0.0, 0.1, "a"), (0.1, 0.2, "b")] {
            project
                .add_interval(&IntervalSpec {
                    tier_id,
                    start_seconds: s,
                    end_seconds: e,
                    label: Some(lbl.into()),
                    parent_annotation_id: None,
                    status: None,
                    note: None,
                    extra: None,
                    ..Default::default()
                })
                .unwrap();
        }

        // Rename: trims, rejects empty, errors on unknown id.
        project.rename_tier(tier_id, "  segments  ").unwrap();
        assert_eq!(project.get_tier(tier_id).unwrap().name, "segments");
        assert!(matches!(
            project.rename_tier(tier_id, "   ").unwrap_err(),
            EngineError::Corpus(_)
        ));
        assert!(matches!(
            project.rename_tier(9_999, "x").unwrap_err(),
            EngineError::Corpus(_)
        ));

        // Delete cascades the tier's annotations, then errors if repeated.
        assert_eq!(project.intervals(tier_id).unwrap().len(), 2);
        project.delete_tier(tier_id).unwrap();
        assert!(project.tiers(Some(bundle_id)).unwrap().is_empty());
        assert!(project.intervals(tier_id).unwrap().is_empty());
        assert!(matches!(
            project.delete_tier(tier_id).unwrap_err(),
            EngineError::Corpus(_)
        ));

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn delete_tier_refuses_when_children_exist() {
        let root = unique_dir("tier_children");
        let _ = std::fs::remove_dir_all(&root);
        let source_wav =
            std::env::temp_dir().join(format!("sadda_tier_child_{}.wav", std::process::id()));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("b", &source_wav).unwrap();
        let parent = project
            .add_tier(&TierSpec::new(bundle_id, "words", TierType::Interval))
            .unwrap();
        let child = project
            .add_tier(&TierSpec {
                parent_id: Some(parent),
                cardinality: Some("one_to_many".into()),
                ..TierSpec::new(bundle_id, "phones", TierType::Interval)
            })
            .unwrap();

        // Refuses while a child references it.
        assert!(matches!(
            project.delete_tier(parent).unwrap_err(),
            EngineError::Corpus(_)
        ));
        // Delete the child first, then the parent succeeds.
        project.delete_tier(child).unwrap();
        project.delete_tier(parent).unwrap();
        assert!(project.tiers(Some(bundle_id)).unwrap().is_empty());

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn record_processing_run_and_query_timeline() {
        let root = unique_dir("processing_run");
        let _ = std::fs::remove_dir_all(&root);
        let source_wav = std::env::temp_dir().join(format!(
            "sadda_engine_prov_source_{}.wav",
            std::process::id()
        ));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("greeting", &source_wav).unwrap();

        // Fresh bundle: no runs yet.
        assert!(project.processing_runs(bundle_id).unwrap().is_empty());

        let mut pitch = ProcessingRunSpec::new(
            bundle_id,
            ProcessingRunKind::DspAlgorithm,
            "sadda.dsp.pitch.autocorrelation",
        );
        pitch.parameters = Some("{\"step\":0.01}".into());
        pitch.output_signal_ids = vec![1];
        let pid = project.record_processing_run(&pitch).unwrap();
        project
            .record_processing_run(&ProcessingRunSpec::new(
                bundle_id,
                ProcessingRunKind::DspAlgorithm,
                "sadda.dsp.mfcc",
            ))
            .unwrap();

        let runs = project.processing_runs(bundle_id).unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].id, pid);
        assert_eq!(runs[0].processor_id, "sadda.dsp.pitch.autocorrelation");
        assert_eq!(runs[0].kind, "dsp_algorithm");
        assert_eq!(runs[0].processor_version, crate::version());
        assert_eq!(runs[0].status, "ok");
        assert!(runs[0].finished_at.is_some());

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn citations_dedup_and_omit_uncited() {
        let root = unique_dir("citations");
        let _ = std::fs::remove_dir_all(&root);
        let source_wav = std::env::temp_dir().join(format!(
            "sadda_engine_cite_source_{}.wav",
            std::process::id()
        ));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("greeting", &source_wav).unwrap();

        // Two pitch runs (same processor) + one mfcc + one uncited tool op.
        for _ in 0..2 {
            project
                .record_processing_run(&ProcessingRunSpec::new(
                    bundle_id,
                    ProcessingRunKind::DspAlgorithm,
                    "sadda.dsp.pitch.windowed_autocorrelation",
                ))
                .unwrap();
        }
        project
            .record_processing_run(&ProcessingRunSpec::new(
                bundle_id,
                ProcessingRunKind::DspAlgorithm,
                "sadda.dsp.mfcc",
            ))
            .unwrap();
        project
            .record_processing_run(&ProcessingRunSpec::new(
                bundle_id,
                ProcessingRunKind::DspAlgorithm,
                "sadda.io.textgrid.import",
            ))
            .unwrap();

        let cites = project.citations(bundle_id).unwrap();
        // pitch (deduped to 1) + mfcc = 2; the import op is uncited.
        assert_eq!(cites.len(), 2);
        assert!(cites[0].reference.contains("Boersma"));
        assert!(cites[1].reference.contains("Davis"));

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn instrument_calibration_roundtrips_and_resolves_via_bundle() {
        let root = unique_dir("instrument_cal");
        let _ = std::fs::remove_dir_all(&root);
        let wav =
            std::env::temp_dir().join(format!("sadda_engine_cal_a_{}.wav", std::process::id()));
        let wav2 =
            std::env::temp_dir().join(format!("sadda_engine_cal_b_{}.wav", std::process::id()));
        write_short_wav(&wav, 16_000);
        write_short_wav(&wav2, 16_000);

        let project = Project::create(&root, "p").unwrap();

        // A 94 dB-SPL calibration tone read -20 dB-FS → +114 dB offset.
        let cal = Calibration {
            reference_spl_db: 94.0,
            reference_db_fs: -20.0,
        };
        let mut ispec = InstrumentSpec::new("B&K 4189");
        ispec.kind = Some("microphone".into());
        ispec.calibration = Some(cal);
        let instrument_id = project.add_instrument(&ispec).unwrap();

        // Round-trips through the calibration JSON column.
        let got = project.get_instrument(instrument_id).unwrap();
        assert_eq!(got.name, "B&K 4189");
        assert_eq!(got.calibration, Some(cal));
        assert!((cal.spl_offset_db() - 114.0).abs() < 1e-9);
        // -26 dB-FS + 114 → 88 dB-SPL.
        assert!((cal.to_spl(crate::units::Decibels::new(-26.0)).value() - 88.0).abs() < 1e-4);

        // bundle → session → instrument resolves the calibration.
        let mut sspec = SessionSpec::new("s1");
        sspec.instrument_id = Some(instrument_id);
        let session_id = project.add_session(&sspec).unwrap();
        let mut bspec = BundleSpec::new("greeting");
        bspec.session_id = Some(session_id);
        let calibrated = project.add_bundle_with(&bspec, &wav).unwrap();
        assert_eq!(project.bundle_calibration(calibrated).unwrap(), Some(cal));

        // A bundle with no session is uncalibrated (dB-FS only).
        let plain = project.add_bundle("plain", &wav2).unwrap();
        assert_eq!(project.bundle_calibration(plain).unwrap(), None);
        assert_eq!(project.bundle_calibration(9_999).unwrap(), None);

        let _ = std::fs::remove_file(&wav);
        let _ = std::fs::remove_file(&wav2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn create_on_existing_empty_path_succeeds() {
        // Most common GUI workflow: the user picks an already-
        // existing empty folder via a file dialog. Project::create
        // should treat that as "use this folder."
        let root = unique_dir("empty_existing");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let project = Project::create(&root, "p").unwrap();
        assert_eq!(project.name().unwrap(), "p");
        assert!(root.join("project.toml").is_file());
        assert!(root.join("corpus.db").is_file());

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn create_on_existing_non_empty_path_errors() {
        // The directory exists AND has content (could be a previous
        // project, could be arbitrary user files). Don't clobber it.
        let root = unique_dir("non_empty_existing");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("some_file.txt"), b"hi").unwrap();

        let err = Project::create(&root, "p").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("not empty"),
            "expected 'not empty' in error, got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn rubric_status_note_and_controlled_vocabulary() {
        let root = unique_dir("rubric");
        let _ = std::fs::remove_dir_all(&root);
        let source_wav =
            std::env::temp_dir().join(format!("sadda_rubric_{}.wav", std::process::id()));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("b", &source_wav).unwrap();
        let tier_id = project
            .add_tier(&TierSpec::new(bundle_id, "phones", TierType::Interval))
            .unwrap();

        // No rubric yet: a plain interval inserts fine (status/note default to
        // None), but setting a status is rejected (no status vocabulary).
        let iv = project
            .add_interval(&IntervalSpec {
                tier_id,
                start_seconds: 0.0,
                end_seconds: 0.1,
                label: Some("a".into()),
                ..Default::default()
            })
            .unwrap();
        assert!(project.intervals(tier_id).unwrap()[0].status.is_none());
        assert!(project.set_interval_status(iv, Some("done"), None).is_err());

        // Define the rubric + an arbitrary status vocabulary.
        let rubric = project
            .set_rubric("IPA segmentation", 1, Some("Label each phone."))
            .unwrap();
        assert_eq!(rubric.id, 1);
        project
            .set_rubric_statuses(&[
                StatusDef {
                    value: "draft".into(),
                    description: None,
                    sort_order: 0,
                },
                StatusDef {
                    value: "done".into(),
                    description: Some("reviewed".into()),
                    sort_order: 1,
                },
            ])
            .unwrap();
        let statuses: Vec<String> = project
            .rubric_statuses()
            .unwrap()
            .into_iter()
            .map(|s| s.value)
            .collect();
        assert_eq!(statuses, ["draft", "done"]);

        // Status + note now round-trip; an undefined status is rejected; a
        // note may be set on an annotation of any status (including none).
        project
            .set_interval_status(iv, Some("done"), Some("looks good"))
            .unwrap();
        let row = project.intervals(tier_id).unwrap().remove(0);
        assert_eq!(row.status.as_deref(), Some("done"));
        assert_eq!(row.note.as_deref(), Some("looks good"));
        assert!(
            project
                .set_interval_status(iv, Some("bogus"), None)
                .is_err()
        );
        project
            .set_interval_status(iv, None, Some("revisit"))
            .unwrap();
        let row = project.intervals(tier_id).unwrap().remove(0);
        assert!(row.status.is_none());
        assert_eq!(row.note.as_deref(), Some("revisit"));

        // Controlled vocabulary, open by default: out-of-vocab is flagged but
        // still accepted on insert.
        project
            .set_controlled_vocabulary(
                "phones",
                &[
                    VocabEntry {
                        value: "a".into(),
                        description: None,
                        sort_order: 0,
                    },
                    VocabEntry {
                        value: "i".into(),
                        description: None,
                        sort_order: 1,
                    },
                ],
            )
            .unwrap();
        assert_eq!(project.controlled_vocabulary("phones").unwrap().len(), 2);
        let chk = project.label_check("phones", Some("a")).unwrap();
        assert!(chk.has_vocabulary && !chk.closed && chk.in_vocabulary);
        let chk = project.label_check("phones", Some("zzz")).unwrap();
        assert!(chk.has_vocabulary && !chk.closed && !chk.in_vocabulary);
        project
            .add_interval(&IntervalSpec {
                tier_id,
                start_seconds: 0.2,
                end_seconds: 0.3,
                label: Some("zzz".into()),
                ..Default::default()
            })
            .expect("open vocabulary accepts out-of-vocab labels");

        // Close the vocabulary: out-of-vocab labels are now rejected at entry,
        // but in-vocab / empty / absent labels still insert.
        project
            .set_rubric_tier("phones", Some("IPA phones"), true)
            .unwrap();
        assert!(
            project
                .rubric_tier("phones")
                .unwrap()
                .unwrap()
                .closed_vocabulary
        );
        assert!(
            project
                .add_interval(&IntervalSpec {
                    tier_id,
                    start_seconds: 0.3,
                    end_seconds: 0.4,
                    label: Some("zzz".into()),
                    ..Default::default()
                })
                .is_err()
        );
        project
            .add_interval(&IntervalSpec {
                tier_id,
                start_seconds: 0.4,
                end_seconds: 0.5,
                label: Some("i".into()),
                ..Default::default()
            })
            .unwrap();
        project
            .add_interval(&IntervalSpec {
                tier_id,
                start_seconds: 0.5,
                end_seconds: 0.6,
                label: None,
                ..Default::default()
            })
            .unwrap();

        // A tier with no rubric configuration is unconstrained.
        let chk = project.label_check("untouched", Some("anything")).unwrap();
        assert!(!chk.has_vocabulary && !chk.closed && chk.in_vocabulary);

        // set_rubric updates the singleton in place, preserving created_at.
        let updated = project
            .set_rubric("IPA segmentation", 2, Some("v2 guidelines"))
            .unwrap();
        assert_eq!(updated.id, 1);
        assert_eq!(updated.version, 2);
        assert_eq!(updated.created_at, rubric.created_at);

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn structured_criterion_runs_proposes_and_accepts() {
        let root = unique_dir("criteria");
        let _ = std::fs::remove_dir_all(&root);
        let source_wav =
            std::env::temp_dir().join(format!("sadda_criteria_{}.wav", std::process::id()));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("b", &source_wav).unwrap();
        let phones = project
            .add_tier(&TierSpec::new(bundle_id, "phones", TierType::Interval))
            .unwrap();
        let words = project
            .add_tier(&TierSpec::new(bundle_id, "words", TierType::Interval))
            .unwrap();
        for (s, e, l) in [(0.0, 0.1, "a"), (0.1, 0.2, "b"), (0.5, 0.6, "a")] {
            project
                .add_interval(&IntervalSpec {
                    tier_id: phones,
                    start_seconds: s,
                    end_seconds: e,
                    label: Some(l.into()),
                    ..Default::default()
                })
                .unwrap();
        }
        // One "word" covering only the first two phones (0.0..0.3).
        project
            .add_interval(&IntervalSpec {
                tier_id: words,
                start_seconds: 0.0,
                end_seconds: 0.3,
                label: Some("stressed".into()),
                ..Default::default()
            })
            .unwrap();

        // Criterion: vowels "a" within a stressed word, emit the midpoint.
        let body = r#"{
            "select": {"tier": "phones", "label_any": ["a"]},
            "within": {"tier": "words", "label_any": ["stressed"]},
            "emit": {"kind": "point", "at": 0.5}
        }"#;
        let crit = project
            .set_criterion("vowel midpoints", None, "structured", body, "landmarks")
            .unwrap();
        assert_eq!(crit.kind, "structured");

        // A python criterion is stored but not runnable in the engine.
        let py = project
            .set_criterion("py", None, "python", "def c(): pass", "x")
            .unwrap();
        assert!(project.run_criterion(py.id, bundle_id).is_err());

        // Run: only the first "a" (0.0..0.1, inside the stressed word) qualifies
        // — the second "a" at 0.5 is outside it. One proposal at its midpoint.
        let n = project.run_criterion(crit.id, bundle_id).unwrap();
        assert_eq!(n, 1);
        let preview = project
            .tiers(Some(bundle_id))
            .unwrap()
            .into_iter()
            .find(|t| t.name == "landmarks (auto)")
            .expect("preview tier created");
        assert_eq!(preview.r#type, TierType::Point);
        let pts = project.points(preview.id).unwrap();
        assert_eq!(pts.len(), 1);
        assert!((pts[0].time_seconds - 0.05).abs() < 1e-9);

        // Re-running replaces (not appends) the proposals.
        assert_eq!(project.run_criterion(crit.id, bundle_id).unwrap(), 1);
        assert_eq!(project.points(preview.id).unwrap().len(), 1);

        // Accept: proposals promote to the "landmarks" tier and the preview
        // tier is cleared.
        let promoted = project.accept_proposals(bundle_id, "landmarks").unwrap();
        assert_eq!(promoted, 1);
        assert!(project.points(preview.id).unwrap().is_empty());
        let landmarks = project
            .tier_by_name(bundle_id, "landmarks")
            .unwrap()
            .unwrap();
        assert_eq!(project.points(landmarks.id).unwrap().len(), 1);

        // Reject path: run again, then clear discards the proposals.
        assert_eq!(project.run_criterion(crit.id, bundle_id).unwrap(), 1);
        assert_eq!(project.clear_proposals(bundle_id, "landmarks").unwrap(), 1);
        assert!(project.points(preview.id).unwrap().is_empty());

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn criterion_run_records_provenance_and_survives_accept() {
        let root = unique_dir("crit_prov");
        let _ = std::fs::remove_dir_all(&root);
        let source_wav =
            std::env::temp_dir().join(format!("sadda_crit_prov_{}.wav", std::process::id()));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("b", &source_wav).unwrap();
        let phones = project
            .add_tier(&TierSpec::new(bundle_id, "phones", TierType::Interval))
            .unwrap();
        for (s, e, l) in [(0.0, 0.1, "a"), (0.1, 0.2, "a")] {
            project
                .add_interval(&IntervalSpec {
                    tier_id: phones,
                    start_seconds: s,
                    end_seconds: e,
                    label: Some(l.into()),
                    ..Default::default()
                })
                .unwrap();
        }
        // Emit a span over every "a" onto the "vowels" target tier.
        let body = r#"{"select": {"tier": "phones", "label_any": ["a"]},
                       "emit": {"kind": "span"}}"#;
        let crit = project
            .set_criterion("vowels", None, "structured", body, "vowels")
            .unwrap();

        // No runs before; running records exactly one criterion_run.
        assert!(project.processing_runs(bundle_id).unwrap().is_empty());
        assert_eq!(project.run_criterion(crit.id, bundle_id).unwrap(), 2);

        let runs = project.processing_runs(bundle_id).unwrap();
        assert_eq!(runs.len(), 1);
        let run = &runs[0];
        assert_eq!(run.kind, "criterion_run");
        assert_eq!(run.processor_id, "sadda.criteria.vowels");
        // parameters capture the criterion id + a body checksum (repeatability).
        let params: serde_json::Value =
            serde_json::from_str(run.parameters.as_deref().unwrap()).unwrap();
        assert_eq!(params["criterion_id"], crit.id);
        assert_eq!(params["criterion_kind"], "structured");
        assert!(params["body_sha256"].as_str().unwrap().len() == 64);

        // get_processing_run resolves the same row.
        assert_eq!(project.get_processing_run(run.id).unwrap().unwrap().id, run.id);
        assert!(project.get_processing_run(99_999).unwrap().is_none());

        // Each preview proposal carries the run link.
        let preview = project.tier_by_name(bundle_id, "vowels (auto)").unwrap().unwrap();
        let proposals = project.intervals(preview.id).unwrap();
        assert_eq!(proposals.len(), 2);
        assert!(proposals.iter().all(|iv| iv.processing_run_id == Some(run.id)));

        // Re-running records a *second* run; the proposals now point at it.
        assert_eq!(project.run_criterion(crit.id, bundle_id).unwrap(), 2);
        let runs = project.processing_runs(bundle_id).unwrap();
        assert_eq!(runs.len(), 2);
        let latest = runs[1].id;
        let proposals = project.intervals(preview.id).unwrap();
        assert!(proposals.iter().all(|iv| iv.processing_run_id == Some(latest)));

        // Accept: the provenance link survives promotion onto the target tier.
        assert_eq!(project.accept_proposals(bundle_id, "vowels").unwrap(), 2);
        let target = project.tier_by_name(bundle_id, "vowels").unwrap().unwrap();
        let promoted = project.intervals(target.id).unwrap();
        assert_eq!(promoted.len(), 2);
        assert!(promoted.iter().all(|iv| iv.processing_run_id == Some(latest)));

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn target_crud_and_status_lifecycle() {
        let root = unique_dir("targets");
        let _ = std::fs::remove_dir_all(&root);
        let source_wav =
            std::env::temp_dir().join(format!("sadda_targets_{}.wav", std::process::id()));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("b", &source_wav).unwrap();

        // Manual target with defaults: status='unassigned', source='manual'.
        let id = project
            .add_target(&TargetSpec::new(bundle_id, 0.2, 0.5, "phones"))
            .unwrap();
        let t = project.get_target(id).unwrap().unwrap();
        assert_eq!(t.status, "unassigned");
        assert_eq!(t.source, "manual");
        assert_eq!(t.target_type, "phones");
        assert_eq!(t.criterion_id, None);

        // A second, earlier target — listing is time-ordered.
        project
            .add_target(&TargetSpec::new(bundle_id, 0.0, 0.1, "phones"))
            .unwrap();
        let listed = project.targets(bundle_id).unwrap();
        assert_eq!(listed.len(), 2);
        assert!(listed[0].start_seconds < listed[1].start_seconds);

        // Status lifecycle + note; bad status rejected; RoI sanity enforced.
        project.update_target_status(id, "in_progress").unwrap();
        project.set_target_note(id, Some("ambiguous vs rubric")).unwrap();
        let t = project.get_target(id).unwrap().unwrap();
        assert_eq!(t.status, "in_progress");
        assert_eq!(t.note.as_deref(), Some("ambiguous vs rubric"));
        assert!(project.update_target_status(id, "bogus").is_err());
        assert!(project.update_target_status(9_999, "done").is_err());
        assert!(
            project
                .add_target(&TargetSpec::new(bundle_id, 0.5, 0.5, "phones"))
                .is_err()
        );

        // Delete is idempotent.
        project.delete_target(id).unwrap();
        project.delete_target(id).unwrap();
        assert_eq!(project.targets(bundle_id).unwrap().len(), 1);

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn generate_targets_from_criterion_uses_roi_selection_and_replaces() {
        let root = unique_dir("gen_targets");
        let _ = std::fs::remove_dir_all(&root);
        let source_wav =
            std::env::temp_dir().join(format!("sadda_gen_targets_{}.wav", std::process::id()));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("b", &source_wav).unwrap();
        let phones = project
            .add_tier(&TierSpec::new(bundle_id, "phones", TierType::Interval))
            .unwrap();
        for (s, e, l) in [(0.0, 0.1, "a"), (0.1, 0.2, "b"), (0.5, 0.6, "a")] {
            project
                .add_interval(&IntervalSpec {
                    tier_id: phones,
                    start_seconds: s,
                    end_seconds: e,
                    label: Some(l.into()),
                    ..Default::default()
                })
                .unwrap();
        }
        // Criterion selecting "a" phones; emit is irrelevant to target RoIs.
        let body = r#"{"select": {"tier": "phones", "label_any": ["a"]},
                       "emit": {"kind": "span"}}"#;
        let crit = project
            .set_criterion("vowels", None, "structured", body, "vowel-detail")
            .unwrap();

        // Two "a" RoIs → two criterion-sourced targets, with the RoI spans and
        // the criterion's target tier as the type.
        let n = project
            .generate_targets_from_criterion(crit.id, bundle_id)
            .unwrap();
        assert_eq!(n, 2);
        let ts = project.targets(bundle_id).unwrap();
        assert_eq!(ts.len(), 2);
        assert!(ts.iter().all(|t| t.source == "criterion"
            && t.criterion_id == Some(crit.id)
            && t.target_type == "vowel-detail"
            && t.status == "unassigned"));
        assert!((ts[0].start_seconds - 0.0).abs() < 1e-9 && (ts[0].end_seconds - 0.1).abs() < 1e-9);
        assert!((ts[1].start_seconds - 0.5).abs() < 1e-9 && (ts[1].end_seconds - 0.6).abs() < 1e-9);

        // Regeneration replaces (not appends) this criterion's targets.
        assert_eq!(
            project
                .generate_targets_from_criterion(crit.id, bundle_id)
                .unwrap(),
            2
        );
        assert_eq!(project.targets(bundle_id).unwrap().len(), 2);

        // A python criterion can't generate targets in the engine.
        let py = project
            .set_criterion("py", None, "python", "def c(): pass", "x")
            .unwrap();
        assert!(
            project
                .generate_targets_from_criterion(py.id, bundle_id)
                .is_err()
        );

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn assignment_crud_and_target_status_management() {
        let root = unique_dir("assign");
        let _ = std::fs::remove_dir_all(&root);
        let source_wav =
            std::env::temp_dir().join(format!("sadda_assign_{}.wav", std::process::id()));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("b", &source_wav).unwrap();
        let tid = project
            .add_target(&TargetSpec::new(bundle_id, 0.0, 0.1, "phones"))
            .unwrap();

        // Assigning advances the target unassigned → assigned.
        let aid = project
            .add_assignment(&AssignmentSpec::new(tid, "alice"))
            .unwrap();
        assert_eq!(project.get_target(tid).unwrap().unwrap().status, "assigned");
        let a = &project.assignments(bundle_id).unwrap()[0];
        assert_eq!(a.annotator, "alice");
        assert_eq!(a.role, "primary");
        assert_eq!(a.status, "assigned");
        assert_eq!(a.seed, None);

        // A second annotator on the same target (overlap → S5 agreement).
        project
            .add_assignment(&AssignmentSpec {
                target_id: tid,
                annotator: "bob".into(),
                role: Some("secondary".into()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(project.assignments_for_target(tid).unwrap().len(), 2);
        // The same annotator twice is rejected (UNIQUE).
        assert!(
            project
                .add_assignment(&AssignmentSpec::new(tid, "alice"))
                .is_err()
        );
        // Bad role / status / missing target are rejected.
        assert!(
            project
                .add_assignment(&AssignmentSpec {
                    target_id: tid,
                    annotator: "x".into(),
                    role: Some("lead".into()),
                    ..Default::default()
                })
                .is_err()
        );
        assert!(
            project
                .add_assignment(&AssignmentSpec::new(9_999, "ghost"))
                .is_err()
        );

        // Editable: status + reassignment.
        project.update_assignment_status(aid, "in_progress").unwrap();
        project.set_assignment_annotator(aid, "carol").unwrap();
        let a = project.get_target(tid).unwrap().unwrap();
        assert_eq!(a.status, "assigned"); // target status unaffected by edits
        assert!(project.update_assignment_status(aid, "bogus").is_err());

        // Deleting assignments: the target reverts to unassigned only when the
        // LAST one is removed (and it was merely 'assigned').
        project.delete_assignment(aid).unwrap();
        assert_eq!(project.get_target(tid).unwrap().unwrap().status, "assigned");
        let bob = project.assignments_for_target(tid).unwrap()[0].id;
        project.delete_assignment(bob).unwrap();
        assert_eq!(
            project.get_target(tid).unwrap().unwrap().status,
            "unassigned"
        );
        project.delete_assignment(bob).unwrap(); // idempotent

        // Deleting a target removes its assignments (no orphans).
        project
            .add_assignment(&AssignmentSpec::new(tid, "dave"))
            .unwrap();
        project.delete_target(tid).unwrap();
        assert!(project.assignments(bundle_id).unwrap().is_empty());

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn assign_targets_randomly_is_deterministic_balanced_and_remainder_only() {
        let build = |tag: &str| {
            let root = unique_dir(tag);
            let _ = std::fs::remove_dir_all(&root);
            let wav = std::env::temp_dir().join(format!(
                "sadda_rand_assign_{}_{}.wav",
                std::process::id(),
                root.file_name().unwrap().to_string_lossy()
            ));
            write_short_wav(&wav, 16_000);
            let project = Project::create(&root, "p").unwrap();
            let bundle_id = project.add_bundle("b", &wav).unwrap();
            let mut ids = Vec::new();
            for i in 0..10 {
                let s = i as f64 * 0.1;
                ids.push(
                    project
                        .add_target(&TargetSpec::new(bundle_id, s, s + 0.05, "phones"))
                        .unwrap(),
                );
            }
            (root, wav, project, bundle_id, ids)
        };

        let roster = vec!["alice".to_string(), "bob".into(), "carol".into()];

        // First project: assign all 10 with seed 42.
        let (root1, wav1, p1, b1, _) = build("rand_assign_1");
        let n = p1.assign_targets_randomly(b1, &roster, 42, None).unwrap();
        assert_eq!(n, 10);
        let a1 = p1.assignments(b1).unwrap();
        assert_eq!(a1.len(), 10);
        // All targets now assigned; seed recorded.
        assert!(p1.targets(b1).unwrap().iter().all(|t| t.status == "assigned"));
        assert!(a1.iter().all(|a| a.seed == Some(42)));
        // Balanced: 10 across 3 → counts in {3,4}, difference ≤ 1.
        let mut counts = std::collections::HashMap::new();
        for a in &a1 {
            *counts.entry(a.annotator.clone()).or_insert(0) += 1;
        }
        let (min, max) = (
            *counts.values().min().unwrap(),
            *counts.values().max().unwrap(),
        );
        assert!(max - min <= 1, "unbalanced: {counts:?}");
        // A second call assigns nothing (no remaining unassigned targets).
        assert_eq!(p1.assign_targets_randomly(b1, &roster, 7, None).unwrap(), 0);
        // Empty roster errors.
        assert!(p1.assign_targets_randomly(b1, &[], 1, None).is_err());

        // Second identical project, same seed → identical (target→annotator) map.
        let (root2, wav2, p2, b2, _) = build("rand_assign_2");
        p2.assign_targets_randomly(b2, &roster, 42, None).unwrap();
        let map1: Vec<(i64, String)> =
            a1.iter().map(|a| (a.target_id, a.annotator.clone())).collect();
        let map2: Vec<(i64, String)> = p2
            .assignments(b2)
            .unwrap()
            .iter()
            .map(|a| (a.target_id, a.annotator.clone()))
            .collect();
        assert_eq!(map1, map2, "same seed must reproduce the same assignment");

        // Re-randomize-of-remainder: add 2 new targets, reassign only those.
        for i in 10..12 {
            let s = i as f64 * 0.1;
            p1.add_target(&TargetSpec::new(b1, s, s + 0.05, "phones"))
                .unwrap();
        }
        assert_eq!(p1.assign_targets_randomly(b1, &roster, 99, None).unwrap(), 2);
        assert_eq!(p1.assignments(b1).unwrap().len(), 12);

        for (root, wav) in [(root1, wav1), (root2, wav2)] {
            let _ = std::fs::remove_file(&wav);
            let _ = std::fs::remove_dir_all(&root);
        }
    }

    #[test]
    fn deterministic_shuffle_is_seed_stable_and_a_permutation() {
        let base: Vec<i32> = (0..50).collect();
        let mut a = base.clone();
        let mut b = base.clone();
        deterministic_shuffle(&mut a, 12345);
        deterministic_shuffle(&mut b, 12345);
        assert_eq!(a, b, "same seed → same permutation");

        let mut c = base.clone();
        deterministic_shuffle(&mut c, 999);
        assert_ne!(a, c, "different seeds should differ for 50 elements");

        // It is a permutation (same multiset) and actually reorders.
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, base);
        assert_ne!(a, base);
    }

    #[test]
    fn export_import_round_trip_lands_per_annotator_tier() {
        let root = unique_dir("pkg_parent");
        let pkg_dir = unique_dir("pkg_export");
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&pkg_dir);
        let source_wav =
            std::env::temp_dir().join(format!("sadda_pkg_{}.wav", std::process::id()));
        write_short_wav(&source_wav, 16_000);

        let parent = Project::create(&root, "study").unwrap();
        let bundle_id = parent.add_bundle("b", &source_wav).unwrap();
        // Context tier the annotator references.
        let phones = parent
            .add_tier(&TierSpec::new(bundle_id, "phones", TierType::Interval))
            .unwrap();
        parent
            .add_interval(&IntervalSpec {
                tier_id: phones,
                start_seconds: 0.0,
                end_seconds: 0.2,
                label: Some("a".into()),
                ..Default::default()
            })
            .unwrap();
        // A rubric, to exercise the rubric copy.
        parent.set_rubric("scheme", 1, Some("annotate vowels")).unwrap();
        // alice and bob each get a "vowels" target; export must include only alice's.
        let ta = parent
            .add_target(&TargetSpec::new(bundle_id, 0.0, 0.2, "vowels"))
            .unwrap();
        parent.add_assignment(&AssignmentSpec::new(ta, "alice")).unwrap();
        let tb = parent
            .add_target(&TargetSpec::new(bundle_id, 0.5, 0.7, "vowels"))
            .unwrap();
        parent.add_assignment(&AssignmentSpec::new(tb, "bob")).unwrap();

        let summary = parent
            .export_annotator_package("alice", &pkg_dir)
            .unwrap();
        assert_eq!((summary.bundles, summary.targets, summary.assignments), (1, 1, 1));

        // Simulate alice working in the package: the context + her target/assignment
        // are present (not bob's); she adds a "vowels" tier and annotates.
        {
            let pkg = Project::open(&pkg_dir).unwrap();
            let pb = pkg.bundles().unwrap()[0].id;
            assert_eq!(pkg.targets(pb).unwrap().len(), 1);
            assert_eq!(pkg.assignments(pb).unwrap()[0].annotator, "alice");
            let ph = pkg.tier_by_name(pb, "phones").unwrap().unwrap();
            assert_eq!(pkg.intervals(ph.id).unwrap().len(), 1); // context copied
            assert_eq!(pkg.rubric().unwrap().unwrap().name, "scheme"); // rubric copied
            let vowels = pkg
                .add_tier(&TierSpec::new(pb, "vowels", TierType::Interval))
                .unwrap();
            pkg.add_interval(&IntervalSpec {
                tier_id: vowels,
                start_seconds: 0.05,
                end_seconds: 0.15,
                label: Some("a".into()),
                ..Default::default()
            })
            .unwrap();
        } // drop the package handle so import can open it

        let imp = parent.import_annotator_package(&pkg_dir).unwrap();
        assert_eq!(imp.annotator, "alice");
        assert_eq!(
            (imp.bundles_matched, imp.tiers_imported, imp.annotations_imported),
            (1, 1, 1)
        );
        assert_eq!(imp.assignments_marked_done, 1);

        // alice's work landed on its own per-annotator tier.
        let valice = parent.tier_by_name(bundle_id, "vowels [alice]").unwrap().unwrap();
        assert_eq!(parent.intervals(valice.id).unwrap().len(), 1);
        // Her assignment is done; bob's is untouched.
        let assigns = parent.assignments(bundle_id).unwrap();
        assert_eq!(
            assigns.iter().find(|a| a.annotator == "alice").unwrap().status,
            "done"
        );
        assert_eq!(
            assigns.iter().find(|a| a.annotator == "bob").unwrap().status,
            "assigned"
        );

        // A directory without a manifest is rejected.
        assert!(parent.import_annotator_package(&root).is_err());

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&pkg_dir);
    }

    #[test]
    fn merge_tiers_unions_sources_in_time_order() {
        let root = unique_dir("merge");
        let _ = std::fs::remove_dir_all(&root);
        let source_wav =
            std::env::temp_dir().join(format!("sadda_merge_{}.wav", std::process::id()));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("b", &source_wav).unwrap();
        let alice = project
            .add_tier(&TierSpec::new(bundle_id, "phones [alice]", TierType::Interval))
            .unwrap();
        let bob = project
            .add_tier(&TierSpec::new(bundle_id, "phones [bob]", TierType::Interval))
            .unwrap();
        for (tier, s) in [(alice, 0.2), (alice, 0.0), (bob, 0.5)] {
            project
                .add_interval(&IntervalSpec {
                    tier_id: tier,
                    start_seconds: s,
                    end_seconds: s + 0.05,
                    label: Some("a".into()),
                    ..Default::default()
                })
                .unwrap();
        }

        let n = project
            .merge_tiers(
                bundle_id,
                &["phones [alice]".into(), "phones [bob]".into()],
                "phones",
            )
            .unwrap();
        assert_eq!(n, 3);
        let merged = project.tier_by_name(bundle_id, "phones").unwrap().unwrap();
        let ivs = project.intervals(merged.id).unwrap();
        // Unioned and time-ordered.
        let starts: Vec<f64> = ivs.iter().map(|i| i.start_seconds).collect();
        assert_eq!(starts, vec![0.0, 0.2, 0.5]);

        // Re-merging replaces (idempotent); a missing source errors.
        assert_eq!(
            project
                .merge_tiers(bundle_id, &["phones [alice]".into()], "phones")
                .unwrap(),
            2
        );
        assert!(
            project
                .merge_tiers(bundle_id, &["nope".into()], "phones")
                .is_err()
        );

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn compare_tiers_reports_agreement_and_guards_types() {
        let root = unique_dir("compare");
        let _ = std::fs::remove_dir_all(&root);
        let source_wav =
            std::env::temp_dir().join(format!("sadda_compare_{}.wav", std::process::id()));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("b", &source_wav).unwrap();
        let alice = project
            .add_tier(&TierSpec::new(bundle_id, "phones [alice]", TierType::Interval))
            .unwrap();
        let bob = project
            .add_tier(&TierSpec::new(bundle_id, "phones [bob]", TierType::Interval))
            .unwrap();
        // Same spans; bob disagrees on the last label.
        for (tier, flip) in [(alice, false), (bob, true)] {
            for (k, (s, e, l)) in [(0.0, 0.1, "a"), (0.1, 0.2, "b"), (0.2, 0.3, "a")]
                .into_iter()
                .enumerate()
            {
                let label = if flip && k == 2 { "c" } else { l };
                project
                    .add_interval(&IntervalSpec {
                        tier_id: tier,
                        start_seconds: s,
                        end_seconds: e,
                        label: Some(label.into()),
                        ..Default::default()
                    })
                    .unwrap();
            }
        }
        let r = project.compare_tiers(bundle_id, alice, bob, None).unwrap();
        assert_eq!(r.tier_type, "interval");
        assert_eq!((r.n_matched, r.n_only_a, r.n_only_b), (3, 0, 0));
        assert!((r.percent_label_agreement - 2.0 / 3.0).abs() < 1e-9);
        assert!(r.mean_abs_boundary_diff < 1e-9);

        // A point tier can't be compared against an interval tier.
        let pt = project
            .add_tier(&TierSpec::new(bundle_id, "marks", TierType::Point))
            .unwrap();
        assert!(project.compare_tiers(bundle_id, alice, pt, None).is_err());

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn target_progress_and_next_target_drive_the_work_queue() {
        let root = unique_dir("workqueue");
        let _ = std::fs::remove_dir_all(&root);
        let source_wav =
            std::env::temp_dir().join(format!("sadda_workqueue_{}.wav", std::process::id()));
        write_short_wav(&source_wav, 16_000);

        let project = Project::create(&root, "p").unwrap();
        let bundle_id = project.add_bundle("b", &source_wav).unwrap();
        let t0 = project
            .add_target(&TargetSpec::new(bundle_id, 0.0, 0.1, "phones"))
            .unwrap();
        let t1 = project
            .add_target(&TargetSpec::new(bundle_id, 0.2, 0.3, "phones"))
            .unwrap();
        let _t2 = project
            .add_target(&TargetSpec::new(bundle_id, 0.4, 0.5, "phones"))
            .unwrap();
        project.update_target_status(t0, "done").unwrap();
        project.update_target_status(t1, "flagged").unwrap();

        let p = project.target_progress(bundle_id).unwrap();
        assert_eq!((p.total, p.done, p.flagged, p.unassigned), (3, 1, 1, 1));

        // Next to-do (unassigned) is the third target; next flagged is t1.
        let todo = project
            .next_target(bundle_id, &["unassigned".into()])
            .unwrap()
            .unwrap();
        assert!((todo.start_seconds - 0.4).abs() < 1e-9);
        let flagged = project
            .next_target(bundle_id, &["flagged".into()])
            .unwrap()
            .unwrap();
        assert_eq!(flagged.id, t1);
        // Nothing in_progress.
        assert!(
            project
                .next_target(bundle_id, &["in_progress".into()])
                .unwrap()
                .is_none()
        );

        let _ = std::fs::remove_file(&source_wav);
        let _ = std::fs::remove_dir_all(&root);
    }
}
