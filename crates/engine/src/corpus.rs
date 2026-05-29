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
    /// `'dsp_algorithm'` | `'ml_model'` | `'clinical_measure'` | `'plugin'` | `'live_recording'`.
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
                 status, note, extra) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8) RETURNING id",
            rusqlite::params![
                spec.tier_id,
                spec.start_seconds,
                spec.end_seconds,
                spec.label,
                spec.parent_annotation_id,
                spec.status,
                spec.note,
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
                    status, note, extra \
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
                    extra: row.get(8)?,
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
                (tier_id, time_seconds, label, parent_annotation_id, status, note, extra) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7) RETURNING id",
            rusqlite::params![
                spec.tier_id,
                spec.time_seconds,
                spec.label,
                spec.parent_annotation_id,
                spec.status,
                spec.note,
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
            "SELECT id, tier_id, time_seconds, label, parent_annotation_id, status, note, extra \
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
                    extra: row.get(7)?,
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
}
