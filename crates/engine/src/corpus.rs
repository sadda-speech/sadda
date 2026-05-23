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

        Ok(Project {
            root,
            conn,
            recipe_run_id: std::cell::Cell::new(None),
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

        Ok(Project {
            root,
            conn,
            recipe_run_id: std::cell::Cell::new(None),
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
        let id: i64 = self.conn.query_row(
            "INSERT INTO annotation_interval \
                (tier_id, start_seconds, end_seconds, label, parent_annotation_id, extra) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6) RETURNING id",
            rusqlite::params![
                spec.tier_id,
                spec.start_seconds,
                spec.end_seconds,
                spec.label,
                spec.parent_annotation_id,
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
        self.conn.execute(
            "UPDATE annotation_interval \
                SET start_seconds = ?2, end_seconds = ?3, label = ?4, \
                    parent_annotation_id = ?5, extra = ?6 \
              WHERE id = ?1",
            rusqlite::params![
                id,
                spec.start_seconds,
                spec.end_seconds,
                spec.label,
                spec.parent_annotation_id,
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
            "SELECT id, tier_id, start_seconds, end_seconds, label, parent_annotation_id, extra \
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
                    extra: row.get(6)?,
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
        let id: i64 = self.conn.query_row(
            "INSERT INTO annotation_point \
                (tier_id, time_seconds, label, parent_annotation_id, extra) \
             VALUES (?1, ?2, ?3, ?4, ?5) RETURNING id",
            rusqlite::params![
                spec.tier_id,
                spec.time_seconds,
                spec.label,
                spec.parent_annotation_id,
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
        self.conn.execute(
            "UPDATE annotation_point \
                SET time_seconds = ?2, label = ?3, parent_annotation_id = ?4, extra = ?5 \
              WHERE id = ?1",
            rusqlite::params![
                id,
                spec.time_seconds,
                spec.label,
                spec.parent_annotation_id,
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
            "SELECT id, tier_id, time_seconds, label, parent_annotation_id, extra \
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
                    extra: row.get(5)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
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
        let output_tier_ids = format!(
            "[{}]",
            new_tier_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        self.conn.execute(
            "INSERT INTO processing_run \
                (bundle_id, kind, processor_id, processor_version, \
                 parameters, output_tier_ids, finished_at, status, recipe_run_id) \
             VALUES (?1, 'dsp_algorithm', 'sadda.io.textgrid.import', ?2, ?3, ?4, \
                     strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 'ok', ?5)",
            rusqlite::params![
                bundle_id,
                crate::version(),
                params,
                output_tier_ids,
                self.recipe_run_id.get(),
            ],
        )?;
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
        let output_tier_ids = format!(
            "[{}]",
            new_tier_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        self.conn.execute(
            "INSERT INTO processing_run \
                (bundle_id, kind, processor_id, processor_version, \
                 parameters, output_tier_ids, finished_at, status, recipe_run_id) \
             VALUES (?1, 'dsp_algorithm', 'sadda.io.eaf.import', ?2, ?3, ?4, \
                     strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 'ok', ?5)",
            rusqlite::params![
                bundle_id,
                crate::version(),
                params,
                output_tier_ids,
                self.recipe_run_id.get(),
            ],
        )?;
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
}
