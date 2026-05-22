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

impl Project {
    /// Creates a new project at `path`. The path must not exist yet. Lays
    /// down the directory tree, opens a fresh `corpus.db`, runs the full
    /// migration chain, and writes the `project.toml` marker.
    pub fn create(path: impl AsRef<Path>, name: &str) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        if root.exists() {
            return Err(EngineError::Corpus(format!(
                "project path already exists: {}",
                root.display()
            )));
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

        Ok(Project { root, conn })
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

        Ok(Project { root, conn })
    }

    /// Returns the project's filesystem root directory.
    pub fn root(&self) -> &Path {
        &self.root
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
        let parent_annotation_id = parent_annotation_id.ok_or_else(|| {
            EngineError::Cardinality(format!(
                "tier {} has parent tier {}; parent_annotation_id is required",
                child_tier.id, parent_tier_id
            ))
        })?;

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
                 parameters, output_tier_ids, finished_at, status) \
             VALUES (?1, 'dsp_algorithm', 'sadda.io.textgrid.import', ?2, ?3, ?4, \
                     strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), 'ok')",
            rusqlite::params![bundle_id, crate::version(), params, output_tier_ids,],
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
    fn create_on_existing_path_errors() {
        let root = unique_dir("already_exists");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let err = Project::create(&root, "p").unwrap_err();
        assert!(matches!(err, EngineError::Corpus(_)));

        let _ = std::fs::remove_dir_all(&root);
    }
}
