//! Project directory + SQLite-backed corpus database. Schema is owned by the
//! [`migrations`] module; this file holds the user-facing types (`Project`,
//! `Bundle`) and the project-directory layout.
//!
//! The full v1 entity model (Speaker, Session, Tier, AuditLog, ProcessingRun, …)
//! lands in the B-cluster slices; today the schema covers only `project`,
//! `bundle`, and `schema_migrations`.

pub mod migrations;

use std::path::{Path, PathBuf};

use rusqlite::Connection;

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
/// At Phase 0 a bundle is a single audio file plus its acoustic header.
/// The full v1 bundle model will additionally own tiers, derived signals,
/// instrument calibration, etc.
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
    /// database. Returns the new bundle's id.
    pub fn add_bundle(&self, name: &str, source_audio_path: impl AsRef<Path>) -> Result<i64> {
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
            "INSERT INTO bundle (name, audio_relative_path, sample_rate, channels, n_frames) \
             VALUES (?1, ?2, ?3, ?4, ?5) RETURNING id",
            (
                name,
                dest_rel.to_string_lossy().as_ref(),
                audio.sample_rate as i64,
                audio.channels as i64,
                audio.frame_count() as i64,
            ),
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Lists all bundles in id order.
    pub fn bundles(&self) -> Result<Vec<Bundle>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, audio_relative_path, sample_rate, channels, n_frames \
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
}

fn backup_corpus_db(conn: &Connection, db_path: &Path, from_version: i64) -> Result<()> {
    // Flush any WAL state into the main file so the copy is self-contained.
    conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")?;
    let backup_path = db_path.with_file_name(format!("corpus.db.bak.{from_version}"));
    std::fs::copy(db_path, &backup_path)?;
    Ok(())
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
