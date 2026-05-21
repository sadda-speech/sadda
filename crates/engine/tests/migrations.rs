//! Integration tests for the corpus-database migration framework. The in-
//! memory smoke tests live inside `corpus::migrations`; this file covers the
//! on-disk lifecycle that `Project::open` drives: Phase-0 DBs upgrading,
//! backups landing, and the forward-compat clamp triggering.

use std::path::{Path, PathBuf};

use rusqlite::Connection;
use sadda_engine::{EngineError, Project};

fn unique_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "sadda_engine_migrations_test_{}_{}",
        std::process::id(),
        name
    ));
    p
}

/// Lays down what a Phase-0 engine would have written to disk: a project
/// directory containing a corpus.db whose schema_migrations table has only
/// the `(version, applied_at)` columns, plus a single row at version=1.
fn make_phase0_project(root: &Path) {
    std::fs::create_dir_all(root.join("signals").join("original")).unwrap();
    std::fs::create_dir_all(root.join("signals").join("derived")).unwrap();
    std::fs::create_dir_all(root.join("attachments")).unwrap();
    std::fs::create_dir_all(root.join("exports")).unwrap();

    let conn = Connection::open(root.join("corpus.db")).unwrap();
    conn.execute_batch(
        "
        CREATE TABLE schema_migrations (
            version    INTEGER NOT NULL PRIMARY KEY,
            applied_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE project (
            id         INTEGER PRIMARY KEY CHECK (id = 1),
            name       TEXT    NOT NULL,
            profile    TEXT    NOT NULL DEFAULT 'phonetician',
            created_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE bundle (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            name                TEXT    NOT NULL,
            audio_relative_path TEXT    NOT NULL UNIQUE,
            sample_rate         INTEGER NOT NULL,
            channels            INTEGER NOT NULL,
            n_frames            INTEGER NOT NULL,
            created_at          TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO project (id, name) VALUES (1, 'phase0_project');
        INSERT INTO schema_migrations (version) VALUES (1);
        ",
    )
    .unwrap();
    drop(conn);
}

#[test]
fn fresh_create_persists_full_provenance() {
    let root = unique_dir("fresh_create_provenance");
    let _ = std::fs::remove_dir_all(&root);

    let _project = Project::create(&root, "fresh").unwrap();

    let conn = Connection::open(root.join("corpus.db")).unwrap();
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM schema_migrations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, sadda_engine::schema_version());

    let (v1_name, v1_checksum): (Option<String>, Option<String>) = conn
        .query_row(
            "SELECT name, checksum FROM schema_migrations WHERE version = 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(v1_name.as_deref(), Some("phase0_baseline"));
    assert!(
        v1_checksum
            .as_deref()
            .is_some_and(|s| s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit())),
        "V1 row should carry a 64-char hex SHA-256 checksum, got {v1_checksum:?}"
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn opening_phase0_db_upgrades_and_writes_backup() {
    let root = unique_dir("upgrade_phase0");
    let _ = std::fs::remove_dir_all(&root);
    make_phase0_project(&root);

    // Confirm pre-state: 2-column schema_migrations, one row.
    {
        let conn = Connection::open(root.join("corpus.db")).unwrap();
        let cols: Vec<String> = conn
            .prepare("PRAGMA table_info(schema_migrations)")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(cols, vec!["version", "applied_at"]);
    }

    let project = Project::open(&root).unwrap();
    assert_eq!(project.name().unwrap(), "phase0_project");

    // Backup file landed under the pre-migration version.
    let backup = root.join("corpus.db.bak.1");
    assert!(
        backup.exists(),
        "expected pre-migration backup at {}",
        backup.display()
    );

    // The backup retains the legacy 2-column shape; the live DB has 4 columns
    // and a backfilled V1 row.
    {
        let bak = Connection::open(&backup).unwrap();
        let bak_cols: Vec<String> = bak
            .prepare("PRAGMA table_info(schema_migrations)")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(bak_cols, vec!["version", "applied_at"]);

        let live = Connection::open(root.join("corpus.db")).unwrap();
        let live_cols: Vec<String> = live
            .prepare("PRAGMA table_info(schema_migrations)")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(
            live_cols,
            vec!["version", "applied_at", "name", "checksum"]
        );

        let v1: (Option<String>, Option<String>) = live
            .query_row(
                "SELECT name, checksum FROM schema_migrations WHERE version = 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(v1.0.as_deref(), Some("phase0_baseline"));
        assert!(v1.1.is_some(), "V1 checksum should be backfilled");
    }

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn second_open_does_not_re_backup_or_re_migrate() {
    let root = unique_dir("idempotent_open");
    let _ = std::fs::remove_dir_all(&root);
    make_phase0_project(&root);

    let _ = Project::open(&root).unwrap();
    let first_backup_mtime = std::fs::metadata(root.join("corpus.db.bak.1"))
        .unwrap()
        .modified()
        .unwrap();

    // Second open should not produce a new backup (already at latest).
    let _ = Project::open(&root).unwrap();
    let second_backup_mtime = std::fs::metadata(root.join("corpus.db.bak.1"))
        .unwrap()
        .modified()
        .unwrap();
    assert_eq!(first_backup_mtime, second_backup_mtime);

    // And no `corpus.db.bak.2` should appear.
    assert!(!root.join("corpus.db.bak.2").exists());

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn opening_a_future_db_returns_schema_too_new() {
    let root = unique_dir("future_db");
    let _ = std::fs::remove_dir_all(&root);
    let _ = Project::create(&root, "p").unwrap();

    // Forge a future migration row.
    {
        let conn = Connection::open(root.join("corpus.db")).unwrap();
        conn.execute(
            "INSERT INTO schema_migrations (version, name, checksum) VALUES (?1, ?2, ?3)",
            rusqlite::params![999, "from_the_future", "deadbeef"],
        )
        .unwrap();
    }

    let err = Project::open(&root).unwrap_err();
    match err {
        EngineError::SchemaTooNew {
            db_version,
            engine_max,
        } => {
            assert_eq!(db_version, 999);
            assert_eq!(engine_max, sadda_engine::schema_version());
        }
        other => panic!("expected SchemaTooNew, got: {other:?}"),
    }

    // No `corpus.db.bak.999` should have been written — clamp fires before
    // the backup step.
    assert!(!root.join("corpus.db.bak.999").exists());

    let _ = std::fs::remove_dir_all(&root);
}
