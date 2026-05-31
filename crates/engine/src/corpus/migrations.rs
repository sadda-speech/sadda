//! Forward-only schema-migration framework for the corpus database.
//!
//! Migrations are static, ordered by integer version, embedded at compile
//! time. Each pending migration runs inside its own transaction and is
//! followed (for `version >= 2`) by an `INSERT INTO schema_migrations`
//! recording version, name, and checksum. The V1 baseline writes its own row
//! because at V1 the table only has the `(version, applied_at)` columns —
//! V2 then extends the table and backfills V1's provenance.
//!
//! Design: see the 2026-05-21 DEVLOG entry "Migration framework (A1)".

use rusqlite::{Connection, Transaction};
use sha2::{Digest, Sha256};

use crate::error::{EngineError, Result};

const V1_SQL: &str = include_str!("../../migrations/V1__phase0_baseline.sql");
const V2_SQL: &str = include_str!("../../migrations/V2__schema_migrations_provenance.sql");
const V3_SQL: &str = include_str!("../../migrations/V3__entity_schema.sql");
const V4_SQL: &str = include_str!("../../migrations/V4__sparse_annotations.sql");
const V5_SQL: &str = include_str!("../../migrations/V5__derived_signal.sql");
const V6_SQL: &str = include_str!("../../migrations/V6__processing_run_kind_live_recording.sql");
const V7_SQL: &str = include_str!("../../migrations/V7__recipe_run.sql");
const V8_SQL: &str = include_str!("../../migrations/V8__annotation_rubric.sql");
const V9_SQL: &str = include_str!("../../migrations/V9__criteria.sql");
const V10_SQL: &str = include_str!("../../migrations/V10__criterion_run_provenance.sql");
const V11_SQL: &str = include_str!("../../migrations/V11__target.sql");

/// One forward-only migration step.
struct Migration {
    /// Monotonically increasing version. Equals index + 1 inside `MIGRATIONS`.
    version: i64,
    /// Short slug, matches the filename between the version prefix and `.sql`.
    name: &'static str,
    /// What the migration actually does.
    kind: Kind,
}

enum Kind {
    /// Pure SQL. The runner `execute_batch`'s the body inside the transaction.
    Sql(&'static str),
    /// Rust closure. Receives the in-flight transaction; runs SQL plus any
    /// logic that doesn't fit a single SQL string (introspection, conditional
    /// backfill, …).
    Rust {
        run: fn(&Transaction) -> Result<()>,
        sql_for_checksum: &'static str,
    },
}

static MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        name: "phase0_baseline",
        kind: Kind::Sql(V1_SQL),
    },
    Migration {
        version: 2,
        name: "schema_migrations_provenance",
        kind: Kind::Rust {
            run: v2_run,
            sql_for_checksum: V2_SQL,
        },
    },
    Migration {
        version: 3,
        name: "entity_schema",
        kind: Kind::Sql(V3_SQL),
    },
    Migration {
        version: 4,
        name: "sparse_annotations",
        kind: Kind::Sql(V4_SQL),
    },
    Migration {
        version: 5,
        name: "derived_signal",
        kind: Kind::Sql(V5_SQL),
    },
    Migration {
        version: 6,
        name: "processing_run_kind_live_recording",
        kind: Kind::Sql(V6_SQL),
    },
    Migration {
        version: 7,
        name: "recipe_run",
        kind: Kind::Sql(V7_SQL),
    },
    Migration {
        version: 8,
        name: "annotation_rubric",
        kind: Kind::Sql(V8_SQL),
    },
    Migration {
        version: 9,
        name: "criteria",
        kind: Kind::Sql(V9_SQL),
    },
    Migration {
        version: 10,
        name: "criterion_run_provenance",
        kind: Kind::Sql(V10_SQL),
    },
    Migration {
        version: 11,
        name: "target",
        kind: Kind::Sql(V11_SQL),
    },
];

fn v2_run(tx: &Transaction) -> Result<()> {
    tx.execute_batch(V2_SQL)?;
    tx.execute(
        "UPDATE schema_migrations \
             SET name = ?1, checksum = ?2 \
           WHERE version = 1",
        rusqlite::params!["phase0_baseline", checksum_of(V1_SQL)],
    )?;
    Ok(())
}

/// Highest schema version this build of the engine knows how to apply.
pub fn engine_max_version() -> i64 {
    MIGRATIONS.last().map(|m| m.version).unwrap_or(0)
}

/// Summary of what `run` did.
#[derive(Debug, Clone, Copy)]
pub struct Outcome {
    /// Version recorded in `schema_migrations` before this run (0 if absent).
    pub from_version: i64,
    /// Version recorded in `schema_migrations` after this run.
    pub to_version: i64,
    /// Number of migrations applied this run.
    pub applied: usize,
}

/// Reads `MAX(version)` from `schema_migrations`. Returns 0 if the table
/// doesn't yet exist (a brand-new database).
pub fn current_db_version(conn: &Connection) -> Result<i64> {
    let exists: i64 = conn.query_row(
        "SELECT COUNT(*) FROM sqlite_master \
          WHERE type = 'table' AND name = 'schema_migrations'",
        [],
        |row| row.get(0),
    )?;
    if exists == 0 {
        return Ok(0);
    }
    let v: Option<i64> =
        conn.query_row("SELECT MAX(version) FROM schema_migrations", [], |row| {
            row.get(0)
        })?;
    Ok(v.unwrap_or(0))
}

/// Applies every pending migration in order. Refuses to run (returning
/// `EngineError::SchemaTooNew`) if the database is at a higher version than
/// this build of the engine recognises.
///
/// The forward-compat clamp and pre-migration backup live in `Project::open`,
/// not here, so unit tests of this module can run migrations against an
/// in-memory connection without touching the filesystem.
pub fn run(conn: &mut Connection) -> Result<Outcome> {
    let from = current_db_version(conn)?;
    let to = engine_max_version();
    if from > to {
        return Err(EngineError::SchemaTooNew {
            db_version: from,
            engine_max: to,
        });
    }
    let mut applied = 0usize;
    for m in MIGRATIONS.iter().filter(|m| m.version > from) {
        apply_one(conn, m)?;
        applied += 1;
    }
    Ok(Outcome {
        from_version: from,
        to_version: to,
        applied,
    })
}

fn apply_one(conn: &mut Connection, m: &Migration) -> Result<()> {
    let checksum = match &m.kind {
        Kind::Sql(sql) => checksum_of(sql),
        Kind::Rust {
            sql_for_checksum, ..
        } => checksum_of(sql_for_checksum),
    };
    let tx = conn.transaction()?;
    match &m.kind {
        Kind::Sql(sql) => tx.execute_batch(sql)?,
        Kind::Rust { run, .. } => run(&tx)?,
    }
    // V1 writes its own row (the table at V1 lacks name/checksum columns).
    // V2+ have the full column set and the runner writes the row.
    if m.version >= 2 {
        tx.execute(
            "INSERT INTO schema_migrations (version, name, checksum) \
             VALUES (?1, ?2, ?3)",
            rusqlite::params![m.version, m.name, checksum],
        )?;
    }
    tx.commit()?;
    Ok(())
}

fn checksum_of(sql: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(sql.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_max_version_matches_static_table() {
        assert_eq!(engine_max_version(), 11);
    }

    #[test]
    fn checksum_is_deterministic_and_distinguishes_bodies() {
        assert_eq!(checksum_of("hello"), checksum_of("hello"));
        assert_ne!(checksum_of("alpha"), checksum_of("beta"));
        assert_eq!(checksum_of("").len(), 64);
    }

    #[test]
    fn fresh_in_memory_db_walks_to_latest() {
        let mut conn = Connection::open_in_memory().unwrap();
        let outcome = run(&mut conn).unwrap();
        assert_eq!(outcome.from_version, 0);
        assert_eq!(outcome.to_version, engine_max_version());
        assert_eq!(outcome.applied, engine_max_version() as usize);
        assert_eq!(current_db_version(&conn).unwrap(), engine_max_version());

        let rows: Vec<(i64, Option<String>, Option<String>)> = conn
            .prepare("SELECT version, name, checksum FROM schema_migrations ORDER BY version")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(rows.len(), engine_max_version() as usize);
        assert_eq!(rows[0].0, 1);
        assert_eq!(rows[0].1.as_deref(), Some("phase0_baseline"));
        assert_eq!(rows[0].2.as_deref(), Some(checksum_of(V1_SQL).as_str()));
        assert_eq!(rows[1].0, 2);
        assert_eq!(rows[1].1.as_deref(), Some("schema_migrations_provenance"));
        assert_eq!(rows[1].2.as_deref(), Some(checksum_of(V2_SQL).as_str()));
        assert_eq!(rows[2].0, 3);
        assert_eq!(rows[2].1.as_deref(), Some("entity_schema"));
        assert_eq!(rows[2].2.as_deref(), Some(checksum_of(V3_SQL).as_str()));
        assert_eq!(rows[3].0, 4);
        assert_eq!(rows[3].1.as_deref(), Some("sparse_annotations"));
        assert_eq!(rows[3].2.as_deref(), Some(checksum_of(V4_SQL).as_str()));
        assert_eq!(rows[4].0, 5);
        assert_eq!(rows[4].1.as_deref(), Some("derived_signal"));
        assert_eq!(rows[4].2.as_deref(), Some(checksum_of(V5_SQL).as_str()));
    }

    #[test]
    fn second_run_is_a_noop() {
        let mut conn = Connection::open_in_memory().unwrap();
        run(&mut conn).unwrap();
        let again = run(&mut conn).unwrap();
        assert_eq!(again.from_version, engine_max_version());
        assert_eq!(again.to_version, engine_max_version());
        assert_eq!(again.applied, 0);
    }

    #[test]
    fn db_newer_than_engine_errors() {
        let mut conn = Connection::open_in_memory().unwrap();
        run(&mut conn).unwrap();
        conn.execute(
            "INSERT INTO schema_migrations (version, name, checksum) VALUES (?1, ?2, ?3)",
            rusqlite::params![99, "future", "deadbeef"],
        )
        .unwrap();
        match run(&mut conn) {
            Err(EngineError::SchemaTooNew {
                db_version,
                engine_max,
            }) => {
                assert_eq!(db_version, 99);
                assert_eq!(engine_max, engine_max_version());
            }
            other => panic!("expected SchemaTooNew, got {other:?}"),
        }
    }
}
