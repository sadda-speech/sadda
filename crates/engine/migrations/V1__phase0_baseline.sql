-- V1: Phase 0 baseline.
--
-- Restated so a fresh DB created by this engine walks the same migration
-- chain as a DB created by the Phase 0 engine. The Phase 0 code created the
-- three tables below via execute_batch and then inserted a single row into
-- schema_migrations (version = 1) directly; this migration preserves both
-- effects, including the self-insert.
--
-- At V1, schema_migrations has only (version, applied_at). The name and
-- checksum columns are added in V2, which also backfills the V1 row.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER NOT NULL PRIMARY KEY,
    applied_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS project (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    name       TEXT    NOT NULL,
    profile    TEXT    NOT NULL DEFAULT 'phonetician',
    created_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bundle (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT    NOT NULL,
    audio_relative_path TEXT    NOT NULL UNIQUE,
    sample_rate         INTEGER NOT NULL,
    channels            INTEGER NOT NULL,
    n_frames            INTEGER NOT NULL,
    created_at          TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO schema_migrations (version) VALUES (1);
