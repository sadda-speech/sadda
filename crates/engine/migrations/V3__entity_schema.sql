-- V3: Full entity schema + AuditLog (Phase 1, slice B1).
--
-- Adds the v1 entity tables (Speaker, Session, Instrument, Protocol, Entity,
-- EntityRef, Tier header, ProcessingRun), the Bundle column extension, the
-- audit_log infrastructure (audit_log + _audit_context + per-table triggers),
-- and the indexes that support common queries.
--
-- Design: see the 2026-05-21 DEVLOG entry "Full entity schema + AuditLog
-- (B1)" and the 2026-05-18 corpus data-model entry.
--
-- IMPORTANT — trigger-rebuild discipline:
-- Any future migration that ALTER-TABLEs one of the audited tables (those
-- listed below) MUST `DROP TRIGGER IF EXISTS <table>_audit_{insert,update,
-- delete}` and recreate them with the new column list. Otherwise the audit
-- payload silently stops including the new columns.
--
-- Audited tables (9):
--   speaker, session, instrument, protocol, entity, entity_ref,
--   bundle, tier, processing_run
-- NOT audited:
--   project (singleton), schema_migrations (managed by migrator),
--   audit_log itself (would recurse), _audit_context (engine-internal).


-- ============================================================
-- Entity tables
-- ============================================================

CREATE TABLE speaker (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    sex         TEXT,
    birth_year  INTEGER,
    notes       TEXT,
    extra       TEXT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE instrument (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    kind         TEXT,
    serial       TEXT,
    calibration  TEXT,
    extra        TEXT,
    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE protocol (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    description  TEXT,
    schema       TEXT,
    extra        TEXT,
    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE session (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL,
    started_at     TEXT,
    ended_at       TEXT,
    location       TEXT,
    instrument_id  INTEGER REFERENCES instrument(id),
    protocol_id    INTEGER REFERENCES protocol(id),
    notes          TEXT,
    extra          TEXT,
    created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX idx_session_instrument ON session(instrument_id);
CREATE INDEX idx_session_protocol   ON session(protocol_id);

CREATE TABLE entity (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    extra       TEXT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX idx_entity_kind ON entity(kind);

CREATE TABLE entity_ref (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id    INTEGER NOT NULL REFERENCES entity(id),
    target_kind  TEXT    NOT NULL CHECK (
        target_kind IN ('bundle', 'session', 'speaker', 'tier', 'annotation')
    ),
    target_id    INTEGER NOT NULL,
    role         TEXT,
    extra        TEXT
);
CREATE INDEX idx_entity_ref_entity ON entity_ref(entity_id);
CREATE INDEX idx_entity_ref_target ON entity_ref(target_kind, target_id);


-- ============================================================
-- Bundle extension (Phase 0 bundle gains session/speaker FKs + extra json)
-- ============================================================

ALTER TABLE bundle ADD COLUMN session_id INTEGER REFERENCES session(id);
ALTER TABLE bundle ADD COLUMN speaker_id INTEGER REFERENCES speaker(id);
ALTER TABLE bundle ADD COLUMN extra      TEXT;
CREATE INDEX idx_bundle_session ON bundle(session_id);
CREATE INDEX idx_bundle_speaker ON bundle(speaker_id);


-- ============================================================
-- Tier header (annotation rows + Parquet sidecar bookkeeping land in B2/B3)
-- ============================================================

CREATE TABLE tier (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    bundle_id    INTEGER NOT NULL REFERENCES bundle(id),
    name         TEXT    NOT NULL,
    type         TEXT    NOT NULL CHECK (
        type IN (
            'interval', 'point', 'reference',
            'continuous_numeric', 'continuous_vector', 'categorical_sampled'
        )
    ),
    parent_id    INTEGER REFERENCES tier(id),
    cardinality  TEXT    CHECK (
        cardinality IN ('one_to_one', 'one_to_many', 'many_to_one', 'none')
    ),
    schema       TEXT,
    extra        TEXT,
    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (bundle_id, name)
);
CREATE INDEX idx_tier_bundle ON tier(bundle_id);
CREATE INDEX idx_tier_parent ON tier(parent_id);


-- ============================================================
-- ProcessingRun (per the 2026-05-20 ML-model-registry entry; covers ML
-- models AND DSP algorithms AND clinical-composite measures AND plugins)
-- ============================================================

CREATE TABLE processing_run (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    bundle_id           INTEGER NOT NULL REFERENCES bundle(id),
    kind                TEXT    NOT NULL CHECK (
        kind IN ('ml_model', 'dsp_algorithm', 'clinical_measure', 'plugin')
    ),
    processor_id        TEXT    NOT NULL,
    processor_version   TEXT    NOT NULL,
    weights_checksum    TEXT,
    parameters          TEXT,
    input_tier_ids      TEXT,
    output_tier_ids     TEXT,
    output_signal_ids   TEXT,
    started_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    finished_at         TEXT,
    status              TEXT    NOT NULL DEFAULT 'ok' CHECK (
        status IN ('ok', 'error', 'partial')
    ),
    error_message       TEXT,
    recipe_run_id       INTEGER
);
CREATE INDEX idx_processing_run_bundle ON processing_run(bundle_id);


-- ============================================================
-- Audit infrastructure
-- ============================================================

-- _audit_context: singleton row holding the current user. Triggers read this
-- via subselect; the engine sets it on connection. Not audited.
CREATE TABLE _audit_context (
    id    INTEGER PRIMARY KEY CHECK (id = 1),
    user  TEXT    NOT NULL DEFAULT 'local'
);
INSERT INTO _audit_context (id, user) VALUES (1, 'local');

-- audit_log: append-only mutation history. Indexed by (table, row) and by
-- timestamp for the two most common access patterns.
CREATE TABLE audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    user        TEXT    NOT NULL,
    table_name  TEXT    NOT NULL,
    row_id      INTEGER NOT NULL,
    op          TEXT    NOT NULL CHECK (op IN ('insert', 'update', 'delete')),
    before      TEXT,
    after       TEXT
);
CREATE INDEX idx_audit_log_table_row ON audit_log(table_name, row_id);
CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp);


-- ============================================================
-- Triggers: 3 per audited table × 9 audited tables = 27 triggers.
-- Pattern: AFTER {INSERT,UPDATE,DELETE} writes an audit_log row whose
-- before/after columns hold json_object(...) snapshots of the row's
-- columns at the moment of the mutation. The user is read from
-- _audit_context; the timestamp from the audit_log column default.
-- ============================================================

-- speaker
CREATE TRIGGER speaker_audit_insert AFTER INSERT ON speaker BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'speaker', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'name', NEW.name, 'sex', NEW.sex,
                    'birth_year', NEW.birth_year, 'notes', NEW.notes,
                    'extra', NEW.extra, 'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER speaker_audit_update AFTER UPDATE ON speaker BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'speaker', NEW.id, 'update',
        json_object('id', OLD.id, 'name', OLD.name, 'sex', OLD.sex,
                    'birth_year', OLD.birth_year, 'notes', OLD.notes,
                    'extra', OLD.extra, 'created_at', OLD.created_at),
        json_object('id', NEW.id, 'name', NEW.name, 'sex', NEW.sex,
                    'birth_year', NEW.birth_year, 'notes', NEW.notes,
                    'extra', NEW.extra, 'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER speaker_audit_delete AFTER DELETE ON speaker BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'speaker', OLD.id, 'delete',
        json_object('id', OLD.id, 'name', OLD.name, 'sex', OLD.sex,
                    'birth_year', OLD.birth_year, 'notes', OLD.notes,
                    'extra', OLD.extra, 'created_at', OLD.created_at),
        NULL
    );
END;

-- instrument
CREATE TRIGGER instrument_audit_insert AFTER INSERT ON instrument BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'instrument', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'name', NEW.name, 'kind', NEW.kind,
                    'serial', NEW.serial, 'calibration', NEW.calibration,
                    'extra', NEW.extra, 'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER instrument_audit_update AFTER UPDATE ON instrument BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'instrument', NEW.id, 'update',
        json_object('id', OLD.id, 'name', OLD.name, 'kind', OLD.kind,
                    'serial', OLD.serial, 'calibration', OLD.calibration,
                    'extra', OLD.extra, 'created_at', OLD.created_at),
        json_object('id', NEW.id, 'name', NEW.name, 'kind', NEW.kind,
                    'serial', NEW.serial, 'calibration', NEW.calibration,
                    'extra', NEW.extra, 'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER instrument_audit_delete AFTER DELETE ON instrument BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'instrument', OLD.id, 'delete',
        json_object('id', OLD.id, 'name', OLD.name, 'kind', OLD.kind,
                    'serial', OLD.serial, 'calibration', OLD.calibration,
                    'extra', OLD.extra, 'created_at', OLD.created_at),
        NULL
    );
END;

-- protocol
CREATE TRIGGER protocol_audit_insert AFTER INSERT ON protocol BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'protocol', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'name', NEW.name, 'description', NEW.description,
                    'schema', NEW.schema, 'extra', NEW.extra,
                    'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER protocol_audit_update AFTER UPDATE ON protocol BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'protocol', NEW.id, 'update',
        json_object('id', OLD.id, 'name', OLD.name, 'description', OLD.description,
                    'schema', OLD.schema, 'extra', OLD.extra,
                    'created_at', OLD.created_at),
        json_object('id', NEW.id, 'name', NEW.name, 'description', NEW.description,
                    'schema', NEW.schema, 'extra', NEW.extra,
                    'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER protocol_audit_delete AFTER DELETE ON protocol BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'protocol', OLD.id, 'delete',
        json_object('id', OLD.id, 'name', OLD.name, 'description', OLD.description,
                    'schema', OLD.schema, 'extra', OLD.extra,
                    'created_at', OLD.created_at),
        NULL
    );
END;

-- session
CREATE TRIGGER session_audit_insert AFTER INSERT ON session BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'session', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'name', NEW.name, 'started_at', NEW.started_at,
                    'ended_at', NEW.ended_at, 'location', NEW.location,
                    'instrument_id', NEW.instrument_id, 'protocol_id', NEW.protocol_id,
                    'notes', NEW.notes, 'extra', NEW.extra,
                    'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER session_audit_update AFTER UPDATE ON session BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'session', NEW.id, 'update',
        json_object('id', OLD.id, 'name', OLD.name, 'started_at', OLD.started_at,
                    'ended_at', OLD.ended_at, 'location', OLD.location,
                    'instrument_id', OLD.instrument_id, 'protocol_id', OLD.protocol_id,
                    'notes', OLD.notes, 'extra', OLD.extra,
                    'created_at', OLD.created_at),
        json_object('id', NEW.id, 'name', NEW.name, 'started_at', NEW.started_at,
                    'ended_at', NEW.ended_at, 'location', NEW.location,
                    'instrument_id', NEW.instrument_id, 'protocol_id', NEW.protocol_id,
                    'notes', NEW.notes, 'extra', NEW.extra,
                    'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER session_audit_delete AFTER DELETE ON session BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'session', OLD.id, 'delete',
        json_object('id', OLD.id, 'name', OLD.name, 'started_at', OLD.started_at,
                    'ended_at', OLD.ended_at, 'location', OLD.location,
                    'instrument_id', OLD.instrument_id, 'protocol_id', OLD.protocol_id,
                    'notes', OLD.notes, 'extra', OLD.extra,
                    'created_at', OLD.created_at),
        NULL
    );
END;

-- entity
CREATE TRIGGER entity_audit_insert AFTER INSERT ON entity BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'entity', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'kind', NEW.kind, 'name', NEW.name,
                    'extra', NEW.extra, 'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER entity_audit_update AFTER UPDATE ON entity BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'entity', NEW.id, 'update',
        json_object('id', OLD.id, 'kind', OLD.kind, 'name', OLD.name,
                    'extra', OLD.extra, 'created_at', OLD.created_at),
        json_object('id', NEW.id, 'kind', NEW.kind, 'name', NEW.name,
                    'extra', NEW.extra, 'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER entity_audit_delete AFTER DELETE ON entity BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'entity', OLD.id, 'delete',
        json_object('id', OLD.id, 'kind', OLD.kind, 'name', OLD.name,
                    'extra', OLD.extra, 'created_at', OLD.created_at),
        NULL
    );
END;

-- entity_ref
CREATE TRIGGER entity_ref_audit_insert AFTER INSERT ON entity_ref BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'entity_ref', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'entity_id', NEW.entity_id,
                    'target_kind', NEW.target_kind, 'target_id', NEW.target_id,
                    'role', NEW.role, 'extra', NEW.extra)
    );
END;
CREATE TRIGGER entity_ref_audit_update AFTER UPDATE ON entity_ref BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'entity_ref', NEW.id, 'update',
        json_object('id', OLD.id, 'entity_id', OLD.entity_id,
                    'target_kind', OLD.target_kind, 'target_id', OLD.target_id,
                    'role', OLD.role, 'extra', OLD.extra),
        json_object('id', NEW.id, 'entity_id', NEW.entity_id,
                    'target_kind', NEW.target_kind, 'target_id', NEW.target_id,
                    'role', NEW.role, 'extra', NEW.extra)
    );
END;
CREATE TRIGGER entity_ref_audit_delete AFTER DELETE ON entity_ref BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'entity_ref', OLD.id, 'delete',
        json_object('id', OLD.id, 'entity_id', OLD.entity_id,
                    'target_kind', OLD.target_kind, 'target_id', OLD.target_id,
                    'role', OLD.role, 'extra', OLD.extra),
        NULL
    );
END;

-- bundle (post-V3 column set)
CREATE TRIGGER bundle_audit_insert AFTER INSERT ON bundle BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'bundle', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'name', NEW.name,
                    'audio_relative_path', NEW.audio_relative_path,
                    'sample_rate', NEW.sample_rate, 'channels', NEW.channels,
                    'n_frames', NEW.n_frames, 'created_at', NEW.created_at,
                    'session_id', NEW.session_id, 'speaker_id', NEW.speaker_id,
                    'extra', NEW.extra)
    );
END;
CREATE TRIGGER bundle_audit_update AFTER UPDATE ON bundle BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'bundle', NEW.id, 'update',
        json_object('id', OLD.id, 'name', OLD.name,
                    'audio_relative_path', OLD.audio_relative_path,
                    'sample_rate', OLD.sample_rate, 'channels', OLD.channels,
                    'n_frames', OLD.n_frames, 'created_at', OLD.created_at,
                    'session_id', OLD.session_id, 'speaker_id', OLD.speaker_id,
                    'extra', OLD.extra),
        json_object('id', NEW.id, 'name', NEW.name,
                    'audio_relative_path', NEW.audio_relative_path,
                    'sample_rate', NEW.sample_rate, 'channels', NEW.channels,
                    'n_frames', NEW.n_frames, 'created_at', NEW.created_at,
                    'session_id', NEW.session_id, 'speaker_id', NEW.speaker_id,
                    'extra', NEW.extra)
    );
END;
CREATE TRIGGER bundle_audit_delete AFTER DELETE ON bundle BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'bundle', OLD.id, 'delete',
        json_object('id', OLD.id, 'name', OLD.name,
                    'audio_relative_path', OLD.audio_relative_path,
                    'sample_rate', OLD.sample_rate, 'channels', OLD.channels,
                    'n_frames', OLD.n_frames, 'created_at', OLD.created_at,
                    'session_id', OLD.session_id, 'speaker_id', OLD.speaker_id,
                    'extra', OLD.extra),
        NULL
    );
END;

-- tier
CREATE TRIGGER tier_audit_insert AFTER INSERT ON tier BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'tier', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'bundle_id', NEW.bundle_id, 'name', NEW.name,
                    'type', NEW.type, 'parent_id', NEW.parent_id,
                    'cardinality', NEW.cardinality, 'schema', NEW.schema,
                    'extra', NEW.extra, 'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER tier_audit_update AFTER UPDATE ON tier BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'tier', NEW.id, 'update',
        json_object('id', OLD.id, 'bundle_id', OLD.bundle_id, 'name', OLD.name,
                    'type', OLD.type, 'parent_id', OLD.parent_id,
                    'cardinality', OLD.cardinality, 'schema', OLD.schema,
                    'extra', OLD.extra, 'created_at', OLD.created_at),
        json_object('id', NEW.id, 'bundle_id', NEW.bundle_id, 'name', NEW.name,
                    'type', NEW.type, 'parent_id', NEW.parent_id,
                    'cardinality', NEW.cardinality, 'schema', NEW.schema,
                    'extra', NEW.extra, 'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER tier_audit_delete AFTER DELETE ON tier BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'tier', OLD.id, 'delete',
        json_object('id', OLD.id, 'bundle_id', OLD.bundle_id, 'name', OLD.name,
                    'type', OLD.type, 'parent_id', OLD.parent_id,
                    'cardinality', OLD.cardinality, 'schema', OLD.schema,
                    'extra', OLD.extra, 'created_at', OLD.created_at),
        NULL
    );
END;

-- processing_run
CREATE TRIGGER processing_run_audit_insert AFTER INSERT ON processing_run BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'processing_run', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'bundle_id', NEW.bundle_id, 'kind', NEW.kind,
                    'processor_id', NEW.processor_id,
                    'processor_version', NEW.processor_version,
                    'weights_checksum', NEW.weights_checksum,
                    'parameters', NEW.parameters,
                    'input_tier_ids', NEW.input_tier_ids,
                    'output_tier_ids', NEW.output_tier_ids,
                    'output_signal_ids', NEW.output_signal_ids,
                    'started_at', NEW.started_at, 'finished_at', NEW.finished_at,
                    'status', NEW.status, 'error_message', NEW.error_message,
                    'recipe_run_id', NEW.recipe_run_id)
    );
END;
CREATE TRIGGER processing_run_audit_update AFTER UPDATE ON processing_run BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'processing_run', NEW.id, 'update',
        json_object('id', OLD.id, 'bundle_id', OLD.bundle_id, 'kind', OLD.kind,
                    'processor_id', OLD.processor_id,
                    'processor_version', OLD.processor_version,
                    'weights_checksum', OLD.weights_checksum,
                    'parameters', OLD.parameters,
                    'input_tier_ids', OLD.input_tier_ids,
                    'output_tier_ids', OLD.output_tier_ids,
                    'output_signal_ids', OLD.output_signal_ids,
                    'started_at', OLD.started_at, 'finished_at', OLD.finished_at,
                    'status', OLD.status, 'error_message', OLD.error_message,
                    'recipe_run_id', OLD.recipe_run_id),
        json_object('id', NEW.id, 'bundle_id', NEW.bundle_id, 'kind', NEW.kind,
                    'processor_id', NEW.processor_id,
                    'processor_version', NEW.processor_version,
                    'weights_checksum', NEW.weights_checksum,
                    'parameters', NEW.parameters,
                    'input_tier_ids', NEW.input_tier_ids,
                    'output_tier_ids', NEW.output_tier_ids,
                    'output_signal_ids', NEW.output_signal_ids,
                    'started_at', NEW.started_at, 'finished_at', NEW.finished_at,
                    'status', NEW.status, 'error_message', NEW.error_message,
                    'recipe_run_id', NEW.recipe_run_id)
    );
END;
CREATE TRIGGER processing_run_audit_delete AFTER DELETE ON processing_run BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'processing_run', OLD.id, 'delete',
        json_object('id', OLD.id, 'bundle_id', OLD.bundle_id, 'kind', OLD.kind,
                    'processor_id', OLD.processor_id,
                    'processor_version', OLD.processor_version,
                    'weights_checksum', OLD.weights_checksum,
                    'parameters', OLD.parameters,
                    'input_tier_ids', OLD.input_tier_ids,
                    'output_tier_ids', OLD.output_tier_ids,
                    'output_signal_ids', OLD.output_signal_ids,
                    'started_at', OLD.started_at, 'finished_at', OLD.finished_at,
                    'status', OLD.status, 'error_message', OLD.error_message,
                    'recipe_run_id', OLD.recipe_run_id),
        NULL
    );
END;
