-- V8: Annotation rubric, controlled vocabulary, and annotation status
-- (Phase 4, slice S1 of the annotation workflow).
--
-- Makes the annotation *rubric* (the scheme: guidelines + the allowed
-- status vocabulary + per-tier-name controlled vocabularies) and each
-- annotation's *status*/*note* first-class, queryable data instead of prose
-- in an external guidelines document. Design: the 2026-05-30 DEVLOG entry
-- "Design: the annotation workflow (rubric-as-data + computational
-- criteria)" plus the S1 decisions:
--   * one rubric per project (the `rubric` row's id is pinned to 1);
--   * status values are an ARBITRARY, rubric-defined set of strings (not a
--     fixed enum) — so `status` cannot be a SQL CHECK and is validated in
--     Rust against `rubric_status`; any annotation (any status, incl. none)
--     may carry a free-text `note`;
--   * controlled vocabularies are keyed by tier NAME, with a per-tier-name
--     open/closed flag (open accepts + soft-flags out-of-vocab labels;
--     closed rejects them at entry, enforced in Rust).
--
-- Trigger-rebuild discipline (per the V3/B1 rule): this migration
-- ALTER-TABLEs the audited `annotation_interval` and `annotation_point`
-- tables, so it DROPs and recreates their three audit triggers each with the
-- new `status`/`note` columns in the payload.
--
-- Audited tables added by this migration: rubric, rubric_status,
-- rubric_tier, controlled_vocabulary (3 triggers each). The same
-- trigger-rebuild discipline applies to any future ALTER TABLE on them.

-- ============================================================
-- Rubric tables
-- ============================================================

-- One rubric per project: the id is pinned to 1 by a CHECK, so the table is
-- a singleton the engine upserts.
CREATE TABLE rubric (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    name        TEXT    NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    guidelines  TEXT,
    extra       TEXT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- The arbitrary set of annotation-status strings the rubric defines.
CREATE TABLE rubric_status (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    rubric_id   INTEGER NOT NULL REFERENCES rubric(id),
    value       TEXT    NOT NULL,
    description TEXT,
    sort_order  INTEGER NOT NULL DEFAULT 0,
    UNIQUE (rubric_id, value)
);

-- Per-tier-name rubric configuration: guidelines + whether its controlled
-- vocabulary is closed (rejects out-of-vocab labels) or open (default).
CREATE TABLE rubric_tier (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    rubric_id         INTEGER NOT NULL REFERENCES rubric(id),
    tier_name         TEXT    NOT NULL,
    description       TEXT,
    closed_vocabulary INTEGER NOT NULL DEFAULT 0 CHECK (closed_vocabulary IN (0, 1)),
    UNIQUE (rubric_id, tier_name)
);

-- Controlled-vocabulary entries (allowed labels) for a rubric_tier.
CREATE TABLE controlled_vocabulary (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    rubric_tier_id INTEGER NOT NULL REFERENCES rubric_tier(id),
    value          TEXT    NOT NULL,
    description    TEXT,
    sort_order     INTEGER NOT NULL DEFAULT 0,
    UNIQUE (rubric_tier_id, value)
);
CREATE INDEX idx_controlled_vocabulary_tier
    ON controlled_vocabulary(rubric_tier_id);

-- ============================================================
-- Annotation status + note (first-class columns)
-- ============================================================
ALTER TABLE annotation_interval ADD COLUMN status TEXT;
ALTER TABLE annotation_interval ADD COLUMN note   TEXT;
ALTER TABLE annotation_point    ADD COLUMN status TEXT;
ALTER TABLE annotation_point    ADD COLUMN note   TEXT;

-- ============================================================
-- Rebuild annotation_interval audit triggers (add status, note)
-- ============================================================
DROP TRIGGER IF EXISTS annotation_interval_audit_insert;
DROP TRIGGER IF EXISTS annotation_interval_audit_update;
DROP TRIGGER IF EXISTS annotation_interval_audit_delete;

CREATE TRIGGER annotation_interval_audit_insert AFTER INSERT ON annotation_interval BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'annotation_interval', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'start_seconds', NEW.start_seconds,
                    'end_seconds', NEW.end_seconds, 'label', NEW.label,
                    'parent_annotation_id', NEW.parent_annotation_id,
                    'status', NEW.status, 'note', NEW.note,
                    'extra', NEW.extra)
    );
END;
CREATE TRIGGER annotation_interval_audit_update AFTER UPDATE ON annotation_interval BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'annotation_interval', NEW.id, 'update',
        json_object('id', OLD.id, 'tier_id', OLD.tier_id,
                    'start_seconds', OLD.start_seconds,
                    'end_seconds', OLD.end_seconds, 'label', OLD.label,
                    'parent_annotation_id', OLD.parent_annotation_id,
                    'status', OLD.status, 'note', OLD.note,
                    'extra', OLD.extra),
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'start_seconds', NEW.start_seconds,
                    'end_seconds', NEW.end_seconds, 'label', NEW.label,
                    'parent_annotation_id', NEW.parent_annotation_id,
                    'status', NEW.status, 'note', NEW.note,
                    'extra', NEW.extra)
    );
END;
CREATE TRIGGER annotation_interval_audit_delete AFTER DELETE ON annotation_interval BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'annotation_interval', OLD.id, 'delete',
        json_object('id', OLD.id, 'tier_id', OLD.tier_id,
                    'start_seconds', OLD.start_seconds,
                    'end_seconds', OLD.end_seconds, 'label', OLD.label,
                    'parent_annotation_id', OLD.parent_annotation_id,
                    'status', OLD.status, 'note', OLD.note,
                    'extra', OLD.extra),
        NULL
    );
END;

-- ============================================================
-- Rebuild annotation_point audit triggers (add status, note)
-- ============================================================
DROP TRIGGER IF EXISTS annotation_point_audit_insert;
DROP TRIGGER IF EXISTS annotation_point_audit_update;
DROP TRIGGER IF EXISTS annotation_point_audit_delete;

CREATE TRIGGER annotation_point_audit_insert AFTER INSERT ON annotation_point BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'annotation_point', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'time_seconds', NEW.time_seconds, 'label', NEW.label,
                    'parent_annotation_id', NEW.parent_annotation_id,
                    'status', NEW.status, 'note', NEW.note,
                    'extra', NEW.extra)
    );
END;
CREATE TRIGGER annotation_point_audit_update AFTER UPDATE ON annotation_point BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'annotation_point', NEW.id, 'update',
        json_object('id', OLD.id, 'tier_id', OLD.tier_id,
                    'time_seconds', OLD.time_seconds, 'label', OLD.label,
                    'parent_annotation_id', OLD.parent_annotation_id,
                    'status', OLD.status, 'note', OLD.note,
                    'extra', OLD.extra),
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'time_seconds', NEW.time_seconds, 'label', NEW.label,
                    'parent_annotation_id', NEW.parent_annotation_id,
                    'status', NEW.status, 'note', NEW.note,
                    'extra', NEW.extra)
    );
END;
CREATE TRIGGER annotation_point_audit_delete AFTER DELETE ON annotation_point BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'annotation_point', OLD.id, 'delete',
        json_object('id', OLD.id, 'tier_id', OLD.tier_id,
                    'time_seconds', OLD.time_seconds, 'label', OLD.label,
                    'parent_annotation_id', OLD.parent_annotation_id,
                    'status', OLD.status, 'note', OLD.note,
                    'extra', OLD.extra),
        NULL
    );
END;

-- ============================================================
-- Audit triggers for the new rubric tables (provenance of rubric edits)
-- ============================================================

-- rubric
CREATE TRIGGER rubric_audit_insert AFTER INSERT ON rubric BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'rubric', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'name', NEW.name, 'version', NEW.version,
                    'guidelines', NEW.guidelines, 'extra', NEW.extra)
    );
END;
CREATE TRIGGER rubric_audit_update AFTER UPDATE ON rubric BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'rubric', NEW.id, 'update',
        json_object('id', OLD.id, 'name', OLD.name, 'version', OLD.version,
                    'guidelines', OLD.guidelines, 'extra', OLD.extra),
        json_object('id', NEW.id, 'name', NEW.name, 'version', NEW.version,
                    'guidelines', NEW.guidelines, 'extra', NEW.extra)
    );
END;
CREATE TRIGGER rubric_audit_delete AFTER DELETE ON rubric BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'rubric', OLD.id, 'delete',
        json_object('id', OLD.id, 'name', OLD.name, 'version', OLD.version,
                    'guidelines', OLD.guidelines, 'extra', OLD.extra),
        NULL
    );
END;

-- rubric_status
CREATE TRIGGER rubric_status_audit_insert AFTER INSERT ON rubric_status BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'rubric_status', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'rubric_id', NEW.rubric_id, 'value', NEW.value,
                    'description', NEW.description, 'sort_order', NEW.sort_order)
    );
END;
CREATE TRIGGER rubric_status_audit_update AFTER UPDATE ON rubric_status BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'rubric_status', NEW.id, 'update',
        json_object('id', OLD.id, 'rubric_id', OLD.rubric_id, 'value', OLD.value,
                    'description', OLD.description, 'sort_order', OLD.sort_order),
        json_object('id', NEW.id, 'rubric_id', NEW.rubric_id, 'value', NEW.value,
                    'description', NEW.description, 'sort_order', NEW.sort_order)
    );
END;
CREATE TRIGGER rubric_status_audit_delete AFTER DELETE ON rubric_status BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'rubric_status', OLD.id, 'delete',
        json_object('id', OLD.id, 'rubric_id', OLD.rubric_id, 'value', OLD.value,
                    'description', OLD.description, 'sort_order', OLD.sort_order),
        NULL
    );
END;

-- rubric_tier
CREATE TRIGGER rubric_tier_audit_insert AFTER INSERT ON rubric_tier BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'rubric_tier', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'rubric_id', NEW.rubric_id, 'tier_name', NEW.tier_name,
                    'description', NEW.description, 'closed_vocabulary', NEW.closed_vocabulary)
    );
END;
CREATE TRIGGER rubric_tier_audit_update AFTER UPDATE ON rubric_tier BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'rubric_tier', NEW.id, 'update',
        json_object('id', OLD.id, 'rubric_id', OLD.rubric_id, 'tier_name', OLD.tier_name,
                    'description', OLD.description, 'closed_vocabulary', OLD.closed_vocabulary),
        json_object('id', NEW.id, 'rubric_id', NEW.rubric_id, 'tier_name', NEW.tier_name,
                    'description', NEW.description, 'closed_vocabulary', NEW.closed_vocabulary)
    );
END;
CREATE TRIGGER rubric_tier_audit_delete AFTER DELETE ON rubric_tier BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'rubric_tier', OLD.id, 'delete',
        json_object('id', OLD.id, 'rubric_id', OLD.rubric_id, 'tier_name', OLD.tier_name,
                    'description', OLD.description, 'closed_vocabulary', OLD.closed_vocabulary),
        NULL
    );
END;

-- controlled_vocabulary
CREATE TRIGGER controlled_vocabulary_audit_insert AFTER INSERT ON controlled_vocabulary BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'controlled_vocabulary', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'rubric_tier_id', NEW.rubric_tier_id, 'value', NEW.value,
                    'description', NEW.description, 'sort_order', NEW.sort_order)
    );
END;
CREATE TRIGGER controlled_vocabulary_audit_update AFTER UPDATE ON controlled_vocabulary BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'controlled_vocabulary', NEW.id, 'update',
        json_object('id', OLD.id, 'rubric_tier_id', OLD.rubric_tier_id, 'value', OLD.value,
                    'description', OLD.description, 'sort_order', OLD.sort_order),
        json_object('id', NEW.id, 'rubric_tier_id', NEW.rubric_tier_id, 'value', NEW.value,
                    'description', NEW.description, 'sort_order', NEW.sort_order)
    );
END;
CREATE TRIGGER controlled_vocabulary_audit_delete AFTER DELETE ON controlled_vocabulary BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'controlled_vocabulary', OLD.id, 'delete',
        json_object('id', OLD.id, 'rubric_tier_id', OLD.rubric_tier_id, 'value', OLD.value,
                    'description', OLD.description, 'sort_order', OLD.sort_order),
        NULL
    );
END;
