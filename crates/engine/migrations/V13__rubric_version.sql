-- V13: rubric version snapshots (Phase 4, slice S6b — rubric versioning).
--
-- The rubric is a per-project singleton (V8) with a monotonic `version` int.
-- S6b records the rubric's *history*: `publish_rubric_version` snapshots the
-- whole rubric (statuses + per-tier config + controlled vocabularies) as JSON
-- under its current version number, so a past scheme can be listed and recalled
-- and "what changed since version V" (impact) is answerable. Annotations are NOT
-- versioned per-rubric-version (the snapshot-history approach chosen over
-- invasive per-annotation tagging); the `criterion_run` provenance already
-- records the active rubric version in its parameters (S2.5 + S6b).
--
-- The snapshot is an opaque JSON blob (shape owned by the engine's
-- `RubricSnapshot`), so the rubric scheme can evolve without a schema change.
--
-- Audited table added: rubric_version (3 triggers), per the V3/B1 audit rule.

CREATE TABLE rubric_version (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    -- The rubric.version this snapshot captures; one snapshot per version.
    version     INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    guidelines  TEXT,
    -- JSON: { statuses: [...], tiers: [{tier_name, description, closed_vocabulary, vocab:[...]}] }
    snapshot    TEXT    NOT NULL,
    note        TEXT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (version)
);

CREATE TRIGGER rubric_version_audit_insert AFTER INSERT ON rubric_version BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'rubric_version', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'version', NEW.version, 'name', NEW.name,
                    'guidelines', NEW.guidelines, 'snapshot', NEW.snapshot, 'note', NEW.note)
    );
END;
CREATE TRIGGER rubric_version_audit_update AFTER UPDATE ON rubric_version BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'rubric_version', NEW.id, 'update',
        json_object('id', OLD.id, 'version', OLD.version, 'name', OLD.name,
                    'guidelines', OLD.guidelines, 'snapshot', OLD.snapshot, 'note', OLD.note),
        json_object('id', NEW.id, 'version', NEW.version, 'name', NEW.name,
                    'guidelines', NEW.guidelines, 'snapshot', NEW.snapshot, 'note', NEW.note)
    );
END;
CREATE TRIGGER rubric_version_audit_delete AFTER DELETE ON rubric_version BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'rubric_version', OLD.id, 'delete',
        json_object('id', OLD.id, 'version', OLD.version, 'name', OLD.name,
                    'guidelines', OLD.guidelines, 'snapshot', OLD.snapshot, 'note', OLD.note),
        NULL
    );
END;
