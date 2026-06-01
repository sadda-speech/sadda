-- V14: the PI lab-notebook (Phase 4, slice S7 — the final annotation-suite slice).
--
-- As the PI explores a corpus to define a study, they jot observations,
-- measurements, and decisions — grouped by `target_type` (the kind of thing the
-- note is about, e.g. "vowels"). A note can then be PROMOTED into a rubric
-- artifact: a `criterion` (the computational rule) or rubric-tier guidance (the
-- prose rule). Promotion stamps `promoted_kind` / `promoted_ref` on the entry,
-- so the rubric's own creation is provenance — "this rule came from that
-- observation". Same iterate-loop the annotators use later, run earlier by the
-- PI; ties to recipes (which record measurement actions).
--
-- `measurement` optionally records the measurement action behind the note (a
-- free record at v1 — e.g. a signal expression or a measured value); deeper
-- recipe integration is a later slice.
--
-- Audited table added: notebook_entry (3 triggers), per the V3/B1 audit rule.

CREATE TABLE notebook_entry (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    -- What kind of thing the note is about (free text; usually a tier name).
    target_type   TEXT    NOT NULL,
    -- The note's nature.
    kind          TEXT    NOT NULL DEFAULT 'observation' CHECK (
        kind IN ('observation', 'measurement', 'decision')
    ),
    text          TEXT    NOT NULL,
    -- Optional recorded measurement action / result behind the note.
    measurement   TEXT,
    -- Optional context: the bundle that prompted the note.
    bundle_id     INTEGER REFERENCES bundle(id),
    -- Set when the note is promoted: what it became + a reference to it.
    promoted_kind TEXT CHECK (promoted_kind IN ('criterion', 'rubric_guidance')),
    promoted_ref  TEXT,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_notebook_entry_target_type ON notebook_entry(target_type);

CREATE TRIGGER notebook_entry_audit_insert AFTER INSERT ON notebook_entry BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'notebook_entry', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'target_type', NEW.target_type, 'kind', NEW.kind,
                    'text', NEW.text, 'measurement', NEW.measurement,
                    'bundle_id', NEW.bundle_id, 'promoted_kind', NEW.promoted_kind,
                    'promoted_ref', NEW.promoted_ref)
    );
END;
CREATE TRIGGER notebook_entry_audit_update AFTER UPDATE ON notebook_entry BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'notebook_entry', NEW.id, 'update',
        json_object('id', OLD.id, 'target_type', OLD.target_type, 'kind', OLD.kind,
                    'text', OLD.text, 'measurement', OLD.measurement,
                    'bundle_id', OLD.bundle_id, 'promoted_kind', OLD.promoted_kind,
                    'promoted_ref', OLD.promoted_ref),
        json_object('id', NEW.id, 'target_type', NEW.target_type, 'kind', NEW.kind,
                    'text', NEW.text, 'measurement', NEW.measurement,
                    'bundle_id', NEW.bundle_id, 'promoted_kind', NEW.promoted_kind,
                    'promoted_ref', NEW.promoted_ref)
    );
END;
CREATE TRIGGER notebook_entry_audit_delete AFTER DELETE ON notebook_entry BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'notebook_entry', OLD.id, 'delete',
        json_object('id', OLD.id, 'target_type', OLD.target_type, 'kind', OLD.kind,
                    'text', OLD.text, 'measurement', OLD.measurement,
                    'bundle_id', OLD.bundle_id, 'promoted_kind', OLD.promoted_kind,
                    'promoted_ref', OLD.promoted_ref),
        NULL
    );
END;
