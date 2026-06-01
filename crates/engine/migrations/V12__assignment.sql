-- V12: the `assignment` object (Phase 4, slice S4b — campaign layer).
--
-- An `assignment` distributes a campaign `target` (V11) to an annotator. It is
-- a DEDICATED first-class object, separate from annotation data and the rubric,
-- so roster churn never bumps the rubric and the QA dashboard can query "who
-- has what" directly. A target may carry N assignments (overlap → S5 agreement);
-- each has its own `role` (primary / secondary) and per-annotator progress
-- `status`. Editable throughout.
--
-- `seed` records the random-assignment seed when a target was batch-assigned by
-- `assign_targets_randomly` (NULL for hand assignment) — reproducibility,
-- consistent with the no-`Math.random` ethos: the shuffle is a deterministic
-- seeded Fisher–Yates in the engine, and the seed is stored so a run is
-- recoverable.
--
-- Design: the 2026-05-30 "annotation campaign management" entry (assignment is
-- first-class + editable; role primary/secondary; seeded random-assign +
-- re-randomize-of-remaining) + the S4 decomposition entry. This slice (S4b)
-- builds the assignment object + its seeded distributor; per-annotator export
-- packages + import/merge are S4c.
--
-- UNIQUE(target_id, annotator): a given annotator is assigned a target at most
-- once. Audited table added: assignment (3 triggers), per the V3/B1 audit rule.

CREATE TABLE assignment (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id    INTEGER NOT NULL REFERENCES target(id),
    annotator    TEXT    NOT NULL,
    role         TEXT    NOT NULL DEFAULT 'primary' CHECK (
        role IN ('primary', 'secondary')
    ),
    status       TEXT    NOT NULL DEFAULT 'assigned' CHECK (
        status IN ('assigned', 'in_progress', 'done')
    ),
    -- The `assign_targets_randomly` seed when batch-assigned; NULL for manual.
    seed         INTEGER,
    extra        TEXT,
    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (target_id, annotator)
);

CREATE INDEX idx_assignment_target ON assignment(target_id);

CREATE TRIGGER assignment_audit_insert AFTER INSERT ON assignment BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'assignment', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'target_id', NEW.target_id, 'annotator', NEW.annotator,
                    'role', NEW.role, 'status', NEW.status, 'seed', NEW.seed,
                    'extra', NEW.extra)
    );
END;
CREATE TRIGGER assignment_audit_update AFTER UPDATE ON assignment BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'assignment', NEW.id, 'update',
        json_object('id', OLD.id, 'target_id', OLD.target_id, 'annotator', OLD.annotator,
                    'role', OLD.role, 'status', OLD.status, 'seed', OLD.seed,
                    'extra', OLD.extra),
        json_object('id', NEW.id, 'target_id', NEW.target_id, 'annotator', NEW.annotator,
                    'role', NEW.role, 'status', NEW.status, 'seed', NEW.seed,
                    'extra', NEW.extra)
    );
END;
CREATE TRIGGER assignment_audit_delete AFTER DELETE ON assignment BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'assignment', OLD.id, 'delete',
        json_object('id', OLD.id, 'target_id', OLD.target_id, 'annotator', OLD.annotator,
                    'role', OLD.role, 'status', OLD.status, 'seed', OLD.seed,
                    'extra', OLD.extra),
        NULL
    );
END;
