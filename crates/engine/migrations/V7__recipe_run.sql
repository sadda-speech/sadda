-- V7: recipe_run table for the F1 reproducibility slice. See the
-- 2026-05-22 "Recipes (F1)" DEVLOG entry for design rationale.
--
-- recipe_run rows are FK'd from processing_run.recipe_run_id (the
-- column already exists in V3) so any processing_run insert inside a
-- recipe block links back. Audit triggers mirror the V3 pattern.

CREATE TABLE recipe_run (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    NOT NULL,
    sadda_version   TEXT    NOT NULL,
    parameters      TEXT,
    started_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at    TEXT,
    status          TEXT    NOT NULL DEFAULT 'in_progress' CHECK (
        status IN ('in_progress', 'ok', 'error')
    ),
    error_message   TEXT,
    UNIQUE (name)
);
CREATE INDEX idx_recipe_run_name ON recipe_run(name);

CREATE TRIGGER recipe_run_audit_insert AFTER INSERT ON recipe_run BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'recipe_run', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'name', NEW.name,
                    'sadda_version', NEW.sadda_version,
                    'parameters', NEW.parameters,
                    'started_at', NEW.started_at,
                    'completed_at', NEW.completed_at,
                    'status', NEW.status,
                    'error_message', NEW.error_message)
    );
END;
CREATE TRIGGER recipe_run_audit_update AFTER UPDATE ON recipe_run BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'recipe_run', NEW.id, 'update',
        json_object('id', OLD.id, 'name', OLD.name,
                    'sadda_version', OLD.sadda_version,
                    'parameters', OLD.parameters,
                    'started_at', OLD.started_at,
                    'completed_at', OLD.completed_at,
                    'status', OLD.status,
                    'error_message', OLD.error_message),
        json_object('id', NEW.id, 'name', NEW.name,
                    'sadda_version', NEW.sadda_version,
                    'parameters', NEW.parameters,
                    'started_at', NEW.started_at,
                    'completed_at', NEW.completed_at,
                    'status', NEW.status,
                    'error_message', NEW.error_message)
    );
END;
CREATE TRIGGER recipe_run_audit_delete AFTER DELETE ON recipe_run BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'recipe_run', OLD.id, 'delete',
        json_object('id', OLD.id, 'name', OLD.name,
                    'sadda_version', OLD.sadda_version,
                    'parameters', OLD.parameters,
                    'started_at', OLD.started_at,
                    'completed_at', OLD.completed_at,
                    'status', OLD.status,
                    'error_message', OLD.error_message),
        NULL
    );
END;
