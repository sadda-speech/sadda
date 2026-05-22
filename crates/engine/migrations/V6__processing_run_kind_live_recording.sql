-- V6: extend processing_run.kind CHECK constraint with 'live_recording'.
--
-- SQLite cannot alter an existing CHECK constraint in place, so we recreate
-- the table via the "create new, copy rows, drop old, rename" pattern. The
-- table's three audit triggers are dropped automatically when the original
-- table is dropped (SQLite cascades trigger drops with the table); we
-- recreate them after the rename.
--
-- See the 2026-05-22 DEVLOG entry "Live recording (E1) …" for the design
-- rationale: live_recording is a distinct event class from DSP runs / ML
-- inferences / clinical measures / plugin calls, and keeping it as its own
-- kind value makes audit queries straightforward.

CREATE TABLE processing_run__v6 (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    bundle_id           INTEGER NOT NULL REFERENCES bundle(id),
    kind                TEXT    NOT NULL CHECK (
        kind IN ('ml_model', 'dsp_algorithm', 'clinical_measure', 'plugin',
                 'live_recording')
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

INSERT INTO processing_run__v6 (
    id, bundle_id, kind, processor_id, processor_version, weights_checksum,
    parameters, input_tier_ids, output_tier_ids, output_signal_ids,
    started_at, finished_at, status, error_message, recipe_run_id
)
SELECT
    id, bundle_id, kind, processor_id, processor_version, weights_checksum,
    parameters, input_tier_ids, output_tier_ids, output_signal_ids,
    started_at, finished_at, status, error_message, recipe_run_id
FROM processing_run;

DROP TABLE processing_run;
ALTER TABLE processing_run__v6 RENAME TO processing_run;

CREATE INDEX idx_processing_run_bundle ON processing_run(bundle_id);

-- Recreate the three audit triggers (verbatim from V3). Kept here so V6 is
-- self-contained: a future contributor reading V6 in isolation sees the
-- triggers reattached to the recreated table.

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
