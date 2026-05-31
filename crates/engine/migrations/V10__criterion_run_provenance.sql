-- V10: criterion-run provenance (annotation-suite slice S2.5).
--
-- Makes a criterion *run* part of the project trace — closing the gap the S2
-- design intended but the S2 implementation left open (proposals carried only
-- start/end/label, with no link back to the criterion that produced them).
-- The fix is two-fold and benefits BOTH structured and python criteria:
--
--   1. a new `processing_run.kind` value, `'criterion_run'`, so a criterion
--      execution is a first-class row in the provenance timeline (with the
--      criterion id + body checksum + rubric id in `parameters`);
--   2. a per-row `processing_run_id` FK on `annotation_interval` /
--      `annotation_point`, so "which criterion produced this annotation" is a
--      query. The link is set when proposals are materialized onto the preview
--      `"<target> (auto)"` tier and COPIED onto the promoted rows when the
--      proposals are accepted, so the trace survives promotion.
--
-- Note (status, NOT overloaded): the older S2 design sketch said an auto
-- proposal would be tagged `status='auto'`. S1 since made `status` a
-- user-defined rubric vocabulary, so "auto" can't live there. Provenance is the
-- `processing_run_id` link + the preview `(auto)` tier — not a magic status.
--
-- Note (rubric version): rubric *versioning* is deferred to S6 (singleton
-- `rubric.id=1`). The run records `rubric_id` in `parameters` now; S6's
-- versioning adds a `version` field there. Schema-ready, not over-built.
--
-- SQLite cannot alter a CHECK in place, so `processing_run` is recreated via
-- the V6 "create new, copy rows, drop old, rename" pattern; its three audit
-- triggers are recreated after the rename. Per the V3/B1 trigger-rebuild
-- discipline, the ALTER-TABLEd `annotation_interval` / `annotation_point` have
-- their three audit triggers each dropped and recreated with the new
-- `processing_run_id` column in the payload.

-- ============================================================
-- processing_run: add the 'criterion_run' kind
-- ============================================================
CREATE TABLE processing_run__v10 (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    bundle_id           INTEGER NOT NULL REFERENCES bundle(id),
    kind                TEXT    NOT NULL CHECK (
        kind IN ('ml_model', 'dsp_algorithm', 'clinical_measure', 'plugin',
                 'live_recording', 'criterion_run')
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

INSERT INTO processing_run__v10 (
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
ALTER TABLE processing_run__v10 RENAME TO processing_run;

CREATE INDEX idx_processing_run_bundle ON processing_run(bundle_id);

-- Recreate the three audit triggers (verbatim from V6, reattached to the
-- recreated table).
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

-- ============================================================
-- Per-annotation provenance link
-- ============================================================
ALTER TABLE annotation_interval ADD COLUMN processing_run_id INTEGER REFERENCES processing_run(id);
ALTER TABLE annotation_point    ADD COLUMN processing_run_id INTEGER REFERENCES processing_run(id);

-- ============================================================
-- Rebuild annotation_interval audit triggers (add processing_run_id)
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
                    'processing_run_id', NEW.processing_run_id,
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
                    'processing_run_id', OLD.processing_run_id,
                    'extra', OLD.extra),
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'start_seconds', NEW.start_seconds,
                    'end_seconds', NEW.end_seconds, 'label', NEW.label,
                    'parent_annotation_id', NEW.parent_annotation_id,
                    'status', NEW.status, 'note', NEW.note,
                    'processing_run_id', NEW.processing_run_id,
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
                    'processing_run_id', OLD.processing_run_id,
                    'extra', OLD.extra),
        NULL
    );
END;

-- ============================================================
-- Rebuild annotation_point audit triggers (add processing_run_id)
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
                    'processing_run_id', NEW.processing_run_id,
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
                    'processing_run_id', OLD.processing_run_id,
                    'extra', OLD.extra),
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'time_seconds', NEW.time_seconds, 'label', NEW.label,
                    'parent_annotation_id', NEW.parent_annotation_id,
                    'status', NEW.status, 'note', NEW.note,
                    'processing_run_id', NEW.processing_run_id,
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
                    'processing_run_id', OLD.processing_run_id,
                    'extra', OLD.extra),
        NULL
    );
END;
