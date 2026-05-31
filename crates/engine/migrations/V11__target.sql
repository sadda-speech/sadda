-- V11: the `target` object (Phase 4, slice S4a — campaign layer foundation).
--
-- A `target` is the first-class UNIT OF WORK in an annotation campaign:
-- `target = (file/bundle, region-of-interest, target-type, status)`. The PI's
-- criteria engine GENERATES targets from its RoI selection (`source='criterion'`,
-- with a `criterion_id` back-link), or they are hand-marked (`source='manual'`).
-- The S4b assignment layer then distributes targets across annotators; the QA
-- dashboard reads completeness straight off the `status` column.
--
-- Design: the 2026-05-30 DEVLOG "annotation campaign management" entry
-- (targets are first-class; status lifecycle; criteria *generate* targets, the
-- assignment layer *distributes* them) + the S4 decomposition entry. This slice
-- (S4a) builds only the target object + its generator; `assignment` and the
-- export/merge package are S4b/S4c.
--
-- A target is a temporal region on a bundle (not tied to a tier): the RoI is
-- `[start_seconds, end_seconds)`, and `target_type` names what kind of work is
-- expected there (in practice the criterion's target tier). Keeping it
-- tier-free matches the design's `(file, RoI, type, status)` shape and lets a
-- target outlive any particular tier.
--
-- Audited table added: target (3 triggers), per the V3/B1 audit rule.

CREATE TABLE target (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    bundle_id     INTEGER NOT NULL REFERENCES bundle(id),
    -- Region of interest on the bundle's audio, in seconds.
    start_seconds REAL    NOT NULL,
    end_seconds   REAL    NOT NULL CHECK (end_seconds > start_seconds),
    -- What kind of annotation work this region needs (usually a tier name).
    target_type   TEXT    NOT NULL,
    -- Work lifecycle. 'flagged' is the ambiguous-vs-rubric escape hatch.
    status        TEXT    NOT NULL DEFAULT 'unassigned' CHECK (
        status IN ('unassigned', 'assigned', 'in_progress', 'done', 'flagged')
    ),
    -- How the target came to exist.
    source        TEXT    NOT NULL DEFAULT 'manual' CHECK (
        source IN ('manual', 'criterion')
    ),
    -- The generating criterion when source='criterion'; NULL for manual targets.
    criterion_id  INTEGER REFERENCES criterion(id),
    note          TEXT,
    extra         TEXT,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX idx_target_bundle ON target(bundle_id);
CREATE INDEX idx_target_criterion ON target(criterion_id);

CREATE TRIGGER target_audit_insert AFTER INSERT ON target BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'target', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'bundle_id', NEW.bundle_id,
                    'start_seconds', NEW.start_seconds, 'end_seconds', NEW.end_seconds,
                    'target_type', NEW.target_type, 'status', NEW.status,
                    'source', NEW.source, 'criterion_id', NEW.criterion_id,
                    'note', NEW.note, 'extra', NEW.extra)
    );
END;
CREATE TRIGGER target_audit_update AFTER UPDATE ON target BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'target', NEW.id, 'update',
        json_object('id', OLD.id, 'bundle_id', OLD.bundle_id,
                    'start_seconds', OLD.start_seconds, 'end_seconds', OLD.end_seconds,
                    'target_type', OLD.target_type, 'status', OLD.status,
                    'source', OLD.source, 'criterion_id', OLD.criterion_id,
                    'note', OLD.note, 'extra', OLD.extra),
        json_object('id', NEW.id, 'bundle_id', NEW.bundle_id,
                    'start_seconds', NEW.start_seconds, 'end_seconds', NEW.end_seconds,
                    'target_type', NEW.target_type, 'status', NEW.status,
                    'source', NEW.source, 'criterion_id', NEW.criterion_id,
                    'note', NEW.note, 'extra', NEW.extra)
    );
END;
CREATE TRIGGER target_audit_delete AFTER DELETE ON target BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'target', OLD.id, 'delete',
        json_object('id', OLD.id, 'bundle_id', OLD.bundle_id,
                    'start_seconds', OLD.start_seconds, 'end_seconds', OLD.end_seconds,
                    'target_type', OLD.target_type, 'status', OLD.status,
                    'source', OLD.source, 'criterion_id', OLD.criterion_id,
                    'note', OLD.note, 'extra', OLD.extra),
        NULL
    );
END;
