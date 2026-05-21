-- V4: Sparse-tier annotation rows (Phase 1, slice B2).
--
-- Adds the three sparse-tier annotation tables (interval / point /
-- reference), the indexes that support common queries, and the audit
-- triggers (3 per table × 3 tables = 9 triggers) per the B1 audit rule.
--
-- Parent-child cardinality is NOT enforced in SQL; the engine validates at
-- insert time in Rust because the right parent annotation table depends on
-- the parent tier's `type`, which a trigger can't dispatch on cleanly. See
-- the 2026-05-21 DEVLOG entry "Sparse tier types (B2)".
--
-- Audited tables added by this migration: annotation_interval,
-- annotation_point, annotation_reference. Trigger-rebuild discipline (per
-- the B1 entry) applies to any future ALTER TABLE on these tables.

CREATE TABLE annotation_interval (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    tier_id              INTEGER NOT NULL REFERENCES tier(id),
    start_seconds        REAL    NOT NULL,
    end_seconds          REAL    NOT NULL,
    label                TEXT,
    parent_annotation_id INTEGER,
    extra                TEXT,
    CHECK (end_seconds > start_seconds)
);
CREATE INDEX idx_annotation_interval_tier
    ON annotation_interval(tier_id);
CREATE INDEX idx_annotation_interval_parent
    ON annotation_interval(parent_annotation_id);

CREATE TABLE annotation_point (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    tier_id              INTEGER NOT NULL REFERENCES tier(id),
    time_seconds         REAL    NOT NULL,
    label                TEXT,
    parent_annotation_id INTEGER,
    extra                TEXT
);
CREATE INDEX idx_annotation_point_tier
    ON annotation_point(tier_id);
CREATE INDEX idx_annotation_point_parent
    ON annotation_point(parent_annotation_id);

CREATE TABLE annotation_reference (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    tier_id              INTEGER NOT NULL REFERENCES tier(id),
    target_kind          TEXT    NOT NULL CHECK (
        target_kind IN ('bundle', 'session', 'speaker', 'tier', 'annotation')
    ),
    target_id            INTEGER NOT NULL,
    label                TEXT,
    parent_annotation_id INTEGER,
    extra                TEXT
);
CREATE INDEX idx_annotation_reference_tier
    ON annotation_reference(tier_id);
CREATE INDEX idx_annotation_reference_parent
    ON annotation_reference(parent_annotation_id);
CREATE INDEX idx_annotation_reference_target
    ON annotation_reference(target_kind, target_id);

-- annotation_interval triggers
CREATE TRIGGER annotation_interval_audit_insert AFTER INSERT ON annotation_interval BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'annotation_interval', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'start_seconds', NEW.start_seconds,
                    'end_seconds', NEW.end_seconds, 'label', NEW.label,
                    'parent_annotation_id', NEW.parent_annotation_id,
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
                    'extra', OLD.extra),
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'start_seconds', NEW.start_seconds,
                    'end_seconds', NEW.end_seconds, 'label', NEW.label,
                    'parent_annotation_id', NEW.parent_annotation_id,
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
                    'extra', OLD.extra),
        NULL
    );
END;

-- annotation_point triggers
CREATE TRIGGER annotation_point_audit_insert AFTER INSERT ON annotation_point BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'annotation_point', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'time_seconds', NEW.time_seconds, 'label', NEW.label,
                    'parent_annotation_id', NEW.parent_annotation_id,
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
                    'extra', OLD.extra),
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'time_seconds', NEW.time_seconds, 'label', NEW.label,
                    'parent_annotation_id', NEW.parent_annotation_id,
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
                    'extra', OLD.extra),
        NULL
    );
END;

-- annotation_reference triggers
CREATE TRIGGER annotation_reference_audit_insert AFTER INSERT ON annotation_reference BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'annotation_reference', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'target_kind', NEW.target_kind, 'target_id', NEW.target_id,
                    'label', NEW.label,
                    'parent_annotation_id', NEW.parent_annotation_id,
                    'extra', NEW.extra)
    );
END;
CREATE TRIGGER annotation_reference_audit_update AFTER UPDATE ON annotation_reference BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'annotation_reference', NEW.id, 'update',
        json_object('id', OLD.id, 'tier_id', OLD.tier_id,
                    'target_kind', OLD.target_kind, 'target_id', OLD.target_id,
                    'label', OLD.label,
                    'parent_annotation_id', OLD.parent_annotation_id,
                    'extra', OLD.extra),
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'target_kind', NEW.target_kind, 'target_id', NEW.target_id,
                    'label', NEW.label,
                    'parent_annotation_id', NEW.parent_annotation_id,
                    'extra', NEW.extra)
    );
END;
CREATE TRIGGER annotation_reference_audit_delete AFTER DELETE ON annotation_reference BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'annotation_reference', OLD.id, 'delete',
        json_object('id', OLD.id, 'tier_id', OLD.tier_id,
                    'target_kind', OLD.target_kind, 'target_id', OLD.target_id,
                    'label', OLD.label,
                    'parent_annotation_id', OLD.parent_annotation_id,
                    'extra', OLD.extra),
        NULL
    );
END;
