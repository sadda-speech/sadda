-- V9: Criteria engine (Phase 4, slice S2).
--
-- A `criterion` is a re-runnable rule that selects regions of interest and
-- emits them as PROPOSALS onto a preview ("auto") tier for review; accepted
-- proposals promote to the target tier. v1 supports two `kind`s:
--   * 'structured' — a JSON rule (see engine `criteria.rs`), evaluated in the
--     engine: cross-tier interval predicates + within-interval anchors/spans;
--   * 'python'     — a Python-function body, evaluated in the python/app layer
--     (the engine stores it but does not execute it).
--
-- Design: the 2026-05-30 DEVLOG annotation-workflow entries and the S2
-- decisions (both representations; cross-tier + anchors/spans; proposals on an
-- auto tier that promote on accept). Proposals themselves are ordinary
-- annotations on the preview tier — no separate table — so they reuse the
-- existing annotation + audit infrastructure.
--
-- Audited table added: criterion (3 triggers), per the V3/B1 audit rule.

CREATE TABLE criterion (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL,
    description  TEXT,
    kind         TEXT    NOT NULL CHECK (kind IN ('structured', 'python')),
    -- JSON rule (structured) or Python source (python).
    body         TEXT    NOT NULL,
    -- Name of the tier accepted proposals promote to. The preview tier is
    -- derived as "<target_tier> (auto)".
    target_tier  TEXT    NOT NULL,
    extra        TEXT,
    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (name)
);

CREATE TRIGGER criterion_audit_insert AFTER INSERT ON criterion BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'criterion', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'name', NEW.name, 'description', NEW.description,
                    'kind', NEW.kind, 'body', NEW.body, 'target_tier', NEW.target_tier,
                    'extra', NEW.extra)
    );
END;
CREATE TRIGGER criterion_audit_update AFTER UPDATE ON criterion BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'criterion', NEW.id, 'update',
        json_object('id', OLD.id, 'name', OLD.name, 'description', OLD.description,
                    'kind', OLD.kind, 'body', OLD.body, 'target_tier', OLD.target_tier,
                    'extra', OLD.extra),
        json_object('id', NEW.id, 'name', NEW.name, 'description', NEW.description,
                    'kind', NEW.kind, 'body', NEW.body, 'target_tier', NEW.target_tier,
                    'extra', NEW.extra)
    );
END;
CREATE TRIGGER criterion_audit_delete AFTER DELETE ON criterion BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'criterion', OLD.id, 'delete',
        json_object('id', OLD.id, 'name', OLD.name, 'description', OLD.description,
                    'kind', OLD.kind, 'body', OLD.body, 'target_tier', OLD.target_tier,
                    'extra', OLD.extra),
        NULL
    );
END;
