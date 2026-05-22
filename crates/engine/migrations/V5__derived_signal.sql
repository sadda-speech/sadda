-- V5: DerivedSignal registration table (Phase 1, slice B3).
--
-- Maps a dense tier (continuous_numeric / continuous_vector /
-- categorical_sampled) to a Parquet sidecar under
-- `signals/derived/bundle_<id>/<tier_name>.parquet`. The file holds the row
-- data; this table holds the metadata that lets the engine find + describe
-- the sidecar without opening it.
--
-- One sidecar per tier in v1 (tier_id is UNIQUE). Rewrites land in a
-- follow-up slice once a real use case appears.
--
-- Audited per the B1 trigger-rebuild discipline; any future ALTER TABLE on
-- this table must DROP+CREATE the three audit triggers.

CREATE TABLE derived_signal (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tier_id         INTEGER NOT NULL UNIQUE REFERENCES tier(id),
    relative_path   TEXT    NOT NULL,
    n_frames        INTEGER NOT NULL,
    n_dims          INTEGER NOT NULL DEFAULT 1,
    sample_rate_hz  REAL,
    dtype           TEXT    NOT NULL,
    extra           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX idx_derived_signal_tier ON derived_signal(tier_id);

-- derived_signal triggers
CREATE TRIGGER derived_signal_audit_insert AFTER INSERT ON derived_signal BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'derived_signal', NEW.id, 'insert', NULL,
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'relative_path', NEW.relative_path,
                    'n_frames', NEW.n_frames, 'n_dims', NEW.n_dims,
                    'sample_rate_hz', NEW.sample_rate_hz, 'dtype', NEW.dtype,
                    'extra', NEW.extra, 'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER derived_signal_audit_update AFTER UPDATE ON derived_signal BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'derived_signal', NEW.id, 'update',
        json_object('id', OLD.id, 'tier_id', OLD.tier_id,
                    'relative_path', OLD.relative_path,
                    'n_frames', OLD.n_frames, 'n_dims', OLD.n_dims,
                    'sample_rate_hz', OLD.sample_rate_hz, 'dtype', OLD.dtype,
                    'extra', OLD.extra, 'created_at', OLD.created_at),
        json_object('id', NEW.id, 'tier_id', NEW.tier_id,
                    'relative_path', NEW.relative_path,
                    'n_frames', NEW.n_frames, 'n_dims', NEW.n_dims,
                    'sample_rate_hz', NEW.sample_rate_hz, 'dtype', NEW.dtype,
                    'extra', NEW.extra, 'created_at', NEW.created_at)
    );
END;
CREATE TRIGGER derived_signal_audit_delete AFTER DELETE ON derived_signal BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context), 'derived_signal', OLD.id, 'delete',
        json_object('id', OLD.id, 'tier_id', OLD.tier_id,
                    'relative_path', OLD.relative_path,
                    'n_frames', OLD.n_frames, 'n_dims', OLD.n_dims,
                    'sample_rate_hz', OLD.sample_rate_hz, 'dtype', OLD.dtype,
                    'extra', OLD.extra, 'created_at', OLD.created_at),
        NULL
    );
END;
