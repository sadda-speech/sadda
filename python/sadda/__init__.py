"""sadda — open-source toolkit for phonetics and speech science research.

This package wraps the ``sadda._native`` Rust extension built by maturin,
re-exporting each public symbol with a stability decorator from
:mod:`sadda._stability` applied. End users should import from ``sadda``
directly rather than reaching into ``sadda._native``.

All Phase-0 + B1 surfaces are STABLE. Subsequent slices add new tiers
(provisional/experimental) per the 2026-05-18 Python API surface DEVLOG
entry and the 2026-05-21 A2 entry.
"""

from __future__ import annotations

from sadda import _native, clinical, criteria, dsp, ml, refdist
from sadda._stability import (
    ExperimentalAPIWarning,
    ProvisionalAPIWarning,
    SaddaWarning,
    experimental,
    provisional,
    stable,
    stable_clinical,
)

__all__ = [
    "Audio",
    "Bundle",
    "Calibration",
    "Citation",
    "Criterion",
    "DerivedSignal",
    "ExperimentalAPIWarning",
    "Instrument",
    "Interval",
    "LabelCheck",
    "Point",
    "ProcessingRun",
    "Project",
    "ProvisionalAPIWarning",
    "Reference",
    "Rubric",
    "RubricTier",
    "SaddaWarning",
    "Session",
    "Speaker",
    "StatusDef",
    "Tier",
    "VocabEntry",
    "clinical",
    "criteria",
    "dsp",
    "experimental",
    "f0",
    "load_wav",
    "ml",
    "new_project",
    "open_project",
    "provisional",
    "refdist",
    "schema_version",
    "stable",
    "stable_clinical",
    "version",
]

# Phase-0 surface.
Audio = stable(_native.Audio)
version = stable(_native.version)
load_wav = stable(_native.load_wav)
f0 = stable(_native.f0)

# B1 surface — corpus entry points.
schema_version = stable(_native.schema_version)
new_project = stable(_native.new_project)
open_project = stable(_native.open_project)
Project = stable(_native.Project)
Bundle = stable(_native.Bundle)
Speaker = stable(_native.Speaker)
Session = stable(_native.Session)

# B2 surface — tier + annotation types.
Tier = stable(_native.Tier)
Interval = stable(_native.Interval)
Point = stable(_native.Point)
Reference = stable(_native.Reference)

# B3 surface — dense-tier Parquet sidecar registration.
DerivedSignal = stable(_native.DerivedSignal)

# S1 surface (Phase 4) — annotation rubric: guidelines, the status
# vocabulary, and per-tier controlled vocabularies. New and evolving with the
# annotation suite, so provisional.
Rubric = provisional(_native.Rubric)
StatusDef = provisional(_native.StatusDef)
RubricTier = provisional(_native.RubricTier)
VocabEntry = provisional(_native.VocabEntry)
LabelCheck = provisional(_native.LabelCheck)

# S2 surface (Phase 4) — criteria engine: re-runnable rules emitting proposals.
Criterion = provisional(_native.Criterion)

# A1 surface (Phase 3) — provenance timeline + citation export.
ProcessingRun = stable(_native.ProcessingRun)
Citation = stable(_native.Citation)

# A3 surface (Phase 3) — instrument calibration + calibrated SPL.
Instrument = stable(_native.Instrument)
Calibration = stable(_native.Calibration)


# [docs:sadda.Project.query]  — monkey-patched onto _native.Project below,
# so the source-link scanner can't derive it from the PyO3 bindings.
def _project_query(self, tier_id):
    """Returns the rows of a sparse tier as a ``polars.DataFrame``. Columns
    depend on the tier's type:

    - ``interval``: id, tier_id, start_seconds, end_seconds, duration_seconds, label, parent_annotation_id, extra
    - ``point``: id, tier_id, time_seconds, label, parent_annotation_id, extra
    - ``reference``: id, tier_id, target_kind, target_id, label, parent_annotation_id, extra

    Dense tiers (continuous_numeric / continuous_vector /
    categorical_sampled) live in Parquet sidecars and arrive in B3.
    """
    import polars as pl

    tier = self.get_tier(tier_id)
    kind = tier.type
    if kind == "interval":
        rows = self.intervals(tier_id)
        return pl.DataFrame(
            {
                "id": [r.id for r in rows],
                "tier_id": [r.tier_id for r in rows],
                "start_seconds": [r.start_seconds for r in rows],
                "end_seconds": [r.end_seconds for r in rows],
                "duration_seconds": [r.duration_seconds for r in rows],
                "label": [r.label for r in rows],
                "parent_annotation_id": [r.parent_annotation_id for r in rows],
                "extra": [r.extra for r in rows],
            },
            schema={
                "id": pl.Int64,
                "tier_id": pl.Int64,
                "start_seconds": pl.Float64,
                "end_seconds": pl.Float64,
                "duration_seconds": pl.Float64,
                "label": pl.Utf8,
                "parent_annotation_id": pl.Int64,
                "extra": pl.Utf8,
            },
        )
    if kind == "point":
        rows = self.points(tier_id)
        return pl.DataFrame(
            {
                "id": [r.id for r in rows],
                "tier_id": [r.tier_id for r in rows],
                "time_seconds": [r.time_seconds for r in rows],
                "label": [r.label for r in rows],
                "parent_annotation_id": [r.parent_annotation_id for r in rows],
                "extra": [r.extra for r in rows],
            },
            schema={
                "id": pl.Int64,
                "tier_id": pl.Int64,
                "time_seconds": pl.Float64,
                "label": pl.Utf8,
                "parent_annotation_id": pl.Int64,
                "extra": pl.Utf8,
            },
        )
    if kind == "reference":
        rows = self.references_for(tier_id)
        return pl.DataFrame(
            {
                "id": [r.id for r in rows],
                "tier_id": [r.tier_id for r in rows],
                "target_kind": [r.target_kind for r in rows],
                "target_id": [r.target_id for r in rows],
                "label": [r.label for r in rows],
                "parent_annotation_id": [r.parent_annotation_id for r in rows],
                "extra": [r.extra for r in rows],
            },
            schema={
                "id": pl.Int64,
                "tier_id": pl.Int64,
                "target_kind": pl.Utf8,
                "target_id": pl.Int64,
                "label": pl.Utf8,
                "parent_annotation_id": pl.Int64,
                "extra": pl.Utf8,
            },
        )
    if kind in ("continuous_numeric", "continuous_vector", "categorical_sampled"):
        path = self.dense_path(tier_id)
        if path is None:
            raise ValueError(
                f"tier {tier_id} has no derived_signal sidecar yet; "
                f"call write_{kind}(...) first"
            )
        return pl.read_parquet(path)
    raise ValueError(f"tier {tier_id} has unknown type {kind!r}")


# Monkey-patch query onto the PyO3 class so callers can write
# `proj.query(tier_id)`. PyO3 classes are not subclass-friendly without
# extra plumbing (they're not Send+Sync); patching the method onto the
# class object is the simplest path. The type-stub file doesn't reflect
# this method yet — that's a known limitation when a Python-side method
# augments a Rust-defined class. The stub_gen workflow may grow a hook
# for this later; for now `Project.query` is documented in the module
# docstring and discoverable via help().
_native.Project.query = _project_query  # type: ignore[attr-defined]
