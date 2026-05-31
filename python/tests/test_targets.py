"""Python-surface tests for S4a campaign targets: the target object (manual +
criterion-generated) and its status lifecycle."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import pytest

import sadda


def _write_wav(path: Path, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * (sample_rate // 2))


def _project(td: Path) -> tuple[sadda.Project, int]:
    proj = sadda.new_project(td / "p", "demo")
    wav = td / "t.wav"
    _write_wav(wav)
    bundle_id = proj.add_bundle("b", wav)
    phones = proj.add_tier(bundle_id, "phones", "interval")
    for s, e, lbl in [(0.0, 0.1, "a"), (0.1, 0.2, "b"), (0.5, 0.6, "a")]:
        proj.add_interval(phones, s, e, label=lbl)
    return proj, bundle_id


def test_manual_target_crud_and_status_lifecycle() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))

        tid = proj.add_target(bundle_id, 0.2, 0.5, "phones")
        t = proj.get_target(tid)
        assert t is not None
        assert t.status == "unassigned"
        assert t.source == "manual"
        assert t.target_type == "phones"
        assert t.criterion_id is None

        # Time-ordered listing.
        proj.add_target(bundle_id, 0.0, 0.1, "phones")
        listed = proj.targets(bundle_id)
        assert [round(x.start_seconds, 3) for x in listed] == [0.0, 0.2]

        # Lifecycle + note; invalid status + RoI rejected.
        proj.update_target_status(tid, "in_progress")
        proj.set_target_note(tid, "ambiguous vs rubric")
        t = proj.get_target(tid)
        assert t.status == "in_progress"
        assert t.note == "ambiguous vs rubric"
        with pytest.raises((ValueError, RuntimeError)):
            proj.update_target_status(tid, "bogus")
        with pytest.raises((ValueError, RuntimeError)):
            proj.add_target(bundle_id, 0.5, 0.5, "phones")

        # Idempotent delete.
        proj.delete_target(tid)
        proj.delete_target(tid)
        assert len(proj.targets(bundle_id)) == 1


def test_generate_targets_from_criterion() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))
        body = '{"select": {"tier": "phones", "label_any": ["a"]}, "emit": {"kind": "span"}}'
        crit = proj.set_criterion("vowels", "structured", body, "vowel-detail")

        n = proj.generate_targets_from_criterion(crit.id, bundle_id)
        assert n == 2  # the two "a" RoIs
        ts = proj.targets(bundle_id)
        assert len(ts) == 2
        assert all(
            t.source == "criterion"
            and t.criterion_id == crit.id
            and t.target_type == "vowel-detail"
            and t.status == "unassigned"
            for t in ts
        )
        assert round(ts[0].start_seconds, 3) == 0.0
        assert round(ts[1].start_seconds, 3) == 0.5

        # Regeneration replaces (not appends).
        assert proj.generate_targets_from_criterion(crit.id, bundle_id) == 2
        assert len(proj.targets(bundle_id)) == 2

        # A python criterion cannot generate targets in the engine.
        py = proj.set_criterion("py", "python", "def c(): pass", "x")
        with pytest.raises((ValueError, RuntimeError)):
            proj.generate_targets_from_criterion(py.id, bundle_id)


def test_target_is_provisional() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.Target) == "provisional"
