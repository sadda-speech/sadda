"""Python-surface tests for S4b campaign assignments: the assignment object,
its lifecycle/editability, and seeded random distribution across a roster."""

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


def _project_with_targets(td: Path, n: int) -> tuple[sadda.Project, int, list[int]]:
    proj = sadda.new_project(td / "p", "demo")
    wav = td / "t.wav"
    _write_wav(wav)
    bundle_id = proj.add_bundle("b", wav)
    ids = [proj.add_target(bundle_id, i * 0.1, i * 0.1 + 0.05, "phones") for i in range(n)]
    return proj, bundle_id, ids


def test_assignment_crud_and_target_status() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id, ids = _project_with_targets(Path(td), 1)
        tid = ids[0]

        aid = proj.add_assignment(tid, "alice")
        assert proj.get_target(tid).status == "assigned"
        a = proj.assignments(bundle_id)[0]
        assert a.annotator == "alice" and a.role == "primary" and a.status == "assigned"
        assert a.seed is None

        proj.add_assignment(tid, "bob", role="secondary")
        assert len(proj.assignments_for_target(tid)) == 2
        # Duplicate annotator + bad role rejected.
        with pytest.raises((ValueError, RuntimeError)):
            proj.add_assignment(tid, "alice")
        with pytest.raises((ValueError, RuntimeError)):
            proj.add_assignment(tid, "x", role="lead")

        # Editable.
        proj.update_assignment_status(aid, "in_progress")
        proj.set_assignment_annotator(aid, "carol")
        assert proj.assignments_for_target(tid)[0].annotator == "carol"
        with pytest.raises((ValueError, RuntimeError)):
            proj.update_assignment_status(aid, "bogus")

        # Removing the last assignment reverts the target to unassigned.
        for a in proj.assignments_for_target(tid):
            proj.delete_assignment(a.id)
        assert proj.get_target(tid).status == "unassigned"


def test_assign_targets_randomly_is_deterministic_and_remainder_only() -> None:
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        roster = ["alice", "bob", "carol"]

        p1, b1, _ = _project_with_targets(Path(td1), 10)
        assert p1.assign_targets_randomly(b1, roster, 42) == 10
        a1 = [(a.target_id, a.annotator) for a in p1.assignments(b1)]
        assert all(a.seed == 42 for a in p1.assignments(b1))
        assert all(t.status == "assigned" for t in p1.targets(b1))
        # No remaining unassigned targets → a second call assigns nothing.
        assert p1.assign_targets_randomly(b1, roster, 7) == 0
        with pytest.raises((ValueError, RuntimeError)):
            p1.assign_targets_randomly(b1, [], 1)

        # Same seed on an identical project reproduces the exact mapping.
        p2, b2, _ = _project_with_targets(Path(td2), 10)
        p2.assign_targets_randomly(b2, roster, 42)
        a2 = [(a.target_id, a.annotator) for a in p2.assignments(b2)]
        assert a1 == a2


def test_assignment_is_provisional() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.Assignment) == "provisional"
