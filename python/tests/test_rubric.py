"""Python-surface tests for the S1 annotation-rubric API: the per-project
rubric, its status vocabulary, per-tier controlled vocabularies (open/closed),
and the status/note columns on interval annotations."""

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
        w.writeframes(b"\x00\x00" * (sample_rate // 4))


def _project_with_tier(td: Path) -> tuple[sadda.Project, int]:
    proj = sadda.new_project(td / "p", "demo")
    wav = td / "t.wav"
    _write_wav(wav)
    bundle_id = proj.add_bundle("b", wav)
    tier_id = proj.add_tier(bundle_id, "phones", "interval")
    return proj, tier_id


def test_rubric_singleton_and_statuses() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, _tier = _project_with_tier(Path(td))

        assert proj.rubric() is None
        r = proj.set_rubric("IPA segmentation", guidelines="Label each phone.")
        assert r.id == 1
        assert r.name == "IPA segmentation"
        assert r.version == 1
        assert proj.rubric().name == "IPA segmentation"

        proj.set_rubric_statuses(
            [("draft", None, 0), ("done", "reviewed", 1), ("flagged", None, 2)]
        )
        values = [s.value for s in proj.rubric_statuses()]
        assert values == ["draft", "done", "flagged"]

        # Update is in place (singleton); created_at preserved.
        r2 = proj.set_rubric("IPA segmentation", version=2)
        assert r2.id == 1
        assert r2.version == 2
        assert r2.created_at == r.created_at


def test_status_and_note_on_intervals() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, tier = _project_with_tier(Path(td))
        iv = proj.add_interval(tier, 0.0, 0.1, label="a")

        # No rubric yet -> setting a status fails; a note alone needs a status
        # vocabulary too only for status, so set_*_status with status=None ok.
        with pytest.raises(RuntimeError):
            proj.set_interval_status(iv, status="done")

        proj.set_rubric("r")
        proj.set_rubric_statuses([("draft", None, 0), ("done", None, 1)])

        proj.set_interval_status(iv, status="done", note="looks good")
        row = proj.intervals(tier)[0]
        assert row.status == "done"
        assert row.note == "looks good"

        # Undefined status rejected.
        with pytest.raises(RuntimeError):
            proj.set_interval_status(iv, status="bogus")

        # A note may be set on any status, including none.
        proj.set_interval_status(iv, status=None, note="revisit")
        row = proj.intervals(tier)[0]
        assert row.status is None
        assert row.note == "revisit"

        # add_interval can carry status/note directly.
        iv2 = proj.add_interval(tier, 0.2, 0.3, label="i", status="draft", note="n")
        row2 = next(r for r in proj.intervals(tier) if r.id == iv2)
        assert row2.status == "draft"
        assert row2.note == "n"


def test_controlled_vocabulary_open_and_closed() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, tier = _project_with_tier(Path(td))
        proj.set_rubric("r")

        # Unconfigured tier: unconstrained.
        chk = proj.label_check("phones", "anything")
        assert not chk.has_vocabulary and not chk.closed and chk.in_vocabulary

        proj.set_controlled_vocabulary("phones", [("a", None, 0), ("i", "close front", 1)])
        assert [v.value for v in proj.controlled_vocabulary("phones")] == ["a", "i"]

        chk = proj.label_check("phones", "a")
        assert chk.has_vocabulary and not chk.closed and chk.in_vocabulary
        chk = proj.label_check("phones", "zzz")
        assert chk.has_vocabulary and not chk.closed and not chk.in_vocabulary

        # Open vocabulary still accepts an out-of-vocab label on insert.
        proj.add_interval(tier, 0.0, 0.1, label="zzz")

        # Close it: out-of-vocab now rejected; in-vocab / empty still accepted.
        rt = proj.set_rubric_tier("phones", description="IPA phones", closed=True)
        assert rt.closed_vocabulary is True
        with pytest.raises(RuntimeError):
            proj.add_interval(tier, 0.1, 0.2, label="zzz")
        proj.add_interval(tier, 0.2, 0.3, label="i")
        proj.add_interval(tier, 0.3, 0.4, label=None)


def test_s1_surface_is_provisional() -> None:
    from sadda._stability import get_stability

    for sym in (sadda.Rubric, sadda.StatusDef, sadda.RubricTier, sadda.VocabEntry, sadda.LabelCheck):
        assert get_stability(sym) == "provisional", sym
