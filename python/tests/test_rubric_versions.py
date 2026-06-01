"""Python-surface tests for S6b: rubric version snapshots (publish / list /
recall) and the rubric-change impact report."""

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
    proj = sadda.new_project(td / "p", "p")
    wav = td / "t.wav"
    _write_wav(wav)
    return proj, proj.add_bundle("b", wav)


def test_publish_recall_and_impact() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))
        phones = proj.add_tier(bundle_id, "phones", "interval")

        proj.set_rubric("scheme", 1, "v1 guide")
        proj.set_controlled_vocabulary("phones", [("a", None, 0), ("b", None, 1)])
        v1 = proj.publish_rubric_version("initial")
        assert v1.version == 1
        assert v1.note == "initial"
        assert [v.value for v in v1.tiers[0].vocab] == ["a", "b"]

        proj.add_interval(phones, 0.0, 0.1, label="a")
        proj.add_interval(phones, 0.1, 0.2, label="b")

        # v2: drop "b", add "c".
        proj.set_rubric("scheme", 2, "v2 guide")
        proj.set_controlled_vocabulary("phones", [("a", None, 0), ("c", None, 1)])
        proj.publish_rubric_version("dropped b, added c")

        assert [v.version for v in proj.rubric_versions()] == [1, 2]
        recalled = proj.get_rubric_version(1)
        assert recalled.guidelines == "v1 guide"
        assert proj.get_rubric_version(99) is None

        impact = {t.tier_name: t for t in proj.rubric_impact(1)}
        ph = impact["phones"]
        assert ph.vocab_added == ["c"]
        assert ph.vocab_removed == ["b"]
        assert ph.affected_annotations == 1  # the "b" annotation

        with pytest.raises((ValueError, RuntimeError)):
            proj.rubric_impact(99)


def test_versioning_types_provisional() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.RubricVersion) == "provisional"
    assert get_stability(sadda.TierImpact) == "provisional"
