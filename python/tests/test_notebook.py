"""Python-surface tests for S7: the PI lab-notebook — capturing exploration
notes and promoting them into a criterion or rubric-tier guidance."""

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


def test_capture_edit_and_filter() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))
        e1 = proj.add_notebook_entry(
            "vowels",
            "creaky below 120 Hz",
            kind="measurement",
            measurement="mean f0 = 110",
            bundle_id=bundle_id,
        )
        proj.add_notebook_entry("words", "function words reduce")

        assert len(proj.notebook_entries("vowels")) == 1
        assert len(proj.notebook_entries()) == 2
        got = proj.get_notebook_entry(e1)
        assert got.kind == "measurement"
        assert got.measurement == "mean f0 = 110"
        assert got.promoted_kind is None

        proj.update_notebook_entry(e1, "creaky below 115 Hz")
        assert proj.get_notebook_entry(e1).text == "creaky below 115 Hz"
        with pytest.raises((ValueError, RuntimeError)):
            proj.add_notebook_entry("x", "t", kind="bogus")


def test_promote_to_criterion_and_guidance() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))
        proj.add_tier(bundle_id, "phones", "interval")
        e = proj.add_notebook_entry("vowels", "creaky vowels exist")

        crit = proj.promote_entry_to_criterion(
            e,
            "creaky vowels",
            "structured",
            '{"select": {"tier": "phones"}, "emit": {"kind": "span"}}',
            "creaky",
        )
        assert crit.name == "creaky vowels"
        promoted = proj.get_notebook_entry(e)
        assert promoted.promoted_kind == "criterion"
        assert promoted.promoted_ref == "creaky vowels"

        w = proj.add_notebook_entry("words", "function words reduce")
        proj.set_rubric("scheme", 1, None)
        proj.set_rubric_tier("words", "existing", False)
        proj.promote_entry_to_rubric_guidance(w)
        rt = proj.rubric_tier("words")
        assert rt.description == "existing\nfunction words reduce"
        assert proj.get_notebook_entry(w).promoted_kind == "rubric_guidance"


def test_notebook_is_provisional() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.NotebookEntry) == "provisional"
