"""Python-surface tests for the S2 criteria engine: the criterion API, the
structured engine path, the python-escape executor, and accept/reject."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import pytest

import sadda
from sadda import criteria


def _write_wav(path: Path, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * (sample_rate // 4))


def _project(td: Path) -> tuple[sadda.Project, int]:
    proj = sadda.new_project(td / "p", "demo")
    wav = td / "t.wav"
    _write_wav(wav)
    bundle_id = proj.add_bundle("b", wav)
    phones = proj.add_tier(bundle_id, "phones", "interval")
    for s, e, lbl in [(0.0, 0.1, "a"), (0.1, 0.2, "b"), (0.5, 0.6, "a")]:
        proj.add_interval(phones, s, e, label=lbl)
    return proj, bundle_id


def test_structured_criterion_run_and_accept() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))

        body = '{"select": {"tier": "phones", "label_any": ["a"]}, "emit": {"kind": "point", "at": 0.5}}'
        crit = proj.set_criterion("vowel midpoints", "structured", body, "landmarks")
        assert crit.kind == "structured"
        assert crit.target_tier == "landmarks"
        assert [c.name for c in proj.criteria()] == ["vowel midpoints"]

        n = proj.run_criterion(crit.id, bundle_id)
        assert n == 2  # two "a" intervals -> two midpoints

        preview = next(t for t in proj.tiers(bundle_id) if t.name == "landmarks (auto)")
        assert preview.type == "point"
        assert len(proj.points(preview.id)) == 2

        promoted = proj.accept_proposals(bundle_id, "landmarks")
        assert promoted == 2
        assert proj.points(preview.id) == []
        landmarks = next(t for t in proj.tiers(bundle_id) if t.name == "landmarks")
        assert len(proj.points(landmarks.id)) == 2


def test_python_escape_criterion() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))

        # A python criterion: emit a span over every "a" phone, label "vowel".
        body = (
            "def criterion(proj, bundle_id):\n"
            "    phones = next(t for t in proj.tiers(bundle_id) if t.name == 'phones')\n"
            "    out = []\n"
            "    for iv in proj.intervals(phones.id):\n"
            "        if iv.label == 'a':\n"
            "            out.append((iv.start_seconds, iv.end_seconds, 'vowel'))\n"
            "    return out\n"
        )
        crit = proj.set_criterion("py vowels", "python", body, "vowels")

        # The engine refuses to run a python criterion directly...
        with pytest.raises(RuntimeError):
            proj.run_criterion(crit.id, bundle_id)
        # ...but the python-escape executor runs it.
        n = criteria.run_criterion(proj, crit.id, bundle_id)
        assert n == 2

        preview = next(t for t in proj.tiers(bundle_id) if t.name == "vowels (auto)")
        assert preview.type == "interval"
        ivs = proj.intervals(preview.id)
        assert len(ivs) == 2
        assert all(iv.label == "vowel" for iv in ivs)

        # Reject discards the proposals.
        assert proj.clear_proposals(bundle_id, "vowels") == 2
        assert proj.intervals(preview.id) == []


def test_criteria_executor_dispatches_structured() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))
        body = '{"select": {"tier": "phones"}, "emit": {"kind": "span"}}'
        crit = proj.set_criterion("all", "structured", body, "copy")
        # The executor delegates structured criteria to the engine.
        assert criteria.run_criterion(proj, crit.id, bundle_id) == 3


def test_structured_run_records_provenance_and_survives_accept() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))
        body = '{"select": {"tier": "phones", "label_any": ["a"]}, "emit": {"kind": "point"}}'
        crit = proj.set_criterion("vowels", "structured", body, "landmarks")

        assert proj.processing_runs(bundle_id) == []
        proj.run_criterion(crit.id, bundle_id)

        runs = proj.processing_runs(bundle_id)
        assert len(runs) == 1
        run = runs[0]
        assert run.kind == "criterion_run"
        assert run.processor_id == "sadda.criteria.vowels"
        assert proj.get_processing_run(run.id).id == run.id
        assert proj.get_processing_run(99_999) is None

        # Each proposal carries the run link...
        preview = next(t for t in proj.tiers(bundle_id) if t.name == "landmarks (auto)")
        pts = proj.points(preview.id)
        assert pts and all(p.processing_run_id == run.id for p in pts)

        # ...and it survives promotion onto the target tier.
        proj.accept_proposals(bundle_id, "landmarks")
        target = next(t for t in proj.tiers(bundle_id) if t.name == "landmarks")
        promoted = proj.points(target.id)
        assert promoted and all(p.processing_run_id == run.id for p in promoted)


def test_python_escape_run_is_also_traced() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))
        body = (
            "def criterion(proj, bundle_id):\n"
            "    phones = next(t for t in proj.tiers(bundle_id) if t.name == 'phones')\n"
            "    return [(iv.start_seconds, iv.end_seconds, 'vowel')\n"
            "            for iv in proj.intervals(phones.id) if iv.label == 'a']\n"
        )
        crit = proj.set_criterion("py vowels", "python", body, "vowels")
        criteria.run_criterion(proj, crit.id, bundle_id)

        # The executor records a criterion_run for the python path too.
        runs = proj.processing_runs(bundle_id)
        assert len(runs) == 1 and runs[0].kind == "criterion_run"
        ivs = proj.intervals(
            next(t for t in proj.tiers(bundle_id) if t.name == "vowels (auto)").id
        )
        assert ivs and all(iv.processing_run_id == runs[0].id for iv in ivs)


def test_s2_surface_is_provisional() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.Criterion) == "provisional"
