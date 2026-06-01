"""Python-surface tests for S6a: the compile + QA dashboard (completeness from
assignments, per-tier QA, and agreement summary over per-annotator tiers)."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

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


def test_completeness_per_annotator() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))
        t0 = proj.add_target(bundle_id, 0.0, 0.1, "phones")
        t1 = proj.add_target(bundle_id, 0.2, 0.3, "phones")
        a0 = proj.add_assignment(t0, "alice")
        proj.add_assignment(t1, "bob")
        proj.update_assignment_status(a0, "done")

        pp = proj.project_target_progress()
        assert pp.total == 2 and pp.assigned == 2

        prog = proj.assignment_progress()
        assert [p.annotator for p in prog] == ["alice", "bob"]
        assert prog[0].done == 1
        assert prog[1].assigned == 1


def test_tier_qa_flags_vocab_and_overlaps() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))
        phones = proj.add_tier(bundle_id, "phones", "interval")
        proj.set_rubric("r", 1, None)
        proj.set_controlled_vocabulary("phones", [("a", None, 0)])
        proj.add_interval(phones, 0.0, 0.1, label="a")
        proj.add_interval(phones, 0.1, 0.2, label="zzz")  # out of vocab
        proj.add_interval(phones, 0.15, 0.25, label=None)  # missing + overlap

        qa = proj.tier_qa(phones)
        assert qa.n_annotations == 3
        assert qa.out_of_vocab == 1
        assert qa.missing_label == 1
        assert qa.overlaps == 1


def test_agreement_summary_over_per_annotator_tiers() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))
        for annot, last in [("alice", "a"), ("bob", "b")]:
            tid = proj.add_tier(bundle_id, f"vowels [{annot}]", "interval")
            proj.add_interval(tid, 0.0, 0.1, label=last)

        pairs = proj.agreement_summary(bundle_id, "vowels")
        assert len(pairs) == 1
        assert (pairs[0].annotator_a, pairs[0].annotator_b) == ("alice", "bob")
        assert pairs[0].report.n_matched == 1
        assert pairs[0].report.percent_label_agreement == 0.0  # a vs b


def test_dashboard_types_provisional() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.AnnotatorProgress) == "provisional"
    assert get_stability(sadda.QaReport) == "provisional"
    assert get_stability(sadda.PairAgreement) == "provisional"
