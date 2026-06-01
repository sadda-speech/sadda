"""Python-surface tests for S5: the comparison/agreement engine and the
campaign work-queue (progress + next-target)."""

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


def test_compare_tiers_inter_annotator() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))
        alice = proj.add_tier(bundle_id, "phones [alice]", "interval")
        bob = proj.add_tier(bundle_id, "phones [bob]", "interval")
        for tier, last in [(alice, "a"), (bob, "c")]:
            proj.add_interval(tier, 0.0, 0.1, label="a")
            proj.add_interval(tier, 0.1, 0.2, label="b")
            proj.add_interval(tier, 0.2, 0.3, label=last)

        r = proj.compare_tiers(bundle_id, alice, bob)
        assert r.tier_type == "interval"
        assert (r.n_matched, r.n_only_a, r.n_only_b) == (3, 0, 0)
        assert abs(r.percent_label_agreement - 2 / 3) < 1e-9
        assert r.cohen_kappa < 1.0
        assert r.mean_abs_boundary_diff < 1e-9
        # Frame metric is reported for interval tiers.
        assert 0.0 <= r.frame_percent_agreement <= 1.0

        # Identical tiers → perfect agreement; tolerance is tunable.
        r2 = proj.compare_tiers(bundle_id, alice, alice, boundary_tolerance_seconds=0.05)
        assert abs(r2.cohen_kappa - 1.0) < 1e-9
        assert abs(r2.frame_kappa - 1.0) < 1e-9
        assert abs(r2.boundary_tolerance_seconds - 0.05) < 1e-9


def test_compare_tiers_type_guard() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))
        iv = proj.add_tier(bundle_id, "iv", "interval")
        pt = proj.add_tier(bundle_id, "pt", "point")
        with pytest.raises((ValueError, RuntimeError)):
            proj.compare_tiers(bundle_id, iv, pt)


def test_target_progress_and_next_target() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project(Path(td))
        t0 = proj.add_target(bundle_id, 0.0, 0.1, "phones")
        t1 = proj.add_target(bundle_id, 0.2, 0.3, "phones")
        proj.add_target(bundle_id, 0.4, 0.5, "phones")
        proj.update_target_status(t0, "done")
        proj.update_target_status(t1, "flagged")

        p = proj.target_progress(bundle_id)
        assert (p.total, p.done, p.flagged, p.unassigned) == (3, 1, 1, 1)

        nxt = proj.next_target(bundle_id, ["unassigned"])
        assert nxt is not None and abs(nxt.start_seconds - 0.4) < 1e-9
        assert proj.next_target(bundle_id, ["flagged"]).id == t1
        assert proj.next_target(bundle_id, ["in_progress"]) is None


def test_agreement_is_provisional() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.AgreementReport) == "provisional"
    assert get_stability(sadda.ProgressCounts) == "provisional"
