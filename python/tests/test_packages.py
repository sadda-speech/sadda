"""Python-surface tests for S4c campaign packages: per-annotator export, import
(landing on per-annotator tiers), and the explicit merge_tiers union."""

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


def test_export_import_round_trip() -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        wav = tdp / "t.wav"
        _write_wav(wav)
        parent = sadda.new_project(tdp / "study", "study")
        bundle_id = parent.add_bundle("b", wav)
        phones = parent.add_tier(bundle_id, "phones", "interval")
        parent.add_interval(phones, 0.0, 0.2, label="a")
        parent.set_rubric("scheme", 1, "annotate vowels")
        ta = parent.add_target(bundle_id, 0.0, 0.2, "vowels")
        parent.add_assignment(ta, "alice")
        tb = parent.add_target(bundle_id, 0.5, 0.7, "vowels")
        parent.add_assignment(tb, "bob")

        pkg_dir = tdp / "alice_pkg"
        summary = parent.export_annotator_package("alice", pkg_dir)
        assert (summary.bundles, summary.targets, summary.assignments) == (1, 1, 1)
        assert summary.annotator == "alice"

        # alice works in the package: add a "vowels" tier + annotation.
        pkg = sadda.open_project(pkg_dir)
        pb = pkg.bundles()[0].id
        assert len(pkg.targets(pb)) == 1
        vowels = pkg.add_tier(pb, "vowels", "interval")
        pkg.add_interval(vowels, 0.05, 0.15, label="a")
        del pkg  # release the package so import can open it

        imp = parent.import_annotator_package(pkg_dir)
        assert imp.annotator == "alice"
        assert (imp.bundles_matched, imp.tiers_imported, imp.annotations_imported) == (1, 1, 1)
        assert imp.assignments_marked_done == 1

        valice = next(t for t in parent.tiers(bundle_id) if t.name == "vowels [alice]")
        assert len(parent.intervals(valice.id)) == 1
        assigns = {a.annotator: a.status for a in parent.assignments(bundle_id)}
        assert assigns["alice"] == "done"
        assert assigns["bob"] == "assigned"


def test_merge_tiers_unions_in_time_order() -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        wav = tdp / "t.wav"
        _write_wav(wav)
        proj = sadda.new_project(tdp / "p", "p")
        bundle_id = proj.add_bundle("b", wav)
        a = proj.add_tier(bundle_id, "phones [alice]", "interval")
        b = proj.add_tier(bundle_id, "phones [bob]", "interval")
        proj.add_interval(a, 0.2, 0.25, label="a")
        proj.add_interval(a, 0.0, 0.05, label="a")
        proj.add_interval(b, 0.5, 0.55, label="a")

        n = proj.merge_tiers(bundle_id, ["phones [alice]", "phones [bob]"], "phones")
        assert n == 3
        merged = next(t for t in proj.tiers(bundle_id) if t.name == "phones")
        starts = [round(i.start_seconds, 3) for i in proj.intervals(merged.id)]
        assert starts == [0.0, 0.2, 0.5]

        with pytest.raises((ValueError, RuntimeError)):
            proj.merge_tiers(bundle_id, ["nope"], "phones")


def test_import_rejects_non_package_dir() -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        wav = tdp / "t.wav"
        _write_wav(wav)
        proj = sadda.new_project(tdp / "p", "p")
        proj.add_bundle("b", wav)
        with pytest.raises((ValueError, RuntimeError)):
            proj.import_annotator_package(tdp / "p")
