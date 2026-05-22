"""Python-surface tests for the D2 EAF round-trip API."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import pytest

import sadda


def _write_silent_wav(path: Path, sample_rate: int = 16_000, duration_s: float = 1.0) -> None:
    n = int(sample_rate * duration_s)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n)


def _project_with_bundle(td: Path, duration_s: float = 1.5) -> tuple[sadda.Project, int]:
    proj = sadda.new_project(td / "p", "demo")
    wav = td / "t.wav"
    _write_silent_wav(wav, duration_s=duration_s)
    bundle_id = proj.add_bundle("b1", wav)
    return proj, bundle_id


def test_export_writes_eaf_xml() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "phones", "interval")
        proj.add_interval(tier, 0.0, 0.5, label="h")
        proj.add_interval(tier, 0.5, 1.0, label="e")
        out = Path(td) / "out.eaf"
        proj.export_eaf(bundle_id, out)
        text = out.read_text()
        assert text.startswith("<?xml")
        assert 'FORMAT="2.8"' in text
        assert 'TIER_ID="phones"' in text


def test_export_then_import_recovers_interval_tier() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "phones", "interval")
        proj.add_interval(tier, 0.0, 0.5, label="h", extra='{"v":1}')
        proj.add_interval(tier, 0.5, 1.0, label="e")
        out = Path(td) / "out.eaf"
        proj.export_eaf(bundle_id, out)

        wav2 = Path(td) / "t2.wav"
        _write_silent_wav(wav2, duration_s=1.5)
        bundle2 = proj.add_bundle("b2", wav2)
        new_tier_ids = proj.import_eaf(out, bundle2)
        assert len(new_tier_ids) == 1
        rows = proj.intervals(new_tier_ids[0])
        assert [r.label for r in rows] == ["h", "e"]
        assert rows[0].extra == '{"v":1}'


def test_point_tier_round_trips_via_degenerate_alignable() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "events", "point")
        proj.add_point(tier, 0.25, label="click")
        proj.add_point(tier, 0.75, label="release", extra='{"force":12}')
        out = Path(td) / "out.eaf"
        proj.export_eaf(bundle_id, out)

        wav2 = Path(td) / "t2.wav"
        _write_silent_wav(wav2)
        bundle2 = proj.add_bundle("b2", wav2)
        new_tier_ids = proj.import_eaf(out, bundle2)
        rows = proj.points(new_tier_ids[0])
        assert [r.label for r in rows] == ["click", "release"]
        assert rows[1].extra == '{"force":12}'


def test_tier_hierarchy_survives_round_trip() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td), duration_s=2.0)
        words = proj.add_tier(bundle_id, "words", "interval")
        phones = proj.add_tier(
            bundle_id, "phones", "interval", parent_id=words, cardinality="one_to_many"
        )
        proj.add_interval(words, 0.0, 1.0, label="hi")
        parent_iv = proj.add_interval(words, 1.0, 2.0, label="there")
        proj.add_interval(
            phones, 1.0, 1.5, label="th", parent_annotation_id=parent_iv
        )

        out = Path(td) / "hierarchy.eaf"
        proj.export_eaf(bundle_id, out)

        wav2 = Path(td) / "t2.wav"
        _write_silent_wav(wav2, duration_s=2.0)
        bundle2 = proj.add_bundle("b2", wav2)
        new_tier_ids = proj.import_eaf(out, bundle2)
        assert len(new_tier_ids) == 2

        # Look up by name (topo-sort order isn't guaranteed by name).
        tiers = {t.name: t for t in proj.tiers(bundle2)}
        assert tiers["words"].parent_id is None
        assert tiers["phones"].parent_id == tiers["words"].id


def test_reference_tier_round_trips() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        words = proj.add_tier(bundle_id, "words", "interval")
        word = proj.add_interval(words, 0.0, 1.0, label="hello")
        lex = proj.add_tier(
            bundle_id, "lex", "reference", parent_id=words, cardinality="one_to_many"
        )
        proj.add_reference(
            lex, "annotation", word, label="greeting", parent_annotation_id=word
        )

        out = Path(td) / "ref.eaf"
        proj.export_eaf(bundle_id, out)

        wav2 = Path(td) / "t2.wav"
        _write_silent_wav(wav2)
        bundle2 = proj.add_bundle("b2", wav2)
        proj.import_eaf(out, bundle2)

        tiers = {t.name: t for t in proj.tiers(bundle2)}
        assert tiers["lex"].type == "reference"
        refs = proj.references_for(tiers["lex"].id)
        assert [r.label for r in refs] == ["greeting"]
        assert refs[0].target_kind == "annotation"


def test_export_subset_via_tier_ids() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        t1 = proj.add_tier(bundle_id, "a", "interval")
        proj.add_interval(t1, 0.0, 0.5, label="x")
        t2 = proj.add_tier(bundle_id, "b", "interval")
        proj.add_interval(t2, 0.0, 0.5, label="y")
        out = Path(td) / "subset.eaf"
        proj.export_eaf(bundle_id, out, tier_ids=[t1])
        text = out.read_text()
        assert 'TIER_ID="a"' in text
        assert 'TIER_ID="b"' not in text


def test_export_skips_dense_tiers() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        proj.add_tier(bundle_id, "phones", "interval")
        proj.add_tier(bundle_id, "f0", "continuous_numeric")
        out = Path(td) / "dense.eaf"
        proj.export_eaf(bundle_id, out)
        text = out.read_text()
        assert 'TIER_ID="phones"' in text
        assert 'TIER_ID="f0"' not in text


def test_import_missing_file_raises_io_error() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        with pytest.raises(OSError):
            proj.import_eaf(Path(td) / "missing.eaf", bundle_id)
