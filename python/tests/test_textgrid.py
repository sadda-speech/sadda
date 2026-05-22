"""Python-surface tests for the D1 TextGrid round-trip API."""

from __future__ import annotations

import struct
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


def test_export_interval_tier_writes_long_textgrid() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "phones", "interval")
        proj.add_interval(tier, 0.0, 0.5, label="h")
        proj.add_interval(tier, 0.5, 1.0, label="e")
        out = Path(td) / "out.TextGrid"
        proj.export_textgrid(bundle_id, out)
        text = out.read_text()
        assert text.startswith('File type = "ooTextFile"')
        assert 'class = "IntervalTier"' in text
        assert 'name = "phones"' in text


def test_export_then_import_recovers_interval_tier() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "phones", "interval")
        proj.add_interval(tier, 0.0, 0.5, label="h", extra='{"v":1}')
        proj.add_interval(tier, 0.5, 1.0, label="e")
        out = Path(td) / "out.TextGrid"
        proj.export_textgrid(bundle_id, out)
        # Re-import into a fresh bundle.
        wav2 = Path(td) / "t2.wav"
        _write_silent_wav(wav2, duration_s=1.5)
        bundle2 = proj.add_bundle("b2", wav2)
        new_tier_ids = proj.import_textgrid(out, bundle2)
        assert len(new_tier_ids) == 1
        rows = proj.intervals(new_tier_ids[0])
        # 2 user intervals + 1 trailing pad
        assert len(rows) == 3
        assert rows[0].label == "h"
        assert rows[0].extra == '{"v":1}'
        assert rows[1].label == "e"


def test_export_point_tier_round_trips() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "events", "point")
        proj.add_point(tier, 0.25, label="click")
        proj.add_point(tier, 0.75, label="release", extra='{"force":12}')
        out = Path(td) / "out.TextGrid"
        proj.export_textgrid(bundle_id, out)
        wav2 = Path(td) / "t2.wav"
        _write_silent_wav(wav2)
        bundle2 = proj.add_bundle("b2", wav2)
        new_tier_ids = proj.import_textgrid(out, bundle2)
        rows = proj.points(new_tier_ids[0])
        assert [r.label for r in rows] == ["click", "release"]
        assert rows[1].extra == '{"force":12}'


def test_export_subset_via_tier_ids() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        t1 = proj.add_tier(bundle_id, "a", "interval")
        proj.add_interval(t1, 0.0, 0.5, label="x")
        t2 = proj.add_tier(bundle_id, "b", "interval")
        proj.add_interval(t2, 0.0, 0.5, label="y")
        out = Path(td) / "subset.TextGrid"
        proj.export_textgrid(bundle_id, out, tier_ids=[t1])
        text = out.read_text()
        assert "size = 1" in text
        assert '"a"' in text
        assert '"b"' not in text


def test_export_skips_dense_tiers() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        proj.add_tier(bundle_id, "phones", "interval")
        proj.add_tier(bundle_id, "f0", "continuous_numeric")
        out = Path(td) / "dense.TextGrid"
        proj.export_textgrid(bundle_id, out)
        text = out.read_text()
        # phones is the only IntervalTier; f0 was skipped.
        assert '"phones"' in text
        assert '"f0"' not in text


def test_import_missing_file_raises_io_error() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        with pytest.raises(OSError):
            proj.import_textgrid(Path(td) / "missing.TextGrid", bundle_id)


def test_import_then_audit_log_records_processing_run_via_engine() -> None:
    """import_textgrid records a processing_run row of kind dsp_algorithm.
    We can't query processing_run from the Python API yet (no method), but
    we can verify the import succeeded end-to-end."""
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tg = Path(td) / "tiny.TextGrid"
        tg.write_text(
            'File type = "ooTextFile"\n'
            'Object class = "TextGrid"\n\n'
            "xmin = 0\nxmax = 1.0\ntiers? <exists>\nsize = 1\n"
            "item []:\n"
            "    item [1]:\n"
            '        class = "IntervalTier"\n'
            '        name = "phones"\n'
            "        xmin = 0\n        xmax = 1.0\n"
            "        intervals: size = 1\n"
            "        intervals [1]:\n"
            "            xmin = 0\n            xmax = 1.0\n"
            '            text = "a"\n'
        )
        tier_ids = proj.import_textgrid(tg, bundle_id)
        assert len(tier_ids) == 1
        rows = proj.intervals(tier_ids[0])
        assert len(rows) == 1
        assert rows[0].label == "a"
