"""A1 — provenance timeline + citation export, Python surface."""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path

import pytest
import sadda


def _write_short_wav(path: Path, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * sample_rate, *([0] * sample_rate)))


def _project_with_bundle(td: str):
    proj = sadda.new_project(Path(td) / "p", "demo")
    wav = Path(td) / "tone.wav"
    _write_short_wav(wav)
    bundle_id = proj.add_bundle("greeting", wav)
    return proj, bundle_id


def test_record_and_query_processing_runs() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(td)
        assert proj.processing_runs(bundle_id) == []

        rid = proj.record_processing_run(
            bundle_id,
            "dsp_algorithm",
            "sadda.dsp.pitch.autocorrelation",
            parameters='{"step":0.01}',
            output_signal_ids=[1],
        )
        runs = proj.processing_runs(bundle_id)
        assert len(runs) == 1
        assert runs[0].id == rid
        assert runs[0].kind == "dsp_algorithm"
        assert runs[0].processor_id == "sadda.dsp.pitch.autocorrelation"
        assert runs[0].status == "ok"
        assert runs[0].processor_version == sadda.version()
        assert runs[0].finished_at is not None


def test_invalid_kind_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(td)
        with pytest.raises(ValueError):
            proj.record_processing_run(bundle_id, "not_a_kind", "sadda.x")


def test_citations_dedup_and_omit_uncited() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(td)
        # Two mfcc runs (dedup to one citation) + one uncited tool op.
        proj.record_processing_run(bundle_id, "dsp_algorithm", "sadda.dsp.mfcc")
        proj.record_processing_run(bundle_id, "dsp_algorithm", "sadda.dsp.mfcc")
        proj.record_processing_run(
            bundle_id, "dsp_algorithm", "sadda.io.textgrid.import"
        )

        cites = proj.citations(bundle_id)
        assert len(cites) == 1
        assert "Davis" in cites[0].reference
        assert cites[0].doi == "10.1109/TASSP.1980.1163420"
        assert cites[0].processor_id == "sadda.dsp.mfcc"
