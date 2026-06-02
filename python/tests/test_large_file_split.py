"""Python-surface tests for the large-file ingest guard: a header-only probe
(size without decoding) and a streaming chunked split that lands each piece as
its own bundle."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import sadda


def _write_wav(path: Path, sample_rate: int, n_frames: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n_frames)


def test_probe_reads_header_without_decoding() -> None:
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "p.wav"
        _write_wav(wav, 16_000, 4_000)
        probe = sadda.probe_wav(wav)
        assert probe.sample_rate == 16_000
        assert probe.channels == 1
        assert probe.n_frames == 4_000
        assert abs(probe.duration_seconds - 0.25) < 1e-9
        assert probe.decoded_bytes == 4_000 * 4
        assert "AudioProbe" in repr(probe)


def test_add_bundle_split_streams_into_chunks() -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        wav = tdp / "long.wav"
        _write_wav(wav, 4_000, 1_000)  # 0.25 s

        proj = sadda.new_project(tdp / "proj", "proj")
        # 0.1 s chunks @ 4 kHz = 400 frames → 400, 400, 200 = 3 bundles.
        ids = proj.add_bundle_split("long", wav, 0.1)
        assert len(ids) == 3

        bundles = proj.bundles()
        assert [b.name for b in bundles] == ["long_001", "long_002", "long_003"]
        assert [b.n_frames for b in bundles] == [400, 400, 200]
        assert sum(b.n_frames for b in bundles) == 1_000
