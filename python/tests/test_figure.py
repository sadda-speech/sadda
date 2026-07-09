"""Python-surface tests for publication figure export (`Project.export_figure`,
G1).

Exports a waveform / spectrogram / tier figure of a bundle to a self-contained
SVG and asserts the structure: an embedded font + spectrogram raster, IPA tier
labels as real text, and the shared time axis. Also covers the include-flag and
error paths (unsupported format, unknown colormap)."""

from __future__ import annotations

import math
import struct
import tempfile
import wave
from pathlib import Path

import pytest

import sadda


def _write_sine_wav(path: Path, sample_rate: int = 16_000, freq: float = 220.0) -> None:
    n = sample_rate // 2  # 0.5 s
    frames = bytearray()
    for i in range(n):
        v = int(0.6 * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
        frames += struct.pack("<h", v)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(frames))


def _project_with_annotated_bundle(td: Path) -> tuple[sadda.Project, int]:
    proj = sadda.new_project(td / "p", "demo")
    wav = td / "praat.wav"
    _write_sine_wav(wav)
    bundle = proj.add_bundle("praat", wav)
    phones = proj.add_tier(bundle, "phones", "interval")
    # IPA labels — the point of embedding Doulos SIL.
    for start, end, label in [
        (0.0, 0.1, "p"),
        (0.1, 0.25, "ɹ"),
        (0.25, 0.4, "ɑː"),
        (0.4, 0.5, "t"),
    ]:
        proj.add_interval(phones, start, end, label=label)
    events = proj.add_tier(bundle, "events", "point")
    proj.add_point(events, 0.1, label="burst")
    return proj, bundle


def test_export_figure_writes_self_contained_svg() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle = _project_with_annotated_bundle(Path(td))
        out = Path(td) / "fig.svg"
        proj.export_figure(bundle, out, title="praat")
        assert out.exists()
        svg = out.read_text()
        assert svg.startswith("<svg")
        assert svg.rstrip().endswith("</svg>")
        # Self-contained: embedded font + embedded raster, no external files.
        assert "@font-face" in svg
        assert "data:font/ttf;base64," in svg
        assert "data:image/png;base64," in svg
        # IPA survives as real text (editable), and the shared axis is present.
        assert ">ɹ</text>" in svg
        assert ">ɑː</text>" in svg
        assert "Time (s)" in svg
        assert ">praat</text>" in svg  # title


def test_export_figure_respects_include_flags() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle = _project_with_annotated_bundle(Path(td))
        out = Path(td) / "no_spec.svg"
        proj.export_figure(bundle, out, spectrogram=False)
        svg = out.read_text()
        # No spectrogram raster when the lane is excluded; waveform still drawn.
        assert "data:image/png;base64," not in svg
        assert "<path" in svg  # the waveform band


def test_export_figure_tier_selection() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle = _project_with_annotated_bundle(Path(td))
        tiers = {t.name: t for t in proj.tiers(bundle)}
        out = Path(td) / "phones_only.svg"
        proj.export_figure(bundle, out, tier_ids=[tiers["phones"].id])
        svg = out.read_text()
        assert ">phones</text>" in svg
        assert ">events</text>" not in svg


def test_export_figure_rejects_unsupported_format() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle = _project_with_annotated_bundle(Path(td))
        out = Path(td) / "fig.pdf"
        with pytest.raises(Exception, match="not supported"):
            proj.export_figure(bundle, out, format="pdf")


def test_export_figure_rejects_unknown_colormap() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle = _project_with_annotated_bundle(Path(td))
        out = Path(td) / "fig.svg"
        with pytest.raises(ValueError, match="unknown colormap"):
            proj.export_figure(bundle, out, colormap="not-a-map")
