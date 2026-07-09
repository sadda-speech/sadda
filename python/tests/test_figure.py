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


def test_export_figure_with_measure_lanes() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle = _project_with_annotated_bundle(Path(td))
        out = Path(td) / "measures.svg"
        proj.export_figure(bundle, out, f0=True, formants=True, intensity=True)
        svg = out.read_text()
        # The intensity lane is deterministic; f0 tracks on the sine clip.
        assert ">intensity</text>" in svg
        assert ">f0</text>" in svg


def test_export_figure_whole_signal_column() -> None:
    """Every lane at once: waveform + spectrogram + f0/formants/intensity +
    MFCC heatmap + tiers — plus a style knob."""
    with tempfile.TemporaryDirectory() as td:
        proj, bundle = _project_with_annotated_bundle(Path(td))
        out = Path(td) / "column.svg"
        proj.export_figure(
            bundle,
            out,
            f0=True,
            formants=True,
            intensity=True,
            mfcc=True,
            font_size=15.0,
            colormap="magma",
        )
        svg = out.read_text()
        assert ">MFCC</text>" in svg
        assert ">intensity</text>" in svg
        # Two rasters embedded: the spectrogram + the MFCC heatmap.
        assert svg.count("data:image/png;base64,") == 2


def test_export_figure_mfcc_tikz_writes_heatmap_sidecar() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle = _project_with_annotated_bundle(Path(td))
        out = Path(td) / "hm.tex"
        proj.export_figure(bundle, out, format="tikz", mfcc=True)
        tex = out.read_text()
        # Both raster sidecars are referenced + written.
        assert "hm-spectrogram.png" in tex
        assert "hm-heatmap0.png" in tex
        assert (Path(td) / "hm-spectrogram.png").exists()
        assert (Path(td) / "hm-heatmap0.png").exists()


def test_export_figure_embedding_heatmap_from_tier() -> None:
    import numpy as np

    with tempfile.TemporaryDirectory() as td:
        proj, bundle = _project_with_annotated_bundle(Path(td))
        emb = proj.add_tier(bundle, "emb", "continuous_vector")
        # 20 frames × 4 dims of dense vectors at 100 Hz.
        mat = np.random.RandomState(0).rand(20, 4).astype(np.float64)
        proj.write_continuous_vector(emb, mat, 100.0)
        out = Path(td) / "emb.svg"
        proj.export_figure(bundle, out, embedding_tier_id=emb)
        svg = out.read_text()
        assert ">embedding</text>" in svg
        # Two rasters: the spectrogram + the embedding heatmap.
        assert svg.count("data:image/png;base64,") == 2


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


def test_export_figure_pdf_is_self_contained() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle = _project_with_annotated_bundle(Path(td))
        out = Path(td) / "fig.pdf"
        proj.export_figure(bundle, out, format="pdf", title="praat")
        assert out.exists()
        data = out.read_bytes()
        assert data.startswith(b"%PDF-")  # a real PDF
        assert len(data) > 1000


def test_export_figure_tikz_writes_tex_and_raster_sidecar() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle = _project_with_annotated_bundle(Path(td))
        out = Path(td) / "fig.tex"
        proj.export_figure(bundle, out, format="tikz", title="praat")
        tex = out.read_text()
        assert "\\begin{tikzpicture}" in tex
        assert "\\end{document}" in tex.rstrip()
        # IPA survives as real LaTeX text (rendered via fontspec + Doulos SIL).
        assert "{ɹ}" in tex
        # The spectrogram is a sidecar PNG next to the .tex.
        sidecar = Path(td) / "fig-spectrogram.png"
        assert sidecar.exists()
        assert "\\includegraphics" in tex and "fig-spectrogram.png" in tex


def test_export_figure_rejects_unsupported_format() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle = _project_with_annotated_bundle(Path(td))
        out = Path(td) / "fig.png"
        with pytest.raises(Exception, match="not supported"):
            proj.export_figure(bundle, out, format="png")


def test_export_figure_rejects_unknown_colormap() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle = _project_with_annotated_bundle(Path(td))
        out = Path(td) / "fig.svg"
        with pytest.raises(ValueError, match="unknown colormap"):
            proj.export_figure(bundle, out, colormap="not-a-map")
