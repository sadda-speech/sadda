"""Python-surface tests for the P3 aggregate concordance view: concatenate all
tokens matching a tier + label filter into a single derived bundle, with a
source-divider tier and surrounding annotations remapped onto the timeline."""

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
        # 0.25 s of silence is enough for the token windows used below.
        w.writeframes(b"\x00\x00" * (sample_rate // 4))


def test_build_concordance_concatenates_tokens() -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        wav1 = tdp / "a.wav"
        wav2 = tdp / "b.wav"
        _write_wav(wav1)
        _write_wav(wav2)

        proj = sadda.new_project(tdp / "study", "study")

        b1 = proj.add_bundle("b1", wav1)
        ph1 = proj.add_tier(b1, "phone", "interval")
        proj.add_interval(ph1, 0.02, 0.05, label="f")
        proj.add_interval(ph1, 0.10, 0.13, label="s")  # non-matching
        proj.add_interval(ph1, 0.15, 0.18, label="f")
        wd1 = proj.add_tier(b1, "word", "interval")
        proj.add_interval(wd1, 0.0, 0.08, label="fish")  # context

        b2 = proj.add_bundle("b2", wav2)
        ph2 = proj.add_tier(b2, "phone", "interval")
        proj.add_interval(ph2, 0.05, 0.09, label="f")

        summary = proj.build_concordance(
            "phone", ["f"], "f-concordance", gap_seconds=0.05
        )

        assert summary.n_tokens == 3
        assert abs(summary.duration_seconds - 0.20) < 0.01
        assert summary.n_context_annotations >= 1
        assert "ConcordanceSummary" in repr(summary)

        new_id = summary.bundle_id
        tiers = {t.name: t for t in proj.tiers(new_id)}
        assert "⟨source⟩" in tiers  # ⟨source⟩ divider tier
        marks = proj.intervals(tiers["⟨source⟩"].id)
        assert len(marks) == 3
        assert marks[0].label.startswith("b1 @ 0.020s")

        # The "word" context tier was clipped + remapped onto the timeline.
        words = proj.intervals(tiers["word"].id)
        assert len(words) == 1
        assert words[0].label == "fish"
        assert abs(words[0].start_seconds) < 1e-6
        assert abs(words[0].end_seconds - 0.03) < 1e-3


def test_build_concordance_default_gap_and_any_label() -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        wav = tdp / "a.wav"
        _write_wav(wav)
        proj = sadda.new_project(tdp / "s2", "s2")
        b1 = proj.add_bundle("b1", wav)
        ph = proj.add_tier(b1, "phone", "interval")
        proj.add_interval(ph, 0.02, 0.05, label="f")
        proj.add_interval(ph, 0.10, 0.13, label="s")

        # Empty label filter = any label; default gap (0.25 s).
        summary = proj.build_concordance("phone", [], "all-phones", gap_seconds=0.0)
        assert summary.n_tokens == 2
