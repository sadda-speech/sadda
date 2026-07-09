"""Adaptive pitch-range (De Looze & Hirst 2008) Python surface:
`estimate_pitch_range` and `voiced_pitch(range_mode="two_pass")`.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

import sadda
from sadda import dsp


def _tone(tmp_path: Path, freq: float) -> "sadda.Audio":
    wav = tmp_path / f"tone_{int(freq)}.wav"
    n = int(16_000 * 0.6)
    samples = np.sin(2 * np.pi * freq * np.arange(n) / 16_000)
    pcm = (samples * 32767).astype(np.int16).tobytes()
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(pcm)
    return sadda.load_wav(str(wav))


def test_estimate_pitch_range_brackets_tone(tmp_path: Path) -> None:
    f0 = 150.0
    rng = dsp.estimate_pitch_range(_tone(tmp_path, f0))
    assert rng is not None
    floor, ceiling = rng
    # De Looze & Hirst on a clean tone: floor ≈ 0.75·f0, ceiling ≈ 1.5·f0.
    assert floor < f0 < ceiling
    assert floor == pytest.approx(0.75 * f0, rel=0.1)
    assert ceiling == pytest.approx(1.5 * f0, rel=0.1)


def test_two_pass_recovers_tone(tmp_path: Path) -> None:
    f0 = 150.0
    _t, freqs, voicing = dsp.voiced_pitch(
        _tone(tmp_path, f0), method="boersma", range_mode="two_pass"
    )
    voiced = freqs[voicing >= 0.45]
    assert len(voiced) > 5
    assert float(np.median(voiced)) == pytest.approx(f0, abs=3.0)


def test_unknown_range_mode_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="range_mode"):
        dsp.voiced_pitch(_tone(tmp_path, 150.0), range_mode="nonsense")
