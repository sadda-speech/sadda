"""LTAS feature — long-term average spectrum + slope/tilt/alpha ratio."""

from __future__ import annotations

from pathlib import Path

import sadda

FIXTURES = Path(__file__).resolve().parents[2] / "crates/engine/tests/clinical/fixtures"


def test_ltas_slope_and_shape() -> None:
    audio = sadda.load_wav(str(FIXTURES / "hnr_high_120hz.wav"))
    l = sadda.dsp.ltas(audio, bin_hz=100.0)
    assert l.bin_hz == 100.0
    assert len(l.levels_db) > 100
    # Harmonic tone rolls off → negative slope and tilt.
    slope = l.slope(0.0, 1000.0, 1000.0, 4000.0)
    assert slope < 0.0
    assert l.tilt(0.0, 5000.0) < 0.0


def test_ltas_is_stable() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.dsp.ltas) == "stable"
