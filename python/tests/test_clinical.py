"""B4 — jitter/shimmer clinical measures, Python surface."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest
import sadda

FIXTURES = Path(__file__).resolve().parents[2] / "crates/engine/tests/clinical/fixtures"


def test_perturbation_on_shimmer_signal() -> None:
    audio = sadda.load_wav(str(FIXTURES / "shimmer_150hz.wav"))
    r = sadda.clinical.perturbation(audio)
    # Injected ~6% shimmer, ~0 jitter.
    assert 0.04 < r.shimmer_local < 0.09
    assert r.jitter_local < 0.01
    assert r.shimmer_local_db > 0.0
    assert r.n_periods > 50


def test_perturbation_on_jitter_signal() -> None:
    audio = sadda.load_wav(str(FIXTURES / "jitter_150hz.wav"))
    r = sadda.clinical.perturbation(audio)
    assert 0.01 < r.jitter_local < 0.04
    assert r.shimmer_local < 0.03


def test_perturbation_unreliable_raises(tmp_path) -> None:
    silent = tmp_path / "silent.wav"
    with wave.open(str(silent), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(struct.pack("<" + "h" * 16_000, *([0] * 16_000)))
    audio = sadda.load_wav(str(silent))
    with pytest.raises(ValueError):
        sadda.clinical.perturbation(audio)


def test_hnr_on_sustained_tones() -> None:
    high = sadda.load_wav(str(FIXTURES / "hnr_high_120hz.wav"))
    mid = sadda.load_wav(str(FIXTURES / "hnr_mid_120hz.wav"))
    # Injected 25 dB and 12 dB HNR.
    assert sadda.clinical.hnr(high) > 20.0
    assert 8.0 < sadda.clinical.hnr(mid) < 16.0


def test_cpps_on_sustained_tones() -> None:
    high = sadda.load_wav(str(FIXTURES / "hnr_high_120hz.wav"))
    mid = sadda.load_wav(str(FIXTURES / "hnr_mid_120hz.wav"))
    cpps_high = sadda.clinical.cpps(high)
    cpps_mid = sadda.clinical.cpps(mid)
    # Cleaner tone → higher cepstral prominence than the noisier one.
    assert cpps_high > cpps_mid
    assert 15.0 < cpps_high < 27.0


def test_clinical_surface_is_stable_clinical() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.clinical.perturbation) == "stable-clinical"
    assert get_stability(sadda.clinical.hnr) == "stable-clinical"
    assert get_stability(sadda.clinical.cpps) == "stable-clinical"
