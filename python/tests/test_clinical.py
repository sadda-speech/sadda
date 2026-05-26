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


def test_h1_h2_on_harmonic_tone() -> None:
    audio = sadda.load_wav(str(FIXTURES / "hnr_high_120hz.wav"))
    # 1/h harmonic amplitudes → H1−H2 = 20·log10(2) ≈ 6 dB.
    assert 4.5 < sadda.clinical.h1_h2(audio) < 7.5


def test_h1_h2_unreliable_raises(tmp_path) -> None:
    silent = tmp_path / "silent.wav"
    with wave.open(str(silent), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(struct.pack("<" + "h" * 16_000, *([0] * 16_000)))
    audio = sadda.load_wav(str(silent))
    with pytest.raises(ValueError):
        sadda.clinical.h1_h2(audio)


def test_perturbation_reports_psd() -> None:
    clean = sadda.clinical.perturbation(sadda.load_wav(str(FIXTURES / "clean_120hz.wav")))
    jittered = sadda.clinical.perturbation(sadda.load_wav(str(FIXTURES / "jitter_150hz.wav")))
    # Period standard deviation (PSD): small, positive, larger with jitter.
    assert jittered.period_std_s > clean.period_std_s >= 0.0
    assert jittered.period_std_s < 0.001


def test_gne_orders_clean_above_noisy_and_is_provisional() -> None:
    import warnings

    from sadda._stability import get_stability

    assert get_stability(sadda.clinical.gne) == "provisional"
    high = sadda.load_wav(str(FIXTURES / "hnr_high_120hz.wav"))
    mid = sadda.load_wav(str(FIXTURES / "hnr_mid_120hz.wav"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g_high = sadda.clinical.gne(high)
        g_mid = sadda.clinical.gne(mid)
    assert 0.0 <= g_mid < g_high <= 1.0


def test_hfno_and_hnr_d_order_clean_above_noisy() -> None:
    import warnings

    from sadda._stability import get_stability

    assert get_stability(sadda.clinical.hfno) == "provisional"
    assert get_stability(sadda.clinical.hnr_d) == "provisional"
    high = sadda.load_wav(str(FIXTURES / "hnr_high_120hz.wav"))
    mid = sadda.load_wav(str(FIXTURES / "hnr_mid_120hz.wav"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sadda.clinical.hfno(high) > sadda.clinical.hfno(mid)
        assert sadda.clinical.hnr_d(high) > sadda.clinical.hnr_d(mid)


def test_abi_formula_orders_and_is_provisional() -> None:
    import warnings

    from sadda._stability import get_stability

    assert get_stability(sadda.clinical.abi) == "provisional"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # (cpps, jit%, gne, hfno, hnr_d, h1h2, shim_db, shim%, psd_s)
        clean = sadda.clinical.abi(12.0, 0.3, 0.85, 2.0, 8.0, 2.0, 0.2, 2.0, 0.0002)
        breathy = sadda.clinical.abi(7.0, 1.0, 0.5, 1.0, 3.0, 8.0, 0.8, 6.0, 0.0005)
    assert clean < breathy
    assert 0.0 <= clean <= 10.0 and 0.0 <= breathy <= 10.0


def test_avqi_formula_orders_and_is_provisional() -> None:
    import warnings

    from sadda._stability import get_stability

    # AVQI is provisional (not yet reference-confirmed) → warns on use.
    assert get_stability(sadda.clinical.avqi) == "provisional"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        normal = sadda.clinical.avqi(14.50, 21.96, 2.77, 0.35, -24.73, -10.66)
        dysphonic = sadda.clinical.avqi(8.57, 16.31, 7.80, 0.75, -31.51, -9.31)
    assert normal < dysphonic
    assert 0.0 <= normal <= 10.0


def test_clinical_surface_is_stable_clinical() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.clinical.perturbation) == "stable-clinical"
    assert get_stability(sadda.clinical.hnr) == "stable-clinical"
    assert get_stability(sadda.clinical.cpps) == "stable-clinical"
    assert get_stability(sadda.clinical.h1_h2) == "stable-clinical"
