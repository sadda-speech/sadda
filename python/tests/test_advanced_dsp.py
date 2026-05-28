"""Python-surface tests for the C2 advanced DSP (formants, MFCC,
voiced_pitch). The DSP method-diversity principle (cite + multiple methods
where they exist) means we test both LPC methods for formants and both
pitch methods for voiced_pitch."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

import sadda


def _write_sine_wav(path: Path, freq: float, sample_rate: int, duration_s: float) -> None:
    n = int(sample_rate * duration_s)
    samples = np.sin(2 * np.pi * freq * np.arange(n) / sample_rate)
    pcm = (samples * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def _write_silent_wav(path: Path, sample_rate: int, duration_s: float) -> None:
    n = int(sample_rate * duration_s)
    pcm = (np.zeros(n, dtype=np.int16)).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def _write_vowel_wav(
    path: Path,
    sample_rate: int,
    duration_s: float,
    f0_hz: float,
    formant_freqs: list[float],
    formant_bws: list[float],
) -> None:
    """Source-filter vowel synthesis: impulse train at f0 → cascade of
    2nd-order resonators at the named formants."""
    n = int(sample_rate * duration_s)
    source = np.zeros(n, dtype=np.float64)
    period = max(1, int(sample_rate / f0_hz))
    source[::period] = 1.0
    signal = source.copy()
    for fi, bi in zip(formant_freqs, formant_bws):
        r = np.exp(-np.pi * bi / sample_rate)
        theta = 2 * np.pi * fi / sample_rate
        a1 = -2 * r * np.cos(theta)
        a2 = r * r
        y1, y2 = 0.0, 0.0
        out = np.empty_like(signal)
        for i, x in enumerate(signal):
            y = x - a1 * y1 - a2 * y2
            out[i] = y
            y2, y1 = y1, y
        signal = out
    signal = signal / np.max(np.abs(signal))
    pcm = (signal * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


# ---------------------------------------------------------------------------
# voiced_pitch
# ---------------------------------------------------------------------------

def test_voiced_pitch_returns_three_arrays() -> None:
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "sine.wav"
        _write_sine_wav(wav, freq=220.0, sample_rate=16_000, duration_s=0.5)
        audio = sadda.load_wav(str(wav))
        times, freqs, voicing = sadda.dsp.voiced_pitch(audio)
        assert times.dtype == np.float64
        assert freqs.dtype == np.float32
        assert voicing.dtype == np.float32
        assert len(times) == len(freqs) == len(voicing)


def test_voiced_pitch_default_method_is_windowed_autocorrelation() -> None:
    """Default method should produce sub-Hz precision on a clean sine
    (windowed_autocorrelation does parabolic interpolation; the naive
    'autocorrelation' method is capped at integer-lag precision)."""
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "sine.wav"
        _write_sine_wav(wav, freq=220.0, sample_rate=16_000, duration_s=0.5)
        audio = sadda.load_wav(str(wav))
        times, freqs, voicing = sadda.dsp.voiced_pitch(audio)
        mid = len(freqs) // 2
        assert abs(float(freqs[mid]) - 220.0) < 1.0
        assert float(voicing[mid]) > 0.7


def test_voiced_pitch_naive_method_also_works() -> None:
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "sine.wav"
        _write_sine_wav(wav, freq=200.0, sample_rate=16_000, duration_s=0.5)
        audio = sadda.load_wav(str(wav))
        times, freqs, voicing = sadda.dsp.voiced_pitch(
            audio, method="autocorrelation"
        )
        mid = len(freqs) // 2
        assert abs(float(freqs[mid]) - 200.0) < 5.0
        assert float(voicing[mid]) > 0.7


def test_voiced_pitch_boersma_method_tracks_clean_tone() -> None:
    """method='boersma' resolves to the faithful Boersma 1993 tracker
    (Praat `Sound: To Pitch (ac)…`). For a clean 220 Hz tone with
    fade-in/out, the median voiced f0 should land within 1 Hz."""
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "sine.wav"
        _write_sine_wav(wav, freq=220.0, sample_rate=16_000, duration_s=0.6)
        audio = sadda.load_wav(str(wav))
        times, freqs, voicing = sadda.dsp.voiced_pitch(audio, method="boersma")
        voiced = freqs[voicing >= 0.45]
        assert len(voiced) > 5, f"expected several voiced frames, got {len(voiced)}"
        import numpy as _np

        median_f0 = float(_np.median(voiced))
        assert abs(median_f0 - 220.0) < 1.0, (
            f"boersma median f0 = {median_f0:.3f} Hz, expected ~220"
        )


def test_voiced_pitch_unknown_method_raises_value_error() -> None:
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "sine.wav"
        _write_sine_wav(wav, freq=220.0, sample_rate=16_000, duration_s=0.2)
        audio = sadda.load_wav(str(wav))
        with pytest.raises(ValueError):
            sadda.dsp.voiced_pitch(audio, method="yin")


def test_voiced_pitch_silent_input_has_low_voicing() -> None:
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "silent.wav"
        _write_silent_wav(wav, 16_000, 0.5)
        audio = sadda.load_wav(str(wav))
        _, _, voicing = sadda.dsp.voiced_pitch(audio)
        assert np.all(voicing < 0.2)


def test_sadda_f0_alias_unchanged_phase0_contract() -> None:
    """sadda.f0 stays as the Phase-0 (times, freqs) 2-tuple form; the new
    3-tuple voicing API is on sadda.dsp.voiced_pitch."""
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "sine.wav"
        _write_sine_wav(wav, freq=220.0, sample_rate=16_000, duration_s=0.2)
        audio = sadda.load_wav(str(wav))
        result = sadda.f0(audio)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# formants
# ---------------------------------------------------------------------------

def test_formants_recovers_synthetic_vowel_with_burg() -> None:
    """A synthesised /a/-like vowel with formants at 700/1220/2600 Hz
    should be recovered within ~150 Hz by the Burg-LPC formant tracker."""
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "vowel.wav"
        targets = [700.0, 1220.0, 2600.0]
        _write_vowel_wav(wav, 16_000, 0.5, 110.0, targets, [50.0] * 3)
        audio = sadda.load_wav(str(wav))
        frames = sadda.dsp.formants(audio)
        # Pick a steady-state frame.
        mid = frames[len(frames) // 2]
        freqs = np.array(mid.frequencies)
        assert len(freqs) >= 3
        for i, target in enumerate(targets):
            assert abs(float(freqs[i]) - target) < 150.0, (
                f"formant {i + 1}: got {freqs[i]} Hz, expected ~{target} Hz; "
                f"all freqs {freqs.tolist()}"
            )


def test_formants_with_autocorrelation_method() -> None:
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "vowel.wav"
        targets = [700.0, 1220.0, 2600.0]
        _write_vowel_wav(wav, 16_000, 0.5, 110.0, targets, [50.0] * 3)
        audio = sadda.load_wav(str(wav))
        frames = sadda.dsp.formants(audio, method="autocorrelation")
        mid = frames[len(frames) // 2]
        assert len(mid.frequencies) >= 3
        for i, target in enumerate(targets):
            assert abs(float(mid.frequencies[i]) - target) < 250.0


def test_formants_unknown_method_raises_value_error() -> None:
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "vowel.wav"
        _write_vowel_wav(wav, 16_000, 0.3, 110.0, [500.0], [50.0])
        audio = sadda.load_wav(str(wav))
        with pytest.raises(ValueError):
            sadda.dsp.formants(audio, method="qcpfb")


def test_formant_frame_class_has_co_indexed_arrays() -> None:
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "vowel.wav"
        _write_vowel_wav(wav, 16_000, 0.3, 110.0, [700.0, 1500.0], [50.0, 50.0])
        audio = sadda.load_wav(str(wav))
        frames = sadda.dsp.formants(audio)
        f = frames[len(frames) // 2]
        assert isinstance(f.frequencies, np.ndarray)
        assert isinstance(f.bandwidths, np.ndarray)
        assert f.frequencies.dtype == np.float32
        assert f.bandwidths.dtype == np.float32
        assert len(f.frequencies) == len(f.bandwidths)


# ---------------------------------------------------------------------------
# mfcc
# ---------------------------------------------------------------------------

def test_mfcc_returns_2d_float32_array_with_default_shape() -> None:
    """Default n_mfcc=13; (n_frames, n_mfcc) layout matches librosa."""
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "sine.wav"
        _write_sine_wav(wav, freq=440.0, sample_rate=16_000, duration_s=1.0)
        audio = sadda.load_wav(str(wav))
        m = sadda.dsp.mfcc(audio)
        assert isinstance(m, np.ndarray)
        assert m.ndim == 2
        assert m.dtype == np.float32
        assert m.shape[1] == 13


def test_mfcc_higher_c0_for_audible_than_silent() -> None:
    """c0 (first cepstral coefficient) tracks log-energy. A loud sine should
    have a c0 well above silence at the same defaults."""
    with tempfile.TemporaryDirectory() as td:
        sine_path = Path(td) / "sine.wav"
        silent_path = Path(td) / "silent.wav"
        _write_sine_wav(sine_path, freq=440.0, sample_rate=16_000, duration_s=1.0)
        _write_silent_wav(silent_path, 16_000, 1.0)
        sine_audio = sadda.load_wav(str(sine_path))
        silent_audio = sadda.load_wav(str(silent_path))
        sine_m = sadda.dsp.mfcc(sine_audio)
        silent_m = sadda.dsp.mfcc(silent_audio)
        sine_c0 = float(sine_m[sine_m.shape[0] // 2, 0])
        silent_c0 = float(silent_m[silent_m.shape[0] // 2, 0])
        assert sine_c0 > silent_c0 + 20.0, (
            f"sine c0 ({sine_c0}) should exceed silent c0 ({silent_c0}) by 20+"
        )


def test_mfcc_custom_n_mfcc_param() -> None:
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "sine.wav"
        _write_sine_wav(wav, freq=440.0, sample_rate=16_000, duration_s=0.5)
        audio = sadda.load_wav(str(wav))
        m = sadda.dsp.mfcc(audio, n_mfcc=20)
        assert m.shape[1] == 20


# ---------------------------------------------------------------------------
# stability tier
# ---------------------------------------------------------------------------

def test_c2_surface_is_stable() -> None:
    from sadda._stability import get_stability

    for sym in (
        sadda.dsp.voiced_pitch,
        sadda.dsp.formants,
        sadda.dsp.mfcc,
        sadda.dsp.FormantFrame,
    ):
        assert get_stability(sym) == "stable", sym
