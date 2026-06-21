"""Python-surface tests for the C1 DSP namespace (sadda.dsp.*)."""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

import sadda


def _write_sine_wav(path: Path, freq: float, sample_rate: int, duration_seconds: float) -> None:
    n = int(sample_rate * duration_seconds)
    samples = np.sin(2 * np.pi * freq * np.arange(n) / sample_rate)
    pcm = (samples * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def test_dsp_namespace_is_a_submodule() -> None:
    import types

    assert isinstance(sadda.dsp, types.ModuleType)
    expected = {"hann", "hamming", "blackman", "gaussian", "kaiser",
                "stft", "spectrogram", "intensity", "f0"}
    assert expected.issubset(set(dir(sadda.dsp)))


def test_top_level_f0_alias_points_at_dsp_f0() -> None:
    """Phase-0 sadda.f0 stays as a back-compat alias for sadda.dsp.f0.
    Both must call the same underlying PyO3 function."""
    assert sadda.f0 is sadda.dsp.f0


def test_hann_endpoints_and_peak_match_formula() -> None:
    w = sadda.dsp.hann(9)
    assert isinstance(w, np.ndarray)
    assert w.dtype == np.float32
    assert abs(float(w[0])) < 1e-6
    assert abs(float(w[8])) < 1e-6
    assert abs(float(w[4]) - 1.0) < 1e-6


def test_hamming_endpoints_match_formula() -> None:
    w = sadda.dsp.hamming(11)
    assert abs(float(w[0]) - 0.08) < 1e-5
    assert abs(float(w[10]) - 0.08) < 1e-5


def test_blackman_endpoints_are_zero() -> None:
    w = sadda.dsp.blackman(13)
    assert abs(float(w[0])) < 1e-5
    assert abs(float(w[12])) < 1e-5


def test_gaussian_peak_at_center() -> None:
    w = sadda.dsp.gaussian(11, 2.0)
    assert abs(float(w[5]) - 1.0) < 1e-6


def test_kaiser_beta_zero_is_rectangular() -> None:
    w = sadda.dsp.kaiser(7, 0.0)
    np.testing.assert_allclose(w, np.ones(7, dtype=np.float32), atol=1e-5)


def test_stft_of_pure_sine_peaks_at_expected_bin() -> None:
    sr = 16_000
    freq = 1_000.0
    n = sr // 2
    samples = np.sin(2 * np.pi * freq * np.arange(n) / sr).astype(np.float32)
    Z = sadda.dsp.stft(samples, frame_size=1024, hop_size=256)
    assert Z.dtype == np.complex64
    assert Z.shape[1] == 1024 // 2 + 1  # n_freq_bins
    expected_bin = round(freq / (sr / 1024))
    mags = np.abs(Z)
    for f in range(Z.shape[0]):
        peak = int(np.argmax(mags[f]))
        assert abs(peak - expected_bin) <= 1, f"frame {f}: peak {peak}, expected near {expected_bin}"


def test_spectrogram_shape_is_freq_first() -> None:
    sr = 16_000
    samples = np.zeros(8_000, dtype=np.float32)
    S = sadda.dsp.spectrogram(samples, frame_size=512, hop_size=128)
    assert S.dtype == np.float32
    # (n_freq_bins, n_frames)
    assert S.shape[0] == 512 // 2 + 1


def test_spectrogram_of_pure_sine_peaks_at_expected_bin() -> None:
    sr = 16_000
    freq = 1_000.0
    n = sr // 2
    samples = np.sin(2 * np.pi * freq * np.arange(n) / sr).astype(np.float32)
    S = sadda.dsp.spectrogram(samples, frame_size=1024, hop_size=256)
    expected_bin = round(freq / (sr / 1024))
    # Pick a middle frame.
    mid = S.shape[1] // 2
    peak = int(np.argmax(S[:, mid]))
    assert abs(peak - expected_bin) <= 1


def test_intensity_of_unit_sine_is_one_over_sqrt_two() -> None:
    sr = 16_000
    freq = 440.0
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "sine.wav"
        _write_sine_wav(wav, freq, sr, 1.0)
        audio = sadda.load_wav(str(wav))
        times, rms, db = sadda.dsp.intensity(audio)
        assert times.dtype == np.float64
        assert rms.dtype == np.float32
        assert db.dtype == np.float32
        assert len(times) == len(rms) == len(db)
        # Mid frame: full-scale sine has RMS = 1/√2 = 0.7071...
        mid = len(rms) // 2
        assert abs(float(rms[mid]) - 1 / np.sqrt(2)) < 0.02
        # dB-FS = 20 log10(1/√2) ≈ -3.01 dB
        assert abs(float(db[mid]) - (-3.0103)) < 0.2


def test_mono_returns_an_audio_that_flows_back_into_dsp() -> None:
    sr = 16_000
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "sine.wav"
        _write_sine_wav(wav, 440.0, sr, 0.5)
        audio = sadda.load_wav(str(wav))

        mono = audio.mono()
        # mono() returns an Audio (not a raw array), so it composes with dsp.*.
        assert isinstance(mono, sadda.Audio)
        assert mono.channels == 1
        assert mono.sample_rate == sr
        assert mono.n_frames == audio.n_frames
        # Raw samples are still one hop away.
        assert isinstance(mono.samples, np.ndarray)
        # The whole point: the result can be passed straight back into dsp.*.
        ltas = sadda.dsp.ltas(mono)
        assert ltas.levels_db.size > 0


def test_stft_window_length_mismatch_raises_value_error() -> None:
    samples = np.zeros(2048, dtype=np.float32)
    wrong_window = np.ones(512, dtype=np.float32)
    with pytest.raises(ValueError):
        sadda.dsp.stft(samples, frame_size=1024, hop_size=256, window=wrong_window)


def test_dsp_surface_is_stable() -> None:
    from sadda._stability import get_stability

    for sym in (
        sadda.dsp.hann,
        sadda.dsp.hamming,
        sadda.dsp.blackman,
        sadda.dsp.gaussian,
        sadda.dsp.kaiser,
        sadda.dsp.stft,
        sadda.dsp.spectrogram,
        sadda.dsp.intensity,
        sadda.dsp.f0,
    ):
        assert get_stability(sym) == "stable", sym
