"""E11 — ML inference (bundled Silero VAD), Python surface.

The inference tests need a runtime ONNX Runtime (`ORT_DYLIB_PATH`); they
skip cleanly when it isn't available, so CI (which has no ORT) stays
green. Run locally with `ORT_DYLIB_PATH=…/libonnxruntime.so` for the real
assertions.
"""

from __future__ import annotations

import warnings
import wave

import numpy as np
import pytest
import sadda


def _silence_wav(path, seconds: float = 1.0, sr: int = 16_000) -> None:
    """Writes a mono 16-bit PCM WAV of silence."""
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)


def _ort_or_skip(fn):
    """Calls `fn`, skipping the test if ONNX Runtime isn't available."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence the @provisional warning
        try:
            return fn()
        except RuntimeError as e:
            msg = str(e).lower()
            if "ml error" in msg or "onnx" in msg:
                pytest.skip(f"ONNX Runtime not available: {e}")
            raise


def test_ml_functions_are_provisional() -> None:
    # Runs without ONNX Runtime — just checks the surface is wired + tiered.
    from sadda._stability import get_stability

    assert get_stability(sadda.ml.vad) == "provisional"
    assert get_stability(sadda.ml.speech_segments) == "provisional"


def test_vad_on_silence(tmp_path) -> None:
    wav = tmp_path / "silence.wav"
    _silence_wav(wav)
    audio = sadda.load_wav(str(wav))
    times, probs = _ort_or_skip(lambda: sadda.ml.vad(audio))
    assert len(times) == len(probs)
    assert len(times) > 0
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    # Silence is overwhelmingly non-speech.
    assert float(np.mean(probs)) < 0.3


def test_speech_segments_on_silence(tmp_path) -> None:
    wav = tmp_path / "silence.wav"
    _silence_wav(wav)
    audio = sadda.load_wav(str(wav))
    segs = _ort_or_skip(lambda: sadda.ml.speech_segments(audio, threshold=0.5))
    assert isinstance(segs, list)
    # No speech in silence.
    assert segs == []
