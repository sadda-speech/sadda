"""Python-surface tests for sadda.asr (A4 — ASR / no-transcript path).

The real recognizer (faster-whisper) is gated — it's a heavy opt-in extra with a
model download, absent in CI. The seams — the backend protocol/registry, the
resample-to-model-rate helper, the not-installed error, and the ASR→align
orchestration (`align_auto`) — run everywhere via a mock backend.
"""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest

import sadda
from sadda import asr
from sadda.align import Emissions, align_auto

pytestmark = pytest.mark.filterwarnings("ignore::sadda.ProvisionalAPIWarning")


class _MockASR:
    """A trivial recognizer that always returns a fixed transcript."""

    name = "mock"

    def __init__(self, text: str = "hi") -> None:
        self._text = text
        self.calls: list[int] = []

    def transcribe(self, audio, sample_rate, *, language=None):
        self.calls.append(sample_rate)
        return asr.Transcription(text=self._text, segments=(), language=language or "en")


class _BlockModel:
    """Mock acoustic model favouring a given phone sequence (see test_align)."""

    def __init__(self, phones, frames_per_phone=3, frame_rate=100.0):
        self._phones = phones
        self._fpp = frames_per_phone
        self._fr = frame_rate
        uniq = list(dict.fromkeys(phones))
        self._vocab = {p: i + 1 for i, p in enumerate(uniq)}

    def emissions(self, audio, sample_rate):
        n = len(self._vocab) + 1
        hi, lo = math.log(0.9), math.log(0.1 / (n - 1))
        rows = []
        for p in self._phones:
            hot = self._vocab[p]
            for _ in range(self._fpp):
                rows.append([hi if c == hot else lo for c in range(n)])
        return Emissions(np.array(rows, dtype=np.float32), self._vocab, self._fr, 0)


# --- protocol + registry ---


def test_mock_backend_satisfies_protocol() -> None:
    assert isinstance(_MockASR(), asr.ASRBackend)


def test_registry_roundtrip() -> None:
    asr.register_backend("mock-test", _MockASR)
    assert "mock-test" in asr.list_backends()
    assert isinstance(asr.get_backend("mock-test"), _MockASR)


def test_get_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown ASR backend"):
        asr.get_backend("no-such-backend")


# --- transcribe() with a passed backend ---


def test_transcribe_uses_given_backend() -> None:
    tr = asr.transcribe(np.zeros(16000, np.float32), 16000, backend=_MockASR("hello world"))
    assert tr.text == "hello world"
    assert tr.language == "en"


# --- resample seam (ungated) ---


def test_to_model_rate_resamples_off_rate_audio() -> None:
    from sadda.asr.backends import SAMPLE_RATE, _to_model_rate

    x16 = np.linspace(-0.3, 0.3, SAMPLE_RATE, dtype=np.float32)
    assert np.allclose(_to_model_rate(x16, SAMPLE_RATE), x16, atol=1e-6)

    t = np.arange(8000, dtype=np.float32) / 8000.0
    y = (0.5 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)
    out = _to_model_rate(y, 8000)
    assert out.dtype == np.float32
    assert abs(len(out) - len(y) * SAMPLE_RATE // 8000) <= 2


# --- not-installed error (no faster-whisper needed) ---


def _fw_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(_fw_available(), reason="faster-whisper is installed")
def test_faster_whisper_backend_errors_clearly_when_absent() -> None:
    with pytest.raises(ImportError, match='sadda\\[asr\\]'):
        asr.FasterWhisperBackend()


@pytest.mark.skipif(not _fw_available(), reason="faster-whisper not installed")
def test_faster_whisper_transcribes_smoke() -> None:
    # Only where faster-whisper is installed: recognize a real clip named by env.
    import os
    import wave

    path = os.environ.get("SADDA_TEST_ASR_AUDIO")
    if not path:
        pytest.skip("SADDA_TEST_ASR_AUDIO not set")
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    tr = asr.transcribe(audio, sr, backend=asr.FasterWhisperBackend("tiny"))
    assert isinstance(tr.text, str) and tr.language


# --- align_auto orchestration (ASR -> transcript -> align) ---


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")
def test_align_auto_recognizes_then_aligns() -> None:
    phones = list(sadda.align.phonemize("hi").words[0].phones)
    model = _BlockModel(phones, frames_per_phone=3, frame_rate=100.0)
    recognizer = _MockASR("hi")
    audio = np.zeros(16000, dtype=np.float32)

    al = align_auto(audio, 16000, model=model, asr_backend=recognizer)

    assert recognizer.calls == [16000]  # ASR ran
    assert [w.text for w in al.words] == ["hi"]  # and its transcript was aligned
    assert [p.label for p in al.phones] == phones


def test_align_auto_rejects_empty_transcript() -> None:
    model = _BlockModel(["a"])
    with pytest.raises(ValueError, match="empty transcript"):
        align_auto(np.zeros(16000, np.float32), 16000, model=model, asr_backend=_MockASR(""))
