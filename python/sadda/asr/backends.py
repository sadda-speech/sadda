"""sadda.asr.backends — automatic speech recognition backends (audio → text).

ASR is the *no-transcript* front-end to forced alignment: recognize what was said,
then align the transcript to the audio (see :func:`sadda.align.align_auto`). It is
first-class, not a mere convenience — plenty of speech data (unprompted
conversational and naturalistic recordings) has no transcript to start from.

Mirrors the :mod:`sadda.tts` design: a small structural :class:`ASRBackend`
protocol + a registry, so backends are pluggable and users can supply their own.
The default is **faster-whisper** (a CTranslate2 reimplementation of OpenAI
Whisper — MIT, torch-free at inference), behind the opt-in ``sadda[asr]`` extra;
if it isn't installed the backend raises an actionable error at construction.

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np

from sadda._stability import provisional

__all__ = [
    "TranscriptSegment",
    "Transcription",
    "ASRBackend",
    "FasterWhisperBackend",
    "register_backend",
    "get_backend",
    "list_backends",
    "transcribe",
    "DEFAULT_BACKEND",
    "SAMPLE_RATE",
]

DEFAULT_BACKEND = "faster-whisper"

#: The sample rate ASR models expect (Whisper is 16 kHz mono); input at any other
#: rate is resampled via :meth:`sadda.Audio.resample` before recognition.
SAMPLE_RATE = 16_000


# [docs:sadda.asr.TranscriptSegment]
@dataclass(frozen=True)
class TranscriptSegment:
    """One recognized span: its text and (approximate) time bounds."""

    text: str
    start_seconds: float
    end_seconds: float


# [docs:sadda.asr.Transcription]
@dataclass(frozen=True)
class Transcription:
    """The result of recognizing an utterance.

    ``text`` is the full transcript (the thing you feed the forced aligner);
    ``segments`` are the recognizer's own coarse time-stamped spans; ``language``
    is the detected/!specified language code (ISO 639-1, e.g. ``"en"``).
    """

    text: str
    segments: tuple[TranscriptSegment, ...]
    language: Optional[str] = None


@runtime_checkable
class ASRBackend(Protocol):
    """The contract every ASR backend implements.

    Structural (runtime-checkable): a backend is any object with a ``name`` and a
    :meth:`transcribe` method — no subclassing required, so a test mock or a
    user's own recognizer qualifies.
    """

    name: str

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        language: Optional[str] = None,
    ) -> Transcription:
        """Recognize mono ``audio`` → a :class:`Transcription`."""
        ...


def _to_model_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono float32 ``audio`` resampled to :data:`SAMPLE_RATE` (Whisper's rate)."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate == SAMPLE_RATE:
        return x
    import sadda  # local import avoids a package-level import cycle

    resampled = sadda.Audio.from_samples(x, sample_rate, channels=1).resample(SAMPLE_RATE)
    return np.asarray(resampled.samples, dtype=np.float32)


def _import_faster_whisper():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "faster-whisper is required for the default ASR backend. Install it "
            'with `pip install "sadda[asr]"`.'
        ) from exc
    return WhisperModel


# [docs:sadda.asr.FasterWhisperBackend]
@provisional
class FasterWhisperBackend:
    """Recognize speech with faster-whisper (CTranslate2 Whisper).

    Wraps ``faster_whisper.WhisperModel``. The model (``model_size``, default
    ``"base"``) is fetched from the HF Hub on first use and cached. Runs on CPU
    with int8 quantization by default; pass ``device="cuda"`` for a GPU.

    Args:
        model_size: A Whisper size (``"tiny"``/``"base"``/``"small"``/``"medium"``/
            ``"large-v3"``) or a path to a converted model.
        device: ``"cpu"`` (default) or ``"cuda"``.
        compute_type: CTranslate2 compute type (default ``"int8"``).
        word_timestamps: Ask the recognizer for word-level times (slower).
    """

    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "base",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        word_timestamps: bool = False,
        **model_kwargs: object,
    ) -> None:
        whisper_model = _import_faster_whisper()
        self._model = whisper_model(
            model_size, device=device, compute_type=compute_type, **model_kwargs
        )
        self._word_timestamps = word_timestamps

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        language: Optional[str] = None,
    ) -> Transcription:
        x = _to_model_rate(audio, sample_rate)
        segments, info = self._model.transcribe(
            x, language=language, word_timestamps=self._word_timestamps
        )
        segs = tuple(
            TranscriptSegment(text=s.text, start_seconds=s.start, end_seconds=s.end)
            for s in segments  # a generator — consuming it runs the recognizer
        )
        text = "".join(s.text for s in segs).strip()
        return Transcription(text=text, segments=segs, language=getattr(info, "language", language))


# name -> zero/kw-arg factory returning an ASRBackend instance.
_REGISTRY: dict[str, Callable[..., ASRBackend]] = {
    "faster-whisper": FasterWhisperBackend,
}


# [docs:sadda.asr.register_backend]
@provisional
def register_backend(name: str, factory: Callable[..., ASRBackend]) -> None:
    """Register a backend factory under ``name`` (overwrites any existing one)."""
    _REGISTRY[name] = factory


# [docs:sadda.asr.list_backends]
@provisional
def list_backends() -> list[str]:
    """Return the registered backend names, sorted."""
    return sorted(_REGISTRY)


# [docs:sadda.asr.get_backend]
@provisional
def get_backend(name: Optional[str] = None, **kwargs: object) -> ASRBackend:
    """Instantiate a backend by name.

    ``name`` defaults to ``$SADDA_ASR_BACKEND`` then :data:`DEFAULT_BACKEND`.
    Extra keyword arguments are forwarded to the backend's constructor.
    """
    if name is None:
        name = os.environ.get("SADDA_ASR_BACKEND", DEFAULT_BACKEND)
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown ASR backend {name!r}; registered: {list_backends()}") from None
    return factory(**kwargs)


# [docs:sadda.asr.transcribe]
@provisional
def transcribe(
    audio: np.ndarray,
    sample_rate: int,
    *,
    backend: Optional[ASRBackend] = None,
    language: Optional[str] = None,
    **backend_kwargs: object,
) -> Transcription:
    """Recognize ``audio`` to a :class:`Transcription`.

    ``backend`` is an :class:`ASRBackend` (default: a fresh :data:`DEFAULT_BACKEND`
    built with ``backend_kwargs``). ``language`` fixes the language (ISO 639-1);
    omit to let the recognizer detect it.
    """
    if backend is None:
        backend = get_backend(**backend_kwargs)
    return backend.transcribe(audio, sample_rate, language=language)
