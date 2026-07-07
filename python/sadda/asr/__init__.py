"""sadda.asr — automatic speech recognition (audio → transcript).

The no-transcript front-end to forced alignment: recognize what was said, then
align it (:func:`sadda.align.align_auto`). First-class for unprompted
conversational / naturalistic recordings that have no transcript to begin with.

    import sadda
    tr = sadda.asr.transcribe(audio, sample_rate)   # needs `pip install "sadda[asr]"`
    print(tr.text)

Backends mirror :mod:`sadda.tts`: a structural :class:`ASRBackend` protocol + a
registry. The default is faster-whisper (opt-in ``sadda[asr]`` extra).

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

from .backends import (
    DEFAULT_BACKEND,
    SAMPLE_RATE,
    ASRBackend,
    FasterWhisperBackend,
    Transcription,
    TranscriptSegment,
    get_backend,
    list_backends,
    register_backend,
    transcribe,
)

__all__ = [
    "Transcription",
    "TranscriptSegment",
    "ASRBackend",
    "FasterWhisperBackend",
    "register_backend",
    "get_backend",
    "list_backends",
    "transcribe",
    "DEFAULT_BACKEND",
    "SAMPLE_RATE",
]
