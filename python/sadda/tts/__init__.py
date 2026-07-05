"""sadda.tts — text-to-speech synthesis (voiceover + general TTS).

A backend-agnostic speech-synthesis toolkit. The immediate driver is voiceover
for generated documentation (narration script -> cached WAVs -> assembled
track), but the surface is deliberately general so it also serves ad-hoc TTS:

    import sadda

    # one-shot
    sadda.tts.synthesize("Hello from sadda.", "hello.wav")

    # a narration script (blank-line-separated paragraphs -> segments)
    script = sadda.tts.parse_script(open("intro.txt").read())
    result = sadda.tts.synthesize_script(script, out_dir="build/vo")
    print(result.combined, result.total_duration_s)

Synthesis goes through a pluggable :class:`~sadda.tts.backends.TTSBackend`. The
default, ``"espeak-ng"``, is a zero-dependency offline formant synthesizer —
robotic but reproducible and CI-safe. A high-quality neural default (Kokoro,
Apache 2.0) and opt-in cloud backends (ElevenLabs / OpenAI) are designed to plug
in the same way; see :mod:`sadda.tts.backends` and the 2026-07-05 DEVLOG entry.

Screencast/gif capture and audio/video muxing — the rest of the doc-automation
vision — are out of scope for this module and tracked separately in BACKLOG.

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

from .backends import (
    DEFAULT_BACKEND,
    EspeakNgBackend,
    SynthesisResult,
    TTSBackend,
    get_backend,
    list_backends,
    register_backend,
)
from .pipeline import (
    ScriptResult,
    cache_key,
    concat_wavs,
    synthesize,
    synthesize_script,
)
from .script import NarrationScript, Segment, load_script, parse_script

__all__ = [
    # backends
    "TTSBackend",
    "SynthesisResult",
    "EspeakNgBackend",
    "get_backend",
    "list_backends",
    "register_backend",
    "DEFAULT_BACKEND",
    # script model
    "Segment",
    "NarrationScript",
    "parse_script",
    "load_script",
    # pipeline
    "synthesize",
    "synthesize_script",
    "concat_wavs",
    "cache_key",
    "ScriptResult",
]
