"""sadda.tts.backends — the pluggable synthesis-backend layer.

A backend turns a string of text into a WAV file. The :class:`TTSBackend`
protocol is the whole contract; everything else in :mod:`sadda.tts` (caching,
script assembly, the pipeline) is backend-agnostic and speaks only to it.

Backends shipped / registered here:

- ``"espeak-ng"`` — :class:`EspeakNgBackend`, a subprocess wrapper around the
  system `eSpeak NG <https://github.com/espeak-ng/espeak-ng>`_ formant
  synthesizer. Always-available (no Python dependency, no model download),
  reproducible, and phonetically apt — but robotic. It is the dependency-free
  default and the CI-safe reference implementation.
- ``"kokoro"`` — the planned high-quality neural default (Kokoro-82M, Apache
  2.0, CPU-capable). **Not yet wired**: its inference dependency lives behind a
  future ``sadda[tts]`` extra, pending the decision recorded in the 2026-07-05
  DEVLOG entry. Requesting it today raises a clear, actionable error rather than
  guessing at an unverified API.

Cloud backends (ElevenLabs / OpenAI) are intended to plug in the same way, as
opt-in add-ons, once the local path is settled.

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, Union, runtime_checkable

from sadda._stability import provisional

__all__ = [
    "SynthesisResult",
    "TTSBackend",
    "EspeakNgBackend",
    "register_backend",
    "get_backend",
    "list_backends",
    "DEFAULT_BACKEND",
]

DEFAULT_BACKEND = "espeak-ng"


# [docs:sadda.tts.SynthesisResult]
@dataclass(frozen=True)
class SynthesisResult:
    """The outcome of synthesizing one span of text to a WAV file."""

    path: Path
    sample_rate: int
    duration_s: float


@runtime_checkable
class TTSBackend(Protocol):
    """The contract every synthesis backend implements.

    A backend is any object with a ``name`` and a :meth:`synthesize` method. It
    need not subclass anything — this is a structural (runtime-checkable)
    protocol so callers can pass their own duck-typed backend.
    """

    name: str

    def synthesize(
        self,
        text: str,
        out_path: Union[str, Path],
        *,
        voice: Optional[str] = None,
        rate: Optional[float] = None,
    ) -> SynthesisResult:
        """Synthesize ``text`` to a WAV at ``out_path`` and describe the result."""
        ...


def _probe_wav(path: Path) -> tuple[int, float]:
    """Return ``(sample_rate, duration_s)`` for a WAV file, via stdlib ``wave``."""
    with wave.open(str(path), "rb") as w:
        framerate = w.getframerate()
        nframes = w.getnframes()
    return framerate, (nframes / framerate if framerate else 0.0)


# [docs:sadda.tts.EspeakNgBackend]
@provisional
class EspeakNgBackend:
    """Synthesize speech by shelling out to the ``espeak-ng`` binary.

    eSpeak NG writes 22.05 kHz mono 16-bit PCM WAV. It is a formant synthesizer:
    robotic, but zero-dependency, fully offline, deterministic, and available on
    every platform — which makes it the reference backend for tests and doc
    builds where reproducibility beats naturalness.

    Args:
        binary: Path to the ``espeak-ng`` executable. Defaults to whatever is on
            ``PATH``. Raises :class:`FileNotFoundError` at construction if none
            is found.
        base_words_per_minute: The rate that a ``rate`` multiplier of ``1.0``
            maps to. eSpeak's own default is 175 wpm.
    """

    name = "espeak-ng"

    # eSpeak clamps speed to roughly this range; mirror it so a wild multiplier
    # produces a valid command rather than an error.
    _MIN_WPM = 80
    _MAX_WPM = 450

    def __init__(
        self,
        binary: Optional[Union[str, Path]] = None,
        *,
        base_words_per_minute: int = 175,
    ) -> None:
        resolved = str(binary) if binary is not None else shutil.which("espeak-ng")
        if resolved is None or (binary is not None and not Path(resolved).exists()):
            raise FileNotFoundError(
                "espeak-ng executable not found. Install it (e.g. "
                "`apt install espeak-ng`, `brew install espeak-ng`) or pass "
                "`binary=`."
            )
        self._binary = resolved
        self._base_wpm = base_words_per_minute

    def _wpm(self, rate: Optional[float]) -> int:
        wpm = round(self._base_wpm * rate) if rate is not None else self._base_wpm
        return max(self._MIN_WPM, min(self._MAX_WPM, wpm))

    # [docs:sadda.tts.EspeakNgBackend.synthesize]
    def synthesize(
        self,
        text: str,
        out_path: Union[str, Path],
        *,
        voice: Optional[str] = None,
        rate: Optional[float] = None,
    ) -> SynthesisResult:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Feed text via a temp file (`-f`) rather than as an argument, to sidestep
        # shell-quoting and argument-length limits on long narration.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", encoding="utf-8", delete=False
        ) as tf:
            tf.write(text)
            text_file = tf.name
        try:
            cmd = [
                self._binary,
                "-w",
                str(out_path),
                "-f",
                text_file,
                "-s",
                str(self._wpm(rate)),
            ]
            if voice is not None:
                cmd += ["-v", voice]
            subprocess.run(cmd, check=True, capture_output=True)
        finally:
            os.unlink(text_file)

        sample_rate, duration_s = _probe_wav(out_path)
        return SynthesisResult(path=out_path, sample_rate=sample_rate, duration_s=duration_s)


def _kokoro_pending(**_kwargs: object) -> TTSBackend:
    raise NotImplementedError(
        "The 'kokoro' backend is not yet wired. It is the planned high-quality "
        "neural default (Kokoro-82M, Apache 2.0), but its inference dependency "
        "is still pending the `sadda[tts]` extra decision — see the 2026-07-05 "
        "DEVLOG entry. Use backend='espeak-ng' for now."
    )


# name -> zero/kw-arg factory returning a TTSBackend instance.
_REGISTRY: dict[str, Callable[..., TTSBackend]] = {
    "espeak-ng": EspeakNgBackend,
    "kokoro": _kokoro_pending,
}


# [docs:sadda.tts.register_backend]
@provisional
def register_backend(name: str, factory: Callable[..., TTSBackend]) -> None:
    """Register a backend factory under ``name`` (overwrites any existing one)."""
    _REGISTRY[name] = factory


# [docs:sadda.tts.list_backends]
@provisional
def list_backends() -> list[str]:
    """Return the registered backend names, sorted."""
    return sorted(_REGISTRY)


# [docs:sadda.tts.get_backend]
@provisional
def get_backend(name: Optional[str] = None, **kwargs: object) -> TTSBackend:
    """Instantiate a backend by name.

    ``name`` defaults to ``$SADDA_TTS_BACKEND`` then :data:`DEFAULT_BACKEND`.
    Extra keyword arguments are forwarded to the backend's constructor.
    """
    if name is None:
        name = os.environ.get("SADDA_TTS_BACKEND", DEFAULT_BACKEND)
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown TTS backend {name!r}; registered: {list_backends()}"
        ) from None
    return factory(**kwargs)
