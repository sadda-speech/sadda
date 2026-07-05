"""sadda.tts.pipeline — synthesis orchestration with content-hash caching.

This is the layer the voiceover use case actually calls. It resolves a backend,
synthesizes each segment (skipping work when an identical segment was already
synthesized), and optionally assembles the segments into a single narration
track with the requested inter-segment pauses.

The cache is keyed on a content hash of ``(backend, voice, rate, text)`` so a
doc build only re-synthesizes the lines that changed — making repeated builds
cheap and reproducible, which is the whole point for generated documentation.

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from sadda._stability import provisional

from .backends import SynthesisResult, TTSBackend, get_backend
from .script import NarrationScript, Segment

__all__ = ["ScriptResult", "cache_key", "synthesize", "synthesize_script", "concat_wavs"]

BackendArg = Union[str, TTSBackend, None]


@dataclass(frozen=True)
class ScriptResult:
    """The outcome of synthesizing a whole :class:`NarrationScript`."""

    segments: tuple[SynthesisResult, ...]
    combined: Optional[Path]
    total_duration_s: float


def _resolve_backend(backend: BackendArg) -> TTSBackend:
    if backend is None or isinstance(backend, str):
        return get_backend(backend)
    return backend


@provisional
def cache_key(text: str, *, backend: str, voice: Optional[str], rate: Optional[float]) -> str:
    """Deterministic hex digest identifying one synthesized span.

    Any change to the text, voice, rate, or backend name yields a different key;
    identical inputs always yield the same key.
    """
    h = hashlib.sha256()
    # NUL-separate the fields so no concatenation collision is possible.
    h.update(backend.encode("utf-8"))
    h.update(b"\0")
    h.update((voice or "").encode("utf-8"))
    h.update(b"\0")
    h.update(("" if rate is None else repr(float(rate))).encode("utf-8"))
    h.update(b"\0")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


@provisional
def concat_wavs(
    paths: Sequence[Union[str, Path]],
    out_path: Union[str, Path],
    *,
    pauses_s: Optional[Sequence[float]] = None,
) -> SynthesisResult:
    """Concatenate WAV files into one, inserting silence between them.

    ``pauses_s[i]`` is the silence appended *after* ``paths[i]``. All inputs must
    share the same sample rate, channel count, and sample width; a mismatch
    raises :class:`ValueError` rather than producing a corrupt file.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("concat_wavs requires at least one input WAV")
    if pauses_s is None:
        pauses_s = [0.0] * len(paths)
    if len(pauses_s) != len(paths):
        raise ValueError("pauses_s must be the same length as paths")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    params = None
    with wave.open(str(out_path), "wb") as out:
        for path, pause in zip(paths, pauses_s):
            with wave.open(str(path), "rb") as w:
                p = w.getparams()
                if params is None:
                    params = p
                    out.setnchannels(p.nchannels)
                    out.setsampwidth(p.sampwidth)
                    out.setframerate(p.framerate)
                elif (p.nchannels, p.sampwidth, p.framerate) != (
                    params.nchannels,
                    params.sampwidth,
                    params.framerate,
                ):
                    raise ValueError(
                        f"WAV format mismatch in {path.name}: "
                        f"{(p.nchannels, p.sampwidth, p.framerate)} != "
                        f"{(params.nchannels, params.sampwidth, params.framerate)}"
                    )
                out.writeframes(w.readframes(p.nframes))
            if pause and pause > 0:
                silent_frames = int(round(pause * params.framerate))
                out.writeframes(b"\x00" * (silent_frames * params.sampwidth * params.nchannels))

    sample_rate = params.framerate
    with wave.open(str(out_path), "rb") as w:
        duration_s = w.getnframes() / sample_rate if sample_rate else 0.0
    return SynthesisResult(path=out_path, sample_rate=sample_rate, duration_s=duration_s)


@provisional
def synthesize(
    text: str,
    out_path: Union[str, Path],
    *,
    backend: BackendArg = None,
    voice: Optional[str] = None,
    rate: Optional[float] = None,
) -> SynthesisResult:
    """One-shot: synthesize ``text`` to ``out_path``. Convenience over a backend."""
    return _resolve_backend(backend).synthesize(text, out_path, voice=voice, rate=rate)


@provisional
def synthesize_script(
    script: NarrationScript,
    out_dir: Union[str, Path],
    *,
    backend: BackendArg = None,
    cache_dir: Union[str, Path, None] = None,
    assemble: bool = True,
    combined_name: str = "narration.wav",
) -> ScriptResult:
    """Synthesize every segment of ``script``, caching by content hash.

    Each segment is synthesized into ``cache_dir`` (default ``out_dir/.cache``)
    under its :func:`cache_key`; a cache hit is reused verbatim, so re-running
    after editing one line only re-synthesizes that line. Per-segment WAVs are
    also copied to ``out_dir`` as ``segment_000.wav`` … for inspection.

    If ``assemble`` (the default), the segments are concatenated into
    ``out_dir/<combined_name>`` honoring each segment's ``pause_after_s``.
    """
    be = _resolve_backend(backend)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir is not None else out_dir / ".cache"
    cache.mkdir(parents=True, exist_ok=True)

    results: list[SynthesisResult] = []
    seg_paths: list[Path] = []
    pauses: list[float] = []
    for i, seg in enumerate(script):
        voice = script.resolved_voice(seg)
        rate = script.resolved_rate(seg)
        key = cache_key(seg.text, backend=be.name, voice=voice, rate=rate)
        cached = cache / f"{key}.wav"
        if not cached.exists():
            be.synthesize(seg.text, cached, voice=voice, rate=rate)

        seg_out = out_dir / f"segment_{i:03d}.wav"
        seg_out.write_bytes(cached.read_bytes())
        sample_rate, duration_s = _probe(seg_out)
        results.append(SynthesisResult(path=seg_out, sample_rate=sample_rate, duration_s=duration_s))
        seg_paths.append(seg_out)
        pauses.append(seg.pause_after_s)

    combined: Optional[Path] = None
    if assemble and seg_paths:
        combined = concat_wavs(seg_paths, out_dir / combined_name, pauses_s=pauses).path

    total = sum(r.duration_s for r in results) + sum(pauses)
    return ScriptResult(segments=tuple(results), combined=combined, total_duration_s=total)


def _probe(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as w:
        framerate = w.getframerate()
        nframes = w.getnframes()
    return framerate, (nframes / framerate if framerate else 0.0)
