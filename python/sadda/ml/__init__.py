"""sadda.ml — ML inference (Phase 3 E11).

ONNX-model inference over audio. The first model is a bundled
`Silero VAD <https://github.com/snakers4/silero-vad>`_ (voice-activity
detection); :func:`vad` returns a speech probability per ~32 ms window and
:func:`speech_segments` merges those into speech regions.

ONNX Runtime is loaded at runtime (not bundled): if it isn't found, these
functions raise a clear error rather than crashing. Point ``ORT_DYLIB_PATH``
at a ``libonnxruntime`` shared library (the desktop app ships one). See the
2026-05-26 E11 DEVLOG entry.

Typical usage::

    import sadda

    audio = sadda.load_wav("utterance.wav")

    # per-window speech probability (NumPy arrays)
    times, probs = sadda.ml.vad(audio)

    # merged speech regions, as (start_s, end_s) tuples
    for start, end in sadda.ml.speech_segments(audio, threshold=0.5):
        print(f"speech {start:.2f}-{end:.2f}s")

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

from typing import Optional

from sadda import _native
from sadda._stability import provisional

__all__ = [
    "Model",
    "load_model",
    "speech_segments",
    "vad",
]

# Return type of load_model; re-exported for type-reference + introspection.
Model = _native.ml.Model


@provisional
def load_model(id):  # noqa: A002
    """Resolve a model by id, returning a :class:`Model`.

    ``id`` is one of: ``"sadda/<name>[@version]"`` (curated registry,
    falling back to the bundled set), ``"local://<path>"`` (a model
    directory with a ``model.toml``, or a bare model file), or
    ``"hf://<repo>"`` (HuggingFace passthrough — arrives in a later
    release). The returned model exposes ``.vad(audio)`` plus ``.id`` /
    ``.version`` / ``.kind`` / ``.weights_checksum`` metadata.
    """
    return _native.ml.load_model(id)


@provisional
def vad(audio, *, model_path: Optional[str] = None):
    """Run Silero VAD over ``audio``.

    Returns ``(times, speech_probs)`` as NumPy arrays — one entry per
    ~32 ms window (the audio is mono-mixed and resampled to 16 kHz).
    Uses the bundled model unless ``model_path`` points at another ONNX
    VAD model. Raises if ONNX Runtime isn't available.
    """
    return _native.ml.vad(audio, model_path=model_path)


@provisional
def speech_segments(audio, *, threshold: float = 0.5, model_path: Optional[str] = None):
    """Speech regions in ``audio`` as ``(start_seconds, end_seconds)``.

    Runs :func:`vad`, then merges consecutive windows whose probability is
    ``>= threshold``. Uses the bundled model unless ``model_path`` is given.
    """
    return _native.ml.speech_segments(audio, threshold=threshold, model_path=model_path)
