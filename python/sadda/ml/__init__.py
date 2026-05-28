"""sadda.ml — ML inference (Phase 3 E11).

ONNX-model inference over audio. The first model is a bundled
`Silero VAD <https://github.com/snakers4/silero-vad>`_ (voice-activity
detection); :func:`vad` returns a speech probability per ~32 ms window and
:func:`speech_segments` merges those into speech regions.

ONNX Runtime is loaded at runtime (not linked into the wheel). To make
this transparent for ``pip install sadda[ml]`` users, this module
auto-discovers a pip- or conda-installed ``onnxruntime`` at import time
and sets ``ORT_DYLIB_PATH`` to its bundled library. A user-set
``ORT_DYLIB_PATH`` is never overridden. If neither is available, the
inference functions raise a clear error rather than crashing.

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

import os
import sys
from pathlib import Path
from typing import Optional

from sadda import _native
from sadda._stability import provisional


def _discover_ort_dylib() -> Optional[str]:
    """Locate ``libonnxruntime`` shipped inside a pip- or conda-installed
    ``onnxruntime`` package. Returns the absolute path, or ``None`` if the
    package isn't importable or no library is found.

    The PyPI/conda ``onnxruntime`` wheel ships its runtime under
    ``<site-packages>/onnxruntime/capi/`` — ``libonnxruntime.so.<ver>`` on
    Linux, ``libonnxruntime.dylib`` on macOS, ``onnxruntime.dll`` on
    Windows — alongside the ``libonnxruntime_providers_shared`` shim, which
    we deliberately exclude (the engine's probe rejects the shim with a
    pointed error, but skipping it here saves the round trip).
    """
    try:
        import onnxruntime  # type: ignore[import-untyped]
    except ImportError:
        return None
    init = getattr(onnxruntime, "__file__", None)
    if not init:
        return None
    capi = Path(init).parent / "capi"
    if not capi.is_dir():
        return None
    if sys.platform == "win32":
        candidates = list(capi.glob("onnxruntime*.dll"))
    elif sys.platform == "darwin":
        candidates = list(capi.glob("libonnxruntime*.dylib"))
    else:
        candidates = list(capi.glob("libonnxruntime.so*"))
    candidates = [p for p in candidates if "providers_shared" not in p.name]
    if not candidates:
        return None
    # The wheel ships exactly one runtime library; prefer the longest
    # filename (a versioned name like libonnxruntime.so.1.26.0 over a
    # plain libonnxruntime.so) for stability across symlink layouts.
    candidates.sort(key=lambda p: len(p.name), reverse=True)
    return str(candidates[0])


# Set ORT_DYLIB_PATH from the installed onnxruntime, but never override
# a user-set value. Failure is silent — the engine raises a clean
# "set ORT_DYLIB_PATH" error at vad-call time if neither path resolves.
if "ORT_DYLIB_PATH" not in os.environ:
    _found = _discover_ort_dylib()
    if _found is not None:
        os.environ["ORT_DYLIB_PATH"] = _found

__all__ = [
    "Model",
    "get_model",
    "install_model",
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
def install_model(src_dir, *, root=None):
    """Install a model directory (a ``model.toml`` + its files) into the
    store by copying it in — how the bundled set seeds the cache and where
    a fetched model lands. Returns the installed :class:`Model`."""
    return _native.ml.install_model(src_dir, root=root)


@provisional
def get_model(id, version, *, root=None):  # noqa: A002
    """The model with this ``id`` + ``version`` in the store (the per-user
    cache by default, or an explicit ``root``), or ``None``."""
    return _native.ml.get_model(id, version, root=root)


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
