"""sadda.dsp — foundational DSP toolkit.

Pure-function API over NumPy float32 arrays. Window functions, STFT,
spectrogram, intensity, and the relocated `f0` from Phase 0 all live here.
Stability tier: STABLE (per the 2026-05-18 Python API surface DEVLOG entry).

The top-level `sadda.f0` stays as a Phase-0 back-compat alias for the same
function.
"""

from __future__ import annotations

from typing import Optional

from sadda import _native
from sadda._stability import provisional, stable

__all__ = [
    "FormantFrame",
    "Ltas",
    "MfccParams",
    "MfccPreset",
    "blackman",
    "builtin_mfcc_presets",
    "delete_mfcc_preset",
    "f0",
    "formants",
    "gaussian",
    "hamming",
    "hann",
    "intensity",
    "kaiser",
    "log_mel_whisper",
    "ltas",
    "mfcc",
    "mfcc_preset",
    "mfcc_preset_store",
    "mfcc_presets",
    "mfcc_user_presets",
    "save_mfcc_preset",
    "spectrogram",
    "stft",
    "voiced_pitch",
]

hann = stable(_native.hann)
hamming = stable(_native.hamming)
blackman = stable(_native.blackman)
gaussian = stable(_native.gaussian)
kaiser = stable(_native.kaiser)
stft = stable(_native.stft)
spectrogram = stable(_native.spectrogram)
intensity = stable(_native.intensity)
f0 = stable(_native.f0)

# C2 surface.
voiced_pitch = stable(_native.voiced_pitch)
formants = stable(_native.formants)


@stable
def mfcc(
    audio,
    *,
    params: "Optional[MfccParams]" = None,
    frame_size_seconds: float = 0.025,
    hop_seconds: float = 0.010,
    n_mels: int = 40,
    n_mfcc: int = 13,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    method: str = "librosa",
):
    """Mel-frequency cepstral coefficients, shape ``(n_frames, n_mfcc)``.

    Two ways to specify the computation:

    - **By named method** (default): ``method`` is one of ``"librosa"``
      (default), ``"kaldi"``, or ``"praat"`` — each a faithful reproduction of
      that reference (see ``MfccMethod``). The other keyword args set the
      common analysis parameters.
    - **By full parameter set**: pass ``params=`` an :class:`MfccParams` (from a
      preset, optionally edited with ``.replace(...)``). When ``params`` is
      given it fully determines the computation and the ``method`` / ``n_mels``
      / ``frame_size_seconds`` / … keywords are ignored.

    ``f_max`` defaults to the Nyquist frequency (``sample_rate / 2``)."""
    if params is not None:
        return _native.mfcc_preset.compute(audio, params)
    return _native.mfcc(
        audio,
        frame_size_seconds=frame_size_seconds,
        hop_seconds=hop_seconds,
        n_mels=n_mels,
        n_mfcc=n_mfcc,
        f_min=f_min,
        f_max=f_max,
        method=method,
    )


# MFCC parameter/preset registry (roadmap item 3/4). The params + preset types
# and the on-disk store. PROVISIONAL — the preset schema may still change.
# Classes are re-exported raw (not `@provisional`): that decorator wraps
# `__init__`, which breaks PyO3's `__new__`-based construction — so the
# provisional tier is carried by the store functions below, as in sadda.refdist.
MfccParams = _native.mfcc_preset.MfccParams
MfccPreset = _native.mfcc_preset.MfccPreset


@provisional
def mfcc_presets(*, root: Optional[str] = None) -> "list[MfccPreset]":
    """All MFCC presets: the built-in authoritative set (librosa / kaldi /
    praat) followed by the user's on-disk presets."""
    return _native.mfcc_preset.list_all(root=root)


@provisional
def mfcc_user_presets(*, root: Optional[str] = None) -> "list[MfccPreset]":
    """The user's on-disk MFCC presets only (no built-ins)."""
    return _native.mfcc_preset.list_user(root=root)


@provisional
def builtin_mfcc_presets() -> "list[MfccPreset]":
    """The built-in authoritative MFCC presets (librosa / kaldi / praat)."""
    return _native.mfcc_preset.builtin()


@provisional
def mfcc_preset(id: str, *, root: Optional[str] = None) -> "Optional[MfccPreset]":  # noqa: A002
    """The MFCC preset with this ``id`` (built-in or on-disk), or ``None``."""
    return _native.mfcc_preset.get(id, root=root)


@provisional
def save_mfcc_preset(preset: "MfccPreset", *, root: Optional[str] = None) -> str:
    """Save a user MFCC preset to the store; returns its file path. Errors if
    the id is invalid or collides with a built-in (which are immutable)."""
    return _native.mfcc_preset.save(preset, root=root)


@provisional
def delete_mfcc_preset(id: str, *, root: Optional[str] = None) -> bool:  # noqa: A002
    """Delete the user MFCC preset with this ``id``. Returns ``True`` if a file
    was removed."""
    return _native.mfcc_preset.delete(id, root=root)


@provisional
def mfcc_preset_store(*, root: Optional[str] = None) -> str:
    """Filesystem path of the active MFCC preset store (default:
    ``~/.local/share/sadda/presets/mfcc/`` or the platform equivalent)."""
    return _native.mfcc_preset.store_root(root=root)
# Whisper-exact log-mel front end (the byte-faithful Whisper-encoder input).
log_mel_whisper = stable(_native.log_mel_whisper)
FormantFrame = stable(_native.FormantFrame)

# LTAS surface — long-term average spectrum + slope/tilt/alpha ratio.
ltas = stable(_native.ltas)
Ltas = stable(_native.Ltas)
