"""sadda.dsp — foundational DSP toolkit.

Pure-function API over NumPy float32 arrays. Window functions, STFT,
spectrogram, intensity, and the relocated `f0` from Phase 0 all live here.
Stability tier: STABLE (per the 2026-05-18 Python API surface DEVLOG entry).

The top-level `sadda.f0` stays as a Phase-0 back-compat alias for the same
function.
"""

from __future__ import annotations

from sadda import _native
from sadda._stability import stable

__all__ = [
    "blackman",
    "f0",
    "gaussian",
    "hamming",
    "hann",
    "intensity",
    "kaiser",
    "spectrogram",
    "stft",
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
