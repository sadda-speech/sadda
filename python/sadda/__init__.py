"""sadda — open-source toolkit for phonetics and speech science research.

This package wraps the ``sadda._native`` Rust extension built by maturin,
re-exporting each public symbol with a stability decorator from
:mod:`sadda._stability` applied. End users should import from ``sadda``
directly rather than reaching into ``sadda._native``.

The Phase-0 surface (``version``, ``load_wav``, ``f0``, ``Audio``) is all
STABLE. Subsequent slices add new tiers (provisional/experimental) per the
2026-05-18 Python API surface DEVLOG entry and the 2026-05-21 A2 entry.
"""

from __future__ import annotations

from sadda import _native
from sadda._stability import (
    ExperimentalAPIWarning,
    ProvisionalAPIWarning,
    SaddaWarning,
    experimental,
    provisional,
    stable,
)

__all__ = [
    "Audio",
    "ExperimentalAPIWarning",
    "ProvisionalAPIWarning",
    "SaddaWarning",
    "experimental",
    "f0",
    "load_wav",
    "provisional",
    "stable",
    "version",
]

Audio = stable(_native.Audio)
version = stable(_native.version)
load_wav = stable(_native.load_wav)
f0 = stable(_native.f0)
