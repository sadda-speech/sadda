"""sadda.clinical — clinical perturbation measures (Phase 3 B4).

Jitter and shimmer over a sustained phonation, in the standard family
Praat reports. Validated against Praat (see the engine's
``tests/clinical`` fixtures and the 2026-05-25 clinical-validation-
references DEVLOG entry).

Stability tier: **STABLE-CLINICAL** — the stronger change-control the
clinical surface carries (per the 2026-05-18 clinical-regulatory entry).
For research, education, and non-diagnostic use only.
"""

from __future__ import annotations

from sadda import _native
from sadda._stability import provisional, stable_clinical

__all__ = [
    "PerturbationReport",
    "avqi",
    "cpps",
    "h1_h2",
    "hnr",
    "perturbation",
]

perturbation = stable_clinical(_native.perturbation)
PerturbationReport = stable_clinical(_native.PerturbationReport)
hnr = stable_clinical(_native.hnr)
cpps = stable_clinical(_native.cpps)
h1_h2 = stable_clinical(_native.h1_h2)

# AVQI is PROVISIONAL, not stable_clinical: the v03.01 formula is
# clean-room from the publications but not yet confirmed against the
# reference Praat script / authors (version + slope/tilt-definition
# questions are open), so its absolute values may change.
avqi = provisional(_native.avqi)
