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
    "abi",
    "avqi",
    "cpps",
    "gne",
    "h1_h2",
    "hfno",
    "hnr",
    "hnr_d",
    "perturbation",
]

perturbation = stable_clinical(_native.perturbation)
PerturbationReport = stable_clinical(_native.PerturbationReport)
hnr = stable_clinical(_native.hnr)
cpps = stable_clinical(_native.cpps)
h1_h2 = stable_clinical(_native.h1_h2)

# GNE is PROVISIONAL: the algorithm follows the canonical published
# parametrization (Michaelis et al. 1997; bw=1000, fshift=300 Hz), and
# its behaviour is validated qualitatively (discriminates pulsatile from
# turbulent excitation, orders clean>noisy), but its absolute values are
# not yet confirmed against a reference oracle — there is no Praat GNE.
gne = provisional(_native.gne)

# ABI component measures. Hfno-6000 and HNR-D are PROVISIONAL: Hfno's
# exact band-level convention and HNR-D's harmonic/noise separation are
# reconstructed from the ABI papers' prose, not confirmed against the
# authors' artifact.
hfno = provisional(_native.hfno)
hnr_d = provisional(_native.hnr_d)

# ABI is PROVISIONAL: the published v01 regression formula, but the
# component definitions (HNR-D/Hfno) and unit conventions aren't yet
# confirmed against the authors' artifact, so its absolute values are not
# to be trusted. Takes the nine components directly (no abi_from_audio).
abi = provisional(_native.abi)

# AVQI is PROVISIONAL, not stable_clinical: the v03.01 formula is
# clean-room from the publications but not yet confirmed against the
# reference Praat script / authors (version + slope/tilt-definition
# questions are open), so its absolute values may change.
avqi = provisional(_native.avqi)
