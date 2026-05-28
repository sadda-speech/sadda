# sadda.clinical

Voice-quality measures over an `Audio` — perturbation (jitter / shimmer),
harmonics-to-noise ratios, cepstral peak prominence, the AVQI / ABI
composite dysphonia indices, and the auxiliary component measures used
to compute them.

**Research use only.** These measures are implemented from peer-reviewed
publications with Praat-anchored validation where a Praat oracle exists,
and as clean-room reproductions from publications where one doesn't (see
the
[clinical-validation-references entry](https://github.com/sadda-speech/sadda/blob/main/DEVLOG.md)
in `DEVLOG.md`). The composite AVQI / ABI indices are PROVISIONAL pending
the authors' artifact for byte-level confirmation. **Not** a diagnostic
tool — see the project README's "Intended use" section.

Stability tier: **stable_clinical** (same API commitment as Stable, with
a distinct tier name to flag the research-use caveat).

## Perturbation

::: sadda.clinical.perturbation

::: sadda.clinical.PerturbationReport

## Harmonics-to-noise

::: sadda.clinical.hnr

::: sadda.clinical.hnr_d

::: sadda.clinical.gne

## Cepstral

::: sadda.clinical.cpps

::: sadda.clinical.h1_h2

## Spectral tilt / noise

::: sadda.clinical.hfno

## Composite indices (provisional)

::: sadda.clinical.avqi

::: sadda.clinical.abi
