# B4 jitter/shimmer validation fixtures

Reference data for the engine's jitter/shimmer tests, per the
2026-05-25 *clinical validation-references* DEVLOG entry: **Praat is the
primary reference**, and its values are committed as golden fixtures so
CI never needs Praat installed.

## Files

| File | Produced by | Role |
|---|---|---|
| `synth_fixtures.py` | — | Synthesizes the test signals (controlled jitter/shimmer, **analytic** ground truth) |
| `jitter_shimmer.praat` | — | Measures the signals with Praat → the **reference** values |
| `../fixtures/*.wav` | `synth_fixtures.py` | The committed test signals |
| `../fixtures/injected.json` | `synth_fixtures.py` | Analytic ground truth (synthesis params + realized local measures) |
| `../fixtures/praat_golden.tsv` | `jitter_shimmer.praat` | Praat's jitter/shimmer per signal |

The engine tests assert their output against **`praat_golden.tsv` within
a per-measure tolerance** (computational reproducibility vs the
reference) and sanity-check it against **`injected.json`** (the analytic
truth).

## Regenerating

From the repo root (Praat **6.2.09**):

```bash
# 1. (Re)synthesize the WAVs + injected.json — deterministic.
python crates/engine/tests/clinical/praat/synth_fixtures.py

# 2. Measure them with Praat → praat_golden.tsv.
#    NOTE: Praat resolves relative paths against the *script* dir, so the
#    fixtures dir must be passed as an ABSOLUTE path.
praat --run crates/engine/tests/clinical/praat/jitter_shimmer.praat \
      "$(pwd)/crates/engine/tests/clinical/fixtures"
```

If the case list in `synth_fixtures.py` changes, delete stale
`fixtures/*.wav` first so the Praat glob doesn't pick them up.

## Measurement parameters (Praat)

Standard Voice-report defaults: period floor 0.1 ms, ceiling 20 ms, max
period factor 1.3, max amplitude factor 1.6; pitch range 75–600 Hz;
PointProcess via `To Pitch` → `To PointProcess (cc)`. Pin the Praat
version with any fixture regeneration — Praat's CPPS/AVQI internals have
shifted across releases (jitter/shimmer are stable, but record it anyway).

## Notes

- Perturbation is deterministic pseudo-random (seeded per signal), not
  alternating — alternating jitter/shimmer drives the pitch tracker into
  period-doubling. See the `synth_fixtures.py` docstring.
- Praat and the analytic truth agree closely (e.g. shimmer_150hz: Praat
  6.29% vs analytic 6.22%); small differences are the detector's, which
  is exactly why the engine is validated against Praat, not the injected
  value alone.
