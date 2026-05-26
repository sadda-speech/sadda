# sadda reference-distribution registry

Reference distributions — tagged statistical summaries (or samples) of
acoustic/articulatory measures over a population, or prescriptive targets —
that sadda can render a measurement against: vowel-formant clouds,
age/sex-normed clinical ranges, f0 statistics, voice-coach target zones, and
so on.

This directory is the **registry** (tiers 2 and 3). It is scaffolded inside
the main repo for now; it is designed to be split out into its own public
repo (`sadda-speech/refdist-registry`) so submissions are ordinary PRs. The
governance model is the 2026-05-18 "Reference distribution governance" DEVLOG
entry; the format is documented in [SCHEMA.md](SCHEMA.md).

> **Status: placeholder.** Every distribution here right now is **synthetic
> and non-authoritative** (titles and `provenance.md` say PLACEHOLDER). They
> exist to exercise the format, validator, index builder, and the engine
> resolver end to end until license-cleared real data is sourced
> (Hillenbrand 1995, Peterson-Barney 1952, clinical normative ranges).
> Regenerate them with `python make_placeholders.py`.

## Tiers

| Tier | Where | Bar |
|---|---|---|
| **1 — bundled** | `../refdist-bundled/` (ships with the app) | Small, vetted, foundational; only redistributable-licensed data |
| **2 — curated** | `tier2/` | Editorial review: provenance, license, min-n, reproducible method. At most one per (measure × population × method) |
| **3 — community** | `tier3/` | Anyone may publish; trust signals (downloads, promotion, flags) over gatekeeping |

All tiers share one format, one query API, and one discovery surface; only
the trust signal differs.

## Layout

```
refdist-registry/
  tier2/<id>/          refdist.toml + data.parquet + provenance.md + LICENSE
  tier3/<id>/          (same)
  validate.py          CI gate: schema / license / min-n / data conformance
  build_index.py       emits index.json (the GitHub-Pages artifact)
  make_placeholders.py regenerates the synthetic placeholder set
```

## Submitting (when this is its own repo)

1. Add a directory under `tier2/` or `tier3/` with the four files above.
2. `python validate.py .` must pass.
3. Open a PR. CI runs `validate.py` and rebuilds `index.json`.
4. Merge publishes; GitHub Pages serves `index.json` + the data files. The
   sadda engine reads `index.json` to discover and resolve distributions.

Tier-2 promotion of a tier-3 entry is a second PR moving the directory.
Yanking sets `yanked = true` in the entry; pinned versions keep resolving
with a warning.

## License policy

Prefer **CC0-1.0 / CC-BY-4.0 / ODC-BY-1.0** for data. Tier 2 **disallows**
NonCommercial / NoDerivatives; tier 3 allows them with a prominent flag. Each
distribution ships a `LICENSE` file (the upstream license verbatim for real
data).
