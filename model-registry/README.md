# sadda model registry

ONNX (and, by exception, other-format) models that sadda can resolve by id
and run — VAD, embeddings, transcription, segmentation, alignment, feature
extractors. The model analogue of the reference-distribution registry; it
reuses the same protocol (TOML manifest, tiers, Pages index, CI gate,
project pinning, in-app publishing) per the 2026-05-20 "ML model registry"
DEVLOG entry. The format is documented in [SCHEMA.md](SCHEMA.md).

This directory is scaffolded inside the main repo for now; it is designed
to split out into its own public repo (`sadda-speech/model-registry`) so
submissions are ordinary PRs.

> **Status: placeholder.** The entries here are **synthetic and
> manifest-only** (titles say PLACEHOLDER; the `url` points at
> `example.invalid`). They exercise the format, validator, and index
> builder until real curated models are listed (wav2vec2-base,
> whisper-tiny/base — landing with the E12 on-demand download path).
> Regenerate them with `python make_placeholders.py`.

## Weights live elsewhere

Unlike the refdist registry (which holds its data inline), model entries
are **manifest-only**: a `model.toml` + `LICENSE`, no weights. Weights are
typically 10 MB–5 GB+, so the manifest declares a `url` (HuggingFace,
Zenodo, a release asset, …) plus a `sha256` checksum, and the engine
fetches + verifies on demand (E12). The one exception is the **tier-1
bundled** set (`../models-bundled/`), whose small, vetted weights ship with
the app — there the manifest names a local `file`.

## Tiers

| Tier | Where | Bar |
|---|---|---|
| **1 — bundled** | `../models-bundled/` (ships with the app) | Small, vetted, foundational; redistributable license; weights inline |
| **2 — curated** | `tier2/` | Editorial review: provenance, license, ONNX canonical format, declared compute + output tier |
| **3 — community** | `tier3/` | Anyone may publish; trust signals over gatekeeping |

All tiers share one manifest format, one `load_model` resolver, and one
discovery surface; only the trust signal differs.

## Layout

```
model-registry/
  tier2/<id>/          model.toml + LICENSE   (weights via url + checksum)
  tier3/<id>/          (same)
  validate.py          CI gate: schema / kind / format / license / weights-resolvable
  build_index.py       emits index.json (the GitHub-Pages artifact)
  make_placeholders.py regenerates the synthetic placeholder entries
```

## Submitting (when this is its own repo)

1. Add a directory under `tier2/` or `tier3/` with `model.toml` + `LICENSE`.
2. Declare the weights via `model.url` + `model.file_checksum` (sha256).
3. `python validate.py .` must pass.
4. Open a PR. CI runs `validate.py` and rebuilds `index.json`.
5. Merge publishes; GitHub Pages serves `index.json`. The sadda engine
   reads it to discover models; `load_model("sadda/<id>")` fetches +
   verifies the weights on demand (E12).

## Format policy

**ONNX is canonical** for the curated tier (sadda runs models via ONNX
Runtime). `gguf` / `safetensors` / `savedmodel` are accepted as documented
exceptions; format conversion belongs to the publishing workflow, not the
runtime. Each entry ships a `LICENSE` (the weights' license verbatim for
real models).
