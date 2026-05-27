# `model.toml` schema

A model entry is a directory containing:

```
model.toml        manifest (this schema)
LICENSE           the weights' license, verbatim
<weights file>    only for the tier-1 bundled set; registry entries use a url
```

The manifest is parsed by `sadda_engine::models` (Rust) and by this
registry's `validate.py` / `build_index.py` (stdlib `tomllib`). Keep the
two in sync.

## Fields

```toml
id = "sadda/wav2vec2-base-960h"      # required — resolvable id
version = "1.0.0"                    # required — semver, immutable once published
title = "wav2vec2-base self-supervised speech model"
upstream_source = "https://huggingface.co/facebook/wav2vec2-base-960h"
license = "Apache-2.0"               # SPDX id; LICENSE file required

[model]
kind = "embedding"        # embedding | transcription | vad | segmentation | alignment | feature
format = "onnx"           # onnx (canonical) | gguf | safetensors | savedmodel
# Weights: a local `file` (tier-1 bundled) OR a `url` + `file_checksum`
# (registry entries; fetched + verified on demand).
file = "model.onnx"                  # for bundled entries
# url = "https://…/model.onnx"       # for registry entries
file_checksum = "sha256:…"           # required when `url` is used

[input]
modality = "audio"                   # audio | video | both
sample_rate_hz = 16000
channels = 1

[output]
tier_kind = "continuous_vector"      # the sadda tier kind inference produces:
                                     # interval | point | reference |
                                     # continuous_numeric | continuous_vector |
                                     # categorical_sampled
channels = 768                       # embedding dim / output channels
sample_rate_hz = 50                  # output frame rate, if a dense signal

[compute]
cpu_min_ram_mb = 1024
gpu = "optional"                     # required | optional | unsupported

[citation]
authors = ["…"]
year = 2026
bibtex = "…"                         # optional, paste-ready
```

## `index.json` (built by `build_index.py`)

One entry per model, in the shape `sadda_engine::models::ModelRegistryIndex`
deserializes:

```json
{
  "schema_version": 1,
  "entries": [
    {
      "id": "sadda/wav2vec2-base-960h", "version": "1.0.0", "tier": 2,
      "title": "…", "kind": "embedding", "format": "onnx",
      "license": "Apache-2.0", "path": "tier2/wav2vec2-base-960h",
      "yanked": false
    }
  ]
}
```

`yanked = true` keeps a pinned version resolvable but surfaces a warning.
