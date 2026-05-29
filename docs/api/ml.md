# sadda.ml

ONNX-model inference over audio — bundled Silero VAD plus a generic
embedding-extraction harness for wav2vec2-style (waveform) and
Whisper-style (log-mel) encoders. PROVISIONAL tier.

ONNX Runtime is loaded at runtime, not linked into the wheel. With
`pip install "sadda[ml]"` the wheel auto-discovers the installed
`onnxruntime` package at import time and sets `ORT_DYLIB_PATH`; the
desktop-app bundles ship the runtime as a sidecar so it just works.
Without ORT available, these calls raise a clear "ONNX Runtime not
available" error rather than crashing — see the
[2026-05-28 ORT-sidecar packaging DEVLOG entry](https://github.com/sadda-speech/sadda/blob/main/DEVLOG.md).

## Downloading models (`hf://`)

`load_model("hf://<org>/<name>/<file>[@<rev>]")` fetches a model from
HuggingFace into the local cache and runs it (unverified passthrough —
prefer a curated `sadda/…` id when one exists). `pip install
"sadda[download]"` is the convenient install (it pulls ONNX Runtime so a
downloaded model is immediately runnable).

**sadda never touches the network unless you opt in.** The fetch is
compiled into the wheel but stays dormant until you set the environment
variable `SADDA_ALLOW_NETWORK=1`; without it, an `hf://` cache miss
raises a clear *"network access is disabled"* error. Cached models and
`local://` / `sadda/…` ids always work offline. Authenticate to private
or gated repos with `HF_TOKEN`. The desktop app does **not** compile this
in — the GUI is network-free by construction.

```python
import os
os.environ["SADDA_ALLOW_NETWORK"] = "1"      # explicit opt-in
m = sadda.ml.load_model("hf://onnx-community/silero-vad/onnx/model.onnx")
```

## Voice activity detection (bundled)

::: sadda.ml.vad

::: sadda.ml.speech_segments

## Model resolution + embeddings

::: sadda.ml.load_model

::: sadda.ml.install_model

::: sadda.ml.get_model

::: sadda.ml.Model
