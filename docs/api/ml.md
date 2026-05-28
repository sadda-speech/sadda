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

## Voice activity detection (bundled)

::: sadda.ml.vad

::: sadda.ml.speech_segments

## Model resolution + embeddings

::: sadda.ml.load_model

::: sadda.ml.install_model

::: sadda.ml.get_model

::: sadda.ml.Model
