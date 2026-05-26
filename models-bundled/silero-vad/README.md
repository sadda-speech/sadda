# Silero VAD (bundled)

`silero_vad.onnx` is the [Silero VAD](https://github.com/snakers4/silero-vad)
voice-activity-detection model, vendored here as sadda's first bundled ML
model (Phase 3 cluster E / E11). It ships with the app so the
`engine::ml::vad` path works offline, with no model download.

- **Model**: Silero VAD, ONNX export (`silero_vad.onnx`).
- **Version**: silero-vad 6.2.1 (PyPI), 2026.
- **License**: MIT (see `LICENSE`) — redistributable, no attribution
  strings attached. © 2020–present Silero Team.
- **Source**: <https://github.com/snakers4/silero-vad>

## I/O contract (drives `engine::ml::vad`)

| Tensor | Direction | Type / shape | Notes |
|--------|-----------|--------------|-------|
| `input` | in | f32 `[batch, samples]` | 512 samples per window at 16 kHz |
| `state` | in | f32 `[2, batch, 128]` | recurrent state, threaded across windows |
| `sr` | in | i64 scalar | 16000 |
| `output` | out | f32 `[batch, 1]` | speech probability for the window |
| `stateN` | out | f32 `[2, batch, 128]` | next `state` |

The engine resamples to 16 kHz, runs the model window-by-window threading
`stateN -> state`, and reports a speech probability per window.
