# sadda

*Sadda* (Pali सद्द — *"sound, voice"*) is an open-source toolkit for
phonetics and speech-science research, with a modern Python API,
NumPy- and Polars-native data types, and a corpus-first project model.

![The sadda desktop app showing a short speech recording — a waveform, a spectrogram, an MFCC heatmap, and stacked f0, formant, and intensity tracks — with word- and utterance-level annotation tiers below and a bundle-list sidebar.](assets/generated/overview.png#only-light){ width="900" }
![The sadda desktop app (dark theme) showing a short speech recording — a waveform, a spectrogram, an MFCC heatmap, and stacked f0, formant, and intensity tracks — with word- and utterance-level annotation tiers below and a bundle-list sidebar.](assets/generated/overview-dark.png#only-dark){ width="900" }

These docs always reflect the latest commit on `main`. For per-release
notes, see the
[CHANGELOG](https://github.com/sadda-speech/sadda/blob/main/CHANGELOG.md)
and [GitHub releases](https://github.com/sadda-speech/sadda/releases).

## Install

```bash
pip install sadda           # core: corpus, DSP, clinical, refdist, recipes
pip install "sadda[ml]"     # also installs onnxruntime — VAD + embeddings
```

Pre-built wheels are available for Linux x86_64, macOS arm64, and
Windows x86_64 on Python 3.10–3.13. Other platforms install from sdist
and need a Rust toolchain locally.

The `[ml]` extra adds ONNX Runtime (~25 MB). The base install is lean
for users who don't need VAD / embeddings; ML calls raise a clean
"ONNX Runtime not available" error if the extra isn't installed.

## What's here

- **[Quickstart](quickstart.md)** — open a project, register a bundle,
  run pitch / formants / a clinical measure, query the results as a
  Polars DataFrame.
- **[Text-to-speech](tts.md)** — synthesize narration for generated docs
  (or any TTS task) through a pluggable backend, with content-hash caching
  and 100+ languages via espeak-ng.
- **API reference** — auto-generated from the Python source for
  [`sadda.corpus`](api/corpus.md), [`sadda.dsp`](api/dsp.md),
  [`sadda.clinical`](api/clinical.md), [`sadda.refdist`](api/refdist.md),
  [`sadda.ml`](api/ml.md), [`sadda.live`](api/live.md),
  [`sadda.recipe`](api/recipe.md), and [`sadda.tts`](api/tts.md).
- **Round-trip lossiness** — what's preserved (and what isn't) on
  [TextGrid](lossiness/textgrid.md) and [EAF](lossiness/eaf.md)
  import/export.

## Stability tiers

Per the
[2026-05-18 Python API surface entry](https://github.com/sadda-speech/sadda/blob/main/DEVLOG.md)
(in `DEVLOG.md`):

| Tier | Modules | Commitment |
|---|---|---|
| **Stable** | `sadda.corpus`, `sadda.dsp`, top-level project loaders | Won't break across minor versions |
| **Stable (clinical)** | `sadda.clinical` | Same commitment as Stable; the separate tier flags that these measures are **research-use only**, not for clinical diagnosis |
| **Provisional** | `sadda.live`, `sadda.recipe`, `sadda.refdist`, `sadda.ml`, `sadda.tts` | May break in minor versions after a deprecation cycle |
| **Experimental** | `sadda.experimental.*` (none yet) | May break any release |

Non-stable APIs emit a one-time `ProvisionalAPIWarning` /
`ExperimentalAPIWarning` on first use (suppressible via the standard
`warnings` machinery).

## Source

Everything lives at [github.com/sadda-speech/sadda](https://github.com/sadda-speech/sadda).
The `DEVLOG.md` in the repo is the running design-decision log; the
docs here are a curated subset of what's most useful when you're
actually using the library.

## License

Dual-licensed under Apache 2.0 OR MIT.
