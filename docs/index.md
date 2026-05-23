# sadda

*Sadda* (Pali सद्द — *"sound, voice"*) is an open-source toolkit for
phonetics and speech-science research. It targets the same audience as
[Praat](https://www.fon.hum.uva.nl/praat/) but with a modern Python API,
NumPy- and Polars-native data types, and a corpus-first project model.

This is the documentation for **sadda 0.1.0**, the first PyPI release.

## Install

```bash
pip install sadda
```

Pre-built wheels are available for Linux x86_64, macOS arm64, and
Windows x86_64 on Python 3.10–3.13. Other platforms install from sdist
and need a Rust toolchain locally.

## What's here

- **[Quickstart](quickstart.md)** — open a project, register a bundle,
  run pitch and formants, query the results as a Polars DataFrame.
- **API reference** — auto-generated from the Python source for
  [`sadda.corpus`](api/corpus.md), [`sadda.dsp`](api/dsp.md),
  [`sadda.live`](api/live.md), and [`sadda.recipe`](api/recipe.md).
- **Round-trip lossiness** — what's preserved (and what isn't) on
  [TextGrid](lossiness/textgrid.md) and [EAF](lossiness/eaf.md)
  import/export.

## Stability tiers

Per the
[2026-05-18 Python API surface entry](https://github.com/sadda-speech/sadda/blob/main/DEVLOG.md)
(in `DEVLOG.md`):

| Tier | Modules at 0.1.0 | Commitment |
|---|---|---|
| **Stable** | `sadda.corpus`, `sadda.dsp`, top-level project loaders | Won't break across minor versions |
| **Provisional** | `sadda.live`, `sadda.recipe` | May break in minor versions after a deprecation cycle |
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
