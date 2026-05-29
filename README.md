# sadda

_Sadda_ ([Pali](https://en.wikipedia.org/wiki/Pali): सद्द) — _sound, voice_.

An open-source toolkit for phonetics and speech-science research.

## Intended use

sadda is **for research, education, and non-diagnostic use only.** It is
not a medical device and makes no diagnostic, therapeutic, or treatment
claims. Its clinical-style measures (jitter, shimmer, HNR, CPP, AVQI,
ABI, …) are provided for research and education; they are implemented
with validation suites and provenance so a downstream entity _could_
pursue regulatory clearance, but sadda itself is not cleared and takes
no liability for clinical decisions made with it. "Research use" by
long convention includes clinical-_research_ contexts.

## Install

```bash
pip install sadda           # core: corpus, DSP, clinical, refdist, recipes
pip install "sadda[ml]"     # also pulls ONNX Runtime — enables VAD + embeddings
```

Pre-built wheels are available for Linux x86_64, macOS arm64, and
Windows x86_64 on Python 3.10–3.13. Other platforms install from
sdist; you'll need a Rust toolchain locally.

The `[ml]` extra adds `onnxruntime` (~25 MB) so the optional ML
features work out-of-the-box; without it those calls raise a clear
"ONNX Runtime not available" error instead of crashing. The base
install is lean for users who only need acoustics + annotation.

## Desktop app

Unsigned bundles for the egui-based desktop app are attached to each
`v*.*.*-app` release on the
[GitHub Releases page](https://github.com/sadda-speech/sadda/releases).
Each bundle is a single archive containing the app binary plus a
sidecar ONNX Runtime — ML features (VAD, embeddings) work without
any extra install:

- `sadda-app-linux-x86_64.tar.gz`
- `sadda-app-macos-arm64.tar.gz` (Apple Silicon)
- `sadda-app-windows-x86_64.zip`

Extract the archive and run the binary from the resulting
`sadda-app-<platform>/` directory — the app picks up the bundled
`onnxruntime/` automatically.

macOS users will see an "unidentified developer" warning on first
launch — right-click → Open to bypass. Proper notarisation lands
in 1.0.

The embedded script panel needs Python 3.11 or 3.12 installed on
the system. Everything else (waveform, spectrogram, measure-track
lanes, tier editing, playback, VAD, refdist overlays) works without
Python.

## Quickstart

```python
import sadda
from pathlib import Path

proj = sadda.new_project(Path("vowels"), name="vowel-study")
bundle_id = proj.add_bundle("speaker_01", Path("rec01.wav"))

audio = proj.load_audio(bundle_id)
times, freqs, voicing = sadda.dsp.voiced_pitch(audio)

# `import_textgrid` returns the integer tier id(s) it created.
[phones_tier] = proj.import_textgrid(Path("phones.TextGrid"), bundle_id)
df = proj.query(phones_tier)
```

Full walk-through at the [quickstart](https://sadda-speech.github.io/sadda/quickstart/).

## What's in the box

- **Corpus model** — projects, bundles, six tier types (interval,
  point, reference, dense numeric / vector / categorical), parent-
  child cardinality, append-only audit log.
- **DSP toolkit** — windowing, STFT, spectrogram, intensity, pitch
  (autocorrelation + voicing), LPC formants, MFCC, LTAS.
- **Clinical measures** — jitter (local / RAP / PPQ5), shimmer
  (local / dB / APQ3 / APQ5), HNR, CPP / CPPS, H1–H2, GNE, and the
  composite AVQI / ABI dysphonia indices (provisional). Praat-
  validated where a Praat oracle exists; clean-room from the source
  publications where one doesn't.
- **Reference distributions** — `sadda.refdist`: install + query
  curated distributions (normative ranges, target zones, observed
  corpora); pin per-project; publish your own back to the registry.
- **ML inference** — bundled Silero VAD plus an embedding-extraction
  harness for wav2vec2 / Whisper-style ONNX models, via load-dynamic
  ONNX Runtime. Inference results land as B3-shaped continuous-vector
  tiers ready for downstream analysis.
- **Interop** — Praat TextGrid and ELAN `.eaf` import/export with
  documented lossiness and a JSON-sentinel for `extra` payloads.
- **Live recording** — cpal-driven capture with streaming meter /
  pitch / intensity / formants subscribers; atomic commit into the
  project.
- **Calibration + provenance** — instrument calibration for absolute
  dB-SPL; every analysis records a `processing_run` row; bundled
  citation export for the methods you used.
- **Desktop GUI** — egui app with waveform / spectrogram / measure
  tracks (f0, formants, intensity, VAD), reference-distribution
  overlays, vowel-space scatter, TextGrid / EAF I/O, live recording,
  and an embedded Python script panel.
- **Recipes** — `with sadda.recipe.record(proj, name="..."):` links
  operations to a named record and emits a runnable `.py` script.

Status by module:

| Tier                  | Modules                                                |
| --------------------- | ------------------------------------------------------ |
| **Stable**            | `sadda.corpus`, `sadda.dsp`, top-level project loaders |
| **Stable (clinical)** | `sadda.clinical` — clinical-research-use only          |
| **Provisional**       | `sadda.live`, `sadda.recipe`, `sadda.refdist`, `sadda.ml` |
| **Experimental**      | none yet                                               |

## Documentation

- **[sadda-speech.github.io/sadda](https://sadda-speech.github.io/sadda/)** — user docs (quickstart, API ref, round-trip lossiness)
- **[`DEVLOG.md`](DEVLOG.md)** — design decision log; the running record of why things are the way they are
- **[`CHANGELOG.md`](CHANGELOG.md)** — versioned release history

## Repository structure

```
sadda/
├── crates/
│   ├── engine/         Core Rust engine — DSP, corpus, clinical, refdist, ML, I/O
│   ├── python/         PyO3 bindings; built into the `sadda` Python module via maturin
│   ├── app/            Desktop GUI (egui + wgpu)
│   ├── script-engine/  Embedded CPython used by the GUI script panel
│   └── uniffi/         UniFFI bindings for mobile (iOS / Android) — planned for v1.x
├── python/sadda/       Python wrapper around the Rust extension
├── docs/               mkdocs-material site source
├── models-bundled/     First-run-seedable model weights (Silero VAD)
├── refdist-bundled/    First-run-seedable reference distributions
├── DEVLOG.md           Design-decision log
├── CHANGELOG.md        Versioned release history
├── THIRD_PARTY_NOTICES.md  Attributions for bundled third-party binaries
├── Cargo.toml          Rust workspace root
├── pyproject.toml      Python project metadata + maturin config
└── mkdocs.yml          Docs site config
```

## Development

Requirements:

- Rust stable (managed via `rust-toolchain.toml`)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for Python environment management

```bash
# Rust workspace
cargo build
cargo test

# Desktop app (egui + wgpu)
cargo run -p sadda-app

# Python extension + tests
uv sync
uv run pytest python/tests/

# Docs site (auto-rebuilds on save)
uv pip install mkdocs-material "mkdocstrings[python]"
mkdocs serve
```

The app embeds CPython for the script panel, so its binary links
against the `libpython` PyO3 picked at build time (via the
`PYO3_PYTHON` env var, defaulting to `python3` on `PATH`). If
`cargo run -p sadda-app` fails with `error while loading shared
libraries: libpython3.X.so.1.0: cannot open shared object file`,
that `libpython` isn't on the runtime loader path. Two fixes:

```bash
# 1. Put the build-time Python's lib dir on the loader path:
LD_LIBRARY_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')" \
    cargo run -p sadda-app

# 2. Or rebuild against a Python whose libpython is already on
#    the loader path (e.g. Ubuntu's /usr/bin/python3):
PYO3_PYTHON=/usr/bin/python3 cargo clean -p sadda-app
PYO3_PYTHON=/usr/bin/python3 cargo run -p sadda-app
```

Common trigger: conda / miniconda `python3` is first on `PATH`.
See `crates/script-engine/README.md` for the same gotcha at the
test layer.

### Validation

DSP and clinical measures are validated against authoritative external
references — the tool or reference implementation that *defines* each
method (Praat, librosa, OpenAI Whisper, and Camacho's own SWIPE' MATLAB
run under Octave). The reference values are committed as small golden
files, so the test suite runs fully offline; the external tools are only
needed to regenerate a golden. See
[`crates/engine/tests/README.md`](crates/engine/tests/README.md) for the
philosophy and which tool produces which golden.

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual-licensed as above, without any additional terms or conditions.

## AI and human acknowledgement

This project was developed with the assistance of an AI tool (Claude Code, made by Anthropic). Large language models like Claude are trained on large corpora of text and code, which include publicly available source code written by many human developers. While the model does not copy or retrieve specific files when generating suggestions, its capabilities are fundamentally built upon patterns learned from this collective body of human work. We acknowledge that the AI assistance used in this project rests on the contributions of countless developers whose code formed part of that training, even though they cannot be individually credited.
