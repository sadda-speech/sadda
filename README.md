# sadda

*Sadda* (Pali: सद्द) — *sound, voice*.

An open-source toolkit for phonetics and speech-science research.

## Install

```bash
pip install sadda
```

Pre-built wheels are available for Linux x86_64, macOS arm64, and
Windows x86_64 on Python 3.10–3.13. Other platforms install from
sdist; you'll need a Rust toolchain locally.

## Desktop app

Unsigned binaries for the egui-based desktop app are attached to
each `v*.*.*-app` release on the
[GitHub Releases page](https://github.com/sadda-speech/sadda/releases).
Download the one for your OS and run it directly:

- `sadda-app-linux-x86_64`
- `sadda-app-macos-arm64` (Apple Silicon)
- `sadda-app-windows-x86_64.exe`

macOS users will see an "unidentified developer" warning on first
launch — right-click → Open to bypass. Proper notarisation lands
in 1.0.

The embedded script panel needs Python 3.11 or 3.12 installed on
the system. Everything else (waveform, spectrogram, tier editing,
playback) works without Python.

## Quickstart

```python
import sadda
from pathlib import Path

proj = sadda.new_project(Path("vowels"), name="vowel-study")
bundle_id = proj.add_bundle("speaker_01", Path("rec01.wav"))

audio = proj.load_audio(bundle_id)
pitch = sadda.dsp.voiced_pitch(
    audio.samples.astype("float32"),
    audio.sample_rate,
)

proj.import_textgrid(Path("phones.TextGrid"), bundle_id)
df = proj.query(tier_id="phones")
```

Full walk-through at the [quickstart](https://sadda-speech.github.io/sadda/quickstart/).

## What's in the box

- **Corpus model** — projects, bundles, six tier types (interval,
  point, reference, dense numeric / vector / categorical), parent-
  child cardinality, append-only audit log.
- **DSP toolkit** — windowing, STFT, spectrogram, intensity, pitch
  (autocorrelation + voicing), LPC formants, MFCC.
- **Interop** — Praat TextGrid and ELAN `.eaf` import/export with
  documented lossiness and a JSON-sentinel for `extra` payloads.
- **Live recording** — cpal-driven capture with streaming meter /
  pitch / intensity / formants subscribers; atomic commit into the
  project.
- **Recipes** — `with sadda.recipe.record(proj, name="..."):` links
  operations to a named record and emits a runnable `.py` script.

Status by module:

| Tier | Modules |
|---|---|
| **Stable** | `sadda.corpus`, `sadda.dsp`, top-level project loaders |
| **Provisional** | `sadda.live`, `sadda.recipe` |
| **Experimental** | none yet |

## Documentation

- **[sadda-speech.github.io/sadda](https://sadda-speech.github.io/sadda/)** — user docs (quickstart, API ref, round-trip lossiness)
- **[`DEVLOG.md`](DEVLOG.md)** — design decision log; the running record of why things are the way they are
- **[`CHANGELOG.md`](CHANGELOG.md)** — versioned release history

## Repository structure

```
sadda/
├── crates/
│   ├── engine/        Core Rust engine — DSP, corpus, I/O
│   ├── python/        PyO3 bindings; built into the `sadda` Python module via maturin
│   ├── app/           Desktop GUI (egui + wgpu) — planned for Phase 2
│   └── uniffi/        UniFFI bindings for mobile (iOS / Android) — planned for Phase 8
├── python/sadda/      Python wrapper around the Rust extension
├── docs/              mkdocs-material site source
├── DEVLOG.md          Design-decision log
├── CHANGELOG.md       Versioned release history
├── Cargo.toml         Rust workspace root
├── pyproject.toml     Python project metadata + maturin config
└── mkdocs.yml         Docs site config
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

# Python extension + tests
uv sync
uv run pytest python/tests/

# Docs site (auto-rebuilds on save)
uv pip install mkdocs-material "mkdocstrings[python]"
mkdocs serve
```

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual-licensed as above, without any additional terms or conditions.
