# sadda

*Sadda* (Pali: सद्द) — *sound, voice*.

An open-source toolkit for phonetics and speech-science research.

## Status

**Pre-alpha — scaffolding only.** The project is in active design and initial bootstrap. Nothing is installable from PyPI or runnable as a usable application yet. See [`DEVLOG.md`](DEVLOG.md) for the design decision log and the v1 milestone plan.

## Repository structure

```
sadda/
├── crates/
│   ├── engine/        Core Rust engine — DSP, corpus, I/O
│   ├── python/        PyO3 bindings, built into the `sadda` Python module via maturin
│   ├── app/           Desktop GUI (egui + wgpu) — planned
│   └── uniffi/        UniFFI bindings for mobile (iOS / Android) — planned
├── DEVLOG.md          Design decision log
├── pyproject.toml     Python project metadata + maturin config
├── Cargo.toml         Rust workspace root
└── rust-toolchain.toml
```

## Development

Requirements:

- Rust stable (managed via `rust-toolchain.toml`)
- Python 3.10+ (for the Python build path; not required for the Rust-only build)
- [uv](https://docs.astral.sh/uv/) for Python environment management

Build and test the Rust workspace:

```bash
cargo build
cargo test
```

Python wheel build instructions will land alongside the first real PyO3 surface.

## License

Dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual-licensed as above, without any additional terms or conditions.
