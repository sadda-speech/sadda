# Changelog

All notable changes to sadda are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-05-25

First desktop-GUI release. Closes Phase 2 of the project plan: an
egui + wgpu application that opens projects, shows synced waveform /
spectrogram / tier views, edits annotations, runs embedded-CPython
scripts, records live audio, and imports/exports TextGrid + EAF —
all on top of the 0.1 engine. Distributed as unsigned prebuilt
binaries for macOS (arm64), Linux (x86_64), and Windows (x86_64).
The Python library surface is unchanged from 0.1.1 apart from one
new corpus method (`rename_bundle`). Tagged `v0.2.0-app` (the app
release track; Python wheels keep the `v*.*.*` tags).

### Added

#### Application shell + navigation

- **Project-aware shell** (`A1`): welcome screen with New / Open /
  Recent, persistent window + recent-projects state.
- **Bundle sidebar**: selectable per-bundle rows with a context menu
  (rename, reveal in file manager, delete) and a live bundle count.

#### Signal views (Cluster B + `C5`)

- **Waveform, spectrogram, and tier-strip panes** sharing one
  pixel-aligned time axis with a synced playback cursor (`C5`).
- **Spectrogram** with matplotlib-faithful viridis / magma /
  greyscale colormaps and configurable window / hop / dynamic range.
- **Audio playback** (cpal output) via spacebar toggle, with
  cursor-follow scrolling.

#### Annotation editing (`D6` / `D7`)

- **Interval + point editing**: drag-to-create, boundary resize,
  point move, inline label edit, and delete — persisted through the
  engine.

#### Scripting (`E8` / `E9`)

- **Embedded CPython script panel** (`E8`): run Python against the
  open project, reusing the Phase-0 script-engine.
- **In-app `sadda.app` namespace + command palette** (`E9`):
  `import sadda.app` from embedded scripts, register commands, and
  invoke them via Ctrl/Cmd+P.

#### I/O, recording, and bundle management (`H1`)

- **TextGrid + EAF import/export** wired into the File menu.
- **Live-recording modal**: device / sample-rate / channel choice,
  level meter, save-to-bundle.
- **Bundle delete** (with confirmation) and **bundle rename** (via
  the sidebar context menu), backed by new engine methods
  `Project::delete_bundle` and `Project::rename_bundle`, both with
  Python bindings.

#### Reliability + distribution (`F10` / `G11`)

- **Single-writer lock** (`F10`): a `.sadda-lock` advisory file
  (PID + hostname) stops two processes writing one corpus.
- **Release-binary CI** (`G11`): `app-release.yml` builds + uploads
  macOS / Linux / Windows binaries on `v*.*.*-app` tags.

### Fixed

- **WSL launch crash**: under WSLg, winit's Wayland backend
  broken-pipes on event-loop init; the app now drops
  `WAYLAND_DISPLAY` when running under WSL so it takes the XWayland
  path.
- **Lane alignment**: the waveform / spectrogram / tier-strip plot
  areas now share exact left and right boundaries (frameless panels
  plus a single shared left-gutter width), so the playback cursor
  draws as one straight line across all three.
- **Spectrogram bounds**: the plot no longer pads past the data into
  negative frequencies or beyond the recording, and now crops to the
  zoom window instead of always drawing the whole file.

[0.2.0]: https://github.com/sadda-speech/sadda/releases/tag/v0.2.0-app

## [0.1.1] — 2026-05-22

Packaging-only release. No code changes from 0.1.0; ships the sdist
that 0.1.0 was missing.

### Fixed

- **sdist now uploads to PyPI.** 0.1.0's sdist failed PyPI's PEP 639
  license-file validation because the deprecated `license = { text
  = ... }` form combined with two on-disk license files
  (`LICENSE-APACHE` + `LICENSE-MIT`) confused maturin's sdist
  generator — the `License-File: LICENSE-APACHE` metadata line had
  no matching file in the tarball. Switched to the PEP 639 SPDX
  expression form (`license = "Apache-2.0 OR MIT"`) with explicit
  `license-files = ["LICENSE-APACHE", "LICENSE-MIT"]`. Both files
  now ship in the sdist and are listed in METADATA consistently.
- **Dropped redundant `License ::` classifiers.** PEP 639 forbids
  using both the SPDX `license` field and the legacy
  `License :: OSI Approved :: ...` classifiers; PyPI rejects the
  combination.

### Notes

0.1.0 stays on PyPI as wheels-only — PyPI doesn't allow re-uploading
the same version. The 12 wheels from 0.1.0 install fine; 0.1.1 adds
the matching sdist plus an equivalent wheel set built from the
post-fix sources. Users on the supported wheel matrix can install
either; users compiling from source need 0.1.1.

[0.1.1]: https://github.com/sadda-speech/sadda/releases/tag/v0.1.1

## [0.1.0] — 2026-05-22

First PyPI release. Closes Phase 1 of the project plan; brings the
Python library, corpus model, DSP toolkit, two interchange formats,
live recording, and reproducibility recipes to a usable state.

### Added

#### Core (Cluster A — infrastructure)

- **Migration framework** (`A1`): forward-only SQLite migrations with
  per-step transactions, schema-version backup, and a forward-compat
  clamp that refuses to open a database newer than the engine knows.
- **Stability decorators + type stubs** (`A2`): `@stable`,
  `@provisional`, `@experimental` decorators emit one-time runtime
  warnings on first use of a non-stable API; `pyo3-stub-gen`
  generates `.pyi` stubs at build time.

#### Corpus (Cluster B)

- **Full entity schema + audit log** (`B1`): `speaker`, `session`,
  `bundle`, `tier`, `entity`, `entity_ref`, `instrument`, `protocol`,
  `processing_run`, and an append-only `audit_log` populated by
  SQLite triggers. User attribution via a singleton
  `_audit_context` row.
- **Sparse tier types** (`B2`): `interval`, `point`, `reference`
  tiers with full CRUD; parent-child cardinality enforced at insert
  time. First cut of `proj.query(tier_id) → polars.DataFrame`.
- **Dense tier types + Parquet sidecars** (`B3`):
  `continuous_numeric`, `continuous_vector`, `categorical_sampled`
  via `DerivedSignal` registration rows pointing at Parquet files
  under `signals/derived/`. mmap-friendly load path.

#### DSP (Cluster C)

- **Foundational DSP** (`C1`): windowing (Hann, Hamming, Blackman,
  Gaussian, Kaiser), STFT, spectrogram, intensity. Pure functions
  over `&[f32]`; no corpus dependency.
- **Advanced DSP** (`C2`): LPC + Aberth-root formants, mel→DCT MFCC,
  voicing decision on the autocorrelation pitch tracker. Method
  diversity principle: every measure cites its source and offers
  multiple non-equivalent implementations where they exist.

#### Interop (Cluster D)

- **TextGrid round-trip** (`D1`): import/export of Praat TextGrid
  (long + short text variants). JSON-sentinel suffix preserves
  per-annotation `extra` JSON across the round-trip.
- **EAF round-trip** (`D2`): import/export of ELAN `.eaf` files via
  `quick-xml`. Tier hierarchy survives via `PARENT_REF` ↔
  `tier.parent_id`; point tiers via the degenerate-alignable
  encoding with the `≤ 2ms` recovery heuristic.

#### Recording (Cluster E)

- **Live recording** (`E1`): cpal-driven capture with rtrb SPSC
  ringbuffer plumbing, `.in_progress/<uuid>/` staging, atomic
  rename on commit. Streaming `on_meter` / `on_pitch` /
  `on_intensity` / `on_formants` subscribers fire from a dispatch
  thread that pops from result rtrbs with the GIL held.

#### Reproducibility (Cluster F)

- **Recipes** (`F1`): `with sadda.recipe.record(proj, name):`
  context manager links every processing-run write inside the
  block to a named `recipe_run` row and emits
  `<project>/recipes/<name>.py` at clean exit.

#### Documentation (Cluster G)

- **mkdocs-material site** (`G12`) with auto-generated Python API
  ref via mkdocstrings, a quickstart tutorial, and lossiness pages
  for TextGrid and EAF.
- **PyPI release** (`G12`): published as `sadda` for Python 3.10–3.13
  on Linux x86_64 / macOS arm64 / Windows x86_64 plus an sdist
  fallback.

### Notes

The Rust crates (`sadda-engine`, `sadda-python`) are **not** published
to crates.io at 0.1.0 — the Rust API is still being shaken out and we
want to iterate without SemVer pressure. Republish under a stable
crate name is a 0.2 candidate.

[0.1.0]: https://github.com/sadda-speech/sadda/releases/tag/v0.1.0
