# Changelog

All notable changes to sadda are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Import a whole folder of recordings.** The desktop app's **File ▸ Add
  Directory…** picks a folder and registers every `.wav` in it (case-insensitive,
  sorted alphabetically so bundle order is predictable) as a bundle, each going
  through the same large-file probe/split guard as a single Add Bundle…. Reports
  how many were added, or an error if the folder has no WAV files.

### Fixed

- **Fresh-start crash on a clean machine.** `PersistedState`'s derived `Default`
  left `ui_scale` at `0.0`, which made egui panic the first time the app ran with
  no saved state. Replaced with an explicit `Default` that seeds `ui_scale` to
  `1.0`.

## [0.4.0] — 2026-06-03

Adds the **annotation campaign suite** — a full labeling-campaign platform
(rubrics, computational criteria, targets, annotator assignment, inter-annotator
agreement, QA dashboard, lab-notebook) — plus **Praat-style keyboard scan /
annotate ergonomics** in the desktop app, a corpus-wide **concordance** view, a
performance pass, and a run of correctness + packaging fixes (notably VAD, which
had been returning zero for every input). Tagged both `v0.4.0` (Python wheel) and
`v0.4.0-app` (desktop binaries).

### Added

#### Annotation campaign suite (migrations V8–V14)

- **Rubric-as-data + controlled vocabulary**, with per-annotation status / note.
- **Computational criteria** — structured rules and a typed *signal-function
  expression* language over open signal + reducer registries — that propose
  annotations on a preview tier for review, with full criterion-run provenance.
- **Targets** (first-class work units), **annotator assignment** with seeded
  random distribution, and **per-annotator package** export / import + tier merge.
- **Agreement engine** — Cohen's κ, both unit- and frame-based — plus a work
  queue, a compile / QA dashboard, **rubric versioning + impact**, and a PI
  **lab-notebook**.

#### Desktop app

- **Scan & annotate ergonomics** — keyboard span playback with loop + pause;
  multi-active tiers via digit keys; click-to-place a point; **Enter** to commit
  an annotation to all active tiers with conflict resolution; **Ctrl-snap** of a
  selection edge to the nearest existing boundary.
- **Concordance view** — concatenate every token matching a tier + label across
  the corpus into one continuous timeline.
- **Help → Memory report**; selection start / end times in the waveform header; a
  spectrogram **Reset** button.

#### Other

- **Large-file ingest guard** — warn + streaming-split files that would exceed
  ~512 MiB decoded.
- The bundled **Silero VAD now ships inside the Python wheel**, so
  `sadda.ml.vad()` works out of the box after `pip install "sadda[ml]"`.

### Changed

- The default measure-track / `sadda.dsp.voiced_pitch` f0 tracker is now the
  octave-robust **Boersma** (was windowed-autocorrelation, which latched onto
  subharmonics of clean tones — e.g. 150 → 75 Hz).
- **Performance** — FFT-based pitch autocorrelation (~700×), a per-bundle signal
  cache, asynchronous DSP, and an adaptive cache budget make bundle switching
  noticeably snappier.

### Fixed

- **VAD returned ~0 for all audio.** The Silero 2024 model requires a 64-sample
  context window prepended to each 512-sample frame (model input `[1, 576]`);
  sadda fed bare frames, so the model never detected speech. Fixed and verified
  end-to-end on real speech.
- **f0 octave / subharmonic errors** on clean tones (resolved by the Boersma
  default above).
- The Python wheel now includes the bundled VAD model — previously
  `sadda.ml.vad()` raised "bundled Silero VAD not found" for pip installs.
- Refreshed the embedded script-panel placeholder text (dropped internal slice
  codes that meant nothing to users).

## [0.3.0] — 2026-05-28

Closes Phase 3 of the project plan: clinical voice-quality measures,
reference distributions, ML inference, calibration, and provenance.
The Python wheel jumps from `0.1.1` and adds four new submodules
(`sadda.clinical`, `sadda.refdist`, `sadda.ml`, plus expanded
`sadda.dsp.ltas`). The desktop app jumps from `0.2.0`-app and gains
measure-track lanes, reference-distribution overlays, a Reference
panel (vowel-space scatter + parameter histogram), a VAD lane, a
provenance / citations modal, and a bundled-ONNX-Runtime sidecar so
ML features work out-of-the-box. Tagged both `v0.3.0` (Python wheel)
and `v0.3.0-app` (desktop binaries) — the first release to ship both
tracks at the same version.

### Added

#### Clinical substrate (Cluster A)

- **Provenance + citations** (`A1`): every analysis records a
  `processing_run` row; `Project.processing_runs(bundle_id)` and
  `Project.citations(bundle_id)` return them. The GUI has a
  read-only "Provenance & citations…" modal with clipboard-copy.
- **Typed units + clinical discipline** (`A2`): engine-side
  `Hertz` / `Decibels` / `Ratio` / `Seconds` newtypes (frequency and
  level only; time stays bare `f64` at the NumPy boundary). New
  `stable_clinical` stability tier — same API commitment as Stable,
  with a distinct tier name flagging the research-use caveat. A
  status-bar "research use only" notice in the GUI.
- **Instrument calibration** (`A3`): `Instrument` + `Calibration`
  data types with reference-pair single-offset SPL model;
  `Project.add_instrument` / `instruments` / `get_instrument` /
  `bundle_calibration`. Calibration is stored as JSON in the
  existing `instrument.calibration` column — no migration needed.

#### Clinical measures (Cluster B)

- **Jitter + shimmer** (`B4`): `sadda.clinical.perturbation(audio)`
  computes pitch-synchronous jitter (local / RAP / PPQ5) and shimmer
  (local / dB / APQ3 / APQ5) and returns a `PerturbationReport`.
  Praat-validated within tolerance via a committed golden-fixture
  validation harness (so CI needs no Praat install).
- **Harmonics-to-noise + cepstral measures** (`B5`):
  `sadda.clinical.hnr` (Praat cross-correlation HNR; Praat-validated)
  and `sadda.clinical.cpps` (cepstral peak prominence, smoothed;
  offset-invariant peak − robust tilt line; Praat-validated).
- **LTAS as a first-class feature**: `sadda.dsp.ltas` returns long-term
  average spectrum levels with `.slope` (band-energy ratio) and
  `.tilt` (regression dB / kHz) and `.alpha_ratio` helpers.
- **Composite dysphonia indices + components** (`B6`):
  `sadda.clinical.avqi` (Acoustic Voice Quality Index, v03.01) and
  `sadda.clinical.abi` (Acoustic Breathiness Index, v01) — both
  **provisional** pending byte-level confirmation against the
  authors' artifact. Component measures `h1_h2`, `gne`
  (Glottal-to-Noise Excitation Ratio), `hfno`, and `hnr_d`
  (Dejonckere–Lebacq HNR) also exposed as standalone functions.
- **Clean-room provenance**: clinical/proprietary-origin measures
  (AVQI / ABI / MDVP) are clean-room reproductions from
  publications; proprietary scripts (e.g. Phonanium) are confirmation
  oracles only, never models.

#### Reference distributions (Cluster C)

- **Resolver + manifest + per-user store** (`C7`): `sadda.refdist`
  exposes `query`, `get`, `list_all`, `install`, `store_root`
  (default `~/.local/share/sadda/refdist/`). Distributions carry a
  `refdist.toml` manifest with `MeasureKind` (observed_distribution
  | summary_normative_range | target_zone) — encoded distinctly in
  the GUI so observed / normative / target never visually collide.
- **Project-level pinning**: `Project.pin_refdist` / `refdist_pins`
  / `remove_refdist_pin` persist a pin in `project.toml`.
- **Registry repo + CI** (`C8`): an in-repo
  [`refdist-registry/`](https://github.com/sadda-speech/sadda/tree/main/refdist-registry)
  with tier2/tier3 directories, a self-contained `validate.py` CI
  gate (license check, min-n, distinct-speaker k-anon proxy), and
  `build_index.py` emitting `index.json`. End-to-end-validated with
  a synthetic placeholder distribution set.
- **Bundled distribution + first-run seeding** (`C8`/`D10`):
  `refdist-bundled/placeholder-amE-vowels` ships with the wheel; the
  desktop app has a View → "Install bundled reference data" command
  that seeds the per-user cache.
- **In-app publishing scaffold** (`C9`): `sadda.refdist.scaffold`
  writes a registry-ready directory (refdist.toml + data.parquet +
  provenance + LICENSE stub) from a Polars DataFrame — round-trips
  through the C8 validator.

#### Desktop GUI (Cluster D / `D10`)

- **Measure-track lanes**: stacked f0 / formants / intensity / VAD
  lanes below the spectrogram, sharing the time-axis gutter and
  playback cursor. View-menu visibility toggles; per-lane configs
  persist across sessions.
- **Reference-distribution overlays**: timeline bands drawn on the
  f0 and intensity lanes, with kind-distinct encoding —
  observed = neutral; normative = green; target = amber + dashed +
  "TARGET" tag — enforcing the design rule that the GUI must never
  conflate "what people do" with "what to aim for".
- **Right-side Reference panel**: vowel-space scatter (F1 × F2 with
  phonetic axis orientation, optional phone filter, live "measured
  vowel" diamond at the cursor) + 1-D parameter histogram with
  p5 / p95 / median markers and a red "you" line.

#### ML inference (Cluster E)

- **Bundled Silero VAD** (`E11`): `sadda.ml.vad(audio)` returns
  per-window speech probabilities; `sadda.ml.speech_segments(audio,
  threshold)` merges those into speech regions. The model is bundled
  under `models-bundled/silero-vad/`. The desktop app gains a "VAD
  (speech)" measure-track lane (speech-probability contour, dashed
  threshold, green shading above threshold).
- **Model registry + resolver**: `sadda.ml.load_model(id)` resolves
  `sadda/<name>[@version]` (curated set) or `local://<path>` (a
  directory with `model.toml`, or a bare ONNX file). Parallel
  registry-repo pattern to refdist: `model-registry/` carries the
  registry CI; bundled VAD lives under `models-bundled/silero-vad/`
  with a `model.toml` for tier-1 proof. `sadda.ml.install_model` /
  `get_model` mirror `sadda.refdist`.
- **HF passthrough + checksum verification** (`E12`): in
  `download`-enabled engine builds, `sadda.ml.load_model("hf://<repo>/<file>[@rev]")`
  fetches an ONNX model from HuggingFace into a per-user cache
  (default-off cargo feature `download` keeps the base engine
  network-free per the local-first principle). `verify_checksum`
  helper for `sha256:…` digests.
- **Embedding-extraction harness**: `Model.embeddings(audio)`
  returns a `(frames, dims)` NumPy array for any wav2vec2-style
  (waveform input) or Whisper-style (log-mel input, optional
  fixed-frame padding) ONNX model. The manifest's `[input]` section
  picks the representation, and the harness reads the model's own
  input/output tensor names so it's model-agnostic.
- **Embedding tiers**: `Project.extract_embeddings(bundle_id,
  model_id, tier_name)` runs an embedding model and persists the
  result as a B3 `continuous_vector` tier with the ml-model
  `processing_run` recorded inline.

#### ORT-sidecar packaging (0.3.x polish)

- **Wheel `[ml]` extra**: `pip install "sadda[ml]"` pulls
  `onnxruntime`; the wheel auto-discovers it at import (no manual
  `ORT_DYLIB_PATH` setup), so `sadda.ml.vad(...)` just works.
- **Desktop bundle includes libonnxruntime**: each release archive
  carries an `onnxruntime/` subdirectory with the platform's ORT
  1.22.0 binary + its upstream LICENSE. The app probes
  `<exe-dir>/onnxruntime/` at startup and validates each candidate
  with the engine's `OrtGetApiBase` symbol probe before pointing the
  runtime at it.
- **THIRD_PARTY_NOTICES.md**: new repo-root file with verbatim
  ONNX Runtime + Silero VAD MIT texts.
- **App release artefact shape changes**: from a bare binary to a
  `.tar.gz` (Unix) / `.zip` (Windows) archive containing the app +
  `onnxruntime/` + `THIRD_PARTY_NOTICES.md` + project licenses.

### Fixed

- **ONNX Runtime probe rejects the provider shim**: pointing
  `ORT_DYLIB_PATH` at `libonnxruntime_providers_shared.so` (a valid
  but wrong shared object) now produces a distinct, actionable error
  pre-empting `ort`'s lazy-loader panic. The probe resolves the
  `OrtGetApiBase` symbol after `dlopen` instead of waving any
  shared-object through.
- **WSLg "hang" → window-geometry persistence under WSL**: a
  restored maximized-at-`(-32768, -32768)` window broke winit's
  pointer-coordinate mapping, making clicks land offset from the
  cursor and the window feel frozen. The desktop app now disables
  window-geometry persistence under WSL (`persist_window:
  !is_wsl()`); app state (recent projects, prefs) is unaffected.
- **Engine refdist Parquet reads**: Polars' `write_parquet` defaults
  to zstd + LargeUtf8, but the engine's `parquet` feature was on the
  snap-only path — so the engine couldn't read any refdist data
  file. Added `zstd` to the engine's parquet features and a
  `LargeStringArray` path.

### Notes

- The 0.2.0 release (app-only) doesn't have a matching PyPI wheel —
  it bumped `v*.*.*-app` but the wheel surface didn't change. 0.3.0
  reunifies the version numbers across both tracks.
- Real-voice clinical validation datasets remain a known gap; v1
  carries on synthetic + Praat-anchored fixtures, with SVD
  (CC-BY-4.0) earmarked as the redistributable real-voice option
  when the validation work lands.
- AVQI and ABI absolute values remain **provisional** until
  byte-level confirmation against the authors' artifact lands; the
  components (jitter, shimmer, HNR, CPPS, LTAS slope, …) are
  Praat-validated where a Praat oracle exists.

[0.3.0]: https://github.com/sadda-speech/sadda/releases/tag/v0.3.0

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
