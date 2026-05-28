# DEVLOG

A running log of research, decisions, and development for the SpeechAnalysisTool project — a planned next-generation phonetics / speech-science tool, conceived as a successor to Praat.

Newest entries at the top. Each entry is dated `YYYY-MM-DD` and tagged with a short topic.

---

## 2026-05-28 — GUI embedding-heatmap lane (E12 GUI follow-on)

Finished the E12 surface set for the desktop app: `Project.extract_embeddings` has been landing wav2vec2 / Whisper-encoder outputs as B3 `continuous_vector` tiers for a few sessions; now there's a way to *see* them. New lane stacks at the top of the measure-track strip (directly under the spectrogram) and renders the embedding as a `(dim × time)` heatmap synced with the rest of the timeline.

**Decisions (DEVLOG-style multi-option Qs with the user, all defaults accepted):** colormap = **Cividis** (consistency with the recent accessibility-default for spectrograms — CVD-safe + luminance-monotonic); normalization = **per-dim z-score** (the SSL-probing-paper convention; each row centered + scaled to unit variance with the z-scores clipped to `[-3, +3]` before mapping to `[0, 1]`, so one outlier neuron can't wash out finer structure across the matrix); tier-selection UX = **View menu submenu** (matches the existing pattern for the f0 / formants / intensity / VAD lane toggles + the refdist-overlay submenus).

Why per-dim z-score matters here: SSL encoders very often have a handful of high-magnitude dims that sweep ±10 while most stay near 0. A global min–max colormap then paints the whole matrix a near-uniform mid-grey except for those outliers. Per-dim z-score equalizes the visual contribution of every dim — structure shows up; magnitude differences don't drown it out. Same convention the Pasad/Hsu/Mohamed SSL-probing papers use in their figures; expected by the speech-AI-engineer audience.

### Implementation

`crates/app/src/state.rs` (pure-data): new `EmbeddingHeatmapConfig { selected_tier_id, colormap, normalization }` field on `PersistedState`; `EmbeddingNormalization` enum (`PerDimZScore` | `GlobalZScore` | `GlobalMinMax`) with `PerDimZScore` as `Default`; new `normalize_embedding(matrix, n_dims, n_frames, mode)` that takes a `(n_dims × n_frames)` dim-major buffer and returns the normalized `[0, 1]` buffer ready for the existing `colormap_bake`. Degenerate row (`std == 0`) → midpoint `0.5` instead of NaN. Empty input → empty out. Five new unit tests cover (a) per-dim z-score erases magnitude differences when two rows have the same *shape* scaled 10×, (b) constant row → midpoint, (c) `[-3, +3]` clip really does clip (verified with a 1000× larger outlier still pegging at 1.0), (d) global min–max spans `0..1`, (e) all modes accept empty input.

`crates/app/src/main.rs`: new `EmbeddingHeatmapCache { bundle_id, config, texture, duration_seconds, n_dims, tier_name }` mirrors `SpectrogramCache` — same egui `TextureHandle` story, same staleness key shape, same `build_*_texture` / `rebuild_*_if_stale` split. `build_embedding_heatmap_texture` reads the `continuous_vector` matrix from the engine (`Project::read_continuous_vector` returns `Array2<f64>` shaped `[n_frames, n_dims]`), transposes to dim-major `f32` in one pass (small enough cost vs. the GPU upload), buckets the time axis if `n_frames > MAX_SPECTROGRAM_WIDTH` using the same averaging logic as the spectrogram, normalizes, bakes RGBA via `colormap_bake`, and uploads. The matrix isn't kept — re-reading the Parquet sidecar on a rebuild is cheap and saves the MB-scale memory cost of holding a 768-dim embedding tier in app state.

Rebuild surface: `rebuild_embedding_heatmap_if_stale(ctx)` runs alongside the existing `rebuild_tracks_if_stale` / `rebuild_overlays_if_stale`; no selected tier drops the cache (lane disappears immediately); a build error (selected tier missing across a project reopen, sidecar unreadable) is parked in `embedding_heatmap_error` so the lane renders the message centred instead of blanking out silently. Bundle change / delete clears both `active_embedding_heatmap` and `embedding_heatmap_error` from every relevant site.

Render: `embedding_heatmap_lane_pane` mirrors `spectrogram_pane` exactly — `egui_plot::Plot` with `set_plot_bounds_x` cropped to the timeline view + `set_plot_bounds_y(0..n_dims)` + a `PlotImage` centred at `(duration/2, n_dims/2)`. Cursor line + click-to-seek + zoom/scroll handler come from the same shared helpers. Lane caption shows tier name + dim count. The lane is registered LAST in the measure-track stack so it sits directly under the spectrogram (matches the visual story: two 2D views stacked). Default height is `MEASURE_LANE_HEIGHT × 2` because a 768-dim heatmap wants more vertical real estate than a 1-D contour.

View menu: a new "Embedding heatmap" submenu under View. Lists the active bundle's `continuous_vector` tiers as a one-shot radio with a "(None — hide lane)" entry first, plus colormap (Cividis / Viridis / Magma / Greyscale) + normalization (PerDimZScore / GlobalZScore / GlobalMinMax) sub-pickers. Empty case ("no continuous_vector tiers yet — run Project.extract_embeddings") gets a weak italic hint, so users see the obvious next step.

### Gates + deferred

Full local gate green on rust 1.95: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` (58 app tests now, +5 from the new normalizer suite; everything else unchanged), `pytest python/tests -q` → 148 passed / 6 skipped. The texture-build side isn't unit-tested (mirrors the spectrogram pattern — egui-dependent assembly is exercised by manual run + the pattern is well-trodden across the existing measure-track lanes).

**Deferred:** (a) hover tooltip showing `(time, dim, raw_value)` — would need access to the un-normalized matrix at render time, easy to add by holding `Vec<f32>` next to the texture if asked; (b) dim sorting (by variance, or by clustering) for finding structure faster in a high-dim model; (c) layer-comparison view stacking N heatmaps for different layers of the same SSL model — the registry-side change to store layer-indexed embedding tiers is a separate slice; (d) a colorbar legend in the lane gutter — `egui_plot` doesn't have a built-in for this, would be drawn manually. None blocking; all reasonable follow-ons if the feature gets used.

---

## 2026-05-28 — ORT-sidecar packaging: ML "just works" for `pip install sadda[ml]` + app release bundles ship libonnxruntime

Picked up the explicit release-engineering follow-on the previous 2026-05-28 entry deferred: making ONNX Runtime *available* to the wheel and the app binary without any manual `ORT_DYLIB_PATH` setup, so VAD / embeddings work out-of-the-box for users on the supported install paths. Two distribution channels, two sourcing strategies, one shared mental model — both end up at "the engine probe accepts whatever path we land on, or you get the same clean error you got before."

**Decisions made first (DEVLOG-style multi-option Qs with the user):** ORT source = **Microsoft's GitHub releases** (canonical, SHA-pinned, MIT-redistributable); wheel approach = **`[ml]` extra dep on `onnxruntime`** (leverages PyPI's existing per-platform wheels — no bundling matrix on our side, base install stays lean); app platform scope = **Linux x64 + macOS arm64 + Win x64** (the three the existing app-release matrix already runs); bundle layout next to the exe = **`<exe-dir>/onnxruntime/`** (named subdir, matches Microsoft's tarball layout, leaves room for future bundled libs).

### Python wheel — auto-discovery (`python/sadda/ml/__init__.py`)

`pyproject.toml` gets `[project.optional-dependencies] ml = ["onnxruntime>=1.22"]` — `pip install sadda[ml]` pulls the matching ORT wheel for the user's platform. At first import of `sadda.ml` the new `_discover_ort_dylib()` looks at `Path(onnxruntime.__file__).parent / "capi"`, globs the platform-appropriate name (`libonnxruntime.so*` Linux / `libonnxruntime*.dylib` macOS / `onnxruntime*.dll` Windows), excludes any `providers_shared` filename (the engine's probe rejects it anyway with a pointed error, but skipping the round trip is cheap), and prefers the longest filename — so a versioned `libonnxruntime.so.1.26.0` beats a bare symlink in mixed layouts. If `ORT_DYLIB_PATH` is already set we never override; if `onnxruntime` isn't importable the discovery is silent and `vad()`/`embeddings()` raise the same clean "set `ORT_DYLIB_PATH`" error as before. Lower bound `>=1.22` matches the API version `ort` 2.0.0-rc.10 was built against (the C API is backward-compatible, so newer ORT also works).

The discovery lives in the `sadda.ml` module rather than top-level `sadda/__init__.py` so it sits next to the code that consumes it — but since the package's `__init__` imports `sadda.ml`, both placements end up triggering at the same time anyway.

### App binary — sibling-sidecar discovery (`crates/app/src/main.rs`)

Release bundles place the runtime under `<exe-dir>/onnxruntime/`; the new `discover_ort_sidecar()` runs at the top of `main()` (next to `force_x11_under_wsl()`), reads `current_exe()`, walks `parent()/onnxruntime/`, applies the same platform-specific filename match + providers-shim exclusion + longest-name preference, **and validates each candidate with the engine's probe** before setting the env var — so a wrong file at the right location is filtered out at startup rather than blowing up at first VAD call.

`engine::ml::probe_ort_dylib` was already doing the `OrtGetApiBase` symbol-verify added yesterday; it just had to flip from crate-private to `pub` and re-export through `sadda_engine::probe_ort_dylib` (behind the `ml` feature). Factored the directory-scan half into `find_ort_in_dir(&Path) -> Option<PathBuf>` (pure, deterministic) so it's unit-testable without touching the process environment: two new tests confirm a missing dir returns `None` and a zero-byte file with a runtime-shaped name is rejected by the probe and discovery returns `None`. The env-mutation wrapper stays `unsafe { set_var }` and is exercised only via the app's startup path.

### Release workflow — download + verify + bundle (`.github/workflows/app-release.yml`)

ONNX Runtime pinned to **1.22.0** (`ORT_VERSION` env var at the workflow level). Per-platform matrix entries carry the archive filename, **SHA-256** (verified locally against the upstream `microsoft/onnxruntime` v1.22.0 release: `8344d55f…` linux-x64 / `cab6dcbd…` osx-arm64 / `174c616e…` win-x64), and the relative path inside the archive to the one runtime library we ship (the providers shim, debug `.pdb`/`.dSYM`, cmake files, etc. are dropped — Windows in particular has a 357 MB `.pdb` we definitely don't want). New steps:

1. **Download + verify** — `curl -fL` the archive, sha256 via Python's `hashlib` (portable across `runner.os` without bringing in `shasum`/`sha256sum`/`Get-FileHash` divergences), fail with `::error::` on mismatch.
2. **Stage release bundle** — extract with `tar -xzf` (Unix) or `unzip -q` (Windows, also via `shell: bash` on the Win runner), copy *only* the one runtime library into `bundle/<asset>/onnxruntime/` along with the upstream `LICENSE`, then drop the app exe + `THIRD_PARTY_NOTICES.md` + `README.md` + `LICENSE-APACHE` + `LICENSE-MIT` alongside the binary.
3. **Pack archive** — `tar -czf` for Unix targets, `shutil.make_archive` (Python) for Windows so the script is identical across runners. Asset name extension becomes `.tar.gz` / `.zip` — the release artifact shape changes from a bare executable to a directory archive, which is fine for the 0.3.2 binary cut (no users on the old format yet).

`if-no-files-found: error` on the upload + `fail_on_unmatched_files` on the release step both still hold; the publish job is unchanged.

### `THIRD_PARTY_NOTICES.md`

New repo-root file listing ONNX Runtime (MIT, v1.22.0, upstream URL, in-place location in the bundle) and Silero VAD (MIT, v6.2.1, bundled under `models-bundled/silero-vad/`). License texts reproduced verbatim from the canonical upstream sources (diffed against `models-bundled/silero-vad/LICENSE` and the extracted ORT `LICENSE` — matches modulo a trailing newline). Per the canonical-sources principle ([[feedback-canonical-sources]]).

### CI gates

Full local gate on rust 1.95: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings` (also `-p sadda-engine --features download` so the network-feature path stays clippy-clean), `cargo test --workspace` green (engine 148, app 53 incl. the two new sidecar tests; ORT-gated suites pass with `ORT_DYLIB_PATH` set to the conda runtime), `pytest python/tests -q` → 148 passed / 6 skipped (the new `test_ort_dylib_autodiscovery` skips cleanly when `onnxruntime` isn't installed in the .venv, identical pattern to the existing ml-suite skips).

Also fixed an unrelated stale assertion in `release.yml`: the wheel's `CIBW_TEST_COMMAND` was still asserting `sadda.version().startswith('0.1')` from the 0.1 cut. Relaxed to `'0.'` so it tracks the broader 0.x line and doesn't fail on every release bump.

### What this slice deliberately doesn't do

- No source-build of ORT in CI; we use Microsoft's prebuilt binaries (the spike entry from 2026-05-26 already settled this).
- No bundling of ORT inside the wheel itself (rejected option in the design Q: `[ml]` extra dep is leaner, leverages PyPI's existing per-platform wheels, and lets users skip the ~25 MB if they're not using VAD/embeddings).
- No platform expansion beyond the existing matrix; Linux-arm64 / Win-arm64 / macOS-x64 wait until there's user demand.
- No ORT-sidecar packaging *inside* the Python wheel beyond the `[ml]` extra — if a user wants `pip install sadda` without internet they can still set `ORT_DYLIB_PATH` manually, same as before.

The two earlier deferred items (auto-discovery + sidecar packaging) collapse into this one slice, both now closed.

---



With the GUI "hang" resolved (entry below), turned to the ONNX Runtime issue surfaced during that same session: the user's `ORT_DYLIB_PATH` had been pointed at `libonnxruntime_providers_shared.so` — the small ORT **provider shim**, not the runtime.

**The trap.** `ensure_ort_available()` (the no-panic guard that runs before any `ort` call, since `ort` *aborts* in its lazy loader on a bad runtime) only checked that the path `dlopen`s. But the provider shim is a perfectly valid shared object — it opens cleanly — it just contains none of the ORT C API. So the probe gave a **green light**, then `ort` failed downstream with an opaque error. The probe was testing the wrong thing: "is this *a* library?" instead of "is this *the runtime*?"

Confirmed on the machine with `nm -D`: the real `libonnxruntime.so.1.26.0` exports `OrtGetApiBase` (the ORT C API entry point); the provider shim does not. (Also noted: conda ships only the versioned `libonnxruntime.so.1.26.0`, no unversioned symlink, so the unset-path fallback `"libonnxruntime.so"` can't resolve there either — but that's an env-setup matter the error message now addresses, not a code bug.)

**Fix (`crates/engine/src/ml.rs`).** Extracted `probe_ort_dylib(path)` from the env-reading `ensure_ort_available()` (so it's testable without touching the process environment). After `dlopen` succeeds it now resolves the `OrtGetApiBase` symbol; if absent, it returns a distinct, actionable `EngineError::Ml` — names the likely cause (the provider shim) and says to point `ORT_DYLIB_PATH` at the runtime itself, noting a versioned filename like `libonnxruntime.so.1.26.0` is fine. Two failure modes → two messages ("not loadable" vs. "loads but isn't the runtime"). The symbol is only *resolved*, never called.

This is a shared engine path, so the better error reaches all three surfaces at once — engine VAD/embeddings, Python `sadda.ml`, and the GUI VAD lane — no per-surface change needed.

**Tests** (`ml::tests`, both pure / ORT-free, run under the standard `cargo test --workspace` gate): a missing path → the "not loadable" error; a guaranteed-present system C library (`libc.so.6` / `libSystem.B.dylib` / `kernel32.dll`) → the `OrtGetApiBase` rejection — a portable stand-in for "wrong `.so`" needing no ONNX Runtime (the test executable can't be used: Linux refuses to `dlopen` a PIE). Verified the happy path is untouched: with `ORT_DYLIB_PATH` set to the real runtime, the ORT-gated `vad_bundled_runs_on_silence` and the full ml suite pass — the real lib clears the new symbol check.

Full gate green on 1.95: fmt, clippy (workspace `--all-targets` + `-p sadda-engine --features ml`, `-D warnings`), `cargo test --workspace` (all groups; the 148-test engine-lib group includes the two new probe tests). (The `sadda-app` test binary needs conda's `libpython3.12.so.1.0` on `LD_LIBRARY_PATH` to run in a bare shell — a pre-existing harness-env note, unrelated to this change.)

**Deferred (unchanged).** Auto-discovering a pip/conda-installed `libonnxruntime.so.N` when `ORT_DYLIB_PATH` is unset, and the per-platform ORT-sidecar packaging for the 0.3.2 binaries, both remain open release-engineering items — the user scoped this slice to the probe.

---

## 2026-05-27 — WSLg GUI "hang" — *actual* root cause: restored window geometry, not GL (fixes the entry below)

The "WSLg GL hang" diagnosis in the entry below was **wrong on every count**. Ran it to ground this session with the user at the machine; the real story:

**Symptom (refined by the user, twice).** The window *does* open and *does* paint the welcome screen — it isn't off-screen and isn't a GL stall. First report was "totally dead to clicks"; the decisive follow-up was that hover highlights fire but **offset from the cursor** — you must hover *above* a Recent-projects row to light it up. That's a pointer-coordinate mismatch, not a freeze.

**Root cause.** eframe's `persistence` feature restores a saved window geometry, and the saved value was `maximized:true` at the INT16 position sentinel `(-32768,-32768)` (confirmed in `~/.local/share/sadda/app.ron`). Under WSLg/Weston the maximized window is shown filling the screen, but **winit believes the window origin is at `-32768,-32768`**, so it translates incoming pointer coordinates against that bogus origin — hit-testing lands offset from where widgets are drawn, so clicks miss and it feels frozen. It **compounds**: the bad position is re-saved every run.

**Why the earlier guesses were wrong.** (a) Renderer is **wgpu → D3D12** (`libd3d12core.so`, `swrast_dri` fallback in the lldb backtrace), *not* glow — the Cargo.toml description was right; the old entry's "glow" claim was unchecked. (b) `Dropped Escape call … 0x03007703` is a benign WSLg GL-virtualisation log line, not the hang. (c) Not ORT (lazy; never at startup). (d) Not conda-lib shadowing: proved with `LD_DEBUG` that the alias loads conda's `libX11/libxcb/libEGL` vs system, isolated libpython to force the WSLg libs — **still froze**, ruling it out. (e) `wsl --shutdown` couldn't have helped: nothing to do with the GPU session.

**How it was actually found.** `rust-lldb` (had to *launch* the app — `ptrace_scope=1` blocks attach) caught the main thread blocked in `winit::Window::is_minimized()` → a synchronous X11 `GetProperty` round-trip via `eframe::is_invisible_or_minimized` (called every paint). That X stall is the *same* Xwayland flakiness the bogus window state induces. `xwininfo`/`xprop` then showed the maximized `-32768` geometry, and `app.ron` confirmed the persisted source.

**Fix (shipped).** `crates/app/src/main.rs`: extracted `is_wsl()` (shared with `force_x11_under_wsl`) and set `NativeOptions { persist_window: !is_wsl(), .. }`. eframe's `persist_window` gates **saving only** — it restores an existing key unconditionally (`create_window` → `load_window_settings`/`apply_window_settings`, `wgpu_integration.rs:997/1007`) — so the flag alone stops the compounding re-saves, and clearing the stale key once leaves nothing to restore. App state (recent projects, prefs) is untouched; it rides the Storage trait, not this flag. Verified by injecting the pathological geometry (still applied → confirms load isn't gated), then clearing + running: window opens windowed on-screen, `_NET_WM_STATE` empty, and **no `"window"` key is written back**. Off-WSL behaviour unchanged.

**Deferred.** eframe restoring geometry unconditionally is a latent footgun if a stale key ever reappears (old build, restored backup). A belt-and-suspenders startup-strip of the `"window"` key under WSL (`eframe::storage_dir(APP_TITLE).join("app.ron")`) was considered and **not** done — it means hand-editing eframe's RON storage from `main()`, fragile for an edge case that `persist_window:false` already prevents in normal use.

---

## 2026-05-27 — GUI: keyboard scrubbing + accessibility (palette / colormap / UI scale); ~~WSLg renderer note~~ (§3 superseded — see entry above)

Three user asks from a session of hands-on testing. Two shipped; one is a host-environment diagnosis pending the user's confirmation.

### 1. Keyboard scrubbing (shipped)

When zoomed into a portion of a recording there was no *discoverable* way to move along the file — shift+scroll-wheel panned (since C5's `handle_zoom_and_scroll`) but nothing advertised it. Added **Left/Right** to pan a quarter-window per press and **Home/End** to snap to the file start/end, built on `TimelineState::scroll_by` (already clamps against bundle bounds + unit-tested). Suppressed while any widget has focus or a label is being edited, so the keys still reach text editors.

### 2. Accessibility: plot colours + font size (shipped)

The user asked for accessible plot palettes + font-size control; chose **CVD-safe presets** over full per-element pickers. Implemented as an Appearance group under View → Theme, all persisted:

- **Plot palette** (`PlotPalette`: Default / Okabe–Ito). The colourblind-safe scheme recolours only where colour must be *discriminated*: the overlaid formants F1…Fn (one shared lane) and the coexisting observed/normative/target reference bands. Single-series lanes (f0, intensity, VAD) are unambiguous and left alone; band meaning stays carried redundantly by the TARGET tag + dashed border, never colour alone. Palette: Okabe & Ito, "Color Universal Design" (2008), <https://jfly.uni-koeln.de/color/>.
- **Spectrogram colormap** gains **Cividis** (CVD-optimised, luminance-monotonic) alongside the existing Viridis/Magma/Greyscale — `colorous::CIVIDIS`, one arm in `sample_colormap`, picked up by the existing `==`-based cache invalidation.
- **UI scale** slider (0.8–2.0×) via egui `zoom_factor`, scaling all text + widgets relative to native density.

`state.rs` stays egui-free (the enum + serde defaults live there; colour resolution stays in `main.rs`). Tests: palette schemes non-aliasing + internally distinct; Cividis sampling distinct; appearance defaults deserialise to native scale (guarding the serde default against f32's `0.0`, which would shrink the UI to nothing). Deferred (logged): per-element colour pickers if fine control is wanted later; recolouring the single-series lanes for full scheme coherence.

### 3. WSLg GL hang (diagnosis, pending user confirmation) — ❌ SUPERSEDED

> **Correction (later same day):** this diagnosis was wrong — see the top-of-log entry "WSLg GUI 'hang' — *actual* root cause: restored window geometry, not GL". The renderer is wgpu/D3D12 (not glow), the `Dropped Escape call` line is benign, and the real cause was a restored maximized/off-screen window geometry breaking winit's pointer mapping. Left below as a record of the wrong turn.

User reported the app hanging at launch after `Dropped Escape call with ulEscapeCode : 0x03007703`. Diagnosis: that line is a WSLg DirectX/GL-virtualisation message emitted during GL context creation; the app renders via **glow (OpenGL)** (eframe with only the `persistence` feature → default renderer). It is **not** ORT (load-dynamic, loaded lazily on first VAD/embedding call, never at startup — and the user's `ORT_DYLIB_PATH` pointed at `libonnxruntime_providers_shared.so`, the tiny provider shim, not `libonnxruntime.so.1.26.0`) and **not** the new window icon (decoded pre-`run_native`, before the GL message). Mitigations offered, in order: `wsl --shutdown` then relaunch; `LIBGL_ALWAYS_SOFTWARE=1` to isolate the hardware-GL path; if persistent, switch the app to the **wgpu** renderer (the original stack plan) or make the renderer env-selectable. Awaiting which one yields a window before hardening.

---

## 2026-05-26 — Roadmap intake: AI-agent-native surface + auto-generated walkthrough demos

Two roadmap additions raised by the user, **logged not designed** (like the 2026-05-25 bundle-rename / TikZ-export intake). Neither is in the Phase 3 plan; each needs a dedicated design session and decomposition into slices when slotted (likely Phase 4+ / v1.x). Captured here with prior art, how they ride on what already exists, and the questions to settle.

### 1. AI-agent-native infrastructure

**Thesis.** As AI agents become routine collaborators, sadda should be a tool an agent can drive *naturally* for the speech-science work we already support — acoustic measurement, corpus-scale analysis, statistical modelling, reports/figures. This is squarely the group-2 (speech-AI-engineer) audience from the 2026-05-16 survey, and an extension of the "scriptable like Parselmouth, ML-native" thesis. The user is themselves this audience.

**Most of the substrate already exists** — the gap is an *agent-shaped* surface over it:
- Python API (`sadda.*`, Polars-native, typed units, stability tiers, `.pyi` stubs) — the natural agent substrate.
- Embedded scripting (E8) + `sadda.app` command registration + palette (E9).
- Recipe record/replay (F1) — reproducible pipelines.
- Provenance + citation export (A1) — auditable, reproducible analyses (critical for agent-generated work).
- Corpus model + dense/sparse tiers + refdist — the data substrate for batch work.

**Candidate infra to think through (the user's list, organised):**
- **MCP server** (Model Context Protocol) — the modern "tool use" infra: expose sadda capabilities (open/scan corpus, measure features, query refdist, run a model, generate a report/figure) as MCP tools so any MCP-capable agent can drive it. Probably the centrepiece.
- **A supported `SKILL.md`** (Claude-skill format) — teaches an agent the common workflows ("measure jitter/shimmer/CPPS over this corpus, compare to a normative refdist, write a report"). Explicitly requested.
- **High-level task API + structured outputs** — batch acoustic measurement over a corpus → tidy DataFrame; one-call "measure set X over corpus Y"; agent-legible errors that say what to do next. Builds on the existing Polars/typed-unit returns.
- **Statistical modelling of speech features** — integration points (Polars → statsmodels/scikit/R) or a thin built-in modelling layer; overlaps the AI-engineer audience's existing tooling.
- **Reports + figure generation** — programmatic, headless report assembly (markdown/HTML/PDF) + figures; ties directly to the deferred **TikZ/specTeX figure-export** item (a scriptable figure IR serves both humans and agents).
- **Headless / sandboxable operation** — agents use the engine/Python path, not the GUI; keep that path first-class and consider a CLI front.

**Open questions for the design session:** MCP scope + transport + auth/sandboxing for agent-driven file/corpus ops; how much is genuinely new vs. polishing the Python API into task-level calls; report + figure IR (shared with the TikZ item); whether agents ever drive the GUI or strictly the headless API; safety posture for destructive corpus operations. Cross-cutting — not a single slice.

### 2. Auto-generated walkthrough demos (instruction + testing)

**Idea.** A full screen-capture walkthrough of every major module/feature, ideally **auto-generated from documentation/markup**, with **synthesized voiceover** — serving *both* onboarding/instruction *and* end-to-end UI testing.

**Why it's two birds:** the GUI rendering currently has no automated coverage (flagged in the D10 entry — panes are validated by mirroring tested code, not by driving the UI). A scripted walkthrough that drives the app and captures frames is, with visual diffing, also a regression test of the whole UI.

**Prior art / shape:**
- **Praat's Demo Window** (already noted in the 2026-05-16 survey) — scripted GUI demonstrations; the closest domain analogue.
- A declarative **tour spec** (ordered steps: open project → select bundle → show spectrogram → toggle f0 track → add a refdist overlay → …), each step carrying narration text. Co-locating the spec with the mkdocs feature pages keeps docs and demos in sync ("auto-generated from markup").
- **GUI driving**: egui is immediate-mode, so this likely means a built-in *tour/demo runner* that scripts app state + injects events (the `sadda.app` scripting from E9 and the recipe replay from F1 are the closest existing levers), plus a screen recorder (ffmpeg) and **TTS** for the voiceover from the same narration text.
- Terminal-side analogues (VHS / asciinema) for the Python/CLI surface.

**Open questions:** tour-spec format + home (docs-embedded vs separate); the egui-driving mechanism (in-app demo runner vs external input injection); capture + assembly pipeline; TTS engine + voice licensing; degree of auto-generation vs hand-authoring; short per-feature clips for the docs site vs one long walkthrough. Likely a Phase 6/7 (polish + docs) investment, or an incremental tooling track once the GUI surface stabilises.

### Status

Both are roadmap intake only. The immediate path is unchanged: finish **E11** (ML inference) and the rest of Phase 3. Revisit each with its own design session when scheduled.

---

## 2026-05-27 — Roadmap decision: in-app community surface = link-out (full chat declined)

Explored a user-raised idea — an in-app community chat — and **made the call**: do **not** build real-time chat; meet the underlying goal (in-app community connection) with a lightweight **link-out surface**. The reasoning is recorded so it isn't re-litigated. The feature itself is ~Phase 4+/v1.x; this entry settles its *shape*.

> **Decision (2026-05-27, with the maintainer):** adopt the lightweight, opt-in **"Community" link-out surface** as sadda's community-connection approach. Full real-time in-app chat is **declined**. The read-only Matrix mirror and the thin Matrix client are **deferred / not planned** unless explicitly revisited.

**Why not full in-app chat.** It's a 24/7 *service* (relay / presence / history / storage), not a feature; it needs moderation (a legal + human burden — an empty or toxic chat is worse than none); and it collides head-on with sadda's local-first / no-telemetry / explicit-opt-in-network ethos (principle #10) and its privacy-non-negotiable audiences (clinical/HIPAA, forensic/chain-of-custody, field/data-sovereignty), who would be alarmed by a speech tool holding a live chat connection. Plus version-skew/rot and community cold-start/fragmentation. Prior art: scientific/desktop tools (Praat, R, Blender) link out; the ones that embed chat are cloud products (Figma, VS Code Live Share).

**Adopted — a lightweight, opt-in "Community" surface.** Deep-links out to existing async, platform-moderated channels (GitHub Discussions, a Matrix/Zulip room, the docs), plus a context-aware "ask the community" (a pre-filled Discussion with sadda version + OS and opt-in, sanitized, no-data context). Zero infra, zero moderation burden, no data leaves, no live connection — ~90% of the value at ~1% of the cost. Composes with the registries' PR-discussion surfaces.

**The "mirror a Slack server" variant (user idea).** Rather than build chat, *embed/mirror* an existing community server. By platform:
- **Matrix is the clean answer** — open, federated, built for clients/embedding, and it **bridges Slack/Discord into a Matrix room**. So "mirror Slack" is best realized as: the community lives on Slack/Discord/Matrix → a Matrix bridge mirrors it → sadda is a thin Matrix client. You ride Matrix + its bridges; you build no server.
- **Slack/Discord directly = the bad path** — closed APIs, OAuth-per-user, ToS restrictions on custom clients re-displaying content, free-tier history limits. Slack is actively hostile to mirroring; Discord offers only a limited read-only widget.
- **Read-only mirror** (an in-app "community feed" of recent public messages + a "join the conversation →" link-out) is the lighter, lower-risk form vs. in-app participation: no compose UI ⇒ no in-app abuse vector, smaller moderation surface, still makes the community feel present. Full participation is the thin Matrix client.

**Deferred "if real chat is ever wanted": a thin Matrix client** (federated, E2E, self-hostable, Matrix's own moderation tooling; Matrix-bridges-Slack/Discord if the community lives there). The only architecture compatible with the privacy ethos — but it still carries version + human-moderation costs, so it's "later, maybe," off-by-default + opt-in.

**Open questions.** Which channels are canonical; whether a read-only Matrix-mirror feed is worth it (auth for non-public rooms, rate limits, the live-connection opt-in UX vs principle #10); moderation delegation; webview-vs-native for any embed.

**Status.** Approach **decided** (link-out surface); the build is ~Phase 4+ and small. Immediate path unchanged: the 0.3.x release, then Phase 4.

---

## 2026-05-27 — Roadmap intake: community script registry

A user-raised feature, logged-not-designed (like the AI-agent-surface / walkthrough-demos / AI-engineer-directions intakes); ~Phase 4+/v1.x; needs its own design session.

**Thesis.** Sadda is scriptable (the E8 embedded-CPython panel + E9 `sadda.app` command registration + the Python API). A **script registry** lets users publish the scripts they write back to the community — the central home the Praat-script ecosystem never had (scripts scattered across forums + personal sites). It is the project's **third registry**, reusing the established pattern: a separate repo, PR-based submission, manifest + artifact, CI validation, a Pages index, in-app discovery.

**Submission pathway (the user's sketch).** Publishing is a PR adding a script + a manifest + docs; the **review is the QA gate** and ensures documentation. It needs *very clear* submission instructions, ideally a **walk-through** (ties directly to the auto-generated-walkthrough-demos intake).

**The distinguishing concern — scripts are executable code.** Unlike the refdist registry (inert data) and the model registry (weights run by a runtime we control), a shared script is **arbitrary code that runs on the user's machine**. So this registry carries a trust/security burden the others don't: review can't fully vouch for safety; the app should signal trust clearly and likely **sandbox / require explicit consent** before running a community script (overlaps the agent-safety/sandboxing open question from the AI-agent-surface intake); and CI "validation" leans toward lint / import-check / a sandboxed smoke-run rather than data conformance. This is the part the design session must get right.

**The promotion loop (a good dynamic the user flagged).** The review process surfaces scripts that prove broadly useful — a documented path for *"popular community script → candidate to build into the engine itself."* Scripts become a proving ground for built-in features (the way shell plugins / git aliases graduate into core).

**Connections.** Scripting surface (E8 / E9 / Python API); the registry mechanism — refdist (C8) + models (E11-3b) + now scripts = **three** registries, which materially strengthens the deferred *parallel-vs-shared registry-core* reassessment (three concrete instances to generalize from); the submission walk-through (#2); in-app browse / install / run (a "script browser," mirroring a refdist/model browser); provenance + attribution for shared scripts.

**Open questions for the design session.** Manifest shape (`script.toml`: title / author / license / `sadda`-version compat / APIs-used / tags / example invocation); curated-tiers vs flat-with-trust-signals; the **security model** (sandbox? capability limits? consent-to-run UX; static checks vs sandboxed smoke-run in CI); how the "promote to engine" path is tracked (labels / issues); and versioning + compat as the `sadda.*` API evolves under shared scripts.

**Status.** Roadmap intake only; ~Phase 4+. Immediate path unchanged: the 0.3.x release, then Phase 4.

---

## 2026-05-27 — E12b-2b: fixed-length mel + real-model validation (E12 / cluster E complete)

The "both" the maintainer asked for: a fixed-length-mel harness feature, plus the embedding harness validated against two *real* models (one per representation). This closes E12, cluster E, and Phase 3's ML line.

### What landed

- **`input.fixed_frames`** (manifest): pad (with the log-mel floor of silence) / truncate the log-mel spectrogram to exactly N frames before inference — the contract fixed-length encoders like Whisper (3000 frames) need. Absent ⇒ variable length. CI-runnable test (ORT-gated, no network): the log-mel fixture padded to 200 and truncated to 20 frames yields exactly that many embedding rows.

### Real-model validation (gated behind `SADDA_NET_TESTS`, never in CI)

Both downloaded via `hf://` and run through the harness with **no code change** — proving the "manifest-declares, harness-handles" design on real models:

- **wav2vec2-base-960h** (waveform, ~378 MB) → harness runs it end-to-end, producing a `frames × dims` matrix. (Note: `-960h` is the CTC ASR fine-tune, so its output is char-vocab logits, not 768-d SSL hidden states — a pretrained `wav2vec2-base` would give those; the test asserts the shape generally.)
- **whisper-tiny.en encoder** (log-mel, ~33 MB) → with `representation = log_mel` + `fixed_frames = 3000`, the harness shapes the mel and runs the encoder to `[1500, 384]`. Caveat documented: `dsp::log_mel` is Slaney/natural-log, not Whisper's exact HTK/log10/normalized mel, so this validates the harness *mechanics* (fixed-length shaping → run → output), not embedding fidelity — model-exact mel preprocessing is a future per-model concern.

### Validation

fmt, clippy default + `--features download`, `cargo test --workspace` (22 groups; the `fixed_frames` test runs + ORT-skips), all green on 1.95. The two real-model tests pass locally with `SADDA_NET_TESTS=1` + ORT.

### Cluster E / Phase-3 status

**E12 done → cluster E (ML inference) complete → Phase 3 (clusters A–E) content-complete.** The 0.3 line — clinical substrate + clinical algorithms + reference distributions + GUI overlays + ML inference — is now all landed. Deferred follow-ups remain (a `download`-enabled wheel/extra; a GUI embedding-heatmap lane; curated url-fetch with checksum verify; the parallel-vs-shared registry-architecture reassessment; model-exact mel preprocessing), but the phase's headline scope is met.

---

## 2026-05-27 — E12b-2a: embedding tiers — `Project::extract_embeddings` + provenance

Wires the harness into the corpus: inference results become first-class, queryable dense tiers — the "embedding tiers" half of E12.

### What landed (engine + Python)

- **`Project::extract_embeddings(bundle_id, &Model, tier_name) -> tier_id`** (cfg `ml`): loads the bundle's audio, runs `Model::embeddings`, creates a `continuous_vector` tier, writes the `(frames × dims)` matrix as a B3 Parquet sidecar (frame rate = actual `frames / duration`), and **records an `ml_model` `ProcessingRun`** (processor = the model id, `weights_checksum` populated, `model_version` in `parameters`) — so the tier's lineage is queryable via the A1 provenance timeline.
- **Python `Project.extract_embeddings(bundle_id, model_id, tier_name)`** — takes a model *id string* (`sadda/…` / `local://…` / `hf://…`), resolved via `load_model` internally. Taking the id rather than a `Model` object keeps it inside the `gen_stub`-typed methods (a `PyModel` param would need `PyStubType`); the stub gains the method (committed). The result is then a normal `proj.query(tier_id)` → Polars DataFrame.

### Validation

A gated engine integration test (`tests/ml_embeddings.rs`): create a project + bundle, run `extract_embeddings` with the waveform fixture, assert the tier is `continuous_vector`, reads back as `frames × 8`, **and** an `ml_model` ProcessingRun naming the model was recorded — ORT-gated skip. A Python test mirrors it (`proj.extract_embeddings(local://fixture)` → non-empty `query`). Gates green on 1.95: fmt, clippy default + `--features download`, `cargo test --workspace` (22 groups), stub regenerated (the one new `Project` method), pytest 148 pass + 5 ORT-skip.

### E12b-2b next

The real-model gated validation (the "both"): wav2vec2-base (waveform) + whisper-tiny encoder (log-mel) via `hf://`, behind `SADDA_NET_TESTS`.

---

## 2026-05-27 — E12b-1: the embedding harness (representation-driven), validated with synthetic fixtures

The general embedding harness (maintainer chose "general harness, validate with one consumer of each input kind"). A new model runs as an embedding extractor with **no code change** — the manifest declares its input representation, and the harness does the preprocessing.

### What landed (engine + Python)

- **Manifest `[input].representation`** + mel params: `waveform` (raw mono `[1, N]`; wav2vec2/HuBERT) or `log_mel` (`[1, n_mels, T]`; the Whisper-encoder contract, with `n_mels` / `n_fft` / `hop_length`).
- **`dsp::log_mel`** — the pre-DCT stage of `mfcc` (Slaney mel-filterbank energies → log), reused by the harness.
- **`Model::embeddings(audio) -> Array2<f64>`** (`frames × dims`): mono-mix → resample to `input.sample_rate_hz` → shape per `representation` → run the ONNX session → output as `[batch, frames, dims]`. The model's **own input/output tensor names are read from the session**, so the harness is model-agnostic (wav2vec2 `input_values`, Whisper `mel`, etc. all work).
- **Python `Model.embeddings(audio) -> np.ndarray[float64, 2]`**.

### Validation — one consumer of each representation (synthetic fixtures)

`crates/engine/tests/ml_fixtures/` (generated by `make_fixtures.py`, torch→ONNX): **waveform-embed** (`[1,N]→[1,T,8]`) and **logmel-embed** (`[1,80,T]→[1,T,8]`) — tiny (3–13 KB) committed ONNX models, one per input representation. Engine tests run `embeddings()` through each (ORT-gated: skip cleanly without ORT) and assert the `frames × 8` shape — proving both preprocessing paths + the session run + output reshape. With ORT both pass; a Python fixture test mirrors it via `local://`.

### Validation green

fmt, clippy `--workspace --all-targets` + `--features download`, `cargo test --workspace` (21 groups), `--features download` (12), stub unchanged, **pytest 148 pass + 4 ORT-skip** — all on 1.95.

### E12b-2 next

`Project::extract_embeddings(bundle_id, model, tier_name)` → write the matrix as a B3 `continuous_vector` dense tier + record an `ml_model` `ProcessingRun` (the provenance hook, at a project-aware call site) + Python; and the **real-model gated validation** — wav2vec2-base (waveform) + whisper-tiny encoder (log-mel) via `hf://` (the "both" the maintainer asked for; large gated downloads). A GUI embedding view (heatmap lane) is a later, separate piece.

---

## 2026-05-27 — E12a: on-demand model download (`hf://` + checksum), behind the `download` feature

Implements the spike's recommended path (maintainer-approved). The engine gains its first network capability — fetching model weights on demand — **strictly opt-in** (architectural principle #10): a default-OFF `download` cargo feature that no workspace member enables, so the base engine, the Python wheel, and the app stay network-free.

### What landed (engine only)

- **`download` feature** (`= ["ml", "dep:ureq"]`, default OFF). `ureq` 3 with `default-features = false, features = ["rustls"]` → rustls (ring) TLS, **no gzip/cookies, no async runtime**. Network-free builds are unchanged.
- **`load_model("hf://<org>/<name>/<file>[@<rev>]")`** — with `download`, fetches the file into `~/.local/share/sadda/models/hf/<repo>/<rev>/<file>` (skipped if cached) and returns a runnable `Model`; without the feature, a clear "needs `download`" error (not a silent no-op). HF passthrough is **unverified/uncurated** (no manifest — the trust tier the 2026-05-20 entry calls out); auth via `HF_TOKEN`.
- **`download_file`** streams to a `.part` temp then renames (no partial file ever looks cached); raises a clean error on HTTP failure. **`verify_checksum(path, "sha256:…")`** (pub, `sha2`) is the trust check for curated/fetched weights — the HF silero copy's sha256 ≠ the bundled pip copy's, so checksums are pinned **per source**.
- No new Python/GUI surface: the wheel doesn't enable `download` (per the approved network-free-wheel decision), so `sadda.ml.load_model("hf://…")` returns the clean "needs download" error there. A `download`-enabled wheel/extra is a later packaging decision.

### Validation

Engine: non-network units (`verify_checksum` match/mismatch, `parse_hf_id`, `hf_resolve_url`, and the no-feature error path) run in the default + `--features download` builds. A real end-to-end test (gated behind `SADDA_NET_TESTS`) downloaded the public HF silero-vad ONNX and ran VAD on it — **`hf://` → fetch → `Model.vad` in 1.2 s**. CI gains a `cargo clippy + test -p sadda-engine --features download` step (network/ORT tests skip there). Full local gate sequence green on 1.95 (fmt, clippy default + `--features download`, build, `cargo test --workspace`, pytest 11 pass + 3 ORT-skip).

### Deferred to E12b / later

- **Embedding tiers** — inference output (wav2vec2/HuBERT) → B3 `continuous_vector` dense tiers (local; the next E12 slice).
- **Curated url-fetch** — `sadda/…` entries whose `model.toml` declares `url` + `file_checksum` (vs. the placeholders' `example.invalid`): download into the store dir and **verify against the manifest checksum** (the `verify_checksum` path, exercised once real curated entries exist).
- A `download`-enabled wheel/extra; `ProcessingRun(ml_model)` recording at project-aware call sites; HF auth-UX beyond `HF_TOKEN`; the parallel-vs-shared registry reassessment.

---

## 2026-05-27 — E12 network spike: ureq3 + checksum + HF passthrough (findings + decisions)

E12 introduces the **engine's first network capability** (HTTP weight download + `hf://` passthrough). That bumps architectural principle #10 (2026-05-16: *"local-first; explicit opt-in for any network feature"*), so — like the `ort` spike — this de-risks the approach before committing a dependency, and **pauses for the maintainer's nod** rather than auto-implementing. Throwaway, `/tmp/net-spike`, nothing committed to the engine.

### Findings

- **`ureq` 3.3 (rustls) downloads cleanly, fully sync — no async runtime.** Fetched a public HF model (silero-vad ONNX, 2.24 MB) over HTTPS in ~0.9 s, streamed to disk, sha256-verified. `ureq::get(url).header("Authorization", …).call()?` → `body_mut().with_config().limit(N).reader()` → `io::copy`. A 404 / bad URL returns a clean `Err` (no panic).
- **No `hf-hub` needed.** `hf://<repo>` resolves to a stable URL we construct ourselves — `https://huggingface.co/{repo}/resolve/{rev}/{file}` — keeping one HTTP client and our own cache layout. (`hf-hub` 0.4 also pulls an *older* `ureq` 2, so using it alongside `ureq` 3 means two HTTP stacks; rejected for E12. Revisit only if we want its caching/auth conveniences later.)
- **Auth** is a `Authorization: Bearer $HF_TOKEN` header — trivial; public models need none.
- **Checksum** verify is trivial with `sha2` (already an engine dependency). The HF silero copy's sha256 (`a4a068cd…`) ≠ the bundled pip copy's (`1a153a22…`), confirming checksums must be pinned **per source**, not assumed identical across mirrors.
- **Cost** (the headline concern): `ureq` 3 + `rustls` + `ring` + `sha2` ≈ 48 transitive crates, ~+2 MB to the (stripped) binary — almost entirely TLS (`rustls`/`ring`). **No tokio/reqwest/hyper.** `ring` is the heaviest compile.

### Recommended approach (for confirmation)

1. **Client: `ureq` 3** (sync, rustls, no async — matches the engine's sync nature). Not `reqwest` (async + tokio + heavier).
2. **TLS: `rustls`** (no system OpenSSL dependency → portable cross-platform builds), not native-tls.
3. **HF: raw HTTP to the self-constructed resolve URL**, auth via `HF_TOKEN` Bearer header. No `hf-hub`.
4. **Feature-gate the network behind a default-OFF cargo feature** (e.g. `download`, or fold into the on-demand-`ml` path) so the base engine + wheel stay network-free — the ~2 MB + 48 crates land only for download-enabled builds (principle #10).
5. **Checksum-verify on fetch** against the manifest's `file_checksum`; cache fetched weights under `~/.local/share/sadda/models/hf/<repo>/<rev>/…` (the 2026-05-20 layout).
6. **Tests/CI: gate actual downloads** behind a network/env marker (like the ORT-gated tests); default CI stays offline.

### Decisions awaiting the maintainer (principle #10 — explicit opt-in)

Whether to add the network dependency at all, the feature-gate name/shape, the HF auth UX surface, and whether the **embedding-tier plumbing** (local, no network — inference output → B3 `continuous_vector` dense tiers) should land *first* as a no-network E12 slice while the network half is blessed. **Paused here for that call.**

### Sources / references

- `ureq`: <https://github.com/algesten/ureq> ; `hf-hub`: <https://github.com/huggingface/hf-hub>
- 2026-05-16 architectural principles (#10 local-first / opt-in network); 2026-05-20 ML-registry entry (cache layout, HF passthrough, auth deferral).

---

## 2026-05-27 — Model registry: repo scaffold + index + CI gate (E11, part 3b)

The C8 analogue: stands up the model registry as a separate-repo-shaped artifact, mirroring `refdist-registry/`. Completes E11; the network half (fetch + on-demand models + embeddings) is E12.

### What landed

- **`model-registry/`** (in-repo, designed to split out to `sadda-speech/model-registry`): `tier2/` (curated) + `tier3/` (community); `README.md` + `SCHEMA.md` (the `model.toml` field reference, kept in sync with `engine::models`); `validate.py` (the CI gate, self-contained stdlib `tomllib` — no polars, no sadda wheel); `build_index.py` (emits `index.json` in the `ModelRegistryIndex` shape); `make_placeholders.py`; `.github/workflows/registry-ci.yml`.
- **Weights live elsewhere.** Unlike refdist (data inline), model entries are **manifest-only** — `model.toml` + `LICENSE`, no weights. The manifest declares a `url` + `sha256` and the engine fetches/verifies on demand (E12). Only the tier-1 bundled set ships weights inline (`models-bundled/silero-vad/`). `validate.py` enforces this: a local `file` must exist, *or* a `url` + `file_checksum` must be declared.
- **Validation is shallower than refdist's, by design** (per the 2026-05-20 entry): manifest fields, known `model.kind` + `format` (tier 2 prefers ONNX), license + LICENSE file, weights-resolvable, `sha256:<64hex>` checksum shape, known `output.tier_kind` / `compute.gpu` — but it can't validate model *accuracy*; curated trust leans on editorial review.
- **Synthetic placeholder set** (`make_placeholders.py`): `tier2/placeholder-embeddings` (Apache-2.0, `continuous_vector` output) + `tier3/placeholder-asr` (MIT), both url-based with `example.invalid` placeholders — exercising the pipeline until real curated models (wav2vec2-base, whisper-tiny/base) land with E12.
- **Engine**: `ModelRegistryIndex` / `ModelRegistryEntry` + `parse_model_index()` (parallel to refdist's), so the engine can read a hosted registry's index.
- **Python**: `sadda.ml.install_model(src_dir, root=…)` + `sadda.ml.get_model(id, version, root=…)` (PROVISIONAL) — install a model dir into the store and resolve it back, mirroring `sadda.refdist.install`/`get`.

### Validation

Engine: 11 `models` units (added `parses_registry_index`). Python: `test_models_registry.py` runs `validate.py` (pass on the placeholder set **and** the file-based bundled set; rejects url-without-checksum, missing LICENSE, unknown kind), `build_index.py` (entries + tier/kind/format/license carried through — the engine-schema contract checked from Python), and the bundled-model install→`get_model` round-trip. Full CI gate sequence green locally on 1.95 (fmt, clippy `--workspace --all-targets -D warnings`, build, `cargo test --workspace`, stub unchanged, pytest 11 pass + 3 ORT-skip).

### E11 complete — what's deferred to E12

`hf://` passthrough + HTTP weight download (+ checksum verify) + the wav2vec2/Whisper curated starter set + inference results as embedding tiers (`continuous_vector`). Also: ProcessingRun(ml_model) recording at project-aware call sites; the architecture-consolidation reassessment (parallel `models` vs shared core with refdist) once both registries are concrete; ORT-sidecar packaging for the 0.3.2 binaries.

---

## 2026-05-27 — Model registry: consume side + load_model/Model (E11, part 3a)

Implements the design entry's part 3a (the C7 analogue): the model-registry **consumption** side, offline-first, no network.

### What landed

- **`engine::models`** (new, parallel to `refdist`, behind `ml`): `ModelManifest` (serde over the `model.toml` blocks `[model]`/`[input]`/`[output]`/`[compute]`/`[citation]`); `ModelStore` (user cache at `~/.local/share/sadda/models/`, nested `<id>/<version>/`, `install_from_dir`/`get`/`get_latest`/`resolve`); `Model { manifest, dir }` with `file_path`, `id`/`version`/`kind`/`weights_checksum`, and `.vad(audio)` (delegates to the part-1 `ml::vad`, enforcing the declared kind).
- **`load_model(id)`** resolver: `sadda/<name>[@version]` (user store → bundled-set fallback), `local://<path>` (a model dir, or a bare file with a synthesized minimal manifest), `hf://…` → a clear "arrives in E12" error.
- **Bundled VAD re-homed** under `models-bundled/silero-vad/model.toml` (kind=vad, sha256 checksum, MIT). `vad_bundled()` moved into `models` and now does `load_model`-of-the-bundled-entry; the part-1 `bundled_vad_path` is gone. `ml` keeps just the inference primitives (`vad(audio, path)`, `speech_segments`). Behaviour identical, re-verified.
- **Python**: `sadda.ml.load_model(id) -> Model` (PROVISIONAL) with `.vad(audio)` + `.id`/`.version`/`.kind`/`.weights_checksum`/`.title`/`.license`; the part-2a `sadda.ml.vad(audio)` convenience unchanged.
- **Provenance**: `Model` exposes `id`/`version`/`weights_checksum` for a project-aware caller to record `ProcessingRun{kind=ml_model, …}`; the recording wiring (where bundle context exists) rides a later step.

### Validation

Engine: 10 `models` units (manifest parse, `Model` accessors, store install/get round-trip, `load_model` for local-dir / bare-file / `hf://`-error / curated-bundled-fallback, kind-check, + an ORT-gated `vad_bundled` e2e that skips cleanly without ORT). Python: 7 `test_ml` (load_model resolves bundled VAD + metadata, `hf://` deferred, provisional tier, `Model.vad` matches the free `vad`), skipping the inference ones without ORT. Full CI gate sequence green locally on 1.95 (fmt, clippy `--workspace --all-targets -D warnings`, build, `cargo test --workspace`, stub unchanged, pytest 4 pass + 3 skip without ORT).

### Next

E11 part 3b — the `model-registry/` repo scaffold (tiers, `validate.py`, `build_index.py`, index), the C8 analogue. Then E12 (network: `hf://` + HTTP download + wav2vec2/Whisper → embedding tiers). Architecture-consolidation reassessment (parallel vs shared with refdist) remains parked until both registries are concrete.

---

## 2026-05-27 — Roadmap intake: speech-AI-engineer feature directions

Ideation session on what would make Sadda *attractive to speech-AI/ML engineers* — beyond the basics, the "take notice" tier, weighted (at the user's direction) toward features that help engineers make **principled and ethical decisions rather than running a parameter grid and picking the best number.** Logged-not-designed (like the AI-agent-surface + walkthrough-demos intake); ~Phase 4+/v1.x; each needs its own design session. Six directions, all endorsed by the user (the last two are the user's additions).

**Unifying thesis.** Not "another training framework" — Sadda is **the measurement / evaluation / interpretation layer for speech ML, phonetically literate** (it knows what F1/F2/VOT/jitter/SNR *mean*, has a corpus model + reference distributions + ML-native time-aligned signals + provenance). The pitch: *the acoustic conscience of a speech-ML pipeline* — making data and model behaviour legible at the phonetic level so decisions are defensible. How speech-AI engineers work today (collect → filter → fine-tune an HF model → eval WER/EER/MOS → eyeball errors → iterate; tools = `datasets`/`transformers`, `torchaudio`/`librosa`, `jiwer`, W&B, notebooks) leaves chronic gaps: data is acoustically opaque, eval collapses to one number, error analysis is manual, bias/representation auditing is rare, reproducibility/documentation are afterthoughts.

### 1. Acoustic dataset intelligence — "know your data"
Corpus acoustic profiling (f0 / formant-space / SNR / bandwidth / clipping / speaking-rate / duration distributions); **coverage & representation auditing against reference distributions** (which phones/contexts are thin; which speaker demographics under-represented vs. a target population — the ethical core: surface gaps *before* training); train/test **leakage & near-duplicate detection** via speaker+acoustic embeddings; label/alignment QA; **auto-datasheets** (prior art: *Datasheets for Datasets*, Gebru et al.) populated from measured facts.

### 2. Corpus comparison *(user addition)*
A/B two or more corpora or subsets on the same acoustic/representation axes — train vs. test, domain A vs. B, pre/post-filtering, candidate datasets — with statistical comparison (effect sizes, not eyeballing). The bridge between #1 and #3. Use: **domain-shift detection**, distribution drift, and **principled dataset selection/curation** (choose for coverage/representation, not convenience; detect when the eval set doesn't match deployment). Reuses the #1 profiling machinery + #3 statistics.

### 3. Principled evaluation — the antidote to grid-and-pick
**Disaggregated slice-based evaluation** (WER/EER/MOS by acoustic + demographic buckets: SNR band, rate, accent, f0 range, sex/age) with confidence intervals; **statistical honesty** — significance tests, effect sizes, multiple-comparison correction, **power analysis** ("is the test set even big enough for that WER gap?"); calibration/uncertainty checks; **auto Model Cards** (Mitchell et al.) from the disaggregated results + provenance. Structurally discourages cherry-picking.

### 4. Model legibility — ML-native interpretability, phonetically grounded
Model internals as first-class **time-aligned signals** (CTC posteriors, attention/alignment, layer-wise SSL embeddings) viewable alongside formants/f0; **acoustic error analysis** (correlate ASR/TTS errors with measurable acoustic factors → failure *modes*); **SSL-layer probing against acoustic ground truth** ("which wav2vec2/HuBERT layer encodes F1 / voicing / speaker?"; prior art: SUPERB, Pasad et al. layer-wise analyses) — research-grade, and only doable well by a tool that *has* the acoustic ground truth; **controlled perturbation/robustness** (principled noise/reverb/pitch/formant manipulations via Sadda DSP → counterfactuals, not random augmentation).

### 5. Automated labeling: ToBI / IPA / phonetic-feature spans *(user addition)*
Auto-annotation that attacks the annotation bottleneck and compounds with Sadda's prosody/formant strength + the ML registry: **ToBI** (pitch accents / boundary tones / break indices; prior art: Rosenberg's AuToBI — genuinely hard, imperfect), **IPA / phone segmentation** (more mature: MFA, Charsiu, Allosaurus), **phonetic-feature spans** (distinctive features over time, derived from phone labels + a feature system). The principled posture is **human-in-the-loop**: the model *proposes* labels as tiers carrying **confidence + provenance** (which model/version), the expert *disposes* — never silently trusting model output. Bridges the annotation model (B1/B2) ↔ ML registry (E11/E12) ↔ measure tracks.

### 6. Reproducibility & accountability — Sadda already has the bones
Provenance + citations on every measurement (A1), recipe record/replay (F1), **licensing/consent tracking** on training data (the refdist license discipline generalizes), and speaker **anonymization / re-identification-risk** analysis (k-anonymity thinking from refdist). Auditable, legally-aware pipelines.

### Flagships + cross-connections

The three most differentiating *and* most on-thesis (principled/ethical, and uniquely Sadda): **representation/coverage auditing vs. reference distributions (#1)**, **disaggregated + statistically-honest evaluation (#3)**, and **SSL-layer probing against acoustic ground truth (#4)**. These compose with already-logged roadmap items: the **AI-agent surface** would *drive* all of these (an agent running a principled eval + emitting a model card); **report/figure generation** *outputs* them; **reference distributions** are the *yardstick* for representation; the **model registry (E11/E12)** powers the labeling models; **provenance (A1)** makes every result auditable.

### Open questions for the eventual design sessions

Statistics engine: built-in (a focused stats layer) vs. lean on Polars→`statsmodels`/R? Where do auto-cards live (docs-embedded, exportable HTML/PDF — shares the report/figure IR)? Which audience surface drives this — the Python API / AI-agent surface primarily, with the GUI for inspection? Labeling: confidence + correction-workflow UX; which models ship curated. Representation auditing needs the *real* reference distributions (still pending license clearance) to be more than a demo.

### Status

Roadmap intake only. Immediate path unchanged: finish **E11** (model registry → E12). Revisit each with its own design session when scheduled; the flagships are the place to start.

---

## 2026-05-27 — Model registry: implementation shape for E11 (design session)

The 2026-05-20 "ML model registry" entry is the *governance* design (the model analogue of the 2026-05-18 refdist-governance entry). This session concretizes it into an **implementation shape for E11**, the way C7/C8/C9 turned refdist governance into code. The manifest format, three ID schemes (`sadda/…`, `hf://…`, `local://…`), ONNX-canonical curated tier, storage layout (`~/.local/share/sadda/models/`), and `ProcessingRun{kind=ml_model, weights_checksum}` provenance are all settled there and not re-litigated.

### Decisions this session

1. **Architecture: parallel for now, consolidation DEFERRED for reassessment.** Build a standalone `engine::models` module mirroring refdist's shape rather than generalizing C7/C8 into a shared generic core. Rationale: it's the *reversible* choice — extracting shared utilities (cache-dir, install-from-dir copy, index parsing) later is easy; un-merging a premature abstraction over two schemas that may diverge (models carry weights/compute/output-tier; refdist carries population/measure) is not. **Reassessment checkpoint**: after E12, when both registries are concrete and the real duplication/divergence is observable rather than guessed — decide then whether to extract a shared core or shared utilities. (Logged at the user's request to revisit.)
2. **Offline-first split (E11) / network (E12)** — mirrors C7 (consume) → C8 (registry repo + bundled set) → C9, with no network in E11.
3. **`load_model(id) -> Model` object** with task methods (`.vad()` now; `.transcribe()` / `.embeddings()` in E12), plus `sadda.ml.vad(audio)` kept as a convenience over the bundled model. Resolve-once-then-call suits repeated use and agent use.

### Concrete shape

- **`engine::models`** (new, parallel to `refdist`):
  - `ModelManifest` — serde over the 2026-05-20 `model.toml` (`id` / `version` / `title` / `upstream_source` / `license`, `[model]` kind+format+file/url+checksum, `[input]`, `[output].tier_kind`, `[compute]`, `[citation]`).
  - `ModelStore` rooted at `~/.local/share/sadda/models/` — `list` / `get` / `install_from_dir` (mirrors `RefdistStore`).
  - `load_model(id) -> Model` resolver: `sadda/…` (curated cache) and `local:///…` land in E11; `hf://…` returns a clear "arrives in E12" error (not silent).
  - `Model { manifest, dir }` — `.vad(audio)` (delegates to the part-1 `ml::vad` with this model's ONNX path), `.kind` / `.output_tier` / compute info, and `.id` / `.version` / `.weights_checksum` for provenance. `.transcribe()` / `.embeddings()` arrive in E12.
- **Bundled VAD re-homed under a manifest** — `models-bundled/silero-vad/model.toml` alongside the existing `silero_vad.onnx` + `LICENSE`. `vad_bundled()` becomes `load_model` of the bundled entry; the bundled set is the registry's tier-1 proof, exactly paralleling `refdist-bundled/`.
- **Provenance hook** — `Model` exposes the fields a project-aware caller needs to record `ProcessingRun{kind=ml_model, processor_id=id, processor_version=version, weights_checksum}` (the table + fields exist since B1/A1). The *recording* is wired where bundle context exists (the GUI VAD path / a project-aware entry point), not in the bare `vad(audio)` function.
- **Python**: `sadda.ml.load_model(id) -> Model` (PROVISIONAL); `Model.vad(audio)`; existing `sadda.ml.vad` convenience unchanged.

### Sub-slices (mirroring the C7→C8 staging)

- **E11 part 3a** — `engine::models` + `load_model`/`Model` + bundled-VAD-via-manifest + Python surface (the C7 analogue: format + store + resolve + consume).
- **E11 part 3b** — `model-registry/` repo scaffold (tiers, README/SCHEMA, `validate.py`, `build_index.py`, index JSON), parallel to `refdist-registry/` (the C8 analogue).

### Deferred (per 2026-05-20, unchanged)

`hf://` passthrough + HTTP weight download + wav2vec2/Whisper starter set + embedding tiers (E12); HF auth UX; curated-tier smoke-test CI; cache eviction (manual GC at v1); the architecture-consolidation reassessment above.

### Sources / references

- 2026-05-20 ML-model-registry entry (the governance this concretizes).
- 2026-05-25 C7/C8/C9 entries (the refdist mechanism this parallels, and the staging it mirrors).

---

## 2026-05-26 — ML inference: GUI VAD lane (E11, part 2b)

Completes the three-surface coverage for the VAD model (engine → Python → GUI). The app's `sadda-engine` dependency now enables `ml`, and VAD joins the D10 measure-track lanes.

### What landed

- **VAD measure-track lane** — a stacked lane (reusing the D10 `measure_lane` scaffolding) showing the per-window speech probability (0–1) as a contour, a dashed threshold line, and green shading over windows above the threshold. View → Measure Tracks → "VAD (speech)" toggles it; `vad_threshold` persists in `MeasureTrackConfig`.
- **Computed + cached** alongside the other tracks: `compute_measure_tracks` now builds the owned `Audio` once and reuses it for both pitch and `vad_bundled`. The result lands in `MeasureTrackCache { vad, vad_error }`.
- **Graceful degradation** — VAD needs ONNX Runtime at runtime, so it can fail where f0/formants/intensity can't. A failure (e.g. ORT absent) is caught into `vad_error` and rendered as a hint in the lane ("VAD unavailable — …"); the app never crashes, and starting it never requires ORT.

### Validation

Full CI gate sequence again green locally on 1.95.0 (fmt, clippy `--workspace --all-targets -D warnings`, build, `cargo test --workspace`, app tests 48). The lane rendering itself isn't unit-tested (consistent with the other panes); the config logic is, and the VAD data path is covered by the engine + Python tests. **Visual QA in a running window is still pending** (same standing caveat as D10).

### E11 status + what's left

E11's first model is now fully surfaced (engine + Python + GUI). Remaining in cluster E: the **on-demand model registry** (parallel to refdist + `hf://` + `weights_checksum`) — which, like refdist governance, deserves its own design pass — and **E12** (wav2vec2/Whisper → embedding tiers). Stopping here pending that design input.

---

## 2026-05-26 — ML inference: Python `sadda.ml` surface + ml-on-by-default (E11, part 2a)

Builds on part 1's engine `ml` foundation. Decision (with the user): **`ml` on by default in the shipped wheel + app** — under `load-dynamic` it costs ~150 KB of bindings, ONNX Runtime stays an optional runtime sidecar, and the surface errors cleanly (never crashes) when ORT is absent.

### What landed

- **`crates/python` enables `sadda-engine`'s `ml` feature** → every wheel build exposes `sadda.ml`. (Feature-unified across the workspace, so the engine's ml tests now also run under CI's `cargo test --workspace` — they skip cleanly without ORT, see below.)
- **`sadda.ml.vad(audio)`** → `(times, speech_probs)` NumPy arrays (one Silero-VAD window per element; audio mono-mixed + resampled to 16 kHz internally). **`sadda.ml.speech_segments(audio, threshold=…)`** → merged `(start, end)` tuples. Both use the bundled model unless given a `model_path`. PROVISIONAL. This is also the **first concrete piece of the roadmap's AI-agent surface** — structured, headless, agent-legible.
- `PyAudio` made `pub(crate)` so the new `crates/python/src/ml.rs` module can take it.

### CI surface (the reason this was verified end-to-end before commit)

Turning `ml` on workspace-wide changes what CI compiles and runs. Ran the **exact CI gate sequence** locally on a 1.95.0 toolchain: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo build --workspace --all-targets`, `cargo test --workspace`, stub regen + diff, and pytest. Two issues caught and fixed pre-push: a `clippy::type_complexity` on the `vad` np-tuple return (matched the existing `voiced_pitch` `#[allow]`), and fmt drift in the new file. The engine's ml e2e test runs under `cargo test --workspace` now and **skips cleanly without ORT** (the part-1 `libloading` probe); `test_ml.py` likewise skips its inference tests without ORT (1 pass + 2 skip) and runs them for real with `ORT_DYLIB_PATH` set (3 pass). The `.pyi` stub is unchanged (the ml submodule, like refdist, carries no `gen_stub` derives).

### Deferred (still ahead in E11 + E12)

GUI VAD tier (part 2b — surface VAD in the app), the on-demand model registry (parallel to refdist + `hf://` + `weights_checksum`), and E12 (wav2vec2/Whisper → embedding tiers). ORT-sidecar packaging for the 0.3.2 binaries remains a release-engineering task.

---

## 2026-05-26 — ML inference: ort-bundling spike + first bundled model (E11, part 1)

Opens cluster E (ML inference, 0.3.2). The plan mandated an **`ort`-bundling spike before E11**, because 0.3.2's headline risks are binary size and cross-platform ONNX Runtime linking. Ran the spike, settled the integration strategy, then landed the engine half of E11: a feature-gated ML path and the first bundled model (Silero VAD), end to end.

### The spike (throwaway, `/tmp`)

Goal: confirm `ort` links + bundles cleanly and measure the size hit, before committing. Findings:

- **`ort` 2.0.0-rc.10 + `load-dynamic` compiles in our toolchain** (Rust 1.93.1) in ~8 s, pulling only `libloading` — **no ONNX Runtime downloaded or linked at build time**.
- **Binary-size answer (the headline risk): `load-dynamic` keeps the ~23 MB ORT out of every artifact.** The spike binary gained ~150 KB (just the bindings); the 23 MB `libonnxruntime.so` is an external sidecar loaded at runtime. So enabling ML never bloats the base engine / wheel / app.
- **End-to-end inference works**: Silero VAD loaded and ran — silence → speech-prob 0.0006, noise → 0.0023.
- **Version tolerance**: `ort` rc.10 requests ORT **API version 22 (= ONNX Runtime 1.22)**; a **1.26** runtime served it fine (ORT's C API is backward-compatible). The runtime is found via `ORT_DYLIB_PATH`.

Decisions taken (with the user, before building): **feature-gated + `load-dynamic`**, **Silero VAD** (MIT) as the first model, **spike first**.

### What landed (engine)

- **`ml` cargo feature, default OFF** (`crates/engine`): `ort` + `libloading` are optional, enabled only by `--features ml`. The default build — and therefore CI, the Python wheel, and the app — is completely unchanged (no ORT, no size hit). Confirmed: default build + 12 test groups green; `--features ml` builds + tests green; clippy clean both ways.
- **`engine::ml`** (`#[cfg(feature = "ml")]`): `vad(audio, model_path)` mono-mixes + resamples to 16 kHz, runs Silero VAD window-by-window (512 samples) threading the recurrent `state`/`stateN`, and returns a `VadFrame { time_seconds, speech_prob }` per window. `vad_bundled(audio)` uses the located bundled model; `speech_segments(frames, threshold)` merges above-threshold windows into `SpeechSegment`s (pure, ORT-free, unit-tested).
- **Bundled Silero VAD** (`models-bundled/silero-vad/`): `silero_vad.onnx` (2.3 MB, MIT, silero-vad 6.2.1) + its `LICENSE` + a `README` documenting the I/O contract. Located via `bundled_vad_path()` (env `SADDA_MODELS_BUNDLED` → exe-dir → dev workspace path, mirroring the refdist bundled-set locator).
- **No-panic discipline**: `ort` *panics* in its lazy loader when the runtime is absent — unacceptable for a library. Added `ensure_ort_available()`, which probes the dylib (`ORT_DYLIB_PATH` or the platform default name) with `libloading` *before* touching `ort`, converting "no ONNX Runtime" into a clean `EngineError::Ml` (new variant). So `vad()` on a machine without ORT returns an error, never aborts.
- **Shared resampler**: lifted `resample_to_hz` out of `clinical` into `engine::dsp` (`pub(crate)`) so GNE (10 kHz) and VAD (16 kHz) share one FFT-domain resampler.

### Validation

`ml` tests run **both** ways: without ORT (the e2e `vad` test skips cleanly — proving the no-panic probe) and with `ORT_DYLIB_PATH` set (real inference: 1 s of silence reads mean speech-prob < 0.3). Pure `speech_segments` + the bundled-model locator are tested unconditionally. Default suites untouched: engine 12 groups, app 48, python 137 — all green.

### Deferred (next E11 sub-steps + E12)

- **Python `sadda.ml` + GUI VAD tier** — the three-surface completion. Engine-first is justified by ML's optionality (the wheel/app stay ML-free by default); these ride the next E11 commits, gated behind the `ml` feature in the wheel build.
- **Model registry** — the parallel-to-refdist registry with `hf://` passthrough + `weights_checksum` provenance (the `ProcessingRunKind::MlModel` plumbing already exists). Next E11 slice.
- **On-demand models + embedding tiers (E12)** — wav2vec2 / Whisper downloaded on demand, written as `continuous_vector` dense tiers.
- **Packaging the ORT sidecar** — how the app ships `libonnxruntime` per-platform and sets `ORT_DYLIB_PATH`; a release-engineering task for when 0.3.2 binaries are cut.

### Sources / references

- 2026-05-25 Phase 3 slicing entry (E11/E12 scope; the `ort`-spike-before-E11 prerequisite).
- 2026-05-20 ML-model-registry entry (ONNX canonical; HF passthrough; embedding tiers; `ProcessingRun kind = ml_model`).
- `ort` (ONNX Runtime for Rust): <https://github.com/pykeio/ort>
- Silero VAD (MIT): <https://github.com/snakers4/silero-vad>

---

## 2026-05-26 — GUI measure tracks + refdist overlays (D10)

The first slice of cluster D and the payoff for clusters B and C: the GUI now *renders* analysis against the timeline and against reference distributions. Three coupled capabilities, on the user's call to ship all three refdist display forms (not just timeline bands).

### Measure-track lanes (D10a)

The engine has emitted `pitch()` / `formants()` / `intensity()` per-frame series since Phase 0, but the GUI never drew them. D10a adds **stacked lanes** below the spectrogram — f0 (dot contour), formants (one colour per slot F1..Fn), intensity (dB line) — each a frameless `egui_plot` lane sharing the `SIGNAL_LEFT_GUTTER` and the `view_start..=view_end` window, so the playback cursor draws one straight line through waveform → spectrogram → lanes → tiers. Computed in-app from the bundle samples and cached in a `MeasureTrackCache` keyed on bundle + `MeasureTrackConfig` (mirrors `SpectrogramCache` / `rebuild_*_if_stale`); a hidden lane costs nothing (the recompute skips it). Visibility + analysis params persist in `PersistedState`. Lanes chosen over Praat-style overlay-on-spectrogram per the user (separate y-axes, clean per-measure scaling).

### Refdist `summary` / `histogram` / `points2d` (D10b, three-surface)

So the band/scatter/hist numbers aren't app-internal, the overlay geometry is computed in the **engine** and exposed to **Python** identically:

- `Summary { n, mean, sd, min, p5, p25, median, p75, p95, max }` — empirical for an observed distribution (`from_samples`, NumPy type-7 percentiles); a **normal model** of the published mean/SD for a `summary_normative_range` (`from_mean_sd`, z-quantile band), so both render through the same fields but only one is measured.
- `Histogram { edges, counts }` (`from_samples`), `RefDist::{column_f64, summary, histogram, points2d}`, all taking an AND-combined `(column, value)` subgroup `filter` (sex / phone). Python: `rd.summary("F1", filter={"phone":"iy"})`, `.histogram`, `.points2d`, returning `Summary` / `Histogram` pyclasses.

**Latent bug found + fixed:** the engine's `parquet` feature set was `["arrow","snap"]`, but polars' `write_parquet` (every refdist data file, and the C9 scaffolder's output) defaults to **zstd** with **LargeUtf8** strings — so the engine could not read its own refdist data at all. Added the `zstd` feature and a `LargeStringArray` path. Verified against the real bundled zstd file (/iy/ F1 mean ≈ 303, matching the synthetic 300 target).

### Timeline band overlays (D10c)

Horizontal bands on the f0 / intensity lanes, drawn behind the contour, with **`MeasureKind`-distinct encodings** so the 2026-05-18 governance rule holds — the GUI never conflates "what people do" with "what to aim for":

- observed → cool neutral percentile band; normative → green clinical band; **target zone → amber goal region with a dashed border + a "TARGET" tag**.
- Each draws an outer p5–p95 fill, inner p25–p75 fill, and a centre line. Picked per lane from a View-menu submenu listing store distributions whose parameter matches the lane, narrowed by subgroup (per-sex entries). Resolved + cached on selection change (a store + parquet read), not per frame.
- **First-run seeding** (the C8-deferred app wiring): View → "Install bundled reference data" copies `refdist-bundled/` into the user store via `install_from_dir` (idempotent), located by `SADDA_REFDIST_BUNDLED` → exe-dir → dev workspace path.

### Vowel-space scatter + 1-D histogram (D10d, D10e)

A right-side **Reference panel** with one distribution picker feeding two views:

- **Vowel space** — `points2d(params[0], params[1])` reference cloud in phonetic orientation (F2 on x and F1 on y, both axes inverted via egui_plot `invert_x/y`), plus the **measured vowel** (F1/F2 from the formant track at the cursor) as a red diamond. Phone filter combo.
- **Histogram** — engine `Histogram` as a bar chart with dashed p5/p95 + solid median markers from the `Summary`, and a red "you" line at the measured value (median voiced f0 for the `f0` param; the cursor formant for F1/F2) with a "(interquartile)" / "(above p95)" position readout. Summary-only distributions say so and point at the band overlay (no raw samples to bin).

### Validation

Engine: 10 new `refdist` units (Summary/Histogram math + normal-band, `column_f64` filter + integer widening, observed vs normative `summary`, `points2d` alignment, histogram rejects summary-only) — 20 refdist units, 129 engine total. App: `MeasureTrackConfig::any_visible` + `nearest_frame_index` units — 48 total. Python: 6 new (`column`/`summary`/`histogram`/`points2d` incl. normative band + the zstd/LargeUtf8 real-data path) — 137 total. clippy clean. The plot rendering itself isn't unit-tested (consistent with the existing waveform/spectrogram panes); the pure logic under it is, and the data path is verified against the real bundled file.

### Deferred

- **Visual QA in a running window** — the lanes/overlays compile + their data is verified, but a human hasn't eyeballed the rendered app this session.
- **Phonetic-convention niceties** — colour-by-phone in the vowel cloud (needs a string-column-aware 2-D reader); a secondary-axis intensity overlay-on-spectrogram option.
- **CHANGELOG** — left for the 0.3.x release, matching the project's release-time-only CHANGELOG cadence (Phase-3 A/B/C aren't in it either).
- **Real reference data** — overlays still ride the synthetic placeholder set until the license-cleared corpora (Hillenbrand/Peterson-Barney/clinical norms) land.

### Sources / references

- 2026-05-16 cross-cutting patterns (A: real-time/measure display undersupported; C: reference distributions are everyone's hidden problem).
- 2026-05-18 reference-distribution governance (the `MeasureKind` distinction the encodings enforce).
- 2026-05-25 C7/C8/C9 entries (the format, store, validator, and the C8-deferred first-run seeding this lands).

---

## 2026-05-25 — In-app refdist publishing: scaffold from an analysis result (C9)

Closes the reference-distribution loop (consume → publish). C9 turns an analysis result into a publishable distribution directory; the actual fork-and-PR submission uses the maintainer's own GitHub credentials, so that step is documented rather than automated (and waits on a live registry — see deferred).

### Scaffold (the buildable, testable core)

`engine::refdist::scaffold(dir, &RefdistManifest, provenance) -> RefDist` writes `refdist.toml` (serialized from the manifest), `provenance.md`, and a `LICENSE` stub keyed off the manifest's SPDX id. The data file is the caller's to write. The written manifest **round-trips through `parse_manifest`** — so a scaffolded distribution is immediately resolvable and passes the C8 validator (once the maintainer swaps the LICENSE stub for the full text and fills in real provenance).

Making serialization work meant adding `skip_serializing_if` to the manifest's `Option`/`Vec`/empty-`String` fields (the `toml` serializer rejects `None`; we want absent-not-null and no noise arrays). Deserialization is unchanged (still `#[serde(default)]`).

### Python ergonomics

`sadda.refdist.scaffold(dest, data: polars.DataFrame, *, id, version, kind, parameters=…, units=…, language=…, sex=…, license=…, shareability=…, min_n_per_subgroup=…, authors=…, year=…, provenance=…)` (provisional): writes `data.parquet` from the DataFrame, infers `schema.columns` from it and `n_speakers` from a `speaker_id` column, then calls the engine scaffolder. One call from a result table to a submission-ready directory.

### Validation

Engine: a `scaffold` round-trip unit test (write → re-parse → fields match, `None`s skipped not nulled). Python: `test_scaffold_produces_a_validatable_distribution` scaffolds from a DataFrame, runs the **C8 `validate.py`** over it (exit 0 — the producer and the gate agree), then installs it into a store and resolves it back with `.data()`. 9 engine refdist units + the registry Python tests all green.

### The submission flow (documented, deferred)

Per the governance entry, "publication is a `git push` to a fork-and-PR flow; auth is the user's GitHub credentials, not ours." The `scaffold` docstring spells out the manual flow (copy under `tier3/<id>/`, branch, `gh pr create`). Automating it is deferred because (a) there is **no hosted registry repo to PR against yet** (same blocker as C8's HTTP fetch), and (b) the natural home for a guided "Publish…" flow is the **app GUI (cluster D)**, on top of this scaffolder. No git/GitHub action is taken automatically.

### Sources / references

- 2026-05-18 governance entry ("Tier 3 publishing is in-app … the manifest is scaffolded automatically").
- 2026-05-25 C7 + C8 entries (the format, store, and validator this produces against).

---

## 2026-05-25 — Reference-distribution registry: scaffold + CI + index + bundled set (C8)

Builds on C7's format + local store. C8 stands up the **registry** (the public, separate-repo artifact) and the bundled starter set, with the engine able to consume a registry index. Real, license-cleared data is still being sourced, so everything here is wired against a **synthetic placeholder set** that exercises the whole pipeline end to end.

### The registry (`refdist-registry/`)

Scaffolded in-repo, designed to split out to `sadda-speech/refdist-registry`:

- **`tier2/` (curated) + `tier3/` (community)** directory structure; `README.md` (tiers, submission-by-PR, license policy) + `SCHEMA.md` (the `refdist.toml` field reference, kept in sync with `engine::refdist`).
- **`validate.py`** — the CI gate, self-contained (stdlib `tomllib` + `polars`, no sadda wheel, so it travels with the repo): required fields, known `measure.kind`, SPDX license + non-empty `LICENSE` file (tier 2 disallows NC/ND), `min_n_per_subgroup` present, **data-file column conformance**, and a k-anonymity proxy (distinct `speaker_id` ≥ `min_n` for `raw_samples` — tier 2 errors, tier 3 warns).
- **`build_index.py`** — walks the tiers and emits `index.json` in exactly the shape `engine::refdist::RegistryIndex` deserializes (the GitHub-Pages artifact).
- **`make_placeholders.py`** — regenerates the synthetic placeholder distributions deterministically.
- **`.github/workflows/registry-ci.yml`** — runs validate + build-index, for when the registry is its own repo.

### Placeholder starter set (synthetic — clearly marked)

`make_placeholders.py` writes three distributions, every title + `provenance.md` shouting PLACEHOLDER, data from fixed seeds (no real corpus): `placeholder-amE-vowels` (tier 1, bundled, F1/F2 observed, CC0), `placeholder-f0-norms` (tier 2, f0 summary-normative, CC-BY-4.0), `placeholder-vot-norms` (tier 3, VOT observed, CC0). Tier 1 lives in **`refdist-bundled/`** (ships with the app); tier 2/3 under the registry.

### Engine

- `RegistryIndex` / `RegistryEntry` (serde) + `parse_index()` — the engine reads a registry's published `index.json` to discover what's available.
- `RefdistStore::install_from_dir()` — copies a distribution into the store under `<id>__<version>/`. This is how the bundled set seeds the user cache and where a fetched-and-unpacked tarball will land. Added a top-level `license` field to `RefdistManifest`.
- Python: `sadda.refdist.install(src_dir, root=…)` (provisional).

### Validation surfaces

Engine: 8 `refdist` unit tests (now incl. `parse_index`, `install_from_dir`). Python: `test_refdist_registry.py` exercises `validate.py` (pass on the placeholder set; **rejects** broken columns + missing LICENSE), `build_index.py` (entries + tier/kind/license carried through — the engine-schema contract checked from the Python side), and `install` round-trip (bundled → store → `.data()`). The main-repo CI runs these via pytest, so the registry gates run here too.

### Deliberately deferred

- **Real datasets** — Hillenbrand 1995 / Peterson-Barney 1952 / a clinical normative set, pending license clearance (user is following up). The placeholder set holds the contract until then.
- **HTTP fetch** — there is no live registry to fetch from yet (the repo isn't pushed/hosted). `parse_index` + `install_from_dir` are the offline halves; the thin network GET lands when the registry goes live. No HTTP dependency added.
- **Creating/pushing the public `refdist-registry` repo** — an outward action for the maintainer; the scaffold + its CI workflow are ready to split out.
- **First-run seeding of the bundled set** by the app — an app-side (cluster D) wiring on top of `install_from_dir`.

### Sources / references

- 2026-05-18 "Reference distribution governance" entry (tiers, format, CI gates, license policy).
- 2026-05-25 C7 entry (the format + `RefdistStore` this builds on).

---

## 2026-05-25 — Reference distributions: format + resolver + query + pinning (C7)

First slice of cluster C and the start of the **0.3.1** sub-release. C7 is the **consumption** side of the reference-distribution system (the 2026-05-18 governance entry's eight deliverables): parse a `refdist.toml` manifest, resolve distributions from a user-level cache, query them by population/measure facets, and let a project pin the versions it used. Deliberately independent of any *hosted* registry — HTTP fetch + the public registry repo + CI + bundled starter set are C8.

### What landed

- **`engine::refdist`** — `RefdistManifest` (+ `Citation` / `Population` / `Measure` / `Privacy` / `Schema` serde structs, all `#[serde(default)]` so partial manifests parse) mirroring the governance entry's `refdist.toml` schema. `MeasureKind` enum (`observed_distribution` | `summary_normative_range` | `target_zone`) keeps "what people sound like" distinct from "what to aim for". `parse_manifest` / `load_manifest`.
- **`RefdistStore`** — rooted at an explicit dir or the per-OS default via the `directories` crate (`~/.local/share/sadda/refdist/` on Linux). `list()` scans subdirectories, skipping any without a valid manifest; `query(&QuerySpec)` does faceted, case-insensitive matching (parameter / language / variety / sex / age_band / phone / kind); `get(id, version)`. `RefDist { manifest, dir }` with `data_path()`.
- **Project pinning** — `Project::pin_refdist` / `refdist_pins` / `remove_refdist_pin` record `[refdist]` `id = version` entries in the existing `project.toml` (parsed/rewritten via the new `toml` dep, preserving the other keys), so the choice travels with the project and reopens reproducibly.

### Three surfaces

`sadda.refdist.{query, get, list_all, store_root}` (all `@provisional`) + a `RefDist` class with manifest getters and a **Polars `.data()`** helper (reads `data.parquet`, mirroring B3's dense-tier path pattern — the engine exposes only the path). `Project.{pin_refdist, refdist_pins, remove_refdist_pin}`. New deps: `toml`, `directories`. Tests: 5 engine unit (manifest parse incl. defaults + invalid, store list/get, query facets) + 1 corpus integration (pin round-trip through `project.toml`) + 5 Python (list/get, facets, `.data()` Polars load, provisional tier, project pins).

### Deferred to C8 / D10

Remote fetch + the public `refdist-registry` repo + CI validation (schema / license / `min_n_per_subgroup`) + the GitHub-Pages index + the bundled tier-1 starter set (Hillenbrand 1995, Peterson-Barney 1952) all ride with **C8**; in-app publishing is **C9**; overlay rendering of `measure.kind` variants is **D10**. C7 standing up the format + local store means C8's fixtures and the GUI both have a concrete contract to build against.

### Sources / references

- 2026-05-18 "Reference distribution governance" DEVLOG entry (three-tier registry, `refdist.toml` schema, the `measure.kind` distinction).
- 2026-05-25 Phase 3 slicing entry (C7 scope; C independent of B, can proceed in parallel).

---

## 2026-05-25 — ABI assembled: Hfno-6000, HNR-D, and the composite (B6 complete)

The last three pieces of the **Acoustic Breathiness Index** (Barsties von Latoszek et al. 2017): the two remaining component measures, plus the composite. With H1–H2, GNE, and PSD already done (and CPPS, jitter, shimmer, shimmer-dB from B4/B5), all **nine** ABI components now exist.

### Found the published formula

An open-access validation paper (Kankare/Finnish, *PMC10743974*) quotes the full v01 regression verbatim:

```
ABI = (5.0447740915 − 0.172·CPPS − 0.193·Jit − 1.283·GNE − 0.396·Hfno-6000
       + 0.01·HNR-D + 0.017·H1−H2 + 1.473·Shim-dB − 0.088·Shim
       − 68.295·PSD) × 2.9257400394,  clamped to [0, 10]
```

### The two new components

- **`hfno(audio)` — Hfno-6000** (dB): the LTAS level difference between the 0–6 kHz and 6–10 kHz bands, `10·log10(P[0–6k]/P[6–10k])`. Reuses the LTAS feature. Needs ≥20 kHz sample rate. Validated: cleaner `hnr_high` **30.6 dB** > noisier `hnr_mid` **17.6 dB** (more HF noise ⇒ smaller difference). Tier **provisional** — the exact band-level convention isn't confirmed.
- **`hnr_d(audio, …)` — HNR-D** (Dejonckere–Lebacq, dB): mean harmonic-peak power vs the inter-harmonic noise-floor power over the 500–1500 Hz formant zone, on an averaged 8192-pt periodogram. **CLEAN-ROOM from the ABI papers' one-line prose, not Dejonckere & Lebacq's exact procedure** (which wasn't accessible) — so explicitly provisional, with the gap flagged in the docstring. Validated: cleaner `hnr_high` **41.3 dB** > noisier `hnr_mid` **28.5 dB**.

### The composite — `abi(...)`, PROVISIONAL, and *not* wired to audio

`engine::clinical::abi(cpps, jit%, gne, hfno, hnr_d, h1_h2, shim_db, shim%, psd_s) -> f32` evaluates the published formula directly from the nine components (mirrors the approved `avqi(...)` pattern). Tiered **provisional**, with two unresolved gaps documented:

1. **Component definitions** — HNR-D (and to a lesser degree Hfno) are reconstructed from prose, not confirmed against the authors' artifact.
2. **Unit/scale conventions** — assumed CPPS/Hfno/HNR-D/H1−H2/shimmer-dB in dB, GNE a [0,1] ratio, jitter+shimmer as percents, PSD in seconds. Building the test surfaced a concrete symptom: feeding our `hfno()` (a ~30 dB level difference) into the −0.396 coefficient drives the score hard to the 0 clamp — i.e. **the regression expects component scales we haven't pinned**. So `abi_from_audio` is **deliberately deferred** (exactly as `avqi_from_audio` was), and the composite's absolute values are not to be trusted until confirmed against the authors' artifact (colleague pursuing the Phonanium ABI plugin). The test therefore checks only the formula arithmetic + ordering on illustrative in-range vectors — not clinical correctness.

### Three surfaces

All on `sadda.clinical`: `hfno`, `hnr_d`, `abi` (+ the earlier `period_std_s` on `PerturbationReport`), all tiered `provisional`. Engine + Python tests; stub regenerated.

### Sources / references

- Barsties von Latoszek, Maryn, et al. (2017), *J. Voice* 31(4): 511.e11–511.e27 — the ABI.
- Kankare et al. (2023), *PMC10743974* — Finnish validation; quotes the full formula + component descriptions.
- Dejonckere & Lebacq — HNR-D (cited via the ABI papers; exact procedure not yet sourced).
- `clean-room-clinical-algorithms` policy.

---

## 2026-05-25 — ABI component: GNE (glottal-to-noise excitation)

Second ABI component. GNE asks whether the excitation is **pulsatile** (glottal-fold vibration drives every frequency band in lock-step) or **turbulent noise** (bands excited independently). A correlation in [0, 1]: ~1 for a clean voice, dropping toward 0 as breathiness/noise grows.

### Algorithm (clean-room from Michaelis, Gramß & Strube 1997)

`engine::clinical::gne(audio, &GneConfig) -> Result<Ratio>`:

1. Band-limit + resample to 10 kHz (FFT-domain, scipy-`resample`-style — the anti-alias low-pass folds into the bin copy).
2. **Inverse-filter** (LPC whitening, order 13) → the excitation residual `e[n]`, stripping the formant envelope so band energies reflect the *source*.
3. For each band of bandwidth **1000 Hz** stepped by **300 Hz** (Michaelis et al.'s optimal screening parameters), take the **Hilbert envelope** (FFT bandpass → analytic signal → magnitude).
4. **GNE = max correlation** over band pairs whose centers differ by ≥ ½·bandwidth (non-overlapping passbands). Because the FFT bandpass is **zero-phase**, synchronous excitation lands at lag 0, so the per-pair correlation is evaluated there (rather than searching the cross-correlation function over lags — equivalent here, fewer DOF for noise to exploit).

The 10 kHz / order-13 / 1000 Hz / 300 Hz parametrization is the canonical one (the screening literature reports 90 % accuracy at bw=1000, fshift=300).

### Validation (no Praat oracle → qualitative + analytic-extreme)

There is **no Praat GNE**, so this is validated by its defining behaviour rather than a golden number:

- **Synthetic extremes** (engine test): a periodic pulse train → GNE > 0.8 (measured ~1.0); white noise → GNE < 0.5. The measure separates the two regimes it's built to distinguish.
- **Fixture ordering**: clean `hnr_high` (25 dB) **0.997** > noisy `hnr_mid` (12 dB) **0.962** > … ; the pure pulse-trains (`clean`, `shimmer`) read 1.000. GNE saturates near 1 for moderately clean voices and only falls with substantial noise — the expected, clinically-reported behaviour.

### Stability tier

`sadda.clinical.gne`, tiered **`provisional`** (like AVQI, unlike the analytically-pinned H1–H2): the algorithm follows the published canonical parametrization, but absolute values aren't yet confirmed against an authors'/reference oracle. Promotable once the Phonanium plugin (colleague pursuing) confirms.

### Sources / references

- Michaelis, Gramß & Strube (1997), *Acta Acustica* 83 — GNE, the original.
- Godino-Llorente et al. (2010), *J. Voice* — "Effectiveness of GNE for screening": bw=1000 / fshift=300 optimal.
- Barsties von Latoszek et al. (2017) — ABI and its component set (the assembly target).

---

## 2026-05-25 — ABI component: H1–H2 (first of the breathiness measures)

The Acoustic Breathiness Index (ABI), like AVQI, is a weighted composite — but of measures we mostly don't have yet (GNE, H1–H2, PSD, Hₙₒ‑6000, HNR‑D). Rather than land ABI as a black box, we're building each component as a **first-class, independently-useful measure**, then assembling the composite once all are validated. Starting with the most tractable and broadly-useful: **H1–H2**.

### What it is

H1–H2 is the level of the first harmonic (at f0) minus the second (at 2·f0), in dB — a classic **glottal-source / open-quotient correlate**. A breathy voice (a wider open quotient, a more sinusoidal glottal flow) has relatively more energy at f0, so H1–H2 rises; a pressed voice lowers it. It is the uncorrected variant here — **no formant correction** (the Hanson / Iseli–Alwan correction is a later refinement; flagged in the docstring).

### What landed

`engine::clinical::h1_h2(audio, &H1H2Config) -> Result<Decibels>` — per Hann-windowed frame (4096, 50 % hop), real-FFT magnitude spectrum, locate the peak within ±15 % of f0 (H1) and of 2·f0 (H2), average `20·log10(A1/A2)` over voiced frames. f0 comes from the shared `median_voiced_f0` helper (refactored out of `estimate_f0`, now used by both perturbation and this). `EngineError::Unreliable` if no voiced f0 / signal shorter than one frame — no silent fallback.

### Validation

The harmonic-tone fixtures use **1/h harmonic amplitudes**, so H1/H2 = 1/(½) = 2 → H1−H2 = 20·log10(2) ≈ **6.02 dB** as an exact analytic target. Engine test asserts within ±1.5 dB of 6.02 on `hnr_high_120hz`; the Python test brackets 4.5–7.5 dB. (No Praat golden needed here — the synthesis defines the truth value. A natural-vowel Praat cross-check rides with the future real-voice corpus.)

### Three surfaces

`engine::clinical::h1_h2` + `sadda.clinical.h1_h2` (tiered **`stable_clinical`** — it's a standard, well-defined measure, unlike provisional AVQI), engine + Python tests. GUI exposure rides with the cluster-D clinical-readout pane.

### Sources / references

- Titze (2000), *Principles of Voice Production* — source spectral tilt / open quotient ↔ H1–H2.
- Hanson (1997), *JASA* 101(1) — H1–H2 (and H1–A1, H1–A3) as breathiness correlates; the formant-corrected variant.
- Barsties von Latoszek et al. (2017) — ABI and its component set (the assembly target).

---

## 2026-05-25 — AVQI (B6, part 1): clean-room v03.01 formula, PROVISIONAL pending author confirmation

The Acoustic Voice Quality Index — a single 0–10 dysphonia score from a weighted combination of CPPS, HNR, shimmer-local (%), shimmer-local-dB, LTAS slope, and LTAS tilt. All six component measures now exist (B4/B5 + LTAS), so this slice adds the composite.

### Clean-room discipline (the governing constraint)

The reference AVQI implementation is the **Phonanium Praat plugin, by Youri Maryn** (the AVQI author) — proprietary/paywalled. We implement the formula **clean-room from the peer-reviewed publications**, and will use the proprietary script (if obtained) *only as a black-box oracle to confirm output*, never as a model. (See the `clean-room-clinical-algorithms` policy.) **Validation oracles must be the authors' own artifacts** — the Phonanium script (Maryn's) or the papers' worked examples — *not* third-party reimplementations (e.g. `superassp`'s `praat_avqi`), which are just other reproductions and validate nothing.

### What landed

`engine::clinical::avqi(cpps, hnr, shimmer_local_pct, shimmer_local_db, slope, tilt) -> f32` — the published **v03.01** formula `2.571·(3.295 − 0.111·CPPS − 0.073·HNR − 0.213·SL + 2.789·SLdB − 0.032·Slope + 0.077·Tilt)`, clamped to [0, 10]. Exposed as `sadda.clinical.avqi`, tiered **`provisional`** (not `stable_clinical`) — a deliberate honesty marker: it is not yet reference-confirmed, so its absolute values may change.

### What's verified vs not

- **Verified**: the formula arithmetic, and that the published worked-example component vectors order correctly (Maryn 2015 Fig 1 normal → 1.13 here < Fig 2 dysphonic → 4.82) and stay in [0, 10].
- **Not verified (open, for the authors)**: ① the v03.01 coefficients couldn't be byte-checked against a *v03.01* worked example — the accessible examples are script v02.02 and read 2.76 / 5.92 for those same vectors (a version-scaling difference); ② AVQI's exact **slope/tilt definitions** (tilt is a dB trendline value, not the dB/kHz `Ltas::tilt`; slope's LTAS bands). Because ② is unsettled, the **audio→AVQI wiring is deliberately deferred** — `avqi()` takes the six components directly, leaving their AVQI-protocol measurement to be pinned once confirmed.

### Deferred

- **`avqi_from_audio`** — pending the slope/tilt definitions.
- **ABI** — needs ~5 measures we don't have (GNE, H1–H2, PSD, Hₙₒ‑6000, HNR‑D); its own cluster of work.

### Sources / references

- Maryn & Weenink (2015), *J. Voice* 29(1) — CPPS + AVQI (primary; worked examples)
- Barsties von Latoszek et al. — AVQI v03.01 (coefficients)
- `clean-room-clinical-algorithms` policy (this session)

---

## 2026-05-25 — Long-term average spectrum (LTAS) as a first-class feature

Promoted from "an AVQI helper" to a first-class `sadda.dsp` feature at the user's request — LTAS is a core phonetics tool (spectral slope, tilt, alpha ratio underpin voice-quality work broadly), and it's also where B6's AVQI slope/tilt parameters come from.

### What it delivers

`engine::dsp::ltas(samples, sr, bin_hz) -> Ltas` — the Welch-averaged power spectrum (overlapping Hann frames → mean power per FFT bin → aggregated into `bin_hz` bands → dB). The `Ltas` carries the per-band dB levels and three derived measures:

- **slope** — high-band/low-band energy ratio in dB (`Ltas: Get slope`, "energy" averaging).
- **tilt** — least-squares regression slope of band dB vs frequency, dB/kHz.
- **alpha ratio** — energy above vs below 1 kHz, dB.

All three are *differences/ratios* of dB levels, so — like CPP, unlike HNR — they're **invariant to the spectrum's overall normalization**; only the shape matters. The slope matched Praat within **3 dB** first try across harmonic-tone and pulse-train fixtures.

### Three surfaces

`engine::dsp::ltas` + `sadda.dsp.ltas` (returns an `Ltas` with `.levels_db` / `.slope()` / `.tilt()` / `.alpha_ratio()`), tiered `stable`. GUI: the spectrum is plottable; a dedicated LTAS pane rides with cluster D's overlays.

### Feeds B6

AVQI's "slope" and "tilt" parameters are exactly `Ltas::slope` and `Ltas::tilt`, so B6 (AVQI/ABI) consumes this directly.

### Layout

- `crates/engine/src/dsp/ltas.rs` (new) — `Ltas` + `ltas`.
- `crates/python/src/lib.rs` — `PyLtas` + `ltas` pyfunction; `python/sadda/dsp/__init__.py` re-export.
- `tests/clinical/praat/jitter_shimmer.praat` — `ltas_slope` column; `clinical_perturbation.rs` + `test_ltas.py` tests.

### Sources / references

- Praat (Boersma & Weenink) — Ltas / Get slope
- 2026-05-25 CPP/CPPS entry (same offset-invariance argument); the validation harness this extends

---

## 2026-05-25 — CPP/CPPS (B5, part 2): cepstral peak prominence, robust tilt, Praat-validated

Completes cluster B's B5. Smoothed cepstral peak prominence — the prominence of the cepstral peak (at the f0 quefrency) above the cepstrum's regression tilt line, averaged over frames. Praat's `PowerCepstrogram` → `Get CPPS`.

### Why this was easier than HNR

CPP is a **prominence** — `peak − tilt_line`, both in dB. Any constant offset on the cepstrum (FFT normalization, `ln` vs `log10`, even log-power vs log-magnitude — all *additive* in dB) **cancels in the subtraction**. So unlike HNR's `r→1` hypersensitivity, CPP is robust to implementation conventions; only the cepstrum *shape* and the regression matter. First implementation was already within ~4 dB of Praat.

### Design

`engine::clinical::cpps`: per frame, Hann window → real FFT → log power spectrum → inverse FFT (the cepstrum) → quefrency-smoothed power → dB → peak in the f0 quefrency band → **robust (IRLS bisquare) straight-line tilt** over [0.001, 0.05] s → `CPP = peak − line(peak_q)`; mean over frames. Two tuned-to-Praat pieces:

- **Robust tilt regression** (not plain least-squares) so the cepstral peak doesn't drag the fitted line up and shrink the prominence — matches Praat's "Robust" fit.
- **Quefrency smoothing in the power domain** (≈0.00015 s, calibrated to Praat). Unsmoothed CPP over-reads by ~4 dB; smoothing the *power* cepstrum lowers the sharp peak ~10·log10 (vs magnitude smoothing's harsher 20·log10, which over-shot to −6 dB). The window is calibrated against the reference rather than replicating Praat's resample-to-10 kHz cepstrogram pipeline.

Matches Praat within **3 dB** on the harmonic-tone fixtures.

### Validation signal choice

CPPS is validated on the **sustained harmonic-tone** fixtures (the HNR signals), which are the appropriate cepstral input — a vowel is harmonic-rich, not an impulse train. The B4 jitter/shimmer **pulse-train** fixtures have a degenerate cepstrum and diverge 15–20 dB from Praat; they aren't valid CPP inputs and are excluded from the CPP test.

### Three surfaces

`engine::clinical::cpps` + `sadda.clinical.cpps` (`stable_clinical`). GUI via the script panel; dedicated display with cluster D.

### What this doesn't do

- **Praat's exact CPPS pipeline** (resample to 2·maxFreq, its precise smoothing/window) — approximated with a calibrated power-domain quefrency smoothing, validated to 3 dB.
- **Time smoothing** — negligible for stationary sustained tones; not implemented.
- **FFT-accelerated** per-frame work — fine at analysis sizes.

### Cluster B status

B4 (jitter/shimmer) + **B5 (HNR + CPP/CPPS)** done. Next: **B6 (AVQI + ABI)** — composite indices that *combine* CPPS + HNR + shimmer + spectral slope/tilt, so B4/B5's measures feed directly into B6.

### Layout

- `crates/engine/src/clinical.rs` — `cpps` + `CppsConfig` + `robust_line` / `moving_average`.
- `crates/python/src/lib.rs` — `cpps` pyfunction; `python/sadda/clinical/__init__.py` re-export.
- `tests/clinical/praat/jitter_shimmer.praat` — CPPS column; `clinical_perturbation.rs` CPPS test.

### Sources / references

- Hillenbrand et al. 1994; Heman-Ackah et al. 2003 (CPP / CPPS)
- Praat (Boersma & Weenink) — PowerCepstrogram / Get CPPS
- 2026-05-25 clinical-validation-references entry; B5-part-1 (HNR) entry (shared harmonic-tone fixtures)

---

## 2026-05-25 — HNR (B5, part 1): faithful Boersma cross-correlation harmonicity, Praat-validated

Second cluster-B measure: mean harmonics-to-noise ratio over a sustained phonation, via the Boersma-1993 **cross-correlation** method (Praat's `To Harmonicity (cc)`). CPP/CPPS (the rest of B5) follows.

### The finding that shaped this

First attempt reused the pitch tracker's voicing — `HNR = 10·log10(r/(1−r))` with `r` = the window-corrected normalized autocorrelation peak. **Dead end**: that mapping is hypersensitive near `r → 1`, where a clean tone lives. On a 25 dB-SNR tone, Praat's `r ≈ 0.997` (→25 dB) but the pitch tracker's `r ≈ 0.99995` (→43 dB) — an **18 dB error** from a 0.003 difference in `r`. The plain (un-corrected) autocorrelation went the other way (Hann taper → r under-reads → 4 dB). Neither tracks Praat.

The fix is the **cross-correlation** `r` Praat actually uses:

```
r(τ) = Σ xᵢ·xᵢ₊τ / √( Σ xᵢ² · Σ xᵢ₊τ² )
```

The **geometric-mean** energy normalization (vs autocorrelation's `R(0)` and its taper/correction games) is what makes `r` track Praat near 1. Implemented standalone in `engine::clinical::hnr` (not via the pitch tracker): per-frame max `r` over the pitch lag range → HNR, mean over non-silent frames (1%-energy silence gate). Matches Praat within **3 dB** on the fixtures (25 dB and 12 dB cases).

### Fixtures

The HNR signals are **sustained harmonic tones** (1/h-harmonic glottal-source-like) + additive noise at a target SNR — *not* the B4 pulse trains, which are built for discrete-pulse jitter/shimmer and whose autocorrelation at the period lag sits well below R(0). The synth harness (`synth_fixtures.py`) grew a harmonic-tone branch; the Praat script grew an HNR column (`To Harmonicity (cc)` → mean); golden values committed.

### Three surfaces

`engine::clinical::hnr` + `sadda.clinical.hnr` (`stable_clinical`). GUI: via the embedded script panel (as with B4); dedicated display with cluster D.

### What this doesn't do

- **CPP / CPPS** — the other half of B5; next.
- **FFT-accelerated cross-correlation** — the per-frame O(lags·window) cc is fine for test/analysis sizes; optimize if a long-file path needs it.

### Layout

- `crates/engine/src/clinical.rs` — `hnr` + `HnrConfig`.
- `crates/python/src/lib.rs` — `hnr` pyfunction; `python/sadda/clinical/__init__.py` re-export.
- `tests/clinical/praat/*` — harmonic-tone synth branch + HNR Praat column; `clinical_perturbation.rs` HNR test.

### Sources / references

- Boersma, P. (1993) — the cross-correlation HNR (also the project's pitch citation)
- 2026-05-25 clinical-validation-references entry (Praat-primary, tolerance-match) + B4 entry (the harness this extends)

---

## 2026-05-25 — Jitter + shimmer (B4): engine::clinical perturbation, pitch-synchronous period extraction, Praat-validated

First **cluster B (clinical algorithms)** slice. Jitter (local / rap / ppq5) and shimmer (local / local-dB / apq3 / apq5) over a sustained phonation, validated against Praat per the same-day validation-references entry.

### What B4 delivers

- **`engine::clinical::perturbation`** — estimates a nominal f0 (autocorrelation pitch median), detects one glottal pulse per period by **pitch-synchronous peak-picking**, and computes the perturbation quotients over the realized period / peak-amplitude sequences. Returns a `PerturbationReport` using the A2 units (jitter + relative shimmers as `Ratio`, `shimmer_local_db` as `Decibels`). Too few periods / no voiced f0 → the A2 `EngineError::Unreliable` (no fabricated number).
- **Validation harness** (`crates/engine/tests/clinical/`) — `synth_fixtures.py` synthesizes controlled-jitter/shimmer signals (deterministic pseudo-random perturbation → analytic ground truth in `injected.json`); `jitter_shimmer.praat` measures them with Praat 6.2.09 → `praat_golden.tsv`. Both committed with a README. The engine + Python tests assert their output is **within per-measure tolerance of the Praat golden value** (≈20–25% relative + an absolute floor) and in the ballpark of the analytic value.
- **Python** — `sadda.clinical.perturbation(audio) -> PerturbationReport`, tiered **`stable_clinical`**.

### Design notes

- **Period extraction = peak-picking, not Praat's cross-correlation.** A one-period search window starting ~0.7 period past the previous peak tolerates the jitter being measured. On the synthetic fixtures it matches Praat within tolerance (e.g. shimmer_local 6.0% vs Praat 6.3%; combined-signal jitter 1.5% vs Praat 1.5%). Cruder than Praat's cc for messy real voice — acceptable for v1's synthetic-primary corpus; a cc/refined extractor is a later improvement.
- **Fixtures use pseudo-random, not alternating, perturbation** — alternating jitter/shimmer drives the pitch tracker into period-doubling (the 200 Hz case read ~0% until switched). Praat golden values are committed, so **CI never needs Praat**.

### Three surfaces

Engine + Python as above. **GUI**: clinical measures are already reachable in-app via the **embedded CPython script panel** (E8/E9) — `sadda.clinical.perturbation` runs there today. A *dedicated* measures display (tracks + readouts) belongs with the **cluster-D overlays**, so no standalone widget lands here (same pattern as A3's deferred SPL display).

### What B4 deliberately doesn't ship

- **Praat's exact period detection** — the peak-picker is an approximation validated to tolerance, not a cc reimplementation.
- **Real-voice validation** — synthetic-primary; the small real set awaits a clean-licensed source (validation-references entry's open item).
- **Auto-recorded provenance** — `perturbation` is a pure function (no `Project`), like the DSP measures; a `clinical_measure` ProcessingRun is recorded when a result is *persisted* (the A1 persistence-boundary pattern), not inside the pure call.
- **A dedicated GUI display** — rides with cluster D.

### Layout

- `crates/engine/src/clinical.rs` (new); `crates/engine/tests/clinical_perturbation.rs` + `tests/clinical/{praat,fixtures}/`.
- `crates/python/src/lib.rs` — `PyPerturbationReport` + `perturbation`; `python/sadda/clinical/__init__.py` (new submodule).
- Tests: 4 engine fixture + 2 engine unit + 4 Python.

### Next

B5 (HNR + CPP / CPPS), then B6 (AVQI + ABI) — each Praat-validated against committed golden fixtures.

### Sources / references

- 2026-05-25 clinical-validation-references entry (the contract this implements)
- Praat (Boersma & Weenink) — Voice report jitter/shimmer definitions
- 2026-05-25 A2 entry (units + `Unreliable`); A1 entry (provenance at the persistence boundary)

---

## 2026-05-25 — Clinical validation references: Praat-primary, tolerance-match, synthetic + small-real corpus

Goal: settle the validation contract for cluster B's clinical measures — the prerequisite both the Phase-3 slicing entry and the 2026-05-18 clinical-regulatory entry flagged as required *before* B4. Cluster A (substrate) is complete; this entry unblocks the algorithms. It fixes *what* each measure is validated against, *how* "validated" is defined, and *on what data* — not the algorithms themselves (those land per B-slice).

### Decisions (2026-05-25 design session)

| Question | Decision |
|---|---|
| **Primary reference** | **Praat primary; MDVP a documented secondary cross-check.** Praat is open, scriptable (we can run it to generate reference values), reproducible, and the de-facto research standard; the AVQI and CPPS references are themselves Praat-based. MDVP (KayPENTAX) is the clinical standard many clinicians know, but it is proprietary — we can neither redistribute its values nor re-derive them — so it cannot be the validation *target*. Where Praat and MDVP are known to diverge for a measure, the divergence is documented in the measure's docs, never silently averaged. |
| **What "validated" means** | **Match the Praat reference value within a per-measure tolerance** (computational reproducibility), for now. **Flagged to revisit:** whether measures merit a deeper **construct-validity** check — see below. |
| **Corpus** | **Both** — synthetic signals with analytic ground truth *and* a small clean-licensed real-voice set. |

### Reference per measure

The Praat features each cluster-B measure validates against (exact parameters + precise published refs land in each B-slice and its A1 citation-registry entries; this fixes the *target*):

| Measure | Praat reference | Literature |
|---|---|---|
| Jitter (local, rap, ppq5) | Voice report via PointProcess (periodic, cc) | Boersma & Weenink, Praat manual |
| Shimmer (local, local dB, apq3, apq5) | Voice report (same PointProcess) | " |
| HNR | `To Harmonicity (cc)` → mean | Boersma 1993 |
| CPP / CPPS | `To PowerCepstrogram` → CPPS | Hillenbrand et al. 1994; Heman-Ackah et al. 2003 |
| AVQI | Maryn AVQI Praat script (v3) | Maryn et al. 2010 + revisions |
| ABI | Barsties ABI Praat script | Barsties von Latoszek et al. |

### Methodology

- **Reference values are generated offline and committed as golden fixtures.** Run Praat (via its scripting interface) once over the shared test inputs, capture the outputs, and check them in as test data. CI asserts our outputs match the golden values within the per-measure tolerance — so **CI never needs Praat installed**, and numerical drift fails the build (the slicing entry's CI drift gate). The Praat version is recorded alongside the fixtures (CPPS/AVQI specifics have shifted across Praat releases).
- **Per-measure tolerance** is set in each B-slice's design pass — tighter for deterministic measures, looser where windowing / pitch-tracking sensitivity makes exact reproduction unrealistic. A measure that can't be computed reliably on a fixture returns the A2 `Unreliable` error rather than a number to compare.
- **Two corpora, two notions of truth:**
  - *Synthetic* — synthesized voiced signals with **injected, analytically-known** perturbation (controlled jitter/shimmer on a glottal-pulse train; HNR via added noise at a known SNR). Truth is the injected value; no licensing; fully reproducible. Primary for unit-level exactness.
  - *Small real set* — one small, clean-licensed sustained-vowel set (normal + disordered). Truth there is the Praat reference value (no analytic truth for real voice); exercises messy real signals. **Sourcing is open** (the standing sample-data-licensing concern); synthetic carries v1 if sourcing slips, with the real-set tests a stub until a set is found.

### Construct validity (deferred — revisit)

Matching Praat establishes *computational reproducibility* — we compute the measure the way the reference does. It does **not** establish *construct validity* — that the measure actually indexes the clinical construct (dysphonia severity, breathiness) on our data. Different validation tiers. v1's contract is reproducibility; a later pass — once measures are landed and a labelled corpus exists — assesses criterion/construct validity (e.g. AVQI vs expert perceptual ratings). Tracked as deferred; not gating cluster B.

### What this entry doesn't decide

- **Exact per-measure tolerances** — per B-slice design pass.
- **The specific real dataset** — sourcing TBD; synthetic is the non-blocking fallback.
- **The pinned Praat version** — pinned when B4 generates the first fixtures.
- **Algorithms** — each B-slice designs its own; this entry fixes only the validation target.

### Unblocks

Cluster B is cleared to start: **B4 (jitter + shimmer)** → B5 (HNR + CPP/CPPS) → B6 (AVQI + ABI), each with a Praat-golden + synthetic validation suite. 0.3.0 = cluster A (done) + cluster B.

### Sources / references

- 2026-05-18 clinical-regulatory-stance entry (deferred this entry; validation-suite commitment #2)
- 2026-05-25 Phase 3 slicing entry (cluster B gated on this entry)
- Praat (Boersma & Weenink) — Voice report / harmonicity / power-cepstrogram manual sections
- Maryn et al. (AVQI); Barsties von Latoszek et al. (ABI); Hillenbrand et al. 1994 and Heman-Ackah et al. 2003 (CPP / CPPS)

---

## 2026-05-25 — Feature intake: microphone registry + frequency-response correction

Logged during A3, **not yet designed** — captured here so it isn't lost and so its connections to existing work are recorded.

**The feature.** A curated registry of microphones and their published properties — frequency range, sensitivity, impedance, dynamic range, self-noise, max SPL, polar pattern, and the **frequency-response function (FRF)** — plus **inverse FRF correction**: flatten a recording's spectral coloration by applying the mic's inverse response, so measured spectra/formants/levels reflect the signal rather than the transducer.

**How it connects to what exists:**

- **Extends A3 calibration.** A3 ships a single *broadband* dB-FS→dB-SPL offset; the FRF is the per-*frequency* generalization. A registry mic supplies its published FRF as the correction basis (refined by a calibration tone where available). The `Calibration` model grows a frequency-dependent variant; the `Instrument` entity gains a "registry mic id" reference.
- **Third instance of the registry pattern.** The refdist-governance + ML-model-registry entries already define the mechanism: TOML-manifest entries, curated/community tiers, GitHub-Pages index, resolve-by-id + cache, provenance + citation, in-app publish. A microphone registry reuses it wholesale — a `mic.toml` with the properties above. Decide at design time: a parallel registry vs a sub-namespace of refdist.

**Open questions for its own design session:**

- **FRF representation** — magnitude-only vs complex (phase); published "typical unit" curve vs per-serial measured response; how the two compose with A3's calibration tone.
- **Inverse-correction method** — minimum-phase inverse filter vs regularized deconvolution vs magnitude-only EQ; how to bound noise amplification at roll-off frequencies.
- **Data + licensing** — manufacturer FRF data is often published as plots/typical curves with unclear redistribution rights; same sourcing concern as the refdist starter set.

**Where in the plan:** Phase 3+ (depends on cluster C's reference-distribution infrastructure being in place) or v1.x. Needs its own design entry; intake only here.

### Sources / references

- 2026-05-25 A3 entry (the single-offset `Calibration` this generalizes)
- 2026-05-18 reference-distribution-governance + 2026-05-20 ML-model-registry entries (the registry mechanism to reuse)

---

## 2026-05-25 — Instrument calibration + calibrated SPL (A3): Instrument CRUD, reference-pair calibration, bundle→session→instrument resolution

Third Phase-3 slice, and the last of **cluster A (clinical substrate)**. Adds the calibration path that turns the engine's relative dB-FS readings into absolute dB-SPL — a precondition for clinically meaningful intensity, which the cluster-B measures will consume.

### What A3 delivers

- **`Calibration`** — a flat single-offset model: a reference tone of known SPL recorded at a measured dB-FS pins `offset = reference_spl_db − reference_db_fs`. The reference *pair* is stored (not just the offset) so the calibration is auditable. `spl_offset_db()` + `to_spl(Decibels) -> Decibels`.
- **Instrument CRUD** — the `instrument` table existed (B1, schema-only); A3 adds `Instrument` / `InstrumentSpec` + `add_instrument` / `instruments` / `get_instrument`. `Calibration` is serialized as JSON into the existing generic `calibration TEXT` column.
- **`Project::bundle_calibration(bundle_id)`** — resolves a bundle's calibration by walking **bundle → session → instrument** (sessions already carried `instrument_id`; this is the first code to populate and read instruments). `None` = levels are dB-FS only.
- **Three surfaces** — Python `Instrument` / `Calibration` classes (`Calibration(reference_spl_db=…, reference_db_fs=…)`, `.to_spl(db_fs)`) + the CRUD + `bundle_calibration`; GUI shows a **Levels: calibrated (dB-SPL, +X dB) / uncalibrated (dB-FS only)** line in the bundle provenance modal.

### Design decisions

| Decision | Reasoning |
|---|---|
| Store `Calibration` as **JSON in the existing `calibration TEXT` column** | No V8 migration, and no audit-trigger rebuild (the B1 discipline: any `ALTER` on an audited table must drop+recreate its 3 triggers). Added `serde` + `serde_json` to the engine — both already in the lock tree |
| **Single-offset** calibration model | The foundational dB-FS→dB-SPL mapping; frequency-response curves (the clinical entry's richer form) are deferred until a real consumer needs them |
| Store the **reference pair**, not just the derived offset | Auditable — clinical provenance wants "how was this calibrated," not just the number |
| GUI surface = a line in the **provenance modal** | Calibration is "how these levels were measured" — provenance-adjacent. Avoids per-frame DB queries (the modal loads a one-shot snapshot) |
| Lenient calibration parse | A null/legacy/unparseable `calibration` value reads as `None` (uncalibrated), never fails the whole instrument query |

### What A3 deliberately doesn't ship

- **Frequency-response curves / per-frequency calibration** — single broadband offset only for now.
- **A full instrument-management GUI** and a **calibrated-SPL display** — there's no intensity track in the GUI yet, so SPL has no plot to render into. Instrument setup is via the Python API / script for now; the management UI + SPL display ride with the cluster-B intensity measure and the cluster-D overlays.

### Cluster A is complete

A1 (provenance) + A2 (units + discipline) + A3 (calibration) close the clinical substrate. **Next is cluster B (clinical algorithms)** — jitter/shimmer/HNR/CPP/AVQI/ABI — which is gated on the prerequisite **clinical validation-references** design entry (pick the reference implementation + values per measure; the slicing entry flagged it must precede B4).

### Layout

- `crates/engine/src/corpus.rs` — `Calibration`, `Instrument`, `InstrumentSpec`, `add_instrument` / `instruments` / `get_instrument`, `bundle_calibration`. `crates/engine/Cargo.toml` — `serde` + `serde_json`.
- `crates/python/src/lib.rs` — `PyCalibration` (`from_py_object`) + `PyInstrument` + the methods.
- `crates/app/src/main.rs` — calibration line in `ProvenanceView`.

### Sources / references

- 2026-05-25 Phase 3 slicing entry (A3 scope; cluster-B gating on validation-references)
- 2026-05-18 clinical-regulatory-stance entry (commitment #4: calibrated SPL + mic profiles on the Instrument entity)
- 2026-05-18 corpus-data-model entry (the `instrument` table + session→instrument link)

---

## 2026-05-25 — Typed units + clinical discipline (A2): lightweight unit newtypes, no-silent-fallback error, stable-clinical marking, research-use-only labeling

Second Phase-3 slice (cluster A, clinical substrate). Lands the measurement-discipline substrate the cluster-B clinical measures build on: typed units, an explicit "couldn't compute reliably" error, a stronger API-stability tier for the clinical surface, and research-use-only labeling.

### Design forks settled this session

- **Unit modelling: lightweight newtypes, not the full `uom` crate.** The clinical-regulatory + slicing entries named `uom`; in practice the engine's quantity set is small and the Python surface returns NumPy arrays of plain floats, so `uom`'s verbose `Quantity` types add ceremony without crossing the FFI boundary as numbers anyway. Newtypes make the unit part of the *type* (a signature reads `Hertz`, not `f32`) and unwrap to the primitive at the boundary — 90% of the discipline at 10% of the cost. Full dimensional algebra (catching Hz+seconds) is the part we give up; not worth it for this quantity set.
- **Retrofit the existing DSP, not just new measures.** Applied now rather than waiting for cluster B.

### What A2 delivers

- **`engine::units`** — `Hertz`, `Decibels`, `Ratio` (f32-backed) and `Seconds` (f64-backed) newtypes: `new` / `value`, `Display` with the unit, ordering for thresholds. The "no bare numbers" substrate.
- **Retrofit** — `PitchFrame.frequency_hz`, `FormantFrame.frequencies` / `bandwidths` (+ the `Live*Frame` mirrors) → `Hertz`; `IntensityFrame.db_fs` / `MeterFrame.rms_db` (+ live) → `Decibels`. Producers wrap; the PyO3 getters and live callbacks unwrap with `.value()`, so **the Python API is unchanged** (the `.pyi` stub diff is empty) — the type safety is Rust-internal.
- **No-silent-fallback** — `EngineError::Unreliable { measure, reason }` + `EngineError::unreliable(…)`. A measure on insufficient signal returns this *instead of a guessed number*; mapped to Python `ValueError`.
- **Stable-for-clinical-use marking** — a `stable_clinical` stability tier (no runtime warning, like `@stable`, but a distinct label `"stable-clinical"` that `get_stability` surfaces) for the stronger change-control the clinical surface needs.
- **Research-use-only labeling** — an always-visible status-bar notice in the app ("For research, education, and non-diagnostic use only") and an **Intended use** section in the README. Posture 3, never a clinical claim.

### Scope decisions

| Decision | Reasoning |
|---|---|
| Units on **frequency + level** only | Hz and dB are the clinically load-bearing, confusion-prone quantities. `time_seconds` stays a bare `f64` (the engine's universal time type — wrapping it everywhere is high-churn, low-value); linear `rms` and `voicing` stay bare (`Seconds` / `Ratio` exist in the module for cluster B) |
| Python keeps returning floats | Newtypes unwrap at the PyO3/NumPy boundary; no API break, stub unchanged |
| `Unreliable` carries `measure` + `reason` | Names what failed and why — honest for clinical contexts, no fabricated value |

### What A2 deliberately doesn't ship

- **Full dimensional analysis** — newtypes don't prevent `Hertz + Seconds`; that mixing isn't a realistic risk for this code, and `uom`'s cost wasn't justified.
- **Per-measure uncertainty quantification** — A2 ships explicit *errors* on bad input; richer "value + uncertainty" returns are deferred (clinical entry).
- **The regression-test / numerical-drift CI gates** for `stable_clinical` measures — those land *with* cluster B's measures, which are the first things to carry the tier.

### Layout

- `crates/engine/src/units.rs` (new); retrofit in `dsp/{intensity,formants}.rs`, `pitch.rs`, `live/mod.rs`; `error.rs` (`Unreliable`).
- `crates/python/src/{lib,live}.rs` — getters/callbacks unwrap; `engine_err_to_py` maps `Unreliable`.
- `python/sadda/_stability.py` + `__init__.py` — `stable_clinical`.
- `crates/app/src/main.rs` — status-bar RUO notice. `README.md` — Intended use.

### Sources / references

- 2026-05-25 Phase 3 slicing entry (A2 scope); this session's A2 design forks (newtypes; retrofit existing DSP)
- 2026-05-18 clinical-regulatory-stance entry (commitments: typed units #5, no-silent-fallback #6, stable-API marking #8, RUO labeling #1/#10)
- 2026-05-21 DSP-method-diversity principle (the measure path the units now type)

---

## 2026-05-25 — Provenance coverage + citation export (A1): `record_processing_run` backbone, per-bundle timeline, citation registry

First Phase-3 slice (cluster A, clinical substrate). Builds the provenance backbone every later clinical / ML / DSP slice records into, plus citation export. The `processing_run` table + `kind` enum already existed (B1 / V6); A1 adds the unified write path, a per-bundle query, and the citation layer on top.

### What A1 delivers

- **`Project::record_processing_run(&ProcessingRunSpec) -> i64`** — the single insert path for a completed run. The engine fills in the sadda version, `started_at` / `finished_at`, and the active recipe id (`current_recipe_id`); the caller supplies `ProcessingRunSpec` (bundle, `ProcessingRunKind`, `processor_id`, optional params / input+output tier ids / output signal ids / weights checksum / status). New `ProcessingRunKind` + `ProcessingRunStatus` enums give the `kind` / `status` CHECK values type-safety.
- **`Project::processing_runs(bundle_id) -> Vec<ProcessingRunRow>`** — the per-bundle provenance timeline (insertion order), alongside the pre-existing `processing_runs_for_recipe`.
- **Citation registry** (`engine::citation`) — `Citation { processor_id, reference, doi }` + `citation_for(processor_id)`, the **machine-readable source of truth** for citation export. Seeded from the DSP modules' curated `## References` blocks (Boersma 1993 for pitch, McCandless 1974 / Markel 1972 for formants, Makhoul 1975 for LPC, Davis & Mermelstein 1980 for MFCC, Allen 1977 for STFT/spectrogram) rather than reproduced from memory. Uncited processors (imports, recording) return `None`.
- **`Project::citations(bundle_id) -> Vec<Citation>`** — walks the bundle's runs, dedups by `processor_id`, preserves first-use order, drops uncited processors.
- **Three surfaces**: Python `record_processing_run` / `processing_runs` / `citations` + `ProcessingRun` / `Citation` pyclasses (stubs regenerated; tiered `@stable`); GUI **"Provenance & citations…"** modal off the bundle context menu — read-only run list + deduped citation list with DOI links and a **Copy references** clipboard action.

### Design decisions

| Decision | Reasoning |
|---|---|
| One `record_processing_run` insert path | Every analysis goes through it, so the timeline + citations stay complete. Refactored the TextGrid / EAF import inserts onto it (behavior-preserving DRY) |
| Citation registry keyed by `processor_id` | Reverse-DNS ids (`sadda.dsp.pitch.autocorrelation`) per the ML-registry entry; the registry is the one machine-readable mirror of the per-method doc citations |
| Empty id lists → SQL `NULL`, not `"[]"` | Distinguishes "no outputs" from an empty array in the stored JSON |
| Citations dedup by processor, first-use order | A reference list wants each work once, in the order analyses first used it |
| `commit_recording` left on its own insert | It writes inside the bundle-commit transaction with rename-back error handling; not worth threading through the new path this slice |

### What A1 deliberately doesn't ship

- **Auto-recording of every DSP / clinical call.** `sadda.dsp.*` are pure functions that don't touch a `Project`; provenance is recorded at the *persistence boundary* (imports / recording today; derived-signal writes and the cluster-B clinical measures as they land). A1 ships the backbone + the API for callers to record; blanket auto-coverage completes through B and a later derived-signal-write hook.
- **Richer citations** — one primary reference + DOI per processor; multiple-refs-per-method and BibTeX/`.bib` export are later enhancements.
- **Start-then-finish runs** — A1 records completed runs (`started_at` ≈ `finished_at`); long-running async runs that stamp start then finish come with the ML cluster.

### Acceptance vs the slicing-entry target

The slicing entry's A1 acceptance ("pitch track then AVQI → complete timeline; `citations()` lists both") can't fully land until clinical measures (B) exist. A1 satisfies the achievable half: recording any cited processor (e.g. `sadda.dsp.mfcc`) makes it appear in both the timeline and `citations()`, verified in engine + Python tests; imports already record provenance through the new path.

### Layout

- `crates/engine/src/citation.rs` (new) — `Citation` + `citation_for`.
- `crates/engine/src/corpus.rs` — `ProcessingRunKind` / `ProcessingRunStatus` / `ProcessingRunSpec`, `record_processing_run`, `processing_runs`, `citations`; import inserts refactored.
- `crates/python/src/lib.rs` — `PyProcessingRun` / `PyCitation` + three `Project` methods + `parse_run_kind`.
- `crates/app/src/main.rs` — `ProvenanceView` + the modal.
- Tests: 3 engine (citation registry) + 2 engine (corpus provenance/citations) + 3 Python.

### Sources / references

- 2026-05-25 Phase 3 slicing entry (A1 scope + the three-surface validation/registry extensions)
- 2026-05-20 ML-model-registry entry (`ProcessingRun` shape, `kind` discriminator, reverse-DNS `processor_id`)
- 2026-05-18 clinical-regulatory-stance entry (provenance as commitment #1; citation export)
- 2026-05-21 DSP-method-diversity principle (every method carries a published source — now surfaced via the citation registry)

---

## 2026-05-25 — Phase 3 slicing: 12 slices in 5 clusters, released incrementally 0.3.0 → 0.3.2

Goal: sequence Phase 3 — "Differentiators part 1," the clinical-ready release — into a commit-by-commit ordering, analogous to the Phase 1 (2026-05-21) and Phase 2 (2026-05-23) slicing entries. Phase 3's *scope* and most of its *design* are already settled across four prior entries (reference-distribution governance, clinical regulatory stance, ML model registry, profile catalog); this entry commits to *cadence and ordering*. It is the largest phase in the milestone plan (3–4 months solo part-time).

### What ships across 0.3 (from the milestone plan)

> Reference distribution infrastructure (format + resolver + GitHub registry + CI + Pages index + in-app publish) + bundled starter set + GUI overlay rendering + clinical algorithms (AVQI / ABI / CPP / jitter / shimmer / HNR) with validation suite + provenance + uom typed units + calibrated SPL + research-use-only labeling + ort runtime + VAD bundled + wav2vec2/Whisper on-demand download + embedding tiers

**Phase 3 design conversation (2026-05-25), confirmed:**

- **Full Phase 3 scope** — refdist + clinical + ML all land in the 0.3 line. Not narrowed despite the size; the milestone plan's "differentiators" thesis keeps clinical and ML together.
- **Incremental sub-releases** rather than one 0.3 at phase end. The largest phase is the one most worth de-risking with intermediate usable artifacts. Each sub-release tags both tracks where applicable: PyPI (`v0.3.x`) for new Python surface, app binaries (`v0.3.x-app`) via the existing G11 workflow.
- **Clinical substrate first** — provenance / units / calibration before the algorithms that depend on them. Infrastructure-first, matching the A1 cadence of every prior phase.

### What changes from Phase 2

| | Phase 2 | Phase 3 |
|---|---|---|
| Slices | 11 | 12 |
| Release | single 0.2 at phase end | **incremental: 0.3.0 → 0.3.1 → 0.3.2** |
| Each slice ends in | a clickable feature | a measured value, an overlay, or an inference — most with a validation suite |
| Surfaces per slice | engine (small) + app + manual verify | engine + Python + GUI **+ validation suite** (clinical) **+ registry repo + CI** (refdist) |
| Headline risk | render performance | **numerical correctness** (clinical measures must match published references) + bundle size (ML runtime) |

The three-surface rule (engine + Python + GUI, adopted after H1) holds, with two Phase-3 extensions: every **clinical** measure adds a fourth surface — a **validation suite** (known-input/known-output vs published reference values, run in CI with numerical-drift gates); every **refdist/model** capability adds a fifth — the **public registry repo + its CI**, a separate artifact from the app repo.

### Prior-art shape (clinical measures + ML runtime; refdist covered in its own entry)

| Source | What we lift | What we leave |
|---|---|---|
| **Praat** | The de-facto reference for jitter / shimmer / HNR / CPPS; published algorithm definitions; AVQI Praat scripts (Maryn) | Its exact windowing quirks where better-documented variants exist |
| **MDVP (KayPENTAX)** | The clinical perturbation-measure vocabulary clinicians expect | Proprietary; disagrees numerically with Praat on "the same" measure — we validate against a *chosen* reference, not both silently |
| **VoiceSauce / COVAREP** | Batch spectral/voice-quality measure suites; reproducible parameterization | MATLAB heritage; no corpus model |
| **AVQI / ABI (Maryn et al.)** | The composite-index formulas + their validated component set (CPPS, HNR, shimmer, slope, tilt) | — (adopt directly, validate against published values) |
| **ONNX Runtime (`ort`)** | Cross-platform CPU/GPU inference, the Phase-3-committed runtime; ONNX as canonical curated format | Training; framework-specific formats (handled at publish time, not runtime) |
| **MFA models registry / HuggingFace Hub** | Model distribution pattern (covered in the ML-registry entry: parallel registry, HF passthrough escape hatch) | — |

### Decomposition: 5 clusters, 12 slices

Slice numbering continues the per-phase counter (1–12); cluster letters restart at A.

**Cluster A — Clinical substrate** (lands first; everything clinical depends on it)

1. **A1 — Provenance coverage + citation export.** The `processing_run` table + `kind` enum already exist (B1 / V6); A1 makes *every* DSP and clinical analysis record a run (`kind = dsp_algorithm | clinical_measure`), populating `processor_id` / `processor_version` / `parameters` / input+output tier ids. Adds project-level **citation export** that walks the table and emits a reference list (refs come from each processor's registered citation). Acceptance: running a pitch track then an AVQI leaves a complete, queryable provenance timeline; `project.citations()` lists both.
2. **A2 — Typed units + clinical discipline.** Introduce `uom` on the clinical-path API (no bare numbers); explicit **no-silent-fallback** error/uncertainty types (a measure on insufficient signal returns an error, never a guess); the **stable-for-clinical-use** API-marking convention (stronger than the general `@stable` tier); and **research-use-only labeling** (startup notice, docs banner, Intended Use statement). Acceptance: a clinical measure on a 50 ms silent clip returns a typed error, not a number; the app shows the RUO notice.
3. **A3 — Instrument calibration + calibrated SPL.** Build the `Instrument` API (table exists, schema-only since B1) with structured calibration: mic sensitivity, frequency-response curve, reference level. Compute **calibrated SPL** (dB-SPL re 20 µPa) when a bundle is tied to a calibrated instrument; fall back to dB-FS otherwise (explicitly, per A2). Likely a V8 migration for structured calibration columns. Acceptance: a bundle with a calibrated `Instrument` reports SPL; one without reports dB-FS and says so.

**Cluster B — Clinical algorithms** (each: engine + Python + GUI readout + validation suite + CI drift gate)

4. **B4 — Jitter + shimmer.** Period-perturbation measures over the C2 pitch/period track; the standard family (local, rap, ppq5 / local-dB, apq3, apq5). Validated against published reference values.
5. **B5 — HNR + CPP / CPPS.** Harmonics-to-noise ratio and cepstral peak prominence (smoothed). CPPS is the single most clinically load-bearing measure (AVQI's dominant term) and the spectral-domain counterpart to B4's time-domain perturbation.
6. **B6 — AVQI + ABI.** The composite indices, built from B4/B5 components plus spectral slope/tilt. Validated against Maryn et al. published values. Acceptance: AVQI on the reference recordings reproduces published scores within tolerance.

**Cluster C — Reference distribution infrastructure** (the cross-cutting differentiator; per the refdist-governance entry)

7. **C7 — Format + resolver + query + pinning.** `refdist.toml` manifest parsing; engine resolver with the user-level cache (`~/.local/share/sadda/refdist/`); `engine.refdist.query(measure=…, population=…)`; project version-pinning in `project.toml`. The *consumption* side — resolve, cache, query, pin — independent of any hosted registry.
8. **C8 — Registry repo + CI + Pages index + bundled starter set.** Stand up the public `refdist-registry` repo (tier 2 / tier 3 dirs); CI validation (TOML schema, license check, `min_n_per_subgroup`, data-file conformance); GitHub-Pages-rendered index JSON; and the **bundled tier-1 starter set** (Hillenbrand 1995, Peterson-Barney 1952, a small clinical normative-range set) shipped with the app. Gated on sourcing redistributable-licensed data.
9. **C9 — In-app publishing.** Auto-scaffold a manifest from an analysis result; fork-and-PR submission flow using the user's GitHub credentials. Closes the refdist loop (consume → publish).

**Cluster D — GUI overlay rendering**

10. **D10 — Measure tracks + refdist overlays.** Render clinical measures as tracks on the signal views, and refdist as **overlays / target zones / percentile bands**, visually distinguishing `measure.kind` (`observed_distribution` vs `summary_normative_range` vs `target_zone`) so "what people sound like" is never conflated with "what to aim for." Surfaces the A1 citation list in the export UI. This is the slice the critical path gates on C (refdist) + B (measures to plot).

**Cluster E — ML inference** (per the ML-registry entry; the model registry reuses the refdist mechanism)

11. **E11 — `ort` runtime + model registry + bundled VAD.** Integrate ONNX Runtime; stand up the parallel model registry (shared protocol with refdist, larger-artifact + compute-hint manifest fields, HF passthrough via `hf://`); ship a small **bundled VAD** as the first end-to-end model and the embedding-tier proof. `ProcessingRun kind = ml_model` with `weights_checksum`.
12. **E12 — On-demand models + embedding tiers.** wav2vec2 / Whisper resolved + downloaded on demand (with checksum validation + external-mirror support for >2 GB weights); inference results written as **embedding tiers** (`continuous_vector` dense signals from B3). Acceptance: `sadda.ml.load_model("sadda/wav2vec2-base-960h")` → embeddings stored as a queryable dense tier.

### Sub-release mapping

| Sub-release | Slices | Headline |
|---|---|---|
| **0.3.0** | A1–A3, B4–B6 | Clinical substrate + the six clinical algorithms with validation suites. The clinical-ready core. |
| **0.3.1** | C7–C9, D10 | Reference distributions end-to-end + GUI overlays. The norms/comparison differentiator. |
| **0.3.2** | E11–E12 | ML inference (ort + VAD + on-demand wav2vec2/Whisper + embedding tiers). |

Each sub-release is independently usable and tagged (`v0.3.x` for the PyPI surface, `v0.3.x-app` for binaries). Sub-release boundaries are the adjustable part — if the substrate (A) stabilizes well before the algorithms (B), it can ship as its own 0.3.0 with the algorithms following as 0.3.1.

### Dependency graph

```
A1 (provenance) ─→ A2 (units/discipline) ─→ A3 (calibration) ─┐
                                                               ├─→ B4 (jitter/shimmer) ─→ B5 (HNR/CPP) ─→ B6 (AVQI/ABI) ─┐
                                                               │                                                          ├─→ D10 (overlays)
C7 (resolve/query/pin) ─→ C8 (registry+CI+starter set) ─→ C9 (publish) ───────────────────────────────────────────────┘
                                                                                                              (refdist norms feed D10)
E11 (ort+registry+VAD) ─→ E12 (on-demand models + embedding tiers)        [independent of A–D; ships 0.3.2]
```

B needs A (units, provenance, no-silent-fallback). D10 needs B (measures) + C (norms to overlay). C is independent of B and can proceed in parallel. E is independent of everything else.

### Confirmed scope decisions

| Item | Decision | Reasoning |
|---|---|---|
| 0.3 scope | **Full Phase 3 (clinical + refdist + ML)** | Keeps the "differentiators" package intact; milestone-plan scope held |
| Release cadence | **Incremental 0.3.0 → 0.3.2** | De-risks the largest phase; ships usable artifacts mid-phase; matches the incremental-0.x principle |
| First cluster | **Clinical substrate (A)** | Provenance/units/calibration gate the algorithms; infra-first cadence |
| Clinical reference target | **One chosen reference per measure, not "match everything"** | Praat and MDVP disagree on "the same" measure; silently averaging is worse than a documented choice. Settled per measure in the validation-references design pass below |
| Units | **`uom` on the clinical path** | Compile-time unit checking; clinical entry's commitment |
| Unreliable inputs | **Typed error / uncertainty, never a guessed number** | No-silent-fallback; false confidence is harmful clinically |
| Model format | **ONNX canonical (curated tier); HF passthrough for the rest** | Phase-3 `ort` commitment; ML-registry entry |
| Registry shape | **Models reuse the refdist registry mechanism** | One protocol, two registries; ML-registry entry |

### Prerequisite design passes + parallel risk spikes

Unlike Phase 2, two clusters need a focused **design entry before their first slice**, plus the milestone plan's standing spikes:

| Item | Type | When | Purpose |
|---|---|---|---|
| **Clinical validation references** | design entry (required) | before B4 | Pick the reference implementation + reference values per measure (AVQI→Maryn; CPPS→Hillenbrand/Heman-Ackah; jitter/shimmer→Praat vs MDVP). Resolve disagreements explicitly. The clinical-regulatory entry flagged this as needing its own entry. |
| **`ort` runtime integration** | spike | before E11 | Confirm ONNX Runtime links + bundles cleanly across macOS/Linux/Windows and the binary-size impact, before E commits. |
| **Native plugin ABI** | spike (milestone plan) | late Phase 3, before Phase 4 | One trivial native plugin from a dylib; settles `abi_stable` vs C ABI vs WASM. |
| **Tongue-segmentation model exploration** | spike (milestone plan) | during Phase 3 | Survey/fine-tune candidates; de-risks Phase 4's heaviest deliverable. |

### Cut lines if timeline pressure hits

In priority order of what to defer first:

1. **Cluster E (ML inference)** → slips to a 0.3.3 / pulled into early Phase 4. It's the most self-contained and least coupled to the clinical headline.
2. **C9 (in-app publishing)** → ship consume + bundled starter set (C7/C8); defer user publishing. Most users consume long before they publish.
3. **B6 (AVQI/ABI)** → ship the component measures (jitter/shimmer/HNR/CPP) at 0.3.0; the composite indices follow. Components are individually useful.
4. **A3 (calibrated SPL)** → dB-FS only at first; calibrated SPL follows once a calibrated `Instrument` workflow is validated with a real clinical user.

**Not cuttable**: provenance + units + no-silent-fallback (A1/A2), at least the perturbation measures with validation (B4), refdist consume + starter set (C7/C8), research-use-only labeling.

### What this entry doesn't decide

- **Per-measure validation tolerances and reference datasets** — the prerequisite validation-references entry settles these (it's a correctness contract, not a cadence question).
- **Whether clinical-path code is restricted to in-tree / audited plugins** — the clinical-regulatory entry flagged a community plugin computing jitter wrongly as a distinct risk; policy TBD, likely alongside the Phase-4 plugin ABI.
- **Sub-release exact boundaries** — A vs A+B for 0.3.0; settled as the substrate stabilizes.
- **Per-measure uncertainty modeling** — A2 ships explicit errors on bad input; richer "value + uncertainty" returns are layered later (clinical entry).
- **Starter-set exact contents** — gated on confirming redistribution rights per dataset; publicly-licensed material only to start (refdist entry's open item).
- **Model cache eviction policy** — manual `sadda model gc` in v1; automatic LRU if disk pressure surfaces (ML-registry entry).

### Pace and revisit cadence

- Milestone plan estimates Phase 3 at 3–4 months solo part-time; 12 slices over ~16 weeks ≈ one slice every 1–2 weeks, same cadence as Phases 1–2.
- Revisit after **0.3.0** ships: the first clinical release with real measured values — clinical-research feedback may reshape B/C/D ordering and surface the workflow features (protocols, reports) deferred to v1.x.
- Revisit after the validation-references entry: if a chosen reference proves unreproducible, re-scope the affected measure.

### Sources / references

- 2026-05-18 milestone-plan entry (Phase 3 row; critical path: refdist before clinical; the after-0.2 revisit point this entry acts on)
- 2026-05-18 reference-distribution-governance entry (three-tier registry, manifest format, hosting, in-app publish)
- 2026-05-18 clinical-regulatory-stance entry (posture 3; the nine architectural commitments; AVQI/ABI/CPP/jitter/shimmer/HNR scope; validation-references deferral)
- 2026-05-20 ML-model-registry entry (ProcessingRun provenance; parallel model registry; ONNX canonical; HF passthrough; embedding tiers)
- 2026-05-20 profile-catalog entry (clinical profile; per-profile recommended distributions)
- 2026-05-21 / 2026-05-23 Phase 1 + Phase 2 slicing entries (analogous structure)
- `ort` (ONNX Runtime for Rust): <https://github.com/pykeio/ort>
- `uom`: <https://github.com/iliekturtles/uom>
- AVQI (Maryn et al.): <https://www.vvl.be/en/avqi>

---

## 2026-05-25 — Phase-2 GUI dogfooding: lane-alignment + spectrogram-bounds fixes shipped; bundle-rename and TikZ figure-export logged for later

Dogfooding the 0.2 desktop GUI on WSL2 surfaced three rendering/launch issues (fixed this session) and two feature gaps (logged here). This is an **intake entry, not a design entry** — the two backlog items still need their own design passes before implementation.

### Shipped this session (Phase-2 polish)

- **X11-under-WSL launch fix.** `cargo run -p sadda-app` aborted with `WinitEventLoop(ExitFailure(1))` under WSLg — winit's Wayland backend broken-pipes against the WSLg compositor. `main()` now drops `WAYLAND_DISPLAY` when WSL is detected (`WSL_INTEROP` / `/proc/sys/kernel/osrelease`), steering winit onto XWayland (`DISPLAY=:0`), which works. No-op off WSL, so native Wayland sessions are untouched.
- **Lane alignment.** The waveform / spectrogram / tier-strip plot areas didn't share left/right boundaries. Root cause: the waveform + tier panels carried `Frame::side_top_panel`'s 8px horizontal inner margin while the spectrogram was drawn frameless on the bare `ui`. Fix: all three lanes are now frameless (`Frame::NONE`); horizontal alignment is owned **solely** by the shared 120px gutter (`SIGNAL_LEFT_GUTTER` on the tier strip + matching `y_axis_min_width` on the two plots). The one outer `CentralPanel` supplies uniform window padding. This is the alignment convention to preserve as more lanes (f0 / intensity / formant tracks) land.
- **Spectrogram bounds.** The plot padded ~5% past the data (a meaningless 0…−1000 Hz band, plus empty time beyond the recording) and — latent — never cropped to the zoom window, because the full-bundle texture dominates egui_plot's auto-bounds (the old `include_x` comment claiming it "crops for free" was wrong). Fix: both plots now set exact bounds each frame via `PlotUi::set_plot_bounds_x/y`, which disables auto-fit (killing the margin) and crops the image to the visible window. x-bounds are now identical across panes, reinforcing the shared-cursor alignment.

### Backlog — not yet designed

**Bundle rename (near-term, fits current Phase 2).** No way to rename a bundle after creation — the selector only creates / deletes / reveals. A GUI editing affordance, which the roadmap's contributor-onramp note pegs to the post-0.2 window we're in. Cheap: the `bundle` table has `name TEXT NOT NULL` with **no** uniqueness constraint (only `audio_relative_path` is `UNIQUE`, and a display-name rename doesn't touch the WAV), so the engine side is a bare `UPDATE bundle SET name=? WHERE id=?`. Three-surface per the standing rule: engine gains `rename_bundle` / `update_bundle_name`, the Python module wraps it, the GUI exposes it from the existing bundle-row context menu (H1) via an inline or modal text edit, each with a test. Small enough to lift into the current phase whenever wanted.

**Publication-quality figure export → TikZ (future, ~Phase 3+).** Export the assembled view — selected panes (waveform, spectrogram), derived-signal overlays, labeled tiers, selections — as a TikZ figure for LaTeX, mirroring the user's existing Praat tooling, **specTeX** (https://github.com/dbqpdb/specTeX). Differentiator-tier and explicitly gated on "once the visual elements are all developed," so it lands after Phase-3 "GUI overlay rendering" at the earliest. Needs its own design session; open questions to settle there:
  - **Intermediate representation.** Does the engine/Python side emit a figure spec (keeping export scriptable + headless, three-surface), with the GUI just feeding it the current view? Strongly preferred over a GUI-only renderer.
  - **Spectrogram in TikZ.** TikZ can't draw the bitmap natively — it'd embed a PNG (the baked colormap texture) and overlay vector axes / tiers / annotations on top. Confirm against how specTeX handles it.
  - **Fidelity + units.** How closely to match on-screen styling; axis scaling; time/frequency units; tier label typesetting.

### What this entry doesn't decide

Neither backlog item is designed here — no IR, no API names beyond the sketch, no GUI layout. The bundle-rename sketch is concrete enough to implement directly when picked up; the TikZ exporter needs a design entry first, with specTeX as the prior-art baseline to study.

### Sources / references

- specTeX — the user's Praat TikZ figure exporter, prior art for the export feature: https://github.com/dbqpdb/specTeX
- 2026-05-20 roadmap entry (Phase 2 GUI scope; Phase 3 overlay rendering; contributor-onramp pegging GUI editing affordances to post-0.2)
- 2026-05-23 H1 entry (the bundle-row context menu the rename UI would extend)

---

## 2026-05-23 — Phase-2 polish (H1): File-menu I/O for TextGrid + EAF, live-recording dialog, bundle context menu, `Project::delete_bundle`

Goal: bring the GUI's I/O surface up to user expectation for a "desktop GUI release." The 2026-05-23 Phase 2 slicing entry's eleven slices delivered the visual editor, scripting host, single-writer lock, and release CI — but the existing engine methods for TextGrid / EAF I/O (D1, D2) and live recording (E1) were left unwired from the GUI menus. First post-0.2 user feedback flagged this as the most obvious gap.

This is a polish slice, not a re-scoping of Phase 2. The slicing entry pinned eleven items; this is a twelfth focused on closing user-discoverability gaps.

### What H1 must deliver

| Action | Engine | GUI wiring before | GUI wiring after |
|---|---|---|---|
| Import TextGrid (Praat) | D1 — `Project::import_textgrid` | ✗ | **File → Import → Praat TextGrid…** |
| Import EAF (ELAN) | D2 — `Project::import_eaf` | ✗ | **File → Import → ELAN .eaf…** |
| Export TextGrid | D1 — `Project::export_textgrid` | ✗ | **File → Export → Praat TextGrid…** |
| Export EAF | D2 — `Project::export_eaf` | ✗ | **File → Export → ELAN .eaf…** |
| Live record from mic | E1 — `LiveSession::start` | ✗ | **File → Record from microphone…** (modal) |
| Show project on disk | n/a | ✗ | **File → Show project folder** |
| Delete bundle | **new** — `Project::delete_bundle` | ✗ | **Right-click bundle → Delete** |
| Show bundle audio on disk | n/a | ✗ | **Right-click bundle → Show in file manager** |

### Engine surface: `Project::delete_bundle`

`Project::delete_bundle(id)` is the one new engine method this slice needs. SQL cascade across the rows that FK to `bundle`:

- `processing_run` (FK `bundle_id REFERENCES bundle(id)`)
- `tier` (FK `bundle_id`)
- `annotation_interval` / `annotation_point` / `annotation_reference` (FK `tier_id` → cascaded by deleting tiers)
- `derived_signal` (FK `tier_id` → same)
- The bundle's WAV under `signals/original/<name>.wav`

The schema's CHECK + FK constraints don't auto-cascade — SQLite needs `PRAGMA foreign_keys = ON` and explicit `ON DELETE CASCADE` declarations. We don't have either on the current schema (V3 didn't add cascade clauses to the FKs). So `delete_bundle` does explicit deletion in topological order inside a transaction:

1. `DELETE FROM derived_signal WHERE tier_id IN (SELECT id FROM tier WHERE bundle_id = ?)`
2. `DELETE FROM annotation_interval WHERE tier_id IN (...)`
3. Same for `annotation_point`, `annotation_reference`
4. `DELETE FROM tier WHERE bundle_id = ?`
5. `DELETE FROM processing_run WHERE bundle_id = ?`
6. `DELETE FROM bundle WHERE id = ?`
7. Best-effort `std::fs::remove_file(signals/original/<name>.wav)` — leaves WAV behind on disk if the corpus row deleted but the file isn't accessible. Audit trail records the DB deletion regardless.

Audit triggers on each table fire automatically, producing a `'delete'` row each. Deleting a busy bundle generates many audit rows but the corpus stays self-consistent.

### Live-recording modal

`File → Record from microphone…` opens a modal `Window`:

```
┌─────────────────────────────────────────────┐
│ Record from microphone                      │
├─────────────────────────────────────────────┤
│ Name:     [practice_take_1                ] │
│ Device:   [▾ default (system mic)       ]   │
│ Format:   44100 Hz / 1 channel              │
│                                             │
│ ┌─ Meter ─────────────────────────────────┐ │
│ │ peak  ▰▰▰▰▰▰▱▱▱▱▱▱  -8.4 dB-FS         │ │
│ │ rms   ▰▰▰▱▱▱▱▱▱▱▱▱  -22.1 dB-FS        │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ Status: idle / recording / stopped          │
│ Elapsed: 00:00:02.345                       │
│                                             │
│ [ ● Record ] [ ■ Stop ] [ Save ] [ Cancel ]  │
└─────────────────────────────────────────────┘
```

Lifecycle, using the existing `sadda_engine::live::LiveSession`:

- On open: enumerate devices via cpal (no-op fallback to default-only on enumerate failure). State = `Idle`.
- Click **Record**: build a `cpal::Stream` against the picked device feeding into `LiveSession`'s push_samples. State → `Recording`. Meter pulls peak/RMS from the result rtrb every frame.
- Click **Stop**: drop the stream + call `LiveSession::stop()`. State → `Stopped`. Show duration.
- Click **Save**: `Project::commit_recording(stopped, name, params_json)` → new bundle in the sidebar. Modal closes.
- Click **Cancel** (or window close at any state): drop the LiveSession; discard via `StoppedSession::discard` if Stopped; modal closes.

State plumbing matches E9's pattern — the cpal Stream stays on the GUI thread, GUI polls atomic counters each frame.

### Bundle row context menu

Right-click on a bundle row in the sidebar pops up:

- **Delete bundle** — confirmation dialog; on confirm, `Project::delete_bundle(id)`. If the deleted bundle was active, clear the selection.
- **Show audio in file manager** — open the OS file manager at the bundle's WAV path. UNIX: `xdg-open <dir>`; macOS: `open <dir>`; Windows: `explorer.exe <dir>`. Wrapped in a `cfg` block.

### File-menu reshuffle

Existing menu becomes nested:

```
File
├── New Project…
├── Open Project…
├── Recent Projects ▸
├── Open Bundle…
├── ─── (separator)
├── Import ▸
│   ├── Praat TextGrid…
│   └── ELAN .eaf…
├── Export ▸                 [greyed unless a bundle is selected]
│   ├── Praat TextGrid…
│   └── ELAN .eaf…
├── Record from microphone…
├── Show project folder
├── ─── (separator)
└── Quit
```

### Confirmed H1 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Engine surface for delete | **New `Project::delete_bundle` with explicit topological cascade** | SQLite's `ON DELETE CASCADE` would need a migration touching every FK; explicit cascade in one method is simpler |
| Audio file deletion | **Best-effort `std::fs::remove_file`; corpus stays consistent on filesystem failure** | The DB is the source of truth; an orphan WAV is recoverable, an orphan DB row isn't |
| Confirmation on delete | **Yes — modal Window with "Delete <name>?" + Cancel** | Hard-to-undo action; user typo protection |
| TextGrid / EAF I/O dialog | **rfd file picker — open for import, save for export** | Same as the existing "Open Bundle…" path |
| Live-recording device picker | **ComboBox listing cpal input devices; "default" first** | E1's API surface already exposes default + named devices |
| Live-recording format | **Bundle's sample rate / channels picked from default device config** | Auto-detect for v1; explicit controls land in 0.3 polish |
| Meter source | **Existing `MeterFrame` on `LiveResults`** | E1 already emits peak/RMS per chunk |
| Status during recording | **Continuous repaint via `ctx.request_repaint()`** | Same pattern as C5 playback |
| Show-in-file-manager | **Platform-specific cmd wrapped in cfg branches** | xdg-open / open / explorer.exe |
| Bundle context menu trigger | **Right-click on bundle row** | Universal convention |

### Layout

- `crates/engine/src/corpus.rs` — `Project::delete_bundle`.
- `crates/engine/tests/sparse_annotations.rs` — round-trip add → delete cascade tests.
- `crates/python/src/lib.rs` — `PyProject.delete_bundle` binding (the **three-surface** principle: every new user-facing engine method ships engine + Python + GUI in the same slice).
- `python/tests/test_corpus.py` — pytest coverage for `delete_bundle` (cascade + idempotent-on-missing-id).
- `python/sadda/_native/__init__.pyi` — type stub for the new method.
- `crates/app/src/main.rs` — file-menu reshuffle; import/export menu items; show-project-folder; bundle context menu; show-in-file-manager helper; **recording modal** (inline, not a new file — the state machine fit cleanly alongside the existing modal-window infrastructure).

### Lossiness / what H1 deliberately doesn't ship

- **Export audio as a separate command.** The WAV is at `signals/original/<name>.wav`; "Show in file manager" gets users there. Bundle-as-zip / re-encode-to-FLAC is a 0.3 polish.
- **Drag-and-drop file import.** Adds when a real user complains.
- **Bulk import** (folder of TextGrids). Same.
- **Rename bundle.** Engine doesn't have an update method for bundle name; this would need a small engine extension. Deferred.
- **Bundle metadata editor** (session / speaker FKs). Deferred.
- **Open recipe `.py` from the GUI / replay button.** Polish.
- **Recording: monitoring during recording** (hear yourself with low latency). Same.
- **Recording: pre-roll buffer / pause/resume.** Same — E1's DEVLOG entry already deferred these.

### What this entry doesn't decide

- **Whether import should auto-create the tier(s)** the imported file references, or only link annotations to existing tiers. The existing `Project::import_textgrid` / `import_eaf` semantics create tiers; we just expose them via the menu.
- **Whether export should default to the project's `exports/` directory.** Likely yes; settled in code.
- **Visual treatment of the recording meter** (linear vs log bars). Whatever looks reasonable.

### Sources / references

- 2026-05-23 Phase 2 slicing entry (the eleven slices this polishes)
- 2026-05-22 D1 entry (TextGrid I/O reused here)
- 2026-05-22 D2 entry (EAF I/O reused here)
- 2026-05-22 E1 entry (LiveSession reused for the recording modal)
- 2026-05-23 G11 entry (release binaries the polish ships in)

---

## 2026-05-23 — 0.2 release binaries (G11): app-release.yml builds + uploads unsigned macOS arm64 / Linux x86_64 / Windows x86_64 binaries on tag push

Goal: settle the eleventh and final Phase 2 slice. G11 stands up the CI workflow that turns a `v0.2.x` git tag into a GitHub Release with desktop binaries attached. Unsigned per the Phase 2 slicing entry — Apple Developer / EV-cert spend lands in Phase 7.

### What G11 must deliver

From the Phase 2 slicing entry: "GitHub Actions workflow building **unsigned** desktop binaries on tag push for macOS arm64 / Linux x86_64 / Windows x86_64; upload as GitHub Release artifacts; update README with download links."

### Tag scheme

The 0.1.x Python releases use `v0.1.0` / `v0.1.1` tags consumed by `release.yml`. To avoid double-firing every workflow on every tag, **the app workflow uses a separate tag prefix**: `v0.2.0-app`, `v0.2.1-app`, etc. (or in general `v*.*.*-app`). Python releases keep `v*.*.*`. Both workflows have explicit `on.push.tags` filters; neither runs on the other's tags.

Alternative: gate by tag content. Cleaner shape: separate tag namespace. The DEVLOG entry picks the separation.

### Matrix

Three runners, three OS triples:

| Runner image | Target triple | Output name |
|---|---|---|
| `ubuntu-latest` | `x86_64-unknown-linux-gnu` | `sadda-app-linux-x86_64` |
| `macos-14` (arm64) | `aarch64-apple-darwin` | `sadda-app-macos-arm64` |
| `windows-latest` | `x86_64-pc-windows-msvc` | `sadda-app-windows-x86_64.exe` |

cibuildwheel is **not** used — that's wheel-specific. Plain `cargo build --release --bin sadda-app` on each runner.

### Per-platform native deps

- **Linux**: ALSA dev headers (`libasound2-dev`) for cpal, libpython for the script-engine. The Ubuntu CI image ships both via apt; we install them explicitly. Plus the `LD_LIBRARY_PATH` workaround for libpython at runtime — for the *user*'s machine, we instead set `LD_RUN_PATH` at link time so the binary records its libpython search path, AND we bundle the OS's libpython location in the README's "what you need installed" note.
- **macOS**: CoreAudio is a system framework; no install needed. libpython ships with system Python; no install needed.
- **Windows**: WASAPI is system; no install. libpython needs to be findable at runtime — Python is usually on the PATH on Windows dev boxes; document the "install Python 3.11 from python.org" requirement in the release notes.

The libpython runtime requirement is awkward for end users — the cleanest fix is the Phase-7 packaging-with-embedded-libpython work the milestone plan calls out. v0.2 ships "needs a Python install on PATH" as a known limitation in the release notes; the script panel + `sadda.app` work without Python only if you also install a matching libpython.

Actually — for 0.2 simplicity, document it as: "requires Python 3.11 or 3.12 installed on the system; if you don't have one, the script panel won't work but everything else will."

### Release flow

1. User bumps `workspace.package.version` in Cargo.toml + commits.
2. User tags `git tag v0.2.0-app` and pushes.
3. CI builds three binaries in parallel; each uploads its single binary as a job artifact.
4. A final `publish` job downloads all three artifacts, creates the GitHub Release, attaches the binaries.

Trusted publishing isn't relevant — these are GitHub Release artifacts, not PyPI uploads. No secrets needed; `GITHUB_TOKEN` (always present in Actions) is sufficient.

### Confirmed G11 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Tag scheme | **`v*.*.*-app` for app releases; `v*.*.*` continues to mean Python wheels** | Avoids both workflows firing on every tag |
| Build tool | **Plain `cargo build --release --bin sadda-app`** | cibuildwheel is wheel-specific; not relevant |
| Signing | **No signing at v0.2** | Phase-7 work per the milestone plan |
| Binary naming | **`sadda-app-<os>-<arch>[.exe]`** | Standard release-artifact convention |
| Linux ALSA deps | **`apt install libasound2-dev`** | Same as the Python wheel workflow |
| Python runtime | **Document as user requirement in release notes** | Bundling libpython is Phase-7 packaging work |
| Permissions | **`contents: write`** so the publish job can create releases | Minimum scope |
| Failure mode | **fail-fast: false** on the matrix | Want partial-success (e.g. macOS works even if Windows breaks) |
| Workflow_dispatch trigger | **Yes** | Dry-run pattern: builds + uploads to Actions artefacts but skips the publish job (gated on `github.event_name == 'push'`) |
| README link | **Single "Latest desktop binaries" link to the GitHub Releases page** | Doesn't go stale; users browse the page for their OS |

### Layout

- `.github/workflows/app-release.yml` (new) — matrix build, artifact upload, publish job.
- `README.md` — add a brief "Desktop app" section pointing at the GitHub Releases page.

### Lossiness / what G11 deliberately doesn't ship

- **Code signing.** macOS gatekeeper warning. Phase 7.
- **Auto-update mechanism inside the app** — release-page download only.
- **Installer packages** (DMG, MSI, AppImage, .deb). Tarball / zip / bare-exe at v0.2.
- **Bundled libpython.** Users must have Python installed for the script panel to work.
- **Per-tag release notes generation.** Empty notes / user-edited via the Releases UI.
- **Smoke-test on the built binary in CI.** GUI binaries are awkward to smoke-test headlessly; the test suite covers everything reachable from non-GUI code.

### What this entry doesn't decide

- **Whether to also build for Linux aarch64 / macOS x86_64.** Same matrix decision as the Python wheel release; can layer in later.
- **Whether to ship a tarball of the asset alongside the bare binary.** Bare binary for v0.2; tarball is a nice-to-have polish.

### Sources / references

- 2026-05-23 Phase 2 slicing entry (G11 row)
- 2026-05-22 G12 entry (Python wheel release workflow — symmetric pattern this slice extends)
- 2026-05-18 milestone-plan entry (Phase-7 signing + packaging cut-line)
- GitHub Actions release-uploading reference: <https://docs.github.com/en/actions/publishing-packages/publishing-docker-images>
- `cargo build --release` cross-platform reference: standard

---

## 2026-05-23 — Single-writer lock on corpus.db (F10): `.sadda-lock` advisory file, PID + hostname, released on Project drop

Goal: settle the tenth Phase 2 slice. F10 prevents two processes (two app instances, or app + a Python script) from writing to the same `corpus.db` and corrupting the audit trail.

### What F10 must deliver

From the Phase 2 slicing entry: "Two `corpus.db` writers running simultaneously (app + script, or two app instances) would corrupt the audit trail. Use SQLite's WAL + a `BEGIN EXCLUSIVE` advisory pattern, OR a `<project>/.sadda-lock` lockfile written at `Project::open`. Refuses to open a project that's locked, with a clear message naming the holder's PID."

### Mechanism choice

The slicing entry left the SQL-vs-file lock choice open. Decision: **filesystem lockfile**. Reasons:

- A SQLite-level `BEGIN EXCLUSIVE` only protects the duration of the transaction; the GUI holds the connection open continuously, but doesn't keep an exclusive transaction running — that would block every other reader, including the Python wrapper that opens the same `corpus.db` read-only via `pl.scan_parquet`-adjacent code paths.
- A file lock survives crashes weirdly (some OSes hold flock past process death; others release). An advisory lockfile with the holder's PID + hostname is **self-documenting**: the error message can name *which process* (or which hostname) holds the lock, and stale-lock recovery is a simple manual `rm .sadda-lock` for the user.
- We've already established `Project::is_project_root` checks lightweight on-disk markers; this adds one more.

### Lockfile shape

`<project_root>/.sadda-lock`, written at `Project::open` and `Project::create` time, containing:

```toml
pid = 12345
hostname = "alice-laptop"
acquired_at = "2026-05-23T15:42:11.789Z"
```

On `Project::open`:

1. If `.sadda-lock` doesn't exist, create it with our `(pid, hostname, now)` and proceed.
2. If it exists and the recorded PID is **our own PID**, take ownership (the previous open in this process leaked; not a real conflict).
3. If it exists and the PID belongs to a **live process on this host**, refuse to open with a clear `EngineError::ProjectLocked { holder_pid, hostname }`.
4. If it exists but the PID is dead (or on a different host), the lock is stale — write a fresh lockfile and proceed. (Cross-host detection: trust the recorded hostname; if it matches ours, check PID liveness via `kill(pid, 0)` on Unix or `OpenProcess` on Windows. If hostname differs, treat as stale-with-warning to the operator since we can't verify.)

On `Project::drop`: best-effort delete the lockfile. If the process is killed (SIGKILL, panic, etc.) the file persists and the next open detects + cleans the stale lock per step 4.

### Engine surface

- New `EngineError::ProjectLocked { holder_pid: u32, hostname: String, lockfile_path: PathBuf }` variant.
- `Project::open` and `Project::create` acquire the lock as part of their existing flow.
- `Project::drop` releases.
- New `Project::is_locked_by(path) -> Option<(pid, hostname)>` helper for the GUI's welcome-screen recent-projects list (grey-out projects currently held by another process).

### Cross-process detection

Use the `sysinfo` crate? Or libc directly? Libc on Unix is `kill(pid, 0)` returning 0 (process exists) or `ESRCH` (doesn't). Windows is `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, ...)`. Both are ~10 lines of platform-specific code with `#[cfg(unix)]` / `#[cfg(windows)]`. No new dep needed; the existing engine deps don't include sysinfo.

### Confirmed F10 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Mechanism | **Filesystem lockfile** | Survives crashes self-documentingly; doesn't block read-only paths |
| Filename | **`.sadda-lock` in the project root** | Hidden (UNIX); doesn't crowd the project dir listing |
| Contents | **TOML with pid + hostname + acquired_at** | Human-readable; matches our existing `project.toml` |
| Stale detection | **Per-host PID liveness check (kill(0) / OpenProcess)** | Standard pattern; no new dep |
| Cross-host policy | **Trust + warn — treat as stale, take over** | Without a network ping we can't verify; warning to the operator is enough |
| Same-PID re-open | **Take ownership silently** | Lets the GUI re-open a project it leaked previously |
| Release | **Best-effort delete on `Project::drop`** | Crash leaves a stale file; next open cleans it |
| GUI integration | **Welcome-screen Recent rows show `(locked by PID N)` next to held projects** | Surfaces conflict before the user clicks |
| Python wrapper | **Goes through `Project::open` too — picks up the lock identically** | One unified path; no special read-only mode for the Python side at v1 |
| Error type | **New `EngineError::ProjectLocked` variant** | Distinguishable from generic `Corpus` errors so the GUI can render a specific message |

### Layout

- `crates/engine/src/error.rs` — `EngineError::ProjectLocked { holder_pid, hostname, lockfile_path }`.
- `crates/engine/src/corpus.rs` — `acquire_lock` / `release_lock` helpers; integration into `Project::open` / `Project::create` / `Project::drop`. PID-liveness helper with `cfg(unix)` / `cfg(windows)` branches.
- `crates/engine/tests/migrations.rs` (or a new `lock.rs`) — integration tests: acquire OK, double-acquire fails, drop releases, stale lock on a non-existent PID is cleared.

### Lossiness / what F10 deliberately doesn't ship

- **Multi-writer with conflict resolution.** Single-writer is the v1 promise. Multi-writer needs operational transformation or a real CRDT; way out of scope.
- **Read-only mode.** A read-only `Project::open_readonly` that doesn't take the lock could come later for "browse but don't edit" workflows.
- **Network-aware locking.** No filesystem-network detection; SMB / NFS shared projects work but cross-host crash recovery is operator-judgement.
- **Lock timeout / auto-release after N minutes.** No timer; users explicitly `rm .sadda-lock` to break a stale lock if the PID-liveness check misjudges.
- **Wrapping every PyO3 wrapper to surface the lock error specifically.** The error bubbles up as `EngineError::ProjectLocked` and lands in the Python `RuntimeError` text; UX polish for a typed exception is a 0.3 item.

### What this entry doesn't decide

- **Whether to log to stderr when acquiring a lock that was stale.** Probably yes; trivial to add.
- **Exact wording of the `(locked by PID N)` row decoration on the welcome screen.** Settled in implementation.

### Sources / references

- 2026-05-23 Phase 2 slicing entry (F10 row)
- 2026-05-23 A1 entry (`Project::is_project_root` helper that F10 mirrors)
- SQLite locking modes reference: <https://www.sqlite.org/lockingv3.html>

---

## 2026-05-23 — `sadda.app` in-app namespace (E9): append_to_inittab built-in module, thread-local snapshot, command palette via Ctrl+P

Goal: settle the ninth Phase 2 slice. E9 turns the script panel from a "run pure Python" toy into the in-app scripting host the 2026-05-18 API-surface entry sketched: scripts read the GUI's current selection / active bundle / cursor, and register commands that show up in a palette.

### What E9 must deliver

From the Phase 2 slicing entry: "`sadda.app` API: current_selection, active_bundle, register_command. Lands the in-app `sadda.app` namespace per the 2026-05-18 API-surface entry. `register_command` adds the user's function to a command palette accessible via Ctrl+P or a menu. Scripts that don't touch `sadda.app` work identically whether run inside the app or via `sadda` from a terminal."

### Two design problems

E9 needs to settle two genuinely-hard problems the prior slices kicked down the road:

**(1) Embed packaging — how does `import sadda.app` work without a pip install?** The embedded interpreter's `sys.path` defaults to the system Python's site-packages. If `sadda` isn't pip-installed there (and on a typical user's machine, it isn't), `import sadda.app` fails. E9's resolution: **register a built-in `sadda.app` module via PyO3's `append_to_inittab!`** before the interpreter starts. No filesystem `sadda` package required — `import sadda.app` resolves to the Rust-implemented module baked into the binary.

The full `sadda` Python package (with stability decorators, DSP wrappers, etc.) is **NOT bundled in the embed**. Users who want it `pip install sadda` separately into whatever Python the embed picked up. The slicing-entry risk-spike note flagged Linux/Windows embed-packaging as deferred to a follow-up; E9 confirms the minimal-namespace approach is the v0.2 answer.

**(2) State plumbing — how do PyO3 functions read SaddaApp's GUI state?** The functions in `sadda.app` need to return things owned by `SaddaApp` (selected annotation, active bundle, cursor). The standard solution for a single-threaded GUI process: **a thread-local pointer to an `AppSnapshot` struct, set immediately before `run_script` and cleared after**. The GIL guarantees the snapshot is valid for the script's lifetime (the GUI thread can't mutate `SaddaApp` while running Python on the same thread).

### `sadda.app` surface at E9

```python
import sadda.app as app

app.project_root()       # → str, project root directory
app.active_bundle()      # → dict | None: {id, name, sample_rate, duration_seconds}
app.current_selection()  # → dict | None: {kind, tier_id, annotation_id}
app.cursor_seconds()     # → float, current playback cursor

app.register_command("My Action", my_func)
# my_func is invoked with no args when the user picks it from
# the Ctrl+P palette
```

Each function with no snapshot active raises `RuntimeError("sadda.app called outside an active app session")`. In practice that's only possible if someone tries to import `sadda.app` from a script run via the standalone sadda Python package — fine; the function names exist, the call surface is consistent.

### Snapshot design

```rust
struct AppSnapshot {
    project_root: PathBuf,
    bundle: Option<BundleInfo>,
    selection: Option<AnnotationSelection>,
    cursor: f64,
}

thread_local! {
    static APP_SNAPSHOT: Cell<Option<NonNull<AppSnapshot>>> = const { Cell::new(None) };
}

fn with_snapshot<R>(f: impl FnOnce(&AppSnapshot) -> R) -> Option<R> {
    APP_SNAPSHOT.with(|cell| cell.get().map(|p| f(unsafe { p.as_ref() })))
}
```

The unsafe is fenced by:
- Pointer set in `Cell` is non-null only between `run_script` setup and teardown.
- The snapshot lives in a stack local of the caller (`run_script_with_snapshot`); it outlives the entire script run by construction.
- GUI thread holds GIL during the run, so the snapshot can't race with any other thread that might want it.

### Command palette

`register_command(name, callable)` appends `(name, callable)` to a `Vec<(String, Py<PyAny>)>` on `SaddaApp`. The PyObject's refcount keeps the callable alive past the script run (it's still owned by Python — egui doesn't garbage-collect it).

**Ctrl+P** (Cmd+P on macOS) opens a modal `Window` with:
- Search input at the top
- Filterable list below (substring match on command name)
- Enter or click on a row invokes the callable

When invoked, the app:
1. Locates the `Py<PyAny>` in `registered_commands`
2. Sets up the snapshot
3. Acquires the GIL
4. Calls `cb.call0(py)` — the script's PyO3 callable runs synchronously, GUI freezes for the duration
5. Tears down

If the callable raises, the exception's repr lands in `script_error` and the script panel's output pane shows it.

Commands persist for the **app session**. After a relaunch, the user re-runs their registration script to repopulate.

### Confirmed E9 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Embed packaging | **`append_to_inittab!` for `sadda.app` only — minimal namespace** | Works without pip install; sidesteps the full-wheel bundling complexity; the 2026-05-18 API-surface entry pinned `sadda.app` as the only in-app-specific module |
| Full `sadda` package availability | **NOT bundled — users pip install for the wider engine API** | Bundling is a Phase-7 polish item per the milestone plan |
| State plumbing | **Thread-local pointer to a stack-local snapshot** | GUI thread guarantees serialise; standard pattern; no Arc / Mutex overhead |
| `sadda.app.active_project` from the slicing entry | **Renamed to `project_root()`** — returns a path string | A full Project PyObject would need to thread PyProject from the python crate; out of scope for the minimal namespace |
| Command palette | **Modal Window opened by Ctrl/Cmd+P** | VS Code / IntelliJ convention; matches the existing label-edit modal pattern |
| Command persistence | **App-session only; users re-run registration scripts** | Auto-running stored code on startup is hazardous |
| Command invocation | **Synchronous on the GUI thread with snapshot set** | Same blocking semantics as `run_script` |
| Snapshot timing | **Set before run_script / command invocation; cleared in a Drop guard so panics still clear it** | Defensive; an unwinding script must not leave a dangling pointer |
| Snapshot contents | **project_root, active_bundle, selection, cursor** | The four pieces every script demo wants |

### Layout

- `crates/app/Cargo.toml` — adds `pyo3 = { version = "0.28", features = ["auto-initialize"] }`. (script-engine already brings pyo3 in transitively; making it a direct dep is necessary for `append_to_inittab!` to be in scope.)
- `crates/app/src/sadda_app.rs` (new) — `AppSnapshot`, thread-local cell, `with_snapshot`, the `#[pymodule] fn sadda(...)` registering the `app` submodule, all functions.
- `crates/app/src/main.rs` — `append_to_inittab!` call in `main` before `eframe::run_native`; `registered_commands` on `SaddaApp`; Ctrl+P handler; command-palette modal; `run_script_buffer` switches to `run_script_with_snapshot` that sets up the cell before calling into `script-engine`.

### Lossiness / what E9 deliberately doesn't ship

- **Bundling the full `sadda` Python wheel into the binary.** Phase-7 packaging work; large enough to be its own slice cluster.
- **Calling engine mutators (`add_interval` etc.) from `sadda.app`.** Doable but requires threading a `PyProject`-equivalent through the snapshot; deferred to keep the v1 surface focused on read-side introspection.
- **`current_selection_info` returning the full annotation data** (label, time span, etc.). E9 returns the identifying triple `(kind, tier_id, annotation_id)`; scripts that want more open the project externally via `sadda.open_project(root)` and look up.
- **Command argument support.** `register_command(name, callable)` calls `cb.call0(py)`. Commands with args wait for a polish slice.
- **Command persistence across launches.** Same reason.
- **`sadda.app.refresh()`** — the API-surface entry mentioned this; egui repaints reactively, so it's a no-op for now. Stub for forward-compat if real users need it.
- **`sadda.app.register_panel` for arbitrary widgets.** Same cut-line as the milestone-plan's "register_command only at 1.0."
- **Cancellable / async command execution.** Same as E8.

### What this entry doesn't decide

- **Exact Ctrl+P modifier per platform.** Egui's `Modifiers::COMMAND` handles Ctrl on Linux/Windows and Cmd on macOS automatically.
- **Whether the palette should fuzzy-match or substring-match.** Substring at v1; fuzzy is polish.
- **What to do if the user registers two commands with the same name.** Just keep both; first-match wins on Ctrl+P selection.
- **How big the snapshot can get.** Currently tiny; if it grows (e.g., a thumbnail of the waveform), revisit.

### Sources / references

- 2026-05-23 Phase 2 slicing entry (E9 row + the embed-packaging spike note)
- 2026-05-23 E8 entry (the script panel this slice extends)
- 2026-05-18 Python API surface entry (`sadda.app` namespace shape)
- 2026-05-23 A1 entry (window state pattern this slice's snapshot mirrors)
- PyO3 `append_to_inittab!`: <https://docs.rs/pyo3/latest/pyo3/macro.append_to_inittab.html>

---

## 2026-05-23 — Embedded CPython script panel (E8): bottom panel reuses Phase-0 script-engine, code persists, output doesn't

Goal: settle the eighth Phase 2 slice. E8 lights up the script-panel UI — the visible half of the embedded-CPython story. The interpreter itself was validated in Phase 0's `crates/script-engine`; E8 just wires it to a panel.

### What E8 must deliver

From the Phase 2 slicing entry: "Reuse Phase 0's `crates/script-engine` for the embed. Egui text editor + Run button + output pane. Scripts can `import sadda` and get the active project handle via `sadda.app.active_project`."

E8 ships the panel + the runner; **`import sadda` and `sadda.app` defer to E9**. The two slices are split because E9 needs to settle a real packaging question (how the embedded interpreter finds the sadda module) that's bigger than the UI work E8 represents.

### Reused infrastructure

`crates/script-engine` from Phase 0 already provides:

```rust
pub fn run_script(code: &str) -> PyResult<ScriptOutput>;
pub struct ScriptOutput { pub stdout: String, pub stderr: String }
```

Each call uses a fresh globals dict; state doesn't persist between runs. `auto-initialize` brings the interpreter up on first call.

### Panel layout

New bottom panel between the tier strip and the status bar, toggleable via **View → Show Script Panel**:

```
┌──────────────────────────────────────────────────────────┐
│ menu                                                     │
├───────────┬──────────────────────────────────────────────┤
│ sidebar   │ waveform                                     │
│           │ ─ spectrogram toolbar ─                      │
│ bundles   │ spectrogram                                  │
│           │ ─ tier strip ─                               │
│           ├──────────────────────────────────────────────┤
│           │ ▸ Script                          [ Run ]    │
│           │                                              │
│           │   import math                                │
│           │   print("hello", math.pi)                    │
│           │                                              │
│           │ ─ output ─                                   │
│           │   hello 3.141592653589793                    │
│           ├──────────────────────────────────────────────┤
│           │ Project: vowels  ·  /home/alice/vowels       │
└───────────┴──────────────────────────────────────────────┘
```

The panel itself splits into top (code editor — multi-line `TextEdit`) and bottom (output — read-only `TextEdit`). Resizable divider between. Defaults: 60/40 (code/output), ~200px tall total.

### Persistence

| What | Persisted? | Where |
|---|---|---|
| Panel open/closed | **Yes** | `PersistedState.script_panel_open` |
| Script source code | **Yes** | `PersistedState.script_buffer` |
| Most recent output | **No** | In-memory only on `SaddaApp` |
| Cursor position / selection inside the editor | **No** | egui's per-frame state |
| Editor / output split ratio | **Implicit** via egui's `Panel::bottom` resize state | Free via the framework |

Persisting the script content matters — users will type non-trivial scripts and re-launch the app; losing them on relaunch is hostile. Persisting output is wasteful (regenerable in <100 ms by clicking Run) and risks bloating the storage blob with stdin dumps over time.

### Run model

- **Run button** (top-right of the script panel header) and **Ctrl/Cmd+Enter** keyboard shortcut both invoke `run_script(self.script_buffer)`.
- Errors from the interpreter (`PyErr`) surface in the output pane formatted as `stderr` would: red colour, no special panel.
- Execution is **blocking** — the GUI freezes for the duration of the script. That's a v1 simplification; long-running scripts are user responsibility. Real-time progress + cancellation is a 0.3 polish item.

### What scripts can't do at E8 (defers to E9)

- **`import sadda`** — the embedded interpreter's `sys.path` doesn't necessarily contain the sadda module (it depends on how the binary was launched; pyo3's auto-initialize uses the same Python the script-engine crate compiled against, which may or may not have `sadda` installed). E9 adds the wiring + a fallback to inject `sadda` into the namespace.
- **`sadda.app.current_selection / active_bundle / register_command`** — entire `sadda.app` namespace lands in E9.
- **Calling Rust-side state directly** (e.g., reading the current `timeline.cursor`) — same; E9 owns that bridge.

Pure-Python scripts work at E8: arithmetic, stdlib calls, `print` / `import math` / `import json`, etc. Enough to verify the embed is live and to play with the interpreter.

### Confirmed E8 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Panel placement | **Bottom panel, between tier strip and status bar** | Doesn't crowd the editor; resizable; toggleable from View |
| Toggle | **View → Show Script Panel** | Discoverable; standard menu placement |
| Default state | **Closed** | Avoids surprising new users; egui storage preserves the previous session's choice |
| Inner layout | **Top: editor; bottom: output. Resizable divider** | Matches most IDE script consoles |
| Code persistence | **Yes (in `PersistedState`)** | Losing typed scripts on relaunch is hostile |
| Output persistence | **No** | Cheap to regenerate; persisting risks unbounded growth |
| Run keybindings | **Button + Ctrl/Cmd+Enter** | Universal IDE convention for "run buffer" |
| Execution model | **Blocking; GUI freezes during run** | v1 simplification; cancellation is polish |
| `import sadda` access | **Deferred to E9** | Needs packaging spike + the `sadda.app` namespace; both are E9's job |
| Output format | **Plain text; stderr in red** | Matches the existing error banner colour |
| Script size cap | **None** | Egui handles multi-KB strings fine; users who want sanity will write to a file and `exec(open(...).read())` |

### State changes on `SaddaApp`

- **New `PersistedState` fields**: `script_panel_open: bool`, `script_buffer: String`.
- **New in-memory fields**: `script_output: Option<ScriptOutput>` (last-run result; `None` if the user hasn't pressed Run since launch).

### Layout

- `crates/app/Cargo.toml` — adds `sadda-script-engine = { path = "../script-engine" }`.
- `crates/app/src/state.rs` — `PersistedState` gets `script_panel_open` + `script_buffer` (both `#[serde(default)]`).
- `crates/app/src/main.rs` — `view_menu` gets a `Show Script Panel` checkbox; new `script_panel` method renders the bottom panel; `Ctrl/Cmd+Enter` keyboard handler when the panel is open.

### Lossiness / what E8 deliberately doesn't ship

- **`import sadda` + `sadda.app` namespace.** E9.
- **Async / cancellable execution.** Polish slice. The GUI freezes during long runs in v1.
- **Multiple script tabs / sessions.** Single buffer.
- **Syntax highlighting.** Plain `TextEdit`; a syntax-highlighter widget can layer in via `egui_extras` or a custom layouter; polish.
- **File open / save** for scripts. v1 users `Ctrl+A → copy → paste into their editor`. Sadda's broader recipe system (F1) is the persistent-script story.
- **Script history / "Run previous" navigation.** Not in scope.
- **Output buffer limit.** v1 stores whatever the script printed; very chatty scripts can fill memory. Cap can layer in if a real complaint surfaces.
- **REPL semantics** (state persists between runs). Each run starts fresh per `script-engine`'s contract.
- **Locale / encoding controls** for `sys.stdout`. UTF-8 everywhere; no per-script override.

### What this entry doesn't decide

- **Whether the script panel should also appear in the welcome screen state (no project loaded).** Probably no — there's nothing useful to do without a project. Settled in implementation; greys out when no project.
- **Visual prominence of the Run button.** Default-styled for v1; can promote to coloured button later.
- **Whether Ctrl+Enter or Cmd+Enter on macOS specifically.** Egui's modifier handling is platform-aware; using `Modifiers::COMMAND` covers both.

### Sources / references

- 2026-05-23 Phase 2 slicing entry (E8 row)
- 2026-05-23 A1 entry (`PersistedState` shape this slice extends)
- Phase 0 `crates/script-engine/README.md` (the reuse target)
- 2026-05-18 Python API surface entry (`sadda.app` design — settled in E9)

---

## 2026-05-23 — Point-tier editing (D7): click-to-add overrides cursor positioning, drag-to-move, shared label modal

Goal: settle the seventh Phase 2 slice. D7 brings the tier strip's point lanes to feature-parity with D6's interval lanes: add, move, edit-label, delete. Lighter than D6 (points are zero-width events; no boundary-drag) but with one design twist where it bumps into B4's click-to-position semantics.

### What D7 must deliver

From the Phase 2 slicing entry: "Click to add a point; drag to move; delete to remove. Same engine-surface extension pattern (`update_point / delete_point`)."

### Engine extension

Surface-symmetric with D6's interval methods:

```rust
impl Project {
    pub fn update_point(&self, id: i64, spec: &PointSpec) -> Result<()>;
    pub fn delete_point(&self, id: i64) -> Result<()>;
}
```

Same conventions:

- `update_point` replaces `(time_seconds, label, extra, parent_annotation_id)`. Rejects `tier_id` changes. Re-validates cardinality.
- `delete_point` is idempotent — no error on a missing id.
- Both fire the V3 audit triggers automatically.

### The click-to-add vs. click-to-position conflict

B4 shipped click-on-empty-lane → position cursor + deselect. D7's "click to add a point" would collide with that *if applied uniformly*. Resolution: **point lanes override**. Two disjoint behaviours by tier type:

| Click target | Interval lane | Point lane |
|---|---|---|
| Empty space | Position cursor + deselect (B4 / C5) | **Add new point at click time (D7)** |
| Existing annotation body | Select + position cursor at start | Select + position cursor at point time |
| Within boundary hit zone | Start drag-resize (D6) | n/a — points are zero-width |
| Drag-and-release on annotation | Drag boundary → resize (D6) | **Drag point → move (D7)** |

This matches Praat — `TextGrid` editor adds a point on PointTier click. Users who want the cursor moved in a point lane can click an existing point (which moves cursor to that point's time per C5).

### Drag-to-move state

A new `DraftEdit` variant:

```rust
enum DraftEdit {
    // ... existing variants
    MovingPoint {
        tier_id: i64,
        annotation_id: i64,
        original_time: f64,
        current_time: f64,
    },
}
```

Mouse-down within ~6px of a point's tick → start `MovingPoint`. Drag updates `current_time`; the rendered tick follows the pointer. Mouse-up commits via `update_point` (or drops the draft if the time hasn't actually changed).

### LabelEdit refactor

D6's `LabelEdit` stored `base: sadda_engine::Interval` — couldn't fit a point's data. D7 lifts it to a polymorphic struct that re-fetches the base row at commit time:

```rust
struct LabelEdit {
    tier_id: i64,
    annotation_id: i64,
    kind: LabelEditKind,    // Interval | Point
    text: String,
    just_started: bool,
}
```

The modal Window UI is unchanged (TextEdit + Save / Cancel + Enter / Escape). Commit-time logic branches on `kind` to call `update_interval` or `update_point` with the re-fetched base row's other fields preserved.

### Delete key

D6's `delete_selected_annotation` already had a `match self.selected_annotation` shape; D7 fills in the `AnnotationSelection::Point` arm to call `delete_point`. No new keybinding.

### Confirmed D7 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Engine update API | **`update_point(id, &PointSpec)`, replace-all** | Surface-symmetric with `update_interval`; same audit-trigger free-lunch |
| Engine delete API | **`delete_point(id)`, idempotent** | Matches `delete_interval`; user mental model |
| Click conflict resolution | **Point lanes override empty-click = add; cursor positioning still happens on click-on-existing-point** | Matches Praat; the "I clicked on a point lane to do something" interpretation is unambiguously "add a point" |
| Point hit zone for drag | **6 pixels** | Same as D6 boundary; same rationale |
| Minimum drag-to-move | **None — any drag commits** | Distinct from drag-to-create (which has a 5ms floor); moving by ε is still a deliberate "I touched it" action |
| No-op move detection | **If `\|current - original\| < 1ms`, drop the draft silently** | Prevents writing an audit row for a noise-level mouse jitter |
| Label edit | **Reuse D6's modal Window** | Already proven; inline overlay is still a polish item |
| Delete key | **Same Delete/Backspace handler from D6, with the Point arm filled in** | Single keybinding for both tier types |
| Audit log | **Free via V3 triggers** | Same as D6 |

### State changes

- `DraftEdit::MovingPoint` variant added.
- `LabelEdit` refactored: drops `base: Interval`; adds `kind: LabelEditKind`.
- `delete_selected_annotation` extended with the `Point` arm.
- `commit_draft_edit` extended with the `MovingPoint` arm.

### Layout

- `crates/engine/src/corpus.rs` — `Project::update_point` + `Project::delete_point`.
- `crates/engine/tests/sparse_annotations.rs` — 5 new tests covering update / tier-change reject / round-trip move / idempotent delete / audit-log trail.
- `crates/app/src/main.rs` — `DraftEdit::MovingPoint` + `LabelEdit` refactor + `render_point_lane` rewrite (boundary-style hit detection for grab; click-add for empty space; double-click label) + `commit_draft_edit` MovingPoint arm + `delete_selected_annotation` Point arm.

### Lossiness / what D7 deliberately doesn't ship

- **Snap-to-cursor on drag-move.** Polish slice; same reasoning as D6.
- **Insert-at-cursor keyboard shortcut.** A "press P to add a point at the cursor" binding could ship later; mouse is enough at v1.
- **Multi-point select + bulk delete.** Single-select only.
- **Drag with shift to constrain to nearest tick** (e.g., zero-crossing snap). Polish.
- **Reference-tier editing.** Reference lanes still show their caption; their visualisation has to come first.
- **Undo / redo.** Same answer as D6 — the audit log captures the data for a future implementation.

### What this entry doesn't decide

- **Whether moving a point should preserve the cursor's position relative to the point** (e.g., follow with the cursor). Settled in implementation: cursor stays where it was unless the user clicks the new location.
- **Visual affordance for "drag me"** (cursor change on hover near a point tick). Polish.
- **Whether duplicate points at the same time are allowed** — engine doesn't reject them; UX doesn't prevent them.

### Sources / references

- 2026-05-23 Phase 2 slicing entry (D7 row)
- 2026-05-23 D6 entry (the surface-symmetric pattern this slice extends to point lanes)
- 2026-05-23 B4 entry (the click-to-position behaviour D7 overrides in point lanes only)
- 2026-05-23 C5 entry (cursor positioning conventions)
- Praat PointTier editor reference: <https://www.fon.hum.uva.nl/praat/manual/PointTier.html>

---

## 2026-05-23 — Interval-tier editing (D6): mouse state machine, drag-create, drag-resize, double-click label, delete key

Goal: settle the sixth Phase 2 slice. D6 turns the tier-strip pane from a read-only viewer into an interval editor. Two engine methods land alongside the GUI work.

### What D6 must deliver

From the Phase 2 slicing entry: "Drag in empty space to create a new interval; drag a boundary to resize; double-click to edit label inline; delete key removes the selected interval. Writes via `Project::add_interval`; engine API extended with `update_interval / delete_interval`."

### Engine extension

Two new methods on `Project`:

```rust
impl Project {
    pub fn update_interval(&self, id: i64, spec: &IntervalSpec) -> Result<()>;
    pub fn delete_interval(&self, id: i64) -> Result<()>;
}
```

- **`update_interval`**: replace-all semantics on `(start_seconds, end_seconds, label, extra, parent_annotation_id)`. Rejects an attempt to change `tier_id` (would corrupt the parent-tier relationship). Cardinality re-validation runs on the new `parent_annotation_id`. The existing `annotation_interval_audit_update` trigger fires automatically — the audit trail records the before/after JSON without any caller cooperation.
- **`delete_interval`**: simple `DELETE FROM annotation_interval WHERE id = ?`. The audit trigger captures the row before deletion. Returns `Ok(())` even when no row matched — idempotent, matches what users expect from a "remove" action.

Surface-symmetric with the existing `add_interval`; same error model.

### App mouse state machine

The tier-strip interval lane handles four mouse interactions; D6's job is the state machine that disambiguates them:

```rust
enum DraftEdit {
    None,
    Creating  { tier_id: i64, start_time: f64, current_time: f64 },
    Resizing  { tier_id: i64, annotation_id: i64, edge: BoundaryEdge,
                fixed_time: f64, current_time: f64 },
}
enum BoundaryEdge { Start, End }
```

The dispatch on mouse-down inside an interval lane:

| Pointer location at mouse-down | Action |
|---|---|
| Within ~6px of an interval boundary | Start `Resizing { edge }`; `fixed_time` = the *other* edge |
| Inside an interval's body (not near a boundary) | Click-only handling (B4 select + C5 cursor positioning); no draft starts |
| Empty lane space | Start `Creating { start_time }` |

On `dragged()`: update `current_time` from `pointer_to_time`. The pane render draws a live preview overlay (different colour so it's clearly a draft).

On `drag_stopped()`:

- `Creating` with `|end - start| ≥ 5ms`: commit `Project::add_interval`. Smaller drags are treated as a stray click (no-op for create — the click handler already did its work).
- `Resizing`: commit `Project::update_interval` with the new boundary; if the new range is negative (user dragged past the fixed edge), swap the endpoints first.

### Inline label editing

Double-click an interval → enter label-edit mode:

```rust
struct LabelEdit {
    tier_id: i64,
    annotation_id: i64,
    text: String,        // mutable buffer the TextEdit binds to
    just_started: bool,  // first-frame flag for request_focus
}
```

The pane renders an `egui::TextEdit` over the interval's rectangle, pre-filled with the current label, focus requested on the first frame. Commit on **Enter** (writes via `Project::update_interval`); cancel on **Escape** (drops the buffer).

Click outside the TextEdit also commits — matches the macOS / Praat convention where focus-loss saves the edit.

### Delete key

When `selected_annotation == Some(AnnotationSelection::Interval { … })` and the user presses **Delete** or **Backspace** (and no TextEdit has focus), call `Project::delete_interval(id)` and clear the selection. Same key binding works for points in D7.

### Overlap policy

The engine doesn't reject overlapping intervals — that's existing behaviour, validated by the fact that TextGrid export already handles it (errors at export time per the D1 contract, not at insert time). D6 inherits that: **users can create overlapping intervals**. The boundary-drag resize doesn't snap to neighbouring boundaries either; users are trusted.

If real users complain about accidental overlap, a soft "snap within 5px to nearest boundary" can land as a 0.3 polish slice. Engine-enforced no-overlap is more invasive — it'd need an annotation-tier `non_overlapping` constraint and a schema migration; out of D6 scope.

### Confirmed D6 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Engine update API | **`update_interval(id, &IntervalSpec)`, replace-all** | Smallest surface; cardinality enforcement reused; same shape as `add_interval` |
| Tier-id change | **Rejected** | Moving an interval between tiers is a different operation; out of scope |
| Engine delete API | **`delete_interval(id)`, idempotent** | Matches user mental model; existing audit trigger captures the OLD row |
| Boundary hit zone | **6 pixels** | Standard for interactive editors; doesn't conflict with body click |
| Minimum drag-to-create | **5 milliseconds** | Filters accidental clicks; 5ms at typical zoom is ~5px |
| Negative-range resize | **Swap endpoints; never insert reversed** | The engine CHECK rejects `start >= end` anyway; swap gives the user the intuitive result |
| Label edit trigger | **Double-click on body** | Universal convention; doesn't conflict with single-click select |
| Label commit | **Enter or focus-loss** | macOS / Praat / VS Code convention |
| Label cancel | **Escape** | Universal convention |
| Delete key | **Delete or Backspace** | Both work; mirrors most editors |
| Overlap policy | **Allowed at engine level; no UX prevention** | Inherits from existing engine behaviour; can layer "snap-to-boundary" later |
| Draft preview colour | **Lighter shade of the base fill** | Visually obvious as in-progress; doesn't conflict with selection highlight |
| Cursor positioning during drag | **Doesn't move; B4 cursor logic only fires on plain clicks** | Drag = edit; click = select + cursor. Two disjoint actions |
| Audit log | **Free via the V3 triggers** | No caller cooperation needed |

### State changes on `SaddaApp`

- **New**: `draft_edit: DraftEdit`, `label_edit: Option<LabelEdit>`.
- **Reset** in `clear_bundle_selection` / `select_bundle`: both cleared along with selection.
- Existing `selected_annotation` from B4 stays the source of truth for "what does Delete affect?"

### Layout

- `crates/engine/src/corpus.rs` — `Project::update_interval`, `Project::delete_interval`.
- `crates/engine/tests/sparse_annotations.rs` — extend with update/delete tests.
- `crates/app/src/main.rs` — `DraftEdit`, `LabelEdit`; rewrite of `render_interval_lane` and the lane's mouse-event plumbing; delete-key handler at the app level; TextEdit overlay.

### Lossiness / what D6 deliberately doesn't ship

- **Point-tier editing.** Slice D7.
- **Reference-tier editing.** Reference lanes still show "(reference — N targets)"; their visualisation needs to come first.
- **Dense-tier editing.** No GUI for editing per-frame dense data at v1.
- **Interval move (drag the body to shift)** — drag-resize only at the boundaries. Body-drag-to-move is a polish item.
- **Multi-select.** Single selection only.
- **Undo / redo.** No app-level undo stack at v1; the audit log captures the data needed to reconstruct one in a later slice.
- **Snap-to-boundary, snap-to-cursor, magnetic guides.** All polish items.
- **Insert-into-gap auto-grow** (Audacity behavior). v1 doesn't auto-resize neighbours.
- **Rich-text labels.** Plain UTF-8 only; matches the engine's string column.
- **Toolbar buttons for "Add / Delete"** — keyboard + drag only at v1; menu / palette can come later.
- **Right-click context menus.** Same.

### What this entry doesn't decide

- **Whether label commits on focus-loss should also trigger when the user clicks another interval** — settled in implementation.
- **Whether to show the new interval's draft preview in the waveform / spectrogram too** — only in the lane being edited for now; could extend.
- **Numerical-precision floor for `start_seconds == end_seconds`** — the engine's CHECK already rejects equal; the app's 5ms min-drag is the soft guard.
- **Cursor flash / blink in the label TextEdit** — egui default.

### Sources / references

- 2026-05-23 Phase 2 slicing entry (D6 row)
- 2026-05-23 B4 entry (the tier-strip selection state this slice extends)
- 2026-05-23 C5 entry (the timeline state edit times map through)
- 2026-05-21 B2 entry (`add_interval` + cardinality model these new methods symmetrically extend)
- 2026-05-22 D1 entry (TextGrid export's overlap handling — confirms engine doesn't reject overlaps at insert time)
- Audacity edit-vocabulary reference: <https://manual.audacityteam.org/man/audacity_selection.html>
- Praat label-edit convention: <https://www.fon.hum.uva.nl/praat/manual/Intro_7__Annotation.html>

---

## 2026-05-23 — Synced cursor + zoom + scroll + playback (C5): shared timeline state, mouse-position-centered zoom, cpal output with linear resample, view follows cursor

Goal: settle the fifth Phase 2 slice. C5 is the slice that turns three independent panes into a coordinated editor — they all share one timeline state, scrub in lockstep, and respond to a single transport. It's also the biggest single slice in Phase 2; budget realistically ~1200–1500 LOC.

### What C5 must deliver

From the Phase 2 slicing entry: "Single shared timeline state across the three view panes. Spacebar plays/stops from the cursor via cpal *output* (reusing the existing cpal dep from E1 input). Click moves cursor. Mouse-wheel zooms; shift-wheel scrolls. Acceptance: scrub through a recording with the spectrogram cursor and tier-strip cursor staying in lockstep."

### Timeline state

One struct, owned by `SaddaApp`, plumbed into each pane render:

```rust
pub struct TimelineState {
    /// View window in seconds: [view_start, view_end). Resets on
    /// bundle change to [0, duration_seconds].
    pub view_start: f64,
    pub view_end: f64,
    /// Cursor position in seconds. Clamped to [0, duration].
    pub cursor: f64,
    /// Total bundle duration in seconds; needed for clamping.
    pub duration: f64,
}
```

Three operations the panes call back into:

- `zoom_at(time, factor)` — multiplies the view range by `factor` (e.g. 1/1.2 zoom-in, 1.2 zoom-out), centred on the time the mouse is over.
- `scroll_by(delta_seconds)` — shifts the window; clamps to `[0, duration]` without changing the range size.
- `set_cursor(time)` — clamps + sets.

All of those are pure-data — unit-testable in `state.rs` without spinning up a single egui frame.

### Interaction model

| Event | Result |
|---|---|
| Click on waveform / spectrogram | `set_cursor(click_time)` |
| Click on tier-strip annotation | Select annotation (B4) + `set_cursor(annotation_start)` |
| Click on tier-strip background | Deselect (B4) — cursor unchanged |
| Mouse wheel up/down on any pane | `zoom_at(mouse_time, 1.0/1.2)` / `zoom_at(mouse_time, 1.2)` |
| Shift + wheel | `scroll_by(±0.1 × view_range)` |
| Spacebar (anywhere in the app) | Toggle playback |

Mouse-position-centred zoom matches Praat / Audacity / WaveSurfer.js / every map app. Shift+wheel for horizontal scroll matches GIMP / Inkscape / IDE editors.

### Per-frame re-bucketing in the waveform

B2's `EnvelopeCache` baked a 2000-bucket envelope over the full bundle at load time. At full zoom that's fine; zoomed in it'd render the same buckets at lower density, losing the spike-preserving property the min/max envelope existed for. C5 replaces the cached envelope with a per-frame call against the visible range:

```rust
fn render_waveform(env: &EnvelopeCache, timeline: &TimelineState, plot_width_px: usize) {
    let buckets = build_envelope_for_range(
        &env.mono_samples,
        env.sample_rate,
        timeline.view_start,
        timeline.view_end,
        plot_width_px,  // one bucket per pixel column
    );
    // draw vertical lines per bucket as before
}
```

This is the "per-frame re-bucketing" promised in the B2 entry. Cost is `O(visible_samples)` per frame, which is bounded — even fully zoomed out on a 10-minute file at 44.1 kHz that's ~26M iterations, well under a 16 ms frame budget.

The cached `EnvelopeCache.envelope` field stays for now (used as a fallback if width is 0 — defensive) but the rendering path no longer reads it.

### Spectrogram: no work to do

The B3 texture covers the full bundle. Narrowing the plot bounds (`Plot::include_x(view_start)…(view_end)`) lets egui_plot crop the displayed region for free. No re-render, no re-upload. The 4096-column cap means even at full zoom-out, one screen pixel maps to multiple texture columns; zooming in, the texture columns spread out — visually fine for a viewer (it's not a per-pixel scientific output).

Future polish: at extreme zoom-in, the spectrogram resolution can look chunky if the visible range is `< 4096 × hop_seconds`. A multi-resolution texture cache could fix that; not C5 scope.

### Playback

Embedded directly on `SaddaApp` as `Option<Playback>`:

```rust
struct Playback {
    _stream: cpal::Stream,            // !Send on Linux ALSA; OK because SaddaApp stays on the main thread
    state: Arc<PlaybackState>,
}

struct PlaybackState {
    samples: Vec<f32>,                // mono mixdown, possibly resampled to the device rate
    device_sample_rate: u32,
    bundle_sample_rate: u32,          // to convert atomic cursor back to bundle time
    cursor_samples: AtomicUsize,      // current sample position; audio thread advances; GUI reads
    finished: AtomicBool,             // set true when cursor reaches the end
}
```

Flow on spacebar (when stopped):

1. Pull mono samples + sample rate from `active_envelope`.
2. Build output stream on the default device. Get device's preferred sample rate.
3. If device rate ≠ bundle rate, linear-resample the samples once into a fresh `Vec<f32>`. Quality is acceptable for monitoring playback; not for studio output.
4. Build `Arc<PlaybackState>` with `cursor_samples` initialised from the current `timeline.cursor`.
5. Build cpal output stream with a callback that reads from `state.samples` advancing the atomic.
6. `stream.play()`.
7. Store `Playback` in `SaddaApp.playback`.

On spacebar (when playing) or at end of audio: drop the `Playback`. cpal's Drop impl stops the stream cleanly.

On each frame: if playback exists, read `cursor_samples` and update `timeline.cursor`. If cursor went off-screen, auto-scroll the view to keep cursor visible (re-centre at cursor when it leaves the right edge during playback).

### Threading

cpal's output callback runs on a real-time audio thread. Same rules as E1 input: no allocations, no locks, no GIL. The callback reads `samples` (immutable shared `Arc<Vec<f32>>` via `PlaybackState`) and advances an `AtomicUsize` — both lock-free. End-of-audio is signalled to the GUI via an `AtomicBool` that the main thread polls each frame.

No `rtrb` for output: the callback doesn't need a queue because the entire sample buffer is in memory. The pattern is closer to "play a static buffer with a moving read head" than to E1's "stream samples through a ring."

### Confirmed C5 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Timeline state location | **One `TimelineState` on `SaddaApp`, plumbed into each pane** | Single source of truth; pure-data; testable in isolation |
| Reset on bundle change | **`[0, duration]` + cursor at 0** | Sensible default; preserves view state across re-selects of the *same* bundle |
| Zoom centre | **Mouse position** | Universal convention; Praat / Audacity / map apps |
| Zoom factor per wheel notch | **1.2× (out) / 0.833× (in)** | Standard geometric step; ~7 notches per 10× zoom |
| Scroll | **Shift + wheel; ±10% of view range per notch** | Familiar from GIMP / IDEs; proportional pan matches zoom level |
| Click-positions-cursor scope | **Waveform + spectrogram. Tier-strip clicks select + position cursor at annotation start** | Tier-strip clicks have to do double duty: existing B4 selection + new cursor positioning |
| Cursor visualisation | **1-px red vertical line on all three panes** | High contrast against viridis/magma/greyscale; minimal visual weight |
| Spacebar binding | **Global at the app level (anywhere)** | Universal transport convention; `egui::Context::input` consumes the key so it doesn't double-fire |
| Waveform re-bucketing | **Per-frame over the visible range; one bucket per pixel column** | The fix B2 promised; cheap; preserves the min/max property at any zoom |
| Spectrogram on zoom | **Just narrow plot bounds against the existing texture** | Re-rendering on every zoom would be ruinous; bound-crop is free |
| Playback device | **`cpal::default_host().default_output_device()`** | No device picker yet; users on multi-output systems can change the OS default |
| Sample-rate mismatch | **One-shot linear resample at playback start** | Sufficient quality for monitoring; avoids dragging in a real SRC library at v1 |
| Resampling library | **None — hand-rolled linear interp** | ~15 LOC; v1 monitoring quality is fine; can swap in `rubato` later if needed |
| End-of-audio behaviour | **Auto-stop; cursor stays at end** | Matches Praat; loop mode is a 0.3 polish item |
| View-follows-cursor during playback | **Yes — re-centre when cursor crosses the right edge** | The most common scrubbing UX; user can still drag the wheel mid-playback |
| Cursor advance smoothness | **Read `AtomicUsize` per frame; convert to seconds** | At 60 fps with audio buffers ~10 ms, motion looks smooth |

### State changes on `SaddaApp`

- **New**: `timeline: TimelineState`, `playback: Option<Playback>`.
- **Reset** in `clear_bundle_selection` and `select_bundle`: timeline resets to the bundle's full range, cursor to 0; playback drops.

### Layout

- `crates/app/src/state.rs` — `TimelineState` + `build_envelope_for_range` + unit tests for zoom/scroll/clamp/cursor/per-range bucket counts.
- `crates/app/src/playback.rs` — new module. `Playback`, `PlaybackState`, `start_playback(env, timeline) -> Result<Playback, String>`, linear resample, cpal stream wiring.
- `crates/app/src/main.rs` — wire `timeline` into the three pane renderers, add the spacebar handler, drive the per-frame cursor advance from `Playback`.

### Lossiness / what C5 deliberately doesn't ship

- **Loop / region playback.** Selecting a time range to loop is the natural extension; 0.3+ polish.
- **Output-device picker.** Default device only at v1. Users who want a specific output change the OS default. Adds when a real workflow asks.
- **High-quality resampler.** Linear interpolation is "fine for monitoring." Swap to `rubato` if a real complaint surfaces.
- **Multi-channel playback.** Mono mixdown only; matches the rest of the GUI.
- **Latency calibration.** cpal default buffer sizes — typically 5–20 ms. No correction for output latency in the cursor display. Sub-frame precision isn't a 0.2 concern.
- **Visible-region cursor follow (smooth-scroll instead of re-centre).** Smooth-scroll is nicer; re-centre is simpler; choose simpler for v1.
- **Keyboard nav** (←/→ to step, Home/End, etc). 0.3 polish.
- **Touch-pad pinch-to-zoom.** Not in scope; mouse-wheel only.
- **Two-finger scroll on macOS trackpads (Ctrl+wheel zoom convention).** Egui handles wheel events platform-agnostically; if natural-scroll directions feel inverted, fix in a polish slice.

### What this entry doesn't decide

- **Whether the cursor should hide when not in focus on any pane.** Just always-on for v1.
- **Whether to show a transport bar with explicit play/stop buttons.** Spacebar only at v1; a transport widget can come later if mouse-only users complain.
- **What happens if the bundle changes while playback is running.** Drop the playback in `select_bundle`; settled in code.
- **Whether resample latency at playback start is worth a progress indicator.** ~10 ms for a 1-minute file; not worth UI.

### Sources / references

- 2026-05-23 Phase 2 slicing entry (C5 row)
- 2026-05-23 B2 entry (per-frame re-bucketing promise this slice fulfils)
- 2026-05-23 B3 entry (spectrogram texture-bound semantics this slice exploits)
- 2026-05-23 B4 entry (annotation-selection model this slice extends with cursor-positioning)
- 2026-05-22 E1 entry (cpal input + atomic-cursor + thread-model patterns reused on the output side)
- cpal output examples: <https://github.com/RustAudio/cpal/tree/master/examples>
- WaveSurfer.js zoom + cursor reference: <https://wavesurfer-js.org/docs/options>

---

## 2026-05-23 — Tier-strip view (B4): three-pane vertical split, per-tier-type lane vocabulary, clickable selection

Goal: settle the fourth Phase 2 slice. B4 adds the tier-strip pane below the spectrogram, completing the Praat-style waveform / spectrogram / tier-strip visual stack. Plus the basic annotation-selection state that C5 (cursor sync) and D-cluster (editing) will both reach into.

### What B4 must deliver

From the Phase 2 slicing entry: "Reads tier rows via `Project::tiers(Some(bundle_id))` + `intervals/points/references_for`. Renders each tier as a horizontal lane below the spectrogram; intervals as filled rectangles, points as vertical ticks; clicking a row selects the annotation."

### Layout: three-pane split

```
┌──────────────────────────────────────────────────────────┐
│ menu                                                     │
├───────────┬──────────────────────────────────────────────┤
│ sidebar   │ caption                                      │
│           ├──────────────────────────────────────────────┤
│ bundles   │                                              │
│           │   waveform                  ← Panel::top      │
│           │                                              │
│           ├──────────────────────────────────────────────┤
│           │ toolbar (window / hop / colormap)            │
│           ├──────────────────────────────────────────────┤
│           │                                              │
│           │   spectrogram               ← CentralPanel   │
│           │                                              │
│           ├──────────────────────────────────────────────┤
│           │ phones   ┃ ━━━ ━━━ ━━ ━━━━ ━━━━ ━━━━━━       │
│           │ words    ┃ ━━━━━━ ━━━━━━━━━━━━━━━━━━         │ ← Panel::bottom (B4)
│           │ events   ┃    ┃   ┃   ┃    ┃                 │
│           │ lex      ┃ (reference)                       │
│           ├──────────────────────────────────────────────┤
│           │ Project: vowels  ·  /home/alice/vowels       │
└───────────┴──────────────────────────────────────────────┘
```

Two `Panel`s now wrap the central pane: `top("waveform_split")` (resizable, ~220px) and `bottom("tier_strip")` (resizable, ~28px × #tiers). Spectrogram fills whatever's left.

### Lane vocabulary

Each tier renders as a 28-pixel-tall row with a 120-pixel label gutter on the left. Past the gutter, the lane spans the bundle's `[0, duration_seconds]` along the x-axis (sharing units with the spectrogram and waveform — sets up C5's synced cursor for free).

| Tier type | Lane contents |
|---|---|
| `interval` | Filled rectangle per interval, spanning `[start, end]`. Label text inside (truncated with `…` if it overflows). Selected interval gets a brighter fill + a stroke. |
| `point` | Vertical tick at each point's time. Label drawn above the tick (truncated similarly). Selected tick gets a brighter colour + 2× width. |
| `reference` | Lane shows a "(reference — N targets)" caption instead of time-positioned content. References don't have time data of their own; sensible rendering needs to resolve the targets, which is a future slice. |
| `continuous_numeric` / `continuous_vector` / `categorical_sampled` | Lane shows "(dense — not displayable in tier strip)". Visualizing these as overlays on the spectrogram is the right idiom; it lands when overlays do (post-0.2). |

### Selection state

New `AppState`-adjacent field:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnnotationSelection {
    Interval { tier_id: i64, annotation_id: i64 },
    Point    { tier_id: i64, annotation_id: i64 },
    // Reference selection added when reference rendering becomes interactive.
}

struct SaddaApp {
    // ...
    selected_annotation: Option<AnnotationSelection>,
}
```

Selection clears on bundle change and on click outside any tick/rectangle. Lives on `SaddaApp` (not `PersistedState`) — selection is an in-memory ephemerality, not a persistent setting.

### Confirmed B4 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Pane placement | **Bottom panel, below spectrogram** | Praat / WaveSurfer / librosa convention; spectrogram + tier strip share the same x-axis units |
| Lane height | **Fixed 28 px** | Compact; readable; consistent regardless of tier count |
| Tier name gutter | **Fixed 120 px on the left** | Fits most names; truncates with `…` |
| Hidden tiers | **Show with a "(can't render this type)" caption** | Don't silently drop — confuses users about what's in the corpus |
| Selection state | **In-memory only on `SaddaApp`** | Selection isn't worth persisting across launches; cluster C5/D6/D7 read it directly |
| Selection clears on | **Bundle change + click outside any item** | Matches every editor's behavior |
| Interval label overflow | **Truncated with `…`, no wrap** | Lane height doesn't accommodate wrapping; hover tooltip can land later |
| Reference-tier preview | **`(reference — N targets)` text only** | Time-less; meaningful rendering requires resolving `(target_kind, target_id)` pairs — separate slice |
| Many tiers | **Vertical scroll** | `ScrollArea` wrap around the lane list |
| Tier ordering | **Engine's `Project::tiers` insertion order** | Deterministic; reordering UX is a D-cluster context-menu concern |

### Pure-data extraction

Most of B4 is rendering, but two helpers belong in `state.rs` (testable):

- `truncate_label(text, max_chars) -> String` — produces `"hello…"` when the label is too long. Used by both interval rectangles and point ticks.
- `format_reference_lane_caption(n_targets: usize) -> String` — `"(reference — 3 targets)"` / `"(reference — 1 target)"` etc.

### Lossiness / what B4 deliberately doesn't ship

- **Editing.** Drag-to-create / drag-boundary / double-click-to-edit-label all land in D6 (interval) / D7 (point). B4 is read-only.
- **Hover tooltips.** Defer to a polish slice.
- **Tier reordering / rename / delete via right-click.** D-cluster context menus.
- **Reference-tier visualisation.** Resolving the targets needs design — show as link? Stretch from the source to the target? Out of B4 scope.
- **Dense-tier overlays on the spectrogram.** Same.
- **Selection visualisation that extends to the waveform / spectrogram.** That's C5 (synced cursor) — selecting an interval should highlight its time range across all panes. B4 marks the selected annotation in the tier strip only.
- **Multiple tier strips (left/right of the spectrogram).** ELAN / Anvil ship multi-pane tier views; we don't yet need that complexity.
- **Tier-level mute / solo / hide toggles.** Same.

### What this entry doesn't decide

- **Colour palette per tier type.** Settled at implementation — light blue for intervals, amber for points, grey for inactive captions. Themed palette work is a polish slice.
- **Whether the lane labels can be rich-text** (e.g. icons indicating tier type). Plain text for now.
- **Whether selection survives a reload of the same bundle.** Currently clears on any bundle re-select.

### Sources / references

- 2026-05-23 Phase 2 slicing entry (B4 row)
- 2026-05-23 B3 entry (which this stacks below)
- Praat tier display reference: <https://www.fon.hum.uva.nl/praat/manual/TextGrid_files_2.html>
- ELAN tier strip layout: <https://www.mpi.nl/corpus/html/elan/ch04s02.html>

---

## 2026-05-23 — Spectrogram view (B3): dB-FS scale + 70dB floor, three colormaps with toolbar toggle, cached texture, central-pane vertical split

Goal: settle the third Phase 2 slice. B3 adds the spectrogram below the waveform, with the toolbar controls + colormap variants the slicing entry called for. Combined with B2 this gives the visual stack a Praat-trained user expects.

### What B3 must deliver

From the Phase 2 slicing entry: "Uses C1's `sadda::dsp::spectrogram`; renders via egui's `TextureHandle` + a viridis-ish colormap. Configurable hop / window via a panel-local control row." Plus the spike note: "Spectrogram render performance on a 10-minute file. Render time for a long file is the most likely 'felt' bottleneck."

### Render pipeline

```
mono samples (cached from B2's bundle load)
        │
        ▼
sadda::dsp::stft(samples, hann_window, hop)        ← C1 primitive
        │
        ▼
sadda::dsp::spectrogram::power_spectrogram(...)    ← C1 primitive
        │
        ▼
10 · log10(power)  (dB-FS, clamped to [-dynamic_range, 0])
        │
        ▼
normalise to [0, 1] then index into the active colormap LUT
        │
        ▼
RGBA8 buffer, written into an egui::ColorImage
        │
        ▼
egui::Context::load_texture → TextureHandle
        │
        ▼
plot_ui.image(...)
```

The expensive bit (STFT + colour bake) runs once per `(bundle_id, window_ms, hop_ms, colormap, dynamic_range_db)` tuple. Cached output is `(TextureHandle, n_freq_bins, n_frames)`. Pan/zoom (when C5 lands) doesn't invalidate the cache.

### Layout

Central panel splits vertically inside `waveform_pane`:

```
┌──────────────────────────────────────────────────────────┐
│ Bundle #1  ·  16000 Hz  ·  4.32s                         │ ← B2 caption row
├──────────────────────────────────────────────────────────┤
│                                                          │
│              [ waveform — B2 ]                           │ ~30% height
│                                                          │
├──────────────────────────────────────────────────────────┤
│ Window: [25] ms   Hop: [5] ms   Colormap: [viridis ▾]    │ ← B3 toolbar
├──────────────────────────────────────────────────────────┤
│                                                          │
│                                                          │
│              [ spectrogram — B3 ]                        │ ~70% height
│                                                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Split implemented as an `egui::Panel::top` for the waveform + the spectrogram filling the rest of the central pane. User can drag the divider to rebalance; default proportions persist across launches via egui's own panel-state storage.

### Confirmed B3 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Frequency range | **Full Nyquist** | Lossless; matches Praat's default behavior; user can read off the y-axis to see what sample rate they're working with |
| Power scale | **dB-FS, `10·log10(power)`** | Universal spectrogram convention; gives the visible dynamic-range structure that linear power flattens |
| Dynamic-range floor | **70 dB by default** | Praat's default; covers the speech-relevant dynamic range without lighting up noise floors |
| Colormaps shipped | **Viridis (default), Magma, Greyscale** | Viridis: modern perceptually-uniform default. Magma: dark-mode alt. Greyscale: Praat refugees |
| Colormap toggle | **Dropdown in the toolbar row** | One-click swap; persisted across launches via the same `PersistedState` already carrying `theme` |
| Colormap source | **`colorous = "1"` crate** | ~50 KB pure-Rust dep that ships matplotlib-faithful viridis + magma. Avoids hand-baking + bit-rot of approximations |
| Default window | **25 ms** | Speech-DSP convention. Configurable in the toolbar |
| Default hop | **5 ms** | Speech-DSP convention; 80% overlap at 25 ms window |
| Hop/window inputs | **Numeric `DragValue` widgets, ms units** | More compact than sliders; precise; allow keyboard entry |
| Cache invalidation key | **`(bundle_id, window_ms, hop_ms, colormap, dynamic_range_db)`** | Recomputes on any of the five changing |
| Long-recording cap | **Texture width capped at 4096 cols; longer audio averages frames into buckets** | egui's typical max texture is 8192; 4096 keeps headroom and gives roughly 1px per ~150ms at 10 minutes |
| Y-axis labelling | **Hz, top = Nyquist** | Standard; matches Praat/librosa |
| X-axis labelling | **seconds, shared with the waveform plot** | Sets up C5's synced-cursor work — the two plots already speak the same x-units |

### State changes

```rust
struct SpectrogramConfig {
    window_ms: f32,           // default 25.0
    hop_ms: f32,              // default 5.0
    colormap: ColormapKind,   // default Viridis
    dynamic_range_db: f32,    // default 70.0
}

struct SpectrogramCache {
    bundle_id: i64,
    config: SpectrogramConfig,
    texture: egui::TextureHandle,
    duration_seconds: f64,
    nyquist_hz: f32,
}
```

`SpectrogramConfig` lives in `PersistedState` so reopening a project remembers the last-used DSP settings.

### Pure-data extraction

`crates/app/src/state.rs` already hosts the envelope downsampler; B3 adds two more pure-data helpers there for unit-testability:

- `power_to_db_normalized(power: &[f32], dynamic_range_db: f32) -> Vec<f32>` — log + clamp + normalize-to-[0,1] in one pass.
- `colormap_bake(values: &[f32], width: usize, height: usize, colormap: ColormapKind) -> Vec<u8>` — apply the colormap to a freq-major (height × width) buffer and emit row-major RGBA8 (4 × width × height bytes).

Both are testable with synthetic inputs; the egui texture upload happens in main.rs.

### Lossiness / what B3 deliberately doesn't ship

- **Per-frame log mel** / **mel spectrogram view.** A different module on a different y-axis; not in 0.2 scope. Could become a "View → Mel" toggle later.
- **Cepstrogram / autocorrelogram.** Same.
- **Reassigned spectrogram** (sharper-look). Same.
- **Pitch / formant overlay on the spectrogram.** The pitch contour overlay is a 0.3 polish item; formants land with their own slice once the visualisation vocabulary settles.
- **Zoom / scroll / cursor.** C5.
- **Pre-emphasis toggle in the toolbar.** Praat exposes it; we don't yet. The user can pre-emphasise externally and re-import; first-class toggle waits for a real workflow ask.
- **Phase spectrogram.** Power only.

### What this entry doesn't decide

- **Whether the spectrogram inherits the waveform pane's vertical split ratio or has its own.** Settled at implementation — start with a single egui-managed top/bottom split for simplicity.
- **Exact toolbar widget shapes.** `DragValue` for the numerics, `ComboBox` for the colormap; can refine later.
- **Whether the cache shares an LRU across bundles.** Single-bundle cache for v1; LRU lands if real users complain about switch latency.

### Sources / references

- 2026-05-23 Phase 2 slicing entry (B3 row + the render-performance spike note)
- 2026-05-23 B2 entry (which this stacks on top of; shares the waveform pane's caption row)
- 2026-05-21 C1 entry (the STFT + power-spectrogram primitives this consumes)
- Matplotlib viridis/magma reference: <https://matplotlib.org/stable/users/explain/colors/colormaps.html>
- `colorous`: <https://github.com/dtolnay/colorous>
- Praat spectrogram defaults reference: <https://www.fon.hum.uva.nl/praat/manual/Spectrogram.html>

---

## 2026-05-23 — Waveform view + bundle sidebar (B2): min/max envelope, fixed-resolution cache, left-sidebar single-select

Goal: settle the second Phase 2 slice. B2 brings the first real content pane — a waveform of the active bundle's audio — and the bundle-selection UI that B3 / B4 / C5 will all share.

### What B2 must deliver

From the Phase 2 slicing entry: "Reads the active bundle's audio via `Project::load_audio`; renders a min/max envelope (better than the Phase-0 step-sampled line). Vertical Y bounds clamped to [-1, 1]. No interaction yet."

Plus: a bundle selector. The slicing entry's A1 "doesn't-decide" note flagged "probably a sidebar tree of bundles unlocked by clicking a project root" as emerging in cluster B; B2 owns it because it's the first slice that needs an active bundle to render anything.

### Sidebar shape

Left side, ~200px wide, resizable down to ~120px, lists every bundle from `Project::bundles()` as a single-select list. Each row shows the bundle name plus a small grey suffix with duration in seconds.

```
┌───────────────────┬──────────────────────────────────────────┐
│ Bundles           │                                          │
│                   │                                          │
│ ▸ practice_take_1 │             [ waveform here ]            │
│   speaker_01      │                                          │
│   greeting_v2     │                                          │
│                   │                                          │
│ (3 bundles)       │                                          │
├───────────────────┴──────────────────────────────────────────┤
│ Project: vowels  ·  /home/alice/projects/vowels              │
└──────────────────────────────────────────────────────────────┘
```

No hierarchy / grouping yet (we don't have bundle-folder semantics in the corpus). Hierarchy lands when sessions / speakers get sidebar treatment in a later slice.

### Min/max envelope

For a plot rendered at `W` pixels wide showing `N` audio samples (mono mixdown):

```
bucket_size = ceil(N / W)
for col in 0..W:
    samples_in_bucket = samples[col * bucket_size .. (col + 1) * bucket_size]
    envelope[col] = (min(samples_in_bucket), max(samples_in_bucket))
```

The plot draws one vertical line segment per column, from `(t, min)` to `(t, max)`. Visually equivalent to Audacity's "Waveform" view; peaks survive any zoom factor (whereas Phase-0's `step_by` step-sampling would silently drop them).

For B2 the bucketing is computed **once at fixed resolution** (~2000 buckets) when the user selects a bundle, and the cached array is drawn at any plot width. This is fine while there's no zoom — the plot width doesn't change frame-to-frame. C5 (zoom + scroll + playback) will replace this with a re-buckets-per-frame strategy when zoom lands.

### State changes

```rust
struct SaddaApp {
    app_state: AppState,
    selected_bundle_id: Option<i64>,            // new
    active_envelope: Option<EnvelopeCache>,     // new
    persisted: PersistedState,
    error: Option<String>,
}

struct EnvelopeCache {
    bundle_id: i64,
    sample_rate: u32,
    duration_seconds: f64,
    /// Per-bucket (min, max) over the mono mixdown.
    envelope: Vec<(f32, f32)>,
}
```

Selecting a bundle = `Project::load_audio(id)` + build envelope + stash in `active_envelope`. On project change, both fields reset.

### Confirmed B2 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Bundle selector | **Left sidebar, single-select, flat list** | First content pane; gates B3/B4/C5 having an active bundle to render |
| Sidebar width | **~200px default, resizable to ~120 minimum** | Bundle names of typical length fit; doesn't crowd the waveform |
| Bundle row label | **`<name>`  ·  `<duration>s` (greyed)** | Quick scanning by duration helps when names collide |
| Downsampler | **Min/max envelope (Audacity convention)** | Peak-preserving at any zoom; same algorithm we'll keep for C5 |
| Cache strategy | **Fixed resolution ~2000 buckets at load time** | Cheap, simple, fine without zoom. C5 replaces with per-frame re-bucketing |
| Channel handling | **Mono mixdown** | Per-channel display is a 0.3 polish item |
| Y bounds | **Clamped to [-1, 1]** | Matches the engine's `Audio::samples` normalisation contract |
| Cursor / playback | **Not in B2** | C5 owns the timeline interaction |
| Empty state | **"Open a bundle via File → Open Bundle…" hint in central panel** | Same hint A1 used, repurposed for "project loaded, no bundle selected" |
| Selection persistence across launches | **No** | Selection is in-memory only. Cluster F / G can revisit if needed |

### Layout

- `crates/app/src/state.rs` — pure-data extension: `EnvelopeCache` struct + `build_envelope` function. Unit tests covering: empty input, single-bucket constant, alternating sine. Stays free of egui.
- `crates/app/src/main.rs` — sidebar pane (left), waveform pane (central). The existing `welcome` / `error` / status-bar code is untouched.

### Lossiness / what B2 deliberately doesn't ship

- **Zoom and scroll.** Slice C5.
- **Cursor.** Slice C5.
- **Playback.** Slice C5.
- **Per-channel waveforms.** 0.3 polish.
- **Bundle reordering / renaming / delete from sidebar.** Engine API doesn't have rename/delete yet; right-click context menus arrive when the editing slices (D6/D7) introduce them on tiers.
- **Session / speaker grouping in the sidebar.** No bundle-folder semantics; deferred until the corpus model justifies it.
- **Waveform overlays (pitch, intensity).** Pure DSP overlays will land on the spectrogram pane (B3) where the colour space is more permissive.
- **Empty-project hint to "create your first bundle".** The greyed-out sidebar + the File menu's enabled "Open Bundle…" is enough signal for a first user.

### What this entry doesn't decide

- **Whether the sidebar collapses to a hamburger icon at small widths.** Egui's resizable side panel handles this for free; the min-width clamp is the only safety net B2 ships.
- **Bundle-row right-click affordances.** Adds when D-cluster's editing surfaces decide their own context-menu vocabulary.
- **Sidebar persistence width across launches.** Egui can persist side-panel width via its own internal storage; whether to opt in here vs let the user resize on every launch is settled in implementation.

### Sources / references

- 2026-05-23 Phase 2 slicing entry (B2 row)
- 2026-05-23 A1 entry (which this extends with the sidebar + bundle selection model)
- Audacity waveform-rendering reference (min/max envelope algorithm): <https://wiki.audacityteam.org/wiki/Audacity_Source_Code>
- egui side-panel reference: <https://docs.rs/egui/latest/egui/containers/panel/struct.SidePanel.html>

---

## 2026-05-23 — App shell + project open/create (A1): welcome screen, persistent recent-projects, greyed Open-Bundle until project loaded

Goal: settle the first Phase 2 slice. A1 replaces the Phase-0 sketch in `crates/app/src/main.rs` (which loaded a bare WAV via file dialog and drew waveform + f0) with a project-aware app shell. Everything cluster B and later builds on this state model.

### What A1 must deliver

From the Phase 2 slicing entry: window with menu (File → New Project / Open Project / Open Bundle); persistent window state via egui's built-in storage; light/dark from system; loads a `Project` and shows "Project: <name>" in the status bar. **No content panes yet.**

Acceptance: opens a real `Project` created by Phase 1's `sadda.new_project`; recent-projects list survives a relaunch; "Open Bundle" is greyed until a project is loaded; trying to open a non-sadda directory shows a clear error.

### Prior art

| Tool | What we lift | What we leave |
|---|---|---|
| **VS Code** | Welcome screen with New / Open + Recent column; folder-as-project model; recent list persists across launches | Multi-folder workspaces; the wider extension ecosystem |
| **Reaper / Logic / Ableton** | Project-centric workflow; recent-projects in File menu | Auto-reopen-last on launch (too opinionated for a 0.2) |
| **Praat** | The terminology ("Open", "New") | The object-list paradigm; no project concept |
| **Audacity** | Single project = single file simplicity | Single-file projects don't fit our directory-tree project layout |

### State model

The app has three top-level states; the central panel renders differently per state:

```
enum AppState {
    NoProject,
    ProjectLoaded { project: Project, root: PathBuf },
    Error { message: String, since: Instant },  // overlay; dismisses on click
}
```

- `NoProject` → welcome card: title, [New Project] button, [Open Project…] button, Recent list (clickable rows).
- `ProjectLoaded` → status bar populated; central panel says "Open a bundle from File → Open Bundle…" (placeholder for cluster B).
- `Error` → red banner at the bottom; dismissible.

The state transitions happen on menu actions and click handlers, never inside the render loop's body (no spawning of file dialogs mid-frame — they block).

### Welcome screen layout

```
┌──────────────────────────────────────────────────────────┐
│  File  Edit  View  Help                                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                   sadda                                  │
│        speech-analysis toolkit                           │
│                                                          │
│        ┌──────────────┐  ┌──────────────┐                │
│        │ New Project  │  │ Open Project │                │
│        └──────────────┘  └──────────────┘                │
│                                                          │
│        Recent                                            │
│        • /home/alice/projects/vowels                     │
│        • /home/alice/projects/connected_speech           │
│        • /home/alice/projects/practice                   │
│                                                          │
│        (No recent projects yet)                          │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Recent list is capped at 5 entries; clicking a row opens it. If a recent row's path no longer exists, the row is rendered greyed with `(missing)` suffix and clicking it removes the entry.

### Persistence

eframe's built-in storage hook (`Storage` trait via `set_value(eframe::APP_KEY, ...)` and `eframe::get_value(...)`) covers everything we persist:

```rust
#[derive(Default, serde::Serialize, serde::Deserialize)]
struct PersistedState {
    recent_projects: Vec<PathBuf>,    // capped at 5; most-recent first
    theme_preference: ThemePref,      // System | Light | Dark
}
```

Window size + position are persisted by eframe automatically once `persist_window: true` is set on `NativeOptions::viewport`.

### Menus

- **File**
  - New Project… — folder picker; `Project::create(path, name = path.file_name())`. Errors if the folder already exists per the existing `Project::create` contract.
  - Open Project… — folder picker; `Project::open(path)`. Errors if the folder isn't a sadda project.
  - Recent Projects → submenu of up to 5 entries.
  - Open Bundle… — greyed unless `AppState::ProjectLoaded`. WAV file picker; calls `Project::add_bundle(name = file_stem, source = path)`. Doesn't *select* the bundle yet (selection is a slice-B concern); just registers it.
  - ─
  - Quit — Cmd/Ctrl+Q.
- **Edit**, **View** — placeholder menus, populated by later slices. Empty in A1 so the menu bar shape is set.
- **Help**
  - About — version + license + repo link.

### Error model

Engine errors bubble up through the existing `EngineError` hierarchy. The app catches them on menu actions and converts to `AppState::Error { message }`. Recoverable: clicking dismisses the banner and returns to the previous state. Common cases:

- `Project::create` on an existing path → "Project path already exists: {path}"
- `Project::open` on a non-sadda directory → "Not a sadda project: {path}"
- `Project::open` on a future-schema project → uses the existing `EngineError::SchemaTooNew` Display impl verbatim.

### Engine-side touchups

Two small additions to `sadda-engine` to support the app cleanly:

- `Project::name() -> Result<String>` — already exists from B1; reused as-is.
- `Project::is_project_root(path: &Path) -> bool` — new helper that returns true iff `<path>/project.toml` + `<path>/corpus.db` exist. Used to grey-out the Recent rows for moved/deleted projects without doing a full `Project::open`.

### Layout

- `crates/app/Cargo.toml` — adds `serde` (for `PersistedState`) and `directories-next = "2"` (for cross-platform recent-projects storage). Actually, eframe's `Storage` handles platform-appropriate storage already; we just need `serde` (which eframe transitively pulls in but isn't a direct dep yet).
- `crates/app/src/main.rs` — refactor end-to-end. Phase-0 waveform code goes away.
- `crates/app/src/state.rs` — `AppState`, `PersistedState`, recent-projects management. Unit-tested.
- `crates/app/src/ui/welcome.rs` — welcome-card widget.
- `crates/app/src/ui/menus.rs` — File/Edit/View/Help bar.
- `crates/engine/src/corpus.rs` — `Project::is_project_root` helper.

### Confirmed A1 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Startup behavior | **Welcome screen with New / Open / Recent** | Most discoverable; matches VS Code / IntelliJ convention |
| "Open Bundle" semantics | **Greyed until a project is loaded** | Bundles only live inside projects per our data model; no ambiguity about where they go |
| Recent-projects depth | **5 entries, most-recent first** | Standard; deeper lists rot |
| Missing recent entries | **Rendered greyed with `(missing)` suffix; click removes** | Self-cleaning without needing a separate "Manage Recent" UI |
| Theme | **Follow system light/dark by default; manual override in View menu** | Defer the custom palette; egui's defaults are fine |
| Persistence backend | **eframe's built-in `Storage`** | Cross-platform, no new dep |
| Auto-reopen-last on launch | **No** | One extra click for repeat users beats the "stale project sitting open" failure mode |
| Phase-0 waveform code | **Removed** | Comes back in slice B2 reading from `Project` |

### Lossiness / what A1 deliberately doesn't ship

- **Sidebar / project navigator pane.** Empty central panel for now; the bundle picker lives in a menu. Sidebar lands when there's something to browse to (slice B).
- **Recently-opened *bundles*** (vs projects). Bundles are project-scoped; per-bundle history would live inside the project's persisted state, deferred until cluster B/C exposes selection.
- **Drag-and-drop project folder onto the app window.** Nice-to-have; egui supports it; doesn't change the slice scope.
- **Welcome-screen graphics / branding.** Plain text + buttons in A1. A bundled logo lands at 0.2 release-polish time.
- **Multi-window / multi-project.** Single project per app instance; multi-instance is the single-writer-lock concern (slice F10).
- **Internationalisation.** All strings English; per the milestone-plan deferral.
- **Accessibility audit.** Egui's AccessKit integration is young; A1 best-effort, dedicated pass at 1.x per the milestone plan.

### What this entry doesn't decide

- **Exact welcome-card visual proportions.** Settled at implementation time; will iterate based on screenshots.
- **Whether "About" links open in the system browser or show a modal.** Both are easy; pick whichever feels right when implementing.
- **Whether `Project::is_project_root` should also check `corpus.db` is openable (not just exists).** Probably overkill — the existence check is cheap; the open-and-validate happens on actual `Project::open`.
- **Persistence file format / location.** eframe handles it; we just hand it serde-derived structs.

### Sources / references

- 2026-05-23 Phase 2 slicing entry (A1 row + the "single-bundle vs project-navigator" open item this entry settles)
- 2026-05-18 milestone-plan entry (Phase 2 row)
- 2026-05-18 Python API surface entry (`sadda.app` namespace; in-app commands)
- eframe Storage trait: <https://docs.rs/eframe/latest/eframe/trait.Storage.html>
- VS Code welcome editor: <https://code.visualstudio.com/docs/getstarted/userinterface#_welcome-page>

---

## 2026-05-23 — Phase 2 slicing: 11 slices in 7 clusters toward 0.2

Goal: sequence Phase 2 — the egui+wgpu desktop GUI — into a concrete commit-by-commit ordering, analogous to the 2026-05-21 Phase 1 slicing entry. The 2026-05-18 milestone-plan entry committed to Phase 2's *scope*; this entry commits to its *cadence and ordering*.

### What ships at 0.2 (from the milestone plan)

> Egui+wgpu app shell + project navigator + waveform/spectrogram/tier-strip with sync cursor + interval/point tier editing + embedded CPython in app + `sadda.app` basics (selection, register_command) + single-writer lock.

Plus, from this entry's design conversation: **audio playback** (bundled with the Sync slice) and **unsigned binaries** at 0.2 (macOS/Linux/Windows; signing is Phase 7 work).

### What changes from Phase 1

| | Phase 1 | Phase 2 |
|---|---|---|
| Slices | 12 | 11 |
| Each slice ends in | a usable Python entry point | a clickable feature you can demo |
| Layers crossed per slice | engine + PyO3 + tests | engine (small) + app + manual GUI verification |
| Release at end | 0.1 — Python library on PyPI | 0.2 — desktop binaries on the GitHub Releases page |
| Testability | unit + integration in CI | mix of unit-testable layers (state, controllers, downsamplers) + manual screenshot verification |

The GUI couples directly to `sadda_engine::Project` (not via PyO3). Python and the GUI share `corpus.db` via the new single-writer lock landing in cluster F.

### Prior-art shape

| Tool | What we lift | What we leave |
|---|---|---|
| **Praat** | Bottom-up tier strips below the spectrogram; per-bundle "Edit" panel | Modal sub-windows per object; the byzantine info-list pane; the menu sprawl |
| **Audacity** | Selection-handle visual vocabulary; spacebar transport; click-to-position cursor | Linear waveform-only view; no tier model |
| **ELAN** | Multi-pane tier editor with stacked lanes; tier-hierarchy visualisation | Video-first layout; no DSP integration |
| **WaveSurfer.js** | Region-selection handles; smooth zoom + pan | Browser stack; no corpus model |
| **librosa display** | The waveform / spectrogram / pitch stacked layout convention | Read-only; no editing affordances |

### Decomposition: 7 clusters

**Cluster A — Shell** (lands first; gates the rest)

1. **App shell + project open/create.** Replace the Phase-0 `main.rs` sketch (which loads a bare WAV) with a project-aware app shell: window with menu (File → New Project / Open Project / Open Bundle); persistent window state via egui's built-in storage; light/dark from system; loads a `Project` and shows "Project: <name>" in the status bar. No content panes yet. Acceptance: opens a real `Project` created by Phase 1's `sadda.new_project`.

**Cluster B — Views** (independent of each other; can interleave; each ends in a visible pane)

2. **Waveform view.** Reads the active bundle's audio via `Project::load_audio`; renders a min/max envelope (better than the Phase-0 step-sampled line). Vertical Y bounds clamped to [-1, 1]. No interaction yet.
3. **Spectrogram view.** Uses C1's `sadda::dsp::spectrogram`; renders via egui's `TextureHandle` + a viridis-ish colormap. Configurable hop / window via a panel-local control row.
4. **Tier-strip view.** Reads tier rows via `Project::tiers(Some(bundle_id))` + `intervals/points/references_for`. Renders each tier as a horizontal lane below the spectrogram; intervals as filled rectangles, points as vertical ticks; clicking a row selects the annotation.

**Cluster C — Sync + playback**

5. **Synced cursor + zoom + scroll + playback.** Single shared timeline state across the three view panes. Spacebar plays/stops from the cursor via cpal *output* (reusing the existing cpal dep from E1 input). Click moves cursor. Mouse-wheel zooms; shift-wheel scrolls. Acceptance: scrub through a recording with the spectrogram cursor and tier-strip cursor staying in lockstep.

**Cluster D — Editing**

6. **Interval-tier editing.** Drag in empty space to create a new interval; drag a boundary to resize; double-click to edit label inline; delete key removes the selected interval. Writes via `Project::add_interval`; engine API extended with `update_interval / delete_interval` (these don't exist yet — small engine surface addition lands in this slice).
7. **Point-tier editing.** Click to add a point; drag to move; delete to remove. Same engine-surface extension pattern (`update_point / delete_point`).

**Cluster E — Scripting**

8. **Embedded CPython script panel.** Reuse Phase 0's `crates/script-engine` for the embed. Egui text editor + Run button + output pane. Scripts can `import sadda` and get the active project handle via `sadda.app.active_project`.
9. **`sadda.app` API: current_selection, active_bundle, register_command.** Lands the in-app `sadda.app` namespace per the 2026-05-18 API-surface entry. `register_command` adds the user's function to a command palette accessible via Ctrl+P or a menu. Scripts that don't touch `sadda.app` work identically whether run inside the app or via `sadda` from a terminal.

**Cluster F — Safety**

10. **Single-writer lock.** Two `corpus.db` writers running simultaneously (app + script, or two app instances) would corrupt the audit trail. Use SQLite's WAL + a `BEGIN EXCLUSIVE` advisory pattern, OR a `<project>/.sadda-lock` lockfile written at `Project::open`. Refuses to open a project that's locked, with a clear message naming the holder's PID. The Phase-1 `Project::open` API picks up the lock without breaking existing callers (PyO3 + tests).

**Cluster G — Release**

11. **0.2 binaries.** GitHub Actions workflow building **unsigned** desktop binaries on tag push for macOS arm64 / Linux x86_64 / Windows x86_64; upload as GitHub Release artifacts; update README with download links. macOS users see an "unidentified developer" warning; Linux + Windows are clean. Apple Developer / EV-cert signing is Phase 7 scope per the milestone plan.

### Dependency graph

```
A1 ─┬─→ B1 (waveform) ────┐
    ├─→ B2 (spectrogram) ─┤
    ├─→ B3 (tier strip) ──┴─→ C5 (sync + playback) ─┬─→ D6 (interval edit)
    │                                                 └─→ D7 (point edit)
    │
    ├─→ F10 (single-writer lock; can land any time after A1)
    │
    └─→ E8 (embed) ─→ E9 (sadda.app API) ─→ G11 (binaries last)
```

E8 (embed) needs A1's project model; E9 needs D-cluster's selection model. G11 is gated on everything else.

### Vertical-slice interpretation for Phase 2

Phase 1's vertical-slice rule was "each slice ends in a usable Python entry point." Phase 2's rule is **"each slice ends in a clickable feature you can demo."** That means:

- Every Phase 2 slice merges with a screenshot or a 10-second screen recording added to a `docs/changelog-media/` directory (which the docs site can pull in later).
- Unit tests cover state/controller/downsampler logic; GUI behaviour is verified manually per slice (no Selenium-for-egui equivalent worth wiring at this scale).
- CI builds and lints the app crate every commit but does not exercise the egui surface (already true in the Phase 1 ci.yml).

### Confirmed scope decisions

| Item | Decision | Reasoning |
|---|---|---|
| Audio playback | **In 0.2, bundled with the Sync slice (C5)** | Without playback, the GUI is visualization-only; feels half-finished for a Praat replacement. cpal output is the same dep as E1 input |
| Embedded CPython timing | **Late in Phase 2 (cluster E)** | Ship the visual editor first; add the script panel once `sadda.app`'s selection model is settled. Lowest risk of churning the embed API |
| 0.2 binary distribution | **Unsigned on the GitHub Releases page** | Skips the Apple Developer / EV-cert spend until 1.0; "unidentified developer" warning is acceptable for a 0.2 |
| GUI ↔ engine coupling | **Direct: app crate uses `sadda_engine::Project`** | Simplest, fastest, no IPC. Python and the GUI share `corpus.db` via the new single-writer lock |
| Tier hierarchy in the strip view | **Hierarchical (parent above child)** | EAF tiers carry parent_id since D2; the strip should reflect that. Matches ELAN |
| Spectrogram colormap | **Viridis-ish (perceptually uniform)** | Modern default; better than jet; avoids the colormap-as-data-distortion footgun |
| Editing model | **Direct manipulation (drag handles)** | Praat / Audacity / WaveSurfer.js convention; less modal than ELAN |
| Command palette | **Ctrl+P (Cmd+P on macOS)** | VS Code / Sublime / IntelliJ convention; familiar to the target audience |
| First slice | **A1 (app shell)** | Infrastructure-first, matches Phase 1's A1 cadence. Replaces the Phase-0 sketch in `main.rs` |

### Parallel risk spikes

Smaller list than Phase 1 since most cross-cutting unknowns settled in Phase 0 (`crates/script-engine` validated the embed) or Phase 1 (cpal, packaging baseline).

| Spike | Concurrent with | Purpose |
|---|---|---|
| **Embedded CPython packaging on Linux/Windows** | Slices A–C | The Phase-1 spike was macOS only. Validate Linux (ELF + libpython.so lookup) + Windows (DLL search path) before E8 commits to the embed. |
| **Spectrogram render performance on a 10-minute file** | Slice B2 | Render time for a long file is the most likely "felt" bottleneck. Spike the texture-upload + segmented-render strategy before B2's design pass. |

### Cut lines if timeline pressure hits

In priority order of what to defer first:

1. **Cluster E (embed + sadda.app)** → defer to 0.3. The visual editor is 0.2's headline; the script panel can layer in later.
2. **Spectrogram view (B2)** → defer to 0.3. Waveform + tier strip alone is a usable Praat-replacement for a labelling workflow. Loses the visualization differentiator.
3. **Point-tier editing (D7)** → ship interval editing only at 0.2; point tiers stay read-only. Most labelling workflows are interval-first.
4. **Windows binary (G11 subset)** → macOS + Linux only at 0.2; Windows later. Loses ~30% of the desktop audience.

**Not cuttable**: app shell (A1), at least one view (B1 or B3), single-writer lock (F10), any binary at all for at least one OS (G11 subset).

### What this entry doesn't decide

- **Project navigator vs single-bundle focus at A1.** The milestone plan says "project navigator"; the A1 acceptance criterion above starts with single-bundle. Settled inside A1's design pass — probably a sidebar tree of bundles unlocked by clicking a project root.
- **Persistent UI state beyond window size.** Recent files, last-open project, last-selected bundle. Egui's built-in storage covers it; the question is what to persist. Settled per slice as the need surfaces.
- **Keyboard shortcuts beyond the playback spacebar + delete-to-remove + Ctrl+P palette.** Mapping table emerges through Phase 2; first cut in A1.
- **Theme palette beyond "follow system light/dark".** Custom colour overrides for the spectrogram / tier-strip lanes happen inside B2/B3.
- **Whether `sadda.app.register_panel`** (arbitrary widget panels) ships in 0.2 or waits — the milestone plan punts the heavier panel API to 1.0. The cut-line list keeps `register_command` only.
- **Crash-reporting / telemetry.** Out of v1 entirely unless a real downstream need surfaces.

### Pace and revisit cadence

- Milestone plan estimates Phase 2 at 3–4 months solo part-time. 11 slices over ~14 weeks ≈ one slice every 1–2 weeks (same cadence as Phase 1).
- Revisit this entry after slice 5 (C5, the sync+playback slice — the first one with real interaction). If the interaction model fights the rest of the plan, re-slice D6/D7.
- After 0.2 ships: real GUI users land. Their feedback reshapes Phase 3+ scope per the milestone plan's "after-0.2" revisit point.

### Sources / references

- 2026-05-18 milestone-plan entry (Phase 2 row; "after-0.2" revisit point)
- 2026-05-18 Python API surface entry (`sadda.app` namespace; register_command shape)
- 2026-05-21 Phase 1 slicing entry (analogous structure)
- 2026-05-22 G12 entry (binary-distribution baseline)
- eframe (egui native shell): <https://github.com/emilk/egui/tree/main/crates/eframe>
- egui_plot: <https://github.com/emilk/egui_plot>
- cpal output examples: <https://github.com/RustAudio/cpal/tree/master/examples>
- WaveSurfer.js (region-selection vocabulary): <https://wavesurfer-js.org/>

---

## 2026-05-22 — Release 0.1.0 (G12): PyPI Python-only, cibuildwheel matrix, mkdocs-material on GitHub Pages

Goal: close Phase 1. G12 puts the Python library on PyPI as `sadda 0.1.0` and stands up the docs site at GitHub Pages. The slicing entry pinned this as "claim the name + first quickstart tutorial + mkdocs-material with mkdocstrings auto-rendering"; the 2026-05-21 docs-strategy entry pinned this as the docs-site start point. This entry settles the cuts.

### What gets published (and what doesn't)

| Surface | 0.1.0 | Rationale |
|---|---|---|
| **`sadda` on PyPI** | **Yes** | The Phase 1 deliverable. Python library is what real users will touch first. |
| `sadda-engine` on crates.io | **No** | Rust API not stable enough to commit. Republishing later under a different name (`sadda-engine` → maybe `sadda-rs`?) is still on the table. |
| `sadda-python` on crates.io | **No** | Build-only crate; no value in publishing the maturin shim. |
| Desktop app binaries | **No** | Phase 2 scope. |
| UniFFI bindings | **No** | Phase 8 scope. |
| docs.rs Rust API ref | **No** | Auto-published only when a crate is on crates.io; with no crate published, no docs.rs page. The "cargo-doc → markdown bridge" the docs-strategy entry mentioned is **dropped from G12** — `///` docs ship with the source for now; docs.rs comes for free once we publish `sadda-engine`. |
| **mkdocs-material site on GitHub Pages** | **Yes** | The docs-strategy entry pinned this as the 0.1 release deliverable. |

Cutting the Rust crate publish removes the largest unknown from G12: the engine's `Project` / `LiveSession` / etc. APIs have not had real outside users yet; locking them by SemVer at this stage would force breaking-change pain we can sidestep entirely.

### Wheel matrix

12 wheels via `cibuildwheel` in GitHub Actions:

- **Linux x86_64**: Python 3.10 / 3.11 / 3.12 / 3.13 — `manylinux_2_28`
- **macOS arm64**: Python 3.10 / 3.11 / 3.12 / 3.13
- **Windows x86_64**: Python 3.10 / 3.11 / 3.12 / 3.13

Excluded from 0.1.0:

- Linux aarch64 — cibuildwheel can build via QEMU but doubles CI time; deferred to 0.1.x if real ARM-server users surface.
- macOS x86_64 (Intel) — Apple has discontinued Intel Macs; arm64-only matches what Apple sells today.
- Python 3.9 — past the practical EOL window; covered by sdist for stragglers.
- musllinux — Alpine users get sdist.

PyPI also ships an sdist alongside the wheels so any unsupported platform can still `pip install sadda` and compile locally.

### Trusted publishing (no PyPI API token in CI)

PyPI's trusted-publishing flow uses OIDC: a GitHub Actions job authenticates to PyPI by presenting its OIDC token, and PyPI verifies the repo + workflow + environment match a pre-registered trust policy. **No long-lived `PYPI_API_TOKEN` secret in the repo's GitHub settings**, which closes the most common exfil-and-supply-chain vector for OSS Python packages. The user-side setup is one PyPI form fill at <https://pypi.org/manage/account/publishing/>.

### Docs surface for 0.1.0

```
docs/
├── index.md                       # Landing page
├── quickstart.md                  # One end-to-end walk-through
├── lossiness/
│   ├── textgrid.md                # D1's lossiness table, surfaced
│   └── eaf.md                     # D2's lossiness table, surfaced
└── api/
    ├── corpus.md                  # ::: sadda.Project / Bundle / Tier / …
    ├── dsp.md                     # ::: sadda.dsp
    ├── live.md                    # ::: sadda.live
    └── recipe.md                  # ::: sadda.recipe
```

Concept guides per module are deferred to 0.1.x — the API ref is generated for free from the `.pyi` stubs via mkdocstrings, so the "guides" tier is the only place we'd write fresh prose, and the time is better spent in the quickstart for 0.1.0.

DEVLOG.md stays in the repo (not republished as docs) — it's a working log, not user-facing. The handful of design decisions users will care about (lossiness on round-trip, stability tiers, etc.) get pulled into the docs site individually.

### Trusted-publishing handoff

Things only the user can do, listed once in the commit body so the release can actually happen:

1. **Claim `sadda` on PyPI.** Create an account if needed; reserve the project name.
2. **Configure trusted publishing.** PyPI account settings → "Pending publishers" → add a publisher for `sadda-speech/sadda` / `release.yml` / environment `pypi`. Trust policy enforces that only this exact workflow can publish.
3. **Activate GitHub Pages.** Repo settings → Pages → source = "GitHub Actions". The `docs.yml` workflow targets the `gh-pages` deployment environment.
4. **Push the prepared branch to main + tag v0.1.0.** The `release.yml` workflow fires on tag push, builds wheels via cibuildwheel, and uploads to PyPI via trusted publishing.
5. **Sanity-check the Pages URL** (`https://sadda-speech.github.io/sadda/`) once the docs workflow finishes.

### Confirmed G12 decisions

| Item | Decision | Reasoning |
|---|---|---|
| What gets published | **Python `sadda` only; no Rust crates** | Lets us iterate on the Rust API without SemVer pressure |
| Wheel matrix | **Linux + macOS arm64 + Windows × Py 3.10–3.13** | 12 wheels covers ~98% of Python users; sdist catches the rest |
| Build orchestration | **cibuildwheel inside `actions/setup-python`** | The de-facto cross-platform Python wheel builder; first-class maturin support |
| PyPI auth | **Trusted publishing (OIDC), no API token** | Closes the most common supply-chain risk for OSS Python; one-form user setup |
| Docs framework | **mkdocs-material + mkdocstrings (Python handler)** | Pinned in the 2026-05-21 docs-strategy entry; reads `.pyi` stubs we already generate |
| Docs scope | **Quickstart + auto-API ref + lossiness pages** | Concept guides deferred to 0.1.x — auto-ref + one tutorial is the minimum viable docs site |
| Docs hosting | **GitHub Pages, deployed from `docs.yml`** | Free; familiar; pinned in the docs-strategy entry |
| Version bump | **0.0.0 → 0.1.0 in `Cargo.toml` + `pyproject.toml`** | Two version strings; CHANGELOG.md drafted from DEVLOG slice headers |
| Release notes | **`CHANGELOG.md` in repo root, sectioned by version** | Standard; pulled into GitHub Release body via the workflow |
| Parselmouth migration guide | **Deferred to 0.1.x or later** | Real artifact for users coming from Praat; needs real content; cut from G12 |
| docs.rs Rust API ref | **Deferred** | Comes for free when we publish the engine crate; not blocking 0.1.0 |

### Layout

- `Cargo.toml`, `pyproject.toml` — version bumps.
- `CHANGELOG.md` (new) — sectioned by version, populated from the DEVLOG slice list.
- `README.md` — refresh status section (drop "pre-alpha", point at `pip install sadda`).
- `mkdocs.yml` (new) — material theme + mkdocstrings python handler.
- `docs/` (new) — index, quickstart, lossiness pages, api pages.
- `.github/workflows/release.yml` (new) — cibuildwheel + trusted publishing on tag push.
- `.github/workflows/docs.yml` (new) — mkdocs build + Pages deploy on main push.
- `pyproject.toml` `[project.urls]` — homepage + docs + repo + issues URLs.

### Lossiness / what G12 deliberately doesn't ship

- **Concept guides per module.** Auto-API ref + quickstart only.
- **Parselmouth migration guide.** Real content needs real Parselmouth-user feedback; defer.
- **Rust crate publish + docs.rs.** Let the engine API breathe; revisit when there's a downstream user.
- **Linux aarch64 / macOS Intel wheels.** sdist covers them; revisit if real users ask.
- **Python 3.9 wheels.** Past practical EOL; sdist covers stragglers.
- **musllinux (Alpine) wheels.** Same.
- **GitHub Release auto-publishing of wheels.** Wheels live on PyPI; the GitHub Release body links there. Cuts a moving part.
- **Versioned docs (multi-version docs site).** mkdocs-material supports it via `mike`; deferred until a real second version ships.
- **Search-engine optimisation on the docs site.** Out of scope at 0.1.0.

### What this entry doesn't decide

- **Whether to keep `0.0.0` on the unpublished Rust crates** post-release — the workspace bump moves all of them to 0.1.0; if a later slice splits the engine versioning from the Python wrapper, that's a future decision.
- **CHANGELOG format** — Keep-a-Changelog style is conventional; can swap if a real downstream tool needs differently.
- **PR preview docs.** Read-the-Docs has them out of the box; GitHub Pages requires extra setup. Defer.
- **Whether to ship a `sadda` GitHub org logo / favicon.** Nice-to-have; not in 0.1.0.

### Sources / references

- 2026-05-21 documentation strategy entry (G12 deliverable; mkdocs-material + GH Pages choice)
- 2026-05-21 Phase 1 slicing entry (G12 row)
- 2026-05-18 Python API surface entry (stability tiers + Parselmouth migration commitment)
- mkdocs-material: <https://squidfunk.github.io/mkdocs-material/>
- mkdocstrings (Python handler): <https://mkdocstrings.github.io/python/>
- cibuildwheel: <https://cibuildwheel.pypa.io/>
- PyPI trusted publishing: <https://docs.pypi.org/trusted-publishers/>
- maturin's release-action docs: <https://www.maturin.rs/distribution.html>

---

## 2026-05-22 — Recipes (F1): `sadda.recipe.record` context manager, recipe_run SQL table, `.py` export at block exit

Goal: settle the eleventh Phase 1 slice. F1 lands the reproducibility primitive — a `with sadda.recipe.record(project, name="..."):` block that links every `processing_run` row written inside it to a named `recipe_run`, plus a generated `<project>/recipes/<name>.py` script at block exit. The user's runnable artifact is the `.py`; the SQL rows are the audit linkage so anyone inspecting the corpus later can answer "which recipe produced this tier?"

### What F1 must deliver

From the Phase-1 slicing entry: "Context manager that logs every analysis call through `ProcessingRun` + `AuditLog`; serializes a recipe as a Python script the user can re-run on the same project or another." The slicing entry explicitly left the serialization format open ("whether the persistent log is JSON, TOML, or SQL rows is internal"). This entry settles it: **SQL rows in `recipe_run` + a generated `.py` script**.

### Capture scope (scoped down from the literal slicing-entry text)

The B1 schema's `processing_run.recipe_run_id` column already exists; F1's job is to populate it. F1 captures only the **three current `processing_run` writers**:

- `Project::import_textgrid` (slice D1)
- `Project::import_eaf` (slice D2)
- `Project::commit_recording` (slice E1)

That covers everything currently writing `processing_run` rows. Pure-DSP calls (`sadda.dsp.f0`, `sadda.dsp.formants`, ...) and corpus mutations that don't currently write `processing_run` rows (tier creates, annotation inserts) are **not captured** in v1. Rationale:

- Pure-DSP work is reproducible from the user's own `.py` script; sadda doesn't need to re-record it.
- Adding `processing_run` rows to dense-tier sidecar writes (`write_continuous_numeric` etc.) is a reasonable extension and is noted as a 0.1.x follow-up. Doing it inside F1 would expand the slice past the cadence target.
- "Every analysis call" in the slicing-entry text reads literally, but means "every `processing_run`-shaped call" once you look at the schema: corpus mutations like `add_interval` have no `parameters` / `outputs` to record under the `processing_run` schema.

### Prior art

| Tool | What it does | What we lift |
|---|---|---|
| **Praat** | GUI "Record" macro: every menu action appends to a `.praat` script the user can re-run | The user-facing artifact is a script, not a UI replay. We adopt the same model. |
| **MLflow** | `mlflow.start_run()` context manager logs params/metrics/artifacts to a side store; user re-runs the original notebook for replay | Context-manager shape; side store of provenance. We don't ship the metrics surface — `processing_run` already covers it. |
| **Jupyter `nbconvert --to script`** | Converts a notebook to a `.py` after the fact | Generation timing: the `.py` is emitted at block exit, not built incrementally. |
| **scikit-learn `Pipeline`** | Declarative pipeline objects you `.fit()` / `.predict()` | Doesn't fit — we're imperative. |

### Schema (V7 migration)

```sql
CREATE TABLE recipe_run (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    sadda_version TEXT    NOT NULL,
    parameters    TEXT,        -- JSON; user-supplied kwargs to record()
    started_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at  TEXT,        -- set on `__exit__`, NULL if the block panicked
    status        TEXT    NOT NULL DEFAULT 'in_progress' CHECK (
        status IN ('in_progress', 'ok', 'error')
    ),
    error_message TEXT,
    UNIQUE (name)
);
CREATE INDEX idx_recipe_run_name ON recipe_run(name);
```

Plus the standard three audit triggers (insert / update / delete) mirroring V3's pattern.

`processing_run.recipe_run_id` (already in V3) is the FK; no schema change there.

### Active-recipe mechanism

`Project` gains a `Cell<Option<i64>>` field carrying the active recipe id. Three methods:

```rust
impl Project {
    pub fn start_recipe(&self, name: &str, params_json: Option<&str>) -> Result<i64>;
    pub fn end_recipe(&self, recipe_run_id: i64, status: &str, error: Option<&str>) -> Result<()>;
    pub fn current_recipe_id(&self) -> Option<i64>;
    fn set_current_recipe_id(&self, id: Option<i64>);  // pub(crate) helper
}
```

The three existing `processing_run` writers pull from `current_recipe_id()` and include the value in their INSERT. `Cell` works because `Project` is single-threaded by design (`unsendable` on the PyO3 side; the engine `Project` holds a non-Sync `rusqlite::Connection`).

### Python surface

```python
import sadda

with sadda.recipe.record(proj, name="vowel_analysis_v1") as recipe:
    proj.import_textgrid("phones.TextGrid", bundle_id)
    proj.import_eaf("annotations.eaf", bundle_id)

# After the block:
# - recipe_run row exists with status='ok', completed_at set
# - the two import processing_run rows have recipe_run_id pointing at it
# - <project>/recipes/vowel_analysis_v1.py exists

sadda.recipe.list(proj)        # → ["vowel_analysis_v1"]
sadda.recipe.script_path(proj, "vowel_analysis_v1")  # → Path(.../recipes/vowel_analysis_v1.py)
```

The context manager:

- `__enter__`: calls `proj.start_recipe(name, params_json)`, returning a recipe id. Stores the id; assigns it as the project's active recipe.
- `__exit__`: clears the active recipe (idempotent), updates the `recipe_run` row's `completed_at` + `status` ('ok' on clean exit, 'error' with the exception's repr on non-clean), and generates the `.py` script.

Replay: the user runs `python recipes/vowel_analysis_v1.py`. The script imports sadda, opens the project (by path passed in `argv[1]` or hardcoded to the same project), and replays each operation. No programmatic `sadda.recipe.replay(...)` API in v1 — matches Jupyter's "convert and run" model and cuts ~30% of the slice's complexity.

### Generated script shape

```python
#!/usr/bin/env python3
# Auto-generated by sadda.recipe at 2026-05-22T15:42:00.123Z.
# Recipe name: vowel_analysis_v1
# Sadda version: 0.0.0
# Source project: /Users/alice/projects/vowels/
#
# Run with: python vowel_analysis_v1.py [project_path]
# Default target is the same project this recipe was recorded from.

from __future__ import annotations
import sys
from pathlib import Path

import sadda


def main() -> None:
    project_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/Users/alice/projects/vowels/"
    )
    proj = sadda.open_project(project_path)

    # processing_run #42 — import_textgrid on bundle 17
    proj.import_textgrid(Path("phones.TextGrid"), 17)
    # processing_run #43 — import_eaf on bundle 17
    proj.import_eaf(Path("annotations.eaf"), 17)


if __name__ == "__main__":
    main()
```

The generator walks `processing_run` rows linked to the recipe, ordered by `started_at`, and emits one call per row. The processor_id (`sadda.io.textgrid.import`, `sadda.io.eaf.import`, `sadda.live`) drives the dispatch table on the generator side — each processor id maps to a Python source template that takes `parameters` JSON + bundle id and emits the right call.

The bundle id is captured verbatim. If the user runs the script against a different project, the bundle ids won't match — the generated script is **most useful for replay on the original project** (re-verifying provenance). For cross-project replay, the user edits the script. Programmatic template-mode replay against a different project is a 0.1.x enhancement.

### Confirmed F1 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Capture scope | **Existing processing_run writers only** | Pure-DSP is reproducible from user's own .py; broader corpus mutations need schema work that doesn't fit this slice |
| Storage | **SQL rows + .py emitted at block exit** | SQL gives audit linkage; .py is the runnable artifact |
| Replay surface | **Run the .py yourself** | Jupyter's nbconvert model; cuts dispatch-table complexity |
| Recipe key | **Name (string), UNIQUE per project** | Simple; user picks the cadence; matches `bundle.name`/`tier.name` UNIQUE pattern |
| Active-recipe mechanism | **`Cell<Option<i64>>` on `Project`** | Project is single-threaded by design; no global / thread-local needed |
| Migration | **V7** | Sequential after E1's V6; adds `recipe_run` table + index + audit triggers |
| Status field values | **`'in_progress'`, `'ok'`, `'error'`** | Mirrors `processing_run.status` (`'ok'`, `'error'`, `'partial'`) without overloading — `in_progress` is meaningful for recipes (the `.py` is generated on transition out of in_progress) |
| Reentrancy | **One recipe at a time per project** | `start_recipe` while one is already active errors. Nesting is an open question; out of v1 scope |
| Failure handling | **Block panic → status='error', error_message=repr(exc), .py NOT generated** | Recipe rows persist for forensics; failed runs aren't replayable |
| Script destination | **`<project>/recipes/<name>.py`** | Inside the project tree; lives with the data it describes |

### Layout

- `crates/engine/migrations/V7__recipe_run.sql` — the new table + indices + audit triggers.
- `crates/engine/src/corpus/migrations.rs` — register V7; bump `engine_max_version()` to 7.
- `crates/engine/src/corpus.rs` — add the `recipe_run_id: Cell<Option<i64>>` field to `Project`; new methods `start_recipe`, `end_recipe`, `current_recipe_id`; thread the FK through the three existing INSERTs.
- `crates/engine/tests/recipes.rs` — engine integration test.
- `crates/python/src/lib.rs` — PyO3 wrappers (`start_recipe`, `end_recipe`, `list_recipes`); register a new `_native.recipe` submodule.
- `crates/python/src/recipe.rs` — script generator. Walks the linked `processing_run` rows and emits the `.py`.
- `python/sadda/recipe/__init__.py` — `record(project, name)` context manager class; `list(project)`, `script_path(project, name)` helpers; `@provisional` decorators.
- `python/tests/test_recipe.py` — end-to-end Python tests.

### Lossiness / what F1 deliberately doesn't ship

- **Capture of pure-DSP calls** — out; the user's own script is the orchestration. Discoverable as a follow-up if real users want it.
- **Capture of dense-tier sidecar writes** — out; needs to extend `write_continuous_numeric` etc. to write `processing_run` rows. 0.1.x.
- **Capture of corpus mutations without `processing_run` shape** — out; needs schema design (parameters JSON for `add_interval`? probably not useful).
- **Programmatic `sadda.recipe.replay()`** — out; user runs the `.py`.
- **Cross-project template replay** — out; user edits bundle ids in the `.py` if running elsewhere.
- **Nested recipes** — out; second `record()` while another is active errors.
- **Renaming / deleting recipes** — out; user removes the `.py` and DELETEs the row manually if needed. CLI helpers are a 0.1.x follow-up.
- **Diffing recipes** — out; `.py` files are just files, users can `diff` them.

### What this entry doesn't decide

- **Where to put the `recipes/` directory inside the project tree** — `<project>/recipes/` chosen for visibility; alternative is `<project>/.sadda/recipes/`. The exposed-by-default placement makes the artifacts feel first-class.
- **JSON shape of `recipe_run.parameters`** — caller-supplied passthrough; not interpreted by sadda. Convention can settle later.
- **Whether `recipe_run` rows should be exportable across projects** — they're plain SQLite rows; users can already `sqlite3` them out. A real export/import surface is a 0.1.x or 0.2 follow-up.
- **Per-recipe metadata** (author, notes, tags) — `parameters` JSON is the escape hatch; first-class columns can come later.

### Sources / references

- 2026-05-21 Phase 1 slicing entry (F1 row + the format-open item)
- 2026-05-18 Python API surface entry (`sadda.recipe.record` shape; PROVISIONAL tier)
- 2026-05-20 ML-model-registry entry (ProcessingRun → renamed from ModelRun; `recipe_run_id` FK lands in V3)
- 2026-05-21 B1 entry (audit log + ProcessingRun)
- 2026-05-22 E1 entry (live recording; second `processing_run` writer)
- Praat scripting reference: <https://www.fon.hum.uva.nl/praat/manual/Scripting.html>
- MLflow tracking model: <https://mlflow.org/docs/latest/tracking.html>
- Jupyter `nbconvert --to script`: <https://nbconvert.readthedocs.io/en/latest/usage.html#convert-script>

---

## 2026-05-22 — Live recording (E1): cpal capture, rtrb ringbuffer, `.in_progress/` atomic commit, streaming pitch/formants/intensity subscribers

Goal: settle the tenth Phase 1 slice. E1 lands the live-recording surface — the second piece of new functionality after the interop slices, and the most architecturally novel (real-time audio thread + cross-thread plumbing + Python callback dispatch). The 2026-05-18 Python-API-surface entry sketched `sadda.live.start_session(...)` plus subscriber decorators (`@session.on_pitch`, `@session.on_formants`); the 2026-05-21 slicing entry punted the concrete spike "to inside E1." This entry settles it.

### What E1 must deliver

From the Phase-1 slicing entry: (1) `.in_progress/` flow → atomic commit; (2) cpal cross-platform input driver; (3) metering callbacks. Plus the live-coaching surface promised by the API-surface entry: subscriber decorators for pitch, formants, intensity. JACK is **explicitly out of scope** — already cut to a 0.1.x stretch in the slicing entry, and confirmed during this design pass.

### Architecture: three threads, two ringbuffers

Live audio capture cannot block. cpal input callbacks run on a real-time audio thread with hard deadlines (typically 5–20 ms between callbacks at 44.1 kHz); inside the callback we cannot allocate, take locks, or acquire the GIL. Standard pattern across the Rust-Audio ecosystem (Bevy audio, fundsp, samplerate-rs) is to ferry samples to a consumer thread via a lock-free SPSC ringbuffer and do all the slow work there.

```
┌──────────────────────────────────┐
│ Audio thread (cpal callback)     │
│ - receives [f32] chunks from OS  │
│ - pushes into raw-samples rtrb   │── push samples ──┐
│ - NO alloc, NO locks, NO GIL     │                  │
└──────────────────────────────────┘                  │
                                                      ▼
                                  ┌──────────────────────────────────┐
                                  │ Consumer thread (Rust)           │
                                  │ - drains raw-samples rtrb        │
                                  │ - writes hound WAV to disk       │
                                  │ - accumulates DSP window         │
                                  │ - per hop, runs pitch/intensity/ │
                                  │   formants on latest window      │
                                  │ - pushes results into per-       │
                                  │   measure rtrb queues            │
                                  └──────────────────────────────────┘
                                                      │
                                                      ▼
                                  ┌──────────────────────────────────┐
                                  │ Dispatch thread (Rust w/ GIL)    │
                                  │ - blocking-pop on result rtrbs   │
                                  │ - Python::with_gil(|py| ...)     │
                                  │ - invokes registered callables   │
                                  │ - exits on stop_signal           │
                                  └──────────────────────────────────┘
                                                      │
                                                      ▼
                                              user's @session.on_pitch
                                              fn pitch_cb(value, time): ...
```

Splitting **consumer** from **dispatch** matters: consumer is alloc-free Rust (cheap to run continuously); dispatch holds the GIL to invoke Python (the slow, serialised part). If we merged them, a slow Python callback would stall WAV writing and cause sample drops.

### Crate choices

| Concern | Choice | Why |
|---|---|---|
| Audio I/O | **`cpal` 0.16** | Cross-platform (WASAPI / CoreAudio / ALSA / PulseAudio / JACK*); pure-Rust; the de-facto Rust audio I/O crate; used by Bevy. *JACK backend opt-in but we don't enable the feature flag in 0.1 |
| Lock-free queue | **`rtrb` 0.3** | SPSC ringbuffer the Rust-Audio working group recommends. Wait-free producer, alloc-free push. Minimal API surface (`Producer::push` / `Consumer::pop`). |
| WAV writer | **`hound` 3.5** | Already a dep (used by tests); supports streamed writes via `write_sample` in a loop |
| UUID for dirnames | **`uuid` 1** with `v4` | Standard; pure-Rust |

Three new top-level deps (`cpal`, `rtrb`, `uuid`). `hound` is already in tree.

### Session lifecycle

A `LiveSession` is created at `sadda.live.start_session(project, device=..., sample_rate=..., channels=..., name=...)`. The name eventually becomes the bundle name; if absent, a timestamp-based default is used. Construction does:

1. Validates that `signals/.in_progress/` exists; mints `<uuid4>` subdir; opens `audio.wav` for streamed write.
2. Spawns the audio thread via `cpal::Device::build_input_stream`.
3. Spawns the consumer thread (Rust).
4. Spawns the dispatch thread (Rust, holds GIL on each iteration).
5. Returns the `LiveSession` Python object.

State transitions:

```
        start_session()         stop()                  commit() | discard()
Idle ───────────────────► Recording ───────────────► Stopped ──────────────► Committed/Discarded
                              │                           │
                              └── auto-flush hound ──────►┘
```

Calling `stop()` joins all three threads (with a 1-second timeout), flushes hound, and returns control. `commit()` then does the atomic move + bundle insert + processing_run insert; `discard()` deletes the `.in_progress/<uuid>/` directory.

Why split `stop()` and `commit()`: lets the caller inspect meter/result history *after* recording but *before* deciding to keep the bundle. A `with sadda.live.start_session(...) as session:` context manager commits on clean exit and discards on exception.

### Python API

```python
session = sadda.live.start_session(
    project,
    device="default",          # or a specific device name from list_devices()
    sample_rate=44100,
    channels=1,
    name="practice_session_1",
    analysis_window_ms=30,     # frame size for streaming DSP
    analysis_hop_ms=10,        # hop between analysis frames
)

@session.on_meter
def on_meter(peak_db, rms_db, t):
    ui.update_meter(peak_db, rms_db)

@session.on_pitch
def on_pitch(f0_hz, voiced, t):
    if voiced:
        print(f"{t:.3f}s  f0={f0_hz:.1f} Hz")

@session.on_intensity
def on_intensity(intensity_db, t):
    ...

@session.on_formants
def on_formants(formants, t):
    # formants: list[float], length = num_formants requested
    ...

session.start()              # actually begins capture
time.sleep(5.0)
session.stop()
bundle_id = session.commit() # atomic: moves WAV in, inserts bundle + processing_run

# or:
with sadda.live.start_session(proj, name="x") as session:
    @session.on_pitch
    def cb(f0, voiced, t): ...
    session.start()
    time.sleep(5.0)
# commits on clean exit; discards on exception
```

Module surface in `sadda.live`:

- `start_session(project, *, device="default", sample_rate=44100, channels=1, name=None, analysis_window_ms=30, analysis_hop_ms=10) → LiveSession`
- `list_input_devices() → list[str]`
- `default_input_device() → str | None`
- `LiveSession` (subscriber decorators + lifecycle methods listed above)

All `PROVISIONAL` per the API-surface entry's stability tiers.

### Streaming-DSP plan

The consumer thread maintains a `VecDeque<f32>` of the last `window_samples` audio samples. Whenever it has advanced by `hop_samples` since the last analysis, it takes a snapshot of the most recent window and runs the DSP suite:

- **Intensity** — RMS of the window in dB-FS via `sadda_engine::dsp::intensity::rms_db` (already in C1).
- **Pitch** — autocorrelation pitch via `sadda_engine::pitch::autocorrelation` (already in Phase 0 + C2's voicing extension). Single-frame call: no buffering needed.
- **Formants** — LPC (Burg) + Aberth root-solver via `sadda_engine::dsp::formants::lpc_formants` (already in C2). Default `num_formants = 4`, `lpc_order = 2 + sample_rate / 1000` per the standard rule.

Meter (peak + RMS) is computed once per *chunk* (cpal's natural buffer size, typically ~512–2048 samples), not once per analysis hop. Faster reaction for level UIs; not a "real DSP frame."

Each measure has its own result `rtrb` (size 256 entries) so a slow Python callback for one measure doesn't back-pressure the others.

### Atomic commit flow

```
signals/
├── original/                  # committed bundles
├── derived/                   # B3 Parquet sidecars
└── .in_progress/              # NEW
    └── <uuid4>/
        ├── audio.wav          # streamed by hound
        └── meta.toml          # device, sr, channels, started_at, intent_name
```

`commit()` runs these inside a single SQLite transaction:

1. `fsync(audio.wav)` + close.
2. `std::fs::rename(.in_progress/<uuid>/audio.wav → signals/original/<sanitized_name>.wav)`. `rename` is atomic on same-filesystem POSIX moves and on Windows (via `MoveFileEx`).
3. `Project::add_bundle_with(name, audio_path)` — same path the existing static-file import uses; reuses its audit trigger on the bundle insert.
4. INSERT into `processing_run` with `kind = 'live_recording'`, `processor_id = 'sadda.live'`, `parameters = {device, sr, channels, duration_s, analysis_hop_ms, analysis_window_ms}`.
5. `std::fs::remove_dir(.in_progress/<uuid>/)` — meta.toml left behind for forensics if step 4 fails.

If any step fails, the transaction rolls back and the `.in_progress/<uuid>/` directory is left intact for inspection.

### `processing_run.kind` migration

Current `CHECK (kind IN (...))` constraint must accept `'live_recording'`. Schema migration V7 adds it. The migration:

```sql
-- V7__processing_run_kind_live_recording.sql
-- SQLite has no ALTER TABLE … CHECK; recreate the table.
CREATE TABLE processing_run__new (
    ... -- identical columns ...,
    kind TEXT NOT NULL CHECK (kind IN (
        'dsp_algorithm', 'ml_inference', 'manual_edit', 'live_recording'
    )),
    ...
);
INSERT INTO processing_run__new SELECT * FROM processing_run;
DROP TABLE processing_run;
ALTER TABLE processing_run__new RENAME TO processing_run;
-- Recreate indices + triggers.
```

This is the first non-trivial migration since B1 — exercises A1's migration framework and forces us to validate the `corpus.db.bak.<old_version>` backup path actually works on a real schema change.

### Confirmed E1 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Audio I/O crate | **`cpal` 0.16** | De-facto Rust cross-platform audio; pure-Rust on three of four backends |
| Cross-thread plumbing | **`rtrb` 0.3 SPSC** | Wait-free producer; alloc-free push; orthodox choice for RT audio in Rust |
| Thread split | **Audio + Consumer + Dispatch (3 threads)** | Keeps GIL acquisition off the consumer path so slow Python callbacks can't drop samples |
| API scope at 0.1 | **Full: recording + metering + on_pitch + on_intensity + on_formants** | Per user direction (this entry); ~1500–1800 LOC slice |
| JACK backend | **Cut from 0.1** | cpal's JACK feature flag stays off; revisit in 0.1.x |
| Bundle attachment | **Session creates the bundle on `commit()`** | No orphan bundle row if recording fails; `.in_progress/` directory is the failure-mode artifact |
| processing_run kind | **`'live_recording'` (new)** | Distinguishable from DSP/import events in audit queries; requires V7 schema migration |
| Default analysis frame | **30 ms window / 10 ms hop** | Standard for pitch + formants; matches Praat's default short-term analysis settings |
| Default sample rate | **44_100 Hz, 1 channel, f32** | Universal device support; speech-band oversampled enough for formants |
| Subscriber dispatch | **Python thread spawned by start(); joined on stop()** | Matches sounddevice convention; no asyncio in v1 (per API-surface entry) |

### Layout

- `crates/engine/Cargo.toml` — adds `cpal = "0.16"`, `rtrb = "0.3"`, `uuid = { version = "1", features = ["v4"] }`.
- `crates/engine/src/live/mod.rs` — `LiveSession`, `LiveConfig`, `SessionState` enum, plus the `pyclass`-friendly Rust API (no pyo3 deps here — just `std`).
- `crates/engine/src/live/capture.rs` — cpal stream construction, audio-thread closure, ringbuffer producer side.
- `crates/engine/src/live/consumer.rs` — consumer thread loop, WAV writer wrapper, DSP-frame scheduler.
- `crates/engine/src/live/results.rs` — per-measure result types + ringbuffer types.
- `crates/engine/src/corpus.rs` — `Project::commit_recording(uuid, name, params) → bundle_id` helper.
- `crates/engine/migrations/V7__processing_run_kind_live_recording.sql` — the CHECK-constraint migration.
- `crates/engine/tests/live_recording.rs` — integration test using a synthetic in-process producer (no cpal in CI).
- `crates/python/src/live.rs` — PyO3 `PyLiveSession`, decorator methods, dispatch thread.
- `python/sadda/live/__init__.py` — pure-Python re-exports + the context manager wrapper.
- `python/tests/test_live.py` — tests against the synthetic-producer path.

### Lossiness / what E1 deliberately doesn't ship

- **Live JACK input** — cut; revisit 0.1.x.
- **Live MIDI / control surface input** — not in scope.
- **Pre-roll buffer** — recording starts when `session.start()` returns; no "rewind 2 seconds" feature.
- **Pause / resume** — single contiguous capture per session. Pause/resume can layer in later as `Session::pause()` / `Session::resume()`; not v1.
- **Live waveform / spectrogram subscribers** — `on_meter` covers peak/RMS for level UIs; live waveform / spectrogram are deferred (would need a high-rate subscriber, separate budget). The frame data is still on disk so post-hoc plotting works fine.
- **Multi-channel analysis** — recording supports `channels >= 1`, but the DSP path runs only on channel 0. Multi-channel formants/pitch fan-out is a future enhancement.
- **Auto device-disconnect handling** — if the OS yanks the device mid-stream, the session enters an `Error` state and the caller must explicitly `discard()`. Reconnection logic is not v1.
- **Live recording into an existing bundle** — `commit()` always creates a new bundle. Replacing or appending to an existing bundle's audio is a separate operation.
- **In-app surface** — `sadda.app` integration (a live-meter widget) is Phase-2 work; E1 ships only the library API.

### What this entry doesn't decide

- **Exact ringbuffer sizes** — depends on chunk size on each platform; tunable constants. Sensible starting points: 4× `chunk_samples` for raw-samples rtrb; 256 entries for each result rtrb.
- **Behaviour on overrun / underrun** — if the consumer can't keep up (e.g. disk stalled), we count drops and surface a `session.dropped_chunks` property. Whether to also raise an error after a threshold is TBD; safe default for v1 is "count but continue."
- **Exact `meta.toml` schema** — minimum is device / sr / channels / started_at / intent_name; can grow as forensics workflows surface.
- **Whether `LiveSession` is reusable** — v1 ships single-use (one session → one bundle). Reusable sessions can layer in later if a workflow needs them.
- **`PROVISIONAL` decorator coverage** — every new public Python entry point in `sadda.live` is marked `@provisional` per the A2 contract. The exact warning text is settled inside A2's helpers.

### Sources / references

- 2026-05-21 Phase 1 slicing entry (E1 row + the live-recording UX open item)
- 2026-05-18 Python API surface entry (`sadda.live` namespace, subscriber decorators, sync-by-default async stance)
- 2026-05-18 corpus data-model entry (processing_run shape; bundle insert semantics)
- 2026-05-21 C1 entry (intensity / windowing primitives reused here)
- 2026-05-21 C2 entry (pitch with voicing; LPC formants reused here)
- `cpal`: <https://github.com/RustAudio/cpal>
- `rtrb`: <https://github.com/mgeier/rtrb>
- The Rust-Audio working group's "real-time safety in audio threads" notes: <https://github.com/RustAudio/cpal#real-time-safety>
- sounddevice (Python; closest API analogue): <https://python-sounddevice.readthedocs.io/>

---

## 2026-05-22 — EAF round-trip (D2): quick-xml, EAF 2.8 target, tier hierarchy via PARENT_REF, points as degenerate alignables

Goal: settle the ninth Phase 1 slice. D2 lands ELAN .eaf import + export — the second-tier interop after Praat TextGrid. The crucial difference from D1: **tier hierarchy survives the round-trip** (EAF natively models `PARENT_REF`).

### What D2 must deliver

From the Phase-1 slicing entry: (1) ALIGNABLE_ANNOTATION, TIME_SUBDIVISION, SYMBOLIC_ASSOCIATION tier types; (2) `parent_tier_ref` preserved; (3) XML round-trip stable enough that ELAN can re-open exports without warnings.

### Library: quick-xml

EAF is XML-based and substantially richer than TextGrid. `quick-xml` 0.40 is the canonical Rust XML reader/writer pair (MSRV 1.56 — fine for our 1.85). Pure Rust, ~5 transitive deps, both pull-parser (events) and builder writer in one crate. Used across the Rust XML ecosystem.

### EAF 2.8 target on write; permissive on read

Write emits `FORMAT="2.8"`; this is the widely-supported version (ELAN 5.0+ reads it transparently and is still common in field linguistics). EAF 3.0 added external CV references we don't use. The parser is permissive on FORMAT — accepts 2.7, 2.8, 3.0.

### Tier-type mapping

The headline feature: tier hierarchy round-trips losslessly via EAF's `PARENT_REF`. Mapping:

| Our tier type | EAF mapping |
|---|---|
| `interval` (no parent) | `ALIGNABLE_ANNOTATION` tier; `LINGUISTIC_TYPE` with no CONSTRAINTS |
| `interval` (parent = interval) | `ALIGNABLE_ANNOTATION` tier; `LINGUISTIC_TYPE` constraint = `Included_In`; `PARENT_REF` set |
| `point` (no parent) | `ALIGNABLE_ANNOTATION` tier; each point becomes a degenerate `[t, t + 1ms]` annotation. On import: a tier where every annotation has `end - start ≤ 2ms` is recovered as a `point` tier |
| `point` (with parent) | Same degenerate-alignable convention; `PARENT_REF` preserved |
| `reference` (with parent) | `REF_ANNOTATION` tier; `LINGUISTIC_TYPE` constraint = `Symbolic_Association`. Sentinel-encoded `(target_kind, target_id)` lives in the value |
| `continuous_*` | **Skipped silently** on export (no EAF analogue beyond heavy EXTERNAL_REFs) |

The ≤2ms heuristic on import is the standard way to recover point semantics from EAF — annotators don't naturally make sub-millisecond intervals, and the `1ms` width on export is small enough that ELAN renders the annotation visually as a vertical line.

### JSON sentinel

Reused from D1: `<label> {json:<inline-json>}` suffix. Plain EAFs without sentinels round-trip cleanly. ELAN displays the sentinel as part of the annotation value; users see and don't manually edit it.

### Drop ELAN-specific metadata

ELAN files often carry `CONTROLLED_VOCABULARY`, `LANGUAGE`, `LICENSE`, `EXTERNAL_REF`, `LEXICON_REF`, `REF_LINK_SET`, `LOCALES`. These don't fit our model. **D2 drops them on round-trip** and documents the loss. Re-importing an exported EAF won't have the user's CV definitions, language tags, or license metadata.

Preserving them opaquely would need a per-tier `extra_xml` column (schema migration), opaque-XML retention logic, and substantially more parser complexity. That's a future enhancement; tracking task TBD if real users request it.

### Project API

```rust
impl Project {
    pub fn import_eaf(&self, path: impl AsRef<Path>, bundle_id: i64) -> Result<Vec<i64>>;
    pub fn export_eaf(
        &self,
        bundle_id: i64,
        path: impl AsRef<Path>,
        tier_ids: Option<&[i64]>,
    ) -> Result<()>;
}
```

Mirror D1's TextGrid API exactly. Import records a `processing_run` row of kind `dsp_algorithm` with `processor_id = "sadda.io.eaf.import"`.

### Confirmed D2 decisions

| Item | Decision | Reasoning |
|---|---|---|
| XML library | **`quick-xml` 0.40** | Pure Rust; MSRV 1.56; both reader + writer in one crate; minimal deps |
| EAF FORMAT on write | **2.8** | Widely supported by ELAN 5.0+; 3.0-only features unused |
| Point-tier mapping | **Degenerate `[t, t+1ms]` alignable; ≤2ms heuristic on import** | Round-trips losslessly between sadda projects; ELAN displays cleanly |
| ELAN-specific metadata | **Dropped on round-trip; documented loss** | Preserving opaquely needs schema migration + retention logic; deferred |
| Tier hierarchy | **Preserved via `PARENT_REF` ↔ `tier.parent_id`** | The headline D2 feature; EAF natively models this |
| Reference-tier mapping | **`REF_ANNOTATION` + `Symbolic_Association` linguistic type + JSON sentinel for target** | Round-trips through ELAN losslessly; sentinel survives any user edit |
| API | **Methods on `Project` only** | Mirrors D1; corpus-first |
| Import provenance | **`processing_run` row of kind `dsp_algorithm`** | Reuses B1 audit infrastructure (same as D1) |

### Layout

- `crates/engine/Cargo.toml` — adds `quick-xml = "0.40"`.
- `crates/engine/src/io/eaf.rs` — parser + writer + `EafFile` in-memory struct.
- `crates/engine/src/io/mod.rs` — `pub mod eaf;` added.
- `crates/engine/src/corpus.rs` — `Project::import_eaf` / `Project::export_eaf` methods.
- `crates/python/src/lib.rs` — PyO3 method wrappers on `PyProject`.
- `crates/engine/tests/eaf_round_trip.rs` — integration tests.
- `python/tests/test_eaf.py` — end-to-end Python tests.

### Lossiness — what EAF drops

Documented in the module + this entry:

- **Controlled vocabularies** (CV_ENTRY, CV_REF on annotations) — not modeled
- **Languages / locales** (LANGUAGE, LOCALE elements)
- **Licenses** (LICENSE, AUTHOR, DATE attributes on ANNOTATION_DOCUMENT)
- **External refs** (EXTERNAL_REF on tier or annotation)
- **Lexicon refs** (LEXICON_REF in HEADER)
- **Stereotypes beyond the three named** (Time_Subdivision is mapped, but Symbolic_Subdivision and Included_In with non-trivial constraints are simplified)
- **Annotation IDs not from our system** — on re-import, we mint fresh `annotation_<n>` IDs; the original IDs are lost
- **Media file references** in HEADER — we emit a placeholder pointing at the bundle's audio; users editing in ELAN will see our path, not whatever they had before

What's recoverable via the JSON sentinel:
- Annotation `extra` JSON (any tier type)
- Reference-tier target `(target_kind, target_id)`

### Implementation notes from the slice

- **`cardinality = "none"` semantics fix.** `Project::enforce_cardinality` previously errored out before checking the cardinality, requiring `parent_annotation_id` whenever the child tier had a `parent_id`. That made `"none"` (which the match arm groups with the "no constraint" branch) effectively dead. Relaxed so `"none"` (or `None`) allows `parent_annotation_id = None` — this is the mechanism `import_eaf` uses to reconstruct tier-level hierarchy without recovering annotation-level parentage. The existing `cardinality_requires_parent_annotation_id_when_parent_tier_set` test (which uses `"one_to_many"`) still passes; the change is scoped to `"none"`.
- **XML entity-ref stitching in the parser.** `quick-xml` 0.40 fires `Event::GeneralRef` for `&quot;` / `&amp;` / `&lt;` / `&gt;` / `&apos;` / `&#N;` as a *separate* event from the surrounding `Event::Text` chunks. The writer's `BytesText::new` escapes inner `"` characters inside JSON sentinels, so the parser has to handle the entity-ref events explicitly — resolving the predefined names inline and using `BytesRef::resolve_char_ref` for numeric refs — and concatenate the resolved chars into the in-progress annotation value. Without this stitching, `{"v":1}` round-trips as `{v:1}` (quotes silently dropped).

### What this entry doesn't decide

- **Opaque preservation of ELAN-specific metadata** — deferred until a real workflow needs it. Would add `tier.extra_xml` column.
- **CONTROLLED_VOCABULARY round-trip** — same; CV semantics tie into our annotation-value validation story which doesn't exist yet.
- **`Time_Subdivision` and `Symbolic_Subdivision` stereotypes** — mapped to our flat hierarchy without preserving the subdivision constraint. A future slice could add a tier-stereotype field.
- **Free `sadda.io.eaf.read/write` functions** — same scope cut as D1.
- **Pympi-style mutation API** (`Eaf.add_tier(...)`) — not in v1; mutations happen via `Project`.

### Sources / references

- 2026-05-21 Phase 1 slicing entry (D2 row)
- 2026-05-18 corpus data-model entry (interop section, EAF row)
- 2026-05-22 D1 entry (TextGrid; mirrors the API shape)
- ELAN EAF format documentation: <https://www.mpi.nl/tools/elan/EAF_Annotation_Format_3.0_and_ELAN.pdf>
- ELAN tier stereotypes reference: <https://www.mpi.nl/corpus/html/elan/ch02s02s05.html>
- `quick-xml`: <https://github.com/tafia/quick-xml>
- `pympi` (API-shape precedent): <https://github.com/dopefishh/pympi>

---

## 2026-05-22 — TextGrid round-trip (D1): hand-rolled parser, long+short read, long-text write, JSON-sentinel suffix

Goal: settle the eighth Phase 1 slice. D1 lands Praat TextGrid import + export — the adoption hinge for users coming from Praat. Per the 2026-05-18 corpus data-model entry, TextGrid is "deliberately lossy"; what D1 commits to is which losses are explicit, which are recoverable via a JSON sentinel, and which trigger errors.

### What D1 must deliver

From the Phase-1 slicing entry: (1) IntervalTier + TextTier import + export; (2) JSON sentinel for attribute round-trip; (3) explicit lossiness documentation.

### Format coverage: read long+short text; write long text

Praat writes three TextGrid formats:
- **long text** (default; verbose, line-by-line, human-readable)
- **short text** (compact; same structure, fewer keywords)
- **binary**

Read covers long+short; write produces long text only. Binary is deferred (rare in research workflows; subtler parsing rules). Encoding is UTF-8 only at v1 (UTF-16 deferred).

### Hand-rolled parser

`engine::io::textgrid` is a new module with a tokeniser + line-based state machine — ~250 LOC, no `nom`/`pest` deps. The format is line-oriented and unambiguous; both long and short variants share enough structure that one parser handles both via a "skip optional keywords" pass.

### JSON-sentinel: suffix after a space

Our annotations carry more fields (`extra` JSON; `parent_annotation_id`) than Praat's TextGrid stores. To round-trip the JSON `extra` through Praat, the exporter encodes it as a suffix on the label:

```
<label> {json:<inline-json>}
```

Examples:
- Plain: `"hello"` with no extra → `text = "hello"`
- With extra: `"hello"` with `extra = {"foo": 1}` → `text = "hello {json:{\"foo\":1}}"`
- Empty plain text with extra: `""` with `extra = {"k":"v"}` → `text = " {json:{\"k\":\"v\"}}"`

Recovery on re-import is a non-greedy regex against the suffix:

```regex
^(.*?)(?:\s\{json:(.*)\})?$
```

This wins over prefix/replace because:
- Plain TextGrids with no sentinel round-trip cleanly (no leading whitespace surprises)
- Users see the sentinel and know data exists they shouldn't edit by hand in Praat
- The opening `{json:` and trailing `}` make it grep-friendly

### Mapping to our tier model

| Our tier type | TextGrid mapping |
|---|---|
| `interval` | `IntervalTier` (direct) |
| `point` | `TextTier` (Praat's name for point tier; `points` keyword) |
| `reference` | `IntervalTier` with degenerate `[0.0, 0.001]` time span and the JSON sentinel carrying `(target_kind, target_id)`. Round-trips losslessly through Praat |
| `continuous_numeric` / `continuous_vector` / `categorical_sampled` | **Skipped** on export (no TextGrid analogue). A future API may report the skipped tier count |

Praat's empty-label convention (`text = ""`) imports as `label = ""` (preserves verbatim). On export, our `label = None` writes as `text = ""`.

### Project API

```rust
impl Project {
    /// Reads `path`, creates new Tier rows attached to `bundle_id`, inserts
    /// annotation_interval / annotation_point / annotation_reference rows.
    /// Returns the new tier IDs.
    pub fn import_textgrid(&self, path: impl AsRef<Path>, bundle_id: i64) -> Result<Vec<i64>>;

    /// Writes all sparse tiers of `bundle_id` (or the subset in `tier_ids`)
    /// to `path` as long-text TextGrid. Dense tiers are skipped.
    pub fn export_textgrid(
        &self,
        bundle_id: i64,
        path: impl AsRef<Path>,
        tier_ids: Option<&[i64]>,
    ) -> Result<()>;
}
```

Import records a `processing_run` row of kind `dsp_algorithm` with `processor_id = "sadda.io.textgrid.import"` for audit trail / provenance — re-using B1's audit infrastructure. (Export does not log a processing_run; it's a read-only snapshot of existing tier data and the corpus state is unchanged.)

### Confirmed D1 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Format support | **Read long+short text; write long text** | Matches Praat's defaults; covers ~99% of real-world files; binary deferred |
| Parser | **Hand-rolled (~250 LOC), UTF-8 only** | No `nom`/`pest` dep; format is line-oriented and unambiguous |
| JSON-sentinel placement | **Suffix: `<label> {json:<inline-json>}`** | Plain TextGrids round-trip cleanly; visible to users; grep-friendly |
| `text = ""` ↔ `label = ""` | **Preserve verbatim both directions** | Round-trip fidelity; no silent null↔empty conversion |
| Reference-tier export | **IntervalTier at `[0.0, 0.001]` with JSON sentinel** | Round-trips losslessly through Praat without losing the reference target |
| Dense tier export | **Skipped silently** | No TextGrid analogue; explicit skip-count reporting deferred |
| API | **Methods on `Project` only** | Corpus-first; pure-file `sadda.io.textgrid.read/write` deferred unless real users need it |
| Import provenance | **`processing_run` row of kind `dsp_algorithm`** | Re-uses B1's audit infrastructure; pins what produced these tiers in the corpus |

### Layout

- `crates/engine/src/io/mod.rs` — new `engine::io` module.
- `crates/engine/src/io/textgrid.rs` — parser + writer + `TextGridFile` in-memory struct + JSON-sentinel codec.
- `crates/engine/src/corpus.rs` — `Project::import_textgrid` / `Project::export_textgrid` methods.
- `crates/python/src/lib.rs` — PyO3 method wrappers on `PyProject`.
- `crates/engine/tests/textgrid_round_trip.rs` — integration tests covering fixture parsing, round-trip stability, JSON sentinel, reference-tier round-trip.
- `python/tests/test_textgrid.py` — end-to-end Project-level tests.

### Lossiness — what TextGrid drops

Documented in the module + this entry so users see the warnings before they bake export → external-edit → import into their workflow:

- **Tier hierarchy** (`Tier.parent_id`, `Tier.cardinality`) — Praat has no parent-tier concept; lost on export, re-import creates flat tiers.
- **Per-annotation parent links** (`annotation.parent_annotation_id`) — lost.
- **Tier `schema` JSON** — lost; tier-level metadata Praat doesn't model.
- **Dense tier sidecars** — skipped entirely.
- **Audit history of the source bundle** — TextGrid carries no provenance; the import creates a fresh `processing_run` row but does not preserve the original `audit_log` chain.

What's recoverable via the JSON sentinel:
- Annotation `extra` JSON
- Reference-tier target (`target_kind`, `target_id`) — when re-imported, the reference is reconstructed in our model

### What this entry doesn't decide

- **Binary TextGrid format** — deferred. Add when a real user needs it.
- **UTF-16 encoding** — deferred. Most modern TextGrids are UTF-8; some old ones aren't. Decode on demand later.
- **Pure-file `sadda.io.textgrid.read/write`** — deferred until real users need a corpus-less workflow.
- **Lossy report on export** (number of skipped dense tiers, lost hierarchy edges) — would be a return value `ExportReport { skipped_dense, lost_parents }` ; out of scope for D1.
- **Diff-mode import** (merge with existing tiers rather than create new ones) — explicitly rejected by the corpus-model entry's "export is a snapshot, not a sync" boundary.

### Sources / references

- 2026-05-21 Phase 1 slicing entry (D1 row)
- 2026-05-18 corpus data-model entry (interop section, "deliberately lossy" framing)
- Praat TextGrid format manual: <https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html>
- praatio (Python TextGrid library, API-shape precedent): <https://github.com/timmahrt/praatIO>

---

## 2026-05-21 — DSP method diversity (project design principle)

Goal: codify a project-wide commitment for the DSP namespace. Raised mid-C2 as a course correction; this entry captures it as a durable design principle.

### The principle

Two coupled commitments for every member of `sadda.dsp.*` (and the underlying `engine::dsp` / `engine::pitch` modules):

1. **Cite the canonical source.** Every public DSP function carries at least one bibliographic reference in its doc comment — typically a paper, textbook section, or canonical implementation (Praat manual / `scipy.signal` / `librosa` source code all count).
2. **Provide multiple non-equivalent methods where they exist.** Domain-norm in speech research is that "the formant tracker" or "the pitch tracker" is shorthand for one of several non-equivalent methods, each with known strengths. Providing alternates lets users compare methods on their own data; defaulting to one without naming alternatives obscures algorithmic choice.

### Why this matters here

- **Phoneticians know the alternates by name.** Praat's `Sound.to_formant_burg` vs `Sound.to_formant_burg_robust` vs autocorrelation-LPC formants are *different* methods producing measurably different tracks — not a numerical-precision difference. Users coming from Praat will look for both.
- **AI/ML users expect citations to be discoverable from API docs.** `help(sadda.dsp.mfcc)` should answer "what mel scale? — Davis & Mermelstein (1980), HTK convention; Slaney (1998) deferred" without context-switching to external references.
- **Forensic / clinical use cases need auditable algorithmic choices.** Citation + method name is the minimum to defend a measurement in court or in a clinical decision.
- **Multiple methods make the namespace honest about what it is.** Pitch tracking is a research area, not a solved problem. Naming the method (`autocorrelation_pitch`, `yin_pitch`, …) instead of `pitch` keeps the API honest.

### What "non-equivalent" means

Two methods are non-equivalent if they can produce *measurably different outputs* on the same input under reasonable parameter choices. Examples:

| Family | Method A | Method B | Difference |
|---|---|---|---|
| LPC | Autocorrelation | Burg | Stability on short frames; Burg is Praat's formant-tracking default |
| Pitch | Autocorrelation | YIN | YIN handles octave errors better; cited in millions of papers |
| Pitch | Heuristic (autocorrelation, YIN, RAPT, SWIPE) | Neural (CREPE) | Substantially different cost & accuracy profiles |
| Mel scale | HTK | Slaney | Different filter-bank cutoff conventions; ~5% energy-distribution shift |
| Window | Symmetric | Periodic | Symmetric for filter design; periodic for STFT (librosa's `sym` flag) |
| Spectrogram | Magnitude | Power | `|X|` vs `|X|²`; users from MATLAB world expect both |

If two methods only differ by floating-point precision or implementation detail, they are *not* non-equivalent and don't need separate functions.

### How to apply (operational rules)

- **Doc-comment citations are required** on every public DSP function. Format: short author/year + a publication link whenever one exists (DOI preferred; arXiv / JSTOR / publisher URL / archived tech-report URL otherwise). Older textbooks / book chapters without a stable URL get cited by ISBN and chapter section. The Praat manual, `scipy.signal` source, and `librosa` source code count as canonical references for well-known methods, with their stable doc URLs included.
- **Multi-method exposure scope:**
  - **Cheap flag (`power=True/False`, `sym=True/False`)**: ship alongside the sibling in the same slice.
  - **Substantial implementation (YIN tracker, Slaney mel scale with full filter recomputation)**: a new function in a later slice; tracked in a `### Deferred alternates` section of the slice's DEVLOG entry.
- **Document the chosen variant when not multi-method.** If C2's MFCC uses HTK mel + pre-emphasis + log (not dB), the doc comment says so + cites Davis & Mermelstein (1980) and notes Slaney (1998) as a known alternative.
- **DEVLOG slice entries for DSP work include a `### Deferred alternates` section** enumerating known non-equivalent methods not shipped yet. This is the running development-target list.
- **Function naming when multi-method**: prefer method-suffixed names (`autocorrelation_pitch`, `yin_pitch`) over polymorphic flags (`pitch(method="yin")`) when each method has substantial code behind it. Flags are fine when the method choice is a small algorithmic variant of a shared implementation.

### Application to C2 and back to C1

- **C2 ships both autocorrelation LPC and Burg's method** (`engine::dsp::lpc::autocorr_lpc` and `engine::dsp::lpc::burg_lpc`), selectable via an `LpcMethod` enum at the formant-tracker boundary. Burg is the default for formants (Praat convention).
- **C2 ships citations on every new DSP function**: LPC (Markel & Gray 1976), formants (Markel 1972; McCandless 1974), MFCC (Davis & Mermelstein 1980), Aberth roots (Aberth 1973; Bini 1996), voicing (Boersma 1993 for the autocorrelation-peak ratio).
- **C1 citations are backfilled** in the same slice: windows (Harris 1978; Kaiser 1980; scipy.signal.windows source), STFT (Oppenheim & Schafer §11), intensity (RMS basics; Boersma & Weenink Praat manual for intensity-object conventions).
- **Deferred alternates list** (tracked across slices; not blockers for v1):
  - YIN pitch (de Cheveigné & Kawahara 2002)
  - RAPT pitch (Talkin 1995)
  - SWIPE / SWIPE' pitch (Camacho & Harris 2008)
  - CREPE pitch (Kim et al. 2018; neural)
  - pyin pitch (Mauch & Dixon 2014)
  - Slaney mel scale (Slaney 1998)
  - Magnitude spectrogram (alongside the current power-only)
  - Symmetric/periodic window toggle
  - Burg-robust LPC (Praat's Robust variant)
  - dB-SPL intensity (needs Instrument calibration)

### Sources / references

- Davis, S.B. & Mermelstein, P. (1980), "Comparison of parametric representations for monosyllabic word recognition…" *IEEE TASSP* 28(4). https://doi.org/10.1109/TASSP.1980.1163420
- de Cheveigné, A. & Kawahara, H. (2002), "YIN, a fundamental frequency estimator for speech and music." *J. Acoust. Soc. Am.* 111(4). https://doi.org/10.1121/1.1458024
- Talkin, D. (1995), "A robust algorithm for pitch tracking (RAPT)." Ch. 14 of *Speech Coding and Synthesis* (Kleijn & Paliwal, eds.), Elsevier. ISBN 978-0-444-82169-1
- Camacho, A. & Harris, J.G. (2008), "A sawtooth waveform inspired pitch estimator for speech and music." *J. Acoust. Soc. Am.* 124(3). https://doi.org/10.1121/1.2951592
- Mauch, M. & Dixon, S. (2014), "pYIN: A fundamental frequency estimator using probabilistic threshold distributions." *Proc. ICASSP*. https://doi.org/10.1109/ICASSP.2014.6853678
- Kim, J.W., Salamon, J., Li, P., Bello, J.P. (2018), "CREPE: A convolutional representation for pitch estimation." *Proc. ICASSP*. https://arxiv.org/abs/1802.06182
- Slaney, M. (1998), "Auditory toolbox v2." Interval Research Tech. Report 1998-010. https://engineering.purdue.edu/~malcolm/interval/1998-010/
- librosa method-diversity precedent: https://librosa.org/doc/latest/feature.html
- Praat formant methods: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html

---

## 2026-05-21 — Advanced DSP (C2): LPC + hand-rolled Aberth roots → formants, mel→DCT MFCC, voicing on PitchFrame

Goal: settle the seventh Phase 1 slice. C2 lands the three advanced DSP families on top of C1's foundation: **formants** via LPC + polynomial root-finding, **MFCC** via mel-filterbank + DCT-II, and a **voicing decision** added to the existing autocorrelation pitch tracker. All three live inside `engine::dsp` and surface through `sadda.dsp.*`.

C2 also marks the first slice to land under the 2026-05-21 **DSP method-diversity principle** (see the entry directly above): every public DSP function carries a doc-comment citation with a publication link, and LPC ships both the autocorrelation method and Burg's method side-by-side.

### What C2 must deliver

From the Phase-1 slicing entry: (1) formants via LPC + root-solver; (2) MFCC (mel-filterbank → DCT); (3) refined pitch with voicing decision (extends Phase 0's autocorrelation tracker).

### Polynomial root-finding: hand-rolled Aberth-Ehrlich

For formants we need to find the complex roots of the LPC predictor polynomial (typically degree ~12 for 5 formants). Three options surfaced — hand-rolled Aberth-Ehrlich, `nalgebra` companion-matrix eigenvalues, a third-party root-finder crate. Hand-rolled won because:

- `nalgebra` would add ~50 transitive deps for one polynomial root-find per frame.
- Aberth-Ehrlich is well-known (parallel-Newton method with deflation correction), converges quadratically, and is ~80 LOC.
- We control numerical tolerance and behavior on ill-conditioned polynomials.

The algorithm:
1. Initialize `degree` complex roots evenly spaced on a circle of radius `(max|coeff|)^(1/degree)`.
2. Iteratively update each root `z_j ← z_j - p(z_j) / (p'(z_j) - p(z_j) · Σ_{k≠j} 1/(z_j - z_k))`.
3. Stop when `max|correction| < 1e-7` or 100 iterations.

LPC polynomials from speech are typically well-conditioned, and degree-12 roots converge in ~20 iterations. The module is `engine::dsp::roots`, fully unit-tested against analytical-truth polynomials.

### LPC: autocorrelation method AND Burg's method

Per the method-diversity principle, C2 ships both standard LPC estimators side-by-side:

- `engine::dsp::lpc::autocorr_lpc` — autocorrelation + Levinson-Durbin. Always produces a stable predictor (`|k_i| < 1` for all reflection coefficients) but tapers signal energy at frame edges (effectively assumes zero-extension outside the frame), biasing formant estimates on short frames.
- `engine::dsp::lpc::burg_lpc` — Burg's method. Estimates reflection coefficients directly from forward/backward prediction errors; avoids the autocorrelation method's implicit windowing. Praat's `Sound.to_formant_burg` default.

Dispatcher `lpc(samples, order, LpcMethod)` selects between them. **The formant tracker defaults to `LpcMethod::Burg`** to match Praat's convention; callers can override.

Citations: Makhoul (1975); Markel & Gray (1976); Burg (1975); Levinson (1947); Durbin (1960). Full links in the LPC module docs and in the C2 references list below.

### Formants

Pipeline per frame:
1. Apply pre-emphasis filter `y[n] = x[n] - α·x[n-1]` with `α = 0.97` (standard speech-DSP convention; caller can pass `0.0` to skip).
2. Window the frame (Hann by default).
3. Compute LPC coefficients via the chosen method (Burg by default; autocorrelation available).
5. Find roots of the predictor polynomial `1 + a_1·z⁻¹ + ... + a_p·z⁻ᵖ` (or equivalently `z^p + a_1·z^(p-1) + ... + a_p`).
6. For each root `z = r·e^(jθ)` with `r < 1` (inside unit circle) and `θ > 0` (upper half — complex conjugate pairs):
   - `frequency = θ · sample_rate / (2π)` Hz
   - `bandwidth = -ln(r) · sample_rate / π` Hz
7. Filter to `freq ∈ [50, sample_rate/2 - 50]` Hz and `bandwidth < 1000` Hz.
8. Sort by frequency.

**Default LPC order**: `2 · n_formants + 2`. For `n_formants = 5` (Praat default) → `p = 12`. Caller can override.

**Output shape: variable-length per frame**, not fixed-N. `FormantFrame { time_seconds, frequencies: Vec<f32>, bandwidths: Vec<f32> }`. Honest about frames where the root-finder didn't return enough valid roots in the F1–F<n_formants> range (silence, noise bursts, etc.). A future helper can pad to a `(n_frames, n_formants)` NumPy array with NaN, but C2 ships the list-of-frames form only.

### MFCC

Pipeline:
1. STFT magnitude → power spectrogram (reuse C1's `engine::dsp::stft` + `power_spectrogram`).
2. Apply mel filterbank: `n_mels` triangular filters between `f_min` and `f_max`, spaced uniformly on the mel scale (`m = 2595 · log10(1 + f/700)`).
3. Log of filterbank energies.
4. DCT-II to decorrelate → cepstral coefficients.
5. Keep the first `n_mfcc` coefficients.

DCT-II via direct matrix multiply (precompute the `n_mels × n_mfcc` cosine matrix once per call — trivially small at v1 defaults).

**Defaults** (matching librosa's defaults exactly so users porting code don't see surprises):

| Param | Default | Notes |
|---|---|---|
| Mel scale | **Slaney** | librosa's default (piecewise linear-then-log); HTK toggle deferred to task #55 |
| `n_mfcc` | 13 | Speech-recognition standard |
| `n_mels` | 40 | Phoneme-level resolution |
| `f_min` | 0.0 Hz | |
| `f_max` | `sample_rate / 2` | Nyquist |
| `frame_size_seconds` | 0.025 | 25 ms — standard speech analysis frame |
| `hop_seconds` | 0.010 | 10 ms |

**No** sinusoidal liftering or Δ/Δ² stacking in C2 — those are layered on top later if a real use case appears.

Output shape: `Array2<f32>` in Rust, `np.ndarray[float32, ndim=2]` in Python, shape `(n_frames, n_coeffs)` (frames-first to match librosa).

### Refined pitch: voicing decision on PitchFrame

Phase 0's `engine::pitch::autocorrelation` already computes the autocorrelation peak when finding the lag. Voicing is essentially free:

```
voicing = peak_autocorr_at_period / R(0)
```

`voicing ∈ [0, 1]` — close to 1 for clean voiced speech, near 0 for noise/silence. Threshold at 0.45 (typical literature value) to get a binary voiced flag.

`PitchFrame` grows a `voicing: f32` field. `PitchConfig` grows `voicing_threshold: f32` (default 0.45). Callers can read voicing directly or filter on the threshold.

**Python API**: keep `sadda.dsp.f0(audio, ...) → (times, freqs)` exactly as in Phase 0 (STABLE contract — no breakage). Add a new `sadda.dsp.voiced_pitch(audio, ...) → (times, freqs, voicing)` that returns the same three columns the Rust `PitchFrame` now has. Both functions call the same underlying autocorrelation tracker; they just project different fields.

### Confirmed C2 decisions

| Item | Decision | Reasoning |
|---|---|---|
| LPC methods | **Both autocorrelation (Levinson-Durbin) and Burg shipped side-by-side** | Method-diversity principle; Burg = Praat default for formants; autocorrelation = textbook default; each addresses a different stability/edge-bias trade-off |
| Default LPC method for formants | **Burg** | Praat's `Sound.to_formant_burg` convention; better short-frame behaviour |
| Polynomial root-solver | **Hand-rolled Aberth-Ehrlich (~80 LOC)** | Avoids ~50 `nalgebra` deps; well-known parallel-Newton method; numerical tolerance under our control |
| Formant output shape | **Variable-length `FormantFrame { time_seconds, frequencies, bandwidths }` per frame** | Honest about frames where the root-finder didn't return enough valid roots; Python wrapper can pad to fixed-N later if needed |
| Voicing API | **Add `sadda.dsp.voiced_pitch(...)` returning `(times, freqs, voicing)`; keep `sadda.dsp.f0(...)` returning `(times, freqs)`** | No Phase-0 surface breakage; new function for callers who want voicing; PitchFrame gains voicing field internally |
| LPC order default | **`2 · n_formants + 2`** | Standard speech-DSP rule of thumb; matches Praat's `Sound.to_formant_burg` default |
| Pre-emphasis | **Applied internally with `α = 0.97`** | Standard pipeline; caller can pass `0.0` to skip |
| MFCC defaults | **`n_mfcc=13`, `n_mels=40`, `f_min=0`, `f_max=sr/2`, 25 ms frame, 10 ms hop, Slaney mel scale** | Matches librosa's defaults exactly so ported code reproduces; HTK mel-scale toggle is a deferred alternate (task #55) |
| MFCC orientation | **`(n_frames, n_coeffs)`** | Frames-first, matching librosa; symmetric with spectrogram's `(n_freq_bins, n_frames)` only in being row-major |
| Citations | **Doc-comment citation required on every DSP function** | Per the method-diversity principle entry |

### Layout

- `crates/engine/src/dsp/lpc.rs` — Levinson-Durbin recursion; returns LPC coefficients + reflection coeffs + prediction gain.
- `crates/engine/src/dsp/roots.rs` — Aberth-Ehrlich polynomial root-solver.
- `crates/engine/src/dsp/formants.rs` — frame loop, pre-emphasis, root→formant conversion, filtering.
- `crates/engine/src/dsp/mfcc.rs` — mel-scale conversion, triangular filterbank, DCT-II matrix multiply.
- `crates/engine/src/pitch.rs` — extends `PitchFrame` with `voicing`; adds `PitchConfig.voicing_threshold`.
- `crates/python/src/lib.rs` — PyO3 wrappers (`formants`, `mfcc`, `voiced_pitch`); new `PyFormantFrame` data class.
- `python/sadda/dsp/__init__.py` — re-exports with `@stable`.
- `crates/engine/tests/advanced_dsp.rs` — analytical-truth tests against synthetic vowels and known polynomials.
- `python/tests/test_advanced_dsp.py` — end-to-end Python tests.

### Recommended defaults for v1

Per the method-diversity principle:

- **LPC** (general use): **Burg's method** — Praat's formant-tracker convention; better short-frame behaviour. Autocorrelation method available for textbook parity.
- **Polynomial root-find**: Aberth-Ehrlich. Robust and standalone (no `nalgebra` dep); converges in fewer iterations than companion-matrix QR for the degree-12-ish polynomials LPC produces.
- **Formant tracker**: LPC-Burg + Aberth-Ehrlich roots + freq/bw conversion. Pre-emphasis α=0.97. Praat-baseline behaviour.
- **MFCC**: Slaney mel scale (librosa default), n_mels=40, n_mfcc=13, 25 ms frame, 10 ms hop, HTK-style log + DCT-II. (HTK mel-scale toggle is a deferred alternate.)
- **Pitch**: `windowed_autocorrelation` — the window-corrected method described above. Strict improvement on Phase 0's naive autocorrelation. Naive `autocorrelation` retained for back-compat with `sadda.f0`.

### Deferred alternates

Tracked per the method-diversity principle; each has a corresponding tracking task:

- **Faithful Boersma 1993 pitch** (task #51) — full Praat pipeline: anti-alias upsample + Gaussian-window option + windowed-sinc + Brent's method peak refinement + multi-candidate Viterbi path-finder + octave/voiced-unvoiced cost terms. The renamed `windowed_autocorrelation` in C2 adopts only Boersma's central window-correction insight; this task tracks the rest.
- **YIN / pYIN / SWIPE' / CREPE pitch trackers** (task #52). pYIN is librosa's modern default; CREPE is the neural SOTA.
- **DeepFormants + QCP-FB formant tracker** (task #53) — Alku et al. 2023, <https://doi.org/10.1016/j.csl.2023.101515>. Modern accuracy upgrade beyond LPC+roots.
- **Noise-robust LPC** (task #54): QCP-FB (Airaksinen 2014) and Burg-robust (Praat's `Sound.to_formant_burg_robust`).
- **PNCC + Slaney mel + HTK toggle for MFCC** (task #55). PNCC (Kim & Stern 2016) is noise-robust; Slaney/HTK toggle for MATLAB/Kaldi parity.
- **Multitaper + reassigned STFT** (task #56) — Babadi & Brown 2014; Auger et al. 2013.
- **Magnitude / log-power spectrogram + periodic-window flag** (task #57).
- **LUFS loudness + calibrated dB-SPL intensity** (task #58).
- **Formant trajectory smoothing** (Viterbi / DP continuity) — C2 ships per-frame independent root-finding.
- **Fixed-N dense formant array helper** with NaN padding for missing formants.
- **MFCC Δ / Δ² stacking and sinusoidal liftering**.
- **Sub-sample lag interpolation** in the naive autocorrelation pitch tracker (the `windowed_autocorrelation` method already does parabolic interpolation; a future slice can add it to the naive tracker too).

### What this entry doesn't decide

- **Move `engine::pitch` under `engine::dsp::pitch`.** Still deferred (same rationale as C1: keep the slice diff additive).

### Sources / references

- 2026-05-21 DSP method-diversity principle entry (above this one)
- 2026-05-21 C1 entry (foundational DSP this builds on)
- 2026-05-21 Phase 1 slicing entry (C2 row)
- **LPC, autocorrelation method**: Makhoul, J. (1975), "Linear prediction: A tutorial review." *Proc. IEEE* 63(4). https://doi.org/10.1109/PROC.1975.9792
- **LPC, Burg's method**: Burg, J.P. (1975), *Maximum Entropy Spectral Analysis*. PhD thesis, Stanford. https://sepwww.stanford.edu/data/media/public/oldreports/sep06/
- **Levinson-Durbin**: Levinson, N. (1947), https://doi.org/10.1002/sapm1946251261 ; Durbin, J. (1960), https://doi.org/10.2307/1401322
- **Markel & Gray** (1976), *Linear Prediction of Speech*, Springer. ISBN 978-3-642-66288-1
- **Formant tracker (Praat-like, autocorrelation + roots)**: McCandless, S.S. (1974), "An algorithm for automatic formant extraction using linear prediction spectra." *IEEE TASSP* 22(2). https://doi.org/10.1109/TASSP.1974.1162572
- **Aberth's method**: Aberth, O. (1973), "Iteration methods for finding all zeros of a polynomial simultaneously." *Math. Comp.* 27(122). https://doi.org/10.1090/S0025-5718-1973-0329236-7
- **Aberth/Ehrlich numerical analysis**: Bini, D.A. (1996), "Numerical computation of polynomial zeros by means of Aberth's method." *Numerical Algorithms* 13. https://doi.org/10.1007/BF02207694
- **MFCC**: Davis, S.B. & Mermelstein, P. (1980), "Comparison of parametric representations for monosyllabic word recognition." *IEEE TASSP* 28(4). https://doi.org/10.1109/TASSP.1980.1163420
- **Voicing via autocorrelation peak ratio**: Boersma, P. (1993), "Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound." *Proc. Inst. Phonetic Sciences* 17. https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf
- Praat formant defaults: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html
- librosa.feature.mfcc: https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html

---

## 2026-05-21 — Foundational DSP (C1): sadda.dsp namespace introduced, rustfft+realfft, intensity = RMS + dB-FS

Goal: settle the sixth Phase 1 slice. C1 introduces the `sadda.dsp.*` namespace and lands the foundational DSP toolkit — windowing functions, STFT, spectrogram, intensity — as pure functions over `&[f32]` with no corpus coupling. The slice's exports become the first STABLE-tier members of `sadda.dsp.*` per the 2026-05-18 Python API surface entry.

### What C1 must deliver

From the Phase-1 slicing entry: (1) windowing functions (Hann, Hamming, Blackman, Gaussian, Kaiser); (2) STFT; (3) spectrogram; (4) intensity; (5) pure functions over `&[f32]`, no corpus dependency; (6) Polars-friendly outputs.

### Namespace: introduce `sadda.dsp.*` now; move `sadda.f0` → `sadda.dsp.f0`

The 2026-05-18 API surface entry pinned `sadda.dsp` as a STABLE namespace. C1 adds 8+ new public symbols — keeping them at the top level would clutter `sadda.*` for every future caller. So C1 is the natural slice to introduce the namespace.

The Phase-0 `sadda.f0(...)` moves to `sadda.dsp.f0(...)` to live alongside the rest of the DSP toolkit. A top-level `sadda.f0` alias stays in place for back-compat — both call paths reach the same underlying function. The alias is documented as a Phase-0 convenience and is part of the STABLE contract (not deprecated; users won't see a warning).

`sadda.Audio`, `sadda.new_project`, `sadda.open_project`, the tier types, etc. stay at the top level; only DSP gets a sub-module in C1.

### FFT: `rustfft` + `realfft`

`rustfft` 6.4 is the canonical Rust FFT crate — pure Rust, no_std-capable, MSRV 1.61 (fine for our 1.85). `realfft` 3.5 wraps it for real-only input (saves roughly half the work for audio, which is always real-valued). Both have minimal feature footprints; default features turn on SIMD intrinsics (AVX/SSE/NEON) that no-op on platforms without them.

### Module layout (Rust)

New `engine::dsp` module, split into focused sub-modules:

```
crates/engine/src/dsp/
├── mod.rs             — re-exports + module docs
├── windowing.rs       — hann, hamming, blackman, gaussian(n, sigma), kaiser(n, beta)
├── stft.rs            — stft(samples, window, hop) → (Vec<Complex<f32>>, Shape)
├── spectrogram.rs     — power_spectrogram(stft, shape) → Vec<f32> (n_freq_bins * n_frames)
└── intensity.rs       — intensity(samples, sample_rate, frame_size_seconds, hop_seconds) → Vec<IntensityFrame>
```

Pure functions over `&[f32]`; no `Project` coupling — unit-testable in isolation just like `engine::storage::dense`. The existing `engine::pitch` (autocorrelation f0) is logically a DSP module too, but stays where it is to keep C1's diff focused on the new surfaces; a follow-up may move it under `engine::dsp::pitch`.

### Window functions

All five return `Vec<f32>` of the requested length. Parameterized windows take their parameter explicitly:

```rust
pub fn hann(n: usize)     -> Vec<f32>;
pub fn hamming(n: usize)  -> Vec<f32>;
pub fn blackman(n: usize) -> Vec<f32>;
pub fn gaussian(n: usize, sigma: f32) -> Vec<f32>;
pub fn kaiser(n: usize, beta: f32)    -> Vec<f32>;
```

Forces callers to declare params (matches `scipy.signal.windows` convention); avoids hiding strong opinions like "Praat-default Kaiser β = 8.6" in the API. Callers compose via:

```rust
let win = sadda_engine::dsp::hann(frame_size);
let windowed: Vec<f32> = samples.iter().zip(win.iter()).map(|(x, w)| x * w).collect();
```

### STFT signature

```rust
pub struct Shape { pub n_frames: usize, pub n_freq_bins: usize }

pub fn stft(samples: &[f32], window: &[f32], hop_size: usize)
    -> (Vec<Complex<f32>>, Shape);
```

Returns the row-major flattened matrix shape `(n_frames, n_freq_bins)`. The companion `Shape` is the cheap structural metadata. Real-input optimized via `realfft::RealFftPlanner::plan_fft_forward(window.len())`. `n_freq_bins = window.len() / 2 + 1` (the unique part of the spectrum for real input).

### Spectrogram

Magnitude-squared (power) of the STFT, real-valued. Shape `(n_freq_bins, n_frames)` matches the API surface entry's documented convention; this is row-major-transposed relative to STFT's internal `(n_frames, n_freq_bins)` to be polars-friendly when each frequency bin becomes a column-like axis.

```rust
pub fn power_spectrogram(stft_out: &[Complex<f32>], shape: Shape)
    -> Vec<f32>;  // length n_freq_bins * n_frames, row-major (n_freq_bins, n_frames)
```

A `magnitude_spectrogram(...)` follow-up can layer in later if a user wants the `|X|` form instead of `|X|²`.

### Intensity: RMS + dB-FS per frame

```rust
pub struct IntensityFrame {
    pub time_seconds: f64,
    pub rms: f32,        // linear amplitude, root-mean-square over the frame
    pub db_fs: f32,      // 20 * log10(rms / 1.0); dB relative to full-scale [-1.0, 1.0]
}

pub fn intensity(
    samples: &[f32], sample_rate: u32,
    frame_size_seconds: f32, hop_seconds: f32,
) -> Vec<IntensityFrame>;
```

Both forms in one frame struct: linear RMS for raw analysis, dB-FS as the calibration-free dB form (relative to digital full-scale at amplitude 1.0). dB-SPL (relative to 2·10⁻⁵ Pa, the Praat convention) is deferred to a later slice that plumbs microphone calibration through the `Instrument` table.

Edge case: silent frames (RMS = 0) produce `db_fs = -∞`. The Rust implementation clamps at a small floor (e.g. `-200.0` dB) to keep downstream callers from having to special-case `NEG_INFINITY`.

### Python surface

`python/sadda/dsp/__init__.py` becomes the public entry point:

```
sadda.dsp.hann(n)               → np.ndarray[float32]
sadda.dsp.hamming(n)            → np.ndarray[float32]
sadda.dsp.blackman(n)           → np.ndarray[float32]
sadda.dsp.gaussian(n, sigma)    → np.ndarray[float32]
sadda.dsp.kaiser(n, beta)       → np.ndarray[float32]
sadda.dsp.stft(samples, frame_size, hop_size, *, window=None)
                                → tuple[np.ndarray[complex64, 2], (n_frames, n_freq_bins)]
sadda.dsp.spectrogram(samples, frame_size, hop_size, *, window=None)
                                → np.ndarray[float32, 2]   # shape (n_freq_bins, n_frames)
sadda.dsp.intensity(audio, *, frame_size_seconds=0.030, hop_seconds=0.010)
                                → tuple[np.ndarray[float64], np.ndarray[float32], np.ndarray[float32]]
                                  # (times, rms, db_fs)
sadda.dsp.f0(audio, ...)         # the existing Phase-0 function, relocated
```

Top-level `sadda.f0` stays as a back-compat alias pointing at the same function. All `sadda.dsp.*` symbols are `@stable`.

The DSP submodule is implemented in pure Python (`python/sadda/dsp/__init__.py`) re-exporting from `_native`; no PyO3 submodule machinery (which would complicate the stub layout). The Rust extension exposes all DSP functions flat in `sadda._native` (e.g. `_native.hann`, `_native.stft`, …); the Python wrapper does the namespacing.

### Confirmed C1 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Namespace | **Introduce `sadda.dsp.*` now; move `sadda.f0` → `sadda.dsp.f0` with top-level alias** | Cleanest namespace from day one; 8+ DSP symbols would clutter top-level; API-surface entry already pinned the namespace; alias keeps existing users working |
| FFT library | **`rustfft` 6 + `realfft` 3** | Canonical Rust FFT pair; pure Rust; MSRV 1.61; SIMD by default; minimal extra deps |
| Window API | **Return `Vec<f32>`; explicit per-param args** | Composable; no hidden defaults; matches scipy convention; testable per param |
| Intensity form | **`IntensityFrame { rms, db_fs, time_seconds }`** | Both calibration-free forms in one struct; dB-SPL deferred until Instrument calibration lands |
| Spectrogram orientation | **`(n_freq_bins, n_frames)`** | Matches the 2026-05-18 API-surface entry's documented convention; polars-friendly |
| Polars integration | **Returns NumPy; Python wraps with polars if desired** | Same pattern as B2/B3 — no polars-rs in the Rust tree |

### Layout

- `crates/engine/Cargo.toml` — adds `rustfft = "6"` and `realfft = "3"`.
- `crates/engine/src/dsp/{mod,windowing,stft,spectrogram,intensity}.rs` — new module with focused submodules.
- `crates/engine/src/lib.rs` — `pub mod dsp;` + re-exports.
- `crates/python/src/lib.rs` — new PyO3 functions for windowing/stft/spectrogram/intensity (the f0 binding already exists).
- `python/sadda/dsp/__init__.py` — re-exports + `@stable` decoration; pulls `f0` from `_native` too.
- `python/sadda/__init__.py` — keeps `f0` as the back-compat alias.
- `crates/engine/tests/dsp.rs` — round-trips and analytical-truth tests (sine → peak in spectrogram, RMS of known signal matches).
- `python/tests/test_dsp.py` — NumPy-side smoke tests + namespace presence + alias equality.

### Recommended defaults for v1

Per the 2026-05-21 method-diversity principle entry, each module names its v1 default explicitly:

- **Windowing**: Hann for general use (good main-lobe/side-lobe tradeoff, low scalloping).
- **STFT**: standard Gabor STFT (Hann window, hop ≤ window/4).
- **Spectrogram**: power (`|X|²`) for downstream computation; log-power for visualisation. C1 ships power only; log-power lands later (task #57).
- **Intensity**: linear RMS + dB-FS together per frame, 30 ms Hann frame, 10 ms hop.

### Deferred alternates

Per the method-diversity principle, the running development-target list for C1:

- **Periodic-window flag** (`sym=False`) for COLA-correct STFT overlap-add (task #57).
- **Multitaper STFT** via DPSS/Slepian tapers — Thomson 1982; review by Babadi & Brown 2014, <https://doi.org/10.1109/TBME.2014.2311996>. Lower-variance estimates, useful for HNR. (task #56)
- **Reassignment / synchrosqueezing STFT** — Auger et al. 2013, <https://doi.org/10.1109/MSP.2013.2265316>. Sharper time-frequency localization. (task #56)
- **Magnitude spectrogram** (`|X|`) and **log-power spectrogram** (`10·log10(|X|²)`) variants. (task #57)
- **LUFS / ITU-R BS.1770 K-weighted loudness** — <https://www.itu.int/rec/R-REC-BS.1770>. Broadcast-standard perceptual loudness. (task #58)
- **Calibrated dB-SPL** intensity once `Instrument` calibration is plumbed through. (task #58)
- **LazyFrame-style streaming STFT** for long recordings (live-recording slice E1 will revisit).
- **Move `engine::pitch` under `engine::dsp::pitch`** (cosmetic; deferred to a future reorganization slice).

### Sources / references

- 2026-05-18 Python API surface entry (`sadda.dsp` namespace, spectrogram shape convention)
- 2026-05-21 Phase 1 slicing entry (C1 row)
- `rustfft`: https://github.com/ejmahler/RustFFT (6.4.1, MSRV 1.61)
- `realfft`: https://github.com/HEnquist/realfft (3.5.0)
- scipy.signal.windows reference: https://docs.scipy.org/doc/scipy/reference/signal.windows.html
- librosa STFT/spectrogram reference: https://librosa.org/doc/latest/generated/librosa.stft.html

---

## 2026-05-21 — Dense tier types + Parquet sidecars (B3): apache parquet+arrow, per-bundle layout, NumPy/buffers in, polars out

Goal: settle the fifth Phase 1 slice. B3 lands the three dense tier types' on-disk format (Parquet sidecars under `signals/derived/`), the `DerivedSignal` registration table that ties them back to the corpus, and the read/write paths so AI-engineer users can either ask for a `polars.DataFrame` via `proj.query(tier_id)` or grab the sidecar path and `pl.scan_parquet(path)` directly.

### What B3 must deliver

From the Phase-1 slicing entry: (1) the three dense tier types — `continuous_numeric` / `continuous_vector` / `categorical_sampled`; (2) `DerivedSignal` registration rows; (3) reader/writer in `engine::storage::parquet`; (4) mmap-friendly load path so external readers work.

### Parquet via Apache `parquet` + `arrow` (version 58)

The corpus data-model entry pinned Parquet as the storage format. For Rust, the Apache `parquet` and `arrow` crates (version-locked) are the canonical pair — they're what `polars-rs` uses under the hood, and the resulting files are bit-for-bit consumable by `polars.scan_parquet(...)` / `pyarrow.parquet.read_table(...)` / `pandas.read_parquet(...)`. Verified compatibility with our `edition = "2024"` + `rust-version = "1.85"` workspace (both crates declare exactly those in their workspace package).

Minimal feature set to keep the dep tree from ballooning:

```toml
parquet = { version = "58", default-features = false, features = ["arrow", "snap"] }
arrow   = { version = "58", default-features = false }
```

- `parquet` defaults include `brotli`/`base64`/`simdutf8`/etc. — dropped; we only need Snappy (Parquet's most common codec).
- `arrow` defaults include `csv`/`ipc`/`json` — dropped; the Float64 / FixedSizeList / Utf8 types live in the always-on core.
- `ndarray` is added as a separate dep for the `Array2<f64>` ergonomics on the Rust side; arrow has no `ndarray` feature, so we marshal manually via `Float64Array::from(arr.as_standard_layout().as_slice().unwrap().to_vec())`.

### Sidecar layout: per-bundle subdirectories

`signals/derived/bundle_<id>/<tier_name>.parquet`. All sidecars for a bundle group together — easy to `ls` by hand, no branching on whether the bundle has a session, and rename-safe at the file level (renaming a tier renames the file via the engine; external readers reference DerivedSignal.relative_path). The original data-model entry sketched `signals/derived/session_<id>/bundle_<id>.<name>.parquet` but that nesting is awkward when bundles have no session.

### DerivedSignal table (V5)

V5 migration adds:

```sql
CREATE TABLE derived_signal (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tier_id         INTEGER NOT NULL UNIQUE REFERENCES tier(id),
    relative_path   TEXT    NOT NULL,
    n_frames        INTEGER NOT NULL,
    n_dims          INTEGER NOT NULL DEFAULT 1,
    sample_rate_hz  REAL,
    dtype           TEXT    NOT NULL,
    extra           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX idx_derived_signal_tier ON derived_signal(tier_id);
```

- `tier_id` UNIQUE: one sidecar per tier (writes after the first error; rewrites land in a follow-up).
- `n_dims = 1` for `continuous_numeric` and `categorical_sampled`; `n_dims >= 1` for `continuous_vector`.
- `sample_rate_hz` NULL for non-sampled / variable-rate signals (none in v1, but the column is forward-compat).
- `dtype` enum stored as text: `f64`, `f32`, `utf8` for v1.
- Audited per the B1 trigger-rebuild discipline (3 triggers added in V5).

### Write API: positional buffers in Rust, NumPy in Python

Three options surfaced — positional buffers (Rust `&[f64]` / `Array2<f64>` / `&[String]`), DataFrame-typed inputs (requires `polars-rs` + `pyo3-polars`), or both. Positional buffers won because:
- No new Rust deps beyond `parquet`+`arrow` (already heavy enough).
- Clean Rust signatures; no DataFrame ceremony at internal call sites.
- The read path can still return a DataFrame; symmetry isn't required.
- DataFrame inputs land in a later ergonomic-pass slice if real users want bulk writes from polars.

Rust signatures:

```rust
impl Project {
    pub fn write_continuous_numeric(
        &self, tier_id: i64,
        samples: &[f64], sample_rate_hz: f64,
    ) -> Result<i64>; // returns derived_signal.id

    pub fn write_continuous_vector(
        &self, tier_id: i64,
        frames: ndarray::ArrayView2<'_, f64>, sample_rate_hz: f64,
    ) -> Result<i64>;

    pub fn write_categorical_sampled(
        &self, tier_id: i64,
        labels: &[String], sample_rate_hz: f64,
    ) -> Result<i64>;

    pub fn read_continuous_numeric(&self, tier_id: i64) -> Result<Vec<f64>>;
    pub fn read_continuous_vector(&self, tier_id: i64) -> Result<ndarray::Array2<f64>>;
    pub fn read_categorical_sampled(&self, tier_id: i64) -> Result<Vec<String>>;

    pub fn derived_signal(&self, tier_id: i64) -> Result<Option<DerivedSignal>>;
    pub fn dense_path(&self, tier_id: i64) -> Result<Option<PathBuf>>;
}
```

The write methods reject if the tier's type doesn't match, if a `derived_signal` row already exists for the tier (no overwrite in v1), and if the input buffer is empty.

### Python: NumPy in, polars out

- `proj.write_continuous_numeric(tier_id, np.ndarray[float64], sample_rate_hz)`
- `proj.write_continuous_vector(tier_id, np.ndarray[float64, ndim=2], sample_rate_hz)`
- `proj.write_categorical_sampled(tier_id, list[str], sample_rate_hz)`
- `proj.read_continuous_numeric(tier_id) → np.ndarray[float64]`
- `proj.read_continuous_vector(tier_id) → np.ndarray[float64, ndim=2]`
- `proj.read_categorical_sampled(tier_id) → list[str]`
- `proj.dense_path(tier_id) → str | None` — for `pl.scan_parquet(path)` external reads
- `proj.derived_signal(tier_id) → DerivedSignal | None` — the registration row
- `proj.query(tier_id)` (the B2 monkey-patch) extended: for dense tiers, calls `pl.read_parquet(proj.dense_path(tier_id))` and returns the resulting DataFrame; for sparse, the B2 dispatch.

### `categorical_sampled` encoding

Plain UTF8 column for v1; Parquet dictionary encoding is an optimization for later. Reasoning: dictionary encoding shrinks VAD/voicing-class files ~3× but adds slightly more arrow-rs API surface and has marginal compatibility risk with older Parquet readers; v1 corpora are small enough that the disk delta doesn't matter. The polars `.cast(pl.Categorical)` escape hatch handles in-memory categorization on read.

### Confirmed B3 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Parquet library | **Apache `parquet` + `arrow` 58 (minimal features)** | Canonical Rust→Parquet path; bit-for-bit compatible with polars/pyarrow consumers; only viable choice without forking `polars-rs` |
| Sidecar layout | **`signals/derived/bundle_<id>/<tier_name>.parquet`** | All sidecars for a bundle grouped; no session-presence branching; rename-safe at the file level |
| `DerivedSignal` shape | **`tier_id UNIQUE` + path + n_frames/n_dims/sample_rate_hz/dtype + extra** | One sidecar per tier in v1; extension columns cover the queries the read path needs without parsing Parquet metadata |
| Write API | **Positional buffers (Rust slices/ndarray; NumPy in Python)** | No `polars-rs` dep; clean signatures; symmetric read path can still return DataFrames |
| `categorical_sampled` encoding | **Plain UTF8; dictionary as a later optimization** | Simpler write path; broad compatibility; size delta acceptable at v1 corpus scale |
| `proj.query(tier_id)` for dense tiers | **`pl.read_parquet(path)`** | Reuses the B2 monkey-patch; matches AI-engineer expectations; LazyFrame variant can layer in later |

### Layout

- `crates/engine/migrations/V5__derived_signal.sql` — table + index + 3 audit triggers.
- `crates/engine/src/storage/mod.rs` + `crates/engine/src/storage/parquet.rs` — new module owning `arrow`/`parquet` boilerplate; pure Rust, no `Project` dep, so it's unit-testable in isolation.
- `crates/engine/src/corpus.rs` — `DerivedSignal` struct; `Project::write_*` / `read_*` / `derived_signal` / `dense_path` methods.
- `crates/python/src/lib.rs` — PyDerivedSignal class; numpy-bridging write/read methods on PyProject; `dense_path` returning Option<str>.
- `python/sadda/__init__.py` — extend the B2 `proj.query` monkey-patch to dispatch on dense tier types via `pl.read_parquet(...)`.
- `crates/engine/tests/dense_tiers.rs` + `python/tests/test_dense_tiers.py` — round-trip + interop tests.

### What this entry doesn't decide

- **Rewrite / append to existing sidecars.** v1: one-shot write per tier; rewrites error. A follow-up slice may add `proj.replace_dense(...)` once a clear use case appears.
- **Streaming writes (chunk-by-chunk).** v1 writes the whole buffer in one call. Streaming arrives with live recording (E1) if needed for long sessions.
- **LazyFrame from `proj.query`.** Materializing the DataFrame is fine at v1 scale (≤ ~1 GB sidecars for the typical user). A `proj.scan(tier_id) → pl.LazyFrame` can layer in later for the ML / large-embedding case.
- **Mixed-type vector columns.** All `continuous_vector` columns are Float64 at v1; Float32 + Int variants land when a real consumer needs them.
- **Audit log of Parquet contents.** The DerivedSignal row is audited; the Parquet *file contents* are not — they're write-once column blobs and the audit trail lives at the registration boundary.
- **Symlink / cache eviction policy.** Sidecars stay until the bundle is deleted (cascade from `bundle` to `tier` to `derived_signal` is a future migration if we add `ON DELETE CASCADE` — not in V5).

### Sources / references

- 2026-05-18 corpus data-model entry (Parquet pinning, sparse/dense split)
- 2026-05-21 Phase 1 slicing entry (B3 row)
- 2026-05-21 B2 entry (sparse tier types this entry parallels)
- Apache arrow-rs: https://github.com/apache/arrow-rs (parquet 58, arrow 58)
- `parquet::arrow::ArrowWriter`: https://docs.rs/parquet/58.3.0/parquet/arrow/index.html
- `FixedSizeListArray`: https://docs.rs/arrow-array/58.3.0/arrow_array/array/struct.FixedSizeListArray.html

---

## 2026-05-21 — Sparse tier types (B2): three annotation tables, Rust-level cardinality, polars wrap in Python

Goal: settle the fourth Phase 1 slice. B2 puts annotation rows into the three sparse-tier tables, wires tier-header + annotation CRUD into the Project API, enforces parent-child cardinality at insert time, and ships the first cut of `proj.query(...) → polars.DataFrame`.

### What B2 must deliver

From the Phase-1 slicing entry: (1) `interval`, `point`, `reference` tier types with CRUD; (2) first cut of `proj.query(...) → polars.DataFrame`; (3) parent-child cardinality enforced at insert time.

### Three separate annotation tables

The 2026-05-18 corpus data-model entry already pinned three per-type tables (versus a single discriminated annotation table). Schemas:

```sql
annotation_interval (
    id, tier_id, start_seconds REAL, end_seconds REAL,
    label, parent_annotation_id, extra,
    CHECK (end_seconds > start_seconds)
);
annotation_point (
    id, tier_id, time_seconds REAL, label, parent_annotation_id, extra
);
annotation_reference (
    id, tier_id, target_kind TEXT, target_id INTEGER,
    label, parent_annotation_id, extra,
    CHECK (target_kind IN ('bundle','session','speaker','tier','annotation'))
);
```

All three are audited per the B1 trigger-rebuild discipline; V4 includes 9 new audit triggers.

### Polars integration: Python-side wrap

Three options surfaced — Python-side wrap of Rust-supplied rows, `polars-rs` + `pyo3-polars` in Rust, or LazyFrame via path scan. Python-side wrap won because:
- No new Rust deps. `polars-rs` pulls ~50 transitive crates and lengthens cold builds substantially; not worth it for sparse-tier scale (≤ ~100K rows per project).
- The Arrow-zero-copy story matters mostly for *dense* tiers (continuous_vector embeddings, continuous_numeric tracks). B3 introduces Parquet sidecars where Arrow buffers are already the native format; that's the natural place to add `polars-rs` if needed.
- The Python wrapper is trivial: Rust returns `Vec<row tuples>`, `__init__.py` calls `polars.DataFrame(rows, schema=...)` with a tier-type-specific column shape.

`polars` joins `numpy` in the runtime deps in `pyproject.toml`.

### Cardinality enforcement: engine-level (Rust)

SQL triggers can't easily express "the parent_annotation_id must reference the correct one of three possible parent annotation tables, chosen by the parent tier's type." Rust-level enforcement at insert time is straightforward, gives clear error messages, and avoids trigger debugging when something goes wrong.

The check in `Project::add_interval` (and the analogous point/reference methods):

```
let tier = self.get_tier(tier_id)?;
match tier.parent_id {
    None => { /* no parent: no check */ }
    Some(parent_tier_id) => {
        let parent_annotation_id = spec.parent_annotation_id
            .ok_or_else(|| EngineError::Cardinality(
                "child tier requires parent_annotation_id".into()))?;
        // Verify parent annotation exists in the parent tier's right table.
        ensure_parent_exists(self, parent_tier_id, parent_annotation_id)?;
        match tier.cardinality.as_deref() {
            Some("one_to_one") => ensure_parent_is_unique(...)?,
            Some("one_to_many") | Some("none") | None => {}
            Some("many_to_one") => return Err(EngineError::Cardinality(
                "many_to_one cardinality is not supported until B-cluster follow-up".into())),
            Some(other) => return Err(...),
        }
    }
}
```

New `EngineError::Cardinality(String)` variant covers all the failure modes — clearer than reusing `Corpus(String)`.

### Many-to-one deferred

The v1 use cases all naturally fit `one_to_many` (word→phones, syllable→phones, …). `many_to_one` (multiple parents per child) is inherently a join table the rest of the model doesn't have. V4 keeps the cardinality enum's `many_to_one` value (the V3 CHECK already lists it), but `add_interval` / `add_point` / `add_reference` reject it with a clear "not supported until follow-up" message. A future B-cluster slice can add the join table when a real use case surfaces.

### Confirmed B2 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Annotation tables | **Three: `annotation_interval`, `annotation_point`, `annotation_reference`** | Already pinned by the corpus data-model entry; per-type tables keep columns typed and queries direct |
| Polars | **Python-side wrap; `polars-rs` deferred to B3** | No new Rust deps; cheap to swap when Parquet sidecars need Arrow zero-copy |
| Cardinality enforcement | **Engine-level Rust check at insert time** | SQL triggers can't dispatch on parent tier's type cleanly; Rust gives clear error messages |
| `many_to_one` | **Deferred** | No v1 use case requires it; the join table can land when a use case appears |
| API surface | **Both DataFrame and typed accessors** | `proj.query(tier_id)` → polars.DataFrame for AI-engineer ergonomics; `proj.intervals(tier_id)` → `list[Interval]` for OO callers and tests |

### Layout

- `crates/engine/migrations/V4__sparse_annotations.sql` — three tables + indexes + 9 audit triggers.
- `crates/engine/src/corpus.rs` — `Tier` / `Interval` / `Point` / `Reference` structs + their `*Spec` builders; `add_tier`, `tiers`, `get_tier`, `add_interval`, `intervals`, `add_point`, `points`, `add_reference`, `references_for` (avoiding the `references` reserved-ish name), plus a Rust helper `query_tier_rows(tier_id) → Vec<RowTuple>` for the Python query wrapper.
- `crates/engine/src/error.rs` — adds `EngineError::Cardinality(String)`.
- `crates/python/src/lib.rs` — PyTier / PyInterval / PyPoint / PyReference + insert/list bindings; raw query method returns Python tuples.
- `python/sadda/__init__.py` — wires `Project.query(tier_id)` (Python method patched onto the Rust class) to call the raw method and build a `polars.DataFrame` with tier-type-aware column schema.
- `pyproject.toml` — adds `polars>=1.0` to runtime deps.

### Trigger-rebuild discipline (reminder)

V4 adds three new audited tables — annotation_interval, annotation_point, annotation_reference. The audit triggers are created in V4 alongside the tables. Any future ALTER TABLE on these tables must DROP+CREATE the triggers per the B1 entry's rule.

### What this entry doesn't decide

- **Dense tier CRUD + Parquet sidecars.** That's B3.
- **Cross-bundle query language.** `proj.query(tier_id)` is the first cut; richer filtering (`proj.query(tier_name="phones", bundles=...)`, EMU-EQL-style traversal) is a later API slice.
- **`many_to_one` join table.** Deferred until a use case appears.
- **Polars schema-typed inserts.** B2 returns DataFrames *from* the corpus; inserting *from* a DataFrame (bulk write of annotations) is a later ergonomic.
- **TextGrid/EAF I/O.** That's D1 / D2 and consumes B2's tier types.

### Sources / references

- 2026-05-18 corpus data-model entry (sparse tier storage decision)
- 2026-05-18 Python API surface entry (polars as primary query-result type)
- 2026-05-21 Phase 1 slicing entry (B2 row)
- 2026-05-21 B1 entry (trigger-rebuild discipline)
- EMU-SDMS level/segment model: https://ips-lmu.github.io/EMU.html
- polars Python API: https://docs.pola.rs/api/python/stable/

---

## 2026-05-21 — Full entity schema + AuditLog (B1): triggers + JSON1, Speaker/Session API only, schema-only for the rest

Goal: settle the third Phase 1 slice. B1 lays down the SQLite-side scaffolding for the v1 entity model (Speaker, Session, Bundle-extension, Tier-header, Entity, EntityRef, Instrument, Protocol, ProcessingRun, AuditLog) and the trigger-based audit infrastructure. Annotation CRUD (B2) and Parquet sidecars (B3) build on top of this schema; F1's recipes write `processing_run` rows.

### What B1 must deliver

From the Phase-1 slicing entry: (1) the schema for nine new entity tables; (2) `extra: json` columns throughout; (3) append-only AuditLog with mutation triggers; (4) ProcessingRun (the renamed ModelRun per the 2026-05-20 ML-registry entry).

### Scope: schema everywhere, API for Speaker + Session + Bundle only

The schema lands in full so subsequent slices (B2, B3, F1, …) build on the same migration. The Rust + Python API surface is intentionally narrower in B1:

| Table | Schema in V3? | Public API in B1? | First public-API slice |
|---|---|---|---|
| `speaker` | ✅ | ✅ | B1 |
| `session` | ✅ | ✅ | B1 |
| `bundle` (extended) | ✅ | ✅ (optional `session_id` + `speaker_id` args) | B1 |
| `instrument` | ✅ | — | B2 or later |
| `protocol` | ✅ | — | E1 (live recording) or experiments slice |
| `entity` | ✅ | — | profile-driven; later |
| `entity_ref` | ✅ | — | B2 (when ref-tier annotations land) |
| `tier` (header) | ✅ | — | B2 (when annotation rows land) |
| `processing_run` | ✅ | — | F1 (recipe.record) |
| `audit_log` + `_audit_context` | ✅ | minimal: `proj.set_audit_user(name)` | B1 |

The narrower API surface keeps B1's commit footprint near the Phase-1 cadence target (~750–1000 LOC). Future slices wire each table into Python as their first real user appears, with no further schema work needed.

### AuditLog: SQLite triggers + `_audit_context` singleton

Three options surfaced — application-level (each engine mutation INSERTs its own audit row), triggers (DB enforces; can't be bypassed), or a hybrid. Triggers won because the regulatory-stance entry calls for "every analysis step recorded" — application-level logging is one direct `INSERT` from a forgotten audit row, and any future plugin / SQL CLI usage bypasses it entirely. Triggers fire regardless of caller.

User attribution: a singleton `_audit_context` row holds the current user; triggers read it (`(SELECT user FROM _audit_context)`); the Rust API sets it on connection (`Project::set_audit_user`). Default is `"local"` (no auth in v1). Recipe replay can override per-block when F1 lands.

JSON payloads via SQLite's `json_object()` (JSON1 extension; bundled with rusqlite since long before our pinned 0.32). Trigger shape per audited table:

```sql
CREATE TRIGGER <table>_audit_insert AFTER INSERT ON <table> BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context),
        '<table>',
        NEW.id,
        'insert',
        NULL,
        json_object('col1', NEW.col1, 'col2', NEW.col2, ...)
    );
END;
-- Plus _audit_update (before + after JSON) and _audit_delete (before only) variants.
```

Audited tables: `speaker`, `session`, `instrument`, `protocol`, `entity`, `entity_ref`, `bundle`, `tier`, `processing_run`. **NOT** audited: `project` (singleton), `schema_migrations` (managed by migrator), `audit_log` itself (would recurse), `_audit_context` (engine-internal).

**Trigger-rebuild discipline**: any future migration that `ALTER TABLE`s an audited table must `DROP TRIGGER IF EXISTS` + recreate the three triggers so JSON column lists stay current. Codified in a comment at the top of V3 and a checklist line in `crates/engine/migrations/README.md` (to be added).

### Confirmed B1 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Audit mechanism | **SQLite triggers + `_audit_context` singleton** | DB-enforced; survives external SQL writes; matches regulatory stance |
| Bundle ↔ Speaker | **Nullable FK `bundle.speaker_id`** | Common case covered cleanly; multi-speaker handled via per-segment tier rows in B2 |
| Python API shape | **Flat methods on `Project`** | Matches Phase-0 / A2 surface; namespacing decision settles once for the whole corpus layer later |
| `user` field | **`_audit_context` table; engine sets on connection; default `"local"`** | No auth in v1; explicit setter lets F1 recipes scope user per-block |
| Migration granularity | **One V3 migration** | Cohesive atomic schema bump; aligns with one-slice-one-commit cadence |
| Schema-only vs API'd tables | **API for Speaker + Session + Bundle-extension; rest schema-only until a first real user appears** | Keeps B1 commit at Phase-1 cadence target; no API churn when later slices add the public surface for Entity/Instrument/…|

### Layout

- `crates/engine/migrations/V3__entity_schema.sql` — full schema bump.
- `crates/engine/src/corpus.rs` — adds `Speaker`, `Session` structs; `Project::{add_speaker, speakers, get_speaker, add_session, sessions, get_session, set_audit_user, audit_user}`; extends `Project::add_bundle` to accept optional `session_id` + `speaker_id`.
- `crates/python/src/lib.rs` — PyO3 wrappers; `#[gen_stub_*]` attributes.
- `python/sadda/__init__.py` — re-exports + `@stable` decoration.
- `crates/engine/tests/migrations.rs` — extended: V3 applies on fresh DB and on a synthesized post-V2 DB; triggers fire and write the expected JSON; bundle extension keeps existing Phase-0 columns intact.
- `python/tests/test_corpus.py` — Speaker/Session add+list round-trips.

### Schema sketch

```sql
-- Entities (each with extra TEXT for JSON payload):
CREATE TABLE speaker (id, name, sex, birth_year, notes, extra, created_at);
CREATE TABLE session (id, name, started_at, ended_at, location,
                     instrument_id REFERENCES instrument(id),
                     protocol_id   REFERENCES protocol(id),
                     notes, extra, created_at);
CREATE TABLE instrument (id, name, kind, serial, calibration, extra, created_at);
CREATE TABLE protocol (id, name, description, schema, extra, created_at);
CREATE TABLE entity (id, kind, name, extra, created_at);
CREATE TABLE entity_ref (id, entity_id, target_kind, target_id, role, extra);
CREATE TABLE tier (id, bundle_id, name, type, parent_id, cardinality,
                  schema, extra, created_at, UNIQUE (bundle_id, name));
CREATE TABLE processing_run (id, bundle_id, kind, processor_id, processor_version,
                             weights_checksum, parameters, input_tier_ids,
                             output_tier_ids, output_signal_ids,
                             started_at, finished_at, status, error_message,
                             recipe_run_id);

-- Bundle extension:
ALTER TABLE bundle ADD COLUMN session_id INTEGER REFERENCES session(id);
ALTER TABLE bundle ADD COLUMN speaker_id INTEGER REFERENCES speaker(id);
ALTER TABLE bundle ADD COLUMN extra TEXT;

-- Audit:
CREATE TABLE _audit_context (id INTEGER PK CHECK(id=1), user TEXT NOT NULL DEFAULT 'local');
INSERT INTO _audit_context VALUES (1, 'local');
CREATE TABLE audit_log (id, timestamp, user, table_name, row_id, op, before, after);

-- 27 triggers (3 per audited table × 9 audited tables).
```

Type enums in CHECK constraints:
- `tier.type IN ('interval', 'point', 'reference', 'continuous_numeric', 'continuous_vector', 'categorical_sampled')`
- `tier.cardinality IN ('one_to_one', 'one_to_many', 'many_to_one', 'none')`
- `entity_ref.target_kind IN ('bundle', 'session', 'speaker', 'tier', 'annotation')`
- `processing_run.kind IN ('ml_model', 'dsp_algorithm', 'clinical_measure', 'plugin')`
- `processing_run.status IN ('ok', 'error', 'partial')`
- `audit_log.op IN ('insert', 'update', 'delete')`

### What this entry doesn't decide

- **Trigger regeneration tooling.** A future `cargo xtask audit-triggers` (introspect tables, generate trigger SQL from `PRAGMA table_info`) is plausible but out of scope; per-migration discipline carries B1.
- **Cross-bundle query API.** The corpus-data-model entry's deferred item ("how a phonetician asks 'all phones across all bundles'") stays deferred — it's a query-language decision, not a schema one. Polars-DataFrame queries via `proj.query(...)` arrive in B2.
- **JSON-schema validation for `extra`.** Profile schemas validate `extra` payloads — they exist as files per the 2026-05-20 profile-catalog entry; wiring them into the engine's write path is a later concern.
- **Audit log retention.** Pruning, archival, vacuum policy — out of scope for B1. Likely a CLI verb after real usage.

### Sources / references

- 2026-05-18 corpus data-model entry (entity tables, audit-log shape)
- 2026-05-20 ML-model-registry entry (ProcessingRun rename + schema)
- 2026-05-18 clinical regulatory entry (audit-trail requirements)
- 2026-05-21 Phase 1 slicing entry (B1 row)
- SQLite JSON1 extension docs: https://sqlite.org/json1.html
- PostgreSQL audit_trigger pattern: https://wiki.postgresql.org/wiki/Audit_trigger_91plus

---

## 2026-05-21 — Stability decorators + type stubs (A2): mixed-project layout, pyo3-stub-gen, all-STABLE Phase-0 tiering

Goal: settle the second Phase 1 slice. A2 introduces the API contract — `@stable` / `@provisional` / `@experimental` decorators with one-time runtime warnings — and the type-stub pipeline that every subsequent slice carries. The 2026-05-18 Python-API-surface entry already pinned the tier semantics; A2 commits to *how* they're implemented and integrated.

### What A2 must deliver

From the Phase-1 slicing entry: (1) `@stable` / `@provisional` / `@experimental` Python decorators emitting one-time runtime warnings; (2) `pyo3-stub-gen` integrated into the build; (3) `py.typed` marker added; (4) existing Phase-0 APIs (`sadda.version`, `sadda.load_wav`, `sadda.f0`, `sadda.Audio`) tiered.

### Tooling: maturin mixed project + pyo3-stub-gen

Decorators are Python-level concerns — wrapping a Rust-built C extension function with `functools.wraps` is straightforward but requires Python source to live alongside the wheel. The maturin **mixed-project layout** is the canonical solution: a `python/sadda/` package directory plus a Rust extension exposed as a submodule (`sadda._native`). This is the exact pattern pydantic-core, orjson, and the pyo3-stub-gen `examples/mixed` template use.

`pyo3-stub-gen` is the only viable stub generator that targets PyO3 0.28 (verified: requires `pyo3 >= 0.26`; current release `0.22.4`). Its workflow is:
- Add `#[gen_stub_pyfunction]` / `#[gen_stub_pyclass]` / `#[gen_stub_pymethods]` *above* the existing PyO3 macros on each public item.
- Call `define_stub_info_gatherer!(stub_info)` at the bottom of the pymodule's `lib.rs`.
- Add `src/bin/stub_gen.rs` that invokes `stub_info()?.generate()` — writes `.pyi` files next to the Python package.
- CI runs `cargo run --bin stub_gen` then `git diff --exit-code python/sadda/_native.pyi`; PRs that change Rust signatures without regenerating fail.

### Confirmed A2 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Package layout | **Maturin mixed project**: `python/sadda/__init__.py` + Rust submodule `sadda._native` | Universal pattern (pydantic-core, orjson); enables Python-side decorators on Rust functions; needed for `py.typed` placement |
| Stub generation | **`cargo run --bin stub_gen` writes committed `python/sadda/_native.pyi`; CI diff-checks** | Best IDE experience (visible in source tree); drift is caught at PR time, not at build time |
| Class @provisional semantics | **Wraps `__init__`; warns once on first instantiation** | Quiet for type imports and `isinstance` checks; matches users' mental model that "using" a class = constructing one |
| Phase-0 tiering | **All four (`version`, `load_wav`, `f0`, `Audio`) → STABLE** | Matches the API-surface entry's pinning; downgrading would break the Phase-0 commitment for users already on 0.0 |
| Warning class hierarchy | **`SaddaWarning(UserWarning)` → `ProvisionalAPIWarning`, `ExperimentalAPIWarning`** | `UserWarning` is visible by default (unlike `DeprecationWarning`); shared base allows `simplefilter("ignore", SaddaWarning)` to silence all tiers at once |
| Warning frequency | **Once per decorated symbol per process** | Stored in a module-level `set[str]` keyed by qualified name; standard pattern (scikit-learn, scipy) |

### Layout

```
sadda/
├── python/sadda/                      # Python package (maturin python-source)
│   ├── __init__.py                    # re-exports + applies @stable to Phase-0 symbols
│   ├── _stability.py                  # decorators + warning classes
│   ├── _native.pyi                    # generated by stub_gen; committed
│   └── py.typed                       # PEP 561 marker (empty file)
├── crates/python/
│   ├── Cargo.toml                     # adds pyo3-stub-gen dep
│   ├── src/
│   │   ├── lib.rs                     # #[pymodule] renamed sadda → _native; #[gen_stub_*] sprinkled
│   │   └── bin/stub_gen.rs            # cargo run --bin stub_gen
└── pyproject.toml                     # [tool.maturin] python-source="python", module-name="sadda._native"
```

### Decorator implementation

`_stability.py` exports three decorators plus the warning classes. Each decorator:
- For a function: returns a `functools.wraps`-decorated wrapper that emits the warning at most once per process, then forwards to the original.
- For a class: replaces `__init__` with a wrapped version that emits the warning at most once on first instantiation, then calls the original `__init__`.
- For `@stable`: a no-op (still tags `fn.__sadda_stability__ = "stable"` for introspection).

The "once" set lives at module scope keyed by `f"{func.__module__}.{func.__qualname__}"`. Warnings are emitted with `stacklevel=2` so the user's calling line is what `warnings` reports.

### Warning categories

```python
class SaddaWarning(UserWarning):
    """Base class for sadda's stability warnings."""

class ProvisionalAPIWarning(SaddaWarning):
    """First use of a PROVISIONAL API. May break in minor versions after a deprecation cycle."""

class ExperimentalAPIWarning(SaddaWarning):
    """First use of an EXPERIMENTAL API. May break in any release without notice."""
```

Subclassing `UserWarning` (not `DeprecationWarning` / `PendingDeprecationWarning`) is deliberate: Python's default filter hides those for non-`__main__` modules, which would silence the signal we want users to see when their script imports something `sadda.ml`.

### Phase-0 tiering

| Symbol | Tier |
|---|---|
| `sadda.version()` | STABLE |
| `sadda.load_wav(path)` | STABLE |
| `sadda.f0(audio, ...)` | STABLE |
| `sadda.Audio` (class) | STABLE |

`@stable` is a no-op at runtime, but applying it now (a) tags the symbol for `inspect`-based audits, and (b) makes the absence of a stability decorator on a future PR a code-smell.

### What this entry doesn't decide

- **Top-level convenience aliases vs namespaced canonical homes.** `sadda.f0` stays as the canonical home for now; a future namespacing pass (likely after B-cluster lands `sadda.dsp`) decides whether `sadda.f0` becomes an alias for `sadda.dsp.f0` or stays the primary entry. Either move is compatible with the STABLE tier — aliases don't break callers.
- **`@stable` audit tooling.** A future `cargo xtask audit-tiers` (assert every public PyO3 symbol has a corresponding decorator in `__init__.py`) is plausible but out of scope.
- **Stub generation for engine-internal types reachable via Python.** All Phase-0 types fit; future provisional surfaces may need manual `.pyi` patches if `pyo3-stub-gen` can't infer their shape.

### Sources / references

- 2026-05-18 Python API surface entry (this entry implements the stability-tier section, lines 841–849)
- 2026-05-21 Phase 1 slicing entry (this entry expands its A2 row)
- pyo3-stub-gen docs and mixed example: https://github.com/Jij-Inc/pyo3-stub-gen
- maturin mixed-project layout: https://www.maturin.rs/project_layout

---

## 2026-05-21 — Migration framework (A1): hand-rolled migrator, forward-only, always back up

Goal: settle the first Phase 1 slice. A1 wires migrations into `engine::corpus` so that every subsequent slice — starting with B1's eight-table schema expansion — rides on proper rails rather than the Phase-0 `CREATE TABLE IF NOT EXISTS` blob.

### What A1 must deliver

From the Phase-1 slicing entry: (1) a real migration framework, (2) the `schema_migrations` table extended for richer provenance, (3) a forward-compat clamp (engine refuses to open a DB newer than itself), (4) `corpus.db.bak.<old_version>` written before any destructive migration, (5) per-migration tests.

### Tool choice: hand-rolled, ~50 LOC

The slicing entry framed the choice as "`sqlx::migrate!` vs `refinery`", but that framing pre-dated the rusqlite commitment in Phase 0. Real candidates with `rusqlite` already in the dep graph:

| | rusqlite_migration | refinery | sqlx::migrate! | hand-rolled |
|---|---|---|---|---|
| Native to `rusqlite::Connection` | ✅ | needs shim | ❌ pulls sqlx | ✅ |
| SQL strings | ✅ | ✅ `V<n>__name.sql` | ✅ | ✅ via `include_str!` |
| Rust closures per migration | ✅ first-class | ❌ | ❌ | ✅ |
| Custom version-tracking table | ❌ (uses `PRAGMA user_version` only, by design) | ❌ (uses `refinery_schema_history`) | ❌ (uses `_sqlx_migrations`) | ✅ |
| Multi-DB | SQLite-only | PG/MySQL/SQLite | PG/MySQL/SQLite/MSSQL | n/a |

The initial recommendation was `rusqlite_migration`, but verifying its API surface (crate version `1.3.1`, the one compatible with our pinned `rusqlite = "0.32"`) revealed a hard constraint: it does **not** support a custom version-tracking table — `PRAGMA user_version` is hard-wired. Upgrading to the 2.x line doesn't fix this; the design is "no custom table" across all versions. `refinery` has the same flavor of issue (its own `refinery_schema_history`). Since Phase 0 deliberately seeded `schema_migrations` as the canonical version-tracking surface and the slicing entry pinned it as the contract A1 extends, sticking with the off-the-shelf tools would mean keeping `schema_migrations` as a parallel audit log written manually from each migration body — collapsing back to most of the hand-rolled code anyway.

Hand-rolled is roughly 50 LOC: a sorted iteration over embedded SQL strings + closure `fn` registrations, each step run inside `conn.transaction()`, each writing its own row into `schema_migrations`. The "supports Rust closures" requirement is trivial without a library. Zero new dependencies, one source of truth for schema version.

### Confirmed A1 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Migration tool | **Hand-rolled (~50 LOC)** | None of the off-the-shelf tools support a custom version-tracking table; Phase 0's `schema_migrations` is canonical; hand-rolled gives one source of truth and zero new deps |
| Migration direction | **Forward-only** | SQLite's column-drop story makes faithful down-migrations rare; backup files are the real recovery path; restoring `corpus.db.bak.<n>` is the documented recovery flow |
| Backup policy | **Always back up before any migration run** | Classifying "destructive" per-migration is error-prone; SQLite file copies are cheap; pruning old backups is a separate concern |
| Version-tracking table | **Keep custom `schema_migrations`, extend it** | Preserves Phase-0 continuity; add `name TEXT` and `checksum TEXT` columns for provenance and tamper detection; written atomically with each migration in the same transaction |

### Layout

- `crates/engine/migrations/V<n>__<short_name>.sql` — versioned SQL migration files, embedded via `include_str!` at compile time. `V1` is the Phase-0 baseline (`project` + `bundle` + `schema_migrations`), restated so a fresh DB walks the same path as upgraded DBs.
- `crates/engine/src/corpus/migrations.rs` — registry: a `static` slice of `Migration { version, name, kind: Sql(&'static str) | Rust(fn(&Transaction) -> Result<()>), checksum: &'static str }`, plus a `pub fn run(conn: &mut Connection) -> Result<MigrationOutcome>` that applies anything missing.
- `crates/engine/src/corpus.rs` — refactored: `Project::create` calls `migrations::run` on a fresh DB; `Project::open` runs the forward-compat clamp first, then the backup, then `migrations::run` if anything is pending.
- `crates/engine/tests/migrations.rs` — integration tests. For each version `N ≥ 2`: seed a DB at `N-1` (using only the SQL up to `V<N-1>`), apply forward, assert post-migration invariants. Plus one "fresh-create from latest" smoke test that walks the full chain on an empty DB.

### Schema_migrations extension

Phase-0 columns: `version INTEGER PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP`.
A1 adds:
- `name TEXT NOT NULL` — short slug from the migration filename, for listing and grepping
- `checksum TEXT NOT NULL` — SHA-256 of the SQL body (or `"rust:<fn-name>"` for closure migrations), so a future engine can detect that a previously-applied migration's contents have since changed (tamper / accidental edit). Computed at compile time via a tiny `build.rs` so the constant lives next to the SQL.

The `V1` migration writes its own row into `schema_migrations` to seed the new columns. Phase-0 DBs already in the wild would have a row with `version=1, name=NULL, checksum=NULL`; a `V2` migration handles the backfill before the B1 schema work lands on top.

### Forward-compat clamp

After opening the connection, read `MAX(version)` from `schema_migrations`. If that exceeds the highest version known at compile time (the last entry in the static migration slice), `Project::open` returns `EngineError::SchemaTooNew { db_version, engine_max }` instead of running any analysis. The error message points the user at upgrading the engine.

### Backup mechanics

Before invoking the migration runner, if `MAX(version) < engine_max`:
1. Issue `PRAGMA wal_checkpoint(TRUNCATE)` to flush WAL state into the main file.
2. Copy `corpus.db` → `corpus.db.bak.<current_version>` via `std::fs::copy`.
3. Apply migrations.

Backups are not garbage-collected by A1 — that's a future concern (a `sadda corpus gc-backups` CLI verb, or a `project.toml` retention policy). Disk overhead for v1 corpora is modest enough that this can wait.

### What this entry doesn't decide

- **Backup retention policy.** Out of scope for A1. Likely a CLI verb later.
- **Closure-migration ergonomics.** The first closure migration arrives in B1 or B2; the exact call-site shape is determined when a real case appears.
- **Migration linting.** A future `cargo xtask check-migrations` (verify checksums on disk against `schema_migrations`, no out-of-order files) is plausible; A1 ships only the runtime checks.

### Sources / references

- 2026-05-21 Phase 1 slicing entry (this entry expands its A1 row)
- 2026-05-18 corpus data-model entry (B-cluster scope this framework will carry)
- `rusqlite_migration` 1.3.1 API verification (custom version table not supported): https://docs.rs/rusqlite_migration/1.3.1/rusqlite_migration/

---

## 2026-05-21 — Phase 1 slicing: 12 slices in 7 clusters toward 0.1

Goal: sequence Phase 1's deliverables (full corpus schema, six tier types, DSP suite, TextGrid + EAF I/O, live recording, stability decorators, type stubs, recipes, migration framework) into a concrete commit-by-commit ordering. The 2026-05-18 milestone-plan entry committed to Phase 1's *scope*; this entry commits to its *cadence and ordering*.

### What "vertical slice" means in Phase 1

Phase 0's slices touched every architectural layer (engine → corpus → Python → app → UniFFI) per commit. Phase 1's release target is **the Python library on PyPI**, so "vertical" re-reads as: **each slice ends in a usable Python entry point**, not "touches the desktop app." The egui surface stays minimal until Phase 2.

### What changes from Phase 0

| | Phase 0 | Phase 1 |
|---|---|---|
| Slices | 8 | 12 |
| Average LOC / slice | ~375 | ~750 (est.) |
| Layers crossed per slice | all of them | 2–3 (engine + PyO3 + tests) |
| Release at end | 0.0 internal | **0.1 to PyPI** |

Slices are roughly 2× thicker than Phase 0 because the work shifts from glue (one layer's worth at a time) to depth (Parquet sidecar I/O alone exceeds the entire Phase 0 corpus crate). Keep one focused commit per slice; CI green after each.

### Decomposition: 7 clusters

Work-items decompose along three orthogonal lines: cross-cutting infrastructure (which gates everything), corpus schema expansion (strict internal dependency chain), and surface work (DSP, I/O, recording, recipes — mostly independent of each other; depends on corpus).

**Cluster A — Cross-cutting infrastructure** (lands first; gates the rest)

1. **Migration framework.** `sqlx::migrate!` or `refinery` wired in; `schema_migrations` table extended (the table itself was seeded in Phase 0 piece 5); engine refuses to open a DB newer than it knows (forward-compat clamp); `corpus.db.bak.<old_version>` written before any destructive migration; per-migration tests. Lands before any schema expansion.
2. **Stability decorators + type stubs scaffolding.** `@stable / @provisional / @experimental` Python decorators that emit one-time runtime warnings; `pyo3-stub-gen` integrated into the maturin build; `py.typed` marker added; existing Phase-0 APIs (`sadda.version`, `sadda.load_wav`, `sadda.f0`) tiered. Sets the contract that every subsequent slice marks its API surface.

**Cluster B — Corpus schema expansion** (strict order: B1 → B2 → B3)

3. **Full entity schema + AuditLog.** Speaker, Session, Bundle (extended), Tier (header), Entity, EntityRef, Instrument, Protocol. `extra: json` columns throughout. Append-only AuditLog with mutation triggers. ProcessingRun table (the renamed ModelRun per the ML-registry entry).
4. **Sparse tier types.** `interval`, `point`, `reference` with CRUD + the first cut of `proj.query(...) → polars.DataFrame`. Parent-child cardinality enforced at insert time.
5. **Dense tier types + Parquet sidecars.** `continuous_numeric`, `continuous_vector`, `categorical_sampled`. `DerivedSignal` registration rows. Reader/writer in `engine::storage::parquet`; mmap-friendly load path so AI-engineer users can `pl.scan_parquet` directly.

**Cluster C — DSP surface** (independent of B; can interleave)

6. **Foundational DSP.** Windowing (Hann, Hamming, Blackman, Gaussian, Kaiser), STFT, spectrogram, intensity. Pure functions over `&[f32]`; no corpus dependency. Polars-friendly outputs.
7. **Advanced DSP.** Formants via LPC + root-solver; MFCC (mel-filterbank → DCT); refined pitch with voicing decision (extends Phase 0's autocorrelation tracker). Stays inside `engine::dsp`; PyO3 wrappers in `crates/python`.

**Cluster D — Interop I/O**

8. **TextGrid round-trip.** IntervalTier and TextTier import + export; JSON-sentinel (`{json:{…}}`) for attribute round-trip; explicit lossiness documentation. The adoption hinge for Praat users.
9. **EAF round-trip (bidirectional).** ELAN tier types (ALIGNABLE_ANNOTATION, TIME_SUBDIVISION, SYMBOLIC_ASSOCIATION); `parent_tier_ref` preserved; XML round-trip stable enough that ELAN can re-open exports without warnings.

**Cluster E — Recording**

10. **Live recording (cpal).** `.in_progress/` flow → atomic commit; cpal cross-platform driver; metering callbacks. JACK as a stretch goal — if cpal absorbs the time budget, push JACK to a 0.1.x patch release.

**Cluster F — Reproducibility**

11. **`sadda.recipe.record()` / `.replay()`.** Context manager that logs every analysis call through `ProcessingRun` + `AuditLog`; serializes a recipe as a Python script the user can re-run on the same project or another. Connects the cross-cutting reproducibility need surfaced across corpus + refdist + clinical entries.

**Cluster G — Release**

12. **0.1.0 to PyPI + mkdocs-material docs site.** Claim `sadda` on PyPI; mkdocs-material with mkdocstrings auto-rendering API reference from the Rust `///` doc comments (via a cargo-doc → markdown bridge) and the Python stubs; GitHub Pages hosting; first quickstart tutorial. The 2026-05-21 docs-strategy entry already pinned this as the docs-site start point.

### Dependency graph

```
A1 ─┐
    ├─→ B1 ─→ B2 ─→ {D1, D2, F1}
A2 ─┘    │
         ├─→ B3 ─→ F1
         └─→ E1

C1, C2 ──→ (independent; interleave with B)

G12 = last (releases everything above)
```

Concretely: A1 and A2 first (parallel-OK between them but both before B*). Then B1. C* can begin any time A2 is done — they touch no schema. B2 + B3 follow B1. E1 follows B1. D1 + D2 follow B2 (need at least sparse tiers). F1 follows B1 + the Python API instrumentation that A2 sets up. G12 last.

### Parallel risk spike, off-track

**Embedded CPython distribution.** Per the milestone plan, runs alongside Phase 1 as a fail-fast experiment: one signed macOS `.app` with bundled Python running a real script end-to-end. Validates the packaging story before Phase 7 commits to it. Does **not** gate the 0.1 release — Phase 0's `crates/script-engine` already proved the embed works; the spike is about *shipping* the embed. Pick this up when Mac time is available; tracked separately from the 12-slice plan.

### Confirmed slicing decisions

| Item | Decision | Reasoning |
|---|---|---|
| First slice | **Migration framework (A1)** | Infrastructure-first. Every schema change after this rides on solid rails. The first commit is invisible to users; that's fine — Phase 1's first commit is not a marketing release |
| Slice granularity | **12 slices, ~1 commit each** | Match Phase 0's cadence; atomic CI; easy review. Resist combining clusters into mega-PRs |
| EAF scope at 0.1 | **Bidirectional** | Don't preemptively apply the cut line. EAF round-trip is in the cut list but cut from default only under pressure; aim for both at 0.1 |
| PyPI timing | **Single 0.1.0 release at end of Phase 1** | No stub 0.0.x. Name-squatting risk is mitigated by the public `sadda-speech` GitHub org claim already being live. Single launch event |

### What this entry doesn't decide

- **Migration tool choice (`sqlx::migrate!` vs `refinery`).** Settled inside A1's design pass. Both are viable; pick after a small spike.
- **Recipe serialization format inside F1.** Python script is the user-facing artifact; whether the persistent log is JSON, TOML, or SQL rows is internal.
- **Which sub-DSP measure goes in C1 vs C2.** Drafted above but the precise cut can shift as code is written.
- **Live recording UX for the Python API.** `sadda.live.start_session()` is the entry point; subscriber-decorator semantics (per the API-surface entry) are settled at design level but need a concrete spike inside E1.
- **mkdocs-material theme + nav layout.** Settled inside G12; not architectural.

### Pace and revisit cadence

- Per the milestone plan, Phase 1 is 3–4 months at solo part-time. 12 slices over ~16 weeks ≈ one slice every 1–2 weeks.
- Revisit this entry after slice 6 (mid-phase). If cadence is slower than 1/week, consider deferring EAF to import-only (the cut-line move) and/or splitting JACK out of E1 entirely.
- After 0.1 ships: real PyPI users land; feedback may reshape Phase 2 scope. Phase 1 cadence numbers feed into the milestone-plan revisit.

### Sources / references

- 2026-05-18 milestone-plan entry (this entry is downstream of its Phase 1 row)
- 2026-05-18 corpus data-model entry (B-cluster scope)
- 2026-05-18 Python API surface entry (A2 stability decorators; F1 recipes)
- 2026-05-21 documentation strategy entry (G12 docs site)
- "Tracer Bullets" — Pragmatic Programmer (vertical-slice principle): https://pragprog.com/tips/

---

## 2026-05-21 — Documentation strategy: discipline now, site at 0.1, polish at 1.0

Goal: settle when and how docs grow. The milestone plan already committed to *"documentation grows incrementally per phase, not as a separate final phase"* but didn't schedule it. This entry pins concrete starts per phase.

### Three layers, three different starts

| Doc layer | What it is | Starts |
|---|---|---|
| **API reference** | `///` doc comments on every public Rust item, docstrings on every PyO3-exposed Python item. Auto-compiles to docs via `cargo doc` and (later) mkdocstrings | **Now (Phase 0).** Enforced via `#![warn(missing_docs)]` on library crates + CI's `-D warnings` |
| **User-facing docs site** | mkdocs-material + mkdocstrings; conceptual guides drawn from DEVLOG entries; quickstart tutorial; auto-generated API reference; GitHub Pages hosting | **End of Phase 1 (0.1 PyPI release)** — when there's a Python API stable enough for outside users to actually try |
| **Polished docs site + migration** | Parselmouth migration docs, notebook tutorials, bundled sample projects, visual GUI walkthroughs | **Phase 7 (toward 1.0)**, per the existing milestone-plan deliverables |

### Why these particular start points

- **API reference now.** Cost of writing a one-line doc comment when you write a function is near-zero; cost of retrofitting hundreds of public items at 0.1 is a multi-day audit nobody wants. Enforcing via a lint means the discipline doesn't depend on remembering. PyO3 `///` comments on `#[pyfunction]` / `#[pymethods]` become Python docstrings automatically — same source of truth.
- **Docs site at end of Phase 1.** Before 0.1 the Python API isn't stable enough to write tutorials against — examples would rot constantly. After 0.1 there's an audience (PyPI installers) who genuinely benefit. Earlier than that is writing for a reader who doesn't exist yet.
- **GUI tutorials at Phase 2.** Screenshots of an unstable GUI rot every week. Wait until 0.2 ships.
- **Migration docs at Phase 7.** Praat-to-sadda migration is high-effort writing; do it once when the API is stable and not before.

### Concrete starts this turn

- `#![warn(missing_docs)]` enabled on the four library crates: `sadda-engine`, `sadda-python`, `sadda-script-engine`, `sadda-uniffi`. The `app` binary crate is excluded — binaries don't expose a library surface.
- Existing public items that lacked doc comments are filled in (audit done across the four library crates in the same commit).
- CI's `-D warnings` makes future PRs that omit doc comments fail. Adding a public item is now contract-bound to come with a one-line doc.

### Cross-references / what this entry updates

- Extends the 2026-05-18 milestone-plan entry's "documentation grows incrementally" principle with a concrete per-phase schedule.
- Touches the API-surface entry (2026-05-18): `sadda.ml` and other PROVISIONAL surfaces still get docs, but their docstrings can call out the stability tier explicitly.

### Sources / references

- mkdocs-material: https://squidfunk.github.io/mkdocs-material
- mkdocstrings (Python API rendering): https://mkdocstrings.github.io
- rustdoc book: https://doc.rust-lang.org/rustdoc/
- PyO3 docs guide on docstrings: https://pyo3.rs/v0.28.0/function/signature.html#documenting-functions

---

## 2026-05-20 — Profile catalog: five v1 profiles over the entity / tier / refdist / measure surfaces

Goal: close the "profile catalog" open follow-up from the corpus entry. Specify which profiles ship at v1, what each profile concretely contains, and the policy around default profile, authoring path, and mid-flight profile changes. The mechanism (profile = schema validator over JSON `extra`) was already settled in the corpus entry; this entry settles content and governance.

### Profile = a bundle of seven things

A sadda profile is a coherent default-state package across seven surfaces:

1. **Entity `extra` schema validators** — which fields are required on Speaker / Patient / Case / Participant entities; drives typed GUI forms over the flexible JSON storage.
2. **Default tier templates** — what tier set a new bundle gets out of the box.
3. **Recommended reference distributions** — the subset of refdist starter entries pre-recommended for this workflow.
4. **Default measure surfaces** — which analyses the GUI prioritizes.
5. **Workflow defaults** — what panels/widgets surface; what GUI mode the project opens to.
6. **Optional capabilities flagged on** — e.g. calibration mandatory, audit logging emphasized, community-consent flow.
7. **Terminology** — labels like "Patient" vs "Speaker" vs "Case" vs "Participant".

### Profiles shipping at v1

Five profiles, mapped from the 8 user groups in the 2026-05-16 entry. Voice Training (voice coaches + L2 learners, combined) deferred to v1.x — it's mobile-heavy per its source entry, so shipping it pre-Phase-6 would feel half-baked. Speech AI / ML engineers don't get a profile because their needs cut across all five via the ML feature surface (`sadda.ml`, embedding tiers). Bioacoustics stays designed-in (frequency-range agnosticism) but doesn't get a v1 profile.

#### 1. Phonetician *(default)*

The user's own profile; also the fallback when no profile is specified.

| Surface | Content |
|---|---|
| **Entities** | `Speaker` with `extra`: l1_language (ISO 639-3), dialect, age_band, sex_at_birth (optional), handedness (optional) |
| **Tier template** | `phones` (interval, IPA) → `words` (interval, parent of phones) → `utterances` (interval, parent of words); `notes` (point, free-text) |
| **Recommended distributions** | Hillenbrand 1995, Peterson-Barney 1952, one VOT reference |
| **Default measures** | F0 (autocorrelation), formants F1–F3, duration, intensity, spectrogram |
| **Workflow defaults** | Praat-like layout: spectrogram + waveform + tier strip with sync cursor; refdist overlays visible by default |
| **Capabilities** | Calibration optional; audit logging standard (always-on, low-emphasis) |
| **Terminology** | "Speaker" / "Session" / "Bundle" |

#### 2. Clinical

| Surface | Content |
|---|---|
| **Entities** | `Patient` with `extra`: MRN, DOB, sex_at_birth, treating_clinician, diagnosis_codes (ICD-10 strings), clinical_history (optional rich text). `Encounter` with `extra`: type (initial_eval / follow_up / post_treatment), notes. Sessions link to encounters; bundles get extras: protocol (CAPE-V / Rainbow / sustained_vowel / connected_speech), calibration_set_id (required when SPL is measured) |
| **Tier template** | Protocol-driven: sustained-vowel marker (single interval); CAPE-V auditory-perceptual rating tier (categorical_sampled); phones tier optional |
| **Recommended distributions** | Age/sex-banded jitter/shimmer/HNR norms; AVQI/ABI normative ranges; Voice Range Profile norms |
| **Default measures** | AVQI, ABI, CPP, jitter (multiple variants), shimmer, HNR, F0 statistics, calibrated SPL; Voice Range Profile / phonetogram view; longitudinal pre/post comparison |
| **Workflow defaults** | Patient-as-first-class: GUI opens to patient list. Protocol-driven recording flow: select protocol → guided record → auto-analyze → draft report. Research-use-only watermark on exports (per the 2026-05-18 clinical regulatory entry, posture 3) |
| **Capabilities** | Calibration *required* (red banner if uncalibrated); audit logging emphasized in UI |
| **Terminology** | "Patient" / "Encounter" / "Treatment Session" / "Bundle" |

#### 3. Forensic

| Surface | Content |
|---|---|
| **Entities** | `Case` with `extra`: case_id, lead_investigator, jurisdiction, case_status (open / closed / archived). `Sample` with `extra`: source (questioned / known), evidence_id, chain_of_custody_events (append-only event log backed by AuditLog). `Speaker` anonymized: speaker_alias (S1, S2, …), demographic_band (no exact age). Bundle `extra`: sample_id (FK), recording_quality_notes, sealed_at, sealed_by |
| **Tier template** | Long-term formant tracking (continuous_numeric); F0 distribution sampling (continuous_numeric); articulation-rate markers (interval) |
| **Recommended distributions** | Population reference data: age/sex/dialect-banded F0; formant LTAS distributions; acoustic-feature distributions for LR computation |
| **Default measures** | Long-term formant distributions; F0 statistics (mean / SD / range); articulation rate; speaker similarity (LR-framework analyses); LTAS |
| **Workflow defaults** | Case-as-first-class: GUI opens to case list. Audit log prominent in the case timeline view. Chain-of-custody UI for evidence-handling events. `sadda.recipe.record` mode default-on (every analysis logged as a recipe). Reproducible "raw audio → report" pipeline export |
| **Capabilities** | Audit logging *emphasized + mandatory* recipe recording for all analyses; encryption at rest emphasized (per cross-cutting pattern J); calibration optional but recommended |
| **Terminology** | "Case" / "Sample" / "Speaker (S1)" / "Recording" |

#### 4. Field linguistics

| Surface | Content |
|---|---|
| **Entities** | `Speaker` with `extra`: l1_language (ISO 639-3, required), community_id (FK), consent_status, consent_events (timestamped log), age_band, sex (optional; community-norm-dependent). `Community` with `extra`: language_iso639_3, region, community_consent_doc_ref, data_governance_policy. Session `extra`: location, session_type (elicitation / narrative / conversation), elicitor |
| **Tier template** | Multi-tier glossed: phonetic (interval, IPA) → morphological_gloss (interval, parent: phonetic) → free_translation (interval, parent: morphological_gloss) → notes (interval, free-text) |
| **Recommended distributions** | Less applicable — field work is usually on previously-undocumented populations. Reserve slot for IPA frequency references by major language family; rely on community contribution |
| **Default measures** | F0, formants (surfaced but not prioritized), spectrogram. Annotation tools take primary screen real estate |
| **Workflow defaults** | Annotation-first GUI: multi-tier strip prominent; spectrogram secondary. IPA palette + feature-based phone search front-and-center (per cross-cutting field-linguistics need). Crash-resilient autosave emphasized. Community-consent badge on Speaker entities |
| **Capabilities** | IPA tooling emphasized; community-consent flow mandatory before public-export operations; archival-format export (DELAMAN, ELAR, PARADISEC) prominent in export menu; calibration optional |
| **Terminology** | "Speaker" / "Session" / "Community" / "Elicitation" |

#### 5. Experimenter

| Surface | Content |
|---|---|
| **Entities** | `Participant` with `extra`: participant_id, age, sex, l1_language, consent_status, recruitment_source (lab / Prolific / MTurk / etc.). `Experiment` with `extra`: protocol_def_ref (link to sequence-editor output), task_description, n_target. `Trial` with `extra`: trial_index, condition, stimulus_id (FK), response_data (JSON), reaction_time_ms. `Stimulus` with `extra`: stimulus_type (audio / image / text), source_path, content_description |
| **Tier template** | Trial-aligned interval tier (one interval per trial); response markers (point tier) |
| **Recommended distributions** | Less applicable — experiments produce their own data. Reserve slot for normative reference distributions if a stock paradigm benefits |
| **Default measures** | Trial-level aggregates: reaction time stats, accuracy by condition, F0 / formants extracted per trial. Auto-process recordings (force-align, extract features, tag with trial info per cross-cutting pattern I) |
| **Workflow defaults** | Experiment list view at GUI open. Sequence-editor surface for protocol building (per the 2026-05-18 experiment-runner entry). Auto-tagging: recordings link to trials by metadata. PsychoPy CSV import on by default |
| **Capabilities** | Trial-runner mode enabled; `sadda.recipe.record` mode default-on; calibration optional |
| **Terminology** | "Experiment" / "Trial" / "Participant" / "Stimulus" / "Block" |

### Policy decisions

**Default profile.** When `sadda.new_project(path)` is called without a `profile=` argument, profile is `phonetician`. Mirrors the tool's primary identity (and the user's own profile).

**Built-in profiles only at v1.** No plugin-shipped or user-authored profiles in v1. Custom per-project `extra` schemas can be declared in `project.toml` under `[profile.extensions]` for minor customizations without forking a profile. Reasoning: profile schema will churn during early use; locking the authoring API prematurely would force false stability.

**Cross-profile changes mid-flight.** Allowed via `sadda.profile.change(project, new_profile)`. Engine validates all existing entities against the new profile's `extra` schemas. Required-field gaps are surfaced ("12 patients missing `treating_clinician`") rather than auto-filled. User fixes via batch-edit GUI or reverts. No data is destroyed by a profile change; the underlying `extra` JSON is preserved, only the validator changes.

**Profile selection at project creation.** The "new project" GUI dialog presents the five profiles with one-line descriptions and a "see what changes" expand. Initial profile selection is significant but not irreversible.

**Profile + GUI customization are orthogonal.** Profile drives *defaults*; users can override panel layout, hide/show widgets, adjust terminology preferences within a profile. Per-user preferences live in the user config, not the project — switching machines preserves panel layouts via cloud-config sync (deferred) or manual export.

### Architectural touch points

- **API.** `sadda.new_project(path, profile="phonetician")` accepts profile id. `sadda.profile.change(project, new_profile)` swaps profile. `sadda.profile.list()` returns the catalog. All under `sadda.profile` namespace, PROVISIONAL stability tier (the API entry's existing tiering).
- **Storage.** `project.toml` has a top-level `profile = "..."` field. Profile id is also persisted as a column on the `Project` row in `corpus.db` (already in the corpus entry's entity table).
- **Profile-extension hooks.** `project.toml` `[profile.extensions]` section can declare additional `extra` schema fields and additional tier templates that augment the chosen profile without replacing it. Format spec to be finalized when first per-project customization is requested.
- **Refdist defaults.** Each profile's "recommended distributions" list is consumed by the in-app refdist picker — entries are highlighted as "recommended for this profile" without restricting which distributions can actually be installed.
- **GUI mode switching.** The 2026-05-18 API surface entry's `sadda.app` namespace gains a `set_workflow_mode(profile_id)` call that re-applies workflow defaults; useful for users who want to temporarily try a clinical view of a phonetic project.
- **Audit-log emphasis is profile-driven, not engine-default.** The `AuditLog` table is always populated regardless of profile; profiles only control how prominent it is in the GUI.

### v1 deliverables this entry commits to

1. Five profiles shipped: `phonetician`, `clinical`, `forensic`, `field`, `experimenter`.
2. Each profile's seven surfaces (entities / tiers / refdist / measures / workflow / capabilities / terminology) implemented to the level specified above.
3. `project.toml` `profile` field and `[profile.extensions]` section.
4. `sadda.new_project(path, profile=...)`, `sadda.profile.change()`, `sadda.profile.list()`.
5. New-project GUI dialog surfacing the five profiles with one-line descriptions.
6. Cross-profile change flow with validation gap surfacing.
7. Per-profile refdist recommendation lists (consumed by refdist picker GUI).

### Open trade-offs / deferred

- **Voice Training profile (v1.x).** Combined voice coach + L2 learner profile. Mobile-heavy; better landed when Phase 6 ships. Workflow shape: own-voice baseline; user-defined target zones (not population norms); progress tracking; recording-and-compare against a model. Reference distributions: target zones (`measure.kind = "target_zone"` from refdist entry) and mobile-app-friendly subset.
- **Plugin-shipped profiles.** Once the profile schema has stabilized through real v1 use, allow plugins to declare profiles via a manifest. v1.x feature; possible drivers: a research lab shipping a custom protocol-bundle profile, or a clinical specialty (singing voice, pediatric) wanting a sub-profile.
- **User-authored profiles.** v1.x feature once authoring API is designed. Probably scaffolded via `sadda profile init <name>` CLI producing a profile-definition TOML the user can edit.
- **Profile catalog versioning.** When profile `extra` schemas evolve mid-v1 (e.g. clinical adds a required field), how do existing projects upgrade? Likely: profile schemas are themselves semver-versioned; project.toml records the profile version at creation; migration utilities walk `extra` payloads (per the corpus entry's migration policy).
- **Bioacoustics adjacent profile.** Per the user-groups entry, bioacoustics has real adoption-crossover potential. v1.x or community-contributed profile makes sense once the plugin-authoring path is open.
- **Distribution recommendations need actual curation.** This entry sketches "recommended distributions" lists at the *category* level (clinical: jitter/shimmer/HNR norms); the actual distribution IDs need to match what's curated in tier 1 of the refdist registry. Sync these lists with refdist starter set as Phase 1 work progresses.

### Cross-references / what this entry updates

- Closes the "profile catalog" follow-up from the 2026-05-18 corpus entry.
- Closes the "profile-driven defaults" follow-up from the 2026-05-18 refdist entry.
- Extends the 2026-05-18 API entry: adds `sadda.profile` namespace with `list / change` and concretizes `new_project(profile=...)`.
- Threads through: clinical entry (research-use-only labeling driven by `clinical` profile), forensic patterns from the user-groups entry, field linguistics archival export emphasis, experimenter trial metadata flow.

### Sources / references

- EMU-SDMS per-database schema: https://ips-lmu.github.io/EMU.html
- ELAN tier templates: https://www.mpi.nl/tools/elan
- PsychoPy Builder paradigm templates: https://www.psychopy.org/builder
- FLEx (SIL FieldWorks): https://software.sil.org/fieldworks
- DELAMAN archive standards: https://www.delaman.org
- ICD-10: https://icd.who.int/browse10
- CAPE-V: Kempster et al. (2009), *American Journal of Speech-Language Pathology*

---

## 2026-05-20 — ML model registry: provenance audit + parallel-to-refdist distribution

Goal: close the deferred "ML model registry scope" item from the corpus entry. The phrase as flagged conflated two structurally separate concerns; this entry separates them and specifies both.

### Two layers, each with its own design

| Layer | Scope | Lives where | Question it answers |
|---|---|---|---|
| **Provenance** | Audit record of every processing run on a bundle | Per-project SQLite | "Where did this tier come from?" |
| **Distribution** | How models reach a user's machine and become loadable by ID | User-level cache + cross-project registry | "Where do I get `wav2vec2-base`?" |

The corpus entry's `ModelRun` table was the provenance layer. The articulatory entry's open question ("downloaded via same registry as refdist, or separate?") was the distribution layer.

### Layer 1: Provenance — rename `ModelRun` → `ProcessingRun`

The clinical entry's commitment ("every measure records which algorithm version was used … plumbed through the ML model registry") already implies coverage of non-ML algorithms (pitch, formants, AVQI) alongside ML models. `ModelRun` was misleading. Rename to **`ProcessingRun`** with a `kind` discriminator.

```
ProcessingRun
  id                primary key
  bundle_id         FK
  kind              enum: ml_model | dsp_algorithm | clinical_measure | plugin
  processor_id      e.g. "sadda/wav2vec2-base-960h" or "sadda.dsp.pitch"
  processor_version semver
  weights_checksum  nullable — populated for ml_model; null for pure-DSP
  parameters        JSON — call args
  input_tier_ids    JSON — what inputs were consumed
  output_tier_ids   JSON — what tiers were produced
  output_signal_ids JSON — for DerivedSignals (Parquet sidecars)
  started_at        timestamp
  finished_at       timestamp
  status            enum: ok | error | partial
  error_message     nullable
  recipe_run_id     nullable FK — set when invoked from sadda.recipe.record
```

Coverage by `kind`:

- `ml_model` — registry-resolved models; processor_id from registry namespace; weights_checksum populated
- `dsp_algorithm` — built-in DSP; processor_id is the function path (`sadda.dsp.pitch`); version is the sadda version
- `clinical_measure` — composite measures; processor_id e.g. `sadda.clinical.avqi`; version is sadda + measure spec version
- `plugin` — plugin-supplied analyzers; processor_id includes the plugin id

A single audit timeline per bundle answers "where did every signal/annotation come from" uniformly. Citation export (planned in the clinical entry) walks this table.

### Layer 2: Distribution — parallel registry, shared protocol with refdist, HF passthrough escape hatch

Locked: separate registry repo, reuses the refdist mechanism. Mirrors three-tier model (bundled / curated / community), TOML-manifest format, GitHub-Pages index, CI validation, in-app publishing flow, project pinning, automatic citation. Where structurally different from refdist:

- **Larger artifacts.** Weights can be 10MB–5GB+. GitHub release assets cap ~2GB; above that, the manifest declares an external mirror (HuggingFace, Zenodo, S3) with file checksum. Engine downloads from the declared URL and validates against the checksum.
- **Compute hints in manifest.** RAM, GPU/Metal optional/required, FP precision. Engine surfaces "won't run on your machine" before download.
- **CI validation is shallower.** Curated-tier CI can checksum, license-check, and (optionally) run a smoke test on a small input — but it can't end-to-end validate accuracy. Trust signals lean more on editorial review than for refdist.
- **Output spec ties into the tier model.** Manifest declares `output.tier_kind` in our tier vocabulary so the engine knows what tier type inference results produce.

#### Manifest sketch

```toml
id = "sadda/wav2vec2-base-960h"
version = "1.0.0"
title = "wav2vec2-base self-supervised speech model"
upstream_source = "https://huggingface.co/facebook/wav2vec2-base-960h"
license = "Apache-2.0"

[model]
kind = "embedding"               # embedding | transcription | vad | segmentation | alignment | feature
format = "onnx"                  # onnx | gguf | safetensors | savedmodel
file = "model.onnx"              # OR url = "..." + url_checksum = "sha256:..." for external mirror
file_checksum = "sha256:..."

[input]
modality = "audio"               # audio | video | both
sample_rate_hz = 16000
channels = 1

[output]
tier_kind = "continuous_vector"
channels = 768
sample_rate_hz = 50

[compute]
cpu_min_ram_mb = 1024
gpu = "optional"                 # required | optional | unsupported
quantization_available = ["int8"]

[citation]
bibtex = "..."
```

#### ID schemes — three resolvable forms via `sadda.ml.load_model(id)`

- `"sadda/wav2vec2-base-960h"` — curated registry; version optional (latest if omitted), should be pinned in `project.toml` for reproducibility
- `"hf://facebook/wav2vec2-base-960h"` — HuggingFace passthrough; no curation, no manifest, no quality guarantee. `ProcessingRun.processor_id` records `"hf://..."` verbatim so provenance is honest about the trust tier
- `"local:///abs/path/model.onnx"` — local file; for in-development or air-gapped use

HF passthrough behavior:
- Downloaded via the HF Hub protocol; cached alongside curated models
- File format opaque at fetch time; load attempts the supported runtime (ort or fallback); fail loud if format unsupported
- Auth via `HF_TOKEN` env or sadda config; not required for public models
- HF revision SHA recorded in `ProcessingRun.weights_checksum`, so recipe replay can pin the exact upstream version

### Canonical format: ONNX, with documented exceptions

The Phase 3 milestone commits to `ort runtime`. Implication: **ONNX is the canonical format for curated-tier publishing.** Manifest's `format` field accepts `gguf` / `safetensors` / `savedmodel` for specific exceptions — the likely one being a whisper.cpp GGUF variant for embedded mobile Whisper in Phase 6 — but the default and recommended publish path is ONNX. Format-conversion (e.g. `optimum`) belongs to the publishing workflow, not the runtime. HF passthrough sidesteps this: we load whatever HF has, attempting available runtimes.

### Storage and cache

```
~/.local/share/sadda/models/
├── sadda/                                    ← curated namespace
│   └── wav2vec2-base-960h/
│       ├── 1.0.0/
│       │   ├── manifest.toml
│       │   ├── model.onnx
│       │   └── LICENSE
│       └── 1.0.1/
├── hf/                                       ← HF passthrough cache
│   └── facebook/
│       └── wav2vec2-base-960h/
│           └── <hf-revision-sha>/...
└── .cache_meta.json                          ← checksums, last-used timestamps
```

Cache eviction is manual via CLI in v1 (`sadda model gc`). Automatic LRU eviction with a configurable disk cap is a follow-up if real users hit disk pressure.

### v1 curated starter set

Per Phase 3 / Phase 4 milestones:

| Model | Purpose | Phase | Notes |
|---|---|---|---|
| `sadda/silero-vad` | Voice activity detection | 3 | Small (~2MB); bundled with the app rather than downloaded |
| `sadda/wav2vec2-base-960h` | Self-supervised speech embeddings | 3 | On-demand download |
| `sadda/whisper-tiny` | Speech transcription (entry-level) | 3 | On-demand; good enough for trial-log alignment |
| `sadda/whisper-base` | Speech transcription (better quality) | 3 | Optional larger variant |
| `sadda/tongue-contour-v1` | Ultrasound tongue-contour segmentation | 4 | Specific model choice still open per articulatory entry |

VAD is the only bundled model; the rest download on first use, with sizes surfaced in the GUI before download begins.

### Architectural touch points

- **API surface.** `sadda.ml` stays at the PROVISIONAL stability tier from the API entry. Loader signature: `sadda.ml.load_model(id, version=None, **load_opts) -> Model`. Model objects expose `extract_embeddings` / `transcribe` / `vad` / `align` / `segment` depending on `model.kind`.
- **Recipe replay.** Curated-registry IDs resolve to the pinned version; HF passthrough IDs resolve to the original HF revision SHA recorded at first run; local-path IDs require the file to be present.
- **Project pinning.** `project.toml` gets a `[models]` block listing `id` + `version` used in the project. Mirrors refdist pinning.
- **In-app publishing.** Same flow as refdist tier-3 publishing: GUI scaffolds a manifest, validates, opens a PR against the model-registry repo using the user's GitHub credentials. For weights, GUI prompts for upload target (GitHub release asset for ≤2GB; external mirror URL otherwise).
- **Plugin interaction.** Native and Python plugins can register models via the same manifest format pointed at a local file — how third parties ship their own analyzers without going through the public registry.

### v1 deliverables this entry commits to

1. `ProcessingRun` table (replaces `ModelRun`) with the kind-discriminator schema enumerated above.
2. Model registry repo on GitHub (`sadda-speech/model-registry`) with three-tier structure mirroring refdist.
3. CI validation pipeline for model entries (manifest schema, license check, file checksum, optional smoke-test runner).
4. GitHub Pages-rendered index JSON consumable by the engine.
5. `sadda.ml.load_model` resolver covering curated / HF passthrough / local schemes.
6. ONNX-canonical curated-tier publishing path; HF passthrough as escape hatch.
7. Project pinning of model versions in `project.toml`.
8. Curated starter set: VAD (bundled), wav2vec2-base, whisper-tiny, whisper-base in Phase 3; tongue-contour-v1 in Phase 4.
9. User-level cache at `~/.local/share/sadda/models/` with manual GC command.

### Open trade-offs / deferred

- **HF passthrough auth UX.** Where does `HF_TOKEN` get configured (env, config file, GUI prompt)? Gated models need this; public ones don't. Defer until first real user hits the wall.
- **Mobile model story.** Phase 6. ONNX Runtime Mobile vs. ORT + quantization vs. whisper.cpp/llama.cpp variants. Likely its own DEVLOG entry when Phase 6 approaches.
- **Output verification on HF passthrough.** No manifest means no `output.tier_kind` declared. Engine has to either ask the caller to specify or sniff via convention. Probably caller-specified in v1.
- **Tongue-contour-v1 model choice.** Still open per articulatory entry. UNet vs. DeepLabCut-style vs. fine-tune of an existing release. Phase 3 risk spike addresses this.
- **Curated-tier CI smoke-test runner.** Running a model in CI costs CPU minutes; GPU CI is hard. Plausible v1: tiny synthetic input + checksum of expected output. Spec out before opening the registry repo for tier-3 submissions.
- **Plugin-shipped models in `ProcessingRun`.** Plugin-supplied models should still record with the same provenance fields; manifest path convention for plugin-shipped weights needs nailing down.
- **Cache eviction.** Manual at v1; LRU eviction with a configurable cap as a follow-up if disk usage becomes a real complaint.

### Cross-references / what this entry updates

- Closes "ML model registry scope" follow-up from the 2026-05-18 corpus entry.
- Closes the "bundled model distribution channel" question raised in the 2026-05-18 articulatory entry (item 7 in that entry's open trade-offs).
- Updates the corpus entity table: rename `ModelRun` → `ProcessingRun`.
- Refines `sadda.ml` surface from the 2026-05-18 API entry: adds explicit `load_model` resolver and the three ID schemes; `extract_embeddings` / `transcribe` / `vad` / `align` move from top-level `sadda.ml.*` to methods on the returned `Model` object.

### Sources / references

- HuggingFace Hub: https://huggingface.co/docs/hub ; huggingface_hub Python lib: https://huggingface.co/docs/huggingface_hub
- ONNX Runtime: https://onnxruntime.ai ; ort (Rust binding): https://ort.pyke.io
- ONNX Model Zoo: https://github.com/onnx/models
- MFA models registry: https://mfa-models.readthedocs.io
- Silero VAD: https://github.com/snakers4/silero-vad
- Whisper: https://github.com/openai/whisper ; whisper.cpp: https://github.com/ggerganov/whisper.cpp
- wav2vec2 (facebook/wav2vec2-base-960h): https://huggingface.co/facebook/wav2vec2-base-960h
- Hugging Face Optimum (format conversion): https://huggingface.co/docs/optimum

---

## 2026-05-20 — naming decision: project is named "sadda"

Closes the naming open-question from the milestone-plan entry. PyPI name is the first irrevocable public commitment, so this needs to land before any code reaches 0.0 internal — even though no code exists yet.

### Search path

Started from "lapa" (Pali, *talkative, chatty*) — direct echo of Praat's naming logic (Dutch *talk*). Two flavors of speech-words considered:

- **Informal/conversational verbs** (parallels Praat): lapa, jalpa, sallāpa, kathā, ālāpa
- **Technical phonological terms** (aligns with what the tool actually does): vāc, dhvani, svara, śabda, vacana, uccāraṇa

The conversational flavor matches Praat's spirit; the technical flavor matches the tool's actual subject matter. The conversational frame is what a user *does*; the technical frame is what the tool *analyzes*. Settled on the latter as more honest about the tool's nature — Praat's name was a friendly disguise on a deeply technical instrument, and we don't need that disguise in 2026.

### Collisions that killed candidates

PyPI screen against the longlist:

- `lapa` — taken (genomics tool, Mortazavi Lab; active; adjacent scientific-Python field — exactly the collision case to avoid)
- `dhvani` — taken, and the taker is itself a phonetics tool (Hinglish phonetic normalization, IPA-bridged). Hard kill: same field
- `svara`, `vac`, `shabda`, `nada`, `bhasha` — all taken
- `vacana`, `sallapa`, `vaak`, `alapa`, `jalpa`, `katha`, `vada`, `sadda` — all available

### Why sadda

Pali सद्द (cognate of Sanskrit śabda, displaced because śabda/shabda are taken on PyPI):

- **Phonetics-coded directly.** *Sadda-sattha* is the Pali term for the science of language/sound — the Pali grammatical tradition's own name for itself (Kaccāyana). It's not a metaphor; it's the indigenous technical term.
- **Phenomenologically primary.** In Pali Abhidhamma, sadda is one of the six sense-objects — the proper object of hearing. Means "the kind of thing this tool measures."
- **Pali matches the user's initial instinct** (entered via "lapa") while moving from the action-frame to the object-frame.
- **Phonotactically clean.** /sad.də/, two syllables, geminate /-dd-/ gives crisp distinction in search, trivial pronunciation across L1s.
- **Praat parallel preserved.** Both names pick from a tradition where the chosen word carries philological weight in the picker's eyes.

### Namespace status

| Asset | Status |
|---|---|
| PyPI `sadda` | available |
| PyPI `saddha` / `sadda-speech` / `pysadda` (qualified fallbacks) | available |
| `github.com/sadda` | taken (active personal account, Lukáš Adam) — GitHub org uses a qualifier instead |
| `github.com/sadda-speech` | **created as GitHub org** (2026-05-20) |
| `sadda.dev` | available |
| `sadda.org` | available |
| `sadda.io` | parked |

GitHub org: **`sadda-speech`** — descriptive, matches the `python-pillow` / `astral-sh` convention of qualifying the brand name when the bare slug is taken. Other candidates considered (`sadda-dev`, `sadda-tool`, `getsadda`) declined: `sadda-dev` too vague, `sadda-tool` undersells the scope, `getsadda` reads commercial-startup rather than open-source-scientific.

### What still needs doing

- **Sanskritist / Indologist sanity check.** Name was chosen with secondhand Pali knowledge; register and connotation in scholarly Theravāda / modern Pali-studies circles should be verified before formal commit. Not blocking — Pali is a textual liturgical language with limited modern-speaker connotation risk — but worth a five-minute check with someone in Buddhist studies.
- **Trademark search.** "Sadda" appears in Bollywood / Punjabi pop-cultural usage ("Sadda Haq") but not as a software / scientific-instruments mark, as far as casual search shows. A real USPTO / WIPO check should happen before the 0.1 release rather than now.
- ~~**GitHub org name lock-in.**~~ Closed 2026-05-20: `sadda-speech` chosen and org created same day.
- **PyPI name claim — defer to end of Phase 0, not now.** PyPI has no formal reservation; claiming the name means uploading a real release. Doing that today would mean a bare stub sitting on PyPI for 1–2+ years before a real release, which is a long PEP 541 ("project is squatting") exposure window for modest insurance against an unlikely independent collision (sadda is non-obvious in an obscure scholarly language; the slot has been free the entire history of PyPI). Plan: claim with `0.0.1` at end of Phase 0, when there's an actual installable artifact — which both closes the collision window and forecloses the squatting-vulnerability window in the same act. If a collision risk materializes earlier (someone publishes a *sadda* package on PyPI mid-Phase-0), revisit immediately.
- **Historical `sat` references in earlier DEVLOG entries left as-is** — they're the record of how thinking evolved during the placeholder period, not authoritative naming. Code from this point forward uses `sadda`.

### Module-import implication

The top-level Python module becomes `sadda`. Replaces `sat` in the API surface entry — `import sadda` rather than `import sat`; `sadda.dsp`, `sadda.corpus`, `sadda.app`, `sadda.recipe.record()`, `sadda.refdist`, etc. The earlier API-surface entry's code examples remain valid as patterns; just substitute the module name.

---

## 2026-05-18 — v1 milestone plan: vertical slice first, seven phases, incremental 0.x releases

Goal: sequence all the v1 commitments accumulated across the previous 9 entries into an executable plan. Unlike the other entries, this is a project plan — a working roadmap, not a binding architecture decision. It will drift; revisit periodically.

### Scale, named honestly

Pulled together across the prior entries, the v1 surface is enormous: Rust engine + egui desktop GUI + PyO3 Python module + iOS + Android UniFFI shells + Python + native plugin architecture + six tier types + video + live recording + DSP suite + ML inference + six clinical algorithms with validation + articulatory imports + ultrasound video with integrated tongue contour tracker + experiment runner + reference distribution registry + embedded CPython + packaging on three desktop and two mobile platforms.

**Pace assumption: solo part-time** (10–15 hours / week sustainable). Realistic v1.0 timeline at this pace: **~3–4 years**. Timeline compresses substantially if pace moves to full-time (~18–24 months) or if contributors join after the first 0.x release. The plan is built around this conservative pace; faster pace shortens phase durations but doesn't change phase ordering.

### Organizing principles

1. **Vertical slice first.** Phase 0 builds end-to-end through every architectural layer (engine → corpus → Python → GUI → UniFFI → embedded CPython) before any layer is built out in depth. If something is broken architecturally, we want to know in month 2, not month 14.
2. **Risk spikes run in parallel** with main-path phases. Small, contained, fail-fast experiments to settle unknowns before they block work.
3. **Incremental 0.x releases** at the end of each phase. Each release is a usable artifact for some real audience; not internal-only.
4. **Contributor onramp** progressively opens as APIs stabilize: Python library contributors after 0.1; GUI contributors after 0.2; plugin authors after 0.4.

### Phases and milestones

| Phase | Duration (solo part-time) | Release | What ships |
|---|---|---|---|
| **0. Vertical slice** | 1–2 mo | 0.0 internal | Engine + minimal corpus + PyO3 + WAV loader + autocorrelation pitch + egui waveform/pitch view + embedded CPython script panel + one UniFFI method to Swift CLI |
| **1. Foundations** | 3–4 mo | **0.1 — Python library on PyPI** | Full corpus schema + six tier types + sparse/dense storage split + live recording (cpal + JACK) + DSP suite (pitch / formants / intensity / spectrogram / MFCC / STFT) + TextGrid + EAF I/O + stability decorators + type stubs + reproducibility recipes + migration framework + Apache-2.0 OR MIT dual license |
| **2. Desktop GUI** | 3–4 mo | **0.2 — desktop GUI** | Egui+wgpu app shell + project navigator + waveform/spectrogram/tier-strip with sync cursor + interval/point tier editing + embedded CPython in app + `sat.app` basics (selection, register_command) + single-writer lock |
| **3. Differentiators part 1** | 3–4 mo | **0.3 — clinical-ready** | Reference distribution infrastructure (format + resolver + GitHub registry + CI + Pages index + in-app publish) + bundled starter set + GUI overlay rendering + clinical algorithms (AVQI / ABI / CPP / jitter / shimmer / HNR) with validation suite + provenance + uom typed units + calibrated SPL + research-use-only labeling + ort runtime + VAD bundled + wav2vec2/Whisper on-demand download + embedding tiers |
| **4. Articulatory** | 5–7 mo | **0.4 — articulatory** | Plugin architecture (Python + native via `abi_stable`) + EGG + Carstens AG501 importer + `video_aligned` tier + channel semantics schema fields + ffmpeg-rs decoder + video player widget + synced multi-pane layout + **tongue contour tracker** (model + per-frame inference + correction UI + validation) |
| **5. Experiment runner + scripting depth** | 2–3 mo | **0.5 — beta** | Trial sequencing (linear / randomized / block / counterbalanced) + simple sequence-editor GUI + shipped templates (CAPE-V / Rainbow / sustained-vowel / discrimination) + drill mode + PsychoPy CSV import + best-effort PsychoPy export + `register_panel` bridge (Markdown / table) |
| **6. Mobile** | 3–4 mo | **0.6 — mobile beta** | UniFFI bindings for mobile API surface + iOS shell (SwiftUI) + Android shell (Compose) — both with record + live feedback + sync + bundle pack export + store internal-testing setup |
| **7. Polish & release** | 2–3 mo | **1.0** | macOS signing + notarization + Windows installer + Linux packaging (AppImage + Flatpak) + embedded CPython distribution validation + docs site (mkdocs-material + mkdocstrings probably) + Parselmouth migration docs + notebook tutorials + bundled sample projects + bug bash + 1.0 release |

**Total: ~22–31 months solo part-time.** Compresses to ~14–18 months full-time, or shorter with contributors past 0.1.

### Parallel risk spikes

Run alongside main phases — small, fail-fast experiments:

| Spike | Concurrent with | Purpose |
|---|---|---|
| **UniFFI proof** | Phase 0 | One method bridged to Swift CLI end-to-end. Validates mobile architecture before any shell work |
| **Embedded CPython distribution** | Phase 1 | One signed macOS `.app` with embedded Python running a real script. Validates packaging story early |
| **Native plugin ABI** | Phase 3 (before Phase 4 starts) | One trivial native plugin loaded from a dylib. Settles `abi_stable` vs hand-curated C ABI vs WASM components |
| **Tongue segmentation model exploration** | Phase 3 | Survey available models (UNet variants, DeepLabCut-style, HuggingFace tongue-segmentation releases), fine-tune candidates on small dataset. De-risks the heaviest single Phase 4 deliverable |

### Critical path and dependencies

- Phase 0 unblocks everything — without the vertical slice working, no architecture can be trusted
- Engine + corpus + Python bindings (Phases 0–1) are the spine; GUI, mobile, plugins, refdist all depend on this
- Reference distribution infrastructure (Phase 3) has to land before clinical because clinical features display against norms
- Plugin architecture (Phase 4 start) has to land before the AG501 importer because the importer IS the first plugin
- Mobile (Phase 6) depends on UniFFI being proven (Phase 0 spike) and on the engine API being stable enough to bridge — tier 1 of stability decorators

### Confirmed scope decisions for v1.0

| Item | Decision | Reasoning |
|---|---|---|
| Tongue contour tracker | **Core, in Phase 4** | Held per articulatory entry. Most ambitious single v1 commitment. Phase 4's 5–7 month estimate accounts for it |
| Mobile platforms at 1.0 | **Both iOS + Android** | Per tech-stack entry. Cross-cutting pattern B (mobile as structural gap) preserved at v1.0 |

### Cut lines if timeline pressure hits

In priority order of what to defer first (most→least):

1. Tongue contour tracker → community-deliverable plugin instead of core (the articulatory entry's flagged scope concern)
2. Android → iOS-only at 1.0; Android in 1.x
3. Drill mode → defer to 1.x
4. PsychoPy script export → import-only at 1.0; export later
5. `register_panel` for arbitrary widgets → `register_command` only at 1.0
6. Some refdist starter-set entries → ship with fewer
7. emuDB export → defer to plugin / 1.x
8. EAF round-trip → import-only at 1.0; export later

**Not cuttable**: engine, corpus, Python bindings, basic DSP, basic GUI, reference distribution infrastructure, clinical algorithms + validation, EGG, packaging on three desktop platforms, the Python library on PyPI by 0.1.

### Contributor-onramp progression

- **After 0.1**: Python library contributors — DSP, ML, format support, validation tests
- **After 0.2**: GUI / UX contributors — egui-side widgets, theme work, annotation editing affordances
- **After 0.3**: broader contributors with stable APIs and lower breakage churn risk
- **After 0.4**: plugin authors can build importers / analyzers independently — biggest contributor unlock; community can fill the modality long tail (EPG, aerodynamic, additional EMA vendors)

### Cross-cutting work happening every phase

- Documentation grows incrementally per phase, not as a separate final phase
- Validation test corpus grows with each clinical algorithm in Phase 3
- Reference distribution curation ongoing (identify candidates, verify licensing)
- Community engagement ongoing pre-v1: Praat / Parselmouth / phonetics-list visibility
- Stability decorator hygiene: every public function/class consciously tiered as added

### Open questions / deferred

- ~~**Naming.**~~ Closed 2026-05-20: project named `sadda` (Pali, *sound / voice*). See 2026-05-20 entry. GitHub org qualifier and trademark check still pending.
- **Public repo structure.** Single monorepo vs split (engine / Python / GUI / mobile shells / registry). Probably monorepo for v1 era, split if scale demands. Decide before Phase 1 ends.
- **Continuous integration scope.** GitHub Actions matrix on three desktop + two mobile platforms is substantial; minimal CI covering build + test + lint suffices through Phase 1; expand from Phase 2.
- **Funding model.** Solo part-time may sustain through 0.3 or so; further depends on grants / sponsorship / commercial backing. Not architectural, but real for timeline.
- **Beta-tester recruitment timing.** Probably starts at 0.2 (desktop GUI usable); 0.3 has the clinical-research audience attached.
- **Sample data licensing.** Need a small, clean-licensed audio set for tutorials and CI tests — bundled with the docs. Sourcing TBD.
- **Translation / i18n.** Deferred entirely from v1. Strings live in source for now; structure for extraction can be added later without re-architecture.
- **Accessibility (screen readers, keyboard nav).** egui's AccessKit integration is young; v1.0 best-effort, dedicated pass in 1.x. Worth marking the limitation explicitly.

### How this plan should be revisited

- After Phase 0 ends: if any spike failed, re-architect before Phase 1.
- After every phase: assess actual time vs estimate; recalibrate later phases proportionally.
- After 0.2: real users land; their feedback may reshape Phase 3+ scoping.
- After 0.3: clinical-research adoption will surface validation requirements (or kill them as priorities).
- Annually: scope cut-line list re-evaluated against actual capacity.

This plan is the working roadmap, not a contract. Re-read at the start of each phase; adjust openly.

### Sources / references

- The other 2026-05-15 through 2026-05-18 entries (this plan is downstream of all of them)
- "Tracer Bullets" — Pragmatic Programmer (vertical-slice principle): https://pragprog.com/tips/
- SemVer (for 0.x cadence semantics): https://semver.org

---

## 2026-05-18 — Python API surface: shape, stability tiers, conventions

Goal: design the Python API surface. Flagged as an open item in the tech-stack entry — once Python is in v1, the API contracts users write scripts against become hard to change. Better to design once than retrofit.

### Two audiences, one surface

- **Library users** (Parselmouth-replacement) — script analyses against a corpus or against raw audio, no GUI. AI engineers + scripting phoneticians.
- **In-app scripting host** — automate workflows inside the desktop app; build custom analyses or panels.

Same module everywhere; one extra namespace (`sat.app`) is populated only when running inside the desktop process.

### Lessons from prior art

| Source | Lifted | Left |
|---|---|---|
| **Parselmouth** | Top-level convenience namespace; NumPy-native; escape hatches | Tied to Praat object model (we're not Praat-compat at script level) |
| **librosa** | Flat-ish functional DSP API; NumPy first; consistent shapes | Almost no state model; no corpus concept |
| **scikit-learn** | Convention consistency (fit/transform/predict everywhere) | Domain mismatch; no single repeating pattern that wide |
| **Blender bpy / Houdini hou** | In-app namespace mirrors GUI data model; build UI panels from scripts; clear library-vs-interactive distinction | Very stateful; we keep DSP/analysis stateless where possible |
| **emuR / wrassp** | Corpus query → typed result; signal-as-ndarray | R conventions don't translate |

Synthesis: **functional NumPy-native API for DSP/analysis; OO for corpus/project; one in-app namespace clearly separated.**

### Module layout

```
sat                               (top-level; `sat` is a placeholder — naming deferred)
├── __version__
├── open_project(path) → Project
├── new_project(path, profile=None) → Project
│
├── corpus                        # OO, STABLE
│   ├── Project, Session, Bundle, Tier
│   ├── Annotation (Interval/Point/Reference variants)
│   ├── DerivedSignal, Entity, EntityRef
│   ├── ModelRun, AuditLog
│   └── query(...) → polars.DataFrame
│
├── dsp                           # functional, stateless, STABLE
│   ├── pitch, formants, intensity, spectrogram
│   ├── mfcc, stft, lpc, autocorrelation, cepstrum
│   └── windowing, resampling, filtering
│
├── clinical                      # STABLE (per regulatory entry)
│   ├── avqi, cpp, cpps
│   ├── jitter, shimmer, hnr, nhr
│   └── (each returns value + uncertainty + algorithm_version)
│
├── ml                            # PROVISIONAL
│   ├── load_model, extract_embeddings
│   ├── transcribe, vad, align
│
├── refdist                       # STABLE (per refdist entry)
│   ├── get, search, publish
│   └── registry
│
├── articulatory                  # PROVISIONAL
│   ├── import_carstens, import_video
│   ├── tongue_contour
│   └── egg.{open_close, oq, cq, …}
│
├── experiments                   # PROVISIONAL
│   ├── Experiment, Trial, Block
│   ├── load_template
│   └── psychopy.import_log / psychopy.export
│
├── live                          # PROVISIONAL
│   └── start_session (with subscriber decorators)
│
├── plugins                       # PROVISIONAL
│   └── register_importer / register_analyzer / register_command
│
├── app                           # IN-APP ONLY; NotInAppError otherwise
│   ├── current_selection, active_bundle, active_project
│   ├── register_command, register_panel
│   ├── refresh()
│   └── prompts (file pickers, dialog boxes)
│
├── recipe                        # reproducibility
│   ├── record() context manager
│   ├── replay(path)
│
├── experimental                  # explicitly unstable
│
└── errors
    └── SatError → CorpusError, AnalysisError, ModelError,
                   NotInAppError, PluginError, MigrationError
```

### Data type conventions

| Concept | Type | Notes |
|---|---|---|
| Audio buffer | `np.ndarray` (float32) | `(n_samples,)` mono; `(n_channels, n_samples)` multichannel |
| Sample rate | `int` (Hz) | — |
| Time stamps | `float` (seconds) | Not frames, not Praat xmin/xmax |
| Frequencies | `float` (Hz) | — |
| Spectrograms | `np.ndarray` `(n_freq_bins, n_frames)` + metadata wrapper | Hop / window / fmin / fmax on the wrapper |
| Tier data | Tier object wrapping NumPy or Parquet | `.to_numpy()` / `.to_dataframe()` escape hatches |
| Corpus query results | `polars.DataFrame` (primary) | `.to_pandas()` escape hatch; Arrow-native; zero-copy from Rust; matches our Parquet sidecars |
| Reference distributions | `Distribution` object | `.samples` / `.summary` NumPy escape hatches |

### Stability tiers

Three tiers, marked **in code with decorators** that emit one-time runtime warnings on first use:

- **Stable** — `corpus`, `dsp` (basic measures), `clinical`, `refdist`, top-level project loaders. Commits not to break across minor versions.
- **Provisional** — `ml`, `articulatory`, `experiments`, `live`, `plugins`, `app`. May break in minor versions after a deprecation cycle.
- **Experimental** — under `sat.experimental.*`. May break anytime.

Mechanism: `@stable`, `@provisional`, `@experimental` decorators on every public function/class. First use of a non-stable API emits a `ProvisionalAPIWarning` / `ExperimentalAPIWarning` (suppressible via the standard `warnings` machinery). Surfaces dependency risk at coding time rather than after the fact.

### GIL, async, error model

- **GIL released** during all long Rust operations (already committed in tech-stack entry).
- **Sync by default** — matches scientific-Python expectations. Async wrappers for I/O-bound things (network refdist fetches, sync inbox watchers) deferred to v1.x; not in initial v1.
- **Real-time / streaming** via subscriber decorators (`@session.on_pitch`) or async iterators. Not asyncio-based in v1.
- **Errors** as a typed exception hierarchy rooted at `SatError`. Rust `Result` types map to specific exception classes via PyO3 conversion traits.

### Reproducibility

`sat.recipe.record()` context manager: every analysis call inside the block logs to the project's `ModelRun` + `AuditLog` tables, and the recipe can be saved as a Python script for replay. Connects the cross-cutting reproducibility need surfaced in corpus + refdist + clinical entries.

```python
with sat.recipe.record(proj, name="vowel_analysis_v1"):
    for bundle in proj.bundles:
        sat.dsp.formants(bundle.audio, bundle.sr, num_formants=5)
# project now has a saved recipe; reproducible via sat.recipe.replay()
```

### In-app surface

`sat.app` is populated only when running inside the desktop process. Outside, accessing it raises `NotInAppError`. Scripts that don't touch `sat.app` work identically in both modes.

In-app scripts can: read GUI state (selection, active bundle), register commands (added to a menu / palette), register custom panels (egui-side hooks via a stable bridge), trigger refreshes after mutations, call file pickers and dialog boxes.

### Representative usage

Library:

```python
import sat
proj = sat.open_project("/path/to/myproject")
for bundle in proj.bundles:
    pitch = sat.dsp.pitch(bundle.audio, bundle.sr, method="autocorrelation")
    bundle.add_tier("F0", type="continuous_numeric", data=pitch.values, hop=pitch.hop)
proj.save()
```

Clinical with reference comparison:

```python
result = sat.clinical.avqi(audio, sr)
print(result.value, result.uncertainty, result.algorithm_version)
norm = sat.refdist.get("voiceevalu8-avqi-norms", version="2.1.0")
percentile = norm.percentile_rank(result.value, population={"sex": "f", "age_band": "adult"})
```

Live (voice-coach):

```python
session = sat.live.start_session(device="default", sr=44100)
target = sat.refdist.get("trans-feminine-resonance-targets")

@session.on_formants
def show(formants, time):
    print(f"F1={formants[0]:.0f} F2={formants[1]:.0f} in_target={target.contains(formants)}")

session.start_recording(target_bundle=proj.new_bundle("practice"))
```

In-app:

```python
import sat, sat.app as app

@app.register_command("Compute custom feature")
def my_analysis():
    bundle = app.active_bundle
    audio = bundle.audio_slice(app.current_selection)
    bundle.add_tier("custom_feature", type="continuous_numeric",
                    data=my_dsp_function(audio))
    app.refresh()
```

### v1 deliverables this entry commits to

1. Module layout as enumerated above.
2. Data type conventions: NumPy primary for signals, Polars primary for corpus query results, with documented escape hatches.
3. Three-tier stability marking via in-code `@stable` / `@provisional` / `@experimental` decorators that emit runtime warnings.
4. GIL released during long Rust ops; sync-by-default API.
5. Typed exception hierarchy rooted at `SatError`.
6. `sat.recipe.record()` reproducibility context manager.
7. `sat.app` namespace populated only inside the desktop process.
8. Type stubs (`.pyi`) generated alongside the module; `py.typed` marker present.
9. Rich `__repr__` (including HTML for Jupyter) on major types.
10. Migration documentation for Parselmouth users (no compat shim — migration docs only).

### Open trade-offs / deferred

- **Top-level module name** — `sat` is a placeholder (collides with the SAT-solver namespace and is overly generic). Naming is its own discussion; defer until closer to release.
- **Praat-script compatibility** — explicitly out of v1 per 2026-05-16 question 5. Worth revisiting after v1 if migration friction proves high.
- **Async / asyncio support** — sync-first in v1; async wrappers for I/O-bound APIs (network fetches, sync inboxes) layered later.
- **Tab-completion / IDE polish** — type stubs + `py.typed` are the v1 floor; richer IDE integration (Pylance plugins, VS Code extensions) is a v1.x consideration.
- **In-app panel API** — `app.register_panel` exists but the panel-rendering bridge between Python callbacks and egui's frame loop needs a focused design. The simple case (register a Markdown / table view) is straightforward; arbitrary widgets are harder.
- **Plugin authoring SDK vs the regular API** — current decision: plugins use the same `sat` API + `sat.plugins.register_*` registration. Whether plugins need a richer SDK (lifecycle hooks, capabilities declaration beyond what `register_*` carries) is a follow-up.
- **Migration helpers for non-Python tools** — a small `sat.io.praat` module for TextGrid round-trip (covered in corpus entry) and basic Sound-object compatibility (`sat.io.praat.read_sound`) likely worth shipping; full Parselmouth-shaped object wrappers are not.
- **Notebook tutorials and documentation infrastructure** — mkdocs-material + mkdocstrings, or sphinx-immaterial? Pick before docs work starts in earnest.
- **Sample-data distribution for tutorials** — small bundled audio files for docs to work without external downloads; pick a license-clean source.

### Sources / references

- PyO3: https://pyo3.rs
- pyo3-stub-gen (type stub generation from PyO3): https://github.com/Jij-Inc/pyo3-stub-gen
- Parselmouth: https://parselmouth.readthedocs.io
- librosa API design: https://librosa.org/doc
- scikit-learn API design glossary: https://scikit-learn.org/stable/glossary.html
- Blender bpy: https://docs.blender.org/api/current/
- Houdini hou: https://www.sidefx.com/docs/houdini/hom/
- Polars: https://pola.rs
- Apache Arrow: https://arrow.apache.org
- PEP 484 (type hints): https://peps.python.org/pep-0484/
- py.typed marker (PEP 561): https://peps.python.org/pep-0561/

---

## 2026-05-18 — Experiment runner: production/protocol/drill focus, desktop-only in v1

Goal: settle scope for the experiment runner — the last open question from 2026-05-16. The decision is not *whether* (cross-cutting pattern I from 2026-05-16 already said yes), but *how deep*: PsychoPy-equivalent depth or Demo Window+ minimum?

### What our user groups actually need from an experiment runner

| Group | Use case | Demand profile |
|---|---|---|
| 8. Psycholinguists | Perception experiments | High — full PsychoPy territory (frame-accurate visual timing, RDKs, eye-tracker integration, web deployment) |
| 1, 6. Phoneticians, field linguists | Production / elicitation (prompt → record) | Low–moderate — text/audio/image prompt + record + tag with trial metadata |
| 3. Clinicians | Protocol-driven workflows (CAPE-V, Rainbow Passage, sustained vowels at fixed F0) | Low — sequential templates with auto-tag, calibrated recording |
| 4, 5. Voice coaches, L2 learners | Drill structure (model → attempt → feedback) | Low–moderate — repetition, comparison feedback, progress tracking |

**Five of six use cases are variants of "present a prompt, record the response, tag it with trial metadata."** Only group 8 needs the deep perception-research surface PsychoPy excels at.

### Defensible position vs PsychoPy

What we should build (unique to us):
- Trial recordings ARE corpus bundles, immediately analyzable in the same project (cross-cutting D — convergence of the record-analyze pipeline)
- Real-time analysis / feedback during practice (voice-coach / L2)
- Calibrated SPL during clinical protocols (cross-references the clinical regulatory entry)
- Articulatory data captured *during* experiments (EMA / ultrasound, via the articulatory entry)
- Reference distributions overlaid on live response visualization

What we should not try to match (PsychoPy does well):
- Frame-accurate visual timing
- Vision-science stimuli (RDKs, gratings, faces, movies)
- Eye-tracker integration
- Specialized hardware (button boxes, parallel/serial port triggers)
- Web deployment via Pavlovia
- Mature visual Builder UI for arbitrary trial structures

**The v1 commitment: a production-and-protocol-oriented experiment runner that converges the record-analyze pipeline. Leave perception-research depth to PsychoPy. Interop both directions.**

### v1 capabilities

**Stimulus types**
- Text (any length, formatted)
- Image (static)
- Audio (sample-accurate scheduling via `cpal`)
- Multi-modal combinations (text + audio, image + audio, …)

**Response types**
- Keyboard / mouse (egui native)
- Audio recording of the participant (real-time path already committed)
- Categorical selection (button-press equivalent in GUI)
- Timed vs untimed responses

**Trial sequencing**
- Trial = (stimulus spec, response spec, optional time limit, optional metadata)
- Sequences: linear, randomized, block-structured
- Basic counterbalancing (Latin square); complex factorial designs via the Python API
- Per-trial logging into corpus

**Protocol templates**
- Shipped: CAPE-V, Rainbow Passage, sustained-vowel at fixed pitches, basic discrimination tasks
- User-extensible: templates as data, editable in the GUI

**Drill mode (voice-coach / L2)**
- Model presentation → user attempt → comparison feedback
- Spaced repetition logic
- Progress tracking across sessions

**Builder UI**
- Simple sequence editor: drag-and-drop linear sequence of trials; per-trial config inspector; block structure via grouping
- Complex trial structures via the Python API rather than a PsychoPy-Builder-equivalent visual editor

**Interop**
- Import: PsychoPy CSV trial logs → ingested as metadata for matching recordings in the corpus
- Export: experiment definition → basic PsychoPy script scaffold (best-effort, no fidelity guarantees)

**Explicitly out of v1 scope**
- Vision-science stimulus types (shapes, gratings, RDKs, faces, movies)
- Specialized hardware integration (eye-trackers, button boxes, serial / parallel triggers)
- Frame-accurate visual timing guarantees (best-effort frame-rate timing only)
- Web deployment of experiments — see below
- Full PsychoPy-Builder-equivalent visual editor

### Browser deployment — deferred to v2+

The 2026-05-16 architectural implication #7 left browser as a deferred third surface. Confirming that stance for experiments specifically: **defer browser deployment to v2+.**

Rationale:
- Crowdsourced experimenters (Prolific / MTurk audiences) already use jsPsych, PCIbex, Gorilla, Pavlovia. We don't need to compete on browser-deployed experiments to be useful.
- A browser surface is meaningful additional build: TypeScript runtime, hosting infrastructure, participant-flow UX (consent, instructions, completion), cross-browser audio quality testing, privacy considerations under HIPAA / GDPR for participant audio in transit.
- It also complicates the local-first principle (#10) — first hosted-component dependency in the stack.
- v1 interop strategy already covers this audience: their existing browser-tool trial logs land in our corpus as metadata for downstream analysis.

When browser deployment becomes a v2+ priority, the architectural shape will likely mirror the mobile companion: a thin TypeScript client that records + uploads to a sync endpoint, with all analysis still happening on the experimenter's desktop project. No Rust-engine-in-WASM required.

### Architectural integration with prior entries

This entry barely costs anything because the existing data model accommodates it:

1. **`Trial` is a generic `Entity`** (`kind = "trial"`) with `extra` JSON for stimulus / response specifications. Already in the corpus entry.
2. **Experiment recordings are normal `Bundle`s** linked to trials via the `reference` tier type or `EntityRef`. Trial metadata flows directly into downstream analyses.
3. **Stimulus presentation runs in the GUI** (egui), using the same wgpu painter / audio scheduler used elsewhere.
4. **Sample-accurate audio scheduling** uses `cpal`'s callback-based output — already in the tech-stack commitments.
5. **Per-trial recording** uses the same atomic `.in_progress/` → commit path from the corpus entry.
6. **Real-time feedback during drills** (voice-coach use case) uses the live-analysis path already committed.
7. **Clinical protocol templates** integrate with calibrated SPL from the regulatory entry.
8. **Articulatory data during experiments** uses the same import path as standalone articulatory recording.

No new infrastructure required — just a stimulus-scheduling layer and a trial-sequencing engine on top of what we're already building.

### v1 deliverables this entry commits to

1. Stimulus types: text, image, audio (single + multi-modal).
2. Response types: keyboard, mouse, audio recording, categorical selection (timed / untimed).
3. Trial-sequencing engine: linear, randomized, block-structured, with basic counterbalancing.
4. Simple sequence-editor GUI (drag-and-drop, per-trial inspector, block grouping).
5. Shipped protocol templates: CAPE-V, Rainbow Passage, sustained-vowel-at-pitch, basic discrimination.
6. User-editable template format.
7. Drill mode with comparison feedback and spaced repetition for voice-coach / L2 use.
8. Per-trial logging into the corpus via `EntityRef` to a `Trial` entity.
9. PsychoPy CSV trial-log import → corpus metadata.
10. Best-effort PsychoPy script export from experiment definitions.

### Open trade-offs / deferred

- **Browser-deployable experiments → v2+.** Re-evaluate when the v1 mobile-companion architecture has stabilized; the TypeScript thin-client pattern likely transfers.
- **Specialized hardware integration (eye-trackers, button boxes, serial / parallel triggers)** → plugins (uses the plugin architecture committed in the articulatory entry). Native plugins ideal for low-latency hardware paths.
- **Advanced experimental designs (complex factorials, adaptive procedures, staircases)** → via Python API in v1; visual builders for these are heavy and PsychoPy already does them well.
- **Vision-science stimuli (RDKs, gratings, faces, movies)** → defer indefinitely; not in any of our scoped audiences.
- **Frame-accurate visual timing guarantees** → defer; would require lower-level GPU compositor integration than egui currently provides.
- **Participant identity / privacy in trial data** → straightforward (uses the corpus's `Speaker` entity with appropriate `extra` schema), but worth its own pass for clinical-experiment use cases under the regulatory entry's posture 3.
- **Drill mode UX details** → spaced-repetition algorithm choice, comparison-feedback visualization style, progress-tracking metric design. Iterate with voice-coach / L2 collaborators.
- **Protocol template authorship** → who curates the shipped templates (CAPE-V is well-defined; voice-coach drills less so). May fold into the reference-distribution registry pattern or warrant its own template registry.

### Sources / references

- PsychoPy: https://www.psychopy.org
- Pavlovia (browser deployment for PsychoPy): https://pavlovia.org
- jsPsych: https://www.jspsych.org
- PCIbex: https://www.pcibex.net
- Gorilla: https://gorilla.sc
- LabVanced: https://www.labvanced.com
- CAPE-V (Consensus Auditory-Perceptual Evaluation of Voice): https://www.asha.org/practice-portal/clinical-topics/voice-disorders/

---

## 2026-05-18 — Clinical regulatory stance: research-use-only, fork-friendly clinical readiness

Goal: settle the regulatory posture — pursue FDA / CE clearance pathways ourselves, or stay research-only and let downstream forks pursue clearance. Carried open from 2026-05-16.

### Five postures considered

| Posture | What it means | Cost | Fit |
|---|---|---|---|
| 1. Pure research use | Explicit "not for clinical use" labeling; clinical features may exist but no clinical claims; Praat's de facto posture | Minimal | Easy default |
| 2. Research-only, no clinical-design intentionality | Same as (1), no special effort to be clinically defensible | Minimal | Cheapest |
| 3. Research-only, fork-friendly clinical readiness | Designed so a downstream commercial entity could fork and pursue clearance without re-architecting | Modest engineering hygiene | Recommendation |
| 4. Self-pursued clearance | Formal 510(k) / CE under MDR, QMS (ISO 13485 / IEC 62304 / IEC 14971), V&V, post-market surveillance | $200k–$2M+ initial, ongoing labor | Not viable for our shape |
| 5. Cleared and shipped | Actually a medical device | Even more | Not viable |

Postures 4 and 5 require a corporate vehicle, regulatory consultancy, and ongoing burden that competes with development. They are not viable for a research-driven open-source project of our shape. Almost every academic phonetics / speech-science tool sits at posture 1 or 2; CSL / MDVP / VoxMetria / Visi-Pitch carry the regulatory burden separately as proprietary commercial products.

### Decision: posture 3

The clinical user group (group 3 from 2026-05-16) needs AVQI / ABI / CPP / jitter / shimmer / HNR, CAPE-V protocols, calibrated SPL, EGG sync, phonetograms, longitudinal patient viz, report generation, HL7 / FHIR export. **None of these require clearance to implement.** They can ship as research-use-only features. The regulatory posture is about *claims and design intentionality*, not about whether features exist.

Posture 3 ships clinical features as research-use-only, builds the engine with the discipline a downstream forking entity would need to pursue clearance, and avoids the regulatory burden ourselves.

### Architectural commitments under posture 3

Mostly good engineering hygiene anyway, with extra rigor for clinical-path code:

1. **Algorithm provenance metadata.** Every measure records which algorithm version was used, with citation to published reference. Plumbed through the ML model registry from the corpus entry.
2. **Validation test suite shipped with engine.** For clinical-style measures, ship known-input known-output tests comparing against published reference implementations and reference values. A forking entity can run them as the seed of a V&V package.
3. **Calibration as first-class.** Calibrated SPL, mic profiles, sensitivity / frequency response curves attached to the `Instrument` entity already in the corpus model. Without calibration, SPL-based measures are nonsense in clinical contexts.
4. **Units pervasive, type-level where possible.** Crates like `uom` give compile-time unit checking; modest overhead, high payoff for clinical paths. Never bare numbers without units.
5. **No silent fallbacks on clinical measures.** If a measure can't be computed reliably (insufficient signal, bad input), return an error or explicit uncertainty — not a guess. False confidence is harmful in clinical contexts.
6. **Regression testing for clinical algorithms in CI.** Numerical drift detected before release; surprising changes are gated on explicit acknowledgement.
7. **Stable-API marking.** Certain APIs designated as "stable for clinical use," with stronger change-control discipline than the rest of the engine. Cross-references the API surface concern from the tech-stack entry.
8. **Audit trail at the analysis level.** Already committed in the corpus entry (`AuditLog`). Every analysis run records inputs, algorithm versions, parameters, outputs. Forensic and clinical share this need.
9. **Documentation of intended use.** Explicit Intended Use statement even under research positioning: what we claim, what we don't, scope of validation, known limitations.
10. **Research-use-only labeling.** At app startup, in documentation, on clinical-style features: "For research, education, and non-diagnostic use only."

### v1 clinical scope

| Feature | v1 | v1.x | Defer |
|---|---|---|---|
| Clinical-style algorithms (AVQI, ABI, CPP, jitter, shimmer, HNR) with validation suite | ✓ | | |
| Calibrated SPL + calibrated mic profiles in `Instrument` | ✓ | | |
| EGG sync + basic EGG analysis (OQ, CQ) | ✓ (via articulatory entry) | | |
| Patient-as-`Entity` with `kind = "patient"`, longitudinal viz hooks | ✓ (via corpus entry) | | |
| Sustained-vowel + connected-speech protocol workflows | | ✓ | |
| CAPE-V / Rainbow Passage protocol templates | | ✓ | |
| Phonetogram / Voice Range Profile renderer | | ✓ | |
| FHIR / HL7 export | | ✓ | |
| Clinical report templates | | ✓ | |
| EHR integration | | | Plugin or v2+ |

Rationale for the v1 cut: ship the analysis engine and calibration in v1 because they're foundational and have the highest validation-design cost. Workflow features (protocols, report templates, FHIR export) wait until we've talked to enough clinical researchers to know what shapes are actually used — these are easy to get wrong without that input, and the cost of getting them wrong is real (workflows are sticky).

### The license tension — noted, not resolved

Posture 3 only works if our license permits commercial forks: permissive (Apache-2.0 OR MIT) or weak copyleft (MPL-2.0). Under GPL-3.0+, the fork-and-clear pathway is effectively closed — clinical vendors don't fork GPL code because they don't want to release their derivative work under GPL, and FDA-cleared products are almost never GPL.

So **posture 3 + GPL is internally inconsistent.** If we land on GPL later, clinical productization by third parties is foreclosed and posture 3 effectively collapses to posture 1.

**The license decision is not forced by this entry.** It remains deferred per the licensing entry's stance. This tension is now documented and feeds into the eventual license call. Concretely: if the GPL ambition is preserved through to release, we should re-read this entry and decide whether to (a) accept the collapse to posture 1 or (b) shift the license away from GPL to preserve posture 3.

### What posture 3 explicitly does NOT do

To be precise about the negative space:

- Does NOT establish a Quality Management System (ISO 13485)
- Does NOT formally document software lifecycle processes (IEC 62304)
- Does NOT do formal risk management (ISO 14971)
- Does NOT make therapeutic, diagnostic, or treatment claims
- Does NOT take liability for clinical decisions made using the tool
- Does NOT prevent clinical research use — "research use only" by long convention includes clinical-research contexts

### v1 deliverables this entry commits to

1. Research-use-only labeling at app startup, in docs, on clinical features. Explicit Intended Use statement.
2. Validation test suite for AVQI, ABI, CPP, jitter, shimmer, HNR comparing against published reference values, runnable from the engine and from CI.
3. Algorithm provenance metadata on every measure, with citation references.
4. Calibrated SPL + mic profile support on the `Instrument` entity.
5. `uom`-style typed units on the clinical-path API.
6. No-silent-fallback discipline on clinical measures (explicit error / uncertainty types).
7. Regression testing for clinical algorithms in CI with numerical-drift gates.
8. Stable-API marking convention for clinical surface.
9. Audit-trail coverage of clinical analyses (already committed via corpus entry; confirmed for clinical scope).

### Open trade-offs / deferred

- **Clinical workflow features (v1.x).** Protocol templates (CAPE-V, Rainbow Passage), phonetogram renderer, FHIR / HL7 export, report templates. Defer until we've validated workflow shape with clinical-research users. Risk: clinical adoption may be limited in v1 without these.
- **Algorithm validation reference sources.** Each clinical measure needs a chosen reference implementation to validate against: AVQI against Maryn et al. (publishes Praat scripts); CPPS against Hillenbrand / Heman-Ackah; jitter / shimmer against Praat and MDVP. Need a focused entry on which references we adopt and how we resolve disagreement between them (Praat and MDVP do not always agree on the same algorithm).
- **Per-measure uncertainty quantification.** Returning "value + uncertainty" rather than "value" is the right shape for clinical use, but requires per-measure work to define what uncertainty means. v1 starts with explicit error returns on unreliable inputs; richer uncertainty modeling layered in later.
- **Recruiting clinical-research collaborators.** Posture 3's value depends partly on whether clinical researchers actually use the tool. Outreach / partnership thinking is a non-engineering question worth its own entry once v1 is closer.
- **Position on supporting a future commercial fork pursuing clearance.** Are we willing to provide upstream stability commitments, validation-data access, or co-signed test reports for a serious forking entity? Affects how stable-API marking is enforced and how much support burden we accept.
- **Plugin-clinical interaction.** Clinical-path code should probably be restricted to in-tree or audited plugins — a community plugin computing jitter incorrectly is a different risk profile than a community plugin computing an experimental feature. Worth a policy.
- **Documentation of "known limitations" per measure.** Posture 3 expects honest documentation of what each measure does and doesn't validate. Mechanical work but real.

### Sources / references

- FDA Software as a Medical Device (SaMD): https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd
- IMDRF SaMD documents: https://www.imdrf.org/working-groups/software-medical-device-samd
- EU MDR (2017/745): https://eur-lex.europa.eu/eli/reg/2017/745/oj
- FDA 510(k): https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/premarket-notification-510k
- IEC 62304 (medical device software lifecycle) and ISO 13485 (QMS for medical devices) and ISO 14971 (risk management): ISO/IEC standards, not free
- AVQI (Maryn et al.) reference Praat scripts: https://www.vvl.be/en/avqi
- CPPS background (Hillenbrand): https://homepages.wmich.edu/~hillenbr/
- `uom` (units of measurement for Rust): https://github.com/iliekturtles/uom

---

## 2026-05-18 — Articulatory channels: first-class data model, plugin-driven vendor glue

Goal: settle whether ultrasound / EMA / EPG / EGG / aerodynamic / video modalities are first-class signals, plugins, or out of scope. The question was carried open from 2026-05-16.

### The modalities and their engineering shape

Stepping back from what each one *measures* and looking at what each one *demands*:

| Modality | Data shape | Sample rate | Vendor lock | Sync need |
|---|---|---|---|---|
| **EGG** | Single channel, audio-rate | 16–96 kHz | Mild | Trivial — usually a 2nd WAV channel |
| **EMA** | 3D positions × N sensors (5–20) | 200–400 Hz | Heavy (Carstens AG501 dominant; NDI Wave) | Sub-ms; vendor format carries it |
| **EPG (palatography)** | ~62–96 binary contact sensors | 100–200 Hz | Heavy (Reading EPG, WinEPG) | Ms-level |
| **Ultrasound** | Video frames (60–200 fps), often with audio | 60–200 fps | Heavy (AAA / Articulate Instruments dominant) | Sub-frame |
| **rtMRI** | Video, very high data rate | ~83 fps | Research-only (USC SPAN ecosystem) | Sub-frame |
| **Aerodynamic** | Multi-channel sensor (oral/nasal flow, pressure) | 1–10 kHz | Heavy (PERCI-SARS, GlottalEnterprises) | Ms-level |
| **High-speed laryngeal video** | Video, very high frame rate (~4000 fps) | 4000 fps | Heavy (Wolf, Pentax) | Sub-frame |

Two structural facts drive the architecture:

- **Hardware-side is fragmented and vendor-locked.** We will not drive the hardware. We import data after recording.
- **Format fragmentation is worse than hardware.** Multiple Carstens formats across model generations; AAA proprietary; EPG vendor-specific; ultrasound sometimes DICOM, sometimes proprietary; rtMRI is research-pipeline-specific.

Current state of the art: **no integrated tool exists.** AAA + MView (UCLA MATLAB) + bespoke MATLAB pipelines is the academic stack. EMU-SDMS gets closest (SSFF multichannel + hierarchical levels) but its articulatory story is thin and R-centric. **This is a genuine differentiator for the phonetician audience if we build it well.**

### Architectural answer: data-model native, vendor glue in plugins

The decision is two-layered, not a single signals-vs-plugins fork:

1. **The data model treats articulatory channels as first-class signals.** They're time-aligned (uni- or multi-) dimensional data anchored to a `Bundle`. The existing corpus tier model already covers sensor modalities via `continuous_numeric` (e.g., nasal airflow), `continuous_vector` (EMA, EPG), and `categorical_sampled` (EPG contacts as binary). Video is the only modality the existing tier model doesn't cover — needs a new tier type.
2. **Vendor-format glue lives in plugins.** Each importer (Carstens `.pos` → vector tier; AAA `.ust` → video + contour tier; EPG vendor format → categorical-sampled tier) is a plugin, not core. The engine ships a plugin API and a small handful of in-tree importers; the long tail lives out-of-tree.

This honors the 2026-05-16 architectural implication #3 ("signal types are pluggable") without bloating the engine for every vendor format.

### Changes to the corpus data model

**New tier type: `video_aligned`.** Storage: original video file in `signals/original/session_X/bundle_Y.US.mp4` (or `.dicom`, `.avi`); the DB tier row carries the file reference + per-frame timing offset + frame-rate / codec metadata. Multiple `video_aligned` tiers per bundle allowed (e.g., ultrasound + lip-camera). Adds a seventh tier type to the six defined in the corpus entry.

**Channel semantics in `continuous_vector` `schema`.** A 21-channel EMA tier needs to declare what each channel means (`TT_x`, `TT_y`, `TT_z`, `TD_x`, …). Adding `channel_names` and `channel_kinds` fields to the `schema` JSON covers EMA, EPG, and aerodynamic without further tier types.

**Sub-millisecond sync via `time_offset_seconds` in tier `schema`.** Articulatory tiers need precise alignment with audio; the offset comes from the vendor format at import. Small addition, easy to overlook.

**Derived per-frame measurements stay standard tiers.** Tongue contour polylines, region masks, tracked landmarks — all `continuous_vector` tiers anchored to the same frame timeline as the parent `video_aligned` tier. Parent-child cardinality (already in the corpus model) handles the relationship.

### Plugin architecture — both Python and native, in v1

Two plugin classes, both available from launch:

**Python plugins.** Loaded via the embedded CPython committed in the tech-stack entry. Discovery: scan plugin directories for packages with a plugin manifest. Use case: vendor-format importers (where work is parsing weird binary formats, not crunching numbers). Easier authoring; aligned with the Python audience we're already serving.

**Native plugins (Rust dylib).** Loaded via dlopen / LoadLibrary at startup. Need a stable ABI — likely via the `abi_stable` crate or carefully-curated C-ABI surface. Use case: performance-critical analyzers (per-frame video analysis, real-time contour inference, anything that needs to run in the audio thread). Higher development cost; higher capability ceiling.

Plugin discovery locations:
- Project-scoped: `<project>/plugins/`
- User-scoped: `~/.config/sat/plugins/`
- Distribution: Python plugins pip-installable into embedded Python OR drop-folder; native plugins ship pre-built per platform.

Plugin trait sketch (conceptual; native and Python both implement equivalent contracts):

```rust
trait Importer {
    fn name() -> &'static str;                  // "carstens-ag501"
    fn supported_extensions() -> &[&str];       // ["pos", "amps"]
    fn detect(file: &Path) -> Confidence;
    fn import(file: &Path, bundle: &mut Bundle) -> Result<()>;
}

trait Analyzer {
    fn name() -> &'static str;                  // "tongue-contour-tracker"
    fn inputs() -> &[InputSpec];                // requires video_aligned tier
    fn outputs() -> &[OutputSpec];              // produces continuous_vector tier
    fn run(bundle: &Bundle, params: &Params) -> Result<()>;
}
```

Plugins declare capabilities in a manifest; the engine indexes them at startup. No sandboxing in v1 (trust model: user installs what they trust); plugin signing / sandboxing is a follow-up if it becomes necessary.

### v1 scope — maximal

| Modality | v1 status | Notes |
|---|---|---|
| **EGG** | **In v1, core.** | 2nd-WAV-channel import; basic analysis (open/closing instants, OQ, CQ from differentiated EGG). High clinical value (group 3); trivial engineering |
| **EMA — Carstens AG501** | **In v1, importer plugin in-tree (Python).** | Parses `.pos` to `continuous_vector` tier with channel semantics. Trajectory rendering in GUI |
| **Ultrasound — generic video** | **In v1, core.** | `video_aligned` tier type; video player widget in GUI; ffmpeg-rs or rsmpeg decoder |
| **Tongue contour tracking** | **In v1, core analyzer (native plugin).** | Bundled (or download-on-first-use) ML segmentation model; per-frame inference; interactive correction UI; contours stored as `continuous_vector` tier under the parent `video_aligned` tier. **This is the most ambitious single v1 commitment scoped so far** — see open trade-offs |
| **Ultrasound — AAA format** | v1.x via plugin | Format reverse-engineering; community plugin candidate |
| **EPG** | v1.x via plugin | Data model handles; waiting on importer |
| **rtMRI** | Defer | USC SPAN ecosystem handles; small overlap with primary audiences |
| **Aerodynamic** | v1.x via plugin | Data model handles trivially via `continuous_numeric`; needs vendor-specific importers |
| **High-speed laryngeal video** | Defer | Clinical/research niche; very large data |

### GUI implications

- **Video player widget synced with audio + spectrogram + tier overlays.** wgpu renders textures from decoded video frames; ffmpeg-rs / rsmpeg is the decoder. Cursor / playhead synchronization across waveform / spectrogram / video / EMA trajectory plot is the central UX claim.
- **Multi-pane synchronized layout.** Audio waveform + spectrogram + EMA trajectory + ultrasound video + tier strip, all scrolling together. egui's painter API handles this; just confirming articulatory data doesn't break the assumption.
- **EMA-specific rendering.** Mid-sagittal 2D projection (X-Y / X-Z), 3D rendering for AG501. Plottable as derived tiers in the same display surface.
- **Contour correction UI.** Per-frame contour editing with ML-model re-fit, propagation of corrections across nearby frames, comparison of model vs corrected. Its own UX subproblem.

### v1 deliverables this entry commits to

1. Seventh tier type `video_aligned` added to the corpus model.
2. Channel semantics (`channel_names`, `channel_kinds`) and `time_offset_seconds` fields added to dense-tier `schema`.
3. Plugin architecture: Python and native, in-tree and out-of-tree, with discovery in project + user dirs.
4. EGG support in core.
5. Carstens AG501 EMA importer (in-tree Python plugin).
6. Ultrasound video display (core): video player widget, sync with audio + spectrogram + tier strip.
7. Tongue contour tracker (core native analyzer): bundled/downloadable ML model, per-frame inference, interactive correction UI, results stored as a tier.

### Open trade-offs / deferred

- **Tongue contour tracker scope within v1.** This is the heaviest single commitment we've made. Realistically it's its own several-month workstream: model choice (UNet variant? DeepLabCut-style? fine-tune an existing model?), training / validation strategy, per-frame inference perf tuning, interactive correction UX, evaluation against existing tools (AAA, GetContours, Ultratrace). Worth phasing within v1: get bundled model + per-frame inference + visualization shipping first, build correction UI iteratively after. Could merit its own follow-up DEVLOG entry once a model is chosen.
- **Bundled model licensing and weights.** Per the licensing entry's discipline, model weights are downloaded not bundled. Need to pick a model with a permissive license (or train our own and publish it). Open-licensed pretrained tongue-segmentation models exist (various academic releases on HuggingFace and OSF) but quality varies.
- **Native plugin ABI stability.** `abi_stable` crate vs hand-curated C ABI vs WASM components. Each has trade-offs; needs a focused decision before v1 plugin API is published, since once it ships it's hard to change.
- **Video decoder dependency.** ffmpeg via `ffmpeg-rs` / `rsmpeg` is the standard answer but adds a heavy native dep with its own licensing constraints (LGPL builds OK for any license; GPL components must be excluded from builds). Worth a focused look.
- **Sub-frame sync semantics.** `time_offset_seconds` covers per-tier offset, but ultrasound + audio sometimes has drift across a long recording (clock-domain difference between video and audio). v1 assumes constant offset; clock-drift correction deferred.
- **3D EMA visualization.** AG501 is 3D; AG500 and older were 2D. v1 starts with 2D mid-sagittal rendering and full 3D becomes a follow-up.
- ~~**Bundled model distribution channel.**~~ Closed 2026-05-20: parallel registry to refdist (shared protocol, separate index), HF passthrough as escape hatch. See 2026-05-20 ML-model-registry entry.

### Sources / references

- EMU-SDMS articulatory support: https://ips-lmu.github.io/EMU.html
- AAA (Articulate Assistant Advanced): https://www.articulateinstruments.com/aaa
- MView (UCLA): https://www.phonetics.ucla.edu/facilities/software/mview.html
- Carstens AG501: https://www.articulograph.de
- NDI Wave: https://www.ndigital.com/products/wave
- USC SPAN rtMRI corpus: https://sail.usc.edu/span
- Ultratrace: https://github.com/Ultratrace/ultratrace
- GetContours: https://github.com/mlml/getContours
- ffmpeg-rs: https://github.com/zmwangx/rust-ffmpeg ; rsmpeg: https://github.com/larksuite/rsmpeg
- abi_stable crate: https://github.com/rodrimati1992/abi_stable_crates

---

## 2026-05-18 — Reference distribution governance: three-tier registry, unified format

Goal: settle the governance, format, and architecture for reference distributions — the open question carried from 2026-05-16 (cross-cutting pattern C) and flagged again in the corpus entry as a deferred follow-up. Reference distributions are the transverse value-add across at least five user groups (voice coaches, L2 learners, clinicians, forensic, phoneticians), and nobody currently has them in one place.

### What a reference distribution IS, in our model

A structured statistical summary of some acoustic / articulatory measurement over a population (or a prescriptive target), in a form the GUI can render against:

- Vowel formant clouds (Hillenbrand 1995, Peterson-Barney 1952, Hagiwara, …)
- Age / sex-normed clinical ranges (jitter, shimmer, HNR, CPP, …)
- F0 statistics by age / sex / dialect (forensic + voice-coaching)
- Speaker-set distributions (forensic likelihood ratios)
- Target zones (voice-coach / L2 — prescriptive, not observed)
- Articulation-rate norms, VOT distributions, prosodic patterns by language

The unifying abstraction: **a tagged sample or summary you can compare a measurement against**.

### Governance decomposes into seven sub-questions

| Sub-question | What it means |
|---|---|
| Curation | Who decides what's authoritative vs experimental |
| Versioning | How users pin for reproducibility (esp. forensic / clinical) |
| Licensing | What licenses are acceptable; can derived distributions ship under different terms than their source corpora |
| Privacy / k-anonymity | When does a distribution leak speaker identity |
| Discovery | How a user finds "the right reference for adult AmE female /i/" |
| Citation | Automating BibTeX / DOI so use in papers is one-click |
| Disputes / errors | Yanking, conflicting evidence, quality flags |

Plus the cross-cutting question of **transport** (how distributions get from publisher to user).

### Prior art surveyed

| Source | Lifted from | Left behind |
|---|---|---|
| **CRAN** | Strict editorial review, immutable versions, citeable | Slow contribution rate; heavy curation labor |
| **HuggingFace Hub** | Anyone publishes; reputation-based discovery; rich metadata cards; tag faceting | No quality gating; trust signals coarse |
| **Zenodo / OSF** | DOIs, persistent versioning, free researcher hosting | Not domain-aware; no schema for our data shape |
| **MFA models registry** | Domain-specific catalog, version-tagged, language-faceted — closest precedent in speech | Narrow scope (just alignment models) |
| **NIST reference data** | Authoritative, slow-moving, foundational tier | Government-scale infrastructure we don't have |
| **NHANES** | Clinical reference data convention; population descriptors | US-only, not directly extensible |
| **OLAC / PARADISEC / ELAR** | Metadata standards (Dublin Core, IMDI, CMDI) for language data | Heavy bureaucratic submission |

The pattern that fits us best is a **CRAN-curated tier on top of a HuggingFace-style open tier**, with MFA's models registry as the closest domain-specific precedent.

### Three-tier model (all three shipping at launch)

**Tier 1 — Bundled starter set.** Ships with the app. Small, vetted, foundational. Classic public-data distributions: Hillenbrand 1995 vowel space, Peterson-Barney 1952, a small set of clinically published normative ranges. Citation metadata baked in. Only redistributable-licensed data here. Updated with app releases.

**Tier 2 — Curated registry.** Editorial standards (CRAN-style): must have provenance, license, min-n, reproducible derivation method. Versioned semantically. Hosted on GitHub Pages. Submissions are PRs to a public repo; CI runs validation; merge publishes. Editorial committee starts as project maintainers, expanded by adding repo maintainers over time.

**Tier 3 — Community registry.** Same infrastructure, lower bar. Anyone can publish. Trust signals (download count, tier 2 promotion, flag-as-questionable) drive discovery rather than gatekeeping. Tier 2 can promote a tier 3 entry after review.

All three tiers share **one format, one query API, one discovery surface**. Users see them together with provenance labels in the GUI; only the trust signal differs.

### Format

A distribution is a directory or tarball:

```
refdist.toml         -- manifest: id, version, citation, population, measure, schema, license, kind
data.parquet         -- the data (samples OR summaries, per shareability declaration)
provenance.md        -- human-readable: paper, method, sample procedure
LICENSE              -- explicit license file
```

The `refdist.toml` schema is load-bearing — discovery, validation, citation, and rendering all key off it:

```toml
id = "hillenbrand-1995-amE-vowels"
version = "1.0.0"
title = "American English vowel formants (Hillenbrand et al. 1995)"
doi = "10.1121/1.411872"

[citation]
authors = ["Hillenbrand, J.", "Getty, L.", "Clark, M.", "Wheeler, K."]
year = 1995
journal = "JASA"
bibtex = "..."

[population]
language = "eng"
variety = "AmE"
sex = ["m", "f", "c"]
age_band = ["adult", "child"]
n_speakers = 139
n_tokens = 1668

[measure]
kind = "observed_distribution"   # or "target_zone" | "summary_normative_range"
parameters = ["F1", "F2", "F3"]
units = "Hz"
phones = ["iy", "ih", "ey", "eh", "ae", "ah", "aw", "ao", "ow", "uw", "uh", "er"]
context = "hVd"
measurement_method = "steady-state, manually selected"

[privacy]
shareability = "raw_samples"     # or "summary_only"
min_n_per_subgroup = 5
community_consent = false        # required true for small-language community data

[schema]
data_file = "data.parquet"
shape = "long"
columns = ["speaker_id", "phone", "F1", "F2", "F3"]
```

The `measure.kind` enum keeps prescriptive targets clearly distinct from observed populations while sharing infrastructure:

- `observed_distribution` — raw samples from a measured population
- `summary_normative_range` — summary stats only (mean / SD / percentiles), no raw values shipped
- `target_zone` — prescriptive goal regions (voice-coach / L2 use); not an empirical claim about a population

GUI renders them differently and labels them clearly so "what people sound like" is never conflated with "what they should aim for."

### Decisions per sub-question

| Sub-question | Decision |
|---|---|
| **Curation** | Three-tier model. Tier 2 editorial committee starts with project maintainers; expansion by adding maintainers to the registry repo |
| **Versioning** | Semantic. Immutable once published. Yanks possible (deprecation flag in index) but old pins keep resolving with a warning |
| **Licensing** | Prefer CC0 or CC-BY-4.0 for data; ODC-BY for databases. **Disallow non-commercial / no-derivatives in tier 2.** Allow in tier 3 with prominent flags. Matches our own permissive lean (cross-references the 2026-05-18 licensing entry) |
| **Privacy / k-anonymity** | Required `min_n_per_subgroup` in manifest (default 5). `shareability` field declares whether raw samples or only summaries are shipped. Tier 2 enforces; tier 3 surfaces. `community_consent` required for small-language community data |
| **Discovery** | Faceted search on manifest tags (language, variety, sex, age, measure, phones, kind); free-text over titles + provenance |
| **Citation** | Manifest carries BibTeX + (optional) DOI. Engine API records which distributions any analysis used. Project export-to-paper produces a citation list automatically |
| **Disputes / errors** | Conflicting distributions co-exist in tier 3. Tier 2 picks at most one per (measure × population × method). GitHub-issue-style error reports against published distributions. Corrections ship as version bumps |
| **Transport** | Plain HTTP-served static registry index + tarball downloads. GitHub Pages-hosted. Cached at user level (`~/.local/share/sat/refdist/`). App ships with official registry URL configured; accepts additional registries |

### Hosting: GitHub Pages, PR-based

Concretely:

- A public GitHub repo (e.g. `speech-analysis-tool/refdist-registry`) holds all tier 2 + tier 3 entries.
- Tier 1 (starter set) lives in the main app repo and is bundled at build time.
- Submission = PR adding a distribution directory under `tier2/` or `tier3/`.
- CI validates: TOML schema, manifest required fields, license file present and acceptable, `min_n_per_subgroup` enforced, license compatible with tier, data-file matches declared schema.
- Merge to main triggers Pages rebuild. The app fetches the rendered index JSON and individual tarballs.
- Tier-2 promotion = a second PR moving a directory from `tier3/` to `tier2/` after editorial review.
- Yanking = `yanked = true` flag in the entry; old pinned versions keep resolving with a warning surfaced in the GUI.
- DOI assignment is optional and separate; Zenodo can mint DOIs for GitHub releases if a publisher wants one.

### Architectural touch points

How this plugs into what we've already designed:

- **Distributions live outside the corpus**, at user level. Projects *reference* distributions by `id` + `version`; the engine resolves to the local cache (or fetches once if missing).
- **Projects pin versions for reproducibility.** `project.toml` records which distribution versions were used; opening on another machine fetches the same pinned versions.
- **Engine API exposes distributions as queryable objects** — e.g. `engine.refdist.query(measure="F1", population={lang:"eng", sex:"f", age_band:"adult"})` returns matching distributions; GUI renders as overlays / target zones / percentile bands.
- **Tier 3 publishing is in-app.** Users can package any analysis result as a distribution from inside the desktop GUI; the manifest is scaffolded automatically. Publication is a `git push` to a fork-and-PR flow (auth is the user's GitHub credentials, not ours).
- **Cited automatically.** When a project uses a distribution in any analysis, the citation is recorded; project export emits the citation list.

### Edge-case tests the model handles

- **Clinical norms from private patient data.** Hospital with n=200 dysphonic patients publishes jitter norms as `shareability = "summary_only"`. Only mean / SD / percentile bands ship in `data.parquet`. Provenance notes the source corpus is private. ✓
- **Voice-coach target zones (heuristic, not population-derived).** Published with `measure.kind = "target_zone"`. GUI labels distinctly from observed distributions. Important not to conflate "what is" with "what should be." ✓
- **Conflicting vowel-formant studies for the same population.** Both live in tier 3; users compare via quality signals (n, year, method, citations). Tier 2 picks at most one per (measure × population × method). ✓
- **Field-linguistics small-language data.** `community_consent = true` required for tier 2; flagged but not enforced for tier 3. CARE / FAIR principles respected by data publishers, surfaced by the platform. ✓
- **Forensic expert using reference from a private corpus.** Published as `summary_only` with provenance pointing to the private corpus (citable but not downloadable). Reproducible at the summary level; underlying data is not. ✓

### v1 deliverables this entry commits to

1. Bundled starter set with the app (Hillenbrand 1995, Peterson-Barney 1952, a small clinical normative-range set).
2. Public registry repo on GitHub with tier 2 / tier 3 directory structure.
3. CI validation pipeline (manifest schema, license check, min-n enforcement, data-file conformance).
4. GitHub Pages-rendered index JSON consumable by the engine.
5. Engine API for distribution resolution, caching, and query.
6. Project pinning of distribution versions in `project.toml`.
7. In-app publishing flow with auto-scaffolded manifests and fork-and-PR submission.
8. GUI rendering for overlays / target zones / percentile bands distinguishing `measure.kind` variants.

### Open trade-offs / deferred

- **Exact starter-set contents.** Need to confirm redistribution rights for each candidate dataset; start with publicly-licensed material only.
- **Initial editorial committee composition.** Starts as just the project maintainers; expansion process and quality bar can codify after launch and a few PRs.
- **Trust-signal richness in tier 3.** v1: download count + flag-as-questionable + tier 2 promotion. Richer signals (verified-publisher badges, citation counts) layered later.
- **Registry repo sharding.** Single repo to start; shard by domain (clinical, phonetic, forensic, …) if PR rate justifies.
- **Federated / additional registries.** App accepts additional registry URLs; whether to actively encourage forks (e.g. an organization's internal registry) is a community-development question, not a v1 architecture one.
- **DOI integration depth.** Zenodo can mint DOIs for tier 2 releases automatically; whether to require this for tier 2 or keep it optional needs a call.
- ~~**Profile-driven defaults.**~~ Closed 2026-05-20: each of the five v1 profiles ships with a "recommended distributions" list consumed by the refdist picker GUI. Specific distribution IDs still need sync with the actual tier-1 starter set as Phase 1 work lands. See 2026-05-20 profile-catalog entry.

### Sources / references

- CRAN policies: https://cran.r-project.org/web/packages/policies.html
- HuggingFace Hub: https://huggingface.co/docs/hub
- Zenodo: https://zenodo.org ; OSF: https://osf.io
- MFA models registry: https://mfa-models.readthedocs.io
- NHANES: https://www.cdc.gov/nchs/nhanes
- OLAC: http://www.language-archives.org ; PARADISEC: https://www.paradisec.org.au ; ELAR: https://www.elararchive.org
- CARE Principles for Indigenous Data Governance: https://www.gida-global.org/care
- FAIR Principles: https://www.go-fair.org/fair-principles
- Hillenbrand et al. 1995, JASA: https://doi.org/10.1121/1.411872
- Peterson & Barney 1952, JASA: https://doi.org/10.1121/1.1906875

---

## 2026-05-18 — Corpus data model: directory + SQLite + Parquet, six-type tier model

Goal: design the corpus data model — the structural decision flagged 2026-05-16 as "probably the single most consequential architectural decision." Cover the container shape, storage technology, annotation tier model, entity tables, round-trip interop, mobile sync, and schema migrations.

### What the corpus model has to carry

Pulling obligations from prior decisions and user groups:

- **8 user groups, 8 shapes of "project."** Forensic case files, clinical patient longitudinal data, field-linguistics sessions, ML datasets, experimental trial logs, voice-coach practice journals — one model has to fit all without collapsing into mush.
- **Three signal categories.** Original audio (WAV / FLAC); derived DSP signals (F0, formants, spectrograms — recomputable but expensive); ML embeddings (wav2vec2-class signals at 768-dim per 20 ms ≈ 38 MB/min — *must* persist).
- **Annotations are the secret center of gravity** (cross-cutting F): TextGrid, EAF, lexicon entries, trial logs, patient charts are all the same shape underneath.
- **Live recording must land cleanly** without corrupting the project on crash.
- **Mobile companion appends via sync** (no concurrent edits).
- **Forensic / clinical need audit trails** — every analysis step recorded.
- **Scale spans 4 orders of magnitude** — phonetician with 50 recordings ↔ ASR dataset with 1M utterances, same model.

### Prior art surveyed

| Tool | Lifted from | Left behind |
|---|---|---|
| **EMU-SDMS (emuDB)** | Hierarchical annotation levels with parent-child relations; bundle = (recording + annotations) as a unit; DBconfig as schema document | R-centric query layer; rigid migration story |
| **ELAN (.eaf)** | Tier types + controlled vocabularies + reference relationships between tiers | XML; no corpus structure beyond a folder of EAFs; no derived signals |
| **Praat TextGrid** | Flat interval/point tiers as the lowest-common-denominator interchange format | File-as-unit; no metadata, schema, or hierarchy |
| **Phon** | SQLite-as-project-file precedent for transcription corpora | Domain-narrow; not signal-rich |
| **NWB / HDF5 (neuroscience)** | Hierarchical container for huge time-series signals; self-describing; language-agnostic | Weak annotation story; not graph-queryable |
| **Apache Arrow / Parquet** | Columnar storage for derived signals and ML embeddings; zero-copy across Rust / Python; ecosystem-native for AI engineers | Not for audio blobs; not for annotation graphs |
| **SQLite (Phon, Anki, Lightroom, Logic Pro precedent)** | Single-file ACID DB as project container; queryable; trivially backed up | Not great for huge audio BLOBs |
| **Lightroom catalog model** | Directory containing a DB + signal files + sidecars; DB holds metadata, signals on disk | — |

EMU is the closest reference; the others contribute pieces.

### Five design axes and decisions

| Axis | Options considered | Decision |
|---|---|---|
| **Container shape** | Single file (SQLite-only); directory with embedded DB; graph-DB server | **Directory with embedded DB**. Plays well with `cp` / `rsync` / Time Machine; audio doesn't bloat the DB; standard scientific-app pattern |
| **Storage tech** | SQLite vs DuckDB for relational; Parquet vs HDF5 for dense signals; WAV vs FLAC for audio | **SQLite + Parquet + WAV (FLAC archival opt-in)**. SQLite for transactional editing maturity; Parquet for zero-copy Rust↔Python and AI-ecosystem fit; WAV for lossless universal originals |
| **Annotation model** | Praat-flat-tiers; ELAN tier-types; EMU hierarchical levels; unified type system | **Six-type tier model with parent-child cardinality** (see below). Unifies Praat / ELAN / EMU / ML signals / trial metadata |
| **Entity model** | Fixed schema + JSON extra; user-defined-schemas first-class; per-profile fixed schemas | **Fixed core schema + JSON `extra` column per entity**. Profiles ship as schema validators for `extra`. Fast queries, simple migrations, extensible per project |
| **Audit / versioning** | Audit log only; soft versioning (immutable annotations); Git-like full history | **Append-only audit log, no time-travel queries in v1**. Forensic-defensible, reproducibility-friendly, simple. Time-travel can layer in later |

### Annotation tier model

A `Tier` is a typed container with an optional parent relation:

```
Tier
  id           uuid
  bundle_id    uuid              -- which recording this tier belongs to
  name         text              -- e.g. "phones", "F0", "trials"
  type         enum              -- six variants, below
  parent_id    uuid?             -- null for top-level tiers
  cardinality  enum?             -- one_to_one | one_to_many | many_to_one | none
  schema       json              -- type-specific config (hop, dims, vocab)
  extra        json              -- user fields
```

Six tier types, with storage shape:

| Type | Storage | Praat / ELAN / EMU analogue | Primary uses |
|---|---|---|---|
| `interval` | rows in `annotation_interval` (DB) | Praat IntervalTier; ELAN ALIGNABLE_ANNOTATION | Phones, words, segments |
| `point` | rows in `annotation_point` (DB) | Praat TextTier; ELAN TIME_SUBDIVISION | Event markers, clicks, glottal pulses |
| `continuous_numeric` | Parquet sidecar (dense array); header in DB | EMU SSFF track | F0, intensity, jitter-over-window |
| `continuous_vector` | Parquet sidecar (n_frames × n_dims) | EMU SSFF multi-channel | wav2vec / HuBERT embeddings, MFCC, mel-spec |
| `categorical_sampled` | Parquet sidecar (RLE or dense) | — | VAD, voicing on/off, model class output |
| `reference` | rows in `annotation_reference` (DB) | ELAN SYMBOLIC_ASSOCIATION; FLEx lexical ref | Lexicon links, trial links, speaker turns |

**Storage split rationale.** Sparse tiers (interval, point, reference) live in the DB — queryable, editable, transactional. Dense tiers (numeric, vector, categorical sampled) live in Parquet sidecars next to the audio — they're huge (an hour of wav2vec layer-N embeddings is ≈ 2 GB), append-friendly, and the AI-engineer audience can mmap them directly without going through SQLite.

**Parent-child cardinality** captures EMU's hierarchical model: `one_to_one`, `one_to_many` (the common case: word → phones), `many_to_one` (the inverse view), or `none` (independent tier).

### v1 entity tables

```
Project           -- one row; project-level metadata, schema version, profile
Speaker           -- person who produced speech (could be participant/patient/case)
Session           -- recording session: time, place, instrument, calibration
Bundle            -- one recording: audio file + everything derived from it
Tier              -- annotation container; lives in a bundle
Annotation*       -- per-type tables: interval, point, reference
DerivedSignal     -- registration row for a Parquet sidecar
Entity            -- generic typed entity (Patient, Case, Trial, Stimulus, …)
EntityRef         -- links from reference tiers / sessions / bundles to entities
Protocol          -- recording protocol / form definition (CAPE-V, …)
Instrument        -- mic / interface / calibration data
ModelRun          -- registry of which ML model produced which signal/annotation
AuditLog          -- append-only log of all mutations
```

A note on `Entity`: rather than separate `Patient`, `Case`, `Trial` tables, a single `Entity` table with a `kind` discriminator and a JSON `extra` payload keeps the schema small. Common queries hit indexed columns (`kind`, `name`, `project_id`); group-specific fields live in `extra`. **Profiles** (clinical, forensic, phonetician, etc.) ship as schema validators for `extra` — the user gets typed forms in the GUI even though storage is flexible.

Every entity table has an `extra: json` column. Every mutation gets a row in `AuditLog` with `(timestamp, user, table, row_id, op, before, after)`.

### Project directory layout

```
my_project/                          ← the "project" is a directory
├── project.toml                     ← human-readable header (name, schema version, profile, …)
├── corpus.db                        ← SQLite: schema, entities, annotations, audit log
├── signals/
│   ├── original/                    ← source recordings (WAV; FLAC archival)
│   │   └── session_001/
│   │       ├── bundle_001.wav
│   │       └── .in_progress/        ← active live recordings land here
│   └── derived/                     ← cached / persisted derived signals (Parquet)
│       └── session_001/
│           ├── bundle_001.f0.parquet
│           ├── bundle_001.formants.parquet
│           └── bundle_001.wav2vec.layer8.parquet
├── attachments/                     ← user docs, images, calibration files
├── exports/                         ← round-trip TextGrid / EAF for interop
└── .lock                            ← single-writer file lock
```

### Live recording integration

Audio writes to a temp file under `signals/original/.in_progress/`. On recording end, a single SQLite transaction atomically moves the file to its final path and inserts the `Bundle` + `DerivedSignal` rows. A crash leaves a recoverable temp file; the project DB stays consistent.

### Round-trip interop

Adoption hinges on TextGrid round-tripping cleanly enough to satisfy existing Praat workflows. Tier-type coverage by format:

| Our tier type | Praat TextGrid | ELAN .eaf | EMU emuDB |
|---|---|---|---|
| `interval` | Direct (IntervalTier); attrs lost unless JSON-in-label | Direct (ALIGNABLE_ANNOTATION); limited attrs | Direct (interval level with attrs) |
| `point` | Direct (TextTier); attrs lost | Direct (TIME_SUBDIVISION) | Direct (point level with attrs) |
| `continuous_numeric` | Separate Pitch/Matrix file | Linked external file | Direct (SSFF track) |
| `continuous_vector` | Not representable | Not representable | Direct (SSFF multi-channel) |
| `categorical_sampled` | Lossy: collapse runs into IntervalTier | Lossy: collapse into time-aligned | Direct |
| `reference` | Lossy: flatten ref to label text | Direct (SYMBOLIC_ASSOCIATION) | Direct (reference relation) |
| Parent-child cardinality | Lost | Direct (parent_tier_ref) | Direct (level hierarchy) |

**Export targets, ranked by adoption value:**

1. **TextGrid (Praat) — table stakes, deliberately lossy.** Interval and point tiers. Optional attribute round-tripping via a JSON sentinel inside the label (`{json:{…}}`) that Praat ignores and we re-parse. Document loudly what's lost.
2. **EAF (ELAN) — primary annotation interop.** Hierarchical, lossless for the subset ELAN supports.
3. **emuDB — most faithful export.** Maps nearly everything we have.
4. **Parquet / Arrow — for dense signals only.** Direct export of `continuous_*` tiers; the format the AI-engineer audience already lives in.
5. **Our own project archive (`.satproj.tar` or similar) — fully lossless.** Tarball of the project directory; the format you use to share a project between instances of the tool.

**Critical boundary: export is a snapshot, not a sync.** A user who exports a TextGrid, edits it in Praat, and re-imports gets a *new annotation set* on the bundle — not a merge. Maintaining a two-way bridge to a richer model has no good failure mode. The exception is our own archive format, which round-trips losslessly because it IS our format.

### Mobile sync — bundle pack format

Mobile is recording-only; sync is one-directional and append-only. The unit of sync is one completed bundle:

```
bundle_001.satbundle.tar
├── manifest.toml       -- bundle id, session id, recording params, instrument,
│                          calibration, speaker ref, timestamps
├── audio.wav           -- the recording
├── tiers/              -- annotations created on phone (live VAD, user-marked points)
│   └── vad.json
└── derived/            -- signals computed on phone (live pitch track, etc.)
    └── pitch.parquet
```

**Ingest flow.** Desktop project watches one or more sync inbox locations (local folder, cloud folder, LAN, …). For each bundle pack: validate manifest → copy audio → copy derived → insert DB rows → mark consumed. Transactional; failures leave the pack in the inbox with an error flag.

**Transport is unspecified.** AirDrop, USB, Syncthing, Dropbox, iCloud, LAN — all valid; all just shuffle a portable archive. Local-first (decision #10) means no required server.

**Session assignment.** Phone either declares "this bundle belongs to session X" (created on desktop, pushed to phone) or drops into a "mobile-default" session the user reorganizes later.

### Schema migrations

- **Tool: `refinery` or `sqlx::migrate!`.** Numbered SQL files versioned in the engine crate. DB stores its schema version in a `schema_migrations` table.
- **Engine refuses to open a DB newer than it knows** (forward-compat clamp).
- **Older-schema DBs**: additive changes (new columns / tables) auto-apply on open; destructive changes (renames / drops) prompt for explicit confirmation. Every upgrade writes `corpus.db.bak.<old_version>` first.
- **JSON `extra` columns don't need DDL migration**, but profile-level `extra` schemas evolve — ship profile migration utilities separately that walk rows and update payloads.
- **Forward compatibility within a minor version**: additive changes don't bump major. Engine SQL must avoid `SELECT *` so older code can ignore unknown columns.
- **Parquet sidecars carry their own embedded schema** — no migration needed. Changes to derivation produce new sidecars; old ones stay valid.

### v1 deliverables this entry commits to

1. Directory-shaped project with `corpus.db` + `signals/` + `attachments/` + `exports/`.
2. SQLite-backed entity schema as enumerated above, with JSON `extra` columns and append-only `AuditLog`.
3. Six-type tier model with parent-child cardinality, split storage (DB for sparse, Parquet for dense).
4. Live recording via `.in_progress/` + atomic commit.
5. TextGrid + EAF + emuDB + Parquet + project-archive export paths. Praat TextGrid round-trip is the only one with bidirectional re-import in v1.
6. Mobile bundle-pack format and append-only sync inbox.
7. Migration policy via `refinery` / `sqlx::migrate!`.

### Open trade-offs / deferred

- ~~**ML model registry scope**~~ Closed 2026-05-20: split into provenance layer (`ProcessingRun` table — replaces `ModelRun` here) and distribution layer (parallel registry to refdist, HF passthrough escape hatch). See 2026-05-20 ML-model-registry entry.
- **Reference distributions** — entangled with the 2026-05-16 governance question (who curates, versions, licenses). Its own follow-up entry.
- **Concurrency** — single-writer file lock for v1; multi-user is a v2+ concern requiring a coordination story we don't owe yet.
- ~~**Profile catalog**~~ Closed 2026-05-20: five profiles ship at v1 — phonetician (default), clinical, forensic, field, experimenter. Voice training deferred to v1.x with Phase 6 mobile. See 2026-05-20 profile-catalog entry.
- **TextGrid attribute-sentinel format** — the `{json:{…}}` convention needs a precise spec so it's robust to round-trips through other tools.
- **Cross-bundle queries** — DB layout supports them (one project = one DB), but the API surface (how a phonetician asks "all phones tagged `[ɛ]` across all bundles from male speakers") is a query-language decision deferred to the API-surface entry.
- **Hierarchical query semantics** — EMU has a full path-expression language (EQL). What subset do we ship in v1?

### Sources / references

- EMU-SDMS / emuDB: https://ips-lmu.github.io/EMU.html
- ELAN EAF format: https://www.mpi.nl/tools/elan
- Praat manual (TextGrid spec): https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html
- Phon: https://www.phon.ca
- NWB (Neurodata Without Borders): https://www.nwb.org
- Apache Arrow: https://arrow.apache.org ; Parquet: https://parquet.apache.org
- SQLite as application file format: https://sqlite.org/appfileformat.html
- refinery: https://github.com/rust-db/refinery
- sqlx: https://github.com/launchbadge/sqlx

---

## 2026-05-18 — Licensing: permissive-leaning, GPL kept on the table

Goal: settle the license direction before the dependency tree is locked in, so we don't discover an incompatible dep later. Triggered by the realization that licensing affects which downstream user groups can actually productize on top of the engine.

### Audit of the stack picked in the previous entry

Almost the entire dep tree is permissive (MIT / Apache-2.0 / dual), which keeps every licensing option open for our own code. Two items worth flagging:

| Dep | License | Note |
|---|---|---|
| Rust toolchain, `rustfft`, `realfft`, `cpal`, `egui`, `wgpu`, `glow`, `ort`, `candle`, `burn`, `PyO3` | MIT and/or Apache-2.0 | Permissive across the board |
| CPython (when embedded) | PSF | BSD-compatible |
| **`UniFFI`** | **MPL-2.0** | File-level copyleft only — using as a dep does *not* infect our code; only modifications to UniFFI's own source files would have to be MPL |
| **`rust-jack` → libjack** | MIT crate; libjack is LGPL-2.1 | Dynamic linking (the cpal default) is fine for any downstream license. Avoid static linking libjack |

Two small disciplines fall out:
- Don't fork UniFFI itself (or accept MPL on the fork). Using it is fine for any license.
- Keep libjack dynamic-linked, which is the default. The PipeWire JACK-compat shim is MIT, which is even cleaner on PipeWire-default distros.

### Patent grants matter for the ML path

Apache-2.0 includes an explicit patent grant; MIT does not. For a project with substantial ML inference code (cross-cutting pattern G), the patent grant is non-trivial protection against future patent assertions. This is why the Rust convention is **dual Apache-2.0 OR MIT** — downstream users pick, but the patent grant is available. We should match that convention regardless of whether we add a stronger copyleft layer.

### The three license shapes considered

1. **Permissive (Apache-2.0 OR MIT).** InFormant, MFA, librosa, ESPnet path. Anyone can fork, integrate, build commercial products on top. Doesn't require improvements back. Maximizes adoption and downstream productization.
2. **Weak copyleft (MPL-2.0 or LGPL-3.0).** Engine improvements come back, but downstream apps that *use* the engine can be any license. MPL is more modern and avoids LGPL's static/dynamic-linking gymnastics. A proprietary clinical app could link our engine; patches to the engine itself would be MPL.
3. **Strong copyleft (GPL-3.0+).** Praat, Phonometrica, ELAN, Parselmouth path. Any derivative work must also be GPL. Strong free-software statement; closes the door to FDA-cleared clinical products, forensic court-deployable products, and most mobile/consumer apps built on top. Phonometrica is the cautionary tale: serious technical effort whose adoption partly stalled on license-driven integrator deterrence.

### How license interacts with the 2026-05-16 user groups

| Group | Sensitivity | Direction |
|---|---|---|
| 1. Phoneticians | Low | Neutral |
| 2. AI engineers | Low | Mild permissive lean (HuggingFace/PyTorch world is mostly permissive) |
| 3. Clinical (SLPs, ENT) | **High** | Strong permissive lean — FDA-cleared products on GPL code is effectively unheard of |
| 4. Voice coaches | Medium | Permissive lean — mobile/consumer apps won't adopt GPL |
| 5. L2 learners | Medium | Permissive lean — same as above |
| 6. Field linguists | Low–medium | Somewhat copyleft-sympathetic culturally, but practical access matters more |
| 7. Forensic | **High** | Strong permissive lean — proprietary court-deployable tools won't touch GPL |
| 8. Experimenters | Low | Mixed (PsychoPy is GPL; jsPsych/PCIbex are MIT) |

Aggregate: **permissive or MPL maximizes the user-group reach we explicitly scoped.** GPL would meaningfully reduce reach in the clinical and forensic directions.

### Provisional stance

**Lean: permissive — dual Apache-2.0 OR MIT** for the engine, the Python module, and the desktop GUI. Matches Rust-ecosystem and speech-ML-ecosystem convention, doesn't cut off clinical/forensic groups, gives downstream users patent protection by default.

**Not committed.** GPL-3.0+ is left on the table as a possible ambition: a deliberate choice to prioritize software freedom over reach. Revisit before the first public release. MPL-2.0 is the principled middle ground if we want engine improvements to come back without infecting downstream integrations.

### What follows regardless of which license we land on

These are non-negotiable independent of the license choice:

1. **Do not bundle model weights.** Ship code that downloads them; let users inherit their licenses (most popular speech model weights are MIT/Apache anyway, but the principle keeps us clean).
2. **Do not redistribute reference corpora that aren't openly licensed.** Reference Hillenbrand / Peterson-Barney etc. by pointing at them; bundle only corpora we have clear redistribution rights for.
3. **Do not wrap or link Praat itself.** Praat is GPL-3.0+; any linking inherits GPL (this is how Parselmouth became GPL). Re-implementing algorithms from the literature is fine; copying Praat source is not. The 2026-05-16 decision to make "running Praat scripts a non-goal" already aligns with this.
4. **Keep libjack dynamic-linked** on Linux (cpal default), and don't fork UniFFI without accepting MPL on the fork.
5. **Engine API stability matters more under permissive licensing** because commercial integrators will lock to versions and resist breaking changes. Already a discipline we owed Python users; permissive licensing just amplifies it.

### Open questions / deferred

- **Final license decision** — defer until closer to first public release; revisit if the user-group prioritization shifts (e.g. if we explicitly de-scope clinical/forensic productization, GPL becomes more palatable).
- **Contribution license.** Whether to require a CLA, use a DCO sign-off, or rely on inbound = outbound. Affects whether we can ever relicense.
- **Trademark.** Project name and any logo — separate from code license, often overlooked, matters for community-fork dynamics.

---

## 2026-05-18 — Tech stack: engine-first Rust + egui + PyO3 + UniFFI

Goal: pick the host language and surface technologies for v1, given the architectural implications surfaced 2026-05-16. The tech-stack discussion was explicitly deferred at the end of that session; this entry closes it.

### Constraints inherited

Six prior decisions narrow the field before any tech is named:

| Source | Constraint |
|---|---|
| Decision #6 | Desktop primary, phone as thin companion (not parity) |
| Decision #5 | Real-time analysis path designed in, not bolted on |
| Decision #8 | Scriptable via Python API; host language separate from API |
| Decision #3 / pattern G | ML inference (ONNX-family + others) as a first-class signal type |
| Decision #10 | Local-first, encryption at rest |
| Implicit | Solo-dev / small-team capacity; can't carry two full native UI codebases at desktop scope |

### Decision axes

Three axes determine almost everything; the rest follows.

- **A. Engine language.** Where DSP + real-time + ML inference live. Realistic candidates: Rust, C++. (Python-as-engine is too slow for real-time; JVM/Dart need a native layer underneath anyway, just deferring the question.)
- **B. Desktop GUI surface.** Native toolkit (Qt), immediate-mode (egui / iced), webview (Tauri / Electron), declarative cross-platform (Compose / Flutter / Slint).
- **C. Engine ↔ GUI ↔ mobile ↔ Python coupling.** Monolithic app that exposes APIs, or a portable engine library that multiple shells consume?

### Decisions, top-down

**Coupling — engine-first.** The engine is the deliverable: a Rust crate with a stable, versioned API. Desktop GUI links it in-process. Mobile shells consume it via UniFFI-generated Swift / Kotlin bindings. The Python module (PyO3) wraps the same engine for library users and for the in-app scripting host. This makes the engine ↔ Python boundary a first-class API surface, designed-for from day one rather than retrofitted.

**Engine language — Rust.** Memory safety in a long-running stateful tool with real-time threads is non-trivially valuable. Mature crates for everything we need: DSP (`rustfft`, `realfft`), audio I/O (`cpal`), GUI (`egui` + `wgpu`), ML inference (`ort`, `candle`), Python bindings (`PyO3`), mobile bindings (`UniFFI`). C++ + Qt is the conservative alternative — better-trodden in the domain (Praat, Phonometrica) — but pulls in CMake/footgun overhead, weaker Python embedding ergonomics, and more painful long-term maintenance for a small team.

**Desktop GUI — egui.** The central UI element is dense custom-rendered viz (spectrograms with tier overlays, formant tracks, real-time scrolling, scrub cursors). egui's painter API plus a `wgpu` backend gives Rust direct GPU access to draw this; no IPC, no serialization boundary, no webview, no Linux-distro variance. Closest reference: Rerun is architecturally the same shape (Rust engine + egui scientific viewer over huge multi-modal time-series data) and uses egui. Trade-off accepted: egui's default widgets look "tool-like" rather than native — fine for a research/clinical tool, less ideal if the app were positioned as a polished consumer voice-coach product. Slint is the runner-up if conventional UX matters more than I'm weighting it now. Tauri was ruled out because its webview boundary buys nothing for viz-heavy UI and risks WebKitGTK performance variance on Linux. Compose Multiplatform was ruled out once we decided mobile is companion, not parity — Compose's mobile-parity advantage doesn't apply.

**ML inference — `ort` primary, `candle` for mobile.** `ort` (ONNX Runtime Rust bindings) runs arbitrary ONNX graphs, covering the breadth of speech models published on HuggingFace: wav2vec2, HuBERT, WavLM, Whisper, ECAPA-TDNN / x-vector speaker embeddings, Silero VAD, etc. Breadth is essential for the "ML features as first-class signal" pattern (G) — users will want to drop in arbitrary fine-tuned models. `candle` (HuggingFace's pure-Rust framework) is narrower — only architectures that have been ported to Rust code — but pure-Rust means clean cross-compilation, especially for mobile, where ONNX Runtime's iOS/Android binaries are heavy. Mobile's ML surface is narrow (live VAD, maybe pitch enhancement), so candle is plausibly enough there. `burn` worth tracking as a third option but not v1.

**Audio I/O — `cpal` with JACK on Linux.** `cpal` is the standard Rust cross-platform audio I/O crate (CoreAudio / WASAPI / ALSA / JACK). On Linux, enable the `jack` feature and prefer the JACK host when available — works against PipeWire's JACK-compatibility API too, so modern distros are covered. ALSA-direct fallback for systems without JACK/PipeWire. Real-time audio thread needs `SCHED_FIFO` priority. If cpal's JACK backend disappoints in practice (historically less polished than its ALSA backend), local fallback is the `jack` crate directly on Linux. Audio I/O is wrapped behind an engine abstraction so the swap is contained.

**Mobile — native shells via UniFFI.** iOS: SwiftUI on top of UniFFI-generated bindings. Android: Jetpack Compose on the same. Scope kept strictly narrow: live recording, live feedback (pitch / formants), upload-and-sync to desktop project. No corpus management, no annotation editing, no scripting on phone. Writing both UIs natively is acceptable because the scope is bounded; we get native audio I/O (AVAudioEngine / AAudio) and native distribution as a side benefit.

**Python — PyO3, included in v1 (revised mid-session).** Initial framing deferred Python to v2 to simplify v1 scope. Reconsidered: scripting and GUI use are equally important user surfaces. The phonetician scripting tradition (group 1) and the AI-engineer / Parselmouth-replacement audience (group 2) are entirely gated on the Python API existing. The engine ↔ Python boundary needs to be designed for Python from day one — retrofitting in v2 would force a re-architecture once real scripting needs surface. Two delivery shapes share the same module: (a) standalone PyO3 module — `import speech_analysis_tool` from any CPython, library-shaped like Parselmouth; (b) embedded in the desktop app — same module imported by an embedded CPython, serving as the in-app scripting host. Costs accepted: GIL discipline around long-running engine ops, NumPy interop for signal arrays, embedded-CPython distribution complexity (well-trodden in DCC tools — Blender, Houdini — but non-trivial), API stability earlier than otherwise.

### Architecture

```
                  ┌────────────────────────────────────┐
                  │  Rust engine crate                 │
                  │  - DSP (F0, formants, spectro, …)  │
                  │  - ML inference (ort; candle on    │
                  │    mobile)                         │
                  │  - Corpus & annotation model       │
                  │  - Real-time audio loop (cpal)     │
                  │  - Reference-distribution store    │
                  └────────────┬───────────────────────┘
                               │ stable Rust API
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼─────┐         ┌──────▼──────┐         ┌─────▼──────┐
   │ Desktop  │         │ PyO3 module │         │ UniFFI     │
   │ egui GUI │         │ (Python)    │         │ bindings   │
   │          │         │             │         │            │
   │ in-proc  │         │ standalone  │         │ SwiftUI    │
   │ wgpu     │         │ library OR  │         │ (iOS)      │
   │ no IPC   │         │ embedded as │         │ Compose    │
   │          │         │ in-app host │         │ (Android)  │
   └──────────┘         └─────────────┘         └────────────┘
```

### v1 deliverables, after this entry

1. Rust engine crate with a stable, versioned public API.
2. Desktop GUI (egui) consuming the engine in-process.
3. PyO3 Python module exposing the same engine — both as a standalone library and as the embedded scripting host inside the desktop app.
4. iOS + Android companion shells via UniFFI, scoped to live record + live feedback + sync to a desktop project.

### Open trade-offs / deferred

- **egui ergonomics for forms / dialogs.** Most of our UI is custom viz, but settings panels, file pickers, batch-job config forms exist and will feel less polished than native. Acceptable if we keep that surface small; revisit if it grows.
- **Embedded CPython distribution.** Signing on macOS, library paths, GIL across the FFI boundary. Worth a focused spike before committing to "embedded scripting host in v1" specifically — the standalone PyO3 module is cheaper and could ship first, with the embedded host following.
- **cpal's JACK backend reliability.** Watch for xruns / sample-rate negotiation issues. Fallback: drop to the `jack` crate directly on Linux behind the audio-I/O abstraction.
- **wgpu vs glow as egui backend.** `wgpu` is the modern path (Vulkan / Metal / DX12 / WebGPU). `glow` is the legacy OpenGL backend, more compatible on older Linux GPUs. Start with `wgpu`; keep `glow` available as a fallback build for distribution edge cases.
- **Python API surface scoping.** What's in v1's public API matters more when Python is in v1 — anything we expose, users will write scripts against and resist us breaking. Worth a dedicated follow-up entry on API surface design.
- **In-app scripting host UX.** Notebook-style? Script-file editor? REPL panel? Praat-script-editor analogue? Affects how Python embedding actually surfaces in the GUI, and overlaps with the open questions still pending from 2026-05-16.

### Sources / references

- Rerun (reference architecture: Rust engine + egui scientific viewer): https://rerun.io ; https://github.com/rerun-io/rerun
- egui: https://github.com/emilk/egui
- Slint: https://slint.dev
- Tauri: https://tauri.app
- cpal: https://github.com/RustAudio/cpal
- rust-jack: https://github.com/RustAudio/rust-jack
- ort (ONNX Runtime Rust bindings): https://ort.pyke.io
- candle: https://github.com/huggingface/candle
- burn: https://github.com/tracel-ai/burn
- PyO3: https://pyo3.rs
- UniFFI: https://mozilla.github.io/uniffi-rs/
- wgpu: https://wgpu.rs

---

## 2026-05-16 — Target user groups: features, ideals, cross-cutting patterns

Goal: enumerate the user groups a Praat successor might serve, list what tooling they use today and what features they actually rely on, sketch what an ideal experience would look like, and surface patterns that should inform architecture.

Scope decisions taken this session (in response to the six open questions from 2026-05-15):

| # | Question | Decision |
|---|----------|----------|
| 1 | Target users | All-in-one ambition; enumerate to find overlaps (this entry). Phonetic research and speech-AI engineering are user-profile givens. |
| 2 | Scope | "All-in-one" as a north star, accepting it may end up "some-in-one." Let developability drive trimming. |
| 3 | Architecture stance | Desktop GUI primary. Mobile companion seriously considered if workflows fit. |
| 4 | Scripting | Definitely scripted. Form TBD. Python is the accessibility incumbent but may not be the right host-language for the app itself. |
| 5 | Praat compat | TextGrid I/O is table stakes. Running Praat scripts is a non-goal for now. |
| 6 | ML vs classical DSP | Both, first-class. |
| 7 (new) | Experimental interfaces | In scope. Production/perception experiment runner (Praat Demo Window / PsychoPy overlap), stimulus elicitation. |

### Groups

User profile context: the user is themselves a phonetician + speech-AI engineer, so groups 1 and 2 are taken as givens and described briefly. Groups 3–8 get fuller treatment.

#### 1. Research phoneticians  *(given — user profile)*

Praat + Parselmouth + MFA + R/Python stats, sometimes EMU-SDMS, ELAN for multi-tier or video. Plugins: FastTrack (formant exploration), ProsodyPro (F0). Articulatory work (ultrasound, EMA, palatography) lives in separate proprietary tools — AAA, MView. The frictions are well-known: arcane scripting, no undo, manual stitching of forced-alignment / annotation / stats stages, weak articulatory channel support, no native ML feature inspection.

#### 2. Speech AI / ML engineers  *(given — user profile)*

`librosa` / `torchaudio` / HuggingFace + bespoke matplotlib notebooks + W&B/MLflow + ad-hoc dataset browsers. Praat used when *acoustic measurement* is genuinely needed. The unmet need is a real GUI for inspecting model outputs at scale — attention/alignment for TTS, embeddings (UMAP) for speaker/dialect models, error analysis (WER by acoustic-context bucket), dataset cleaning, side-by-side audio diff. None of this lives in one tool.

#### 3. Clinical voice (SLPs, ENT, voice therapists)

**Today.** CSL / MDVP, Visi-Pitch, Sona-Speech, VoxMetria, Lingwaves — all proprietary. Some clinical Praat scripts (VoiceEvalU8, AVQI, ABI). Stroboscopy and EGG hardware lives outside the analysis app.

**Features relied on.** Standardized parameter sets with normative ranges (MDVP's 33, AVQI, ABI, CPP-based scores); CAPE-V / Rainbow Passage protocols; sustained-vowel + connected-speech workflows; phonetogram / Voice Range Profile; calibrated SPL (calibrated mic); EGG sync; report generation with EHR fields; pre/post-treatment longitudinal comparison.

**Pain points.** CSL is expensive, Windows-only, locked. Praat parameter values aren't directly interchangeable with MDVP. Longitudinal patient tracking is manual. Report generation is manual. Regulatory/validation friction makes switching tools risky.

**Ideal.** Open, documented MDVP-equivalent algorithms; AVQI/ABI validated against literature; calibrated-SPL support via known-mic profiles; patient-as-first-class-object with longitudinal viz; protocol-driven workflows (load protocol → guided recording → auto-report); HL7 / FHIR export; teletherapy via browser delivery.

#### 4. Voice coaches and trans-voice training

**Today.** Praat (the trans-voice community has popularized it heavily), InFormant (the strongest real-time tool), mobile apps (Voice Tools, Eva, ChristellaVoice, Vocaberry). Singing coaches additionally use VoceVista, Sygyt's Overtone Analyzer.

**Features relied on.** Real-time F0; real-time F1/F2 (resonance targets); user-defined target zones (not population norms); pitch range / register exploration; recording-and-compare against a model; visual feedback during practice; phonetogram; progress over weeks.

**Pain points.** Praat is intimidating and not designed for this audience. Mobile apps are paywalled and analytically shallow. Clinical norms don't fit (voice modification is not pathology). Singing-specific needs (vibrato, register transitions, vowel modification across registers) aren't standard in any one tool.

**Ideal.** Mobile-first real-time; user-definable target zones with own-voice baselines; practice sessions with auto-tagged recordings and longitudinal viz; singing modes (semitones/cents, vibrato analysis, formant tuning); privacy-respecting (local by default, no cloud uploads); onboarding that doesn't require linguistics training.

#### 5. L2 / pronunciation learners and tutors

**Today.** Praat (advanced learners and pronunciation researchers); commercial apps (ELSA Speak, Speechling, Sounds, Forvo). Most commercial offerings are ASR-feedback driven.

**Features relied on.** Compare learner to native model; minimal pairs; IPA display; stress/intonation viz; spaced repetition.

**Pain points.** ASR-based feedback says "wrong" without saying *why*. Native models are usually one speaker, not a distribution. No phonetic-feature-level feedback. Progress tracking is shallow.

**Ideal.** Feedback grounded in measured acoustics (your F2 trajectory vs target distribution), not just ASR pass/fail; native-speaker reference *distributions*; minimal pairs with measured targets; mobile delivery; pluggable for arbitrary L1→L2 pairs.

#### 6. Field linguists / language documenters

**Today.** ELAN (heavily) + Praat + SayMore + FieldWorks (FLEx) + Toolbox/Shoebox + Phonology Assistant + Audacity. Often offline, often on modest hardware in remote conditions.

**Features relied on.** Excellent IPA input (palette, feature search, novel diacritics); multi-tier annotation with glossing layers (audio → IPA → morphological gloss → free translation → metadata); lexicon-aware annotation; time-aligned video + audio + transcript; long-session recording reliability; archival-format export (DELAMAN, ELAR, PARADISEC).

**Pain points.** ELAN is powerful but UX-heavy. Praat is annotation-poor (one TextGrid, no lexicon link). Nothing spans recording → transcription → lexicon → publication cleanly. IPA keyboarding is universally bad. Field hardware constraints (battery, low power, sometimes offline for weeks).

**Ideal.** Lightweight, cross-platform, offline-capable. Excellent IPA input. Lexicon-aware annotation (this word's gloss persists across files). Tier-rich annotation matching ELAN's expressiveness. Crash-resilient recording with autosave. Direct export to archival formats. Phone as a recording companion that syncs back to the desktop project.

#### 7. Forensic phoneticians

**Today.** Praat + proprietary speaker-comparison tools (Batvox, VOCALISE / iVOCALISE) + bespoke likelihood-ratio scripts. Reference populations are scarce and often proprietary.

**Features relied on.** Speaker comparison with likelihood-ratio frameworks; long-term formant distributions; F0 statistics, articulation rate; defensible/auditable methodology; chain-of-custody; population reference data.

**Pain points.** Tools are niche and proprietary. Reproducibility / audit trail for court is hand-rolled. Reference population data is fragmented.

**Ideal.** Built-in audit logging (every analysis step recorded); standard LR frameworks; community-shared reference population corpora; reproducible "raw audio → court-ready report" pipelines.

#### 8. Experimental psycholinguists / lab experimenters

**Today.** PsychoPy (Python, lab), E-Prime (proprietary Windows), OpenSesame, PCIbex / Gorilla / jsPsych (browser), LabVanced. Praat's Demo Window for quick perception probes. Recording is one tool, analysis is another.

**Features relied on.** Millisecond-accurate stimulus timing; randomization / counterbalancing; audio playback + response (keyboard / button / mouse / voice); production tasks (prompt → record participant); trial-level data export; multi-modal stimuli; browser deployment for Prolific/MTurk; informed-consent flows.

**Pain points.** PsychoPy is code-heavy for non-programmers. Web platforms have audio timing issues. Recording in one tool, analyzing in another, with no shared project model. Browser recording quality varies wildly.

**Ideal.** Visual experiment builder (stimulus list → trial structure → response collection) that shares a project with the analysis layer; trial metadata flows directly into downstream analyses; auto-process recordings (force-align, extract features, tag with trial info); browser-deliverable version for crowdsourced participants; open / replicable experiment artifacts.

#### (Adjacent) Bioacoustics

Worth flagging because Parselmouth's bioacoustics paper showed real adoption crossover. Tools: Raven Pro (Cornell), Avisoft, BatSound. Needs: configurable frequency ranges (ultrasonic / infrasonic), long-recording event detection, multichannel arrays for localization, species-classifier plugins. Probably not a primary audience but should not be designed *out* — frequency-range assumptions in particular are easy to bake in by accident.

### Cross-cutting patterns

These are the patterns that recur across groups and should drive architecture, not just feature lists.

**A. Real-time analysis is universally undersupported.** Voice coaches, L2 learners, clinicians during therapy, and experimenters during pilot recording all want live feedback. Only InFormant takes this seriously. Praat's real-time facilities are afterthoughts. A first-class low-latency analysis path (live spectrogram, F0, formants, configurable feature overlays) would serve four groups at once.

**B. Mobile is a structural gap, not a nice-to-have.** Voice coaches, L2 learners, field linguists (recording companion), clinical teletherapy, crowdsourced experiment participants — all benefit from a phone form factor. Today's mobile speech apps are commercial walled gardens with weak underlying analysis. A research-grade engine deliverable to phone is genuinely novel and aligns with the desktop-primary / mobile-companion stance from question 3.

**C. Reference distributions are everyone's hidden problem.** Voice coaches want target zones; L2 learners want native-speaker distributions; clinicians want age/sex-normed ranges; forensic wants population statistics; phoneticians constantly cite Hillenbrand / Peterson-Barney / Hagiwara. Nobody has them in one place. A reference-distribution layer — bake in known corpora, make it easy to publish new ones, treat distributions as a first-class data type the GUI can render against — would be a transverse value-add across at least five groups.

**D. The "record → annotate → analyze" pipeline is fractured for every group.** Praat handles annotate + analyze but is weak at record; PsychoPy handles record + present but not analyze; ELAN annotates but doesn't analyze; field tools cover record + annotate but not analyze. Every group stitches. The opportunity is not "better analyzer" — it's *one continuous project model* that holds recordings, annotations, analyses, and derived data, with sensible defaults per workflow.

**E. The corpus / project model itself is missing in Praat.** EMU-SDMS is the only mainstream tool that takes this seriously. Forensic case files, clinical patient longitudinal data, field-linguistics sessions, ML datasets, experimental trial logs — every group wants "this is the project; here are its files and metadata." This is probably the single most consequential architectural decision: design corpus-first, not file-first.

**F. Annotation is the secret center of gravity.** TextGrid (Praat), EAF (ELAN), lexicon entries (FLEx), trial logs (PsychoPy), patient charts (clinical) are all annotation schemas under different names. A flexible tier + attribute model — interval/point/sequence/categorical/numeric/embedding-vector tiers — could unify all of them. EMU's hierarchical model is the closest prior art; ELAN's tier model is the most practically expressive.

**G. ML features are a unifying "new signal type."** Every group has a use:

- phoneticians → SSL embeddings as analogues to formants
- AI engineers → native viz of model internals
- clinicians → diagnostic embeddings (Parkinson's, dysphonia screening)
- voice coaches → pretrained voice-quality classifiers
- L2 learners → pronunciation scoring via SSL
- forensic → speaker embeddings (already used)
- bioacoustics → species classifiers
- experimenters → automatic post-trial feature extraction

Treating model outputs (logits, attention, hidden states, embeddings) as *just another time-aligned signal* alongside F0 and formants is a structural decision that pays off in seven directions.

**H. Scripting is needed by power users in every group, but Python's pull is decisive.** Phoneticians (R/Python now), AI engineers (Python first), psycholinguists (PsychoPy *is* Python), field linguists (some Python), forensic (some Python). R has a strong foothold in EMU-SDMS and in stats post-analysis, but no other group is R-native. The right shape is probably *the app exposes a Python API* — whether or not the app is itself written in Python.

**I. Experimentation is a bonus capability with high reuse.** Stimulus presentation + response collection overlaps with field elicitation, clinical protocols (CAPE-V), L2 drills, perception experiments, and crowdsourced data collection. The shared abstraction is "scripted protocol = sequence of (present stimulus, collect response, record participant, tag with metadata)." Building this once serves five groups.

**J. Privacy / on-device is non-negotiable for several groups.** Clinical (HIPAA), voice coaches (sensitive voice modification), forensic (chain of custody), field linguistics (community data sovereignty). Cloud-first is a non-starter. Architecturally: local-first, sync optional, encryption at rest.

### Architectural implications (provisional)

Translating the patterns above into stance, before any tech-stack discussion:

1. **Corpus-first, not file-first.** A "project" is the primary noun. Recordings, annotations, analyses, and derived signals live inside it. EMU-SDMS as prior art.
2. **Tiered annotation as a unifying data model.** Interval / point / continuous / categorical / numeric / embedding tiers. ELAN + EMU as prior art.
3. **Signal types are pluggable.** F0, formants, intensity, mel-spec, MFCC, wav2vec layer-N embedding, custom feature — all the same kind of object to the GUI. ML features = first-class.
4. **Reference distributions are first-class.** Compare any signal against a stored reference; ship some, let users publish more.
5. **Real-time path is designed in, not bolted on.** Live analysis as a peer to offline analysis.
6. **Desktop primary; phone is a thin client.** Phone records and runs the live-feedback path, syncs to the desktop project. Phone is not where corpora are managed.
7. **Browser embed is a deferred third surface.** Useful for crowdsourced experiments and teletherapy; not the primary delivery path.
8. **Scriptable via Python API,** independent of host language. The app embeds or exposes Python; the app itself need not be written in Python.
9. **Experimentation runner is a built-in module,** not a separate product. Shared protocol abstraction across elicitation / perception / clinical-protocol / drill.
10. **Local-first; encryption at rest; explicit opt-in for any network feature.**

### Open questions surfaced this session

- **Reference-distribution governance.** If we want a community-shareable distribution store, who curates? Version? License?
- **Articulatory channels.** Do we treat ultrasound video / EMA / palatography as in-scope multichannel signals, or push them to plugins? Phonetician-side feature with no overlap to other groups except via "signals are pluggable."
- **Clinical regulatory stance.** Build with potential FDA / CE pathway in mind, or stay explicitly "research use only" and let a downstream fork pursue clearance?
- **Experiment-runner ambitions.** Match PsychoPy's depth, or stay at "good enough for production/perception probes" (i.e. Demo Window+) and let serious experimenters use PsychoPy?
- **Host-language candidates** for the app itself, given the Python-API decision: Rust (Tauri), C++ (Qt), Kotlin Multiplatform, Swift+Kotlin twin native, Electron + native engine, Flutter + native engine. Deferred to a future entry.

---

## 2026-05-15 — Landscape survey: existing Praat alternatives

First entry. Project is greenfield; no code, language, or architecture chosen yet. Goal of this session: survey the current state of phonetics / speech-science software to understand where Praat sits, what alternatives exist, and where the real gaps are.

### Findings, grouped by what each tool is actually for

The field is fragmented. Most practitioners glue 3–4 tools together rather than using one. Nothing is currently positioned as a true Praat successor.

#### 1. Direct GUI alternatives (Praat-shaped tools)

- **Praat** (Boersma & Weenink, UvA) — still the incumbent, actively released (6.4.x series). Architecture, scripting language, and Motif-derived GUI are essentially frozen in shape. Well-known pain points unchanged: no undo, no autosave, idiosyncratic scripting (no inline `#` comments, whitespace-sensitive, opaque error messages), no native package/extension story.
- **Phonometrica** (GPL, C++/Qt) — the most ambitious "redo Praat properly" attempt. Corpus-first design, Lua scripting, modern Qt UI. Small community; intermittent development.
- **InFormant** (open source, `in-formant/in-formant`) — modern, real-time-focused. Spectrogram, formant tracking, glottal inverse filtering. Strong live-analysis UX (popular in trans-voice / vocal coaching). Not a corpus/annotation tool.
- **VoiceLab** (MIT, `Voice-Lab/VoiceLab`) — GUI wrapper over Parselmouth for batch automated voice analysis. Reproducibility-focused; loads many files, outputs CSV of pitch/jitter/shimmer/formants. Not interactive in the Praat sense.
- **Speech Analyzer** (SIL) — free, Windows-only, oriented toward field linguistics.
- **WaveSurfer** (KTH) — historically the main alternative; effectively unmaintained.
- **xkl** — modernized GTK port of Klatt's classic analysis/synthesis tool. Niche but technically interesting.
- **TrackDraw** — research toy for Klatt-synthesizer formant track drawing.

#### 2. R ecosystem — EMU-SDMS

The most complete *non-Praat* ecosystem. Either prior art or an integration target.

- **emuDB** — corpus/database format (audio + hierarchical annotations).
- **wrassp** — R wrapper around `libassp` (formants, F0, RMS, spectral, zero-crossings, filtering). Replaces Praat's signal-processing engine.
- **emuR** — query and analysis in R.
- **EMU-webApp** — browser-based annotation/inspection front-end; talks to emuR over websocket.

Closest thing to "Praat done as a database + R + web stack." Weakness: GUI is annotation-focused, audience is mostly R users.

#### 3. Python ecosystem

- **Parselmouth** — Python bindings calling Praat's actual C++ code. Now the de facto way to script Praat analyses; cited heavily in bioacoustics and clinical voice papers. No GUI — library only.
- **librosa**, **torchaudio**, **opensmile**, **SPTK** — general MIR/audio-ML libraries; partial overlap with phonetics needs.

#### 4. Forced alignment (used alongside Praat)

- **Montreal Forced Aligner (MFA)** — Kaldi GMM-HMM; current standard. Outputs TextGrids straight into Praat. Active 2025 work on low-resource fine-tuning.
- **WebMAUS / BAS** — Munich's web service; integrates with ELAN.
- **Charsiu** — transformer-based, newer alternative.

#### 5. Annotation-centric

- **ELAN** (Max Planck) — multimedia tier-based annotation, gold standard for video+audio corpora. Not an analysis tool.
- **Phon** — child phonology corpora.
- **Phonology Assistant** (SIL) — phonetic data charting.

#### 6. Proprietary clinical

- **CSL / MDVP** (Pentax / KayPENTAX) — clinical voice labs; MDVP module is why hospitals still buy it. Expensive, locked.
- **Dr. Speech**, **TF32** — niche clinical tools.

#### 7. General audio (used as crutches)

- **Sonic Visualiser** — extensible via Vamp plugins, good general spectrogram inspection, no phonetics-specific affordances.
- **Audacity**, **Ocenaudio** — general audio editing.

### Read on the gap

The field has bifurcated:

- **GUI inspection** → still Praat (because nothing else is full-featured); InFormant taking real-time voice work.
- **Reproducible analysis** → Parselmouth (Python) or wrassp/emuR (R) — both library-shaped, neither GUI.
- **Corpus annotation** → ELAN + MFA + Praat-as-viewer stitched together.
- **Clinical** → proprietary CSL.

The real gap a successor could fill: a single tool that is **interactive like Praat**, **scriptable like Parselmouth**, **corpus-aware like EMU-SDMS**, and **ML-native** — i.e. wav2vec2 / HuBERT / Whisper features as first-class signal types alongside formants and F0. Phonometrica reaches for some of this but hasn't achieved escape velocity.

### Open questions to resolve next

1. **Target users.** Research phoneticians? Clinicians? L2 learners? Voice coaches? Field linguists? The answer drives nearly everything else.
2. **Scope.** All-in-one (Praat-style) vs. focused (pick one of: analysis, annotation, corpus management, ML-feature inspection).
3. **Architecture stance.** Desktop GUI (Qt / Tauri / native), web-first (browser app + local engine), or library-with-thin-GUI?
4. **Scripting story.** Embedded Python? Lua? A DSL? None?
5. **Praat-compat surface.** Read/write TextGrids — table stakes. Run Praat scripts — explicit non-goal? Or interop layer?
6. **ML integration.** First-class transformer-based features (SSL embeddings, alignment, ASR) vs. classical-DSP-only.

### Sources consulted

- A Phonetician's Software Toolkit — Will Styler: https://wstyler.ucsd.edu/posts/phoneticians_software.html
- Praat: https://www.fon.hum.uva.nl/praat/
- Phonometrica: https://github.com/phonometrica/phonometrica
- InFormant: https://in-formant.app/ and https://github.com/in-formant/in-formant
- VoiceLab: https://voice-lab.github.io/VoiceLab/ ; Interspeech 2022 paper https://www.isca-archive.org/interspeech_2022/feinberg22_interspeech.pdf
- EMU-SDMS: https://ips-lmu.github.io/EMU.html ; emuR https://github.com/IPS-LMU/emuR ; CSL 2017 paper https://www.phonetik.uni-muenchen.de/~jmh/papers/emucsl.pdf
- Parselmouth for bioacoustics (2023): https://www.tandfonline.com/doi/full/10.1080/09524622.2023.2259327
- Montreal Forced Aligner: https://montrealcorpustools.github.io/Montreal-Forced-Aligner/
- SIL Speech Analyzer: https://software.sil.org/speech-analyzer/ ; SIL Phonology Assistant: https://software.sil.org/phonologyassistant/
- Evaluation of Praat (Bamberg): https://www.uni-bamberg.de/fileadmin/eng-ling/fs/Chapter_13/324EvaluationofPraat.html
- xkl modernization: https://www.researchgate.net/publication/373229441_xkl_A_legacy_software_for_detailed_acoustic_analysis_of_speech_made_modern
