# DEVLOG

A running log of research, decisions, and development for the SpeechAnalysisTool project — a planned next-generation phonetics / speech-science tool.

Newest entries at the top. Each entry is dated `YYYY-MM-DD` and tagged with a short topic. This file holds the **current month**; earlier months are rotated into [`devlog/`](devlog/) (index at the bottom).

---

## 2026-06-19 — PipeWire audio playback fix (retry + device fallback)

Audio playback failed on first attempt with "device is no longer available" on
systems using PipeWire's ALSA emulation. Root cause: PipeWire can return stale
device handles that fail when queried for their config — the device *exists* in
the enumeration, but its state is out of sync with the actual audio graph.

**The fix (`d6d73c2`).** Two-part workaround in `playback.rs`:

1. **Retry logic**: if starting a stream fails, pause 50 ms and retry once.
   PipeWire often settles between attempts. The retry is internal to
   `start_span`; callers see either success or the final error.

2. **Device fallback with HDMI deprioritization**: if the default device fails,
   enumerate all output devices and try the first one whose
   `default_output_config()` succeeds. Devices are sorted so non-HDMI ones come
   first (checked via `DeviceDescription::interface_type() == Hdmi` or a
   name-contains-"hdmi" fallback), since HDMI outputs are rarely the intended
   playback target when a system also has speakers/headphones.

The actual fix on the user's machine was a missing ALSA→PipeWire routing symlink
(`/etc/alsa/conf.d/99-pipewire-default.conf`), but the retry+fallback makes the
app more resilient to transient PipeWire state mismatches generally. Also fixed
a cpal 0.17 deprecation warning: `device.name()` → `device.description().name()`.

---

## 2026-06-19 — Fresh-start crash fix + folder import (cross-machine debugging)

Two app changes out of a debugging session on a second machine.

**The fix (`f789b92`).** The app panicked on first launch on a clean machine —
no `~/.config` state yet. Cause: `PersistedState` derived `Default`, so every
field fell to its type default, and `ui_scale: f32` became `0.0`. We only ever
set `ui_scale` from `default_ui_scale()` (`1.0`) via serde's `#[serde(default =
…)]` *when deserializing existing state*; a from-scratch `Default::default()`
bypassed that and handed egui a zero scale, which it panics on. Replaced the
derive with a hand-written `Default` that calls `default_ui_scale()` for that one
field (and spells out the rest explicitly so the next field added to the struct
can't silently inherit a bad zero). Lesson logged for the [[reference_wslg_gui_debugging]]
pile: a `Default` derived over a struct that serde otherwise back-fills is a
latent first-run footgun — the defaults you see at runtime aren't the ones a
cold start uses.

**The feature (`42ce2b9`).** Added **File ▸ Add Directory…** — pick a folder,
register every `.wav` in it as a bundle. Extension match is case-insensitive;
files are sorted alphabetically before import so the bundle list is in a
predictable order; each file still goes through `add_bundle_guarded` (so the
large-file probe/split guard applies per file). Empty folder → an error toast;
otherwise an info toast with the count. Enabled only when a project is open
(disabled-hover explains why), mirroring Add Bundle…. Engine/Python already
import a folder by looping `add_bundle`, so this is the GUI surface of an
existing capability rather than a new one.

## 2026-06-15 — Export + import annotations as CSV / JSON (three surfaces)

The backlog's "export annotation data to CSV / JSON" — built as a round-trip
(the user asked for import too, the natural pairing, and it mirrors how
TextGrid / EAF already do both directions). Engine + Python + GUI in one slice.

**Two shapes for two audiences** (new `engine/src/io/tabular.rs`, pure +
unit-tested):

- **CSV** = one *tidy* long table, one row per annotation across all sparse
  tiers (the shape pandas / polars / R want). The column set is the union over
  the three sparse kinds (interval / point / reference); cells that don't apply
  to a row's kind are left empty (`time_seconds` on an interval, `start_seconds`
  on a reference). RFC 4180 quoting, hand-rolled — no new crate dep (the project
  is deliberately dep-conservative; `serde_json` covers JSON, a ~20-line escaper
  covers CSV).
- **JSON** = a *faithful* nested document: `{bundle, tiers:[…]}`, each tier
  carrying only its native rows, so the per-tier structure CSV flattens away is
  recoverable. `extra` (DB-stored JSON-as-TEXT) is embedded as parsed JSON, not
  an escaped string.

Dense tiers (`continuous_*` / `categorical_sampled`) are skipped in both,
matching the TextGrid / EAF exporters — their samples live in Parquet sidecars
(`Project.query`).

**Import** is the inverse (`parse_csv` / `parse_json`, also pure-tested incl. a
hand-rolled RFC-4180 *reader* that handles quoted commas / doubled quotes /
embedded newlines). CSV columns are matched **by header name**, so reordered or
extra columns are tolerated; rows group into tiers by `(tier_name, tier_type)`.
**v1 limits (documented):** only interval + point tiers import (reference rows
are skipped — their `(target_kind, target_id)` is project-local; dense isn't
sparse-annotation data). The source-project / rubric-bound columns — `status`,
`parent_annotation_id`, `processing_run_id` — are dropped; times, `label`,
`note`, `extra` are honoured. Each import records a `processing_run`
(`sadda.io.{csv,json}.import`) for provenance, like the TextGrid / EAF importers.

Surfaces:
- **Engine**: `Project::export_csv/export_json/import_csv/import_json`
  `(bundle_id, path, tier_ids?)`, sharing a `gather_export_tiers` /
  `create_imported_tiers` back end with the existing exporters' signature.
- **Python**: same four methods on `Project` (pyo3); stubs regenerated.
- **GUI**: "CSV (annotations)…" / "JSON (annotations)…" in the File ▸ Import and
  ▸ Export submenus, reusing `suggest_export_path` + the rfd pick-file pattern.

Tests: 9 pure unit tests in `tabular.rs` (CSV escaping, the RFC-4180 reader,
column-by-name matching, export→parse round-trip, JSON extra-embedding), 3
engine DB round-trip integration tests (`annotation_export_import.rs`, incl. a
comma/quote/newline label torture case + the `tier_ids` filter), and 3 Python
round-trip tests. Full gate green (the only stop was the known stubs-vs-HEAD
pre-commit false positive — stubs are current; backlog item still open).

## 2026-06-15 — Hard-gate releases on CI (reusable gate workflow)

`v0.4.0-app` shipped broken — twice in one cycle (the debug-only egui API, then
re-cut) — because the release workflows never *ran* the gate; they only trusted
"main was green when we tagged." Tagging an unverified commit could publish to
PyPI / cut a GitHub Release with nothing standing in the way. Closed that gap.

Restructure (the backlog's "GitHub-Release-driven" item):

- Extracted CI's full `test` job verbatim into a **reusable workflow**
  (`.github/workflows/gate.yml`, `on: workflow_call`): fmt · clippy · debug build
  · `cargo check --release -p sadda-app` · `cargo test` · download-feature
  clippy+test · stub-drift · pytest. One definition, no copy.
- `ci.yml` is now a thin caller (`uses: ./.github/workflows/gate.yml`).
- **Both release workflows call the same gate and `publish` `needs:` it:**
  `release.yml` → `publish.needs: [gate, build-wheels, build-sdist]`;
  `app-release.yml` → `publish.needs: [gate, build]`. Gate runs in parallel with
  the builds; if it fails, publish is skipped and **nothing is uploaded** even
  though the tag exists.

Why reusable (not copy-paste the steps into each release file): the recurring
failure mode in this project is *drift* — the gate and its mirror disagreeing
(cf. the `cargo fmt` omission that left CI silently red for a day). One
`workflow_call` definition makes "CI is green" and "safe to publish" the same
checks by construction. `just gate` remains the local mirror; the justfile header
now points at `gate.yml`.

Note: the per-OS *builds* aren't gated (only `publish` is), so a broken commit
still burns matrix build minutes producing artifacts that never publish — the
cheap, request-matching choice. `main` is unprotected, so no required-check-name
rule needed updating despite the now-nested check context. Config-only; the gate
itself is unchanged, so this couldn't regress a green tree. Validated by YAML
parse + dependency-graph review; first real exercise is the next tag.

## 2026-06-04 — Live recording now populates the main view (waveform + spectrogram + measure tracks)

Recording previously showed only an elapsed timer + a dB-FS level meter in the
record window — no live visual. The user expected (Praat-style) to *watch* the
waveform and spectrogram fill in as they speak. Built it: while recording, the
**main view's own lanes** render the in-progress capture in a scrolling ~10 s
window (ending at the live edge), then revert to the selected bundle on stop.

Three slices, all app-side (no engine change):

- **Sample tap (waveform).** The engine already had a raw-sample ring feeding the
  WAV writer + DSP, but nothing exposed samples to the GUI. Rather than touch the
  engine, the cpal callback now *tees* each sample into a **second, app-owned
  ring** (`spawn_cpal_input` gained a `display_tap`). A new `LiveView` drains it
  each frame, downmixes interleaved → mono, and accumulates into an `Arc<Vec<f32>>`.
  Key property: a UI stall overflows only the *display* ring → a momentary
  waveform glitch, **never** a dropped sample in the saved file. The waveform pane
  synthesizes an `EnvelopeCache` (sentinel `bundle_id = -1`) over that buffer so
  the existing per-visible-range re-bucketer draws it unchanged; the timeline is
  pinned to `[dur − 10s, dur]` so the window scrolls. `active_envelope` is left
  untouched, so the prior bundle reappears the instant recording stops.
- **Live spectrogram.** The async P2 spectrogram path is keyed on
  `(bundle_id, config)` and `poll_analysis` discards results whose bundle no longer
  matches — so the `-1` sentinel couldn't reuse it directly. Added a dedicated
  throttled path: `rebuild_live_spectrogram_if_stale` dispatches a worker STFT of
  the capture-so-far at ≈5/s (one build in flight at a time), delivered via a new
  `AnalysisResult::LiveSpectrogram` and installed into `live_spectrogram`. Worker
  thread, so the UI never blocks. The spectrogram pane already positions its
  texture over `[0, duration]` and crops to the view, so the scrolling window falls
  out for free.
- **Live measure tracks.** The engine was *already* streaming f0 / intensity /
  formant frames over the result rings — and the dialog was draining-and-discarding
  them every frame. Now they accumulate into a `live_tracks` cache (the live frame
  types are field-identical to the dsp `PitchFrame` / `FormantFrame`; intensity
  back-computes `rms` from `db_fs`). The four measure lanes draw `live_tracks` while
  recording. No live VAD (not streamed) — that lane stays empty.

Borrow-checker note worth recording: a `current_tracks(&self)` helper returning the
live-or-active cache borrowed *all* of `self`, colliding with the lane panes' later
`self.timeline` mutation. Inlining the `if recording { … }` pick makes the borrow
field-specific (`live_tracks` / `active_tracks`), disjoint from `timeline`. Same
disjoint-field reasoning lets the per-frame poll write `self.live_*` while `handle`
holds `self.record_dialog`.

Tested: pure drain/downmix logic (mono passthrough, stereo downmix, partial-frame
carry across drains, duration) as `live_view_tests`. Live audio itself is a manual
GUI check (no mic in CI). `just gate` green.

**v1 limits (deferred):** the view auto-follows the live edge (manual scroll/zoom
won't stick mid-record); the take isn't reviewable in the Stopped state (clears on
stop, full take shown after Save); the live spectrogram re-STFTs the whole growing
capture each tick (fine for typical takes, heavier — but non-blocking — on long ones).

## 2026-06-03 — Fix: 0.4.0 app-release broke on a debug-only egui API + gate now release-checks

The `v0.4.0-app` release failed to build on all three platforms:
`error[E0599]: no method set_debug_on_hover on &egui::Context`. That method is
`debug_assertions`-gated in egui (present in debug, absent in release), and the
`just gate` / CI gate build only in **debug** — so a release-only compile error
sailed straight through to the release workflow's `cargo build --release`.

- Gate the `SADDA_DEBUG` hover-overlay call behind `#[cfg(debug_assertions)]` (a
  dev-build aid; egui's overlay painting is debug-only anyway). App builds release.
- **Gate gap closed:** added `cargo check --release -p sadda-app` to the justfile
  gate and `ci.yml`, so debug-vs-release API drift is caught on every push, not
  at release time. (`check`, not `build`, to stay quick.)

App-only; the 0.4.0 **PyPI wheel is unaffected** (it builds `sadda-python`, not
the app). The wheel track published fine; the app binaries get re-cut from the fix.

## 2026-06-03 — Fix: VAD returned ~0 for everyone (missing Silero context window)

The big one in the VAD-debugging thread. `sadda.ml.vad()` / the GUI VAD lane
returned ~0 speech probability on **all** audio. Bisected via raw onnxruntime: the
bundled model is byte-identical to the official Silero (sha256 `1a153a22…`), and
the **official model also returns ~0** when fed bare 512-sample windows — so it
was never a sadda-vs-model or audio problem (the test clip was confirmed real
speech: 83% energy in 300–3400 Hz, no DC offset).

Root cause: the Silero **2024** model needs **64 "context" samples** (the tail of
the previous window) prepended to each 512-sample window — the model input is
`[1, 576]`, not `[1, 512]`. sadda fed bare 512, so the model never saw the
lookahead it was trained with → flat ~0. The only VAD test ran on *silence*
(~0 either way), so it never caught it.

Fix (`engine/src/ml.rs::vad`): carry a 64-sample `context` across windows and feed
`context ++ window`. Verified end-to-end through sadda on real speech: **max
0.003 → 1.000**, mean 0.640, 226/360 windows detected. Extracted `vad_model_input`
+ a unit test guarding the 576-sample input. (Backlogged: a real-speech ORT-gated
integration test — the silence-only test was the gap.)

Found while a collaborator stress-tested ML VAD — the same session surfaced the
wheel-missing-model packaging gap (fixed in the entry below).

## 2026-06-03 — Fix: bundled Silero VAD now ships in the wheel

The PyPI wheel didn't actually include the bundled Silero VAD, so
`sadda.ml.vad()` failed with "bundled Silero VAD not found" for any pip user
without the repo checked out + `SADDA_MODELS_BUNDLED` set — the engine's
`bundled_model_dir` searches that env var, then next-to-exe, then a *compile-time*
repo path, none of which resolve for a pip wheel. (The GUI app sidesteps it: a dev
build finds the repo's `models-bundled/`.) Found while testing VAD end-to-end.

Fix (mirrors the existing ORT auto-discovery in `sadda.ml`):
- Ship the model as **package data**: `python/sadda/_bundled/silero-vad/`
  {`model.toml`, `silero_vad.onnx`, `LICENSE`}. **Verified it lands in the built
  wheel** as `sadda/_bundled/silero-vad/…`.
- `ml/__init__.py` gains `_discover_bundled_models()` and sets
  `SADDA_MODELS_BUNDLED` to the package dir at import (never overriding a user
  value), so `vad_bundled` finds it.
- Tests: ships-with-package + discovery-sets-env + a **drift guard** asserting the
  in-package copy stays byte-identical to the repo's canonical `models-bundled/`
  (the copy the engine / GUI build uses). The duplication is deliberate + guarded.

App/wheel-only; gate green (+3 tests). (Separately noted but NOT the cause of the
near-zero VAD output under investigation — that's the recording/model question.)

## 2026-06-02 — Ctrl-snap boundary reuse (Slice 3c — scan ergonomics COMPLETE)

Holding **Ctrl** while defining/moving a selection edge snaps it to the nearest
existing interval boundary across the active interval tiers — usable mid-drag or on
a click. Completes the scan-ergonomics feature (Slices 1–3).

- Pure `snap_to_nearest(t, boundaries, max_dist)` (tested); `active_interval_boundaries`
  gathers start+end of every interval on active interval tiers.
- `apply_lane_selection_drag` gains `ctrl_held` + `boundaries`; snaps the drag anchor /
  drag end / click when Ctrl is held. Wired on the waveform / spectrogram / heatmap
  panes (the `measure_lane` free fn has no project handle → no snap there; documented
  cut). Ctrl = egui COMMAND; always snaps to nearest (Ctrl means "reuse a boundary").

App-only; +1 test. **Scan-ergonomics feature COMPLETE: Slice 1 (span playback) → 2
(multi-active tiers + digits) → 3a (click=point) → 3b (Enter-commit + conflict
resolution) → 3c (Ctrl-snap).** All on main, each slice green.

## 2026-06-02 — Enter-to-commit + conflict resolution (Slice 3b: scan ergonomics)

Bare **Enter** (when not text-editing / no modal / no focused widget) commits the
current selection to all active tiers of the matching type: a span → intervals on
active interval tiers, a point → points on active point tiers.

- Pre-insert conflict detection (pure, tested): `overlapping_intervals` (positive
  overlap only — touching boundaries are allowed) + `colliding_points` (within
  `POINT_COLLISION_TOL_SECONDS` = 1 ms).
- `enter_commit`: commits the non-conflicting tiers immediately; queues conflicting
  tiers into `pending_commit`.
- Resolution prompt (`render_pending_commit`): per-tier **Skip / Replace** + Skip-all /
  Replace-all; **Commit** applies, **Cancel** discards. Replace = delete the
  conflicting existing annotation(s) (`delete_interval`/`delete_point`) then add the
  new one (`apply_pending_commit`).

App-only; +2 tests (conflict detection). The Enter/modal flow is GUI-driven (not
unit-tested) — worth an end-to-end check in the app. Flags (tweakable): point
collision tol = 1 ms; Enter is guarded on "no focused widget", so a rare
open-but-unfocused modal could still see Enter. Next: 3c (Ctrl-snap boundary reuse).

## 2026-06-02 — Click places a point selection (Slice 3a: scan ergonomics)

Per the locked Slice 3 decisions, a plain click on a signal pane now drops a
zero-width **selection point** at that time (+ moves the playhead) instead of
clearing the selection — so a time can be committed as a point. Drag still makes
a span; clicking an annotation in a tier lane still selects it (separate path).

- `TimelineState::set_point_selection(t)` (selection = `(t, t)`);
  `apply_lane_selection_drag`'s click branch uses it.
- The selection now drives the commit **type**: a point commits to active **point**
  tiers (single point), a span to active **interval** tiers. The strip button + label
  adapt (Add point / Add interval); `commit_selection_to_tier` no longer adds two
  points at the span edges.
- Rendering: a point selection shows as a single vertical line in plot lanes
  (coincident band edges) and tier lanes (`draw_selection_band_rect`). Header reads
  `point t s`. Space falls back to the view for a point selection.
- Pure helper `selection_is_point` + tests.

App-only; +2 tests. Next: 3b (Enter-to-commit + conflict resolution), 3c (Ctrl-snap).

## 2026-06-02 — Multi-active tiers + digit activation (Slice 2: scan ergonomics)

Second slice. The single `active_tier_id: Option<i64>` becomes a **set**
`active_tier_ids: Vec<i64>` — several annotation tiers can be active at once.

- **Digit keys**: bare **1–9** selects the tier at that lane position (top = 1),
  replacing the set; **Shift+digit** toggles it in/out; **0** clears.
  `set_active_by_position` resolves the tier id under a scoped `&project` borrow,
  then mutates the set. Clicking a tier name still toggles it (now into the set).
- **UI**: all active lanes highlighted (`SELECTION_EDGE`); the strip status lists
  the active tiers (pure `format_active_tiers_status`, tested) or a hint; the
  Add-selection button commits to **all** active interval/point tiers (label
  adapts: Add interval / Add points / Add to active tiers). Delete-tier prunes
  the set.

Groundwork for Slice 3 (Enter-to-commit + conflict resolution). App-only; +2
tests. Notes (tweakable): Shift+digit chosen for the toggle; the multi-tier
commit button has **no conflict checks yet** — that lands in Slice 3.

## 2026-06-02 — Praat-style span playback (Slice 1: scanning ergonomics)

First slice of "make scanning & annotating enjoyable" (plan + Q&A 2026-06-02;
plan file `fuzzy-shimmying-tulip.md`). The playback engine — which only played
from a start point **to the end of the file** — now plays an arbitrary **span**
with **loop** (silent
inter-repetition gap) and **pause/resume**, all in a pure, real-time-safe
`next_mono_sample` state machine (paused → inter-loop pause → span body →
end/loop) that the tests drive **without an audio device**.

- `playback.rs`: `Playback::start_span(samples, sr, start_s, end_s, LoopMode)`,
  `LoopMode = Once | Loop { pause_seconds }`, `set_paused`/`is_paused`;
  `PlaybackState` gains span bounds + paused + looping + loop-pause countdown; the
  three cpal `fill_buffer_*` share `next_mono_sample`. 8 engine tests.
- `main.rs`: pure `span_for_action(action, view, cursor, selection) → Option<(lo,hi)>`
  for the five view-relative spans (5 tests); transport methods `play_span` /
  `play_action` / `toggle_pause` / `stop_and_return` (+ `playback_origin`).
- Keymap (refinable, just key constants): **Space** = play selection-or-view once /
  pause-continue while playing; **Ctrl/Cmd+Space** = loop it (0.5 s gap); **`,`/`.`**
  = left/right of playhead; **`[`/`]`** = left/right of selection; **Ctrl/Cmd+**span =
  loop; **Esc** = stop + return to span start. Subsumes the simple "play selection on
  space" backlog item (note: second Space *pauses*; Esc *stops* — refine in testing).

App-only; gate green (+13 tests). Next: Slice 2 (digit tier activation), Slice 3
(Enter-to-commit + conflict resolution, with a dedicated boundary-reuse key).

## 2026-06-02 — Fix: script-panel placeholder was outdated (E8/E9 jargon)

The embedded Python panel's ghost text read "pure stdlib only at E8 / `import
sadda` lands in E9" — internal slice codes, and "lands in E9" implied the GUI
namespace wasn't available yet. E9 shipped long ago, so updated to "Embedded
Python (stdlib). `import sadda.app` reads the live GUI:" with a runnable example
(`sadda.app.active_bundle()`) — correct because `run_script_buffer` executes
inside `with_snapshot_active`. One-line `hint_text` change.

## 2026-06-02 — GUI: selection timestamps + reset-spectrogram-settings button

Two small app conveniences:
- The **waveform header** now shows the active span selection's boundary times
  and duration — `sel A–B s  (Δ C s)`, rendered strong against the weak
  bundle/view line — whenever a selection exists (pure `format_selection`,
  unit-tested; reads `TimelineState.selection`).
- The **spectrogram toolbar** gained a **Reset** button (after Colormap) that
  reverts window / hop / range / colormap to `SpectrogramConfig::default()`
  (25 ms / 5 ms / Viridis / 70 dB); disabled when already at default (so it
  reads as a no-op).

App-only; +3 tests; gate green.

## 2026-06-02 — README + docs screenshot

Wired the existing `assets/sadda_screenshot.png` (waveform + spectrogram + f0 /
formant / intensity measure tracks + bundle sidebar + reference panel) into the
README hero slot and the mkdocs landing page — copied to `docs/assets/` for the
site (mkdocs serves only under `docs/`), referenced as `assets/sadda_screenshot.png`
from both, matching the existing `annotation-cycle.svg` pattern. Validated with
`mkdocs build --strict`. The asset predates today's f0 fix — fine as a
representative shot; a fresh capture is worth doing eventually.

## 2026-06-02 — Help → Memory report (diagnostic)

A snapshot diagnostic under the Help menu: system RAM (total / used / available,
each as a % of total) plus **this process's resident size (RSS)** — sadda's
actual memory outlay against the machine. Pairs naturally with the adaptive
cache budget that just landed.

- `sysinfo 0.36` (the `system` feature only — disk / network / component / user
  off, to keep the build lean) for cross-platform system memory + per-process
  RSS (Linux / macOS / Windows). Distinct from the budget's `libc` `sysconf`
  path, which stays for its lighter total-RAM-only query.
- `MemoryReport` with `Option` fields (a figure the platform can't supply shows
  "unavailable", not a misleading zero); `gather_memory_report()` (sysinfo
  `new_all`); pure `format_memory_report()` (reuses `human_bytes`, %-of-total),
  unit-tested for full / all-unavailable / RSS-unavailable cases.
- Help → "Memory report" pops the green info panel (`set_info`), matching the
  About snapshot pattern (chosen: snapshot dialog over a live-refreshing window).

Sample on this 16 GB host: `System RAM: 15.6 GiB · used 1.1 GiB (7%) · available
14.5 GiB (93%) · sadda RSS: 8 MiB (0%)`. App-only; gate green (+3 tests).

## 2026-06-02 — Adaptive signal-cache budget (low-RAM win)

The P1 per-bundle signal swap-cache was bounded by a hard 768 MiB — fine on a
16 GB workstation, hostile on a 4 GB box where it competes with everything else.
Now adaptive: budget = **`min(768 MiB cap, ~15% of system RAM)`**, falling back
to the cap when RAM can't be determined.

- `system_ram_bytes()` — total physical RAM via POSIX `sysconf(_SC_PHYS_PAGES) ×
  sysconf(_SC_PAGESIZE)` on Linux + macOS (`libc` was already in the tree
  transitively → no new build cost); `None` on Windows → cap fallback.
- `cache_budget_for_ram(ram, cap)` — pure, unit-tested policy fn (16 GB → cap;
  4 GB → ~614 MiB; `None` → cap; boundary just over ~5 GiB).
- `signal_cache_budget_bytes()` wires them and logs the choice once under
  `SADDA_DEBUG`; all three `SignalCache` construction sites use it.

App-only (the cache lives in the app — no engine/Python surface). Verified
end-to-end: on this 16 GB host the budget stays 768 MiB (15%·16 GB > cap), so
workstations are unchanged; a 4 GB box now gets ~614 MiB. Gate green (app +5
tests).

## 2026-06-02 — Fix: f0 octave-down errors — default tracker → Boersma

The app's measure-track f0 — and Python `voiced_pitch`, and the criteria `f0`
signal — defaulted to `windowed_autocorrelation`, which on clean tones latches
onto **subharmonics**. A diagnostic across the band showed **150→75, 250→83.3,
300→100** under `PitchConfig::default()`: the tracker picks the global max of
`r_a(τ)/r_w(τ)`, and the window-correction over-inflates long-lag subharmonic
peaks because it has no octave cost and no path-finding. The faithful `Boersma`
tracker (octave-cost + octave-jump-cost + Viterbi) — which already existed and
already had an octave-robustness test — reports every tone correctly
(150/200/220/250/300/400). It simply predated the app default and was never
wired in.

**Fix:** make `PitchMethod::Boersma` the **canonical default** (`impl Default
for PitchMethod`) and route all three default call sites through
`PitchMethod::default()`: app `compute_measure_tracks`, engine `signal_set`
(criteria `f0`), and Python `voiced_pitch(method="boersma")` (docstring + stub
updated). Perf is a non-issue: Boersma is ~1.6× `windowed_ac` but only ~39 ms
for 30 s of 44.1 kHz audio (release), and the f0 lane is async (P2).
`windowed_autocorrelation` stays a selectable method, now with a doc-comment
warning about its subharmonic weakness.

Three surfaces: **engine** (`impl Default`; `signal_set`; tests
`default_pitch_method_is_boersma` + `boersma_tracks_pure_sines_without_subharmonic_errors`),
**Python** (default + docstring + stub + `test_voiced_pitch_default_method_is_boersma_and_octave_robust`
guarding 150/220/250 Hz), **app** (measure-track default).

While here, hardened the local gate: `just pytest` now rebuilds the extension
(`maturin develop`, `CONDA_PREFIX` unset) before running — `uv run` alone won't
rebuild on Rust-source changes, so pytest had been testing a **stale wheel**,
which masked this fix's Python side until caught. (Separately backlogged: the
`stubs` recipe's pre-commit `git diff` ergonomics.)

Gate: green — fmt · clippy · build · test · download · stubs · pytest (221 passed / 6 skipped).

## 2026-06-02 — Large-file ingest guard: warn-and-split on add

The pragmatic stand-in for the (deferred) windowed reader — meet the problem where it bites, at ingest. When a user adds a WAV whose **full decode would exceed ~512 MiB** of RAM (interleaved f32; the honest predictor of the load cost — ≈ a 2.3 h mono 16 kHz file, or ~13 min of 44.1 kHz stereo, same RAM hit), warn them and offer to **split it into contiguous pieces**, each its own bundle. The split **streams** the source (read-a-sample-write-a-sample, rolling to a fresh chunk file every N frames), so memory stays flat regardless of length — a file too large to *load* still gets *in*. Also the key low-RAM mitigation: it turns one un-openable long file into pieces that fit a 4 GB box.

Three surfaces:
- **Engine**: `Audio::probe(path) -> AudioProbe` (header-only — `hound` `duration()` reads the data-chunk size, no samples decoded; reports `decoded_bytes`); `Project::add_bundle_split(name_prefix, source, chunk_seconds) -> Vec<i64>` streaming chunked split, preserving the source's exact format (sample rate / channels / bit depth), final chunk = remainder, clean cuts (no overlap), each chunk landed as `"<prefix>_NNN"`. Refactored the bundle INSERT into a shared `insert_bundle_row`. `TierType: Hash` (from P3) unrelated. 2 tests (probe header math; 1000-frame file → 400/400/200 chunks summing back, format preserved, files on disk).
- **Python**: `sadda.probe_wav(path) -> AudioProbe` + `Project.add_bundle_split(...)`, provisional; stubs regenerated; 2 pytests.
- **GUI**: `add_bundle_guarded` probes on Add Bundle…; over-threshold raises a "Large audio file" dialog (live piece-count as you edit the per-piece minutes, default ≈ half the ceiling capped at 15 min) offering **Split / Add as-is / Cancel**. Pure helpers `human_bytes` + `split_piece_count`, both tested.

Gate: engine 203 lib tests, python +2, app 83 tests, clippy clean, stubs no-drift. Deliberately *not* built: the windowed reader / peak cache (deferred — see the design entry below), reference-in-place ingest, FLAC. v1 cut: split is whole-file contiguous only (no per-RoI or silence-aware splitting).

## 2026-06-02 — Design: windowed reader + multi-resolution peak cache (scale to long files) — DEFERRED

> **UPDATE 2026-06-02 (course correction):** on reflection the user judged this **premature / possibly unnecessary** for now — ultra-long *single* files are uncommon in practice. So this full design is **deferred to "planning for if long files become an issue"** (backlogged), NOT built. What ships *instead* is a small, pragmatic **warn-and-split-on-ingest guard**: when a user adds a file large enough to be problematic, warn them and offer to break it into manageable contiguous pieces (each its own bundle). The design below is kept verbatim as the record for if/when the windowed reader is revisited; treat the "Decisions / Slices (W0–W3)" as the *future* plan, not the current one.
>
> **Low-RAM framing (2026-06-02 follow-up):** the stronger argument for the windowed reader isn't "hundreds of hours" — it's *older / lower-RAM machines*. Today peak RAM ≈ longest open file × ~3–4 (decode + mono copy + spectrogram), so a single ~6 hr file (~4–5 GB working set) is un-openable on a 4–8 GB box, while short-file phonetics work is already fine there. So: **warn-and-split is the low-RAM mitigation now** (splitting streams, flat memory; turns one un-openable file into pieces that fit), and the **windowed reader is what would let long files open on the *least* capable hardware without splitting** — more valuable on small machines than big ones. Two cheap low-RAM wins captured separately in BACKLOG: (1) make the P1 cache budget `min(768 MiB, ~15% system RAM)` instead of a constant; (2) reference-in-place / FLAC ingest to ease disk doubling (small SSD/eMMC).

Scoping the engine for "hundreds of hours" corpora (ML-research / long sociophonetic sessions). Per the user, ultra-long *single* files are uncommon — the goal is for the engine to **do something sensible** at the extremes rather than OOM. The windowed reader + peak cache were sketched as the eventual proper fix; this entry records that design before any build. (Superseded for now by the warn-and-split guard — see the update banner above and the dedicated ship entry.)

### The hard wall today
`load_audio` → `Audio::from_wav_path` decodes the *entire* WAV into a `Vec<f32>`, and the renderer `.collect()`s a mono copy on top — no windowed/streaming read anywhere. Numbers (16 kHz mono): 1 hr ≈ 230 MB for `samples` alone; the 768 MB P1 cache holds ~2 such bundles; a single ~6 hr file (~1.4 GB) is effectively un-openable. Ingest also `std::fs::copy`s every file (2× disk; WAV uncompressed ≈ 115 MB/hr mono, ≈ 635 MB/hr 44.1k stereo). The just-shipped P3 `build_concordance` loads *all* matched bundles into one HashMap → OOMs on a big corpus (acute, easy fix: stream bundle-by-bundle since tokens are already bundle-sorted).

### The two pieces, and their division of labor
- **Peak cache** — whole-file, tiny (~1 MB/hr), persisted. Answers *"what does the file look like, zoomed out."* A waveform pane is ~1500 px; one column covers tens of thousands of samples and can only draw a **min/max** vertical line. So the cache stores the exact per-bucket **min/max(/rms)** — NOT an interpolation; at a given zoom a peak-drawn waveform is pixel-identical to a sample-drawn one. **Multi-resolution**: precompute at geometric decimation levels (base bucket 256, ×4 per level, up to ~1 peak/file). min/max compose associatively, so rendering any zoom = pick the finest level whose bucket ≤ the column span and aggregate a handful of peaks (min-of-mins/max-of-maxes), exact. Build folds `hound`'s sample *iterator* into buckets (streamed, O(1) memory) → safe for arbitrarily long files. Storage ≈ 1.3× the finest level.
- **Windowed reader** — `read_window(start_frame, n_frames) -> Audio` via `hound` seek-to-frame + read N (fixed-stride PCM; WAV-only, our only ingest format). Answers *"give me real samples for the slab on screen"* for spectrogram / f0+intensity / playback / deep zoom. Short files just call it once for the whole range (the eager fast path).

### How it generalizes P1/P2 (not thrown away)
`EnvelopeCache { Arc<Vec<f32>> }` goes from *whole file* → *current window* (+ frame offset) + a handle to the whole-file peak cache. Waveform renders from `PeakCache::render_range(view, n_px)` (never touches raw samples); spectrogram/measure-tracks compute over the **visible window** read via `read_window`. P1 `SignalCache` splits into many tiny **peak caches** (whole working set) + a bounded **window-signal cache** keyed by `(bundle, window, config)`. P2 async now fires on **bundle switch AND pan/zoom past the loaded window**; `poll_analysis` staleness key gains the window range. The renderer's `active_*` reads keep their shape; only what fills them changes.

### Behavior change (explicit)
Detail views cover the **visible slab + margin** and **recompute on pan** (the Praat-LongSound / Audacity tradeoff) for files past a **size threshold**; short files keep the eager whole-file path (free panning, simpler). Nothing regresses for the common case; long files trade free-pan for being openable.

### Prior art
Praat **LongSound** (in-RAM `Sound` vs on-demand-windowed `LongSound` — the direct precedent); Audacity block files + 256:1/65536:1 **summary** levels; BBC `audiowaveform`/peaks.js precomputed peak files; Lhotse/WebDataset manifest-of-references (reference-not-copy) from speech-ML; DAW `.reapeaks` + streaming + reference-in-place media.

### Decisions (Q&A 2026-06-02)
- **Storage**: compact **binary blob** per bundle in `signals/derived/peaks/` (display infra, not analysis data — keep it tiny/fast; own a minor format rather than bend Parquet to it).
- **Build timing**: **configurable** — lazy-on-first-open by default (persist, rebuild if missing or `n_frames` mismatch), with an opt-in **precompute-on-ingest** for bulk imports you'll browse later.
- **v1 scope**: **all of W0–W3 in one push** — long files work end-to-end, not just navigable.

### Slices (three-surface each)
- **W0** — sensible guard: refuse/warn on open if decoded size would blow a RAM ceiling (the safety net; lands first).
- **W1** — peak cache: engine build+persist (streaming)+`render_range`; Python `bundle.waveform_peaks(start,end,cols)`; app waveform renders from peaks. (Helps short files too — cheaper render.)
- **W2** — windowed reader: engine `read_window`; Python `bundle.read_window`; tests. (Pure addition.)
- **W3** — window-driven detail views (the invasive integration): spectrogram + measure-tracks over the visible window; generalize P1/P2 cache + staleness to `(bundle, window, config)`; eager-vs-windowed threshold; async re-read on pan/zoom.

Tactical aside to fold in: fix `build_concordance` to stream bundle-by-bundle (don't hold all source audio at once), and FLAC/compressed ingest stays on the backlog (orthogonal ~2× disk win).

## 2026-06-01 — P3: aggregate concordance view — concatenate corpus tokens into one bundle

The "aggregate view" the user asked for (see the design entry below): a single waveform/spectrogram/tier view that shows *all* of a query's tokens as if they were one sound file in sequence. Built as `Project::build_concordance(tier_name, labels, dest_name, gap_seconds)` — chosen design (per the user): **tier + label filter** as the token source, **token + remapped context**, materialised as a **read-only derived bundle** (not a virtual overlay), so it rides the *existing* render + P1/P2 cache/async layer for free rather than needing a new playback path.

What it does: gathers every interval on `tier_name` across all bundles whose label is in `labels` (empty = any), in `(bundle, time)` order; requires the matched bundles to share one sample rate (mixed rates error — v1); down-mixes each source to mono once; concatenates each token's `[start,end]` slice with `gap_seconds` of silence between; writes the result as a 16-bit PCM WAV (`write_mono_wav_i16`) and re-ingests it via `add_bundle`. Then it lays down a `"⟨source⟩"` **divider tier** (one interval per token, labelled `"<bundle> @ <orig-time>s"`) and **remaps each token's surrounding context**: every interval/point tier on the source bundle (skipping reference/dense + the divider name) is clipped to the token window and shifted onto the concordance timeline, grouped by source tier name via `ensure_tier`. Returns a `ConcordanceSummary { bundle_id, n_tokens, duration_seconds, n_context_annotations }`.

Three surfaces: **engine** `build_concordance` + `write_mono_wav_i16` helper, two round-trip tests (concat + divider + context-clip math; empty-match error); **Python** `Project.build_concordance(tier_name, labels, dest_name, gap_seconds=0.25) -> ConcordanceSummary` (frozen pyclass, stubs regenerated), 2 pytests; **GUI** an *Annotate → Concordance…* form (token tier / labels / new bundle name / gap), which on Build runs the engine call and `select_bundle`s the result so it opens immediately — label-field parsing extracted to a tested `parse_label_filter` free fn. Incidental: `TierType` now derives `Hash` (needed to key the per-tier-name dest map). Engine 201 lib tests, app 81 tests, clippy clean across the workspace.

Limitations (v1, all logged in the doc comment): single sample rate, mono only, no reference/dense tiers, annotation parent links not carried, edits don't flow back to sources. Natural follow-ups: cross-rate resampling, a "jump to source" affordance from a divider interval, and re-running a concordance when its query's matches change.

## 2026-06-01 — Perf P2: async DSP — first visits no longer freeze the UI

P1 made revisits free; P2 stops the *first* visit from blocking. The spectrogram + measure-track builds now run on **worker threads**: `rebuild_*_if_stale` dispatch a `std::thread::spawn` (sharing the envelope via the new `Arc<Vec<f32>>` mono samples + an `egui::Context` clone, so the worker can `request_repaint` on completion) instead of computing inline. Results return over an `mpsc` channel that `poll_analysis` drains each frame, installing only those still matching the current `(bundle, config)` — and uploading the spectrogram's `ColorImage` to a GPU texture on the UI thread (`build_spectrogram_texture` split into a worker-safe `compute_spectrogram_image` → `ColorImage`, plus a UI-thread `spectrogram_cache_from_image`). Per-kind in-flight guards (`pending_spectrogram` / `pending_tracks`) stop per-frame re-dispatch; a result that's gone stale (user switched / changed config mid-compute) is dropped. So on a cache **miss** the waveform paints immediately and the spectrogram / f0 / formants fill in a moment later, no frozen frame; on a **hit** (P1) everything still installs synchronously and instantly. `load_audio` + down-mix stay synchronous (cheap — async-loading is a later tail for hour-long files). App 80 tests, clippy clean. Known residual: VAD (ONNX) now runs on a worker thread — untested there, but it's off by default and failures already degrade to an in-lane hint. **P1+P2 complete** → P3 (the aggregate concatenated-timeline view) rides this same cache + async layer when it's next.

## 2026-06-01 — Perf P1: per-bundle signal cache — instant revisits

With the DSP now fast (f0 FFT + dev-profile), the remaining bundle-switch cost was paid *again on every revisit* — a switch invalidated everything, so scrubbing back and forth across a corpus re-loaded + re-ran the DSP each time. P1 adds a per-bundle **swap cache**: `select_bundle` pops the target's cached signals (envelope + spectrogram + measure tracks) and stashes the bundle it's leaving — **popping the target before stashing the old one**, so stashing can never evict the bundle you're entering. The renderer and the `rebuild_*_if_stale` paths are untouched (they still read `active_*`); computed signals get stashed naturally on the next switch and restored on return, where the existing config-staleness checks recompute only if the spectrogram/track config changed while away. **Byte-budgeted** LRU (`SIGNAL_CACHE_BUDGET_BYTES` ≈ 768 MB, dominated by the mono envelope) rather than count-bounded, since recordings span seconds to hours; cleared on project change (bundle ids are per-project), invalidated on bundle delete. Result: revisiting a recently viewed bundle skips the re-load *and* the DSP entirely — `SADDA_PERF` shows a lone `cache_hit` instead of the load/DSP lines. Unit-tested (`signal_cache_is_lru_and_byte_budgeted`); app 80 tests, clippy clean. **Next: P2 async**, so the *first* visit doesn't freeze the UI either.

## 2026-06-01 — Perf: the bundle-switch "slowness" was mostly a DEBUG build — optimise DSP in the dev profile

Per-lane instrumentation revealed the alarming `measure_tracks` numbers were a `cargo run` **debug** build. Same 104 s signal, release vs debug: **f0 73 ms vs 4033 ms (55×)**, **formants 664 ms vs 10491 ms (16×)** — unoptimised Rust strips the SIMD + inlining that `rustfft` and the autocorr/LPC inner loops depend on. So every numeric lane was 16–55× slower than reality in debug, swamping the algorithmic picture (and explaining why the f0 FFT win "didn't show" — debug penalty hid it; the FFT fix still cut `measure_tracks` 52.8 → 14 s in debug, then this took it to ~1 s).

Fix (workspace `Cargo.toml`): optimise *only the hot crates* in the dev profile — `[profile.dev.package.{sadda-engine, rustfft, realfft}] opt-level = 3` — leaving the app + binding crates at opt-level 0 (debuggable). **Verified**: a debug build's f0 dropped 4033 → 106 ms and formants 10491 → 893 ms (release-like). So plain `cargo run` is now usable for audio analysis; no `--release` needed for day-to-day testing. (One-time cost: a clean build recompiles those three crates optimised; incremental app rebuilds stay fast.)

Net for a 104 s bundle switch (debug, after both perf fixes): `measure_tracks` ~1 s (f0 ~0.1 s + formants ~0.9 s + intensity ~0.01 s), from 52.8 s. **Residual for HOUR-long files**: formants (~30 s/hr) is now the dominant DSP lane → next optimisation target (FFT-based LPC autocorrelation and/or frame parallelisation), alongside the spectrogram and the LRU-cache / async layer (P1/P2) for compute-once + non-blocking. Per-lane track timing (`· f0 / · formants / · intensity`) added to the `SADDA_PERF` output.

## 2026-06-01 — Perf: FFT-based pitch autocorrelation — ~700× faster, behaviour-preserving (P1)

The `SADDA_PERF` instrumentation (design entry below) showed `measure_tracks` dominating a bundle switch — **52.8 s for a 104 s recording** (~0.5× realtime; a 1-hour sociophonetic session would be ~30 min, unusable). Cause: `windowed_autocorrelation`'s per-frame autocorrelation was the naive time-domain `O(N · max_lag)` double loop (`autocorr_full`) — ~1–5M strided mults/frame over ~10 k frames.

Replaced `autocorr_full` with an **FFT autocorrelation** (`IFFT(|FFT(x)|²)`, zero-padded to `N + max_lag` for the *linear* result), `O(N log N)`, reusing thread-cached `realfft` plans (the spectrogram already pulls in `realfft`/`rustfft`). It returns the **same values** as the naive sum — new test `fft_autocorrelation_matches_naive_sum` asserts ≤0.1 % across all lags — so every tracker that uses it is unchanged; all 31 lib + 5 integration pitch tests stay green. Measured (`voiced_pitch` on synthetic tones): **~1300× realtime** (120 s → 83 ms), i.e. ~700× faster than before; an hour-long file's pitch now costs single-digit seconds. Both `autocorr_full` call sites benefit, and Python's `sadda.dsp` gets the speedup for free.

**Surfaced separately (pre-existing, NOT from this change — the value-equality test proves it):** `windowed_autocorrelation` makes **octave-down errors** on pure tones when `2·period ≤ max_lag` (200 Hz→100, 150 Hz→75; 120 Hz ok) — the `r_a/r_w` window-correction boosts subharmonics, and the method's docstring already flags the missing octave-cost / Viterbi terms. Backlogged; the app's default measure-track f0 may want pYIN/SWIPE or octave-cost terms.

Next in P1: the per-bundle LRU cache (free revisits) + frame parallelisation; then P2 async. The spectrogram is now the larger residual for very long files.

## 2026-06-01 — Design: bundle-switch responsiveness + the aggregate view — one signal-cache + async-compute layer (logged, not built)

Responsiveness when switching bundles across a corpus is, per the user, make-or-break for sadda being usable as intended. The user also flagged that this is **coupled** to the planned "aggregate" view (all of a query's tokens shown as one concatenated timeline) — and they're right: the machinery that makes a switch snappy is exactly what the aggregate view needs. So this designs **one shared layer** for both, before any code.

### What a bundle switch costs today
`select_bundle` (on click) runs `load_audio` (WAV read + decode) + a full mono `.collect()`, then invalidates the spectrogram so it — and the measure tracks — **rebuild on the *next frame*, on the UI thread**:
1. `load_audio` — I/O + decode (UI thread, on click)
2. mono down-mix collect — O(n) (UI thread, on click)
3. spectrogram — STFT + colormap + GPU upload (**UI thread**, next frame)
4. measure tracks — pitch (autocorr/Boersma) + formants (LPC) + intensity over the whole file (**UI thread**, next frame)

Two structural problems: **(a)** the heavy DSP (3, 4) blocks the frame after the click → the stutter; **(b)** **no cross-bundle cache** — a switch invalidates everything, so switching *back* recomputes from scratch. Scrubbing across a corpus pays full price every time. (A worker-thread + lock-free result-ring pattern already exists in the app, but only for *live recording* — a pattern to reuse, not new ground.)

### The architecture: three layers
Separate the concerns that are currently fused in `select_bundle`:

1. **View / time-map** — maps a *timeline position* to a `(bundle_id, time)`. The single-bundle view is the identity map (one bundle fills the timeline). The **aggregate view is just a different time-map** over an ordered segment list `[(bundle_id, start, end)]`. Nothing about signals or compute is view-specific.
2. **Signal cache** — a per-bundle `BundleSignals` keyed by `bundle_id` (+ the configs that affect each part): a **down-sampled min/max envelope pyramid** (cheap waveform at any zoom), the **spectrogram** (CPU dB grid + its uploaded `TextureHandle`), and the **measure tracks** (f0 / formants / intensity). Held in a small **LRU** (count-bounded to start, e.g. 6) so revisits are instant. Audio for a bundle is immutable, so only config changes invalidate the derived parts.
3. **Async producer** — a background worker (reusing the live-recording worker+channel pattern) that computes a `BundleSignals` for a requested `(bundle_id, configs)` and hands it back via a channel the UI drains each frame (exactly like the record dialog drains its rings).

### The flow that removes the stutter
On `select_bundle`:
- **cache hit** → display immediately (instant revisits — fixes (b));
- **miss** → load audio, build the **down-sampled envelope** (cheap) so the **waveform paints this frame**, mark the bundle selected, and **dispatch** spectrogram + tracks to the worker; those panels show a quiet "computing…" until the result lands and goes into the cache (fixes (a) — the UI never blocks on DSP).
- **Progressive reveal**: nothing → (decode) waveform → (DSP) spectrogram + tracks.
- **Staleness**: a generation token guards *display* ("is this result still the selected bundle?"); a late result for a now-unselected bundle still **enters the cache** (useful for the inevitable switch-back), so no work is wasted.

### How the aggregate view rides on the same layer (the payoff)
The aggregate view is a new **time-map** (step 1) over a segment list — and segment lists come straight from the criteria RoI query (the "one object, three faces" insight). To render, for each visible segment it needs that source bundle's signals over `[start, end]` — which it pulls from the **same** `BundleSignals` cache + async producer: cached → instant, else compute lazily as segments scroll in. The down-sampled envelope makes per-segment waveforms cheap; the spectrogram grid slices per segment. So the aggregate view adds **only** a time-map + a scroll-driven prefetch policy — the model and producer are unchanged. Build the cache+async layer once; both features ride it.

### Down-sampled waveform (a win on its own)
A min/max envelope pyramid (mip levels) renders the waveform in O(visible pixels) regardless of file length or zoom — standard in DAWs (Audacity, REAPER). Cheap to build (one O(n) pass), independent of the cache/async work, and required by the aggregate view (many segment envelopes).

### Decisions / recommendations (open to refine)
- **Async scope v1:** async the *DSP* only, keep `load_audio` sync → simplest, and decode is usually fast next to pitch/formants. Promote `load_audio` to the worker only if measurement shows decode dominates. **(rec)**
- **Cache eviction:** count-based LRU (e.g. 6) to start; revisit to memory-bounded if long recordings blow the budget. **(rec)**
- **Spectrogram cache granularity:** cache the uploaded `TextureHandle` (same egui ctx) keyed by `bundle + cfg`, so revisits skip both STFT *and* upload. **(rec)**
- **Slicing:** P1 — down-sampled envelope + per-bundle LRU cache (instant revisits, no threading); P2 — async producer + progressive reveal (kills first-visit stutter); P3 — aggregate view as a time-map on top. Each independently shippable + three-surface where relevant. **(rec)**

### Still measure — to *tune*, not to decide direction
Even with the layer decided, instrument the four cost centers (env-gated) to tune: LRU capacity (memory vs hit-rate), whether `load_audio` needs async, envelope pyramid depth. So an instrumentation pass is step 0 of P1.

## 2026-06-01 — Fix: Criteria editor's right panel collapsed (egui infinite-width footgun)

Found during user testing of the notebook→criterion flow: the Criteria editor's left-list / right-editor split rendered only the left list — the right panel (Name / Kind / Target tier / Rule body / Save / Run / Accept / Reject) was squeezed to zero width, so the editor looked dead (clicking a criterion or "+ New criterion" did nothing *visible*; the interactions fired but had nowhere to show). Cause (S2 code): the rule-body `TextEdit::multiline` used `.desired_width(f32::INFINITY)` **inside a `horizontal_top` layout** — an infinite-width child collapses its siblings in a horizontal layout. Fix: fixed-width (170) left column + bound the body to `available_width().max(280)`, and widened the default window (560→640). The other two `INFINITY` boxes (Rubric guidelines, annotation Note) live in *vertical* layouts where it means "fill width" correctly — left as-is. App 79 tests green, clippy clean; engine/python untouched. (User confirmed the editor + Run → `… (auto)` preview tier now work.)

## 2026-06-01 — Annotation workflow S7: the PI lab-notebook (shipped) — the suite is complete

The final slice. As the PI explores a corpus to define a study, they capture observations / measurements / decisions, then **promote** them into rubric artifacts — so the rubric's own creation is provenance ("this rule came from that observation"). Same iterate-loop the annotators use later, run earlier by the PI.

**Engine** (migration **V14** + `corpus.rs`): a `notebook_entry` table — `(target_type, kind, text, measurement, bundle_id, promoted_kind, promoted_ref, timestamps)` + index + 3 audit triggers. `kind ∈ {observation, measurement, decision}`; `target_type` is the free-text topic (usually a tier name); `measurement` optionally records the action/result behind a note (a free record at v1 — deeper recipe integration deferred). CRUD: `add_notebook_entry`, `notebook_entries(target_type?)` (newest-first, optional topic filter), `get_notebook_entry`, `update_notebook_entry`, `delete_notebook_entry`. **Two promote paths**, each stamping `promoted_kind` / `promoted_ref` on the entry:
- `promote_entry_to_criterion(entry, name, kind, body, target_tier)` — creates a criterion via `set_criterion` and links it (the computational rule).
- `promote_entry_to_rubric_guidance(entry)` — appends the note text to the `target_type` tier's rubric description (upserting `rubric_tier`) and links it (the prose rule).

**Decisions:** annotators/topics are free text (consistent with S4b/S6); guidance promotion *appends* to existing tier description (notebook accumulates guidance) rather than replacing; promotion is one-directional provenance (no auto-sync if the note later changes).

**Python**: the seven methods + `NotebookEntry` (provisional `sadda.NotebookEntry`). Stubs regenerated (additive).

**GUI**: an **Annotate → Notebook…** window — an add-note form (target type / kind / note / measurement), a topic-filtered list, and per-note **→criterion** (creates a template criterion to refine in the Criteria editor) / **→guidance** / delete, with a pure unit-tested `format_notebook_entry` (showing the promotion marker).

**Deferred / later:** a live measurement-runner feeding `measurement` (it's a recorded note at v1); recipe linkage for replaying measurement actions; promoting to a controlled-vocabulary *label* (guidance promotion targets the tier description).

**Gate (all green):** engine 293 lib + integration (incl. `notebook_captures_and_promotes_to_criterion_and_guidance`), clippy clean; python 190 passed / 6 skipped (`test_notebook.py`); app 79 (incl. `notebook_entry_line_shows_topic_kind_and_promotion`), clippy clean; stubs no drift.

**The annotation-workflow campaign suite (S1–S7) is complete:** S1 rubric-as-data → S2 criteria engine → S2.5 criterion-run provenance → S3 signal-function expressions → S4 campaign layer (a targets, b assignment, c distribution) → S5 agreement engine + work-queue → S6 dashboard (a) + rubric versioning/impact (b) → S7 lab-notebook. Migrations V8–V14. Next focus is open (validation runs / polish; a 0.4.0 cut bundling the suite is a natural milestone).

## 2026-06-01 — Annotation workflow S6b: rubric versioning (snapshot history) + impact (shipped)

The "evolve" half of S6, finishing the rubric loop (flag → refine → revisit). Snapshot-history approach (user's call), so **no per-annotation versioning** — annotations stay untouched; provenance carries the version.

**Engine** (migration **V13** + `corpus.rs`): a `rubric_version` table — `(version UNIQUE, name, guidelines, snapshot JSON, note, created_at)` + 3 audit triggers. The snapshot is an opaque JSON blob (engine-owned `RubricSnapshot`: statuses + per-tier config + controlled vocabularies), so the rubric scheme can evolve without a schema change. `StatusDef` / `VocabEntry` gained serde derives for it.
- `publish_rubric_version(note)` snapshots the current rubric under its current `version` (upsert on version — tweak before bumping; `set_rubric(version+1)` starts a new one). `rubric_versions()` lists; `get_rubric_version(v)` recalls the full scheme.
- **Impact** (`rubric_impact(version) → [TierImpact]`): per tier, the vocabulary values added / removed since a past version, and how many *current* annotations are now out of the current vocabulary (need revisiting — the step-7 loop). Only changed/affected tiers, tier-ordered. Reuses S6a's out-of-vocab counting.
- `record_criterion_run` now records `rubric_version` in its params alongside `rubric_id` (the schema-ready slot from S2.5).

**Decisions:** publish upserts the current version's snapshot rather than erroring on re-publish (edit-then-snapshot ergonomics); impact is measured against the *current* rubric's vocab (so a removed label shows as affected annotations); annotations are never rubric-version-tagged (snapshot history + provenance suffice — the invasive per-annotation column was explicitly rejected).

**Python**: `publish_rubric_version` / `rubric_versions` / `get_rubric_version` / `rubric_impact` + `RubricVersion` / `RubricTierSnapshot` / `TierImpact` (provisional; snapshots expose the existing `StatusDef` / `VocabEntry` pyclasses). Stubs regenerated (additive).

**GUI**: the **Dashboard** window gained a *Rubric versions* section — a publish-with-note control, the published-version list, and an "impact since version N" report via a pure unit-tested `format_tier_impact`.

**Deferred to S7 / later:** rubric *rollback* (recall is read-only — re-applying a snapshot to the live rubric is not wired); diffing two arbitrary past versions (impact compares a version to *current*); the protocol-registry (4th registry) sharing of versioned schemes.

**Gate (all green):** engine 292 lib + integration (incl. `rubric_versioning_snapshots_recalls_and_reports_impact`), clippy clean; python 187 passed / 6 skipped (`test_rubric_versions.py`); app 78 (incl. `tier_impact_line_reads_naturally`), clippy clean; stubs no drift. **S6 complete (S6a dashboard + S6b versioning/impact). Next: S7 — the PI lab-notebook (measurement-actions + notes per target-type → promote-to-rubric/criterion), the final roadmap slice.**

## 2026-06-01 — Annotation workflow S6a: the compile + QA dashboard (shipped)

S6 is the "monitor and evolve" layer; user chose to **decompose it dashboard-first**. This slice (S6a) is the *compile + QA dashboard* — pure read-only aggregation over what S4/S5 built, **no migration**. S6b (rubric *versioning* + impact) is next, and will use the snapshot-history approach (user's call).

Three reads, the three dashboard panes:
- **Completeness** (from assignments/targets): `project_target_progress()` sums `target_progress` across all bundles (the headline), and `assignment_progress()` rolls assignments up per annotator (`assigned`/`in_progress`/`done`, annotator-sorted) — "who has how much left".
- **QA sanity** (per tier): `tier_qa(tier_id) → QaReport` flags out-of-vocabulary labels (against the tier's S1 controlled vocabulary), empty/missing labels, and — for interval tiers — overlapping interval pairs. Reference/dense tiers report zeros.
- **Accuracy** (from the S5 agreement engine): `agreement_summary(bundle, base) → [PairAgreement]` finds every `"<base> [annotator]"` tier (the per-annotator tiers S4c import produces), parses the annotator out of the bracket, and runs `compare_tiers` on each annotator pair — closing the loop "S4c lands per-annotator tiers → S5 compares them → S6 summarizes".

**Decisions:** all aggregation lives on `Project` (no new module — these are thin reads over existing tables); annotator identity is parsed from the `"<base> [annotator]"` tier-name convention rather than stored (consistent with S4b's free-text annotators); QA `overlaps` is an all-pairs positive-intersection count (fine at tier scale).

**Python**: `project_target_progress` / `assignment_progress` / `tier_qa` / `agreement_summary` + the `AnnotatorProgress` / `QaReport` / `PairAgreement` result types (provisional `sadda.*`; `PairAgreement.report` is the S5 `AgreementReport`). Stubs regenerated (additive).

**GUI**: a dedicated **Annotate → Dashboard…** window (`dashboard_window`) — a live Completeness pane (overall + per-annotator) and an on-demand QA & agreement pane (pick a tier → Run QA; type a base tier → Summarize agreement). Pure unit-tested `format_annotator_progress` / `format_qa_report` (with the existing `format_target_progress` / `format_agreement_report`).

**Deferred to S6b / later:** rubric version *history* (snapshot table + publish/recall) and **impact tracking** (re-check annotations against a chosen version's vocab); a curator/adjudication view; CSV/report export of the dashboard.

**Gate (all green):** engine 291 lib + integration (incl. `dashboard_compiles_completeness_qa_and_agreement`), clippy clean; python 185 passed / 6 skipped (`test_dashboard.py`); app 77 (incl. `dashboard_lines_read_naturally`), clippy clean; stubs no drift. **Next: S6b — rubric versioning (snapshot history) + impact tracking. Then S7 (PI lab-notebook).**

## 2026-06-01 — Annotation workflow S5: the agreement engine + work-queue (shipped)

S4 (the campaign layer) is complete, so S5 adds the **QA core**: the comparison/agreement engine and the annotator throughput/work-queue. Built both together (user's call) with the agreement engine reporting **both** the unit-based and frame-based paradigms (method diversity).

**The "one comparison engine, three uses" realised** (`agreement.rs`, a pure module like `dsp/` — no `Project` coupling, unit-tested): `compare_intervals` / `compare_points` over plain `Segment` / `Mark` lists. The same engine serves inter-annotator agreement (the `"phones [alice]"` vs `"phones [bob]"` tiers S4c import produces), auto-criteria-vs-gold (a preview `(auto)` tier vs a manual tier), and rubric-version impact (S6) — all "compare two label sequences over one time base".
- **Unit-based** (forced-alignment tradition): greedy max-overlap 1:1 matching → **Cohen's κ** (Cohen 1960, cited) + % label agreement over matched pairs + mean boundary deviation + % boundaries within tolerance (default 20 ms) + insertions/deletions for unmatched units.
- **Frame-based** (diarization tradition): sample a fixed grid (default 10 ms), compare the per-frame label each side assigns (a `∅` category for gaps) → frame κ + agreement. No matching; robust to divergent segmentation. Reported alongside the unit metrics because they answer different questions.
- κ degenerate-case conventions documented (no pairs → 0; single-category → 1 iff perfect else 0). Points get nearest-1:1 matching + time deviation; frame metrics are N/A (0.0).

**Work queue** (`corpus.rs`): `target_progress(bundle) → ProgressCounts` (targets by status) and `next_target(bundle, statuses) → Option<Target>` (time-ordered — `["unassigned","assigned"]` = next-to-do, `["flagged"]` = next-flagged). Flag/status itself reuses S4a's `update_target_status` (`'flagged'` is already a target status).

**Engine wrapper:** `Project::compare_tiers(bundle, a_id, b_id, opts)` adapts stored interval/point tiers into the pure engine; guards that both tiers are on the bundle, share a type, and are interval/point.

**Python**: `compare_tiers` (kwargs `boundary_tolerance_seconds` / `frame_step_seconds`) → `AgreementReport`; `target_progress` → `ProgressCounts`; `next_target`. Both result types provisional `sadda.*`. Stubs regenerated (additive).

**GUI**: the Targets… panel gained a QA section — a progress line (`format_target_progress`), **Next to do** / **Next flagged** buttons (`next_target`), and a **Compare** A-vs-B tier picker showing a compact report via the pure `format_agreement_report` (κ, label %, match counts, boundary Δ/tolerance, frame κ). Both helpers unit-tested.

**Deferred:** multi-rater (Fleiss' κ; we do two-rater Cohen); a dedicated adjudication *view* (side-by-side diff with accept-from-A/B) beyond the numeric report; the rubric-version-impact use awaits S6 versioning; a real waveform jump on "next-target" (the button reports it as a status line for now).

**Gate (all green):** engine 290 lib + integration (agreement.rs 8 unit + `compare_tiers_…`, `target_progress_…`), clippy clean; python 181 passed / 6 skipped (`test_agreement.py`); app 76 (incl. `progress_and_agreement_lines_read_naturally`), clippy clean; stubs no drift. **Next: S6 (compile + QA dashboard + rubric versioning + impact tracking), then S7 (PI lab-notebook).**

## 2026-06-01 — Annotation workflow S4c: per-annotator package export / import / merge (shipped)

The last piece of the campaign layer, and the one I'd flagged as heaviest: **distribution**. Local-first / no-server → hand-off is a *package*, not a shared web app. The PI exports each annotator a self-contained slice, they work offline, the PI imports it back. Across the three surfaces.

**Design forks (user-decided):** package format = a **self-contained sub-project directory** (a real sadda project: copied audio + a `corpus.db` + manifest — dep-free; the annotator just opens it; zipping for transfer is the user's call). Merge model = **per-annotator tiers PLUS an explicit `merge_tiers`** — the user's refinement of my "smart merge": import never silently combines; each annotator's work lands on `"<tier> [annotator]"`, and a separate PI-driven `merge_tiers` unions selected tiers. Cleaner separation than auto-merging disjoint vs overlap on import.

**No migration.** A package *is* a normal sadda project (same schema V12), so S4c is pure orchestration — no V13.

**Engine** (`corpus.rs`):
- `export_annotator_package(annotator, dest_dir) → ExportSummary`: the bundles with a target assigned to `annotator` (audio via `add_bundle`'s copy), their **sparse interval/point tiers + annotations** copied with tier-`parent_id` and annotation-`parent_annotation_id` **remapped** through id maps (tiers placed **parent-first** via `parent_first_order`), the annotator's targets+assignments, the rubric (`copy_rubric_into` — name/version/guidelines/status vocab + per-tier config & CVs), and a `sadda_export.json` manifest (`{format, annotator, source_project, schema_version}`, serde_json).
- `import_annotator_package(package_dir) → ImportSummary`: reads the manifest, opens the package, matches bundles **by name**, and for each assigned target-type lands the package tier's annotations onto `"<tier> [annotator]"` (created/refilled), then marks that annotator's assignments on matched bundles `done` (importing the package = "they finished here").
- `merge_tiers(bundle, source_names, dest_name)`: unions same-type (interval/point) source tiers into a destination in time order (read-all-before-clear, so a destination that is also a source isn't wiped early).

**v1 scope cuts (documented):** dense (measure-track/vector) + reference tiers aren't copied; rubric *versioning* is S6 (current rubric copied as-is); the criterion behind a target isn't exported (targets keep their RoI/type/status, `criterion_id` dropped); bundle matching is by name.

**Python**: `export_annotator_package` / `import_annotator_package` / `merge_tiers` on `Project` (paths as `str`/`PathLike`), returning `ExportSummary` / `ImportSummary` (provisional `sadda.*`). Stubs regenerated (additive).

**GUI**: the Targets… panel gained a Package row (**Export for annotator…** / **Import package…** via `rfd` folder pickers) and a **Merge tiers** row (sources + dest), with pure unit-tested `format_export_summary` / `format_import_summary` status lines.

**Gate (all green):** engine 280 lib + integration (incl. `export_import_round_trip_lands_per_annotator_tier`, `merge_tiers_unions_sources_in_time_order`), clippy clean; python 177 passed / 6 skipped (`test_packages.py`); app 75 (incl. `package_summaries_read_naturally`), clippy clean; stubs no drift. **S4 (the campaign layer) is complete — S4a targets + S4b assignment + S4c distribution. Next: S5 (annotator throughput + QA core: flag/status UX + work queue + the comparison/agreement engine).**

---

## Archives

Older months are rotated into [`devlog/`](devlog/) to keep this file lean
(one file per month). Newest first:

- **[2026-05](devlog/2026-05.md)** — project genesis → 0.2.0 / 0.3.0 releases → annotation suite S1–S7 → perf arc + large-file ingest guard
