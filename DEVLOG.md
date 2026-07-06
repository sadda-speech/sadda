# DEVLOG

A running log of research, decisions, and development for the SpeechAnalysisTool project — a planned next-generation phonetics / speech-science tool.

Newest entries at the top. Each entry is dated `YYYY-MM-DD` and tagged with a short topic. This file holds the **current month**; earlier months are rotated into [`devlog/`](devlog/) (index at the bottom).

---

## 2026-07-06 — Forced-alignment silence handling (prototype, both detectors)

A1's aligner absorbed non-speech into edge phones (the demo clip's leading "I"
ran 0.00–0.98 s; "a" stretched over a 0.66 s pause). This adds silence detection.

**Standard problem:** *optional silence* in forced alignment. Field-standard
aligners (HTK/P2FA, Kaldi, MFA) model an explicit optional `sil` between words
and at edges. In the CTC setting the model has no `sil` phone, but its **blank**
(`<pad>`) fires during silence ("no phoneme emitted").

**Decided forks (user-confirmed):**
- **Detector → both, user-selected.** `align(..., detector="blank"|"vad"|None)`.
  `"blank"` (default): CTC-blank runs ≥ `min_silence_seconds` (~120 ms) → silence;
  reuses the model's own posteriors. `"vad"`: Silero VAD (`sadda.ml`), external
  per-frame mask. Both combine in the engine (mask OR blank-run).
- **Representation → empty-labeled intervals, not gaps.** The user's point: an
  interval *tier* is a full partition; silence is an **empty-labeled** interval,
  not a hole. So the Word/Phone results stay contiguous over `0..T`; silence
  intervals carry `label=""` (Praat empty intervals on export). A1's contiguity
  property is preserved — we insert empties, not gaps.

**Implementation.** Engine `forced_align` gains `min_silence_frames` + optional
`silence_mask`; a **carve** step (leaving the DP + collapse untouched) splits the
contiguous phone spans wherever silence is detected → `TokenSpan.is_silence`.
Added `Audio::from_samples` (+ `sadda.Audio.from_samples`) so the VAD detector can
build an `Audio` from the numpy waveform for `sadda.ml` VAD.

**Blank vs VAD (validated on demo.wav):** blank keeps every word and its silence
boundaries agree with the alignment (leading "I" → pause 0.00–0.94, "I" 0.94–0.98;
the pause after "a" broken out). VAD is **coarser** — its independent boundary
swallowed the 40 ms leading "I" — and can *drop* a word whose phones fully overlap
VAD-silence. Hence **blank is the default**; VAD is opt-in.

**The modeled-silence caveat (user's wish):** neither detector is a *trained*
silence model — both infer it. Truly modeled silence (an explicit `sil` state)
arrives with **A2 (MFA passthrough)**, whose HMM models optional silence directly.
Backlogged: a silence-aware neural model, and a VAD∩blank hybrid to fix VAD
dropping short phones.

**Status:** prototype on `feat/align-silence` (draft PR) — engine 263 + python 313,
clippy/fmt clean. **Pending review:** confirm the `min_silence_seconds` default and
the blank-default choice.

## 2026-07-05 — Design: ASR + phone-level forced alignment (A1–A5)

Design for the "ASR + forced alignment engine" backlog item (produces Words /
Syllables / Phones tiers, "defaults that just work"). Framed → prior art →
decided the forks → picked the model → sliced it.

**The distinction that structures everything.** Two problems get bundled here and
are separable: **forced alignment** (audio *+ a known transcript* → time-aligned
Word/Phone boundaries — the phonetician's classic tool, a *solved* problem whose
field standard is the **Montreal Forced Aligner**) and **ASR** (audio → text, for
when there's no transcript). Most phoneticians *have* transcripts, so alignment is
the core deliverable and ASR is a convenience layer on top. **Syllables are not a
separate model** — they're derived from the Phone tier by rule.

**Decided forks (user-confirmed):**
- **Flow → alignment-first** (bring-your-own transcript). Whisper ASR is a later
  slice (A4), not the core.
- **Engine → both**, per the DSP-method-diversity convention: a built-in **neural
  aligner** as the zero-setup default + **MFA passthrough** for the gold standard.
- **Phones → IPA, multilingual**, via espeak-ng G2P (not English-only ARPABET) —
  aligns with the localization thread and reuses the TTS espeak-ng dependency.

**Architecture — two cleanly separable stages, and unlike TTS this one has a real
Rust-engine core:**
1. **Acoustic posteriors** — audio → per-frame phone (CTC) probabilities. Neural,
   ONNX → `sadda.ml`. (MFA does its own GMM-HMM version internally.)
2. **Constrained alignment** — target phone sequence (from G2P) + posteriors →
   boundaries via a constrained-Viterbi / CTC forced-align DP. **Pure algorithm →
   the Rust `engine`**, like the agreement engine and DSP: fast, deterministic,
   unit-testable. (Contrast TTS, which had no engine role, so its GUI/engine
   surfaces were deferred; forced alignment earns all three surfaces.)

Backends plug in at stage 1 (which posteriors); espeak-ng feeds stage 1's target
sequence; the DP + tier construction are shared.

**Model decision (researched, clean-license gate applied — CC-BY-NC is out, as
with XTTS for TTS):**
- **Neural default → `facebook/wav2vec2-lv-60-espeak-cv-ft`** — **Apache-2.0**,
  CTC, multilingual, and its output tokens **are espeak IPA phonemes** (trained on
  eSpeak-phonemized CommonVoice), so it matches espeak-ng G2P with no mapping
  layer. The only candidate that satisfies *all* of {clean license · espeak-IPA
  match · CTC · multilingual}. Exported to ONNX, run via the `sadda.ml` ORT path;
  wav2vec2-large (~315M) → a chunky download, so it lives behind a **`sadda[align]`
  extra + `hf://` fetch** (int8 to shrink it), never the base install.
- **MFA 3.0 (MIT) → gold-standard passthrough** (A2): subprocess `mfa align` →
  TextGrid import (already supported). Mean boundary error <15 ms, harmonized IPA
  dicts; heavy conda/Kaldi install, so it's opt-in, detected-or-clear-error (the
  Kokoro-pending pattern).
- **charsiu (MIT)** — noted alternative; does **text-independent** alignment (align
  with no transcript — a future option) but IPA support is *TBD* and models are
  per-language, so not the default.
- **MMS → rejected**: `uroman` (not IPA) + CC-BY-NC weights on several checkpoints.
- **G2P**: espeak-ng `--ipa` via the binary already wrapped for TTS — avoids the
  GPL `phonemizer` Python dep. A1 must verify espeak-ng's token set lines up with
  the model's (small normalization/mapping may be needed).
- **Syllabification**: Phone tier → Syllable tier by sonority-sequencing +
  maximal-onset (Selkirk; language-tunable). Pure `engine` module, no model.

**QA loop, nearly free:** pipe an aligner's output + a manual tier into the S5
**agreement engine** → boundary deviation + κ. Gives users alignment validation
*and* a way to benchmark neural-vs-MFA on their own data.

**Where it lives (three surfaces):** `engine` (the forced-align DP + syllabifier +
tier construction), Python (`sadda.align` + the ONNX model in `sadda.ml`), GUI
(A5). Provenance records aligner + model + params.

**Slice plan:**
- **A1** — engine forced-align DP + espeak-ng IPA G2P + the ONNX phone model →
  **Words + Phones** interval tiers, Python API. The core.
- **A2** — MFA 3.0 passthrough (subprocess → TextGrid import).
- **A3** — syllabification (Phones → Syllables).
- **A4** — Whisper ASR (the no-transcript path).
- **A5** — GUI surface.

**Open risks:** model ONNX size (int8/distillation); exact espeak-ng↔model token
match; neural-vs-MFA accuracy gap (neural still trails HMM-GMM on boundary
precision — real for sub-15 ms phonetics work); non-English syllabification rules.

**Refs:** MFA & the state of alignment in 2026 (arXiv 2606.18466); Xu et al.,
zero-shot cross-lingual phoneme recognition (the espeak wav2vec2 model); torchaudio
`forced_align` / CTC-segmentation (Kürzinger et al. 2020); Zhu et al., charsiu
(text-free phone alignment); BFA (arXiv 2509.23147). **Next: A1, starting by
confirming the espeak-ng↔model phoneme token alignment.**

## 2026-07-05 — TTS pipeline T1: the backend-agnostic voiceover core (shipped)

First slice of a text-to-speech capability. The **immediate** driver is voiceover
for auto-generated documentation (screencasts / tutorial videos); the **general**
mandate is that the same surface serve any user's ad-hoc TTS. Those two goals have
different centres of gravity — the docs use case wants *reproducible, free,
offline, CI-runnable*; the general use case wants *pluggable backends + a stable
API* — and a single backend abstraction serves both as long as the docs use case
doesn't quietly dictate the public API.

**Design forks (I proposed defaults; the user stepped away mid-question, so I
proceeded on the recommendation — flagged here as pending confirmation):**

- **Scope / home → ship `sadda.tts` (Python), GUI deferred.** Not the tooling-only
  `tools/` option, and *not* full three-surface. This is a deliberate, flagged
  departure from the three-surface principle (engine + Python + GUI for every
  user-facing capability): neural TTS has **no meaningful Rust-engine
  implementation** to write (it's ONNX / Python), and a phonetics *analysis* tool
  has no proven user story for a "type text, hear voice" GUI button yet. A
  GUI-native path would force the egui app to either shell to Python or run
  Kokoro-ONNX via the `ort` crate — real cost, deferred until a concrete analysis
  use case (analysis-by-synthesis, perception-experiment stimuli) appears.
- **Backend → local default + pluggable, cloud later.** A structural
  `TTSBackend` protocol (`name` + `synthesize(text, out_path, *, voice, rate)
  → SynthesisResult`) is the whole contract; caching / assembly / pipeline speak
  only to it. Cloud backends (ElevenLabs / OpenAI) are designed to plug in the
  same way as opt-in add-ons.
- **Default engine → espeak-ng now; Kokoro is the planned quality default.**
  `EspeakNgBackend` shells out to the system `espeak-ng` (22.05 kHz mono 16-bit):
  robotic, but zero-dependency, offline, deterministic, and phonetically apt — the
  right *reference / CI* backend where reproducibility beats naturalness. **Kokoro**
  (82M, Apache 2.0, CPU faster-than-real-time, near-ElevenLabs for clean narration)
  is registered but **not yet wired**: requesting it raises an actionable error
  rather than shipping guessed API calls, pending the `sadda[tts]` extra decision.
  Kokoro's Apache-2.0 aligns with sadda's Apache/MIT stance; **Piper was passed
  over** because its active fork is now GPL-3.0 (copyleft — awkward to bundle even
  if only invoked as a subprocess) and it's more robotic; XTTS/F5/Fish are
  non-commercial → out.
- **Pipeline scope → audio voiceover only.** Screencast/gif capture + A/V muxing
  (the rest of the doc-automation vision) is a genuinely different problem
  (GUI-driving under WSLg/xvfb + ffmpeg) → BACKLOG, not this slice.

**What shipped (`python/sadda/tts/`, pure Python, PROVISIONAL):**

- **`script.py`** — the narration model: `Segment` (text + optional voice / rate /
  `pause_after_s` / stable `id`) and `NarrationScript` (segments + script-wide
  voice/rate defaults, with a segment-wins fallback chain). `parse_script` is a
  deliberately minimal text convenience (blank-line-separated paragraphs →
  segments; soft-wraps collapsed). A richer on-disk format (per-scene ids, inline
  directives, screencast timing markers) is an **open design question** → BACKLOG.
- **`backends.py`** — `TTSBackend` protocol, `SynthesisResult`, `EspeakNgBackend`
  (text fed via `-f tmpfile` to dodge quoting / arg-length; `rate` multiplier →
  clamped wpm), and a name→factory registry (`get_backend` / `list_backends` /
  `register_backend`; default via `$SADDA_TTS_BACKEND` → `espeak-ng`).
- **`pipeline.py`** — the layer voiceover calls. **Content-hash caching**
  (`cache_key` over `(backend, voice, rate, text)`) so a doc rebuild only
  re-synthesizes changed lines — the crux of cheap, reproducible generated docs.
  `synthesize` (one-shot), `synthesize_script` (per-segment cached synthesis +
  optional assembly), `concat_wavs` (stdlib `wave`; inserts per-segment silence,
  rejects sample-rate/width/channel mismatches).

Wired into the top-level package (`import sadda; sadda.tts.…`), matching the
`dsp`/`ml` eager-import convention. **No new dependencies** — espeak-ng is a system
binary, not a pip dep, so the base install stays lean; the future `sadda[tts]`
extra is where Kokoro/torch will live.

**Verified end-to-end** (not just unit tests): a two-paragraph script → espeak →
a 9.34 s assembled 22.05 kHz mono `narration.wav` (4.51 + 4.42 s segments + 0.4 s
pause), 2 cache entries; a re-run with one line changed re-synthesized exactly one
segment.

**Deferred → BACKLOG:** Kokoro backend + `sadda[tts]` extra (needs the dependency
decision); cloud backends (ElevenLabs/OpenAI); screencast/gif capture + A/V mux;
richer narration-script format; a GUI surface (only if an analysis user story
lands); a docs API-reference page for `sadda.tts`.

**Gate:** Python **251 passed / 6 skipped** (incl. `test_tts.py`: 13 — script
parse/fallback, registry + Kokoro-pending error, cache-key stability, WAV concat +
mismatch guard, pipeline assembly + cache-hit + single-segment re-synth, and an
espeak-ng integration test that skips when the binary is absent so CI stays green).
Rust side untouched (pure-Python slice; stubs/clippy unaffected). **Next: confirm
the forks above, then wire Kokoro (the `sadda[tts]` extra) as the quality default.**

## 2026-07-02 — Doc-image catalog + Phase-1 figures rendered from a real clip

Planned the documentation-image set (what would most help users) now that the
pipeline exists, and rendered the first batch. The docs had **almost no
screenshots** — one hand-taken hero + a logo — so this fills the biggest gap.

**Demo clip.** A ~6s CC0 recording (user-provided), *"I don't know, should I
make a picture of the app? I guess I should make a picture of the app"* — chosen
for its clear question-vs-statement intonation and varied vowels. Committed at
`docs/recipes/assets/demo.wav` (mono 16 kHz).

**Catalog (organised by user need, not feature enumeration):**

- *Group A — the signal surfaces* (renderable now): **overview/hero** (the whole
  app, maximal), **signal-view** (waveform+spectrogram), **spectrogram**,
  **pitch-contour**, **formant-tracks**, **intensity**, **mfcc**,
  **measure-stack**. Light + dark for the hero. → `docs/recipes/overview.py`,
  output to `docs/assets/generated/`.
- *Group B — annotation + interaction* (needs API extensions): **annotated
  tiers** (done this pass — see below), then **selection/measurement**,
  **reference-distribution panel**, **DSP-method comparison**, **corpus/bundle
  navigation** (deferred; each needs a small recipe primitive — selection,
  refdist panel toggle + install, DSP-method, multiple bundles).

**B1 shipped — the annotation piping.** Added `shot(textgrid=…)`: the runner
imports a Praat TextGrid into the bundle so the tier strip has content
(`RecipeShot.textgrid` → `apply_shot` → `project.import_textgrid`). The hero is
now **maximal**: menu bar + sidebar + waveform + spectrogram + MFCC + f0 +
formants + intensity + an annotation tier, showing sadda's *signal* and *corpus*
sides in one figure. Guarded by `textgrid_import_adds_tiers`.

**Annotation.** `docs/recipes/assets/demo.TextGrid` now holds the **real**
annotation — an **Utterance** tier (2 phrases) + a **Words** tier (21 words),
created *in sadda* (dogfooding the annotation workflow) and exported from the
project DB (gaps filled as empty intervals for valid Praat tiers). The hero shows
both, aligned under the speech; the narrow word intervals also exercise the new
label width-fit + red truncation indicator. (An earlier placeholder phrase tier
was superseded.)

**Notable:** lane-focused shots need the *focal* lane sized explicitly, because
the spectrogram is the flex/remainder pane and otherwise eats the height — e.g.
`heights=[("f0", 220)]` for the pitch figure. The hero also re-confirmed the
backlogged y-axis-label clipping bug ("formants" clipped on the left).

**Next:** wire the images into the doc pages (Home hero, a "tour" section,
annotation-cycle), the user's real demo annotation, and the Group-B primitives.

---

## 2026-07-02 — Shipped: view-composition scripting + headless documentation-image pipeline (S2–S8)

Built the whole documentation-automation strand designed in the two entries
below, on `feat/figure-export`. The result: **compose the app's view, capture a
region, and regenerate documentation images headlessly from a Python recipe —
all drift-tested against the real app.**

- **S2 — hand-draw capture** (`crates/app/src/capture.rs`). Rubber-band a region
  → crop the framebuffer → save PNG; echoes the pixel rect so a hand-drawn
  selection lifts into a scriptable `capture(rect=…)`.
- **S3 — visibility & selection model.** Every signal subpane is now show/hideable
  (waveform/spectrogram/tier strip had no toggle before) and every tier is
  selectable in/out; a dynamic **flex-lane** chooser so hiding *any* lane reflows
  instead of leaving a hole. Scriptable: `sadda.app.set_pane_visible` /
  `set_tier_visible`.
- **S4 — named-rect registry + named capture.** A per-frame registry of named
  regions (composites `whole-window`/`signal-column`, every lane, the side
  columns); a "Capture ▸ region" menu; capture by name or pixel rect.
- **S5/S6.1/S6.2 — reproducible layout, all scriptable.** `set_window_size` (doc
  presets + zoom pin), `set_pane_height`, `set_column_width` — the last two by
  writing egui's `PanelState`, so drag stays intact and sizes persist.
- **S6 — the headless spine** (`crates/app/src/doc_render.rs`, `#[cfg(test)]`).
  `egui_kittest` + wgpu drives the *real* `SaddaApp` offscreen through the same
  egui/wgpu stack users see, on **lavapipe** (software Vulkan) — no window, no
  display. This is the anti-drift guarantee made concrete.
- **S7 — the recipe runner.** `sadda.doc.shot(...)` declarative Python recipes
  (project/audio, bundle, size, theme, visibility, heights, widths, capture);
  `audio=` builds a throwaway project from a WAV so recipes are self-contained;
  external recipe *files* under `docs/recipes/`; `just docs-images`; a light/dark
  `set_theme` knob.
- **S8 — the drift gate.** The cropped figure is an `egui_kittest` snapshot golden
  (`crates/app/tests/snapshots/`); `.github/workflows/docs-images.yml` renders on
  lavapipe in CI — structural checks blocking, the pixel snapshot advisory until
  goldens are regenerated from CI's own renderer.

**Scripting surface** (drained after a run, like `register_command`):
`sadda.app.{set_pane_visible, set_tier_visible, set_window_size, set_pane_height,
set_column_width, set_theme}` + `sadda.doc.shot`. Every one is exercised by a
Python-through-the-interpreter integration test; the layout ones are additionally
verified by *rendering* and measuring the real geometry.

**Notable engineering:** wgpu segfaults on the WSL/CI default adapter → pinned to
lavapipe (auto-detected ICD; render tests `#[ignore]` so a normal `cargo test`
never touches it). egui stores a panel's *content* rect in `PanelState`, so the
column-width test verifies by *difference* (widen 100pt → panel moves 100pt),
cancelling the frame-margin offset. Render tests share the GPU + one embedded
Python interpreter, so `just docs-images` runs them serially and per-run scratch
dirs avoid a project-clobber race.

**Deferred (in BACKLOG):** a clean-licensed demo speech clip (fixture is a
synthetic tone), committing real doc images into the mkdocs site, and promoting
the pixel snapshot to a blocking gate once CI-native goldens are validated.

---

## 2026-07-02 — Design: automatable documentation-image pathway (headless, drift-tested)

Strand 2 (the 2026-07-01 entry) shipped hand-drawn region capture. The user then
expanded the goal into its real shape: **an automatable pathway to regenerate a
set of documentation images from a scripted, repeatable recipe** — so docs stay
in sync when the UI changes. Explicit requirements added in this session:

1. Select regions to capture **by name/category**, not only by hand-draw (keep
   hand-draw too).
2. **Standard window sizes** (presets) so screenshots share consistent
   proportions/look.
3. **Every signal subpane show/hideable** and **every annotation tier selectable
   in/out**.
4. **Scriptable, repeatable** recipes.
5. **Headless.**
6. **No drift** from the GUI users actually see.

### The load-bearing question: headless *and* faithful

Requirements 5 and 6 look opposed (a headless offscreen renderer is the classic
place drift creeps in). Resolution — **there is no separate doc renderer**:

- **Structural anti-drift.** Both the live app and the headless pipeline run the
  *identical* `SaddaApp::ui` through the *identical* egui + wgpu stack. Confirmed
  `eframe` 0.34 renders via wgpu (`egui-wgpu` on by default), and
  [`egui_kittest`](https://crates.io/crates/egui_kittest) with its `wgpu` feature
  renders through that same stack. The only things that can differ are
  environmental — fonts, theme, `pixels_per_point`, window size — which we pin to
  identical values in both paths.
- **Enforced anti-drift.** The doc images *are* `egui_kittest` snapshot goldens.
  CI re-renders and diffs them, so any UI change that would alter a documentation
  image **fails the build** until the image is regenerated and reviewed. This is
  exactly what kittest exists for; drift becomes a caught test failure, not a
  silent doc rot.
- **Headless falls out for free.** kittest renders offscreen via wgpu — a
  software Vulkan adapter (lavapipe) in CI, WSLg's adapter locally. No display
  server needed.

### Prior art

docs-as-code screenshots: Playwright/Puppeteer element screenshots + fixed
viewport; Storybook + Chromatic; `VHS` (terminal); and `egui_kittest` itself
(egui's own snapshot-test suite). The named-rect registry is a curated version of
an accessibility/testing tree (AccessKit, which eframe already enables).

### Architecture

- **Shared primitives** (driver-agnostic): per-subpane visibility; per-tier
  in/out; a per-frame **named-rect registry** (panes + lanes + composites like
  `signal-column` and `whole-window`); window-size presets; a pinned zoom so
  output pixels are deterministic.
- **A recipe** = an ordered list of *shots*: `{ size, project, bundle, visible
  panes, visible tiers, cursor/selection, region → output file }`.
- **Two drivers off the shared primitives:**
  - *Live (eframe)* — interactive/authoring. Actions run against the live window;
    capture via `ViewportCommand::Screenshot`. Best-effort/async (the script
    engine can't pump frames — it runs synchronously between them). This is the
    already-shipped hand-draw + the forthcoming named-capture menu.
  - *Headless (`egui_kittest` + wgpu)* — **the automation spine.** A step-driven
    harness owns the frame loop, so it can drive `SaddaApp` directly, run frames
    until DSP analysis settles, snapshot the named region, and write the PNG —
    all *synchronously* and deterministically. Drift-tested via snapshots.

### Feasibility & risks (checked)

- eframe→wgpu confirmed; kittest+wgpu shares it. **Low renderer-drift risk.**
- `egui_kittest` must match egui **0.34** (pin the dev-dep; fallback = bump
  egui/eframe to 0.35 — larger, deferred unless forced).
- CI needs a wgpu adapter → **lavapipe** (software Vulkan) for deterministic
  headless pixels; allow a small snapshot diff tolerance (as egui's CI does).

### Reconciliation with the 2026-07-01 entry

The user now wants **real GUI toggles** for every subpane + per-tier in/out. This
supersedes the 07-01 note that "the export dialog owns per-element include flags
(rather than force GUI changes)" and **absorbs** the parked "separable
structural-lane toggles" backlog item into this work. The interactive hand-draw
capture (shipped) and the named-capture menu remain as the manual/authoring tier
beneath the headless spine.

### Slice plan

- **S3 — Visibility & selection model** (shared): show/hide for *every* signal
  subpane (waveform, spectrogram, tier strip — none today — unified with the
  existing f0/formant/intensity/VAD/MFCC toggles) + per-tier in/out; a
  `visible_lanes()` accessor. GUI menus.
- **S4 — Named-rect registry + interactive named capture**: per-frame registry
  (panes + lanes + composites); "Capture ▸ *region*" submenu (live driver).
- **S5 — Standard size presets**: `View ▸ Doc size ▸` {1280×800, 1600×1000,
  1024×768} via `ViewportCommand::InnerSize`; zoom pinned to 1.0 in doc size.
- **S6 — Headless doc-render harness** (the spine): `egui_kittest` + wgpu binary/
  test driving `SaddaApp`, capturing named regions, writing PNGs.
- **S7 — Recipe API + in-repo recipes**: a shared **Python** primitive API
  (decided 2026-07-02 — recipes are Python; the user authors in Python and the
  app already embeds CPython) — `size`, `open`, `select`, `show_only`,
  `show_tiers`, `cursor`, `capture`, `quit`. `capture` accepts **either** a named
  region **or** an explicit pixel rect `(x, y, w, h)` — pixel rects are
  reproducible precisely because doc-mode pins size + zoom. The interactive
  hand-draw (S2) **echoes its final pixel rect** (info banner / console) so a
  hand-drawn selection lifts straight into a scriptable `capture(rect=…)`. A
  `just docs-images` regeneration target.
- **S8 — CI snapshot-diff gate**: lavapipe adapter + golden diffing.

### North star: scripted screencast + narration (future)

Logged, not scoped now — the fuller vision the user has carried since
[`devlog/2026-05.md`](devlog/2026-05.md) (the "auto-generated feature walkthrough
with synthesized voiceover, doubling as end-to-end UI testing" intake): script a
whole in-app workflow (create/record a sound → measure → annotate → …) and emit a
**screencast video with TTS narration**. Why it belongs here: a screencast is the
S6 headless driver rendering a **timed frame sequence** + an audio track, rather
than one snapshot per shot — so **S6 should be built to render sequences, not
hard-code single frames** (each "shot" = a settle-then-grab; a screencast = grab
every frame at a fixed fps). Additional pieces it needs, none built: a fixed-fps
frame-sequence recorder, `ffmpeg` muxing (frames + audio → mp4/gif), an audio
track (real playback capture and/or synthesized), and **TTS for the narration —
which is not yet on the roadmap** (see the new BACKLOG item). Kept as a north star
so nothing in S6/S7 forecloses it; not on the critical path for doc images.

### What this doesn't decide

Recipe format (Python vs data file — settle at S7); exact lavapipe/tolerance
tuning (S8); whether the live driver ever gains a full async action-queue or
stays hand-draw + named-menu only (revisit after the headless spine lands).

### Sources / references

- egui_kittest — egui's snapshot-testing harness: crates.io/crates/egui_kittest
- 2026-07-01 strand-2 entry (hand-draw capture; the primitives this builds on)
- `devlog/2026-05.md` — the auto-generated feature-walkthrough + voiceover intake
  (the screencast/TTS north star this pathway seeds)

---

## 2026-07-01 — Design session: figure export + GUI-region capture

The **publication-quality figure export** logged on 2026-05-25 (intake only,
gated on "once the visual elements are all developed") gets its design pass. The
session also split out a second, smaller strand the user raised alongside it.
This is a **design entry** — no code yet; it settles the route, architecture,
and slice plan, and reframes a premise the intake note had baked in.

### Two strands

1. **Publication figures.** A simple way to write out professional, journal-ready
   figures *from within the app* — the waveform / spectrogram / annotation-tier
   figure that is the staple of a phonetics paper.
2. **GUI-region capture.** Separately: write out a raster image of an arbitrary
   region of the GUI, for documentation/slides. No signal semantics — just a
   crop-and-save. Small; shipped **first** as a quick win.

### The gate had quietly changed shape (stale premise)

The intake note gated the exporter on "after Phase-3 GUI *overlay rendering*" —
meaning f0/formants composited **on** the spectrogram. That was never built.
Instead the GUI pivoted to a **stacked-lane** layout: waveform → spectrogram →
separate f0 / formant / intensity / MFCC / VAD / embedding lanes → tier strips
(`crates/app/src/main.rs:10899+`). So the gate's *premise* is stale, not failed:
the visual elements are all developed, just as sibling lanes rather than
overlays. Not a blocker.

### Prior art — specTeX sample studied

The user's own Praat exporter, **specTeX** (github.com/dbqpdb/specTeX), is the
style baseline. Studied its rendered sample (`examples/demo_document.pdf`): two
stacked panels sharing a time axis — thin black waveform (y = "Pressure (Pa)",
min/max labels + zero line) over a greyscale spectrogram (y = "Frequency (Hz)"
on the right); a **tier header row** (`p ɹ ɑː t` / `praat`) whose interval
**boundary lines extend down through both panels**; Computer-Modern serif with
cleanly typeset IPA; "Time (s)" along the bottom. Architecture: a Praat script
exports raster PDFs (waveform + spectrogram) + a `.tex` data file of TikZ
commands, assembled via `\specfigure{…}`. Deps: tikz/graphicx/calc/fontspec,
XeLaTeX/LuaLaTeX, Doulos SIL for IPA.

**Key reframe:** in Praat, that export-rasters-and-reassemble-in-LaTeX dance was
a *limitation Praat forced* (Praat can't draw the finished figure itself), not a
workflow the user prefers. Sadda has no such constraint — it can render the
finished figure directly. So "must this go through LaTeX/TikZ at all?" was
genuinely re-opened rather than inherited. Everything that makes the target look
good (crisp vector axes/ticks/tier-boxes/boundary lines, embedded greyscale
raster spectrogram, journal proportions) is achievable by **any** good vector
renderer. The *only* thing that strictly needs the LaTeX route is having figure
text match the surrounding document's fonts automatically; a direct SVG/PDF gets
very close by embedding a Latin-Modern-like serif + a Unicode IPA font (and can
outline glyphs to paths for a fully self-contained file).

### Standard problem & alternatives

Reproducible scientific-figure generation — vector export of a mixed
raster+vector plot. Phonetics prior art: Praat Picture→EPS/PDF, **praatpicture**
(R/ggplot, Puggaard-Rode), parselmouth+matplotlib, specTeX (TikZ). Route options
weighed: (A) **direct vector** SVG/PDF — one file, one click, no toolchain, drops
into LaTeX/Word/web; (B) **TikZ/LaTeX** — document-font-matched + hand-editable,
but multi-file + a LaTeX build; (C) **both off a shared IR**. The IR investment
is route-agnostic, so (C) dominates on flexibility for a bounded extra cost.

### Decisions locked

- **Architecture: a `FigureSpec` IR + pluggable serializers.** New
  `crates/engine/src/io/figure.rs`, mirroring the tabular exporter's
  `ExportBundle`/`ExportTier` → `to_csv`/`to_json` split (data-IR → serializer).
- **Two backends, both in this effort:** **SVG/PDF** (the easy default; PDF via
  an SVG→PDF step) **and** **TikZ** (`.tex` fragment + raster assets, specTeX
  integration model, for LaTeX-native tuning). Same IR feeds both.
- **Content: the whole signal column** (not the Python console) by default, with
  **per-element include flags on the `FigureSpec`** so a figure can differ from
  the screen (e.g. spectrogram without waveform). Defaults from current
  visibility.
- **Style: clean publication defaults**, with the IR carrying overridable style
  fields (colormap, palette, fonts, dimensions, bounds) — knobs surfaced to
  Python/GUI later.
- **Strand 2 (GUI capture) ships first.**

### Toggle reality (the requirement to confirm)

Audited whether the signal column is fully togglable today (so "export what's
shown" is user-controlled). It is **not, yet**: f0/formants/intensity/VAD are
View-menu checkboxes (`persisted.tracks.*`, visibility coupled to computation),
MFCC has "Show MFCC lane" (`persisted.mfcc.show`), embedding is shown iff a tier
is picked (`persisted.embedding.selected_tier_id`) — but **waveform, spectrogram,
and tier strips have no toggle** (always drawn), and there is **no single
"visible lanes" descriptor** — visibility is scattered across three
`PersistedState` fields plus the layout control-flow in `bundle_content_pane`.
Resolution (better than forcing GUI changes): the **export dialog owns its own
per-element include checkboxes** — including the always-on lanes — defaulting
from a new consolidating `visible_lanes()` accessor the app should have anyway.
Adding real GUI show/hide toggles for the structural lanes is a **separable** UI
enhancement, not part of this feature.

### The one real refactor

The spectrogram exists only as a **baked GPU texture**, and the colormap bake
(`power_to_db_normalized`/`colormap_bake`/`sample_colormap`) currently lives in
the **app** (`state.rs`), not the engine. Headless, three-surface export needs
that bake (or the raw STFT power matrix) available engine-side — so it moves into
the engine (where those pure-data functions arguably belong). Everything else the
exporter needs already exists as clean engine data: `PitchFrame`/`FormantFrame`/
`IntensityFrame` frame vectors (vector-friendly polylines), tiers, and a min/max
waveform envelope (drawn as a **vector** band — improving on specTeX, which
rasters the waveform).

### Slice plan

Strand 2 first, then strand 1 by capability; each slice ships engine + Python +
GUI + tests per the three-surface rule (from G1 on).

- **S2 — GUI region capture (first).** Rubber-band region select over the app →
  `ViewportCommand::Screenshot` → crop `ColorImage` to the rect (logical→physical
  px via `pixels_per_point`) → save PNG via file dialog. Reuses/un-gates the F12
  screenshot path (`debug::save_screenshot`, `main.rs:8900-8916`). GUI-only.
- **G0 — groundwork.** Move the colormap/spectrogram bake into the engine +
  expose the spectrogram raster/matrix; add `visible_lanes()` to the app.
  Bake-parity test. No user surface.
- **G1 — first shippable figure.** `FigureSpec` IR + **SVG** serializer for
  **waveform + spectrogram + tiers** (specTeX-parity core) + PDF via SVG→PDF;
  Python `export_figure(...)`; GUI "Export figure…" dialog with per-element
  include checkboxes (default from `visible_lanes()`) + format choice.
  Golden-SVG + round-trip tests.
- **G2 — TikZ backend.** TikZ serializer off the same IR + a standalone `.tex`
  wrapper for one-shot preview. Golden-`.tex` test.
- **G3 — measure lanes.** f0 / formants / intensity / VAD as stacked rows in the
  IR, both backends.
- **G4 — heatmap lanes + style knobs.** MFCC + embedding rasters; expose
  colormap/palette/font/dimension overrides across Python + GUI. Completes the
  "whole signal column" goal.

### Steelman & disconfirmers

- **Steelman for TikZ-only (path not fully taken):** the user already owns a
  mature TikZ renderer; document-font-matched, hand-editable output is a real
  advantage for camera-ready figures. Kept alive as the G2 backend — not
  discarded. Disconfirmer for direct-SVG-primary: if in practice every figure
  needs LaTeX-native font matching and hand-tuning, the SVG path is dead weight
  and we should have led with TikZ. Watch which backend the user actually reaches
  for once both exist.
- **Disconfirmer for the whole feature:** if the direct SVG figure can't hit the
  specTeX look convincingly (IPA typesetting, spectrogram fidelity), the
  "simple/easy" promise fails and the LaTeX route was the honest answer. G1's
  SVG sample against `demo_document.pdf` is the checkpoint.

### What this entry doesn't decide

Exact `FigureSpec` field names; the SVG font-embedding vs glyph-outlining choice
for IPA (decided at G1 against the specTeX sample); the SVG→PDF crate
(`svg2pdf`/`resvg` — evaluated at G1); whether structural-lane GUI toggles ever
get added (separate).

### Sources / references

- specTeX — Praat TikZ figure exporter; style baseline: github.com/dbqpdb/specTeX
  (`examples/demo_document.pdf`, `specTeX.sty`, `specTeX.praat`)
- 2026-05-25 intake entry (the deferred figure-export + bundle-rename items)
- praatpicture (R) — Puggaard-Rode; prior art for reproducing Praat-style figures

---

## 2026-07-01 — Git history rewrite: removed AI co-author trailers

Rewrote the **entire commit history** to strip the `Co-Authored-By: Claude …`
trailers that had been appended to commits (181 of 187). Rationale: Claude Code
is a development *tool*, not a project contributor — like Vim, a compiler, or a
browser, tools aren't credited as authors. The trailers had surfaced Claude in
GitHub's Contributors list (~183 commits). The README **"AI and human
acknowledgement"** section remains the appropriate, deliberate acknowledgement of
the tool's role; only the authorship credit was removed.

**Scope / consequences:**

- Every commit hash on `main` (and the active feature branches) changed. The 9
  tags — the published releases `v0.3.0` … `v0.5.0-app` and earlier — were **left
  pointing at their original, pre-rewrite commits**: those tags are protected from
  updates, and moving them isn't necessary anyway, because GitHub's Contributors
  graph is computed from the **default branch** — trailers on commits that are no
  longer on `main` don't count. Consequence: each release tag references a commit
  that is off `main`'s new line but byte-identical in content. GitHub Releases and
  the PyPI wheels are unaffected.
- The rewrite was content-preserving: every rewritten tree is byte-identical to
  its original (only commit messages changed), verified before pushing.
- Open PR branches were rewritten in the same pass so they stayed valid.
- Going forward, commits in this repo do **not** carry AI co-author trailers.

**⚠️ If you have an existing clone (another machine, a fork, an old checkout):**
the old and new histories cannot be reconciled by a normal `pull` (it will try to
merge two unrelated lines). Reset each checkout to the rewritten history:

```sh
git fetch origin
git checkout main && git reset --hard origin/main
# repeat the reset for any feature branch you had checked out,
# or simply delete local branches and re-create them from origin.
```

If in doubt, the cleanest fix is a fresh `git clone`.

## 2026-07-01 — Compositional-DSP perf spike: what composition actually costs

Follow-on to the in-app-help / signal-flow-explorer design. The open question
was whether a **unified compositional DSP path** (every lane a chain of
composable stage elements — enabling break-out of any intermediate for
annotation, chain editing, custom user functions, and the viz for free) is
performant enough to be the *production* whole-file path, not just an
interactive toy. Decided to measure before committing.

**Benchmark infra (new, kept).** Added `divan` (dev-dep) with its
`AllocProfiler` so benches report bytes-allocated alongside wall time — the
memory story matters as much as CPU here. `just bench` runs it. The throwaway
spike code lives behind an **off-by-default `spike` Cargo feature** so it is
never compiled into the app or wheel (holdout-code convention — see CONTRIBUTING);
`just bench` passes `--features spike` and verifies compositional == production
first. Each compositional variant is pinned output-equal to production by a test,
so every comparison is like-for-like (fast-correct vs fast-correct).

**Method.** Three MFCC implementations (production fused / naive whole-signal
materialisation / streaming per-frame boxed stages), plus streaming mirrors of
formants (LPC + root-solving) and pitch autocorrelation. Synthetic harmonic
signal, 10 / 60 / 300 s @ 16 kHz, release build, 100 samples each.

**Results (median time; peak = `max alloc`):**

| pipeline @300s | production | streaming-comp | naive-comp |
|---|---|---|---|
| MFCC time | 191 ms | 188 ms (1.00×) | 265 ms (1.39×) |
| MFCC peak mem | 25.6 MB | 25.6 MB (1.00×) | 115 MB (4.5×) |
| formants time | 1.482 s | 1.478 s (1.00×) | — |
| pitch time (fine-split) | 1.241 s | **4.80 s (3.9×)** | — |
| pitch time (fused stage) | 1.241 s | 1.249 s (1.00×) | — |

**The refined conclusion — composition's cost is not dispatch.** Dynamic
dispatch at frame granularity is free: 4 vtable calls per frame are nothing next
to the per-frame numeric work. The pitch autocorrelation profile *looked* like a
3.9× counterexample, but isolating it showed the penalty was entirely the
**decomposition**, not the boundary: a **fused** boxed stage (one `dyn` call per
frame around the intact hot loop, slice in, no copy) matched production to 1%.
What cost 3.9× was (a) per-frame heap allocation and (b) **splitting a
compiler-optimizable hot loop across a stage boundary and materialising its
intermediate** (the autocorrelation curve), which defeats register/SIMD reuse and
adds memory traffic.

So the rule:

- Wrap an **opaque heavy kernel** (FFT, LPC, root-solving) in stages → free
  (MFCC, formants).
- Wrap a **simple hot numeric loop coarsely** (one dispatch, buffers reused) →
  free (pitch-fused).
- **Finely split** that hot loop + **materialise** its intermediate → multiples
  slower (pitch fine-split 3.9×; the naive whole-signal MFCC's 4.5× memory is the
  same disease at the whole-signal scale).

**Decision-relevant upshot.** A unified compositional path *is* perf-viable —
conditional on three concrete constraints, not just "use streaming":

1. Stage boundaries go **around** expensive/opaque kernels, **never through**
   tight inner loops.
2. **Reuse buffers** — no per-frame allocation.
3. Whole-file production must **not materialise per-stage intermediates**.

Note (3) means the "break out every intermediate" desire — the viz/annotation
dream — *is* the expensive pattern (it's exactly what naive-comp does). Empirical
case confirmed for keeping break-out / viz on a **short window**, while whole-file
production fuses. This grounds the earlier two-level granularity idea: fine
primitives for reuse are fine, but *execution* must keep hot loops fused.

**Still not measured / open:** the streaming-state engineering for globally
non-streamable stages (Boersma Viterbi, pYIN HMM — the top_db floor was a trivial
O(N) instance) and the parity re-work of re-establishing librosa/kaldi/praat
goldens through composed structure (the passing equality tests show the
decomposition *can* be bit-faithful).

## 2026-06-30 — Design: in-app help + DSP signal-flow explorer

Design session unifying two backlog items — the general **GUI in-line
help/information system** (2026-06-28) and **communicating DSP parameter/method
choices** (2026-06-29). Decision: unify them at the *delivery* and *content-model*
layers, but not force a single content shape; the DSP tier gets a bespoke,
richer affordance on top.

**Standard problem.** This is *embedded / contextual in-product help* (tooltips,
info affordances, a contextual help panel, progressive disclosure). The DSP case
adds *scientific parameter documentation* (the numpydoc "Parameters + References
+ Notes" tradition) plus, in its ambitious form, an *explorable explanation* /
*signal-flow visualization*.

**Prior art surveyed.** Ableton Live's "Info View" (a fixed panel describing
whatever the cursor is over — zero-click, always-on) and jamovi / JASP
(contextual help panels in a scientific GUI) for the general layer; Praat's
per-dialog "Help → manual page" as the cautionary heavyweight (help as a
*destination*, not inline); numpydoc / scikit-learn as the gold standard for
*content* of scientific param docs; Bret-Victor-style reactive documents and the
interactive-FFT explainer genre for the visualizer. No production audio tool ships
an interactive stage-by-stage DSP explorer — this would be a genuine
differentiator, but built not borrowed.

**Decisions (four forks):**

1. **Two-tier content, unified only at delivery.** *Tier A (DSP params/methods):*
   engine-owned **structured descriptors** (label, units, valid range, default,
   effect-text, citation, and an `affects: [stages]` field) as the single source
   of truth, generated into Python docstrings + GUI tooltips/Info + a docs table.
   This satisfies the *three-surface* and *methods-cite-their-sources* house rules
   structurally rather than by discipline. *Tier B (general UI):* a lightweight
   GUI-side help-string catalog keyed by widget — freeform prose, no registry
   machinery (which would be over-engineering for one-liners). Rejected a single
   unified registry for everything as premature convergence; the honest shared
   thing is the delivery surface, not the content shape.
2. **DSP delivery = a staged signal-flow explorer.** A "visualize" toggle in the
   parameter window reveals the DSP chain as a **column of stages**, each showing
   its intermediate representation; changing a parameter re-renders **only the
   stages downstream of where it acts**. (User's concept; it fits Tier A exactly —
   see synergy below.)
3. **Full staged column, committed, phased** (not the cheaper final-output-only
   preview — see path-not-taken).
4. **MFCC first** — the longest, most linear, most teachable chain, and its
   stages are already factored in the engine.

**Architecture / key points:**

- **The param→stage map *is* the content model.** "Only affected stages update"
  requires each parameter to declare which stage(s) it drives — that's the
  descriptor's `affects` field. The reactive behavior falls out of Tier A; the
  two decisions reinforce.
- **Engine needs a "traced" compute path** emitting intermediates, not just the
  final matrix. `mfcc_with_params` (`crates/engine/src/dsp/mfcc.rs:1015`) is
  currently monolithic, but the stages are already factored into `stft.rs`,
  `windowing.rs`, `spectrogram.rs`, `mel_filterbank_general`, `build_dct` — so
  this is bounded plumbing to capture intermediates, not a math rewrite.
- **Three-surface, not GUI-only.** Once the engine emits `{framed, windowed,
  power_spectrum, mel, log_mel, cepstrum}`, Python/notebook users get the same
  staged arrays — real value lands in Phase 2, before any GUI visualizer exists.
- **Perf (house rule #2).** Visualize a **short window** (a few frames around the
  playhead/selection), not the whole file, and recompute only downstream of the
  changed stage. Live-running the full pipeline on a multi-hour file per slider
  tick would violate our own latency rule.

**Phasing (each phase stands alone and ships):**

- **P1 — descriptors (content).** Engine-owned structured descriptors →
  generated Python docstrings + GUI tooltips/Info text + docs table, plus the
  Tier-B UI catalog. Ships help *everywhere*, cheaply.
- **P2 — traced engine compute.** Intermediates API, exposed engine + Python
  (notebook value lands here).
- **P3 — GUI signal-flow column.** The "visualize" toggle, per-stage rendering,
  reactive downstream-only updates driven by P1's `affects` map, short-window,
  MFCC first.

**Path not taken / disconfirmer.** A live **final-output preview** (drag a param,
watch the final MFCC/pitch/formant plot re-render, no intermediate stages) would
give most of the "what does this do" value for a fraction of the work. We chose
the staged depth as the scientific-instrument differentiator and because the
primary user is deepening method understanding (dogfooding). Disconfirmer: if in
practice users only want "is my output better," the stage column is teaching
luxury and P3 should collapse back to a final-output preview.

*Open for the next session:* the exact descriptor schema (fields + Rust type),
the canonical MFCC stage list and how each stage renders, and the reactive
recompute structure. Backlog items 2026-06-28 / 2026-06-29 remain the trackers.

## 2026-06-29 — Preset registry polish: GUI save/delete, uniform param shapes, generalized schema

Three loose ends from the preset work, after an honest audit found them:

- **GUI save/delete user presets.** The GUI exposed only the registry *read*
  side (pickers + param editors); creating a named user preset was Python-only.
  Added a shared "Save current as preset…" dialog (id + title, surfaces
  invalid-id / built-in-collision errors) and a "Delete \"<id>\"" affordance —
  shown only for user presets (built-in ids gated in-memory) — wired into all
  three lane submenus (MFCC / f0 / Formants) via one `PresetTarget`-dispatched
  handler. Saving records `based_on` = the preset it was derived from and
  `faithful = false`; deleting resets the lane to that family's first built-in.
  Now all three surfaces (engine / Python / GUI) can both read and write presets.

- **Uniform Python param shapes.** `PitchParams` / `FormantsParams` only had a
  generic `for_method(...)`, while `MfccParams` had named per-reference
  constructors. Parallelized them: added `PitchParams.boersma/yin/pyin/swipe`
  and `FormantsParams.burg/autocorrelation` (taking each family's common
  analysis args with defaults, parallel to `MfccParams.librosa/kaldi/praat`),
  and gave `MfccParams` a matching `for_method`. All three now expose the
  identical surface — `for_method` + named constructors + `.replace(...)` +
  getters + `to_toml` — verified equivalent by test.

- **Generalized `preset-registry/SCHEMA.md`** from MFCC-only to all three
  domains: a shared top-level-fields section + per-domain `[params]` sections
  (MFCC / pitch `method`+`[params.config]` / formant) + a domain table.

Validation: engine 230 + app 116 tests, Python 31 preset tests green; no stub
drift; `fmt`/`clippy` clean; app boots cleanly. Remaining preset items are the
intentional backlog (f64 pipeline → faithful Praat preset; `htk()` preset;
real-Kaldi golden; the DSP-parameter-effects design session).

---

## 2026-06-29 — Formant presets (three surfaces) — item 6 complete

Closed out roadmap item 6 by adding **formant** presets, the lightest domain:
`FormantsConfig` already bundles the LPC method, so it *is* the preset payload
(no `PitchParams`-style wrapper needed). The generic `crate::preset` core +
the pitch template made this fast.

- **Engine:** serde + `PartialEq` + `Copy` on `FormantsConfig` / `LpcMethod`;
  `dsp/formant_preset.rs` with `impl PresetDomain for FormantsConfig` (subdir
  `formant`) + two built-ins at reference defaults (`praat-burg` /
  `autocorrelation`), both faithful. 4 tests.
- **Python:** `sadda._native.formant_preset` submodule — `FormantsParams`
  pyclass (`for_method` + getters + `.replace` + `to_toml`; `lpc_order`'s
  nested-`Option` left out of `replace` as a rare advanced knob), `FormantPreset`,
  store fns, `compute` (returns the same `FormantFrame` list as `formants`).
  `sadda.dsp.formants(audio, params=…)` dispatches; PROVISIONAL. (Made
  `PyFormantFrame` `pub(crate)` so the submodule can construct it.) 8 tests.
- **GUI:** unified the formant lane onto `tracks.formant_params: FormantsConfig`
  (`Copy`, so `MeasureTrackConfig` stays `Copy`), replacing the
  `formant_lpc_method` mirror + the `formant_count` field (values identical to
  `FormantsConfig::default()`; `formant_max_hz` stays separate as the
  display-only y-axis bound). Removed the `LpcMethodChoice` mirror. View ▸ DSP
  methods ▸ Formants is now a preset picker + Edit-parameters modal +
  "(modified)" flag — mirroring the f0 work.

**Item 6 done across pitch + formants + the generic core.** Validation: engine
230 lib tests, app 115 tests, Python 50 dsp-preset tests green; no stub drift;
`fmt`/`clippy` clean workspace-wide; app boots cleanly (old persisted state
migrates via `#[serde(default)]`). The whole MFCC→pitch→formants arc now shares
one `Preset<P>` / `PresetStore<P>` / `PresetDomain` core.

---

## 2026-06-29 — Generic preset core + pitch presets (three surfaces); item 6 pitch done

Extended the preset pattern from MFCC to **pitch** (roadmap item 6), first
**generalizing the registry** so the third/fourth domains don't duplicate it.
Decisions taken with the user: *pitch first, full stack*; *generic `Preset<P>`
core*.

### Generic core (`crate::preset`)

Extracted the MFCC store into a payload-generic registry: `Preset<P>` (id /
version / title / description / `based_on` / `faithful` / `reference` /
`params: P`), a `PresetStore<P>`, and a `PresetDomain` trait each param type
implements to declare its on-disk `subdir()` + code-sourced `builtins()`. MFCC
became `type MfccPreset = Preset<MfccParams>` / `MfccPresetStore =
PresetStore<MfccParams>` via `impl PresetDomain for MfccParams` — ~250 lines of
store code deleted, MFCC tests still green. The MFCC-specific `PresetLineage`
enum collapsed into a free-text `based_on: String` (lineage vocabularies differ
per domain: librosa/kaldi/praat vs praat/yin/pyin/swipe); the Python `based_on`
getter/ctor simplified accordingly (no stub drift).

### Pitch (engine + Python + GUI)

- **Engine:** serde + `PartialEq` + `Copy` on `PitchConfig` / `PitchMethod`;
  new `PitchParams { method, config }` (the pitch analogue of `MfccParams` —
  the tracker API takes method *separately*, so the preset payload bundles
  both) + `pitch_with_params`. `pitch_preset.rs`: `impl PresetDomain for
  PitchParams` (subdir `pitch`) + four built-ins at their reference defaults
  (`praat-ac` Boersma / `yin` / `pyin` / `swipe`), all `faithful` since
  `PitchConfig::default()` already matches the Praat/paper defaults. Pinned
  `PYin`'s serde name to `"pyin"` (not snake_case `p_yin`) to match the
  `voiced_pitch(method=…)` vocabulary. 4 tests.
- **Python:** `sadda._native.pitch_preset` submodule mirroring `mfcc_preset` —
  `PitchParams` pyclass (`for_method` + getters + `.replace(**kwargs)` over all
  16 knobs + `to_toml`), `PitchPreset`, store fns, `compute` (returns the same
  `(times, freqs, voicing)` as `voiced_pitch`). `sadda.dsp.voiced_pitch(audio,
  params=…)` now dispatches; preset surface PROVISIONAL. **`voiced_pitch(
  params=preset)` is bit-equal to `voiced_pitch(method=…)`.** 9 tests.
- **GUI:** unified the f0 lane onto a single `tracks.pitch_params: PitchParams`
  (engine type, now `Copy` so `MeasureTrackConfig` stays `Copy`), *replacing*
  the old `pitch_method` mirror + the three separate `f0_min/max/voicing`
  fields (their values were identical to `PitchConfig::default()`, so behaviour
  is preserved; y-axis bounds now read from the params). Removed the
  `PitchMethodChoice` mirror. View ▸ DSP methods ▸ f0 is now a **preset
  picker** + **Edit-parameters modal** (method ComboBox + min/max/voicing/
  frame/hop, plus method-specific advanced sliders shown for the active
  method); `pitch_preset_id` on `PersistedState` drives the menu label + a
  "(modified)" flag.

**Validation:** engine 226 lib tests, app 115 tests, Python 20 dsp-preset tests
green; no stub drift; `fmt`/`clippy` clean workspace-wide; app boots cleanly
(old persisted state without the removed `f0_*` fields migrates via
`#[serde(default)]`). *Remaining:* item 6 **formants** (the generic core + this
pattern make it a fast follow), and the live GUI-interaction check (boot- and
pattern-validated only).

---

## 2026-06-29 — MFCC preset registry: three surfaces (engine on-disk store + Python + GUI lane)

Branch `feat/dsp-method-diversity`. Built roadmap items **3–5** in one slice
(scope chosen with the user: all three surfaces). The earlier `MfccParams` /
`mfcc_with_params` unification (item 2) made presets *expressible*; this makes
them **persistable, named, user-extensible, and visible in the GUI**.

### Engine — on-disk preset registry (`crates/engine/src/dsp/preset.rs`)

- **serde on the param space.** Added `Serialize`/`Deserialize` to every
  `Mfcc*` enum + `MfccParams`. Unit enums → `rename_all = "snake_case"`; the two
  **data** enums (`MfccFilters`, `MfccLog`) → internally tagged (`tag = "kind"`).
  That required converting `MfccFilters::NMels(usize)` from a tuple to a struct
  variant `NMels { n_mels }` (breaking change to a public enum; rode the
  in-flight 0.4.0-follow-up window). Verified the `toml` 0.8 crate round-trips
  internally-tagged enums — it does (was the main technical risk).
- **`MfccPreset`** = `MfccParams` + provenance (`based_on` lineage, `faithful`,
  `reference`, id/version/title/description). One self-contained `<id>.toml`
  per preset — *no* directory-per-entry, because a preset has no data payload
  (the contrast with `refdist`/`model` registries, which bundle Parquet/weights).
- **`MfccPresetStore`** mirrors `RefdistStore`: `new`/`user_default`
  (`~/.local/share/sadda/presets/mfcc/`) + `list`/`list_user`/`get`/`save`/
  `delete`. Built-ins (`builtin_presets()`) are **code** (golden-tested), never
  written to disk; their ids are reserved (save rejects them) so the
  authoritative presets can't drift. `is_valid_id` guards against path
  traversal. New `EngineError::Preset`. 8 unit tests (round-trip incl. the
  mel-spacing arm, builtin-params-match-constructors, store CRUD, id validation).
- **Honesty (the DSP-diversity discipline):** `faithful` means "reproduces
  `based_on`'s golden *through `mfcc_with_params`* to tolerance" → librosa/kaldi
  `true`, **praat `false`** (its pipeline path is f32-approximate; faithful
  Praat still needs `mfcc(method=Praat)`, the dedicated f64 path).

### Python (`crates/python/src/mfcc_preset.rs`, `sadda.dsp`)

- `sadda._native.mfcc_preset` submodule (unstubbed, per the refdist/ml
  convention). `MfccParams` pyclass (`PyCalibration` template: `frozen` +
  `from_py_object` + `Clone`) with `librosa/kaldi/praat` constructors, getters,
  a `.replace(**kwargs)` override path (scalars/bools + enum strings; `n_mels`
  only on the n-mels bank), and `to_toml`. `MfccPreset` pyclass (`#[new]` +
  getters + `to_toml`). Store fns: `list_all`/`list_user`/`builtin`/`get`/
  `save`/`delete`/`store_root`/`compute`.
- `sadda.dsp.mfcc(audio, params=…)` now dispatches to the params pipeline (the
  wrapper is pure-Python so the flat native `mfcc` stub is untouched → **no
  stub drift**). Preset surface re-exported PROVISIONAL: `mfcc_presets`,
  `mfcc_user_presets`, `builtin_mfcc_presets`, `mfcc_preset`, `save_mfcc_preset`,
  `delete_mfcc_preset`, `mfcc_preset_store`, `MfccParams`, `MfccPreset`. (Classes
  re-exported raw — `@provisional` wraps `__init__`, which breaks PyO3
  construction, same as `sadda.refdist`.) 11 tests; **`mfcc(params=preset)` is
  bit-equal to `mfcc(method=…)`**.

### GUI — togglable MFCC heatmap lane + preset picker + param editor

Prompted by the user ("why can't we create a togglable display for MFCCs?") —
a fair challenge: I'd over-scoped it out. An MFCC result is a `(frames ×
coeffs)` matrix, structurally identical to the **embedding heatmap** lane the
app already renders, so it reuses that whole path (`normalize_embedding` /
`colormap_bake` / texture upload / cache-by-`==`). What's actually backlogged
is the deeper "*communicating parameter effects*" design, not a basic display.

- `MfccLaneConfig` in `state.rs` persists the active preset id + the engine
  `MfccParams` **directly** (now that it's serde+PartialEq+Clone — a 21-field
  mirror would be pure drift-prone duplication; documented the departure from
  the `*MethodChoice`-mirror convention) + colormap/normalization/c0-display.
- `build_mfcc_heatmap_texture` + `rebuild_mfcc_if_stale` + `mfcc_lane_pane`
  mirror the embedding-heatmap trio (synchronous; MFCC is cheap). c0 (energy)
  is orders larger than c1+, so it gets a 3-way display (`MfccC0Display`):
  **Separate** (default — on its own per-coeff scale, set apart by a small
  transparent gap), **Inline** (shared scale), **Hidden**; combined with
  per-coefficient z-score normalization this keeps the heatmap legible. Lane
  caption flags "(modified)" when params ≠ the named preset, and the c0 mode.
- **View ▸ MFCC** submenu: show toggle, preset picker (built-in + on-disk),
  colormap/normalization/c0 knobs, and **Edit parameters…** → a modal
  (`rubric_editor_window` pattern) editing all scalar/bool/enum knobs with
  DragValue/ComboBox; Apply writes back (invalidates cache), with an explicit
  "editing voids faithfulness" note.

`+ preset-registry/` (README + SCHEMA) alongside `refdist-registry`/
`model-registry` — leaner (local config, no tiered-governance CI; presets are
user-owned scalars, not redistributed corpora/weights).

**Validation:** engine 223 lib tests + 8 preset tests green; Python 33 dsp
tests green; stubs no-drift; `cargo fmt`/`clippy` clean across engine/python/
app; app compiles. *Not yet done:* live GUI run (compile- + pattern-validated
only — the lane copies a working lane verbatim, but I haven't launched it under
WSLg); migrate the pipeline to f64 so the Praat *preset* is faithful (item 2
backlog); item 6 (extend to pitch/formants).

---

## 2026-06-29 — DSP method diversity: named MFCC methods + GUI method pickers; unification design (in progress)

Branch `feat/dsp-method-diversity` (3 commits, gate green). Started from the
2026-06-27 DSP review, which found the MFCC code claimed to "match librosa
exactly" while actually using natural-log (not librosa's `10·log10`) and
symmetric-Hann/leading-edge framing — a chimera matching **no** published
definition. Fixing that honestly turned into a method-diversity build, then a
parameterized-pipeline + presets design.

### Shipped (committed, `just gate` green: 229 passed / 6 skipped)

**Named MFCC methods** (`engine/src/dsp/mfcc.rs`, `MfccMethod` enum). "MFCC"
is a family, not one algorithm; each variant is faithful to one reference,
validated against that reference's *own* output. Goldens + generator committed
under `crates/engine/tests/dsp/mfcc/` (CI needs no librosa/torch/parselmouth).

- **Librosa** (default) — `librosa.feature.mfcc` 0.11: Slaney mel + area norm,
  power, `10·log10` + 80 dB global floor, periodic Hann, `center=True` framing,
  ortho DCT-II. Validated max ≈5e-4.
- **Kaldi** — `compute-mfcc-feats`: DC removal, pre-emph 0.97, Povey window,
  pow2 FFT, HTK mel, unit-peak filters, natural-log, ortho DCT-II, cepstral
  lifter 22, snip-edges. Validated vs **torchaudio kaldi-compliance** (PyTorch's
  faithful Kaldi reproduction — *not* Kaldi-proper; real `compute-mfcc-feats`
  golden backlogged). Max ≈3e-3.
- **Praat** — `Sound: To MFCC…`: Gaussian window (`2×` analysis width), HTK mel,
  unit-peak filters, **un-normalised** DCT, c0 in column 0. **Approximate** —
  see below.

Three surfaces: engine + Python (`sadda.dsp.mfcc(method=…)`, default librosa) +
tests. The dropped chimera was a `stable`-tier behaviour change; rode the
in-flight `0.4.0`-follow-up breaking window.

**GUI DSP method selection** (`crates/app`): View ▸ DSP methods submenu —
f0 tracker (Boersma/AC/windowed-AC/YIN/pYIN/SWIPE′) and formant LPC
(Burg/AC); both were hardcoded before. Persisted, cache-invalidating,
default-preserving. New `PitchMethodChoice`/`LpcMethodChoice` in `state.rs`.

### Praat: confirmed-correct vs. residual (key reference for resuming)

Reverse-engineered from Praat source (`NUMhertzToMel2`, `Sound_to_MelSpectrogram`,
`MelSpectrogram_to_MFCC`, `NUMtriangularfilter_amplitude`):
- **Confirmed**: `NUMhertzToMel2 = 2595·log10(1+f/700)` = the HTK / O'Shaughnessy–
  Makhoul constant (same curve Kaldi uses, `1127·ln`). Unit-peak triangles
  (area-norm is commented out in Praat). Filter count `N = round((mel(Nyq)−100)/100)
  = 27` (swept: 26/28 far worse). Un-normalised DCT `c_m = Σ_j P_j·cos(πm(j+0.5)/N)`,
  `c0 = Σ_j P_j`. Framing `floor((dur−2·win)/hop)+1 = 46` frames. dB ref `4e-10`.
  **No pre-emphasis**.
- **Residual — RESOLVED to tolerance (see roadmap 1).** The c1/c2 error was
  *not* the window (Gaussian-2 confirmed via parselmouth bisection; window-count
  swept) — it was (a) missing `_Spectrogram_windowCorrection` (÷ window
  mean-square → c0) and (b) f32 underflow on near-empty high filters in Praat's
  un-normalised dB DCT. Fixed by adding windowCorrection and computing in f64.
  Remaining ~10/~20 is irreducible FFT-library noise (realfft vs NUMrealft on
  ~1e-30 filter powers), now documented; `MfccMethod::Praat` = faithful-to-
  tolerance, test runs.

### Design: one parameterized pipeline + presets (feasibility CONFIRMED; engine BUILT additively)

**UPDATE (same session):** the engine half is now **built and committed**
(`feat(dsp): unified MfccParams pipeline + reference presets`). `MfccParams`
exposes every knob; `MfccParams::librosa/kaldi/praat` are presets; one
`mfcc_with_params()` pipeline runs them. Proven: `mfcc_with_params(preset)`
reproduces the librosa + Kaldi goldens to tolerance *and* agrees bit-for-bit
with `mfcc(method)` (agreement test). The unification is real, not just
designed. (Found en route: Kaldi triangulates filters linearly in *mel*,
librosa/Praat in *Hz* — the `triangle_in_mel` knob.) Additive: the enum
dispatch still calls the dedicated fns; collapsing it is mechanical (step 2
below, now mostly done).


User goal: *set the parameters that can vary, offer presets by authoritative
reference (Praat/librosa/Kaldi/HTK) or user-defined, and "select a preset then
modify individual parameters."* Distinguish "the algorithm" from "the reference's
default options."

**Feasibility analysis (the key result): yes, it unifies cleanly.** All
references share one skeleton — `frame → window → (pre-emph/DC) → FFT → power →
mel filterbank → log → DCT → lifter` — and differ only by a per-stage scalar or
enum, never an incompatible structure:

| stage | librosa | Kaldi | Praat | HTK | knob |
|---|---|---|---|---|---|
| window length | 25 ms | 25 ms | 2× | 25 ms | scalar |
| window fn | periodic Hann | Povey | Gaussian | Hamming | enum |
| DC / pre-emph | no/0 | yes/0.97 | no/0 | no/0.97 | bool+scalar |
| framing | center zero-pad | snip-edges | snip-edges(2×win) | snip-edges | enum |
| FFT size | =win | next pow2 | next pow2 | next pow2 | enum |
| mel scale | Slaney | HTK | HTK | HTK | enum |
| filter spec | n_mels/[fmin,fmax] | n_mels | first/step mel→derived N | n_mels | enum (2 modes) |
| filter norm | area (Slaney) | unit-peak | unit-peak | unit-peak | enum |
| log | 10log10+80dB floor | natural ln | 10log10 / 4e-10 | natural ln | enum+scalars |
| DCT norm | ortho | ortho | none | sqrt(2/N) | enum |
| lifter | 0 | 22 | 0 | 22 | scalar |
| power scale | raw | raw | 2/(nfft·n) | raw | enum |
| exclude Nyquist bin | no | yes | no | — | bool |

(Note: librosa "framing=center" is the only non-snip-edges; Kaldi vs Praat
differ by window fn + 2× length, not framing.) The faithful reproductions
already built *become* the validated authoritative presets + regression tests.
Caveats: a few knobs are enums not sliders; filter-spec has two modes; any combo
*computes* but only reference-matching ones are golden-validated — editing a
reference preset honestly makes it "custom (based on X)", voiding faithfulness.

**Decisions locked**: user-defined presets → **on-disk registry** (TOML + schema,
alongside `model-registry`/`refdist-registry`). Preset-then-edit is the core
interaction (requires `MfccParams`, not the opaque enum). Same pattern then
extends to pitch/formants.

### Roadmap (resume here)
1. ✅ **Praat** — now validated to tolerance (was approximate). Added Praat's
   `_Spectrogram_windowCorrection` (÷ window mean-square, fixes c0) and moved
   the whole path to **f64** (FFT included), cutting the residual from
   (c0 165, c1+ 54) to (c0 ~20, c1+ ~10, ~0.3% typical). The remaining gap is
   *irreducible* FFT-library noise — Praat's un-normalised dB DCT sums
   `10·log10` of every filter, so near-empty high filters (~1e-30) amplify
   sub-1e-15 realfft-vs-NUMrealft differences. Bit-exact would need Praat's FFT.
   Bisected via parselmouth (`mel(100)`→27 filters, Gaussian-2 confirmed,
   triangle-in-Hz). Test un-ignored (c1+ < 15, c0 < 25).
2. ✅ **`MfccParams` + `mfcc_with_params`** general pipeline + presets — built,
   golden-validated, agrees with the enum. Dispatch **collapsed** for Librosa +
   Kaldi (now route through `mfcc_with_params`; dedicated fns deleted). *Still
   pending (backlogged):* migrate the pipeline to **f64** so the Praat *preset*
   matches the dedicated f64 `mfcc(Praat)` path, then route Praat through it too;
   and an `htk()` preset (blocked on a power/magnitude knob + an HTK golden).
3. ✅ **On-disk preset registry** — built (`dsp/preset.rs`); serde on all
   `Mfcc*` enums + `MfccParams` (incl. the `NMels` tuple→struct refactor for
   internal tagging); `MfccPresetStore` + built-in/user split + `<id>.toml`
   format + `preset-registry/` schema. See the 2026-06-29 top entry.
4. ✅ **Python** — `MfccParams`/`MfccPreset` pyclasses + preset store fns +
   `mfcc(audio, params=…)` dispatch + `.replace()` override; 11 tests; no stub
   drift.
5. ✅ **GUI** — togglable MFCC heatmap lane (reuses the embedding-heatmap path)
   + View ▸ MFCC preset picker + modal per-parameter editor. The backlogged
   piece remains the deeper *communicating parameter effects* visualization.
6. ✅ **Extend** the params+presets pattern to pitch + formants — DONE.
   - ✅ **Generic core** (`crate::preset`: `Preset<P>` / `PresetStore<P>` /
     `PresetDomain`) — MFCC refactored onto it; `based_on` now free-text.
   - ✅ **Pitch** — `PitchParams` (method+config) + `pitch_preset.rs` builtins +
     Python + GUI (f0 lane unified onto `pitch_params`; preset picker + editor).
   - ✅ **Formants** — `impl PresetDomain for FormantsConfig` + `formant_preset.rs`
     builtins + Python + GUI (formant lane unified onto `formant_params`; preset
     picker + editor). All three DSP families now share the generic core.

Backlog updated with: real-Kaldi golden, MFCC-in-GUI, DSP-parameter-communication
design, GUI in-line help, plus the 2026-06-27 review items.

## 2026-06-21 — Python API ergonomics: three papercuts from a live-API probe

A test run of the live Python API surfaced three discoverability footguns,
all fixed cleanly (breaking, on the heels of 0.4.0) on `fix/python-api-ergonomics`.

1. **`Audio.mono()` returned a raw ndarray.** It read like "give me a mono
   `Audio`" but handed back samples, so the result couldn't flow back into the
   `dsp.*` functions that take an `Audio`. Now returns a single-channel `Audio`
   (new `Audio::to_mono()` in the engine; PyO3 `mono()` wraps it); reach the
   samples via `audio.mono().samples`. Worth noting the original footgun was
   half-illusory — every `&Audio`-taking `dsp.*`/`clinical.*` function already
   mono-mixes internally (`audio.mono_samples()`), so you rarely need `.mono()`
   before them at all. The docstring now says so.

2. **`Ltas` mixed methods and attributes with no stated rule.** `levels_db` /
   `bin_hz` / `sample_rate` are attributes; `slope(...)` / `tilt(...)` /
   `alpha_ratio()` are methods. The rule is principled (stored data = attribute,
   band-derived scalar = method) but undocumented, and parameterless
   `alpha_ratio()` made it look arbitrary. Documented the convention on the
   class docstring; kept `alpha_ratio` a method (it computes over a fixed 1 kHz
   split, consistent with the other measures).

3. **`schema_version` was callable, despite reading like a value.** Replaced the
   `sadda.version()` / `sadda.schema_version()` functions with Pythonic module
   constants `sadda.__version__` (str) and `sadda.SCHEMA_VERSION` (int),
   computed once at import from the native engine. The native `_native.version`
   / `_native.schema_version` functions stay as the internal value source.
   Constants carry no stability tier — the `@stable`/`@provisional` machinery
   decorates callables/classes, not plain values.

Call sites updated: `test_corpus`, `test_provenance`, `test_stability` (incl.
the PyO3-registry probe, repointed at `_native.load_wav`), and the release
workflow's `CIBW_TEST_COMMAND` smoke test. Stubs regenerated; full `just gate`
green (228 passed, 6 skipped).

---

## 2026-06-20 — Pane focus + annotation navigation (keymap, round 2)

Follow-up to the home-row keymap (merged the same day), adding keyboard
annotation work and console focus — still aimed at mouse-free scanning.

**Annotation navigation (its own keys, deliberately separate from the timeline
selection keys so the two don't fight).** The upper-right row drives a
*current-annotation* highlight: `u`/`i` = previous/next annotation on the
current tier, `o`/`p` = previous/next tier. It reuses the existing
`selected_annotation` highlight as the "current" annotation (so a click and the
keys share one concept) and only pans the view to keep it visible — it does
**not** move the timeline cursor/selection. `y` ("grab") is the deliberate
bridge: it pulls the current annotation into the timeline selection + cursor
(via the new `Timeline::set_selection_range`), so playback (`d`/`s`/`f`) then
acts on it. `Enter` edits the current annotation's label when one is
highlighted, else falls back to committing the timeline selection; `Esc` (or any
home-row cursor move) clears the highlight, back to commit-mode.

**Console focus (`Shift+↓` / `Shift+↑`).** We collapsed the originally-planned
three-pane focus stack to what actually changes behavior: the Python console vs.
everything else. `Shift+↓` opens + focuses the console; `Shift+↑` leaves it. The
"all keys pass through to the console" requirement falls out of the existing
text-edit guard (every shortcut already skips while a text field is focused), so
the console is a clean slate for a future Vim/Emacs mode. `Shift+↑/↓` are
reserved globally — consumed before the panels render — so they move focus even
from inside the console editor (which would otherwise eat them for line-select).
A ring marks the focused console.

**Why hybrid, not modal.** Playback, view scroll/zoom, and bundle nav stay
global; only annotation nav got new dedicated keys. Dedicated keys (vs. routing
the home row by focus) mean annotation nav and cursor movement are live at once
with no mode-switching — the smoother flow for actual annotating.

Also carried `Timeline::set_selection_range(start, end)` (the selection analogue
of `set_view_range`) onto this branch. Engine `step_id` helper unit-tested;
cheatsheet updated.

---

## 2026-06-20 — Two-handed, layout-independent keyboard navigation

Reworked the desktop transport/navigation keymap into a mouse-free, two-handed
home-row scheme, driven by real dogfooding (an L2 sound-file audio survey). Goal:
audition and scan a recording without touching the mouse.

**The keymap.** Left hand = playback, right hand = move, lower-right = view.

- **Left hand (playback):** `a` stop, `s` play-left-of-focus, `d` play/pause
  (selection-or-view), `f` play-right-of-focus; `Shift`+`s`/`d`/`f` loops. "Focus"
  resolves to the selection if a real span is selected, else the cursor. `Space`
  aliases `d`, `Esc` aliases `a`. The loop modifier moved from Ctrl → **Shift**.
  The old `,`/`.`/`[`/`]` directional keys are gone (subsumed by `s`/`f`).
- **Right hand (cursor/selection):** `h` rec-start, `j` view-start, `k`/`l`
  smooth glide left/right (frame-driven while held, with a speed ramp), `;`
  view-end, `'` rec-end. The **modifier picks the target**: bare = cursor,
  `Shift` = selection start, `Alt` = selection end (seeded at the cursor when no
  selection exists). The view follows the moved point off-screen.
- **Lower-right (view):** `n`/`m`/`,`/`.` scroll (view→start, pan-left, pan-right,
  view→end); `Shift` over the same keys zooms (fit-all, out, in, zoom-to-selection),
  reusing the wheel's 1.2 factor and the arrows' quarter-window pan step.
- **Bundles:** `q`/`w`/`e`/`r` = first/previous/next/last.

**Layout independence.** Every positional binding matches on egui's
`Event::Key.physical_key` (US-QWERTY position), not the logical key. A Dvorak/
AZERTY typist gets the same hand *shape* — the action follows where the key sits,
not the character their layout types. A small `consume_physical_key` helper
mirrors egui's `consume_key` but matches the physical key (falling back to the
logical key when the backend reports none, e.g. web) with exact modifier matching
(so bare / `Shift` / `Alt` drive three different actions on the same key). The
held `k`/`l` glide is tracked from raw key events because egui's `keys_down` is
logical-only.

**Three surfaces.** Per the project norm, the navigation primitives became a
real API, not just keybindings. The pure state moved out of the app into the
engine as `sadda_engine::Timeline` (cursor + view + selection), with a
**move-to (absolute) / move-by (relative)** pair for each action
(`set_cursor`/`move_cursor_by`, `set_selection_start`/`move_selection_start_by`,
`set_view_start`/`scroll_by`, `set_view_range` for fit/zoom-to-selection, …).
The app re-exports it as `TimelineState` (zero churn at call sites); Python gets
`sadda.Timeline` (provisional). Unit tests live with the engine type; a pytest
suite covers the Python surface. New docs page: [Keyboard cheatsheet](cheatsheet.md)
(all navigation + annotation hotkeys).

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
