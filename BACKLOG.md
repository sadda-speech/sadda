# Backlog

The running development queue for sadda. Companion to `DEVLOG.md` (which records
what *was* built); this records what we *might* build next.

**Capture:** type `/backlog <idea>` any time — it appends to **Inbox** below and
returns to the current task without discussion.
**Groom:** on request ("groom the backlog") — triage Inbox items into the
sections below, drop dead ones, and pull the next thing to work on.

Status: `[ ]` open · `[~]` in progress · `[x]` done (move to DEVLOG when shipped).

---

## Inbox

_Raw captures land here; groomed into the sections below on request._

- [ ] Refine / reorganize the `annotation-cycle.md` docs page — structure + flow need careful attention; revisit in the coming days (draft is uncommitted in the working tree) — _added 2026-06-01_
- [ ] Check how we handle stereo / multi-channel WAV files — verify behaviour end to end (down-mix vs per-channel; loader, waveform, DSP, playback) — _added 2026-06-01_
- [ ] Enable sound file conversion / extraction — import non-WAV audio (convert to WAV) and/or extract audio from other containers/formats on ingest — _added 2026-06-01_
- [ ] **VAD regression test on real speech** (ORT-gated) — vendor a short licensed speech clip + assert `vad()` detects it (max prob ≫ 0). The existing test only ran on *silence*, which is why the missing-Silero-context bug (VAD ~0 for everyone) slipped through. — _added 2026-06-03_
- [ ] **App-side auto-discovery of a pip-installed `onnxruntime`** — locate `…/onnxruntime/capi/libonnxruntime.so*` from an installed `sadda[ml]` and set `ORT_DYLIB_PATH` automatically, so dev builds find ORT without a manual export (the wheel already does this for the Python side; the app only auto-finds a packaged `<exe-dir>/onnxruntime/` sidecar today). — _added 2026-06-03_
- [ ] **Corpus fetch passthrough** (dataset analogue of the `hf://` model fetch) + a **prep stage** that formats a fetched corpus for Sadda: convert audio to WAV if needed, ingest existing annotations into tiers (TextGrid/CSV/dataset labels → interval/point tiers), build bundles. — _added 2026-06-03_
- [ ] Annotation tier navigation/activation when there are more than 9 tiers (digit keys 1–9 only cover the first nine) — _added 2026-06-03_
- [ ] Make the script panel resizable — _added 2026-06-03_
- [ ] Reorder annotation tiers — _added 2026-06-02_
- [ ] Multiple selection for points/boundaries; Mouse2 (right-click) context menu for points/boundaries (e.g. move left 0.3s, move to nearest zero-crossing) — _added 2026-06-02_
- [ ] Auto-generate demo screenshot, run at each release — _added 2026-06-02_
- [ ] **Custom / interactive aggregate view** — the *imperative* counterpart to P3's query-driven concordance: hand-pick regions across files into an editable "working aggregate" ("add this selection to the aggregate"). Design discussed 2026-06-02 (deferred — user wants to live with P3 first to find the right abstractions). Framing from that conversation: a **persisted ordered "clip-set"** (region refs `(bundle, start, end, label?)`); **two populators, one set** (manual "add selection" + P3's query redirected to append); refactor `build_concordance` into split *select-clips* / *concatenate* steps so it becomes the "flatten/export" terminal for both. Three open forks: (1) **object model** — dedicated clip-set table vs reuse ordered targets vs ephemeral scratch buffer; (2) **rendering** — materialize-and-cache (rebuild bundle on edit, reuse P3 + perf layer; v1) vs truly-virtual EDL (synth in-memory buffer, new render hook, flatten on export); (3) **layout** — main view + dockable side panel vs two windows vs dedicated split workspace. Prior art: EDL/NLE timelines, concordancer KWIC list, ELAN multi-file search. The P1/P2 in-memory `Arc<Vec<f32>>` envelope is what makes virtual render plausible. — _added 2026-06-02_

- [ ] **`just gate` stubs check is unfriendly pre-commit** — the `stubs` recipe does `git diff --exit-code` vs HEAD, so it flags a legitimate *uncommitted* stub change as "stale" (you must `git add` first). Make it compare a fresh regen against the working tree (idempotence check) instead, so it passes pre-commit when stubs are current-with-code and still mirrors CI on a clean tree. — _added 2026-06-02_

---

## Performance & responsiveness

- [ ] **Speed up bundle switching across a corpus** (HIGH — user: "responsiveness
  is really critical for this to be useable as intended"). Profile the switch
  path in `crates/app` first (audio load · f0/intensity/spectrogram recompute ·
  egui render of waveform/spectrogram/tiers); suspect DSP recompute on every
  visit dominates. Levers: cache derived signals per bundle (reuse the B3 Parquet
  sidecar machinery) · async/progressive paint (waveform now, spectrogram/f0
  when ready) · downsampled waveform + tiled/cached spectrogram · virtualize tier
  render · prefetch neighbours. _Measure before optimizing._ (raised 2026-06-01)
  - **Done so far (0.3.x):** FFT autocorrelation (~700×), dev-profile DSP
    opt-level, P1 per-bundle signal cache, P2 async DSP. Switching within a
    working set is now snappy; remaining levers above are for very long files.

- [ ] **Windowed reader + multi-resolution peak cache** (DEFERRED — planned for
  *if/when long files become an issue*; full design in DEVLOG 2026-06-02). Decode
  only the visible window instead of whole files; render the waveform from a
  persisted binary peak cache (min/max(/rms) at geometric zoom levels). Decouples
  peak RAM from file length. **Low-RAM relevance (the real argument):** on an
  older/4 GB machine this is what lets a long file open *at all* without
  splitting — more valuable on the least-capable hardware than on a workstation.
  Slices W0–W3 (guard → peak cache → windowed reader → window-driven detail
  views) in the DEVLOG entry. (deferred 2026-06-02; interim mitigation =
  warn-and-split-on-ingest, shipped)

## Views

- [ ] **Aggregate "concatenated-timeline" view over a corpus** (design-laden;
  user sees this as a primary exploration surface, not a viewer). Show all
  selected targets/intervals (e.g. all `f`-tier `='f'` tokens across bundles) as
  one continuous sound file in the *existing* waveform/spectrogram/tier view,
  with a **source-divider tier** (interval per segment, labelled with source file
  + original time for jump-back) and **context tiers remapped by offset** (see
  each token *with* its annotations). Big win: reuses the existing renderer.
  - **Implementation fork:** (A) **materialized** — extract each segment's audio
    (Audio slice + hound WAV writer), concat, `add_bundle`, build divider +
    remapped tiers; simplest, renders like any bundle, but a copy (no write-back).
    (B) **virtual** — a projection `[(bundle,start,end)]` + time-map, lazy
    per-segment render; live + editable write-back, but more plumbing.
  - **Key decision:** read-only inspection → materialized v1 (rec); edit-write-
    back → virtual. Subtleties: silence at seams for playback; per-segment DSP to
    avoid seam smear.
  - **Unifying insight:** the criteria RoI query is one object with three faces —
    the aggregate view's segment list, a notebook entry's subject, and the target
    generator. Strong synergy with the S7 lab-notebook (it's the instrument for
    the explore→note flow). Integration: "add to notebook" from the view
    (target_type prefilled + back-link); measure-over-the-set → a `measurement`
    note. Do a design session first. (raised + refined 2026-06-01)

## Annotation suite tweaks

- [ ] **Make the notebook `kind` set editable / reconsider the trio.** S7 fixed
  it to `observation | measurement | decision` via a CHECK constraint (not
  user-editable). If the trio feels wrong after real use, revisit names/extent;
  cheap change. (surfaced 2026-06-01 while testing the lab-notebook)

## Docs

_(empty)_
