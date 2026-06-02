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

- [ ] Display selection timestamps — show the boundary times (start + end) of the current selection — _added 2026-06-01_
- [ ] Refine / reorganize the `annotation-cycle.md` docs page — structure + flow need careful attention; revisit in the coming days (draft is uncommitted in the working tree) — _added 2026-06-01_
- [ ] Check how we handle stereo / multi-channel WAV files — verify behaviour end to end (down-mix vs per-channel; loader, waveform, DSP, playback) — _added 2026-06-01_
- [ ] Enable sound file conversion / extraction — import non-WAV audio (convert to WAV) and/or extract audio from other containers/formats on ingest — _added 2026-06-01_
- [ ] f0 **octave-down errors** in `windowed_autocorrelation` (the app's default measure-track f0): pure 200 Hz→100, 150 Hz→75 when `2·period ≤ max_lag` — the window-correction `r_a/r_w` boosts subharmonics. PRE-EXISTING (proven not the FFT-autocorr change). Fix options: default the measure track to pYIN/SWIPE, or add octave-cost / Viterbi path terms to the Boersma-style tracker. — _added 2026-06-01_

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

- [ ] **Add a screenshot to the README** (mechanical). Capture under WSLg during
  a run, commit the asset (`assets/` or `docs/`), add a markdown image to
  `README.md`. (raised 2026-06-01)
