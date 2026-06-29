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

- [ ] **GUI in-line help/information system** — make it easy for users to get info about a button, window, function, operation, etc. (e.g. hover tooltips / `?` affordances / contextual help panel) — _added 2026-06-28_
- [ ] **MFCC `method="kaldi"`: validate against real Kaldi `compute-mfcc-feats`** — currently validated against torchaudio's kaldi-compliance (PyTorch's faithful reproduction), not Kaldi-proper. Regenerate the golden from an actual `compute-mfcc-feats` run when a Kaldi toolchain is available, to drop the one-step-removed caveat. (`crates/engine/tests/dsp/mfcc/`) — _added 2026-06-28_
- [ ] **MFCC method selection in the GUI** — once MFCC has a GUI lane/feature (it has none today), add an `MfccMethod` picker alongside the View ▸ DSP methods f0/formant pickers. — _added 2026-06-28_
- [ ] **MFCC `htk()` preset + power/magnitude knob + f64 pipeline** — HTK-proper MFCC needs (a) a magnitude-vs-power spectrum knob (HTK USEPOWER=F uses magnitude; pipeline is power-only) and an HTK DCT/lifter, and (b) an HTK golden to validate. Also: migrate `mfcc_with_params` to f64 so the Praat *preset* matches the dedicated f64 `mfcc(Praat)` path, then route Praat through the unified pipeline too (completes the dispatch collapse). — _added 2026-06-29_
- [ ] **Explore FFT method/backend options** — the Praat MFCC residual is irreducible FFT-library noise (realfft vs Praat's NUMrealft amplified by the un-normalised dB DCT on ~1e-30 filter powers). Options: vendor/port Praat's `NUMrealft` (would let Praat MFCC go byte-exact), expose a selectable FFT backend, or evaluate alternatives (rustfft variants, pocketfft port) for accuracy/perf. Relevant to any pipeline where FFT-library differences surface. — _added 2026-06-29_
- [ ] **Design session: communicating DSP parameter/method choices** — workshop a more organized way to present DSP algorithm parameter choices and convey the *effects* of each choice (e.g. visualizations, side-by-side previews, inline explanations of trade-offs). Spans pitch/formant/MFCC method pickers + their numeric params across Python + GUI. — _added 2026-06-29_

- [ ] **Rhythm metrics** — speech rhythm/timing measures (e.g. %V, ΔC/ΔV, VarcoV/VarcoC, nPVI/rPVI) computed over interval tiers — _added 2026-06-05_
- [ ] Fix table alignment in docs Quickstart → "Query annotations" → `print(df.head())` output — _added 2026-06-04_

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

- [ ] **Broaden the agreement engine beyond Cohen's κ — measurement-level-aware metrics** — S5 reports unit- and frame-based paradigms but both use Cohen's κ (Fleiss deferred). Survey + design alternatives keyed to the *measurement level* of a tier's ratings, with **Krippendorff's α** as the unifying default (any number of raters, missing data, and a configurable distance metric → distance-weighted boundary agreement, which κ can't express). By level: **categorical** — Cohen/Fleiss κ, Scott's π, Krippendorff nominal α, Gwet's AC1/AC2 (addresses the "kappa paradox" on skewed marginals); **ordinal** (ranked scales) — weighted κ, Krippendorff ordinal α, Kendall's W; **interval/numeric** (continuous measurements) — ICC, Krippendorff interval/ratio α, Bland–Altman, Lin's concordance correlation; **boundary/unitizing** (segmentation) — Krippendorff's unitized α (uα). Canonical reference: Artstein & Poesio 2008, "Inter-Coder Agreement for Computational Linguistics." Do a `/broaden` or design session first — the engine has no measurement-level concept yet (tiers carry controlled vocab but not a scale type). — _added 2026-06-28_

- [ ] **MFCC / pitch "matches library X exactly" over-claims** (from 2026-06-27 review) — correct close-but-not-identical parity claims. (1) pitch `PitchConfig` default ceiling is 500 Hz but the docstring says "defaults match Praat 6.x" — Praat's default is 600 Hz (`pitch.rs:118-119,190`); fix the value or the claim. (2) pYIN `pyin_bins_per_semitone=20` while the field doc admits librosa's default is 12, and the module says "matches librosa" (`pitch.rs:163,177-181,201`); the pYIN HMM is a reimplementation, not librosa-identical — soften. (3) MFCC HTK-mel toggle / librosa-faithful mode is deferred — fold into the MFCC fix decision (a true-librosa-parity mode would need 10·log10 + periodic-Hann + center framing + a golden fixture). — _added 2026-06-28_

- [ ] **HNR: fix the Boersma-1993 citation + add the autocorrelation variant** — `clinical.rs:253-255` attributes the cross-correlation HNR `(cc)` to Boersma 1993, which is the *autocorrelation* harmonicity paper; cite the `(cc)` method correctly. Only `(cc)` is exposed — add the `(ac)` variant as the DSP-method-diversity partner. — _added 2026-06-28_

- [ ] **Missing primary citations (cite-every-method gap)** — CPPS cites only Praat, not Hillenbrand & Houde 1996 / Heman-Ackah et al. 2003 (`clinical.rs:360-369`); LTAS slope/tilt/alpha have no primary refs (`ltas.rs`). Also verify CPPS default `quefrency_smooth_s=0.00015` (~2 samples @ 16 kHz) against the Praat oracle — looks small and directly affects peak prominence. — _added 2026-06-28_

- [ ] **Docs accuracy pass (from 2026-06-27 review)** — (1) README Validation section (`README.md:204-210`) over-claims "validated against authoritative references"; reconcile to the accurate weaker claim at README.md:96-97 (Praat-validated where an oracle exists, clean-room/provisional otherwise). (2) Add a provisional caveat for AVQI/ABI in quickstart's clinical section (`quickstart.md:116-133`). (3) CHANGELOG never logs the YIN/pYIN/SWIPE' trackers though they exist (`pitch.rs:852/1003/1488`) — add. (4) `CONTRIBUTING.md:4` "next-generation" positioning → neutral framing matching README/docs. (5) `docs/index.md:45-47` links a May entry to the live DEVLOG.md; repoint to `devlog/2026-05.md`. (6) Internal "B3" slice code leaks into `README.md:104` / `quickstart.md:162`. — _added 2026-06-28_

- [ ] **Burg "no windowing" docstring oversold + stray doc comment** — `lpc.rs:10-13` sells Burg's avoid-windowing advantage, but `formants.rs:124-129` Hann-windows the frame before calling it (Praat does too); soften the framing. Separately, a doc-comment paragraph about frequency-domain resampling sits above `pub fn hfno` (`clinical.rs:791-796`), so `hfno`'s rendered docs open with unrelated text — reattach/move. — _added 2026-06-28_

- [ ] **Hand-rolled RFC-4180 vs the `csv` crate** — annotation CSV import/export hand-rolls an RFC-4180 reader (`io/tabular.rs`) for dep-conservatism. The writer is low-risk; the reader is where edge-case bugs live (BOM, lone `\r`, `\r\n` inside quoted fields). Revisit whether the maintenance cost justifies hand-rolling vs BurntSushi's `csv`; at minimum add edge-case tests for the currently-untested cases. — _added 2026-06-28_

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
