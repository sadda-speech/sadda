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

- [ ] **Two-pass adaptive pitch floor/ceiling** — implement the now-standard two-pass prosody strategy: run pitch once over a wide range, derive the floor/ceiling from the resulting f0 distribution (e.g. quantile-based — floor ≈ 0.75×q25, ceiling ≈ 1.5×q75, exact formula TBD from the source), then re-run pitch with the speaker-tuned range. Removes the hardcoded-default guesswork (see the static 500-vs-600 Hz ceiling question) and adapts per speaker/recording. Prior art: Hirst, D.J. (2011), "The analysis by synthesis of speech melody" (Journal of Speech Sciences 1(1):55–83) and the De Looze & Hirst two-pass Praat scripts. Should ship as an opt-in pitch-config mode across engine + Python + GUI (three surfaces); cite the source per the DSP-method-diversity principle. — _added 2026-07-09_

- [ ] **Corpus-level Bayesian speaker-adaptive pitch range** — beyond the single-recording two-pass method above, pool f0 information about a *speaker across all their recordings in a corpus* to set that speaker's floor/ceiling (and priors) — a hierarchical/Bayesian estimator where each recording's range is informed by the speaker-level posterior, not just its own distribution. More robust for short or atypical recordings where a single-file two-pass estimate is noisy. Needs the corpus/speaker-metadata layer to identify same-speaker recordings. Design session + a citable method before implementing (survey hierarchical-Bayes f0 / speaker-normalization literature); ships across the three surfaces. — _added 2026-07-09_

- [ ] **Tighten pYIN's alignment with librosa (faithful port)** — sadda's pYIN keeps two *deliberate* deviations from `librosa.pyin`: a finer HMM grid (`pyin_bins_per_semitone = 20` vs librosa's `resolution = 0.1` semitone ≈ 10 bins/semitone) and a Gaussian-σ semitone-distance frequency transition (reusing sadda's Boersma HMM idiom) instead of porting librosa's `max_transition_rate` cap. **User call (2026-07-09):** prefer as faithful a port as our tooling reasonably allows over these divergences — align the grid resolution and port the `max_transition_rate` transition semantics, then validate against a *tighter* librosa golden (closer than the current median-f0-within-tolerance check). Note: `switch_prob` was already aligned to librosa's 0.01 in the 2026-07-09 citation/claims pass (it had been a wrong-value 0.05 mistakenly documented as "librosa's value"). Engine-only change (the pyin params already thread through); keep the three-surface exposure and update the docstring's "deliberate deviation" reasoning once aligned. — _added 2026-07-09_

- [ ] **Yearly citation-link check** — once a year, verify every weblink in a citation (engine `citation_for` registry + `## References` blocks + module/docstring refs + model cards) still resolves *and* points to the intended material. Fix drifted URLs; for a link that's gone dead with no replacement, remove the link and leave a note that no weblink is available (per the weblink convention: a citation has either a working link or an explicit "no weblink available" note). Could be a scripted link-checker over the repo's http(s)/doi.org URLs, run manually or in a scheduled CI job. — _added 2026-07-06_

- [ ] **Systematic pass on annotation keybindings** — selection, label typing, tier selection etc. behave oddly with the new keybindings; work through it together (needs a joint design/debug session, not a solo fix) — _added 2026-07-02_
- [ ] **Glyph-distinguishing font for annotations (maybe global)** — use a font that clearly separates O/0, 1/I/l/|, etc.; leaning toward a teletype monospace for annotation text. Candidates: JetBrains Mono / IBM Plex Mono / DejaVu Sans Mono / Iosevka (all disambiguate). Decide scope (annotation text vs global) + embed the font (licensing/size). — _added 2026-07-02_
- [ ] **Key pattern: chain an interval from the previous endpoint** — after labeling an interval, a fast keypress should start a new interval whose left edge is the just-finished interval's right edge (common annotation flow); design the keybinding (ties into the keybindings pass). — _added 2026-07-02_
- [ ] **Automatic prosodic annotation** — auto prosody-annotation system with defaults that just work — _added 2026-07-02_
- [ ] **ASR + phone-level forced alignment (A1–A5)** — produces Words / Syllables / Phones tiers, "defaults that just work". **Designed 2026-07-05** (see DEVLOG): alignment-first · both engines (neural default + MFA passthrough) · IPA-multilingual via espeak-ng G2P. Model: `facebook/wav2vec2-lv-60-espeak-cv-ft` (Apache-2.0, CTC, espeak-IPA) behind `sadda[align]`. Architecture = ONNX acoustic posteriors (`sadda.ml`) + a constrained-Viterbi DP in the Rust `engine`. Slices: **A1** engine DP + G2P + ONNX phone model → Words+Phones; **A2** MFA 3.0 passthrough; **A3** syllabification (sonority + maximal-onset); **A4** Whisper ASR (no-transcript path); **A5** GUI. QA via the S5 agreement engine (boundary Δ / κ). **Shipped 2026-07-06/07:** **A1** (`sadda.align`, real-audio validated) + **silence detection** (`detector="blank"`/`"vad"`, empty intervals) + **input resampling** (`Audio.resample` → 16 kHz) + **A2 MFA** (`mfa_align`/`mfa_align_corpus` → same `Alignment`; `import_alignment` writes any Alignment to a bundle; modeled `sil`/`sp` kept **labelled**; delivers "truly modeled silence"; fixed a UTF-8/IPA TextGrid-tokenizer bug) + **A4 ASR** (`sadda.asr`: `ASRBackend` + `FasterWhisperBackend`, opt-in `sadda[asr]`, MIT/torch-free; `transcribe` + `align_auto`; ASR first-class for conversational/naturalistic data) + **A3 syllabification** (`syllabify(alignment)` → `TimedSyllable` by universal Sonority-Sequencing + Maximal-Onset, Clements 1990, pure engine `syllable.rs`, cited). **A5 GUI in progress** (Rust-native neural-first, user-confirmed — no runtime Python dep; see DEVLOG): **A5.1 native engine G2P shipped 2026-07-07** (`sadda_engine::g2p` — espeak-ng `phonemize` + `tokenize`, mirrors the Python target). **A5.2 native orchestration shipped 2026-07-07** (`align::align_transcript` — emissions + transcript → `Alignment` {Word/Phone/Syllable}, mirrors `sadda.align.align`; fully unit-tested with synthetic emissions). **A5.2b native model emissions + align_bundle shipped 2026-07-07** (`Model::emissions` = ONNX harness + log-softmax + wav2vec2 `input.normalize`; `[alignment]` manifest block; `Project::write_alignment` writes Words/Syllables/Phones tiers + `ml_model` provenance; `Project::align_bundle` glues it — ONNX-gated, faithful port, not yet run against the real model). **A5.2c shipped 2026-07-07** — Rust `Model::emissions` verified numerically against the Python reference (synthetic ONNX cross-check, max |Δ| 5.6e-5); app gains the `download` feature to fetch acoustic models via `hf://`. **A5.3 shipped 2026-07-07** — Annotate ▸ Align… panel: transcript + voice → Words/Syllables/Phones tiers; model + DP on a worker thread (no `Project` — decompose pattern), `write_alignment` on the UI thread; `resolve_align_model` fetches + stamps the manifest. **A5 complete → the A-series (A1–A5) ships end to end** (neural + MFA alignment, ASR, syllabification, native GUI). GUI interaction + real-model end-to-end need hands-on confirmation (can't be driven headlessly). Remaining silence work: `sil`-aware neural model, VAD∩blank hybrid. Refinement: auto-derive alignment `voice=` from the ASR-detected language (needs a language→espeak-voice map; today pass `voice=` explicitly). — _added 2026-07-02, groomed 2026-07-05_
- [ ] **Syllabification: per-language onset-legality table** — A3 (shipped 2026-07-07) uses a *universal* Sonority-Sequencing + Maximal-Onset rule with no phonotactic legality data, so it mis-splits `sC` onsets (`extra` → `ɛks.trə` not `ɛk.strə`) and merges true vowel hiatus (`ˈke.ɒs`). Add a language-tunable legal-onset inventory (and/or a language-specific sonority scale) so MOP is constrained by attested onsets; consider a data-driven syllabifier (Bartlett et al. 2009) as an alternate backend per the method-diversity principle. — _added 2026-07-07_
- [ ] **Import from image → sound** — synthesize audio from a figure (waveform image, spectrogram, cepstrogram, or MFCC heatmap; consider other representations) — _added 2026-07-02_
- [ ] **Interval-create → Enter opens label edit (cursor in field), Enter accepts, Esc cancels** — currently Enter on a just-created interval throws an overlap warning instead of entering label editing — _added 2026-07-02_
- [ ] **Bug: signal-lane y-axis labels clipped at the top** — rotated labels like "f0" lose the top of the glyphs; add a bit more left/top padding — _added 2026-07-02_
- [ ] **Bug: "Recording too short for spectrogram window" shows mid-recording** — message bar reports it before the recording is finished — _added 2026-07-02_
- [ ] **Bug: View ▸ UI scale slider moves as it scales** — the slider repositions under the cursor while dragging, breaking the selector — _added 2026-07-02_
- [ ] **Rework the annotation project cycle** — the current flow is clunky/confusing; refine it to feel like an extension of the hand, not a mechsuit — _added 2026-07-02_
- [ ] **Accessibility for blind researchers** — explore what it would take to do phonetic analyses in sadda with no visual info (non-visual access to waveform/spectrogram/measures — e.g. sonification, screen-reader-friendly Python surface, audio cues). Design exploration. — _added 2026-07-05_
- [ ] **TTS: build a Guiyang-dialect (Chinese) voice model** — train/adapt a TTS voice for the Guiyang dialect; personal-interest, **low priority**. Would slot in as another `TTSBackend` once a model exists. — _added 2026-07-05_
- [ ] **TTS: wire the Kokoro backend + `sadda[tts]` extra** — the planned high-quality neural default (Kokoro-82M, hexgrad; weights **Apache-2.0**, confirmed). T1 (2026-07-05) registered `"kokoro"` as a pending backend that raises an actionable error. **Prep done (2026-07-05):** two pip packages exist — `kokoro` (pulls **torch**) vs **`kokoro-onnx`** (ONNX-runtime based). Prefer `kokoro-onnx`: sadda's `[ml]` extra already ships `onnxruntime>=1.22`, so the `[tts]` extra can reuse ORT instead of dragging torch into the wheel, and the model (`onnx-community/Kokoro-82M-v1.0-ONNX`) can likely plug into the existing `hf://` model-registry/fetch path (SADDA_ALLOW_NETWORK-gated). Open: `kokoro-onnx`'s own package license + its phonemizer dep (espeak-ng-backed) licensing; confirm before committing the extra. — _added 2026-07-05_
- [ ] **TTS: cloud backends (ElevenLabs / OpenAI)** — opt-in `TTSBackend` implementations for top-quality narration when a network + API key is acceptable (i.e. not reproducible CI doc builds). Plug into the existing registry; key via env var; document the reproducibility trade-off. — _added 2026-07-05_
- [ ] **TTS: localization preprocessing per backend** — voice `voice=` codes are already threaded through the surface, but text needs per-language preprocessing before espeak-ng. **Finding (2026-07-05, espeak-ng 1.50):** Japanese `ja` needs (1) kanji→kana (no kanji dict — it names unknown chars as "Japanese letter"), (2) **whitespace word-segmentation** (no built-in segmenter — an unspaced run orphans the `ー` chōonpu into "Japanese letter"; spacing was the actual fix), and (3) particles written phonetically (は→ワ, へ→エ — no sandhi). A `fugashi`/MeCab + `pykakasi` pass does all three. Verify which other languages need similar prep (e.g. Chinese numerals/segmentation) as localization scope firms up. — _added 2026-07-05_
- [ ] **TTS: richer narration-script format** — T1 ships a minimal `parse_script` (blank-line paragraphs → segments). Design a fuller on-disk format: per-scene stable ids, inline voice/rate directives, and screencast **timing markers** so narration can be aligned to captured video. Do a design session (prior art: subtitle/caption formats, SSML, storyboard/EDL). — _added 2026-07-05_
- [ ] **Doc-automation: screencast/gif capture + A/V mux** — the other half of the "auto-generate all docs" vision that T1 TTS deliberately scoped out. Capture the egui app (GUI-driving under WSLg/xvfb) → gif/video, then mux the `sadda.tts` voiceover track against it with ffmpeg (already on the machine). Entangles with the WSLg GUI-debugging gotchas; its own project. Related: the existing "Auto-generate demo screenshot, run at each release" item. — _added 2026-07-05_
- [ ] **TTS: docs API-reference page + GUI surface (conditional)** — add a `sadda.tts` page to the mkdocs API nav once the surface firms up. A GUI surface (egui "type text → hear voice") stays deferred until a concrete phonetics analysis story appears (analysis-by-synthesis, perception-experiment stimuli) — noted because it's the flagged departure from the three-surface principle. — _added 2026-07-05_
- [ ] **GUI in-line help/information system** — make it easy for users to get info about a button, window, function, operation, etc. (e.g. hover tooltips / `?` affordances / contextual help panel) — _added 2026-06-28_
- [ ] **MFCC `method="kaldi"`: validate against real Kaldi `compute-mfcc-feats`** — currently validated against torchaudio's kaldi-compliance (PyTorch's faithful reproduction), not Kaldi-proper. Regenerate the golden from an actual `compute-mfcc-feats` run when a Kaldi toolchain is available, to drop the one-step-removed caveat. (`crates/engine/tests/dsp/mfcc/`) — _added 2026-06-28_
- [ ] **MFCC method selection in the GUI** — once MFCC has a GUI lane/feature (it has none today), add an `MfccMethod` picker alongside the View ▸ DSP methods f0/formant pickers. — _added 2026-06-28_
- [ ] **MFCC `htk()` preset + power/magnitude knob + f64 pipeline** — HTK-proper MFCC needs (a) a magnitude-vs-power spectrum knob (HTK USEPOWER=F uses magnitude; pipeline is power-only) and an HTK DCT/lifter, and (b) an HTK golden to validate. Also: migrate `mfcc_with_params` to f64 so the Praat *preset* matches the dedicated f64 `mfcc(Praat)` path, then route Praat through the unified pipeline too (completes the dispatch collapse). — _added 2026-06-29_
- [ ] **Explore FFT method/backend options** — the Praat MFCC residual is irreducible FFT-library noise (realfft vs Praat's NUMrealft amplified by the un-normalised dB DCT on ~1e-30 filter powers). Options: vendor/port Praat's `NUMrealft` (would let Praat MFCC go byte-exact), expose a selectable FFT backend, or evaluate alternatives (rustfft variants, pocketfft port) for accuracy/perf. Relevant to any pipeline where FFT-library differences surface. — _added 2026-06-29_
- [ ] **Design session: communicating DSP parameter/method choices** — workshop a more organized way to present DSP algorithm parameter choices and convey the *effects* of each choice (e.g. visualizations, side-by-side previews, inline explanations of trade-offs). Spans pitch/formant/MFCC method pickers + their numeric params across Python + GUI. — _added 2026-06-29_
- [ ] **Live display mode (no recording)** — monitor/demo a running signal live without creating bundles or writing files (transient scope/spectrogram/lane view). — _added 2026-06-29_
- [ ] **UI miscellany (next round)** — aggregator for small UI tweaks to batch together; add sub-bullets as they come up — _added 2026-06-20_
  <!-- pane focus-shifting + annotation navigation promoted to active work (feat/pane-focus-nav); cursor-at-window-center dropped (start time is fine) -->
- [ ] **Docs site: rotating "thank-you" banner for upstream OSS** — on home (or any) page load, spotlight one open-source project sadda draws from, celebrating OSS generosity + noting sadda pays it forward with equally open licensing; careful not to imply those projects endorse/agree with sadda. — _added 2026-06-30_

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
- [ ] **Corpus "View" abstraction** — a first-class subset over corpus samples (by acoustic, annotation, derived-signal, or metadata criteria) with all attendant properties; generalizes aggregate view for nimbler corpus navigation/operation — _added 2026-06-25_
- [ ] **Bundle metadata panel** — display selected bundle's metadata in bottom section of bundle navigation column — _added 2026-06-25_

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

## Export & figures

_Designed 2026-07-01 (see DEVLOG design entry). `FigureSpec` IR in
`crates/engine/src/io/figure.rs` → pluggable serializers; strand 2 ships first._

- [x] **S2 — GUI-region capture → PNG** _(done, feat/figure-export)_ — rubber-band
  region select over the app, crop the framebuffer `ColorImage` to the rect, save
  PNG via a file dialog (reuses/un-gates the F12 screenshot path). The manual/
  hand-draw tier; named capture + headless automation build on top (S3–S8). —
  _added 2026-07-01_
- [ ] **G0 — figure-export groundwork** — move the colormap/spectrogram bake into
  the engine + expose the spectrogram raster/matrix; add a `visible_lanes()`
  accessor to the app. Bake-parity test; no user surface. — _added 2026-07-01_
- [ ] **G1 — first shippable figure** — `FigureSpec` IR + **SVG** serializer for
  waveform + spectrogram + tiers (specTeX-parity core) + PDF via SVG→PDF; Python
  `export_figure(...)`; GUI "Export figure…" dialog with per-element include
  checkboxes (default from `visible_lanes()`) + format choice. — _added 2026-07-01_
- [ ] **G2 — TikZ backend** — TikZ serializer off the same IR + standalone `.tex`
  preview wrapper (specTeX integration model). — _added 2026-07-01_
- [ ] **G3 — measure lanes in figures** — f0 / formants / intensity / VAD as
  stacked rows, both backends. — _added 2026-07-01_
- [ ] **G4 — heatmap lanes + style knobs** — MFCC + embedding rasters; expose
  colormap/palette/font/dimension overrides across Python + GUI (completes the
  "whole signal column" default). — _added 2026-07-01_
## Documentation-image pathway

_Designed 2026-07-02 (see DEVLOG design entry). North star: an **automatable,
headless, drift-tested** pipeline to regenerate documentation images from a
scripted recipe. Anti-drift = same `SaddaApp::ui` + same egui/wgpu renderer in
both the live app and the headless path, enforced by `egui_kittest` snapshot
goldens. Shared primitives (S3–S5) feed both a live driver and the headless
spine (S6). Absorbs the former "structural-lane toggles" item into S3._

- [~] **S3 — Visibility & selection model** _(mostly done, feat/figure-export)_ —
  ✅ show/hide for every structural subpane (waveform/spectrogram/tier strip) via
  View ▸ Signal panes; ✅ per-tier in/out (strip context "Hide tier" + View ▸ Tiers
  checkboxes, `hidden_tier_ids`); ✅ Python control — `sadda.app.set_pane_visible`
  / `set_tier_visible` drained + applied after a run. ⬜ Remaining: a
  `visible_lanes()` accessor (lands with the S4 named-rect registry; also unblocks
  figure-export G0). — _added 2026-07-02_
- [~] **S4 — Named-rect registry + interactive named capture** _(mostly done,
  feat/figure-export)_ — ✅ per-frame double-buffered registry (`capture_rects`) of
  named rects: composites `whole-window`/`signal-column` + waveform/spectrogram/
  each measure lane/tier strip; ✅ **View ▸ Capture image → PNG** submenu (hand-drawn
  *or* named region, greyed when not visible); ✅ `PendingNamed` deferred one frame
  so the menu leaves the shot; ✅ pixel-rect echo on save (`reproduce with
  capture(rect=(x,y,w,h))`). ⬜ Remaining: record sidebar/console/side-panels too
  (only signal-column components + whole-window today); a config-derived
  `visible_lanes()` for figure-export G0 (the render-time registry covers capture).
  — _added 2026-07-02_
- [x] **S6.2 — Programmatic column widths** _(done, feat/figure-export)_ — scriptable
  `sadda.app.set_column_width(name, width)` for the GUI columns (`bundles` sidebar,
  `annotation`, `reference`), via the same `PanelState` mechanism as heights. Signal
  column = the flex/remainder column (set the sides + window size; rejected with a
  clear error). Also added the three columns to the named-capture registry (closes
  the S4 gap). Verified headless by difference: widening the request 100pt widens the
  rendered panel 100pt. — _added 2026-07-02_
- [x] **S6.1 — Programmatic pane heights** _(done, feat/figure-export)_ — scriptable
  `sadda.app.set_pane_height(name, height)` for the individual signal-column panes
  (waveform / tier strip / f0 / formants / intensity / vad / mfcc), by writing
  egui's `PanelState` — keeps drag intact, persists via eframe, reproducible in the
  headless harness. Spectrogram = the flex/remainder pane, so it's sized indirectly
  (rejected with a clear error). Verified headless: a scripted 120px waveform renders
  at ~120px. ⬜ Optional later: a GUI numeric height control + embedding-lane name. —
  _added 2026-07-02_
- [x] **S5 — Standard doc-size presets** _(done, feat/figure-export)_ — ✅ `View ▸
  Doc size ▸` {1280×800, 1600×1000, 1024×768} via `ViewportCommand::InnerSize`,
  UI zoom pinned to 100%; ✅ scriptable `sadda.app.set_window_size(w, h)` (queued
  during a run, applied next frame where the `Context` is in hand). Pixel density
  on the live window still tracks monitor DPI — the S6 headless path fixes it for
  byte-reproducibility. — _added 2026-07-02_
- [~] **S6 — Headless doc-render harness (spine)** _(working, feat/figure-export)_ —
  ✅ `egui_kittest` 0.34 (eframe+wgpu+snapshot, dev-dep) drives the real `SaddaApp`
  offscreen in `doc_render.rs`; ✅ compose view via app state → settle async DSP →
  resolve a named `capture_rect` → wgpu render → crop → PNG (verified: faithful
  waveform+spectrogram figure of a fixture bundle); ✅ headless via **lavapipe**
  (software Vulkan) — `configure_headless_gpu` auto-points wgpu at the ICD;
  render tests `#[ignore]` so default `cargo test` never hits the crashy WSL GPU.
  ⬜ Remaining: theme/light-mode knob, sequence rendering (for the screencast north
  star), and folding into the recipe runner (S7). — _added 2026-07-02_
- [~] **S7 — Recipe API + in-repo recipes** _(core done, feat/figure-export)_ —
  ✅ `sadda.doc.shot(to, capture, project, bundle, size, theme, show, heights,
  widths)` declarative Python recipe API; ✅ headless executor (open → select →
  compose → settle → render → crop → write); ✅ `capture` = named target **or**
  `rect:x,y,w,h`; ✅ **light/dark theme** (`sadda.app.set_theme` + shot `theme=`);
  ✅ `just docs-images` (self-configures lavapipe, runs serially). ✅ external
  recipe **files** (`run_recipe_file` + `docs/recipes/*.py`); ✅ `audio=` builds a
  throwaway project from a WAV so recipes are self-contained (no committed DB); ✅
  committed example `docs/recipes/overview.py` rendered by a test. Verified: the
  file recipe renders end-to-end; parallel scratch-dir race fixed. ⬜ Remaining:
  a clean-licensed demo speech clip (fixture is synthetic), committing real doc
  images, and wiring them into the mkdocs site. — _added 2026-07-02_
- [x] **S8 — CI snapshot-diff gate** _(done, feat/figure-export)_ — ✅ `egui_kittest`
  `try_image_snapshot` diffs the cropped figure vs a committed golden
  (`crates/app/tests/snapshots/doc-signal-column.png`) with egui's cross-platform
  tolerance; ✅ `.github/workflows/docs-images.yml` runs lavapipe headless — structural
  checks blocking, pixel snapshot advisory (cross-machine determinism unproven until
  goldens are regenerated from CI); ✅ `just docs-images` (check) + `docs-images-update`
  (refresh goldens). ⬜ Promote the pixel gate to blocking once CI-native goldens land.
  — _added 2026-07-02_
- [~] **Doc-image catalog — Phase 1 + B1** _(rendered, feat/figure-export)_ — Group A
  (overview/hero, signal-view, spectrogram, pitch, formants, intensity, mfcc,
  measure-stack; light+dark hero) + B1 annotated tiers via `shot(textgrid=…)`,
  from the CC0 demo clip → `docs/assets/generated/`. ✅ Real Utterance+Words
  annotation wired in; ✅ Home hero on `index.md` (light/dark via Material
  `#only-light/#only-dark`). ⬜ Remaining: a "tour" section + annotation-cycle
  images; README still uses the old hand-taken screenshot. — _added 2026-07-02_
- [ ] **Doc-image recipe primitives (Group B)** — small recipe additions for the
  remaining figures: `selection`/`cursor` (measurement + annotation-editing
  shots), reference-panel open + distribution install/select (vowel-space
  figure), DSP-method choice (f0/formant method comparison), multiple bundles
  (corpus/sidebar navigation). — _added 2026-07-02_
- [ ] **(future north star) Scripted screencast + TTS narration** — the fuller
  vision from `devlog/2026-05.md`: script an in-app workflow (create/record → measure
  → annotate → …) and emit a **screencast video with narration**, doubling as
  end-to-end UI testing. Rides on the S6 headless driver rendering a **timed frame
  sequence** (not single frames) + `ffmpeg` muxing (frames + audio → mp4/gif) + an
  audio track. Design S6/S7 so they don't foreclose it; not on the doc-image
  critical path. — _added 2026-07-02_
- [ ] **TTS / speech synthesis in the app (roadmap dependency)** — prerequisite for
  the screencast narration above (and generally useful: synthesized stimuli,
  narrated tutorials). Not yet on any roadmap; only appeared in the 2026-05
  walkthrough idea. Needs its own design pass (engine choice + voice licensing —
  e.g. Piper/Coqui/espeak-ng — offline vs cloud, quality vs footprint). —
  _added 2026-07-02_
  - Build a **default TTS pipeline in sadda that just works** out of the box —
    good-enough (not perfect) quality, iterated on later. — _added 2026-07-02_

## Annotation suite tweaks

- [ ] **Make the notebook `kind` set editable / reconsider the trio.** S7 fixed
  it to `observation | measurement | decision` via a CHECK constraint (not
  user-editable). If the trio feels wrong after real use, revisit names/extent;
  cheap change. (surfaced 2026-06-01 while testing the lab-notebook)

## Docs

_(empty)_
