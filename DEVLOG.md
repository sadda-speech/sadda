# DEVLOG

A running log of research, decisions, and development for **sadda** — an
open-source toolkit for phonetics and speech-science research
([`README.md`](README.md)). Releases and user-facing changes live in
[`CHANGELOG.md`](CHANGELOG.md); planned and parked work in
[`BACKLOG.md`](BACKLOG.md). This log is the narrative behind those: why
things were built the way they were, what was tried and rejected, and what
is still open.

Newest entries at the top. Each entry is dated `YYYY-MM-DD` and tagged with a
short topic. This file holds the **current month**; earlier months are rotated
into [`devlog/`](devlog/) (index at the bottom).

---

## 2026-07-07 — A5.3: forced alignment in the GUI (completes A5 + the A-series)

The last A-series slice: an **Annotate ▸ Align…** action that force-aligns a
transcript to the selected recording, natively, and writes Words / Syllables /
Phones tiers — no runtime Python.

**Panel.** `AlignPanel` (mirrors the `CriteriaEditor` window idiom): a transcript
box, an espeak-ng `voice`, a min-silence control, and Align. Gated on a loaded
bundle; a spinner + status line while it runs.

**Threading — the `Project` constraint.** `Project` isn't `Clone`/`Sync` and holds
a single-writer lock, so (like the VAD lane) the worker touches **no** `Project`:
it clones the audio envelope + the mpsc `tx` + the egui `Context`, spawns
`compute_alignment` (espeak-ng G2P → `Model::emissions` → `align::align_transcript`
— all off the UI thread, reusing A5.1/A5.2/A5.2b), and sends an
`AnalysisResult::AlignmentDone`. `poll_analysis` (which owns the `Project`) then
calls `write_alignment` and activates the new tiers. So the multi-second run — and
the first-run model download — never freeze the UI, and the tier strip renders the
result on the next frame (it re-reads tiers every frame).

**Model resolution.** `resolve_align_model` fetches `model.onnx` + `vocab.json`
from the HF Hub into the model cache on first use (the app's `download` feature,
A5.2c), then stamps the `input.normalize` + `[alignment]` manifest the bare
`hf://` loader omits — so `emissions` gets the wav2vec2 normalization it needs.

**Verification.** Compile- + clippy-clean; the underlying pipeline is all verified
engine code (align_transcript unit-tested A5.2; emissions numerically matched to
the Python reference A5.2c). The GUI *interaction* + the real end-to-end run
(egui under WSLg + the 635 MB model + ORT) are for hands-on confirmation — they
can't be driven headlessly here.

**A5 done → the A-series (A1–A5) is complete:** neural + MFA alignment, ASR,
syllabification, and now a native GUI, across engine + Python + app.

## 2026-07-07 — A5.2c: verify the ONNX port + wire the model fetch

De-risks A5.2b's ONNX-gated `Model::emissions` — the one part that couldn't run
in CI — by cross-checking it against the Python reference, and wires the app to
fetch acoustic models.

**Verification (the point of this slice).** A synthetic tiny CTC ONNX model
(`onnx`-authored, `[1,N] → [1,N,C]`) is run through **both** the Python
`Wav2Vec2EspeakModel.emissions` reference and the Rust `Model::emissions`, on the
same audio. They agree to **max |Δ| = 5.6e-5** over 2000 values — exactly the
expected f32-vs-f64 rounding (Python log-softmaxes in f32, Rust in f64). So the
port (zero-mean/unit-var normalization + ONNX forward pass + per-frame log-softmax)
is numerically correct against the reference; since Python works with the real
635 MB wav2vec2-espeak model, Rust produces the same emissions with it too (same
code path). The check lives as a `#[ignore]` + env-gated engine test
(`emissions_match_python_reference`), run locally with `ORT_DYLIB_PATH` +
`SADDA_XCHECK_DIR`; the harness that builds the synthetic model is kept out of the
repo (throwaway).

**Fetch wiring.** The app's engine dep gains the `download` feature (implies `ml`,
pulls `ureq`) so the GUI can fetch an acoustic model via the engine's `hf://`
resolver — dormant until network is explicitly allowed. A5.3 uses it to load the
model behind the "Align…" action.

## 2026-07-07 — A5.2b: native model emissions + `align_bundle`

The ONNX half of native alignment: run the acoustic model in Rust and write
aligned tiers onto a bundle with provenance. Completes the engine's ability to
align natively (the GUI in A5.3 just drives it on a worker thread).

- **`Model::emissions(audio) -> [frames × classes]`** — the existing `models.rs`
  ONNX harness (`embeddings`) + a per-frame **log-softmax** (a CTC net emits
  logits). Plus `alignment_vocab()` (loads the manifest's `[alignment].vocab_file`),
  `alignment_frame_rate()`, and the blank token.
- **Manifest support** — new `input.normalize = "zero_mean_unit_var"` (the
  wav2vec2 feature-extractor contract — the one real correctness subtlety: the
  generic embedding harness fed raw samples, but wav2vec2 needs zero-mean/unit-var
  input, so `emissions` would silently degrade without it) applied in `embeddings`;
  new `[alignment]` block (`vocab_file`/`blank_token`/`frame_rate_hz`, defaults =
  the wav2vec2-espeak values).
- **`Project::write_alignment(bundle, &Alignment, …)`** — writes **Words /
  Syllables / Phones** interval tiers, records an `ml_model` `processing_run`
  (processor_id + params + `weights_checksum`), and stamps every interval with the
  run id. The engine counterpart of the Python `import_alignment`.
- **`Project::align_bundle(bundle, transcript, voice, model, min_silence_frames)`**
  (behind the `ml` feature) — the full glue: `load_audio` → `g2p::phonemize` →
  `model.emissions` → `align::align_transcript` → `write_alignment`.

**Tested (no ONNX, runs in CI):** the normalization math, per-frame log-softmax
(rows sum to 1), the `[alignment]` manifest parse + defaults, and a
`write_alignment` round-trip (synthetic `Alignment` → bundle → read back tiers +
the stamped `ml_model` provenance run). **ONNX-gated (not run in CI):** `emissions`
and `align_bundle` — they need ONNX Runtime + the ~635 MB model. Faithful ports of
`sadda.align.acoustic` (same normalization + log-softmax), but **not yet verified
end-to-end against the real model** (couldn't fetch it here); A5.3 wires the fetch
and I'll confirm Rust emissions match the Python reference then.

## 2026-07-07 — A5.2: native alignment orchestration (`align_transcript`)

The "brain" of native forced alignment, in the engine: turn per-frame emissions +
a phonemized transcript into a full `Alignment` (Word + Phone + Syllable tiers),
reusing the pieces that already exist — `g2p::tokenize` (A5.1), `forced_align`, and
`syllable::syllabify` (A3).

`align::align_transcript(emissions, words, vocab, blank, frame_rate,
min_silence_frames, silence_mask) -> Alignment` mirrors `sadda.align.align`'s
orchestration exactly, so the GUI and Python paths produce the same tiers:
tokenize each word's IPA → run the DP → contiguous **Phone** tier (silence as
empty-label intervals) + **Word** tier (inter-word pauses as empty-`text` words)
+ per-word **Syllable** tier. New engine types `Alignment` / `AlignedWord` /
`AlignedPhone` / `AlignedSyllable` (the native analog of the Python dataclasses).

**Fully unit-tested with synthetic emissions** — the Rust analog of the Python
mock-model tests (a block-favoured emission matrix): a two-syllable word →
hə·loʊ, edge-silence carving → empty phone/word intervals, unknown-phone
rejection. No ONNX needed, so it runs in CI.

**Deferred to A5.2b** (needs the real ONNX model, so it's gated like the Python
tests): `Model::emissions` (log-softmax over the existing `models.rs` ONNX
harness — with one real subtlety, the wav2vec2 input needs zero-mean/unit-var
normalization the generic harness doesn't do yet), `Project::align_bundle` (audio
→ emissions → `align_transcript` → write tiers + provenance), and acoustic-model
fetch (the `download` feature in the app). Then **A5.3** the GUI menu/panel.

## 2026-07-07 — A5 design + slice 1: native G2P in the engine

Bringing forced alignment into the desktop app (egui). Design decided with the
user, then the first (foundational) slice.

**Architecture fork — how the GUI runs the aligner.** A recon of `crates/app` +
`crates/engine` found the app already holds the load-bearing pieces natively:
a generic Rust ONNX harness (`models.rs`, the `ort` path VAD uses) that can emit
CTC logits, the `forced_align` DP (`align.rs`, pure Rust), the GUI "run → write
tiers → they render" pattern (`run_criterion`), an off-UI-thread worker pattern
(`analysis_tx`/`poll_analysis`), and provenance recording. Options:
- **(A) Finish orchestration in Rust** — port espeak-ng G2P + the IPA↔vocab
  tokenizer (the only genuinely missing native piece), add `log_softmax` to the
  ONNX output, wrap as `Project::align_bundle`, drive from a menu on a worker
  thread. Self-contained native binary, no runtime Python dependency.
- **(B) Embed Python** — the app already links CPython, so call the tested
  `sadda.align`/`mfa`/`asr`/`syllabify` in-process. All backends at once, no
  reimplementation — but the shipped desktop binary would then need a configured
  Python env with `sadda[align,asr]` importable at runtime.

**Decision: (A), neural-first** (user-confirmed). sadda ships as a native app run
on varied hardware; a runtime Python-env dependency (B) is a real
distribution/reliability tax, and the A-series design already justified the third
surface on the DP living in the engine. MFA (an external subprocess anyway) and
ASR get GUI surfaces in later slices; syllables come nearly free (the syllabifier
is already pure-engine).

**Slice plan.** **A5.1** (this) engine G2P; **A5.2** `Model::emissions`
(log-softmax over the ONNX harness) + `Project::align_bundle` (G2P → emissions →
`forced_align` → Word/Phone/Syllable tiers + provenance) + acoustic-model fetch
(add the `download` feature to the app); **A5.3** the GUI "Align…" menu + panel +
worker-thread wiring.

**A5.1 — engine `g2p` module.** Ports `python/sadda/align/g2p.py` +
`sadda.align.tokenize` to Rust so the native path produces the *same* alignment
target: `phonemize(text, voice)` shells `espeak-ng -q --ipa` per word (edge
punctuation stripped, stress marks removed), and `tokenize(ipa, vocab)` does the
greedy longest-match against a model vocab (multi-char tokens like `dʒ` win).
Key simplification found while porting: the aligner tokenizes the IPA **string**,
so `split_phones` (and its Unicode-category tables) is *not* needed on the target
path — it's pure string work, no new crate deps. New `EngineError::Align`
variant. Tests: `strip_stress`/`tokenize`/edge-punct pure; `phonemize` gated on
espeak-ng presence (early-returns when absent, mirroring the Python skipif).

## 2026-07-07 — A3: syllabification (Phones → Syllables by rule)

Derives a **Syllable** tier from the Phone tier — no model, per the design. Pure
engine module (`crates/engine/src/syllable.rs`) + a thin Python surface.

**Standard problem / approach.** Automatic syllabification. Textbook rule:
**Sonority Sequencing Principle** (nuclei = sonority peaks) + **Maximal Onset
Principle** (intervocalic consonants attach to the following onset as far as a
rising-sonority onset allows) — Clements (1990, doi:10.1017/CBO9780511627736.017),
Selkirk (1982). Alternatives weighed: SSP+MOP + a per-language **onset-legality
table** (more accurate, needs phonotactic data per language), and **data-driven**
syllabifiers (Bartlett et al. 2009 — best per-language, needs training corpora).

**Decision: universal SSP+MOP for v1** — language-agnostic, deterministic, no
data, fits the IPA-multilingual stance; a language-tunable sonority scale +
legality table is the accuracy refinement (backlogged). **Honest limitations,
documented:** pure sonority mis-splits `sC` onsets (`extra` → `ɛks.trə`, not
English's `ɛk.strə`), and adjacent vowels merge to one nucleus (diphthong-
friendly — necessary since A1's `split_phones` splits untied diphthongs like
`oʊ` — but it under-splits true hiatus `ˈke.ɒs`).

**Implementation.** `syllable::syllabify(phones) -> Vec<(start,end)>`: sonority
by IPA base symbol (vowel > glide > rhotic > lateral > nasal > voiced fric >
voiceless fric > voiced stop > voiceless stop); nuclei = maximal vowel runs (or
a syllabicity-diacritic consonant); the boundary between two nuclei is the
longest rising-sonority suffix of the intervocalic cluster (maximal onset).
Python `sadda.align.syllabify(alignment) -> tuple[TimedSyllable, ...]` runs it
word-internally, skipping empty/pause words — so it works on both neural and MFA
alignments and never turns a modeled-silence phone into a syllable.

**Surfaces.** Engine + Python (the align GUI is still A5, as with A1/A2). Cited
in the engine registry (`sadda.align.syllabify` → Clements 1990).

**Tests:** engine (sonority ordering, diphthong = one nucleus, maximal-onset vs
coda split, syllabic consonant, monosyllable, no-nucleus); python (native index
ranges + `syllabify` over hand-built alignments: two-syllable word, monosyllable,
pause words skipped, no cross-word syllables). All ungated (pure rule).

## 2026-07-07 — A4: ASR (the no-transcript path) via faster-whisper

Recognize a transcript from audio, then feed the forced aligner — for speech that
arrives with no transcript.

**Reframing (user correction, 2026-07-06):** the 2026-07-05 design called ASR "a
convenience layer" on "most phoneticians have transcripts." Too strong — unprompted
**conversational / naturalistic** recordings have no prompt, a common research mode,
so ASR is a *primary* workflow. A4 is built accordingly: `sadda.asr` is its own
first-class module (sibling of `sadda.tts`), not a corner of `sadda.align`.

**Runtime decision.** Standard problem: efficient Whisper inference for embedding.
Options: **faster-whisper** (CTranslate2 — MIT, ~4× faster than reference, int8,
**torch-free at inference**); **ONNX-over-ORT** (would reuse sadda.ml's runtime, but
autoregressive Whisper over raw ORT is a lot of fragile generation code *and* ONNX
is weaker for transformers — worse perf for more work); **openai-whisper / transformers**
(pull torch). **Chose faster-whisper** behind an opt-in `sadda[asr]` extra. Tradeoff
accepted and flagged: a second inference engine (CTranslate2) distinct from ORT — but
it's opt-in, torch-free, and MIT, and the ORT alternative isn't worth its cost.

**Shape (mirrors the TTS backend pattern).** `sadda.asr`: an `ASRBackend` structural
protocol (`transcribe(audio, sr) -> Transcription`) + a name→factory registry
(`get_backend`/`register_backend`/`list_backends`, `$SADDA_ASR_BACKEND`), default
`FasterWhisperBackend`. `Transcription` = text + coarse segments + language.
`transcribe(audio, sr)` resamples to Whisper's 16 kHz via `Audio.resample` (reusing
the resampler that landed with the align input-resampling work). Bridge:
`sadda.align.align_auto(audio, sr, model=...)` = transcribe → align in one call (the
recognize→align pipeline); for a review-before-align flow, call `transcribe` + `align`
separately.

**No new Rust (flagged three-surface departure, like TTS/MFA).** faster-whisper is an
external engine; there's no engine algorithm to write. Python-only; the align GUI is
still A5. **Deferred:** auto-deriving the alignment G2P `voice` from the recognized
language (needs a language→espeak-voice map) — for now pass `voice=` explicitly.

**Tests:** all seams ungated via a mock backend — protocol/registry, resample helper,
not-installed error (faster-whisper absent → actionable `sadda[asr]` message), and the
`align_auto` ASR→transcript→align orchestration (mock recognizer + mock acoustic model).
Real faster-whisper is gated (heavy extra + model download).

## 2026-07-06 — A2: MFA gold-standard alignment passthrough

The second aligner backend, per the "both engines" fork (neural default + MFA
gold standard). MFA 3.x (HMM-GMM + speaker adaptation, ~15 ms boundary error) is
a heavy conda/Kaldi tool, so this is a **passthrough**: shell out to the `mfa`
CLI, import its TextGrid into the **same** `Alignment` the neural aligner returns,
so the backends are interchangeable. Strictly opt-in (detect `mfa` on PATH →
actionable error otherwise, the espeak/model-fetch pattern).

**User-decided forks (both "both"):**
- **Granularity → both.** `mfa_align(audio, transcript)` via MFA 3.0's
  `mfa align_one` (fast single file, no corpus/db overhead) **and**
  `mfa_align_corpus(dir)` via `mfa align` (batch + speaker adaptation) →
  `{stem: Alignment}`.
- **Return shape → both.** Functions return an `Alignment`; `import_alignment`
  (new, **backend-agnostic** — works for neural A1 too) writes any `Alignment`
  onto a bundle as Word + Phone interval tiers (Phone child-of-Word by time
  containment).

**Modeled vs imputed silence (user correction).** I'd first planned to map MFA
silence to empty intervals like A1's detectors. Wrong: A1's blank/VAD *infer*
silence (an absence → empty), but MFA *models* it in the HMM — a positive
assertion — so its `sil`/`sp`/`spn` intervals keep their **labels** verbatim.
Empty = "we didn't model it"; labelled = "the model says silence here". The two
silence kinds now read differently on the tier (and survive `import_alignment`).

**No new Rust algorithm (flagged three-surface departure, like TTS).** MFA is a
subprocess; the only engine work is exposing the existing TextGrid reader to
Python (`parse_textgrid_intervals`) to turn MFA output into an `Alignment`. So
this is a Python slice + that one binding; the alignment GUI stays A5.

**Bug found + fixed while wiring the parser:** the engine TextGrid tokenizer did
`bytes[i] as char` when accumulating quoted strings — Latin-1-decoding multi-byte
UTF-8, so **any IPA label was mangled** (`ɪ`'s two UTF-8 bytes `0xC9 0xAA` read as
Latin-1 → `Éª`, so `aɪ` became `aÉª`). Latent across
`import_textgrid` too (importing a Praat TextGrid with IPA would corrupt it),
just never exercised since existing tiers were ASCII. Fixed to accumulate bytes +
`String::from_utf8`; guarded by `ipa_labels_survive_parse_round_trip` (engine).

**ASR is first-class (user correction, recorded).** The 2026-07-05 design called
ASR "a convenience layer" on "most phoneticians have transcripts." Too strong —
unprompted conversational/naturalistic recordings have no prompt, a legitimate and
common research mode, so A4 (recognize → transcript → align) is a primary
workflow. The alignment surfaces already accept a plain-text transcript an ASR
front-end can feed.

**Tests:** engine — IPA round-trip. python (all ungated via a canned MFA-style
long_textgrid) — TextGrid→Alignment mapping, modeled silence stays labelled,
missing-tier error, `import_alignment` round-trip into a bundle (hierarchical +
flat), not-installed error. Real `mfa` integration is gated (skips without the
binary + `SADDA_TEST_MFA_*`). `TimedPhone.score` is now `Optional` (`None` for MFA).

## 2026-07-06 — Forced-alignment input resampling (any-rate audio "just works")

A1 shipped with a hard 16 kHz requirement: `Wav2Vec2EspeakModel.emissions`
raised `ValueError("…expects 16 kHz…— resample first")` on anything else. Since
most recordings are 44.1/48 kHz, alignment failed on ordinary audio unless the
caller resampled by hand — against the "defaults that just work" goal.

**Standard problem:** fixed-rate model input. The model was trained at 16 kHz
mono; the fix is to resample the caller's audio to that rate transparently.

**Decided forks:**
- **Where → in the model, not the orchestration.** The model is the authority
  on its own input rate, so `emissions()` resamples. This makes *any* caller
  robust (not just `align()`), needs no change to the minimal `AcousticModel`
  protocol (mocks stay trivial: `emissions` only), and can't desync the VAD
  path — `_vad_silence_mask` keys silence to the emission `frame_rate`, so it's
  rate-agnostic and unaffected.
- **Which resampler → the engine's existing one.** Reuse `dsp::resample_to_hz`
  (FFT-domain, scipy-`resample`-style) — the same resampler the VAD/GNE paths
  use — rather than pulling scipy into the Python layer. One documented method,
  consistent with the cite-one-method convention.
- **Surface → `Audio.resample`.** Exposed as a general `Audio::resample_to` in
  the engine + `sadda.Audio.resample(target_hz)` in Python (preserves channel
  count; de-interleave → per-channel resample → re-interleave). The natural
  sibling of the `from_samples`/`mono` methods, independently testable.

**Two surfaces, not three.** Resampling here is align-internal plumbing (its
GUI is the still-deferred A5 slice), so engine + Python only — the same way
`from_samples` landed in the silence work without a GUI control.

**Implementation.** `Audio::resample_to(target_hz)` (engine) + `PyAudio.resample`
(binding, stub regenerated). In `sadda.align.acoustic`, a module-level
`_resample_to_model_rate(audio, sample_rate)` helper (a unit-testable seam that
doesn't need the ONNX-gated model) resamples to `SAMPLE_RATE = 16_000`;
`emissions()` calls it instead of raising. Docs `align.md` caveat updated
(mono in, any rate — resampled for you).

**Tests:** engine — resample up-samples proportionally, matching-rate is a copy,
stereo keeps its channel count. python — `Audio.resample` rate/length + no-op,
`_resample_to_model_rate` at-rate/off-rate; the model-gated test flips from
asserting an 8 kHz *raise* to asserting an 8 kHz input resamples to the same
frame count as 16 kHz.

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
clippy/fmt clean. `min_silence_seconds` default set to **0.20 s** (2026-07-06),
grounded in the pause literature: above stop-closure durations, between Praat's
0.1 s silence default and Goldman-Eisler's (1968) 0.25 s articulatory-vs-pause
boundary. **Pending review:** confirm the blank-default choice; derive a
corpus-tuned threshold later (empirically, via the S5 agreement engine).

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

---

## Archives

Older months are rotated into [`devlog/`](devlog/) to keep this file lean
(one file per month). Newest first:

- **[2026-06](devlog/2026-06.md)** — annotation suite S4c–S7 → perf arc + scan ergonomics → live recording → CSV / JSON annotation I/O → DSP method diversity + preset registries → 0.5.0
- **[2026-05](devlog/2026-05.md)** — project genesis → 0.2.0 / 0.3.0 releases → annotation suite S1–S7 → perf arc + large-file ingest guard
