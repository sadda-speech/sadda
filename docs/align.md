# Forced alignment

`sadda.align` time-aligns a **transcript** to audio, producing **Word** and
**Phone** results with IPA phone labels — the phonetician's classic tool. Give it
audio plus the words that were said, and it tells you *when* each word and phone
occurs.

PROVISIONAL tier. See the API reference at [`sadda.align`](api/align.md), and the
2026-07-05 design entry in the DEVLOG for the architecture.

## What you need

- The **`espeak-ng`** system binary (the grapheme→phoneme engine, also used by
  TTS): `apt install espeak-ng` / `brew install espeak-ng`.
- **`pip install "sadda[align]"`** — pulls ONNX Runtime + `huggingface_hub` for
  the neural acoustic model. The model itself (~635 MB, Apache-2.0) is fetched
  from the Hub on first use and cached.

## Quickstart

```python
import sadda
import numpy as np

# audio: a mono float32 waveform at any sample rate (resampled to 16 kHz for you)
audio, sample_rate = load_your_audio_as_mono()

model = sadda.align.Wav2Vec2EspeakModel.from_pretrained("sadda-speech/wav2vec2-espeak-ctc")
alignment = sadda.align.align(audio, sample_rate, "the words that were said", model=model)

for w in alignment.words:
    print(f"{w.start_seconds:.2f}-{w.end_seconds:.2f}  {w.text}  [{' '.join(p.label for p in w.phones)}]")
```

`align` returns an [`Alignment`](api/align.md) with `words` ([`TimedWord`](api/align.md))
and a flat `phones` list ([`TimedPhone`](api/align.md)) — each carrying
`start_seconds`, `end_seconds`, and (for phones) an alignment-confidence `score`
(`None` for backends that don't emit one, e.g. MFA).

## In the desktop app

The GUI exposes alignment natively (no Python): select a recording, then
**Annotate ▸ Align…**, type the transcript and a language voice, and run. It
phonemizes, runs the neural CTC aligner on a background thread (the first run
downloads the model), and writes **Words / Syllables / Phones** tiers onto the
bundle — the same result as `sadda.align.align`.

## How it works

Three stages, cleanly separated (which is why each is swappable and testable):

1. **G2P** — `espeak-ng` turns the transcript into per-word IPA phones (the
   *target*). Multilingual: pick the language with `voice=` (e.g. `"en-us"`,
   `"de"`, `"cmn"`).
2. **Acoustic model** — an ONNX wav2vec 2.0 CTC network emits per-frame phone
   probabilities. The default, `Wav2Vec2EspeakModel`, outputs espeak IPA phonemes
   that line up exactly with the G2P.
3. **Forced-align DP** — a constrained-Viterbi over the CTC posteriors (in the
   Rust engine) places each target phone in time; words are grouped from their
   phones.

## With or without a transcript

Alignment is **transcript-first**: you supply the words, and it finds where they
fall. The transcript needn't be perfectly tokenized — punctuation is stripped,
and each whitespace-separated word becomes one `TimedWord`.

But plenty of speech has no transcript — unprompted conversational and
naturalistic recordings especially. For those, recognize the words first with
**ASR**, then align. `sadda.asr` (opt-in `pip install "sadda[asr]"`, default
[faster-whisper](https://github.com/SYSTRAN/faster-whisper)) does the recognizing:

```python
tr = sadda.asr.transcribe(audio, sample_rate)   # -> Transcription(text=..., language=...)
alignment = sadda.align.align(audio, sample_rate, tr.text, model=model)
```

`align_auto` does both in one call (recognize → align) when you don't need to
review the transcript first:

```python
alignment = sadda.align.align_auto(audio, sample_rate, model=model)
```

Set `voice=` to the spoken language for the alignment G2P (deriving it from the
recognized language automatically is a planned refinement). For a human-in-the-loop
workflow — recognize, correct the transcript, then align — call `sadda.asr.transcribe`
and `align` separately.

## Languages

`espeak-ng` covers [100+ languages](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md),
and the acoustic model is multilingual (Xu et al. 2022, cited in the References
below), so
alignment is: translated/target-language transcript + the matching `voice=`:

```python
sadda.align.align(audio, 16000, transcript_de, model=model, voice="de")
```

You can also run just the G2P:

```python
utt = sadda.align.phonemize("hello world", voice="en-us")
print(utt.words[0].phones)   # ('h', 'ə', 'l', 'o', 'ʊ')
```

## Custom acoustic models

The aligner is backend-agnostic: any object satisfying the
[`AcousticModel`](api/align.md) protocol (an `emissions(audio, sample_rate)`
returning [`Emissions`](api/align.md) = log-probs + vocab + frame-rate + blank
id) can drive it — your own model, or a test mock.

## Gold-standard alignment with MFA

The neural aligner needs no setup, but the field's gold standard is the
[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) (HMM-GMM
+ speaker adaptation, ~15 ms mean boundary error). `sadda.align.mfa` is a
passthrough: it shells out to an external `mfa` install and returns the **same**
`Alignment`, so the two backends are interchangeable.

MFA is a heavy conda/Kaldi tool, so it's strictly opt-in — install it separately
and download models, or you get an actionable error:

```bash
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
```

```python
from sadda.align import mfa_align, mfa_align_corpus

# one recording (mfa align_one — fast, no corpus setup)
al = mfa_align("utt.wav", "the words that were said")

# a whole corpus directory of paired audio + .lab/.txt files
alignments = mfa_align_corpus("corpus/")   # -> {file_stem: Alignment}
```

Defaults are the IPA-flavoured `english_mfa` dictionary + acoustic model (pass
`dictionary=` / `acoustic_model=` names or paths for other languages). One honest
caveat: MFA's phone inventory isn't identical to espeak-ng's, so labels won't
match the neural aligner's byte-for-byte — and `align_one` skips speaker
adaptation, so it's marginally less precise than the full-corpus path.

## Onto a bundle

Either backend's `Alignment` can be written straight into a project bundle as
Word + Phone interval tiers (Phone as a child of Word) with
[`import_alignment`](api/align.md) — the "align, then annotate/edit" step:

```python
word_tier, phone_tier = sadda.align.import_alignment(project, bundle_id, al)
```

## Syllables

`syllabify` derives a **Syllable** tier from an `Alignment` by rule — no model:

```python
al = sadda.align.align(audio, 16000, "hello", model=model)
for syl in sadda.align.syllabify(al):
    print(f"{syl.start_seconds:.2f}-{syl.end_seconds:.2f}  {syl.label}")
# 0.00-0.20  hə
# 0.20-0.60  loʊ
```

It applies the **Sonority Sequencing Principle** (syllable nuclei are sonority
peaks) and the **Maximal Onset Principle** (intervocalic consonants attach to the
following onset as far as a rising-sonority onset allows) — Clements
([1990](https://doi.org/10.1017/CBO9780511627736.017)), Selkirk (1982).
Syllabification is word-internal: each word's phones are syllabified on their
own, and pauses/silence between words belong to no syllable.

!!! warning "Universal v1 — known limitations"
    This is a *language-agnostic* pass: one universal sonority scale and **no
    per-language onset-legality table**. So it can't know a language's specific
    legal clusters — pure sonority splits `extra` as `ɛks.trə` (because `s` is
    more sonorous than the following `t`), where English, allowing `/str/` as an
    onset, prefers `ɛk.strə` (the `sC`-cluster exception). It also merges adjacent
    vowels into one nucleus (diphthong-friendly, but it under-splits true
    **hiatus** like `ˈke.ɒs`). A language-tunable legality table is the planned
    refinement.

## Silence and pauses

Leading/trailing silence and inter-word pauses are detected and left as
**empty-labeled intervals** (the tiers stay a contiguous partition of the
recording — on TextGrid export these are Praat's empty intervals):

```python
sadda.align.align(audio, 16000, transcript, model=model,
                  detector="blank", min_silence_seconds=0.20)
```

The `min_silence_seconds` default (0.20 s) follows the pause literature — above
typical stop-closure durations, between Praat's 0.1 s silence-detector default
([`To TextGrid (silences)`](https://www.fon.hum.uva.nl/praat/manual/Sound__To_TextGrid__silences____.html))
and Goldman-Eisler's (1968) 0.25 s articulatory-vs-pause boundary
(*Psycholinguistics: Experiments in Spontaneous Speech*, Academic Press — no
stable weblink available). Derive a value tuned to your own corpus later (the S5
agreement engine scores an aligner against a hand-corrected reference).

- `detector="blank"` (default) marks long runs of the CTC **blank** as silence —
  it reuses the acoustic model's own posteriors, so it's consistent with the
  alignment and needs no second model.
- `detector="vad"` uses Silero VAD (`sadda.ml`). More independent, but coarser —
  its speech/non-speech boundaries can disagree with the alignment and swallow
  very short edge phones. Prefer `blank` unless VAD robustness matters.
- `detector=None` disables silence handling.

!!! note "Inferred vs modeled silence"
    The neural detectors *infer* silence — `blank` from "no phoneme emitted",
    `vad` from a separate detector — so it's an **empty** interval (an absence we
    didn't model). The **MFA** backend *models* optional silence in its HMM, so
    its silence intervals carry real labels (`sil`/`sp`/`spn`) and are preserved
    verbatim. Two different claims, two representations: empty = "we didn't model
    it here", labelled = "the model says silence here".

## Caveats

- **Mono input.** Pass a mono waveform. Any sample rate is accepted —
  `Wav2Vec2EspeakModel` resamples to its 16 kHz training rate internally (via
  the engine's FFT-domain resampler, `Audio.resample`), so you don't have to.

## References

The forced-align DP:

- Graves et al. (2006), *Connectionist temporal classification* —
  <https://doi.org/10.1145/1143844.1143891>
- Kürzinger et al. (2020), *CTC-Segmentation of Large Corpora* —
  <https://doi.org/10.1007/978-3-030-60276-5_27>

The acoustic model:

- Xu et al. (2022), *Simple and Effective Zero-shot Cross-lingual Phoneme
  Recognition* — <https://doi.org/10.21437/Interspeech.2022-60>
- Baevski et al. (2020), *wav2vec 2.0* — <https://arxiv.org/abs/2006.11477>

Full citations are in the [`sadda.align`](api/align.md) module and the engine
citation registry (`citation_for("sadda.align.forced_align")` /
`"sadda.align.wav2vec2_espeak"`), each of which carries a weblink.
