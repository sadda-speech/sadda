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

# audio: a mono 16 kHz float32 waveform (resample first if yours isn't 16 kHz)
audio = load_your_audio_as_16k_mono()

model = sadda.align.Wav2Vec2EspeakModel.from_pretrained("sadda-speech/wav2vec2-espeak-ctc")
alignment = sadda.align.align(audio, 16000, "the words that were said", model=model)

for w in alignment.words:
    print(f"{w.start_seconds:.2f}-{w.end_seconds:.2f}  {w.text}  [{' '.join(p.label for p in w.phones)}]")
```

`align` returns an [`Alignment`](api/align.md) with `words` ([`TimedWord`](api/align.md))
and a flat `phones` list ([`TimedPhone`](api/align.md)) — each carrying
`start_seconds`, `end_seconds`, and (for phones) an alignment-confidence `score`.

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

## Bring your own transcript (for now)

This slice is **alignment-first**: you supply the transcript. Automatic
transcription (Whisper ASR) for the no-transcript case is a later slice. The
transcript needn't be perfectly tokenized — punctuation is stripped, and each
whitespace-separated word becomes one `TimedWord`.

## Languages

`espeak-ng` covers 100+ languages, and the acoustic model is multilingual, so
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

!!! note "Inferred, not modeled, silence"
    Neither detector is a *trained* silence model — `blank` infers silence from
    "no phoneme emitted", `vad` from a separate detector. Truly **modeled**
    silence (an explicit `sil` state) comes with the planned **MFA passthrough**
    backend, whose HMM models optional silence directly.

## Caveats

- **16 kHz mono input.** `Wav2Vec2EspeakModel` requires 16 kHz; resample first
  (a built-in resample is a planned refinement).

## References

The forced-align DP: Graves et al. (2006, CTC), Kürzinger et al. (2020,
CTC-Segmentation). The acoustic model: Xu et al. (2022), Baevski et al. (2020).
Full citations are in the [`sadda.align`](api/align.md) module and the engine
citation registry (`citation_for("sadda.align.forced_align")` /
`"sadda.align.wav2vec2_espeak"`).
