# Text-to-speech & voiceover

`sadda.tts` is a small, backend-agnostic text-to-speech toolkit. Its
immediate job is **voiceover for generated documentation** (narration for
screencasts and tutorial videos), but the surface is deliberately general —
it works just as well for any ad-hoc "turn this text into a WAV" task.

PROVISIONAL tier — the API may change in minor versions after a deprecation
cycle. See the API reference at [`sadda.tts`](api/tts.md).

## What you need

Synthesis goes through a pluggable **backend**. The default backend,
`espeak-ng`, shells out to the [eSpeak NG](https://github.com/espeak-ng/espeak-ng)
formant synthesizer — so you need the `espeak-ng` binary on your system:

```bash
apt install espeak-ng      # Debian/Ubuntu
brew install espeak-ng     # macOS
# Windows: install from the espeak-ng releases page
```

No Python extra is required for espeak-ng — it is a system binary, not a pip
dependency, so the base `pip install sadda` is enough. eSpeak NG is robotic
but **zero-dependency, fully offline, deterministic, and available on every
platform**, which makes it the right default for reproducible doc builds and
CI. A high-quality neural backend (Kokoro) is planned behind a future
`sadda[tts]` extra; see [Backends](#backends).

## One-shot synthesis

The simplest call — text in, WAV out:

```python
import sadda

result = sadda.tts.synthesize("Hello from sadda.", "hello.wav")
print(result.path, result.sample_rate, result.duration_s)
# hello.wav 22050 1.02
```

`synthesize` returns a [`SynthesisResult`](api/tts.md) describing the file it
wrote (path, sample rate, duration).

## Narration scripts

For anything longer than a line, build a **narration script**: an ordered list
of [`Segment`](api/tts.md)s, each a span of text plus optional per-segment
overrides. The quickest way in is [`parse_script`](api/tts.md), which turns
blank-line-separated paragraphs into one segment each:

```python
script = sadda.tts.parse_script("""
Welcome to sadda, an open-source toolkit for phonetics and speech science.

This narration was generated automatically by the built-in text-to-speech pipeline.
""")

result = sadda.tts.synthesize_script(script, out_dir="build/vo")
print(result.combined)            # build/vo/narration.wav
print(result.total_duration_s)    # 9.34
```

[`synthesize_script`](api/tts.md) synthesizes each segment, writes them to
`out_dir` as `segment_000.wav`, `segment_001.wav`, … and (unless you pass
`assemble=False`) concatenates them into a single `narration.wav`.

### Building scripts explicitly

For control over voice, speaking rate, and pauses, construct the segments
yourself. Segment-level settings win over the script-wide defaults, which win
over the backend defaults:

```python
from sadda.tts import NarrationScript, Segment

script = NarrationScript(
    segments=[
        Segment("First, open a project.", pause_after_s=0.4),
        Segment("Then register a recording as a bundle.", rate=0.95),
    ],
    voice="en-us",     # script-wide default voice
)
```

- **`pause_after_s`** — silence (seconds) appended after the segment when the
  script is assembled into one track.
- **`rate`** — a relative multiplier where `1.0` is the backend's natural
  rate, `1.2` is 20 % faster, `0.9` is 10 % slower.
- **`voice`** — a backend-specific voice id (see [Languages](#languages-and-localization)).
- **`id`** — an optional stable name for the segment (e.g. a scene name),
  carried through untouched so you can align a segment with a screencast
  marker or caption.

## Caching and reproducibility

`synthesize_script` caches every segment by a **content hash** of
`(backend, voice, rate, text)`. Re-running after editing one line
re-synthesizes only that line — everything else is reused from the cache
(default `out_dir/.cache`, override with `cache_dir=`). This is what makes
regenerating documentation cheap and reproducible: no per-run cost, no
network, identical inputs give identical output.

```python
# Edit one paragraph, rebuild — only the changed segment is re-synthesized.
sadda.tts.synthesize_script(edited_script, out_dir="build/vo")
```

## Assembling audio yourself

If you manage per-segment audio separately, [`concat_wavs`](api/tts.md) joins
WAVs into one, inserting silence between them. All inputs must share the same
sample rate, channel count, and sample width (it raises rather than produce a
corrupt file):

```python
sadda.tts.concat_wavs(
    ["intro.wav", "body.wav", "outro.wav"],
    "full.wav",
    pauses_s=[0.5, 0.5, 0.0],   # silence appended after each
)
```

## Backends

A backend is any object satisfying the [`TTSBackend`](api/tts.md) protocol —
a `name` and a `synthesize(text, out_path, *, voice, rate)` method returning a
`SynthesisResult`. Select one by name, or set `$SADDA_TTS_BACKEND`:

```python
sadda.tts.list_backends()          # ['espeak-ng', 'kokoro']
sadda.tts.synthesize("hi", "hi.wav", backend="espeak-ng")
```

| Backend | Status | Notes |
|---|---|---|
| `espeak-ng` | **default** | Offline formant synth; robotic but reproducible, CI-safe, [100+ languages](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md) |
| `kokoro` | *pending* | Planned high-quality neural default (Kokoro-82M, Apache-2.0). Registered but not yet wired — raises an actionable error until the `sadda[tts]` extra lands |

Cloud backends (ElevenLabs / OpenAI) are intended to plug in the same way as
opt-in add-ons.

### Writing your own backend

Because the contract is a structural protocol, you can supply your own backend
— no subclassing required — and register it by name:

```python
from pathlib import Path
from sadda.tts import SynthesisResult, register_backend

class MyBackend:
    name = "my-tts"

    def synthesize(self, text, out_path, *, voice=None, rate=None):
        out_path = Path(out_path)
        # ... write a WAV to out_path ...
        return SynthesisResult(path=out_path, sample_rate=24000, duration_s=...)

register_backend("my-tts", MyBackend)
sadda.tts.synthesize("hi", "hi.wav", backend="my-tts")
```

You can also pass a backend *instance* directly (`backend=MyBackend()`), which
is handy for tests and for backends that need constructor arguments.

## Languages and localization

The `espeak-ng` backend covers [**100+ languages and regional
variants**](https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md)
— far more than the neural backends here, which (like most neural TTS voices)
are trained per-language. Pass the language as the `voice`; list what your
install supports with `espeak-ng --voices`:

```python
en = NarrationScript.from_texts(sentences_en, voice="en-us")
de = NarrationScript.from_texts(sentences_de, voice="de")
fr = NarrationScript.from_texts(sentences_fr, voice="fr-fr")
```

Regional variants are available where they matter — e.g. `en-us`,
`en-gb`, `en-gb-scotland`, `en-gb-x-rp`; `es` (Spain) vs `es-419` (Latin
America); `pt` vs `pt-br`; `cmn` (Mandarin) vs `yue` (Cantonese). Because the
cache keys on voice, a multi-language build synthesizes each locale
independently and only redoes what changed.

Two honest caveats for localization:

1. **`sadda.tts` gives you the *voice*, not the *translation*.** The pipeline
   speaks whatever text you hand it; producing the translated strings per
   locale is your responsibility (your own translations, or a translation step
   upstream).
2. **espeak-ng quality is uneven across languages,** and some need
   preprocessing. Its letter-to-sound rules are far more polished for major
   languages than long-tail ones — listen to each target locale before relying
   on it.

!!! note "espeak-ng and Japanese"
    In our testing (espeak-ng 1.50), eSpeak NG does no Japanese
    word-segmentation and has no kanji dictionary, so raw Japanese text
    mispronounces badly. To get clean output you must
    preprocess: convert **kanji → kana**, insert **spaces between words**
    (without them the `ー` long-vowel mark is misread), and write grammatical
    particles phonetically (は → ワ, へ → エ). A tokenizer such as
    `fugashi`/MeCab plus a kana converter like `pykakasi` does all three. This
    kind of per-language preprocessing is on the roadmap.

## Scope

This module covers **audio voiceover only**. Capturing the screencast/gif and
muxing the narration against video are a separate concern (and a separate
roadmap item) — `sadda.tts` produces the audio track you would then combine
with captured video using a tool like `ffmpeg`.
