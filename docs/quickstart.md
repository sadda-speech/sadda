# Quickstart

This walk-through covers the things you'll do in your first session
with sadda: create a project, register a recording as a bundle, run
pitch and formants, and query the results.

## Install

```bash
pip install sadda           # core
pip install "sadda[ml]"     # adds onnxruntime — needed for VAD + embeddings
```

## Create a project

A *project* is a directory containing audio, derived signals,
annotations, and a SQLite-backed corpus database. Create one:

```python
import sadda
from pathlib import Path

proj = sadda.new_project(Path("vowels"), name="vowel-study")
```

The directory `vowels/` now contains:

```
vowels/
├── corpus.db              # SQLite — bundles, tiers, annotations, provenance
├── project.toml           # Project metadata
├── signals/
│   ├── original/          # Source audio (copied at registration)
│   ├── derived/           # Parquet sidecars for dense tiers
│   └── .in_progress/      # Live-recording staging
├── attachments/
├── exports/
└── recipes/               # Auto-generated reproducibility scripts
```

## Register a bundle

A *bundle* is one recording with its metadata. Registering it copies
the WAV into `signals/original/` and inserts a `bundle` row:

```python
bundle_id = proj.add_bundle("speaker_01_take_1", Path("rec01.wav"))
```

You can also attach a JSON `extra` payload, and link the bundle to a
`Speaker` or `Session` (created via `proj.add_speaker(...)` /
`proj.add_session(...)`) by passing their ids:

```python
speaker_id = proj.add_speaker("S01")

bundle_id = proj.add_bundle(
    "speaker_01_take_2",
    Path("rec02.wav"),
    speaker_id=speaker_id,
    extra='{"elicitation": "rainbow_passage", "take": 2}',
)
```

## Large recordings

Loading a bundle decodes the whole file into memory, so a single very
long recording (hours) can be slow — or exceed RAM on a smaller machine.
You can **probe** a file's size from its header alone first, with no
samples decoded:

```python
info = sadda.probe_wav(Path("interview_3h.wav"))
print(info.duration_seconds, info.decoded_bytes)  # decoded_bytes ≈ the RAM a full load costs
```

If a file is too large to work with comfortably, split it into
contiguous pieces — each registered as its own bundle (`<prefix>_001`,
`_002`, …). The split **streams** the source, so memory stays flat
regardless of length:

```python
ids = proj.add_bundle_split(
    "interview", Path("interview_3h.wav"), chunk_seconds=600  # 10-minute pieces
)
```

In the desktop app this is automatic: **File → Add Bundle…** probes the
file first, and if it's large enough to be risky it offers to split it
(or add it as-is) before loading. (`probe_wav` is provisional and warns
once on first use.) To bring in a whole corpus at once, **File → Add
Directory…** registers every `.wav` in a folder (sorted by name), each
through the same large-file guard.

## Run pitch and formants

The DSP surface (`sadda.dsp.*`) is functional — every function takes
an `Audio` (or, for some, NumPy `float32` arrays plus a sample rate)
and returns NumPy or dataclass results. No corpus dependency:

```python
audio = proj.load_audio(bundle_id)

times, freqs, voicing = sadda.dsp.voiced_pitch(
    audio,
    frame_size_seconds=0.030,
    hop_size_seconds=0.010,
    min_freq_hz=75.0,
    max_freq_hz=500.0,
)

formants = sadda.dsp.formants(audio, n_formants=4)
```

## Clinical measures

`sadda.clinical.*` adds voice-quality measures — jitter, shimmer,
HNR, CPP / CPPS, H1–H2, GNE, and the AVQI / ABI composite indices.
Every measure is a pure function over an `Audio`:

```python
perturbation = sadda.clinical.perturbation(audio)
print(f"jitter local: {perturbation.jitter_local:.4f}")
print(f"shimmer local dB: {perturbation.shimmer_local_db:.3f}")

hnr_db  = sadda.clinical.hnr(audio)
cpps_db = sadda.clinical.cpps(audio)
```

All clinical measures are research-use only. They live in their own
`stable_clinical` tier — the API commitment is the same as Stable,
but the tier name flags the clinical-research caveat (see the
[stability tiers](index.md#stability-tiers) table).

## Reference distributions

`sadda.refdist.*` lets you compare a measurement against normative
ranges, target zones, or observed corpora. The bundled set ships with
the wheel; the desktop app's View menu has an "Install bundled
reference data" command that seeds the per-user cache, and the same
distributions are available from Python:

```python
sadda.refdist.install("refdist-bundled/placeholder-amE-vowels")
vowels = sadda.refdist.get("placeholder-amE-vowels", "0.1.0")
print(vowels.summary("F1").mean, vowels.summary("F1").sd)
```

## ML inference (voice activity, embeddings)

With `sadda[ml]` installed, `sadda.ml.vad` runs the bundled Silero
VAD over an `Audio` and returns per-window speech probabilities:

```python
times, probs = sadda.ml.vad(audio)
for start, end in sadda.ml.speech_segments(audio, threshold=0.5):
    print(f"speech {start:.2f}–{end:.2f}s")
```

`sadda.ml.load_model("hf://<org>/<repo>/<file>")` (download-enabled
builds only) lets you pull wav2vec2 / Whisper-style ONNX models and
extract embeddings as a B3 continuous-vector tier — see
[`sadda.ml`](api/ml.md) for the full surface.

## Import existing annotations

Open Praat TextGrids or ELAN .eaf files directly into a bundle:

```python
proj.import_textgrid(Path("phones.TextGrid"), bundle_id)
proj.import_eaf(Path("annotations.eaf"), bundle_id)
```

Round-trip semantics (what's preserved, what's lost) are documented
under [Round-trip lossiness](lossiness/textgrid.md).

## Query annotations as a DataFrame

Every tier can be pulled into a Polars DataFrame. `proj.query` takes
the integer tier id returned by `add_tier(...)` or `import_textgrid(...)`:

```python
import polars as pl

[phones_tier] = proj.import_textgrid(Path("phones.TextGrid"), bundle_id)
df = proj.query(phones_tier)
print(df.head())
# shape: (2, 8)
# ┌─────┬─────────┬───────────────┬─────────────┬──────────────────┬───────┬──────────────────────┬───────┐
# │ id  ┆ tier_id ┆ start_seconds ┆ end_seconds ┆ duration_seconds ┆ label ┆ parent_annotation_id ┆ extra │
# ├─────┼─────────┼───────────────┼─────────────┼──────────────────┼───────┼──────────────────────┼───────┤
# │ 1   ┆ 1       ┆ 0.0           ┆ 0.12        ┆ 0.12             ┆ h     ┆ null                 ┆ null  │
# │ 2   ┆ 1       ┆ 0.12          ┆ 0.27        ┆ 0.15             ┆ ɛ     ┆ null                 ┆ null  │
# └─────┴─────────┴───────────────┴─────────────┴──────────────────┴───────┴──────────────────────┴───────┘
```

## Record an analysis recipe

A *recipe* is a reproducibility primitive — the operations you run
inside a `with sadda.recipe.record(...):` block are linked to a named
record in the corpus, and a runnable `.py` script is emitted to
`<project>/recipes/<name>.py`:

```python
with sadda.recipe.record(proj, name="phone_import_v1"):
    proj.import_textgrid(Path("phones.TextGrid"), bundle_id)
    proj.import_eaf(Path("annotations.eaf"), bundle_id)

# Re-run later:
#   python vowels/recipes/phone_import_v1.py
```

Recipes capture the calls that already produce `processing_run` rows
in the corpus: TextGrid / EAF imports and live recordings in 0.1.0.
Pure-DSP calls are reproducible from your own script.

## Live recording

To record from a microphone:

```python
session = sadda.live.start_session(proj, name="practice_take_1",
                                   sample_rate=44100, channels=1)

@session.on_meter
def show(peak, rms, rms_db, t):
    print(f"{t:.2f}s  rms={rms_db:+.1f} dB-FS")

session.start()
import time; time.sleep(5.0)
session.stop()

bundle_id = session.commit(proj)
```

The live surface includes `on_meter`, `on_pitch`, `on_intensity`,
`on_formants` subscribers — see [`sadda.live`](api/live.md) for the
full surface.

## Where to go next

- [API reference](api/corpus.md) for every public class and function.
- [Round-trip lossiness](lossiness/textgrid.md) when you need to know
  exactly what survives an import/export round-trip.
- The `DEVLOG.md` in the repo for the design history.
