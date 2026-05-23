# Quickstart

This walk-through covers the things you'll do in your first session
with sadda: create a project, register a recording as a bundle, run
pitch and formants, and query the results.

## Install

```bash
pip install sadda
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

You can also register with extras:

```python
bundle_id = proj.add_bundle("speaker_01_take_1", Path("rec01.wav"))
```

## Run pitch and formants

The DSP surface (`sadda.dsp.*`) is functional — every function takes
NumPy `float32` arrays and a sample rate, and returns NumPy or
dataclass results. No corpus dependency:

```python
audio = proj.load_audio(bundle_id)

pitch = sadda.dsp.voiced_pitch(
    audio.samples.astype("float32"),
    audio.sample_rate,
    frame_size_seconds=0.030,
    hop_size_seconds=0.010,
    min_freq_hz=75.0,
    max_freq_hz=500.0,
)

formants = sadda.dsp.formants(
    audio.samples.astype("float32"),
    audio.sample_rate,
    n_formants=4,
)
```

## Import existing annotations

Open Praat TextGrids or ELAN .eaf files directly into a bundle:

```python
proj.import_textgrid(Path("phones.TextGrid"), bundle_id)
proj.import_eaf(Path("annotations.eaf"), bundle_id)
```

Round-trip semantics (what's preserved, what's lost) are documented
under [Round-trip lossiness](lossiness/textgrid.md).

## Query annotations as a DataFrame

Every tier can be pulled into a Polars DataFrame:

```python
import polars as pl

df = proj.query(tier_id="phones")
print(df)
# ┌─────┬──────────┬──────────┬───────┐
# │ id  ┆ start_s  ┆ end_s    ┆ label │
# ├─────┼──────────┼──────────┼───────┤
# │ 1   ┆ 0.0      ┆ 0.12     ┆ h     │
# │ 2   ┆ 0.12     ┆ 0.27     ┆ ɛ     │
# └─────┴──────────┴──────────┴───────┘
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
