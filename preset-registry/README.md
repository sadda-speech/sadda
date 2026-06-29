# sadda MFCC preset registry

An MFCC **preset** is a named, reusable parameter set for
`sadda.dsp.mfcc(audio, params=…)` — a point in the [`MfccParams`] space plus
provenance (which authoritative reference it derives from, whether it's a
faithful reproduction, a citation).

Unlike the [reference-distribution](../refdist-registry/) and
[model](../model-registry/) registries — which pair a metadata manifest with a
separate **data payload** (Parquet, model weights), hence a directory per
entry — a preset has *no payload*: the parameters **are** the content. So a
preset is a single self-contained `<id>.toml` file, and a store is a flat
directory of them.

The format is documented in [SCHEMA.md](SCHEMA.md). It is parsed by
`sadda_engine::dsp::preset` (Rust) and surfaced through Python as
`sadda.dsp.mfcc_presets()` / `mfcc_preset(id)` / `save_mfcc_preset(...)` and in
the desktop app's **View ▸ MFCC** menu.

## Built-in vs. user presets

- **Built-in** presets (`librosa-default`, `kaldi-default`, `praat-default`)
  live in **code** (`sadda_engine::dsp::preset::builtin_presets`), where they
  are golden-tested against their references. They are surfaced by every store
  but are **never written to disk**, and their ids are reserved — saving a
  preset with a built-in id is rejected, so the authoritative presets can
  never drift. This directory therefore ships **no** built-in `.toml` files;
  SCHEMA.md shows one inline for reference.

- **User** presets live in the per-user store
  (`~/.local/share/sadda/presets/mfcc/` on Linux, the platform equivalent
  elsewhere), one `<id>.toml` per preset. Create them in the GUI (View ▸ MFCC ▸
  Edit parameters… → save) or from Python (`save_mfcc_preset`).

## Faithfulness

`faithful = true` means: running the preset through the unified
`mfcc_with_params` pipeline reproduces `based_on`'s reference output to
tolerance. The built-in `librosa`/`kaldi` presets are faithful; the `praat`
preset is **not** (the shared pipeline is f32, while faithful Praat needs the
dedicated f64 path — `sadda.dsp.mfcc(audio, method="praat")`). Editing any
reference-defining knob of a preset voids faithfulness, and the GUI/Python
flag the result as a custom set.

## Status

Local-config registry — no tiered governance / CI validator (those exist for
the data registries because they vet redistributed corpora and weights; a
preset is a few dozen scalars the user owns). If presets later become
shareable, this is where a validator/index would land.
