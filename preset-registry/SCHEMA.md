# MFCC preset `<id>.toml` schema

A preset is a single self-contained TOML file. The store is a flat directory
of them; the file stem is the preset id.

The manifest is parsed by `sadda_engine::dsp::preset::MfccPreset` (Rust) and
written by the same type's `to_toml`. The `[params]` table is a serialized
`sadda_engine::dsp::MfccParams`; keep this doc in sync with that struct.

## Top-level fields

```toml
id = "my-asr"                 # required — stable, filesystem-safe slug
                              #   ([A-Za-z0-9_-]+, ≤128 chars; it's the file stem)
version = "1.0.0"             # required — semver
title = "My ASR front-end"   # optional — human-readable label (shown in pickers)
description = "…"             # optional — what it's for, caveats
based_on = "librosa"         # librosa | kaldi | praat | htk | custom  (default: custom)
faithful = false             # true iff it reproduces `based_on`'s reference
                             #   through mfcc_with_params to tolerance (default: false)
reference = "librosa 0.11 — librosa.feature.mfcc"   # optional — citation / source
```

## `[params]` — the parameter set

Every stage of the MFCC pipeline is one scalar or enum. Enum values are the
snake_case names below.

```toml
[params]
frame_size_seconds = 0.025    # base analysis-window duration (before the factor)
hop_seconds = 0.010           # frame advance
n_mfcc = 13                   # output columns (column 0 is c0)
f_min = 0.0                   # lowest filterbank frequency (Hz)
f_max = 8000.0                # highest filterbank frequency (Hz; typically Nyquist)
window = "periodic_hann"      # periodic_hann | povey | praat_gaussian | hamming
window_duration_factor = 1.0  # window length = frame_size · factor · sr (Praat: 2.0)
framing = "centered"          # centered | snip_edges
fft = "window_length"         # window_length | next_pow2
remove_dc = false             # subtract per-frame mean before windowing (Kaldi)
pre_emphasis = 0.0            # y[n] = x[n] − α·x[n−1]  (0 = none)
mel_scale = "slaney"          # slaney | htk
filter_norm = "area_slaney"   # area_slaney | unit_peak
triangle_in_mel = false       # triangle slopes linear in mel (Kaldi) vs Hz
exclude_nyquist_bin = false   # drop the Nyquist FFT bin from the bank (Kaldi)
dct = "ortho"                 # ortho | unnormalized
power_norm = "raw"            # raw | praat_duration
lifter = 0.0                  # sinusoidal cepstral lifter L  (0 = none)
```

### `[params.filters]` — filterbank layout (internally tagged by `kind`)

```toml
# n filters spread over [f_min, f_max] (librosa, Kaldi):
[params.filters]
kind = "n_mels"
n_mels = 40

# OR: centres every step_mel from first_mel, count derived to Nyquist (Praat):
[params.filters]
kind = "mel_spacing"
first_mel = 100.0
step_mel = 100.0
```

### `[params.log]` — log compression (internally tagged by `kind`)

```toml
# 10·log10(max(e, amin)/reference), optional global top_db floor (librosa/Praat):
[params.log]
kind = "db"
reference = 1.0
amin = 1e-10
top_db = 80.0        # optional — omit for no global dynamic-range floor (Praat)

# OR: ln(max(e, floor))  (Kaldi):
[params.log]
kind = "natural_ln"
floor = 1.1920929e-7
```

## Notes

- `f_min` / `f_max` / `frame_size_seconds` / `hop_seconds` / `n_mfcc` /
  `n_mels` are **analysis** choices, not reference-defining — editing them does
  not void `faithful`. The *algorithmic* knobs (window, mel scale, filter norm,
  DCT, log, power scaling, framing, FFT rule, pre-emphasis) are what make a
  preset faithful to its reference; editing those should set `faithful = false`.
- Built-in ids (`librosa-default`, `kaldi-default`, `praat-default`) are
  reserved — the store rejects saving over them.
