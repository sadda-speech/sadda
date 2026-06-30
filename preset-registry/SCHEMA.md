# Preset `<id>.toml` schema

A preset is a single self-contained TOML file. The store is a flat directory
of them (one per domain — `presets/mfcc/`, `presets/pitch/`, `presets/formant/`);
the file stem is the preset id.

All domains share the same generic envelope (`sadda_engine::preset::Preset<P>`,
parsed/written by its `from_toml`/`to_toml`) — the **top-level fields** below —
and differ only in the `[params]` table, which is the serialized domain
parameter type. Keep each `[params]` section in sync with its Rust struct.

| Domain | `[params]` = | Rust type | Built-in ids |
|---|---|---|---|
| MFCC | a full `MfccParams` | `sadda_engine::dsp::MfccParams` | librosa-default / kaldi-default / praat-default |
| Pitch | a `method` + a `[params.config]` | `sadda_engine::pitch::PitchParams` | praat-ac / yin / pyin / swipe |
| Formants | LPC method + analysis knobs | `sadda_engine::dsp::FormantsConfig` | praat-burg / autocorrelation |

## Top-level fields (all domains)

```toml
id = "my-asr"                 # required — stable, filesystem-safe slug
                              #   ([A-Za-z0-9_-]+, ≤128 chars; it's the file stem)
version = "1.0.0"             # required — semver
title = "My ASR front-end"   # optional — human-readable label (shown in pickers)
description = "…"             # optional — what it's for, caveats
based_on = "librosa"         # free-text lineage label (domain-specific vocabulary:
                             #   librosa/kaldi/praat · praat/yin/pyin/swipe · praat/…)
faithful = false             # true iff it reproduces `based_on`'s reference output
                             #   to tolerance (default: false)
reference = "librosa 0.11 — librosa.feature.mfcc"   # optional — citation / source
```

## MFCC `[params]`

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

## Pitch `[params]`

The tracking `method` plus a nested `[params.config]` (`PitchConfig`). The
`boersma_*` knobs are read only by `method = "boersma"`, the `yin_*`/`pyin_*`
knobs only by `yin`/`pyin`.

```toml
[params]
method = "boersma"            # autocorrelation | windowed_autocorrelation |
                              #   boersma | yin | pyin | swipe

[params.config]
frame_size_seconds = 0.030
hop_size_seconds = 0.010
min_freq_hz = 75.0            # pitch floor (also the lane's y-axis minimum)
max_freq_hz = 500.0          # pitch ceiling (also the lane's y-axis maximum)
voicing_threshold = 0.45
boersma_max_candidates = 15
boersma_silence_threshold = 0.03
boersma_octave_cost = 0.01
boersma_octave_jump_cost = 0.35
boersma_voiced_unvoiced_cost = 0.14
yin_threshold = 0.1
pyin_n_thresholds = 100
pyin_transition_semitone_cost = 0.5
pyin_voiced_unvoiced_cost = 0.05
pyin_bins_per_semitone = 20
```

## Formant `[params]`

A `FormantsConfig` — the LPC method lives directly in `[params]` (the config
already bundles it; no nested table).

```toml
[params]
frame_size_seconds = 0.025
hop_seconds = 0.010
n_formants = 5
pre_emphasis = 0.97
lpc_method = "burg"           # burg | autocorrelation
max_bandwidth_hz = 1000.0     # drop formants wider than this
min_frequency_hz = 50.0       # ignore roots below this
# lpc_order omitted ⇒ auto (2·n_formants + 2); set an integer to pin it
```

## Notes

- Some fields are **analysis** choices, not reference-defining — editing them
  does not void `faithful` (MFCC `f_min`/`f_max`/`frame_size`/`hop`/`n_mfcc`/
  `n_mels`; pitch `min_freq_hz`/`max_freq_hz`/frame/hop; formant `n_formants`/
  frame/hop). The *algorithmic* knobs (MFCC window/mel-scale/DCT/log/…; the
  pitch `method`; the formant `lpc_method`) are what make a preset faithful to
  its reference; editing those should set `faithful = false`.
- Each domain's built-in ids (e.g. `librosa-default`, `praat-ac`, `praat-burg`)
  are reserved — the store rejects saving over them.
