# `refdist.toml` schema

A distribution is a directory containing:

```
refdist.toml      manifest (this schema)
data.parquet      the data — raw samples OR summaries, per `privacy.shareability`
provenance.md     human-readable: paper, method, sampling procedure
LICENSE           the data license, verbatim
```

The manifest is parsed by `sadda_engine::refdist` (Rust) and by this
registry's `validate.py` / `build_index.py` (stdlib `tomllib`). Keep the two
in sync.

## Fields

```toml
id = "hillenbrand-1995-amE-vowels"   # required — stable slug
version = "1.0.0"                    # required — semver, immutable once published
title = "American English vowel formants (Hillenbrand et al. 1995)"
doi = "10.1121/1.411872"             # optional
license = "CC-BY-4.0"                # SPDX id; enforced per tier by CI

[citation]
authors = ["Hillenbrand, J.", "Getty, L.", "Clark, M.", "Wheeler, K."]
year = 1995
journal = "JASA"
bibtex = "..."                       # optional, paste-ready

[population]                         # the facets `query` matches on
language = "eng"                     # ISO 639-3
variety = "AmE"
sex = ["m", "f", "c"]
age_band = ["adult", "child"]
n_speakers = 139
n_tokens = 1668

[measure]
kind = "observed_distribution"       # observed_distribution | summary_normative_range | target_zone
parameters = ["F1", "F2", "F3"]
units = "Hz"
phones = ["iy", "ih", "eh", "ae"]    # optional
context = "hVd"                      # optional
measurement_method = "steady-state, manually selected"

[privacy]
shareability = "raw_samples"         # raw_samples | summary_only
min_n_per_subgroup = 5               # required; k-anonymity floor
community_consent = false            # required true for small-language community data

[schema]
data_file = "data.parquet"
shape = "long"                       # long | wide
columns = ["speaker_id", "phone", "F1", "F2", "F3"]
```

## `measure.kind`

Kept distinct so the GUI never conflates "what people sound like" with "what
to aim for":

- **`observed_distribution`** — raw samples from a measured population.
- **`summary_normative_range`** — summary stats only (mean / SD / percentiles); no raw values shipped.
- **`target_zone`** — prescriptive goal region (voice-coach / L2 use); not an empirical claim.

## CI gates (`validate.py`)

- required fields present; `measure.kind` is one of the three;
- `license` declared + a non-empty `LICENSE` file; tier-2 disallows NC/ND;
- `privacy.min_n_per_subgroup` present; for `raw_samples`, distinct
  `speaker_id` count ≥ `min_n_per_subgroup` (tier 2 errors, tier 3 warns);
- `data_file` exists and its columns equal `schema.columns`.
