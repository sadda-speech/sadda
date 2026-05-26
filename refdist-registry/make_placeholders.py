#!/usr/bin/env python3
"""Generate the PLACEHOLDER reference distributions (C8).

These are **synthetic, non-authoritative** distributions whose only job is
to exercise the registry format, validator, index builder, and the
engine's resolver/`install_from_dir` end to end *before* real,
license-cleared data is sourced (Hillenbrand 1995, Peterson-Barney 1952,
clinical normative ranges). Every manifest's title + provenance says
PLACEHOLDER in capitals, and the data is drawn from fixed seeds — not from
any real corpus.

Run from the repo root::

    python refdist-registry/make_placeholders.py

Writes:
  refdist-bundled/<dist>/                  (tier 1 — bundled with the app)
  refdist-registry/tier2/<dist>/           (curated registry)
  refdist-registry/tier3/<dist>/           (community registry)

Each distribution directory gets: refdist.toml, data.parquet,
provenance.md, LICENSE. Deterministic — re-running reproduces it.
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
BUNDLED = REPO / "refdist-bundled"
REGISTRY = REPO / "refdist-registry"

# SPDX id → short LICENSE notice. We do NOT reproduce the full legal text
# from memory; the notice points at the canonical source (real
# distributions ship the upstream LICENSE verbatim).
LICENSE_NOTICE = {
    "CC0-1.0": (
        "This PLACEHOLDER dataset is dedicated to the public domain under "
        "Creative Commons CC0 1.0 Universal.\n"
        "Full legal code: https://creativecommons.org/publicdomain/zero/1.0/legalcode\n"
    ),
    "CC-BY-4.0": (
        "This PLACEHOLDER dataset is licensed under Creative Commons "
        "Attribution 4.0 International (CC BY 4.0).\n"
        "Full legal code: https://creativecommons.org/licenses/by/4.0/legalcode\n"
    ),
}


def _write(dist_dir: Path, manifest: str, df: pl.DataFrame, spdx: str, provenance: str) -> None:
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "refdist.toml").write_text(textwrap.dedent(manifest).lstrip())
    df.write_parquet(dist_dir / "data.parquet")
    (dist_dir / "LICENSE").write_text(LICENSE_NOTICE[spdx])
    (dist_dir / "provenance.md").write_text(textwrap.dedent(provenance).lstrip())


def formant_norms_placeholder() -> None:
    """Tier 1 (bundled): synthetic AmE vowel F1/F2 observed distribution."""
    rng = random.Random("formant-placeholder")
    # Rough vowel targets so the cloud at least looks vowel-shaped; jittered.
    targets = {"iy": (300, 2300), "ae": (700, 1700), "uw": (350, 900), "ah": (700, 1200)}
    rows = []
    speaker = 0
    for sex in ("m", "f"):
        for _ in range(8):  # 8 speakers per sex → 16 total (≥ min_n)
            speaker += 1
            for phone, (f1, f2) in targets.items():
                rows.append(
                    {
                        "speaker_id": speaker,
                        "sex": sex,
                        "phone": phone,
                        "F1": round(f1 + rng.gauss(0, 25), 1),
                        "F2": round(f2 + rng.gauss(0, 60), 1),
                    }
                )
    df = pl.DataFrame(rows)
    manifest = """
        id = "placeholder-amE-vowels"
        version = "0.1.0"
        title = "PLACEHOLDER American English vowel formants (synthetic)"
        license = "CC0-1.0"

        [citation]
        authors = ["sadda placeholder generator"]
        year = 2026

        [population]
        language = "eng"
        variety = "AmE"
        sex = ["m", "f"]
        age_band = ["adult"]
        n_speakers = 16

        [measure]
        kind = "observed_distribution"
        parameters = ["F1", "F2"]
        units = "Hz"
        phones = ["iy", "ae", "uw", "ah"]
        context = "isolated"
        measurement_method = "synthetic placeholder — NOT a real measurement"

        [privacy]
        shareability = "raw_samples"
        min_n_per_subgroup = 5

        [schema]
        data_file = "data.parquet"
        shape = "long"
        columns = ["speaker_id", "sex", "phone", "F1", "F2"]
    """
    provenance = """
        # PLACEHOLDER — synthetic data

        This is **not** a real reference distribution. It exists only to
        exercise the registry/resolver pipeline (C8) until license-cleared
        data (e.g. Hillenbrand et al. 1995) is sourced. Values are random
        draws around rough vowel targets with a fixed seed.
    """
    _write(BUNDLED / "placeholder-amE-vowels", manifest, df, "CC0-1.0", provenance)


def f0_norms_placeholder() -> None:
    """Tier 2 (curated): synthetic adult f0 summary normative range."""
    rows = [
        {"sex": "m", "stat": "mean", "f0": 120.0},
        {"sex": "m", "stat": "sd", "f0": 18.0},
        {"sex": "f", "stat": "mean", "f0": 210.0},
        {"sex": "f", "stat": "sd", "f0": 22.0},
    ]
    df = pl.DataFrame(rows)
    manifest = """
        id = "placeholder-f0-norms"
        version = "0.1.0"
        title = "PLACEHOLDER adult speaking f0 normative range (synthetic)"
        license = "CC-BY-4.0"

        [citation]
        authors = ["sadda placeholder generator"]
        year = 2026

        [population]
        language = "eng"
        sex = ["m", "f"]
        age_band = ["adult"]
        n_speakers = 200

        [measure]
        kind = "summary_normative_range"
        parameters = ["f0"]
        units = "Hz"

        [privacy]
        shareability = "summary_only"
        min_n_per_subgroup = 5

        [schema]
        data_file = "data.parquet"
        shape = "long"
        columns = ["sex", "stat", "f0"]
    """
    provenance = """
        # PLACEHOLDER — synthetic summary

        Not a real normative range. Summary-only (no raw samples), as a
        clinical-norm distribution typically would be. Replace with a
        license-cleared published normative set.
    """
    _write(REGISTRY / "tier2" / "placeholder-f0-norms", manifest, df, "CC-BY-4.0", provenance)


def vot_norms_placeholder() -> None:
    """Tier 3 (community): synthetic VOT observed distribution."""
    rng = random.Random("vot-placeholder")
    rows = []
    speaker = 0
    for _ in range(10):  # 10 speakers
        speaker += 1
        for stop in ("p", "t", "k"):
            base = {"p": 60, "t": 75, "k": 85}[stop]
            rows.append(
                {
                    "speaker_id": speaker,
                    "stop": stop,
                    "vot_ms": round(base + rng.gauss(0, 10), 1),
                }
            )
    df = pl.DataFrame(rows)
    manifest = """
        id = "placeholder-vot-norms"
        version = "0.1.0"
        title = "PLACEHOLDER voiceless-stop VOT (synthetic)"
        license = "CC0-1.0"

        [citation]
        authors = ["sadda placeholder generator"]
        year = 2026

        [population]
        language = "eng"
        sex = ["m", "f"]
        age_band = ["adult"]
        n_speakers = 10

        [measure]
        kind = "observed_distribution"
        parameters = ["vot_ms"]
        units = "ms"
        phones = ["p", "t", "k"]

        [privacy]
        shareability = "raw_samples"
        min_n_per_subgroup = 5

        [schema]
        data_file = "data.parquet"
        shape = "long"
        columns = ["speaker_id", "stop", "vot_ms"]
    """
    provenance = """
        # PLACEHOLDER — synthetic data

        Community-tier example only. Not a real VOT corpus.
    """
    _write(REGISTRY / "tier3" / "placeholder-vot-norms", manifest, df, "CC0-1.0", provenance)


def main() -> None:
    formant_norms_placeholder()
    f0_norms_placeholder()
    vot_norms_placeholder()
    print("wrote placeholder distributions to refdist-bundled/ and refdist-registry/{tier2,tier3}/")


if __name__ == "__main__":
    main()
