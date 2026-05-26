#!/usr/bin/env python3
"""Validate reference-distribution directories against the registry rules
(C8 CI gate). Self-contained — stdlib ``tomllib`` + ``polars`` only, no
dependency on the sadda wheel, so this travels when the registry becomes
its own repo.

Checks per distribution directory:
  * `refdist.toml` parses and has the required fields (id, version,
    measure.kind, schema.data_file);
  * `measure.kind` is one of the three known kinds;
  * a `license` SPDX id is declared and is on the allowlist for the tier
    (tier 2 disallows NonCommercial / NoDerivatives; tier 3 warns);
  * a non-empty `LICENSE` file is present;
  * the declared data file exists and its columns match `schema.columns`;
  * `privacy.min_n_per_subgroup` is present, and for `raw_samples` the
    data has at least that many distinct `speaker_id`s (a k-anonymity
    proxy; tier 2 errors, tier 3 warns).

Usage::

    python validate.py <root> [<root> ...]      # each root holds dist dirs
    python validate.py --dist <dist_dir>        # one distribution dir

Exit code is nonzero if any ERROR was found. WARN does not fail tier 3.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import polars as pl

KNOWN_KINDS = {"observed_distribution", "summary_normative_range", "target_zone"}
# SPDX ids acceptable for the curated tier; tier 3 may use others (warned).
TIER2_ALLOWED = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "ODC-BY-1.0", "ODbL-1.0"}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: str, msg: str) -> None:
        self.errors.append(f"ERROR {where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"WARN  {where}: {msg}")


def _tier_of(dist_dir: Path) -> int:
    """Infer tier from a `tierN` ancestor directory; default 3 (lowest bar)."""
    for part in dist_dir.parts:
        if part == "tier1":
            return 1
        if part == "tier2":
            return 2
        if part == "tier3":
            return 3
    return 3


def validate_dist(dist_dir: Path, rep: Report) -> None:
    where = str(dist_dir)
    manifest_path = dist_dir / "refdist.toml"
    if not manifest_path.is_file():
        rep.error(where, "no refdist.toml")
        return
    try:
        m = tomllib.loads(manifest_path.read_text())
    except tomllib.TOMLDecodeError as e:
        rep.error(where, f"refdist.toml does not parse: {e}")
        return

    for field in ("id", "version"):
        if not m.get(field):
            rep.error(where, f"missing required top-level field '{field}'")

    tier = _tier_of(dist_dir)
    measure = m.get("measure", {})
    kind = measure.get("kind", "observed_distribution")
    if kind not in KNOWN_KINDS:
        rep.error(where, f"unknown measure.kind {kind!r}")

    # License: SPDX id + LICENSE file.
    spdx = m.get("license")
    if not spdx:
        rep.error(where, "no top-level `license` SPDX id")
    else:
        upper = spdx.upper()
        if "-NC" in upper or "-ND" in upper:
            if tier == 2:
                rep.error(where, f"license {spdx} (NC/ND) not allowed in tier 2")
            else:
                rep.warn(where, f"license {spdx} is NC/ND — flagged")
        elif tier == 2 and spdx not in TIER2_ALLOWED:
            rep.warn(where, f"license {spdx} not on the tier-2 allowlist {sorted(TIER2_ALLOWED)}")
    license_file = dist_dir / "LICENSE"
    if not (license_file.is_file() and license_file.read_text().strip()):
        rep.error(where, "missing or empty LICENSE file")

    # Privacy / k-anonymity.
    privacy = m.get("privacy", {})
    min_n = privacy.get("min_n_per_subgroup")
    if min_n is None:
        rep.error(where, "privacy.min_n_per_subgroup is required")

    # Data file + schema conformance.
    schema = m.get("schema", {})
    data_file = schema.get("data_file")
    if not data_file:
        rep.error(where, "schema.data_file is required")
        return
    data_path = dist_dir / data_file
    if not data_path.is_file():
        rep.error(where, f"declared data file {data_file} is missing")
        return
    try:
        df = pl.read_parquet(data_path)
    except Exception as e:  # noqa: BLE001
        rep.error(where, f"cannot read {data_file}: {e}")
        return
    declared = schema.get("columns")
    if declared and list(df.columns) != list(declared):
        rep.error(
            where,
            f"data columns {df.columns} != declared schema.columns {declared}",
        )

    # k-anonymity proxy for raw-sample distributions.
    if privacy.get("shareability") == "raw_samples" and min_n is not None:
        if "speaker_id" in df.columns:
            n_speakers = df["speaker_id"].n_unique()
            if n_speakers < min_n:
                msg = f"{n_speakers} distinct speakers < min_n_per_subgroup {min_n}"
                (rep.error if tier == 2 else rep.warn)(where, msg)


def iter_dist_dirs(root: Path):
    """A directory is a distribution iff it contains a refdist.toml."""
    if (root / "refdist.toml").is_file():
        yield root
        return
    for child in sorted(root.rglob("refdist.toml")):
        yield child.parent


def main(argv: list[str]) -> int:
    args = argv[1:]
    rep = Report()
    if args and args[0] == "--dist":
        roots = [Path(a) for a in args[1:]]
        dirs = roots
    else:
        roots = [Path(a) for a in args] or [Path(__file__).resolve().parent]
        dirs = [d for r in roots for d in iter_dist_dirs(r)]

    if not dirs:
        print("no distributions found", file=sys.stderr)
        return 1
    for d in dirs:
        validate_dist(d, rep)

    for w in rep.warnings:
        print(w)
    for e in rep.errors:
        print(e, file=sys.stderr)
    n = len(dirs)
    if rep.errors:
        print(f"\n{len(rep.errors)} error(s) across {n} distribution(s)", file=sys.stderr)
        return 1
    print(f"\nOK — {n} distribution(s) valid ({len(rep.warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
