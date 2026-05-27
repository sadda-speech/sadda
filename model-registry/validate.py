#!/usr/bin/env python3
"""Validate model-registry entries against the registry rules (E11 part 3b
CI gate). Self-contained — stdlib ``tomllib`` only (no ``polars``, no sadda
wheel), so it travels when the registry becomes its own repo.

Model-registry validation is **shallower than refdist's** by design (per
the 2026-05-20 ML-registry entry): it can check the manifest, license, and
that the weights are *resolvable*, but it can't end-to-end validate model
accuracy. Trust leans on editorial review for the curated tier.

Checks per model directory (a dir with ``model.toml``):
  * `model.toml` parses and has the required fields (id, version);
  * `model.kind` and `model.format` are known (tier 2 prefers `onnx`);
  * a `license` SPDX id is declared and a non-empty `LICENSE` file exists;
  * the weights are resolvable — EITHER `model.file` is present in the
    directory (bundled/curated-with-weights) OR `model.url` +
    `model.file_checksum` are declared (registry entries are manifest-only;
    weights live at the url, fetched on demand in E12);
  * `model.file_checksum`, if present, looks like `sha256:<64 hex>`;
  * `output.tier_kind`, if present, is a known sadda tier kind;
  * `compute.gpu`, if present, is `required` | `optional` | `unsupported`.

Usage::

    python validate.py <root> [<root> ...]      # each root holds model dirs
    python validate.py --dist <model_dir>       # one model dir

Exit code is nonzero if any ERROR was found. WARN does not fail.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

KNOWN_KINDS = {"embedding", "transcription", "vad", "segmentation", "alignment", "feature"}
KNOWN_FORMATS = {"onnx", "gguf", "safetensors", "savedmodel"}
# sadda's tier vocabulary (B2/B3); what inference results become.
KNOWN_TIER_KINDS = {
    "interval",
    "point",
    "reference",
    "continuous_numeric",
    "continuous_vector",
    "categorical_sampled",
}
KNOWN_GPU = {"required", "optional", "unsupported"}
CHECKSUM_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: str, msg: str) -> None:
        self.errors.append(f"ERROR {where}: {msg}")

    def warn(self, where: str, msg: str) -> None:
        self.warnings.append(f"WARN  {where}: {msg}")


def _tier_of(model_dir: Path) -> int:
    for part in model_dir.parts:
        if part == "tier1":
            return 1
        if part == "tier2":
            return 2
        if part == "tier3":
            return 3
    return 3


def validate_model(model_dir: Path, rep: Report) -> None:
    where = str(model_dir)
    manifest_path = model_dir / "model.toml"
    if not manifest_path.is_file():
        rep.error(where, "no model.toml")
        return
    try:
        m = tomllib.loads(manifest_path.read_text())
    except tomllib.TOMLDecodeError as e:
        rep.error(where, f"model.toml does not parse: {e}")
        return

    for field in ("id", "version"):
        if not m.get(field):
            rep.error(where, f"missing required top-level field '{field}'")

    tier = _tier_of(model_dir)
    model = m.get("model", {})

    kind = model.get("kind", "")
    if kind not in KNOWN_KINDS:
        rep.error(where, f"unknown model.kind {kind!r} (known: {sorted(KNOWN_KINDS)})")

    fmt = model.get("format", "")
    if fmt not in KNOWN_FORMATS:
        rep.error(where, f"unknown model.format {fmt!r} (known: {sorted(KNOWN_FORMATS)})")
    elif tier == 2 and fmt != "onnx":
        rep.warn(where, f"tier-2 curated format is {fmt!r}; ONNX is the canonical curated format")

    # License: SPDX id + LICENSE file.
    if not m.get("license"):
        rep.error(where, "no top-level `license` SPDX id")
    license_file = model_dir / "LICENSE"
    if not (license_file.is_file() and license_file.read_text().strip()):
        rep.error(where, "missing or empty LICENSE file")

    # Weights resolvable: local file, or url + checksum.
    file = model.get("file")
    url = model.get("url")
    checksum = model.get("file_checksum")
    if file:
        if not (model_dir / file).is_file():
            rep.error(where, f"declared model.file {file!r} is missing")
    elif url:
        if not checksum:
            rep.error(where, "model.url given without model.file_checksum")
    else:
        rep.error(where, "no model.file (local) or model.url (+ file_checksum) declared")

    if checksum is not None and not CHECKSUM_RE.match(checksum):
        rep.error(where, f"model.file_checksum {checksum!r} is not 'sha256:<64 hex>'")

    # Output tier kind, if declared.
    tier_kind = m.get("output", {}).get("tier_kind")
    if tier_kind is not None and tier_kind not in KNOWN_TIER_KINDS:
        rep.warn(where, f"output.tier_kind {tier_kind!r} not a known sadda tier kind")

    # Compute hints.
    gpu = m.get("compute", {}).get("gpu")
    if gpu is not None and gpu not in KNOWN_GPU:
        rep.error(where, f"compute.gpu {gpu!r} must be one of {sorted(KNOWN_GPU)}")


def iter_model_dirs(root: Path):
    """A directory is a model iff it contains a model.toml."""
    if (root / "model.toml").is_file():
        yield root
        return
    for child in sorted(root.rglob("model.toml")):
        yield child.parent


def main(argv: list[str]) -> int:
    args = argv[1:]
    rep = Report()
    if args and args[0] == "--dist":
        dirs = [Path(a) for a in args[1:]]
    else:
        roots = [Path(a) for a in args] or [Path(__file__).resolve().parent]
        dirs = [d for r in roots for d in iter_model_dirs(r)]

    if not dirs:
        print("no models found", file=sys.stderr)
        return 1
    for d in dirs:
        validate_model(d, rep)

    for w in rep.warnings:
        print(w)
    for e in rep.errors:
        print(e, file=sys.stderr)
    n = len(dirs)
    if rep.errors:
        print(f"\n{len(rep.errors)} error(s) across {n} model(s)", file=sys.stderr)
        return 1
    print(f"\nOK — {n} model(s) valid ({len(rep.warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
