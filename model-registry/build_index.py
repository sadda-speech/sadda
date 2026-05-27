#!/usr/bin/env python3
"""Build the model registry's ``index.json`` — the GitHub-Pages artifact
the sadda engine reads to discover what a hosted model registry offers
(E11 part 3b).

Walks ``tier2/`` and ``tier3/`` under the registry root, reads each
``model.toml``, and emits one entry per model in the shape
``sadda_engine::models::ModelRegistryIndex`` deserializes. Stdlib only
(``tomllib`` + ``json``), so it travels with the registry repo.

Usage::

    python build_index.py [registry_root] [-o index.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

SCHEMA_VERSION = 1


def entry_for(model_dir: Path, tier: int, registry_root: Path) -> dict:
    m = tomllib.loads((model_dir / "model.toml").read_text())
    model = m.get("model", {})
    return {
        "id": m.get("id", ""),
        "version": m.get("version", ""),
        "tier": tier,
        "title": m.get("title", ""),
        "kind": model.get("kind", ""),
        "format": model.get("format", ""),
        "license": m.get("license"),
        "path": str(model_dir.relative_to(registry_root).as_posix()),
        "yanked": bool(m.get("yanked", False)),
    }


def build(registry_root: Path) -> dict:
    entries = []
    for tier in (2, 3):
        tier_dir = registry_root / f"tier{tier}"
        if not tier_dir.is_dir():
            continue
        for manifest in sorted(tier_dir.rglob("model.toml")):
            entries.append(entry_for(manifest.parent, tier, registry_root))
    entries.sort(key=lambda e: (e["id"], e["version"]))
    return {"schema_version": SCHEMA_VERSION, "entries": entries}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "root",
        nargs="?",
        default=str(Path(__file__).resolve().parent),
        help="registry root (default: this script's directory)",
    )
    ap.add_argument("-o", "--output", default="-", help="output path, or - for stdout")
    args = ap.parse_args(argv[1:])

    index = build(Path(args.root))
    text = json.dumps(index, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        Path(args.output).write_text(text)
        print(f"wrote {args.output} ({len(index['entries'])} entries)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
