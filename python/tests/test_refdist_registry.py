"""C8 — registry tooling (validate.py + build_index.py) and the bundled
starter-set install path. Exercised in the main repo's CI so the registry
gates run here too (the registry also carries its own workflow for when it
is split into a standalone repo)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import polars as pl
import sadda

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "refdist-registry"
BUNDLED = REPO / "refdist-bundled"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )


def test_placeholder_registry_validates() -> None:
    r = _run(str(REGISTRY / "validate.py"), str(REGISTRY))
    assert r.returncode == 0, r.stdout + r.stderr
    b = _run(str(REGISTRY / "validate.py"), str(BUNDLED))
    assert b.returncode == 0, b.stdout + b.stderr


def test_validator_rejects_broken_distribution(tmp_path) -> None:
    # Copy a good distribution, then break its declared columns.
    src = REGISTRY / "tier3" / "placeholder-vot-norms"
    dst = tmp_path / "tier3" / "broken"
    shutil.copytree(src, dst)
    pl.DataFrame({"wrong": [1, 2]}).write_parquet(dst / "data.parquet")
    r = _run(str(REGISTRY / "validate.py"), str(tmp_path))
    assert r.returncode != 0
    assert "columns" in (r.stdout + r.stderr)


def test_validator_flags_missing_license(tmp_path) -> None:
    src = REGISTRY / "tier2" / "placeholder-f0-norms"
    dst = tmp_path / "tier2" / "nolicense"
    shutil.copytree(src, dst)
    (dst / "LICENSE").unlink()
    r = _run(str(REGISTRY / "validate.py"), str(tmp_path))
    assert r.returncode != 0
    assert "LICENSE" in (r.stdout + r.stderr)


def test_build_index_matches_engine_schema() -> None:
    r = _run(str(REGISTRY / "build_index.py"), str(REGISTRY))
    assert r.returncode == 0, r.stderr
    index = json.loads(r.stdout)
    assert index["schema_version"] == 1
    ids = {e["id"] for e in index["entries"]}
    assert {"placeholder-f0-norms", "placeholder-vot-norms"} <= ids
    # Tier and kind are carried through for the engine's RegistryIndex.
    f0 = next(e for e in index["entries"] if e["id"] == "placeholder-f0-norms")
    assert f0["tier"] == 2
    assert f0["kind"] == "summary_normative_range"
    assert f0["license"] == "CC-BY-4.0"


def test_scaffold_produces_a_validatable_distribution(tmp_path) -> None:
    # C9: scaffold a distribution from an analysis result, then prove it
    # passes the C8 validator and resolves from a store.
    df = pl.DataFrame(
        {
            "speaker_id": list(range(1, 9)),
            "phone": ["iy"] * 8,
            "F1": [300.0 + i for i in range(8)],
            "F2": [2300.0 - i for i in range(8)],
        }
    )
    dest = tmp_path / "tier3" / "my-vowels"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = sadda.refdist.scaffold(
            str(dest),
            df,
            id="my-amE-vowels",
            version="0.1.0",
            kind="observed_distribution",
            parameters=["F1", "F2"],
            units="Hz",
            language="eng",
            sex=["m", "f"],
            license="CC0-1.0",
            shareability="raw_samples",
            min_n_per_subgroup=5,
            authors=["Me"],
            year=2026,
            provenance="Real measurements from my study.",
        )
    assert rd.id == "my-amE-vowels"
    assert (dest / "refdist.toml").is_file()
    assert (dest / "data.parquet").is_file()

    # The LICENSE stub must be replaced with real text before it validates
    # under a strict reading, but presence + SPDX already pass the gate.
    r = _run(str(REGISTRY / "validate.py"), str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr

    # And it resolves once installed into a store.
    store = str(tmp_path / "store")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sadda.refdist.install(str(dest), root=store)
        got = sadda.refdist.get("my-amE-vowels", "0.1.0", root=store)
    assert got is not None
    assert got.units == "Hz"
    assert got.data().height == 8


def test_install_bundled_into_store(tmp_path) -> None:
    store = str(tmp_path / "store")
    src = str(BUNDLED / "placeholder-amE-vowels")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        installed = sadda.refdist.install(src, root=store)
        assert installed.id == "placeholder-amE-vowels"
        # Now resolvable from the store, with its data readable.
        rd = sadda.refdist.get("placeholder-amE-vowels", "0.1.0", root=store)
        assert rd is not None
        df = rd.data()
    assert df.columns == ["speaker_id", "sex", "phone", "F1", "F2"]
    assert df["speaker_id"].n_unique() == 16
