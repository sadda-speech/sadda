"""E11 part 3b — model-registry tooling (validate.py + build_index.py) and
the bundled-model install path. Exercised in the main repo's CI so the
model-registry gates run here too (the registry also carries its own
workflow for when it is split into a standalone repo)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import sadda

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "model-registry"
BUNDLED = REPO / "models-bundled"


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
    # The tier-1 bundled set (file-based weights) also validates.
    b = _run(str(REGISTRY / "validate.py"), str(BUNDLED))
    assert b.returncode == 0, b.stdout + b.stderr


def test_validator_rejects_url_without_checksum(tmp_path) -> None:
    src = REGISTRY / "tier3" / "placeholder-asr"
    dst = tmp_path / "tier3" / "broken"
    shutil.copytree(src, dst)
    toml = (dst / "model.toml").read_text()
    # Drop the file_checksum line, leaving a url with no checksum.
    toml = "\n".join(ln for ln in toml.splitlines() if "file_checksum" not in ln)
    (dst / "model.toml").write_text(toml)
    r = _run(str(REGISTRY / "validate.py"), str(tmp_path))
    assert r.returncode != 0
    assert "checksum" in (r.stdout + r.stderr)


def test_validator_flags_missing_license(tmp_path) -> None:
    src = REGISTRY / "tier2" / "placeholder-embeddings"
    dst = tmp_path / "tier2" / "nolicense"
    shutil.copytree(src, dst)
    (dst / "LICENSE").unlink()
    r = _run(str(REGISTRY / "validate.py"), str(tmp_path))
    assert r.returncode != 0
    assert "LICENSE" in (r.stdout + r.stderr)


def test_validator_rejects_unknown_kind(tmp_path) -> None:
    src = REGISTRY / "tier2" / "placeholder-embeddings"
    dst = tmp_path / "tier2" / "weird"
    shutil.copytree(src, dst)
    toml = (dst / "model.toml").read_text().replace('kind = "embedding"', 'kind = "telepathy"')
    (dst / "model.toml").write_text(toml)
    r = _run(str(REGISTRY / "validate.py"), str(tmp_path))
    assert r.returncode != 0
    assert "kind" in (r.stdout + r.stderr)


def test_build_index_matches_engine_schema() -> None:
    r = _run(str(REGISTRY / "build_index.py"), str(REGISTRY))
    assert r.returncode == 0, r.stderr
    index = json.loads(r.stdout)
    assert index["schema_version"] == 1
    ids = {e["id"] for e in index["entries"]}
    assert {"sadda/placeholder-embeddings", "sadda/placeholder-asr"} <= ids
    emb = next(e for e in index["entries"] if e["id"] == "sadda/placeholder-embeddings")
    assert emb["tier"] == 2
    assert emb["kind"] == "embedding"
    assert emb["format"] == "onnx"
    assert emb["license"] == "Apache-2.0"


def test_install_bundled_model_into_store(tmp_path) -> None:
    store = str(tmp_path / "store")
    src = str(BUNDLED / "silero-vad")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        installed = sadda.ml.install_model(src, root=store)
        assert installed.id == "sadda/silero-vad"
        assert installed.kind == "vad"
        # Now resolvable from that store.
        m = sadda.ml.get_model("sadda/silero-vad", "1.0.0", root=store)
    assert m is not None
    assert m.weights_checksum.startswith("sha256:")


def test_registry_functions_are_provisional() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.ml.install_model) == "provisional"
    assert get_stability(sadda.ml.get_model) == "provisional"
