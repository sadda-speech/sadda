"""The bundled Silero VAD ships inside the wheel so ``sadda.ml.vad()`` works
out of the box (no ``SADDA_MODELS_BUNDLED`` needed) after ``pip install
sadda[ml]``."""

from __future__ import annotations

import filecmp
from pathlib import Path

import sadda.ml


def _pkg_bundled() -> Path:
    return Path(sadda.ml.__file__).resolve().parents[1] / "_bundled"


def test_bundled_silero_ships_with_the_package() -> None:
    sv = _pkg_bundled() / "silero-vad"
    assert (sv / "model.toml").is_file(), "model.toml missing from packaged bundle"
    assert (sv / "silero_vad.onnx").is_file(), "ONNX weights missing from packaged bundle"


def test_discovery_points_models_bundled_at_the_package() -> None:
    found = sadda.ml._discover_bundled_models()
    assert found is not None
    assert (Path(found) / "silero-vad" / "model.toml").is_file()


def test_package_copy_matches_repo_canonical() -> None:
    # Drift guard: the in-package copy must stay byte-identical to the repo's
    # canonical models-bundled/ (the copy the engine / GUI build uses). Only
    # meaningful in a source checkout; skipped for a bare pip install.
    repo = (
        Path(__file__).resolve().parents[2]
        / "models-bundled"
        / "silero-vad"
        / "silero_vad.onnx"
    )
    if not repo.is_file():
        return
    pkg = _pkg_bundled() / "silero-vad" / "silero_vad.onnx"
    assert filecmp.cmp(pkg, repo, shallow=False), (
        "python/sadda/_bundled/silero-vad/silero_vad.onnx drifted from "
        "models-bundled/silero-vad/silero_vad.onnx — keep them identical"
    )
