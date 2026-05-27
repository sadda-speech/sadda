#!/usr/bin/env python3
"""Generate the PLACEHOLDER model-registry entries (E11 part 3b).

These are **synthetic, non-authoritative, manifest-only** entries: the
registry holds a `model.toml` + `LICENSE` per model, and the *weights*
live at the declared `url` (fetched on demand in E12 — large weights don't
belong in git). They exist to exercise the format, validator, and index
builder until real curated models (wav2vec2-base, whisper-tiny/base) are
listed. The tier-1 bundled model (Silero VAD) ships its weights under
`../models-bundled/`.

Run from the repo root::

    python model-registry/make_placeholders.py

Deterministic — re-running reproduces the entries.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent

APACHE_NOTICE = (
    "This PLACEHOLDER entry's weights would be licensed under the Apache "
    "License 2.0.\nFull text: https://www.apache.org/licenses/LICENSE-2.0\n"
)
MIT_NOTICE = "This PLACEHOLDER entry's weights would be licensed under the MIT License.\n"


def _write(dist_dir: Path, manifest: str, notice: str) -> None:
    dist_dir.mkdir(parents=True, exist_ok=True)
    (dist_dir / "model.toml").write_text(textwrap.dedent(manifest).lstrip())
    (dist_dir / "LICENSE").write_text(notice)


def embeddings_placeholder() -> None:
    """Tier 2 (curated): synthetic SSL embedding model entry."""
    manifest = """
        id = "sadda/placeholder-embeddings"
        version = "0.1.0"
        title = "PLACEHOLDER speech embedding model (synthetic entry)"
        upstream_source = "https://example.invalid/placeholder-embeddings"
        license = "Apache-2.0"

        [model]
        kind = "embedding"
        format = "onnx"
        url = "https://example.invalid/placeholder-embeddings/model.onnx"
        file_checksum = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

        [input]
        modality = "audio"
        sample_rate_hz = 16000
        channels = 1

        [output]
        tier_kind = "continuous_vector"
        channels = 768
        sample_rate_hz = 50

        [compute]
        cpu_min_ram_mb = 1024
        gpu = "optional"

        [citation]
        authors = ["sadda placeholder generator"]
        year = 2026
    """
    _write(REGISTRY / "tier2" / "placeholder-embeddings", manifest, APACHE_NOTICE)


def transcription_placeholder() -> None:
    """Tier 3 (community): synthetic transcription model entry."""
    manifest = """
        id = "sadda/placeholder-asr"
        version = "0.1.0"
        title = "PLACEHOLDER transcription model (synthetic entry)"
        upstream_source = "https://example.invalid/placeholder-asr"
        license = "MIT"

        [model]
        kind = "transcription"
        format = "onnx"
        url = "https://example.invalid/placeholder-asr/model.onnx"
        file_checksum = "sha256:1111111111111111111111111111111111111111111111111111111111111111"

        [input]
        modality = "audio"
        sample_rate_hz = 16000

        [output]
        tier_kind = "interval"

        [compute]
        cpu_min_ram_mb = 2048
        gpu = "optional"

        [citation]
        authors = ["sadda placeholder generator"]
        year = 2026
    """
    _write(REGISTRY / "tier3" / "placeholder-asr", manifest, MIT_NOTICE)


def main() -> None:
    embeddings_placeholder()
    transcription_placeholder()
    print("wrote placeholder model entries to model-registry/{tier2,tier3}/")


if __name__ == "__main__":
    main()
