"""E11 — ML inference (bundled Silero VAD), Python surface.

The inference tests need a runtime ONNX Runtime (`ORT_DYLIB_PATH`); they
skip cleanly when it isn't available, so CI (which has no ORT) stays
green. Run locally with `ORT_DYLIB_PATH=…/libonnxruntime.so` for the real
assertions.
"""

from __future__ import annotations

import warnings
import wave

import numpy as np
import pytest
import sadda


def _silence_wav(path, seconds: float = 1.0, sr: int = 16_000) -> None:
    """Writes a mono 16-bit PCM WAV of silence."""
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)


def _ort_or_skip(fn):
    """Calls `fn`, skipping the test if ONNX Runtime isn't available."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence the @provisional warning
        try:
            return fn()
        except RuntimeError as e:
            msg = str(e).lower()
            if "ml error" in msg or "onnx" in msg:
                pytest.skip(f"ONNX Runtime not available: {e}")
            raise


def test_ml_functions_are_provisional() -> None:
    # Runs without ONNX Runtime — just checks the surface is wired + tiered.
    from sadda._stability import get_stability

    assert get_stability(sadda.ml.vad) == "provisional"
    assert get_stability(sadda.ml.speech_segments) == "provisional"


def test_vad_on_silence(tmp_path) -> None:
    wav = tmp_path / "silence.wav"
    _silence_wav(wav)
    audio = sadda.load_wav(str(wav))
    times, probs = _ort_or_skip(lambda: sadda.ml.vad(audio))
    assert len(times) == len(probs)
    assert len(times) > 0
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    # Silence is overwhelmingly non-speech.
    assert float(np.mean(probs)) < 0.3


def test_speech_segments_on_silence(tmp_path) -> None:
    wav = tmp_path / "silence.wav"
    _silence_wav(wav)
    audio = sadda.load_wav(str(wav))
    segs = _ort_or_skip(lambda: sadda.ml.speech_segments(audio, threshold=0.5))
    assert isinstance(segs, list)
    # No speech in silence.
    assert segs == []


def test_load_model_resolves_bundled_vad() -> None:
    # No ONNX Runtime needed — just resolution + manifest metadata.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sadda.ml.load_model("sadda/silero-vad")
    assert m.id == "sadda/silero-vad"
    assert m.kind == "vad"
    assert m.weights_checksum.startswith("sha256:")


def test_load_model_hf_refused_without_network_opt_in() -> None:
    """The `download` capability is compiled into the wheel, but a network
    fetch is refused unless SADDA_ALLOW_NETWORK is set — and the error names
    the opt-in. Runs in CI (no network: the gate fires before any request).
    """
    import os

    if os.environ.get("SADDA_ALLOW_NETWORK"):
        pytest.skip("SADDA_ALLOW_NETWORK is set; gate would allow the fetch")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Exception) as exc:
            # Well-formed id (org/name/file) so it passes parsing and reaches
            # the network gate rather than failing on the id shape.
            sadda.ml.load_model("hf://sadda-test/does-not-exist-xyz/model.onnx")
    assert "SADDA_ALLOW_NETWORK" in str(exc.value)


def test_load_model_is_provisional() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.ml.load_model) == "provisional"


def test_model_embeddings_from_fixture(tmp_path) -> None:
    # Load a synthetic embedding fixture (waveform-input) via local:// and
    # run it; skips cleanly without ORT.
    from pathlib import Path

    fixture = (
        Path(__file__).resolve().parents[2]
        / "crates/engine/tests/ml_fixtures/waveform-embed"
    )
    wav = tmp_path / "silence.wav"
    _silence_wav(wav)
    audio = sadda.load_wav(str(wav))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sadda.ml.load_model(f"local://{fixture}")
        emb = _ort_or_skip(lambda: m.embeddings(audio))
    assert emb.ndim == 2
    assert emb.shape[1] == 8  # fixture DIMS
    assert emb.shape[0] > 0


def test_extract_embeddings_writes_tier(tmp_path) -> None:
    from pathlib import Path

    fixture = (
        Path(__file__).resolve().parents[2]
        / "crates/engine/tests/ml_fixtures/waveform-embed"
    )
    proj = sadda.new_project(str(tmp_path / "proj"), "emb")
    wav = tmp_path / "rec.wav"
    _silence_wav(wav)
    bundle_id = proj.add_bundle("b", str(wav))
    tier_id = _ort_or_skip(
        lambda: proj.extract_embeddings(bundle_id, f"local://{fixture}", "ssl_emb")
    )
    df = proj.query(tier_id)
    assert df.height > 0  # one row per embedding frame


def test_ort_dylib_autodiscovery() -> None:
    """The ``sadda.ml`` import hook sets ``ORT_DYLIB_PATH`` from a
    pip/conda ``onnxruntime`` install when the user hasn't already pointed
    it elsewhere. Skips if ``onnxruntime`` isn't installed in this env.
    """
    pytest.importorskip("onnxruntime")
    from pathlib import Path

    from sadda.ml import _discover_ort_dylib

    found = _discover_ort_dylib()
    assert found is not None, "onnxruntime is importable but no library was located"
    p = Path(found)
    assert p.exists(), f"discovered path does not exist: {p}"
    assert "providers_shared" not in p.name, (
        f"discovery picked up the provider shim: {p.name}"
    )


def test_model_vad_matches_free_vad(tmp_path) -> None:
    wav = tmp_path / "silence.wav"
    _silence_wav(wav)
    audio = sadda.load_wav(str(wav))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sadda.ml.load_model("sadda/silero-vad")
        times, probs = _ort_or_skip(lambda: m.vad(audio))
    assert len(times) == len(probs) > 0
    assert float(np.mean(probs)) < 0.3
