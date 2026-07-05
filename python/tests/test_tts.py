"""Python-surface tests for the TTS pipeline (sadda.tts.*).

These avoid any external dependency: the pipeline/cache/assembly logic is
exercised with an in-test fake backend, and the real espeak-ng backend is only
touched by a test that skips when the binary is absent (so CI stays green
without espeak-ng installed).
"""

from __future__ import annotations

import shutil
import types
import wave
from pathlib import Path

import pytest

import sadda
from sadda.tts.backends import SynthesisResult

pytestmark = pytest.mark.filterwarnings("ignore::sadda.ProvisionalAPIWarning")


def _write_silence(path: Path, seconds: float, sample_rate: int = 22050) -> None:
    n = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n)


class _FakeBackend:
    """Writes a fixed-length silent WAV; counts how often it's invoked."""

    name = "fake"

    def __init__(self, seconds: float = 0.1, sample_rate: int = 22050) -> None:
        self.seconds = seconds
        self.sample_rate = sample_rate
        self.calls = 0

    def synthesize(self, text, out_path, *, voice=None, rate=None) -> SynthesisResult:
        self.calls += 1
        _write_silence(Path(out_path), self.seconds, self.sample_rate)
        return SynthesisResult(path=Path(out_path), sample_rate=self.sample_rate, duration_s=self.seconds)


# --- namespace + surface ----------------------------------------------------


def test_tts_namespace_is_a_submodule() -> None:
    assert isinstance(sadda.tts, types.ModuleType)


def test_fake_backend_satisfies_protocol() -> None:
    # Structural protocol: a duck-typed backend is accepted.
    assert isinstance(_FakeBackend(), sadda.tts.TTSBackend)


# --- script model -----------------------------------------------------------


def test_parse_script_splits_on_blank_lines_and_collapses_wraps() -> None:
    script = sadda.tts.parse_script("First line.\nstill first.\n\nSecond.\n\n\n")
    assert [s.text for s in script.segments] == ["First line. still first.", "Second."]


def test_from_texts_and_fallback_chain() -> None:
    script = sadda.tts.NarrationScript.from_texts(["a", "b"], voice="en-us", rate=1.1)
    seg_default, seg_override = script.segments[0], sadda.tts.Segment("c", voice="en-gb")
    # segment with no voice falls back to the script default
    assert script.resolved_voice(seg_default) == "en-us"
    # a segment's own voice wins
    assert script.resolved_voice(seg_override) == "en-gb"
    assert script.resolved_rate(seg_default) == 1.1


# --- backend registry -------------------------------------------------------


def test_registry_lists_espeak_and_kokoro() -> None:
    names = sadda.tts.list_backends()
    assert "espeak-ng" in names and "kokoro" in names


def test_kokoro_backend_is_pending_with_actionable_error() -> None:
    with pytest.raises(NotImplementedError, match="not yet wired"):
        sadda.tts.get_backend("kokoro")


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown TTS backend"):
        sadda.tts.get_backend("nope")


# --- cache key --------------------------------------------------------------


def test_cache_key_is_stable_and_sensitive() -> None:
    base = dict(backend="espeak-ng", voice="en-us", rate=1.0)
    k = sadda.tts.cache_key("hello", **base)
    assert k == sadda.tts.cache_key("hello", **base)  # deterministic
    assert k != sadda.tts.cache_key("hello!", **base)  # text
    assert k != sadda.tts.cache_key("hello", backend="espeak-ng", voice="en-gb", rate=1.0)  # voice
    assert k != sadda.tts.cache_key("hello", backend="espeak-ng", voice="en-us", rate=1.2)  # rate


# --- assembly ---------------------------------------------------------------


def test_concat_wavs_sums_duration_with_pause(tmp_path: Path) -> None:
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_silence(a, 0.10)
    _write_silence(b, 0.20)
    out = sadda.tts.concat_wavs([a, b], tmp_path / "out.wav", pauses_s=[0.05, 0.0])
    assert out.duration_s == pytest.approx(0.35, abs=1e-3)


def test_concat_wavs_rejects_format_mismatch(tmp_path: Path) -> None:
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_silence(a, 0.1, sample_rate=22050)
    _write_silence(b, 0.1, sample_rate=16000)
    with pytest.raises(ValueError, match="format mismatch"):
        sadda.tts.concat_wavs([a, b], tmp_path / "out.wav")


# --- pipeline ---------------------------------------------------------------


def test_synthesize_script_assembles_and_caches(tmp_path: Path) -> None:
    fake = _FakeBackend(seconds=0.1)
    script = sadda.tts.NarrationScript(
        segments=(
            sadda.tts.Segment("one", pause_after_s=0.05),
            sadda.tts.Segment("two"),
        )
    )
    result = sadda.tts.synthesize_script(script, tmp_path / "vo", backend=fake)

    assert len(result.segments) == 2
    assert result.combined is not None and result.combined.exists()
    assert result.total_duration_s == pytest.approx(0.25, abs=1e-3)  # 0.1 + 0.1 + 0.05 pause
    assert fake.calls == 2

    # Re-run: identical segments hit the cache, no new synthesis.
    fake.calls = 0
    sadda.tts.synthesize_script(script, tmp_path / "vo2", backend=fake, cache_dir=tmp_path / "vo" / ".cache")
    assert fake.calls == 0


def test_synthesize_script_resynthesizes_only_changed_segment(tmp_path: Path) -> None:
    fake = _FakeBackend(seconds=0.1)
    cache = tmp_path / "cache"
    s1 = sadda.tts.NarrationScript.from_texts(["alpha", "beta"])
    sadda.tts.synthesize_script(s1, tmp_path / "a", backend=fake, cache_dir=cache)
    assert fake.calls == 2

    fake.calls = 0
    s2 = sadda.tts.NarrationScript.from_texts(["alpha", "gamma"])  # only 2nd changed
    sadda.tts.synthesize_script(s2, tmp_path / "b", backend=fake, cache_dir=cache)
    assert fake.calls == 1


# --- espeak-ng integration (skipped when the binary is absent) --------------


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")
def test_espeak_backend_produces_audio(tmp_path: Path) -> None:
    out = sadda.tts.synthesize("Hello from sadda.", tmp_path / "hello.wav", backend="espeak-ng")
    assert out.path.exists()
    assert out.duration_s > 0.1
    assert out.sample_rate > 0
