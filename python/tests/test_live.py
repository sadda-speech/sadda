"""Python-surface tests for the E1 live-recording API.

These tests use ``session.push_samples_for_tests(...)`` to bypass cpal —
CI does not have audio hardware. The cpal path itself is exercised
indirectly by the same engine pipeline that ``push_samples_for_tests``
feeds.
"""

from __future__ import annotations

import math
import tempfile
import time
import wave
import warnings
from pathlib import Path

import pytest

import sadda
import sadda.live


def _silent_wav(path: Path, sample_rate: int = 16_000, duration_s: float = 1.0) -> None:
    n = int(sample_rate * duration_s)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n)


def _sine(sample_rate: int, freq: float, duration_s: float, amplitude: float = 0.3) -> list[float]:
    n = int(sample_rate * duration_s)
    return [
        amplitude * math.sin(2 * math.pi * freq * i / sample_rate) for i in range(n)
    ]


def _new_project(td: Path) -> sadda.Project:
    return sadda.new_project(td / "proj", "demo")


def _start_session(proj: sadda.Project, **kw) -> sadda.live.LiveSession:
    # Suppress the ProvisionalAPIWarning that fires on first use per test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=sadda.ProvisionalAPIWarning)
        defaults = dict(sample_rate=16_000, channels=1, name="t")
        defaults.update(kw)
        return sadda.live.start_session(proj, **defaults)


def test_session_creates_in_progress_directory() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = _new_project(Path(td))
        session = _start_session(proj)
        in_progress = Path(session.in_progress_dir)
        assert in_progress.exists()
        assert in_progress.is_dir()
        assert (in_progress / "audio.wav").exists()
        session.stop()
        session.discard()


def test_commit_creates_bundle_and_processing_run() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = _new_project(Path(td))
        session = _start_session(proj, name="practice_take")
        samples = _sine(16_000, 440.0, 0.5)
        session.push_samples_for_tests(samples)
        time.sleep(0.1)
        session.stop()
        bundle_id = session.commit(proj)
        bundles = [b for b in proj.bundles()]
        assert len(bundles) == 1
        assert bundles[0].id == bundle_id
        assert bundles[0].name == "practice_take"
        wav = Path(td) / "proj" / "signals" / "original" / "practice_take.wav"
        assert wav.exists()


def test_pitch_callback_observes_440hz() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = _new_project(Path(td))
        session = _start_session(proj)
        observed_pitches: list[float] = []

        @session.on_pitch
        def collect(f0_hz: float, voiced: bool, t: float) -> None:
            observed_pitches.append(f0_hz)

        samples = _sine(16_000, 440.0, 1.0)
        session.push_samples_for_tests(samples)
        # Give the dispatch thread enough cycles to flush.
        time.sleep(0.3)
        session.stop()
        session.discard()
        assert len(observed_pitches) >= 10, observed_pitches
        observed_pitches.sort()
        median = observed_pitches[len(observed_pitches) // 2]
        assert abs(median - 440.0) < 5.0


def test_meter_callback_fires_on_push() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = _new_project(Path(td))
        session = _start_session(proj)
        peaks: list[float] = []

        @session.on_meter
        def collect(peak: float, rms: float, rms_db: float, t: float) -> None:
            peaks.append(peak)

        session.push_samples_for_tests(_sine(16_000, 440.0, 0.2, amplitude=0.4))
        time.sleep(0.2)
        session.stop()
        session.discard()
        assert peaks, "expected at least one meter frame"
        assert max(peaks) > 0.3


def test_intensity_callback_fires_with_db_range() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = _new_project(Path(td))
        session = _start_session(proj)
        dbs: list[float] = []

        @session.on_intensity
        def collect(db_fs: float, t: float) -> None:
            dbs.append(db_fs)

        session.push_samples_for_tests(_sine(16_000, 440.0, 0.3))
        time.sleep(0.2)
        session.stop()
        session.discard()
        assert dbs, "expected at least one intensity frame"
        # 0.3-amplitude sine: RMS ≈ 0.212 → -13 dB-FS. Allow generous range.
        for d in dbs:
            assert -40.0 < d < 0.0, d


def test_formants_callback_fires() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = _new_project(Path(td))
        session = _start_session(proj)
        frames: list[tuple] = []

        @session.on_formants
        def collect(freqs: list[float], bws: list[float], t: float) -> None:
            frames.append((freqs, bws, t))

        # A periodic-but-rich source produces multiple LPC roots; pure sine
        # would mostly give one. Use a sum of harmonics.
        sr = 16_000
        n = sr // 2
        samples = [
            0.3 * (math.sin(2 * math.pi * 200 * i / sr)
                   + 0.5 * math.sin(2 * math.pi * 800 * i / sr))
            for i in range(n)
        ]
        session.push_samples_for_tests(samples)
        time.sleep(0.2)
        session.stop()
        session.discard()
        assert frames, "expected at least one formants frame"
        for freqs, bws, _t in frames:
            assert len(freqs) == len(bws)


def test_stop_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = _new_project(Path(td))
        session = _start_session(proj)
        session.push_samples_for_tests(_sine(16_000, 440.0, 0.1))
        time.sleep(0.1)
        session.stop()
        session.stop()  # second call is a no-op
        session.discard()


def test_commit_without_stop_errors() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = _new_project(Path(td))
        session = _start_session(proj)
        with pytest.raises(RuntimeError):
            session.commit(proj)
        session.stop()
        session.discard()


def test_discard_removes_in_progress_dir() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj = _new_project(Path(td))
        session = _start_session(proj)
        session.push_samples_for_tests(_sine(16_000, 440.0, 0.05))
        time.sleep(0.1)
        session.stop()
        in_progress = Path(session.in_progress_dir)
        session.discard()
        assert not in_progress.exists()


def test_default_input_device_returns_string_or_none() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=sadda.ProvisionalAPIWarning)
        d = sadda.live.default_input_device()
    # In a containerised CI there may be no audio device; either result
    # is valid.
    assert d is None or isinstance(d, str)


def test_list_input_devices_returns_list() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=sadda.ProvisionalAPIWarning)
        devices = sadda.live.list_input_devices()
    assert isinstance(devices, list)
    for d in devices:
        assert isinstance(d, str)
