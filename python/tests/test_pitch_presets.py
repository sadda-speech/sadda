"""Pitch parameter/preset registry (roadmap item 6).

Mirrors test_mfcc_presets: built-in + user presets, PitchParams build /
.replace(...) override, voiced_pitch(audio, params=...) dispatch, and the
on-disk store. The store is pointed at a temp root so tests never touch the
real user cache.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

import sadda
from sadda import dsp


def _write_sine_wav(path: Path, freq: float, sample_rate: int, duration_s: float) -> None:
    n = int(sample_rate * duration_s)
    samples = np.sin(2 * np.pi * freq * np.arange(n) / sample_rate)
    pcm = (samples * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


@pytest.fixture
def tone(tmp_path: Path):
    wav = tmp_path / "tone.wav"
    _write_sine_wav(wav, freq=150.0, sample_rate=16_000, duration_s=0.6)
    return sadda.load_wav(str(wav))


def test_builtin_presets_present_and_tagged() -> None:
    by_id = {p.id: p for p in dsp.builtin_pitch_presets()}
    assert {"praat-ac", "yin", "pyin", "swipe"} <= set(by_id)
    assert by_id["praat-ac"].params.method == "boersma"
    assert by_id["praat-ac"].based_on == "praat"
    assert all(p.faithful for p in by_id.values())


def test_empty_store_lists_only_builtins(tmp_path: Path) -> None:
    root = str(tmp_path)
    assert dsp.pitch_user_presets(root=root) == []
    ids = [p.id for p in dsp.pitch_presets(root=root)]
    assert ids[:4] == ["praat-ac", "yin", "pyin", "swipe"]


def test_params_path_matches_method_path(tone) -> None:
    praat = dsp.pitch_preset("praat-ac")
    by_params = dsp.voiced_pitch(tone, params=praat.params)
    by_method = dsp.voiced_pitch(tone, method="boersma", min_freq_hz=75.0, max_freq_hz=500.0)
    for a, b in zip(by_params, by_method):
        assert np.array_equal(a, b)


def test_default_preset_is_octave_robust(tone) -> None:
    # praat-ac (Boersma) must report the true 150 Hz, not a subharmonic.
    _t, freqs, voicing = dsp.voiced_pitch(tone, params=dsp.pitch_preset("praat-ac").params)
    voiced = freqs[voicing >= 0.45]
    assert len(voiced) > 5
    assert abs(float(np.median(voiced)) - 150.0) < 3.0


def test_named_constructors_parallel_mfcc_shape() -> None:
    # Named per-method constructors (like MfccParams.librosa/.kaldi/.praat),
    # taking the common analysis args; equivalent to for_method at defaults.
    assert dsp.PitchParams.boersma().method == "boersma"
    assert dsp.PitchParams.yin().method == "yin"
    assert dsp.PitchParams.pyin().method == "pyin"
    assert dsp.PitchParams.swipe().method == "swipe"
    assert dsp.PitchParams.boersma(max_freq_hz=400.0).max_freq_hz == pytest.approx(400.0)
    # Named constructor == for_method at the same (default) settings.
    a = dsp.PitchParams.boersma()
    b = dsp.PitchParams.for_method("boersma")
    assert a.to_toml() == b.to_toml()


def test_for_method_and_replace() -> None:
    p = dsp.PitchParams.for_method("yin")
    assert p.method == "yin"
    edited = p.replace(method="pyin", max_freq_hz=400.0, voicing_threshold=0.5)
    assert edited.method == "pyin"
    assert edited.max_freq_hz == pytest.approx(400.0)
    assert edited.voicing_threshold == pytest.approx(0.5)
    # original unchanged
    assert p.method == "yin"


def test_replace_rejects_bad_method() -> None:
    with pytest.raises(ValueError):
        dsp.PitchParams.for_method("boersma").replace(method="not-a-method")


def test_save_reload_delete(tmp_path: Path) -> None:
    root = str(tmp_path)
    base = dsp.pitch_preset("praat-ac", root=root)
    edited = base.params.replace(method="yin", max_freq_hz=400.0)
    preset = dsp.PitchPreset("my-pitch", edited, based_on="praat", faithful=False)
    path = dsp.save_pitch_preset(preset, root=root)
    assert Path(path).is_file()
    assert [p.id for p in dsp.pitch_user_presets(root=root)] == ["my-pitch"]
    got = dsp.pitch_preset("my-pitch", root=root)
    assert got.params.method == "yin"
    assert got.faithful is False
    assert dsp.delete_pitch_preset("my-pitch", root=root) is True
    assert dsp.delete_pitch_preset("my-pitch", root=root) is False


def test_save_rejects_builtin_id(tmp_path: Path) -> None:
    root = str(tmp_path)
    with pytest.raises(ValueError):
        dsp.save_pitch_preset(
            dsp.PitchPreset("praat-ac", dsp.PitchParams.for_method("boersma")), root=root
        )


def test_preset_round_trips_through_toml() -> None:
    p = dsp.pitch_preset("pyin")
    text = p.to_toml()
    assert 'id = "pyin"' in text
    assert 'method = "pyin"' in text  # not the snake_case default p_yin
