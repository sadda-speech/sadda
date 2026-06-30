"""MFCC parameter/preset registry (roadmap item 3/4).

Covers the Python surface: built-in + user presets, ``MfccParams`` build /
``.replace(...)`` override, ``mfcc(audio, params=...)`` dispatch, and the
on-disk store (save / reload / delete / built-in immutability). The store is
pointed at a temp ``root`` so tests never touch the real user cache.
"""

from __future__ import annotations

import tempfile
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
    _write_sine_wav(wav, freq=220.0, sample_rate=16_000, duration_s=0.5)
    return sadda.load_wav(str(wav))


# ---- presets ---------------------------------------------------------------

def test_builtin_presets_present_and_tagged() -> None:
    by_id = {p.id: p for p in dsp.builtin_mfcc_presets()}
    assert {"librosa-default", "kaldi-default", "praat-default"} <= set(by_id)
    # librosa/kaldi are faithful through the pipeline; praat is not (its
    # pipeline path is f32-approximate — use mfcc(method="praat")).
    assert by_id["librosa-default"].faithful is True
    assert by_id["kaldi-default"].faithful is True
    assert by_id["praat-default"].faithful is False
    assert by_id["librosa-default"].based_on == "librosa"


def test_empty_store_lists_only_builtins(tmp_path: Path) -> None:
    root = str(tmp_path)
    assert dsp.mfcc_user_presets(root=root) == []
    ids = [p.id for p in dsp.mfcc_presets(root=root)]
    assert ids[:3] == ["librosa-default", "kaldi-default", "praat-default"]


# ---- params + override -----------------------------------------------------

def test_params_preset_path_matches_method_path(tone) -> None:
    # mfcc(audio, params=librosa-preset) must reproduce mfcc(method="librosa")
    # bit-for-bit (the preset *is* the validated librosa point).
    lib = dsp.mfcc_preset("librosa-default")
    by_params = dsp.mfcc(tone, params=lib.params)
    by_method = dsp.mfcc(tone, method="librosa", n_mels=40, n_mfcc=13, f_min=0.0, f_max=8000.0)
    assert by_params.shape == by_method.shape
    assert np.array_equal(by_params, by_method)


def test_replace_overrides_fields(tone) -> None:
    lib = dsp.mfcc_preset("librosa-default")
    edited = lib.params.replace(n_mfcc=20, window="hamming", pre_emphasis=0.95)
    assert edited.n_mfcc == 20
    assert edited.window == "hamming"
    assert edited.pre_emphasis == pytest.approx(0.95)
    # Original is unchanged (replace returns a copy).
    assert lib.params.n_mfcc == 13
    out = dsp.mfcc(tone, params=edited)
    assert out.shape[1] == 20


def test_replace_rejects_bad_enum_string() -> None:
    params = dsp.MfccParams.librosa()
    with pytest.raises(ValueError):
        params.replace(window="not-a-window")


def test_n_mels_override_rejected_on_mel_spacing() -> None:
    # The Praat preset uses the mel-spacing filterbank, so n_mels doesn't apply.
    praat = dsp.mfcc_preset("praat-default").params
    assert praat.n_mels is None
    with pytest.raises(ValueError):
        praat.replace(n_mels=30)


def test_params_constructors() -> None:
    assert dsp.MfccParams.librosa().window == "periodic_hann"
    assert dsp.MfccParams.kaldi().window == "povey"
    assert dsp.MfccParams.praat().window == "praat_gaussian"
    assert dsp.MfccParams.kaldi().mel_scale == "htk"


def test_for_method_parallels_named_constructors() -> None:
    # for_method (the generic form shared with PitchParams/FormantsParams)
    # matches the named per-reference constructors at default settings.
    assert dsp.MfccParams.for_method("librosa").to_toml() == dsp.MfccParams.librosa().to_toml()
    assert dsp.MfccParams.for_method("kaldi").to_toml() == dsp.MfccParams.kaldi().to_toml()
    assert dsp.MfccParams.for_method("praat").to_toml() == dsp.MfccParams.praat().to_toml()
    with pytest.raises(ValueError):
        dsp.MfccParams.for_method("not-a-reference")


# ---- on-disk store ---------------------------------------------------------

def test_save_reload_delete_user_preset(tmp_path: Path) -> None:
    root = str(tmp_path)
    base = dsp.mfcc_preset("librosa-default", root=root)
    edited = base.params.replace(n_mfcc=20)
    preset = dsp.MfccPreset(
        "my-asr", edited, title="My ASR front-end", based_on="librosa", faithful=False
    )
    path = dsp.save_mfcc_preset(preset, root=root)
    assert Path(path).is_file()

    assert [p.id for p in dsp.mfcc_user_presets(root=root)] == ["my-asr"]
    got = dsp.mfcc_preset("my-asr", root=root)
    assert got is not None
    assert got.params.n_mfcc == 20
    assert got.based_on == "librosa"
    assert got.faithful is False

    assert dsp.delete_mfcc_preset("my-asr", root=root) is True
    assert dsp.delete_mfcc_preset("my-asr", root=root) is False
    assert dsp.mfcc_preset("my-asr", root=root) is None


def test_save_rejects_builtin_id(tmp_path: Path) -> None:
    root = str(tmp_path)
    params = dsp.MfccParams.librosa()
    with pytest.raises(ValueError):
        dsp.save_mfcc_preset(dsp.MfccPreset("librosa-default", params), root=root)


def test_save_rejects_bad_id(tmp_path: Path) -> None:
    root = str(tmp_path)
    params = dsp.MfccParams.librosa()
    with pytest.raises(ValueError):
        dsp.save_mfcc_preset(dsp.MfccPreset("../escape", params), root=root)


def test_preset_round_trips_through_toml() -> None:
    lib = dsp.mfcc_preset("librosa-default")
    text = lib.to_toml()
    assert 'id = "librosa-default"' in text
    # The params table is embedded.
    assert "[params" in text
