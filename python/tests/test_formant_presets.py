"""Formant parameter/preset registry (roadmap item 6).

Mirrors test_pitch_presets: built-in + user presets, FormantsParams build /
.replace(...) override, formants(audio, params=...) dispatch, and the on-disk
store. The store is pointed at a temp root so tests never touch the real cache.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

import sadda
from sadda import dsp


def _write_vowel_wav(path: Path, sample_rate: int, duration_s: float) -> None:
    # Impulse-train source through two formant resonators (F1≈700, F2≈1200).
    n = int(sample_rate * duration_s)
    source = np.zeros(n)
    period = max(1, int(sample_rate / 120.0))
    source[::period] = 1.0
    sig = source.copy()
    for f, bw in [(700.0, 80.0), (1220.0, 90.0)]:
        r = np.exp(-np.pi * bw / sample_rate)
        theta = 2 * np.pi * f / sample_rate
        a1, a2 = -2 * r * np.cos(theta), r * r
        y1 = y2 = 0.0
        out = np.empty_like(sig)
        for i, x in enumerate(sig):
            y = x - a1 * y1 - a2 * y2
            out[i] = y
            y2, y1 = y1, y
        sig = out
    sig = sig / np.max(np.abs(sig))
    pcm = (sig * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


@pytest.fixture
def vowel(tmp_path: Path):
    wav = tmp_path / "vowel.wav"
    _write_vowel_wav(wav, sample_rate=16_000, duration_s=0.4)
    return sadda.load_wav(str(wav))


def test_builtin_presets_present_and_tagged() -> None:
    by_id = {p.id: p for p in dsp.builtin_formant_presets()}
    assert {"praat-burg", "autocorrelation"} <= set(by_id)
    assert by_id["praat-burg"].params.method == "burg"
    assert by_id["praat-burg"].based_on == "praat"
    assert all(p.faithful for p in by_id.values())


def test_empty_store_lists_only_builtins(tmp_path: Path) -> None:
    root = str(tmp_path)
    assert dsp.formant_user_presets(root=root) == []
    ids = [p.id for p in dsp.formant_presets(root=root)]
    assert ids[:2] == ["praat-burg", "autocorrelation"]


def test_params_path_matches_method_path(vowel) -> None:
    burg = dsp.formant_preset("praat-burg")
    by_params = dsp.formants(vowel, params=burg.params)
    by_method = dsp.formants(vowel, method="burg", n_formants=5)
    assert len(by_params) == len(by_method)
    # Same number of detected formants per frame.
    for a, b in zip(by_params, by_method):
        assert len(a.frequencies) == len(b.frequencies)


def test_for_method_and_replace() -> None:
    p = dsp.FormantsParams.for_method("burg")
    assert p.method == "burg"
    edited = p.replace(method="autocorrelation", n_formants=4, pre_emphasis=0.9)
    assert edited.method == "autocorrelation"
    assert edited.n_formants == 4
    assert edited.pre_emphasis == pytest.approx(0.9)
    assert p.n_formants == 5  # original unchanged


def test_replace_rejects_bad_method() -> None:
    with pytest.raises(ValueError):
        dsp.FormantsParams.for_method("burg").replace(method="not-a-method")


def test_save_reload_delete(tmp_path: Path) -> None:
    root = str(tmp_path)
    base = dsp.formant_preset("praat-burg", root=root)
    edited = base.params.replace(method="autocorrelation", n_formants=4)
    preset = dsp.FormantPreset("my-formants", edited, based_on="praat", faithful=False)
    path = dsp.save_formant_preset(preset, root=root)
    assert Path(path).is_file()
    assert [p.id for p in dsp.formant_user_presets(root=root)] == ["my-formants"]
    got = dsp.formant_preset("my-formants", root=root)
    assert got.params.method == "autocorrelation"
    assert got.params.n_formants == 4
    assert dsp.delete_formant_preset("my-formants", root=root) is True
    assert dsp.delete_formant_preset("my-formants", root=root) is False


def test_save_rejects_builtin_id(tmp_path: Path) -> None:
    root = str(tmp_path)
    with pytest.raises(ValueError):
        dsp.save_formant_preset(
            dsp.FormantPreset("praat-burg", dsp.FormantsParams.for_method("burg")), root=root
        )


def test_preset_round_trips_through_toml() -> None:
    p = dsp.formant_preset("praat-burg")
    text = p.to_toml()
    assert 'id = "praat-burg"' in text
    assert 'lpc_method = "burg"' in text
