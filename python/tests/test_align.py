"""Python-surface tests for sadda.align (A1 — G2P slice).

The pure helpers (strip_stress, split_phones) run everywhere; the espeak-ng
phonemize test skips when the binary is absent so CI stays green without it.
"""

from __future__ import annotations

import shutil
import types

import pytest

import sadda

pytestmark = pytest.mark.filterwarnings("ignore::sadda.ProvisionalAPIWarning")


def test_align_namespace_is_a_submodule() -> None:
    assert isinstance(sadda.align, types.ModuleType)


def test_strip_stress_removes_primary_and_secondary() -> None:
    assert sadda.align.strip_stress("həlˈoʊ ˌwɜːld") == "həloʊ wɜːld"
    assert sadda.align.strip_stress("no marks") == "no marks"


def test_split_phones_segments_base_plus_modifiers() -> None:
    # length mark ː binds to the preceding vowel; diphthong splits (untied)
    assert sadda.align.split_phones("wɜːld") == ["w", "ɜː", "l", "d"]
    assert sadda.align.split_phones("həloʊ") == ["h", "ə", "l", "o", "ʊ"]
    # a tied affricate stays one phone; whitespace is ignored
    assert sadda.align.split_phones("t͡ʃ iz") == ["t͡ʃ", "i", "z"]


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")
def test_phonemize_produces_stress_free_per_word_phones() -> None:
    utt = sadda.align.phonemize("hello world", voice="en-us")
    assert [w.text for w in utt.words] == ["hello", "world"]
    # every word got phones, and no stress marks survive into them
    for w in utt.words:
        assert w.phones, f"{w.text!r} produced no phones"
        assert "ˈ" not in "".join(w.phones) and "ˌ" not in "".join(w.phones)
    # the utterance-level phone sequence is the concatenation
    assert utt.phones == utt.words[0].phones + utt.words[1].phones


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")
def test_phonemize_strips_edge_punctuation() -> None:
    utt = sadda.align.phonemize("Hello, world!")
    assert [w.text for w in utt.words] == ["Hello", "world"]
