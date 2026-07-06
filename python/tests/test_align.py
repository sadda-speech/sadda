"""Python-surface tests for sadda.align (A1 — G2P slice).

The pure helpers (strip_stress, split_phones) run everywhere; the espeak-ng
phonemize test skips when the binary is absent so CI stays green without it.
"""

from __future__ import annotations

import math
import os
import shutil
import types

import numpy as np
import pytest

import sadda
from sadda.align import Emissions

pytestmark = pytest.mark.filterwarnings("ignore::sadda.ProvisionalAPIWarning")


class _BlockModel:
    """Mock acoustic model: emits posteriors that favour the given phone
    sequence in equal consecutive blocks — so a correct aligner recovers the
    blocks. Vocab is built from the phones it's told to expect."""

    def __init__(self, phones: list[str], frames_per_phone: int = 2, frame_rate: float = 100.0):
        self._phones = phones
        self._fpp = frames_per_phone
        self._frame_rate = frame_rate
        # 0 = blank; phones get ids 1..N (deduped, first-seen order)
        uniq = list(dict.fromkeys(phones))
        self._vocab = {p: i + 1 for i, p in enumerate(uniq)}

    def emissions(self, audio, sample_rate) -> Emissions:
        n_classes = len(self._vocab) + 1
        hi, lo = math.log(0.9), math.log(0.1 / (n_classes - 1))
        rows = []
        for p in self._phones:
            hot = self._vocab[p]
            for _ in range(self._fpp):
                rows.append([hi if c == hot else lo for c in range(n_classes)])
        return Emissions(
            log_probs=np.array(rows, dtype=np.float32),
            vocab=self._vocab,
            frame_rate=self._frame_rate,
            blank_id=0,
        )


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


# --- tokenizer (pure, no espeak) ---


def test_tokenize_greedy_longest_match() -> None:
    vocab = {"d": 1, "ʒ": 2, "dʒ": 3, "a": 4}
    # 'dʒ' (a vocab token) beats 'd' + 'ʒ'
    assert sadda.align.tokenize("dʒa", vocab) == [3, 4]
    # whitespace skipped
    assert sadda.align.tokenize("d a", vocab) == [1, 4]


def test_tokenize_rejects_unknown_phone() -> None:
    with pytest.raises(ValueError, match="not in acoustic-model vocab"):
        sadda.align.tokenize("dx", {"d": 1})


# --- end-to-end orchestration with a mock acoustic model ---


def test_mock_model_satisfies_protocol() -> None:
    assert isinstance(_BlockModel(["a"]), sadda.align.AcousticModel)


@pytest.mark.skipif(shutil.which("espeak-ng") is None, reason="espeak-ng not installed")
def test_align_end_to_end_with_mock() -> None:
    phones = list(sadda.align.phonemize("hi").words[0].phones)
    model = _BlockModel(phones, frames_per_phone=3, frame_rate=100.0)
    audio = np.zeros(16000, dtype=np.float32)  # mock ignores the audio

    al = sadda.align.align(audio, 16000, "hi", model=model)

    # one word "hi", with exactly the phonemized phones, correctly labelled
    assert [w.text for w in al.words] == ["hi"]
    assert [p.label for p in al.phones] == phones
    # contiguous, starts at 0, covers the full emission duration
    assert al.phones[0].start_seconds == 0.0
    assert al.phones[-1].end_seconds == pytest.approx(len(phones) * 3 / 100.0)
    for a, b in zip(al.phones, al.phones[1:]):
        assert a.end_seconds == b.start_seconds
    # the word span wraps its phones
    assert al.words[0].start_seconds == al.phones[0].start_seconds
    assert al.words[0].end_seconds == al.phones[-1].end_seconds


# --- neural acoustic model (onnxruntime + model-gated; skips in CI) ---


def _ort_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


_ALIGN_MODEL = os.environ.get("SADDA_TEST_ALIGN_MODEL")
_ALIGN_VOCAB = os.environ.get("SADDA_TEST_ALIGN_VOCAB")


@pytest.mark.skipif(
    not (_ort_available() and _ALIGN_MODEL and _ALIGN_VOCAB),
    reason="onnxruntime + SADDA_TEST_ALIGN_MODEL/VOCAB required",
)
def test_wav2vec2_espeak_model_produces_emissions() -> None:
    model = sadda.align.Wav2Vec2EspeakModel(_ALIGN_MODEL, _ALIGN_VOCAB)
    assert len(model.vocab) == 392
    assert model.blank_id == model.vocab["<pad>"]

    em = model.emissions(np.zeros(16000, dtype=np.float32), 16000)
    assert em.log_probs.ndim == 2 and em.log_probs.shape[1] == 392
    assert em.frame_rate == 50.0 and em.blank_id == model.blank_id
    # log-probs: each frame's exp-sum ~ 1
    assert np.allclose(np.exp(em.log_probs).sum(axis=1), 1.0, atol=1e-3)

    with pytest.raises(ValueError, match="16 kHz"):
        model.emissions(np.zeros(8000, dtype=np.float32), 8000)
