"""Python-surface tests for sadda.align.syllabify (A3 — syllabification).

All ungated: syllabification is a pure rule (no model, no espeak), driven here
off hand-built Alignments.
"""

from __future__ import annotations

import pytest

import sadda
from sadda.align import Alignment, TimedPhone, TimedWord, syllabify

pytestmark = pytest.mark.filterwarnings("ignore::sadda.ProvisionalAPIWarning")


def _word(text: str, phones: list[tuple[str, float, float]]) -> TimedWord:
    tp = tuple(TimedPhone(label=l, start_seconds=s, end_seconds=e, score=1.0) for l, s, e in phones)
    start = tp[0].start_seconds if tp else 0.0
    end = tp[-1].end_seconds if tp else 0.0
    return TimedWord(text=text, start_seconds=start, end_seconds=end, phones=tp)


def _alignment(*words: TimedWord) -> Alignment:
    phones = tuple(p for w in words for p in w.phones)
    return Alignment(words=words, phones=phones)


# --- native index-level algorithm ---


def test_native_syllabify_maximal_onset() -> None:
    # /a b l a/ -> a.bla ; /a l b a/ -> al.ba
    assert sadda._native.syllabify(["a", "b", "l", "a"]) == [(0, 1), (1, 4)]
    assert sadda._native.syllabify(["a", "l", "b", "a"]) == [(0, 2), (2, 4)]


# --- Alignment-level syllabify ---


def test_two_syllable_word() -> None:
    # hello /h ə l o ʊ/ -> hə . loʊ (the split diphthong o+ʊ is one nucleus)
    al = _alignment(
        _word("hello", [("h", 0.0, 0.1), ("ə", 0.1, 0.2), ("l", 0.2, 0.3), ("o", 0.3, 0.45), ("ʊ", 0.45, 0.6)])
    )
    syls = syllabify(al)
    assert [s.label for s in syls] == ["hə", "loʊ"]
    # spans run from first phone start to last phone end, contiguous within the word
    assert (syls[0].start_seconds, syls[0].end_seconds) == pytest.approx((0.0, 0.2))
    assert (syls[1].start_seconds, syls[1].end_seconds) == pytest.approx((0.2, 0.6))
    assert [p.label for p in syls[1].phones] == ["l", "o", "ʊ"]


def test_monosyllable() -> None:
    al = _alignment(_word("strength", [(p, i * 0.1, (i + 1) * 0.1) for i, p in enumerate(["s", "t", "r", "ɛ", "ŋ", "θ"])]))
    syls = syllabify(al)
    assert len(syls) == 1
    assert syls[0].label == "strɛŋθ"


def test_pause_words_are_not_syllabified() -> None:
    # leading empty (pause) word, a real word, trailing empty word
    al = _alignment(
        TimedWord(text="", start_seconds=0.0, end_seconds=0.2, phones=()),
        _word("cat", [("k", 0.2, 0.3), ("æ", 0.3, 0.4), ("t", 0.4, 0.5)]),
        TimedWord(text="", start_seconds=0.5, end_seconds=0.8, phones=()),
    )
    syls = syllabify(al)
    # only the real word yields a syllable; empty words contribute nothing
    assert [s.label for s in syls] == ["kæt"]


def test_modeled_silence_phone_in_pause_word_is_skipped() -> None:
    # An MFA-style pause word (empty text) may carry a labelled 'sil' phone —
    # it must still be skipped, not turned into a "sil" syllable.
    pause = TimedWord(
        text="",
        start_seconds=0.0,
        end_seconds=0.3,
        phones=(TimedPhone(label="sil", start_seconds=0.0, end_seconds=0.3, score=1.0),),
    )
    al = _alignment(pause, _word("a", [("a", 0.3, 0.5)]))
    syls = syllabify(al)
    assert [s.label for s in syls] == ["a"]


def test_multiword_syllables_do_not_span_words() -> None:
    al = _alignment(
        _word("to", [("t", 0.0, 0.1), ("u", 0.1, 0.3)]),
        _word("day", [("d", 0.3, 0.4), ("e", 0.4, 0.55), ("ɪ", 0.55, 0.7)]),
    )
    syls = syllabify(al)
    assert [s.label for s in syls] == ["tu", "deɪ"]
    # each syllable's phones come from a single word
    assert all(len(s.phones) >= 1 for s in syls)


def test_empty_alignment_gives_no_syllables() -> None:
    assert syllabify(_alignment()) == ()
