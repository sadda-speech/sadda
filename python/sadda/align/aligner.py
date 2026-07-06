"""sadda.align.aligner — the forced-alignment orchestrator.

Ties the pieces together (see the 2026-07-05 DEVLOG design entry): espeak-ng G2P
(the target phones) + an :class:`~sadda.align.model.AcousticModel` (per-frame CTC
posteriors) + the engine's forced-align DP (`sadda._native.forced_align`) →
time-aligned **Word** and **Phone** results.

The phone→class-id step is a greedy longest-match of each word's IPA against the
model's vocabulary (so multi-character vocab tokens like ``dʒ`` are matched
whole), which is why G2P stays model-agnostic and the tokenization lives here.
Syllable derivation and writing these onto a project's tiers are later slices;
this returns a plain :class:`Alignment` result.

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from sadda import _native
from sadda._stability import provisional

from .g2p import phonemize
from .model import AcousticModel

__all__ = ["TimedPhone", "TimedWord", "Alignment", "tokenize", "align"]


# [docs:sadda.align.TimedPhone]
@dataclass(frozen=True)
class TimedPhone:
    """One aligned phone with its time span and confidence."""

    label: str
    start_seconds: float
    end_seconds: float
    score: float


# [docs:sadda.align.TimedWord]
@dataclass(frozen=True)
class TimedWord:
    """One aligned word: its span and the phones inside it."""

    text: str
    start_seconds: float
    end_seconds: float
    phones: tuple[TimedPhone, ...]


# [docs:sadda.align.Alignment]
@dataclass(frozen=True)
class Alignment:
    """A forced alignment: aligned words and the flat phone sequence."""

    words: tuple[TimedWord, ...]
    phones: tuple[TimedPhone, ...]


# [docs:sadda.align.tokenize]
def tokenize(ipa: str, vocab: Mapping[str, int]) -> list[int]:
    """Greedy longest-match an IPA string into ``vocab`` class ids.

    At each position, the longest vocab key that matches wins (so ``dʒ`` beats
    ``d`` when both are in the vocab). Whitespace is skipped. Raises
    :class:`ValueError` naming the offending substring if no key matches — which
    is how a phone the acoustic model doesn't cover surfaces.
    """
    keys = sorted((k for k in vocab if k), key=len, reverse=True)
    out: list[int] = []
    i, n = 0, len(ipa)
    while i < n:
        if ipa[i].isspace():
            i += 1
            continue
        for k in keys:
            if ipa.startswith(k, i):
                out.append(vocab[k])
                i += len(k)
                break
        else:
            raise ValueError(f"phone not in acoustic-model vocab at {ipa[i:]!r}")
    return out


# [docs:sadda.align.align]
@provisional
def align(
    audio: np.ndarray,
    sample_rate: int,
    transcript: str,
    *,
    model: AcousticModel,
    voice: str = "en-us",
) -> Alignment:
    """Force-align ``transcript`` to ``audio`` with ``model``.

    Phonemizes the transcript (espeak-ng, ``voice``), gets per-frame posteriors
    from ``model``, tokenizes each word against the model vocab, runs the CTC
    forced-align DP, and returns time-aligned words and phones.
    """
    utt = phonemize(transcript, voice=voice)
    em = model.emissions(np.asarray(audio, dtype=np.float32), sample_rate)
    id_to_phone = {i: p for p, i in em.vocab.items()}

    target: list[int] = []
    word_ranges: list[tuple[str, int, int]] = []
    for w in utt.words:
        start = len(target)
        target.extend(tokenize(w.ipa, em.vocab))
        word_ranges.append((w.text, start, len(target)))
    if not target:
        raise ValueError("transcript produced no phones to align")

    log_probs = np.asarray(em.log_probs, dtype=np.float32)
    spans = _native.forced_align(log_probs, target, blank=em.blank_id)

    fr = float(em.frame_rate)
    phones = tuple(
        TimedPhone(
            label=id_to_phone.get(label, str(label)),
            start_seconds=start_frame / fr,
            end_seconds=end_frame / fr,
            score=score,
        )
        for (_token, label, start_frame, end_frame, score) in spans
    )

    words: list[TimedWord] = []
    for text, s, e in word_ranges:
        wp = phones[s:e]
        if not wp:
            continue
        words.append(
            TimedWord(
                text=text,
                start_seconds=wp[0].start_seconds,
                end_seconds=wp[-1].end_seconds,
                phones=wp,
            )
        )
    return Alignment(words=tuple(words), phones=phones)
