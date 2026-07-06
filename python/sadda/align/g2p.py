"""sadda.align.g2p — grapheme-to-phoneme via espeak-ng.

Turns a transcript into per-word IPA phone sequences, the *target* side of forced
alignment (see the 2026-07-05 DEVLOG design entry). Uses the system ``espeak-ng``
binary — the same rule-based G2P engine bundled for TTS — so it covers 100+
languages and needs no Python dependency or model download.

This layer is **model-agnostic**: it emits IPA phones with suprasegmental stress
marks stripped, but does *not* map them to a specific acoustic model's token
vocabulary — that reconciliation (a greedy longest-match against the model's
vocab) happens at alignment time, because it's model-specific. The espeak-ng
token match was validated against ``facebook/wav2vec2-lv-60-espeak-cv-ft`` (every
segmental phone is a model token once ``ˈ ˌ ː`` are handled); see the design entry.

espeak-ng is a rule-based tool, not an academic method, so it carries no citation
in the :mod:`~sadda.citation` registry (like TextGrid import) — the reference is
the eSpeak NG project itself.

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

import shutil
import string
import subprocess
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from sadda._stability import provisional

__all__ = ["Word", "Utterance", "phonemize", "strip_stress", "split_phones"]

# Combining double inverted breve — the affricate/diphthong tie (t͡ʃ, a͡ɪ).
_TIE = "͡"
# IPA primary (ˈ) and secondary (ˌ) stress marks — suprasegmental, never phones.
_STRESS = "ˈˌ"
# Punctuation to peel off word edges before phonemizing (keeps word-internal
# apostrophes, e.g. don't).
_EDGE_PUNCT = string.punctuation + "…—–«»¡¿"


# [docs:sadda.align.Word]
@dataclass(frozen=True)
class Word:
    """One word and its phonemization.

    Attributes:
        text: The word as written (edge punctuation removed).
        ipa: The espeak-ng IPA string for the word, stress marks stripped.
        phones: ``ipa`` segmented into individual IPA phones (base character +
            its trailing length / tie / diacritic modifiers). The aligner
            reconciles these against the acoustic model's vocabulary.
    """

    text: str
    ipa: str
    phones: tuple[str, ...]


# [docs:sadda.align.Utterance]
@dataclass(frozen=True)
class Utterance:
    """A phonemized transcript: an ordered list of :class:`Word`\\ s."""

    words: tuple[Word, ...]
    voice: str

    # [docs:sadda.align.Utterance.phones]
    @property
    def phones(self) -> tuple[str, ...]:
        """The full phone sequence across all words, in order."""
        return tuple(p for w in self.words for p in w.phones)


# [docs:sadda.align.strip_stress]
def strip_stress(ipa: str) -> str:
    """Remove IPA primary/secondary stress marks (``ˈ ˌ``)."""
    return ipa.translate({ord(c): None for c in _STRESS})


# [docs:sadda.align.split_phones]
def split_phones(ipa: str) -> list[str]:
    """Segment an IPA string into phones.

    A phone is a base character plus any trailing modifiers that bind to it:
    combining diacritics (Unicode ``Mn``, including the affricate/diphthong tie),
    and modifier letters (``Lm`` — length ``ː``, aspiration ``ʰ``, palatalization
    ``ʲ``, …). A tie makes the following base join the current phone (``t͡ʃ`` → one
    phone); untied sequences split (``tʃ`` → ``t``, ``ʃ``), matching how espeak-ng
    emits them.
    """
    phones: list[str] = []
    for ch in ipa:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        joins = phones and (cat in ("Mn", "Lm") or phones[-1].endswith(_TIE))
        if joins:
            phones[-1] += ch
        else:
            phones.append(ch)
    return phones


@lru_cache(maxsize=1)
def _espeak_binary() -> str:
    binary = shutil.which("espeak-ng")
    if binary is None:
        raise FileNotFoundError(
            "espeak-ng executable not found. Install it (e.g. `apt install "
            "espeak-ng`, `brew install espeak-ng`)."
        )
    return binary


def _espeak_ipa(text: str, voice: str) -> str:
    out = subprocess.run(
        [_espeak_binary(), "-q", "--ipa", "-v", voice, text],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.replace("\n", " ").strip()


# [docs:sadda.align.phonemize]
@provisional
def phonemize(text: str, *, voice: str = "en-us") -> Utterance:
    """Phonemize a transcript to per-word IPA phones via espeak-ng.

    Words are phonemized individually (a clean 1:1 word→phones mapping, at the
    cost of cross-word coarticulation — acceptable for an alignment target).
    Edge punctuation is stripped; stress marks are removed. ``voice`` is an
    espeak-ng language id (e.g. ``"en-us"``, ``"de"``, ``"cmn"``).
    """
    words: list[Word] = []
    for token in text.split():
        w = token.strip(_EDGE_PUNCT)
        if not w:
            continue
        ipa = strip_stress(_espeak_ipa(w, voice)).strip()
        words.append(Word(text=w, ipa=ipa, phones=tuple(split_phones(ipa))))
    return Utterance(words=tuple(words), voice=voice)
