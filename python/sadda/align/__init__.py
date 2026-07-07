"""sadda.align — phone-level forced alignment (ASR + alignment).

Aligns a transcript to audio, producing time-aligned **Words** and **Phones**
tiers (Syllables and Whisper ASR come in later slices). See the 2026-07-05
DEVLOG design entry for the architecture: espeak-ng G2P (the target) + an ONNX
CTC acoustic model (per-frame posteriors, in :mod:`sadda.ml`) + a constrained-
Viterbi forced-align DP (the Rust engine).

This slice (A1) is landing incrementally; the G2P surface is here first::

    import sadda
    utt = sadda.align.phonemize("hello world", voice="en-us")
    for w in utt.words:
        print(w.text, w.phones)

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

from .acoustic import Wav2Vec2EspeakModel
from .aligner import (
    Alignment,
    TimedPhone,
    TimedSyllable,
    TimedWord,
    align,
    align_auto,
    import_alignment,
    syllabify,
    tokenize,
)
from .g2p import Utterance, Word, phonemize, split_phones, strip_stress
from .mfa import alignment_from_textgrid, mfa_align, mfa_align_corpus
from .model import AcousticModel, Emissions

__all__ = [
    # G2P
    "Word",
    "Utterance",
    "phonemize",
    "split_phones",
    "strip_stress",
    # acoustic model interface + the neural model
    "AcousticModel",
    "Emissions",
    "Wav2Vec2EspeakModel",
    # alignment
    "align",
    "align_auto",
    "tokenize",
    "Alignment",
    "TimedWord",
    "TimedPhone",
    "import_alignment",
    # syllabification (A3)
    "syllabify",
    "TimedSyllable",
    # MFA (gold-standard passthrough)
    "mfa_align",
    "mfa_align_corpus",
    "alignment_from_textgrid",
]
