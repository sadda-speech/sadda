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

from .g2p import Utterance, Word, phonemize, split_phones, strip_stress

__all__ = [
    "Word",
    "Utterance",
    "phonemize",
    "split_phones",
    "strip_stress",
]
