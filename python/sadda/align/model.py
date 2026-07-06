"""sadda.align.model — the acoustic-model interface for forced alignment.

An acoustic model turns audio into per-frame CTC log-probabilities over a phone
vocabulary (stage 1 of the pipeline; see the 2026-07-05 DEVLOG design entry).
The default will be the espeak-IPA wav2vec2 CTC net (`facebook/wav2vec2-lv-60-
espeak-cv-ft`, Apache-2.0) run via ONNX in `sadda.ml` — a later slice. The
:class:`AcousticModel` protocol keeps the aligner backend-agnostic (and lets
tests supply a mock), the same way :mod:`sadda.tts` abstracts its backends.

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

__all__ = ["Emissions", "AcousticModel"]


@dataclass(frozen=True)
class Emissions:
    """Per-frame acoustic posteriors for one audio clip.

    Attributes:
        log_probs: ``(T, C)`` float32 array of log-probabilities over the ``C``
            classes (phones + blank), one row per frame.
        vocab: Maps each phone label to its class id (column of ``log_probs``).
        frame_rate: Emission frames per second (e.g. 50.0 for a wav2vec2 CTC
            head with a 20 ms stride) — used to convert frame spans to seconds.
        blank_id: The CTC blank class id.
    """

    log_probs: np.ndarray
    vocab: Mapping[str, int]
    frame_rate: float
    blank_id: int


@runtime_checkable
class AcousticModel(Protocol):
    """Anything that turns audio into :class:`Emissions`.

    Structural protocol — a backend need only provide ``emissions``; no
    subclassing required (so a test mock or a user's own model both qualify).
    """

    def emissions(self, audio: np.ndarray, sample_rate: int) -> Emissions:
        """Return per-frame CTC log-probabilities for ``audio``."""
        ...
