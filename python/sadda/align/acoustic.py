"""sadda.align.acoustic — the neural acoustic model for forced alignment.

:class:`Wav2Vec2EspeakModel` wraps ``facebook/wav2vec2-lv-60-espeak-cv-ft``
(Apache-2.0) — a wav2vec 2.0 CTC network fine-tuned to emit **espeak IPA
phonemes** — exported to ONNX and run via ``onnxruntime`` (the ``sadda[align]``
extra). It produces per-frame CTC log-probabilities (:class:`~sadda.align.model.Emissions`)
for the forced-align DP; its 392-token espeak-IPA output vocabulary is exactly
what :func:`sadda.align.tokenize` matches the espeak-ng G2P against.

The model is supplied by path (``model_path`` + ``vocab_path``). A hosted
``hf://`` fetch (behind ``sadda[align]``) is the packaging step and lands with
the model hosting.

References
---------
- Xu, Q., Baevski, A. & Auli, M. (2022). Simple and Effective Zero-shot
  Cross-lingual Phoneme Recognition. *Interspeech 2022*, 2113–2117.
  doi:10.21437/Interspeech.2022-60. — the espeak-IPA fine-tuned model; the
  citation the ``sadda.align.wav2vec2_espeak`` processor registers.
- Baevski, A., Zhou, Y., Mohamed, A. & Auli, M. (2020). wav2vec 2.0: A
  Framework for Self-Supervised Learning of Speech Representations. *NeurIPS
  33*, 12449–12460. — the underlying self-supervised architecture.

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

from sadda._stability import provisional

from .model import Emissions

__all__ = ["Wav2Vec2EspeakModel"]

#: Reverse-DNS processor id whose citation lives in the engine registry
#: (`crates/engine/src/citation.rs` → Xu et al. 2022).
PROCESSOR_ID = "sadda.align.wav2vec2_espeak"


def _import_onnxruntime():
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "onnxruntime is required for the neural acoustic model. Install it "
            'with `pip install "sadda[align]"`.'
        ) from exc
    return onnxruntime


@provisional
class Wav2Vec2EspeakModel:
    """espeak-IPA wav2vec2 CTC acoustic model (ONNX) — an ``AcousticModel``.

    Args:
        model_path: ONNX model file. Input: a mono 16 kHz waveform ``(1, N)``;
            output: ``(1, T, C)`` logits.
        vocab_path: ``vocab.json`` mapping each phone to its class id.
        frame_rate: Emission frames per second — 50.0 for wav2vec2's 320× stride
            at 16 kHz (used to turn frame spans into seconds).
        blank_token: The vocab key that is the CTC blank (default ``"<pad>"``).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        vocab_path: Union[str, Path],
        *,
        frame_rate: float = 50.0,
        blank_token: str = "<pad>",
    ) -> None:
        ort = _import_onnxruntime()
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self._output = self._session.get_outputs()[0].name
        self.vocab: dict[str, int] = json.loads(
            Path(vocab_path).read_text(encoding="utf-8")
        )
        if blank_token not in self.vocab:
            raise ValueError(f"blank token {blank_token!r} not in vocab")
        self.blank_id = self.vocab[blank_token]
        self.frame_rate = float(frame_rate)

    def emissions(self, audio: np.ndarray, sample_rate: int) -> Emissions:
        """Run the model over ``audio`` (mono, 16 kHz) → per-frame log-probs."""
        if sample_rate != 16000:
            raise ValueError(
                f"Wav2Vec2EspeakModel expects 16 kHz audio, got {sample_rate} Hz "
                "— resample first."
            )
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        # wav2vec2 input normalisation: zero mean, unit variance.
        x = (x - x.mean()) / (x.std() + 1e-7)
        logits = self._session.run([self._output], {self._input: x[None, :]})[0][0]
        logits = np.asarray(logits, dtype=np.float32)
        # logits → log-softmax (the DP consumes log-probabilities).
        mx = logits.max(axis=-1, keepdims=True)
        log_probs = (logits - mx) - np.log(
            np.exp(logits - mx).sum(axis=-1, keepdims=True)
        )
        return Emissions(
            log_probs=log_probs,
            vocab=self.vocab,
            frame_rate=self.frame_rate,
            blank_id=self.blank_id,
        )
