#!/usr/bin/env python
"""Export ``facebook/wav2vec2-lv-60-espeak-cv-ft`` to the ONNX artifact that
``sadda.align`` uses (hosted at ``sadda-speech/wav2vec2-espeak-ctc``).

This is a **format conversion** of Meta's Apache-2.0 weights — no retraining —
so the artifact stays Apache-2.0. It produces a single-file **fp16**
``model.onnx`` + the 392-token espeak-IPA ``vocab.json``.

Dev-only; NOT a shipped dependency. Run once in a throwaway environment::

    pip install torch transformers onnx onnxruntime onnxconverter-common
    python tools/align/export_onnx.py --out ./out
    # then upload out/{model.onnx,vocab.json,README.md} to the HF repo

Why this exact recipe:

* **Dynamo exporter, not the legacy TorchScript one.** The legacy
  ``torch.onnx.export`` mistraces wav2vec2's scaled-dot-product attention and
  produces wrong logits (verified: max|Δlogit| ≈ 15 vs the PyTorch model). The
  dynamo exporter is faithful (max|Δlogit| ≈ 8e-4).
* **Embed + fp16.** The dynamo exporter emits weights as external data; we embed
  them into one file, then cast to fp16 — which halves the download (~1.27 GB →
  ~635 MB) and gives *identical* alignment boundaries to fp32 on test audio.

References: Xu, Q., Baevski, A. & Auli, M. (2022), Simple and Effective Zero-shot
Cross-lingual Phoneme Recognition, Interspeech 2022 (doi:10.21437/Interspeech.2022-60);
Baevski et al. (2020), wav2vec 2.0, NeurIPS 33.
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2ForCTC

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"


class _LogitsOnly(torch.nn.Module):
    """Wrap the CTC model so ONNX sees a single logits tensor, not a dataclass."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.model(input_values=input_values).logits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--fp32", action="store_true", help="keep fp32 (skip the fp16 cast)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).eval()
    wrap = _LogitsOnly(model)

    tmp = os.path.join(args.out, "_dynamo.onnx")
    torch.onnx.export(
        wrap,
        (torch.zeros(1, 16000),),
        tmp,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "length"},
            "logits": {0: "batch", 1: "frames"},
        },
        opset_version=17,  # dynamo exporter (torch>=2.9 default)
    )

    # Embed the external-data weights into a single file.
    model_path = os.path.join(args.out, "model.onnx")
    onnx.save(onnx.load(tmp, load_external_data=True), model_path, save_as_external_data=False)

    if not args.fp32:
        from onnxconverter_common import float16

        m16 = float16.convert_float_to_float16(onnx.load(model_path), keep_io_types=True)
        onnx.save(m16, model_path)

    for f in os.listdir(args.out):
        if f.startswith("_dynamo"):
            os.remove(os.path.join(args.out, f))

    # The 392-token espeak-IPA vocab.json ships with the source model. Fetch it
    # directly rather than via the HF processor — that tokenizer pulls in the
    # GPL `phonemizer` library, which sadda deliberately avoids.
    shutil.copy(hf_hub_download(MODEL_ID, "vocab.json"), os.path.join(args.out, "vocab.json"))

    # Sanity parity check: PyTorch vs the exported ONNX on random audio.
    x = np.random.randn(1, 16000).astype(np.float32)
    with torch.no_grad():
        pt = wrap(torch.from_numpy(x)).numpy()
    on = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"]).run(
        ["logits"], {"input_values": x}
    )[0]
    print(f"wrote {model_path} + vocab.json  (max|Δlogit| = {np.abs(pt - on).max():.2e})")


if __name__ == "__main__":
    main()
