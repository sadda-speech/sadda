#!/usr/bin/env python3
"""Generate tiny synthetic ONNX embedding models — one per input
representation — to validate the E12b embedding harness in CI without
large downloads. Each is a single Conv1d that maps its input to a
`[batch, frames, dims]` embedding, exercising the harness's preprocessing
+ ONNX run + output→tier path deterministically.

  waveform-embed:  input [1, N]          -> embeddings [1, T, 8]
  logmel-embed:    input [1, n_mels, T]  -> embeddings [1, T, 8]

Run from the repo root (needs torch):
    python crates/engine/tests/ml_fixtures/make_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
DIMS = 8
N_MELS = 80


class WaveformEmbed(nn.Module):
    """Raw audio [B, N] -> [B, T, DIMS] via a strided Conv1d (frames)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, DIMS, kernel_size=400, stride=160)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # [B, 1, N]
        y = self.conv(x)  # [B, DIMS, T]
        return y.transpose(1, 2)  # [B, T, DIMS]


class LogMelEmbed(nn.Module):
    """Log-mel [B, n_mels, T] -> [B, T, DIMS] via a 1x1 Conv1d over mels."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(N_MELS, DIMS, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)  # [B, DIMS, T]
        return y.transpose(1, 2)  # [B, T, DIMS]


def export(model: nn.Module, dummy: torch.Tensor, path: Path, dyn_axis: int) -> None:
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["embeddings"],
        dynamic_axes={"input": {dyn_axis: "len"}, "embeddings": {1: "frames"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {path}")


def main() -> None:
    export(WaveformEmbed(), torch.zeros(1, 16000), HERE / "waveform-embed" / "model.onnx", dyn_axis=1)
    export(LogMelEmbed(), torch.zeros(1, N_MELS, 100), HERE / "logmel-embed" / "model.onnx", dyn_axis=2)


if __name__ == "__main__":
    (HERE / "waveform-embed").mkdir(parents=True, exist_ok=True)
    (HERE / "logmel-embed").mkdir(parents=True, exist_ok=True)
    main()
