#!/usr/bin/env python3
"""Generate librosa golden values for sadda's YIN + pYIN validation.

For each WAV in the fixtures directory, runs:

  - ``librosa.yin(fmin=75, fmax=500, trough_threshold=0.1)``
  - ``librosa.pyin(fmin=75, fmax=500, n_thresholds=100, ...)``

with the same frame_length / hop_length defaults as sadda's pitch
config (30 ms / 10 ms at the file's sample rate). Writes one TSV row
per (fixture, method) capturing the voiced-frame count and median
voiced f0 — the same scalars the Rust integration test compares.

These values are the *record of how the target (golden) data was
produced*. Per the 2026-05-21 DSP method-diversity principle, librosa
is the primary reference for the YIN family. The resulting TSV is
committed so CI never needs librosa.

Run from the repo root:

    python crates/engine/tests/dsp/librosa/pitch_yin_pyin.py

Writes ``<fixtures>/pitch_yin_pyin_golden.tsv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np

FIXTURES = (
    Path(__file__).resolve().parent.parent.parent / "clinical" / "fixtures"
)

# Matches `PitchConfig::default()` on the Rust side: 30 ms frame, 10 ms
# hop, 75 Hz floor, 500 Hz ceiling. librosa uses these directly.
FMIN = 75.0
FMAX = 500.0
FRAME_MS = 30.0
HOP_MS = 10.0


def median_voiced(f0: np.ndarray, voiced: np.ndarray) -> tuple[int, float]:
    """Returns (n_voiced_frames, median_voiced_f0)."""
    voiced_f0 = f0[voiced & np.isfinite(f0)]
    if voiced_f0.size == 0:
        return 0, float("nan")
    return int(voiced_f0.size), float(np.median(voiced_f0))


def main() -> None:
    out = FIXTURES / "pitch_yin_pyin_golden.tsv"
    rows: list[str] = [
        "\t".join(["signal", "method", "n_voiced_frames", "median_f0_hz"])
    ]

    wavs = sorted(FIXTURES.glob("*.wav"))
    if not wavs:
        sys.exit(f"no WAVs found under {FIXTURES}")

    for wav in wavs:
        y, sr = librosa.load(str(wav), sr=None, mono=True)
        frame_length = int(round(FRAME_MS * 1e-3 * sr))
        hop_length = int(round(HOP_MS * 1e-3 * sr))

        # YIN: f0 array per frame; librosa uses NaN for "no valid f0",
        # so we infer voiced = isfinite.
        yin_f0 = librosa.yin(
            y,
            fmin=FMIN,
            fmax=FMAX,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            trough_threshold=0.1,
        )
        yin_voiced = np.isfinite(yin_f0)
        n_yin, med_yin = median_voiced(yin_f0, yin_voiced)
        rows.append(f"{wav.stem}\tyin\t{n_yin}\t{med_yin:.6f}")

        # pYIN: returns (f0, voiced_flag, voiced_prob).
        pyin_f0, pyin_voiced, _pyin_p = librosa.pyin(
            y,
            fmin=FMIN,
            fmax=FMAX,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
            n_thresholds=100,
        )
        # Librosa's pYIN reports unvoiced frames as NaN; treat
        # finite + voiced=True as the voiced set.
        pyin_voiced_mask = pyin_voiced & np.isfinite(pyin_f0)
        n_pyin, med_pyin = median_voiced(pyin_f0, pyin_voiced_mask)
        rows.append(f"{wav.stem}\tpyin\t{n_pyin}\t{med_pyin:.6f}")

    out.write_text("\n".join(rows) + "\n")
    print(f"Wrote {len(wavs)} fixtures × 2 methods to {out}")


if __name__ == "__main__":
    main()
