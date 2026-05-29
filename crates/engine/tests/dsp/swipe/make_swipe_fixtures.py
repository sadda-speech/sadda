"""Generate the deterministic SWIPE' input fixtures (harmonic tones).

These are plain sums of sines — no pitch algorithm here, so the inputs are
unambiguous and language-agnostic. The *golden* (expected f0 medians) is
produced separately by running Camacho's own MATLAB under Octave
(`run_swipe_octave.m`), which is the authoritative reference.

    python crates/engine/tests/dsp/swipe/make_swipe_fixtures.py

(numpy only; run once locally — the *_input.tsv files are committed.)
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
FS = 16000
CASES = {150, 220, 330}  # true f0 in Hz; filenames f0_<hz>_input.tsv


def harmonic_tone(f0: float, fs: int, dur: float, n_harm: int = 10) -> np.ndarray:
    t = np.arange(int(fs * dur)) / fs
    x = sum(np.sin(2 * np.pi * h * f0 * t) / h for h in range(1, n_harm + 1))
    return (x / np.max(np.abs(x))).astype(np.float64)


def main() -> None:
    for f0 in sorted(CASES):
        x = harmonic_tone(float(f0), FS, 0.5)
        path = HERE / f"f0_{f0}_input.tsv"
        with path.open("w") as fh:
            fh.write(f"n_samples\t{len(x)}\n")
            for v in x:
                fh.write(f"{v:.8e}\n")
        print(f"wrote {path.name} ({len(x)} samples, true f0 {f0} Hz)")


if __name__ == "__main__":
    main()
