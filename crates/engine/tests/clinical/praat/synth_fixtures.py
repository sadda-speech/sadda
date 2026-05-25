#!/usr/bin/env python3
"""Synthesize the B4 (jitter + shimmer) validation signals.

Each signal is a train of damped-sinusoid glottal pulses with a *known*,
analytically-controlled period jitter and amplitude shimmer, so the test
suite has a ground truth independent of any reference implementation.
Companion to ``jitter_shimmer.praat`` (which measures these WAVs to
produce the Praat golden values).

Perturbation is deterministic pseudo-random (a per-signal seed), not
strictly alternating: alternating jitter/shimmer drives the pitch
tracker into period-doubling, and real voice perturbation is irregular
anyway. Each period and amplitude is perturbed by an independent
uniform draw:

  intervals   P_i = T0 * (1 + jitter_frac · u_i),  u_i ~ U(-1, 1)
  amplitudes  A_i = 1 + shimmer_frac · v_i,         v_i ~ U(-1, 1)

The "local" measures are then computed exactly from the *realized*
period/amplitude sequences (mean consecutive |Δ| / mean), so injected.json
records the true analytic value regardless of the draw.

Run from the repo root:
    python crates/engine/tests/clinical/praat/synth_fixtures.py

Writes ``<...>/fixtures/<name>.wav`` + ``<...>/fixtures/injected.json``.
Deterministic — re-running reproduces identical bytes.
"""

from __future__ import annotations

import json
import math
import random
import struct
import wave
from pathlib import Path

SR = 44_100
DUR = 1.0
RING_HZ = 950.0  # damped-sinusoid "formant" ring frequency
DECAY_S = 0.0016  # ring decay time constant

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

# (name, f0, jitter_frac, shimmer_frac). The realized "local" measure is
# a fraction of `frac` (pseudo-random draws); injected.json records the
# exact analytic value computed from the realized sequence.
CASES = [
    ("clean_120hz", 120.0, 0.0, 0.0),
    ("jitter_150hz", 150.0, 0.03, 0.0),
    ("shimmer_150hz", 150.0, 0.0, 0.10),
    ("jitter_shimmer_200hz", 200.0, 0.025, 0.08),
]


def synth(seed: str, f0: float, jitter_frac: float, shimmer_frac: float):
    rng = random.Random(seed)
    n = int(DUR * SR)
    x = [0.0] * n
    t0 = 1.0 / f0
    ring_len = int(0.9 * t0 * SR)

    t = t0  # first pulse one period in
    periods: list[float] = []
    amps: list[float] = []
    prev_t = None
    while t < DUR - t0:
        amp = 1.0 + shimmer_frac * rng.uniform(-1.0, 1.0)
        start = int(round(t * SR))
        for k in range(ring_len):
            if start + k >= n:
                break
            tau = k / SR
            x[start + k] += amp * math.exp(-tau / DECAY_S) * math.sin(
                2.0 * math.pi * RING_HZ * tau
            )
        amps.append(amp)
        if prev_t is not None:
            periods.append(t - prev_t)
        prev_t = t
        # advance by a perturbed period
        t += t0 * (1.0 + jitter_frac * rng.uniform(-1.0, 1.0))

    # Normalize to 0.9 peak to stay clear of clipping.
    peak = max(abs(v) for v in x) or 1.0
    x = [0.9 * v / peak for v in x]

    # Analytic local measures from the realized sequences.
    def local_rel(seq: list[float]) -> float:
        if len(seq) < 2:
            return 0.0
        diffs = sum(abs(seq[j] - seq[j - 1]) for j in range(1, len(seq)))
        mean = sum(seq) / len(seq)
        return (diffs / (len(seq) - 1)) / mean

    jitter_local = local_rel(periods)
    shimmer_local = local_rel(amps)
    return x, jitter_local, shimmer_local


def write_wav(path: Path, samples: list[float]) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        frames = b"".join(
            struct.pack("<h", max(-32768, min(32767, int(round(v * 32767))))) for v in samples
        )
        w.writeframes(frames)


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    injected = {"sample_rate": SR, "duration_s": DUR, "signals": {}}
    for name, f0, jf, sf in CASES:
        samples, jit, shim = synth(name, f0, jf, sf)
        write_wav(FIXTURES / f"{name}.wav", samples)
        injected["signals"][name] = {
            "f0_hz": f0,
            "jitter_frac": jf,
            "shimmer_frac": sf,
            "analytic_jitter_local": round(jit, 6),
            "analytic_shimmer_local": round(shim, 6),
        }
        print(f"{name}: analytic jitter_local={jit:.4%} shimmer_local={shim:.4%}")
    (FIXTURES / "injected.json").write_text(json.dumps(injected, indent=2) + "\n")
    print(f"wrote {len(CASES)} WAVs + injected.json to {FIXTURES}")


if __name__ == "__main__":
    main()
