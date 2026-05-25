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

# (name, f0, jitter_frac, shimmer_frac, noise_hnr_db). The realized
# jitter/shimmer "local" measure is a fraction of `frac` (pseudo-random
# draws); injected.json records the exact analytic value. noise_hnr_db
# (or None) injects additive Gaussian noise at a target harmonics-to-
# noise ratio for the HNR fixtures (B5).
CASES = [
    ("clean_120hz", 120.0, 0.0, 0.0, None),
    ("jitter_150hz", 150.0, 0.03, 0.0, None),
    ("shimmer_150hz", 150.0, 0.0, 0.10, None),
    ("jitter_shimmer_200hz", 200.0, 0.025, 0.08, None),
    ("hnr_high_120hz", 120.0, 0.0, 0.0, 25.0),
    ("hnr_mid_120hz", 120.0, 0.0, 0.0, 12.0),
]


def synth(
    seed: str,
    f0: float,
    jitter_frac: float,
    shimmer_frac: float,
    noise_hnr_db: float | None = None,
):
    rng = random.Random(seed)
    n = int(DUR * SR)
    t0 = 1.0 / f0
    periods: list[float] = []
    amps: list[float] = []

    if noise_hnr_db is not None:
        # HNR fixtures: a sustained harmonic tone (glottal-source-like,
        # 1/h harmonics). A continuous periodic signal makes the
        # autocorrelation at the period lag ≈ R(0), so plain-ACF HNR
        # recovers the injected ratio — unlike the pulse train below,
        # which is built for jitter/shimmer.
        n_harmonics = min(30, int((SR / 2) / f0) - 1)
        x = [0.0] * n
        for i in range(n):
            t = i / SR
            x[i] = sum(
                (1.0 / h) * math.sin(2.0 * math.pi * h * f0 * t)
                for h in range(1, n_harmonics + 1)
            )
    else:
        # Jitter/shimmer fixtures: a train of damped-sinusoid glottal
        # pulses with controlled period/amplitude perturbation.
        x = [0.0] * n
        ring_len = int(0.9 * t0 * SR)
        t = t0  # first pulse one period in
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

    # Normalize the (harmonic) signal to 0.9 peak.
    peak = max(abs(v) for v in x) or 1.0
    x = [0.9 * v / peak for v in x]

    # Optional additive noise at a target harmonics-to-noise ratio.
    hnr_db: float | None = None
    if noise_hnr_db is not None:
        sig_power = sum(v * v for v in x) / len(x)
        noise_std = math.sqrt(sig_power / (10.0 ** (noise_hnr_db / 10.0)))
        noise = [rng.gauss(0.0, noise_std) for _ in range(len(x))]
        noise_power = sum(e * e for e in noise) / len(noise)
        hnr_db = 10.0 * math.log10(sig_power / noise_power)
        x = [s + e for s, e in zip(x, noise)]
        # Renormalize (signal+noise) to 0.9 — preserves the HNR ratio.
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
    return x, jitter_local, shimmer_local, hnr_db


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
    for name, f0, jf, sf, noise_hnr in CASES:
        samples, jit, shim, hnr = synth(name, f0, jf, sf, noise_hnr)
        write_wav(FIXTURES / f"{name}.wav", samples)
        entry = {
            "f0_hz": f0,
            "jitter_frac": jf,
            "shimmer_frac": sf,
            "analytic_jitter_local": round(jit, 6),
            "analytic_shimmer_local": round(shim, 6),
        }
        if hnr is not None:
            entry["analytic_hnr_db"] = round(hnr, 3)
        injected["signals"][name] = entry
        hnr_str = f" hnr={hnr:.2f}dB" if hnr is not None else ""
        print(f"{name}: jitter_local={jit:.4%} shimmer_local={shim:.4%}{hnr_str}")
    (FIXTURES / "injected.json").write_text(json.dumps(injected, indent=2) + "\n")
    print(f"wrote {len(CASES)} WAVs + injected.json to {FIXTURES}")


if __name__ == "__main__":
    main()
