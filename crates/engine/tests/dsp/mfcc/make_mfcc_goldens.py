"""Generate goldens for sadda's named MFCC methods (`engine::dsp::mfcc`).

Each method is a faithful reproduction of a reference implementation; this
script runs the *actual* reference and commits its output as a TSV, so CI
never needs librosa / parselmouth / torch (it reads the committed goldens).
Per the 2026-05-21 DSP method-diversity principle: each MFCC variant names
and validates against the toolkit that defines it.

Backends (run with whichever interpreter has the lib; each writes its own
golden, the shared input.tsv is pure-numpy and identical across runs):

  - librosa  -> librosa.feature.mfcc            (pip: librosa)
  - praat    -> parselmouth "To MFCC..."        (pip: praat-parselmouth)
  - kaldi    -> torchaudio.compliance.kaldi.mfcc (pip: torchaudio)
               NB: torchaudio's kaldi-compliance is PyTorch's faithful Kaldi
               reproduction, NOT Kaldi-proper. A real compute-mfcc-feats
               golden is a future refinement (see BACKLOG).

Shared analysis params mirror sadda's defaults:
  sr=16000, frame=25 ms (n_fft=400), hop=10 ms (160), n_mels=40,
  n_mfcc=13, f_min=0, f_max=sr/2.

Run from repo root, e.g.:
    <venv-with-librosa>/bin/python crates/engine/tests/dsp/mfcc/make_mfcc_goldens.py
    /path/to/python-with-torchaudio  crates/engine/tests/dsp/mfcc/make_mfcc_goldens.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SR, N_FFT, HOP, N_MELS, N_MFCC = 16000, 400, 160, 40, 13
F_MIN, F_MAX = 0.0, SR / 2.0


def make_audio() -> np.ndarray:
    """Deterministic multi-tone, 0.5 s. Committed verbatim so the Rust test
    reads identical samples (no cross-language sin() drift)."""
    t = np.arange(8000, dtype=np.float64) / SR
    x = (
        0.5 * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2 * np.pi * 1700.0 * t)
        + 0.1 * np.sin(2 * np.pi * 3500.0 * t)
    )
    return x.astype(np.float32)


def write_tsv(path: Path, header: str, rows) -> None:
    with path.open("w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write("\t".join(f"{v:.8e}" for v in row) + "\n")


def write_input(audio: np.ndarray) -> None:
    write_tsv(HERE / "mfcc_input.tsv", f"n_samples\t{audio.shape[0]}", [[v] for v in audio])
    print(f"wrote mfcc_input.tsv ({audio.shape[0]} samples)")


def gen_librosa(audio: np.ndarray) -> None:
    import librosa  # type: ignore

    m = librosa.feature.mfcc(
        y=audio.astype(np.float64),
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
    )  # (n_mfcc, n_frames)
    n_mfcc, n_frames = m.shape
    write_tsv(
        HERE / "mfcc_librosa_golden.tsv",
        f"n_mfcc\t{n_mfcc}\tn_frames\t{n_frames}\tlibrosa\t{librosa.__version__}",
        m,
    )
    print(f"wrote mfcc_librosa_golden.tsv  [{n_mfcc}, {n_frames}]  librosa {librosa.__version__}")


def gen_praat(audio: np.ndarray) -> None:
    import parselmouth  # type: ignore
    from parselmouth.praat import call  # type: ignore

    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=SR)
    # Praat "To MFCC...": coefficients, window length (s), time step (s),
    # first filter (mel), distance between filters (mel), max freq (mel).
    mfcc_obj = call(
        snd, "To MFCC...", N_MFCC, N_FFT / SR, HOP / SR, 100.0, 100.0, 0.0
    )
    arr = mfcc_obj.to_array()  # (n_coeffs, n_frames); row 0 is c0
    n_coeffs, n_frames = arr.shape
    write_tsv(
        HERE / "mfcc_praat_golden.tsv",
        f"n_coeffs\t{n_coeffs}\tn_frames\t{n_frames}\tparselmouth\t{parselmouth.__version__}",
        arr,
    )
    print(f"wrote mfcc_praat_golden.tsv  [{n_coeffs}, {n_frames}]  parselmouth {parselmouth.__version__}")


def gen_kaldi(audio: np.ndarray) -> None:
    import torch  # type: ignore
    import torchaudio  # type: ignore

    wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # [1, N]
    feats = torchaudio.compliance.kaldi.mfcc(
        wav,
        sample_frequency=float(SR),
        frame_length=1000.0 * N_FFT / SR,  # ms
        frame_shift=1000.0 * HOP / SR,  # ms
        num_mel_bins=N_MELS,
        num_ceps=N_MFCC,
        low_freq=F_MIN,
        high_freq=F_MAX,
        dither=0.0,  # determinism
        snip_edges=True,
    )  # [n_frames, n_ceps]
    arr = feats.numpy().T  # -> [n_ceps, n_frames] to match the others
    n_ceps, n_frames = arr.shape
    write_tsv(
        HERE / "mfcc_kaldi_golden.tsv",
        f"n_ceps\t{n_ceps}\tn_frames\t{n_frames}\ttorchaudio\t{torchaudio.__version__}",
        arr,
    )
    print(f"wrote mfcc_kaldi_golden.tsv  [{n_ceps}, {n_frames}]  torchaudio {torchaudio.__version__} (kaldi-compliance)")


def main() -> None:
    audio = make_audio()
    write_input(audio)
    for name, fn in (("librosa", gen_librosa), ("praat", gen_praat), ("kaldi", gen_kaldi)):
        try:
            fn(audio)
        except ImportError as e:
            print(f"--  {name}: skipped ({e.name} not importable in this interpreter)")


if __name__ == "__main__":
    main()
