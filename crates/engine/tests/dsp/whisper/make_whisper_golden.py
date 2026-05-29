"""Generate the golden for `engine::dsp::log_mel_whisper`.

Reproduces OpenAI Whisper's `log_mel_spectrogram` **verbatim** (the body of
`whisper/audio.py`, `padding=0`) using Whisper's own pre-computed mel
filterbank asset — so the golden is an independent reference, not a
re-derivation of our Rust code. Run once locally; the TSVs it writes are
committed and CI compares against them (no torch/whisper needed in CI).

Prereqs (conda env): torch + numpy, and Whisper's filterbank:

    curl -sSL -o /tmp/mel_filters.npz \
      https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
    python crates/engine/tests/dsp/whisper/make_whisper_golden.py

Reference: https://github.com/openai/whisper/blob/main/whisper/audio.py
"""

from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
SR, N_FFT, HOP, N_MELS = 16000, 400, 160, 80
FILTERS_NPZ = "/tmp/mel_filters.npz"


def make_audio() -> np.ndarray:
    # Deterministic multi-tone, 0.5 s. Committed verbatim so the Rust test
    # reads the identical samples (no cross-language sin() drift).
    t = np.arange(8000, dtype=np.float32) / SR
    x = (
        0.5 * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2 * np.pi * 1700.0 * t)
        + 0.1 * np.sin(2 * np.pi * 3500.0 * t)
    )
    return x.astype(np.float32)


def whisper_log_mel(audio: np.ndarray) -> np.ndarray:
    """Verbatim `whisper.audio.log_mel_spectrogram(audio, padding=0)`."""
    filters = torch.from_numpy(np.load(FILTERS_NPZ)["mel_80"])
    audio_t = torch.from_numpy(audio)
    window = torch.hann_window(N_FFT)
    stft = torch.stft(audio_t, N_FFT, HOP, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.numpy()  # [n_mels, n_frames]


def write_tsv(path: Path, header: str, rows) -> None:
    with path.open("w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write("\t".join(f"{v:.8e}" for v in row) + "\n")


def main() -> None:
    audio = make_audio()
    mel = whisper_log_mel(audio)  # [80, T]
    n_mels, n_frames = mel.shape
    print(f"audio {audio.shape[0]} samples -> mel [{n_mels}, {n_frames}]")

    write_tsv(HERE / "whisper_mel_input.tsv", "n_samples\t8000", [[v] for v in audio])
    # One row per mel band, n_frames columns.
    write_tsv(HERE / "whisper_mel_golden.tsv", f"n_mels\t{n_mels}\tn_frames\t{n_frames}", mel)
    print("wrote whisper_mel_input.tsv + whisper_mel_golden.tsv")


if __name__ == "__main__":
    main()
