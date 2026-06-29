//! Foundational DSP — pure functions over `&[f32]`, no corpus coupling.
//!
//! Includes window functions (Hann / Hamming / Blackman / Gaussian / Kaiser),
//! the short-time Fourier transform, power spectrograms, and per-frame
//! intensity (linear RMS + dB-FS).
//!
//! Design: see the 2026-05-21 DEVLOG entry "Foundational DSP (C1)".

pub mod formant_preset;
pub mod formants;
pub mod intensity;
pub mod lpc;
pub mod ltas;
pub mod mfcc;
pub mod preset;
pub mod roots;
pub mod spectrogram;
pub mod stft;
pub mod windowing;

pub use formant_preset::{FormantPreset, FormantPresetStore, formant_builtin_presets};
pub use formants::{FormantFrame, FormantsConfig, formants};
pub use intensity::{IntensityFrame, intensity};
pub use lpc::{LpcMethod, LpcResult, autocorr_lpc, burg_lpc, lpc};
pub use ltas::{Ltas, ltas};
pub use mfcc::{
    MelScaleKind, MfccDct, MfccFft, MfccFilterNorm, MfccFilters, MfccFraming, MfccLog, MfccMethod,
    MfccParams, MfccPowerNorm, MfccWindow, log_mel, log_mel_whisper, mfcc, mfcc_with_params,
};
pub use preset::{MfccPreset, MfccPresetStore, builtin_presets};
pub use roots::polynomial_roots;
pub use spectrogram::power_spectrogram;
pub use stft::{Shape, stft};
pub use windowing::{blackman, gaussian, hamming, hann, kaiser};

/// FFT-domain resampling (scipy `resample`-style): the anti-alias
/// low-pass folds into the frequency-bin copy/truncation. Used by GNE
/// (downsample to 10 kHz) and by the E11 VAD path (resample to the
/// model's 16 kHz). Returns `signal` unchanged when rates match or the
/// input is too short.
pub(crate) fn resample_to_hz(signal: &[f32], fs_in: u32, fs_out: u32) -> Vec<f32> {
    use rustfft::{FftPlanner, num_complex::Complex};

    if fs_in == fs_out || signal.len() < 2 {
        return signal.to_vec();
    }
    let n = signal.len();
    let m = ((n as u64 * fs_out as u64) / fs_in as u64) as usize;
    if m < 2 {
        return signal.to_vec();
    }
    let mut planner = FftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(m);
    let mut buf: Vec<Complex<f32>> = signal.iter().map(|&v| Complex::new(v, 0.0)).collect();
    fwd.process(&mut buf);

    let mut out = vec![Complex::<f32>::new(0.0, 0.0); m];
    let half = n.min(m) / 2;
    out[..=half].copy_from_slice(&buf[..=half]); // positive freqs incl. DC
    for k in 1..=half {
        out[m - k] = buf[n - k]; // mirror negatives
    }
    inv.process(&mut out);
    let scale = 1.0 / n as f32;
    out.iter().map(|c| c.re * scale).collect()
}
