//! Foundational DSP — pure functions over `&[f32]`, no corpus coupling.
//!
//! Includes window functions (Hann / Hamming / Blackman / Gaussian / Kaiser),
//! the short-time Fourier transform, power spectrograms, and per-frame
//! intensity (linear RMS + dB-FS).
//!
//! Design: see the 2026-05-21 DEVLOG entry "Foundational DSP (C1)".

pub mod intensity;
pub mod spectrogram;
pub mod stft;
pub mod windowing;

pub use intensity::{IntensityFrame, intensity};
pub use spectrogram::power_spectrogram;
pub use stft::{Shape, stft};
pub use windowing::{blackman, gaussian, hamming, hann, kaiser};
