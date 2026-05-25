//! Foundational DSP — pure functions over `&[f32]`, no corpus coupling.
//!
//! Includes window functions (Hann / Hamming / Blackman / Gaussian / Kaiser),
//! the short-time Fourier transform, power spectrograms, and per-frame
//! intensity (linear RMS + dB-FS).
//!
//! Design: see the 2026-05-21 DEVLOG entry "Foundational DSP (C1)".

pub mod formants;
pub mod intensity;
pub mod lpc;
pub mod ltas;
pub mod mfcc;
pub mod roots;
pub mod spectrogram;
pub mod stft;
pub mod windowing;

pub use formants::{FormantFrame, FormantsConfig, formants};
pub use intensity::{IntensityFrame, intensity};
pub use lpc::{LpcMethod, LpcResult, autocorr_lpc, burg_lpc, lpc};
pub use ltas::{Ltas, ltas};
pub use mfcc::mfcc;
pub use roots::polynomial_roots;
pub use spectrogram::power_spectrogram;
pub use stft::{Shape, stft};
pub use windowing::{blackman, gaussian, hamming, hann, kaiser};
