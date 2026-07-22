//! Long-term average spectrum (LTAS) and its derived voice-quality
//! measures.
//!
//! The LTAS is the mean power spectrum over the whole signal (Welch
//! averaging), binned into fixed-Hz bands and expressed in dB. It's a
//! core phonetics tool in its own right, and the source of the spectral
//! **slope** and **tilt** parameters the AVQI clinical index (B6) builds
//! on. The derived measures (slope, tilt, alpha ratio) are all
//! *differences/ratios* of dB levels, so they're invariant to the
//! spectrum's overall normalization — only its shape matters, which is
//! why they match Praat readily.
//!
//! ## References
//! - Frøkjær-Jensen, B. & Prytz, S. (1976). "Registration of voice quality."
//!   *Brüel & Kjær Technical Review* 3: 3–17. — origin of the **alpha ratio**
//!   (the >1 kHz vs <1 kHz spectral-balance measure). No stable weblink is
//!   available (a Brüel & Kjær technical review with no DOI).
//! - **Slope** and **tilt** are standard LTAS spectral-balance descriptors
//!   rather than any single group's coinage; sadda mirrors Praat's
//!   `Ltas: Get slope` (energy averaging) and a least-squares dB-vs-frequency
//!   fit, validated against Praat
//!   (<https://www.fon.hum.uva.nl/praat/manual/Ltas.html>).

use realfft::RealFftPlanner;

use crate::dsp::windowing::hann;
use crate::units::Decibels;

/// A long-term average spectrum: mean power per frequency band, in dB.
#[derive(Debug, Clone)]
pub struct Ltas {
    /// Source sample rate, Hz.
    pub sample_rate: u32,
    /// Band width, Hz. Band `i` spans `[i·bin_hz, (i+1)·bin_hz)`.
    pub bin_hz: f32,
    /// Mean power per band, in dB (arbitrary 0 dB reference — only
    /// level *differences* are meaningful).
    pub levels_db: Vec<f32>,
}

impl Ltas {
    /// Mean band *power* (linear) over `[lo, hi)` Hz.
    fn band_power(&self, lo: f32, hi: f32) -> f64 {
        let b0 = (lo / self.bin_hz).floor() as usize;
        let b1 = ((hi / self.bin_hz).ceil() as usize).min(self.levels_db.len());
        if b1 <= b0 {
            return 0.0;
        }
        let sum: f64 = self.levels_db[b0..b1]
            .iter()
            .map(|&d| 10f64.powf(d as f64 / 10.0))
            .sum();
        sum / (b1 - b0) as f64
    }

    /// Spectral **slope** (dB): the high-band / low-band energy ratio,
    /// `10·log10(mean_power_high / mean_power_low)`. Matches Praat's
    /// `Ltas: Get slope` with "energy" averaging. Negative = energy
    /// falls with frequency.
    pub fn slope(&self, low: (f32, f32), high: (f32, f32)) -> Decibels {
        let lo = self.band_power(low.0, low.1).max(1e-30);
        let hi = self.band_power(high.0, high.1).max(1e-30);
        Decibels::new((10.0 * (hi / lo).log10()) as f32)
    }

    /// Spectral **tilt**: the slope of a least-squares straight line fit
    /// to the band dB levels over `[f_lo, f_hi)`, in dB per kHz.
    pub fn tilt(&self, f_lo: f32, f_hi: f32) -> f32 {
        let b0 = (f_lo / self.bin_hz).floor() as usize;
        let b1 = ((f_hi / self.bin_hz).ceil() as usize).min(self.levels_db.len());
        if b1 <= b0 + 1 {
            return 0.0;
        }
        // Regress level (dB) on frequency (kHz).
        let xs: Vec<f64> = (b0..b1)
            .map(|b| (b as f32 + 0.5) as f64 * self.bin_hz as f64 / 1000.0)
            .collect();
        let ys: Vec<f64> = self.levels_db[b0..b1].iter().map(|&d| d as f64).collect();
        let n = xs.len() as f64;
        let mx = xs.iter().sum::<f64>() / n;
        let my = ys.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut den = 0.0;
        for (x, y) in xs.iter().zip(&ys) {
            num += (x - mx) * (y - my);
            den += (x - mx) * (x - mx);
        }
        if den.abs() < 1e-12 {
            0.0
        } else {
            (num / den) as f32
        }
    }

    /// Alpha ratio: energy above 1 kHz relative to below 1 kHz, in dB
    /// (a common breathiness/brightness measure; Frøkjær-Jensen & Prytz 1976 —
    /// see the module references).
    pub fn alpha_ratio(&self) -> Decibels {
        let nyq = self.sample_rate as f32 / 2.0;
        self.slope((0.0, 1000.0), (1000.0, nyq))
    }
}

/// Computes the long-term average spectrum of `samples` with `bin_hz`-wide
/// bands. Welch averaging: overlapping Hann frames → mean power per FFT
/// bin → aggregated into `bin_hz` bands. Returns an empty LTAS if the
/// signal is shorter than one frame.
pub fn ltas(samples: &[f32], sample_rate: u32, bin_hz: f32) -> Ltas {
    let n = 2048usize;
    let empty = Ltas {
        sample_rate,
        bin_hz,
        levels_db: Vec::new(),
    };
    if samples.len() < n || bin_hz <= 0.0 {
        return empty;
    }
    let window = hann(n);
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();

    // Mean power per FFT bin (Welch, 50% overlap).
    let mut bin_power = vec![0.0_f64; output.len()];
    let hop = n / 2;
    let mut frames = 0usize;
    let mut s = 0;
    while s + n <= samples.len() {
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = samples[s + i] * window[i];
        }
        fft.process(&mut input, &mut output)
            .expect("fft sized via make_*_vec");
        for (acc, c) in bin_power.iter_mut().zip(output.iter()) {
            *acc += c.norm_sqr() as f64;
        }
        frames += 1;
        s += hop;
    }
    for p in &mut bin_power {
        *p /= frames as f64;
    }

    // Aggregate FFT bins (width sr/n) into bin_hz bands.
    let fft_bin_hz = sample_rate as f32 / n as f32;
    let n_bands = (sample_rate as f32 / 2.0 / bin_hz).ceil() as usize;
    let mut band_power = vec![0.0_f64; n_bands];
    let mut band_count = vec![0usize; n_bands];
    for (k, &p) in bin_power.iter().enumerate() {
        let f = k as f32 * fft_bin_hz;
        let b = (f / bin_hz) as usize;
        if b < n_bands {
            band_power[b] += p;
            band_count[b] += 1;
        }
    }
    let levels_db = band_power
        .iter()
        .zip(&band_count)
        .map(|(&p, &c)| {
            let mean = if c > 0 { p / c as f64 } else { 1e-30 };
            10.0 * (mean + 1e-30).log10() as f32
        })
        .collect();

    Ltas {
        sample_rate,
        bin_hz,
        levels_db,
    }
}
