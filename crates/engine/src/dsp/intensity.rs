//! Per-frame intensity: linear RMS amplitude + dB-FS (dB relative to digital
//! full-scale at amplitude 1.0). dB-SPL (Praat convention) is a later slice
//! that needs microphone calibration plumbed through `Instrument`.

/// One frame of intensity output: linear RMS amplitude, its dB-FS conversion,
/// and the time at the center of the frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntensityFrame {
    /// Time at the center of the analysis frame, in seconds.
    pub time_seconds: f64,
    /// Linear RMS amplitude over the frame: `sqrt(mean(samples²))`.
    pub rms: f32,
    /// dB-FS: `20 * log10(rms)`. Clamped to [`DB_FS_FLOOR`] for silent
    /// frames so callers don't have to special-case `-∞`.
    pub db_fs: f32,
}

/// Floor for dB-FS to avoid `-inf` on completely silent frames.
pub const DB_FS_FLOOR: f32 = -200.0;

/// Computes per-frame intensity over `samples` with a sliding window.
///
/// `frame_size_seconds` and `hop_seconds` are converted to samples via
/// `sample_rate`. Frames overflowing the end of `samples` are dropped (no
/// zero-padding); if `samples` is shorter than one frame, returns empty.
pub fn intensity(
    samples: &[f32],
    sample_rate: u32,
    frame_size_seconds: f32,
    hop_seconds: f32,
) -> Vec<IntensityFrame> {
    assert!(sample_rate > 0, "intensity: sample_rate must be > 0");
    assert!(
        frame_size_seconds > 0.0,
        "intensity: frame_size_seconds must be > 0"
    );
    assert!(hop_seconds > 0.0, "intensity: hop_seconds must be > 0");

    let frame_size = (frame_size_seconds * sample_rate as f32).round() as usize;
    let hop_size = (hop_seconds * sample_rate as f32).round() as usize;
    if frame_size == 0 || hop_size == 0 || samples.len() < frame_size {
        return Vec::new();
    }
    let n_frames = (samples.len() - frame_size) / hop_size + 1;
    let mut out = Vec::with_capacity(n_frames);
    let half_frame_seconds = frame_size as f64 / (2.0 * sample_rate as f64);
    for f in 0..n_frames {
        let start = f * hop_size;
        let frame = &samples[start..start + frame_size];
        let mean_sq: f32 = frame.iter().map(|x| x * x).sum::<f32>() / frame_size as f32;
        let rms = mean_sq.sqrt();
        let db_fs = if rms > 0.0 {
            20.0 * rms.log10()
        } else {
            DB_FS_FLOOR
        };
        out.push(IntensityFrame {
            time_seconds: start as f64 / sample_rate as f64 + half_frame_seconds,
            rms,
            db_fs,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    /// A pure sine at amplitude A has RMS = A / √2 = 0.7071 * A.
    /// A=1.0 → RMS ≈ 0.7071 → dB-FS ≈ -3.01 dB.
    #[test]
    fn rms_of_unit_sine_is_one_over_sqrt_two() {
        let sample_rate = 16_000u32;
        let freq = 440.0f32;
        let n = sample_rate as usize; // 1 second
        let samples: Vec<f32> = (0..n)
            .map(|i| (TAU * freq * (i as f32 / sample_rate as f32)).sin())
            .collect();
        let frames = intensity(&samples, sample_rate, 0.030, 0.010);
        assert!(!frames.is_empty());
        // Pick a frame from the steady-state middle of the signal.
        let mid = &frames[frames.len() / 2];
        let expected_rms = 1.0_f32 / std::f32::consts::SQRT_2;
        assert!(
            (mid.rms - expected_rms).abs() < 0.01,
            "got {} expected ~{}",
            mid.rms,
            expected_rms
        );
        // 20 * log10(1/√2) = -3.0103 dB.
        assert!(
            (mid.db_fs - (-3.0103)).abs() < 0.1,
            "got db_fs={}",
            mid.db_fs
        );
    }

    /// Silent input → RMS = 0; db_fs floored at DB_FS_FLOOR.
    #[test]
    fn silent_frame_floors_db_fs() {
        let sample_rate = 16_000u32;
        let samples = vec![0.0_f32; sample_rate as usize / 2];
        let frames = intensity(&samples, sample_rate, 0.030, 0.010);
        assert!(!frames.is_empty());
        for f in &frames {
            assert_eq!(f.rms, 0.0);
            assert_eq!(f.db_fs, DB_FS_FLOOR);
        }
    }

    #[test]
    fn time_seconds_advances_by_hop() {
        let sample_rate = 16_000u32;
        let samples = vec![0.5_f32; sample_rate as usize];
        let frame_size_secs = 0.030;
        let hop_secs = 0.010;
        let frames = intensity(&samples, sample_rate, frame_size_secs, hop_secs);
        assert!(frames.len() >= 2);
        let delta = frames[1].time_seconds - frames[0].time_seconds;
        assert!((delta - hop_secs as f64).abs() < 1e-6, "delta = {delta}");
    }

    #[test]
    fn input_shorter_than_frame_returns_empty() {
        let frames = intensity(&[0.5; 100], 16_000, 0.030, 0.010);
        assert!(frames.is_empty());
    }
}
