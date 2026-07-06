//! WAV loading and the in-memory `Audio` type. Format support today: PCM
//! 16/24/32-bit integer and 32-bit float; FLAC and other formats follow later.

use std::path::Path;

use crate::error::{EngineError, Result};

/// Audio data with interleaved samples normalized to f32 in `[-1.0, 1.0]`.
#[derive(Debug, Clone)]
pub struct Audio {
    /// Interleaved samples; for stereo the layout is `[L0, R0, L1, R1, ...]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz (samples per second per channel).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono, 2 = stereo, …).
    pub channels: u16,
}

impl Audio {
    /// Loads a WAV file. PCM 16/24/32-bit integer and 32-bit float are supported;
    /// samples are converted to f32 in `[-1.0, 1.0]` regardless of source depth.
    pub fn from_wav_path(path: impl AsRef<Path>) -> Result<Self> {
        let mut reader = hound::WavReader::open(path.as_ref())?;
        let spec = reader.spec();

        let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, 32) => reader
                .samples::<f32>()
                .collect::<std::result::Result<_, _>>()?,
            (hound::SampleFormat::Int, 16) => {
                let scale = 1.0 / i16::MAX as f32;
                reader
                    .samples::<i16>()
                    .map(|s| s.map(|v| v as f32 * scale))
                    .collect::<std::result::Result<_, _>>()?
            }
            (hound::SampleFormat::Int, 24) => {
                let scale = 1.0 / (1i32 << 23) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| v as f32 * scale))
                    .collect::<std::result::Result<_, _>>()?
            }
            (hound::SampleFormat::Int, 32) => {
                let scale = 1.0 / i32::MAX as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| v as f32 * scale))
                    .collect::<std::result::Result<_, _>>()?
            }
            (fmt, bits) => {
                return Err(EngineError::UnsupportedFormat(format!(
                    "{fmt:?} {bits}-bit"
                )));
            }
        };

        Ok(Audio {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        })
    }

    /// Construct an `Audio` from raw interleaved f32 samples (expected in
    /// `[-1.0, 1.0]`). For stereo the layout is `[L0, R0, L1, R1, ...]`.
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Audio {
            samples,
            sample_rate,
            channels,
        }
    }

    /// Number of frames (samples per channel).
    pub fn frame_count(&self) -> usize {
        self.samples.len() / self.channels as usize
    }

    /// Duration in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.frame_count() as f64 / self.sample_rate as f64
    }

    /// Iterator over a mono mixdown. For multi-channel audio, frames are averaged.
    pub fn mono_samples(&self) -> impl Iterator<Item = f32> + '_ {
        let channels = self.channels as usize;
        self.samples
            .chunks_exact(channels)
            .map(move |chunk| chunk.iter().sum::<f32>() / channels as f32)
    }

    /// Returns a new single-channel `Audio` whose samples are the mono mixdown
    /// (multi-channel frames averaged). Already-mono audio is copied unchanged.
    /// Unlike [`mono_samples`](Self::mono_samples), this yields a full `Audio`
    /// so the result can flow back into functions that take an `Audio`.
    pub fn to_mono(&self) -> Audio {
        Audio {
            samples: self.mono_samples().collect(),
            sample_rate: self.sample_rate,
            channels: 1,
        }
    }

    /// Reads only a WAV file's header to learn its size without decoding any
    /// samples — cheap regardless of file length. Used to decide, *before*
    /// committing to a full in-memory load, whether a file is large enough to
    /// warrant warning the user and offering to split it.
    pub fn probe(path: impl AsRef<Path>) -> Result<AudioProbe> {
        let reader = hound::WavReader::open(path.as_ref())?;
        let spec = reader.spec();
        // `duration()` is frames-per-channel, read from the data-chunk size in
        // the header — no samples are decoded.
        let n_frames = reader.duration() as u64;
        let channels = spec.channels;
        Ok(AudioProbe {
            sample_rate: spec.sample_rate,
            channels,
            n_frames,
            duration_seconds: n_frames as f64 / spec.sample_rate as f64,
            // What a full decode would cost in RAM: interleaved f32 samples.
            decoded_bytes: n_frames * channels as u64 * 4,
        })
    }
}

/// Cheap, header-only summary of a WAV file (see [`Audio::probe`]). Lets a
/// caller gauge the in-memory cost of a file before deciding to load it.
#[derive(Debug, Clone)]
pub struct AudioProbe {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Number of frames (samples per channel).
    pub n_frames: u64,
    /// Duration in seconds.
    pub duration_seconds: f64,
    /// Bytes a full decode would occupy (interleaved f32): `n_frames ×
    /// channels × 4`. The honest predictor of the load's RAM cost.
    pub decoded_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_sine_wav(
        path: &Path,
        sample_rate: u32,
        channels: u16,
        freq_hz: f32,
        duration_s: f32,
        bits_per_sample: u16,
        sample_format: hound::SampleFormat,
    ) {
        let spec = hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample,
            sample_format,
        };
        let mut writer = hound::WavWriter::create(path, spec).unwrap();
        let n_frames = (sample_rate as f32 * duration_s) as u32;
        for i in 0..n_frames {
            let t = i as f32 / sample_rate as f32;
            let s = 0.5 * (2.0 * std::f32::consts::PI * freq_hz * t).sin();
            for _ in 0..channels {
                match sample_format {
                    hound::SampleFormat::Float => writer.write_sample(s).unwrap(),
                    hound::SampleFormat::Int => {
                        let v = (s * i16::MAX as f32) as i16;
                        writer.write_sample(v).unwrap();
                    }
                }
            }
        }
        writer.finalize().unwrap();
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "sadda_engine_audio_test_{}_{}.wav",
            std::process::id(),
            name
        ));
        p
    }

    #[test]
    fn loads_mono_pcm16() {
        let path = tmp_path("mono_pcm16");
        write_sine_wav(&path, 16_000, 1, 440.0, 1.0, 16, hound::SampleFormat::Int);

        let audio = Audio::from_wav_path(&path).unwrap();

        assert_eq!(audio.sample_rate, 16_000);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.frame_count(), 16_000);
        assert!((audio.duration_seconds() - 1.0).abs() < 1e-6);

        let peak = audio.samples.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!((0.49..0.51).contains(&peak), "peak was {peak}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn loads_stereo_pcm16_and_mixes_down() {
        let path = tmp_path("stereo_pcm16");
        write_sine_wav(&path, 22_050, 2, 220.0, 0.25, 16, hound::SampleFormat::Int);

        let audio = Audio::from_wav_path(&path).unwrap();

        assert_eq!(audio.channels, 2);
        assert_eq!(audio.frame_count(), 22_050 / 4);
        assert_eq!(audio.samples.len(), audio.frame_count() * 2);

        let mono: Vec<f32> = audio.mono_samples().collect();
        assert_eq!(mono.len(), audio.frame_count());

        let down = audio.to_mono();
        assert_eq!(down.channels, 1);
        assert_eq!(down.sample_rate, audio.sample_rate);
        assert_eq!(down.frame_count(), audio.frame_count());
        assert_eq!(down.samples, mono);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn loads_float32() {
        let path = tmp_path("mono_float32");
        write_sine_wav(
            &path,
            48_000,
            1,
            1000.0,
            0.1,
            32,
            hound::SampleFormat::Float,
        );

        let audio = Audio::from_wav_path(&path).unwrap();

        assert_eq!(audio.sample_rate, 48_000);
        assert_eq!(audio.frame_count(), 4_800);
        let peak = audio.samples.iter().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!((0.49..0.51).contains(&peak), "peak was {peak}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_returns_an_error() {
        let path = tmp_path("does_not_exist");
        let err = Audio::from_wav_path(&path).unwrap_err();
        assert!(matches!(
            err,
            EngineError::Io(_) | EngineError::WavDecode(_)
        ));
    }
}
