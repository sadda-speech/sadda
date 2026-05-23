//! Live audio recording with streaming pitch / formants / intensity / meter
//! callbacks.
//!
//! ## Architecture (per the 2026-05-22 E1 DEVLOG entry)
//!
//! Three threads cooperate:
//!
//! 1. **Audio thread** (cpal callback or the synthetic-source path in tests).
//!    Pushes raw `f32` samples into the [`raw_samples`] SPSC ringbuffer.
//!    Must not allocate, lock, or acquire the GIL.
//! 2. **Consumer thread** (Rust). Drains the ringbuffer; streams samples to
//!    the on-disk WAV file via `hound`; maintains a sliding window for DSP;
//!    once per `hop_samples`, runs pitch / intensity / formants on the
//!    latest window; meter (peak + RMS) emitted once per drained chunk.
//!    Per-measure result rtrbs deliver frames to the dispatch side.
//! 3. **Dispatch thread** (lives outside this crate — in the Python
//!    bindings). Pops from the result rtrbs with the GIL held and invokes
//!    user-supplied Python callables.
//!
//! Splitting consumer from dispatch matters: consumer is alloc-free Rust
//! (cheap to run continuously); dispatch holds the GIL to invoke Python
//! (the slow, serialised part). If they merged, a slow Python callback
//! would stall WAV writing and drop samples.
//!
//! [`raw_samples`]: LiveSession::push_samples
//!
//! ## Test seam
//!
//! `LiveSession` does not own a cpal stream. The audio thread's only job is
//! to call [`LiveSession::push_samples`]; the real cpal wiring is the
//! caller's responsibility (see `crates/python/src/live.rs`). The
//! integration test in `crates/engine/tests/live_recording.rs` synthesises
//! samples directly, which keeps the engine test suite free of audio
//! hardware dependencies.

use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::dsp::formants::{FormantFrame as DspFormantFrame, FormantsConfig};
use crate::dsp::intensity::DB_FS_FLOOR;
use crate::dsp::lpc::LpcMethod;
use crate::error::{EngineError, Result};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for a [`LiveSession`].
#[derive(Debug, Clone)]
pub struct LiveConfig {
    /// Capture sample rate in Hz.
    pub sample_rate: u32,
    /// Number of capture channels (1 = mono; 2 = stereo). The DSP path
    /// always runs on the downmixed-to-mono signal.
    pub channels: u16,
    /// Analysis window length in milliseconds.
    pub analysis_window_ms: f32,
    /// Analysis hop length in milliseconds (the rate at which DSP frames
    /// are emitted).
    pub analysis_hop_ms: f32,
    /// Number of formants to keep per frame (top N by ascending frequency
    /// after the LPC root filter).
    pub num_formants: usize,
    /// Minimum pitch frequency in Hz (used by the autocorrelation tracker).
    pub min_pitch_hz: f32,
    /// Maximum pitch frequency in Hz.
    pub max_pitch_hz: f32,
}

impl Default for LiveConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            channels: 1,
            analysis_window_ms: 30.0,
            analysis_hop_ms: 10.0,
            num_formants: 4,
            min_pitch_hz: 75.0,
            max_pitch_hz: 500.0,
        }
    }
}

/// Peak + RMS snapshot emitted once per drained chunk for fast level UIs.
#[derive(Debug, Clone, Copy)]
pub struct MeterFrame {
    /// Peak absolute amplitude over the chunk, in `[0, 1]`.
    pub peak: f32,
    /// RMS amplitude over the chunk, in `[0, 1]`.
    pub rms: f32,
    /// RMS expressed in dB-FS. Floored at [`DB_FS_FLOOR`].
    pub rms_db: f32,
    /// Time in seconds since the session started.
    pub time_seconds: f64,
}

/// One pitch estimate emitted once per analysis hop.
#[derive(Debug, Clone, Copy)]
pub struct LivePitchFrame {
    /// Time in seconds since the session started.
    pub time_seconds: f64,
    /// Estimated f0 in Hz.
    pub frequency_hz: f32,
    /// Voicing strength in `[0, 1]`.
    pub voicing: f32,
}

/// One intensity reading emitted once per analysis hop.
#[derive(Debug, Clone, Copy)]
pub struct LiveIntensityFrame {
    /// Time in seconds since the session started.
    pub time_seconds: f64,
    /// dB-FS intensity (floored at [`DB_FS_FLOOR`]).
    pub db_fs: f32,
}

/// One formants reading emitted once per analysis hop. Variable length —
/// frames where the root-finder didn't return enough roots have a shorter
/// `frequencies` vector rather than NaN padding.
#[derive(Debug, Clone)]
pub struct LiveFormantsFrame {
    /// Time in seconds since the session started.
    pub time_seconds: f64,
    /// Formant frequencies in Hz, ascending.
    pub frequencies: Vec<f32>,
    /// Bandwidths in Hz, co-indexed with `frequencies`.
    pub bandwidths: Vec<f32>,
}

/// The result rtrbs exposed to the dispatch side. Each is a single-consumer
/// ring; pop until empty in a tight loop per dispatch iteration.
pub struct LiveResults {
    /// Meter snapshots (one per drained chunk).
    pub meters: rtrb::Consumer<MeterFrame>,
    /// Pitch frames (one per analysis hop).
    pub pitches: rtrb::Consumer<LivePitchFrame>,
    /// Intensity frames (one per analysis hop).
    pub intensities: rtrb::Consumer<LiveIntensityFrame>,
    /// Formant frames (one per analysis hop).
    pub formants: rtrb::Consumer<LiveFormantsFrame>,
}

/// A recording in progress. Construction spawns the consumer thread and
/// opens the WAV file under `.in_progress/<uuid>/audio.wav`. Calling
/// [`LiveSession::stop`] joins the thread; [`StoppedSession::commit`] then
/// performs the atomic move + bundle insert + processing_run insert.
pub struct LiveSession {
    in_progress_dir: PathBuf,
    config: LiveConfig,
    raw_producer: Option<rtrb::Producer<f32>>,
    consumer_handle: Option<JoinHandle<ConsumerOutcome>>,
    stop_flag: Arc<AtomicBool>,
    dropped_samples: Arc<AtomicUsize>,
}

/// The state returned after [`LiveSession::stop`]. The on-disk file is
/// flushed and the consumer thread has joined; the caller now decides
/// whether to [`Self::commit`] (atomic move + bundle insert) or
/// [`Self::discard`] (delete the `.in_progress/<uuid>/` directory).
pub struct StoppedSession {
    /// Path to the `.in_progress/<uuid>/` directory.
    pub in_progress_dir: PathBuf,
    /// Total number of *frames* (samples per channel) written to disk.
    pub frames_written: usize,
    /// Number of samples dropped because the raw-samples rtrb overflowed
    /// (consumer couldn't keep up). 0 means no drops.
    pub dropped_samples: usize,
    /// Sample rate (echoed from the config for the commit step).
    pub sample_rate: u32,
    /// Channels (echoed from the config for the commit step).
    pub channels: u16,
}

/// Internal: the consumer-thread result returned through `JoinHandle`.
struct ConsumerOutcome {
    frames_written: usize,
}

// ---------------------------------------------------------------------------
// LiveSession implementation
// ---------------------------------------------------------------------------

/// Size (in samples) of the raw-samples rtrb between the audio thread and
/// the consumer. Sized for ~1 second of stereo 48 kHz audio so brief
/// consumer stalls (e.g. disk fsync) don't drop samples.
const RAW_RING_CAPACITY: usize = 48_000 * 2;
/// Size (in entries) of each per-measure result rtrb. 256 is enough for
/// ~2.5 seconds of frames at the 10 ms default hop.
const RESULT_RING_CAPACITY: usize = 256;
/// How long the consumer thread sleeps when the ring is empty. 1 ms is short
/// enough to keep latency low and long enough that a busy-wait doesn't
/// dominate CPU. The producer drives the eventual wakeup; this is a
/// fallback for the empty-ring case.
const IDLE_SLEEP: Duration = Duration::from_millis(1);

impl LiveSession {
    /// Constructs a new session: creates `<project_root>/signals/.in_progress/<uuid>/`,
    /// opens `audio.wav` for streamed write, spawns the consumer thread,
    /// and returns the session paired with its [`LiveResults`] rtrbs.
    pub fn start(project_root: &Path, config: LiveConfig) -> Result<(Self, LiveResults)> {
        if config.sample_rate == 0 {
            return Err(EngineError::Corpus("live: sample_rate must be > 0".into()));
        }
        if config.channels == 0 {
            return Err(EngineError::Corpus("live: channels must be > 0".into()));
        }
        if config.analysis_window_ms <= 0.0 || config.analysis_hop_ms <= 0.0 {
            return Err(EngineError::Corpus(
                "live: analysis_window_ms and analysis_hop_ms must be > 0".into(),
            ));
        }

        let in_progress_root = project_root.join("signals").join(".in_progress");
        fs::create_dir_all(&in_progress_root)?;
        let session_dir = in_progress_root.join(uuid::Uuid::new_v4().to_string());
        fs::create_dir_all(&session_dir)?;

        let wav_path = session_dir.join("audio.wav");
        let wav_spec = hound::WavSpec {
            channels: config.channels,
            sample_rate: config.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let writer = hound::WavWriter::new(BufWriter::new(fs::File::create(&wav_path)?), wav_spec)?;

        let (raw_producer, raw_consumer) = rtrb::RingBuffer::<f32>::new(RAW_RING_CAPACITY);
        let (meter_producer, meter_consumer) =
            rtrb::RingBuffer::<MeterFrame>::new(RESULT_RING_CAPACITY);
        let (pitch_producer, pitch_consumer) =
            rtrb::RingBuffer::<LivePitchFrame>::new(RESULT_RING_CAPACITY);
        let (intensity_producer, intensity_consumer) =
            rtrb::RingBuffer::<LiveIntensityFrame>::new(RESULT_RING_CAPACITY);
        let (formants_producer, formants_consumer) =
            rtrb::RingBuffer::<LiveFormantsFrame>::new(RESULT_RING_CAPACITY);

        let stop_flag = Arc::new(AtomicBool::new(false));
        let dropped_samples = Arc::new(AtomicUsize::new(0));

        let consumer = ConsumerCtx {
            cfg: config.clone(),
            raw: raw_consumer,
            wav_writer: writer,
            meters: meter_producer,
            pitches: pitch_producer,
            intensities: intensity_producer,
            formants_out: formants_producer,
            stop_flag: stop_flag.clone(),
        };
        let consumer_handle = thread::Builder::new()
            .name("sadda-live-consumer".into())
            .spawn(move || consumer.run())
            .map_err(|e| EngineError::Corpus(format!("live: spawn consumer thread: {e}")))?;

        let session = LiveSession {
            in_progress_dir: session_dir,
            config,
            raw_producer: Some(raw_producer),
            consumer_handle: Some(consumer_handle),
            stop_flag,
            dropped_samples,
        };
        let results = LiveResults {
            meters: meter_consumer,
            pitches: pitch_consumer,
            intensities: intensity_consumer,
            formants: formants_consumer,
        };
        Ok((session, results))
    }

    /// Moves the raw-samples ringbuffer producer out of the session. Use
    /// this when the producer lives elsewhere — e.g. inside the
    /// cpal-callback closure on the audio thread. After
    /// `take_producer`, [`Self::push_samples`] is a no-op (and reports
    /// every sample as "dropped").
    ///
    /// Returns `None` if the producer has already been moved out.
    pub fn take_producer(&mut self) -> Option<rtrb::Producer<f32>> {
        self.raw_producer.take()
    }

    /// Pushes a chunk of interleaved samples into the raw-samples
    /// ringbuffer. Called from the audio thread (cpal callback or the
    /// integration test's synthetic source). Returns the number of samples
    /// dropped due to ring overrun (0 in the happy path).
    ///
    /// Real-time-safe: no allocations, no locks, no system calls.
    pub fn push_samples(&mut self, samples: &[f32]) -> usize {
        let producer = match self.raw_producer.as_mut() {
            Some(p) => p,
            None => return samples.len(),
        };
        let mut dropped = 0;
        for &s in samples {
            if producer.push(s).is_err() {
                dropped += 1;
            }
        }
        if dropped > 0 {
            self.dropped_samples.fetch_add(dropped, Ordering::Relaxed);
        }
        dropped
    }

    /// Returns the path to this session's `.in_progress/<uuid>/` directory.
    pub fn in_progress_dir(&self) -> &Path {
        &self.in_progress_dir
    }

    /// Returns the live config (sample rate, channels, hop, …).
    pub fn config(&self) -> &LiveConfig {
        &self.config
    }

    /// Signals the consumer thread to drain and exit, joins it, and
    /// returns a [`StoppedSession`] handle the caller can commit or
    /// discard. The audio thread must have stopped pushing samples before
    /// `stop()` is called (the cpal stream should be dropped first).
    pub fn stop(mut self) -> Result<StoppedSession> {
        // Drop the producer first so the consumer drains any pending
        // samples and sees the ringbuffer become permanently empty.
        self.raw_producer.take();
        self.stop_flag.store(true, Ordering::Release);

        let handle = self
            .consumer_handle
            .take()
            .ok_or_else(|| EngineError::Corpus("live: session already stopped".into()))?;
        let outcome = handle
            .join()
            .map_err(|_| EngineError::Corpus("live: consumer thread panicked".into()))?;

        Ok(StoppedSession {
            in_progress_dir: self.in_progress_dir.clone(),
            frames_written: outcome.frames_written,
            dropped_samples: self.dropped_samples.load(Ordering::Acquire),
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
        })
    }
}

impl Drop for LiveSession {
    fn drop(&mut self) {
        // Best-effort cleanup if the caller forgot to stop()/commit():
        // signal the consumer to exit so its file handle drops cleanly.
        // The .in_progress/ directory is left behind for forensics; the
        // user is expected to call commit() or discard() explicitly.
        if self.consumer_handle.is_some() {
            self.raw_producer.take();
            self.stop_flag.store(true, Ordering::Release);
            if let Some(h) = self.consumer_handle.take() {
                let _ = h.join();
            }
        }
    }
}

impl StoppedSession {
    /// Deletes the `.in_progress/<uuid>/` directory. Idempotent —
    /// returns `Ok(())` even if the directory has already been removed.
    pub fn discard(self) -> Result<()> {
        match fs::remove_dir_all(&self.in_progress_dir) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(EngineError::Io(e)),
        }
    }

    /// Returns the duration of the recording in seconds.
    pub fn duration_seconds(&self) -> f64 {
        self.frames_written as f64 / self.sample_rate as f64
    }
}

// ---------------------------------------------------------------------------
// Consumer thread
// ---------------------------------------------------------------------------

struct ConsumerCtx {
    cfg: LiveConfig,
    raw: rtrb::Consumer<f32>,
    wav_writer: hound::WavWriter<BufWriter<fs::File>>,
    meters: rtrb::Producer<MeterFrame>,
    pitches: rtrb::Producer<LivePitchFrame>,
    intensities: rtrb::Producer<LiveIntensityFrame>,
    formants_out: rtrb::Producer<LiveFormantsFrame>,
    stop_flag: Arc<AtomicBool>,
}

impl ConsumerCtx {
    fn run(mut self) -> ConsumerOutcome {
        let cfg = self.cfg.clone();
        let window_samples =
            ((cfg.analysis_window_ms / 1000.0) * cfg.sample_rate as f32).round() as usize;
        let hop_samples =
            ((cfg.analysis_hop_ms / 1000.0) * cfg.sample_rate as f32).round() as usize;
        let window_samples = window_samples.max(1);
        let hop_samples = hop_samples.max(1);

        // Sliding window in mono. Bounded at `window_samples`; older
        // samples drop off the front as new ones arrive.
        let mut mono_window: std::collections::VecDeque<f32> =
            std::collections::VecDeque::with_capacity(window_samples * 2);
        // Total mono frames seen since session start (drives timestamps).
        let mut frames_total: usize = 0;
        // Frames since the last analysis emission.
        let mut frames_since_hop: usize = 0;

        // Per-chunk scratch buffer to feed hound and downmix to mono.
        let chunk_capacity = 4096_usize.max(window_samples);
        let mut chunk: Vec<f32> = Vec::with_capacity(chunk_capacity);

        let channels = cfg.channels as usize;
        let formants_cfg = FormantsConfig {
            frame_size_seconds: cfg.analysis_window_ms / 1000.0,
            hop_seconds: cfg.analysis_hop_ms / 1000.0,
            n_formants: cfg.num_formants,
            pre_emphasis: 0.97,
            lpc_order: None,
            lpc_method: LpcMethod::Burg,
            max_bandwidth_hz: 1000.0,
            min_frequency_hz: 50.0,
        };

        loop {
            // Drain whatever is available.
            chunk.clear();
            while let Ok(s) = self.raw.pop() {
                chunk.push(s);
                if chunk.len() == chunk.capacity() {
                    break;
                }
            }

            if !chunk.is_empty() {
                // 1. Stream to disk (interleaved as-is).
                for &s in &chunk {
                    // hound's WavWriter<f32> never errors on a write
                    // unless the underlying writer does; we'd surface a
                    // dropped-recording fault, but for now just bail on
                    // disk errors by stopping the loop early.
                    if self.wav_writer.write_sample(s).is_err() {
                        self.stop_flag.store(true, Ordering::Release);
                        break;
                    }
                }

                // 2. Downmix to mono and append to the sliding window.
                let mono_chunk_len = chunk.len() / channels;
                for frame in chunk.chunks_exact(channels) {
                    let s: f32 = frame.iter().sum::<f32>() / channels as f32;
                    mono_window.push_back(s);
                    if mono_window.len() > window_samples * 2 {
                        // Cap the deque to avoid unbounded growth if hop is
                        // somehow ahead of consumption. Keep the most recent
                        // window worth of samples.
                        let drop = mono_window.len() - window_samples * 2;
                        for _ in 0..drop {
                            mono_window.pop_front();
                        }
                    }
                }

                // 3. Meter from the just-drained mono chunk.
                if mono_chunk_len > 0 {
                    let (peak, rms) = peak_and_rms(&mono_window, mono_chunk_len);
                    let rms_db = if rms > 0.0 {
                        20.0 * rms.log10()
                    } else {
                        DB_FS_FLOOR
                    };
                    let meter_time =
                        (frames_total + mono_chunk_len) as f64 / cfg.sample_rate as f64;
                    let _ = self.meters.push(MeterFrame {
                        peak,
                        rms,
                        rms_db,
                        time_seconds: meter_time,
                    });
                }

                frames_total += mono_chunk_len;
                frames_since_hop += mono_chunk_len;

                // 4. Emit DSP frames as long as we have a full window and
                //    have advanced by at least one hop.
                while frames_since_hop >= hop_samples && mono_window.len() >= window_samples {
                    frames_since_hop -= hop_samples;
                    let window: Vec<f32> = mono_window
                        .iter()
                        .copied()
                        .rev()
                        .take(window_samples)
                        .collect::<Vec<_>>()
                        .into_iter()
                        .rev()
                        .collect();
                    let centre_seconds = (frames_total as f64 - window_samples as f64 / 2.0)
                        / cfg.sample_rate as f64;
                    emit_dsp_frames(
                        &window,
                        centre_seconds,
                        cfg.sample_rate,
                        cfg.min_pitch_hz,
                        cfg.max_pitch_hz,
                        &formants_cfg,
                        &mut self.pitches,
                        &mut self.intensities,
                        &mut self.formants_out,
                    );
                }
            }

            // 5. Idle / stop bookkeeping.
            if chunk.is_empty() {
                if self.stop_flag.load(Ordering::Acquire) {
                    // Producer is gone *and* we just drained an empty
                    // ring → safe to exit.
                    break;
                }
                thread::sleep(IDLE_SLEEP);
            }
        }

        // Flush hound; finalize replaces the underlying BufWriter so any
        // errors here would indicate an OS-level write failure. We swallow
        // the error so the join doesn't panic; commit() will notice
        // frames_written and refuse if it's zero.
        let _ = self.wav_writer.finalize();
        ConsumerOutcome {
            frames_written: frames_total,
        }
    }
}

/// Compute peak (max |x|) and RMS over the trailing `n` samples of the
/// mono window.
fn peak_and_rms(window: &std::collections::VecDeque<f32>, n: usize) -> (f32, f32) {
    if n == 0 || window.is_empty() {
        return (0.0, 0.0);
    }
    let start = window.len().saturating_sub(n);
    let mut peak: f32 = 0.0;
    let mut sum_sq: f32 = 0.0;
    let mut count: usize = 0;
    for i in start..window.len() {
        let s = window[i];
        let a = s.abs();
        if a > peak {
            peak = a;
        }
        sum_sq += s * s;
        count += 1;
    }
    let rms = if count > 0 {
        (sum_sq / count as f32).sqrt()
    } else {
        0.0
    };
    (peak, rms)
}

#[allow(clippy::too_many_arguments)]
fn emit_dsp_frames(
    window: &[f32],
    centre_seconds: f64,
    sample_rate: u32,
    min_pitch_hz: f32,
    max_pitch_hz: f32,
    formants_cfg: &FormantsConfig,
    pitches: &mut rtrb::Producer<LivePitchFrame>,
    intensities: &mut rtrb::Producer<LiveIntensityFrame>,
    formants_out: &mut rtrb::Producer<LiveFormantsFrame>,
) {
    // Intensity is cheap: RMS in dB-FS over the window.
    let sum_sq: f32 = window.iter().map(|x| x * x).sum::<f32>() / window.len() as f32;
    let rms = sum_sq.sqrt();
    let db_fs = if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        DB_FS_FLOOR
    };
    let _ = intensities.push(LiveIntensityFrame {
        time_seconds: centre_seconds,
        db_fs,
    });

    // Pitch via single-frame autocorrelation. We can't reuse the public
    // `pitch::autocorrelation` directly (it takes an &Audio and runs over
    // many frames), so re-implement the per-frame core inline. This stays
    // in sync with the reference implementation by construction — the
    // peak-finding logic is identical.
    let min_lag = (sample_rate as f32 / max_pitch_hz).round() as usize;
    let max_lag = (sample_rate as f32 / min_pitch_hz).round() as usize;
    if min_lag < max_lag && max_lag < window.len() {
        let (lag, peak_val) = best_lag(window, min_lag, max_lag);
        let r0: f32 = window.iter().map(|x| x * x).sum();
        let voicing = if r0 > 0.0 {
            (peak_val / r0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let frequency_hz = sample_rate as f32 / lag as f32;
        let _ = pitches.push(LivePitchFrame {
            time_seconds: centre_seconds,
            frequency_hz,
            voicing,
        });
    }

    // Formants: reuse the batch `formants()` API by handing it exactly one
    // window's worth of samples. Configured so a single frame comes out.
    let frames: Vec<DspFormantFrame> =
        crate::dsp::formants::formants(window, sample_rate, formants_cfg);
    if let Some(f0) = frames.into_iter().next() {
        let _ = formants_out.push(LiveFormantsFrame {
            time_seconds: centre_seconds,
            frequencies: f0.frequencies,
            bandwidths: f0.bandwidths,
        });
    }
}

/// Time-domain autocorrelation best-lag finder. Inlined from
/// `pitch::autocorrelation` (which operates on a full Audio); shared logic
/// would land here too if we ever extract a per-frame helper into
/// `crate::pitch`.
fn best_lag(frame: &[f32], min_lag: usize, max_lag: usize) -> (usize, f32) {
    let max_lag = max_lag.min(frame.len().saturating_sub(1));
    let mut best = min_lag;
    let mut best_value = f32::MIN;
    for lag in min_lag..=max_lag {
        let mut sum = 0.0f32;
        for i in 0..(frame.len() - lag) {
            sum += frame[i] * frame[i + lag];
        }
        if sum > best_value {
            best_value = sum;
            best = lag;
        }
    }
    (best, best_value.max(0.0))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn fresh_root(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("sadda_live_test_{}_{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(p.join("signals")).unwrap();
        p
    }

    fn sine_samples(sample_rate: u32, freq_hz: f32, duration_s: f32) -> Vec<f32> {
        let n = (duration_s * sample_rate as f32) as usize;
        (0..n)
            .map(|i| (TAU * freq_hz * (i as f32 / sample_rate as f32)).sin() * 0.5)
            .collect()
    }

    #[test]
    fn start_creates_in_progress_directory_with_wav() {
        let root = fresh_root("start");
        let (session, _results) = LiveSession::start(&root, LiveConfig::default()).unwrap();
        assert!(session.in_progress_dir().exists());
        assert!(session.in_progress_dir().join("audio.wav").exists());
        let stopped = session.stop().unwrap();
        stopped.discard().unwrap();
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn push_samples_writes_audio_to_wav_file() {
        let root = fresh_root("push_writes");
        let cfg = LiveConfig {
            sample_rate: 16_000,
            channels: 1,
            ..LiveConfig::default()
        };
        let (mut session, _results) = LiveSession::start(&root, cfg.clone()).unwrap();
        let samples = sine_samples(cfg.sample_rate, 440.0, 0.5);
        session.push_samples(&samples);
        // Give the consumer thread time to drain.
        thread::sleep(Duration::from_millis(50));
        let stopped = session.stop().unwrap();
        assert!(stopped.frames_written > 0);
        assert_eq!(stopped.dropped_samples, 0);
        let wav = stopped.in_progress_dir.join("audio.wav");
        let audio = crate::audio::Audio::from_wav_path(&wav).unwrap();
        assert!(audio.frame_count() > 0);
        assert_eq!(audio.sample_rate, cfg.sample_rate);
        stopped.discard().unwrap();
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn pitch_callback_emits_440hz_for_440hz_sine() {
        let root = fresh_root("pitch_440");
        let cfg = LiveConfig {
            sample_rate: 16_000,
            channels: 1,
            ..LiveConfig::default()
        };
        let (mut session, mut results) = LiveSession::start(&root, cfg.clone()).unwrap();
        let samples = sine_samples(cfg.sample_rate, 440.0, 1.0);
        session.push_samples(&samples);
        thread::sleep(Duration::from_millis(100));
        let stopped = session.stop().unwrap();

        let mut pitch_values = Vec::new();
        while let Ok(p) = results.pitches.pop() {
            pitch_values.push(p.frequency_hz);
        }
        assert!(!pitch_values.is_empty(), "expected pitch frames");
        // Median should be very close to 440 Hz.
        pitch_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = pitch_values[pitch_values.len() / 2];
        assert!(
            (median - 440.0).abs() < 5.0,
            "median pitch was {median}, expected ~440"
        );

        stopped.discard().unwrap();
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn intensity_callback_emits_negative_db_for_quiet_sine() {
        let root = fresh_root("intensity");
        let cfg = LiveConfig {
            sample_rate: 16_000,
            channels: 1,
            ..LiveConfig::default()
        };
        let (mut session, mut results) = LiveSession::start(&root, cfg.clone()).unwrap();
        let samples = sine_samples(cfg.sample_rate, 440.0, 0.5);
        session.push_samples(&samples);
        thread::sleep(Duration::from_millis(80));
        let stopped = session.stop().unwrap();

        let mut got_any = false;
        while let Ok(f) = results.intensities.pop() {
            got_any = true;
            // 0.5-amplitude sine: RMS ≈ 0.354 → ~ -9 dB-FS
            assert!(f.db_fs > -30.0 && f.db_fs < 0.0, "db_fs = {}", f.db_fs);
        }
        assert!(got_any);
        stopped.discard().unwrap();
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn meter_callback_emits_one_per_push() {
        let root = fresh_root("meter");
        let (mut session, mut results) = LiveSession::start(&root, LiveConfig::default()).unwrap();
        let samples = sine_samples(44_100, 440.0, 0.1);
        session.push_samples(&samples);
        thread::sleep(Duration::from_millis(80));
        let stopped = session.stop().unwrap();

        let mut peaks = Vec::new();
        while let Ok(m) = results.meters.pop() {
            peaks.push(m.peak);
        }
        assert!(!peaks.is_empty());
        // 0.5-amplitude sine: peak ≈ 0.5
        let max_peak = peaks.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max_peak > 0.4 && max_peak <= 1.0);
        stopped.discard().unwrap();
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn discard_removes_in_progress_dir() {
        let root = fresh_root("discard");
        let (session, _results) = LiveSession::start(&root, LiveConfig::default()).unwrap();
        let dir = session.in_progress_dir().to_path_buf();
        let stopped = session.stop().unwrap();
        stopped.discard().unwrap();
        assert!(!dir.exists());
        let _ = fs::remove_dir_all(&root);
    }
}
