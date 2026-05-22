//! Integration tests for the E1 live-recording slice. cpal is *not*
//! exercised here — we feed synthetic samples directly into
//! `LiveSession::push_samples`. CI / headless environments don't have an
//! audio device, and the cpal wiring lives in `crates/python`.

use std::f32::consts::TAU;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use sadda_engine::{LiveConfig, LiveSession, Project};

fn unique_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "sadda_engine_e1_test_{}_{}",
        std::process::id(),
        name
    ));
    p
}

fn sine(sample_rate: u32, freq_hz: f32, duration_s: f32, amplitude: f32) -> Vec<f32> {
    let n = (duration_s * sample_rate as f32) as usize;
    (0..n)
        .map(|i| (TAU * freq_hz * (i as f32 / sample_rate as f32)).sin() * amplitude)
        .collect()
}

fn new_project(root: &Path) -> Project {
    let _ = std::fs::remove_dir_all(root);
    Project::create(root, "e1_test").unwrap()
}

#[test]
fn commit_creates_bundle_and_processing_run() {
    let root = unique_dir("commit_basic");
    let proj = new_project(&root);
    let cfg = LiveConfig {
        sample_rate: 16_000,
        channels: 1,
        ..LiveConfig::default()
    };
    let (mut session, _results) = LiveSession::start(&root, cfg.clone()).unwrap();
    let samples = sine(cfg.sample_rate, 440.0, 0.5, 0.5);
    session.push_samples(&samples);
    thread::sleep(Duration::from_millis(80));
    let stopped = session.stop().unwrap();

    let params = format!(
        "{{\"sample_rate\":{},\"channels\":{},\"duration_s\":{}}}",
        cfg.sample_rate,
        cfg.channels,
        stopped.duration_seconds()
    );
    let bundle_id = proj
        .commit_recording(stopped, "practice_take_1", &params)
        .unwrap();

    // Bundle row exists with the right shape.
    let bundles = proj.bundles().unwrap();
    assert_eq!(bundles.len(), 1);
    assert_eq!(bundles[0].id, bundle_id);
    assert_eq!(bundles[0].name, "practice_take_1");
    assert_eq!(bundles[0].sample_rate, 16_000);
    assert_eq!(bundles[0].channels, 1);
    assert!(bundles[0].n_frames > 0);

    // WAV landed at signals/original/<name>.wav.
    let wav = root.join("signals").join("original").join("practice_take_1.wav");
    assert!(wav.exists(), "expected wav at {}", wav.display());

    // processing_run audited as live_recording.
    let conn = rusqlite::Connection::open(root.join("corpus.db")).unwrap();
    let (kind, processor, status): (String, String, String) = conn
        .query_row(
            "SELECT kind, processor_id, status FROM processing_run \
             WHERE bundle_id = ?1 ORDER BY id DESC LIMIT 1",
            [bundle_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(kind, "live_recording");
    assert_eq!(processor, "sadda.live");
    assert_eq!(status, "ok");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn pitch_subscriber_observes_440hz_for_440hz_sine() {
    let root = unique_dir("pitch_440");
    let _proj = new_project(&root);
    let cfg = LiveConfig {
        sample_rate: 16_000,
        channels: 1,
        ..LiveConfig::default()
    };
    let (mut session, mut results) = LiveSession::start(&root, cfg.clone()).unwrap();
    let samples = sine(cfg.sample_rate, 440.0, 1.0, 0.5);
    session.push_samples(&samples);
    thread::sleep(Duration::from_millis(150));
    let stopped = session.stop().unwrap();

    let mut pitches: Vec<f32> = Vec::new();
    while let Ok(p) = results.pitches.pop() {
        pitches.push(p.frequency_hz);
    }
    assert!(pitches.len() >= 10, "got {} pitch frames", pitches.len());
    pitches.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = pitches[pitches.len() / 2];
    assert!(
        (median - 440.0).abs() < 5.0,
        "median pitch {median} Hz, expected ~440"
    );
    stopped.discard().unwrap();
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn discard_after_stop_leaves_no_bundle_or_processing_run() {
    let root = unique_dir("discard");
    let proj = new_project(&root);
    let (mut session, _results) =
        LiveSession::start(&root, LiveConfig::default()).unwrap();
    session.push_samples(&sine(44_100, 440.0, 0.05, 0.5));
    thread::sleep(Duration::from_millis(60));
    let stopped = session.stop().unwrap();
    let in_progress = stopped.in_progress_dir.clone();
    stopped.discard().unwrap();
    assert!(!in_progress.exists());
    assert_eq!(proj.bundles().unwrap().len(), 0);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn commit_with_zero_frames_errors() {
    let root = unique_dir("commit_zero");
    let proj = new_project(&root);
    let (session, _results) = LiveSession::start(&root, LiveConfig::default()).unwrap();
    let stopped = session.stop().unwrap();
    // Nothing was pushed.
    let err = proj
        .commit_recording(stopped, "empty", "{}")
        .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("frames_written = 0"),
        "unexpected error: {msg}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn commit_refuses_duplicate_destination() {
    let root = unique_dir("commit_dup");
    let proj = new_project(&root);
    // First recording.
    let (mut s1, _r1) = LiveSession::start(&root, LiveConfig::default()).unwrap();
    s1.push_samples(&sine(44_100, 440.0, 0.05, 0.5));
    thread::sleep(Duration::from_millis(60));
    let stopped1 = s1.stop().unwrap();
    proj.commit_recording(stopped1, "take", "{}").unwrap();

    // Second recording with the same name should error and leave the
    // .in_progress dir intact for retry.
    let (mut s2, _r2) = LiveSession::start(&root, LiveConfig::default()).unwrap();
    s2.push_samples(&sine(44_100, 440.0, 0.05, 0.5));
    thread::sleep(Duration::from_millis(60));
    let stopped2 = s2.stop().unwrap();
    let in_progress_dir = stopped2.in_progress_dir.clone();
    let err = proj
        .commit_recording(stopped2, "take", "{}")
        .unwrap_err();
    assert!(format!("{err}").contains("destination already exists"));
    // .in_progress still has the WAV (we didn't even attempt the rename).
    assert!(in_progress_dir.join("audio.wav").exists());

    let _ = std::fs::remove_dir_all(&root);
}
