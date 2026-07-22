//! YIN + pYIN validation: median voiced f0 per fixture within tolerance
//! of librosa 0.11's `yin` and `pyin` over the same WAVs.
//!
//! The golden TSV at `tests/clinical/fixtures/pitch_yin_pyin_golden.tsv`
//! is produced by `tests/dsp/librosa/pitch_yin_pyin.py` (committed so CI
//! never needs librosa). Both sides use the same defaults: 30 ms frame,
//! 10 ms hop, fmin = 75 Hz, fmax = 500 Hz, trough_threshold = 0.1 for
//! YIN; n_thresholds = 100 for pYIN.

use std::path::PathBuf;

use sadda_engine::Audio;
use sadda_engine::pitch::{PitchConfig, PitchMethod, pitch};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/clinical/fixtures")
}

fn load_wav(stem: &str) -> Audio {
    let path = fixtures_dir().join(format!("{stem}.wav"));
    Audio::from_wav_path(&path).unwrap_or_else(|e| panic!("load {stem}.wav: {e}"))
}

fn median_voiced_f0(frames: &[sadda_engine::PitchFrame], voicing_threshold: f32) -> Option<f32> {
    let mut voiced: Vec<f32> = frames
        .iter()
        .filter(|f| f.voicing >= voicing_threshold)
        .map(|f| f.frequency_hz.value())
        .collect();
    if voiced.is_empty() {
        return None;
    }
    voiced.sort_by(|a, b| a.total_cmp(b));
    Some(voiced[voiced.len() / 2])
}

#[derive(Debug)]
struct GoldenRow {
    signal: String,
    method: String,
    n_voiced_frames: usize,
    median_f0_hz: f32,
}

fn read_golden() -> Vec<GoldenRow> {
    let path = fixtures_dir().join("pitch_yin_pyin_golden.tsv");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 4, "golden tsv row {i}: {line}");
        rows.push(GoldenRow {
            signal: cols[0].to_string(),
            method: cols[1].to_string(),
            n_voiced_frames: cols[2].parse().unwrap(),
            median_f0_hz: cols[3].parse().unwrap(),
        });
    }
    rows
}

#[test]
fn yin_median_f0_matches_librosa_within_tolerance() {
    let golden = read_golden();
    // Pin the ceiling to the librosa golden's 500 Hz (see the module doc) so
    // this comparison stays valid independent of the engine default, which is
    // 600 Hz to match Praat's `To Pitch (ac)`.
    let cfg = PitchConfig {
        max_freq_hz: 500.0,
        ..PitchConfig::default()
    };
    let mut checked = 0;
    for row in &golden {
        if row.method != "yin" {
            continue;
        }
        let audio = load_wav(&row.signal);
        let frames = pitch(&audio, &cfg, PitchMethod::Yin);
        let our_median = median_voiced_f0(&frames, cfg.voicing_threshold).unwrap_or_else(|| {
            panic!(
                "yin found no voiced frames for {} (librosa: {} voiced)",
                row.signal, row.n_voiced_frames,
            )
        });
        // 1.5 Hz for clean / HNR fixtures, 3 Hz for jitter / shimmer
        // (which inject ±3% period jitter so the realised median shifts
        // by the pseudo-random draw).
        let tol = if row.signal.starts_with("jitter") || row.signal.starts_with("shimmer") {
            3.0
        } else {
            1.5
        };
        let diff = (our_median - row.median_f0_hz).abs();
        assert!(
            diff <= tol,
            "yin {}: ours={:.3} librosa={:.3} diff={:.3} tol={:.1}",
            row.signal,
            our_median,
            row.median_f0_hz,
            diff,
            tol,
        );
        checked += 1;
    }
    assert!(
        checked >= 6,
        "expected at least 6 YIN comparisons, got {checked}"
    );
}

#[test]
fn pyin_median_f0_matches_librosa_within_tolerance() {
    let golden = read_golden();
    // Pin the ceiling to the librosa golden's 500 Hz (see the module doc) so
    // this comparison stays valid independent of the engine default, which is
    // 600 Hz to match Praat's `To Pitch (ac)`.
    let cfg = PitchConfig {
        max_freq_hz: 500.0,
        ..PitchConfig::default()
    };
    let mut checked = 0;
    for row in &golden {
        if row.method != "pyin" {
            continue;
        }
        let audio = load_wav(&row.signal);
        let frames = pitch(&audio, &cfg, PitchMethod::PYin);
        let our_median = median_voiced_f0(&frames, cfg.voicing_threshold).unwrap_or_else(|| {
            panic!(
                "pyin found no voiced frames for {} (librosa: {} voiced)",
                row.signal, row.n_voiced_frames,
            )
        });
        // pYIN snaps to its log-bin grid (both ours and librosa's), so
        // exact match is rare. 4 Hz tolerance comfortably covers the
        // grid resolution at f0 ≈ 200 Hz (≈ 1 bin = 0.6 Hz with
        // bins_per_semitone = 20, but the grid alignment between our
        // fmin/fmax span and librosa's can land us a few bins off).
        let tol = if row.signal.starts_with("jitter") || row.signal.starts_with("shimmer") {
            5.0
        } else {
            4.0
        };
        let diff = (our_median - row.median_f0_hz).abs();
        assert!(
            diff <= tol,
            "pyin {}: ours={:.3} librosa={:.3} diff={:.3} tol={:.1}",
            row.signal,
            our_median,
            row.median_f0_hz,
            diff,
            tol,
        );
        checked += 1;
    }
    assert!(
        checked >= 6,
        "expected at least 6 pYIN comparisons, got {checked}"
    );
}
