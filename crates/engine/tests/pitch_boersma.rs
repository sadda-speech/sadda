//! Faithful-Boersma pitch validation: median voiced f0 per fixture
//! within tolerance of Praat 6.x's `Sound: To Pitch (ac)…` output.
//!
//! The golden TSV at `tests/clinical/fixtures/pitch_boersma_golden.tsv`
//! is produced by `tests/dsp/praat/pitch_boersma.praat` against the
//! sustained-tone fixtures from synth_fixtures.py. Both use the same
//! Boersma defaults (max_candidates=15, very_accurate=false,
//! silence_threshold=0.03, voicing_threshold=0.45, octave_cost=0.01,
//! octave_jump_cost=0.35, voiced_unvoiced_cost=0.14), so a direct
//! median comparison is meaningful.
//!
//! Tolerance: 1.5 Hz (~1%) for clean and HNR fixtures, 3 Hz for the
//! jitter / shimmer fixtures (which inject ±3% period jitter). Both
//! are well inside the parabolic-refinement noise floor noted in the
//! pitch module docs.

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
    n_voiced_frames: usize,
    median_f0_hz: f32,
}

fn read_golden() -> Vec<GoldenRow> {
    let path = fixtures_dir().join("pitch_boersma_golden.tsv");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 || line.is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 3, "golden tsv row {i}: {line}");
        rows.push(GoldenRow {
            signal: cols[0].to_string(),
            n_voiced_frames: cols[1].parse().unwrap(),
            median_f0_hz: cols[2].parse().unwrap(),
        });
    }
    rows
}

#[test]
fn boersma_median_f0_matches_praat_within_tolerance() {
    let golden = read_golden();
    assert!(!golden.is_empty(), "no golden rows found");

    let cfg = PitchConfig::default();

    for row in &golden {
        let audio = load_wav(&row.signal);
        let frames = pitch(&audio, &cfg, PitchMethod::Boersma);
        let our_median = median_voiced_f0(&frames, cfg.voicing_threshold).unwrap_or_else(|| {
            panic!(
                "boersma found no voiced frames for {} (praat: {} voiced)",
                row.signal, row.n_voiced_frames,
            )
        });

        // Tighter tolerance for clean / HNR fixtures (no period jitter);
        // wider for fixtures with injected ±3% period jitter where the
        // realized median is a function of the pseudo-random draws.
        let tolerance_hz = if row.signal.starts_with("jitter") {
            3.0
        } else {
            1.5
        };
        let diff = (our_median - row.median_f0_hz).abs();
        assert!(
            diff <= tolerance_hz,
            "{}: boersma median = {:.3} Hz, praat = {:.3} Hz (diff {:.3} Hz, tol {:.1} Hz)",
            row.signal,
            our_median,
            row.median_f0_hz,
            diff,
            tolerance_hz,
        );
    }
}

#[test]
fn boersma_finds_voiced_frames_for_every_sustained_fixture() {
    let golden = read_golden();
    let cfg = PitchConfig::default();
    for row in &golden {
        let audio = load_wav(&row.signal);
        let frames = pitch(&audio, &cfg, PitchMethod::Boersma);
        let n_voiced = frames
            .iter()
            .filter(|f| f.voicing >= cfg.voicing_threshold)
            .count();
        // Praat counted at least 96 voiced frames in every fixture;
        // require at least 60 from us (≈ 60% match), which is loose
        // enough to absorb hop-rate differences (Praat auto-derives a
        // 0.0025s hop from a 75 Hz floor; we default to 0.01s) without
        // false negatives.
        assert!(
            n_voiced >= 60,
            "{}: only {} voiced frames found (praat: {})",
            row.signal,
            n_voiced,
            row.n_voiced_frames,
        );
    }
}
