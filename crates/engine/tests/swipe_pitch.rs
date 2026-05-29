//! Validates `engine::pitch::swipe` (SWIPE') against a golden produced by
//! running Camacho's **own** dissertation MATLAB (`swipep`) under Octave —
//! see `tests/dsp/swipe/run_swipe_octave.m`. CI reads the committed TSVs
//! (no Octave/scipy needed). The fixtures are deterministic harmonic tones
//! (`make_swipe_fixtures.py`).

use std::fs;
use std::path::Path;

use sadda_engine::Audio;
use sadda_engine::pitch::{PitchConfig, PitchMethod, pitch};

fn read_input(path: &Path) -> Vec<f32> {
    fs::read_to_string(path)
        .expect("read input tsv")
        .lines()
        .skip(1)
        .map(|l| l.trim().parse::<f32>().expect("sample"))
        .collect()
}

/// Golden rows: (true_f0, author-exact SWIPE' median Hz).
fn read_golden(path: &Path) -> Vec<(f64, f64)> {
    fs::read_to_string(path)
        .expect("read golden tsv")
        .lines()
        .skip(1)
        .map(|l| {
            let mut it = l.split('\t');
            let a = it.next().unwrap().parse::<f64>().unwrap();
            let b = it.next().unwrap().parse::<f64>().unwrap();
            (a, b)
        })
        .collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

#[test]
fn swipe_matches_octave_author_golden() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/dsp/swipe");
    let golden = read_golden(&dir.join("swipe_golden.tsv"));
    assert!(!golden.is_empty());

    let cfg = PitchConfig {
        min_freq_hz: 100.0,
        max_freq_hz: 600.0,
        hop_size_seconds: 0.01,
        ..PitchConfig::default()
    };

    for (true_f0, golden_median) in golden {
        let samples = read_input(&dir.join(format!("f0_{}_input.tsv", true_f0 as i64)));
        let audio = Audio {
            samples,
            sample_rate: 16_000,
            channels: 1,
        };
        let frames = pitch(&audio, &cfg, PitchMethod::Swipe);
        // Camacho's sTHR = 0.30 voicing threshold.
        let voiced: Vec<f64> = frames
            .iter()
            .filter(|f| f.voicing >= 0.30)
            .map(|f| f.frequency_hz.value() as f64)
            .collect();
        assert!(!voiced.is_empty(), "f0={true_f0}: no voiced frames");
        let med = median(voiced);
        // Tight tolerance: our Rust reproduces the author's algorithm with no
        // deviation (not-a-knot spline, hanning N+1), so it should track the
        // Octave golden to well under a Hz (f32 vs f64 FFT aside).
        let diff = (med - golden_median).abs();
        assert!(
            diff < 0.05,
            "f0={true_f0}: swipe median {med:.3} Hz vs author golden {golden_median:.3} Hz \
             (|Δ| = {diff:.3} > 0.05)"
        );
    }
}
