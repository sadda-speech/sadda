//! Two-pass adaptive pitch-range estimation (De Looze & Hirst 2008):
//! floor = 0.75·q25, ceiling = 1.5·q75 from a wide (60–750 Hz) first pass, then
//! re-track. Validated on the sustained-tone fixtures (known f0), where the
//! derived range must bracket the true f0 and be tighter than the wide first
//! pass. Reference: <https://doi.org/10.21437/SpeechProsody.2008-32>.

use std::path::PathBuf;

use sadda_engine::Audio;
use sadda_engine::pitch::{
    ADAPTIVE_MIN_VOICED, PitchConfig, PitchFrame, PitchMethod, TWO_PASS_CEILING_HZ,
    TWO_PASS_FLOOR_HZ, estimate_pitch_range, two_pass_pitch, voiced_f0_quantiles,
};
use sadda_engine::units::Hertz;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/clinical/fixtures")
}

fn load_wav(stem: &str) -> Audio {
    let path = fixtures_dir().join(format!("{stem}.wav"));
    Audio::from_wav_path(&path).unwrap_or_else(|e| panic!("load {stem}.wav: {e}"))
}

fn median_voiced_f0(frames: &[PitchFrame], vt: f32) -> Option<f32> {
    let mut v: Vec<f32> = frames
        .iter()
        .filter(|f| f.voicing >= vt)
        .map(|f| f.frequency_hz.value())
        .collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    Some(v[v.len() / 2])
}

fn frame(hz: f32, voicing: f32, t: usize) -> PitchFrame {
    PitchFrame {
        time_seconds: t as f64,
        frequency_hz: Hertz::new(hz),
        voicing,
    }
}

#[test]
fn estimated_range_brackets_true_f0_and_tightens_ceiling() {
    // Each fixture is a sustained tone. The derived range must bracket the true
    // f0 and pull the ceiling well in from the 750 Hz first-pass bracket. (The
    // floor is *not* asserted above 60: on the jittered fixtures ±3% jitter
    // causes occasional octave-down errors in the wide first pass, which drag
    // q25 — and thus 0.75·q25 — a little below 60. That is faithful to the
    // method; De Looze & Hirst note the quartiles are only fairly robust to
    // detection errors.)
    for (stem, f0) in [
        ("clean_120hz", 120.0f32),
        ("jitter_150hz", 150.0),
        ("jitter_shimmer_200hz", 200.0),
    ] {
        let cfg = PitchConfig::default();
        let (floor, ceiling) = estimate_pitch_range(&load_wav(stem), &cfg, PitchMethod::Boersma)
            .unwrap_or_else(|| panic!("no range estimate for {stem}"));
        assert!(
            floor < f0 && f0 < ceiling,
            "{stem}: range ({floor:.1}, {ceiling:.1}) must bracket {f0}"
        );
        assert!(
            ceiling < TWO_PASS_CEILING_HZ,
            "{stem}: ceiling {ceiling:.1} should be tighter than the {TWO_PASS_CEILING_HZ} first pass"
        );
    }
}

#[test]
fn estimated_range_matches_formula_on_clean_tone() {
    // On a clean tone the octave-robust Boersma first pass gives quartiles at the
    // tone, so the De Looze & Hirst formula lands squarely: floor ≈ 0.75·f0,
    // ceiling ≈ 1.5·f0.
    let f0 = 120.0f32;
    let cfg = PitchConfig::default();
    let (floor, ceiling) =
        estimate_pitch_range(&load_wav("clean_120hz"), &cfg, PitchMethod::Boersma)
            .expect("clean tone yields a range");
    assert!(
        floor > TWO_PASS_FLOOR_HZ,
        "clean floor {floor:.1} should tighten above 60"
    );
    assert!(
        (floor - 0.75 * f0).abs() < 0.1 * f0,
        "floor {floor:.1} ≉ 0.75·{f0}"
    );
    assert!(
        (ceiling - 1.5 * f0).abs() < 0.1 * f0,
        "ceiling {ceiling:.1} ≉ 1.5·{f0}"
    );
}

#[test]
fn two_pass_recovers_true_f0() {
    for (stem, f0) in [("clean_120hz", 120.0f32), ("jitter_shimmer_200hz", 200.0)] {
        let cfg = PitchConfig::default();
        let frames = two_pass_pitch(&load_wav(stem), &cfg, PitchMethod::Boersma);
        let med = median_voiced_f0(&frames, cfg.voicing_threshold)
            .unwrap_or_else(|| panic!("no voiced frames for {stem}"));
        assert!(
            (med - f0).abs() < 3.0,
            "{stem}: two-pass median {med:.1} vs true {f0}"
        );
    }
}

#[test]
fn quantiles_interpolate_type7_and_ignore_unvoiced() {
    // A ramp 100..=200 in 11 steps: type-7 q25 = 125, q50 = 150, q75 = 175.
    let ramp: Vec<PitchFrame> = (0..=10)
        .map(|i| frame(100.0 + 10.0 * i as f32, 1.0, i))
        .collect();
    let q = voiced_f0_quantiles(&ramp, 0.45, &[0.25, 0.5, 0.75]).expect("enough voiced frames");
    assert!((q[0] - 125.0).abs() < 0.5, "q25 = {}", q[0]);
    assert!((q[1] - 150.0).abs() < 0.5, "q50 = {}", q[1]);
    assert!((q[2] - 175.0).abs() < 0.5, "q75 = {}", q[2]);

    // Unvoiced frames (below threshold) are dropped from the distribution: mix
    // in loud low-voicing outliers and the quartiles must not move.
    let mut with_noise = ramp.clone();
    for i in 0..20 {
        with_noise.push(frame(1000.0, 0.1, 100 + i));
    }
    let q2 = voiced_f0_quantiles(&with_noise, 0.45, &[0.25, 0.75]).unwrap();
    assert!(
        (q2[0] - 125.0).abs() < 0.5 && (q2[1] - 175.0).abs() < 0.5,
        "outliers leaked in: {q2:?}"
    );
}

#[test]
fn sparse_voicing_yields_no_estimate() {
    // Fewer than ADAPTIVE_MIN_VOICED voiced frames → None (too sparse to trust).
    let few: Vec<PitchFrame> = (0..ADAPTIVE_MIN_VOICED - 1)
        .map(|i| frame(120.0, 1.0, i))
        .collect();
    assert!(voiced_f0_quantiles(&few, 0.45, &[0.25, 0.75]).is_none());
}

#[test]
fn two_pass_falls_back_when_range_unestimable() {
    // Silence has too few voiced frames to estimate a range; two_pass_pitch must
    // still run (falling back to the config's range), not panic or hang.
    let silence = Audio::from_samples(vec![0.0f32; 16_000], 16_000, 1);
    let cfg = PitchConfig::default();
    let _ = two_pass_pitch(&silence, &cfg, PitchMethod::Boersma);
}
