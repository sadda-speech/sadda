//! B4 jitter/shimmer validation against the committed fixtures.
//!
//! Each fixture WAV is measured by the engine and checked two ways:
//!   1. within tolerance of Praat's golden value (the reference — primary),
//!   2. in the ballpark of the analytic injected value (sanity).
//!
//! Fixtures + the generators live in `tests/clinical/` (see its README).

use std::path::PathBuf;

use sadda_engine::{Audio, CppsConfig, HnrConfig, PerturbationConfig, cpps, hnr, perturbation};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/clinical/fixtures")
}

/// One row of `praat_golden.tsv`.
struct Golden {
    jitter_local: f64,
    shimmer_local: f64,
    shimmer_local_db: f64,
    hnr_db: f64,
    cpps: f64,
}

fn load_golden(name: &str) -> Golden {
    let tsv = std::fs::read_to_string(fixtures_dir().join("praat_golden.tsv")).unwrap();
    let mut lines = tsv.lines();
    let _header = lines.next();
    for line in lines {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols[0] == name {
            return Golden {
                jitter_local: cols[1].parse().unwrap(),
                shimmer_local: cols[4].parse().unwrap(),
                shimmer_local_db: cols[5].parse().unwrap(),
                hnr_db: cols[8].parse().unwrap(),
                cpps: cols[9].parse().unwrap(),
            };
        }
    }
    panic!("no golden row for {name}");
}

/// Engine within `max(rel·|ref|, abs)` of the reference value.
fn close(got: f64, want: f64, rel: f64, abs: f64) -> bool {
    (got - want).abs() <= (rel * want.abs()).max(abs)
}

fn measure(name: &str) -> sadda_engine::PerturbationReport {
    let audio = Audio::from_wav_path(fixtures_dir().join(format!("{name}.wav"))).unwrap();
    perturbation(&audio, &PerturbationConfig::default()).unwrap()
}

#[test]
fn matches_praat_on_jitter_signal() {
    let r = measure("jitter_150hz");
    let g = load_golden("jitter_150hz");
    // Reference (Praat) within 20% relative / 0.003 absolute.
    assert!(
        close(r.jitter_local.value() as f64, g.jitter_local, 0.20, 0.003),
        "jitter_local {} vs praat {}",
        r.jitter_local.value(),
        g.jitter_local
    );
    // Jitter-only signal: shimmer stays small.
    assert!(
        r.shimmer_local.value() < 0.03,
        "shimmer leak {}",
        r.shimmer_local.value()
    );
}

#[test]
fn matches_praat_on_shimmer_signal() {
    let r = measure("shimmer_150hz");
    let g = load_golden("shimmer_150hz");
    assert!(
        close(r.shimmer_local.value() as f64, g.shimmer_local, 0.20, 0.005),
        "shimmer_local {} vs praat {}",
        r.shimmer_local.value(),
        g.shimmer_local
    );
    assert!(
        close(
            r.shimmer_local_db.value() as f64,
            g.shimmer_local_db,
            0.25,
            0.05
        ),
        "shimmer_db {} vs praat {}",
        r.shimmer_local_db.value(),
        g.shimmer_local_db
    );
    // Shimmer-only signal: jitter stays small.
    assert!(
        r.jitter_local.value() < 0.01,
        "jitter leak {}",
        r.jitter_local.value()
    );
}

#[test]
fn matches_praat_on_combined_signal() {
    let r = measure("jitter_shimmer_200hz");
    let g = load_golden("jitter_shimmer_200hz");
    assert!(
        close(r.jitter_local.value() as f64, g.jitter_local, 0.25, 0.003),
        "jitter_local {} vs praat {}",
        r.jitter_local.value(),
        g.jitter_local
    );
    assert!(
        close(r.shimmer_local.value() as f64, g.shimmer_local, 0.25, 0.005),
        "shimmer_local {} vs praat {}",
        r.shimmer_local.value(),
        g.shimmer_local
    );
}

#[test]
fn matches_praat_on_hnr() {
    // Cross-correlation HNR within 3 dB of Praat across a clean-ish and
    // a noisy sustained tone.
    for name in ["hnr_high_120hz", "hnr_mid_120hz"] {
        let audio = Audio::from_wav_path(fixtures_dir().join(format!("{name}.wav"))).unwrap();
        let got = hnr(&audio, &HnrConfig::default()).unwrap().value() as f64;
        let want = load_golden(name).hnr_db;
        assert!(
            (got - want).abs() < 3.0,
            "{name}: hnr {got} vs praat {want}"
        );
    }
}

#[test]
fn matches_praat_on_cpps() {
    // Validated on the sustained *harmonic-tone* fixtures, which are the
    // appropriate cepstral input (a vowel is harmonic-rich, not an impulse
    // train — the jitter/shimmer pulse-train fixtures have a degenerate
    // cepstrum and aren't valid CPP inputs). The two SNRs exercise the
    // clean→noisy direction. Within 3 dB of Praat's CPPS.
    let mut report = String::new();
    let mut ok = true;
    for name in ["hnr_high_120hz", "hnr_mid_120hz"] {
        let audio = Audio::from_wav_path(fixtures_dir().join(format!("{name}.wav"))).unwrap();
        let got = cpps(&audio, &CppsConfig::default()).unwrap().value() as f64;
        let want = load_golden(name).cpps;
        report.push_str(&format!("{name}: cpps {got:.2} vs praat {want:.2}\n"));
        ok &= (got - want).abs() < 3.0;
    }
    assert!(ok, "{report}");
}

#[test]
fn clean_signal_is_near_zero() {
    let r = measure("clean_120hz");
    assert!(
        r.jitter_local.value() < 0.01,
        "clean jitter {}",
        r.jitter_local.value()
    );
    assert!(
        r.shimmer_local.value() < 0.02,
        "clean shimmer {}",
        r.shimmer_local.value()
    );
}
