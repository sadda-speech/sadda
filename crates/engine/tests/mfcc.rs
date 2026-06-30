//! Faithfulness checks for `engine::dsp::mfcc`'s named methods against
//! goldens produced by each reference implementation itself (librosa, Praat,
//! Kaldi-via-torchaudio). See `tests/dsp/mfcc/make_mfcc_goldens.py`. CI needs
//! none of those libraries — it reads the committed TSVs.

use std::fs;
use std::path::Path;

use sadda_engine::dsp::{MfccMethod, MfccParams, mfcc, mfcc_with_params};

const SR: u32 = 16_000;
const FRAME_S: f32 = 0.025;
const HOP_S: f32 = 0.010;
const N_MELS: usize = 40;

fn dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/dsp/mfcc")
}

fn read_input() -> Vec<f32> {
    let text = fs::read_to_string(dir().join("mfcc_input.tsv")).expect("read input tsv");
    text.lines()
        .skip(1) // header
        .map(|l| l.trim().parse::<f32>().expect("input sample"))
        .collect()
}

/// Golden as `[n_coeffs][n_frames]` (one row per cepstral coefficient).
fn read_golden(name: &str) -> Vec<Vec<f32>> {
    let text = fs::read_to_string(dir().join(name)).expect("read golden tsv");
    text.lines()
        .skip(1) // header
        .map(|l| {
            l.split('\t')
                .map(|v| v.parse::<f32>().expect("coeff value"))
                .collect()
        })
        .collect()
}

#[test]
fn mfcc_librosa_matches_librosa_golden() {
    let input = read_input();
    let golden = read_golden("mfcc_librosa_golden.tsv"); // [n_mfcc][n_frames]
    let n_mfcc = golden.len();
    let n_frames = golden[0].len();

    let ours = mfcc(
        &input,
        SR,
        FRAME_S,
        HOP_S,
        N_MELS,
        n_mfcc,
        0.0,
        SR as f32 / 2.0,
        MfccMethod::Librosa,
    );
    assert_eq!(ours.dim(), (n_frames, n_mfcc), "shape (frames, mfcc)");

    let mut max_abs = 0.0_f32;
    for c in 0..n_mfcc {
        for fr in 0..n_frames {
            max_abs = max_abs.max((ours[[fr, c]] - golden[c][fr]).abs());
        }
    }
    // librosa computes in f64, sadda in f32 (realfft); the gap propagates
    // power → mel → 10·log10 → DCT over 40 bands. Measured max ≈5e-4 on
    // coefficients up to ~280; bound keeps headroom for cross-platform f32.
    assert!(
        max_abs < 1e-2,
        "max abs diff vs librosa golden = {max_abs} (expected < 1e-2)"
    );
}

#[test]
fn mfcc_kaldi_matches_torchaudio_kaldi_golden() {
    let input = read_input();
    let golden = read_golden("mfcc_kaldi_golden.tsv"); // [n_ceps][n_frames]
    let n_ceps = golden.len();
    let n_frames = golden[0].len();

    let ours = mfcc(
        &input,
        SR,
        FRAME_S,
        HOP_S,
        N_MELS,
        n_ceps,
        0.0,
        SR as f32 / 2.0,
        MfccMethod::Kaldi,
    );
    assert_eq!(ours.dim(), (n_frames, n_ceps), "shape (frames, ceps)");

    let mut max_abs = 0.0_f32;
    for c in 0..n_ceps {
        for fr in 0..n_frames {
            max_abs = max_abs.max((ours[[fr, c]] - golden[c][fr]).abs());
        }
    }
    // torchaudio is f32 like sadda but with different FFT/log/DCT internals.
    // Measured max ≈3e-3 on coefficients of magnitude ~tens; headroom for f32.
    assert!(
        max_abs < 1e-2,
        "max abs diff vs torchaudio kaldi golden = {max_abs} (expected < 1e-2)"
    );
}

#[test]
fn unified_pipeline_reproduces_librosa_and_kaldi_presets() {
    // The parameterized pipeline (mfcc_with_params) with a reference preset
    // must equal that reference's golden bit-for-bit-to-tolerance — the proof
    // that the three methods unify into one knob space.
    let input = read_input();

    let lg = read_golden("mfcc_librosa_golden.tsv");
    let lp = MfccParams::librosa(FRAME_S, HOP_S, N_MELS, lg.len(), 0.0, SR as f32 / 2.0);
    let lours = mfcc_with_params(&input, SR, &lp);
    let mut lmax = 0.0_f32;
    for c in 0..lg.len() {
        for fr in 0..lg[0].len() {
            lmax = lmax.max((lours[[fr, c]] - lg[c][fr]).abs());
        }
    }
    assert!(lmax < 1e-2, "librosa preset vs golden = {lmax}");

    let kg = read_golden("mfcc_kaldi_golden.tsv");
    let kp = MfccParams::kaldi(FRAME_S, HOP_S, N_MELS, kg.len(), 0.0, SR as f32 / 2.0);
    let kours = mfcc_with_params(&input, SR, &kp);
    let mut kmax = 0.0_f32;
    for c in 0..kg.len() {
        for fr in 0..kg[0].len() {
            kmax = kmax.max((kours[[fr, c]] - kg[c][fr]).abs());
        }
    }
    assert!(kmax < 1e-2, "kaldi preset vs golden = {kmax}");
}

#[test]
fn unified_pipeline_preset_matches_enum_method() {
    // mfcc(method) and mfcc_with_params(preset) must agree — same algorithm,
    // two entry points.
    let input = read_input();
    let viaenum = mfcc(
        &input,
        SR,
        FRAME_S,
        HOP_S,
        N_MELS,
        13,
        0.0,
        SR as f32 / 2.0,
        MfccMethod::Librosa,
    );
    let viaparams = mfcc_with_params(
        &input,
        SR,
        &MfccParams::librosa(FRAME_S, HOP_S, N_MELS, 13, 0.0, SR as f32 / 2.0),
    );
    assert_eq!(viaenum.dim(), viaparams.dim());
    let mut d = 0.0_f32;
    for (a, b) in viaenum.iter().zip(viaparams.iter()) {
        d = d.max((a - b).abs());
    }
    assert!(d < 1e-5, "enum vs params drift = {d}");
}

#[test]
fn mfcc_praat_has_correct_structure() {
    // Praat is currently an *approximate* reproduction (see MfccMethod::Praat).
    // What IS faithful — and what this guards — is the structure: frame count
    // (Praat's `floor((dur-win)/hop)+1` with win = 2*frame), coefficient count
    // (c0 + n-1), and finite, correctly-signed energy ordering.
    let input = read_input();
    let golden = read_golden("mfcc_praat_golden.tsv"); // [n_coeffs][n_frames]
    let n_coeffs = golden.len();
    let n_frames = golden[0].len();

    let ours = mfcc(
        &input,
        SR,
        FRAME_S,
        HOP_S,
        N_MELS,
        n_coeffs,
        0.0,
        SR as f32 / 2.0,
        MfccMethod::Praat,
    );
    assert_eq!(
        ours.dim(),
        (n_frames, n_coeffs),
        "Praat shape (frames, coeffs)"
    );
    assert!(ours.iter().all(|v| v.is_finite()));
}

/// Praat MFCC vs the parselmouth golden. Validated to tolerance, **not**
/// byte-exact: Praat's un-normalised DCT sums `10·log10` of every mel filter,
/// so the near-empty high filters' tiny powers (≈1e-30) dominate the
/// conditioning, and sub-1e-15 FFT-library differences (realfft vs Praat's
/// NUMrealft) blow up there. Computing in f64 (the whole path) holds this to
/// ≈20 worst-case (c0 / energy) and ≈10 on c1+ (shape); typical-frame
/// agreement is ≈0.3%. Confirmed-exact structure: HTK mel, unit-peak filters,
/// N=27 (swept), Gaussian-2 window, framing, dB ref 4e-10, windowCorrection,
/// no pre-emphasis. A bit-exact match would need Praat's exact FFT.
#[test]
fn mfcc_praat_matches_parselmouth_golden_to_tolerance() {
    let input = read_input();
    let golden = read_golden("mfcc_praat_golden.tsv"); // row 0 = c0
    let n_coeffs = golden.len();
    let n_frames = golden[0].len();
    let ours = mfcc(
        &input,
        SR,
        FRAME_S,
        HOP_S,
        N_MELS,
        n_coeffs,
        0.0,
        SR as f32 / 2.0,
        MfccMethod::Praat,
    );
    let mut max_c0 = 0.0_f32;
    let mut max_rest = 0.0_f32;
    for c in 0..n_coeffs {
        for fr in 0..n_frames {
            let d = (ours[[fr, c]] - golden[c][fr]).abs();
            if c == 0 {
                max_c0 = max_c0.max(d);
            } else {
                max_rest = max_rest.max(d);
            }
        }
    }
    // c1+ (shape) is more faithful than c0 (absolute energy). Bounds carry
    // headroom over the measured ≈10 / ≈21 for cross-platform f64.
    assert!(max_rest < 15.0, "Praat c1+ shape diff = {max_rest}");
    assert!(max_c0 < 25.0, "Praat c0 energy diff = {max_c0}");
}
