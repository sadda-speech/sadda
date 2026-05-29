//! Byte-faithfulness check for `engine::dsp::log_mel_whisper` against a
//! golden produced by OpenAI Whisper's own `log_mel_spectrogram` (its exact
//! filterbank asset + the verbatim audio.py steps). See
//! `tests/dsp/whisper/make_whisper_golden.py`. CI needs no torch/whisper —
//! it reads the committed TSVs.

use std::fs;
use std::path::Path;

fn read_input(path: &Path) -> Vec<f32> {
    let text = fs::read_to_string(path).expect("read input tsv");
    text.lines()
        .skip(1) // header
        .map(|l| l.trim().parse::<f32>().expect("input sample"))
        .collect()
}

/// Golden as `[n_mels][n_frames]`.
fn read_golden(path: &Path) -> Vec<Vec<f32>> {
    let text = fs::read_to_string(path).expect("read golden tsv");
    text.lines()
        .skip(1) // header
        .map(|l| {
            l.split('\t')
                .map(|v| v.parse::<f32>().expect("mel value"))
                .collect()
        })
        .collect()
}

#[test]
fn log_mel_whisper_matches_openai_whisper_golden() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/dsp/whisper");
    let input = read_input(&dir.join("whisper_mel_input.tsv"));
    let golden = read_golden(&dir.join("whisper_mel_golden.tsv"));
    let n_mels = golden.len();
    let n_frames = golden[0].len();

    // None target_frames + Whisper's `padding=0`: both compute over the raw
    // audio, so the frame counts line up directly.
    let ours = sadda_engine::dsp::log_mel_whisper(&input, 16_000, 400, 160, n_mels, None);
    assert_eq!(ours.dim(), (n_frames, n_mels), "shape (frames, mels)");

    let mut max_abs = 0.0_f32;
    for m in 0..n_mels {
        for fr in 0..n_frames {
            max_abs = max_abs.max((ours[[fr, m]] - golden[m][fr]).abs());
        }
    }
    // f32 FFT-implementation differences (realfft vs torch) propagate
    // through power → mel → log10 → normalise; a few ×10⁻³ is expected.
    assert!(
        max_abs < 5e-3,
        "max abs diff vs Whisper golden = {max_abs} (expected < 5e-3)"
    );
}
