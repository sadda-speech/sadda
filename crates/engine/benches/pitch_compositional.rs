//! **Compositional-DSP perf spike — pitch (autocorrelation profile).**
//! Checks whether "streaming composition is free" holds for an autocorrelation
//! inner loop (memory-access-bound `O(frame·lag)`, no FFT / no root-solving).
//! Output-equality with production is pinned by a test in `pitch::spike`.
//!
//! Run with `just bench` (needs `--features spike`).

use divan::{Bencher, black_box};
use sadda_engine::audio::Audio;
use sadda_engine::dsp::mfcc::spike::synth_signal;
use sadda_engine::pitch::{PitchConfig, autocorrelation, spike};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

const SR: u32 = 16_000;
const SECS: &[usize] = &[10, 60, 300];

#[divan::bench(args = SECS)]
fn production(bencher: Bencher, secs: usize) {
    let cfg = PitchConfig::default();
    let audio = Audio {
        samples: synth_signal(secs * SR as usize, SR),
        sample_rate: SR,
        channels: 1,
    };
    bencher.bench_local(|| autocorrelation(black_box(&audio), black_box(&cfg)));
}

#[divan::bench(args = SECS)]
fn stream_comp(bencher: Bencher, secs: usize) {
    let cfg = PitchConfig::default();
    let sig = synth_signal(secs * SR as usize, SR);
    bencher.bench_local(|| spike::streaming_compositional(black_box(&sig), SR, black_box(&cfg)));
}

#[divan::bench(args = SECS)]
fn stream_fused(bencher: Bencher, secs: usize) {
    let cfg = PitchConfig::default();
    let sig = synth_signal(secs * SR as usize, SR);
    bencher.bench_local(|| spike::streaming_fused(black_box(&sig), SR, black_box(&cfg)));
}
