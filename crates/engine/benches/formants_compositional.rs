//! **Compositional-DSP perf spike — formants (LPC + root-solving profile).**
//! Checks whether "streaming composition is free" generalises past MFCC's
//! FFT-heavy per-frame work to formant tracking, whose per-frame cost is LPC
//! plus polynomial root-solving. Output-equality with production is pinned by a
//! test in `dsp::formants::spike`.
//!
//! Run with `just bench` (needs `--features spike`).

use divan::{Bencher, black_box};
use sadda_engine::dsp::formants::{FormantsConfig, formants, spike};
use sadda_engine::dsp::mfcc::spike::synth_signal;

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

const SR: u32 = 16_000;
const SECS: &[usize] = &[10, 60, 300];

#[divan::bench(args = SECS)]
fn production(bencher: Bencher, secs: usize) {
    let cfg = FormantsConfig::default();
    let sig = synth_signal(secs * SR as usize, SR);
    bencher.bench_local(|| formants(black_box(&sig), SR, black_box(&cfg)));
}

#[divan::bench(args = SECS)]
fn stream_comp(bencher: Bencher, secs: usize) {
    let cfg = FormantsConfig::default();
    let sig = synth_signal(secs * SR as usize, SR);
    bencher.bench_local(|| spike::streaming_compositional(black_box(&sig), SR, black_box(&cfg)));
}
