//! **Compositional-DSP perf spike.** Benchmarks three MFCC implementations of
//! the *same* computation (output-equality pinned by a test in `dsp::mfcc::spike`):
//!
//! 1. `production`  — the hand-fused per-frame loop (`mfcc_with_params`).
//! 2. `naive_comp`  — whole-signal stage-by-stage, materialising every
//!    intermediate (the "~2 GB on a long file" model).
//! 3. `stream_comp` — one frame at a time through boxed `dyn` stages
//!    (compositional done right; production-like memory).
//!
//! `divan::AllocProfiler` reports bytes-allocated per run alongside wall time,
//! so the memory story (not just the CPU story) shows up in the table.
//!
//! Run with `just bench` (or `cargo bench -p sadda-engine`).

use divan::{Bencher, black_box};
use sadda_engine::dsp::mfcc::{MfccParams, mfcc_with_params, spike};

#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

const SR: u32 = 16_000;
/// Signal lengths in seconds. 300 s (5 min) is where materialisation memory
/// starts to bite; keep the top end modest so the suite stays quick.
const SECS: &[usize] = &[10, 60, 300];

fn params() -> MfccParams {
    MfccParams::librosa(0.025, 0.010, 40, 13, 0.0, 8_000.0)
}

#[divan::bench(args = SECS)]
fn production(bencher: Bencher, secs: usize) {
    let p = params();
    let sig = spike::synth_signal(secs * SR as usize, SR);
    bencher.bench_local(|| mfcc_with_params(black_box(&sig), SR, black_box(&p)));
}

#[divan::bench(args = SECS)]
fn naive_comp(bencher: Bencher, secs: usize) {
    let p = params();
    let sig = spike::synth_signal(secs * SR as usize, SR);
    bencher.bench_local(|| spike::naive_compositional(black_box(&sig), SR, black_box(&p)));
}

#[divan::bench(args = SECS)]
fn stream_comp(bencher: Bencher, secs: usize) {
    let p = params();
    let sig = spike::synth_signal(secs * SR as usize, SR);
    bencher.bench_local(|| spike::streaming_compositional(black_box(&sig), SR, black_box(&p)));
}
