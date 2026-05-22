# DEVLOG

A running log of research, decisions, and development for the SpeechAnalysisTool project — a planned next-generation phonetics / speech-science tool, conceived as a successor to Praat.

Newest entries at the top. Each entry is dated `YYYY-MM-DD` and tagged with a short topic.

---

## 2026-05-21 — DSP method diversity (project design principle)

Goal: codify a project-wide commitment for the DSP namespace. Raised mid-C2 as a course correction; this entry captures it as a durable design principle.

### The principle

Two coupled commitments for every member of `sadda.dsp.*` (and the underlying `engine::dsp` / `engine::pitch` modules):

1. **Cite the canonical source.** Every public DSP function carries at least one bibliographic reference in its doc comment — typically a paper, textbook section, or canonical implementation (Praat manual / `scipy.signal` / `librosa` source code all count).
2. **Provide multiple non-equivalent methods where they exist.** Domain-norm in speech research is that "the formant tracker" or "the pitch tracker" is shorthand for one of several non-equivalent methods, each with known strengths. Providing alternates lets users compare methods on their own data; defaulting to one without naming alternatives obscures algorithmic choice.

### Why this matters here

- **Phoneticians know the alternates by name.** Praat's `Sound.to_formant_burg` vs `Sound.to_formant_burg_robust` vs autocorrelation-LPC formants are *different* methods producing measurably different tracks — not a numerical-precision difference. Users coming from Praat will look for both.
- **AI/ML users expect citations to be discoverable from API docs.** `help(sadda.dsp.mfcc)` should answer "what mel scale? — Davis & Mermelstein (1980), HTK convention; Slaney (1998) deferred" without context-switching to external references.
- **Forensic / clinical use cases need auditable algorithmic choices.** Citation + method name is the minimum to defend a measurement in court or in a clinical decision.
- **Multiple methods make the namespace honest about what it is.** Pitch tracking is a research area, not a solved problem. Naming the method (`autocorrelation_pitch`, `yin_pitch`, …) instead of `pitch` keeps the API honest.

### What "non-equivalent" means

Two methods are non-equivalent if they can produce *measurably different outputs* on the same input under reasonable parameter choices. Examples:

| Family | Method A | Method B | Difference |
|---|---|---|---|
| LPC | Autocorrelation | Burg | Stability on short frames; Burg is Praat's formant-tracking default |
| Pitch | Autocorrelation | YIN | YIN handles octave errors better; cited in millions of papers |
| Pitch | Heuristic (autocorrelation, YIN, RAPT, SWIPE) | Neural (CREPE) | Substantially different cost & accuracy profiles |
| Mel scale | HTK | Slaney | Different filter-bank cutoff conventions; ~5% energy-distribution shift |
| Window | Symmetric | Periodic | Symmetric for filter design; periodic for STFT (librosa's `sym` flag) |
| Spectrogram | Magnitude | Power | `|X|` vs `|X|²`; users from MATLAB world expect both |

If two methods only differ by floating-point precision or implementation detail, they are *not* non-equivalent and don't need separate functions.

### How to apply (operational rules)

- **Doc-comment citations are required** on every public DSP function. Format: short author/year + a publication link whenever one exists (DOI preferred; arXiv / JSTOR / publisher URL / archived tech-report URL otherwise). Older textbooks / book chapters without a stable URL get cited by ISBN and chapter section. The Praat manual, `scipy.signal` source, and `librosa` source code count as canonical references for well-known methods, with their stable doc URLs included.
- **Multi-method exposure scope:**
  - **Cheap flag (`power=True/False`, `sym=True/False`)**: ship alongside the sibling in the same slice.
  - **Substantial implementation (YIN tracker, Slaney mel scale with full filter recomputation)**: a new function in a later slice; tracked in a `### Deferred alternates` section of the slice's DEVLOG entry.
- **Document the chosen variant when not multi-method.** If C2's MFCC uses HTK mel + pre-emphasis + log (not dB), the doc comment says so + cites Davis & Mermelstein (1980) and notes Slaney (1998) as a known alternative.
- **DEVLOG slice entries for DSP work include a `### Deferred alternates` section** enumerating known non-equivalent methods not shipped yet. This is the running development-target list.
- **Function naming when multi-method**: prefer method-suffixed names (`autocorrelation_pitch`, `yin_pitch`) over polymorphic flags (`pitch(method="yin")`) when each method has substantial code behind it. Flags are fine when the method choice is a small algorithmic variant of a shared implementation.

### Application to C2 and back to C1

- **C2 ships both autocorrelation LPC and Burg's method** (`engine::dsp::lpc::autocorr_lpc` and `engine::dsp::lpc::burg_lpc`), selectable via an `LpcMethod` enum at the formant-tracker boundary. Burg is the default for formants (Praat convention).
- **C2 ships citations on every new DSP function**: LPC (Markel & Gray 1976), formants (Markel 1972; McCandless 1974), MFCC (Davis & Mermelstein 1980), Aberth roots (Aberth 1973; Bini 1996), voicing (Boersma 1993 for the autocorrelation-peak ratio).
- **C1 citations are backfilled** in the same slice: windows (Harris 1978; Kaiser 1980; scipy.signal.windows source), STFT (Oppenheim & Schafer §11), intensity (RMS basics; Boersma & Weenink Praat manual for intensity-object conventions).
- **Deferred alternates list** (tracked across slices; not blockers for v1):
  - YIN pitch (de Cheveigné & Kawahara 2002)
  - RAPT pitch (Talkin 1995)
  - SWIPE / SWIPE' pitch (Camacho & Harris 2008)
  - CREPE pitch (Kim et al. 2018; neural)
  - pyin pitch (Mauch & Dixon 2014)
  - Slaney mel scale (Slaney 1998)
  - Magnitude spectrogram (alongside the current power-only)
  - Symmetric/periodic window toggle
  - Burg-robust LPC (Praat's Robust variant)
  - dB-SPL intensity (needs Instrument calibration)

### Sources / references

- Davis, S.B. & Mermelstein, P. (1980), "Comparison of parametric representations for monosyllabic word recognition…" *IEEE TASSP* 28(4). https://doi.org/10.1109/TASSP.1980.1163420
- de Cheveigné, A. & Kawahara, H. (2002), "YIN, a fundamental frequency estimator for speech and music." *J. Acoust. Soc. Am.* 111(4). https://doi.org/10.1121/1.1458024
- Talkin, D. (1995), "A robust algorithm for pitch tracking (RAPT)." Ch. 14 of *Speech Coding and Synthesis* (Kleijn & Paliwal, eds.), Elsevier. ISBN 978-0-444-82169-1
- Camacho, A. & Harris, J.G. (2008), "A sawtooth waveform inspired pitch estimator for speech and music." *J. Acoust. Soc. Am.* 124(3). https://doi.org/10.1121/1.2951592
- Mauch, M. & Dixon, S. (2014), "pYIN: A fundamental frequency estimator using probabilistic threshold distributions." *Proc. ICASSP*. https://doi.org/10.1109/ICASSP.2014.6853678
- Kim, J.W., Salamon, J., Li, P., Bello, J.P. (2018), "CREPE: A convolutional representation for pitch estimation." *Proc. ICASSP*. https://arxiv.org/abs/1802.06182
- Slaney, M. (1998), "Auditory toolbox v2." Interval Research Tech. Report 1998-010. https://engineering.purdue.edu/~malcolm/interval/1998-010/
- librosa method-diversity precedent: https://librosa.org/doc/latest/feature.html
- Praat formant methods: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html

---

## 2026-05-21 — Advanced DSP (C2): LPC + hand-rolled Aberth roots → formants, mel→DCT MFCC, voicing on PitchFrame

Goal: settle the seventh Phase 1 slice. C2 lands the three advanced DSP families on top of C1's foundation: **formants** via LPC + polynomial root-finding, **MFCC** via mel-filterbank + DCT-II, and a **voicing decision** added to the existing autocorrelation pitch tracker. All three live inside `engine::dsp` and surface through `sadda.dsp.*`.

C2 also marks the first slice to land under the 2026-05-21 **DSP method-diversity principle** (see the entry directly above): every public DSP function carries a doc-comment citation with a publication link, and LPC ships both the autocorrelation method and Burg's method side-by-side.

### What C2 must deliver

From the Phase-1 slicing entry: (1) formants via LPC + root-solver; (2) MFCC (mel-filterbank → DCT); (3) refined pitch with voicing decision (extends Phase 0's autocorrelation tracker).

### Polynomial root-finding: hand-rolled Aberth-Ehrlich

For formants we need to find the complex roots of the LPC predictor polynomial (typically degree ~12 for 5 formants). Three options surfaced — hand-rolled Aberth-Ehrlich, `nalgebra` companion-matrix eigenvalues, a third-party root-finder crate. Hand-rolled won because:

- `nalgebra` would add ~50 transitive deps for one polynomial root-find per frame.
- Aberth-Ehrlich is well-known (parallel-Newton method with deflation correction), converges quadratically, and is ~80 LOC.
- We control numerical tolerance and behavior on ill-conditioned polynomials.

The algorithm:
1. Initialize `degree` complex roots evenly spaced on a circle of radius `(max|coeff|)^(1/degree)`.
2. Iteratively update each root `z_j ← z_j - p(z_j) / (p'(z_j) - p(z_j) · Σ_{k≠j} 1/(z_j - z_k))`.
3. Stop when `max|correction| < 1e-7` or 100 iterations.

LPC polynomials from speech are typically well-conditioned, and degree-12 roots converge in ~20 iterations. The module is `engine::dsp::roots`, fully unit-tested against analytical-truth polynomials.

### LPC: autocorrelation method AND Burg's method

Per the method-diversity principle, C2 ships both standard LPC estimators side-by-side:

- `engine::dsp::lpc::autocorr_lpc` — autocorrelation + Levinson-Durbin. Always produces a stable predictor (`|k_i| < 1` for all reflection coefficients) but tapers signal energy at frame edges (effectively assumes zero-extension outside the frame), biasing formant estimates on short frames.
- `engine::dsp::lpc::burg_lpc` — Burg's method. Estimates reflection coefficients directly from forward/backward prediction errors; avoids the autocorrelation method's implicit windowing. Praat's `Sound.to_formant_burg` default.

Dispatcher `lpc(samples, order, LpcMethod)` selects between them. **The formant tracker defaults to `LpcMethod::Burg`** to match Praat's convention; callers can override.

Citations: Makhoul (1975); Markel & Gray (1976); Burg (1975); Levinson (1947); Durbin (1960). Full links in the LPC module docs and in the C2 references list below.

### Formants

Pipeline per frame:
1. Apply pre-emphasis filter `y[n] = x[n] - α·x[n-1]` with `α = 0.97` (standard speech-DSP convention; caller can pass `0.0` to skip).
2. Window the frame (Hann by default).
3. Compute LPC coefficients via the chosen method (Burg by default; autocorrelation available).
5. Find roots of the predictor polynomial `1 + a_1·z⁻¹ + ... + a_p·z⁻ᵖ` (or equivalently `z^p + a_1·z^(p-1) + ... + a_p`).
6. For each root `z = r·e^(jθ)` with `r < 1` (inside unit circle) and `θ > 0` (upper half — complex conjugate pairs):
   - `frequency = θ · sample_rate / (2π)` Hz
   - `bandwidth = -ln(r) · sample_rate / π` Hz
7. Filter to `freq ∈ [50, sample_rate/2 - 50]` Hz and `bandwidth < 1000` Hz.
8. Sort by frequency.

**Default LPC order**: `2 · n_formants + 2`. For `n_formants = 5` (Praat default) → `p = 12`. Caller can override.

**Output shape: variable-length per frame**, not fixed-N. `FormantFrame { time_seconds, frequencies: Vec<f32>, bandwidths: Vec<f32> }`. Honest about frames where the root-finder didn't return enough valid roots in the F1–F<n_formants> range (silence, noise bursts, etc.). A future helper can pad to a `(n_frames, n_formants)` NumPy array with NaN, but C2 ships the list-of-frames form only.

### MFCC

Pipeline:
1. STFT magnitude → power spectrogram (reuse C1's `engine::dsp::stft` + `power_spectrogram`).
2. Apply mel filterbank: `n_mels` triangular filters between `f_min` and `f_max`, spaced uniformly on the mel scale (`m = 2595 · log10(1 + f/700)`).
3. Log of filterbank energies.
4. DCT-II to decorrelate → cepstral coefficients.
5. Keep the first `n_mfcc` coefficients.

DCT-II via direct matrix multiply (precompute the `n_mels × n_mfcc` cosine matrix once per call — trivially small at v1 defaults).

**Defaults** (matching librosa's defaults exactly so users porting code don't see surprises):

| Param | Default | Notes |
|---|---|---|
| Mel scale | **Slaney** | librosa's default (piecewise linear-then-log); HTK toggle deferred to task #55 |
| `n_mfcc` | 13 | Speech-recognition standard |
| `n_mels` | 40 | Phoneme-level resolution |
| `f_min` | 0.0 Hz | |
| `f_max` | `sample_rate / 2` | Nyquist |
| `frame_size_seconds` | 0.025 | 25 ms — standard speech analysis frame |
| `hop_seconds` | 0.010 | 10 ms |

**No** sinusoidal liftering or Δ/Δ² stacking in C2 — those are layered on top later if a real use case appears.

Output shape: `Array2<f32>` in Rust, `np.ndarray[float32, ndim=2]` in Python, shape `(n_frames, n_coeffs)` (frames-first to match librosa).

### Refined pitch: voicing decision on PitchFrame

Phase 0's `engine::pitch::autocorrelation` already computes the autocorrelation peak when finding the lag. Voicing is essentially free:

```
voicing = peak_autocorr_at_period / R(0)
```

`voicing ∈ [0, 1]` — close to 1 for clean voiced speech, near 0 for noise/silence. Threshold at 0.45 (typical literature value) to get a binary voiced flag.

`PitchFrame` grows a `voicing: f32` field. `PitchConfig` grows `voicing_threshold: f32` (default 0.45). Callers can read voicing directly or filter on the threshold.

**Python API**: keep `sadda.dsp.f0(audio, ...) → (times, freqs)` exactly as in Phase 0 (STABLE contract — no breakage). Add a new `sadda.dsp.voiced_pitch(audio, ...) → (times, freqs, voicing)` that returns the same three columns the Rust `PitchFrame` now has. Both functions call the same underlying autocorrelation tracker; they just project different fields.

### Confirmed C2 decisions

| Item | Decision | Reasoning |
|---|---|---|
| LPC methods | **Both autocorrelation (Levinson-Durbin) and Burg shipped side-by-side** | Method-diversity principle; Burg = Praat default for formants; autocorrelation = textbook default; each addresses a different stability/edge-bias trade-off |
| Default LPC method for formants | **Burg** | Praat's `Sound.to_formant_burg` convention; better short-frame behaviour |
| Polynomial root-solver | **Hand-rolled Aberth-Ehrlich (~80 LOC)** | Avoids ~50 `nalgebra` deps; well-known parallel-Newton method; numerical tolerance under our control |
| Formant output shape | **Variable-length `FormantFrame { time_seconds, frequencies, bandwidths }` per frame** | Honest about frames where the root-finder didn't return enough valid roots; Python wrapper can pad to fixed-N later if needed |
| Voicing API | **Add `sadda.dsp.voiced_pitch(...)` returning `(times, freqs, voicing)`; keep `sadda.dsp.f0(...)` returning `(times, freqs)`** | No Phase-0 surface breakage; new function for callers who want voicing; PitchFrame gains voicing field internally |
| LPC order default | **`2 · n_formants + 2`** | Standard speech-DSP rule of thumb; matches Praat's `Sound.to_formant_burg` default |
| Pre-emphasis | **Applied internally with `α = 0.97`** | Standard pipeline; caller can pass `0.0` to skip |
| MFCC defaults | **`n_mfcc=13`, `n_mels=40`, `f_min=0`, `f_max=sr/2`, 25 ms frame, 10 ms hop, Slaney mel scale** | Matches librosa's defaults exactly so ported code reproduces; HTK mel-scale toggle is a deferred alternate (task #55) |
| MFCC orientation | **`(n_frames, n_coeffs)`** | Frames-first, matching librosa; symmetric with spectrogram's `(n_freq_bins, n_frames)` only in being row-major |
| Citations | **Doc-comment citation required on every DSP function** | Per the method-diversity principle entry |

### Layout

- `crates/engine/src/dsp/lpc.rs` — Levinson-Durbin recursion; returns LPC coefficients + reflection coeffs + prediction gain.
- `crates/engine/src/dsp/roots.rs` — Aberth-Ehrlich polynomial root-solver.
- `crates/engine/src/dsp/formants.rs` — frame loop, pre-emphasis, root→formant conversion, filtering.
- `crates/engine/src/dsp/mfcc.rs` — mel-scale conversion, triangular filterbank, DCT-II matrix multiply.
- `crates/engine/src/pitch.rs` — extends `PitchFrame` with `voicing`; adds `PitchConfig.voicing_threshold`.
- `crates/python/src/lib.rs` — PyO3 wrappers (`formants`, `mfcc`, `voiced_pitch`); new `PyFormantFrame` data class.
- `python/sadda/dsp/__init__.py` — re-exports with `@stable`.
- `crates/engine/tests/advanced_dsp.rs` — analytical-truth tests against synthetic vowels and known polynomials.
- `python/tests/test_advanced_dsp.py` — end-to-end Python tests.

### Recommended defaults for v1

Per the method-diversity principle:

- **LPC** (general use): **Burg's method** — Praat's formant-tracker convention; better short-frame behaviour. Autocorrelation method available for textbook parity.
- **Polynomial root-find**: Aberth-Ehrlich. Robust and standalone (no `nalgebra` dep); converges in fewer iterations than companion-matrix QR for the degree-12-ish polynomials LPC produces.
- **Formant tracker**: LPC-Burg + Aberth-Ehrlich roots + freq/bw conversion. Pre-emphasis α=0.97. Praat-baseline behaviour.
- **MFCC**: Slaney mel scale (librosa default), n_mels=40, n_mfcc=13, 25 ms frame, 10 ms hop, HTK-style log + DCT-II. (HTK mel-scale toggle is a deferred alternate.)
- **Pitch**: `windowed_autocorrelation` — the window-corrected method described above. Strict improvement on Phase 0's naive autocorrelation. Naive `autocorrelation` retained for back-compat with `sadda.f0`.

### Deferred alternates

Tracked per the method-diversity principle; each has a corresponding tracking task:

- **Faithful Boersma 1993 pitch** (task #51) — full Praat pipeline: anti-alias upsample + Gaussian-window option + windowed-sinc + Brent's method peak refinement + multi-candidate Viterbi path-finder + octave/voiced-unvoiced cost terms. The renamed `windowed_autocorrelation` in C2 adopts only Boersma's central window-correction insight; this task tracks the rest.
- **YIN / pYIN / SWIPE' / CREPE pitch trackers** (task #52). pYIN is librosa's modern default; CREPE is the neural SOTA.
- **DeepFormants + QCP-FB formant tracker** (task #53) — Alku et al. 2023, <https://doi.org/10.1016/j.csl.2023.101515>. Modern accuracy upgrade beyond LPC+roots.
- **Noise-robust LPC** (task #54): QCP-FB (Airaksinen 2014) and Burg-robust (Praat's `Sound.to_formant_burg_robust`).
- **PNCC + Slaney mel + HTK toggle for MFCC** (task #55). PNCC (Kim & Stern 2016) is noise-robust; Slaney/HTK toggle for MATLAB/Kaldi parity.
- **Multitaper + reassigned STFT** (task #56) — Babadi & Brown 2014; Auger et al. 2013.
- **Magnitude / log-power spectrogram + periodic-window flag** (task #57).
- **LUFS loudness + calibrated dB-SPL intensity** (task #58).
- **Formant trajectory smoothing** (Viterbi / DP continuity) — C2 ships per-frame independent root-finding.
- **Fixed-N dense formant array helper** with NaN padding for missing formants.
- **MFCC Δ / Δ² stacking and sinusoidal liftering**.
- **Sub-sample lag interpolation** in the naive autocorrelation pitch tracker (the `windowed_autocorrelation` method already does parabolic interpolation; a future slice can add it to the naive tracker too).

### What this entry doesn't decide

- **Move `engine::pitch` under `engine::dsp::pitch`.** Still deferred (same rationale as C1: keep the slice diff additive).

### Sources / references

- 2026-05-21 DSP method-diversity principle entry (above this one)
- 2026-05-21 C1 entry (foundational DSP this builds on)
- 2026-05-21 Phase 1 slicing entry (C2 row)
- **LPC, autocorrelation method**: Makhoul, J. (1975), "Linear prediction: A tutorial review." *Proc. IEEE* 63(4). https://doi.org/10.1109/PROC.1975.9792
- **LPC, Burg's method**: Burg, J.P. (1975), *Maximum Entropy Spectral Analysis*. PhD thesis, Stanford. https://sepwww.stanford.edu/data/media/public/oldreports/sep06/
- **Levinson-Durbin**: Levinson, N. (1947), https://doi.org/10.1002/sapm1946251261 ; Durbin, J. (1960), https://doi.org/10.2307/1401322
- **Markel & Gray** (1976), *Linear Prediction of Speech*, Springer. ISBN 978-3-642-66288-1
- **Formant tracker (Praat-like, autocorrelation + roots)**: McCandless, S.S. (1974), "An algorithm for automatic formant extraction using linear prediction spectra." *IEEE TASSP* 22(2). https://doi.org/10.1109/TASSP.1974.1162572
- **Aberth's method**: Aberth, O. (1973), "Iteration methods for finding all zeros of a polynomial simultaneously." *Math. Comp.* 27(122). https://doi.org/10.1090/S0025-5718-1973-0329236-7
- **Aberth/Ehrlich numerical analysis**: Bini, D.A. (1996), "Numerical computation of polynomial zeros by means of Aberth's method." *Numerical Algorithms* 13. https://doi.org/10.1007/BF02207694
- **MFCC**: Davis, S.B. & Mermelstein, P. (1980), "Comparison of parametric representations for monosyllabic word recognition." *IEEE TASSP* 28(4). https://doi.org/10.1109/TASSP.1980.1163420
- **Voicing via autocorrelation peak ratio**: Boersma, P. (1993), "Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound." *Proc. Inst. Phonetic Sciences* 17. https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf
- Praat formant defaults: https://www.fon.hum.uva.nl/praat/manual/Sound__To_Formant__burg____.html
- librosa.feature.mfcc: https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html

---

## 2026-05-21 — Foundational DSP (C1): sadda.dsp namespace introduced, rustfft+realfft, intensity = RMS + dB-FS

Goal: settle the sixth Phase 1 slice. C1 introduces the `sadda.dsp.*` namespace and lands the foundational DSP toolkit — windowing functions, STFT, spectrogram, intensity — as pure functions over `&[f32]` with no corpus coupling. The slice's exports become the first STABLE-tier members of `sadda.dsp.*` per the 2026-05-18 Python API surface entry.

### What C1 must deliver

From the Phase-1 slicing entry: (1) windowing functions (Hann, Hamming, Blackman, Gaussian, Kaiser); (2) STFT; (3) spectrogram; (4) intensity; (5) pure functions over `&[f32]`, no corpus dependency; (6) Polars-friendly outputs.

### Namespace: introduce `sadda.dsp.*` now; move `sadda.f0` → `sadda.dsp.f0`

The 2026-05-18 API surface entry pinned `sadda.dsp` as a STABLE namespace. C1 adds 8+ new public symbols — keeping them at the top level would clutter `sadda.*` for every future caller. So C1 is the natural slice to introduce the namespace.

The Phase-0 `sadda.f0(...)` moves to `sadda.dsp.f0(...)` to live alongside the rest of the DSP toolkit. A top-level `sadda.f0` alias stays in place for back-compat — both call paths reach the same underlying function. The alias is documented as a Phase-0 convenience and is part of the STABLE contract (not deprecated; users won't see a warning).

`sadda.Audio`, `sadda.new_project`, `sadda.open_project`, the tier types, etc. stay at the top level; only DSP gets a sub-module in C1.

### FFT: `rustfft` + `realfft`

`rustfft` 6.4 is the canonical Rust FFT crate — pure Rust, no_std-capable, MSRV 1.61 (fine for our 1.85). `realfft` 3.5 wraps it for real-only input (saves roughly half the work for audio, which is always real-valued). Both have minimal feature footprints; default features turn on SIMD intrinsics (AVX/SSE/NEON) that no-op on platforms without them.

### Module layout (Rust)

New `engine::dsp` module, split into focused sub-modules:

```
crates/engine/src/dsp/
├── mod.rs             — re-exports + module docs
├── windowing.rs       — hann, hamming, blackman, gaussian(n, sigma), kaiser(n, beta)
├── stft.rs            — stft(samples, window, hop) → (Vec<Complex<f32>>, Shape)
├── spectrogram.rs     — power_spectrogram(stft, shape) → Vec<f32> (n_freq_bins * n_frames)
└── intensity.rs       — intensity(samples, sample_rate, frame_size_seconds, hop_seconds) → Vec<IntensityFrame>
```

Pure functions over `&[f32]`; no `Project` coupling — unit-testable in isolation just like `engine::storage::dense`. The existing `engine::pitch` (autocorrelation f0) is logically a DSP module too, but stays where it is to keep C1's diff focused on the new surfaces; a follow-up may move it under `engine::dsp::pitch`.

### Window functions

All five return `Vec<f32>` of the requested length. Parameterized windows take their parameter explicitly:

```rust
pub fn hann(n: usize)     -> Vec<f32>;
pub fn hamming(n: usize)  -> Vec<f32>;
pub fn blackman(n: usize) -> Vec<f32>;
pub fn gaussian(n: usize, sigma: f32) -> Vec<f32>;
pub fn kaiser(n: usize, beta: f32)    -> Vec<f32>;
```

Forces callers to declare params (matches `scipy.signal.windows` convention); avoids hiding strong opinions like "Praat-default Kaiser β = 8.6" in the API. Callers compose via:

```rust
let win = sadda_engine::dsp::hann(frame_size);
let windowed: Vec<f32> = samples.iter().zip(win.iter()).map(|(x, w)| x * w).collect();
```

### STFT signature

```rust
pub struct Shape { pub n_frames: usize, pub n_freq_bins: usize }

pub fn stft(samples: &[f32], window: &[f32], hop_size: usize)
    -> (Vec<Complex<f32>>, Shape);
```

Returns the row-major flattened matrix shape `(n_frames, n_freq_bins)`. The companion `Shape` is the cheap structural metadata. Real-input optimized via `realfft::RealFftPlanner::plan_fft_forward(window.len())`. `n_freq_bins = window.len() / 2 + 1` (the unique part of the spectrum for real input).

### Spectrogram

Magnitude-squared (power) of the STFT, real-valued. Shape `(n_freq_bins, n_frames)` matches the API surface entry's documented convention; this is row-major-transposed relative to STFT's internal `(n_frames, n_freq_bins)` to be polars-friendly when each frequency bin becomes a column-like axis.

```rust
pub fn power_spectrogram(stft_out: &[Complex<f32>], shape: Shape)
    -> Vec<f32>;  // length n_freq_bins * n_frames, row-major (n_freq_bins, n_frames)
```

A `magnitude_spectrogram(...)` follow-up can layer in later if a user wants the `|X|` form instead of `|X|²`.

### Intensity: RMS + dB-FS per frame

```rust
pub struct IntensityFrame {
    pub time_seconds: f64,
    pub rms: f32,        // linear amplitude, root-mean-square over the frame
    pub db_fs: f32,      // 20 * log10(rms / 1.0); dB relative to full-scale [-1.0, 1.0]
}

pub fn intensity(
    samples: &[f32], sample_rate: u32,
    frame_size_seconds: f32, hop_seconds: f32,
) -> Vec<IntensityFrame>;
```

Both forms in one frame struct: linear RMS for raw analysis, dB-FS as the calibration-free dB form (relative to digital full-scale at amplitude 1.0). dB-SPL (relative to 2·10⁻⁵ Pa, the Praat convention) is deferred to a later slice that plumbs microphone calibration through the `Instrument` table.

Edge case: silent frames (RMS = 0) produce `db_fs = -∞`. The Rust implementation clamps at a small floor (e.g. `-200.0` dB) to keep downstream callers from having to special-case `NEG_INFINITY`.

### Python surface

`python/sadda/dsp/__init__.py` becomes the public entry point:

```
sadda.dsp.hann(n)               → np.ndarray[float32]
sadda.dsp.hamming(n)            → np.ndarray[float32]
sadda.dsp.blackman(n)           → np.ndarray[float32]
sadda.dsp.gaussian(n, sigma)    → np.ndarray[float32]
sadda.dsp.kaiser(n, beta)       → np.ndarray[float32]
sadda.dsp.stft(samples, frame_size, hop_size, *, window=None)
                                → tuple[np.ndarray[complex64, 2], (n_frames, n_freq_bins)]
sadda.dsp.spectrogram(samples, frame_size, hop_size, *, window=None)
                                → np.ndarray[float32, 2]   # shape (n_freq_bins, n_frames)
sadda.dsp.intensity(audio, *, frame_size_seconds=0.030, hop_seconds=0.010)
                                → tuple[np.ndarray[float64], np.ndarray[float32], np.ndarray[float32]]
                                  # (times, rms, db_fs)
sadda.dsp.f0(audio, ...)         # the existing Phase-0 function, relocated
```

Top-level `sadda.f0` stays as a back-compat alias pointing at the same function. All `sadda.dsp.*` symbols are `@stable`.

The DSP submodule is implemented in pure Python (`python/sadda/dsp/__init__.py`) re-exporting from `_native`; no PyO3 submodule machinery (which would complicate the stub layout). The Rust extension exposes all DSP functions flat in `sadda._native` (e.g. `_native.hann`, `_native.stft`, …); the Python wrapper does the namespacing.

### Confirmed C1 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Namespace | **Introduce `sadda.dsp.*` now; move `sadda.f0` → `sadda.dsp.f0` with top-level alias** | Cleanest namespace from day one; 8+ DSP symbols would clutter top-level; API-surface entry already pinned the namespace; alias keeps existing users working |
| FFT library | **`rustfft` 6 + `realfft` 3** | Canonical Rust FFT pair; pure Rust; MSRV 1.61; SIMD by default; minimal extra deps |
| Window API | **Return `Vec<f32>`; explicit per-param args** | Composable; no hidden defaults; matches scipy convention; testable per param |
| Intensity form | **`IntensityFrame { rms, db_fs, time_seconds }`** | Both calibration-free forms in one struct; dB-SPL deferred until Instrument calibration lands |
| Spectrogram orientation | **`(n_freq_bins, n_frames)`** | Matches the 2026-05-18 API-surface entry's documented convention; polars-friendly |
| Polars integration | **Returns NumPy; Python wraps with polars if desired** | Same pattern as B2/B3 — no polars-rs in the Rust tree |

### Layout

- `crates/engine/Cargo.toml` — adds `rustfft = "6"` and `realfft = "3"`.
- `crates/engine/src/dsp/{mod,windowing,stft,spectrogram,intensity}.rs` — new module with focused submodules.
- `crates/engine/src/lib.rs` — `pub mod dsp;` + re-exports.
- `crates/python/src/lib.rs` — new PyO3 functions for windowing/stft/spectrogram/intensity (the f0 binding already exists).
- `python/sadda/dsp/__init__.py` — re-exports + `@stable` decoration; pulls `f0` from `_native` too.
- `python/sadda/__init__.py` — keeps `f0` as the back-compat alias.
- `crates/engine/tests/dsp.rs` — round-trips and analytical-truth tests (sine → peak in spectrogram, RMS of known signal matches).
- `python/tests/test_dsp.py` — NumPy-side smoke tests + namespace presence + alias equality.

### Recommended defaults for v1

Per the 2026-05-21 method-diversity principle entry, each module names its v1 default explicitly:

- **Windowing**: Hann for general use (good main-lobe/side-lobe tradeoff, low scalloping).
- **STFT**: standard Gabor STFT (Hann window, hop ≤ window/4).
- **Spectrogram**: power (`|X|²`) for downstream computation; log-power for visualisation. C1 ships power only; log-power lands later (task #57).
- **Intensity**: linear RMS + dB-FS together per frame, 30 ms Hann frame, 10 ms hop.

### Deferred alternates

Per the method-diversity principle, the running development-target list for C1:

- **Periodic-window flag** (`sym=False`) for COLA-correct STFT overlap-add (task #57).
- **Multitaper STFT** via DPSS/Slepian tapers — Thomson 1982; review by Babadi & Brown 2014, <https://doi.org/10.1109/TBME.2014.2311996>. Lower-variance estimates, useful for HNR. (task #56)
- **Reassignment / synchrosqueezing STFT** — Auger et al. 2013, <https://doi.org/10.1109/MSP.2013.2265316>. Sharper time-frequency localization. (task #56)
- **Magnitude spectrogram** (`|X|`) and **log-power spectrogram** (`10·log10(|X|²)`) variants. (task #57)
- **LUFS / ITU-R BS.1770 K-weighted loudness** — <https://www.itu.int/rec/R-REC-BS.1770>. Broadcast-standard perceptual loudness. (task #58)
- **Calibrated dB-SPL** intensity once `Instrument` calibration is plumbed through. (task #58)
- **LazyFrame-style streaming STFT** for long recordings (live-recording slice E1 will revisit).
- **Move `engine::pitch` under `engine::dsp::pitch`** (cosmetic; deferred to a future reorganization slice).

### Sources / references

- 2026-05-18 Python API surface entry (`sadda.dsp` namespace, spectrogram shape convention)
- 2026-05-21 Phase 1 slicing entry (C1 row)
- `rustfft`: https://github.com/ejmahler/RustFFT (6.4.1, MSRV 1.61)
- `realfft`: https://github.com/HEnquist/realfft (3.5.0)
- scipy.signal.windows reference: https://docs.scipy.org/doc/scipy/reference/signal.windows.html
- librosa STFT/spectrogram reference: https://librosa.org/doc/latest/generated/librosa.stft.html

---

## 2026-05-21 — Dense tier types + Parquet sidecars (B3): apache parquet+arrow, per-bundle layout, NumPy/buffers in, polars out

Goal: settle the fifth Phase 1 slice. B3 lands the three dense tier types' on-disk format (Parquet sidecars under `signals/derived/`), the `DerivedSignal` registration table that ties them back to the corpus, and the read/write paths so AI-engineer users can either ask for a `polars.DataFrame` via `proj.query(tier_id)` or grab the sidecar path and `pl.scan_parquet(path)` directly.

### What B3 must deliver

From the Phase-1 slicing entry: (1) the three dense tier types — `continuous_numeric` / `continuous_vector` / `categorical_sampled`; (2) `DerivedSignal` registration rows; (3) reader/writer in `engine::storage::parquet`; (4) mmap-friendly load path so external readers work.

### Parquet via Apache `parquet` + `arrow` (version 58)

The corpus data-model entry pinned Parquet as the storage format. For Rust, the Apache `parquet` and `arrow` crates (version-locked) are the canonical pair — they're what `polars-rs` uses under the hood, and the resulting files are bit-for-bit consumable by `polars.scan_parquet(...)` / `pyarrow.parquet.read_table(...)` / `pandas.read_parquet(...)`. Verified compatibility with our `edition = "2024"` + `rust-version = "1.85"` workspace (both crates declare exactly those in their workspace package).

Minimal feature set to keep the dep tree from ballooning:

```toml
parquet = { version = "58", default-features = false, features = ["arrow", "snap"] }
arrow   = { version = "58", default-features = false }
```

- `parquet` defaults include `brotli`/`base64`/`simdutf8`/etc. — dropped; we only need Snappy (Parquet's most common codec).
- `arrow` defaults include `csv`/`ipc`/`json` — dropped; the Float64 / FixedSizeList / Utf8 types live in the always-on core.
- `ndarray` is added as a separate dep for the `Array2<f64>` ergonomics on the Rust side; arrow has no `ndarray` feature, so we marshal manually via `Float64Array::from(arr.as_standard_layout().as_slice().unwrap().to_vec())`.

### Sidecar layout: per-bundle subdirectories

`signals/derived/bundle_<id>/<tier_name>.parquet`. All sidecars for a bundle group together — easy to `ls` by hand, no branching on whether the bundle has a session, and rename-safe at the file level (renaming a tier renames the file via the engine; external readers reference DerivedSignal.relative_path). The original data-model entry sketched `signals/derived/session_<id>/bundle_<id>.<name>.parquet` but that nesting is awkward when bundles have no session.

### DerivedSignal table (V5)

V5 migration adds:

```sql
CREATE TABLE derived_signal (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tier_id         INTEGER NOT NULL UNIQUE REFERENCES tier(id),
    relative_path   TEXT    NOT NULL,
    n_frames        INTEGER NOT NULL,
    n_dims          INTEGER NOT NULL DEFAULT 1,
    sample_rate_hz  REAL,
    dtype           TEXT    NOT NULL,
    extra           TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX idx_derived_signal_tier ON derived_signal(tier_id);
```

- `tier_id` UNIQUE: one sidecar per tier (writes after the first error; rewrites land in a follow-up).
- `n_dims = 1` for `continuous_numeric` and `categorical_sampled`; `n_dims >= 1` for `continuous_vector`.
- `sample_rate_hz` NULL for non-sampled / variable-rate signals (none in v1, but the column is forward-compat).
- `dtype` enum stored as text: `f64`, `f32`, `utf8` for v1.
- Audited per the B1 trigger-rebuild discipline (3 triggers added in V5).

### Write API: positional buffers in Rust, NumPy in Python

Three options surfaced — positional buffers (Rust `&[f64]` / `Array2<f64>` / `&[String]`), DataFrame-typed inputs (requires `polars-rs` + `pyo3-polars`), or both. Positional buffers won because:
- No new Rust deps beyond `parquet`+`arrow` (already heavy enough).
- Clean Rust signatures; no DataFrame ceremony at internal call sites.
- The read path can still return a DataFrame; symmetry isn't required.
- DataFrame inputs land in a later ergonomic-pass slice if real users want bulk writes from polars.

Rust signatures:

```rust
impl Project {
    pub fn write_continuous_numeric(
        &self, tier_id: i64,
        samples: &[f64], sample_rate_hz: f64,
    ) -> Result<i64>; // returns derived_signal.id

    pub fn write_continuous_vector(
        &self, tier_id: i64,
        frames: ndarray::ArrayView2<'_, f64>, sample_rate_hz: f64,
    ) -> Result<i64>;

    pub fn write_categorical_sampled(
        &self, tier_id: i64,
        labels: &[String], sample_rate_hz: f64,
    ) -> Result<i64>;

    pub fn read_continuous_numeric(&self, tier_id: i64) -> Result<Vec<f64>>;
    pub fn read_continuous_vector(&self, tier_id: i64) -> Result<ndarray::Array2<f64>>;
    pub fn read_categorical_sampled(&self, tier_id: i64) -> Result<Vec<String>>;

    pub fn derived_signal(&self, tier_id: i64) -> Result<Option<DerivedSignal>>;
    pub fn dense_path(&self, tier_id: i64) -> Result<Option<PathBuf>>;
}
```

The write methods reject if the tier's type doesn't match, if a `derived_signal` row already exists for the tier (no overwrite in v1), and if the input buffer is empty.

### Python: NumPy in, polars out

- `proj.write_continuous_numeric(tier_id, np.ndarray[float64], sample_rate_hz)`
- `proj.write_continuous_vector(tier_id, np.ndarray[float64, ndim=2], sample_rate_hz)`
- `proj.write_categorical_sampled(tier_id, list[str], sample_rate_hz)`
- `proj.read_continuous_numeric(tier_id) → np.ndarray[float64]`
- `proj.read_continuous_vector(tier_id) → np.ndarray[float64, ndim=2]`
- `proj.read_categorical_sampled(tier_id) → list[str]`
- `proj.dense_path(tier_id) → str | None` — for `pl.scan_parquet(path)` external reads
- `proj.derived_signal(tier_id) → DerivedSignal | None` — the registration row
- `proj.query(tier_id)` (the B2 monkey-patch) extended: for dense tiers, calls `pl.read_parquet(proj.dense_path(tier_id))` and returns the resulting DataFrame; for sparse, the B2 dispatch.

### `categorical_sampled` encoding

Plain UTF8 column for v1; Parquet dictionary encoding is an optimization for later. Reasoning: dictionary encoding shrinks VAD/voicing-class files ~3× but adds slightly more arrow-rs API surface and has marginal compatibility risk with older Parquet readers; v1 corpora are small enough that the disk delta doesn't matter. The polars `.cast(pl.Categorical)` escape hatch handles in-memory categorization on read.

### Confirmed B3 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Parquet library | **Apache `parquet` + `arrow` 58 (minimal features)** | Canonical Rust→Parquet path; bit-for-bit compatible with polars/pyarrow consumers; only viable choice without forking `polars-rs` |
| Sidecar layout | **`signals/derived/bundle_<id>/<tier_name>.parquet`** | All sidecars for a bundle grouped; no session-presence branching; rename-safe at the file level |
| `DerivedSignal` shape | **`tier_id UNIQUE` + path + n_frames/n_dims/sample_rate_hz/dtype + extra** | One sidecar per tier in v1; extension columns cover the queries the read path needs without parsing Parquet metadata |
| Write API | **Positional buffers (Rust slices/ndarray; NumPy in Python)** | No `polars-rs` dep; clean signatures; symmetric read path can still return DataFrames |
| `categorical_sampled` encoding | **Plain UTF8; dictionary as a later optimization** | Simpler write path; broad compatibility; size delta acceptable at v1 corpus scale |
| `proj.query(tier_id)` for dense tiers | **`pl.read_parquet(path)`** | Reuses the B2 monkey-patch; matches AI-engineer expectations; LazyFrame variant can layer in later |

### Layout

- `crates/engine/migrations/V5__derived_signal.sql` — table + index + 3 audit triggers.
- `crates/engine/src/storage/mod.rs` + `crates/engine/src/storage/parquet.rs` — new module owning `arrow`/`parquet` boilerplate; pure Rust, no `Project` dep, so it's unit-testable in isolation.
- `crates/engine/src/corpus.rs` — `DerivedSignal` struct; `Project::write_*` / `read_*` / `derived_signal` / `dense_path` methods.
- `crates/python/src/lib.rs` — PyDerivedSignal class; numpy-bridging write/read methods on PyProject; `dense_path` returning Option<str>.
- `python/sadda/__init__.py` — extend the B2 `proj.query` monkey-patch to dispatch on dense tier types via `pl.read_parquet(...)`.
- `crates/engine/tests/dense_tiers.rs` + `python/tests/test_dense_tiers.py` — round-trip + interop tests.

### What this entry doesn't decide

- **Rewrite / append to existing sidecars.** v1: one-shot write per tier; rewrites error. A follow-up slice may add `proj.replace_dense(...)` once a clear use case appears.
- **Streaming writes (chunk-by-chunk).** v1 writes the whole buffer in one call. Streaming arrives with live recording (E1) if needed for long sessions.
- **LazyFrame from `proj.query`.** Materializing the DataFrame is fine at v1 scale (≤ ~1 GB sidecars for the typical user). A `proj.scan(tier_id) → pl.LazyFrame` can layer in later for the ML / large-embedding case.
- **Mixed-type vector columns.** All `continuous_vector` columns are Float64 at v1; Float32 + Int variants land when a real consumer needs them.
- **Audit log of Parquet contents.** The DerivedSignal row is audited; the Parquet *file contents* are not — they're write-once column blobs and the audit trail lives at the registration boundary.
- **Symlink / cache eviction policy.** Sidecars stay until the bundle is deleted (cascade from `bundle` to `tier` to `derived_signal` is a future migration if we add `ON DELETE CASCADE` — not in V5).

### Sources / references

- 2026-05-18 corpus data-model entry (Parquet pinning, sparse/dense split)
- 2026-05-21 Phase 1 slicing entry (B3 row)
- 2026-05-21 B2 entry (sparse tier types this entry parallels)
- Apache arrow-rs: https://github.com/apache/arrow-rs (parquet 58, arrow 58)
- `parquet::arrow::ArrowWriter`: https://docs.rs/parquet/58.3.0/parquet/arrow/index.html
- `FixedSizeListArray`: https://docs.rs/arrow-array/58.3.0/arrow_array/array/struct.FixedSizeListArray.html

---

## 2026-05-21 — Sparse tier types (B2): three annotation tables, Rust-level cardinality, polars wrap in Python

Goal: settle the fourth Phase 1 slice. B2 puts annotation rows into the three sparse-tier tables, wires tier-header + annotation CRUD into the Project API, enforces parent-child cardinality at insert time, and ships the first cut of `proj.query(...) → polars.DataFrame`.

### What B2 must deliver

From the Phase-1 slicing entry: (1) `interval`, `point`, `reference` tier types with CRUD; (2) first cut of `proj.query(...) → polars.DataFrame`; (3) parent-child cardinality enforced at insert time.

### Three separate annotation tables

The 2026-05-18 corpus data-model entry already pinned three per-type tables (versus a single discriminated annotation table). Schemas:

```sql
annotation_interval (
    id, tier_id, start_seconds REAL, end_seconds REAL,
    label, parent_annotation_id, extra,
    CHECK (end_seconds > start_seconds)
);
annotation_point (
    id, tier_id, time_seconds REAL, label, parent_annotation_id, extra
);
annotation_reference (
    id, tier_id, target_kind TEXT, target_id INTEGER,
    label, parent_annotation_id, extra,
    CHECK (target_kind IN ('bundle','session','speaker','tier','annotation'))
);
```

All three are audited per the B1 trigger-rebuild discipline; V4 includes 9 new audit triggers.

### Polars integration: Python-side wrap

Three options surfaced — Python-side wrap of Rust-supplied rows, `polars-rs` + `pyo3-polars` in Rust, or LazyFrame via path scan. Python-side wrap won because:
- No new Rust deps. `polars-rs` pulls ~50 transitive crates and lengthens cold builds substantially; not worth it for sparse-tier scale (≤ ~100K rows per project).
- The Arrow-zero-copy story matters mostly for *dense* tiers (continuous_vector embeddings, continuous_numeric tracks). B3 introduces Parquet sidecars where Arrow buffers are already the native format; that's the natural place to add `polars-rs` if needed.
- The Python wrapper is trivial: Rust returns `Vec<row tuples>`, `__init__.py` calls `polars.DataFrame(rows, schema=...)` with a tier-type-specific column shape.

`polars` joins `numpy` in the runtime deps in `pyproject.toml`.

### Cardinality enforcement: engine-level (Rust)

SQL triggers can't easily express "the parent_annotation_id must reference the correct one of three possible parent annotation tables, chosen by the parent tier's type." Rust-level enforcement at insert time is straightforward, gives clear error messages, and avoids trigger debugging when something goes wrong.

The check in `Project::add_interval` (and the analogous point/reference methods):

```
let tier = self.get_tier(tier_id)?;
match tier.parent_id {
    None => { /* no parent: no check */ }
    Some(parent_tier_id) => {
        let parent_annotation_id = spec.parent_annotation_id
            .ok_or_else(|| EngineError::Cardinality(
                "child tier requires parent_annotation_id".into()))?;
        // Verify parent annotation exists in the parent tier's right table.
        ensure_parent_exists(self, parent_tier_id, parent_annotation_id)?;
        match tier.cardinality.as_deref() {
            Some("one_to_one") => ensure_parent_is_unique(...)?,
            Some("one_to_many") | Some("none") | None => {}
            Some("many_to_one") => return Err(EngineError::Cardinality(
                "many_to_one cardinality is not supported until B-cluster follow-up".into())),
            Some(other) => return Err(...),
        }
    }
}
```

New `EngineError::Cardinality(String)` variant covers all the failure modes — clearer than reusing `Corpus(String)`.

### Many-to-one deferred

The v1 use cases all naturally fit `one_to_many` (word→phones, syllable→phones, …). `many_to_one` (multiple parents per child) is inherently a join table the rest of the model doesn't have. V4 keeps the cardinality enum's `many_to_one` value (the V3 CHECK already lists it), but `add_interval` / `add_point` / `add_reference` reject it with a clear "not supported until follow-up" message. A future B-cluster slice can add the join table when a real use case surfaces.

### Confirmed B2 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Annotation tables | **Three: `annotation_interval`, `annotation_point`, `annotation_reference`** | Already pinned by the corpus data-model entry; per-type tables keep columns typed and queries direct |
| Polars | **Python-side wrap; `polars-rs` deferred to B3** | No new Rust deps; cheap to swap when Parquet sidecars need Arrow zero-copy |
| Cardinality enforcement | **Engine-level Rust check at insert time** | SQL triggers can't dispatch on parent tier's type cleanly; Rust gives clear error messages |
| `many_to_one` | **Deferred** | No v1 use case requires it; the join table can land when a use case appears |
| API surface | **Both DataFrame and typed accessors** | `proj.query(tier_id)` → polars.DataFrame for AI-engineer ergonomics; `proj.intervals(tier_id)` → `list[Interval]` for OO callers and tests |

### Layout

- `crates/engine/migrations/V4__sparse_annotations.sql` — three tables + indexes + 9 audit triggers.
- `crates/engine/src/corpus.rs` — `Tier` / `Interval` / `Point` / `Reference` structs + their `*Spec` builders; `add_tier`, `tiers`, `get_tier`, `add_interval`, `intervals`, `add_point`, `points`, `add_reference`, `references_for` (avoiding the `references` reserved-ish name), plus a Rust helper `query_tier_rows(tier_id) → Vec<RowTuple>` for the Python query wrapper.
- `crates/engine/src/error.rs` — adds `EngineError::Cardinality(String)`.
- `crates/python/src/lib.rs` — PyTier / PyInterval / PyPoint / PyReference + insert/list bindings; raw query method returns Python tuples.
- `python/sadda/__init__.py` — wires `Project.query(tier_id)` (Python method patched onto the Rust class) to call the raw method and build a `polars.DataFrame` with tier-type-aware column schema.
- `pyproject.toml` — adds `polars>=1.0` to runtime deps.

### Trigger-rebuild discipline (reminder)

V4 adds three new audited tables — annotation_interval, annotation_point, annotation_reference. The audit triggers are created in V4 alongside the tables. Any future ALTER TABLE on these tables must DROP+CREATE the triggers per the B1 entry's rule.

### What this entry doesn't decide

- **Dense tier CRUD + Parquet sidecars.** That's B3.
- **Cross-bundle query language.** `proj.query(tier_id)` is the first cut; richer filtering (`proj.query(tier_name="phones", bundles=...)`, EMU-EQL-style traversal) is a later API slice.
- **`many_to_one` join table.** Deferred until a use case appears.
- **Polars schema-typed inserts.** B2 returns DataFrames *from* the corpus; inserting *from* a DataFrame (bulk write of annotations) is a later ergonomic.
- **TextGrid/EAF I/O.** That's D1 / D2 and consumes B2's tier types.

### Sources / references

- 2026-05-18 corpus data-model entry (sparse tier storage decision)
- 2026-05-18 Python API surface entry (polars as primary query-result type)
- 2026-05-21 Phase 1 slicing entry (B2 row)
- 2026-05-21 B1 entry (trigger-rebuild discipline)
- EMU-SDMS level/segment model: https://ips-lmu.github.io/EMU.html
- polars Python API: https://docs.pola.rs/api/python/stable/

---

## 2026-05-21 — Full entity schema + AuditLog (B1): triggers + JSON1, Speaker/Session API only, schema-only for the rest

Goal: settle the third Phase 1 slice. B1 lays down the SQLite-side scaffolding for the v1 entity model (Speaker, Session, Bundle-extension, Tier-header, Entity, EntityRef, Instrument, Protocol, ProcessingRun, AuditLog) and the trigger-based audit infrastructure. Annotation CRUD (B2) and Parquet sidecars (B3) build on top of this schema; F1's recipes write `processing_run` rows.

### What B1 must deliver

From the Phase-1 slicing entry: (1) the schema for nine new entity tables; (2) `extra: json` columns throughout; (3) append-only AuditLog with mutation triggers; (4) ProcessingRun (the renamed ModelRun per the 2026-05-20 ML-registry entry).

### Scope: schema everywhere, API for Speaker + Session + Bundle only

The schema lands in full so subsequent slices (B2, B3, F1, …) build on the same migration. The Rust + Python API surface is intentionally narrower in B1:

| Table | Schema in V3? | Public API in B1? | First public-API slice |
|---|---|---|---|
| `speaker` | ✅ | ✅ | B1 |
| `session` | ✅ | ✅ | B1 |
| `bundle` (extended) | ✅ | ✅ (optional `session_id` + `speaker_id` args) | B1 |
| `instrument` | ✅ | — | B2 or later |
| `protocol` | ✅ | — | E1 (live recording) or experiments slice |
| `entity` | ✅ | — | profile-driven; later |
| `entity_ref` | ✅ | — | B2 (when ref-tier annotations land) |
| `tier` (header) | ✅ | — | B2 (when annotation rows land) |
| `processing_run` | ✅ | — | F1 (recipe.record) |
| `audit_log` + `_audit_context` | ✅ | minimal: `proj.set_audit_user(name)` | B1 |

The narrower API surface keeps B1's commit footprint near the Phase-1 cadence target (~750–1000 LOC). Future slices wire each table into Python as their first real user appears, with no further schema work needed.

### AuditLog: SQLite triggers + `_audit_context` singleton

Three options surfaced — application-level (each engine mutation INSERTs its own audit row), triggers (DB enforces; can't be bypassed), or a hybrid. Triggers won because the regulatory-stance entry calls for "every analysis step recorded" — application-level logging is one direct `INSERT` from a forgotten audit row, and any future plugin / SQL CLI usage bypasses it entirely. Triggers fire regardless of caller.

User attribution: a singleton `_audit_context` row holds the current user; triggers read it (`(SELECT user FROM _audit_context)`); the Rust API sets it on connection (`Project::set_audit_user`). Default is `"local"` (no auth in v1). Recipe replay can override per-block when F1 lands.

JSON payloads via SQLite's `json_object()` (JSON1 extension; bundled with rusqlite since long before our pinned 0.32). Trigger shape per audited table:

```sql
CREATE TRIGGER <table>_audit_insert AFTER INSERT ON <table> BEGIN
    INSERT INTO audit_log (user, table_name, row_id, op, before, after)
    VALUES (
        (SELECT user FROM _audit_context),
        '<table>',
        NEW.id,
        'insert',
        NULL,
        json_object('col1', NEW.col1, 'col2', NEW.col2, ...)
    );
END;
-- Plus _audit_update (before + after JSON) and _audit_delete (before only) variants.
```

Audited tables: `speaker`, `session`, `instrument`, `protocol`, `entity`, `entity_ref`, `bundle`, `tier`, `processing_run`. **NOT** audited: `project` (singleton), `schema_migrations` (managed by migrator), `audit_log` itself (would recurse), `_audit_context` (engine-internal).

**Trigger-rebuild discipline**: any future migration that `ALTER TABLE`s an audited table must `DROP TRIGGER IF EXISTS` + recreate the three triggers so JSON column lists stay current. Codified in a comment at the top of V3 and a checklist line in `crates/engine/migrations/README.md` (to be added).

### Confirmed B1 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Audit mechanism | **SQLite triggers + `_audit_context` singleton** | DB-enforced; survives external SQL writes; matches regulatory stance |
| Bundle ↔ Speaker | **Nullable FK `bundle.speaker_id`** | Common case covered cleanly; multi-speaker handled via per-segment tier rows in B2 |
| Python API shape | **Flat methods on `Project`** | Matches Phase-0 / A2 surface; namespacing decision settles once for the whole corpus layer later |
| `user` field | **`_audit_context` table; engine sets on connection; default `"local"`** | No auth in v1; explicit setter lets F1 recipes scope user per-block |
| Migration granularity | **One V3 migration** | Cohesive atomic schema bump; aligns with one-slice-one-commit cadence |
| Schema-only vs API'd tables | **API for Speaker + Session + Bundle-extension; rest schema-only until a first real user appears** | Keeps B1 commit at Phase-1 cadence target; no API churn when later slices add the public surface for Entity/Instrument/…|

### Layout

- `crates/engine/migrations/V3__entity_schema.sql` — full schema bump.
- `crates/engine/src/corpus.rs` — adds `Speaker`, `Session` structs; `Project::{add_speaker, speakers, get_speaker, add_session, sessions, get_session, set_audit_user, audit_user}`; extends `Project::add_bundle` to accept optional `session_id` + `speaker_id`.
- `crates/python/src/lib.rs` — PyO3 wrappers; `#[gen_stub_*]` attributes.
- `python/sadda/__init__.py` — re-exports + `@stable` decoration.
- `crates/engine/tests/migrations.rs` — extended: V3 applies on fresh DB and on a synthesized post-V2 DB; triggers fire and write the expected JSON; bundle extension keeps existing Phase-0 columns intact.
- `python/tests/test_corpus.py` — Speaker/Session add+list round-trips.

### Schema sketch

```sql
-- Entities (each with extra TEXT for JSON payload):
CREATE TABLE speaker (id, name, sex, birth_year, notes, extra, created_at);
CREATE TABLE session (id, name, started_at, ended_at, location,
                     instrument_id REFERENCES instrument(id),
                     protocol_id   REFERENCES protocol(id),
                     notes, extra, created_at);
CREATE TABLE instrument (id, name, kind, serial, calibration, extra, created_at);
CREATE TABLE protocol (id, name, description, schema, extra, created_at);
CREATE TABLE entity (id, kind, name, extra, created_at);
CREATE TABLE entity_ref (id, entity_id, target_kind, target_id, role, extra);
CREATE TABLE tier (id, bundle_id, name, type, parent_id, cardinality,
                  schema, extra, created_at, UNIQUE (bundle_id, name));
CREATE TABLE processing_run (id, bundle_id, kind, processor_id, processor_version,
                             weights_checksum, parameters, input_tier_ids,
                             output_tier_ids, output_signal_ids,
                             started_at, finished_at, status, error_message,
                             recipe_run_id);

-- Bundle extension:
ALTER TABLE bundle ADD COLUMN session_id INTEGER REFERENCES session(id);
ALTER TABLE bundle ADD COLUMN speaker_id INTEGER REFERENCES speaker(id);
ALTER TABLE bundle ADD COLUMN extra TEXT;

-- Audit:
CREATE TABLE _audit_context (id INTEGER PK CHECK(id=1), user TEXT NOT NULL DEFAULT 'local');
INSERT INTO _audit_context VALUES (1, 'local');
CREATE TABLE audit_log (id, timestamp, user, table_name, row_id, op, before, after);

-- 27 triggers (3 per audited table × 9 audited tables).
```

Type enums in CHECK constraints:
- `tier.type IN ('interval', 'point', 'reference', 'continuous_numeric', 'continuous_vector', 'categorical_sampled')`
- `tier.cardinality IN ('one_to_one', 'one_to_many', 'many_to_one', 'none')`
- `entity_ref.target_kind IN ('bundle', 'session', 'speaker', 'tier', 'annotation')`
- `processing_run.kind IN ('ml_model', 'dsp_algorithm', 'clinical_measure', 'plugin')`
- `processing_run.status IN ('ok', 'error', 'partial')`
- `audit_log.op IN ('insert', 'update', 'delete')`

### What this entry doesn't decide

- **Trigger regeneration tooling.** A future `cargo xtask audit-triggers` (introspect tables, generate trigger SQL from `PRAGMA table_info`) is plausible but out of scope; per-migration discipline carries B1.
- **Cross-bundle query API.** The corpus-data-model entry's deferred item ("how a phonetician asks 'all phones across all bundles'") stays deferred — it's a query-language decision, not a schema one. Polars-DataFrame queries via `proj.query(...)` arrive in B2.
- **JSON-schema validation for `extra`.** Profile schemas validate `extra` payloads — they exist as files per the 2026-05-20 profile-catalog entry; wiring them into the engine's write path is a later concern.
- **Audit log retention.** Pruning, archival, vacuum policy — out of scope for B1. Likely a CLI verb after real usage.

### Sources / references

- 2026-05-18 corpus data-model entry (entity tables, audit-log shape)
- 2026-05-20 ML-model-registry entry (ProcessingRun rename + schema)
- 2026-05-18 clinical regulatory entry (audit-trail requirements)
- 2026-05-21 Phase 1 slicing entry (B1 row)
- SQLite JSON1 extension docs: https://sqlite.org/json1.html
- PostgreSQL audit_trigger pattern: https://wiki.postgresql.org/wiki/Audit_trigger_91plus

---

## 2026-05-21 — Stability decorators + type stubs (A2): mixed-project layout, pyo3-stub-gen, all-STABLE Phase-0 tiering

Goal: settle the second Phase 1 slice. A2 introduces the API contract — `@stable` / `@provisional` / `@experimental` decorators with one-time runtime warnings — and the type-stub pipeline that every subsequent slice carries. The 2026-05-18 Python-API-surface entry already pinned the tier semantics; A2 commits to *how* they're implemented and integrated.

### What A2 must deliver

From the Phase-1 slicing entry: (1) `@stable` / `@provisional` / `@experimental` Python decorators emitting one-time runtime warnings; (2) `pyo3-stub-gen` integrated into the build; (3) `py.typed` marker added; (4) existing Phase-0 APIs (`sadda.version`, `sadda.load_wav`, `sadda.f0`, `sadda.Audio`) tiered.

### Tooling: maturin mixed project + pyo3-stub-gen

Decorators are Python-level concerns — wrapping a Rust-built C extension function with `functools.wraps` is straightforward but requires Python source to live alongside the wheel. The maturin **mixed-project layout** is the canonical solution: a `python/sadda/` package directory plus a Rust extension exposed as a submodule (`sadda._native`). This is the exact pattern pydantic-core, orjson, and the pyo3-stub-gen `examples/mixed` template use.

`pyo3-stub-gen` is the only viable stub generator that targets PyO3 0.28 (verified: requires `pyo3 >= 0.26`; current release `0.22.4`). Its workflow is:
- Add `#[gen_stub_pyfunction]` / `#[gen_stub_pyclass]` / `#[gen_stub_pymethods]` *above* the existing PyO3 macros on each public item.
- Call `define_stub_info_gatherer!(stub_info)` at the bottom of the pymodule's `lib.rs`.
- Add `src/bin/stub_gen.rs` that invokes `stub_info()?.generate()` — writes `.pyi` files next to the Python package.
- CI runs `cargo run --bin stub_gen` then `git diff --exit-code python/sadda/_native.pyi`; PRs that change Rust signatures without regenerating fail.

### Confirmed A2 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Package layout | **Maturin mixed project**: `python/sadda/__init__.py` + Rust submodule `sadda._native` | Universal pattern (pydantic-core, orjson); enables Python-side decorators on Rust functions; needed for `py.typed` placement |
| Stub generation | **`cargo run --bin stub_gen` writes committed `python/sadda/_native.pyi`; CI diff-checks** | Best IDE experience (visible in source tree); drift is caught at PR time, not at build time |
| Class @provisional semantics | **Wraps `__init__`; warns once on first instantiation** | Quiet for type imports and `isinstance` checks; matches users' mental model that "using" a class = constructing one |
| Phase-0 tiering | **All four (`version`, `load_wav`, `f0`, `Audio`) → STABLE** | Matches the API-surface entry's pinning; downgrading would break the Phase-0 commitment for users already on 0.0 |
| Warning class hierarchy | **`SaddaWarning(UserWarning)` → `ProvisionalAPIWarning`, `ExperimentalAPIWarning`** | `UserWarning` is visible by default (unlike `DeprecationWarning`); shared base allows `simplefilter("ignore", SaddaWarning)` to silence all tiers at once |
| Warning frequency | **Once per decorated symbol per process** | Stored in a module-level `set[str]` keyed by qualified name; standard pattern (scikit-learn, scipy) |

### Layout

```
sadda/
├── python/sadda/                      # Python package (maturin python-source)
│   ├── __init__.py                    # re-exports + applies @stable to Phase-0 symbols
│   ├── _stability.py                  # decorators + warning classes
│   ├── _native.pyi                    # generated by stub_gen; committed
│   └── py.typed                       # PEP 561 marker (empty file)
├── crates/python/
│   ├── Cargo.toml                     # adds pyo3-stub-gen dep
│   ├── src/
│   │   ├── lib.rs                     # #[pymodule] renamed sadda → _native; #[gen_stub_*] sprinkled
│   │   └── bin/stub_gen.rs            # cargo run --bin stub_gen
└── pyproject.toml                     # [tool.maturin] python-source="python", module-name="sadda._native"
```

### Decorator implementation

`_stability.py` exports three decorators plus the warning classes. Each decorator:
- For a function: returns a `functools.wraps`-decorated wrapper that emits the warning at most once per process, then forwards to the original.
- For a class: replaces `__init__` with a wrapped version that emits the warning at most once on first instantiation, then calls the original `__init__`.
- For `@stable`: a no-op (still tags `fn.__sadda_stability__ = "stable"` for introspection).

The "once" set lives at module scope keyed by `f"{func.__module__}.{func.__qualname__}"`. Warnings are emitted with `stacklevel=2` so the user's calling line is what `warnings` reports.

### Warning categories

```python
class SaddaWarning(UserWarning):
    """Base class for sadda's stability warnings."""

class ProvisionalAPIWarning(SaddaWarning):
    """First use of a PROVISIONAL API. May break in minor versions after a deprecation cycle."""

class ExperimentalAPIWarning(SaddaWarning):
    """First use of an EXPERIMENTAL API. May break in any release without notice."""
```

Subclassing `UserWarning` (not `DeprecationWarning` / `PendingDeprecationWarning`) is deliberate: Python's default filter hides those for non-`__main__` modules, which would silence the signal we want users to see when their script imports something `sadda.ml`.

### Phase-0 tiering

| Symbol | Tier |
|---|---|
| `sadda.version()` | STABLE |
| `sadda.load_wav(path)` | STABLE |
| `sadda.f0(audio, ...)` | STABLE |
| `sadda.Audio` (class) | STABLE |

`@stable` is a no-op at runtime, but applying it now (a) tags the symbol for `inspect`-based audits, and (b) makes the absence of a stability decorator on a future PR a code-smell.

### What this entry doesn't decide

- **Top-level convenience aliases vs namespaced canonical homes.** `sadda.f0` stays as the canonical home for now; a future namespacing pass (likely after B-cluster lands `sadda.dsp`) decides whether `sadda.f0` becomes an alias for `sadda.dsp.f0` or stays the primary entry. Either move is compatible with the STABLE tier — aliases don't break callers.
- **`@stable` audit tooling.** A future `cargo xtask audit-tiers` (assert every public PyO3 symbol has a corresponding decorator in `__init__.py`) is plausible but out of scope.
- **Stub generation for engine-internal types reachable via Python.** All Phase-0 types fit; future provisional surfaces may need manual `.pyi` patches if `pyo3-stub-gen` can't infer their shape.

### Sources / references

- 2026-05-18 Python API surface entry (this entry implements the stability-tier section, lines 841–849)
- 2026-05-21 Phase 1 slicing entry (this entry expands its A2 row)
- pyo3-stub-gen docs and mixed example: https://github.com/Jij-Inc/pyo3-stub-gen
- maturin mixed-project layout: https://www.maturin.rs/project_layout

---

## 2026-05-21 — Migration framework (A1): hand-rolled migrator, forward-only, always back up

Goal: settle the first Phase 1 slice. A1 wires migrations into `engine::corpus` so that every subsequent slice — starting with B1's eight-table schema expansion — rides on proper rails rather than the Phase-0 `CREATE TABLE IF NOT EXISTS` blob.

### What A1 must deliver

From the Phase-1 slicing entry: (1) a real migration framework, (2) the `schema_migrations` table extended for richer provenance, (3) a forward-compat clamp (engine refuses to open a DB newer than itself), (4) `corpus.db.bak.<old_version>` written before any destructive migration, (5) per-migration tests.

### Tool choice: hand-rolled, ~50 LOC

The slicing entry framed the choice as "`sqlx::migrate!` vs `refinery`", but that framing pre-dated the rusqlite commitment in Phase 0. Real candidates with `rusqlite` already in the dep graph:

| | rusqlite_migration | refinery | sqlx::migrate! | hand-rolled |
|---|---|---|---|---|
| Native to `rusqlite::Connection` | ✅ | needs shim | ❌ pulls sqlx | ✅ |
| SQL strings | ✅ | ✅ `V<n>__name.sql` | ✅ | ✅ via `include_str!` |
| Rust closures per migration | ✅ first-class | ❌ | ❌ | ✅ |
| Custom version-tracking table | ❌ (uses `PRAGMA user_version` only, by design) | ❌ (uses `refinery_schema_history`) | ❌ (uses `_sqlx_migrations`) | ✅ |
| Multi-DB | SQLite-only | PG/MySQL/SQLite | PG/MySQL/SQLite/MSSQL | n/a |

The initial recommendation was `rusqlite_migration`, but verifying its API surface (crate version `1.3.1`, the one compatible with our pinned `rusqlite = "0.32"`) revealed a hard constraint: it does **not** support a custom version-tracking table — `PRAGMA user_version` is hard-wired. Upgrading to the 2.x line doesn't fix this; the design is "no custom table" across all versions. `refinery` has the same flavor of issue (its own `refinery_schema_history`). Since Phase 0 deliberately seeded `schema_migrations` as the canonical version-tracking surface and the slicing entry pinned it as the contract A1 extends, sticking with the off-the-shelf tools would mean keeping `schema_migrations` as a parallel audit log written manually from each migration body — collapsing back to most of the hand-rolled code anyway.

Hand-rolled is roughly 50 LOC: a sorted iteration over embedded SQL strings + closure `fn` registrations, each step run inside `conn.transaction()`, each writing its own row into `schema_migrations`. The "supports Rust closures" requirement is trivial without a library. Zero new dependencies, one source of truth for schema version.

### Confirmed A1 decisions

| Item | Decision | Reasoning |
|---|---|---|
| Migration tool | **Hand-rolled (~50 LOC)** | None of the off-the-shelf tools support a custom version-tracking table; Phase 0's `schema_migrations` is canonical; hand-rolled gives one source of truth and zero new deps |
| Migration direction | **Forward-only** | SQLite's column-drop story makes faithful down-migrations rare; backup files are the real recovery path; restoring `corpus.db.bak.<n>` is the documented recovery flow |
| Backup policy | **Always back up before any migration run** | Classifying "destructive" per-migration is error-prone; SQLite file copies are cheap; pruning old backups is a separate concern |
| Version-tracking table | **Keep custom `schema_migrations`, extend it** | Preserves Phase-0 continuity; add `name TEXT` and `checksum TEXT` columns for provenance and tamper detection; written atomically with each migration in the same transaction |

### Layout

- `crates/engine/migrations/V<n>__<short_name>.sql` — versioned SQL migration files, embedded via `include_str!` at compile time. `V1` is the Phase-0 baseline (`project` + `bundle` + `schema_migrations`), restated so a fresh DB walks the same path as upgraded DBs.
- `crates/engine/src/corpus/migrations.rs` — registry: a `static` slice of `Migration { version, name, kind: Sql(&'static str) | Rust(fn(&Transaction) -> Result<()>), checksum: &'static str }`, plus a `pub fn run(conn: &mut Connection) -> Result<MigrationOutcome>` that applies anything missing.
- `crates/engine/src/corpus.rs` — refactored: `Project::create` calls `migrations::run` on a fresh DB; `Project::open` runs the forward-compat clamp first, then the backup, then `migrations::run` if anything is pending.
- `crates/engine/tests/migrations.rs` — integration tests. For each version `N ≥ 2`: seed a DB at `N-1` (using only the SQL up to `V<N-1>`), apply forward, assert post-migration invariants. Plus one "fresh-create from latest" smoke test that walks the full chain on an empty DB.

### Schema_migrations extension

Phase-0 columns: `version INTEGER PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP`.
A1 adds:
- `name TEXT NOT NULL` — short slug from the migration filename, for listing and grepping
- `checksum TEXT NOT NULL` — SHA-256 of the SQL body (or `"rust:<fn-name>"` for closure migrations), so a future engine can detect that a previously-applied migration's contents have since changed (tamper / accidental edit). Computed at compile time via a tiny `build.rs` so the constant lives next to the SQL.

The `V1` migration writes its own row into `schema_migrations` to seed the new columns. Phase-0 DBs already in the wild would have a row with `version=1, name=NULL, checksum=NULL`; a `V2` migration handles the backfill before the B1 schema work lands on top.

### Forward-compat clamp

After opening the connection, read `MAX(version)` from `schema_migrations`. If that exceeds the highest version known at compile time (the last entry in the static migration slice), `Project::open` returns `EngineError::SchemaTooNew { db_version, engine_max }` instead of running any analysis. The error message points the user at upgrading the engine.

### Backup mechanics

Before invoking the migration runner, if `MAX(version) < engine_max`:
1. Issue `PRAGMA wal_checkpoint(TRUNCATE)` to flush WAL state into the main file.
2. Copy `corpus.db` → `corpus.db.bak.<current_version>` via `std::fs::copy`.
3. Apply migrations.

Backups are not garbage-collected by A1 — that's a future concern (a `sadda corpus gc-backups` CLI verb, or a `project.toml` retention policy). Disk overhead for v1 corpora is modest enough that this can wait.

### What this entry doesn't decide

- **Backup retention policy.** Out of scope for A1. Likely a CLI verb later.
- **Closure-migration ergonomics.** The first closure migration arrives in B1 or B2; the exact call-site shape is determined when a real case appears.
- **Migration linting.** A future `cargo xtask check-migrations` (verify checksums on disk against `schema_migrations`, no out-of-order files) is plausible; A1 ships only the runtime checks.

### Sources / references

- 2026-05-21 Phase 1 slicing entry (this entry expands its A1 row)
- 2026-05-18 corpus data-model entry (B-cluster scope this framework will carry)
- `rusqlite_migration` 1.3.1 API verification (custom version table not supported): https://docs.rs/rusqlite_migration/1.3.1/rusqlite_migration/

---

## 2026-05-21 — Phase 1 slicing: 12 slices in 7 clusters toward 0.1

Goal: sequence Phase 1's deliverables (full corpus schema, six tier types, DSP suite, TextGrid + EAF I/O, live recording, stability decorators, type stubs, recipes, migration framework) into a concrete commit-by-commit ordering. The 2026-05-18 milestone-plan entry committed to Phase 1's *scope*; this entry commits to its *cadence and ordering*.

### What "vertical slice" means in Phase 1

Phase 0's slices touched every architectural layer (engine → corpus → Python → app → UniFFI) per commit. Phase 1's release target is **the Python library on PyPI**, so "vertical" re-reads as: **each slice ends in a usable Python entry point**, not "touches the desktop app." The egui surface stays minimal until Phase 2.

### What changes from Phase 0

| | Phase 0 | Phase 1 |
|---|---|---|
| Slices | 8 | 12 |
| Average LOC / slice | ~375 | ~750 (est.) |
| Layers crossed per slice | all of them | 2–3 (engine + PyO3 + tests) |
| Release at end | 0.0 internal | **0.1 to PyPI** |

Slices are roughly 2× thicker than Phase 0 because the work shifts from glue (one layer's worth at a time) to depth (Parquet sidecar I/O alone exceeds the entire Phase 0 corpus crate). Keep one focused commit per slice; CI green after each.

### Decomposition: 7 clusters

Work-items decompose along three orthogonal lines: cross-cutting infrastructure (which gates everything), corpus schema expansion (strict internal dependency chain), and surface work (DSP, I/O, recording, recipes — mostly independent of each other; depends on corpus).

**Cluster A — Cross-cutting infrastructure** (lands first; gates the rest)

1. **Migration framework.** `sqlx::migrate!` or `refinery` wired in; `schema_migrations` table extended (the table itself was seeded in Phase 0 piece 5); engine refuses to open a DB newer than it knows (forward-compat clamp); `corpus.db.bak.<old_version>` written before any destructive migration; per-migration tests. Lands before any schema expansion.
2. **Stability decorators + type stubs scaffolding.** `@stable / @provisional / @experimental` Python decorators that emit one-time runtime warnings; `pyo3-stub-gen` integrated into the maturin build; `py.typed` marker added; existing Phase-0 APIs (`sadda.version`, `sadda.load_wav`, `sadda.f0`) tiered. Sets the contract that every subsequent slice marks its API surface.

**Cluster B — Corpus schema expansion** (strict order: B1 → B2 → B3)

3. **Full entity schema + AuditLog.** Speaker, Session, Bundle (extended), Tier (header), Entity, EntityRef, Instrument, Protocol. `extra: json` columns throughout. Append-only AuditLog with mutation triggers. ProcessingRun table (the renamed ModelRun per the ML-registry entry).
4. **Sparse tier types.** `interval`, `point`, `reference` with CRUD + the first cut of `proj.query(...) → polars.DataFrame`. Parent-child cardinality enforced at insert time.
5. **Dense tier types + Parquet sidecars.** `continuous_numeric`, `continuous_vector`, `categorical_sampled`. `DerivedSignal` registration rows. Reader/writer in `engine::storage::parquet`; mmap-friendly load path so AI-engineer users can `pl.scan_parquet` directly.

**Cluster C — DSP surface** (independent of B; can interleave)

6. **Foundational DSP.** Windowing (Hann, Hamming, Blackman, Gaussian, Kaiser), STFT, spectrogram, intensity. Pure functions over `&[f32]`; no corpus dependency. Polars-friendly outputs.
7. **Advanced DSP.** Formants via LPC + root-solver; MFCC (mel-filterbank → DCT); refined pitch with voicing decision (extends Phase 0's autocorrelation tracker). Stays inside `engine::dsp`; PyO3 wrappers in `crates/python`.

**Cluster D — Interop I/O**

8. **TextGrid round-trip.** IntervalTier and TextTier import + export; JSON-sentinel (`{json:{…}}`) for attribute round-trip; explicit lossiness documentation. The adoption hinge for Praat users.
9. **EAF round-trip (bidirectional).** ELAN tier types (ALIGNABLE_ANNOTATION, TIME_SUBDIVISION, SYMBOLIC_ASSOCIATION); `parent_tier_ref` preserved; XML round-trip stable enough that ELAN can re-open exports without warnings.

**Cluster E — Recording**

10. **Live recording (cpal).** `.in_progress/` flow → atomic commit; cpal cross-platform driver; metering callbacks. JACK as a stretch goal — if cpal absorbs the time budget, push JACK to a 0.1.x patch release.

**Cluster F — Reproducibility**

11. **`sadda.recipe.record()` / `.replay()`.** Context manager that logs every analysis call through `ProcessingRun` + `AuditLog`; serializes a recipe as a Python script the user can re-run on the same project or another. Connects the cross-cutting reproducibility need surfaced across corpus + refdist + clinical entries.

**Cluster G — Release**

12. **0.1.0 to PyPI + mkdocs-material docs site.** Claim `sadda` on PyPI; mkdocs-material with mkdocstrings auto-rendering API reference from the Rust `///` doc comments (via a cargo-doc → markdown bridge) and the Python stubs; GitHub Pages hosting; first quickstart tutorial. The 2026-05-21 docs-strategy entry already pinned this as the docs-site start point.

### Dependency graph

```
A1 ─┐
    ├─→ B1 ─→ B2 ─→ {D1, D2, F1}
A2 ─┘    │
         ├─→ B3 ─→ F1
         └─→ E1

C1, C2 ──→ (independent; interleave with B)

G12 = last (releases everything above)
```

Concretely: A1 and A2 first (parallel-OK between them but both before B*). Then B1. C* can begin any time A2 is done — they touch no schema. B2 + B3 follow B1. E1 follows B1. D1 + D2 follow B2 (need at least sparse tiers). F1 follows B1 + the Python API instrumentation that A2 sets up. G12 last.

### Parallel risk spike, off-track

**Embedded CPython distribution.** Per the milestone plan, runs alongside Phase 1 as a fail-fast experiment: one signed macOS `.app` with bundled Python running a real script end-to-end. Validates the packaging story before Phase 7 commits to it. Does **not** gate the 0.1 release — Phase 0's `crates/script-engine` already proved the embed works; the spike is about *shipping* the embed. Pick this up when Mac time is available; tracked separately from the 12-slice plan.

### Confirmed slicing decisions

| Item | Decision | Reasoning |
|---|---|---|
| First slice | **Migration framework (A1)** | Infrastructure-first. Every schema change after this rides on solid rails. The first commit is invisible to users; that's fine — Phase 1's first commit is not a marketing release |
| Slice granularity | **12 slices, ~1 commit each** | Match Phase 0's cadence; atomic CI; easy review. Resist combining clusters into mega-PRs |
| EAF scope at 0.1 | **Bidirectional** | Don't preemptively apply the cut line. EAF round-trip is in the cut list but cut from default only under pressure; aim for both at 0.1 |
| PyPI timing | **Single 0.1.0 release at end of Phase 1** | No stub 0.0.x. Name-squatting risk is mitigated by the public `sadda-speech` GitHub org claim already being live. Single launch event |

### What this entry doesn't decide

- **Migration tool choice (`sqlx::migrate!` vs `refinery`).** Settled inside A1's design pass. Both are viable; pick after a small spike.
- **Recipe serialization format inside F1.** Python script is the user-facing artifact; whether the persistent log is JSON, TOML, or SQL rows is internal.
- **Which sub-DSP measure goes in C1 vs C2.** Drafted above but the precise cut can shift as code is written.
- **Live recording UX for the Python API.** `sadda.live.start_session()` is the entry point; subscriber-decorator semantics (per the API-surface entry) are settled at design level but need a concrete spike inside E1.
- **mkdocs-material theme + nav layout.** Settled inside G12; not architectural.

### Pace and revisit cadence

- Per the milestone plan, Phase 1 is 3–4 months at solo part-time. 12 slices over ~16 weeks ≈ one slice every 1–2 weeks.
- Revisit this entry after slice 6 (mid-phase). If cadence is slower than 1/week, consider deferring EAF to import-only (the cut-line move) and/or splitting JACK out of E1 entirely.
- After 0.1 ships: real PyPI users land; feedback may reshape Phase 2 scope. Phase 1 cadence numbers feed into the milestone-plan revisit.

### Sources / references

- 2026-05-18 milestone-plan entry (this entry is downstream of its Phase 1 row)
- 2026-05-18 corpus data-model entry (B-cluster scope)
- 2026-05-18 Python API surface entry (A2 stability decorators; F1 recipes)
- 2026-05-21 documentation strategy entry (G12 docs site)
- "Tracer Bullets" — Pragmatic Programmer (vertical-slice principle): https://pragprog.com/tips/

---

## 2026-05-21 — Documentation strategy: discipline now, site at 0.1, polish at 1.0

Goal: settle when and how docs grow. The milestone plan already committed to *"documentation grows incrementally per phase, not as a separate final phase"* but didn't schedule it. This entry pins concrete starts per phase.

### Three layers, three different starts

| Doc layer | What it is | Starts |
|---|---|---|
| **API reference** | `///` doc comments on every public Rust item, docstrings on every PyO3-exposed Python item. Auto-compiles to docs via `cargo doc` and (later) mkdocstrings | **Now (Phase 0).** Enforced via `#![warn(missing_docs)]` on library crates + CI's `-D warnings` |
| **User-facing docs site** | mkdocs-material + mkdocstrings; conceptual guides drawn from DEVLOG entries; quickstart tutorial; auto-generated API reference; GitHub Pages hosting | **End of Phase 1 (0.1 PyPI release)** — when there's a Python API stable enough for outside users to actually try |
| **Polished docs site + migration** | Parselmouth migration docs, notebook tutorials, bundled sample projects, visual GUI walkthroughs | **Phase 7 (toward 1.0)**, per the existing milestone-plan deliverables |

### Why these particular start points

- **API reference now.** Cost of writing a one-line doc comment when you write a function is near-zero; cost of retrofitting hundreds of public items at 0.1 is a multi-day audit nobody wants. Enforcing via a lint means the discipline doesn't depend on remembering. PyO3 `///` comments on `#[pyfunction]` / `#[pymethods]` become Python docstrings automatically — same source of truth.
- **Docs site at end of Phase 1.** Before 0.1 the Python API isn't stable enough to write tutorials against — examples would rot constantly. After 0.1 there's an audience (PyPI installers) who genuinely benefit. Earlier than that is writing for a reader who doesn't exist yet.
- **GUI tutorials at Phase 2.** Screenshots of an unstable GUI rot every week. Wait until 0.2 ships.
- **Migration docs at Phase 7.** Praat-to-sadda migration is high-effort writing; do it once when the API is stable and not before.

### Concrete starts this turn

- `#![warn(missing_docs)]` enabled on the four library crates: `sadda-engine`, `sadda-python`, `sadda-script-engine`, `sadda-uniffi`. The `app` binary crate is excluded — binaries don't expose a library surface.
- Existing public items that lacked doc comments are filled in (audit done across the four library crates in the same commit).
- CI's `-D warnings` makes future PRs that omit doc comments fail. Adding a public item is now contract-bound to come with a one-line doc.

### Cross-references / what this entry updates

- Extends the 2026-05-18 milestone-plan entry's "documentation grows incrementally" principle with a concrete per-phase schedule.
- Touches the API-surface entry (2026-05-18): `sadda.ml` and other PROVISIONAL surfaces still get docs, but their docstrings can call out the stability tier explicitly.

### Sources / references

- mkdocs-material: https://squidfunk.github.io/mkdocs-material
- mkdocstrings (Python API rendering): https://mkdocstrings.github.io
- rustdoc book: https://doc.rust-lang.org/rustdoc/
- PyO3 docs guide on docstrings: https://pyo3.rs/v0.28.0/function/signature.html#documenting-functions

---

## 2026-05-20 — Profile catalog: five v1 profiles over the entity / tier / refdist / measure surfaces

Goal: close the "profile catalog" open follow-up from the corpus entry. Specify which profiles ship at v1, what each profile concretely contains, and the policy around default profile, authoring path, and mid-flight profile changes. The mechanism (profile = schema validator over JSON `extra`) was already settled in the corpus entry; this entry settles content and governance.

### Profile = a bundle of seven things

A sadda profile is a coherent default-state package across seven surfaces:

1. **Entity `extra` schema validators** — which fields are required on Speaker / Patient / Case / Participant entities; drives typed GUI forms over the flexible JSON storage.
2. **Default tier templates** — what tier set a new bundle gets out of the box.
3. **Recommended reference distributions** — the subset of refdist starter entries pre-recommended for this workflow.
4. **Default measure surfaces** — which analyses the GUI prioritizes.
5. **Workflow defaults** — what panels/widgets surface; what GUI mode the project opens to.
6. **Optional capabilities flagged on** — e.g. calibration mandatory, audit logging emphasized, community-consent flow.
7. **Terminology** — labels like "Patient" vs "Speaker" vs "Case" vs "Participant".

### Profiles shipping at v1

Five profiles, mapped from the 8 user groups in the 2026-05-16 entry. Voice Training (voice coaches + L2 learners, combined) deferred to v1.x — it's mobile-heavy per its source entry, so shipping it pre-Phase-6 would feel half-baked. Speech AI / ML engineers don't get a profile because their needs cut across all five via the ML feature surface (`sadda.ml`, embedding tiers). Bioacoustics stays designed-in (frequency-range agnosticism) but doesn't get a v1 profile.

#### 1. Phonetician *(default)*

The user's own profile; also the fallback when no profile is specified.

| Surface | Content |
|---|---|
| **Entities** | `Speaker` with `extra`: l1_language (ISO 639-3), dialect, age_band, sex_at_birth (optional), handedness (optional) |
| **Tier template** | `phones` (interval, IPA) → `words` (interval, parent of phones) → `utterances` (interval, parent of words); `notes` (point, free-text) |
| **Recommended distributions** | Hillenbrand 1995, Peterson-Barney 1952, one VOT reference |
| **Default measures** | F0 (autocorrelation), formants F1–F3, duration, intensity, spectrogram |
| **Workflow defaults** | Praat-like layout: spectrogram + waveform + tier strip with sync cursor; refdist overlays visible by default |
| **Capabilities** | Calibration optional; audit logging standard (always-on, low-emphasis) |
| **Terminology** | "Speaker" / "Session" / "Bundle" |

#### 2. Clinical

| Surface | Content |
|---|---|
| **Entities** | `Patient` with `extra`: MRN, DOB, sex_at_birth, treating_clinician, diagnosis_codes (ICD-10 strings), clinical_history (optional rich text). `Encounter` with `extra`: type (initial_eval / follow_up / post_treatment), notes. Sessions link to encounters; bundles get extras: protocol (CAPE-V / Rainbow / sustained_vowel / connected_speech), calibration_set_id (required when SPL is measured) |
| **Tier template** | Protocol-driven: sustained-vowel marker (single interval); CAPE-V auditory-perceptual rating tier (categorical_sampled); phones tier optional |
| **Recommended distributions** | Age/sex-banded jitter/shimmer/HNR norms; AVQI/ABI normative ranges; Voice Range Profile norms |
| **Default measures** | AVQI, ABI, CPP, jitter (multiple variants), shimmer, HNR, F0 statistics, calibrated SPL; Voice Range Profile / phonetogram view; longitudinal pre/post comparison |
| **Workflow defaults** | Patient-as-first-class: GUI opens to patient list. Protocol-driven recording flow: select protocol → guided record → auto-analyze → draft report. Research-use-only watermark on exports (per the 2026-05-18 clinical regulatory entry, posture 3) |
| **Capabilities** | Calibration *required* (red banner if uncalibrated); audit logging emphasized in UI |
| **Terminology** | "Patient" / "Encounter" / "Treatment Session" / "Bundle" |

#### 3. Forensic

| Surface | Content |
|---|---|
| **Entities** | `Case` with `extra`: case_id, lead_investigator, jurisdiction, case_status (open / closed / archived). `Sample` with `extra`: source (questioned / known), evidence_id, chain_of_custody_events (append-only event log backed by AuditLog). `Speaker` anonymized: speaker_alias (S1, S2, …), demographic_band (no exact age). Bundle `extra`: sample_id (FK), recording_quality_notes, sealed_at, sealed_by |
| **Tier template** | Long-term formant tracking (continuous_numeric); F0 distribution sampling (continuous_numeric); articulation-rate markers (interval) |
| **Recommended distributions** | Population reference data: age/sex/dialect-banded F0; formant LTAS distributions; acoustic-feature distributions for LR computation |
| **Default measures** | Long-term formant distributions; F0 statistics (mean / SD / range); articulation rate; speaker similarity (LR-framework analyses); LTAS |
| **Workflow defaults** | Case-as-first-class: GUI opens to case list. Audit log prominent in the case timeline view. Chain-of-custody UI for evidence-handling events. `sadda.recipe.record` mode default-on (every analysis logged as a recipe). Reproducible "raw audio → report" pipeline export |
| **Capabilities** | Audit logging *emphasized + mandatory* recipe recording for all analyses; encryption at rest emphasized (per cross-cutting pattern J); calibration optional but recommended |
| **Terminology** | "Case" / "Sample" / "Speaker (S1)" / "Recording" |

#### 4. Field linguistics

| Surface | Content |
|---|---|
| **Entities** | `Speaker` with `extra`: l1_language (ISO 639-3, required), community_id (FK), consent_status, consent_events (timestamped log), age_band, sex (optional; community-norm-dependent). `Community` with `extra`: language_iso639_3, region, community_consent_doc_ref, data_governance_policy. Session `extra`: location, session_type (elicitation / narrative / conversation), elicitor |
| **Tier template** | Multi-tier glossed: phonetic (interval, IPA) → morphological_gloss (interval, parent: phonetic) → free_translation (interval, parent: morphological_gloss) → notes (interval, free-text) |
| **Recommended distributions** | Less applicable — field work is usually on previously-undocumented populations. Reserve slot for IPA frequency references by major language family; rely on community contribution |
| **Default measures** | F0, formants (surfaced but not prioritized), spectrogram. Annotation tools take primary screen real estate |
| **Workflow defaults** | Annotation-first GUI: multi-tier strip prominent; spectrogram secondary. IPA palette + feature-based phone search front-and-center (per cross-cutting field-linguistics need). Crash-resilient autosave emphasized. Community-consent badge on Speaker entities |
| **Capabilities** | IPA tooling emphasized; community-consent flow mandatory before public-export operations; archival-format export (DELAMAN, ELAR, PARADISEC) prominent in export menu; calibration optional |
| **Terminology** | "Speaker" / "Session" / "Community" / "Elicitation" |

#### 5. Experimenter

| Surface | Content |
|---|---|
| **Entities** | `Participant` with `extra`: participant_id, age, sex, l1_language, consent_status, recruitment_source (lab / Prolific / MTurk / etc.). `Experiment` with `extra`: protocol_def_ref (link to sequence-editor output), task_description, n_target. `Trial` with `extra`: trial_index, condition, stimulus_id (FK), response_data (JSON), reaction_time_ms. `Stimulus` with `extra`: stimulus_type (audio / image / text), source_path, content_description |
| **Tier template** | Trial-aligned interval tier (one interval per trial); response markers (point tier) |
| **Recommended distributions** | Less applicable — experiments produce their own data. Reserve slot for normative reference distributions if a stock paradigm benefits |
| **Default measures** | Trial-level aggregates: reaction time stats, accuracy by condition, F0 / formants extracted per trial. Auto-process recordings (force-align, extract features, tag with trial info per cross-cutting pattern I) |
| **Workflow defaults** | Experiment list view at GUI open. Sequence-editor surface for protocol building (per the 2026-05-18 experiment-runner entry). Auto-tagging: recordings link to trials by metadata. PsychoPy CSV import on by default |
| **Capabilities** | Trial-runner mode enabled; `sadda.recipe.record` mode default-on; calibration optional |
| **Terminology** | "Experiment" / "Trial" / "Participant" / "Stimulus" / "Block" |

### Policy decisions

**Default profile.** When `sadda.new_project(path)` is called without a `profile=` argument, profile is `phonetician`. Mirrors the tool's primary identity (and the user's own profile).

**Built-in profiles only at v1.** No plugin-shipped or user-authored profiles in v1. Custom per-project `extra` schemas can be declared in `project.toml` under `[profile.extensions]` for minor customizations without forking a profile. Reasoning: profile schema will churn during early use; locking the authoring API prematurely would force false stability.

**Cross-profile changes mid-flight.** Allowed via `sadda.profile.change(project, new_profile)`. Engine validates all existing entities against the new profile's `extra` schemas. Required-field gaps are surfaced ("12 patients missing `treating_clinician`") rather than auto-filled. User fixes via batch-edit GUI or reverts. No data is destroyed by a profile change; the underlying `extra` JSON is preserved, only the validator changes.

**Profile selection at project creation.** The "new project" GUI dialog presents the five profiles with one-line descriptions and a "see what changes" expand. Initial profile selection is significant but not irreversible.

**Profile + GUI customization are orthogonal.** Profile drives *defaults*; users can override panel layout, hide/show widgets, adjust terminology preferences within a profile. Per-user preferences live in the user config, not the project — switching machines preserves panel layouts via cloud-config sync (deferred) or manual export.

### Architectural touch points

- **API.** `sadda.new_project(path, profile="phonetician")` accepts profile id. `sadda.profile.change(project, new_profile)` swaps profile. `sadda.profile.list()` returns the catalog. All under `sadda.profile` namespace, PROVISIONAL stability tier (the API entry's existing tiering).
- **Storage.** `project.toml` has a top-level `profile = "..."` field. Profile id is also persisted as a column on the `Project` row in `corpus.db` (already in the corpus entry's entity table).
- **Profile-extension hooks.** `project.toml` `[profile.extensions]` section can declare additional `extra` schema fields and additional tier templates that augment the chosen profile without replacing it. Format spec to be finalized when first per-project customization is requested.
- **Refdist defaults.** Each profile's "recommended distributions" list is consumed by the in-app refdist picker — entries are highlighted as "recommended for this profile" without restricting which distributions can actually be installed.
- **GUI mode switching.** The 2026-05-18 API surface entry's `sadda.app` namespace gains a `set_workflow_mode(profile_id)` call that re-applies workflow defaults; useful for users who want to temporarily try a clinical view of a phonetic project.
- **Audit-log emphasis is profile-driven, not engine-default.** The `AuditLog` table is always populated regardless of profile; profiles only control how prominent it is in the GUI.

### v1 deliverables this entry commits to

1. Five profiles shipped: `phonetician`, `clinical`, `forensic`, `field`, `experimenter`.
2. Each profile's seven surfaces (entities / tiers / refdist / measures / workflow / capabilities / terminology) implemented to the level specified above.
3. `project.toml` `profile` field and `[profile.extensions]` section.
4. `sadda.new_project(path, profile=...)`, `sadda.profile.change()`, `sadda.profile.list()`.
5. New-project GUI dialog surfacing the five profiles with one-line descriptions.
6. Cross-profile change flow with validation gap surfacing.
7. Per-profile refdist recommendation lists (consumed by refdist picker GUI).

### Open trade-offs / deferred

- **Voice Training profile (v1.x).** Combined voice coach + L2 learner profile. Mobile-heavy; better landed when Phase 6 ships. Workflow shape: own-voice baseline; user-defined target zones (not population norms); progress tracking; recording-and-compare against a model. Reference distributions: target zones (`measure.kind = "target_zone"` from refdist entry) and mobile-app-friendly subset.
- **Plugin-shipped profiles.** Once the profile schema has stabilized through real v1 use, allow plugins to declare profiles via a manifest. v1.x feature; possible drivers: a research lab shipping a custom protocol-bundle profile, or a clinical specialty (singing voice, pediatric) wanting a sub-profile.
- **User-authored profiles.** v1.x feature once authoring API is designed. Probably scaffolded via `sadda profile init <name>` CLI producing a profile-definition TOML the user can edit.
- **Profile catalog versioning.** When profile `extra` schemas evolve mid-v1 (e.g. clinical adds a required field), how do existing projects upgrade? Likely: profile schemas are themselves semver-versioned; project.toml records the profile version at creation; migration utilities walk `extra` payloads (per the corpus entry's migration policy).
- **Bioacoustics adjacent profile.** Per the user-groups entry, bioacoustics has real adoption-crossover potential. v1.x or community-contributed profile makes sense once the plugin-authoring path is open.
- **Distribution recommendations need actual curation.** This entry sketches "recommended distributions" lists at the *category* level (clinical: jitter/shimmer/HNR norms); the actual distribution IDs need to match what's curated in tier 1 of the refdist registry. Sync these lists with refdist starter set as Phase 1 work progresses.

### Cross-references / what this entry updates

- Closes the "profile catalog" follow-up from the 2026-05-18 corpus entry.
- Closes the "profile-driven defaults" follow-up from the 2026-05-18 refdist entry.
- Extends the 2026-05-18 API entry: adds `sadda.profile` namespace with `list / change` and concretizes `new_project(profile=...)`.
- Threads through: clinical entry (research-use-only labeling driven by `clinical` profile), forensic patterns from the user-groups entry, field linguistics archival export emphasis, experimenter trial metadata flow.

### Sources / references

- EMU-SDMS per-database schema: https://ips-lmu.github.io/EMU.html
- ELAN tier templates: https://www.mpi.nl/tools/elan
- PsychoPy Builder paradigm templates: https://www.psychopy.org/builder
- FLEx (SIL FieldWorks): https://software.sil.org/fieldworks
- DELAMAN archive standards: https://www.delaman.org
- ICD-10: https://icd.who.int/browse10
- CAPE-V: Kempster et al. (2009), *American Journal of Speech-Language Pathology*

---

## 2026-05-20 — ML model registry: provenance audit + parallel-to-refdist distribution

Goal: close the deferred "ML model registry scope" item from the corpus entry. The phrase as flagged conflated two structurally separate concerns; this entry separates them and specifies both.

### Two layers, each with its own design

| Layer | Scope | Lives where | Question it answers |
|---|---|---|---|
| **Provenance** | Audit record of every processing run on a bundle | Per-project SQLite | "Where did this tier come from?" |
| **Distribution** | How models reach a user's machine and become loadable by ID | User-level cache + cross-project registry | "Where do I get `wav2vec2-base`?" |

The corpus entry's `ModelRun` table was the provenance layer. The articulatory entry's open question ("downloaded via same registry as refdist, or separate?") was the distribution layer.

### Layer 1: Provenance — rename `ModelRun` → `ProcessingRun`

The clinical entry's commitment ("every measure records which algorithm version was used … plumbed through the ML model registry") already implies coverage of non-ML algorithms (pitch, formants, AVQI) alongside ML models. `ModelRun` was misleading. Rename to **`ProcessingRun`** with a `kind` discriminator.

```
ProcessingRun
  id                primary key
  bundle_id         FK
  kind              enum: ml_model | dsp_algorithm | clinical_measure | plugin
  processor_id      e.g. "sadda/wav2vec2-base-960h" or "sadda.dsp.pitch"
  processor_version semver
  weights_checksum  nullable — populated for ml_model; null for pure-DSP
  parameters        JSON — call args
  input_tier_ids    JSON — what inputs were consumed
  output_tier_ids   JSON — what tiers were produced
  output_signal_ids JSON — for DerivedSignals (Parquet sidecars)
  started_at        timestamp
  finished_at       timestamp
  status            enum: ok | error | partial
  error_message     nullable
  recipe_run_id     nullable FK — set when invoked from sadda.recipe.record
```

Coverage by `kind`:

- `ml_model` — registry-resolved models; processor_id from registry namespace; weights_checksum populated
- `dsp_algorithm` — built-in DSP; processor_id is the function path (`sadda.dsp.pitch`); version is the sadda version
- `clinical_measure` — composite measures; processor_id e.g. `sadda.clinical.avqi`; version is sadda + measure spec version
- `plugin` — plugin-supplied analyzers; processor_id includes the plugin id

A single audit timeline per bundle answers "where did every signal/annotation come from" uniformly. Citation export (planned in the clinical entry) walks this table.

### Layer 2: Distribution — parallel registry, shared protocol with refdist, HF passthrough escape hatch

Locked: separate registry repo, reuses the refdist mechanism. Mirrors three-tier model (bundled / curated / community), TOML-manifest format, GitHub-Pages index, CI validation, in-app publishing flow, project pinning, automatic citation. Where structurally different from refdist:

- **Larger artifacts.** Weights can be 10MB–5GB+. GitHub release assets cap ~2GB; above that, the manifest declares an external mirror (HuggingFace, Zenodo, S3) with file checksum. Engine downloads from the declared URL and validates against the checksum.
- **Compute hints in manifest.** RAM, GPU/Metal optional/required, FP precision. Engine surfaces "won't run on your machine" before download.
- **CI validation is shallower.** Curated-tier CI can checksum, license-check, and (optionally) run a smoke test on a small input — but it can't end-to-end validate accuracy. Trust signals lean more on editorial review than for refdist.
- **Output spec ties into the tier model.** Manifest declares `output.tier_kind` in our tier vocabulary so the engine knows what tier type inference results produce.

#### Manifest sketch

```toml
id = "sadda/wav2vec2-base-960h"
version = "1.0.0"
title = "wav2vec2-base self-supervised speech model"
upstream_source = "https://huggingface.co/facebook/wav2vec2-base-960h"
license = "Apache-2.0"

[model]
kind = "embedding"               # embedding | transcription | vad | segmentation | alignment | feature
format = "onnx"                  # onnx | gguf | safetensors | savedmodel
file = "model.onnx"              # OR url = "..." + url_checksum = "sha256:..." for external mirror
file_checksum = "sha256:..."

[input]
modality = "audio"               # audio | video | both
sample_rate_hz = 16000
channels = 1

[output]
tier_kind = "continuous_vector"
channels = 768
sample_rate_hz = 50

[compute]
cpu_min_ram_mb = 1024
gpu = "optional"                 # required | optional | unsupported
quantization_available = ["int8"]

[citation]
bibtex = "..."
```

#### ID schemes — three resolvable forms via `sadda.ml.load_model(id)`

- `"sadda/wav2vec2-base-960h"` — curated registry; version optional (latest if omitted), should be pinned in `project.toml` for reproducibility
- `"hf://facebook/wav2vec2-base-960h"` — HuggingFace passthrough; no curation, no manifest, no quality guarantee. `ProcessingRun.processor_id` records `"hf://..."` verbatim so provenance is honest about the trust tier
- `"local:///abs/path/model.onnx"` — local file; for in-development or air-gapped use

HF passthrough behavior:
- Downloaded via the HF Hub protocol; cached alongside curated models
- File format opaque at fetch time; load attempts the supported runtime (ort or fallback); fail loud if format unsupported
- Auth via `HF_TOKEN` env or sadda config; not required for public models
- HF revision SHA recorded in `ProcessingRun.weights_checksum`, so recipe replay can pin the exact upstream version

### Canonical format: ONNX, with documented exceptions

The Phase 3 milestone commits to `ort runtime`. Implication: **ONNX is the canonical format for curated-tier publishing.** Manifest's `format` field accepts `gguf` / `safetensors` / `savedmodel` for specific exceptions — the likely one being a whisper.cpp GGUF variant for embedded mobile Whisper in Phase 6 — but the default and recommended publish path is ONNX. Format-conversion (e.g. `optimum`) belongs to the publishing workflow, not the runtime. HF passthrough sidesteps this: we load whatever HF has, attempting available runtimes.

### Storage and cache

```
~/.local/share/sadda/models/
├── sadda/                                    ← curated namespace
│   └── wav2vec2-base-960h/
│       ├── 1.0.0/
│       │   ├── manifest.toml
│       │   ├── model.onnx
│       │   └── LICENSE
│       └── 1.0.1/
├── hf/                                       ← HF passthrough cache
│   └── facebook/
│       └── wav2vec2-base-960h/
│           └── <hf-revision-sha>/...
└── .cache_meta.json                          ← checksums, last-used timestamps
```

Cache eviction is manual via CLI in v1 (`sadda model gc`). Automatic LRU eviction with a configurable disk cap is a follow-up if real users hit disk pressure.

### v1 curated starter set

Per Phase 3 / Phase 4 milestones:

| Model | Purpose | Phase | Notes |
|---|---|---|---|
| `sadda/silero-vad` | Voice activity detection | 3 | Small (~2MB); bundled with the app rather than downloaded |
| `sadda/wav2vec2-base-960h` | Self-supervised speech embeddings | 3 | On-demand download |
| `sadda/whisper-tiny` | Speech transcription (entry-level) | 3 | On-demand; good enough for trial-log alignment |
| `sadda/whisper-base` | Speech transcription (better quality) | 3 | Optional larger variant |
| `sadda/tongue-contour-v1` | Ultrasound tongue-contour segmentation | 4 | Specific model choice still open per articulatory entry |

VAD is the only bundled model; the rest download on first use, with sizes surfaced in the GUI before download begins.

### Architectural touch points

- **API surface.** `sadda.ml` stays at the PROVISIONAL stability tier from the API entry. Loader signature: `sadda.ml.load_model(id, version=None, **load_opts) -> Model`. Model objects expose `extract_embeddings` / `transcribe` / `vad` / `align` / `segment` depending on `model.kind`.
- **Recipe replay.** Curated-registry IDs resolve to the pinned version; HF passthrough IDs resolve to the original HF revision SHA recorded at first run; local-path IDs require the file to be present.
- **Project pinning.** `project.toml` gets a `[models]` block listing `id` + `version` used in the project. Mirrors refdist pinning.
- **In-app publishing.** Same flow as refdist tier-3 publishing: GUI scaffolds a manifest, validates, opens a PR against the model-registry repo using the user's GitHub credentials. For weights, GUI prompts for upload target (GitHub release asset for ≤2GB; external mirror URL otherwise).
- **Plugin interaction.** Native and Python plugins can register models via the same manifest format pointed at a local file — how third parties ship their own analyzers without going through the public registry.

### v1 deliverables this entry commits to

1. `ProcessingRun` table (replaces `ModelRun`) with the kind-discriminator schema enumerated above.
2. Model registry repo on GitHub (`sadda-speech/model-registry`) with three-tier structure mirroring refdist.
3. CI validation pipeline for model entries (manifest schema, license check, file checksum, optional smoke-test runner).
4. GitHub Pages-rendered index JSON consumable by the engine.
5. `sadda.ml.load_model` resolver covering curated / HF passthrough / local schemes.
6. ONNX-canonical curated-tier publishing path; HF passthrough as escape hatch.
7. Project pinning of model versions in `project.toml`.
8. Curated starter set: VAD (bundled), wav2vec2-base, whisper-tiny, whisper-base in Phase 3; tongue-contour-v1 in Phase 4.
9. User-level cache at `~/.local/share/sadda/models/` with manual GC command.

### Open trade-offs / deferred

- **HF passthrough auth UX.** Where does `HF_TOKEN` get configured (env, config file, GUI prompt)? Gated models need this; public ones don't. Defer until first real user hits the wall.
- **Mobile model story.** Phase 6. ONNX Runtime Mobile vs. ORT + quantization vs. whisper.cpp/llama.cpp variants. Likely its own DEVLOG entry when Phase 6 approaches.
- **Output verification on HF passthrough.** No manifest means no `output.tier_kind` declared. Engine has to either ask the caller to specify or sniff via convention. Probably caller-specified in v1.
- **Tongue-contour-v1 model choice.** Still open per articulatory entry. UNet vs. DeepLabCut-style vs. fine-tune of an existing release. Phase 3 risk spike addresses this.
- **Curated-tier CI smoke-test runner.** Running a model in CI costs CPU minutes; GPU CI is hard. Plausible v1: tiny synthetic input + checksum of expected output. Spec out before opening the registry repo for tier-3 submissions.
- **Plugin-shipped models in `ProcessingRun`.** Plugin-supplied models should still record with the same provenance fields; manifest path convention for plugin-shipped weights needs nailing down.
- **Cache eviction.** Manual at v1; LRU eviction with a configurable cap as a follow-up if disk usage becomes a real complaint.

### Cross-references / what this entry updates

- Closes "ML model registry scope" follow-up from the 2026-05-18 corpus entry.
- Closes the "bundled model distribution channel" question raised in the 2026-05-18 articulatory entry (item 7 in that entry's open trade-offs).
- Updates the corpus entity table: rename `ModelRun` → `ProcessingRun`.
- Refines `sadda.ml` surface from the 2026-05-18 API entry: adds explicit `load_model` resolver and the three ID schemes; `extract_embeddings` / `transcribe` / `vad` / `align` move from top-level `sadda.ml.*` to methods on the returned `Model` object.

### Sources / references

- HuggingFace Hub: https://huggingface.co/docs/hub ; huggingface_hub Python lib: https://huggingface.co/docs/huggingface_hub
- ONNX Runtime: https://onnxruntime.ai ; ort (Rust binding): https://ort.pyke.io
- ONNX Model Zoo: https://github.com/onnx/models
- MFA models registry: https://mfa-models.readthedocs.io
- Silero VAD: https://github.com/snakers4/silero-vad
- Whisper: https://github.com/openai/whisper ; whisper.cpp: https://github.com/ggerganov/whisper.cpp
- wav2vec2 (facebook/wav2vec2-base-960h): https://huggingface.co/facebook/wav2vec2-base-960h
- Hugging Face Optimum (format conversion): https://huggingface.co/docs/optimum

---

## 2026-05-20 — naming decision: project is named "sadda"

Closes the naming open-question from the milestone-plan entry. PyPI name is the first irrevocable public commitment, so this needs to land before any code reaches 0.0 internal — even though no code exists yet.

### Search path

Started from "lapa" (Pali, *talkative, chatty*) — direct echo of Praat's naming logic (Dutch *talk*). Two flavors of speech-words considered:

- **Informal/conversational verbs** (parallels Praat): lapa, jalpa, sallāpa, kathā, ālāpa
- **Technical phonological terms** (aligns with what the tool actually does): vāc, dhvani, svara, śabda, vacana, uccāraṇa

The conversational flavor matches Praat's spirit; the technical flavor matches the tool's actual subject matter. The conversational frame is what a user *does*; the technical frame is what the tool *analyzes*. Settled on the latter as more honest about the tool's nature — Praat's name was a friendly disguise on a deeply technical instrument, and we don't need that disguise in 2026.

### Collisions that killed candidates

PyPI screen against the longlist:

- `lapa` — taken (genomics tool, Mortazavi Lab; active; adjacent scientific-Python field — exactly the collision case to avoid)
- `dhvani` — taken, and the taker is itself a phonetics tool (Hinglish phonetic normalization, IPA-bridged). Hard kill: same field
- `svara`, `vac`, `shabda`, `nada`, `bhasha` — all taken
- `vacana`, `sallapa`, `vaak`, `alapa`, `jalpa`, `katha`, `vada`, `sadda` — all available

### Why sadda

Pali सद्द (cognate of Sanskrit śabda, displaced because śabda/shabda are taken on PyPI):

- **Phonetics-coded directly.** *Sadda-sattha* is the Pali term for the science of language/sound — the Pali grammatical tradition's own name for itself (Kaccāyana). It's not a metaphor; it's the indigenous technical term.
- **Phenomenologically primary.** In Pali Abhidhamma, sadda is one of the six sense-objects — the proper object of hearing. Means "the kind of thing this tool measures."
- **Pali matches the user's initial instinct** (entered via "lapa") while moving from the action-frame to the object-frame.
- **Phonotactically clean.** /sad.də/, two syllables, geminate /-dd-/ gives crisp distinction in search, trivial pronunciation across L1s.
- **Praat parallel preserved.** Both names pick from a tradition where the chosen word carries philological weight in the picker's eyes.

### Namespace status

| Asset | Status |
|---|---|
| PyPI `sadda` | available |
| PyPI `saddha` / `sadda-speech` / `pysadda` (qualified fallbacks) | available |
| `github.com/sadda` | taken (active personal account, Lukáš Adam) — GitHub org uses a qualifier instead |
| `github.com/sadda-speech` | **created as GitHub org** (2026-05-20) |
| `sadda.dev` | available |
| `sadda.org` | available |
| `sadda.io` | parked |

GitHub org: **`sadda-speech`** — descriptive, matches the `python-pillow` / `astral-sh` convention of qualifying the brand name when the bare slug is taken. Other candidates considered (`sadda-dev`, `sadda-tool`, `getsadda`) declined: `sadda-dev` too vague, `sadda-tool` undersells the scope, `getsadda` reads commercial-startup rather than open-source-scientific.

### What still needs doing

- **Sanskritist / Indologist sanity check.** Name was chosen with secondhand Pali knowledge; register and connotation in scholarly Theravāda / modern Pali-studies circles should be verified before formal commit. Not blocking — Pali is a textual liturgical language with limited modern-speaker connotation risk — but worth a five-minute check with someone in Buddhist studies.
- **Trademark search.** "Sadda" appears in Bollywood / Punjabi pop-cultural usage ("Sadda Haq") but not as a software / scientific-instruments mark, as far as casual search shows. A real USPTO / WIPO check should happen before the 0.1 release rather than now.
- ~~**GitHub org name lock-in.**~~ Closed 2026-05-20: `sadda-speech` chosen and org created same day.
- **PyPI name claim — defer to end of Phase 0, not now.** PyPI has no formal reservation; claiming the name means uploading a real release. Doing that today would mean a bare stub sitting on PyPI for 1–2+ years before a real release, which is a long PEP 541 ("project is squatting") exposure window for modest insurance against an unlikely independent collision (sadda is non-obvious in an obscure scholarly language; the slot has been free the entire history of PyPI). Plan: claim with `0.0.1` at end of Phase 0, when there's an actual installable artifact — which both closes the collision window and forecloses the squatting-vulnerability window in the same act. If a collision risk materializes earlier (someone publishes a *sadda* package on PyPI mid-Phase-0), revisit immediately.
- **Historical `sat` references in earlier DEVLOG entries left as-is** — they're the record of how thinking evolved during the placeholder period, not authoritative naming. Code from this point forward uses `sadda`.

### Module-import implication

The top-level Python module becomes `sadda`. Replaces `sat` in the API surface entry — `import sadda` rather than `import sat`; `sadda.dsp`, `sadda.corpus`, `sadda.app`, `sadda.recipe.record()`, `sadda.refdist`, etc. The earlier API-surface entry's code examples remain valid as patterns; just substitute the module name.

---

## 2026-05-18 — v1 milestone plan: vertical slice first, seven phases, incremental 0.x releases

Goal: sequence all the v1 commitments accumulated across the previous 9 entries into an executable plan. Unlike the other entries, this is a project plan — a working roadmap, not a binding architecture decision. It will drift; revisit periodically.

### Scale, named honestly

Pulled together across the prior entries, the v1 surface is enormous: Rust engine + egui desktop GUI + PyO3 Python module + iOS + Android UniFFI shells + Python + native plugin architecture + six tier types + video + live recording + DSP suite + ML inference + six clinical algorithms with validation + articulatory imports + ultrasound video with integrated tongue contour tracker + experiment runner + reference distribution registry + embedded CPython + packaging on three desktop and two mobile platforms.

**Pace assumption: solo part-time** (10–15 hours / week sustainable). Realistic v1.0 timeline at this pace: **~3–4 years**. Timeline compresses substantially if pace moves to full-time (~18–24 months) or if contributors join after the first 0.x release. The plan is built around this conservative pace; faster pace shortens phase durations but doesn't change phase ordering.

### Organizing principles

1. **Vertical slice first.** Phase 0 builds end-to-end through every architectural layer (engine → corpus → Python → GUI → UniFFI → embedded CPython) before any layer is built out in depth. If something is broken architecturally, we want to know in month 2, not month 14.
2. **Risk spikes run in parallel** with main-path phases. Small, contained, fail-fast experiments to settle unknowns before they block work.
3. **Incremental 0.x releases** at the end of each phase. Each release is a usable artifact for some real audience; not internal-only.
4. **Contributor onramp** progressively opens as APIs stabilize: Python library contributors after 0.1; GUI contributors after 0.2; plugin authors after 0.4.

### Phases and milestones

| Phase | Duration (solo part-time) | Release | What ships |
|---|---|---|---|
| **0. Vertical slice** | 1–2 mo | 0.0 internal | Engine + minimal corpus + PyO3 + WAV loader + autocorrelation pitch + egui waveform/pitch view + embedded CPython script panel + one UniFFI method to Swift CLI |
| **1. Foundations** | 3–4 mo | **0.1 — Python library on PyPI** | Full corpus schema + six tier types + sparse/dense storage split + live recording (cpal + JACK) + DSP suite (pitch / formants / intensity / spectrogram / MFCC / STFT) + TextGrid + EAF I/O + stability decorators + type stubs + reproducibility recipes + migration framework + Apache-2.0 OR MIT dual license |
| **2. Desktop GUI** | 3–4 mo | **0.2 — desktop GUI** | Egui+wgpu app shell + project navigator + waveform/spectrogram/tier-strip with sync cursor + interval/point tier editing + embedded CPython in app + `sat.app` basics (selection, register_command) + single-writer lock |
| **3. Differentiators part 1** | 3–4 mo | **0.3 — clinical-ready** | Reference distribution infrastructure (format + resolver + GitHub registry + CI + Pages index + in-app publish) + bundled starter set + GUI overlay rendering + clinical algorithms (AVQI / ABI / CPP / jitter / shimmer / HNR) with validation suite + provenance + uom typed units + calibrated SPL + research-use-only labeling + ort runtime + VAD bundled + wav2vec2/Whisper on-demand download + embedding tiers |
| **4. Articulatory** | 5–7 mo | **0.4 — articulatory** | Plugin architecture (Python + native via `abi_stable`) + EGG + Carstens AG501 importer + `video_aligned` tier + channel semantics schema fields + ffmpeg-rs decoder + video player widget + synced multi-pane layout + **tongue contour tracker** (model + per-frame inference + correction UI + validation) |
| **5. Experiment runner + scripting depth** | 2–3 mo | **0.5 — beta** | Trial sequencing (linear / randomized / block / counterbalanced) + simple sequence-editor GUI + shipped templates (CAPE-V / Rainbow / sustained-vowel / discrimination) + drill mode + PsychoPy CSV import + best-effort PsychoPy export + `register_panel` bridge (Markdown / table) |
| **6. Mobile** | 3–4 mo | **0.6 — mobile beta** | UniFFI bindings for mobile API surface + iOS shell (SwiftUI) + Android shell (Compose) — both with record + live feedback + sync + bundle pack export + store internal-testing setup |
| **7. Polish & release** | 2–3 mo | **1.0** | macOS signing + notarization + Windows installer + Linux packaging (AppImage + Flatpak) + embedded CPython distribution validation + docs site (mkdocs-material + mkdocstrings probably) + Parselmouth migration docs + notebook tutorials + bundled sample projects + bug bash + 1.0 release |

**Total: ~22–31 months solo part-time.** Compresses to ~14–18 months full-time, or shorter with contributors past 0.1.

### Parallel risk spikes

Run alongside main phases — small, fail-fast experiments:

| Spike | Concurrent with | Purpose |
|---|---|---|
| **UniFFI proof** | Phase 0 | One method bridged to Swift CLI end-to-end. Validates mobile architecture before any shell work |
| **Embedded CPython distribution** | Phase 1 | One signed macOS `.app` with embedded Python running a real script. Validates packaging story early |
| **Native plugin ABI** | Phase 3 (before Phase 4 starts) | One trivial native plugin loaded from a dylib. Settles `abi_stable` vs hand-curated C ABI vs WASM components |
| **Tongue segmentation model exploration** | Phase 3 | Survey available models (UNet variants, DeepLabCut-style, HuggingFace tongue-segmentation releases), fine-tune candidates on small dataset. De-risks the heaviest single Phase 4 deliverable |

### Critical path and dependencies

- Phase 0 unblocks everything — without the vertical slice working, no architecture can be trusted
- Engine + corpus + Python bindings (Phases 0–1) are the spine; GUI, mobile, plugins, refdist all depend on this
- Reference distribution infrastructure (Phase 3) has to land before clinical because clinical features display against norms
- Plugin architecture (Phase 4 start) has to land before the AG501 importer because the importer IS the first plugin
- Mobile (Phase 6) depends on UniFFI being proven (Phase 0 spike) and on the engine API being stable enough to bridge — tier 1 of stability decorators

### Confirmed scope decisions for v1.0

| Item | Decision | Reasoning |
|---|---|---|
| Tongue contour tracker | **Core, in Phase 4** | Held per articulatory entry. Most ambitious single v1 commitment. Phase 4's 5–7 month estimate accounts for it |
| Mobile platforms at 1.0 | **Both iOS + Android** | Per tech-stack entry. Cross-cutting pattern B (mobile as structural gap) preserved at v1.0 |

### Cut lines if timeline pressure hits

In priority order of what to defer first (most→least):

1. Tongue contour tracker → community-deliverable plugin instead of core (the articulatory entry's flagged scope concern)
2. Android → iOS-only at 1.0; Android in 1.x
3. Drill mode → defer to 1.x
4. PsychoPy script export → import-only at 1.0; export later
5. `register_panel` for arbitrary widgets → `register_command` only at 1.0
6. Some refdist starter-set entries → ship with fewer
7. emuDB export → defer to plugin / 1.x
8. EAF round-trip → import-only at 1.0; export later

**Not cuttable**: engine, corpus, Python bindings, basic DSP, basic GUI, reference distribution infrastructure, clinical algorithms + validation, EGG, packaging on three desktop platforms, the Python library on PyPI by 0.1.

### Contributor-onramp progression

- **After 0.1**: Python library contributors — DSP, ML, format support, validation tests
- **After 0.2**: GUI / UX contributors — egui-side widgets, theme work, annotation editing affordances
- **After 0.3**: broader contributors with stable APIs and lower breakage churn risk
- **After 0.4**: plugin authors can build importers / analyzers independently — biggest contributor unlock; community can fill the modality long tail (EPG, aerodynamic, additional EMA vendors)

### Cross-cutting work happening every phase

- Documentation grows incrementally per phase, not as a separate final phase
- Validation test corpus grows with each clinical algorithm in Phase 3
- Reference distribution curation ongoing (identify candidates, verify licensing)
- Community engagement ongoing pre-v1: Praat / Parselmouth / phonetics-list visibility
- Stability decorator hygiene: every public function/class consciously tiered as added

### Open questions / deferred

- ~~**Naming.**~~ Closed 2026-05-20: project named `sadda` (Pali, *sound / voice*). See 2026-05-20 entry. GitHub org qualifier and trademark check still pending.
- **Public repo structure.** Single monorepo vs split (engine / Python / GUI / mobile shells / registry). Probably monorepo for v1 era, split if scale demands. Decide before Phase 1 ends.
- **Continuous integration scope.** GitHub Actions matrix on three desktop + two mobile platforms is substantial; minimal CI covering build + test + lint suffices through Phase 1; expand from Phase 2.
- **Funding model.** Solo part-time may sustain through 0.3 or so; further depends on grants / sponsorship / commercial backing. Not architectural, but real for timeline.
- **Beta-tester recruitment timing.** Probably starts at 0.2 (desktop GUI usable); 0.3 has the clinical-research audience attached.
- **Sample data licensing.** Need a small, clean-licensed audio set for tutorials and CI tests — bundled with the docs. Sourcing TBD.
- **Translation / i18n.** Deferred entirely from v1. Strings live in source for now; structure for extraction can be added later without re-architecture.
- **Accessibility (screen readers, keyboard nav).** egui's AccessKit integration is young; v1.0 best-effort, dedicated pass in 1.x. Worth marking the limitation explicitly.

### How this plan should be revisited

- After Phase 0 ends: if any spike failed, re-architect before Phase 1.
- After every phase: assess actual time vs estimate; recalibrate later phases proportionally.
- After 0.2: real users land; their feedback may reshape Phase 3+ scoping.
- After 0.3: clinical-research adoption will surface validation requirements (or kill them as priorities).
- Annually: scope cut-line list re-evaluated against actual capacity.

This plan is the working roadmap, not a contract. Re-read at the start of each phase; adjust openly.

### Sources / references

- The other 2026-05-15 through 2026-05-18 entries (this plan is downstream of all of them)
- "Tracer Bullets" — Pragmatic Programmer (vertical-slice principle): https://pragprog.com/tips/
- SemVer (for 0.x cadence semantics): https://semver.org

---

## 2026-05-18 — Python API surface: shape, stability tiers, conventions

Goal: design the Python API surface. Flagged as an open item in the tech-stack entry — once Python is in v1, the API contracts users write scripts against become hard to change. Better to design once than retrofit.

### Two audiences, one surface

- **Library users** (Parselmouth-replacement) — script analyses against a corpus or against raw audio, no GUI. AI engineers + scripting phoneticians.
- **In-app scripting host** — automate workflows inside the desktop app; build custom analyses or panels.

Same module everywhere; one extra namespace (`sat.app`) is populated only when running inside the desktop process.

### Lessons from prior art

| Source | Lifted | Left |
|---|---|---|
| **Parselmouth** | Top-level convenience namespace; NumPy-native; escape hatches | Tied to Praat object model (we're not Praat-compat at script level) |
| **librosa** | Flat-ish functional DSP API; NumPy first; consistent shapes | Almost no state model; no corpus concept |
| **scikit-learn** | Convention consistency (fit/transform/predict everywhere) | Domain mismatch; no single repeating pattern that wide |
| **Blender bpy / Houdini hou** | In-app namespace mirrors GUI data model; build UI panels from scripts; clear library-vs-interactive distinction | Very stateful; we keep DSP/analysis stateless where possible |
| **emuR / wrassp** | Corpus query → typed result; signal-as-ndarray | R conventions don't translate |

Synthesis: **functional NumPy-native API for DSP/analysis; OO for corpus/project; one in-app namespace clearly separated.**

### Module layout

```
sat                               (top-level; `sat` is a placeholder — naming deferred)
├── __version__
├── open_project(path) → Project
├── new_project(path, profile=None) → Project
│
├── corpus                        # OO, STABLE
│   ├── Project, Session, Bundle, Tier
│   ├── Annotation (Interval/Point/Reference variants)
│   ├── DerivedSignal, Entity, EntityRef
│   ├── ModelRun, AuditLog
│   └── query(...) → polars.DataFrame
│
├── dsp                           # functional, stateless, STABLE
│   ├── pitch, formants, intensity, spectrogram
│   ├── mfcc, stft, lpc, autocorrelation, cepstrum
│   └── windowing, resampling, filtering
│
├── clinical                      # STABLE (per regulatory entry)
│   ├── avqi, cpp, cpps
│   ├── jitter, shimmer, hnr, nhr
│   └── (each returns value + uncertainty + algorithm_version)
│
├── ml                            # PROVISIONAL
│   ├── load_model, extract_embeddings
│   ├── transcribe, vad, align
│
├── refdist                       # STABLE (per refdist entry)
│   ├── get, search, publish
│   └── registry
│
├── articulatory                  # PROVISIONAL
│   ├── import_carstens, import_video
│   ├── tongue_contour
│   └── egg.{open_close, oq, cq, …}
│
├── experiments                   # PROVISIONAL
│   ├── Experiment, Trial, Block
│   ├── load_template
│   └── psychopy.import_log / psychopy.export
│
├── live                          # PROVISIONAL
│   └── start_session (with subscriber decorators)
│
├── plugins                       # PROVISIONAL
│   └── register_importer / register_analyzer / register_command
│
├── app                           # IN-APP ONLY; NotInAppError otherwise
│   ├── current_selection, active_bundle, active_project
│   ├── register_command, register_panel
│   ├── refresh()
│   └── prompts (file pickers, dialog boxes)
│
├── recipe                        # reproducibility
│   ├── record() context manager
│   ├── replay(path)
│
├── experimental                  # explicitly unstable
│
└── errors
    └── SatError → CorpusError, AnalysisError, ModelError,
                   NotInAppError, PluginError, MigrationError
```

### Data type conventions

| Concept | Type | Notes |
|---|---|---|
| Audio buffer | `np.ndarray` (float32) | `(n_samples,)` mono; `(n_channels, n_samples)` multichannel |
| Sample rate | `int` (Hz) | — |
| Time stamps | `float` (seconds) | Not frames, not Praat xmin/xmax |
| Frequencies | `float` (Hz) | — |
| Spectrograms | `np.ndarray` `(n_freq_bins, n_frames)` + metadata wrapper | Hop / window / fmin / fmax on the wrapper |
| Tier data | Tier object wrapping NumPy or Parquet | `.to_numpy()` / `.to_dataframe()` escape hatches |
| Corpus query results | `polars.DataFrame` (primary) | `.to_pandas()` escape hatch; Arrow-native; zero-copy from Rust; matches our Parquet sidecars |
| Reference distributions | `Distribution` object | `.samples` / `.summary` NumPy escape hatches |

### Stability tiers

Three tiers, marked **in code with decorators** that emit one-time runtime warnings on first use:

- **Stable** — `corpus`, `dsp` (basic measures), `clinical`, `refdist`, top-level project loaders. Commits not to break across minor versions.
- **Provisional** — `ml`, `articulatory`, `experiments`, `live`, `plugins`, `app`. May break in minor versions after a deprecation cycle.
- **Experimental** — under `sat.experimental.*`. May break anytime.

Mechanism: `@stable`, `@provisional`, `@experimental` decorators on every public function/class. First use of a non-stable API emits a `ProvisionalAPIWarning` / `ExperimentalAPIWarning` (suppressible via the standard `warnings` machinery). Surfaces dependency risk at coding time rather than after the fact.

### GIL, async, error model

- **GIL released** during all long Rust operations (already committed in tech-stack entry).
- **Sync by default** — matches scientific-Python expectations. Async wrappers for I/O-bound things (network refdist fetches, sync inbox watchers) deferred to v1.x; not in initial v1.
- **Real-time / streaming** via subscriber decorators (`@session.on_pitch`) or async iterators. Not asyncio-based in v1.
- **Errors** as a typed exception hierarchy rooted at `SatError`. Rust `Result` types map to specific exception classes via PyO3 conversion traits.

### Reproducibility

`sat.recipe.record()` context manager: every analysis call inside the block logs to the project's `ModelRun` + `AuditLog` tables, and the recipe can be saved as a Python script for replay. Connects the cross-cutting reproducibility need surfaced in corpus + refdist + clinical entries.

```python
with sat.recipe.record(proj, name="vowel_analysis_v1"):
    for bundle in proj.bundles:
        sat.dsp.formants(bundle.audio, bundle.sr, num_formants=5)
# project now has a saved recipe; reproducible via sat.recipe.replay()
```

### In-app surface

`sat.app` is populated only when running inside the desktop process. Outside, accessing it raises `NotInAppError`. Scripts that don't touch `sat.app` work identically in both modes.

In-app scripts can: read GUI state (selection, active bundle), register commands (added to a menu / palette), register custom panels (egui-side hooks via a stable bridge), trigger refreshes after mutations, call file pickers and dialog boxes.

### Representative usage

Library:

```python
import sat
proj = sat.open_project("/path/to/myproject")
for bundle in proj.bundles:
    pitch = sat.dsp.pitch(bundle.audio, bundle.sr, method="autocorrelation")
    bundle.add_tier("F0", type="continuous_numeric", data=pitch.values, hop=pitch.hop)
proj.save()
```

Clinical with reference comparison:

```python
result = sat.clinical.avqi(audio, sr)
print(result.value, result.uncertainty, result.algorithm_version)
norm = sat.refdist.get("voiceevalu8-avqi-norms", version="2.1.0")
percentile = norm.percentile_rank(result.value, population={"sex": "f", "age_band": "adult"})
```

Live (voice-coach):

```python
session = sat.live.start_session(device="default", sr=44100)
target = sat.refdist.get("trans-feminine-resonance-targets")

@session.on_formants
def show(formants, time):
    print(f"F1={formants[0]:.0f} F2={formants[1]:.0f} in_target={target.contains(formants)}")

session.start_recording(target_bundle=proj.new_bundle("practice"))
```

In-app:

```python
import sat, sat.app as app

@app.register_command("Compute custom feature")
def my_analysis():
    bundle = app.active_bundle
    audio = bundle.audio_slice(app.current_selection)
    bundle.add_tier("custom_feature", type="continuous_numeric",
                    data=my_dsp_function(audio))
    app.refresh()
```

### v1 deliverables this entry commits to

1. Module layout as enumerated above.
2. Data type conventions: NumPy primary for signals, Polars primary for corpus query results, with documented escape hatches.
3. Three-tier stability marking via in-code `@stable` / `@provisional` / `@experimental` decorators that emit runtime warnings.
4. GIL released during long Rust ops; sync-by-default API.
5. Typed exception hierarchy rooted at `SatError`.
6. `sat.recipe.record()` reproducibility context manager.
7. `sat.app` namespace populated only inside the desktop process.
8. Type stubs (`.pyi`) generated alongside the module; `py.typed` marker present.
9. Rich `__repr__` (including HTML for Jupyter) on major types.
10. Migration documentation for Parselmouth users (no compat shim — migration docs only).

### Open trade-offs / deferred

- **Top-level module name** — `sat` is a placeholder (collides with the SAT-solver namespace and is overly generic). Naming is its own discussion; defer until closer to release.
- **Praat-script compatibility** — explicitly out of v1 per 2026-05-16 question 5. Worth revisiting after v1 if migration friction proves high.
- **Async / asyncio support** — sync-first in v1; async wrappers for I/O-bound APIs (network fetches, sync inboxes) layered later.
- **Tab-completion / IDE polish** — type stubs + `py.typed` are the v1 floor; richer IDE integration (Pylance plugins, VS Code extensions) is a v1.x consideration.
- **In-app panel API** — `app.register_panel` exists but the panel-rendering bridge between Python callbacks and egui's frame loop needs a focused design. The simple case (register a Markdown / table view) is straightforward; arbitrary widgets are harder.
- **Plugin authoring SDK vs the regular API** — current decision: plugins use the same `sat` API + `sat.plugins.register_*` registration. Whether plugins need a richer SDK (lifecycle hooks, capabilities declaration beyond what `register_*` carries) is a follow-up.
- **Migration helpers for non-Python tools** — a small `sat.io.praat` module for TextGrid round-trip (covered in corpus entry) and basic Sound-object compatibility (`sat.io.praat.read_sound`) likely worth shipping; full Parselmouth-shaped object wrappers are not.
- **Notebook tutorials and documentation infrastructure** — mkdocs-material + mkdocstrings, or sphinx-immaterial? Pick before docs work starts in earnest.
- **Sample-data distribution for tutorials** — small bundled audio files for docs to work without external downloads; pick a license-clean source.

### Sources / references

- PyO3: https://pyo3.rs
- pyo3-stub-gen (type stub generation from PyO3): https://github.com/Jij-Inc/pyo3-stub-gen
- Parselmouth: https://parselmouth.readthedocs.io
- librosa API design: https://librosa.org/doc
- scikit-learn API design glossary: https://scikit-learn.org/stable/glossary.html
- Blender bpy: https://docs.blender.org/api/current/
- Houdini hou: https://www.sidefx.com/docs/houdini/hom/
- Polars: https://pola.rs
- Apache Arrow: https://arrow.apache.org
- PEP 484 (type hints): https://peps.python.org/pep-0484/
- py.typed marker (PEP 561): https://peps.python.org/pep-0561/

---

## 2026-05-18 — Experiment runner: production/protocol/drill focus, desktop-only in v1

Goal: settle scope for the experiment runner — the last open question from 2026-05-16. The decision is not *whether* (cross-cutting pattern I from 2026-05-16 already said yes), but *how deep*: PsychoPy-equivalent depth or Demo Window+ minimum?

### What our user groups actually need from an experiment runner

| Group | Use case | Demand profile |
|---|---|---|
| 8. Psycholinguists | Perception experiments | High — full PsychoPy territory (frame-accurate visual timing, RDKs, eye-tracker integration, web deployment) |
| 1, 6. Phoneticians, field linguists | Production / elicitation (prompt → record) | Low–moderate — text/audio/image prompt + record + tag with trial metadata |
| 3. Clinicians | Protocol-driven workflows (CAPE-V, Rainbow Passage, sustained vowels at fixed F0) | Low — sequential templates with auto-tag, calibrated recording |
| 4, 5. Voice coaches, L2 learners | Drill structure (model → attempt → feedback) | Low–moderate — repetition, comparison feedback, progress tracking |

**Five of six use cases are variants of "present a prompt, record the response, tag it with trial metadata."** Only group 8 needs the deep perception-research surface PsychoPy excels at.

### Defensible position vs PsychoPy

What we should build (unique to us):
- Trial recordings ARE corpus bundles, immediately analyzable in the same project (cross-cutting D — convergence of the record-analyze pipeline)
- Real-time analysis / feedback during practice (voice-coach / L2)
- Calibrated SPL during clinical protocols (cross-references the clinical regulatory entry)
- Articulatory data captured *during* experiments (EMA / ultrasound, via the articulatory entry)
- Reference distributions overlaid on live response visualization

What we should not try to match (PsychoPy does well):
- Frame-accurate visual timing
- Vision-science stimuli (RDKs, gratings, faces, movies)
- Eye-tracker integration
- Specialized hardware (button boxes, parallel/serial port triggers)
- Web deployment via Pavlovia
- Mature visual Builder UI for arbitrary trial structures

**The v1 commitment: a production-and-protocol-oriented experiment runner that converges the record-analyze pipeline. Leave perception-research depth to PsychoPy. Interop both directions.**

### v1 capabilities

**Stimulus types**
- Text (any length, formatted)
- Image (static)
- Audio (sample-accurate scheduling via `cpal`)
- Multi-modal combinations (text + audio, image + audio, …)

**Response types**
- Keyboard / mouse (egui native)
- Audio recording of the participant (real-time path already committed)
- Categorical selection (button-press equivalent in GUI)
- Timed vs untimed responses

**Trial sequencing**
- Trial = (stimulus spec, response spec, optional time limit, optional metadata)
- Sequences: linear, randomized, block-structured
- Basic counterbalancing (Latin square); complex factorial designs via the Python API
- Per-trial logging into corpus

**Protocol templates**
- Shipped: CAPE-V, Rainbow Passage, sustained-vowel at fixed pitches, basic discrimination tasks
- User-extensible: templates as data, editable in the GUI

**Drill mode (voice-coach / L2)**
- Model presentation → user attempt → comparison feedback
- Spaced repetition logic
- Progress tracking across sessions

**Builder UI**
- Simple sequence editor: drag-and-drop linear sequence of trials; per-trial config inspector; block structure via grouping
- Complex trial structures via the Python API rather than a PsychoPy-Builder-equivalent visual editor

**Interop**
- Import: PsychoPy CSV trial logs → ingested as metadata for matching recordings in the corpus
- Export: experiment definition → basic PsychoPy script scaffold (best-effort, no fidelity guarantees)

**Explicitly out of v1 scope**
- Vision-science stimulus types (shapes, gratings, RDKs, faces, movies)
- Specialized hardware integration (eye-trackers, button boxes, serial / parallel triggers)
- Frame-accurate visual timing guarantees (best-effort frame-rate timing only)
- Web deployment of experiments — see below
- Full PsychoPy-Builder-equivalent visual editor

### Browser deployment — deferred to v2+

The 2026-05-16 architectural implication #7 left browser as a deferred third surface. Confirming that stance for experiments specifically: **defer browser deployment to v2+.**

Rationale:
- Crowdsourced experimenters (Prolific / MTurk audiences) already use jsPsych, PCIbex, Gorilla, Pavlovia. We don't need to compete on browser-deployed experiments to be useful.
- A browser surface is meaningful additional build: TypeScript runtime, hosting infrastructure, participant-flow UX (consent, instructions, completion), cross-browser audio quality testing, privacy considerations under HIPAA / GDPR for participant audio in transit.
- It also complicates the local-first principle (#10) — first hosted-component dependency in the stack.
- v1 interop strategy already covers this audience: their existing browser-tool trial logs land in our corpus as metadata for downstream analysis.

When browser deployment becomes a v2+ priority, the architectural shape will likely mirror the mobile companion: a thin TypeScript client that records + uploads to a sync endpoint, with all analysis still happening on the experimenter's desktop project. No Rust-engine-in-WASM required.

### Architectural integration with prior entries

This entry barely costs anything because the existing data model accommodates it:

1. **`Trial` is a generic `Entity`** (`kind = "trial"`) with `extra` JSON for stimulus / response specifications. Already in the corpus entry.
2. **Experiment recordings are normal `Bundle`s** linked to trials via the `reference` tier type or `EntityRef`. Trial metadata flows directly into downstream analyses.
3. **Stimulus presentation runs in the GUI** (egui), using the same wgpu painter / audio scheduler used elsewhere.
4. **Sample-accurate audio scheduling** uses `cpal`'s callback-based output — already in the tech-stack commitments.
5. **Per-trial recording** uses the same atomic `.in_progress/` → commit path from the corpus entry.
6. **Real-time feedback during drills** (voice-coach use case) uses the live-analysis path already committed.
7. **Clinical protocol templates** integrate with calibrated SPL from the regulatory entry.
8. **Articulatory data during experiments** uses the same import path as standalone articulatory recording.

No new infrastructure required — just a stimulus-scheduling layer and a trial-sequencing engine on top of what we're already building.

### v1 deliverables this entry commits to

1. Stimulus types: text, image, audio (single + multi-modal).
2. Response types: keyboard, mouse, audio recording, categorical selection (timed / untimed).
3. Trial-sequencing engine: linear, randomized, block-structured, with basic counterbalancing.
4. Simple sequence-editor GUI (drag-and-drop, per-trial inspector, block grouping).
5. Shipped protocol templates: CAPE-V, Rainbow Passage, sustained-vowel-at-pitch, basic discrimination.
6. User-editable template format.
7. Drill mode with comparison feedback and spaced repetition for voice-coach / L2 use.
8. Per-trial logging into the corpus via `EntityRef` to a `Trial` entity.
9. PsychoPy CSV trial-log import → corpus metadata.
10. Best-effort PsychoPy script export from experiment definitions.

### Open trade-offs / deferred

- **Browser-deployable experiments → v2+.** Re-evaluate when the v1 mobile-companion architecture has stabilized; the TypeScript thin-client pattern likely transfers.
- **Specialized hardware integration (eye-trackers, button boxes, serial / parallel triggers)** → plugins (uses the plugin architecture committed in the articulatory entry). Native plugins ideal for low-latency hardware paths.
- **Advanced experimental designs (complex factorials, adaptive procedures, staircases)** → via Python API in v1; visual builders for these are heavy and PsychoPy already does them well.
- **Vision-science stimuli (RDKs, gratings, faces, movies)** → defer indefinitely; not in any of our scoped audiences.
- **Frame-accurate visual timing guarantees** → defer; would require lower-level GPU compositor integration than egui currently provides.
- **Participant identity / privacy in trial data** → straightforward (uses the corpus's `Speaker` entity with appropriate `extra` schema), but worth its own pass for clinical-experiment use cases under the regulatory entry's posture 3.
- **Drill mode UX details** → spaced-repetition algorithm choice, comparison-feedback visualization style, progress-tracking metric design. Iterate with voice-coach / L2 collaborators.
- **Protocol template authorship** → who curates the shipped templates (CAPE-V is well-defined; voice-coach drills less so). May fold into the reference-distribution registry pattern or warrant its own template registry.

### Sources / references

- PsychoPy: https://www.psychopy.org
- Pavlovia (browser deployment for PsychoPy): https://pavlovia.org
- jsPsych: https://www.jspsych.org
- PCIbex: https://www.pcibex.net
- Gorilla: https://gorilla.sc
- LabVanced: https://www.labvanced.com
- CAPE-V (Consensus Auditory-Perceptual Evaluation of Voice): https://www.asha.org/practice-portal/clinical-topics/voice-disorders/

---

## 2026-05-18 — Clinical regulatory stance: research-use-only, fork-friendly clinical readiness

Goal: settle the regulatory posture — pursue FDA / CE clearance pathways ourselves, or stay research-only and let downstream forks pursue clearance. Carried open from 2026-05-16.

### Five postures considered

| Posture | What it means | Cost | Fit |
|---|---|---|---|
| 1. Pure research use | Explicit "not for clinical use" labeling; clinical features may exist but no clinical claims; Praat's de facto posture | Minimal | Easy default |
| 2. Research-only, no clinical-design intentionality | Same as (1), no special effort to be clinically defensible | Minimal | Cheapest |
| 3. Research-only, fork-friendly clinical readiness | Designed so a downstream commercial entity could fork and pursue clearance without re-architecting | Modest engineering hygiene | Recommendation |
| 4. Self-pursued clearance | Formal 510(k) / CE under MDR, QMS (ISO 13485 / IEC 62304 / IEC 14971), V&V, post-market surveillance | $200k–$2M+ initial, ongoing labor | Not viable for our shape |
| 5. Cleared and shipped | Actually a medical device | Even more | Not viable |

Postures 4 and 5 require a corporate vehicle, regulatory consultancy, and ongoing burden that competes with development. They are not viable for a research-driven open-source project of our shape. Almost every academic phonetics / speech-science tool sits at posture 1 or 2; CSL / MDVP / VoxMetria / Visi-Pitch carry the regulatory burden separately as proprietary commercial products.

### Decision: posture 3

The clinical user group (group 3 from 2026-05-16) needs AVQI / ABI / CPP / jitter / shimmer / HNR, CAPE-V protocols, calibrated SPL, EGG sync, phonetograms, longitudinal patient viz, report generation, HL7 / FHIR export. **None of these require clearance to implement.** They can ship as research-use-only features. The regulatory posture is about *claims and design intentionality*, not about whether features exist.

Posture 3 ships clinical features as research-use-only, builds the engine with the discipline a downstream forking entity would need to pursue clearance, and avoids the regulatory burden ourselves.

### Architectural commitments under posture 3

Mostly good engineering hygiene anyway, with extra rigor for clinical-path code:

1. **Algorithm provenance metadata.** Every measure records which algorithm version was used, with citation to published reference. Plumbed through the ML model registry from the corpus entry.
2. **Validation test suite shipped with engine.** For clinical-style measures, ship known-input known-output tests comparing against published reference implementations and reference values. A forking entity can run them as the seed of a V&V package.
3. **Calibration as first-class.** Calibrated SPL, mic profiles, sensitivity / frequency response curves attached to the `Instrument` entity already in the corpus model. Without calibration, SPL-based measures are nonsense in clinical contexts.
4. **Units pervasive, type-level where possible.** Crates like `uom` give compile-time unit checking; modest overhead, high payoff for clinical paths. Never bare numbers without units.
5. **No silent fallbacks on clinical measures.** If a measure can't be computed reliably (insufficient signal, bad input), return an error or explicit uncertainty — not a guess. False confidence is harmful in clinical contexts.
6. **Regression testing for clinical algorithms in CI.** Numerical drift detected before release; surprising changes are gated on explicit acknowledgement.
7. **Stable-API marking.** Certain APIs designated as "stable for clinical use," with stronger change-control discipline than the rest of the engine. Cross-references the API surface concern from the tech-stack entry.
8. **Audit trail at the analysis level.** Already committed in the corpus entry (`AuditLog`). Every analysis run records inputs, algorithm versions, parameters, outputs. Forensic and clinical share this need.
9. **Documentation of intended use.** Explicit Intended Use statement even under research positioning: what we claim, what we don't, scope of validation, known limitations.
10. **Research-use-only labeling.** At app startup, in documentation, on clinical-style features: "For research, education, and non-diagnostic use only."

### v1 clinical scope

| Feature | v1 | v1.x | Defer |
|---|---|---|---|
| Clinical-style algorithms (AVQI, ABI, CPP, jitter, shimmer, HNR) with validation suite | ✓ | | |
| Calibrated SPL + calibrated mic profiles in `Instrument` | ✓ | | |
| EGG sync + basic EGG analysis (OQ, CQ) | ✓ (via articulatory entry) | | |
| Patient-as-`Entity` with `kind = "patient"`, longitudinal viz hooks | ✓ (via corpus entry) | | |
| Sustained-vowel + connected-speech protocol workflows | | ✓ | |
| CAPE-V / Rainbow Passage protocol templates | | ✓ | |
| Phonetogram / Voice Range Profile renderer | | ✓ | |
| FHIR / HL7 export | | ✓ | |
| Clinical report templates | | ✓ | |
| EHR integration | | | Plugin or v2+ |

Rationale for the v1 cut: ship the analysis engine and calibration in v1 because they're foundational and have the highest validation-design cost. Workflow features (protocols, report templates, FHIR export) wait until we've talked to enough clinical researchers to know what shapes are actually used — these are easy to get wrong without that input, and the cost of getting them wrong is real (workflows are sticky).

### The license tension — noted, not resolved

Posture 3 only works if our license permits commercial forks: permissive (Apache-2.0 OR MIT) or weak copyleft (MPL-2.0). Under GPL-3.0+, the fork-and-clear pathway is effectively closed — clinical vendors don't fork GPL code because they don't want to release their derivative work under GPL, and FDA-cleared products are almost never GPL.

So **posture 3 + GPL is internally inconsistent.** If we land on GPL later, clinical productization by third parties is foreclosed and posture 3 effectively collapses to posture 1.

**The license decision is not forced by this entry.** It remains deferred per the licensing entry's stance. This tension is now documented and feeds into the eventual license call. Concretely: if the GPL ambition is preserved through to release, we should re-read this entry and decide whether to (a) accept the collapse to posture 1 or (b) shift the license away from GPL to preserve posture 3.

### What posture 3 explicitly does NOT do

To be precise about the negative space:

- Does NOT establish a Quality Management System (ISO 13485)
- Does NOT formally document software lifecycle processes (IEC 62304)
- Does NOT do formal risk management (ISO 14971)
- Does NOT make therapeutic, diagnostic, or treatment claims
- Does NOT take liability for clinical decisions made using the tool
- Does NOT prevent clinical research use — "research use only" by long convention includes clinical-research contexts

### v1 deliverables this entry commits to

1. Research-use-only labeling at app startup, in docs, on clinical features. Explicit Intended Use statement.
2. Validation test suite for AVQI, ABI, CPP, jitter, shimmer, HNR comparing against published reference values, runnable from the engine and from CI.
3. Algorithm provenance metadata on every measure, with citation references.
4. Calibrated SPL + mic profile support on the `Instrument` entity.
5. `uom`-style typed units on the clinical-path API.
6. No-silent-fallback discipline on clinical measures (explicit error / uncertainty types).
7. Regression testing for clinical algorithms in CI with numerical-drift gates.
8. Stable-API marking convention for clinical surface.
9. Audit-trail coverage of clinical analyses (already committed via corpus entry; confirmed for clinical scope).

### Open trade-offs / deferred

- **Clinical workflow features (v1.x).** Protocol templates (CAPE-V, Rainbow Passage), phonetogram renderer, FHIR / HL7 export, report templates. Defer until we've validated workflow shape with clinical-research users. Risk: clinical adoption may be limited in v1 without these.
- **Algorithm validation reference sources.** Each clinical measure needs a chosen reference implementation to validate against: AVQI against Maryn et al. (publishes Praat scripts); CPPS against Hillenbrand / Heman-Ackah; jitter / shimmer against Praat and MDVP. Need a focused entry on which references we adopt and how we resolve disagreement between them (Praat and MDVP do not always agree on the same algorithm).
- **Per-measure uncertainty quantification.** Returning "value + uncertainty" rather than "value" is the right shape for clinical use, but requires per-measure work to define what uncertainty means. v1 starts with explicit error returns on unreliable inputs; richer uncertainty modeling layered in later.
- **Recruiting clinical-research collaborators.** Posture 3's value depends partly on whether clinical researchers actually use the tool. Outreach / partnership thinking is a non-engineering question worth its own entry once v1 is closer.
- **Position on supporting a future commercial fork pursuing clearance.** Are we willing to provide upstream stability commitments, validation-data access, or co-signed test reports for a serious forking entity? Affects how stable-API marking is enforced and how much support burden we accept.
- **Plugin-clinical interaction.** Clinical-path code should probably be restricted to in-tree or audited plugins — a community plugin computing jitter incorrectly is a different risk profile than a community plugin computing an experimental feature. Worth a policy.
- **Documentation of "known limitations" per measure.** Posture 3 expects honest documentation of what each measure does and doesn't validate. Mechanical work but real.

### Sources / references

- FDA Software as a Medical Device (SaMD): https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd
- IMDRF SaMD documents: https://www.imdrf.org/working-groups/software-medical-device-samd
- EU MDR (2017/745): https://eur-lex.europa.eu/eli/reg/2017/745/oj
- FDA 510(k): https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/premarket-notification-510k
- IEC 62304 (medical device software lifecycle) and ISO 13485 (QMS for medical devices) and ISO 14971 (risk management): ISO/IEC standards, not free
- AVQI (Maryn et al.) reference Praat scripts: https://www.vvl.be/en/avqi
- CPPS background (Hillenbrand): https://homepages.wmich.edu/~hillenbr/
- `uom` (units of measurement for Rust): https://github.com/iliekturtles/uom

---

## 2026-05-18 — Articulatory channels: first-class data model, plugin-driven vendor glue

Goal: settle whether ultrasound / EMA / EPG / EGG / aerodynamic / video modalities are first-class signals, plugins, or out of scope. The question was carried open from 2026-05-16.

### The modalities and their engineering shape

Stepping back from what each one *measures* and looking at what each one *demands*:

| Modality | Data shape | Sample rate | Vendor lock | Sync need |
|---|---|---|---|---|
| **EGG** | Single channel, audio-rate | 16–96 kHz | Mild | Trivial — usually a 2nd WAV channel |
| **EMA** | 3D positions × N sensors (5–20) | 200–400 Hz | Heavy (Carstens AG501 dominant; NDI Wave) | Sub-ms; vendor format carries it |
| **EPG (palatography)** | ~62–96 binary contact sensors | 100–200 Hz | Heavy (Reading EPG, WinEPG) | Ms-level |
| **Ultrasound** | Video frames (60–200 fps), often with audio | 60–200 fps | Heavy (AAA / Articulate Instruments dominant) | Sub-frame |
| **rtMRI** | Video, very high data rate | ~83 fps | Research-only (USC SPAN ecosystem) | Sub-frame |
| **Aerodynamic** | Multi-channel sensor (oral/nasal flow, pressure) | 1–10 kHz | Heavy (PERCI-SARS, GlottalEnterprises) | Ms-level |
| **High-speed laryngeal video** | Video, very high frame rate (~4000 fps) | 4000 fps | Heavy (Wolf, Pentax) | Sub-frame |

Two structural facts drive the architecture:

- **Hardware-side is fragmented and vendor-locked.** We will not drive the hardware. We import data after recording.
- **Format fragmentation is worse than hardware.** Multiple Carstens formats across model generations; AAA proprietary; EPG vendor-specific; ultrasound sometimes DICOM, sometimes proprietary; rtMRI is research-pipeline-specific.

Current state of the art: **no integrated tool exists.** AAA + MView (UCLA MATLAB) + bespoke MATLAB pipelines is the academic stack. EMU-SDMS gets closest (SSFF multichannel + hierarchical levels) but its articulatory story is thin and R-centric. **This is a genuine differentiator for the phonetician audience if we build it well.**

### Architectural answer: data-model native, vendor glue in plugins

The decision is two-layered, not a single signals-vs-plugins fork:

1. **The data model treats articulatory channels as first-class signals.** They're time-aligned (uni- or multi-) dimensional data anchored to a `Bundle`. The existing corpus tier model already covers sensor modalities via `continuous_numeric` (e.g., nasal airflow), `continuous_vector` (EMA, EPG), and `categorical_sampled` (EPG contacts as binary). Video is the only modality the existing tier model doesn't cover — needs a new tier type.
2. **Vendor-format glue lives in plugins.** Each importer (Carstens `.pos` → vector tier; AAA `.ust` → video + contour tier; EPG vendor format → categorical-sampled tier) is a plugin, not core. The engine ships a plugin API and a small handful of in-tree importers; the long tail lives out-of-tree.

This honors the 2026-05-16 architectural implication #3 ("signal types are pluggable") without bloating the engine for every vendor format.

### Changes to the corpus data model

**New tier type: `video_aligned`.** Storage: original video file in `signals/original/session_X/bundle_Y.US.mp4` (or `.dicom`, `.avi`); the DB tier row carries the file reference + per-frame timing offset + frame-rate / codec metadata. Multiple `video_aligned` tiers per bundle allowed (e.g., ultrasound + lip-camera). Adds a seventh tier type to the six defined in the corpus entry.

**Channel semantics in `continuous_vector` `schema`.** A 21-channel EMA tier needs to declare what each channel means (`TT_x`, `TT_y`, `TT_z`, `TD_x`, …). Adding `channel_names` and `channel_kinds` fields to the `schema` JSON covers EMA, EPG, and aerodynamic without further tier types.

**Sub-millisecond sync via `time_offset_seconds` in tier `schema`.** Articulatory tiers need precise alignment with audio; the offset comes from the vendor format at import. Small addition, easy to overlook.

**Derived per-frame measurements stay standard tiers.** Tongue contour polylines, region masks, tracked landmarks — all `continuous_vector` tiers anchored to the same frame timeline as the parent `video_aligned` tier. Parent-child cardinality (already in the corpus model) handles the relationship.

### Plugin architecture — both Python and native, in v1

Two plugin classes, both available from launch:

**Python plugins.** Loaded via the embedded CPython committed in the tech-stack entry. Discovery: scan plugin directories for packages with a plugin manifest. Use case: vendor-format importers (where work is parsing weird binary formats, not crunching numbers). Easier authoring; aligned with the Python audience we're already serving.

**Native plugins (Rust dylib).** Loaded via dlopen / LoadLibrary at startup. Need a stable ABI — likely via the `abi_stable` crate or carefully-curated C-ABI surface. Use case: performance-critical analyzers (per-frame video analysis, real-time contour inference, anything that needs to run in the audio thread). Higher development cost; higher capability ceiling.

Plugin discovery locations:
- Project-scoped: `<project>/plugins/`
- User-scoped: `~/.config/sat/plugins/`
- Distribution: Python plugins pip-installable into embedded Python OR drop-folder; native plugins ship pre-built per platform.

Plugin trait sketch (conceptual; native and Python both implement equivalent contracts):

```rust
trait Importer {
    fn name() -> &'static str;                  // "carstens-ag501"
    fn supported_extensions() -> &[&str];       // ["pos", "amps"]
    fn detect(file: &Path) -> Confidence;
    fn import(file: &Path, bundle: &mut Bundle) -> Result<()>;
}

trait Analyzer {
    fn name() -> &'static str;                  // "tongue-contour-tracker"
    fn inputs() -> &[InputSpec];                // requires video_aligned tier
    fn outputs() -> &[OutputSpec];              // produces continuous_vector tier
    fn run(bundle: &Bundle, params: &Params) -> Result<()>;
}
```

Plugins declare capabilities in a manifest; the engine indexes them at startup. No sandboxing in v1 (trust model: user installs what they trust); plugin signing / sandboxing is a follow-up if it becomes necessary.

### v1 scope — maximal

| Modality | v1 status | Notes |
|---|---|---|
| **EGG** | **In v1, core.** | 2nd-WAV-channel import; basic analysis (open/closing instants, OQ, CQ from differentiated EGG). High clinical value (group 3); trivial engineering |
| **EMA — Carstens AG501** | **In v1, importer plugin in-tree (Python).** | Parses `.pos` to `continuous_vector` tier with channel semantics. Trajectory rendering in GUI |
| **Ultrasound — generic video** | **In v1, core.** | `video_aligned` tier type; video player widget in GUI; ffmpeg-rs or rsmpeg decoder |
| **Tongue contour tracking** | **In v1, core analyzer (native plugin).** | Bundled (or download-on-first-use) ML segmentation model; per-frame inference; interactive correction UI; contours stored as `continuous_vector` tier under the parent `video_aligned` tier. **This is the most ambitious single v1 commitment scoped so far** — see open trade-offs |
| **Ultrasound — AAA format** | v1.x via plugin | Format reverse-engineering; community plugin candidate |
| **EPG** | v1.x via plugin | Data model handles; waiting on importer |
| **rtMRI** | Defer | USC SPAN ecosystem handles; small overlap with primary audiences |
| **Aerodynamic** | v1.x via plugin | Data model handles trivially via `continuous_numeric`; needs vendor-specific importers |
| **High-speed laryngeal video** | Defer | Clinical/research niche; very large data |

### GUI implications

- **Video player widget synced with audio + spectrogram + tier overlays.** wgpu renders textures from decoded video frames; ffmpeg-rs / rsmpeg is the decoder. Cursor / playhead synchronization across waveform / spectrogram / video / EMA trajectory plot is the central UX claim.
- **Multi-pane synchronized layout.** Audio waveform + spectrogram + EMA trajectory + ultrasound video + tier strip, all scrolling together. egui's painter API handles this; just confirming articulatory data doesn't break the assumption.
- **EMA-specific rendering.** Mid-sagittal 2D projection (X-Y / X-Z), 3D rendering for AG501. Plottable as derived tiers in the same display surface.
- **Contour correction UI.** Per-frame contour editing with ML-model re-fit, propagation of corrections across nearby frames, comparison of model vs corrected. Its own UX subproblem.

### v1 deliverables this entry commits to

1. Seventh tier type `video_aligned` added to the corpus model.
2. Channel semantics (`channel_names`, `channel_kinds`) and `time_offset_seconds` fields added to dense-tier `schema`.
3. Plugin architecture: Python and native, in-tree and out-of-tree, with discovery in project + user dirs.
4. EGG support in core.
5. Carstens AG501 EMA importer (in-tree Python plugin).
6. Ultrasound video display (core): video player widget, sync with audio + spectrogram + tier strip.
7. Tongue contour tracker (core native analyzer): bundled/downloadable ML model, per-frame inference, interactive correction UI, results stored as a tier.

### Open trade-offs / deferred

- **Tongue contour tracker scope within v1.** This is the heaviest single commitment we've made. Realistically it's its own several-month workstream: model choice (UNet variant? DeepLabCut-style? fine-tune an existing model?), training / validation strategy, per-frame inference perf tuning, interactive correction UX, evaluation against existing tools (AAA, GetContours, Ultratrace). Worth phasing within v1: get bundled model + per-frame inference + visualization shipping first, build correction UI iteratively after. Could merit its own follow-up DEVLOG entry once a model is chosen.
- **Bundled model licensing and weights.** Per the licensing entry's discipline, model weights are downloaded not bundled. Need to pick a model with a permissive license (or train our own and publish it). Open-licensed pretrained tongue-segmentation models exist (various academic releases on HuggingFace and OSF) but quality varies.
- **Native plugin ABI stability.** `abi_stable` crate vs hand-curated C ABI vs WASM components. Each has trade-offs; needs a focused decision before v1 plugin API is published, since once it ships it's hard to change.
- **Video decoder dependency.** ffmpeg via `ffmpeg-rs` / `rsmpeg` is the standard answer but adds a heavy native dep with its own licensing constraints (LGPL builds OK for any license; GPL components must be excluded from builds). Worth a focused look.
- **Sub-frame sync semantics.** `time_offset_seconds` covers per-tier offset, but ultrasound + audio sometimes has drift across a long recording (clock-domain difference between video and audio). v1 assumes constant offset; clock-drift correction deferred.
- **3D EMA visualization.** AG501 is 3D; AG500 and older were 2D. v1 starts with 2D mid-sagittal rendering and full 3D becomes a follow-up.
- ~~**Bundled model distribution channel.**~~ Closed 2026-05-20: parallel registry to refdist (shared protocol, separate index), HF passthrough as escape hatch. See 2026-05-20 ML-model-registry entry.

### Sources / references

- EMU-SDMS articulatory support: https://ips-lmu.github.io/EMU.html
- AAA (Articulate Assistant Advanced): https://www.articulateinstruments.com/aaa
- MView (UCLA): https://www.phonetics.ucla.edu/facilities/software/mview.html
- Carstens AG501: https://www.articulograph.de
- NDI Wave: https://www.ndigital.com/products/wave
- USC SPAN rtMRI corpus: https://sail.usc.edu/span
- Ultratrace: https://github.com/Ultratrace/ultratrace
- GetContours: https://github.com/mlml/getContours
- ffmpeg-rs: https://github.com/zmwangx/rust-ffmpeg ; rsmpeg: https://github.com/larksuite/rsmpeg
- abi_stable crate: https://github.com/rodrimati1992/abi_stable_crates

---

## 2026-05-18 — Reference distribution governance: three-tier registry, unified format

Goal: settle the governance, format, and architecture for reference distributions — the open question carried from 2026-05-16 (cross-cutting pattern C) and flagged again in the corpus entry as a deferred follow-up. Reference distributions are the transverse value-add across at least five user groups (voice coaches, L2 learners, clinicians, forensic, phoneticians), and nobody currently has them in one place.

### What a reference distribution IS, in our model

A structured statistical summary of some acoustic / articulatory measurement over a population (or a prescriptive target), in a form the GUI can render against:

- Vowel formant clouds (Hillenbrand 1995, Peterson-Barney 1952, Hagiwara, …)
- Age / sex-normed clinical ranges (jitter, shimmer, HNR, CPP, …)
- F0 statistics by age / sex / dialect (forensic + voice-coaching)
- Speaker-set distributions (forensic likelihood ratios)
- Target zones (voice-coach / L2 — prescriptive, not observed)
- Articulation-rate norms, VOT distributions, prosodic patterns by language

The unifying abstraction: **a tagged sample or summary you can compare a measurement against**.

### Governance decomposes into seven sub-questions

| Sub-question | What it means |
|---|---|
| Curation | Who decides what's authoritative vs experimental |
| Versioning | How users pin for reproducibility (esp. forensic / clinical) |
| Licensing | What licenses are acceptable; can derived distributions ship under different terms than their source corpora |
| Privacy / k-anonymity | When does a distribution leak speaker identity |
| Discovery | How a user finds "the right reference for adult AmE female /i/" |
| Citation | Automating BibTeX / DOI so use in papers is one-click |
| Disputes / errors | Yanking, conflicting evidence, quality flags |

Plus the cross-cutting question of **transport** (how distributions get from publisher to user).

### Prior art surveyed

| Source | Lifted from | Left behind |
|---|---|---|
| **CRAN** | Strict editorial review, immutable versions, citeable | Slow contribution rate; heavy curation labor |
| **HuggingFace Hub** | Anyone publishes; reputation-based discovery; rich metadata cards; tag faceting | No quality gating; trust signals coarse |
| **Zenodo / OSF** | DOIs, persistent versioning, free researcher hosting | Not domain-aware; no schema for our data shape |
| **MFA models registry** | Domain-specific catalog, version-tagged, language-faceted — closest precedent in speech | Narrow scope (just alignment models) |
| **NIST reference data** | Authoritative, slow-moving, foundational tier | Government-scale infrastructure we don't have |
| **NHANES** | Clinical reference data convention; population descriptors | US-only, not directly extensible |
| **OLAC / PARADISEC / ELAR** | Metadata standards (Dublin Core, IMDI, CMDI) for language data | Heavy bureaucratic submission |

The pattern that fits us best is a **CRAN-curated tier on top of a HuggingFace-style open tier**, with MFA's models registry as the closest domain-specific precedent.

### Three-tier model (all three shipping at launch)

**Tier 1 — Bundled starter set.** Ships with the app. Small, vetted, foundational. Classic public-data distributions: Hillenbrand 1995 vowel space, Peterson-Barney 1952, a small set of clinically published normative ranges. Citation metadata baked in. Only redistributable-licensed data here. Updated with app releases.

**Tier 2 — Curated registry.** Editorial standards (CRAN-style): must have provenance, license, min-n, reproducible derivation method. Versioned semantically. Hosted on GitHub Pages. Submissions are PRs to a public repo; CI runs validation; merge publishes. Editorial committee starts as project maintainers, expanded by adding repo maintainers over time.

**Tier 3 — Community registry.** Same infrastructure, lower bar. Anyone can publish. Trust signals (download count, tier 2 promotion, flag-as-questionable) drive discovery rather than gatekeeping. Tier 2 can promote a tier 3 entry after review.

All three tiers share **one format, one query API, one discovery surface**. Users see them together with provenance labels in the GUI; only the trust signal differs.

### Format

A distribution is a directory or tarball:

```
refdist.toml         -- manifest: id, version, citation, population, measure, schema, license, kind
data.parquet         -- the data (samples OR summaries, per shareability declaration)
provenance.md        -- human-readable: paper, method, sample procedure
LICENSE              -- explicit license file
```

The `refdist.toml` schema is load-bearing — discovery, validation, citation, and rendering all key off it:

```toml
id = "hillenbrand-1995-amE-vowels"
version = "1.0.0"
title = "American English vowel formants (Hillenbrand et al. 1995)"
doi = "10.1121/1.411872"

[citation]
authors = ["Hillenbrand, J.", "Getty, L.", "Clark, M.", "Wheeler, K."]
year = 1995
journal = "JASA"
bibtex = "..."

[population]
language = "eng"
variety = "AmE"
sex = ["m", "f", "c"]
age_band = ["adult", "child"]
n_speakers = 139
n_tokens = 1668

[measure]
kind = "observed_distribution"   # or "target_zone" | "summary_normative_range"
parameters = ["F1", "F2", "F3"]
units = "Hz"
phones = ["iy", "ih", "ey", "eh", "ae", "ah", "aw", "ao", "ow", "uw", "uh", "er"]
context = "hVd"
measurement_method = "steady-state, manually selected"

[privacy]
shareability = "raw_samples"     # or "summary_only"
min_n_per_subgroup = 5
community_consent = false        # required true for small-language community data

[schema]
data_file = "data.parquet"
shape = "long"
columns = ["speaker_id", "phone", "F1", "F2", "F3"]
```

The `measure.kind` enum keeps prescriptive targets clearly distinct from observed populations while sharing infrastructure:

- `observed_distribution` — raw samples from a measured population
- `summary_normative_range` — summary stats only (mean / SD / percentiles), no raw values shipped
- `target_zone` — prescriptive goal regions (voice-coach / L2 use); not an empirical claim about a population

GUI renders them differently and labels them clearly so "what people sound like" is never conflated with "what they should aim for."

### Decisions per sub-question

| Sub-question | Decision |
|---|---|
| **Curation** | Three-tier model. Tier 2 editorial committee starts with project maintainers; expansion by adding maintainers to the registry repo |
| **Versioning** | Semantic. Immutable once published. Yanks possible (deprecation flag in index) but old pins keep resolving with a warning |
| **Licensing** | Prefer CC0 or CC-BY-4.0 for data; ODC-BY for databases. **Disallow non-commercial / no-derivatives in tier 2.** Allow in tier 3 with prominent flags. Matches our own permissive lean (cross-references the 2026-05-18 licensing entry) |
| **Privacy / k-anonymity** | Required `min_n_per_subgroup` in manifest (default 5). `shareability` field declares whether raw samples or only summaries are shipped. Tier 2 enforces; tier 3 surfaces. `community_consent` required for small-language community data |
| **Discovery** | Faceted search on manifest tags (language, variety, sex, age, measure, phones, kind); free-text over titles + provenance |
| **Citation** | Manifest carries BibTeX + (optional) DOI. Engine API records which distributions any analysis used. Project export-to-paper produces a citation list automatically |
| **Disputes / errors** | Conflicting distributions co-exist in tier 3. Tier 2 picks at most one per (measure × population × method). GitHub-issue-style error reports against published distributions. Corrections ship as version bumps |
| **Transport** | Plain HTTP-served static registry index + tarball downloads. GitHub Pages-hosted. Cached at user level (`~/.local/share/sat/refdist/`). App ships with official registry URL configured; accepts additional registries |

### Hosting: GitHub Pages, PR-based

Concretely:

- A public GitHub repo (e.g. `speech-analysis-tool/refdist-registry`) holds all tier 2 + tier 3 entries.
- Tier 1 (starter set) lives in the main app repo and is bundled at build time.
- Submission = PR adding a distribution directory under `tier2/` or `tier3/`.
- CI validates: TOML schema, manifest required fields, license file present and acceptable, `min_n_per_subgroup` enforced, license compatible with tier, data-file matches declared schema.
- Merge to main triggers Pages rebuild. The app fetches the rendered index JSON and individual tarballs.
- Tier-2 promotion = a second PR moving a directory from `tier3/` to `tier2/` after editorial review.
- Yanking = `yanked = true` flag in the entry; old pinned versions keep resolving with a warning surfaced in the GUI.
- DOI assignment is optional and separate; Zenodo can mint DOIs for GitHub releases if a publisher wants one.

### Architectural touch points

How this plugs into what we've already designed:

- **Distributions live outside the corpus**, at user level. Projects *reference* distributions by `id` + `version`; the engine resolves to the local cache (or fetches once if missing).
- **Projects pin versions for reproducibility.** `project.toml` records which distribution versions were used; opening on another machine fetches the same pinned versions.
- **Engine API exposes distributions as queryable objects** — e.g. `engine.refdist.query(measure="F1", population={lang:"eng", sex:"f", age_band:"adult"})` returns matching distributions; GUI renders as overlays / target zones / percentile bands.
- **Tier 3 publishing is in-app.** Users can package any analysis result as a distribution from inside the desktop GUI; the manifest is scaffolded automatically. Publication is a `git push` to a fork-and-PR flow (auth is the user's GitHub credentials, not ours).
- **Cited automatically.** When a project uses a distribution in any analysis, the citation is recorded; project export emits the citation list.

### Edge-case tests the model handles

- **Clinical norms from private patient data.** Hospital with n=200 dysphonic patients publishes jitter norms as `shareability = "summary_only"`. Only mean / SD / percentile bands ship in `data.parquet`. Provenance notes the source corpus is private. ✓
- **Voice-coach target zones (heuristic, not population-derived).** Published with `measure.kind = "target_zone"`. GUI labels distinctly from observed distributions. Important not to conflate "what is" with "what should be." ✓
- **Conflicting vowel-formant studies for the same population.** Both live in tier 3; users compare via quality signals (n, year, method, citations). Tier 2 picks at most one per (measure × population × method). ✓
- **Field-linguistics small-language data.** `community_consent = true` required for tier 2; flagged but not enforced for tier 3. CARE / FAIR principles respected by data publishers, surfaced by the platform. ✓
- **Forensic expert using reference from a private corpus.** Published as `summary_only` with provenance pointing to the private corpus (citable but not downloadable). Reproducible at the summary level; underlying data is not. ✓

### v1 deliverables this entry commits to

1. Bundled starter set with the app (Hillenbrand 1995, Peterson-Barney 1952, a small clinical normative-range set).
2. Public registry repo on GitHub with tier 2 / tier 3 directory structure.
3. CI validation pipeline (manifest schema, license check, min-n enforcement, data-file conformance).
4. GitHub Pages-rendered index JSON consumable by the engine.
5. Engine API for distribution resolution, caching, and query.
6. Project pinning of distribution versions in `project.toml`.
7. In-app publishing flow with auto-scaffolded manifests and fork-and-PR submission.
8. GUI rendering for overlays / target zones / percentile bands distinguishing `measure.kind` variants.

### Open trade-offs / deferred

- **Exact starter-set contents.** Need to confirm redistribution rights for each candidate dataset; start with publicly-licensed material only.
- **Initial editorial committee composition.** Starts as just the project maintainers; expansion process and quality bar can codify after launch and a few PRs.
- **Trust-signal richness in tier 3.** v1: download count + flag-as-questionable + tier 2 promotion. Richer signals (verified-publisher badges, citation counts) layered later.
- **Registry repo sharding.** Single repo to start; shard by domain (clinical, phonetic, forensic, …) if PR rate justifies.
- **Federated / additional registries.** App accepts additional registry URLs; whether to actively encourage forks (e.g. an organization's internal registry) is a community-development question, not a v1 architecture one.
- **DOI integration depth.** Zenodo can mint DOIs for tier 2 releases automatically; whether to require this for tier 2 or keep it optional needs a call.
- ~~**Profile-driven defaults.**~~ Closed 2026-05-20: each of the five v1 profiles ships with a "recommended distributions" list consumed by the refdist picker GUI. Specific distribution IDs still need sync with the actual tier-1 starter set as Phase 1 work lands. See 2026-05-20 profile-catalog entry.

### Sources / references

- CRAN policies: https://cran.r-project.org/web/packages/policies.html
- HuggingFace Hub: https://huggingface.co/docs/hub
- Zenodo: https://zenodo.org ; OSF: https://osf.io
- MFA models registry: https://mfa-models.readthedocs.io
- NHANES: https://www.cdc.gov/nchs/nhanes
- OLAC: http://www.language-archives.org ; PARADISEC: https://www.paradisec.org.au ; ELAR: https://www.elararchive.org
- CARE Principles for Indigenous Data Governance: https://www.gida-global.org/care
- FAIR Principles: https://www.go-fair.org/fair-principles
- Hillenbrand et al. 1995, JASA: https://doi.org/10.1121/1.411872
- Peterson & Barney 1952, JASA: https://doi.org/10.1121/1.1906875

---

## 2026-05-18 — Corpus data model: directory + SQLite + Parquet, six-type tier model

Goal: design the corpus data model — the structural decision flagged 2026-05-16 as "probably the single most consequential architectural decision." Cover the container shape, storage technology, annotation tier model, entity tables, round-trip interop, mobile sync, and schema migrations.

### What the corpus model has to carry

Pulling obligations from prior decisions and user groups:

- **8 user groups, 8 shapes of "project."** Forensic case files, clinical patient longitudinal data, field-linguistics sessions, ML datasets, experimental trial logs, voice-coach practice journals — one model has to fit all without collapsing into mush.
- **Three signal categories.** Original audio (WAV / FLAC); derived DSP signals (F0, formants, spectrograms — recomputable but expensive); ML embeddings (wav2vec2-class signals at 768-dim per 20 ms ≈ 38 MB/min — *must* persist).
- **Annotations are the secret center of gravity** (cross-cutting F): TextGrid, EAF, lexicon entries, trial logs, patient charts are all the same shape underneath.
- **Live recording must land cleanly** without corrupting the project on crash.
- **Mobile companion appends via sync** (no concurrent edits).
- **Forensic / clinical need audit trails** — every analysis step recorded.
- **Scale spans 4 orders of magnitude** — phonetician with 50 recordings ↔ ASR dataset with 1M utterances, same model.

### Prior art surveyed

| Tool | Lifted from | Left behind |
|---|---|---|
| **EMU-SDMS (emuDB)** | Hierarchical annotation levels with parent-child relations; bundle = (recording + annotations) as a unit; DBconfig as schema document | R-centric query layer; rigid migration story |
| **ELAN (.eaf)** | Tier types + controlled vocabularies + reference relationships between tiers | XML; no corpus structure beyond a folder of EAFs; no derived signals |
| **Praat TextGrid** | Flat interval/point tiers as the lowest-common-denominator interchange format | File-as-unit; no metadata, schema, or hierarchy |
| **Phon** | SQLite-as-project-file precedent for transcription corpora | Domain-narrow; not signal-rich |
| **NWB / HDF5 (neuroscience)** | Hierarchical container for huge time-series signals; self-describing; language-agnostic | Weak annotation story; not graph-queryable |
| **Apache Arrow / Parquet** | Columnar storage for derived signals and ML embeddings; zero-copy across Rust / Python; ecosystem-native for AI engineers | Not for audio blobs; not for annotation graphs |
| **SQLite (Phon, Anki, Lightroom, Logic Pro precedent)** | Single-file ACID DB as project container; queryable; trivially backed up | Not great for huge audio BLOBs |
| **Lightroom catalog model** | Directory containing a DB + signal files + sidecars; DB holds metadata, signals on disk | — |

EMU is the closest reference; the others contribute pieces.

### Five design axes and decisions

| Axis | Options considered | Decision |
|---|---|---|
| **Container shape** | Single file (SQLite-only); directory with embedded DB; graph-DB server | **Directory with embedded DB**. Plays well with `cp` / `rsync` / Time Machine; audio doesn't bloat the DB; standard scientific-app pattern |
| **Storage tech** | SQLite vs DuckDB for relational; Parquet vs HDF5 for dense signals; WAV vs FLAC for audio | **SQLite + Parquet + WAV (FLAC archival opt-in)**. SQLite for transactional editing maturity; Parquet for zero-copy Rust↔Python and AI-ecosystem fit; WAV for lossless universal originals |
| **Annotation model** | Praat-flat-tiers; ELAN tier-types; EMU hierarchical levels; unified type system | **Six-type tier model with parent-child cardinality** (see below). Unifies Praat / ELAN / EMU / ML signals / trial metadata |
| **Entity model** | Fixed schema + JSON extra; user-defined-schemas first-class; per-profile fixed schemas | **Fixed core schema + JSON `extra` column per entity**. Profiles ship as schema validators for `extra`. Fast queries, simple migrations, extensible per project |
| **Audit / versioning** | Audit log only; soft versioning (immutable annotations); Git-like full history | **Append-only audit log, no time-travel queries in v1**. Forensic-defensible, reproducibility-friendly, simple. Time-travel can layer in later |

### Annotation tier model

A `Tier` is a typed container with an optional parent relation:

```
Tier
  id           uuid
  bundle_id    uuid              -- which recording this tier belongs to
  name         text              -- e.g. "phones", "F0", "trials"
  type         enum              -- six variants, below
  parent_id    uuid?             -- null for top-level tiers
  cardinality  enum?             -- one_to_one | one_to_many | many_to_one | none
  schema       json              -- type-specific config (hop, dims, vocab)
  extra        json              -- user fields
```

Six tier types, with storage shape:

| Type | Storage | Praat / ELAN / EMU analogue | Primary uses |
|---|---|---|---|
| `interval` | rows in `annotation_interval` (DB) | Praat IntervalTier; ELAN ALIGNABLE_ANNOTATION | Phones, words, segments |
| `point` | rows in `annotation_point` (DB) | Praat TextTier; ELAN TIME_SUBDIVISION | Event markers, clicks, glottal pulses |
| `continuous_numeric` | Parquet sidecar (dense array); header in DB | EMU SSFF track | F0, intensity, jitter-over-window |
| `continuous_vector` | Parquet sidecar (n_frames × n_dims) | EMU SSFF multi-channel | wav2vec / HuBERT embeddings, MFCC, mel-spec |
| `categorical_sampled` | Parquet sidecar (RLE or dense) | — | VAD, voicing on/off, model class output |
| `reference` | rows in `annotation_reference` (DB) | ELAN SYMBOLIC_ASSOCIATION; FLEx lexical ref | Lexicon links, trial links, speaker turns |

**Storage split rationale.** Sparse tiers (interval, point, reference) live in the DB — queryable, editable, transactional. Dense tiers (numeric, vector, categorical sampled) live in Parquet sidecars next to the audio — they're huge (an hour of wav2vec layer-N embeddings is ≈ 2 GB), append-friendly, and the AI-engineer audience can mmap them directly without going through SQLite.

**Parent-child cardinality** captures EMU's hierarchical model: `one_to_one`, `one_to_many` (the common case: word → phones), `many_to_one` (the inverse view), or `none` (independent tier).

### v1 entity tables

```
Project           -- one row; project-level metadata, schema version, profile
Speaker           -- person who produced speech (could be participant/patient/case)
Session           -- recording session: time, place, instrument, calibration
Bundle            -- one recording: audio file + everything derived from it
Tier              -- annotation container; lives in a bundle
Annotation*       -- per-type tables: interval, point, reference
DerivedSignal     -- registration row for a Parquet sidecar
Entity            -- generic typed entity (Patient, Case, Trial, Stimulus, …)
EntityRef         -- links from reference tiers / sessions / bundles to entities
Protocol          -- recording protocol / form definition (CAPE-V, …)
Instrument        -- mic / interface / calibration data
ModelRun          -- registry of which ML model produced which signal/annotation
AuditLog          -- append-only log of all mutations
```

A note on `Entity`: rather than separate `Patient`, `Case`, `Trial` tables, a single `Entity` table with a `kind` discriminator and a JSON `extra` payload keeps the schema small. Common queries hit indexed columns (`kind`, `name`, `project_id`); group-specific fields live in `extra`. **Profiles** (clinical, forensic, phonetician, etc.) ship as schema validators for `extra` — the user gets typed forms in the GUI even though storage is flexible.

Every entity table has an `extra: json` column. Every mutation gets a row in `AuditLog` with `(timestamp, user, table, row_id, op, before, after)`.

### Project directory layout

```
my_project/                          ← the "project" is a directory
├── project.toml                     ← human-readable header (name, schema version, profile, …)
├── corpus.db                        ← SQLite: schema, entities, annotations, audit log
├── signals/
│   ├── original/                    ← source recordings (WAV; FLAC archival)
│   │   └── session_001/
│   │       ├── bundle_001.wav
│   │       └── .in_progress/        ← active live recordings land here
│   └── derived/                     ← cached / persisted derived signals (Parquet)
│       └── session_001/
│           ├── bundle_001.f0.parquet
│           ├── bundle_001.formants.parquet
│           └── bundle_001.wav2vec.layer8.parquet
├── attachments/                     ← user docs, images, calibration files
├── exports/                         ← round-trip TextGrid / EAF for interop
└── .lock                            ← single-writer file lock
```

### Live recording integration

Audio writes to a temp file under `signals/original/.in_progress/`. On recording end, a single SQLite transaction atomically moves the file to its final path and inserts the `Bundle` + `DerivedSignal` rows. A crash leaves a recoverable temp file; the project DB stays consistent.

### Round-trip interop

Adoption hinges on TextGrid round-tripping cleanly enough to satisfy existing Praat workflows. Tier-type coverage by format:

| Our tier type | Praat TextGrid | ELAN .eaf | EMU emuDB |
|---|---|---|---|
| `interval` | Direct (IntervalTier); attrs lost unless JSON-in-label | Direct (ALIGNABLE_ANNOTATION); limited attrs | Direct (interval level with attrs) |
| `point` | Direct (TextTier); attrs lost | Direct (TIME_SUBDIVISION) | Direct (point level with attrs) |
| `continuous_numeric` | Separate Pitch/Matrix file | Linked external file | Direct (SSFF track) |
| `continuous_vector` | Not representable | Not representable | Direct (SSFF multi-channel) |
| `categorical_sampled` | Lossy: collapse runs into IntervalTier | Lossy: collapse into time-aligned | Direct |
| `reference` | Lossy: flatten ref to label text | Direct (SYMBOLIC_ASSOCIATION) | Direct (reference relation) |
| Parent-child cardinality | Lost | Direct (parent_tier_ref) | Direct (level hierarchy) |

**Export targets, ranked by adoption value:**

1. **TextGrid (Praat) — table stakes, deliberately lossy.** Interval and point tiers. Optional attribute round-tripping via a JSON sentinel inside the label (`{json:{…}}`) that Praat ignores and we re-parse. Document loudly what's lost.
2. **EAF (ELAN) — primary annotation interop.** Hierarchical, lossless for the subset ELAN supports.
3. **emuDB — most faithful export.** Maps nearly everything we have.
4. **Parquet / Arrow — for dense signals only.** Direct export of `continuous_*` tiers; the format the AI-engineer audience already lives in.
5. **Our own project archive (`.satproj.tar` or similar) — fully lossless.** Tarball of the project directory; the format you use to share a project between instances of the tool.

**Critical boundary: export is a snapshot, not a sync.** A user who exports a TextGrid, edits it in Praat, and re-imports gets a *new annotation set* on the bundle — not a merge. Maintaining a two-way bridge to a richer model has no good failure mode. The exception is our own archive format, which round-trips losslessly because it IS our format.

### Mobile sync — bundle pack format

Mobile is recording-only; sync is one-directional and append-only. The unit of sync is one completed bundle:

```
bundle_001.satbundle.tar
├── manifest.toml       -- bundle id, session id, recording params, instrument,
│                          calibration, speaker ref, timestamps
├── audio.wav           -- the recording
├── tiers/              -- annotations created on phone (live VAD, user-marked points)
│   └── vad.json
└── derived/            -- signals computed on phone (live pitch track, etc.)
    └── pitch.parquet
```

**Ingest flow.** Desktop project watches one or more sync inbox locations (local folder, cloud folder, LAN, …). For each bundle pack: validate manifest → copy audio → copy derived → insert DB rows → mark consumed. Transactional; failures leave the pack in the inbox with an error flag.

**Transport is unspecified.** AirDrop, USB, Syncthing, Dropbox, iCloud, LAN — all valid; all just shuffle a portable archive. Local-first (decision #10) means no required server.

**Session assignment.** Phone either declares "this bundle belongs to session X" (created on desktop, pushed to phone) or drops into a "mobile-default" session the user reorganizes later.

### Schema migrations

- **Tool: `refinery` or `sqlx::migrate!`.** Numbered SQL files versioned in the engine crate. DB stores its schema version in a `schema_migrations` table.
- **Engine refuses to open a DB newer than it knows** (forward-compat clamp).
- **Older-schema DBs**: additive changes (new columns / tables) auto-apply on open; destructive changes (renames / drops) prompt for explicit confirmation. Every upgrade writes `corpus.db.bak.<old_version>` first.
- **JSON `extra` columns don't need DDL migration**, but profile-level `extra` schemas evolve — ship profile migration utilities separately that walk rows and update payloads.
- **Forward compatibility within a minor version**: additive changes don't bump major. Engine SQL must avoid `SELECT *` so older code can ignore unknown columns.
- **Parquet sidecars carry their own embedded schema** — no migration needed. Changes to derivation produce new sidecars; old ones stay valid.

### v1 deliverables this entry commits to

1. Directory-shaped project with `corpus.db` + `signals/` + `attachments/` + `exports/`.
2. SQLite-backed entity schema as enumerated above, with JSON `extra` columns and append-only `AuditLog`.
3. Six-type tier model with parent-child cardinality, split storage (DB for sparse, Parquet for dense).
4. Live recording via `.in_progress/` + atomic commit.
5. TextGrid + EAF + emuDB + Parquet + project-archive export paths. Praat TextGrid round-trip is the only one with bidirectional re-import in v1.
6. Mobile bundle-pack format and append-only sync inbox.
7. Migration policy via `refinery` / `sqlx::migrate!`.

### Open trade-offs / deferred

- ~~**ML model registry scope**~~ Closed 2026-05-20: split into provenance layer (`ProcessingRun` table — replaces `ModelRun` here) and distribution layer (parallel registry to refdist, HF passthrough escape hatch). See 2026-05-20 ML-model-registry entry.
- **Reference distributions** — entangled with the 2026-05-16 governance question (who curates, versions, licenses). Its own follow-up entry.
- **Concurrency** — single-writer file lock for v1; multi-user is a v2+ concern requiring a coordination story we don't owe yet.
- ~~**Profile catalog**~~ Closed 2026-05-20: five profiles ship at v1 — phonetician (default), clinical, forensic, field, experimenter. Voice training deferred to v1.x with Phase 6 mobile. See 2026-05-20 profile-catalog entry.
- **TextGrid attribute-sentinel format** — the `{json:{…}}` convention needs a precise spec so it's robust to round-trips through other tools.
- **Cross-bundle queries** — DB layout supports them (one project = one DB), but the API surface (how a phonetician asks "all phones tagged `[ɛ]` across all bundles from male speakers") is a query-language decision deferred to the API-surface entry.
- **Hierarchical query semantics** — EMU has a full path-expression language (EQL). What subset do we ship in v1?

### Sources / references

- EMU-SDMS / emuDB: https://ips-lmu.github.io/EMU.html
- ELAN EAF format: https://www.mpi.nl/tools/elan
- Praat manual (TextGrid spec): https://www.fon.hum.uva.nl/praat/manual/TextGrid_file_formats.html
- Phon: https://www.phon.ca
- NWB (Neurodata Without Borders): https://www.nwb.org
- Apache Arrow: https://arrow.apache.org ; Parquet: https://parquet.apache.org
- SQLite as application file format: https://sqlite.org/appfileformat.html
- refinery: https://github.com/rust-db/refinery
- sqlx: https://github.com/launchbadge/sqlx

---

## 2026-05-18 — Licensing: permissive-leaning, GPL kept on the table

Goal: settle the license direction before the dependency tree is locked in, so we don't discover an incompatible dep later. Triggered by the realization that licensing affects which downstream user groups can actually productize on top of the engine.

### Audit of the stack picked in the previous entry

Almost the entire dep tree is permissive (MIT / Apache-2.0 / dual), which keeps every licensing option open for our own code. Two items worth flagging:

| Dep | License | Note |
|---|---|---|
| Rust toolchain, `rustfft`, `realfft`, `cpal`, `egui`, `wgpu`, `glow`, `ort`, `candle`, `burn`, `PyO3` | MIT and/or Apache-2.0 | Permissive across the board |
| CPython (when embedded) | PSF | BSD-compatible |
| **`UniFFI`** | **MPL-2.0** | File-level copyleft only — using as a dep does *not* infect our code; only modifications to UniFFI's own source files would have to be MPL |
| **`rust-jack` → libjack** | MIT crate; libjack is LGPL-2.1 | Dynamic linking (the cpal default) is fine for any downstream license. Avoid static linking libjack |

Two small disciplines fall out:
- Don't fork UniFFI itself (or accept MPL on the fork). Using it is fine for any license.
- Keep libjack dynamic-linked, which is the default. The PipeWire JACK-compat shim is MIT, which is even cleaner on PipeWire-default distros.

### Patent grants matter for the ML path

Apache-2.0 includes an explicit patent grant; MIT does not. For a project with substantial ML inference code (cross-cutting pattern G), the patent grant is non-trivial protection against future patent assertions. This is why the Rust convention is **dual Apache-2.0 OR MIT** — downstream users pick, but the patent grant is available. We should match that convention regardless of whether we add a stronger copyleft layer.

### The three license shapes considered

1. **Permissive (Apache-2.0 OR MIT).** InFormant, MFA, librosa, ESPnet path. Anyone can fork, integrate, build commercial products on top. Doesn't require improvements back. Maximizes adoption and downstream productization.
2. **Weak copyleft (MPL-2.0 or LGPL-3.0).** Engine improvements come back, but downstream apps that *use* the engine can be any license. MPL is more modern and avoids LGPL's static/dynamic-linking gymnastics. A proprietary clinical app could link our engine; patches to the engine itself would be MPL.
3. **Strong copyleft (GPL-3.0+).** Praat, Phonometrica, ELAN, Parselmouth path. Any derivative work must also be GPL. Strong free-software statement; closes the door to FDA-cleared clinical products, forensic court-deployable products, and most mobile/consumer apps built on top. Phonometrica is the cautionary tale: serious technical effort whose adoption partly stalled on license-driven integrator deterrence.

### How license interacts with the 2026-05-16 user groups

| Group | Sensitivity | Direction |
|---|---|---|
| 1. Phoneticians | Low | Neutral |
| 2. AI engineers | Low | Mild permissive lean (HuggingFace/PyTorch world is mostly permissive) |
| 3. Clinical (SLPs, ENT) | **High** | Strong permissive lean — FDA-cleared products on GPL code is effectively unheard of |
| 4. Voice coaches | Medium | Permissive lean — mobile/consumer apps won't adopt GPL |
| 5. L2 learners | Medium | Permissive lean — same as above |
| 6. Field linguists | Low–medium | Somewhat copyleft-sympathetic culturally, but practical access matters more |
| 7. Forensic | **High** | Strong permissive lean — proprietary court-deployable tools won't touch GPL |
| 8. Experimenters | Low | Mixed (PsychoPy is GPL; jsPsych/PCIbex are MIT) |

Aggregate: **permissive or MPL maximizes the user-group reach we explicitly scoped.** GPL would meaningfully reduce reach in the clinical and forensic directions.

### Provisional stance

**Lean: permissive — dual Apache-2.0 OR MIT** for the engine, the Python module, and the desktop GUI. Matches Rust-ecosystem and speech-ML-ecosystem convention, doesn't cut off clinical/forensic groups, gives downstream users patent protection by default.

**Not committed.** GPL-3.0+ is left on the table as a possible ambition: a deliberate choice to prioritize software freedom over reach. Revisit before the first public release. MPL-2.0 is the principled middle ground if we want engine improvements to come back without infecting downstream integrations.

### What follows regardless of which license we land on

These are non-negotiable independent of the license choice:

1. **Do not bundle model weights.** Ship code that downloads them; let users inherit their licenses (most popular speech model weights are MIT/Apache anyway, but the principle keeps us clean).
2. **Do not redistribute reference corpora that aren't openly licensed.** Reference Hillenbrand / Peterson-Barney etc. by pointing at them; bundle only corpora we have clear redistribution rights for.
3. **Do not wrap or link Praat itself.** Praat is GPL-3.0+; any linking inherits GPL (this is how Parselmouth became GPL). Re-implementing algorithms from the literature is fine; copying Praat source is not. The 2026-05-16 decision to make "running Praat scripts a non-goal" already aligns with this.
4. **Keep libjack dynamic-linked** on Linux (cpal default), and don't fork UniFFI without accepting MPL on the fork.
5. **Engine API stability matters more under permissive licensing** because commercial integrators will lock to versions and resist breaking changes. Already a discipline we owed Python users; permissive licensing just amplifies it.

### Open questions / deferred

- **Final license decision** — defer until closer to first public release; revisit if the user-group prioritization shifts (e.g. if we explicitly de-scope clinical/forensic productization, GPL becomes more palatable).
- **Contribution license.** Whether to require a CLA, use a DCO sign-off, or rely on inbound = outbound. Affects whether we can ever relicense.
- **Trademark.** Project name and any logo — separate from code license, often overlooked, matters for community-fork dynamics.

---

## 2026-05-18 — Tech stack: engine-first Rust + egui + PyO3 + UniFFI

Goal: pick the host language and surface technologies for v1, given the architectural implications surfaced 2026-05-16. The tech-stack discussion was explicitly deferred at the end of that session; this entry closes it.

### Constraints inherited

Six prior decisions narrow the field before any tech is named:

| Source | Constraint |
|---|---|
| Decision #6 | Desktop primary, phone as thin companion (not parity) |
| Decision #5 | Real-time analysis path designed in, not bolted on |
| Decision #8 | Scriptable via Python API; host language separate from API |
| Decision #3 / pattern G | ML inference (ONNX-family + others) as a first-class signal type |
| Decision #10 | Local-first, encryption at rest |
| Implicit | Solo-dev / small-team capacity; can't carry two full native UI codebases at desktop scope |

### Decision axes

Three axes determine almost everything; the rest follows.

- **A. Engine language.** Where DSP + real-time + ML inference live. Realistic candidates: Rust, C++. (Python-as-engine is too slow for real-time; JVM/Dart need a native layer underneath anyway, just deferring the question.)
- **B. Desktop GUI surface.** Native toolkit (Qt), immediate-mode (egui / iced), webview (Tauri / Electron), declarative cross-platform (Compose / Flutter / Slint).
- **C. Engine ↔ GUI ↔ mobile ↔ Python coupling.** Monolithic app that exposes APIs, or a portable engine library that multiple shells consume?

### Decisions, top-down

**Coupling — engine-first.** The engine is the deliverable: a Rust crate with a stable, versioned API. Desktop GUI links it in-process. Mobile shells consume it via UniFFI-generated Swift / Kotlin bindings. The Python module (PyO3) wraps the same engine for library users and for the in-app scripting host. This makes the engine ↔ Python boundary a first-class API surface, designed-for from day one rather than retrofitted.

**Engine language — Rust.** Memory safety in a long-running stateful tool with real-time threads is non-trivially valuable. Mature crates for everything we need: DSP (`rustfft`, `realfft`), audio I/O (`cpal`), GUI (`egui` + `wgpu`), ML inference (`ort`, `candle`), Python bindings (`PyO3`), mobile bindings (`UniFFI`). C++ + Qt is the conservative alternative — better-trodden in the domain (Praat, Phonometrica) — but pulls in CMake/footgun overhead, weaker Python embedding ergonomics, and more painful long-term maintenance for a small team.

**Desktop GUI — egui.** The central UI element is dense custom-rendered viz (spectrograms with tier overlays, formant tracks, real-time scrolling, scrub cursors). egui's painter API plus a `wgpu` backend gives Rust direct GPU access to draw this; no IPC, no serialization boundary, no webview, no Linux-distro variance. Closest reference: Rerun is architecturally the same shape (Rust engine + egui scientific viewer over huge multi-modal time-series data) and uses egui. Trade-off accepted: egui's default widgets look "tool-like" rather than native — fine for a research/clinical tool, less ideal if the app were positioned as a polished consumer voice-coach product. Slint is the runner-up if conventional UX matters more than I'm weighting it now. Tauri was ruled out because its webview boundary buys nothing for viz-heavy UI and risks WebKitGTK performance variance on Linux. Compose Multiplatform was ruled out once we decided mobile is companion, not parity — Compose's mobile-parity advantage doesn't apply.

**ML inference — `ort` primary, `candle` for mobile.** `ort` (ONNX Runtime Rust bindings) runs arbitrary ONNX graphs, covering the breadth of speech models published on HuggingFace: wav2vec2, HuBERT, WavLM, Whisper, ECAPA-TDNN / x-vector speaker embeddings, Silero VAD, etc. Breadth is essential for the "ML features as first-class signal" pattern (G) — users will want to drop in arbitrary fine-tuned models. `candle` (HuggingFace's pure-Rust framework) is narrower — only architectures that have been ported to Rust code — but pure-Rust means clean cross-compilation, especially for mobile, where ONNX Runtime's iOS/Android binaries are heavy. Mobile's ML surface is narrow (live VAD, maybe pitch enhancement), so candle is plausibly enough there. `burn` worth tracking as a third option but not v1.

**Audio I/O — `cpal` with JACK on Linux.** `cpal` is the standard Rust cross-platform audio I/O crate (CoreAudio / WASAPI / ALSA / JACK). On Linux, enable the `jack` feature and prefer the JACK host when available — works against PipeWire's JACK-compatibility API too, so modern distros are covered. ALSA-direct fallback for systems without JACK/PipeWire. Real-time audio thread needs `SCHED_FIFO` priority. If cpal's JACK backend disappoints in practice (historically less polished than its ALSA backend), local fallback is the `jack` crate directly on Linux. Audio I/O is wrapped behind an engine abstraction so the swap is contained.

**Mobile — native shells via UniFFI.** iOS: SwiftUI on top of UniFFI-generated bindings. Android: Jetpack Compose on the same. Scope kept strictly narrow: live recording, live feedback (pitch / formants), upload-and-sync to desktop project. No corpus management, no annotation editing, no scripting on phone. Writing both UIs natively is acceptable because the scope is bounded; we get native audio I/O (AVAudioEngine / AAudio) and native distribution as a side benefit.

**Python — PyO3, included in v1 (revised mid-session).** Initial framing deferred Python to v2 to simplify v1 scope. Reconsidered: scripting and GUI use are equally important user surfaces. The phonetician scripting tradition (group 1) and the AI-engineer / Parselmouth-replacement audience (group 2) are entirely gated on the Python API existing. The engine ↔ Python boundary needs to be designed for Python from day one — retrofitting in v2 would force a re-architecture once real scripting needs surface. Two delivery shapes share the same module: (a) standalone PyO3 module — `import speech_analysis_tool` from any CPython, library-shaped like Parselmouth; (b) embedded in the desktop app — same module imported by an embedded CPython, serving as the in-app scripting host. Costs accepted: GIL discipline around long-running engine ops, NumPy interop for signal arrays, embedded-CPython distribution complexity (well-trodden in DCC tools — Blender, Houdini — but non-trivial), API stability earlier than otherwise.

### Architecture

```
                  ┌────────────────────────────────────┐
                  │  Rust engine crate                 │
                  │  - DSP (F0, formants, spectro, …)  │
                  │  - ML inference (ort; candle on    │
                  │    mobile)                         │
                  │  - Corpus & annotation model       │
                  │  - Real-time audio loop (cpal)     │
                  │  - Reference-distribution store    │
                  └────────────┬───────────────────────┘
                               │ stable Rust API
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼─────┐         ┌──────▼──────┐         ┌─────▼──────┐
   │ Desktop  │         │ PyO3 module │         │ UniFFI     │
   │ egui GUI │         │ (Python)    │         │ bindings   │
   │          │         │             │         │            │
   │ in-proc  │         │ standalone  │         │ SwiftUI    │
   │ wgpu     │         │ library OR  │         │ (iOS)      │
   │ no IPC   │         │ embedded as │         │ Compose    │
   │          │         │ in-app host │         │ (Android)  │
   └──────────┘         └─────────────┘         └────────────┘
```

### v1 deliverables, after this entry

1. Rust engine crate with a stable, versioned public API.
2. Desktop GUI (egui) consuming the engine in-process.
3. PyO3 Python module exposing the same engine — both as a standalone library and as the embedded scripting host inside the desktop app.
4. iOS + Android companion shells via UniFFI, scoped to live record + live feedback + sync to a desktop project.

### Open trade-offs / deferred

- **egui ergonomics for forms / dialogs.** Most of our UI is custom viz, but settings panels, file pickers, batch-job config forms exist and will feel less polished than native. Acceptable if we keep that surface small; revisit if it grows.
- **Embedded CPython distribution.** Signing on macOS, library paths, GIL across the FFI boundary. Worth a focused spike before committing to "embedded scripting host in v1" specifically — the standalone PyO3 module is cheaper and could ship first, with the embedded host following.
- **cpal's JACK backend reliability.** Watch for xruns / sample-rate negotiation issues. Fallback: drop to the `jack` crate directly on Linux behind the audio-I/O abstraction.
- **wgpu vs glow as egui backend.** `wgpu` is the modern path (Vulkan / Metal / DX12 / WebGPU). `glow` is the legacy OpenGL backend, more compatible on older Linux GPUs. Start with `wgpu`; keep `glow` available as a fallback build for distribution edge cases.
- **Python API surface scoping.** What's in v1's public API matters more when Python is in v1 — anything we expose, users will write scripts against and resist us breaking. Worth a dedicated follow-up entry on API surface design.
- **In-app scripting host UX.** Notebook-style? Script-file editor? REPL panel? Praat-script-editor analogue? Affects how Python embedding actually surfaces in the GUI, and overlaps with the open questions still pending from 2026-05-16.

### Sources / references

- Rerun (reference architecture: Rust engine + egui scientific viewer): https://rerun.io ; https://github.com/rerun-io/rerun
- egui: https://github.com/emilk/egui
- Slint: https://slint.dev
- Tauri: https://tauri.app
- cpal: https://github.com/RustAudio/cpal
- rust-jack: https://github.com/RustAudio/rust-jack
- ort (ONNX Runtime Rust bindings): https://ort.pyke.io
- candle: https://github.com/huggingface/candle
- burn: https://github.com/tracel-ai/burn
- PyO3: https://pyo3.rs
- UniFFI: https://mozilla.github.io/uniffi-rs/
- wgpu: https://wgpu.rs

---

## 2026-05-16 — Target user groups: features, ideals, cross-cutting patterns

Goal: enumerate the user groups a Praat successor might serve, list what tooling they use today and what features they actually rely on, sketch what an ideal experience would look like, and surface patterns that should inform architecture.

Scope decisions taken this session (in response to the six open questions from 2026-05-15):

| # | Question | Decision |
|---|----------|----------|
| 1 | Target users | All-in-one ambition; enumerate to find overlaps (this entry). Phonetic research and speech-AI engineering are user-profile givens. |
| 2 | Scope | "All-in-one" as a north star, accepting it may end up "some-in-one." Let developability drive trimming. |
| 3 | Architecture stance | Desktop GUI primary. Mobile companion seriously considered if workflows fit. |
| 4 | Scripting | Definitely scripted. Form TBD. Python is the accessibility incumbent but may not be the right host-language for the app itself. |
| 5 | Praat compat | TextGrid I/O is table stakes. Running Praat scripts is a non-goal for now. |
| 6 | ML vs classical DSP | Both, first-class. |
| 7 (new) | Experimental interfaces | In scope. Production/perception experiment runner (Praat Demo Window / PsychoPy overlap), stimulus elicitation. |

### Groups

User profile context: the user is themselves a phonetician + speech-AI engineer, so groups 1 and 2 are taken as givens and described briefly. Groups 3–8 get fuller treatment.

#### 1. Research phoneticians  *(given — user profile)*

Praat + Parselmouth + MFA + R/Python stats, sometimes EMU-SDMS, ELAN for multi-tier or video. Plugins: FastTrack (formant exploration), ProsodyPro (F0). Articulatory work (ultrasound, EMA, palatography) lives in separate proprietary tools — AAA, MView. The frictions are well-known: arcane scripting, no undo, manual stitching of forced-alignment / annotation / stats stages, weak articulatory channel support, no native ML feature inspection.

#### 2. Speech AI / ML engineers  *(given — user profile)*

`librosa` / `torchaudio` / HuggingFace + bespoke matplotlib notebooks + W&B/MLflow + ad-hoc dataset browsers. Praat used when *acoustic measurement* is genuinely needed. The unmet need is a real GUI for inspecting model outputs at scale — attention/alignment for TTS, embeddings (UMAP) for speaker/dialect models, error analysis (WER by acoustic-context bucket), dataset cleaning, side-by-side audio diff. None of this lives in one tool.

#### 3. Clinical voice (SLPs, ENT, voice therapists)

**Today.** CSL / MDVP, Visi-Pitch, Sona-Speech, VoxMetria, Lingwaves — all proprietary. Some clinical Praat scripts (VoiceEvalU8, AVQI, ABI). Stroboscopy and EGG hardware lives outside the analysis app.

**Features relied on.** Standardized parameter sets with normative ranges (MDVP's 33, AVQI, ABI, CPP-based scores); CAPE-V / Rainbow Passage protocols; sustained-vowel + connected-speech workflows; phonetogram / Voice Range Profile; calibrated SPL (calibrated mic); EGG sync; report generation with EHR fields; pre/post-treatment longitudinal comparison.

**Pain points.** CSL is expensive, Windows-only, locked. Praat parameter values aren't directly interchangeable with MDVP. Longitudinal patient tracking is manual. Report generation is manual. Regulatory/validation friction makes switching tools risky.

**Ideal.** Open, documented MDVP-equivalent algorithms; AVQI/ABI validated against literature; calibrated-SPL support via known-mic profiles; patient-as-first-class-object with longitudinal viz; protocol-driven workflows (load protocol → guided recording → auto-report); HL7 / FHIR export; teletherapy via browser delivery.

#### 4. Voice coaches and trans-voice training

**Today.** Praat (the trans-voice community has popularized it heavily), InFormant (the strongest real-time tool), mobile apps (Voice Tools, Eva, ChristellaVoice, Vocaberry). Singing coaches additionally use VoceVista, Sygyt's Overtone Analyzer.

**Features relied on.** Real-time F0; real-time F1/F2 (resonance targets); user-defined target zones (not population norms); pitch range / register exploration; recording-and-compare against a model; visual feedback during practice; phonetogram; progress over weeks.

**Pain points.** Praat is intimidating and not designed for this audience. Mobile apps are paywalled and analytically shallow. Clinical norms don't fit (voice modification is not pathology). Singing-specific needs (vibrato, register transitions, vowel modification across registers) aren't standard in any one tool.

**Ideal.** Mobile-first real-time; user-definable target zones with own-voice baselines; practice sessions with auto-tagged recordings and longitudinal viz; singing modes (semitones/cents, vibrato analysis, formant tuning); privacy-respecting (local by default, no cloud uploads); onboarding that doesn't require linguistics training.

#### 5. L2 / pronunciation learners and tutors

**Today.** Praat (advanced learners and pronunciation researchers); commercial apps (ELSA Speak, Speechling, Sounds, Forvo). Most commercial offerings are ASR-feedback driven.

**Features relied on.** Compare learner to native model; minimal pairs; IPA display; stress/intonation viz; spaced repetition.

**Pain points.** ASR-based feedback says "wrong" without saying *why*. Native models are usually one speaker, not a distribution. No phonetic-feature-level feedback. Progress tracking is shallow.

**Ideal.** Feedback grounded in measured acoustics (your F2 trajectory vs target distribution), not just ASR pass/fail; native-speaker reference *distributions*; minimal pairs with measured targets; mobile delivery; pluggable for arbitrary L1→L2 pairs.

#### 6. Field linguists / language documenters

**Today.** ELAN (heavily) + Praat + SayMore + FieldWorks (FLEx) + Toolbox/Shoebox + Phonology Assistant + Audacity. Often offline, often on modest hardware in remote conditions.

**Features relied on.** Excellent IPA input (palette, feature search, novel diacritics); multi-tier annotation with glossing layers (audio → IPA → morphological gloss → free translation → metadata); lexicon-aware annotation; time-aligned video + audio + transcript; long-session recording reliability; archival-format export (DELAMAN, ELAR, PARADISEC).

**Pain points.** ELAN is powerful but UX-heavy. Praat is annotation-poor (one TextGrid, no lexicon link). Nothing spans recording → transcription → lexicon → publication cleanly. IPA keyboarding is universally bad. Field hardware constraints (battery, low power, sometimes offline for weeks).

**Ideal.** Lightweight, cross-platform, offline-capable. Excellent IPA input. Lexicon-aware annotation (this word's gloss persists across files). Tier-rich annotation matching ELAN's expressiveness. Crash-resilient recording with autosave. Direct export to archival formats. Phone as a recording companion that syncs back to the desktop project.

#### 7. Forensic phoneticians

**Today.** Praat + proprietary speaker-comparison tools (Batvox, VOCALISE / iVOCALISE) + bespoke likelihood-ratio scripts. Reference populations are scarce and often proprietary.

**Features relied on.** Speaker comparison with likelihood-ratio frameworks; long-term formant distributions; F0 statistics, articulation rate; defensible/auditable methodology; chain-of-custody; population reference data.

**Pain points.** Tools are niche and proprietary. Reproducibility / audit trail for court is hand-rolled. Reference population data is fragmented.

**Ideal.** Built-in audit logging (every analysis step recorded); standard LR frameworks; community-shared reference population corpora; reproducible "raw audio → court-ready report" pipelines.

#### 8. Experimental psycholinguists / lab experimenters

**Today.** PsychoPy (Python, lab), E-Prime (proprietary Windows), OpenSesame, PCIbex / Gorilla / jsPsych (browser), LabVanced. Praat's Demo Window for quick perception probes. Recording is one tool, analysis is another.

**Features relied on.** Millisecond-accurate stimulus timing; randomization / counterbalancing; audio playback + response (keyboard / button / mouse / voice); production tasks (prompt → record participant); trial-level data export; multi-modal stimuli; browser deployment for Prolific/MTurk; informed-consent flows.

**Pain points.** PsychoPy is code-heavy for non-programmers. Web platforms have audio timing issues. Recording in one tool, analyzing in another, with no shared project model. Browser recording quality varies wildly.

**Ideal.** Visual experiment builder (stimulus list → trial structure → response collection) that shares a project with the analysis layer; trial metadata flows directly into downstream analyses; auto-process recordings (force-align, extract features, tag with trial info); browser-deliverable version for crowdsourced participants; open / replicable experiment artifacts.

#### (Adjacent) Bioacoustics

Worth flagging because Parselmouth's bioacoustics paper showed real adoption crossover. Tools: Raven Pro (Cornell), Avisoft, BatSound. Needs: configurable frequency ranges (ultrasonic / infrasonic), long-recording event detection, multichannel arrays for localization, species-classifier plugins. Probably not a primary audience but should not be designed *out* — frequency-range assumptions in particular are easy to bake in by accident.

### Cross-cutting patterns

These are the patterns that recur across groups and should drive architecture, not just feature lists.

**A. Real-time analysis is universally undersupported.** Voice coaches, L2 learners, clinicians during therapy, and experimenters during pilot recording all want live feedback. Only InFormant takes this seriously. Praat's real-time facilities are afterthoughts. A first-class low-latency analysis path (live spectrogram, F0, formants, configurable feature overlays) would serve four groups at once.

**B. Mobile is a structural gap, not a nice-to-have.** Voice coaches, L2 learners, field linguists (recording companion), clinical teletherapy, crowdsourced experiment participants — all benefit from a phone form factor. Today's mobile speech apps are commercial walled gardens with weak underlying analysis. A research-grade engine deliverable to phone is genuinely novel and aligns with the desktop-primary / mobile-companion stance from question 3.

**C. Reference distributions are everyone's hidden problem.** Voice coaches want target zones; L2 learners want native-speaker distributions; clinicians want age/sex-normed ranges; forensic wants population statistics; phoneticians constantly cite Hillenbrand / Peterson-Barney / Hagiwara. Nobody has them in one place. A reference-distribution layer — bake in known corpora, make it easy to publish new ones, treat distributions as a first-class data type the GUI can render against — would be a transverse value-add across at least five groups.

**D. The "record → annotate → analyze" pipeline is fractured for every group.** Praat handles annotate + analyze but is weak at record; PsychoPy handles record + present but not analyze; ELAN annotates but doesn't analyze; field tools cover record + annotate but not analyze. Every group stitches. The opportunity is not "better analyzer" — it's *one continuous project model* that holds recordings, annotations, analyses, and derived data, with sensible defaults per workflow.

**E. The corpus / project model itself is missing in Praat.** EMU-SDMS is the only mainstream tool that takes this seriously. Forensic case files, clinical patient longitudinal data, field-linguistics sessions, ML datasets, experimental trial logs — every group wants "this is the project; here are its files and metadata." This is probably the single most consequential architectural decision: design corpus-first, not file-first.

**F. Annotation is the secret center of gravity.** TextGrid (Praat), EAF (ELAN), lexicon entries (FLEx), trial logs (PsychoPy), patient charts (clinical) are all annotation schemas under different names. A flexible tier + attribute model — interval/point/sequence/categorical/numeric/embedding-vector tiers — could unify all of them. EMU's hierarchical model is the closest prior art; ELAN's tier model is the most practically expressive.

**G. ML features are a unifying "new signal type."** Every group has a use:

- phoneticians → SSL embeddings as analogues to formants
- AI engineers → native viz of model internals
- clinicians → diagnostic embeddings (Parkinson's, dysphonia screening)
- voice coaches → pretrained voice-quality classifiers
- L2 learners → pronunciation scoring via SSL
- forensic → speaker embeddings (already used)
- bioacoustics → species classifiers
- experimenters → automatic post-trial feature extraction

Treating model outputs (logits, attention, hidden states, embeddings) as *just another time-aligned signal* alongside F0 and formants is a structural decision that pays off in seven directions.

**H. Scripting is needed by power users in every group, but Python's pull is decisive.** Phoneticians (R/Python now), AI engineers (Python first), psycholinguists (PsychoPy *is* Python), field linguists (some Python), forensic (some Python). R has a strong foothold in EMU-SDMS and in stats post-analysis, but no other group is R-native. The right shape is probably *the app exposes a Python API* — whether or not the app is itself written in Python.

**I. Experimentation is a bonus capability with high reuse.** Stimulus presentation + response collection overlaps with field elicitation, clinical protocols (CAPE-V), L2 drills, perception experiments, and crowdsourced data collection. The shared abstraction is "scripted protocol = sequence of (present stimulus, collect response, record participant, tag with metadata)." Building this once serves five groups.

**J. Privacy / on-device is non-negotiable for several groups.** Clinical (HIPAA), voice coaches (sensitive voice modification), forensic (chain of custody), field linguistics (community data sovereignty). Cloud-first is a non-starter. Architecturally: local-first, sync optional, encryption at rest.

### Architectural implications (provisional)

Translating the patterns above into stance, before any tech-stack discussion:

1. **Corpus-first, not file-first.** A "project" is the primary noun. Recordings, annotations, analyses, and derived signals live inside it. EMU-SDMS as prior art.
2. **Tiered annotation as a unifying data model.** Interval / point / continuous / categorical / numeric / embedding tiers. ELAN + EMU as prior art.
3. **Signal types are pluggable.** F0, formants, intensity, mel-spec, MFCC, wav2vec layer-N embedding, custom feature — all the same kind of object to the GUI. ML features = first-class.
4. **Reference distributions are first-class.** Compare any signal against a stored reference; ship some, let users publish more.
5. **Real-time path is designed in, not bolted on.** Live analysis as a peer to offline analysis.
6. **Desktop primary; phone is a thin client.** Phone records and runs the live-feedback path, syncs to the desktop project. Phone is not where corpora are managed.
7. **Browser embed is a deferred third surface.** Useful for crowdsourced experiments and teletherapy; not the primary delivery path.
8. **Scriptable via Python API,** independent of host language. The app embeds or exposes Python; the app itself need not be written in Python.
9. **Experimentation runner is a built-in module,** not a separate product. Shared protocol abstraction across elicitation / perception / clinical-protocol / drill.
10. **Local-first; encryption at rest; explicit opt-in for any network feature.**

### Open questions surfaced this session

- **Reference-distribution governance.** If we want a community-shareable distribution store, who curates? Version? License?
- **Articulatory channels.** Do we treat ultrasound video / EMA / palatography as in-scope multichannel signals, or push them to plugins? Phonetician-side feature with no overlap to other groups except via "signals are pluggable."
- **Clinical regulatory stance.** Build with potential FDA / CE pathway in mind, or stay explicitly "research use only" and let a downstream fork pursue clearance?
- **Experiment-runner ambitions.** Match PsychoPy's depth, or stay at "good enough for production/perception probes" (i.e. Demo Window+) and let serious experimenters use PsychoPy?
- **Host-language candidates** for the app itself, given the Python-API decision: Rust (Tauri), C++ (Qt), Kotlin Multiplatform, Swift+Kotlin twin native, Electron + native engine, Flutter + native engine. Deferred to a future entry.

---

## 2026-05-15 — Landscape survey: existing Praat alternatives

First entry. Project is greenfield; no code, language, or architecture chosen yet. Goal of this session: survey the current state of phonetics / speech-science software to understand where Praat sits, what alternatives exist, and where the real gaps are.

### Findings, grouped by what each tool is actually for

The field is fragmented. Most practitioners glue 3–4 tools together rather than using one. Nothing is currently positioned as a true Praat successor.

#### 1. Direct GUI alternatives (Praat-shaped tools)

- **Praat** (Boersma & Weenink, UvA) — still the incumbent, actively released (6.4.x series). Architecture, scripting language, and Motif-derived GUI are essentially frozen in shape. Well-known pain points unchanged: no undo, no autosave, idiosyncratic scripting (no inline `#` comments, whitespace-sensitive, opaque error messages), no native package/extension story.
- **Phonometrica** (GPL, C++/Qt) — the most ambitious "redo Praat properly" attempt. Corpus-first design, Lua scripting, modern Qt UI. Small community; intermittent development.
- **InFormant** (open source, `in-formant/in-formant`) — modern, real-time-focused. Spectrogram, formant tracking, glottal inverse filtering. Strong live-analysis UX (popular in trans-voice / vocal coaching). Not a corpus/annotation tool.
- **VoiceLab** (MIT, `Voice-Lab/VoiceLab`) — GUI wrapper over Parselmouth for batch automated voice analysis. Reproducibility-focused; loads many files, outputs CSV of pitch/jitter/shimmer/formants. Not interactive in the Praat sense.
- **Speech Analyzer** (SIL) — free, Windows-only, oriented toward field linguistics.
- **WaveSurfer** (KTH) — historically the main alternative; effectively unmaintained.
- **xkl** — modernized GTK port of Klatt's classic analysis/synthesis tool. Niche but technically interesting.
- **TrackDraw** — research toy for Klatt-synthesizer formant track drawing.

#### 2. R ecosystem — EMU-SDMS

The most complete *non-Praat* ecosystem. Either prior art or an integration target.

- **emuDB** — corpus/database format (audio + hierarchical annotations).
- **wrassp** — R wrapper around `libassp` (formants, F0, RMS, spectral, zero-crossings, filtering). Replaces Praat's signal-processing engine.
- **emuR** — query and analysis in R.
- **EMU-webApp** — browser-based annotation/inspection front-end; talks to emuR over websocket.

Closest thing to "Praat done as a database + R + web stack." Weakness: GUI is annotation-focused, audience is mostly R users.

#### 3. Python ecosystem

- **Parselmouth** — Python bindings calling Praat's actual C++ code. Now the de facto way to script Praat analyses; cited heavily in bioacoustics and clinical voice papers. No GUI — library only.
- **librosa**, **torchaudio**, **opensmile**, **SPTK** — general MIR/audio-ML libraries; partial overlap with phonetics needs.

#### 4. Forced alignment (used alongside Praat)

- **Montreal Forced Aligner (MFA)** — Kaldi GMM-HMM; current standard. Outputs TextGrids straight into Praat. Active 2025 work on low-resource fine-tuning.
- **WebMAUS / BAS** — Munich's web service; integrates with ELAN.
- **Charsiu** — transformer-based, newer alternative.

#### 5. Annotation-centric

- **ELAN** (Max Planck) — multimedia tier-based annotation, gold standard for video+audio corpora. Not an analysis tool.
- **Phon** — child phonology corpora.
- **Phonology Assistant** (SIL) — phonetic data charting.

#### 6. Proprietary clinical

- **CSL / MDVP** (Pentax / KayPENTAX) — clinical voice labs; MDVP module is why hospitals still buy it. Expensive, locked.
- **Dr. Speech**, **TF32** — niche clinical tools.

#### 7. General audio (used as crutches)

- **Sonic Visualiser** — extensible via Vamp plugins, good general spectrogram inspection, no phonetics-specific affordances.
- **Audacity**, **Ocenaudio** — general audio editing.

### Read on the gap

The field has bifurcated:

- **GUI inspection** → still Praat (because nothing else is full-featured); InFormant taking real-time voice work.
- **Reproducible analysis** → Parselmouth (Python) or wrassp/emuR (R) — both library-shaped, neither GUI.
- **Corpus annotation** → ELAN + MFA + Praat-as-viewer stitched together.
- **Clinical** → proprietary CSL.

The real gap a successor could fill: a single tool that is **interactive like Praat**, **scriptable like Parselmouth**, **corpus-aware like EMU-SDMS**, and **ML-native** — i.e. wav2vec2 / HuBERT / Whisper features as first-class signal types alongside formants and F0. Phonometrica reaches for some of this but hasn't achieved escape velocity.

### Open questions to resolve next

1. **Target users.** Research phoneticians? Clinicians? L2 learners? Voice coaches? Field linguists? The answer drives nearly everything else.
2. **Scope.** All-in-one (Praat-style) vs. focused (pick one of: analysis, annotation, corpus management, ML-feature inspection).
3. **Architecture stance.** Desktop GUI (Qt / Tauri / native), web-first (browser app + local engine), or library-with-thin-GUI?
4. **Scripting story.** Embedded Python? Lua? A DSL? None?
5. **Praat-compat surface.** Read/write TextGrids — table stakes. Run Praat scripts — explicit non-goal? Or interop layer?
6. **ML integration.** First-class transformer-based features (SSL embeddings, alignment, ASR) vs. classical-DSP-only.

### Sources consulted

- A Phonetician's Software Toolkit — Will Styler: https://wstyler.ucsd.edu/posts/phoneticians_software.html
- Praat: https://www.fon.hum.uva.nl/praat/
- Phonometrica: https://github.com/phonometrica/phonometrica
- InFormant: https://in-formant.app/ and https://github.com/in-formant/in-formant
- VoiceLab: https://voice-lab.github.io/VoiceLab/ ; Interspeech 2022 paper https://www.isca-archive.org/interspeech_2022/feinberg22_interspeech.pdf
- EMU-SDMS: https://ips-lmu.github.io/EMU.html ; emuR https://github.com/IPS-LMU/emuR ; CSL 2017 paper https://www.phonetik.uni-muenchen.de/~jmh/papers/emucsl.pdf
- Parselmouth for bioacoustics (2023): https://www.tandfonline.com/doi/full/10.1080/09524622.2023.2259327
- Montreal Forced Aligner: https://montrealcorpustools.github.io/Montreal-Forced-Aligner/
- SIL Speech Analyzer: https://software.sil.org/speech-analyzer/ ; SIL Phonology Assistant: https://software.sil.org/phonologyassistant/
- Evaluation of Praat (Bamberg): https://www.uni-bamberg.de/fileadmin/eng-ling/fs/Chapter_13/324EvaluationofPraat.html
- xkl modernization: https://www.researchgate.net/publication/373229441_xkl_A_legacy_software_for_detailed_acoustic_analysis_of_speech_made_modern
