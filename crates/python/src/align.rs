//! PyO3 bindings for forced alignment — the engine's CTC forced-align DP
//! exposed to Python as `sadda._native.forced_align`. The `sadda.align` package
//! wraps this with G2P (espeak-ng) + the acoustic model (`sadda.ml`) to produce
//! Word/Phone tiers. Thin wrapper over [`sadda_engine::forced_align`].

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use sadda_engine::AlignError;

fn align_err_to_py(e: AlignError) -> PyErr {
    let msg = match e {
        AlignError::EmptyTargets => "targets is empty".to_string(),
        AlignError::NoFrames => "emissions has no frames".to_string(),
        AlignError::LabelOutOfRange => {
            "a target id or blank is outside the emission's class range".to_string()
        }
        AlignError::TooFewFrames { needed, have } => {
            format!("too few frames to align: need {needed}, have {have}")
        }
        AlignError::SilenceMaskLength => {
            "silence_mask length does not match the number of frames".to_string()
        }
    };
    PyValueError::new_err(msg)
}

/// CTC forced alignment. `emissions` is a `(T, C)` float32 array of per-frame
/// log-probabilities (blank included); `targets` is the phone-token id sequence
/// (none equal to `blank`). With `min_silence_frames > 0`, long CTC-blank runs
/// become silence intervals; `silence_mask` is an optional length-`T` bool array
/// (e.g. from VAD) whose true frames are silence.
///
/// Returns one span per interval as `(token, label, start_frame, end_frame,
/// score, is_silence)` — contiguous over `0..T`. Silence spans have
/// `token == 2**64-1` and `is_silence == True`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (emissions, targets, *, blank=0, min_silence_frames=0, silence_mask=None))]
#[allow(clippy::type_complexity)]
pub(crate) fn forced_align(
    emissions: PyReadonlyArray2<'_, f32>,
    targets: Vec<usize>,
    blank: usize,
    min_silence_frames: usize,
    silence_mask: Option<Vec<bool>>,
) -> PyResult<Vec<(usize, usize, usize, usize, f32, bool)>> {
    let arr = emissions.as_array();
    let t = arr.nrows();
    // Owned rows sidestep row-contiguity assumptions on the incoming array.
    let owned: Vec<Vec<f32>> = (0..t).map(|i| arr.row(i).to_vec()).collect();
    let rows: Vec<&[f32]> = owned.iter().map(Vec::as_slice).collect();
    let spans = sadda_engine::forced_align(
        &rows,
        &targets,
        blank,
        min_silence_frames,
        silence_mask.as_deref(),
    )
    .map_err(align_err_to_py)?;
    Ok(spans
        .into_iter()
        .map(|s| {
            (
                s.token,
                s.label,
                s.start_frame,
                s.end_frame,
                s.score,
                s.is_silence,
            )
        })
        .collect())
}
