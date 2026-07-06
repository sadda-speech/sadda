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
    };
    PyValueError::new_err(msg)
}

/// CTC forced alignment. `emissions` is a `(T, C)` float32 array of per-frame
/// log-probabilities (blank included); `targets` is the phone-token id sequence
/// (none equal to `blank`). Returns one span per target as
/// `(token, label, start_frame, end_frame, score)` — contiguous over `0..T`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (emissions, targets, *, blank=0))]
#[allow(clippy::type_complexity)]
pub(crate) fn forced_align(
    emissions: PyReadonlyArray2<'_, f32>,
    targets: Vec<usize>,
    blank: usize,
) -> PyResult<Vec<(usize, usize, usize, usize, f32)>> {
    let arr = emissions.as_array();
    let t = arr.nrows();
    // Owned rows sidestep row-contiguity assumptions on the incoming array.
    let owned: Vec<Vec<f32>> = (0..t).map(|i| arr.row(i).to_vec()).collect();
    let rows: Vec<&[f32]> = owned.iter().map(Vec::as_slice).collect();
    let spans = sadda_engine::forced_align(&rows, &targets, blank).map_err(align_err_to_py)?;
    Ok(spans
        .into_iter()
        .map(|s| (s.token, s.label, s.start_frame, s.end_frame, s.score))
        .collect())
}
