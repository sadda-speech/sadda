//! E9 — built-in `sadda.app` Python module exposing the GUI's
//! current state to embedded scripts. Registered via
//! `pyo3::append_to_inittab!` before the interpreter starts so
//! `import sadda.app` works without a pip install.
//!
//! ## State plumbing
//!
//! Functions in this module read from a thread-local pointer to an
//! `AppSnapshot` set up by [`with_snapshot_active`] immediately
//! before the embed runs. Outside an active session, every
//! function raises `RuntimeError`. The snapshot lives on the GUI
//! thread's stack; the GIL is held for the duration of the
//! script, so no other thread can race on it.

use std::cell::Cell;
use std::path::PathBuf;
use std::ptr::NonNull;

use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Snapshot of the GUI's current state, set up by the caller
/// before running an embedded script. Read by the `sadda.app`
/// functions below.
pub struct AppSnapshot {
    pub project_root: PathBuf,
    pub bundle: Option<BundleInfo>,
    pub selection: Option<SelectionInfo>,
    pub cursor_seconds: f64,
}

/// Active bundle metadata exposed to scripts.
pub struct BundleInfo {
    pub id: i64,
    pub name: String,
    pub sample_rate: u32,
    pub duration_seconds: f64,
}

/// Identifies the currently-selected annotation. Mirrors the
/// app-side `AnnotationSelection` enum but stays defined here so
/// the PyO3 module doesn't pull in the rest of `main.rs`.
pub struct SelectionInfo {
    pub kind: SelectionKind,
    pub tier_id: i64,
    pub annotation_id: i64,
}

#[derive(Debug, Clone, Copy)]
pub enum SelectionKind {
    Interval,
    Point,
}

impl SelectionKind {
    fn as_str(self) -> &'static str {
        match self {
            SelectionKind::Interval => "interval",
            SelectionKind::Point => "point",
        }
    }
}

/// Output of a script run that registered commands. The caller
/// drains this after `with_snapshot_active` returns and appends to
/// the app's command list.
#[derive(Default)]
pub struct ScriptSessionExtras {
    pub registered_commands: Vec<(String, Py<PyAny>)>,
}

thread_local! {
    static APP_SNAPSHOT: Cell<Option<NonNull<AppSnapshot>>> = const { Cell::new(None) };
    static SESSION_EXTRAS: Cell<Option<NonNull<ScriptSessionExtras>>> = const { Cell::new(None) };
}

/// Runs `f` with the given snapshot + extras installed in the
/// thread-local cells. Cleared in a `Drop` guard so panics still
/// clear the pointer.
///
/// SAFETY: `snapshot` and `extras` must outlive the body of `f`.
/// In practice the caller stack-allocates both, calls this with
/// `&mut` references, and immediately reads `extras` after the
/// call returns — that pattern is sound by construction.
pub fn with_snapshot_active<R>(
    snapshot: &AppSnapshot,
    extras: &mut ScriptSessionExtras,
    f: impl FnOnce() -> R,
) -> R {
    struct ClearOnDrop;
    impl Drop for ClearOnDrop {
        fn drop(&mut self) {
            APP_SNAPSHOT.with(|c| c.set(None));
            SESSION_EXTRAS.with(|c| c.set(None));
        }
    }
    APP_SNAPSHOT.with(|c| {
        c.set(Some(NonNull::from(snapshot)));
    });
    SESSION_EXTRAS.with(|c| {
        c.set(Some(NonNull::from(extras)));
    });
    let _guard = ClearOnDrop;
    f()
}

/// Returns true iff a snapshot is currently installed — useful for
/// tests / debug asserts.
#[cfg(test)]
pub fn snapshot_active() -> bool {
    APP_SNAPSHOT.with(|c| c.get().is_some())
}

fn with_snapshot<R>(f: impl FnOnce(&AppSnapshot) -> R) -> PyResult<R> {
    APP_SNAPSHOT.with(|cell| {
        let ptr = cell.get().ok_or_else(|| {
            PyRuntimeError::new_err("sadda.app called outside an active app session")
        })?;
        // SAFETY: contract on `with_snapshot_active`.
        Ok(f(unsafe { ptr.as_ref() }))
    })
}

fn with_extras<R>(f: impl FnOnce(&mut ScriptSessionExtras) -> R) -> PyResult<R> {
    SESSION_EXTRAS.with(|cell| {
        let mut ptr = cell.get().ok_or_else(|| {
            PyRuntimeError::new_err("sadda.app called outside an active app session")
        })?;
        // SAFETY: contract on `with_snapshot_active`.
        Ok(f(unsafe { ptr.as_mut() }))
    })
}

// ---------------------------------------------------------------------------
// PyO3 module
// ---------------------------------------------------------------------------

/// Top-level `sadda` module — re-exposes `app` as a submodule so
/// `import sadda.app` works. Not tagged `#[pymodule]` itself; the
/// outer wrapper in `main.rs` carries the macro (so the symbol
/// generated is `PyInit_sadda` exactly once across the crate).
pub fn sadda(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let app_module = PyModule::new(py, "app")?;
    register_app_module(&app_module)?;
    m.add_submodule(&app_module)?;
    // Register the submodule under sys.modules so `import sadda.app`
    // works as well as `from sadda import app`.
    let sys = py.import("sys")?;
    let sys_modules = sys.getattr("modules")?;
    sys_modules.set_item("sadda.app", &app_module)?;
    Ok(())
}

fn register_app_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(project_root, m)?)?;
    m.add_function(wrap_pyfunction!(active_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(current_selection, m)?)?;
    m.add_function(wrap_pyfunction!(cursor_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(register_command, m)?)?;
    m.add_function(wrap_pyfunction!(registered_command_names, m)?)?;
    m.add_function(wrap_pyfunction!(refresh, m)?)?;
    Ok(())
}

/// Returns the project root path as a string.
#[pyfunction]
fn project_root() -> PyResult<String> {
    with_snapshot(|s| s.project_root.to_string_lossy().into_owned())
}

/// Returns the active bundle's metadata, or `None` if no bundle
/// is selected.
#[pyfunction]
fn active_bundle(py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
    with_snapshot(|s| {
        s.bundle.as_ref().map(|b| {
            let d = PyDict::new(py);
            d.set_item("id", b.id).ok();
            d.set_item("name", &b.name).ok();
            d.set_item("sample_rate", b.sample_rate).ok();
            d.set_item("duration_seconds", b.duration_seconds).ok();
            d.unbind()
        })
    })
}

/// Returns a dict describing the currently-selected annotation,
/// or `None` if no annotation is selected. Keys: `kind` (one of
/// `"interval"` / `"point"`), `tier_id` (int), `annotation_id` (int).
#[pyfunction]
fn current_selection(py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
    with_snapshot(|s| {
        s.selection.as_ref().map(|sel| {
            let d = PyDict::new(py);
            d.set_item("kind", sel.kind.as_str()).ok();
            d.set_item("tier_id", sel.tier_id).ok();
            d.set_item("annotation_id", sel.annotation_id).ok();
            d.unbind()
        })
    })
}

/// Returns the current playback / scrub cursor in seconds.
#[pyfunction]
fn cursor_seconds() -> PyResult<f64> {
    with_snapshot(|s| s.cursor_seconds)
}

/// Registers a Python callable as a named command. The command
/// appears in the Ctrl/Cmd+P palette and runs synchronously on
/// the GUI thread when invoked.
#[pyfunction]
fn register_command(name: String, callable: Bound<'_, PyAny>) -> PyResult<()> {
    if !callable.is_callable() {
        return Err(PyTypeError::new_err(
            "register_command: second argument must be callable",
        ));
    }
    with_extras(|extras| {
        extras
            .registered_commands
            .push((name, callable.clone().unbind()));
    })
}

/// Returns the names of every command currently registered. Mainly
/// for diagnostics / scripts that want to verify their registration
/// landed.
#[pyfunction]
fn registered_command_names<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
    let extras_names: Vec<String> = with_extras(|extras| {
        extras
            .registered_commands
            .iter()
            .map(|(n, _)| n.clone())
            .collect()
    })?;
    let list = PyList::empty(py);
    for n in extras_names {
        list.append(n)?;
    }
    Ok(list)
}

/// No-op shim for forward-compatibility with the API-surface
/// entry's `sadda.app.refresh()`. Egui repaints reactively, so
/// scripts don't need to ask for a redraw — but exposing the call
/// surface now means scripts that already use it won't break.
#[pyfunction]
fn refresh() -> PyResult<()> {
    // Nothing to do; egui handles repaints on event input.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_inactive_by_default() {
        assert!(!snapshot_active());
    }

    #[test]
    fn with_snapshot_active_sets_and_clears() {
        let snap = AppSnapshot {
            project_root: PathBuf::from("/tmp/x"),
            bundle: None,
            selection: None,
            cursor_seconds: 0.0,
        };
        let mut extras = ScriptSessionExtras::default();
        let saw_active = with_snapshot_active(&snap, &mut extras, snapshot_active);
        assert!(saw_active);
        assert!(!snapshot_active(), "snapshot must clear after callback");
    }
}
