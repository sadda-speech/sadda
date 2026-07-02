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

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// A visibility change requested by a script (`sadda.app.set_pane_visible` /
/// `set_tier_visible`), applied by the app after the run. Kept as a plain enum
/// so this module stays independent of the app's config types — the foundation
/// for the scripted documentation-capture pathway (a recipe composes the exact
/// view it wants, then captures it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisibilityAction {
    Pane { pane: SignalPaneId, visible: bool },
    Tier { tier_id: i64, visible: bool },
}

/// The signal subpanes a script can show/hide by name. Excludes the embedding
/// heatmap, whose visibility is driven by *selecting a tier* rather than a
/// boolean (see `EmbeddingHeatmapConfig`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalPaneId {
    Waveform,
    Spectrogram,
    TierStrip,
    F0,
    Formants,
    Intensity,
    Vad,
    Mfcc,
}

impl SignalPaneId {
    /// Parses the canonical pane name (case-insensitive), accepting a few
    /// natural aliases (`pitch`→f0, `tiers`→tier_strip).
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "waveform" | "wave" => Some(Self::Waveform),
            "spectrogram" | "spec" => Some(Self::Spectrogram),
            "tier_strip" | "tier-strip" | "tiers" => Some(Self::TierStrip),
            "f0" | "pitch" => Some(Self::F0),
            "formants" | "formant" => Some(Self::Formants),
            "intensity" => Some(Self::Intensity),
            "vad" => Some(Self::Vad),
            "mfcc" => Some(Self::Mfcc),
            _ => None,
        }
    }

    /// Human-readable list of accepted names, for error messages.
    pub const ACCEPTED_NAMES: &'static str =
        "waveform, spectrogram, tier_strip, f0, formants, intensity, vad, mfcc";
}

/// A resizable GUI *column* a script can width-set (`set_column_width`). The
/// central signal column fills the remaining width, so it isn't listed — set
/// the side columns plus the window size instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuiColumn {
    /// Left bundle-list sidebar.
    Bundles,
    /// Right inline annotation panel.
    Annotation,
    /// Right reference-distribution panel.
    Reference,
}

impl GuiColumn {
    /// Parses the column name (case-insensitive) with natural aliases.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "bundles" | "bundle" | "sidebar" | "bundle_sidebar" => Some(Self::Bundles),
            "annotation" | "annotations" | "annotation_panel" => Some(Self::Annotation),
            "reference" | "reference_distros" | "reference_panel" | "distributions" => {
                Some(Self::Reference)
            }
            _ => None,
        }
    }

    /// Human-readable list of accepted names, for error messages.
    pub const ACCEPTED_NAMES: &'static str = "bundles, annotation, reference";
}

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
    /// Visibility changes the script requested; drained + applied by the app
    /// after the run (like `registered_commands`).
    pub visibility_actions: Vec<VisibilityAction>,
    /// S5: a window resize the script requested, in logical points
    /// (`set_window_size`). Last write wins. Applied by the app after the run.
    pub window_size: Option<(f32, f32)>,
    /// S6.1: per-pane height changes the script requested, in logical points
    /// (`set_pane_height`) — for reproducible figure layouts. Applied by the app
    /// after the run.
    pub pane_heights: Vec<(SignalPaneId, f32)>,
    /// S6.2: GUI column width changes the script requested, in logical points
    /// (`set_column_width`). Applied by the app after the run.
    pub column_widths: Vec<(GuiColumn, f32)>,
    /// S7: a theme change the script requested (`set_theme`) — `"light"`,
    /// `"dark"`, or `"system"`. Last write wins.
    pub theme: Option<String>,
}

/// S7: a single documentation shot collected by `sadda.doc.shot(...)` — a fully
/// declarative spec of *one* image. The headless recipe runner applies it,
/// settles the DSP, renders, crops to `capture`, and writes `to`. Everything
/// except `to` + `capture` is optional (unset = leave as-is / app default).
// Fields are populated by `shot()` (compiled into the live binary) but read
// only by the `#[cfg(test)]` headless runner, so the non-test build sees them
// as write-only.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct RecipeShot {
    /// Output path for the PNG (resolved by the runner).
    pub to: String,
    /// Region to capture: a named target (e.g. `"signal-column"`) or a pixel
    /// rect as `"rect:x,y,w,h"`.
    pub capture: String,
    /// Project to open (path). Mutually exclusive with `audio`.
    pub project: Option<String>,
    /// Audio file to build a throwaway one-bundle project from — keeps a recipe
    /// self-contained (no committed project DB). Mutually exclusive with
    /// `project`.
    pub audio: Option<String>,
    /// Bundle to select, by name.
    pub bundle: Option<String>,
    /// Window size in logical points.
    pub size: Option<(f32, f32)>,
    /// Theme: `"light"`, `"dark"`, or `"system"`.
    pub theme: Option<String>,
    /// If set, exactly these signal panes are shown (all others hidden).
    pub show: Option<Vec<String>>,
    /// Per-pane heights, `(pane_name, points)`.
    pub heights: Vec<(String, f32)>,
    /// Per-column widths, `(column_name, points)`.
    pub widths: Vec<(String, f32)>,
}

thread_local! {
    /// Shots accumulated by `sadda.doc.shot` during a recipe run, drained by
    /// the headless runner via [`take_recipe_shots`].
    static RECIPE_SHOTS: std::cell::RefCell<Vec<RecipeShot>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Drains the shots a recipe queued. Called by the headless runner after
/// executing the recipe source. Test-only: the runner (which depends on the
/// `egui_kittest` dev-dependency) is the sole consumer.
#[cfg(test)]
pub fn take_recipe_shots() -> Vec<RecipeShot> {
    RECIPE_SHOTS.with(|s| std::mem::take(&mut *s.borrow_mut()))
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
    let sys = py.import("sys")?;
    let sys_modules = sys.getattr("modules")?;

    // `sadda.app` — live GUI state + interactive setters.
    let app_module = PyModule::new(py, "app")?;
    register_app_module(&app_module)?;
    m.add_submodule(&app_module)?;
    // Register under sys.modules so `import sadda.app` works as well as
    // `from sadda import app`.
    sys_modules.set_item("sadda.app", &app_module)?;

    // `sadda.doc` — S7 documentation-recipe authoring (declarative shots).
    let doc_module = PyModule::new(py, "doc")?;
    doc_module.add_function(wrap_pyfunction!(shot, &doc_module)?)?;
    m.add_submodule(&doc_module)?;
    sys_modules.set_item("sadda.doc", &doc_module)?;

    Ok(())
}

fn register_app_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(project_root, m)?)?;
    m.add_function(wrap_pyfunction!(active_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(current_selection, m)?)?;
    m.add_function(wrap_pyfunction!(cursor_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(register_command, m)?)?;
    m.add_function(wrap_pyfunction!(registered_command_names, m)?)?;
    m.add_function(wrap_pyfunction!(set_pane_visible, m)?)?;
    m.add_function(wrap_pyfunction!(set_tier_visible, m)?)?;
    m.add_function(wrap_pyfunction!(set_window_size, m)?)?;
    m.add_function(wrap_pyfunction!(set_pane_height, m)?)?;
    m.add_function(wrap_pyfunction!(set_column_width, m)?)?;
    m.add_function(wrap_pyfunction!(set_theme, m)?)?;
    m.add_function(wrap_pyfunction!(refresh, m)?)?;
    Ok(())
}

/// Normalises a theme name, or returns `None` if unrecognised. Accepted:
/// `light`, `dark`, `system` (case-insensitive; `auto` aliases `system`).
pub fn normalize_theme(name: &str) -> Option<&'static str> {
    match name.trim().to_ascii_lowercase().as_str() {
        "light" => Some("light"),
        "dark" => Some("dark"),
        "system" | "auto" => Some("system"),
        _ => None,
    }
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

/// Shows or hides a signal subpane by name. Valid names:
/// `waveform`, `spectrogram`, `tier_strip`, `f0`, `formants`, `intensity`,
/// `vad`, `mfcc` (aliases: `pitch`→f0, `tiers`→tier_strip). The change is
/// applied to the live GUI right after the script finishes. Raises
/// `ValueError` for an unknown pane.
#[pyfunction]
fn set_pane_visible(name: String, visible: bool) -> PyResult<()> {
    let pane = SignalPaneId::from_name(&name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown pane {name:?}; expected one of: {}",
            SignalPaneId::ACCEPTED_NAMES
        ))
    })?;
    with_extras(|e| {
        e.visibility_actions
            .push(VisibilityAction::Pane { pane, visible });
    })
}

/// Shows or hides one annotation tier in the strip, by tier id. Ids come from
/// `active_bundle()` / a tier listing; an unknown id is a harmless no-op. The
/// change is applied to the live GUI right after the script finishes.
#[pyfunction]
fn set_tier_visible(tier_id: i64, visible: bool) -> PyResult<()> {
    with_extras(|e| {
        e.visibility_actions
            .push(VisibilityAction::Tier { tier_id, visible });
    })
}

/// Resizes the app window to `width` × `height` **logical points** (the same
/// unit as the View ▸ Doc size presets). Applied to the live window right after
/// the script finishes, with UI zoom pinned to 100% so documentation shots keep
/// consistent proportions. Raises `ValueError` for a non-positive size.
#[pyfunction]
fn set_window_size(width: f64, height: f64) -> PyResult<()> {
    if !(width.is_finite() && height.is_finite()) || width < 1.0 || height < 1.0 {
        return Err(PyValueError::new_err(
            "set_window_size: width and height must be positive, finite numbers",
        ));
    }
    with_extras(|e| {
        e.window_size = Some((width as f32, height as f32));
    })
}

/// Sets the height of one signal-column pane to `height` **logical points**,
/// for reproducible figure layouts. Valid panes: `waveform`, `tier_strip`,
/// `f0`, `formants`, `intensity`, `vad`, `mfcc` (same names/aliases as
/// [`set_pane_visible`]). The **spectrogram fills the remaining height**, so it
/// can't be sized directly — set the other panes plus the window size instead.
/// Applied to the live layout right after the script finishes; raises
/// `ValueError` for an unknown pane, the spectrogram, or a non-positive height.
#[pyfunction]
fn set_pane_height(name: String, height: f64) -> PyResult<()> {
    let pane = SignalPaneId::from_name(&name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown pane {name:?}; expected one of: {}",
            SignalPaneId::ACCEPTED_NAMES
        ))
    })?;
    if matches!(pane, SignalPaneId::Spectrogram) {
        return Err(PyValueError::new_err(
            "set_pane_height: the spectrogram fills the remaining height — \
             set the other pane heights and the window size instead",
        ));
    }
    if !height.is_finite() || height < 1.0 {
        return Err(PyValueError::new_err(
            "set_pane_height: height must be a positive, finite number of points",
        ));
    }
    with_extras(|e| e.pane_heights.push((pane, height as f32)))
}

/// Sets the width of one GUI column to `width` **logical points**. Valid
/// columns: `bundles` (left sidebar), `annotation`, `reference` (right panels).
/// The **signal column fills the remaining width**, so it can't be sized
/// directly — set the side columns plus the window size instead. Applied to the
/// live layout right after the script finishes; raises `ValueError` for an
/// unknown column, the signal column, or a non-positive width.
#[pyfunction]
fn set_column_width(name: String, width: f64) -> PyResult<()> {
    // Friendly rejection for the central signal column (the remainder).
    if matches!(
        name.trim().to_ascii_lowercase().as_str(),
        "signals" | "signal" | "signal_column" | "signals_column" | "center" | "centre"
    ) {
        return Err(PyValueError::new_err(
            "set_column_width: the signal column fills the remaining width — \
             set the side columns and the window size instead",
        ));
    }
    let col = GuiColumn::from_name(&name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown column {name:?}; expected one of: {}",
            GuiColumn::ACCEPTED_NAMES
        ))
    })?;
    if !width.is_finite() || width < 1.0 {
        return Err(PyValueError::new_err(
            "set_column_width: width must be a positive, finite number of points",
        ));
    }
    with_extras(|e| e.column_widths.push((col, width as f32)))
}

/// Sets the app theme: `light`, `dark`, or `system` (case-insensitive).
/// Applied to the live GUI right after the script finishes. Raises `ValueError`
/// for an unrecognised name.
#[pyfunction]
fn set_theme(name: String) -> PyResult<()> {
    let normalized = normalize_theme(&name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "unknown theme {name:?}; expected light, dark, or system"
        ))
    })?;
    with_extras(|e| e.theme = Some(normalized.to_string()))
}

/// S7: append a documentation shot. Every field except `to` and `capture` is
/// optional. Called from a Python recipe (`import sadda.doc as doc`); the
/// headless runner executes the accumulated shots. Does not touch the live GUI.
#[pyfunction]
#[pyo3(signature = (to, capture, project=None, audio=None, bundle=None, size=None, theme=None, show=None, heights=None, widths=None))]
#[allow(clippy::too_many_arguments)]
fn shot(
    to: String,
    capture: String,
    project: Option<String>,
    audio: Option<String>,
    bundle: Option<String>,
    size: Option<(f32, f32)>,
    theme: Option<String>,
    show: Option<Vec<String>>,
    heights: Option<Vec<(String, f32)>>,
    widths: Option<Vec<(String, f32)>>,
) -> PyResult<()> {
    if project.is_some() && audio.is_some() {
        return Err(PyValueError::new_err(
            "shot(): pass either project= or audio=, not both",
        ));
    }
    if let Some(t) = &theme {
        if normalize_theme(t).is_none() {
            return Err(PyValueError::new_err(format!(
                "shot(theme=...): unknown theme {t:?}; expected light, dark, or system"
            )));
        }
    }
    RECIPE_SHOTS.with(|s| {
        s.borrow_mut().push(RecipeShot {
            to,
            capture,
            project,
            audio,
            bundle,
            size,
            theme,
            show,
            heights: heights.unwrap_or_default(),
            widths: widths.unwrap_or_default(),
        });
    });
    Ok(())
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

    /// End-to-end: build the real `sadda` module, call the visibility setters
    /// *through Python*, and confirm they populate the session extras (and that
    /// an unknown pane raises `ValueError`). Calls the functions directly on the
    /// module object rather than `import sadda.app`, so the test needs neither
    /// `append_to_inittab!` (which must run before the interpreter starts — too
    /// fragile across a shared test binary) nor `sys.modules` fixup.
    #[test]
    fn python_visibility_setters_populate_extras() {
        let snap = AppSnapshot {
            project_root: PathBuf::from("/tmp/x"),
            bundle: None,
            selection: None,
            cursor_seconds: 0.0,
        };
        let mut extras = ScriptSessionExtras::default();
        with_snapshot_active(&snap, &mut extras, || {
            Python::attach(|py| {
                let m = PyModule::new(py, "sadda").expect("build module");
                sadda(&m).expect("register submodule");
                let app = m.getattr("app").expect("sadda.app submodule");

                app.call_method1("set_pane_visible", ("spectrogram", false))
                    .expect("set_pane_visible ok");
                app.call_method1("set_tier_visible", (7_i64, false))
                    .expect("set_tier_visible ok");
                // Aliases resolve (pitch → f0).
                app.call_method1("set_pane_visible", ("pitch", true))
                    .expect("alias ok");
                // Unknown pane → ValueError.
                let err = app
                    .call_method1("set_pane_visible", ("bogus", true))
                    .expect_err("unknown pane should raise");
                assert!(err.is_instance_of::<PyValueError>(py));

                // S5 window sizing: valid resize queues; non-positive rejects.
                app.call_method1("set_window_size", (1280.0, 800.0))
                    .expect("set_window_size ok");
                let bad = app
                    .call_method1("set_window_size", (0.0, 100.0))
                    .expect_err("non-positive size should raise");
                assert!(bad.is_instance_of::<PyValueError>(py));

                // S6.1 pane heights: a normal pane queues; the spectrogram (the
                // flex pane) is rejected.
                app.call_method1("set_pane_height", ("waveform", 140.0))
                    .expect("set_pane_height ok");
                let spec = app
                    .call_method1("set_pane_height", ("spectrogram", 200.0))
                    .expect_err("sizing the spectrogram should raise");
                assert!(spec.is_instance_of::<PyValueError>(py));

                // S6.2 column widths: a side column queues; the signal column
                // (the flex column) is rejected.
                app.call_method1("set_column_width", ("bundles", 180.0))
                    .expect("set_column_width ok");
                let sig = app
                    .call_method1("set_column_width", ("signals", 400.0))
                    .expect_err("sizing the signal column should raise");
                assert!(sig.is_instance_of::<PyValueError>(py));
            });
        });

        assert_eq!(extras.window_size, Some((1280.0, 800.0)));
        assert_eq!(
            extras.pane_heights,
            vec![(SignalPaneId::Waveform, 140.0)],
            "only the valid pane-height call should have queued"
        );
        assert_eq!(
            extras.column_widths,
            vec![(GuiColumn::Bundles, 180.0)],
            "only the valid column-width call should have queued"
        );

        assert!(extras.visibility_actions.contains(&VisibilityAction::Pane {
            pane: SignalPaneId::Spectrogram,
            visible: false,
        }));
        assert!(extras.visibility_actions.contains(&VisibilityAction::Pane {
            pane: SignalPaneId::F0,
            visible: true,
        }));
        assert!(extras.visibility_actions.contains(&VisibilityAction::Tier {
            tier_id: 7,
            visible: false,
        }));
        // The rejected call must not have enqueued anything.
        assert_eq!(extras.visibility_actions.len(), 3);
    }

    #[test]
    fn pane_name_parsing_accepts_aliases_and_rejects_unknown() {
        assert_eq!(
            SignalPaneId::from_name("Waveform"),
            Some(SignalPaneId::Waveform)
        );
        assert_eq!(SignalPaneId::from_name("pitch"), Some(SignalPaneId::F0));
        assert_eq!(
            SignalPaneId::from_name("tiers"),
            Some(SignalPaneId::TierStrip)
        );
        assert_eq!(
            SignalPaneId::from_name("  SPEC "),
            Some(SignalPaneId::Spectrogram)
        );
        assert_eq!(SignalPaneId::from_name("nope"), None);
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
