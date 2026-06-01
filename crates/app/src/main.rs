//! Sadda desktop GUI. Slice A1 ships the project-aware shell —
//! welcome screen with New / Open / Recent, a `Project`-backed loaded
//! state, persistent window + recent-projects state. No content panes
//! yet; those land in cluster B (waveform / spectrogram / tier strip).
//!
//! See the 2026-05-23 DEVLOG entry "App shell + project open/create
//! (A1)" for the design rationale and the cut-list for what
//! deliberately doesn't ship at A1.

mod debug;
mod playback;
mod sadda_app;
mod state;

use std::path::{Path, PathBuf};

use eframe::egui;
use egui_plot::{
    Bar, BarChart, Line, LineStyle, Plot, PlotPoint, PlotPoints, Points, Polygon, Text, VLine,
};
use pyo3::prelude::*;
use sadda_engine::{LiveConfig, LiveResults, LiveSession, Project, StoppedSession, TierType};
// D10 measure tracks: the engine already emits these per-frame
// time series; the GUI computes + caches + renders them as lanes.
use sadda_engine::dsp::{FormantFrame, FormantsConfig, IntensityFrame, formants, intensity};
use sadda_engine::pitch::{PitchConfig, PitchFrame, PitchMethod, pitch};
// D10 refdist overlays: resolve a distribution from the store and turn
// it into a band the lane draws behind its contour.
use sadda_engine::{Histogram, MeasureKind, RefDist, RefdistStore, Summary};
// E11 VAD lane: the engine's `ml` feature is enabled for the app.
use sadda_engine::{VadFrame, vad_bundled};

use crate::playback::Playback;
use crate::sadda_app::{
    AppSnapshot, BundleInfo, ScriptSessionExtras, SelectionInfo, SelectionKind,
    with_snapshot_active,
};
use crate::state::{
    ColormapKind, EmbeddingHeatmapConfig, EnvelopeCache, MeasureTrackConfig, PersistedState,
    PlotPalette, RefdistOverlay, SpectrogramConfig, ThemePref, TimelineState,
    build_envelope_for_range, colormap_bake, format_reference_lane_caption, nearest_frame_index,
    normalize_embedding, power_to_db_normalized, truncate_label,
};

/// Maximum characters drawn inside an interval rectangle or above a
/// point tick before truncation kicks in (with an ellipsis).
const TIER_LABEL_MAX_CHARS: usize = 20;
/// Vertical pixels per lane in the tier strip.
const TIER_LANE_HEIGHT: f32 = 28.0;
/// Shared width of the left gutter that holds the y-axis ticks /
/// labels on the waveform + spectrogram plots and the tier name on
/// the tier strip. Using one value for all three keeps the time-axis
/// plot areas pixel-aligned, so the playback cursor draws a single
/// straight line top-to-bottom and all three views x-scale together.
/// Sized to fit a 16-char tier name; comfortably wider than the
/// widest expected Hz tickmark ("22050" at a 44.1 kHz sample rate).
const SIGNAL_LEFT_GUTTER: f32 = 120.0;
/// D10: default height (px) of each stacked measure-track lane. The
/// panels are resizable, so this is just the initial split.
const MEASURE_LANE_HEIGHT: f32 = 96.0;

const APP_TITLE: &str = "sadda";
/// Cap on spectrogram texture width. egui's typical max texture size
/// is 8192; 4096 keeps headroom and gives roughly 1px per ~150 ms at
/// 10 minutes — fine resolution for the long-recording case the B3
/// spike note flagged. Longer audio averages frames into buckets.
const MAX_SPECTROGRAM_WIDTH: usize = 4096;

fn main() -> eframe::Result<()> {
    // WSLg advertises a Wayland compositor (WAYLAND_DISPLAY=wayland-0),
    // but winit's Wayland backend broken-pipes against it during
    // event-loop construction and eframe bails with
    // `WinitEventLoop(ExitFailure(1))`. XWayland (DISPLAY=:0) works,
    // so under WSL we drop WAYLAND_DISPLAY before the event loop is
    // built, steering winit onto its X11 backend. No effect off WSL.
    force_x11_under_wsl();

    // Release bundles ship libonnxruntime as a sidecar under
    // `<exe-dir>/onnxruntime/`. If the user hasn't set ORT_DYLIB_PATH
    // themselves, pick the sidecar up so ML features (VAD lane,
    // embeddings) work without manual setup. The probe inside ensures
    // we don't set the env var unless the file actually exports the
    // ORT C API, avoiding a startup-time discovery that points the
    // runtime at a bogus file.
    discover_ort_sidecar();

    // E9: register the built-in `sadda` module BEFORE the embedded
    // CPython interpreter starts, so embedded scripts can
    // `import sadda.app` without needing the wheel pip-installed.
    // Must happen before any pyo3 call that might trigger
    // auto-initialize (the script-engine's first run_script).
    pyo3::append_to_inittab!(sadda);

    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1024.0, 720.0])
        .with_min_inner_size([640.0, 480.0])
        .with_title(APP_TITLE);
    // Window / taskbar / dock icon, embedded at build time. Decoded at
    // startup; if the bundled PNG ever fails to decode we just launch
    // without an icon rather than refusing to start.
    if let Ok(icon) = eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon.png")) {
        viewport = viewport.with_icon(icon);
    }
    let options = eframe::NativeOptions {
        viewport,
        // Under WSLg/Xwayland, a saved window geometry comes back maximized
        // at the INT16 position sentinel (-32768,-32768); winit then maps
        // pointer coordinates against that bogus origin, so hover/clicks
        // land offset from the cursor and the window feels frozen. Worse,
        // the bad position is re-saved every run, so it compounds. Disable
        // window-geometry persistence under WSL: eframe never *writes* the
        // "window" key, so once it's absent the app always opens at the
        // ViewportBuilder size below. (eframe restores an existing key
        // unconditionally — `persist_window` only gates saving — but with
        // saving off there's nothing to restore.) App state (recent
        // projects, prefs) is unaffected; it rides the Storage trait, not
        // this flag. No-op off WSL, where geometry restore is desirable.
        persist_window: !is_wsl(),
        ..Default::default()
    };
    eframe::run_native(
        APP_TITLE,
        options,
        Box::new(|cc| Ok(Box::new(SaddaApp::new(cc)))),
    )
}

/// Detects WSL and, if found, removes `WAYLAND_DISPLAY` so winit
/// selects its X11/XWayland backend instead of the WSLg Wayland one
/// (which fails on event-loop init — see the `main` preamble). Must
/// run before `eframe::run_native` builds the event loop.
fn force_x11_under_wsl() {
    if is_wsl() && std::env::var_os("WAYLAND_DISPLAY").is_some() {
        // SAFETY: called at the top of `main` before any thread is
        // spawned, so there is no concurrent access to the environment.
        unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
    }
}

/// True when running under WSL (WSLg). Gates the WSL-specific GUI
/// workarounds: forcing XWayland over the broken Wayland backend
/// (`force_x11_under_wsl`) and disabling window-geometry persistence
/// (which otherwise restores a maximized window parked at the off-screen
/// position sentinel and breaks pointer-coordinate mapping).
fn is_wsl() -> bool {
    std::env::var_os("WSL_INTEROP").is_some()
        || std::fs::read_to_string("/proc/sys/kernel/osrelease")
            .map(|s| s.to_ascii_lowercase().contains("microsoft"))
            .unwrap_or(false)
}

/// If `ORT_DYLIB_PATH` is unset and a libonnxruntime sidecar is bundled
/// next to the executable (the layout the release workflow ships:
/// `<exe-dir>/onnxruntime/`), validate it with the engine's probe and
/// point `ORT_DYLIB_PATH` at it. Silent no-op otherwise — the user gets
/// the same `set ORT_DYLIB_PATH` error from the engine at first ML use
/// as before.
fn discover_ort_sidecar() {
    if std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return;
    }
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let Some(dir) = exe.parent() else {
        return;
    };
    let Some(path) = find_ort_in_dir(&dir.join("onnxruntime")) else {
        return;
    };
    let Some(s) = path.to_str() else { return };
    // SAFETY: called at the top of `main` before any thread is spawned,
    // so there is no concurrent access to the environment.
    unsafe { std::env::set_var("ORT_DYLIB_PATH", s) };
}

/// Looks for a probed-OK ONNX Runtime dylib inside `dir`. The match is
/// platform-specific by filename; the engine's probe then rejects
/// anything that doesn't export `OrtGetApiBase` (filtering out the
/// `libonnxruntime_providers_shared` shim, which is a valid shared object
/// but not the runtime). Returns the first candidate that passes the
/// probe, preferring longer filenames so a versioned name like
/// `libonnxruntime.so.1.26.0` beats a bare `libonnxruntime.so` symlink.
fn find_ort_in_dir(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;

    #[cfg(target_os = "windows")]
    let matches = |name: &str| {
        let lower = name.to_ascii_lowercase();
        lower.starts_with("onnxruntime") && lower.ends_with(".dll")
    };
    #[cfg(target_os = "macos")]
    let matches = |name: &str| name.starts_with("libonnxruntime") && name.ends_with(".dylib");
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let matches = |name: &str| name.starts_with("libonnxruntime.so");

    let mut candidates: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| matches(n) && !n.contains("providers_shared"))
                .unwrap_or(false)
        })
        .collect();
    candidates.sort_by_key(|p| std::cmp::Reverse(p.as_os_str().len()));

    candidates.into_iter().find(|p| {
        p.to_str()
            .is_some_and(|s| sadda_engine::probe_ort_dylib(s).is_ok())
    })
}

// `append_to_inittab!` registers under the wrapper function's name,
// which becomes the Python module name. The actual module body lives
// in `sadda_app.rs`; this wrapper exists purely so the registered
// name is `sadda` (and not `sadda_app`).
#[pymodule]
fn sadda(m: &Bound<'_, PyModule>) -> PyResult<()> {
    sadda_app::sadda(m)
}

/// Top-level app state. `ProjectLoaded` carries the engine handle plus
/// the cached display name (avoids re-querying SQLite every frame).
enum AppState {
    NoProject,
    ProjectLoaded {
        project: Project,
        root: PathBuf,
        name: String,
    },
}

struct SaddaApp {
    app_state: AppState,
    persisted: PersistedState,
    /// Error message currently shown in the bottom banner, or `None`.
    /// Dismissed by user click.
    error: Option<String>,
    /// Currently-selected bundle. `None` means "no bundle selected;
    /// show the central-panel placeholder." Reset on project change.
    selected_bundle_id: Option<i64>,
    /// Cached waveform envelope (and raw mono samples) for the
    /// selected bundle. Rebuilt only when the selection changes.
    active_envelope: Option<EnvelopeCache>,
    /// Cached spectrogram texture for the selected bundle + current
    /// `SpectrogramConfig`. Rebuilt only when either changes.
    active_spectrogram: Option<SpectrogramCache>,
    /// D10: cached measure-track analysis (f0 / formants / intensity)
    /// for the selected bundle + current `MeasureTrackConfig`.
    /// Recomputed only when the bundle or the config changes.
    active_tracks: Option<MeasureTrackCache>,
    /// D10: resolved reference-distribution overlay bands per lane.
    /// Refreshed when the View-menu selection changes.
    overlays: OverlayCache,
    /// D10: resolved data for the right-side Reference panel (vowel-space
    /// scatter + histogram). Refreshed when its selection changes.
    reference: ReferenceView,
    /// Currently-selected annotation in the tier strip. In-memory
    /// only — clears on bundle change. Reached by C5 (cursor sync)
    /// and D6/D7 (editing) when those slices land.
    selected_annotation: Option<AnnotationSelection>,
    /// Cached embedding-heatmap render. `None` when the lane is hidden
    /// (no tier selected) or no bundle is loaded. Rebuilt only when the
    /// bundle, the selected tier, or the colormap/normalization changes.
    active_embedding_heatmap: Option<EmbeddingHeatmapCache>,
    /// Sticky error message from the last attempt to (re)build the
    /// embedding-heatmap cache — e.g. the selected tier no longer exists,
    /// the Parquet sidecar can't be read. Surfaced as a hint inside the
    /// lane so the lane keeps drawing instead of blanking out silently.
    embedding_heatmap_error: Option<String>,
    /// Shared timeline state — cursor, view window, duration —
    /// plumbed into every C5+ pane. Reset on bundle change.
    timeline: TimelineState,
    /// Live playback handle. `Some` while audio is playing; the
    /// app polls `is_finished()` each frame and drops it on
    /// completion (or on a second spacebar press).
    playback: Option<Playback>,
    /// In-progress mouse-driven edit on an interval lane. Drag-create
    /// or drag-resize lives here between mouse-down and mouse-up.
    draft_edit: DraftEdit,
    /// In-progress inline label edit triggered by double-clicking an
    /// interval. Commits on Enter / focus-loss; cancels on Escape.
    label_edit: Option<LabelEdit>,
    /// Open rubric-editor window (slice S1), or `None` when closed.
    rubric_editor: Option<RubricEditor>,
    /// Open criteria-editor window (slice S2), or `None` when closed.
    criteria_editor: Option<CriteriaEditor>,
    /// Open targets panel (slice S4a — campaign work units), or `None`.
    targets_panel: Option<TargetsPanel>,
    /// Open campaign QA dashboard (slice S6a), or `None`.
    dashboard: Option<DashboardWindow>,
    /// Working state for the inline Annotation panel (modal-free editor).
    annotation_inspector: AnnotationInspector,
    /// E8: most recent `script-engine` output (stdout + stderr).
    /// `None` until the user clicks Run for the first time this
    /// session. Not persisted across launches — regenerable cheaply.
    script_output: Option<sadda_script_engine::ScriptOutput>,
    /// E8: error from the most recent script run (e.g. Python
    /// syntax error). Rendered in the output pane alongside stderr.
    script_error: Option<String>,
    /// E9: commands registered via `sadda.app.register_command`.
    /// Lives for the app session. Surfaced in the Ctrl/Cmd+P
    /// palette. Cleared on bundle change? No — commands are global
    /// to the session, not bundle-scoped.
    registered_commands: Vec<(String, Py<PyAny>)>,
    /// E9: whether the command palette is currently visible.
    command_palette_open: bool,
    /// E9: search text in the command palette.
    command_palette_query: String,
    /// H1: informational (non-error) banner — green tone, dismissed
    /// by user click. Used to confirm successful imports / exports.
    info: Option<String>,
    /// H1: live-recording modal state. `Some` while the modal is
    /// open.
    record_dialog: Option<RecordDialog>,
    /// H1: pending bundle-deletion confirmation. `Some` while the
    /// confirm modal is open.
    pending_delete: Option<PendingBundleDelete>,
    /// Pending bundle rename. `Some` while the rename modal is open;
    /// `name` is the editable buffer, seeded with the current name.
    pending_rename: Option<PendingBundleRename>,
    /// Tier-lifecycle modals (create / rename / delete), each `Some`
    /// while its modal is open.
    pending_new_tier: Option<NewTierDraft>,
    pending_tier_rename: Option<PendingTierRename>,
    pending_tier_delete: Option<PendingTierDelete>,
    /// Target tier for span-selection commits (boundaries / points). Set by
    /// clicking a tier's gutter name; highlighted in the strip.
    active_tier_id: Option<i64>,
    /// `(y_axis_gutter_width, data_area_width)` of the signal plots,
    /// measured from the waveform each frame. The painter-based tier lanes
    /// use these *widths* (applied from their own panel's left) so their
    /// time axis aligns with the egui_plot lanes — whose y-axis width is
    /// dynamic. Stored as widths (not an absolute rect) so it's robust to
    /// the tier-strip panel having a different left inset than the waveform.
    lane_geom: Option<(f32, f32)>,
    /// A1: open provenance/citations modal. `Some` holds the snapshot
    /// of the bundle's processing runs + citations, loaded once when
    /// the modal opens.
    provenance: Option<ProvenanceView>,
}

/// A1: a one-shot snapshot of a bundle's provenance timeline and the
/// citations for the analyses it used, rendered in a read-only modal.
struct ProvenanceView {
    name: String,
    runs: Vec<sadda_engine::ProcessingRunRow>,
    citations: Vec<sadda_engine::Citation>,
    /// A3: the bundle's resolved calibration, if any. Drives the
    /// "levels are dB-SPL vs dB-FS" line.
    calibration: Option<sadda_engine::Calibration>,
}

/// H1: identifies a bundle the user has requested be deleted,
/// rendered as a confirmation modal.
struct PendingBundleDelete {
    id: i64,
    name: String,
}

/// Pending "New tier…" modal: a draft tier awaiting name + type.
struct NewTierDraft {
    bundle_id: i64,
    name: String,
    tier_type: TierType,
    just_started: bool,
}

/// Pending tier-rename modal.
struct PendingTierRename {
    id: i64,
    name: String,
    just_started: bool,
}

/// Pending tier-delete confirmation.
struct PendingTierDelete {
    id: i64,
    name: String,
}

/// A tier-lifecycle request raised from the tier strip, applied after the
/// `&project` borrow ends (mirrors the selection/draft snapshot pattern).
enum TierOp {
    New,
    Rename(i64, String),
    Delete(i64, String),
}

/// A bundle the user is renaming via the modal text-edit. `name` is
/// the live edit buffer; `just_started` grabs focus on the first
/// frame.
struct PendingBundleRename {
    id: i64,
    name: String,
    just_started: bool,
}

/// Mouse-driven edit-in-progress for tier lanes. Idle between drags;
/// transitions live in `render_interval_lane` / `render_point_lane`.
#[derive(Debug, Clone)]
enum DraftEdit {
    /// No drag in progress.
    None,
    /// User is dragging in empty interval-lane space to create a new
    /// interval.
    Creating {
        tier_id: i64,
        start_time: f64,
        current_time: f64,
    },
    /// User is dragging an interval's start or end edge to resize it.
    Resizing {
        tier_id: i64,
        annotation_id: i64,
        edge: BoundaryEdge,
        /// The *other* edge, held fixed during the resize.
        fixed_time: f64,
        current_time: f64,
    },
    /// User is dragging an existing point tick to move it (D7).
    MovingPoint {
        tier_id: i64,
        annotation_id: i64,
        original_time: f64,
        current_time: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundaryEdge {
    Start,
    End,
}

/// Inline label-edit state. Created on double-click; drained on
/// commit / cancel. Polymorphic over interval / point lanes: the
/// `kind` tells the commit-time handler which engine `update_*`
/// method to call; the base row is re-fetched at commit so
/// non-label fields aren't snapshotted into a stale buffer.
#[derive(Default)]
struct LabelEdit {
    tier_id: i64,
    annotation_id: i64,
    kind: LabelEditKind,
    /// The label text being edited.
    text: String,
    /// Working copy of the annotation's status (a rubric status value), or
    /// `None`. Populated from the row when the edit opens.
    status: Option<String>,
    /// Working copy of the free-text note.
    note: String,
    /// Set to `true` for the first frame so the label field grabs
    /// focus; cleared after.
    just_started: bool,
    /// Guards the one-time fetch of the rubric context below (the free
    /// lane-render fns that create a `LabelEdit` have no project handle, so
    /// the window fills these on its first frame).
    loaded: bool,
    /// The tier's controlled vocabulary (allowed label values).
    vocab: Vec<String>,
    /// Whether the tier's vocabulary is closed (out-of-vocab rejected).
    closed: bool,
    /// The rubric's status vocabulary (options for the status picker).
    statuses: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default)]
enum LabelEditKind {
    #[default]
    Interval,
    Point,
}

/// State for the rubric-editor window (slice S1): a working copy of the
/// project's annotation rubric — name, guidelines, the status vocabulary,
/// and one tier's controlled vocabulary — that the user edits and commits
/// with "Apply". Loaded from the project once when the window opens.
#[derive(Default)]
struct RubricEditor {
    /// Guards the one-time load from the project.
    loaded: bool,
    name: String,
    /// Preserved across edits (full version history is a later slice).
    version: i64,
    guidelines: String,
    /// Editable status rows: (value, description).
    statuses: Vec<(String, String)>,
    /// The tier name whose controlled vocabulary is being edited.
    tier_name: Option<String>,
    /// Whether the selected tier's vocabulary is closed.
    tier_closed: bool,
    /// Editable controlled-vocabulary rows for the selected tier:
    /// (value, description).
    vocab: Vec<(String, String)>,
}

/// State for the criteria-editor window (slice S2): browse/edit criteria and
/// run them to produce proposals on a preview tier, then accept or reject.
#[derive(Default)]
struct CriteriaEditor {
    /// Guards the one-time load of the criteria list.
    loaded: bool,
    /// Existing criteria as (id, name) for the list.
    list: Vec<(i64, String)>,
    /// Id of the criterion being edited; `None` while drafting a new one.
    selected: Option<i64>,
    /// Working-copy fields.
    name: String,
    /// `"structured"` or `"python"`.
    kind: String,
    body: String,
    target_tier: String,
    description: String,
    /// Last run/accept/reject result, shown in the window.
    status_msg: Option<String>,
}

/// State for the targets panel (slice S4a): the campaign work-unit list for the
/// selected bundle, plus the small "generate from a criterion" and "add manual
/// target" affordances. Targets are read live from the project each frame, so
/// the panel only holds draft inputs and the last status line.
#[derive(Default)]
struct TargetsPanel {
    /// Criterion chosen for "Generate targets from criterion" as `(id, name)`.
    gen_criterion: Option<(i64, String)>,
    /// Draft fields for "Add manual target".
    new_start: String,
    new_end: String,
    new_type: String,
    /// S4b — annotator typed in for the per-target "Assign" button.
    assign_annotator: String,
    /// S4b — comma-separated roster + seed for "Assign randomly".
    roster: String,
    seed: String,
    /// S4c — merge_tiers inputs: comma-separated source tier names + destination.
    merge_sources: String,
    merge_dest: String,
    /// S5 — the two tiers selected for the agreement comparison, as `(id, name)`.
    compare_a: Option<(i64, String)>,
    compare_b: Option<(i64, String)>,
    /// Last action result, shown in the panel.
    status_msg: Option<String>,
}

/// State for the campaign QA dashboard (slice S6a): overall + per-annotator
/// completeness (read live), plus on-demand per-tier QA and inter-annotator
/// agreement. Holds only the draft inputs and the last computed result lines.
#[derive(Default)]
struct DashboardWindow {
    /// Tier selected for the QA check, as `(id, name)`.
    qa_tier: Option<(i64, String)>,
    /// Base tier name for the agreement summary (e.g. `phones`).
    agreement_base: String,
    /// Last QA result line.
    qa_msg: Option<String>,
    /// Last agreement-summary result lines.
    agreement_msgs: Vec<String>,
    /// S6b — note for the next published rubric version + impact version input.
    publish_note: String,
    impact_version: String,
    /// Last versioning result line + impact lines.
    version_msg: Option<String>,
    impact_msgs: Vec<String>,
}

/// Working state for the inline Annotation panel (the modal-free editor): a
/// live working copy of the currently-selected annotation's label / status /
/// note, plus the rubric context for that tier. Reloaded whenever the
/// selection changes; edits apply to the project on commit (Enter / focus
/// loss / status pick / Apply).
#[derive(Default)]
struct AnnotationInspector {
    /// The annotation the working copy was loaded for; used to detect a
    /// changed selection and reload.
    loaded_for: Option<AnnotationSelection>,
    tier_name: String,
    kind: LabelEditKind,
    label: String,
    status: Option<String>,
    note: String,
    /// The tier's controlled vocabulary (for suggestion chips + OOV flag).
    vocab: Vec<String>,
    /// Whether the tier's vocabulary is closed.
    closed: bool,
    /// The rubric's status vocabulary (status-picker options).
    statuses: Vec<String>,
    /// Provenance one-liner when the annotation came from a criterion run
    /// (its `processing_run_id` resolved to a `criterion_run`), else `None`.
    provenance: Option<String>,
}

impl CriteriaEditor {
    /// Resets the working-copy fields to a blank new criterion.
    fn reset_to_new(&mut self) {
        self.selected = None;
        self.name = String::new();
        self.kind = "structured".into();
        self.body =
            "{\n  \"select\": {\"tier\": \"\"},\n  \"emit\": {\"kind\": \"span\"}\n}".into();
        self.target_tier = String::new();
        self.description = String::new();
        self.status_msg = None;
    }
}

/// H1: live-recording modal. State machine: Idle (configuring the
/// session) → Recording (cpal stream running, meter updating) →
/// Stopped (awaiting Save or Discard). On Save, commits to a bundle
/// and selects it; on Discard or cancel, drops the in-progress
/// directory.
struct RecordDialog {
    /// cpal device label. `"default"` resolves at start() to the
    /// host's default input device.
    device: String,
    /// Available device labels for the picker. Populated once at
    /// dialog construction.
    device_options: Vec<String>,
    /// Capture sample rate in Hz.
    sample_rate: u32,
    /// 1 (mono) or 2 (stereo). DSP path always runs on the
    /// downmixed-to-mono signal.
    channels: u16,
    /// Bundle name the recording will be committed as.
    name: String,
    /// Current state of the dialog state machine.
    state: RecordDialogState,
    /// Live peak dB-FS for the meter (Recording state only).
    meter_db: f32,
    /// Seconds since the recording started (Recording state only).
    elapsed_seconds: f64,
    /// Sticky status message rendered below the action buttons.
    status: String,
}

#[allow(clippy::large_enum_variant)]
enum RecordDialogState {
    Idle,
    Recording(RecordingHandle),
    Stopped {
        stopped: StoppedSession,
        duration: f64,
        dropped: usize,
    },
}

struct RecordingHandle {
    /// Engine session — owns the WAV writer + consumer thread.
    engine: LiveSession,
    /// Result rings drained each frame; otherwise the engine's
    /// consumer thread would back-pressure on full rings and start
    /// dropping DSP frames.
    results: LiveResults,
    /// Holds the cpal stream alive on a dedicated thread (`cpal::Stream`
    /// is `!Send` on Linux ALSA). Drop = stop signal + join.
    cpal: CpalStreamHandle,
    /// Wall-clock start instant — drives the elapsed-seconds display.
    started_at: std::time::Instant,
}

/// Owns a `cpal::Stream` on a dedicated thread. The stream is
/// `!Send` on Linux ALSA, so we cannot move it into the GUI's
/// `App` field without contaminating eframe's threading. Dropping
/// this handle stops the stream and joins the thread.
struct CpalStreamHandle {
    stop_tx: std::sync::mpsc::Sender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for CpalStreamHandle {
    fn drop(&mut self) {
        let _ = self.stop_tx.send(());
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl RecordDialog {
    fn new() -> Self {
        let device_options = enumerate_input_devices();
        let default_device = default_input_device_label().unwrap_or_else(|| "default".into());
        Self {
            device: default_device,
            device_options,
            sample_rate: 44_100,
            channels: 1,
            name: default_recording_name(),
            state: RecordDialogState::Idle,
            meter_db: -120.0,
            elapsed_seconds: 0.0,
            status: String::new(),
        }
    }
}

fn default_recording_name() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("recording_{secs}")
}

fn enumerate_input_devices() -> Vec<String> {
    use cpal::traits::HostTrait;
    let host = cpal::default_host();
    let mut out = vec!["default".to_string()];
    if let Ok(devices) = host.input_devices() {
        for d in devices {
            if let Ok(n) = cpal_device_name(&d) {
                out.push(n);
            }
        }
    }
    out
}

fn default_input_device_label() -> Option<String> {
    use cpal::traits::HostTrait;
    cpal::default_host()
        .default_input_device()
        .and_then(|d| cpal_device_name(&d).ok())
}

#[allow(deprecated)]
fn cpal_device_name(d: &cpal::Device) -> std::result::Result<String, cpal::DeviceNameError> {
    use cpal::traits::DeviceTrait;
    d.name()
}

/// Pending action for the recording modal's render closure. Captured
/// inside the egui closure (which borrows `dialog` mutably) and
/// dispatched afterwards against `&mut self` to avoid a double
/// borrow.
#[derive(Debug, Clone, Copy)]
enum RecordDialogAction {
    Start,
    Stop,
    Commit,
    Discard,
    Close,
}

/// Builds the cpal input stream against `device_label` and pushes
/// captured samples into `producer`. The stream lives on a
/// dedicated thread (cpal::Stream is `!Send` on Linux ALSA).
fn spawn_cpal_input(
    device_label: &str,
    cfg: &LiveConfig,
    mut producer: rtrb::Producer<f32>,
) -> std::result::Result<CpalStreamHandle, String> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let host = cpal::default_host();
    let device = if device_label == "default" {
        host.default_input_device()
            .ok_or_else(|| "no default input device".to_string())?
    } else {
        let mut found = None;
        let devices = host
            .input_devices()
            .map_err(|e| format!("enumerating input devices: {e}"))?;
        for d in devices {
            if cpal_device_name(&d).ok().as_deref() == Some(device_label) {
                found = Some(d);
                break;
            }
        }
        found.ok_or_else(|| format!("input device not found: {device_label}"))?
    };
    let stream_cfg = cpal::StreamConfig {
        channels: cfg.channels,
        sample_rate: cfg.sample_rate,
        buffer_size: cpal::BufferSize::Default,
    };
    let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
    let thread = std::thread::Builder::new()
        .name("sadda-app-cpal".into())
        .spawn(move || {
            let stream = match device.build_input_stream::<f32, _, _>(
                &stream_cfg,
                move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                    for &s in data {
                        let _ = producer.push(s);
                    }
                },
                move |err| eprintln!("sadda-app cpal error: {err}"),
                None,
            ) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("sadda-app: build_input_stream failed: {e}");
                    return;
                }
            };
            use cpal::traits::StreamTrait;
            if let Err(e) = stream.play() {
                eprintln!("sadda-app: stream.play() failed: {e}");
                return;
            }
            let _ = stop_rx.recv();
            drop(stream);
        })
        .map_err(|e| format!("spawn cpal thread: {e}"))?;
    Ok(CpalStreamHandle {
        stop_tx,
        thread: Some(thread),
    })
}

/// Open `path` in the OS's native file manager. Best-effort: failure
/// is silently ignored on the assumption the user can navigate to
/// the path manually using the message they were just shown.
fn open_in_file_manager(path: &std::path::Path) {
    #[cfg(target_os = "macos")]
    let result = std::process::Command::new("open").arg(path).spawn();
    #[cfg(target_os = "windows")]
    let result = std::process::Command::new("explorer").arg(path).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let result = std::process::Command::new("xdg-open").arg(path).spawn();
    let _ = result;
}

/// Identifies the currently-selected annotation in the tier strip.
/// Reference selection is omitted in B4 (reference lanes don't have
/// time-positioned content yet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnnotationSelection {
    Interval { tier_id: i64, annotation_id: i64 },
    Point { tier_id: i64, annotation_id: i64 },
}

/// Cached spectrogram render. Invalidates whenever the bundle or the
/// DSP config changes.
struct SpectrogramCache {
    bundle_id: i64,
    config: SpectrogramConfig,
    /// GPU texture handle for the colormapped image. Egui keeps the
    /// texture alive for the lifetime of this handle.
    texture: egui::TextureHandle,
    /// Echoed from the source bundle for the x-axis bounds.
    duration_seconds: f64,
    /// Echoed from the source bundle for the y-axis bounds.
    nyquist_hz: f32,
}

/// Cached embedding-heatmap render. Mirrors [`SpectrogramCache`]: a
/// baked egui texture for the colormapped matrix plus the metadata the
/// lane needs to lay it out (duration, dim count) and the config it was
/// built from (so a colormap / normalization / tier-selection change
/// invalidates the cache cheaply). The original matrix is *not* kept —
/// re-reading the Parquet sidecar on a rebuild is cheap and saves the
/// MB-scale memory cost.
struct EmbeddingHeatmapCache {
    bundle_id: i64,
    config: EmbeddingHeatmapConfig,
    /// GPU texture handle for the colormapped matrix.
    texture: egui::TextureHandle,
    /// Bundle duration in seconds (x-axis upper bound).
    duration_seconds: f64,
    /// Embedding dimensionality (y-axis upper bound — `[0, n_dims)`).
    n_dims: usize,
    /// User-facing tier name, surfaced in the lane caption.
    tier_name: String,
}

/// D10: cached measure-track analysis for the selected bundle under
/// the current [`MeasureTrackConfig`]. Mirrors [`SpectrogramCache`]:
/// recomputed only when the bundle or the config changes (see
/// `rebuild_tracks_if_stale`). Holds the raw per-frame engine output;
/// the lane panes read straight from these vectors each frame.
struct MeasureTrackCache {
    bundle_id: i64,
    config: MeasureTrackConfig,
    /// f0 estimates over the whole bundle. Every frame is retained
    /// (the tracker emits a frequency for all frames); frames below
    /// the config's voicing threshold are filtered at draw time so an
    /// unvoiced stretch leaves a gap rather than a spurious contour.
    f0: Vec<PitchFrame>,
    /// Per-frame formant estimates (ascending F1..Fn).
    formants: Vec<FormantFrame>,
    /// Per-frame intensity (dB-FS).
    intensity: Vec<IntensityFrame>,
    /// Per-window VAD speech probabilities (empty if the lane is hidden
    /// or VAD failed — see `vad_error`).
    vad: Vec<VadFrame>,
    /// Set when the VAD lane is visible but inference failed (e.g. ONNX
    /// Runtime not available); rendered as a hint in the lane.
    vad_error: Option<String>,
}

/// D10: a resolved reference-distribution band, ready to draw on a lane.
/// The `kind` drives the visual encoding so a normative band and a
/// target zone never look alike (the 2026-05-18 governance rule that the
/// GUI must not conflate "what people do" with "what to aim for").
struct OverlayBand {
    /// Distribution summary the band geometry comes from.
    summary: Summary,
    /// Observed vs normative vs target — selects the encoding.
    kind: MeasureKind,
    /// Short label (the subgroup-qualified distribution title) drawn in
    /// the lane corner.
    label: String,
}

/// D10: per-lane cache of the resolved overlay band. Each entry holds the
/// selection that produced it plus the band (`None` if the distribution
/// couldn't be resolved / summarised), so the store read only happens
/// when the selection changes — not every frame.
#[derive(Default)]
struct OverlayCache {
    f0: Option<(RefdistOverlay, Option<OverlayBand>)>,
    intensity: Option<(RefdistOverlay, Option<OverlayBand>)>,
}

/// D10: resolved data for the right-side Reference panel — the vowel-space
/// cloud and the 1-D histogram + summary for the active parameter. Cached
/// against the (distribution, phone, param) selection so the parquet read
/// happens on a selection change, not every frame (matters during
/// continuous-repaint playback).
#[derive(Default)]
struct ReferenceView {
    /// Selection that produced this view: (distribution, phone, param).
    key: Option<(RefdistOverlay, Option<String>, Option<String>)>,
    /// Subgroup-qualified distribution title.
    title: String,
    /// Distribution kind (drives the histogram/scatter framing).
    kind: Option<MeasureKind>,
    /// The distribution's declared parameters (vowel space uses [0], [1]).
    params: Vec<String>,
    /// Declared phones, offered as a vowel-space filter.
    phones: Vec<String>,
    /// Vowel-space cloud: `[params[0], params[1]]` pairs (empty for a
    /// <2-parameter or summary-only distribution).
    cloud: Vec<[f64; 2]>,
    /// Histogram of the active parameter (observed/target only).
    histogram: Option<Histogram>,
    /// Summary of the active parameter (percentile markers).
    summary: Option<Summary>,
    /// The parameter `histogram` / `summary` describe.
    active_param: Option<String>,
}

impl SaddaApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let persisted: PersistedState = cc
            .storage
            .and_then(|s| eframe::get_value::<PersistedState>(s, eframe::APP_KEY))
            .unwrap_or_default();
        Self {
            app_state: AppState::NoProject,
            persisted,
            error: None,
            selected_bundle_id: None,
            active_envelope: None,
            active_spectrogram: None,
            active_tracks: None,
            overlays: OverlayCache::default(),
            reference: ReferenceView::default(),
            selected_annotation: None,
            active_embedding_heatmap: None,
            embedding_heatmap_error: None,
            timeline: TimelineState::default(),
            playback: None,
            draft_edit: DraftEdit::None,
            label_edit: None,
            rubric_editor: None,
            criteria_editor: None,
            targets_panel: None,
            dashboard: None,
            annotation_inspector: AnnotationInspector::default(),
            script_output: None,
            script_error: None,
            registered_commands: Vec::new(),
            command_palette_open: false,
            command_palette_query: String::new(),
            info: None,
            record_dialog: None,
            pending_delete: None,
            pending_rename: None,
            pending_new_tier: None,
            pending_tier_rename: None,
            pending_tier_delete: None,
            active_tier_id: None,
            lane_geom: None,
            provenance: None,
        }
    }

    // ----- Mutators that change AppState -------------------------------

    fn open_project_at(&mut self, path: PathBuf) {
        match Project::open(&path) {
            Ok(project) => {
                let name = project.name().unwrap_or_else(|_| "(unknown)".to_string());
                self.persisted.record_open(&path);
                self.app_state = AppState::ProjectLoaded {
                    project,
                    root: path,
                    name,
                };
                self.clear_bundle_selection();
                self.error = None;
            }
            Err(e) => self.set_error(format!("Failed to open project: {e}")),
        }
    }

    fn create_project_at(&mut self, path: PathBuf) {
        let name = project_name_from_path(&path);
        match Project::create(&path, &name) {
            Ok(project) => {
                self.persisted.record_open(&path);
                self.app_state = AppState::ProjectLoaded {
                    project,
                    root: path,
                    name,
                };
                self.clear_bundle_selection();
                self.error = None;
            }
            Err(e) => self.set_error(format!("Failed to create project: {e}")),
        }
    }

    fn add_bundle_from_wav(&mut self, wav_path: PathBuf) {
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        let bundle_name = wav_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "bundle".into());
        match project.add_bundle(&bundle_name, &wav_path) {
            Ok(id) => {
                self.error = None;
                self.select_bundle(id);
            }
            Err(e) => self.set_error(format!("Failed to add bundle: {e}")),
        }
    }

    // ----- H1 Import / Export helpers ---------------------------------

    fn import_textgrid_for_active_bundle(&mut self, path: PathBuf) {
        let Some(bundle_id) = self.selected_bundle_id else {
            return;
        };
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        match project.import_textgrid(&path, bundle_id) {
            Ok(tier_ids) => {
                self.error = None;
                self.set_info(format!(
                    "Imported {} tier{} from {}",
                    tier_ids.len(),
                    if tier_ids.len() == 1 { "" } else { "s" },
                    path.display(),
                ));
            }
            Err(e) => self.set_error(format!("TextGrid import failed: {e}")),
        }
    }

    fn import_eaf_for_active_bundle(&mut self, path: PathBuf) {
        let Some(bundle_id) = self.selected_bundle_id else {
            return;
        };
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        match project.import_eaf(&path, bundle_id) {
            Ok(tier_ids) => {
                self.error = None;
                self.set_info(format!(
                    "Imported {} tier{} from {}",
                    tier_ids.len(),
                    if tier_ids.len() == 1 { "" } else { "s" },
                    path.display(),
                ));
            }
            Err(e) => self.set_error(format!("EAF import failed: {e}")),
        }
    }

    fn export_textgrid_for_active_bundle(&mut self, path: PathBuf) {
        let Some(bundle_id) = self.selected_bundle_id else {
            return;
        };
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        match project.export_textgrid(bundle_id, &path, None) {
            Ok(()) => {
                self.error = None;
                self.set_info(format!("Wrote TextGrid to {}", path.display()));
            }
            Err(e) => self.set_error(format!("TextGrid export failed: {e}")),
        }
    }

    fn export_eaf_for_active_bundle(&mut self, path: PathBuf) {
        let Some(bundle_id) = self.selected_bundle_id else {
            return;
        };
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        match project.export_eaf(bundle_id, &path, None) {
            Ok(()) => {
                self.error = None;
                self.set_info(format!("Wrote EAF to {}", path.display()));
            }
            Err(e) => self.set_error(format!("EAF export failed: {e}")),
        }
    }

    /// Pops a save-file dialog defaulting to the project's
    /// `exports/` directory + the active bundle's name + the
    /// requested extension. Returns `None` if the user cancels.
    fn suggest_export_path(&self, extension: &str) -> Option<PathBuf> {
        let (root, bundle_id) = match &self.app_state {
            AppState::ProjectLoaded { root, .. } => (root.clone(), self.selected_bundle_id?),
            AppState::NoProject => return None,
        };
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return None;
        };
        let bundle_name = project
            .bundles()
            .ok()?
            .into_iter()
            .find(|b| b.id == bundle_id)?
            .name;
        let exports = root.join("exports");
        let _ = std::fs::create_dir_all(&exports);
        let filename = format!("{bundle_name}.{extension}");
        rfd::FileDialog::new()
            .set_directory(&exports)
            .set_file_name(&filename)
            .save_file()
    }

    /// Opens the project root in the OS's native file manager.
    fn show_project_folder(&mut self) {
        let AppState::ProjectLoaded { root, .. } = &self.app_state else {
            return;
        };
        open_in_file_manager(root);
    }

    /// Opens the parent directory of a bundle's WAV file in the OS
    /// file manager.
    fn reveal_bundle(&mut self, audio_rel: &str) {
        let AppState::ProjectLoaded { root, .. } = &self.app_state else {
            return;
        };
        let abs = root.join(audio_rel);
        let parent = abs.parent().unwrap_or(root.as_path());
        open_in_file_manager(parent);
    }

    /// Executes the pending bundle deletion. Clears the selection
    /// if the deleted bundle was active; refreshes the bundle list
    /// on the next frame.
    fn confirm_pending_delete(&mut self) {
        let Some(pending) = self.pending_delete.take() else {
            return;
        };
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        match project.delete_bundle(pending.id) {
            Ok(()) => {
                self.error = None;
                if self.selected_bundle_id == Some(pending.id) {
                    self.selected_bundle_id = None;
                    self.active_envelope = None;
                    self.active_spectrogram = None;
                    self.active_embedding_heatmap = None;
                    self.embedding_heatmap_error = None;
                    self.selected_annotation = None;
                    self.timeline = TimelineState::default();
                    self.playback = None;
                }
                self.set_info(format!("Deleted bundle “{}”.", pending.name));
            }
            Err(e) => self.set_error(format!("Delete failed: {e}")),
        }
    }

    /// Renders the bundle-delete confirmation modal when one is
    /// pending.
    fn render_pending_delete(&mut self, ctx: &egui::Context) {
        let Some(pending) = self.pending_delete.as_ref() else {
            return;
        };
        let name = pending.name.clone();
        let mut action: Option<bool> = None; // true = confirm, false = cancel
        let mut is_open = true;
        egui::Window::new("Delete bundle?")
            .open(&mut is_open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!(
                    "Permanently delete bundle “{name}” and all its tiers, \
                     annotations, and derived signals?"
                ));
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("The recorded WAV will also be removed from disk.").weak(),
                );
                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .button(
                            egui::RichText::new("Delete")
                                .color(egui::Color32::from_rgb(220, 80, 80)),
                        )
                        .clicked()
                    {
                        action = Some(true);
                    }
                    if ui.button("Cancel").clicked() {
                        action = Some(false);
                    }
                });
            });
        if !is_open {
            action = Some(false);
        }
        match action {
            Some(true) => self.confirm_pending_delete(),
            Some(false) => {
                self.pending_delete = None;
            }
            None => {}
        }
    }

    /// Renders the bundle-rename modal when one is pending. Commits on
    /// Enter / Save; cancels on Escape / Cancel / window close. Save is
    /// disabled while the (trimmed) name is empty.
    fn render_pending_rename(&mut self, ctx: &egui::Context) {
        if self.pending_rename.is_none() {
            return;
        }
        let mut commit = false;
        let mut cancel = false;
        let mut keep_open = true;
        if let Some(pending) = self.pending_rename.as_mut() {
            egui::Window::new("Rename bundle")
                .collapsible(false)
                .resizable(false)
                .open(&mut keep_open)
                .show(ctx, |ui| {
                    let resp =
                        ui.add(egui::TextEdit::singleline(&mut pending.name).desired_width(260.0));
                    if pending.just_started {
                        resp.request_focus();
                        pending.just_started = false;
                    }
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        commit = true;
                    }
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        let can_save = !pending.name.trim().is_empty();
                        if ui
                            .add_enabled(can_save, egui::Button::new("Save"))
                            .clicked()
                        {
                            commit = true;
                        }
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }
                    });
                });
        }
        if !keep_open {
            cancel = true;
        }
        // Escape cancels even when the text field isn't focused; consume
        // it so it doesn't also reach the tier-editing key handlers.
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            cancel = true;
        }
        if commit {
            self.commit_pending_rename();
        } else if cancel {
            self.pending_rename = None;
        }
    }

    /// Applies the pending rename via the engine. Closes the modal on
    /// success; on failure surfaces the error and leaves the modal open
    /// for a retry. The sidebar re-queries `bundles()` each frame, so
    /// the new name appears without any cached-name invalidation.
    fn commit_pending_rename(&mut self) {
        let Some(pending) = self.pending_rename.as_ref() else {
            return;
        };
        let id = pending.id;
        let new_name = pending.name.trim().to_string();
        if new_name.is_empty() {
            return;
        }
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        match project.rename_bundle(id, &new_name) {
            Ok(()) => {
                self.error = None;
                self.pending_rename = None;
                self.set_info(format!("Renamed bundle to “{new_name}”."));
            }
            Err(e) => self.set_error(format!("Rename failed: {e}")),
        }
    }

    /// "New tier…" modal: name field + type picker (interval / point /
    /// reference). Creates the tier on the active bundle via the engine.
    fn render_new_tier(&mut self, ctx: &egui::Context) {
        if self.pending_new_tier.is_none() {
            return;
        }
        let mut create = false;
        let mut cancel = false;
        let mut keep_open = true;
        if let Some(draft) = self.pending_new_tier.as_mut() {
            egui::Window::new("New tier")
                .collapsible(false)
                .resizable(false)
                .open(&mut keep_open)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        let resp = ui
                            .add(egui::TextEdit::singleline(&mut draft.name).desired_width(220.0));
                        if draft.just_started {
                            resp.request_focus();
                            draft.just_started = false;
                        }
                        if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            create = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Type:");
                        ui.selectable_value(&mut draft.tier_type, TierType::Interval, "Interval");
                        ui.selectable_value(&mut draft.tier_type, TierType::Point, "Point");
                        ui.selectable_value(&mut draft.tier_type, TierType::Reference, "Reference");
                    });
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        let can_create = !draft.name.trim().is_empty();
                        if ui
                            .add_enabled(can_create, egui::Button::new("Create"))
                            .clicked()
                        {
                            create = true;
                        }
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }
                    });
                });
        }
        if !keep_open {
            cancel = true;
        }
        if create {
            self.commit_new_tier();
        } else if cancel {
            self.pending_new_tier = None;
        }
    }

    fn commit_new_tier(&mut self) {
        let Some(draft) = self.pending_new_tier.as_ref() else {
            return;
        };
        let (bundle_id, name, tier_type) = (
            draft.bundle_id,
            draft.name.trim().to_string(),
            draft.tier_type,
        );
        if name.is_empty() {
            return;
        }
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        match project.add_tier(&sadda_engine::TierSpec::new(bundle_id, &name, tier_type)) {
            Ok(_) => {
                self.error = None;
                self.pending_new_tier = None;
                self.set_info(format!("Created tier “{name}”."));
            }
            Err(e) => self.set_error(format!("Create tier failed: {e}")),
        }
    }

    /// Tier-rename modal (mirrors the bundle rename).
    fn render_pending_tier_rename(&mut self, ctx: &egui::Context) {
        if self.pending_tier_rename.is_none() {
            return;
        }
        let mut commit = false;
        let mut cancel = false;
        let mut keep_open = true;
        if let Some(pending) = self.pending_tier_rename.as_mut() {
            egui::Window::new("Rename tier")
                .collapsible(false)
                .resizable(false)
                .open(&mut keep_open)
                .show(ctx, |ui| {
                    let resp =
                        ui.add(egui::TextEdit::singleline(&mut pending.name).desired_width(260.0));
                    if pending.just_started {
                        resp.request_focus();
                        pending.just_started = false;
                    }
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        commit = true;
                    }
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        let can_save = !pending.name.trim().is_empty();
                        if ui
                            .add_enabled(can_save, egui::Button::new("Save"))
                            .clicked()
                        {
                            commit = true;
                        }
                        if ui.button("Cancel").clicked() {
                            cancel = true;
                        }
                    });
                });
        }
        if !keep_open {
            cancel = true;
        }
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            cancel = true;
        }
        if commit {
            let Some(pending) = self.pending_tier_rename.as_ref() else {
                return;
            };
            let (id, new_name) = (pending.id, pending.name.trim().to_string());
            if new_name.is_empty() {
                return;
            }
            let AppState::ProjectLoaded { project, .. } = &self.app_state else {
                return;
            };
            match project.rename_tier(id, &new_name) {
                Ok(()) => {
                    self.error = None;
                    self.pending_tier_rename = None;
                    self.set_info(format!("Renamed tier to “{new_name}”."));
                }
                Err(e) => self.set_error(format!("Rename tier failed: {e}")),
            }
        } else if cancel {
            self.pending_tier_rename = None;
        }
    }

    /// Tier-delete confirmation (mirrors the bundle delete).
    fn render_pending_tier_delete(&mut self, ctx: &egui::Context) {
        let Some(pending) = self.pending_tier_delete.as_ref() else {
            return;
        };
        let (id, name) = (pending.id, pending.name.clone());
        let mut action: Option<bool> = None;
        let mut is_open = true;
        egui::Window::new("Delete tier?")
            .open(&mut is_open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(format!(
                    "Permanently delete tier “{name}” and all of its annotations?"
                ));
                ui.separator();
                ui.horizontal(|ui| {
                    if ui
                        .button(
                            egui::RichText::new("Delete")
                                .color(egui::Color32::from_rgb(220, 80, 80)),
                        )
                        .clicked()
                    {
                        action = Some(true);
                    }
                    if ui.button("Cancel").clicked() {
                        action = Some(false);
                    }
                });
            });
        if !is_open {
            action = Some(false);
        }
        match action {
            Some(true) => {
                if let AppState::ProjectLoaded { project, .. } = &self.app_state {
                    match project.delete_tier(id) {
                        Ok(()) => {
                            self.error = None;
                            // Clear any selection/draft/active-tier that
                            // referenced the gone tier.
                            self.selected_annotation = None;
                            self.draft_edit = DraftEdit::None;
                            if self.active_tier_id == Some(id) {
                                self.active_tier_id = None;
                            }
                            self.set_info(format!("Deleted tier “{name}”."));
                        }
                        Err(e) => self.set_error(format!("Delete tier failed: {e}")),
                    }
                }
                self.pending_tier_delete = None;
            }
            Some(false) => self.pending_tier_delete = None,
            None => {}
        }
    }

    /// Commits the current time-span selection to the active tier: an
    /// interval `[lo, hi]` (interval tier) or two boundary points at the
    /// edges (point tier). Keeps the selection so it can be placed on
    /// several tiers in turn.
    fn commit_selection_to_tier(&mut self, tier_id: i64, tier_type: TierType, lo: f64, hi: f64) {
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        let result = match tier_type {
            TierType::Interval => project
                .add_interval(&sadda_engine::IntervalSpec {
                    tier_id,
                    start_seconds: lo,
                    end_seconds: hi,
                    label: None,
                    parent_annotation_id: None,
                    status: None,
                    note: None,
                    extra: None,
                    ..Default::default()
                })
                .map(|_| ()),
            TierType::Point => project
                .add_point(&sadda_engine::PointSpec {
                    tier_id,
                    time_seconds: lo,
                    label: None,
                    parent_annotation_id: None,
                    status: None,
                    note: None,
                    extra: None,
                    ..Default::default()
                })
                .and_then(|_| {
                    project.add_point(&sadda_engine::PointSpec {
                        tier_id,
                        time_seconds: hi,
                        label: None,
                        parent_annotation_id: None,
                        status: None,
                        note: None,
                        extra: None,
                        ..Default::default()
                    })
                })
                .map(|_| ()),
            _ => return,
        };
        match result {
            Ok(()) => {
                self.error = None;
                self.set_info("Added annotation from selection.".to_string());
            }
            Err(e) => self.set_error(format!("Add from selection failed: {e}")),
        }
    }

    /// A1: loads a bundle's provenance timeline + citations into a
    /// modal snapshot. One-shot query; the modal renders the snapshot
    /// without re-querying each frame.
    fn open_provenance_view(&mut self, bundle_id: i64, name: String) {
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        let runs = match project.processing_runs(bundle_id) {
            Ok(r) => r,
            Err(e) => return self.set_error(format!("Failed to load provenance: {e}")),
        };
        let citations = match project.citations(bundle_id) {
            Ok(c) => c,
            Err(e) => return self.set_error(format!("Failed to load citations: {e}")),
        };
        // A3: a failed calibration lookup is non-fatal — treat as
        // uncalibrated rather than blocking the whole view.
        let calibration = project.bundle_calibration(bundle_id).ok().flatten();
        self.provenance = Some(ProvenanceView {
            name,
            runs,
            citations,
            calibration,
        });
    }

    /// A1: renders the provenance/citations modal (read-only). Lists the
    /// bundle's processing runs and the deduplicated citation list, with
    /// a button to copy the references to the clipboard.
    fn render_provenance_view(&mut self, ctx: &egui::Context) {
        let Some(view) = self.provenance.as_ref() else {
            return;
        };
        let mut keep_open = true;
        let mut copy_citations = false;
        egui::Window::new(format!("Provenance — {}", view.name))
            .open(&mut keep_open)
            .collapsible(false)
            .resizable(true)
            .default_width(560.0)
            .show(ctx, |ui| {
                // A3: calibration status — are levels absolute dB-SPL or
                // relative dB-FS?
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Levels:").strong());
                    match &view.calibration {
                        Some(c) => ui.label(format!(
                            "calibrated — dB-SPL (+{:.1} dB from dB-FS)",
                            c.spl_offset_db()
                        )),
                        None => ui.label(egui::RichText::new("uncalibrated — dB-FS only").weak()),
                    };
                });
                ui.separator();

                ui.label(egui::RichText::new("Processing runs").strong());
                if view.runs.is_empty() {
                    ui.label(egui::RichText::new("(no recorded analyses yet)").weak());
                } else {
                    egui::ScrollArea::vertical()
                        .max_height(220.0)
                        .show(ui, |ui| {
                            for r in &view.runs {
                                let ts = r.finished_at.as_deref().unwrap_or(&r.started_at);
                                ui.label(format!(
                                    "{} · {} · {} · {}",
                                    ts, r.kind, r.processor_id, r.status
                                ));
                            }
                        });
                }

                ui.add_space(8.0);
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Citations").strong());
                    if !view.citations.is_empty() && ui.button("Copy references").clicked() {
                        copy_citations = true;
                    }
                });
                if view.citations.is_empty() {
                    ui.label(egui::RichText::new("(no citeable analyses on this bundle)").weak());
                } else {
                    for c in &view.citations {
                        ui.label(&c.reference);
                        if let Some(doi) = &c.doi {
                            ui.hyperlink_to(format!("doi:{doi}"), format!("https://doi.org/{doi}"));
                        }
                        ui.add_space(4.0);
                    }
                }
            });

        if copy_citations {
            let text = view
                .citations
                .iter()
                .map(|c| match &c.doi {
                    Some(doi) => format!("{} https://doi.org/{doi}", c.reference),
                    None => c.reference.clone(),
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            ctx.copy_text(text);
            self.set_info("Citations copied to clipboard.".into());
        }
        if !keep_open {
            self.provenance = None;
        }
    }

    /// Opens the recording-modal dialog. State is reset on every
    /// open.
    fn open_record_dialog(&mut self) {
        self.record_dialog = Some(RecordDialog::new());
    }

    /// Begins capture using `self.record_dialog`'s configured device
    /// and format. Builds the engine session, spawns the cpal stream
    /// thread, and flips the dialog into `Recording`. Errors surface
    /// in the dialog's status line.
    fn record_dialog_start(&mut self) {
        let AppState::ProjectLoaded { root, .. } = &self.app_state else {
            return;
        };
        let root = root.clone();
        let Some(dialog) = self.record_dialog.as_mut() else {
            return;
        };
        let cfg = LiveConfig {
            sample_rate: dialog.sample_rate,
            channels: dialog.channels,
            ..Default::default()
        };
        let (mut engine, results) = match LiveSession::start(&root, cfg.clone()) {
            Ok(pair) => pair,
            Err(e) => {
                dialog.status = format!("Start failed: {e}");
                return;
            }
        };
        let producer = match engine.take_producer() {
            Some(p) => p,
            None => {
                dialog.status = "Start failed: engine produced no sample queue".into();
                return;
            }
        };
        let cpal = match spawn_cpal_input(&dialog.device, &cfg, producer) {
            Ok(handle) => handle,
            Err(e) => {
                // Engine session leaks an .in_progress dir if we abandon
                // it; stop+discard cleans it up.
                if let Ok(stopped) = engine.stop() {
                    let _ = stopped.discard();
                }
                dialog.status = format!("Audio device error: {e}");
                return;
            }
        };
        dialog.state = RecordDialogState::Recording(RecordingHandle {
            engine,
            results,
            cpal,
            started_at: std::time::Instant::now(),
        });
        dialog.status = "Recording…".into();
        dialog.meter_db = -120.0;
        dialog.elapsed_seconds = 0.0;
    }

    /// Stops the recording, transitioning into `Stopped`. The WAV
    /// file is flushed but not yet committed.
    fn record_dialog_stop(&mut self) {
        let Some(dialog) = self.record_dialog.as_mut() else {
            return;
        };
        let state = std::mem::replace(&mut dialog.state, RecordDialogState::Idle);
        let RecordDialogState::Recording(handle) = state else {
            dialog.state = state;
            return;
        };
        let RecordingHandle { engine, cpal, .. } = handle;
        // 1. Drop the cpal stream first so no further callbacks fire.
        drop(cpal);
        // 2. Stop the engine session, which joins the consumer thread.
        match engine.stop() {
            Ok(stopped) => {
                let duration = stopped.duration_seconds();
                let dropped = stopped.dropped_samples;
                dialog.elapsed_seconds = duration;
                dialog.status = if dropped > 0 {
                    format!("Stopped — {duration:.1}s captured ({dropped} samples dropped)")
                } else {
                    format!("Stopped — {duration:.1}s captured")
                };
                dialog.state = RecordDialogState::Stopped {
                    stopped,
                    duration,
                    dropped,
                };
            }
            Err(e) => {
                dialog.status = format!("Stop failed: {e}");
            }
        }
    }

    /// Commits the stopped recording into the project as a new bundle
    /// and selects it.
    fn record_dialog_commit(&mut self) {
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        let Some(dialog) = self.record_dialog.as_mut() else {
            return;
        };
        let state = std::mem::replace(&mut dialog.state, RecordDialogState::Idle);
        let (stopped, _duration, _dropped) = match state {
            RecordDialogState::Stopped {
                stopped,
                duration,
                dropped,
            } => (stopped, duration, dropped),
            other => {
                dialog.state = other;
                dialog.status = "Stop the recording before saving.".into();
                return;
            }
        };
        let name = dialog.name.trim().to_string();
        if name.is_empty() {
            dialog.status = "Give the recording a name.".into();
            // Recover the stopped session into the dialog for retry.
            dialog.state = RecordDialogState::Stopped {
                duration: stopped.duration_seconds(),
                dropped: stopped.dropped_samples,
                stopped,
            };
            return;
        }
        let params_json = format!(
            "{{\"device\":{:?},\"sample_rate\":{},\"channels\":{},\"duration_s\":{}}}",
            dialog.device,
            dialog.sample_rate,
            dialog.channels,
            stopped.duration_seconds(),
        );
        match project.commit_recording(stopped, &name, &params_json) {
            Ok(bundle_id) => {
                let saved_name = name.clone();
                self.record_dialog = None;
                self.set_info(format!("Saved recording as bundle “{saved_name}”."));
                self.select_bundle(bundle_id);
            }
            Err(e) => {
                if let Some(d) = self.record_dialog.as_mut() {
                    d.status = format!("Save failed: {e}");
                }
            }
        }
    }

    /// Discards the stopped recording (deletes the in-progress dir).
    fn record_dialog_discard(&mut self) {
        let Some(dialog) = self.record_dialog.as_mut() else {
            return;
        };
        let state = std::mem::replace(&mut dialog.state, RecordDialogState::Idle);
        match state {
            RecordDialogState::Stopped { stopped, .. } => {
                let _ = stopped.discard();
                self.record_dialog = None;
            }
            RecordDialogState::Recording(handle) => {
                // Closing while still recording: stop then discard.
                let RecordingHandle { engine, cpal, .. } = handle;
                drop(cpal);
                if let Ok(stopped) = engine.stop() {
                    let _ = stopped.discard();
                }
                self.record_dialog = None;
            }
            RecordDialogState::Idle => {
                self.record_dialog = None;
            }
        }
    }

    /// Renders the recording modal (when open). Drains the engine's
    /// result rings each frame so the meter stays current and the
    /// engine's consumer thread doesn't back-pressure.
    fn render_record_dialog(&mut self, ctx: &egui::Context) {
        if self.record_dialog.is_none() {
            return;
        }
        // Poll engine result rings to update the meter + elapsed time.
        if let Some(dialog) = self.record_dialog.as_mut() {
            if let RecordDialogState::Recording(handle) = &mut dialog.state {
                let mut latest_peak = dialog.meter_db;
                while let Ok(m) = handle.results.meters.pop() {
                    latest_peak = m.rms_db.value();
                }
                while handle.results.pitches.pop().is_ok() {}
                while handle.results.intensities.pop().is_ok() {}
                while handle.results.formants.pop().is_ok() {}
                dialog.meter_db = latest_peak;
                dialog.elapsed_seconds = handle.started_at.elapsed().as_secs_f64();
                ctx.request_repaint_after(std::time::Duration::from_millis(33));
            }
        }

        // Track which action to take after the closure (so we don't
        // borrow `self.record_dialog` mutably while also calling
        // `&mut self` methods on `SaddaApp`).
        let mut action: Option<RecordDialogAction> = None;
        let mut is_open = true;
        // Borrowing `self.record_dialog` here releases before action dispatch.
        if let Some(dialog) = self.record_dialog.as_mut() {
            egui::Window::new("Record from microphone")
                .open(&mut is_open)
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    let recording = matches!(dialog.state, RecordDialogState::Recording(_));
                    let stopped = matches!(dialog.state, RecordDialogState::Stopped { .. });
                    let idle = matches!(dialog.state, RecordDialogState::Idle);

                    ui.add_enabled_ui(idle, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Device:");
                            egui::ComboBox::from_id_salt("record-device")
                                .selected_text(&dialog.device)
                                .show_ui(ui, |ui| {
                                    for d in &dialog.device_options {
                                        ui.selectable_value(&mut dialog.device, d.clone(), d);
                                    }
                                });
                        });
                        ui.horizontal(|ui| {
                            ui.label("Sample rate:");
                            for &rate in &[16_000u32, 22_050, 44_100, 48_000] {
                                ui.selectable_value(
                                    &mut dialog.sample_rate,
                                    rate,
                                    format!("{rate}"),
                                );
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Channels:");
                            ui.selectable_value(&mut dialog.channels, 1, "1 (mono)");
                            ui.selectable_value(&mut dialog.channels, 2, "2 (stereo)");
                        });
                    });

                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.add_enabled(
                            !recording,
                            egui::TextEdit::singleline(&mut dialog.name).desired_width(220.0),
                        );
                    });

                    ui.separator();

                    if recording {
                        ui.label(format!("Elapsed: {:.1}s", dialog.elapsed_seconds));
                        ui.label(format!("Level: {:.1} dB-FS", dialog.meter_db));
                    } else if stopped {
                        ui.label(format!("Captured: {:.1}s", dialog.elapsed_seconds));
                    }

                    if !dialog.status.is_empty() {
                        ui.separator();
                        ui.label(&dialog.status);
                    }

                    ui.separator();
                    ui.horizontal(|ui| {
                        if idle {
                            if ui.button("Start").clicked() {
                                action = Some(RecordDialogAction::Start);
                            }
                            if ui.button("Cancel").clicked() {
                                action = Some(RecordDialogAction::Close);
                            }
                        } else if recording {
                            if ui.button("Stop").clicked() {
                                action = Some(RecordDialogAction::Stop);
                            }
                        } else if stopped {
                            if ui.button("Save").clicked() {
                                action = Some(RecordDialogAction::Commit);
                            }
                            if ui.button("Discard").clicked() {
                                action = Some(RecordDialogAction::Discard);
                            }
                        }
                    });
                });
        }

        if !is_open {
            action = Some(RecordDialogAction::Close);
        }

        match action {
            None => {}
            Some(RecordDialogAction::Start) => self.record_dialog_start(),
            Some(RecordDialogAction::Stop) => self.record_dialog_stop(),
            Some(RecordDialogAction::Commit) => self.record_dialog_commit(),
            Some(RecordDialogAction::Discard) => self.record_dialog_discard(),
            Some(RecordDialogAction::Close) => self.record_dialog_discard(),
        }
    }

    /// Informational banner — same look as `self.error` but green.
    /// Used to confirm successful imports / exports.
    fn set_info(&mut self, msg: String) {
        self.info = Some(msg);
    }

    /// Loads the bundle's audio, builds the envelope cache, and sets
    /// it as the active bundle. Errors surface in the bottom banner.
    /// Invalidates the spectrogram cache; it gets rebuilt lazily on
    /// the next frame.
    fn select_bundle(&mut self, bundle_id: i64) {
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        let audio = match project.load_audio(bundle_id) {
            Ok(a) => a,
            Err(e) => {
                self.set_error(format!("Failed to load bundle audio: {e}"));
                return;
            }
        };
        let mono: Vec<f32> = audio.mono_samples().collect();
        self.active_envelope = Some(EnvelopeCache {
            bundle_id,
            sample_rate: audio.sample_rate,
            duration_seconds: audio.duration_seconds(),
            mono_samples: mono,
        });
        self.selected_bundle_id = Some(bundle_id);
        self.active_spectrogram = None;
        self.active_embedding_heatmap = None;
        self.embedding_heatmap_error = None;
        self.selected_annotation = None;
        self.playback = None;
        self.draft_edit = DraftEdit::None;
        self.label_edit = None;
        self.timeline
            .reset_for_bundle(self.active_envelope.as_ref().unwrap().duration_seconds);
    }

    fn clear_bundle_selection(&mut self) {
        self.selected_bundle_id = None;
        self.active_envelope = None;
        self.active_spectrogram = None;
        self.active_embedding_heatmap = None;
        self.embedding_heatmap_error = None;
        self.selected_annotation = None;
        self.playback = None;
        self.draft_edit = DraftEdit::None;
        self.label_edit = None;
        self.timeline = TimelineState::default();
    }

    /// Deletes the currently-selected annotation (interval or
    /// point). No-op if nothing is selected.
    fn delete_selected_annotation(&mut self) {
        let Some(sel) = self.selected_annotation else {
            return;
        };
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        let result = match sel {
            AnnotationSelection::Interval { annotation_id, .. } => project
                .delete_interval(annotation_id)
                .map_err(|e| format!("Failed to delete interval: {e}")),
            AnnotationSelection::Point { annotation_id, .. } => project
                .delete_point(annotation_id)
                .map_err(|e| format!("Failed to delete point: {e}")),
        };
        if let Err(msg) = result {
            self.set_error(msg);
            return;
        }
        self.selected_annotation = None;
    }

    /// Renders the modal label-edit window when one is active.
    /// Commits on Enter / Save button; cancels on Escape / Cancel /
    /// window close. Inline overlay over the interval rect is a
    /// polish item (see the 2026-05-23 D6 DEVLOG entry).
    /// The rubric-editor window (slice S1): edit the project's rubric name,
    /// guidelines, status vocabulary, and per-tier controlled vocabularies,
    /// committing with "Apply".
    fn rubric_editor_window(&mut self, ctx: &egui::Context) {
        if self.rubric_editor.is_none() {
            return;
        }

        // One-time load of the current rubric into the working copy.
        if !self
            .rubric_editor
            .as_ref()
            .map(|r| r.loaded)
            .unwrap_or(true)
        {
            let AppState::ProjectLoaded { project, .. } = &self.app_state else {
                self.rubric_editor = None;
                return;
            };
            let rubric = project.rubric().ok().flatten();
            let statuses: Vec<(String, String)> = project
                .rubric_statuses()
                .map(|v| {
                    v.into_iter()
                        .map(|s| (s.value, s.description.unwrap_or_default()))
                        .collect()
                })
                .unwrap_or_default();
            let ed = self.rubric_editor.as_mut().unwrap();
            match rubric {
                Some(r) => {
                    ed.name = r.name;
                    ed.version = r.version;
                    ed.guidelines = r.guidelines.unwrap_or_default();
                }
                None => {
                    ed.name = "Annotation rubric".into();
                    ed.version = 1;
                }
            }
            ed.statuses = statuses;
            ed.loaded = true;
        }

        // Tier names for the controlled-vocabulary picker (active bundle).
        let tier_names: Vec<String> = match (self.selected_bundle_id, &self.app_state) {
            (Some(bid), AppState::ProjectLoaded { project, .. }) => project
                .tiers(Some(bid))
                .map(|ts| ts.into_iter().map(|t| t.name).collect())
                .unwrap_or_default(),
            _ => Vec::new(),
        };

        let mut apply = false;
        let mut close = false;
        let mut load_tier: Option<String> = None;
        let mut keep_open = true;
        egui::Window::new("Rubric")
            .open(&mut keep_open)
            .resizable(true)
            .default_width(440.0)
            .show(ctx, |ui| {
                let ed = self.rubric_editor.as_mut().expect("checked above");
                ui.horizontal(|ui| {
                    ui.label("Name:");
                    ui.text_edit_singleline(&mut ed.name);
                });
                ui.label(egui::RichText::new("Guidelines").small());
                ui.add(
                    egui::TextEdit::multiline(&mut ed.guidelines)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY),
                );
                ui.separator();

                // Status vocabulary.
                ui.label(egui::RichText::new("Statuses").strong());
                let mut remove_status: Option<usize> = None;
                for (i, (val, desc)) in ed.statuses.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::TextEdit::singleline(val)
                                .desired_width(110.0)
                                .hint_text("value"),
                        );
                        ui.add(
                            egui::TextEdit::singleline(desc)
                                .desired_width(200.0)
                                .hint_text("description"),
                        );
                        if ui.small_button("✖").clicked() {
                            remove_status = Some(i);
                        }
                    });
                }
                if let Some(i) = remove_status {
                    ed.statuses.remove(i);
                }
                if ui.button("➕ Add status").clicked() {
                    ed.statuses.push((String::new(), String::new()));
                }
                ui.separator();

                // Per-tier controlled vocabulary.
                ui.label(egui::RichText::new("Controlled vocabulary").strong());
                ui.horizontal(|ui| {
                    ui.label("Tier:");
                    let sel = ed
                        .tier_name
                        .clone()
                        .unwrap_or_else(|| "(select a tier)".into());
                    egui::ComboBox::from_id_salt("rubric_tier_pick")
                        .selected_text(sel)
                        .show_ui(ui, |ui| {
                            for tn in &tier_names {
                                let is_sel = ed.tier_name.as_deref() == Some(tn.as_str());
                                if ui.selectable_label(is_sel, tn).clicked() {
                                    load_tier = Some(tn.clone());
                                }
                            }
                        });
                });
                if ed.tier_name.is_some() {
                    ui.checkbox(
                        &mut ed.tier_closed,
                        "Closed vocabulary (reject out-of-vocab labels)",
                    );
                    let mut remove_vocab: Option<usize> = None;
                    for (i, (val, desc)) in ed.vocab.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(val)
                                    .desired_width(110.0)
                                    .hint_text("label"),
                            );
                            ui.add(
                                egui::TextEdit::singleline(desc)
                                    .desired_width(200.0)
                                    .hint_text("description"),
                            );
                            if ui.small_button("✖").clicked() {
                                remove_vocab = Some(i);
                            }
                        });
                    }
                    if let Some(i) = remove_vocab {
                        ed.vocab.remove(i);
                    }
                    if ui.button("➕ Add entry").clicked() {
                        ed.vocab.push((String::new(), String::new()));
                    }
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Apply").clicked() {
                        apply = true;
                    }
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
            });

        if !keep_open || close {
            self.rubric_editor = None;
            return;
        }

        // Load a newly-picked tier's controlled vocabulary into the editor.
        if let Some(tn) = load_tier {
            let AppState::ProjectLoaded { project, .. } = &self.app_state else {
                return;
            };
            let vocab: Vec<(String, String)> = project
                .controlled_vocabulary(&tn)
                .map(|v| {
                    v.into_iter()
                        .map(|e| (e.value, e.description.unwrap_or_default()))
                        .collect()
                })
                .unwrap_or_default();
            let closed = project
                .rubric_tier(&tn)
                .ok()
                .flatten()
                .map(|rt| rt.closed_vocabulary)
                .unwrap_or(false);
            if let Some(ed) = self.rubric_editor.as_mut() {
                ed.tier_name = Some(tn);
                ed.vocab = vocab;
                ed.tier_closed = closed;
            }
        }

        // Apply: commit the working copy to the project.
        if apply {
            // Snapshot the editor data so we can borrow the project mutably.
            let Some(ed) = self.rubric_editor.as_ref() else {
                return;
            };
            let name = ed.name.trim().to_string();
            let version = ed.version.max(1);
            let guidelines = (!ed.guidelines.trim().is_empty()).then(|| ed.guidelines.clone());
            let statuses: Vec<(String, Option<String>, i64)> = ed
                .statuses
                .iter()
                .filter(|(v, _)| !v.trim().is_empty())
                .enumerate()
                .map(|(i, (v, d))| {
                    (
                        v.trim().to_string(),
                        (!d.trim().is_empty()).then(|| d.clone()),
                        i as i64,
                    )
                })
                .collect();
            let tier = ed.tier_name.clone();
            let tier_closed = ed.tier_closed;
            let vocab: Vec<(String, Option<String>, i64)> = ed
                .vocab
                .iter()
                .filter(|(v, _)| !v.trim().is_empty())
                .enumerate()
                .map(|(i, (v, d))| {
                    (
                        v.trim().to_string(),
                        (!d.trim().is_empty()).then(|| d.clone()),
                        i as i64,
                    )
                })
                .collect();

            let result: Result<(), String> = (|| {
                let AppState::ProjectLoaded { project, .. } = &self.app_state else {
                    return Ok(());
                };
                if name.is_empty() {
                    return Err("Rubric name cannot be empty".into());
                }
                project
                    .set_rubric(&name, version, guidelines.as_deref())
                    .map_err(|e| format!("Failed to save rubric: {e}"))?;
                let status_defs: Vec<sadda_engine::StatusDef> = statuses
                    .iter()
                    .map(|(v, d, o)| sadda_engine::StatusDef {
                        value: v.clone(),
                        description: d.clone(),
                        sort_order: *o,
                    })
                    .collect();
                project
                    .set_rubric_statuses(&status_defs)
                    .map_err(|e| format!("Failed to save statuses: {e}"))?;
                if let Some(tn) = &tier {
                    project
                        .set_rubric_tier(tn, None, tier_closed)
                        .map_err(|e| format!("Failed to save tier config: {e}"))?;
                    let entries: Vec<sadda_engine::VocabEntry> = vocab
                        .iter()
                        .map(|(v, d, o)| sadda_engine::VocabEntry {
                            value: v.clone(),
                            description: d.clone(),
                            sort_order: *o,
                        })
                        .collect();
                    project
                        .set_controlled_vocabulary(tn, &entries)
                        .map_err(|e| format!("Failed to save vocabulary: {e}"))?;
                }
                Ok(())
            })();
            if let Err(msg) = result {
                self.set_error(msg);
            }
        }
    }

    /// The criteria-editor window (slice S2): browse/edit criteria, run a
    /// structured criterion to produce proposals on its preview tier, and
    /// accept/reject them.
    fn criteria_editor_window(&mut self, ctx: &egui::Context) {
        if self.criteria_editor.is_none() {
            return;
        }
        // One-time load of the criteria list.
        if !self
            .criteria_editor
            .as_ref()
            .map(|e| e.loaded)
            .unwrap_or(true)
        {
            let list = match &self.app_state {
                AppState::ProjectLoaded { project, .. } => project
                    .criteria()
                    .map(|cs| cs.into_iter().map(|c| (c.id, c.name)).collect())
                    .unwrap_or_default(),
                _ => Vec::new(),
            };
            let ed = self.criteria_editor.as_mut().unwrap();
            ed.list = list;
            ed.loaded = true;
        }

        let bundle_id = self.selected_bundle_id;
        let mut close = false;
        let mut load_criterion: Option<i64> = None;
        let mut save = false;
        let mut delete_id: Option<i64> = None;
        let mut run = false;
        let mut accept = false;
        let mut reject = false;
        let mut keep_open = true;

        egui::Window::new("Criteria")
            .open(&mut keep_open)
            .resizable(true)
            .default_width(560.0)
            .show(ctx, |ui| {
                let ed = self.criteria_editor.as_mut().expect("checked above");
                ui.horizontal_top(|ui| {
                    // Left: the criteria list.
                    ui.vertical(|ui| {
                        ui.set_min_width(150.0);
                        if ui.button("➕ New criterion").clicked() {
                            ed.reset_to_new();
                        }
                        ui.separator();
                        for (id, name) in &ed.list {
                            if ui
                                .selectable_label(ed.selected == Some(*id), name)
                                .clicked()
                            {
                                load_criterion = Some(*id);
                            }
                        }
                    });
                    ui.separator();
                    // Right: the editor for the selected/new criterion.
                    ui.vertical(|ui| {
                        egui::Grid::new("criterion_fields")
                            .num_columns(2)
                            .spacing([8.0, 6.0])
                            .show(ui, |ui| {
                                ui.label("Name");
                                ui.text_edit_singleline(&mut ed.name);
                                ui.end_row();
                                ui.label("Kind");
                                egui::ComboBox::from_id_salt("criterion_kind")
                                    .selected_text(&ed.kind)
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut ed.kind,
                                            "structured".to_string(),
                                            "structured",
                                        );
                                        ui.selectable_value(
                                            &mut ed.kind,
                                            "python".to_string(),
                                            "python",
                                        );
                                    });
                                ui.end_row();
                                ui.label("Target tier");
                                ui.text_edit_singleline(&mut ed.target_tier);
                                ui.end_row();
                            });
                        ui.label(
                            egui::RichText::new(if ed.kind == "structured" {
                                "Rule (JSON)"
                            } else {
                                "Python body — define criterion(proj, bundle_id)"
                            })
                            .small(),
                        );
                        ui.add(
                            egui::TextEdit::multiline(&mut ed.body)
                                .code_editor()
                                .desired_rows(8)
                                .desired_width(f32::INFINITY),
                        );
                        ui.horizontal(|ui| {
                            if ui.button("Save").clicked() {
                                save = true;
                            }
                            if let Some(id) = ed.selected {
                                if ui.button("Delete").clicked() {
                                    delete_id = Some(id);
                                }
                            }
                        });
                        ui.separator();
                        // Run + review (needs a selected criterion + bundle).
                        let has_target = !ed.target_tier.trim().is_empty();
                        let run_enabled =
                            bundle_id.is_some() && ed.selected.is_some() && ed.kind == "structured";
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(run_enabled, egui::Button::new("Run"))
                                .on_disabled_hover_text(if ed.kind == "python" {
                                    "Run python criteria from the script panel: \
                                     sadda.criteria.run_criterion(proj, id, bundle)"
                                } else {
                                    "Select a saved criterion and a bundle first"
                                })
                                .clicked()
                            {
                                run = true;
                            }
                            if ui
                                .add_enabled(
                                    bundle_id.is_some() && has_target,
                                    egui::Button::new("Accept proposals"),
                                )
                                .clicked()
                            {
                                accept = true;
                            }
                            if ui
                                .add_enabled(
                                    bundle_id.is_some() && has_target,
                                    egui::Button::new("Reject proposals"),
                                )
                                .clicked()
                            {
                                reject = true;
                            }
                        });
                        if let Some(msg) = &ed.status_msg {
                            ui.label(egui::RichText::new(msg).weak());
                        }
                    });
                });
                ui.separator();
                if ui.button("Close").clicked() {
                    close = true;
                }
            });

        if !keep_open || close {
            self.criteria_editor = None;
            return;
        }

        // ---- Apply requests after the window's &mut self borrow ----
        if let Some(id) = load_criterion {
            if let AppState::ProjectLoaded { project, .. } = &self.app_state {
                if let Ok(Some(c)) = project.get_criterion(id) {
                    let ed = self.criteria_editor.as_mut().unwrap();
                    ed.selected = Some(c.id);
                    ed.name = c.name;
                    ed.kind = c.kind;
                    ed.body = c.body;
                    ed.target_tier = c.target_tier;
                    ed.description = c.description.unwrap_or_default();
                    ed.status_msg = None;
                }
            }
        }
        if save {
            self.criteria_save();
        }
        if let Some(id) = delete_id {
            if let AppState::ProjectLoaded { project, .. } = &self.app_state {
                let _ = project.delete_criterion(id);
            }
            self.criteria_refresh_list();
            if let Some(ed) = self.criteria_editor.as_mut() {
                if ed.selected == Some(id) {
                    ed.reset_to_new();
                }
                ed.status_msg = Some("Deleted.".into());
            }
        }
        if run {
            let sel = self.criteria_editor.as_ref().and_then(|e| e.selected);
            let msg = match (sel, bundle_id, &self.app_state) {
                (Some(id), Some(bid), AppState::ProjectLoaded { project, .. }) => {
                    match project.run_criterion(id, bid) {
                        Ok(n) => format!("Ran — {n} proposal(s) on the preview tier."),
                        Err(e) => format!("Run failed: {e}"),
                    }
                }
                _ => "Select a saved criterion and a bundle.".into(),
            };
            if let Some(ed) = self.criteria_editor.as_mut() {
                ed.status_msg = Some(msg);
            }
        }
        if accept || reject {
            let target = self
                .criteria_editor
                .as_ref()
                .map(|e| e.target_tier.trim().to_string())
                .unwrap_or_default();
            let msg = match (bundle_id, &self.app_state) {
                (Some(bid), AppState::ProjectLoaded { project, .. }) if !target.is_empty() => {
                    let res = if accept {
                        project
                            .accept_proposals(bid, &target)
                            .map(|n| format!("Accepted {n}."))
                    } else {
                        project
                            .clear_proposals(bid, &target)
                            .map(|n| format!("Rejected {n}."))
                    };
                    res.unwrap_or_else(|e| format!("Failed: {e}"))
                }
                _ => "Select a bundle and set a target tier.".into(),
            };
            if let Some(ed) = self.criteria_editor.as_mut() {
                ed.status_msg = Some(msg);
            }
        }
    }

    /// The targets panel (slice S4a): lists the selected bundle's campaign work
    /// units, lets the user advance each target's status or delete it, generate
    /// targets from a saved criterion, or hand-mark a new one. Targets are read
    /// live from the project each frame; requested mutations are applied after
    /// the window's borrow ends, mirroring the criteria editor.
    fn targets_panel_window(&mut self, ctx: &egui::Context) {
        if self.targets_panel.is_none() {
            return;
        }
        let bundle_id = self.selected_bundle_id;

        // Read the data the panel renders BEFORE borrowing panel state, so the
        // window closure never holds two conflicting borrows of `self`.
        let (targets, criteria, assigns): (
            Vec<sadda_engine::Target>,
            Vec<(i64, String)>,
            Vec<sadda_engine::Assignment>,
        ) = match (bundle_id, &self.app_state) {
            (Some(bid), AppState::ProjectLoaded { project, .. }) => (
                project.targets(bid).unwrap_or_default(),
                project
                    .criteria()
                    .map(|cs| cs.into_iter().map(|c| (c.id, c.name)).collect())
                    .unwrap_or_default(),
                project.assignments(bid).unwrap_or_default(),
            ),
            _ => (Vec::new(), Vec::new(), Vec::new()),
        };
        // Group assignments by their target for the per-row summary.
        let mut assigns_by_target: std::collections::HashMap<i64, Vec<sadda_engine::Assignment>> =
            std::collections::HashMap::new();
        for a in assigns {
            assigns_by_target.entry(a.target_id).or_default().push(a);
        }
        // S5 — compare-tier choices (interval/point tiers) + progress readout.
        let (tier_choices, progress): (Vec<(i64, String)>, Option<sadda_engine::ProgressCounts>) =
            match (bundle_id, &self.app_state) {
                (Some(bid), AppState::ProjectLoaded { project, .. }) => (
                    project
                        .tiers(Some(bid))
                        .unwrap_or_default()
                        .into_iter()
                        .filter(|t| {
                            matches!(
                                t.r#type,
                                sadda_engine::TierType::Interval | sadda_engine::TierType::Point
                            )
                        })
                        .map(|t| (t.id, t.name))
                        .collect(),
                    project.target_progress(bid).ok(),
                ),
                _ => (Vec::new(), None),
            };

        let mut close = false;
        let mut keep_open = true;
        let mut generate = false;
        let mut add_manual = false;
        let mut random_assign = false;
        let mut assign_one: Option<i64> = None;
        let mut export_pkg = false;
        let mut import_pkg = false;
        let mut merge = false;
        let mut compare = false;
        let mut next_kind: Option<&'static str> = None;
        let mut status_change: Option<(i64, String)> = None;
        let mut delete_id: Option<i64> = None;

        egui::Window::new("Targets")
            .open(&mut keep_open)
            .resizable(true)
            .default_width(520.0)
            .show(ctx, |ui| {
                let panel = self.targets_panel.as_mut().expect("checked above");
                if bundle_id.is_none() {
                    ui.label("Select a bundle to manage its targets.");
                    return;
                }

                // Generate targets from a saved criterion.
                ui.horizontal(|ui| {
                    ui.label("Generate from criterion:");
                    let label = panel
                        .gen_criterion
                        .as_ref()
                        .map(|(_, n)| n.as_str())
                        .unwrap_or("— pick —");
                    egui::ComboBox::from_id_salt("target_gen_criterion")
                        .selected_text(label)
                        .show_ui(ui, |ui| {
                            for (id, name) in &criteria {
                                if ui
                                    .selectable_label(
                                        panel.gen_criterion.as_ref().map(|(i, _)| *i) == Some(*id),
                                        name,
                                    )
                                    .clicked()
                                {
                                    panel.gen_criterion = Some((*id, name.clone()));
                                }
                            }
                        });
                    if ui
                        .add_enabled(
                            panel.gen_criterion.is_some(),
                            egui::Button::new("Generate"),
                        )
                        .clicked()
                    {
                        generate = true;
                    }
                });

                // Hand-mark a manual target.
                ui.horizontal(|ui| {
                    ui.label("Add manual:");
                    ui.add(
                        egui::TextEdit::singleline(&mut panel.new_start)
                            .hint_text("start s")
                            .desired_width(60.0),
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut panel.new_end)
                            .hint_text("end s")
                            .desired_width(60.0),
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut panel.new_type)
                            .hint_text("type")
                            .desired_width(100.0),
                    );
                    if ui.button("Add").clicked() {
                        add_manual = true;
                    }
                });

                // S4b — assignment: one annotator for the per-row Assign button,
                // plus seeded random distribution across a comma-separated roster.
                ui.horizontal(|ui| {
                    ui.label("Annotator:");
                    ui.add(
                        egui::TextEdit::singleline(&mut panel.assign_annotator)
                            .hint_text("name")
                            .desired_width(110.0),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Assign randomly:");
                    ui.add(
                        egui::TextEdit::singleline(&mut panel.roster)
                            .hint_text("alice, bob, …")
                            .desired_width(150.0),
                    );
                    ui.add(
                        egui::TextEdit::singleline(&mut panel.seed)
                            .hint_text("seed")
                            .desired_width(60.0),
                    );
                    if ui
                        .button("Assign randomly")
                        .on_hover_text("Distribute unassigned targets across the roster (seeded)")
                        .clicked()
                    {
                        random_assign = true;
                    }
                });

                // S4c — package hand-off + explicit tier merge.
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Package:");
                    if ui
                        .add_enabled(
                            !panel.assign_annotator.trim().is_empty(),
                            egui::Button::new("Export for annotator…"),
                        )
                        .on_hover_text("Export the annotator above a self-contained sub-project")
                        .clicked()
                    {
                        export_pkg = true;
                    }
                    if ui
                        .button("Import package…")
                        .on_hover_text("Merge a returned annotator package back in")
                        .clicked()
                    {
                        import_pkg = true;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Merge tiers:");
                    ui.add(
                        egui::TextEdit::singleline(&mut panel.merge_sources)
                            .hint_text("phones [alice], phones [bob]")
                            .desired_width(180.0),
                    );
                    ui.label("→");
                    ui.add(
                        egui::TextEdit::singleline(&mut panel.merge_dest)
                            .hint_text("phones")
                            .desired_width(90.0),
                    );
                    if ui.button("Merge").clicked() {
                        merge = true;
                    }
                });

                // S5 — agreement + work queue.
                ui.separator();
                if let Some(p) = &progress {
                    ui.label(format_target_progress(p));
                }
                ui.horizontal(|ui| {
                    if ui
                        .button("Next to do")
                        .on_hover_text("Jump to the next unassigned/assigned target")
                        .clicked()
                    {
                        next_kind = Some("todo");
                    }
                    if ui.button("Next flagged").clicked() {
                        next_kind = Some("flagged");
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Compare:");
                    let combo = |ui: &mut egui::Ui,
                                 salt: &str,
                                 slot: &mut Option<(i64, String)>,
                                 choices: &[(i64, String)]| {
                        let text = slot.as_ref().map(|(_, n)| n.as_str()).unwrap_or("— tier —");
                        egui::ComboBox::from_id_salt(salt)
                            .selected_text(text)
                            .show_ui(ui, |ui| {
                                for (id, name) in choices {
                                    if ui
                                        .selectable_label(
                                            slot.as_ref().map(|(i, _)| *i) == Some(*id),
                                            name,
                                        )
                                        .clicked()
                                    {
                                        *slot = Some((*id, name.clone()));
                                    }
                                }
                            });
                    };
                    combo(ui, "compare_a", &mut panel.compare_a, &tier_choices);
                    ui.label("vs");
                    combo(ui, "compare_b", &mut panel.compare_b, &tier_choices);
                    if ui
                        .add_enabled(
                            panel.compare_a.is_some() && panel.compare_b.is_some(),
                            egui::Button::new("Compare"),
                        )
                        .clicked()
                    {
                        compare = true;
                    }
                });

                ui.separator();
                if targets.is_empty() {
                    ui.label(egui::RichText::new("No targets yet.").weak());
                }
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for t in &targets {
                        ui.horizontal(|ui| {
                            ui.label(format_target_row(t));
                            let mut status = t.status.clone();
                            egui::ComboBox::from_id_salt(("target_status", t.id))
                                .selected_text(&status)
                                .show_ui(ui, |ui| {
                                    for s in sadda_engine::TARGET_STATUSES {
                                        ui.selectable_value(&mut status, s.to_string(), s);
                                    }
                                });
                            if status != t.status {
                                status_change = Some((t.id, status));
                            }
                            if ui
                                .add_enabled(
                                    !panel.assign_annotator.trim().is_empty(),
                                    egui::Button::new("Assign"),
                                )
                                .on_hover_text("Assign the annotator above to this target")
                                .clicked()
                            {
                                assign_one = Some(t.id);
                            }
                            if ui.button("🗑").on_hover_text("Delete target").clicked() {
                                delete_id = Some(t.id);
                            }
                        });
                        let summary = format_assignment_summary(
                            assigns_by_target.get(&t.id).map(Vec::as_slice).unwrap_or(&[]),
                        );
                        ui.label(egui::RichText::new(summary).weak().small());
                    }
                });

                if let Some(msg) = &panel.status_msg {
                    ui.separator();
                    ui.label(egui::RichText::new(msg).weak());
                }
                ui.separator();
                if ui.button("Close").clicked() {
                    close = true;
                }
            });

        if !keep_open || close {
            self.targets_panel = None;
            return;
        }

        // ---- Apply requests after the window's &mut self borrow ----
        if generate {
            let crit = self
                .targets_panel
                .as_ref()
                .and_then(|p| p.gen_criterion.as_ref().map(|(i, _)| *i));
            let msg = match (crit, bundle_id, &self.app_state) {
                (Some(cid), Some(bid), AppState::ProjectLoaded { project, .. }) => {
                    match project.generate_targets_from_criterion(cid, bid) {
                        Ok(n) => format!("Generated {n} target(s)."),
                        Err(e) => format!("Generate failed: {e}"),
                    }
                }
                _ => "Pick a criterion and a bundle.".into(),
            };
            if let Some(p) = self.targets_panel.as_mut() {
                p.status_msg = Some(msg);
            }
        }
        if add_manual {
            let (start, end, ttype) = self
                .targets_panel
                .as_ref()
                .map(|p| (p.new_start.clone(), p.new_end.clone(), p.new_type.clone()))
                .unwrap_or_default();
            let msg = match (start.trim().parse::<f64>(), end.trim().parse::<f64>()) {
                _ if ttype.trim().is_empty() => "Enter a target type.".to_string(),
                (Ok(s), Ok(e)) => match (bundle_id, &self.app_state) {
                    (Some(bid), AppState::ProjectLoaded { project, .. }) => {
                        match project
                            .add_target(&sadda_engine::TargetSpec::new(bid, s, e, ttype.trim()))
                        {
                            Ok(_) => "Added target.".to_string(),
                            Err(err) => format!("Add failed: {err}"),
                        }
                    }
                    _ => "Select a bundle first.".to_string(),
                },
                _ => "Start and end must be numbers.".to_string(),
            };
            if let Some(p) = self.targets_panel.as_mut() {
                if msg == "Added target." {
                    p.new_start.clear();
                    p.new_end.clear();
                    p.new_type.clear();
                }
                p.status_msg = Some(msg);
            }
        }
        if let Some(target_id) = assign_one {
            let annotator = self
                .targets_panel
                .as_ref()
                .map(|p| p.assign_annotator.trim().to_string())
                .unwrap_or_default();
            let msg = match &self.app_state {
                AppState::ProjectLoaded { project, .. } if !annotator.is_empty() => {
                    match project.add_assignment(&sadda_engine::AssignmentSpec::new(
                        target_id, &annotator,
                    )) {
                        Ok(_) => format!("Assigned to {annotator}."),
                        Err(e) => format!("Assign failed: {e}"),
                    }
                }
                _ => "Enter an annotator.".into(),
            };
            if let Some(p) = self.targets_panel.as_mut() {
                p.status_msg = Some(msg);
            }
        }
        if random_assign {
            let (roster, seed) = self
                .targets_panel
                .as_ref()
                .map(|p| (p.roster.clone(), p.seed.clone()))
                .unwrap_or_default();
            let names: Vec<String> = roster
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let msg = match (names.is_empty(), seed.trim().parse::<i64>(), bundle_id, &self.app_state)
            {
                (true, _, _, _) => "Enter a roster (comma-separated).".to_string(),
                (_, Err(_), _, _) => "Seed must be an integer.".to_string(),
                (_, Ok(s), Some(bid), AppState::ProjectLoaded { project, .. }) => {
                    match project.assign_targets_randomly(bid, &names, s, None) {
                        Ok(n) => format!("Randomly assigned {n} target(s)."),
                        Err(e) => format!("Assign failed: {e}"),
                    }
                }
                _ => "Select a bundle first.".to_string(),
            };
            if let Some(p) = self.targets_panel.as_mut() {
                p.status_msg = Some(msg);
            }
        }
        if export_pkg {
            let annotator = self
                .targets_panel
                .as_ref()
                .map(|p| p.assign_annotator.trim().to_string())
                .unwrap_or_default();
            let msg = if annotator.is_empty() {
                Some("Enter an annotator to export.".to_string())
            } else if let Some(parent) = rfd::FileDialog::new()
                .set_title("Pick a folder to write the annotator package into")
                .pick_folder()
            {
                let dest = parent.join(format!("sadda_package_{annotator}"));
                match &self.app_state {
                    AppState::ProjectLoaded { project, .. } => {
                        Some(match project.export_annotator_package(&annotator, &dest) {
                            Ok(s) => format_export_summary(&s),
                            Err(e) => format!("Export failed: {e}"),
                        })
                    }
                    _ => None,
                }
            } else {
                None // cancelled
            };
            if let (Some(p), Some(msg)) = (self.targets_panel.as_mut(), msg) {
                p.status_msg = Some(msg);
            }
        }
        if import_pkg {
            let msg = if let Some(dir) = rfd::FileDialog::new()
                .set_title("Pick the returned annotator package directory")
                .pick_folder()
            {
                match &self.app_state {
                    AppState::ProjectLoaded { project, .. } => {
                        Some(match project.import_annotator_package(&dir) {
                            Ok(s) => format_import_summary(&s),
                            Err(e) => format!("Import failed: {e}"),
                        })
                    }
                    _ => None,
                }
            } else {
                None
            };
            if let (Some(p), Some(msg)) = (self.targets_panel.as_mut(), msg) {
                p.status_msg = Some(msg);
            }
        }
        if merge {
            let (sources, dest) = self
                .targets_panel
                .as_ref()
                .map(|p| (p.merge_sources.clone(), p.merge_dest.clone()))
                .unwrap_or_default();
            let names: Vec<String> = sources
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let msg = match (names.is_empty() || dest.trim().is_empty(), bundle_id, &self.app_state) {
                (true, _, _) => "Enter source tiers and a destination.".to_string(),
                (_, Some(bid), AppState::ProjectLoaded { project, .. }) => {
                    match project.merge_tiers(bid, &names, dest.trim()) {
                        Ok(n) => format!("Merged {n} annotation(s) into “{}”.", dest.trim()),
                        Err(e) => format!("Merge failed: {e}"),
                    }
                }
                _ => "Select a bundle first.".to_string(),
            };
            if let Some(p) = self.targets_panel.as_mut() {
                p.status_msg = Some(msg);
            }
        }
        if compare {
            let ids = self
                .targets_panel
                .as_ref()
                .and_then(|p| match (&p.compare_a, &p.compare_b) {
                    (Some((a, _)), Some((b, _))) => Some((*a, *b)),
                    _ => None,
                });
            let msg = match (ids, bundle_id, &self.app_state) {
                (Some((a, b)), Some(bid), AppState::ProjectLoaded { project, .. }) => {
                    Some(match project.compare_tiers(bid, a, b, None) {
                        Ok(r) => format_agreement_report(&r),
                        Err(e) => format!("Compare failed: {e}"),
                    })
                }
                _ => Some("Pick two tiers to compare.".to_string()),
            };
            if let (Some(p), Some(msg)) = (self.targets_panel.as_mut(), msg) {
                p.status_msg = Some(msg);
            }
        }
        if let Some(kind) = next_kind {
            let statuses: Vec<String> = match kind {
                "flagged" => vec!["flagged".into()],
                _ => vec!["unassigned".into(), "assigned".into()],
            };
            let msg = match (bundle_id, &self.app_state) {
                (Some(bid), AppState::ProjectLoaded { project, .. }) => {
                    Some(match project.next_target(bid, &statuses) {
                        Ok(Some(t)) => format!("Next {kind}: {}", format_target_row(&t)),
                        Ok(None) => format!("No {kind} targets."),
                        Err(e) => format!("Lookup failed: {e}"),
                    })
                }
                _ => None,
            };
            if let (Some(p), Some(msg)) = (self.targets_panel.as_mut(), msg) {
                p.status_msg = Some(msg);
            }
        }
        if let Some((id, status)) = status_change {
            if let AppState::ProjectLoaded { project, .. } = &self.app_state {
                let _ = project.update_target_status(id, &status);
            }
        }
        if let Some(id) = delete_id {
            if let AppState::ProjectLoaded { project, .. } = &self.app_state {
                let _ = project.delete_target(id);
            }
        }
    }

    /// The campaign QA dashboard (slice S6a): project + per-annotator
    /// completeness (read live), and on-demand per-tier QA + inter-annotator
    /// agreement. Read-live / apply-after-borrow, like the targets panel.
    fn dashboard_window(&mut self, ctx: &egui::Context) {
        if self.dashboard.is_none() {
            return;
        }
        let bundle_id = self.selected_bundle_id;

        // Read live aggregates before borrowing dashboard state.
        let (progress, annotators, tier_choices): (
            Option<sadda_engine::ProgressCounts>,
            Vec<sadda_engine::AnnotatorProgress>,
            Vec<(i64, String)>,
        ) = match &self.app_state {
            AppState::ProjectLoaded { project, .. } => (
                project.project_target_progress().ok(),
                project.assignment_progress().unwrap_or_default(),
                bundle_id
                    .and_then(|bid| project.tiers(Some(bid)).ok())
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|t| {
                        matches!(
                            t.r#type,
                            sadda_engine::TierType::Interval | sadda_engine::TierType::Point
                        )
                    })
                    .map(|t| (t.id, t.name))
                    .collect(),
            ),
            _ => (None, Vec::new(), Vec::new()),
        };
        // S6b — published rubric versions (version + note).
        let versions: Vec<(i64, Option<String>)> = match &self.app_state {
            AppState::ProjectLoaded { project, .. } => project
                .rubric_versions()
                .unwrap_or_default()
                .into_iter()
                .map(|v| (v.version, v.note))
                .collect(),
            _ => Vec::new(),
        };

        let mut close = false;
        let mut keep_open = true;
        let mut run_qa = false;
        let mut run_agreement = false;
        let mut publish = false;
        let mut run_impact = false;

        egui::Window::new("Campaign dashboard")
            .open(&mut keep_open)
            .resizable(true)
            .default_width(480.0)
            .show(ctx, |ui| {
                let dash = self.dashboard.as_mut().expect("checked above");

                ui.heading("Completeness");
                if let Some(p) = &progress {
                    ui.label(format_target_progress(p));
                }
                if annotators.is_empty() {
                    ui.label(egui::RichText::new("No assignments yet.").weak());
                } else {
                    for a in &annotators {
                        ui.label(format_annotator_progress(a));
                    }
                }

                ui.separator();
                ui.heading("QA & agreement");
                if bundle_id.is_none() {
                    ui.label(
                        egui::RichText::new("Select a bundle for per-tier QA / agreement.").weak(),
                    );
                } else {
                    ui.horizontal(|ui| {
                        ui.label("QA tier:");
                        let text = dash
                            .qa_tier
                            .as_ref()
                            .map(|(_, n)| n.as_str())
                            .unwrap_or("— tier —");
                        egui::ComboBox::from_id_salt("dash_qa_tier")
                            .selected_text(text)
                            .show_ui(ui, |ui| {
                                for (id, name) in &tier_choices {
                                    if ui
                                        .selectable_label(
                                            dash.qa_tier.as_ref().map(|(i, _)| *i) == Some(*id),
                                            name,
                                        )
                                        .clicked()
                                    {
                                        dash.qa_tier = Some((*id, name.clone()));
                                    }
                                }
                            });
                        if ui
                            .add_enabled(dash.qa_tier.is_some(), egui::Button::new("Run QA"))
                            .clicked()
                        {
                            run_qa = true;
                        }
                    });
                    if let Some(msg) = &dash.qa_msg {
                        ui.label(egui::RichText::new(msg).weak());
                    }
                    ui.horizontal(|ui| {
                        ui.label("Agreement for base tier:");
                        ui.add(
                            egui::TextEdit::singleline(&mut dash.agreement_base)
                                .hint_text("phones")
                                .desired_width(120.0),
                        );
                        if ui.button("Summarize").clicked() {
                            run_agreement = true;
                        }
                    });
                    for line in &dash.agreement_msgs {
                        ui.label(egui::RichText::new(line).weak());
                    }
                }

                // S6b — rubric versioning + impact.
                ui.separator();
                ui.heading("Rubric versions");
                ui.horizontal(|ui| {
                    ui.label("Publish note:");
                    ui.add(
                        egui::TextEdit::singleline(&mut dash.publish_note)
                            .hint_text("e.g. added creaky")
                            .desired_width(150.0),
                    );
                    if ui
                        .button("Publish current")
                        .on_hover_text("Snapshot the current rubric under its version number")
                        .clicked()
                    {
                        publish = true;
                    }
                });
                if let Some(msg) = &dash.version_msg {
                    ui.label(egui::RichText::new(msg).weak());
                }
                if versions.is_empty() {
                    ui.label(egui::RichText::new("No versions published yet.").weak());
                } else {
                    for (v, note) in &versions {
                        ui.label(
                            egui::RichText::new(format!(
                                "v{v}{}",
                                note.as_deref()
                                    .map(|n| format!(" — {n}"))
                                    .unwrap_or_default()
                            ))
                            .weak(),
                        );
                    }
                }
                ui.horizontal(|ui| {
                    ui.label("Impact since version:");
                    ui.add(
                        egui::TextEdit::singleline(&mut dash.impact_version)
                            .hint_text("1")
                            .desired_width(40.0),
                    );
                    if ui.button("Report").clicked() {
                        run_impact = true;
                    }
                });
                for line in &dash.impact_msgs {
                    ui.label(egui::RichText::new(line).weak());
                }

                ui.separator();
                if ui.button("Close").clicked() {
                    close = true;
                }
            });

        if !keep_open || close {
            self.dashboard = None;
            return;
        }

        if run_qa {
            let tier = self.dashboard.as_ref().and_then(|d| d.qa_tier.as_ref().map(|(i, _)| *i));
            let msg = match (tier, &self.app_state) {
                (Some(tid), AppState::ProjectLoaded { project, .. }) => {
                    Some(match project.tier_qa(tid) {
                        Ok(q) => format_qa_report(&q),
                        Err(e) => format!("QA failed: {e}"),
                    })
                }
                _ => None,
            };
            if let (Some(d), Some(msg)) = (self.dashboard.as_mut(), msg) {
                d.qa_msg = Some(msg);
            }
        }
        if run_agreement {
            let base = self
                .dashboard
                .as_ref()
                .map(|d| d.agreement_base.trim().to_string())
                .unwrap_or_default();
            let lines: Vec<String> = match (base.is_empty(), bundle_id, &self.app_state) {
                (true, _, _) => vec!["Enter a base tier name.".to_string()],
                (_, Some(bid), AppState::ProjectLoaded { project, .. }) => {
                    match project.agreement_summary(bid, &base) {
                        Ok(pairs) if pairs.is_empty() => {
                            vec![format!("No \"{base} [annotator]\" tiers on this bundle.")]
                        }
                        Ok(pairs) => pairs
                            .iter()
                            .map(|p| {
                                format!(
                                    "{} vs {}: {}",
                                    p.annotator_a,
                                    p.annotator_b,
                                    format_agreement_report(&p.report)
                                )
                            })
                            .collect(),
                        Err(e) => vec![format!("Agreement failed: {e}")],
                    }
                }
                _ => vec![],
            };
            if let Some(d) = self.dashboard.as_mut() {
                d.agreement_msgs = lines;
            }
        }
        if publish {
            let note = self
                .dashboard
                .as_ref()
                .map(|d| d.publish_note.trim().to_string())
                .unwrap_or_default();
            let msg = match &self.app_state {
                AppState::ProjectLoaded { project, .. } => {
                    let note = (!note.is_empty()).then_some(note.as_str());
                    Some(match project.publish_rubric_version(note) {
                        Ok(v) => format!("Published rubric v{}.", v.version),
                        Err(e) => format!("Publish failed: {e}"),
                    })
                }
                _ => None,
            };
            if let (Some(d), Some(msg)) = (self.dashboard.as_mut(), msg) {
                d.version_msg = Some(msg);
            }
        }
        if run_impact {
            let raw = self
                .dashboard
                .as_ref()
                .map(|d| d.impact_version.trim().to_string())
                .unwrap_or_default();
            let lines: Vec<String> = match (raw.parse::<i64>(), &self.app_state) {
                (Err(_), _) => vec!["Enter a version number.".to_string()],
                (Ok(v), AppState::ProjectLoaded { project, .. }) => {
                    match project.rubric_impact(v) {
                        Ok(impacts) if impacts.is_empty() => {
                            vec![format!("No changes since v{v}.")]
                        }
                        Ok(impacts) => impacts.iter().map(format_tier_impact).collect(),
                        Err(e) => vec![format!("Impact failed: {e}")],
                    }
                }
                _ => vec![],
            };
            if let Some(d) = self.dashboard.as_mut() {
                d.impact_msgs = lines;
            }
        }
    }

    /// Saves the criteria-editor working copy via the engine, then refreshes
    /// the list and selection.
    fn criteria_save(&mut self) {
        let Some(ed) = self.criteria_editor.as_ref() else {
            return;
        };
        let name = ed.name.trim().to_string();
        let kind = ed.kind.clone();
        let body = ed.body.clone();
        let target_tier = ed.target_tier.trim().to_string();
        let description = ed.description.clone();
        let result: Result<i64, String> = if name.is_empty() {
            Err("Name cannot be empty.".into())
        } else if target_tier.is_empty() {
            Err("Target tier cannot be empty.".into())
        } else if let AppState::ProjectLoaded { project, .. } = &self.app_state {
            let desc = (!description.trim().is_empty()).then_some(description.as_str());
            project
                .set_criterion(&name, desc, &kind, &body, &target_tier)
                .map(|c| c.id)
                .map_err(|e| format!("Save failed: {e}"))
        } else {
            Err("No project open.".into())
        };
        match result {
            Ok(id) => {
                self.criteria_refresh_list();
                if let Some(ed) = self.criteria_editor.as_mut() {
                    ed.selected = Some(id);
                    ed.status_msg = Some("Saved.".into());
                }
            }
            Err(msg) => {
                if let Some(ed) = self.criteria_editor.as_mut() {
                    ed.status_msg = Some(msg);
                }
            }
        }
    }

    /// Reloads the criteria list into the editor from the project.
    fn criteria_refresh_list(&mut self) {
        let list = match &self.app_state {
            AppState::ProjectLoaded { project, .. } => project
                .criteria()
                .map(|cs| cs.into_iter().map(|c| (c.id, c.name)).collect())
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        if let Some(ed) = self.criteria_editor.as_mut() {
            ed.list = list;
        }
    }

    /// The inline Annotation panel (modal-free editor): shows the selected
    /// annotation and edits its label / status / note in place, applying on
    /// commit (Enter / focus loss / status pick / Apply). Reloads its working
    /// copy whenever the selection changes.
    fn annotation_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        let Some(sel) = self.selected_annotation else {
            self.annotation_inspector.loaded_for = None;
            ui.label(egui::RichText::new("Select an annotation in a tier lane to edit it.").weak());
            return;
        };
        if self.annotation_inspector.loaded_for != Some(sel) {
            self.load_inspector(sel);
        }

        let mut commit = false;
        {
            let insp = &mut self.annotation_inspector;
            let kind_label = match insp.kind {
                LabelEditKind::Interval => "interval",
                LabelEditKind::Point => "point",
            };
            ui.label(egui::RichText::new(format!("{} · {kind_label}", insp.tier_name)).strong());
            if let Some(prov) = &insp.provenance {
                ui.label(
                    egui::RichText::new(prov)
                        .small()
                        .italics()
                        .color(egui::Color32::from_rgb(150, 120, 190)),
                )
                .on_hover_text(
                    "This annotation was produced by a criterion run (its provenance \
                     link). Editing it doesn't change that origin.",
                );
            }
            ui.separator();

            ui.label(egui::RichText::new("Label").small());
            // Commit on Enter (and via the Apply button below). We avoid
            // committing on bare focus-loss: that can lag a frame and land on
            // a different annotation if focus is lost by clicking another one.
            let label_resp = ui.text_edit_singleline(&mut insp.label);
            if label_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                commit = true;
            }
            if is_out_of_vocab(&insp.vocab, &insp.label) {
                let (color, msg) = if insp.closed {
                    (
                        egui::Color32::from_rgb(220, 110, 110),
                        "⚠ not in the closed vocabulary — will be rejected",
                    )
                } else {
                    (egui::Color32::from_rgb(210, 160, 70), "⚠ out of vocabulary")
                };
                ui.colored_label(color, msg);
            }
            if !insp.vocab.is_empty() {
                let vocab = insp.vocab.clone();
                ui.horizontal_wrapped(|ui| {
                    for v in &vocab {
                        if ui.selectable_label(insp.label == *v, v).clicked() {
                            insp.label = v.clone();
                            commit = true;
                        }
                    }
                });
            }

            ui.add_space(6.0);
            ui.label(egui::RichText::new("Status").small());
            if insp.statuses.is_empty() {
                ui.label(
                    egui::RichText::new("(no statuses in rubric)")
                        .italics()
                        .weak(),
                );
            } else {
                let selected = insp.status.clone().unwrap_or_else(|| "(none)".into());
                let statuses = insp.statuses.clone();
                egui::ComboBox::from_id_salt("inspector_status")
                    .selected_text(selected)
                    .show_ui(ui, |ui| {
                        if ui
                            .selectable_label(insp.status.is_none(), "(none)")
                            .clicked()
                        {
                            insp.status = None;
                            commit = true;
                        }
                        for s in &statuses {
                            let is_sel = insp.status.as_deref() == Some(s.as_str());
                            if ui.selectable_label(is_sel, s).clicked() {
                                insp.status = Some(s.clone());
                                commit = true;
                            }
                        }
                    });
            }

            ui.add_space(6.0);
            ui.label(egui::RichText::new("Note").small());
            // Multiline: Enter inserts a newline, so the note commits via the
            // Apply button (or alongside a label-Enter / status pick).
            ui.add(
                egui::TextEdit::multiline(&mut insp.note)
                    .desired_rows(3)
                    .desired_width(f32::INFINITY),
            );

            ui.add_space(6.0);
            if ui
                .button("Apply")
                .on_hover_text("save label, status, and note")
                .clicked()
            {
                commit = true;
            }
        }
        if commit {
            self.apply_inspector(sel);
        }
    }

    /// Loads the selected annotation's label / status / note and its tier's
    /// rubric context (vocabulary, closed flag, status options) into the
    /// inspector working copy.
    fn load_inspector(&mut self, sel: AnnotationSelection) {
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        let (tier_id, kind) = match sel {
            AnnotationSelection::Interval { tier_id, .. } => (tier_id, LabelEditKind::Interval),
            AnnotationSelection::Point { tier_id, .. } => (tier_id, LabelEditKind::Point),
        };
        let tier_name = project
            .get_tier(tier_id)
            .map(|t| t.name)
            .unwrap_or_default();
        let (label, status, note, run_id): (String, Option<String>, String, Option<i64>) =
            match sel {
                AnnotationSelection::Interval { annotation_id, .. } => project
                    .intervals(tier_id)
                    .ok()
                    .and_then(|rs| rs.into_iter().find(|r| r.id == annotation_id))
                    .map(|r| {
                        (
                            r.label.unwrap_or_default(),
                            r.status,
                            r.note.unwrap_or_default(),
                            r.processing_run_id,
                        )
                    })
                    .unwrap_or_default(),
                AnnotationSelection::Point { annotation_id, .. } => project
                    .points(tier_id)
                    .ok()
                    .and_then(|rs| rs.into_iter().find(|r| r.id == annotation_id))
                    .map(|r| {
                        (
                            r.label.unwrap_or_default(),
                            r.status,
                            r.note.unwrap_or_default(),
                            r.processing_run_id,
                        )
                    })
                    .unwrap_or_default(),
            };
        // Resolve a criterion-run link to a provenance one-liner.
        let provenance = run_id
            .and_then(|id| project.get_processing_run(id).ok().flatten())
            .filter(|run| run.kind == "criterion_run")
            .map(|run| format_provenance_line(&run.processor_id, &run.started_at));
        let vocab: Vec<String> = project
            .controlled_vocabulary(&tier_name)
            .map(|v| v.into_iter().map(|e| e.value).collect())
            .unwrap_or_default();
        let closed = project
            .rubric_tier(&tier_name)
            .ok()
            .flatten()
            .map(|rt| rt.closed_vocabulary)
            .unwrap_or(false);
        let statuses: Vec<String> = project
            .rubric_statuses()
            .map(|v| v.into_iter().map(|s| s.value).collect())
            .unwrap_or_default();

        let insp = &mut self.annotation_inspector;
        insp.loaded_for = Some(sel);
        insp.tier_name = tier_name;
        insp.kind = kind;
        insp.label = label;
        insp.status = status;
        insp.note = note;
        insp.vocab = vocab;
        insp.closed = closed;
        insp.statuses = statuses;
        insp.provenance = provenance;
    }

    /// Applies the inspector working copy to the selected annotation. The
    /// engine validates against the live rubric; on rejection the reason goes
    /// to the banner and the working copy is left intact for correction.
    fn apply_inspector(&mut self, sel: AnnotationSelection) {
        let (label, status, note) = {
            let i = &self.annotation_inspector;
            (i.label.clone(), i.status.clone(), i.note.clone())
        };
        let new_label = (!label.is_empty()).then_some(label);
        let new_status = status;
        let new_note = (!note.is_empty()).then_some(note);
        let result: Result<(), String> = match &self.app_state {
            AppState::ProjectLoaded { project, .. } => match sel {
                AnnotationSelection::Interval {
                    tier_id,
                    annotation_id,
                } => match project.intervals(tier_id) {
                    Ok(rows) => match rows.into_iter().find(|r| r.id == annotation_id) {
                        Some(existing) => project
                            .update_interval(
                                annotation_id,
                                &sadda_engine::IntervalSpec {
                                    tier_id,
                                    start_seconds: existing.start_seconds,
                                    end_seconds: existing.end_seconds,
                                    label: new_label,
                                    parent_annotation_id: existing.parent_annotation_id,
                                    status: new_status,
                                    note: new_note,
                                    processing_run_id: existing.processing_run_id,
                                    extra: existing.extra,
                                },
                            )
                            .map_err(|e| format!("Failed to save annotation: {e}")),
                        None => Ok(()),
                    },
                    Err(e) => Err(format!("Failed to reload interval: {e}")),
                },
                AnnotationSelection::Point {
                    tier_id,
                    annotation_id,
                } => match project.points(tier_id) {
                    Ok(rows) => match rows.into_iter().find(|r| r.id == annotation_id) {
                        Some(existing) => project
                            .update_point(
                                annotation_id,
                                &sadda_engine::PointSpec {
                                    tier_id,
                                    time_seconds: existing.time_seconds,
                                    label: new_label,
                                    parent_annotation_id: existing.parent_annotation_id,
                                    status: new_status,
                                    note: new_note,
                                    processing_run_id: existing.processing_run_id,
                                    extra: existing.extra,
                                },
                            )
                            .map_err(|e| format!("Failed to save annotation: {e}")),
                        None => Ok(()),
                    },
                    Err(e) => Err(format!("Failed to reload point: {e}")),
                },
            },
            _ => Ok(()),
        };
        if let Err(msg) = result {
            self.set_error(msg);
        }
    }

    fn label_edit_window(&mut self, ctx: &egui::Context) {
        if self.label_edit.is_none() {
            return;
        }

        // One-time load of the rubric context (controlled vocabulary, the
        // tier's open/closed flag, and the status options). The lane-render
        // free fns that create a `LabelEdit` have no project handle, so the
        // window fills these on its first frame.
        if !self.label_edit.as_ref().map(|le| le.loaded).unwrap_or(true) {
            let AppState::ProjectLoaded { project, .. } = &self.app_state else {
                self.label_edit = None;
                return;
            };
            let tier_id = self.label_edit.as_ref().unwrap().tier_id;
            let tier_name = project
                .get_tier(tier_id)
                .map(|t| t.name)
                .unwrap_or_default();
            let vocab: Vec<String> = project
                .controlled_vocabulary(&tier_name)
                .map(|v| v.into_iter().map(|e| e.value).collect())
                .unwrap_or_default();
            let closed = project
                .rubric_tier(&tier_name)
                .ok()
                .flatten()
                .map(|rt| rt.closed_vocabulary)
                .unwrap_or(false);
            let statuses: Vec<String> = project
                .rubric_statuses()
                .map(|v| v.into_iter().map(|s| s.value).collect())
                .unwrap_or_default();
            let le = self.label_edit.as_mut().unwrap();
            le.vocab = vocab;
            le.closed = closed;
            le.statuses = statuses;
            le.loaded = true;
        }

        let mut commit = false;
        let mut cancel = false;
        let mut keep_open = true;
        egui::Window::new("Edit annotation")
            .collapsible(false)
            .resizable(false)
            .open(&mut keep_open)
            .show(ctx, |ui| {
                let le = self.label_edit.as_mut().expect("checked above");
                ui.horizontal(|ui| {
                    ui.label("Label:");
                    let resp = ui.text_edit_singleline(&mut le.text);
                    if le.just_started {
                        resp.request_focus();
                        le.just_started = false;
                    }
                    if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        commit = true;
                    }
                });
                // Out-of-vocabulary flag (soft, unless the tier is closed).
                if is_out_of_vocab(&le.vocab, &le.text) {
                    let (color, msg) = if le.closed {
                        (
                            egui::Color32::from_rgb(220, 110, 110),
                            "⚠ not in the closed vocabulary — will be rejected",
                        )
                    } else {
                        (egui::Color32::from_rgb(210, 160, 70), "⚠ out of vocabulary")
                    };
                    ui.colored_label(color, msg);
                }
                // Controlled-vocabulary suggestion chips.
                if !le.vocab.is_empty() {
                    ui.label(egui::RichText::new("Vocabulary").small());
                    let vocab = le.vocab.clone();
                    ui.horizontal_wrapped(|ui| {
                        for v in &vocab {
                            if ui.selectable_label(le.text == *v, v).clicked() {
                                le.text = v.clone();
                            }
                        }
                    });
                }
                ui.separator();
                // Status picker (rubric-defined). Disabled with a hint when
                // the rubric defines no statuses yet.
                ui.horizontal(|ui| {
                    ui.label("Status:");
                    if le.statuses.is_empty() {
                        ui.label(
                            egui::RichText::new("(no statuses in rubric)")
                                .italics()
                                .weak(),
                        );
                    } else {
                        let selected = le.status.clone().unwrap_or_else(|| "(none)".into());
                        let statuses = le.statuses.clone();
                        egui::ComboBox::from_id_salt("annotation_status")
                            .selected_text(selected)
                            .show_ui(ui, |ui| {
                                if ui.selectable_label(le.status.is_none(), "(none)").clicked() {
                                    le.status = None;
                                }
                                for s in &statuses {
                                    let is_sel = le.status.as_deref() == Some(s.as_str());
                                    if ui.selectable_label(is_sel, s).clicked() {
                                        le.status = Some(s.clone());
                                    }
                                }
                            });
                    }
                });
                // Free-text note (allowed on any status, including none).
                ui.label(egui::RichText::new("Note").small());
                ui.text_edit_multiline(&mut le.note);
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Save").clicked() {
                        commit = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });
        if !keep_open {
            cancel = true;
        }

        if commit {
            // Snapshot the edited values (immutable borrow), then apply via
            // the project. The engine validates the label against the *live*
            // controlled vocabulary — so a closed-tier rejection is a clear
            // banner error, and the window stays open so the edit isn't lost.
            // (We deliberately do NOT pre-block on the cached `le.vocab`,
            // which can be stale if the rubric changed while the window was
            // open — that silently dropped valid labels.)
            let le = self.label_edit.as_ref().expect("checked above");
            let kind = le.kind;
            let tier_id = le.tier_id;
            let annotation_id = le.annotation_id;
            let new_label = (!le.text.is_empty()).then(|| le.text.clone());
            let new_status = le.status.clone();
            let new_note = (!le.note.is_empty()).then(|| le.note.clone());

            let result: Result<(), String> = match &self.app_state {
                AppState::ProjectLoaded { project, .. } => match kind {
                    LabelEditKind::Interval => match project.intervals(tier_id) {
                        Ok(rows) => match rows.into_iter().find(|r| r.id == annotation_id) {
                            // Re-fetch the base row so non-edited fields
                            // (bounds, parent, extra) survive concurrent edits.
                            Some(existing) => project
                                .update_interval(
                                    annotation_id,
                                    &sadda_engine::IntervalSpec {
                                        tier_id,
                                        start_seconds: existing.start_seconds,
                                        end_seconds: existing.end_seconds,
                                        label: new_label,
                                        parent_annotation_id: existing.parent_annotation_id,
                                        status: new_status,
                                        note: new_note,
                                        processing_run_id: existing.processing_run_id,
                                        extra: existing.extra,
                                    },
                                )
                                .map_err(|e| format!("Failed to save annotation: {e}")),
                            None => Ok(()), // deleted concurrently
                        },
                        Err(e) => Err(format!("Failed to reload interval: {e}")),
                    },
                    LabelEditKind::Point => match project.points(tier_id) {
                        Ok(rows) => match rows.into_iter().find(|r| r.id == annotation_id) {
                            Some(existing) => project
                                .update_point(
                                    annotation_id,
                                    &sadda_engine::PointSpec {
                                        tier_id,
                                        time_seconds: existing.time_seconds,
                                        label: new_label,
                                        parent_annotation_id: existing.parent_annotation_id,
                                        status: new_status,
                                        note: new_note,
                                        processing_run_id: existing.processing_run_id,
                                        extra: existing.extra,
                                    },
                                )
                                .map_err(|e| format!("Failed to save annotation: {e}")),
                            None => Ok(()),
                        },
                        Err(e) => Err(format!("Failed to reload point: {e}")),
                    },
                },
                _ => Ok(()),
            };
            match result {
                // Success closes the window; failure keeps it open with the
                // reason in the banner so the user can correct the label.
                Ok(()) => self.label_edit = None,
                Err(msg) => self.set_error(msg),
            }
        } else if cancel {
            self.label_edit = None;
        }
    }

    /// Spacebar toggle: start playback from the current cursor, or
    /// stop if already playing. Surfaces cpal errors in the
    /// error banner.
    fn toggle_playback(&mut self) {
        if self.playback.is_some() {
            self.playback = None;
            return;
        }
        let Some(env) = &self.active_envelope else {
            return;
        };
        match Playback::start(&env.mono_samples, env.sample_rate, self.timeline.cursor) {
            Ok(p) => self.playback = Some(p),
            Err(e) => self.set_error(format!("Playback failed: {e}")),
        }
    }

    /// Per-frame playback bookkeeping: pull the audio thread's
    /// atomic cursor into `timeline.cursor`, scroll the view if the
    /// cursor went offscreen, and drop the stream when it finishes.
    fn poll_playback(&mut self) {
        let Some(p) = &self.playback else {
            return;
        };
        let new_cursor = p.cursor_seconds();
        self.timeline.set_cursor(new_cursor);
        self.timeline.ensure_cursor_visible();
        if p.is_finished() {
            self.playback = None;
        }
    }

    /// Rebuilds the spectrogram cache if stale (i.e. cached bundle id
    /// or config differs from the current pair). On error, sets the
    /// error banner and leaves the previous cache (if any) in place.
    fn rebuild_spectrogram_if_stale(&mut self, ctx: &egui::Context) {
        let Some(env) = &self.active_envelope else {
            return;
        };
        let bundle_id = env.bundle_id;
        let cfg = self.persisted.spectrogram;
        if let Some(sc) = &self.active_spectrogram {
            if sc.bundle_id == bundle_id && sc.config == cfg {
                return;
            }
        }
        match build_spectrogram_texture(ctx, env, cfg) {
            Ok(sc) => self.active_spectrogram = Some(sc),
            Err(msg) => self.set_error(msg),
        }
    }

    /// Rebuilds the embedding-heatmap cache if stale. Drops the cache
    /// when no tier is selected (lane hidden) so the lane disappears
    /// immediately. On rebuild failure (selected tier missing, sidecar
    /// unreadable) sets `embedding_heatmap_error` for the lane to render
    /// as an in-lane hint instead of crashing or blanking out.
    fn rebuild_embedding_heatmap_if_stale(&mut self, ctx: &egui::Context) {
        let Some(env) = &self.active_envelope else {
            self.active_embedding_heatmap = None;
            self.embedding_heatmap_error = None;
            return;
        };
        let bundle_id = env.bundle_id;
        let cfg = self.persisted.embedding.clone();
        // No tier selected → lane hidden, drop any prior cache.
        if cfg.selected_tier_id.is_none() {
            self.active_embedding_heatmap = None;
            self.embedding_heatmap_error = None;
            return;
        }
        if let Some(c) = &self.active_embedding_heatmap {
            if c.bundle_id == bundle_id && c.config == cfg {
                return;
            }
        }
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        match build_embedding_heatmap_texture(ctx, project, env, cfg) {
            Ok(c) => {
                self.active_embedding_heatmap = Some(c);
                self.embedding_heatmap_error = None;
            }
            Err(e) => {
                self.active_embedding_heatmap = None;
                self.embedding_heatmap_error = Some(e);
            }
        }
    }

    /// D10: recompute the measure tracks if the cache is stale for the
    /// selected bundle + current [`MeasureTrackConfig`]. Pure CPU work
    /// (no GPU upload, unlike the spectrogram), so it takes no `ctx`.
    /// Skips the analysis entirely when every lane is hidden.
    fn rebuild_tracks_if_stale(&mut self) {
        let Some(env) = &self.active_envelope else {
            return;
        };
        let bundle_id = env.bundle_id;
        let cfg = self.persisted.tracks;
        if let Some(tc) = &self.active_tracks {
            if tc.bundle_id == bundle_id && tc.config == cfg {
                return;
            }
        }
        if !cfg.any_visible() {
            // Nothing to draw; don't spend the FFT/LPC budget. Leave
            // any prior cache in place so re-enabling a lane that was
            // just toggled off doesn't pay to recompute.
            return;
        }
        self.active_tracks = Some(compute_measure_tracks(env, cfg));
    }

    /// D10: refresh the per-lane overlay bands if the View-menu selection
    /// changed. Reads the refdist store (a filesystem + parquet hit), so
    /// it only runs on a selection change, not every frame.
    fn rebuild_overlays_if_stale(&mut self) {
        Self::sync_overlay(&self.persisted.f0_overlay, "f0", &mut self.overlays.f0);
        Self::sync_overlay(
            &self.persisted.intensity_overlay,
            "intensity",
            &mut self.overlays.intensity,
        );
    }

    fn sync_overlay(
        selection: &Option<RefdistOverlay>,
        parameter: &str,
        slot: &mut Option<(RefdistOverlay, Option<OverlayBand>)>,
    ) {
        match selection {
            None => *slot = None,
            Some(sel) => {
                if slot.as_ref().map(|(key, _)| key) == Some(sel) {
                    return; // already resolved for this selection
                }
                *slot = Some((sel.clone(), resolve_overlay_band(sel, parameter)));
            }
        }
    }

    /// D10: refresh the Reference-panel data if its (distribution, phone,
    /// param) selection changed.
    fn rebuild_reference_if_stale(&mut self) {
        let Some(sel) = self.persisted.reference_dist.clone() else {
            if self.reference.key.is_some() {
                self.reference = ReferenceView::default();
            }
            return;
        };
        let phone = self.persisted.reference_phone.clone();
        let param = self.persisted.reference_param.clone();
        let key = (sel.clone(), phone.clone(), param.clone());
        if self.reference.key.as_ref() == Some(&key) {
            return;
        }
        self.reference = build_reference_view(&sel, phone, param);
    }

    fn set_error(&mut self, msg: String) {
        self.error = Some(msg);
    }
}

/// Returns a sensible default project name from a folder path: the
/// folder's basename, falling back to `"untitled"`.
fn project_name_from_path(path: &Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "untitled".into())
}

/// Runs STFT + power-spectrogram + dB-normalise + colormap on the
/// envelope cache's mono samples and uploads the result as an egui
/// texture. Returns a fresh [`SpectrogramCache`] or an error message
/// suitable for the bottom banner.
fn build_spectrogram_texture(
    ctx: &egui::Context,
    env: &EnvelopeCache,
    cfg: SpectrogramConfig,
) -> Result<SpectrogramCache, String> {
    let sr = env.sample_rate as f32;
    let window_samples = ((cfg.window_ms / 1000.0) * sr).round() as usize;
    let hop_samples = ((cfg.hop_ms / 1000.0) * sr).round() as usize;
    if window_samples < 4 || hop_samples == 0 {
        return Err(format!(
            "Spectrogram: window ({} ms) / hop ({} ms) are too small at {} Hz",
            cfg.window_ms, cfg.hop_ms, sr,
        ));
    }
    if env.mono_samples.len() < window_samples {
        return Err(format!(
            "Spectrogram: bundle too short ({} samples) for window {} ms",
            env.mono_samples.len(),
            cfg.window_ms,
        ));
    }

    let window = sadda_engine::dsp::hann(window_samples);
    let (stft_out, shape) = sadda_engine::dsp::stft(&env.mono_samples, &window, hop_samples);
    let power = sadda_engine::dsp::power_spectrogram(&stft_out, shape);
    let normalized = power_to_db_normalized(&power, cfg.dynamic_range_db);

    // Optionally downsample the time dimension so the texture stays
    // under MAX_SPECTROGRAM_WIDTH. The spike note in the B3 DEVLOG
    // entry flagged 10-minute files; this is the cap that handles it.
    let (display_width, display) = if shape.n_frames > MAX_SPECTROGRAM_WIDTH {
        let stride = shape.n_frames.div_ceil(MAX_SPECTROGRAM_WIDTH);
        let new_width = shape.n_frames.div_ceil(stride);
        let mut out = vec![0.0_f32; shape.n_freq_bins * new_width];
        for b in 0..shape.n_freq_bins {
            for x in 0..new_width {
                let start = x * stride;
                let end = (start + stride).min(shape.n_frames);
                let mut acc = 0.0_f32;
                for f in start..end {
                    acc += normalized[b * shape.n_frames + f];
                }
                out[b * new_width + x] = acc / (end - start) as f32;
            }
        }
        (new_width, out)
    } else {
        (shape.n_frames, normalized)
    };

    let rgba = colormap_bake(&display, display_width, shape.n_freq_bins, cfg.colormap);
    let image = egui::ColorImage::from_rgba_unmultiplied([display_width, shape.n_freq_bins], &rgba);
    let texture = ctx.load_texture("spectrogram", image, egui::TextureOptions::LINEAR);

    Ok(SpectrogramCache {
        bundle_id: env.bundle_id,
        config: cfg,
        texture,
        duration_seconds: env.duration_seconds,
        nyquist_hz: sr / 2.0,
    })
}

/// Reads the selected `continuous_vector` tier, normalises + bakes a
/// `(n_dims × n_frames)` colormapped texture, and uploads it via egui.
/// Returns a fresh [`EmbeddingHeatmapCache`] or an actionable error
/// message (no selected tier → caller skips this entirely; tier missing
/// / wrong type / sidecar unreadable → message shown inside the lane).
fn build_embedding_heatmap_texture(
    ctx: &egui::Context,
    project: &Project,
    env: &EnvelopeCache,
    cfg: EmbeddingHeatmapConfig,
) -> Result<EmbeddingHeatmapCache, String> {
    let tier_id = cfg
        .selected_tier_id
        .ok_or_else(|| "no embedding tier selected".to_string())?;

    let tier = project
        .get_tier(tier_id)
        .map_err(|e| format!("read tier {tier_id}: {e}"))?;
    if tier.r#type != TierType::ContinuousVector {
        return Err(format!(
            "tier {tier_id} ({}) is not a continuous_vector tier",
            tier.name,
        ));
    }
    let matrix = project
        .read_continuous_vector(tier_id)
        .map_err(|e| format!("read embedding tier {tier_id}: {e}"))?;
    let (n_frames, n_dims) = matrix.dim();
    if n_frames == 0 || n_dims == 0 {
        return Err(format!(
            "tier {tier_id} ({}) is empty ({n_frames} frames × {n_dims} dims)",
            tier.name,
        ));
    }

    // The engine returns `Array2<f64>` shaped `[n_frames, n_dims]`. We
    // need a dim-major / frame-minor `[n_dims * n_frames]` `f32` buffer
    // for `colormap_bake`. Transpose + downcast in one pass; small
    // enough cost compared to the GPU upload.
    let mut dim_major: Vec<f32> = vec![0.0; n_dims * n_frames];
    for f in 0..n_frames {
        for d in 0..n_dims {
            dim_major[d * n_frames + f] = matrix[(f, d)] as f32;
        }
    }

    // Bucket the time axis if the matrix has more frames than the
    // colormap texture cap — same strategy as the spectrogram, so a
    // very long file doesn't try to upload a 100k-wide texture.
    let (display_width, display) = if n_frames > MAX_SPECTROGRAM_WIDTH {
        let stride = n_frames.div_ceil(MAX_SPECTROGRAM_WIDTH);
        let new_width = n_frames.div_ceil(stride);
        let mut out = vec![0.0f32; n_dims * new_width];
        for d in 0..n_dims {
            for x in 0..new_width {
                let start = x * stride;
                let end = (start + stride).min(n_frames);
                let mut acc = 0.0f32;
                for f in start..end {
                    acc += dim_major[d * n_frames + f];
                }
                out[d * new_width + x] = acc / (end - start) as f32;
            }
        }
        (new_width, out)
    } else {
        (n_frames, dim_major)
    };

    let normalized = normalize_embedding(&display, n_dims, display_width, cfg.normalization);
    let rgba = colormap_bake(&normalized, display_width, n_dims, cfg.colormap);
    let image = egui::ColorImage::from_rgba_unmultiplied([display_width, n_dims], &rgba);
    let texture = ctx.load_texture(
        format!("embedding_heatmap_{tier_id}"),
        image,
        egui::TextureOptions::LINEAR,
    );

    Ok(EmbeddingHeatmapCache {
        bundle_id: env.bundle_id,
        config: cfg,
        texture,
        duration_seconds: env.duration_seconds,
        n_dims,
        tier_name: tier.name,
    })
}

/// D10: run the engine's per-frame analyses for the visible lanes and
/// pack them into a [`MeasureTrackCache`]. Hidden lanes are skipped so
/// toggling a lane off reclaims its analysis cost. Pitch needs an
/// owned [`Audio`]; formants/intensity take the sample slice directly.
fn compute_measure_tracks(env: &EnvelopeCache, cfg: MeasureTrackConfig) -> MeasureTrackCache {
    let sr = env.sample_rate;

    // pitch() and the VAD model both want an owned Audio; build it once if
    // either lane needs it. Recompute is rare (bundle / config change
    // only), so the one-off sample clone is fine.
    let audio = (cfg.f0_visible || cfg.vad_visible).then(|| sadda_engine::Audio {
        samples: env.mono_samples.clone(),
        sample_rate: sr,
        channels: 1,
    });

    let mut f0 = Vec::new();
    let mut vad = Vec::new();
    let mut vad_error = None;
    if let Some(audio) = &audio {
        if cfg.f0_visible {
            let pcfg = PitchConfig {
                min_freq_hz: cfg.f0_min_hz,
                max_freq_hz: cfg.f0_max_hz,
                voicing_threshold: cfg.f0_voicing_threshold,
                ..PitchConfig::default()
            };
            f0 = pitch(audio, &pcfg, PitchMethod::WindowedAutocorrelation);
        }
        if cfg.vad_visible {
            // VAD needs ONNX Runtime at runtime; a missing/incompatible
            // ORT surfaces as an error, shown as a hint in the lane.
            match vad_bundled(audio) {
                Ok(frames) => vad = frames,
                Err(e) => vad_error = Some(e.to_string()),
            }
        }
    }

    let formants = if cfg.formants_visible {
        let fcfg = FormantsConfig {
            n_formants: cfg.formant_count,
            ..FormantsConfig::default()
        };
        formants(&env.mono_samples, sr, &fcfg)
    } else {
        Vec::new()
    };

    let intensity_frames = if cfg.intensity_visible {
        // 30 ms / 10 ms — the standard Praat-like intensity analysis
        // window; long enough to span a pitch period at the low end.
        intensity(&env.mono_samples, sr, 0.030, 0.010)
    } else {
        Vec::new()
    };

    MeasureTrackCache {
        bundle_id: env.bundle_id,
        config: cfg,
        f0,
        formants,
        intensity: intensity_frames,
        vad,
        vad_error,
    }
}

/// D10: resolve an overlay selection against the refdist store and
/// summarise it into a drawable [`OverlayBand`]. Returns `None` (rather
/// than erroring) if the store is unavailable, the distribution isn't
/// installed, or it has no values for the parameter — the lane then just
/// draws without a band.
fn resolve_overlay_band(sel: &RefdistOverlay, parameter: &str) -> Option<OverlayBand> {
    let store = RefdistStore::user_default().ok()?;
    let rd = store.get(&sel.id, &sel.version)?;
    let filters: Vec<(&str, &str)> = match &sel.sex {
        Some(s) => vec![("sex", s.as_str())],
        None => Vec::new(),
    };
    let summary = rd.summary(parameter, &filters).ok()?;
    Some(OverlayBand {
        summary,
        kind: rd.manifest.measure.kind,
        label: overlay_label(&rd, sel),
    })
}

/// Subgroup-qualified label for an overlay band ("f0 norms (m)").
fn overlay_label(rd: &RefDist, sel: &RefdistOverlay) -> String {
    let base = if rd.manifest.title.is_empty() {
        rd.manifest.id.clone()
    } else {
        rd.manifest.title.clone()
    };
    match &sel.sex {
        Some(s) => format!("{base} ({s})"),
        None => base,
    }
}

/// D10: resolve the Reference-panel selection into a [`ReferenceView`] —
/// the vowel-space cloud (first two parameters) plus the histogram +
/// summary for the active parameter. Best-effort: any read failure
/// leaves the corresponding field empty rather than erroring.
fn build_reference_view(
    sel: &RefdistOverlay,
    phone: Option<String>,
    param: Option<String>,
) -> ReferenceView {
    let mut view = ReferenceView {
        key: Some((sel.clone(), phone.clone(), param.clone())),
        ..ReferenceView::default()
    };
    let Ok(store) = RefdistStore::user_default() else {
        return view;
    };
    let Some(rd) = store.get(&sel.id, &sel.version) else {
        return view;
    };
    view.title = overlay_label(&rd, sel);
    view.kind = Some(rd.manifest.measure.kind);
    let params = rd.manifest.measure.parameters.clone();
    view.phones = rd.manifest.measure.phones.clone();

    let mut filters: Vec<(&str, &str)> = Vec::new();
    if let Some(s) = sel.sex.as_deref() {
        filters.push(("sex", s));
    }
    if let Some(p) = phone.as_deref() {
        filters.push(("phone", p));
    }

    // Vowel-space cloud from the first two parameters.
    if params.len() >= 2 {
        if let Ok(pts) = rd.points2d(&params[0], &params[1], &filters) {
            view.cloud = pts.into_iter().map(|(x, y)| [x, y]).collect();
        }
    }

    // Histogram + summary of the active parameter (default: first param).
    let active = param.or_else(|| params.first().cloned());
    if let Some(p) = active {
        if let Ok(s) = rd.summary(&p, &filters) {
            view.summary = Some(s);
        }
        if rd.manifest.measure.kind != MeasureKind::SummaryNormativeRange {
            if let Ok(h) = rd.histogram(&p, 24, &filters) {
                view.histogram = Some(h);
            }
        }
        view.active_param = Some(p);
    }
    view.params = params;
    view
}

/// D10: the measured `(F1, F2)` at the cursor, from the formant track —
/// the point plotted against the reference vowel cloud. `None` if there's
/// no formant frame or it has fewer than two formants.
fn formant_point_at_cursor(frames: &[FormantFrame], cursor: f64) -> Option<(f64, f64)> {
    let times: Vec<f64> = frames.iter().map(|f| f.time_seconds).collect();
    let i = nearest_frame_index(&times, cursor)?;
    let f = &frames[i];
    if f.frequencies.len() >= 2 {
        Some((
            f.frequencies[0].value() as f64,
            f.frequencies[1].value() as f64,
        ))
    } else {
        None
    }
}

/// D10: the median of the voiced f0 estimates — the bundle's "your value"
/// f0 marker against an f0 reference histogram. `None` if nothing is
/// voiced above `threshold`.
fn median_voiced_f0(frames: &[PitchFrame], threshold: f32) -> Option<f64> {
    let mut vals: Vec<f64> = frames
        .iter()
        .filter(|f| f.voicing >= threshold)
        .map(|f| f.frequency_hz.value() as f64)
        .collect();
    if vals.is_empty() {
        return None;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(vals[vals.len() / 2])
}

/// D10: draws the Reference panel's 1-D histogram — bars from the engine
/// [`Histogram`], dashed grey p5 / p95 percentile lines + a solid median
/// line from `summary`, and (if present) a red "you" line at the measured
/// value so the user sees where their measurement falls in the reference.
fn draw_reference_histogram(
    ui: &mut egui::Ui,
    hist: &Histogram,
    summary: Option<&Summary>,
    measured: Option<f64>,
) {
    let bars: Vec<Bar> = hist
        .counts
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let lo = hist.edges[i];
            let hi = hist.edges[i + 1];
            let center = 0.5 * (lo + hi);
            let width = (hi - lo) * 0.92;
            Bar::new(center, c as f64)
                .width(width)
                .fill(egui::Color32::from_rgba_unmultiplied(120, 140, 200, 160))
        })
        .collect();

    Plot::new("ref_histogram")
        .height(180.0)
        .y_axis_label("count")
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(BarChart::new("reference", bars));
            if let Some(s) = summary {
                let pct = egui::Color32::from_gray(140);
                for x in [s.p5, s.p95] {
                    plot_ui.vline(
                        VLine::new("", x)
                            .color(pct)
                            .style(LineStyle::dashed_loose()),
                    );
                }
                plot_ui.vline(
                    VLine::new("median", s.median)
                        .color(egui::Color32::from_gray(180))
                        .width(1.5),
                );
            }
            if let Some(x) = measured {
                plot_ui.vline(
                    VLine::new("you", x)
                        .color(egui::Color32::from_rgb(230, 70, 70))
                        .width(2.0),
                );
            }
        });

    // Legend / readout line beneath the plot.
    if let Some(x) = measured {
        let pctile = summary.map(|s| describe_position(x, s)).unwrap_or_default();
        ui.label(
            egui::RichText::new(format!("your value: {x:.1}{pctile}"))
                .color(egui::Color32::from_rgb(210, 90, 90))
                .small(),
        );
    }
}

/// D10: a short "(below p5)" / "(near median)" / "(above p95)" tag locating
/// a measured value within a reference summary, for the histogram readout.
fn describe_position(x: f64, s: &Summary) -> String {
    let where_ = if x < s.p5 {
        "below p5"
    } else if x < s.p25 {
        "p5–p25"
    } else if x <= s.p75 {
        "interquartile"
    } else if x <= s.p95 {
        "p75–p95"
    } else {
        "above p95"
    };
    format!("  ({where_})")
}

/// D10: draws a reference-distribution band across the lane's full x
/// width. The encoding is keyed on [`MeasureKind`] so the three kinds are
/// never visually confused: an **observed** distribution reads as a cool
/// neutral percentile band, a **normative** range as a green clinical
/// band, and a **target zone** as a distinct amber goal region with a
/// dashed border and a "TARGET" tag. Each draws an outer p5–p95 fill, an
/// inner p25–p75 fill, and a centre line at the median/mean. `y_top` is
/// the lane's y-axis maximum, used only to place the corner label.
fn draw_refdist_band(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    band: &OverlayBand,
    x0: f64,
    x1: f64,
    y_top: f64,
    palette: PlotPalette,
) {
    let s = &band.summary;
    // Observed / normative / target bands can share a lane, so their
    // colours are a discrimination case → Okabe–Ito under the CVD palette
    // (blue / bluish-green / vermillion). The "TARGET" tag and the dashed
    // border carry the meaning redundantly, never colour alone.
    let (base, tag) = match (palette, band.kind) {
        (PlotPalette::Default, MeasureKind::ObservedDistribution) => {
            (egui::Color32::from_rgb(90, 140, 210), "observed")
        }
        (PlotPalette::Default, MeasureKind::SummaryNormativeRange) => {
            (egui::Color32::from_rgb(70, 165, 95), "norm")
        }
        (PlotPalette::Default, MeasureKind::TargetZone) => {
            (egui::Color32::from_rgb(230, 160, 40), "TARGET")
        }
        (PlotPalette::OkabeIto, MeasureKind::ObservedDistribution) => {
            (egui::Color32::from_rgb(0, 114, 178), "observed")
        }
        (PlotPalette::OkabeIto, MeasureKind::SummaryNormativeRange) => {
            (egui::Color32::from_rgb(0, 158, 115), "norm")
        }
        (PlotPalette::OkabeIto, MeasureKind::TargetZone) => {
            (egui::Color32::from_rgb(213, 94, 0), "TARGET")
        }
    };
    let alpha = |a: u8| egui::Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), a);

    let rect = |y_lo: f64, y_hi: f64| {
        PlotPoints::from(vec![[x0, y_lo], [x1, y_lo], [x1, y_hi], [x0, y_hi]])
    };

    // Outer p5–p95, then inner p25–p75 (drawn over it, more opaque).
    plot_ui.polygon(
        Polygon::new("", rect(s.p5, s.p95))
            .fill_color(alpha(28))
            .stroke(egui::Stroke::NONE),
    );
    let inner = Polygon::new("", rect(s.p25, s.p75)).fill_color(alpha(60));
    // A target zone gets a bold dashed border so it reads as prescriptive,
    // not descriptive; observed/normative bands stay borderless.
    let inner = if band.kind == MeasureKind::TargetZone {
        inner
            .stroke(egui::Stroke::new(1.5, base))
            .style(LineStyle::dashed_loose())
    } else {
        inner.stroke(egui::Stroke::NONE)
    };
    plot_ui.polygon(inner);

    // Centre line (median for observed, mean for normative — both land in
    // `Summary::median`).
    plot_ui.line(
        Line::new("", PlotPoints::from(vec![[x0, s.median], [x1, s.median]]))
            .color(base)
            .width(1.5),
    );

    // Corner tag so the kind + source is legible without a legend.
    let label_x = x0 + 0.01 * (x1 - x0);
    plot_ui.text(
        Text::new(
            "",
            PlotPoint::new(label_x, y_top),
            format!("{tag} · {}", band.label),
        )
        .color(base)
        .anchor(egui::Align2::LEFT_TOP),
    );
}

/// D10: populates an overlay-picker submenu for `parameter`. Lists every
/// installed distribution whose measure parameters include `parameter`,
/// one radio entry per subgroup (`sex`) when the distribution declares
/// them, plus a "None" option. Writes the choice into `slot`.
fn refdist_overlay_submenu(
    ui: &mut egui::Ui,
    parameter: Option<&str>,
    slot: &mut Option<RefdistOverlay>,
) {
    if ui.radio(slot.is_none(), "None").clicked() {
        *slot = None;
        ui.close();
    }
    let dists = RefdistStore::user_default()
        .map(|s| s.list())
        .unwrap_or_default();
    let matching: Vec<&RefDist> = dists
        .iter()
        .filter(|rd| match parameter {
            Some(p) => rd
                .manifest
                .measure
                .parameters
                .iter()
                .any(|x| x.eq_ignore_ascii_case(p)),
            None => true,
        })
        .collect();
    if matching.is_empty() {
        ui.separator();
        ui.label(
            egui::RichText::new(
                "(no matching distributions installed —\nuse \"Install bundled reference data\")",
            )
            .weak(),
        );
        return;
    }
    ui.separator();
    for rd in matching {
        let id = rd.manifest.id.clone();
        let version = rd.manifest.version.clone();
        let sexes = &rd.manifest.population.sex;
        if sexes.is_empty() {
            let sel = RefdistOverlay {
                id,
                version,
                sex: None,
            };
            if ui
                .radio(slot.as_ref() == Some(&sel), &rd.manifest.title)
                .clicked()
            {
                *slot = Some(sel);
                ui.close();
            }
        } else {
            for sx in sexes {
                let sel = RefdistOverlay {
                    id: id.clone(),
                    version: version.clone(),
                    sex: Some(sx.clone()),
                };
                let label = format!("{} ({sx})", rd.manifest.title);
                if ui.radio(slot.as_ref() == Some(&sel), label).clicked() {
                    *slot = Some(sel);
                    ui.close();
                }
            }
        }
    }
}

/// D10: installs the bundled tier-1 reference distributions into the user
/// store (the first-run seeding the C8 entry left to cluster D). Copies
/// every distribution directory under the located `refdist-bundled/`;
/// idempotent (re-running overwrites in place). Returns how many were
/// installed.
fn seed_bundled_refdists() -> std::result::Result<usize, String> {
    let store = RefdistStore::user_default().map_err(|e| e.to_string())?;
    let dir = locate_bundled_refdist_dir()
        .ok_or_else(|| "bundled reference-distribution directory not found".to_string())?;
    let mut n = 0;
    for entry in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.join("refdist.toml").is_file() {
            store.install_from_dir(&path).map_err(|e| e.to_string())?;
            n += 1;
        }
    }
    Ok(n)
}

/// Locates the `refdist-bundled/` directory: an explicit
/// `SADDA_REFDIST_BUNDLED` override, then next to the executable (the
/// shipped layout), then the workspace copy relative to this crate (dev).
fn locate_bundled_refdist_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SADDA_REFDIST_BUNDLED") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return Some(p);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(cand) = exe.parent().map(|d| d.join("refdist-bundled")) {
            if cand.is_dir() {
                return Some(cand);
            }
        }
    }
    let dev = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../refdist-bundled");
    if dev.is_dir() {
        return Some(dev);
    }
    None
}

// ---------------------------------------------------------------------------
// Shared C5 plot helpers: cursor line + wheel-driven zoom / scroll
// ---------------------------------------------------------------------------

/// Draws the synced playback cursor on a plot. Bound the segment
/// by `[y_min, y_max]` so it spans the visible y-axis (e.g.
/// -1..1 for the waveform, 0..nyquist for the spectrogram).
fn draw_cursor_line(plot_ui: &mut egui_plot::PlotUi<'_>, cursor: f64, y_min: f64, y_max: f64) {
    let points = PlotPoints::from(vec![[cursor, y_min], [cursor, y_max]]);
    plot_ui.line(
        Line::new("cursor", points)
            .color(egui::Color32::from_rgb(230, 70, 70))
            .width(1.0),
    );
}

/// Selection band fill + edge colours, shared by the plot-lane and
/// tier-lane (painter) renderers so the band reads identically everywhere.
const SELECTION_FILL: egui::Color32 = egui::Color32::from_rgba_premultiplied(40, 62, 102, 70);
const SELECTION_EDGE: egui::Color32 = egui::Color32::from_rgb(96, 150, 235);

/// Draws the time-span selection as a translucent band + edge lines in an
/// `egui_plot` lane (waveform / spectrogram / measure tracks).
fn draw_selection_band(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    selection: Option<(f64, f64)>,
    y_min: f64,
    y_max: f64,
) {
    let Some((lo, hi)) = selection else {
        return;
    };
    plot_ui.polygon(
        egui_plot::Polygon::new(
            "selection",
            PlotPoints::from(vec![[lo, y_min], [hi, y_min], [hi, y_max], [lo, y_max]]),
        )
        .fill_color(SELECTION_FILL)
        .stroke(egui::Stroke::NONE),
    );
    for x in [lo, hi] {
        plot_ui.line(
            Line::new("sel_edge", PlotPoints::from(vec![[x, y_min], [x, y_max]]))
                .color(SELECTION_EDGE)
                .width(1.0),
        );
    }
}

/// Draws the selection band into a painter-based lane (tier lanes), mapping
/// time → x the same way the interval/point renderers do.
fn draw_selection_band_rect(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_start: f64,
    view_end: f64,
    selection: Option<(f64, f64)>,
) {
    let Some((lo, hi)) = selection else {
        return;
    };
    let x_per_second = rect.width() as f64 / (view_end - view_start).max(1e-6);
    let x0 = (rect.left() + ((lo - view_start) * x_per_second) as f32).max(rect.left());
    let x1 = (rect.left() + ((hi - view_start) * x_per_second) as f32).min(rect.right());
    if x1 <= x0 {
        return;
    }
    let band = egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x1, rect.bottom()));
    painter.rect_filled(band, 0.0, SELECTION_FILL);
    for x in [x0, x1] {
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(1.0, SELECTION_EDGE),
        );
    }
}

/// D10: per-formant dot colour (F1, F2, F3, …). The formants share one
/// lane, so this is the main place colour has to be *discriminated* —
/// hence the colourblind-safe alternate. Wraps for slots beyond the set.
fn formant_color(palette: PlotPalette, slot: usize) -> egui::Color32 {
    const DEFAULT: [egui::Color32; 5] = [
        egui::Color32::from_rgb(220, 60, 60),  // F1 red
        egui::Color32::from_rgb(230, 140, 40), // F2 orange
        egui::Color32::from_rgb(200, 80, 170), // F3 magenta
        egui::Color32::from_rgb(120, 90, 200), // F4 violet
        egui::Color32::from_rgb(90, 160, 90),  // F5 green
    ];
    // Okabe–Ito qualitative palette, ordered for maximum separation:
    // vermillion, bluish-green, reddish-purple, sky-blue, orange.
    const OKABE_ITO: [egui::Color32; 5] = [
        egui::Color32::from_rgb(213, 94, 0),
        egui::Color32::from_rgb(0, 158, 115),
        egui::Color32::from_rgb(204, 121, 167),
        egui::Color32::from_rgb(86, 180, 233),
        egui::Color32::from_rgb(230, 159, 0),
    ];
    let set = match palette {
        PlotPalette::Default => DEFAULT,
        PlotPalette::OkabeIto => OKABE_ITO,
    };
    set[slot % set.len()]
}

/// D10: shared scaffolding for a measure-track lane. Owns the plot
/// bounds (visible window in x, the lane's own range in y — so each
/// lane x-aligns with the waveform/spectrogram and the cursor draws a
/// single straight line through them all), draws the synced cursor,
/// and routes click-to-seek + wheel zoom/scroll back into `timeline`.
/// The lane-specific contour is drawn by `draw`. The x-axis is hidden
/// (`show_axes([false, true])`) to keep stacked lanes compact; the
/// waveform and spectrogram already carry the time ruler.
fn measure_lane(
    ui: &mut egui::Ui,
    plot_id: &str,
    timeline: &mut TimelineState,
    y_range: (f64, f64),
    y_axis_label: &str,
    draw: impl FnOnce(&mut egui_plot::PlotUi<'_>),
) {
    let view_start = timeline.view_start;
    let view_end = timeline.view_end;
    let cursor = timeline.cursor;
    let selection = timeline.selection;
    let (y_min, y_max) = y_range;
    let mut clicked_time: Option<f64> = None;
    let mut drag_start: Option<f64> = None;
    let mut drag_to: Option<f64> = None;
    let mut drag_ended = false;

    let plot_response = Plot::new(plot_id)
        .show_axes([false, true])
        .y_axis_label(y_axis_label)
        .y_axis_min_width(SIGNAL_LEFT_GUTTER)
        .allow_drag(false)
        .allow_zoom(false)
        .allow_scroll(false)
        .show(ui, |plot_ui| {
            // Own the bounds outright (matches the waveform pane): the
            // explicit set disables auto-fit so there's no edge margin
            // and the x-window matches the other lanes exactly.
            plot_ui.set_plot_bounds_x(view_start..=view_end);
            plot_ui.set_plot_bounds_y(y_min..=y_max);
            draw(plot_ui);
            draw_cursor_line(plot_ui, cursor, y_min, y_max);
            draw_selection_band(plot_ui, selection, y_min, y_max);

            let resp = plot_ui.response();
            if resp.drag_started() {
                drag_start = resp
                    .interact_pointer_pos()
                    .map(|p| plot_ui.plot_from_screen(p).x);
            }
            if resp.dragged() {
                drag_to = resp
                    .interact_pointer_pos()
                    .map(|p| plot_ui.plot_from_screen(p).x);
            }
            if resp.drag_stopped() {
                drag_ended = true;
            }
            if resp.clicked() {
                clicked_time = resp
                    .interact_pointer_pos()
                    .map(|p| plot_ui.plot_from_screen(p).x);
            }
        })
        .response;

    apply_lane_selection_drag(timeline, drag_start, drag_to, drag_ended, clicked_time);
    handle_zoom_and_scroll(&plot_response, timeline);
}

/// Wheel-driven zoom + shift-wheel scroll. Reads raw scroll deltas
/// from the response's hover state — egui_plot's own zoom/scroll is
/// disabled in every C5 pane so we own the input vocabulary.
fn handle_zoom_and_scroll(response: &egui::Response, timeline: &mut TimelineState) {
    if !response.hovered() {
        return;
    }
    let (scroll, modifiers, pointer) = response
        .ctx
        .input(|i| (i.smooth_scroll_delta, i.modifiers, i.pointer.hover_pos()));
    let dy = scroll.y;
    if dy == 0.0 {
        return;
    }
    // Estimate the time the pointer is over so zoom centres on it.
    // Egui doesn't expose plot-space coords outside of `show`, so we
    // approximate using the response rect + the timeline range.
    let pointer_time = pointer
        .map(|p| {
            let rect = response.rect;
            let width = rect.width().max(1.0) as f64;
            let rel = ((p.x - rect.left()) as f64).clamp(0.0, width);
            timeline.pixel_to_time(rel, width)
        })
        .unwrap_or(timeline.cursor);

    if modifiers.shift {
        // Shift+wheel: pan horizontally. Positive scroll = pan
        // right (matches IDE / GIMP convention on most platforms).
        let pan_secs = -(dy as f64 / 60.0) * 0.1 * timeline.view_range();
        timeline.scroll_by(pan_secs);
    } else {
        // Wheel only: zoom around the pointer. Positive scroll
        // (wheel up) zooms in; negative zooms out.
        let factor = if dy > 0.0 { 1.0 / 1.2 } else { 1.2 };
        timeline.zoom_at(pointer_time, factor);
    }
}

/// Applies a signal lane's drag/click outcome to the timeline: a drag
/// draws a span selection; a plain click clears any selection and moves
/// the cursor. Shared by the waveform + spectrogram panes.
fn apply_lane_selection_drag(
    timeline: &mut TimelineState,
    drag_start: Option<f64>,
    drag_to: Option<f64>,
    drag_ended: bool,
    clicked_time: Option<f64>,
) {
    if let Some(t) = drag_start {
        timeline.begin_selection(t);
    }
    if let Some(t) = drag_to {
        timeline.update_selection(t);
    }
    if drag_ended {
        timeline.end_selection();
    }
    if let Some(t) = clicked_time {
        timeline.set_cursor(t);
        timeline.clear_selection();
    }
}

// ---------------------------------------------------------------------------
// Tier-strip lane renderers
// ---------------------------------------------------------------------------

/// Pixel hit-zone width for grabbing an interval boundary.
const BOUNDARY_HIT_ZONE_PX: f32 = 6.0;
/// Minimum drag length (in seconds) before a drag-to-create commits.
/// Smaller drags are treated as plain clicks (no new interval).
const MIN_DRAFT_CREATE_SECONDS: f64 = 0.005;

/// Allocates one tier row's gutter + time-lane within `ui`, advancing the
/// cursor left-to-right, and returns their rects and responses.
///
/// The gutter reserves the signal plots' y-axis width (from the measured
/// `lane_geom = (gutter_width, data_area_width)`) via `allocate_exact_size`
/// — **not** `allocate_ui_with_layout`, which shrinks to its content and was
/// the cause of the tier-lane misalignment bug. With `item_spacing.x` zeroed,
/// the lane's left edge lands exactly at `row_left + gutter_w`, so the lane's
/// data area lines up with the egui_plot signal lanes that share the panel's
/// left edge. Before the waveform is measured (`None`), the fixed
/// `SIGNAL_LEFT_GUTTER` and remaining width are used.
///
/// Shared by the live tier strip and the `layout_tests` regression test so
/// the alignment-critical allocation is exercised by exactly one code path.
fn allocate_tier_row(
    ui: &mut egui::Ui,
    lane_geom: Option<(f32, f32)>,
) -> (egui::Rect, egui::Response, egui::Rect, egui::Response) {
    ui.spacing_mut().item_spacing.x = 0.0;
    let gutter_w = lane_geom
        .map(|(g, _)| g.max(40.0))
        .unwrap_or(SIGNAL_LEFT_GUTTER);
    let (gutter_rect, gutter_resp) = ui.allocate_exact_size(
        egui::Vec2::new(gutter_w, TIER_LANE_HEIGHT),
        egui::Sense::click(),
    );
    let avail = ui.available_size_before_wrap();
    let lane_w = lane_geom.map(|(_, w)| w).unwrap_or(avail.x);
    let (lane_rect, lane_resp) = ui.allocate_exact_size(
        egui::Vec2::new(lane_w, TIER_LANE_HEIGHT),
        egui::Sense::click_and_drag(),
    );
    (gutter_rect, gutter_resp, lane_rect, lane_resp)
}

/// The suffix the criteria engine appends to a target tier name for its
/// preview ("auto") tier — must match the engine's `preview_tier_name`.
const PREVIEW_TIER_SUFFIX: &str = " (auto)";

/// Whether a tier name is a criteria-engine preview tier (holds proposals).
fn is_preview_tier(name: &str) -> bool {
    name.ends_with(PREVIEW_TIER_SUFFIX)
}

/// A one-line provenance label for an annotation produced by a criterion run,
/// from the run's `processor_id` (`sadda.criteria.<name>`) and `started_at`
/// ISO timestamp. The `sadda.criteria.` prefix is stripped to the criterion
/// name; the timestamp is trimmed to whole seconds (drops the `T`, the
/// fractional part, and the trailing `Z`).
fn format_provenance_line(processor_id: &str, started_at: &str) -> String {
    let name = processor_id
        .strip_prefix("sadda.criteria.")
        .unwrap_or(processor_id);
    let when = started_at
        .split_once('.')
        .map(|(head, _frac)| head)
        .unwrap_or(started_at)
        .trim_end_matches('Z')
        .replace('T', " ");
    format!("↻ from criterion “{name}” · {when}")
}

/// One-line summary of a campaign target for the targets panel: its RoI, type,
/// and origin (the status is shown separately as an editable combo). E.g.
/// `[0.20–0.50s] phones · criterion`.
fn format_target_row(t: &sadda_engine::Target) -> String {
    format!(
        "[{:.2}–{:.2}s] {} · {}",
        t.start_seconds, t.end_seconds, t.target_type, t.source
    )
}

/// One-line "who's on this target" summary for the targets panel (slice S4b):
/// `→ alice(primary), bob(secondary)`, or `— unassigned` when empty.
fn format_assignment_summary(assignments: &[sadda_engine::Assignment]) -> String {
    if assignments.is_empty() {
        return "— unassigned".to_string();
    }
    let who: Vec<String> = assignments
        .iter()
        .map(|a| format!("{}({})", a.annotator, a.role))
        .collect();
    format!("→ {}", who.join(", "))
}

/// Status line for a completed package export (slice S4c).
fn format_export_summary(s: &sadda_engine::ExportSummary) -> String {
    format!(
        "Exported {} bundle(s), {} target(s), {} assignment(s) for “{}” → {}",
        s.bundles,
        s.targets,
        s.assignments,
        s.annotator,
        s.path.display()
    )
}

/// Campaign progress one-liner for the targets panel (slice S5).
fn format_target_progress(p: &sadda_engine::ProgressCounts) -> String {
    format!(
        "Progress: {}/{} done · {} in progress · {} flagged · {} to do",
        p.done,
        p.total,
        p.in_progress,
        p.flagged,
        p.unassigned + p.assigned
    )
}

/// One annotator's completeness line for the dashboard (slice S6).
fn format_annotator_progress(a: &sadda_engine::AnnotatorProgress) -> String {
    format!(
        "{}: {} done · {} in progress · {} to do",
        a.annotator, a.done, a.in_progress, a.assigned
    )
}

/// QA findings line for a tier on the dashboard (slice S6).
fn format_qa_report(q: &sadda_engine::QaReport) -> String {
    format!(
        "{} annotations · {} out-of-vocab · {} missing · {} overlaps",
        q.n_annotations, q.out_of_vocab, q.missing_label, q.overlaps
    )
}

/// Per-tier rubric-change impact line for the dashboard (slice S6b).
fn format_tier_impact(t: &sadda_engine::TierImpact) -> String {
    format!(
        "{}: +[{}] −[{}] · {} to revisit",
        t.tier_name,
        t.vocab_added.join(", "),
        t.vocab_removed.join(", "),
        t.affected_annotations
    )
}

/// Compact agreement readout for the targets panel (slice S5): κ + label
/// agreement, unit match counts, boundary deviation/tolerance, and frame κ.
fn format_agreement_report(r: &sadda_engine::AgreementReport) -> String {
    format!(
        "κ={:.2} ({:.0}% labels) · {} matched / {}+{} extra · Δbound {:.0}ms ({:.0}% ≤{:.0}ms) · frame κ={:.2}",
        r.cohen_kappa,
        r.percent_label_agreement * 100.0,
        r.n_matched,
        r.n_only_a,
        r.n_only_b,
        r.mean_abs_boundary_diff * 1000.0,
        r.boundary_within_tolerance * 100.0,
        r.boundary_tolerance_seconds * 1000.0,
        r.frame_kappa,
    )
}

/// Status line for a completed package import (slice S4c).
fn format_import_summary(s: &sadda_engine::ImportSummary) -> String {
    format!(
        "Imported “{}”: {} bundle(s) matched, {} tier(s) / {} annotation(s) landed, {} assignment(s) done",
        s.annotator,
        s.bundles_matched,
        s.tiers_imported,
        s.annotations_imported,
        s.assignments_marked_done
    )
}

/// Whether `label` is out of the controlled `vocab`: non-empty and absent.
/// An empty label, or an empty vocabulary, is never "out of vocab".
fn is_out_of_vocab(vocab: &[String], label: &str) -> bool {
    !label.is_empty() && !vocab.is_empty() && !vocab.iter().any(|v| v == label)
}

/// A stable background tint for an annotation status, chosen by the status's
/// position in the rubric's status list so progress reads at a glance.
/// `None` (or a status not in the list) yields no tint.
fn status_tint(status: Option<&str>, statuses: &[String]) -> Option<egui::Color32> {
    let s = status?;
    let idx = statuses.iter().position(|v| v == s)?;
    // Muted, distinguishable tints; cycled if more statuses than colors.
    const PALETTE: [egui::Color32; 6] = [
        egui::Color32::from_rgb(120, 120, 130), // 0 — neutral / draft-ish
        egui::Color32::from_rgb(90, 150, 100),  // 1 — green / done-ish
        egui::Color32::from_rgb(190, 150, 70),  // 2 — amber / flagged-ish
        egui::Color32::from_rgb(150, 100, 170), // 3 — violet
        egui::Color32::from_rgb(90, 140, 180),  // 4 — blue
        egui::Color32::from_rgb(180, 110, 110), // 5 — red
    ];
    Some(PALETTE[idx % PALETTE.len()])
}

#[allow(clippy::too_many_arguments)]
fn render_interval_lane(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_start: f64,
    view_end: f64,
    tier_id: i64,
    rows: &[sadda_engine::Interval],
    status_palette: &[String],
    is_preview: bool,
    selection: Option<AnnotationSelection>,
    response: &egui::Response,
    draft: &DraftEdit,
    new_selection: &mut Option<AnnotationSelection>,
    new_cursor: &mut Option<f64>,
    new_draft_action: &mut Option<DraftAction>,
    label_edit_request: &mut Option<LabelEdit>,
    request_delete: &mut bool,
    open_annotation_panel: &mut bool,
) {
    let view_range = (view_end - view_start).max(1e-6);
    let lane_width = rect.width() as f64;
    let x_per_second = lane_width / view_range;
    // Proposals on a criteria preview ("auto") tier render in a distinct
    // amber so they read as not-yet-accepted.
    let base_fill = if is_preview {
        egui::Color32::from_rgb(190, 150, 70)
    } else {
        egui::Color32::from_rgb(82, 138, 198)
    };
    let selected_fill = egui::Color32::from_rgb(160, 200, 250);
    let draft_fill = egui::Color32::from_rgba_premultiplied(60, 180, 120, 160);
    let text_color = egui::Color32::WHITE;

    // Draw existing intervals first.
    for r in rows {
        // Cull intervals entirely outside the view.
        if r.end_seconds < view_start || r.start_seconds > view_end {
            continue;
        }
        let x0 = rect.left() + ((r.start_seconds - view_start) * x_per_second) as f32;
        let x1 = rect.left() + ((r.end_seconds - view_start) * x_per_second) as f32;
        let item_rect = egui::Rect::from_min_max(
            egui::Pos2::new(x0.max(rect.left()), rect.top() + 2.0),
            egui::Pos2::new(x1.min(rect.right()), rect.bottom() - 2.0),
        );
        let is_selected = matches!(
            selection,
            Some(AnnotationSelection::Interval { tier_id: t, annotation_id: a })
                if t == tier_id && a == r.id
        );
        painter.rect_filled(
            item_rect,
            2.0,
            if is_selected {
                selected_fill
            } else {
                base_fill
            },
        );
        if is_selected {
            painter.rect_stroke(
                item_rect,
                2.0,
                egui::Stroke::new(1.5, egui::Color32::WHITE),
                egui::StrokeKind::Inside,
            );
        }
        // Status indicator: a thin colored strip along the interval's bottom
        // edge, tinted by the annotation's rubric status (if any).
        if let Some(tint) = status_tint(r.status.as_deref(), status_palette) {
            let strip = egui::Rect::from_min_max(
                egui::Pos2::new(item_rect.left(), item_rect.bottom() - 3.0),
                egui::Pos2::new(item_rect.right(), item_rect.bottom()),
            );
            painter.rect_filled(strip, 0.0, tint);
        }
        if let Some(label) = &r.label {
            if !label.is_empty() && item_rect.width() > 20.0 {
                painter.text(
                    item_rect.left_center() + egui::Vec2::new(4.0, 0.0),
                    egui::Align2::LEFT_CENTER,
                    truncate_label(label, TIER_LABEL_MAX_CHARS),
                    egui::FontId::proportional(11.0),
                    text_color,
                );
            }
        }
    }

    // Draw the live draft preview on top of existing intervals.
    if let DraftEdit::Creating {
        tier_id: t,
        start_time,
        current_time,
    } = draft
        && *t == tier_id
    {
        let (lo, hi) = if start_time <= current_time {
            (*start_time, *current_time)
        } else {
            (*current_time, *start_time)
        };
        if hi > view_start && lo < view_end {
            let x0 = rect.left() + ((lo - view_start) * x_per_second) as f32;
            let x1 = rect.left() + ((hi - view_start) * x_per_second) as f32;
            let preview = egui::Rect::from_min_max(
                egui::Pos2::new(x0.max(rect.left()), rect.top() + 2.0),
                egui::Pos2::new(x1.min(rect.right()), rect.bottom() - 2.0),
            );
            painter.rect_filled(preview, 2.0, draft_fill);
            painter.rect_stroke(
                preview,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(40, 140, 90)),
                egui::StrokeKind::Inside,
            );
        }
    }
    if let DraftEdit::Resizing {
        tier_id: t,
        annotation_id,
        edge,
        fixed_time,
        current_time,
    } = draft
        && *t == tier_id
        && let Some(r) = rows.iter().find(|r| r.id == *annotation_id)
    {
        let _ = r;
        let (lo, hi) = match edge {
            BoundaryEdge::Start => {
                if current_time <= fixed_time {
                    (*current_time, *fixed_time)
                } else {
                    (*fixed_time, *current_time)
                }
            }
            BoundaryEdge::End => {
                if fixed_time <= current_time {
                    (*fixed_time, *current_time)
                } else {
                    (*current_time, *fixed_time)
                }
            }
        };
        if hi > view_start && lo < view_end {
            let x0 = rect.left() + ((lo - view_start) * x_per_second) as f32;
            let x1 = rect.left() + ((hi - view_start) * x_per_second) as f32;
            let preview = egui::Rect::from_min_max(
                egui::Pos2::new(x0.max(rect.left()), rect.top() + 2.0),
                egui::Pos2::new(x1.min(rect.right()), rect.bottom() - 2.0),
            );
            painter.rect_stroke(
                preview,
                2.0,
                egui::Stroke::new(2.0, egui::Color32::from_rgb(40, 140, 90)),
                egui::StrokeKind::Inside,
            );
        }
    }

    // ----- Mouse interaction dispatch ---------------------------------

    // Right-click an interval → context menu (edit / delete). Right-click
    // also selects the interval under the pointer, so the menu's target stays
    // stable while it's open: this frame we use the just-clicked id, on later
    // frames the (now-applied) selection. `hover_pos` can't be used — it moves
    // onto the menu popup and would blank the target.
    let mut clicked_target: Option<i64> = None;
    if response.secondary_clicked() {
        if let Some(p) = response.interact_pointer_pos() {
            if let Some(r) = rows.iter().find(|r| {
                let x0 = rect.left() + ((r.start_seconds - view_start) * x_per_second) as f32;
                let x1 = rect.left() + ((r.end_seconds - view_start) * x_per_second) as f32;
                p.x >= x0 && p.x <= x1 && rect.y_range().contains(p.y)
            }) {
                clicked_target = Some(r.id);
                *new_selection = Some(AnnotationSelection::Interval {
                    tier_id,
                    annotation_id: r.id,
                });
            }
        }
    }
    let target_id = clicked_target.or(match selection {
        Some(AnnotationSelection::Interval {
            tier_id: t,
            annotation_id,
        }) if t == tier_id => Some(annotation_id),
        _ => None,
    });
    let menu_target = target_id.and_then(|id| rows.iter().find(|r| r.id == id));
    response.context_menu(|ui| match menu_target {
        Some(r) => {
            ui.label(egui::RichText::new(r.label.as_deref().unwrap_or("(no label)")).strong());
            if ui.button("Edit annotation…").clicked() {
                *new_selection = Some(AnnotationSelection::Interval {
                    tier_id,
                    annotation_id: r.id,
                });
                *label_edit_request = Some(LabelEdit {
                    tier_id,
                    annotation_id: r.id,
                    kind: LabelEditKind::Interval,
                    text: r.label.clone().unwrap_or_default(),
                    status: r.status.clone(),
                    note: r.note.clone().unwrap_or_default(),
                    just_started: true,
                    ..Default::default()
                });
                ui.close();
            }
            if ui.button("Delete interval").clicked() {
                *new_selection = Some(AnnotationSelection::Interval {
                    tier_id,
                    annotation_id: r.id,
                });
                *request_delete = true;
                ui.close();
            }
        }
        None => {
            ui.label("Right-click an interval to edit or delete it");
        }
    });

    // Double-click on an interval body → select it and open the inline
    // Annotation panel (the modal-free editor).
    if response.double_clicked() {
        if let Some(p) = response.interact_pointer_pos() {
            for r in rows {
                let x0 = rect.left() + ((r.start_seconds - view_start) * x_per_second) as f32;
                let x1 = rect.left() + ((r.end_seconds - view_start) * x_per_second) as f32;
                if p.x >= x0 && p.x <= x1 && rect.y_range().contains(p.y) {
                    *new_selection = Some(AnnotationSelection::Interval {
                        tier_id,
                        annotation_id: r.id,
                    });
                    *open_annotation_panel = true;
                    return;
                }
            }
        }
    }

    // Mouse-down begins a drag. Disambiguate via the down position:
    // near a boundary → resize; in a body → no draft (selection
    // handled by plain click later); empty → create.
    if response.drag_started_by(egui::PointerButton::Primary) {
        if let Some(p) = response.interact_pointer_pos() {
            let t_at = view_start + ((p.x - rect.left()) as f64) / x_per_second;
            // Boundary check first — wins over body if both apply.
            for r in rows {
                let x0 = rect.left() + ((r.start_seconds - view_start) * x_per_second) as f32;
                let x1 = rect.left() + ((r.end_seconds - view_start) * x_per_second) as f32;
                if (p.x - x0).abs() <= BOUNDARY_HIT_ZONE_PX {
                    *new_draft_action = Some(DraftAction::Start(DraftEdit::Resizing {
                        tier_id,
                        annotation_id: r.id,
                        edge: BoundaryEdge::Start,
                        fixed_time: r.end_seconds,
                        current_time: r.start_seconds,
                    }));
                    return;
                }
                if (p.x - x1).abs() <= BOUNDARY_HIT_ZONE_PX {
                    *new_draft_action = Some(DraftAction::Start(DraftEdit::Resizing {
                        tier_id,
                        annotation_id: r.id,
                        edge: BoundaryEdge::End,
                        fixed_time: r.start_seconds,
                        current_time: r.end_seconds,
                    }));
                    return;
                }
            }
            // Not on a boundary — was it on an existing body? If so,
            // don't start a draft (lets the click semantics win).
            let on_body = rows.iter().any(|r| {
                let x0 = rect.left() + ((r.start_seconds - view_start) * x_per_second) as f32;
                let x1 = rect.left() + ((r.end_seconds - view_start) * x_per_second) as f32;
                p.x >= x0 && p.x <= x1
            });
            if !on_body {
                *new_draft_action = Some(DraftAction::Start(DraftEdit::Creating {
                    tier_id,
                    start_time: t_at,
                    current_time: t_at,
                }));
                return;
            }
        }
    }

    // Drag-in-progress: update the active draft if it belongs to
    // this tier. Setting Update overwrites the matching draft only.
    if response.dragged() && matches_this_tier(draft, tier_id) {
        if let Some(p) = response.interact_pointer_pos() {
            let t = view_start + ((p.x - rect.left()) as f64) / x_per_second;
            *new_draft_action = Some(DraftAction::Update(t));
        }
    }

    if response.drag_stopped() && matches_this_tier(draft, tier_id) {
        *new_draft_action = Some(DraftAction::Commit);
        return;
    }

    // Plain click without drag → B4 selection + C5 cursor positioning.
    // Skip if a drag is in progress (mouse-down → drag → mouse-up).
    if response.clicked() && !matches!(draft, DraftEdit::None) {
        return;
    }
    if response.clicked() {
        if let Some(p) = response.interact_pointer_pos() {
            for r in rows {
                let x0 = rect.left() + ((r.start_seconds - view_start) * x_per_second) as f32;
                let x1 = rect.left() + ((r.end_seconds - view_start) * x_per_second) as f32;
                if p.x >= x0 && p.x <= x1 && rect.y_range().contains(p.y) {
                    *new_selection = Some(AnnotationSelection::Interval {
                        tier_id,
                        annotation_id: r.id,
                    });
                    *new_cursor = Some(r.start_seconds);
                    return;
                }
            }
        }
    }
}

/// Cross-frame draft mutation requested by a lane. Translated to
/// `SaddaApp.draft_edit` mutations by `tier_strip_pane` after the
/// per-lane render loop.
#[derive(Debug, Clone)]
enum DraftAction {
    Start(DraftEdit),
    Update(f64),
    Commit,
    /// Click-to-add for point lanes (D7). Distinct from `Start +
    /// Commit` because there's no draft state to inspect — the
    /// caller knows the tier + time at click-resolve time.
    AddPointNow {
        tier_id: i64,
        time: f64,
    },
}

fn matches_this_tier(draft: &DraftEdit, tier_id: i64) -> bool {
    match draft {
        DraftEdit::None => false,
        DraftEdit::Creating { tier_id: t, .. }
        | DraftEdit::Resizing { tier_id: t, .. }
        | DraftEdit::MovingPoint { tier_id: t, .. } => *t == tier_id,
    }
}

/// Pixel hit-zone width for grabbing a point tick.
const POINT_HIT_ZONE_PX: f32 = 6.0;
/// Minimum drag distance (in seconds) for a point-move to count as
/// a real change. Tiny mouse jitter shouldn't write audit rows.
const MIN_POINT_MOVE_SECONDS: f64 = 0.001;

#[allow(clippy::too_many_arguments)]
fn render_point_lane(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_start: f64,
    view_end: f64,
    tier_id: i64,
    rows: &[sadda_engine::Point],
    status_palette: &[String],
    is_preview: bool,
    selection: Option<AnnotationSelection>,
    response: &egui::Response,
    draft: &DraftEdit,
    new_selection: &mut Option<AnnotationSelection>,
    new_cursor: &mut Option<f64>,
    new_draft_action: &mut Option<DraftAction>,
    label_edit_request: &mut Option<LabelEdit>,
    request_delete: &mut bool,
    open_annotation_panel: &mut bool,
) {
    let view_range = (view_end - view_start).max(1e-6);
    let lane_width = rect.width() as f64;
    let x_per_second = lane_width / view_range;
    // Proposals on a criteria preview ("auto") tier render in a distinct
    // violet so they read as not-yet-accepted.
    let base_color = if is_preview {
        egui::Color32::from_rgb(175, 120, 205)
    } else {
        egui::Color32::from_rgb(230, 180, 70)
    };
    let selected_color = egui::Color32::from_rgb(255, 220, 120);
    let draft_color = egui::Color32::from_rgb(60, 200, 130);

    // Draw existing ticks.
    for p in rows {
        if p.time_seconds < view_start || p.time_seconds > view_end {
            continue;
        }
        // If this point is being moved, skip the static render —
        // the moved position is drawn below in the draft pass.
        if matches!(
            draft,
            DraftEdit::MovingPoint { tier_id: t, annotation_id: a, .. }
                if *t == tier_id && *a == p.id
        ) {
            continue;
        }
        let x = rect.left() + ((p.time_seconds - view_start) * x_per_second) as f32;
        let is_selected = matches!(
            selection,
            Some(AnnotationSelection::Point { tier_id: t, annotation_id: a })
                if t == tier_id && a == p.id
        );
        let stroke_width = if is_selected { 2.0 } else { 1.0 };
        let colour = if is_selected {
            selected_color
        } else {
            base_color
        };
        painter.line_segment(
            [
                egui::Pos2::new(x, rect.top() + 2.0),
                egui::Pos2::new(x, rect.bottom() - 2.0),
            ],
            egui::Stroke::new(stroke_width, colour),
        );
        // Status indicator: a small filled square at the tick's base, tinted
        // by the point's rubric status (if any).
        if let Some(tint) = status_tint(p.status.as_deref(), status_palette) {
            painter.rect_filled(
                egui::Rect::from_center_size(
                    egui::Pos2::new(x, rect.bottom() - 3.0),
                    egui::Vec2::splat(5.0),
                ),
                1.0,
                tint,
            );
        }
        if let Some(label) = &p.label
            && !label.is_empty()
        {
            painter.text(
                egui::Pos2::new(x + 3.0, rect.top() + 2.0),
                egui::Align2::LEFT_TOP,
                truncate_label(label, TIER_LABEL_MAX_CHARS),
                egui::FontId::proportional(11.0),
                colour,
            );
        }
    }

    // Draw the live moving-point preview (if it belongs to this tier).
    if let DraftEdit::MovingPoint {
        tier_id: t,
        current_time,
        ..
    } = draft
        && *t == tier_id
        && *current_time >= view_start
        && *current_time <= view_end
    {
        let x = rect.left() + ((current_time - view_start) * x_per_second) as f32;
        painter.line_segment(
            [
                egui::Pos2::new(x, rect.top() + 2.0),
                egui::Pos2::new(x, rect.bottom() - 2.0),
            ],
            egui::Stroke::new(2.0, draft_color),
        );
    }

    // ----- Mouse interaction dispatch ---------------------------------

    // Right-click a point → context menu (edit / delete). Right-click also
    // selects the point, so the menu's target stays stable while open (see
    // the interval-lane note on why `hover_pos` can't be used here).
    let mut clicked_target: Option<i64> = None;
    if response.secondary_clicked() {
        if let Some(p) = response.interact_pointer_pos() {
            if let Some(row) = rows.iter().find(|row| {
                let x = rect.left() + ((row.time_seconds - view_start) * x_per_second) as f32;
                (p.x - x).abs() <= POINT_HIT_ZONE_PX && rect.y_range().contains(p.y)
            }) {
                clicked_target = Some(row.id);
                *new_selection = Some(AnnotationSelection::Point {
                    tier_id,
                    annotation_id: row.id,
                });
            }
        }
    }
    let target_id = clicked_target.or(match selection {
        Some(AnnotationSelection::Point {
            tier_id: t,
            annotation_id,
        }) if t == tier_id => Some(annotation_id),
        _ => None,
    });
    let menu_target = target_id.and_then(|id| rows.iter().find(|r| r.id == id));
    response.context_menu(|ui| match menu_target {
        Some(row) => {
            ui.label(egui::RichText::new(row.label.as_deref().unwrap_or("(no label)")).strong());
            if ui.button("Edit annotation…").clicked() {
                *new_selection = Some(AnnotationSelection::Point {
                    tier_id,
                    annotation_id: row.id,
                });
                *label_edit_request = Some(LabelEdit {
                    tier_id,
                    annotation_id: row.id,
                    kind: LabelEditKind::Point,
                    text: row.label.clone().unwrap_or_default(),
                    status: row.status.clone(),
                    note: row.note.clone().unwrap_or_default(),
                    just_started: true,
                    ..Default::default()
                });
                ui.close();
            }
            if ui.button("Delete point").clicked() {
                *new_selection = Some(AnnotationSelection::Point {
                    tier_id,
                    annotation_id: row.id,
                });
                *request_delete = true;
                ui.close();
            }
        }
        None => {
            ui.label("Right-click a point to edit or delete it");
        }
    });

    // Double-click on an existing point → label edit.
    if response.double_clicked() {
        if let Some(p) = response.interact_pointer_pos() {
            for row in rows {
                let x = rect.left() + ((row.time_seconds - view_start) * x_per_second) as f32;
                if (p.x - x).abs() <= POINT_HIT_ZONE_PX && rect.y_range().contains(p.y) {
                    *new_selection = Some(AnnotationSelection::Point {
                        tier_id,
                        annotation_id: row.id,
                    });
                    *open_annotation_panel = true;
                    return;
                }
            }
        }
    }

    // Mouse-down: if near an existing tick, start MovingPoint;
    // otherwise no-op (empty-click handled below via response.clicked).
    if response.drag_started_by(egui::PointerButton::Primary) {
        if let Some(p) = response.interact_pointer_pos() {
            for row in rows {
                let x = rect.left() + ((row.time_seconds - view_start) * x_per_second) as f32;
                if (p.x - x).abs() <= POINT_HIT_ZONE_PX {
                    *new_draft_action = Some(DraftAction::Start(DraftEdit::MovingPoint {
                        tier_id,
                        annotation_id: row.id,
                        original_time: row.time_seconds,
                        current_time: row.time_seconds,
                    }));
                    return;
                }
            }
            // No tick under pointer — fall through; the click handler
            // below will add a new point on mouse-up.
        }
    }

    if response.dragged() && matches_this_tier(draft, tier_id) {
        if let Some(p) = response.interact_pointer_pos() {
            let t = view_start + ((p.x - rect.left()) as f64) / x_per_second;
            *new_draft_action = Some(DraftAction::Update(t));
        }
    }

    if response.drag_stopped() && matches_this_tier(draft, tier_id) {
        *new_draft_action = Some(DraftAction::Commit);
        return;
    }

    // Plain click (no drag): if it hit an existing tick, select +
    // position cursor. Otherwise, add a new point at the click time.
    if response.clicked() && !matches!(draft, DraftEdit::None) {
        return;
    }
    if response.clicked() {
        if let Some(p) = response.interact_pointer_pos() {
            for row in rows {
                let x = rect.left() + ((row.time_seconds - view_start) * x_per_second) as f32;
                if (p.x - x).abs() <= POINT_HIT_ZONE_PX && rect.y_range().contains(p.y) {
                    *new_selection = Some(AnnotationSelection::Point {
                        tier_id,
                        annotation_id: row.id,
                    });
                    *new_cursor = Some(row.time_seconds);
                    return;
                }
            }
            // Empty space — add a new point at the click time. This
            // overrides B4's click-to-position-cursor behaviour
            // *only for point lanes* (matches Praat's PointTier
            // editor convention).
            if rect.contains(p) {
                let t = view_start + ((p.x - rect.left()) as f64) / x_per_second;
                *new_draft_action = Some(DraftAction::Start(DraftEdit::MovingPoint {
                    tier_id,
                    annotation_id: -1, // sentinel: "new, not yet committed"
                    original_time: t,
                    current_time: t,
                }));
                // Immediately commit — no drag needed for a click-add.
                *new_draft_action = Some(DraftAction::AddPointNow { tier_id, time: t });
            }
        }
    }
}

impl eframe::App for SaddaApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.persisted);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // `SADDA_DEBUG` aids: egui's hover-debug overlays (widget rects on
        // hover) plus F12 screenshot capture to a PNG we can read back. All
        // no-ops when the env var is unset. See `debug.rs`.
        if debug::enabled() {
            let ctx = ui.ctx();
            ctx.set_debug_on_hover(true);
            if ctx.input(|i| i.key_pressed(egui::Key::F12)) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                debug::log("screenshot requested (F12)");
            }
            let shots: Vec<std::sync::Arc<egui::ColorImage>> = ctx.input(|i| {
                i.raw
                    .events
                    .iter()
                    .filter_map(|e| match e {
                        egui::Event::Screenshot { image, .. } => Some(image.clone()),
                        _ => None,
                    })
                    .collect()
            });
            for img in &shots {
                debug::save_screenshot(img);
            }
        }

        self.apply_theme(ui.ctx());
        // Accessibility: apply the persisted UI zoom factor. Idempotent
        // and cheap — egui only relays out when the value actually
        // changes — so setting it every frame keeps it in force without
        // tracking dirtiness ourselves.
        ui.ctx().set_zoom_factor(self.persisted.ui_scale);

        // Drive the playback-cursor advance before any pane
        // renders, so they all see the same `timeline.cursor` this
        // frame. Repaint continuously while playing so the cursor
        // line stays in sync without user input.
        self.poll_playback();
        if self.playback.is_some() {
            ui.ctx().request_repaint();
        }

        // When a text field (the script editor, an inline label edit,
        // the command-palette query, or a dialog input) holds keyboard
        // focus, bare single-key shortcuts must NOT fire — the keystroke
        // belongs to the text field. Without this gate, typing in the
        // script panel toggled transport on Space and deleted the
        // selected annotation on Backspace/Delete instead of editing
        // text. Modifier combos (Ctrl/Cmd+Enter to run, Ctrl/Cmd+P for
        // the palette) are intentionally still allowed through so they
        // work while the editor is focused.
        let text_editing = ui.ctx().text_edit_focused();

        // Spacebar toggles transport. `consume_key` ensures the
        // press doesn't fall through to any focused widget. Skipped
        // while a text field is focused so Space reaches the editor.
        if !text_editing
            && ui
                .ctx()
                .input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Space))
        {
            self.toggle_playback();
        }

        // Arrow keys scrub the view left / right (a quarter-window per
        // press); Home / End snap to the start / end of the file. This
        // is the discoverable, trackpad-free way to move through a long
        // recording while zoomed in — shift+scroll-wheel pans too (see
        // handle_zoom_and_scroll). Skipped while any widget has focus or
        // a label is being edited, so the keys reach the editor instead.
        // scroll_by clamps against the bundle bounds, so a full-width
        // pan at either edge is a harmless no-op.
        if self.selected_bundle_id.is_some()
            && self.timeline.duration > 0.0
            && self.label_edit.is_none()
            && ui.ctx().memory(|m| m.focused().is_none())
        {
            let step = self.timeline.view_range() * 0.25;
            let ctx = ui.ctx();
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowRight)) {
                self.timeline.scroll_by(step);
            }
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowLeft)) {
                self.timeline.scroll_by(-step);
            }
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Home)) {
                self.timeline.scroll_by(-self.timeline.duration);
            }
            if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::End)) {
                self.timeline.scroll_by(self.timeline.duration);
            }
        }

        // E8: Ctrl/Cmd+Enter runs the script buffer when the panel
        // is open. `Modifiers::COMMAND` covers Ctrl on Linux/Windows
        // and Cmd on macOS via egui's platform-aware handling.
        if self.persisted.script_panel_open
            && ui
                .ctx()
                .input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter))
        {
            self.run_script_buffer();
        }

        // E9: Ctrl/Cmd+P opens the command palette.
        if ui
            .ctx()
            .input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::P))
        {
            self.command_palette_open = !self.command_palette_open;
            self.command_palette_query.clear();
        }

        // Delete / Backspace removes the selected interval. Skip
        // whenever a text field is focused (inline label edit, script
        // editor, dialogs) — those keys need to reach the TextEdit
        // instead. (`text_editing` already covers the inline label
        // edit; the explicit `label_edit.is_none()` is kept as a
        // belt-and-braces guard on that distinct editing mode.)
        if self.label_edit.is_none() && !text_editing {
            let delete_pressed = ui.ctx().input_mut(|i| {
                i.consume_key(egui::Modifiers::NONE, egui::Key::Delete)
                    || i.consume_key(egui::Modifiers::NONE, egui::Key::Backspace)
            });
            if delete_pressed {
                self.delete_selected_annotation();
            }
        }

        // Inline label-edit modal. Rendered before the menu so it
        // overlays everything; commit / cancel logic lives inside.
        self.label_edit_window(ui.ctx());
        self.rubric_editor_window(ui.ctx());
        self.criteria_editor_window(ui.ctx());
        self.targets_panel_window(ui.ctx());
        self.dashboard_window(ui.ctx());

        // E9 command palette. Same overlay pattern.
        self.command_palette_window(ui.ctx());

        // H1 live-recording modal.
        self.render_record_dialog(ui.ctx());

        // H1 bundle-delete confirmation modal.
        self.render_pending_delete(ui.ctx());

        // Bundle-rename modal.
        self.render_pending_rename(ui.ctx());

        // Tier-lifecycle modals (create / rename / delete).
        self.render_new_tier(ui.ctx());
        self.render_pending_tier_rename(ui.ctx());
        self.render_pending_tier_delete(ui.ctx());

        // A1 provenance & citations modal.
        self.render_provenance_view(ui.ctx());

        egui::Panel::top("menu").show_inside(ui, |ui| self.menu_bar(ui));

        egui::Panel::bottom("status").show_inside(ui, |ui| {
            ui.horizontal(|ui| {
                if let AppState::ProjectLoaded { name, root, .. } = &self.app_state {
                    ui.label(format!("Project: {name}  ·  {}", root.display()));
                    ui.separator();
                }
                // Research-use-only labeling (clinical-regulatory posture 3,
                // Phase 3 A2): always visible, never a clinical claim.
                ui.label(
                    egui::RichText::new("For research, education, and non-diagnostic use only")
                        .weak(),
                );
            });
        });

        if let Some(msg) = self.error.clone() {
            egui::Panel::bottom("error").show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::from_rgb(220, 80, 80), &msg);
                    if ui.button("Dismiss").clicked() {
                        self.error = None;
                    }
                });
            });
        }

        if let Some(msg) = self.info.clone() {
            egui::Panel::bottom("info").show_inside(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::from_rgb(60, 170, 90), &msg);
                    if ui.button("Dismiss").clicked() {
                        self.info = None;
                    }
                });
            });
        }

        // E8 script panel — bottom panel above status/error, shown
        // only when toggled on via View → Show Script Panel AND a
        // project is loaded (no point running scripts otherwise).
        if self.persisted.script_panel_open
            && matches!(self.app_state, AppState::ProjectLoaded { .. })
        {
            egui::Panel::bottom("script_panel")
                .resizable(true)
                .default_size(220.0)
                .min_size(120.0)
                .show_inside(ui, |ui| self.script_panel(ui));
        }

        match &self.app_state {
            AppState::NoProject => {
                egui::CentralPanel::default().show_inside(ui, |ui| self.welcome(ui));
            }
            AppState::ProjectLoaded { .. } => {
                egui::Panel::left("bundle_sidebar")
                    .resizable(true)
                    .default_size(200.0)
                    .min_size(120.0)
                    .show_inside(ui, |ui| self.bundle_sidebar(ui));
                // Inline Annotation panel (modal-free editor for the selected
                // annotation). Shown as its own right column.
                if self.persisted.annotation_panel_open {
                    egui::Panel::right("annotation_panel")
                        .resizable(true)
                        .default_size(280.0)
                        .min_size(200.0)
                        .show_inside(ui, |ui| {
                            egui::ScrollArea::vertical().show(ui, |ui| self.annotation_panel(ui));
                        });
                }
                // D10: right-side Reference panel. Refresh its cached data
                // before showing it, so a changed selection lands next frame.
                if self.persisted.reference_panel_open {
                    self.rebuild_reference_if_stale();
                    egui::Panel::right("reference_panel")
                        .resizable(true)
                        .default_size(320.0)
                        .min_size(220.0)
                        .show_inside(ui, |ui| {
                            egui::ScrollArea::vertical().show(ui, |ui| self.reference_panel(ui))
                        });
                }
                egui::CentralPanel::default().show_inside(ui, |ui| self.bundle_content_pane(ui));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

impl SaddaApp {
    fn apply_theme(&self, ctx: &egui::Context) {
        match self.persisted.theme {
            ThemePref::System => {
                // egui follows the OS preference by default; nothing to do.
            }
            ThemePref::Light => ctx.set_visuals(egui::Visuals::light()),
            ThemePref::Dark => ctx.set_visuals(egui::Visuals::dark()),
        }
    }

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            self.file_menu(ui);
            self.annotate_menu(ui);
            self.view_menu(ui);
            self.help_menu(ui);
        });
    }

    fn annotate_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Annotate", |ui| {
            let project_open = matches!(self.app_state, AppState::ProjectLoaded { .. });
            if ui
                .add_enabled(project_open, egui::Button::new("Rubric…"))
                .on_disabled_hover_text("Open or create a project first")
                .on_hover_text(
                    "Edit the annotation rubric: guidelines, status vocabulary, \
                     and per-tier controlled vocabularies",
                )
                .clicked()
            {
                ui.close();
                if self.rubric_editor.is_none() {
                    self.rubric_editor = Some(RubricEditor::default());
                }
            }
            if ui
                .add_enabled(project_open, egui::Button::new("Criteria…"))
                .on_disabled_hover_text("Open or create a project first")
                .on_hover_text(
                    "Define re-runnable rules that propose annotations on a preview tier",
                )
                .clicked()
            {
                ui.close();
                if self.criteria_editor.is_none() {
                    let mut ed = CriteriaEditor::default();
                    ed.reset_to_new();
                    self.criteria_editor = Some(ed);
                }
            }
            if ui
                .add_enabled(project_open, egui::Button::new("Targets…"))
                .on_disabled_hover_text("Open or create a project first")
                .on_hover_text(
                    "Manage campaign work units: regions to annotate, generated \
                     from a criterion or hand-marked, each with a status",
                )
                .clicked()
            {
                ui.close();
                if self.targets_panel.is_none() {
                    self.targets_panel = Some(TargetsPanel::default());
                }
            }
            if ui
                .add_enabled(project_open, egui::Button::new("Dashboard…"))
                .on_disabled_hover_text("Open or create a project first")
                .on_hover_text(
                    "Campaign QA dashboard: completeness, per-annotator progress, \
                     tier QA, and inter-annotator agreement",
                )
                .clicked()
            {
                ui.close();
                if self.dashboard.is_none() {
                    self.dashboard = Some(DashboardWindow::default());
                }
            }
        });
    }

    fn file_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("File", |ui| {
            if ui.button("New Project…").clicked() {
                ui.close();
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Create a new sadda project — pick a directory")
                    .pick_folder()
                {
                    self.create_project_at(path);
                }
            }
            if ui.button("Open Project…").clicked() {
                ui.close();
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Open a sadda project — pick the project directory")
                    .pick_folder()
                {
                    self.open_project_at(path);
                }
            }
            ui.menu_button("Recent Projects", |ui| self.recent_projects_submenu(ui));
            ui.separator();
            let project_open = matches!(self.app_state, AppState::ProjectLoaded { .. });
            let bundle_selected = project_open && self.selected_bundle_id.is_some();
            if ui
                .add_enabled(project_open, egui::Button::new("Open Bundle…"))
                .on_disabled_hover_text("Open or create a project first")
                .clicked()
            {
                ui.close();
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("WAV", &["wav"])
                    .set_title("Pick a WAV file to register as a bundle")
                    .pick_file()
                {
                    self.add_bundle_from_wav(path);
                }
            }
            // ---- H1 Import submenu --------------------------------
            ui.menu_button("Import", |ui| {
                let import_enabled = bundle_selected;
                ui.add_enabled_ui(import_enabled, |ui| {
                    if ui.button("Praat TextGrid…").clicked() {
                        ui.close();
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Praat TextGrid", &["TextGrid", "textgrid"])
                            .set_title("Import a Praat TextGrid into the active bundle")
                            .pick_file()
                        {
                            self.import_textgrid_for_active_bundle(path);
                        }
                    }
                    if ui.button("ELAN .eaf…").clicked() {
                        ui.close();
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("ELAN EAF", &["eaf"])
                            .set_title("Import an ELAN .eaf into the active bundle")
                            .pick_file()
                        {
                            self.import_eaf_for_active_bundle(path);
                        }
                    }
                });
                if !import_enabled {
                    ui.label(
                        egui::RichText::new("(select a bundle first)")
                            .weak()
                            .small(),
                    );
                }
            });
            // ---- H1 Export submenu --------------------------------
            ui.menu_button("Export", |ui| {
                let export_enabled = bundle_selected;
                ui.add_enabled_ui(export_enabled, |ui| {
                    if ui.button("Praat TextGrid…").clicked() {
                        ui.close();
                        if let Some(path) = self.suggest_export_path("TextGrid") {
                            self.export_textgrid_for_active_bundle(path);
                        }
                    }
                    if ui.button("ELAN .eaf…").clicked() {
                        ui.close();
                        if let Some(path) = self.suggest_export_path("eaf") {
                            self.export_eaf_for_active_bundle(path);
                        }
                    }
                });
                if !export_enabled {
                    ui.label(
                        egui::RichText::new("(select a bundle first)")
                            .weak()
                            .small(),
                    );
                }
            });
            // ---- H1 Recording -------------------------------------
            if ui
                .add_enabled(project_open, egui::Button::new("Record from microphone…"))
                .on_disabled_hover_text("Open or create a project first")
                .clicked()
            {
                ui.close();
                self.open_record_dialog();
            }
            // ---- H1 Show project folder ---------------------------
            if ui
                .add_enabled(project_open, egui::Button::new("Show project folder"))
                .clicked()
            {
                ui.close();
                self.show_project_folder();
            }
            ui.separator();
            if ui.button("Quit").clicked() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
    }

    fn recent_projects_submenu(&mut self, ui: &mut egui::Ui) {
        if self.persisted.recent_projects.is_empty() {
            ui.label("(none yet)");
            return;
        }
        // Collect actions in a side buffer; we can't mutate
        // `self.persisted.recent_projects` while iterating it.
        let mut to_open: Option<PathBuf> = None;
        let mut to_remove: Option<PathBuf> = None;
        for p in &self.persisted.recent_projects {
            let exists = Project::is_project_root(p);
            let label = if exists {
                p.display().to_string()
            } else {
                format!("{}  (missing)", p.display())
            };
            if ui.button(label).clicked() {
                ui.close();
                if exists {
                    to_open = Some(p.clone());
                } else {
                    to_remove = Some(p.clone());
                }
            }
        }
        if let Some(p) = to_open {
            self.open_project_at(p);
        }
        if let Some(p) = to_remove {
            self.persisted.remove_recent(&p);
        }
    }

    fn view_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("View", |ui| {
            ui.label("Theme");
            ui.radio_value(&mut self.persisted.theme, ThemePref::System, "System");
            ui.radio_value(&mut self.persisted.theme, ThemePref::Light, "Light");
            ui.radio_value(&mut self.persisted.theme, ThemePref::Dark, "Dark");
            ui.separator();
            // Accessibility: plot colour scheme + UI zoom, both persisted.
            // The palette only recolours the cases where colour has to be
            // told apart (overlaid formants, coexisting reference bands);
            // the spectrogram has its own Cividis option in its toolbar.
            ui.label("Plot palette");
            ui.radio_value(
                &mut self.persisted.palette,
                PlotPalette::Default,
                PlotPalette::Default.label(),
            );
            ui.radio_value(
                &mut self.persisted.palette,
                PlotPalette::OkabeIto,
                PlotPalette::OkabeIto.label(),
            );
            ui.horizontal(|ui| {
                ui.label("UI scale");
                ui.add(
                    egui::Slider::new(&mut self.persisted.ui_scale, 0.8..=2.0)
                        .step_by(0.1)
                        .suffix("×"),
                );
            });
            ui.separator();
            // D10: measure-track lane visibility. Each toggle persists
            // across launches; enabling a lane triggers the analysis
            // on the next frame via `rebuild_tracks_if_stale`.
            ui.label("Measure Tracks");
            ui.checkbox(&mut self.persisted.tracks.f0_visible, "f0");
            ui.checkbox(&mut self.persisted.tracks.formants_visible, "Formants");
            ui.checkbox(&mut self.persisted.tracks.intensity_visible, "Intensity");
            ui.checkbox(&mut self.persisted.tracks.vad_visible, "VAD (speech)");
            // E12: embedding-heatmap submenu — tier picker (lists the
            // continuous_vector tiers of the active bundle) + colormap +
            // normalization. Hidden when no project / bundle is loaded.
            ui.menu_button("Embedding heatmap", |ui| {
                self.embedding_heatmap_submenu(ui);
            });
            // D10: reference-distribution overlay pickers, one per
            // band-capable lane. Each lists installed distributions whose
            // parameter matches the lane, narrowed by subgroup.
            ui.menu_button("f0 reference overlay", |ui| {
                refdist_overlay_submenu(ui, Some("f0"), &mut self.persisted.f0_overlay);
            });
            ui.menu_button("Intensity reference overlay", |ui| {
                refdist_overlay_submenu(
                    ui,
                    Some("intensity"),
                    &mut self.persisted.intensity_overlay,
                );
            });
            if ui.button("Install bundled reference data").clicked() {
                ui.close();
                match seed_bundled_refdists() {
                    Ok(n) => self.set_info(format!(
                        "Installed {n} bundled reference distribution(s) into the store."
                    )),
                    Err(e) => {
                        self.set_error(format!("Could not install bundled reference data: {e}"))
                    }
                }
            }
            ui.separator();
            // D10: right-side Reference panel (vowel space + histogram).
            ui.checkbox(
                &mut self.persisted.reference_panel_open,
                "Show Reference Panel",
            );
            // Inline Annotation panel (modal-free editor for the selection).
            ui.checkbox(
                &mut self.persisted.annotation_panel_open,
                "Show Annotation Panel",
            );
            // E8: script-panel toggle. Persists across launches.
            ui.checkbox(&mut self.persisted.script_panel_open, "Show Script Panel");
        });
    }

    /// E12 embedding-heatmap submenu: list the active bundle's
    /// `continuous_vector` tiers (the natural output of
    /// `Project.extract_embeddings`) as a one-shot radio selection, with
    /// a "(None — hide lane)" entry first and colormap / normalization
    /// sub-pickers afterwards. Renders a single "(no continuous_vector
    /// tiers yet)" hint when nothing's extractable.
    fn embedding_heatmap_submenu(&mut self, ui: &mut egui::Ui) {
        let (Some(env), AppState::ProjectLoaded { project, .. }) =
            (&self.active_envelope, &self.app_state)
        else {
            ui.label(egui::RichText::new("(no bundle loaded)").weak());
            return;
        };

        // Tier picker first — most-used control.
        let tiers = match project.tiers(Some(env.bundle_id)) {
            Ok(t) => t,
            Err(e) => {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 80, 80),
                    format!("Couldn't list tiers: {e}"),
                );
                return;
            }
        };
        let vector_tiers: Vec<_> = tiers
            .into_iter()
            .filter(|t| t.r#type == TierType::ContinuousVector)
            .collect();

        if ui
            .radio(
                self.persisted.embedding.selected_tier_id.is_none(),
                "None (hide lane)",
            )
            .clicked()
        {
            self.persisted.embedding.selected_tier_id = None;
        }
        if vector_tiers.is_empty() {
            ui.separator();
            ui.label(
                egui::RichText::new(
                    "(no continuous_vector tiers yet — run Project.extract_embeddings)",
                )
                .weak()
                .italics(),
            );
        } else {
            ui.separator();
            for t in &vector_tiers {
                let selected = self.persisted.embedding.selected_tier_id == Some(t.id);
                if ui.radio(selected, &t.name).clicked() {
                    self.persisted.embedding.selected_tier_id = Some(t.id);
                }
            }
        }

        ui.separator();
        ui.label("Colormap");
        for &(cm, label) in &[
            (ColormapKind::Cividis, "Cividis (CVD-safe)"),
            (ColormapKind::Viridis, "Viridis"),
            (ColormapKind::Magma, "Magma"),
            (ColormapKind::Greyscale, "Greyscale"),
        ] {
            ui.radio_value(&mut self.persisted.embedding.colormap, cm, label);
        }

        ui.separator();
        ui.label("Normalization");
        for &(mode, label) in &[
            (
                crate::state::EmbeddingNormalization::PerDimZScore,
                "Per-dim z-score (default)",
            ),
            (
                crate::state::EmbeddingNormalization::GlobalZScore,
                "Global z-score",
            ),
            (
                crate::state::EmbeddingNormalization::GlobalMinMax,
                "Global min–max",
            ),
        ] {
            ui.radio_value(&mut self.persisted.embedding.normalization, mode, label);
        }
    }

    fn help_menu(&mut self, ui: &mut egui::Ui) {
        ui.menu_button("Help", |ui| {
            if ui.button("About").clicked() {
                ui.close();
                self.set_error(format!(
                    "sadda {} — open-source toolkit for phonetics and speech-science research.\n\
                     Apache-2.0 OR MIT. https://github.com/sadda-speech/sadda",
                    sadda_engine::version()
                ));
            }
        });
    }

    fn bundle_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.heading("Bundles");
        ui.add_space(4.0);
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        let bundles = match project.bundles() {
            Ok(b) => b,
            Err(e) => {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 80, 80),
                    format!("Failed to list bundles: {e}"),
                );
                return;
            }
        };
        if bundles.is_empty() {
            ui.label(egui::RichText::new("(no bundles yet)").italics());
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Add one via File → Open Bundle…").weak());
            return;
        }
        let selected = self.selected_bundle_id;
        let mut to_select: Option<i64> = None;
        let mut to_reveal: Option<(i64, String)> = None;
        let mut to_delete_prompt: Option<(i64, String)> = None;
        let mut to_rename_prompt: Option<(i64, String)> = None;
        let mut to_provenance: Option<(i64, String)> = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            for b in &bundles {
                let is_selected = selected == Some(b.id);
                let duration_secs = b.n_frames as f64 / b.sample_rate as f64;
                let row = ui.selectable_label(
                    is_selected,
                    egui::RichText::new(format!("{}  ·  {:.2}s", b.name, duration_secs)),
                );
                if row.clicked() && !is_selected {
                    to_select = Some(b.id);
                }
                row.context_menu(|ui| {
                    if ui.button("Rename…").clicked() {
                        to_rename_prompt = Some((b.id, b.name.clone()));
                        ui.close();
                    }
                    if ui.button("Reveal in file manager").clicked() {
                        to_reveal = Some((b.id, b.audio_relative_path.clone()));
                        ui.close();
                    }
                    if ui.button("Provenance & citations…").clicked() {
                        to_provenance = Some((b.id, b.name.clone()));
                        ui.close();
                    }
                    ui.separator();
                    if ui
                        .button(
                            egui::RichText::new("Delete bundle…")
                                .color(egui::Color32::from_rgb(220, 80, 80)),
                        )
                        .clicked()
                    {
                        to_delete_prompt = Some((b.id, b.name.clone()));
                        ui.close();
                    }
                });
            }
        });
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(format!(
                "({} bundle{})",
                bundles.len(),
                if bundles.len() == 1 { "" } else { "s" }
            ))
            .weak(),
        );
        if let Some(id) = to_select {
            self.select_bundle(id);
        }
        if let Some((_id, audio_rel)) = to_reveal {
            self.reveal_bundle(&audio_rel);
        }
        if let Some((id, name)) = to_delete_prompt {
            self.pending_delete = Some(PendingBundleDelete { id, name });
        }
        if let Some((id, name)) = to_rename_prompt {
            self.pending_rename = Some(PendingBundleRename {
                id,
                name,
                just_started: true,
            });
        }
        if let Some((id, name)) = to_provenance {
            self.open_provenance_view(id, name);
        }
    }

    /// Central content for a loaded project: waveform on top
    /// (resizable), tier strip on the bottom (resizable, sized by
    /// tier count), spectrogram filling the middle.
    fn bundle_content_pane(&mut self, ui: &mut egui::Ui) {
        // Top sub-panel: waveform. Resizable; user can drag the
        // divider to rebalance with the spectrogram.
        egui::Panel::top("waveform_split")
            .resizable(true)
            .default_size(220.0)
            .min_size(80.0)
            // Frameless: no implicit panel margin. Horizontal
            // alignment across all lanes is owned by the shared
            // SIGNAL_LEFT_GUTTER / y-axis gutter, not panel frames.
            .frame(egui::Frame::NONE)
            .show_inside(ui, |ui| self.waveform_pane(ui));

        // Bottom sub-panel: tier strip. Resizable; default height
        // scales with the number of tiers up to a sensible cap.
        let n_lanes = self.estimate_tier_lane_count();
        let default_strip_height = ((n_lanes as f32).max(1.0) * TIER_LANE_HEIGHT + 8.0).min(220.0);
        egui::Panel::bottom("tier_strip")
            .resizable(true)
            .default_size(default_strip_height)
            .min_size(TIER_LANE_HEIGHT + 8.0)
            // Frameless, like the waveform panel — see note there.
            .frame(egui::Frame::NONE)
            .show_inside(ui, |ui| self.tier_strip_pane(ui));

        // D10 measure-track lanes, stacked between the spectrogram and
        // the tier strip. Each is a frameless bottom panel so it shares
        // the SIGNAL_LEFT_GUTTER / view-window with the other lanes and
        // the playback cursor draws one straight line through them all.
        // Registered bottom-up — intensity sits just above the tier
        // strip, then formants, then f0 just under the spectrogram —
        // because egui stacks the first-registered bottom panel
        // outermost. Only shown when a bundle is loaded and the lane is
        // enabled in View → Measure Tracks.
        self.rebuild_tracks_if_stale();
        self.rebuild_overlays_if_stale();
        self.rebuild_embedding_heatmap_if_stale(ui.ctx());
        if self.active_envelope.is_some() {
            let tracks = self.persisted.tracks;
            // Registered first → bottommost lane (just above the tiers).
            if tracks.vad_visible {
                egui::Panel::bottom("vad_lane")
                    .resizable(true)
                    .default_size(MEASURE_LANE_HEIGHT)
                    .min_size(48.0)
                    .frame(egui::Frame::NONE)
                    .show_inside(ui, |ui| self.vad_lane_pane(ui));
            }
            if tracks.intensity_visible {
                egui::Panel::bottom("intensity_lane")
                    .resizable(true)
                    .default_size(MEASURE_LANE_HEIGHT)
                    .min_size(48.0)
                    .frame(egui::Frame::NONE)
                    .show_inside(ui, |ui| self.intensity_lane_pane(ui));
            }
            if tracks.formants_visible {
                egui::Panel::bottom("formant_lane")
                    .resizable(true)
                    .default_size(MEASURE_LANE_HEIGHT)
                    .min_size(48.0)
                    .frame(egui::Frame::NONE)
                    .show_inside(ui, |ui| self.formant_lane_pane(ui));
            }
            if tracks.f0_visible {
                egui::Panel::bottom("f0_lane")
                    .resizable(true)
                    .default_size(MEASURE_LANE_HEIGHT)
                    .min_size(48.0)
                    .frame(egui::Frame::NONE)
                    .show_inside(ui, |ui| self.f0_lane_pane(ui));
            }
            // Embedding heatmap lane — registered LAST so it sits at the
            // top of the lane stack, directly under the spectrogram. The
            // taller default reflects that it's a 2D view (dim × time);
            // a 768-dim wav2vec2 embedding wants more vertical real
            // estate than a 1-D contour.
            if self.persisted.embedding.selected_tier_id.is_some()
                || self.embedding_heatmap_error.is_some()
            {
                egui::Panel::bottom("embedding_heatmap_lane")
                    .resizable(true)
                    .default_size(MEASURE_LANE_HEIGHT * 2.0)
                    .min_size(64.0)
                    .frame(egui::Frame::NONE)
                    .show_inside(ui, |ui| self.embedding_heatmap_lane_pane(ui));
            }
        }

        // Centre: spectrogram fills the remainder, drawn directly on
        // the panel `ui`. With the waveform/tier panels now frameless,
        // all three lanes share this `ui`'s content rect exactly, so
        // their plot areas line up with no per-element margins to
        // reconcile — alignment lives entirely in the 120px gutter.
        self.rebuild_spectrogram_if_stale(ui.ctx());
        self.spectrogram_pane(ui);
    }

    /// How many lanes the tier strip will render for the current
    /// project / bundle. Used only to size the bottom panel; returns
    /// `0` when no bundle is selected.
    fn estimate_tier_lane_count(&self) -> usize {
        let (Some(env), AppState::ProjectLoaded { project, .. }) =
            (&self.active_envelope, &self.app_state)
        else {
            return 0;
        };
        project
            .tiers(Some(env.bundle_id))
            .map(|t| t.len())
            .unwrap_or(0)
    }

    fn waveform_pane(&mut self, ui: &mut egui::Ui) {
        let Some(env) = &self.active_envelope else {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("Select a bundle from the sidebar").weak());
            });
            return;
        };
        if env.mono_samples.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("(empty waveform)").italics());
            });
            return;
        }

        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!(
                    "Bundle #{}  ·  {} Hz  ·  {:.3}s  ·  view {:.3}–{:.3}s",
                    env.bundle_id,
                    env.sample_rate,
                    env.duration_seconds,
                    self.timeline.view_start,
                    self.timeline.view_end,
                ))
                .weak(),
            );
        });

        let plot_width_px = ui.available_width().max(1.0) as usize;
        // One bucket per pixel column of the visible range — the
        // per-frame re-bucketing the B2 entry promised C5 would
        // land. Cost scales with `visible_samples`, not the full
        // bundle.
        let buckets = build_envelope_for_range(
            &env.mono_samples,
            env.sample_rate,
            self.timeline.view_start,
            self.timeline.view_end,
            plot_width_px,
        );
        let n_buckets = buckets.len();
        let view_range = self.timeline.view_range();
        let view_start = self.timeline.view_start;
        let view_end = self.timeline.view_end;
        let cursor = self.timeline.cursor;
        let selection = self.timeline.selection;
        let mut clicked_time: Option<f64> = None;
        let mut drag_start: Option<f64> = None;
        let mut drag_to: Option<f64> = None;
        let mut drag_ended = false;
        // Left edge of where the plot widget will be allocated (egui_plot
        // uses available_rect_before_wrap().min); the data area's left
        // minus this is the true y-axis gutter width.
        let widget_left = ui.available_rect_before_wrap().left();

        let plot_response = Plot::new("waveform")
            .show_axes([true, true])
            .y_axis_label("amplitude")
            .x_axis_label("seconds")
            .y_axis_min_width(SIGNAL_LEFT_GUTTER)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                // Own the bounds outright: visible window in x, full
                // amplitude range in y. This disables auto-fit (so no
                // edge margin) and matches the spectrogram's x-bounds
                // exactly, so the shared cursor lands on the same pixel
                // column in both panes.
                plot_ui.set_plot_bounds_x(view_start..=view_end);
                plot_ui.set_plot_bounds_y(-1.0..=1.0);
                if n_buckets > 0 {
                    let dt = view_range / n_buckets as f64;
                    for (i, (mn, mx)) in buckets.iter().enumerate() {
                        let t = view_start + (i as f64 + 0.5) * dt;
                        let segment = PlotPoints::from(vec![[t, *mn as f64], [t, *mx as f64]]);
                        plot_ui.line(
                            Line::new("", segment).color(egui::Color32::from_rgb(80, 140, 220)),
                        );
                    }
                }
                draw_cursor_line(plot_ui, cursor, -1.0, 1.0);
                draw_selection_band(plot_ui, selection, -1.0, 1.0);

                // Drag draws a time-span selection; a plain click clears
                // it and positions the cursor. Times come from the
                // pointer-coord in plot space.
                let resp = plot_ui.response();
                if resp.drag_started() {
                    drag_start = resp
                        .interact_pointer_pos()
                        .map(|p| plot_ui.plot_from_screen(p).x);
                }
                if resp.dragged() {
                    drag_to = resp
                        .interact_pointer_pos()
                        .map(|p| plot_ui.plot_from_screen(p).x);
                }
                if resp.drag_stopped() {
                    drag_ended = true;
                }
                if resp.clicked() {
                    clicked_time = resp
                        .interact_pointer_pos()
                        .map(|p| plot_ui.plot_from_screen(p).x);
                }
            });

        // Measure the lane geometry as WIDTHS (not absolute coords): the
        // y-axis gutter = data-area left minus the plot WIDGET's left
        // (NB egui_plot's `response.rect` is the data area, not the widget,
        // so we use the captured `widget_left`), plus the data-area width.
        {
            let frame = plot_response.transform.frame();
            let gutter_w = frame.left() - widget_left;
            self.lane_geom = Some((gutter_w, frame.width()));
            dlog!(
                "[layout] waveform widget_left={:.1} frame=[{:.1},{:.1}] gutter_w={:.1} data_w={:.1}",
                widget_left,
                frame.left(),
                frame.right(),
                gutter_w,
                frame.width()
            );
        }

        apply_lane_selection_drag(
            &mut self.timeline,
            drag_start,
            drag_to,
            drag_ended,
            clicked_time,
        );
        handle_zoom_and_scroll(&plot_response.response, &mut self.timeline);
    }

    fn spectrogram_pane(&mut self, ui: &mut egui::Ui) {
        // Toolbar row above the plot (window / hop / colormap).
        self.spectrogram_toolbar(ui);

        let Some(sc) = &self.active_spectrogram else {
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new(if self.active_envelope.is_some() {
                        "(building spectrogram…)"
                    } else {
                        "Select a bundle to see its spectrogram"
                    })
                    .weak(),
                );
            });
            return;
        };

        let duration = sc.duration_seconds;
        let nyquist = sc.nyquist_hz as f64;
        // The texture spans the whole bundle ([0, duration] × [0,
        // nyquist]); the explicit set_plot_bounds in the closure crops
        // it to the visible window. `include_x` does NOT crop — the
        // image's full extent dominates auto-bounds, which is why the
        // plot used to show the whole file and pad past 0 Hz and the
        // recording edges.
        let centre = egui_plot::PlotPoint::new(duration / 2.0, nyquist / 2.0);
        let size = egui::Vec2::new(duration as f32, nyquist as f32);
        let texture_id = sc.texture.id();
        let cursor = self.timeline.cursor;
        let view_start = self.timeline.view_start;
        let view_end = self.timeline.view_end;
        let selection = self.timeline.selection;
        let mut clicked_time: Option<f64> = None;
        let mut drag_start: Option<f64> = None;
        let mut drag_to: Option<f64> = None;
        let mut drag_ended = false;

        let plot_response = Plot::new("spectrogram")
            .show_axes([true, true])
            .y_axis_label("Hz")
            .x_axis_label("seconds")
            .y_axis_min_width(SIGNAL_LEFT_GUTTER)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                // Crop to the visible window in x and clamp y to
                // [0, nyquist]: no negative-frequency band, nothing
                // plotted past the recording edges, x aligned to the
                // waveform.
                plot_ui.set_plot_bounds_x(view_start..=view_end);
                plot_ui.set_plot_bounds_y(0.0..=nyquist);
                plot_ui.image(egui_plot::PlotImage::new(
                    "spectrogram_img",
                    texture_id,
                    centre,
                    size,
                ));
                draw_cursor_line(plot_ui, cursor, 0.0, nyquist);
                draw_selection_band(plot_ui, selection, 0.0, nyquist);

                let resp = plot_ui.response();
                if resp.drag_started() {
                    drag_start = resp
                        .interact_pointer_pos()
                        .map(|p| plot_ui.plot_from_screen(p).x);
                }
                if resp.dragged() {
                    drag_to = resp
                        .interact_pointer_pos()
                        .map(|p| plot_ui.plot_from_screen(p).x);
                }
                if resp.drag_stopped() {
                    drag_ended = true;
                }
                if resp.clicked() {
                    clicked_time = resp
                        .interact_pointer_pos()
                        .map(|p| plot_ui.plot_from_screen(p).x);
                }
            });

        apply_lane_selection_drag(
            &mut self.timeline,
            drag_start,
            drag_to,
            drag_ended,
            clicked_time,
        );
        handle_zoom_and_scroll(&plot_response.response, &mut self.timeline);
    }

    fn spectrogram_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Window:");
            let win = ui.add(
                egui::DragValue::new(&mut self.persisted.spectrogram.window_ms)
                    .speed(1.0)
                    .range(5.0..=200.0)
                    .suffix(" ms"),
            );
            ui.add_space(8.0);
            ui.label("Hop:");
            let hop = ui.add(
                egui::DragValue::new(&mut self.persisted.spectrogram.hop_ms)
                    .speed(0.5)
                    .range(1.0..=100.0)
                    .suffix(" ms"),
            );
            ui.add_space(8.0);
            ui.label("Range:");
            let dr = ui.add(
                egui::DragValue::new(&mut self.persisted.spectrogram.dynamic_range_db)
                    .speed(1.0)
                    .range(20.0..=120.0)
                    .suffix(" dB"),
            );
            ui.add_space(8.0);
            let mut cmap = self.persisted.spectrogram.colormap;
            let combo = egui::ComboBox::from_label("Colormap")
                .selected_text(cmap.label())
                .show_ui(ui, |ui| {
                    let mut changed = false;
                    for kind in [
                        ColormapKind::Viridis,
                        ColormapKind::Magma,
                        ColormapKind::Cividis,
                        ColormapKind::Greyscale,
                    ] {
                        if ui.selectable_value(&mut cmap, kind, kind.label()).clicked() {
                            changed = true;
                        }
                    }
                    changed
                });
            self.persisted.spectrogram.colormap = cmap;

            // The cache invalidates by `==` comparison, so any
            // change to the DragValues / ComboBox above is picked
            // up on the next frame's `rebuild_spectrogram_if_stale`.
            // The bindings here exist mostly to silence unused
            // warnings on the response objects.
            let _ = (win, hop, dr, combo);
        });
    }

    /// D10: f0 measure-track lane. Draws voiced pitch estimates as a
    /// dot contour (Praat draws f0 as dots, not a connected line, so
    /// unvoiced gaps read as gaps). Frames below the voicing threshold
    /// are dropped at draw time.
    fn f0_lane_pane(&mut self, ui: &mut egui::Ui) {
        let cfg = self.persisted.tracks;
        let Some(tc) = self.active_tracks.as_ref() else {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("(computing f0…)").weak());
            });
            return;
        };
        let frames = &tc.f0;
        let threshold = cfg.f0_voicing_threshold;
        let band = self.overlays.f0.as_ref().and_then(|(_, b)| b.as_ref());
        let palette = self.persisted.palette;
        let x0 = self.timeline.view_start;
        let x1 = self.timeline.view_end;
        let y_top = cfg.f0_max_hz as f64;
        measure_lane(
            ui,
            "f0_lane_plot",
            &mut self.timeline,
            (cfg.f0_min_hz as f64, cfg.f0_max_hz as f64),
            "f0 (Hz)",
            |plot_ui| {
                // Band first, behind the contour.
                if let Some(band) = band {
                    draw_refdist_band(plot_ui, band, x0, x1, y_top, palette);
                }
                let pts: Vec<[f64; 2]> = frames
                    .iter()
                    .filter(|f| f.voicing >= threshold)
                    .map(|f| [f.time_seconds, f.frequency_hz.value() as f64])
                    .collect();
                if !pts.is_empty() {
                    plot_ui.points(
                        Points::new("f0", PlotPoints::from(pts))
                            .radius(1.6)
                            .color(egui::Color32::from_rgb(40, 120, 230)),
                    );
                }
            },
        );
    }

    /// D10: formant measure-track lane. One dot series per formant
    /// slot (F1..Fn), each its own colour, so vowel formant trajectories
    /// are separable by eye.
    fn formant_lane_pane(&mut self, ui: &mut egui::Ui) {
        let cfg = self.persisted.tracks;
        let Some(tc) = self.active_tracks.as_ref() else {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("(computing formants…)").weak());
            });
            return;
        };
        let frames = &tc.formants;
        let n = cfg.formant_count;
        let palette = self.persisted.palette;
        measure_lane(
            ui,
            "formant_lane_plot",
            &mut self.timeline,
            (0.0, cfg.formant_max_hz as f64),
            "formants (Hz)",
            |plot_ui| {
                for slot in 0..n {
                    let pts: Vec<[f64; 2]> = frames
                        .iter()
                        .filter_map(|f| {
                            f.frequencies
                                .get(slot)
                                .map(|hz| [f.time_seconds, hz.value() as f64])
                        })
                        .collect();
                    if !pts.is_empty() {
                        plot_ui.points(
                            Points::new(format!("F{}", slot + 1), PlotPoints::from(pts))
                                .radius(1.5)
                                .color(formant_color(palette, slot)),
                        );
                    }
                }
            },
        );
    }

    /// D10: intensity measure-track lane. A connected dB-FS contour
    /// (intensity is continuous, so unlike f0 it reads best as a line).
    fn intensity_lane_pane(&mut self, ui: &mut egui::Ui) {
        let cfg = self.persisted.tracks;
        let Some(tc) = self.active_tracks.as_ref() else {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("(computing intensity…)").weak());
            });
            return;
        };
        let frames = &tc.intensity;
        let band = self
            .overlays
            .intensity
            .as_ref()
            .and_then(|(_, b)| b.as_ref());
        let palette = self.persisted.palette;
        let x0 = self.timeline.view_start;
        let x1 = self.timeline.view_end;
        measure_lane(
            ui,
            "intensity_lane_plot",
            &mut self.timeline,
            (cfg.intensity_floor_db as f64, 0.0),
            "intensity (dB)",
            |plot_ui| {
                if let Some(band) = band {
                    draw_refdist_band(plot_ui, band, x0, x1, 0.0, palette);
                }
                let pts: Vec<[f64; 2]> = frames
                    .iter()
                    .map(|f| [f.time_seconds, f.db_fs.value() as f64])
                    .collect();
                if pts.len() >= 2 {
                    plot_ui.line(
                        Line::new("intensity", PlotPoints::from(pts))
                            .color(egui::Color32::from_rgb(225, 170, 40)),
                    );
                }
            },
        );
    }

    /// E11: VAD (voice-activity) lane. Draws the per-window speech
    /// probability (0–1) with a dashed threshold line; windows above the
    /// threshold are shaded as speech. If VAD couldn't run (e.g. ONNX
    /// Runtime not available) the lane shows the reason instead.
    fn vad_lane_pane(&mut self, ui: &mut egui::Ui) {
        let cfg = self.persisted.tracks;
        let Some(tc) = self.active_tracks.as_ref() else {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("(computing VAD…)").weak());
            });
            return;
        };
        if let Some(err) = &tc.vad_error {
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new(format!("VAD unavailable — {err}"))
                        .weak()
                        .small(),
                );
            });
            return;
        }
        let frames = &tc.vad;
        let threshold = cfg.vad_threshold as f64;
        let x0 = self.timeline.view_start;
        let x1 = self.timeline.view_end;
        // Half a Silero window (~16 ms at 16 kHz) for shading extents.
        let half = (512.0 / 16_000.0) / 2.0;
        measure_lane(
            ui,
            "vad_lane_plot",
            &mut self.timeline,
            (0.0, 1.0),
            "VAD (speech p)",
            |plot_ui| {
                // Shade windows above threshold as speech regions.
                for f in frames.iter().filter(|f| f.speech_prob as f64 >= threshold) {
                    let (lo, hi) = (f.time_seconds - half, f.time_seconds + half);
                    plot_ui.polygon(
                        Polygon::new(
                            "",
                            PlotPoints::from(vec![[lo, 0.0], [hi, 0.0], [hi, 1.0], [lo, 1.0]]),
                        )
                        .fill_color(egui::Color32::from_rgba_unmultiplied(90, 160, 90, 40))
                        .stroke(egui::Stroke::NONE),
                    );
                }
                // Threshold line.
                plot_ui.line(
                    Line::new(
                        "threshold",
                        PlotPoints::from(vec![[x0, threshold], [x1, threshold]]),
                    )
                    .color(egui::Color32::from_gray(140))
                    .style(LineStyle::dashed_loose()),
                );
                // Speech-probability contour.
                let pts: Vec<[f64; 2]> = frames
                    .iter()
                    .map(|f| [f.time_seconds, f.speech_prob as f64])
                    .collect();
                if pts.len() >= 2 {
                    plot_ui.line(
                        Line::new("speech_p", PlotPoints::from(pts))
                            .color(egui::Color32::from_rgb(70, 165, 95)),
                    );
                }
            },
        );
    }

    /// E12 embedding-heatmap lane. Draws the cached texture as a
    /// `PlotImage` spanning `[0, duration] × [0, n_dims]`, cropped to
    /// the current x-view (matching the spectrogram + measure-track
    /// lanes for cursor alignment). Hover surfaces (time, dim, raw
    /// value) for inspection. When the cache failed to build (selected
    /// tier missing, sidecar unreadable) the error message renders
    /// centred in the lane rather than blanking it out.
    fn embedding_heatmap_lane_pane(&mut self, ui: &mut egui::Ui) {
        // Sticky build error → render as centred hint and bail. The lane
        // stays visible so the user sees the explanation without having
        // to chase a missing-tier message into the bottom banner.
        if let Some(msg) = self.embedding_heatmap_error.clone() {
            ui.centered_and_justified(|ui| {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 80, 80),
                    format!("Embedding heatmap: {msg}"),
                );
            });
            return;
        }
        let Some(cache) = &self.active_embedding_heatmap else {
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new("(building embedding heatmap…)")
                        .weak()
                        .italics(),
                );
            });
            return;
        };

        // Lane caption: tier name + dim count, weak so it doesn't
        // compete with the heatmap colours.
        ui.horizontal(|ui| {
            ui.add_space(SIGNAL_LEFT_GUTTER + 4.0);
            ui.label(
                egui::RichText::new(format!(
                    "Embedding · {} · {} dim{}",
                    cache.tier_name,
                    cache.n_dims,
                    if cache.n_dims == 1 { "" } else { "s" },
                ))
                .weak(),
            );
        });

        let duration = cache.duration_seconds;
        let n_dims = cache.n_dims as f64;
        let centre = egui_plot::PlotPoint::new(duration / 2.0, n_dims / 2.0);
        let size = egui::Vec2::new(duration as f32, n_dims as f32);
        let texture_id = cache.texture.id();
        let cursor = self.timeline.cursor;
        let view_start = self.timeline.view_start;
        let view_end = self.timeline.view_end;
        let selection = self.timeline.selection;
        let mut clicked_time: Option<f64> = None;
        let mut drag_start: Option<f64> = None;
        let mut drag_to: Option<f64> = None;
        let mut drag_ended = false;

        let plot_response = Plot::new("embedding_heatmap_plot")
            .show_axes([true, true])
            .y_axis_label("dim")
            .x_axis_label("seconds")
            .y_axis_min_width(SIGNAL_LEFT_GUTTER)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                plot_ui.set_plot_bounds_x(view_start..=view_end);
                plot_ui.set_plot_bounds_y(0.0..=n_dims);
                plot_ui.image(egui_plot::PlotImage::new(
                    "embedding_heatmap_img",
                    texture_id,
                    centre,
                    size,
                ));
                draw_cursor_line(plot_ui, cursor, 0.0, n_dims);
                draw_selection_band(plot_ui, selection, 0.0, n_dims);

                let resp = plot_ui.response();
                if resp.drag_started() {
                    drag_start = resp
                        .interact_pointer_pos()
                        .map(|p| plot_ui.plot_from_screen(p).x);
                }
                if resp.dragged() {
                    drag_to = resp
                        .interact_pointer_pos()
                        .map(|p| plot_ui.plot_from_screen(p).x);
                }
                if resp.drag_stopped() {
                    drag_ended = true;
                }
                if resp.clicked() {
                    clicked_time = resp
                        .interact_pointer_pos()
                        .map(|p| plot_ui.plot_from_screen(p).x);
                }
            })
            .response;

        apply_lane_selection_drag(
            &mut self.timeline,
            drag_start,
            drag_to,
            drag_ended,
            clicked_time,
        );
        handle_zoom_and_scroll(&plot_response, &mut self.timeline);
    }

    fn tier_strip_pane(&mut self, ui: &mut egui::Ui) {
        let (Some(env), AppState::ProjectLoaded { project, .. }) =
            (&self.active_envelope, &self.app_state)
        else {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("Select a bundle to see its tiers").weak());
            });
            return;
        };
        let bundle_id = env.bundle_id;
        let view_start = self.timeline.view_start;
        let view_end = self.timeline.view_end;
        let cursor = self.timeline.cursor;
        let selection = self.timeline.selection;
        let active_tier_id = self.active_tier_id;
        let lane_geom = self.lane_geom;
        let tiers = match project.tiers(Some(bundle_id)) {
            Ok(t) => t,
            Err(e) => {
                ui.colored_label(
                    egui::Color32::from_rgb(220, 80, 80),
                    format!("Failed to list tiers: {e}"),
                );
                return;
            }
        };
        // Tier-lifecycle + selection-commit requests raised below are
        // applied after the `&project` borrow ends — same snapshot
        // discipline as the selection state.
        let mut tier_op: Option<TierOp> = None;
        let mut clicked_active: Option<i64> = None;
        let mut selection_commit: Option<(i64, TierType, f64, f64)> = None;
        let mut clear_selection = false;
        let active = active_tier_id.and_then(|aid| tiers.iter().find(|t| t.id == aid));
        ui.horizontal(|ui| {
            if ui.button("➕ New tier…").clicked() {
                tier_op = Some(TierOp::New);
            }
            ui.separator();
            match active {
                Some(t) => {
                    ui.label(format!("Active tier: {}", truncate_label(&t.name, 16)));
                }
                None => {
                    ui.label(egui::RichText::new("Active tier: none — click a tier name").weak());
                }
            }
            if let Some((lo, hi)) = selection {
                ui.separator();
                ui.label(egui::RichText::new(format!("selection {lo:.3}–{hi:.3}s")).small());
                let can_commit = active
                    .map(|t| matches!(t.r#type, TierType::Interval | TierType::Point))
                    .unwrap_or(false);
                let hint = if active.map(|t| t.r#type) == Some(TierType::Point) {
                    "Add points at edges"
                } else {
                    "Add interval"
                };
                if ui
                    .add_enabled(can_commit, egui::Button::new(hint))
                    .on_hover_text("add the selection to the active tier")
                    .clicked()
                {
                    if let Some(t) = active {
                        selection_commit = Some((t.id, t.r#type, lo, hi));
                    }
                }
                if ui.button("Clear").clicked() {
                    clear_selection = true;
                }
            }
        });
        if tiers.is_empty() {
            ui.label(egui::RichText::new("(no tiers in this bundle yet)").italics());
        }
        // Snapshot the selection up-front; any click handled inside
        // the lane render mutates the snapshot, which we copy back
        // at the end. Borrow-checker-friendly: avoids needing &mut
        // self while we also hold &project.
        let mut new_selection = self.selected_annotation;
        // Same idea for the error banner.
        let mut new_error: Option<String> = None;
        // Click on a lane (background or annotation) can also move
        // the playback cursor, kept here so we apply at the end.
        let mut new_cursor: Option<f64> = None;
        // Draft-mutation requests from interval lanes (one per frame
        // at most — only the tier the cursor is over fires).
        let mut new_draft_action: Option<DraftAction> = None;
        // Label-edit request: double-clicking an interval body.
        let mut label_edit_request: Option<LabelEdit> = None;
        // Right-click "Delete" request from a lane; applied after the
        // &project borrow ends, reusing `delete_selected_annotation`.
        let mut request_delete = false;
        // Double-click on an annotation opens the inline Annotation panel.
        let mut open_annotation_panel = false;
        // The rubric's status vocabulary, fetched once per frame, used to
        // tint each annotation's status strip.
        let status_palette: Vec<String> = project
            .rubric_statuses()
            .map(|v| v.into_iter().map(|s| s.value).collect())
            .unwrap_or_default();

        egui::ScrollArea::vertical().show(ui, |ui| {
            for tier in &tiers {
                ui.horizontal(|ui| {
                    // Drop the default inter-widget spacing so the
                    // lane's left edge sits at exactly
                    // (row_left + SIGNAL_LEFT_GUTTER), matching where
                    // the plots' inner plot areas start.
                    // Reserve the gutter + time-lane via the shared helper so
                    // the alignment-critical allocation is the same code the
                    // `layout_tests` regression test exercises.
                    let (gutter_rect, gutter_resp, rect, response) =
                        allocate_tier_row(ui, lane_geom);
                    // Active tier (the span-selection target) is highlighted;
                    // the name is painted into the reserved gutter rect.
                    let name_color = if active_tier_id == Some(tier.id) {
                        SELECTION_EDGE
                    } else {
                        ui.visuals().strong_text_color()
                    };
                    ui.painter_at(gutter_rect).text(
                        gutter_rect.left_center() + egui::vec2(4.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        truncate_label(&tier.name, 16),
                        egui::FontId::proportional(13.0),
                        name_color,
                    );
                    let gutter_resp = gutter_resp.on_hover_text(
                        "click to make this the active tier; right-click for actions",
                    );
                    if gutter_resp.clicked() {
                        clicked_active = Some(tier.id);
                    }
                    gutter_resp.context_menu(|ui| {
                        if ui.button("Rename tier…").clicked() {
                            tier_op = Some(TierOp::Rename(tier.id, tier.name.clone()));
                            ui.close();
                        }
                        if ui.button("Delete tier").clicked() {
                            tier_op = Some(TierOp::Delete(tier.id, tier.name.clone()));
                            ui.close();
                        }
                    });
                    dlog!(
                        "[layout] tier '{}' gutter_w={:.1} lane=[{:.1},{:.1}]",
                        tier.name,
                        gutter_rect.width(),
                        rect.left(),
                        rect.right()
                    );
                    let painter = ui.painter_at(rect);
                    let visuals = ui.visuals();

                    // Background.
                    painter.rect_filled(rect, 2.0, visuals.extreme_bg_color);
                    painter.rect_stroke(
                        rect,
                        2.0,
                        egui::Stroke::new(1.0, visuals.widgets.noninteractive.bg_stroke.color),
                        egui::StrokeKind::Inside,
                    );

                    match tier.r#type {
                        TierType::Interval => match project.intervals(tier.id) {
                            Ok(rows) => render_interval_lane(
                                &painter,
                                rect,
                                view_start,
                                view_end,
                                tier.id,
                                &rows,
                                &status_palette,
                                is_preview_tier(&tier.name),
                                self.selected_annotation,
                                &response,
                                &self.draft_edit,
                                &mut new_selection,
                                &mut new_cursor,
                                &mut new_draft_action,
                                &mut label_edit_request,
                                &mut request_delete,
                                &mut open_annotation_panel,
                            ),
                            Err(e) => {
                                new_error = Some(format!(
                                    "Failed to list intervals for tier {}: {e}",
                                    tier.id,
                                ));
                            }
                        },
                        TierType::Point => match project.points(tier.id) {
                            Ok(rows) => render_point_lane(
                                &painter,
                                rect,
                                view_start,
                                view_end,
                                tier.id,
                                &rows,
                                &status_palette,
                                is_preview_tier(&tier.name),
                                self.selected_annotation,
                                &response,
                                &self.draft_edit,
                                &mut new_selection,
                                &mut new_cursor,
                                &mut new_draft_action,
                                &mut label_edit_request,
                                &mut request_delete,
                                &mut open_annotation_panel,
                            ),
                            Err(e) => {
                                new_error = Some(format!(
                                    "Failed to list points for tier {}: {e}",
                                    tier.id,
                                ));
                            }
                        },
                        TierType::Reference => match project.references_for(tier.id) {
                            Ok(rows) => {
                                let caption = format_reference_lane_caption(rows.len());
                                painter.text(
                                    rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    caption,
                                    egui::FontId::proportional(12.0),
                                    visuals.weak_text_color(),
                                );
                            }
                            Err(e) => {
                                new_error = Some(format!(
                                    "Failed to list references for tier {}: {e}",
                                    tier.id,
                                ));
                            }
                        },
                        TierType::ContinuousNumeric
                        | TierType::ContinuousVector
                        | TierType::CategoricalSampled => {
                            painter.text(
                                rect.center(),
                                egui::Align2::CENTER_CENTER,
                                "(dense — not displayable in tier strip)",
                                egui::FontId::proportional(12.0),
                                visuals.weak_text_color(),
                            );
                        }
                    }

                    // Selection band spans every lane so it lines up with
                    // the waveform / spectrogram. Drawn under the cursor.
                    draw_selection_band_rect(&painter, rect, view_start, view_end, selection);

                    // Draw the synced cursor over every lane (incl.
                    // reference / dense captions) at this frame's
                    // timeline.cursor — clipped to the lane rect.
                    let view_range = (view_end - view_start).max(1e-6);
                    if cursor >= view_start && cursor <= view_end {
                        let x_per_second = rect.width() as f64 / view_range;
                        let cx = rect.left() + ((cursor - view_start) * x_per_second) as f32;
                        painter.line_segment(
                            [
                                egui::Pos2::new(cx, rect.top()),
                                egui::Pos2::new(cx, rect.bottom()),
                            ],
                            egui::Stroke::new(1.0, egui::Color32::from_rgb(230, 70, 70)),
                        );
                    }

                    // Click on the lane background (not on a hit
                    // item) deselects and moves the cursor to the
                    // click position. Detect by: response.clicked()
                    // AND new_selection hasn't changed AND the
                    // click pos maps inside the lane.
                    if response.clicked()
                        && new_selection == self.selected_annotation
                        && new_cursor.is_none()
                    {
                        if let Some(p) = response.interact_pointer_pos() {
                            if rect.contains(p) {
                                let x_per_second = rect.width() as f64 / view_range;
                                let t = view_start + ((p.x - rect.left()) as f64) / x_per_second;
                                new_cursor = Some(t);
                                if self.selected_annotation.is_some() {
                                    new_selection = None;
                                }
                            }
                        }
                    }
                });
            }
        });

        self.selected_annotation = new_selection;
        // Right-click "Delete" acts on the just-selected annotation, reusing
        // the same path as the Delete/Backspace key.
        if request_delete {
            self.delete_selected_annotation();
        }
        // Double-clicking an annotation reveals the inline editor.
        if open_annotation_panel {
            self.persisted.annotation_panel_open = true;
        }
        if let Some(t) = new_cursor {
            self.timeline.set_cursor(t);
        }
        if let Some(msg) = new_error {
            self.set_error(msg);
        }

        // ---- Apply draft mutations from the lanes ------------------
        if let Some(action) = new_draft_action {
            match action {
                DraftAction::Start(draft) => {
                    self.draft_edit = draft;
                }
                DraftAction::Update(t) => match &mut self.draft_edit {
                    DraftEdit::Creating { current_time, .. }
                    | DraftEdit::Resizing { current_time, .. }
                    | DraftEdit::MovingPoint { current_time, .. } => {
                        *current_time = t;
                    }
                    DraftEdit::None => {}
                },
                DraftAction::Commit => self.commit_draft_edit(),
                DraftAction::AddPointNow { tier_id, time } => {
                    self.add_point_immediate(tier_id, time);
                }
            }
        }

        // ---- Inline label edit request -----------------------------
        if let Some(req) = label_edit_request {
            self.label_edit = Some(req);
        }

        // ---- Active-tier + selection-commit (apply after &project) -
        if let Some(id) = clicked_active {
            // Toggle off if clicking the already-active tier.
            self.active_tier_id = if self.active_tier_id == Some(id) {
                None
            } else {
                Some(id)
            };
        }
        if clear_selection {
            self.timeline.clear_selection();
        }
        if let Some((tier_id, tier_type, lo, hi)) = selection_commit {
            self.commit_selection_to_tier(tier_id, tier_type, lo, hi);
        }

        // ---- Tier-lifecycle requests (apply after &project ended) --
        match tier_op {
            Some(TierOp::New) => {
                self.pending_new_tier = Some(NewTierDraft {
                    bundle_id,
                    name: String::new(),
                    tier_type: TierType::Interval,
                    just_started: true,
                });
            }
            Some(TierOp::Rename(id, name)) => {
                self.pending_tier_rename = Some(PendingTierRename {
                    id,
                    name,
                    just_started: true,
                });
            }
            Some(TierOp::Delete(id, name)) => {
                self.pending_tier_delete = Some(PendingTierDelete { id, name });
            }
            None => {}
        }
    }

    /// Resolves the active draft (create or resize) by writing to the
    /// engine. Clears the draft on success; on error, surfaces in the
    /// banner and still clears the draft (the user can retry).
    fn commit_draft_edit(&mut self) {
        let draft = std::mem::replace(&mut self.draft_edit, DraftEdit::None);
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        let result: Result<(), String> = match draft {
            DraftEdit::None => Ok(()),
            DraftEdit::Creating {
                tier_id,
                start_time,
                current_time,
            } => {
                let (lo, hi) = if start_time <= current_time {
                    (start_time, current_time)
                } else {
                    (current_time, start_time)
                };
                if (hi - lo) < MIN_DRAFT_CREATE_SECONDS {
                    // Too small a drag: treat as accidental click;
                    // don't create.
                    return;
                }
                project
                    .add_interval(&sadda_engine::IntervalSpec {
                        tier_id,
                        start_seconds: lo,
                        end_seconds: hi,
                        label: None,
                        extra: None,
                        ..Default::default()
                    })
                    .map(|id| {
                        self.selected_annotation = Some(AnnotationSelection::Interval {
                            tier_id,
                            annotation_id: id,
                        });
                        self.timeline.set_cursor(lo);
                    })
                    .map_err(|e| format!("Failed to create interval: {e}"))
            }
            DraftEdit::Resizing {
                tier_id,
                annotation_id,
                edge: _,
                fixed_time,
                current_time,
            } => {
                let (lo, hi) = if fixed_time <= current_time {
                    (fixed_time, current_time)
                } else {
                    (current_time, fixed_time)
                };
                // Don't write an invalid (zero or reversed) span;
                // engine CHECK would reject it anyway.
                if (hi - lo) < MIN_DRAFT_CREATE_SECONDS {
                    return;
                }
                let existing = match project.intervals(tier_id) {
                    Ok(rows) => rows.into_iter().find(|r| r.id == annotation_id),
                    Err(e) => {
                        return self.set_error(format!("Failed to reload interval: {e}"));
                    }
                };
                let Some(existing) = existing else {
                    return; // Annotation was deleted concurrently
                };
                project
                    .update_interval(
                        annotation_id,
                        &sadda_engine::IntervalSpec {
                            tier_id,
                            start_seconds: lo,
                            end_seconds: hi,
                            label: existing.label,
                            parent_annotation_id: existing.parent_annotation_id,
                            status: existing.status,
                            note: existing.note,
                            processing_run_id: existing.processing_run_id,
                            extra: existing.extra,
                        },
                    )
                    .map_err(|e| format!("Failed to resize interval: {e}"))
            }
            DraftEdit::MovingPoint {
                tier_id,
                annotation_id,
                original_time,
                current_time,
            } => {
                // Mouse jitter under MIN_POINT_MOVE_SECONDS is
                // dropped without writing a no-op audit row.
                if (current_time - original_time).abs() < MIN_POINT_MOVE_SECONDS {
                    return;
                }
                let existing = match project.points(tier_id) {
                    Ok(rows) => rows.into_iter().find(|r| r.id == annotation_id),
                    Err(e) => {
                        return self.set_error(format!("Failed to reload point: {e}"));
                    }
                };
                let Some(existing) = existing else {
                    return;
                };
                let new_time = current_time.max(0.0).min(self.timeline.duration);
                project
                    .update_point(
                        annotation_id,
                        &sadda_engine::PointSpec {
                            tier_id,
                            time_seconds: new_time,
                            label: existing.label,
                            parent_annotation_id: existing.parent_annotation_id,
                            status: existing.status,
                            note: existing.note,
                            processing_run_id: existing.processing_run_id,
                            extra: existing.extra,
                        },
                    )
                    .map(|_| {
                        self.timeline.set_cursor(new_time);
                    })
                    .map_err(|e| format!("Failed to move point: {e}"))
            }
        };
        if let Err(msg) = result {
            self.set_error(msg);
        }
    }

    /// Click-to-add for point lanes (D7). The lane render fires
    /// this as a `DraftAction::AddPointNow` when the user clicks
    /// empty space in a point lane.
    fn add_point_immediate(&mut self, tier_id: i64, time: f64) {
        let AppState::ProjectLoaded { project, .. } = &self.app_state else {
            return;
        };
        let t = time.max(0.0).min(self.timeline.duration);
        match project.add_point(&sadda_engine::PointSpec {
            tier_id,
            time_seconds: t,
            label: None,
            extra: None,
            ..Default::default()
        }) {
            Ok(id) => {
                self.selected_annotation = Some(AnnotationSelection::Point {
                    tier_id,
                    annotation_id: id,
                });
                self.timeline.set_cursor(t);
            }
            Err(e) => self.set_error(format!("Failed to add point: {e}")),
        }
    }

    /// E8 script panel: top = code editor, bottom = output. Run via
    /// the button OR Ctrl/Cmd+Enter (handled at the app level so
    /// the shortcut works whether or not the editor has focus).
    /// D10: right-side Reference panel — a vowel-space scatter (the
    /// reference cloud + the measured vowel at the cursor) and a 1-D
    /// histogram of a chosen parameter with percentile + measured-value
    /// markers. Both read from the cached [`ReferenceView`], refreshed in
    /// `rebuild_reference_if_stale`.
    fn reference_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Reference").strong());
        });
        ui.label(
            egui::RichText::new("Compare your measurements against a reference distribution.")
                .weak(),
        );
        ui.separator();

        // Distribution picker (all installed distributions).
        let picker_label = self
            .persisted
            .reference_dist
            .as_ref()
            .map(|_| self.reference.title.clone())
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| "Select distribution…".to_string());
        ui.menu_button(picker_label, |ui| {
            refdist_overlay_submenu(ui, None, &mut self.persisted.reference_dist);
        });

        if self.persisted.reference_dist.is_none() {
            ui.label(
                egui::RichText::new(
                    "Pick a distribution above. If none are listed, run \
                     View → Install bundled reference data.",
                )
                .weak(),
            );
            return;
        }

        // Snapshot the cached view + the current measurements into owned
        // locals, so the picker/selectors below can freely mutate
        // `self.persisted` without aliasing the borrows.
        let params = self.reference.params.clone();
        let phones = self.reference.phones.clone();
        let cloud = self.reference.cloud.clone();
        let histogram = self.reference.histogram.clone();
        let summary = self.reference.summary;
        let active_param = self.reference.active_param.clone();
        let kind = self.reference.kind;
        let cursor = self.timeline.cursor;
        let threshold = self.persisted.tracks.f0_voicing_threshold;
        let measured_vowel = self
            .active_tracks
            .as_ref()
            .and_then(|tc| formant_point_at_cursor(&tc.formants, cursor));
        let median_f0 = self
            .active_tracks
            .as_ref()
            .and_then(|tc| median_voiced_f0(&tc.f0, threshold));

        // Vowel-space scatter (first two parameters), phonetic orientation.
        if params.len() >= 2 && !cloud.is_empty() {
            ui.separator();
            if !phones.is_empty() {
                ui.horizontal(|ui| {
                    ui.label("Phone:");
                    let cur = self
                        .persisted
                        .reference_phone
                        .clone()
                        .unwrap_or_else(|| "all".to_string());
                    egui::ComboBox::from_id_salt("ref_phone")
                        .selected_text(cur)
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(self.persisted.reference_phone.is_none(), "all")
                                .clicked()
                            {
                                self.persisted.reference_phone = None;
                            }
                            for ph in &phones {
                                let on = self.persisted.reference_phone.as_deref() == Some(ph);
                                if ui.selectable_label(on, ph).clicked() {
                                    self.persisted.reference_phone = Some(ph.clone());
                                }
                            }
                        });
                });
            }
            // Convention: F2 on x (high → left), F1 on y (high → down).
            let x_name = params[1].clone();
            let y_name = params[0].clone();
            ui.label(format!("Vowel space — x: {x_name}, y: {y_name}"));
            Plot::new("vowel_space")
                .invert_x(true)
                .invert_y(true)
                .x_axis_label(x_name)
                .y_axis_label(y_name)
                .height(220.0)
                .show(ui, |plot_ui| {
                    let pts: Vec<[f64; 2]> = cloud.iter().map(|p| [p[1], p[0]]).collect();
                    plot_ui.points(
                        Points::new("reference", PlotPoints::from(pts))
                            .radius(2.5)
                            .color(egui::Color32::from_rgba_unmultiplied(120, 140, 200, 160)),
                    );
                    if let Some((f1, f2)) = measured_vowel {
                        plot_ui.points(
                            Points::new("you", PlotPoints::from(vec![[f2, f1]]))
                                .radius(6.0)
                                .shape(egui_plot::MarkerShape::Diamond)
                                .color(egui::Color32::from_rgb(230, 70, 70)),
                        );
                    }
                });
            if measured_vowel.is_none() {
                ui.label(
                    egui::RichText::new("enable the Formants track to plot your vowel")
                        .weak()
                        .small(),
                );
            }
        }

        // Histogram of the active parameter.
        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Histogram:");
            for p in &params {
                let on = active_param.as_deref() == Some(p);
                if ui.selectable_label(on, p).clicked() {
                    self.persisted.reference_param = Some(p.clone());
                }
            }
        });

        let measured_scalar: Option<f64> = match active_param.as_deref() {
            Some(p) if p.eq_ignore_ascii_case("f0") => median_f0,
            Some("F1") => measured_vowel.map(|(f1, _)| f1),
            Some("F2") => measured_vowel.map(|(_, f2)| f2),
            _ => None,
        };

        if let Some(h) = &histogram {
            draw_reference_histogram(ui, h, summary.as_ref(), measured_scalar);
        } else if kind == Some(MeasureKind::SummaryNormativeRange) {
            ui.label(
                egui::RichText::new(
                    "summary-only distribution — no per-sample histogram; \
                     use the band overlay on the f0 lane.",
                )
                .weak(),
            );
            if let Some(s) = summary {
                ui.label(format!(
                    "mean {:.1}  ·  sd {:.1}  ·  p5–p95 {:.1}–{:.1}",
                    s.mean, s.sd, s.p5, s.p95
                ));
            }
        }
    }

    fn script_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Script").strong());
            ui.add_space(8.0);
            ui.label(egui::RichText::new("Ctrl/Cmd+Enter to run").weak().small());
            let mut run_clicked = false;
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Run").clicked() {
                    run_clicked = true;
                }
            });
            if run_clicked {
                self.run_script_buffer();
            }
        });
        ui.separator();

        // Top half: code editor. Bottom half: output. The split is
        // not user-resizable inside the panel for v1; the outer
        // Panel::bottom is resizable.
        let total_h = ui.available_height();
        let code_h = (total_h * 0.6).max(60.0);
        let output_h = (total_h - code_h - 4.0).max(40.0);

        // Python syntax highlighting via a custom layouter. The closure
        // re-tokenizes the buffer each frame; layouts are cheap relative
        // to a human-sized script, and egui caches the resulting galley
        // keyed on the job, so an unchanged buffer re-lays out only when
        // the font/theme changes.
        let mut layouter = |ui: &egui::Ui, buf: &dyn egui::TextBuffer, wrap_width: f32| {
            let mut job = python_layout_job(ui, buf.as_str());
            job.wrap.max_width = wrap_width;
            ui.fonts_mut(|f| f.layout_job(job))
        };
        egui::ScrollArea::vertical()
            .id_salt("script_code_scroll")
            .max_height(code_h)
            .show(ui, |ui| {
                ui.add_sized(
                    [ui.available_width(), code_h],
                    egui::TextEdit::multiline(&mut self.persisted.script_buffer)
                        .desired_rows(8)
                        .code_editor()
                        .layouter(&mut layouter)
                        .hint_text("# Python — pure stdlib only at E8.\n# `import sadda` lands in E9.\nprint('hello from sadda')\n"),
                );
            });

        ui.separator();
        ui.label(egui::RichText::new("Output").weak().small());
        egui::ScrollArea::vertical()
            .id_salt("script_output_scroll")
            .max_height(output_h)
            // Fill the full panel width so the scrollbar sits at the far
            // right edge, not hugging the right of the output text. The
            // default `auto_shrink` shrinks horizontally to content width;
            // we keep vertical shrink so the area still collapses to the
            // text height (bounded by `max_height`).
            .auto_shrink([false, true])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                if let Some(output) = &self.script_output {
                    if !output.stdout.is_empty() {
                        ui.label(egui::RichText::new(&output.stdout).monospace());
                    }
                    if !output.stderr.is_empty() {
                        ui.colored_label(
                            egui::Color32::from_rgb(220, 80, 80),
                            egui::RichText::new(&output.stderr).monospace(),
                        );
                    }
                    if output.stdout.is_empty() && output.stderr.is_empty() {
                        ui.label(egui::RichText::new("(no output)").italics().weak());
                    }
                }
                if let Some(err) = &self.script_error {
                    ui.colored_label(
                        egui::Color32::from_rgb(220, 80, 80),
                        egui::RichText::new(err).monospace(),
                    );
                }
                if self.script_output.is_none() && self.script_error.is_none() {
                    ui.label(
                        egui::RichText::new("(no runs yet — click Run or press Ctrl/Cmd+Enter)")
                            .italics()
                            .weak(),
                    );
                }
            });
    }

    /// Runs the persisted script buffer through the embedded
    /// CPython runtime, with the `sadda.app` snapshot installed.
    /// Captures stdout / stderr into `self.script_output`; Python
    /// exceptions land in `self.script_error`. Drains any commands
    /// the script registered into `self.registered_commands`.
    fn run_script_buffer(&mut self) {
        let code = self.persisted.script_buffer.clone();
        if code.trim().is_empty() {
            self.script_output = None;
            self.script_error = Some("(script buffer is empty)".to_string());
            return;
        }
        let snapshot = self.snapshot_now();
        let mut extras = ScriptSessionExtras::default();
        let result = with_snapshot_active(&snapshot, &mut extras, || {
            sadda_script_engine::run_script(&code)
        });
        match result {
            Ok(output) => {
                self.script_output = Some(output);
                self.script_error = None;
            }
            Err(e) => {
                self.script_output = None;
                self.script_error = Some(e.to_string());
            }
        }
        // Append any newly-registered commands. PyObject refcount
        // keeps the callable alive past this scope.
        self.registered_commands
            .extend(std::mem::take(&mut extras.registered_commands));
    }

    /// Invokes the command at index `i` of `registered_commands`,
    /// passing no args. Surfaces Python exceptions in the script
    /// panel's error slot.
    fn invoke_command(&mut self, i: usize) {
        let Some((name, callable_ref)) = self.registered_commands.get(i) else {
            return;
        };
        let name = name.clone();
        // Py<PyAny> needs the GIL to bump its refcount; do the
        // clone + the call inside one `attach` block so we don't
        // pay the GIL hit twice.
        let snapshot = self.snapshot_now();
        let mut extras = ScriptSessionExtras::default();
        let result = with_snapshot_active(&snapshot, &mut extras, || {
            Python::attach(|py| {
                let cb = callable_ref.clone_ref(py);
                cb.call0(py)
            })
        });
        match result {
            Ok(_) => {
                // Successful invocation — no special UI update.
                let _ = name;
            }
            Err(e) => {
                self.script_error = Some(format!("Command {name:?} raised: {e}"));
                // Also pop the script panel open if it isn't,
                // so the user sees the error.
                self.persisted.script_panel_open = true;
            }
        }
        // Commands invoked from the palette can themselves
        // register more commands — drain the extras.
        self.registered_commands
            .extend(std::mem::take(&mut extras.registered_commands));
    }

    /// Builds an `AppSnapshot` describing the current GUI state,
    /// used by the `sadda.app.*` PyO3 functions during a script
    /// run / command invocation.
    fn snapshot_now(&self) -> AppSnapshot {
        let project_root = match &self.app_state {
            AppState::NoProject => PathBuf::new(),
            AppState::ProjectLoaded { root, .. } => root.clone(),
        };
        let bundle = self.active_envelope.as_ref().and_then(|env| {
            let AppState::ProjectLoaded { project, .. } = &self.app_state else {
                return None;
            };
            project.bundles().ok().and_then(|all| {
                all.into_iter()
                    .find(|b| b.id == env.bundle_id)
                    .map(|b| BundleInfo {
                        id: b.id,
                        name: b.name,
                        sample_rate: env.sample_rate,
                        duration_seconds: env.duration_seconds,
                    })
            })
        });
        let selection = self.selected_annotation.map(|sel| match sel {
            AnnotationSelection::Interval {
                tier_id,
                annotation_id,
            } => SelectionInfo {
                kind: SelectionKind::Interval,
                tier_id,
                annotation_id,
            },
            AnnotationSelection::Point {
                tier_id,
                annotation_id,
            } => SelectionInfo {
                kind: SelectionKind::Point,
                tier_id,
                annotation_id,
            },
        });
        AppSnapshot {
            project_root,
            bundle,
            selection,
            cursor_seconds: self.timeline.cursor,
        }
    }

    /// Renders the Ctrl/Cmd+P command palette as a modal Window.
    /// Filtering by substring on the command name; Enter or click
    /// invokes.
    fn command_palette_window(&mut self, ctx: &egui::Context) {
        if !self.command_palette_open {
            return;
        }
        let mut keep_open = true;
        let mut invoke_index: Option<usize> = None;
        egui::Window::new("Command palette")
            .collapsible(false)
            .resizable(false)
            .open(&mut keep_open)
            .default_width(420.0)
            .show(ctx, |ui| {
                let resp = ui.text_edit_singleline(&mut self.command_palette_query);
                resp.request_focus();
                ui.separator();
                if self.registered_commands.is_empty() {
                    ui.label(
                        egui::RichText::new(
                            "(no commands registered yet — call sadda.app.register_command from the script panel)",
                        )
                        .italics()
                        .weak(),
                    );
                    return;
                }
                let query = self.command_palette_query.to_lowercase();
                let mut visible: Vec<usize> = (0..self.registered_commands.len())
                    .filter(|&i| {
                        let name = &self.registered_commands[i].0;
                        query.is_empty() || name.to_lowercase().contains(&query)
                    })
                    .collect();
                if visible.is_empty() {
                    ui.label(
                        egui::RichText::new("(no matches)").italics().weak(),
                    );
                    return;
                }
                // Enter invokes the first match.
                if resp.lost_focus()
                    && ui.input(|i| i.key_pressed(egui::Key::Enter))
                {
                    invoke_index = visible.first().copied();
                }
                egui::ScrollArea::vertical()
                    .max_height(300.0)
                    .show(ui, |ui| {
                        for i in visible.drain(..) {
                            let row = ui.selectable_label(
                                false,
                                &self.registered_commands[i].0,
                            );
                            if row.clicked() {
                                invoke_index = Some(i);
                            }
                        }
                    });
            });
        if let Some(i) = invoke_index {
            self.command_palette_open = false;
            self.command_palette_query.clear();
            self.invoke_command(i);
        }
        if !keep_open {
            self.command_palette_open = false;
            self.command_palette_query.clear();
        }
        // Escape closes.
        if self.command_palette_open && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.command_palette_open = false;
            self.command_palette_query.clear();
        }
    }

    fn welcome(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.add_space(60.0);
            ui.heading(egui::RichText::new(APP_TITLE).size(48.0).strong());
            ui.label("speech-analysis toolkit");
            ui.add_space(36.0);

            ui.horizontal(|ui| {
                // Centre the two buttons in the available row.
                let total = 180.0 + 8.0 + 180.0;
                let pad = (ui.available_width() - total).max(0.0) / 2.0;
                ui.add_space(pad);
                if ui
                    .add_sized([180.0, 36.0], egui::Button::new("New Project"))
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .set_title("Create a new sadda project")
                        .pick_folder()
                    {
                        self.create_project_at(path);
                    }
                }
                ui.add_space(8.0);
                if ui
                    .add_sized([180.0, 36.0], egui::Button::new("Open Project"))
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .set_title("Open a sadda project")
                        .pick_folder()
                    {
                        self.open_project_at(path);
                    }
                }
            });

            ui.add_space(36.0);
            ui.label(egui::RichText::new("Recent").strong());
            ui.add_space(8.0);
            self.recent_projects_list(ui);
        });
    }

    fn recent_projects_list(&mut self, ui: &mut egui::Ui) {
        if self.persisted.recent_projects.is_empty() {
            ui.label(egui::RichText::new("(no recent projects yet)").italics());
            return;
        }
        let mut to_open: Option<PathBuf> = None;
        let mut to_remove: Option<PathBuf> = None;
        for p in &self.persisted.recent_projects {
            let exists = Project::is_project_root(p);
            let (text, colour) = if exists {
                (p.display().to_string(), egui::Color32::PLACEHOLDER)
            } else {
                (format!("{}  (missing)", p.display()), egui::Color32::GRAY)
            };
            let response = ui.add(
                egui::Label::new(egui::RichText::new(text).color(colour))
                    .sense(egui::Sense::click()),
            );
            if response.clicked() {
                if exists {
                    to_open = Some(p.clone());
                } else {
                    to_remove = Some(p.clone());
                }
            }
        }
        if let Some(p) = to_open {
            self.open_project_at(p);
        }
        if let Some(p) = to_remove {
            self.persisted.remove_recent(&p);
        }
    }
}

/// A coarse Python token class, enough to colour the script editor.
/// Deliberately lexical-only (no parsing) — the goal is a readable
/// editor, not a linter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PyTok {
    Keyword,
    Builtin,
    StringLit,
    Comment,
    Number,
    Other,
}

/// Python keywords (3.12 `keyword.kwlist` plus the soft keywords
/// `match`/`case`). Looked up by exact identifier match.
const PY_KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield", "match", "case",
];

/// A pragmatic subset of common Python builtins worth tinting so they
/// stand out from user names. Not exhaustive — `builtins.dir()` has ~150
/// entries; these are the ones a phonetics script touches most.
const PY_BUILTINS: &[&str] = &[
    "abs",
    "all",
    "any",
    "bool",
    "bytes",
    "dict",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "getattr",
    "hasattr",
    "id",
    "input",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "open",
    "ord",
    "chr",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "setattr",
    "sorted",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "zip",
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "IndexError",
    "RuntimeError",
    "self",
    "cls",
];

/// Colours per token class, chosen per theme. Values follow the
/// familiar VS Code light/dark editor palette so the highlighting reads
/// as conventional rather than bespoke.
struct SyntaxPalette {
    keyword: egui::Color32,
    builtin: egui::Color32,
    string: egui::Color32,
    comment: egui::Color32,
    number: egui::Color32,
    other: egui::Color32,
}

impl SyntaxPalette {
    fn for_visuals(v: &egui::Visuals) -> Self {
        use egui::Color32;
        if v.dark_mode {
            Self {
                keyword: Color32::from_rgb(86, 156, 214),
                builtin: Color32::from_rgb(78, 201, 176),
                string: Color32::from_rgb(206, 145, 120),
                comment: Color32::from_rgb(106, 153, 85),
                number: Color32::from_rgb(181, 206, 168),
                other: v.text_color(),
            }
        } else {
            Self {
                keyword: Color32::from_rgb(0, 0, 200),
                builtin: Color32::from_rgb(38, 127, 153),
                string: Color32::from_rgb(163, 21, 21),
                comment: Color32::from_rgb(0, 128, 0),
                number: Color32::from_rgb(9, 134, 88),
                other: v.text_color(),
            }
        }
    }

    fn color(&self, tok: PyTok) -> egui::Color32 {
        match tok {
            PyTok::Keyword => self.keyword,
            PyTok::Builtin => self.builtin,
            PyTok::StringLit => self.string,
            PyTok::Comment => self.comment,
            PyTok::Number => self.number,
            PyTok::Other => self.other,
        }
    }
}

/// Lexes Python source into `(run, class)` pairs. The concatenation of
/// every run is byte-for-byte the input — this is load-bearing: the
/// editor's galley text must equal the buffer or cursor positioning and
/// selection break. Whitespace and newlines ride along in `Other` runs.
fn tokenize_python(text: &str) -> Vec<(String, PyTok)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out: Vec<(String, PyTok)> = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c == '#' {
            // Comment: to end of line.
            let start = i;
            while i < n && chars[i] != '\n' {
                i += 1;
            }
            out.push((chars[start..i].iter().collect(), PyTok::Comment));
        } else if c == '"' || c == '\'' {
            // String literal: triple- or single-quoted, escape-aware.
            let quote = c;
            let triple = i + 2 < n && chars[i + 1] == quote && chars[i + 2] == quote;
            let start = i;
            if triple {
                i += 3;
                while i < n {
                    if chars[i] == '\\' && i + 1 < n {
                        i += 2;
                        continue;
                    }
                    if chars[i] == quote
                        && i + 2 < n
                        && chars[i + 1] == quote
                        && chars[i + 2] == quote
                    {
                        i += 3;
                        break;
                    }
                    i += 1;
                }
            } else {
                i += 1; // opening quote
                while i < n {
                    if chars[i] == '\\' && i + 1 < n {
                        i += 2;
                        continue;
                    }
                    if chars[i] == quote {
                        i += 1;
                        break;
                    }
                    if chars[i] == '\n' {
                        break; // unterminated single-line string
                    }
                    i += 1;
                }
            }
            out.push((chars[start..i].iter().collect(), PyTok::StringLit));
        } else if c.is_ascii_digit() {
            // Number: consume a maximal alnum/./_ run (covers 0x1F, 1_000, 3.14).
            let start = i;
            while i < n && (chars[i].is_alphanumeric() || chars[i] == '.' || chars[i] == '_') {
                i += 1;
            }
            out.push((chars[start..i].iter().collect(), PyTok::Number));
        } else if c.is_alphabetic() || c == '_' {
            // Identifier: classify against keyword / builtin tables.
            let start = i;
            while i < n && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let tok = if PY_KEYWORDS.contains(&word.as_str()) {
                PyTok::Keyword
            } else if PY_BUILTINS.contains(&word.as_str()) {
                PyTok::Builtin
            } else {
                PyTok::Other
            };
            out.push((word, tok));
        } else {
            // Everything else (operators, punctuation, whitespace,
            // newlines): a maximal run that doesn't start another token.
            let start = i;
            i += 1;
            while i < n {
                let d = chars[i];
                if d == '#'
                    || d == '"'
                    || d == '\''
                    || d.is_ascii_digit()
                    || d.is_alphabetic()
                    || d == '_'
                {
                    break;
                }
                i += 1;
            }
            out.push((chars[start..i].iter().collect(), PyTok::Other));
        }
    }
    out
}

/// Builds a syntax-highlighted `LayoutJob` for the script editor's
/// `TextEdit::layouter`. Monospace font from the active style; colours
/// from the active theme.
fn python_layout_job(ui: &egui::Ui, text: &str) -> egui::text::LayoutJob {
    let palette = SyntaxPalette::for_visuals(ui.visuals());
    let font_id = egui::TextStyle::Monospace.resolve(ui.style());
    let mut job = egui::text::LayoutJob::default();
    for (run, tok) in tokenize_python(text) {
        job.append(
            &run,
            0.0,
            egui::TextFormat {
                font_id: font_id.clone(),
                color: palette.color(tok),
                ..Default::default()
            },
        );
    }
    job
}

#[cfg(test)]
mod layout_tests {
    //! Headless regression tests for tier-lane horizontal alignment, run
    //! through egui's built-in single-frame harness (`Context::run`) — no GUI,
    //! no extra deps. These guard the bug we hit where the gutter, allocated
    //! with `allocate_ui_with_layout`, collapsed to the tier-name width and
    //! left-shifted the lane out of alignment with the signal plots.
    use super::*;
    use eframe::egui;

    /// Runs `add` inside one egui frame on a `width`×200 screen, within a
    /// horizontal layout off the root `Ui`. The alignment assertions are
    /// relative (lane vs gutter), so the absolute origin is immaterial — what
    /// matters is that the same allocation runs in a real egui layout pass.
    fn run_one_frame<R>(width: f32, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
        let ctx = egui::Context::default();
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width, 200.0),
            )),
            ..Default::default()
        };
        // `run_ui`'s closure is `FnMut`; move the one-shot `add` through an
        // `Option` so it is consumed on the single invocation.
        let mut add = Some(add);
        let mut out = None;
        let _ = ctx.run_ui(raw, |ui| {
            if let Some(add) = add.take() {
                out = Some(ui.horizontal(add).inner);
            }
        });
        out.expect("run_ui invokes the ui closure once")
    }

    #[test]
    fn tier_row_reserves_full_gutter_and_aligns_lane() {
        // Simulate a measured signal plot: 120px y-axis gutter, 800px data.
        let geom = Some((120.0_f32, 800.0_f32));
        let (gutter, _g, lane, _l) = run_one_frame(1000.0, |ui| allocate_tier_row(ui, geom));
        // The gutter must reserve the FULL measured width — the original bug
        // collapsed it to the (much narrower) tier-name width.
        assert!(
            (gutter.width() - 120.0).abs() < 0.5,
            "gutter width = {} (expected 120)",
            gutter.width()
        );
        // The lane sits flush against the gutter — no gap, no overlap — so its
        // left edge coincides with the signal plots' data-area left.
        assert!(
            (lane.left() - gutter.right()).abs() < 0.5,
            "lane.left {} != gutter.right {}",
            lane.left(),
            gutter.right()
        );
        // And it spans exactly the plot data-area width, so the right edges
        // line up too.
        assert!(
            (lane.width() - 800.0).abs() < 0.5,
            "lane width = {} (expected 800)",
            lane.width()
        );
    }

    #[test]
    fn tier_row_falls_back_to_fixed_gutter_before_measurement() {
        // Before the waveform is measured the gutter uses the fixed default.
        let (gutter, _g, _lane, _l) = run_one_frame(1000.0, |ui| allocate_tier_row(ui, None));
        assert!(
            (gutter.width() - SIGNAL_LEFT_GUTTER).abs() < 0.5,
            "gutter width = {} (expected {SIGNAL_LEFT_GUTTER})",
            gutter.width()
        );
    }
}

#[cfg(test)]
mod rubric_ui_tests {
    use super::*;

    #[test]
    fn out_of_vocab_only_flags_nonempty_unknown_labels_with_a_vocabulary() {
        let vocab = vec!["a".to_string(), "i".to_string()];
        // In-vocabulary, empty label, and empty vocabulary are never OOV.
        assert!(!is_out_of_vocab(&vocab, "a"));
        assert!(!is_out_of_vocab(&vocab, ""));
        assert!(!is_out_of_vocab(&[], "anything"));
        // A non-empty label absent from a non-empty vocabulary is OOV.
        assert!(is_out_of_vocab(&vocab, "zzz"));
    }

    #[test]
    fn preview_tier_detection_matches_the_engine_suffix() {
        // Must match the engine's `preview_tier_name` = "<target> (auto)".
        assert!(is_preview_tier("vowels (auto)"));
        assert!(is_preview_tier(&format!("landmarks{PREVIEW_TIER_SUFFIX}")));
        assert!(!is_preview_tier("vowels"));
        assert!(!is_preview_tier("(auto) prefix"));
    }

    #[test]
    fn status_tint_is_stable_by_position_and_none_for_unknown() {
        let statuses = vec!["draft".to_string(), "done".to_string()];
        assert_eq!(status_tint(None, &statuses), None);
        assert_eq!(status_tint(Some("nope"), &statuses), None);
        let a = status_tint(Some("draft"), &statuses).unwrap();
        let b = status_tint(Some("done"), &statuses).unwrap();
        // Distinct statuses get distinct tints; the same status is stable.
        assert_ne!(a, b);
        assert_eq!(status_tint(Some("draft"), &statuses).unwrap(), a);
    }

    #[test]
    fn provenance_line_strips_prefix_and_trims_timestamp() {
        let line =
            format_provenance_line("sadda.criteria.vowel midpoints", "2026-05-31T12:34:56.789Z");
        assert_eq!(line, "↻ from criterion “vowel midpoints” · 2026-05-31 12:34:56");
        // A processor_id without the expected prefix is shown verbatim; a
        // timestamp without a fractional part is handled too.
        let line = format_provenance_line("custom.proc", "2026-05-31T00:00:00Z");
        assert_eq!(line, "↻ from criterion “custom.proc” · 2026-05-31 00:00:00");
    }

    #[test]
    fn target_row_shows_roi_type_and_source() {
        let t = sadda_engine::Target {
            id: 1,
            bundle_id: 1,
            start_seconds: 0.2,
            end_seconds: 0.5,
            target_type: "phones".into(),
            status: "in_progress".into(),
            source: "criterion".into(),
            criterion_id: Some(7),
            note: None,
            created_at: String::new(),
            updated_at: String::new(),
        };
        assert_eq!(format_target_row(&t), "[0.20–0.50s] phones · criterion");
    }

    #[test]
    fn assignment_summary_lists_annotators_or_unassigned() {
        assert_eq!(format_assignment_summary(&[]), "— unassigned");
        let mk = |annotator: &str, role: &str| sadda_engine::Assignment {
            id: 1,
            target_id: 1,
            annotator: annotator.into(),
            role: role.into(),
            status: "assigned".into(),
            seed: None,
            created_at: String::new(),
            updated_at: String::new(),
        };
        let a = [mk("alice", "primary"), mk("bob", "secondary")];
        assert_eq!(
            format_assignment_summary(&a),
            "→ alice(primary), bob(secondary)"
        );
    }

    #[test]
    fn package_summaries_read_naturally() {
        let exp = sadda_engine::ExportSummary {
            annotator: "alice".into(),
            path: std::path::PathBuf::from("/tmp/pkg"),
            bundles: 2,
            targets: 5,
            assignments: 5,
        };
        assert_eq!(
            format_export_summary(&exp),
            "Exported 2 bundle(s), 5 target(s), 5 assignment(s) for “alice” → /tmp/pkg"
        );
        let imp = sadda_engine::ImportSummary {
            annotator: "alice".into(),
            bundles_matched: 2,
            tiers_imported: 3,
            annotations_imported: 40,
            assignments_marked_done: 5,
        };
        assert_eq!(
            format_import_summary(&imp),
            "Imported “alice”: 2 bundle(s) matched, 3 tier(s) / 40 annotation(s) landed, 5 assignment(s) done"
        );
    }

    #[test]
    fn progress_and_agreement_lines_read_naturally() {
        let p = sadda_engine::ProgressCounts {
            total: 10,
            unassigned: 2,
            assigned: 1,
            in_progress: 1,
            done: 5,
            flagged: 1,
        };
        assert_eq!(
            format_target_progress(&p),
            "Progress: 5/10 done · 1 in progress · 1 flagged · 3 to do"
        );
        let r = sadda_engine::AgreementReport {
            tier_type: "interval".into(),
            n_a: 3,
            n_b: 3,
            n_matched: 3,
            n_only_a: 0,
            n_only_b: 0,
            percent_label_agreement: 2.0 / 3.0,
            cohen_kappa: 0.5,
            mean_abs_boundary_diff: 0.012,
            boundary_within_tolerance: 0.75,
            boundary_tolerance_seconds: 0.020,
            frame_percent_agreement: 0.9,
            frame_kappa: 0.6,
            frame_step_seconds: 0.010,
        };
        assert_eq!(
            format_agreement_report(&r),
            "κ=0.50 (67% labels) · 3 matched / 0+0 extra · Δbound 12ms (75% ≤20ms) · frame κ=0.60"
        );
    }

    #[test]
    fn dashboard_lines_read_naturally() {
        let a = sadda_engine::AnnotatorProgress {
            annotator: "alice".into(),
            assigned: 2,
            in_progress: 1,
            done: 4,
        };
        assert_eq!(
            format_annotator_progress(&a),
            "alice: 4 done · 1 in progress · 2 to do"
        );
        let q = sadda_engine::QaReport {
            tier_id: 1,
            n_annotations: 12,
            out_of_vocab: 2,
            missing_label: 1,
            overlaps: 3,
        };
        assert_eq!(
            format_qa_report(&q),
            "12 annotations · 2 out-of-vocab · 1 missing · 3 overlaps"
        );
    }

    #[test]
    fn tier_impact_line_reads_naturally() {
        let t = sadda_engine::TierImpact {
            tier_name: "phones".into(),
            vocab_added: vec!["c".into()],
            vocab_removed: vec!["b".into()],
            affected_annotations: 4,
        };
        assert_eq!(
            format_tier_impact(&t),
            "phones: +[c] −[b] · 4 to revisit"
        );
    }
}

#[cfg(test)]
mod python_highlight_tests {
    use super::*;

    /// The lexer must be lossless: concatenating every run reproduces
    /// the input exactly, or the editor galley desyncs from the buffer.
    #[test]
    fn tokenize_preserves_text_exactly() {
        let src = "def f(x):  # comment\n    s = 'hi\\n'\n    t = \"\"\"multi\nline\"\"\"\n    return 42 + x_2\n";
        let toks = tokenize_python(src);
        let rebuilt: String = toks.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(rebuilt, src);
    }

    fn class_of(toks: &[(String, PyTok)], run: &str) -> Option<PyTok> {
        toks.iter().find(|(s, _)| s == run).map(|(_, t)| *t)
    }

    #[test]
    fn classifies_common_tokens() {
        let src = "def go():\n    print('hi')  # note\n    return 0xFF\n";
        let toks = tokenize_python(src);
        assert_eq!(class_of(&toks, "def"), Some(PyTok::Keyword));
        assert_eq!(class_of(&toks, "return"), Some(PyTok::Keyword));
        assert_eq!(class_of(&toks, "print"), Some(PyTok::Builtin));
        assert_eq!(class_of(&toks, "go"), Some(PyTok::Other)); // user name
        assert_eq!(class_of(&toks, "'hi'"), Some(PyTok::StringLit));
        assert_eq!(class_of(&toks, "0xFF"), Some(PyTok::Number));
        // The comment run stops before the newline; the `\n` rides in the
        // following `Other` run.
        assert_eq!(class_of(&toks, "# note"), Some(PyTok::Comment));
    }

    #[test]
    fn unterminated_string_stops_at_newline() {
        let src = "x = 'oops\ny = 1\n";
        let toks = tokenize_python(src);
        // The unterminated literal must not swallow the rest of the file.
        assert_eq!(class_of(&toks, "'oops"), Some(PyTok::StringLit));
        assert_eq!(class_of(&toks, "y"), Some(PyTok::Other));
        let rebuilt: String = toks.iter().map(|(s, _)| s.as_str()).collect();
        assert_eq!(rebuilt, src);
    }
}

#[cfg(test)]
mod ort_sidecar_tests {
    use super::find_ort_in_dir;

    #[test]
    fn missing_dir_returns_none() {
        let tmp = std::env::temp_dir().join("sadda-ort-sidecar-missing");
        let _ = std::fs::remove_dir_all(&tmp);
        assert!(find_ort_in_dir(&tmp).is_none());
    }

    #[test]
    fn dir_without_runtime_returns_none() {
        // A directory whose only "library-shaped" file isn't actually the
        // runtime is rejected by the probe (matching the deferred-shim
        // case). We use a libc-shaped name so the filename filter accepts
        // it but the OrtGetApiBase symbol check fails.
        let tmp =
            std::env::temp_dir().join(format!("sadda-ort-sidecar-no-rt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        // Empty file under a name the platform-glob would accept. The
        // probe will fail to dlopen it (zero-byte ELF), so discovery
        // returns None.
        #[cfg(target_os = "windows")]
        let name = "onnxruntime.dll";
        #[cfg(target_os = "macos")]
        let name = "libonnxruntime.dylib";
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        let name = "libonnxruntime.so";
        std::fs::write(tmp.join(name), b"").unwrap();
        assert!(find_ort_in_dir(&tmp).is_none());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

#[cfg(test)]
mod palette_tests {
    use super::{PlotPalette, formant_color};

    #[test]
    fn formant_palette_differs_by_scheme_and_is_internally_distinct() {
        // Switching schemes must actually change the colours — a no-op
        // accessibility toggle is worse than none.
        assert_ne!(
            formant_color(PlotPalette::Default, 0),
            formant_color(PlotPalette::OkabeIto, 0),
        );
        // Within each scheme the first five formant slots are all
        // distinct, so F1..F5 are separable by colour.
        for palette in [PlotPalette::Default, PlotPalette::OkabeIto] {
            let colours: Vec<_> = (0..5).map(|s| formant_color(palette, s)).collect();
            for i in 0..colours.len() {
                for j in (i + 1)..colours.len() {
                    assert_ne!(
                        colours[i], colours[j],
                        "slots {i}/{j} collide in {palette:?}"
                    );
                }
            }
        }
        // Slots past the set wrap back to the start.
        assert_eq!(
            formant_color(PlotPalette::OkabeIto, 0),
            formant_color(PlotPalette::OkabeIto, 5),
        );
    }
}
