//! Sadda desktop GUI. Slice A1 ships the project-aware shell —
//! welcome screen with New / Open / Recent, a `Project`-backed loaded
//! state, persistent window + recent-projects state. No content panes
//! yet; those land in cluster B (waveform / spectrogram / tier strip).
//!
//! See the 2026-05-23 DEVLOG entry "App shell + project open/create
//! (A1)" for the design rationale and the cut-list for what
//! deliberately doesn't ship at A1.

mod playback;
mod sadda_app;
mod state;

use std::path::{Path, PathBuf};

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use pyo3::prelude::*;
use sadda_engine::{LiveConfig, LiveResults, LiveSession, Project, StoppedSession, TierType};

use crate::playback::Playback;
use crate::sadda_app::{
    AppSnapshot, BundleInfo, ScriptSessionExtras, SelectionInfo, SelectionKind,
    with_snapshot_active,
};
use crate::state::{
    ColormapKind, EnvelopeCache, PersistedState, SpectrogramConfig, ThemePref, TimelineState,
    build_envelope_for_range, colormap_bake, format_reference_lane_caption, power_to_db_normalized,
    truncate_label,
};

/// Maximum characters drawn inside an interval rectangle or above a
/// point tick before truncation kicks in (with an ellipsis).
const TIER_LABEL_MAX_CHARS: usize = 20;
/// Vertical pixels per lane in the tier strip.
const TIER_LANE_HEIGHT: f32 = 28.0;
/// Width of the left gutter holding the tier name in the tier strip.
const TIER_LABEL_GUTTER: f32 = 120.0;

const APP_TITLE: &str = "sadda";
/// Cap on spectrogram texture width. egui's typical max texture size
/// is 8192; 4096 keeps headroom and gives roughly 1px per ~150 ms at
/// 10 minutes — fine resolution for the long-recording case the B3
/// spike note flagged. Longer audio averages frames into buckets.
const MAX_SPECTROGRAM_WIDTH: usize = 4096;

fn main() -> eframe::Result<()> {
    // E9: register the built-in `sadda` module BEFORE the embedded
    // CPython interpreter starts, so embedded scripts can
    // `import sadda.app` without needing the wheel pip-installed.
    // Must happen before any pyo3 call that might trigger
    // auto-initialize (the script-engine's first run_script).
    pyo3::append_to_inittab!(sadda);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 720.0])
            .with_min_inner_size([640.0, 480.0])
            .with_title(APP_TITLE),
        ..Default::default()
    };
    eframe::run_native(
        APP_TITLE,
        options,
        Box::new(|cc| Ok(Box::new(SaddaApp::new(cc)))),
    )
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
    /// Currently-selected annotation in the tier strip. In-memory
    /// only — clears on bundle change. Reached by C5 (cursor sync)
    /// and D6/D7 (editing) when those slices land.
    selected_annotation: Option<AnnotationSelection>,
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
}

/// H1: identifies a bundle the user has requested be deleted,
/// rendered as a confirmation modal.
struct PendingBundleDelete {
    id: i64,
    name: String,
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
struct LabelEdit {
    tier_id: i64,
    annotation_id: i64,
    kind: LabelEditKind,
    text: String,
    /// Set to `true` for the first frame so the TextEdit grabs
    /// focus; cleared after.
    just_started: bool,
}

#[derive(Debug, Clone, Copy)]
enum LabelEditKind {
    Interval,
    Point,
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
            selected_annotation: None,
            timeline: TimelineState::default(),
            playback: None,
            draft_edit: DraftEdit::None,
            label_edit: None,
            script_output: None,
            script_error: None,
            registered_commands: Vec::new(),
            command_palette_open: false,
            command_palette_query: String::new(),
            info: None,
            record_dialog: None,
            pending_delete: None,
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

    /// Opens the recording-modal dialog. State is reset on every
    /// open.
    fn open_record_dialog(&mut self) {
        self.record_dialog = Some(RecordDialog::new());
    }

    /// Begins capture using `self.record_dialog`'s configured device
    /// + format. Builds the engine session, spawns the cpal stream
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
                    latest_peak = m.rms_db;
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
    fn label_edit_window(&mut self, ctx: &egui::Context) {
        let Some(le) = self.label_edit.as_mut() else {
            return;
        };
        let mut commit = false;
        let mut cancel = false;
        let mut keep_open = true;
        egui::Window::new("Edit label")
            .collapsible(false)
            .resizable(false)
            .open(&mut keep_open)
            .show(ctx, |ui| {
                let resp = ui.text_edit_singleline(&mut le.text);
                if le.just_started {
                    resp.request_focus();
                    le.just_started = false;
                }
                if resp.lost_focus() {
                    if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        commit = true;
                    } else if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                        cancel = true;
                    }
                }
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
            let le = self.label_edit.take().expect("checked above");
            let AppState::ProjectLoaded { project, .. } = &self.app_state else {
                return;
            };
            let new_label = if le.text.is_empty() {
                None
            } else {
                Some(le.text)
            };
            // Re-fetch the base row at commit so we preserve any
            // non-label fields that might have changed elsewhere
            // (resize, move, parent_annotation_id update) since
            // the label-edit window opened.
            let result: Result<(), String> = match le.kind {
                LabelEditKind::Interval => match project.intervals(le.tier_id) {
                    Ok(rows) => match rows.into_iter().find(|r| r.id == le.annotation_id) {
                        Some(existing) => project
                            .update_interval(
                                le.annotation_id,
                                &sadda_engine::IntervalSpec {
                                    tier_id: le.tier_id,
                                    start_seconds: existing.start_seconds,
                                    end_seconds: existing.end_seconds,
                                    label: new_label,
                                    parent_annotation_id: existing.parent_annotation_id,
                                    extra: existing.extra,
                                },
                            )
                            .map_err(|e| format!("Failed to save label: {e}")),
                        None => Ok(()), // deleted concurrently
                    },
                    Err(e) => Err(format!("Failed to reload interval: {e}")),
                },
                LabelEditKind::Point => match project.points(le.tier_id) {
                    Ok(rows) => match rows.into_iter().find(|r| r.id == le.annotation_id) {
                        Some(existing) => project
                            .update_point(
                                le.annotation_id,
                                &sadda_engine::PointSpec {
                                    tier_id: le.tier_id,
                                    time_seconds: existing.time_seconds,
                                    label: new_label,
                                    parent_annotation_id: existing.parent_annotation_id,
                                    extra: existing.extra,
                                },
                            )
                            .map_err(|e| format!("Failed to save label: {e}")),
                        None => Ok(()),
                    },
                    Err(e) => Err(format!("Failed to reload point: {e}")),
                },
            };
            if let Err(msg) = result {
                self.set_error(msg);
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

// ---------------------------------------------------------------------------
// Tier-strip lane renderers
// ---------------------------------------------------------------------------

/// Pixel hit-zone width for grabbing an interval boundary.
const BOUNDARY_HIT_ZONE_PX: f32 = 6.0;
/// Minimum drag length (in seconds) before a drag-to-create commits.
/// Smaller drags are treated as plain clicks (no new interval).
const MIN_DRAFT_CREATE_SECONDS: f64 = 0.005;

#[allow(clippy::too_many_arguments)]
fn render_interval_lane(
    painter: &egui::Painter,
    rect: egui::Rect,
    view_start: f64,
    view_end: f64,
    tier_id: i64,
    rows: &[sadda_engine::Interval],
    selection: Option<AnnotationSelection>,
    response: &egui::Response,
    draft: &DraftEdit,
    new_selection: &mut Option<AnnotationSelection>,
    new_cursor: &mut Option<f64>,
    new_draft_action: &mut Option<DraftAction>,
    label_edit_request: &mut Option<LabelEdit>,
) {
    let view_range = (view_end - view_start).max(1e-6);
    let lane_width = rect.width() as f64;
    let x_per_second = lane_width / view_range;
    let base_fill = egui::Color32::from_rgb(82, 138, 198);
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

    // Double-click on an interval body → start label edit.
    if response.double_clicked() {
        if let Some(p) = response.interact_pointer_pos() {
            for r in rows {
                let x0 = rect.left() + ((r.start_seconds - view_start) * x_per_second) as f32;
                let x1 = rect.left() + ((r.end_seconds - view_start) * x_per_second) as f32;
                if p.x >= x0 && p.x <= x1 && rect.y_range().contains(p.y) {
                    *label_edit_request = Some(LabelEdit {
                        tier_id,
                        annotation_id: r.id,
                        kind: LabelEditKind::Interval,
                        text: r.label.clone().unwrap_or_default(),
                        just_started: true,
                    });
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
    selection: Option<AnnotationSelection>,
    response: &egui::Response,
    draft: &DraftEdit,
    new_selection: &mut Option<AnnotationSelection>,
    new_cursor: &mut Option<f64>,
    new_draft_action: &mut Option<DraftAction>,
    label_edit_request: &mut Option<LabelEdit>,
) {
    let view_range = (view_end - view_start).max(1e-6);
    let lane_width = rect.width() as f64;
    let x_per_second = lane_width / view_range;
    let base_color = egui::Color32::from_rgb(230, 180, 70);
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

    // Double-click on an existing point → label edit.
    if response.double_clicked() {
        if let Some(p) = response.interact_pointer_pos() {
            for row in rows {
                let x = rect.left() + ((row.time_seconds - view_start) * x_per_second) as f32;
                if (p.x - x).abs() <= POINT_HIT_ZONE_PX && rect.y_range().contains(p.y) {
                    *label_edit_request = Some(LabelEdit {
                        tier_id,
                        annotation_id: row.id,
                        kind: LabelEditKind::Point,
                        text: row.label.clone().unwrap_or_default(),
                        just_started: true,
                    });
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
        self.apply_theme(ui.ctx());

        // Drive the playback-cursor advance before any pane
        // renders, so they all see the same `timeline.cursor` this
        // frame. Repaint continuously while playing so the cursor
        // line stays in sync without user input.
        self.poll_playback();
        if self.playback.is_some() {
            ui.ctx().request_repaint();
        }

        // Spacebar toggles transport. `consume_key` ensures the
        // press doesn't fall through to any focused widget.
        if ui
            .ctx()
            .input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Space))
        {
            self.toggle_playback();
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
        // when label editing is active — those keys need to reach
        // the TextEdit instead.
        if self.label_edit.is_none() {
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

        // E9 command palette. Same overlay pattern.
        self.command_palette_window(ui.ctx());

        // H1 live-recording modal.
        self.render_record_dialog(ui.ctx());

        // H1 bundle-delete confirmation modal.
        self.render_pending_delete(ui.ctx());

        egui::Panel::top("menu").show_inside(ui, |ui| self.menu_bar(ui));

        if let AppState::ProjectLoaded { name, root, .. } = &self.app_state {
            let label = format!("Project: {name}  ·  {}", root.display());
            egui::Panel::bottom("status").show_inside(ui, |ui| {
                ui.label(label);
            });
        }

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
            self.view_menu(ui);
            self.help_menu(ui);
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
            // E8: script-panel toggle. Persists across launches.
            ui.checkbox(&mut self.persisted.script_panel_open, "Show Script Panel");
        });
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
                    if ui.button("Reveal in file manager").clicked() {
                        to_reveal = Some((b.id, b.audio_relative_path.clone()));
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
            .show_inside(ui, |ui| self.waveform_pane(ui));

        // Bottom sub-panel: tier strip. Resizable; default height
        // scales with the number of tiers up to a sensible cap.
        let n_lanes = self.estimate_tier_lane_count();
        let default_strip_height = ((n_lanes as f32).max(1.0) * TIER_LANE_HEIGHT + 8.0).min(220.0);
        egui::Panel::bottom("tier_strip")
            .resizable(true)
            .default_size(default_strip_height)
            .min_size(TIER_LANE_HEIGHT + 8.0)
            .show_inside(ui, |ui| self.tier_strip_pane(ui));

        // Centre: spectrogram fills the remainder.
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
        let cursor = self.timeline.cursor;
        let mut clicked_time: Option<f64> = None;

        let plot_response = Plot::new("waveform")
            .show_axes([true, true])
            .y_axis_label("amplitude")
            .x_axis_label("seconds")
            .include_y(-1.0)
            .include_y(1.0)
            .include_x(self.timeline.view_start)
            .include_x(self.timeline.view_end)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
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

                // Click in the plot positions the cursor; map the
                // pointer-coord (in plot space) directly.
                let resp = plot_ui.response();
                if resp.clicked() {
                    if let Some(pos) = resp.interact_pointer_pos() {
                        let tx = plot_ui.plot_from_screen(pos);
                        clicked_time = Some(tx.x);
                    }
                }
            })
            .response;

        if let Some(t) = clicked_time {
            self.timeline.set_cursor(t);
        }
        handle_zoom_and_scroll(&plot_response, &mut self.timeline);
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
        // Image still covers the full bundle; the plot's
        // include_x bounds crop the visible region for free.
        let centre = egui_plot::PlotPoint::new(duration / 2.0, nyquist / 2.0);
        let size = egui::Vec2::new(duration as f32, nyquist as f32);
        let texture_id = sc.texture.id();
        let cursor = self.timeline.cursor;
        let mut clicked_time: Option<f64> = None;

        let plot_response = Plot::new("spectrogram")
            .show_axes([true, true])
            .y_axis_label("Hz")
            .x_axis_label("seconds")
            .include_x(self.timeline.view_start)
            .include_x(self.timeline.view_end)
            .include_y(0.0)
            .include_y(nyquist)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .show(ui, |plot_ui| {
                plot_ui.image(egui_plot::PlotImage::new(
                    "spectrogram_img",
                    texture_id,
                    centre,
                    size,
                ));
                draw_cursor_line(plot_ui, cursor, 0.0, nyquist);

                let resp = plot_ui.response();
                if resp.clicked() {
                    if let Some(pos) = resp.interact_pointer_pos() {
                        let tx = plot_ui.plot_from_screen(pos);
                        clicked_time = Some(tx.x);
                    }
                }
            })
            .response;

        if let Some(t) = clicked_time {
            self.timeline.set_cursor(t);
        }
        handle_zoom_and_scroll(&plot_response, &mut self.timeline);
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
        if tiers.is_empty() {
            ui.label(egui::RichText::new("(no tiers in this bundle yet)").italics());
            return;
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

        egui::ScrollArea::vertical().show(ui, |ui| {
            for tier in &tiers {
                ui.horizontal(|ui| {
                    // Left gutter: tier name + type hint.
                    ui.allocate_ui_with_layout(
                        egui::Vec2::new(TIER_LABEL_GUTTER, TIER_LANE_HEIGHT),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            ui.label(egui::RichText::new(truncate_label(&tier.name, 16)).strong());
                        },
                    );
                    // Right: time-positioned lane.
                    let avail = ui.available_size_before_wrap();
                    let lane_size = egui::Vec2::new(avail.x, TIER_LANE_HEIGHT);
                    let (rect, response) =
                        ui.allocate_exact_size(lane_size, egui::Sense::click_and_drag());
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
                                self.selected_annotation,
                                &response,
                                &self.draft_edit,
                                &mut new_selection,
                                &mut new_cursor,
                                &mut new_draft_action,
                                &mut label_edit_request,
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
                                self.selected_annotation,
                                &response,
                                &self.draft_edit,
                                &mut new_selection,
                                &mut new_cursor,
                                &mut new_draft_action,
                                &mut label_edit_request,
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

        egui::ScrollArea::vertical()
            .id_salt("script_code_scroll")
            .max_height(code_h)
            .show(ui, |ui| {
                ui.add_sized(
                    [ui.available_width(), code_h],
                    egui::TextEdit::multiline(&mut self.persisted.script_buffer)
                        .desired_rows(8)
                        .code_editor()
                        .hint_text("# Python — pure stdlib only at E8.\n# `import sadda` lands in E9.\nprint('hello from sadda')\n"),
                );
            });

        ui.separator();
        ui.label(egui::RichText::new("Output").weak().small());
        egui::ScrollArea::vertical()
            .id_salt("script_output_scroll")
            .max_height(output_h)
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
