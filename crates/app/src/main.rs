//! Sadda desktop GUI. Slice A1 ships the project-aware shell —
//! welcome screen with New / Open / Recent, a `Project`-backed loaded
//! state, persistent window + recent-projects state. No content panes
//! yet; those land in cluster B (waveform / spectrogram / tier strip).
//!
//! See the 2026-05-23 DEVLOG entry "App shell + project open/create
//! (A1)" for the design rationale and the cut-list for what
//! deliberately doesn't ship at A1.

mod playback;
mod state;

use std::path::{Path, PathBuf};

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use sadda_engine::{Project, TierType};

use crate::playback::Playback;
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
        self.timeline
            .reset_for_bundle(self.active_envelope.as_ref().unwrap().duration_seconds);
    }

    fn clear_bundle_selection(&mut self) {
        self.selected_bundle_id = None;
        self.active_envelope = None;
        self.active_spectrogram = None;
        self.selected_annotation = None;
        self.playback = None;
        self.timeline = TimelineState::default();
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
    new_selection: &mut Option<AnnotationSelection>,
    new_cursor: &mut Option<f64>,
) {
    let view_range = (view_end - view_start).max(1e-6);
    let lane_width = rect.width() as f64;
    let x_per_second = lane_width / view_range;
    let base_fill = egui::Color32::from_rgb(82, 138, 198);
    let selected_fill = egui::Color32::from_rgb(160, 200, 250);
    let text_color = egui::Color32::WHITE;
    let click_pos = response.interact_pointer_pos();

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
        if let Some(p) = click_pos
            && item_rect.contains(p)
            && response.clicked()
        {
            *new_selection = Some(AnnotationSelection::Interval {
                tier_id,
                annotation_id: r.id,
            });
            // Per the C5 design: clicking an annotation also moves
            // the cursor to its start, so the cluster of panes all
            // re-centre on the selection.
            *new_cursor = Some(r.start_seconds);
        }
    }
}

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
    new_selection: &mut Option<AnnotationSelection>,
    new_cursor: &mut Option<f64>,
) {
    let view_range = (view_end - view_start).max(1e-6);
    let lane_width = rect.width() as f64;
    let x_per_second = lane_width / view_range;
    let base_color = egui::Color32::from_rgb(230, 180, 70);
    let selected_color = egui::Color32::from_rgb(255, 220, 120);
    let click_pos = response.interact_pointer_pos();
    // Pick the nearest point to a click within this tolerance.
    let click_tolerance_px = 6.0;

    for p in rows {
        if p.time_seconds < view_start || p.time_seconds > view_end {
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
        if let Some(label) = &p.label {
            if !label.is_empty() {
                painter.text(
                    egui::Pos2::new(x + 3.0, rect.top() + 2.0),
                    egui::Align2::LEFT_TOP,
                    truncate_label(label, TIER_LABEL_MAX_CHARS),
                    egui::FontId::proportional(11.0),
                    colour,
                );
            }
        }
        if let Some(cp) = click_pos
            && (cp.x - x).abs() <= click_tolerance_px
            && rect.contains(cp)
            && response.clicked()
        {
            *new_selection = Some(AnnotationSelection::Point {
                tier_id,
                annotation_id: p.id,
            });
            *new_cursor = Some(p.time_seconds);
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
            let bundle_enabled = matches!(self.app_state, AppState::ProjectLoaded { .. });
            if ui
                .add_enabled(bundle_enabled, egui::Button::new("Open Bundle…"))
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
                    let (rect, response) = ui.allocate_exact_size(lane_size, egui::Sense::click());
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
                                &mut new_selection,
                                &mut new_cursor,
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
                                &mut new_selection,
                                &mut new_cursor,
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
