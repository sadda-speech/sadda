//! Sadda desktop GUI. Slice A1 ships the project-aware shell —
//! welcome screen with New / Open / Recent, a `Project`-backed loaded
//! state, persistent window + recent-projects state. No content panes
//! yet; those land in cluster B (waveform / spectrogram / tier strip).
//!
//! See the 2026-05-23 DEVLOG entry "App shell + project open/create
//! (A1)" for the design rationale and the cut-list for what
//! deliberately doesn't ship at A1.

mod state;

use std::path::{Path, PathBuf};

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use sadda_engine::Project;

use crate::state::{
    ColormapKind, EnvelopeCache, PersistedState, SpectrogramConfig, ThemePref, build_envelope,
    colormap_bake, power_to_db_normalized,
};

const APP_TITLE: &str = "sadda";
/// Bucket count for the B2 fixed-resolution waveform envelope. C5 will
/// replace this with per-frame re-bucketing once zoom lands.
const ENVELOPE_BUCKETS: usize = 2000;
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
        let envelope = build_envelope(&mono, ENVELOPE_BUCKETS);
        self.active_envelope = Some(EnvelopeCache {
            bundle_id,
            sample_rate: audio.sample_rate,
            duration_seconds: audio.duration_seconds(),
            envelope,
            mono_samples: mono,
        });
        self.selected_bundle_id = Some(bundle_id);
        self.active_spectrogram = None;
    }

    fn clear_bundle_selection(&mut self) {
        self.selected_bundle_id = None;
        self.active_envelope = None;
        self.active_spectrogram = None;
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

impl eframe::App for SaddaApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.persisted);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.apply_theme(ui.ctx());

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

    /// Central content for a loaded project: caption + waveform on
    /// top (resizable), spectrogram filling the rest.
    fn bundle_content_pane(&mut self, ui: &mut egui::Ui) {
        // Top sub-panel: waveform. Resizable; user can drag the
        // divider to rebalance with the spectrogram.
        egui::Panel::top("waveform_split")
            .resizable(true)
            .default_size(220.0)
            .min_size(80.0)
            .show_inside(ui, |ui| self.waveform_pane(ui));

        // Bottom: spectrogram fills the remainder.
        self.rebuild_spectrogram_if_stale(ui.ctx());
        self.spectrogram_pane(ui);
    }

    fn waveform_pane(&mut self, ui: &mut egui::Ui) {
        let Some(env) = &self.active_envelope else {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("Select a bundle from the sidebar").weak());
            });
            return;
        };
        if env.envelope.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label(egui::RichText::new("(empty waveform)").italics());
            });
            return;
        }
        // Caption above the plot summarises the loaded bundle's
        // header so users see sample-rate / duration at a glance.
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!(
                    "Bundle #{}  ·  {} Hz  ·  {:.3}s",
                    env.bundle_id, env.sample_rate, env.duration_seconds,
                ))
                .weak(),
            );
        });
        // Build (time, min) and (time, max) point arrays once per
        // frame from the cached envelope, then plot each bucket as a
        // single vertical line segment from min to max.
        let n = env.envelope.len() as f64;
        let dt = env.duration_seconds / n;
        Plot::new("waveform")
            .show_axes([true, true])
            .y_axis_label("amplitude")
            .x_axis_label("seconds")
            .include_y(-1.0)
            .include_y(1.0)
            .include_x(0.0)
            .include_x(env.duration_seconds)
            .show(ui, |plot_ui| {
                for (i, (mn, mx)) in env.envelope.iter().enumerate() {
                    let t = (i as f64 + 0.5) * dt;
                    let segment = PlotPoints::from(vec![[t, *mn as f64], [t, *mx as f64]]);
                    plot_ui
                        .line(Line::new("", segment).color(egui::Color32::from_rgb(80, 140, 220)));
                }
            });
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
        let centre = egui_plot::PlotPoint::new(duration / 2.0, nyquist / 2.0);
        let size = egui::Vec2::new(duration as f32, nyquist as f32);
        let texture_id = sc.texture.id();

        Plot::new("spectrogram")
            .show_axes([true, true])
            .y_axis_label("Hz")
            .x_axis_label("seconds")
            .include_x(0.0)
            .include_x(duration)
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
            });
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
