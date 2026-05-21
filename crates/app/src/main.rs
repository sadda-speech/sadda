//! Desktop GUI for sadda. Phase 0 vertical-slice version: loads a WAV via the
//! engine and renders waveform + f0 overlay. egui+wgpu native window.

use std::path::PathBuf;

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use sadda_engine::{Audio, PitchConfig, PitchFrame, autocorrelation};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 640.0])
            .with_title("sadda — pre-alpha"),
        ..Default::default()
    };
    eframe::run_native(
        "sadda",
        options,
        Box::new(|_cc| Ok(Box::<SaddaApp>::default())),
    )
}

#[derive(Default)]
struct SaddaApp {
    audio: Option<Audio>,
    audio_path: Option<PathBuf>,
    pitch_frames: Vec<PitchFrame>,
    waveform_points: Vec<[f64; 2]>,
    status: String,
    error: Option<String>,
}

impl SaddaApp {
    fn open_file(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("WAV", &["wav"])
            .pick_file()
        {
            self.load_path(path);
        }
    }

    fn load_path(&mut self, path: PathBuf) {
        self.error = None;
        match Audio::from_wav_path(&path) {
            Ok(audio) => {
                self.status = format!(
                    "{} — {} Hz, {} ch, {:.3}s",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    audio.sample_rate,
                    audio.channels,
                    audio.duration_seconds()
                );
                self.pitch_frames = autocorrelation(&audio, &PitchConfig::default());
                self.waveform_points = downsample_for_display(&audio, 2000);
                self.audio = Some(audio);
                self.audio_path = Some(path);
            }
            Err(e) => {
                self.error = Some(e.to_string());
                self.status.clear();
            }
        }
    }
}

/// Downsamples the mono mixdown to roughly `target_points` (time, amplitude)
/// pairs for waveform display. Picks one sample per bucket — fast and visually
/// adequate for Phase 0. A min/max envelope would be a refinement later.
fn downsample_for_display(audio: &Audio, target_points: usize) -> Vec<[f64; 2]> {
    let mono: Vec<f32> = audio.mono_samples().collect();
    if mono.is_empty() {
        return Vec::new();
    }
    let bucket = (mono.len() / target_points).max(1);
    let sample_period = 1.0 / audio.sample_rate as f64;
    mono.iter()
        .step_by(bucket)
        .enumerate()
        .map(|(i, &s)| [i as f64 * bucket as f64 * sample_period, s as f64])
        .collect()
}

impl eframe::App for SaddaApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("menu").show_inside(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open WAV…").clicked() {
                        ui.close();
                        self.open_file();
                    }
                });
                ui.separator();
                ui.label(&self.status);
            });
        });

        if let Some(err) = self.error.clone() {
            egui::Panel::bottom("error").show_inside(ui, |ui| {
                ui.colored_label(egui::Color32::RED, err);
            });
        }

        egui::CentralPanel::default().show_inside(ui, |ui| {
            if self.audio.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label("Open a WAV file via File → Open WAV…");
                });
                return;
            }

            let available_height = ui.available_height();
            let pane_height = (available_height - 32.0) * 0.5;

            ui.heading("Waveform");
            Plot::new("waveform")
                .height(pane_height)
                .show_axes([true, true])
                .y_axis_label("amplitude")
                .x_axis_label("seconds")
                .include_y(-1.0)
                .include_y(1.0)
                .show(ui, |plot_ui| {
                    let pts = PlotPoints::from(self.waveform_points.clone());
                    plot_ui.line(Line::new("waveform", pts));
                });

            ui.separator();
            ui.heading("f0 (autocorrelation)");
            Plot::new("pitch")
                .height(pane_height)
                .show_axes([true, true])
                .y_axis_label("Hz")
                .x_axis_label("seconds")
                .show(ui, |plot_ui| {
                    let pts: Vec<[f64; 2]> = self
                        .pitch_frames
                        .iter()
                        .map(|f| [f.time_seconds, f.frequency_hz as f64])
                        .collect();
                    plot_ui.line(Line::new("f0", PlotPoints::from(pts)));
                });
        });
    }
}
