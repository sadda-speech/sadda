//! Pure-data app state. No egui types here — that's the point. Tests
//! cover the recent-projects ordering / capping / cleanup logic plus
//! the waveform-envelope downsampler without spinning up an egui
//! context.

use std::path::{Path, PathBuf};

/// Maximum number of recent-project entries kept in persisted state
/// and shown on the welcome screen.
pub const MAX_RECENT_PROJECTS: usize = 5;

/// Theme preference. `System` means follow the OS dark/light setting
/// at startup; the user can override via the View menu.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ThemePref {
    /// Follow the OS preference at app startup.
    #[default]
    System,
    /// Force the light palette.
    Light,
    /// Force the dark palette.
    Dark,
}

impl ThemePref {
    /// Maps a normalized theme name (`"light"` / `"dark"` / `"system"`) to a
    /// preference. Used by the `sadda.app.set_theme` / `sadda.doc` scripting
    /// surface; returns `None` for anything else.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "light" => Some(Self::Light),
            "dark" => Some(Self::Dark),
            "system" => Some(Self::System),
            _ => None,
        }
    }
}

/// Horizontal alignment of interval labels within their box in the tier strip.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LabelAlign {
    #[default]
    Left,
    Center,
    Right,
}

impl LabelAlign {
    /// Menu label for the alignment picker.
    pub fn label(self) -> &'static str {
        match self {
            Self::Left => "Left",
            Self::Center => "Center",
            Self::Right => "Right",
        }
    }
}

/// State that survives across launches. Eframe's `Storage` hook
/// serializes this via serde; window size + position are persisted
/// separately by eframe itself.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersistedState {
    /// Most-recently-opened projects, newest first. Capped at
    /// [`MAX_RECENT_PROJECTS`].
    #[serde(default)]
    pub recent_projects: Vec<PathBuf>,
    /// Theme preference; defaults to `System`.
    #[serde(default)]
    pub theme: ThemePref,
    /// Last-used spectrogram window / hop / colormap / dB-range.
    #[serde(default)]
    pub spectrogram: SpectrogramConfig,
    /// D10: last-used measure-track lane visibility + analysis params.
    #[serde(default)]
    pub tracks: MeasureTrackConfig,
    /// D10: reference distribution overlaid on the f0 lane, if any.
    #[serde(default)]
    pub f0_overlay: Option<RefdistOverlay>,
    /// D10: reference distribution overlaid on the intensity lane, if any.
    #[serde(default)]
    pub intensity_overlay: Option<RefdistOverlay>,
    /// D10: whether the right-side Reference panel (vowel-space scatter +
    /// 1-D histogram) is shown.
    #[serde(default)]
    pub reference_panel_open: bool,
    /// D10: distribution shown in the Reference panel (`sex` narrows the
    /// subgroup). The vowel-space scatter uses its first two parameters;
    /// the histogram uses [`PersistedState::reference_param`].
    #[serde(default)]
    pub reference_dist: Option<RefdistOverlay>,
    /// D10: phone filter for the Reference panel's vowel-space scatter.
    #[serde(default)]
    pub reference_phone: Option<String>,
    /// D10: which parameter the Reference panel's histogram bins. Defaults
    /// to the distribution's first parameter when unset.
    #[serde(default)]
    pub reference_param: Option<String>,
    /// E8: whether the embedded-CPython script panel is currently
    /// shown at the bottom of the app.
    #[serde(default)]
    pub script_panel_open: bool,
    /// S2/annotation: whether the right-side Annotation panel (inline editor
    /// for the selected annotation's label / status / note) is shown.
    #[serde(default)]
    pub annotation_panel_open: bool,
    /// E8: persisted script-editor buffer. Survives relaunches so
    /// users don't lose typed scripts.
    #[serde(default)]
    pub script_buffer: String,
    /// Accessibility: colour scheme for the measure-track lanes and
    /// reference overlays. See [`PlotPalette`].
    #[serde(default)]
    pub palette: PlotPalette,
    /// Accessibility: UI zoom factor (egui `zoom_factor`) — scales all
    /// text and widgets relative to the native pixel density. 1.0 =
    /// native; the Appearance menu exposes ~0.8–2.0.
    #[serde(default = "default_ui_scale")]
    pub ui_scale: f32,
    /// Embedding-heatmap lane state: which `continuous_vector` tier is
    /// selected, plus colormap + normalization knobs. Default leaves the
    /// lane hidden (no selected tier).
    #[serde(default)]
    pub embedding: EmbeddingHeatmapConfig,
    /// MFCC lane state: active preset + parameters + colormap/normalization.
    /// Default leaves the lane hidden.
    #[serde(default)]
    pub mfcc: MfccLaneConfig,
    /// Id of the f0 preset currently selected in View ▸ DSP methods (display
    /// only — the actual params live in `tracks.pitch_params`). Lets the menu
    /// show the active preset and flag edits as "(modified)".
    #[serde(default = "default_pitch_preset_id")]
    pub pitch_preset_id: String,
    /// Id of the formant preset selected in View ▸ DSP methods (display only —
    /// the actual params live in `tracks.formant_params`).
    #[serde(default = "default_formant_preset_id")]
    pub formant_preset_id: String,
    /// S3: visibility of the structural signal subpanes (waveform, spectrogram,
    /// tier strip), which had no toggle before. Complements the measure-lane
    /// toggles in `tracks` / `mfcc` / `embedding`. Default: all shown.
    #[serde(default)]
    pub panes: SignalPaneVisibility,
    /// S3: annotation tiers explicitly hidden from the tier strip, by tier id
    /// (per-tier in/out selection). Stale ids — deleted tiers, or ids from
    /// another project — are harmless no-ops. Default: empty (all tiers shown).
    #[serde(default)]
    pub hidden_tier_ids: std::collections::HashSet<i64>,
    /// Horizontal alignment of interval labels within their box. Default left.
    #[serde(default)]
    pub interval_label_align: LabelAlign,
}

fn default_pitch_preset_id() -> String {
    "praat-ac".to_string()
}

fn default_formant_preset_id() -> String {
    "praat-burg".to_string()
}

/// Default UI zoom factor (native size). A free fn because `serde`'s
/// `default` needs a path, and `f32::default()` would give `0.0`.
fn default_ui_scale() -> f32 {
    1.0
}

impl Default for PersistedState {
    fn default() -> Self {
        Self {
            recent_projects: Vec::new(),
            theme: ThemePref::default(),
            spectrogram: SpectrogramConfig::default(),
            tracks: MeasureTrackConfig::default(),
            f0_overlay: None,
            intensity_overlay: None,
            reference_panel_open: false,
            reference_dist: None,
            reference_phone: None,
            reference_param: None,
            script_panel_open: false,
            annotation_panel_open: false,
            script_buffer: String::new(),
            palette: PlotPalette::default(),
            ui_scale: default_ui_scale(), // 1.0, not 0.0
            embedding: EmbeddingHeatmapConfig::default(),
            mfcc: MfccLaneConfig::default(),
            pitch_preset_id: default_pitch_preset_id(),
            formant_preset_id: default_formant_preset_id(),
            panes: SignalPaneVisibility::default(),
            hidden_tier_ids: std::collections::HashSet::new(),
            interval_label_align: LabelAlign::default(),
        }
    }
}

/// G0 (figure-export groundwork): a single consolidated descriptor of which
/// signal-column lanes are currently visible.
///
/// Lane visibility is otherwise scattered across several `PersistedState`
/// fields — structural panes in `panes`, measure lanes in `tracks`,
/// `mfcc.show`, and the embedding tier selection — with no one place to ask
/// "what is on screen?". The figure-export dialog (G1) defaults its
/// per-element include checkboxes from this, and the headless exporter reads
/// it to decide which lanes to draw, so a figure matches what the user saw.
///
// Consumed by the figure-export dialog + serializer in G1; unused until then,
// hence the `allow` (the accessor + its tests are the G0 deliverable).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisibleLanes {
    pub waveform: bool,
    pub spectrogram: bool,
    pub f0: bool,
    pub formants: bool,
    pub intensity: bool,
    pub vad: bool,
    pub mfcc: bool,
    pub embedding: bool,
    pub tier_strip: bool,
}

impl PersistedState {
    /// Builds the current [`VisibleLanes`] descriptor from the scattered
    /// per-lane visibility flags. Pure — reads only persisted state, no live
    /// GUI context, so the headless doc/figure path can call it too.
    ///
    /// The embedding lane is "visible" exactly when a tier is selected for it
    /// (its default `None` selection is what hides it).
    // Wired into the G1 export dialog; unused until that slice lands.
    #[allow(dead_code)]
    pub fn visible_lanes(&self) -> VisibleLanes {
        VisibleLanes {
            waveform: self.panes.waveform,
            spectrogram: self.panes.spectrogram,
            f0: self.tracks.f0_visible,
            formants: self.tracks.formants_visible,
            intensity: self.tracks.intensity_visible,
            vad: self.tracks.vad_visible,
            mfcc: self.mfcc.show,
            embedding: self.embedding.selected_tier_id.is_some(),
            tier_strip: self.panes.tier_strip,
        }
    }

    /// Records a project open. Moves the path to the front of the list,
    /// removing duplicates and capping the total at
    /// [`MAX_RECENT_PROJECTS`].
    pub fn record_open(&mut self, path: impl Into<PathBuf>) {
        let path = path.into();
        self.recent_projects.retain(|p| p != &path);
        self.recent_projects.insert(0, path);
        self.recent_projects.truncate(MAX_RECENT_PROJECTS);
    }

    /// Removes a specific path from the recent list. Returns true if it
    /// was present. Used when a recent row's path no longer exists on
    /// disk and the user clicks it to dismiss the entry.
    pub fn remove_recent(&mut self, path: &Path) -> bool {
        let before = self.recent_projects.len();
        self.recent_projects.retain(|p| p != path);
        self.recent_projects.len() != before
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn measure_track_default_shows_only_f0() {
        let cfg = MeasureTrackConfig::default();
        assert!(cfg.f0_visible);
        assert!(!cfg.formants_visible);
        assert!(!cfg.intensity_visible);
        assert!(cfg.any_visible());
    }

    #[test]
    fn visible_lanes_reflects_defaults() {
        // Default: structural panes all shown, only f0 among measure lanes,
        // mfcc + embedding hidden.
        let lanes = PersistedState::default().visible_lanes();
        assert!(lanes.waveform);
        assert!(lanes.spectrogram);
        assert!(lanes.tier_strip);
        assert!(lanes.f0);
        assert!(!lanes.formants);
        assert!(!lanes.intensity);
        assert!(!lanes.vad);
        assert!(!lanes.mfcc);
        assert!(!lanes.embedding);
    }

    #[test]
    fn visible_lanes_tracks_each_source_field() {
        let mut s = PersistedState::default();
        s.panes.waveform = false;
        s.tracks.formants_visible = true;
        s.mfcc.show = true;
        s.embedding.selected_tier_id = Some(7);
        let lanes = s.visible_lanes();
        assert!(!lanes.waveform, "structural pane toggle feeds through");
        assert!(lanes.formants, "measure-lane toggle feeds through");
        assert!(lanes.mfcc, "mfcc.show feeds through");
        assert!(
            lanes.embedding,
            "a selected embedding tier makes it visible"
        );
    }

    #[test]
    fn measure_track_any_visible_false_when_all_off() {
        let cfg = MeasureTrackConfig {
            f0_visible: false,
            formants_visible: false,
            intensity_visible: false,
            ..MeasureTrackConfig::default()
        };
        assert!(!cfg.any_visible());
    }

    #[test]
    fn dsp_method_defaults_match_engine_defaults() {
        // The GUI defaults must mirror the engine's chosen defaults so an
        // untouched install behaves identically to the API's `*::default()`.
        let cfg = MeasureTrackConfig::default();
        assert_eq!(
            cfg.pitch_params,
            sadda_engine::pitch::PitchParams::default()
        );
        assert_eq!(
            cfg.pitch_params.method,
            sadda_engine::pitch::PitchMethod::Boersma
        );
        assert_eq!(
            cfg.formant_params,
            sadda_engine::dsp::FormantsConfig::default()
        );
        assert_eq!(
            cfg.formant_params.lpc_method,
            sadda_engine::dsp::lpc::LpcMethod::Burg
        );
    }

    #[test]
    fn dsp_method_changes_invalidate_cache() {
        // Changing the f0 or formant method must change config equality
        // (that's how the track cache detects staleness).
        let a = MeasureTrackConfig::default();
        let mut b = MeasureTrackConfig::default();
        b.pitch_params.method = sadda_engine::pitch::PitchMethod::Yin;
        assert_ne!(a, b);
        let mut c = MeasureTrackConfig::default();
        c.formant_params.lpc_method = sadda_engine::dsp::lpc::LpcMethod::Autocorrelation;
        assert_ne!(a, c);
    }

    #[test]
    fn measure_track_any_visible_true_when_one_on() {
        let cfg = MeasureTrackConfig {
            f0_visible: false,
            formants_visible: true,
            intensity_visible: false,
            ..MeasureTrackConfig::default()
        };
        assert!(cfg.any_visible());
    }

    #[test]
    fn signal_panes_default_to_all_shown() {
        let p = SignalPaneVisibility::default();
        assert!(p.waveform && p.spectrogram && p.tier_strip);
    }

    #[test]
    fn persisted_state_without_pane_fields_restores_all_shown() {
        // Older persisted state predates `panes` / `hidden_tier_ids`. Missing
        // fields must restore to "everything visible", preserving the old
        // always-on behaviour rather than blanking the signal column.
        let s: PersistedState = serde_json::from_str("{}").expect("empty object is valid");
        assert!(s.panes.waveform && s.panes.spectrogram && s.panes.tier_strip);
        assert!(s.hidden_tier_ids.is_empty());
    }

    #[test]
    fn signal_pane_visibility_survives_a_round_trip() {
        let p = SignalPaneVisibility {
            waveform: true,
            spectrogram: false,
            tier_strip: true,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(
            serde_json::from_str::<SignalPaneVisibility>(&json).unwrap(),
            p
        );
    }

    #[test]
    fn nearest_frame_index_picks_closest() {
        let times = [0.0, 0.1, 0.2, 0.3];
        assert_eq!(nearest_frame_index(&times, 0.0), Some(0));
        assert_eq!(nearest_frame_index(&times, 0.17), Some(2));
        assert_eq!(nearest_frame_index(&times, 0.24), Some(2));
        assert_eq!(nearest_frame_index(&times, 99.0), Some(3));
    }

    #[test]
    fn nearest_frame_index_empty_is_none() {
        assert_eq!(nearest_frame_index(&[], 0.5), None);
    }

    #[test]
    fn record_open_inserts_at_front() {
        let mut s = PersistedState::default();
        s.record_open(PathBuf::from("/a"));
        s.record_open(PathBuf::from("/b"));
        assert_eq!(
            s.recent_projects,
            vec![PathBuf::from("/b"), PathBuf::from("/a")]
        );
    }

    #[test]
    fn record_open_dedupes_existing_entry_to_front() {
        let mut s = PersistedState::default();
        s.record_open(PathBuf::from("/a"));
        s.record_open(PathBuf::from("/b"));
        s.record_open(PathBuf::from("/a"));
        assert_eq!(
            s.recent_projects,
            vec![PathBuf::from("/a"), PathBuf::from("/b")]
        );
    }

    #[test]
    fn record_open_caps_at_max() {
        let mut s = PersistedState::default();
        for i in 0..10 {
            s.record_open(PathBuf::from(format!("/p{i}")));
        }
        assert_eq!(s.recent_projects.len(), MAX_RECENT_PROJECTS);
        // Most-recent (/p9) at the front.
        assert_eq!(s.recent_projects[0], PathBuf::from("/p9"));
    }

    #[test]
    fn remove_recent_takes_one_entry() {
        let mut s = PersistedState::default();
        s.record_open(PathBuf::from("/a"));
        s.record_open(PathBuf::from("/b"));
        assert!(s.remove_recent(Path::new("/a")));
        assert_eq!(s.recent_projects, vec![PathBuf::from("/b")]);
        assert!(!s.remove_recent(Path::new("/never_there")));
    }

    #[test]
    fn theme_pref_defaults_to_system() {
        let s = PersistedState::default();
        assert_eq!(s.theme, ThemePref::System);
    }

    // ----- Envelope tests -------------------------------------------------

    #[test]
    fn envelope_empty_input_returns_empty() {
        let env = build_envelope(&[], 100);
        assert!(env.is_empty());
    }

    #[test]
    fn envelope_zero_buckets_returns_empty() {
        let env = build_envelope(&[1.0, -1.0, 0.5], 0);
        assert!(env.is_empty());
    }

    #[test]
    fn envelope_buckets_capture_min_and_max() {
        // 8 samples bucketed to 4 = 2 samples per bucket.
        let samples = [0.1, 0.9, -0.2, -0.8, 0.3, 0.7, -0.4, -0.6];
        let env = build_envelope(&samples, 4);
        assert_eq!(
            env,
            vec![(0.1, 0.9), (-0.8, -0.2), (0.3, 0.7), (-0.6, -0.4)]
        );
    }

    #[test]
    fn envelope_short_input_distributes_samples_one_per_bucket() {
        // bucket_size = ceil(3 / 8) = 1 → 3 buckets, not 8.
        let env = build_envelope(&[0.5, -0.5, 0.25], 8);
        assert_eq!(env.len(), 3);
        assert_eq!(env[0], (0.5, 0.5));
        assert_eq!(env[1], (-0.5, -0.5));
        assert_eq!(env[2], (0.25, 0.25));
    }

    #[test]
    fn envelope_preserves_peaks_at_lower_resolution() {
        // A sharp peak in a sea of zeros must survive the downsample.
        let mut samples = vec![0.0; 1000];
        samples[497] = 0.99;
        samples[503] = -0.99;
        let env = build_envelope(&samples, 100);
        let global_max = env.iter().map(|(_, mx)| *mx).fold(f32::MIN, f32::max);
        let global_min = env.iter().map(|(mn, _)| *mn).fold(f32::MAX, f32::min);
        assert!((global_max - 0.99).abs() < 1e-6);
        assert!((global_min - -0.99).abs() < 1e-6);
    }
}

// ---------------------------------------------------------------------------
// Waveform envelope
// ---------------------------------------------------------------------------

/// Bundle audio + metadata cached on selection. B2 originally also
/// cached a fixed-resolution waveform envelope here; C5 replaced that
/// with per-frame re-bucketing via [`build_envelope_for_range`], so
/// the cache now holds only the raw mono samples + the header info
/// the panes use for axis bounds.
#[derive(Debug, Clone)]
pub struct EnvelopeCache {
    /// Bundle this cache was built for; reset when the user selects
    /// a different bundle.
    pub bundle_id: i64,
    /// Bundle audio sample rate; used for the x-axis tick labels and
    /// for sample-count → time conversions in B3's STFT call.
    pub sample_rate: u32,
    /// Bundle audio duration in seconds; used for the x-axis bounds.
    pub duration_seconds: f64,
    /// Mono-mixdown samples. Used by the waveform pane's per-frame
    /// re-bucketer (C5) and by the spectrogram cache's STFT
    /// (rebuilt on (window, hop, colormap, dynamic-range) change).
    /// `Arc` so the async-analysis worker (P2) can share it with the UI
    /// without copying — important for hour-long recordings.
    pub mono_samples: std::sync::Arc<Vec<f32>>,
}

/// Computes a min/max envelope over `samples` at `target_buckets`
/// resolution. Each bucket pair is `(min, max)` over the samples
/// falling inside that bucket. Empty input or zero buckets returns
/// an empty vector. If the sample count is smaller than the bucket
/// count, the result has one bucket per sample (no padding, no
/// repetition).
pub fn build_envelope(samples: &[f32], target_buckets: usize) -> Vec<(f32, f32)> {
    if samples.is_empty() || target_buckets == 0 {
        return Vec::new();
    }
    let n = samples.len();
    let bucket_size = n.div_ceil(target_buckets).max(1);
    let n_buckets = n.div_ceil(bucket_size);
    let mut out = Vec::with_capacity(n_buckets);
    for b in 0..n_buckets {
        let start = b * bucket_size;
        let end = (start + bucket_size).min(n);
        let chunk = &samples[start..end];
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &s in chunk {
            if s < mn {
                mn = s;
            }
            if s > mx {
                mx = s;
            }
        }
        out.push((mn, mx));
    }
    out
}

// ---------------------------------------------------------------------------
// Spectrogram (B3): config + pure-data helpers
// ---------------------------------------------------------------------------

// `ColormapKind` (with its `label()` and colormap sampling) lives in the
// engine (`sadda_engine::dsp::colormap`) as of G0 and is re-exported below
// alongside the bake helpers.

/// Colour scheme for the measure-track lanes and reference overlays.
/// `Default` keeps the original warm-leaning scheme; `OkabeIto` swaps in
/// the Okabe–Ito colourblind-safe qualitative palette where colour
/// actually has to be *discriminated* — the overlaid formants F1…Fn that
/// share one lane, and the observed / normative / target reference bands
/// that coexist on a lane. Single-series lanes (f0, intensity, VAD) are
/// already unambiguous, so they're left alone. Lives in [`PersistedState`].
///
/// Okabe & Ito, "Color Universal Design" (2008);
/// <https://jfly.uni-koeln.de/color/>.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PlotPalette {
    /// Original warm-leaning scheme.
    #[default]
    Default,
    /// Okabe–Ito colourblind-safe qualitative palette.
    OkabeIto,
}

impl PlotPalette {
    /// Human-readable label for the Appearance menu.
    pub fn label(self) -> &'static str {
        match self {
            PlotPalette::Default => "Default",
            PlotPalette::OkabeIto => "Colourblind-safe (Okabe–Ito)",
        }
    }
}

/// Toolbar-controlled spectrogram configuration. Lives in
/// [`PersistedState`] so the user's last-used settings survive a
/// relaunch.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SpectrogramConfig {
    /// STFT window length in milliseconds.
    pub window_ms: f32,
    /// STFT hop length in milliseconds.
    pub hop_ms: f32,
    /// Colour scheme.
    pub colormap: ColormapKind,
    /// Dynamic-range floor in dB. Values below `-dynamic_range_db`
    /// (relative to the maximum) clamp to black/the lowest colormap
    /// entry.
    pub dynamic_range_db: f32,
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self {
            window_ms: 25.0,
            hop_ms: 5.0,
            colormap: ColormapKind::Viridis,
            dynamic_range_db: 70.0,
        }
    }
}

/// [`PersistedState`] so the user's lane visibility + analysis
/// parameters survive a relaunch. Changing any field invalidates the
/// app's track cache (see `MeasureTrackCache`), so this derives
/// `PartialEq` to make staleness a cheap equality check — exactly the
/// pattern [`SpectrogramConfig`] uses.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MeasureTrackConfig {
    /// Whether the f0 lane is shown.
    pub f0_visible: bool,
    /// Whether the formant lane is shown.
    pub formants_visible: bool,
    /// Whether the intensity lane is shown.
    pub intensity_visible: bool,
    /// Whether the VAD (speech-activity) lane is shown. Requires ONNX
    /// Runtime at runtime; the lane shows a hint if it isn't available.
    #[serde(default)]
    pub vad_visible: bool,
    /// Formant lane y-axis maximum (Hz). Formants above this aren't
    /// plotted; the lane scales to a fixed range so vowels are
    /// comparable across bundles. (Display only — not part of the analysis
    /// config, hence kept separate from `formant_params`.)
    pub formant_max_hz: f32,
    /// Intensity lane y-axis floor (dB-FS). The ceiling is fixed at 0.
    pub intensity_floor_db: f32,
    /// VAD speech-probability threshold, drawn as a line on the VAD lane.
    #[serde(default = "default_vad_threshold")]
    pub vad_threshold: f32,
    /// Full f0-tracking spec for the pitch lane (method + config: search
    /// floor/ceiling, voicing threshold, and method-specific knobs). Doubles
    /// as the f0 lane's y-axis bounds (`min_freq_hz`..`max_freq_hz`). Persists
    /// across launches; changing any field invalidates the track cache.
    #[serde(default)]
    pub pitch_params: sadda_engine::pitch::PitchParams,
    /// Full formant spec for the formant lane (LPC method + count + analysis
    /// knobs). Changing any field invalidates the track cache.
    #[serde(default)]
    pub formant_params: sadda_engine::dsp::FormantsConfig,
}

fn default_vad_threshold() -> f32 {
    0.5
}

impl MeasureTrackConfig {
    /// True when at least one lane is enabled — used to skip the
    /// analysis recompute entirely when every lane is hidden.
    pub fn any_visible(&self) -> bool {
        self.f0_visible || self.formants_visible || self.intensity_visible || self.vad_visible
    }
}

/// S3: show/hide state for the three *structural* signal subpanes — waveform,
/// spectrogram, and the tier strip — which were previously always drawn. The
/// measure lanes (f0 / formants / intensity / VAD / MFCC / embedding) carry
/// their own visibility in [`MeasureTrackConfig`] / [`MfccLaneConfig`] /
/// [`EmbeddingHeatmapConfig`]; this fills the gap so *every* subpane is
/// togglable — the foundation for named/scripted documentation capture.
///
/// Each field defaults to `true` (shown) so existing persisted state that
/// predates this struct restores with everything visible, matching the old
/// always-on behaviour.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignalPaneVisibility {
    #[serde(default = "shown")]
    pub waveform: bool,
    #[serde(default = "shown")]
    pub spectrogram: bool,
    #[serde(default = "shown")]
    pub tier_strip: bool,
}

fn shown() -> bool {
    true
}

impl Default for SignalPaneVisibility {
    fn default() -> Self {
        Self {
            waveform: true,
            spectrogram: true,
            tier_strip: true,
        }
    }
}

/// D10: index of the frame whose `time` is closest to `cursor`, or `None`
/// for an empty slice. Used to read the measured vowel (F1/F2) at the
/// playback cursor for the Reference panel's vowel-space scatter. Frame
/// times are assumed ascending but the scan is order-agnostic.
pub fn nearest_frame_index(times: &[f64], cursor: f64) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, &t) in times.iter().enumerate() {
        let d = (t - cursor).abs();
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    best.map(|(i, _)| i)
}

/// D10: a reference distribution selected to overlay on a measure-track
/// lane, plus the subgroup it's narrowed to. Identifies the distribution
/// by `id` + `version` (resolved against the store at draw time) so the
/// choice persists across launches and survives a store update. `sex` is
/// the optional subgroup filter (a distribution with both sexes pooled
/// reads as a bimodal band, so the picker usually narrows it).
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RefdistOverlay {
    /// Distribution id (e.g. `"placeholder-f0-norms"`).
    pub id: String,
    /// Pinned version.
    pub version: String,
    /// Subgroup filter on the `sex` column, if narrowed.
    #[serde(default)]
    pub sex: Option<String>,
}

impl Default for MeasureTrackConfig {
    fn default() -> Self {
        Self {
            // f0 is the single most-requested track across user groups
            // (the 2026-05-16 survey's pattern A); on by default. The
            // others stay opt-in to keep the first view uncluttered.
            f0_visible: true,
            formants_visible: false,
            intensity_visible: false,
            vad_visible: false,
            formant_max_hz: 5500.0,
            intensity_floor_db: -80.0,
            vad_threshold: 0.5,
            // Default: Boersma + Praat-default config (75–500 Hz, voicing 0.45)
            // — same numbers the old f0_min/max/voicing fields carried.
            pitch_params: sadda_engine::pitch::PitchParams::default(),
            // Burg + 5 formants — same as the old formant_count/lpc_method.
            formant_params: sadda_engine::dsp::FormantsConfig::default(),
        }
    }
}

/// Embedding-heatmap configuration — which `continuous_vector` tier the
/// lane is rendering, plus colormap + normalization. Lives in
/// [`PersistedState`] so the user's last view survives a relaunch.
///
/// Default leaves `selected_tier_id` as `None`, which hides the lane.
/// Colormap defaults to [`ColormapKind::Cividis`] for consistency with the
/// accessibility-default spectrogram colormap (CVD-safe, luminance-
/// monotonic). Normalization defaults to per-dim z-score, matching the
/// SSL-probing-paper convention so one high-magnitude dim can't wash out
/// the rest of the heatmap.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingHeatmapConfig {
    /// Tier id of the displayed `continuous_vector` tier, or `None` when
    /// the lane is hidden. Persisted across launches; tier ids are
    /// project-scoped, so a stale id from a different project clears
    /// itself silently on the next refresh (the rebuild can't find it).
    #[serde(default)]
    pub selected_tier_id: Option<i64>,
    /// Colormap applied to the normalized values. Defaults to Cividis.
    #[serde(default = "default_embedding_colormap")]
    pub colormap: ColormapKind,
    /// Normalization applied before the colormap. Defaults to per-dim
    /// z-score.
    #[serde(default)]
    pub normalization: EmbeddingNormalization,
}

fn default_embedding_colormap() -> ColormapKind {
    ColormapKind::Cividis
}

impl Default for EmbeddingHeatmapConfig {
    fn default() -> Self {
        Self {
            selected_tier_id: None,
            colormap: ColormapKind::Cividis,
            normalization: EmbeddingNormalization::default(),
        }
    }
}

/// Normalization strategies for the embedding heatmap. **Per-dim
/// z-score** is the SSL-probing-paper standard — each row centered +
/// scaled to unit variance so dim-magnitude differences (very common in
/// SSL encoders, where a handful of neurons sweep ±10 while most stay
/// near zero) don't wash out finer structure. The other two are
/// available for when raw cross-dim magnitudes matter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EmbeddingNormalization {
    /// Each row (one dim) is centered to its mean and scaled to its std.
    /// Z-scores are clipped to `[-3, +3]` before mapping to `[0, 1]`.
    #[default]
    PerDimZScore,
    /// One mean + one std computed across the whole matrix.
    GlobalZScore,
    /// Min and max computed across the whole matrix; map to `[0, 1]`.
    GlobalMinMax,
}

/// Normalizes a `[n_dims × n_frames]` row-major matrix into a `[0, 1]`
/// buffer ready for [`colormap_bake`]. Output has the same shape and
/// memory layout as the input (`[dim * n_frames + frame]`).
///
/// Empty inputs return an empty vector. A degenerate row (constant
/// values, so `std == 0`) maps to the midpoint `0.5` under z-score
/// normalization rather than dividing by zero. Z-score outputs are
/// clipped to `[-3, +3]` before scaling so a single outlier doesn't
/// compress the rest of the dim into a sliver of the colour range.
pub fn normalize_embedding(
    matrix: &[f32],
    n_dims: usize,
    n_frames: usize,
    mode: EmbeddingNormalization,
) -> Vec<f32> {
    debug_assert_eq!(
        matrix.len(),
        n_dims * n_frames,
        "normalize_embedding: shape mismatch",
    );
    if matrix.is_empty() {
        return Vec::new();
    }
    match mode {
        EmbeddingNormalization::PerDimZScore => {
            let mut out = vec![0.0f32; matrix.len()];
            for d in 0..n_dims {
                let row = &matrix[d * n_frames..(d + 1) * n_frames];
                let mean = row.iter().copied().sum::<f32>() / row.len() as f32;
                let var = row
                    .iter()
                    .map(|&x| {
                        let dx = x - mean;
                        dx * dx
                    })
                    .sum::<f32>()
                    / row.len() as f32;
                let std = var.sqrt();
                let out_row = &mut out[d * n_frames..(d + 1) * n_frames];
                if std == 0.0 {
                    // Constant row → midpoint, not NaN.
                    for cell in out_row.iter_mut() {
                        *cell = 0.5;
                    }
                } else {
                    for (i, &x) in row.iter().enumerate() {
                        let z = ((x - mean) / std).clamp(-3.0, 3.0);
                        // z ∈ [-3, +3] → [0, 1]
                        out_row[i] = (z + 3.0) / 6.0;
                    }
                }
            }
            out
        }
        EmbeddingNormalization::GlobalZScore => {
            let n = matrix.len() as f32;
            let mean = matrix.iter().copied().sum::<f32>() / n;
            let var = matrix
                .iter()
                .map(|&x| {
                    let dx = x - mean;
                    dx * dx
                })
                .sum::<f32>()
                / n;
            let std = var.sqrt();
            if std == 0.0 {
                return vec![0.5; matrix.len()];
            }
            matrix
                .iter()
                .map(|&x| {
                    let z = ((x - mean) / std).clamp(-3.0, 3.0);
                    (z + 3.0) / 6.0
                })
                .collect()
        }
        EmbeddingNormalization::GlobalMinMax => {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &x in matrix {
                if x < lo {
                    lo = x;
                }
                if x > hi {
                    hi = x;
                }
            }
            let span = hi - lo;
            if span == 0.0 {
                return vec![0.5; matrix.len()];
            }
            matrix.iter().map(|&x| (x - lo) / span).collect()
        }
    }
}

/// MFCC heatmap-lane state: which preset is active, the full parameter set it
/// resolves to (possibly edited away from the preset), and the display knobs
/// (colormap / normalization / whether to drop c0). Default leaves the lane
/// hidden with the librosa preset loaded.
///
/// Unlike the sibling `*MethodChoice` mirrors, this stores the engine
/// [`sadda_engine::dsp::MfccParams`] directly: that type is now
/// `Serialize`/`Deserialize`/`Clone`/`PartialEq`, so a hand-written mirror of
/// its ~21 fields (plus the data enums) would be pure, drift-prone
/// duplication. `PartialEq` is load-bearing — the lane cache invalidates by
/// `==` on this whole struct (see `rebuild_mfcc_if_stale`).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MfccLaneConfig {
    /// Whether the MFCC lane is shown.
    #[serde(default)]
    pub show: bool,
    /// Id of the active preset (the menu's current selection). When `params`
    /// has been edited away from this preset, the lane caption flags it.
    #[serde(default = "default_mfcc_preset_id")]
    pub preset_id: String,
    /// The resolved parameter set actually used to compute the lane.
    #[serde(default = "default_mfcc_params")]
    pub params: sadda_engine::dsp::MfccParams,
    /// Colormap applied to the normalized coefficients.
    #[serde(default = "default_mfcc_colormap")]
    pub colormap: ColormapKind,
    /// Normalization applied before the colormap. Per-coefficient z-score by
    /// default, which keeps c0 (overall log-energy, orders larger than c1+)
    /// from washing out the rest of the heatmap.
    #[serde(default)]
    pub normalization: EmbeddingNormalization,
    /// How c0 (the overall log-energy coefficient) is shown. Defaults to
    /// `Separate` — visible but on its own scale, set apart by a small gap.
    #[serde(default)]
    pub c0: MfccC0Display,
}

fn default_mfcc_preset_id() -> String {
    "librosa-default".to_string()
}

fn default_mfcc_params() -> sadda_engine::dsp::MfccParams {
    sadda_engine::dsp::MfccParams::librosa(0.025, 0.010, 40, 13, 0.0, 8000.0)
}

fn default_mfcc_colormap() -> ColormapKind {
    ColormapKind::Cividis
}

impl Default for MfccLaneConfig {
    fn default() -> Self {
        Self {
            show: false,
            preset_id: default_mfcc_preset_id(),
            params: default_mfcc_params(),
            colormap: default_mfcc_colormap(),
            normalization: EmbeddingNormalization::default(),
            c0: MfccC0Display::default(),
        }
    }
}

/// How the MFCC lane displays c0 (overall log-energy). c0 is orders larger
/// than the spectral-shape coefficients c1+, so mixing it in is misleading
/// (and, under a shared-scale normalization, washes the rest out).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MfccC0Display {
    /// Show c0 on its own normalization scale, separated from c1+ by a small
    /// gap. The default — keeps the energy visible without it dominating or
    /// being mistaken for a shape coefficient.
    #[default]
    Separate,
    /// Show c0 in the matrix on the same scale as the other coefficients.
    Inline,
    /// Omit c0 entirely; show only c1+.
    Hidden,
}

impl MfccC0Display {
    pub fn label(self) -> &'static str {
        match self {
            MfccC0Display::Separate => "Separate scale + gap",
            MfccC0Display::Inline => "Inline (shared scale)",
            MfccC0Display::Hidden => "Hidden",
        }
    }

    pub fn all() -> [MfccC0Display; 3] {
        [
            MfccC0Display::Separate,
            MfccC0Display::Inline,
            MfccC0Display::Hidden,
        ]
    }
}

// G0 (figure-export groundwork): the spectrogram bake pipeline
// (`power_to_db_normalized` + `colormap_bake`) and the `ColormapKind` enum
// moved into the engine (`sadda_engine::dsp`) so headless figure export can
// bake a spectrogram raster without the GUI. Re-exported here so the existing
// `crate::state::{…}` references across the app keep resolving unchanged; the
// unit tests for these moved with them.
pub use sadda_engine::dsp::{ColormapKind, colormap_bake, power_to_db_normalized};

#[cfg(test)]
mod appearance_default_tests {
    use super::*;

    #[test]
    fn appearance_defaults_are_native_scale_and_default_palette() {
        // A persisted state written before these fields existed must
        // deserialise to native size + the default scheme, never f32's
        // 0.0 (which would shrink the whole UI to nothing).
        assert_eq!(default_ui_scale(), 1.0);
        assert_eq!(PlotPalette::default(), PlotPalette::Default);
        assert_eq!(ColormapKind::default(), ColormapKind::Viridis);
    }
}

// ---------------------------------------------------------------------------
// Tier strip (B4): label helpers
// ---------------------------------------------------------------------------

/// Truncates `text` to at most `max_chars`, replacing the trailing
/// characters with `…` when truncation happens. ASCII-aware: counts
/// chars (not bytes) so non-ASCII labels truncate cleanly. `max_chars`
/// of `0` returns an empty string.
pub fn truncate_label(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_string();
    }
    // Reserve the last char slot for the ellipsis; if max_chars == 1
    // the result is just "…".
    let keep = max_chars.saturating_sub(1);
    let mut out: String = text.chars().take(keep).collect();
    out.push('…');
    out
}

/// Caption text for a reference-tier lane: `"(reference — N
/// targets)"`. Singularises when `n == 1`.
pub fn format_reference_lane_caption(n_targets: usize) -> String {
    if n_targets == 1 {
        "(reference — 1 target)".to_string()
    } else {
        format!("(reference — {n_targets} targets)")
    }
}

#[cfg(test)]
mod tier_strip_tests {
    use super::*;

    #[test]
    fn truncate_label_short_string_unchanged() {
        assert_eq!(truncate_label("hi", 10), "hi");
    }

    #[test]
    fn truncate_label_exact_length_unchanged() {
        assert_eq!(truncate_label("abcdef", 6), "abcdef");
    }

    #[test]
    fn truncate_label_long_string_gets_ellipsis() {
        assert_eq!(truncate_label("abcdefghij", 5), "abcd…");
    }

    #[test]
    fn truncate_label_one_char_keeps_only_ellipsis() {
        assert_eq!(truncate_label("abc", 1), "…");
    }

    #[test]
    fn truncate_label_zero_max_returns_empty() {
        assert_eq!(truncate_label("abc", 0), "");
    }

    #[test]
    fn truncate_label_unicode_aware() {
        // 8 chars; truncate to 5 → "héllo" (5 chars) wait, 4 + ellipsis
        assert_eq!(truncate_label("héllo world", 6), "héllo…");
    }

    #[test]
    fn reference_caption_singular_vs_plural() {
        assert_eq!(format_reference_lane_caption(0), "(reference — 0 targets)");
        assert_eq!(format_reference_lane_caption(1), "(reference — 1 target)");
        assert_eq!(format_reference_lane_caption(3), "(reference — 3 targets)");
    }
}

// ---------------------------------------------------------------------------
// Timeline state (C5): cursor + view window + zoom + scroll
// ---------------------------------------------------------------------------

/// The shared timeline navigation state (cursor, view window, selection) used
/// by every C5+ pane. The type and its move-to / move-by API live in the engine
/// (`sadda_engine::Timeline`) so the desktop keybindings, the Python bindings,
/// and unit tests share one tested implementation; the app keeps the historic
/// `TimelineState` name via this re-export.
pub use sadda_engine::Timeline as TimelineState;

/// Per-frame waveform downsampler over a time range. Replaces the
/// fixed-resolution `build_envelope` for C5's zoomable view. Returns
/// one `(min, max)` bucket per pixel column.
///
/// `mono` is the full bundle mono mixdown (from
/// [`EnvelopeCache::mono_samples`]); `view_start` / `view_end` clamp
/// to `[0, mono.len() / sample_rate]` automatically.
pub fn build_envelope_for_range(
    mono: &[f32],
    sample_rate: u32,
    view_start: f64,
    view_end: f64,
    target_buckets: usize,
) -> Vec<(f32, f32)> {
    if mono.is_empty() || target_buckets == 0 || sample_rate == 0 {
        return Vec::new();
    }
    let sr = sample_rate as f64;
    let n = mono.len();
    let start_sample = ((view_start * sr).max(0.0) as usize).min(n);
    let end_sample = ((view_end * sr).max(0.0) as usize).min(n);
    if end_sample <= start_sample {
        return Vec::new();
    }
    let slice = &mono[start_sample..end_sample];
    build_envelope(slice, target_buckets)
}

#[cfg(test)]
mod timeline_tests {
    use super::*;

    // ----- build_envelope_for_range -----

    #[test]
    fn envelope_for_range_empty_input() {
        let env = build_envelope_for_range(&[], 16_000, 0.0, 1.0, 100);
        assert!(env.is_empty());
    }

    #[test]
    fn envelope_for_range_zero_sample_rate() {
        let env = build_envelope_for_range(&[0.1, 0.2], 0, 0.0, 1.0, 100);
        assert!(env.is_empty());
    }

    #[test]
    fn envelope_for_range_inverted_window() {
        let env = build_envelope_for_range(&[0.1, 0.2], 16_000, 1.0, 0.0, 100);
        assert!(env.is_empty());
    }

    #[test]
    fn envelope_for_range_subset() {
        // 16-sample buffer at 16 kHz → 1 ms long. Take the middle
        // 0.25–0.75 ms slice (samples 4..12) and bucket to 4.
        let samples: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
        let env = build_envelope_for_range(&samples, 16_000, 0.25e-3, 0.75e-3, 4);
        // 8 samples bucketed into 4 = 2 per bucket; values
        // (4/16, 5/16), (6/16, 7/16), (8/16, 9/16), (10/16, 11/16).
        assert_eq!(env.len(), 4);
        assert!((env[0].0 - 4.0 / 16.0).abs() < 1e-6);
        assert!((env[0].1 - 5.0 / 16.0).abs() < 1e-6);
        assert!((env[3].1 - 11.0 / 16.0).abs() < 1e-6);
    }

    #[test]
    fn embedding_per_dim_zscore_centers_each_row() {
        // 2 dims × 4 frames. Both rows are arithmetic with the same
        // *shape* (zero mean, identical relative spacing), just scaled
        // 10× apart. Per-dim z-score should produce identical output
        // rows — magnitude scaling washes out, structure stays.
        let matrix = [-3.0_f32, -1.0, 1.0, 3.0, -30.0, -10.0, 10.0, 30.0];
        let out = normalize_embedding(&matrix, 2, 4, EmbeddingNormalization::PerDimZScore);
        assert_eq!(out.len(), 8);
        for i in 0..4 {
            assert!(
                (out[i] - out[4 + i]).abs() < 1e-6,
                "z-score should erase dim-magnitude differences: dim0[{i}]={} vs dim1[{i}]={}",
                out[i],
                out[4 + i],
            );
        }
        // Symmetric input centres at 0.5.
        let mid = (out[0] + out[3]) / 2.0;
        assert!((mid - 0.5).abs() < 1e-3, "midpoint {mid}");
    }

    #[test]
    fn embedding_per_dim_zscore_constant_row_returns_midpoint() {
        // A constant row has std = 0; without a guard this divides by
        // zero and produces NaN. Should map to the midpoint instead.
        let matrix = [5.0_f32; 4];
        let out = normalize_embedding(&matrix, 1, 4, EmbeddingNormalization::PerDimZScore);
        for v in out {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "constant row not mapped to 0.5: {v}"
            );
        }
    }

    #[test]
    fn embedding_per_dim_zscore_clips_outliers() {
        // 1 dim × 10 frames: nine zeros + one 1000. With this shape the
        // outlier z-score is exactly +3 (mean=100, std=300, z=900/300=3),
        // so the [-3, +3] clip is just kissing it — the outlier pegs at
        // 1.0 and the rest cluster well below the midpoint.
        let mut matrix = vec![0.0_f32; 10];
        matrix[9] = 1000.0;
        let out = normalize_embedding(&matrix, 1, 10, EmbeddingNormalization::PerDimZScore);
        assert!(
            (out[9] - 1.0).abs() < 1e-3,
            "outlier should peg at 1.0, got {}",
            out[9],
        );
        // A far larger outlier would still clip at 1.0 — confirms the
        // clip is doing its job rather than passing the raw z-score.
        let mut matrix_huge = vec![0.0_f32; 10];
        matrix_huge[9] = 1_000_000.0;
        let out_huge =
            normalize_embedding(&matrix_huge, 1, 10, EmbeddingNormalization::PerDimZScore);
        assert!(
            (out_huge[9] - 1.0).abs() < 1e-3,
            "huge outlier should still clip at 1.0, got {}",
            out_huge[9],
        );
        // Non-outliers all share the same raw value (0.0); should be
        // well below the midpoint.
        for &v in &out[..9] {
            assert!(v < 0.5, "non-outlier should be below midpoint, got {v}");
        }
    }

    #[test]
    fn embedding_global_min_max_spans_zero_to_one() {
        let matrix = [-1.0_f32, 0.0, 1.0, 2.0];
        let out = normalize_embedding(&matrix, 1, 4, EmbeddingNormalization::GlobalMinMax);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn embedding_empty_matrix_returns_empty() {
        for mode in [
            EmbeddingNormalization::PerDimZScore,
            EmbeddingNormalization::GlobalZScore,
            EmbeddingNormalization::GlobalMinMax,
        ] {
            assert!(normalize_embedding(&[], 0, 0, mode).is_empty());
        }
    }
}
