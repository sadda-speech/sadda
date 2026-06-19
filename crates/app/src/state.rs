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
        }
    }
}

impl PersistedState {
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

/// Which colormap the spectrogram pane uses to map normalised power
/// values into RGB. Default is `Viridis` (perceptually uniform).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ColormapKind {
    /// Modern perceptually-uniform default; dark purple → blue → green → yellow.
    #[default]
    Viridis,
    /// Dark-mode-friendly perceptually-uniform alternate; black → purple → red → yellow.
    Magma,
    /// Perceptually-uniform map optimised for colour-vision deficiency
    /// (dark blue → grey → yellow). The accessibility pick — it stays
    /// monotonic in luminance for all common forms of CVD.
    Cividis,
    /// Classic black-and-white spectrogram; Praat refugees.
    Greyscale,
}

impl ColormapKind {
    /// Human-readable label for the toolbar `ComboBox`.
    pub fn label(self) -> &'static str {
        match self {
            ColormapKind::Viridis => "Viridis",
            ColormapKind::Magma => "Magma",
            ColormapKind::Cividis => "Cividis (CVD-safe)",
            ColormapKind::Greyscale => "Greyscale",
        }
    }
}

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

/// D10: configuration for the stacked measure-track lanes (f0,
/// formants, intensity) drawn below the spectrogram. Lives in
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
    /// f0 search floor (Hz). Doubles as the f0 lane's y-axis minimum.
    pub f0_min_hz: f32,
    /// f0 search ceiling (Hz). Doubles as the f0 lane's y-axis maximum.
    pub f0_max_hz: f32,
    /// Drop f0 estimates whose voicing strength is below this; unvoiced
    /// frames leave a gap rather than a spurious pitch point.
    pub f0_voicing_threshold: f32,
    /// Number of formants to track (and plot, ascending: F1..Fn).
    pub formant_count: usize,
    /// Formant lane y-axis maximum (Hz). Formants above this aren't
    /// plotted; the lane scales to a fixed range so vowels are
    /// comparable across bundles.
    pub formant_max_hz: f32,
    /// Intensity lane y-axis floor (dB-FS). The ceiling is fixed at 0.
    pub intensity_floor_db: f32,
    /// VAD speech-probability threshold, drawn as a line on the VAD lane.
    #[serde(default = "default_vad_threshold")]
    pub vad_threshold: f32,
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
            f0_min_hz: 75.0,
            f0_max_hz: 500.0,
            f0_voicing_threshold: 0.45,
            formant_count: 5,
            formant_max_hz: 5500.0,
            intensity_floor_db: -80.0,
            vad_threshold: 0.5,
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

/// Floor for converting linear power to dB-FS without blowing up on
/// silent frames. Matches the floor [`crate::state::power_to_db_normalized`]
/// applies when `power == 0`.
const POWER_DB_FLOOR: f32 = -200.0;

/// Converts linear power values into `[0, 1]` normalised dB-FS,
/// suitable for direct colormap indexing.
///
/// Pipeline per cell:
/// 1. `db = 10 · log10(power)` (or `POWER_DB_FLOOR` for silent cells).
/// 2. Find the global max across the buffer.
/// 3. Re-reference: `db_rel = db - max_db`.
/// 4. Clamp to `[-dynamic_range_db, 0]`.
/// 5. Normalise to `[0, 1]`: `(db_rel + dynamic_range_db) / dynamic_range_db`.
///
/// Returns an empty vector for empty input. `dynamic_range_db` must
/// be `> 0`; values `<=0` are treated as `1.0` to avoid div-by-zero.
pub fn power_to_db_normalized(power: &[f32], dynamic_range_db: f32) -> Vec<f32> {
    if power.is_empty() {
        return Vec::new();
    }
    let dr = if dynamic_range_db > 0.0 {
        dynamic_range_db
    } else {
        1.0
    };
    // 1. power → dB (with floor for zeros).
    let mut db: Vec<f32> = power
        .iter()
        .map(|&p| {
            if p > 0.0 {
                10.0 * p.log10()
            } else {
                POWER_DB_FLOOR
            }
        })
        .collect();
    // 2. global max.
    let max_db = db.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // 3–5. re-reference, clamp, normalise. In-place to save the alloc.
    for d in db.iter_mut() {
        let rel = (*d - max_db).clamp(-dr, 0.0);
        *d = (rel + dr) / dr;
    }
    db
}

/// Bakes a normalised `[0, 1]` freq-major buffer into a row-major
/// `RGBA8` image suitable for `egui::ColorImage::from_rgba_unmultiplied`.
///
/// `power` is laid out as `height` rows of `width` columns (a
/// freq-major spectrogram, `[freq_bin * width + frame]`). The output
/// flips the y-axis so frequency increases bottom→top in image
/// coordinates, which is the convention every spectrogram viewer
/// uses.
pub fn colormap_bake(
    power: &[f32],
    width: usize,
    height: usize,
    colormap: ColormapKind,
) -> Vec<u8> {
    debug_assert_eq!(power.len(), width * height, "colormap_bake: shape mismatch");
    let mut out = vec![0u8; width * height * 4];
    for y in 0..height {
        // Flip: image row 0 is at the top, which should show the
        // highest frequency bin (`height - 1`).
        let bin = height - 1 - y;
        for x in 0..width {
            let v = power[bin * width + x].clamp(0.0, 1.0);
            let (r, g, b) = sample_colormap(colormap, v);
            let i = (y * width + x) * 4;
            out[i] = r;
            out[i + 1] = g;
            out[i + 2] = b;
            out[i + 3] = 255;
        }
    }
    out
}

fn sample_colormap(kind: ColormapKind, t: f32) -> (u8, u8, u8) {
    let t = t.clamp(0.0, 1.0) as f64;
    match kind {
        ColormapKind::Viridis => {
            let c = colorous::VIRIDIS.eval_continuous(t);
            (c.r, c.g, c.b)
        }
        ColormapKind::Magma => {
            let c = colorous::MAGMA.eval_continuous(t);
            (c.r, c.g, c.b)
        }
        ColormapKind::Cividis => {
            let c = colorous::CIVIDIS.eval_continuous(t);
            (c.r, c.g, c.b)
        }
        ColormapKind::Greyscale => {
            let v = (t * 255.0).round() as u8;
            (v, v, v)
        }
    }
}

#[cfg(test)]
mod spectrogram_tests {
    use super::*;

    #[test]
    fn power_to_db_normalized_empty_returns_empty() {
        assert!(power_to_db_normalized(&[], 70.0).is_empty());
    }

    #[test]
    fn power_to_db_normalized_constant_input_returns_ones() {
        // All cells equal → all re-referenced to 0 dB → normalised to 1.
        let out = power_to_db_normalized(&[1.0, 1.0, 1.0, 1.0], 70.0);
        for v in out {
            assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn power_to_db_normalized_clamps_below_dynamic_range() {
        // Max is 1.0 (0 dB); silent cell is at POWER_DB_FLOOR (very
        // negative), should clamp to the bottom of the range (= 0.0).
        let out = power_to_db_normalized(&[1.0, 0.0, 0.0, 0.0], 70.0);
        assert!((out[0] - 1.0).abs() < 1e-6);
        for &v in &out[1..] {
            assert!(v.abs() < 1e-6, "silent cell should clamp to 0.0, got {v}");
        }
    }

    #[test]
    fn power_to_db_normalized_midpoint_is_half_at_floor_minus_half() {
        // Cell at -35 dB relative to max with 70 dB range → 0.5 after
        // normalisation.
        // power that gives -35 dB relative: 10^(-3.5) ≈ 0.000316.
        let out = power_to_db_normalized(&[1.0, 10f32.powf(-3.5)], 70.0);
        assert!((out[1] - 0.5).abs() < 1e-3, "got {}", out[1]);
    }

    #[test]
    fn colormap_bake_shape_is_rgba_height_times_width() {
        let power = vec![0.5_f32; 6]; // 2 freq bins × 3 frames
        let rgba = colormap_bake(&power, 3, 2, ColormapKind::Greyscale);
        assert_eq!(rgba.len(), 3 * 2 * 4);
        // Greyscale @ 0.5 ≈ (128, 128, 128, 255).
        for chunk in rgba.chunks_exact(4) {
            assert!((chunk[0] as i32 - 128).abs() < 2);
            assert_eq!(chunk[0], chunk[1]);
            assert_eq!(chunk[0], chunk[2]);
            assert_eq!(chunk[3], 255);
        }
    }

    #[test]
    fn colormap_bake_flips_y_axis_so_highest_freq_is_at_top() {
        // 2 freq bins × 1 frame. bin 0 (low) = 0.0, bin 1 (high) = 1.0.
        let power = vec![0.0_f32, 1.0_f32];
        let rgba = colormap_bake(&power, 1, 2, ColormapKind::Greyscale);
        // Image row 0 (top) should reflect the high freq (1.0 → 255).
        assert_eq!(rgba[0], 255, "image top row should be the high-freq cell");
        // Image row 1 (bottom) should reflect the low freq (0.0 → 0).
        assert_eq!(rgba[4], 0, "image bottom row should be the low-freq cell");
    }

    #[test]
    fn cividis_colormap_is_distinct() {
        // Endpoints differ, and Cividis differs from Viridis at the
        // midpoint — guards the new match arm against accidentally
        // aliasing another scheme.
        assert_ne!(
            sample_colormap(ColormapKind::Cividis, 0.0),
            sample_colormap(ColormapKind::Cividis, 1.0),
        );
        assert_ne!(
            sample_colormap(ColormapKind::Cividis, 0.5),
            sample_colormap(ColormapKind::Viridis, 0.5),
        );
    }

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

/// Minimum view-range size in seconds; prevents the user from zooming
/// to a window so small that floats start losing precision.
const MIN_VIEW_RANGE_SECONDS: f64 = 0.005;
/// Drags shorter than this (in seconds) don't "stick" as a selection —
/// they read as a plain click (which clears the selection + sets cursor).
const MIN_SELECTION_SECONDS: f64 = 0.002;

/// Shared timeline state used by every C5+ pane: cursor, view window,
/// and the bundle duration the window clamps against. Pure-data —
/// pane render code calls back into this for zoom / scroll / cursor
/// mutations. Reset on bundle change via [`TimelineState::reset_for_bundle`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelineState {
    /// Left edge of the visible time range, in seconds.
    pub view_start: f64,
    /// Right edge of the visible time range, in seconds (exclusive).
    pub view_end: f64,
    /// Cursor position in seconds. Clamped to `[0, duration]`.
    pub cursor: f64,
    /// Bundle duration in seconds. Acts as the upper bound for
    /// `view_end` and `cursor` clamping.
    pub duration: f64,
    /// Active time-span selection `(lo, hi)` in seconds, drawn as a band
    /// across every lane. `None` when nothing is selected. Drag on the
    /// waveform / spectrogram sets it; a plain click clears it.
    pub selection: Option<(f64, f64)>,
    /// Transient drag anchor while a selection is being dragged out.
    selection_anchor: Option<f64>,
}

impl Default for TimelineState {
    fn default() -> Self {
        Self {
            view_start: 0.0,
            view_end: 0.0,
            cursor: 0.0,
            duration: 0.0,
            selection: None,
            selection_anchor: None,
        }
    }
}

impl TimelineState {
    /// Reinitialises the timeline for a freshly-loaded bundle:
    /// view spans `[0, duration]`, cursor at 0.
    pub fn reset_for_bundle(&mut self, duration_seconds: f64) {
        let d = duration_seconds.max(0.0);
        self.duration = d;
        self.view_start = 0.0;
        self.view_end = d.max(MIN_VIEW_RANGE_SECONDS);
        self.cursor = 0.0;
        self.selection = None;
        self.selection_anchor = None;
    }

    /// Begins a drag-selection anchored at `t` (seconds).
    pub fn begin_selection(&mut self, t: f64) {
        let t = t.clamp(0.0, self.duration);
        self.selection_anchor = Some(t);
        self.selection = Some((t, t));
    }

    /// Sets a zero-width "selection point" at `t` (clamped to the bundle). A
    /// plain click uses this so the time can be committed as a point; a drag
    /// still produces a span via [`begin_selection`](Self::begin_selection).
    pub fn set_point_selection(&mut self, t: f64) {
        let t = t.clamp(0.0, self.duration);
        self.selection_anchor = None;
        self.selection = Some((t, t));
    }

    /// Extends the in-progress selection to `t` (sorted into `(lo, hi)`).
    pub fn update_selection(&mut self, t: f64) {
        if let Some(a) = self.selection_anchor {
            let t = t.clamp(0.0, self.duration);
            self.selection = Some(if a <= t { (a, t) } else { (t, a) });
        }
    }

    /// Finalises a drag-selection; discards spans too short to be intentional.
    pub fn end_selection(&mut self) {
        self.selection_anchor = None;
        if let Some((lo, hi)) = self.selection {
            if (hi - lo) < MIN_SELECTION_SECONDS {
                self.selection = None;
            }
        }
    }

    /// Clears any selection (and drag anchor).
    pub fn clear_selection(&mut self) {
        self.selection = None;
        self.selection_anchor = None;
    }

    /// Returns `view_end - view_start` (always > 0 for an
    /// initialised state).
    pub fn view_range(&self) -> f64 {
        (self.view_end - self.view_start).max(MIN_VIEW_RANGE_SECONDS)
    }

    /// Maps a pixel-x within `[0, plot_width_px)` to seconds in the
    /// current view range.
    pub fn pixel_to_time(&self, pixel_x: f64, plot_width_px: f64) -> f64 {
        if plot_width_px <= 0.0 {
            return self.view_start;
        }
        let t = (pixel_x / plot_width_px).clamp(0.0, 1.0);
        self.view_start + t * self.view_range()
    }

    /// Sets the cursor, clamped to `[0, duration]`.
    pub fn set_cursor(&mut self, time_seconds: f64) {
        self.cursor = time_seconds.clamp(0.0, self.duration);
    }

    /// Zooms the view around `time_seconds`, by `factor` (e.g. 0.833
    /// to zoom in, 1.2 to zoom out). Clamps the resulting view to
    /// `[0, duration]` and prevents the range from shrinking below
    /// [`MIN_VIEW_RANGE_SECONDS`].
    pub fn zoom_at(&mut self, time_seconds: f64, factor: f64) {
        let old_range = self.view_range();
        let new_range = (old_range * factor).clamp(
            MIN_VIEW_RANGE_SECONDS,
            self.duration.max(MIN_VIEW_RANGE_SECONDS),
        );
        // Position of `time_seconds` within the old window (0..1):
        let t_norm = ((time_seconds - self.view_start) / old_range).clamp(0.0, 1.0);
        // Re-anchor: keep `time_seconds` at the same normalised
        // position inside the new (smaller / larger) window.
        let new_start = time_seconds - t_norm * new_range;
        self.set_view(new_start, new_start + new_range);
    }

    /// Pans the view by `delta_seconds`, clamped against the bundle's
    /// bounds. Range size is preserved.
    pub fn scroll_by(&mut self, delta_seconds: f64) {
        let range = self.view_range();
        let new_start = self.view_start + delta_seconds;
        self.set_view(new_start, new_start + range);
    }

    /// Ensures the cursor is inside `[view_start, view_end]`. If not,
    /// shifts the view (preserving range) to put the cursor a quarter
    /// of the way in from the left edge — the convention that gives
    /// the user upcoming audio to look at during playback.
    pub fn ensure_cursor_visible(&mut self) {
        let range = self.view_range();
        if self.cursor < self.view_start || self.cursor > self.view_end {
            let new_start = (self.cursor - range * 0.25).max(0.0);
            self.set_view(new_start, new_start + range);
        }
    }

    /// Internal: clamps a candidate `[start, end]` to the bundle.
    fn set_view(&mut self, mut start: f64, mut end: f64) {
        let duration = self.duration.max(MIN_VIEW_RANGE_SECONDS);
        let range = (end - start).max(MIN_VIEW_RANGE_SECONDS);
        if start < 0.0 {
            start = 0.0;
            end = (start + range).min(duration);
        }
        if end > duration {
            end = duration;
            start = (end - range).max(0.0);
        }
        self.view_start = start;
        self.view_end = end.max(start + MIN_VIEW_RANGE_SECONDS);
    }
}

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

    fn fresh(duration: f64) -> TimelineState {
        let mut t = TimelineState::default();
        t.reset_for_bundle(duration);
        t
    }

    #[test]
    fn reset_for_bundle_spans_full_range() {
        let t = fresh(5.0);
        assert_eq!(t.view_start, 0.0);
        assert_eq!(t.view_end, 5.0);
        assert_eq!(t.cursor, 0.0);
        assert_eq!(t.duration, 5.0);
    }

    #[test]
    fn reset_clamps_negative_duration() {
        let mut t = TimelineState::default();
        t.reset_for_bundle(-1.0);
        assert_eq!(t.duration, 0.0);
        assert!(t.view_end >= MIN_VIEW_RANGE_SECONDS);
    }

    #[test]
    fn selection_drag_sorts_and_sticks() {
        let mut t = fresh(5.0);
        // Drag right-to-left still yields a sorted (lo, hi).
        t.begin_selection(3.0);
        t.update_selection(1.0);
        assert_eq!(t.selection, Some((1.0, 3.0)));
        t.end_selection();
        assert_eq!(t.selection, Some((1.0, 3.0)));
    }

    #[test]
    fn tiny_selection_is_discarded_as_a_click() {
        let mut t = fresh(5.0);
        t.begin_selection(2.0);
        t.update_selection(2.0 + MIN_SELECTION_SECONDS / 2.0);
        t.end_selection();
        assert_eq!(t.selection, None);
    }

    #[test]
    fn clear_and_reset_drop_selection() {
        let mut t = fresh(5.0);
        t.begin_selection(1.0);
        t.update_selection(2.0);
        t.clear_selection();
        assert_eq!(t.selection, None);
        t.begin_selection(1.0);
        t.update_selection(2.0);
        t.reset_for_bundle(5.0);
        assert_eq!(t.selection, None);
    }

    #[test]
    fn selection_clamps_to_duration() {
        let mut t = fresh(2.0);
        t.begin_selection(-1.0);
        t.update_selection(9.0);
        assert_eq!(t.selection, Some((0.0, 2.0)));
    }

    #[test]
    fn set_cursor_clamps_to_duration() {
        let mut t = fresh(2.0);
        t.set_cursor(5.0);
        assert_eq!(t.cursor, 2.0);
        t.set_cursor(-1.0);
        assert_eq!(t.cursor, 0.0);
        t.set_cursor(1.3);
        assert_eq!(t.cursor, 1.3);
    }

    #[test]
    fn zoom_in_keeps_anchor_at_same_normalised_position() {
        let mut t = fresh(10.0);
        // Anchor at the cursor (middle of the view), zoom in 5x.
        t.zoom_at(5.0, 0.2);
        // The anchored time should still be at the same normalised
        // position (0.5) inside the new view, so:
        let normalised = (5.0 - t.view_start) / t.view_range();
        assert!((normalised - 0.5).abs() < 1e-9);
        assert!((t.view_range() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn zoom_clamps_to_minimum_range() {
        let mut t = fresh(10.0);
        // Try to zoom to a microscopic range.
        t.zoom_at(5.0, 0.000001);
        assert!(t.view_range() >= MIN_VIEW_RANGE_SECONDS);
    }

    #[test]
    fn zoom_out_past_bundle_clamps_to_full_range() {
        let mut t = fresh(10.0);
        t.zoom_at(5.0, 100.0);
        assert!((t.view_start - 0.0).abs() < 1e-9);
        assert!((t.view_end - 10.0).abs() < 1e-9);
    }

    #[test]
    fn scroll_preserves_range_and_clamps_left() {
        let mut t = fresh(10.0);
        t.zoom_at(5.0, 0.2); // range 2.0
        t.scroll_by(-100.0);
        assert!((t.view_start - 0.0).abs() < 1e-9);
        assert!((t.view_range() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn scroll_preserves_range_and_clamps_right() {
        let mut t = fresh(10.0);
        t.zoom_at(5.0, 0.2); // range 2.0
        t.scroll_by(100.0);
        assert!((t.view_end - 10.0).abs() < 1e-9);
        assert!((t.view_range() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn ensure_cursor_visible_shifts_view() {
        let mut t = fresh(10.0);
        t.zoom_at(2.0, 0.2); // range 2.0, view about [1, 3]
        t.cursor = 7.0;
        t.ensure_cursor_visible();
        assert!(t.cursor >= t.view_start && t.cursor <= t.view_end);
        // Range preserved.
        assert!((t.view_range() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn pixel_to_time_maps_within_view() {
        let mut t = fresh(10.0);
        t.view_start = 2.0;
        t.view_end = 6.0;
        // Left edge → view_start; right edge → view_end.
        assert!((t.pixel_to_time(0.0, 100.0) - 2.0).abs() < 1e-9);
        assert!((t.pixel_to_time(100.0, 100.0) - 6.0).abs() < 1e-9);
        assert!((t.pixel_to_time(50.0, 100.0) - 4.0).abs() < 1e-9);
    }

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
