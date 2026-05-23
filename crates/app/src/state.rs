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
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
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
    /// E8: whether the embedded-CPython script panel is currently
    /// shown at the bottom of the app.
    #[serde(default)]
    pub script_panel_open: bool,
    /// E8: persisted script-editor buffer. Survives relaunches so
    /// users don't lose typed scripts.
    #[serde(default)]
    pub script_buffer: String,
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
    pub mono_samples: Vec<f32>,
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
    /// Classic black-and-white spectrogram; Praat refugees.
    Greyscale,
}

impl ColormapKind {
    /// Human-readable label for the toolbar `ComboBox`.
    pub fn label(self) -> &'static str {
        match self {
            ColormapKind::Viridis => "Viridis",
            ColormapKind::Magma => "Magma",
            ColormapKind::Greyscale => "Greyscale",
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
}

impl Default for TimelineState {
    fn default() -> Self {
        Self {
            view_start: 0.0,
            view_end: 0.0,
            cursor: 0.0,
            duration: 0.0,
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
}
