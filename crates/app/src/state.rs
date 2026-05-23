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

/// Cached min/max envelope of a bundle's mono mixdown. Computed once
/// at bundle-select time and reused across frames at any plot width
/// (B2 has no zoom yet — when C5 lands zoom, this cache gets a
/// per-frame re-bucketing path).
#[derive(Debug, Clone)]
pub struct EnvelopeCache {
    /// Bundle this cache was built for; reset when the user selects
    /// a different bundle.
    pub bundle_id: i64,
    /// Bundle audio sample rate; used for the x-axis tick labels.
    pub sample_rate: u32,
    /// Bundle audio duration in seconds; used for the x-axis bounds.
    pub duration_seconds: f64,
    /// Per-bucket (min, max). Length = the resolution requested at
    /// build time (clamped to the sample count for short audio).
    pub envelope: Vec<(f32, f32)>,
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
