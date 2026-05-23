//! Pure-data app state. No egui types here — that's the point. Tests
//! cover the recent-projects ordering / capping / cleanup logic
//! without spinning up an egui context.

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
}
