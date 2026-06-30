//! Generic on-disk preset registry — the shared core behind the per-domain
//! preset stores (MFCC, and — roadmap item 6 — pitch / formants).
//!
//! A *preset* is a named, serializable parameter set plus provenance: which
//! authoritative reference it derives from (`based_on`), whether it is a
//! faithful reproduction of that reference, and a citation. Unlike a
//! [`crate::refdist::RefdistStore`] entry — which pairs a metadata manifest
//! with a separate binary data file, hence a directory per entry — a preset
//! has *no payload*: the parameters **are** the content. So a preset is a
//! single self-contained `<id>.toml` file, and a store is a flat directory of
//! them under `<data>/presets/<subdir>/`.
//!
//! Each parameter type implements [`PresetDomain`] to declare its on-disk
//! subdirectory and its built-in authoritative presets. Built-ins are the
//! source of truth in *code* (golden-tested); the store surfaces them
//! alongside the user's on-disk presets, and saving cannot overwrite a
//! built-in id, so the authoritative presets never drift.

use std::fs;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};

/// A named preset: a parameter set `P` plus provenance. Serialized as one
/// `<id>.toml` file in a [`PresetStore`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Preset<P> {
    /// Stable, filesystem-safe slug, e.g. `"librosa-default"`. Used as the
    /// file stem; validated by [`is_valid_id`].
    pub id: String,
    /// Semantic version of this preset.
    pub version: String,
    /// Human-readable title.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
    /// Free-text description (what it's for, caveats).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Which authoritative reference this preset derives from — a free-text
    /// lineage label whose vocabulary is domain-specific (`"librosa"` /
    /// `"praat"` / `"rapt"` / `"burg"` / `"custom"` / …). Recorded so the
    /// GUI/Python can say "praat" or "custom (based on praat)" honestly: a
    /// preset edited away from its reference keeps `based_on` but flips
    /// [`faithful`](Preset::faithful) to `false`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub based_on: String,
    /// `true` iff this preset reproduces `based_on`'s reference output to
    /// tolerance. Any user edit to a reference-defining knob should clear it.
    #[serde(default)]
    pub faithful: bool,
    /// Citation / source for the reference (paper, docs URL, toolkit version).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    /// The full parameter set.
    pub params: P,
}

impl<P: Serialize + DeserializeOwned> Preset<P> {
    /// Parses an `<id>.toml` preset manifest from a TOML string.
    pub fn from_toml(toml_str: &str) -> Result<Self> {
        toml::from_str(toml_str).map_err(|e| EngineError::Preset(format!("invalid preset: {e}")))
    }

    /// Serializes this preset to TOML.
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self)
            .map_err(|e| EngineError::Preset(format!("cannot serialize preset {:?}: {e}", self.id)))
    }
}

/// Is `id` a valid preset id — a non-empty, filesystem-safe slug with no path
/// separators or traversal? Ids become file stems, so this guards the store
/// against escaping its root.
pub fn is_valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id != "."
        && id != ".."
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// A domain of presets: ties a parameter type to its on-disk subdirectory and
/// its built-in authoritative presets. Implemented by each parameter type
/// (`MfccParams`, `PitchConfig`, …); a [`PresetStore`] is then generic over it.
pub trait PresetDomain: Sized + Clone + PartialEq + Serialize + DeserializeOwned {
    /// Subdirectory under `<data>/presets/` (e.g. `"mfcc"`, `"pitch"`).
    fn subdir() -> &'static str;
    /// The built-in authoritative presets (code-sourced, golden-tested).
    fn builtins() -> Vec<Preset<Self>>;
}

/// A user-level store of presets for one [`PresetDomain`]: a flat directory of
/// `<id>.toml` files. [`list`](Self::list) merges the built-in presets with
/// the user's on-disk ones; saving cannot overwrite a built-in id.
#[derive(Debug, Clone)]
pub struct PresetStore<P: PresetDomain> {
    root: PathBuf,
    _marker: PhantomData<P>,
}

impl<P: PresetDomain> PresetStore<P> {
    /// A store rooted at an explicit directory (used in tests and for
    /// alternative locations).
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            _marker: PhantomData,
        }
    }

    /// The default per-user store at the OS data directory —
    /// `~/.local/share/sadda/presets/<subdir>/` on Linux, the platform
    /// equivalent elsewhere. The directory is created if missing.
    pub fn user_default() -> Result<Self> {
        let dirs = directories::ProjectDirs::from("", "", "sadda")
            .ok_or_else(|| EngineError::Preset("cannot determine user data directory".into()))?;
        let root = dirs.data_dir().join("presets").join(P::subdir());
        fs::create_dir_all(&root)?;
        Ok(Self::new(root))
    }

    /// The store's root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The user's on-disk presets only (no built-ins), sorted by id, skipping
    /// files that don't parse as a valid preset.
    pub fn list_user(&self) -> Vec<Preset<P>> {
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(&self.root) else {
            return out;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            if let Ok(text) = fs::read_to_string(&path) {
                if let Ok(preset) = Preset::<P>::from_toml(&text) {
                    out.push(preset);
                }
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// All presets the user can choose from: the built-in authoritative set
    /// followed by the user's on-disk presets. Built-in ids are reserved
    /// (saving rejects them), so there is no shadowing.
    pub fn list(&self) -> Vec<Preset<P>> {
        let mut out = P::builtins();
        out.extend(self.list_user());
        out
    }

    /// The preset with this `id` — built-in or on-disk. `None` if absent.
    pub fn get(&self, id: &str) -> Option<Preset<P>> {
        P::builtins()
            .into_iter()
            .find(|p| p.id == id)
            .or_else(|| self.list_user().into_iter().find(|p| p.id == id))
    }

    /// Writes `preset` to `<root>/<id>.toml`, creating the store directory if
    /// needed, and returns the file path. Errors if the id is invalid or
    /// collides with a built-in (the authoritative presets are immutable).
    /// Overwrites an existing user preset with the same id.
    pub fn save(&self, preset: &Preset<P>) -> Result<PathBuf> {
        if !is_valid_id(&preset.id) {
            return Err(EngineError::Preset(format!(
                "invalid preset id {:?}: use letters, digits, '-' or '_'",
                preset.id
            )));
        }
        if P::builtins().iter().any(|b| b.id == preset.id) {
            return Err(EngineError::Preset(format!(
                "{:?} is a built-in preset id and cannot be overwritten; choose another id",
                preset.id
            )));
        }
        fs::create_dir_all(&self.root)?;
        let path = self.root.join(format!("{}.toml", preset.id));
        fs::write(&path, preset.to_toml()?)?;
        Ok(path)
    }

    /// Deletes the user preset with this `id`. Returns `true` if a file was
    /// removed, `false` if none existed. Built-in ids never have a file, so
    /// this is a no-op for them.
    pub fn delete(&self, id: &str) -> Result<bool> {
        if !is_valid_id(id) {
            return Err(EngineError::Preset(format!("invalid preset id {id:?}")));
        }
        let path = self.root.join(format!("{id}.toml"));
        if path.is_file() {
            fs::remove_file(&path)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny domain so the generic store can be exercised without depending
    /// on a real parameter type.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct DummyParams {
        gain: f32,
        mode: String,
    }

    impl PresetDomain for DummyParams {
        fn subdir() -> &'static str {
            "dummy"
        }
        fn builtins() -> Vec<Preset<DummyParams>> {
            vec![Preset {
                id: "builtin-a".into(),
                version: "1.0.0".into(),
                title: "Built-in A".into(),
                description: String::new(),
                based_on: "reference-x".into(),
                faithful: true,
                reference: None,
                params: DummyParams {
                    gain: 1.0,
                    mode: "x".into(),
                },
            }]
        }
    }

    fn temp_root() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "sadda_genpreset_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn round_trips_through_toml() {
        let p = &DummyParams::builtins()[0];
        let back = Preset::<DummyParams>::from_toml(&p.to_toml().unwrap()).unwrap();
        assert_eq!(*p, back);
    }

    #[test]
    fn store_lists_builtins_plus_user_and_protects_builtins() {
        let root = temp_root();
        let store: PresetStore<DummyParams> = PresetStore::new(&root);
        assert!(store.list_user().is_empty());
        assert_eq!(store.list().len(), 1); // the one built-in

        let mut mine = DummyParams::builtins()[0].clone();
        mine.id = "mine".into();
        mine.faithful = false;
        mine.params.gain = 2.0;
        store.save(&mine).unwrap();
        assert_eq!(store.list_user().len(), 1);
        assert_eq!(store.list().len(), 2);
        assert_eq!(store.get("mine").unwrap().params.gain, 2.0);
        assert!(store.get("builtin-a").is_some());

        // Built-in id reserved; bad ids rejected.
        let mut clash = mine.clone();
        clash.id = "builtin-a".into();
        assert!(store.save(&clash).is_err());
        clash.id = "../escape".into();
        assert!(store.save(&clash).is_err());

        assert!(store.delete("mine").unwrap());
        assert!(!store.delete("mine").unwrap());
        assert!(!store.delete("builtin-a").unwrap()); // built-in: no file
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn id_validation() {
        assert!(is_valid_id("librosa-default"));
        assert!(is_valid_id("my_asr_2"));
        assert!(!is_valid_id(""));
        assert!(!is_valid_id(".."));
        assert!(!is_valid_id("has/slash"));
        assert!(!is_valid_id("dot.toml"));
    }
}
