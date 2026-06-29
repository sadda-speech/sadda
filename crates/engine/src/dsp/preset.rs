//! On-disk MFCC preset registry (the 2026-06-29 "one parameterized pipeline +
//! presets" design, roadmap item 3).
//!
//! An MFCC *preset* is a named, serializable point in the [`MfccParams`] space
//! plus provenance metadata — which authoritative reference it derives from
//! ([`PresetLineage`]), whether it is a faithful reproduction of that
//! reference, and a citation. Built-in presets ([`builtin_presets`]) are the
//! validated authoritative reproductions (librosa / Kaldi / Praat); users can
//! save their own to an on-disk store, edit individual parameters, and reload
//! them.
//!
//! Unlike a [`crate::refdist::RefdistStore`] entry — which pairs a metadata
//! manifest with a separate binary data file, hence a directory per entry — a
//! preset has *no payload*: the parameters **are** the content. So a preset is
//! a single self-contained `<id>.toml` file, and the store is a flat directory
//! of them (`~/.local/share/sadda/presets/mfcc/`).
//!
//! Built-in presets are the source of truth in *code* (they are golden-tested
//! against their references); the store surfaces them alongside the user's
//! on-disk presets. Saving cannot overwrite a built-in id, so the authoritative
//! presets can never drift.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::dsp::mfcc::MfccParams;
use crate::error::{EngineError, Result};

/// The authoritative reference an MFCC preset derives from. Recorded so the
/// GUI/Python can say "librosa" or "custom (based on librosa)" honestly: a
/// preset edited away from its reference keeps its `based_on` but flips
/// [`MfccPreset::faithful`] to `false`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresetLineage {
    /// `librosa.feature.mfcc`.
    Librosa,
    /// Kaldi `compute-mfcc-feats`.
    Kaldi,
    /// Praat `Sound: To MFCC…`.
    Praat,
    /// HTK `HCopy`.
    Htk,
    /// Not derived from a single reference (built from scratch or heavily
    /// edited).
    #[default]
    Custom,
}

/// A named MFCC preset: a [`MfccParams`] point plus provenance. Serialized as
/// one `<id>.toml` file in an [`MfccPresetStore`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MfccPreset {
    /// Stable, filesystem-safe slug, e.g. `"librosa-default"` or `"my-asr"`.
    /// Used as the file stem; validated by [`is_valid_id`].
    pub id: String,
    /// Semantic version of this preset.
    pub version: String,
    /// Human-readable title.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
    /// Free-text description (what it's for, caveats).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// Which authoritative reference this preset derives from.
    #[serde(default)]
    pub based_on: PresetLineage,
    /// `true` iff running this preset through [`crate::dsp::mfcc_with_params`]
    /// reproduces `based_on`'s reference golden to tolerance. Built-in
    /// librosa/Kaldi are faithful; the Praat preset is **not** (its pipeline
    /// path is f32-approximate — use `mfcc(…, MfccMethod::Praat)` for the
    /// dedicated f64 path). Any user edit to a reference's defining knobs
    /// should set this `false`.
    #[serde(default)]
    pub faithful: bool,
    /// Citation / source for the reference (paper, docs URL, toolkit version).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    /// The full parameter set.
    pub params: MfccParams,
}

impl MfccPreset {
    /// Parses an `<id>.toml` preset manifest from a TOML string.
    pub fn from_toml(toml_str: &str) -> Result<MfccPreset> {
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

/// The built-in authoritative presets, in code (golden-tested against their
/// references). The scalar analysis parameters (frame size, hop, `n_mels`,
/// `n_mfcc`, `f_min`, `f_max`) are sensible 16-kHz defaults — they are *not*
/// reference-defining and are expected to be overridden per recording; the
/// *algorithmic* knobs are what make each faithful to its reference.
pub fn builtin_presets() -> Vec<MfccPreset> {
    vec![
        MfccPreset {
            id: "librosa-default".into(),
            version: "1.0.0".into(),
            title: "librosa (default)".into(),
            description: "librosa.feature.mfcc 0.11: Slaney mel + area norm, \
                power, 10·log10 + 80 dB floor, periodic Hann, center framing, \
                orthonormal DCT-II."
                .into(),
            based_on: PresetLineage::Librosa,
            faithful: true,
            reference: Some("librosa 0.11 — librosa.feature.mfcc".into()),
            params: MfccParams::librosa(0.025, 0.010, 40, 13, 0.0, 8000.0),
        },
        MfccPreset {
            id: "kaldi-default".into(),
            version: "1.0.0".into(),
            title: "Kaldi (compute-mfcc-feats)".into(),
            description: "Kaldi compute-mfcc-feats: DC removal, pre-emph 0.97, \
                Povey window, pow2 FFT, HTK mel, unit-peak filters, natural log, \
                orthonormal DCT-II, cepstral lifter 22, snip-edges."
                .into(),
            based_on: PresetLineage::Kaldi,
            faithful: true,
            reference: Some("Kaldi — compute-mfcc-feats (torchaudio kaldi-compliant)".into()),
            params: MfccParams::kaldi(0.025, 0.010, 23, 13, 20.0, 8000.0),
        },
        MfccPreset {
            id: "praat-default".into(),
            version: "1.0.0".into(),
            title: "Praat (Sound: To MFCC…)".into(),
            description: "Praat Sound: To MFCC…: Gaussian-2 window, HTK mel, \
                unit-peak filters, un-normalised DCT, c0 in column 0. NOTE: \
                through the f32 mfcc_with_params pipeline this is an \
                approximation — for faithful output use mfcc(…, Praat), the \
                dedicated f64 path."
                .into(),
            based_on: PresetLineage::Praat,
            // The params are Praat-faithful, but the shared pipeline they run
            // through is f32 (the dedicated Praat path is f64), so the preset
            // is honestly not byte-faithful yet. See the 2026-06-29 DEVLOG.
            faithful: false,
            reference: Some("Praat — Sound: To MFCC…".into()),
            params: MfccParams::praat(0.025, 0.010, 13, 8000.0),
        },
    ]
}

/// A user-level store of MFCC presets: a flat directory of `<id>.toml` files.
/// [`list`](Self::list) merges the built-in authoritative presets with the
/// user's on-disk ones; saving cannot overwrite a built-in id.
#[derive(Debug, Clone)]
pub struct MfccPresetStore {
    root: PathBuf,
}

impl MfccPresetStore {
    /// A store rooted at an explicit directory (used in tests and for
    /// alternative locations).
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    /// The default per-user store at the OS data directory —
    /// `~/.local/share/sadda/presets/mfcc/` on Linux, the platform equivalent
    /// elsewhere. The directory is created if missing.
    pub fn user_default() -> Result<Self> {
        let dirs = directories::ProjectDirs::from("", "", "sadda")
            .ok_or_else(|| EngineError::Preset("cannot determine user data directory".into()))?;
        let root = dirs.data_dir().join("presets").join("mfcc");
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// The store's root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The user's on-disk presets only (no built-ins), sorted by id, skipping
    /// files that don't parse as a valid preset.
    pub fn list_user(&self) -> Vec<MfccPreset> {
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
                if let Ok(preset) = MfccPreset::from_toml(&text) {
                    out.push(preset);
                }
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        out
    }

    /// All presets the user can choose from: the built-in authoritative set
    /// followed by the user's on-disk presets, sorted by id within each group.
    /// Built-in ids are reserved (saving rejects them), so there is no
    /// shadowing.
    pub fn list(&self) -> Vec<MfccPreset> {
        let mut out = builtin_presets();
        out.extend(self.list_user());
        out
    }

    /// The preset with this `id` — built-in or on-disk. `None` if absent.
    pub fn get(&self, id: &str) -> Option<MfccPreset> {
        builtin_presets()
            .into_iter()
            .find(|p| p.id == id)
            .or_else(|| self.list_user().into_iter().find(|p| p.id == id))
    }

    /// Writes `preset` to `<root>/<id>.toml`, creating the store directory if
    /// needed, and returns the file path. Errors if the id is invalid or
    /// collides with a built-in (the authoritative presets are immutable).
    /// Overwrites an existing user preset with the same id.
    pub fn save(&self, preset: &MfccPreset) -> Result<PathBuf> {
        if !is_valid_id(&preset.id) {
            return Err(EngineError::Preset(format!(
                "invalid preset id {:?}: use letters, digits, '-' or '_'",
                preset.id
            )));
        }
        if builtin_presets().iter().any(|b| b.id == preset.id) {
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
    use crate::dsp::mfcc::{MfccFilters, mfcc_with_params};

    fn temp_root() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "sadda_preset_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn params_round_trip_through_toml() {
        // The serde representation must survive TOML round-trip, including the
        // internally-tagged data enums (MfccFilters / MfccLog).
        for preset in builtin_presets() {
            let toml = preset.to_toml().unwrap();
            let back = MfccPreset::from_toml(&toml).unwrap();
            assert_eq!(preset, back, "preset {:?} did not round-trip", preset.id);
        }
    }

    #[test]
    fn mel_spacing_variant_round_trips() {
        // Praat uses the MelSpacing filter variant (not NMels) — exercise the
        // other internally-tagged arm explicitly.
        let praat = builtin_presets()
            .into_iter()
            .find(|p| p.id == "praat-default")
            .unwrap();
        assert!(matches!(
            praat.params.filters,
            MfccFilters::MelSpacing { .. }
        ));
        let back = MfccPreset::from_toml(&praat.to_toml().unwrap()).unwrap();
        assert_eq!(praat.params.filters, back.params.filters);
    }

    #[test]
    fn builtin_params_match_their_constructors() {
        // The on-disk preset must carry exactly the code-validated params, so
        // the registry never becomes a second source of truth.
        let by_id = |id: &str| builtin_presets().into_iter().find(|p| p.id == id).unwrap();
        assert_eq!(
            by_id("librosa-default").params,
            MfccParams::librosa(0.025, 0.010, 40, 13, 0.0, 8000.0)
        );
        assert_eq!(
            by_id("kaldi-default").params,
            MfccParams::kaldi(0.025, 0.010, 23, 13, 20.0, 8000.0)
        );
        assert_eq!(
            by_id("praat-default").params,
            MfccParams::praat(0.025, 0.010, 13, 8000.0)
        );
    }

    #[test]
    fn builtin_presets_run() {
        // A short tone through each built-in preset produces finite output of
        // the declared width — the presets are runnable, not just storable.
        let sr = 16_000;
        let tone: Vec<f32> = (0..8000)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sr as f32).sin())
            .collect();
        for preset in builtin_presets() {
            let out = mfcc_with_params(&tone, sr, &preset.params);
            assert_eq!(out.dim().1, preset.params.n_mfcc, "preset {:?}", preset.id);
            assert!(
                out.iter().all(|v| v.is_finite()),
                "preset {:?} produced non-finite output",
                preset.id
            );
        }
    }

    #[test]
    fn store_lists_builtins_plus_user() {
        let root = temp_root();
        let store = MfccPresetStore::new(&root);
        // Empty store: only built-ins.
        assert_eq!(store.list_user().len(), 0);
        assert_eq!(store.list().len(), builtin_presets().len());

        let mut mine = builtin_presets()
            .into_iter()
            .find(|p| p.id == "librosa-default")
            .unwrap();
        mine.id = "my-asr".into();
        mine.title = "My ASR front-end".into();
        mine.based_on = PresetLineage::Librosa;
        mine.faithful = false; // edited from a reference
        mine.params.n_mfcc = 20;
        let path = store.save(&mine).unwrap();
        assert!(path.is_file());

        assert_eq!(store.list_user().len(), 1);
        assert_eq!(store.list().len(), builtin_presets().len() + 1);

        let got = store.get("my-asr").unwrap();
        assert_eq!(got.params.n_mfcc, 20);
        assert!(!got.faithful);
        // Built-ins resolve through the same store.
        assert!(store.get("librosa-default").is_some());
        assert!(store.get("nonexistent").is_none());

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn save_rejects_builtin_id_and_bad_id() {
        let root = temp_root();
        let store = MfccPresetStore::new(&root);
        let mut p = builtin_presets().into_iter().next().unwrap();
        // Built-in id is reserved.
        assert!(store.save(&p).is_err());
        // Path-traversal / invalid ids are rejected.
        p.id = "../escape".into();
        assert!(store.save(&p).is_err());
        p.id = "".into();
        assert!(store.save(&p).is_err());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn delete_removes_user_preset_only() {
        let root = temp_root();
        let store = MfccPresetStore::new(&root);
        let mut p = builtin_presets().into_iter().next().unwrap();
        p.id = "scratch".into();
        store.save(&p).unwrap();
        assert!(store.delete("scratch").unwrap());
        assert!(!store.delete("scratch").unwrap()); // already gone
        assert!(!store.delete("librosa-default").unwrap()); // built-in: no file
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn id_validation() {
        assert!(is_valid_id("librosa-default"));
        assert!(is_valid_id("my_asr_2"));
        assert!(!is_valid_id(""));
        assert!(!is_valid_id(".."));
        assert!(!is_valid_id("has/slash"));
        assert!(!is_valid_id("has space"));
        assert!(!is_valid_id("dot.toml"));
    }
}
