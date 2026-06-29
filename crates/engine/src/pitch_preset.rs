//! Pitch preset domain (roadmap item 6) — the pitch instantiation of the
//! generic [`crate::preset`] registry, mirroring [`crate::dsp::preset`] (MFCC).
//!
//! A pitch preset is a named [`PitchParams`] (a [`PitchMethod`] + its
//! [`PitchConfig`]) plus provenance, stored as one `<id>.toml` file under
//! `~/.local/share/sadda/presets/pitch/`. The built-in presets
//! ([`pitch_builtin_presets`]) are the reference trackers at their
//! authoritative defaults (Praat AC / YIN / pYIN / SWIPE′), code-sourced;
//! `PitchParams`'s [`PresetDomain`] impl ties them to the `pitch` subdirectory.

use crate::pitch::{PitchConfig, PitchMethod, PitchParams};
use crate::preset::{Preset, PresetDomain, PresetStore};

/// A named pitch preset: a [`PitchParams`] plus provenance.
pub type PitchPreset = Preset<PitchParams>;

/// A user-level store of pitch presets (built-ins + on-disk `<id>.toml` files).
pub type PitchPresetStore = PresetStore<PitchParams>;

impl PresetDomain for PitchParams {
    fn subdir() -> &'static str {
        "pitch"
    }
    fn builtins() -> Vec<PitchPreset> {
        pitch_builtin_presets()
    }
}

/// The built-in authoritative pitch presets, in code: each reference tracker
/// at its published defaults. `PitchConfig::default()` already matches Praat
/// 6.x's `Sound: To Pitch (ac)…` and the YIN/pYIN paper/librosa defaults
/// (see the field docs), so a method + default config is faithful to that
/// method's reference.
pub fn pitch_builtin_presets() -> Vec<PitchPreset> {
    let preset =
        |id: &str, title: &str, based_on: &str, reference: &str, method: PitchMethod| PitchPreset {
            id: id.into(),
            version: "1.0.0".into(),
            title: title.into(),
            description: String::new(),
            based_on: based_on.into(),
            faithful: true,
            reference: Some(reference.into()),
            params: PitchParams {
                method,
                config: PitchConfig::default(),
            },
        };
    vec![
        PitchPreset {
            description: "Faithful Boersma 1993 / Praat Sound: To Pitch (ac)…: \
                multi-candidate autocorrelation + Viterbi path-finding with \
                octave / octave-jump / voiced-unvoiced costs. Octave-robust; \
                the default."
                .into(),
            ..preset(
                "praat-ac",
                "Praat autocorrelation (Boersma)",
                "praat",
                "Boersma 1993 — Praat Sound: To Pitch (ac)…",
                PitchMethod::Boersma,
            )
        },
        PitchPreset {
            description: "de Cheveigné & Kawahara 2002 YIN — cumulative-mean-\
                normalized-difference tracker (threshold 0.1)."
                .into(),
            ..preset(
                "yin",
                "YIN",
                "yin",
                "de Cheveigné & Kawahara 2002 — YIN",
                PitchMethod::Yin,
            )
        },
        PitchPreset {
            description: "Mauch & Dixon 2014 pYIN — probabilistic YIN with a \
                beta-prior over thresholds + HMM smoothing. librosa's default."
                .into(),
            ..preset(
                "pyin",
                "pYIN (librosa)",
                "librosa",
                "Mauch & Dixon 2014 — pYIN (librosa pyin)",
                PitchMethod::PYin,
            )
        },
        PitchPreset {
            description: "Camacho & Harris 2008 SWIPE′ — sawtooth-inspired \
                spectral pitch estimator over ERB-scale loudness."
                .into(),
            ..preset(
                "swipe",
                "SWIPE′",
                "swipe",
                "Camacho & Harris 2008 — SWIPE′",
                PitchMethod::Swipe,
            )
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_presets_round_trip_through_toml() {
        for preset in pitch_builtin_presets() {
            let back = PitchPreset::from_toml(&preset.to_toml().unwrap()).unwrap();
            assert_eq!(preset, back, "preset {:?} did not round-trip", preset.id);
        }
    }

    #[test]
    fn methods_serialize_with_the_python_vocabulary() {
        let by_id = |id: &str| {
            pitch_builtin_presets()
                .into_iter()
                .find(|p| p.id == id)
                .unwrap()
                .to_toml()
                .unwrap()
        };
        assert!(by_id("praat-ac").contains("method = \"boersma\""));
        // pyin, not the snake_case default `p_yin`.
        assert!(
            by_id("pyin").contains("method = \"pyin\""),
            "{}",
            by_id("pyin")
        );
    }

    #[test]
    fn store_resolves_builtins() {
        let store = PitchPresetStore::new(std::env::temp_dir().join("sadda_pitch_preset_smoke"));
        assert!(store.get("praat-ac").is_some());
        assert!(store.get("yin").is_some());
        assert_eq!(store.list().len(), pitch_builtin_presets().len());
    }

    #[test]
    fn builtin_ids_are_unique_and_valid() {
        let presets = pitch_builtin_presets();
        for p in &presets {
            assert!(crate::preset::is_valid_id(&p.id), "bad id {:?}", p.id);
        }
        let mut ids: Vec<_> = presets.iter().map(|p| p.id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), presets.len(), "duplicate built-in ids");
    }
}
