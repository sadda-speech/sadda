//! Formant preset domain (roadmap item 6) — the formant instantiation of the
//! generic [`crate::preset`] registry, mirroring [`crate::dsp::preset`] (MFCC)
//! and [`crate::pitch_preset`] (pitch).
//!
//! A formant preset is a named [`FormantsConfig`] (which already bundles the
//! [`LpcMethod`](crate::dsp::lpc::LpcMethod) — so, unlike pitch, no wrapper is
//! needed) plus provenance, stored as one `<id>.toml` file under
//! `~/.local/share/sadda/presets/formant/`. The built-in presets
//! ([`formant_builtin_presets`]) are the reference LPC methods at their
//! defaults (Praat Burg / autocorrelation).

use crate::dsp::formants::FormantsConfig;
use crate::dsp::lpc::LpcMethod;
use crate::preset::{Preset, PresetDomain, PresetStore};

/// A named formant preset: a [`FormantsConfig`] plus provenance.
pub type FormantPreset = Preset<FormantsConfig>;

/// A user-level store of formant presets (built-ins + on-disk `<id>.toml`).
pub type FormantPresetStore = PresetStore<FormantsConfig>;

impl PresetDomain for FormantsConfig {
    fn subdir() -> &'static str {
        "formant"
    }
    fn builtins() -> Vec<FormantPreset> {
        formant_builtin_presets()
    }
}

/// The built-in authoritative formant presets, in code: each reference LPC
/// method at its default config. `FormantsConfig::default()` is Praat's
/// `Sound.to_formant_burg` convention (Burg, pre-emph 0.97, 5 formants).
pub fn formant_builtin_presets() -> Vec<FormantPreset> {
    vec![
        FormantPreset {
            id: "praat-burg".into(),
            version: "1.0.0".into(),
            title: "Praat Burg".into(),
            description: "Praat Sound.to_formant_burg convention: Burg LPC, \
                pre-emphasis 0.97, 5 formants. Burg estimates reflection \
                coefficients directly (no autocorrelation windowing), so it's \
                more accurate on short frames — the default."
                .into(),
            based_on: "praat".into(),
            faithful: true,
            reference: Some("Praat — Sound.to_formant_burg".into()),
            params: FormantsConfig {
                lpc_method: LpcMethod::Burg,
                ..FormantsConfig::default()
            },
        },
        FormantPreset {
            id: "autocorrelation".into(),
            version: "1.0.0".into(),
            title: "Autocorrelation (Levinson–Durbin)".into(),
            description: "Autocorrelation-method LPC via Levinson–Durbin: always \
                stable, but tapers energy at frame edges (zero-extension), which \
                biases formant estimates on short frames."
                .into(),
            based_on: "autocorrelation".into(),
            faithful: true,
            reference: Some("Makhoul 1975; Markel & Gray 1976".into()),
            params: FormantsConfig {
                lpc_method: LpcMethod::Autocorrelation,
                ..FormantsConfig::default()
            },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_presets_round_trip_through_toml() {
        for preset in formant_builtin_presets() {
            let back = FormantPreset::from_toml(&preset.to_toml().unwrap()).unwrap();
            assert_eq!(preset, back, "preset {:?} did not round-trip", preset.id);
        }
    }

    #[test]
    fn methods_serialize_as_snake_case() {
        let burg = formant_builtin_presets()
            .into_iter()
            .find(|p| p.id == "praat-burg")
            .unwrap()
            .to_toml()
            .unwrap();
        assert!(burg.contains("lpc_method = \"burg\""), "{burg}");
    }

    #[test]
    fn store_resolves_builtins() {
        let store =
            FormantPresetStore::new(std::env::temp_dir().join("sadda_formant_preset_smoke"));
        assert!(store.get("praat-burg").is_some());
        assert!(store.get("autocorrelation").is_some());
        assert_eq!(store.list().len(), formant_builtin_presets().len());
    }

    #[test]
    fn builtin_ids_are_unique_and_valid() {
        let presets = formant_builtin_presets();
        for p in &presets {
            assert!(crate::preset::is_valid_id(&p.id), "bad id {:?}", p.id);
        }
        let mut ids: Vec<_> = presets.iter().map(|p| p.id.clone()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), presets.len(), "duplicate built-in ids");
    }
}
