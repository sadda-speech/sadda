//! MFCC preset domain (roadmap item 3) — the MFCC instantiation of the generic
//! [`crate::preset`] registry.
//!
//! An MFCC preset is a named [`MfccParams`] point plus provenance, stored as
//! one `<id>.toml` file under `~/.local/share/sadda/presets/mfcc/`. The
//! built-in presets ([`builtin_presets`]) are the validated authoritative
//! reproductions (librosa / Kaldi / Praat), code-sourced and golden-tested;
//! `MfccParams`'s [`PresetDomain`] impl ties them to the `mfcc` subdirectory.

use crate::dsp::mfcc::MfccParams;
use crate::preset::{Preset, PresetDomain, PresetStore};

/// A named MFCC preset: an [`MfccParams`] point plus provenance.
pub type MfccPreset = Preset<MfccParams>;

/// A user-level store of MFCC presets (built-ins + on-disk `<id>.toml` files).
pub type MfccPresetStore = PresetStore<MfccParams>;

impl PresetDomain for MfccParams {
    fn subdir() -> &'static str {
        "mfcc"
    }
    fn builtins() -> Vec<MfccPreset> {
        builtin_presets()
    }
}

/// The built-in authoritative MFCC presets, in code (golden-tested against
/// their references). The scalar analysis parameters (frame size, hop,
/// `n_mels`, `n_mfcc`, `f_min`, `f_max`) are sensible 16-kHz defaults — they
/// are *not* reference-defining and are expected to be overridden per
/// recording; the *algorithmic* knobs are what make each faithful to its
/// reference.
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
            based_on: "librosa".into(),
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
            based_on: "kaldi".into(),
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
            based_on: "praat".into(),
            // The params are Praat-faithful, but the shared pipeline they run
            // through is f32 (the dedicated Praat path is f64), so the preset
            // is honestly not byte-faithful yet. See the 2026-06-29 DEVLOG.
            faithful: false,
            reference: Some("Praat — Sound: To MFCC…".into()),
            params: MfccParams::praat(0.025, 0.010, 13, 8000.0),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::mfcc::{MfccFilters, mfcc_with_params};

    #[test]
    fn builtin_presets_round_trip_through_toml() {
        // The MFCC serde representation must survive TOML round-trip, including
        // the internally-tagged data enums (MfccFilters / MfccLog).
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
    fn builtin_presets_run_and_resolve_via_store() {
        // A short tone through each built-in preset produces finite output of
        // the declared width — runnable, not just storable.
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
        // The domain wiring resolves built-ins through a store.
        let store = MfccPresetStore::new(std::env::temp_dir().join("sadda_mfcc_preset_smoke"));
        assert!(store.get("librosa-default").is_some());
        assert_eq!(store.list().len(), builtin_presets().len());
    }
}
