//! Citation registry: maps a `processor_id` (the reverse-DNS identifier
//! recorded in `processing_run.processor_id`) to the publication a user
//! should cite when an analysis used that processor.
//!
//! This is the single machine-readable source of truth for citation
//! export; the per-method `## References` blocks in the `dsp` module
//! docs are the human-readable mirror, and the references here are
//! copied from those curated lists rather than reproduced from memory
//! (cross-references the DSP-method-diversity principle: every method
//! carries a published source).
//!
//! Processors with no academic source to cite — tool operations like
//! TextGrid / EAF import, live recording, or measures whose only
//! reference is a software manual — return `None`. They still appear in
//! the [`crate::Project::processing_runs`] provenance timeline; they
//! just don't contribute to a paper's reference list.

/// A literature reference for a processor, suitable for a reference
/// list in a paper or report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Citation {
    /// The processor this cites (matches `processing_run.processor_id`).
    pub processor_id: String,
    /// Human-readable reference string.
    pub reference: String,
    /// Bare DOI (e.g. `10.1109/PROC.1975.9792`) when one exists.
    pub doi: Option<String>,
}

/// Returns the citation for a `processor_id`, or `None` for processors
/// that have no academic source to cite.
///
/// Keyed on the canonical reverse-DNS ids the DSP / clinical / ML
/// slices record into `processing_run`. The map grows as those slices
/// land; an unknown id is simply uncited (`None`).
pub fn citation_for(processor_id: &str) -> Option<Citation> {
    let (reference, doi): (&str, Option<&str>) = match processor_id {
        "sadda.dsp.pitch.autocorrelation" | "sadda.dsp.pitch.windowed_autocorrelation" => (
            "Boersma, P. (1993). Accurate short-term analysis of the fundamental \
             frequency and the harmonics-to-noise ratio of a sampled sound. \
             Proceedings of the Institute of Phonetic Sciences 17: 97–110.",
            None,
        ),
        "sadda.dsp.formants.burg" => (
            "McCandless, S.S. (1974). An algorithm for automatic formant extraction \
             using linear prediction spectra. IEEE Trans. Acoust. Speech Signal \
             Process. 22(2): 135–141.",
            Some("10.1109/TASSP.1974.1162572"),
        ),
        "sadda.dsp.formants.autocorrelation" => (
            "Markel, J.D. (1972). Digital inverse filtering — a new tool for formant \
             trajectory estimation. IEEE Trans. Audio Electroacoust. 20(2): 129–137.",
            Some("10.1109/TAU.1972.1162366"),
        ),
        "sadda.dsp.lpc.burg" => (
            "Makhoul, J. (1975). Linear prediction: A tutorial review. \
             Proc. IEEE 63(4): 561–580.",
            Some("10.1109/PROC.1975.9792"),
        ),
        "sadda.dsp.mfcc" => (
            "Davis, S.B. & Mermelstein, P. (1980). Comparison of parametric \
             representations for monosyllabic word recognition in continuously \
             spoken sentences. IEEE Trans. Acoust. Speech Signal Process. 28(4): \
             357–366.",
            Some("10.1109/TASSP.1980.1163420"),
        ),
        "sadda.dsp.stft" | "sadda.dsp.spectrogram" => (
            "Allen, J.B. (1977). Short term spectral analysis, synthesis, and \
             modification by discrete Fourier transform. IEEE Trans. Acoust. \
             Speech Signal Process. 25(3): 235–238.",
            Some("10.1109/TASSP.1977.1162950"),
        ),
        "sadda.align.forced_align" => (
            "Graves, A., Fernández, S., Gomez, F. & Schmidhuber, J. (2006). \
             Connectionist temporal classification: labelling unsegmented \
             sequence data with recurrent neural networks. ICML 2006, 369–376.",
            Some("10.1145/1143844.1143891"),
        ),
        "sadda.align.wav2vec2_espeak" => (
            "Xu, Q., Baevski, A. & Auli, M. (2022). Simple and Effective \
             Zero-shot Cross-lingual Phoneme Recognition. Interspeech 2022, \
             2113–2117.",
            Some("10.21437/Interspeech.2022-60"),
        ),
        _ => return None,
    };
    Some(Citation {
        processor_id: processor_id.to_string(),
        reference: reference.to_string(),
        doi: doi.map(str::to_string),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_processor_has_citation() {
        let c = citation_for("sadda.dsp.mfcc").expect("mfcc is cited");
        assert!(c.reference.contains("Davis"));
        assert_eq!(c.doi.as_deref(), Some("10.1109/TASSP.1980.1163420"));
    }

    #[test]
    fn pitch_methods_share_boersma() {
        let a = citation_for("sadda.dsp.pitch.autocorrelation").unwrap();
        let b = citation_for("sadda.dsp.pitch.windowed_autocorrelation").unwrap();
        assert_eq!(a.reference, b.reference);
        assert!(a.reference.contains("Boersma"));
    }

    #[test]
    fn forced_align_cites_ctc() {
        let c = citation_for("sadda.align.forced_align").expect("forced align is cited");
        assert!(c.reference.contains("Graves"));
        assert_eq!(c.doi.as_deref(), Some("10.1145/1143844.1143891"));
    }

    #[test]
    fn acoustic_model_cites_xu() {
        let c = citation_for("sadda.align.wav2vec2_espeak").expect("acoustic model is cited");
        assert!(c.reference.contains("Xu"));
        assert_eq!(c.doi.as_deref(), Some("10.21437/Interspeech.2022-60"));
    }

    #[test]
    fn tool_operations_are_uncited() {
        assert!(citation_for("sadda.io.textgrid.import").is_none());
        assert!(citation_for("sadda.unknown.processor").is_none());
    }
}
