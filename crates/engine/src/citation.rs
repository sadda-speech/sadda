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
//!
//! Weblink convention: every cited entry must resolve to a weblink via
//! [`Citation::weblink`] — a DOI (rendered `https://doi.org/<doi>`) or an
//! explicit [`Citation::url`]. A publication with no stable link is allowed,
//! but the entry must then carry a source comment stating that no weblink was
//! available (so "uncited-link" is a deliberate, recorded decision, never an
//! oversight). This mirrors the human-readable `## References` blocks.

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
    /// Explicit canonical URL for sources without a DOI (a publisher page,
    /// official manual, or open PDF). `None` when the DOI already yields the
    /// weblink — see [`Citation::weblink`].
    pub url: Option<String>,
}

impl Citation {
    /// A resolvable weblink for this citation, or `None` if none is known.
    ///
    /// Prefers an explicit [`Citation::url`]; otherwise derives the canonical
    /// `https://doi.org/<doi>` link from the DOI. The working practice is to give
    /// every citation a weblink; some publications genuinely have no stable link
    /// (an older book, a manual with no permalink), and `None` is a legitimate,
    /// documented outcome for those — the entry should say so in a comment.
    pub fn weblink(&self) -> Option<String> {
        self.url
            .clone()
            .or_else(|| self.doi.as_ref().map(|d| format!("https://doi.org/{d}")))
    }
}

/// Returns the citation for a `processor_id`, or `None` for processors
/// that have no academic source to cite.
///
/// Keyed on the canonical reverse-DNS ids the DSP / clinical / ML
/// slices record into `processing_run`. The map grows as those slices
/// land; an unknown id is simply uncited (`None`).
pub fn citation_for(processor_id: &str) -> Option<Citation> {
    let (reference, doi, url): (&str, Option<&str>, Option<&str>) = match processor_id {
        "sadda.dsp.pitch.autocorrelation" | "sadda.dsp.pitch.windowed_autocorrelation" => (
            "Boersma, P. (1993). Accurate short-term analysis of the fundamental \
             frequency and the harmonics-to-noise ratio of a sampled sound. \
             Proceedings of the Institute of Phonetic Sciences 17: 97–110.",
            None,
            // No DOI (IFA proceedings); the author's canonical open PDF.
            Some("https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf"),
        ),
        "sadda.dsp.formants.burg" => (
            "McCandless, S.S. (1974). An algorithm for automatic formant extraction \
             using linear prediction spectra. IEEE Trans. Acoust. Speech Signal \
             Process. 22(2): 135–141.",
            Some("10.1109/TASSP.1974.1162572"),
            None,
        ),
        "sadda.dsp.formants.autocorrelation" => (
            "Markel, J.D. (1972). Digital inverse filtering — a new tool for formant \
             trajectory estimation. IEEE Trans. Audio Electroacoust. 20(2): 129–137.",
            Some("10.1109/TAU.1972.1162366"),
            None,
        ),
        "sadda.dsp.lpc.burg" => (
            "Makhoul, J. (1975). Linear prediction: A tutorial review. \
             Proc. IEEE 63(4): 561–580.",
            Some("10.1109/PROC.1975.9792"),
            None,
        ),
        "sadda.dsp.mfcc" => (
            "Davis, S.B. & Mermelstein, P. (1980). Comparison of parametric \
             representations for monosyllabic word recognition in continuously \
             spoken sentences. IEEE Trans. Acoust. Speech Signal Process. 28(4): \
             357–366.",
            Some("10.1109/TASSP.1980.1163420"),
            None,
        ),
        "sadda.dsp.stft" | "sadda.dsp.spectrogram" => (
            "Allen, J.B. (1977). Short term spectral analysis, synthesis, and \
             modification by discrete Fourier transform. IEEE Trans. Acoust. \
             Speech Signal Process. 25(3): 235–238.",
            Some("10.1109/TASSP.1977.1162950"),
            None,
        ),
        "sadda.align.forced_align" => (
            "Graves, A., Fernández, S., Gomez, F. & Schmidhuber, J. (2006). \
             Connectionist temporal classification: labelling unsegmented \
             sequence data with recurrent neural networks. ICML 2006, 369–376.",
            Some("10.1145/1143844.1143891"),
            None,
        ),
        "sadda.align.wav2vec2_espeak" => (
            "Xu, Q., Baevski, A. & Auli, M. (2022). Simple and Effective \
             Zero-shot Cross-lingual Phoneme Recognition. Interspeech 2022, \
             2113–2117.",
            Some("10.21437/Interspeech.2022-60"),
            None,
        ),
        _ => return None,
    };
    Some(Citation {
        processor_id: processor_id.to_string(),
        reference: reference.to_string(),
        doi: doi.map(str::to_string),
        url: url.map(str::to_string),
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

    #[test]
    fn doi_citation_keeps_doi_and_derives_weblink() {
        // DOI-bearing entries keep the bare DOI *and* resolve to a weblink.
        let c = citation_for("sadda.dsp.mfcc").unwrap();
        assert_eq!(c.doi.as_deref(), Some("10.1109/TASSP.1980.1163420"));
        assert_eq!(c.url, None);
        assert_eq!(
            c.weblink().as_deref(),
            Some("https://doi.org/10.1109/TASSP.1980.1163420")
        );
    }

    #[test]
    fn doi_free_citation_has_explicit_weblink() {
        // Boersma (1993) has no DOI; it still resolves to a link via `url`.
        let c = citation_for("sadda.dsp.pitch.autocorrelation").unwrap();
        assert_eq!(c.doi, None);
        assert_eq!(
            c.weblink().as_deref(),
            Some("https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf")
        );
    }

    #[test]
    fn weblink_present_exactly_when_a_link_source_is() {
        // The contract: weblink() resolves iff the entry has a DOI or an
        // explicit URL. A citation with neither is allowed (a documented
        // linkless source) and simply returns None — it is not a bug.
        for id in [
            "sadda.dsp.pitch.autocorrelation",
            "sadda.dsp.pitch.windowed_autocorrelation",
            "sadda.dsp.formants.burg",
            "sadda.dsp.formants.autocorrelation",
            "sadda.dsp.lpc.burg",
            "sadda.dsp.mfcc",
            "sadda.dsp.stft",
            "sadda.dsp.spectrogram",
            "sadda.align.forced_align",
            "sadda.align.wav2vec2_espeak",
        ] {
            let c = citation_for(id).unwrap();
            assert_eq!(
                c.weblink().is_some(),
                c.doi.is_some() || c.url.is_some(),
                "{id}: weblink must resolve iff a DOI or URL is present"
            );
        }

        // A citation with neither a DOI nor a URL is a legitimate linkless
        // source — weblink() is simply None.
        let linkless = Citation {
            processor_id: "x".into(),
            reference: "Some Author (1968). A Book With No DOI. Publisher.".into(),
            doi: None,
            url: None,
        };
        assert_eq!(linkless.weblink(), None);
    }
}
