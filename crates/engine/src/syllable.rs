//! Rule-based syllabification of an IPA phone sequence (the A3 slice).
//!
//! Derives a **Syllable** tier from the **Phone** tier by rule — no model. Two
//! classic principles:
//!
//! - **Sonority Sequencing Principle (SSP):** syllable nuclei are sonority
//!   peaks (vowels); sonority rises through the onset to the nucleus and falls
//!   through the coda.
//! - **Maximal Onset Principle (MOP):** an intervocalic consonant cluster is
//!   assigned to the following syllable's onset as far as a rising-sonority
//!   onset allows; the remainder is the preceding syllable's coda.
//!
//! Reference: Clements, G. N. (1990), *The role of the sonority cycle in core
//! syllabification* (doi:10.1017/CBO9780511627736.017); the Maximal Onset
//! Principle follows Selkirk (1982), *The syllable*.
//!
//! **Scope / known limitations (v1).** This is a *universal*, language-agnostic
//! pass over IPA: one sonority scale, and **no per-language onset-legality
//! table**. So it cannot know that, e.g., English allows `/str/` as an onset —
//! pure SSP splits `s`+stop as sonority-falling (the well-known `sC`-cluster
//! exception), and it merges adjacent vowels into one nucleus (diphthong-
//! friendly, but it under-splits true vowel **hiatus** like `ˈke.ɒs`). A
//! language-tunable legality table is the accuracy refinement (backlogged).

/// Sonority rank of an IPA phone label — higher is more sonorous. Classified by
/// the phone's base symbol (leading tie-bars/spacing are skipped; diacritics and
/// length marks are ignored). The scale is a standard sonority hierarchy
/// (Clements 1990 — see the module references; the ranking of fine distinctions
/// such as the obstruent voicing split varies across authors):
/// vowel > glide > rhotic > lateral > nasal > voiced fricative > voiceless
/// fricative > voiced stop > voiceless stop. Unclassifiable labels rank lowest.
pub fn sonority(phone: &str) -> u8 {
    match base_char(phone) {
        Some(c) if is_vowel_char(c) => 10,
        Some(c) => consonant_sonority(c),
        None => 1,
    }
}

/// Whether a phone is a syllable nucleus: a vowel, or a consonant bearing a
/// syllabicity diacritic (combining below `U+0329` / above `U+030D`).
pub fn is_nucleus(phone: &str) -> bool {
    if phone.chars().any(|c| c == '\u{0329}' || c == '\u{030D}') {
        return true;
    }
    matches!(base_char(phone), Some(c) if is_vowel_char(c))
}

/// Syllabify a sequence of IPA phone labels into `[start, end)` index ranges,
/// one per syllable (contiguous, covering `0..phones.len()`). Adjacent vowels
/// form a single nucleus (diphthongs); intervocalic clusters split by SSP+MOP.
/// A sequence with no nucleus is returned as a single span. Pass one *word's*
/// phones — syllabification here is word-internal (callers segment on silence
/// and word boundaries first).
pub fn syllabify(phones: &[impl AsRef<str>]) -> Vec<(usize, usize)> {
    let n = phones.len();
    if n == 0 {
        return Vec::new();
    }
    let son: Vec<u8> = phones.iter().map(|p| sonority(p.as_ref())).collect();
    let nuc: Vec<bool> = phones.iter().map(|p| is_nucleus(p.as_ref())).collect();

    // Nucleus spans: maximal runs of adjacent nucleus phones (a diphthong or a
    // long vowel written as separate phones collapses to one nucleus).
    let mut nuclei: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        if nuc[i] {
            let start = i;
            while i < n && nuc[i] {
                i += 1;
            }
            nuclei.push((start, i));
        } else {
            i += 1;
        }
    }

    // No nucleus (e.g. a stray consonant run): one degenerate syllable.
    if nuclei.is_empty() {
        return vec![(0, n)];
    }

    // Boundary between each adjacent nucleus pair = the onset start of the later
    // syllable: the maximal suffix of the intervocalic cluster whose sonority
    // strictly rises toward the following nucleus.
    let mut starts: Vec<usize> = vec![0];
    for w in nuclei.windows(2) {
        let left_end = w[0].1; // first consonant index after the left nucleus
        let right_start = w[1].0; // the following nucleus's first phone
        let mut onset_start = right_start;
        let mut k = right_start; // walk left through the consonant gap
        while k > left_end {
            k -= 1;
            if son[k] < son[onset_start] {
                onset_start = k;
            } else {
                break;
            }
        }
        starts.push(onset_start);
    }

    // Ranges from the boundary starts; the last syllable runs to the end.
    let mut ranges = Vec::with_capacity(starts.len());
    for j in 0..starts.len() {
        let end = if j + 1 < starts.len() {
            starts[j + 1]
        } else {
            n
        };
        ranges.push((starts[j], end));
    }
    ranges
}

/// The phone's base symbol for classification: the first character that isn't a
/// tie-bar or combining mark. Returns `None` for an empty/all-diacritic label.
fn base_char(phone: &str) -> Option<char> {
    phone
        .chars()
        .find(|&c| c != '\u{0361}' && c != '\u{035C}' && !is_combining(c))
}

/// Combining diacritics and IPA length/suprasegmental marks that attach to a
/// base symbol and must not be treated as the base.
fn is_combining(c: char) -> bool {
    matches!(c, '\u{0300}'..='\u{036F}')          // combining diacritical marks
        || matches!(c, 'ː' | 'ˑ')                 // length marks (U+02D0, U+02D1)
        || matches!(c, 'ˈ' | 'ˌ') // stress marks
}

fn is_vowel_char(c: char) -> bool {
    matches!(
        c,
        'i' | 'y'
            | 'ɨ'
            | 'ʉ'
            | 'ɯ'
            | 'u'
            | 'ɪ'
            | 'ʏ'
            | 'ʊ'
            | 'e'
            | 'ø'
            | 'ɘ'
            | 'ɵ'
            | 'ɤ'
            | 'o'
            | 'ə'
            | 'ɛ'
            | 'œ'
            | 'ɜ'
            | 'ɞ'
            | 'ʌ'
            | 'ɔ'
            | 'æ'
            | 'ɐ'
            | 'a'
            | 'ɶ'
            | 'ä'
            | 'ɑ'
            | 'ɒ'
    )
}

/// Sonority of a consonant base symbol, on the same scale as [`sonority`].
fn consonant_sonority(c: char) -> u8 {
    match c {
        // Glides / non-lateral approximants.
        'j' | 'w' | 'ɥ' | 'ɰ' | 'ʋ' | 'ɹ' | 'ɻ' => 9,
        // Rhotics (trills / taps / flaps).
        'r' | 'ɾ' | 'ɽ' | 'ʀ' | 'ⱱ' => 8,
        // Laterals.
        'l' | 'ɭ' | 'ʎ' | 'ʟ' | 'ɫ' => 7,
        // Nasals.
        'm' | 'ɱ' | 'n' | 'ɳ' | 'ɲ' | 'ŋ' | 'ɴ' => 6,
        // Voiced fricatives.
        'v' | 'z' | 'ʒ' | 'ð' | 'ɣ' | 'ʁ' | 'ʐ' | 'ʑ' | 'β' | 'ʕ' => 5,
        // Voiceless fricatives.
        'f' | 's' | 'ʃ' | 'θ' | 'x' | 'χ' | 'h' | 'ç' | 'ħ' | 'ɸ' | 'ʂ' | 'ɕ' | 'ʜ' => 4,
        // Voiced stops (and voiced affricates, keyed on their stop onset).
        'b' | 'd' | 'ɡ' | 'g' | 'ɖ' | 'ɟ' | 'ɢ' => 3,
        // Voiceless stops (and voiceless affricates).
        'p' | 't' | 'k' | 'q' | 'ʈ' | 'c' | 'ʔ' => 2,
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn syl(phones: &[&str]) -> Vec<Vec<String>> {
        syllabify(phones)
            .into_iter()
            .map(|(s, e)| phones[s..e].iter().map(|p| p.to_string()).collect())
            .collect()
    }

    #[test]
    fn sonority_orders_classes() {
        assert!(sonority("a") > sonority("j")); // vowel > glide
        assert!(sonority("j") > sonority("l")); // glide > lateral
        assert!(sonority("l") > sonority("n")); // lateral > nasal
        assert!(sonority("n") > sonority("z")); // nasal > voiced fric
        assert!(sonority("z") > sonority("s")); // voiced > voiceless fric
        assert!(sonority("s") > sonority("t")); // fric > stop
        assert!(sonority("d") > sonority("t")); // voiced > voiceless stop
    }

    #[test]
    fn diphthong_is_one_nucleus() {
        // hello /h ə l o ʊ/ — the split o+ʊ diphthong is one nucleus → 2 syllables
        assert_eq!(
            syl(&["h", "ə", "l", "o", "ʊ"]),
            vec![vec!["h", "ə"], vec!["l", "o", "ʊ"]]
        );
    }

    #[test]
    fn maximal_onset_moves_cluster_to_next_syllable() {
        // /a b l a/ → a.bla (both consonants onset the 2nd syllable: b<l<a rises)
        assert_eq!(
            syl(&["a", "b", "l", "a"]),
            vec![vec!["a"], vec!["b", "l", "a"]]
        );
    }

    #[test]
    fn falling_sonority_splits_into_coda_plus_onset() {
        // /a l b a/ → al.ba (l>b, so l can't onset with b; l is coda, b onsets)
        assert_eq!(
            syl(&["a", "l", "b", "a"]),
            vec![vec!["a", "l"], vec!["b", "a"]]
        );
    }

    #[test]
    fn single_intervocalic_consonant_onsets_second() {
        // /a t a/ → a.ta
        assert_eq!(syl(&["a", "t", "a"]), vec![vec!["a"], vec!["t", "a"]]);
    }

    #[test]
    fn monosyllable_stays_one() {
        // /s t r ɛ ŋ θ/ (strength) — one nucleus, one syllable
        assert_eq!(syllabify(&["s", "t", "r", "ɛ", "ŋ", "θ"]), vec![(0, 6)]);
    }

    #[test]
    fn leading_and_trailing_consonants_attach() {
        // /p a t/ → one syllable pat (onset p, coda t)
        assert_eq!(syllabify(&["p", "a", "t"]), vec![(0, 3)]);
    }

    #[test]
    fn no_nucleus_is_single_span() {
        assert_eq!(syllabify(&["s", "t"]), vec![(0, 2)]);
        let empty: [&str; 0] = [];
        assert_eq!(syllabify(&empty), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn affricate_and_length_classify_by_base() {
        assert_eq!(sonority("t͡ʃ"), 2); // voiceless-stop onset of the affricate
        assert_eq!(sonority("d͡ʒ"), 3); // voiced
        assert_eq!(sonority("ɜː"), 10); // long vowel is still a vowel
        assert!(is_nucleus("ɜː"));
    }

    #[test]
    fn syllabic_consonant_is_a_nucleus() {
        // /b ʌ t n̩/ (button, syllabic n) → 2 syllables, the syllabic n a nucleus
        assert!(is_nucleus("n\u{0329}"));
        assert_eq!(
            syl(&["b", "ʌ", "t", "n\u{0329}"]),
            vec![vec!["b", "ʌ"], vec!["t", "n\u{0329}"]]
        );
    }
}
