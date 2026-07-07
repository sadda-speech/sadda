//! Grapheme-to-phoneme via the `espeak-ng` binary — the *target* side of forced
//! alignment, in the engine so the native GUI path (A5) produces the same
//! alignment target as the Python API (`python/sadda/align/g2p.py`).
//!
//! espeak-ng is a rule-based tool covering 100+ languages; it needs no model
//! download and carries no academic citation (the reference is the eSpeak NG
//! project itself). This layer is model-agnostic: it emits IPA with stress marks
//! stripped, and [`tokenize`] reconciles that IPA against a specific acoustic
//! model's vocabulary (a greedy longest-match, model-specific) at align time —
//! matching `sadda.align.tokenize`.

use std::collections::HashMap;
use std::process::Command;

use crate::error::EngineError;

/// IPA primary (ˈ, U+02C8) and secondary (ˌ, U+02CC) stress marks —
/// suprasegmental, never phones.
const STRESS: [char; 2] = ['ˈ', 'ˌ'];

/// One word and its espeak-ng IPA (stress marks stripped).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    /// The word as written, with edge punctuation removed.
    pub text: String,
    /// The espeak-ng IPA for the word, stress marks stripped.
    pub ipa: String,
}

/// Remove IPA primary/secondary stress marks (`ˈ ˌ`) from an IPA string.
pub fn strip_stress(ipa: &str) -> String {
    ipa.chars().filter(|c| !STRESS.contains(c)).collect()
}

/// Peel punctuation off a word's edges (word-internal apostrophes are kept, so
/// `don't` survives). Mirrors Python's `str.strip(string.punctuation + …)`.
fn strip_edge_punct(word: &str) -> &str {
    word.trim_matches(|c: char| c.is_ascii_punctuation() || "…—–«»¡¿".contains(c))
}

/// Run `espeak-ng -q --ipa -v <voice> <text>` and return its IPA output
/// (newlines folded to spaces, trimmed).
fn espeak_ipa(text: &str, voice: &str) -> Result<String, EngineError> {
    let output = Command::new("espeak-ng")
        .args(["-q", "--ipa", "-v", voice, text])
        .output()
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                EngineError::Align(
                    "espeak-ng executable not found. Install it (e.g. `apt install \
                     espeak-ng`, `brew install espeak-ng`)."
                        .to_string(),
                )
            } else {
                EngineError::Io(e)
            }
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(EngineError::Align(format!(
            "espeak-ng failed for voice {voice:?}: {}",
            stderr.trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .replace('\n', " ")
        .trim()
        .to_string())
}

/// Phonemize a transcript to per-word IPA (stress stripped) via espeak-ng.
///
/// Each whitespace-separated word is phonemized on its own (a clean 1:1
/// word→phones mapping, at the cost of cross-word coarticulation — acceptable
/// for an alignment target); edge punctuation is stripped. `voice` is an
/// espeak-ng language id (`"en-us"`, `"de"`, `"cmn"`, …). Mirrors
/// `sadda.align.phonemize`.
pub fn phonemize(text: &str, voice: &str) -> Result<Vec<Word>, EngineError> {
    let mut words = Vec::new();
    for token in text.split_whitespace() {
        let w = strip_edge_punct(token);
        if w.is_empty() {
            continue;
        }
        let ipa = strip_stress(espeak_ipa(w, voice)?.trim());
        words.push(Word {
            text: w.to_string(),
            ipa,
        });
    }
    Ok(words)
}

/// Greedy longest-match an IPA string into `vocab` class ids: at each position
/// the longest vocab key that matches wins (so a multi-character token like `dʒ`
/// beats `d`). Whitespace is skipped. Errors ([`EngineError::Align`]) naming the
/// offending substring when no key matches — how a phone the acoustic model
/// doesn't cover surfaces. Mirrors `sadda.align.tokenize`.
pub fn tokenize(ipa: &str, vocab: &HashMap<String, usize>) -> Result<Vec<usize>, EngineError> {
    // Longest keys first so greedy matching prefers multi-char tokens.
    let mut keys: Vec<&String> = vocab.keys().filter(|k| !k.is_empty()).collect();
    keys.sort_by_key(|k| std::cmp::Reverse(k.len()));

    let mut out = Vec::new();
    let mut rest = ipa;
    'scan: while let Some(first) = rest.chars().next() {
        if first.is_whitespace() {
            rest = &rest[first.len_utf8()..];
            continue;
        }
        for k in &keys {
            if rest.starts_with(k.as_str()) {
                out.push(vocab[*k]);
                rest = &rest[k.len()..];
                continue 'scan;
            }
        }
        return Err(EngineError::Align(format!(
            "phone not in acoustic-model vocab at {rest:?}"
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn espeak_available() -> bool {
        Command::new("espeak-ng")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    }

    #[test]
    fn strip_stress_removes_primary_and_secondary() {
        assert_eq!(strip_stress("həlˈoʊ ˌwɜːld"), "həloʊ wɜːld");
        assert_eq!(strip_stress("no marks"), "no marks");
    }

    #[test]
    fn tokenize_greedy_longest_match() {
        let vocab: HashMap<String, usize> = [("d", 1), ("ʒ", 2), ("dʒ", 3), ("a", 4)]
            .map(|(k, v)| (k.to_string(), v))
            .into();
        // 'dʒ' (a vocab token) beats 'd' + 'ʒ'
        assert_eq!(tokenize("dʒa", &vocab).unwrap(), vec![3, 4]);
        // whitespace skipped
        assert_eq!(tokenize("d a", &vocab).unwrap(), vec![1, 4]);
    }

    #[test]
    fn tokenize_rejects_unknown_phone() {
        let vocab: HashMap<String, usize> = [("d".to_string(), 1)].into();
        let err = tokenize("dx", &vocab).unwrap_err();
        assert!(matches!(err, EngineError::Align(m) if m.contains("not in acoustic-model vocab")));
    }

    #[test]
    fn strip_edge_punct_keeps_word_internal_apostrophe() {
        assert_eq!(strip_edge_punct("don't"), "don't");
        assert_eq!(strip_edge_punct("(hello,)"), "hello");
        assert_eq!(strip_edge_punct("¿que?"), "que");
    }

    #[test]
    fn phonemize_produces_stress_free_per_word_ipa() {
        if !espeak_available() {
            return; // espeak-ng not installed — skip (mirrors the Python skipif)
        }
        let words = phonemize("hello world", "en-us").unwrap();
        assert_eq!(
            words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
            ["hello", "world"]
        );
        for w in &words {
            assert!(!w.ipa.is_empty(), "{:?} produced no IPA", w.text);
            assert!(
                !w.ipa.contains('ˈ') && !w.ipa.contains('ˌ'),
                "stress survived: {:?}",
                w.ipa
            );
        }
    }

    #[test]
    fn phonemize_strips_edge_punctuation() {
        if !espeak_available() {
            return;
        }
        let words = phonemize("Hello, world!", "en-us").unwrap();
        assert_eq!(
            words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
            ["Hello", "world"]
        );
    }
}
