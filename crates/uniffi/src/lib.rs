//! UniFFI bindings for sadda. Exposes a small subset of the engine surface
//! through UniFFI so it can be consumed from Swift / Kotlin / Python clients.
//!
//! At Phase 0 only `engine_version()` is exposed — the architectural smoke
//! test for the Rust ↔ Swift bridge. Real mobile API expansion lands in
//! Phase 6 (per the 2026-05-18 milestone plan).

uniffi::setup_scaffolding!();

#[uniffi::export]
pub fn engine_version() -> String {
    sadda_engine::version().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_engine_version() {
        assert!(!engine_version().is_empty());
    }
}
