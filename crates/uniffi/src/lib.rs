//! UniFFI bindings for sadda. Placeholder skeleton; bindings land in the Phase 0 spike → Phase 6.

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
