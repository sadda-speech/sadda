//! PyO3 bindings for sadda. Placeholder skeleton; `#[pymodule]` wiring lands when there's something worth exposing.

pub fn engine_version() -> &'static str {
    sadda_engine::version()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxies_engine_version() {
        assert_eq!(engine_version(), sadda_engine::version());
    }
}
