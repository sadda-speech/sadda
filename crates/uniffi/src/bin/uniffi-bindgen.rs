//! Thin wrapper around `uniffi::uniffi_bindgen_main()` so the bindgen tool
//! is version-locked to the same uniffi crate as the rest of the build.
//! Usage: `cargo run -p sadda-uniffi --bin uniffi-bindgen -- generate ...`

fn main() {
    uniffi::uniffi_bindgen_main()
}
