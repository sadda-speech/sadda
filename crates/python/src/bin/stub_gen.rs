//! Generates `python/sadda/_native.pyi` from the `#[gen_stub_*]` attributes
//! on the pyo3 items in this crate.
//!
//! Usage: `cargo run --bin stub_gen` from the workspace root. CI runs this
//! plus `git diff --exit-code python/sadda/_native.pyi` to catch drift.

fn main() -> pyo3_stub_gen::Result<()> {
    let stub = sadda_python::stub_info()?;
    stub.generate()?;
    Ok(())
}
