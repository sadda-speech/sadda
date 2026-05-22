//! Round-trip I/O for external annotation / signal formats.
//!
//! - [`textgrid`] — Praat TextGrid (long + short text variants).
//! - [`eaf`] — ELAN .eaf (EAF 2.8 target on write; permissive on read).

pub mod eaf;
pub mod textgrid;
