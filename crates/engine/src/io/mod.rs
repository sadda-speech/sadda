//! Round-trip I/O for external annotation / signal formats.
//!
//! - [`textgrid`] — Praat TextGrid (long + short text variants).
//! - [`eaf`] — ELAN .eaf (EAF 2.8 target on write; permissive on read).
//! - [`tabular`] — flat CSV + structured JSON export of sparse annotations.

pub mod eaf;
pub mod tabular;
pub mod textgrid;
