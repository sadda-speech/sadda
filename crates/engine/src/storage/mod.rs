//! Storage primitives owned by the corpus: Parquet sidecar I/O for dense
//! tier types lives in [`dense`]. The audio loader lives in [`crate::audio`]
//! and the corpus database in [`crate::corpus`].

pub mod dense;
