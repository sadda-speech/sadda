//! Parquet sidecar I/O for the three dense tier types
//! (`continuous_numeric`, `continuous_vector`, `categorical_sampled`).
//! Pure read/write helpers — no [`crate::corpus::Project`] coupling — so
//! they're unit-testable against a temporary file path.
//!
//! Design: see the 2026-05-21 DEVLOG entry "Dense tier types + Parquet
//! sidecars (B3)".

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, FixedSizeListArray, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use ndarray::{Array2, ArrayView2};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::error::{EngineError, Result};

const VALUES_FIELD_NAME: &str = "item";

fn writer_props() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build()
}

fn map_arrow_err(e: arrow::error::ArrowError) -> EngineError {
    EngineError::Corpus(format!("arrow error: {e}"))
}

fn map_parquet_err(e: parquet::errors::ParquetError) -> EngineError {
    EngineError::Corpus(format!("parquet error: {e}"))
}

/// Writes a single-column Float64 Parquet file at `path`. The column is
/// named `value`.
pub fn write_continuous_numeric(path: &Path, samples: &[f64]) -> Result<()> {
    if samples.is_empty() {
        return Err(EngineError::Corpus(
            "write_continuous_numeric: empty input".into(),
        ));
    }
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Float64,
        false,
    )]));
    let array: ArrayRef = Arc::new(Float64Array::from(samples.to_vec()));
    let batch = RecordBatch::try_new(schema.clone(), vec![array]).map_err(map_arrow_err)?;

    let file = std::fs::File::create(path)?;
    let mut writer =
        ArrowWriter::try_new(file, schema, Some(writer_props())).map_err(map_parquet_err)?;
    writer.write(&batch).map_err(map_parquet_err)?;
    writer.close().map_err(map_parquet_err)?;
    Ok(())
}

/// Reads a Parquet file written by [`write_continuous_numeric`] into a
/// `Vec<f64>`. Returns the row count as `Vec::len()`.
pub fn read_continuous_numeric(path: &Path) -> Result<Vec<f64>> {
    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(map_parquet_err)?
        .build()
        .map_err(map_parquet_err)?;
    let mut out: Vec<f64> = Vec::new();
    for batch in reader {
        let batch = batch.map_err(map_arrow_err)?;
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                EngineError::Corpus("expected Float64 column for continuous_numeric".into())
            })?;
        out.extend(col.values().iter().copied());
    }
    Ok(out)
}

/// Writes a 2-D `Array2<f64>` (shape `[n_frames, n_dims]`) as a Parquet file
/// with a single `FixedSizeList<Float64>` column named `value`.
pub fn write_continuous_vector(path: &Path, frames: ArrayView2<'_, f64>) -> Result<()> {
    let (n_frames, n_dims) = frames.dim();
    if n_frames == 0 || n_dims == 0 {
        return Err(EngineError::Corpus(
            "write_continuous_vector: empty input".into(),
        ));
    }
    // Flatten to row-major contiguous storage.
    let standard = frames.as_standard_layout();
    let flat: Vec<f64> = standard.iter().copied().collect();
    let values: ArrayRef = Arc::new(Float64Array::from(flat));

    let item_field = Arc::new(Field::new(VALUES_FIELD_NAME, DataType::Float64, false));
    let fsl = FixedSizeListArray::new(item_field.clone(), n_dims as i32, values, None);
    let outer_field = Field::new(
        "value",
        DataType::FixedSizeList(item_field, n_dims as i32),
        false,
    );
    let schema = Arc::new(Schema::new(vec![outer_field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(fsl)]).map_err(map_arrow_err)?;

    let file = std::fs::File::create(path)?;
    let mut writer =
        ArrowWriter::try_new(file, schema, Some(writer_props())).map_err(map_parquet_err)?;
    writer.write(&batch).map_err(map_parquet_err)?;
    writer.close().map_err(map_parquet_err)?;
    Ok(())
}

/// Reads a Parquet file written by [`write_continuous_vector`] back into an
/// `Array2<f64>` of shape `[n_frames, n_dims]`.
pub fn read_continuous_vector(path: &Path) -> Result<Array2<f64>> {
    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(map_parquet_err)?
        .build()
        .map_err(map_parquet_err)?;
    let mut all_rows: Vec<f64> = Vec::new();
    let mut n_dims: Option<usize> = None;
    let mut n_frames = 0usize;
    for batch in reader {
        let batch = batch.map_err(map_arrow_err)?;
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                EngineError::Corpus("expected FixedSizeList column for continuous_vector".into())
            })?;
        let dims_here = col.value_length() as usize;
        match n_dims {
            None => n_dims = Some(dims_here),
            Some(prev) if prev != dims_here => {
                return Err(EngineError::Corpus(format!(
                    "FixedSizeList width changed mid-file: {prev} vs {dims_here}"
                )));
            }
            _ => {}
        }
        let values = col
            .values()
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| EngineError::Corpus("FixedSizeList values must be Float64".into()))?;
        // Each FixedSizeList row owns `dims_here` contiguous values starting
        // at row_index * dims_here. The whole-values buffer is the flat
        // row-major contents.
        let start = col.value_offset(0) as usize;
        let count = col.len() * dims_here;
        all_rows.extend(values.values()[start..start + count].iter().copied());
        n_frames += col.len();
    }
    let n_dims = n_dims.unwrap_or(0);
    if n_dims == 0 || n_frames == 0 {
        return Err(EngineError::Corpus(
            "read_continuous_vector: empty Parquet file".into(),
        ));
    }
    Array2::from_shape_vec((n_frames, n_dims), all_rows).map_err(|e| {
        EngineError::Corpus(format!(
            "Array2::from_shape_vec failed for [{n_frames}, {n_dims}]: {e}"
        ))
    })
}

/// Writes a single Utf8 column (plain string; no dictionary encoding at v1)
/// named `value`.
pub fn write_categorical_sampled(path: &Path, labels: &[String]) -> Result<()> {
    if labels.is_empty() {
        return Err(EngineError::Corpus(
            "write_categorical_sampled: empty input".into(),
        ));
    }
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Utf8,
        false,
    )]));
    let array: ArrayRef = Arc::new(StringArray::from(
        labels.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(schema.clone(), vec![array]).map_err(map_arrow_err)?;

    let file = std::fs::File::create(path)?;
    let mut writer =
        ArrowWriter::try_new(file, schema, Some(writer_props())).map_err(map_parquet_err)?;
    writer.write(&batch).map_err(map_parquet_err)?;
    writer.close().map_err(map_parquet_err)?;
    Ok(())
}

/// Reads a `categorical_sampled` Parquet file back into `Vec<String>`.
pub fn read_categorical_sampled(path: &Path) -> Result<Vec<String>> {
    let file = std::fs::File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(map_parquet_err)?
        .build()
        .map_err(map_parquet_err)?;
    let mut out: Vec<String> = Vec::new();
    for batch in reader {
        let batch = batch.map_err(map_arrow_err)?;
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                EngineError::Corpus("expected Utf8 column for categorical_sampled".into())
            })?;
        for i in 0..col.len() {
            if col.is_null(i) {
                return Err(EngineError::Corpus(
                    "null label in categorical_sampled column".into(),
                ));
            }
            out.push(col.value(i).to_string());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "sadda_storage_dense_{}_{}.parquet",
            std::process::id(),
            name
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn continuous_numeric_round_trip() {
        let path = tmp_path("numeric");
        let samples: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect();
        write_continuous_numeric(&path, &samples).unwrap();
        let back = read_continuous_numeric(&path).unwrap();
        assert_eq!(samples, back);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn continuous_vector_round_trip() {
        let path = tmp_path("vector");
        let n_frames = 50usize;
        let n_dims = 4usize;
        let arr = Array::from_shape_fn((n_frames, n_dims), |(r, c)| (r * 10 + c) as f64 * 0.5);
        write_continuous_vector(&path, arr.view()).unwrap();
        let back = read_continuous_vector(&path).unwrap();
        assert_eq!(back.dim(), (n_frames, n_dims));
        assert_eq!(arr, back);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn categorical_sampled_round_trip() {
        let path = tmp_path("categorical");
        let labels: Vec<String> = ["a", "b", "b", "c", "a"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        write_categorical_sampled(&path, &labels).unwrap();
        let back = read_categorical_sampled(&path).unwrap();
        assert_eq!(labels, back);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_continuous_numeric_rejects_empty() {
        let path = tmp_path("empty_numeric");
        let err = write_continuous_numeric(&path, &[]).unwrap_err();
        assert!(matches!(err, EngineError::Corpus(_)));
    }

    /// Regression guard for the "S3 dense-read ordering" scare (DEVLOG
    /// 2026-05-31): a degraded session reported `read_continuous_numeric`
    /// returning rotated/reordered samples intermittently. Parquet reads are
    /// row-ordered, so a rotation is structurally impossible here — this test
    /// pins that. It writes the exact `[10×10, 40×10]` series from the S3 end-
    /// to-end test, forces *many small row groups* (so the reader yields
    /// multiple batches that must concatenate in order), and reads it back many
    /// times from many threads. Any rotation/reordering would fail an exact
    /// `assert_eq` on every read.
    #[test]
    fn read_continuous_numeric_preserves_order_under_repeated_and_concurrent_reads() {
        use std::sync::Arc;
        use std::thread;

        let path = tmp_path("order_stress");
        // The S3 series, plus a longer monotonic series to span many row groups.
        let short: Vec<f64> = [10.0; 10].into_iter().chain([40.0; 10]).collect();
        let long: Vec<f64> = (0..5000).map(|i| i as f64).collect();

        // Force tiny row groups so the reader must stitch many batches in order.
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_max_row_group_row_count(Some(7))
            .build();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Float64Array::from(long.clone())) as ArrayRef],
        )
        .unwrap();
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        // The short series via the normal writer (single row group).
        let short_path = tmp_path("order_stress_short");
        write_continuous_numeric(&short_path, &short).unwrap();

        // Repeated sequential reads.
        for _ in 0..200 {
            assert_eq!(read_continuous_numeric(&path).unwrap(), long);
            assert_eq!(read_continuous_numeric(&short_path).unwrap(), short);
        }

        // Concurrent reads from many threads sharing the same files.
        let long = Arc::new(long);
        let short = Arc::new(short);
        let path = Arc::new(path);
        let short_path = Arc::new(short_path);
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let (long, short, path, short_path) = (
                    Arc::clone(&long),
                    Arc::clone(&short),
                    Arc::clone(&path),
                    Arc::clone(&short_path),
                );
                thread::spawn(move || {
                    for _ in 0..100 {
                        assert_eq!(read_continuous_numeric(&path).unwrap(), *long);
                        assert_eq!(read_continuous_numeric(&short_path).unwrap(), *short);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        let _ = std::fs::remove_file(&*path);
        let _ = std::fs::remove_file(&*short_path);
    }
}
