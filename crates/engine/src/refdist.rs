//! Reference distributions — the consumption side (Phase 3 C7).
//!
//! A *reference distribution* is a tagged statistical summary (or sample)
//! of an acoustic/articulatory measure over a population, or a
//! prescriptive target, that the GUI can render a measurement against —
//! vowel-formant clouds, age/sex-normed clinical ranges, f0 statistics,
//! voice-coach target zones, and so on. The governance, three-tier
//! registry, and `refdist.toml` format are settled in the 2026-05-18
//! "Reference distribution governance" DEVLOG entry.
//!
//! This module implements the **consumption** half: parse a `refdist.toml`
//! manifest, resolve distributions from a user-level cache directory
//! (`~/.local/share/sadda/refdist/`), and query them by population /
//! measure facets. It is deliberately independent of any *hosted*
//! registry — fetching tarballs over HTTP lands with the registry itself
//! (C8). A [`crate::Project`] pins the versions it used in `project.toml`
//! for reproducibility.

use std::fs;
use std::path::{Path, PathBuf};

use arrow::array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray, StringArray,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};

/// A parsed `refdist.toml` manifest. Discovery, validation, citation, and
/// rendering all key off this. Most fields are optional so distributions
/// can omit sections that don't apply to them.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefdistManifest {
    /// Stable distribution id, e.g. `"hillenbrand-1995-amE-vowels"`.
    pub id: String,
    /// Semantic version of this distribution.
    pub version: String,
    /// Human-readable title.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
    /// DOI, if one was minted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doi: Option<String>,
    /// SPDX license identifier of the distribution data, e.g.
    /// `"CC0-1.0"` / `"CC-BY-4.0"` / `"ODC-BY-1.0"`. The registry CI
    /// (C8) enforces the per-tier license policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Citation metadata (authors / year / journal / bibtex).
    #[serde(default)]
    pub citation: Citation,
    /// Who/what the distribution describes.
    #[serde(default)]
    pub population: Population,
    /// What is measured, and whether it's observed vs prescriptive.
    #[serde(default)]
    pub measure: Measure,
    /// Privacy / shareability declarations.
    #[serde(default)]
    pub privacy: Privacy,
    /// Pointer to the data file and its layout.
    #[serde(default)]
    pub schema: Schema,
}

/// Citation block of a [`RefdistManifest`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Citation {
    /// Author list.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub authors: Vec<String>,
    /// Publication year.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub year: Option<i32>,
    /// Journal / venue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub journal: Option<String>,
    /// A ready-to-paste BibTeX entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bibtex: Option<String>,
}

/// Population block — the facets discovery and `query` match on.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Population {
    /// ISO 639-3 language code, e.g. `"eng"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Variety / dialect, e.g. `"AmE"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variety: Option<String>,
    /// Sexes represented, e.g. `["m", "f", "c"]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sex: Vec<String>,
    /// Age bands represented, e.g. `["adult", "child"]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub age_band: Vec<String>,
    /// Number of speakers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_speakers: Option<u64>,
    /// Number of tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_tokens: Option<u64>,
}

/// What kind of reference a distribution is — kept distinct so the GUI
/// never conflates "what people sound like" with "what to aim for".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasureKind {
    /// Raw samples from a measured population.
    #[default]
    ObservedDistribution,
    /// Summary statistics only (mean / SD / percentiles); no raw values.
    SummaryNormativeRange,
    /// A prescriptive goal region (voice-coach / L2 use).
    TargetZone,
}

/// Measure block — which parameters, units, and (for speech) phones.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Measure {
    /// Observed vs summary vs prescriptive.
    #[serde(default)]
    pub kind: MeasureKind,
    /// Measured parameters, e.g. `["F1", "F2", "F3"]` or `["jitter_local"]`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parameters: Vec<String>,
    /// Units of the parameters, e.g. `"Hz"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub units: Option<String>,
    /// Phones the distribution covers (ARPABET/IPA strings), if applicable.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub phones: Vec<String>,
    /// Phonetic context, e.g. `"hVd"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    /// Free-text description of how the measurement was made.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement_method: Option<String>,
}

/// Privacy / shareability block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Privacy {
    /// `"raw_samples"` or `"summary_only"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shareability: Option<String>,
    /// k-anonymity floor per subgroup (default 5 enforced by the registry).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_n_per_subgroup: Option<u64>,
    /// Required `true` for small-language community data.
    #[serde(default)]
    pub community_consent: bool,
}

/// Schema block — the data file and its column layout.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Schema {
    /// Data file name relative to the distribution directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_file: Option<String>,
    /// `"long"` or `"wide"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<String>,
    /// Column names in the data file.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub columns: Vec<String>,
}

/// A reference distribution resolved on disk: its manifest plus the
/// directory it lives in (so the data file can be located).
#[derive(Debug, Clone)]
pub struct RefDist {
    /// The parsed manifest.
    pub manifest: RefdistManifest,
    /// Directory containing `refdist.toml` and the data file.
    pub dir: PathBuf,
}

/// Distribution-shape summary of a 1-D measure (D10): enough to draw a
/// band overlay or a numeric readout without re-reading the data file.
/// For an [`MeasureKind::ObservedDistribution`] every field is empirical;
/// for a [`MeasureKind::SummaryNormativeRange`] the percentiles are a
/// **normal approximation** of the published mean/SD (see
/// [`Summary::from_mean_sd`]) — so a normative band and an observed band
/// render through the same fields, but the normative one is modelled, not
/// measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    /// Number of underlying values (raw samples, or declared speakers
    /// for a normative summary).
    pub n: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Standard deviation (sample SD, `n-1`, for raw samples).
    pub sd: f64,
    /// Minimum (observed) or `mean - 2·SD` (normative plotting range).
    pub min: f64,
    /// 5th percentile.
    pub p5: f64,
    /// 25th percentile (lower quartile).
    pub p25: f64,
    /// 50th percentile (median).
    pub median: f64,
    /// 75th percentile (upper quartile).
    pub p75: f64,
    /// 95th percentile.
    pub p95: f64,
    /// Maximum (observed) or `mean + 2·SD` (normative plotting range).
    pub max: f64,
}

impl Summary {
    /// Empirical summary of raw samples. `None` if `samples` is empty.
    /// Percentiles use linear interpolation between order statistics
    /// (the NumPy / "type 7" convention).
    pub fn from_samples(samples: &[f64]) -> Option<Summary> {
        if samples.is_empty() {
            return None;
        }
        let n = samples.len();
        let mut sorted: Vec<f64> = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let sd = if n > 1 {
            let var = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
            var.sqrt()
        } else {
            0.0
        };
        Some(Summary {
            n,
            mean,
            sd,
            min: sorted[0],
            p5: percentile_sorted(&sorted, 0.05),
            p25: percentile_sorted(&sorted, 0.25),
            median: percentile_sorted(&sorted, 0.50),
            p75: percentile_sorted(&sorted, 0.75),
            p95: percentile_sorted(&sorted, 0.95),
            max: sorted[n - 1],
        })
    }

    /// Builds a summary from a published mean and SD, modelling the
    /// distribution as normal. Percentiles use the standard-normal
    /// quantile multipliers (z(0.05)=±1.6449, z(0.25)=±0.6745); `min` /
    /// `max` are set to ±2 SD as a plotting range, not a hard bound.
    /// This is how a `summary_normative_range` distribution — which has
    /// no raw samples — yields a band with the same shape as an observed
    /// one.
    pub fn from_mean_sd(mean: f64, sd: f64, n: usize) -> Summary {
        Summary {
            n,
            mean,
            sd,
            min: mean - 2.0 * sd,
            p5: mean - 1.6449 * sd,
            p25: mean - 0.6745 * sd,
            median: mean,
            p75: mean + 0.6745 * sd,
            p95: mean + 1.6449 * sd,
            max: mean + 2.0 * sd,
        }
    }
}

/// Equal-width histogram of a 1-D measure (D10): `counts[i]` is the
/// number of samples in `[edges[i], edges[i+1])` (the final bin is
/// closed on the right). `edges.len() == counts.len() + 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct Histogram {
    /// Bin boundaries, ascending; `edges.len() == counts.len() + 1`.
    pub edges: Vec<f64>,
    /// Per-bin sample counts.
    pub counts: Vec<u64>,
}

impl Histogram {
    /// Bins `samples` into `bins` equal-width buckets spanning
    /// `[min, max]`. `None` if `samples` is empty or `bins == 0`. When
    /// every sample is identical, returns a single unit-width bin holding
    /// them all (avoids a zero-width range).
    pub fn from_samples(samples: &[f64], bins: usize) -> Option<Histogram> {
        if samples.is_empty() || bins == 0 {
            return None;
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &x in samples {
            lo = lo.min(x);
            hi = hi.max(x);
        }
        if (hi - lo).abs() < f64::EPSILON {
            // Degenerate: all equal. One bin centred on the value.
            return Some(Histogram {
                edges: vec![lo - 0.5, lo + 0.5],
                counts: vec![samples.len() as u64],
            });
        }
        let width = (hi - lo) / bins as f64;
        let edges: Vec<f64> = (0..=bins).map(|i| lo + i as f64 * width).collect();
        let mut counts = vec![0u64; bins];
        for &x in samples {
            // Clamp into the last bin so x == hi (which would map to
            // index `bins`) lands in `bins - 1`.
            let idx = (((x - lo) / width) as usize).min(bins - 1);
            counts[idx] += 1;
        }
        Some(Histogram { edges, counts })
    }
}

/// Linear-interpolated percentile of an already-sorted slice. `q` in
/// `[0, 1]`. Matches NumPy's default ("type 7") quantile.
fn percentile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

impl RefDist {
    /// Absolute path to the manifest's declared data file, if any.
    pub fn data_path(&self) -> Option<PathBuf> {
        self.manifest
            .schema
            .data_file
            .as_ref()
            .map(|f| self.dir.join(f))
    }

    /// Reads a numeric column from the data file as `f64`, keeping only
    /// rows where every `(column, value)` in `filters` matches (string
    /// columns, ASCII-case-insensitive). Integer and 32/64-bit float
    /// columns are all widened to `f64`. The natural way to pull "F1 for
    /// the /iy/ rows" or "f0 where sex = m".
    pub fn column_f64(&self, name: &str, filters: &[(&str, &str)]) -> Result<Vec<f64>> {
        let path = self
            .data_path()
            .ok_or_else(|| EngineError::RefDist("distribution has no data file".into()))?;
        let file = fs::File::open(&path)
            .map_err(|e| EngineError::RefDist(format!("cannot open {}: {e}", path.display())))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(map_parquet_err)?
            .build()
            .map_err(map_parquet_err)?;

        let mut out = Vec::new();
        for batch in reader {
            let batch = batch.map_err(map_arrow_err)?;
            let schema = batch.schema();
            let col_idx = schema.index_of(name).map_err(|_| {
                EngineError::RefDist(format!("no column {name:?} in {}", path.display()))
            })?;
            let values = batch.column(col_idx);
            // Resolve each filter column once per batch, materialised to
            // strings so Utf8 and LargeUtf8 (polars' default) are handled
            // uniformly.
            let filter_cols: Vec<(Vec<Option<String>>, &str)> = filters
                .iter()
                .map(|(fname, fval)| {
                    let idx = schema
                        .index_of(fname)
                        .map_err(|_| EngineError::RefDist(format!("no filter column {fname:?}")))?;
                    let vals = string_column(batch.column(idx).as_ref()).ok_or_else(|| {
                        EngineError::RefDist(format!("filter column {fname:?} is not a string"))
                    })?;
                    Ok((vals, *fval))
                })
                .collect::<Result<_>>()?;

            for row in 0..batch.num_rows() {
                let keep = filter_cols.iter().all(|(vals, want)| {
                    vals.get(row)
                        .and_then(|v| v.as_deref())
                        .is_some_and(|s| s.eq_ignore_ascii_case(want))
                });
                if keep {
                    out.push(numeric_at(values.as_ref(), row).ok_or_else(|| {
                        EngineError::RefDist(format!("column {name:?} is not numeric"))
                    })?);
                }
            }
        }
        Ok(out)
    }

    /// Empirical/normative summary of a 1-D parameter (D10 band overlays
    /// and readouts). For an observed distribution this is the empirical
    /// [`Summary`] of the raw column; for a `summary_normative_range` it
    /// is built from the `mean` / `sd` rows of the `stat` column via
    /// [`Summary::from_mean_sd`] (a normal model). `filters` subsets by
    /// subgroup (e.g. `sex = m`). Errors if there are no matching values.
    pub fn summary(&self, parameter: &str, filters: &[(&str, &str)]) -> Result<Summary> {
        match self.manifest.measure.kind {
            MeasureKind::SummaryNormativeRange => {
                let mut mean_filters = filters.to_vec();
                mean_filters.push(("stat", "mean"));
                let means = self.column_f64(parameter, &mean_filters)?;
                if means.is_empty() {
                    return Err(EngineError::RefDist(format!(
                        "normative distribution {:?} has no `mean` row for {parameter:?}",
                        self.manifest.id
                    )));
                }
                let mean = means.iter().sum::<f64>() / means.len() as f64;

                let mut sd_filters = filters.to_vec();
                sd_filters.push(("stat", "sd"));
                let sds = self.column_f64(parameter, &sd_filters)?;
                let sd = if sds.is_empty() {
                    0.0
                } else {
                    sds.iter().sum::<f64>() / sds.len() as f64
                };
                let n = self.manifest.population.n_speakers.unwrap_or(0) as usize;
                Ok(Summary::from_mean_sd(mean, sd, n))
            }
            _ => {
                let samples = self.column_f64(parameter, filters)?;
                Summary::from_samples(&samples).ok_or_else(|| {
                    EngineError::RefDist(format!(
                        "no values for {parameter:?} in {:?}",
                        self.manifest.id
                    ))
                })
            }
        }
    }

    /// Equal-width [`Histogram`] of a 1-D parameter's raw samples (D10
    /// histogram panel). Only meaningful for distributions that ship raw
    /// samples; errors on a `summary_normative_range` (which has none).
    pub fn histogram(
        &self,
        parameter: &str,
        bins: usize,
        filters: &[(&str, &str)],
    ) -> Result<Histogram> {
        if self.manifest.measure.kind == MeasureKind::SummaryNormativeRange {
            return Err(EngineError::RefDist(
                "cannot histogram a summary_normative_range distribution (no raw samples)".into(),
            ));
        }
        let samples = self.column_f64(parameter, filters)?;
        Histogram::from_samples(&samples, bins).ok_or_else(|| {
            EngineError::RefDist(format!(
                "no samples for {parameter:?} in {:?}",
                self.manifest.id
            ))
        })
    }

    /// Reads two numeric columns as aligned `(x, y)` pairs (D10 vowel
    /// space: `x_param = "F1"`, `y_param = "F2"`). Both columns are read
    /// over the same filtered rows, so the pairs stay row-aligned.
    pub fn points2d(
        &self,
        x_param: &str,
        y_param: &str,
        filters: &[(&str, &str)],
    ) -> Result<Vec<(f64, f64)>> {
        let xs = self.column_f64(x_param, filters)?;
        let ys = self.column_f64(y_param, filters)?;
        if xs.len() != ys.len() {
            return Err(EngineError::RefDist(format!(
                "column length mismatch reading 2-D points ({} vs {})",
                xs.len(),
                ys.len()
            )));
        }
        Ok(xs.into_iter().zip(ys).collect())
    }
}

/// Reads the value at `row` of a numeric Arrow array as `f64`, widening
/// from the integer/float types polars writes. `None` for non-numeric
/// arrays or null cells.
fn numeric_at(arr: &dyn Array, row: usize) -> Option<f64> {
    if arr.is_null(row) {
        return None;
    }
    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
        Some(a.value(row))
    } else if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
        Some(a.value(row) as f64)
    } else if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
        Some(a.value(row) as f64)
    } else {
        arr.as_any()
            .downcast_ref::<Int32Array>()
            .map(|a| a.value(row) as f64)
    }
}

/// Materialises a string Arrow column to `Vec<Option<String>>`, handling
/// both `Utf8` (`StringArray`) and `LargeUtf8` (`LargeStringArray`, which
/// is what polars writes). `None` if the array is not a string type.
fn string_column(arr: &dyn Array) -> Option<Vec<Option<String>>> {
    if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
        Some(
            (0..a.len())
                .map(|i| (!a.is_null(i)).then(|| a.value(i).to_string()))
                .collect(),
        )
    } else {
        arr.as_any().downcast_ref::<LargeStringArray>().map(|a| {
            (0..a.len())
                .map(|i| (!a.is_null(i)).then(|| a.value(i).to_string()))
                .collect()
        })
    }
}

fn map_parquet_err(e: parquet::errors::ParquetError) -> EngineError {
    EngineError::RefDist(format!("parquet error: {e}"))
}

fn map_arrow_err(e: arrow::error::ArrowError) -> EngineError {
    EngineError::RefDist(format!("arrow error: {e}"))
}

/// Parses a `refdist.toml` manifest from a TOML string.
pub fn parse_manifest(toml_str: &str) -> Result<RefdistManifest> {
    toml::from_str(toml_str).map_err(|e| EngineError::RefDist(format!("invalid refdist.toml: {e}")))
}

/// Loads the `refdist.toml` manifest from a distribution directory.
pub fn load_manifest(dir: impl AsRef<Path>) -> Result<RefdistManifest> {
    let path = dir.as_ref().join("refdist.toml");
    let text = fs::read_to_string(&path)
        .map_err(|e| EngineError::RefDist(format!("cannot read {}: {e}", path.display())))?;
    parse_manifest(&text)
}

/// Scaffolds a distribution directory from `manifest` (C9 in-app
/// publishing): writes `refdist.toml`, a `provenance.md` carrying
/// `provenance`, and a `LICENSE` stub keyed off the manifest's SPDX id.
/// The **data file** named by `manifest.schema.data_file` is the caller's
/// responsibility (e.g. `polars` `write_parquet` from the analysis
/// result). Creates `dir` if needed and returns the resolved [`RefDist`].
///
/// The written manifest round-trips through [`parse_manifest`], so a
/// scaffolded distribution is immediately resolvable and passes the
/// registry validator (modulo the maintainer replacing the LICENSE stub
/// with the full license text before submission).
pub fn scaffold(
    dir: impl AsRef<Path>,
    manifest: &RefdistManifest,
    provenance: &str,
) -> Result<RefDist> {
    let dir = dir.as_ref().to_path_buf();
    fs::create_dir_all(&dir)?;
    let toml_str = toml::to_string(manifest)
        .map_err(|e| EngineError::RefDist(format!("cannot serialize manifest: {e}")))?;
    fs::write(dir.join("refdist.toml"), toml_str)?;

    let provenance = if provenance.trim().is_empty() {
        "# Provenance\n\nTODO: describe the source corpus, sampling, and method.\n".to_string()
    } else {
        format!("{}\n", provenance.trim_end())
    };
    fs::write(dir.join("provenance.md"), provenance)?;

    let spdx = manifest
        .license
        .as_deref()
        .unwrap_or("LicenseRef-UNSPECIFIED");
    let license = format!(
        "SPDX-License-Identifier: {spdx}\n\n\
         TODO: replace this stub with the full text of the {spdx} license \
         before submitting to the registry.\n"
    );
    fs::write(dir.join("LICENSE"), license)?;

    Ok(RefDist {
        manifest: manifest.clone(),
        dir,
    })
}

/// A faceted query over a [`RefdistStore`]. Every `Some` / non-empty field
/// is a constraint the manifest must satisfy; `None` / empty matches
/// anything. String matches are case-insensitive.
#[derive(Debug, Clone, Default)]
pub struct QuerySpec {
    /// Require this parameter in `measure.parameters` (e.g. `"F1"`).
    pub parameter: Option<String>,
    /// Require this `population.language`.
    pub language: Option<String>,
    /// Require this `population.variety`.
    pub variety: Option<String>,
    /// Require this sex in `population.sex`.
    pub sex: Option<String>,
    /// Require this age band in `population.age_band`.
    pub age_band: Option<String>,
    /// Require this phone in `measure.phones`.
    pub phone: Option<String>,
    /// Require this `measure.kind`.
    pub kind: Option<MeasureKind>,
}

fn eq_ci(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

impl QuerySpec {
    /// Does `m` satisfy every constraint in this spec?
    fn matches(&self, m: &RefdistManifest) -> bool {
        if let Some(p) = &self.parameter {
            if !m.measure.parameters.iter().any(|x| eq_ci(x, p)) {
                return false;
            }
        }
        if let Some(l) = &self.language {
            if m.population.language.as_deref().map(|x| eq_ci(x, l)) != Some(true) {
                return false;
            }
        }
        if let Some(v) = &self.variety {
            if m.population.variety.as_deref().map(|x| eq_ci(x, v)) != Some(true) {
                return false;
            }
        }
        if let Some(s) = &self.sex {
            if !m.population.sex.iter().any(|x| eq_ci(x, s)) {
                return false;
            }
        }
        if let Some(a) = &self.age_band {
            if !m.population.age_band.iter().any(|x| eq_ci(x, a)) {
                return false;
            }
        }
        if let Some(ph) = &self.phone {
            if !m.measure.phones.iter().any(|x| eq_ci(x, ph)) {
                return false;
            }
        }
        if let Some(k) = self.kind {
            if m.measure.kind != k {
                return false;
            }
        }
        true
    }
}

/// A user-level cache of reference distributions: a directory whose
/// immediate subdirectories each hold one distribution (a `refdist.toml`
/// plus its data file). The resolver scans these — it does not fetch from
/// a hosted registry (that's C8).
#[derive(Debug, Clone)]
pub struct RefdistStore {
    root: PathBuf,
}

impl RefdistStore {
    /// A store rooted at an explicit directory (used in tests and for
    /// additional/!default registries).
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    /// The default per-user store at the OS data directory —
    /// `~/.local/share/sadda/refdist/` on Linux, the platform equivalent
    /// elsewhere. The directory is created if missing.
    pub fn user_default() -> Result<Self> {
        let dirs = directories::ProjectDirs::from("", "", "sadda")
            .ok_or_else(|| EngineError::RefDist("cannot determine user data directory".into()))?;
        let root = dirs.data_dir().join("refdist");
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// The store's root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// All distributions in the store, skipping subdirectories without a
    /// readable, valid `refdist.toml`.
    pub fn list(&self) -> Vec<RefDist> {
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(&self.root) else {
            return out;
        };
        for entry in entries.flatten() {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            if let Ok(manifest) = load_manifest(&dir) {
                out.push(RefDist { manifest, dir });
            }
        }
        out.sort_by(|a, b| {
            a.manifest
                .id
                .cmp(&b.manifest.id)
                .then(a.manifest.version.cmp(&b.manifest.version))
        });
        out
    }

    /// Distributions matching `spec`, sorted by id then version.
    pub fn query(&self, spec: &QuerySpec) -> Vec<RefDist> {
        self.list()
            .into_iter()
            .filter(|d| spec.matches(&d.manifest))
            .collect()
    }

    /// The distribution with this `id` and `version`, if present.
    pub fn get(&self, id: &str, version: &str) -> Option<RefDist> {
        self.list()
            .into_iter()
            .find(|d| d.manifest.id == id && d.manifest.version == version)
    }

    /// Installs the distribution at `src_dir` (a directory holding a
    /// `refdist.toml` + its data file) into this store by copying it under
    /// `<id>__<version>/`. This is how the bundled starter set seeds the
    /// cache and how a fetched tarball lands once unpacked. Returns the
    /// installed [`RefDist`]. Errors if `src_dir` has no valid manifest.
    pub fn install_from_dir(&self, src_dir: impl AsRef<Path>) -> Result<RefDist> {
        let src = src_dir.as_ref();
        let manifest = load_manifest(src)?;
        let dest = self
            .root
            .join(format!("{}__{}", manifest.id, manifest.version));
        fs::create_dir_all(&dest)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                fs::copy(entry.path(), dest.join(entry.file_name()))?;
            }
        }
        Ok(RefDist {
            manifest,
            dir: dest,
        })
    }
}

/// One entry in a registry's published `index.json` — the discovery
/// metadata for a distribution available from a hosted registry, without
/// its data file. The engine reads the index to list what's available;
/// the data is fetched/installed separately (C8+).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegistryEntry {
    /// Distribution id.
    pub id: String,
    /// Distribution version.
    pub version: String,
    /// Registry tier (2 = curated, 3 = community).
    #[serde(default)]
    pub tier: u8,
    /// Human-readable title.
    #[serde(default)]
    pub title: String,
    /// Measure kind.
    #[serde(default)]
    pub kind: MeasureKind,
    /// Measured parameters.
    #[serde(default)]
    pub parameters: Vec<String>,
    /// Language (ISO 639-3).
    #[serde(default)]
    pub language: Option<String>,
    /// SPDX license id.
    #[serde(default)]
    pub license: Option<String>,
    /// Path to the distribution within the registry (e.g. `tier2/<dir>`),
    /// resolved against the registry's base URL when fetching.
    #[serde(default)]
    pub path: Option<String>,
    /// Whether this version has been yanked (still resolvable for pins,
    /// surfaced with a warning).
    #[serde(default)]
    pub yanked: bool,
}

/// A registry's published index — the GitHub-Pages JSON the engine reads
/// to discover what a hosted registry offers (C8).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegistryIndex {
    /// Index format version.
    #[serde(default)]
    pub schema_version: u32,
    /// The distributions the registry publishes.
    #[serde(default)]
    pub entries: Vec<RegistryEntry>,
}

/// Parses a registry `index.json`.
pub fn parse_index(json: &str) -> Result<RegistryIndex> {
    serde_json::from_str(json)
        .map_err(|e| EngineError::RefDist(format!("invalid registry index.json: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_toml(id: &str, version: &str) -> String {
        format!(
            r#"
id = "{id}"
version = "{version}"
title = "Test distribution {id}"

[citation]
authors = ["Doe, J."]
year = 2025

[population]
language = "eng"
variety = "AmE"
sex = ["m", "f"]
age_band = ["adult"]
n_speakers = 42

[measure]
kind = "observed_distribution"
parameters = ["F1", "F2"]
units = "Hz"
phones = ["iy", "ae"]

[privacy]
shareability = "raw_samples"
min_n_per_subgroup = 5

[schema]
data_file = "data.parquet"
shape = "long"
columns = ["speaker_id", "phone", "F1", "F2"]
"#
        )
    }

    fn write_dist(store_root: &Path, dirname: &str, toml: &str) {
        let dir = store_root.join(dirname);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("refdist.toml"), toml).unwrap();
        fs::write(
            dir.join("data.parquet"),
            b"not really parquet, just a marker",
        )
        .unwrap();
    }

    fn temp_store() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "sadda_refdist_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn parses_full_manifest() {
        let m = parse_manifest(&manifest_toml("hillenbrand-1995-amE-vowels", "1.0.0")).unwrap();
        assert_eq!(m.id, "hillenbrand-1995-amE-vowels");
        assert_eq!(m.version, "1.0.0");
        assert_eq!(m.measure.kind, MeasureKind::ObservedDistribution);
        assert_eq!(m.measure.parameters, ["F1", "F2"]);
        assert_eq!(m.population.language.as_deref(), Some("eng"));
        assert_eq!(m.population.sex, ["m", "f"]);
        assert_eq!(m.privacy.min_n_per_subgroup, Some(5));
        assert_eq!(m.schema.data_file.as_deref(), Some("data.parquet"));
    }

    #[test]
    fn minimal_manifest_defaults() {
        let m = parse_manifest("id = \"x\"\nversion = \"0.1.0\"\n").unwrap();
        assert_eq!(m.measure.kind, MeasureKind::ObservedDistribution);
        assert!(m.measure.parameters.is_empty());
        assert!(m.population.language.is_none());
    }

    #[test]
    fn invalid_manifest_errors() {
        // Missing the required `version` field.
        assert!(parse_manifest("id = \"x\"\n").is_err());
    }

    #[test]
    fn store_lists_and_resolves() {
        let root = temp_store();
        write_dist(&root, "a", &manifest_toml("dist-a", "1.0.0"));
        write_dist(&root, "b", &manifest_toml("dist-b", "2.1.0"));
        fs::create_dir_all(root.join("not-a-dist")).unwrap(); // no refdist.toml

        let store = RefdistStore::new(&root);
        let all = store.list();
        assert_eq!(all.len(), 2, "skips the dir without a manifest");
        assert_eq!(all[0].manifest.id, "dist-a");

        let got = store.get("dist-b", "2.1.0").unwrap();
        assert_eq!(got.manifest.version, "2.1.0");
        assert_eq!(
            got.data_path().unwrap(),
            root.join("b").join("data.parquet")
        );
        assert!(store.get("dist-b", "9.9.9").is_none());

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn query_facets() {
        let root = temp_store();
        write_dist(&root, "a", &manifest_toml("dist-a", "1.0.0"));
        let store = RefdistStore::new(&root);

        let by_param = store.query(&QuerySpec {
            parameter: Some("f1".into()), // case-insensitive
            ..Default::default()
        });
        assert_eq!(by_param.len(), 1);

        let by_sex = store.query(&QuerySpec {
            sex: Some("f".into()),
            language: Some("eng".into()),
            ..Default::default()
        });
        assert_eq!(by_sex.len(), 1);

        let no_match = store.query(&QuerySpec {
            parameter: Some("F3".into()),
            ..Default::default()
        });
        assert!(no_match.is_empty());

        let wrong_kind = store.query(&QuerySpec {
            kind: Some(MeasureKind::TargetZone),
            ..Default::default()
        });
        assert!(wrong_kind.is_empty());

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn install_from_dir_copies_into_store() {
        // A "bundled" source dir, installed into an (initially empty) store.
        let src_root = temp_store();
        write_dist(
            &src_root,
            "bundled-src",
            &manifest_toml("starter-vowels", "1.0.0"),
        );
        let store_root = temp_store();
        let store = RefdistStore::new(&store_root);
        assert!(store.list().is_empty());

        let installed = store
            .install_from_dir(src_root.join("bundled-src"))
            .unwrap();
        assert_eq!(installed.manifest.id, "starter-vowels");
        // Now resolvable from the store, with its data file copied over.
        let got = store.get("starter-vowels", "1.0.0").unwrap();
        assert!(got.data_path().unwrap().is_file());
        assert_eq!(store.list().len(), 1);

        fs::remove_dir_all(&src_root).ok();
        fs::remove_dir_all(&store_root).ok();
    }

    #[test]
    fn scaffold_round_trips_through_parser() {
        let store = temp_store();
        let manifest = RefdistManifest {
            id: "my-vowels".into(),
            version: "0.1.0".into(),
            title: "My vowels".into(),
            doi: None,
            license: Some("CC-BY-4.0".into()),
            citation: Citation {
                authors: vec!["Me".into()],
                year: Some(2026),
                ..Default::default()
            },
            population: Population {
                language: Some("eng".into()),
                sex: vec!["m".into(), "f".into()],
                ..Default::default()
            },
            measure: Measure {
                kind: MeasureKind::ObservedDistribution,
                parameters: vec!["F1".into(), "F2".into()],
                units: Some("Hz".into()),
                ..Default::default()
            },
            privacy: Privacy {
                shareability: Some("raw_samples".into()),
                min_n_per_subgroup: Some(5),
                community_consent: false,
            },
            schema: Schema {
                data_file: Some("data.parquet".into()),
                shape: Some("long".into()),
                columns: vec!["speaker_id".into(), "F1".into(), "F2".into()],
            },
        };
        let dist_dir = store.join("my-vowels");
        let rd = scaffold(&dist_dir, &manifest, "Synthetic test data.").unwrap();
        assert!(dist_dir.join("refdist.toml").is_file());
        assert!(dist_dir.join("provenance.md").is_file());
        assert!(dist_dir.join("LICENSE").is_file());
        assert_eq!(rd.data_path().unwrap(), dist_dir.join("data.parquet"));

        // The written manifest parses back to the same fields (None
        // Options were skipped, not emitted as nulls).
        let reparsed = load_manifest(&dist_dir).unwrap();
        assert_eq!(reparsed.id, "my-vowels");
        assert_eq!(reparsed.license.as_deref(), Some("CC-BY-4.0"));
        assert!(reparsed.doi.is_none());
        assert_eq!(reparsed.measure.parameters, ["F1", "F2"]);
        assert_eq!(reparsed.population.sex, ["m", "f"]);
        assert_eq!(reparsed.privacy.min_n_per_subgroup, Some(5));

        fs::remove_dir_all(&store).ok();
    }

    #[test]
    fn parses_registry_index() {
        let json = r#"
        {
          "schema_version": 1,
          "entries": [
            {
              "id": "hillenbrand-1995-amE-vowels", "version": "1.0.0", "tier": 2,
              "title": "AmE vowels", "kind": "observed_distribution",
              "parameters": ["F1", "F2"], "language": "eng",
              "license": "CC0-1.0", "path": "tier2/hillenbrand-1995", "yanked": false
            },
            { "id": "f0-norms", "version": "0.1.0", "tier": 3 }
          ]
        }"#;
        let index = parse_index(json).unwrap();
        assert_eq!(index.schema_version, 1);
        assert_eq!(index.entries.len(), 2);
        assert_eq!(index.entries[0].tier, 2);
        assert_eq!(index.entries[0].kind, MeasureKind::ObservedDistribution);
        assert_eq!(index.entries[0].license.as_deref(), Some("CC0-1.0"));
        // Defaulted fields on the sparse entry.
        assert_eq!(index.entries[1].tier, 3);
        assert!(!index.entries[1].yanked);
        assert!(index.entries[1].parameters.is_empty());
    }

    // ----- D10: Summary / Histogram + data-file readers -----------------

    #[test]
    fn summary_from_samples_matches_numpy_type7() {
        let s = Summary::from_samples(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(s.n, 5);
        assert!((s.mean - 3.0).abs() < 1e-9);
        assert!((s.median - 3.0).abs() < 1e-9);
        assert!((s.min - 1.0).abs() < 1e-9);
        assert!((s.max - 5.0).abs() < 1e-9);
        assert!((s.p25 - 2.0).abs() < 1e-9);
        assert!((s.p75 - 4.0).abs() < 1e-9);
        // Sample SD (n-1) of 1..5 = sqrt(2.5).
        assert!((s.sd - 2.5_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn summary_from_samples_empty_is_none() {
        assert!(Summary::from_samples(&[]).is_none());
    }

    #[test]
    fn summary_from_mean_sd_is_normal_band() {
        let s = Summary::from_mean_sd(100.0, 10.0, 200);
        assert_eq!(s.n, 200);
        assert!((s.mean - 100.0).abs() < 1e-9);
        assert!((s.median - 100.0).abs() < 1e-9);
        // z(0.95) ≈ 1.6449 ⇒ p95 ≈ 116.449; band is symmetric.
        assert!((s.p95 - 116.449).abs() < 1e-2);
        assert!((s.p5 - 83.551).abs() < 1e-2);
        assert!((s.p95 - s.mean) - (s.mean - s.p5) < 1e-9);
    }

    #[test]
    fn histogram_buckets_and_edges() {
        let h = Histogram::from_samples(&[0.0, 1.0, 2.0, 3.0, 4.0], 4).unwrap();
        assert_eq!(h.edges.len(), 5);
        assert_eq!(h.counts.len(), 4);
        // All five samples accounted for; 4.0 (== max) folds into last bin.
        assert_eq!(h.counts.iter().sum::<u64>(), 5);
        assert_eq!(*h.counts.last().unwrap(), 2); // 3.0 and 4.0
    }

    #[test]
    fn histogram_all_equal_is_single_bin() {
        let h = Histogram::from_samples(&[5.0, 5.0, 5.0], 4).unwrap();
        assert_eq!(h.counts, vec![3]);
        assert_eq!(h.edges.len(), 2);
    }

    // Writes a real Parquet file with the given columns into `dir`, so the
    // `RefDist` data-readers can be exercised against actual bytes (the
    // older `write_dist` helper writes only a marker).
    fn write_parquet(dir: &Path, columns: Vec<(&str, ParquetCol)>) {
        use arrow::array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let fields: Vec<Field> = columns
            .iter()
            .map(|(name, col)| {
                let dt = match col {
                    ParquetCol::Int(_) => DataType::Int64,
                    ParquetCol::F64(_) => DataType::Float64,
                    ParquetCol::Str(_) => DataType::Utf8,
                };
                Field::new(*name, dt, false)
            })
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let arrays: Vec<ArrayRef> = columns
            .into_iter()
            .map(|(_, col)| match col {
                ParquetCol::Int(v) => Arc::new(Int64Array::from(v)) as ArrayRef,
                ParquetCol::F64(v) => Arc::new(Float64Array::from(v)) as ArrayRef,
                ParquetCol::Str(v) => Arc::new(StringArray::from(v)) as ArrayRef,
            })
            .collect();
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        let file = fs::File::create(dir.join("data.parquet")).unwrap();
        let mut w = ArrowWriter::try_new(file, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
    }

    enum ParquetCol {
        Int(Vec<i64>),
        F64(Vec<f64>),
        Str(Vec<&'static str>),
    }

    fn observed_dist() -> RefDist {
        let dir = temp_store().join("amE-vowels");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("refdist.toml"),
            manifest_toml("amE-vowels", "1.0.0"),
        )
        .unwrap();
        write_parquet(
            &dir,
            vec![
                ("speaker_id", ParquetCol::Int(vec![1, 1, 2, 2, 3, 3])),
                (
                    "phone",
                    ParquetCol::Str(vec!["iy", "ae", "iy", "ae", "iy", "ae"]),
                ),
                (
                    "F1",
                    ParquetCol::F64(vec![300.0, 700.0, 310.0, 710.0, 290.0, 690.0]),
                ),
                (
                    "F2",
                    ParquetCol::F64(vec![2300.0, 1700.0, 2320.0, 1710.0, 2280.0, 1690.0]),
                ),
            ],
        );
        let manifest = load_manifest(&dir).unwrap();
        RefDist { manifest, dir }
    }

    fn normative_dist() -> RefDist {
        let dir = temp_store().join("f0-norms");
        fs::create_dir_all(&dir).unwrap();
        let toml = r#"
id = "f0-norms"
version = "0.1.0"
title = "Test f0 norms"

[citation]
authors = ["Doe, J."]
year = 2025

[population]
language = "eng"
sex = ["m", "f"]
n_speakers = 200

[measure]
kind = "summary_normative_range"
parameters = ["f0"]
units = "Hz"

[privacy]
shareability = "summary_only"

[schema]
data_file = "data.parquet"
shape = "long"
columns = ["sex", "stat", "f0"]
"#;
        fs::write(dir.join("refdist.toml"), toml).unwrap();
        write_parquet(
            &dir,
            vec![
                ("sex", ParquetCol::Str(vec!["m", "m", "f", "f"])),
                ("stat", ParquetCol::Str(vec!["mean", "sd", "mean", "sd"])),
                ("f0", ParquetCol::F64(vec![120.0, 18.0, 210.0, 22.0])),
            ],
        );
        let manifest = load_manifest(&dir).unwrap();
        RefDist { manifest, dir }
    }

    #[test]
    fn column_f64_reads_and_filters() {
        let rd = observed_dist();
        let all = rd.column_f64("F1", &[]).unwrap();
        assert_eq!(all.len(), 6);
        let iy = rd.column_f64("F1", &[("phone", "iy")]).unwrap();
        assert_eq!(iy, vec![300.0, 310.0, 290.0]);
        // Filter match is ASCII-case-insensitive.
        let iy_upper = rd.column_f64("F1", &[("phone", "IY")]).unwrap();
        assert_eq!(iy_upper, vec![300.0, 310.0, 290.0]);
    }

    #[test]
    fn column_f64_widens_integer_column() {
        let rd = observed_dist();
        let ids = rd.column_f64("speaker_id", &[]).unwrap();
        assert_eq!(ids, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn summary_observed_uses_raw_samples() {
        let rd = observed_dist();
        let s = rd.summary("F1", &[("phone", "iy")]).unwrap();
        assert_eq!(s.n, 3);
        assert!((s.mean - 300.0).abs() < 1e-9);
        assert!((s.median - 300.0).abs() < 1e-9);
    }

    #[test]
    fn summary_normative_builds_band_from_mean_sd() {
        let rd = normative_dist();
        // Subgroup-filtered: the male band.
        let m = rd.summary("f0", &[("sex", "m")]).unwrap();
        assert!((m.mean - 120.0).abs() < 1e-9);
        assert!((m.sd - 18.0).abs() < 1e-9);
        assert!((m.median - 120.0).abs() < 1e-9);
        assert_eq!(m.n, 200);
        // No filter pools the two means (120, 210) → 165.
        let pooled = rd.summary("f0", &[]).unwrap();
        assert!((pooled.mean - 165.0).abs() < 1e-9);
    }

    #[test]
    fn histogram_rejects_summary_normative() {
        let rd = normative_dist();
        assert!(rd.histogram("f0", 10, &[]).is_err());
    }

    #[test]
    fn points2d_pairs_aligned_columns() {
        let rd = observed_dist();
        let ae = rd.points2d("F1", "F2", &[("phone", "ae")]).unwrap();
        assert_eq!(ae, vec![(700.0, 1700.0), (710.0, 1710.0), (690.0, 1690.0)]);
    }
}
