//! PyO3 bindings for C7 reference distributions — the consumption side.
//! Thin wrappers over `sadda_engine::refdist`: resolve distributions from
//! the user-level cache, query by population/measure facets, and read the
//! manifest. The `sadda/refdist/__init__.py` re-exports these under the
//! user-facing `sadda.refdist.*` path and adds a Polars `.data()` helper.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sadda_engine::{
    Histogram, Measure, MeasureKind, Population, Privacy, QuerySpec, RefDist, RefdistManifest,
    RefdistStore, Summary,
};
use sadda_engine::{RefdistCitation, RefdistSchema};

use crate::engine_err_to_py;

/// Converts a Python `filter={"col": "value"}` mapping into the
/// `(column, value)` pairs the engine readers take. Order is irrelevant
/// (filters are AND-combined), so the HashMap's arbitrary order is fine.
fn filter_pairs(filter: Option<HashMap<String, String>>) -> Vec<(String, String)> {
    filter.map(|m| m.into_iter().collect()).unwrap_or_default()
}

/// One resolved reference distribution (its parsed manifest + on-disk
/// location).
#[pyclass(module = "sadda._native.refdist", name = "RefDist")]
pub(crate) struct PyRefDist {
    inner: RefDist,
}

fn kind_str(k: MeasureKind) -> &'static str {
    match k {
        MeasureKind::ObservedDistribution => "observed_distribution",
        MeasureKind::SummaryNormativeRange => "summary_normative_range",
        MeasureKind::TargetZone => "target_zone",
    }
}

fn parse_kind(s: &str) -> PyResult<MeasureKind> {
    match s {
        "observed_distribution" => Ok(MeasureKind::ObservedDistribution),
        "summary_normative_range" => Ok(MeasureKind::SummaryNormativeRange),
        "target_zone" => Ok(MeasureKind::TargetZone),
        other => Err(PyValueError::new_err(format!(
            "unknown measure kind {other:?}; expected observed_distribution | \
             summary_normative_range | target_zone"
        ))),
    }
}

#[pymethods]
impl PyRefDist {
    /// Stable distribution id.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.manifest.id
    }
    /// Semantic version.
    #[getter]
    fn version(&self) -> &str {
        &self.inner.manifest.version
    }
    /// Human-readable title.
    #[getter]
    fn title(&self) -> &str {
        &self.inner.manifest.title
    }
    /// DOI, if any.
    #[getter]
    fn doi(&self) -> Option<String> {
        self.inner.manifest.doi.clone()
    }
    /// Measure kind: `observed_distribution` | `summary_normative_range`
    /// | `target_zone`.
    #[getter]
    fn kind(&self) -> &'static str {
        kind_str(self.inner.manifest.measure.kind)
    }
    /// Measured parameters (e.g. `["F1", "F2"]`).
    #[getter]
    fn parameters(&self) -> Vec<String> {
        self.inner.manifest.measure.parameters.clone()
    }
    /// Parameter units (e.g. `"Hz"`), if declared.
    #[getter]
    fn units(&self) -> Option<String> {
        self.inner.manifest.measure.units.clone()
    }
    /// Phones covered, if applicable.
    #[getter]
    fn phones(&self) -> Vec<String> {
        self.inner.manifest.measure.phones.clone()
    }
    /// ISO 639-3 language code, if declared.
    #[getter]
    fn language(&self) -> Option<String> {
        self.inner.manifest.population.language.clone()
    }
    /// Variety / dialect, if declared.
    #[getter]
    fn variety(&self) -> Option<String> {
        self.inner.manifest.population.variety.clone()
    }
    /// Sexes represented.
    #[getter]
    fn sex(&self) -> Vec<String> {
        self.inner.manifest.population.sex.clone()
    }
    /// Age bands represented.
    #[getter]
    fn age_band(&self) -> Vec<String> {
        self.inner.manifest.population.age_band.clone()
    }
    /// Number of speakers, if declared.
    #[getter]
    fn n_speakers(&self) -> Option<u64> {
        self.inner.manifest.population.n_speakers
    }
    /// Citation authors.
    #[getter]
    fn authors(&self) -> Vec<String> {
        self.inner.manifest.citation.authors.clone()
    }
    /// Publication year, if declared.
    #[getter]
    fn year(&self) -> Option<i32> {
        self.inner.manifest.citation.year
    }
    /// BibTeX entry, if declared.
    #[getter]
    fn bibtex(&self) -> Option<String> {
        self.inner.manifest.citation.bibtex.clone()
    }
    /// Shareability declaration (`raw_samples` | `summary_only`).
    #[getter]
    fn shareability(&self) -> Option<String> {
        self.inner.manifest.privacy.shareability.clone()
    }
    /// Absolute path to the data file, if the manifest declares one.
    fn data_path(&self) -> Option<String> {
        self.inner
            .data_path()
            .map(|p| p.to_string_lossy().into_owned())
    }

    /// Reads a numeric data-file column as a list of floats, keeping only
    /// rows where every `filter` entry matches (string columns,
    /// case-insensitive). E.g. `column("F1", filter={"phone": "iy"})`.
    /// (D10, provisional.)
    #[pyo3(signature = (name, *, filter=None))]
    fn column(&self, name: &str, filter: Option<HashMap<String, String>>) -> PyResult<Vec<f64>> {
        let owned = filter_pairs(filter);
        let refs: Vec<(&str, &str)> = owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        self.inner.column_f64(name, &refs).map_err(engine_err_to_py)
    }

    /// Distribution summary (mean / SD / percentiles) of a 1-D parameter.
    /// Empirical for an observed distribution; a normal model of the
    /// published mean/SD for a `summary_normative_range`. `filter` subsets
    /// by subgroup (e.g. `filter={"sex": "m"}`). (D10, provisional.)
    #[pyo3(signature = (parameter, *, filter=None))]
    fn summary(
        &self,
        parameter: &str,
        filter: Option<HashMap<String, String>>,
    ) -> PyResult<PySummary> {
        let owned = filter_pairs(filter);
        let refs: Vec<(&str, &str)> = owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        Ok(PySummary {
            inner: self
                .inner
                .summary(parameter, &refs)
                .map_err(engine_err_to_py)?,
        })
    }

    /// Equal-width histogram of a parameter's raw samples. Errors on a
    /// `summary_normative_range` (no raw samples). (D10, provisional.)
    #[pyo3(signature = (parameter, *, bins=20, filter=None))]
    fn histogram(
        &self,
        parameter: &str,
        bins: usize,
        filter: Option<HashMap<String, String>>,
    ) -> PyResult<PyHistogram> {
        let owned = filter_pairs(filter);
        let refs: Vec<(&str, &str)> = owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        Ok(PyHistogram {
            inner: self
                .inner
                .histogram(parameter, bins, &refs)
                .map_err(engine_err_to_py)?,
        })
    }

    /// Reads two numeric columns as aligned `(xs, ys)` lists — e.g.
    /// `points2d("F1", "F2", filter={"phone": "iy"})` for a vowel-space
    /// scatter. (D10, provisional.)
    #[pyo3(signature = (x_param, y_param, *, filter=None))]
    fn points2d(
        &self,
        x_param: &str,
        y_param: &str,
        filter: Option<HashMap<String, String>>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let owned = filter_pairs(filter);
        let refs: Vec<(&str, &str)> = owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let pts = self
            .inner
            .points2d(x_param, y_param, &refs)
            .map_err(engine_err_to_py)?;
        let xs = pts.iter().map(|p| p.0).collect();
        let ys = pts.iter().map(|p| p.1).collect();
        Ok((xs, ys))
    }

    fn __repr__(&self) -> String {
        format!(
            "RefDist(id={:?}, version={:?}, kind={:?})",
            self.inner.manifest.id,
            self.inner.manifest.version,
            kind_str(self.inner.manifest.measure.kind)
        )
    }
}

/// Distribution summary of a 1-D measure (D10): mean, SD, and
/// percentiles. Returned by [`PyRefDist::summary`].
#[pyclass(module = "sadda._native.refdist", name = "Summary")]
pub(crate) struct PySummary {
    inner: Summary,
}

#[pymethods]
impl PySummary {
    /// Number of underlying values (raw samples, or declared speakers).
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }
    /// Arithmetic mean.
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean
    }
    /// Standard deviation.
    #[getter]
    fn sd(&self) -> f64 {
        self.inner.sd
    }
    /// Minimum (observed) / `mean − 2·SD` (normative range).
    #[getter]
    fn min(&self) -> f64 {
        self.inner.min
    }
    /// 5th percentile.
    #[getter]
    fn p5(&self) -> f64 {
        self.inner.p5
    }
    /// 25th percentile.
    #[getter]
    fn p25(&self) -> f64 {
        self.inner.p25
    }
    /// Median.
    #[getter]
    fn median(&self) -> f64 {
        self.inner.median
    }
    /// 75th percentile.
    #[getter]
    fn p75(&self) -> f64 {
        self.inner.p75
    }
    /// 95th percentile.
    #[getter]
    fn p95(&self) -> f64 {
        self.inner.p95
    }
    /// Maximum (observed) / `mean + 2·SD` (normative range).
    #[getter]
    fn max(&self) -> f64 {
        self.inner.max
    }

    fn __repr__(&self) -> String {
        format!(
            "Summary(n={}, mean={:.4}, sd={:.4}, median={:.4})",
            self.inner.n, self.inner.mean, self.inner.sd, self.inner.median
        )
    }
}

/// Equal-width histogram of a 1-D measure (D10). Returned by
/// [`PyRefDist::histogram`]; `len(edges) == len(counts) + 1`.
#[pyclass(module = "sadda._native.refdist", name = "Histogram")]
pub(crate) struct PyHistogram {
    inner: Histogram,
}

#[pymethods]
impl PyHistogram {
    /// Bin boundaries, ascending.
    #[getter]
    fn edges(&self) -> Vec<f64> {
        self.inner.edges.clone()
    }
    /// Per-bin sample counts.
    #[getter]
    fn counts(&self) -> Vec<u64> {
        self.inner.counts.clone()
    }

    fn __repr__(&self) -> String {
        format!("Histogram(bins={})", self.inner.counts.len())
    }
}

fn store_for(root: Option<String>) -> PyResult<RefdistStore> {
    match root {
        Some(r) => Ok(RefdistStore::new(r)),
        None => RefdistStore::user_default().map_err(engine_err_to_py),
    }
}

/// Filesystem path of the active reference-distribution store. With
/// `root=None`, the per-user default (`~/.local/share/sadda/refdist/` or
/// the platform equivalent), created if missing.
#[pyfunction]
#[pyo3(signature = (*, root=None))]
pub(crate) fn store_root(root: Option<String>) -> PyResult<String> {
    Ok(store_for(root)?.root().to_string_lossy().into_owned())
}

/// All distributions in the store.
#[pyfunction]
#[pyo3(signature = (*, root=None))]
pub(crate) fn list_all(root: Option<String>) -> PyResult<Vec<PyRefDist>> {
    Ok(store_for(root)?
        .list()
        .into_iter()
        .map(|inner| PyRefDist { inner })
        .collect())
}

/// Distributions matching the given facets (any omitted facet matches
/// anything; string matches are case-insensitive).
#[pyfunction]
#[pyo3(signature = (*, parameter=None, language=None, variety=None, sex=None, age_band=None, phone=None, kind=None, root=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn query(
    parameter: Option<String>,
    language: Option<String>,
    variety: Option<String>,
    sex: Option<String>,
    age_band: Option<String>,
    phone: Option<String>,
    kind: Option<String>,
    root: Option<String>,
) -> PyResult<Vec<PyRefDist>> {
    let spec = QuerySpec {
        parameter,
        language,
        variety,
        sex,
        age_band,
        phone,
        kind: kind.map(|k| parse_kind(&k)).transpose()?,
    };
    Ok(store_for(root)?
        .query(&spec)
        .into_iter()
        .map(|inner| PyRefDist { inner })
        .collect())
}

/// The distribution with this `id` and `version`, or `None`.
#[pyfunction]
#[pyo3(signature = (id, version, *, root=None))]
pub(crate) fn get(id: &str, version: &str, root: Option<String>) -> PyResult<Option<PyRefDist>> {
    Ok(store_for(root)?
        .get(id, version)
        .map(|inner| PyRefDist { inner }))
}

/// Installs a distribution directory (a `refdist.toml` + its data file)
/// into the store by copying it in — how the bundled starter set seeds
/// the user cache. Returns the installed distribution.
#[pyfunction]
#[pyo3(signature = (src_dir, *, root=None))]
pub(crate) fn install(src_dir: &str, root: Option<String>) -> PyResult<PyRefDist> {
    let inner = store_for(root)?
        .install_from_dir(src_dir)
        .map_err(engine_err_to_py)?;
    Ok(PyRefDist { inner })
}

/// Scaffolds a distribution directory (C9 publishing): writes
/// `refdist.toml`, `provenance.md`, and a `LICENSE` stub from the given
/// metadata. The caller writes the data file separately. `columns` should
/// match the data file's columns. Returns the scaffolded distribution.
#[pyfunction]
#[pyo3(signature = (
    dest_dir, *, id, version, kind, columns,
    parameters=None, data_file=None, title=None, doi=None, license=None,
    language=None, variety=None, sex=None, age_band=None, n_speakers=None,
    units=None, phones=None, shareability=None, min_n_per_subgroup=None,
    authors=None, year=None, provenance=None,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn scaffold(
    dest_dir: &str,
    id: String,
    version: String,
    kind: String,
    columns: Vec<String>,
    parameters: Option<Vec<String>>,
    data_file: Option<String>,
    title: Option<String>,
    doi: Option<String>,
    license: Option<String>,
    language: Option<String>,
    variety: Option<String>,
    sex: Option<Vec<String>>,
    age_band: Option<Vec<String>>,
    n_speakers: Option<u64>,
    units: Option<String>,
    phones: Option<Vec<String>>,
    shareability: Option<String>,
    min_n_per_subgroup: Option<u64>,
    authors: Option<Vec<String>>,
    year: Option<i32>,
    provenance: Option<String>,
) -> PyResult<PyRefDist> {
    let manifest = RefdistManifest {
        id,
        version,
        title: title.unwrap_or_default(),
        doi,
        license,
        citation: RefdistCitation {
            authors: authors.unwrap_or_default(),
            year,
            journal: None,
            bibtex: None,
        },
        population: Population {
            language,
            variety,
            sex: sex.unwrap_or_default(),
            age_band: age_band.unwrap_or_default(),
            n_speakers,
            n_tokens: None,
        },
        measure: Measure {
            kind: parse_kind(&kind)?,
            parameters: parameters.unwrap_or_default(),
            units,
            phones: phones.unwrap_or_default(),
            context: None,
            measurement_method: None,
        },
        privacy: Privacy {
            shareability,
            min_n_per_subgroup,
            community_consent: false,
        },
        schema: RefdistSchema {
            data_file: Some(data_file.unwrap_or_else(|| "data.parquet".to_string())),
            shape: Some("long".to_string()),
            columns,
        },
    };
    let inner = sadda_engine::scaffold(dest_dir, &manifest, provenance.as_deref().unwrap_or(""))
        .map_err(engine_err_to_py)?;
    Ok(PyRefDist { inner })
}
