//! PyO3 bindings for C7 reference distributions — the consumption side.
//! Thin wrappers over `sadda_engine::refdist`: resolve distributions from
//! the user-level cache, query by population/measure facets, and read the
//! manifest. The `sadda/refdist/__init__.py` re-exports these under the
//! user-facing `sadda.refdist.*` path and adds a Polars `.data()` helper.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use sadda_engine::{MeasureKind, QuerySpec, RefDist, RefdistStore};

use crate::engine_err_to_py;

/// One resolved reference distribution (its parsed manifest + on-disk
/// location).
#[pyclass(name = "RefDist")]
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

    fn __repr__(&self) -> String {
        format!(
            "RefDist(id={:?}, version={:?}, kind={:?})",
            self.inner.manifest.id,
            self.inner.manifest.version,
            kind_str(self.inner.manifest.measure.kind)
        )
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
