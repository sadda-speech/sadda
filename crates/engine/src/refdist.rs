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
    #[serde(default)]
    pub title: String,
    /// DOI, if one was minted.
    #[serde(default)]
    pub doi: Option<String>,
    /// SPDX license identifier of the distribution data, e.g.
    /// `"CC0-1.0"` / `"CC-BY-4.0"` / `"ODC-BY-1.0"`. The registry CI
    /// (C8) enforces the per-tier license policy.
    #[serde(default)]
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
    #[serde(default)]
    pub authors: Vec<String>,
    /// Publication year.
    #[serde(default)]
    pub year: Option<i32>,
    /// Journal / venue.
    #[serde(default)]
    pub journal: Option<String>,
    /// A ready-to-paste BibTeX entry.
    #[serde(default)]
    pub bibtex: Option<String>,
}

/// Population block — the facets discovery and `query` match on.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Population {
    /// ISO 639-3 language code, e.g. `"eng"`.
    #[serde(default)]
    pub language: Option<String>,
    /// Variety / dialect, e.g. `"AmE"`.
    #[serde(default)]
    pub variety: Option<String>,
    /// Sexes represented, e.g. `["m", "f", "c"]`.
    #[serde(default)]
    pub sex: Vec<String>,
    /// Age bands represented, e.g. `["adult", "child"]`.
    #[serde(default)]
    pub age_band: Vec<String>,
    /// Number of speakers.
    #[serde(default)]
    pub n_speakers: Option<u64>,
    /// Number of tokens.
    #[serde(default)]
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
    #[serde(default)]
    pub parameters: Vec<String>,
    /// Units of the parameters, e.g. `"Hz"`.
    #[serde(default)]
    pub units: Option<String>,
    /// Phones the distribution covers (ARPABET/IPA strings), if applicable.
    #[serde(default)]
    pub phones: Vec<String>,
    /// Phonetic context, e.g. `"hVd"`.
    #[serde(default)]
    pub context: Option<String>,
    /// Free-text description of how the measurement was made.
    #[serde(default)]
    pub measurement_method: Option<String>,
}

/// Privacy / shareability block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Privacy {
    /// `"raw_samples"` or `"summary_only"`.
    #[serde(default)]
    pub shareability: Option<String>,
    /// k-anonymity floor per subgroup (default 5 enforced by the registry).
    #[serde(default)]
    pub min_n_per_subgroup: Option<u64>,
    /// Required `true` for small-language community data.
    #[serde(default)]
    pub community_consent: bool,
}

/// Schema block — the data file and its column layout.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Schema {
    /// Data file name relative to the distribution directory.
    #[serde(default)]
    pub data_file: Option<String>,
    /// `"long"` or `"wide"`.
    #[serde(default)]
    pub shape: Option<String>,
    /// Column names in the data file.
    #[serde(default)]
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

impl RefDist {
    /// Absolute path to the manifest's declared data file, if any.
    pub fn data_path(&self) -> Option<PathBuf> {
        self.manifest
            .schema
            .data_file
            .as_ref()
            .map(|f| self.dir.join(f))
    }
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
}
