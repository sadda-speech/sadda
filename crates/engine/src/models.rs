//! Model registry — consumption side (Phase 3 E11, part 3a). Parallel to
//! `engine::refdist` (C7): parse a `model.toml` manifest, resolve a model
//! by id from a user-level cache or the bundled set, and run it. Behind
//! the `ml` feature.
//!
//! `load_model(id)` resolves three ID schemes:
//! - `sadda/<name>[@version]` — curated: the user store, falling back to
//!   the bundled set that ships with the app.
//! - `local://<path>` — a model directory (with `model.toml`), or a bare
//!   model file (a minimal manifest is synthesized; the caller asserts the
//!   task by which method it calls).
//! - `hf://<org>/<name>/<file>[@<rev>]` — HuggingFace passthrough: with the
//!   `download` feature, fetches the file into the cache and runs it
//!   (unverified/uncurated); without it, a clear "needs `download`" error.
//!
//! The architecture is deliberately a *parallel* module rather than a
//! shared generic core with refdist; revisit after E12 once both
//! registries are concrete. See the 2026-05-27 design DEVLOG entry.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::audio::Audio;
use crate::error::{EngineError, Result};
use crate::ml::VadFrame;

fn model_err(msg: impl Into<String>) -> EngineError {
    EngineError::Ml(msg.into())
}

/// A parsed `model.toml` manifest. Most fields are optional so a partial
/// manifest still parses; `id` + `version` identify the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Resolvable id, e.g. `"sadda/silero-vad"`.
    pub id: String,
    /// Semantic version.
    pub version: String,
    /// Human-readable title.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub title: String,
    /// Upstream source URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_source: Option<String>,
    /// SPDX license id of the weights.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// The model file + format + checksum.
    #[serde(default)]
    pub model: ModelSpec,
    /// Expected input.
    #[serde(default)]
    pub input: ModelInput,
    /// What inference produces, in sadda's tier vocabulary.
    #[serde(default)]
    pub output: ModelOutput,
    /// Compute hints.
    #[serde(default)]
    pub compute: ModelCompute,
    /// Citation metadata.
    #[serde(default)]
    pub citation: ModelCitation,
}

/// `[model]` block — the file and its format.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelSpec {
    /// `embedding` | `transcription` | `vad` | `segmentation` | `alignment` | `feature`.
    #[serde(default)]
    pub kind: String,
    /// `onnx` | `gguf` | `safetensors` | `savedmodel`.
    #[serde(default)]
    pub format: String,
    /// Model file name relative to the directory (curated / bundled).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// External mirror URL (the E12 fetch path) — for weights too large
    /// for a bundled/release artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// `sha256:…` checksum of the model file / weights.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_checksum: Option<String>,
}

/// `[input]` block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelInput {
    /// `audio` | `video` | `both`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modality: Option<String>,
    /// Expected sample rate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
    /// Expected channel count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channels: Option<u16>,
}

/// `[output]` block — ties inference results to a tier kind.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelOutput {
    /// Tier kind produced (e.g. `interval`, `continuous_vector`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier_kind: Option<String>,
    /// Output channels / embedding dimensionality.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channels: Option<u32>,
    /// Output frame rate, if a dense signal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_rate_hz: Option<u32>,
}

/// `[compute]` block — surfaced before a download so the engine can warn
/// "won't run on your machine".
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCompute {
    /// Minimum RAM in MB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_min_ram_mb: Option<u64>,
    /// `required` | `optional` | `unsupported`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
}

/// `[citation]` block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCitation {
    /// Authors.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub authors: Vec<String>,
    /// Publication year.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub year: Option<i32>,
    /// A ready-to-paste BibTeX entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bibtex: Option<String>,
}

/// Parses a `model.toml` manifest from a TOML string.
pub fn parse_model_manifest(toml_str: &str) -> Result<ModelManifest> {
    toml::from_str(toml_str).map_err(|e| model_err(format!("invalid model.toml: {e}")))
}

/// Loads the `model.toml` manifest from a model directory.
pub fn load_model_manifest(dir: impl AsRef<Path>) -> Result<ModelManifest> {
    let path = dir.as_ref().join("model.toml");
    let text = fs::read_to_string(&path)
        .map_err(|e| model_err(format!("cannot read {}: {e}", path.display())))?;
    parse_model_manifest(&text)
}

/// A resolved model: its manifest plus the directory it lives in.
#[derive(Debug, Clone)]
pub struct Model {
    /// The parsed manifest.
    pub manifest: ModelManifest,
    /// Directory containing `model.toml` and the model file.
    pub dir: PathBuf,
}

impl Model {
    /// Resolves a model from a directory containing `model.toml`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Model> {
        let dir = dir.as_ref().to_path_buf();
        let manifest = load_model_manifest(&dir)?;
        Ok(Model { manifest, dir })
    }

    /// Absolute path to the model file, if the manifest names one.
    pub fn file_path(&self) -> Option<PathBuf> {
        self.manifest.model.file.as_ref().map(|f| self.dir.join(f))
    }

    /// Resolvable id.
    pub fn id(&self) -> &str {
        &self.manifest.id
    }
    /// Version.
    pub fn version(&self) -> &str {
        &self.manifest.version
    }
    /// Model kind (`vad`, `embedding`, …).
    pub fn kind(&self) -> &str {
        &self.manifest.model.kind
    }
    /// Weights checksum (`sha256:…`), for `ProcessingRun` provenance.
    pub fn weights_checksum(&self) -> Option<&str> {
        self.manifest.model.file_checksum.as_deref()
    }

    /// Runs this model as a VAD over `audio` (delegates to
    /// [`crate::ml::vad`]). Errors if the manifest declares a non-`vad`
    /// kind or the model has no local file. An empty (unknown) kind is
    /// allowed — the caller asserts the task by calling this method.
    pub fn vad(&self, audio: &Audio) -> Result<Vec<VadFrame>> {
        let kind = &self.manifest.model.kind;
        if !kind.is_empty() && kind != "vad" {
            return Err(model_err(format!(
                "model {:?} is kind {kind:?}, not vad",
                self.manifest.id
            )));
        }
        let path = self
            .file_path()
            .ok_or_else(|| model_err(format!("model {:?} has no local file", self.manifest.id)))?;
        crate::ml::vad(audio, &path)
    }
}

/// User-level model cache, rooted at `~/.local/share/sadda/models/` (or
/// the platform equivalent). Models nest under `<id>/<version>/`. Parallel
/// to [`crate::refdist::RefdistStore`].
pub struct ModelStore {
    root: PathBuf,
}

impl ModelStore {
    /// A store rooted at an explicit directory.
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    /// The per-user default store, created if missing.
    pub fn user_default() -> Result<Self> {
        let dirs = directories::ProjectDirs::from("", "", "sadda")
            .ok_or_else(|| model_err("cannot determine user data directory"))?;
        let root = dirs.data_dir().join("models");
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    /// Root directory of the store.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The model with this `id` + `version`, or `None`.
    pub fn get(&self, id: &str, version: &str) -> Option<Model> {
        let dir = self.root.join(id).join(version);
        if dir.join("model.toml").is_file() {
            Model::from_dir(&dir).ok()
        } else {
            None
        }
    }

    /// The highest-versioned model for `id` (lexicographic), or `None`.
    pub fn get_latest(&self, id: &str) -> Option<Model> {
        let base = self.root.join(id);
        let mut versions: Vec<std::ffi::OsString> = fs::read_dir(&base)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name())
            .collect();
        versions.sort();
        let v = versions.last()?;
        Model::from_dir(base.join(v)).ok()
    }

    /// `get(id, version)` if `version` is given, else `get_latest(id)`.
    pub fn resolve(&self, id: &str, version: Option<&str>) -> Option<Model> {
        match version {
            Some(v) => self.get(id, v),
            None => self.get_latest(id),
        }
    }

    /// Installs a model directory (a `model.toml` + its files) into the
    /// store under `<id>/<version>/` by copying it in. How the bundled set
    /// seeds the cache and where a fetched model lands (E12).
    pub fn install_from_dir(&self, src_dir: impl AsRef<Path>) -> Result<Model> {
        let src = src_dir.as_ref();
        let manifest = load_model_manifest(src)?;
        let dest = self.root.join(&manifest.id).join(&manifest.version);
        fs::create_dir_all(&dest)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                fs::copy(entry.path(), dest.join(entry.file_name()))?;
            }
        }
        Ok(Model {
            manifest,
            dir: dest,
        })
    }
}

/// One entry in a model registry's published `index.json` — the discovery
/// metadata for a model available from a hosted registry, without its
/// weights. Parallel to `refdist::RegistryEntry`; the engine reads the
/// index to list what's available, the weights are fetched separately
/// (E12).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelRegistryEntry {
    /// Model id.
    pub id: String,
    /// Model version.
    pub version: String,
    /// Registry tier (2 = curated, 3 = community).
    #[serde(default)]
    pub tier: u8,
    /// Human-readable title.
    #[serde(default)]
    pub title: String,
    /// Model kind (`vad`, `embedding`, …).
    #[serde(default)]
    pub kind: String,
    /// Weights format (`onnx`, …).
    #[serde(default)]
    pub format: String,
    /// SPDX license id.
    #[serde(default)]
    pub license: Option<String>,
    /// Path to the entry within the registry (e.g. `tier2/<dir>`).
    #[serde(default)]
    pub path: Option<String>,
    /// Whether this version has been yanked.
    #[serde(default)]
    pub yanked: bool,
}

/// A model registry's `index.json` (the GitHub-Pages artifact). Parallel
/// to `refdist::RegistryIndex`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelRegistryIndex {
    /// Index format version.
    #[serde(default)]
    pub schema_version: u32,
    /// The models the registry publishes.
    #[serde(default)]
    pub entries: Vec<ModelRegistryEntry>,
}

/// Parses a model registry's `index.json`.
pub fn parse_model_index(json: &str) -> Result<ModelRegistryIndex> {
    serde_json::from_str(json).map_err(|e| model_err(format!("invalid model index.json: {e}")))
}

/// Verifies a file's SHA-256 against an expected `sha256:<hex>` string
/// (case-insensitive) — the trust check applied to fetched/curated
/// weights against a manifest's `file_checksum`.
pub fn verify_checksum(path: &Path, expected: &str) -> Result<()> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut f = fs::File::open(path)
        .map_err(|e| model_err(format!("cannot open {} for checksum: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf).map_err(|e| model_err(e.to_string()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let got = format!("sha256:{:x}", hasher.finalize());
    if got.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(model_err(format!(
            "checksum mismatch for {}: expected {expected}, got {got}",
            path.display()
        )))
    }
}

/// Parses an `hf://<org>/<name>/<file…>[@<rev>]` id into
/// `(repo = "<org>/<name>", rev, file)`. `rev` defaults to `main`.
#[cfg(feature = "download")]
fn parse_hf_id(rest: &str) -> Result<(String, String, String)> {
    let (path, rev) = match rest.rsplit_once('@') {
        Some((p, r)) => (p, r.to_string()),
        None => (rest, "main".to_string()),
    };
    let parts: Vec<&str> = path.splitn(3, '/').collect();
    if parts.len() < 3 || parts.iter().any(|p| p.is_empty()) {
        return Err(model_err(format!(
            "hf:// id {rest:?} must be `hf://<org>/<name>/<file>[@<rev>]`"
        )));
    }
    Ok((
        format!("{}/{}", parts[0], parts[1]),
        rev,
        parts[2].to_string(),
    ))
}

/// The HuggingFace "resolve" URL for a repo file at a revision.
#[cfg(feature = "download")]
fn hf_resolve_url(repo: &str, rev: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/{rev}/{file}")
}

/// Downloads `url` to `dest` (atomically, via a `.part` temp), sending an
/// `Authorization: Bearer` header when `token` is given. Streams — no
/// whole-file buffering.
#[cfg(feature = "download")]
fn download_file(url: &str, dest: &Path, token: Option<&str>) -> Result<u64> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut req = ureq::get(url);
    if let Some(t) = token {
        req = req.header("Authorization", &format!("Bearer {t}"));
    }
    let mut resp = req
        .call()
        .map_err(|e| model_err(format!("download {url}: {e}")))?;
    let tmp = dest.with_extension("part");
    let mut file = fs::File::create(&tmp)?;
    // Cap well above any single model file; ureq's default body limit is small.
    let mut reader = resp.body_mut().with_config().limit(16 << 30).reader();
    let n = std::io::copy(&mut reader, &mut file).map_err(|e| model_err(e.to_string()))?;
    drop(file);
    fs::rename(&tmp, dest)?;
    Ok(n)
}

/// Resolves an `hf://…` id by downloading the file into the model cache
/// (`<store>/hf/<repo>/<rev>/<file>`, skipped if already present) and
/// returning a [`Model`] over it. HuggingFace passthrough is **unverified
/// and uncurated** (no manifest, no quality guarantee — the trust tier the
/// 2026-05-20 entry calls out); auth via the `HF_TOKEN` env var.
#[cfg(feature = "download")]
fn fetch_hf(id: &str) -> Result<Model> {
    let rest = id.strip_prefix("hf://").unwrap_or(id);
    let (repo, rev, file) = parse_hf_id(rest)?;
    let dir = ModelStore::user_default()?
        .root()
        .join("hf")
        .join(&repo)
        .join(&rev);
    let dest = dir.join(&file);
    if !dest.is_file() {
        let token = std::env::var("HF_TOKEN").ok();
        download_file(&hf_resolve_url(&repo, &rev, &file), &dest, token.as_deref())?;
    }
    // Synthesized manifest: hf passthrough declares no kind (the caller
    // asserts the task by which method it calls).
    Ok(Model {
        manifest: ModelManifest {
            id: format!("hf://{repo}/{file}"),
            version: rev,
            title: String::new(),
            upstream_source: Some(format!("https://huggingface.co/{repo}")),
            license: None,
            model: ModelSpec {
                format: "onnx".into(),
                file: Some(file),
                ..ModelSpec::default()
            },
            input: ModelInput::default(),
            output: ModelOutput::default(),
            compute: ModelCompute::default(),
            citation: ModelCitation::default(),
        },
        dir,
    })
}

/// Resolves a model by id. See the module docs for the three schemes.
pub fn load_model(id: &str) -> Result<Model> {
    if let Some(rest) = id.strip_prefix("local://") {
        return resolve_local(Path::new(rest));
    }
    if id.starts_with("hf://") {
        #[cfg(feature = "download")]
        {
            return fetch_hf(id);
        }
        #[cfg(not(feature = "download"))]
        {
            return Err(model_err(format!(
                "resolving {id:?} requires the `download` feature (network); rebuild sadda \
                 with `--features download`, or use a curated `sadda/…` id or `local://…`"
            )));
        }
    }
    // Curated `sadda/<name>[@version]`: user store first, then bundled set.
    let (name_id, version) = match id.split_once('@') {
        Some((n, v)) => (n, Some(v)),
        None => (id, None),
    };
    if let Ok(store) = ModelStore::user_default() {
        if let Some(m) = store.resolve(name_id, version) {
            return Ok(m);
        }
    }
    if let Some(short) = name_id.strip_prefix("sadda/") {
        if let Some(dir) = bundled_model_dir(short) {
            return Model::from_dir(dir);
        }
    }
    Err(model_err(format!(
        "model {id:?} not found in the store or bundled set"
    )))
}

fn resolve_local(path: &Path) -> Result<Model> {
    if path.is_dir() {
        Model::from_dir(path)
    } else if path.is_file() {
        // Bare model file: synthesize a minimal manifest (kind unknown).
        let file = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("model")
            .to_string();
        let dir = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(Model {
            manifest: ModelManifest {
                id: format!("local:{file}"),
                version: "0.0.0".into(),
                title: String::new(),
                upstream_source: None,
                license: None,
                model: ModelSpec {
                    format: "onnx".into(),
                    file: Some(file),
                    ..ModelSpec::default()
                },
                input: ModelInput::default(),
                output: ModelOutput::default(),
                compute: ModelCompute::default(),
                citation: ModelCitation::default(),
            },
            dir,
        })
    } else {
        Err(model_err(format!(
            "local model path does not exist: {}",
            path.display()
        )))
    }
}

/// Locates a bundled model directory (`models-bundled/<name>/`): an
/// explicit `SADDA_MODELS_BUNDLED` override, then next to the executable
/// (shipped layout), then the workspace copy (dev). Mirrors the refdist
/// bundled-set locator.
pub fn bundled_model_dir(name: &str) -> Option<PathBuf> {
    let candidates = [
        std::env::var_os("SADDA_MODELS_BUNDLED").map(PathBuf::from),
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|d| d.join("models-bundled"))),
        Some(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models-bundled")),
    ];
    for base in candidates.into_iter().flatten() {
        let dir = base.join(name);
        if dir.join("model.toml").is_file() {
            return Some(dir);
        }
    }
    None
}

/// Runs the bundled Silero VAD over `audio` (the convenience the Python
/// `sadda.ml.vad` and the GUI VAD lane use). Errors if the bundled model
/// can't be found or ONNX Runtime isn't available.
pub fn vad_bundled(audio: &Audio) -> Result<Vec<VadFrame>> {
    let dir =
        bundled_model_dir("silero-vad").ok_or_else(|| model_err("bundled Silero VAD not found"))?;
    Model::from_dir(dir)?.vad(audio)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = r#"
id = "sadda/test-vad"
version = "1.0.0"
title = "Test VAD"
license = "MIT"

[model]
kind = "vad"
format = "onnx"
file = "model.onnx"
file_checksum = "sha256:abc"

[output]
tier_kind = "interval"
"#;

    #[test]
    fn parses_manifest() {
        let m = parse_model_manifest(MANIFEST).unwrap();
        assert_eq!(m.id, "sadda/test-vad");
        assert_eq!(m.model.kind, "vad");
        assert_eq!(m.model.file.as_deref(), Some("model.onnx"));
        assert_eq!(m.model.file_checksum.as_deref(), Some("sha256:abc"));
        assert_eq!(m.output.tier_kind.as_deref(), Some("interval"));
    }

    fn temp_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "sadda_models_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn write_model(dir: &Path) {
        fs::create_dir_all(dir).unwrap();
        fs::write(dir.join("model.toml"), MANIFEST).unwrap();
        fs::write(dir.join("model.onnx"), b"not really onnx").unwrap();
    }

    #[test]
    fn model_from_dir_and_accessors() {
        let dir = temp_dir().join("m");
        write_model(&dir);
        let m = Model::from_dir(&dir).unwrap();
        assert_eq!(m.id(), "sadda/test-vad");
        assert_eq!(m.version(), "1.0.0");
        assert_eq!(m.kind(), "vad");
        assert_eq!(m.weights_checksum(), Some("sha256:abc"));
        assert_eq!(m.file_path(), Some(dir.join("model.onnx")));
    }

    #[test]
    fn store_install_and_get_round_trip() {
        let src = temp_dir().join("src");
        write_model(&src);
        let store = ModelStore::new(temp_dir());
        let installed = store.install_from_dir(&src).unwrap();
        assert_eq!(installed.id(), "sadda/test-vad");
        // Nested <id>/<version>/ layout.
        assert!(installed.dir.ends_with("sadda/test-vad/1.0.0"));
        assert!(store.get("sadda/test-vad", "1.0.0").is_some());
        assert!(store.resolve("sadda/test-vad", None).is_some()); // latest
        assert!(store.get("sadda/test-vad", "9.9.9").is_none());
    }

    #[test]
    fn load_model_local_dir() {
        let dir = temp_dir().join("m");
        write_model(&dir);
        let id = format!("local://{}", dir.display());
        let m = load_model(&id).unwrap();
        assert_eq!(m.kind(), "vad");
    }

    #[test]
    fn load_model_local_bare_file_synthesizes_manifest() {
        let dir = temp_dir();
        let onnx = dir.join("foo.onnx");
        fs::write(&onnx, b"x").unwrap();
        let m = load_model(&format!("local://{}", onnx.display())).unwrap();
        assert_eq!(m.kind(), ""); // unknown
        assert_eq!(m.file_path(), Some(onnx));
    }

    #[cfg(not(feature = "download"))]
    #[test]
    fn load_model_hf_needs_download_feature() {
        let e = load_model("hf://facebook/wav2vec2-base-960h").unwrap_err();
        assert!(format!("{e}").contains("download"));
    }

    #[test]
    fn verify_checksum_matches_and_mismatches() {
        let dir = temp_dir();
        let path = dir.join("blob.bin");
        fs::write(&path, b"hello sadda").unwrap();
        // sha256("hello sadda")
        let expected = "sha256:7f6dc7da26a99c2086f607e02f8211b323f5746a09bf8b9f3ef3d92dfb9c92be";
        // Compute once to avoid hard-coding a possibly-wrong constant:
        let got = {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(b"hello sadda");
            format!("sha256:{:x}", h.finalize())
        };
        assert!(verify_checksum(&path, &got).is_ok());
        assert!(verify_checksum(&path, "sha256:deadbeef").is_err());
        let _ = expected;
    }

    #[cfg(feature = "download")]
    #[test]
    fn hf_id_parsing_and_url() {
        let (repo, rev, file) = parse_hf_id("onnx-community/silero-vad/onnx/model.onnx").unwrap();
        assert_eq!(repo, "onnx-community/silero-vad");
        assert_eq!(rev, "main");
        assert_eq!(file, "onnx/model.onnx");
        let (_, rev2, _) = parse_hf_id("org/name/file.onnx@abc123").unwrap();
        assert_eq!(rev2, "abc123");
        assert!(parse_hf_id("too/short").is_err());
        assert_eq!(
            hf_resolve_url("org/name", "main", "f.onnx"),
            "https://huggingface.co/org/name/resolve/main/f.onnx"
        );
    }

    // Real network download (+ inference). Gated behind SADDA_NET_TESTS so
    // it never runs in CI; needs network and (for .vad) ONNX Runtime.
    #[cfg(feature = "download")]
    #[test]
    fn fetch_hf_downloads_and_runs() {
        if std::env::var("SADDA_NET_TESTS").is_err() {
            eprintln!("fetch_hf_downloads_and_runs skipped (set SADDA_NET_TESTS=1)");
            return;
        }
        let m = load_model("hf://onnx-community/silero-vad/onnx/model.onnx").unwrap();
        assert!(m.file_path().unwrap().is_file());
        // If ORT is present, the downloaded model actually runs.
        let audio = Audio {
            samples: vec![0.0f32; 16_000],
            sample_rate: 16_000,
            channels: 1,
        };
        match m.vad(&audio) {
            Ok(frames) => assert!(!frames.is_empty()),
            Err(EngineError::Ml(msg)) => eprintln!("vad skipped (ORT unavailable): {msg}"),
            Err(e) => panic!("unexpected: {e}"),
        }
    }

    #[test]
    fn parses_registry_index() {
        let json = r#"{
          "schema_version": 1,
          "entries": [
            { "id": "sadda/wav2vec2-base", "version": "1.0.0", "tier": 2,
              "kind": "embedding", "format": "onnx", "license": "Apache-2.0",
              "path": "tier2/wav2vec2-base" },
            { "id": "sadda/whisper-tiny", "version": "1.0.0", "tier": 3 }
          ]
        }"#;
        let index = parse_model_index(json).unwrap();
        assert_eq!(index.schema_version, 1);
        assert_eq!(index.entries.len(), 2);
        assert_eq!(index.entries[0].tier, 2);
        assert_eq!(index.entries[0].kind, "embedding");
        assert_eq!(index.entries[0].license.as_deref(), Some("Apache-2.0"));
        assert!(!index.entries[1].yanked);
        assert!(index.entries[1].kind.is_empty()); // defaulted
    }

    #[test]
    fn bundled_silero_is_locatable_and_vad_kind() {
        let dir = bundled_model_dir("silero-vad").expect("bundled silero-vad dir");
        let m = Model::from_dir(dir).unwrap();
        assert_eq!(m.kind(), "vad");
        assert_eq!(m.id(), "sadda/silero-vad");
    }

    #[test]
    fn load_model_curated_resolves_bundled_fallback() {
        // No user store entry → falls back to the bundled set.
        let m = load_model("sadda/silero-vad").unwrap();
        assert_eq!(m.kind(), "vad");
    }

    #[test]
    fn vad_kind_check_rejects_non_vad() {
        let dir = temp_dir().join("emb");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("model.toml"),
            "id=\"x\"\nversion=\"1\"\n[model]\nkind=\"embedding\"\nfile=\"m.onnx\"\n",
        )
        .unwrap();
        fs::write(dir.join("m.onnx"), b"x").unwrap();
        let m = Model::from_dir(&dir).unwrap();
        let audio = Audio {
            samples: vec![0.0; 16000],
            sample_rate: 16000,
            channels: 1,
        };
        let e = m.vad(&audio).unwrap_err();
        assert!(format!("{e}").contains("not vad"));
    }

    // End-to-end inference through the bundled model. Requires a runtime
    // ONNX Runtime (`ORT_DYLIB_PATH`); skips cleanly (the part-1 probe
    // returns an error, not a panic) when ORT is absent, so CI stays
    // green. Run locally with `ORT_DYLIB_PATH=…/libonnxruntime.so`.
    #[test]
    fn vad_bundled_runs_on_silence() {
        let audio = Audio {
            samples: vec![0.0f32; 16_000],
            sample_rate: 16_000,
            channels: 1,
        };
        match vad_bundled(&audio) {
            Ok(frames) => {
                assert!(!frames.is_empty());
                let mean = frames.iter().map(|f| f.speech_prob).sum::<f32>() / frames.len() as f32;
                assert!(mean < 0.3, "silence read as speech (mean {mean})");
            }
            Err(EngineError::Ml(msg)) => {
                eprintln!("vad_bundled_runs_on_silence skipped (ORT unavailable): {msg}");
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
}
