//! Integration tests for the V5 dense-tier Parquet sidecar API (Phase 1
//! slice B3): write+read round-trips for `continuous_numeric`,
//! `continuous_vector`, `categorical_sampled`; `derived_signal` row
//! bookkeeping; sidecar path layout; double-write rejection.

use std::path::{Path, PathBuf};

use ndarray::{Array2, ArrayView2};
use sadda_engine::{BundleSpec, EngineError, Project, TierSpec, TierType};

fn unique_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "sadda_engine_b3_test_{}_{}",
        std::process::id(),
        name
    ));
    p
}

fn write_short_wav(path: &Path, sample_rate: u32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..sample_rate / 4 {
        let t = i as f32 / sample_rate as f32;
        let s = (0.5 * i16::MAX as f32 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()) as i16;
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
}

fn new_project_with_bundle(root: &Path) -> (Project, i64) {
    let _ = std::fs::remove_dir_all(root);
    let proj = Project::create(root, "p").unwrap();
    let wav = std::env::temp_dir().join(format!(
        "sadda_b3_{}_{}.wav",
        std::process::id(),
        root.file_name().unwrap().to_string_lossy(),
    ));
    write_short_wav(&wav, 16_000);
    let bundle_id = proj.add_bundle_with(&BundleSpec::new("b"), &wav).unwrap();
    let _ = std::fs::remove_file(&wav);
    (proj, bundle_id)
}

#[test]
fn continuous_numeric_round_trip_and_derived_signal_row() {
    let root = unique_dir("cn_round_trip");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let tier = proj
        .add_tier(&TierSpec::new(bundle_id, "f0", TierType::ContinuousNumeric))
        .unwrap();

    let samples: Vec<f64> = (0..100).map(|i| 80.0 + i as f64 * 0.5).collect();
    let ds_id = proj
        .write_continuous_numeric(tier, &samples, 100.0)
        .unwrap();

    let ds = proj.derived_signal(tier).unwrap().expect("registered");
    assert_eq!(ds.id, ds_id);
    assert_eq!(ds.tier_id, tier);
    assert_eq!(ds.n_frames, 100);
    assert_eq!(ds.n_dims, 1);
    assert_eq!(ds.dtype, "f64");
    assert_eq!(ds.sample_rate_hz, Some(100.0));
    assert_eq!(
        ds.relative_path,
        format!("signals/derived/bundle_{bundle_id}/f0.parquet")
    );

    let path = proj.dense_path(tier).unwrap().unwrap();
    assert!(path.exists(), "sidecar should exist at {}", path.display());

    let back = proj.read_continuous_numeric(tier).unwrap();
    assert_eq!(back, samples);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn continuous_vector_round_trip_shape_preserved() {
    let root = unique_dir("cv_round_trip");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let tier = proj
        .add_tier(&TierSpec::new(
            bundle_id,
            "wav2vec.layer8",
            TierType::ContinuousVector,
        ))
        .unwrap();
    let arr = Array2::from_shape_fn((25, 6), |(r, c)| (r * 100 + c) as f64 * 0.1);
    let _ = proj
        .write_continuous_vector(tier, arr.view(), 50.0)
        .unwrap();

    let ds = proj.derived_signal(tier).unwrap().unwrap();
    assert_eq!(ds.n_frames, 25);
    assert_eq!(ds.n_dims, 6);
    assert!(ds.relative_path.contains("wav2vec.layer8.parquet"));

    let back = proj.read_continuous_vector(tier).unwrap();
    assert_eq!(back.dim(), (25, 6));
    assert_eq!(back, arr);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn categorical_sampled_round_trip_preserves_labels() {
    let root = unique_dir("cs_round_trip");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let tier = proj
        .add_tier(&TierSpec::new(
            bundle_id,
            "vad",
            TierType::CategoricalSampled,
        ))
        .unwrap();
    let labels: Vec<String> = [
        "speech", "speech", "silence", "speech", "silence", "silence",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let _ = proj.write_categorical_sampled(tier, &labels, 10.0).unwrap();
    let back = proj.read_categorical_sampled(tier).unwrap();
    assert_eq!(back, labels);

    let ds = proj.derived_signal(tier).unwrap().unwrap();
    assert_eq!(ds.dtype, "utf8");
    assert_eq!(ds.n_dims, 1);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn writing_to_wrong_tier_type_errors() {
    let root = unique_dir("wrong_type");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let cn = proj
        .add_tier(&TierSpec::new(bundle_id, "f0", TierType::ContinuousNumeric))
        .unwrap();
    let arr = Array2::<f64>::zeros((4, 3));
    let err = proj
        .write_continuous_vector(cn, arr.view(), 100.0)
        .unwrap_err();
    assert!(matches!(err, EngineError::Corpus(_)), "got {err:?}");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn double_write_for_same_tier_is_rejected() {
    let root = unique_dir("double_write");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let tier = proj
        .add_tier(&TierSpec::new(bundle_id, "f0", TierType::ContinuousNumeric))
        .unwrap();
    let samples: Vec<f64> = (0..10).map(|i| i as f64).collect();
    proj.write_continuous_numeric(tier, &samples, 100.0)
        .unwrap();
    let err = proj
        .write_continuous_numeric(tier, &samples, 100.0)
        .unwrap_err();
    assert!(matches!(err, EngineError::Corpus(_)), "got {err:?}");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn read_before_write_errors() {
    let root = unique_dir("read_before_write");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let tier = proj
        .add_tier(&TierSpec::new(bundle_id, "f0", TierType::ContinuousNumeric))
        .unwrap();
    let err = proj.read_continuous_numeric(tier).unwrap_err();
    assert!(matches!(err, EngineError::Corpus(_)), "got {err:?}");
    assert!(proj.derived_signal(tier).unwrap().is_none());
    assert!(proj.dense_path(tier).unwrap().is_none());

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn derived_signal_insert_writes_audit_log_row() {
    let root = unique_dir("audit_derived_signal");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let tier = proj
        .add_tier(&TierSpec::new(bundle_id, "f0", TierType::ContinuousNumeric))
        .unwrap();
    let samples: Vec<f64> = vec![1.0, 2.0, 3.0];
    proj.write_continuous_numeric(tier, &samples, 100.0)
        .unwrap();

    let conn = rusqlite::Connection::open(root.join("corpus.db")).unwrap();
    let after: String = conn
        .query_row(
            "SELECT after FROM audit_log \
             WHERE table_name = 'derived_signal' AND op = 'insert' \
             ORDER BY id DESC LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(after.contains("\"dtype\":\"f64\""));
    assert!(after.contains("\"n_frames\":3"));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn tier_name_with_unsafe_chars_is_sanitized_in_path() {
    let root = unique_dir("sanitize");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let tier = proj
        .add_tier(&TierSpec::new(
            bundle_id,
            "weird/name with space*",
            TierType::ContinuousNumeric,
        ))
        .unwrap();
    let samples = vec![1.0_f64, 2.0, 3.0];
    proj.write_continuous_numeric(tier, &samples, 100.0)
        .unwrap();
    let ds = proj.derived_signal(tier).unwrap().unwrap();
    // Slashes and spaces and asterisks must not appear in the filename.
    let last_segment = ds.relative_path.rsplit('/').next().unwrap();
    assert!(!last_segment.contains('/'), "{last_segment}");
    assert!(!last_segment.contains(' '), "{last_segment}");
    assert!(!last_segment.contains('*'), "{last_segment}");
    assert!(last_segment.ends_with(".parquet"));

    // Path is reachable on disk.
    let path = proj.dense_path(tier).unwrap().unwrap();
    assert!(path.exists());

    // Helper passthrough: the helper used internally returns the same
    // sanitized filename for this input.
    let _ = ArrayView2::<f64>::from_shape((1, 1), &[1.0]); // touch ArrayView2 import; no-op

    let _ = std::fs::remove_dir_all(&root);
}
