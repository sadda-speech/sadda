//! Integration tests for the V4 sparse-annotation schema (Phase 1 slice B2):
//! tier CRUD, interval/point/reference round-trips, and Rust-level
//! parent-child cardinality enforcement.

use std::path::{Path, PathBuf};

use rusqlite::Connection;
use sadda_engine::{
    BundleSpec, EngineError, IntervalSpec, PointSpec, Project, ReferenceSpec, TierSpec, TierType,
};

fn unique_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "sadda_engine_b2_test_{}_{}",
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

/// Builds a project with one bundle and returns `(project, bundle_id)`.
fn new_project_with_bundle(root: &Path) -> (Project, i64) {
    let _ = std::fs::remove_dir_all(root);
    let proj = Project::create(root, "p").unwrap();
    let wav = std::env::temp_dir().join(format!(
        "sadda_b2_proj_{}_{}.wav",
        std::process::id(),
        root.file_name().unwrap().to_string_lossy(),
    ));
    write_short_wav(&wav, 16_000);
    let bundle_id = proj.add_bundle_with(&BundleSpec::new("b"), &wav).unwrap();
    let _ = std::fs::remove_file(&wav);
    (proj, bundle_id)
}

#[test]
fn tier_crud_round_trips_with_type_enum() {
    let root = unique_dir("tier_crud");
    let (proj, bundle_id) = new_project_with_bundle(&root);

    let words = proj
        .add_tier(&TierSpec::new(bundle_id, "words", TierType::Interval))
        .unwrap();
    let phones = proj
        .add_tier(&TierSpec {
            bundle_id,
            name: "phones".into(),
            r#type: Some(TierType::Interval),
            parent_id: Some(words),
            cardinality: Some("one_to_many".into()),
            ..Default::default()
        })
        .unwrap();

    let all = proj.tiers(Some(bundle_id)).unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].name, "words");
    assert_eq!(all[0].r#type, TierType::Interval);
    assert!(all[0].parent_id.is_none());
    assert_eq!(all[1].name, "phones");
    assert_eq!(all[1].parent_id, Some(words));
    assert_eq!(all[1].cardinality.as_deref(), Some("one_to_many"));

    let p = proj.get_tier(phones).unwrap();
    assert_eq!(p.id, phones);
    assert_eq!(p.bundle_id, bundle_id);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn interval_round_trip_basic() {
    let root = unique_dir("interval_basic");
    let (proj, bundle_id) = new_project_with_bundle(&root);

    let tier = proj
        .add_tier(&TierSpec::new(bundle_id, "words", TierType::Interval))
        .unwrap();
    let i1 = proj
        .add_interval(&IntervalSpec {
            tier_id: tier,
            start_seconds: 0.0,
            end_seconds: 0.5,
            label: Some("hello".into()),
            ..Default::default()
        })
        .unwrap();
    let i2 = proj
        .add_interval(&IntervalSpec {
            tier_id: tier,
            start_seconds: 0.5,
            end_seconds: 1.0,
            label: Some("world".into()),
            ..Default::default()
        })
        .unwrap();

    let rows = proj.intervals(tier).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].id, i1);
    assert_eq!(rows[0].label.as_deref(), Some("hello"));
    assert_eq!(rows[1].id, i2);
    assert_eq!(rows[1].start_seconds, 0.5);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn interval_check_constraint_rejects_zero_or_reversed_span() {
    let root = unique_dir("interval_check");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let tier = proj
        .add_tier(&TierSpec::new(bundle_id, "t", TierType::Interval))
        .unwrap();
    let err = proj
        .add_interval(&IntervalSpec {
            tier_id: tier,
            start_seconds: 0.5,
            end_seconds: 0.5,
            ..Default::default()
        })
        .unwrap_err();
    assert!(matches!(err, EngineError::Sqlite(_)), "got {err:?}");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn add_interval_on_non_interval_tier_errors() {
    let root = unique_dir("wrong_tier_type");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let point_tier = proj
        .add_tier(&TierSpec::new(bundle_id, "events", TierType::Point))
        .unwrap();
    let err = proj
        .add_interval(&IntervalSpec {
            tier_id: point_tier,
            start_seconds: 0.0,
            end_seconds: 1.0,
            ..Default::default()
        })
        .unwrap_err();
    assert!(matches!(err, EngineError::Corpus(_)), "got {err:?}");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn cardinality_one_to_many_allows_multiple_children_per_parent() {
    let root = unique_dir("card_one_to_many");
    let (proj, bundle_id) = new_project_with_bundle(&root);

    let words = proj
        .add_tier(&TierSpec::new(bundle_id, "words", TierType::Interval))
        .unwrap();
    let phones = proj
        .add_tier(&TierSpec {
            bundle_id,
            name: "phones".into(),
            r#type: Some(TierType::Interval),
            parent_id: Some(words),
            cardinality: Some("one_to_many".into()),
            ..Default::default()
        })
        .unwrap();

    let w1 = proj
        .add_interval(&IntervalSpec {
            tier_id: words,
            start_seconds: 0.0,
            end_seconds: 1.0,
            label: Some("hello".into()),
            ..Default::default()
        })
        .unwrap();

    let _ = proj
        .add_interval(&IntervalSpec {
            tier_id: phones,
            start_seconds: 0.0,
            end_seconds: 0.5,
            label: Some("h".into()),
            parent_annotation_id: Some(w1),
            ..Default::default()
        })
        .unwrap();
    let _ = proj
        .add_interval(&IntervalSpec {
            tier_id: phones,
            start_seconds: 0.5,
            end_seconds: 1.0,
            label: Some("e".into()),
            parent_annotation_id: Some(w1),
            ..Default::default()
        })
        .unwrap();

    assert_eq!(proj.intervals(phones).unwrap().len(), 2);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn cardinality_one_to_one_rejects_second_child_for_same_parent() {
    let root = unique_dir("card_one_to_one");
    let (proj, bundle_id) = new_project_with_bundle(&root);

    let words = proj
        .add_tier(&TierSpec::new(bundle_id, "words", TierType::Interval))
        .unwrap();
    let labels = proj
        .add_tier(&TierSpec {
            bundle_id,
            name: "stress".into(),
            r#type: Some(TierType::Point),
            parent_id: Some(words),
            cardinality: Some("one_to_one".into()),
            ..Default::default()
        })
        .unwrap();

    let w1 = proj
        .add_interval(&IntervalSpec {
            tier_id: words,
            start_seconds: 0.0,
            end_seconds: 1.0,
            ..Default::default()
        })
        .unwrap();

    let _ = proj
        .add_point(&PointSpec {
            tier_id: labels,
            time_seconds: 0.4,
            label: Some("primary".into()),
            parent_annotation_id: Some(w1),
            ..Default::default()
        })
        .unwrap();

    let err = proj
        .add_point(&PointSpec {
            tier_id: labels,
            time_seconds: 0.6,
            parent_annotation_id: Some(w1),
            ..Default::default()
        })
        .unwrap_err();
    assert!(matches!(err, EngineError::Cardinality(_)), "got {err:?}");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn cardinality_requires_parent_annotation_id_when_parent_tier_set() {
    let root = unique_dir("card_missing_parent_ann");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let words = proj
        .add_tier(&TierSpec::new(bundle_id, "words", TierType::Interval))
        .unwrap();
    let phones = proj
        .add_tier(&TierSpec {
            bundle_id,
            name: "phones".into(),
            r#type: Some(TierType::Interval),
            parent_id: Some(words),
            cardinality: Some("one_to_many".into()),
            ..Default::default()
        })
        .unwrap();

    let err = proj
        .add_interval(&IntervalSpec {
            tier_id: phones,
            start_seconds: 0.0,
            end_seconds: 0.1,
            ..Default::default()
        })
        .unwrap_err();
    assert!(matches!(err, EngineError::Cardinality(_)), "got {err:?}");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn cardinality_rejects_dangling_parent_annotation_id() {
    let root = unique_dir("card_dangling_parent");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let words = proj
        .add_tier(&TierSpec::new(bundle_id, "words", TierType::Interval))
        .unwrap();
    let phones = proj
        .add_tier(&TierSpec {
            bundle_id,
            name: "phones".into(),
            r#type: Some(TierType::Interval),
            parent_id: Some(words),
            cardinality: Some("one_to_many".into()),
            ..Default::default()
        })
        .unwrap();

    let err = proj
        .add_interval(&IntervalSpec {
            tier_id: phones,
            start_seconds: 0.0,
            end_seconds: 0.1,
            parent_annotation_id: Some(99_999),
            ..Default::default()
        })
        .unwrap_err();
    assert!(matches!(err, EngineError::Cardinality(_)), "got {err:?}");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn cardinality_many_to_one_is_deferred() {
    let root = unique_dir("card_many_to_one");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let words = proj
        .add_tier(&TierSpec::new(bundle_id, "words", TierType::Interval))
        .unwrap();
    let m2o = proj
        .add_tier(&TierSpec {
            bundle_id,
            name: "m2o".into(),
            r#type: Some(TierType::Interval),
            parent_id: Some(words),
            cardinality: Some("many_to_one".into()),
            ..Default::default()
        })
        .unwrap();
    let w1 = proj
        .add_interval(&IntervalSpec {
            tier_id: words,
            start_seconds: 0.0,
            end_seconds: 1.0,
            ..Default::default()
        })
        .unwrap();
    let err = proj
        .add_interval(&IntervalSpec {
            tier_id: m2o,
            start_seconds: 0.0,
            end_seconds: 0.5,
            parent_annotation_id: Some(w1),
            ..Default::default()
        })
        .unwrap_err();
    match err {
        EngineError::Cardinality(msg) => assert!(msg.contains("many_to_one"), "msg: {msg}"),
        other => panic!("expected Cardinality, got {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn point_and_reference_round_trip() {
    let root = unique_dir("point_ref");
    let (proj, bundle_id) = new_project_with_bundle(&root);

    let event_tier = proj
        .add_tier(&TierSpec::new(bundle_id, "events", TierType::Point))
        .unwrap();
    let _ = proj
        .add_point(&PointSpec {
            tier_id: event_tier,
            time_seconds: 0.25,
            label: Some("click".into()),
            ..Default::default()
        })
        .unwrap();

    let ref_tier = proj
        .add_tier(&TierSpec::new(
            bundle_id,
            "speaker_turns",
            TierType::Reference,
        ))
        .unwrap();
    let r1 = proj
        .add_reference(&ReferenceSpec {
            tier_id: ref_tier,
            target_kind: "speaker".into(),
            target_id: 1,
            label: Some("alice".into()),
            ..Default::default()
        })
        .unwrap();

    let points = proj.points(event_tier).unwrap();
    let refs = proj.references_for(ref_tier).unwrap();
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].label.as_deref(), Some("click"));
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].id, r1);
    assert_eq!(refs[0].target_kind, "speaker");
    assert_eq!(refs[0].target_id, 1);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn annotation_insert_writes_audit_log_row() {
    let root = unique_dir("audit_annotation");
    let (proj, bundle_id) = new_project_with_bundle(&root);
    let tier = proj
        .add_tier(&TierSpec::new(bundle_id, "phones", TierType::Interval))
        .unwrap();
    let _ = proj
        .add_interval(&IntervalSpec {
            tier_id: tier,
            start_seconds: 0.0,
            end_seconds: 0.5,
            label: Some("h".into()),
            ..Default::default()
        })
        .unwrap();

    let conn = Connection::open(root.join("corpus.db")).unwrap();
    let after: String = conn
        .query_row(
            "SELECT after FROM audit_log \
             WHERE table_name = 'annotation_interval' AND op = 'insert' \
             ORDER BY id DESC LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(after.contains("\"label\":\"h\""));
    assert!(after.contains("\"start_seconds\":0"));

    let _ = std::fs::remove_dir_all(&root);
}
