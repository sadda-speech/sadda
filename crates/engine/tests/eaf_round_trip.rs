//! Integration tests for the D2 EAF round-trip surface: import + export
//! through a real `Project` with a real on-disk audio bundle. Mirrors
//! `textgrid_round_trip.rs` in structure; the extra coverage is tier
//! hierarchy preservation via `PARENT_REF` (D2's headline feature).

use std::path::{Path, PathBuf};

use sadda_engine::{
    BundleSpec, IntervalSpec, PointSpec, Project, ReferenceSpec, TierSpec, TierType,
};

fn unique_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "sadda_engine_d2_test_{}_{}",
        std::process::id(),
        name,
    ));
    p
}

fn write_short_wav(path: &Path, sample_rate: u32, duration_seconds: f32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    let n = (duration_seconds * sample_rate as f32) as usize;
    for _ in 0..n {
        writer.write_sample(0_i16).unwrap();
    }
    writer.finalize().unwrap();
}

fn new_project_with_bundle(root: &Path, duration_seconds: f32) -> (Project, i64) {
    let _ = std::fs::remove_dir_all(root);
    let proj = Project::create(root, "p").unwrap();
    let wav = std::env::temp_dir().join(format!(
        "sadda_d2_{}_{}.wav",
        std::process::id(),
        root.file_name().unwrap().to_string_lossy(),
    ));
    write_short_wav(&wav, 16_000, duration_seconds);
    let bundle_id = proj.add_bundle_with(&BundleSpec::new("b"), &wav).unwrap();
    let _ = std::fs::remove_file(&wav);
    (proj, bundle_id)
}

#[test]
fn export_then_import_round_trips_interval_tier() {
    let root = unique_dir("rt_interval");
    let (proj, bundle_id) = new_project_with_bundle(&root, 1.5);
    let tier = proj
        .add_tier(&TierSpec::new(bundle_id, "phones", TierType::Interval))
        .unwrap();
    proj.add_interval(&IntervalSpec {
        tier_id: tier,
        start_seconds: 0.0,
        end_seconds: 0.5,
        label: Some("h".into()),
        extra: Some(r#"{"v":1}"#.into()),
        ..Default::default()
    })
    .unwrap();
    proj.add_interval(&IntervalSpec {
        tier_id: tier,
        start_seconds: 0.5,
        end_seconds: 1.0,
        label: Some("e".into()),
        ..Default::default()
    })
    .unwrap();
    let out = root.join("out.eaf");
    proj.export_eaf(bundle_id, &out, None).unwrap();

    let wav2 = std::env::temp_dir().join(format!("sadda_d2_rt2_{}.wav", std::process::id()));
    write_short_wav(&wav2, 16_000, 1.5);
    let bundle2 = proj.add_bundle_with(&BundleSpec::new("b2"), &wav2).unwrap();
    let new_tier_ids = proj.import_eaf(&out, bundle2).unwrap();
    assert_eq!(new_tier_ids.len(), 1);

    let intervals = proj.intervals(new_tier_ids[0]).unwrap();
    assert_eq!(intervals.len(), 2);
    assert_eq!(intervals[0].label.as_deref(), Some("h"));
    assert_eq!(intervals[0].extra.as_deref(), Some(r#"{"v":1}"#));
    assert!((intervals[0].start_seconds - 0.0).abs() < 1e-6);
    assert!((intervals[0].end_seconds - 0.5).abs() < 1e-6);
    assert_eq!(intervals[1].label.as_deref(), Some("e"));
    let _ = std::fs::remove_file(&wav2);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn export_then_import_round_trips_point_tier() {
    let root = unique_dir("rt_point");
    let (proj, bundle_id) = new_project_with_bundle(&root, 1.0);
    let tier = proj
        .add_tier(&TierSpec::new(bundle_id, "events", TierType::Point))
        .unwrap();
    proj.add_point(&PointSpec {
        tier_id: tier,
        time_seconds: 0.25,
        label: Some("click".into()),
        ..Default::default()
    })
    .unwrap();
    proj.add_point(&PointSpec {
        tier_id: tier,
        time_seconds: 0.75,
        label: Some("release".into()),
        extra: Some(r#"{"force":12}"#.into()),
        ..Default::default()
    })
    .unwrap();
    let out = root.join("out.eaf");
    proj.export_eaf(bundle_id, &out, None).unwrap();

    let wav2 = std::env::temp_dir().join(format!("sadda_d2_rtp_{}.wav", std::process::id()));
    write_short_wav(&wav2, 16_000, 1.0);
    let bundle2 = proj.add_bundle_with(&BundleSpec::new("b2"), &wav2).unwrap();
    let new_tier_ids = proj.import_eaf(&out, bundle2).unwrap();
    let points = proj.points(new_tier_ids[0]).unwrap();
    assert_eq!(points.len(), 2);
    assert_eq!(points[0].label.as_deref(), Some("click"));
    assert!((points[0].time_seconds - 0.25).abs() < 2e-3);
    assert_eq!(points[1].label.as_deref(), Some("release"));
    assert_eq!(points[1].extra.as_deref(), Some(r#"{"force":12}"#));
    let _ = std::fs::remove_file(&wav2);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn tier_hierarchy_survives_round_trip() {
    let root = unique_dir("rt_hierarchy");
    let (proj, bundle_id) = new_project_with_bundle(&root, 2.0);
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
    proj.add_interval(&IntervalSpec {
        tier_id: words,
        start_seconds: 0.0,
        end_seconds: 1.0,
        label: Some("hi".into()),
        ..Default::default()
    })
    .unwrap();
    let parent_iv = proj
        .add_interval(&IntervalSpec {
            tier_id: words,
            start_seconds: 1.0,
            end_seconds: 2.0,
            label: Some("there".into()),
            ..Default::default()
        })
        .unwrap();
    proj.add_interval(&IntervalSpec {
        tier_id: phones,
        start_seconds: 1.0,
        end_seconds: 1.5,
        label: Some("ð".into()),
        parent_annotation_id: Some(parent_iv),
        ..Default::default()
    })
    .unwrap();

    let out = root.join("hierarchy.eaf");
    proj.export_eaf(bundle_id, &out, None).unwrap();

    // Re-import into a fresh bundle.
    let wav2 = std::env::temp_dir().join(format!("sadda_d2_rth_{}.wav", std::process::id()));
    write_short_wav(&wav2, 16_000, 2.0);
    let bundle2 = proj.add_bundle_with(&BundleSpec::new("b2"), &wav2).unwrap();
    let new_tier_ids = proj.import_eaf(&out, bundle2).unwrap();
    assert_eq!(new_tier_ids.len(), 2);

    // Look the new tiers up by name. The order of new_tier_ids depends on
    // topo sort (parents first), so we resolve by name.
    let all_tiers = proj.tiers(Some(bundle2)).unwrap();
    let new_words = all_tiers.iter().find(|t| t.name == "words").unwrap();
    let new_phones = all_tiers.iter().find(|t| t.name == "phones").unwrap();
    assert_eq!(new_words.parent_id, None);
    assert_eq!(new_phones.parent_id, Some(new_words.id));

    let _ = std::fs::remove_file(&wav2);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn reference_tier_round_trips_via_symbolic_association() {
    let root = unique_dir("rt_reference");
    let (proj, bundle_id) = new_project_with_bundle(&root, 1.0);
    let parent_tier = proj
        .add_tier(&TierSpec::new(bundle_id, "words", TierType::Interval))
        .unwrap();
    let parent_iv = proj
        .add_interval(&IntervalSpec {
            tier_id: parent_tier,
            start_seconds: 0.0,
            end_seconds: 1.0,
            label: Some("hello".into()),
            ..Default::default()
        })
        .unwrap();
    let ref_tier = proj
        .add_tier(&TierSpec {
            bundle_id,
            name: "lex".into(),
            r#type: Some(TierType::Reference),
            parent_id: Some(parent_tier),
            cardinality: Some("one_to_many".into()),
            ..Default::default()
        })
        .unwrap();
    proj.add_reference(&ReferenceSpec {
        tier_id: ref_tier,
        target_kind: "annotation".into(),
        target_id: parent_iv,
        label: Some("greeting".into()),
        parent_annotation_id: Some(parent_iv),
        ..Default::default()
    })
    .unwrap();

    let out = root.join("ref.eaf");
    proj.export_eaf(bundle_id, &out, None).unwrap();

    let wav2 = std::env::temp_dir().join(format!("sadda_d2_rtr_{}.wav", std::process::id()));
    write_short_wav(&wav2, 16_000, 1.0);
    let bundle2 = proj.add_bundle_with(&BundleSpec::new("b2"), &wav2).unwrap();
    let _ = proj.import_eaf(&out, bundle2).unwrap();

    let all = proj.tiers(Some(bundle2)).unwrap();
    let new_ref_tier = all.iter().find(|t| t.name == "lex").unwrap();
    assert!(matches!(new_ref_tier.r#type, TierType::Reference));
    let refs = proj.references_for(new_ref_tier.id).unwrap();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].label.as_deref(), Some("greeting"));
    assert_eq!(refs[0].target_kind, "annotation");
    let _ = std::fs::remove_file(&wav2);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn export_skips_dense_tiers_silently() {
    let root = unique_dir("export_dense_skip");
    let (proj, bundle_id) = new_project_with_bundle(&root, 1.0);
    let int_tier = proj
        .add_tier(&TierSpec::new(bundle_id, "phones", TierType::Interval))
        .unwrap();
    proj.add_interval(&IntervalSpec {
        tier_id: int_tier,
        start_seconds: 0.0,
        end_seconds: 1.0,
        label: Some("a".into()),
        ..Default::default()
    })
    .unwrap();
    let _ = proj
        .add_tier(&TierSpec::new(bundle_id, "f0", TierType::ContinuousNumeric))
        .unwrap();
    let out = root.join("dense_skip.eaf");
    proj.export_eaf(bundle_id, &out, None).unwrap();
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("TIER_ID=\"phones\""));
    assert!(!text.contains("TIER_ID=\"f0\""));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn import_records_processing_run_for_audit() {
    let root = unique_dir("import_processing_run");
    let (proj, bundle_id) = new_project_with_bundle(&root, 1.0);
    let eaf_path = root.join("input.eaf");
    std::fs::write(
        &eaf_path,
        r#"<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="" DATE="2025-01-01T00:00:00Z" FORMAT="2.8" VERSION="2.8">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"/>
    <TIME_ORDER>
        <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>
        <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="1000"/>
    </TIME_ORDER>
    <TIER LINGUISTIC_TYPE_REF="default" TIER_ID="phones">
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
                <ANNOTATION_VALUE>a</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
    </TIER>
    <LINGUISTIC_TYPE LINGUISTIC_TYPE_ID="default" TIME_ALIGNABLE="true"/>
</ANNOTATION_DOCUMENT>
"#,
    )
    .unwrap();
    let _ = proj.import_eaf(&eaf_path, bundle_id).unwrap();

    let conn = rusqlite::Connection::open(root.join("corpus.db")).unwrap();
    let (processor, status): (String, String) = conn
        .query_row(
            "SELECT processor_id, status FROM processing_run \
             WHERE bundle_id = ?1 ORDER BY id DESC LIMIT 1",
            [bundle_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(processor, "sadda.io.eaf.import");
    assert_eq!(status, "ok");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn empty_eaf_imports_zero_tiers() {
    let root = unique_dir("import_empty");
    let (proj, bundle_id) = new_project_with_bundle(&root, 1.0);
    let path = root.join("empty.eaf");
    std::fs::write(
        &path,
        r#"<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT FORMAT="2.8" VERSION="2.8">
    <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"/>
    <TIME_ORDER/>
</ANNOTATION_DOCUMENT>
"#,
    )
    .unwrap();
    let tier_ids = proj.import_eaf(&path, bundle_id).unwrap();
    assert!(tier_ids.is_empty());
    let _ = std::fs::remove_dir_all(&root);
}
