//! Integration tests for the D1 TextGrid round-trip surface: import +
//! export through a real `Project` with a real on-disk audio bundle.

use std::path::{Path, PathBuf};

use sadda_engine::{BundleSpec, IntervalSpec, PointSpec, Project, TierSpec, TierType};

fn unique_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "sadda_engine_d1_test_{}_{}",
        std::process::id(),
        name
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
        "sadda_d1_{}_{}.wav",
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
    let out = root.join("out.TextGrid");
    proj.export_textgrid(bundle_id, &out, None).unwrap();

    // Re-import into the same project under a new bundle (so the tier-name
    // UNIQUE constraint per bundle isn't violated).
    let wav2 = std::env::temp_dir().join(format!("sadda_d1_rt2_{}.wav", std::process::id()));
    write_short_wav(&wav2, 16_000, 1.5);
    let bundle2 = proj.add_bundle_with(&BundleSpec::new("b2"), &wav2).unwrap();
    let new_tier_ids = proj.import_textgrid(&out, bundle2).unwrap();
    assert_eq!(new_tier_ids.len(), 1);

    let intervals = proj.intervals(new_tier_ids[0]).unwrap();
    // 2 user intervals + 1 trailing pad to file_xmax.
    assert_eq!(intervals.len(), 3);
    assert_eq!(intervals[0].label.as_deref(), Some("h"));
    assert_eq!(intervals[0].extra.as_deref(), Some(r#"{"v":1}"#));
    assert_eq!(intervals[1].label.as_deref(), Some("e"));
    assert_eq!(intervals[2].label.as_deref(), Some(""));
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
    let out = root.join("out.TextGrid");
    proj.export_textgrid(bundle_id, &out, None).unwrap();

    let wav2 = std::env::temp_dir().join(format!("sadda_d1_rtp_{}.wav", std::process::id()));
    write_short_wav(&wav2, 16_000, 1.0);
    let bundle2 = proj.add_bundle_with(&BundleSpec::new("b2"), &wav2).unwrap();
    let new_tier_ids = proj.import_textgrid(&out, bundle2).unwrap();
    let points = proj.points(new_tier_ids[0]).unwrap();
    assert_eq!(points.len(), 2);
    assert_eq!(points[0].label.as_deref(), Some("click"));
    assert_eq!(points[1].label.as_deref(), Some("release"));
    assert_eq!(points[1].extra.as_deref(), Some(r#"{"force":12}"#));
    let _ = std::fs::remove_file(&wav2);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn export_skips_dense_tiers_silently() {
    let root = unique_dir("export_dense_skip");
    let (proj, bundle_id) = new_project_with_bundle(&root, 1.0);
    // One interval tier + one dense tier.
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
    let out = root.join("dense_skip.TextGrid");
    proj.export_textgrid(bundle_id, &out, None).unwrap();
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("size = 1"));
    assert!(text.contains("\"phones\""));
    assert!(!text.contains("\"f0\""));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn export_pads_gaps_with_empty_intervals() {
    let root = unique_dir("export_pad");
    let (proj, bundle_id) = new_project_with_bundle(&root, 2.0);
    let tier = proj
        .add_tier(&TierSpec::new(bundle_id, "phones", TierType::Interval))
        .unwrap();
    // Annotation from 0.5 to 1.0 (leaves a 0..0.5 gap and a 1.0..2.0 gap).
    proj.add_interval(&IntervalSpec {
        tier_id: tier,
        start_seconds: 0.5,
        end_seconds: 1.0,
        label: Some("a".into()),
        ..Default::default()
    })
    .unwrap();
    let out = root.join("pad.TextGrid");
    proj.export_textgrid(bundle_id, &out, None).unwrap();
    let text = std::fs::read_to_string(&out).unwrap();
    // Three intervals: leading pad, "a", trailing pad.
    assert!(text.contains("intervals: size = 3"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn import_records_processing_run_for_audit() {
    let root = unique_dir("import_processing_run");
    let (proj, bundle_id) = new_project_with_bundle(&root, 1.0);
    // Write a minimal TextGrid by hand.
    let tg_path = root.join("input.TextGrid");
    std::fs::write(
        &tg_path,
        r#"File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1.0
            text = "a"
"#,
    )
    .unwrap();
    let _ = proj.import_textgrid(&tg_path, bundle_id).unwrap();

    // Verify a processing_run row exists.
    let conn = rusqlite::Connection::open(root.join("corpus.db")).unwrap();
    let (processor, status): (String, String) = conn
        .query_row(
            "SELECT processor_id, status FROM processing_run \
             WHERE bundle_id = ?1 ORDER BY id DESC LIMIT 1",
            [bundle_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(processor, "sadda.io.textgrid.import");
    assert_eq!(status, "ok");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn empty_textgrid_imports_zero_tiers() {
    let root = unique_dir("import_empty");
    let (proj, bundle_id) = new_project_with_bundle(&root, 1.0);
    let tg_path = root.join("empty.TextGrid");
    std::fs::write(
        &tg_path,
        r#"File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <absent>
"#,
    )
    .unwrap();
    let tier_ids = proj.import_textgrid(&tg_path, bundle_id).unwrap();
    assert!(tier_ids.is_empty());
    let _ = std::fs::remove_dir_all(&root);
}
