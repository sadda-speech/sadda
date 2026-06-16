//! Integration tests for CSV / JSON annotation export + import
//! (`Project::export_csv` / `export_json` / `import_csv` / `import_json`).
//!
//! These exercise the full DB round-trip — export a bundle's annotations to a
//! file, re-import into a *second* bundle, and assert the honoured fields
//! survive. The pure CSV/JSON serialization is unit-tested in
//! `engine/src/io/tabular.rs`; this covers the corpus glue + provenance.

use std::path::{Path, PathBuf};

use sadda_engine::{BundleSpec, IntervalSpec, PointSpec, Project, TierSpec, TierType};

fn unique_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "sadda_engine_exim_test_{}_{}",
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
    for _ in 0..sample_rate / 4 {
        writer.write_sample(0i16).unwrap();
    }
    writer.finalize().unwrap();
}

/// Project with two bundles (`src`, `dst`) from distinct WAVs (add_bundle
/// copies the source keyed by filename, so the two can't share a path).
fn project_with_two_bundles(root: &Path) -> (Project, i64, i64) {
    let _ = std::fs::remove_dir_all(root);
    let proj = Project::create(root, "p").unwrap();
    let mk = |tag: &str| {
        let wav = std::env::temp_dir().join(format!(
            "sadda_exim_{}_{}_{}.wav",
            std::process::id(),
            root.file_name().unwrap().to_string_lossy(),
            tag,
        ));
        write_short_wav(&wav, 16_000);
        wav
    };
    let src_wav = mk("src");
    let dst_wav = mk("dst");
    let src = proj
        .add_bundle_with(&BundleSpec::new("src"), &src_wav)
        .unwrap();
    let dst = proj
        .add_bundle_with(&BundleSpec::new("dst"), &dst_wav)
        .unwrap();
    let _ = std::fs::remove_file(&src_wav);
    let _ = std::fs::remove_file(&dst_wav);
    (proj, src, dst)
}

/// Annotates `bundle_id` with an interval tier (incl. a CSV-hostile label)
/// and a point tier. Returns the two tier ids.
fn annotate(proj: &Project, bundle_id: i64) {
    let words = proj
        .add_tier(&TierSpec::new(bundle_id, "words", TierType::Interval))
        .unwrap();
    proj.add_interval(&IntervalSpec {
        tier_id: words,
        start_seconds: 0.0,
        end_seconds: 0.5,
        label: Some("hello".into()),
        ..Default::default()
    })
    .unwrap();
    proj.add_interval(&IntervalSpec {
        tier_id: words,
        start_seconds: 0.5,
        end_seconds: 1.0,
        // comma + quote + newline: the CSV quoting torture case.
        label: Some("a, b \"c\"\nd".into()),
        note: Some("a note".into()),
        extra: Some(r#"{"k":1}"#.into()),
        ..Default::default()
    })
    .unwrap();
    let pulses = proj
        .add_tier(&TierSpec::new(bundle_id, "pulses", TierType::Point))
        .unwrap();
    proj.add_point(&PointSpec {
        tier_id: pulses,
        time_seconds: 0.25,
        label: Some("p".into()),
        ..Default::default()
    })
    .unwrap();
}

#[test]
fn csv_export_import_round_trips_through_db() {
    let root = unique_dir("csv_round_trip");
    let (proj, src, dst) = project_with_two_bundles(&root);
    annotate(&proj, src);

    let csv = root.join("ann.csv");
    proj.export_csv(src, &csv, None).unwrap();

    let new_tiers = proj.import_csv(&csv, dst).unwrap();
    assert_eq!(new_tiers.len(), 2);

    let tiers = proj.tiers(Some(dst)).unwrap();
    let words = tiers.iter().find(|t| t.name == "words").unwrap();
    let pulses = tiers.iter().find(|t| t.name == "pulses").unwrap();
    assert_eq!(words.r#type, TierType::Interval);
    assert_eq!(pulses.r#type, TierType::Point);

    let ivs = proj.intervals(words.id).unwrap();
    assert_eq!(ivs.len(), 2);
    assert_eq!(ivs[0].label.as_deref(), Some("hello"));
    // The hostile label and note/extra survive the quote round-trip.
    assert_eq!(ivs[1].label.as_deref(), Some("a, b \"c\"\nd"));
    assert_eq!(ivs[1].note.as_deref(), Some("a note"));
    assert_eq!(ivs[1].extra.as_deref(), Some(r#"{"k":1}"#));

    let pts = proj.points(pulses.id).unwrap();
    assert_eq!(pts.len(), 1);
    assert!((pts[0].time_seconds - 0.25).abs() < 1e-9);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn json_export_import_round_trips_through_db() {
    let root = unique_dir("json_round_trip");
    let (proj, src, dst) = project_with_two_bundles(&root);
    annotate(&proj, src);

    let json = root.join("ann.json");
    proj.export_json(src, &json, None).unwrap();

    // The file is structured (bundle + tiers), and extra embeds as an object.
    let text = std::fs::read_to_string(&json).unwrap();
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(doc["bundle"]["name"], "src");
    assert_eq!(doc["tiers"][0]["intervals"][1]["extra"]["k"], 1);

    let new_tiers = proj.import_json(&json, dst).unwrap();
    assert_eq!(new_tiers.len(), 2);
    let tiers = proj.tiers(Some(dst)).unwrap();
    let words = tiers.iter().find(|t| t.name == "words").unwrap();
    let ivs = proj.intervals(words.id).unwrap();
    assert_eq!(ivs[1].label.as_deref(), Some("a, b \"c\"\nd"));
    assert_eq!(ivs[1].extra.as_deref(), Some(r#"{"k":1}"#));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn tier_ids_filter_limits_the_export() {
    let root = unique_dir("tier_filter");
    let (proj, src, dst) = project_with_two_bundles(&root);
    annotate(&proj, src);

    // Export only the "words" interval tier.
    let words_src = proj
        .tiers(Some(src))
        .unwrap()
        .into_iter()
        .find(|t| t.name == "words")
        .unwrap()
        .id;
    let csv = root.join("words_only.csv");
    proj.export_csv(src, &csv, Some(&[words_src])).unwrap();

    let new_tiers = proj.import_csv(&csv, dst).unwrap();
    assert_eq!(new_tiers.len(), 1);
    let tiers = proj.tiers(Some(dst)).unwrap();
    assert_eq!(tiers.len(), 1);
    assert_eq!(tiers[0].name, "words");

    let _ = std::fs::remove_dir_all(&root);
}
