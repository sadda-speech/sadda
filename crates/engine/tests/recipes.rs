//! Integration tests for the F1 recipe surface. Exercises
//! `Project::start_recipe` / `end_recipe`, the `processing_run.recipe_run_id`
//! wiring through `import_textgrid` + `commit_recording`, and the
//! recipe enumeration API.

use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use sadda_engine::{BundleSpec, LiveConfig, LiveSession, Project};

fn unique_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "sadda_engine_f1_test_{}_{}",
        std::process::id(),
        name
    ));
    p
}

fn write_silent_wav(path: &Path, sr: u32, duration_s: f32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    let n = (duration_s * sr as f32) as usize;
    for _ in 0..n {
        w.write_sample(0_i16).unwrap();
    }
    w.finalize().unwrap();
}

fn new_project_with_bundle(root: &Path, duration_s: f32) -> (Project, i64) {
    let _ = std::fs::remove_dir_all(root);
    let proj = Project::create(root, "recipes_test").unwrap();
    // Use a temp WAV path scoped to *this* project root rather than a
    // shared /tmp/sadda_<pid>.wav: cargo test runs tests in parallel and
    // they were racing on the shared filename.
    let wav = root.with_extension("source.wav");
    write_silent_wav(&wav, 16_000, duration_s);
    let bundle_id = proj.add_bundle_with(&BundleSpec::new("b"), &wav).unwrap();
    let _ = std::fs::remove_file(&wav);
    (proj, bundle_id)
}

#[test]
fn start_and_end_recipe_round_trip() {
    let root = unique_dir("start_end");
    let proj = Project::create(&root, "p").unwrap();
    let id = proj
        .start_recipe("test_recipe", Some(r#"{"k":"v"}"#))
        .unwrap();
    assert_eq!(proj.current_recipe_id(), Some(id));
    proj.end_recipe(id, "ok", None).unwrap();
    assert!(proj.current_recipe_id().is_none());

    let recipes = proj.recipes().unwrap();
    assert_eq!(recipes.len(), 1);
    assert_eq!(recipes[0].name, "test_recipe");
    assert_eq!(recipes[0].status, "ok");
    assert!(recipes[0].completed_at.is_some());
    assert_eq!(recipes[0].parameters.as_deref(), Some(r#"{"k":"v"}"#));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn second_start_while_active_errors() {
    let root = unique_dir("double_start");
    let proj = Project::create(&root, "p").unwrap();
    let _ = proj.start_recipe("first", None).unwrap();
    let err = proj.start_recipe("second", None).unwrap_err();
    assert!(format!("{err}").contains("already active"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn unique_name_constraint_holds_across_recipes() {
    let root = unique_dir("unique_name");
    let proj = Project::create(&root, "p").unwrap();
    let id = proj.start_recipe("dup", None).unwrap();
    proj.end_recipe(id, "ok", None).unwrap();
    let err = proj.start_recipe("dup", None).unwrap_err();
    // Surfaces as a Sqlite UNIQUE-constraint error wrapped in EngineError::Sqlite.
    let msg = format!("{err}");
    assert!(
        msg.contains("UNIQUE") || msg.contains("constraint"),
        "got: {msg}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn import_textgrid_inside_recipe_links_processing_run() {
    let root = unique_dir("link_textgrid");
    let (proj, bundle_id) = new_project_with_bundle(&root, 1.0);
    let tg = root.join("input.TextGrid");
    std::fs::write(
        &tg,
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
    let recipe_id = proj.start_recipe("rcp", None).unwrap();
    proj.import_textgrid(&tg, bundle_id).unwrap();
    proj.end_recipe(recipe_id, "ok", None).unwrap();

    let runs = proj.processing_runs_for_recipe(recipe_id).unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].processor_id, "sadda.io.textgrid.import");
    assert_eq!(runs[0].bundle_id, bundle_id);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn live_commit_inside_recipe_links_processing_run() {
    let root = unique_dir("link_live");
    let proj = Project::create(&root, "p").unwrap();
    let recipe_id = proj.start_recipe("live_rec", None).unwrap();

    let cfg = LiveConfig {
        sample_rate: 16_000,
        channels: 1,
        ..LiveConfig::default()
    };
    let (mut session, _results) = LiveSession::start(&root, cfg.clone()).unwrap();
    let samples: Vec<f32> = (0..16_000 / 2)
        .map(|i| (i as f32 / 1000.0).sin() * 0.3)
        .collect();
    session.push_samples(&samples);
    thread::sleep(Duration::from_millis(80));
    let stopped = session.stop().unwrap();
    let bundle_id = proj
        .commit_recording(stopped, "take1", r#"{"device":"test"}"#)
        .unwrap();
    proj.end_recipe(recipe_id, "ok", None).unwrap();

    let runs = proj.processing_runs_for_recipe(recipe_id).unwrap();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].processor_id, "sadda.live");
    assert_eq!(runs[0].bundle_id, bundle_id);
    assert_eq!(runs[0].kind, "live_recording");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn processing_run_outside_recipe_has_null_recipe_run_id() {
    let root = unique_dir("outside");
    let (proj, bundle_id) = new_project_with_bundle(&root, 1.0);
    let tg = root.join("input.TextGrid");
    std::fs::write(
        &tg,
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
    proj.import_textgrid(&tg, bundle_id).unwrap();

    // Query for any processing_run row with non-null recipe_run_id.
    let conn = rusqlite::Connection::open(root.join("corpus.db")).unwrap();
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM processing_run WHERE recipe_run_id IS NOT NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 0);

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn recipe_by_name_returns_the_row() {
    let root = unique_dir("by_name");
    let proj = Project::create(&root, "p").unwrap();
    let id = proj.start_recipe("findme", None).unwrap();
    proj.end_recipe(id, "ok", None).unwrap();
    let r = proj.recipe_by_name("findme").unwrap();
    assert_eq!(r.id, id);
    assert_eq!(r.status, "ok");
    let _ = std::fs::remove_dir_all(&root);
}
