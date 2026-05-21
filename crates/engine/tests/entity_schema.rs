//! Integration tests for the V3 entity schema (Phase 1 slice B1):
//! Speaker / Session / Bundle-FK round-trips and the trigger-based audit log.

use std::path::PathBuf;

use rusqlite::Connection;
use sadda_engine::{BundleSpec, Project, SessionSpec, SpeakerSpec};

fn unique_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "sadda_engine_b1_test_{}_{}",
        std::process::id(),
        name
    ));
    p
}

fn write_short_wav(path: &std::path::Path, sample_rate: u32) {
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

#[test]
fn speaker_round_trip_through_add_and_list() {
    let root = unique_dir("speaker_round_trip");
    let _ = std::fs::remove_dir_all(&root);
    let proj = Project::create(&root, "p").unwrap();

    let alice_id = proj
        .add_speaker(&SpeakerSpec {
            name: "Alice".into(),
            sex: Some("f".into()),
            birth_year: Some(1990),
            ..Default::default()
        })
        .unwrap();
    let _bob_id = proj.add_speaker(&SpeakerSpec::new("Bob")).unwrap();

    let all = proj.speakers().unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].name, "Alice");
    assert_eq!(all[0].sex.as_deref(), Some("f"));
    assert_eq!(all[0].birth_year, Some(1990));
    assert_eq!(all[1].name, "Bob");
    assert!(all[1].sex.is_none());

    let alice = proj.get_speaker(alice_id).unwrap();
    assert_eq!(alice.name, "Alice");
    assert!(!alice.created_at.is_empty());

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn session_round_trip_through_add_and_list() {
    let root = unique_dir("session_round_trip");
    let _ = std::fs::remove_dir_all(&root);
    let proj = Project::create(&root, "p").unwrap();

    let s1 = proj
        .add_session(&SessionSpec {
            name: "lab_session_1".into(),
            location: Some("lab_b".into()),
            started_at: Some("2026-05-21T14:00:00Z".into()),
            ..Default::default()
        })
        .unwrap();
    let sessions = proj.sessions().unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].id, s1);
    assert_eq!(sessions[0].location.as_deref(), Some("lab_b"));
    assert_eq!(
        sessions[0].started_at.as_deref(),
        Some("2026-05-21T14:00:00Z")
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn bundle_carries_session_and_speaker_fks() {
    let root = unique_dir("bundle_fks");
    let _ = std::fs::remove_dir_all(&root);
    let proj = Project::create(&root, "p").unwrap();

    let speaker_id = proj.add_speaker(&SpeakerSpec::new("Alice")).unwrap();
    let session_id = proj.add_session(&SessionSpec::new("s1")).unwrap();

    let wav = std::env::temp_dir().join(format!("sadda_b1_bundle_fk_{}.wav", std::process::id()));
    write_short_wav(&wav, 16_000);

    let bundle_id = proj
        .add_bundle_with(
            &BundleSpec {
                name: "greeting".into(),
                session_id: Some(session_id),
                speaker_id: Some(speaker_id),
                extra: Some(r#"{"take":2}"#.into()),
            },
            &wav,
        )
        .unwrap();

    let bundles = proj.bundles().unwrap();
    assert_eq!(bundles.len(), 1);
    assert_eq!(bundles[0].id, bundle_id);
    assert_eq!(bundles[0].session_id, Some(session_id));
    assert_eq!(bundles[0].speaker_id, Some(speaker_id));
    assert_eq!(bundles[0].extra.as_deref(), Some(r#"{"take":2}"#));

    let _ = std::fs::remove_file(&wav);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn add_speaker_writes_insert_row_to_audit_log() {
    let root = unique_dir("audit_speaker_insert");
    let _ = std::fs::remove_dir_all(&root);
    let proj = Project::create(&root, "p").unwrap();

    let speaker_id = proj.add_speaker(&SpeakerSpec::new("Alice")).unwrap();

    let conn = Connection::open(root.join("corpus.db")).unwrap();
    let row: (String, i64, String, Option<String>, String) = conn
        .query_row(
            "SELECT user, row_id, op, before, after FROM audit_log \
             WHERE table_name = 'speaker' ORDER BY id DESC LIMIT 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .unwrap();
    assert_eq!(row.0, "local");
    assert_eq!(row.1, speaker_id);
    assert_eq!(row.2, "insert");
    assert!(row.3.is_none());
    assert!(row.4.contains("\"name\":\"Alice\""));

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn updating_speaker_writes_before_and_after_to_audit_log() {
    let root = unique_dir("audit_speaker_update");
    let _ = std::fs::remove_dir_all(&root);
    let proj = Project::create(&root, "p").unwrap();
    let id = proj.add_speaker(&SpeakerSpec::new("Alice")).unwrap();

    // Force a direct update so the test doesn't depend on a public mutate API.
    {
        let conn = Connection::open(root.join("corpus.db")).unwrap();
        conn.execute(
            "UPDATE speaker SET sex = ?1 WHERE id = ?2",
            rusqlite::params!["f", id],
        )
        .unwrap();
    }

    let conn = Connection::open(root.join("corpus.db")).unwrap();
    let (op, before, after): (String, Option<String>, Option<String>) = conn
        .query_row(
            "SELECT op, before, after FROM audit_log \
             WHERE table_name = 'speaker' AND op = 'update' \
             ORDER BY id DESC LIMIT 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(op, "update");
    let before = before.expect("update should write a before payload");
    let after = after.expect("update should write an after payload");
    // before.sex was NULL; after.sex is "f".
    assert!(before.contains("\"sex\":null"), "before: {before}");
    assert!(after.contains("\"sex\":\"f\""), "after: {after}");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn deleting_speaker_writes_before_only_to_audit_log() {
    let root = unique_dir("audit_speaker_delete");
    let _ = std::fs::remove_dir_all(&root);
    let proj = Project::create(&root, "p").unwrap();
    let id = proj.add_speaker(&SpeakerSpec::new("Alice")).unwrap();

    {
        let conn = Connection::open(root.join("corpus.db")).unwrap();
        conn.execute("DELETE FROM speaker WHERE id = ?1", [id])
            .unwrap();
    }

    let conn = Connection::open(root.join("corpus.db")).unwrap();
    let (op, before, after): (String, Option<String>, Option<String>) = conn
        .query_row(
            "SELECT op, before, after FROM audit_log \
             WHERE table_name = 'speaker' AND op = 'delete'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(op, "delete");
    assert!(before.is_some());
    assert!(after.is_none());

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn set_audit_user_flows_through_to_subsequent_rows() {
    let root = unique_dir("audit_user_flow");
    let _ = std::fs::remove_dir_all(&root);
    let proj = Project::create(&root, "p").unwrap();

    assert_eq!(proj.audit_user().unwrap(), "local");
    proj.set_audit_user("alice").unwrap();
    assert_eq!(proj.audit_user().unwrap(), "alice");

    let _ = proj.add_speaker(&SpeakerSpec::new("BobAfter")).unwrap();

    let conn = Connection::open(root.join("corpus.db")).unwrap();
    let user: String = conn
        .query_row(
            "SELECT user FROM audit_log \
             WHERE table_name = 'speaker' ORDER BY id DESC LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(user, "alice");

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn bundle_insert_writes_audit_with_session_and_speaker_fields() {
    let root = unique_dir("audit_bundle_extension");
    let _ = std::fs::remove_dir_all(&root);
    let proj = Project::create(&root, "p").unwrap();

    let speaker_id = proj.add_speaker(&SpeakerSpec::new("Alice")).unwrap();
    let session_id = proj.add_session(&SessionSpec::new("s")).unwrap();
    let wav =
        std::env::temp_dir().join(format!("sadda_b1_audit_bundle_{}.wav", std::process::id()));
    write_short_wav(&wav, 16_000);
    let _ = proj
        .add_bundle_with(
            &BundleSpec {
                name: "g".into(),
                session_id: Some(session_id),
                speaker_id: Some(speaker_id),
                extra: None,
            },
            &wav,
        )
        .unwrap();

    let conn = Connection::open(root.join("corpus.db")).unwrap();
    let after: String = conn
        .query_row(
            "SELECT after FROM audit_log \
             WHERE table_name = 'bundle' AND op = 'insert' ORDER BY id DESC LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(after.contains(&format!("\"session_id\":{session_id}")));
    assert!(after.contains(&format!("\"speaker_id\":{speaker_id}")));

    let _ = std::fs::remove_file(&wav);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn project_table_is_not_audited() {
    // The singleton `project` row is created in Project::create; no trigger
    // should fire and no audit_log row should reference table_name='project'.
    let root = unique_dir("project_not_audited");
    let _ = std::fs::remove_dir_all(&root);
    let _proj = Project::create(&root, "p").unwrap();

    let conn = Connection::open(root.join("corpus.db")).unwrap();
    let n: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM audit_log WHERE table_name = 'project'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 0);

    let _ = std::fs::remove_dir_all(&root);
}
