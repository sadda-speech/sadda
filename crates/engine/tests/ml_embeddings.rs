//! E12b-2 — `Project::extract_embeddings` writes a `continuous_vector` tier
//! and records an `ml_model` processing run. Behind the `ml` feature;
//! ORT-gated so it skips cleanly without ONNX Runtime (CI stays green).
#![cfg(feature = "ml")]

use std::path::{Path, PathBuf};

use sadda_engine::{BundleSpec, EngineError, Model, Project, TierType};

fn write_silent_wav(path: &Path, sr: u32, duration_s: f32) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for _ in 0..(duration_s * sr as f32) as usize {
        w.write_sample(0_i16).unwrap();
    }
    w.finalize().unwrap();
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/ml_fixtures")
        .join(name)
}

#[test]
fn extract_embeddings_writes_continuous_vector_tier_with_provenance() {
    let root = std::env::temp_dir().join(format!("sadda_emb_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let proj = Project::create(&root, "emb").unwrap();
    let wav = root.with_extension("src.wav");
    write_silent_wav(&wav, 16_000, 1.0);
    let bundle_id = proj.add_bundle_with(&BundleSpec::new("b"), &wav).unwrap();
    let _ = std::fs::remove_file(&wav);

    let model = Model::from_dir(fixture("waveform-embed")).unwrap();
    match proj.extract_embeddings(bundle_id, &model, "ssl_embeddings") {
        Ok(tier_id) => {
            // The tier exists, is a continuous_vector, and reads back as
            // frames × 8 (the fixture's DIMS).
            let tier = proj.get_tier(tier_id).unwrap();
            assert_eq!(tier.r#type, TierType::ContinuousVector);
            let emb = proj.read_continuous_vector(tier_id).unwrap();
            assert_eq!(emb.ncols(), 8);
            assert!(emb.nrows() > 0);

            // Provenance: an ml_model run naming the model was recorded.
            let runs = proj.processing_runs(bundle_id).unwrap();
            assert!(
                runs.iter()
                    .any(|r| r.kind == "ml_model" && r.processor_id == "fixture/waveform-embed"),
                "no ml_model ProcessingRun recorded; got {:?}",
                runs.iter()
                    .map(|r| (&r.kind, &r.processor_id))
                    .collect::<Vec<_>>()
            );
        }
        Err(EngineError::Ml(msg)) => {
            eprintln!("extract_embeddings test skipped (ORT unavailable): {msg}");
        }
        Err(e) => panic!("unexpected error: {e}"),
    }
    let _ = std::fs::remove_dir_all(&root);
}
