//! S6 — headless documentation-image renderer (the automation spine).
//!
//! Drives the *real* [`SaddaApp`](crate::SaddaApp) offscreen through
//! `egui_kittest` + wgpu — the same egui/wgpu stack the live app renders with —
//! so documentation images can't drift from what users see (the anti-drift
//! argument in the 2026-07-02 DEVLOG entry). No window, no display server.
//!
//! This module is `#[cfg(test)]`: it reaches `SaddaApp`'s crate-private state
//! (it lives in the `main.rs` binary, not a library), and the doc images double
//! as `egui_kittest` snapshot goldens — a UI change that would alter one fails
//! CI until it's regenerated and reviewed.
//!
//! ## Running
//!
//! The render tests are `#[ignore]` because the default wgpu adapter under
//! WSL/headless can segfault; they need a **software Vulkan** adapter
//! (lavapipe). [`configure_headless_gpu`] points wgpu at lavapipe when it finds
//! the ICD. Run them explicitly:
//!
//! ```text
//! VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json \
//!   WGPU_BACKEND=vulkan cargo test -p sadda-app --bins -- --ignored doc_render
//! ```
//!
//! (S7 wraps this in a `just docs-images` recipe; S8 wires the CI adapter.)

#[cfg(test)]
mod tests {
    use crate::{CaptureTarget, SaddaApp};
    use eframe::egui;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    /// Point wgpu at a software-Vulkan (lavapipe) adapter when present, so the
    /// headless render doesn't touch the crash-prone WSL GPU passthrough. A
    /// no-op if the caller already set the env (CI does).
    fn configure_headless_gpu() {
        if std::env::var_os("WGPU_BACKEND").is_none() {
            // SAFETY: single-threaded test setup, before any wgpu init.
            unsafe { std::env::set_var("WGPU_BACKEND", "vulkan") };
        }
        if std::env::var_os("VK_ICD_FILENAMES").is_none() {
            for p in [
                "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
                "/usr/local/share/vulkan/icd.d/lvp_icd.x86_64.json",
            ] {
                if Path::new(p).exists() {
                    unsafe { std::env::set_var("VK_ICD_FILENAMES", p) };
                    break;
                }
            }
        }
    }

    /// Build a harness driving the real app at a fixed size + 1.0 device scale,
    /// so output pixels equal logical points (deterministic geometry).
    fn doc_harness<'a>(size: egui::Vec2) -> egui_kittest::Harness<'a, SaddaApp> {
        configure_headless_gpu();
        egui_kittest::Harness::builder()
            .with_size(size)
            .with_pixels_per_point(1.0)
            .build_eframe(|cc| SaddaApp::new(cc))
    }

    /// A short, real, clean-licensed fixture WAV from the engine test corpus.
    fn fixture_wav() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../engine/tests/clinical/fixtures/hnr_high_120hz.wav")
    }

    /// Fresh scratch dir for a throwaway project (per process + label).
    fn scratch_dir(label: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("sadda-doc-render-{}-{label}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// Load a one-bundle project into the app, headlessly (post-dialog handlers
    /// take paths directly, so no GUI needed).
    fn load_fixture_bundle(app: &mut SaddaApp, project_root: PathBuf) {
        app.create_project_at(project_root);
        app.add_bundle_from_wav(fixture_wav()); // auto-selects the new bundle
        assert!(
            app.selected_bundle_id.is_some(),
            "fixture bundle should be selected"
        );
    }

    /// Step frames until the async DSP for the shown bundle has landed (or a
    /// timeout). The analysis runs on worker threads and is polled each frame,
    /// so we alternate stepping with short sleeps to let the workers finish.
    fn settle_analysis(h: &mut egui_kittest::Harness<'_, SaddaApp>, timeout: Duration) {
        let start = Instant::now();
        loop {
            h.step();
            let ready = h.state().active_spectrogram.is_some();
            if ready || start.elapsed() > timeout {
                break;
            }
            std::thread::sleep(Duration::from_millis(15));
        }
        // A few more frames so the just-installed textures/tracks paint.
        h.run();
    }

    /// Crop a rendered full-window image to a rect (points == pixels at ppp
    /// 1.0), clamped to the image. `None` for a zero-area rect.
    fn crop_rect(img: &image::RgbaImage, rect: egui::Rect) -> Option<image::RgbaImage> {
        if rect.width() < 1.0 || rect.height() < 1.0 {
            return None;
        }
        let (iw, ih) = (img.width(), img.height());
        let x = (rect.min.x.max(0.0) as u32).min(iw);
        let y = (rect.min.y.max(0.0) as u32).min(ih);
        let w = ((rect.width().round() as u32).min(iw.saturating_sub(x))).max(1);
        let h = ((rect.height().round() as u32).min(ih.saturating_sub(y))).max(1);
        Some(image::imageops::crop_imm(img, x, y, w, h).to_image())
    }

    /// Crop to a named region's registered rect.
    fn crop_named(
        img: &image::RgbaImage,
        app: &SaddaApp,
        target: CaptureTarget,
    ) -> Option<image::RgbaImage> {
        crop_rect(img, app.capture_rect_for(target)?)
    }

    /// Crop for a recipe's `capture` field: a named target, or `rect:x,y,w,h`.
    fn crop_for_capture(
        full: &image::RgbaImage,
        app: &SaddaApp,
        capture: &str,
    ) -> Option<image::RgbaImage> {
        if let Some(rest) = capture.strip_prefix("rect:") {
            let n: Vec<f32> = rest
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if n.len() != 4 {
                return None;
            }
            return crop_rect(
                full,
                egui::Rect::from_min_size(egui::pos2(n[0], n[1]), egui::vec2(n[2], n[3])),
            );
        }
        crop_named(full, app, CaptureTarget::from_name(capture)?)
    }

    // ----- S7 recipe runner ------------------------------------------------

    /// Run a Python recipe source and collect its `sadda.doc.shot(...)` calls.
    /// Builds the real `sadda` module so the recipe can `import sadda.doc`.
    fn collect_shots(src: &str) -> Vec<crate::sadda_app::RecipeShot> {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;
        Python::attach(|py| {
            let m = PyModule::new(py, "sadda").expect("build sadda module");
            crate::sadda_app::sadda(&m).expect("register sadda");
            py.import("sys")
                .and_then(|s| s.getattr("modules"))
                .and_then(|m2| m2.set_item("sadda", &m))
                .expect("register in sys.modules");
            let code = std::ffi::CString::new(src).expect("recipe source contains a NUL byte");
            py.run(code.as_c_str(), None, None)
                .expect("recipe executed");
        });
        crate::sadda_app::take_recipe_shots()
    }

    /// Select the bundle named `name` in the open project, if present.
    fn select_bundle_by_name(app: &mut SaddaApp, name: &str) -> bool {
        let id = if let crate::AppState::ProjectLoaded { project, .. } = &app.app_state {
            project
                .bundles()
                .ok()
                .and_then(|bs| bs.into_iter().find(|b| b.name == name).map(|b| b.id))
        } else {
            None
        };
        if let Some(id) = id {
            app.select_bundle(id);
            true
        } else {
            false
        }
    }

    /// Show exactly the named signal panes (hide the rest).
    fn apply_show_only(app: &mut SaddaApp, show: &[String]) {
        use crate::VisibilityAction;
        use crate::sadda_app::SignalPaneId::{
            F0, Formants, Intensity, Mfcc, Spectrogram, TierStrip, Vad, Waveform,
        };
        let want: Vec<_> = show
            .iter()
            .filter_map(|s| crate::sadda_app::SignalPaneId::from_name(s))
            .collect();
        for p in [
            Waveform,
            Spectrogram,
            TierStrip,
            F0,
            Formants,
            Intensity,
            Vad,
            Mfcc,
        ] {
            app.apply_visibility_actions(vec![VisibilityAction::Pane {
                pane: p,
                visible: want.contains(&p),
            }]);
        }
    }

    /// Apply one shot's spec to the harness (state + queued layout changes).
    /// `source_base` resolves input paths (`project`/`audio`); `scratch_root` +
    /// `idx` give an `audio` shot a fresh throwaway project dir.
    fn apply_shot(
        harness: &mut egui_kittest::Harness<'_, SaddaApp>,
        shot: &crate::sadda_app::RecipeShot,
        source_base: &Path,
        scratch_root: &Path,
        idx: usize,
    ) {
        use crate::sadda_app::{GuiColumn, SignalPaneId};

        if let Some((w, h)) = shot.size {
            harness.set_size(egui::vec2(w, h));
        }
        let app = harness.state_mut();
        app.persisted.ui_scale = 1.0;
        if let Some(theme) = shot
            .theme
            .as_deref()
            .and_then(crate::state::ThemePref::from_name)
        {
            app.persisted.theme = theme;
        }
        // Scene source: build a throwaway project from `audio`, or open an
        // existing `project`.
        if let Some(audio) = &shot.audio {
            let proj = scratch_root.join(format!("proj-{idx}"));
            let _ = std::fs::remove_dir_all(&proj);
            app.create_project_at(proj);
            app.add_bundle_from_wav(source_base.join(audio)); // auto-selects
        } else if let Some(project) = &shot.project {
            app.open_project_at(source_base.join(project));
        }
        if let Some(bundle) = &shot.bundle {
            select_bundle_by_name(app, bundle);
        }
        // B1: import a TextGrid into the selected bundle so its tiers show.
        if let Some(tg) = &shot.textgrid {
            let path = source_base.join(tg);
            match (&app.app_state, app.selected_bundle_id) {
                (crate::AppState::ProjectLoaded { project, .. }, Some(bid)) if path.exists() => {
                    if let Err(e) = project.import_textgrid(&path, bid) {
                        eprintln!("[doc_render] TextGrid import failed ({tg}): {e}");
                    }
                }
                _ => eprintln!("[doc_render] TextGrid skipped (not found / no bundle): {tg}"),
            }
        }
        if let Some(show) = &shot.show {
            apply_show_only(app, show);
        }
        for (pane, ht) in &shot.heights {
            if let Some(p) = SignalPaneId::from_name(pane) {
                app.pending_pane_heights.push((p, *ht));
            }
        }
        for (col, wd) in &shot.widths {
            if let Some(c) = GuiColumn::from_name(col) {
                app.pending_column_widths.push((c, *wd));
            }
        }
    }

    /// Execute a Python recipe headlessly. Input paths (`project`/`audio`)
    /// resolve against `source_base`; each shot's PNG is written under
    /// `output_base.join(shot.to)`. Returns the written paths.
    fn run_recipe(src: &str, source_base: &Path, output_base: &Path) -> Vec<PathBuf> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static RUN: AtomicUsize = AtomicUsize::new(0);

        let shots = collect_shots(src);
        // Unique per run so concurrent recipe runs don't clobber each other's
        // throwaway projects (each run wipes its own scratch root).
        let scratch_root = std::env::temp_dir().join(format!(
            "sadda-doc-recipe-{}-{}",
            std::process::id(),
            RUN.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&scratch_root);
        let mut harness = doc_harness(egui::vec2(1280.0, 800.0));
        let mut written = Vec::new();
        for (idx, shot) in shots.iter().enumerate() {
            apply_shot(&mut harness, shot, source_base, &scratch_root, idx);
            settle_analysis(&mut harness, Duration::from_secs(10));
            let full = harness.render().expect("wgpu render");
            let img = crop_for_capture(&full, harness.state(), &shot.capture)
                .unwrap_or_else(|| panic!("capture {:?} produced no image", shot.capture));
            let path = output_base.join(&shot.to);
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            img.save(&path).expect("write recipe PNG");
            written.push(path);
        }
        written
    }

    /// Run a recipe from a `.py` file. Input paths resolve against
    /// `source_base`, outputs under `output_base`.
    fn run_recipe_file(recipe: &Path, source_base: &Path, output_base: &Path) -> Vec<PathBuf> {
        let src = std::fs::read_to_string(recipe)
            .unwrap_or_else(|e| panic!("read recipe {recipe:?}: {e}"));
        run_recipe(&src, source_base, output_base)
    }

    /// Workspace root (two levels up from this crate).
    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// Where rendered doc images land for inspection (under the workspace
    /// `target/`, which is git-ignored). Returned so the test can log it.
    fn out_dir() -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/doc-render");
        std::fs::create_dir_all(&dir).expect("out dir");
        dir
    }

    /// Spike: the headless wgpu renderer produces a full-window image at the
    /// requested size. Proves the offscreen path works before anything else.
    #[test]
    #[ignore = "needs a software wgpu adapter (lavapipe); run with --ignored"]
    fn headless_render_produces_an_image() {
        let mut h = doc_harness(egui::vec2(800.0, 600.0));
        h.run();
        let img = h.render().expect("headless wgpu render should succeed");
        assert_eq!((img.width(), img.height()), (800, 600));
    }

    /// End-to-end: load a real bundle at a doc size, hide everything but the
    /// spectrogram, settle the DSP, and crop the `SignalColumn` region to a PNG.
    /// This exercises the whole S6 spine: drive the app → settle async analysis
    /// → resolve a named rect → render → crop → write.
    #[test]
    #[ignore = "needs a software wgpu adapter (lavapipe); run with --ignored"]
    fn renders_signal_column_of_a_bundle() {
        let (w, h) = (1280.0f32, 800.0f32);
        let mut harness = doc_harness(egui::vec2(w, h));

        // Compose the view via the same state the GUI/scripts drive.
        {
            let app = harness.state_mut();
            load_fixture_bundle(app, scratch_dir("signal-column"));
            // A clean spectrogram-only figure.
            app.persisted.panes.waveform = true;
            app.persisted.panes.spectrogram = true;
            app.persisted.panes.tier_strip = false;
            app.persisted.tracks.f0_visible = false;
            app.persisted.tracks.formants_visible = false;
            app.persisted.tracks.intensity_visible = false;
            app.persisted.tracks.vad_visible = false;
        }

        settle_analysis(&mut harness, Duration::from_secs(10));

        // The spectrogram must have actually built (not just timed out).
        assert!(
            harness.state().active_spectrogram.is_some(),
            "spectrogram analysis did not settle in time"
        );

        let full = harness.render().expect("wgpu render");
        assert_eq!((full.width(), full.height()), (w as u32, h as u32));

        let region = crop_named(&full, harness.state(), CaptureTarget::SignalColumn)
            .expect("SignalColumn rect should be registered once the bundle renders");
        assert!(
            region.width() > 100 && region.height() > 100,
            "signal column crop looks too small: {}x{}",
            region.width(),
            region.height()
        );

        let path = out_dir().join("signal-column.png");
        region.save(&path).expect("write PNG");
        eprintln!("[doc_render] wrote {}", path.display());
    }

    /// S6.1: a scripted pane height is honoured by the *real* layout — proves
    /// the `PanelState` write drives egui's resizable panel. Pin the waveform to
    /// an exact height and confirm its rendered rect matches.
    #[test]
    #[ignore = "needs a software wgpu adapter (lavapipe); run with --ignored"]
    fn respects_a_scripted_pane_height() {
        use crate::sadda_app::SignalPaneId;

        let mut harness = doc_harness(egui::vec2(1000.0, 900.0));
        {
            let app = harness.state_mut();
            load_fixture_bundle(app, scratch_dir("pane-height"));
            // A clean waveform-over-spectrogram layout (spectrogram flexes).
            app.persisted.tracks.f0_visible = false;
            app.persisted.tracks.formants_visible = false;
            app.persisted.tracks.intensity_visible = false;
            app.persisted.tracks.vad_visible = false;
            app.persisted.panes.tier_strip = false;
            // Pin the waveform to an exact height (as a script would).
            app.pending_pane_heights
                .push((SignalPaneId::Waveform, 120.0));
        }

        settle_analysis(&mut harness, Duration::from_secs(10));

        let wf = harness
            .state()
            .capture_rect_for(CaptureTarget::Waveform)
            .expect("waveform rect should be registered");
        assert!(
            (wf.height() - 120.0).abs() <= 6.0,
            "waveform rendered at {}px, expected ~120px",
            wf.height()
        );
    }

    /// S6.2: a scripted column width is honoured by the real layout — the same
    /// `PanelState` mechanism on the horizontal axis. Pin the bundle sidebar to
    /// an exact width and confirm its rendered rect matches.
    #[test]
    #[ignore = "needs a software wgpu adapter (lavapipe); run with --ignored"]
    fn respects_a_scripted_column_width() {
        use crate::sadda_app::GuiColumn;

        let mut harness = doc_harness(egui::vec2(1200.0, 800.0));
        load_fixture_bundle(harness.state_mut(), scratch_dir("column-width"));
        settle_analysis(&mut harness, Duration::from_secs(10));

        // egui stores each panel's *content* rect (inside the frame margin) in
        // `PanelState`, so absolute widths carry a constant margin offset. Verify
        // by difference instead — widening the request by 100pt must widen the
        // rendered panel by 100pt (the offset cancels, and it also catches an
        // unstable/animating panel that would drift).
        let width_of = |h: &egui_kittest::Harness<'_, SaddaApp>| {
            egui::containers::panel::PanelState::load(&h.ctx, egui::Id::new("bundle_sidebar"))
                .expect("bundle sidebar panel state")
                .rect
                .width()
        };

        harness
            .state_mut()
            .pending_column_widths
            .push((GuiColumn::Bundles, 175.0));
        harness.run();
        let narrow = width_of(&harness);

        harness
            .state_mut()
            .pending_column_widths
            .push((GuiColumn::Bundles, 275.0));
        harness.run();
        let wide = width_of(&harness);

        assert!(
            ((wide - narrow) - 100.0).abs() <= 3.0,
            "widening the request by 100pt changed the sidebar by {}pt (narrow={narrow}, wide={wide})",
            wide - narrow
        );
    }

    /// S7: a full Python recipe → two documentation PNGs. Proves the authoring
    /// surface (`sadda.doc.shot`), the executor (open → select → compose →
    /// settle → render → crop → write), the light theme, and per-shot layout.
    #[test]
    #[ignore = "needs a software wgpu adapter (lavapipe); run with --ignored"]
    fn runs_a_python_recipe() {
        let base = scratch_dir("recipe");
        // A project for the recipe to open — created straight via the engine
        // (no GUI needed).
        {
            let project =
                sadda_engine::Project::create(base.join("proj"), "demo").expect("create project");
            project
                .add_bundle("demo", fixture_wav())
                .expect("add bundle");
        }

        let recipe = r#"
import sadda.doc as doc

doc.shot(
    project="proj", bundle="demo",
    size=(1200, 760), theme="light",
    show=["waveform", "spectrogram"],
    heights=[("waveform", 130)],
    capture="signal-column", to="img/overview.png",
)
doc.shot(
    project="proj", bundle="demo",
    theme="dark", show=["spectrogram"],
    capture="spectrogram", to="img/spectrogram.png",
)
"#;

        let written = run_recipe(recipe, &base, &base);
        assert_eq!(written.len(), 2, "recipe defines two shots");
        for p in &written {
            let img = image::open(p).unwrap_or_else(|e| panic!("decode {p:?}: {e}"));
            assert!(
                img.width() > 100 && img.height() > 100,
                "{p:?} looks too small: {}x{}",
                img.width(),
                img.height()
            );
        }
        // Also drop a copy where a human can eyeball it.
        let preview = out_dir().join("recipe-overview.png");
        let _ = std::fs::copy(&written[0], &preview);
        eprintln!(
            "[doc_render] recipe wrote {written:?}  (preview: {})",
            preview.display()
        );
    }

    /// S8: the drift gate. Render a fixed scene, crop the signal column, and
    /// compare against a committed golden (`tests/snapshots/doc-signal-column.png`)
    /// with egui's cross-platform-tuned tolerance. A UI change that alters the
    /// image fails here until it's regenerated (`UPDATE_SNAPSHOTS=1`) and
    /// reviewed — this is what makes drift a *caught* failure, not silent rot.
    #[test]
    #[ignore = "needs a software wgpu adapter (lavapipe); run with --ignored"]
    fn snapshot_signal_column_matches_golden() {
        let mut harness = doc_harness(egui::vec2(1280.0, 800.0));
        {
            let app = harness.state_mut();
            load_fixture_bundle(app, scratch_dir("snapshot"));
            app.persisted.theme = crate::state::ThemePref::Light;
            app.persisted.panes.tier_strip = false;
            app.persisted.tracks.f0_visible = false;
            app.persisted.tracks.formants_visible = false;
            app.persisted.tracks.intensity_visible = false;
            app.persisted.tracks.vad_visible = false;
        }
        settle_analysis(&mut harness, Duration::from_secs(10));

        let full = harness.render().expect("wgpu render");
        let img = crop_named(&full, harness.state(), CaptureTarget::SignalColumn)
            .expect("signal-column rect");

        let opts = egui_kittest::SnapshotOptions::new()
            .output_path(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots"));
        egui_kittest::try_image_snapshot_options(&img, "doc-signal-column", &opts)
            .expect("signal-column drifted from golden — run UPDATE_SNAPSHOTS=1 to refresh");
    }

    /// B1: `textgrid=` imports annotations so the tier strip has content — guards
    /// the "image annotation piping" from silently regressing (a broken import
    /// would still render an image, just without tiers).
    #[test]
    #[ignore = "needs a software wgpu adapter (lavapipe); run with --ignored"]
    fn textgrid_import_adds_tiers() {
        let mut harness = doc_harness(egui::vec2(900.0, 700.0));
        let shot = crate::sadda_app::RecipeShot {
            to: "unused".into(),
            capture: "signal-column".into(),
            audio: Some("docs/recipes/assets/demo.wav".into()),
            textgrid: Some("docs/recipes/assets/demo.TextGrid".into()),
            show: Some(vec!["waveform".into(), "tier_strip".into()]),
            ..Default::default()
        };
        apply_shot(&mut harness, &shot, &repo_root(), &scratch_dir("tg"), 0);
        settle_analysis(&mut harness, Duration::from_secs(10));

        let app = harness.state();
        let n_tiers = match (&app.app_state, app.selected_bundle_id) {
            (crate::AppState::ProjectLoaded { project, .. }, Some(bid)) => {
                project.tiers(Some(bid)).map(|t| t.len()).unwrap_or(0)
            }
            _ => 0,
        };
        assert!(
            n_tiers >= 1,
            "textgrid= should have imported at least one tier"
        );
    }

    /// S7b: run the committed recipe *files* under `docs/recipes/` against the
    /// repo (inputs + outputs resolve from the repo root). This is what `just
    /// docs-images` runs to regenerate the documentation images.
    #[test]
    #[ignore = "needs a software wgpu adapter (lavapipe); run with --ignored"]
    fn generates_images_from_recipe_files() {
        let root = repo_root();
        let recipes_dir = root.join("docs/recipes");
        let mut recipes: Vec<PathBuf> = std::fs::read_dir(&recipes_dir)
            .unwrap_or_else(|e| panic!("read {recipes_dir:?}: {e}"))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "py"))
            .collect();
        recipes.sort();
        assert!(!recipes.is_empty(), "no recipe files under {recipes_dir:?}");

        for recipe in &recipes {
            let written = run_recipe_file(recipe, &root, &root);
            assert!(!written.is_empty(), "{recipe:?} produced no images");
            for p in &written {
                let img = image::open(p).unwrap_or_else(|e| panic!("decode {p:?}: {e}"));
                assert!(
                    img.width() > 100 && img.height() > 100,
                    "{p:?} looks too small: {}x{}",
                    img.width(),
                    img.height()
                );
            }
            eprintln!("[doc_render] {recipe:?} → {written:?}");
        }
    }
}
