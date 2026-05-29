//! Opt-in debugging aids, all gated by the `SADDA_DEBUG` environment
//! variable so normal/release runs pay nothing (one cached bool check).
//!
//! Set `SADDA_DEBUG=1` to enable. Diagnostics are appended to a **log file**
//! (default `<tmp>/sadda-debug.log`, override with `SADDA_DEBUG_LOG`) *and*
//! echoed to stderr — the file matters because it can be read back directly,
//! independent of terminal copy/scroll quirks. `SADDA_DEBUG` also turns on
//! egui's hover-debug overlays and the F12 screenshot capture.

use eframe::egui;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

/// Whether `SADDA_DEBUG` is set to a non-falsey value (cached per process).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SADDA_DEBUG")
            .map(|v| !matches!(v.trim(), "" | "0" | "false" | "no"))
            .unwrap_or(false)
    })
}

/// Path of the debug log file (`SADDA_DEBUG_LOG`, else a temp-dir default).
pub fn log_path() -> PathBuf {
    std::env::var_os("SADDA_DEBUG_LOG")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("sadda-debug.log"))
}

/// Appends `msg` to the debug log file and echoes it to stderr. No-op when
/// `SADDA_DEBUG` is off. Prefer the [`dlog!`](crate::dlog) macro at call
/// sites so the `format!` is skipped entirely when debugging is disabled.
pub fn log(msg: &str) {
    if !enabled() {
        return;
    }
    eprintln!("[sadda] {msg}");
    static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let file = FILE.get_or_init(|| {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path())
            .ok()
            .map(Mutex::new)
    });
    if let Some(m) = file {
        if let Ok(mut f) = m.lock() {
            let _ = writeln!(f, "{msg}");
        }
    }
}

/// Saves an egui screenshot to a sequentially-numbered PNG in the temp dir;
/// logs and returns the path. Used by the F12 capture in debug mode.
pub fn save_screenshot(ci: &egui::ColorImage) -> Option<PathBuf> {
    static SEQ: AtomicU32 = AtomicU32::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let (w, h) = (ci.width() as u32, ci.height() as u32);
    let mut rgba = Vec::with_capacity(w as usize * h as usize * 4);
    for px in &ci.pixels {
        rgba.extend_from_slice(&px.to_array());
    }
    let path = std::env::temp_dir().join(format!("sadda-shot-{n}.png"));
    let img = image::RgbaImage::from_raw(w, h, rgba)?;
    img.save(&path).ok()?;
    log(&format!("screenshot saved: {}", path.display()));
    Some(path)
}

/// Logs a debug message only when `SADDA_DEBUG` is set, skipping the
/// `format!` otherwise. Usage: `dlog!("[layout] frame={:?}", rect);`.
#[macro_export]
macro_rules! dlog {
    ($($arg:tt)*) => {
        if $crate::debug::enabled() {
            $crate::debug::log(&format!($($arg)*));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::Color32;

    // The PNG encoding path is the only screenshot logic that runs without a
    // live GUI; exercise it on a tiny image. (The `log()` call inside is a
    // no-op unless `SADDA_DEBUG` is set, so this writes the PNG either way.)
    #[test]
    fn save_screenshot_writes_a_decodable_png() {
        let ci = egui::ColorImage::filled([3, 2], Color32::from_rgb(10, 20, 30));
        let path = save_screenshot(&ci).expect("screenshot should be written");
        assert!(path.exists(), "PNG file should exist at {}", path.display());
        let decoded = image::open(&path).expect("written file should decode as an image");
        assert_eq!((decoded.width(), decoded.height()), (3, 2));
        let _ = std::fs::remove_file(&path);
    }
}
