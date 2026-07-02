//! Strand 2: capture an arbitrary rectangular region of the GUI to a PNG,
//! for documentation and slides.
//!
//! This is deliberately *not* the signal figure exporter (that renders a
//! vector `FigureSpec` — see the DEVLOG 2026-07-01 design entry). This is a raw
//! framebuffer crop with no signal semantics: the user rubber-bands a rectangle
//! over the running app, and we crop the screenshot to it.
//!
//! The pixel plumbing mirrors [`crate::debug::save_screenshot`] (egui delivers
//! the whole framebuffer as an [`egui::ColorImage`] via `Event::Screenshot`);
//! the only extra step here is the crop, kept pure so it can be unit-tested
//! without a live GUI.

use eframe::egui;

/// Crop a full-window screenshot (`ci`, in physical pixels) to `region`, given
/// in logical screen points, scaling by `pixels_per_point`.
///
/// Unlike [`egui::ColorImage::region`], which panics on an inverted or
/// out-of-bounds rectangle, this clamps `region` to the image and returns
/// `None` for a zero-area result — a rubber-band that ends outside the window,
/// or a stray click, simply yields nothing instead of crashing.
pub fn crop_region(
    ci: &egui::ColorImage,
    region: egui::Rect,
    pixels_per_point: f32,
) -> Option<image::RgbaImage> {
    let (img_w, img_h) = (ci.width(), ci.height());
    let ppp = pixels_per_point.max(f32::MIN_POSITIVE);

    // Logical points -> physical pixels. Round outward so the crop encloses the
    // drawn selection rather than shaving a pixel off an edge.
    let clamp_x = |v: f32| (v as i64).clamp(0, img_w as i64) as usize;
    let clamp_y = |v: f32| (v as i64).clamp(0, img_h as i64) as usize;
    let min_x = clamp_x((region.min.x * ppp).floor());
    let min_y = clamp_y((region.min.y * ppp).floor());
    let max_x = clamp_x((region.max.x * ppp).ceil());
    let max_y = clamp_y((region.max.y * ppp).ceil());

    let (w, h) = (max_x.saturating_sub(min_x), max_y.saturating_sub(min_y));
    if w == 0 || h == 0 {
        return None;
    }

    let mut rgba = Vec::with_capacity(w * h * 4);
    for y in min_y..max_y {
        let row = y * img_w;
        for x in min_x..max_x {
            rgba.extend_from_slice(&ci.pixels[row + x].to_array());
        }
    }
    image::RgbaImage::from_raw(w as u32, h as u32, rgba)
}

#[cfg(test)]
mod tests {
    use super::*;
    use eframe::egui::{Color32, ColorImage, Pos2, Rect};

    /// A `w×h` image whose red channel encodes the column and green the row, so
    /// a crop can be checked pixel-exactly against its source coordinates.
    fn ramp(w: usize, h: usize) -> ColorImage {
        let mut pixels = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                pixels.push(Color32::from_rgb(x as u8, y as u8, 0));
            }
        }
        ColorImage::new([w, h], pixels)
    }

    #[test]
    fn crops_the_requested_rectangle() {
        let img = ramp(8, 8);
        let region = Rect::from_min_max(Pos2::new(2.0, 3.0), Pos2::new(5.0, 6.0));
        let out = crop_region(&img, region, 1.0).expect("non-empty crop");
        assert_eq!(out.dimensions(), (3, 3));
        // Top-left of the crop is source pixel (2, 3).
        assert_eq!(out.get_pixel(0, 0).0, [2, 3, 0, 255]);
        assert_eq!(out.get_pixel(2, 2).0, [4, 5, 0, 255]);
    }

    #[test]
    fn scales_by_pixels_per_point() {
        // A HiDPI framebuffer: 8 logical points wide but 16 physical px.
        let img = ramp(16, 16);
        let region = Rect::from_min_max(Pos2::new(1.0, 1.0), Pos2::new(3.0, 3.0));
        let out = crop_region(&img, region, 2.0).expect("non-empty crop");
        // 2 points * 2 ppp = 4 physical px each side.
        assert_eq!(out.dimensions(), (4, 4));
        assert_eq!(out.get_pixel(0, 0).0, [2, 2, 0, 255]);
    }

    #[test]
    fn clamps_a_region_spilling_past_the_edge() {
        let img = ramp(4, 4);
        // Extends well beyond the 4×4 image; must clamp, not panic.
        let region = Rect::from_min_max(Pos2::new(2.0, 2.0), Pos2::new(100.0, 100.0));
        let out = crop_region(&img, region, 1.0).expect("clamped crop");
        assert_eq!(out.dimensions(), (2, 2));
    }

    #[test]
    fn zero_area_selection_yields_none() {
        let img = ramp(4, 4);
        let dot = Rect::from_min_max(Pos2::new(1.0, 1.0), Pos2::new(1.0, 1.0));
        assert!(crop_region(&img, dot, 1.0).is_none());
        // Fully off-screen (both edges clamp to the same column) is also empty.
        let off = Rect::from_min_max(Pos2::new(10.0, 10.0), Pos2::new(20.0, 20.0));
        assert!(crop_region(&img, off, 1.0).is_none());
    }
}
