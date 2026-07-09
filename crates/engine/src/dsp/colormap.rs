//! Colormaps for rendering a normalised `[0, 1]` scalar field (a
//! spectrogram, MFCC, or embedding raster) into RGB.
//!
//! Moved engine-side in G0 (figure-export groundwork) so the spectrogram
//! bake can run headlessly — the same colormap the on-screen GUI uses
//! feeds the SVG/PDF/TikZ figure exporter, guaranteeing the figure matches
//! what the user saw.
//!
//! ## References
//! - Viridis / Magma / Cividis — perceptually-uniform maps from matplotlib;
//!   Cividis is optimised for colour-vision deficiency
//!   (Nuñez, Anderton & Renslow 2018, "Optimizing colormaps with consideration
//!   for color vision deficiency to enable accurate interpretation of
//!   scientific data"), <https://doi.org/10.1371/journal.pone.0199239>.
//! - `colorous` crate (the sampling backend):
//!   <https://docs.rs/colorous>

/// Which colormap maps normalised `[0, 1]` power values into RGB. Default
/// is [`ColormapKind::Viridis`] (perceptually uniform).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ColormapKind {
    /// Modern perceptually-uniform default; dark purple → blue → green → yellow.
    #[default]
    Viridis,
    /// Dark-mode-friendly perceptually-uniform alternate; black → purple → red → yellow.
    Magma,
    /// matplotlib / MATLAB `hot`: black → red → orange → yellow → white — a
    /// piecewise-linear RGB ramp (red fills first, then green, then blue).
    /// Not perceptually uniform, unlike Viridis / Magma / Cividis.
    Hot,
    /// Perceptually-uniform map optimised for colour-vision deficiency
    /// (dark blue → grey → yellow). The accessibility pick — it stays
    /// monotonic in luminance for all common forms of CVD.
    Cividis,
    /// Classic black-and-white spectrogram; Praat refugees.
    Greyscale,
}

impl ColormapKind {
    /// Human-readable label for a toolbar `ComboBox`.
    pub fn label(self) -> &'static str {
        match self {
            ColormapKind::Viridis => "Viridis",
            ColormapKind::Magma => "Magma",
            ColormapKind::Hot => "Hot",
            ColormapKind::Cividis => "Cividis (CVD-safe)",
            ColormapKind::Greyscale => "Greyscale",
        }
    }

    /// Samples the colormap at `t ∈ [0, 1]` (clamped), returning an `(r, g, b)`
    /// 8-bit triple.
    pub fn sample(self, t: f32) -> (u8, u8, u8) {
        let t = t.clamp(0.0, 1.0) as f64;
        match self {
            ColormapKind::Viridis => {
                let c = colorous::VIRIDIS.eval_continuous(t);
                (c.r, c.g, c.b)
            }
            ColormapKind::Magma => {
                let c = colorous::MAGMA.eval_continuous(t);
                (c.r, c.g, c.b)
            }
            ColormapKind::Hot => {
                // matplotlib / MATLAB `hot`: black → red → orange → yellow →
                // white. Red fills over the first 3/8, green over the next
                // 3/8, blue over the last 2/8.
                let t = t as f32;
                let r = (8.0 / 3.0 * t).clamp(0.0, 1.0);
                let g = ((8.0 * t - 3.0) / 3.0).clamp(0.0, 1.0);
                let b = (4.0 * t - 3.0).clamp(0.0, 1.0);
                (
                    (r * 255.0).round() as u8,
                    (g * 255.0).round() as u8,
                    (b * 255.0).round() as u8,
                )
            }
            ColormapKind::Cividis => {
                let c = colorous::CIVIDIS.eval_continuous(t);
                (c.r, c.g, c.b)
            }
            ColormapKind::Greyscale => {
                let v = (t as f32 * 255.0).round() as u8;
                (v, v, v)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greyscale_is_linear_and_monochrome() {
        let (r, g, b) = ColormapKind::Greyscale.sample(0.5);
        assert!((r as i32 - 128).abs() < 2);
        assert_eq!(r, g);
        assert_eq!(g, b);
    }

    #[test]
    fn endpoints_and_clamping() {
        // Out-of-range inputs clamp rather than panic.
        assert_eq!(
            ColormapKind::Greyscale.sample(-1.0),
            ColormapKind::Greyscale.sample(0.0)
        );
        assert_eq!(
            ColormapKind::Greyscale.sample(2.0),
            ColormapKind::Greyscale.sample(1.0)
        );
    }

    #[test]
    fn cividis_distinct_from_viridis_at_midpoint() {
        assert_ne!(
            ColormapKind::Cividis.sample(0.5),
            ColormapKind::Viridis.sample(0.5)
        );
    }

    #[test]
    fn hot_ramps_black_to_white_and_is_not_magma() {
        // black → red → orange → yellow → white (matplotlib/MATLAB `hot`).
        assert_eq!(ColormapKind::Hot.sample(0.0), (0, 0, 0));
        assert_eq!(ColormapKind::Hot.sample(0.375), (255, 0, 0)); // red
        assert_eq!(ColormapKind::Hot.sample(0.75), (255, 255, 0)); // yellow
        assert_eq!(ColormapKind::Hot.sample(1.0), (255, 255, 255));
        // Magma is preserved as its own (distinct) perceptually-uniform map.
        assert_ne!(
            ColormapKind::Hot.sample(0.5),
            ColormapKind::Magma.sample(0.5)
        );
    }
}
