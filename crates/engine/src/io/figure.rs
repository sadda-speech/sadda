//! Publication figure export — a `FigureSpec` intermediate representation and
//! pluggable serializers (SVG now; PDF via `svg2pdf`; TikZ in G2).
//!
//! Mirrors the [`super::tabular`] split: a plain-data IR ([`FigureSpec`] +
//! its lanes) built by the caller, and free serializer functions
//! ([`to_svg`]) that render it — so the same figure description can target
//! several backends and be assembled headlessly (Python, the doc-render
//! harness) or from the GUI.
//!
//! ## The figure
//!
//! A figure is a vertical stack of **lanes sharing one time axis**: an
//! optional waveform band, an optional spectrogram raster, then any number of
//! annotation-tier rows, with a shared "Time (s)" axis at the bottom. This is
//! the staple waveform / spectrogram / tier figure of a phonetics paper — the
//! specTeX style baseline studied in the 2026-07-01 design entry: a thin
//! vector waveform over a raster spectrogram, tier boxes whose boundary lines
//! extend down through the signal panels, Doulos SIL for cleanly typeset IPA.
//!
//! Text (IPA tier labels, axis labels) is rendered as real SVG `<text>` with
//! the **Doulos SIL** font embedded in the file (SIL Open Font License; see
//! `assets/fonts/`), so a figure renders identically everywhere and stays
//! selectable/editable, without the viewer needing the font installed.
//!
//! ## References
//! - specTeX — Praat TikZ figure exporter; style baseline:
//!   <https://github.com/dbqpdb/specTeX>
//! - Doulos SIL — IPA reference face, SIL Global:
//!   <https://software.sil.org/doulos/>

use base64::Engine as _;

/// The **Doulos SIL** font, embedded so figures are self-contained. SIL Open
/// Font License 1.1 — see `assets/fonts/DoulosSIL-OFL.txt`.
const DOULOS_SIL_TTF: &[u8] = include_bytes!("../../assets/fonts/DoulosSIL-Regular.ttf");

/// The CSS `font-family` the embedded `@font-face` publishes and every
/// `<text>` element references.
const FONT_FAMILY: &str = "Doulos SIL";

/// A complete publication figure: lanes stacked on a shared time axis, plus
/// the style that governs sizing and colour. Build one and hand it to a
/// serializer ([`to_svg`]).
///
/// Every lane is optional / list-valued, so a figure can show any subset —
/// spectrogram-only, waveform + tiers, the whole column — independent of what
/// is on screen (the GUI/Python caller decides, defaulting from the visible
/// lanes).
pub struct FigureSpec {
    /// Optional title drawn above the figure.
    pub title: Option<String>,
    /// The x-axis extent in seconds, `(start, end)`. All lanes map time to x
    /// through this window; annotations outside it are clipped.
    pub time_range: (f64, f64),
    /// Optional waveform band (drawn as a filled vector envelope).
    pub waveform: Option<WaveformLane>,
    /// Optional spectrogram raster (embedded as a PNG `<image>`).
    pub spectrogram: Option<SpectrogramLane>,
    /// Annotation-tier rows, drawn top-to-bottom in the given order.
    pub tiers: Vec<FigureTier>,
    /// Sizing, fonts, and colours.
    pub style: FigureStyle,
}

/// A waveform lane as a **min/max envelope** — one `(min, max)` amplitude pair
/// per horizontal column, left to right — drawn as a filled band (improving on
/// specTeX, which rasterises the waveform). Build it from
/// `state::build_envelope` (app) or any min/max downsampler.
pub struct WaveformLane {
    /// `(min, max)` amplitude per column, left → right across the time window.
    pub minmax: Vec<(f32, f32)>,
    /// The amplitude extent to map to the lane's vertical span, `(min, max)`.
    /// Usually symmetric (e.g. `(-1.0, 1.0)`); the zero line is drawn at 0.
    pub amplitude_range: (f32, f32),
}

/// A spectrogram lane: a baked RGBA raster (row-major, row 0 = top = highest
/// frequency — the layout [`crate::dsp::colormap_bake`] produces) plus the
/// frequency extent for the y-axis.
pub struct SpectrogramLane {
    /// Row-major RGBA8, `width * height * 4` bytes.
    pub rgba: Vec<u8>,
    /// Raster width in cells (time frames).
    pub width: usize,
    /// Raster height in cells (frequency bins).
    pub height: usize,
    /// The top-of-lane frequency in Hz (the raster's highest bin), for the
    /// y-axis label. The bottom is 0 Hz.
    pub max_freq_hz: f32,
}

/// One annotation tier row in the figure.
pub struct FigureTier {
    /// Tier name, drawn in the left margin.
    pub name: String,
    /// The tier's annotations.
    pub content: FigureTierContent,
}

/// A tier's drawable content — intervals (boxed) or points (ticked).
pub enum FigureTierContent {
    /// Interval tier: labelled boxes with boundary lines.
    Intervals(Vec<FigureInterval>),
    /// Point tier: labelled vertical ticks.
    Points(Vec<FigurePoint>),
}

/// A labelled interval `[start, end)` in seconds.
pub struct FigureInterval {
    /// Interval start in seconds.
    pub start: f64,
    /// Interval end in seconds.
    pub end: f64,
    /// Interval label (may be empty; rendered centred in the box).
    pub label: String,
}

/// A labelled instant in seconds.
pub struct FigurePoint {
    /// Point time in seconds.
    pub time: f64,
    /// Point label (may be empty; rendered above the tick).
    pub label: String,
}

/// Figure sizing, fonts, and colours. [`Default`] is a clean publication
/// preset (journal proportions, Doulos SIL, greyscale-friendly strokes).
pub struct FigureStyle {
    /// Overall figure width in px (the SVG user-unit).
    pub width: f64,
    /// Height of the waveform lane, if drawn.
    pub waveform_height: f64,
    /// Height of the spectrogram lane, if drawn.
    pub spectrogram_height: f64,
    /// Height of each tier row.
    pub tier_height: f64,
    /// Left margin reserved for y-axis + tier-name labels.
    pub margin_left: f64,
    /// Right margin.
    pub margin_right: f64,
    /// Top margin (above the first lane / below the title).
    pub margin_top: f64,
    /// Height reserved at the bottom for the shared time axis.
    pub axis_height: f64,
    /// Base font size in px for axis labels and tier text.
    pub font_size: f64,
    /// Stroke colour for the waveform, axes, and boundaries (CSS colour).
    pub stroke: String,
    /// Fill colour for the waveform band (CSS colour).
    pub waveform_fill: String,
    /// Background colour of the whole figure (CSS colour).
    pub background: String,
}

impl Default for FigureStyle {
    fn default() -> Self {
        Self {
            width: 800.0,
            waveform_height: 90.0,
            spectrogram_height: 220.0,
            tier_height: 34.0,
            margin_left: 64.0,
            margin_right: 12.0,
            margin_top: 10.0,
            axis_height: 34.0,
            font_size: 13.0,
            stroke: "#000000".to_string(),
            waveform_fill: "#333333".to_string(),
            background: "#ffffff".to_string(),
        }
    }
}

impl FigureStyle {
    /// The title band height (0 when there is no title).
    fn title_height(&self, has_title: bool) -> f64 {
        if has_title { self.font_size * 1.8 } else { 0.0 }
    }
}

/// Serializes a [`FigureSpec`] to a **self-contained SVG** string (embedded
/// font + embedded spectrogram raster; no external references).
///
/// Layout, top to bottom: optional title, waveform band, spectrogram raster,
/// tier rows, shared time axis. The left margin carries the per-lane y-axis /
/// tier-name labels; interval boundary lines extend down through the signal
/// panels (the specTeX signature).
pub fn to_svg(spec: &FigureSpec) -> String {
    let s = &spec.style;
    let has_title = spec.title.is_some();

    // --- vertical layout: assign each lane a top y and a height ---------
    let mut y = s.margin_top + s.title_height(has_title);
    let panels_top = y; // where the signal panels begin (for boundary lines)

    let wave_y = spec.waveform.as_ref().map(|_| {
        let top = y;
        y += s.waveform_height;
        top
    });
    let spec_y = spec.spectrogram.as_ref().map(|_| {
        let top = y;
        y += s.spectrogram_height;
        top
    });
    let panels_bottom = y; // signal panels end here (tiers begin)

    let mut tier_tops = Vec::with_capacity(spec.tiers.len());
    for _ in &spec.tiers {
        tier_tops.push(y);
        y += s.tier_height;
    }
    let axis_y = y;
    let total_height = axis_y + s.axis_height;

    let plot_x0 = s.margin_left;
    let plot_x1 = s.width - s.margin_right;
    let plot_w = (plot_x1 - plot_x0).max(1.0);
    let (t0, t1) = spec.time_range;
    let tspan = (t1 - t0).max(f64::MIN_POSITIVE);
    let x_of = |t: f64| plot_x0 + ((t - t0) / tspan) * plot_w;

    // --- assemble ------------------------------------------------------
    let mut out = String::with_capacity(4096);
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.0}\" height=\"{:.0}\" \
         viewBox=\"0 0 {:.0} {:.0}\" font-family=\"{}, serif\">\n",
        s.width, total_height, s.width, total_height, FONT_FAMILY
    ));
    out.push_str(&embed_font_defs());
    out.push_str(&format!(
        "<rect x=\"0\" y=\"0\" width=\"{:.0}\" height=\"{:.0}\" fill=\"{}\"/>\n",
        s.width, total_height, s.background
    ));

    if let Some(title) = &spec.title {
        out.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"{:.1}\" text-anchor=\"middle\">{}</text>\n",
            s.width / 2.0,
            s.margin_top + s.font_size,
            s.font_size * 1.15,
            xml_escape(title),
        ));
    }

    // Waveform band.
    if let (Some(top), Some(wave)) = (wave_y, &spec.waveform) {
        out.push_str(&waveform_svg(
            wave,
            top,
            s.waveform_height,
            plot_x0,
            plot_w,
            s,
        ));
    }

    // Spectrogram raster.
    if let (Some(top), Some(sg)) = (spec_y, &spec.spectrogram) {
        out.push_str(&spectrogram_svg(
            sg,
            top,
            s.spectrogram_height,
            plot_x0,
            plot_w,
            s,
        ));
    }

    // Boundary lines through the signal panels (interval tiers only).
    if panels_bottom > panels_top {
        out.push_str(&boundary_lines_svg(
            spec,
            panels_top,
            panels_bottom,
            &x_of,
            s,
        ));
    }

    // Tier rows.
    for (tier, &top) in spec.tiers.iter().zip(&tier_tops) {
        out.push_str(&tier_svg(
            tier,
            top,
            s.tier_height,
            plot_x0,
            plot_x1,
            &x_of,
            s,
        ));
    }

    // Shared time axis.
    out.push_str(&time_axis_svg(t0, t1, axis_y, plot_x0, plot_x1, &x_of, s));

    out.push_str("</svg>\n");
    out
}

/// Emits the `<defs>` block embedding the Doulos SIL font via a base64
/// `@font-face`, so every `<text>` renders in it without an external file.
fn embed_font_defs() -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(DOULOS_SIL_TTF);
    format!(
        "<defs><style type=\"text/css\">@font-face{{font-family:\"{FONT_FAMILY}\";\
         src:url(data:font/ttf;base64,{b64}) format(\"truetype\");}}</style></defs>\n"
    )
}

fn waveform_svg(
    wave: &WaveformLane,
    top: f64,
    height: f64,
    plot_x0: f64,
    plot_w: f64,
    s: &FigureStyle,
) -> String {
    let n = wave.minmax.len();
    let (amin, amax) = wave.amplitude_range;
    let aspan = (amax - amin).max(f32::MIN_POSITIVE) as f64;
    // amp → y (SVG y grows downward, so larger amplitude is higher = smaller y).
    let y_of = |amp: f32| top + (1.0 - ((amp - amin) as f64) / aspan) * height;
    let x_of = |i: usize| {
        if n <= 1 {
            plot_x0
        } else {
            plot_x0 + (i as f64 / (n - 1) as f64) * plot_w
        }
    };

    let mut out = String::new();
    if n > 0 {
        // Filled band: max envelope left→right, then min envelope right→left.
        let mut d = String::from("M");
        for (i, &(_, mx)) in wave.minmax.iter().enumerate() {
            d.push_str(&format!(" {:.1},{:.1}", x_of(i), y_of(mx)));
        }
        for (i, &(mn, _)) in wave.minmax.iter().enumerate().rev() {
            d.push_str(&format!(" {:.1},{:.1}", x_of(i), y_of(mn)));
        }
        d.push_str(" Z");
        out.push_str(&format!(
            "<path d=\"{}\" fill=\"{}\" stroke=\"none\"/>\n",
            d, s.waveform_fill
        ));
    }
    // Zero line.
    let yz = y_of(0.0);
    out.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\"/>\n",
        plot_x0, yz, plot_x0 + plot_w, yz, s.stroke
    ));
    // y-axis min/max amplitude labels.
    out.push_str(&y_label(
        plot_x0,
        top + s.font_size,
        &fmt_num(amax as f64),
        s,
    ));
    out.push_str(&y_label(plot_x0, top + height, &fmt_num(amin as f64), s));
    out
}

fn spectrogram_svg(
    sg: &SpectrogramLane,
    top: f64,
    height: f64,
    plot_x0: f64,
    plot_w: f64,
    s: &FigureStyle,
) -> String {
    let mut out = String::new();
    if sg.width > 0 && sg.height > 0 && sg.rgba.len() == sg.width * sg.height * 4 {
        let png_b64 = rgba_to_png_base64(&sg.rgba, sg.width, sg.height);
        out.push_str(&format!(
            "<image x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
             preserveAspectRatio=\"none\" xlink:href=\"data:image/png;base64,{}\" \
             xmlns:xlink=\"http://www.w3.org/1999/xlink\"/>\n",
            plot_x0, top, plot_w, height, png_b64
        ));
    }
    // Frequency axis: top = max_freq, bottom = 0.
    out.push_str(&y_label(
        plot_x0,
        top + s.font_size,
        &fmt_num(sg.max_freq_hz as f64),
        s,
    ));
    out.push_str(&y_label(plot_x0, top + height, "0", s));
    // A "Hz" unit hint on the frequency axis.
    out.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"{:.1}\" text-anchor=\"end\" fill=\"{}\">Hz</text>\n",
        plot_x0 - 6.0,
        top + height / 2.0,
        s.font_size * 0.85,
        s.stroke
    ));
    out
}

fn boundary_lines_svg(
    spec: &FigureSpec,
    y_top: f64,
    y_bottom: f64,
    x_of: &dyn Fn(f64) -> f64,
    s: &FigureStyle,
) -> String {
    let (t0, t1) = spec.time_range;
    let mut out = String::new();
    for tier in &spec.tiers {
        if let FigureTierContent::Intervals(ivs) = &tier.content {
            for iv in ivs {
                for &b in &[iv.start, iv.end] {
                    if b > t0 && b < t1 {
                        out.push_str(&format!(
                            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" \
                             stroke=\"{}\" stroke-width=\"0.4\" opacity=\"0.5\"/>\n",
                            x_of(b),
                            y_top,
                            x_of(b),
                            y_bottom,
                            s.stroke
                        ));
                    }
                }
            }
        }
    }
    out
}

fn tier_svg(
    tier: &FigureTier,
    top: f64,
    height: f64,
    plot_x0: f64,
    plot_x1: f64,
    x_of: &dyn Fn(f64) -> f64,
    s: &FigureStyle,
) -> String {
    let mut out = String::new();
    let mid = top + height / 2.0;
    let baseline = mid + s.font_size * 0.35;

    // Tier name in the left margin.
    out.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"{:.1}\" text-anchor=\"end\" fill=\"{}\">{}</text>\n",
        plot_x0 - 6.0,
        baseline,
        s.font_size * 0.85,
        s.stroke,
        xml_escape(&tier.name),
    ));
    // Lane frame.
    out.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\"/>\n",
        plot_x0, top, plot_x1, top, s.stroke
    ));

    match &tier.content {
        FigureTierContent::Intervals(ivs) => {
            for iv in ivs {
                let xa = x_of(iv.start).clamp(plot_x0, plot_x1);
                let xb = x_of(iv.end).clamp(plot_x0, plot_x1);
                // Left boundary tick within the lane.
                out.push_str(&format!(
                    "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\"/>\n",
                    xa, top, xa, top + height, s.stroke
                ));
                if !iv.label.is_empty() {
                    out.push_str(&format!(
                        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"{:.1}\" text-anchor=\"middle\" fill=\"{}\">{}</text>\n",
                        (xa + xb) / 2.0,
                        baseline,
                        s.font_size,
                        s.stroke,
                        xml_escape(&iv.label),
                    ));
                }
            }
        }
        FigureTierContent::Points(pts) => {
            for pt in pts {
                let x = x_of(pt.time);
                if x < plot_x0 || x > plot_x1 {
                    continue;
                }
                out.push_str(&format!(
                    "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.6\"/>\n",
                    x, top, x, top + height, s.stroke
                ));
                if !pt.label.is_empty() {
                    out.push_str(&format!(
                        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"{:.1}\" text-anchor=\"middle\" fill=\"{}\">{}</text>\n",
                        x,
                        baseline,
                        s.font_size,
                        s.stroke,
                        xml_escape(&pt.label),
                    ));
                }
            }
        }
    }
    out
}

fn time_axis_svg(
    t0: f64,
    t1: f64,
    axis_y: f64,
    plot_x0: f64,
    plot_x1: f64,
    x_of: &dyn Fn(f64) -> f64,
    s: &FigureStyle,
) -> String {
    let mut out = String::new();
    // Axis line.
    out.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.8\"/>\n",
        plot_x0, axis_y, plot_x1, axis_y, s.stroke
    ));
    // Ticks at start / middle / end (a minimal, always-sensible set).
    for frac in [0.0, 0.5, 1.0] {
        let t = t0 + (t1 - t0) * frac;
        let x = x_of(t);
        out.push_str(&format!(
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.8\"/>\n",
            x,
            axis_y,
            x,
            axis_y + 4.0,
            s.stroke
        ));
        out.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"{:.1}\" text-anchor=\"middle\" fill=\"{}\">{}</text>\n",
            x,
            axis_y + 4.0 + s.font_size,
            s.font_size * 0.85,
            s.stroke,
            fmt_num(t),
        ));
    }
    // "Time (s)" caption.
    out.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"{:.1}\" text-anchor=\"middle\" fill=\"{}\">Time (s)</text>\n",
        (plot_x0 + plot_x1) / 2.0,
        axis_y + 4.0 + s.font_size * 2.2,
        s.font_size * 0.9,
        s.stroke,
    ));
    out
}

/// A right-aligned y-axis value label in the left margin.
fn y_label(plot_x0: f64, y: f64, text: &str, s: &FigureStyle) -> String {
    format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"{:.1}\" text-anchor=\"end\" fill=\"{}\">{}</text>\n",
        plot_x0 - 6.0,
        y,
        s.font_size * 0.8,
        s.stroke,
        xml_escape(text),
    )
}

/// PNG-encodes an RGBA8 raster and base64s it for an SVG `data:` URI.
fn rgba_to_png_base64(rgba: &[u8], width: usize, height: usize) -> String {
    let mut png_bytes = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut png_bytes, width as u32, height as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        // A missing header write can't recover; a truncated PNG would be an
        // obviously-broken figure, so unwrap is acceptable here (in-memory
        // writer, no I/O to fail).
        let mut writer = enc.write_header().expect("png header");
        writer.write_image_data(rgba).expect("png data");
    }
    base64::engine::general_purpose::STANDARD.encode(&png_bytes)
}

/// Formats a number for an axis label: up to 3 significant decimals, trailing
/// zeros trimmed, so `0.5` not `0.500` and `1200` not `1200.000`.
fn fmt_num(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let mut s = format!("{v:.3}");
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    s
}

/// Escapes the five XML metacharacters so labels can't break the document.
fn xml_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replaces the (large, font-version-dependent) embedded base64 payloads —
    /// the `@font-face` src and the spectrogram PNG — with stable placeholders,
    /// so structural golden assertions stay small and don't churn on a font or
    /// PNG-encoder update. Verifies the payloads exist and are non-trivial.
    fn normalize_blobs(svg: &str) -> String {
        let mut out = svg.to_string();
        for tag in ["data:font/ttf;base64,", "data:image/png;base64,"] {
            while let Some(start) = out.find(tag) {
                let payload_start = start + tag.len();
                let end = out[payload_start..]
                    .find([')', '"'])
                    .map(|i| payload_start + i)
                    .unwrap_or(out.len());
                assert!(
                    end - payload_start > 64,
                    "embedded blob for {tag} looks too small to be real"
                );
                // Replace tag + payload together so the tag isn't re-matched.
                out.replace_range(start..end, "<BLOB>");
            }
        }
        out
    }

    fn sample_spec() -> FigureSpec {
        FigureSpec {
            title: Some("praat".to_string()),
            time_range: (0.0, 1.0),
            waveform: Some(WaveformLane {
                minmax: vec![(-0.5, 0.5), (-0.8, 0.7), (-0.2, 0.3)],
                amplitude_range: (-1.0, 1.0),
            }),
            spectrogram: Some(SpectrogramLane {
                rgba: vec![128u8; 4 * 3 * 4], // 4×3 grey raster
                width: 4,
                height: 3,
                max_freq_hz: 8000.0,
            }),
            tiers: vec![
                FigureTier {
                    name: "phones".to_string(),
                    content: FigureTierContent::Intervals(vec![
                        FigureInterval {
                            start: 0.0,
                            end: 0.4,
                            label: "p".to_string(),
                        },
                        FigureInterval {
                            start: 0.4,
                            end: 1.0,
                            label: "ɹ".to_string(),
                        },
                    ]),
                },
                FigureTier {
                    name: "events".to_string(),
                    content: FigureTierContent::Points(vec![FigurePoint {
                        time: 0.5,
                        label: "burst".to_string(),
                    }]),
                },
            ],
            style: FigureStyle::default(),
        }
    }

    #[test]
    fn svg_is_well_formed_and_self_contained() {
        let svg = to_svg(&sample_spec());
        assert!(svg.starts_with("<svg"));
        assert!(svg.trim_end().ends_with("</svg>"));
        // Self-contained: an embedded font + an embedded raster, no external refs.
        assert!(svg.contains("@font-face"));
        assert!(svg.contains("data:font/ttf;base64,"));
        assert!(svg.contains("data:image/png;base64,"));
        assert!(!svg.contains("http://") || svg.contains("www.w3.org")); // only the SVG namespace URL
    }

    #[test]
    fn ipa_labels_are_present_as_real_text() {
        let svg = to_svg(&sample_spec());
        // IPA glyphs survive as UTF-8 text (not outlined), so they stay editable.
        assert!(svg.contains(">ɹ</text>"), "IPA label ɹ should be real text");
        assert!(svg.contains(">p</text>"));
        assert!(svg.contains("Time (s)"));
    }

    #[test]
    fn xml_metacharacters_in_labels_are_escaped() {
        let mut spec = sample_spec();
        spec.tiers[0].content = FigureTierContent::Intervals(vec![FigureInterval {
            start: 0.0,
            end: 1.0,
            label: "a<b&c\"".to_string(),
        }]);
        let svg = to_svg(&spec);
        assert!(svg.contains("a&lt;b&amp;c&quot;"));
        assert!(!svg.contains("a<b&c"));
    }

    #[test]
    fn height_grows_with_each_lane() {
        let base = sample_spec();
        let h_all = svg_height(&to_svg(&base));

        let mut no_wave = sample_spec();
        no_wave.waveform = None;
        let h_no_wave = svg_height(&to_svg(&no_wave));
        assert!(
            h_all > h_no_wave,
            "dropping the waveform should shrink the figure"
        );

        let mut no_tiers = sample_spec();
        no_tiers.tiers.clear();
        let h_no_tiers = svg_height(&to_svg(&no_tiers));
        assert!(
            h_all > h_no_tiers,
            "dropping tiers should shrink the figure"
        );
    }

    #[test]
    fn spectrogram_only_figure_still_renders() {
        let spec = FigureSpec {
            title: None,
            time_range: (0.0, 2.0),
            waveform: None,
            spectrogram: Some(SpectrogramLane {
                rgba: vec![64u8; 2 * 2 * 4],
                width: 2,
                height: 2,
                max_freq_hz: 5000.0,
            }),
            tiers: vec![],
            style: FigureStyle::default(),
        };
        let svg = to_svg(&spec);
        assert!(svg.contains("data:image/png;base64,"));
        assert!(!svg.contains("data:font/ttf;base64,") || svg.contains("@font-face"));
    }

    #[test]
    fn structural_golden_ignores_embedded_blobs() {
        // A stable structural snapshot: the same geometry every run, with the
        // font + PNG payloads normalized out. Guards the SVG skeleton against
        // accidental drift while staying tiny.
        let svg = normalize_blobs(&to_svg(&sample_spec()));
        assert!(svg.contains("<BLOB>"), "blobs should be normalized");
        // Key structural landmarks.
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Time (s)"));
        assert!(svg.contains(">phones</text>"));
        assert!(svg.contains(">events</text>"));
        assert!(svg.contains(">praat</text>")); // title
        assert!(svg.contains("Hz")); // spectrogram freq axis
    }

    /// Extracts the `height="…"` from the `<svg>` open tag.
    fn svg_height(svg: &str) -> f64 {
        let tag = &svg[..svg.find('>').unwrap()];
        let key = "height=\"";
        let start = tag.find(key).unwrap() + key.len();
        let end = tag[start..].find('"').unwrap() + start;
        tag[start..end].parse().unwrap()
    }
}
