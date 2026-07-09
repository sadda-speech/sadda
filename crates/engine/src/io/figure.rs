//! Publication figure export — a `FigureSpec` intermediate representation and
//! pluggable serializers: [`to_svg`] (self-contained SVG), `to_pdf` (via
//! `svg2pdf`, `figure-pdf` feature), and [`to_tikz`] (standalone TikZ/LaTeX).
//!
//! Mirrors the [`super::tabular`] split: a plain-data IR ([`FigureSpec`] +
//! its lanes) built by the caller, and free serializer functions that render
//! it off a shared [`FigureLayout`] — so the same figure description can
//! target every backend consistently and be assembled headlessly (Python, the
//! doc-render harness) or from the GUI.
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
    /// Measure-track lanes (f0 / formants / intensity …), each its own stacked
    /// row between the spectrogram and the tiers.
    pub measures: Vec<MeasureLane>,
    /// Heatmap lanes (MFCC / embedding rasters), each a baked colormap raster
    /// in its own row.
    pub heatmaps: Vec<HeatmapLane>,
    /// Annotation-tier rows, drawn top-to-bottom in the given order.
    pub tiers: Vec<FigureTier>,
    /// Sizing, fonts, and colours.
    pub style: FigureStyle,
}

/// A stacked measure-track lane — one or more time-series (f0, the formants,
/// intensity, …) drawn over the lane's own y-axis, in its own row.
pub struct MeasureLane {
    /// Lane name, drawn in the left margin (e.g. `"f0"`, `"formants"`).
    pub name: String,
    /// y-axis unit label (e.g. `"Hz"`, `"dB"`; empty for none).
    pub unit: String,
    /// The value extent mapped to the lane's vertical span, `(min, max)`.
    pub y_range: (f64, f64),
    /// One or more series drawn in the lane (e.g. F1…F5 as separate series).
    pub series: Vec<MeasureSeries>,
}

/// A heatmap lane — a baked colormap raster (MFCC coefficients, an embedding
/// matrix, …) in its own row, like the spectrogram but for a generic matrix.
pub struct HeatmapLane {
    /// Lane name, drawn in the left margin (e.g. `"MFCC"`).
    pub name: String,
    /// Row-major RGBA8 (`width * height * 4`), row 0 = top — as
    /// [`crate::dsp::colormap_bake`] produces.
    pub rgba: Vec<u8>,
    /// Raster width in cells (time frames).
    pub width: usize,
    /// Raster height in cells (matrix rows, e.g. coefficients).
    pub height: usize,
    /// Top y-axis label (e.g. the highest coefficient index).
    pub top_label: String,
    /// Bottom y-axis label (e.g. the lowest coefficient index).
    pub bottom_label: String,
    /// Sidecar PNG filename for backends that can't inline a raster (TikZ);
    /// `None` for the inlining backends (SVG/PDF). Set by the exporter.
    pub raster_ref: Option<String>,
}

/// One time-series in a [`MeasureLane`].
pub struct MeasureSeries {
    /// Polyline segments — each an unbroken run of `(time_seconds, value)`
    /// points; gaps (e.g. f0 unvoiced frames) split into separate segments so
    /// the line doesn't bridge them.
    pub segments: Vec<Vec<(f64, f64)>>,
    /// Draw discrete dots (the convention for formants) rather than connecting
    /// the points into a line.
    pub dots: bool,
    /// CSS stroke/fill colour; falls back to the style stroke when `None`.
    pub color: Option<String>,
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
    /// Height of each measure-track lane row (f0 / formants / intensity).
    pub measure_height: f64,
    /// Height of each heatmap lane row (MFCC / embedding).
    pub heatmap_height: f64,
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
            measure_height: 80.0,
            heatmap_height: 120.0,
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

/// The computed vertical + horizontal geometry of a figure — where each lane
/// sits and how time maps to x. Shared by every serializer ([`to_svg`],
/// [`to_tikz`]) so the SVG, PDF, and TikZ backends can't drift apart.
struct FigureLayout {
    /// Top y of the signal-panel region (boundary lines start here).
    panels_top: f64,
    /// Top y of the waveform lane, if present.
    wave_y: Option<f64>,
    /// Top y of the spectrogram lane, if present.
    spec_y: Option<f64>,
    /// Top y of each measure lane, in order.
    measure_ys: Vec<f64>,
    /// Top y of each heatmap lane, in order.
    heatmap_ys: Vec<f64>,
    /// Bottom y of the signal-panel region (tiers start here; boundary lines
    /// end here — they cross the waveform, spectrogram, measure, and heatmap
    /// lanes).
    panels_bottom: f64,
    /// Top y of each tier row, in order.
    tier_tops: Vec<f64>,
    /// y of the shared time axis line.
    axis_y: f64,
    /// Total figure height.
    total_height: f64,
    /// Left / right / width of the plot area (between the margins).
    plot_x0: f64,
    plot_x1: f64,
    plot_w: f64,
    /// Time window.
    t0: f64,
    t1: f64,
}

impl FigureLayout {
    fn compute(spec: &FigureSpec) -> Self {
        let s = &spec.style;
        let mut y = s.margin_top + s.title_height(spec.title.is_some());
        let panels_top = y;

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
        let mut measure_ys = Vec::with_capacity(spec.measures.len());
        for _ in &spec.measures {
            measure_ys.push(y);
            y += s.measure_height;
        }
        let mut heatmap_ys = Vec::with_capacity(spec.heatmaps.len());
        for _ in &spec.heatmaps {
            heatmap_ys.push(y);
            y += s.heatmap_height;
        }
        let panels_bottom = y;

        let mut tier_tops = Vec::with_capacity(spec.tiers.len());
        for _ in &spec.tiers {
            tier_tops.push(y);
            y += s.tier_height;
        }
        let axis_y = y;
        let total_height = axis_y + s.axis_height;

        let plot_x0 = s.margin_left;
        let plot_x1 = s.width - s.margin_right;
        let (t0, t1) = spec.time_range;
        Self {
            panels_top,
            wave_y,
            spec_y,
            measure_ys,
            heatmap_ys,
            panels_bottom,
            tier_tops,
            axis_y,
            total_height,
            plot_x0,
            plot_x1,
            plot_w: (plot_x1 - plot_x0).max(1.0),
            t0,
            t1,
        }
    }

    /// Maps a time in seconds to an x coordinate in the plot area.
    fn x_of(&self, t: f64) -> f64 {
        let span = (self.t1 - self.t0).max(f64::MIN_POSITIVE);
        self.plot_x0 + ((t - self.t0) / span) * self.plot_w
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
    let lay = FigureLayout::compute(spec);
    let FigureLayout {
        panels_top,
        wave_y,
        spec_y,
        ref measure_ys,
        ref heatmap_ys,
        panels_bottom,
        ref tier_tops,
        axis_y,
        total_height,
        plot_x0,
        plot_x1,
        plot_w,
        t0,
        t1,
    } = lay;
    let x_of = |t: f64| lay.x_of(t);

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

    // Measure lanes (f0 / formants / intensity).
    for (lane, &top) in spec.measures.iter().zip(measure_ys) {
        out.push_str(&measure_svg(
            lane,
            top,
            s.measure_height,
            plot_x0,
            plot_x1,
            &x_of,
            s,
        ));
    }

    // Heatmap lanes (MFCC / embedding rasters).
    for (lane, &top) in spec.heatmaps.iter().zip(heatmap_ys) {
        out.push_str(&heatmap_svg(
            lane,
            top,
            s.heatmap_height,
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
    for (tier, &top) in spec.tiers.iter().zip(tier_tops) {
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

/// Serializes a [`FigureSpec`] to a **standalone TikZ/LaTeX document** (the
/// specTeX integration model). Compile with **XeLaTeX or LuaLaTeX** — it uses
/// `fontspec` + Doulos SIL so IPA typesets in the document's own font.
///
/// TikZ can't embed a raster, so the spectrogram is not inlined: if the figure
/// has one, pass `raster_ref` = the filename of a sidecar PNG (written next to
/// the `.tex` by [`crate::corpus::Project::export_figure`]) and it is
/// `\includegraphics`'d; pass `None` to omit it.
///
/// Everything else (waveform band, tier boxes, panel-crossing boundary lines,
/// axes, text) is native vector TikZ off the shared [`FigureLayout`], so it
/// matches the SVG/PDF backends. Coordinates are px-as-pt, y flipped for TikZ's
/// y-up convention; scale on `\includegraphics` when embedding.
pub fn to_tikz(spec: &FigureSpec, raster_ref: Option<&str>) -> String {
    let s = &spec.style;
    let lay = FigureLayout::compute(spec);
    let h = lay.total_height;
    // TikZ is y-up; map top-down SVG-space y to (h - y). Emit px as pt.
    let ty = |y: f64| h - y;

    let mut b = String::with_capacity(4096);
    b.push_str("% Publication figure exported by sadda.\n");
    b.push_str("% Compile with XeLaTeX or LuaLaTeX (fontspec + Doulos SIL for IPA).\n");
    b.push_str("% The spectrogram is the sidecar PNG referenced below.\n");
    b.push_str("\\documentclass[border=5pt]{standalone}\n");
    b.push_str("\\usepackage{tikz}\n\\usepackage{graphicx}\n\\usepackage{fontspec}\n");
    b.push_str(
        "\\setmainfont{Doulos SIL}% IPA reference face (SIL OFL); install it or add [Path=...]\n",
    );
    b.push_str("\\begin{document}\n\\begin{tikzpicture}[x=1pt,y=1pt]\n");
    b.push_str(&format!(
        "\\definecolor{{figstroke}}{{HTML}}{{{}}}\n",
        hex_rgb(&s.stroke)
    ));
    b.push_str(&format!(
        "\\definecolor{{figwave}}{{HTML}}{{{}}}\n",
        hex_rgb(&s.waveform_fill)
    ));

    if let Some(title) = &spec.title {
        b.push_str(&format!(
            "\\node[anchor=north] at ({:.1},{:.1}) {{\\large {}}};\n",
            s.width / 2.0,
            ty(s.margin_top),
            tex_escape(title),
        ));
    }

    // Waveform band + zero line + amplitude labels.
    if let (Some(top), Some(wave)) = (lay.wave_y, &spec.waveform) {
        let (amin, amax) = wave.amplitude_range;
        let aspan = (amax - amin).max(f32::MIN_POSITIVE) as f64;
        let n = wave.minmax.len();
        let wx = |i: usize| {
            if n <= 1 {
                lay.plot_x0
            } else {
                lay.plot_x0 + (i as f64 / (n - 1) as f64) * lay.plot_w
            }
        };
        let wy = |amp: f32| ty(top + (1.0 - ((amp - amin) as f64) / aspan) * s.waveform_height);
        if n > 0 {
            b.push_str("\\fill[figwave] ");
            for (i, &(_, mx)) in wave.minmax.iter().enumerate() {
                b.push_str(&format!("({:.1},{:.1}) -- ", wx(i), wy(mx)));
            }
            for (i, &(mn, _)) in wave.minmax.iter().enumerate().rev() {
                b.push_str(&format!("({:.1},{:.1}) -- ", wx(i), wy(mn)));
            }
            b.push_str("cycle;\n");
        }
        let yz = wy(0.0);
        b.push_str(&format!(
            "\\draw[figstroke,line width=0.4pt] ({:.1},{:.1}) -- ({:.1},{:.1});\n",
            lay.plot_x0,
            yz,
            lay.plot_x0 + lay.plot_w,
            yz,
        ));
        b.push_str(&tikz_ylabel(
            lay.plot_x0,
            ty(top + s.font_size),
            &fmt_num(amax as f64),
        ));
        b.push_str(&tikz_ylabel(
            lay.plot_x0,
            ty(top + s.waveform_height),
            &fmt_num(amin as f64),
        ));
    }

    // Spectrogram: the sidecar raster + frequency labels.
    if let (Some(top), Some(sg)) = (lay.spec_y, &spec.spectrogram) {
        if let Some(r) = raster_ref {
            b.push_str(&format!(
                "\\node[anchor=north west,inner sep=0] at ({:.1},{:.1}) \
                 {{\\includegraphics[width={:.1}pt,height={:.1}pt]{{{}}}}};\n",
                lay.plot_x0,
                ty(top),
                lay.plot_w,
                s.spectrogram_height,
                r,
            ));
        }
        b.push_str(&tikz_ylabel(
            lay.plot_x0,
            ty(top + s.font_size),
            &fmt_num(sg.max_freq_hz as f64),
        ));
        b.push_str(&tikz_ylabel(
            lay.plot_x0,
            ty(top + s.spectrogram_height),
            "0",
        ));
        b.push_str(&format!(
            "\\node[anchor=east,font=\\small] at ({:.1},{:.1}) {{Hz}};\n",
            lay.plot_x0 - 6.0,
            ty(top + s.spectrogram_height / 2.0),
        ));
    }

    // Measure lanes (f0 / formants / intensity).
    for (lane, &top) in spec.measures.iter().zip(&lay.measure_ys) {
        let (lo, hi) = lane.y_range;
        let vspan = (hi - lo).max(f64::MIN_POSITIVE);
        let my = |v: f64| ty(top + (1.0 - (v - lo) / vspan) * s.measure_height);
        b.push_str(&format!(
            "\\node[anchor=east,font=\\small] at ({:.1},{:.1}) {{{}}};\n",
            lay.plot_x0 - 6.0,
            ty(top + s.measure_height / 2.0),
            tex_escape(&lane.name),
        ));
        b.push_str(&format!(
            "\\draw[figstroke,line width=0.4pt] ({:.1},{:.1}) -- ({:.1},{:.1});\n",
            lay.plot_x0,
            ty(top),
            lay.plot_x1,
            ty(top),
        ));
        b.push_str(&tikz_ylabel(
            lay.plot_x0,
            ty(top + s.font_size),
            &measure_top_label(hi, &lane.unit),
        ));
        b.push_str(&tikz_ylabel(
            lay.plot_x0,
            ty(top + s.measure_height),
            &fmt_num(lo),
        ));
        for series in &lane.series {
            let col = series
                .color
                .as_deref()
                .map(hex_rgb)
                .unwrap_or_else(|| hex_rgb(&s.stroke));
            b.push_str(&format!("\\definecolor{{figseries}}{{HTML}}{{{col}}}\n"));
            for seg in &series.segments {
                if series.dots {
                    for &(t, v) in seg {
                        let x = lay.x_of(t);
                        if x >= lay.plot_x0 && x <= lay.plot_x1 {
                            b.push_str(&format!(
                                "\\fill[figseries] ({:.1},{:.1}) circle (1.0pt);\n",
                                x,
                                my(v),
                            ));
                        }
                    }
                } else if seg.len() >= 2 {
                    b.push_str("\\draw[figseries,line width=0.7pt] ");
                    for (i, &(t, v)) in seg.iter().enumerate() {
                        b.push_str(&format!(
                            "{}({:.1},{:.1})",
                            if i == 0 { "" } else { " -- " },
                            lay.x_of(t).clamp(lay.plot_x0, lay.plot_x1),
                            my(v),
                        ));
                    }
                    b.push_str(";\n");
                }
            }
        }
    }

    // Heatmap lanes (MFCC / embedding): each a sidecar raster, like the
    // spectrogram, plus name + top/bottom labels.
    for (lane, &top) in spec.heatmaps.iter().zip(&lay.heatmap_ys) {
        if let Some(r) = &lane.raster_ref {
            b.push_str(&format!(
                "\\node[anchor=north west,inner sep=0] at ({:.1},{:.1}) \
                 {{\\includegraphics[width={:.1}pt,height={:.1}pt]{{{}}}}};\n",
                lay.plot_x0,
                ty(top),
                lay.plot_w,
                s.heatmap_height,
                r,
            ));
        }
        b.push_str(&format!(
            "\\node[anchor=east,font=\\small] at ({:.1},{:.1}) {{{}}};\n",
            lay.plot_x0 - 6.0,
            ty(top + s.heatmap_height / 2.0),
            tex_escape(&lane.name),
        ));
        b.push_str(&tikz_ylabel(
            lay.plot_x0,
            ty(top + s.font_size),
            &lane.top_label,
        ));
        b.push_str(&tikz_ylabel(
            lay.plot_x0,
            ty(top + s.heatmap_height),
            &lane.bottom_label,
        ));
    }

    // Boundary lines through the signal panels (interval tiers only).
    if lay.panels_bottom > lay.panels_top {
        for tier in &spec.tiers {
            if let FigureTierContent::Intervals(ivs) = &tier.content {
                for iv in ivs {
                    for &bd in &[iv.start, iv.end] {
                        if bd > lay.t0 && bd < lay.t1 {
                            let x = lay.x_of(bd);
                            b.push_str(&format!(
                                "\\draw[figstroke,opacity=0.5,line width=0.3pt] \
                                 ({:.1},{:.1}) -- ({:.1},{:.1});\n",
                                x,
                                ty(lay.panels_top),
                                x,
                                ty(lay.panels_bottom),
                            ));
                        }
                    }
                }
            }
        }
    }

    // Tier rows.
    for (tier, &top) in spec.tiers.iter().zip(&lay.tier_tops) {
        let mid = top + s.tier_height / 2.0;
        b.push_str(&format!(
            "\\node[anchor=east,font=\\small] at ({:.1},{:.1}) {{{}}};\n",
            lay.plot_x0 - 6.0,
            ty(mid),
            tex_escape(&tier.name),
        ));
        b.push_str(&format!(
            "\\draw[figstroke,line width=0.4pt] ({:.1},{:.1}) -- ({:.1},{:.1});\n",
            lay.plot_x0,
            ty(top),
            lay.plot_x1,
            ty(top),
        ));
        match &tier.content {
            FigureTierContent::Intervals(ivs) => {
                for iv in ivs {
                    let xa = lay.x_of(iv.start).clamp(lay.plot_x0, lay.plot_x1);
                    let xb = lay.x_of(iv.end).clamp(lay.plot_x0, lay.plot_x1);
                    b.push_str(&format!(
                        "\\draw[figstroke,line width=0.4pt] ({:.1},{:.1}) -- ({:.1},{:.1});\n",
                        xa,
                        ty(top),
                        xa,
                        ty(top + s.tier_height),
                    ));
                    if !iv.label.is_empty() {
                        b.push_str(&format!(
                            "\\node at ({:.1},{:.1}) {{{}}};\n",
                            (xa + xb) / 2.0,
                            ty(mid),
                            tex_escape(&iv.label),
                        ));
                    }
                }
            }
            FigureTierContent::Points(pts) => {
                for pt in pts {
                    let x = lay.x_of(pt.time);
                    if x < lay.plot_x0 || x > lay.plot_x1 {
                        continue;
                    }
                    b.push_str(&format!(
                        "\\draw[figstroke,line width=0.5pt] ({:.1},{:.1}) -- ({:.1},{:.1});\n",
                        x,
                        ty(top),
                        x,
                        ty(top + s.tier_height),
                    ));
                    if !pt.label.is_empty() {
                        b.push_str(&format!(
                            "\\node at ({:.1},{:.1}) {{{}}};\n",
                            x,
                            ty(mid),
                            tex_escape(&pt.label),
                        ));
                    }
                }
            }
        }
    }

    // Shared time axis.
    b.push_str(&format!(
        "\\draw[figstroke,line width=0.6pt] ({:.1},{:.1}) -- ({:.1},{:.1});\n",
        lay.plot_x0,
        ty(lay.axis_y),
        lay.plot_x1,
        ty(lay.axis_y),
    ));
    for frac in [0.0, 0.5, 1.0] {
        let t = lay.t0 + (lay.t1 - lay.t0) * frac;
        let x = lay.x_of(t);
        b.push_str(&format!(
            "\\draw[figstroke,line width=0.6pt] ({:.1},{:.1}) -- ({:.1},{:.1});\n",
            x,
            ty(lay.axis_y),
            x,
            ty(lay.axis_y + 4.0),
        ));
        b.push_str(&format!(
            "\\node[anchor=north,font=\\small] at ({:.1},{:.1}) {{{}}};\n",
            x,
            ty(lay.axis_y + 5.0),
            fmt_num(t),
        ));
    }
    b.push_str(&format!(
        "\\node[anchor=north,font=\\small] at ({:.1},{:.1}) {{Time (s)}};\n",
        (lay.plot_x0 + lay.plot_x1) / 2.0,
        ty(lay.axis_y + 5.0 + s.font_size * 1.6),
    ));

    b.push_str("\\end{tikzpicture}\n\\end{document}\n");
    b
}

/// A right-aligned TikZ y-axis label node.
fn tikz_ylabel(plot_x0: f64, y: f64, text: &str) -> String {
    format!(
        "\\node[anchor=east,font=\\small] at ({:.1},{:.1}) {{{}}};\n",
        plot_x0 - 6.0,
        y,
        tex_escape(text),
    )
}

/// Extracts a 6-hex-digit `RRGGBB` from a `#rrggbb` CSS colour for xcolor's
/// `{HTML}` model, falling back to black for anything else (named colours,
/// short hex — the style defaults are all full hex).
fn hex_rgb(css: &str) -> String {
    let h = css.trim_start_matches('#');
    if h.len() == 6 && h.bytes().all(|c| c.is_ascii_hexdigit()) {
        h.to_ascii_uppercase()
    } else {
        "000000".to_string()
    }
}

/// Escapes the LaTeX special characters so a tier/axis label can't break the
/// document. IPA and other Unicode pass through (rendered via fontspec).
fn tex_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("\\&"),
            '%' => out.push_str("\\%"),
            '$' => out.push_str("\\$"),
            '#' => out.push_str("\\#"),
            '_' => out.push_str("\\_"),
            '{' => out.push_str("\\{"),
            '}' => out.push_str("\\}"),
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            '\\' => out.push_str("\\textbackslash{}"),
            _ => out.push(c),
        }
    }
    out
}

/// Serializes a [`FigureSpec`] to a **PDF** (via SVG → `svg2pdf`). Requires the
/// `figure-pdf` feature (the app and Python wheel enable it).
///
/// `usvg` flattens the figure's `<text>` to vector outlines — so the PDF is
/// fully self-contained (no font dependency to *view*) at the cost of the text
/// no longer being selectable, unlike the SVG — and decodes the embedded
/// spectrogram PNG. The bundled Doulos SIL is loaded into the font database
/// directly so `Doulos SIL` always resolves, independent of the consumer's
/// `@font-face` handling.
#[cfg(feature = "figure-pdf")]
pub fn to_pdf(spec: &FigureSpec) -> Result<Vec<u8>, String> {
    use svg2pdf::usvg;
    let svg = to_svg(spec);
    let mut options = usvg::Options::default();
    options.fontdb_mut().load_font_data(DOULOS_SIL_TTF.to_vec());
    let tree = usvg::Tree::from_str(&svg, &options)
        .map_err(|e| format!("figure PDF: SVG parse failed: {e}"))?;
    svg2pdf::to_pdf(
        &tree,
        svg2pdf::ConversionOptions::default(),
        svg2pdf::PageOptions::default(),
    )
    .map_err(|e| format!("figure PDF: conversion failed: {e}"))
}

/// Rasterises a [`FigureSpec`] to an `(width, height, rgba)` bitmap (via SVG →
/// `resvg`). Requires the `figure-pdf` feature. Used by the GUI's
/// figure→clipboard action (clipboard images must be raster, not SVG).
///
/// The bytes are straight (un-premultiplied) RGBA8, row-major; the figure's own
/// opaque background means alpha is `255` throughout, so premultiplied and
/// straight coincide.
#[cfg(feature = "figure-pdf")]
pub fn to_rgba(spec: &FigureSpec) -> Result<(u32, u32, Vec<u8>), String> {
    use resvg::tiny_skia;
    use resvg::usvg;
    let svg = to_svg(spec);
    let mut options = usvg::Options::default();
    options.fontdb_mut().load_font_data(DOULOS_SIL_TTF.to_vec());
    let tree = usvg::Tree::from_str(&svg, &options)
        .map_err(|e| format!("figure raster: SVG parse failed: {e}"))?;
    let size = tree.size().to_int_size();
    let (w, h) = (size.width(), size.height());
    let mut pixmap =
        tiny_skia::Pixmap::new(w, h).ok_or_else(|| "figure raster: invalid size".to_string())?;
    resvg::render(
        &tree,
        tiny_skia::Transform::identity(),
        &mut pixmap.as_mut(),
    );
    Ok((w, h, pixmap.take()))
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

fn measure_svg(
    lane: &MeasureLane,
    top: f64,
    height: f64,
    plot_x0: f64,
    plot_x1: f64,
    x_of: &dyn Fn(f64) -> f64,
    s: &FigureStyle,
) -> String {
    let (lo, hi) = lane.y_range;
    let span = (hi - lo).max(f64::MIN_POSITIVE);
    let y_of = |v: f64| top + (1.0 - (v - lo) / span) * height;

    let mut out = String::new();
    // Lane name + top frame line.
    out.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"{:.1}\" text-anchor=\"end\" fill=\"{}\">{}</text>\n",
        plot_x0 - 6.0,
        top + height / 2.0 + s.font_size * 0.35,
        s.font_size * 0.85,
        s.stroke,
        xml_escape(&lane.name),
    ));
    out.push_str(&format!(
        "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\"/>\n",
        plot_x0, top, plot_x1, top, s.stroke
    ));
    // y-axis extent labels — the unit rides on the top (max) label so it can't
    // collide with the lane name at mid-height.
    out.push_str(&y_label(
        plot_x0,
        top + s.font_size,
        &measure_top_label(hi, &lane.unit),
        s,
    ));
    out.push_str(&y_label(plot_x0, top + height, &fmt_num(lo), s));
    // Series.
    for series in &lane.series {
        let color = series.color.as_deref().unwrap_or(&s.stroke);
        for seg in &series.segments {
            if series.dots {
                for &(t, v) in seg {
                    let x = x_of(t);
                    if x < plot_x0 || x > plot_x1 {
                        continue;
                    }
                    out.push_str(&format!(
                        "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"1.3\" fill=\"{}\"/>\n",
                        x,
                        y_of(v),
                        color,
                    ));
                }
            } else if seg.len() >= 2 {
                let mut d = String::from("M");
                for (i, &(t, v)) in seg.iter().enumerate() {
                    d.push_str(&format!(
                        "{} {:.1},{:.1}",
                        if i == 0 { "" } else { " L" },
                        x_of(t).clamp(plot_x0, plot_x1),
                        y_of(v),
                    ));
                }
                out.push_str(&format!(
                    "<path d=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1\"/>\n",
                    d, color
                ));
            }
        }
    }
    out
}

fn heatmap_svg(
    lane: &HeatmapLane,
    top: f64,
    height: f64,
    plot_x0: f64,
    plot_w: f64,
    s: &FigureStyle,
) -> String {
    let mut out = String::new();
    if lane.width > 0 && lane.height > 0 && lane.rgba.len() == lane.width * lane.height * 4 {
        let png_b64 = rgba_to_png_base64(&lane.rgba, lane.width, lane.height);
        out.push_str(&format!(
            "<image x=\"{:.1}\" y=\"{:.1}\" width=\"{:.1}\" height=\"{:.1}\" \
             preserveAspectRatio=\"none\" xlink:href=\"data:image/png;base64,{}\" \
             xmlns:xlink=\"http://www.w3.org/1999/xlink\"/>\n",
            plot_x0, top, plot_w, height, png_b64
        ));
    }
    // Lane name + top/bottom labels.
    out.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{:.1}\" font-size=\"{:.1}\" text-anchor=\"end\" fill=\"{}\">{}</text>\n",
        plot_x0 - 6.0,
        top + height / 2.0 + s.font_size * 0.35,
        s.font_size * 0.85,
        s.stroke,
        xml_escape(&lane.name),
    ));
    out.push_str(&y_label(plot_x0, top + s.font_size, &lane.top_label, s));
    out.push_str(&y_label(plot_x0, top + height, &lane.bottom_label, s));
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

/// PNG-encodes an RGBA8 raster to bytes — for writing a heatmap-lane sidecar
/// (the TikZ backend can't inline a raster). Public counterpart of
/// [`spectrogram_png`] for the heatmap lanes.
pub fn rgba_to_png_bytes(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
    rgba_to_png(rgba, width, height)
}

/// PNG-encodes an RGBA8 raster to bytes.
fn rgba_to_png(rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
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
    png_bytes
}

/// PNG-encodes an RGBA8 raster and base64s it for an SVG `data:` URI.
fn rgba_to_png_base64(rgba: &[u8], width: usize, height: usize) -> String {
    base64::engine::general_purpose::STANDARD.encode(rgba_to_png(rgba, width, height))
}

/// The figure's spectrogram lane as a standalone PNG (the sidecar the TikZ
/// backend `\includegraphics`'s), or `None` if the figure has no spectrogram.
pub fn spectrogram_png(spec: &FigureSpec) -> Option<Vec<u8>> {
    let sg = spec.spectrogram.as_ref()?;
    if sg.width == 0 || sg.height == 0 || sg.rgba.len() != sg.width * sg.height * 4 {
        return None;
    }
    Some(rgba_to_png(&sg.rgba, sg.width, sg.height))
}

/// The top (max) y-axis label for a measure lane: the value plus its unit
/// (e.g. `"500 Hz"`), so the unit rides here instead of a separate mid-height
/// label that would collide with the lane name.
fn measure_top_label(hi: f64, unit: &str) -> String {
    if unit.is_empty() {
        fmt_num(hi)
    } else {
        format!("{} {unit}", fmt_num(hi))
    }
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

/// Options for assembling a [`FigureSpec`] from a bundle's audio + tiers.
/// [`Default`] mirrors the GUI's spectrogram defaults (25 ms window / 5 ms hop
/// / Viridis / 70 dB range) and a full waveform + spectrogram figure at the
/// default figure width.
pub struct FigureExportOptions {
    /// Optional figure title.
    pub title: Option<String>,
    /// Draw the waveform band.
    pub include_waveform: bool,
    /// Draw the spectrogram raster.
    pub include_spectrogram: bool,
    /// Draw the f0 (pitch) measure lane.
    pub include_f0: bool,
    /// Draw the formants measure lane.
    pub include_formants: bool,
    /// Draw the intensity measure lane.
    pub include_intensity: bool,
    /// Draw the MFCC heatmap lane.
    pub include_mfcc: bool,
    /// If set, draw an embedding heatmap lane from this `continuous_vector`
    /// tier's `(frames × dims)` matrix. Resolved by the exporter (which has the
    /// project); `build_spec` alone can't read tier data.
    pub embedding_tier_id: Option<i64>,
    /// Overall figure width in px.
    pub width: f64,
    /// Base font-size override in px; `None` keeps the default.
    pub font_size: Option<f64>,
    /// Waveform-lane height override in px; `None` keeps the default.
    pub waveform_height: Option<f64>,
    /// Spectrogram-lane height override in px; `None` keeps the default.
    pub spectrogram_height: Option<f64>,
    /// Measure-lane height override in px; `None` keeps the default.
    pub measure_height: Option<f64>,
    /// Heatmap-lane height override in px; `None` keeps the default.
    pub heatmap_height: Option<f64>,
    /// Tier-row height override in px; `None` keeps the default.
    pub tier_height: Option<f64>,
    /// Background colour override (CSS); `None` keeps the default.
    pub background: Option<String>,
    /// Stroke colour override (CSS); `None` keeps the default.
    pub stroke: Option<String>,
    /// Waveform-fill colour override (CSS); `None` keeps the default.
    pub waveform_fill: Option<String>,
    /// STFT window length in milliseconds.
    pub window_ms: f32,
    /// STFT hop length in milliseconds.
    pub hop_ms: f32,
    /// Spectrogram dynamic-range floor in dB.
    pub dynamic_range_db: f32,
    /// Spectrogram colormap.
    pub colormap: crate::dsp::ColormapKind,
    /// Heatmap (MFCC / embedding) colormap; `None` reuses [`Self::colormap`].
    pub heatmap_colormap: Option<crate::dsp::ColormapKind>,
    /// Pitch floor in Hz for the f0 tracker; `None` uses the config default.
    pub f0_min_hz: Option<f32>,
    /// Pitch ceiling in Hz for the f0 tracker; `None` uses the config default.
    pub f0_max_hz: Option<f32>,
}

impl FigureExportOptions {
    /// The colormap for heatmap lanes — the explicit override, else the
    /// spectrogram colormap.
    pub(crate) fn heatmap_cmap(&self) -> crate::dsp::ColormapKind {
        self.heatmap_colormap.unwrap_or(self.colormap)
    }
}

impl Default for FigureExportOptions {
    fn default() -> Self {
        Self {
            title: None,
            include_waveform: true,
            include_spectrogram: true,
            // Measure lanes are opt-in: a plain figure is waveform + spectrogram
            // + tiers unless a lane is explicitly requested.
            include_f0: false,
            include_formants: false,
            include_intensity: false,
            include_mfcc: false,
            embedding_tier_id: None,
            width: FigureStyle::default().width,
            font_size: None,
            waveform_height: None,
            spectrogram_height: None,
            measure_height: None,
            heatmap_height: None,
            tier_height: None,
            background: None,
            stroke: None,
            waveform_fill: None,
            window_ms: 25.0,
            hop_ms: 5.0,
            dynamic_range_db: 70.0,
            colormap: crate::dsp::ColormapKind::Viridis,
            heatmap_colormap: None,
            f0_min_hz: None,
            f0_max_hz: None,
        }
    }
}

/// Max spectrogram raster width (time cells) before time-downsampling, so the
/// embedded PNG stays bounded on long files (mirrors the app's cap).
const MAX_FIGURE_SPECTROGRAM_WIDTH: usize = 4096;

/// Assembles a [`FigureSpec`] from a bundle's mono audio + already-converted
/// drawable tiers, computing the waveform envelope and the spectrogram raster
/// (STFT → power → dB-normalise → colormap bake) headlessly — so Python and
/// the GUI share one figure-assembly path.
///
/// A signal too short for the requested window simply yields no spectrogram
/// (and empty audio yields no waveform) rather than an error.
pub fn build_spec(
    samples: &[f32],
    sample_rate: u32,
    tiers: Vec<FigureTier>,
    opts: &FigureExportOptions,
) -> FigureSpec {
    let duration = if sample_rate > 0 {
        samples.len() as f64 / sample_rate as f64
    } else {
        0.0
    };
    let d = FigureStyle::default();
    let style = FigureStyle {
        width: opts.width,
        waveform_height: opts.waveform_height.unwrap_or(d.waveform_height),
        spectrogram_height: opts.spectrogram_height.unwrap_or(d.spectrogram_height),
        measure_height: opts.measure_height.unwrap_or(d.measure_height),
        heatmap_height: opts.heatmap_height.unwrap_or(d.heatmap_height),
        tier_height: opts.tier_height.unwrap_or(d.tier_height),
        margin_left: d.margin_left,
        margin_right: d.margin_right,
        margin_top: d.margin_top,
        axis_height: d.axis_height,
        font_size: opts.font_size.unwrap_or(d.font_size),
        stroke: opts.stroke.clone().unwrap_or(d.stroke),
        waveform_fill: opts.waveform_fill.clone().unwrap_or(d.waveform_fill),
        background: opts.background.clone().unwrap_or(d.background),
    };

    let waveform = if opts.include_waveform && !samples.is_empty() {
        let cols = (opts.width - style.margin_left - style.margin_right).max(1.0);
        let minmax = minmax_envelope(samples, cols as usize);
        // Scale the band to the actual peak so quiet signals still fill the lane.
        let peak = minmax
            .iter()
            .flat_map(|&(mn, mx)| [mn.abs(), mx.abs()])
            .fold(0.0_f32, f32::max)
            .max(1e-6);
        Some(WaveformLane {
            minmax,
            amplitude_range: (-peak, peak),
        })
    } else {
        None
    };

    let spectrogram = if opts.include_spectrogram {
        build_spectrogram_lane(samples, sample_rate, opts)
    } else {
        None
    };

    let measures = build_measure_lanes(samples, sample_rate, opts);
    let heatmaps = build_heatmap_lanes(samples, sample_rate, opts);

    FigureSpec {
        title: opts.title.clone(),
        time_range: (0.0, duration.max(f64::MIN_POSITIVE)),
        waveform,
        spectrogram,
        measures,
        heatmaps,
        tiers,
        style,
    }
}

/// Computes the requested heatmap lanes (MFCC …) from audio. Each matrix is
/// per-row min-max normalised (so lower coefficients aren't washed out by the
/// energy term) and baked with the figure's colormap.
fn build_heatmap_lanes(
    samples: &[f32],
    sample_rate: u32,
    opts: &FigureExportOptions,
) -> Vec<HeatmapLane> {
    let mut lanes = Vec::new();
    if sample_rate == 0 || samples.is_empty() {
        return lanes;
    }
    if opts.include_mfcc {
        let arr = crate::dsp::mfcc(
            samples,
            sample_rate,
            0.025,
            0.010,
            40,
            13,
            0.0,
            sample_rate as f32 / 2.0,
            crate::dsp::MfccMethod::default(),
        );
        let (n_frames, n_mfcc) = (arr.nrows(), arr.ncols());
        if let Some(lane) = matrix_heatmap(
            "MFCC",
            |f, c| arr[[f, c]],
            n_frames,
            n_mfcc,
            opts.heatmap_cmap(),
            &format!("c{}", n_mfcc.saturating_sub(1)),
            "c0",
        ) {
            lanes.push(lane);
        }
    }
    lanes
}

/// Bakes a `(n_frames × n_features)` matrix (accessed as `get(frame, feature)`)
/// into a [`HeatmapLane`]: **per-feature min-max normalised** so no single row
/// (e.g. the MFCC energy term) dominates the colour scale, laid out
/// feature-major (row 0 = top = highest feature index) and colormap-baked.
/// Returns `None` for an empty matrix.
///
/// Shared by the MFCC lane and the embedding-tier lane.
pub fn matrix_heatmap(
    name: &str,
    get: impl Fn(usize, usize) -> f32,
    n_frames: usize,
    n_features: usize,
    colormap: crate::dsp::ColormapKind,
    top_label: &str,
    bottom_label: &str,
) -> Option<HeatmapLane> {
    if n_frames == 0 || n_features == 0 {
        return None;
    }
    let mut norm = vec![0.0_f32; n_features * n_frames];
    for feat in 0..n_features {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for f in 0..n_frames {
            let v = get(f, feat);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let span = (hi - lo).max(1e-9);
        for f in 0..n_frames {
            norm[feat * n_frames + f] = (get(f, feat) - lo) / span;
        }
    }
    let rgba = crate::dsp::colormap_bake(&norm, n_frames, n_features, colormap);
    Some(HeatmapLane {
        name: name.to_string(),
        rgba,
        width: n_frames,
        height: n_features,
        top_label: top_label.to_string(),
        bottom_label: bottom_label.to_string(),
        raster_ref: None,
    })
}

/// Computes the requested measure lanes (f0 / formants / intensity) from audio
/// via the engine's DSP, each with a data-driven y-axis. Lanes that come back
/// empty (e.g. a fully unvoiced clip) are dropped.
fn build_measure_lanes(
    samples: &[f32],
    sample_rate: u32,
    opts: &FigureExportOptions,
) -> Vec<MeasureLane> {
    let mut lanes = Vec::new();
    if sample_rate == 0 || samples.is_empty() {
        return lanes;
    }

    if opts.include_f0 {
        let audio = crate::audio::Audio::from_samples(samples.to_vec(), sample_rate, 1);
        // Boersma (octave cost + Viterbi path-finding) rather than raw
        // autocorrelation — the latter's octave-doubling errors inflated the
        // data-driven f0 axis.
        let mut config = crate::pitch::PitchConfig::default();
        if let Some(lo) = opts.f0_min_hz {
            config.min_freq_hz = lo;
        }
        if let Some(hi) = opts.f0_max_hz {
            config.max_freq_hz = hi;
        }
        let frames = crate::pitch::pitch(&audio, &config, crate::pitch::PitchMethod::Boersma);
        // Voiced runs → segments (break the contour across unvoiced frames).
        let mut segments: Vec<Vec<(f64, f64)>> = Vec::new();
        let mut cur: Vec<(f64, f64)> = Vec::new();
        for fr in &frames {
            let v = fr.frequency_hz.value() as f64;
            if fr.voicing >= 0.5 && v > 0.0 {
                cur.push((fr.time_seconds, v));
            } else if !cur.is_empty() {
                segments.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            segments.push(cur);
        }
        if segments.iter().any(|s| !s.is_empty()) {
            // Robust axis: the 5th–95th percentile of voiced f0 (padded), so a
            // stray octave-error frame can't blow the range up. Falls back to a
            // speech-typical window when there are too few points.
            let mut voiced: Vec<f64> = segments.iter().flatten().map(|&(_, v)| v).collect();
            voiced.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let (lo, hi) = if voiced.len() >= 8 {
                let pct = |p: f64| voiced[((voiced.len() - 1) as f64 * p).round() as usize];
                let lo = (pct(0.05) * 0.9).floor();
                let hi = (pct(0.95) * 1.1).ceil();
                if hi > lo { (lo, hi) } else { (75.0, 500.0) }
            } else {
                (75.0, 500.0)
            };
            lanes.push(MeasureLane {
                name: "f0".to_string(),
                unit: "Hz".to_string(),
                y_range: (lo, hi),
                series: vec![MeasureSeries {
                    segments,
                    dots: false,
                    color: Some("#2b6cb0".to_string()),
                }],
            });
        }
    }

    if opts.include_formants {
        let frames =
            crate::dsp::formants(samples, sample_rate, &crate::dsp::FormantsConfig::default());
        let n_formants = frames
            .iter()
            .map(|f| f.frequencies.len())
            .max()
            .unwrap_or(0);
        let mut series = Vec::new();
        let mut hi = 0.0_f64;
        for i in 0..n_formants {
            let pts: Vec<(f64, f64)> = frames
                .iter()
                .filter_map(|f| {
                    f.frequencies
                        .get(i)
                        .map(|h| (f.time_seconds, h.value() as f64))
                })
                .filter(|&(_, v)| v > 0.0)
                .collect();
            for &(_, v) in &pts {
                hi = hi.max(v);
            }
            if !pts.is_empty() {
                series.push(MeasureSeries {
                    segments: vec![pts],
                    dots: true,
                    color: Some("#c0392b".to_string()),
                });
            }
        }
        if !series.is_empty() {
            lanes.push(MeasureLane {
                name: "formants".to_string(),
                unit: "Hz".to_string(),
                y_range: (0.0, (hi * 1.05).ceil().max(1.0)),
                series,
            });
        }
    }

    if opts.include_intensity {
        let frames = crate::dsp::intensity(samples, sample_rate, 0.03, 0.01);
        let pts: Vec<(f64, f64)> = frames
            .iter()
            .map(|f| (f.time_seconds, f.db_fs.value() as f64))
            .filter(|&(_, v)| v.is_finite())
            .collect();
        if pts.len() >= 2 {
            let lo = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
            let hi = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
            let (lo, hi) = if hi > lo {
                (lo, hi)
            } else {
                (lo - 1.0, lo + 1.0)
            };
            lanes.push(MeasureLane {
                name: "intensity".to_string(),
                unit: "dB".to_string(),
                y_range: (lo.floor(), hi.ceil()),
                series: vec![MeasureSeries {
                    segments: vec![pts],
                    dots: false,
                    color: Some("#2f855a".to_string()),
                }],
            });
        }
    }

    lanes
}

/// Computes a [`SpectrogramLane`] from audio, or `None` if the signal is too
/// short / the rate is unknown.
fn build_spectrogram_lane(
    samples: &[f32],
    sample_rate: u32,
    opts: &FigureExportOptions,
) -> Option<SpectrogramLane> {
    if sample_rate == 0 {
        return None;
    }
    let sr = sample_rate as f32;
    let window_samples = ((opts.window_ms / 1000.0) * sr).round() as usize;
    let hop_samples = ((opts.hop_ms / 1000.0) * sr).round() as usize;
    if window_samples < 4 || hop_samples == 0 || samples.len() < window_samples {
        return None;
    }
    let window = crate::dsp::hann(window_samples);
    let (stft_out, shape) = crate::dsp::stft(samples, &window, hop_samples);
    if shape.n_frames == 0 || shape.n_freq_bins == 0 {
        return None;
    }
    let power = crate::dsp::power_spectrogram(&stft_out, shape);
    let normalized = crate::dsp::power_to_db_normalized(&power, opts.dynamic_range_db);

    // Time-downsample (average pooling) if wider than the cap.
    let (width, display) = if shape.n_frames > MAX_FIGURE_SPECTROGRAM_WIDTH {
        let stride = shape.n_frames.div_ceil(MAX_FIGURE_SPECTROGRAM_WIDTH);
        let new_width = shape.n_frames.div_ceil(stride);
        let mut out = vec![0.0_f32; shape.n_freq_bins * new_width];
        for b in 0..shape.n_freq_bins {
            for x in 0..new_width {
                let start = x * stride;
                let end = (start + stride).min(shape.n_frames);
                let mut acc = 0.0_f32;
                for f in start..end {
                    acc += normalized[b * shape.n_frames + f];
                }
                out[b * new_width + x] = acc / (end - start) as f32;
            }
        }
        (new_width, out)
    } else {
        (shape.n_frames, normalized)
    };

    let rgba = crate::dsp::colormap_bake(&display, width, shape.n_freq_bins, opts.colormap);
    Some(SpectrogramLane {
        rgba,
        width,
        height: shape.n_freq_bins,
        max_freq_hz: sr / 2.0,
    })
}

/// A min/max amplitude envelope of `samples` bucketed into `n_cols` columns,
/// left → right. Each column is the `(min, max)` over its slice of samples.
fn minmax_envelope(samples: &[f32], n_cols: usize) -> Vec<(f32, f32)> {
    if samples.is_empty() || n_cols == 0 {
        return Vec::new();
    }
    let n = samples.len();
    let mut out = Vec::with_capacity(n_cols);
    for c in 0..n_cols {
        let start = c * n / n_cols;
        let end = (((c + 1) * n / n_cols).max(start + 1)).min(n);
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &s in &samples[start..end] {
            mn = mn.min(s);
            mx = mx.max(s);
        }
        out.push((mn, mx));
    }
    out
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
            measures: vec![MeasureLane {
                name: "f0".to_string(),
                unit: "Hz".to_string(),
                y_range: (75.0, 300.0),
                series: vec![MeasureSeries {
                    segments: vec![vec![(0.1, 120.0), (0.2, 140.0), (0.3, 130.0)]],
                    dots: false,
                    color: Some("#2b6cb0".to_string()),
                }],
            }],
            heatmaps: vec![HeatmapLane {
                name: "MFCC".to_string(),
                rgba: vec![64u8; 5 * 3 * 4], // 5 frames × 3 coeffs
                width: 5,
                height: 3,
                top_label: "c2".to_string(),
                bottom_label: "c0".to_string(),
                raster_ref: None,
            }],
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
            measures: vec![],
            heatmaps: vec![],
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

    #[cfg(feature = "figure-pdf")]
    #[test]
    fn to_pdf_produces_a_valid_pdf() {
        let pdf = to_pdf(&sample_spec()).expect("pdf conversion");
        assert!(pdf.starts_with(b"%PDF-"), "output should be a PDF");
        assert!(pdf.len() > 1000, "PDF looks too small to be real");
    }

    #[cfg(feature = "figure-pdf")]
    #[test]
    fn to_rgba_rasterises_to_the_svg_dimensions() {
        let spec = sample_spec();
        let (w, h, rgba) = to_rgba(&spec).expect("raster");
        assert!(w > 0 && h > 0);
        assert_eq!(rgba.len(), (w * h * 4) as usize);
        // The figure has an opaque white background → every pixel opaque.
        assert!(rgba.chunks_exact(4).all(|px| px[3] == 255));
        // Width matches the style width (the SVG viewBox width).
        assert_eq!(w, spec.style.width as u32);
    }

    #[test]
    fn to_tikz_is_a_compilable_standalone_with_raster_and_ipa() {
        let tikz = to_tikz(&sample_spec(), Some("fig-spectrogram.png"));
        assert!(tikz.contains("\\documentclass[border=5pt]{standalone}"));
        assert!(tikz.contains("\\usepackage{fontspec}"));
        assert!(tikz.contains("\\begin{tikzpicture}"));
        assert!(tikz.trim_end().ends_with("\\end{document}"));
        // Spectrogram is the sidecar raster, not inlined.
        assert!(tikz.contains("\\includegraphics"));
        assert!(tikz.contains("fig-spectrogram.png"));
        // IPA + axis text survive as real LaTeX text.
        assert!(tikz.contains("{ɹ}"));
        assert!(tikz.contains("Time (s)"));
        assert!(tikz.contains("{\\large praat}")); // title
    }

    #[test]
    fn to_tikz_without_raster_ref_omits_the_image() {
        let tikz = to_tikz(&sample_spec(), None);
        assert!(!tikz.contains("\\includegraphics"));
        // The rest still renders.
        assert!(tikz.contains("\\begin{tikzpicture}"));
        assert!(tikz.contains("{ɹ}"));
    }

    #[test]
    fn tex_escape_escapes_latex_specials_not_ipa() {
        assert_eq!(tex_escape("a & b_c #1 %x"), "a \\& b\\_c \\#1 \\%x");
        // IPA passes through untouched.
        assert_eq!(tex_escape("ɑː"), "ɑː");
    }

    #[test]
    fn hex_rgb_parses_full_hex_and_falls_back() {
        assert_eq!(hex_rgb("#333333"), "333333");
        assert_eq!(hex_rgb("#ffffff"), "FFFFFF");
        assert_eq!(hex_rgb("red"), "000000"); // named → fallback
        assert_eq!(hex_rgb("#abc"), "000000"); // short hex → fallback
    }

    #[test]
    fn spectrogram_png_round_trips_a_raster() {
        let png = spectrogram_png(&sample_spec()).expect("has a spectrogram");
        // PNG magic number.
        assert_eq!(
            &png[..8],
            &[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n']
        );
    }

    #[test]
    fn build_spec_from_audio_populates_lanes() {
        let sr = 16_000u32;
        // 0.25 s, 440 Hz sine.
        let n = sr as usize / 4;
        let samples: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * 440.0 * i as f32 / sr as f32).sin() * 0.6)
            .collect();
        let tiers = vec![FigureTier {
            name: "w".to_string(),
            content: FigureTierContent::Intervals(vec![FigureInterval {
                start: 0.0,
                end: 0.25,
                label: "aː".to_string(),
            }]),
        }];
        let spec = build_spec(&samples, sr, tiers, &FigureExportOptions::default());
        // Time range ≈ duration.
        assert!((spec.time_range.1 - 0.25).abs() < 1e-3);
        // Both signal lanes materialised.
        let wave = spec.waveform.as_ref().expect("waveform");
        assert!(!wave.minmax.is_empty());
        assert!(wave.amplitude_range.1 > 0.0);
        let sg = spec.spectrogram.as_ref().expect("spectrogram");
        assert_eq!(sg.rgba.len(), sg.width * sg.height * 4);
        assert!((sg.max_freq_hz - 8000.0).abs() < 1.0); // Nyquist
        // And it serialises.
        assert!(to_svg(&spec).contains(">aː</text>"));
    }

    #[test]
    fn build_spec_too_short_for_window_skips_spectrogram() {
        // 3 samples can't fill a 25 ms window; no spectrogram, but no panic.
        let spec = build_spec(
            &[0.1, -0.1, 0.2],
            16_000,
            vec![],
            &FigureExportOptions::default(),
        );
        assert!(spec.spectrogram.is_none());
        assert!(spec.waveform.is_some());
    }

    #[test]
    fn build_spec_respects_include_flags() {
        let samples = vec![0.3_f32; 16_000];
        let opts = FigureExportOptions {
            include_spectrogram: false,
            ..FigureExportOptions::default()
        };
        let spec = build_spec(&samples, 16_000, vec![], &opts);
        assert!(spec.spectrogram.is_none());
        assert!(spec.waveform.is_some());
        // Measure lanes are opt-in — none by default.
        assert!(spec.measures.is_empty());
    }

    #[test]
    fn build_spec_computes_measure_lanes_and_renders_them() {
        // 0.3 s, 150 Hz sine → a voiced f0 lane + an intensity lane.
        let sr = 16_000u32;
        let n = (sr as f64 * 0.3) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * 150.0 * i as f32 / sr as f32).sin() * 0.6)
            .collect();
        let opts = FigureExportOptions {
            include_f0: true,
            include_formants: true,
            include_intensity: true,
            ..FigureExportOptions::default()
        };
        let spec = build_spec(&samples, sr, vec![], &opts);
        let names: Vec<&str> = spec.measures.iter().map(|m| m.name.as_str()).collect();
        // Intensity is deterministic from any signal.
        assert!(names.contains(&"intensity"), "got {names:?}");
        // A clean 150 Hz sine should track as voiced f0.
        assert!(names.contains(&"f0"), "got {names:?}");
        // The robust (percentile) axis brackets 150 Hz rather than blowing up on
        // an octave-error frame.
        let f0 = spec.measures.iter().find(|m| m.name == "f0").unwrap();
        assert!(
            f0.y_range.0 <= 150.0 && f0.y_range.1 >= 150.0 && f0.y_range.1 < 400.0,
            "f0 axis {:?} should bracket 150 Hz tightly",
            f0.y_range
        );
        // The lanes render (names + a polyline) in both backends.
        let svg = to_svg(&spec);
        assert!(svg.contains(">intensity</text>"));
        assert!(svg.contains(">f0</text>"));
        let tikz = to_tikz(&spec, None);
        assert!(tikz.contains("{intensity}"));
    }

    #[test]
    fn build_spec_computes_mfcc_heatmap_and_renders_it() {
        let sr = 16_000u32;
        let n = (sr as f64 * 0.3) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * 200.0 * i as f32 / sr as f32).sin() * 0.5)
            .collect();
        let opts = FigureExportOptions {
            include_mfcc: true,
            ..FigureExportOptions::default()
        };
        let spec = build_spec(&samples, sr, vec![], &opts);
        assert_eq!(spec.heatmaps.len(), 1);
        let hm = &spec.heatmaps[0];
        assert_eq!(hm.name, "MFCC");
        assert_eq!(hm.height, 13); // 13 coefficients
        assert_eq!(hm.rgba.len(), hm.width * hm.height * 4);
        // Renders as an embedded raster in SVG, with its lane label.
        let svg = to_svg(&spec);
        assert!(svg.contains(">MFCC</text>"));
    }

    #[test]
    fn font_size_style_knob_applies() {
        let opts = FigureExportOptions {
            font_size: Some(20.0),
            ..FigureExportOptions::default()
        };
        let spec = build_spec(&[0.1_f32; 16_000], 16_000, vec![], &opts);
        assert_eq!(spec.style.font_size, 20.0);
    }

    #[test]
    fn minmax_envelope_captures_extremes() {
        let samples = vec![0.0, 1.0, -1.0, 0.5, -0.5, 0.2];
        let env = minmax_envelope(&samples, 2);
        assert_eq!(env.len(), 2);
        // First bucket [0,1,-1] → (-1, 1); second [0.5,-0.5,0.2] → (-0.5, 0.5).
        assert_eq!(env[0], (-1.0, 1.0));
        assert_eq!(env[1], (-0.5, 0.5));
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
