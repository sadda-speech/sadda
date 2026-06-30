//! Pure timeline navigation state: cursor + view window + selection, with a
//! `move-to` (absolute) / `move-by` (relative) API over each.
//!
//! This lives in the engine (not the GUI) so the same navigation primitives
//! back the desktop keybindings, the Python bindings, and unit tests — one
//! tested source of truth. It is pure data: no rendering, no I/O, no egui.

/// Minimum view-range size in seconds; prevents zooming to a window so small
/// that floats start losing precision.
const MIN_VIEW_RANGE_SECONDS: f64 = 0.005;
/// Drags shorter than this (in seconds) don't "stick" as a selection — they
/// read as a plain click (which clears the selection + sets the cursor).
const MIN_SELECTION_SECONDS: f64 = 0.002;

/// Shared timeline navigation state: the cursor (playhead), the visible view
/// window, an optional span selection, and the bundle duration everything
/// clamps against.
///
/// Times are in seconds. The view is always `[view_start, view_end)` with
/// `view_end > view_start`; the cursor and selection are clamped to
/// `[0, duration]`. Mutate it through the methods below rather than the fields
/// directly so clamping invariants hold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timeline {
    /// Left edge of the visible time range, in seconds.
    pub view_start: f64,
    /// Right edge of the visible time range, in seconds (exclusive).
    pub view_end: f64,
    /// Cursor position in seconds. Clamped to `[0, duration]`.
    pub cursor: f64,
    /// Bundle duration in seconds. Acts as the upper bound for `view_end` and
    /// `cursor` clamping.
    pub duration: f64,
    /// Active time-span selection `(lo, hi)` in seconds. `None` when nothing is
    /// selected.
    pub selection: Option<(f64, f64)>,
    /// Transient drag anchor while a selection is being dragged out.
    selection_anchor: Option<f64>,
}

impl Default for Timeline {
    fn default() -> Self {
        Self {
            view_start: 0.0,
            view_end: 0.0,
            cursor: 0.0,
            duration: 0.0,
            selection: None,
            selection_anchor: None,
        }
    }
}

impl Timeline {
    /// Builds a timeline for a bundle of `duration_seconds`, with the view
    /// spanning the whole recording and the cursor at the start.
    pub fn new(duration_seconds: f64) -> Self {
        let mut t = Self::default();
        t.reset_for_bundle(duration_seconds);
        t
    }

    /// Reinitialises the timeline for a freshly-loaded bundle: view spans
    /// `[0, duration]`, cursor at 0, no selection.
    pub fn reset_for_bundle(&mut self, duration_seconds: f64) {
        let d = duration_seconds.max(0.0);
        self.duration = d;
        self.view_start = 0.0;
        self.view_end = d.max(MIN_VIEW_RANGE_SECONDS);
        self.cursor = 0.0;
        self.selection = None;
        self.selection_anchor = None;
    }

    // ----- cursor -----

    /// Moves the cursor **to** `time_seconds` (absolute), clamped to
    /// `[0, duration]`.
    pub fn set_cursor(&mut self, time_seconds: f64) {
        self.cursor = time_seconds.clamp(0.0, self.duration);
    }

    /// Moves the cursor **by** `delta_seconds` (negative = left), clamped to
    /// `[0, duration]`.
    pub fn move_cursor_by(&mut self, delta_seconds: f64) {
        self.set_cursor(self.cursor + delta_seconds);
    }

    // ----- selection -----

    /// The current selection, seeding a zero-width one at the cursor when none
    /// exists yet — so the first edge-move creates a selection anchored where
    /// the user was.
    fn selection_or_seed(&self) -> (f64, f64) {
        self.selection.unwrap_or((self.cursor, self.cursor))
    }

    /// Sets the selection's **start** edge to `t` (absolute seconds), seeding a
    /// selection at the cursor when none exists and clamping so `start ≤ end`.
    pub fn set_selection_start(&mut self, t: f64) {
        let (_, hi) = self.selection_or_seed();
        let lo = t.clamp(0.0, hi);
        self.selection = Some((lo, hi));
    }

    /// Moves the selection's **start** edge by `delta_seconds`.
    pub fn move_selection_start_by(&mut self, delta_seconds: f64) {
        let (lo, _) = self.selection_or_seed();
        self.set_selection_start(lo + delta_seconds);
    }

    /// Sets the selection's **end** edge to `t` (absolute seconds), seeding a
    /// selection at the cursor when none exists and clamping so `end ≥ start`.
    pub fn set_selection_end(&mut self, t: f64) {
        let (lo, _) = self.selection_or_seed();
        let hi = t.clamp(lo, self.duration);
        self.selection = Some((lo, hi));
    }

    /// Moves the selection's **end** edge by `delta_seconds`.
    pub fn move_selection_end_by(&mut self, delta_seconds: f64) {
        let (_, hi) = self.selection_or_seed();
        self.set_selection_end(hi + delta_seconds);
    }

    /// Sets the selection to exactly `[start, end]` seconds in one call (sorted,
    /// each edge clamped to the recording). The selection analogue of
    /// [`set_view_range`](Self::set_view_range); unlike the edge setters it
    /// needs no existing selection to seed from.
    pub fn set_selection_range(&mut self, start: f64, end: f64) {
        let lo = start.min(end).clamp(0.0, self.duration);
        let hi = start.max(end).clamp(0.0, self.duration);
        self.selection_anchor = None;
        self.selection = Some((lo, hi));
    }

    /// Begins a drag-selection anchored at `t` (seconds).
    pub fn begin_selection(&mut self, t: f64) {
        let t = t.clamp(0.0, self.duration);
        self.selection_anchor = Some(t);
        self.selection = Some((t, t));
    }

    /// Sets a zero-width "selection point" at `t` (clamped to the bundle). A
    /// plain click uses this so the time can be committed as a point; a drag
    /// still produces a span via [`begin_selection`](Self::begin_selection).
    pub fn set_point_selection(&mut self, t: f64) {
        let t = t.clamp(0.0, self.duration);
        self.selection_anchor = None;
        self.selection = Some((t, t));
    }

    /// Extends the in-progress drag-selection to `t` (sorted into `(lo, hi)`).
    pub fn update_selection(&mut self, t: f64) {
        if let Some(a) = self.selection_anchor {
            let t = t.clamp(0.0, self.duration);
            self.selection = Some(if a <= t { (a, t) } else { (t, a) });
        }
    }

    /// Finalises a drag-selection; discards spans too short to be intentional.
    pub fn end_selection(&mut self) {
        self.selection_anchor = None;
        if let Some((lo, hi)) = self.selection {
            if (hi - lo) < MIN_SELECTION_SECONDS {
                self.selection = None;
            }
        }
    }

    /// Clears any selection (and drag anchor).
    pub fn clear_selection(&mut self) {
        self.selection = None;
        self.selection_anchor = None;
    }

    // ----- view: scroll & zoom -----

    /// Returns `view_end - view_start` (always > 0 for an initialised state).
    pub fn view_range(&self) -> f64 {
        (self.view_end - self.view_start).max(MIN_VIEW_RANGE_SECONDS)
    }

    /// Maps a pixel-x within `[0, plot_width_px)` to seconds in the current
    /// view range.
    pub fn pixel_to_time(&self, pixel_x: f64, plot_width_px: f64) -> f64 {
        if plot_width_px <= 0.0 {
            return self.view_start;
        }
        let t = (pixel_x / plot_width_px).clamp(0.0, 1.0);
        self.view_start + t * self.view_range()
    }

    /// Zooms the view around `time_seconds`, by `factor` (e.g. 0.833 to zoom
    /// in, 1.2 to zoom out). Clamps the resulting view to `[0, duration]` and
    /// prevents the range from shrinking below [`MIN_VIEW_RANGE_SECONDS`].
    pub fn zoom_at(&mut self, time_seconds: f64, factor: f64) {
        let old_range = self.view_range();
        let new_range = (old_range * factor).clamp(
            MIN_VIEW_RANGE_SECONDS,
            self.duration.max(MIN_VIEW_RANGE_SECONDS),
        );
        // Position of `time_seconds` within the old window (0..1):
        let t_norm = ((time_seconds - self.view_start) / old_range).clamp(0.0, 1.0);
        // Re-anchor: keep `time_seconds` at the same normalised position inside
        // the new (smaller / larger) window.
        let new_start = time_seconds - t_norm * new_range;
        self.set_view(new_start, new_start + new_range);
    }

    /// Pans the view **by** `delta_seconds`, clamped against the bundle's
    /// bounds. Range size is preserved.
    pub fn scroll_by(&mut self, delta_seconds: f64) {
        let range = self.view_range();
        let new_start = self.view_start + delta_seconds;
        self.set_view(new_start, new_start + range);
    }

    /// Pans the view so it starts **at** `t` (absolute seconds), preserving the
    /// range and clamping against the bundle's bounds.
    pub fn set_view_start(&mut self, t: f64) {
        let range = self.view_range();
        self.set_view(t, t + range);
    }

    /// Frames the view to exactly `[start, end]` (clamped to the bundle). Used
    /// by "fit whole recording" and "zoom to selection".
    pub fn set_view_range(&mut self, start: f64, end: f64) {
        self.set_view(start, end);
    }

    /// Ensures the cursor is inside `[view_start, view_end]`. If not, shifts the
    /// view (preserving range) to put the cursor a quarter of the way in from
    /// the left edge — the convention that gives the user upcoming audio to look
    /// at during playback.
    pub fn ensure_cursor_visible(&mut self) {
        let range = self.view_range();
        if self.cursor < self.view_start || self.cursor > self.view_end {
            let new_start = (self.cursor - range * 0.25).max(0.0);
            self.set_view(new_start, new_start + range);
        }
    }

    /// Pans the minimum amount needed to bring `t` into the visible window.
    /// Unlike [`ensure_cursor_visible`](Self::ensure_cursor_visible) this keeps
    /// `t` flush against whichever edge it left, so a smoothly-gliding cursor
    /// scrolls the view smoothly rather than jumping a quarter-window.
    pub fn scroll_into_view(&mut self, t: f64) {
        if t < self.view_start {
            self.scroll_by(t - self.view_start);
        } else if t > self.view_end {
            self.scroll_by(t - self.view_end);
        }
    }

    /// Internal: clamps a candidate `[start, end]` to the bundle.
    fn set_view(&mut self, mut start: f64, mut end: f64) {
        let duration = self.duration.max(MIN_VIEW_RANGE_SECONDS);
        let range = (end - start).max(MIN_VIEW_RANGE_SECONDS);
        if start < 0.0 {
            start = 0.0;
            end = (start + range).min(duration);
        }
        if end > duration {
            end = duration;
            start = (end - range).max(0.0);
        }
        self.view_start = start;
        self.view_end = end.max(start + MIN_VIEW_RANGE_SECONDS);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh(duration: f64) -> Timeline {
        Timeline::new(duration)
    }

    #[test]
    fn reset_for_bundle_spans_full_range() {
        let t = fresh(5.0);
        assert_eq!(t.view_start, 0.0);
        assert_eq!(t.view_end, 5.0);
        assert_eq!(t.cursor, 0.0);
        assert_eq!(t.duration, 5.0);
    }

    #[test]
    fn reset_clamps_negative_duration() {
        let mut t = Timeline::default();
        t.reset_for_bundle(-1.0);
        assert_eq!(t.duration, 0.0);
        assert!(t.view_end >= MIN_VIEW_RANGE_SECONDS);
    }

    #[test]
    fn selection_drag_sorts_and_sticks() {
        let mut t = fresh(5.0);
        // Drag right-to-left still yields a sorted (lo, hi).
        t.begin_selection(3.0);
        t.update_selection(1.0);
        assert_eq!(t.selection, Some((1.0, 3.0)));
        t.end_selection();
        assert_eq!(t.selection, Some((1.0, 3.0)));
    }

    #[test]
    fn tiny_selection_is_discarded_as_a_click() {
        let mut t = fresh(5.0);
        t.begin_selection(2.0);
        t.update_selection(2.0 + MIN_SELECTION_SECONDS / 2.0);
        t.end_selection();
        assert_eq!(t.selection, None);
    }

    #[test]
    fn clear_and_reset_drop_selection() {
        let mut t = fresh(5.0);
        t.begin_selection(1.0);
        t.update_selection(2.0);
        t.clear_selection();
        assert_eq!(t.selection, None);
        t.begin_selection(1.0);
        t.update_selection(2.0);
        t.reset_for_bundle(5.0);
        assert_eq!(t.selection, None);
    }

    #[test]
    fn selection_clamps_to_duration() {
        let mut t = fresh(2.0);
        t.begin_selection(-1.0);
        t.update_selection(9.0);
        assert_eq!(t.selection, Some((0.0, 2.0)));
    }

    #[test]
    fn set_cursor_clamps_to_duration() {
        let mut t = fresh(2.0);
        t.set_cursor(5.0);
        assert_eq!(t.cursor, 2.0);
        t.set_cursor(-1.0);
        assert_eq!(t.cursor, 0.0);
        t.set_cursor(1.3);
        assert_eq!(t.cursor, 1.3);
    }

    #[test]
    fn move_cursor_by_is_relative_and_clamps() {
        let mut t = fresh(10.0);
        t.set_cursor(4.0);
        t.move_cursor_by(2.5);
        assert!((t.cursor - 6.5).abs() < 1e-9);
        t.move_cursor_by(-100.0);
        assert_eq!(t.cursor, 0.0);
        t.move_cursor_by(100.0);
        assert_eq!(t.cursor, 10.0);
    }

    #[test]
    fn set_selection_start_seeds_at_cursor_and_clamps() {
        let mut t = fresh(10.0);
        t.set_cursor(6.0);
        // No selection yet: seeds (cursor, cursor), then moves the start.
        t.set_selection_start(2.0);
        assert_eq!(t.selection, Some((2.0, 6.0)));
        // Start cannot cross the end.
        t.set_selection_start(9.0);
        assert_eq!(t.selection, Some((6.0, 6.0)));
    }

    #[test]
    fn set_selection_end_seeds_at_cursor_and_clamps() {
        let mut t = fresh(10.0);
        t.set_cursor(3.0);
        t.set_selection_end(7.0);
        assert_eq!(t.selection, Some((3.0, 7.0)));
        // End cannot cross the start.
        t.set_selection_end(1.0);
        assert_eq!(t.selection, Some((3.0, 3.0)));
        // End clamps to duration.
        t.set_selection_end(99.0);
        assert_eq!(t.selection, Some((3.0, 10.0)));
    }

    #[test]
    fn move_selection_edges_by_is_relative() {
        let mut t = fresh(10.0);
        t.set_cursor(5.0);
        t.set_selection_end(8.0); // selection (5, 8)
        t.move_selection_start_by(-2.0); // (3, 8)
        assert_eq!(t.selection, Some((3.0, 8.0)));
        t.move_selection_end_by(1.0); // (3, 9)
        assert_eq!(t.selection, Some((3.0, 9.0)));
    }

    #[test]
    fn set_selection_range_sorts_and_clamps() {
        let mut t = fresh(10.0);
        // Needs no existing selection; sorts a reversed pair.
        t.set_selection_range(7.0, 3.0);
        assert_eq!(t.selection, Some((3.0, 7.0)));
        // Each edge clamps to the recording.
        t.set_selection_range(-5.0, 99.0);
        assert_eq!(t.selection, Some((0.0, 10.0)));
    }

    #[test]
    fn zoom_in_keeps_anchor_at_same_normalised_position() {
        let mut t = fresh(10.0);
        // Anchor at the cursor (middle of the view), zoom in 5x.
        t.zoom_at(5.0, 0.2);
        let normalised = (5.0 - t.view_start) / t.view_range();
        assert!((normalised - 0.5).abs() < 1e-9);
        assert!((t.view_range() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn zoom_clamps_to_minimum_range() {
        let mut t = fresh(10.0);
        t.zoom_at(5.0, 0.000001);
        assert!(t.view_range() >= MIN_VIEW_RANGE_SECONDS);
    }

    #[test]
    fn zoom_out_past_bundle_clamps_to_full_range() {
        let mut t = fresh(10.0);
        t.zoom_at(5.0, 100.0);
        assert!((t.view_start - 0.0).abs() < 1e-9);
        assert!((t.view_end - 10.0).abs() < 1e-9);
    }

    #[test]
    fn scroll_preserves_range_and_clamps_left() {
        let mut t = fresh(10.0);
        t.zoom_at(5.0, 0.2); // range 2.0
        t.scroll_by(-100.0);
        assert!((t.view_start - 0.0).abs() < 1e-9);
        assert!((t.view_range() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn scroll_preserves_range_and_clamps_right() {
        let mut t = fresh(10.0);
        t.zoom_at(5.0, 0.2); // range 2.0
        t.scroll_by(100.0);
        assert!((t.view_end - 10.0).abs() < 1e-9);
        assert!((t.view_range() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn set_view_start_moves_to_absolute_and_preserves_range() {
        let mut t = fresh(10.0);
        t.zoom_at(5.0, 0.2); // range 2.0
        t.set_view_start(3.0);
        assert!((t.view_start - 3.0).abs() < 1e-9);
        assert!((t.view_range() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn set_view_range_frames_a_span() {
        let mut t = fresh(10.0);
        t.set_view_range(2.0, 6.0);
        assert!((t.view_start - 2.0).abs() < 1e-9);
        assert!((t.view_end - 6.0).abs() < 1e-9);
    }

    #[test]
    fn ensure_cursor_visible_shifts_view() {
        let mut t = fresh(10.0);
        t.zoom_at(2.0, 0.2); // range 2.0, view about [1, 3]
        t.cursor = 7.0;
        t.ensure_cursor_visible();
        assert!(t.cursor >= t.view_start && t.cursor <= t.view_end);
        assert!((t.view_range() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn scroll_into_view_pans_minimally() {
        let mut t = fresh(10.0);
        t.set_view_range(4.0, 6.0); // range 2.0
        // A point just past the right edge scrolls only enough to reach it.
        t.scroll_into_view(6.5);
        assert!((t.view_end - 6.5).abs() < 1e-9);
        assert!((t.view_range() - 2.0).abs() < 1e-9);
        // A point already inside leaves the view untouched.
        let before = (t.view_start, t.view_end);
        t.scroll_into_view(5.5);
        assert_eq!((t.view_start, t.view_end), before);
    }

    #[test]
    fn pixel_to_time_maps_within_view() {
        let mut t = fresh(10.0);
        t.view_start = 2.0;
        t.view_end = 6.0;
        assert!((t.pixel_to_time(0.0, 100.0) - 2.0).abs() < 1e-9);
        assert!((t.pixel_to_time(100.0, 100.0) - 6.0).abs() < 1e-9);
        assert!((t.pixel_to_time(50.0, 100.0) - 4.0).abs() < 1e-9);
    }
}
