"""Tests for the `sadda.Timeline` navigation type — the move-to / move-by API
that backs the desktop app's keyboard navigation."""

from __future__ import annotations

import sadda


def test_new_spans_whole_recording() -> None:
    t = sadda.Timeline(5.0)
    assert t.duration == 5.0
    assert t.view_start == 0.0
    assert t.view_end == 5.0
    assert t.cursor == 0.0
    assert t.selection is None


def test_set_and_move_cursor_clamp() -> None:
    t = sadda.Timeline(10.0)
    t.set_cursor(4.0)
    assert t.cursor == 4.0
    t.move_cursor_by(2.5)
    assert abs(t.cursor - 6.5) < 1e-9
    t.move_cursor_by(-100.0)
    assert t.cursor == 0.0
    t.move_cursor_by(100.0)
    assert t.cursor == 10.0


def test_selection_edges_seed_at_cursor_and_clamp() -> None:
    t = sadda.Timeline(10.0)
    t.set_cursor(6.0)
    # No selection yet: the first edge-move seeds (cursor, cursor).
    t.set_selection_start(2.0)
    assert t.selection == (2.0, 6.0)
    # Start cannot cross the end.
    t.set_selection_start(9.0)
    assert t.selection == (6.0, 6.0)

    t2 = sadda.Timeline(10.0)
    t2.set_cursor(3.0)
    t2.set_selection_end(7.0)
    assert t2.selection == (3.0, 7.0)
    t2.move_selection_start_by(-1.0)
    assert t2.selection == (2.0, 7.0)
    t2.move_selection_end_by(100.0)  # clamps to duration
    assert t2.selection == (2.0, 10.0)
    t2.clear_selection()
    assert t2.selection is None


def test_view_scroll_and_zoom() -> None:
    t = sadda.Timeline(10.0)
    t.zoom_at(5.0, 0.2)  # range 2.0 around t=5
    assert abs(t.view_range - 2.0) < 1e-9
    # Pan to an absolute start, preserving the range.
    t.set_view_start(3.0)
    assert abs(t.view_start - 3.0) < 1e-9
    assert abs(t.view_range - 2.0) < 1e-9
    # Relative pan, clamped at the right edge.
    t.scroll_by(100.0)
    assert abs(t.view_end - 10.0) < 1e-9
    # Frame an explicit span (fit / zoom-to-selection).
    t.set_view_range(2.0, 6.0)
    assert abs(t.view_start - 2.0) < 1e-9
    assert abs(t.view_end - 6.0) < 1e-9


def test_repr_is_informative() -> None:
    t = sadda.Timeline(3.0)
    t.set_cursor(1.5)
    r = repr(t)
    assert "Timeline(" in r
    assert "cursor=1.500" in r
