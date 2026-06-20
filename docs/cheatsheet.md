# Keyboard cheatsheet

sadda's desktop app is built for **mouse-free scanning and annotation**: a
two-handed home-row scheme drives playback, cursor/selection movement, and
view scroll/zoom, so you can audition and mark up a recording without leaving
the keyboard.

!!! note "Bound by physical position, not layout"
    Every shortcut on this page is matched by the **physical key position**
    (US-QWERTY), not the character your layout produces. If you type Dvorak or
    AZERTY, the keys are where `a s d f` / `h j k l` *sit* on the board — your
    hands fall on the same shape regardless of layout. All shortcuts are
    suspended while a text field is focused, so they reach the editor instead.

## Playback — left hand

| Key | Action |
| --- | --- |
| ++d++ | Play the **selection** (or the whole view if nothing is selected). Press again to pause/resume. |
| ++s++ | Play **left of focus** — left of the selection if there is one, else left of the cursor. |
| ++f++ | Play **right of focus**. |
| ++shift+d++ / ++shift+s++ / ++shift+f++ | Loop the same span (0.5 s gap between repetitions). |
| ++a++ | Stop and return the cursor to where playback started. |
| ++space++ | Alias for ++d++ (play / pause). ++shift+space++ loops. |
| ++esc++ | Alias for ++a++ (stop). |

## Cursor & selection — right hand

The right-hand home row moves a point through the recording. The **modifier
picks which point**:

- **no modifier** → the **cursor** (playhead)
- ++shift++ → the **selection start** edge
- ++alt++ → the **selection end** edge

With no selection yet, the first ++shift++/++alt++ move seeds one at the cursor,
then moves that edge.

| Key | Movement |
| --- | --- |
| ++h++ | Jump to the **start of the recording** |
| ++j++ | Jump to the **start of the window** |
| ++k++ | **Glide left** (hold to keep moving; speeds up the longer you hold) |
| ++l++ | **Glide right** |
| ++semicolon++ | Jump to the **end of the window** |
| ++quote++ | Jump to the **end of the recording** |

The view follows the moved point off-screen, so a glide scrolls smoothly and a
jump brings its target into view.

## View: scroll & zoom — lower-right row

The lower-right row controls the **window** itself. Bare keys scroll; ++shift++
zooms.

| Key | Scroll (bare) | Zoom (++shift++) |
| --- | --- | --- |
| ++n++ | View → start of recording | **Fit** the whole recording |
| ++m++ | Pan left (¼ window) | Zoom **out** |
| ++comma++ | Pan right (¼ window) | Zoom **in** |
| ++period++ | View → end of recording | **Zoom to selection** |

Zoom is anchored at the cursor. The mouse wheel also zooms (++shift++ + wheel
pans); the arrow keys pan a quarter-window and ++home++ / ++end++ jump the view
to the file ends.

## Bundles

| Key | Action |
| --- | --- |
| ++q++ | First bundle |
| ++w++ | Previous bundle |
| ++e++ | Next bundle |
| ++r++ | Last bundle |

## Annotation

| Key | Action |
| --- | --- |
| ++1++ … ++9++ | Activate the tier at that lane position (top = 1) |
| ++shift+1++ … ++shift+9++ | Toggle that tier in / out of the active set (several at once) |
| ++0++ | Clear all active tiers |
| ++enter++ | Commit the current selection to the active tiers (span → intervals, point → points) |
| ++backspace++ / ++delete++ | Delete the selected annotation |

## Global

| Key | Action |
| --- | --- |
| ++ctrl+p++ / ++cmd+p++ | Open the command palette |
| ++ctrl+enter++ / ++cmd+enter++ | Run the script buffer (when the script panel is open) |

## Mouse

| Action | Effect |
| --- | --- |
| Click | Place the cursor (and a selection point) |
| Drag on a lane | Draw a span selection |
| Double-click an interval | Edit its label |
| Wheel | Zoom around the pointer |
| ++shift++ + wheel | Pan the view |

## Scripting the same navigation

The cursor/view/selection model is also exposed to Python as
`sadda.Timeline`, with a **move-to** (absolute) / **move-by** (relative)
method for each action — e.g. `set_cursor(t)` vs
`move_cursor_by(dt)`, `set_view_range(start, end)` for fit/zoom-to-selection.
The desktop keys and the Python API share one implementation in the engine.
