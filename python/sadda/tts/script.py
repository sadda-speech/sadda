"""sadda.tts.script — the narration-script model.

A *narration script* is an ordered list of :class:`Segment`\\ s. Each segment is
a span of text to speak plus optional per-segment overrides: the voice, a
relative speaking rate, a trailing pause, and an optional stable ``id`` (e.g. a
scene name) that downstream tooling can use to align a segment with a
screencast marker or a caption.

The Python API (building a :class:`NarrationScript` from :class:`Segment`\\ s, or
via :meth:`NarrationScript.from_texts`) is the primary surface. :func:`parse_script`
is a deliberately minimal text convenience — blank-line-separated paragraphs
become segments — enough to voice a tutorial today. A richer on-disk format
(per-scene ids, inline voice/rate directives, screencast timing markers) is an
open design question tracked in the 2026-07-05 DEVLOG entry; it is intentionally
not committed here.

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from sadda._stability import provisional

__all__ = ["Segment", "NarrationScript", "parse_script", "load_script"]


@dataclass(frozen=True)
class Segment:
    """One span of narration.

    Attributes:
        text: The words to speak. Leading/trailing whitespace is ignored.
        voice: Backend-specific voice id (e.g. ``"en-us"`` for espeak-ng),
            or ``None`` to fall back to the script default, then the backend
            default. Voice strings are intentionally backend-specific — the
            abstraction passes them through unchanged.
        rate: Relative speaking-rate multiplier where ``1.0`` is the backend's
            natural rate, ``1.2`` is 20% faster, ``0.9`` is 10% slower.
            ``None`` falls back to the script default, then the backend default.
        pause_after_s: Silence, in seconds, appended after this segment when a
            script is assembled into a single track.
        id: Optional stable identifier (e.g. a scene/slide name). Not used for
            synthesis; carried through so callers can align segments with
            screencast markers or captions.
    """

    text: str
    voice: Optional[str] = None
    rate: Optional[float] = None
    pause_after_s: float = 0.0
    id: Optional[str] = None


@provisional
@dataclass(frozen=True)
class NarrationScript:
    """An ordered collection of :class:`Segment`\\ s with script-wide defaults.

    A segment's own ``voice`` / ``rate`` win over the script-wide defaults,
    which in turn win over the backend's defaults.
    """

    segments: tuple[Segment, ...] = field(default_factory=tuple)
    voice: Optional[str] = None
    rate: Optional[float] = None

    def __post_init__(self) -> None:
        # Accept any iterable of segments but store an immutable tuple.
        object.__setattr__(self, "segments", tuple(self.segments))

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        *,
        voice: Optional[str] = None,
        rate: Optional[float] = None,
        pause_after_s: float = 0.0,
    ) -> "NarrationScript":
        """Build a script from plain strings, one :class:`Segment` each."""
        segs = tuple(Segment(text=t, pause_after_s=pause_after_s) for t in texts)
        return cls(segments=segs, voice=voice, rate=rate)

    def resolved_voice(self, segment: Segment) -> Optional[str]:
        """The voice a segment should use, applying the fallback chain."""
        return segment.voice if segment.voice is not None else self.voice

    def resolved_rate(self, segment: Segment) -> Optional[float]:
        """The rate a segment should use, applying the fallback chain."""
        return segment.rate if segment.rate is not None else self.rate

    def __len__(self) -> int:
        return len(self.segments)

    def __iter__(self):
        return iter(self.segments)


@provisional
def parse_script(text: str) -> NarrationScript:
    """Parse a minimal plain-text narration script.

    Blank-line-separated paragraphs become one :class:`Segment` each; internal
    single newlines are collapsed to spaces so a soft-wrapped paragraph reads as
    one utterance. Empty paragraphs are dropped.

    This is intentionally minimal (see the module docstring). For anything
    richer, build :class:`Segment`\\ s directly.
    """
    paragraphs = [blk.strip() for blk in text.replace("\r\n", "\n").split("\n\n")]
    segs = tuple(
        Segment(text=" ".join(line.strip() for line in blk.splitlines()))
        for blk in paragraphs
        if blk
    )
    return NarrationScript(segments=segs)


@provisional
def load_script(path: Union[str, Path]) -> NarrationScript:
    """Load a narration script from a text file via :func:`parse_script`."""
    return parse_script(Path(path).read_text(encoding="utf-8"))
