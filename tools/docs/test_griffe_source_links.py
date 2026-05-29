"""Tests for the source-link griffe extension (rendering side).

End-to-end rendering is verified by the `mkdocs build` in CI; these
tests pin the extension's pure logic — URL/HTML composition and the
docstring-injection behaviour (including idempotency) — without standing
up a full mkdocs build.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import griffe

sys.path.insert(0, str(Path(__file__).parent))
import griffe_source_links as gsl  # noqa: E402
import source_links as sl  # noqa: E402


def _entry(role, path, line):
    return sl.Entry(role=role, path=path, line=line, source="rust")


def test_render_binding_only():
    html = gsl._render({"binding": _entry("binding", "crates/python/src/lib.rs", 10)})
    assert "Source:" in html
    assert 'blob/main/crates/python/src/lib.rs#L10' in html
    assert "impl:" not in html
    assert gsl._MARKER in html


def test_render_binding_and_impl():
    html = gsl._render(
        {
            "binding": _entry("binding", "crates/python/src/lib.rs", 10),
            "impl": _entry("impl", "crates/engine/src/pitch.rs", 226),
        }
    )
    assert "lib.rs#L10" in html
    assert "impl: " in html
    assert "pitch.rs#L226" in html
    assert html.count("<a ") == 2


def _make_ext(map_):
    # Build a real extension, then swap in a controlled map so the test
    # doesn't depend on the live repo state.
    ext = gsl.SourceLinks.__new__(gsl.SourceLinks)
    ext._map = map_
    return ext


def test_on_object_appends_to_existing_docstring():
    roles = {"binding": _entry("binding", "crates/python/src/lib.rs", 10)}
    ext = _make_ext({"sadda.dsp.f0": roles})
    obj = SimpleNamespace(
        path="sadda.dsp.f0",
        canonical_path="sadda.dsp.f0",
        docstring=griffe.Docstring("Estimate f0."),
    )
    ext.on_object(obj=obj)
    assert "Estimate f0." in obj.docstring.value  # original preserved
    assert "lib.rs#L10" in obj.docstring.value  # link appended

    # Idempotent: a second visit must not append a duplicate.
    before = obj.docstring.value
    ext.on_object(obj=obj)
    assert obj.docstring.value == before


def test_on_object_creates_docstring_when_missing():
    roles = {"binding": _entry("binding", "crates/python/src/lib.rs", 10)}
    ext = _make_ext({"sadda.dsp.f0": roles})
    obj = SimpleNamespace(
        path="sadda.dsp.f0", canonical_path="sadda.dsp.f0", docstring=None
    )
    ext.on_object(obj=obj)
    assert obj.docstring is not None
    assert "lib.rs#L10" in obj.docstring.value


def test_on_object_ignores_unmapped_symbol():
    ext = _make_ext({})
    obj = SimpleNamespace(
        path="sadda.dsp.f0", canonical_path="sadda.dsp.f0", docstring=None
    )
    ext.on_object(obj=obj)
    assert obj.docstring is None  # untouched


def test_lookup_falls_back_to_canonical_path():
    roles = {"binding": _entry("binding", "x.rs", 1)}
    ext = _make_ext({"sadda._native.thing": roles})
    obj = SimpleNamespace(path="sadda.thing", canonical_path="sadda._native.thing")
    assert ext._lookup(obj) is roles
