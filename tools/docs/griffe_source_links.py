"""griffe extension: append a repo source link to each documented symbol.

Slice 2 of the API-reference source-links feature (see the 2026-05-28
plan-item and 2026-05-29 slice-1 DEVLOG entries). Slice 1 built the
resolver (``source_links.build_map``); this extension consumes its
``qualname -> {role: Entry}`` map and, during ``mkdocs build``, appends a
"Source" line to every rendered API entry pointing at the definition in
the repo.

Why inject into the docstring rather than override a template: nearly
the whole ``sadda`` surface is re-exported from the Rust ``_native``
extension, so mkdocstrings resolves these objects by *runtime
inspection* and the rendered object's ``path`` is the documented dotted
name (``sadda.dsp.f0``). Appending a small HTML block to the object's
docstring renders reliably across mkdocstrings/griffe versions without
depending on the internal template layout — the link shows at the foot
of each entry.

Wired in ``mkdocs.yml`` under the python handler's ``options.extensions``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import griffe

# Import the slice-1 resolver as a sibling module regardless of how
# griffe loads this file (by path or by module name).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_links as sl  # noqa: E402

# Sentinel so we never append the link twice if an object is visited
# more than once during loading.
_MARKER = "doc-source-link"


def _render(roles: dict) -> str:
    """Build the HTML source line for one symbol's role map."""
    parts = []
    binding = roles.get("binding")
    if binding is not None:
        parts.append(
            f'<a href="{binding.url()}" title="Python binding / definition">'
            f"<code>{binding.path}:{binding.line}</code></a>"
        )
    impl = roles.get("impl")
    if impl is not None:
        parts.append(
            f'<span class="doc-source-impl">impl: '
            f'<a href="{impl.url()}" title="engine implementation">'
            f"<code>{impl.path}:{impl.line}</code></a></span>"
        )
    if not parts:
        return ""
    inner = " · ".join(parts)
    return f'<p class="{_MARKER}"><small>Source: {inner}</small></p>'


class SourceLinks(griffe.Extension):
    """Appends a repo source link to each documented object's docstring."""

    def __init__(self, repo_root: str | None = None) -> None:
        # The docs build runs from the repo root (where mkdocs.yml lives);
        # fall back to inferring from this file's location.
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        try:
            self._map, _unresolved = sl.build_map(root)
        except Exception as exc:  # never break the docs build over links
            print(f"[source-links] failed to build link map: {exc}", file=sys.stderr)
            self._map = {}

    def _lookup(self, obj: griffe.Object) -> dict | None:
        # Match on the documented dotted path first, then the canonical
        # path (covers the rare case where mkdocstrings hands us the
        # resolved target rather than the documented alias).
        for key in (getattr(obj, "path", None), getattr(obj, "canonical_path", None)):
            if key and key in self._map:
                return self._map[key]
        return None

    def on_object(self, *, obj: griffe.Object, **kwargs) -> None:
        roles = self._lookup(obj)
        if roles is None:
            return
        snippet = _render(roles)
        if not snippet:
            return
        if obj.docstring is None:
            obj.docstring = griffe.Docstring(snippet, parent=obj)
            return
        if _MARKER in obj.docstring.value:
            return  # already injected
        obj.docstring.value = obj.docstring.value.rstrip() + "\n\n" + snippet
