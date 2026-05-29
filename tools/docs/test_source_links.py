"""Tests for the API-reference source-link scanner.

Two layers:

* **Real-repo** tests double as the CI gate — every documented symbol
  must resolve to a binding link, and a spot-check table guards the
  resolution of each symbol *kind* (alias, method, pure-Python wrapper,
  module class, marker overrides) against regressions.
* **Synthetic-fixture** tests exercise the parsers in isolation so they
  stay meaningful as the real codebase evolves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import source_links as sl  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Real repo — the gate + per-kind spot checks.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_map():
    return sl.build_map(REPO_ROOT)


def test_every_documented_symbol_has_a_binding(real_map):
    """The CI gate: no documented API symbol may lack a binding link."""
    _result, unresolved = real_map
    assert unresolved == [], (
        "documented symbols with no resolvable binding link — add a "
        f"# [docs:<qualname>] marker at the definition:\n{unresolved}"
    )


@pytest.mark.parametrize(
    "qualname, source, path_suffix",
    [
        # alias -> Rust #[pyfunction]
        ("sadda.dsp.f0", "rust", "crates/python/src/lib.rs"),
        # pymethods method
        ("sadda.Project.add_bundle", "rust", "crates/python/src/lib.rs"),
        # pure-Python wrapper def (and a rename: rust fn is list_recipes)
        ("sadda.recipe.record", "python", "python/sadda/recipe/__init__.py"),
        ("sadda.recipe.list", "python", "python/sadda/recipe/__init__.py"),
        # module-namespaced class + its method
        ("sadda.refdist.RefDist", "rust", "crates/python/src/refdist.rs"),
        ("sadda.live.LiveSession.start", "rust", "crates/python/src/live.rs"),
        # top-level class + function
        ("sadda.Audio", "rust", "crates/python/src/lib.rs"),
        ("sadda.new_project", "rust", "crates/python/src/lib.rs"),
        # monkey-patched -> resolved via explicit marker
        ("sadda.Project.query", "marker", "python/sadda/__init__.py"),
    ],
)
def test_binding_resolution_by_kind(real_map, qualname, source, path_suffix):
    result, _ = real_map
    entry = result[qualname]["binding"]
    assert entry.source == source
    assert entry.path == path_suffix
    assert entry.line >= 1


def test_canonical_native_keys_emitted(real_map):
    """The map also carries `sadda._native...` alias keys so the griffe
    extension matches objects under mkdocstrings' runtime inspection."""
    result, _ = real_map
    # Re-exported function: documented sadda.dsp.f0 lives at _native.f0.
    assert result["sadda._native.f0"] is result["sadda.dsp.f0"]
    # Class method: documented sadda.Project.add_bundle.
    assert "sadda._native.Project.add_bundle" in result
    # Submodule class: documented sadda.refdist.RefDist.
    assert "sadda._native.refdist.RefDist" in result
    # Pure-Python wrappers keep their documented path (no _native object).
    assert "sadda._native.recipe.record" not in result


def test_impl_marker_produces_an_impl_link(real_map):
    result, _ = real_map
    impl = result["sadda.dsp.f0"].get("impl")
    assert impl is not None, "the [docs-impl:sadda.dsp.f0] marker should resolve"
    assert impl.path == "crates/engine/src/pitch.rs"
    assert impl.url().startswith(
        "https://github.com/sadda-speech/sadda/blob/main/"
    )


def test_resolved_lines_point_at_real_definitions(real_map):
    """A line a binding resolves to must actually contain a definition
    head — guards against off-by-one drift in the parsers."""
    result, _ = real_map
    for qualname in ("sadda.dsp.f0", "sadda.Project.add_bundle", "sadda.Audio"):
        entry = result[qualname]["binding"]
        text = (REPO_ROOT / entry.path).read_text().splitlines()[entry.line - 1]
        assert ("fn " in text) or ("struct " in text), f"{qualname}: {text!r}"


# --------------------------------------------------------------------------
# Synthetic fixtures — parser units.
# --------------------------------------------------------------------------


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_parse_rust_bindings(tmp_path):
    _write(
        tmp_path,
        "crates/python/src/lib.rs",
        "\n".join(
            [
                '#[pyclass(name = "Audio")]',  # 1
                "struct PyAudio {}",  # 2
                "",  # 3
                "#[pymethods]",  # 4
                "impl PyAudio {",  # 5
                "    fn sample_rate(&self) -> u32 {",  # 6
                "        fn helper() -> u32 { 1 }",  # 7 nested, must NOT count
                "        helper()",  # 8
                "    }",  # 9
                "    fn r#type(&self) -> &str { \"x\" }",  # 10 raw ident
                "}",  # 11
                "",  # 12
                "#[pyfunction]",  # 13
                "fn f0() {}",  # 14
                "",  # 15
                "#[pyfunction]",  # 16
                "#[pyo3(signature = (x))]",  # 17 attribute, skipped
                "fn formants(x: i32) {}",  # 18
            ]
        ),
    )
    idx = sl.parse_rust_bindings(tmp_path)
    assert idx.classes["Audio"] == ("crates/python/src/lib.rs", 2)
    assert idx.struct_to_pyname["PyAudio"] == "Audio"
    assert idx.methods[("PyAudio", "sample_rate")][1] == 6
    assert idx.methods[("PyAudio", "type")][1] == 10  # r# stripped
    assert ("PyAudio", "helper") not in idx.methods  # nested fn ignored
    assert idx.funcs["f0"] == [("crates/python/src/lib.rs", 14)]
    assert idx.funcs["formants"][0][1] == 18  # attribute line skipped


def test_parse_python_wrappers(tmp_path):
    _write(
        tmp_path,
        "python/sadda/__init__.py",
        "Audio = stable(_native.Audio)\n"
        "new_project = stable(_native.new_project)\n",
    )
    _write(
        tmp_path,
        "python/sadda/live/__init__.py",
        "start_session = provisional(_native.live.start_session)\n"
        "LiveSession = _native.live.LiveSession\n",
    )
    wrappers = sl.parse_python_wrappers(tmp_path)
    top = wrappers[""]
    assert top.aliases["Audio"] == sl.NativeAlias(sub=None, leaf="Audio")
    live = wrappers["live"]
    assert live.aliases["start_session"] == sl.NativeAlias(
        sub="live", leaf="start_session"
    )


def test_collect_markers_skips_doc_comments(tmp_path):
    _write(
        tmp_path,
        "crates/engine/src/pitch.rs",
        "\n".join(
            [
                "// [docs-impl:sadda.dsp.f0]  some note",  # 1 marker
                "// continuation of the note",  # 2 comment
                "/// Doc comment line.",  # 3 doc comment
                "/// More docs.",  # 4
                "pub fn autocorrelation() {}",  # 5 <- target
            ]
        ),
    )
    _write(
        tmp_path,
        "python/sadda/__init__.py",
        "# [docs:sadda.Project.query]\n@provisional\ndef _q(self): ...\n",
    )
    markers = sl.collect_markers(tmp_path)
    impl = markers["sadda.dsp.f0"]["impl"]
    assert impl.role == "impl"
    assert impl.line == 5  # landed on the fn, not a doc comment
    binding = markers["sadda.Project.query"]["binding"]
    assert binding.role == "binding"
    assert binding.line == 3  # skipped the @decorator


def test_discover_doc_symbols(tmp_path):
    _write(
        tmp_path,
        "docs/api/corpus.md",
        "\n".join(
            [
                "::: sadda.Project",
                "    options:",
                "      members:",
                "        - add_bundle",
                "        - query",
                "",
                "::: sadda.Audio",
            ]
        ),
    )
    syms = sl.discover_doc_symbols(tmp_path)
    assert ("sadda.Project", None) in syms
    assert ("sadda.Project.add_bundle", "sadda.Project") in syms
    assert ("sadda.Project.query", "sadda.Project") in syms
    assert ("sadda.Audio", None) in syms


def test_build_map_end_to_end_with_rename_and_marker(tmp_path):
    # A pure-Python wrapper whose name differs from the native fn it calls,
    # plus a monkey-patched method resolved by an explicit marker.
    _write(tmp_path, "docs/api/recipe.md", "::: sadda.recipe.record\n")
    _write(
        tmp_path,
        "docs/api/corpus.md",
        "::: sadda.Project\n    options:\n      members:\n        - query\n",
    )
    _write(tmp_path, "python/sadda/__init__.py", "Project = stable(_native.Project)\n")
    _write(
        tmp_path,
        "python/sadda/recipe/__init__.py",
        "@provisional\ndef record(p, name):\n    return _native.recipe.start(p, name)\n",
    )
    _write(
        tmp_path,
        "crates/python/src/lib.rs",
        '#[pyclass(name = "Project")]\nstruct PyProject {}\n',
    )
    # The monkey-patch lives in a separate file with a marker.
    _write(
        tmp_path,
        "python/sadda/_patches.py",
        "# [docs:sadda.Project.query]\ndef _project_query(self, t): ...\n",
    )
    result, unresolved = sl.build_map(tmp_path)
    assert unresolved == []
    rec = result["sadda.recipe.record"]["binding"]
    assert rec.source == "python"
    assert rec.path == "python/sadda/recipe/__init__.py"
    qry = result["sadda.Project.query"]["binding"]
    assert qry.source == "marker"
    assert qry.path == "python/sadda/_patches.py"
