"""Resolve documented API symbols to their definition sites in the repo.

This is the load-bearing first slice of the "API-reference source links"
feature (see the 2026-05-28 DEVLOG plan-item entry). It builds a map

    qualified_name -> { "binding": Entry, "impl": Entry? }

where each ``Entry`` is a repo-relative ``(path, line)`` plus a
``main``-pinned GitHub URL. A later slice feeds this map into a griffe
extension so every API-reference heading renders a "source" link; this
module is the resolver + the CI gate, runnable standalone.

Why a bespoke resolver and not mkdocstrings' built-in ``show_source``:
nearly the whole ``sadda`` public surface is re-exported from the Rust
``_native`` extension (``hann = stable(_native.hann)``; ``Project`` etc.),
so the documented objects are PyO3 builtins with no Python source file —
griffe's source-link machinery covers ~none of them. We therefore
resolve names to their *real* definitions ourselves.

Resolution is **hybrid** (per the design decision):

1. **Explicit markers** win: a ``# [docs:<qualname>]`` (Python) or
   ``// [docs:<qualname>]`` (Rust) comment above a definition pins its
   binding link; ``[docs-impl:<qualname>]`` pins its engine-impl link.
2. **Derivation** otherwise: PyO3 ``#[pyclass]`` / ``#[pymethods]`` /
   ``#[pyfunction]`` names for Rust-backed symbols, and ``def``/``class``
   heads for the pure-Python wrapper functions (``sadda.recipe.record``
   etc.).

The **binding** link is required for every explicitly documented symbol
(the CI gate fails on an unresolved binding). The **impl** link — the
engine algorithm behind a thin PyO3 shim — is best-effort: present where
a ``[docs-impl:...]`` marker exists, never gate-required (data classes,
getters, and pure-Python wrappers have no distinct engine impl).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Pinned ref for generated links. The design decision was to pin to
# `main` (always-current code, accepting line drift between docs
# rebuilds) rather than to a commit SHA.
REPO_URL = "https://github.com/sadda-speech/sadda"
PIN_REF = "main"

# Python submodules that prefix qualified names (`sadda.<module>.<sym>`).
# `corpus` is documented via top-level symbols (`sadda.new_project`,
# `sadda.Project`), so it is not a name prefix here.
KNOWN_MODULES = {"dsp", "clinical", "ml", "refdist", "recipe", "live"}

# Repo-relative roots.
PY_BINDINGS_DIR = Path("crates/python/src")
PY_PACKAGE_DIR = Path("python/sadda")
DOCS_API_DIR = Path("docs/api")
ENGINE_DIR = Path("crates/engine/src")


@dataclass(frozen=True)
class Entry:
    """A resolved definition site."""

    role: str  # "binding" | "impl"
    path: str  # repo-relative, forward-slash
    line: int  # 1-based
    source: str  # "rust" | "python" | "marker"

    def url(self) -> str:
        return f"{REPO_URL}/blob/{PIN_REF}/{self.path}#L{self.line}"


# --------------------------------------------------------------------------
# Docs universe: which symbols must resolve.
# --------------------------------------------------------------------------

_DIRECTIVE_RE = re.compile(r"^:::\s+(\S+)\s*$")
_MEMBER_RE = re.compile(r"^\s+-\s+(\w+)\s*$")
_MEMBERS_KEY_RE = re.compile(r"^\s+members:\s*$")


def discover_doc_symbols(repo_root: Path) -> list[tuple[str, Optional[str]]]:
    """Parse ``docs/api/*.md`` for every explicitly documented symbol.

    Returns ``(qualname, parent_qualname_or_None)`` pairs: each ``:::``
    directive, plus every entry of an indented ``members:`` list (whose
    parent is the directive it sits under).
    """
    out: list[tuple[str, Optional[str]]] = []
    for md in sorted((repo_root / DOCS_API_DIR).glob("*.md")):
        lines = md.read_text(encoding="utf-8").splitlines()
        current: Optional[str] = None
        in_members = False
        for line in lines:
            m = _DIRECTIVE_RE.match(line)
            if m:
                current = m.group(1)
                in_members = False
                out.append((current, None))
                continue
            if _MEMBERS_KEY_RE.match(line):
                in_members = True
                continue
            if in_members:
                mm = _MEMBER_RE.match(line)
                if mm and current is not None:
                    out.append((f"{current}.{mm.group(1)}", current))
                elif line.strip() and not line.startswith(" "):
                    in_members = False
    return out


# --------------------------------------------------------------------------
# Rust PyO3 binding index.
# --------------------------------------------------------------------------


@dataclass
class RustIndex:
    classes: dict[str, tuple[str, int]]  # pyclass name -> (path, line)
    struct_to_pyname: dict[str, str]  # rust struct -> pyclass name
    methods: dict[tuple[str, str], tuple[str, int]]  # (struct, method) -> (path, line)
    funcs: dict[str, list[tuple[str, int]]]  # pyfunction name -> [(path, line)]


_PYCLASS_RE = re.compile(r'#\[pyclass\(.*?name\s*=\s*"(\w+)"')
_STRUCT_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)")
_IMPL_RE = re.compile(r"^\s*impl\s+(\w+)")
_FN_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(?:r#)?(\w+)")
_ATTR_RE = re.compile(r"^\s*#\[")


def parse_rust_bindings(repo_root: Path) -> RustIndex:
    idx = RustIndex(classes={}, struct_to_pyname={}, methods={}, funcs={})
    for rs in sorted((repo_root / PY_BINDINGS_DIR).glob("*.rs")):
        rel = rs.relative_to(repo_root).as_posix()
        lines = rs.read_text(encoding="utf-8").splitlines()
        pending_pyclass: Optional[str] = None
        pymethods_pending = False
        impl_struct: Optional[str] = None
        impl_depth = 0
        pyfunction_pending = False
        for i, line in enumerate(lines):
            lineno = i + 1
            # --- pyclass head: remember the name, attach at the struct. ---
            mc = _PYCLASS_RE.search(line)
            if mc:
                pending_pyclass = mc.group(1)
                continue
            ms = _STRUCT_RE.match(line)
            if ms and pending_pyclass is not None:
                idx.classes[pending_pyclass] = (rel, lineno)
                idx.struct_to_pyname[ms.group(1)] = pending_pyclass
                pending_pyclass = None
                continue
            # --- pymethods impl block (brace-counted) ---
            if "#[pymethods]" in line:
                pymethods_pending = True
                continue
            if pymethods_pending:
                mi = _IMPL_RE.match(line)
                if mi:
                    impl_struct = mi.group(1)
                    pymethods_pending = False
                    impl_depth = line.count("{") - line.count("}")
                    continue
            if impl_struct is not None:
                if impl_depth > 0:
                    mf = _FN_RE.match(line)
                    # Only count method heads at the block's top level
                    # (depth 1), so nested fns/closures don't register.
                    if mf and impl_depth == 1:
                        idx.methods[(impl_struct, mf.group(1))] = (rel, lineno)
                impl_depth += line.count("{") - line.count("}")
                if impl_depth <= 0:
                    impl_struct = None
                continue
            # --- free pyfunction ---
            if "#[pyfunction]" in line:
                pyfunction_pending = True
                continue
            if pyfunction_pending:
                if _ATTR_RE.match(line):  # skip #[pyo3(...)] etc.
                    continue
                mf = _FN_RE.match(line)
                if mf:
                    idx.funcs.setdefault(mf.group(1), []).append((rel, lineno))
                    pyfunction_pending = False
    return idx


# --------------------------------------------------------------------------
# Python wrapper index (aliases + pure-Python defs).
# --------------------------------------------------------------------------


@dataclass
class NativeAlias:
    sub: Optional[str]  # _native submodule, e.g. "live"; None at top level
    leaf: str


@dataclass
class WrapperModule:
    aliases: dict[str, NativeAlias]  # pyname -> native target
    pydefs: dict[str, tuple[str, int]]  # pyname -> (path, line)


_ALIAS_RE = re.compile(
    r"^(\w+)\s*=\s*"
    r"(?:(?:stable|stable_clinical|provisional|experimental)\()?"
    r"_native((?:\.\w+)+)\)?\s*$"
)
_PYDEF_RE = re.compile(r"^(?:async\s+)?(?:def|class)\s+(\w+)")


def _parse_wrapper_file(path: Path, rel: str) -> WrapperModule:
    mod = WrapperModule(aliases={}, pydefs={})
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        ma = _ALIAS_RE.match(line)
        if ma:
            chain = ma.group(2).strip(".").split(".")
            leaf = chain[-1]
            sub = chain[-2] if len(chain) > 1 else None
            mod.aliases[ma.group(1)] = NativeAlias(sub=sub, leaf=leaf)
            continue
        md = _PYDEF_RE.match(line)  # column-0 def/class only
        if md:
            mod.pydefs[md.group(1)] = (rel, i + 1)
    return mod


def parse_python_wrappers(repo_root: Path) -> dict[str, WrapperModule]:
    """Index ``python/sadda/__init__.py`` (key ``""``) and each
    ``python/sadda/<module>/__init__.py`` (key ``<module>``)."""
    out: dict[str, WrapperModule] = {}
    top = repo_root / PY_PACKAGE_DIR / "__init__.py"
    out[""] = _parse_wrapper_file(top, top.relative_to(repo_root).as_posix())
    for module in KNOWN_MODULES:
        f = repo_root / PY_PACKAGE_DIR / module / "__init__.py"
        if f.exists():
            out[module] = _parse_wrapper_file(f, f.relative_to(repo_root).as_posix())
    return out


# --------------------------------------------------------------------------
# Explicit markers.
# --------------------------------------------------------------------------

_MARKER_RE = re.compile(r"\[docs(-impl)?:\s*([\w.]+)\s*\]")
_DEF_HEAD_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
    r"(?:fn|struct|def|class|impl)\b"
)
_SKIPPABLE_RE = re.compile(r"^\s*(?://|#|@|$)")


def _is_skippable(line: str) -> bool:
    """A comment (``//``/``///``/``#``), a Rust attribute (``#[...]``), a
    Python decorator (``@...``), or a blank line — the lines that
    legitimately sit between a marker and the definition head it tags."""
    return bool(_SKIPPABLE_RE.match(line))


def collect_markers(repo_root: Path) -> dict[str, dict[str, Entry]]:
    """Scan Rust + Python sources for ``[docs:...]`` / ``[docs-impl:...]``
    comments. Each marker pins the next definition head below it."""
    out: dict[str, dict[str, Entry]] = {}
    roots = [
        repo_root / PY_BINDINGS_DIR,
        repo_root / PY_PACKAGE_DIR,
        repo_root / ENGINE_DIR,
    ]
    files: list[Path] = []
    for root in roots:
        files += root.rglob("*.rs")
        files += root.rglob("*.py")
    for f in sorted(set(files)):
        rel = f.relative_to(repo_root).as_posix()
        lines = f.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            mm = _MARKER_RE.search(line)
            if not mm:
                continue
            role = "impl" if mm.group(1) else "binding"
            qual = mm.group(2)
            # Point at the definition head below the marker, skipping the
            # comment / doc-comment / attribute lines (incl. the marker's
            # own continuation) that sit between a marker and the `fn` /
            # `def` / `struct` it tags. Stop at the first real code line.
            target = i + 2  # default: the line right after the marker
            for j in range(i + 1, len(lines)):
                if _DEF_HEAD_RE.match(lines[j]):
                    target = j + 1
                    break
                if _is_skippable(lines[j]):
                    continue
                break  # a non-comment, non-def line: give up, keep default
            out.setdefault(qual, {})[role] = Entry(role, rel, target, "marker")
    return out


# --------------------------------------------------------------------------
# Resolution.
# --------------------------------------------------------------------------


def _split_qual(qualname: str) -> tuple[str, list[str]]:
    parts = qualname.split(".")
    assert parts[0] == "sadda", f"unexpected qualname root: {qualname}"
    return parts[0], parts[1:]


def _resolve_binding(
    qualname: str,
    rust: RustIndex,
    wrappers: dict[str, WrapperModule],
) -> Optional[Entry]:
    _, rest = _split_qual(qualname)

    # Module directive itself (e.g. `sadda.dsp`): link the wrapper module.
    if len(rest) == 1 and rest[0] in KNOWN_MODULES:
        mod = wrappers.get(rest[0])
        if mod is not None:
            # The package __init__ — line 1 is a fine, stable head.
            rel = (PY_PACKAGE_DIR / rest[0] / "__init__.py").as_posix()
            return Entry("binding", rel, 1, "python")
        return None

    # Determine module prefix, leaf, and optional class.
    if rest and rest[0] in KNOWN_MODULES:
        module, tail = rest[0], rest[1:]
    else:
        module, tail = "", rest

    if len(tail) == 2 and tail[0] in rust.classes:
        # module.Class.method  or  Class.method (module == "")
        struct = _struct_for(rust, tail[0])
        if struct is not None:
            return _entry(rust.methods.get((struct, tail[1])), "binding", "rust")
        return None

    if len(tail) == 1:
        leaf = tail[0]
        wrapper = wrappers.get(module)
        # A pure-Python wrapper def/class wins (recipe.record, ml.vad, ...).
        if wrapper is not None and leaf in wrapper.pydefs:
            path, line = wrapper.pydefs[leaf]
            return Entry("binding", path, line, "python")
        # A class alias -> the Rust #[pyclass].
        if leaf in rust.classes:
            return _entry(rust.classes.get(leaf), "binding", "rust")
        # A free-function alias -> the Rust #[pyfunction].
        if leaf in rust.funcs:
            cands = rust.funcs[leaf]
            chosen = _disambiguate_func(cands, module)
            return _entry(chosen, "binding", "rust")
        return None

    return None


def _struct_for(rust: RustIndex, pyname: str) -> Optional[str]:
    for struct, name in rust.struct_to_pyname.items():
        if name == pyname:
            return struct
    return None


def _disambiguate_func(
    cands: list[tuple[str, int]], module: str
) -> Optional[tuple[str, int]]:
    if len(cands) == 1:
        return cands[0]
    # Prefer the binding file matching the submodule; top-level -> lib.rs.
    want = f"{module}.rs" if module else "lib.rs"
    for path, line in cands:
        if path.endswith(want):
            return (path, line)
    return cands[0] if cands else None


def _entry(
    loc: Optional[tuple[str, int]], role: str, source: str
) -> Optional[Entry]:
    if loc is None:
        return None
    return Entry(role, loc[0], loc[1], source)


def _canonical_native_key(
    qualname: str,
    rust: RustIndex,
    wrappers: dict[str, WrapperModule],
) -> Optional[str]:
    """The `sadda._native...` path the runtime object actually lives at.

    Under mkdocstrings' runtime inspection, the griffe object handed to the
    extension is the *real* PyO3 object (path `sadda._native.f0`), while the
    documented name (`sadda.dsp.f0`) is an alias to it. Emitting the entry
    under this canonical key too lets the extension match either. Returns
    None for symbols whose real object already sits at the documented path
    (pure-Python wrapper defs, module directives)."""
    _, rest = _split_qual(qualname)
    if len(rest) == 1 and rest[0] in KNOWN_MODULES:
        return None
    if rest and rest[0] in KNOWN_MODULES:
        module, tail = rest[0], rest[1:]
    else:
        module, tail = "", rest
    if len(tail) == 2 and tail[0] in rust.classes:
        sub = f".{module}" if module else ""
        return f"sadda._native{sub}.{tail[0]}.{tail[1]}"
    if len(tail) == 1:
        wrapper = wrappers.get(module)
        if wrapper is None:
            return None
        leaf = tail[0]
        if leaf in wrapper.pydefs:
            return None  # real object already at the documented path
        if leaf in wrapper.aliases:
            ref = wrapper.aliases[leaf]
            sub = f".{ref.sub}" if ref.sub else ""
            return f"sadda._native{sub}.{ref.leaf}"
    return None


def build_map(
    repo_root: Path,
) -> tuple[dict[str, dict[str, Entry]], list[str]]:
    """Build the qualname -> {role: Entry} map. Returns the map plus the
    list of documented symbols with no resolvable **binding** (the CI
    gate fails when this list is non-empty)."""
    doc_symbols = discover_doc_symbols(repo_root)
    rust = parse_rust_bindings(repo_root)
    wrappers = parse_python_wrappers(repo_root)
    markers = collect_markers(repo_root)

    result: dict[str, dict[str, Entry]] = {}
    unresolved: list[str] = []
    for qualname, _parent in doc_symbols:
        roles: dict[str, Entry] = {}
        # Binding: marker override, else derive.
        if qualname in markers and "binding" in markers[qualname]:
            roles["binding"] = markers[qualname]["binding"]
        else:
            b = _resolve_binding(qualname, rust, wrappers)
            if b is not None:
                roles["binding"] = b
        # Impl: marker only (best-effort).
        if qualname in markers and "impl" in markers[qualname]:
            roles["impl"] = markers[qualname]["impl"]
        if "binding" not in roles:
            unresolved.append(qualname)
        if roles:
            result[qualname] = roles
            # Also register under the canonical _native path so the griffe
            # extension matches when mkdocstrings inspects the runtime.
            canon = _canonical_native_key(qualname, rust, wrappers)
            if canon and canon not in result:
                result[canon] = roles
    return result, sorted(set(unresolved))


def _serialize(result: dict[str, dict[str, Entry]]) -> dict:
    return {
        qual: {
            role: {**asdict(entry), "url": entry.url()}
            for role, entry in roles.items()
        }
        for qual, roles in sorted(result.items())
    }


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: inferred from this file's location)",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        help="write the JSON map here (default: stdout)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any documented symbol has no binding link",
    )
    args = ap.parse_args(argv)

    result, unresolved = build_map(args.repo_root)
    payload = json.dumps(_serialize(result), indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

    # Count documented symbols only — `result` also carries canonical
    # `sadda._native...` alias keys for the griffe extension's benefit.
    documented = {q: r for q, r in result.items() if not q.startswith("sadda._native")}
    n_impl = sum(1 for r in documented.values() if "impl" in r)
    print(
        f"resolved {len(documented)} documented symbols "
        f"({len(documented) - n_impl} binding-only, {n_impl} with impl links); "
        f"{len(unresolved)} unresolved",
        file=sys.stderr,
    )
    if unresolved:
        print("UNRESOLVED (no binding link):", file=sys.stderr)
        for q in unresolved:
            print(f"  - {q}", file=sys.stderr)
        if args.check:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
