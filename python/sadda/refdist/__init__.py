"""sadda.refdist — reference distributions, consumption side (C7).

A *reference distribution* is a tagged statistical summary (or sample) of
an acoustic/articulatory measure over a population — vowel-formant clouds,
age/sex-normed clinical ranges, f0 statistics — or a prescriptive
``target_zone``. This module resolves them from a user-level cache
(``~/.local/share/sadda/refdist/`` or the platform equivalent) and queries
them by population/measure facets.

This is the *consumption* half only: parse, resolve, query, and (on the
project) pin versions for reproducibility. Fetching from a hosted registry
arrives with the registry itself (C8). The governance + ``refdist.toml``
format are in the 2026-05-18 "Reference distribution governance" DEVLOG
entry.

Typical usage::

    import sadda

    # everything in the local store
    sadda.refdist.list_all()

    # faceted query (case-insensitive; omitted facets match anything)
    hits = sadda.refdist.query(parameter="F1", language="eng", sex="f")
    rd = hits[0]
    rd.id, rd.version, rd.kind, rd.parameters, rd.units
    df = rd.data()                 # → polars.DataFrame (reads data.parquet)

    # D10 band / histogram / vowel-space helpers (computed in the engine,
    # so a script and the GUI overlays see identical numbers):
    s = rd.summary("F1", filter={"phone": "iy"})   # → Summary(mean, sd, p5…p95)
    h = rd.histogram("F1", bins=20)                # → Histogram(edges, counts)
    xs, ys = rd.points2d("F1", "F2", filter={"phone": "iy"})

    # pin a version into the project for reproducibility
    proj.pin_refdist(rd.id, rd.version)
    proj.refdist_pins()            # → {id: version}

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

from typing import Any, Optional

from sadda import _native
from sadda._stability import provisional

__all__ = [
    "Histogram",
    "RefDist",
    "Summary",
    "get",
    "install",
    "list_all",
    "query",
    "scaffold",
    "store_root",
]

RefDist = _native.refdist.RefDist
# D10: return types of RefDist.summary / .histogram, re-exported so they
# can be type-referenced and introspected from user code.
Summary = _native.refdist.Summary
Histogram = _native.refdist.Histogram


def _data(self: Any):
    """Read this distribution's data file into a ``polars.DataFrame``.

    Returns ``None`` if the manifest declares no data file."""
    import polars as pl

    path = self.data_path()
    return pl.read_parquet(path) if path is not None else None


# Add the Polars convenience onto the PyO3 class (it has no native data
# loader — the engine exposes only the path, mirroring the dense-tier
# `Project.query` pattern from B3).
RefDist.data = _data


@provisional
def list_all(*, root: Optional[str] = None) -> "list[RefDist]":
    """Every distribution in the store (default: the per-user cache)."""
    return _native.refdist.list_all(root=root)


@provisional
def query(
    *,
    parameter: Optional[str] = None,
    language: Optional[str] = None,
    variety: Optional[str] = None,
    sex: Optional[str] = None,
    age_band: Optional[str] = None,
    phone: Optional[str] = None,
    kind: Optional[str] = None,
    root: Optional[str] = None,
) -> "list[RefDist]":
    """Distributions matching the given facets. Any omitted facet matches
    anything; string matches are case-insensitive. ``kind`` is one of
    ``observed_distribution`` | ``summary_normative_range`` |
    ``target_zone``."""
    return _native.refdist.query(
        parameter=parameter,
        language=language,
        variety=variety,
        sex=sex,
        age_band=age_band,
        phone=phone,
        kind=kind,
        root=root,
    )


@provisional
def get(id: str, version: str, *, root: Optional[str] = None) -> Optional[RefDist]:  # noqa: A002
    """The distribution with this ``id`` and ``version``, or ``None``."""
    return _native.refdist.get(id, version, root=root)


@provisional
def install(src_dir: str, *, root: Optional[str] = None) -> RefDist:
    """Install a distribution directory (a ``refdist.toml`` + its data
    file) into the store by copying it in — how the bundled starter set
    seeds the user cache. Returns the installed distribution."""
    return _native.refdist.install(src_dir, root=root)


@provisional
def scaffold(
    dest_dir: str,
    data: Any,
    *,
    id: str,  # noqa: A002
    version: str,
    kind: str,
    parameters: "Optional[list[str]]" = None,
    title: Optional[str] = None,
    doi: Optional[str] = None,
    license: Optional[str] = None,  # noqa: A002
    language: Optional[str] = None,
    variety: Optional[str] = None,
    sex: "Optional[list[str]]" = None,
    age_band: "Optional[list[str]]" = None,
    units: Optional[str] = None,
    phones: "Optional[list[str]]" = None,
    shareability: Optional[str] = None,
    min_n_per_subgroup: Optional[int] = None,
    authors: "Optional[list[str]]" = None,
    year: Optional[int] = None,
    provenance: Optional[str] = None,
) -> RefDist:
    """Scaffold a publishable distribution directory from an analysis
    result (C9). Writes ``data.parquet`` from ``data`` (a
    ``polars.DataFrame``), then ``refdist.toml`` + ``provenance.md`` + a
    ``LICENSE`` stub from the metadata. ``schema.columns`` is taken from
    the DataFrame, and ``n_speakers`` is inferred from a ``speaker_id``
    column if present.

    The result is immediately resolvable and passes the registry
    validator once you (a) replace the LICENSE stub with the full license
    text and (b) fill in real provenance. To submit, copy the directory
    under the registry's ``tier3/<id>/`` and open a fork-and-PR (the auth
    is your GitHub credentials, not sadda's).
    """
    import polars as pl
    from pathlib import Path

    if not isinstance(data, pl.DataFrame):
        raise TypeError("data must be a polars.DataFrame")
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    data.write_parquet(dest / "data.parquet")

    n_speakers = data["speaker_id"].n_unique() if "speaker_id" in data.columns else None
    return _native.refdist.scaffold(
        str(dest),
        id=id,
        version=version,
        kind=kind,
        columns=list(data.columns),
        parameters=parameters,
        data_file="data.parquet",
        title=title,
        doi=doi,
        license=license,
        language=language,
        variety=variety,
        sex=sex,
        age_band=age_band,
        n_speakers=n_speakers,
        units=units,
        phones=phones,
        shareability=shareability,
        min_n_per_subgroup=min_n_per_subgroup,
        authors=authors,
        year=year,
        provenance=provenance,
    )


@provisional
def store_root(*, root: Optional[str] = None) -> str:
    """Filesystem path of the active store (the per-user cache by
    default, created if missing)."""
    return _native.refdist.store_root(root=root)
