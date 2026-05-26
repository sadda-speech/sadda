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

    # pin a version into the project for reproducibility
    proj.pin_refdist(rd.id, rd.version)
    proj.refdist_pins()            # → {id: version}

Stability tier: PROVISIONAL.
"""

from __future__ import annotations

from typing import Any, Optional

from sadda import _native
from sadda._stability import provisional

__all__ = ["RefDist", "get", "install", "list_all", "query", "store_root"]

RefDist = _native.refdist.RefDist


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
def store_root(*, root: Optional[str] = None) -> str:
    """Filesystem path of the active store (the per-user cache by
    default, created if missing)."""
    return _native.refdist.store_root(root=root)
