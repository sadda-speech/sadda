"""C7 — reference distributions (consumption side), Python surface."""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path

import polars as pl
import pytest
import sadda


def _write_dist(store: Path, dirname: str, dist_id: str, version: str) -> None:
    d = store / dirname
    d.mkdir(parents=True, exist_ok=True)
    (d / "refdist.toml").write_text(
        textwrap.dedent(
            f"""
            id = "{dist_id}"
            version = "{version}"
            title = "Test {dist_id}"

            [citation]
            authors = ["Doe, J."]
            year = 2025

            [population]
            language = "eng"
            variety = "AmE"
            sex = ["m", "f"]
            age_band = ["adult"]
            n_speakers = 42

            [measure]
            kind = "observed_distribution"
            parameters = ["F1", "F2"]
            units = "Hz"
            phones = ["iy", "ae"]

            [schema]
            data_file = "data.parquet"
            shape = "long"
            columns = ["speaker_id", "phone", "F1", "F2"]
            """
        )
    )
    pl.DataFrame(
        {
            "speaker_id": [1, 2],
            "phone": ["iy", "ae"],
            "F1": [300.0, 700.0],
            "F2": [2300.0, 1600.0],
        }
    ).write_parquet(d / "data.parquet")


@pytest.fixture
def store(tmp_path) -> str:
    root = tmp_path / "refdist_store"
    _write_dist(root, "hb95", "hillenbrand-1995-amE-vowels", "1.0.0")
    _write_dist(root, "other", "other-vowels", "2.0.0")
    (root / "junk").mkdir()  # not a distribution
    return str(root)


def test_list_and_get(store) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        all_dists = sadda.refdist.list_all(root=store)
        assert len(all_dists) == 2
        rd = sadda.refdist.get("hillenbrand-1995-amE-vowels", "1.0.0", root=store)
    assert rd is not None
    assert rd.version == "1.0.0"
    assert rd.kind == "observed_distribution"
    assert rd.parameters == ["F1", "F2"]
    assert rd.units == "Hz"
    assert rd.language == "eng"
    assert "f" in rd.sex
    assert rd.year == 2025


def test_query_facets(store) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert len(sadda.refdist.query(parameter="f1", root=store)) == 2  # case-insensitive
        assert len(sadda.refdist.query(sex="f", language="eng", root=store)) == 2
        assert sadda.refdist.query(parameter="F3", root=store) == []
        assert sadda.refdist.query(kind="target_zone", root=store) == []


def test_data_loads_parquet(store) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rd = sadda.refdist.get("other-vowels", "2.0.0", root=store)
        df = rd.data()
    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["speaker_id", "phone", "F1", "F2"]
    assert df.height == 2


def test_query_is_provisional() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.refdist.query) == "provisional"
    assert get_stability(sadda.refdist.get) == "provisional"


# ----- D10: summary / histogram / points2d data helpers -------------------


def _write_observed(store: Path) -> None:
    d = store / "obs"
    d.mkdir(parents=True, exist_ok=True)
    (d / "refdist.toml").write_text(
        textwrap.dedent(
            """
            id = "obs-vowels"
            version = "1.0.0"
            title = "Observed vowels"

            [measure]
            kind = "observed_distribution"
            parameters = ["F1", "F2"]
            units = "Hz"
            phones = ["iy", "ae"]

            [schema]
            data_file = "data.parquet"
            shape = "long"
            columns = ["phone", "F1", "F2"]
            """
        )
    )
    pl.DataFrame(
        {
            "phone": ["iy", "iy", "iy", "ae", "ae", "ae"],
            "F1": [300.0, 310.0, 290.0, 700.0, 710.0, 690.0],
            "F2": [2300.0, 2320.0, 2280.0, 1700.0, 1710.0, 1690.0],
        }
    ).write_parquet(d / "data.parquet")


def _write_normative(store: Path) -> None:
    d = store / "norm"
    d.mkdir(parents=True, exist_ok=True)
    (d / "refdist.toml").write_text(
        textwrap.dedent(
            """
            id = "f0-norms"
            version = "1.0.0"
            title = "f0 norms"

            [population]
            n_speakers = 200

            [measure]
            kind = "summary_normative_range"
            parameters = ["f0"]
            units = "Hz"

            [schema]
            data_file = "data.parquet"
            shape = "long"
            columns = ["sex", "stat", "f0"]
            """
        )
    )
    pl.DataFrame(
        {
            "sex": ["m", "m", "f", "f"],
            "stat": ["mean", "sd", "mean", "sd"],
            "f0": [120.0, 18.0, 210.0, 22.0],
        }
    ).write_parquet(d / "data.parquet")


def _get(store: Path, dist_id: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sadda.refdist.get(dist_id, "1.0.0", root=str(store))


def test_column_reads_and_filters(tmp_path) -> None:
    _write_observed(tmp_path)
    rd = _get(tmp_path, "obs-vowels")
    assert rd.column("F1") == [300.0, 310.0, 290.0, 700.0, 710.0, 690.0]
    # case-insensitive filter
    assert rd.column("F1", filter={"phone": "IY"}) == [300.0, 310.0, 290.0]


def test_summary_observed(tmp_path) -> None:
    _write_observed(tmp_path)
    rd = _get(tmp_path, "obs-vowels")
    s = rd.summary("F1", filter={"phone": "iy"})
    assert s.n == 3
    assert s.mean == pytest.approx(300.0)
    assert s.median == pytest.approx(300.0)
    assert s.min == pytest.approx(290.0)
    assert s.max == pytest.approx(310.0)
    assert repr(s).startswith("Summary(")


def test_histogram_observed(tmp_path) -> None:
    _write_observed(tmp_path)
    rd = _get(tmp_path, "obs-vowels")
    h = rd.histogram("F1", bins=5)
    assert len(h.edges) == len(h.counts) + 1
    assert sum(h.counts) == 6


def test_points2d_pairs_columns(tmp_path) -> None:
    _write_observed(tmp_path)
    rd = _get(tmp_path, "obs-vowels")
    xs, ys = rd.points2d("F1", "F2", filter={"phone": "ae"})
    assert xs == [700.0, 710.0, 690.0]
    assert ys == [1700.0, 1710.0, 1690.0]


def test_summary_normative_band(tmp_path) -> None:
    _write_normative(tmp_path)
    rd = _get(tmp_path, "f0-norms")
    m = rd.summary("f0", filter={"sex": "m"})
    assert m.mean == pytest.approx(120.0)
    assert m.sd == pytest.approx(18.0)
    assert m.median == pytest.approx(120.0)
    assert m.n == 200
    # No filter pools the two means (120, 210) → 165.
    assert rd.summary("f0").mean == pytest.approx(165.0)


def test_histogram_rejects_normative(tmp_path) -> None:
    _write_normative(tmp_path)
    rd = _get(tmp_path, "f0-norms")
    with pytest.raises(Exception):
        rd.histogram("f0")


def test_project_pins_refdist(tmp_path) -> None:
    proj = sadda.new_project(str(tmp_path / "proj"), "pins")
    assert proj.refdist_pins() == {}
    proj.pin_refdist("hillenbrand-1995-amE-vowels", "1.0.0")
    proj.pin_refdist("clinical-jitter-norms", "0.2.0")
    assert proj.refdist_pins() == {
        "hillenbrand-1995-amE-vowels": "1.0.0",
        "clinical-jitter-norms": "0.2.0",
    }
    assert proj.remove_refdist_pin("clinical-jitter-norms") is True
    assert proj.remove_refdist_pin("clinical-jitter-norms") is False
    assert proj.refdist_pins() == {"hillenbrand-1995-amE-vowels": "1.0.0"}
