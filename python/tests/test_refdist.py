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
