"""Python-surface tests for the B2 sparse-annotation API (Tier / Interval /
Point / Reference + the `proj.query(tier_id)` polars wrapper)."""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path

import polars as pl
import pytest

import sadda


def _write_wav(path: Path, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * (sample_rate // 4))


def _project_with_bundle(td: Path) -> tuple[sadda.Project, int]:
    proj = sadda.new_project(td / "p", "demo")
    wav = td / "t.wav"
    _write_wav(wav)
    bundle_id = proj.add_bundle("b", wav)
    return proj, bundle_id


def test_tier_crud_basic() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tid = proj.add_tier(bundle_id, "phones", "interval")
        all_tiers = proj.tiers(bundle_id)
        assert len(all_tiers) == 1
        assert all_tiers[0].id == tid
        assert all_tiers[0].name == "phones"
        assert all_tiers[0].type == "interval"
        assert all_tiers[0].parent_id is None


def test_add_tier_with_unknown_type_errors() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        with pytest.raises(RuntimeError):
            proj.add_tier(bundle_id, "x", "not_a_real_type")


def test_interval_round_trip_through_intervals_accessor() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "words", "interval")
        proj.add_interval(tier, 0.0, 0.5, label="hello")
        proj.add_interval(tier, 0.5, 1.0, label="world")
        rows = proj.intervals(tier)
        assert [r.label for r in rows] == ["hello", "world"]
        assert rows[0].duration_seconds == pytest.approx(0.5)


def test_query_returns_polars_dataframe_for_interval_tier() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "phones", "interval")
        proj.add_interval(tier, 0.0, 0.2, label="h")
        proj.add_interval(tier, 0.2, 0.5, label="e")
        df = proj.query(tier)
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (2, 8)
        assert df.columns == [
            "id",
            "tier_id",
            "start_seconds",
            "end_seconds",
            "duration_seconds",
            "label",
            "parent_annotation_id",
            "extra",
        ]
        assert df["label"].to_list() == ["h", "e"]
        assert df["start_seconds"].dtype == pl.Float64


def test_query_for_point_tier() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "events", "point")
        proj.add_point(tier, 0.25, label="click")
        proj.add_point(tier, 0.75, label="release")
        df = proj.query(tier)
        assert df.columns == [
            "id",
            "tier_id",
            "time_seconds",
            "label",
            "parent_annotation_id",
            "extra",
        ]
        assert df["time_seconds"].to_list() == [0.25, 0.75]


def test_query_for_reference_tier() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        speaker_id = proj.add_speaker("Alice")
        tier = proj.add_tier(bundle_id, "speaker_turns", "reference")
        proj.add_reference(tier, "speaker", speaker_id, label="alice")
        df = proj.query(tier)
        assert df.columns == [
            "id",
            "tier_id",
            "target_kind",
            "target_id",
            "label",
            "parent_annotation_id",
            "extra",
        ]
        assert df["target_kind"].to_list() == ["speaker"]
        assert df["target_id"].to_list() == [speaker_id]


def test_one_to_one_cardinality_violation_raises_value_error() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        words = proj.add_tier(bundle_id, "words", "interval")
        stress = proj.add_tier(
            bundle_id,
            "stress",
            "point",
            parent_id=words,
            cardinality="one_to_one",
        )
        w1 = proj.add_interval(words, 0.0, 1.0)
        proj.add_point(stress, 0.4, parent_annotation_id=w1)
        with pytest.raises(ValueError) as excinfo:
            proj.add_point(stress, 0.6, parent_annotation_id=w1)
        assert "one_to_one" in str(excinfo.value)


def test_missing_parent_annotation_id_raises_value_error() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        words = proj.add_tier(bundle_id, "words", "interval")
        phones = proj.add_tier(
            bundle_id,
            "phones",
            "interval",
            parent_id=words,
            cardinality="one_to_many",
        )
        with pytest.raises(ValueError):
            proj.add_interval(phones, 0.0, 0.1)


def test_many_to_one_is_deferred() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        parent = proj.add_tier(bundle_id, "parent", "interval")
        child = proj.add_tier(
            bundle_id,
            "child",
            "interval",
            parent_id=parent,
            cardinality="many_to_one",
        )
        w1 = proj.add_interval(parent, 0.0, 1.0)
        with pytest.raises(ValueError) as excinfo:
            proj.add_interval(child, 0.0, 0.5, parent_annotation_id=w1)
        assert "many_to_one" in str(excinfo.value)


def test_query_on_dense_tier_errors_with_b3_hint() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "f0", "continuous_numeric")
        with pytest.raises(ValueError) as excinfo:
            proj.query(tier)
        assert "B3" in str(excinfo.value)


def test_b2_surface_is_stable() -> None:
    from sadda._stability import get_stability

    for sym in (sadda.Tier, sadda.Interval, sadda.Point, sadda.Reference):
        assert get_stability(sym) == "stable", sym
