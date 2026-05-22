"""Python-surface tests for the B3 dense-tier API:
write/read round-trips through NumPy + the polars query dispatch + the
`pl.scan_parquet(proj.dense_path(tier_id))` external-reader path."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
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


def test_continuous_numeric_numpy_round_trip() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "f0", "continuous_numeric")
        samples = np.linspace(80.0, 220.0, 100, dtype=np.float64)
        ds_id = proj.write_continuous_numeric(tier, samples, sample_rate_hz=100.0)
        ds = proj.derived_signal(tier)
        assert ds is not None
        assert ds.id == ds_id
        assert ds.n_frames == 100
        assert ds.n_dims == 1
        assert ds.dtype == "f64"
        assert ds.sample_rate_hz == 100.0
        back = proj.read_continuous_numeric(tier)
        assert isinstance(back, np.ndarray)
        assert back.dtype == np.float64
        np.testing.assert_allclose(back, samples)


def test_continuous_vector_numpy_round_trip() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "wav2vec", "continuous_vector")
        frames = np.random.default_rng(0).standard_normal((20, 4))
        proj.write_continuous_vector(tier, frames, sample_rate_hz=50.0)
        back = proj.read_continuous_vector(tier)
        assert back.shape == (20, 4)
        np.testing.assert_allclose(back, frames)


def test_categorical_sampled_round_trip() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "vad", "categorical_sampled")
        labels = ["speech", "speech", "silence", "speech", "silence"]
        proj.write_categorical_sampled(tier, labels, sample_rate_hz=10.0)
        back = proj.read_categorical_sampled(tier)
        assert back == labels


def test_query_returns_polars_dataframe_for_dense_tier() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "f0", "continuous_numeric")
        samples = np.array([100.0, 110.0, 120.0])
        proj.write_continuous_numeric(tier, samples, sample_rate_hz=100.0)
        df = proj.query(tier)
        assert isinstance(df, pl.DataFrame)
        # write_continuous_numeric writes a single `value` column
        assert df.columns == ["value"]
        assert df["value"].to_list() == [100.0, 110.0, 120.0]


def test_query_for_dense_without_sidecar_raises_helpful_error() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "f0", "continuous_numeric")
        with pytest.raises(ValueError) as excinfo:
            proj.query(tier)
        msg = str(excinfo.value)
        assert "write_continuous_numeric" in msg


def test_dense_path_works_with_polars_scan_parquet() -> None:
    """AI-engineer path: skip the engine API entirely and use polars
    directly on the sidecar."""
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "wav2vec", "continuous_vector")
        frames = np.arange(40.0).reshape(10, 4)
        proj.write_continuous_vector(tier, frames, sample_rate_hz=50.0)
        path = proj.dense_path(tier)
        assert path is not None
        # External read via pure polars — no engine API call.
        df = pl.scan_parquet(path).collect()
        assert df.height == 10  # n_frames


def test_writing_wrong_type_errors_at_python_layer() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        cn = proj.add_tier(bundle_id, "f0", "continuous_numeric")
        with pytest.raises(RuntimeError):
            proj.write_continuous_vector(cn, np.zeros((4, 3)), sample_rate_hz=100.0)


def test_double_write_rejected() -> None:
    with tempfile.TemporaryDirectory() as td:
        proj, bundle_id = _project_with_bundle(Path(td))
        tier = proj.add_tier(bundle_id, "f0", "continuous_numeric")
        proj.write_continuous_numeric(tier, np.array([1.0, 2.0]), sample_rate_hz=100.0)
        with pytest.raises(RuntimeError):
            proj.write_continuous_numeric(tier, np.array([3.0, 4.0]), sample_rate_hz=100.0)


def test_b3_surface_is_stable() -> None:
    from sadda._stability import get_stability

    assert get_stability(sadda.DerivedSignal) == "stable"
