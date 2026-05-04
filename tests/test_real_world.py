from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from src.datasets.real_world import (
    _extract_windows,
    _label_change_point,
    _label_seasonality,
    _label_seasonality_binary,
    _label_stationarity,
    _label_trend,
    load_real_world_dataset,
)


def test_extract_windows_known_array_shape_and_values() -> None:
    series = np.arange(10, dtype=np.float64)
    windows = _extract_windows(series, seq_len=4, stride=2)
    expected = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )
    assert windows.shape == (4, 4)
    assert np.array_equal(windows, expected)


def test_extract_windows_with_num_samples_subsamples_correctly() -> None:
    series = np.arange(10, dtype=np.float64)
    full_windows = _extract_windows(series, seq_len=4, stride=1)
    sampled_1 = _extract_windows(series, seq_len=4, stride=1, num_samples=3, seed=123)
    sampled_2 = _extract_windows(series, seq_len=4, stride=1, num_samples=3, seed=123)

    assert sampled_1.shape == (3, 4)
    assert np.array_equal(sampled_1, sampled_2)
    assert all(any(np.array_equal(row, w) for w in full_windows) for row in sampled_1)


def test_extract_windows_raises_on_short_series() -> None:
    series = np.arange(5, dtype=np.float64)
    with pytest.raises(ValueError):
        _ = _extract_windows(series, seq_len=6, stride=1)


def test_label_trend_upward_returns_zero() -> None:
    window = np.linspace(0.0, 10.0, 100, dtype=np.float64)
    assert _label_trend(window) == 0


def test_label_trend_downward_returns_one() -> None:
    window = np.linspace(10.0, 0.0, 100, dtype=np.float64)
    assert _label_trend(window) == 1


def test_label_trend_flat_returns_two() -> None:
    window = np.ones(100, dtype=np.float64)
    assert _label_trend(window) == 2


def test_label_seasonality_known_sinusoid_period() -> None:
    seq_len = 120
    period = 12.0
    t = np.arange(seq_len, dtype=np.float64)
    window = np.sin(2.0 * np.pi * t / period)
    estimated = _label_seasonality(window)
    assert estimated == pytest.approx(period, rel=1e-6)


def test_label_seasonality_binary_periodic_signal_returns_one() -> None:
    seq_len = 256
    period = 24.0
    t = np.arange(seq_len, dtype=np.float64)
    window = np.sin(2.0 * np.pi * t / period)
    assert _label_seasonality_binary(window) == 1


def test_label_seasonality_binary_random_noise_returns_zero() -> None:
    rng = np.random.default_rng(42)
    window = rng.standard_normal(512)
    assert _label_seasonality_binary(window.astype(np.float64)) == 0


def test_label_stationarity_random_noise_returns_zero() -> None:
    """Stationarity labeling requires statsmodels; random noise should be stationary."""
    rng = np.random.default_rng(123)
    window = rng.standard_normal(512).astype(np.float64)
    assert _label_stationarity(window) == 0


def test_label_stationarity_raises_without_statsmodels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without statsmodels, _label_stationarity must raise ImportError."""
    original_import = builtins.__import__

    def fake_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "statsmodels.tsa.stattools":
            raise ImportError("forced for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    rng = np.random.default_rng(123)
    window = rng.standard_normal(512).astype(np.float64)
    with pytest.raises(ImportError, match="statsmodels is required"):
        _label_stationarity(window)


def test_label_change_point_no_change_returns_zero() -> None:
    window = np.zeros(256, dtype=np.float64)
    assert _label_change_point(window) == 0


def test_label_change_point_mean_shift_returns_one() -> None:
    left = np.zeros(128, dtype=np.float64)
    right = np.full(128, 5.0, dtype=np.float64)
    window = np.concatenate([left, right])
    assert _label_change_point(window) == 1


def test_load_real_world_dataset_invalid_dataset_name_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _ = load_real_world_dataset("unknown_dataset", "trend", data_dir=str(tmp_path))


def test_load_real_world_dataset_invalid_property_name_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _ = load_real_world_dataset("etth1", "unknown_property")


def test_load_real_world_dataset_nonexistent_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _ = load_real_world_dataset("etth1", "trend", data_dir=str(tmp_path))
