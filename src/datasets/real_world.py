"""Real-world time series dataset loaders with auto-extracted temporal property labels.

Loads ETTh1 (electricity transformer) and Jena Climate (weather) datasets, extracts
sliding windows, and automatically labels each window for temporal properties (trend,
stationarity, seasonality, change point) using statistical methods. Returns
SyntheticDataset objects for seamless integration with the existing probing pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.datasets.synthetic import SyntheticDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_CONFIGS: dict[str, dict[str, str | None]] = {
    "etth1": {
        "filename": "ETTh1.csv",
        "date_column": "date",
        "default_target": "OT",
        "format": None,
    },
    "weather": {
        "filename": "jena_climate_2009_2016.csv",
        "date_column": "Date Time",
        "default_target": "T (degC)",
        "format": None,
    },
    "electricity": {
        "filename": "LD2011_2014.txt",
        "date_column": None,
        "default_target": "MT_004",
        "format": "electricity",
    },
    "traffic": {
        "filename": "traffic.txt",
        "date_column": None,
        "default_target": None,
        "format": "plain_matrix",
    },
    "exchange_rate": {
        "filename": "exchange_rate.txt",
        "date_column": None,
        "default_target": None,
        "format": "plain_matrix",
    },
}

PROPERTY_NAMES = ["trend", "stationarity", "seasonality", "seasonality_binary", "change_point"]

_ADF_PVALUE_THRESHOLD: float = 0.05
_VARIANCE_RATIO_THRESHOLD: float = 2.0
_ACF_PEAK_THRESHOLD: float = 0.3
_CHANGEPOINT_STAT_THRESHOLD: float = 2.0
_MISSING_VALUE_FILLER: float = -9000.0
_MIN_PERIOD_LENGTH: float = 2.0


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_univariate_series(
    dataset_name: str,
    data_dir: str = "data",
    target_column: str | None = None,
) -> NDArray[np.float64]:
    """Load a univariate time series from CSV, normalized to zero-mean unit-variance.

    Args:
        dataset_name: One of the keys in DATASET_CONFIGS.
        data_dir: Directory containing CSV files (relative to project root).
        target_column: Column to extract. None uses the dataset default.

    Returns:
        1D float64 array of z-score normalized values.

    Raises:
        ValueError: If dataset_name is not recognized.
        FileNotFoundError: If CSV file does not exist.
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. Choose from {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    fmt = config.get("format")
    filename = config.get("filename")
    if not isinstance(filename, str):
        raise ValueError(f"Dataset {dataset_name!r} does not define a valid filename")
    csv_path = Path(data_dir) / filename

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}. Download it to the {data_dir}/ directory first."
        )

    if fmt == "plain_matrix":
        # Headerless comma-separated matrix (traffic, exchange_rate)
        data = np.loadtxt(str(csv_path), delimiter=",")
        col_idx = 0  # default: first column
        if target_column is not None:
            col_idx = int(target_column)
        series = data[:, col_idx].astype(np.float64)
    elif fmt == "electricity":
        # Semicolon-separated with decimal comma (LD2011_2014.txt)
        df = pd.read_csv(csv_path, sep=";", decimal=",")
        col = target_column if target_column is not None else config["default_target"]
        if col not in df.columns:
            raise ValueError(
                f"Column {col!r} not found in {csv_path.name}. Available: {list(df.columns)[:10]}"
            )
        series = df[col].to_numpy(dtype=np.float64)
    else:
        # Standard CSV (etth1, weather)
        df = pd.read_csv(csv_path)
        col = target_column if target_column is not None else config["default_target"]
        if col not in df.columns:
            raise ValueError(
                f"Column {col!r} not found in {csv_path.name}. Available: {list(df.columns)}"
            )
        series = df[col].to_numpy(dtype=np.float64)

    # Filter out sentinel values (-9999) used in Jena Climate
    valid_mask = series > _MISSING_VALUE_FILLER
    series = series[valid_mask]

    # Z-score normalization using train portion only (first 80%) to prevent leakage
    train_end = int(len(series) * 0.8)
    mean = series[:train_end].mean()
    std = series[:train_end].std()
    if std < 1e-8:
        raise ValueError(f"Target column has near-zero variance (std={std:.2e})")
    series = (series - mean) / std

    return series


def _extract_windows(
    series: NDArray[np.float64],
    seq_len: int = 512,
    stride: int = 256,
    num_samples: int | None = None,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Extract sliding windows from a 1D time series.

    Args:
        series: 1D float64 array.
        seq_len: Window length.
        stride: Step size between consecutive windows.
        num_samples: If provided, randomly subsample to this count.
        seed: Random seed for subsampling.

    Returns:
        2D array of shape (N, seq_len).

    Raises:
        ValueError: If series is shorter than seq_len.
    """
    if len(series) < seq_len:
        raise ValueError(f"Series length ({len(series)}) is shorter than seq_len ({seq_len})")

    num_windows = (len(series) - seq_len) // stride + 1
    windows = np.empty((num_windows, seq_len), dtype=np.float64)

    for i in range(num_windows):
        start = i * stride
        windows[i] = series[start : start + seq_len]

    if num_samples is not None and num_samples < num_windows:
        rng = np.random.default_rng(seed)
        indices = rng.choice(num_windows, size=num_samples, replace=False)
        indices.sort()
        windows = windows[indices]

    return windows


def temporal_train_test_split(
    windows: NDArray[np.float64],
    labels: NDArray[np.int64] | NDArray[np.float64],
    test_ratio: float = 0.2,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64] | NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64] | NDArray[np.float64],
]:
    """Split windows temporally (no shuffle) to avoid overlap leakage.

    Earlier windows go to train, later windows go to test.
    This prevents information leakage from overlapping sliding windows.

    Args:
        windows: 2D array (N, seq_len).
        labels: 1D array (N,).
        test_ratio: Fraction of windows for test set.

    Returns:
        (train_windows, train_labels, test_windows, test_labels)
    """
    n = len(windows)
    split_idx = int(n * (1 - test_ratio))
    return windows[:split_idx], labels[:split_idx], windows[split_idx:], labels[split_idx:]


# ---------------------------------------------------------------------------
# Auto-labeling functions
# ---------------------------------------------------------------------------


def _label_trend(window: NDArray[np.float64]) -> int:
    """Classify trend direction of a single window.

    Uses linear regression slope with adaptive threshold based on window variance.

    Args:
        window: 1D array of length seq_len.

    Returns:
        0 (upward), 1 (downward), or 2 (flat).
    """
    seq_len = len(window)
    t = np.arange(seq_len, dtype=np.float64)
    slope, _ = np.polyfit(t, window, 1)

    std = window.std()
    threshold = 0.5 * std / seq_len if std > 1e-8 else 1e-8

    if slope > threshold:
        return 0  # upward
    elif slope < -threshold:
        return 1  # downward
    else:
        return 2  # flat


def _label_stationarity(window: NDArray[np.float64]) -> int:
    """Classify stationarity of a single window.

    Tries statsmodels ADF test first; falls back to variance-ratio test.

    Args:
        window: 1D array of length seq_len.

    Returns:
        0 (stationary) or 1 (non-stationary).
    """
    try:
        adfuller = __import__("statsmodels.tsa.stattools", fromlist=["adfuller"]).adfuller
    except ImportError:
        raise ImportError(
            "statsmodels is required for ADF-based stationarity labeling. "
            "Install it: pip install statsmodels"
        ) from None

    result = adfuller(window, autolag="AIC")
    p_value = result[1]
    return 0 if p_value < _ADF_PVALUE_THRESHOLD else 1


def _label_seasonality(window: NDArray[np.float64]) -> float:
    """Extract dominant period of a single window via FFT.

    Args:
        window: 1D array of length seq_len.

    Returns:
        Dominant period as a float, clipped to [2, seq_len // 2].
    """
    seq_len = len(window)
    spectrum = np.abs(np.fft.rfft(window))
    # Exclude DC component (index 0) and very low frequencies (index 1)
    spectrum[0] = 0.0
    if len(spectrum) > 1:
        spectrum[1] = 0.0

    dominant_idx = int(np.argmax(spectrum))
    if dominant_idx == 0:
        dominant_idx = 2  # fallback to avoid division by zero

    period = seq_len / dominant_idx
    period = float(np.clip(period, _MIN_PERIOD_LENGTH, seq_len // 2))
    return period


def _label_seasonality_binary(window: NDArray[np.float64]) -> int:
    """Classify whether a window exhibits significant seasonality via ACF.

    Uses autocorrelation function peaks to detect periodic patterns,
    which is more robust than FFT for noisy real-world data.

    Args:
        window: 1D array of length seq_len.

    Returns:
        0 (no clear seasonality) or 1 (has seasonality).
    """
    seq_len = len(window)
    centered = window - window.mean()
    norm = np.sum(centered**2)
    if norm < 1e-12:
        return 0

    # Compute normalized autocorrelation for positive lags
    acf = np.correlate(centered, centered, mode="full")
    acf = acf[seq_len - 1 :] / norm

    # Search for peaks in ACF between lag 4 and seq_len//2
    min_lag = 4
    max_lag = seq_len // 2
    acf_segment = acf[min_lag:max_lag]

    if len(acf_segment) < 3:
        return 0

    # Find local maxima
    best_peak = 0.0
    for i in range(1, len(acf_segment) - 1):
        if acf_segment[i] > acf_segment[i - 1] and acf_segment[i] > acf_segment[i + 1]:
            if acf_segment[i] > best_peak:
                best_peak = float(acf_segment[i])

    return 1 if best_peak > _ACF_PEAK_THRESHOLD else 0


def _label_change_point(window: NDArray[np.float64]) -> int:
    """Detect presence of a distribution change point via mean-shift statistic.

    Evaluates candidate split points and checks if any produces a significant
    difference in segment means relative to pooled standard deviation.

    Args:
        window: 1D array of length seq_len.

    Returns:
        0 (no change point) or 1 (has change point).
    """
    seq_len = len(window)
    max_stat = 0.0

    for split in range(64, seq_len - 64, 32):
        left = window[:split]
        right = window[split:]

        mean_diff = abs(left.mean() - right.mean())
        pooled_var = (left.var() * len(left) + right.var() * len(right)) / seq_len
        pooled_std = np.sqrt(pooled_var) if pooled_var > 1e-10 else 1e-5

        stat = mean_diff / pooled_std
        if stat > max_stat:
            max_stat = stat

    return 1 if max_stat > _CHANGEPOINT_STAT_THRESHOLD else 0


# Mapping from property name to (labeling function, label_type)
_LABELERS: dict[str, tuple[Callable[[NDArray[np.float64]], int | float], str]] = {
    "trend": (_label_trend, "classification"),
    "stationarity": (_label_stationarity, "classification"),
    "seasonality": (_label_seasonality, "regression"),
    "seasonality_binary": (_label_seasonality_binary, "classification"),
    "change_point": (_label_change_point, "classification"),
}


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def load_real_world_dataset(
    dataset_name: str,
    property_name: str,
    *,
    data_dir: str = "data",
    seq_len: int = 512,
    stride: int = 256,
    num_samples: int | None = None,
    target_column: str | None = None,
    seed: int = 42,
) -> SyntheticDataset:
    """Load a real-world dataset and extract temporal property labels.

    Loads a CSV, extracts univariate sliding windows, and auto-labels each
    window for the requested temporal property.

    Args:
        dataset_name: One of the keys in DATASET_CONFIGS.
        property_name: One of "trend", "stationarity", "seasonality", "change_point".
        data_dir: Directory containing CSV files.
        seq_len: Window length.
        stride: Step size between consecutive windows.
        num_samples: If provided, randomly subsample to this count.
        target_column: Column to extract. None uses the dataset default.
        seed: Random seed for subsampling.

    Returns:
        SyntheticDataset with auto-extracted labels.

    Raises:
        ValueError: If dataset_name or property_name is not recognized.
    """
    if property_name not in _LABELERS:
        raise ValueError(f"Unknown property: {property_name!r}. Choose from {PROPERTY_NAMES}")

    series = _load_univariate_series(dataset_name, data_dir, target_column)
    windows = _extract_windows(series, seq_len, stride, num_samples, seed)

    labeler_fn, label_type = _LABELERS[property_name]
    raw_labels = [labeler_fn(w) for w in windows]

    if label_type == "classification":
        labels = np.array(raw_labels, dtype=np.int64)
    else:
        labels = np.array(raw_labels, dtype=np.float64)

    col_name = target_column or DATASET_CONFIGS[dataset_name]["default_target"]

    return SyntheticDataset(
        sequences=windows,
        labels=labels,
        label_type=label_type,
        property_name=property_name,
        metadata={
            "dataset": dataset_name,
            "source": "real_world",
            "target_column": col_name,
            "seq_len": seq_len,
            "stride": stride,
            "num_samples": len(windows),
            "total_series_length": len(series),
            "seed": seed,
        },
    )


# ---------------------------------------------------------------------------
# Convenience wrappers matching synthetic generator naming convention
# ---------------------------------------------------------------------------


def generate_etth1_trend_dataset(
    num_samples: int | None = None,
    seq_len: int = 512,
    *,
    stride: int = 256,
    seed: int = 42,
) -> SyntheticDataset:
    """Load ETTh1 and label windows by trend direction."""
    return load_real_world_dataset(
        "etth1",
        "trend",
        seq_len=seq_len,
        stride=stride,
        num_samples=num_samples,
        seed=seed,
    )


def generate_etth1_stationarity_dataset(
    num_samples: int | None = None,
    seq_len: int = 512,
    *,
    stride: int = 256,
    seed: int = 42,
) -> SyntheticDataset:
    """Load ETTh1 and label windows by stationarity."""
    return load_real_world_dataset(
        "etth1",
        "stationarity",
        seq_len=seq_len,
        stride=stride,
        num_samples=num_samples,
        seed=seed,
    )


def generate_etth1_seasonality_dataset(
    num_samples: int | None = None,
    seq_len: int = 512,
    *,
    stride: int = 256,
    seed: int = 42,
) -> SyntheticDataset:
    """Load ETTh1 and label windows by dominant seasonality period."""
    return load_real_world_dataset(
        "etth1",
        "seasonality",
        seq_len=seq_len,
        stride=stride,
        num_samples=num_samples,
        seed=seed,
    )


def generate_etth1_change_point_dataset(
    num_samples: int | None = None,
    seq_len: int = 512,
    *,
    stride: int = 256,
    seed: int = 42,
) -> SyntheticDataset:
    """Load ETTh1 and label windows by change point presence."""
    return load_real_world_dataset(
        "etth1",
        "change_point",
        seq_len=seq_len,
        stride=stride,
        num_samples=num_samples,
        seed=seed,
    )


def generate_weather_trend_dataset(
    num_samples: int | None = None,
    seq_len: int = 512,
    *,
    stride: int = 256,
    seed: int = 42,
) -> SyntheticDataset:
    """Load Jena Climate data and label windows by trend direction."""
    return load_real_world_dataset(
        "weather",
        "trend",
        seq_len=seq_len,
        stride=stride,
        num_samples=num_samples,
        seed=seed,
    )


def generate_weather_stationarity_dataset(
    num_samples: int | None = None,
    seq_len: int = 512,
    *,
    stride: int = 256,
    seed: int = 42,
) -> SyntheticDataset:
    """Load Jena Climate data and label windows by stationarity."""
    return load_real_world_dataset(
        "weather",
        "stationarity",
        seq_len=seq_len,
        stride=stride,
        num_samples=num_samples,
        seed=seed,
    )


def generate_weather_seasonality_dataset(
    num_samples: int | None = None,
    seq_len: int = 512,
    *,
    stride: int = 256,
    seed: int = 42,
) -> SyntheticDataset:
    """Load Jena Climate data and label windows by dominant seasonality period."""
    return load_real_world_dataset(
        "weather",
        "seasonality",
        seq_len=seq_len,
        stride=stride,
        num_samples=num_samples,
        seed=seed,
    )


def generate_weather_change_point_dataset(
    num_samples: int | None = None,
    seq_len: int = 512,
    *,
    stride: int = 256,
    seed: int = 42,
) -> SyntheticDataset:
    """Load Jena Climate data and label windows by change point presence."""
    return load_real_world_dataset(
        "weather",
        "change_point",
        seq_len=seq_len,
        stride=stride,
        num_samples=num_samples,
        seed=seed,
    )
