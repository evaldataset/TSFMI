"""Controlled synthetic time series generators with exact ground-truth labels.

Each generator produces sequences with mathematically exact labels derived from
generation parameters — NOT from statistical estimation. This follows the approach
of Wiliński et al. (2024, arXiv:2409.12915) for probing foundation model representations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SyntheticDataset:
    """Container for synthetic time series data with ground-truth labels.

    Attributes:
        sequences: Time series array of shape (N, seq_len).
        labels: Ground-truth labels — (N,) int for classification, (N,) float for regression.
        label_type: Either "classification" or "regression".
        property_name: Temporal property being probed (e.g. "trend", "seasonality").
        metadata: Generation parameters for full reproducibility.
    """

    sequences: NDArray[np.float64]
    labels: NDArray[np.int64] | NDArray[np.float64]
    label_type: str
    property_name: str
    metadata: dict[str, object]


def generate_trend_dataset(
    num_samples: int = 1000,
    seq_len: int = 512,
    *,
    noise_std: float = 0.1,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate time series classified by trend direction.

    Classes:
        0 = upward (positive slope), 1 = downward (negative slope), 2 = flat (no trend).

    Args:
        num_samples: Total number of sequences (split equally among 3 classes).
        seq_len: Length of each time series.
        noise_std: Standard deviation of additive Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticDataset with classification labels in {0, 1, 2}.
    """
    rng = np.random.default_rng(seed)
    samples_per_class = num_samples // 3
    t = np.arange(seq_len, dtype=np.float64)

    sequences = []
    labels = []

    # Class 0: upward trend
    for _ in range(samples_per_class):
        slope = rng.uniform(0.01, 0.1)
        noise = rng.normal(0, noise_std, size=seq_len)
        y = slope * t + noise
        y -= y.mean()
        sequences.append(y)
        labels.append(0)

    # Class 1: downward trend
    for _ in range(samples_per_class):
        slope = rng.uniform(0.01, 0.1)
        noise = rng.normal(0, noise_std, size=seq_len)
        y = -slope * t + noise
        y -= y.mean()
        sequences.append(y)
        labels.append(1)

    # Class 2: flat (pure noise)
    remaining = num_samples - 2 * samples_per_class
    for _ in range(remaining):
        noise = rng.normal(0, noise_std, size=seq_len)
        y = noise.copy()
        y -= y.mean()
        sequences.append(y)
        labels.append(2)

    return SyntheticDataset(
        sequences=np.array(sequences),
        labels=np.array(labels, dtype=np.int64),
        label_type="classification",
        property_name="trend",
        metadata={
            "num_samples": num_samples,
            "seq_len": seq_len,
            "noise_std": noise_std,
            "seed": seed,
            "classes": {0: "up", 1: "down", 2: "flat"},
            "slope_range": (0.01, 0.1),
        },
    )


def generate_seasonality_dataset(
    num_samples: int = 1000,
    seq_len: int = 512,
    *,
    periods: list[int] | None = None,
    noise_std: float = 0.1,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate sinusoidal time series with regression target = period length.

    Each sequence is a sine wave with a randomly chosen period from `periods`,
    random amplitude, and additive noise.

    Args:
        num_samples: Total number of sequences.
        seq_len: Length of each time series.
        periods: List of candidate period lengths. Defaults to [8, 16, 32, 64].
        noise_std: Standard deviation of additive Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticDataset with float regression labels (period values).
    """
    if periods is None:
        periods = [8, 16, 32, 64]

    rng = np.random.default_rng(seed)
    t = np.arange(seq_len, dtype=np.float64)

    sequences = np.empty((num_samples, seq_len), dtype=np.float64)
    labels = np.empty(num_samples, dtype=np.float64)

    for i in range(num_samples):
        period = periods[rng.integers(len(periods))]
        amplitude = rng.uniform(0.5, 2.0)
        noise = rng.normal(0, noise_std, size=seq_len)
        sequences[i] = amplitude * np.sin(2 * np.pi * t / period) + noise
        labels[i] = float(period)

    return SyntheticDataset(
        sequences=sequences,
        labels=labels,
        label_type="regression",
        property_name="seasonality",
        metadata={
            "num_samples": num_samples,
            "seq_len": seq_len,
            "periods": periods,
            "noise_std": noise_std,
            "seed": seed,
            "amplitude_range": (0.5, 2.0),
        },
    )


def generate_frequency_dataset(
    num_samples: int = 1000,
    seq_len: int = 512,
    *,
    freq_bins: int = 8,
    noise_std: float = 0.05,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate sinusoidal time series classified by dominant frequency band.

    Frequency band k has frequency f_k = (k+1) / seq_len cycles per sample.
    Each sequence is a sine at one of the `freq_bins` frequencies plus noise.

    Args:
        num_samples: Total number of sequences.
        seq_len: Length of each time series.
        freq_bins: Number of discrete frequency bands (classes).
        noise_std: Standard deviation of additive Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticDataset with classification labels in {0, ..., freq_bins-1}.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(seq_len, dtype=np.float64)

    sequences = np.empty((num_samples, seq_len), dtype=np.float64)
    labels = np.empty(num_samples, dtype=np.int64)

    for i in range(num_samples):
        k = rng.integers(freq_bins)
        f_k = (k + 1) / seq_len
        amplitude = rng.uniform(0.5, 2.0)
        noise = rng.normal(0, noise_std, size=seq_len)
        sequences[i] = amplitude * np.sin(2 * np.pi * f_k * t) + noise
        labels[i] = k

    return SyntheticDataset(
        sequences=sequences,
        labels=labels,
        label_type="classification",
        property_name="frequency",
        metadata={
            "num_samples": num_samples,
            "seq_len": seq_len,
            "freq_bins": freq_bins,
            "noise_std": noise_std,
            "seed": seed,
            "frequencies": [(k + 1) / seq_len for k in range(freq_bins)],
        },
    )


def generate_stationarity_dataset(
    num_samples: int = 1000,
    seq_len: int = 512,
    *,
    noise_std: float = 0.1,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate time series classified as stationary or non-stationary.

    Stationary sequences are zero-mean Gaussian noise. Non-stationary sequences
    add a random walk component (cumulative sum of noise) to create drift.

    Args:
        num_samples: Total number of sequences.
        seq_len: Length of each time series.
        noise_std: Standard deviation of the base Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticDataset with binary classification labels {0=stationary, 1=non-stationary}.
    """
    rng = np.random.default_rng(seed)
    half = num_samples // 2

    sequences = []
    labels = []

    # Class 0: stationary — pure Gaussian noise
    for _ in range(half):
        y = rng.normal(0, noise_std, size=seq_len)
        sequences.append(y)
        labels.append(0)

    # Class 1: non-stationary — Gaussian noise + random walk drift
    remaining = num_samples - half
    for _ in range(remaining):
        noise = rng.normal(0, noise_std, size=seq_len)
        walk = np.cumsum(rng.normal(0, noise_std, size=seq_len))
        y = noise + walk
        sequences.append(y)
        labels.append(1)

    return SyntheticDataset(
        sequences=np.array(sequences),
        labels=np.array(labels, dtype=np.int64),
        label_type="classification",
        property_name="stationarity",
        metadata={
            "num_samples": num_samples,
            "seq_len": seq_len,
            "noise_std": noise_std,
            "seed": seed,
            "classes": {0: "stationary", 1: "non-stationary"},
        },
    )


def generate_anomaly_dataset(
    num_samples: int = 1000,
    seq_len: int = 512,
    *,
    anomaly_fraction: float = 0.5,
    anomaly_magnitude: float = 5.0,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate time series classified by presence of a point anomaly.

    Normal sequences are standard Gaussian noise. Anomaly sequences have a single
    spike of `anomaly_magnitude` at a random position.

    Args:
        num_samples: Total number of sequences.
        seq_len: Length of each time series.
        anomaly_fraction: Fraction of sequences that contain an anomaly.
        anomaly_magnitude: Absolute magnitude of the anomaly spike.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticDataset with binary classification labels {0=normal, 1=anomaly}.
    """
    rng = np.random.default_rng(seed)
    num_anomaly = int(num_samples * anomaly_fraction)
    num_normal = num_samples - num_anomaly

    sequences = []
    labels = []

    # Class 0: normal — standard Gaussian noise
    for _ in range(num_normal):
        y = rng.standard_normal(seq_len)
        sequences.append(y)
        labels.append(0)

    # Class 1: anomaly — Gaussian noise + single spike
    for _ in range(num_anomaly):
        y = rng.standard_normal(seq_len)
        spike_pos = rng.integers(seq_len)
        spike_sign = rng.choice([-1.0, 1.0])
        y[spike_pos] += spike_sign * anomaly_magnitude
        sequences.append(y)
        labels.append(1)

    return SyntheticDataset(
        sequences=np.array(sequences),
        labels=np.array(labels, dtype=np.int64),
        label_type="classification",
        property_name="anomaly",
        metadata={
            "num_samples": num_samples,
            "seq_len": seq_len,
            "anomaly_fraction": anomaly_fraction,
            "anomaly_magnitude": anomaly_magnitude,
            "seed": seed,
            "classes": {0: "normal", 1: "anomaly"},
        },
    )


def generate_change_point_dataset(
    num_samples: int = 1000,
    seq_len: int = 512,
    *,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate time series classified by presence of a distribution change point.

    No-CP sequences are stationary N(0,1) noise. CP sequences have the first half
    drawn from N(0,1) and the second half from N(mu_shift, sigma_shift) where
    mu_shift ~ U(2,4) and sigma_shift ~ U(2,4).

    Args:
        num_samples: Total number of sequences.
        seq_len: Length of each time series.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticDataset with binary classification labels {0=no_cp, 1=has_cp}.
    """
    rng = np.random.default_rng(seed)
    half = num_samples // 2
    midpoint = seq_len // 2

    sequences = []
    labels = []

    # Class 0: no change point — stationary N(0,1)
    for _ in range(half):
        y = rng.standard_normal(seq_len)
        sequences.append(y)
        labels.append(0)

    # Class 1: change point at midpoint
    remaining = num_samples - half
    for _ in range(remaining):
        mu_shift = rng.uniform(2.0, 4.0)
        sigma_shift = rng.uniform(2.0, 4.0)
        first_half = rng.standard_normal(midpoint)
        second_half = rng.normal(mu_shift, sigma_shift, size=seq_len - midpoint)
        y = np.concatenate([first_half, second_half])
        sequences.append(y)
        labels.append(1)

    return SyntheticDataset(
        sequences=np.array(sequences),
        labels=np.array(labels, dtype=np.int64),
        label_type="classification",
        property_name="change_point",
        metadata={
            "num_samples": num_samples,
            "seq_len": seq_len,
            "seed": seed,
            "change_point_position": midpoint,
            "mu_shift_range": (2.0, 4.0),
            "sigma_shift_range": (2.0, 4.0),
            "classes": {0: "no_change_point", 1: "has_change_point"},
        },
    )


def generate_trend_hard_dataset(
    num_samples: int = 1000,
    seq_len: int = 512,
    *,
    noise_std: float = 1.0,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate harder trend classification with low signal-to-noise ratio.

    Compared to the standard trend generator, this version:
    - Uses slopes much closer to zero (0.001-0.01) relative to high noise (std=1.0)
    - Adds confounding seasonality to all classes
    - Adds slow random-walk drift to flat class to create near-boundary samples

    Classes:
        0 = upward, 1 = downward, 2 = flat.

    Args:
        num_samples: Total number of sequences (split equally among 3 classes).
        seq_len: Length of each time series.
        noise_std: Standard deviation of additive Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticDataset with classification labels in {0, 1, 2}.
    """
    rng = np.random.default_rng(seed)
    samples_per_class = num_samples // 3
    t = np.arange(seq_len, dtype=np.float64)

    sequences: list[NDArray[np.float64]] = []
    labels: list[int] = []

    def _add_confounders(y: NDArray[np.float64]) -> NDArray[np.float64]:
        period = int(rng.choice([16, 32, 64, 128]))
        amplitude = rng.uniform(0.3, 1.5)
        phase = rng.uniform(0, 2 * np.pi)
        seasonal = amplitude * np.sin(2 * np.pi * t / period + phase)
        noise = rng.normal(0, noise_std, size=seq_len)
        return y + seasonal + noise

    # Class 0: upward trend (weak slopes buried in noise)
    for _ in range(samples_per_class):
        slope = rng.uniform(0.001, 0.01)
        y = _add_confounders(slope * t)
        y -= y.mean()
        sequences.append(y)
        labels.append(0)

    # Class 1: downward trend (weak slopes buried in noise)
    for _ in range(samples_per_class):
        slope = rng.uniform(0.001, 0.01)
        y = _add_confounders(-slope * t)
        y -= y.mean()
        sequences.append(y)
        labels.append(1)

    # Class 2: flat with slight random-walk drift (near-boundary)
    remaining = num_samples - 2 * samples_per_class
    for _ in range(remaining):
        drift = np.cumsum(rng.normal(0, 0.001, size=seq_len))
        y = _add_confounders(drift)
        y -= y.mean()
        sequences.append(y)
        labels.append(2)

    return SyntheticDataset(
        sequences=np.array(sequences),
        labels=np.array(labels, dtype=np.int64),
        label_type="classification",
        property_name="trend",
        metadata={
            "num_samples": num_samples,
            "seq_len": seq_len,
            "noise_std": noise_std,
            "seed": seed,
            "difficulty": "hard",
            "classes": {0: "up", 1: "down", 2: "flat"},
            "slope_range": (0.001, 0.01),
        },
    )


def generate_frequency_hard_dataset(
    num_samples: int = 1000,
    seq_len: int = 512,
    *,
    freq_bins: int = 8,
    noise_std: float = 0.5,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate harder frequency classification with overlapping harmonics.

    Compared to the standard frequency generator, this version:
    - Uses closely-spaced frequency bins (1-8 cycles, adjacent bins differ by 1 cycle)
    - Adds random harmonic overtones at 30-70% of fundamental amplitude
    - Uses much higher noise (std=0.5 vs 0.05)
    - Adds random amplitude modulation across the sequence

    Args:
        num_samples: Total number of sequences.
        seq_len: Length of each time series.
        freq_bins: Number of discrete frequency bands (classes).
        noise_std: Standard deviation of additive Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticDataset with classification labels in {0, ..., freq_bins-1}.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(seq_len, dtype=np.float64)

    sequences = np.empty((num_samples, seq_len), dtype=np.float64)
    labels = np.empty(num_samples, dtype=np.int64)

    for i in range(num_samples):
        k = rng.integers(freq_bins)
        f_k = (k + 1) / seq_len
        amplitude = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 2 * np.pi)

        # Fundamental
        y = amplitude * np.sin(2 * np.pi * f_k * t + phase)

        # Add 1-2 harmonic overtones with reduced amplitude
        num_harmonics = int(rng.integers(1, 3))
        for h in range(1, num_harmonics + 1):
            harm_amp = amplitude * rng.uniform(0.3, 0.7)
            harm_phase = rng.uniform(0, 2 * np.pi)
            y += harm_amp * np.sin(2 * np.pi * f_k * (h + 1) * t + harm_phase)

        # Amplitude modulation (slow envelope)
        mod_period = rng.uniform(seq_len / 4, seq_len)
        mod_depth = rng.uniform(0.1, 0.4)
        envelope = 1.0 + mod_depth * np.sin(2 * np.pi * t / mod_period)
        y *= envelope

        # Noise
        noise = rng.normal(0, noise_std, size=seq_len)
        sequences[i] = y + noise
        labels[i] = k

    return SyntheticDataset(
        sequences=sequences,
        labels=labels,
        label_type="classification",
        property_name="frequency",
        metadata={
            "num_samples": num_samples,
            "seq_len": seq_len,
            "freq_bins": freq_bins,
            "noise_std": noise_std,
            "seed": seed,
            "difficulty": "hard",
            "frequencies": [(k + 1) / seq_len for k in range(freq_bins)],
        },
    )


def generate_anomaly_hard_dataset(
    num_samples: int = 1000,
    seq_len: int = 512,
    *,
    anomaly_fraction: float = 0.5,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate harder anomaly detection with subtle, varied anomaly types.

    Compared to the standard anomaly generator, this version:
    - Uses smaller anomaly magnitudes (2-4 sigma instead of 5 sigma)
    - Includes multiple anomaly types: point spike, level shift, variance change
    - Adds structured background (seasonality) instead of pure noise

    Args:
        num_samples: Total number of sequences.
        seq_len: Length of each time series.
        anomaly_fraction: Fraction of sequences that contain an anomaly.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticDataset with binary classification labels {0=normal, 1=anomaly}.
    """
    rng = np.random.default_rng(seed)
    num_anomaly = int(num_samples * anomaly_fraction)
    num_normal = num_samples - num_anomaly
    t = np.arange(seq_len, dtype=np.float64)

    sequences: list[NDArray[np.float64]] = []
    labels: list[int] = []

    def _make_background() -> NDArray[np.float64]:
        period = int(rng.choice([16, 32, 64]))
        amplitude = rng.uniform(0.5, 1.5)
        phase = rng.uniform(0, 2 * np.pi)
        seasonal = amplitude * np.sin(2 * np.pi * t / period + phase)
        return seasonal + rng.normal(0, 0.3, size=seq_len)

    # Class 0: normal — structured background
    for _ in range(num_normal):
        sequences.append(_make_background())
        labels.append(0)

    # Class 1: anomaly — various subtle anomaly types in structured background
    for _ in range(num_anomaly):
        y = _make_background()
        anomaly_type = int(rng.integers(3))

        if anomaly_type == 0:
            # Point anomaly with small magnitude (2-4 sigma)
            pos = int(rng.integers(seq_len))
            y[pos] += rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 4.0)
        elif anomaly_type == 1:
            # Short level shift (5-20 steps)
            shift_len = int(rng.integers(5, 21))
            start = int(rng.integers(0, seq_len - shift_len))
            y[start : start + shift_len] += rng.uniform(1.5, 3.0) * rng.choice([-1.0, 1.0])
        else:
            # Variance change in a segment
            seg_len = int(rng.integers(20, 60))
            start = int(rng.integers(0, seq_len - seg_len))
            y[start : start + seg_len] *= rng.uniform(2.0, 4.0)

        sequences.append(y)
        labels.append(1)

    return SyntheticDataset(
        sequences=np.array(sequences),
        labels=np.array(labels, dtype=np.int64),
        label_type="classification",
        property_name="anomaly",
        metadata={
            "num_samples": num_samples,
            "seq_len": seq_len,
            "anomaly_fraction": anomaly_fraction,
            "seed": seed,
            "difficulty": "hard",
            "anomaly_types": ["point", "level_shift", "variance_change"],
            "classes": {0: "normal", 1: "anomaly"},
        },
    )


def generate_stationarity_hard_dataset(
    num_samples: int = 1000,
    seq_len: int = 512,
    *,
    noise_std: float = 0.1,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate harder stationarity classification with subtle variance drift.

    Compared to the standard stationarity generator, this version:
    - Uses gradual variance changes instead of obvious random walks
    - Adds confounding seasonality to both classes
    - Non-stationary class has slow variance drift and mild mean drift

    Classes:
        0 = stationary, 1 = non-stationary.

    Args:
        num_samples: Total number of sequences (split equally).
        seq_len: Length of each time series.
        noise_std: Standard deviation of the base Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticDataset with binary classification labels.
    """
    rng = np.random.default_rng(seed)
    half = num_samples // 2
    t = np.arange(seq_len, dtype=np.float64)

    sequences: list[NDArray[np.float64]] = []
    labels: list[int] = []

    def _add_seasonality(y: NDArray[np.float64]) -> NDArray[np.float64]:
        period = int(rng.choice([16, 32, 64, 128]))
        amplitude = rng.uniform(0.3, 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        return y + amplitude * np.sin(2 * np.pi * t / period + phase)

    # Class 0: stationary — Gaussian noise + seasonality
    for _ in range(half):
        y = rng.normal(0, noise_std, size=seq_len)
        y = _add_seasonality(y)
        sequences.append(y)
        labels.append(0)

    # Class 1: non-stationary — slow variance drift + seasonality
    remaining = num_samples - half
    for _ in range(remaining):
        drift_factor = rng.uniform(1.5, 3.0)
        direction = rng.choice([-1.0, 1.0])
        if direction > 0:
            var_envelope = np.linspace(1.0, drift_factor, seq_len)
        else:
            var_envelope = np.linspace(drift_factor, 1.0, seq_len)

        noise = rng.normal(0, noise_std, size=seq_len) * var_envelope
        mean_drift = rng.uniform(0.001, 0.005) * direction
        y = noise + mean_drift * t
        y = _add_seasonality(y)
        sequences.append(y)
        labels.append(1)

    return SyntheticDataset(
        sequences=np.array(sequences),
        labels=np.array(labels, dtype=np.int64),
        label_type="classification",
        property_name="stationarity",
        metadata={
            "num_samples": num_samples,
            "seq_len": seq_len,
            "noise_std": noise_std,
            "seed": seed,
            "difficulty": "hard",
            "classes": {0: "stationary", 1: "non-stationary"},
        },
    )


def generate_change_point_hard_dataset(
    num_samples: int = 1000,
    seq_len: int = 512,
    *,
    seed: int = 42,
) -> SyntheticDataset:
    """Generate harder change point detection with subtle distribution shifts.

    Compared to the standard change point generator, this version:
    - Much smaller mean shifts (0.3-1.0 instead of 2-4)
    - Change point position varies (25%-75%, not always at midpoint)
    - Adds confounding seasonality to both classes
    - Mild sigma shifts (1.2-2.0 instead of 2-4)

    Classes:
        0 = no change point, 1 = has change point.

    Args:
        num_samples: Total number of sequences (split equally).
        seq_len: Length of each time series.
        seed: Random seed for reproducibility.

    Returns:
        SyntheticDataset with binary classification labels.
    """
    rng = np.random.default_rng(seed)
    half = num_samples // 2
    t = np.arange(seq_len, dtype=np.float64)

    sequences: list[NDArray[np.float64]] = []
    labels: list[int] = []

    def _add_seasonality(y: NDArray[np.float64]) -> NDArray[np.float64]:
        period = int(rng.choice([16, 32, 64]))
        amplitude = rng.uniform(0.3, 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        return y + amplitude * np.sin(2 * np.pi * t / period + phase)

    # Class 0: no change point — stationary with seasonality
    for _ in range(half):
        y = rng.standard_normal(seq_len)
        y = _add_seasonality(y)
        sequences.append(y)
        labels.append(0)

    # Class 1: subtle change point at varying positions
    remaining = num_samples - half
    for _ in range(remaining):
        cp_pos = int(rng.integers(seq_len // 4, 3 * seq_len // 4))
        mu_shift = rng.uniform(0.3, 1.0) * rng.choice([-1.0, 1.0])
        sigma_shift = rng.uniform(1.2, 2.0)

        first_part = rng.standard_normal(cp_pos)
        second_part = rng.normal(mu_shift, sigma_shift, size=seq_len - cp_pos)
        y = np.concatenate([first_part, second_part])
        y = _add_seasonality(y)
        sequences.append(y)
        labels.append(1)

    return SyntheticDataset(
        sequences=np.array(sequences),
        labels=np.array(labels, dtype=np.int64),
        label_type="classification",
        property_name="change_point",
        metadata={
            "num_samples": num_samples,
            "seq_len": seq_len,
            "seed": seed,
            "difficulty": "hard",
            "cp_position_range": (0.25, 0.75),
            "mu_shift_range": (0.3, 1.0),
            "sigma_shift_range": (1.2, 2.0),
            "classes": {0: "no_change_point", 1: "has_change_point"},
        },
    )
