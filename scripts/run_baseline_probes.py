"""Baseline probing experiments: raw-input and hand-crafted feature baselines.

Establishes the "floor" that model representations must beat to demonstrate
that pre-training adds value beyond trivial signal separability.

Baselines:
  1. Raw-input: Linear probe on raw time series (flattened or mean/var summary)
  2. Hand-crafted features: Linear probe on statistical features
     (slope, ADF p-value, dominant frequency, kurtosis, etc.)
  3. Random projection: Linear probe on random linear projection of raw input

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/run_baseline_probes.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.synthetic import (
    SyntheticDataset,
    generate_anomaly_dataset,
    generate_change_point_dataset,
    generate_frequency_dataset,
    generate_seasonality_dataset,
    generate_stationarity_dataset,
    generate_trend_dataset,
)
from src.utils.seed import seed_everything

OUTPUT_DIR = Path("outputs/baseline_probes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 1000
SEQ_LEN = 512
SEED = 42
N_FOLDS = 5


@dataclass
class BaselineResult:
    property_name: str
    baseline_type: str
    metric_name: str
    mean_score: float
    std_score: float
    feature_dim: int


def extract_hand_crafted_features(sequences: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract interpretable statistical features from raw time series."""
    n, seq_len = sequences.shape
    features = np.zeros((n, 8), dtype=np.float64)

    t = np.arange(seq_len, dtype=np.float64)
    t_centered = t - t.mean()
    t_var = np.sum(t_centered**2)

    for i in range(n):
        x = sequences[i]
        x_centered = x - x.mean()

        # 1. Linear slope (OLS)
        features[i, 0] = np.sum(t_centered * x_centered) / t_var

        # 2. Residual std after detrending
        trend_line = features[i, 0] * t_centered
        features[i, 1] = np.std(x_centered - trend_line)

        # 3. Mean
        features[i, 2] = x.mean()

        # 4. Std
        features[i, 3] = x.std()

        # 5. Kurtosis
        std = x.std()
        if std > 1e-10:
            features[i, 4] = np.mean(((x - x.mean()) / std) ** 4) - 3.0
        else:
            features[i, 4] = 0.0

        # 6. Dominant frequency (via FFT)
        fft_vals = np.abs(np.fft.rfft(x_centered))
        fft_vals[0] = 0  # ignore DC
        features[i, 5] = float(np.argmax(fft_vals))

        # 7. Spectral entropy
        power = fft_vals**2
        power_sum = power.sum()
        if power_sum > 1e-10:
            p = power / power_sum
            p = p[p > 0]
            features[i, 6] = -np.sum(p * np.log(p + 1e-12))
        else:
            features[i, 6] = 0.0

        # 8. First-order difference std (proxy for stationarity)
        features[i, 7] = np.std(np.diff(x))

    return features


def extract_raw_summary(sequences: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract raw summary statistics: subsample + moments."""
    n, seq_len = sequences.shape

    indices = np.linspace(0, seq_len - 1, 64, dtype=int)
    subsampled = sequences[:, indices]

    means = sequences.mean(axis=1, keepdims=True)
    stds = sequences.std(axis=1, keepdims=True)
    mins = sequences.min(axis=1, keepdims=True)
    maxs = sequences.max(axis=1, keepdims=True)

    return np.hstack([subsampled, means, stds, mins, maxs])


def extract_random_projection(
    sequences: NDArray[np.float64],
    dim: int = 256,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Project raw time series with a fixed random matrix."""
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((sequences.shape[1], dim)).astype(np.float64)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    return sequences @ proj


def run_probe(
    features: NDArray[np.float64],
    labels: NDArray[np.int64] | NDArray[np.float64],
    label_type: str,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> tuple[float, float, str]:
    """Run cross-validated probe and return (mean, std, metric_name)."""
    if label_type == "classification":
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
        )
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(pipe, features, labels, cv=cv, scoring="accuracy")
        return float(scores.mean()), float(scores.std()), "accuracy"
    else:
        from sklearn.model_selection import KFold

        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(pipe, features, labels, cv=cv, scoring="r2")
        return float(scores.mean()), float(scores.std()), "r2"


def main() -> None:
    seed_everything(SEED)

    datasets: list[tuple[str, SyntheticDataset]] = [
        ("trend", generate_trend_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED)),
        ("seasonality", generate_seasonality_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED)),
        ("frequency", generate_frequency_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED)),
        ("stationarity", generate_stationarity_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED)),
        ("anomaly", generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED)),
        ("change_point", generate_change_point_dataset(NUM_SAMPLES, SEQ_LEN, seed=SEED)),
    ]

    all_results: list[dict[str, str | float | int]] = []

    for prop_name, dataset in datasets:
        sequences = dataset.sequences
        labels = dataset.labels
        label_type = dataset.label_type

        print(f"\n{'=' * 60}")
        print(f"Property: {prop_name} ({label_type})")
        print(f"{'=' * 60}")

        hc_features = extract_hand_crafted_features(sequences)
        mean_hc, std_hc, metric = run_probe(hc_features, labels, label_type)
        print(f"  Hand-crafted ({hc_features.shape[1]}d): {metric}={mean_hc:.4f}±{std_hc:.4f}")
        all_results.append(
            {
                "property": prop_name,
                "baseline": "hand_crafted",
                "metric": metric,
                "mean": round(mean_hc, 4),
                "std": round(std_hc, 4),
                "feature_dim": hc_features.shape[1],
            }
        )

        raw_features = extract_raw_summary(sequences)
        mean_raw, std_raw, metric = run_probe(raw_features, labels, label_type)
        print(f"  Raw summary ({raw_features.shape[1]}d): {metric}={mean_raw:.4f}±{std_raw:.4f}")
        all_results.append(
            {
                "property": prop_name,
                "baseline": "raw_summary",
                "metric": metric,
                "mean": round(mean_raw, 4),
                "std": round(std_raw, 4),
                "feature_dim": raw_features.shape[1],
            }
        )

        rand_features = extract_random_projection(sequences, dim=256, seed=SEED)
        mean_rand, std_rand, metric = run_probe(rand_features, labels, label_type)
        print(f"  Random projection (256d):       {metric}={mean_rand:.4f}±{std_rand:.4f}")
        all_results.append(
            {
                "property": prop_name,
                "baseline": "random_projection",
                "metric": metric,
                "mean": round(mean_rand, 4),
                "std": round(std_rand, 4),
                "feature_dim": 256,
            }
        )

        if SEQ_LEN <= 512:
            mean_full, std_full, metric = run_probe(sequences, labels, label_type)
            print(f"  Full raw ({sequences.shape[1]}d): {metric}={mean_full:.4f}±{std_full:.4f}")
            all_results.append(
                {
                    "property": prop_name,
                    "baseline": "full_raw",
                    "metric": metric,
                    "mean": round(mean_full, 4),
                    "std": round(std_full, 4),
                    "feature_dim": sequences.shape[1],
                }
            )

    output_path = OUTPUT_DIR / "baseline_probe_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("BASELINE PROBE SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Property':<15} {'Baseline':<20} {'Metric':<10} {'Score':<15}")
    print("-" * 60)
    for r in all_results:
        print(
            f"{r['property']:<15} {r['baseline']:<20} "
            f"{r['metric']:<10} {r['mean']:.4f}±{r['std']:.4f}"
        )


if __name__ == "__main__":
    main()
