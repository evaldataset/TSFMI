"""Canonical baseline controls with the same protocol as model probes.

Uses the SAME protocol as run_canonical_benchmark.py:
    - train / val / test 3-way split (60/20/20)
    - 5 seeds with bootstrap 95% CI
    - sklearn LogisticRegression (classification) / Ridge (regression)
    - StandardScaler inside pipeline

Baselines:
    - hand-crafted features (8D: slope, residual std, mean, std, kurtosis,
      dominant freq, spectral entropy, diff std)
    - raw signal (full 512D flattened time series)
    - random projection (256D Gaussian projection of raw signal)

Output: outputs/canonical_baselines/<property>/canonical_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.synthetic import (
    generate_anomaly_dataset,
    generate_change_point_dataset,
    generate_frequency_dataset,
    generate_seasonality_dataset,
    generate_stationarity_dataset,
    generate_trend_dataset,
)
from src.utils.seed import seed_everything

OUT_ROOT = Path("outputs/canonical_baselines")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 1000
NUM_SAMPLES = 1000
SEQ_LEN = 512


def hand_crafted_features(seq: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Extract an 8-dim interpretable hand-crafted feature vector per sequence.

    Features: (0) linear slope, (1) detrended residual std, (2) mean, (3) std,
    (4) excess kurtosis, (5) argmax FFT bin (dominant frequency),
    (6) spectral entropy, (7) first-difference std.

    Note: this compact 8-D set intentionally does not attempt to match the
    raw-signal baseline on frequency. Raw signal (512-D) linearly decodes
    frequency exactly via FFT; any bounded-dimension hand-crafted set will
    therefore under-perform raw there. The 8-D set remains the published
    baseline because it is interpretable and standard in TS feature engineering
    (cf.\ TSFresh, catch22). We report the gap (HC 0.944 vs.\ raw 1.000 on
    frequency) as a legitimate observation about feature-dimensionality
    sufficiency, not as a baseline calibration issue.
    """
    n, L = seq.shape
    t = np.arange(L, dtype=np.float64) - L / 2
    tv = (t**2).sum()
    feats = np.zeros((n, 8))
    for i in range(n):
        x = seq[i]
        xc = x - x.mean()
        slope = (t * xc).sum() / tv
        feats[i, 0] = slope
        feats[i, 1] = np.std(xc - slope * t)
        feats[i, 2] = x.mean()
        feats[i, 3] = x.std()
        std = x.std()
        feats[i, 4] = np.mean(((x - x.mean()) / (std + 1e-10)) ** 4) - 3
        fft = np.abs(np.fft.rfft(xc))
        fft[0] = 0
        feats[i, 5] = float(np.argmax(fft))
        power = fft**2
        ps = power.sum()
        if ps > 1e-10:
            p = power / ps
            p = p[p > 0]
            feats[i, 6] = -np.sum(p * np.log(p + 1e-12))
        feats[i, 7] = np.std(np.diff(x))
    return feats


def random_projection(
    seq: NDArray[np.float64], dim: int = 256, seed: int = 42
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((seq.shape[1], dim)).astype(np.float64)
    proj /= np.linalg.norm(proj, axis=0, keepdims=True)
    return seq @ proj


def three_way_split(
    X: NDArray[np.float64], y: NDArray[np.float64], seed: int
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.20)
    n_val = int(n * 0.20)
    te = idx[:n_test]
    va = idx[n_test : n_test + n_val]
    tr = idx[n_test + n_val :]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def score(
    X_tr: NDArray[np.float64],
    y_tr: NDArray[np.float64],
    X_te: NDArray[np.float64],
    y_te: NDArray[np.float64],
    task: str,
    seed: int,
) -> float:
    if task == "classification":
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
        )
        pipe.fit(X_tr, y_tr.astype(np.int64))
        return float(accuracy_score(y_te.astype(np.int64), pipe.predict(X_te)))
    else:
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        pipe.fit(X_tr, y_tr)
        return float(r2_score(y_te, pipe.predict(X_te)))


def bootstrap_ci(
    values: list[float], n_bootstrap: int = N_BOOTSTRAP, ci: float = 0.95
) -> tuple[float, float, float]:
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boot = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1 - ci) / 2
    return (
        float(arr.mean()),
        float(np.percentile(boot, alpha * 100)),
        float(np.percentile(boot, (1 - alpha) * 100)),
    )


DATASETS = {
    "trend": ("classification", generate_trend_dataset),
    "seasonality": ("regression", generate_seasonality_dataset),
    "frequency": ("classification", generate_frequency_dataset),
    "stationarity": ("classification", generate_stationarity_dataset),
    "anomaly": ("classification", generate_anomaly_dataset),
    "change_point": ("classification", generate_change_point_dataset),
}


def main() -> None:
    seed_everything(42)
    all_results = []

    for prop_name, (task, gen_fn) in DATASETS.items():
        ds = gen_fn(NUM_SAMPLES, SEQ_LEN, seed=42)
        sequences = ds.sequences
        labels = ds.labels

        baselines = {
            "hand_crafted": hand_crafted_features(sequences),
            "raw_signal": sequences,
            "random_projection": random_projection(sequences),
        }

        for bl_name, features in baselines.items():
            per_seed_test: list[float] = []
            for seed in SEEDS:
                Xtr, ytr, Xva, yva, Xte, yte = three_way_split(
                    features, labels, seed
                )
                s = score(Xtr, ytr, Xte, yte, task, seed)
                per_seed_test.append(s)

            mean, ci_low, ci_high = bootstrap_ci(per_seed_test)
            result = {
                "property": prop_name,
                "baseline": bl_name,
                "task_type": task,
                "test_mean": round(mean, 4),
                "test_ci95_low": round(ci_low, 4),
                "test_ci95_high": round(ci_high, 4),
                "per_seed_test": [round(v, 4) for v in per_seed_test],
                "protocol": "60/20/20 train/val/test, 5 seeds, matched estimator",
            }
            all_results.append(result)
            print(
                f"  {prop_name:15s} {bl_name:20s} "
                f"test={mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
            )

        # Save per-property
        out_dir = OUT_ROOT / prop_name
        out_dir.mkdir(parents=True, exist_ok=True)
        prop_results = [r for r in all_results if r["property"] == prop_name]
        (out_dir / "canonical_results.json").write_text(
            json.dumps(prop_results, indent=2)
        )

    # Save all
    (OUT_ROOT / "all_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
