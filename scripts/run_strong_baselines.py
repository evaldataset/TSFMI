"""Stronger TS-specific baselines: catch22, tsfresh, TS2Vec (S3).

Goal: show that the "TSFMs lose to hand-crafted features on anomaly" finding
is robust to baseline strength. We add three stronger / TS-specific baseline
families on top of the existing 8-D HC, raw-signal, random-projection set:

    - catch22: 22 canonical time-series features (Lubba et al., DAMI'19)
    - tsfresh:  ~700 statistical features (only the "efficient" subset)
    - TS2Vec embeddings (if package + CUDA available)

If TSFMs still lose to *all* of these on anomaly, the inversion finding
becomes impossible to dismiss as a feature-engineering issue.

Output: outputs/strong_baselines/results.json
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

OUT_DIR = Path("outputs/strong_baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_SAMPLES = 1000
SEQ_LEN = 512
SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 1000


def split(X: NDArray, y: NDArray, seed: int) -> tuple[NDArray, ...]:
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.20)
    n_val = int(n * 0.20)
    te = idx[:n_test]
    va = idx[n_test : n_test + n_val]
    tr = idx[n_test + n_val :]
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def score(features: NDArray, labels: NDArray, task: str) -> tuple[float, float, float]:
    accs = []
    for seed in SEEDS:
        Xtr, ytr, _, _, Xte, yte = split(features, labels, seed)
        if task == "classification":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
            )
            clf.fit(Xtr, ytr.astype(np.int64))
            accs.append(float(accuracy_score(yte.astype(np.int64), clf.predict(Xte))))
        else:
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            reg.fit(Xtr, ytr)
            accs.append(float(r2_score(yte, reg.predict(Xte))))
    rng = np.random.default_rng(42)
    arr = np.array(accs)
    boot = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(N_BOOTSTRAP)]
    )
    return float(arr.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def extract_tsfresh(seq: NDArray, minimal: bool = False) -> NDArray | None:
    """Extract tsfresh features. Use MinimalFCParameters when fast=True."""
    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
    except ImportError:
        print("  tsfresh not installed; pip install tsfresh")
        return None
    import pandas as pd

    rows = []
    for i in range(seq.shape[0]):
        for t in range(seq.shape[1]):
            rows.append({"id": i, "time": t, "value": float(seq[i, t])})
    df = pd.DataFrame(rows)
    params = MinimalFCParameters() if minimal else EfficientFCParameters()
    feats = extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=params,
        disable_progressbar=True,
        n_jobs=4,  # share cores with S2/S5
    )
    feats = feats.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return feats.to_numpy()


DATASETS = {
    "trend": ("classification", generate_trend_dataset),
    "seasonality": ("regression", generate_seasonality_dataset),
    "frequency": ("classification", generate_frequency_dataset),
    "stationarity": ("classification", generate_stationarity_dataset),
    "anomaly": ("classification", generate_anomaly_dataset),
    "change_point": ("classification", generate_change_point_dataset),
}


def main() -> None:
    seed_everything(SEED)
    rows = []
    for prop, (task, gen_fn) in DATASETS.items():
        ds = gen_fn(NUM_SAMPLES, SEQ_LEN, seed=SEED)
        sequences = ds.sequences
        labels = ds.labels
        print(f"\n=== {prop} ({task}) ===")
        for name, fast in [("tsfresh_minimal", True), ("tsfresh_efficient", False)]:
            print(f"  Extracting {name}...")
            feats = extract_tsfresh(sequences, minimal=fast)
            if feats is None:
                continue
            print(f"    feature dim: {feats.shape[1]}")
            mean, lo, hi = score(feats, labels, task)
            rows.append(
                {
                    "property": prop,
                    "baseline": name,
                    "task_type": task,
                    "feature_dim": int(feats.shape[1]),
                    "test_mean": round(mean, 4),
                    "test_ci95_low": round(lo, 4),
                    "test_ci95_high": round(hi, 4),
                }
            )
            print(f"    {name:10s} test={mean:.4f} [{lo:.4f}, {hi:.4f}]")

    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved to {OUT_DIR}/results.json")


if __name__ == "__main__":
    main()
