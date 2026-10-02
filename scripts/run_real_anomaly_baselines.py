"""Baseline-controlled probing on real UCR anomaly tasks (Phase-1 / D1).

Runs the non-model controls (hand-crafted 8-D, raw signal, random projection) on the
UCR anomaly datasets under the official leak-free split, reporting threshold-free
metrics (ROC-AUC, Average Precision) plus accuracy. This is the real-data counterpart
to the synthetic anomaly baselines; TSFM probes on the same tasks are added once
representations are extracted.

Usage:
    PYTHONPATH=. python scripts/run_real_anomaly_baselines.py
Output:
    outputs/real_anomaly_baselines/results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.run_per_feature_anomaly import hand_crafted_features
from src.datasets.ucr_anomaly import ANOMALY_DATASETS, load_ucr_anomaly
from src.metrics.tsad_metrics import window_threshold_free_metrics
from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/real_anomaly_baselines")
SEED = 42


def _fit_predict(
    Xtr: NDArray, ytr: NDArray, Xte: NDArray
) -> tuple[NDArray, NDArray]:
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=SEED, solver="lbfgs"),
    )
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = pipe.predict(Xte)
    return proba, pred


def _random_projection(X: NDArray, k: int, seed: int) -> NDArray:
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(X.shape[1], min(k, X.shape[1])))
    return X @ R


def evaluate_baseline(name: str, task, baseline: str) -> dict:
    if baseline == "hand_crafted":
        Xtr = hand_crafted_features(task.train_sequences)
        Xte = hand_crafted_features(task.test_sequences)
    elif baseline == "raw_signal":
        Xtr, Xte = task.train_sequences, task.test_sequences
    elif baseline == "random_projection":
        k = 64
        Xtr = _random_projection(task.train_sequences, k, SEED)
        Xte = _random_projection(task.test_sequences, k, SEED)
    else:
        raise ValueError(baseline)

    proba, pred = _fit_predict(Xtr, task.train_labels, Xte)
    tf = window_threshold_free_metrics(proba, task.test_labels)
    return {
        "dataset": name,
        "baseline": baseline,
        "n_train": int(task.train_sequences.shape[0]),
        "n_test": int(task.test_sequences.shape[0]),
        "seq_len": int(task.train_sequences.shape[1]),
        "anomaly_rate": round(task.anomaly_rate, 4),
        "roc_auc": round(tf["roc_auc"], 4),
        "average_precision": round(tf["average_precision"], 4),
        "accuracy": round(float(accuracy_score(task.test_labels, pred)), 4),
    }


def main() -> None:
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ANOMALY_DATASETS:
        task = load_ucr_anomaly(name)
        for baseline in ("hand_crafted", "raw_signal", "random_projection"):
            row = evaluate_baseline(name, task, baseline)
            rows.append(row)
            print(
                f"{name:14} {baseline:18} ROC-AUC={row['roc_auc']:.3f} "
                f"AP={row['average_precision']:.3f} acc={row['accuracy']:.3f}"
            )
    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
