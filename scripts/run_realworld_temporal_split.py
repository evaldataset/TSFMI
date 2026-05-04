"""Real-world probing under strict temporal split (S5).

Addresses the random-split-overlap concern for sliding-window real-world
evaluation. Uses a strict temporal split with embargo gap to prevent
adjacent windows leaking between train and test.

Protocol:
    - Z-score normalization computed on train portion only (already train-only)
    - Sliding windows extracted with stride 256, length 512
    - Temporal split: first 60% train, next 20% val, last 20% test
    - Embargo: drop windows where train and test overlap (i.e., windows whose
      seq_len region crosses the train/test boundary)
    - Best layer selected on val, final accuracy on test

Output: outputs/realworld_temporal/results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path("outputs/realworld_temporal")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPR_ROOT = Path("outputs/representations")

MODELS = {
    "MOMENT": "moment_pca512",
    "Chronos": "chronos",
    "PatchTST": "patchtst_pretrained",
    "GPT4TS": "gpt4ts_pca512",
    "Timer": "timer_meanpool",
    "TimesFM": "timesfm_meanpool",
    "Moirai": "moirai_meanpool",
}

DATASETS = ["etth1", "weather", "electricity"]
PROPERTIES = ["trend", "stationarity"]
SEEDS = [0, 1, 2, 3, 4]
EMBARGO = 1  # drop 1 window between train/val/test boundaries


def temporal_split_with_embargo(
    n: int, train_ratio: float = 0.6, val_ratio: float = 0.2, embargo: int = 1
) -> tuple[NDArray, NDArray, NDArray]:
    """Strict temporal indices with embargo at boundaries."""
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = np.arange(n_train - embargo)
    val_idx = np.arange(n_train + embargo, n_train + n_val - embargo)
    test_idx = np.arange(n_train + n_val + embargo, n)
    return train_idx, val_idx, test_idx


def find_best_layer_temporal(
    model_key: str, ds_name: str, y: NDArray, seed: int
) -> tuple[NDArray, str] | None:
    d = REPR_ROOT / model_key / ds_name
    if not d.exists():
        return None
    layer_files = sorted(f for f in d.glob("*.pt") if f.stem not in ("labels", "metadata"))
    best_acc = -1.0
    best_X = None
    best_layer_name = ""
    for lf in layer_files:
        t = torch.load(lf, map_location="cpu", weights_only=True)
        if t.ndim > 2:
            t = t.reshape(t.shape[0], -1)
        if t.ndim < 2 or t.shape[0] != len(y) or t.shape[1] == 0:
            continue
        X = t.numpy().astype(np.float64)
        tr_idx, va_idx, _ = temporal_split_with_embargo(len(X))
        if len(va_idx) < 5:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=seed, solver="lbfgs"),
        )
        try:
            clf.fit(X[tr_idx], y[tr_idx].astype(np.int64))
            acc = accuracy_score(y[va_idx].astype(np.int64), clf.predict(X[va_idx]))
        except Exception:
            continue
        if acc > best_acc:
            best_acc = acc
            best_X = X
            best_layer_name = lf.stem
    return (best_X, best_layer_name) if best_X is not None else None


def evaluate_test(X: NDArray, y: NDArray, seed: int) -> float:
    tr_idx, _, te_idx = temporal_split_with_embargo(len(X))
    if len(te_idx) < 3:
        return float("nan")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, random_state=seed, solver="lbfgs"),
    )
    try:
        clf.fit(X[tr_idx], y[tr_idx].astype(np.int64))
        return float(accuracy_score(y[te_idx].astype(np.int64), clf.predict(X[te_idx])))
    except Exception:
        return float("nan")


def main() -> None:
    rows = []
    for ds in DATASETS:
        for prop in PROPERTIES:
            ds_name = f"{ds}_{prop}"
            for model_name, model_key in MODELS.items():
                labels_path = REPR_ROOT / model_key / ds_name / "labels.pt"
                if not labels_path.exists():
                    continue
                y = (
                    torch.load(labels_path, map_location="cpu", weights_only=True)
                    .numpy()
                    .astype(np.int64)
                )
                if len(np.unique(y)) < 2:
                    continue
                ret = find_best_layer_temporal(model_key, ds_name, y, seed=0)
                if ret is None:
                    continue
                X, best_layer = ret
                test_accs = [evaluate_test(X, y, s) for s in SEEDS]
                test_accs = [a for a in test_accs if not np.isnan(a)]
                if not test_accs:
                    continue
                tr_idx, va_idx, te_idx = temporal_split_with_embargo(len(X))
                rows.append(
                    {
                        "dataset": ds,
                        "property": prop,
                        "model": model_name,
                        "best_layer": best_layer,
                        "n_train": int(len(tr_idx)),
                        "n_val": int(len(va_idx)),
                        "n_test": int(len(te_idx)),
                        "test_mean": round(float(np.mean(test_accs)), 4),
                        "test_std": round(float(np.std(test_accs)), 4),
                    }
                )
                print(
                    f"  {ds:12s} {prop:12s} {model_name:8s} layer={best_layer} "
                    f"n_tr={len(tr_idx)} n_te={len(te_idx)} test={np.mean(test_accs):.3f}"
                )

    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\nSaved to {OUT_DIR}/results.json ({len(rows)} entries)")


if __name__ == "__main__":
    main()
