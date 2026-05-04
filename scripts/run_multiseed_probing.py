"""Multi-seed probing with bootstrap confidence intervals.

Runs linear probes on pre-extracted representations with 5 different seeds,
then computes bootstrap 95% CIs for best-layer accuracy.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_multiseed_probing.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.seed import seed_everything

OUTPUT_DIR = Path("outputs/multiseed_probing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPR_ROOT = Path("outputs/representations")

SEEDS = [0, 1, 2, 3, 4]
N_FOLDS = 5
N_BOOTSTRAP = 1000

MODELS = {
    "moment_pca512": "MOMENT",
    "chronos": "Chronos",
    "patchtst_pretrained": "PatchTST",
    "gpt4ts_pca512": "GPT4TS",
    "timer_meanpool": "Timer",
    "timesfm_meanpool": "TimesFM",
    "moirai_meanpool": "Moirai",
}

PROPERTIES = ["stationarity", "anomaly", "frequency"]


def load_best_layer(
    model_key: str, prop: str
) -> tuple[NDArray[np.float64], NDArray[np.int64], str] | None:
    """Load the best-performing layer representations for a model-property pair."""
    repr_dir = REPR_ROOT / model_key / f"synthetic_{prop}"
    if not repr_dir.exists():
        return None

    labels_path = repr_dir / "labels.pt"
    if not labels_path.exists():
        return None
    labels = torch.load(labels_path, map_location="cpu", weights_only=True).numpy()

    layer_files = sorted(f for f in repr_dir.glob("*.pt") if f.stem not in ("labels", "metadata"))
    if not layer_files:
        return None

    best_file = layer_files[0]
    best_acc = -1.0

    for lf in layer_files:
        t = torch.load(lf, map_location="cpu", weights_only=True)
        if t.ndim > 2:
            t = t.mean(dim=1)
        if t.ndim != 2:
            continue
        X = t.numpy().astype(np.float64)
        n = len(X)
        split = int(0.8 * n)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[:split])
        X_val = scaler.transform(X[split:])
        clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(X_train, labels[:split])
        acc = float(clf.score(X_val, labels[split:]))
        if acc > best_acc:
            best_acc = acc
            best_file = lf

    t = torch.load(best_file, map_location="cpu", weights_only=True)
    if t.ndim > 2:
        t = t.mean(dim=1)
    if t.ndim != 2:
        return None
    return t.numpy().astype(np.float64), labels.astype(np.int64), best_file.stem


def run_multiseed_cv(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    seeds: list[int],
    n_folds: int,
) -> list[float]:
    """Run cross-validated probing across multiple seeds, return per-seed mean accuracy."""
    per_seed_scores: list[float] = []

    for seed in seeds:
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"),
        )
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        per_seed_scores.append(float(scores.mean()))

    return per_seed_scores


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = N_BOOTSTRAP,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval. Returns (mean, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    boot_means = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1 - ci) / 2
    return (
        float(arr.mean()),
        float(np.percentile(boot_means, alpha * 100)),
        float(np.percentile(boot_means, (1 - alpha) * 100)),
    )


def main() -> None:
    seed_everything(42)
    all_results: list[dict[str, object]] = []

    for model_key, model_name in MODELS.items():
        for prop in PROPERTIES:
            loaded = load_best_layer(model_key, prop)
            if loaded is None:
                print(f"  {model_name}/{prop}: skipped (no data)")
                continue

            X, y, layer_name = loaded
            per_seed = run_multiseed_cv(X, y, SEEDS, N_FOLDS)
            mean_val, ci_low, ci_high = bootstrap_ci(per_seed)

            print(
                f"  {model_name:12s} {prop:15s} "
                f"{mean_val:.4f} [{ci_low:.4f}, {ci_high:.4f}] "
                f"(std={np.std(per_seed):.4f}) layer={layer_name}"
            )

            all_results.append(
                {
                    "model": model_key,
                    "model_name": model_name,
                    "property": prop,
                    "layer": layer_name,
                    "seeds": SEEDS,
                    "per_seed_accuracy": [round(v, 4) for v in per_seed],
                    "mean": round(mean_val, 4),
                    "std": round(float(np.std(per_seed)), 4),
                    "ci_95_low": round(ci_low, 4),
                    "ci_95_high": round(ci_high, 4),
                }
            )

    output_path = OUTPUT_DIR / "multiseed_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
