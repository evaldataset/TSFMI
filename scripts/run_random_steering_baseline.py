"""Random-direction steering baseline for LDA steering experiments.

Tests whether LDA steering drops are specific to the concept direction or
just an effect of perturbation magnitude. For each model-property pair,
applies random-direction perturbations of the same norm as LDA steering
and measures the accuracy drop.

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/run_random_steering_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.seed import seed_everything

INTERVENTIONS_DIR = Path("outputs/interventions")
REPR_ROOT = Path("outputs/representations")
OUTPUT_DIR = Path("outputs/random_steering_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_RANDOM_DIRECTIONS = 20
ALPHA_VALUES = [1.0, 2.0, 3.0, 5.0]

MODEL_REPR_MAP = {
    "gpt4ts_pca512": "gpt4ts_pca512",
    "moment_pca512": "moment_pca512",
    "chronos": "chronos",
    "patchtst_pretrained": "patchtst_pretrained",
}

PROPERTIES = ["trend", "frequency_hard", "stationarity"]


def load_representations_and_labels(
    model_key: str, prop: str
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]] | None:
    """Load best-layer representations and labels for a model-property pair."""
    base = REPR_ROOT / model_key
    synth_dir = base / f"synthetic_{prop}"
    if not synth_dir.exists():
        synth_dir = base / prop
    if not synth_dir.exists():
        return None

    labels_path = synth_dir / "labels.pt"
    if not labels_path.exists():
        return None

    labels = torch.load(labels_path, map_location="cpu", weights_only=True).numpy()

    layer_files = sorted(
        f for f in synth_dir.glob("*.pt") if f.stem != "labels" and f.stem != "metadata"
    )
    if not layer_files:
        return None

    best_layer_file = layer_files[0]
    best_acc = 0.0

    for lf in layer_files:
        repr_t = torch.load(lf, map_location="cpu", weights_only=True)
        if repr_t.ndim > 2:
            repr_t = repr_t.mean(dim=1)
        if repr_t.ndim != 2:
            continue
        X = repr_t.numpy().astype(np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        n = len(X_scaled)
        split = int(0.8 * n)
        clf = LogisticRegression(max_iter=500, random_state=SEED)
        clf.fit(X_scaled[:split], labels[:split])
        acc = float(clf.score(X_scaled[split:], labels[split:]))
        if acc > best_acc:
            best_acc = acc
            best_layer_file = lf

    repr_t = torch.load(best_layer_file, map_location="cpu", weights_only=True)
    if repr_t.ndim > 2:
        repr_t = repr_t.mean(dim=1)
    if repr_t.ndim != 2:
        return None

    return repr_t.numpy().astype(np.float64), labels.astype(np.int64), [best_layer_file.stem]


def compute_lda_direction(X: NDArray[np.float64], y: NDArray[np.int64]) -> NDArray[np.float64]:
    """Compute LDA direction maximizing class separation."""
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    S_b = np.zeros((X.shape[1], X.shape[1]))

    for c in classes:
        X_c = X[y == c]
        mean_c = X_c.mean(axis=0)
        diff = (mean_c - overall_mean).reshape(-1, 1)
        S_b += len(X_c) * (diff @ diff.T)

    eigenvalues, eigenvectors = np.linalg.eigh(S_b)
    direction = eigenvectors[:, -1]
    return direction / (np.linalg.norm(direction) + 1e-12)


def steer_and_evaluate(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
    direction: NDArray[np.float64],
    alpha: float,
) -> float:
    """Apply steering and return accuracy."""
    X_test_steered = X_test + alpha * direction
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test_steered)
    clf = LogisticRegression(max_iter=500, random_state=SEED)
    clf.fit(X_train_s, y_train)
    return float(clf.score(X_test_s, y_test))


def main() -> None:
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    all_results: list[dict[str, object]] = []

    for model_key in MODEL_REPR_MAP:
        for prop in PROPERTIES:
            print(f"\n{'=' * 60}")
            print(f"{model_key} / {prop}")
            print(f"{'=' * 60}")

            loaded = load_representations_and_labels(model_key, prop)
            if loaded is None:
                print("  Skipped (no data)")
                continue

            X, y, layer_names = loaded
            n = len(X)
            split = int(0.8 * n)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            clf_base = LogisticRegression(max_iter=500, random_state=SEED)
            clf_base.fit(X_train_s, y_train)
            baseline_acc = float(clf_base.score(X_test_s, y_test))
            print(f"  Baseline accuracy: {baseline_acc:.4f}")

            lda_dir = compute_lda_direction(X_train, y_train)

            for alpha in ALPHA_VALUES:
                lda_acc = steer_and_evaluate(X_train, y_train, X_test, y_test, lda_dir, alpha)
                lda_drop = baseline_acc - lda_acc

                random_drops = []
                for _ in range(N_RANDOM_DIRECTIONS):
                    rand_dir = rng.standard_normal(X.shape[1])
                    rand_dir /= np.linalg.norm(rand_dir) + 1e-12
                    rand_acc = steer_and_evaluate(X_train, y_train, X_test, y_test, rand_dir, alpha)
                    random_drops.append(baseline_acc - rand_acc)

                mean_random_drop = float(np.mean(random_drops))
                std_random_drop = float(np.std(random_drops))
                max_random_drop = float(np.max(random_drops))

                print(
                    f"  α={alpha:.1f}: LDA drop={lda_drop:.4f}, "
                    f"Random drop={mean_random_drop:.4f}±{std_random_drop:.4f} "
                    f"(max={max_random_drop:.4f})"
                )

                all_results.append(
                    {
                        "model": model_key,
                        "property": prop,
                        "alpha": alpha,
                        "baseline_acc": round(baseline_acc, 4),
                        "lda_acc": round(lda_acc, 4),
                        "lda_drop": round(lda_drop, 4),
                        "random_drop_mean": round(mean_random_drop, 4),
                        "random_drop_std": round(std_random_drop, 4),
                        "random_drop_max": round(max_random_drop, 4),
                        "lda_specificity": round(lda_drop - mean_random_drop, 4),
                        "best_layer": layer_names[0],
                    }
                )

    output_path = OUTPUT_DIR / "random_steering_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
