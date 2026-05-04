"""Real-world transfer analysis: label stability and error diagnosis.

Addresses the synthetic→real transfer weakness by:
1. Label stability: bootstrap perturbation to measure auto-label reliability
2. Confusion matrices: analyze misclassification patterns on real-world probes
3. Error characterization: identify which windows cause probe failures

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_real_world_analysis.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from src.datasets.real_world import (
    _label_change_point,
    _label_stationarity,
    _label_trend,
)
from src.utils.seed import seed_everything

LABEL_FUNCTIONS: dict[str, Callable[[NDArray[np.float64]], int]] = {
    "trend": _label_trend,
    "stationarity": _label_stationarity,
    "change_point": _label_change_point,
}

MODEL_REPR_KEYS: dict[str, str] = {
    "MOMENT": "moment_pca512",
    "Chronos": "chronos",
    "PatchTST": "patchtst_pretrained",
    "GPT4TS": "gpt4ts_pca512",
}

REAL_DATASETS = ["etth1", "weather"]
PROPERTIES = ["trend", "stationarity", "change_point"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-world transfer analysis: label stability + error diagnosis.",
    )
    parser.add_argument("--repr_root", type=str, default="outputs/representations")
    parser.add_argument("--output_dir", type=str, default="outputs/real_world_analysis")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--n_bootstrap", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.2)
    return parser.parse_args(argv)


def bootstrap_label_stability(
    windows: NDArray[np.float64],
    label_fn: Callable[[NDArray[np.float64]], int],
    n_bootstrap: int = 100,
    noise_std: float = 0.01,
    seed: int = 42,
) -> dict[str, float]:
    """Measure label stability under small perturbations.

    For each window, adds small Gaussian noise n_bootstrap times and checks
    whether the auto-label changes. High agreement = stable labels.

    Args:
        windows: Time series windows (N, seq_len).
        label_fn: Function mapping window → label.
        n_bootstrap: Number of perturbation trials.
        noise_std: Standard deviation of perturbation noise.
        seed: Random seed.

    Returns:
        Dict with agreement_rate, flip_rate, per_class_stability.
    """
    rng = np.random.default_rng(seed)
    n_windows = len(windows)

    # Get original labels
    original_labels = np.array([label_fn(w) for w in windows])

    # Count how often label stays the same under noise
    n_stable = 0
    n_total = 0
    flip_counts_per_class: dict[int, list[int]] = {}

    for i in range(n_windows):
        cls = int(original_labels[i])
        if cls not in flip_counts_per_class:
            flip_counts_per_class[cls] = []
        flips = 0

        for _ in range(n_bootstrap):
            noise = rng.normal(0, noise_std * windows[i].std(), size=windows[i].shape)
            perturbed = windows[i] + noise
            new_label = label_fn(perturbed)
            if new_label == original_labels[i]:
                n_stable += 1
            else:
                flips += 1
            n_total += 1

        flip_counts_per_class[cls].append(flips)

    agreement_rate = n_stable / n_total if n_total > 0 else 0.0

    per_class_stability = {}
    for cls, flip_list in sorted(flip_counts_per_class.items()):
        avg_flips = np.mean(flip_list)
        stability = 1.0 - avg_flips / n_bootstrap
        per_class_stability[str(cls)] = float(stability)

    return {
        "agreement_rate": float(agreement_rate),
        "flip_rate": float(1.0 - agreement_rate),
        "n_windows": n_windows,
        "n_bootstrap": n_bootstrap,
        "noise_std": noise_std,
        "per_class_stability": per_class_stability,
    }


def compute_probe_confusion(
    repr_dir: Path,
    val_split: float,
    seed: int,
) -> dict[str, object]:
    """Train Ridge probe and compute confusion matrix on real-world data.

    Args:
        repr_dir: Directory with layer .pt files and labels.pt.
        val_split: Validation split fraction.
        seed: Random seed.

    Returns:
        Dict with accuracy, confusion_matrix, best_layer.
    """
    import torch

    labels_path = repr_dir / "labels.pt"
    if not labels_path.exists():
        return {}

    labels = torch.load(labels_path, map_location="cpu", weights_only=True).numpy()
    if labels.ndim > 1:
        labels = labels.squeeze()

    layer_files = sorted(
        [f for f in repr_dir.glob("*.pt") if f.stem != "labels"],
    )
    if not layer_files:
        return {}

    best_acc = -1.0
    best_cm = None
    best_layer = ""

    for layer_file in layer_files:
        reprs = torch.load(layer_file, map_location="cpu", weights_only=True)
        if reprs.ndim >= 3:
            reprs = reprs.view(reprs.shape[0], -1)
        X = reprs.numpy().astype(np.float32)

        if X.shape[0] != len(labels):
            continue

        # Temporal split: earlier windows for train, later for test (no shuffle)
        split_idx = int(X.shape[0] * (1 - val_split))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]

        clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))

        if acc > best_acc:
            best_acc = acc
            best_cm = confusion_matrix(y_test, y_pred).tolist()
            best_layer = layer_file.stem

    if best_acc < 0:
        return {}

    return {
        "accuracy": best_acc,
        "confusion_matrix": best_cm,
        "best_layer": best_layer,
    }


def load_real_world_windows(
    dataset_name: str,
    data_dir: str,
    seq_len: int = 512,
    stride: int = 256,
    max_windows: int = 200,
) -> NDArray[np.float64] | None:
    """Load real-world data and extract sliding windows.

    Args:
        dataset_name: One of "etth1", "weather".
        data_dir: Directory containing data files.
        seq_len: Window length.
        stride: Step between windows.
        max_windows: Maximum number of windows to extract.

    Returns:
        Array of shape (N, seq_len) or None if data not found.
    """
    from src.datasets.real_world import _load_univariate_series

    try:
        series = _load_univariate_series(dataset_name, data_dir)
    except (FileNotFoundError, ValueError):
        return None

    # Z-score normalize
    mean, std = series.mean(), series.std()
    if std > 1e-8:
        series = (series - mean) / std

    # Extract windows
    windows = []
    for start in range(0, len(series) - seq_len + 1, stride):
        windows.append(series[start : start + seq_len])
        if len(windows) >= max_windows:
            break

    if not windows:
        return None

    return np.array(windows, dtype=np.float64)


def main() -> None:
    """Run real-world transfer analysis."""
    args = parse_args()
    seed_everything(args.seed)
    repr_root = Path(args.repr_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {
        "label_stability": {},
        "confusion_matrices": {},
    }

    # 1. Label stability analysis
    print("=" * 60)
    print("LABEL STABILITY ANALYSIS")
    print("=" * 60)

    for dataset_name in REAL_DATASETS:
        windows = load_real_world_windows(dataset_name, args.data_dir)
        if windows is None:
            print(f"SKIP {dataset_name}: data not found")
            continue

        print(f"\n{dataset_name} ({len(windows)} windows)")
        all_results["label_stability"][dataset_name] = {}

        for prop, label_fn in LABEL_FUNCTIONS.items():
            stability = bootstrap_label_stability(
                windows, label_fn,
                n_bootstrap=args.n_bootstrap,
                noise_std=args.noise_std,
                seed=args.seed,
            )
            all_results["label_stability"][dataset_name][prop] = stability

            print(
                f"  {prop:<14}: agreement={stability['agreement_rate']:.3f}  "
                f"flip_rate={stability['flip_rate']:.3f}"
            )
            for cls, stab in stability["per_class_stability"].items():
                print(f"    class {cls}: stability={stab:.3f}")

    # 2. Confusion matrices on real-world probes
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)

    for model_name, repr_key in MODEL_REPR_KEYS.items():
        all_results["confusion_matrices"][model_name] = {}

        for dataset_name in REAL_DATASETS:
            for prop in PROPERTIES:
                repr_dir = repr_root / repr_key / f"{dataset_name}_{prop}"
                if not repr_dir.exists():
                    continue

                result = compute_probe_confusion(
                    repr_dir, args.val_split, args.seed,
                )
                if result:
                    key = f"{dataset_name}_{prop}"
                    all_results["confusion_matrices"][model_name][key] = result
                    print(
                        f"  {model_name}/{dataset_name}/{prop}: "
                        f"acc={result['accuracy']:.3f} "
                        f"layer={result['best_layer']}"
                    )

    # Save results
    results_path = output_dir / "analysis_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved all results to {results_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Label Stability Across Properties")
    print("=" * 60)
    print(f"{'Dataset':<12} {'Property':<14} {'Agreement':>10} {'Flip Rate':>10}")
    print("-" * 50)
    for dataset_name in REAL_DATASETS:
        ds_results = all_results["label_stability"].get(dataset_name, {})
        for prop in PROPERTIES:
            stab = ds_results.get(prop)
            if stab:
                print(
                    f"{dataset_name:<12} {prop:<14} "
                    f"{stab['agreement_rate']:>10.3f} "
                    f"{stab['flip_rate']:>10.3f}"
                )


if __name__ == "__main__":
    main()
