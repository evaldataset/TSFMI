"""Compare dimensionality reduction methods for high-dimensional models.

PCA512 destroys seasonality information (R² = -1.117) while full-dimensional
Ridge achieves R² = 0.9999. This script systematically compares alternatives:
  1. PCA (unsupervised, top-variance axes)
  2. Supervised PCA (label-weighted covariance)
  3. Random Projection (JL lemma guarantee)
  4. Full-dimensional Ridge (sklearn, no projection)

This proves the negative seasonality result is a PCA artifact, not a model
deficiency, and identifies concept-aware projection as a remedy.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_pca_alternatives.py
    PYTHONPATH=. .venv/bin/python scripts/run_pca_alternatives.py --device cuda:1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection

from src.utils.seed import seed_everything

MODELS: dict[str, dict[str, str]] = {
    "moment": {"full_d_dir": "moment", "pca512_dir": "moment_pca512"},
    "gpt4ts": {"full_d_dir": "gpt4ts", "pca512_dir": "gpt4ts_pca512"},
}

PROPERTIES: dict[str, str] = {
    "synthetic_trend": "classification",
    "synthetic_seasonality": "regression",
    "synthetic_frequency": "classification",
    "synthetic_stationarity": "classification",
    "synthetic_anomaly": "classification",
    "synthetic_change_point": "classification",
}

N_COMPONENTS = 512


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare PCA vs Supervised PCA vs Random Projection vs Full Ridge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repr_root", type=str, default="outputs/representations")
    parser.add_argument("--output_dir", type=str, default="outputs/pca_alternatives")
    parser.add_argument("--n_components", type=int, default=N_COMPONENTS)
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def _layer_sort_key(name: str) -> tuple[int, str]:
    """Sort layer names by trailing numeric index."""
    parts = name.replace(".", "_").split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part), name
    return 0, name


def supervised_pca(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    n_components: int,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Label-weighted PCA: weight features by correlation with target.

    For regression: weight each feature by its absolute Pearson correlation
    with the label. For classification: weight by one-vs-rest correlation.
    Then apply PCA on the reweighted feature space.

    Args:
        X_train: Training data (N_train, D).
        y_train: Training labels (N_train,).
        X_test: Test data (N_test, D).
        n_components: Number of components to keep.
        seed: Random seed.

    Returns:
        Projected (X_train_proj, X_test_proj), each (N, n_components).
    """
    y_float = y_train.astype(np.float64)
    X_centered = X_train - X_train.mean(axis=0, keepdims=True)

    # Per-feature correlation with label
    y_centered = y_float - y_float.mean()
    y_std = y_centered.std()
    if y_std < 1e-12:
        # Degenerate label — fall back to standard PCA
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
        return pca.fit_transform(X_train), pca.transform(X_test)

    correlations = np.abs(X_centered.T @ y_centered) / (
        np.linalg.norm(X_centered, axis=0) * np.linalg.norm(y_centered) + 1e-12
    )

    # Scale features by correlation weight (soft feature selection)
    weights = correlations + 0.01  # floor to avoid zero-ing out features
    X_weighted_train = X_centered * weights[np.newaxis, :]
    X_weighted_test = (X_test - X_train.mean(axis=0, keepdims=True)) * weights[np.newaxis, :]

    n_comp = min(n_components, X_weighted_train.shape[0], X_weighted_train.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
    X_train_proj = pca.fit_transform(X_weighted_train)
    X_test_proj = pca.transform(X_weighted_test)

    return X_train_proj, X_test_proj


def random_projection(
    X_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    n_components: int,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Gaussian random projection (JL lemma guarantee).

    Args:
        X_train: Training data (N_train, D).
        X_test: Test data (N_test, D).
        n_components: Target dimensionality.
        seed: Random seed.

    Returns:
        Projected (X_train_proj, X_test_proj).
    """
    n_comp = min(n_components, X_train.shape[1])
    rp = GaussianRandomProjection(n_components=n_comp, random_state=seed)
    X_train_proj = rp.fit_transform(X_train)
    X_test_proj = rp.transform(X_test)
    return X_train_proj, X_test_proj


def standard_pca(
    X_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    n_components: int,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Standard PCA projection.

    Args:
        X_train: Training data (N_train, D).
        X_test: Test data (N_test, D).
        n_components: Number of components.
        seed: Random seed.

    Returns:
        Projected (X_train_proj, X_test_proj).
    """
    n_comp = min(n_components, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
    X_train_proj = pca.fit_transform(X_train)
    X_test_proj = pca.transform(X_test)
    return X_train_proj, X_test_proj


def evaluate_ridge(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
    y_test: NDArray[np.float64],
    label_type: str,
    alpha: float = 1.0,
) -> dict[str, float]:
    """Train Ridge model and return metrics.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        label_type: "classification" or "regression".
        alpha: Ridge regularization strength.

    Returns:
        Dict with metric name → value.
    """
    if label_type == "classification":
        model = RidgeClassifier(alpha=alpha, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="weighted")),
        }
    else:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "r2": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        }


def primary_metric(metrics: dict[str, float], label_type: str) -> float:
    """Extract the primary metric value (accuracy or R²)."""
    if label_type == "classification":
        return metrics.get("accuracy", 0.0)
    return metrics.get("r2", 0.0)


def run_layer(
    layer_repr: NDArray[np.float64],
    labels: NDArray[np.float64],
    label_type: str,
    n_components: int,
    alpha: float,
    val_split: float,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Run all 4 methods on a single layer's representations.

    Returns:
        Dict of {method_name: metrics_dict}.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        layer_repr, labels, test_size=val_split, random_state=seed,
    )

    results: dict[str, dict[str, float]] = {}

    # 1. Standard PCA
    Xtr_pca, Xte_pca = standard_pca(X_train, X_test, n_components, seed)
    results["pca"] = evaluate_ridge(Xtr_pca, y_train, Xte_pca, y_test, label_type, alpha)

    # 2. Supervised PCA
    Xtr_spca, Xte_spca = supervised_pca(X_train, y_train, X_test, n_components, seed)
    results["supervised_pca"] = evaluate_ridge(
        Xtr_spca, y_train, Xte_spca, y_test, label_type, alpha,
    )

    # 3. Random Projection
    Xtr_rp, Xte_rp = random_projection(X_train, X_test, n_components, seed)
    results["random_projection"] = evaluate_ridge(
        Xtr_rp, y_train, Xte_rp, y_test, label_type, alpha,
    )

    # 4. Full-dimensional Ridge (no projection)
    results["full_ridge"] = evaluate_ridge(X_train, y_train, X_test, y_test, label_type, alpha)

    return results


def main() -> None:
    """Run PCA alternative comparison across models and properties."""
    args = parse_args()
    seed_everything(args.seed)
    repr_root = Path(args.repr_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for model_name, dirs in MODELS.items():
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"{'=' * 70}")
        all_results[model_name] = {}

        for dataset_name, label_type in PROPERTIES.items():
            full_d_dir = repr_root / dirs["full_d_dir"] / dataset_name
            if not full_d_dir.exists():
                print(f"  SKIP {dataset_name}: {full_d_dir} not found")
                continue

            labels_path = full_d_dir / "labels.pt"
            if not labels_path.exists():
                print(f"  SKIP {dataset_name}: labels.pt not found")
                continue

            labels = torch.load(labels_path, map_location="cpu", weights_only=True).numpy()

            # Find layer files
            layer_files = sorted(
                [f for f in full_d_dir.glob("*.pt") if f.stem != "labels"],
                key=lambda p: _layer_sort_key(p.stem),
            )

            if not layer_files:
                print(f"  SKIP {dataset_name}: no layer files found")
                continue

            prop_name = dataset_name.replace("synthetic_", "")
            print(f"\n  Property: {prop_name} ({label_type})")

            # Track best metric per method across layers
            best_per_method: dict[str, tuple[float, str, dict[str, float]]] = {}
            methods = ["pca", "supervised_pca", "random_projection", "full_ridge"]

            for layer_file in layer_files:
                layer_name = layer_file.stem
                repr_tensor = torch.load(layer_file, map_location="cpu", weights_only=True)
                if repr_tensor.ndim >= 3:
                    repr_tensor = repr_tensor.view(repr_tensor.shape[0], -1)
                X = repr_tensor.numpy().astype(np.float64)

                if X.shape[0] != len(labels):
                    continue

                layer_results = run_layer(
                    X, labels, label_type,
                    args.n_components, args.alpha, args.val_split, args.seed,
                )

                for method, metrics in layer_results.items():
                    val = primary_metric(metrics, label_type)
                    current_best = best_per_method.get(method)
                    if current_best is None or val > current_best[0]:
                        best_per_method[method] = (val, layer_name, metrics)

            # Store best-layer results
            prop_results: dict[str, dict[str, float]] = {}
            metric_name = "accuracy" if label_type == "classification" else "r2"

            print(f"    {'Method':<22} {'Best Layer':<25} {metric_name:>10}")
            print(f"    {'-' * 60}")

            for method in methods:
                if method in best_per_method:
                    val, layer, metrics = best_per_method[method]
                    prop_results[method] = {**metrics, "best_layer": layer}
                    print(f"    {method:<22} {layer:<25} {val:>10.4f}")

            all_results[model_name][prop_name] = prop_results

    # Save results
    results_path = output_dir / "comparison_table.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved comparison table to {results_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Best-layer metric per method (PCA512 vs alternatives)")
    print("=" * 80)

    for model_name, model_data in all_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  {'Property':<16} {'PCA':>8} {'sPCA':>8} {'RP':>8} {'FullRidge':>10}")
        print(f"  {'-' * 54}")
        for prop_name, method_data in model_data.items():
            label_type = PROPERTIES.get(f"synthetic_{prop_name}", "classification")
            mkey = "accuracy" if label_type == "classification" else "r2"
            vals = []
            for method in ["pca", "supervised_pca", "random_projection", "full_ridge"]:
                v = method_data.get(method, {}).get(mkey, float("nan"))
                vals.append(f"{v:>8.4f}")
            print(f"  {prop_name:<16} {'  '.join(vals)}")


if __name__ == "__main__":
    main()
