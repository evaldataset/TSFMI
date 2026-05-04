"""Full-dimensional probing using Ridge regression (scikit-learn).

Avoids OOM issues with high-dimensional representations (MOMENT 65536D, GPT4TS 24576D)
by using sklearn's Ridge instead of PyTorch linear probe.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_fulld_probe.py \
        --representations_dir outputs/representations/moment/synthetic_seasonality/ \
        --output_dir outputs/fulld_probes/moment_seasonality/ \
        --alpha 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import train_test_split

from src.utils.seed import seed_everything


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Full-dimensional probing with Ridge regression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--representations_dir",
        type=str,
        required=True,
        help="Directory containing {layer}.pt, labels.pt, metadata.json.",
    )
    _ = parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results.",
    )
    _ = parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization.")
    _ = parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    _ = parser.add_argument("--val_split", type=float, default=0.2, help="Test fraction.")
    _ = parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to use (subsamples if needed).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Run full-dimensional Ridge probing across layers."""
    args = parse_args()
    seed_everything(args.seed)
    repr_dir = Path(args.representations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = repr_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    label_type = metadata.get("label_type", "classification")

    labels = torch.load(repr_dir / "labels.pt", map_location="cpu", weights_only=True).numpy()
    layer_files = sorted(f for f in repr_dir.glob("*.pt") if f.stem != "labels")

    if args.max_samples and len(labels) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(labels), size=args.max_samples, replace=False)
        idx.sort()
        labels = labels[idx]
        subsample_idx = idx
    else:
        subsample_idx = None

    print(f"Label type: {label_type}, Samples: {len(labels)}, Layers: {len(layer_files)}")
    print(f"Ridge alpha: {args.alpha}")
    print()

    all_results: dict[str, dict[str, float]] = {}

    for layer_file in layer_files:
        layer_name = layer_file.stem
        repr_tensor = torch.load(layer_file, map_location="cpu", weights_only=True)
        if repr_tensor.ndim >= 3:
            repr_tensor = repr_tensor.view(repr_tensor.shape[0], -1)
        X = repr_tensor.numpy().astype(np.float32)

        if subsample_idx is not None:
            X = X[subsample_idx]

        if X.shape[0] != len(labels):
            print(f"[{layer_name}] Sample mismatch, skipping")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            labels,
            test_size=args.val_split,
            random_state=args.seed,
        )

        input_dim = X.shape[1]

        if label_type == "classification":
            model = RidgeClassifier(alpha=args.alpha, class_weight="balanced")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            metrics = {"accuracy": float(acc), "f1": float(f1)}
            print(f"[{layer_name}] dim={input_dim:>6d} acc={acc:.4f} f1={f1:.4f}")
        else:
            model = Ridge(alpha=args.alpha)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = float(np.mean(np.abs(y_test - y_pred)))
            metrics = {"r2": float(r2), "mae": mae}
            print(f"[{layer_name}] dim={input_dim:>6d} r2={r2:.4f} mae={mae:.4f}")

        all_results[layer_name] = metrics

    results_path = output_dir / "fulld_probe_results.json"
    _ = results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
