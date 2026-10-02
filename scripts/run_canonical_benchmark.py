"""Canonical benchmark rerun with clean protocol.

This script implements the canonical protocol described in PLAN.md T1.1:
    - train / val / test 3-way split (60/20/20)
    - best-layer selection on val only
    - final reporting on held-out test only
    - PCA fit on train split only: either upstream (legacy default), or inside the
      per-seed pipeline with --pca-in-pipeline N (StandardScaler -> PCA(N) -> estimator,
      fit on the training rows of each split seed only; pass full-dimensional
      representations)
    - Scaler inside CV loop (via Pipeline)
    - Matched estimator (sklearn LogisticRegression for classification,
      Ridge for regression) for both baseline and model probes
    - 5 seeds with 95% CI (bootstrap)

Usage:
    PYTHONPATH=. python scripts/run_canonical_benchmark.py \
        --model moment_pca512 \
        --property trend \
        --output_dir outputs/canonical/moment_pca512_trend/

Outputs:
    <output_dir>/canonical_results.json with:
        - per-seed val accuracies (for best-layer selection)
        - per-seed test accuracy (for final reporting)
        - best layer (selected on val)
        - bootstrap 95% CI on mean test accuracy
        - metadata: split sizes, seeds, estimator family
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2, 3, 4]
N_BOOTSTRAP = 1000
VAL_RATIO = 0.20
TEST_RATIO = 0.20


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canonical benchmark rerun")
    p.add_argument("--representations_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--property", type=str, required=True)
    p.add_argument(
        "--task_type",
        choices=["classification", "regression"],
        required=True,
    )
    p.add_argument(
        "--pca-in-pipeline",
        dest="pca_in_pipeline",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Insert PCA(N) after StandardScaler inside the per-seed probe pipeline, so PCA is "
            "fit on the training split of each split seed only. Pass full-dimensional "
            "(flattened) representations. Layers with <= N features skip PCA. Default: off "
            "(legacy behaviour; any reduction was applied upstream)."
        ),
    )
    return p.parse_args(argv)


def three_way_split(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Deterministic 60/20/20 train/val/test split."""
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return (
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        X[test_idx],
        y[test_idx],
    )


def build_pipeline(
    task_type: str,
    seed: int,
    n_features: int,
    pca_components: int | None = None,
) -> Pipeline:
    """Build the matched-estimator probe pipeline.

    Args:
        task_type: "classification" (LogisticRegression) or "regression" (Ridge).
        seed: Split seed; seeds the estimator and, if used, the PCA.
        n_features: Input dimensionality of the layer.
        pca_components: If set and smaller than ``n_features``, a PCA with this many
            components is inserted after the StandardScaler. Everything in the pipeline is
            fit on the training rows passed to ``fit`` only.

    Returns:
        Unfitted sklearn pipeline.
    """
    steps: list = [StandardScaler()]
    if pca_components is not None and n_features > pca_components:
        steps.append(PCA(n_components=pca_components, random_state=seed))
    if task_type == "classification":
        steps.append(LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs"))
    else:
        steps.append(Ridge(alpha=1.0))
    return make_pipeline(*steps)


def train_and_score(
    X_tr: NDArray[np.float64],
    y_tr: NDArray[np.float64],
    X_eval: NDArray[np.float64],
    y_eval: NDArray[np.float64],
    task_type: str,
    seed: int,
    pca_components: int | None = None,
) -> float:
    """Train matched-estimator probe on the train split and return its score on eval.

    With ``pca_components=None`` (default) the pipeline is StandardScaler -> estimator,
    identical to the original canonical protocol.
    """
    pipe = build_pipeline(task_type, seed, X_tr.shape[1], pca_components)
    if task_type == "classification":
        pipe.fit(X_tr, y_tr.astype(np.int64))
        return float(accuracy_score(y_eval.astype(np.int64), pipe.predict(X_eval)))
    pipe.fit(X_tr, y_tr)
    return float(r2_score(y_eval, pipe.predict(X_eval)))


def bootstrap_ci(
    values: list[float], n_bootstrap: int = N_BOOTSTRAP, ci: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    boot = np.array(
        [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_bootstrap)]
    )
    alpha = (1 - ci) / 2
    return (
        float(arr.mean()),
        float(np.percentile(boot, alpha * 100)),
        float(np.percentile(boot, (1 - alpha) * 100)),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    pca_n = args.pca_in_pipeline
    repr_dir = Path(args.representations_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = torch.load(repr_dir / "labels.pt", map_location="cpu", weights_only=True).numpy()
    layer_files = sorted(
        f for f in repr_dir.glob("*.pt") if f.stem not in ("labels", "metadata")
    )
    if not layer_files:
        raise FileNotFoundError(f"No layer .pt files in {repr_dir}")

    per_seed_test: list[float] = []
    per_seed_best_layer: list[str] = []
    per_seed_val: list[float] = []

    for seed in SEEDS:
        best_val = -np.inf
        best_test_score: float | None = None
        best_layer = layer_files[0].stem

        for lf in layer_files:
            t = torch.load(lf, map_location="cpu", weights_only=True)
            # Collapse all non-sample dims by flattening (N, ...) -> (N, D)
            if t.ndim > 2:
                t = t.reshape(t.shape[0], -1)
            if t.ndim < 2:
                t = t.unsqueeze(-1)
            X = t.numpy().astype(np.float64)
            if X.shape[1] == 0:
                continue
            X_tr, y_tr, X_val, y_val, X_te, y_te = three_way_split(X, labels, seed)

            val_score = train_and_score(
                X_tr, y_tr, X_val, y_val, args.task_type, seed, pca_components=pca_n
            )
            if val_score > best_val:
                best_val = val_score
                best_test_score = train_and_score(
                    X_tr, y_tr, X_te, y_te, args.task_type, seed, pca_components=pca_n
                )
                best_layer = lf.stem

        if best_test_score is None:
            print(f"  seed={seed} WARN: no usable layer, using first file shape fallback")
            best_test_score = float("nan")
            best_val = float("nan")
        per_seed_val.append(best_val)
        per_seed_test.append(best_test_score)
        per_seed_best_layer.append(best_layer)
        print(
            f"  seed={seed} best_layer={best_layer} val={best_val:.4f} test={best_test_score:.4f}"
        )

    mean, ci_low, ci_high = bootstrap_ci(per_seed_test)
    result = {
        "representations_dir": str(repr_dir),
        "property": args.property,
        "task_type": args.task_type,
        "seeds": SEEDS,
        "val_score_per_seed": [round(v, 4) for v in per_seed_val],
        "test_score_per_seed": [round(v, 4) for v in per_seed_test],
        "best_layer_per_seed": per_seed_best_layer,
        "test_mean": round(mean, 4),
        "test_ci95_low": round(ci_low, 4),
        "test_ci95_high": round(ci_high, 4),
        "protocol": {
            "split": "60/20/20 train/val/test",
            "selection": "best layer chosen on val only",
            "reporting": "test score only",
            "estimator": (
                "sklearn LogisticRegression"
                if args.task_type == "classification"
                else "sklearn Ridge"
            ),
            "n_bootstrap": N_BOOTSTRAP,
        },
    }
    if pca_n is not None:
        result["protocol"]["pca_in_pipeline"] = {
            "n_components": pca_n,
            "pipeline": "StandardScaler -> PCA -> estimator",
            "fit_scope": "training split of each split seed only",
            "input": "flattened full-dimensional layer representations",
            "skipped_when": f"layer dim <= {pca_n}",
        }
    out_path = out_dir / "canonical_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[{args.property}] test mean={mean:.4f} 95%CI=[{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
