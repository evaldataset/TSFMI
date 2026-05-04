"""Causal controls: necessity, sufficiency, and random-subspace analysis.

Triangulates causal claims about concept encoding by testing three conditions:
  1. NECESSITY (LEACE): Erase concept subspace → accuracy drops
  2. SUFFICIENCY: Project onto concept subspace only → accuracy preserved
  3. RANDOM CONTROL: Erase random subspace of same dim → accuracy barely drops

If concept subspace is both necessary AND sufficient, and random erasure has
minimal effect, then the concept is genuinely encoded in that linear subspace.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_causal_controls.py
    PYTHONPATH=. .venv/bin/python scripts/run_causal_controls.py --device cuda:1
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.utils.seed import seed_everything

MODEL_REPR_KEYS: dict[str, str] = {
    "MOMENT": "moment_pca512",
    "Chronos": "chronos",
    "PatchTST": "patchtst_pretrained",
    "GPT4TS": "gpt4ts_pca512",
    "Timer": "timer_meanpool",
    "TimesFM": "timesfm_meanpool",
    "Moirai": "moirai_meanpool",
}

PROPERTIES = ["trend", "stationarity", "frequency", "change_point"]

# Use best layer per model (representative middle or best probing layer)
BEST_LAYERS: dict[str, int] = {
    "MOMENT": 11,
    "Chronos": 2,
    "PatchTST": 1,
    "GPT4TS": 5,
    "Timer": 3,
    "TimesFM": 25,
    "Moirai": 2,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Causal controls: necessity, sufficiency, random subspace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repr_root", type=str, default="outputs/representations")
    parser.add_argument("--output_dir", type=str, default="outputs/causal_controls")
    parser.add_argument("--n_random_seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def find_best_layer_file(repr_dir: Path, target_idx: int) -> Path | None:
    """Find the layer file matching the target index."""
    for f in repr_dir.glob("*.pt"):
        if f.stem == "labels":
            continue
        parts = f.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) == target_idx:
            return f
    return None


def leace_erase(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    X_test: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply LEACE concept erasure, fitting only on train data.

    Args:
        X_train: Training features (N_train, D).
        y_train: Training labels (N_train,).
        X_test: Test features (N_test, D).

    Returns:
        Tuple of (erased_train, erased_test).
    """
    concept_erasure = importlib.import_module("concept_erasure")
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    fitter = concept_erasure.LeaceFitter.fit(X_tr_t, y_tr_t)
    eraser = fitter.eraser
    erased_train = eraser(X_tr_t).numpy()
    erased_test = eraser(torch.tensor(X_test, dtype=torch.float32)).numpy()
    return erased_train, erased_test


def extract_concept_subspace(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    n_directions: int = 1,
) -> NDArray[np.float64]:
    """Extract concept-aligned directions via LDA (independent of probing classifier).

    Uses Linear Discriminant Analysis to find directions that maximally separate
    classes, avoiding circularity with the Ridge classifier used for probing.

    Args:
        X_train: Training features (N, D).
        y_train: Training labels (N,).
        n_directions: Number of LDA discriminant directions.

    Returns:
        Orthonormal basis (n_directions, D).
    """
    n_classes = len(np.unique(y_train))
    n_comp = min(n_directions, n_classes - 1, X_train.shape[1])
    n_comp = max(n_comp, 1)
    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    lda.fit(X_train, y_train)
    # scalings_ has shape (D, n_comp) — each column is a discriminant direction
    directions = lda.scalings_[:, :n_comp].T  # (n_comp, D)
    # Orthonormalize
    Q, _R = np.linalg.qr(directions.T)
    return Q[:, :n_comp].T.astype(np.float64)  # (n_comp, D)


def project_onto_subspace(
    X: NDArray[np.float64],
    subspace: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project representations onto concept subspace (sufficiency test).

    Args:
        X: Features (N, D).
        subspace: Orthonormal basis (k, D).

    Returns:
        Projected features (N, D) — only components in subspace preserved.
    """
    # P = V^T V (projection matrix)
    return X @ subspace.T @ subspace


def erase_random_subspace(
    X: NDArray[np.float64],
    n_directions: int,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Erase random directions of same dimensionality as concept subspace.

    Args:
        X: Features (N, D).
        n_directions: Number of random directions to erase.
        seed: Random seed.

    Returns:
        Features with random subspace removed (N, D).
    """
    rng = np.random.default_rng(seed)
    D = X.shape[1]
    # Generate random orthonormal directions
    random_matrix = rng.standard_normal((n_directions, D))
    Q, _R = np.linalg.qr(random_matrix.T)
    random_dirs = Q[:, :n_directions].T  # (n_directions, D)
    # Remove: X_erased = X - X @ V^T @ V
    return X - X @ random_dirs.T @ random_dirs


def run_causal_analysis(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    val_split: float,
    seed: int,
    n_random_seeds: int,
) -> dict[str, float]:
    """Run necessity/sufficiency/random analysis for one representation.

    Args:
        X: Features (N, D).
        y: Labels (N,).
        val_split: Validation fraction.
        seed: Random seed.
        n_random_seeds: Number of random subspace seeds to average.

    Returns:
        Dict with original_acc, leace_acc, subspace_acc, random_erased_acc.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=val_split, random_state=seed,
    )

    # Original accuracy
    clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf.fit(X_train, y_train)
    original_acc = float(accuracy_score(y_test, clf.predict(X_test)))

    # Number of concept directions = min(n_classes - 1, D)
    n_classes = len(np.unique(y))
    n_dirs = min(n_classes - 1, X.shape[1])
    n_dirs = max(n_dirs, 1)

    # 1. NECESSITY: LEACE erasure (fit on train only, apply to both)
    X_er_train, X_er_test = leace_erase(X_train, y_train, X_test)
    clf_er = RidgeClassifier(alpha=1.0, class_weight="balanced")
    clf_er.fit(X_er_train, y_train)
    leace_acc = float(accuracy_score(y_test, clf_er.predict(X_er_test)))

    # 2. SUFFICIENCY: project onto LDA concept subspace, re-classify with Logistic
    #    (LDA directions are independent of the Ridge probe → no circularity)
    subspace = extract_concept_subspace(X_train, y_train, n_directions=n_dirs)
    X_sub_train = project_onto_subspace(X_train, subspace)
    X_sub_test = project_onto_subspace(X_test, subspace)
    clf_sub = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf_sub.fit(X_sub_train, y_train)
    subspace_acc = float(accuracy_score(y_test, clf_sub.predict(X_sub_test)))

    # 3. RANDOM CONTROL: erase random subspace (average over seeds)
    random_accs = []
    for rs in range(n_random_seeds):
        X_rand = erase_random_subspace(X, n_directions=n_dirs, seed=seed + rs)
        X_rand_train, X_rand_test, _, _ = train_test_split(
            X_rand, y, test_size=val_split, random_state=seed,
        )
        clf_rand = RidgeClassifier(alpha=1.0, class_weight="balanced")
        clf_rand.fit(X_rand_train, y_train)
        random_accs.append(float(accuracy_score(y_test, clf_rand.predict(X_rand_test))))

    return {
        "original_acc": original_acc,
        "leace_erased_acc": leace_acc,
        "subspace_only_acc": subspace_acc,
        "random_erased_acc_mean": float(np.mean(random_accs)),
        "random_erased_acc_std": float(np.std(random_accs)),
        "n_concept_directions": n_dirs,
        "necessity_drop": original_acc - leace_acc,
        "sufficiency_preserved": subspace_acc,
        "random_drop": original_acc - float(np.mean(random_accs)),
        "specificity": (original_acc - leace_acc) - (original_acc - float(np.mean(random_accs))),
    }


def main() -> None:
    """Run causal controls across models and properties."""
    args = parse_args()
    seed_everything(args.seed)
    repr_root = Path(args.repr_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for model_name, repr_key in MODEL_REPR_KEYS.items():
        all_results[model_name] = {}
        best_idx = BEST_LAYERS[model_name]

        for prop in PROPERTIES:
            repr_dir = repr_root / repr_key / f"synthetic_{prop}"
            if not repr_dir.exists():
                print(f"SKIP {model_name}/{prop}: {repr_dir} not found")
                continue

            layer_file = find_best_layer_file(repr_dir, best_idx)
            if layer_file is None:
                print(f"SKIP {model_name}/{prop}: layer {best_idx} not found")
                continue

            labels = torch.load(
                repr_dir / "labels.pt", map_location="cpu", weights_only=True,
            ).numpy()
            if labels.ndim > 1:
                labels = labels.squeeze()

            reprs = torch.load(layer_file, map_location="cpu", weights_only=True)
            if reprs.ndim >= 3:
                reprs = reprs.view(reprs.shape[0], -1)
            X = reprs.numpy().astype(np.float32)

            print(f"\n{model_name} / {prop} (layer {best_idx}, dim={X.shape[1]})")
            result = run_causal_analysis(
                X, labels, args.val_split, args.seed, args.n_random_seeds,
            )
            all_results[model_name][prop] = result

            print(
                f"  Original: {result['original_acc']:.3f}  "
                f"LEACE: {result['leace_erased_acc']:.3f}  "
                f"Subspace: {result['subspace_only_acc']:.3f}  "
                f"Random: {result['random_erased_acc_mean']:.3f}"
            )

    # Save
    results_path = output_dir / "necessity_sufficiency.json"
    results_path.write_text(json.dumps(all_results, indent=2))

    # Summary table
    print("\n" + "=" * 90)
    print("CAUSAL CONTROLS SUMMARY")
    print("=" * 90)
    print(
        f"{'Model':<10} {'Property':<14} {'Orig':>6} {'LEACE':>7} "
        f"{'Subsp':>7} {'Random':>7} {'Specif':>7}"
    )
    print("-" * 90)

    for model_name in MODEL_REPR_KEYS:
        for prop in PROPERTIES:
            r = all_results.get(model_name, {}).get(prop)
            if r is None:
                continue
            print(
                f"{model_name:<10} {prop:<14} "
                f"{r['original_acc']:>6.3f} {r['leace_erased_acc']:>7.3f} "
                f"{r['subspace_only_acc']:>7.3f} "
                f"{r['random_erased_acc_mean']:>7.3f} "
                f"{r['specificity']:>7.3f}"
            )

    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
