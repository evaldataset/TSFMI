"""Hard variant benchmark: compare easy vs hard probing across all models.

Ceiling saturation (5/6 properties at 100%) obscures model differences. This
script evaluates all models on hard synthetic variants (weaker slopes, overlapping
frequencies, subtle anomalies, etc.) and produces an easy-vs-hard comparison table.

Models with genuinely robust representations maintain accuracy on hard variants;
those exploiting trivial signals collapse.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_hard_variant_benchmark.py
    PYTHONPATH=. .venv/bin/python scripts/run_hard_variant_benchmark.py --device cuda:1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.utils.seed import seed_everything

MODELS: dict[str, str] = {
    "MOMENT": "moment_pca512",
    "Chronos": "chronos",
    "PatchTST": "patchtst_pretrained",
    "GPT4TS": "gpt4ts_pca512",
    "Timer": "timer_meanpool",
    "TimesFM": "timesfm_meanpool",
    "Moirai": "moirai_meanpool",
}

HARD_PROPERTIES = ["trend", "frequency", "stationarity", "anomaly", "change_point"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark easy vs hard synthetic probing across models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repr_root", type=str, default="outputs/representations")
    parser.add_argument("--output_dir", type=str, default="outputs/hard_variant_benchmark")
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


def probe_best_layer(
    repr_dir: Path,
    alpha: float,
    val_split: float,
    seed: int,
) -> dict[str, float]:
    """Train Ridge classifier on each layer, return best-layer metrics.

    Args:
        repr_dir: Directory with layer .pt files and labels.pt.
        alpha: Ridge regularization.
        val_split: Fraction held out for evaluation.
        seed: Random seed.

    Returns:
        Dict with best accuracy, f1, and best_layer name.
        Empty dict if data is missing.
    """
    labels_path = repr_dir / "labels.pt"
    if not labels_path.exists():
        return {}

    labels = torch.load(labels_path, map_location="cpu", weights_only=True).numpy()
    if labels.ndim > 1:
        labels = labels.squeeze()

    layer_files = sorted(
        [f for f in repr_dir.glob("*.pt") if f.stem != "labels"],
        key=lambda p: _layer_sort_key(p.stem),
    )
    if not layer_files:
        return {}

    best_acc = -1.0
    best_layer = ""
    best_f1 = 0.0

    for layer_file in layer_files:
        repr_tensor = torch.load(layer_file, map_location="cpu", weights_only=True)
        if repr_tensor.ndim >= 3:
            repr_tensor = repr_tensor.view(repr_tensor.shape[0], -1)
        X = repr_tensor.numpy().astype(np.float32)

        if X.shape[0] != len(labels):
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=val_split, random_state=seed,
        )

        clf = RidgeClassifier(alpha=alpha, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="weighted"))

        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            best_layer = layer_file.stem

    if not best_layer:
        return {}

    return {"accuracy": best_acc, "f1": best_f1, "best_layer": best_layer}


def main() -> None:
    """Run easy vs hard comparison across all models and properties."""
    args = parse_args()
    seed_everything(args.seed)
    repr_root = Path(args.repr_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    missing: list[str] = []

    for model_name, repr_key in MODELS.items():
        results[model_name] = {}

        for prop in HARD_PROPERTIES:
            easy_dir = repr_root / repr_key / f"synthetic_{prop}"
            hard_dir = repr_root / repr_key / f"synthetic_{prop}_hard"

            easy_result: dict[str, float] = {}
            hard_result: dict[str, float] = {}

            if easy_dir.exists():
                easy_result = probe_best_layer(
                    easy_dir, args.alpha, args.val_split, args.seed,
                )
            else:
                missing.append(f"{model_name}/{prop}/easy")

            if hard_dir.exists():
                hard_result = probe_best_layer(
                    hard_dir, args.alpha, args.val_split, args.seed,
                )
            else:
                missing.append(f"{model_name}/{prop}/hard")

            drop = 0.0
            if easy_result and hard_result:
                drop = easy_result["accuracy"] - hard_result["accuracy"]

            results[model_name][prop] = {
                "easy": easy_result,
                "hard": hard_result,
                "drop": drop,
            }

    # Save
    results_path = output_dir / "easy_vs_hard_table.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {results_path}")

    # Print summary
    print("\n" + "=" * 95)
    print("EASY vs HARD Probing Accuracy (best layer)")
    print("=" * 95)

    header = f"{'Model':<10}"
    for prop in HARD_PROPERTIES:
        header += f" | {prop:>14} (E→H)"
    print(header)
    print("-" * 95)

    for model_name in MODELS:
        row = f"{model_name:<10}"
        for prop in HARD_PROPERTIES:
            data = results[model_name][prop]
            e = data["easy"].get("accuracy", float("nan")) if data["easy"] else float("nan")
            h = data["hard"].get("accuracy", float("nan")) if data["hard"] else float("nan")
            if np.isnan(h):
                row += f" | {e:>5.1%}→  N/A   "
            else:
                drop = data["drop"]
                row += f" | {e:>5.1%}→{h:>5.1%} ({drop:>+5.1%})"
        print(row)

    if missing:
        print(f"\nMissing representations ({len(missing)}):")
        for m in missing:
            print(f"  - {m}")
        print("\nTo extract missing hard representations, run:")
        print("  PYTHONPATH=. .venv/bin/python scripts/extract_representations.py \\")
        print("    --model <model> --dataset synthetic_<prop>_hard --layers all \\")
        print("    --output_dir outputs/representations/")


if __name__ == "__main__":
    main()
