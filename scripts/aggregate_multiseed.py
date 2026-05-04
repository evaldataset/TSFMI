"""Aggregate multi-seed probing results and compute mean ± std.

Reads probe results from:
  outputs/probes/ (seed 42, original)
  outputs/probes_seed123/ (seed 123)
  outputs/probes_seed456/ (seed 456)

Produces:
  outputs/analysis/multiseed_results.json
  Console table with mean ± std
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED_DIRS = {
    42: Path("outputs/eval"),
    123: Path("outputs/probes_seed123"),
    456: Path("outputs/probes_seed456"),
}

MODEL_MAP = {
    "moment_pca512": "MOMENT-PCA512",
    "chronos": "Chronos-Bolt",
    "patchtst_pretrained": "PatchTST-Pre",
    "gpt4ts_pca512": "GPT4TS-PCA512",
}

MODELS = ["moment_pca512", "chronos", "patchtst_pretrained", "gpt4ts_pca512"]
PROPS = [
    "synthetic_trend",
    "synthetic_frequency",
    "synthetic_stationarity",
    "synthetic_anomaly",
    "synthetic_change_point",
]
OUTPUT_DIR = Path("outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_best_accuracy_from_eval(model: str, prop: str) -> float | None:
    """Get best accuracy from original eval directory (seed 42)."""
    # Try with and without synthetic_ prefix
    eval_name_no_prefix = f"{model}_{prop.replace('synthetic_', '')}"
    eval_name_with_prefix = f"{model}_{prop}"
    for eval_name in [eval_name_no_prefix, eval_name_with_prefix]:
        eval_path = SEED_DIRS[42] / eval_name / "layer_metrics.json"
        if eval_path.exists():
            break
    else:
        return None
    data = json.loads(eval_path.read_text())
    metric = "val_r2" if "seasonality" in prop else "val_accuracy"
    best = max(entry.get(metric, -999) for entry in data)
    return best


def get_best_accuracy_from_probes(seed_dir: Path, model: str, prop: str) -> float | None:
    """Get best accuracy from probe summary (seeds 123, 456)."""
    probe_dir = seed_dir / f"{model}_{prop}_linear"
    summary = probe_dir / "summary.json"
    if not summary.exists():
        return None
    data = json.loads(summary.read_text())
    # Summary has 'layers' dict with per-layer results
    metric = "val_r2" if "seasonality" in prop else "val_accuracy"
    if "layers" in data:
        vals = [v.get(metric, -999) for v in data["layers"].values()]
        return max(vals) if vals else None
    if "best_val_accuracy" in data:
        return data["best_val_accuracy"]
    if "best_val_r2" in data:
        return data["best_val_r2"]
    return None


def main() -> None:
    results: dict[str, dict[str, dict[int, float | None]]] = defaultdict(lambda: defaultdict(dict))

    for model in MODELS:
        for prop in PROPS:
            # Seed 42 (original)
            val_42 = get_best_accuracy_from_eval(model, prop)
            results[model][prop][42] = val_42

            # Seeds 123, 456
            for seed in [123, 456]:
                val = get_best_accuracy_from_probes(SEED_DIRS[seed], model, prop)
                results[model][prop][seed] = val

    # Print table
    print(
        "\n"
        f"{'Model':<20} | {'Property':<15} | {'Seed42':>8} | {'Seed123':>8} | "
        f"{'Seed456':>8} | {'Mean±Std':>12}"
    )
    print("-" * 85)

    summary = {}
    for model in MODELS:
        for prop in PROPS:
            vals = []
            for seed in [42, 123, 456]:
                v = results[model][prop].get(seed)
                if v is not None:
                    vals.append(v)

            v42 = results[model][prop].get(42)
            v123 = results[model][prop].get(123)
            v456 = results[model][prop].get(456)

            s42 = f"{v42:.3f}" if v42 is not None else "---"
            s123 = f"{v123:.3f}" if v123 is not None else "---"
            s456 = f"{v456:.3f}" if v456 is not None else "---"

            if len(vals) >= 2:
                mean = np.mean(vals)
                std = np.std(vals)
                mean_std = f"{mean:.3f}±{std:.3f}"
            else:
                mean_std = "---"

            prop_short = prop.replace("synthetic_", "")
            print(
                f"{MODEL_MAP[model]:<20} | {prop_short:<15} | {s42:>8} | {s123:>8} | "
                f"{s456:>8} | {mean_std:>12}"
            )

            summary[f"{model}/{prop}"] = {
                "seeds": {str(s): v for s, v in results[model][prop].items()},
                "mean": float(np.mean(vals)) if vals else None,
                "std": float(np.std(vals)) if len(vals) >= 2 else None,
                "n_seeds": len(vals),
            }
        print("-" * 85)

    with open(OUTPUT_DIR / "multiseed_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR / 'multiseed_results.json'}")


if __name__ == "__main__":
    main()
