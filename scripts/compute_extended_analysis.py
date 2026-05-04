"""Compute extended analyses from existing experiment data.

Produces:
  A2: Full selectivity table (Linear - MLP Control) for all models
  B1: Linear vs MLP accuracy comparison
  C2: Layer-wise pattern typology (flat/increasing/decreasing/U-shape)
  C4: Cross-LEACE asymmetry quantification
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EVAL_DIR = Path("outputs/eval")
CROSS_LEACE_DIR = Path("outputs/cross_leace")
OUTPUT_DIR = Path("outputs/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical model names for paper
MODEL_MAP = {
    "moment_pca512": "MOMENT-PCA512",
    "chronos": "Chronos-Bolt",
    "patchtst_pretrained": "PatchTST-Pre",
    "gpt4ts_pca512": "GPT4TS-PCA512",
    "patchtst": "PatchTST-Rnd",
    "itransformer": "iTransf.-Rnd",
    "moment": "MOMENT-FullD",
}

PRETRAINED_MODELS = ["moment_pca512", "chronos", "patchtst_pretrained", "gpt4ts_pca512"]
BASELINE_MODELS = ["patchtst", "itransformer"]
SYNTHETIC_PROPS = ["trend", "seasonality", "frequency", "stationarity", "anomaly", "change_point"]

MetricEntry = dict[str, float | int | str | None]
SelectivityEntry = dict[str, float | None]
PatternEntry = dict[str, str | list[float] | float | None]
CrossLeaceEntry = dict[str, float | None]
AsymmetryEntry = dict[str, float]


def _metric_number(value: float | int | str | None, default: float = -999.0) -> float:
    if isinstance(value, int | float):
        return float(value)
    return default


def load_eval_results(eval_name: str) -> list[MetricEntry] | None:
    """Load layer_metrics.json from eval directory."""
    path = EVAL_DIR / eval_name / "layer_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_best_metric(
    layers: list[MetricEntry], metric: str = "val_accuracy"
) -> tuple[float, str, int]:
    """Return (best_value, best_layer_name, best_layer_index)."""
    best_val = -999.0
    best_name = ""
    best_idx = 0
    for i, entry in enumerate(layers):
        val = _metric_number(entry.get(metric, -999.0))
        if val > best_val:
            best_val = val
            best_name = str(entry.get("layer", ""))
            best_idx = i
    return best_val, best_name, best_idx


def classify_layer_pattern(values: list[float]) -> str:
    """Classify layer-wise accuracy pattern."""
    if len(values) < 3:
        return "too_few_layers"

    arr = np.array(values)
    n = len(arr)
    x = np.arange(n)

    # Linear regression slope
    slope = np.polyfit(x, arr, 1)[0]

    # Range of variation
    val_range = arr.max() - arr.min()

    if val_range < 0.02:
        return "flat"

    # Check for U-shape or inverted-U
    mid = n // 2
    first_half_mean = arr[:mid].mean()
    second_half_mean = arr[mid:].mean()
    middle_mean = arr[max(0, mid - 1) : min(n, mid + 2)].mean()

    if middle_mean < first_half_mean - 0.03 and middle_mean < second_half_mean - 0.03:
        return "U-shape"
    if middle_mean > first_half_mean + 0.03 and middle_mean > second_half_mean + 0.03:
        return "inverted-U"

    if slope > 0.005:
        return "increasing"
    elif slope < -0.005:
        return "decreasing"
    return "flat"


def compute_selectivity() -> dict[str, dict[str, SelectivityEntry]]:
    """A2 + B1: Compute selectivity and linear vs MLP comparison."""
    results = {}

    all_models = PRETRAINED_MODELS + BASELINE_MODELS
    for model in all_models:
        results[model] = {}
        for prop in SYNTHETIC_PROPS:
            # Try direct name first
            lin_data = load_eval_results(f"{model}_{prop}")
            if lin_data is None:
                lin_data = load_eval_results(f"{model}_synthetic_{prop}")

            # MLP control — not in eval dir, check probes dir
            # Actually eval dir has the results for both
            # The eval dirs don't have _linear or _mlp_control suffix
            # Let me check what exists
            metric = "val_r2" if prop == "seasonality" else "val_accuracy"

            if lin_data is not None:
                lin_best, lin_layer, _ = get_best_metric(lin_data, metric)
            else:
                lin_best = None

            # For MLP control, we need probe results
            # Check if we have separate eval for mlp_control
            # Actually probes dir has the trained models, eval dir has evaluation
            # Let's check probe dir for mlp_control results
            mlp_best = None
            probe_mlp_dir = Path(f"outputs/probes/{model}_{prop}_mlp_control")
            if probe_mlp_dir.exists():
                # Check for evaluation_results.json or metrics files
                for f in probe_mlp_dir.rglob("*.json"):
                    try:
                        data = json.loads(f.read_text())
                        if isinstance(data, dict):
                            val = data.get(
                                "val_accuracy", data.get("val_r2", data.get("accuracy", None))
                            )
                            numeric_val = _metric_number(val, default=np.nan)
                            if not np.isnan(numeric_val) and (
                                mlp_best is None or numeric_val > mlp_best
                            ):
                                mlp_best = numeric_val
                    except (json.JSONDecodeError, KeyError):
                        pass

            # Also check if there's a synthetic_ prefix variant
            if mlp_best is None:
                probe_mlp_dir2 = Path(f"outputs/probes/{model}_synthetic_{prop}_mlp_control")
                if probe_mlp_dir2.exists():
                    for f in probe_mlp_dir2.rglob("*.json"):
                        try:
                            data = json.loads(f.read_text())
                            if isinstance(data, dict):
                                val = data.get(
                                    "val_accuracy", data.get("val_r2", data.get("accuracy", None))
                                )
                                numeric_val = _metric_number(val, default=np.nan)
                                if not np.isnan(numeric_val) and (
                                    mlp_best is None or numeric_val > mlp_best
                                ):
                                    mlp_best = numeric_val
                        except (json.JSONDecodeError, KeyError):
                            pass

            results[model][prop] = {
                "linear": lin_best,
                "mlp": mlp_best,
                "selectivity": (lin_best - mlp_best)
                if lin_best is not None and mlp_best is not None
                else None,
            }

    return results


def compute_layer_patterns() -> dict[str, dict[str, PatternEntry]]:
    """C2: Layer-wise pattern typology."""
    patterns = {}

    for model in PRETRAINED_MODELS:
        patterns[model] = {}
        for prop in SYNTHETIC_PROPS:
            data = load_eval_results(f"{model}_{prop}")
            if data is None:
                data = load_eval_results(f"{model}_synthetic_{prop}")
            if data is None:
                continue

            metric = "val_r2" if prop == "seasonality" else "val_accuracy"
            values = [_metric_number(entry.get(metric, 0.0), default=0.0) for entry in data]
            pattern = classify_layer_pattern(values)
            patterns[model][prop] = {
                "pattern": pattern,
                "values": values,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "range": max(values) - min(values) if values else None,
            }

    return patterns


def compute_cross_leace_asymmetry() -> dict[
    str, dict[str, dict[str, CrossLeaceEntry] | dict[str, AsymmetryEntry]]
]:
    """C4: Cross-LEACE asymmetry quantification."""
    results = {}

    for model_dir in CROSS_LEACE_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        results[model] = {}

        for result_file in model_dir.glob("*.json"):
            data = json.loads(result_file.read_text())
            erased = data.get("erased_property", "")
            evaluated = data.get("evaluated_property", "")
            drop = data.get("accuracy_drop", data.get("drop", 0))

            key = f"{erased} → {evaluated}"
            results[model][key] = {
                "before": data.get("before_accuracy", data.get("accuracy_before", None)),
                "after": data.get("after_accuracy", data.get("accuracy_after", None)),
                "drop": drop,
            }

    # Compute asymmetry for bidirectional pairs
    asymmetry = {}
    for model, entries in results.items():
        asymmetry[model] = {}
        pairs_seen = set()
        for key, val in entries.items():
            parts = key.split(" → ")
            if len(parts) == 2:
                reverse_key = f"{parts[1]} → {parts[0]}"
                if reverse_key in entries and key not in pairs_seen:
                    pairs_seen.add(key)
                    pairs_seen.add(reverse_key)
                    fwd_drop = val["drop"]
                    rev_drop = entries[reverse_key]["drop"]
                    asym = abs(fwd_drop - rev_drop)
                    asymmetry[model][f"{parts[0]} ↔ {parts[1]}"] = {
                        "forward_drop": fwd_drop,
                        "reverse_drop": rev_drop,
                        "asymmetry": asym,
                        "mean_drop": (fwd_drop + rev_drop) / 2,
                    }

    return {"raw": results, "asymmetry": asymmetry}


def main() -> None:
    print("=" * 60)
    print("EXTENDED ANALYSIS")
    print("=" * 60)

    # A2 + B1: Selectivity
    print("\n[A2/B1] Computing selectivity (Linear - MLP Control)...")
    sel = compute_selectivity()
    with open(OUTPUT_DIR / "selectivity_full.json", "w") as f:
        json.dump(sel, f, indent=2, default=str)

    print("\nSelectivity Table (Best Layer):")
    print(f"{'Model':<20} | {'Property':<15} | {'Linear':>8} | {'MLP':>8} | {'Select.':>8}")
    print("-" * 70)
    for model in PRETRAINED_MODELS + BASELINE_MODELS:
        for prop in SYNTHETIC_PROPS:
            entry = sel[model].get(prop, {})
            lin = entry.get("linear")
            mlp = entry.get("mlp")
            s = entry.get("selectivity")
            lin_s = f"{lin:.3f}" if lin is not None else "---"
            mlp_s = f"{mlp:.3f}" if mlp is not None else "---"
            sel_s = f"{s:+.3f}" if s is not None else "---"
            print(
                f"{MODEL_MAP.get(model, model):<20} | {prop:<15} | {lin_s:>8} | "
                f"{mlp_s:>8} | {sel_s:>8}"
            )
        print("-" * 70)

    # C2: Layer patterns
    print("\n[C2] Computing layer-wise patterns...")
    patterns = compute_layer_patterns()
    with open(OUTPUT_DIR / "layer_patterns.json", "w") as f:
        json.dump(patterns, f, indent=2, default=str)

    print("\nLayer Pattern Table:")
    print(f"{'Model':<20} | {'Property':<15} | {'Pattern':<12} | {'Range':>8}")
    print("-" * 65)
    for model in PRETRAINED_MODELS:
        for prop in SYNTHETIC_PROPS:
            entry = patterns[model].get(prop, {})
            pat = entry.get("pattern", "---")
            rng = entry.get("range")
            rng_s = f"{rng:.3f}" if rng is not None else "---"
            print(f"{MODEL_MAP.get(model, model):<20} | {prop:<15} | {pat:<12} | {rng_s:>8}")
        print("-" * 65)

    # C4: Cross-LEACE asymmetry
    print("\n[C4] Computing cross-LEACE asymmetry...")
    cross = compute_cross_leace_asymmetry()
    with open(OUTPUT_DIR / "cross_leace_asymmetry.json", "w") as f:
        json.dump(cross, f, indent=2, default=str)

    print("\nCross-LEACE Asymmetry:")
    for model, pairs in cross["asymmetry"].items():
        print(f"\n  {model}:")
        for pair, vals in pairs.items():
            print(
                f"    {pair}: fwd={vals['forward_drop']:.1f}%, rev={vals['reverse_drop']:.1f}%, "
                f"asym={vals['asymmetry']:.1f}%, mean={vals['mean_drop']:.1f}%"
            )

    print("\n✅ All analyses saved to outputs/analysis/")


if __name__ == "__main__":
    main()
