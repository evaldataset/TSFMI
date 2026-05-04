"""Aggregate probes_v2 results into paper-ready tables.

Reads all summary.json files from outputs/probes_v2/, extracts test metrics
(with val-based best-layer selection), and produces CSV/JSON tables for the paper.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/aggregate_v2_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROBES_DIR = Path("outputs/probes_v2")
OUTPUT_DIR = Path("outputs/tables_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model display names
MODEL_NAMES = {
    "moment_pca512": "MOMENT-PCA512",
    "chronos": "Chronos-Bolt",
    "patchtst_pretrained": "PatchTST-Pre",
    "gpt4ts_pca512": "GPT4TS-PCA512",
    "timer_meanpool": "Timer-Base",
    "timesfm_meanpool": "TimesFM-500M",
    "moirai_meanpool": "Moirai-Small",
    "patchtst": "PatchTST-Rnd",
    "itransformer": "iTransf.-Rnd",
    "autoformer": "Autoformer-Rnd",
    "timesnet": "TimesNet-Rnd",
    "fedformer": "FEDformer-Rnd",
}

PROPERTIES = [
    "synthetic_trend",
    "synthetic_seasonality",
    "synthetic_frequency",
    "synthetic_stationarity",
    "synthetic_anomaly",
    "synthetic_change_point",
]

PROP_SHORT = {
    "synthetic_trend": "Trend",
    "synthetic_seasonality": "Season. (R²)",
    "synthetic_frequency": "Frequency",
    "synthetic_stationarity": "Station.",
    "synthetic_anomaly": "Anomaly",
    "synthetic_change_point": "Change Pt.",
}


def find_best_layer_test_metric(
    summary: dict, label_type: str
) -> tuple[str, float, float | None]:
    """Find best layer by val metric, return its test metric.

    Returns: (best_layer_name, test_metric_value, val_metric_value)
    """
    layers = summary.get("layers", summary.get("per_layer_metrics", {}))
    if not layers:
        # Fallback: scan individual layer dirs
        return "", 0.0, None

    best_layer = ""
    best_val = -1e9
    best_test = 0.0

    for layer_name, metrics in layers.items():
        if label_type == "regression":
            val_metric = metrics.get("val_r2", metrics.get("val_mae", -1e9))
            test_metric = metrics.get("test_r2", val_metric)
        else:
            val_metric = metrics.get("val_accuracy", -1e9)
            test_metric = metrics.get("test_accuracy", val_metric)

        if val_metric > best_val:
            best_val = val_metric
            best_test = test_metric
            best_layer = layer_name

    return best_layer, best_test, best_val


def main() -> None:
    results: list[dict] = []

    for model_key, model_name in MODEL_NAMES.items():
        for prop in PROPERTIES:
            probe_dir = PROBES_DIR / f"{model_key}_{prop}_linear"
            summary_path = probe_dir / "summary.json"

            if not summary_path.exists():
                # Try reading individual layer metrics
                if not probe_dir.exists():
                    continue
                # Scan layer subdirectories
                layer_metrics = {}
                for layer_dir in sorted(probe_dir.iterdir()):
                    mf = layer_dir / "metrics.json"
                    if mf.exists():
                        layer_metrics[layer_dir.name] = json.loads(mf.read_text())

                if not layer_metrics:
                    continue

                label_type = "regression" if "seasonality" in prop else "classification"

                # Find best layer by val, report test
                best_layer = ""
                best_val = -1e9
                best_test = 0.0

                for lname, m in layer_metrics.items():
                    if label_type == "regression":
                        val_m = m.get("val_r2", -1e9)
                        test_m = m.get("test_r2", val_m)
                    else:
                        val_m = m.get("val_accuracy", -1e9)
                        test_m = m.get("test_accuracy", val_m)

                    if val_m > best_val:
                        best_val = val_m
                        best_test = test_m
                        best_layer = lname

                results.append({
                    "model": model_name,
                    "model_key": model_key,
                    "property": PROP_SHORT.get(prop, prop),
                    "property_key": prop,
                    "best_layer": best_layer,
                    "test_metric": round(best_test, 4),
                    "val_metric": round(best_val, 4),
                    "metric_type": "R²" if label_type == "regression" else "Accuracy",
                })
                continue

            summary = json.loads(summary_path.read_text())
            label_type = summary.get("label_type", "classification")
            best_layer, test_metric, val_metric = find_best_layer_test_metric(
                summary, label_type
            )

            results.append({
                "model": model_name,
                "model_key": model_key,
                "property": PROP_SHORT.get(prop, prop),
                "property_key": prop,
                "best_layer": best_layer,
                "test_metric": round(test_metric, 4),
                "val_metric": round(val_metric, 4) if val_metric else None,
                "metric_type": "R²" if label_type == "regression" else "Accuracy",
            })

    # Save full results
    out_path = OUTPUT_DIR / "v2_probe_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {out_path}")

    # Print summary table
    print(f"\n{'Model':<20} {'Property':<15} {'Test Metric':<12} {'Best Layer':<20}")
    print("-" * 67)
    for r in results:
        print(
            f"{r['model']:<20} {r['property']:<15} "
            f"{r['test_metric']:<12.4f} {r['best_layer']:<20}"
        )

    # Print LaTeX-ready table
    print("\n\n=== LaTeX Table (test metrics) ===\n")
    pretrained = [k for k in MODEL_NAMES if k in [
        "moment_pca512", "chronos", "patchtst_pretrained",
        "gpt4ts_pca512", "timer_meanpool", "timesfm_meanpool", "moirai_meanpool"
    ]]

    for mk in pretrained:
        mn = MODEL_NAMES[mk]
        row = [mn]
        for prop in PROPERTIES:
            match = [r for r in results if r["model_key"] == mk and r["property_key"] == prop]
            if match:
                v = match[0]["test_metric"]
                bl = match[0]["best_layer"]
                if "seasonality" in prop:
                    row.append(f"{v:.3f} ({bl})")
                else:
                    row.append(f"{v*100:.1f}% ({bl})")
            else:
                row.append("---")
        print(" & ".join(row) + " \\\\")


if __name__ == "__main__":
    main()
