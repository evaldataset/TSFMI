"""Aggregate all experiment results into a single consolidated CSV + summary plots.

Scans outputs/eval/ for layer_metrics.json files, combines them into a single DataFrame,
and generates multi-property comparison plots.

Usage:
    python scripts/aggregate_results.py \
        --eval_dir outputs/eval/ \
        --cka_dir outputs/cka/ \
        --output_dir outputs/summary/
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for results aggregation.

    Args:
        argv: Argument list. Defaults to sys.argv[1:] if None.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Aggregate all experiment results into consolidated tables and plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="outputs/eval/",
        help="Root directory containing per-experiment eval subdirectories.",
    )
    parser.add_argument(
        "--cka_dir",
        type=str,
        default="outputs/cka/",
        help="Root directory containing CKA heatmap results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/summary/",
        help="Directory for saving aggregated CSV and summary plots.",
    )
    return parser.parse_args(argv)


def _load_eval_results(eval_dir: Path) -> list[dict[str, object]]:
    """Load all layer_metrics.json files from eval subdirectories.

    Args:
        eval_dir: Root eval directory.

    Returns:
        List of dicts with model, dataset, property, layer, and metric fields.
    """
    # Known model prefixes sorted by length descending to match longest first
    _MODEL_PREFIXES = sorted(
        [
            "moment_pca512",
            "moment",
            "patchtst_pretrained",
            "patchtst",
            "itransformer",
            "chronos",
            "gpt4ts_pca512",
            "gpt4ts",
        ],
        key=len,
        reverse=True,
    )

    # Known real-world dataset prefixes (sorted longest first)
    _DATASET_PREFIXES = sorted(
        ["etth1", "weather", "electricity", "traffic", "exchange_rate"],
        key=len,
        reverse=True,
    )

    rows: list[dict[str, object]] = []

    for subdir in sorted(eval_dir.iterdir()):
        if not subdir.is_dir():
            continue
        json_path = subdir / "layer_metrics.json"
        if not json_path.exists():
            continue

        name = subdir.name

        # Parse model name using known prefixes
        model: str | None = None
        remainder: str | None = None
        for prefix in _MODEL_PREFIXES:
            if name.startswith(prefix + "_"):
                model = prefix
                remainder = name[len(prefix) + 1 :]
                break
        if model is None or remainder is None:
            continue

        # Parse dataset and property from remainder
        # remainder could be:
        #   'trend'  (synthetic, property only — old naming)
        #   'synthetic_trend'  (synthetic, full dataset name)
        #   'synthetic_trend_hard'  (synthetic hard variant)
        #   'etth1_trend'  (real-world dataset + property)
        #   'electricity_change_point'  (real-world dataset + compound property)
        dataset = "synthetic"
        prop = remainder

        # Check if remainder starts with a real-world dataset prefix
        matched_ds = False
        for ds_prefix in _DATASET_PREFIXES:
            if remainder.startswith(ds_prefix + "_"):
                dataset = ds_prefix
                prop = remainder[len(ds_prefix) + 1 :]
                matched_ds = True
                break

        # If no real-world prefix matched, strip 'synthetic_' prefix if present
        if not matched_ds and remainder.startswith("synthetic_"):
            prop = remainder[len("synthetic_") :]

        layer_metrics: list[dict[str, object]] = json.loads(json_path.read_text())
        for entry in layer_metrics:
            row: dict[str, object] = {
                "model": model,
                "dataset": dataset,
                "property": prop,
                **entry,
            }
            rows.append(row)

    return rows


def _metric_value(entry: dict[str, object], key: str) -> float:
    value = entry.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _generate_comparison_plots(
    rows: list[dict[str, object]],
    output_dir: Path,
) -> None:
    """Generate multi-property comparison plots grouped by model.

    Args:
        rows: Aggregated result rows.
        output_dir: Directory to save plots.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group by model
    models: dict[str, dict[str, list[dict[str, object]]]] = {}
    for row in rows:
        model = str(row["model"])
        prop = str(row["property"])
        if model not in models:
            models[model] = {}
        if prop not in models[model]:
            models[model][prop] = []
        models[model][prop].append(row)

    for model, properties in models.items():
        # Classification properties: accuracy comparison
        cls_props = {}
        reg_props = {}
        for prop, entries in properties.items():
            if "val_accuracy" in entries[0]:
                cls_props[prop] = entries
            elif "val_r2" in entries[0]:
                reg_props[prop] = entries

        # Plot 1: Classification accuracy across layers
        if cls_props:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            for prop, entries in sorted(cls_props.items()):
                layers = list(range(len(entries)))
                accs = [_metric_value(e, "val_accuracy") for e in entries]
                ax.plot(layers, accs, marker="o", label=prop, linewidth=2)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{model} — Classification Accuracy by Layer")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            # Plot 2: F1 macro across layers
            ax = axes[1]
            for prop, entries in sorted(cls_props.items()):
                layers = list(range(len(entries)))
                f1s = [_metric_value(e, "val_f1_macro") for e in entries]
                ax.plot(layers, f1s, marker="s", label=prop, linewidth=2)
            ax.set_xlabel("Layer")
            ax.set_ylabel("F1 Macro")
            ax.set_title(f"{model} — F1 Macro by Layer")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plot_path = output_dir / f"{model}_classification_comparison.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Saved {plot_path}")

        # Plot 3: Regression metrics across layers
        if reg_props:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            ax = axes[0]
            for prop, entries in sorted(reg_props.items()):
                layers = list(range(len(entries)))
                r2s = [_metric_value(e, "val_r2") for e in entries]
                ax.plot(layers, r2s, marker="o", label=prop, linewidth=2)
            ax.set_xlabel("Layer")
            ax.set_ylabel("R²")
            ax.set_title(f"{model} — Regression R² by Layer")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            ax = axes[1]
            for prop, entries in sorted(reg_props.items()):
                layers = list(range(len(entries)))
                maes = [_metric_value(e, "val_mae") for e in entries]
                ax.plot(layers, maes, marker="s", label=prop, linewidth=2)
            ax.set_xlabel("Layer")
            ax.set_ylabel("MAE")
            ax.set_title(f"{model} — Regression MAE by Layer")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            plot_path = output_dir / f"{model}_regression_comparison.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Saved {plot_path}")

        # Plot 4: Selectivity across layers (all properties)
        sel_props = {
            prop: entries
            for prop, entries in properties.items()
            if any("selectivity" in e for e in entries)
        }
        if sel_props:
            fig, ax = plt.subplots(figsize=(10, 5))
            for prop, entries in sorted(sel_props.items()):
                layers = list(range(len(entries)))
                sels = [_metric_value(e, "selectivity") for e in entries]
                ax.plot(layers, sels, marker="D", label=prop, linewidth=2)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Selectivity (Linear − MLP)")
            ax.set_title(f"{model} — Selectivity by Layer")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plot_path = output_dir / f"{model}_selectivity_comparison.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Saved {plot_path}")


def main() -> None:
    """Aggregate results and generate summary reports."""
    args = parse_args()

    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")

    rows = _load_eval_results(eval_dir)
    if not rows:
        print("No results found to aggregate.")
        return

    print(f"Loaded {len(rows)} result entries from {eval_dir}")

    # Determine all field names
    all_fields: list[str] = ["model", "dataset", "property", "layer"]
    for row in rows:
        for key in row:
            if key not in all_fields and not str(key).startswith("_"):
                all_fields.append(key)

    # Save consolidated CSV
    csv_path = output_dir / "all_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved consolidated CSV to {csv_path}")

    # Save as JSON
    json_path = output_dir / "all_results.json"
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"Saved consolidated JSON to {json_path}")

    # Generate comparison plots
    _generate_comparison_plots(rows, output_dir)

    print(f"\nDone! Summary in {output_dir}/")


if __name__ == "__main__":
    main()
