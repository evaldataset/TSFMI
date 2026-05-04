"""Final analysis script: synthesize all probing results into research findings.

Generates publication-quality tables and figures answering 4 research questions:
  RQ1: What temporal properties are linearly encoded at which layers?
  RQ2: How do Foundation Models differ from Transformer TS and LLM-adapted models?
  RQ3: How much information is linearly vs nonlinearly accessible? (Selectivity)
  RQ4: What is the cross-model redundancy structure? (CKA)

Usage:
    python scripts/final_analysis.py \
        --results_csv outputs/summary/all_results.csv \
        --interventions_dir outputs/interventions/ \
        --leace_dir outputs/leace/ \
        --cka_dir outputs/cka/ \
        --output_dir outputs/analysis/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate final analysis tables and figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_csv", type=str, default="outputs/summary/all_results.csv")
    parser.add_argument("--interventions_dir", type=str, default="outputs/interventions/")
    parser.add_argument("--leace_dir", type=str, default="outputs/leace/")
    parser.add_argument("--cka_dir", type=str, default="outputs/cka/")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis/")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Model display names and grouping
# ---------------------------------------------------------------------------

MODEL_DISPLAY = {
    "moment_pca512": "MOMENT-Large",
    "moment": "MOMENT-65K",
    "chronos": "Chronos-Bolt",
    "patchtst_pretrained": "PatchTST-Pre",
    "patchtst": "PatchTST-Rnd",
    "itransformer": "iTransformer-Rnd",
    "gpt4ts_pca512": "GPT4TS",
    "gpt4ts": "GPT4TS-Raw",
}

MODEL_CATEGORY = {
    "moment_pca512": "Foundation",
    "chronos": "Foundation",
    "patchtst_pretrained": "Transformer-TS",
    "gpt4ts_pca512": "LLM-Adapted",
    "patchtst": "Random-Init",
    "itransformer": "Random-Init",
}

# Preferred model order for tables
MODEL_ORDER = [
    "moment_pca512",
    "chronos",
    "patchtst_pretrained",
    "gpt4ts_pca512",
    "patchtst",
    "itransformer",
]

PROPERTY_ORDER = [
    "trend",
    "seasonality",
    "frequency",
    "stationarity",
    "anomaly",
    "change_point",
]

InterventionInfo: TypeAlias = dict[str, float | bool]
LeaceInfo: TypeAlias = dict[str, float]


def _filter_eq(df: pd.DataFrame, column: str, value: object) -> pd.DataFrame:
    return df.loc[df[column] == value].copy()


def _filter_ne(df: pd.DataFrame, column: str, value: object) -> pd.DataFrame:
    return df.loc[df[column] != value].copy()


def _sort_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.sort_values(by=column).copy()


def _best_row(df: pd.DataFrame, metric: str) -> pd.Series | None:
    if df.empty or metric not in df.columns:
        return None
    metric_values = np.asarray(pd.to_numeric(df[metric], errors="coerce"), dtype=np.float64)
    valid_mask = ~np.isnan(metric_values)
    if not valid_mask.any():
        return None
    best_pos = int(np.nanargmax(metric_values))
    return df.iloc[best_pos]


def _best_metric_value(df: pd.DataFrame, metric: str) -> float | None:
    best = _best_row(df, metric)
    if best is None:
        return None
    value = best.get(metric)
    return float(value) if isinstance(value, int | float) else None


def _unique_str_values(df: pd.DataFrame, column: str) -> list[str]:
    values = df[column].dropna().tolist()
    return sorted(str(value) for value in values)


# ---------------------------------------------------------------------------
# RQ1: What temporal properties are linearly encoded at which layers?
# ---------------------------------------------------------------------------


def rq1_best_layer_table(df: pd.DataFrame, output_dir: Path) -> str:
    """Generate Table 1: Best probing performance per model × property (synthetic).

    For each (model, property), find the layer with highest accuracy/R² and report
    the metric value and layer index.

    Returns:
        Markdown table string.
    """
    synthetic = _filter_eq(df, "dataset", "synthetic")
    if synthetic.empty:
        return "No synthetic results found."

    lines = ["## Table 1: Best Linear Probing Accuracy/R² (Synthetic Data)", ""]
    lines.append("| Model | " + " | ".join(PROPERTY_ORDER) + " |")
    lines.append("|" + "---|" * (len(PROPERTY_ORDER) + 1))

    for model_key in MODEL_ORDER:
        model_data = _filter_eq(synthetic, "model", model_key)
        if model_data.empty:
            continue
        display = MODEL_DISPLAY.get(model_key, model_key)
        row = [display]
        for prop in PROPERTY_ORDER:
            prop_data = _filter_eq(model_data, "property", prop)
            if prop_data.empty:
                row.append("—")
                continue
            if prop == "seasonality":
                # Regression: use R²
                metric = "val_r2"
                best = _best_row(prop_data, metric)
                if best is None:
                    row.append("—")
                    continue
                val = best.get(metric)
                if not isinstance(val, int | float) or np.isnan(float(val)):
                    row.append("—")
                    continue
                layer_idx = _layer_index(str(best.get("layer", "")))
                row.append(f"R²={val:.4f} (L{layer_idx})")
            else:
                # Classification: use accuracy
                metric = "val_accuracy"
                if metric not in prop_data.columns:
                    row.append("—")
                    continue
                best = _best_row(prop_data, metric)
                if best is None:
                    row.append("—")
                    continue
                val = best.get(metric)
                if not isinstance(val, int | float):
                    row.append("—")
                    continue
                layer_idx = _layer_index(str(best.get("layer", "")))
                row.append(f"{val * 100:.1f}% (L{layer_idx})")
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    (output_dir / "table1_best_performance.md").write_text(table)
    return table


def rq1_layer_progression(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate Figure 1: Layer-wise probing accuracy curves per model.

    Shows how information emerges and transforms across layers.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    synthetic = _filter_eq(df, "dataset", "synthetic")
    if synthetic.empty:
        return

    # Focus on pre-trained models
    pretrained = ["moment_pca512", "chronos", "patchtst_pretrained", "gpt4ts_pca512"]
    cls_props = ["trend", "frequency", "stationarity", "anomaly", "change_point"]

    for model_key in pretrained:
        model_data = _filter_eq(synthetic, "model", model_key)
        if model_data.empty:
            continue

        display = MODEL_DISPLAY.get(model_key, model_key)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Classification properties
        ax = axes[0]
        for prop in cls_props:
            prop_data = _sort_by(_filter_eq(model_data, "property", prop), "layer")
            if prop_data.empty or "val_accuracy" not in prop_data.columns:
                continue
            layers = list(range(len(prop_data)))
            accs = prop_data["val_accuracy"].values
            ax.plot(layers, accs, marker="o", label=prop, linewidth=2, markersize=4)
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"{display} — Classification Properties", fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Regression (seasonality)
        ax = axes[1]
        prop_data = _sort_by(_filter_eq(model_data, "property", "seasonality"), "layer")
        if not prop_data.empty and "val_r2" in prop_data.columns:
            layers = list(range(len(prop_data)))
            r2s = prop_data["val_r2"].values
            ax.plot(
                layers,
                r2s,
                marker="s",
                label="seasonality",
                linewidth=2,
                markersize=4,
                color="tab:orange",
            )
            ax.set_ylabel("R²", fontsize=12)
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_title(f"{display} — Seasonality (Regression)", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(
            output_dir / f"fig1_{model_key}_layer_progression.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# RQ2: Foundation vs Transformer-TS vs LLM-Adapted
# ---------------------------------------------------------------------------


def rq2_model_comparison(df: pd.DataFrame, output_dir: Path) -> str:
    """Generate Table 2: Cross-model comparison on synthetic data.

    Groups models by category and compares best-layer performance.
    """
    synthetic = _filter_eq(df, "dataset", "synthetic")
    if synthetic.empty:
        return "No synthetic results found."

    lines = ["## Table 2: Model Comparison by Category (Best Layer, Synthetic)", ""]
    lines.append("| Category | Model | Trend | Season. | Freq. | Station. | Anomaly | Change Pt. |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for model_key in MODEL_ORDER:
        model_data = _filter_eq(synthetic, "model", model_key)
        if model_data.empty:
            continue
        display = MODEL_DISPLAY.get(model_key, model_key)
        category = MODEL_CATEGORY.get(model_key, "Other")
        row = [category, display]
        for prop in PROPERTY_ORDER:
            prop_data = _filter_eq(model_data, "property", prop)
            if prop_data.empty:
                row.append("—")
                continue
            if prop == "seasonality":
                metric = "val_r2"
                if metric in prop_data.columns:
                    best_val = _best_metric_value(prop_data, metric)
                    row.append(f"{best_val:.4f}" if best_val is not None else "—")
                else:
                    row.append("—")
            else:
                metric = "val_accuracy"
                if metric in prop_data.columns:
                    best_val = _best_metric_value(prop_data, metric)
                    row.append(f"{best_val * 100:.1f}%" if best_val is not None else "—")
                else:
                    row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    (output_dir / "table2_model_comparison.md").write_text(table)
    return table


def rq2_real_world_table(df: pd.DataFrame, output_dir: Path) -> str:
    real = _filter_ne(df, "dataset", "synthetic")
    if real.empty:
        return "No real-world results found."

    datasets = _unique_str_values(real, "dataset")
    properties = ["trend", "stationarity", "seasonality", "change_point"]
    pretrained_models = ["moment_pca512", "chronos", "patchtst_pretrained", "gpt4ts_pca512"]

    lines = ["## Table 3: Real-World Dataset Results (Best Layer)", ""]

    for ds in datasets:
        ds_data = _filter_eq(real, "dataset", ds)
        lines.append(f"### {ds.upper()}")
        lines.append("")
        lines.append("| Model | Trend | Stationarity | Seasonality | Change Point |")
        lines.append("|---|---|---|---|---|")

        for model_key in pretrained_models:
            model_data = _filter_eq(ds_data, "model", model_key)
            if model_data.empty:
                continue
            display = MODEL_DISPLAY.get(model_key, model_key)
            row = [display]
            for prop in properties:
                prop_data = _filter_eq(model_data, "property", prop)
                if prop_data.empty:
                    row.append("—")
                    continue
                if prop == "seasonality":
                    metric = "val_r2"
                    if metric in prop_data.columns:
                        best_val = _best_metric_value(prop_data, metric)
                        row.append(f"R²={best_val:.4f}" if best_val is not None else "—")
                    else:
                        row.append("—")
                else:
                    metric = "val_accuracy"
                    if metric in prop_data.columns:
                        best_val = _best_metric_value(prop_data, metric)
                        row.append(f"{best_val * 100:.1f}%" if best_val is not None else "—")
                    else:
                        row.append("—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    table = "\n".join(lines)
    (output_dir / "table3_real_world.md").write_text(table)
    return table


# ---------------------------------------------------------------------------
# RQ3: Selectivity analysis
# ---------------------------------------------------------------------------


def rq3_selectivity_table(df: pd.DataFrame, output_dir: Path) -> str:
    """Generate Table 4: Selectivity (Linear acc − MLP acc) for each model × property."""
    synthetic = _filter_eq(df, "dataset", "synthetic")
    if synthetic.empty or "selectivity" not in synthetic.columns:
        return "No selectivity data found."

    sel_data = synthetic.dropna(subset=["selectivity"]).copy()
    if sel_data.empty:
        return "No selectivity data found."

    lines = ["## Table 4: Selectivity (Linear − MLP Control) — Best Layer", ""]
    lines.append("| Model | " + " | ".join(PROPERTY_ORDER) + " |")
    lines.append("|" + "---|" * (len(PROPERTY_ORDER) + 1))

    for model_key in MODEL_ORDER:
        model_data = _filter_eq(sel_data, "model", model_key)
        if model_data.empty:
            continue
        display = MODEL_DISPLAY.get(model_key, model_key)
        row = [display]
        for prop in PROPERTY_ORDER:
            prop_data = _filter_eq(model_data, "property", prop)
            if prop_data.empty:
                row.append("—")
                continue
            # Report selectivity at the best-accuracy layer
            if prop == "seasonality":
                metric = "val_r2"
            else:
                metric = "val_accuracy"
            if metric not in prop_data.columns:
                row.append("—")
                continue
            best = _best_row(prop_data, metric)
            if best is None:
                row.append("—")
                continue
            sel = best.get("selectivity")
            row.append(f"{float(sel):+.3f}" if isinstance(sel, int | float) else "—")
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    (output_dir / "table4_selectivity.md").write_text(table)
    return table


# ---------------------------------------------------------------------------
# RQ4: Cross-model CKA
# ---------------------------------------------------------------------------


def rq4_cka_summary(cka_dir: Path, output_dir: Path) -> str:
    """Load cross-model CKA results and generate Table 5."""
    cka_path = Path(cka_dir)
    if not cka_path.exists():
        return "CKA directory not found."

    lines = ["## Table 5: Cross-Model CKA Summary", ""]

    cross_dirs = sorted(d for d in cka_path.iterdir() if d.is_dir() and "_vs_" in d.name)
    if not cross_dirs:
        return "No cross-model CKA results found."

    lines.append("| Model A | Model B | Mean CKA | Max CKA | Min CKA |")
    lines.append("|---|---|---|---|---|")

    for d in cross_dirs:
        json_path = d / "cross_model_cka_matrix.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text())
        matrix = np.array(data.get("cka_matrix", []))
        if matrix.size == 0:
            continue
        model_a = data.get("model_a", d.name.split("_vs_")[0])
        model_b = data.get("model_b", d.name.split("_vs_")[-1])
        disp_a = MODEL_DISPLAY.get(model_a, model_a)
        disp_b = MODEL_DISPLAY.get(model_b, model_b)
        mean_cka = matrix.mean()
        max_cka = matrix.max()
        min_cka = matrix.min()
        lines.append(f"| {disp_a} | {disp_b} | {mean_cka:.4f} | {max_cka:.4f} | {min_cka:.4f} |")

    table = "\n".join(lines)
    (output_dir / "table5_cross_cka.md").write_text(table)
    return table


# ---------------------------------------------------------------------------
# Intervention & LEACE summaries
# ---------------------------------------------------------------------------


def intervention_summary(interventions_dir: Path, output_dir: Path) -> str:
    int_path = Path(interventions_dir)
    if not int_path.exists():
        return "Interventions directory not found."

    results: dict[str, dict[str, InterventionInfo]] = defaultdict(dict)

    for d in sorted(int_path.iterdir()):
        if not d.is_dir():
            continue
        json_path = d / "intervention_results.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text())

        name = d.name
        # Parse: {model}_{property}
        _MODEL_PREFIXES = sorted(MODEL_DISPLAY.keys(), key=len, reverse=True)
        model: str | None = None
        prop: str | None = None
        for prefix in _MODEL_PREFIXES:
            if name.startswith(prefix + "_"):
                model = prefix
                prop = name[len(prefix) + 1 :]
                break
        if model is None or prop is None:
            continue

        # Find maximum accuracy/R² drop from LDA steering across layers
        max_drop = 0.0
        is_regression = False
        for _layer_name, layer_data in data.items():
            if not isinstance(layer_data, dict):
                continue
            baseline = layer_data.get("baseline", {})
            lda_sweep = layer_data.get("lda_sweep", [])

            if "accuracy" in baseline:
                primary_metric = "accuracy"
            elif "r2" in baseline:
                primary_metric = "r2"
                is_regression = True
            else:
                continue

            baseline_val = baseline[primary_metric]
            if not lda_sweep:
                continue

            for entry in lda_sweep:
                if not isinstance(entry, dict):
                    continue
                steered_val = entry.get(primary_metric, baseline_val)
                drop = baseline_val - steered_val
                if drop > max_drop:
                    max_drop = drop

        results[model][prop] = {"drop": max_drop, "is_regression": is_regression}

    if not results:
        return "No intervention results found."

    lines = ["## Table 6: LDA Steering — Maximum Performance Drop", ""]
    all_props = sorted({p for model_props in results.values() for p in model_props})
    lines.append("| Model | " + " | ".join(all_props) + " |")
    lines.append("|" + "---|" * (len(all_props) + 1))

    for model_key in MODEL_ORDER:
        if model_key not in results:
            continue
        display = MODEL_DISPLAY.get(model_key, model_key)
        row = [display]
        for prop in all_props:
            if prop in results[model_key]:
                info = results[model_key][prop]
                drop = info["drop"]
                if info["is_regression"]:
                    row.append(f"ΔR²={drop:.3f}")
                else:
                    row.append(f"{drop * 100:.1f}%")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    (output_dir / "table6_interventions.md").write_text(table)
    return table


def leace_summary(leace_dir: Path, output_dir: Path) -> str:
    """Generate Table 7: LEACE concept erasure — before vs after accuracy."""
    leace_path = Path(leace_dir)
    if not leace_path.exists():
        return "LEACE directory not found."

    results: dict[str, dict[str, LeaceInfo]] = defaultdict(dict)

    for d in sorted(leace_path.iterdir()):
        if not d.is_dir():
            continue
        json_path = d / "leace_results.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text())

        name = d.name
        _MODEL_PREFIXES = sorted(MODEL_DISPLAY.keys(), key=len, reverse=True)
        model: str | None = None
        prop: str | None = None
        for prefix in _MODEL_PREFIXES:
            if name.startswith(prefix + "_"):
                model = prefix
                prop = name[len(prefix) + 1 :]
                break
        if model is None or prop is None:
            continue

        # Find best-layer before/after
        best_before = 0.0
        best_after = 0.0
        for _layer_name, layer_data in data.items():
            if not isinstance(layer_data, dict):
                continue
            before = layer_data.get("before", {}).get(
                "accuracy", layer_data.get("before", {}).get("r2", 0)
            )
            after = layer_data.get("after", {}).get(
                "accuracy", layer_data.get("after", {}).get("r2", 0)
            )
            if before > best_before:
                best_before = before
                best_after = after

        results[model][prop] = {"before": best_before, "after": best_after}

    if not results:
        return "No LEACE results found."

    lines = ["## Table 7: LEACE Concept Erasure — Before/After (Best Layer)", ""]
    all_props = sorted({p for model_props in results.values() for p in model_props})
    lines.append("| Model | Property | Before | After | Drop |")
    lines.append("|---|---|---|---|---|")

    for model_key in MODEL_ORDER:
        if model_key not in results:
            continue
        display = MODEL_DISPLAY.get(model_key, model_key)
        for prop in all_props:
            if prop not in results[model_key]:
                continue
            before = results[model_key][prop]["before"]
            after = results[model_key][prop]["after"]
            drop = before - after
            lines.append(
                f"| {display} | {prop} | {before * 100:.1f}% | "
                f"{after * 100:.1f}% | {drop * 100:.1f}% |"
            )

    table = "\n".join(lines)
    (output_dir / "table7_leace.md").write_text(table)
    return table


# ---------------------------------------------------------------------------
# Hard synthetic comparison
# ---------------------------------------------------------------------------


def hard_synthetic_table(df: pd.DataFrame, output_dir: Path) -> str:
    """Generate Table 8: Easy vs Hard synthetic comparison."""
    synthetic = _filter_eq(df, "dataset", "synthetic")
    if synthetic.empty:
        return "No synthetic results found."

    pairs = [
        ("trend", "trend_hard"),
        ("frequency", "frequency_hard"),
        ("anomaly", "anomaly_hard"),
    ]

    lines = ["## Table 8: Easy vs Hard Synthetic Variants (Best Layer Accuracy)", ""]
    lines.append("| Model | Trend | Trend Hard | Freq. | Freq. Hard | Anomaly | Anomaly Hard |")
    lines.append("|---|---|---|---|---|---|---|")

    for model_key in MODEL_ORDER:
        model_data = _filter_eq(synthetic, "model", model_key)
        if model_data.empty:
            continue
        display = MODEL_DISPLAY.get(model_key, model_key)
        row = [display]
        for easy, hard in pairs:
            for prop in [easy, hard]:
                prop_data = _filter_eq(model_data, "property", prop)
                if prop_data.empty or "val_accuracy" not in prop_data.columns:
                    row.append("—")
                    continue
                best_val = _best_metric_value(prop_data, "val_accuracy")
                row.append(f"{best_val * 100:.1f}%" if best_val is not None else "—")
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    (output_dir / "table8_hard_synthetic.md").write_text(table)
    return table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer_index(layer_name: str) -> int:
    """Extract numeric layer index from layer name string."""
    import re

    match = re.search(r"(\d+)", str(layer_name))
    return int(match.group(1)) if match else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all analysis outputs."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        print(f"Results CSV not found: {csv_path}")
        print("Run aggregate_results.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} result entries from {csv_path}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print()

    # Generate all tables
    print("=" * 60)
    print("RQ1: What temporal properties are linearly encoded?")
    print("=" * 60)
    t1 = rq1_best_layer_table(df, output_dir)
    print(t1)
    print()
    rq1_layer_progression(df, output_dir)
    print("Layer progression plots saved.\n")

    print("=" * 60)
    print("RQ2: Foundation vs Transformer-TS vs LLM-Adapted")
    print("=" * 60)
    t2 = rq2_model_comparison(df, output_dir)
    print(t2)
    print()
    t3 = rq2_real_world_table(df, output_dir)
    print(t3)
    print()

    print("=" * 60)
    print("RQ3: Selectivity (Linear vs MLP Control)")
    print("=" * 60)
    t4 = rq3_selectivity_table(df, output_dir)
    print(t4)
    print()

    print("=" * 60)
    print("RQ4: Cross-Model CKA")
    print("=" * 60)
    t5 = rq4_cka_summary(Path(args.cka_dir), output_dir)
    print(t5)
    print()

    print("=" * 60)
    print("Interventions: LDA Steering")
    print("=" * 60)
    t6 = intervention_summary(Path(args.interventions_dir), output_dir)
    print(t6)
    print()

    print("=" * 60)
    print("LEACE: Concept Erasure")
    print("=" * 60)
    t7 = leace_summary(Path(args.leace_dir), output_dir)
    print(t7)
    print()

    print("=" * 60)
    print("Easy vs Hard Synthetic Variants")
    print("=" * 60)
    t8 = hard_synthetic_table(df, output_dir)
    print(t8)
    print()

    # Combine all into a single report
    report = "\n\n".join(
        [
            "# TSFMI: Time Series Foundation Model Interpretability — Results Report",
            "## Research Questions",
            "1. What temporal properties are linearly encoded at which layers?",
            "2. How do Foundation Models differ from Transformer-TS and LLM-adapted models?",
            "3. How much information is linearly vs nonlinearly accessible? (Selectivity)",
            "4. What is the cross-model representational redundancy structure? (CKA)",
            "",
            t1,
            t2,
            t3,
            t4,
            t5,
            t6,
            t7,
            t8,
        ]
    )
    report_path = output_dir / "full_report.md"
    report_path.write_text(report)
    print(f"\nFull report saved to {report_path}")
    print(f"All tables saved to {output_dir}/")


if __name__ == "__main__":
    main()
