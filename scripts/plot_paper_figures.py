"""Generate publication-quality figures for the TSFMI paper.

Creates:
- Fig 3: LEACE before/after bar chart (4 models × 6 properties)
- Fig 4: LDA steering alpha sweep curves
- Fig 5: Cross-model CKA summary heatmap

Usage:
    PYTHONPATH=. .venv/bin/python scripts/plot_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


LEACE_DIR = Path("outputs/leace")
INTERVENTION_DIR = Path("outputs/interventions")
CKA_DIR = Path("outputs/cka")
OUTPUT_DIR = Path("outputs/paper_figures")

MODELS = ["moment_pca512", "chronos", "patchtst_pretrained", "gpt4ts_pca512"]
MODEL_LABELS = ["MOMENT", "Chronos", "PatchTST-Pre", "GPT4TS"]
PROPERTIES = ["trend", "stationarity", "change_point", "frequency", "anomaly", "seasonality"]

COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]


def plot_leace_before_after() -> None:
    """Fig 3: LEACE before/after bar chart grouped by model."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharey=False)
    axes_flat = axes.flatten()

    for prop_idx, prop in enumerate(PROPERTIES):
        ax = axes_flat[prop_idx]
        before_vals = []
        after_vals = []
        model_labels_found = []

        for model, label in zip(MODELS, MODEL_LABELS, strict=False):
            result_path = LEACE_DIR / f"{model}_{prop}" / "leace_results.json"
            if not result_path.exists():
                continue

            results = json.loads(result_path.read_text())
            # Find best layer (highest before metric)
            best_before = -float("inf")
            best_after = -float("inf")
            is_regression = prop in ("seasonality",)

            metric_key = "r2" if is_regression else "accuracy"

            for layer_data in results.values():
                if not isinstance(layer_data, dict):
                    continue
                b = layer_data.get("before", {})
                a = layer_data.get("after", {})
                if isinstance(b, dict) and metric_key in b:
                    bv = float(b[metric_key])
                    av = float(a[metric_key])
                    if bv > best_before:
                        best_before = bv
                        best_after = av

            if best_before > -float("inf"):
                before_vals.append(best_before)
                after_vals.append(best_after)
                model_labels_found.append(label)

        if not before_vals:
            ax.set_title(prop.replace("_", " ").title(), fontsize=12, fontweight="bold")
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        x = np.arange(len(model_labels_found))
        width = 0.35

        ax.bar(x - width / 2, before_vals, width, label="Before", color="#2196F3", alpha=0.85)
        ax.bar(x + width / 2, after_vals, width, label="After", color="#F44336", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels_found, rotation=30, ha="right", fontsize=10)
        ax.set_title(prop.replace("_", " ").title(), fontsize=12, fontweight="bold")
        metric_label = "R²" if prop in ("seasonality",) else "Accuracy"
        ax.set_ylabel(metric_label, fontsize=10)
        if prop_idx == 0:
            ax.legend(fontsize=9)

    plt.suptitle(
        "LEACE Concept Erasure: Before vs After (Best Layer)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    out_path = OUTPUT_DIR / "fig3_leace_before_after.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_lda_steering_curves() -> None:
    """Fig 4: LDA steering alpha sweep curves for key properties."""
    target_props = ["trend_hard", "frequency_hard", "stationarity"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for prop_idx, prop in enumerate(target_props):
        ax = axes[prop_idx]

        for model, label, color in zip(MODELS, MODEL_LABELS, COLORS, strict=False):
            result_path = INTERVENTION_DIR / f"{model}_{prop}" / "intervention_results.json"
            if not result_path.exists():
                continue

            results = json.loads(result_path.read_text())

            # Find best layer data with lda_sweep
            best_layer_data = None
            best_base_acc = -1.0

            for _layer_key, layer_data in results.items():
                if not isinstance(layer_data, dict):
                    continue
                sweep = layer_data.get("lda_sweep")
                if not sweep:
                    continue
                base = layer_data.get("base_accuracy", layer_data.get("base_r2", 0))
                if base > best_base_acc:
                    best_base_acc = base
                    best_layer_data = layer_data

            if best_layer_data is None:
                continue

            sweep = best_layer_data["lda_sweep"]
            alphas = [s["alpha"] for s in sweep]
            is_regression = "r2" in sweep[0]
            if is_regression:
                vals = [s.get("r2", 0) for s in sweep]
            else:
                vals = [s.get("accuracy", 0) for s in sweep]

            ax.plot(alphas, vals, "-o", label=label, color=color, markersize=3, linewidth=1.5)

        ax.set_xlabel("Steering α", fontsize=11)
        ax.set_ylabel("Accuracy" if prop_idx == 0 else "", fontsize=11)
        ax.set_title(prop.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle(
        "LDA Steering: Accuracy vs Steering Strength (Best Layer)", fontsize=14, fontweight="bold"
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    out_path = OUTPUT_DIR / "fig4_lda_steering_curves.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_cross_model_cka_summary() -> None:
    """Fig 5: Cross-model CKA heatmap summary."""
    model_names = ["MOMENT", "Chronos", "PatchTST-Pre", "GPT4TS"]
    n = len(model_names)
    cka_matrix = np.ones((n, n))

    # Map model pairs to matrix positions
    pair_files = {
        ("MOMENT", "Chronos"): "moment_pca512_vs_chronos_trend",
        ("MOMENT", "PatchTST-Pre"): "moment_pca512_vs_patchtst_pretrained_trend",
        ("MOMENT", "GPT4TS"): "moment_pca512_vs_gpt4ts_pca512_trend",
        ("Chronos", "PatchTST-Pre"): "chronos_vs_patchtst_pretrained_trend",
        ("Chronos", "GPT4TS"): "chronos_vs_gpt4ts_pca512_trend",
        ("PatchTST-Pre", "GPT4TS"): "patchtst_pretrained_vs_gpt4ts_pca512_trend",
    }

    for (m1, m2), dirname in pair_files.items():
        result_path = CKA_DIR / dirname / "cross_model_cka_matrix.json"
        if not result_path.exists():
            continue
        data = json.loads(result_path.read_text())
        cka_mat = np.array(data["cka_matrix"])
        mean_cka = float(cka_mat.mean())
        i = model_names.index(m1)
        j = model_names.index(m2)
        cka_matrix[i, j] = mean_cka
        cka_matrix[j, i] = mean_cka

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cka_matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(model_names, fontsize=11)

    for i in range(n):
        for j in range(n):
            color = "white" if cka_matrix[i, j] > 0.7 else "black"
            ax.text(
                j,
                i,
                f"{cka_matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=color,
            )

    plt.colorbar(im, ax=ax, label="Mean CKA", shrink=0.8)
    ax.set_title("Cross-Model CKA Similarity\n(on synthetic_trend)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = OUTPUT_DIR / "fig5_cross_model_cka.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    """Generate all paper figures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating paper figures...")
    print()

    plot_leace_before_after()
    plot_lda_steering_curves()
    plot_cross_model_cka_summary()

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
