"""Generate all publication-quality figures for the NeurIPS paper.

Figures:
    fig1: Layer-wise probing progression (4 models × 6 properties)
    fig2: Cross-model CKA heatmap matrix
    fig3: LEACE before/after grouped bar chart
    fig4: Cross-property LEACE entanglement heatmap
    fig5: Full-D vs PCA512 comparison
    fig6: Real-world seasonality binary bar chart
    fig7: Structural probe temporal distance
    fig8: Fine-tuning comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT_DIR = Path("outputs/paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        # savefig.bbox intentionally omitted — legends placed outside need explicit margins
        "font.family": "serif",
    }
)

MODEL_COLORS = {
    "MOMENT": "#1f77b4",
    "Chronos": "#ff7f0e",
    "PatchTST": "#2ca02c",
    "GPT4TS": "#d62728",
}

MODEL_MARKERS = {
    "MOMENT": "o",
    "Chronos": "s",
    "PatchTST": "^",
    "GPT4TS": "D",
}


# ── Fig 1: Layer-wise probing progression ──────────────────────────────
def fig1_layer_progression() -> None:
    """Layer-wise probing accuracy for 4 models × key properties."""
    eval_dir = Path("outputs/eval")

    properties = {
        "trend": ("classification", "val_accuracy"),
        "stationarity": ("classification", "val_accuracy"),
        "anomaly": ("classification", "val_accuracy"),
        "change_point": ("classification", "val_accuracy"),
    }

    model_dirs = {
        "MOMENT": "moment_pca512",
        "Chronos": "chronos",
        "PatchTST": "patchtst_pretrained",
        "GPT4TS": "gpt4ts_pca512",
    }

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    axes = axes.flatten()

    for idx, (prop, (task_type, metric_key)) in enumerate(properties.items()):
        ax = axes[idx]
        for model_label, model_key in model_dirs.items():
            metrics_file = eval_dir / f"{model_key}_{prop}" / "layer_metrics.json"
            if not metrics_file.exists():
                # Try synthetic prefix
                metrics_file = eval_dir / f"{model_key}_synthetic_{prop}" / "layer_metrics.json"
            if not metrics_file.exists():
                continue

            data = json.loads(metrics_file.read_text())
            if isinstance(data, list):
                layers_data = [(d["layer"], d.get(metric_key, 0)) for d in data]
            else:
                layers_data = [(k, v.get(metric_key, 0)) for k, v in data.items()]

            # Sort by layer index
            def sort_key(item: tuple[str, object]) -> int:
                name = str(item[0])
                parts = name.replace(".", "_").split("_")
                for p in reversed(parts):
                    if p.isdigit():
                        return int(p)
                return 0

            layers_data.sort(key=sort_key)
            x = list(range(len(layers_data)))
            y = [v for _, v in layers_data]
            ax.plot(
                x,
                y,
                color=MODEL_COLORS[model_label],
                marker=MODEL_MARKERS[model_label],
                markersize=3,
                linewidth=1.2,
                label=model_label,
            )

        ax.set_title(prop.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Accuracy" if task_type == "classification" else "R²")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right")

    fig.suptitle("Layer-wise Linear Probing Accuracy (Synthetic Data)", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig1_layer_progression.pdf")
    plt.savefig(OUT_DIR / "fig1_layer_progression.png")
    plt.close()
    print("Saved fig1_layer_progression")


# ── Fig 2: Cross-model CKA ────────────────────────────────────────────
def fig2_cross_model_cka() -> None:
    """Cross-model CKA heatmap loaded from artifact JSONs."""
    CKA_DIR = Path("outputs/cka")
    models = ["MOMENT", "Chronos", "PatchTST", "GPT4TS", "Timer", "TimesFM", "Moirai"]
    model_keys = {
        "MOMENT": "moment_pca512",
        "Chronos": "chronos",
        "PatchTST": "patchtst_pretrained",
        "GPT4TS": "gpt4ts_pca512",
        "Timer": "timer_meanpool",
        "TimesFM": "timesfm_meanpool",
        "Moirai": "moirai_meanpool",
    }

    # Build pairwise CKA matrix from saved artifacts (max over layer pairs)
    n = len(models)
    matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            key_a, key_b = model_keys[models[i]], model_keys[models[j]]
            # Try both orderings of the directory name
            candidates = [
                CKA_DIR / f"{key_a}_vs_{key_b}_trend" / "cross_model_cka_matrix.json",
                CKA_DIR / f"{key_b}_vs_{key_a}_trend" / "cross_model_cka_matrix.json",
            ]
            loaded = False
            for cpath in candidates:
                if cpath.exists():
                    data = json.loads(cpath.read_text())
                    cka_mat = np.array(data["cka_matrix"])
                    max_cka = float(cka_mat.max())
                    matrix[i, j] = max_cka
                    matrix[j, i] = max_cka
                    loaded = True
                    break
            if not loaded:
                print(f"  WARNING: No CKA artifact for {models[i]} vs {models[j]}, using 0.0")
                matrix[i, j] = 0.0
                matrix[j, i] = 0.0

    fig, ax = plt.subplots(figsize=(7.8, 6.1))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.2, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=13)
    ax.set_yticklabels(models, fontsize=13)

    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] > 0.7 else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=13,
                color=color,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, label="Mean CKA", fraction=0.046, pad=0.08)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Mean CKA", fontsize=13)
    ax.set_title("Cross-Model CKA Similarity", fontweight="bold", fontsize=17)
    plt.tight_layout(rect=(0.0, 0.0, 0.93, 1.0))
    plt.savefig(OUT_DIR / "fig2_cross_model_cka.pdf")
    plt.savefig(OUT_DIR / "fig2_cross_model_cka.png")
    plt.close()
    print("Saved fig2_cross_model_cka")


# ── Fig 3: LEACE before/after ─────────────────────────────────────────
def fig3_leace() -> None:
    """LEACE concept erasure grouped bar chart — loaded from artifact JSONs."""
    LEACE_DIR = Path("outputs/leace")
    models = ["MOMENT", "Chronos", "PatchTST", "GPT4TS"]
    model_keys = {
        "MOMENT": "moment_pca512",
        "Chronos": "chronos",
        "PatchTST": "patchtst_pretrained",
        "GPT4TS": "gpt4ts_pca512",
    }
    properties = ["trend", "stationarity", "change_point", "frequency"]

    # Load from artifact JSONs — pick max-drop layer for each model-property
    before: dict[str, dict[str, float]] = {m: {} for m in models}
    after: dict[str, dict[str, float]] = {m: {} for m in models}
    for m in models:
        for p in properties:
            jpath = LEACE_DIR / f"{model_keys[m]}_{p}" / "leace_results.json"
            if jpath.exists():
                data = json.loads(jpath.read_text())
                best_layer = max(data, key=lambda k: data[k].get("drop", 0))
                before[m][p] = data[best_layer]["before"]["accuracy"] * 100
                after[m][p] = data[best_layer]["after"]["accuracy"] * 100
            else:
                print(f"  WARNING: No LEACE artifact for {m}/{p}")
                before[m][p] = 100.0
                after[m][p] = 100.0

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 4.6), sharey=True)
    x = np.arange(len(models))
    width = 0.35

    for idx, prop in enumerate(properties):
        ax = axes[idx]
        before_vals = [before[m][prop] for m in models]
        after_vals = [after[m][prop] for m in models]

        ax.bar(x - width / 2, before_vals, width, color="#4ECDC4", label="Before")
        ax.bar(x + width / 2, after_vals, width, color="#FF6B6B", label="After")

        ax.set_title(prop.replace("_", " ").title(), fontweight="bold", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([m[:4] for m in models], rotation=30, ha="right", fontsize=11)
        ax.set_ylim(0, 110)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3)

        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=13)
        ax.tick_params(axis="y", labelsize=11)

    handles = [
        Patch(facecolor="#4ECDC4", label="Before"),
        Patch(facecolor="#FF6B6B", label="After"),
    ]
    labels = ["Before", "After"]
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.87, 0.5),
        fontsize=12,
        frameon=False,
    )

    fig.suptitle("LEACE Concept Erasure: Before vs After", fontweight="bold", y=1.02, fontsize=15)
    fig.subplots_adjust(right=0.82)
    plt.savefig(OUT_DIR / "fig3_leace_erasure.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig3_leace_erasure.png", bbox_inches="tight")
    plt.close()
    print("Saved fig3_leace_erasure")


# ── Fig 4: Cross-Property LEACE Entanglement Heatmap ──────────────────
def fig4_cross_property_leace() -> None:
    """Heatmap showing property entanglement via cross-LEACE."""
    cross_dir = Path("outputs/cross_leace")

    for model_key, model_label in [("moment_pca512", "MOMENT"), ("gpt4ts_pca512", "GPT4TS")]:
        model_dir = cross_dir / model_key
        if not model_dir.exists():
            continue

        properties = ["trend", "stationarity", "change_point", "frequency"]
        n = len(properties)
        drop_matrix = np.zeros((n, n))

        for i, erase_prop in enumerate(properties):
            for j, eval_prop in enumerate(properties):
                if i == j:
                    drop_matrix[i, j] = np.nan  # Self
                    continue
                fname = model_dir / f"cross_leace_{erase_prop}_on_{eval_prop}.json"
                if fname.exists():
                    data = json.loads(fname.read_text())
                    drop_matrix[i, j] = data.get("drop", 0) * 100
                else:
                    drop_matrix[i, j] = 0

        fig, ax = plt.subplots(figsize=(3.5, 3))
        masked = np.ma.array(drop_matrix, mask=np.isnan(drop_matrix))
        im = ax.imshow(masked, cmap="Reds", vmin=0, vmax=50, aspect="auto")

        labels = [p.replace("_", "\n") for p in properties]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Evaluate Property", fontsize=8)
        ax.set_ylabel("Erase Property", fontsize=8)

        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="gray")
                else:
                    val = drop_matrix[i, j]
                    color = "white" if val > 30 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=color,
                        fontweight="bold",
                    )

        plt.colorbar(im, ax=ax, label="Accuracy Drop (%)", shrink=0.8)
        ax.set_title(f"Cross-Property Entanglement ({model_label})", fontweight="bold", fontsize=9)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"fig4_cross_leace_{model_key}.pdf")
        plt.savefig(OUT_DIR / f"fig4_cross_leace_{model_key}.png")
        plt.close()
        print(f"Saved fig4_cross_leace_{model_key}")


# ── Fig 5: Full-D vs PCA512 ───────────────────────────────────────────
def fig5_fulld_vs_pca() -> None:
    """Bar chart: Full-D Ridge vs PCA512 linear probe."""
    models = ["MOMENT", "GPT4TS"]

    pca_vals = {
        "MOMENT": {"seasonality": -1.117, "anomaly": 0.486},
        "GPT4TS": {"seasonality": -1.040, "anomaly": 0.468},
    }
    fulld_vals = {
        "MOMENT": {"seasonality": 0.9999, "anomaly": 0.565},
        "GPT4TS": {"seasonality": 0.9999, "anomaly": 0.555},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))

    for idx, (prop_label, prop_key) in enumerate(
        [("Seasonality (R²)", "seasonality"), ("Anomaly (Accuracy)", "anomaly")]
    ):
        ax = axes[idx]
        x = np.arange(len(models))
        width = 0.35

        pca = [pca_vals[m][prop_key] for m in models]
        full = [fulld_vals[m][prop_key] for m in models]

        ax.bar(x - width / 2, pca, width, color="#FF6B6B", label="PCA-512")
        ax.bar(x + width / 2, full, width, color="#4ECDC4", label="Full-D Ridge")

        ax.set_title(prop_label, fontweight="bold", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=14)
        ax.tick_params(axis="y", labelsize=13)
        ax.grid(True, axis="y", alpha=0.3)
        if idx == 0:
            ax.set_ylim(-1.3, 1.1)
            ax.axhline(y=0, color="black", linewidth=0.5)
        else:
            ax.set_ylim(0, 0.7)
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        fontsize=14,
        frameon=False,
    )

    fig.suptitle("Full-Dimensional vs PCA-Reduced Probing", fontweight="bold", y=1.02, fontsize=17)
    plt.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    plt.savefig(OUT_DIR / "fig5_fulld_vs_pca.pdf")
    plt.savefig(OUT_DIR / "fig5_fulld_vs_pca.png")
    plt.close()
    print("Saved fig5_fulld_vs_pca")


# ── Fig 6: Real-World Seasonality Binary ───────────────────────────────
def fig6_seasonality_binary() -> None:
    """Real-world seasonality binary classification accuracy."""
    models = ["MOMENT\n(PCA512)", "PatchTST\n(Pre)", "Chronos", "GPT4TS\n(PCA512)"]
    datasets = ["ETTh1", "Weather", "Electricity", "Traffic", "Exchange"]

    # From our results
    data = {
        "ETTh1": [57.7, 84.6, 88.5, 30.8],
        "Weather": [89.6, 89.6, 87.8, 78.7],
        "Electricity": [92.6, 100.0, 100.0, 100.0],
        "Traffic": [33.3, 100.0, 100.0, 33.3],
        "Exchange": [50.0, 100.0, 100.0, 100.0],
    }

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    x = np.arange(len(models))
    width = 0.15
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

    for i, ds in enumerate(datasets):
        offset = (i - len(datasets) / 2 + 0.5) * width
        ax.bar(x + offset, data[ds], width, label=ds, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.set_ylim(0, 110)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Real-World Seasonality Binary Classification", fontweight="bold", fontsize=14)
    ax.tick_params(axis="y", labelsize=11)
    fig.subplots_adjust(right=0.78)
    plt.savefig(OUT_DIR / "fig6_seasonality_binary.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig6_seasonality_binary.png", bbox_inches="tight")
    plt.close()
    print("Saved fig6_seasonality_binary")


# ── Fig 7: Structural Probe ───────────────────────────────────────────
def fig7_structural_probe() -> None:
    """Temporal distance preservation across layers."""
    struct_dir = Path("outputs/structural_probe")
    models = {
        "MOMENT": "moment_trend",
        "Chronos": "chronos_trend",
        "GPT4TS": "gpt4ts_trend",
    }

    fig, ax = plt.subplots(figsize=(11.0, 5.4))

    for model_label, dir_name in models.items():
        results_file = struct_dir / dir_name / "structural_probe_results.json"
        if not results_file.exists():
            continue
        data = json.loads(results_file.read_text())

        # Sort layers
        items = list(data.items())

        def sort_key(item: tuple[str, object]) -> int:
            name = str(item[0])
            parts = name.replace(".", "_").split("_")
            for p in reversed(parts):
                if p.isdigit():
                    return int(p)
            return 0

        items.sort(key=sort_key)
        x = list(range(len(items)))
        y = [v["spearman_rho"] for _, v in items]

        ax.plot(
            x,
            y,
            color=MODEL_COLORS[model_label],
            marker=MODEL_MARKERS[model_label],
            markersize=8,
            linewidth=2.5,
            label=model_label,
        )

    ax.set_xlabel("Layer Index", fontsize=16)
    ax.set_ylabel("Spearman ρ", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_title("Temporal Distance Preservation", fontweight="bold", fontsize=17)
    fig.subplots_adjust(right=0.80)
    plt.savefig(OUT_DIR / "fig7_structural_probe.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig7_structural_probe.png", bbox_inches="tight")
    plt.close()
    print("Saved fig7_structural_probe")


# ── Fig 8: Fine-tuning Comparison ──────────────────────────────────────
def fig8_finetune() -> None:
    """Fine-tuning vs frozen accuracy comparison."""
    ft_file = Path("outputs/finetune_compare/patchtst_pretrained_etth1/finetune_comparison.json")
    if not ft_file.exists():
        print("Skipping fig8 — no fine-tuning results")
        return

    data = json.loads(ft_file.read_text())
    layers = list(data["layers"].keys())
    frozen = [data["layers"][layer_name]["frozen_accuracy"] * 100 for layer_name in layers]
    finetuned = [data["layers"][layer_name]["finetuned_accuracy"] * 100 for layer_name in layers]
    layer_labels = [f"L{i}" for i in range(len(layers))]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(layers))
    width = 0.35

    ax.bar(x - width / 2, frozen, width, color="#6c757d", label="Frozen")
    ax.bar(x + width / 2, finetuned, width, color="#198754", label="Fine-tuned")

    # Add delta labels
    for i in range(len(layers)):
        delta = finetuned[i] - frozen[i]
        ax.annotate(
            f"+{delta:.1f}%",
            xy=(x[i] + width / 2, finetuned[i]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color="#198754",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_ylim(70, 90)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("PatchTST-Pre / ETTh1 Trend", fontweight="bold", fontsize=15)
    fig.subplots_adjust(right=0.78)
    plt.savefig(OUT_DIR / "fig8_finetune.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig8_finetune.png", bbox_inches="tight")
    plt.close()
    print("Saved fig8_finetune")


# ── Fig 9: LDA Steering Vulnerability ─────────────────────────────────
def fig9_lda_steering() -> None:
    """LDA steering vulnerability comparison across models."""
    models = ["MOMENT", "Chronos", "PatchTST", "GPT4TS"]
    properties = ["Trend\nHard", "Freq\nHard", "Station.", "Anomaly\nHard", "Change\nPt."]

    # Max drop from Table 6
    data = {
        "MOMENT": [53.3, 63.7, 2.2, 39.7, 1.0],
        "Chronos": [0.2, 3.0, 0.0, 27.4, 0.0],
        "PatchTST": [0.0, 0.0, 0.0, 2.2, 0.0],
        "GPT4TS": [46.3, 59.6, 49.3, 25.2, 1.7],
    }

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    x = np.arange(len(properties))
    width = 0.2

    for i, model in enumerate(models):
        offset = (i - len(models) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            data[model],
            width,
            label=model,
            color=list(MODEL_COLORS.values())[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(properties, fontsize=12)
    ax.set_ylabel("Max Accuracy Drop (%)", fontsize=14)
    ax.set_ylim(0, 75)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("LDA Steering Vulnerability", fontweight="bold", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    fig.subplots_adjust(right=0.80)
    plt.savefig(OUT_DIR / "fig9_lda_steering.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig9_lda_steering.png", bbox_inches="tight")
    plt.close()
    print("Saved fig9_lda_steering")


if __name__ == "__main__":
    fig1_layer_progression()
    fig2_cross_model_cka()
    fig3_leace()
    fig4_cross_property_leace()
    fig5_fulld_vs_pca()
    fig6_seasonality_binary()
    fig7_structural_probe()
    fig8_finetune()
    fig9_lda_steering()
    print(f"\nAll figures saved to {OUT_DIR}/")
