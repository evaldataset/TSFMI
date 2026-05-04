#!/usr/bin/env python
"""Generate publication-ready attention analysis figure for PatchTST and Chronos.

Creates a 2x2 grid showing attention entropy and mean distance across layers
for trend and seasonality classification tasks.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Configure matplotlib for publication quality
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
        "font.family": "serif",
    }
)

# Color scheme for classes
CLASS_COLORS = {
    0: "#1f77b4",  # Blue for Down
    1: "#7f7f7f",  # Gray for Flat
    2: "#d62728",  # Red for Up
}
CLASS_LABELS = {
    0: "Down",
    1: "Flat",
    2: "Up",
}

AttentionLayerData = dict[str, float]
PerClassAttentionData = dict[int, dict[str, list[float]]]


def load_attention_data(json_path: str) -> dict[str, AttentionLayerData]:
    """Load attention analysis data from JSON file."""
    path = Path(json_path)
    if not path.exists():
        return {}

    with open(path) as f:
        return json.load(f)


def extract_layer_metrics(
    data: dict[str, AttentionLayerData],
) -> tuple[list[str], list[float], list[float], PerClassAttentionData]:
    """Extract layer names, entropy, and mean distance from attention data.

    Returns:
        Tuple of (layer_names, entropies, mean_distances, per_class_data)
    """
    layer_names = []
    entropies = []
    mean_distances = []
    per_class_data: PerClassAttentionData = {}

    for layer_name in sorted(data.keys()):
        layer_data = data[layer_name]
        layer_names.append(layer_name.split(".")[-2])  # Extract layer number
        entropies.append(layer_data.get("entropy", 0.0))
        mean_distances.append(layer_data.get("mean_distance", 0.0))

        # Extract per-class data if available
        for class_id in range(3):
            if f"class_{class_id}_entropy" in layer_data:
                if class_id not in per_class_data:
                    per_class_data[class_id] = {"entropy": [], "distance": []}
                per_class_data[class_id]["entropy"].append(layer_data[f"class_{class_id}_entropy"])
                per_class_data[class_id]["distance"].append(
                    layer_data[f"class_{class_id}_distance"]
                )

    return layer_names, entropies, mean_distances, per_class_data


def plot_attention_metric(
    ax,
    layer_names: list[str],
    overall_values: list[float],
    per_class_data: PerClassAttentionData,
    metric_name: str,
    title: str,
) -> None:
    """Plot attention metric (entropy or mean distance) across layers."""
    x = np.arange(len(layer_names))

    # Plot overall metric
    ax.plot(x, overall_values, "ko-", linewidth=2, markersize=6, label="Overall", zorder=3)

    # Plot per-class metrics if available
    if per_class_data:
        for class_id in sorted(per_class_data.keys()):
            class_values = per_class_data[class_id][metric_name]
            if len(class_values) == len(x):
                ax.plot(
                    x,
                    class_values,
                    "o-",
                    color=CLASS_COLORS[class_id],
                    linewidth=1.5,
                    markersize=5,
                    label=f"{CLASS_LABELS[class_id]}",
                    alpha=0.7,
                    zorder=2,
                )

    ax.set_xlabel("Layer", fontsize=9)
    ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", framealpha=0.9)


def main() -> None:
    """Generate attention analysis figure."""
    # Load data
    patchtst_trend = load_attention_data(
        "outputs/attention/patchtst_pretrained_trend/attention_analysis.json"
    )
    patchtst_seasonality = load_attention_data(
        "outputs/attention/patchtst_pretrained_seasonality/attention_analysis.json"
    )

    # Extract metrics
    trend_layers, trend_entropy, trend_distance, trend_per_class = extract_layer_metrics(
        patchtst_trend
    )
    season_layers, season_entropy, season_distance, season_per_class = extract_layer_metrics(
        patchtst_seasonality
    )

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        "Attention Analysis: PatchTST Trend vs. Seasonality",
        fontsize=12,
        fontweight="bold",
    )

    # Top-left: Trend entropy
    plot_attention_metric(
        axes[0, 0],
        trend_layers,
        trend_entropy,
        trend_per_class,
        "entropy",
        "Trend: Attention Entropy",
    )

    # Top-right: Trend mean distance
    plot_attention_metric(
        axes[0, 1],
        trend_layers,
        trend_distance,
        trend_per_class,
        "distance",
        "Trend: Mean Attention Distance",
    )

    # Bottom-left: Seasonality entropy
    plot_attention_metric(
        axes[1, 0],
        season_layers,
        season_entropy,
        season_per_class,
        "entropy",
        "Seasonality: Attention Entropy",
    )

    # Bottom-right: Seasonality mean distance
    plot_attention_metric(
        axes[1, 1],
        season_layers,
        season_distance,
        season_per_class,
        "distance",
        "Seasonality: Mean Attention Distance",
    )

    plt.tight_layout()

    # Save figure
    output_dir = Path("outputs/paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "fig14_attention_analysis.pdf"
    png_path = output_dir / "fig14_attention_analysis.png"

    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(png_path, format="png", bbox_inches="tight", dpi=300)

    print(f"✓ Saved PDF: {pdf_path}")
    print(f"✓ Saved PNG: {png_path}")

    plt.close()


if __name__ == "__main__":
    main()
