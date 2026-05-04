"""Build the canonical 7-model x 6-property gap heatmap (PLAN.md overview figure).

This is the single most reviewer-memorable figure we can add: a 7x6 grid where
each cell shows (best model test score) - (best baseline test score). Cells
where baselines saturate (gap <= 0) are marked gray; positive gaps are shown
in blue; negative gaps (model loses to baselines) are shown in red.

Reads from:
    outputs/canonical/summary.json
    outputs/canonical_baselines/all_results.json

Writes to:
    outputs/paper/latex/figures/fig0_gap_heatmap.pdf
    outputs/paper/latex/figures/fig0_gap_heatmap.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

OUT_DIR = Path("outputs/paper/latex/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("moment_pca512", "MOMENT"),
    ("chronos", "Chronos"),
    ("patchtst_pretrained", "PatchTST"),
    ("gpt4ts_pca512", "GPT4TS"),
    ("timer_meanpool", "Timer"),
    ("timesfm_meanpool", "TimesFM"),
    ("moirai_meanpool", "Moirai"),
]
PROPERTIES = ["trend", "seasonality", "frequency", "stationarity", "anomaly", "change_point"]
PROP_LABELS = {
    "trend": "Trend",
    "seasonality": "Seasonality",
    "frequency": "Frequency",
    "stationarity": "Stationarity",
    "anomaly": "Anomaly",
    "change_point": "Change Pt.",
}


def main() -> None:
    canon = json.loads(Path("outputs/canonical/summary.json").read_text())
    baselines = json.loads(Path("outputs/canonical_baselines/all_results.json").read_text())

    # Best baseline per property (max over hand_crafted / raw_signal / random_projection)
    best_baseline: dict[str, float] = {}
    for prop in PROPERTIES:
        prop_rows = [r for r in baselines if r["property"] == prop]
        if prop_rows:
            best_baseline[prop] = max(r["test_mean"] for r in prop_rows)
        else:
            best_baseline[prop] = float("nan")

    # Gap matrix: model best - best baseline
    n_m = len(MODELS)
    n_p = len(PROPERTIES)
    gap = np.full((n_m, n_p), np.nan)
    model_abs = np.full((n_m, n_p), np.nan)
    for i, (mk, _) in enumerate(MODELS):
        for j, prop in enumerate(PROPERTIES):
            entry = canon.get(mk, {}).get(prop)
            if entry is None:
                continue
            model_abs[i, j] = entry["mean"]
            gap[i, j] = entry["mean"] - best_baseline[prop]

    # Plot: dual-panel layout, left = model abs, right = gap vs best baseline
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1]})

    # ---- Left: model best-layer test score (abs) ----
    ax = axes[0]
    im0 = ax.imshow(model_abs, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_p))
    ax.set_xticklabels([PROP_LABELS[p] for p in PROPERTIES], rotation=30, ha="right")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([lbl for _, lbl in MODELS])
    ax.set_title("(a) Model best-layer test score", fontweight="bold")
    for i in range(n_m):
        for j in range(n_p):
            v = model_abs[i, j]
            if np.isnan(v):
                continue
            color = "white" if v < 0.6 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=color)
    cbar0 = fig.colorbar(im0, ax=ax, shrink=0.85)
    cbar0.set_label("Test accuracy / $R^2$")

    # ---- Right: gap vs best baseline ----
    ax = axes[1]
    vmin = np.nanmin(gap)
    vmax = np.nanmax(gap)
    bound = max(abs(vmin), abs(vmax), 0.15)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    im1 = ax.imshow(gap, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(range(n_p))
    ax.set_xticklabels([PROP_LABELS[p] for p in PROPERTIES], rotation=30, ha="right")
    ax.set_yticks(range(n_m))
    ax.set_yticklabels([lbl for _, lbl in MODELS])
    ax.set_title(
        "(b) Gap vs.\\ best non-model baseline\n"
        "(blue $=$ model ahead, red $=$ baseline ahead)",
        fontweight="bold",
    )
    for i in range(n_m):
        for j in range(n_p):
            v = gap[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:+.3f}" if abs(v) >= 0.001 else "0.000"
            color = "white" if abs(v) > 0.08 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    cbar1 = fig.colorbar(im1, ax=ax, shrink=0.85)
    cbar1.set_label("Model $-$ baseline gap")

    # Annotation: baselines under the right panel
    baseline_text = "Best baselines: " + ", ".join(
        f"{PROP_LABELS[p]} {best_baseline[p]:.3f}" for p in PROPERTIES
    )
    fig.text(
        0.51,
        -0.02,
        baseline_text,
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig0_gap_heatmap.pdf", bbox_inches="tight")
    plt.savefig(OUT_DIR / "fig0_gap_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR}/fig0_gap_heatmap.pdf")
    print(f"Saved {OUT_DIR}/fig0_gap_heatmap.png")
    print("\nGap summary (rows=models, cols=properties):")
    print("   " + "  ".join(f"{p[:6]:>8}" for p in PROPERTIES))
    for i, (_, lbl) in enumerate(MODELS):
        row = "  ".join(
            f"{gap[i, j]:+8.3f}" if not np.isnan(gap[i, j]) else f"{'n/a':>8}"
            for j in range(n_p)
        )
        print(f"{lbl[:10]:<10} {row}")


if __name__ == "__main__":
    main()
