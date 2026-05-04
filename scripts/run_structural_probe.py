"""Structural probe: test if representation distances preserve temporal structure.

Inspired by Hewitt & Manning (2019) structural probes for syntax trees.
For time series, we test whether the pairwise distances in representation space
preserve temporal ordering and periodic structure.

Tests:
1. Temporal distance correlation: Does ||h_t - h_s|| correlate with |t - s|?
2. Periodic structure: For periodic signals, does representation distance
   reflect within-period position similarity?

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_structural_probe.py \
        --representations_dir outputs/representations/moment_pca512/synthetic_seasonality/ \
        --output_dir outputs/structural_probe/moment_pca512_seasonality/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

StructuralProbeResult = dict[str, str | float | int]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Structural probing of temporal representations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--representations_dir",
        type=str,
        required=True,
        help="Dir with per-layer {layer}.pt files. Must have patch-level representations.",
    )
    _ = parser.add_argument("--output_dir", type=str, required=True)
    _ = parser.add_argument("--num_samples", type=int, default=100, help="Samples to analyze.")
    _ = parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def compute_temporal_distance_correlation(
    representations: torch.Tensor,
    num_samples: int = 100,
) -> StructuralProbeResult:
    """Compute correlation between representation distance and temporal distance.

    For patch-level representations (N, num_patches, d_model), computes pairwise
    distances between patches and correlates with temporal (position) distance.

    Args:
        representations: Tensor of shape (N, num_patches, d_model) or (N, features).
        num_samples: Number of samples to use.

    Returns:
        Dict with Spearman correlation and p-value.
    """
    if representations.ndim == 2:
        # Flat representations — can't do temporal distance analysis
        return {"error": "flat_representations", "spearman_rho": 0.0, "p_value": 1.0}

    # Use first num_samples
    repr_subset = representations[:num_samples].float()
    n, num_patches, d_model = repr_subset.shape

    # Compute pairwise representation distances for each sample
    all_repr_dists = []
    all_temp_dists = []

    for i in range(min(n, num_samples)):
        patches = repr_subset[i]  # (num_patches, d_model)
        # Pairwise L2 distances between patches
        repr_dists = torch.cdist(patches.unsqueeze(0), patches.unsqueeze(0)).squeeze(0)
        # Temporal distances
        positions = torch.arange(num_patches, dtype=torch.float32)
        temp_dists = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()

        # Extract upper triangle (avoid duplicates and diagonal)
        mask = torch.triu(torch.ones(num_patches, num_patches, dtype=torch.bool), diagonal=1)
        all_repr_dists.append(repr_dists[mask].numpy())
        all_temp_dists.append(temp_dists[mask].numpy())

    repr_dists_flat = np.concatenate(all_repr_dists)
    temp_dists_flat = np.concatenate(all_temp_dists)

    rho_raw, p_value_raw = spearmanr(repr_dists_flat, temp_dists_flat)
    rho = float(np.asarray(rho_raw).reshape(-1)[0])
    p_value = float(np.asarray(p_value_raw).reshape(-1)[0])

    return {
        "spearman_rho": rho,
        "p_value": p_value,
        "num_patches": int(num_patches),
        "num_samples_used": min(n, num_samples),
    }


def main() -> None:
    """Run structural probe analysis across layers."""
    args = parse_args()
    repr_dir = Path(args.representations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_files = sorted(f for f in repr_dir.glob("*.pt") if f.stem != "labels")
    print(f"Found {len(layer_files)} layers in {repr_dir}")

    all_results: dict[str, StructuralProbeResult] = {}

    for layer_file in layer_files:
        layer_name = layer_file.stem
        repr_tensor = torch.load(layer_file, map_location="cpu", weights_only=True)

        print(f"[{layer_name}] shape={list(repr_tensor.shape)}", end=" ")

        if repr_tensor.ndim >= 3:
            # Has patch/position dimension — can analyze temporal structure
            result = compute_temporal_distance_correlation(repr_tensor, args.num_samples)
            print(
                f"rho={result['spearman_rho']:.4f} p={result['p_value']:.2e}"
                if "spearman_rho" in result
                else "error"
            )
        else:
            result = {"error": "2d_representations", "spearman_rho": 0.0, "p_value": 1.0}
            print("flat (skipped)")

        all_results[layer_name] = result

    # Save results
    results_path = output_dir / "structural_probe_results.json"
    _ = results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {results_path}")

    # Plot correlation across layers
    valid_layers = [(k, v) for k, v in all_results.items() if v.get("error") is None]
    if valid_layers:
        fig, ax = plt.subplots(figsize=(10, 5))
        layer_labels = [layer_entry[0] for layer_entry in valid_layers]
        rhos = [layer_entry[1]["spearman_rho"] for layer_entry in valid_layers]

        ax.bar(range(len(layer_labels)), rhos, color="steelblue")
        ax.set_xticks(range(len(layer_labels)))
        ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Spearman ρ (repr dist vs temporal dist)")
        ax.set_title("Temporal Structure Preservation Across Layers")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        fig.savefig(output_dir / "structural_probe_plot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {output_dir}/structural_probe_plot.png")


if __name__ == "__main__":
    main()
