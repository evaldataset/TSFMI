"""Compute full pairwise CKA heatmap between all layers for a given representation set.

Loads all layer .pt files from a representations directory, computes linear CKA
between every pair of layers, and saves the resulting matrix as JSON + heatmap PNG.

Usage:
    python scripts/compute_cka_heatmap.py \
        --representations_dir outputs/representations/itransformer/synthetic_trend/ \
        --output_dir outputs/cka/itransformer_trend/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for CKA heatmap computation.

    Args:
        argv: Argument list. Defaults to sys.argv[1:] if None.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Compute full pairwise CKA heatmap between all layers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--representations_dir",
        type=str,
        required=True,
        help="Directory containing {layer}.pt files (output of extract_representations.py).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/cka/",
        help="Directory for saving CKA matrix and heatmap.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string ("cuda", "cpu"). Auto-selects if None.',
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2000,
        help="Subsample to this many samples for CKA (memory efficiency).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Compute full pairwise CKA matrix and generate heatmap."""
    args = parse_args()

    import torch

    from src.metrics.probing_metrics import compute_cka
    from src.utils.device import resolve_device

    device = resolve_device(args.device)
    repr_dir = Path(args.representations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not repr_dir.exists():
        raise FileNotFoundError(f"Representations directory not found: {repr_dir}")

    # Discover and load all layer files
    layer_files = sorted(f for f in repr_dir.glob("*.pt") if f.stem != "labels")
    if not layer_files:
        raise FileNotFoundError(f"No layer .pt files found in {repr_dir}")

    # Determine shared subsample indices ONCE (critical: must be same across all layers)
    first_tensor = torch.load(layer_files[0], map_location="cpu", weights_only=True)
    total_samples = first_tensor.shape[0]
    if total_samples > args.max_samples:
        torch.manual_seed(42)
        shared_indices = torch.randperm(total_samples)[: args.max_samples]
    else:
        shared_indices = None
    del first_tensor

    layer_names: list[str] = []
    representations: list[torch.Tensor] = []

    for lf in layer_files:
        name = lf.stem
        tensor = torch.load(lf, map_location=device, weights_only=True)
        # Flatten high-dimensional tensors to 2D
        if tensor.ndim >= 3:
            tensor = tensor.view(tensor.shape[0], -1)
        # Subsample with shared indices
        if shared_indices is not None:
            tensor = tensor[shared_indices]
        layer_names.append(name)
        representations.append(tensor)

    num_layers = len(layer_names)
    print(f"Loaded {num_layers} layers from {repr_dir}")
    print(f"  Sample size: {representations[0].shape[0]}, Device: {device}")

    # Compute full pairwise CKA matrix
    cka_matrix: list[list[float]] = [[0.0] * num_layers for _ in range(num_layers)]

    total_pairs = num_layers * (num_layers + 1) // 2
    computed = 0
    for i in range(num_layers):
        for j in range(i, num_layers):
            cka_val = compute_cka(representations[i], representations[j])
            cka_matrix[i][j] = cka_val
            cka_matrix[j][i] = cka_val
            computed += 1
            if computed % 5 == 0 or computed == total_pairs:
                print(f"  CKA pairs: {computed}/{total_pairs}")

    # Save CKA matrix as JSON
    result = {
        "layer_names": layer_names,
        "cka_matrix": cka_matrix,
    }
    json_path = output_dir / "cka_matrix.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"Saved CKA matrix to {json_path}")

    # Generate heatmap
    _plot_cka_heatmap(cka_matrix, layer_names, output_dir)


def _plot_cka_heatmap(
    cka_matrix: list[list[float]],
    layer_names: list[str],
    output_dir: Path,
) -> None:
    """Generate and save a CKA heatmap as PNG.

    Args:
        cka_matrix: Square matrix of CKA values.
        layer_names: Labels for each layer.
        output_dir: Directory to save the heatmap PNG.
    """
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array(cka_matrix)
    short_names = [n.replace("encoder_layers_", "L") for n in layer_names]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")

    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title("Linear CKA Inter-Layer Similarity")

    # Annotate cells
    for i in range(len(short_names)):
        for j in range(len(short_names)):
            val = matrix[i, j]
            color = "white" if val < 0.7 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="CKA")
    plt.tight_layout()

    plot_path = output_dir / "cka_heatmap.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap to {plot_path}")


if __name__ == "__main__":
    main()
