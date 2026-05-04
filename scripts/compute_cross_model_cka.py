"""Compute cross-model CKA heatmap between all layers of two representation sets.

Loads all layer .pt files from two representations directories, computes linear CKA
between every layer pair across model A and model B, and saves the resulting
rectangular matrix as JSON + heatmap PNG.

Usage:
    python scripts/compute_cross_model_cka.py \
        --repr_dir_a outputs/representations/moment_pca512/synthetic_trend/ \
        --repr_dir_b outputs/representations/chronos/synthetic_trend/ \
        --output_dir outputs/cka/moment_vs_chronos_trend/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for cross-model CKA computation.

    Args:
        argv: Argument list. Defaults to sys.argv[1:] if None.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Compute cross-model CKA heatmap between all layers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--repr_dir_a",
        type=str,
        required=True,
        help="Representation directory for model A.",
    )
    _ = parser.add_argument(
        "--repr_dir_b",
        type=str,
        required=True,
        help="Representation directory for model B.",
    )
    _ = parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving CKA matrix JSON and heatmap PNG.",
    )
    _ = parser.add_argument(
        "--model_a_name",
        type=str,
        default=None,
        help="Display name for model A. Inferred from directory if omitted.",
    )
    _ = parser.add_argument(
        "--model_b_name",
        type=str,
        default=None,
        help="Display name for model B. Inferred from directory if omitted.",
    )
    _ = parser.add_argument(
        "--max_samples",
        type=int,
        default=2000,
        help="Subsample to this many samples for CKA (memory efficiency).",
    )
    _ = parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string ("cuda", "cpu"). Auto-selects if None.',
    )
    return parser.parse_args(argv)


def _infer_model_name(repr_dir: Path) -> str:
    """Infer model display name from a representation directory path.

    Args:
        repr_dir: Representation directory path.

    Returns:
        Inferred model name.
    """
    if repr_dir.parent.name and repr_dir.parent.name != "representations":
        return repr_dir.parent.name
    return repr_dir.name


def _short_layer_name(layer_name: str) -> str:
    """Create short layer labels for compact heatmap tick text.

    Args:
        layer_name: Original layer name.

    Returns:
        Shortened layer name.
    """
    return (
        layer_name.replace("encoder_block_", "B")
        .replace("encoder_layers_", "L")
        .replace("transformer_h_", "H")
    )


def _load_layer_files(repr_dir: Path) -> list[Path]:
    """Discover all representation layer files in a directory.

    Args:
        repr_dir: Directory containing layer .pt files.

    Returns:
        Sorted list of layer file paths excluding labels.pt.

    Raises:
        FileNotFoundError: If directory does not exist or has no layer files.
    """
    if not repr_dir.exists():
        raise FileNotFoundError(f"Representations directory not found: {repr_dir}")

    layer_files = sorted(f for f in repr_dir.glob("*.pt") if f.stem != "labels")
    if not layer_files:
        raise FileNotFoundError(f"No layer .pt files found in {repr_dir}")
    return layer_files


def _load_representations(
    layer_files: list[Path],
    *,
    device: torch.device,
    shared_indices: torch.Tensor | None,
) -> tuple[list[str], list[torch.Tensor]]:
    """Load and preprocess representation tensors for all layer files.

    Args:
        layer_files: Layer file paths to load.
        device: Device to place tensors on.
        shared_indices: Shared sample indices for subsampling, if any.

    Returns:
        Tuple of (layer_names, processed_representation_tensors).
    """
    layer_names: list[str] = []
    representations: list[torch.Tensor] = []
    index_tensor = shared_indices.to(device) if shared_indices is not None else None

    for layer_file in layer_files:
        layer_name = layer_file.stem
        tensor = torch.load(layer_file, map_location=device, weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor in {layer_file}, got {type(tensor).__name__}")
        if tensor.ndim >= 3:
            tensor = tensor.view(tensor.shape[0], -1)
        if index_tensor is not None:
            tensor = tensor[index_tensor]
        layer_names.append(layer_name)
        representations.append(tensor)

    return layer_names, representations


def _plot_cka_heatmap(
    cka_matrix: list[list[float]],
    *,
    layers_a: list[str],
    layers_b: list[str],
    model_a_name: str,
    model_b_name: str,
    output_dir: Path,
) -> None:
    """Generate and save a rectangular cross-model CKA heatmap as PNG.

    Args:
        cka_matrix: CKA matrix with shape (len(layers_a), len(layers_b)).
        layers_a: Layer labels for model A (Y-axis).
        layers_b: Layer labels for model B (X-axis).
        model_a_name: Display name for model A.
        model_b_name: Display name for model B.
        output_dir: Directory to save heatmap PNG.
    """
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array(cka_matrix)
    short_a = [_short_layer_name(name) for name in layers_a]
    short_b = [_short_layer_name(name) for name in layers_b]

    fig_width = max(8.0, len(short_b) * 0.7)
    fig_height = max(6.0, len(short_a) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")

    _ = ax.set_xticks(range(len(short_b)))
    _ = ax.set_yticks(range(len(short_a)))
    _ = ax.set_xticklabels(short_b, fontsize=10, rotation=45, ha="right")
    _ = ax.set_yticklabels(short_a, fontsize=10)
    _ = ax.set_xlabel(f"{model_b_name} layers")
    _ = ax.set_ylabel(f"{model_a_name} layers")
    _ = ax.set_title(f"Linear CKA: {model_a_name} vs {model_b_name}")

    for i in range(len(short_a)):
        for j in range(len(short_b)):
            value = matrix[i, j]
            color = "white" if value < 0.7 else "black"
            _ = ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color=color)

    _ = fig.colorbar(im, ax=ax, shrink=0.8, label="CKA")
    plt.tight_layout()

    plot_path = output_dir / "cross_model_cka_heatmap.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap to {plot_path}")


def main() -> None:
    """Compute cross-model CKA matrix and generate rectangular heatmap."""
    args = parse_args()

    from src.metrics.probing_metrics import compute_cka
    from src.utils.device import resolve_device

    device = resolve_device(args.device)
    repr_dir_a = Path(args.repr_dir_a)
    repr_dir_b = Path(args.repr_dir_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_a_name = args.model_a_name or _infer_model_name(repr_dir_a)
    model_b_name = args.model_b_name or _infer_model_name(repr_dir_b)

    layer_files_a = _load_layer_files(repr_dir_a)
    layer_files_b = _load_layer_files(repr_dir_b)

    first_a = torch.load(layer_files_a[0], map_location="cpu", weights_only=True)
    first_b = torch.load(layer_files_b[0], map_location="cpu", weights_only=True)
    if not isinstance(first_a, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor in {layer_files_a[0]}, got {type(first_a).__name__}"
        )
    if not isinstance(first_b, torch.Tensor):
        raise TypeError(
            f"Expected torch.Tensor in {layer_files_b[0]}, got {type(first_b).__name__}"
        )
    samples_a = first_a.shape[0]
    samples_b = first_b.shape[0]
    min_samples = min(samples_a, samples_b, args.max_samples)

    # Use shared indices to ensure sample alignment across models
    _ = torch.manual_seed(42)
    shared_indices = torch.randperm(min_samples)[:min_samples]
    if samples_a > min_samples:
        indices_a = shared_indices
    else:
        indices_a = None
    if samples_b > min_samples:
        indices_b = shared_indices
    else:
        indices_b = None

    del first_a
    del first_b

    layers_a, reps_a = _load_representations(
        layer_files_a,
        device=device,
        shared_indices=indices_a,
    )
    layers_b, reps_b = _load_representations(
        layer_files_b,
        device=device,
        shared_indices=indices_b,
    )

    print(f"Loaded {len(layers_a)} layers from {repr_dir_a} ({model_a_name})")
    print(f"Loaded {len(layers_b)} layers from {repr_dir_b} ({model_b_name})")
    print(f"  Sample size: {reps_a[0].shape[0]}, Device: {device}")

    num_layers_a = len(layers_a)
    num_layers_b = len(layers_b)
    cka_matrix: list[list[float]] = [[0.0] * num_layers_b for _ in range(num_layers_a)]

    total_pairs = num_layers_a * num_layers_b
    computed = 0
    for i, rep_a in enumerate(reps_a):
        for j, rep_b in enumerate(reps_b):
            cka_value = compute_cka(rep_a, rep_b)
            cka_matrix[i][j] = cka_value
            computed += 1
            if computed % 5 == 0 or computed == total_pairs:
                print(f"  CKA pairs: {computed}/{total_pairs}")

    result = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "layers_a": layers_a,
        "layers_b": layers_b,
        "cka_matrix": cka_matrix,
    }
    json_path = output_dir / "cross_model_cka_matrix.json"
    _ = json_path.write_text(json.dumps(result, indent=2))
    print(f"Saved CKA matrix to {json_path}")

    _plot_cka_heatmap(
        cka_matrix,
        layers_a=layers_a,
        layers_b=layers_b,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
