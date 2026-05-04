# YK|"""CLI script for visualizing layer representations with t-SNE and UMAP.
# KM|
# PX|Load frozen model representations, reduce to 2D using t-SNE/UMAP, and save scatter plots
# NS|colored by ground-truth labels. Useful for interpreting what temporal properties each layer
# BT|encodes (trend, seasonality, frequency, stationarity, etc.).
# HT|
# JP|Usage:
# WX|    python scripts/visualize_representations.py \
# ZY|        --representations_dir outputs/representations/chronos/synthetic_stationarity \
# SM|        --output_dir outputs/viz/chronos_stationarity \
# MQ|        --layers 0,2,4 \
# BQ|        --max_samples 1000
# ZQ|"""
# RJ|

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for representation visualization.

    Args:
        argv: Argument list. Defaults to sys.argv[1:] if None.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Visualize layer representations with t-SNE and UMAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--representations_dir",
        type=str,
        required=True,
        help="Directory containing layer representations (.pt files) and labels.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/viz/",
        help="Directory to save visualization plots",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help='Layers to visualize: "all" or comma-separated indices like "0,2,4".',
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2000,
        help="Maximum samples to use for t-SNE/UMAP (subsampled if needed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="tsne,umap",
        help="Comma-separated visualization methods (tsne, umap)",
    )
    return parser.parse_args(argv)


def load_layer_representations(
    representations_dir: Path, layers: list[str]
) -> dict[str, torch.Tensor]:
    """Load per-layer representation tensors from disk.

    Args:
        representations_dir: Directory containing .pt files.
        layers: List of layer names to load.

    Returns:
        Dictionary mapping layer names to tensors.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    representations: dict[str, torch.Tensor] = {}

    for layer_name in layers:
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        file_path = representations_dir / f"{safe_name}.pt"

        if not file_path.exists():
            raise FileNotFoundError(f"Missing representation file: {file_path}")

        representations[layer_name] = torch.load(file_path, weights_only=True)

    return representations


def load_labels(representations_dir: Path) -> torch.Tensor:
    """Load ground-truth labels from disk.

    Args:
        representations_dir: Directory containing labels.pt.

    Returns:
        Label tensor.
    """
    labels_path = representations_dir / "labels.pt"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    return torch.load(labels_path, weights_only=True)


def bin_regression_labels(labels: torch.Tensor, num_bins: int = 8) -> torch.Tensor:
    """Bin continuous labels into discrete categories for coloring.

    Args:
        labels: Continuous label tensor.
        num_bins: Number of bins to create.

    Returns:
        Binned integer labels.
    """
    labels_np = labels.cpu().numpy().astype(np.float64)
    bin_edges = np.linspace(labels_np.min(), labels_np.max(), num_bins + 1)
    binned = np.digitize(labels_np, bin_edges) - 1
    binned = np.clip(binned, 0, num_bins - 1)
    return torch.from_numpy(binned).long()


def flatten_representations(representations: torch.Tensor) -> torch.Tensor:
    """Flatten multi-dimensional representations to 2D (N, features).

    Args:
        representations: Input tensor of shape (N, *shape).

    Returns:
        Flattened tensor of shape (N, features).
    """
    if representations.ndim == 2:
        return representations

    if representations.ndim == 3:
        return representations.reshape(representations.shape[0], -1)

    raise ValueError(
        f"Unsupported representation shape: {representations.shape}. "
        f"Expected 2D or 3D (N, hidden_dim) or (N, heads, hidden_dim)."
    )


def run_tsne(
    representations: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_samples: int = 2000,
    perplexity: int = 30,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run t-SNE dimensionality reduction.

    Args:
        representations: 2D representation tensor of shape (N, features).
        labels: Ground-truth labels for coloring.
        max_samples: Maximum number of samples to use.
        perplexity: t-SNE perplexity parameter.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (2D embeddings, color labels).
    """
    if representations.shape[0] > max_samples:
        indices = torch.randperm(representations.shape[0])[:max_samples]
        representations = representations[indices]
        labels = labels[indices]

    embeddings = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=1000,
        init="pca",
    ).fit_transform(representations.cpu().numpy())

    return torch.from_numpy(embeddings), labels


def run_umap(
    representations: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_samples: int = 2000,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run UMAP dimensionality reduction.

    Args:
        representations: 2D representation tensor of shape (N, features).
        labels: Ground-truth labels for coloring.
        max_samples: Maximum number of samples to use.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (2D embeddings, color labels).
    """
    try:
        import umap
    except ImportError as e:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn") from e

    if representations.shape[0] > max_samples:
        indices = torch.randperm(representations.shape[0])[:max_samples]
        representations = representations[indices]
        labels = labels[indices]

    embeddings = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    ).fit_transform(representations.cpu().numpy())

    return torch.from_numpy(embeddings), labels


def create_scatter_plot(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    title: str,
    output_path: Path,
) -> None:
    """Create a scatter plot colored by labels.

    Args:
        embeddings: 2D embedding tensor of shape (N, 2).
        labels: Color labels (int for classification, binned for regression).
        title: Plot title.
        output_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))

    unique_labels = np.unique(labels_np)
    cmap = plt.get_cmap("tab10")

    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        ax.scatter(
            embeddings_np[mask, 0],
            embeddings_np[mask, 1],
            c=[cmap(i % 10)],
            label=f"Label {label}",
            alpha=0.6,
            edgecolor="none",
        )

    ax.set_xlabel("Dimension 1", fontsize=12)
    ax.set_ylabel("Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_layer_grid(
    layer_name_to_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
    method: str,
    output_path: Path,
) -> None:
    """Create a grid of scatter plots for all layers side-by-side.

    Args:
        layer_name_to_embeddings: Dictionary mapping layer names to (embeddings, labels).
        method: Visualization method name (tsne/umap).
        output_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    num_layers = len(layer_name_to_embeddings)
    n_cols = min(3, num_layers)
    n_rows = (num_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if num_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    cmap = plt.get_cmap("tab10")

    for idx, (layer_name, (embeddings, labels)) in enumerate(layer_name_to_embeddings.items()):
        if idx >= len(axes):
            break

        embeddings_np = embeddings.cpu().numpy()
        labels_np = labels.cpu().numpy()
        unique_labels = np.unique(labels_np)

        for i, label in enumerate(unique_labels):
            mask = labels_np == label
            axes[idx].scatter(
                embeddings_np[mask, 0],
                embeddings_np[mask, 1],
                c=[cmap(i % 10)],
                label=f"Label {label}",
                alpha=0.6,
                edgecolor="none",
            )

        axes[idx].set_xlabel("Dimension 1", fontsize=9)
        axes[idx].set_ylabel("Dimension 2", fontsize=9)
        axes[idx].set_title(
            f"{layer_name.replace('_', ' ').replace('/', '/')}", fontsize=10, fontweight="bold"
        )
        axes[idx].grid(True, alpha=0.3)

        if idx == 0:
            axes[idx].legend(loc="best", fontsize=8)

    for idx in range(len(axes)):
        if idx >= num_layers:
            axes[idx].axis("off")

    plt.suptitle(f"{method.upper()} — All Layers", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_layer_names(layers_arg: str) -> list[str]:
    """Parse the --layers argument.

    Args:
        layers_arg: Either "all" or comma-separated string like "0,2,4".

    Returns:
        List of layer names.
    """
    if layers_arg == "all":
        return [
            "encoder_block_0",
            "encoder_block_1",
            "encoder_block_2",
            "encoder_block_3",
            "encoder_block_4",
            "encoder_block_5",
        ]

    return [f"encoder_block_{idx}" for idx in map(int, layers_arg.split(","))]


def main() -> None:
    """Load representations, run t-SNE/UMAP, and save scatter plots."""
    args = parse_args()

    representations_dir = Path(args.representations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading representations from: {representations_dir}")
    print(f"Output directory: {output_dir}")

    # Load labels
    labels = load_labels(representations_dir)
    print(f"Labels: shape={labels.shape}, dtype={labels.dtype}, unique={torch.unique(labels)}")

    # Parse layers
    layer_names = parse_layer_names(args.layers)
    print(f"Processing {len(layer_names)} layers: {layer_names}")

    # Load representations
    representations_dict = load_layer_representations(representations_dir, layer_names)
    for layer_name, tensor in representations_dict.items():
        print(f"  {layer_name}: {tensor.shape}")

    # Determine label type
    label_type = "classification"
    if labels.dtype == torch.float64 or labels.dtype == torch.float32:
        label_type = "regression"

    if label_type == "regression":
        labels = bin_regression_labels(labels, num_bins=8)
        print(f"Binned regression labels: {torch.unique(labels)}")

    # Parse methods
    methods = [m.strip() for m in args.methods.split(",")]

    # Process each method
    for method in methods:
        method_name = method.lower()
        if method_name == "tsne":
            reduce_fn = run_tsne
        elif method_name == "umap":
            reduce_fn = run_umap
        else:
            raise ValueError(f"Unknown method: {method}")

        print(f"\nRunning {method.upper()}...")

        method_output_dir = output_dir / f"{method_name}"
        method_output_dir.mkdir(exist_ok=True)

        # Prepare layer name mapping
        layer_name_to_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        for layer_name in layer_names:
            if layer_name not in representations_dict:
                print(f"  Skipping {layer_name}: not in representations_dict")
                continue

            print(f"  Processing {layer_name}...")

            # Flatten and reduce
            repr_flat = flatten_representations(representations_dict[layer_name])
            embeddings, color_labels = reduce_fn(
                repr_flat, labels, max_samples=args.max_samples, seed=args.seed
            )
            layer_name_to_embeddings[layer_name] = (embeddings, color_labels)

            # Create individual scatter plot
            layer_name_display = layer_name.replace("_", " ").replace("/", "/")
            scatter_path = method_output_dir / f"{layer_name.replace('_', '-')}_scatter.png"
            create_scatter_plot(
                embeddings, color_labels, f"{method.upper()} — {layer_name_display}", scatter_path
            )
            print(f"    Saved: {scatter_path}")

        # Create layer grid
        if layer_name_to_embeddings:
            grid_path = method_output_dir / f"grid_{method_name}.png"
            create_layer_grid(layer_name_to_embeddings, method_name, grid_path)
            print(f"    Saved: {grid_path}")
        else:
            print("    No layers to plot — skipping grid")

    print(f"\nDone! Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
