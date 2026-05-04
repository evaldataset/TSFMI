"""Reduce high-dimensional representations via PCA for tractable CKA and MLP probing.

Loads per-layer .pt files, flattens to 2D, fits PCA, and saves reduced representations
to a new directory alongside the original metadata and labels.

Usage:
    python scripts/reduce_representations.py \
        --input_dir outputs/representations/moment/synthetic_trend \
        --output_dir outputs/representations/moment_pca512/synthetic_trend \
        --n_components 512
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for PCA reduction.

    Args:
        argv: Argument list. Defaults to sys.argv[1:] if None.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Reduce representation dimensionality via PCA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing {layer}.pt files and labels.pt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving PCA-reduced representations.",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=512,
        help="Target dimensionality after PCA reduction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of samples to use for fitting PCA (train split). "
        "Set to 1.0 to fit on all data (legacy behavior).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Apply PCA to each layer's representations and save reduced versions."""
    args = parse_args()

    import numpy as np
    import torch
    from sklearn.decomposition import PCA

    from src.utils.seed import seed_everything

    seed_everything(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Copy labels and metadata unchanged
    for fname in ("labels.pt", "metadata.json"):
        src = input_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    # Update metadata with PCA info
    meta_path = output_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["pca_n_components"] = args.n_components
        meta["pca_source_dir"] = str(input_dir)
        meta["pca_train_ratio"] = args.train_ratio
        meta["pca_fit_scope"] = "train_only" if args.train_ratio < 1.0 else "all_data"
        meta["pca_seed"] = args.seed
        meta_path.write_text(json.dumps(meta, indent=2))

    # Discover layer files
    layer_files = sorted(f for f in input_dir.glob("*.pt") if f.stem != "labels")
    if not layer_files:
        raise FileNotFoundError(f"No layer .pt files found in {input_dir}")

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Layers: {len(layer_files)}, Target dim: {args.n_components}")

    for lf in layer_files:
        tensor = torch.load(lf, map_location="cpu", weights_only=True)
        original_shape = tensor.shape

        # Flatten to 2D: (N, features)
        flat = tensor.view(tensor.shape[0], -1).numpy().astype(np.float32)
        original_dim = flat.shape[1]

        if original_dim <= args.n_components:
            print(f"  [{lf.stem}] dim={original_dim} <= {args.n_components}, copying unchanged")
            torch.save(tensor, output_dir / lf.name)
            continue

        # Clamp n_components to valid range for PCA
        max_components = min(flat.shape[0], flat.shape[1])
        effective_components = min(args.n_components, max_components)
        if effective_components < args.n_components:
            print(
                f"  [{lf.stem}] Clamping n_components from {args.n_components}"
                f" to {effective_components} (min(n_samples={flat.shape[0]},"
                f" n_features={flat.shape[1]}))"
            )

        # Fit PCA on train split only, then transform all data
        pca = PCA(n_components=effective_components, random_state=args.seed)
        n_train = int(flat.shape[0] * args.train_ratio)
        if args.train_ratio < 1.0 and n_train >= effective_components:
            pca.fit(flat[:n_train])
            reduced = pca.transform(flat)
        else:
            reduced = pca.fit_transform(flat)
        explained_var = pca.explained_variance_ratio_.sum()

        reduced_tensor = torch.from_numpy(reduced)
        torch.save(reduced_tensor, output_dir / lf.name)

        print(
            f"  [{lf.stem}] {original_shape} ({original_dim}D)"
            f" → ({reduced_tensor.shape[0]}, {args.n_components})"
            f" | explained variance: {explained_var:.4f}"
        )

    print(f"\nDone! Reduced representations saved to {output_dir}")


if __name__ == "__main__":
    main()
