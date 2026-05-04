"""CLI script for training linear probes on pre-extracted representations.

Loads per-layer representation tensors and labels from a representations directory,
trains a LinearProbe or MLPControlProbe per layer, and saves probe weights + metrics.

Usage:
    python scripts/train_probe.py \
        --representations_dir outputs/representations/moment/synthetic_trend/ \
        --probe_type linear \
        --output_dir outputs/probes/moment/trend/ \
        --epochs 100 \
        --verbose
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for probe training.

    Args:
        argv: Argument list. Defaults to sys.argv[1:] if None.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train linear probes on pre-extracted frozen representations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--representations_dir",
        type=str,
        required=True,
        help="Directory containing {layer}.pt and labels.pt files (output of"
        " extract_representations.py).",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        choices=["linear", "mlp_control"],
        default="linear",
        help="Probe architecture to train.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/probes/",
        help="Root directory for saving trained probes and metrics.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Adam optimizer learning rate.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size for training.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data held out for validation / early stopping.",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Fraction of data held out for final test reporting (0 = no test).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string ("cuda", "cpu", "cuda:0"). Auto-selects if None.',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=None,
        help="Hidden layer width for mlp_control probe. Ignored for linear probe.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-epoch training metrics.",
    )
    return parser.parse_args(argv)


def discover_layer_files(repr_dir: Path) -> list[Path]:
    """Find all layer representation .pt files in a directory.

    Excludes labels.pt and returns files sorted alphabetically.

    Args:
        repr_dir: Path to the representations directory.

    Returns:
        Sorted list of layer .pt file paths.

    Raises:
        FileNotFoundError: If no layer .pt files are found.
    """
    layer_files = sorted(f for f in repr_dir.glob("*.pt") if f.stem != "labels")
    if not layer_files:
        raise FileNotFoundError(f"No layer .pt files found in {repr_dir}")
    return layer_files


def load_metadata(repr_dir: Path) -> dict[str, object]:
    """Load extraction metadata if available.

    Args:
        repr_dir: Path to the representations directory.

    Returns:
        Metadata dict, or empty dict if metadata.json doesn't exist.
    """
    meta_path = repr_dir / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {}


def main() -> None:
    """Train probes on pre-extracted representations for each layer."""
    args = parse_args()

    # Lazy imports — keep --help fast without torch
    import torch

    from src.probes.linear_probe import LinearProbe
    from src.probes.mlp_control_probe import MLPControlProbe
    from src.probes.probe_trainer import ProbeTrainer, ProbeTrainerConfig
    from src.utils.device import resolve_device
    from src.utils.seed import seed_everything

    seed_everything(args.seed)
    device = resolve_device(args.device)

    repr_dir = Path(args.representations_dir)
    if not repr_dir.exists():
        raise FileNotFoundError(f"Representations directory not found: {repr_dir}")

    # Load labels
    labels_path = repr_dir / "labels.pt"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    labels: torch.Tensor = torch.load(labels_path, map_location="cpu", weights_only=True)
    if labels.numel() == 0:
        raise ValueError(f"Labels tensor is empty: {labels_path}")

    # Discover layer files
    layer_files = discover_layer_files(repr_dir)

    # Determine task type from metadata
    metadata = load_metadata(repr_dir)
    label_type = str(metadata.get("label_type", "classification"))

    print(f"Device: {device}")
    print(f"Probe type: {args.probe_type}")
    print(f"Label type: {label_type}")
    print(f"Labels shape: {labels.shape}")
    print(f"Layers to train: {len(layer_files)}")
    print(f"Epochs: {args.epochs}, LR: {args.learning_rate}, Batch: {args.batch_size}")
    print()

    # Prepare labels based on task type
    if label_type == "classification":
        labels_prepared = labels.long()
        num_classes = int(labels_prepared.max().item()) + 1
        output_dim = num_classes
    else:
        labels_prepared = labels.float()
        output_dim = 1

    all_layer_metrics: dict[str, dict[str, float]] = {}

    for layer_file in layer_files:
        layer_name = layer_file.stem
        representations: torch.Tensor = torch.load(
            layer_file,
            map_location="cpu",
            weights_only=True,
        )

        # Flatten high-dimensional representations to 2D (N, features)
        # (N, variates, d_model) → (N, variates * d_model)
        # (N, channels, patches, d_model) → (N, channels * patches * d_model)
        if representations.ndim >= 3:
            n = representations.shape[0]
            representations = representations.view(n, -1)

        if representations.ndim != 2:
            print(f"[{layer_name}] Skipping — unexpected shape {representations.shape}")
            continue

        input_dim = representations.shape[1]

        # Validate sample count matches labels
        if representations.shape[0] != labels_prepared.shape[0]:
            print(
                f"[{layer_name}] Skipping — sample count mismatch: "
                f"representations={representations.shape[0]}, labels={labels_prepared.shape[0]}"
            )
            continue

        # Create probe
        if args.probe_type == "linear":
            probe = LinearProbe(input_dim=input_dim, output_dim=output_dim)
        else:
            probe = MLPControlProbe(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=args.hidden_dim,
            )

        # Configure trainer
        config = ProbeTrainerConfig(
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            probe_type=label_type,
            val_split=args.val_split,
            test_split=args.test_split,
            split_seed=args.seed,
            device=str(device),
            verbose=args.verbose,
        )
        trainer = ProbeTrainer(probe, config)

        print(
            f"[{layer_name}] Training {args.probe_type} probe"
            f" — input_dim={input_dim}, output_dim={output_dim}"
        )
        trained_probe, metrics = trainer.train(representations, labels_prepared)

        # Save probe weights and metrics
        layer_out_dir = Path(args.output_dir) / layer_name
        layer_out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(trained_probe.state_dict(), layer_out_dir / "probe.pt")
        (layer_out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        all_layer_metrics[layer_name] = metrics
        print(f"[{layer_name}] Saved probe + metrics to {layer_out_dir}")
        print()

    # Save summary across all layers
    if all_layer_metrics:
        summary_path = Path(args.output_dir) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "probe_type": args.probe_type,
            "label_type": label_type,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "layers": all_layer_metrics,
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
