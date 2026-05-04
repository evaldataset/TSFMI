"""Run LEACE concept erasure on frozen representations across layers.

Loads pre-extracted representations and labels, fits a LEACE eraser per layer on
the training split, and compares probe performance before/after erasure on the
evaluation split.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_leace_erasure.py \
        --representations_dir outputs/representations/moment/synthetic_trend/ \
        --probe_dir outputs/probes/moment_synthetic_trend_linear/ \
        --output_dir outputs/leace/moment_trend/
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for LEACE erasure experiments.

    Args:
        argv: Argument list. Defaults to sys.argv[1:] if None.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run LEACE erasure and evaluate probe degradation per layer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument(
        "--representations_dir",
        type=str,
        required=True,
        help="Directory containing {layer}.pt, labels.pt, and metadata.json.",
    )
    _ = parser.add_argument(
        "--probe_dir",
        type=str,
        required=True,
        help="Directory with trained linear probes (layer subdirs with probe.pt).",
    )
    _ = parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving LEACE erasure results and plots.",
    )
    _ = parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string ("cuda", "cpu", "cuda:0"). Auto-selects if None.',
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    _ = parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of samples used for evaluation.",
    )
    return parser.parse_args(argv)


def discover_layer_files(repr_dir: Path) -> list[Path]:
    """Find all layer representation files in a directory.

    Args:
        repr_dir: Representations directory path.

    Returns:
        Sorted list of layer .pt file paths.

    Raises:
        FileNotFoundError: If no layer representation files are present.
    """
    layer_files = sorted(f for f in repr_dir.glob("*.pt") if f.stem != "labels")
    if not layer_files:
        raise FileNotFoundError(f"No layer .pt files found in {repr_dir}")
    return layer_files


def prepare_labels(labels: torch.Tensor, label_type: str) -> torch.Tensor:
    """Prepare labels based on task type.

    Args:
        labels: Raw label tensor loaded from disk.
        label_type: Task type, either "classification" or "regression".

    Returns:
        Labels converted to the expected dtype.
    """
    if label_type == "classification":
        return labels.long()
    return labels.float()


def make_train_eval_split(
    num_samples: int,
    val_split: float,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Build deterministic train/eval split indices.

    Args:
        num_samples: Number of samples.
        val_split: Fraction for evaluation.
        seed: RNG seed.

    Returns:
        Tuple of (train_indices, eval_indices).

    Raises:
        ValueError: If val_split is not in (0, 1), or split is degenerate.
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    split = int(num_samples * (1 - val_split))
    if split <= 0 or split >= num_samples:
        raise ValueError(
            f"Invalid split with num_samples={num_samples}, val_split={val_split}: split={split}"
        )
    return indices[:split], indices[split:]


def evaluate_probe_on_data(
    probe: nn.Module,
    representations: torch.Tensor,
    labels: torch.Tensor,
    label_type: str,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a trained probe on representations.

    Args:
        probe: Trained probe.
        representations: Representation tensor of shape (N, D).
        labels: Label tensor of shape (N,).
        label_type: Task type, "classification" or "regression".
        device: Target torch device.

    Returns:
        Dictionary of evaluation metrics.
    """
    _ = probe.eval()
    _ = probe.to(device)

    with torch.no_grad():
        outputs = probe(representations.to(device)).cpu()

    if label_type == "classification":
        preds = outputs.argmax(dim=1)
        true = labels
        accuracy = float((preds == true).float().mean().item())
        return {
            "accuracy": accuracy,
        }

    preds = outputs.squeeze(-1)
    true = labels.float()
    diff = true - preds
    ss_res = float((diff * diff).sum().item())
    centered = true - true.mean()
    ss_tot = float((centered * centered).sum().item())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return {
        "r2": float(r2),
        "mae": float(diff.abs().mean().item()),
    }


def fit_leace_and_erase(
    train_repr: torch.Tensor,
    train_labels: torch.Tensor,
    eval_repr: torch.Tensor,
    label_type: str,
) -> torch.Tensor:
    """Fit LEACE eraser on training data and erase evaluation representations.

    Args:
        train_repr: Training representations of shape (N_train, D).
        train_labels: Training labels of shape (N_train,).
        eval_repr: Evaluation representations of shape (N_eval, D).
        label_type: Task type, "classification" or "regression".

    Returns:
        Erased evaluation representations of shape (N_eval, D).
    """
    concept_erasure = importlib.import_module("concept_erasure")
    leace_fitter_cls = concept_erasure.LeaceFitter

    if label_type == "classification":
        z_train = train_labels.long()
    else:
        z_train = train_labels.float().unsqueeze(-1)

    fitter = leace_fitter_cls.fit(train_repr.float(), z_train)
    eraser = fitter.eraser
    erased = eraser(eval_repr.float())
    if not isinstance(erased, torch.Tensor):
        raise TypeError("LEACE eraser did not return a torch.Tensor")
    return erased


def _plot_before_after(
    results: dict[str, dict[str, object]],
    label_type: str,
    output_dir: Path,
) -> None:
    """Plot before/after primary metric for each layer.

    Args:
        results: Per-layer LEACE evaluation results.
        label_type: Task type, "classification" or "regression".
        output_dir: Output directory for saving plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(results.keys())
    if not layers:
        return

    primary_metric = "accuracy" if label_type == "classification" else "r2"
    before_vals: list[float] = []
    after_vals: list[float] = []
    for layer in layers:
        layer_result = results[layer]
        before = layer_result.get("before")
        after = layer_result.get("after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError(f"Malformed results entry for layer {layer}")
        before_vals.append(float(before[primary_metric]))
        after_vals.append(float(after[primary_metric]))

    x = np.arange(len(layers))
    width = 0.36

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.8), 5))
    _ = ax.bar(x - width / 2, before_vals, width=width, label="Before LEACE", color="steelblue")
    _ = ax.bar(x + width / 2, after_vals, width=width, label="After LEACE", color="coral")

    _ = ax.set_xticks(x)
    _ = ax.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
    _ = ax.set_xlabel("Layer")
    _ = ax.set_ylabel(primary_metric)
    _ = ax.set_title(f"Probe {primary_metric}: Before vs After LEACE Erasure")
    _ = ax.legend()
    _ = plt.tight_layout()

    plot_path = output_dir / "leace_before_after.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {plot_path}")


def main() -> None:
    """Run LEACE erasure and evaluate probe performance across layers."""
    args = parse_args()

    from src.probes.linear_probe import LinearProbe
    from src.utils.device import resolve_device
    from src.utils.seed import seed_everything

    representations_dir = str(args.representations_dir)
    probe_root_dir = str(args.probe_dir)
    output_root_dir = str(args.output_dir)
    device_arg = None if args.device is None else str(args.device)
    seed = int(args.seed)
    val_split = float(args.val_split)

    seed_everything(seed)
    device = resolve_device(device_arg)

    repr_dir = Path(representations_dir)
    probe_dir = Path(probe_root_dir)
    output_dir = Path(output_root_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not repr_dir.exists():
        raise FileNotFoundError(f"Representations directory not found: {repr_dir}")
    if not probe_dir.exists():
        raise FileNotFoundError(f"Probe directory not found: {probe_dir}")

    meta_path = repr_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {repr_dir}")
    metadata_raw = json.loads(meta_path.read_text())
    if not isinstance(metadata_raw, dict):
        raise ValueError(f"metadata.json must contain a JSON object, got {type(metadata_raw)}")
    label_type_value = metadata_raw.get("label_type", "classification")
    label_type = str(label_type_value)
    if label_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported label_type in metadata: {label_type}")

    labels_path = repr_dir / "labels.pt"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.pt not found in {repr_dir}")
    labels_loaded = torch.load(labels_path, map_location="cpu", weights_only=True)
    if not isinstance(labels_loaded, torch.Tensor):
        raise TypeError("labels.pt must contain a torch.Tensor")
    labels = prepare_labels(labels_loaded, label_type)
    if labels.numel() == 0:
        raise ValueError(f"labels.pt is empty: {labels_path}")

    layer_files = discover_layer_files(repr_dir)
    train_idx, eval_idx = make_train_eval_split(len(labels), val_split, seed)
    eval_labels = labels[eval_idx]

    print(f"Device: {device}")
    print(f"Label type: {label_type}")
    print(f"Samples: {len(labels)} (train={len(train_idx)}, eval={len(eval_idx)})")
    print(f"Layers: {len(layer_files)}")
    print()

    primary_metric = "accuracy" if label_type == "classification" else "r2"
    all_results: dict[str, dict[str, object]] = {}

    for layer_file in layer_files:
        layer_name = layer_file.stem

        layer_loaded = torch.load(layer_file, map_location="cpu", weights_only=True)
        if not isinstance(layer_loaded, torch.Tensor):
            print(f"[{layer_name}] Layer file does not contain a tensor, skipping")
            continue
        representations = layer_loaded
        if representations.ndim >= 3:
            representations = representations.view(representations.shape[0], -1)
        representations = representations.float()

        if representations.ndim != 2:
            print(
                f"[{layer_name}] Unexpected representation shape {representations.shape}, skipping"
            )
            continue
        if representations.shape[0] != labels.shape[0]:
            print(
                f"[{layer_name}] Sample mismatch repr={representations.shape[0]} "
                + f"vs labels={labels.shape[0]}, skipping"
            )
            continue

        probe_path = probe_dir / layer_name / "probe.pt"
        if not probe_path.exists():
            print(f"[{layer_name}] Probe not found at {probe_path}, skipping")
            continue

        input_dim = representations.shape[1]
        output_dim = int(labels.max().item()) + 1 if label_type == "classification" else 1

        probe = LinearProbe(input_dim=input_dim, output_dim=output_dim)
        state_obj = torch.load(probe_path, map_location="cpu", weights_only=True)
        if not isinstance(state_obj, dict):
            print(f"[{layer_name}] Probe checkpoint is not a state dict, skipping")
            continue
        _ = probe.load_state_dict(state_obj)

        train_repr = representations[train_idx]
        eval_repr = representations[eval_idx]
        train_labels = labels[train_idx]

        erased_eval_repr = fit_leace_and_erase(train_repr, train_labels, eval_repr, label_type)

        before_metrics = evaluate_probe_on_data(probe, eval_repr, eval_labels, label_type, device)
        after_metrics = evaluate_probe_on_data(
            probe, erased_eval_repr, eval_labels, label_type, device
        )
        drop = float(before_metrics[primary_metric] - after_metrics[primary_metric])

        all_results[layer_name] = {
            "before": before_metrics,
            "after": after_metrics,
            "drop": drop,
        }

        message = (
            f"[{layer_name}] {primary_metric}: before={before_metrics[primary_metric]:.4f} "
            + f"after={after_metrics[primary_metric]:.4f} drop={drop:.4f}"
        )
        print(message)

    results_path = output_dir / "leace_results.json"
    _ = results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results to {results_path}")

    _plot_before_after(all_results, label_type, output_dir)


if __name__ == "__main__":
    main()
