"""Intervention experiments: LDA steering vectors on frozen representations.

Computes LDA-based steering directions from class-conditional representations,
applies additive perturbations at varying strengths (alpha), and measures the
effect on linear probe accuracy. Follows Wiliński et al. (2024) methodology.

Usage:
    python scripts/run_intervention.py \
        --representations_dir outputs/representations/moment_pca512/synthetic_trend \
        --probe_dir outputs/probes/moment_pca512_synthetic_trend_linear \
        --output_dir outputs/interventions/moment_pca512_trend \
        --property trend
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, r2_score


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for intervention experiments."""
    parser = argparse.ArgumentParser(
        description="Run LDA steering vector intervention experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--representations_dir",
        type=str,
        required=True,
        help="Directory containing {layer}.pt and labels.pt.",
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        required=True,
        help="Directory with trained linear probes (layer subdirs with probe.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for saving intervention results.",
    )
    parser.add_argument(
        "--alpha_min",
        type=float,
        default=-2.0,
        help="Minimum steering strength.",
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        default=2.0,
        help="Maximum steering strength.",
    )
    parser.add_argument(
        "--alpha_steps",
        type=int,
        default=41,
        help="Number of alpha values to sweep.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data for evaluation (steering computed on rest).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string. Auto-selects if None.",
    )
    return parser.parse_args(argv)


def discover_layer_files(repr_dir: Path) -> list[Path]:
    """Find all layer .pt files (excluding labels.pt) sorted by name."""
    layer_files = sorted(f for f in repr_dir.glob("*.pt") if f.stem != "labels")
    if not layer_files:
        raise FileNotFoundError(f"No layer .pt files found in {repr_dir}")
    return layer_files


def compute_lda_steering_vector(
    representations: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute LDA-based steering direction for maximum class separation.

    For binary classification, returns a single direction vector.
    For multi-class, returns the first LDA discriminant direction.

    Args:
        representations: (N, D) array of representations.
        labels: (N,) integer class labels.

    Returns:
        (D,) steering direction vector (unit-normalized).
    """
    n_classes = len(np.unique(labels))
    n_components = min(n_classes - 1, representations.shape[1])

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(representations, labels)

    # Use the first LDA discriminant direction
    direction = lda.coef_[0] if lda.coef_.ndim == 2 else lda.coef_
    direction = direction.flatten().astype(np.float64)

    # Normalize to unit length
    norm = np.linalg.norm(direction)
    if norm > 1e-10:
        direction = direction / norm

    return direction


def compute_mean_shift_vector(
    representations: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Compute mean-difference steering vector (class 1 mean - class 0 mean).

    For multi-class, computes direction from overall mean to class-0 mean
    (and the vector is interpretable as the "concept direction").

    Args:
        representations: (N, D) array of representations.
        labels: (N,) integer class labels.

    Returns:
        (D,) steering direction vector (unit-normalized).
    """
    classes = np.unique(labels)

    if len(classes) == 2:
        mean_0 = representations[labels == classes[0]].mean(axis=0)
        mean_1 = representations[labels == classes[1]].mean(axis=0)
        direction = mean_1 - mean_0
    else:
        # Multi-class: direction from global mean to class-0 mean
        global_mean = representations.mean(axis=0)
        class_0_mean = representations[labels == classes[0]].mean(axis=0)
        direction = class_0_mean - global_mean

    norm = np.linalg.norm(direction)
    if norm > 1e-10:
        direction = direction / norm

    return direction


def evaluate_probe_on_data(
    probe: nn.Module,
    representations: torch.Tensor,
    labels: torch.Tensor,
    label_type: str,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a trained probe on given representations.

    Args:
        probe: Trained linear probe module.
        representations: (N, D) tensor.
        labels: (N,) tensor.
        label_type: "classification" or "regression".
        device: Torch device.

    Returns:
        Dict with metric values.
    """
    probe.eval()
    probe.to(device)

    with torch.no_grad():
        inputs = representations.to(device)
        outputs = probe(inputs).cpu()

    if label_type == "classification":
        preds = outputs.argmax(dim=1).numpy()
        true = labels.numpy()
        return {
            "accuracy": float(accuracy_score(true, preds)),
            "f1_macro": float(f1_score(true, preds, average="macro", zero_division="warn")),
        }
    else:
        preds = outputs.squeeze().numpy()
        true = labels.numpy()
        return {
            "r2": float(r2_score(true, preds)),
            "mae": float(np.abs(true - preds).mean()),
        }


def main() -> None:
    """Run LDA steering intervention experiments across layers."""
    args = parse_args()

    from src.probes.linear_probe import LinearProbe
    from src.utils.device import resolve_device
    from src.utils.seed import seed_everything

    seed_everything(args.seed)
    device = resolve_device(args.device)

    repr_dir = Path(args.representations_dir)
    probe_dir = Path(args.probe_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta_path = repr_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {repr_dir}")
    meta = json.loads(meta_path.read_text())
    label_type = meta.get("label_type", "classification")

    # Load labels
    labels_path = repr_dir / "labels.pt"
    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    if label_type == "classification":
        labels = labels.long()
    else:
        labels = labels.float()
    if labels.numel() == 0:
        raise ValueError(f"labels.pt is empty: {labels_path}")

    # Train/eval split
    n = len(labels)
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(n)
    split = int(n * (1 - args.val_split))
    train_idx = indices[:split]
    eval_idx = indices[split:]

    train_labels = labels[train_idx]
    eval_labels = labels[eval_idx]

    # Alpha sweep
    alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)

    # Discover layers
    layer_files = discover_layer_files(repr_dir)

    print(f"Device: {device}")
    print(f"Label type: {label_type}")
    print(f"Samples: {n} (train={len(train_idx)}, eval={len(eval_idx)})")
    print(f"Layers: {len(layer_files)}")
    print(f"Alpha range: [{args.alpha_min}, {args.alpha_max}], steps={args.alpha_steps}")
    print()

    all_results: dict[str, Any] = {}

    for lf in layer_files:
        layer_name = lf.stem

        # Load representations
        full_repr = torch.load(lf, map_location="cpu", weights_only=True)
        if full_repr.ndim >= 3:
            full_repr = full_repr.view(full_repr.shape[0], -1)
        full_repr = full_repr.float()

        train_repr = full_repr[train_idx]
        eval_repr = full_repr[eval_idx]

        # Load trained probe
        probe_path = probe_dir / layer_name / "probe.pt"
        if not probe_path.exists():
            print(f"[{layer_name}] No probe found, skipping")
            continue

        input_dim = full_repr.shape[1]
        if label_type == "classification":
            output_dim = int(labels.max().item()) + 1
        else:
            output_dim = 1

        probe = LinearProbe(input_dim=input_dim, output_dim=output_dim)
        probe.load_state_dict(torch.load(probe_path, map_location="cpu", weights_only=True))

        # Baseline accuracy (no perturbation)
        baseline = evaluate_probe_on_data(probe, eval_repr, eval_labels, label_type, device)

        # Compute LDA steering vector from training set
        train_np = train_repr.numpy()
        train_labels_np = train_labels.numpy()

        n_unique = len(np.unique(train_labels_np))
        if label_type == "classification" and n_unique >= 2:
            try:
                steering_lda = compute_lda_steering_vector(train_np, train_labels_np)
            except Exception as e:
                print(f"[{layer_name}] LDA failed: {e}, using mean-shift")
                steering_lda = compute_mean_shift_vector(train_np, train_labels_np)
        else:
            # Regression: use mean-shift between above/below median
            median_val = np.median(train_labels_np)
            binary_labels = (train_labels_np > median_val).astype(np.int64)
            steering_lda = compute_lda_steering_vector(train_np, binary_labels)

        steering_mean = compute_mean_shift_vector(
            train_np,
            train_labels_np
            if label_type == "classification"
            else (train_labels_np > np.median(train_labels_np)).astype(np.int64),
        )

        # Scale steering vectors to match representation magnitude
        repr_scale = float(np.std(train_np))

        # Sweep alpha values
        lda_results: list[dict[str, float]] = []
        mean_results: list[dict[str, float]] = []

        for alpha in alphas:
            # LDA steering
            perturbed_lda = eval_repr + alpha * repr_scale * torch.from_numpy(steering_lda).float()
            metrics_lda = evaluate_probe_on_data(
                probe,
                perturbed_lda,
                eval_labels,
                label_type,
                device,
            )
            metrics_lda["alpha"] = float(alpha)
            lda_results.append(metrics_lda)

            # Mean-shift steering
            perturbed_mean = (
                eval_repr + alpha * repr_scale * torch.from_numpy(steering_mean).float()
            )
            metrics_mean = evaluate_probe_on_data(
                probe,
                perturbed_mean,
                eval_labels,
                label_type,
                device,
            )
            metrics_mean["alpha"] = float(alpha)
            mean_results.append(metrics_mean)

        # Store results
        primary_metric = "accuracy" if label_type == "classification" else "r2"
        baseline_val = baseline[primary_metric]
        best_drop_lda = baseline_val - min(r[primary_metric] for r in lda_results)
        best_drop_mean = baseline_val - min(r[primary_metric] for r in mean_results)

        layer_result = {
            "baseline": baseline,
            "lda_sweep": lda_results,
            "mean_sweep": mean_results,
            "primary_metric": primary_metric,
            "baseline_value": baseline_val,
            "max_drop_lda": float(best_drop_lda),
            "max_drop_mean": float(best_drop_mean),
            "repr_scale": float(repr_scale),
        }
        all_results[layer_name] = layer_result

        print(
            f"[{layer_name}] baseline {primary_metric}={baseline_val:.4f}"
            f"  |  LDA max_drop={best_drop_lda:.4f}"
            f"  |  mean max_drop={best_drop_mean:.4f}"
        )

    # Save results
    results_path = output_dir / "intervention_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved intervention results to {results_path}")

    # Generate intervention plots
    _plot_intervention_curves(all_results, output_dir, label_type)


def _plot_intervention_curves(
    results: dict[str, Any],
    output_dir: Path,
    label_type: str,
) -> None:
    """Generate per-layer intervention sweep plots.

    Creates a grid of subplots showing probe accuracy vs. alpha for each layer,
    comparing LDA and mean-shift steering methods.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(results.keys())
    n_layers = len(layers)
    if n_layers == 0:
        return

    primary_metric = "accuracy" if label_type == "classification" else "r2"
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for idx, layer_name in enumerate(layers):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        lr = results[layer_name]
        lda_alphas = [r["alpha"] for r in lr["lda_sweep"]]
        lda_vals = [r[primary_metric] for r in lr["lda_sweep"]]
        mean_alphas = [r["alpha"] for r in lr["mean_sweep"]]
        mean_vals = [r[primary_metric] for r in lr["mean_sweep"]]
        baseline = lr["baseline_value"]

        ax.plot(lda_alphas, lda_vals, "b-", label="LDA", linewidth=1.5)
        ax.plot(mean_alphas, mean_vals, "r--", label="Mean-shift", linewidth=1.5)
        ax.axhline(y=baseline, color="gray", linestyle=":", alpha=0.7, label="baseline")
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(layer_name, fontsize=9)
        ax.set_xlabel("α", fontsize=8)
        ax.set_ylabel(primary_metric, fontsize=8)
        ax.tick_params(labelsize=7)

        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n_layers, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.suptitle(f"Intervention: Probe {primary_metric} vs. Steering Strength", fontsize=12)
    plt.tight_layout()

    plot_path = output_dir / "intervention_curves.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved intervention plot to {plot_path}")

    # Also generate a summary plot: max drop per layer
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    layer_indices = range(len(layers))
    lda_drops = [results[layer]["max_drop_lda"] for layer in layers]
    mean_drops = [results[layer]["max_drop_mean"] for layer in layers]

    bar_width = 0.35
    ax2.bar(
        [x - bar_width / 2 for x in layer_indices],
        lda_drops,
        bar_width,
        label="LDA steering",
        color="steelblue",
    )
    ax2.bar(
        [x + bar_width / 2 for x in layer_indices],
        mean_drops,
        bar_width,
        label="Mean-shift steering",
        color="coral",
    )
    ax2.set_xticks(list(layer_indices))
    ax2.set_xticklabels(layers, rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel(f"Max {primary_metric} drop")
    ax2.set_title("Intervention Effect by Layer")
    ax2.legend()
    plt.tight_layout()

    summary_path = output_dir / "intervention_summary.png"
    fig2.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved intervention summary to {summary_path}")


if __name__ == "__main__":
    main()
