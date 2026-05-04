"""Measure downstream forecasting impact of LEACE concept erasure.

Interventional experiment: erase a temporal property from intermediate
representations via LEACE, then measure how reconstruction quality degrades.
This quantifies the causal importance of encoded concepts for downstream tasks.

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/run_downstream_impact.py
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.datasets.synthetic import (
    generate_stationarity_dataset,
    generate_trend_dataset,
)
from src.utils.seed import seed_everything


@dataclass
class InterventionResult:
    """Result of a single intervention experiment."""

    model_name: str
    property_name: str
    layer_name: str
    mse_original: float
    mse_erased: float
    mse_increase_pct: float
    mae_original: float
    mae_erased: float


def load_moment_model(device: torch.device) -> nn.Module:
    """Load frozen MOMENT model for reconstruction."""
    from momentfm import MOMENTPipeline

    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction"},
    )
    if not isinstance(model, nn.Module):
        raise RuntimeError("MOMENTPipeline is not an nn.Module")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def prepare_moment_input(sequences: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (N, seq_len) sequences to MOMENT input format (N, 1, seq_len)."""
    x = torch.from_numpy(sequences).float().unsqueeze(1).to(device)
    return x


def get_reconstruction(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Get MOMENT reconstruction output."""
    with torch.no_grad():
        output = model(x_enc=x)
    return output.reconstruction  # (N, 1, seq_len)


def fit_leace_eraser(
    representations: torch.Tensor,
    labels: torch.Tensor,
    label_type: str,
) -> object:
    """Fit LEACE eraser on representations.

    Args:
        representations: Representations of shape (N, D) or (N, T, D).
        labels: Labels of shape (N,).
        label_type: "classification" or "regression".

    Returns:
        Fitted LEACE eraser object.
    """
    concept_erasure = importlib.import_module("concept_erasure")

    # For 3D: mean-pool over sequence dim to (N, D) for efficient LEACE fitting
    if representations.ndim == 3:
        pooled_repr = representations.mean(dim=1).float()  # (N, D)
        flat_repr = pooled_repr
        flat_labels = labels
    else:
        flat_repr = representations.float()
        flat_labels = labels

    if label_type == "classification":
        z = flat_labels.long()
    else:
        z = flat_labels.float().unsqueeze(-1)

    fitter = concept_erasure.LeaceFitter.fit(flat_repr.cpu(), z.cpu())
    return fitter.eraser


def run_intervention(
    model: nn.Module,
    x: torch.Tensor,
    layer_name: str,
    eraser: object,
    device: torch.device,
) -> torch.Tensor:
    """Run MOMENT forward pass with LEACE-erased representations at a specific layer.

    Args:
        model: Frozen MOMENT model.
        x: Input tensor of shape (N, 1, seq_len).
        layer_name: Encoder block name (e.g., "encoder.block.23").
        eraser: Fitted LEACE eraser.
        device: Device.

    Returns:
        Reconstruction tensor after intervention of shape (N, 1, seq_len).
    """
    # Parse layer index from name
    parts = layer_name.split(".")
    layer_idx = int(parts[-1])

    def intervention_hook(
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Replace hidden states with LEACE-erased version."""
        hidden_states = output[0]  # (N, T, D)
        original_shape = hidden_states.shape

        # Flatten to 2D for LEACE
        flat = hidden_states.reshape(-1, hidden_states.shape[-1]).float().cpu()
        erased = eraser(flat)
        if not isinstance(erased, torch.Tensor):
            raise TypeError("LEACE eraser did not return a torch.Tensor")

        erased_reshaped = erased.reshape(original_shape).to(device)

        # Return modified output (keep attention weights unchanged)
        return (erased_reshaped,) + output[1:]

    handle = model.encoder.block[layer_idx].register_forward_hook(intervention_hook)
    try:
        with torch.no_grad():
            output = model(x_enc=x)
        return output.reconstruction
    finally:
        handle.remove()


def compute_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> tuple[float, float]:
    """Compute MSE and MAE between original input and reconstruction."""
    diff = (original - reconstructed).float()
    mse = float(diff.pow(2).mean().item())
    mae = float(diff.abs().mean().item())
    return mse, mae


def run_experiment_for_property(
    model: nn.Module,
    property_name: str,
    device: torch.device,
    layers_to_test: list[int],
    num_samples: int = 2000,
    batch_size: int = 64,
) -> list[InterventionResult]:
    """Run downstream impact experiment for a single property.

    Args:
        model: Frozen MOMENT model.
        property_name: "trend" or "stationarity".
        device: Device.
        layers_to_test: List of encoder block indices to test.
        num_samples: Number of synthetic samples.
        batch_size: Batch size for forward passes.

    Returns:
        List of InterventionResult for each tested layer.
    """
    # Generate data
    if property_name == "trend":
        dataset = generate_trend_dataset(num_samples=num_samples, seq_len=512, seed=42)
        label_type = "classification"
    elif property_name == "stationarity":
        dataset = generate_stationarity_dataset(num_samples=num_samples, seq_len=512, seed=42)
        label_type = "classification"
    else:
        raise ValueError(f"Unsupported property: {property_name}")

    x_all = prepare_moment_input(dataset.sequences, device)
    labels = torch.from_numpy(dataset.labels)

    results: list[InterventionResult] = []

    for layer_idx in layers_to_test:
        layer_name = f"encoder.block.{layer_idx}"
        print(f"  Layer {layer_idx}: extracting representations...")

        # Step 1: Extract representations from this layer via hook
        layer_reprs: list[torch.Tensor] = []

        append_repr = layer_reprs.append

        def capture_hook(
            module: nn.Module,
            input_args: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, ...],
            *,
            append_repr_fn=append_repr,
        ) -> None:
            append_repr_fn(output[0].cpu())

        handle = model.encoder.block[layer_idx].register_forward_hook(capture_hook)

        # Run forward in batches to capture representations
        for i in range(0, num_samples, batch_size):
            batch = x_all[i : i + batch_size]
            with torch.no_grad():
                _ = model(x_enc=batch)

        handle.remove()
        all_reprs = torch.cat(layer_reprs, dim=0)  # (N, T, D)
        print(f"    Representations shape: {all_reprs.shape}")

        # Step 2: Fit LEACE eraser
        print("    Fitting LEACE eraser...")
        eraser = fit_leace_eraser(all_reprs, labels, label_type)

        # Step 3: Compute original reconstruction MSE (in batches)
        print("    Computing original reconstruction...")
        orig_recons: list[torch.Tensor] = []
        for i in range(0, num_samples, batch_size):
            batch = x_all[i : i + batch_size]
            recon = get_reconstruction(model, batch)
            orig_recons.append(recon.cpu())
        orig_reconstruction = torch.cat(orig_recons, dim=0)

        # Step 4: Compute intervened reconstruction MSE (in batches)
        print("    Computing intervened reconstruction...")
        erased_recons: list[torch.Tensor] = []
        for i in range(0, num_samples, batch_size):
            batch = x_all[i : i + batch_size]
            recon = run_intervention(model, batch, layer_name, eraser, device)
            erased_recons.append(recon.cpu())
        erased_reconstruction = torch.cat(erased_recons, dim=0)

        # Step 5: Compute metrics
        input_cpu = x_all.cpu()
        mse_orig, mae_orig = compute_metrics(input_cpu, orig_reconstruction)
        mse_erased, mae_erased = compute_metrics(input_cpu, erased_reconstruction)
        increase_pct = ((mse_erased - mse_orig) / max(mse_orig, 1e-8)) * 100

        result = InterventionResult(
            model_name="MOMENT",
            property_name=property_name,
            layer_name=layer_name,
            mse_original=mse_orig,
            mse_erased=mse_erased,
            mse_increase_pct=increase_pct,
            mae_original=mae_orig,
            mae_erased=mae_erased,
        )
        results.append(result)

        print(
            f"    MSE: {mse_orig:.6f} → {mse_erased:.6f} "
            f"(+{increase_pct:.1f}%), MAE: {mae_orig:.4f} → {mae_erased:.4f}"
        )

    return results


def plot_downstream_impact(
    all_results: dict[str, list[InterventionResult]],
    output_dir: Path,
) -> None:
    """Create downstream impact figure.

    Args:
        all_results: Mapping from property name to list of results.
        output_dir: Output directory.
    """
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

    properties = list(all_results.keys())
    n_props = len(properties)
    fig, axes = plt.subplots(1, n_props, figsize=(4.5 * n_props, 3.8), squeeze=False)

    prop_colors = {"trend": "#d62728", "stationarity": "#1f77b4"}

    for col, prop_name in enumerate(properties):
        ax = axes[0, col]
        results = all_results[prop_name]

        layers = [r.layer_name.split(".")[-1] for r in results]
        layer_indices = list(range(len(layers)))
        mse_increases = [r.mse_increase_pct for r in results]

        color = prop_colors.get(prop_name, "#333333")
        ax.bar(layer_indices, mse_increases, color=color, alpha=0.8, width=0.6)
        ax.set_xticks(layer_indices)
        ax.set_xticklabels([f"L{layer}" for layer in layers], rotation=45, ha="right")
        ax.set_xlabel("Encoder Layer")
        ax.set_ylabel("MSE Increase (%)")
        ax.set_title(f"Erasing {prop_name.title()}")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.suptitle(
        "Downstream Reconstruction Impact of LEACE Erasure (MOMENT)",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    stem = "fig16_downstream_impact"
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_dir / stem}.pdf")


def main() -> None:
    """Run downstream impact analysis."""
    seed_everything(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("outputs/paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = Path("outputs/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MOMENT model...")
    model = load_moment_model(device)

    # Test key layers: early (0), middle (11), and late (23)
    layers_to_test = [0, 11, 23]

    all_results: dict[str, list[InterventionResult]] = {}

    for prop_name in ["trend", "stationarity"]:
        print(f"\n{'=' * 60}")
        print(f"Property: {prop_name}")
        print(f"{'=' * 60}")
        results = run_experiment_for_property(
            model, prop_name, device, layers_to_test, num_samples=500, batch_size=64
        )
        all_results[prop_name] = results

    # Save results as JSON
    json_results: dict[str, list[dict[str, object]]] = {}
    for prop_name, results in all_results.items():
        json_results[prop_name] = [
            {
                "model": r.model_name,
                "property": r.property_name,
                "layer": r.layer_name,
                "mse_original": r.mse_original,
                "mse_erased": r.mse_erased,
                "mse_increase_pct": r.mse_increase_pct,
                "mae_original": r.mae_original,
                "mae_erased": r.mae_erased,
            }
            for r in results
        ]

    results_path = analysis_dir / "downstream_impact_results.json"
    results_path.write_text(json.dumps(json_results, indent=2))
    print(f"\nSaved results to {results_path}")

    # Plot
    plot_downstream_impact(all_results, output_dir)


if __name__ == "__main__":
    main()
