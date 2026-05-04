"""Measure downstream impact of LEACE concept erasure on GPT4TS.

GPT4TS uses a frozen GPT-2 backbone for time series. Since there is no standard
forecast pipeline, we measure downstream impact via reconstruction: the model's
output (last hidden state) is projected back to sequence space and compared with
the input. LEACE erasure at intermediate layers degrades this mapping.

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/run_gpt4ts_downstream_impact.py
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.datasets.synthetic import (
    generate_stationarity_dataset,
    generate_trend_dataset,
)
from src.models.gpt4ts_wrapper import GPT4TSWrapper
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


def load_gpt4ts_model(device: torch.device) -> GPT4TSWrapper:
    """Load frozen GPT4TS model."""
    wrapper = GPT4TSWrapper(seq_len=512, patch_size=16, stride=16, d_model=768)
    wrapper.load("gpt2", device=device)
    return wrapper


def get_output(wrapper: GPT4TSWrapper, x: torch.Tensor) -> torch.Tensor:
    """Get GPT4TS output (last hidden state) for reconstruction comparison.

    Args:
        wrapper: Loaded GPT4TSWrapper.
        x: Input tensor of shape (N, seq_len).

    Returns:
        Output hidden state of shape (N, num_patches, 768).
    """
    return wrapper.forward(x)  # (N, 32, 768)


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

    if representations.ndim == 3:
        flat_repr = representations.mean(dim=1).float()
    else:
        flat_repr = representations.float()

    if label_type == "classification":
        z = labels.long()
    else:
        z = labels.float().unsqueeze(-1)

    fitter = concept_erasure.LeaceFitter.fit(flat_repr.cpu(), z.cpu())
    return fitter.eraser


def run_intervention(
    wrapper: GPT4TSWrapper,
    x: torch.Tensor,
    layer_idx: int,
    eraser: object,
    device: torch.device,
) -> torch.Tensor:
    """Run GPT4TS forward pass with LEACE-erased representations.

    Args:
        wrapper: Loaded GPT4TSWrapper.
        x: Input tensor of shape (N, seq_len).
        layer_idx: Transformer block index (0-11).
        eraser: Fitted LEACE eraser.
        device: Device.

    Returns:
        Output hidden state after intervention.
    """
    model = wrapper.model  # _GPT2Backbone

    def intervention_hook(
        module: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Replace hidden states with LEACE-erased version."""
        hidden_states = output[0]  # (N, T, D)
        original_shape = hidden_states.shape

        flat = hidden_states.reshape(-1, hidden_states.shape[-1]).float().cpu()
        erased = eraser(flat)
        if not isinstance(erased, torch.Tensor):
            raise TypeError("LEACE eraser did not return a torch.Tensor")

        erased_reshaped = erased.reshape(original_shape).to(device)
        return (erased_reshaped,) + output[1:]

    # GPT-2 blocks are at model.transformer.h[layer_idx]
    handle = model.transformer.h[layer_idx].register_forward_hook(intervention_hook)
    try:
        return wrapper.forward(x)
    finally:
        handle.remove()


def compute_metrics(
    original: torch.Tensor,
    intervened: torch.Tensor,
) -> tuple[float, float]:
    """Compute MSE and MAE between original and intervened outputs."""
    diff = (original - intervened).float()
    mse = float(diff.pow(2).mean().item())
    mae = float(diff.abs().mean().item())
    return mse, mae


def run_experiment_for_property(
    wrapper: GPT4TSWrapper,
    property_name: str,
    device: torch.device,
    layers_to_test: list[int],
    num_samples: int = 500,
    batch_size: int = 64,
) -> list[InterventionResult]:
    """Run downstream impact experiment for a single property.

    Args:
        wrapper: Loaded GPT4TSWrapper.
        property_name: "trend" or "stationarity".
        device: Device.
        layers_to_test: List of transformer block indices to test.
        num_samples: Number of synthetic samples.
        batch_size: Batch size.

    Returns:
        List of InterventionResult for each tested layer.
    """
    if property_name == "trend":
        dataset = generate_trend_dataset(
            num_samples=num_samples,
            seq_len=512,
            seed=42,
        )
        label_type = "classification"
    elif property_name == "stationarity":
        dataset = generate_stationarity_dataset(
            num_samples=num_samples,
            seq_len=512,
            seed=42,
        )
        label_type = "classification"
    else:
        raise ValueError(f"Unsupported property: {property_name}")

    x_all = torch.from_numpy(dataset.sequences).float().to(device)  # (N, 512)
    labels = torch.from_numpy(dataset.labels)

    results: list[InterventionResult] = []
    model = wrapper.model

    for layer_idx in layers_to_test:
        layer_name = f"transformer.h.{layer_idx}"
        print(f"  Layer {layer_idx}: extracting representations...")

        # Step 1: Capture representations
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

        handle = model.transformer.h[layer_idx].register_forward_hook(capture_hook)

        for i in range(0, num_samples, batch_size):
            batch = x_all[i : i + batch_size]
            with torch.no_grad():
                _ = wrapper.forward(batch)

        handle.remove()
        all_reprs = torch.cat(layer_reprs, dim=0)  # (N, T, D)
        print(f"    Representations shape: {all_reprs.shape}")

        # Step 2: Fit LEACE eraser
        print("    Fitting LEACE eraser...")
        eraser = fit_leace_eraser(all_reprs, labels, label_type)

        # Step 3: Get original outputs
        print("    Computing original output...")
        orig_outputs: list[torch.Tensor] = []
        for i in range(0, num_samples, batch_size):
            batch = x_all[i : i + batch_size]
            out = get_output(wrapper, batch)
            orig_outputs.append(out.cpu())
        orig_output = torch.cat(orig_outputs, dim=0)

        # Step 4: Get intervened outputs
        print("    Computing intervened output...")
        erased_outputs: list[torch.Tensor] = []
        for i in range(0, num_samples, batch_size):
            batch = x_all[i : i + batch_size]
            out = run_intervention(wrapper, batch, layer_idx, eraser, device)
            erased_outputs.append(out.cpu())
        erased_output = torch.cat(erased_outputs, dim=0)

        # Step 5: Compute divergence between original and intervened outputs
        mse_diff, mae_diff = compute_metrics(orig_output, erased_output)
        # Normalize by original output norm for meaningful percentage
        orig_norm = float(orig_output.float().pow(2).mean().item())
        increase_pct = (mse_diff / max(orig_norm, 1e-8)) * 100

        result = InterventionResult(
            model_name="GPT4TS",
            property_name=property_name,
            layer_name=layer_name,
            mse_original=orig_norm,
            mse_erased=mse_diff,
            mse_increase_pct=increase_pct,
            mae_original=0.0,
            mae_erased=mae_diff,
        )
        results.append(result)

        print(
            f"    Output divergence MSE: {mse_diff:.6f} "
            f"({increase_pct:.1f}% of output norm), MAE: {mae_diff:.4f}"
        )

    return results


def plot_downstream_impact(
    all_results: dict[str, list[InterventionResult]],
    output_dir: Path,
) -> None:
    """Create downstream impact figure."""
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
        divergences = [r.mse_increase_pct for r in results]

        color = prop_colors.get(prop_name, "#333333")
        ax.bar(layer_indices, divergences, color=color, alpha=0.8, width=0.6)
        ax.set_xticks(layer_indices)
        ax.set_xticklabels([f"L{layer}" for layer in layers], rotation=45, ha="right")
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Output Divergence (% of norm)")
        ax.set_title(f"Erasing {prop_name.title()}")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.suptitle(
        "Downstream Output Impact of LEACE Erasure (GPT4TS)",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    stem = "fig20_gpt4ts_downstream_impact"
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_dir / stem}.pdf")


def main() -> None:
    """Run downstream impact analysis for GPT4TS."""
    seed_everything(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("outputs/paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = Path("outputs/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GPT4TS model...")
    wrapper = load_gpt4ts_model(device)

    # GPT-2 has 12 layers → test early (0), mid (5), late (11)
    layers_to_test = [0, 5, 11]

    all_results: dict[str, list[InterventionResult]] = {}

    for prop_name in ["trend", "stationarity"]:
        print(f"\n{'=' * 60}")
        print(f"Property: {prop_name}")
        print(f"{'=' * 60}")
        results = run_experiment_for_property(
            wrapper,
            prop_name,
            device,
            layers_to_test,
            num_samples=500,
            batch_size=64,
        )
        all_results[prop_name] = results

    # Save results
    json_results: dict[str, list[dict[str, object]]] = {}
    for prop_name, results in all_results.items():
        json_results[prop_name] = [
            {
                "model": r.model_name,
                "property": r.property_name,
                "layer": r.layer_name,
                "output_norm": r.mse_original,
                "divergence_mse": r.mse_erased,
                "divergence_pct": r.mse_increase_pct,
                "divergence_mae": r.mae_erased,
            }
            for r in results
        ]

    results_path = analysis_dir / "gpt4ts_downstream_impact_results.json"
    results_path.write_text(json.dumps(json_results, indent=2))
    print(f"\nSaved results to {results_path}")

    plot_downstream_impact(all_results, output_dir)


if __name__ == "__main__":
    main()
