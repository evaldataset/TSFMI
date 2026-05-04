"""Measure downstream forecasting impact of LEACE concept erasure on Moirai v2.

Moirai v2 is an encoder-based foundation model. We generate synthetic time series
with known properties (trend, stationarity), run the encoder forward, then erase
the property from intermediate encoder layer representations via LEACE and
measure how the output changes.

Since Moirai v2 (small) has only 6 encoder layers, we test all of them.

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/run_moirai_downstream_impact.py
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
from src.utils.seed import seed_everything

PATCH_SIZE = 16
SEQ_LEN = 512


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


def load_moirai_model(device: torch.device) -> nn.Module:
    """Load frozen Moirai v2 module.

    Args:
        device: Target device.

    Returns:
        Moirai2Module (nn.Module).
    """
    from uni2ts.model.moirai2 import Moirai2Module

    module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")
    if not isinstance(module, nn.Module):
        raise RuntimeError("Failed to load Moirai2Module as nn.Module")

    module = module.to(device)
    module.eval()
    for param in module.parameters():
        param.requires_grad = False

    return module


def prepare_moirai_inputs(
    sequences: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Prepare Moirai-compatible input tensors from raw sequences.

    Args:
        sequences: Tensor of shape (N, seq_len).
        device: Target device.

    Returns:
        Dict of keyword arguments for Moirai forward pass.
    """
    batch_size, seq_len = sequences.shape
    if seq_len % PATCH_SIZE != 0:
        raise ValueError(f"seq_len must be divisible by {PATCH_SIZE}, got {seq_len}")

    num_patches = seq_len // PATCH_SIZE
    target = sequences.to(device, dtype=torch.float32).reshape(batch_size, num_patches, PATCH_SIZE)
    observed_mask = torch.ones_like(target, dtype=torch.bool)
    sample_id = torch.zeros(batch_size, num_patches, dtype=torch.long, device=device)
    time_id = torch.arange(num_patches, device=device).unsqueeze(0).expand(batch_size, -1)
    variate_id = torch.zeros(batch_size, num_patches, dtype=torch.long, device=device)
    prediction_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

    return {
        "target": target,
        "observed_mask": observed_mask,
        "sample_id": sample_id,
        "time_id": time_id,
        "variate_id": variate_id,
        "prediction_mask": prediction_mask,
        "training_mode": False,
    }


def get_moirai_output(
    module: nn.Module,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Get Moirai output tensor.

    Args:
        module: Moirai2Module.
        inputs: Prepared input dict.

    Returns:
        Output tensor (shape varies by model output).
    """
    with torch.no_grad():
        output = module(**inputs)

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    for attr in ("last_hidden_state", "prediction", "predictions", "logits"):
        val = getattr(output, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    raise RuntimeError("Cannot extract tensor output from Moirai")


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
    module: nn.Module,
    inputs: dict[str, torch.Tensor],
    layer_idx: int,
    eraser: object,
    device: torch.device,
) -> torch.Tensor:
    """Run Moirai forward with LEACE-erased representations at a specific layer.

    Args:
        module: Moirai2Module.
        inputs: Prepared input dict.
        layer_idx: Encoder layer index to intervene on.
        eraser: Fitted LEACE eraser.
        device: Device.

    Returns:
        Output tensor after intervention.
    """

    def intervention_hook(
        mod: nn.Module,
        input_args: tuple[torch.Tensor, ...],
        output: tuple[torch.Tensor, ...] | torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | torch.Tensor:
        """Replace hidden states with LEACE-erased version."""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_states.shape[-1]).float().cpu()
        erased = eraser(flat)
        if not isinstance(erased, torch.Tensor):
            raise TypeError("LEACE eraser did not return a torch.Tensor")

        erased_reshaped = erased.reshape(original_shape).to(device)

        if isinstance(output, tuple):
            return (erased_reshaped,) + output[1:]
        return erased_reshaped

    handle = module.encoder.layers[layer_idx].register_forward_hook(intervention_hook)
    try:
        return get_moirai_output(module, inputs)
    finally:
        handle.remove()


def compute_metrics(
    original: torch.Tensor,
    intervened: torch.Tensor,
) -> tuple[float, float]:
    """Compute MSE and MAE between original and intervened outputs.

    Args:
        original: Original output tensor.
        intervened: Intervened output tensor.

    Returns:
        Tuple of (MSE, MAE).
    """
    diff = (original - intervened).float()
    mse = float(diff.pow(2).mean().item())
    mae = float(diff.abs().mean().item())
    return mse, mae


def run_experiment_for_property(
    module: nn.Module,
    property_name: str,
    device: torch.device,
    layers_to_test: list[int],
    num_samples: int = 500,
    batch_size: int = 64,
) -> list[InterventionResult]:
    """Run downstream impact experiment for a single property.

    Args:
        module: Moirai2Module.
        property_name: "trend" or "stationarity".
        device: Device.
        layers_to_test: List of encoder layer indices to test.
        num_samples: Number of synthetic samples.
        batch_size: Batch size.

    Returns:
        List of InterventionResult for each tested layer.
    """
    if property_name == "trend":
        dataset = generate_trend_dataset(
            num_samples=num_samples,
            seq_len=SEQ_LEN,
            seed=42,
        )
        label_type = "classification"
    elif property_name == "stationarity":
        dataset = generate_stationarity_dataset(
            num_samples=num_samples,
            seq_len=SEQ_LEN,
            seed=42,
        )
        label_type = "classification"
    else:
        raise ValueError(f"Unsupported property: {property_name}")

    x_all = torch.from_numpy(dataset.sequences).float()  # (N, 512)
    labels = torch.from_numpy(dataset.labels)

    results: list[InterventionResult] = []

    for layer_idx in layers_to_test:
        layer_name = f"encoder.layers.{layer_idx}"
        print(f"  Layer {layer_idx}: extracting representations...")

        # Step 1: Capture representations
        layer_reprs: list[torch.Tensor] = []

        append_repr = layer_reprs.append

        def capture_hook(
            mod: nn.Module,
            input_args: tuple[torch.Tensor, ...],
            output: tuple[torch.Tensor, ...] | torch.Tensor,
            *,
            append_repr_fn=append_repr,
        ) -> None:
            if isinstance(output, tuple):
                append_repr_fn(output[0].cpu())
            else:
                append_repr_fn(output.cpu())

        handle = module.encoder.layers[layer_idx].register_forward_hook(capture_hook)

        for i in range(0, num_samples, batch_size):
            batch = x_all[i : i + batch_size]
            inputs = prepare_moirai_inputs(batch, device)
            with torch.no_grad():
                _ = module(**inputs)

        handle.remove()
        all_reprs = torch.cat(layer_reprs, dim=0)
        print(f"    Representations shape: {all_reprs.shape}")

        # Step 2: Fit LEACE eraser
        print("    Fitting LEACE eraser...")
        eraser = fit_leace_eraser(all_reprs, labels, label_type)

        # Step 3: Get original outputs (in batches)
        print("    Computing original output...")
        orig_outputs: list[torch.Tensor] = []
        for i in range(0, num_samples, batch_size):
            batch = x_all[i : i + batch_size]
            inputs = prepare_moirai_inputs(batch, device)
            out = get_moirai_output(module, inputs)
            orig_outputs.append(out.cpu())
        orig_output = torch.cat(orig_outputs, dim=0)

        # Step 4: Get intervened outputs (in batches)
        print("    Computing intervened output...")
        erased_outputs: list[torch.Tensor] = []
        for i in range(0, num_samples, batch_size):
            batch = x_all[i : i + batch_size]
            inputs = prepare_moirai_inputs(batch, device)
            out = run_intervention(module, inputs, layer_idx, eraser, device)
            erased_outputs.append(out.cpu())
        erased_output = torch.cat(erased_outputs, dim=0)

        # Step 5: Compute divergence
        mse_diff, mae_diff = compute_metrics(orig_output, erased_output)
        orig_norm = float(orig_output.float().pow(2).mean().item())
        increase_pct = (mse_diff / max(orig_norm, 1e-8)) * 100

        result = InterventionResult(
            model_name="Moirai",
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
        divergences = [r.mse_increase_pct for r in results]

        color = prop_colors.get(prop_name, "#333333")
        ax.bar(layer_indices, divergences, color=color, alpha=0.8, width=0.6)
        ax.set_xticks(layer_indices)
        ax.set_xticklabels([f"L{layer}" for layer in layers], rotation=45, ha="right")
        ax.set_xlabel("Encoder Layer")
        ax.set_ylabel("Output Divergence (% of norm)")
        ax.set_title(f"Erasing {prop_name.title()}")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.suptitle(
        "Downstream Output Impact of LEACE Erasure (Moirai v2)",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    stem = "fig22_moirai_downstream_impact"
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_dir / stem}.pdf")


def main() -> None:
    """Run downstream impact analysis for Moirai v2."""
    seed_everything(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("outputs/paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = Path("outputs/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Moirai v2 model...")
    module = load_moirai_model(device)

    # Moirai v2 small has 6 encoder layers — test all
    layers_to_test = [0, 1, 2, 3, 4, 5]

    all_results: dict[str, list[InterventionResult]] = {}

    for prop_name in ["trend", "stationarity"]:
        print(f"\n{'=' * 60}")
        print(f"Property: {prop_name}")
        print(f"{'=' * 60}")
        results = run_experiment_for_property(
            module,
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

    results_path = analysis_dir / "moirai_downstream_impact_results.json"
    results_path.write_text(json.dumps(json_results, indent=2))
    print(f"\nSaved results to {results_path}")

    plot_downstream_impact(all_results, output_dir)


if __name__ == "__main__":
    main()
