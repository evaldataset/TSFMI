"""Extract and analyze attention patterns from time series models.

Extracts attention weight matrices from each layer/head, computes attention
entropy, mean attention distance, and visualizes attention patterns for
different temporal property classes.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/extract_attention.py \
        --model moment \
        --dataset synthetic_trend \
        --output_dir outputs/attention/moment_trend/ \
        --num_samples 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.hooks import RemovableHandle


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and analyze attention patterns.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--model", type=str, required=True)
    _ = parser.add_argument("--dataset", type=str, required=True)
    _ = parser.add_argument("--output_dir", type=str, required=True)
    _ = parser.add_argument("--num_samples", type=int, default=500)
    _ = parser.add_argument("--seq_len", type=int, default=512)
    _ = parser.add_argument("--seed", type=int, default=42)
    _ = parser.add_argument("--device", type=str, default=None)
    return parser.parse_args(argv)


def register_attention_hooks(
    model: torch.nn.Module,
    model_name: str,
) -> tuple[list[str], dict[str, list[torch.Tensor]]]:
    """Register hooks to capture attention weights from all layers.

    Args:
        model: The model to hook.
        model_name: Model identifier for selecting correct attention module paths.

    Returns:
        Tuple of (layer names, dict mapping layer name to list of attention tensors).
    """
    attention_store: dict[str, list[torch.Tensor]] = {}
    hook_handles: list[RemovableHandle] = []
    layer_names: list[str] = []

    for name, module in model.named_modules():
        # Match attention modules in various architectures
        is_attention = False
        if "self_attn" in name and hasattr(module, "forward"):
            is_attention = True
        elif "attention" in name.lower() and hasattr(module, "forward"):
            # Check it's a leaf attention, not a parent container
            children = list(module.children())
            if len(children) <= 3:  # Attention modules typically have few sub-modules
                is_attention = True

        if not is_attention:
            continue

        layer_names.append(name)
        attention_store[name] = []

        def _make_hook(layer_name: str):
            def hook_fn(mod, inp, out):
                # Most attention modules return (attn_output, attn_weights) or just attn_output
                if isinstance(out, tuple) and len(out) >= 2:
                    attn_weights = out[1]
                    if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                        attention_store[layer_name].append(attn_weights.detach().cpu())

            return hook_fn

        handle = module.register_forward_hook(_make_hook(name))
        hook_handles.append(handle)

    return layer_names, attention_store


def compute_attention_entropy(attn_weights: torch.Tensor) -> float:
    """Compute mean entropy of attention distributions.

    Args:
        attn_weights: Attention tensor of shape (batch, heads, seq, seq).

    Returns:
        Mean entropy across all heads and queries.
    """
    # Clamp to avoid log(0)
    attn = attn_weights.clamp(min=1e-12)
    entropy = -(attn * attn.log()).sum(dim=-1)  # (batch, heads, seq)
    return float(entropy.mean().item())


def compute_attention_distance(attn_weights: torch.Tensor) -> float:
    """Compute mean attention distance (how far each position looks).

    Args:
        attn_weights: Attention tensor of shape (batch, heads, seq, seq).

    Returns:
        Mean absolute distance of attention.
    """
    seq_len = attn_weights.shape[-1]
    positions = torch.arange(seq_len, dtype=torch.float32)
    # Distance matrix: |i - j|
    dist_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    # Weighted mean distance
    mean_dist = (attn_weights * dist_matrix.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
    return float(mean_dist.mean().item())


def main() -> None:
    """Extract attention patterns and generate analysis."""
    args = parse_args()

    from src.utils.device import resolve_device
    from src.utils.seed import seed_everything

    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import extraction helpers
    from scripts.extract_representations import load_dataset_by_name, load_model_wrapper

    print(f"Loading model: {args.model}")
    wrapper = load_model_wrapper(args.model, None, device, args.seq_len)
    model = wrapper.model

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset_by_name(args.dataset, args.num_samples, args.seq_len, args.seed)

    # Register attention hooks
    # Need to enable attention output in model config
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "output_attentions"):
        object.__setattr__(config, "output_attentions", True)

    layer_names, attention_store = register_attention_hooks(model, args.model)

    if not layer_names:
        print("WARNING: No attention modules found. Trying output_attentions approach...")
        # For HuggingFace models, try calling with output_attentions=True
        print("Available modules:")
        for name, _ in model.named_modules():
            print(f"  {name}")
        print("\nNo attention hooks registered. Exiting.")
        return

    print(f"Registered {len(layer_names)} attention hooks: {layer_names}")

    # Run forward pass in batches
    sequences = torch.tensor(dataset.sequences, dtype=torch.float32)
    labels = dataset.labels
    batch_size = 32

    print(f"Running inference on {len(sequences)} samples...")
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size].unsqueeze(-1).to(device)  # (B, seq, 1)
            try:
                _ = wrapper.forward(batch)
            except Exception as e:
                print(f"Forward pass error at batch {i}: {e}")
                break

    # Analyze attention patterns
    results: dict[str, dict[str, float | list[int]]] = {}

    for layer_name in layer_names:
        stored = attention_store[layer_name]
        if not stored:
            print(f"[{layer_name}] No attention weights captured")
            continue

        # Concatenate all batches
        all_attn = torch.cat(stored, dim=0)  # (N, heads, seq, seq)
        entropy = compute_attention_entropy(all_attn)
        distance = compute_attention_distance(all_attn)

        results[layer_name] = {
            "entropy": entropy,
            "mean_distance": distance,
            "shape": list(all_attn.shape),
        }
        print(f"[{layer_name}] entropy={entropy:.4f}, mean_dist={distance:.2f}")

        # Per-class analysis
        if dataset.label_type == "classification":
            unique_labels = np.unique(labels)
            for cls in unique_labels:
                mask = labels == cls
                cls_attn = all_attn[mask]
                cls_entropy = compute_attention_entropy(cls_attn)
                cls_distance = compute_attention_distance(cls_attn)
                results[layer_name][f"class_{cls}_entropy"] = cls_entropy
                results[layer_name][f"class_{cls}_distance"] = cls_distance

    # Save results
    results_path = output_dir / "attention_analysis.json"
    _ = results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved attention analysis to {results_path}")

    # Plot attention entropy across layers
    if results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        layers = list(results.keys())
        entropies = [results[layer]["entropy"] for layer in layers]
        distances = [results[layer]["mean_distance"] for layer in layers]

        ax1.bar(range(len(layers)), entropies, color="steelblue")
        ax1.set_xticks(range(len(layers)))
        ax1.set_xticklabels(
            [layer.split(".")[-1] for layer in layers],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax1.set_ylabel("Mean Attention Entropy")
        ax1.set_title("Attention Entropy by Layer")

        ax2.bar(range(len(layers)), distances, color="coral")
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels(
            [layer.split(".")[-1] for layer in layers],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax2.set_ylabel("Mean Attention Distance")
        ax2.set_title("Attention Distance by Layer")

        plt.tight_layout()
        fig.savefig(output_dir / "attention_patterns.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved attention plot to {output_dir}/attention_patterns.png")


if __name__ == "__main__":
    main()
