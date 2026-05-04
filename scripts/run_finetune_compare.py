"""Fine-tuning vs Frozen probing comparison.

Fine-tunes a model (PatchTST-Pre) on a simple forecasting task, then extracts
representations from the fine-tuned model and compares probing accuracy to
the frozen (pre-trained) baseline.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_finetune_compare.py \
        --model patchtst_pretrained \
        --dataset etth1 \
        --output_dir outputs/finetune_compare/patchtst_pretrained_etth1/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tuning vs frozen probing comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--model", type=str, default="patchtst_pretrained")
    _ = parser.add_argument("--dataset", type=str, default="etth1")
    _ = parser.add_argument("--output_dir", type=str, required=True)
    _ = parser.add_argument("--ft_epochs", type=int, default=10, help="Fine-tuning epochs.")
    _ = parser.add_argument("--ft_lr", type=float, default=1e-4, help="Fine-tuning learning rate.")
    _ = parser.add_argument("--num_samples", type=int, default=1000)
    _ = parser.add_argument("--seq_len", type=int, default=512)
    _ = parser.add_argument("--device", type=str, default=None)
    _ = parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def extract_representations_from_model(
    wrapper,
    sequences: torch.Tensor,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, torch.Tensor]:
    """Extract representations from all layers of a model.

    Args:
        wrapper: Model wrapper with hook_manager.
        sequences: Input tensor (N, seq_len).
        device: Torch device.
        batch_size: Batch size for inference.

    Returns:
        Dict mapping layer name to representation tensor.
    """
    from src.extractors.hook_manager import HookManager

    layer_names = wrapper.get_layer_names()
    all_representations: dict[str, list[torch.Tensor]] = {name: [] for name in layer_names}

    with HookManager(wrapper.model, layer_names) as hm:
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size].unsqueeze(-1).to(device)
            with torch.no_grad():
                _ = wrapper.forward(batch)
            activations = hm.get_activations()
            for name, act in activations.items():
                all_representations[name].append(act.detach().cpu())
            hm.clear()

    result = {}
    for name in layer_names:
        if all_representations[name]:
            repr_tensor = torch.cat(all_representations[name], dim=0)
            if repr_tensor.ndim >= 3:
                repr_tensor = repr_tensor.view(repr_tensor.shape[0], -1)
            result[name] = repr_tensor

    return result


def probe_representations(
    representations: dict[str, torch.Tensor],
    labels: NDArray[np.int64] | NDArray[np.float64],
    seed: int = 42,
) -> dict[str, float]:
    """Probe each layer with RidgeClassifier and return accuracy.

    Args:
        representations: Dict mapping layer name to tensor.
        labels: Label array.
        seed: Random seed.

    Returns:
        Dict mapping layer name to accuracy.
    """
    results = {}
    for layer_name, repr_tensor in representations.items():
        X = repr_tensor.numpy().astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            labels,
            test_size=0.2,
            random_state=seed,
        )
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        results[layer_name] = float(acc)
    return results


def main() -> None:
    """Run fine-tuning comparison."""
    args = parse_args()

    from src.utils.device import resolve_device
    from src.utils.seed import seed_everything

    seed_everything(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    from scripts.extract_representations import load_dataset_by_name

    # Use trend as the probing property
    dataset_name = f"{args.dataset}_trend"
    dataset = load_dataset_by_name(
        dataset_name, args.num_samples, args.seq_len, args.seed, stride=64
    )
    sequences = torch.tensor(dataset.sequences, dtype=torch.float32)
    labels = dataset.labels

    # === Step 1: Frozen representations ===
    print("=== Frozen Model ===")
    from scripts.extract_representations import load_model_wrapper

    wrapper = load_model_wrapper(args.model, None, device, args.seq_len)
    frozen_reprs = extract_representations_from_model(wrapper, sequences, device)
    frozen_results = probe_representations(frozen_reprs, labels, args.seed)

    for layer, acc in frozen_results.items():
        print(f"  [{layer}] frozen_acc={acc:.4f}")

    # === Step 2: Fine-tune on simple next-step prediction ===
    print("\n=== Fine-tuning ===")
    from src.extractors.hook_manager import HookManager

    model = wrapper.model
    model.train()

    # Unfreeze
    for param in model.parameters():
        param.requires_grad = True

    # Use last layer representation for fine-tuning
    last_layer_name = wrapper.get_layer_names()[-1]
    sample_repr = frozen_reprs[last_layer_name]
    repr_dim = sample_repr.shape[1]
    forecast_head = nn.Linear(repr_dim, args.seq_len // 4).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(forecast_head.parameters()),
        lr=args.ft_lr,
    )

    # Simple self-supervised task: predict last quarter from representation
    train_data = sequences[: int(len(sequences) * 0.8)]
    train_loader = DataLoader(
        TensorDataset(train_data),
        batch_size=32,
        shuffle=True,
    )

    for epoch in range(args.ft_epochs):
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch_in = batch.unsqueeze(-1).to(device)  # (B, seq, 1)
            target = batch[:, -args.seq_len // 4 :].to(device)  # Last quarter

            # Extract last layer repr with hook during fine-tuning
            with HookManager(model, [last_layer_name]) as hm:
                _ = wrapper.forward(batch_in)
                activations = hm.get_activations()

            repr_out = activations[last_layer_name]
            if repr_out.ndim >= 3:
                repr_flat = repr_out.view(repr_out.shape[0], -1)
            else:
                repr_flat = repr_out

            # Trim if needed
            if repr_flat.shape[1] > repr_dim:
                repr_flat = repr_flat[:, :repr_dim]

            pred = forecast_head(repr_flat)
            loss = nn.functional.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"  Epoch {epoch + 1}/{args.ft_epochs}, loss={epoch_loss / len(train_loader):.6f}")

    # === Step 3: Extract fine-tuned representations ===
    print("\n=== Fine-tuned Model ===")
    model.eval()

    # Re-freeze for extraction
    for param in model.parameters():
        param.requires_grad = False

    ft_reprs = extract_representations_from_model(wrapper, sequences, device)
    ft_results = probe_representations(ft_reprs, labels, args.seed)

    for layer, acc in ft_results.items():
        print(f"  [{layer}] finetuned_acc={acc:.4f}")

    # === Save comparison ===
    comparison = {
        "model": args.model,
        "dataset": args.dataset,
        "ft_epochs": args.ft_epochs,
        "ft_lr": args.ft_lr,
        "layers": {},
    }
    for layer in frozen_results:
        frozen_acc = frozen_results[layer]
        ft_acc = ft_results.get(layer, 0.0)
        comparison["layers"][layer] = {
            "frozen_accuracy": frozen_acc,
            "finetuned_accuracy": ft_acc,
            "change": ft_acc - frozen_acc,
        }
        print(
            f"  [{layer}] frozen={frozen_acc:.4f} -> finetuned={ft_acc:.4f} "
            f"(Δ={ft_acc - frozen_acc:+.4f})"
        )

    results_path = output_dir / "finetune_comparison.json"
    _ = results_path.write_text(json.dumps(comparison, indent=2))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
