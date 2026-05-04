"""Cross-property LEACE: erase property A, evaluate probe for property B.

Tests whether erasing one temporal concept (e.g., trend) affects the probing
accuracy of another concept (e.g., stationarity), revealing shared encoding.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_cross_property_leace.py \
        --model moment_pca512 \
        --erase_property trend \
        --eval_property stationarity \
        --output_dir outputs/cross_leace/moment_pca512/
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cross-property LEACE erasure analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _ = parser.add_argument("--model", type=str, required=True, help="Model name prefix.")
    _ = parser.add_argument(
        "--erase_property",
        type=str,
        required=True,
        help="Property to erase (e.g., trend, stationarity).",
    )
    _ = parser.add_argument(
        "--eval_property",
        type=str,
        required=True,
        help="Property to evaluate after erasure.",
    )
    _ = parser.add_argument(
        "--repr_base",
        type=str,
        default="outputs/representations",
        help="Base directory for representations.",
    )
    _ = parser.add_argument(
        "--probe_base",
        type=str,
        default="outputs/probes",
        help="Base directory for probes.",
    )
    _ = parser.add_argument("--output_dir", type=str, required=True)
    _ = parser.add_argument("--seed", type=int, default=42)
    _ = parser.add_argument("--val_split", type=float, default=0.2)
    _ = parser.add_argument(
        "--layer",
        type=str,
        default="best",
        help="Specific layer name or 'best' to auto-select.",
    )
    return parser.parse_args(argv)


def find_best_layer(probe_dir: Path, label_type: str) -> str:
    """Find the best-performing layer from probe metrics."""
    metric_key = "accuracy" if label_type == "classification" else "r2"
    best_layer = None
    best_value = -float("inf")

    for layer_dir in sorted(probe_dir.iterdir()):
        metrics_path = layer_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        val = metrics.get(f"val_{metric_key}", metrics.get(metric_key, -float("inf")))
        if val > best_value:
            best_value = val
            best_layer = layer_dir.name

    if best_layer is None:
        raise FileNotFoundError(f"No probe metrics found in {probe_dir}")
    return best_layer


def main() -> None:
    """Run cross-property LEACE erasure."""
    args = parse_args()

    from src.probes.linear_probe import LinearProbe
    from src.utils.seed import seed_everything

    seed_everything(args.seed)

    repr_base = Path(args.repr_base)
    probe_base = Path(args.probe_base)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate representation dirs for both properties
    erase_repr_dir = repr_base / args.model / f"synthetic_{args.erase_property}"
    eval_repr_dir = repr_base / args.model / f"synthetic_{args.eval_property}"

    if not erase_repr_dir.exists():
        raise FileNotFoundError(f"Erase repr dir not found: {erase_repr_dir}")
    if not eval_repr_dir.exists():
        raise FileNotFoundError(f"Eval repr dir not found: {eval_repr_dir}")

    # Load eval metadata
    eval_meta = json.loads((eval_repr_dir / "metadata.json").read_text())
    eval_label_type = eval_meta.get("label_type", "classification")

    # Load erase metadata
    erase_meta = json.loads((erase_repr_dir / "metadata.json").read_text())
    erase_label_type = erase_meta.get("label_type", "classification")

    # Find probe dir for eval property
    eval_probe_dir = probe_base / f"{args.model}_synthetic_{args.eval_property}_linear"
    if not eval_probe_dir.exists():
        # Try without synthetic_ prefix
        eval_probe_dir = probe_base / f"{args.model}_{args.eval_property}_linear"
    if not eval_probe_dir.exists():
        raise FileNotFoundError(f"Eval probe dir not found for {args.eval_property}")

    # Determine layer
    if args.layer == "best":
        layer_name = find_best_layer(eval_probe_dir, eval_label_type)
    else:
        layer_name = args.layer

    print(f"Model: {args.model}")
    print(f"Erase: {args.erase_property} ({erase_label_type})")
    print(f"Eval: {args.eval_property} ({eval_label_type})")
    print(f"Layer: {layer_name}")

    # Load eval representations and labels
    eval_repr = torch.load(
        eval_repr_dir / f"{layer_name}.pt", map_location="cpu", weights_only=True
    )
    if eval_repr.ndim >= 3:
        eval_repr = eval_repr.view(eval_repr.shape[0], -1)
    eval_repr = eval_repr.float()

    eval_labels = torch.load(eval_repr_dir / "labels.pt", map_location="cpu", weights_only=True)
    if eval_label_type == "classification":
        eval_labels = eval_labels.long()
    else:
        eval_labels = eval_labels.float()

    # Load erase labels (for fitting LEACE on the erase concept)
    erase_labels = torch.load(erase_repr_dir / "labels.pt", map_location="cpu", weights_only=True)

    # Note: erase and eval representations are from DIFFERENT datasets with different
    # samples. We need representations from the SAME samples but with different labels.
    # Since both synthetic datasets use same model, the representations are from
    # different data. For cross-property LEACE, we need to:
    # 1. Use the eval representations as-is
    # 2. Fit LEACE on eval representations with erase-concept labels re-generated
    #
    # Alternative simpler approach: load SAME representations, use erase labels for
    # LEACE fitting and eval labels for evaluation. But labels come from different
    # generators with different samples.
    #
    # Most practical: use the eval representations, re-generate erase labels for those
    # samples. But we don't have the raw sequences saved.
    #
    # Simplest correct approach: use the erase representations (which are from the
    # erase dataset) to fit LEACE, then apply the eraser to eval representations.
    # This works because the eraser is a linear projection that doesn't depend on
    # specific samples — it depends on the concept direction.

    erase_repr = torch.load(
        erase_repr_dir / f"{layer_name}.pt", map_location="cpu", weights_only=True
    )
    if erase_repr.ndim >= 3:
        erase_repr = erase_repr.view(erase_repr.shape[0], -1)
    erase_repr = erase_repr.float()

    # Fit LEACE on erase representations with erase labels
    concept_erasure = importlib.import_module("concept_erasure")
    if erase_label_type == "classification":
        z_erase = erase_labels.long()
    else:
        z_erase = erase_labels.float().unsqueeze(-1)

    print(f"Fitting LEACE on {args.erase_property} ({erase_repr.shape})...")
    fitter = concept_erasure.LeaceFitter.fit(erase_repr, z_erase)
    eraser = fitter.eraser

    # Apply eraser to eval representations
    erased_eval_repr = eraser(eval_repr)

    # Split eval data
    rng = np.random.default_rng(args.seed)
    n = len(eval_labels)
    indices = rng.permutation(n)
    split = int(n * (1 - args.val_split))
    test_idx = indices[split:]

    eval_repr_test = eval_repr[test_idx]
    erased_eval_repr_test = erased_eval_repr[test_idx]
    eval_labels_test = eval_labels[test_idx]

    # Load eval probe
    input_dim = eval_repr.shape[1]
    output_dim = int(eval_labels.max().item()) + 1 if eval_label_type == "classification" else 1
    probe = LinearProbe(input_dim=input_dim, output_dim=output_dim)
    probe_path = eval_probe_dir / layer_name / "probe.pt"
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_path}")
    _ = probe.load_state_dict(torch.load(probe_path, map_location="cpu", weights_only=True))
    _ = probe.eval()

    # Evaluate before and after
    with torch.no_grad():
        before_out = probe(eval_repr_test).cpu()
        after_out = probe(erased_eval_repr_test).cpu()

    result: dict[str, float | str] = {}
    if eval_label_type == "classification":
        before_acc = float((before_out.argmax(1) == eval_labels_test).float().mean())
        after_acc = float((after_out.argmax(1) == eval_labels_test).float().mean())
        result["before_accuracy"] = before_acc
        result["after_accuracy"] = after_acc
        result["drop"] = before_acc - after_acc
        print(
            f"Before: {before_acc:.4f}, After: {after_acc:.4f}, Drop: {before_acc - after_acc:.4f}"
        )
    else:
        true = eval_labels_test.float()
        before_pred = before_out.squeeze(-1)
        after_pred = after_out.squeeze(-1)
        ss_tot = float(((true - true.mean()) ** 2).sum())
        before_r2 = 1.0 - float(((true - before_pred) ** 2).sum()) / max(ss_tot, 1e-12)
        after_r2 = 1.0 - float(((true - after_pred) ** 2).sum()) / max(ss_tot, 1e-12)
        result["before_r2"] = before_r2
        result["after_r2"] = after_r2
        result["drop"] = before_r2 - after_r2
        print(
            f"Before R²: {before_r2:.4f}, After R²: {after_r2:.4f}, "
            f"Drop: {before_r2 - after_r2:.4f}"
        )

    result["erase_property"] = args.erase_property
    result["eval_property"] = args.eval_property
    result["model"] = args.model
    result["layer"] = layer_name

    out_file = output_dir / f"cross_leace_{args.erase_property}_on_{args.eval_property}.json"
    _ = out_file.write_text(json.dumps(result, indent=2))
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
