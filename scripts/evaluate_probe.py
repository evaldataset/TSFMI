"""CLI script for evaluating trained linear probes and generating layer-wise analysis.

Loads per-layer probe metrics from a probe directory, optionally computes selectivity
against control probes, generates layer-wise metric plots, and saves consolidated CSV results.

Usage:
    python scripts/evaluate_probe.py \
        --probe_dir outputs/probes/moment/trend/ \
        --representations_dir outputs/representations/moment/synthetic_trend/ \
        --output_dir outputs/eval/moment/trend/ \
        --metrics accuracy f1 r2 \
        --plot
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for probe evaluation.

    Args:
        argv: Argument list. Defaults to sys.argv[1:] if None.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained probes and generate layer-wise analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        required=True,
        help="Directory with layer subdirs each containing probe.pt + metrics.json.",
    )
    parser.add_argument(
        "--representations_dir",
        type=str,
        required=True,
        help="Directory containing {layer}.pt and labels.pt files (output of"
        " extract_representations.py).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval/",
        help="Directory for saving CSV results and plots.",
    )
    parser.add_argument(
        "--control_probe_dir",
        type=str,
        default=None,
        help="If provided, compute selectivity from control probe metrics.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["accuracy", "f1", "r2", "selectivity", "cka"],
        default=["accuracy", "f1", "r2"],
        help="Metrics to include in the evaluation report.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate matplotlib layer-wise metric plots.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string ("cuda", "cpu", "cuda:0"). Auto-selects if None.',
    )
    return parser.parse_args(argv)


def _discover_layer_dirs(probe_dir: Path) -> list[Path]:
    """Find layer subdirectories that contain probe.pt or metrics.json.

    Args:
        probe_dir: Root probe directory.

    Returns:
        Sorted list of layer subdirectory paths.

    Raises:
        FileNotFoundError: If no valid layer directories are found.
    """
    layer_dirs = sorted(d for d in probe_dir.iterdir() if d.is_dir() and (d / "probe.pt").exists())
    if not layer_dirs:
        raise FileNotFoundError(f"No layer directories with probe.pt found in {probe_dir}")
    return layer_dirs


def _plot_layer_curves(
    results: list[dict[str, object]],
    output_dir: Path,
    label_type: str,
) -> None:
    """Generate layer-wise metric curve plots using matplotlib.

    Saves plots as PNG files to output_dir.

    Args:
        results: List of per-layer metric dicts (must include "layer" key).
        output_dir: Directory to save PNG plots.
        label_type: "classification" or "regression" — determines which metrics to plot.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [str(r["layer"]) for r in results]

    if label_type == "classification":
        metric_keys = ["val_accuracy", "val_f1_macro"]
        ylabel = "Score"
        title = "Layer-wise Probe Performance (Classification)"
    else:
        metric_keys = ["val_r2", "val_mae"]
        ylabel = "Metric Value"
        title = "Layer-wise Probe Performance (Regression)"

    # Add selectivity if present
    if any("selectivity" in r for r in results):
        metric_keys.append("selectivity")

    fig, ax = plt.subplots(figsize=(10, 5))
    for key in metric_keys:
        values = [r.get(key, None) for r in results]
        valid = [(i, v) for i, v in enumerate(values) if v is not None]
        if valid:
            xs, ys = zip(*valid, strict=False)
            ax.plot(xs, ys, marker="o", label=key)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(layer) for layer in layers], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    plot_path = output_dir / "layer_metrics.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {plot_path}")


def main() -> None:
    """Evaluate trained probes across layers and generate analysis reports."""
    args = parse_args()

    # Lazy imports — keep --help fast without torch
    import csv
    import json
    from pathlib import Path

    import torch

    from src.metrics.probing_metrics import compute_cka, compute_selectivity
    from src.utils.device import resolve_device

    device = resolve_device(args.device)
    probe_dir = Path(args.probe_dir)
    repr_dir = Path(args.representations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not probe_dir.exists():
        raise FileNotFoundError(f"Probe directory not found: {probe_dir}")
    if not repr_dir.exists():
        raise FileNotFoundError(f"Representations directory not found: {repr_dir}")

    # Load metadata to determine label_type
    meta_path = repr_dir / "metadata.json"
    label_type = "classification"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        label_type = meta.get("label_type", "classification")

    # Load labels
    labels_path = repr_dir / "labels.pt"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Expected tensor labels in {labels_path}, got {type(labels)!r}")

    # Discover layer directories
    layer_dirs = _discover_layer_dirs(probe_dir)

    print(f"Device: {device}")
    print(f"Label type: {label_type}")
    print(f"Layers found: {len(layer_dirs)}")
    print(f"Requested metrics: {args.metrics}")
    print()

    results: list[dict[str, object]] = []
    all_fieldnames: list[str] = ["layer"]

    for layer_dir in layer_dirs:
        layer_name = layer_dir.name

        # Load saved training metrics
        metrics_path = layer_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"Warning: metrics.json not found for layer {layer_name}, skipping")
            continue
        saved_metrics: dict[str, float] = json.loads(metrics_path.read_text())

        row: dict[str, object] = {"layer": layer_name, **saved_metrics}

        # Compute selectivity if control probes are available
        if args.control_probe_dir and "selectivity" in args.metrics:
            control_dir = Path(args.control_probe_dir) / layer_name
            control_metrics_path = control_dir / "metrics.json"
            if control_metrics_path.exists():
                control_metrics: dict[str, float] = json.loads(control_metrics_path.read_text())
                if label_type == "classification":
                    linear_acc = saved_metrics.get("val_accuracy", 0.0)
                    control_acc = control_metrics.get("val_accuracy", 0.0)
                else:
                    linear_acc = saved_metrics.get("val_r2", 0.0)
                    control_acc = control_metrics.get("val_r2", 0.0)
                row["selectivity"] = compute_selectivity(linear_acc, control_acc)
            else:
                print(
                    f"Warning: control metrics not found for layer {layer_name}, "
                    "skipping selectivity"
                )

        # Compute CKA between adjacent layers if requested
        if "cka" in args.metrics:
            repr_file = repr_dir / f"{layer_name}.pt"
            if repr_file.exists():
                row["_repr_file"] = str(repr_file)

        # Track all field names for CSV header
        for key in row:
            if key not in all_fieldnames and not key.startswith("_"):
                all_fieldnames.append(key)

        results.append(row)
        print(f"[{layer_name}] Loaded metrics: {saved_metrics}")

    # Compute CKA between consecutive layers
    if "cka" in args.metrics and len(results) >= 2:
        print("\nComputing inter-layer CKA...")
        prev_repr: torch.Tensor | None = None
        prev_name: str = ""
        for row in results:
            repr_file_obj = row.pop("_repr_file", None)
            if repr_file_obj is None:
                prev_repr = None
                continue

            repr_file = str(repr_file_obj)

            curr_repr = torch.load(repr_file, map_location=device, weights_only=True)
            if curr_repr.ndim >= 3:
                n = curr_repr.shape[0]
                curr_repr = curr_repr.view(n, -1)

            if prev_repr is not None and prev_repr.shape[0] == curr_repr.shape[0]:
                cka_val = compute_cka(prev_repr, curr_repr)
                row[f"cka_vs_{prev_name}"] = cka_val
                if f"cka_vs_{prev_name}" not in all_fieldnames:
                    all_fieldnames.append(f"cka_vs_{prev_name}")

            prev_repr = curr_repr
            prev_name = str(row["layer"])
    else:
        # Clean up internal keys
        for row in results:
            row.pop("_repr_file", None)

    print()

    # Save consolidated CSV
    if results:
        csv_path = output_dir / "layer_metrics.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=all_fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved layer metrics to {csv_path}")

        # Save as JSON too for programmatic access
        json_path = output_dir / "layer_metrics.json"
        # Filter out internal keys for JSON output
        clean_results = [
            {k: v for k, v in r.items() if not str(k).startswith("_")} for r in results
        ]
        json_path.write_text(json.dumps(clean_results, indent=2))
        print(f"Saved layer metrics to {json_path}")
    else:
        print("No results to save.")

    # Generate plots if requested
    if args.plot and results:
        _plot_layer_curves(results, output_dir, label_type)


if __name__ == "__main__":
    main()
