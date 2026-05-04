"""Analyze linear-probe sample efficiency on synthetic properties.

For each model and property, this script selects the best layer from existing
evaluation outputs, trains linear probes on progressively larger subsets, and
plots mean ± std accuracy across random seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.probes.linear_probe import LinearProbe
from src.probes.probe_trainer import ProbeTrainer, ProbeTrainerConfig
from src.utils.device import resolve_device
from src.utils.seed import seed_everything

OUT_DIR = Path("outputs/paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

MODEL_COLORS = {
    "MOMENT": "#1f77b4",
    "Chronos": "#ff7f0e",
    "PatchTST": "#2ca02c",
    "GPT4TS": "#d62728",
}

MODEL_MARKERS = {
    "MOMENT": "o",
    "Chronos": "s",
    "PatchTST": "^",
    "GPT4TS": "D",
}

MODELS = {
    "moment_pca512": "MOMENT",
    "chronos": "Chronos",
    "patchtst_pretrained": "PatchTST",
    "gpt4ts_pca512": "GPT4TS",
}

PROPERTIES = {
    "trend": "synthetic_trend",
    "stationarity": "synthetic_stationarity",
}

DEFAULT_SAMPLE_SIZES = [100, 250, 500, 1000, 2500, 5000]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to sys.argv[1:] when None.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run sample-efficiency probe analysis and generate learning curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--representations_root",
        type=str,
        default="outputs/representations",
        help="Root directory containing representations organized by model/dataset.",
    )
    parser.add_argument(
        "--eval_root",
        type=str,
        default="outputs/eval",
        help="Root directory containing layer-wise evaluation JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUT_DIR),
        help="Directory to save figure and tabular analysis outputs.",
    )
    parser.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLE_SIZES,
        help="Sample sizes to evaluate.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=3,
        help="Number of random seeds per sample size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Probe training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Probe optimizer learning rate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Probe mini-batch size.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split used by ProbeTrainer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string (e.g., "cuda:1", "cuda", "cpu"). Auto-select if None.',
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base seed used to derive per-run seeds.",
    )
    return parser.parse_args(argv)


def _layer_sort_key(layer_name: str) -> tuple[int, str]:
    """Sort layers by trailing numeric index when available."""
    parts = layer_name.replace(".", "_").split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part), layer_name
    return 0, layer_name


def _find_eval_metrics_path(eval_root: Path, model_key: str, property_name: str) -> Path | None:
    """Locate evaluation metrics JSON for a model/property pair."""
    candidates = [
        eval_root / f"{model_key}_synthetic_{property_name}" / "layer_metrics.json",
        eval_root / f"{model_key}_{property_name}" / "layer_metrics.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _discover_layers(repr_dir: Path) -> list[str]:
    """Discover and sort representation layer names."""
    layer_files = [path for path in repr_dir.glob("*.pt") if path.stem != "labels"]
    return sorted([path.stem for path in layer_files], key=_layer_sort_key)


def _find_best_layer(eval_root: Path, model_key: str, property_name: str) -> str | None:
    """Find the best-performing layer by validation accuracy."""
    metrics_path = _find_eval_metrics_path(eval_root, model_key, property_name)
    if metrics_path is None:
        return None

    data = json.loads(metrics_path.read_text())
    if not isinstance(data, list) or not data:
        return None

    best_row = max(data, key=lambda row: float(row.get("val_accuracy", float("-inf"))))
    best_layer = best_row.get("layer")
    if isinstance(best_layer, str):
        return best_layer
    return None


def _load_classification_data(repr_dir: Path, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and validate representations and labels for classification probing."""
    layer_path = repr_dir / f"{layer_name}.pt"
    labels_path = repr_dir / "labels.pt"

    if not layer_path.exists():
        raise FileNotFoundError(f"Layer file not found: {layer_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    representations = torch.load(layer_path, map_location="cpu", weights_only=True)
    labels = torch.load(labels_path, map_location="cpu", weights_only=True)

    if representations.ndim > 2:
        representations = representations.reshape(representations.shape[0], -1)
    if representations.ndim != 2:
        raise ValueError(f"Expected 2D representations, got {tuple(representations.shape)}")

    if labels.ndim > 1:
        labels = labels.squeeze(-1)
    labels = labels.long()

    if representations.shape[0] != labels.shape[0]:
        raise ValueError(
            "Sample mismatch between representations and labels: "
            f"{representations.shape[0]} vs {labels.shape[0]}"
        )

    return representations.float(), labels


def _train_probe_once(
    representations: torch.Tensor,
    labels: torch.Tensor,
    *,
    sample_size: int,
    seed: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    val_split: float,
    device: str,
) -> float:
    """Train a linear probe on a sampled subset and return validation accuracy."""
    seed_everything(seed)
    n_samples = representations.shape[0]
    take = min(sample_size, n_samples)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(n_samples, generator=generator)[:take]

    subset_repr = representations[indices]
    subset_labels = labels[indices]
    if subset_labels.numel() == 0:
        raise ValueError("Subset labels tensor is empty")

    num_classes = int(subset_labels.max().item()) + 1
    probe = LinearProbe(input_dim=subset_repr.shape[1], output_dim=num_classes)
    trainer = ProbeTrainer(
        probe,
        ProbeTrainerConfig(
            learning_rate=learning_rate,
            num_epochs=epochs,
            batch_size=batch_size,
            probe_type="classification",
            val_split=val_split,
            split_seed=seed,
            device=device,
            verbose=False,
        ),
    )

    _, metrics = trainer.train(subset_repr, subset_labels)
    return float(metrics.get("val_accuracy", 0.0))


def _plot_learning_curves(
    summary: dict[str, dict[str, dict[int, tuple[float, float]]]],
    sample_sizes: list[int],
    output_dir: Path,
) -> None:
    """Plot mean±std sample-efficiency curves for trend and stationarity."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)

    for ax, property_name in zip(axes, PROPERTIES.keys(), strict=False):
        for model_key, model_label in MODELS.items():
            means = [summary[property_name][model_key][size][0] for size in sample_sizes]
            stds = [summary[property_name][model_key][size][1] for size in sample_sizes]
            x = np.array(sample_sizes, dtype=np.float64)
            y = np.array(means, dtype=np.float64)
            s = np.array(stds, dtype=np.float64)

            ax.plot(
                x,
                y,
                color=MODEL_COLORS[model_label],
                marker=MODEL_MARKERS[model_label],
                linewidth=1.8,
                markersize=4,
                label=model_label,
            )
            ax.fill_between(
                x,
                np.clip(y - s, 0.0, 1.0),
                np.clip(y + s, 0.0, 1.0),
                color=MODEL_COLORS[model_label],
                alpha=0.15,
            )

        ax.set_title(property_name.title(), fontweight="bold")
        ax.set_xlabel("Training Samples")
        ax.set_xscale("log")
        ax.set_xticks(sample_sizes)
        ax.set_xticklabels([str(size) for size in sample_sizes], rotation=30, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Validation Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Sample Efficiency of Linear Probes (Best Layer)", fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))

    fig.savefig(output_dir / "sample_efficiency_learning_curves.pdf")
    fig.savefig(output_dir / "sample_efficiency_learning_curves.png")
    plt.close(fig)


def _save_raw_results(
    raw_scores: list[dict[str, object]],
    summary: dict[str, dict[str, dict[int, tuple[float, float]]]],
    output_dir: Path,
) -> None:
    """Save per-run and aggregated sample-efficiency results."""
    json_path = output_dir / "sample_efficiency_results.json"
    json_path.write_text(json.dumps(raw_scores, indent=2))

    summary_path = output_dir / "sample_efficiency_summary.json"
    serializable_summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for property_name, model_data in summary.items():
        serializable_summary[property_name] = {}
        for model_key, size_data in model_data.items():
            serializable_summary[property_name][model_key] = {}
            for size, (mean_val, std_val) in size_data.items():
                serializable_summary[property_name][model_key][str(size)] = {
                    "mean": mean_val,
                    "std": std_val,
                }
    summary_path.write_text(json.dumps(serializable_summary, indent=2))

    csv_path = output_dir / "sample_efficiency_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "property",
                "model_key",
                "model_label",
                "layer",
                "sample_size",
                "seed",
                "val_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_scores)


def _resolve_training_device(device_arg: str | None) -> str:
    """Resolve a requested device with fallback for unavailable CUDA indices."""
    if device_arg is None:
        return str(resolve_device(None))

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        if ":" in device_arg:
            _, index_str = device_arg.split(":", maxsplit=1)
            if index_str.isdigit() and int(index_str) >= torch.cuda.device_count():
                print(
                    f"Warning: Requested {device_arg} unavailable with visible devices. "
                    "Falling back to cuda:0."
                )
                return "cuda:0"
    return device_arg


def main() -> None:
    """Run sample-efficiency experiments and save figures/tables."""
    args = parse_args()

    representations_root = Path(args.representations_root)
    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_training_device(args.device)
    sample_sizes = sorted(set(args.sample_sizes))
    seeds = [args.base_seed + idx for idx in range(args.num_seeds)]

    raw_scores: list[dict[str, object]] = []
    summary: dict[str, dict[str, dict[int, tuple[float, float]]]] = {
        property_name: {model_key: {} for model_key in MODELS} for property_name in PROPERTIES
    }

    print(f"Using device: {device}")

    for property_name, dataset_name in PROPERTIES.items():
        for model_key, model_label in MODELS.items():
            repr_dir = representations_root / model_key / dataset_name
            if not repr_dir.exists():
                raise FileNotFoundError(f"Representation directory not found: {repr_dir}")

            available_layers = _discover_layers(repr_dir)
            if not available_layers:
                raise FileNotFoundError(f"No layer files found in {repr_dir}")

            best_layer = _find_best_layer(eval_root, model_key, property_name)
            if best_layer is None or (repr_dir / f"{best_layer}.pt").exists() is False:
                best_layer = available_layers[-1]

            representations, labels = _load_classification_data(repr_dir, best_layer)
            max_samples = representations.shape[0]

            print(f"[{model_label} | {property_name}] layer={best_layer}, N={max_samples}")
            for sample_size in sample_sizes:
                effective_size = min(sample_size, max_samples)
                run_scores: list[float] = []

                for seed in seeds:
                    accuracy = _train_probe_once(
                        representations,
                        labels,
                        sample_size=effective_size,
                        seed=seed,
                        learning_rate=args.learning_rate,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        val_split=args.val_split,
                        device=device,
                    )
                    run_scores.append(accuracy)
                    raw_scores.append(
                        {
                            "property": property_name,
                            "model_key": model_key,
                            "model_label": model_label,
                            "layer": best_layer,
                            "sample_size": effective_size,
                            "seed": seed,
                            "val_accuracy": accuracy,
                        }
                    )

                summary[property_name][model_key][sample_size] = (
                    float(np.mean(run_scores)),
                    float(np.std(run_scores)),
                )

    _save_raw_results(raw_scores, summary, output_dir)
    _plot_learning_curves(summary, sample_sizes, output_dir)
    print(f"Saved sample-efficiency outputs to {output_dir}")


if __name__ == "__main__":
    main()
