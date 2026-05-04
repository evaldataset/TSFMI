"""Generate t-SNE and UMAP visualizations for Layer 0 vs best layer.

This script loads pre-extracted frozen representations from
``outputs/representations/{model}/{dataset}/`` and creates publication-ready
2D projection figures for:
    - synthetic_trend (3-class)
    - synthetic_stationarity (binary)

For each model and property, it compares Layer 0 against the best layer chosen
from probe evaluation metrics in ``outputs/eval``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypeAlias

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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

PROPERTY_INFO = {
    "trend": {
        "dataset": "synthetic_trend",
        "class_names": ["Down", "Flat", "Up"],
        "metric_key": "val_accuracy",
    },
    "stationarity": {
        "dataset": "synthetic_stationarity",
        "class_names": ["Non-stationary", "Stationary"],
        "metric_key": "val_accuracy",
    },
}

CLASS_COLORS = {
    "trend": ["#3b4cc0", "#7f7f7f", "#b40426"],
    "stationarity": ["#4c78a8", "#f58518"],
}

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list. Defaults to sys.argv[1:] when None.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate Layer0 vs best-layer t-SNE/UMAP paper figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--representations_root",
        type=str,
        default="outputs/representations",
        help="Root directory that contains model representation subdirectories.",
    )
    parser.add_argument(
        "--eval_root",
        type=str,
        default="outputs/eval",
        help="Root directory that contains per-model evaluation summaries.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUT_DIR),
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(MODELS.keys()),
        help="Model keys to visualize.",
    )
    parser.add_argument(
        "--properties",
        type=str,
        nargs="+",
        default=list(PROPERTY_INFO.keys()),
        help="Temporal properties to visualize.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["tsne", "umap"],
        choices=["tsne", "umap"],
        help="Projection methods to run.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2000,
        help="Maximum number of samples used for each projection.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Base t-SNE perplexity (auto-clipped by sample size).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling and projection reproducibility.",
    )
    return parser.parse_args(argv)


def _layer_sort_key(layer_name: str) -> tuple[int, str]:
    """Sort layers by trailing numeric index when available."""
    parts = layer_name.replace(".", "_").split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part), layer_name
    return 0, layer_name


def _discover_layers(repr_dir: Path) -> list[str]:
    """Discover layer filenames in a representations directory.

    Args:
        repr_dir: Directory containing ``{layer}.pt`` and ``labels.pt``.

    Returns:
        Layer names sorted by numeric index.
    """
    layer_files = [path for path in repr_dir.glob("*.pt") if path.stem != "labels"]
    layer_names = [path.stem for path in layer_files]
    return sorted(layer_names, key=_layer_sort_key)


def _load_labels(repr_dir: Path) -> IntArray:
    """Load integer labels from representation directory.

    Args:
        repr_dir: Directory that contains ``labels.pt``.

    Returns:
        Numpy array of shape ``(N,)`` and dtype ``int64``.
    """
    labels_path = repr_dir / "labels.pt"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels file: {labels_path}")

    labels = torch.load(labels_path, map_location="cpu", weights_only=True)
    if labels.ndim > 1:
        labels = labels.squeeze(-1)
    return np.asarray(labels.long().cpu().numpy(), dtype=np.int64)


def _flatten_representations(representations: torch.Tensor) -> FloatArray:
    """Flatten representations to ``(N, features)``.

    Args:
        representations: Tensor of shape ``(N, D)`` or higher dimensional form.

    Returns:
        2D numpy array.
    """
    if representations.ndim < 2:
        raise ValueError(f"Expected at least 2D tensor, got shape {tuple(representations.shape)}")
    if representations.ndim > 2:
        representations = representations.reshape(representations.shape[0], -1)
    return np.asarray(representations.float().cpu().numpy(), dtype=np.float64)


def _find_eval_metrics_path(eval_root: Path, model_key: str, property_name: str) -> Path | None:
    """Locate evaluation metrics JSON for a model/property combination."""
    candidates = [
        eval_root / f"{model_key}_synthetic_{property_name}" / "layer_metrics.json",
        eval_root / f"{model_key}_{property_name}" / "layer_metrics.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _find_best_layer(
    eval_root: Path, model_key: str, property_name: str, metric_key: str
) -> str | None:
    """Find the best layer name from evaluation metrics.

    Args:
        eval_root: Root directory containing evaluation outputs.
        model_key: Model directory key.
        property_name: Property key (e.g., trend).
        metric_key: Metric used to choose best layer.

    Returns:
        Best layer name or None if unavailable.
    """
    metrics_path = _find_eval_metrics_path(eval_root, model_key, property_name)
    if metrics_path is None:
        return None

    data = json.loads(metrics_path.read_text())
    if not isinstance(data, list) or not data:
        return None

    best_row = max(data, key=lambda row: float(row.get(metric_key, float("-inf"))))
    layer_name = best_row.get("layer")
    if isinstance(layer_name, str):
        return layer_name
    return None


def _subsample_indices(num_samples: int, max_samples: int, seed: int) -> IntArray:
    """Return deterministic subset indices up to max_samples."""
    if num_samples <= max_samples:
        return np.arange(num_samples, dtype=np.int64)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_samples, size=max_samples, replace=False))


def _run_tsne(features: FloatArray, *, seed: int, base_perplexity: float) -> FloatArray:
    """Compute t-SNE projection on standardized features."""
    n_samples = features.shape[0]
    max_perplexity = max(5.0, min(float(n_samples - 1) / 3.0, 50.0))
    perplexity = min(base_perplexity, max_perplexity)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(features_scaled)
    return np.asarray(embedding, dtype=np.float64)


def _run_umap(features: FloatArray, *, seed: int) -> FloatArray:
    """Compute UMAP projection on standardized features."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP dependency missing. Install with: .venv/bin/python -m pip install umap-learn"
        ) from exc

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=seed,
    )
    embedding = reducer.fit_transform(features_scaled)
    return np.asarray(embedding, dtype=np.float64)


def _plot_method_figure(
    *,
    method: str,
    property_name: str,
    class_names: list[str],
    model_keys: list[str],
    projections: dict[tuple[str, str], tuple[FloatArray, IntArray, str]],
    output_dir: Path,
) -> None:
    """Plot all models in a grid for one property and one projection method.

    Args:
        method: ``"tsne"`` or ``"umap"``.
        property_name: Property key.
        class_names: Ordered class names for legend.
        projections: Mapping from (model_key, layer_tag) to projection tuple.
        output_dir: Directory for saving figures.
    """
    layer_tags = ["layer0", "best"]
    fig, axes = plt.subplots(
        len(model_keys), len(layer_tags), figsize=(8.4, 9.6), sharex=False, sharey=False
    )
    if len(model_keys) == 1:
        axes = np.expand_dims(axes, axis=0)
    colors = CLASS_COLORS[property_name]

    for row_idx, model_key in enumerate(model_keys):
        for col_idx, layer_tag in enumerate(layer_tags):
            ax = axes[row_idx, col_idx]
            emb, labels, layer_name = projections[(model_key, layer_tag)]

            for class_idx, class_name in enumerate(class_names):
                mask = labels == class_idx
                if not np.any(mask):
                    continue
                ax.scatter(
                    emb[mask, 0],
                    emb[mask, 1],
                    s=6,
                    alpha=0.65,
                    c=colors[class_idx],
                    edgecolors="none",
                    label=class_name,
                )

            if row_idx == 0:
                column_name = "Layer 0" if layer_tag == "layer0" else "Best Layer"
                ax.set_title(column_name, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(MODELS[model_key], fontweight="bold")

            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.02,
                0.98,
                layer_name,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.2},
            )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=colors[idx],
            markeredgecolor="none",
            markersize=5,
            label=name,
        )
        for idx, name in enumerate(class_names)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(class_names), frameon=False)
    fig.suptitle(
        f"{method.upper()} Projections: Synthetic {property_name.title()} (Layer 0 vs Best Layer)",
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))

    stem = f"{method}_synthetic_{property_name}_layer0_vs_best"
    fig.savefig(output_dir / f"{stem}.pdf")
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)


def main() -> None:
    """Generate all requested t-SNE/UMAP paper figures."""
    args = parse_args()

    representations_root = Path(args.representations_root)
    eval_root = Path(args.eval_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_key in args.models:
        if model_key not in MODELS:
            raise ValueError(f"Unknown model key: {model_key}")
    for property_name in args.properties:
        if property_name not in PROPERTY_INFO:
            raise ValueError(f"Unknown property: {property_name}")

    for property_name in args.properties:
        info = PROPERTY_INFO[property_name]
        dataset_name = str(info["dataset"])
        class_names = list(info["class_names"])
        metric_key = str(info["metric_key"])

        projections_tsne: dict[tuple[str, str], tuple[FloatArray, IntArray, str]] = {}
        projections_umap: dict[tuple[str, str], tuple[FloatArray, IntArray, str]] = {}

        for model_key in args.models:
            repr_dir = representations_root / model_key / dataset_name
            if not repr_dir.exists():
                raise FileNotFoundError(f"Representation directory not found: {repr_dir}")

            layer_names = _discover_layers(repr_dir)
            if not layer_names:
                raise FileNotFoundError(f"No layer tensors found in {repr_dir}")

            layer0_name = layer_names[0]
            best_layer_name = _find_best_layer(eval_root, model_key, property_name, metric_key)
            if best_layer_name is None or (repr_dir / f"{best_layer_name}.pt").exists() is False:
                best_layer_name = layer_names[-1]

            labels = _load_labels(repr_dir)
            sample_idx = _subsample_indices(labels.shape[0], args.max_samples, args.seed)
            labels_sub = labels[sample_idx]

            for layer_tag, layer_name in [("layer0", layer0_name), ("best", best_layer_name)]:
                layer_path = repr_dir / f"{layer_name}.pt"
                representations = torch.load(layer_path, map_location="cpu", weights_only=True)
                features = _flatten_representations(representations)[sample_idx]

                if "tsne" in args.methods:
                    embeddings_tsne = _run_tsne(
                        features, seed=args.seed, base_perplexity=args.perplexity
                    )
                    projections_tsne[(model_key, layer_tag)] = (
                        embeddings_tsne,
                        labels_sub,
                        layer_name,
                    )
                if "umap" in args.methods:
                    embeddings_umap = _run_umap(features, seed=args.seed)
                    projections_umap[(model_key, layer_tag)] = (
                        embeddings_umap,
                        labels_sub,
                        layer_name,
                    )

        if "tsne" in args.methods:
            _plot_method_figure(
                method="tsne",
                property_name=property_name,
                class_names=class_names,
                model_keys=args.models,
                projections=projections_tsne,
                output_dir=output_dir,
            )
        if "umap" in args.methods:
            _plot_method_figure(
                method="umap",
                property_name=property_name,
                class_names=class_names,
                model_keys=args.models,
                projections=projections_umap,
                output_dir=output_dir,
            )

    print(f"Saved t-SNE/UMAP figures to {output_dir}")


if __name__ == "__main__":
    main()
