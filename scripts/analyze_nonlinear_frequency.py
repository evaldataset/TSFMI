"""Analyze nonlinear frequency encoding across time series models.

LEACE removes the linear projection of a concept from representations. If a
concept survives LEACE, the information is encoded nonlinearly. This script:
1. Loads original representations and fits LEACE erasers for frequency
2. Trains LINEAR probes on erased representations (should drop)
3. Trains MLP probes on erased representations (should recover if nonlinear)
4. Compares linear vs MLP post-erasure across models → proves nonlinear encoding

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/analyze_nonlinear_frequency.py
"""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.seed import seed_everything


@dataclass
class NonlinearResult:
    """Result for one model × one layer."""

    model_name: str
    layer_name: str
    original_linear_acc: float
    erased_linear_acc: float
    erased_mlp_acc: float
    nonlinear_recovery: float  # erased_mlp_acc - erased_linear_acc


MODEL_CONFIGS: dict[str, dict[str, str | list[str] | list[int] | None]] = {
    "MOMENT": {
        "repr_dir": "outputs/representations/moment_pca512/synthetic_frequency",
        "probe_dir": "outputs/probes/moment_pca512_frequency_linear",
        "leace_dir": "outputs/leace/moment_pca512_frequency",
        "selected_layers": [0, 6, 11, 16, 20, 23],  # Representative early/mid/late
    },
    "Chronos": {
        "repr_dir": "outputs/representations/chronos/synthetic_frequency",
        "probe_dir": "outputs/probes/chronos_frequency_linear",
        "leace_dir": "outputs/leace/chronos_frequency",
        "selected_layers": None,  # Only 6 layers, use all
    },
    "PatchTST": {
        "repr_dir": "outputs/representations/patchtst_pretrained/synthetic_frequency",
        "probe_dir": "outputs/probes/patchtst_pretrained_frequency_linear",
        "leace_dir": "outputs/leace/patchtst_pretrained_frequency",
        "selected_layers": None,  # Only 3 layers, use all
    },
    "GPT4TS": {
        "repr_dir": "outputs/representations/gpt4ts_pca512/synthetic_frequency",
        "probe_dir": "outputs/probes/gpt4ts_pca512_synthetic_frequency_linear",
        "leace_dir": "outputs/leace/gpt4ts_pca512_frequency",
        "selected_layers": [0, 2, 5, 8, 11],  # Representative subset of 12
    },
    "Timer": {
        "repr_dir": "outputs/representations/timer_meanpool/synthetic_frequency",
        "probe_dir": "outputs/probes/timer_meanpool_synthetic_frequency_linear",
        "leace_dir": "outputs/leace/timer_meanpool_frequency",
        "selected_layers": None,  # Only 8 layers, use all
    },
}

ModelConfig = dict[str, str | list[str] | list[int] | None]
LeaceLayerMetrics = dict[str, dict[str, float]]


class MLPProbe(nn.Module):
    """Two-layer MLP probe for detecting nonlinear encoding."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input, got {x.shape}")
        return self.net(x)


def train_probe_on_data(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    num_classes: int,
    *,
    use_mlp: bool = False,
    num_epochs: int = 200,
    lr: float = 0.001,
    batch_size: int = 256,
    device: torch.device | None = None,
) -> float:
    """Train a probe and return validation accuracy.

    Args:
        x_train: Training features (N_train, D).
        y_train: Training labels (N_train,).
        x_val: Validation features (N_val, D).
        y_val: Validation labels (N_val,).
        input_dim: Feature dimensionality.
        num_classes: Number of target classes.
        use_mlp: If True, use MLP probe; otherwise linear.
        num_epochs: Training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        device: Device for training.

    Returns:
        Validation accuracy.
    """
    if device is None:
        device = torch.device("cpu")

    if use_mlp:
        hidden_dim = min(256, input_dim)
        probe = MLPProbe(input_dim, hidden_dim, num_classes).to(device)
    else:
        probe = nn.Linear(input_dim, num_classes).to(device)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(x_train.to(device), y_train.to(device))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    probe.train()
    for _epoch in range(num_epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = probe(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(x_val.to(device))
        preds = logits.argmax(dim=1)
        acc = float((preds == y_val.to(device)).float().mean().item())

    return acc


def erase_representations(
    representations: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Apply LEACE erasure to remove linear frequency information.

    Args:
        representations: Features of shape (N, D) — must be 2D.
        labels: Integer labels of shape (N,).

    Returns:
        Erased representations of shape (N, D).
    """
    if representations.ndim != 2:
        raise ValueError(f"Expected 2D representations, got {representations.shape}")
    concept_erasure = importlib.import_module("concept_erasure")
    fitter = concept_erasure.LeaceFitter.fit(
        representations.float().cpu(),
        labels.long().cpu(),
    )
    erased = fitter.eraser(representations.float().cpu())
    return erased


def run_analysis_for_model(
    model_name: str,
    config: ModelConfig,
    device: torch.device,
    val_split: float = 0.2,
) -> list[NonlinearResult]:
    """Run nonlinear frequency analysis for a single model.

    Args:
        model_name: Display name (e.g., "MOMENT").
        config: Paths dict with repr_dir, probe_dir, leace_dir.
        device: Device for training.
        val_split: Fraction for validation split.

    Returns:
        List of NonlinearResult per layer.
    """
    repr_dir = Path(str(config["repr_dir"]))
    leace_dir = Path(str(config["leace_dir"]))

    if not repr_dir.exists():
        print(f"  SKIP: {repr_dir} not found")
        return []

    # Load labels
    labels = torch.load(repr_dir / "labels.pt", map_location="cpu", weights_only=True)
    if labels.ndim > 1:
        labels = labels.squeeze()
    labels = labels.long()

    # Load metadata for number of classes
    with open(repr_dir / "metadata.json") as f:
        metadata = json.load(f)

    num_classes = int(metadata.get("num_classes", len(torch.unique(labels))))

    # Load existing LEACE results for reference
    leace_results: dict[str, LeaceLayerMetrics] = {}
    leace_path = leace_dir / "leace_results.json"
    if leace_path.exists():
        with open(leace_path) as f:
            leace_results = json.load(f)

    # Find layer files
    layer_files = sorted(repr_dir.glob("*.pt"))
    layer_files = [f for f in layer_files if f.stem != "labels"]

    # Filter to selected layers if specified
    selected = config.get("selected_layers")
    if isinstance(selected, list):
        # Map layer index to filename pattern
        selected_stems = set()
        for f in layer_files:
            # Extract numeric suffix from stem (e.g., 'encoder_block_6' → 6)
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                if int(parts[1]) in selected:
                    selected_stems.add(f.stem)
        layer_files = [f for f in layer_files if f.stem in selected_stems]
        print(f"  Using {len(layer_files)} selected layers: {[f.stem for f in layer_files]}")

    results: list[NonlinearResult] = []

    n_samples = len(labels)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    # Fixed split
    torch.manual_seed(42)
    perm = torch.randperm(n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    for layer_file in layer_files:
        layer_name = layer_file.stem
        print(f"  {layer_name}:")

        reprs = torch.load(layer_file, map_location="cpu", weights_only=True)

        # Handle multi-dimensional representations (3D → mean pool, 4D → reshape + mean)
        if reprs.ndim == 4:
            # PatchTST: (N, 1, T, D) → (N, T, D) → (N, D)
            reprs = reprs.squeeze(1).mean(dim=1)
        elif reprs.ndim == 3:
            reprs = reprs.mean(dim=1)

        input_dim = reprs.shape[1]

        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # 1. Original linear accuracy (from LEACE results)
        orig_acc = 1.0
        leace_key = layer_name
        if leace_key in leace_results:
            orig_acc = leace_results[leace_key]["before"]["accuracy"]

        # 2. LEACE erase
        erased = erase_representations(reprs.float(), labels)
        x_train_erased = erased[train_idx]
        x_val_erased = erased[val_idx]

        # 3. Linear probe on erased → should drop
        erased_linear_acc = train_probe_on_data(
            x_train_erased,
            y_train,
            x_val_erased,
            y_val,
            input_dim,
            num_classes,
            use_mlp=False,
            device=device,
        )
        print(f"    Linear (erased): {erased_linear_acc:.3f}")

        # 4. MLP probe on erased → should recover if nonlinear
        erased_mlp_acc = train_probe_on_data(
            x_train_erased,
            y_train,
            x_val_erased,
            y_val,
            input_dim,
            num_classes,
            use_mlp=True,
            device=device,
            num_epochs=150,
        )
        print(f"    MLP (erased):    {erased_mlp_acc:.3f}")

        nonlinear_recovery = erased_mlp_acc - erased_linear_acc
        print(f"    Recovery:        {nonlinear_recovery:+.3f}")

        results.append(
            NonlinearResult(
                model_name=model_name,
                layer_name=layer_name,
                original_linear_acc=orig_acc,
                erased_linear_acc=erased_linear_acc,
                erased_mlp_acc=erased_mlp_acc,
                nonlinear_recovery=nonlinear_recovery,
            )
        )

    return results


def plot_nonlinear_frequency(
    all_results: dict[str, list[NonlinearResult]],
    output_dir: Path,
) -> None:
    """Create figure comparing linear vs MLP post-LEACE across models."""
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

    model_names = list(all_results.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 3.5), squeeze=False)

    model_colors = {
        "MOMENT": "#1f77b4",
        "Chronos": "#ff7f0e",
        "PatchTST": "#2ca02c",
        "GPT4TS": "#9467bd",
        "Timer": "#d62728",
    }

    for col, model_name in enumerate(model_names):
        ax = axes[0, col]
        results = all_results[model_name]

        if not results:
            ax.set_visible(False)
            continue

        n_layers = len(results)
        x_pos = np.arange(n_layers)
        bar_width = 0.35

        linear_accs = [r.erased_linear_acc for r in results]
        mlp_accs = [r.erased_mlp_acc for r in results]
        orig_accs = [r.original_linear_acc for r in results]
        layer_labels = [r.layer_name.split("_")[-1] for r in results]

        color = model_colors.get(model_name, "#333333")
        ax.bar(
            x_pos - bar_width / 2,
            linear_accs,
            bar_width,
            label="Linear (post-LEACE)",
            color=color,
            alpha=0.5,
        )
        ax.bar(
            x_pos + bar_width / 2,
            mlp_accs,
            bar_width,
            label="MLP (post-LEACE)",
            color=color,
            alpha=0.9,
        )

        # Original accuracy reference line
        ax.axhline(
            y=np.mean(orig_accs), color="gray", linestyle="--", linewidth=0.8, label="Original"
        )

        # Chance level
        ax.axhline(y=0.2, color="red", linestyle=":", linewidth=0.6, label="Chance (1/5)")

        ax.set_xticks(x_pos)

        if n_layers <= 8:
            ax.set_xticklabels([f"L{layer}" for layer in layer_labels], rotation=45, ha="right")
        else:
            # Sparse labels for models with many layers
            tick_every = max(1, n_layers // 6)
            tick_labels = [
                f"L{layer_labels[i]}" if i % tick_every == 0 else "" for i in range(n_layers)
            ]
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")

        ax.set_ylabel("Accuracy")
        ax.set_title(model_name)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="lower right", fontsize=6)

    fig.suptitle(
        "Nonlinear Frequency Encoding: Linear vs MLP Probes After LEACE Erasure",
        fontweight="bold",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    stem = "fig21_nonlinear_frequency"
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure to {output_dir / stem}.pdf")


def plot_summary_bar(
    all_results: dict[str, list[NonlinearResult]],
    output_dir: Path,
) -> None:
    """Create summary bar chart: avg nonlinear recovery per model."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
        }
    )

    model_colors = {
        "MOMENT": "#1f77b4",
        "Chronos": "#ff7f0e",
        "PatchTST": "#2ca02c",
        "GPT4TS": "#9467bd",
        "Timer": "#d62728",
    }

    models = []
    avg_recoveries = []
    max_recoveries = []
    colors = []

    for model_name, results in all_results.items():
        if not results:
            continue
        recoveries = [r.nonlinear_recovery for r in results]
        models.append(model_name)
        avg_recoveries.append(float(np.mean(recoveries)))
        max_recoveries.append(float(np.max(recoveries)))
        colors.append(model_colors.get(model_name, "#333333"))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(models))
    bar_width = 0.35

    ax.bar(
        x - bar_width / 2, avg_recoveries, bar_width, color=colors, alpha=0.6, label="Avg. Recovery"
    )
    ax.bar(
        x + bar_width / 2, max_recoveries, bar_width, color=colors, alpha=0.9, label="Max Recovery"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Accuracy Recovery (MLP − Linear)")
    ax.set_title("Nonlinear Frequency Recovery After LEACE Erasure", fontweight="bold")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    stem = "fig21b_nonlinear_frequency_summary"
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved summary to {output_dir / stem}.pdf")


def main() -> None:
    """Run nonlinear frequency analysis across all models."""
    seed_everything(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("outputs/paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = Path("outputs/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[NonlinearResult]] = {}

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")
        results = run_analysis_for_model(model_name, config, device)
        all_results[model_name] = results

    # Save results
    json_results: dict[str, list[dict[str, float | str]]] = {}
    for model_name, results in all_results.items():
        json_results[model_name] = [asdict(r) for r in results]

    results_path = analysis_dir / "nonlinear_frequency_results.json"
    results_path.write_text(json.dumps(json_results, indent=2))
    print(f"\nSaved results to {results_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Nonlinear Frequency Encoding")
    print("=" * 80)
    print(
        f"{'Model':<12} {'Layers':<8} {'Avg Linear':<12} {'Avg MLP':<12} "
        f"{'Avg Recovery':<14} {'Max Recovery'}"
    )
    print("-" * 80)
    for model_name, results in all_results.items():
        if not results:
            continue
        n = len(results)
        avg_lin = np.mean([r.erased_linear_acc for r in results])
        avg_mlp = np.mean([r.erased_mlp_acc for r in results])
        avg_rec = np.mean([r.nonlinear_recovery for r in results])
        max_rec = np.max([r.nonlinear_recovery for r in results])
        print(
            f"{model_name:<12} {n:<8} {avg_lin:<12.3f} {avg_mlp:<12.3f} "
            f"{avg_rec:<14.3f} {max_rec:.3f}"
        )

    # Generate figures
    plot_nonlinear_frequency(all_results, output_dir)
    plot_summary_bar(all_results, output_dir)


if __name__ == "__main__":
    main()
