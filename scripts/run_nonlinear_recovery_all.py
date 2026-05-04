"""Nonlinear recovery analysis across all classification properties.

Extends analyze_nonlinear_frequency.py to all 5 classification properties:
trend, frequency, stationarity, anomaly, change_point.

For each model-property pair:
1. LEACE-erase representations (removes linear concept information)
2. Train LINEAR probe on erased data (should drop to chance)
3. Train MLP probe on erased data (recovers if nonlinear encoding exists)
4. nonlinear_recovery = erased_mlp_acc - erased_linear_acc

High recovery means the concept survives linear erasure — it is encoded
nonlinearly. This addresses the "linear-only methodology" weakness.

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/run_nonlinear_recovery_all.py
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

PROPERTIES = ["trend", "frequency", "stationarity", "anomaly", "change_point"]

MODEL_REPR_KEYS: dict[str, str] = {
    "MOMENT": "moment_pca512",
    "Chronos": "chronos",
    "PatchTST": "patchtst_pretrained",
    "GPT4TS": "gpt4ts_pca512",
    "Timer": "timer_meanpool",
    "TimesFM": "timesfm_meanpool",
    "Moirai": "moirai_meanpool",
}

SELECTED_LAYERS: dict[str, list[int] | None] = {
    "MOMENT": [0, 6, 11, 16, 20, 23],
    "Chronos": None,
    "PatchTST": None,
    "GPT4TS": [0, 2, 5, 8, 11],
    "Timer": None,
    "TimesFM": [0, 10, 20, 30, 40, 49],
    "Moirai": None,
}


@dataclass
class NonlinearResult:
    """Result for one model × property × layer."""

    model_name: str
    property_name: str
    layer_name: str
    original_linear_acc: float
    erased_linear_acc: float
    erased_mlp_acc: float
    nonlinear_recovery: float


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
        """Forward pass."""
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
    """Train a probe and return validation accuracy."""
    if device is None:
        device = torch.device("cpu")

    if use_mlp:
        hidden_dim = min(256, input_dim)
        probe: nn.Module = MLPProbe(input_dim, hidden_dim, num_classes).to(device)
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
    train_repr: torch.Tensor,
    train_labels: torch.Tensor,
    eval_repr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply LEACE erasure, fitting only on training data.

    Args:
        train_repr: Training features (N_train, D).
        train_labels: Training labels (N_train,).
        eval_repr: Evaluation features (N_eval, D).

    Returns:
        Tuple of (erased_train, erased_eval).
    """
    if train_repr.ndim != 2:
        raise ValueError(
            f"Expected 2D representations, got {train_repr.shape}"
        )
    concept_erasure = importlib.import_module("concept_erasure")
    fitter = concept_erasure.LeaceFitter.fit(
        train_repr.float().cpu(),
        train_labels.long().cpu(),
    )
    eraser = fitter.eraser
    erased_train = eraser(train_repr.float().cpu())
    erased_eval = eraser(eval_repr.float().cpu())
    return erased_train, erased_eval


def run_analysis_for_model_property(
    model_name: str,
    property_name: str,
    repr_dir: Path,
    device: torch.device,
    selected_layers: list[int] | None = None,
    val_split: float = 0.2,
) -> list[NonlinearResult]:
    """Run nonlinear recovery analysis for one model-property pair."""
    if not repr_dir.exists():
        print(f"  SKIP: {repr_dir} not found")
        return []

    labels = torch.load(
        repr_dir / "labels.pt", map_location="cpu", weights_only=True,
    )
    if labels.ndim > 1:
        labels = labels.squeeze()
    labels = labels.long()

    meta_path = repr_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        num_classes = int(metadata.get("num_classes", len(torch.unique(labels))))
    else:
        num_classes = int(len(torch.unique(labels)))

    layer_files = sorted(
        [f for f in repr_dir.glob("*.pt") if f.stem != "labels"],
    )

    if selected_layers is not None:
        filtered = []
        for f in layer_files:
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                if int(parts[1]) in selected_layers:
                    filtered.append(f)
        layer_files = filtered

    if not layer_files:
        return []

    n_samples = len(labels)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    torch.manual_seed(42)
    perm = torch.randperm(n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    results: list[NonlinearResult] = []

    for layer_file in layer_files:
        layer_name = layer_file.stem
        reprs = torch.load(layer_file, map_location="cpu", weights_only=True)
        if reprs.ndim == 4:
            reprs = reprs.squeeze(1).mean(dim=1)
        elif reprs.ndim == 3:
            reprs = reprs.mean(dim=1)

        input_dim = reprs.shape[1]

        # Original linear accuracy
        orig_acc = train_probe_on_data(
            reprs[train_idx], labels[train_idx],
            reprs[val_idx], labels[val_idx],
            input_dim, num_classes,
            use_mlp=False, device=device,
        )

        # LEACE erase (fit on train only)
        x_train_e, x_val_e = erase_representations(
            reprs[train_idx].float(), labels[train_idx],
            reprs[val_idx].float(),
        )

        # Linear on erased
        erased_linear_acc = train_probe_on_data(
            x_train_e, labels[train_idx],
            x_val_e, labels[val_idx],
            input_dim, num_classes,
            use_mlp=False, device=device,
        )

        # MLP on erased
        erased_mlp_acc = train_probe_on_data(
            x_train_e, labels[train_idx],
            x_val_e, labels[val_idx],
            input_dim, num_classes,
            use_mlp=True, device=device, num_epochs=150,
        )

        recovery = erased_mlp_acc - erased_linear_acc
        print(
            f"    {layer_name}: orig={orig_acc:.3f} "
            f"erased_lin={erased_linear_acc:.3f} "
            f"erased_mlp={erased_mlp_acc:.3f} "
            f"recovery={recovery:+.3f}"
        )

        results.append(NonlinearResult(
            model_name=model_name,
            property_name=property_name,
            layer_name=layer_name,
            original_linear_acc=orig_acc,
            erased_linear_acc=erased_linear_acc,
            erased_mlp_acc=erased_mlp_acc,
            nonlinear_recovery=recovery,
        ))

    return results


def plot_recovery_heatmap(
    all_results: dict[str, dict[str, list[NonlinearResult]]],
    output_dir: Path,
) -> None:
    """Create 7×5 heatmap of max nonlinear recovery per model-property."""
    plt.rcParams.update({
        "font.size": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
    })

    models = list(all_results.keys())
    props = PROPERTIES
    matrix = np.zeros((len(models), len(props)))

    for i, model in enumerate(models):
        for j, prop in enumerate(props):
            results = all_results.get(model, {}).get(prop, [])
            if results:
                matrix[i, j] = max(r.nonlinear_recovery for r in results)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(props)))
    ax.set_xticklabels(props, rotation=45, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(props)):
            val = matrix[i, j]
            color = "white" if val > 0.25 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, label="Max Nonlinear Recovery")
    ax.set_title(
        "Nonlinear Recovery After LEACE Erasure (MLP − Linear)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "nonlinear_recovery_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "nonlinear_recovery_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run nonlinear recovery analysis across all models and properties."""
    seed_everything(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("outputs/nonlinear_recovery_all")
    output_dir.mkdir(parents=True, exist_ok=True)

    repr_root = Path("outputs/representations")
    all_results: dict[str, dict[str, list[NonlinearResult]]] = {}

    for model_name, repr_key in MODEL_REPR_KEYS.items():
        all_results[model_name] = {}
        for prop in PROPERTIES:
            repr_dir = repr_root / repr_key / f"synthetic_{prop}"
            print(f"\n{'=' * 60}")
            print(f"{model_name} / {prop}")
            print(f"{'=' * 60}")

            results = run_analysis_for_model_property(
                model_name, prop, repr_dir, device,
                selected_layers=SELECTED_LAYERS.get(model_name),
            )
            all_results[model_name][prop] = results

    # Save JSON
    json_results: dict[str, dict[str, list[dict]]] = {}
    for model, props_dict in all_results.items():
        json_results[model] = {}
        for prop, results in props_dict.items():
            json_results[model][prop] = [asdict(r) for r in results]

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(json_results, indent=2))
    print(f"\nSaved results to {results_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Max Nonlinear Recovery (MLP_erased - Linear_erased)")
    print("=" * 80)
    header = f"{'Model':<12}"
    for prop in PROPERTIES:
        header += f" {prop:>14}"
    print(header)
    print("-" * 80)

    for model_name in MODEL_REPR_KEYS:
        row = f"{model_name:<12}"
        for prop in PROPERTIES:
            results = all_results.get(model_name, {}).get(prop, [])
            if results:
                max_rec = max(r.nonlinear_recovery for r in results)
                row += f" {max_rec:>+14.3f}"
            else:
                row += f" {'N/A':>14}"
        print(row)

    # Heatmap
    plot_recovery_heatmap(all_results, output_dir)
    print(f"\nHeatmap saved to {output_dir}/nonlinear_recovery_heatmap.pdf")


if __name__ == "__main__":
    main()
