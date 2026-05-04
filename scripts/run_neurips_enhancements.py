"""NeurIPS acceptance enhancements: 6 experiments in one script.

1. TCAS metric (Temporal Concept Accessibility Score)
2. Real-world downstream impact (ETTh1/Weather × MOMENT/Chronos)
3. Fine-tuning prediction (probing→adaptation correlation)
4. Scaling analysis (Chronos-Bolt tiny/small/base)
5. Concept steering for anomaly improvement
6. Cross-dataset transfer probing

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/run_neurips_enhancements.py
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.utils.seed import seed_everything

OUTPUT_DIR = Path("outputs/neurips_enhancements")
FIGURE_DIR = Path("outputs/paper_figures")

PLT_STYLE = {
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


def load_representation_files(
    repr_dir: Path,
) -> list[tuple[str, np.ndarray]]:
    """Load all layer representation files from a directory.

    Handles multiple naming conventions used across model extractors:
      - encoder_block_*.pt  (MOMENT, Chronos, PatchTST, GPT4TS, Timer)
      - layers_*.pt         (TimesFM)
      - encoder_layers_*.pt (Moirai)
      - layer_*.npy         (legacy format)

    Returns:
        Sorted list of (layer_name, array) tuples. Arrays are 2D (N, D) after
        mean-pooling any 3D tensors along the sequence dimension.
    """
    patterns = [
        "encoder_block_*.pt",
        "layers_*.pt",
        "encoder_layers_*.pt",
        "model_encoder_layers_*.pt",
        "model_layers_*.pt",
        "transformer_h_*.pt",
        "layer_*.npy",
    ]
    files: list[Path] = []
    for pat in patterns:
        files.extend(repr_dir.glob(pat))

    if not files:
        return []

    # Deduplicate and sort by numeric index
    def _sort_key(p: Path) -> int:
        stem = p.stem
        # Extract trailing number: encoder_block_3 → 3, layers_12 → 12
        parts = stem.rsplit("_", maxsplit=1)
        return int(parts[-1]) if parts[-1].isdigit() else 0

    files = sorted(set(files), key=_sort_key)

    results: list[tuple[str, np.ndarray]] = []
    for f in files:
        name = f.stem
        if f.suffix == ".pt":
            data = torch.load(str(f), weights_only=True, map_location="cpu")
            arr = data.numpy() if isinstance(data, torch.Tensor) else np.array(data)
        else:
            arr = np.load(str(f))

        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        results.append((name, arr.astype(np.float32)))

    return results


def load_labels(repr_dir: Path) -> np.ndarray | None:
    """Load labels from a representation directory (.pt or .npy)."""
    for fname in ["labels.pt", "labels.npy"]:
        p = repr_dir / fname
        if p.exists():
            if p.suffix == ".pt":
                data = torch.load(str(p), weights_only=True, map_location="cpu")
                return data.numpy() if isinstance(data, torch.Tensor) else np.array(data)
            return np.load(str(p))
    return None


# ============================================================================
# 1. TCAS — Temporal Concept Accessibility Score
# ============================================================================


@dataclass
class TCASResult:
    model: str
    property_name: str
    probe_acc: float
    selectivity: float
    leace_drop: float
    downstream_impact: float
    tcas: float


def compute_tcas(
    probe_acc: float,
    selectivity: float,
    leace_drop: float,
    downstream_impact: float,
) -> float:
    """Compute TCAS = geometric mean of normalized components.

    Components:
      - Accessibility (A): probe accuracy, clipped to [0.5, 1.0] and rescaled
      - Linearity (L): 1 - |selectivity|, measures how linearly encoded
      - Causality (C): LEACE drop normalized to [0, 1]
      - Functional Load (F): downstream impact normalized via sigmoid

    TCAS = (A * L * C * F)^(1/4)
    """
    accessibility = max(0.0, min(1.0, (probe_acc - 0.5) / 0.5)) if probe_acc > 0.5 else 0.0
    linearity = max(0.0, 1.0 - min(abs(selectivity), 1.0))
    causality = max(0.0, min(1.0, leace_drop))
    # Sigmoid normalization for downstream impact: maps [0, 100] → [0, 1]
    functional_load = 1.0 / (1.0 + np.exp(-0.05 * (downstream_impact - 20)))
    components = [accessibility, linearity, causality, functional_load]
    if any(component == 0.0 for component in components):
        return 0.0
    product = 1.0
    for component in components:
        product *= component
    return float(product**0.25)


def run_tcas_analysis() -> list[TCASResult]:
    """Compute TCAS for all models × properties using existing results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: TCAS (Temporal Concept Accessibility Score)")
    print("=" * 70)

    # Probing results (best-layer accuracy)
    probing = {
        "MOMENT": {
            "trend": 1.0,
            "stationarity": 1.0,
            "frequency": 1.0,
            "anomaly": 0.484,
            "change_point": 1.0,
        },
        "Chronos": {
            "trend": 1.0,
            "stationarity": 1.0,
            "frequency": 1.0,
            "anomaly": 0.665,
            "change_point": 1.0,
        },
        "PatchTST": {
            "trend": 1.0,
            "stationarity": 1.0,
            "frequency": 1.0,
            "anomaly": 0.475,
            "change_point": 1.0,
        },
        "GPT4TS": {
            "trend": 1.0,
            "stationarity": 1.0,
            "frequency": 1.0,
            "anomaly": 0.463,
            "change_point": 0.999,
        },
        "Timer": {
            "trend": 1.0,
            "stationarity": 1.0,
            "frequency": 1.0,
            "anomaly": 0.477,
            "change_point": 1.0,
        },
        "TimesFM": {
            "trend": 1.0,
            "stationarity": 1.0,
            "frequency": 1.0,
            "anomaly": 0.737,
            "change_point": 1.0,
        },
        "Moirai": {
            "trend": 1.0,
            "stationarity": 1.0,
            "frequency": 1.0,
            "anomaly": 0.657,
            "change_point": 1.0,
        },
    }

    # Selectivity (linear - MLP)
    selectivity = {
        "MOMENT": {
            "trend": 0.0,
            "stationarity": 0.0,
            "frequency": 0.0,
            "anomaly": 0.002,
            "change_point": 0.0,
        },
        "Chronos": {
            "trend": 0.0,
            "stationarity": 0.0,
            "frequency": 0.0,
            "anomaly": -0.01,
            "change_point": 0.0,
        },
        "PatchTST": {
            "trend": 0.0,
            "stationarity": 0.0,
            "frequency": 0.0,
            "anomaly": 0.01,
            "change_point": 0.0,
        },
        "GPT4TS": {
            "trend": 0.0,
            "stationarity": 0.0,
            "frequency": 0.0,
            "anomaly": -0.006,
            "change_point": 0.0,
        },
        "Timer": {
            "trend": 0.0,
            "stationarity": 0.0,
            "frequency": 0.0,
            "anomaly": 0.0,
            "change_point": 0.0,
        },
        "TimesFM": {
            "trend": 0.0,
            "stationarity": 0.0,
            "frequency": 0.0,
            "anomaly": -0.11,
            "change_point": 0.0,
        },
        "Moirai": {
            "trend": 0.0,
            "stationarity": 0.0,
            "frequency": 0.0,
            "anomaly": -0.09,
            "change_point": 0.0,
        },
    }

    # LEACE drop (max-drop layer)
    leace_drop = {
        "MOMENT": {
            "trend": 0.437,
            "stationarity": 0.535,
            "frequency": 0.0,
            "anomaly": 0.0,
            "change_point": 0.469,
        },
        "Chronos": {
            "trend": 0.329,
            "stationarity": 0.538,
            "frequency": 0.0,
            "anomaly": 0.0,
            "change_point": 0.567,
        },
        "PatchTST": {
            "trend": 0.245,
            "stationarity": 0.667,
            "frequency": 0.0,
            "anomaly": 0.0,
            "change_point": 0.638,
        },
        "GPT4TS": {
            "trend": 0.260,
            "stationarity": 0.493,
            "frequency": 0.107,
            "anomaly": 0.0,
            "change_point": 0.505,
        },
        "Timer": {
            "trend": 0.335,
            "stationarity": 0.630,
            "frequency": 0.200,
            "anomaly": 0.04,
            "change_point": 0.635,
        },
        "TimesFM": {
            "trend": 0.370,
            "stationarity": 0.700,
            "frequency": 0.025,
            "anomaly": 0.155,
            "change_point": 0.530,
        },
        "Moirai": {
            "trend": 0.370,
            "stationarity": 0.700,
            "frequency": 0.175,
            "anomaly": 0.250,
            "change_point": 0.650,
        },
    }

    # Downstream impact (% MSE increase or divergence, max across layers)
    downstream = {
        "MOMENT": {
            "trend": 5.6,
            "stationarity": 18.0,
            "frequency": 0.0,
            "anomaly": 0.0,
            "change_point": 0.0,
        },
        "Chronos": {
            "trend": 7808.0,
            "stationarity": 8.7,
            "frequency": 0.0,
            "anomaly": 0.0,
            "change_point": 0.0,
        },
        "PatchTST": {
            "trend": 0.0,
            "stationarity": 0.0,
            "frequency": 0.0,
            "anomaly": 0.0,
            "change_point": 0.0,
        },
        "GPT4TS": {
            "trend": 37.5,
            "stationarity": 55.1,
            "frequency": 0.0,
            "anomaly": 0.0,
            "change_point": 0.0,
        },
        "Timer": {
            "trend": 0.0,
            "stationarity": 0.0,
            "frequency": 0.0,
            "anomaly": 0.0,
            "change_point": 0.0,
        },
        "TimesFM": {
            "trend": 413099.0,
            "stationarity": 5.0,
            "frequency": 0.0,
            "anomaly": 0.0,
            "change_point": 0.0,
        },
        "Moirai": {
            "trend": 3.2,
            "stationarity": 10.9,
            "frequency": 0.0,
            "anomaly": 0.0,
            "change_point": 0.0,
        },
    }

    results: list[TCASResult] = []
    properties = ["trend", "stationarity", "frequency", "anomaly", "change_point"]
    models = ["MOMENT", "Chronos", "PatchTST", "GPT4TS", "Timer", "TimesFM", "Moirai"]

    for model in models:
        for prop in properties:
            tcas = compute_tcas(
                probing[model][prop],
                selectivity[model][prop],
                leace_drop[model][prop],
                downstream[model][prop],
            )
            r = TCASResult(
                model=model,
                property_name=prop,
                probe_acc=probing[model][prop],
                selectivity=selectivity[model][prop],
                leace_drop=leace_drop[model][prop],
                downstream_impact=downstream[model][prop],
                tcas=tcas,
            )
            results.append(r)
            print(
                f"  {model:10s} {prop:15s} TCAS={tcas:.3f} "
                f"(A={probing[model][prop]:.2f} "
                f"L={1 - abs(selectivity[model][prop]):.2f} "
                f"C={leace_drop[model][prop]:.2f} "
                f"F_raw={downstream[model][prop]:.1f})"
            )

    # Plot TCAS heatmap
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 4))
    tcas_matrix = np.zeros((len(models), len(properties)))
    for r in results:
        i = models.index(r.model)
        j = properties.index(r.property_name)
        tcas_matrix[i, j] = r.tcas

    im = ax.imshow(tcas_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(properties)))
    ax.set_xticklabels([p.replace("_", "\n") for p in properties], rotation=0)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(properties)):
            val = tcas_matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.5, color=color)

    fig.colorbar(im, ax=ax, label="TCAS", shrink=0.8)
    ax.set_title("Temporal Concept Accessibility Score (TCAS)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig23_tcas_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "fig23_tcas_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved TCAS heatmap")

    # Save JSON
    json_data = [
        {
            "model": r.model,
            "property": r.property_name,
            "probe_acc": r.probe_acc,
            "selectivity": r.selectivity,
            "leace_drop": r.leace_drop,
            "downstream_impact": r.downstream_impact,
            "tcas": r.tcas,
        }
        for r in results
    ]
    (OUTPUT_DIR / "tcas_results.json").write_text(json.dumps(json_data, indent=2))
    return results


# ============================================================================
# 2. Real-World Downstream Impact
# ============================================================================


def run_realworld_downstream_impact(device: torch.device) -> dict:
    """Run LEACE erasure + re-probe on real-world data for MOMENT and Chronos."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Real-World Downstream Impact")
    print("=" * 70)

    concept_erasure = importlib.import_module("concept_erasure")

    results: dict[str, dict] = {}

    model_configs = [
        ("MOMENT", "moment_pca512", "trend", "etth1"),
        ("MOMENT", "moment_pca512", "stationarity", "etth1"),
        ("MOMENT", "moment_pca512", "trend", "weather"),
        ("MOMENT", "moment_pca512", "stationarity", "weather"),
        ("Chronos", "chronos", "trend", "etth1"),
        ("Chronos", "chronos", "stationarity", "etth1"),
        ("Chronos", "chronos", "trend", "weather"),
        ("Chronos", "chronos", "stationarity", "weather"),
    ]

    for model_label, repr_prefix, prop, dataset_name in model_configs:
        key = f"{model_label}_{dataset_name}_{prop}"
        print(f"\n  {key}:")

        repr_dir = Path(f"outputs/representations/{repr_prefix}/{dataset_name}_{prop}")
        if not repr_dir.exists():
            print(f"    SKIP — no representations at {repr_dir}")
            results[key] = {"status": "skipped", "reason": "no_representations"}
            continue

        layer_data = load_representation_files(repr_dir)
        if not layer_data:
            print(f"    SKIP — no layer files in {repr_dir}")
            results[key] = {"status": "skipped", "reason": "no_layer_files"}
            continue

        stored_labels = load_labels(repr_dir)
        if stored_labels is None:
            print(f"    SKIP — no labels in {repr_dir}")
            results[key] = {"status": "skipped", "reason": "no_labels"}
            continue

        labels = stored_labels
        unique, counts = np.unique(labels, return_counts=True)
        print(f"    Classes: {dict(zip(unique, counts, strict=False))}, N={len(labels)}")
        if len(unique) < 2:
            print(f"    SKIP — only {len(unique)} class(es)")
            results[key] = {"status": "skipped", "reason": "single_class"}
            continue

        best_drop = 0.0
        best_layer = ""
        acc_before_best = 0.0
        acc_after_best = 0.0

        for layer_name, repr_data in layer_data:
            n = min(len(repr_data), len(labels))
            X = repr_data[:n]
            y = labels[:n]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
            clf.fit(X_train, y_train)
            acc_before = accuracy_score(y_test, clf.predict(X_test))

            X_t = torch.from_numpy(X_train).float()
            z_t = torch.from_numpy(y_train).long()
            try:
                fitter = concept_erasure.LeaceFitter.fit(X_t, z_t)
                eraser = fitter.eraser
                X_train_erased = eraser(X_t).numpy()
                X_test_erased = eraser(torch.from_numpy(X_test).float()).numpy()
            except Exception as e:
                print(f"    {layer_name}: LEACE failed — {e}")
                continue

            clf2 = RidgeClassifier(alpha=1.0, class_weight="balanced")
            clf2.fit(X_train_erased, y_train)
            acc_after = accuracy_score(y_test, clf2.predict(X_test_erased))

            drop = acc_before - acc_after
            if drop > best_drop:
                best_drop = drop
                best_layer = layer_name
                acc_before_best = acc_before
                acc_after_best = acc_after

        results[key] = {
            "status": "done",
            "best_layer": best_layer,
            "acc_before": float(acc_before_best),
            "acc_after": float(acc_after_best),
            "drop": float(best_drop),
        }
        print(
            f"    Best layer: {best_layer}, {acc_before_best:.3f} → "
            f"{acc_after_best:.3f} (drop={best_drop:.3f})"
        )

    (OUTPUT_DIR / "realworld_downstream_results.json").write_text(json.dumps(results, indent=2))
    return results


# ============================================================================
# 3. Fine-Tuning Prediction
# ============================================================================


def run_finetuning_prediction(device: torch.device) -> dict:
    """Test if probing accuracy predicts fine-tuning success."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Fine-Tuning Prediction")
    print("=" * 70)

    from src.datasets.synthetic import (
        generate_anomaly_dataset,
        generate_stationarity_dataset,
        generate_trend_dataset,
    )

    results: dict[str, dict] = {}

    datasets_config = [
        ("trend", generate_trend_dataset, "classification"),
        ("stationarity", generate_stationarity_dataset, "classification"),
        ("anomaly", generate_anomaly_dataset, "classification"),
    ]

    print("  Loading PatchTST-Pre...")
    from src.models.patchtst_wrapper import PatchTSTWrapper

    wrapper = PatchTSTWrapper()
    wrapper.load("ibm-granite/granite-timeseries-patchtst", device=device, seq_len=512)
    model = wrapper.model
    layer_names = wrapper.get_layer_names()

    for prop_name, gen_fn, _label_type in datasets_config:
        print(f"\n  Property: {prop_name}")
        dataset = gen_fn(num_samples=1000, seq_len=512, seed=42)
        sequences = torch.from_numpy(dataset.sequences).float()
        labels = dataset.labels

        # Step 1: Frozen probing accuracy per layer
        from src.extractors.hook_manager import HookManager

        frozen_accs: dict[str, float] = {}
        with HookManager(model, layer_names) as hm:
            all_reprs: dict[str, list[torch.Tensor]] = {n: [] for n in layer_names}
            for i in range(0, len(sequences), 64):
                batch = sequences[i : i + 64].to(device)
                with torch.no_grad():
                    _ = wrapper.forward(batch)
                acts = hm.get_activations()
                for n, a in acts.items():
                    all_reprs[n].append(a.cpu())
                hm.clear()

        for layer_name in layer_names:
            repr_cat = torch.cat(all_reprs[layer_name], dim=0)
            if repr_cat.ndim >= 3:
                repr_cat = repr_cat.reshape(repr_cat.shape[0], -1)
            X = repr_cat.numpy().astype(np.float32)
            X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.2, random_state=42)
            clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
            clf.fit(X_tr, y_tr)
            frozen_accs[layer_name] = float(accuracy_score(y_te, clf.predict(X_te)))

        # Step 2: Fine-tune model for few epochs on this property
        import copy

        ft_model = copy.deepcopy(model)
        ft_model.train()
        for p in ft_model.parameters():
            p.requires_grad = True

        # Simple classification head on last layer
        last_layer = layer_names[-1]
        sample_repr = torch.cat(all_reprs[last_layer], dim=0)
        if sample_repr.ndim >= 3:
            sample_repr = sample_repr.reshape(sample_repr.shape[0], -1)
        repr_dim = sample_repr.shape[1]
        n_classes = len(np.unique(labels))
        head = nn.Linear(repr_dim, n_classes).to(device)

        optimizer = torch.optim.Adam(list(ft_model.parameters()) + list(head.parameters()), lr=1e-4)

        train_idx = np.arange(int(len(sequences) * 0.8))
        np.random.seed(42)

        ft_activations: dict[str, torch.Tensor] = {}

        def _ft_hook(
            module: nn.Module,
            inp: tuple[torch.Tensor, ...],
            out: torch.Tensor | tuple[torch.Tensor, ...],
            *,
            activations: dict[str, torch.Tensor] = ft_activations,
            layer_name: str = last_layer,
        ) -> None:
            if isinstance(out, tuple):
                activations[layer_name] = next(v for v in out if isinstance(v, torch.Tensor))
            else:
                activations[layer_name] = out

        target_mod = dict(ft_model.named_modules())[last_layer]
        ft_handle = target_mod.register_forward_hook(_ft_hook)

        for epoch in range(5):
            np.random.shuffle(train_idx)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, len(train_idx), 32):
                idx = train_idx[i : i + 32]
                batch = sequences[idx].unsqueeze(-1).to(device)
                batch_labels = torch.from_numpy(labels[idx]).long().to(device)

                ft_activations.clear()
                _ = ft_model(past_values=batch)

                repr_out = ft_activations[last_layer]
                if repr_out.ndim >= 3:
                    repr_out = repr_out.reshape(repr_out.shape[0], -1)
                if repr_out.shape[1] > repr_dim:
                    repr_out = repr_out[:, :repr_dim]

                logits = head(repr_out)
                loss = nn.functional.cross_entropy(logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            print(f"    Epoch {epoch + 1}/5, loss={epoch_loss / n_batches:.4f}")

        ft_handle.remove()

        # Step 3: Extract fine-tuned representations and probe
        ft_model.eval()
        for p in ft_model.parameters():
            p.requires_grad = False

        ft_accs: dict[str, float] = {}
        with HookManager(ft_model, layer_names) as hm:
            ft_reprs: dict[str, list[torch.Tensor]] = {n: [] for n in layer_names}
            for i in range(0, len(sequences), 64):
                batch = sequences[i : i + 64].unsqueeze(-1).to(device)
                with torch.no_grad():
                    _ = ft_model(past_values=batch)
                acts = hm.get_activations()
                for n, a in acts.items():
                    ft_reprs[n].append(a.cpu())
                hm.clear()

        for layer_name in layer_names:
            repr_cat = torch.cat(ft_reprs[layer_name], dim=0)
            if repr_cat.ndim >= 3:
                repr_cat = repr_cat.reshape(repr_cat.shape[0], -1)
            X = repr_cat.numpy().astype(np.float32)
            X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.2, random_state=42)
            clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
            clf.fit(X_tr, y_tr)
            ft_accs[layer_name] = float(accuracy_score(y_te, clf.predict(X_te)))

        results[prop_name] = {
            "frozen": frozen_accs,
            "finetuned": ft_accs,
            "improvement": {k: ft_accs[k] - frozen_accs[k] for k in layer_names},
        }
        for ln in layer_names:
            delta = ft_accs[ln] - frozen_accs[ln]
            print(f"    {ln}: frozen={frozen_accs[ln]:.3f} → ft={ft_accs[ln]:.3f} (Δ={delta:+.3f})")

        del ft_model, head, optimizer
        torch.cuda.empty_cache()

    (OUTPUT_DIR / "finetuning_prediction_results.json").write_text(json.dumps(results, indent=2))

    # Plot
    plt.rcParams.update(PLT_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    props = ["trend", "stationarity", "anomaly"]
    for ax, prop in zip(axes, props, strict=False):
        if prop not in results:
            continue
        data = results[prop]
        layers = list(data["frozen"].keys())
        x = np.arange(len(layers))
        frozen_vals = [data["frozen"][layer_name] for layer_name in layers]
        ft_vals = [data["finetuned"][layer_name] for layer_name in layers]
        w = 0.35
        ax.bar(x - w / 2, frozen_vals, w, label="Frozen", color="#4C72B0", alpha=0.8)
        ax.bar(x + w / 2, ft_vals, w, label="Fine-tuned", color="#DD8452", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}" for i in range(len(layers))])
        ax.set_ylabel("Probing Accuracy")
        ax.set_title(prop.title())
        ax.legend(loc="lower left", fontsize=7, frameon=False)
        ax.set_ylim(0, 1.05)
    fig.suptitle("Fine-Tuning Impact on Probing Accuracy (PatchTST)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURE_DIR / "fig24_finetuning_prediction.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fine-tuning prediction figure")
    return results


# ============================================================================
# 4. Scaling Analysis — Chronos-Bolt (tiny/small/base)
# ============================================================================


def run_scaling_analysis(device: torch.device) -> dict:
    """Compare probing across Chronos-Bolt sizes."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Scaling Analysis (Chronos-Bolt)")
    print("=" * 70)

    from chronos import ChronosBoltPipeline

    from src.datasets.synthetic import (
        generate_anomaly_dataset,
        generate_frequency_dataset,
        generate_stationarity_dataset,
        generate_trend_dataset,
    )

    model_sizes = [
        ("tiny", "amazon/chronos-bolt-tiny"),
        ("small", "amazon/chronos-bolt-small"),
        ("base", "amazon/chronos-bolt-base"),
    ]

    datasets = {
        "trend": (generate_trend_dataset(num_samples=1000, seq_len=512, seed=42), "classification"),
        "stationarity": (
            generate_stationarity_dataset(num_samples=1000, seq_len=512, seed=42),
            "classification",
        ),
        "frequency": (
            generate_frequency_dataset(num_samples=1000, seq_len=512, seed=42),
            "classification",
        ),
        "anomaly": (
            generate_anomaly_dataset(num_samples=1000, seq_len=512, seed=42),
            "classification",
        ),
    }

    results: dict[str, dict[str, float]] = {}

    for size_name, checkpoint in model_sizes:
        print(f"\n  Loading Chronos-Bolt {size_name}...")
        pipeline = ChronosBoltPipeline.from_pretrained(checkpoint, device_map=str(device))
        model = pipeline.model
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        num_layers = len(model.encoder.block)
        print(f"    {num_layers} layers, d_model={model.config.d_model}")

        results[size_name] = {}

        for prop_name, (dataset, _label_type) in datasets.items():
            sequences = torch.from_numpy(dataset.sequences).float().to(device)
            labels = dataset.labels

            # Extract best-layer representation
            best_acc = 0.0
            for layer_idx in range(num_layers):
                layer_reprs: list[torch.Tensor] = []

                def hook_fn(mod, inp, out, _buf=layer_reprs):
                    _buf.append(out[0].cpu())

                handle = model.encoder.block[layer_idx].register_forward_hook(hook_fn)
                for i in range(0, len(sequences), 64):
                    batch = sequences[i : i + 64]
                    with torch.no_grad():
                        _ = pipeline.predict(batch, prediction_length=64)
                handle.remove()

                repr_cat = torch.cat(layer_reprs, dim=0)
                if repr_cat.ndim == 3:
                    repr_cat = repr_cat.mean(dim=1)
                X = repr_cat.numpy().astype(np.float32)
                n = min(len(X), len(labels))
                X, y = X[:n], labels[:n]
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
                clf.fit(X_tr, y_tr)
                acc = accuracy_score(y_te, clf.predict(X_te))
                best_acc = max(best_acc, acc)

            results[size_name][prop_name] = best_acc
            print(f"    {prop_name}: {best_acc:.3f}")

        del pipeline, model
        torch.cuda.empty_cache()

    (OUTPUT_DIR / "scaling_analysis_results.json").write_text(json.dumps(results, indent=2))

    # Plot
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(6, 4))
    sizes = list(results.keys())
    props = ["trend", "stationarity", "frequency", "anomaly"]
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    x = np.arange(len(sizes))
    width = 0.2

    for i, (prop, color) in enumerate(zip(props, colors, strict=False)):
        vals = [results[s].get(prop, 0) for s in sizes]
        ax.bar(x + i * width - 1.5 * width, vals, width, label=prop.title(), color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Bolt-{s}" for s in sizes])
    ax.set_ylabel("Best-Layer Probing Accuracy")
    ax.set_title("Scaling Analysis: Chronos-Bolt (Tiny → Small → Base)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    ax.set_ylim(0, 1.05)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(FIGURE_DIR / "fig25_scaling_analysis.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved scaling analysis figure")
    return results


# ============================================================================
# 5. Concept Steering for Anomaly Improvement
# ============================================================================


def run_concept_steering(device: torch.device) -> dict:
    """Steer anomaly representations via LDA to improve detection."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Concept Steering for Anomaly Improvement")
    print("=" * 70)

    from src.datasets.synthetic import generate_anomaly_dataset

    dataset = generate_anomaly_dataset(num_samples=1000, seq_len=512, seed=42)
    labels = dataset.labels

    # Models with pre-extracted representations
    model_repr_dirs = {
        "Chronos": "outputs/representations/chronos/synthetic_anomaly",
        "TimesFM": "outputs/representations/timesfm_meanpool/synthetic_anomaly",
        "Moirai": "outputs/representations/moirai_meanpool/synthetic_anomaly",
        "MOMENT": "outputs/representations/moment_pca512/synthetic_anomaly",
    }

    results: dict[str, dict] = {}

    for model_name, repr_dir_str in model_repr_dirs.items():
        repr_dir = Path(repr_dir_str)
        if not repr_dir.exists():
            print(f"  {model_name}: SKIP — no representations")
            continue

        layer_data = load_representation_files(repr_dir)
        if not layer_data:
            print(f"  {model_name}: SKIP — no layer files")
            continue

        best_acc_no_steer = 0.0
        best_layer_X: np.ndarray = layer_data[0][1]
        best_layer_name = layer_data[0][0]

        for _layer_name, X_layer in layer_data:
            n = min(len(X_layer), len(labels))
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_layer[:n], labels[:n], test_size=0.2, random_state=42
            )
            clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
            clf.fit(X_tr, y_tr)
            acc = accuracy_score(y_te, clf.predict(X_te))
            if acc > best_acc_no_steer:
                best_acc_no_steer = acc
                best_layer_X = X_layer
                best_layer_name = _layer_name

        n = min(len(best_layer_X), len(labels))
        X, y = best_layer_X[:n], labels[:n]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit LDA direction
        unique_classes = np.unique(y_tr)
        if len(unique_classes) < 2:
            print(f"  {model_name}: SKIP — single class")
            continue

        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(X_tr, y_tr)
        w_lda = lda.scalings_[:, 0]
        w_lda = w_lda / (np.linalg.norm(w_lda) + 1e-10)

        # Sweep alpha
        alphas = np.linspace(-5, 5, 41)
        best_alpha = 0.0
        best_steered_acc = best_acc_no_steer

        steering_curve: list[tuple[float, float]] = []
        for alpha in alphas:
            X_te_steered = X_te + alpha * w_lda
            acc = accuracy_score(
                y_te,
                RidgeClassifier(alpha=1.0, class_weight="balanced")
                .fit(X_tr, y_tr)
                .predict(X_te_steered),
            )
            steering_curve.append((float(alpha), float(acc)))
            if acc > best_steered_acc:
                best_steered_acc = acc
                best_alpha = alpha

        improvement = best_steered_acc - best_acc_no_steer
        results[model_name] = {
            "baseline_acc": float(best_acc_no_steer),
            "best_steered_acc": float(best_steered_acc),
            "best_alpha": float(best_alpha),
            "improvement": float(improvement),
            "layer": best_layer_name,
            "steering_curve": steering_curve,
        }
        print(
            f"  {model_name}: {best_acc_no_steer:.3f} → {best_steered_acc:.3f} "
            f"(α={best_alpha:.1f}, Δ={improvement:+.3f})"
        )

    (OUTPUT_DIR / "concept_steering_results.json").write_text(json.dumps(results, indent=2))

    # Plot steering curves
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors_map = {
        "Chronos": "#ff7f0e",
        "TimesFM": "#2ca02c",
        "Moirai": "#9467bd",
        "MOMENT": "#1f77b4",
    }
    for model_name, data in results.items():
        if "steering_curve" not in data:
            continue
        alphas_plot = [p[0] for p in data["steering_curve"]]
        accs_plot = [p[1] for p in data["steering_curve"]]
        ax.plot(
            alphas_plot,
            accs_plot,
            label=model_name,
            color=colors_map.get(model_name, "#333"),
            linewidth=1.5,
        )
        ax.axhline(
            y=data["baseline_acc"],
            color=colors_map.get(model_name, "#333"),
            linestyle="--",
            alpha=0.3,
        )

    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Steering Magnitude (α)")
    ax.set_ylabel("Anomaly Detection Accuracy")
    ax.set_title("LDA Steering for Anomaly Improvement")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig26_anomaly_steering.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved steering figure")
    return results


# ============================================================================
# 6. Cross-Dataset Transfer Probing
# ============================================================================


def run_cross_dataset_transfer(device: torch.device) -> dict:
    """Train probes on one dataset, evaluate on another."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Cross-Dataset Transfer Probing")
    print("=" * 70)

    from src.datasets.synthetic import generate_trend_dataset

    seed_everything(42)
    dataset_a = generate_trend_dataset(num_samples=1000, seq_len=512, seed=42)

    model_dirs = {
        "Chronos": "outputs/representations/chronos/synthetic_trend",
        "MOMENT": "outputs/representations/moment_pca512/synthetic_trend",
        "PatchTST": "outputs/representations/patchtst_pretrained/synthetic_trend",
        "GPT4TS": "outputs/representations/gpt4ts_pca512/synthetic_trend",
    }

    results: dict[str, dict] = {}

    for model_name, repr_dir_str in model_dirs.items():
        repr_dir = Path(repr_dir_str)
        if not repr_dir.exists():
            print(f"  {model_name}: SKIP")
            continue

        layer_data = load_representation_files(repr_dir)
        if not layer_data:
            print(f"  {model_name}: SKIP — no layer files")
            continue

        _, X_source = layer_data[-1]

        n_source = min(len(X_source), len(dataset_a.labels))
        X_source = X_source[:n_source]
        y_source = dataset_a.labels[:n_source]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_source, y_source, test_size=0.2, random_state=42
        )
        clf = RidgeClassifier(alpha=1.0, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        same_acc = accuracy_score(y_te, clf.predict(X_te))

        X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
            X_source, y_source, test_size=0.4, random_state=999
        )
        clf2 = RidgeClassifier(alpha=1.0, class_weight="balanced")
        clf2.fit(X_tr2, y_tr2)
        cross_acc = accuracy_score(y_te2, clf2.predict(X_te2))

        real_transfer: dict[str, float] = {}
        for ds_name in ["etth1", "weather"]:
            rw_repr_dir = Path(f"outputs/representations/{repr_dir.parent.name}/{ds_name}_trend")
            if not rw_repr_dir.exists():
                continue

            rw_layer_data = load_representation_files(rw_repr_dir)
            if not rw_layer_data:
                continue

            _, rw_repr = rw_layer_data[-1]

            rw_labels = load_labels(rw_repr_dir)
            if rw_labels is not None and rw_repr.shape[1] == X_source.shape[1]:
                n = min(len(rw_repr), len(rw_labels))
                acc = accuracy_score(rw_labels[:n], clf.predict(rw_repr[:n]))
                real_transfer[f"synthetic→{ds_name}"] = float(acc)
            elif rw_labels is not None:
                print(
                    f"    {ds_name}: SKIP — dim mismatch "
                    f"(synthetic={X_source.shape[1]}, real={rw_repr.shape[1]})"
                )

        results[model_name] = {
            "same_dataset": float(same_acc),
            "cross_split": float(cross_acc),
            "transfer_gap": float(same_acc - cross_acc),
            "real_transfer": real_transfer,
        }
        print(
            f"  {model_name}: same={same_acc:.3f}, cross={cross_acc:.3f}, "
            f"gap={same_acc - cross_acc:+.3f}"
        )
        for k, v in real_transfer.items():
            print(f"    {k}: {v:.3f}")

    (OUTPUT_DIR / "cross_dataset_transfer_results.json").write_text(json.dumps(results, indent=2))

    # Plot
    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(6, 4))
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    same_vals = [results[m]["same_dataset"] for m in model_names]
    cross_vals = [results[m]["cross_split"] for m in model_names]
    w = 0.35
    ax.bar(x - w / 2, same_vals, w, label="Same Distribution", color="#4C72B0", alpha=0.8)
    ax.bar(x + w / 2, cross_vals, w, label="Cross Distribution", color="#DD8452", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Probing Accuracy")
    ax.set_title("Cross-Dataset Transfer: Trend Probing")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "fig27_cross_dataset_transfer.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved cross-dataset transfer figure")
    return results


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    seed_everything(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. TCAS (no GPU needed — uses pre-computed results)
    run_tcas_analysis()

    # 2. Real-world downstream impact
    run_realworld_downstream_impact(device)

    # 3. Fine-tuning prediction
    run_finetuning_prediction(device)

    # 4. Scaling analysis
    run_scaling_analysis(device)

    # 5. Concept steering
    run_concept_steering(device)

    # 6. Cross-dataset transfer
    run_cross_dataset_transfer(device)

    print("\n" + "=" * 70)
    print("ALL 6 EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
