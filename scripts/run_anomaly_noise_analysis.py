# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.datasets.synthetic import SyntheticDataset, generate_anomaly_dataset
from src.extractors.hook_manager import HookManager
from src.models.base import BaseModelWrapper
from src.models.chronos_wrapper import ChronosBoltWrapper
from src.models.gpt4ts_wrapper import GPT4TSWrapper
from src.models.moment_wrapper import MOMENTWrapper
from src.models.patchtst_wrapper import PatchTSTWrapper
from src.probes.linear_probe import LinearProbe
from src.probes.probe_trainer import ProbeTrainer, ProbeTrainerConfig
from src.utils.seed import seed_everything


@dataclass
class ModelSpec:
    key: str
    display_name: str
    color: str
    checkpoint: str
    pca_dim: int | None


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        key="moment",
        display_name="MOMENT",
        color="#1f77b4",
        checkpoint="AutonLab/MOMENT-1-large",
        pca_dim=512,
    ),
    ModelSpec(
        key="chronos",
        display_name="Chronos",
        color="#ff7f0e",
        checkpoint="amazon/chronos-bolt-small",
        pca_dim=None,
    ),
    ModelSpec(
        key="patchtst_pretrained",
        display_name="PatchTST",
        color="#2ca02c",
        checkpoint="ibm-granite/granite-timeseries-patchtst",
        pca_dim=None,
    ),
    ModelSpec(
        key="gpt4ts",
        display_name="GPT4TS",
        color="#d62728",
        checkpoint="gpt2",
        pca_dim=512,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run anomaly noise robustness analysis across 4 model wrappers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--magnitudes",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 5.0, 8.0],
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_val_split", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--outputs_root", type=str, default="outputs/anomaly_noise")
    parser.add_argument(
        "--eval_root",
        type=str,
        default="outputs/eval",
        help="Directory containing historical layer_metrics.json files.",
    )
    parser.add_argument(
        "--figure_path",
        type=str,
        default="outputs/paper_figures/fig15_anomaly_noise_robustness.pdf",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="outputs/analysis/anomaly_noise_results.json",
    )
    return parser.parse_args()


def _sanitize_layer_name(layer_name: str) -> str:
    return layer_name.replace(".", "_").replace("/", "_")


def _instantiate_wrapper(model_key: str, seq_len: int) -> BaseModelWrapper:
    if model_key == "moment":
        return MOMENTWrapper()
    if model_key == "chronos":
        return ChronosBoltWrapper()
    if model_key == "patchtst_pretrained":
        return PatchTSTWrapper()
    if model_key == "gpt4ts":
        return GPT4TSWrapper(seq_len=seq_len)
    raise ValueError(f"Unsupported model key: {model_key}")


def _load_wrapper(
    spec: ModelSpec,
    device: torch.device,
    seq_len: int,
) -> BaseModelWrapper:
    wrapper = _instantiate_wrapper(spec.key, seq_len=seq_len)

    if spec.key == "patchtst_pretrained":
        if not isinstance(wrapper, PatchTSTWrapper):
            raise RuntimeError("patchtst_pretrained expected PatchTSTWrapper instance")
        wrapper.load(spec.checkpoint, device=device, seq_len=seq_len)
    else:
        wrapper.load(spec.checkpoint, device=device)

    if not wrapper.is_frozen():
        raise RuntimeError(f"Model {spec.key} is expected to be frozen after loading.")
    return wrapper


def _shuffle_dataset(dataset: SyntheticDataset, seed: int) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    indices = np.arange(dataset.sequences.shape[0])
    rng.shuffle(indices)
    return SyntheticDataset(
        sequences=dataset.sequences[indices],
        labels=dataset.labels[indices],
        label_type=dataset.label_type,
        property_name=dataset.property_name,
        metadata=dataset.metadata,
    )


def _save_magnitude_dataset(dataset: SyntheticDataset, magnitude: float, root: Path) -> None:
    magnitude_dir = root / "data" / f"magnitude_{magnitude:g}"
    magnitude_dir.mkdir(parents=True, exist_ok=True)
    np.save(magnitude_dir / "sequences.npy", dataset.sequences)
    np.save(magnitude_dir / "labels.npy", dataset.labels)
    metadata = dict(dataset.metadata)
    metadata["anomaly_magnitude"] = float(magnitude)
    (magnitude_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _extract_best_layer_from_metrics_file(metrics_path: Path) -> tuple[str, float] | None:
    raw = json.loads(metrics_path.read_text())
    if not isinstance(raw, list) or len(raw) == 0:
        return None

    best_layer = ""
    best_acc = float("-inf")
    for row in raw:
        if not isinstance(row, dict):
            continue
        layer = row.get("layer")
        val_accuracy = row.get("val_accuracy")
        if isinstance(layer, str) and isinstance(val_accuracy, (int, float)):
            score = float(val_accuracy)
            if score > best_acc:
                best_acc = score
                best_layer = layer

    if best_layer == "":
        return None
    return best_layer, best_acc


def _score_eval_dir_name(model_key: str, dirname: str) -> int:
    score = 0
    if model_key in dirname:
        score += 100
    if "anomaly" in dirname:
        score += 50
    if "synthetic_anomaly" in dirname:
        score += 20
    if "hard" in dirname:
        score -= 30
    return score


def _find_best_layer_from_eval(
    eval_root: Path,
    model_key: str,
    available_layer_names: list[str],
) -> tuple[str, dict[str, object]]:
    fallback_layer = available_layer_names[-1]
    fallback_meta: dict[str, object] = {
        "selection_source": "fallback_last_layer",
        "selected_layer": fallback_layer,
    }

    candidate_files = list(eval_root.glob("*anomaly*/layer_metrics.json"))
    candidate_files = [path for path in candidate_files if model_key in path.parent.name]
    if not candidate_files:
        return fallback_layer, fallback_meta

    sanitized_to_actual = {
        _sanitize_layer_name(layer_name): layer_name for layer_name in available_layer_names
    }

    best_choice: tuple[int, float, str, Path] | None = None
    for metrics_path in candidate_files:
        best_row = _extract_best_layer_from_metrics_file(metrics_path)
        if best_row is None:
            continue

        layer_sanitized, val_accuracy = best_row
        if layer_sanitized not in sanitized_to_actual:
            continue

        dir_score = _score_eval_dir_name(model_key, metrics_path.parent.name)
        if best_choice is None or (dir_score, val_accuracy) > (best_choice[0], best_choice[1]):
            best_choice = (dir_score, val_accuracy, layer_sanitized, metrics_path)

    if best_choice is None:
        return fallback_layer, fallback_meta

    _, best_acc, layer_sanitized, metrics_path = best_choice
    selected_layer = sanitized_to_actual[layer_sanitized]
    meta: dict[str, object] = {
        "selection_source": "outputs_eval",
        "metrics_file": str(metrics_path),
        "selected_layer": selected_layer,
        "selected_layer_sanitized": layer_sanitized,
        "historical_val_accuracy": best_acc,
    }
    return selected_layer, meta


def _extract_layer_representations(
    wrapper: BaseModelWrapper,
    sequences: torch.Tensor,
    layer_name: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    activations: list[torch.Tensor] = []

    with HookManager(wrapper.model, [layer_name]) as hook_manager:
        for start in range(0, sequences.shape[0], batch_size):
            batch = sequences[start : start + batch_size].to(device)
            _ = wrapper.forward(batch)
            acts = hook_manager.get_activations()[layer_name].detach().cpu()
            activations.append(acts)
            hook_manager.clear()

    return torch.cat(activations, dim=0)


def _prepare_features(
    representations: torch.Tensor,
    pca_dim: int | None,
    seed: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    flat = representations.view(representations.shape[0], -1).float()
    info: dict[str, object] = {
        "original_shape": list(representations.shape),
        "flattened_shape": list(flat.shape),
        "pca_applied": False,
    }

    if pca_dim is None or flat.shape[1] <= pca_dim:
        return flat, info

    n_components = min(pca_dim, flat.shape[0], flat.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    reduced = pca.fit_transform(flat.numpy())
    reduced_tensor = torch.from_numpy(reduced).float()

    info["pca_applied"] = True
    info["pca_dim"] = int(n_components)
    info["pca_explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))
    info["post_pca_shape"] = list(reduced_tensor.shape)
    return reduced_tensor, info


def _train_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
    learning_rate: float,
    num_epochs: int,
    val_split: float,
    device: torch.device,
) -> dict[str, float]:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got {features.shape}")

    labels_long = labels.long()
    if labels_long.numel() == 0:
        raise ValueError("Labels tensor is empty")
    output_dim = int(labels_long.max().item()) + 1
    probe = LinearProbe(input_dim=features.shape[1], output_dim=output_dim)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(features.shape[0], generator=generator)
    shuffled_features = features[permutation]
    shuffled_labels = labels_long[permutation]

    trainer_config = ProbeTrainerConfig(
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=256,
        probe_type="classification",
        val_split=val_split,
        split_seed=seed,
        device=str(device),
        verbose=False,
    )
    trainer = ProbeTrainer(probe=probe, config=trainer_config)
    _, metrics = trainer.train(shuffled_features, shuffled_labels)
    return metrics


def _make_publication_plot(
    results: dict[str, list[dict[str, float]]],
    model_specs: list[ModelSpec],
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        }
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=300)

    for spec in model_specs:
        model_rows = results.get(spec.key, [])
        x_values = [row["anomaly_magnitude"] for row in model_rows]
        y_values = [row["val_accuracy"] for row in model_rows]
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            color=spec.color,
            label=spec.display_name,
        )

    ax.set_xlabel("Anomaly Magnitude (SNR)")
    ax.set_ylabel("Anomaly Detection Accuracy")
    ax.set_title("Anomaly Noise Robustness")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
    ax.set_xlim(0.8, 8.2)
    ax.set_ylim(0.0, 1.0)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    outputs_root = Path(args.outputs_root)
    eval_root = Path(args.eval_root)
    figure_path = Path(args.figure_path)
    summary_path = Path(args.summary_path)

    outputs_root.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Magnitudes: {args.magnitudes}")

    datasets_by_magnitude: dict[float, SyntheticDataset] = {}
    for magnitude in args.magnitudes:
        generated = generate_anomaly_dataset(
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            anomaly_magnitude=float(magnitude),
            seed=args.seed,
        )
        shuffled = _shuffle_dataset(generated, seed=args.seed)
        datasets_by_magnitude[float(magnitude)] = shuffled
        _save_magnitude_dataset(shuffled, magnitude=float(magnitude), root=outputs_root)

    results: dict[str, list[dict[str, float]]] = {spec.key: [] for spec in MODEL_SPECS}
    layer_selection: dict[str, dict[str, object]] = {}

    for model_index, spec in enumerate(MODEL_SPECS):
        print(f"\n=== [{model_index + 1}/{len(MODEL_SPECS)}] {spec.display_name} ===")
        wrapper = _load_wrapper(spec=spec, device=device, seq_len=args.seq_len)
        available_layer_names = wrapper.get_layer_names()
        if not available_layer_names:
            raise RuntimeError(f"No hookable layers found for model: {spec.key}")

        selected_layer, selection_meta = _find_best_layer_from_eval(
            eval_root=eval_root,
            model_key=spec.key,
            available_layer_names=available_layer_names,
        )
        layer_selection[spec.key] = selection_meta
        print(f"Selected layer: {selected_layer}")
        print(f"Layer source: {selection_meta['selection_source']}")

        for magnitude in args.magnitudes:
            magnitude_value = float(magnitude)
            print(f"  -> magnitude={magnitude_value:g}")
            dataset = datasets_by_magnitude[magnitude_value]
            sequences = torch.from_numpy(dataset.sequences).float()
            labels = torch.from_numpy(dataset.labels).long()

            raw_repr = _extract_layer_representations(
                wrapper=wrapper,
                sequences=sequences,
                layer_name=selected_layer,
                batch_size=args.batch_size,
                device=device,
            )
            features, repr_info = _prepare_features(
                representations=raw_repr,
                pca_dim=spec.pca_dim,
                seed=args.seed,
            )

            metrics = _train_linear_probe(
                features=features,
                labels=labels,
                seed=args.seed,
                learning_rate=args.probe_lr,
                num_epochs=args.probe_epochs,
                val_split=args.probe_val_split,
                device=device,
            )
            val_accuracy = float(metrics.get("val_accuracy", 0.0))

            repr_dir = (
                outputs_root / "representations" / spec.key / f"magnitude_{magnitude_value:g}"
            )
            repr_dir.mkdir(parents=True, exist_ok=True)
            torch.save(features, repr_dir / f"{_sanitize_layer_name(selected_layer)}.pt")
            torch.save(labels, repr_dir / "labels.pt")

            probe_dir = outputs_root / "probes" / spec.key / f"magnitude_{magnitude_value:g}"
            probe_dir.mkdir(parents=True, exist_ok=True)
            (probe_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            (probe_dir / "repr_info.json").write_text(json.dumps(repr_info, indent=2))

            results[spec.key].append(
                {
                    "anomaly_magnitude": magnitude_value,
                    "val_accuracy": val_accuracy,
                    "val_f1_macro": float(metrics.get("val_f1_macro", 0.0)),
                    "val_f1_weighted": float(metrics.get("val_f1_weighted", 0.0)),
                }
            )
            print(f"     val_accuracy={val_accuracy:.4f}")

        del wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for spec in MODEL_SPECS:
        results[spec.key] = sorted(results[spec.key], key=lambda x: x["anomaly_magnitude"])

    _make_publication_plot(results=results, model_specs=MODEL_SPECS, figure_path=figure_path)

    summary = {
        "analysis": "anomaly_noise_robustness",
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "device": str(device),
        "magnitudes": [float(value) for value in args.magnitudes],
        "models": [asdict(spec) for spec in MODEL_SPECS],
        "layer_selection": layer_selection,
        "results": results,
        "figure_path": str(figure_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nCompleted anomaly noise robustness analysis.")
    print(f"Figure: {figure_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
