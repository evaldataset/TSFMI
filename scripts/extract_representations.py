# pyright: reportMissingImports=false
"""CLI script for extracting intermediate layer activations from frozen time series models.

Extracts hidden representations from specified layers using HookManager, saves per-layer
tensors and metadata for downstream linear probing experiments.

Usage:
    python scripts/extract_representations.py \
        --model moment \
        --dataset synthetic_trend \
        --layers all \
        --output_dir outputs/representations/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from src.datasets.synthetic import SyntheticDataset
    from src.models.base import BaseModelWrapper

MODEL_CHOICES = [
    "moment",
    "patchtst",
    "patchtst_pretrained",
    "itransformer",
    "chronos",
    "gpt4ts",
    "timer",
    "timesfm",
    "moirai",
    "autoformer",
    "timesnet",
    "fedformer",
]
DATASET_CHOICES = [
    "synthetic_trend",
    "synthetic_seasonality",
    "synthetic_frequency",
    "synthetic_stationarity",
    "synthetic_anomaly",
    "synthetic_change_point",
    "synthetic_trend_hard",
    "synthetic_frequency_hard",
    "synthetic_anomaly_hard",
    "synthetic_stationarity_hard",
    "synthetic_change_point_hard",
    "etth1_trend",
    "etth1_stationarity",
    "etth1_seasonality",
    "etth1_seasonality_binary",
    "etth1_change_point",
    "weather_trend",
    "weather_stationarity",
    "weather_seasonality",
    "weather_seasonality_binary",
    "weather_change_point",
    "electricity_trend",
    "electricity_stationarity",
    "electricity_seasonality",
    "electricity_seasonality_binary",
    "electricity_change_point",
    "traffic_trend",
    "traffic_stationarity",
    "traffic_seasonality",
    "traffic_seasonality_binary",
    "traffic_change_point",
    "exchange_rate_trend",
    "exchange_rate_stationarity",
    "exchange_rate_seasonality",
    "exchange_rate_seasonality_binary",
    "exchange_rate_change_point",
]
DEFAULT_CHECKPOINTS: dict[str, str] = {
    "moment": "AutonLab/MOMENT-1-large",
    "patchtst": "namctin/patchtst_etth1_forecast",
    "patchtst_pretrained": "ibm-granite/granite-timeseries-patchtst",
    "itransformer": "",
    "chronos": "amazon/chronos-bolt-small",
    "gpt4ts": "gpt2",
    "timer": "thuml/timer-base-84m",
    "timesfm": "google/timesfm-2.0-500m-pytorch",
    "moirai": "Salesforce/moirai-2.0-R-small",
    "autoformer": "",
    "timesnet": "",
    "fedformer": "",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for representation extraction.

    Args:
        argv: Argument list. Defaults to sys.argv[1:] if None.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Extract intermediate layer representations from a frozen time series model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODEL_CHOICES,
        help="Target model to extract representations from.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASET_CHOICES,
        help=(
            "Dataset to use as input (synthetic_* or real-world "
            "etth1_*/weather_*/electricity_*/traffic_*/exchange_rate_*)."
        ),
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help='Layers to extract: "all" or comma-separated indices like "0,6,12,18".',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/representations/",
        help="Root directory for saving extracted representations.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Sequence length for synthetic data.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for forward passes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=256,
        help="Stride for real-world dataset windowing (ignored for synthetic).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string ("cuda", "cpu", "cuda:0"). Auto-selects if None.',
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Model checkpoint path or HuggingFace ID. Uses model default if None.",
    )
    return parser.parse_args(argv)


def load_model_wrapper(
    model_name: str,
    checkpoint: str | None,
    device: torch.device,
    seq_len: int = 512,
) -> BaseModelWrapper:
    """Instantiate, load, and freeze the specified model wrapper.

    Args:
        model_name: One of MODEL_CHOICES.
        checkpoint: Checkpoint path or HuggingFace ID. None uses model default.
        device: Target torch device.
        seq_len: Sequence length (needed for iTransformer architecture config).

    Returns:
        Loaded and frozen BaseModelWrapper instance.

    Raises:
        ValueError: If model_name is not recognized.
    """
    from src.models.itransformer_wrapper import iTransformerWrapper
    from src.models.moirai_wrapper import MoiraiWrapper
    from src.models.moment_wrapper import MOMENTWrapper
    from src.models.patchtst_wrapper import PatchTSTWrapper
    from src.models.timer_wrapper import TimerWrapper
    from src.models.timesfm_wrapper import TimesFMWrapper

    ckpt = checkpoint if checkpoint is not None else DEFAULT_CHECKPOINTS[model_name]

    if model_name == "moment":
        wrapper = MOMENTWrapper()
        wrapper.load(ckpt, device=device)
    elif model_name in {"patchtst", "patchtst_pretrained"}:
        wrapper = PatchTSTWrapper()
        wrapper.load(ckpt, device=device, seq_len=seq_len)
    elif model_name == "itransformer":
        wrapper = iTransformerWrapper()
        wrapper.load(ckpt, device=device, seq_len=seq_len)
    elif model_name == "chronos":
        from src.models.chronos_wrapper import ChronosBoltWrapper

        wrapper = ChronosBoltWrapper()
        wrapper.load(ckpt, device=device)
    elif model_name == "gpt4ts":
        from src.models.gpt4ts_wrapper import GPT4TSWrapper

        wrapper = GPT4TSWrapper(seq_len=seq_len)
        wrapper.load(ckpt, device=device)
    elif model_name == "timer":
        wrapper = TimerWrapper()
        wrapper.load(ckpt, device=device)
    elif model_name == "timesfm":
        wrapper = TimesFMWrapper()
        wrapper.load(ckpt, device=device)
    elif model_name == "moirai":
        wrapper = MoiraiWrapper()
        wrapper.load(ckpt, device=device)
    elif model_name == "autoformer":
        from src.models.autoformer_wrapper import AutoformerWrapper

        wrapper = AutoformerWrapper()
        wrapper.load(ckpt, device=device, seq_len=seq_len)
    elif model_name == "timesnet":
        from src.models.timesnet_wrapper import TimesNetWrapper

        wrapper = TimesNetWrapper()
        wrapper.load(ckpt, device=device, seq_len=seq_len)
    elif model_name == "fedformer":
        from src.models.fedformer_wrapper import FEDformerWrapper

        wrapper = FEDformerWrapper()
        wrapper.load(ckpt, device=device, seq_len=seq_len)
    else:
        raise ValueError(f"Unknown model: {model_name!r}. Choose from {MODEL_CHOICES}")

    return wrapper


def load_dataset_by_name(
    name: str,
    num_samples: int,
    seq_len: int,
    seed: int,
    stride: int = 256,
) -> SyntheticDataset:
    """Load a synthetic or real-world dataset by name.

    Args:
        name: Dataset name — 'synthetic_*' for generated data, 'etth1_*',
            'weather_*', 'electricity_*', 'traffic_*', or 'exchange_rate_*'
            for real.
        num_samples: Number of samples (generated for synthetic, subsampled for real).
        seq_len: Length of each time series.
        seed: Random seed for reproducibility.
        stride: Window stride for real-world datasets (ignored for synthetic).

    Returns:
        SyntheticDataset with sequences and ground-truth labels.

    Raises:
        ValueError: If dataset name is not recognized.
    """
    if name.startswith("synthetic_"):
        return _load_synthetic(name, num_samples, seq_len, seed)
    elif (
        name.startswith("etth1_")
        or name.startswith("weather_")
        or name.startswith("electricity_")
        or name.startswith("traffic_")
        or name.startswith("exchange_rate_")
    ):
        return _load_real_world(name, num_samples, seq_len, seed, stride)
    else:
        raise ValueError(f"Unknown dataset: {name!r}. Choose from {DATASET_CHOICES}")


def _load_synthetic(
    name: str,
    num_samples: int,
    seq_len: int,
    seed: int,
) -> SyntheticDataset:
    """Generate a synthetic dataset by name."""
    from src.datasets.synthetic import (
        generate_anomaly_dataset,
        generate_anomaly_hard_dataset,
        generate_change_point_dataset,
        generate_change_point_hard_dataset,
        generate_frequency_dataset,
        generate_frequency_hard_dataset,
        generate_seasonality_dataset,
        generate_stationarity_dataset,
        generate_stationarity_hard_dataset,
        generate_trend_dataset,
        generate_trend_hard_dataset,
    )

    generators = {
        "synthetic_trend": generate_trend_dataset,
        "synthetic_seasonality": generate_seasonality_dataset,
        "synthetic_frequency": generate_frequency_dataset,
        "synthetic_stationarity": generate_stationarity_dataset,
        "synthetic_anomaly": generate_anomaly_dataset,
        "synthetic_change_point": generate_change_point_dataset,
        "synthetic_trend_hard": generate_trend_hard_dataset,
        "synthetic_frequency_hard": generate_frequency_hard_dataset,
        "synthetic_anomaly_hard": generate_anomaly_hard_dataset,
        "synthetic_stationarity_hard": generate_stationarity_hard_dataset,
        "synthetic_change_point_hard": generate_change_point_hard_dataset,
    }

    generator = generators.get(name)
    if generator is None:
        raise ValueError(f"Unknown synthetic dataset: {name!r}")
    return generator(num_samples=num_samples, seq_len=seq_len, seed=seed)


def _load_real_world(
    name: str,
    num_samples: int,
    seq_len: int,
    seed: int,
    stride: int = 256,
) -> SyntheticDataset:
    """Load a real-world dataset by name.

    Supported prefixes: etth1_*, weather_*, electricity_*, traffic_*,
    exchange_rate_*.
    """
    from src.datasets.real_world import load_real_world_dataset

    _REAL_WORLD_PREFIXES = ["etth1", "weather", "electricity", "traffic", "exchange_rate"]
    dataset_name = None
    property_name = None
    for prefix in sorted(_REAL_WORLD_PREFIXES, key=len, reverse=True):
        if name.startswith(prefix + "_"):
            dataset_name = prefix
            property_name = name[len(prefix) + 1 :]
            break
    if dataset_name is None or property_name is None:
        raise ValueError(f"Cannot parse real-world dataset name: {name!r}")

    return load_real_world_dataset(
        dataset_name,
        property_name,
        seq_len=seq_len,
        stride=stride,
        num_samples=num_samples,
        seed=seed,
    )


def parse_layer_names(layers_arg: str, all_names: list[str]) -> list[str]:
    """Parse the --layers argument into a list of layer names.

    Args:
        layers_arg: Either "all" or comma-separated integer indices like "0,6,12".
        all_names: Complete list of hookable layer names from the model.

    Returns:
        Selected layer names.

    Raises:
        IndexError: If any index is out of range.
        ValueError: If indices cannot be parsed as integers.
    """
    if layers_arg == "all":
        return all_names

    indices = [int(x.strip()) for x in layers_arg.split(",")]
    num_layers = len(all_names)
    for idx in indices:
        if idx < 0 or idx >= num_layers:
            raise IndexError(
                f"Layer index {idx} out of range. Model has {num_layers} layers"
                f" (0-{num_layers - 1})."
            )
    return [all_names[i] for i in indices]


def main() -> None:
    """Extract and save intermediate layer representations from a frozen model."""
    args = parse_args()

    import torch

    from src.extractors.hook_manager import HookManager
    from src.utils.device import resolve_device
    from src.utils.seed import seed_everything

    seed_everything(args.seed)
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.num_samples}, Seq len: {args.seq_len}")

    # 1. Load model wrapper
    print("Loading model...")
    wrapper = load_model_wrapper(args.model, args.checkpoint, device, seq_len=args.seq_len)
    print(f"  Frozen: {wrapper.is_frozen()}")

    # 2. Load dataset (synthetic or real-world)
    print("Loading dataset...")
    dataset = load_dataset_by_name(
        args.dataset,
        args.num_samples,
        args.seq_len,
        args.seed,
        stride=args.stride,
    )
    sequences = torch.from_numpy(dataset.sequences).float()
    labels = torch.from_numpy(dataset.labels)
    print(f"  Sequences: {sequences.shape}, Labels: {labels.shape}")

    # 3. Determine layers to extract
    all_layer_names = wrapper.get_layer_names()
    layer_names = parse_layer_names(args.layers, all_layer_names)
    print(f"  Extracting {len(layer_names)} layers: {layer_names}")

    # 4. Extract representations with HookManager
    all_activations: dict[str, list[torch.Tensor]] = {name: [] for name in layer_names}
    num_batches = (len(sequences) + args.batch_size - 1) // args.batch_size

    with HookManager(wrapper.model, layer_names) as hm:
        for batch_idx in range(0, len(sequences), args.batch_size):
            batch = sequences[batch_idx : batch_idx + args.batch_size].to(device)
            _ = wrapper.forward(batch)
            batch_acts = hm.get_activations()
            for layer_name, act in batch_acts.items():
                all_activations[layer_name].append(act.cpu())
            hm.clear()

            current_batch = batch_idx // args.batch_size + 1
            if current_batch % 10 == 0 or current_batch == num_batches:
                print(f"  Batch {current_batch}/{num_batches}")

    # 5. Concatenate and save
    output_dir = Path(args.output_dir) / args.model / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_name, acts in all_activations.items():
        tensor = torch.cat(acts, dim=0)
        safe_name = layer_name.replace(".", "_").replace("/", "_")
        save_path = output_dir / f"{safe_name}.pt"
        torch.save(tensor, save_path)
        print(f"  Saved {save_path} — shape {tensor.shape}")

    # Save labels
    torch.save(labels, output_dir / "labels.pt")

    # Save metadata
    meta = {
        "model": args.model,
        "checkpoint": args.checkpoint or DEFAULT_CHECKPOINTS[args.model],
        "dataset": args.dataset,
        "layers": layer_names,
        "num_samples": len(sequences),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "label_type": dataset.label_type,
        "property_name": dataset.property_name,
        "seed": args.seed,
        "device": str(device),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(meta, indent=2))

    print(f"Saved representations to {output_dir}")
    print(f"  Layers: {len(layer_names)}, Labels: {labels.shape}, Metadata: {metadata_path.name}")


if __name__ == "__main__":
    main()
