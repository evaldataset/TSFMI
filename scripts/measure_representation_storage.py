"""Measure the on-disk footprint of extracted representations against the raw input windows.

Probing every layer of a frozen TSFM means materializing per-layer activations. This script records,
for one extracted dataset, the number of stored layers, the tensor shape and dtype of each view, and
the total bytes on disk, next to the size of the input windows themselves. It reads tensor headers
through ``torch.load`` with ``mmap=True`` so that multi-gigabyte files are not read into memory.

Usage:
    PYTHONPATH=. python scripts/measure_representation_storage.py \
        --repr-root outputs/representations_v2 \
        --dataset synthetic_anomaly --out outputs/representation_storage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from src.datasets.synthetic import generate_anomaly_dataset

# model key -> (subdirectory under --repr-root, stored view)
VIEWS: dict[str, tuple[str, str]] = {
    "moment": ("raw/moment", "all layers, per patch token"),
    "gpt4ts": ("raw/gpt4ts", "all layers, per token"),
    "chronos": ("chronos_meanpool", "all layers, mean-pooled"),
    "timer": ("timer_meanpool", "all layers, mean-pooled"),
    "timesfm": ("timesfm_meanpool", "all layers, mean-pooled"),
    "moirai": ("moirai_meanpool", "all layers, mean-pooled"),
    "patchtst_fm": ("patchtst_fm_meanpool", "all layers, mean-pooled"),
}
NUM_SAMPLES, SEQ_LEN, DATA_SEED = 1000, 512, 42


def layer_files(d: Path) -> list[Path]:
    return sorted(f for f in d.glob("*.pt") if f.stem not in ("labels", "metadata"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repr-root", type=Path, required=True)
    ap.add_argument("--dataset", default="synthetic_anomaly")
    ap.add_argument("--out", type=Path, default=Path("outputs/representation_storage"))
    args = ap.parse_args()

    ds = generate_anomaly_dataset(NUM_SAMPLES, SEQ_LEN, seed=DATA_SEED)
    x = torch.as_tensor(ds.sequences, dtype=torch.float32)
    rows: dict[str, dict] = {}
    for model, (sub, view) in VIEWS.items():
        d = args.repr_root / sub / args.dataset
        files = layer_files(d)
        if not files:
            raise SystemExit(f"{model}: no layer tensors in {d}")
        t = torch.load(files[0], map_location="cpu", weights_only=True, mmap=True)
        rows[model] = {
            "directory": f"{sub}/{args.dataset}",
            "view": view,
            "n_layers": len(files),
            "layer_shape": list(t.shape),
            "dtype": str(t.dtype).removeprefix("torch."),
            "bytes": sum(f.stat().st_size for f in files),
        }
        print(
            f"{model:<12} {len(files):>3} layers {tuple(t.shape)} "
            f"{rows[model]['bytes'] / 2**20:,.0f} MiB"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "num_samples": NUM_SAMPLES,
                "seq_len": SEQ_LEN,
                "input": {
                    "shape": list(x.shape),
                    "dtype": "float32",
                    "bytes": x.element_size() * x.nelement(),
                },
                "models": rows,
                "command": " ".join(sys.argv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
