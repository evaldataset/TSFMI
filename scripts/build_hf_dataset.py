"""Materialize the canonical TSFMI-Synthetic datasets as Parquet for HuggingFace.

Generates the 6 standard temporal-property datasets and 5 hard variants under
the canonical seed=42 instantiation, splits each into train/val/test under the
same 60/20/20 protocol used in the paper, and writes Parquet files at
``outputs/hf_dataset/<task>/<split>.parquet``.

Output structure::

    outputs/hf_dataset/
        trend/{train,val,test}.parquet
        seasonality/{train,val,test}.parquet
        ...
        change_point_hard/{train,val,test}.parquet
        README.md         (dataset card with YAML front-matter)
        CROISSANT.json    (copied from repo root)

Schema per row::

    sequence: list[float]   # length 512
    label:    int | float   # int for classification, float for regression
    seed:     int           # data-generation seed (always 42 here)
    split_seed: int         # split seed used for this row's partition

Usage::

    PYTHONPATH=. .venv/bin/python scripts/build_hf_dataset.py

Then upload with::

    hf auth login
    hf repo create datasets/<user>/TSFMI --type dataset
    hf upload <user>/TSFMI outputs/hf_dataset/ --repo-type dataset
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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

OUT_DIR = Path("outputs/hf_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLES = 1000
SEQ_LEN = 512
DATA_SEED = 42
SPLIT_SEED = 0  # canonical paper split seed for the train/val/test partition

GENERATORS = {
    "trend": (generate_trend_dataset, "classification"),
    "seasonality": (generate_seasonality_dataset, "regression"),
    "frequency": (generate_frequency_dataset, "classification"),
    "stationarity": (generate_stationarity_dataset, "classification"),
    "anomaly": (generate_anomaly_dataset, "classification"),
    "change_point": (generate_change_point_dataset, "classification"),
    "trend_hard": (generate_trend_hard_dataset, "classification"),
    "frequency_hard": (generate_frequency_hard_dataset, "classification"),
    "stationarity_hard": (generate_stationarity_hard_dataset, "classification"),
    "anomaly_hard": (generate_anomaly_hard_dataset, "classification"),
    "change_point_hard": (generate_change_point_hard_dataset, "classification"),
}


def three_way_split(
    n: int, seed: int, val_ratio: float = 0.20, test_ratio: float = 0.20
) -> tuple[NDArray, NDArray, NDArray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]


def to_parquet(seqs: NDArray, labels: NDArray, dest: Path) -> None:
    df = pd.DataFrame(
        {
            "sequence": list(seqs.astype(np.float32)),
            "label": labels.tolist(),
            "seed": [DATA_SEED] * len(seqs),
            "split_seed": [SPLIT_SEED] * len(seqs),
        }
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)


def main() -> None:
    for task, (gen_fn, task_type) in GENERATORS.items():
        print(f"-- {task} ({task_type})")
        ds = gen_fn(NUM_SAMPLES, SEQ_LEN, seed=DATA_SEED)
        seqs: NDArray = ds.sequences
        labels: NDArray = ds.labels
        train_idx, val_idx, test_idx = three_way_split(len(seqs), seed=SPLIT_SEED)
        out_task = OUT_DIR / task
        to_parquet(seqs[train_idx], labels[train_idx], out_task / "train.parquet")
        to_parquet(seqs[val_idx], labels[val_idx], out_task / "val.parquet")
        to_parquet(seqs[test_idx], labels[test_idx], out_task / "test.parquet")
        print(
            f"   train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} -> {out_task}"
        )

    # Copy Croissant metadata next to the dataset.
    shutil.copy("CROISSANT.json", OUT_DIR / "CROISSANT.json")
    print(f"\n=> outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
