"""Real-world anomaly-style probing tasks from the UCR archive (Phase-1 / D1).

Frames binary UCR classification datasets as window-level anomaly detection: the
minority (abnormal/rare) class is the positive "anomaly" label. Uses the official
UCR train/test split, which is leak-free by construction. Data is fetched on demand
via ``aeon`` and cached under ``extract_path`` (default: the tsfmi_store on the large
partition).

This provides the external-validity counterpart to the synthetic anomaly task: the
same baseline-controlled probing protocol can be run on real signals whose anomalies
are not a single-spike Gaussian construction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Datasets whose minority class is a genuine anomaly (abnormal / rare event).
ANOMALY_DATASETS: tuple[str, ...] = ("ECG200", "Wafer", "Earthquakes")

# Cache dir for the downloaded UCR archive. Override with the TSFMI_UCR_CACHE env var;
# defaults to a repo-relative path so the release contains no machine-specific paths.
DEFAULT_EXTRACT_PATH = os.environ.get("TSFMI_UCR_CACHE", "data/ucr_cache")


@dataclass
class RealAnomalyTask:
    """A real-world anomaly probing task with the official leak-free split."""

    name: str
    train_sequences: NDArray[np.float64]  # (N_tr, L)
    train_labels: NDArray[np.int64]  # (N_tr,) 1 = anomaly
    test_sequences: NDArray[np.float64]  # (N_te, L)
    test_labels: NDArray[np.int64]  # (N_te,)
    anomaly_class: str
    anomaly_rate: float


def _to_2d(x: NDArray) -> NDArray[np.float64]:
    """aeon returns (N, 1, L) for univariate; squeeze the channel axis to (N, L)."""
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[:, 0, :]
    return x.astype(np.float64)


def load_ucr_anomaly(
    name: str,
    *,
    extract_path: str = DEFAULT_EXTRACT_PATH,
    anomaly_class: str | None = None,
) -> RealAnomalyTask:
    """Load a UCR binary dataset as a real anomaly probing task.

    Args:
        name: UCR dataset name (e.g. "ECG200").
        extract_path: cache directory for the downloaded archive.
        anomaly_class: label string to treat as the anomaly; if None, the minority
            class over the official training split is used.

    Returns:
        A :class:`RealAnomalyTask` with the official train/test split and binary
        labels (1 = anomaly).
    """
    from aeon.datasets import load_classification

    Xtr, ytr = load_classification(name, split="train", extract_path=extract_path)
    Xte, yte = load_classification(name, split="test", extract_path=extract_path)
    Xtr, Xte = _to_2d(Xtr), _to_2d(Xte)
    ytr, yte = np.asarray(ytr).astype(str), np.asarray(yte).astype(str)

    if anomaly_class is None:
        vals, counts = np.unique(ytr, return_counts=True)
        anomaly_class = str(vals[int(np.argmin(counts))])

    ytr_bin = (ytr == anomaly_class).astype(np.int64)
    yte_bin = (yte == anomaly_class).astype(np.int64)
    rate = float((ytr_bin.sum() + yte_bin.sum()) / (len(ytr_bin) + len(yte_bin)))
    return RealAnomalyTask(
        name=name,
        train_sequences=Xtr,
        train_labels=ytr_bin,
        test_sequences=Xte,
        test_labels=yte_bin,
        anomaly_class=anomaly_class,
        anomaly_rate=rate,
    )
