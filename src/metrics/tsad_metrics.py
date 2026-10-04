"""Threshold-independent metrics for anomaly evaluation (Phase-1 / D1).

Two settings are supported:

1. **Window-level probing** (our protocol): each window is one sample with a binary
   label. Threshold-free comparison uses ROC-AUC and Average Precision (AUPRC).
   Use :func:`window_threshold_free_metrics`.

2. **Point-wise scoring** (TSAD-style, for real-data comparison): a per-timestep
   anomaly score with range tolerance. :func:`vus_roc` / :func:`vus_pr` average the
   AUC over a buffer of tolerances, following the range-tolerant idea of Paparrizos
   et al. (VLDB 2022). This is a self-contained approximation; for exact TSB-UAD
   parity, swap in their reference implementation.

All functions take numpy arrays and return plain floats/dicts so results serialize
directly to the JSON/CSV outputs used elsewhere in the pipeline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score


def window_threshold_free_metrics(
    scores: NDArray[np.float64], labels: NDArray[np.int64]
) -> dict[str, float]:
    """ROC-AUC and Average Precision for window-level binary anomaly probing.

    Args:
        scores: Predicted anomaly scores/probabilities, shape (N,).
        labels: Binary ground truth (1 = anomaly), shape (N,).

    Returns:
        Dict with ``roc_auc`` and ``average_precision``. If only one class is
        present (AUC undefined), the affected entry is ``nan``.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    out: dict[str, float] = {}
    if len(np.unique(labels)) < 2:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
        return out
    out["roc_auc"] = float(roc_auc_score(labels, scores))
    out["average_precision"] = float(average_precision_score(labels, scores))
    return out


def _dilate_labels(labels: NDArray[np.int64], buffer: int) -> NDArray[np.int64]:
    """Binary dilation of positive labels by ``buffer`` on each side.

    Points within ``buffer`` timesteps of any true anomaly become positive. This is
    the existence/range-tolerance used by range-based TSAD metrics.
    """
    if buffer <= 0:
        return labels.astype(np.int64)
    n = len(labels)
    pos = np.flatnonzero(labels > 0)
    dil = np.zeros(n, dtype=np.int64)
    for p in pos:
        lo = max(0, p - buffer)
        hi = min(n, p + buffer + 1)
        dil[lo:hi] = 1
    return dil


def _auc_over_buffers(
    scores: NDArray[np.float64],
    labels: NDArray[np.int64],
    max_buffer: int,
    metric: str,
) -> float:
    """Average ROC-AUC or AP of ``scores`` against labels dilated over buffers 0..L."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    fn = roc_auc_score if metric == "roc" else average_precision_score
    vals: list[float] = []
    for b in range(max_buffer + 1):
        y = _dilate_labels(labels, b)
        if len(np.unique(y)) < 2:
            continue
        vals.append(float(fn(y, scores)))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def vus_roc(
    scores: NDArray[np.float64], labels: NDArray[np.int64], max_buffer: int = 10
) -> float:
    """Volume-Under-Surface ROC (range-tolerant), averaged over buffers 0..max_buffer."""
    return _auc_over_buffers(scores, labels, max_buffer, "roc")


def vus_pr(
    scores: NDArray[np.float64], labels: NDArray[np.int64], max_buffer: int = 10
) -> float:
    """Volume-Under-Surface PR (range-tolerant), averaged over buffers 0..max_buffer."""
    return _auc_over_buffers(scores, labels, max_buffer, "pr")
