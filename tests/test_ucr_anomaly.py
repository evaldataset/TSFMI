"""Tests for the UCR real-anomaly loader (Phase-1 / D1). Marked slow (network fetch)."""

import numpy as np
import pytest

from src.datasets.ucr_anomaly import ANOMALY_DATASETS, load_ucr_anomaly


@pytest.mark.slow
def test_load_ecg200_binary_and_split():
    task = load_ucr_anomaly("ECG200")
    # official split sizes
    assert task.train_sequences.shape[0] == 100
    assert task.test_sequences.shape[0] == 100
    # 2D sequences, binary labels
    assert task.train_sequences.ndim == 2
    assert set(np.unique(task.train_labels)).issubset({0, 1})
    assert set(np.unique(task.test_labels)).issubset({0, 1})
    # anomaly = minority class, so anomaly rate < 0.5
    assert 0.0 < task.anomaly_rate < 0.5
    # train and test are the official disjoint partitions (both non-empty positives)
    assert task.train_labels.sum() > 0 and task.test_labels.sum() > 0


def test_anomaly_dataset_registry():
    assert "ECG200" in ANOMALY_DATASETS
    assert all(isinstance(n, str) for n in ANOMALY_DATASETS)
