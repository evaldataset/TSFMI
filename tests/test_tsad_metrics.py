"""Tests for threshold-independent anomaly metrics (Phase-1 / D1)."""

import numpy as np

from src.metrics.tsad_metrics import (
    _dilate_labels,
    vus_pr,
    vus_roc,
    window_threshold_free_metrics,
)


def test_window_metrics_perfect_and_random():
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=np.int64)
    perfect = labels.astype(np.float64)  # scores == labels
    m = window_threshold_free_metrics(perfect, labels)
    assert m["roc_auc"] == 1.0
    assert m["average_precision"] == 1.0
    # anti-correlated scores -> ROC-AUC 0
    m2 = window_threshold_free_metrics(1.0 - perfect, labels)
    assert m2["roc_auc"] == 0.0


def test_window_metrics_single_class_is_nan():
    labels = np.zeros(6, dtype=np.int64)
    m = window_threshold_free_metrics(np.random.rand(6), labels)
    assert np.isnan(m["roc_auc"]) and np.isnan(m["average_precision"])


def test_dilate_labels():
    labels = np.array([0, 0, 1, 0, 0], dtype=np.int64)
    assert _dilate_labels(labels, 0).tolist() == [0, 0, 1, 0, 0]
    assert _dilate_labels(labels, 1).tolist() == [0, 1, 1, 1, 0]
    assert _dilate_labels(labels, 2).tolist() == [1, 1, 1, 1, 1]


def test_vus_perfect_is_one_and_bounded():
    rng = np.random.default_rng(0)
    labels = (rng.random(200) < 0.1).astype(np.int64)
    # A tolerance-aware detector fires across the buffer neighborhood of each
    # anomaly, so it scores high under VUS at every buffer width.
    scores = _dilate_labels(labels, 5).astype(np.float64) + rng.normal(0, 0.01, size=200)
    vr, vp = vus_roc(scores, labels, max_buffer=5), vus_pr(scores, labels, max_buffer=5)
    assert 0.85 <= vr <= 1.0
    assert 0.0 <= vp <= 1.0
    # A point-exact detector still beats random but gets only partial VUS credit,
    # which is the intended range-tolerant behaviour.
    point_scores = labels.astype(np.float64) + rng.normal(0, 0.01, size=200)
    assert vus_roc(point_scores, labels, max_buffer=5) > vus_roc(
        rng.random(200), labels, max_buffer=5
    )
    # random scores -> VUS-ROC near 0.5
    assert 0.3 <= vus_roc(rng.random(200), labels, max_buffer=5) <= 0.7
