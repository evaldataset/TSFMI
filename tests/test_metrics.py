"""Tests for probing metric functions."""

import pytest
import torch

from src.metrics.probing_metrics import (
    compute_cka,
    compute_classification_metrics,
    compute_regression_metrics,
    compute_selectivity,
)

# --- Classification metrics ---


def test_classification_perfect_accuracy():
    """Perfect classifier should have accuracy=1.0."""
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0]])
    labels = torch.tensor([0, 1, 0])
    metrics = compute_classification_metrics(logits, labels)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_classification_zero_accuracy():
    """Worst classifier should have accuracy=0.0."""
    logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])  # predicts 1, 0
    labels = torch.tensor([0, 1])
    metrics = compute_classification_metrics(logits, labels)
    assert metrics["accuracy"] == pytest.approx(0.0)


def test_classification_keys():
    torch.manual_seed(42)
    logits = torch.randn(20, 3)
    labels = torch.randint(0, 3, (20,))
    metrics = compute_classification_metrics(logits, labels)
    assert set(metrics.keys()) == {"accuracy", "f1_macro", "f1_weighted"}
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_classification_multiclass():
    """5-class classification should work."""
    torch.manual_seed(42)
    logits = torch.randn(50, 5)
    labels = torch.randint(0, 5, (50,))
    metrics = compute_classification_metrics(logits, labels)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_classification_binary_1d_logits():
    """Binary classification with 1D logits (sigmoid-style)."""
    logits = torch.tensor([5.0, -5.0, 3.0])  # predicts 1, 0, 1
    labels = torch.tensor([1, 0, 1])
    metrics = compute_classification_metrics(logits, labels)
    assert metrics["accuracy"] == pytest.approx(1.0)


# --- Regression metrics ---


def test_regression_perfect_r2():
    """Perfect predictor should have r2=1.0."""
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    preds = targets.clone()
    metrics = compute_regression_metrics(preds, targets)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-5)


def test_regression_keys():
    torch.manual_seed(42)
    preds = torch.randn(30)
    targets = torch.randn(30)
    metrics = compute_regression_metrics(preds, targets)
    assert set(metrics.keys()) == {"r2", "mae", "mse"}


def test_regression_handles_N1_shape():
    """Probe output (N, 1) should work with targets (N,)."""
    torch.manual_seed(42)
    preds = torch.randn(20, 1)
    targets = torch.randn(20)
    metrics = compute_regression_metrics(preds, targets)
    assert "r2" in metrics


def test_regression_mae_known_values():
    preds = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([2.0, 2.0, 5.0])
    metrics = compute_regression_metrics(preds, targets)
    # MAE = (1 + 0 + 2) / 3 = 1.0
    assert metrics["mae"] == pytest.approx(1.0)
    # MSE = (1 + 0 + 4) / 3 = 5/3
    assert metrics["mse"] == pytest.approx(5.0 / 3.0, abs=1e-5)


def test_regression_nonnegative_mae_mse():
    torch.manual_seed(42)
    preds = torch.randn(50)
    targets = torch.randn(50)
    metrics = compute_regression_metrics(preds, targets)
    assert metrics["mae"] >= 0.0
    assert metrics["mse"] >= 0.0


# --- Selectivity ---


def test_selectivity_positive():
    assert compute_selectivity(0.8, 0.6) == pytest.approx(0.2)


def test_selectivity_negative():
    assert compute_selectivity(0.5, 0.7) == pytest.approx(-0.2)


def test_selectivity_zero():
    assert compute_selectivity(0.7, 0.7) == pytest.approx(0.0)


def test_selectivity_extreme():
    assert compute_selectivity(1.0, 0.0) == pytest.approx(1.0)
    assert compute_selectivity(0.0, 1.0) == pytest.approx(-1.0)


# --- CKA ---


def test_cka_identical_matrices():
    """CKA of a matrix with itself should be 1.0."""
    torch.manual_seed(42)
    X = torch.randn(50, 64)
    assert compute_cka(X, X) == pytest.approx(1.0, abs=1e-4)


def test_cka_range():
    torch.manual_seed(42)
    X = torch.randn(50, 64)
    Y = torch.randn(50, 32)
    cka = compute_cka(X, Y)
    assert 0.0 <= cka <= 1.0


def test_cka_different_feature_dims():
    """CKA should work when d1 != d2."""
    torch.manual_seed(42)
    X = torch.randn(100, 768)
    Y = torch.randn(100, 256)
    cka = compute_cka(X, Y)
    assert 0.0 <= cka <= 1.0


def test_cka_raises_on_mismatched_samples():
    X = torch.randn(50, 64)
    Y = torch.randn(40, 64)
    with pytest.raises(ValueError):
        compute_cka(X, Y)


def test_cka_symmetric():
    """CKA(X, Y) should approximately equal CKA(Y, X)."""
    torch.manual_seed(42)
    X = torch.randn(50, 64)
    Y = torch.randn(50, 32)
    assert compute_cka(X, Y) == pytest.approx(compute_cka(Y, X), abs=1e-5)
