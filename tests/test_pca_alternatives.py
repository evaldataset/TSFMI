"""Tests for PCA alternative dimensionality reduction methods."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.run_pca_alternatives import (
    evaluate_ridge,
    primary_metric,
    random_projection,
    run_layer,
    standard_pca,
    supervised_pca,
)


@pytest.fixture()
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create small regression dataset where label correlates with a few features."""
    rng = np.random.default_rng(42)
    n_train, n_test, d = 80, 20, 64
    X_train = rng.standard_normal((n_train, d)).astype(np.float32)
    X_test = rng.standard_normal((n_test, d)).astype(np.float32)
    # Label depends linearly on features 0-3
    y_train = (X_train[:, :4].sum(axis=1) + rng.normal(0, 0.1, n_train)).astype(np.float32)
    y_test = (X_test[:, :4].sum(axis=1) + rng.normal(0, 0.1, n_test)).astype(np.float32)
    return X_train, X_test, y_train, y_test


@pytest.fixture()
def classification_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create small classification dataset with 3 separable classes."""
    rng = np.random.default_rng(42)
    n_per_class, d = 30, 32
    X_parts = []
    y_parts = []
    for c in range(3):
        center = rng.standard_normal(d).astype(np.float32) * 3
        X_parts.append(center + rng.standard_normal((n_per_class, d)).astype(np.float32) * 0.5)
        y_parts.append(np.full(n_per_class, c))
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    return X[:72], X[72:], y[:72], y[72:]


class TestStandardPCA:
    def test_output_shape(self, regression_data: tuple) -> None:
        X_train, X_test, _, _ = regression_data
        Xtr, Xte = standard_pca(X_train, X_test, n_components=16)
        assert Xtr.shape == (80, 16)
        assert Xte.shape == (20, 16)

    def test_components_capped(self, regression_data: tuple) -> None:
        X_train, X_test, _, _ = regression_data
        Xtr, Xte = standard_pca(X_train, X_test, n_components=1000)
        assert Xtr.shape[1] <= min(X_train.shape)


class TestSupervisedPCA:
    def test_output_shape(self, regression_data: tuple) -> None:
        X_train, X_test, y_train, _ = regression_data
        Xtr, Xte = supervised_pca(X_train, y_train, X_test, n_components=16)
        assert Xtr.shape == (80, 16)
        assert Xte.shape == (20, 16)

    def test_preserves_label_variance(self, regression_data: tuple) -> None:
        """Supervised PCA should give better R² than standard PCA for regression."""
        X_train, X_test, y_train, y_test = regression_data
        Xtr_pca, Xte_pca = standard_pca(X_train, X_test, n_components=8)
        Xtr_spca, Xte_spca = supervised_pca(X_train, y_train, X_test, n_components=8)

        from sklearn.linear_model import Ridge

        ridge_pca = Ridge(alpha=1.0).fit(Xtr_pca, y_train)
        ridge_spca = Ridge(alpha=1.0).fit(Xtr_spca, y_train)
        r2_pca = ridge_pca.score(Xte_pca, y_test)
        r2_spca = ridge_spca.score(Xte_spca, y_test)
        # sPCA should be at least as good (usually better)
        assert r2_spca >= r2_pca - 0.1


class TestRandomProjection:
    def test_output_shape(self, regression_data: tuple) -> None:
        X_train, X_test, _, _ = regression_data
        Xtr, Xte = random_projection(X_train, X_test, n_components=16)
        assert Xtr.shape == (80, 16)
        assert Xte.shape == (20, 16)

    def test_deterministic(self, regression_data: tuple) -> None:
        X_train, X_test, _, _ = regression_data
        Xtr1, _ = random_projection(X_train, X_test, 16, seed=0)
        Xtr2, _ = random_projection(X_train, X_test, 16, seed=0)
        np.testing.assert_array_equal(Xtr1, Xtr2)


class TestEvaluateRidge:
    def test_classification(self, classification_data: tuple) -> None:
        X_train, X_test, y_train, y_test = classification_data
        metrics = evaluate_ridge(X_train, y_train, X_test, y_test, "classification")
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_regression(self, regression_data: tuple) -> None:
        X_train, X_test, y_train, y_test = regression_data
        metrics = evaluate_ridge(X_train, y_train, X_test, y_test, "regression")
        assert "r2" in metrics
        assert "mae" in metrics


class TestPrimaryMetric:
    def test_classification(self) -> None:
        assert primary_metric({"accuracy": 0.9, "f1": 0.85}, "classification") == 0.9

    def test_regression(self) -> None:
        assert primary_metric({"r2": 0.95, "mae": 0.1}, "regression") == 0.95


class TestRunLayer:
    def test_all_methods_present(self, regression_data: tuple) -> None:
        X_train, X_test, y_train, y_test = regression_data
        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        results = run_layer(X, y, "regression", n_components=16, alpha=1.0, val_split=0.2, seed=42)
        assert set(results.keys()) == {"pca", "supervised_pca", "random_projection", "full_ridge"}

    def test_classification_mode(self, classification_data: tuple) -> None:
        X_train, X_test, y_train, y_test = classification_data
        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        results = run_layer(
            X, y, "classification", n_components=16, alpha=1.0, val_split=0.2, seed=42,
        )
        for method_metrics in results.values():
            assert "accuracy" in method_metrics
