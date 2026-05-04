"""Tests for causal controls: necessity, sufficiency, random subspace."""

from __future__ import annotations

import numpy as np

from scripts.run_causal_controls import (
    erase_random_subspace,
    extract_concept_subspace,
    project_onto_subspace,
    run_causal_analysis,
)


class TestExtractConceptSubspace:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 16)).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        subspace = extract_concept_subspace(X, y, n_directions=2)
        assert subspace.shape == (2, 16)

    def test_orthonormal(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 16)).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20 + [2] * 20)
        subspace = extract_concept_subspace(X, y, n_directions=2)
        # Rows should be approximately orthonormal
        gram = subspace @ subspace.T
        np.testing.assert_allclose(gram, np.eye(2), atol=0.1)


class TestProjectOntoSubspace:
    def test_output_shape(self) -> None:
        X = np.random.default_rng(42).standard_normal((20, 8)).astype(np.float32)
        subspace = np.eye(2, 8, dtype=np.float32)  # first 2 dims
        projected = project_onto_subspace(X, subspace)
        assert projected.shape == X.shape

    def test_idempotent(self) -> None:
        X = np.random.default_rng(42).standard_normal((20, 8)).astype(np.float32)
        subspace = np.eye(2, 8, dtype=np.float32)
        p1 = project_onto_subspace(X, subspace)
        p2 = project_onto_subspace(p1, subspace)
        np.testing.assert_allclose(p1, p2, atol=1e-6)


class TestEraseRandomSubspace:
    def test_output_shape(self) -> None:
        X = np.random.default_rng(42).standard_normal((20, 8)).astype(np.float32)
        erased = erase_random_subspace(X, n_directions=2, seed=42)
        assert erased.shape == X.shape

    def test_reduces_variance(self) -> None:
        X = np.random.default_rng(42).standard_normal((50, 16)).astype(np.float32)
        erased = erase_random_subspace(X, n_directions=4, seed=42)
        # Erasing directions should reduce total variance
        assert np.var(erased) <= np.var(X) + 1e-6

    def test_deterministic(self) -> None:
        X = np.random.default_rng(42).standard_normal((20, 8)).astype(np.float32)
        e1 = erase_random_subspace(X, n_directions=2, seed=0)
        e2 = erase_random_subspace(X, n_directions=2, seed=0)
        np.testing.assert_array_equal(e1, e2)


class TestRunCausalAnalysis:
    def test_returns_all_keys(self) -> None:
        rng = np.random.default_rng(42)
        n, d = 80, 16
        # Separable 2-class data
        X = np.concatenate([
            rng.standard_normal((n // 2, d)).astype(np.float32) + 3,
            rng.standard_normal((n // 2, d)).astype(np.float32) - 3,
        ])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        result = run_causal_analysis(X, y, val_split=0.2, seed=42, n_random_seeds=3)
        expected_keys = {
            "original_acc", "leace_erased_acc", "subspace_only_acc",
            "random_erased_acc_mean", "random_erased_acc_std",
            "n_concept_directions", "necessity_drop",
            "sufficiency_preserved", "random_drop", "specificity",
        }
        assert set(result.keys()) == expected_keys

    def test_separable_data_high_original(self) -> None:
        rng = np.random.default_rng(42)
        n, d = 80, 16
        X = np.concatenate([
            rng.standard_normal((n // 2, d)).astype(np.float32) + 5,
            rng.standard_normal((n // 2, d)).astype(np.float32) - 5,
        ])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        result = run_causal_analysis(X, y, val_split=0.2, seed=42, n_random_seeds=3)
        assert result["original_acc"] > 0.8
        # LEACE should cause larger drop than random erasure (specificity > 0)
        assert result["necessity_drop"] >= result["random_drop"]
