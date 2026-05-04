"""Tests for real-world transfer analysis."""

from __future__ import annotations

import numpy as np

from scripts.run_real_world_analysis import (
    bootstrap_label_stability,
)


def _simple_label_fn(window: np.ndarray) -> int:
    """Label by mean: positive → 0, negative → 1."""
    return 0 if window.mean() > 0 else 1


class TestBootstrapLabelStability:
    def test_returns_required_keys(self) -> None:
        rng = np.random.default_rng(42)
        windows = rng.standard_normal((20, 64))
        result = bootstrap_label_stability(
            windows, _simple_label_fn, n_bootstrap=10, noise_std=0.01, seed=42,
        )
        assert "agreement_rate" in result
        assert "flip_rate" in result
        assert "n_windows" in result
        assert "per_class_stability" in result

    def test_agreement_bounded(self) -> None:
        rng = np.random.default_rng(42)
        windows = rng.standard_normal((20, 64))
        result = bootstrap_label_stability(
            windows, _simple_label_fn, n_bootstrap=10, noise_std=0.01, seed=42,
        )
        assert 0.0 <= result["agreement_rate"] <= 1.0
        assert 0.0 <= result["flip_rate"] <= 1.0
        assert abs(result["agreement_rate"] + result["flip_rate"] - 1.0) < 1e-10

    def test_high_stability_for_clear_signal(self) -> None:
        # Windows with very strong signal should be stable
        windows = np.ones((20, 64)) * 10.0  # very positive
        result = bootstrap_label_stability(
            windows, _simple_label_fn, n_bootstrap=50, noise_std=0.01, seed=42,
        )
        assert result["agreement_rate"] > 0.95

    def test_low_stability_for_boundary(self) -> None:
        # Windows near decision boundary should be less stable
        rng = np.random.default_rng(42)
        windows = rng.standard_normal((20, 64)) * 0.001  # near zero mean
        result = bootstrap_label_stability(
            windows, _simple_label_fn, n_bootstrap=50, noise_std=1.0, seed=42,
        )
        # Should be less stable than clear signal
        assert result["agreement_rate"] < 0.95

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(42)
        windows = rng.standard_normal((10, 32))
        r1 = bootstrap_label_stability(
            windows, _simple_label_fn, n_bootstrap=20, noise_std=0.05, seed=0,
        )
        r2 = bootstrap_label_stability(
            windows, _simple_label_fn, n_bootstrap=20, noise_std=0.05, seed=0,
        )
        assert r1["agreement_rate"] == r2["agreement_rate"]
