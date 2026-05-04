"""Tests for hard variant benchmark script."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_hard_variant_benchmark import (
    probe_best_layer,
)


@pytest.fixture()
def tmp_repr_dir(tmp_path: Path) -> Path:
    """Create a temp directory with fake layer representations and labels."""
    n_samples = 60
    hidden_dim = 32
    n_classes = 3
    rng = np.random.default_rng(42)

    # Create separable data
    X_parts = []
    y_parts = []
    for c in range(n_classes):
        center = rng.standard_normal(hidden_dim).astype(np.float32) * 5
        noise = rng.standard_normal((n_samples // n_classes, hidden_dim)).astype(np.float32)
        X_parts.append(center + noise)
        y_parts.append(np.full(n_samples // n_classes, c, dtype=np.int64))
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)

    # Save 2 layers
    torch.save(torch.tensor(X, dtype=torch.float32), tmp_path / "encoder_block_0.pt")
    X_noisy = X + rng.standard_normal(X.shape).astype(np.float32) * 0.1
    torch.save(
        torch.tensor(X_noisy, dtype=torch.float32),
        tmp_path / "encoder_block_1.pt",
    )
    torch.save(torch.tensor(y, dtype=torch.long), tmp_path / "labels.pt")

    return tmp_path


class TestProbeBestLayer:
    def test_returns_metrics(self, tmp_repr_dir: Path) -> None:
        result = probe_best_layer(tmp_repr_dir, alpha=1.0, val_split=0.2, seed=42)
        assert "accuracy" in result
        assert "f1" in result
        assert "best_layer" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_accuracy_above_chance(self, tmp_repr_dir: Path) -> None:
        result = probe_best_layer(tmp_repr_dir, alpha=1.0, val_split=0.2, seed=42)
        # 3-class well-separated data (centers at ±5σ) should achieve near-perfect
        assert result["accuracy"] > 0.8

    def test_missing_labels(self, tmp_path: Path) -> None:
        # No labels.pt → empty result
        torch.save(torch.randn(10, 8), tmp_path / "layer_0.pt")
        result = probe_best_layer(tmp_path, alpha=1.0, val_split=0.2, seed=42)
        assert result == {}

    def test_empty_dir(self, tmp_path: Path) -> None:
        # No layer files, just labels
        torch.save(torch.randint(0, 3, (30,)), tmp_path / "labels.pt")
        result = probe_best_layer(tmp_path, alpha=1.0, val_split=0.2, seed=42)
        assert result == {}

    def test_deterministic(self, tmp_repr_dir: Path) -> None:
        r1 = probe_best_layer(tmp_repr_dir, alpha=1.0, val_split=0.2, seed=42)
        r2 = probe_best_layer(tmp_repr_dir, alpha=1.0, val_split=0.2, seed=42)
        assert r1["accuracy"] == r2["accuracy"]
        assert r1["best_layer"] == r2["best_layer"]
