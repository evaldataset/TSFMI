"""Tests for nonlinear recovery analysis."""

from __future__ import annotations

import pytest
import torch

from scripts.run_nonlinear_recovery_all import (
    MLPProbe,
    erase_representations,
    train_probe_on_data,
)


class TestMLPProbe:
    def test_output_shape(self) -> None:
        probe = MLPProbe(input_dim=32, hidden_dim=16, output_dim=3)
        x = torch.randn(10, 32)
        out = probe(x)
        assert out.shape == (10, 3)

    def test_rejects_3d_input(self) -> None:
        probe = MLPProbe(input_dim=32, hidden_dim=16, output_dim=3)
        x = torch.randn(10, 5, 32)
        with pytest.raises(ValueError):
            probe(x)


class TestTrainProbeOnData:
    def test_linear_returns_accuracy(self) -> None:
        torch.manual_seed(42)
        n, d, c = 100, 16, 3
        x = torch.cat([torch.randn(n // c, d) + i * 3 for i in range(c)])
        y = torch.cat([torch.full((n // c,), i, dtype=torch.long) for i in range(c)])
        acc = train_probe_on_data(
            x[:80], y[:80], x[80:], y[80:], d, c,
            use_mlp=False, num_epochs=50,
        )
        assert 0.0 <= acc <= 1.0
        assert acc > 0.4

    def test_mlp_returns_accuracy(self) -> None:
        torch.manual_seed(42)
        n, d, c = 100, 16, 3
        x = torch.cat([torch.randn(n // c, d) + i * 3 for i in range(c)])
        y = torch.cat([torch.full((n // c,), i, dtype=torch.long) for i in range(c)])
        acc = train_probe_on_data(
            x[:80], y[:80], x[80:], y[80:], d, c,
            use_mlp=True, num_epochs=50,
        )
        assert 0.0 <= acc <= 1.0


class TestEraseRepresentations:
    def test_output_shape_preserved(self) -> None:
        torch.manual_seed(42)
        n_train, n_test, d = 40, 10, 16
        x_train = torch.randn(n_train, d)
        x_test = torch.randn(n_test, d)
        y_train = torch.randint(0, 2, (n_train,))
        erased_train, erased_test = erase_representations(x_train, y_train, x_test)
        assert erased_train.shape == (n_train, d)
        assert erased_test.shape == (n_test, d)

    def test_rejects_3d(self) -> None:
        x = torch.randn(10, 5, 8)
        y = torch.randint(0, 2, (10,))
        with pytest.raises(ValueError):
            erase_representations(x, y, x)

    def test_erased_differs_from_original(self) -> None:
        torch.manual_seed(42)
        n, d = 100, 16
        x = torch.randn(n, d)
        y = (x[:, 0] > 0).long()
        erased_train, _ = erase_representations(x[:80], y[:80], x[80:])
        assert not torch.allclose(x[:80], erased_train, atol=1e-6)
