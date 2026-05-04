"""Tests for LinearProbe and MLPControlProbe."""

import pytest
import torch

from src.probes.linear_probe import LinearProbe
from src.probes.mlp_control_probe import MLPControlProbe


class TestLinearProbe:
    def test_output_shape_binary(self, synthetic_representations):
        probe = LinearProbe(input_dim=768, output_dim=2)
        out = probe(synthetic_representations)
        assert out.shape == (100, 2)

    def test_output_shape_multiclass(self, synthetic_representations):
        probe = LinearProbe(input_dim=768, output_dim=5)
        out = probe(synthetic_representations)
        assert out.shape == (100, 5)

    def test_output_shape_regression(self, synthetic_representations):
        probe = LinearProbe(input_dim=768, output_dim=1)
        out = probe(synthetic_representations)
        assert out.shape == (100, 1)

    def test_raises_on_wrong_ndim(self, synthetic_representations):
        probe = LinearProbe(input_dim=768, output_dim=2)
        with pytest.raises(ValueError):
            probe(synthetic_representations.unsqueeze(0))  # 3D input

    def test_raises_on_wrong_dim(self, synthetic_representations):
        probe = LinearProbe(input_dim=512, output_dim=2)  # wrong dim
        with pytest.raises(ValueError):
            probe(synthetic_representations)  # 768 != 512

    def test_gradients_flow(self, tiny_representations, tiny_labels):
        """Verify probe parameters get gradients during backward."""
        probe = LinearProbe(input_dim=64, output_dim=2)
        out = probe(tiny_representations)
        loss = torch.nn.CrossEntropyLoss()(out, tiny_labels)
        loss.backward()
        assert probe.linear.weight.grad is not None

    def test_no_bias_option(self, synthetic_representations):
        probe = LinearProbe(input_dim=768, output_dim=3, bias=False)
        assert probe.linear.bias is None
        out = probe(synthetic_representations)
        assert out.shape == (100, 3)


class TestMLPControlProbe:
    def test_output_shape(self, synthetic_representations):
        probe = MLPControlProbe(input_dim=768, output_dim=3)
        out = probe(synthetic_representations)
        assert out.shape == (100, 3)

    def test_custom_hidden_dim(self, synthetic_representations):
        probe = MLPControlProbe(input_dim=768, output_dim=3, hidden_dim=256)
        out = probe(synthetic_representations)
        assert out.shape == (100, 3)

    def test_default_hidden_dim(self):
        """Default hidden_dim = max(input_dim // 2, 64)."""
        probe = MLPControlProbe(input_dim=768, output_dim=3)
        # hidden should be 384 = max(768 // 2, 64)
        assert probe.net[0].out_features == 384

    def test_default_hidden_dim_small_input(self):
        """For small input_dim, hidden_dim floor is 64."""
        probe = MLPControlProbe(input_dim=32, output_dim=2)
        # hidden should be max(16, 64) = 64
        assert probe.net[0].out_features == 64

    def test_raises_on_wrong_ndim(self, synthetic_representations):
        probe = MLPControlProbe(input_dim=768, output_dim=2)
        with pytest.raises(ValueError):
            probe(synthetic_representations.unsqueeze(0))

    def test_raises_on_wrong_dim(self, synthetic_representations):
        probe = MLPControlProbe(input_dim=512, output_dim=2)
        with pytest.raises(ValueError):
            probe(synthetic_representations)

    def test_gradients_flow(self, tiny_representations, tiny_labels):
        """Verify MLP probe parameters get gradients during backward."""
        probe = MLPControlProbe(input_dim=64, output_dim=2)
        out = probe(tiny_representations)
        loss = torch.nn.CrossEntropyLoss()(out, tiny_labels)
        loss.backward()
        # First linear layer should have gradients
        assert probe.net[0].weight.grad is not None

    def test_dropout_option(self, synthetic_representations):
        probe = MLPControlProbe(input_dim=768, output_dim=3, dropout=0.5)
        out = probe(synthetic_representations)
        assert out.shape == (100, 3)
