"""Tests for HookManager intermediate activation extraction."""

import pytest
import torch
import torch.nn as nn

from src.extractors.hook_manager import HookManager


class TinyModel(nn.Module):
    """Minimal 2-layer model for testing hooks."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(16, 32)
        self.layer2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    return TinyModel()


def test_hookmanager_captures_activations(tiny_model):
    x = torch.randn(4, 16)
    with HookManager(tiny_model, layer_names=["layer1", "layer2"]) as hm:
        _ = tiny_model(x)
        acts = hm.get_activations()

    assert "layer1" in acts
    assert "layer2" in acts
    assert acts["layer1"].shape == (4, 32)
    assert acts["layer2"].shape == (4, 8)


def test_hookmanager_detaches_tensors(tiny_model):
    x = torch.randn(4, 16)
    with HookManager(tiny_model, layer_names=["layer1"]) as hm:
        _ = tiny_model(x)
        acts = hm.get_activations()
    assert not acts["layer1"].requires_grad


def test_hookmanager_removes_hooks(tiny_model):
    x = torch.randn(4, 16)
    with HookManager(tiny_model, layer_names=["layer1"]) as hm:
        _ = tiny_model(x)
    # After context exit, handles should be cleared
    assert len(hm._handles) == 0


def test_hookmanager_raises_outside_context(tiny_model):
    hm = HookManager(tiny_model, layer_names=["layer1"])
    with pytest.raises(RuntimeError, match="context manager"):
        hm.get_activations()


def test_hookmanager_raises_on_invalid_layer(tiny_model):
    with pytest.raises(ValueError):
        HookManager(tiny_model, layer_names=["nonexistent_layer"])


def test_hookmanager_clear(tiny_model):
    x = torch.randn(4, 16)
    with HookManager(tiny_model, layer_names=["layer1"]) as hm:
        _ = tiny_model(x)
        hm.clear()
        assert len(hm.get_activations()) == 0
        # Run again after clear — should capture new activations
        _ = tiny_model(x)
        acts = hm.get_activations()
    assert "layer1" in acts


def test_hookmanager_single_layer(tiny_model):
    x = torch.randn(4, 16)
    with HookManager(tiny_model, layer_names=["layer2"]) as hm:
        _ = tiny_model(x)
        acts = hm.get_activations()
    assert len(acts) == 1
    assert acts["layer2"].shape == (4, 8)


def test_hookmanager_multiple_forward_passes(tiny_model):
    """Later forward pass overwrites activations for same layer."""
    x1 = torch.randn(4, 16)
    x2 = torch.randn(8, 16)
    with HookManager(tiny_model, layer_names=["layer1"]) as hm:
        _ = tiny_model(x1)
        _ = tiny_model(x2)
        acts = hm.get_activations()
    # Second pass had batch=8, should overwrite
    assert acts["layer1"].shape == (8, 32)
