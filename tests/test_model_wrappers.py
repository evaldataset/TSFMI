"""Smoke tests for inline model wrappers."""

from __future__ import annotations

import torch

from src.models.autoformer_wrapper import AutoformerWrapper
from src.models.fedformer_wrapper import FEDformerWrapper
from src.models.timesnet_wrapper import TimesNetWrapper


def test_autoformer_wrapper_smoke() -> None:
    """Verify Autoformer wrapper loads, runs forward, and freezes on CPU.

    Args:
        None.
    """
    torch.manual_seed(42)
    wrapper = AutoformerWrapper()
    wrapper.load(
        checkpoint="",
        device=torch.device("cpu"),
        seq_len=32,
        d_model=16,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
    )

    x = torch.randn(4, 32)
    out = wrapper.forward(x)

    assert out.shape == (4, 32, 16)
    assert len(wrapper.get_layer_names()) > 0
    wrapper.freeze()
    assert wrapper.is_frozen() is True


def test_timesnet_wrapper_smoke() -> None:
    """Verify TimesNet wrapper loads, runs forward, and freezes on CPU.

    Args:
        None.
    """
    torch.manual_seed(42)
    wrapper = TimesNetWrapper()
    wrapper.load(
        checkpoint="",
        device=torch.device("cpu"),
        num_variates=1,
        seq_len=32,
        d_model=16,
        e_layers=2,
        top_k=2,
        dropout=0.0,
    )

    x = torch.randn(4, 32)
    out = wrapper.forward(x)

    assert out.shape == (4, 32, 16)
    assert len(wrapper.get_layer_names()) > 0
    wrapper.freeze()
    assert wrapper.is_frozen() is True


def test_fedformer_wrapper_smoke() -> None:
    """Verify FEDformer wrapper loads, runs forward, and freezes on CPU.

    Args:
        None.
    """
    torch.manual_seed(42)
    wrapper = FEDformerWrapper()
    wrapper.load(
        checkpoint="",
        device=torch.device("cpu"),
        seq_len=32,
        d_model=16,
        n_heads=4,
        e_layers=2,
        dropout=0.0,
        moving_avg=5,
    )

    x = torch.randn(4, 32)
    out = wrapper.forward(x)

    assert out.shape == (4, 32, 16)
    assert len(wrapper.get_layer_names()) > 0
    wrapper.freeze()
    assert wrapper.is_frozen() is True
