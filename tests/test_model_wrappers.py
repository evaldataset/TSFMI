"""Smoke tests for inline model wrappers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.autoformer_wrapper import AutoformerWrapper
from src.models.fedformer_wrapper import FEDformerWrapper
from src.models.patchtst_wrapper import (
    DEFAULT_PATCHTST_CHECKPOINT,
    LEGACY_ETTH1_CHECKPOINT,
    PatchTSTWrapper,
)
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


def test_patchtst_default_checkpoint_is_patchtst_fm() -> None:
    """The default PatchTST checkpoint is the ETT-free PatchTST-FM-r1 foundation model."""
    assert DEFAULT_PATCHTST_CHECKPOINT == "ibm-granite/granite-timeseries-patchtst-fm-r1"
    assert LEGACY_ETTH1_CHECKPOINT == "namctin/patchtst_etth1_forecast"


def test_patchtst_random_init_smoke() -> None:
    """Random-init HF PatchTST (checkpoint="") still loads and exposes encoder layers."""
    torch.manual_seed(42)
    wrapper = PatchTSTWrapper()
    wrapper.load(checkpoint="", device=torch.device("cpu"), seq_len=64)

    out = wrapper.forward(torch.randn(2, 64))

    assert out.shape == (2, 24, 1)
    assert wrapper.is_patchtst_fm is False
    assert wrapper.get_layer_names() == [f"model.encoder.layers.{i}" for i in range(3)]
    assert wrapper.is_frozen() is True


def test_patchtst_fm_local_tiny_checkpoint_shapes(tmp_path: Path) -> None:
    """PatchTST-FM path: tiny random checkpoint saved locally, loaded via load().

    Hooked block activations must cover only the observed input patches.

    Args:
        tmp_path: Pytest temporary directory for the tiny checkpoint.
    """
    pytest.importorskip("tsfm_public")
    from tsfm_public.models.patchtst_fm import PatchTSTFMConfig, PatchTSTFMForPrediction

    from src.extractors.hook_manager import HookManager

    torch.manual_seed(0)
    config = PatchTSTFMConfig(
        context_length=256, d_patch=16, d_model=32, n_head=4, n_layer=2, num_quantile=9
    )
    PatchTSTFMForPrediction(config).save_pretrained(tmp_path)

    wrapper = PatchTSTWrapper()
    wrapper.load(str(tmp_path), device=torch.device("cpu"))
    names = wrapper.get_layer_names()
    assert wrapper.is_patchtst_fm is True
    assert names == ["backbone.blocks.0", "backbone.blocks.1"]

    with HookManager(wrapper.model, names) as hm:
        out = wrapper.forward(torch.randn(3, 100))
        acts = hm.get_activations()

    assert out.shape == (3, 112, 9)  # ceil(100 / 16) * 16 steps x num_quantile
    for name in names:
        assert acts[name].shape == (3, 7, 32)  # ceil(100 / 16) context patches
    assert wrapper.is_frozen() is True

    with pytest.raises(ValueError, match="univariate"):
        wrapper.forward(torch.randn(2, 100, 3))


@pytest.mark.slow
def test_patchtst_fm_r1_pretrained_shapes() -> None:
    """Download PatchTST-FM-r1 and check architecture facts and hooked shapes."""
    pytest.importorskip("tsfm_public")
    from src.extractors.hook_manager import HookManager

    wrapper = PatchTSTWrapper()
    wrapper.load(device=torch.device("cpu"))
    names = wrapper.get_layer_names()

    assert names == [f"backbone.blocks.{i}" for i in range(20)]
    assert sum(p.numel() for p in wrapper.model.parameters()) == 257_895_552

    with HookManager(wrapper.model, names[:1] + names[-1:]) as hm:
        out = wrapper.forward(torch.randn(2, 512))
        acts = hm.get_activations()

    assert out.shape == (2, 512, 99)
    assert acts[names[0]].shape == (2, 32, 1024)
    assert acts[names[-1]].shape == (2, 32, 1024)
