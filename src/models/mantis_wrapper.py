# pyright: reportMissingImports=false, reportImplicitOverride=false
"""Mantis-8M wrapper for probing experiments (classification-pretrained TSFM)."""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseModelWrapper
from src.utils.device import resolve_device

SEQ_LEN = 512  # Mantis-8M is trained on windows of 512 steps


def _tapped_forward(self: nn.Module, x: torch.Tensor, return_layer: int = -1) -> torch.Tensor:
    """Mantis Transformer.forward with an identity tap after each block's residual stream."""
    for i, (attn, ff) in enumerate(self.layers):
        x = attn(x) + x
        x = ff(x) + x
        x = self.taps[i](x)
        if return_layer == i:
            return x
    return x


class MantisWrapper(BaseModelWrapper):
    """Wrapper for Mantis-8M, a contrastively pretrained time-series classifier backbone.

    The package's transformer applies each block as ``x = attn(x) + x; x = ff(x) + x`` with no
    module whose output is the residual stream, so ``load()`` attaches an ``nn.Identity`` tap after
    every block (same computation, hookable output). Layer names are those taps. Inputs are
    univariate; windows of another length are linearly resized to 512 steps. Mantis applies its own
    instance scaling inside the token generator.
    """

    def __init__(self) -> None:
        """Initialize wrapper state and default device."""
        self._model: nn.Module | None = None
        self._device: torch.device = resolve_device()
        self._depth = 0

    def load(
        self,
        checkpoint: str = "paris-noah/Mantis-8M",
        *,
        device: torch.device | None = None,
    ) -> None:
        """Load Mantis-8M from HuggingFace, attach block taps, and freeze all parameters.

        Args:
            checkpoint: HuggingFace checkpoint ID or local path.
            device: Target device. If None, auto-resolves via resolve_device().

        Raises:
            RuntimeError: If the mantis-tsfm package is not installed.
        """
        self._device = device if device is not None else resolve_device()
        try:
            from mantis.architecture import Mantis8M
        except ImportError as exc:
            raise RuntimeError("mantis-tsfm not installed. Run: pip install mantis-tsfm") from exc
        model = Mantis8M.from_pretrained(checkpoint)
        transformer = model.vit_unit.transformer
        self._depth = len(transformer.layers)
        transformer.taps = nn.ModuleList([nn.Identity() for _ in range(self._depth)])
        transformer.forward = types.MethodType(_tapped_forward, transformer)
        self._model = model.to(self._device)
        self.freeze()

    def freeze(self) -> None:
        """Freeze all model parameters and set eval mode."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        for parameter in self._model.parameters():
            parameter.requires_grad = False
        _ = self._model.eval()

    def get_layer_names(self) -> list[str]:
        """Return the residual-stream taps after each transformer block."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return [f"vit_unit.transformer.taps.{i}" for i in range(self._depth)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run Mantis on univariate windows.

        Args:
            x: Tensor of shape (batch, seq_len) or (batch, seq_len, 1).

        Returns:
            The model's 256-dimensional output embedding, shape (batch, 256).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if x.ndim == 3:
            if x.shape[-1] != 1:
                raise ValueError(f"Mantis wrapper is univariate; got {x.shape[-1]} channels.")
            x = x[..., 0]
        if x.ndim != 2:
            raise ValueError(
                f"Expected (batch, seq_len) or (batch, seq_len, 1), got {tuple(x.shape)}."
            )
        x = x.to(self._device, dtype=torch.float32)[:, None, :]
        if x.shape[-1] != SEQ_LEN:
            x = F.interpolate(x, size=SEQ_LEN, mode="linear", align_corners=True)
        with torch.no_grad():
            return self._model(x)
