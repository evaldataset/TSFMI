"""Abstract base class for probing-ready time series model wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModelWrapper(ABC):
    """Abstract base class for probing-ready model wrappers.

    Subclasses wrap a pretrained time series model, freeze all parameters,
    and expose named layers for hook-based activation extraction.

    All model parameters are frozen after load() — weights are NEVER modified.

    Note:
        Multivariate input handling varies across wrappers. Autoformer uses
        the first channel only, TimesNet expands a 1d signal via projection,
        and FEDformer takes the mean across channels. See individual wrapper
        ``forward()`` docstrings for details.
    """

    # Class-level annotation so mypy knows _model exists; concrete subclasses set it in load().
    _model: nn.Module | None = None

    @abstractmethod
    def load(self, checkpoint: str, *, device: torch.device | None = None) -> None:
        """Load pretrained weights from checkpoint.

        Args:
            checkpoint: HuggingFace model ID or local path.
            device: Target device. If None, auto-resolves via resolve_device().
        """

    @abstractmethod
    def freeze(self) -> None:
        """Freeze all model parameters (set requires_grad=False).

        Must be called after load(). After freeze(), no parameters can be
        updated by optimizers.
        """

    @abstractmethod
    def get_layer_names(self) -> list[str]:
        """Return all hookable layer names for this model.

        Returns:
            List of string names as returned by model.named_modules(),
            filtered to meaningful transformer blocks (not embedding layers,
            norm layers, etc. — only the main encoder/decoder blocks).
        """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass on input time series.

        Args:
            x: Input tensor of shape (batch, seq_len) or (batch, seq_len, channels).

        Returns:
            Model output tensor. Shape depends on model architecture.
        """

    @property
    def model(self) -> nn.Module:
        """The underlying nn.Module. Required for HookManager."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    def is_frozen(self) -> bool:
        """Check whether all parameters have requires_grad=False.

        Raises:
            RuntimeError: If model has not been loaded yet.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return all(not p.requires_grad for p in self._model.parameters())
