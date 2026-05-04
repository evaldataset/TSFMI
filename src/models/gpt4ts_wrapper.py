# pyright: reportMissingImports=false, reportImplicitOverride=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false
"""GPT4TS wrapper with frozen GPT-2 backbone for probing experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

from src.models.base import BaseModelWrapper
from src.utils.device import resolve_device


class PatchEmbedding(nn.Module):
    """Patch embedding layer for univariate time series input.

    This module segments a sequence into fixed-size patches, projects each patch
    into ``d_model`` space, and adds learnable positional embeddings.
    """

    def __init__(
        self,
        *,
        seq_len: int = 512,
        patch_size: int = 16,
        stride: int = 16,
        d_model: int = 768,
    ) -> None:
        super().__init__()
        if seq_len < patch_size:
            raise ValueError(f"seq_len ({seq_len}) must be >= patch_size ({patch_size}).")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}.")

        self.seq_len: int = seq_len
        self.patch_size: int = patch_size
        self.stride: int = stride
        self.d_model: int = d_model

        self.num_patches: int = ((seq_len - patch_size) // stride) + 1
        if self.num_patches <= 0:
            raise ValueError(
                f"Patch configuration yields no patches: "
                f"seq_len={seq_len}, patch_size={patch_size}, stride={stride}."
            )

        self.proj: nn.Linear = nn.Linear(patch_size, d_model)
        self.position_embedding: nn.Parameter = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model),
        )
        _ = nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw series into patch embeddings.

        Args:
            x: Input tensor of shape ``(batch, seq_len)``.

        Returns:
            Embedded patches of shape ``(batch, num_patches, d_model)``.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (batch, seq_len), got {x.shape}.")
        if x.shape[1] < self.patch_size:
            raise ValueError(
                f"Input seq_len ({x.shape[1]}) must be >= patch_size ({self.patch_size})."
            )

        patches: torch.Tensor = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        embeddings: torch.Tensor = self.proj(patches)

        num_patches: int = int(embeddings.shape[1])
        if num_patches > self.num_patches:
            raise ValueError(
                f"Input produces {num_patches} patches, exceeds max {self.num_patches}. Configured "
                + f"seq_len={self.seq_len}."
            )

        position = self.position_embedding[:, :num_patches, :]
        return embeddings + position


class _GPT2Backbone(nn.Module):
    """Container exposing GPT-2 modules under ``transformer.*`` names."""

    def __init__(self, transformer: GPT2Model) -> None:
        super().__init__()
        self.transformer: GPT2Model = transformer

    def forward(self, *, inputs_embeds: torch.Tensor) -> object:
        """Run GPT-2 transformer forward pass with external embeddings."""
        return self.transformer(inputs_embeds=inputs_embeds)


class GPT4TSWrapper(BaseModelWrapper):
    """Wrapper for GPT4TS-style frozen GPT-2 probing setup."""

    def __init__(
        self,
        *,
        seq_len: int = 512,
        patch_size: int = 16,
        stride: int = 16,
        d_model: int = 768,
    ) -> None:
        """Initialize frozen patch embedding configuration for GPT4TS."""
        self._model: nn.Module | None = None
        self._backbone: _GPT2Backbone | None = None
        self._device: torch.device = resolve_device()
        self._patch_embedding: PatchEmbedding = PatchEmbedding(
            seq_len=seq_len,
            patch_size=patch_size,
            stride=stride,
            d_model=d_model,
        )
        for parameter in self._patch_embedding.parameters():
            parameter.requires_grad = False

    def load(self, checkpoint: str = "gpt2", *, device: torch.device | None = None) -> None:
        """Load pretrained GPT-2 backbone and freeze all parameters.

        Args:
            checkpoint: HuggingFace model ID or local path.
            device: Target device. If None, resolves with ``resolve_device()``.
        """
        self._device = device if device is not None else resolve_device()

        transformer = GPT2Model.from_pretrained(checkpoint)
        config: GPT2Config = transformer.config

        hidden_size = int(config.hidden_size)
        if hidden_size != self._patch_embedding.d_model:
            raise ValueError(
                f"Patch embedding d_model ({self._patch_embedding.d_model}) "
                f"must match GPT-2 hidden size ({hidden_size})."
            )
        n_layer = int(config.n_layer)
        if n_layer != 12:
            raise ValueError(f"GPT4TSWrapper expects 12 GPT-2 layers, got n_layer={n_layer}.")

        self._backbone = _GPT2Backbone(transformer=transformer).to(self._device)
        self._model = self._backbone
        self._patch_embedding = self._patch_embedding.to(self._device)
        self.freeze()

    def freeze(self) -> None:
        """Freeze GPT-2 backbone and patch embedding parameters."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self._backbone is None:
            raise RuntimeError("Backbone not loaded. Call load() first.")

        for parameter in self._model.parameters():
            parameter.requires_grad = False
        for parameter in self._patch_embedding.parameters():
            parameter.requires_grad = False

        _ = self._model.eval()
        _ = self._patch_embedding.eval()

    def get_layer_names(self) -> list[str]:
        """Return GPT-2 transformer block names for hook-based extraction."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self._backbone is None:
            raise RuntimeError("Backbone not loaded. Call load() first.")

        return [f"transformer.h.{i}" for i in range(12)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run frozen GPT4TS forward pass and return hidden representations.

        Args:
            x: Input tensor of shape ``(batch, seq_len)`` or ``(batch, seq_len, 1)``.

        Returns:
            Last hidden state tensor of shape ``(batch, num_patches, 768)``.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self._backbone is None:
            raise RuntimeError("Backbone not loaded. Call load() first.")

        if x.ndim == 2:
            series = x
        elif x.ndim == 3 and x.shape[-1] == 1:
            series = x.squeeze(-1)
        else:
            raise ValueError(
                f"Expected input shape (batch, seq_len) or (batch, seq_len, 1), got {x.shape}."
            )

        with torch.no_grad():
            series = series.to(self._device, dtype=torch.float32)
            patch_embeddings = self._patch_embedding(series)
            output = self._backbone(inputs_embeds=patch_embeddings)

            last_hidden_state = getattr(output, "last_hidden_state", None)
            if isinstance(last_hidden_state, torch.Tensor):
                return last_hidden_state

            if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                return output[0]

        raise RuntimeError("Unable to extract last_hidden_state from GPT-2 output.")
