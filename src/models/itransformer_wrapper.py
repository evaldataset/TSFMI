# pyright: reportMissingImports=false, reportImplicitOverride=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedBaseClass=false
"""iTransformer wrapper for probing experiments."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from src.models.base import BaseModelWrapper
from src.utils.device import resolve_device


class iTransformerEncoder(nn.Module):
    """Lightweight iTransformer encoder backbone for probing.

    Implements the inverted attention mechanism from Liu et al. (ICLR 2024):
    instead of attending over time steps, attention is applied over variables/channels.

    Each token represents one channel's time series. Self-attention learns
    inter-variable relationships.

    Args:
        num_variates: Number of input channels/variables.
        seq_len: Input sequence length.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer encoder layers.
        d_ff: Feedforward dimension. Defaults to 4 * d_model.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_variates: int = 7,
        seq_len: int = 96,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        feedforward_dim = d_ff or d_model * 4
        self.value_embedding: nn.Linear = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of time series with variate-wise attention.

        Args:
            x: Input tensor of shape (batch, seq_len, num_variates).

        Returns:
            Encoded representations of shape (batch, num_variates, d_model).
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, channels), got shape {x.shape}")

        x = x.transpose(1, 2)
        x = self.value_embedding(x)
        x = self.encoder(x)
        # cast: torch stubs return Any from nn.Module.__call__
        return cast(torch.Tensor, self.norm(x))


class iTransformerWrapper(BaseModelWrapper):
    """Wrapper for iTransformer encoder (Liu et al., ICLR 2024) for probing experiments.

    Since no official HuggingFace checkpoint exists for iTransformer, this wrapper
    uses a lightweight iTransformerEncoder backbone. The backbone is randomly
    initialized by default and can optionally load a local state dict.

    Reference: Liu et al. "iTransformer: Inverted Transformers Are Effective for Time
    Series Forecasting." ICLR 2024. arXiv:2310.06625.
    """

    def __init__(self) -> None:
        """Initialize wrapper state and default device."""
        self._model: nn.Module | None = None
        self._device: torch.device = resolve_device()

    def load(
        self,
        checkpoint: str = "",
        *,
        device: torch.device | None = None,
        num_variates: int = 7,
        seq_len: int = 96,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        """Initialize or load iTransformer encoder weights.

        Args:
            checkpoint: Path to a local .pt state dict. Empty string uses random init.
            device: Target device.
            num_variates: Number of input channels.
            seq_len: Expected input sequence length.
            d_model: Transformer hidden dimension.
            n_heads: Number of attention heads.
            n_layers: Number of encoder layers.
            d_ff: Feedforward dimension. Defaults to 4 * d_model.
            dropout: Dropout probability.
        """
        self._device = device if device is not None else resolve_device()
        model = iTransformerEncoder(
            num_variates=num_variates,
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        if checkpoint:
            state_dict = torch.load(checkpoint, map_location=self._device, weights_only=True)
            model.load_state_dict(state_dict)

        self._model = model.to(self._device)
        self.freeze()

    def freeze(self) -> None:
        """Freeze all model parameters and set eval mode."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for parameter in self._model.parameters():
            parameter.requires_grad = False
        self._model.eval()

    def get_layer_names(self) -> list[str]:
        """Return iTransformer encoder layer names for HookManager."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        layer_names = [
            name
            for name, _ in self._model.named_modules()
            if name.startswith("encoder.layers.") and name.count(".") == 2
        ]
        return sorted(layer_names, key=self._layer_sort_key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a frozen forward pass with iTransformer-compatible input layout.

        Args:
            x: Input tensor of shape (batch, seq_len) or (batch, seq_len, channels).

        Returns:
            Encoded tensor of shape (batch, num_variates, d_model).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if x.ndim == 2:
            x_3d = x.unsqueeze(-1)
        elif x.ndim == 3:
            x_3d = x
        else:
            raise ValueError(
                f"Expected input shape (batch, seq_len) or (batch, seq_len, channels),"
                f" got {x.shape}"
            )

        if not isinstance(self._model, iTransformerEncoder):
            raise RuntimeError("Loaded model is not an iTransformerEncoder instance.")

        expected_seq_len = self._model.value_embedding.in_features
        if x_3d.shape[1] != expected_seq_len:
            raise ValueError(f"Expected seq_len={expected_seq_len}, got input shape {x_3d.shape}.")

        x_3d = x_3d.to(self._device)
        with torch.no_grad():
            # cast: torch stubs return Any from nn.Module.__call__
            return cast(torch.Tensor, self._model(x_3d))

    @staticmethod
    def _layer_sort_key(layer_name: str) -> tuple[int, str]:
        tail = layer_name.rsplit(".", maxsplit=1)[-1]
        return (int(tail), layer_name) if tail.isdigit() else (-1, layer_name)
