# pyright: reportMissingImports=false, reportImplicitOverride=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedBaseClass=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false
"""FEDformer wrapper for probing experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import BaseModelWrapper
from src.utils.device import resolve_device


class SeriesDecomposition(nn.Module):
    """Series decomposition via moving average trend extraction.

    Args:
        moving_avg: Kernel size for moving-average trend estimation.
    """

    def __init__(self, moving_avg: int = 25) -> None:
        super().__init__()
        if moving_avg <= 0:
            raise ValueError(f"moving_avg must be positive, got {moving_avg}.")
        self.moving_avg = moving_avg
        self.avg_pool = nn.AvgPool1d(kernel_size=moving_avg, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split input into seasonal residual and moving-average trend.

        Args:
            x: Input tensor of shape (batch, seq_len, channels).

        Returns:
            Tuple of (seasonal, trend), both shaped (batch, seq_len, channels).
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, channels), got shape {x.shape}")

        pad_front = (self.moving_avg - 1) // 2
        pad_back = self.moving_avg - 1 - pad_front
        x_t = x.transpose(1, 2)
        x_padded = torch.nn.functional.pad(x_t, (pad_front, pad_back), mode="replicate")
        trend = self.avg_pool(x_padded).transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class FEDformerEncoderLayer(nn.Module):
    """Single FEDformer encoder block with pre-norm attention and FFN.

    Uses standard multi-head self-attention as a stand-in for FEDformer's
    frequency-domain auto-correlation mechanism.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        d_ff: Feedforward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm attention and feedforward blocks.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, d_model), got shape {x.shape}")

        attn_input = self.norm_attn(x)
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.attn_dropout(attn_output)

        ffn_input = self.norm_ffn(x)
        x = x + self.ffn(ffn_input)
        return x


class FEDformerBackbone(nn.Module):
    """Simplified FEDformer encoder backbone for probing.

    This implementation follows FEDformer decomposition + encoder structure,
    but uses standard multi-head self-attention as a simplified stand-in for
    the paper's frequency-domain auto-correlation attention.

    Args:
        seq_len: Expected input sequence length.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        e_layers: Number of encoder layers.
        d_ff: Feedforward hidden dimension. Defaults to 4 * d_model.
        dropout: Dropout probability.
        moving_avg: Kernel size for moving-average decomposition.
    """

    def __init__(
        self,
        seq_len: int = 512,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int | None = None,
        dropout: float = 0.1,
        moving_avg: int = 25,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}.")

        self.seq_len = seq_len
        self.d_model = d_model
        feedforward_dim = d_ff or d_model * 4

        self.series_decomposition = SeriesDecomposition(moving_avg=moving_avg)
        self.input_projection = nn.Linear(1, d_model)
        self.trend_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [
                FEDformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=feedforward_dim,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence into hidden representations.

        Args:
            x: Input tensor of shape (batch, seq_len, channels).

        Returns:
            Encoded tensor of shape (batch, seq_len, d_model).
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, channels), got shape {x.shape}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got input shape {x.shape}.")

        seasonal, trend = self.series_decomposition(x)
        seasonal_scalar = seasonal.mean(dim=-1, keepdim=True)
        trend_scalar = trend.mean(dim=-1, keepdim=True)

        x_hidden = self.input_projection(seasonal_scalar)
        trend_hidden = self.trend_projection(trend_scalar)
        x_hidden = self.dropout(x_hidden + self.position_embedding)

        for encoder_layer in self.encoder_layers:
            x_hidden = encoder_layer(x_hidden)

        return self.output_norm(x_hidden + trend_hidden)


class FEDformerWrapper(BaseModelWrapper):
    """Wrapper for simplified FEDformer encoder (Zhou et al., ICML 2022)."""

    def __init__(self) -> None:
        """Initialize wrapper state and default device."""
        self._model: nn.Module | None = None
        self._device: torch.device = resolve_device()

    def load(
        self,
        checkpoint: str = "",
        *,
        device: torch.device | None = None,
        seq_len: int = 512,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 3,
        d_ff: int | None = None,
        dropout: float = 0.1,
        moving_avg: int = 25,
    ) -> None:
        """Initialize or load FEDformer backbone weights.

        Args:
            checkpoint: Path to a local .pt state dict. Empty string uses random init.
            device: Target device.
            seq_len: Expected input sequence length.
            d_model: Transformer hidden dimension.
            n_heads: Number of attention heads.
            e_layers: Number of encoder layers.
            d_ff: Feedforward hidden dimension. Defaults to 4 * d_model.
            dropout: Dropout probability.
            moving_avg: Moving-average kernel size for decomposition.
        """
        self._device = device if device is not None else resolve_device()
        model = FEDformerBackbone(
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            moving_avg=moving_avg,
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
        """Return FEDformer encoder layer names for HookManager."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        layer_names = [
            name
            for name, _ in self._model.named_modules()
            if name.startswith("encoder_layers.") and name.count(".") == 1
        ]
        return sorted(layer_names, key=self._layer_sort_key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a frozen forward pass with FEDformer-compatible input layout.

        Args:
            x: Input tensor of shape (batch, seq_len) or (batch, seq_len, channels).

        Returns:
            Encoded tensor of shape (batch, seq_len, d_model).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if x.ndim == 2:
            x_3d = x.unsqueeze(-1)
        elif x.ndim == 3:
            x_3d = x
        else:
            raise ValueError(
                f"Expected (batch, seq_len) or (batch, seq_len, channels), got {x.shape}"
            )

        if not isinstance(self._model, FEDformerBackbone):
            raise RuntimeError("Loaded model is not a FEDformerBackbone instance.")

        if x_3d.shape[1] != self._model.seq_len:
            raise ValueError(
                f"Expected seq_len={self._model.seq_len}, got input shape {x_3d.shape}."
            )

        x_3d = x_3d.to(self._device)
        with torch.no_grad():
            return self._model(x_3d)

    @staticmethod
    def _layer_sort_key(layer_name: str) -> tuple[int, str]:
        tail = layer_name.rsplit(".", maxsplit=1)[-1]
        return (int(tail), layer_name) if tail.isdigit() else (-1, layer_name)
