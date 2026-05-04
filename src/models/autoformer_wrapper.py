# pyright: reportMissingImports=false, reportImplicitOverride=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedBaseClass=false
"""Autoformer model wrapper for probing experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.base import BaseModelWrapper
from src.utils.device import resolve_device


class _AutoCorrelation(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.scale = (d_model // n_heads) ** -0.5
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        q = rearrange(self.q(x), "b l (h d) -> b h l d", h=self.n_heads)
        k = rearrange(self.k(x), "b l (h d) -> b h l d", h=self.n_heads)
        v = rearrange(self.v(x), "b l (h d) -> b h l d", h=self.n_heads)

        # Simplified auto-correlation: standard scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.out_proj(out), attn


class _AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = _AutoCorrelation(d_model, n_heads, dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_new, _ = self.attention(self.attention_norm(x))
        x = x + x_new
        x = x + self.ffn(self.ffn_norm(x))
        return x


class _AutoformerEncoder(nn.Module):
    def __init__(
        self,
        seq_len: int = 512,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.value_embedding = nn.Linear(1, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [_AutoformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.value_embedding(x) + self.position_embedding
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class AutoformerWrapper(BaseModelWrapper):
    """Wrapper for Autoformer (Wu et al., ICLR 2022) for probing experiments.

    Inline implementation with simplified auto-correlation attention.

    Note:
        Uses a lightweight inline backbone (random init) rather than HuggingFace's
        ``AutoformerModel``, because the HF version couples encoder access to
        forecasting-specific preprocessing (lags, scalers, time features).
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
        seq_len: int = 512,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        self._device = device if device is not None else resolve_device()

        model = _AutoformerEncoder(
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
        """Freeze all parameters and set eval mode."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        for parameter in self._model.parameters():
            parameter.requires_grad = False
        self._model.eval()

    def get_layer_names(self) -> list[str]:
        """Return top-level encoder layer names for hook registration."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        return [
            name
            for name, _ in self._model.named_modules()
            if name.startswith("layers.") and name.count(".") == 1
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a frozen forward pass with Autoformer-compatible input layout."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if x.ndim == 2:
            x = x.unsqueeze(-1)
        elif x.ndim != 3:
            raise ValueError(
                f"Expected (batch, seq_len) or (batch, seq_len, channels), got {x.shape}"
            )

        x = x.to(self._device)
        with torch.no_grad():
            return self._model(x)

    @staticmethod
    def _layer_sort_key(layer_name: str) -> tuple[int, str]:
        tail = layer_name.rsplit(".", maxsplit=1)[-1]
        return (int(tail), layer_name) if tail.isdigit() else (-1, layer_name)
