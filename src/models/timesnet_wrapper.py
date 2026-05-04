# pyright: reportMissingImports=false, reportImplicitOverride=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUntypedBaseClass=false, reportAny=false, reportUnannotatedClassAttribute=false, reportImplicitStringConcatenation=false, reportUnusedCallResult=false
"""TimesNet wrapper for probing experiments."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseModelWrapper
from src.utils.device import resolve_device


class InceptionBlock1D(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        kernel_sizes: tuple[int, ...] = (1, 3, 5),
    ) -> None:
        super().__init__()
        self.branches: nn.ModuleList = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_ff,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
                for kernel_size in kernel_sizes
            ]
        )
        self.proj: nn.Conv1d = nn.Conv1d(d_ff * len(kernel_sizes), d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (batch, channels, length), got shape {x.shape}")

        branch_outputs = [F.gelu(branch(x)) for branch in self.branches]
        merged = torch.cat(branch_outputs, dim=1)
        return self.proj(merged)


class TimesBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        *,
        d_ff: int,
        top_k: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.top_k: int = top_k
        self.inception: InceptionBlock1D = InceptionBlock1D(d_model=d_model, d_ff=d_ff)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.norm: nn.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D tensor (batch, seq_len, d_model), got shape {x.shape}")

        batch_size, seq_len, d_model = x.shape
        periods, period_weights = self._select_periods(x)
        if not periods:
            return x

        period_outputs: list[torch.Tensor] = []
        for period in periods:
            padded_len = math.ceil(seq_len / period) * period
            if padded_len > seq_len:
                x_padded = F.pad(x, (0, 0, 0, padded_len - seq_len))
            else:
                x_padded = x

            num_periods = padded_len // period
            y = x_padded.reshape(batch_size, num_periods, period, d_model)
            y = y.reshape(batch_size * num_periods, period, d_model).transpose(1, 2)
            y = self.inception(y)
            y = y.transpose(1, 2).reshape(batch_size, num_periods, period, d_model)
            y = y.reshape(batch_size, padded_len, d_model)[:, :seq_len, :]
            period_outputs.append(y)

        stacked = torch.stack(period_outputs, dim=-1)
        attn = torch.softmax(period_weights, dim=0).view(1, 1, 1, -1)
        aggregated = (stacked * attn).sum(dim=-1)
        return self.norm(x + self.dropout(aggregated))

    def _select_periods(self, x: torch.Tensor) -> tuple[list[int], torch.Tensor]:
        seq_len = x.shape[1]
        spectrum = torch.fft.rfft(x.float(), dim=1)
        amplitude = spectrum.abs().mean(dim=(0, 2))

        if amplitude.numel() <= 1:
            return [max(1, seq_len)], x.new_tensor([1.0])

        amplitude = amplitude.clone()
        amplitude[0] = 0.0
        max_k = min(self.top_k, amplitude.numel() - 1)
        if max_k <= 0:
            return [max(1, seq_len)], x.new_tensor([1.0])

        values, indices = torch.topk(amplitude, k=max_k)

        periods: list[int] = []
        weights: list[float] = []
        for index, value in zip(indices.tolist(), values.tolist(), strict=True):
            if index <= 0:
                continue
            period = max(1, math.ceil(seq_len / index))
            if period in periods:
                continue
            periods.append(period)
            weights.append(float(value))

        if not periods:
            return [max(1, seq_len)], x.new_tensor([1.0])

        return periods, x.new_tensor(weights)


class TimesNetBackbone(nn.Module):
    def __init__(
        self,
        num_variates: int = 7,
        seq_len: int = 512,
        d_model: int = 64,
        e_layers: int = 3,
        top_k: int = 3,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        feedforward_dim = d_ff or d_model
        self.seq_len: int = seq_len
        self.value_embedding: nn.Linear = nn.Linear(num_variates, d_model)
        self.position_embedding: nn.Parameter = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.layers: nn.ModuleList = nn.ModuleList(
            [
                TimesBlock(
                    d_model=d_model,
                    d_ff=feedforward_dim,
                    top_k=top_k,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, channels), got shape {x.shape}")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got input shape {x.shape}")
        if x.shape[2] != self.value_embedding.in_features:
            raise ValueError(
                f"Expected input channels to match num_variates "
                f"({self.value_embedding.in_features}), got {x.shape[2]}"
            )

        x = self.value_embedding(x)
        x = x + self.position_embedding[:, : x.shape[1], :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TimesNetWrapper(BaseModelWrapper):
    """Wrapper for a random-init TimesNet-style encoder for probing experiments."""

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
        seq_len: int = 512,
        d_model: int = 64,
        n_heads: int = 4,
        e_layers: int = 3,
        top_k: int = 3,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        del n_heads
        if checkpoint:
            raise ValueError(
                "TimesNetWrapper supports random initialization only; checkpoint must be empty."
            )

        self._device = device if device is not None else resolve_device()
        model = TimesNetBackbone(
            num_variates=num_variates,
            seq_len=seq_len,
            d_model=d_model,
            e_layers=e_layers,
            top_k=top_k,
            d_ff=d_ff,
            dropout=dropout,
        )
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
        """Return TimesNet block names for HookManager."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        layer_names = [
            name
            for name, _ in self._model.named_modules()
            if name.startswith("layers.") and name.count(".") == 1
        ]
        return sorted(layer_names, key=self._layer_sort_key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a frozen forward pass with TimesNet-compatible input layout."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if not isinstance(self._model, TimesNetBackbone):
            raise RuntimeError("Loaded model is not a TimesNetBackbone instance.")

        if x.ndim == 2:
            x_3d = x.unsqueeze(-1)
        elif x.ndim == 3:
            x_3d = x
        else:
            raise ValueError(
                "Expected input shape (batch, seq_len) or (batch, seq_len, channels), "
                f"got {x.shape}"
            )

        if x_3d.shape[1] != self._model.seq_len:
            raise ValueError(
                f"Expected seq_len={self._model.seq_len}, got input shape {x_3d.shape}."
            )

        expected_channels = self._model.value_embedding.in_features
        if x_3d.shape[2] != expected_channels:
            if x_3d.shape[2] == 1 and expected_channels > 1:
                x_3d = x_3d.expand(-1, -1, expected_channels)
            else:
                raise ValueError(
                    f"Expected channels={expected_channels}, got input shape {x_3d.shape}."
                )

        with torch.no_grad():
            return self._model(x_3d.to(self._device))

    @staticmethod
    def _layer_sort_key(layer_name: str) -> tuple[int, str]:
        tail = layer_name.rsplit(".", maxsplit=1)[-1]
        return (int(tail), layer_name) if tail.isdigit() else (-1, layer_name)
