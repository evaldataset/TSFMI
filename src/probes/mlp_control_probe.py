"""MLP control probe for selectivity measurement (Hewitt & Liang, 2019)."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn


class MLPControlProbe(nn.Module):
    """Two-layer MLP control probe for selectivity measurement.

    Following Hewitt & Liang (2019), the control probe provides a nonlinear upper
    bound on probe accuracy. Selectivity = linear_acc - control_acc measures how much
    of the probe's success is due to LINEAR accessibility vs. memorization/nonlinear
    task capability.

    Architecture: Linear(input_dim, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, output_dim)

    Args:
        input_dim: Dimensionality of frozen representation.
        output_dim: Number of classes or 1 for regression.
        hidden_dim: Width of the hidden layer. Defaults to max(input_dim // 2, 64).
        dropout: Dropout probability on hidden layer (0.0 = disabled).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = hidden_dim if hidden_dim is not None else max(input_dim // 2, 64)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, output_dim),
        )
        self._input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP control probe.

        Args:
            x: Frozen representations of shape (N, input_dim).

        Returns:
            Logits/predictions of shape (N, output_dim).

        Raises:
            ValueError: If x.ndim != 2 or x.shape[1] != input_dim.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (N, input_dim), got {x.shape}")
        if x.shape[1] != self._input_dim:
            raise ValueError(f"Expected input_dim={self._input_dim}, got {x.shape[1]}")
        # cast: torch stubs return Any from nn.Module.__call__
        return cast(torch.Tensor, self.net(x))
