"""Linear probe for measuring linear separability of frozen representations."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """Single linear layer probe for measuring linear separability.

    Implements the classic probing classifier (Alain & Bengio, 2016; Hewitt & Liang, 2019).
    A linear probe that achieves high accuracy indicates the target property is
    LINEARLY ENCODED in the representation. Low accuracy → not linearly accessible.

    Supports both classification (integer labels) and regression (float targets).

    Args:
        input_dim: Dimensionality of the frozen representation (hidden_dim).
        output_dim: Number of classes (classification) or 1 (regression).
        bias: Whether to include a bias term.
    """

    def __init__(self, input_dim: int, output_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear probe.

        Args:
            x: Frozen representations of shape (N, input_dim).

        Returns:
            Logits/predictions of shape (N, output_dim).

        Raises:
            ValueError: If x.ndim != 2 or x.shape[1] != input_dim.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (N, input_dim), got {x.shape}")
        if x.shape[1] != self.linear.in_features:
            raise ValueError(f"Expected input_dim={self.linear.in_features}, got {x.shape[1]}")
        # cast: nn.Linear.__call__ return type is Any in torch stubs; runtime always returns Tensor
        return cast(torch.Tensor, self.linear(x))
