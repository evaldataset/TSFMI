"""Device resolution utilities."""

from __future__ import annotations

import torch


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve a device string to a torch.device.

    If device is None, auto-selects CUDA if available, else CPU.

    Args:
        device: Device string ("cuda", "cpu", "cuda:0") or None for auto.

    Returns:
        Resolved torch.device.
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
