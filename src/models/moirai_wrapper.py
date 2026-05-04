# pyright: reportMissingImports=false, reportImplicitOverride=false
"""Moirai model wrapper for probing experiments."""

from __future__ import annotations

import re

import torch
import torch.nn as nn

from src.models.base import BaseModelWrapper
from src.utils.device import resolve_device


class MoiraiWrapper(BaseModelWrapper):
    """Wrapper for Moirai encoder representations."""

    def __init__(self) -> None:
        """Initialize wrapper state and default device."""
        self._model: nn.Module | None = None
        self._module: nn.Module | None = None
        self._device: torch.device = resolve_device()

    def load(
        self,
        checkpoint: str = "Salesforce/moirai-2.0-R-small",
        *,
        device: torch.device | None = None,
    ) -> None:
        """Load Moirai checkpoint and freeze parameters.

        Args:
            checkpoint: HuggingFace checkpoint ID or local path.
            device: Target device. If None, auto-resolves via resolve_device().
        """
        self._device = device if device is not None else resolve_device()

        try:
            from uni2ts.model.moirai2 import Moirai2Module
        except ImportError as exc:
            raise RuntimeError("uni2ts not installed. Run: pip install uni2ts") from exc

        module = Moirai2Module.from_pretrained(checkpoint)
        if not isinstance(module, nn.Module):
            raise RuntimeError("Failed to resolve Moirai nn.Module from Moirai2Module")

        self._module = module.to(self._device)
        self._model = self._module
        self.freeze()

    def freeze(self) -> None:
        """Freeze all model parameters and set eval mode."""
        if self._module is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for parameter in self._module.parameters():
            parameter.requires_grad = False

        self._module.eval()

    def get_layer_names(self) -> list[str]:
        """Return Moirai encoder layer names for hook registration."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        pattern = re.compile(r"^encoder\.layers\.\d+$")
        layer_names = [name for name, _ in self._model.named_modules() if pattern.match(name)]
        return sorted(layer_names, key=self._layer_sort_key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run Moirai forward pass with patched 1D input series.

        Args:
            x: Input tensor of shape (batch, seq_len) or (batch, seq_len, 1).

        Returns:
            Tensor output from Moirai forward pass.
        """
        if self._module is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if x.ndim == 2:
            series = x
        elif x.ndim == 3 and x.shape[-1] == 1:
            series = x.squeeze(-1)
        else:
            raise ValueError(
                f"Expected input shape (batch, seq_len) or (batch, seq_len, 1), got {x.shape}",
            )

        patch_size = 16
        batch_size, seq_len = series.shape
        if seq_len % patch_size != 0:
            raise ValueError(
                f"Expected seq_len divisible by {patch_size}, got {seq_len}",
            )

        num_patches = seq_len // patch_size
        with torch.no_grad():
            target = series.to(self._device, dtype=torch.float32).reshape(
                batch_size, num_patches, patch_size
            )
            observed_mask = torch.ones_like(target, dtype=torch.bool)
            sample_id = torch.zeros(batch_size, num_patches, dtype=torch.long, device=self._device)
            time_id = (
                torch.arange(num_patches, device=self._device).unsqueeze(0).expand(batch_size, -1)
            )
            variate_id = torch.zeros(batch_size, num_patches, dtype=torch.long, device=self._device)
            prediction_mask = torch.zeros(
                batch_size, num_patches, dtype=torch.bool, device=self._device
            )

            output = self._module(
                target=target,
                observed_mask=observed_mask,
                sample_id=sample_id,
                time_id=time_id,
                variate_id=variate_id,
                prediction_mask=prediction_mask,
                training_mode=False,
            )
            return self._extract_tensor_output(output)

    @staticmethod
    def _layer_sort_key(layer_name: str) -> tuple[int, str]:
        tail = layer_name.rsplit(".", maxsplit=1)[-1]
        return (int(tail), layer_name) if tail.isdigit() else (-1, layer_name)

    @staticmethod
    def _extract_tensor_output(output: object) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return output[0]

        if isinstance(output, dict):
            for key in ("last_hidden_state", "prediction", "predictions", "logits"):
                value = output.get(key)
                if isinstance(value, torch.Tensor):
                    return value

        for attr in ("last_hidden_state", "prediction", "predictions", "logits"):
            value = getattr(output, attr, None)
            if isinstance(value, torch.Tensor):
                return value

        raise RuntimeError("Unable to extract tensor output from Moirai forward pass")
