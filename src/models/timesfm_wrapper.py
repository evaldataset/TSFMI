# pyright: reportMissingImports=false, reportImplicitOverride=false
"""TimesFM model wrapper for probing experiments."""

from __future__ import annotations

import re

import torch
import torch.nn as nn

from src.models.base import BaseModelWrapper
from src.utils.device import resolve_device


def _move_module_to_device(module: nn.Module, device: torch.device) -> nn.Module:
    return module.to(device)


class TimesFMWrapper(BaseModelWrapper):
    """Wrapper for TimesFM decoder representations."""

    def __init__(self) -> None:
        """Initialize wrapper state and default device."""
        self._model: nn.Module | None = None
        self._prediction_model: nn.Module | None = None
        self._device: torch.device = resolve_device()

    def load(
        self,
        checkpoint: str = "google/timesfm-2.0-500m-pytorch",
        *,
        device: torch.device | None = None,
    ) -> None:
        """Load TimesFM prediction model and expose decoder backbone for hooks.

        Args:
            checkpoint: HuggingFace checkpoint ID or local path.
            device: Target device. If None, auto-resolves via resolve_device().
        """
        self._device = device if device is not None else resolve_device()

        try:
            from transformers import TimesFmModelForPrediction
        except ImportError as exc:
            raise RuntimeError("transformers not installed. Run: pip install transformers") from exc

        prediction_model = TimesFmModelForPrediction.from_pretrained(checkpoint)
        if not isinstance(prediction_model, nn.Module):
            raise RuntimeError("Failed to resolve TimesFM prediction nn.Module")
        prediction_module: nn.Module = prediction_model

        decoder = getattr(prediction_module, "decoder", None)
        if not isinstance(decoder, nn.Module):
            raise RuntimeError("TimesFM prediction model does not expose a decoder module")

        self._prediction_model = _move_module_to_device(prediction_module, self._device)
        self._model = _move_module_to_device(decoder, self._device)
        self.freeze()

    def freeze(self) -> None:
        """Freeze all model parameters and set eval mode."""
        if self._prediction_model is None or self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for parameter in self._prediction_model.parameters():
            parameter.requires_grad = False

        self._prediction_model.eval()
        self._model.eval()

    def get_layer_names(self) -> list[str]:
        """Return TimesFM decoder block names for hook registration."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        pattern = re.compile(r"^layers\.\d+$")
        layer_names = [name for name, _ in self._model.named_modules() if pattern.match(name)]
        return sorted(layer_names, key=self._layer_sort_key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run TimesFM prediction forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len) or (batch, seq_len, 1).

        Returns:
            Tensor output from prediction model.
        """
        if self._prediction_model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if x.ndim == 2:
            series = x
        elif x.ndim == 3 and x.shape[-1] == 1:
            series = x.squeeze(-1)
        else:
            raise ValueError(
                f"Expected input shape (batch, seq_len) or (batch, seq_len, 1), got {x.shape}",
            )

        with torch.no_grad():
            series = series.to(self._device, dtype=torch.float32)
            batch_size = series.shape[0]
            freq = torch.zeros(batch_size, dtype=torch.long, device=self._device)
            output = self._prediction_model(past_values=series, freq=freq)
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
            for key in ("mean_predictions", "last_hidden_state", "predictions", "logits"):
                value = output.get(key)
                if isinstance(value, torch.Tensor):
                    return value

        for attr in ("mean_predictions", "last_hidden_state", "predictions", "logits"):
            value = getattr(output, attr, None)
            if isinstance(value, torch.Tensor):
                return value

        raise RuntimeError("Unable to extract tensor output from TimesFM forward pass")
