# pyright: reportMissingImports=false, reportImplicitOverride=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false
"""MOMENT foundation model wrapper for probing experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import BaseModelWrapper
from src.utils.device import resolve_device


class MOMENTWrapper(BaseModelWrapper):
    """Wrapper for MOMENT Foundation Model (Goswami et al., ICML 2024)."""

    def __init__(self) -> None:
        """Initialize wrapper state and default device."""
        self._model: nn.Module | None = None
        self._device: torch.device = resolve_device()

    def load(
        self,
        checkpoint: str = "AutonLab/MOMENT-1-large",
        *,
        device: torch.device | None = None,
    ) -> None:
        """Load MOMENT from HuggingFace and freeze all parameters."""
        self._device = device if device is not None else resolve_device()

        try:
            from momentfm import MOMENTPipeline
        except ImportError as exc:
            raise RuntimeError("momentfm not installed. Run: pip install momentfm") from exc

        pipeline = MOMENTPipeline.from_pretrained(
            checkpoint,
            model_kwargs={"task_name": "reconstruction"},
        )
        # MOMENTPipeline IS the nn.Module (wraps MOMENT internally)
        if not isinstance(pipeline, nn.Module):
            raise RuntimeError("Failed to resolve MOMENT nn.Module from MOMENTPipeline")

        self._model = pipeline.to(self._device)
        self.freeze()

    def freeze(self) -> None:
        """Freeze all model parameters."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for parameter in self._model.parameters():
            parameter.requires_grad = False
        self._model.eval()

    def get_layer_names(self) -> list[str]:
        """Return MOMENT encoder transformer block names for HookManager."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        layer_names = [
            name
            for name, _ in self._model.named_modules()
            if name.startswith("encoder.block.") and name.count(".") == 2
        ]
        if not layer_names:
            layer_names = [
                name
                for name, _ in self._model.named_modules()
                if name.startswith("encoder.layer.") and name.count(".") == 2
            ]
        if not layer_names:
            layer_names = [
                name
                for name, _ in self._model.named_modules()
                if "block" in name and name.count(".") == 2
            ]
        return sorted(layer_names, key=self._layer_sort_key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a frozen forward pass with MOMENT-compatible input layout."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if x.ndim == 2:
            model_input = x.unsqueeze(1)
        elif x.ndim == 3:
            model_input = x.transpose(1, 2) if x.shape[1] > x.shape[2] else x
        else:
            raise ValueError(
                f"Expected input shape (batch, seq_len) or (batch, seq_len, channels),"
                f" got {x.shape}"
            )

        model_input = model_input.to(self._device)
        with torch.no_grad():
            output = self._model(x_enc=model_input)

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
            _keys = ("reconstruction", "logits", "prediction", "predictions", "forecast")
            for key in _keys:
                value = output.get(key)
                if isinstance(value, torch.Tensor):
                    return value

        for attr in ("reconstruction", "logits", "prediction", "predictions", "forecast", "output"):
            value = getattr(output, attr, None)
            if isinstance(value, torch.Tensor):
                return value
        raise RuntimeError("Unable to extract tensor output from MOMENT forward pass")
