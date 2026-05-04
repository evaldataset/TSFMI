# pyright: reportMissingImports=false, reportImplicitOverride=false
"""Timer foundation model wrapper for probing experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import BaseModelWrapper
from src.utils.device import resolve_device


class TimerWrapper(BaseModelWrapper):
    """Wrapper for Timer decoder-only model (Liu et al., ICLR 2024)."""

    def __init__(self) -> None:
        """Initialize wrapper state and default device."""
        self._model: nn.Module | None = None
        self._device: torch.device = resolve_device()

    def load(
        self,
        checkpoint: str = "thuml/timer-base-84m",
        *,
        device: torch.device | None = None,
    ) -> None:
        """Load Timer from HuggingFace and freeze all parameters.

        Args:
            checkpoint: HuggingFace checkpoint ID or local path.
            device: Target device. If None, auto-resolves via resolve_device().

        Raises:
            RuntimeError: If transformers is not installed.
        """
        self._device = device if device is not None else resolve_device()

        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("transformers not installed. Run: pip install transformers") from exc

        loaded_model: nn.Module = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            device_map=str(self._device),
        )

        self._model = loaded_model
        self.freeze()

    def freeze(self) -> None:
        """Freeze all model parameters and set eval mode."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for parameter in self._model.parameters():
            parameter.requires_grad = False
        _ = self._model.eval()

    def get_layer_names(self) -> list[str]:
        """Return Timer decoder block names for HookManager."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import re

        module_names = set(dict(self._model.named_modules()).keys())
        # Match only top-level block names (e.g., "model.model.layers.3"), not sub-modules
        deep = [n for n in module_names if re.fullmatch(r"model\.model\.layers\.\d+", n)]
        if deep:
            return sorted(deep, key=lambda n: int(n.rsplit(".", 1)[1]))
        shallow = [n for n in module_names if re.fullmatch(r"model\.layers\.\d+", n)]
        if shallow:
            return sorted(shallow, key=lambda n: int(n.rsplit(".", 1)[1]))
        raise RuntimeError("Could not detect Timer decoder layers from module names")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run Timer forward pass with raw time-series input_ids.

        Args:
            x: Input tensor of shape (batch, seq_len) or (batch, seq_len, 1).

        Returns:
            Last hidden state tensor.

        Raises:
            RuntimeError: If model has not been loaded.
            ValueError: If input shape is unsupported.
            RuntimeError: If hidden states are unavailable in model output.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if x.ndim == 2:
            series = x
        elif x.ndim == 3 and x.shape[-1] == 1:
            series = x.squeeze(-1)
        else:
            raise ValueError(
                f"Expected input shape (batch, seq_len) or (batch, seq_len, 1), got {x.shape}.",
            )

        with torch.no_grad():
            series = series.to(self._device, dtype=torch.float32)
            output = self._model(
                input_ids=series,
                output_hidden_states=True,
                use_cache=False,
            )

            hidden_states = getattr(output, "hidden_states", None)
            if isinstance(hidden_states, tuple) and hidden_states:
                last_hidden_state = hidden_states[-1]
                if isinstance(last_hidden_state, torch.Tensor):
                    return last_hidden_state

            if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                return output[0]

        raise RuntimeError("Unable to extract hidden states from Timer forward output.")
