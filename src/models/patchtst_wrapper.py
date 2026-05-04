# pyright: reportMissingImports=false, reportImplicitOverride=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false
"""PatchTST model wrapper for probing experiments."""

from __future__ import annotations

import re
from typing import cast

import torch
import torch.nn as nn

from src.models.base import BaseModelWrapper
from src.utils.device import resolve_device


class PatchTSTWrapper(BaseModelWrapper):
    """Wrapper for PatchTST time series Transformer (Nie et al., ICLR 2023).

    Loads PatchTST from HuggingFace transformers library. Freezes all parameters
    and exposes encoder transformer blocks for hook-based activation extraction.

    Architecture: Patch-based Transformer with channel independence. Each input
    subsequence is split into patches and processed by a standard Transformer encoder.

    Reference: Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting with
    Transformers." ICLR 2023.
    """

    def __init__(self) -> None:
        """Initialize wrapper state and default device."""
        self._model: nn.Module | None = None
        self._device: torch.device = resolve_device()

    def load(
        self,
        checkpoint: str = "namctin/patchtst_etth1_forecast",
        *,
        device: torch.device | None = None,
        seq_len: int = 96,
    ) -> None:
        """Load PatchTST from HuggingFace or create with random-init config.

        Args:
            checkpoint: HuggingFace checkpoint ID, local path, or empty string for random-init.
            device: Target device. If None, auto-resolve via resolve_device().
            seq_len: Context length (only used for random-init config).

        Raises:
            RuntimeError: If transformers is not installed.
            ValueError: If the checkpoint is invalid.
            RuntimeError: If the loaded object is not an nn.Module.
        """
        self._device = device if device is not None else resolve_device()

        try:
            from transformers import PatchTSTConfig, PatchTSTForPrediction
        except ImportError as exc:
            raise RuntimeError("transformers not installed. Run: pip install transformers") from exc

        if checkpoint:
            try:
                model = PatchTSTForPrediction.from_pretrained(
                    checkpoint,
                    context_length=seq_len,
                )
            except (ValueError, OSError) as exc:
                raise ValueError(f"Cannot load PatchTST checkpoint: {checkpoint}") from exc

            # Some pretrained PatchTST checkpoints (e.g., ibm-granite) are trained with
            # multivariate inputs. For our probing setup we always feed univariate series,
            # so adapt channel-dependent modules to 1 input channel before moving devices.
            if model.config.num_input_channels != 1:
                import torch.nn as nn_fix

                d_model = model.config.d_model
                model.config.num_input_channels = 1
                model.model.encoder.embedder.num_input_channels = 1
                model.model.encoder.positional_encoder.num_input_channels = 1
                if hasattr(model.model.encoder.positional_encoder, "cls_token"):
                    model.model.encoder.positional_encoder.cls_token = nn_fix.Parameter(
                        torch.zeros(1, 1, 1, d_model),
                    )
                model.head.num_input_channels = 1
        else:
            # Random-init with architecture matching original paper
            config = PatchTSTConfig(
                num_input_channels=1,
                context_length=seq_len,
                patch_length=16,
                stride=8,
                d_model=128,
                num_attention_heads=4,
                num_hidden_layers=3,
                prediction_length=24,
                ffn_dim=256,
            )
            model = PatchTSTForPrediction(config)

        if not isinstance(model, nn.Module):
            raise RuntimeError("Failed to load PatchTST nn.Module")

        # cast: transformers 5.x PreTrainedModel.to() overload typing ambiguity
        self._model = cast(nn.Module, model).to(self._device)
        self.freeze()

    def freeze(self) -> None:
        """Freeze all model parameters and set eval mode."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for parameter in self._model.parameters():
            parameter.requires_grad = False
        self._model.eval()

    def get_layer_names(self) -> list[str]:
        """Return PatchTST encoder layer names for HookManager."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        names: list[str] = []
        # Match only top-level encoder blocks: model.encoder.layers.{N}
        pattern = re.compile(r"^model\.encoder\.layers\.\d+$")
        for name, _ in self._model.named_modules():
            if pattern.match(name):
                names.append(name)

        if not names:
            pattern = re.compile(r"(^|\.)encoder\.layers\.\d+$")
            names = [name for name, _ in self._model.named_modules() if pattern.search(name)]

        return sorted(set(names), key=self._layer_sort_key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a frozen forward pass with PatchTST-compatible input layout.

        Args:
            x: Input tensor of shape (batch, seq_len) or (batch, seq_len, channels).

        Returns:
            PatchTST output tensor.

        Raises:
            RuntimeError: If model has not been loaded.
            ValueError: If input shape is not 2D or 3D.
            RuntimeError: If no tensor output is found in model return object.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if x.ndim == 2:
            x_3d = x.unsqueeze(-1)
        elif x.ndim == 3:
            x_3d = x
        else:
            raise ValueError(
                f"Expected input shape (batch, seq_len) or (batch, seq_len, channels),"
                f" got {x.shape}"
            )

        x_3d = x_3d.to(self._device)
        with torch.no_grad():
            output = self._model(past_values=x_3d)

        return self._extract_tensor_output(output)

    @staticmethod
    def _layer_sort_key(layer_name: str) -> tuple[int, str]:
        """Sort layers by final numeric suffix when present."""
        tail = layer_name.rsplit(".", maxsplit=1)[-1]
        return (int(tail), layer_name) if tail.isdigit() else (-1, layer_name)

    @staticmethod
    def _extract_tensor_output(output: object) -> torch.Tensor:
        """Extract a tensor from common HuggingFace model output containers."""
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        if isinstance(output, dict):
            for key in (
                "prediction_output",
                "prediction_outputs",
                "last_hidden_state",
                "logits",
                "predictions",
                "output",
            ):
                value = output.get(key)
                if isinstance(value, torch.Tensor):
                    return value

        for attr in (
            "prediction_output",
            "prediction_outputs",
            "last_hidden_state",
            "logits",
            "predictions",
            "output",
        ):
            value = getattr(output, attr, None)
            if isinstance(value, torch.Tensor):
                return value

        raise RuntimeError("Unable to extract tensor output from PatchTST forward pass")
