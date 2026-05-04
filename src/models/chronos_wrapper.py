# pyright: reportMissingImports=false, reportImplicitOverride=false
"""Chronos-Bolt wrapper for probing experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn

from src.models.base import BaseModelWrapper
from src.utils.device import resolve_device


@runtime_checkable
class _ChronosConfigProtocol(Protocol):
    context_length: int
    use_reg_token: bool


@runtime_checkable
class _ChronosT5ConfigProtocol(Protocol):
    reg_token_id: int


@runtime_checkable
class _ChronosEncoderProtocol(Protocol):
    block: Sequence[nn.Module]

    def __call__(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> object: ...


@runtime_checkable
class _ChronosModelProtocol(Protocol):
    dtype: torch.dtype
    encoder: _ChronosEncoderProtocol
    chronos_config: _ChronosConfigProtocol
    config: _ChronosT5ConfigProtocol

    def instance_norm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def patch(self, x: torch.Tensor) -> torch.Tensor: ...
    def input_patch_embedding(self, x: torch.Tensor) -> torch.Tensor: ...
    def shared(self, x: torch.Tensor) -> torch.Tensor: ...


class ChronosBoltWrapper(BaseModelWrapper):
    """Wrapper for Chronos-Bolt encoder representations.

    This wrapper loads a ``ChronosBoltPipeline`` and exposes the internal
    ``ChronosBoltModelForForecasting`` encoder blocks for hook-based activation
    extraction.
    """

    def __init__(self) -> None:
        """Initialize wrapper state and default device."""
        self._model: nn.Module | None = None
        self._pipeline: object | None = None
        self._device: torch.device = resolve_device()

    def load(
        self,
        checkpoint: str = "amazon/chronos-bolt-small",
        *,
        device: torch.device | None = None,
    ) -> None:
        """Load Chronos-Bolt from pretrained checkpoint and freeze parameters."""
        self._device = device if device is not None else resolve_device()

        try:
            from chronos import ChronosBoltPipeline
        except ImportError as exc:
            raise RuntimeError(
                "chronos-forecasting not installed. Run: pip install chronos-forecasting"
            ) from exc

        pipeline = ChronosBoltPipeline.from_pretrained(
            checkpoint,
            device_map=str(self._device),
        )
        model = pipeline.model
        if not isinstance(model, nn.Module):
            raise RuntimeError("Failed to resolve Chronos-Bolt nn.Module from pipeline.model")

        self._pipeline = pipeline
        self._model = model
        self.freeze()

    def freeze(self) -> None:
        """Freeze all model parameters and set eval mode."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for parameter in self._model.parameters():
            parameter.requires_grad = False
        self._model.eval()

    def _chronos_model(self) -> _ChronosModelProtocol:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if not isinstance(self._model, _ChronosModelProtocol):
            raise RuntimeError("Chronos-Bolt model does not expose the expected encoder API.")
        return self._model

    def get_layer_names(self) -> list[str]:
        """Return Chronos-Bolt encoder block names for HookManager."""
        model = self._chronos_model()
        num_blocks = len(model.encoder.block)
        return [f"encoder.block.{i}" for i in range(num_blocks)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run Chronos-Bolt encoder forward pass for representation extraction."""
        if self._pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load() first.")
        model = self._chronos_model()

        if x.ndim == 2:
            context = x
        elif x.ndim == 3 and x.shape[-1] == 1:
            context = x.squeeze(-1)
        else:
            raise ValueError(
                f"Expected input shape (batch, seq_len) or (batch, seq_len, 1), got {x.shape}",
            )

        with torch.no_grad():
            context = context.to(self._device)

            if context.shape[-1] > model.chronos_config.context_length:
                context = context[..., -model.chronos_config.context_length :]

            mask = torch.isnan(context).logical_not().to(context.dtype)

            context, _ = model.instance_norm(context)
            context = context.to(model.dtype)
            mask = mask.to(model.dtype)

            patched_context = model.patch(context)
            patched_mask = torch.nan_to_num(model.patch(mask), nan=0.0)
            patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
            patched_context = torch.cat([patched_context, patched_mask], dim=-1)

            attention_mask = patched_mask.sum(dim=-1) > 0
            embeddings = model.input_patch_embedding(patched_context)

            use_reg_token = model.chronos_config.use_reg_token
            if use_reg_token:
                reg_input_ids = torch.full(
                    (embeddings.size(0), 1),
                    model.config.reg_token_id,
                    device=embeddings.device,
                    dtype=torch.long,
                )
                reg_embeds = model.shared(reg_input_ids)
                embeddings = torch.cat([embeddings, reg_embeds], dim=-2)
                attention_mask = torch.cat(
                    [
                        attention_mask.to(model.dtype),
                        torch.ones_like(reg_input_ids).to(model.dtype),
                    ],
                    dim=-1,
                )

            encoder_outputs = model.encoder(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
            )
            if isinstance(encoder_outputs, tuple) and encoder_outputs:
                first = encoder_outputs[0]
                if isinstance(first, torch.Tensor):
                    return first

            last_hidden_state = getattr(encoder_outputs, "last_hidden_state", None)
            if isinstance(last_hidden_state, torch.Tensor):
                return last_hidden_state

        raise RuntimeError("Unable to extract encoder hidden states from Chronos-Bolt output.")

    @property
    def model(self) -> nn.Module:
        """Return underlying Chronos-Bolt nn.Module for hook registration."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model
