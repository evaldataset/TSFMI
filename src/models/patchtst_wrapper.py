# pyright: reportMissingImports=false, reportImplicitOverride=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false
"""PatchTST model wrapper for probing experiments."""

from __future__ import annotations

import re
from typing import cast

import torch
import torch.nn as nn

from src.models.base import BaseModelWrapper
from src.utils.device import resolve_device

# Default: IBM Granite PatchTST-FM-r1, a general-purpose pretrained PatchTST-style
# foundation model whose disclosed training corpus (GIFT-Eval-Pretrain + KernelSynth +
# TSMixup restricted to non-GIFT-Eval datasets) excludes ETT, UCI Electricity and Jena
# Weather. See outputs/patchtst_replacement/README.md for the model-card evidence.
DEFAULT_PATCHTST_CHECKPOINT = "ibm-granite/granite-timeseries-patchtst-fm-r1"

# Legacy checkpoints, kept reachable via ``load(checkpoint=...)``. Both are the same
# supervised ETTh1-forecasting PatchTST (identical config, ``_name_or_path`` =
# namctin/patchtst_etth1_forecast); the namctin id now redirects to a gated repo.
# Do NOT use either on ETTh1 / ETT-family real-world probes (train-set leakage).
LEGACY_ETTH1_CHECKPOINT = "namctin/patchtst_etth1_forecast"
LEGACY_GRANITE_ETTH1_CHECKPOINT = "ibm-granite/granite-timeseries-patchtst"

_PATCHTST_FM_MODEL_TYPE = "patchtst_fm"


class PatchTSTWrapper(BaseModelWrapper):
    """Wrapper for PatchTST time series Transformers (Nie et al., ICLR 2023).

    Supports two checkpoint families, selected automatically from the checkpoint's
    ``config.json`` ``model_type``:

    * ``patchtst_fm`` (default, :data:`DEFAULT_PATCHTST_CHECKPOINT`): IBM Granite
      PatchTST-FM, loaded through ``tsfm_public`` (``granite-tsfm``). Non-overlapping
      16-step patches, 20 pre-norm Transformer blocks, d_model 1024. Hookable layers are
      ``backbone.blocks.{i}``; each emits ``(batch, n_context_patches, d_model)``.
    * ``patchtst`` (HuggingFace ``PatchTSTForPrediction``, e.g. the legacy ETTh1
      checkpoints or random-init with ``checkpoint=""``). Hookable layers are
      ``model.encoder.layers.{i}``.

    Architecture: Patch-based Transformer with channel independence. Each input
    subsequence is split into patches and processed by a standard Transformer encoder.

    References:
        Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting with
        Transformers." ICLR 2023.
        Wen et al. "Revisiting the Generic Transformer: Deconstructing a Strong Baseline
        for Time Series Foundation Models." arXiv:2602.06909, 2026 (PatchTST-FM).
    """

    def __init__(self) -> None:
        """Initialize wrapper state and default device."""
        self._model: nn.Module | None = None
        self._device: torch.device = resolve_device()
        self._is_fm: bool = False

    @property
    def is_patchtst_fm(self) -> bool:
        """Whether the loaded checkpoint is a PatchTST-FM (``tsfm_public``) model."""
        return self._is_fm

    def load(
        self,
        checkpoint: str = DEFAULT_PATCHTST_CHECKPOINT,
        *,
        device: torch.device | None = None,
        seq_len: int = 96,
    ) -> None:
        """Load PatchTST from HuggingFace or create with random-init config.

        Args:
            checkpoint: HuggingFace checkpoint ID, local path, or empty string for random-init.
                Defaults to :data:`DEFAULT_PATCHTST_CHECKPOINT` (PatchTST-FM-r1). Pass
                :data:`LEGACY_GRANITE_ETTH1_CHECKPOINT` to reproduce the original
                ``patchtst_pretrained`` results.
            device: Target device. If None, auto-resolve via resolve_device().
            seq_len: Context length for HF PatchTST checkpoints and random-init config.
                Ignored for PatchTST-FM, which accepts any input length up to its
                8192-step context.

        Raises:
            RuntimeError: If transformers (or granite-tsfm for PatchTST-FM) is not installed.
            ValueError: If the checkpoint is invalid.
            RuntimeError: If the loaded object is not an nn.Module.
        """
        self._device = device if device is not None else resolve_device()
        self._is_fm = False

        try:
            from transformers import PatchTSTConfig, PatchTSTForPrediction, PretrainedConfig
        except ImportError as exc:
            raise RuntimeError("transformers not installed. Run: pip install transformers") from exc

        model_type: str | None = None
        if checkpoint:
            try:
                config_dict, _ = PretrainedConfig.get_config_dict(checkpoint)
            except (ValueError, OSError) as exc:
                raise ValueError(f"Cannot load PatchTST checkpoint: {checkpoint}") from exc
            model_type = config_dict.get("model_type")

        if model_type == _PATCHTST_FM_MODEL_TYPE:
            model = self._load_patchtst_fm(checkpoint)
            self._is_fm = True
        elif checkpoint:
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

    @staticmethod
    def _load_patchtst_fm(checkpoint: str) -> nn.Module:
        """Load a PatchTST-FM checkpoint through ``tsfm_public``.

        Args:
            checkpoint: HuggingFace checkpoint ID or local path with ``model_type=patchtst_fm``.

        Returns:
            The loaded ``PatchTSTFMForPrediction`` module.

        Raises:
            RuntimeError: If ``granite-tsfm`` is not installed.
            ValueError: If the checkpoint cannot be loaded.
        """
        try:
            from tsfm_public.models.patchtst_fm import PatchTSTFMForPrediction
        except ImportError as exc:
            raise RuntimeError(
                "PatchTST-FM requires granite-tsfm. Install without touching the pinned "
                "torch/scikit-learn: pip install --no-deps granite-tsfm==0.3.9 datasets"
            ) from exc
        try:
            return PatchTSTFMForPrediction.from_pretrained(checkpoint)
        except (ValueError, OSError) as exc:
            raise ValueError(f"Cannot load PatchTST-FM checkpoint: {checkpoint}") from exc

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
        # Match only top-level encoder blocks: model.encoder.layers.{N} (HF PatchTST)
        # or backbone.blocks.{N} (PatchTST-FM).
        top_level = r"^backbone\.blocks\.\d+$" if self._is_fm else r"^model\.encoder\.layers\.\d+$"
        pattern = re.compile(top_level)
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
        if self._is_fm:
            return self._forward_fm(x_3d)

        with torch.no_grad():
            output = self._model(past_values=x_3d)

        return self._extract_tensor_output(output)

    def _forward_fm(self, x_3d: torch.Tensor) -> torch.Tensor:
        """Encode a univariate context with PatchTST-FM, without forecast slots.

        The series is left-padded with its own mean to the model's full context length and
        the pad region is flagged in ``pad_mask`` (exactly how PatchTST-FM treats short
        inputs). No forecast horizon is appended (``pred_mask`` is all False), and the
        backbone's pruning drops the leading all-pad patches, so every hooked
        ``backbone.blocks.{i}`` activation covers only the observed input:
        ``(batch, ceil(seq_len / 16), d_model)``. Normalization statistics are computed
        from observed values only.

        Args:
            x_3d: Input tensor of shape (batch, seq_len, 1) on the model device.

        Returns:
            Reconstruction quantiles in the model's normalized (RevIN + asinh) space, shape
            (batch, ceil(seq_len / 16) * 16, num_quantile).

        Raises:
            ValueError: If the input is multivariate or longer than the model context.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if x_3d.shape[-1] != 1:
            raise ValueError(
                "PatchTST-FM is channel-independent; pass univariate input "
                f"(batch, seq_len) or (batch, seq_len, 1), got {tuple(x_3d.shape)}"
            )

        config = self._model.config
        max_len = int(config.context_length)
        series = x_3d[..., 0].float()
        batch, seq_len = series.shape
        if seq_len > max_len:
            raise ValueError(f"seq_len {seq_len} exceeds PatchTST-FM context {max_len}")

        observed = ~torch.isnan(series)
        series_mean = torch.nanmean(series, dim=1, keepdim=True)
        series = torch.where(observed, series, series_mean.expand_as(series))
        left = max_len - seq_len
        inputs = torch.cat([series_mean.expand(batch, left), series], dim=1)
        pad_mask = torch.zeros_like(inputs, dtype=torch.bool)
        pad_mask[:, :left] = True
        miss_mask = torch.zeros_like(inputs, dtype=torch.bool)
        miss_mask[:, left:] = ~observed
        pred_mask = torch.zeros_like(inputs, dtype=torch.bool)

        backbone = cast(nn.Module, self._model.backbone)
        with torch.no_grad():
            output = backbone(
                inputs=inputs,
                pred_mask=pred_mask,
                miss_mask=miss_mask,
                pad_mask=pad_mask,
                return_loss=False,
                context_length=seq_len,
            )
        quantiles = cast(torch.Tensor, output.quantile_outputs)
        n_kept = quantiles.shape[1]
        return quantiles[:, n_kept - self._n_output_steps(seq_len, int(config.d_patch)) :]

    @staticmethod
    def _n_output_steps(seq_len: int, d_patch: int) -> int:
        """Number of reconstructed time steps covering the observed input patches."""
        return -(-seq_len // d_patch) * d_patch

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
