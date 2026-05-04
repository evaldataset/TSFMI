# pyright: reportMissingImports=false
"""PyTorch hook manager for extracting intermediate layer activations."""

from __future__ import annotations

import torch
import torch.nn as nn


class HookManager:
    """Context manager for extracting intermediate layer activations via PyTorch hooks.

    Registers forward hooks on named modules, buffers their outputs, and guarantees
    hook removal on context exit to prevent memory leaks.

    Example:
        model = SomeModel()
        with HookManager(model, layer_names=["encoder.layer.0", "encoder.layer.6"]) as hm:
            _ = model(inputs)
            activations = hm.get_activations()  # dict[str, torch.Tensor]
    """

    def __init__(self, model: nn.Module, layer_names: list[str]) -> None:
        """Initialize HookManager.

        Args:
            model: The frozen model to extract activations from.
            layer_names: Names of modules to hook, as returned by model.named_modules().

        Raises:
            ValueError: If any layer name is not found in the model.
        """
        self._model = model
        self._layer_names = layer_names
        self._activations: dict[str, torch.Tensor] = {}
        self._handles: list = []
        self._inside_context: bool = False

        available_modules = dict(model.named_modules())
        for name in layer_names:
            if name not in available_modules:
                raise ValueError(
                    f"Layer '{name}' not found in model. "
                    f"Available: {list(available_modules.keys())}"
                )
        self._target_modules = {name: available_modules[name] for name in layer_names}

    def __enter__(self) -> HookManager:
        """Register forward hooks on specified layers."""
        self._inside_context = True
        self._activations.clear()

        for name, module in self._target_modules.items():
            handle = module.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

        return self

    def __exit__(self, *args: object) -> None:
        """Remove all hooks regardless of exceptions."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._inside_context = False

    def get_activations(self) -> dict[str, torch.Tensor]:
        """Return copy of buffered activations.

        Returns:
            Dictionary mapping layer names to their output tensors.

        Raises:
            RuntimeError: If called outside the context manager.
        """
        if not self._inside_context:
            raise RuntimeError("HookManager must be used as context manager")
        return dict(self._activations)

    def clear(self) -> None:
        """Clear activation buffer while keeping hooks registered."""
        self._activations.clear()

    def _make_hook(self, layer_name: str):
        """Create a forward hook callback that captures output for the given layer.

        Args:
            layer_name: Name to key the stored activation under.

        Returns:
            Hook callable compatible with register_forward_hook.
        """

        def hook(
            module: nn.Module,
            input: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            if isinstance(output, tuple):
                tensor_output = next(
                    (value for value in output if isinstance(value, torch.Tensor)),
                    None,
                )
                if tensor_output is None:
                    raise RuntimeError(
                        f"Hook output for layer '{layer_name}' does not contain a tensor",
                    )
                self._activations[layer_name] = tensor_output.detach()
            else:
                self._activations[layer_name] = output.detach()

        return hook
