from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.base import BaseModelWrapper


class ConcreteModel(BaseModelWrapper):
    def load(self, checkpoint: str, *, device: torch.device | None = None) -> None:
        del checkpoint
        del device
        self._model = nn.Linear(4, 2)

    def freeze(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        for parameter in self._model.parameters():
            parameter.requires_grad = False

    def get_layer_names(self) -> list[str]:
        if self._model is None:
            return []
        return [name for name, _ in self._model.named_modules()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def test_model_property_raises_before_load() -> None:
    wrapper = ConcreteModel()
    with pytest.raises(RuntimeError):
        _ = wrapper.model


def test_model_property_returns_module_after_load() -> None:
    wrapper = ConcreteModel()
    wrapper.load("checkpoint")
    assert isinstance(wrapper.model, nn.Module)


def test_is_frozen_raises_before_load() -> None:
    wrapper = ConcreteModel()
    with pytest.raises(RuntimeError, match="not loaded"):
        wrapper.is_frozen()


def test_is_frozen_true_after_load_and_freeze() -> None:
    wrapper = ConcreteModel()
    wrapper.load("checkpoint")
    wrapper.freeze()
    assert wrapper.is_frozen() is True


def test_is_frozen_false_with_trainable_parameter() -> None:
    wrapper = ConcreteModel()
    wrapper.load("checkpoint")
    wrapper.freeze()
    first_parameter = next(wrapper.model.parameters())
    first_parameter.requires_grad = True
    assert wrapper.is_frozen() is False
