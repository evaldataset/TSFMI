"""Re-initialise a loaded wrapper's weights to separate architecture from pretraining.

The random-initialisation control keeps everything a wrapper does to its input (tokenisation,
scaling, patching, the forward graph, the hooked layer names) and replaces every learned weight with
a draw from a standard initialiser. A probe score that survives re-initialisation is a property of
the architecture acting as a random feature map, not of pretraining.

Weights are reset in two steps: every submodule that defines ``reset_parameters`` is reset with
PyTorch's default initialiser; any parameter left unchanged afterwards (custom norms, learned
scales, special tokens, positional tables held as bare parameters) is set to one for 1-D norm/scale
weights, zero for other 1-D parameters, and N(0, 0.02^2) otherwise.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.base import BaseModelWrapper

# Wrapper attributes that hold modules used in forward() besides ``wrapper.model``.
_EXTRA_MODULE_ATTRS = ("_prediction_model", "_pipeline", "_patch_embedding", "_module")


def wrapper_modules(wrapper: BaseModelWrapper) -> list[nn.Module]:
    """Return the distinct modules a wrapper's forward pass runs through."""
    found: list[nn.Module] = [wrapper.model]
    for attr in _EXTRA_MODULE_ATTRS:
        obj = getattr(wrapper, attr, None)
        if isinstance(obj, nn.Module):
            found.append(obj)
        inner = getattr(obj, "model", None)  # e.g. a pipeline object holding the network
        if obj is not None and not isinstance(obj, nn.Module) and isinstance(inner, nn.Module):
            found.append(inner)
    unique: dict[int, nn.Module] = {}
    for m in found:
        unique.setdefault(id(m), m)
    return list(unique.values())


def _named_parameters(modules: list[nn.Module]) -> dict[int, tuple[str, nn.Parameter]]:
    params: dict[int, tuple[str, nn.Parameter]] = {}
    for root in modules:
        for name, p in root.named_parameters():
            params.setdefault(id(p), (name, p))
    return params


@torch.no_grad()
def reinitialize_(wrapper: BaseModelWrapper, seed: int) -> dict[str, int]:
    """Replace every learned weight of ``wrapper`` with a random draw, in place.

    Args:
        wrapper: A loaded wrapper.
        seed: Seed of the initialiser.

    Returns:
        Counts of parameter tensors reset by ``reset_parameters``, set by the fallback rule, and
        the total, for the run metadata.

    Raises:
        RuntimeError: If any parameter tensor still equals its pretrained value.
    """
    modules = wrapper_modules(wrapper)
    params = _named_parameters(modules)
    before = {k: p.detach().clone() for k, (_, p) in params.items()}
    torch.manual_seed(seed)
    for root in modules:
        for m in root.modules():
            reset = getattr(m, "reset_parameters", None)
            if callable(reset):
                reset()
    n_fallback = 0
    for k, (name, p) in params.items():
        if not torch.equal(p, before[k]):
            continue
        n_fallback += 1
        lowered = name.lower()
        if p.ndim <= 1 and any(t in lowered for t in ("norm", "scale", "ln")):
            p.fill_(1.0)
        elif p.ndim <= 1:
            p.zero_()
        else:
            p.normal_(0.0, 0.02)
    # 1-D parameters may legitimately land on their pretrained value (a norm weight of ones).
    unchanged = [
        name for k, (name, p) in params.items() if p.ndim >= 2 and torch.equal(p, before[k])
    ]
    if unchanged:
        raise RuntimeError(f"parameters not re-initialised: {unchanged[:5]}")
    return {
        "n_parameter_tensors": len(params),
        "n_reset_by_module": len(params) - n_fallback,
        "n_reset_by_fallback": n_fallback,
    }
