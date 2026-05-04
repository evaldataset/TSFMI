"""Shared pytest fixtures for TSFMI test suite."""

import pytest
import torch


@pytest.fixture
def synthetic_representations():
    """100 samples of 768-dim frozen representations (like MOMENT's hidden dim)."""
    torch.manual_seed(42)
    return torch.randn(100, 768)


@pytest.fixture
def binary_labels():
    """100 binary labels (0 or 1)."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (100,))


@pytest.fixture
def multiclass_labels():
    """100 labels in {0,1,2}."""
    torch.manual_seed(42)
    return torch.randint(0, 3, (100,))


@pytest.fixture
def float_labels():
    """100 float regression targets."""
    torch.manual_seed(42)
    return torch.randn(100)


@pytest.fixture
def tiny_representations():
    """50 samples of 64-dim representations (faster for training tests)."""
    torch.manual_seed(42)
    return torch.randn(50, 64)


@pytest.fixture
def tiny_labels():
    """50 binary labels for tiny_representations."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (50,))
