"""RNG seeding utilities for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Set all RNG seeds for reproducibility.

    Sets: torch, torch.cuda, numpy, random, and Python hash seed.
    Also configures CuDNN for deterministic behavior.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
