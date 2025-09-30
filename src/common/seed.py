"""Random seed utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class SeedConfig:
    value: int = 42
    deterministic_cudnn: bool = False


def set_seed(config: SeedConfig) -> None:
    """Set random seeds for python, numpy and torch."""
    seed = config.value
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if config.deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "auto") -> torch.device:
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)
