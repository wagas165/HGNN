"""Optimizer utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 5e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    lbfgs_history_size: int = 10
    lbfgs_line_search: str = "strong_wolfe"


def build_adam(parameters: Iterable[nn.Parameter], config: OptimizerConfig) -> torch.optim.Adam:
    return torch.optim.Adam(
        parameters,
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )


def build_lbfgs(parameters: Iterable[nn.Parameter], config: OptimizerConfig) -> torch.optim.LBFGS:
    return torch.optim.LBFGS(
        parameters,
        lr=1.0,
        history_size=config.lbfgs_history_size,
        line_search_fn=config.lbfgs_line_search,
        max_iter=20,
    )
