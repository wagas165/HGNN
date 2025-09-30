"""Training callbacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EarlyStoppingState:
    patience: int
    best_metric: float = float("-inf")
    counter: int = 0
    best_state_dict: Optional[dict] = None

    def update(self, metric: float, model: torch.nn.Module) -> bool:
        improved = metric > self.best_metric
        if improved:
            self.best_metric = metric
            self.counter = 0
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return improved

    def should_stop(self) -> bool:
        return self.counter >= self.patience

    def restore_best(self, model: torch.nn.Module) -> None:
        if self.best_state_dict:
            model.load_state_dict(self.best_state_dict)
