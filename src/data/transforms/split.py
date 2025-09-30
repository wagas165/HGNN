"""Dataset splitting utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class SplitConfig:
    strategy: str = "stratified"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    num_splits: int = 1
    random_state: int = 42


@dataclass
class DataSplits:
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor


def create_splits(labels: Optional[torch.Tensor], config: SplitConfig, num_nodes: int) -> DataSplits:
    if abs(config.train_ratio + config.val_ratio + config.test_ratio - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0")

    indices = np.arange(num_nodes)
    if labels is None or config.strategy != "stratified":
        np.random.seed(config.random_state)
        np.random.shuffle(indices)
        n_train = int(config.train_ratio * num_nodes)
        n_val = int(config.val_ratio * num_nodes)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=config.train_ratio,
            test_size=config.val_ratio + config.test_ratio,
            random_state=config.random_state,
        )
        train_idx, tmp_idx = next(splitter.split(indices, labels.cpu().numpy()))
        remaining_labels = labels[tmp_idx].cpu().numpy()
        remaining_indices = indices[tmp_idx]
        val_split = config.val_ratio / (config.val_ratio + config.test_ratio)
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=val_split,
            random_state=config.random_state,
        )
        val_rel_idx, test_rel_idx = next(splitter.split(remaining_indices, remaining_labels))
        val_idx = remaining_indices[val_rel_idx]
        test_idx = remaining_indices[test_rel_idx]

    return DataSplits(
        train_idx=torch.tensor(train_idx, dtype=torch.long),
        val_idx=torch.tensor(val_idx, dtype=torch.long),
        test_idx=torch.tensor(test_idx, dtype=torch.long),
    )
