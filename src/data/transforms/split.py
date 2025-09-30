"""Dataset splitting utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class SplitConfig:
    strategy: str = "stratified"
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    ood_ratio: float = 0.0
    num_splits: int = 1
    random_state: int = 42


@dataclass
class DataSplits:
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    test_idx: torch.Tensor
    ood_idx: Optional[torch.Tensor] = None


def create_splits(labels: Optional[torch.Tensor], config: SplitConfig, num_nodes: int) -> DataSplits:
    total_ratio = config.train_ratio + config.val_ratio + config.test_ratio + config.ood_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("Train/val/test/OOD ratios must sum to 1.0")

    indices = np.arange(num_nodes)
    if labels is None or config.strategy != "stratified":
        np.random.seed(config.random_state)
        np.random.shuffle(indices)
        n_train = int(config.train_ratio * num_nodes)
        n_val = int(config.val_ratio * num_nodes)
        n_test = int(config.test_ratio * num_nodes)

        start = 0
        train_idx = indices[start : start + n_train]
        start += n_train
        val_idx = indices[start : start + n_val]
        start += n_val
        test_idx = indices[start : start + n_test]
        start += n_test
        if config.ood_ratio > 0:
            ood_idx = indices[start:]
        else:
            ood_idx = None
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=config.train_ratio,
            test_size=config.val_ratio + config.test_ratio + config.ood_ratio,
            random_state=config.random_state,
        )
        train_idx, tmp_idx = next(splitter.split(indices, labels.cpu().numpy()))
        remaining_labels = labels[tmp_idx].cpu().numpy()
        remaining_indices = indices[tmp_idx]
        remaining_total = config.val_ratio + config.test_ratio + config.ood_ratio
        val_ratio = config.val_ratio / remaining_total if remaining_total > 0 else 0
        test_ratio = config.test_ratio / remaining_total if remaining_total > 0 else 0
        ood_ratio = config.ood_ratio / remaining_total if remaining_total > 0 else 0

        if ood_ratio > 0:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=ood_ratio,
                random_state=config.random_state,
            )
            ood_rel_idx, remaining_rel_idx = next(
                splitter.split(remaining_indices, remaining_labels)
            )
            ood_idx = remaining_indices[ood_rel_idx]
            remaining_indices = remaining_indices[remaining_rel_idx]
            remaining_labels = remaining_labels[remaining_rel_idx]
        else:
            ood_idx = None

        if (val_ratio + test_ratio) > 0 and remaining_indices.size > 0:
            val_split = val_ratio / (val_ratio + test_ratio)
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=val_split,
                random_state=config.random_state,
            )
            val_rel_idx, test_rel_idx = next(
                splitter.split(remaining_indices, remaining_labels)
            )
            val_idx = remaining_indices[val_rel_idx]
            test_idx = remaining_indices[test_rel_idx]
        else:
            val_idx = np.array([], dtype=np.int64)
            test_idx = remaining_indices

    return DataSplits(
        train_idx=torch.tensor(train_idx, dtype=torch.long),
        val_idx=torch.tensor(val_idx, dtype=torch.long),
        test_idx=torch.tensor(test_idx, dtype=torch.long),
        ood_idx=torch.tensor(ood_idx, dtype=torch.long) if ood_idx is not None else None,
    )
