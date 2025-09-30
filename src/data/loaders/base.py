"""Dataset loader interfaces."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch


@dataclass
class HypergraphData:
    num_nodes: int
    incidence: torch.Tensor  # dense or sparse N x E
    edge_weights: torch.Tensor
    node_features: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]
    timestamps: Optional[torch.Tensor]
    metadata: Dict[str, object]


class HypergraphDatasetLoader(ABC):
    """Abstract loader for hypergraph datasets."""

    def __init__(self, root: str) -> None:
        self.root = Path(root)

    @abstractmethod
    def load(self) -> HypergraphData:
        """Load dataset into memory."""

    def _check_exists(self, *filenames: str) -> None:
        for fname in filenames:
            file_path = self.root / fname
            if not file_path.exists():
                raise FileNotFoundError(f"Required dataset file not found: {file_path}")
