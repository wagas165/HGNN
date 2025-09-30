"""Loader for the StackOverflow Answers hypergraph dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base import HypergraphData
from .generic import EdgeListHypergraphConfig, EdgeListHypergraphLoader


@dataclass
class StackOverflowAnswersConfig(EdgeListHypergraphConfig):
    """Configuration defaults for StackOverflow Answers."""

    comment_prefix: str = "%"
    min_cardinality: int = 2
    label_file: Optional[str] = "stackoverflow-answers-node-labels.txt"


class StackOverflowAnswersLoader(EdgeListHypergraphLoader):
    """Loader that knows how to interpret StackOverflow-specific label files."""

    def __init__(self, root: str, config: StackOverflowAnswersConfig) -> None:
        super().__init__(root, config)

    def load(self) -> HypergraphData:  # type: ignore[override]
        # Bypass default label loading to support StackOverflow's node-id-prefixed format
        label_file = self.config.label_file
        self.config.label_file = None
        data = super().load()
        self.config.label_file = label_file
        if label_file:
            labels = self._load_labels(self.root / label_file, data.num_nodes)
            return HypergraphData(
                num_nodes=data.num_nodes,
                incidence=data.incidence,
                edge_weights=data.edge_weights,
                node_features=data.node_features,
                labels=labels,
                timestamps=data.timestamps,
                metadata=data.metadata,
            )
        return data

    def _load_labels(self, path: Path, num_nodes: int) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Required StackOverflow label file missing: {path}")

        labels = np.full(num_nodes, fill_value=-1, dtype=np.int64)
        with path.open("r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith(self.config.comment_prefix):
                    continue
                tokens = line.split()
                try:
                    node_id = int(tokens[0])
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid label line {line_num} in {path}: {raw_line!r}"
                    ) from exc
                if node_id < 0 or node_id >= num_nodes:
                    raise ValueError(
                        f"Node id {node_id} in {path} exceeds inferred node count {num_nodes}"
                    )
                if len(tokens) < 2:
                    raise ValueError(
                        f"Label line {line_num} in {path} does not specify a label id"
                    )
                try:
                    label_id = int(tokens[1])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid label id on line {line_num} in {path}: {tokens[1]!r}"
                    ) from exc
                labels[node_id] = label_id

        if np.any(labels < 0):
            raise ValueError(
                "StackOverflow label file is missing assignments for some nodes"
            )
        return torch.tensor(labels, dtype=torch.long)
