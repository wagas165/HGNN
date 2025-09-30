"""Loader for the email-Eu-full hypergraph dataset."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .base import HypergraphData, HypergraphDatasetLoader


@dataclass
class EmailEuFullConfig:
    vertices_file: str
    simplices_file: str
    times_file: Optional[str] = None
    feature_cache: Optional[str] = None
    label_file: Optional[str] = None


class EmailEuFullLoader(HypergraphDatasetLoader):
    def __init__(self, root: str, config: EmailEuFullConfig) -> None:
        super().__init__(root)
        self.config = config

    def load(self) -> HypergraphData:
        files = [self.config.vertices_file, self.config.simplices_file]
        if self.config.times_file:
            files.append(self.config.times_file)
        self._check_exists(*files)

        vertices = self._read_vertices(self.root / self.config.vertices_file)
        simplices = self._read_simplices(self.root / self.config.simplices_file)
        times = None
        if self.config.times_file:
            times = self._read_times(self.root / self.config.times_file, len(simplices))

        num_nodes = len(vertices)
        incidence = self._build_incidence(num_nodes, simplices)
        edge_weights = torch.ones(incidence.shape[1], dtype=torch.float32)

        node_features = self._load_features(num_nodes)
        labels = self._load_labels(num_nodes)

        metadata = {
            "num_hyperedges": incidence.shape[1],
            "avg_cardinality": float(np.mean([len(s) for s in simplices])) if simplices else 0.0,
        }

        return HypergraphData(
            num_nodes=num_nodes,
            incidence=incidence,
            edge_weights=edge_weights,
            node_features=node_features,
            labels=labels,
            timestamps=times,
            metadata=metadata,
        )

    def _read_vertices(self, path: Path) -> List[int]:
        with path.open("r", encoding="utf-8") as f:
            vertices = [int(line.strip()) for line in f if line.strip()]
        if vertices and min(vertices) != 0:
            raise ValueError("email-Eu-full vertices are expected to start from 0")
        if vertices != list(range(len(vertices))):
            raise ValueError("Vertices must form a contiguous range starting at 0")
        return vertices

    def _read_simplices(self, path: Path) -> List[List[int]]:
        simplices: List[List[int]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = [int(tok) for tok in line.strip().split() if tok]
                if len(parts) >= 2:
                    simplices.append(parts)
        return simplices

    def _read_times(self, path: Path, expected: int) -> torch.Tensor:
        values = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values.append(float(line))
        if len(values) != expected:
            raise ValueError(
                f"Times file length {len(values)} does not match number of simplices {expected}"
            )
        return torch.tensor(values, dtype=torch.float32)

    def _build_incidence(self, num_nodes: int, simplices: List[List[int]]) -> torch.Tensor:
        num_edges = len(simplices)
        incidence = torch.zeros((num_nodes, num_edges), dtype=torch.float32)
        for e_idx, simplex in enumerate(simplices):
            for v in simplex:
                incidence[v, e_idx] = 1.0
        return incidence

    def _load_features(self, num_nodes: int) -> torch.Tensor:
        if self.config.feature_cache:
            cache_path = self.root / self.config.feature_cache
            if cache_path.exists():
                data = np.load(cache_path)
                if "X" in data:
                    return torch.tensor(data["X"], dtype=torch.float32)
        return torch.zeros((num_nodes, 1), dtype=torch.float32)

    def _load_labels(self, num_nodes: int) -> Optional[torch.Tensor]:
        if self.config.label_file:
            label_path = self.root / self.config.label_file
            if label_path.exists():
                labels = np.load(label_path)
                return torch.tensor(labels, dtype=torch.long)
        return None
