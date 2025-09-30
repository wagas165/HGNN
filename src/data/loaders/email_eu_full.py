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

        cardinalities = self._read_cardinalities(self.root / self.config.vertices_file)
        flat_simplices = self._read_flat_simplices(self.root / self.config.simplices_file)

        total_entries = sum(cardinalities)
        if total_entries != len(flat_simplices):
            raise ValueError(
                "Total simplex entries do not match cardinality counts: "
                f"expected {total_entries}, got {len(flat_simplices)}"
            )

        simplices = self._slice_simplices(cardinalities, flat_simplices)
        num_nodes = self._infer_num_nodes(flat_simplices)
        times = None
        if self.config.times_file:
            times = self._read_times(self.root / self.config.times_file, len(simplices))

        incidence = self._build_incidence(num_nodes, simplices)
        edge_weights = torch.ones(incidence.shape[1], dtype=torch.float32)

        node_features = self._load_features(num_nodes)
        labels = self._load_labels(num_nodes)

        metadata = {
            "num_hyperedges": incidence.shape[1],
            "avg_cardinality": float(np.mean(cardinalities)) if cardinalities else 0.0,
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

    def _read_cardinalities(self, path: Path) -> List[int]:
        with path.open("r", encoding="utf-8") as f:
            return [int(line.strip()) for line in f if line.strip()]

    def _read_flat_simplices(self, path: Path) -> List[int]:
        values: List[int] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                values.extend(int(tok) for tok in line.split())
        return values

    def _slice_simplices(
        self, cardinalities: List[int], flat_simplices: List[int]
    ) -> List[List[int]]:
        simplices: List[List[int]] = []
        index = 0
        for count in cardinalities:
            next_index = index + count
            simplices.append(flat_simplices[index:next_index])
            index = next_index
        return simplices

    def _infer_num_nodes(self, flat_simplices: List[int]) -> int:
        if not flat_simplices:
            return 0
        return max(flat_simplices) + 1

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
