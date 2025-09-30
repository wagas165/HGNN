"""Deterministic feature engineering for hypergraphs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class DeterministicFeatureConfig:
    spectral_topk: int = 32
    use_spectral: bool = True
    use_hodge: bool = False
    cache_dir: Optional[str] = None


class DeterministicFeatureBank:
    def __init__(self, config: DeterministicFeatureConfig) -> None:
        self.config = config

    def __call__(self, incidence: Tensor, edge_weights: Tensor) -> Tensor:
        cache = self._load_cache(incidence)
        if cache is not None:
            return cache

        features = [self._node_degree(incidence, edge_weights), self._avg_edge_cardinality(incidence)]
        if self.config.use_spectral:
            features.append(self._spectral_embeddings(incidence, edge_weights, self.config.spectral_topk))
        if self.config.use_hodge:
            features.append(self._hodge_proxy(incidence))

        combined = torch.cat(features, dim=1)
        combined = self._standardize(combined)
        self._save_cache(incidence, combined)
        return combined

    def _node_degree(self, incidence: Tensor, edge_weights: Tensor) -> Tensor:
        degrees = torch.matmul(incidence, edge_weights.unsqueeze(1))
        return degrees

    def _avg_edge_cardinality(self, incidence: Tensor) -> Tensor:
        edge_sizes = incidence.sum(dim=0, keepdim=True)
        node_edge_counts = (incidence > 0).sum(dim=1, keepdim=True).clamp(min=1)
        avg_cardinality = torch.matmul(incidence, 1.0 / edge_sizes.t()) / node_edge_counts
        return avg_cardinality

    def _spectral_embeddings(self, incidence: Tensor, edge_weights: Tensor, topk: int) -> Tensor:
        theta = self._normalized_operator(incidence, edge_weights)
        evals, evecs = torch.linalg.eigh(theta + 1e-6 * torch.eye(theta.shape[0], device=theta.device))
        k = min(topk, evecs.shape[1])
        return evecs[:, -k:]

    def _hodge_proxy(self, incidence: Tensor) -> Tensor:
        edge_sizes = incidence.sum(dim=0)
        normalized = incidence / edge_sizes.clamp(min=1.0)
        return normalized @ normalized.t()

    def _standardize(self, features: Tensor) -> Tensor:
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        standardized = (features - mean) / std
        return torch.nan_to_num(standardized)

    def _normalized_operator(self, incidence: Tensor, edge_weights: Tensor) -> Tensor:
        node_degree = torch.matmul(incidence, edge_weights.unsqueeze(1)).flatten()
        edge_degree = incidence.sum(dim=0)
        d_v_inv_sqrt = torch.diag(torch.pow(node_degree.clamp(min=1e-6), -0.5))
        d_e_inv = torch.diag(torch.pow(edge_degree.clamp(min=1.0), -1.0))
        w_diag = torch.diag(edge_weights)
        theta = d_v_inv_sqrt @ incidence @ w_diag @ d_e_inv @ incidence.t() @ d_v_inv_sqrt
        return theta

    def _cache_path(self, incidence: Tensor) -> Optional[Path]:
        if not self.config.cache_dir:
            return None
        num_nodes, num_edges = incidence.shape
        identifier = f"det_{num_nodes}_{num_edges}.npz"
        path = Path(self.config.cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path / identifier

    def _load_cache(self, incidence: Tensor) -> Optional[Tensor]:
        cache_path = self._cache_path(incidence)
        if cache_path and cache_path.exists():
            data = np.load(cache_path)
            return torch.tensor(data["features"], dtype=torch.float32)
        return None

    def _save_cache(self, incidence: Tensor, features: Tensor) -> None:
        cache_path = self._cache_path(incidence)
        if cache_path:
            np.savez_compressed(cache_path, features=features.cpu().numpy())
