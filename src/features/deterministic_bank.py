"""Deterministic feature engineering for hypergraphs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch import Tensor


def _quantile_clip(features: Tensor, clip: float) -> Tensor:
    """Clip features using per-dimension quantiles."""

    if features.numel() == 0:
        return features
    lower = torch.quantile(features, clip, dim=0, keepdim=True)
    upper = torch.quantile(features, 1.0 - clip, dim=0, keepdim=True)
    return torch.clamp(features, lower, upper)


def robust_standardize(features: Tensor, clip: float) -> Tensor:
    """Apply quantile clipping followed by zero-mean/unit-variance scaling."""

    if features.numel() == 0:
        return features
    clipped = _quantile_clip(features, clip)
    mean = clipped.mean(dim=0, keepdim=True)
    std = clipped.std(dim=0, keepdim=True).clamp(min=1e-6)
    standardized = (clipped - mean) / std
    return torch.nan_to_num(standardized)


@dataclass
class DeterministicFeatureConfig:
    spectral_topk: int = 32
    use_spectral: bool = True
    use_hodge: bool = False
    use_temporal: bool = True
    quantile_clip: float = 0.01
    cache_dir: Optional[str] = None
    device: str = "auto"
    precision: Literal["float32", "float64"] = "float32"
    expansion_chunk_size: Optional[int] = None


class DeterministicFeatureBank:
    """Collection of reproducible hypergraph descriptors."""

    def __init__(self, config: DeterministicFeatureConfig) -> None:
        self.config = config
        self._compute_dtype = self._resolve_dtype()

    def __call__(
        self,
        incidence: Tensor,
        edge_weights: Tensor,
        timestamps: Optional[Tensor] = None,
    ) -> Tensor:
        cache = self._load_cache(incidence, timestamps)
        if cache is not None:
            return cache

        compute_device = self._resolve_device(incidence)
        with torch.inference_mode():
            incidence = incidence.to(
                compute_device, dtype=self._compute_dtype, non_blocking=True
            )
            edge_weights = edge_weights.to(
                compute_device, dtype=self._compute_dtype, non_blocking=True
            )
            timestamps_tensor = (
                timestamps.to(
                    compute_device, dtype=self._compute_dtype, non_blocking=True
                )
                if timestamps is not None
                else None
            )

            features = [
                self._hyperdegree(incidence, edge_weights),
                self._incident_cardinality_stats(incidence),
                self._expansion_graph_descriptors(incidence, edge_weights),
            ]

            if self.config.use_spectral:
                features.append(
                    self._spectral_embeddings(
                        incidence, edge_weights, self.config.spectral_topk
                    )
                )
            if self.config.use_hodge:
                features.append(self._hodge_proxies(incidence, edge_weights))
            if self.config.use_temporal and timestamps_tensor is not None:
                features.append(
                    self._temporal_statistics(incidence, timestamps_tensor)
                )

            combined = torch.cat(features, dim=1)
            combined = robust_standardize(combined, self.config.quantile_clip)

        self._save_cache(incidence, timestamps, combined)
        return combined

    def _hyperdegree(self, incidence: Tensor, edge_weights: Tensor) -> Tensor:
        weighted_degree = incidence @ edge_weights.unsqueeze(1)
        unweighted_degree = incidence.gt(0).sum(dim=1, keepdim=True).to(weighted_degree.dtype)
        return torch.cat([weighted_degree, unweighted_degree], dim=1)

    def _incident_cardinality_stats(self, incidence: Tensor) -> Tensor:
        edge_sizes = incidence.sum(dim=0, keepdim=True)
        mask = incidence.gt(0)
        node_edge_counts = (
            mask.sum(dim=1, keepdim=True).to(dtype=incidence.dtype).clamp(min=1.0)
        )
        edge_sizes_expanded = edge_sizes.expand_as(incidence)
        masked_sizes = torch.where(mask, edge_sizes_expanded, torch.zeros_like(edge_sizes_expanded))
        mean = masked_sizes.sum(dim=1, keepdim=True) / node_edge_counts
        centered = torch.where(mask, masked_sizes - mean, torch.zeros_like(masked_sizes))
        var = (centered**2).sum(dim=1, keepdim=True) / node_edge_counts
        return torch.cat([mean, var], dim=1)

    def _expansion_graph_descriptors(self, incidence: Tensor, edge_weights: Tensor) -> Tensor:
        edge_sizes = incidence.sum(dim=0).clamp(min=1.0)
        pair_scaling = edge_weights / (edge_sizes - 1.0).clamp(min=1.0)
        weighted_incidence = incidence * pair_scaling.unsqueeze(0)

        num_nodes = incidence.shape[0]
        chunk_size = self.config.expansion_chunk_size
        if chunk_size is None or chunk_size >= num_nodes:
            adjacency = weighted_incidence @ incidence.t()
            adjacency = (adjacency + adjacency.t()) * 0.5
        else:
            adjacency = incidence.new_zeros((num_nodes, num_nodes))
            for start in range(0, num_nodes, chunk_size):
                end = min(start + chunk_size, num_nodes)
                block_left = weighted_incidence[start:end] @ incidence.t()
                block_right = incidence[start:end] @ weighted_incidence.t()
                adjacency[start:end] = 0.5 * (block_left + block_right)

        adjacency.fill_diagonal_(0.0)
        clique_degree = adjacency.sum(dim=1, keepdim=True)
        star_degree = incidence.sum(dim=1, keepdim=True)
        triangles = (adjacency @ adjacency * adjacency).sum(dim=1, keepdim=True) / 2.0
        clustering = triangles / (
            clique_degree.clamp(min=1.0)
            * (clique_degree.clamp(min=1.0) - 1.0)
            + 1e-6
        )
        return torch.cat([star_degree, clique_degree, clustering], dim=1)

    def _spectral_embeddings(
        self, incidence: Tensor, edge_weights: Tensor, topk: int
    ) -> Tensor:
        theta = self._normalized_operator(incidence, edge_weights)
        identity = torch.eye(theta.shape[0], device=theta.device, dtype=theta.dtype)
        _, evecs = torch.linalg.eigh(theta + 1e-6 * identity)
        k = min(topk, evecs.shape[1])
        return evecs[:, -k:]

    def _hodge_proxies(self, incidence: Tensor, edge_weights: Tensor) -> Tensor:
        weighted_incidence = incidence * edge_weights.unsqueeze(0)
        gram = weighted_incidence @ weighted_incidence.t()
        cycle_strength = torch.diag(gram).unsqueeze(1)
        flux_strength = weighted_incidence.std(dim=1, keepdim=True)
        return torch.cat([cycle_strength, flux_strength], dim=1)

    def _temporal_statistics(self, incidence: Tensor, timestamps: Tensor) -> Tensor:
        timestamps = timestamps.to(dtype=incidence.dtype)
        mask = incidence.gt(0)
        node_counts = mask.sum(dim=1, keepdim=True).to(dtype=incidence.dtype)
        node_counts_clamped = node_counts.clamp(min=1.0)
        ts_matrix = timestamps.unsqueeze(0).expand_as(incidence)
        masked_ts = torch.where(mask, ts_matrix, torch.zeros_like(ts_matrix))
        mean_time = masked_ts.sum(dim=1, keepdim=True) / node_counts_clamped
        centered = torch.where(mask, ts_matrix - mean_time, torch.zeros_like(ts_matrix))
        var_time = (centered**2).sum(dim=1, keepdim=True) / node_counts_clamped
        duration = timestamps.max() - timestamps.min() + 1e-6
        interaction_rate = node_counts / duration
        stability = 1.0 / (1.0 + var_time)
        return torch.cat([interaction_rate, mean_time / duration, stability], dim=1)

    def _normalized_operator(self, incidence: Tensor, edge_weights: Tensor) -> Tensor:
        node_degree = (incidence @ edge_weights.unsqueeze(1)).flatten()
        edge_degree = incidence.sum(dim=0)

        # Avoid forming large diagonal matrices explicitly to keep memory usage
        # manageable for hypergraphs with many nodes or edges.
        d_v_inv_sqrt = torch.pow(node_degree.clamp(min=1e-6), -0.5)
        d_e_inv = torch.pow(edge_degree.clamp(min=1.0), -1.0)

        weighted_incidence = incidence * (edge_weights * d_e_inv).unsqueeze(0)
        normalized = weighted_incidence @ incidence.t()
        normalized = d_v_inv_sqrt.unsqueeze(1) * normalized * d_v_inv_sqrt.unsqueeze(0)
        return normalized

    def _cache_path(
        self, incidence: Tensor, timestamps: Optional[Tensor]
    ) -> Optional[Path]:
        if not self.config.cache_dir or (timestamps is not None and self.config.use_temporal):
            return None
        num_nodes, num_edges = incidence.shape
        identifier = f"det_{num_nodes}_{num_edges}.npz"
        path = Path(self.config.cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path / identifier

    def _load_cache(
        self, incidence: Tensor, timestamps: Optional[Tensor]
    ) -> Optional[Tensor]:
        cache_path = self._cache_path(incidence, timestamps)
        if cache_path and cache_path.exists():
            data = np.load(cache_path)
            return torch.tensor(data["features"], dtype=torch.float32)
        return None

    def _save_cache(
        self, incidence: Tensor, timestamps: Optional[Tensor], features: Tensor
    ) -> None:
        cache_path = self._cache_path(incidence, timestamps)
        if cache_path:
            np.savez_compressed(
                cache_path, features=features.to(dtype=torch.float32).cpu().numpy()
            )

    def _resolve_device(self, tensor: Tensor) -> torch.device:
        if self.config.device == "auto":
            if tensor.is_cuda:
                return tensor.device
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(self.config.device)

    def _resolve_dtype(self) -> torch.dtype:
        if self.config.precision == "float64":
            return torch.float64
        return torch.float32
