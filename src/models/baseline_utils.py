"""Utility helpers for baseline hypergraph models."""
from __future__ import annotations

import torch
from torch import Tensor


def fuse_features(x: Tensor, z: Tensor) -> Tensor:
    """Concatenate raw and deterministic features when available."""
    tensors = []
    if x is not None and x.numel() > 0:
        tensors.append(x)
    if z is not None and z.numel() > 0:
        tensors.append(z)
    if not tensors:
        raise ValueError("Baseline models require at least one input feature source")
    return torch.cat(tensors, dim=1)


def _prepare_edge_weights(
    incidence: Tensor, edge_weights: Tensor | None, dtype: torch.dtype, device: torch.device
) -> Tensor:
    num_edges = incidence.shape[1]
    if edge_weights is None or edge_weights.numel() == 0:
        return torch.ones(num_edges, device=device, dtype=dtype)
    weights = edge_weights.view(-1).to(device=device, dtype=dtype)
    if weights.shape[0] != num_edges:
        raise ValueError(
            "Edge-weight vector must match the number of hyperedges: "
            f"expected {num_edges}, got {weights.shape[0]}"
        )
    return weights


def aggregate_edges(node_features: Tensor, incidence: Tensor, edge_weights: Tensor | None) -> Tensor:
    """Project node representations to the hyperedge domain."""
    dtype = node_features.dtype
    device = node_features.device
    weights = _prepare_edge_weights(incidence, edge_weights, dtype, device)
    incidence = incidence.to(device=device, dtype=dtype)
    weighted_incidence = incidence * weights.unsqueeze(0)
    edge_deg = weighted_incidence.sum(dim=0, keepdim=True).clamp_min(1.0)
    aggregated = (weighted_incidence.t() @ node_features) / edge_deg.t()
    return aggregated


def aggregate_nodes(edge_features: Tensor, incidence: Tensor, edge_weights: Tensor | None) -> Tensor:
    """Diffuse hyperedge representations back to the node domain."""
    dtype = edge_features.dtype
    device = edge_features.device
    weights = _prepare_edge_weights(incidence, edge_weights, dtype, device)
    incidence = incidence.to(device=device, dtype=dtype)
    weighted_incidence = incidence * weights.unsqueeze(0)
    node_deg = weighted_incidence.sum(dim=1, keepdim=True).clamp_min(1.0)
    aggregated = (weighted_incidence @ edge_features) / node_deg
    return aggregated


def hypergraph_normalised_adjacency(
    incidence: Tensor, edge_weights: Tensor | None, dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Compute the symmetric normalised adjacency used by HyperGCN."""
    num_nodes, num_edges = incidence.shape
    if num_edges == 0:
        return torch.eye(num_nodes, device=device, dtype=dtype)

    incidence = incidence.to(device=device, dtype=dtype)
    weights = _prepare_edge_weights(incidence, edge_weights, dtype, device)

    edge_degree = incidence.sum(dim=0).clamp_min(1.0)
    node_degree = (incidence * weights.unsqueeze(0)).sum(dim=1).clamp_min(1.0)

    sqrt_weights = torch.sqrt(weights)
    inv_sqrt_edge_degree = torch.rsqrt(edge_degree)

    scaled_incidence = incidence * (sqrt_weights * inv_sqrt_edge_degree).unsqueeze(0)
    adjacency = scaled_incidence @ scaled_incidence.t()
    adjacency.fill_diagonal_(0.0)

    inv_sqrt_node_degree = torch.rsqrt(node_degree)
    adjacency = (
        inv_sqrt_node_degree.unsqueeze(1) * adjacency * inv_sqrt_node_degree.unsqueeze(0)
    )
    adjacency = adjacency + torch.eye(num_nodes, device=device, dtype=dtype)
    return adjacency


def zero_gate(reference: Tensor) -> Tensor:
    """Return a gate tensor that keeps the trainer regularisers well-defined."""
    return reference.new_zeros(reference.shape)


__all__ = [
    "aggregate_edges",
    "aggregate_nodes",
    "fuse_features",
    "hypergraph_normalised_adjacency",
    "zero_gate",
]
