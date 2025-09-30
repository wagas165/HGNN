"""Hypergraph operators."""
from __future__ import annotations

import torch
from torch import Tensor


def normalized_hypergraph_operator(incidence: Tensor, edge_weights: Tensor) -> Tensor:
    """Compute D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}."""

    node_degree = torch.matmul(incidence, edge_weights.unsqueeze(-1)).squeeze(-1)
    edge_degree = incidence.sum(dim=0)

    d_v_inv_sqrt = torch.pow(node_degree.clamp(min=1e-6), -0.5)
    edge_scale = edge_weights / edge_degree.clamp(min=1.0)

    weighted_incidence = incidence * edge_scale
    theta_inner = torch.matmul(weighted_incidence, incidence.t())

    theta = d_v_inv_sqrt.unsqueeze(1) * theta_inner * d_v_inv_sqrt.unsqueeze(0)
    return theta


def hypergraph_message_passing(x: Tensor, incidence: Tensor, edge_weights: Tensor) -> Tensor:
    theta = normalized_hypergraph_operator(incidence, edge_weights)
    return theta @ x
