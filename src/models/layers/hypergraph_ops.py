"""Hypergraph operators."""
from __future__ import annotations

import torch
from torch import Tensor


def normalized_hypergraph_operator(incidence: Tensor, edge_weights: Tensor) -> Tensor:
    """Compute D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}."""
    node_degree = torch.matmul(incidence, edge_weights.unsqueeze(1)).flatten()
    edge_degree = incidence.sum(dim=0)
    d_v_inv_sqrt = torch.diag(torch.pow(node_degree.clamp(min=1e-6), -0.5))
    d_e_inv = torch.diag(torch.pow(edge_degree.clamp(min=1.0), -1.0))
    w_diag = torch.diag(edge_weights)
    theta = d_v_inv_sqrt @ incidence @ w_diag @ d_e_inv @ incidence.t() @ d_v_inv_sqrt
    return theta


def hypergraph_message_passing(x: Tensor, incidence: Tensor, edge_weights: Tensor) -> Tensor:
    theta = normalized_hypergraph_operator(incidence, edge_weights)
    return theta @ x
