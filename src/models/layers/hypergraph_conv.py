"""Hypergraph convolutional layers."""
from __future__ import annotations

import torch
from torch import nn
from torch import Tensor

from .hypergraph_ops import hypergraph_message_passing, normalized_hypergraph_operator


class HypergraphMPConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor, incidence: Tensor, edge_weights: Tensor) -> Tensor:
        support = self.linear(x)
        return hypergraph_message_passing(support, incidence, edge_weights)


class HypergraphChebConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, K: int = 2) -> None:
        super().__init__()
        self.K = K
        self.linear = nn.Linear(in_channels * (K + 1), out_channels)
        self._cached_key: int | None = None
        self._cached_theta: Tensor | None = None

    def forward(self, x: Tensor, incidence: Tensor, edge_weights: Tensor) -> Tensor:
        theta = self._get_cached_theta(incidence, edge_weights)
        l_tilde = 2 * theta - torch.eye(theta.shape[0], device=theta.device)

        outputs = [x]
        if self.K >= 1:
            outputs.append(l_tilde @ x)
        for _ in range(2, self.K + 1):
            outputs.append(2 * l_tilde @ outputs[-1] - outputs[-2])
        combined = torch.cat(outputs, dim=1)
        return self.linear(combined)

    def _get_cached_theta(self, incidence: Tensor, edge_weights: Tensor) -> Tensor:
        key = hash(
            (
                incidence.data_ptr(),
                edge_weights.data_ptr(),
                incidence.shape,
                edge_weights.shape,
                incidence.device,
            )
        )
        if self._cached_key != key or self._cached_theta is None:
            self._cached_key = key
            self._cached_theta = normalized_hypergraph_operator(incidence, edge_weights)
        return self._cached_theta
