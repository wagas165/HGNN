"""Implementation of the DF-HGNN model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from .layers.hypergraph_conv import HypergraphChebConv, HypergraphMPConv


@dataclass
class DFHGNNConfig:
    in_dim: int
    det_dim: int
    hidden_dim: int = 128
    out_dim: int = 2
    dropout: float = 0.3
    conv_type: Literal["mp", "cheb"] = "mp"
    chebyshev_order: int = 2
    lambda_align: float = 0.1
    lambda_gate: float = 0.001


class DFHGNN(nn.Module):
    def __init__(self, config: DFHGNNConfig) -> None:
        super().__init__()
        half_hidden = config.hidden_dim // 2
        self.config = config
        self.phi = nn.Linear(config.det_dim, half_hidden)
        self.psi = nn.Linear(config.in_dim, half_hidden)
        self.gate = nn.Sequential(
            nn.Linear(half_hidden * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, half_hidden),
        )
        if config.conv_type == "cheb":
            self.conv1 = HypergraphChebConv(half_hidden, config.hidden_dim, K=config.chebyshev_order)
            self.conv2 = HypergraphChebConv(config.hidden_dim, config.hidden_dim, K=config.chebyshev_order)
        else:
            self.conv1 = HypergraphMPConv(half_hidden, config.hidden_dim)
            self.conv2 = HypergraphMPConv(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_dim, config.out_dim)

    def forward(self, x: Tensor, z: Tensor, incidence: Tensor, edge_weights: Tensor) -> Tuple[Tensor, Tensor]:
        fused, gate = self._fuse_features(x, z)
        h = self.conv1(fused, incidence, edge_weights)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, incidence, edge_weights)
        h = F.relu(h)
        h = self.dropout(h)
        logits = self.output(h)
        return logits, gate

    def _fuse_features(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        proj_x = self.psi(x)
        proj_z = self.phi(z)
        gate_logits = self.gate(torch.cat([proj_x, proj_z], dim=1))
        gate = torch.sigmoid(gate_logits)
        fused = gate * proj_z + (1 - gate) * proj_x
        return fused, gate

    def alignment_loss(self, x: Tensor, z: Tensor) -> Tensor:
        proj_x = F.normalize(self.psi(x), dim=1)
        proj_z = F.normalize(self.phi(z), dim=1)
        return F.mse_loss(proj_x, proj_z)

    def gate_regularization(self, gate: Tensor) -> Tensor:
        return gate.mean()
