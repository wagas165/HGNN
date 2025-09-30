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
    fusion_dim: int | None = None


class DFHGNN(nn.Module):
    def __init__(self, config: DFHGNNConfig) -> None:
        super().__init__()
        self.config = config
        fusion_dim = config.fusion_dim or max(1, config.hidden_dim // 2)
        self.fusion_dim = fusion_dim

        self.has_raw = config.in_dim > 0
        self.has_det = config.det_dim > 0
        if not self.has_raw and not self.has_det:
            raise ValueError("DF-HGNN requires at least one feature source")

        if self.has_det:
            self.phi = nn.Sequential(
                nn.Linear(config.det_dim, fusion_dim),
                nn.ReLU(),
            )
        else:
            self.phi = None

        if self.has_raw:
            self.psi = nn.Sequential(
                nn.Linear(config.in_dim, fusion_dim),
                nn.ReLU(),
            )
        else:
            self.psi = None

        gate_input = 0
        if self.has_raw:
            gate_input += fusion_dim
        if self.has_det:
            gate_input += fusion_dim

        if self.has_raw and self.has_det:
            self.gate = nn.Sequential(
                nn.Linear(gate_input, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, fusion_dim),
            )
        else:
            self.gate = None

        if config.conv_type == "cheb":
            self.conv1 = HypergraphChebConv(fusion_dim, config.hidden_dim, K=config.chebyshev_order)
            self.conv2 = HypergraphChebConv(config.hidden_dim, config.hidden_dim, K=config.chebyshev_order)
        else:
            self.conv1 = HypergraphMPConv(fusion_dim, config.hidden_dim)
            self.conv2 = HypergraphMPConv(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_dim, config.out_dim)

    def forward(
        self, x: Tensor, z: Tensor, incidence: Tensor, edge_weights: Tensor
    ) -> Tuple[Tensor, Tensor]:
        fused, gate = self._fuse_features(x, z)
        h = self.conv1(fused, incidence, edge_weights)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, incidence, edge_weights)
        h = F.relu(h)
        h = self.dropout(h)
        logits = self.output(h)
        return logits, gate

    def _project_raw(self, x: Tensor) -> Tensor | None:
        if not self.has_raw:
            return None
        if x.numel() == 0:
            return None
        return self.psi(x)

    def _project_det(self, z: Tensor) -> Tensor | None:
        if not self.has_det:
            return None
        return self.phi(z)

    def _fuse_features(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        proj_x = self._project_raw(x)
        proj_z = self._project_det(z)

        if proj_x is None and proj_z is None:
            raise ValueError("At least one feature projection must be available")

        if proj_x is None:
            fused = proj_z
            gate = torch.ones_like(fused)
        elif proj_z is None:
            fused = proj_x
            gate = torch.zeros_like(fused)
        else:
            gate_logits = self.gate(torch.cat([proj_z, proj_x], dim=1))
            gate = torch.sigmoid(gate_logits)
            fused = gate * proj_z + (1 - gate) * proj_x
        return fused, gate

    def alignment_loss(self, x: Tensor, z: Tensor) -> Tensor:
        if not (self.has_raw and self.has_det):
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)
        proj_x = F.normalize(self._project_raw(x), dim=1)
        proj_z = F.normalize(self._project_det(z), dim=1)
        return F.mse_loss(proj_x, proj_z)

    def gate_regularization(self, gate: Tensor) -> Tensor:
        if not (self.has_raw and self.has_det):
            return gate.new_tensor(0.0)
        return gate.abs().mean()
