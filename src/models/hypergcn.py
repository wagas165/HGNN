"""HyperGCN baseline with symmetric normalised diffusion."""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .baseline_utils import fuse_features, hypergraph_normalised_adjacency, zero_gate


@dataclass
class HyperGCNConfig:
    in_dim: int
    det_dim: int
    hidden_dim: int = 128
    out_dim: int = 2
    num_layers: int = 2
    dropout: float = 0.2


class _HyperGCNLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, node_embeddings: Tensor, incidence: Tensor, edge_weights: Tensor | None
    ) -> Tensor:
        adjacency = hypergraph_normalised_adjacency(
            incidence, edge_weights, node_embeddings.dtype, node_embeddings.device
        )
        propagated = adjacency @ node_embeddings
        updated = self.activation(self.linear(self.dropout(propagated)))
        return updated


class HyperGCN(nn.Module):
    """HyperGCN baseline that shares DF-HGNN's training interface."""

    def __init__(self, config: HyperGCNConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.in_dim + config.det_dim
        if input_dim <= 0:
            raise ValueError("HyperGCN requires at least one feature dimension")

        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            [_HyperGCNLayer(config.hidden_dim, config.dropout) for _ in range(config.num_layers)]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_dim, config.out_dim)

    def forward(
        self, x: Tensor, z: Tensor, incidence: Tensor, edge_weights: Tensor
    ) -> tuple[Tensor, Tensor]:
        fused = fuse_features(x, z)
        h = self.input_proj(fused)
        for layer in self.layers:
            h = layer(h, incidence, edge_weights)
        h = self.dropout(h)
        logits = self.output(h)
        gate = zero_gate(h)
        return logits, gate

    def alignment_loss(self, x: Tensor, z: Tensor) -> Tensor:
        reference = x if x is not None and x.numel() > 0 else z
        if reference is None or reference.numel() == 0:
            reference = next(self.parameters()).detach()
        return reference.new_zeros(())

    def gate_regularization(self, gate: Tensor) -> Tensor:
        return gate.new_zeros(())


__all__ = ["HyperGCN", "HyperGCNConfig"]
