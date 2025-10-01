"""UniGNN baseline with DF-HGNN compatible interface."""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .baseline_utils import aggregate_edges, aggregate_nodes, fuse_features, zero_gate


@dataclass
class UniGNNConfig:
    in_dim: int
    det_dim: int
    hidden_dim: int = 128
    out_dim: int = 2
    num_layers: int = 2
    dropout: float = 0.3


class _UniGNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.edge_linear = nn.Linear(hidden_dim, hidden_dim)
        self.node_linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, node_embeddings: Tensor, incidence: Tensor, edge_weights: Tensor | None
    ) -> Tensor:
        edge_messages = aggregate_edges(node_embeddings, incidence, edge_weights)
        edge_messages = self.activation(self.edge_linear(edge_messages))
        node_messages = aggregate_nodes(edge_messages, incidence, edge_weights)
        node_messages = self.activation(self.node_linear(node_messages))
        updated = node_embeddings + self.dropout(node_messages)
        return updated


class UniGNN(nn.Module):
    """Simplified UniGNN encoder with residual diffusion blocks."""

    def __init__(self, config: UniGNNConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.in_dim + config.det_dim
        if input_dim <= 0:
            raise ValueError("UniGNN requires at least one feature dimension")

        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            [_UniGNNLayer(config.hidden_dim, config.dropout) for _ in range(config.num_layers)]
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


__all__ = ["UniGNN", "UniGNNConfig"]
