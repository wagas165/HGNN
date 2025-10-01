"""AllSet Transformer baseline for hypergraph learning."""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from .baseline_utils import aggregate_edges, fuse_features, zero_gate


@dataclass
class AllSetTransformerConfig:
    in_dim: int
    det_dim: int
    hidden_dim: int = 128
    out_dim: int = 2
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.2


class _AllSetTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        mlp_hidden = max(hidden_dim, int(hidden_dim * mlp_ratio))
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, node_embeddings: Tensor, incidence: Tensor, edge_weights: Tensor | None
    ) -> Tensor:
        edge_embeddings = aggregate_edges(node_embeddings, incidence, edge_weights)
        attn_output, _ = self.attention(
            node_embeddings.unsqueeze(0),
            edge_embeddings.unsqueeze(0),
            edge_embeddings.unsqueeze(0),
        )
        attn_output = attn_output.squeeze(0)
        h = self.norm1(node_embeddings + self.dropout(attn_output))
        ff = self.feedforward(h)
        h = self.norm2(h + self.dropout(ff))
        return h


class AllSetTransformer(nn.Module):
    """A minimal AllSet Transformer implementation with a DF-HGNN compatible API."""

    def __init__(self, config: AllSetTransformerConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.in_dim + config.det_dim
        if input_dim <= 0:
            raise ValueError("AllSet Transformer requires at least one feature dimension")

        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                _AllSetTransformerBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_dim, config.out_dim)

    def forward(
        self, x: Tensor, z: Tensor, incidence: Tensor, edge_weights: Tensor
    ) -> tuple[Tensor, Tensor]:
        fused = fuse_features(x, z)
        h = self.input_proj(fused)
        for block in self.blocks:
            h = block(h, incidence, edge_weights)
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


__all__ = ["AllSetTransformer", "AllSetTransformerConfig"]
