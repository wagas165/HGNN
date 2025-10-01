"""AllSet Transformer baseline for hypergraph learning."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .baseline_utils import aggregate_edges, fuse_features, zero_gate


try:  # pragma: no-cover - older PyTorch releases may lack SDPA
    from torch.nn.functional import scaled_dot_product_attention as _sdpa

    _SDPA_AVAILABLE = True
except ImportError:  # pragma: no cover - fall back to manual attention
    _sdpa = None
    _SDPA_AVAILABLE = False


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


class _CrossAttention(nn.Module):
    """Memory-efficient multi-head attention between nodes and hyperedges."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "Hidden dimension must be divisible by the number of heads: "
                f"got hidden_dim={hidden_dim}, num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor | None = None) -> Tensor:
        if value is None:
            value = key

        squeeze_batch = query.dim() == 2
        if squeeze_batch:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        elif query.dim() != 3:
            raise ValueError("Attention inputs must be 2D or 3D tensors")

        batch_size, target_len, hidden_dim = query.shape
        source_len = key.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)

        if _SDPA_AVAILABLE:
            attn = _sdpa(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training and self.dropout > 0 else 0.0,
            )
        else:  # pragma: no cover - compatibility path for older PyTorch
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            weights = torch.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                weights = F.dropout(weights, p=self.dropout)
            attn = torch.matmul(weights, v)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, target_len, hidden_dim)
        attn = self.out_proj(attn)
        if squeeze_batch:
            attn = attn.squeeze(0)
        return attn


class _AllSetTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.attention = _CrossAttention(hidden_dim, num_heads, dropout)
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
        attn_output = self.attention(node_embeddings, edge_embeddings)
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
