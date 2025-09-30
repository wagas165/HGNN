"""Model registry for DF-HGNN project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from .df_hgnn import DFHGNN, DFHGNNConfig


@dataclass
class ModelFactoryInput:
    in_dim: int
    det_dim: int
    out_dim: int
    hidden_dim: int
    dropout: float
    conv_type: str
    chebyshev_order: int
    lambda_align: float
    lambda_gate: float


def create_model(name: str, config: ModelFactoryInput):
    if name.lower() == "df_hgnn":
        df_config = DFHGNNConfig(
            in_dim=config.in_dim,
            det_dim=config.det_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            dropout=config.dropout,
            conv_type=config.conv_type,
            chebyshev_order=config.chebyshev_order,
            lambda_align=config.lambda_align,
            lambda_gate=config.lambda_gate,
        )
        return DFHGNN(df_config)
    raise ValueError(f"Unknown model name: {name}")
