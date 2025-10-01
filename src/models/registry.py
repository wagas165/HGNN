"""Model registry for DF-HGNN project."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .allset_transformer import AllSetTransformer, AllSetTransformerConfig
from .df_hgnn import DFHGNN, DFHGNNConfig
from .hypergcn import HyperGCN, HyperGCNConfig
from .unignn import UniGNN, UniGNNConfig


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
    fusion_dim: int | None = None
    extras: Dict[str, object] | None = None


def create_model(name: str, config: ModelFactoryInput):
    extras = config.extras or {}
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
            fusion_dim=config.fusion_dim,
        )
        return DFHGNN(df_config)
    if name.lower() == "allset_transformer":
        model_config = AllSetTransformerConfig(
            in_dim=config.in_dim,
            det_dim=config.det_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            num_layers=int(extras.get("num_layers", 2)),
            num_heads=int(extras.get("num_heads", 4)),
            mlp_ratio=float(extras.get("mlp_ratio", 2.0)),
            dropout=float(extras.get("dropout", config.dropout)),
        )
        return AllSetTransformer(model_config)
    if name.lower() == "unignn":
        model_config = UniGNNConfig(
            in_dim=config.in_dim,
            det_dim=config.det_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            num_layers=int(extras.get("num_layers", 2)),
            dropout=float(extras.get("dropout", config.dropout)),
        )
        return UniGNN(model_config)
    if name.lower() == "hypergcn":
        model_config = HyperGCNConfig(
            in_dim=config.in_dim,
            det_dim=config.det_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.out_dim,
            num_layers=int(extras.get("num_layers", 2)),
            dropout=float(extras.get("dropout", config.dropout)),
        )
        return HyperGCN(model_config)
    raise ValueError(f"Unknown model name: {name}")
