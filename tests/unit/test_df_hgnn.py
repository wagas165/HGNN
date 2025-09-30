from __future__ import annotations

import torch

from src.models.df_hgnn import DFHGNN, DFHGNNConfig


def test_df_hgnn_forward() -> None:
    config = DFHGNNConfig(in_dim=4, det_dim=3, hidden_dim=16, out_dim=2)
    model = DFHGNN(config)
    x = torch.randn(10, 4)
    z = torch.randn(10, 3)
    incidence = torch.randint(0, 2, (10, 5)).float()
    weights = torch.ones(5)
    logits, gate = model(x, z, incidence, weights)
    assert logits.shape == (10, 2)
    assert gate.shape == (10, config.hidden_dim // 2)
