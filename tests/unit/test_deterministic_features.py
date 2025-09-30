from __future__ import annotations

import torch

from src.features.deterministic_bank import DeterministicFeatureBank, DeterministicFeatureConfig


def test_deterministic_feature_dimensions() -> None:
    incidence = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    weights = torch.ones(2)
    bank = DeterministicFeatureBank(DeterministicFeatureConfig(spectral_topk=2, use_spectral=True))
    features = bank(incidence, weights)
    assert features.shape[0] == incidence.shape[0]
    assert features.shape[1] >= 2
