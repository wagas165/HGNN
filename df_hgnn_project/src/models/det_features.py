
from typing import List
import torch, torch.nn as nn, torch.nn.functional as F
from .layers import normalized_hypergraph_operator

class DeterministicFeatures(nn.Module):
    def __init__(self, topk_spectral: int = 16, use_spectral: bool = True):
        super().__init__()
        self.k = topk_spectral
        self.use_spectral = use_spectral

    @torch.no_grad()
    def forward(self, X: torch.Tensor, H: torch.Tensor, w: torch.Tensor):
        N, M = H.shape
        hyperdeg = (H * w.unsqueeze(0)).sum(dim=1, keepdim=True)
        edge_card = H.sum(dim=0, keepdim=True)
        avg_card = (H @ (1.0 / edge_card.clamp(min=1.0)).t()).sum(dim=1, keepdim=True) / H.gt(0).sum(1, keepdim=True).clamp(min=1.0)
        feats = [hyperdeg, avg_card]
        if self.use_spectral:
            Theta = normalized_hypergraph_operator(H, w)
            k = min(self.k, N)
            evals, evecs = torch.linalg.eigh(Theta + 1e-6*torch.eye(N, device=Theta.device))
            pe = evecs[:, -k:]
            feats.append(pe)
        z = torch.cat(feats, dim=1)
        z = (z - z.mean(0, keepdim=True)) / (z.std(0, keepdim=True) + 1e-6)
        return z
