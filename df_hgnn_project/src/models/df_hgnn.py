
import torch, torch.nn as nn, torch.nn.functional as F
from .layers import HypergraphMPConv
from .det_features import DeterministicFeatures

class DFHGNN(nn.Module):
    def __init__(self, in_dim, det_dim, hidden=64, out_dim=2, dropout=0.2):
        super().__init__()
        self.phi = nn.Linear(det_dim, hidden//2) if det_dim>0 else None
        self.psi = nn.Linear(in_dim, hidden//2) if in_dim>0 else None
        self.gate = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2)
        )
        self.conv1 = HypergraphMPConv(hidden//2, hidden)
        self.conv2 = HypergraphMPConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, out_dim)

    def fuse(self, x, z):
        parts = []
        if self.psi is not None and x is not None:
            x1 = self.psi(x)
            parts.append(x1)
        else:
            x1 = 0.0
        if self.phi is not None and z is not None:
            z1 = self.phi(z)
            parts.append(z1)
        else:
            z1 = 0.0

        if len(parts)==2:
            g = torch.sigmoid(self.gate(torch.cat(parts, dim=1)))
            fused = g * z1 + (1 - g) * x1
        elif len(parts)==1:
            fused = parts[0]
            g = torch.zeros_like(fused)
        else:
            raise ValueError("Both raw and deterministic features are missing.")
        return self.dropout(fused), g

    def forward(self, x, z, H, w):
        h, g = self.fuse(x, z)
        h = F.relu(self.conv1(h, H, w))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, H, w))
        logits = self.head(h)
        return logits, g
