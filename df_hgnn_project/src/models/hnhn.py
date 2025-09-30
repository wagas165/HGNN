
import torch, torch.nn as nn, torch.nn.functional as F
from .layers import build_incidence, BiConvHNHN

class HNHN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.2):
        super().__init__()
        self.conv1 = BiConvHNHN(in_dim, in_dim, hidden)
        self.conv2 = BiConvHNHN(hidden, hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, H):
        # initial hyperedge features: mean of incident node features
        He = H / torch.clamp(H.sum(0, keepdim=True), min=1.0)
        e0 = He.t() @ x
        h = self.conv1(x, e0, H)
        h = self.dropout(h)
        h = self.conv2(h, e0, H)
        return self.head(h)
