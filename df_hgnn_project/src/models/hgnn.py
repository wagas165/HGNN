
import torch, torch.nn as nn, torch.nn.functional as F
from .layers import build_incidence, HypergraphMPConv

class HGNN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.2):
        super().__init__()
        self.conv1 = HypergraphMPConv(in_dim, hidden)
        self.conv2 = HypergraphMPConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, H, w):
        h = F.relu(self.conv1(x, H, w))
        h = self.dropout(h)
        h = F.relu(self.conv2(h, H, w))
        return self.head(h)
