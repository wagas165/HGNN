
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_incidence(n_nodes: int, hyperedges: List[List[int]], device=None):
    M = len(hyperedges)
    H = torch.zeros((n_nodes, M), dtype=torch.float32, device=device)
    for e, nodes in enumerate(hyperedges):
        for v in nodes:
            H[v, e] = 1.0
    w = torch.ones(M, dtype=torch.float32, device=device)
    return H, w

def normalized_hypergraph_operator(H: torch.Tensor, w: torch.Tensor):
    De = H.sum(dim=0)
    Dv = torch.matmul(H, w.unsqueeze(-1)).squeeze(-1)

    Dv_inv_sqrt = torch.pow(Dv.clamp(min=1e-6), -0.5)
    edge_scale = w / De.clamp(min=1.0)

    weighted_H = H * edge_scale
    Theta_inner = weighted_H @ H.t()

    Theta = Dv_inv_sqrt.unsqueeze(1) * Theta_inner * Dv_inv_sqrt.unsqueeze(0)
    return Theta

class HypergraphMPConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x, H, w):
        Theta = normalized_hypergraph_operator(H, w)
        return Theta @ self.lin(x)

class BiConvHNHN(nn.Module):
    """
    Node-Hyperedge bipartite message passing used by HNHN-style layers.
    """
    def __init__(self, in_dim_node, in_dim_edge, out_dim_node):
        super().__init__()
        self.lin_e = nn.Linear(in_dim_edge, out_dim_node)
        self.lin_n = nn.Linear(in_dim_node, out_dim_node)

    def forward(self, x_node, x_edge, H):
        # node -> edge
        He = H / torch.clamp(H.sum(0, keepdim=True), min=1.0)
        x_edge_upd = He.t() @ self.lin_n(x_node)
        # edge -> node
        Hv = H / torch.clamp(H.sum(1, keepdim=True), min=1.0)
        x_node_upd = Hv @ self.lin_e(x_edge_upd)
        return F.relu(x_node_upd)
