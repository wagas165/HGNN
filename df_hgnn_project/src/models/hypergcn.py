
import torch, torch.nn as nn, torch.nn.functional as F

def clique_adjacency(n_nodes, hyperedges):
    A = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    for e in hyperedges:
        for i in range(len(e)):
            for j in range(i+1, len(e)):
                u, v = e[i], e[j]
                A[u,v] = 1.0
                A[v,u] = 1.0
    # add self loops
    A += torch.eye(n_nodes)
    # sym norm
    D = torch.diag(torch.pow(A.sum(1).clamp(min=1.0), -0.5))
    return D @ A @ D

class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.2):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden)
        self.W2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, Anorm):
        h = F.relu(Anorm @ self.W1(x))
        h = self.dropout(h)
        h = F.relu(Anorm @ self.W2(h))
        return self.head(h)

class HyperGCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.2):
        super().__init__()
        self.gcn = GCN(in_dim, hidden, out_dim, dropout=dropout)

    def forward(self, x, hyperedges):
        Anorm = clique_adjacency(x.size(0), hyperedges)
        return self.gcn(x, Anorm)
