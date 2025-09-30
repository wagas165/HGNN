
import os, argparse, math
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from .data.loaders import load_processed
from .models.layers import build_incidence
from .models.hgnn import HGNN
from .models.hnhn import HNHN
from .models.hypergcn import HyperGCN
from .models.df_hgnn import DFHGNN
from .models.det_features import DeterministicFeatures
from .metrics import classification_metrics

def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)

def build_features(X_np, H, w, topk_spectral=16, use_spectral=True, use_raw_features=True):
    device = H.device
    det = DeterministicFeatures(topk_spectral=topk_spectral, use_spectral=use_spectral)
    with torch.no_grad():
        Z = det(None, H, w)
    X = None
    if use_raw_features and X_np is not None:
        X = torch.tensor(X_np, dtype=torch.float32, device=device)
        # standardize raw features
        X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
    return X, Z

def run_node_classification(data, args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    N = data['N']
    hyperedges = data['hyperedges']
    X_np = data['X']
    y = torch.tensor(data['y_node'], dtype=torch.long, device=device)
    tr, va, te = [torch.tensor(idx, dtype=torch.long, device=device) for idx in data['splits'].values()]

    H, w = build_incidence(N, hyperedges, device=device)
    X, Z = build_features(X_np, H, w, args.topk_spectral, args.use_spectral, args.use_raw_features)

    in_dim = 0 if X is None else X.size(1)
    det_dim = Z.size(1)

    if args.model == 'hgnn':
        model = HGNN(in_dim if in_dim>0 else det_dim, args.hidden, y.max().item()+1, dropout=args.dropout).to(device)
    elif args.model == 'hnhn':
        model = HNHN(in_dim if in_dim>0 else det_dim, args.hidden, y.max().item()+1, dropout=args.dropout).to(device)
    elif args.model == 'hypergcn':
        model = HyperGCN(in_dim if in_dim>0 else det_dim, args.hidden, y.max().item()+1, dropout=args.dropout).to(device)
    elif args.model == 'df_hgnn':
        model = DFHGNN(in_dim, det_dim, hidden=args.hidden, out_dim=y.max().item()+1, dropout=args.dropout).to(device)
    else:
        raise ValueError(f"Unknown model {args.model}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()

    best_state = None; best_val = -1
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        if args.model == 'df_hgnn':
            logits, g = model(X if X is not None else torch.zeros(N,0,device=device), Z, H, w)
            loss = ce(logits[tr], y[tr])
            # alignment reg
            if X is not None:
                # use internal layers
                proj_x = F.normalize(model.psi(X), dim=1)
                proj_z = F.normalize(model.phi(Z), dim=1)
                loss = loss + args.align_lambda * F.mse_loss(proj_x, proj_z)
        elif args.model == 'hnhn':
            logits = model(X if X is not None else Z, H)
            loss = ce(logits[tr], y[tr])
        elif args.model in ('hgnn','hypergcn'):
            inp = X if X is not None else Z
            logits = model(inp, H if args.model=='hgnn' else hyperedges)
            loss = ce(logits[tr], y[tr])
        else:
            raise RuntimeError

        loss.backward(); opt.step()

        if (ep+1)%10==0:
            model.eval()
            with torch.no_grad():
                if args.model == 'df_hgnn':
                    logits,_ = model(X if X is not None else torch.zeros(N,0,device=device), Z, H, w)
                elif args.model == 'hnhn':
                    logits = model(X if X is not None else Z, H)
                else:
                    inp = X if X is not None else Z
                    logits = model(inp, H if args.model=='hgnn' else hyperedges)
                pred = logits[va].argmax(1).cpu().numpy()
                metrics = classification_metrics(y[va].cpu().numpy(), pred)
                if metrics['acc'] > best_val:
                    best_val = metrics['acc']
                    best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # L-BFGS fine-tune (optional)
    if args.lbfgs_epochs>0 and args.model == 'df_hgnn':
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=args.lbfgs_epochs, history_size=10, line_search_fn='strong_wolfe')
        def closure():
            lbfgs.zero_grad()
            logits,_ = model(X if X is not None else torch.zeros(N,0,device=device), Z, H, w)
            loss = ce(logits[tr], y[tr])
            if X is not None:
                proj_x = F.normalize(model.psi(X), dim=1)
                proj_z = F.normalize(model.phi(Z), dim=1)
                loss = loss + args.align_lambda * F.mse_loss(proj_x, proj_z)
            loss.backward()
            return loss
        lbfgs.step(closure)

    # Evaluate
    model.eval()
    with torch.no_grad():
        if args.model == 'df_hgnn':
            logits,_ = model(X if X is not None else torch.zeros(N,0,device=device), Z, H, w)
        elif args.model == 'hnhn':
            logits = model(X if X is not None else Z, H)
        else:
            inp = X if X is not None else Z
            logits = model(inp, H if args.model=='hgnn' else hyperedges)
        y_pred = logits.argmax(1).cpu().numpy()
        res_tr = classification_metrics(y[tr].cpu().numpy(), y_pred[tr.cpu().numpy()])
        res_va = classification_metrics(y[va].cpu().numpy(), y_pred[va.cpu().numpy()])
        res_te = classification_metrics(y[te].cpu().numpy(), y_pred[te.cpu().numpy()])
    return {'train':res_tr, 'val':res_va, 'test':res_te}

def run_edge_classification(data, args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    N = data['N']
    hyperedges = data['hyperedges']
    y = torch.tensor(data['y_edge'], dtype=torch.long, device=device)
    tr, va, te = [torch.tensor(idx, dtype=torch.long, device=device) for idx in data['splits'].values()]

    H, w = build_incidence(N, hyperedges, device=device)
    # Edge features: mean of incident node features (raw or deterministic node features)
    # Build node features same as node classification; then aggregate to edges
    X_np = data['X']
    X, Z = build_features(X_np, H, w, args.topk_spectral, args.use_spectral, args.use_raw_features)
    node_embed = X if X is not None else Z
    He = H / torch.clamp(H.sum(0, keepdim=True), min=1.0)
    Efeat = He.t() @ node_embed  # (M, d)

    out_dim = int(y.max().item()+1)
    in_dim = Efeat.size(1)
    clf = torch.nn.Linear(in_dim, out_dim).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.wd)
    ce = nn.CrossEntropyLoss()

    best, best_state = -1, None
    for ep in range(args.epochs):
        clf.train(); opt.zero_grad()
        logits = clf(Efeat)
        loss = ce(logits[tr], y[tr]); loss.backward(); opt.step()
        if (ep+1)%10==0:
            clf.eval()
            with torch.no_grad():
                pred = logits[va].argmax(1).cpu().numpy()
            from .metrics import classification_metrics
            m = classification_metrics(y[va].cpu().numpy(), pred)
            if m['acc']>best:
                best, best_state = m['acc'], {k:v.detach().cpu().clone() for k,v in clf.state_dict().items()}
    if best_state is not None: clf.load_state_dict(best_state)

    clf.eval()
    with torch.no_grad():
        logits = clf(Efeat); pred = logits.argmax(1).cpu().numpy()
    from .metrics import classification_metrics
    return {
        'train': classification_metrics(y[tr].cpu().numpy(), pred[tr.cpu().numpy()]),
        'val':   classification_metrics(y[va].cpu().numpy(), pred[va.cpu().numpy()]),
        'test':  classification_metrics(y[te].cpu().numpy(), pred[te.cpu().numpy()]),
    }

def run_link_prediction(data, args):
    # Simple hyperedge completion: learn node embeddings with HGNN on pseudo-task,
    # score hyperedge as average pairwise dot-product; use negative sampling.
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    N = data['N']; hyperedges = data['hyperedges']
    H, w = build_incidence(N, hyperedges, device=device)
    X, Z = build_features(data['X'], H, w, args.topk_spectral, args.use_spectral, args.use_raw_features)
    inp = X if X is not None else Z

    # Learnable node encoder
    enc = torch.nn.Sequential(
        torch.nn.Linear(inp.size(1), args.hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden, args.hidden)
    ).to(device)

    opt = torch.optim.Adam(enc.parameters(), lr=args.lr, weight_decay=args.wd)

    # Build train/val/test edges by random split
    import numpy as np, random
    E = len(hyperedges)
    idx = np.arange(E); np.random.shuffle(idx)
    n_tr = int(0.6*E); n_va = int(0.2*E)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

    def sample_neg(size, min_len=2, max_len=8):
        # sample random hyperedges as negatives
        res = []
        for _ in range(size):
            k = np.random.randint(min_len, max_len+1)
            nodes = np.random.choice(N, size=k, replace=False).tolist()
            res.append(nodes)
        return res

    bce = torch.nn.BCEWithLogitsLoss()

    def score(hyperedge, Znode):
        # score by average pairwise dot
        nodes = torch.tensor(hyperedge, dtype=torch.long, device=device)
        vecs = Znode[nodes]  # k x d
        S = torch.matmul(vecs, vecs.t())
        mask = 1 - torch.eye(S.size(0), device=device)
        return (S*mask).sum() / (mask.sum() + 1e-6)

    for ep in range(args.epochs):
        enc.train(); opt.zero_grad()
        Znode = enc(inp)
        pos_s = torch.stack([score(hyperedges[i], Znode) for i in tr])
        neg_edges = sample_neg(len(tr))
        neg_s = torch.stack([score(e, Znode) for e in neg_edges])
        logits = torch.cat([pos_s, neg_s], dim=0)
        labels = torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)], dim=0)
        loss = bce(logits, labels)
        loss.backward(); opt.step()

    # Eval AUC/AP on val/test
    from sklearn.metrics import roc_auc_score, average_precision_score
    enc.eval()
    with torch.no_grad():
        Znode = enc(inp)
        def eval_edges(idxs):
            pos = torch.stack([score(hyperedges[i], Znode) for i in idxs]).cpu().numpy()
            neg = torch.stack([score(e, Znode) for e in sample_neg(len(idxs))]).cpu().numpy()
            y = np.array([1]*len(idxs) + [0]*len(idxs))
            s = np.concatenate([pos, neg])
            return float(roc_auc_score(y, s)), float(average_precision_score(y, s))
        auc_va, ap_va = eval_edges(va); auc_te, ap_te = eval_edges(te)
    return {'val': {'auc': auc_va, 'ap': ap_va}, 'test': {'auc': auc_te, 'ap': ap_te}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True, help='path to processed/data.pt')
    ap.add_argument('--model', type=str, default='df_hgnn', choices=['hgnn','hnhn','hypergcn','df_hgnn'])
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--lbfgs_epochs', type=int, default=30)
    ap.add_argument('--align_lambda', type=float, default=0.1)
    ap.add_argument('--topk_spectral', type=int, default=16)
    ap.add_argument('--use_spectral', action='store_true', default=True)
    ap.add_argument('--no_spectral', dest='use_spectral', action='store_false')
    ap.add_argument('--use_raw_features', action='store_true', default=True)
    ap.add_argument('--only_det', dest='use_raw_features', action='store_false')
    ap.add_argument('--cpu', action='store_true', default=False)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    data = load_processed(args.data)
    task = data.get('task','node_clf')
    if task == 'node_clf':
        res = run_node_classification(data, args)
    elif task == 'edge_clf':
        res = run_edge_classification(data, args)
    else:
        res = run_link_prediction(data, args)
    print(res)

if __name__ == '__main__':
    main()
