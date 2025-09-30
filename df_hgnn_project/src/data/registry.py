
import os, yaml, numpy as np, torch
from .utils_io import read_array, read_labels, read_hyperedges, _maybe_path
from .splits import stratified_splits, random_splits

def load_config(cfg_path: str):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def auto_discover_files(data_dir: str):
    # Try to find common filenames inside a raw data directory.
    choices = {'node_features': None, 'node_labels': None, 'edge_labels': None, 'hyperedges': None}
    for fn in os.listdir(data_dir):
        l = fn.lower()
        p = os.path.join(data_dir, fn)
        if l.startswith('node_features') or l.endswith('_features.npy') or l.endswith('_features.csv'):
            choices['node_features'] = p
        elif l.startswith('node_labels') or l.endswith('_node_labels.npy') or l.endswith('_node_labels.csv'):
            choices['node_labels'] = p
        elif l.startswith('edge_labels') or l.endswith('_edge_labels.npy') or l.endswith('_edge_labels.csv'):
            choices['edge_labels'] = p
        elif l.startswith('hyperedges') or l.endswith('hyperedges.txt') or l.endswith('hyperedges.json') or l.endswith('.npz'):
            choices['hyperedges'] = p
    return choices

def build_dataset(files: dict, fmt: dict, splits_cfg: dict, options: dict, out_dir: str):
    os.makedirs(os.path.join(out_dir, 'processed'), exist_ok=True)
    feat_path = _maybe_path(files.get('node_features', None))
    y_node_path = _maybe_path(files.get('node_labels', None))
    y_edge_path = _maybe_path(files.get('edge_labels', None))
    hedges_path = _maybe_path(files.get('hyperedges', None))

    if hedges_path is None:
        raise ValueError("hyperedges file is required.")

    fmt_edges = fmt.get('hyperedges', 'auto')
    delimiter = fmt.get('delimiter', 'auto')
    hyperedges = read_hyperedges(hedges_path, fmt=fmt_edges, delimiter=delimiter)

    # Num nodes = max node id + 1 if features not provided
    N_guess = max([max(e) for e in hyperedges]) + 1 if len(hyperedges)>0 else 0

    X = read_array(feat_path) if feat_path else None
    if X is not None and X.shape[0] != N_guess:
        raise ValueError(f"node_features length {X.shape[0]} not equal to N={N_guess} inferred from hyperedges")

    y_node = read_labels(y_node_path) if y_node_path else None
    y_edge  = read_labels(y_edge_path) if y_edge_path else None

    task = 'link'
    if y_node is not None:
        task = 'node_clf'
    elif y_edge is not None:
        task = 'edge_clf'

    # splits
    if task == 'node_clf':
        if y_node is None: raise ValueError("node labels required for node_clf")
        tr, va, te = stratified_splits(y_node, train=splits_cfg.get('train',0.6),
                                       val=splits_cfg.get('val',0.2), test=splits_cfg.get('test',0.2))
        splits = {'train': tr, 'val': va, 'test': te}
    elif task == 'edge_clf':
        if y_edge is None: raise ValueError("edge labels required for edge_clf")
        import numpy as np
        nE = len(hyperedges)
        idx = np.arange(nE)
        tr, va, te = random_splits(nE, train=splits_cfg.get('train',0.6),
                                   val=splits_cfg.get('val',0.2), test=splits_cfg.get('test',0.2))
        splits = {'train': tr, 'val': va, 'test': te}
    else:
        splits = {}

    # save torch file
    data = {
        'N': N_guess,
        'hyperedges': hyperedges,
        'X': None if X is None else X.astype('float32'),
        'y_node': None if y_node is None else y_node.astype('int64'),
        'y_edge': None if y_edge is None else y_edge.astype('int64'),
        'task': task,
        'splits': splits,
        'options': options or {}
    }
    out_path = os.path.join(out_dir, 'processed', 'data.pt')
    torch.save(data, out_path)
    return out_path
