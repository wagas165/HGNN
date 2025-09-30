
import os, json, csv
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy import sparse

def _maybe_path(p):
    return None if p is None or str(p).strip()=="" else p

def read_array(path: str):
    if path is None or path=="":
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path)
    elif ext == '.csv':
        return pd.read_csv(path, header=None).values
    elif ext == '.txt':
        # assume one value per line
        with open(path, 'r', encoding='utf-8') as f:
            vals = [float(x.strip()) for x in f if len(x.strip())>0]
        return np.array(vals)
    else:
        raise ValueError(f"Unsupported array file: {path}")

def read_labels(path: str):
    arr = read_array(path)
    if arr is None:
        return None
    arr = arr.squeeze()
    if arr.ndim != 1:
        raise ValueError(f"Labels must be 1-D, got shape {arr.shape}")
    return arr.astype(int)

def guess_delimiter(txt_path: str) -> str:
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s: 
                continue
            if ',' in s and ' ' in s:
                # choose the dominant
                return ',' if s.count(',') >= s.count(' ') else ' '
            if ',' in s:
                return ','
            if ' ' in s:
                return ' '
            return ' '  # default single token
    return ' '

def read_hyperedges(path: str, fmt: str='auto', delimiter: str='auto') -> List[List[int]]:
    ext = os.path.splitext(path)[1].lower()
    if fmt=='npz' or ext=='.npz':
        H = sparse.load_npz(path)  # N x M
        H = H.tocoo()
        N, M = H.shape
        edges = [[] for _ in range(M)]
        for r,c,v in zip(H.row, H.col, H.data):
            if v != 0:
                edges[c].append(int(r))
        return edges
    if fmt=='json' or ext=='.json':
        with open(path, 'r', encoding='utf-8') as f:
            arr = json.load(f)
        return [[int(x) for x in e] for e in arr]
    # txt
    if delimiter=='auto':
        delimiter = guess_delimiter(path)
    edges = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(delimiter) if delimiter!=' ' else s.split()
            nodes = [int(x) for x in parts if x!='']
            if len(nodes) >= 2:
                edges.append(nodes)
    return edges
