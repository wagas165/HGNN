
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split

def stratified_splits(y: np.ndarray, train=0.6, val=0.2, test=0.2, random_state=42):
    assert abs(train+val+test-1.0) < 1e-6
    n = len(y)
    idx = np.arange(n)
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(idx, y, train_size=train, stratify=y, random_state=random_state)
    val_ratio = val / (val + test)
    idx_va, idx_te, _, _ = train_test_split(idx_tmp, y_tmp, train_size=val_ratio, stratify=y_tmp, random_state=random_state+1)
    return idx_tr, idx_va, idx_te

def random_splits(n: int, train=0.6, val=0.2, test=0.2, random_state=42):
    idx = np.arange(n)
    rng = np.random.default_rng(random_state)
    rng.shuffle(idx)
    n_tr = int(train*n); n_va = int(val*n)
    return idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
