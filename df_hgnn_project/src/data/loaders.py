
import torch, numpy as np
from typing import Dict, Any

def load_processed(data_path: str) -> Dict[str, Any]:
    data = torch.load(data_path, map_location='cpu')
    # sanity
    assert 'hyperedges' in data and 'N' in data
    # convert lists to python lists
    data['hyperedges'] = [list(map(int, e)) for e in data['hyperedges']]
    return data
