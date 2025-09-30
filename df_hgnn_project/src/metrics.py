
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

def classification_metrics(y_true, y_pred, y_prob=None):
    out = {
        'acc': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro'))
    }
    if y_prob is not None:
        try:
            if y_prob.ndim==1 or y_prob.shape[1]==2:
                # binary case
                pos = y_prob if y_prob.ndim==1 else y_prob[:,1]
                out['auc'] = float(roc_auc_score(y_true, pos))
                out['ap'] = float(average_precision_score(y_true, pos))
        except Exception:
            pass
    return out
