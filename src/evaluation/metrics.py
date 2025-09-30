"""Evaluation metrics for classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass
class MetricConfig:
    name: str


class MetricRegistry:
    def __init__(self, metrics: List[MetricConfig]) -> None:
        self.metrics = metrics

    def compute(self, probs: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        preds = probs.argmax(dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        probs_np = probs.detach().cpu().numpy()

        results: Dict[str, float] = {}
        for metric in self.metrics:
            if metric.name == "accuracy":
                results[metric.name] = float(accuracy_score(labels_np, preds))
            elif metric.name == "macro_f1":
                results[metric.name] = float(f1_score(labels_np, preds, average="macro", zero_division=0))
            elif metric.name == "roc_auc":
                if probs_np.shape[1] == 2:
                    results[metric.name] = float(roc_auc_score(labels_np, probs_np[:, 1]))
                else:
                    results[metric.name] = float(
                        roc_auc_score(labels_np, probs_np, multi_class="ovr", average="macro")
                    )
            else:
                raise ValueError(f"Unsupported metric: {metric.name}")
        return results
