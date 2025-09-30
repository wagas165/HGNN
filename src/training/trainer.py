"""Training pipeline for DF-HGNN."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from src.common.logging import get_logger
from src.common.seed import get_device
from src.evaluation.metrics import MetricRegistry
from src.features.deterministic_bank import DeterministicFeatureBank, DeterministicFeatureConfig
from src.models.registry import ModelFactoryInput, create_model
from src.training.callbacks import EarlyStoppingState
from src.training.optimizers import OptimizerConfig, build_adam, build_lbfgs


LOGGER = get_logger(__name__)


@dataclass
class TrainerConfig:
    max_epochs: int
    adam_epochs: int
    lbfgs_epochs: int
    lr: float
    weight_decay: float
    grad_clip: float
    device: str
    amp: bool
    early_stopping_patience: int


class DFHGNNTrainer:
    def __init__(
        self,
        trainer_config: TrainerConfig,
        model_config: Dict[str, object],
        feature_config: DeterministicFeatureConfig,
        optimizer_config: OptimizerConfig,
        metrics: MetricRegistry,
    ) -> None:
        self.trainer_config = trainer_config
        self.feature_bank = DeterministicFeatureBank(feature_config)
        self.optimizer_config = optimizer_config
        self.metrics = metrics
        self.device = get_device(trainer_config.device)
        self.model_config = model_config

    def train(
        self,
        incidence: torch.Tensor,
        edge_weights: torch.Tensor,
        node_features: torch.Tensor,
        labels: torch.Tensor,
        splits: Dict[str, torch.Tensor],
        num_classes: int,
    ) -> Dict[str, float]:
        deterministic_features = self.feature_bank(incidence, edge_weights)
        in_dim = node_features.shape[1]
        det_dim = deterministic_features.shape[1]
        model = create_model(
            name=self.model_config["name"],
            config=ModelFactoryInput(
                in_dim=in_dim,
                det_dim=det_dim,
                out_dim=num_classes,
                hidden_dim=int(self.model_config["hidden_dim"]),
                dropout=float(self.model_config["dropout"]),
                conv_type=str(self.model_config.get("conv_type", "mp")),
                chebyshev_order=int(self.model_config.get("chebyshev_order", 2)),
                lambda_align=float(self.model_config.get("lambda_align", 0.1)),
                lambda_gate=float(self.model_config.get("lambda_gate", 0.001)),
            ),
        ).to(self.device)

        LOGGER.info("Model initialized with %s parameters", sum(p.numel() for p in model.parameters()))

        train_idx = splits["train"].to(self.device)
        val_idx = splits["val"].to(self.device)
        test_idx = splits["test"].to(self.device)

        x = node_features.to(self.device)
        z = deterministic_features.to(self.device)
        incidence = incidence.to(self.device)
        edge_weights = edge_weights.to(self.device)
        labels = labels.to(self.device)

        scaler = GradScaler(enabled=self.trainer_config.amp)
        optimizer_cfg = OptimizerConfig(
            lr=self.trainer_config.lr,
            weight_decay=self.trainer_config.weight_decay,
            betas=self.optimizer_config.betas,
            lbfgs_history_size=self.optimizer_config.lbfgs_history_size,
            lbfgs_line_search=self.optimizer_config.lbfgs_line_search,
        )
        adam = build_adam(model.parameters(), optimizer_cfg)
        early_stopping = EarlyStoppingState(patience=self.trainer_config.early_stopping_patience)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.trainer_config.adam_epochs):
            model.train()
            adam.zero_grad()
            with autocast(enabled=self.trainer_config.amp):
                logits, gate = model(x, z, incidence, edge_weights)
                loss = criterion(logits[train_idx], labels[train_idx])
                align_loss = model.alignment_loss(x, z)
                gate_loss = model.gate_regularization(gate)
                total_loss = (
                    loss
                    + self.model_config.get("lambda_align", 0.1) * align_loss
                    + self.model_config.get("lambda_gate", 0.001) * gate_loss
                )
            scaler.scale(total_loss).backward()
            if self.trainer_config.grad_clip > 0:
                scaler.unscale_(adam)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.trainer_config.grad_clip)
            scaler.step(adam)
            scaler.update()

            metrics = self.evaluate(model, x, z, incidence, edge_weights, labels, train_idx, val_idx, test_idx)
            LOGGER.info(
                "Epoch %d | loss=%.4f | val_accuracy=%.4f",
                epoch + 1,
                float(total_loss.item()),
                metrics.get("val_accuracy", 0.0),
            )
            early_stopping.update(metrics.get("val_accuracy", 0.0), model)
            if early_stopping.should_stop():
                LOGGER.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        early_stopping.restore_best(model)

        lbfgs = build_lbfgs(model.parameters(), optimizer_cfg)

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            logits, _ = model(x, z, incidence, edge_weights)
            loss = criterion(logits[train_idx], labels[train_idx])
            loss.backward()
            return loss

        for _ in range(self.trainer_config.lbfgs_epochs):
            lbfgs.step(closure)

        final_metrics = self.evaluate(model, x, z, incidence, edge_weights, labels, train_idx, val_idx, test_idx)
        LOGGER.info("Training complete. Test accuracy=%.4f", final_metrics.get("test_accuracy", 0.0))
        return final_metrics

    def evaluate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        z: torch.Tensor,
        incidence: torch.Tensor,
        edge_weights: torch.Tensor,
        labels: torch.Tensor,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        test_idx: torch.Tensor,
    ) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, z, incidence, edge_weights)
            probs = torch.softmax(logits, dim=1)
        metrics: Dict[str, float] = {}
        for split_name, idx in ("train", train_idx), ("val", val_idx), ("test", test_idx):
            split_metrics = self.metrics.compute(probs[idx], labels[idx])
            metrics.update({f"{split_name}_{k}": v for k, v in split_metrics.items()})
        return metrics
