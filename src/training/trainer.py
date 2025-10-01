"""Training pipeline for DF-HGNN."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast

from src.common.logging import get_logger
from src.common.seed import get_device
from src.evaluation.metrics import MetricRegistry
from src.features.deterministic_bank import (
    DeterministicFeatureBank,
    DeterministicFeatureConfig,
    robust_standardize,
)
from src.models.registry import ModelFactoryInput, create_model
from src.training.callbacks import EarlyStoppingState
from src.training.optimizers import OptimizerConfig, build_adam, build_lbfgs


LOGGER = get_logger(__name__)

PredictionCache = Dict[str, Dict[str, torch.Tensor]]


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
    pin_memory: bool = True


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
        self.feature_config = feature_config
        self.feature_bank = DeterministicFeatureBank(feature_config)
        self.optimizer_config = optimizer_config
        self.metrics = metrics
        self.device = get_device(trainer_config.device)
        self.model_config = model_config

    def train(
        self,
        incidence: torch.Tensor,
        edge_weights: torch.Tensor,
        node_features: Optional[torch.Tensor],
        labels: torch.Tensor,
        splits: Dict[str, torch.Tensor],
        num_classes: int,
        timestamps: Optional[torch.Tensor] = None,
    ) -> tuple[Dict[str, float], nn.Module, PredictionCache]:
        deterministic_features = self.feature_bank(
            incidence, edge_weights, timestamps=timestamps
        )
        node_features = self._prepare_node_features(
            node_features, deterministic_features.dtype, deterministic_features.shape[0]
        )
        in_dim = node_features.shape[1]
        det_dim = deterministic_features.shape[1]
        reserved_keys = {
            "name",
            "hidden_dim",
            "dropout",
            "conv_type",
            "chebyshev_order",
            "lambda_align",
            "lambda_gate",
            "fusion_dim",
        }
        extras = {k: v for k, v in self.model_config.items() if k not in reserved_keys}
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
                fusion_dim=self.model_config.get("fusion_dim"),
                extras=extras,
            ),
        ).to(self.device)

        LOGGER.info("Model initialized with %s parameters", sum(p.numel() for p in model.parameters()))

        split_indices = {name: self._move_to_device(idx) for name, idx in splits.items()}
        train_idx = split_indices["train"]
        val_idx = split_indices.get("val")
        test_idx = split_indices.get("test")

        x = self._move_to_device(node_features)
        z = self._move_to_device(deterministic_features)
        del deterministic_features
        incidence = self._move_to_device(incidence)
        edge_weights = self._move_to_device(edge_weights)
        labels = self._move_to_device(labels)

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

            metrics = self.evaluate(
                model,
                x,
                z,
                incidence,
                edge_weights,
                labels,
                split_indices,
            )
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
            logits, gate = model(x, z, incidence, edge_weights)
            loss = criterion(logits[train_idx], labels[train_idx])
            align_loss = model.alignment_loss(x, z)
            gate_loss = model.gate_regularization(gate)
            total_loss = (
                loss
                + self.model_config.get("lambda_align", 0.1) * align_loss
                + self.model_config.get("lambda_gate", 0.001) * gate_loss
            )
            total_loss.backward()
            return total_loss

        for _ in range(self.trainer_config.lbfgs_epochs):
            lbfgs.step(closure)

        final_evaluation = self.evaluate(
            model,
            x,
            z,
            incidence,
            edge_weights,
            labels,
            split_indices,
            return_cache=True,
        )
        final_metrics, prediction_cache = final_evaluation
        LOGGER.info("Training complete. Test accuracy=%.4f", final_metrics.get("test_accuracy", 0.0))
        return final_metrics, model, prediction_cache

    def evaluate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        z: torch.Tensor,
        incidence: torch.Tensor,
        edge_weights: torch.Tensor,
        labels: torch.Tensor,
        split_indices: Dict[str, torch.Tensor],
        return_cache: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], PredictionCache]]:
        model.eval()
        with torch.inference_mode():
            logits, _ = model(x, z, incidence, edge_weights)
            probs = torch.softmax(logits, dim=1)
        metrics: Dict[str, float] = {}
        split_tensors = {
            name: (probs[idx], labels[idx])
            for name, idx in split_indices.items()
            if idx.numel() > 0
        }

        with ThreadPoolExecutor(max_workers=len(split_tensors)) as executor:
            futures = {
                executor.submit(self.metrics.compute, tensors[0], tensors[1]): name
                for name, tensors in split_tensors.items()
            }
            for future, split_name in futures.items():
                split_metrics = future.result()
                metrics.update({f"{split_name}_{k}": v for k, v in split_metrics.items()})

        if not return_cache:
            return metrics

        prediction_cache: PredictionCache = {}
        for split_name, (split_probs, split_labels) in split_tensors.items():
            prediction_cache[split_name] = {
                "probs": split_probs.detach().cpu(),
                "labels": split_labels.detach().cpu(),
            }
        return metrics, prediction_cache

    def evaluate_model(
        self,
        model: nn.Module,
        incidence: torch.Tensor,
        edge_weights: torch.Tensor,
        node_features: Optional[torch.Tensor],
        labels: torch.Tensor,
        splits: Dict[str, torch.Tensor],
        timestamps: Optional[torch.Tensor] = None,

        return_cache: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], PredictionCache]]:
        deterministic_features = self.feature_bank(
            incidence, edge_weights, timestamps=timestamps

        )
        node_features_prepared = self._prepare_node_features(
            node_features,
            deterministic_features.dtype,
            deterministic_features.shape[0],
        )
        x = self._move_to_device(node_features_prepared)
        z = self._move_to_device(deterministic_features)
        incidence = self._move_to_device(incidence)
        edge_weights = self._move_to_device(edge_weights)
        labels = self._move_to_device(labels)
        split_indices = {name: self._move_to_device(idx) for name, idx in splits.items()}
        return self.evaluate(
            model,
            x,
            z,
            incidence,
            edge_weights,
            labels,
            split_indices,
            return_cache=return_cache,
        )

    def _prepare_node_features(
        self,
        node_features: Optional[torch.Tensor],
        dtype: torch.dtype,
        num_nodes: int,
    ) -> torch.Tensor:
        if node_features is None or node_features.numel() == 0:
            return torch.zeros((num_nodes, 0), dtype=dtype)
        node_features = node_features.to(dtype=dtype)
        if node_features.dim() == 1:
            node_features = node_features.unsqueeze(1)
        processed = robust_standardize(
            node_features, self.feature_config.quantile_clip
        )
        return processed

    def _compute_deterministic_features(
        self,
        incidence: torch.Tensor,
        edge_weights: torch.Tensor,
        node_features: Optional[torch.Tensor],
        timestamps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.feature_bank is not None:
            return self.feature_bank(incidence, edge_weights, timestamps=timestamps)
        return self._create_placeholder_features(node_features, incidence.shape[0])

    def _create_placeholder_features(
        self, node_features: Optional[torch.Tensor], num_nodes: int
    ) -> torch.Tensor:
        if node_features is not None and node_features.numel() > 0:
            return node_features.new_zeros((num_nodes, 0))
        return torch.zeros((num_nodes, 0), dtype=torch.get_default_dtype())

    def _move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == self.device:
            return tensor
        if self.device.type == "cuda" and tensor.device.type == "cpu" and self.trainer_config.pin_memory:
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=self.device.type == "cuda")
