"""Train DF-HGNN model with Hydra configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.common.logging import setup_logging, get_logger
from src.common.seed import SeedConfig, set_seed
from src.data.loaders.email_eu_full import EmailEuFullConfig, EmailEuFullLoader
from src.data.transforms.split import DataSplits, SplitConfig, create_splits
from src.evaluation.metrics import MetricConfig, MetricRegistry
from src.evaluation.reporting import save_metrics_report
from src.features.deterministic_bank import DeterministicFeatureConfig
from src.training.trainer import DFHGNNTrainer, TrainerConfig
from src.training.optimizers import OptimizerConfig

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DF-HGNN")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file")
    return parser.parse_args()


def load_config(path: str) -> dict:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(SeedConfig(value=int(cfg.get("seed", 42))))

    data_cfg = cfg["data"]
    loader = EmailEuFullLoader(
        data_cfg["root"],
        EmailEuFullConfig(
            vertices_file=data_cfg.get("files", {}).get("vertices", "email-Eu-full-nverts.txt"),
            simplices_file=data_cfg.get("files", {}).get("simplices", "email-Eu-full-simplices.txt"),
            times_file=data_cfg.get("files", {}).get("times", "email-Eu-full-times.txt"),
            feature_cache=data_cfg.get("feature_cache"),
            label_file=data_cfg.get("label_file"),
        ),
    )
    data = loader.load()
    if data.labels is None:
        raise ValueError("Dataset must provide labels for supervised training")

    splits = create_splits(
        labels=data.labels,
        config=SplitConfig(
            strategy=data_cfg.get("split", {}).get("strategy", "stratified"),
            train_ratio=data_cfg.get("split", {}).get("train_ratio", 0.6),
            val_ratio=data_cfg.get("split", {}).get("val_ratio", 0.2),
            test_ratio=data_cfg.get("split", {}).get("test_ratio", 0.2),
            random_state=cfg.get("seed", 42),
        ),
        num_nodes=data.num_nodes,
    )

    trainer = DFHGNNTrainer(
        trainer_config=TrainerConfig(
            max_epochs=cfg["trainer"]["max_epochs"],
            adam_epochs=cfg["trainer"]["adam_epochs"],
            lbfgs_epochs=cfg["trainer"]["lbfgs_epochs"],
            lr=cfg["trainer"]["lr"],
            weight_decay=cfg["trainer"]["weight_decay"],
            grad_clip=cfg["trainer"].get("grad_clip", 0.0),
            device=cfg["trainer"].get("device", "auto"),
            amp=cfg["trainer"].get("amp", False),
            early_stopping_patience=cfg["trainer"].get("early_stopping_patience", 20),
        ),
        model_config=cfg["model"],
        feature_config=DeterministicFeatureConfig(
            spectral_topk=cfg["features"]["deterministic"].get("spectral_topk", 32),
            use_spectral=cfg["features"]["deterministic"].get("use_spectral", True),
            use_hodge=cfg["features"]["deterministic"].get("use_hodge", False),
            cache_dir=cfg["features"]["deterministic"].get("cache_dir"),
        ),
        optimizer_config=OptimizerConfig(
            lr=cfg["trainer"]["lr"],
            weight_decay=cfg["trainer"]["weight_decay"],
            betas=tuple(cfg["optimizer"]["adam"].get("betas", (0.9, 0.999))),
            lbfgs_history_size=cfg["optimizer"]["lbfgs"].get("history_size", 10),
            lbfgs_line_search=cfg["optimizer"]["lbfgs"].get("line_search", "strong_wolfe"),
        ),
        metrics=MetricRegistry([MetricConfig(name=m["name"]) for m in cfg.get("metrics", [])]),
    )

    metrics = trainer.train(
        incidence=data.incidence,
        edge_weights=data.edge_weights,
        node_features=data.node_features,
        labels=data.labels,
        splits={"train": splits.train_idx, "val": splits.val_idx, "test": splits.test_idx},
        num_classes=int(cfg.get("num_classes", data.labels.max().item() + 1)),
    )

    report_path = save_metrics_report(metrics, cfg["reporting"].get("dir", "outputs/reports"))
    LOGGER.info("Saved metrics report to %s", report_path)


if __name__ == "__main__":
    main()
