"""Train DF-HGNN model with Hydra configuration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_ROOT = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "default.yaml"

import torch
from omegaconf import OmegaConf

from src.common.logging import setup_logging, get_logger
from src.common.path import DatasetRootResolutionError, resolve_dataset_root
from src.common.seed import SeedConfig, set_seed
from src.data.loaders import create_loader
from src.data.transforms.split import SplitConfig, create_splits
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


def _normalize_default_key(raw_key: str) -> str:
    key = raw_key.strip()
    if not key:
        return key
    for prefix in ("override ", "optional "):
        if key.startswith(prefix):
            key = key[len(prefix) :].strip()
            break
    if key.startswith("/"):
        key = key[1:]
    if "@" in key:
        key = key.split("@", 1)[0]
    return key


def _compose_config_from_path(path: Path) -> "OmegaConf":
    cfg = OmegaConf.load(path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    defaults = cfg_dict.pop("defaults", [])

    merged = OmegaConf.create()
    for entry in defaults:
        if isinstance(entry, str):
            key = _normalize_default_key(entry)
            value = None
        elif isinstance(entry, dict) and len(entry) == 1:
            raw_key, value = next(iter(entry.items()))
            key = _normalize_default_key(str(raw_key))
        else:
            raise TypeError(f"Unsupported defaults entry: {entry!r}")

        if key in {"", "_self_"} or value in {None, "_self_"}:
            continue

        group_path = CONFIG_ROOT / key
        if group_path.is_dir():
            candidate_path = group_path / f"{value}.yaml"
        else:
            candidate_path = group_path.with_suffix(".yaml")

        if not candidate_path.exists():
            raise FileNotFoundError(
                f"Unable to resolve config default '{key}: {value}' at {candidate_path}"
            )

        resolved_cfg = _compose_config_from_path(candidate_path)
        merged = OmegaConf.merge(merged, OmegaConf.create({key: resolved_cfg}))

    current_cfg = OmegaConf.create(cfg_dict)
    return OmegaConf.merge(merged, current_cfg)


def load_config(path: str) -> dict:
    target_path = Path(path)
    composed = _compose_config_from_path(target_path)

    if target_path.resolve() != DEFAULT_CONFIG_PATH.resolve():
        default_cfg = _compose_config_from_path(DEFAULT_CONFIG_PATH)
        if "data" in composed and "data" in default_cfg:
            for key in ("name", "root", "files", "label_file"):
                if key in default_cfg.data:
                    del default_cfg.data[key]
        composed = OmegaConf.merge(default_cfg, composed)

    return OmegaConf.to_container(composed, resolve=True)


def main() -> None:
    setup_logging()
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(str(config_path))

    set_seed(SeedConfig(value=int(cfg.get("seed", 42))))

    data_cfg = cfg["data"]

    try:
        data_root = resolve_dataset_root(
            data_cfg["root"],
            PROJECT_ROOT,
            config_path=config_path,
        )
    except DatasetRootResolutionError as exc:
        LOGGER.error("%s", exc)
        raise


    loader = create_loader(data_cfg.get("name", "email_eu_full"), str(data_root), data_cfg)
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
            ood_ratio=data_cfg.get("split", {}).get("ood_ratio", 0.0),
            random_state=cfg.get("seed", 42),
        ),
        num_nodes=data.num_nodes,
    )

    label_fraction = float(data_cfg.get("label_fraction", 1.0))
    if label_fraction <= 0 or label_fraction > 1:
        raise ValueError("data.label_fraction must be in (0, 1]")

    train_idx = splits.train_idx
    if label_fraction < 1.0:
        generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
        num_train = train_idx.shape[0]
        keep = max(1, int(round(num_train * label_fraction)))
        perm = torch.randperm(num_train, generator=generator)
        train_idx = train_idx[perm[:keep]]
        LOGGER.info("Using %d/%d training nodes (%.2f%%)", keep, num_train, 100 * label_fraction)

    split_tensors = {
        "train": train_idx,
        "val": splits.val_idx,
        "test": splits.test_idx,
    }
    if splits.ood_idx is not None and splits.ood_idx.numel() > 0:
        split_tensors["ood"] = splits.ood_idx

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
            pin_memory=cfg["trainer"].get("pin_memory", True),
        ),
        model_config=cfg["model"],
        feature_config=DeterministicFeatureConfig(
            spectral_topk=cfg["features"]["deterministic"].get("spectral_topk", 32),
            use_spectral=cfg["features"]["deterministic"].get("use_spectral", True),
            use_hodge=cfg["features"]["deterministic"].get("use_hodge", False),
            use_temporal=cfg["features"]["deterministic"].get("use_temporal", True),
            quantile_clip=cfg["features"]["deterministic"].get("quantile_clip", 0.01),
            cache_dir=cfg["features"]["deterministic"].get("cache_dir"),
            device=cfg["features"]["deterministic"].get("device", "auto"),
            precision=cfg["features"]["deterministic"].get("precision", "float32"),
            expansion_chunk_size=cfg["features"]["deterministic"].get("expansion_chunk_size"),
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

    metrics, model = trainer.train(
        incidence=data.incidence,
        edge_weights=data.edge_weights,
        node_features=data.node_features,
        labels=data.labels,
        splits=split_tensors,
        num_classes=int(cfg.get("num_classes", data.labels.max().item() + 1)),
        timestamps=data.timestamps,
    )

    transfer_cfg = data_cfg.get("transfer")
    if transfer_cfg:
        target_name = transfer_cfg.get("target_name")
        if not target_name:
            raise ValueError("data.transfer.target_name must be provided when transfer block is set")
        target_root = transfer_cfg.get("target_root", data_cfg.get("root"))
        try:
            resolved_target_root = resolve_dataset_root(target_root, PROJECT_ROOT, config_path=config_path)
        except DatasetRootResolutionError as exc:
            LOGGER.error("%s", exc)
            raise

        target_loader_cfg = {}
        if "target_config" in transfer_cfg:
            target_loader_cfg.update(transfer_cfg["target_config"])
        if "target_files" in transfer_cfg:
            target_loader_cfg["files"] = transfer_cfg["target_files"]
        if "label_file" in transfer_cfg:
            target_loader_cfg["label_file"] = transfer_cfg["label_file"]

        target_loader = create_loader(target_name, str(resolved_target_root), target_loader_cfg)
        target_data = target_loader.load()
        if target_data.labels is None:
            raise ValueError("Transfer target dataset must provide labels for evaluation")

        transfer_split_dict = transfer_cfg.get("split", {})
        target_splits = create_splits(
            labels=target_data.labels,
            config=SplitConfig(
                strategy=transfer_split_dict.get("strategy", "stratified"),
                train_ratio=transfer_split_dict.get("train_ratio", 0.6),
                val_ratio=transfer_split_dict.get("val_ratio", 0.2),
                test_ratio=transfer_split_dict.get("test_ratio", 0.2),
                ood_ratio=transfer_split_dict.get("ood_ratio", 0.0),
                random_state=cfg.get("seed", 42),
            ),
            num_nodes=target_data.num_nodes,
        )
        transfer_split_tensors = {
            "train": target_splits.train_idx,
            "val": target_splits.val_idx,
            "test": target_splits.test_idx,
        }
        if target_splits.ood_idx is not None and target_splits.ood_idx.numel() > 0:
            transfer_split_tensors["ood"] = target_splits.ood_idx

        transfer_metrics = trainer.evaluate_model(
            model,
            target_data.incidence,
            target_data.edge_weights,
            target_data.node_features,
            target_data.labels,
            transfer_split_tensors,
            timestamps=target_data.timestamps,
        )
        metrics.update({f"transfer_{k}": v for k, v in transfer_metrics.items()})

    report_path = save_metrics_report(metrics, cfg["reporting"].get("dir", "outputs/reports"))
    LOGGER.info("Saved metrics report to %s", report_path)


if __name__ == "__main__":
    main()
