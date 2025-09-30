"""Dataset loader registry and helpers."""
from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Tuple, Type

from .base import HypergraphDatasetLoader
from .cat_edge_dawn import CatEdgeDawnConfig, CatEdgeDawnLoader
from .coauth_dblp_full import CoauthDblpFullConfig, CoauthDblpFullLoader
from .email_eu_full import EmailEuFullConfig, EmailEuFullLoader
from .generic import EdgeListHypergraphConfig, EdgeListHypergraphLoader
from .stackoverflow_answers import StackOverflowAnswersConfig, StackOverflowAnswersLoader

DatasetRegistryEntry = Tuple[Type[Any], Type[HypergraphDatasetLoader]]

DATASET_REGISTRY: Dict[str, DatasetRegistryEntry] = {
    "email_eu_full": (EmailEuFullConfig, EmailEuFullLoader),
    "cat_edge_dawn": (CatEdgeDawnConfig, CatEdgeDawnLoader),
    "coauth_dblp_full": (CoauthDblpFullConfig, CoauthDblpFullLoader),
    "stackoverflow_answers": (StackOverflowAnswersConfig, StackOverflowAnswersLoader),
}


def _build_config_kwargs(config_cls: Type[Any], config_dict: Dict[str, Any]) -> Dict[str, Any]:
    field_names = {f.name for f in fields(config_cls)}
    kwargs: Dict[str, Any] = {}

    for key, value in config_dict.items():
        if key == "files" and isinstance(value, dict):
            continue
        if key in field_names:
            kwargs[key] = value

    files_dict = config_dict.get("files", {})
    if isinstance(files_dict, dict):
        for key, value in files_dict.items():
            if key in field_names:
                kwargs[key] = value
                continue
            candidate = f"{key}_file"
            if candidate in field_names:
                kwargs[candidate] = value
                continue
            if key == "features" and "node_features_file" in field_names:
                kwargs["node_features_file"] = value
            elif key == "labels" and "label_file" in field_names:
                kwargs["label_file"] = value

    return kwargs


def create_loader(name: str, root: str, config: Dict[str, Any]) -> HypergraphDatasetLoader:
    """Instantiate a dataset loader from the registry."""

    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")

    config_cls, loader_cls = DATASET_REGISTRY[name]
    kwargs = _build_config_kwargs(config_cls, config)
    dataset_config = config_cls(**kwargs)
    return loader_cls(root, dataset_config)


__all__ = [
    "EdgeListHypergraphConfig",
    "EdgeListHypergraphLoader",
    "EmailEuFullConfig",
    "EmailEuFullLoader",
    "CatEdgeDawnConfig",
    "CatEdgeDawnLoader",
    "CoauthDblpFullConfig",
    "CoauthDblpFullLoader",
    "StackOverflowAnswersConfig",
    "StackOverflowAnswersLoader",
    "create_loader",
    "DATASET_REGISTRY",
]
