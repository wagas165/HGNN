"""Preprocess the Cat-Edge-DAWN dataset into the unified tensor bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common.logging import get_logger, setup_logging
from src.data.loaders.cat_edge_dawn import CatEdgeDawnConfig, CatEdgeDawnLoader

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Cat-Edge-DAWN")
    parser.add_argument(
        "--root",
        type=str,
        default="data/raw/cat-edge-DAWN",
        help="Directory containing the raw Cat-Edge-DAWN files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/cat-edge-DAWN",
        help="Destination directory for processed tensors",
    )
    parser.add_argument(
        "--simplices-file",
        type=str,
        default="cat-edge-DAWN-simplices.txt",
        help="Relative path to the hyperedge list",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default="",
        help="Relative path to node features (optional)",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default="cat-edge-DAWN-node-labels.txt",
        help="Relative path to node labels",
    )
    parser.add_argument(
        "--times-file",
        type=str,
        default="",
        help="Relative path to hyperedge timestamps (optional)",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    config = CatEdgeDawnConfig(
        simplices_file=args.simplices_file,
        node_features_file=args.features_file if args.features_file else None,
        label_file=args.labels_file if args.labels_file else None,
        times_file=args.times_file if args.times_file else None,
    )
    loader = CatEdgeDawnLoader(args.root, config)
    data = loader.load()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "incidence": data.incidence,
            "edge_weights": data.edge_weights,
            "node_features": data.node_features,
            "labels": data.labels,
            "timestamps": data.timestamps,
            "metadata": data.metadata,
        },
        output_dir / "data.pt",
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(data.metadata, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Saved Cat-Edge-DAWN tensors to %s", output_dir)


if __name__ == "__main__":
    main()
