"""Preprocess the coauth-DBLP-full dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common.logging import get_logger, setup_logging
from src.data.loaders.coauth_dblp_full import CoauthDblpFullConfig, CoauthDblpFullLoader

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess coauth-DBLP-full")
    parser.add_argument(
        "--root",
        type=str,
        default="data/raw/coauth-DBLP-full",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/coauth-DBLP-full",
        help="Output directory",
    )
    parser.add_argument(
        "--simplices-file",
        type=str,
        default="coauth-DBLP-full-simplices.txt",
        help="Relative path to the hyperedge list",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default="coauth-DBLP-full-node-features.npy",
        help="Relative path to node features (optional)",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default="coauth-DBLP-full-node-labels.npy",
        help="Relative path to node labels (optional)",
    )
    parser.add_argument(
        "--times-file",
        type=str,
        default="coauth-DBLP-full-times.txt",
        help="Relative path to hyperedge timestamps (optional)",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    config = CoauthDblpFullConfig(
        simplices_file=args.simplices_file,
        node_features_file=args.features_file if args.features_file else None,
        label_file=args.labels_file if args.labels_file else None,
        times_file=args.times_file if args.times_file else None,
    )
    loader = CoauthDblpFullLoader(args.root, config)
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
    LOGGER.info("Saved coauth-DBLP-full tensors to %s", output_dir)


if __name__ == "__main__":
    main()
