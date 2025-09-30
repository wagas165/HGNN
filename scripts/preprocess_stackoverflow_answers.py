"""Preprocess the StackOverflow Answers dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common.logging import get_logger, setup_logging
from src.data.loaders.stackoverflow_answers import (
    StackOverflowAnswersConfig,
    StackOverflowAnswersLoader,
)

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess StackOverflow Answers")
    parser.add_argument(
        "--root",
        type=str,
        default="data/raw/stackoverflow-answers",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/stackoverflow-answers",
        help="Output directory",
    )
    parser.add_argument(
        "--simplices-file",
        type=str,
        default="hyperedges-stackoverflow-answers.txt",
        help="Relative path to hyperedge list",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default="node-labels-stackoverflow-answers.txt",
        help="Relative path to node labels",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default="",
        help="Optional node features file",
    )
    parser.add_argument(
        "--times-file",
        type=str,
        default="",
        help="Optional hyperedge timestamp file",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    config = StackOverflowAnswersConfig(
        simplices_file=args.simplices_file,
        label_file=args.labels_file if args.labels_file else None,
        node_features_file=args.features_file or None,
        times_file=args.times_file or None,
    )
    loader = StackOverflowAnswersLoader(args.root, config)
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
    LOGGER.info("Saved StackOverflow Answers tensors to %s", output_dir)


if __name__ == "__main__":
    main()
