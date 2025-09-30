"""Preprocess the email-Eu-full dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.common.logging import setup_logging, get_logger
from src.data.loaders.email_eu_full import EmailEuFullConfig, EmailEuFullLoader

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess email-Eu-full dataset")
    parser.add_argument("--root", type=str, default="data/raw/email-Eu-full", help="Dataset root")
    parser.add_argument("--output", type=str, default="data/processed/email-Eu-full", help="Output dir")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    config = EmailEuFullConfig(
        vertices_file="email-Eu-full-nverts.txt",
        simplices_file="email-Eu-full-simplices.txt",
        times_file="email-Eu-full-times.txt",
    )
    loader = EmailEuFullLoader(args.root, config)
    data = loader.load()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "incidence": data.incidence,
        "edge_weights": data.edge_weights,
        "node_features": data.node_features,
        "labels": data.labels,
        "timestamps": data.timestamps,
        "metadata": data.metadata,
    }, output_dir / "data.pt")

    (output_dir / "metadata.json").write_text(json.dumps(data.metadata, indent=2), encoding="utf-8")
    LOGGER.info("Preprocessing complete. Saved artifacts to %s", output_dir)


if __name__ == "__main__":
    main()
