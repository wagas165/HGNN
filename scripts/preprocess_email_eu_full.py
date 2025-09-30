"""Preprocess the email-Eu-full dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.logging import setup_logging, get_logger
from src.data.loaders.email_eu_full import EmailEuFullConfig, EmailEuFullLoader

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess email-Eu-full dataset")
    parser.add_argument("--root", type=str, default="data/raw/email-Eu-full", help="Dataset root")
    parser.add_argument("--output", type=str, default="data/processed/email-Eu-full", help="Output dir")
    return parser.parse_args()

def ensure_label_file(root: Path, filename: str) -> Path:
    """Ensure the numeric label artifact exists, materialising it from JSON if possible."""

    label_path = root / filename
    if label_path.exists():
        return label_path

    json_fallback = label_path.with_suffix(".json")
    if json_fallback.exists():
        with json_fallback.open("r", encoding="utf-8") as f:
            raw_values = json.load(f)

        labels = np.asarray(raw_values, dtype=np.int64)
        if labels.ndim != 1:
            raise ValueError(
                "email-Eu-full labels JSON must describe a 1D array of class ids"
            )

        np.save(label_path, labels)
        LOGGER.info("Materialised %s from %s", label_path.name, json_fallback.name)
        return label_path

    raise FileNotFoundError(
        "email-Eu-full-labels.npy is required for node classification tasks. "
        "Generate or place the label file alongside the raw dataset before preprocessing."
    )


def main() -> None:
    setup_logging()
    args = parse_args()

    root_path = Path(args.root)
    label_path = ensure_label_file(root_path, "email-Eu-full-labels.npy")

    config = EmailEuFullConfig(
        vertices_file="email-Eu-full-nverts.txt",
        simplices_file="email-Eu-full-simplices.txt",
        times_file="email-Eu-full-times.txt",
        label_file="email-Eu-full-labels.npy",
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
