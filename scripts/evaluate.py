"""Aggregate evaluation metrics across experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

from src.common.logging import setup_logging, get_logger

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metrics")
    parser.add_argument("--inputs", nargs="+", help="Metrics JSON files")
    parser.add_argument("--output", type=str, default="outputs/reports/summary.json")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    metrics_list: List[Dict[str, float]] = []
    for path in args.inputs:
        metrics_list.append(json.loads(Path(path).read_text(encoding="utf-8")))

    summary: Dict[str, Dict[str, float]] = {}
    for key in metrics_list[0].keys():
        values = [metrics[key] for metrics in metrics_list if key in metrics]
        summary[key] = {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Saved summary to %s", output_path)


if __name__ == "__main__":
    main()
