"""Reporting utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def save_metrics_report(metrics: Dict[str, float], output_dir: str, filename: str = "metrics.json") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    report_path = path / filename
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return report_path
