#!/usr/bin/env python3
"""Unified entry point for running DF-HGNN experiments."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _stream_subprocess(command: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process, log_path.open("a", encoding="utf-8") as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DF-HGNN training with convenience helpers")
    parser.add_argument("--config", required=True, help="Path to the Hydra configuration file")
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate comparison plots with scripts/analyze_results.py after training",
    )
    parser.add_argument(
        "--compare",
        nargs="*",
        default=[],
        help="Additional experiment directories or metrics paths to include in post-training analysis",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config_name = config_path.stem or "experiment"
    results_dir = Path("results") / config_name
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / "train.log"
    print(f"Running experiment {config_name} with configuration {config_path}")
    command = [sys.executable, "scripts/train_df_hgnn.py", "--config", str(config_path)]
    _stream_subprocess(command, log_path)

    if args.save_plots:
        analysis_inputs = [f"{config_name}={results_dir}"] + args.compare
        analysis_cmd = [
            sys.executable,
            "scripts/analyze_results.py",
            "--inputs",
        ] + analysis_inputs + ["--output-dir", str(results_dir / "reports")]
        print("Generating comparison plots ...")
        subprocess.run(analysis_cmd, check=True)


if __name__ == "__main__":
    main()
