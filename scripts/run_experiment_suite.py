#!/usr/bin/env python3
"""Run a suite of DF-HGNN experiments and generate comparison plots."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
DEFAULT_METRICS = ("test_accuracy", "test_macro_f1", "test_roc_auc")
DEFAULT_CPU_WORKERS = 120
MIN_REQUIRED_RUNS = 10


@dataclass(frozen=True)
class Experiment:
    """Metadata describing a single configuration to execute."""

    key: str
    config: Path
    label: str
    groups: Sequence[str]

    @property
    def results_root(self) -> Path:
        return PROJECT_ROOT / "results" / self.config.stem


EXPERIMENTS: Dict[str, Experiment] = {
    "baseline": Experiment(
        key="baseline",
        config=CONFIG_ROOT / "experiment" / "baseline.yaml",
        label="DF-HGNN (Ours)",
        groups=("comparison", "ablation"),
    ),
    "baseline_no_deterministic": Experiment(
        key="baseline_no_deterministic",
        config=CONFIG_ROOT / "experiment" / "baseline_no_deterministic.yaml",
        label="DF-HGNN w/o Deterministic",
        groups=("ablation",),
    ),
    "allset": Experiment(
        key="allset",
        config=CONFIG_ROOT / "experiment" / "allset_transformer.yaml",
        label="AllSet Transformer",
        groups=("comparison",),
    ),
    "hypergcn": Experiment(
        key="hypergcn",
        config=CONFIG_ROOT / "experiment" / "hypergcn.yaml",
        label="HyperGCN",
        groups=("comparison",),
    ),
    "unignn": Experiment(
        key="unignn",
        config=CONFIG_ROOT / "experiment" / "unignn.yaml",
        label="UniGNN",
        groups=("comparison",),
    ),
}


def _default_seeds(num_runs: int) -> List[int]:
    if num_runs < MIN_REQUIRED_RUNS:
        raise ValueError(
            f"At least {MIN_REQUIRED_RUNS} runs are required for boxplot generation; "
            f"received num_runs={num_runs}."
        )
    return list(range(num_runs))


def _build_train_command(exp: Experiment, seed: int, run_id: str) -> List[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_df_hgnn.py"),
        "--config",
        str(exp.config),
        "--seed",
        str(seed),
        "--run-id",
        run_id,
    ]


def _should_skip(exp: Experiment, run_id: str, skip_existing: bool) -> bool:
    if not skip_existing:
        return False
    metrics_path = exp.results_root / run_id / "reports" / "metrics.json"
    return metrics_path.exists()


def _run_experiment(
    exp: Experiment,
    seed: int,
    skip_existing: bool,
    device_id: Optional[str],
    print_lock: threading.Lock,
) -> None:
    run_id = f"seed{seed}"
    if _should_skip(exp, run_id, skip_existing):
        with print_lock:
            print(f"[skip] {exp.key} (seed={seed}) already has metrics.json")
        return

    cmd = _build_train_command(exp, seed=seed, run_id=run_id)
    env = os.environ.copy()
    if device_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    with print_lock:
        device_msg = f" gpu={device_id}" if device_id is not None else ""
        print(f"[run]{device_msg} {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def _determine_available_gpus(requested: Optional[Sequence[int]]) -> List[str]:
    if requested is not None:
        return [str(device) for device in requested]

    env_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_devices:
        return [token.strip() for token in env_devices.split(",") if token.strip()]

    try:
        import torch

        if torch.cuda.is_available():
            return [str(idx) for idx in range(torch.cuda.device_count())]
    except Exception:
        return []

    return []


def _partition_tasks(tasks: Sequence[Tuple[Experiment, int]], partitions: int) -> List[List[Tuple[Experiment, int]]]:
    buckets: List[List[Tuple[Experiment, int]]] = [[] for _ in range(partitions)]
    for index, task in enumerate(tasks):
        buckets[index % partitions].append(task)
    return buckets


def _execute_tasks(
    tasks: Sequence[Tuple[Experiment, int]],
    gpu_devices: Sequence[str],
    cpu_workers: int,
    skip_existing: bool,
) -> None:
    if not tasks:
        return

    print_lock = threading.Lock()
    if gpu_devices:
        worker_count = min(len(tasks), len(gpu_devices), cpu_workers)
        if worker_count == 0:
            return
        partitions = _partition_tasks(tasks, worker_count)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for device, partition in zip(gpu_devices[:worker_count], partitions):
                futures.append(
                    executor.submit(
                        _run_partition,
                        partition,
                        skip_existing,
                        device,
                        print_lock,
                    )
                )
            for future in futures:
                future.result()
    else:
        worker_count = min(len(tasks), cpu_workers)
        if worker_count == 0:
            return
        partitions = _partition_tasks(tasks, worker_count)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = []
            for partition in partitions:
                futures.append(
                    executor.submit(
                        _run_partition,
                        partition,
                        skip_existing,
                        None,
                        print_lock,
                    )
                )
            for future in futures:
                future.result()


def _run_partition(
    partition: Sequence[Tuple[Experiment, int]],
    skip_existing: bool,
    device_id: Optional[str],
    print_lock: threading.Lock,
) -> None:
    for exp, seed in partition:
        _run_experiment(exp, seed, skip_existing=skip_existing, device_id=device_id, print_lock=print_lock)


def _collect_group_inputs(group: str) -> List[str]:
    tokens: List[str] = []
    for exp in EXPERIMENTS.values():
        if group not in exp.groups:
            continue
        tokens.append(f"{exp.label}={exp.results_root}")
    return tokens


def _run_analysis(group: str, output_dir: Path, metrics: Sequence[str], box_metric: str) -> None:
    inputs = _collect_group_inputs(group)
    if not inputs:
        print(f"[warn] No experiments registered for group '{group}'")
        return

    if all(not Path(token.split("=", 1)[1]).exists() for token in inputs):
        print(f"[warn] Results directories missing for group '{group}', skipping analysis")
        return

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "analyze_results.py"),
        "--inputs",
        *inputs,
        "--metrics-keys",
        *metrics,
        "--boxplot-metric",
        box_metric,
        "--output-dir",
        str(output_dir),
    ]
    print(f"[analyze] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline and ablation experiment suites")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Explicit list of seeds to evaluate",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=MIN_REQUIRED_RUNS,
        help=(
            "Number of seeds to generate when --seeds is not provided. "
            f"Must be at least {MIN_REQUIRED_RUNS} to satisfy boxplot requirements."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose metrics.json already exists",
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "analysis"),
        help="Base directory for generated analysis artefacts",
    )
    parser.add_argument(
        "--metrics-keys",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric keys passed to analyze_results.py",
    )
    parser.add_argument(
        "--boxplot-metric",
        type=str,
        default="test_accuracy",
        help="Metric used for boxplot visualization",
    )
    parser.add_argument(
        "--max-cpu-workers",
        type=int,
        default=DEFAULT_CPU_WORKERS,
        help="Maximum number of concurrent CPU workers for launching experiments",
    )
    parser.add_argument(
        "--gpu-devices",
        nargs="+",
        type=int,
        help=(
            "Explicit list of GPU device indices to cycle through. "
            "Defaults to detected devices or CPU-only execution if none are available."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else _default_seeds(args.num_runs)
    if not seeds:
        raise ValueError("At least one seed must be specified")

    if len(seeds) < MIN_REQUIRED_RUNS:
        raise ValueError(
            f"At least {MIN_REQUIRED_RUNS} seeds are required to generate boxplots. "
            f"Received {len(seeds)}"
        )

    tasks: List[Tuple[Experiment, int]] = []
    for seed in seeds:
        for exp in EXPERIMENTS.values():
            tasks.append((exp, seed))

    gpu_devices = _determine_available_gpus(args.gpu_devices)
    _execute_tasks(tasks, gpu_devices, args.max_cpu_workers, args.skip_existing)

    analysis_root = Path(args.analysis_dir).expanduser().resolve()
    comparison_dir = analysis_root / "method_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    _run_analysis("comparison", comparison_dir, metrics=args.metrics_keys, box_metric=args.boxplot_metric)

    ablation_dir = analysis_root / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    _run_analysis("ablation", ablation_dir, metrics=args.metrics_keys, box_metric=args.boxplot_metric)

    print("\nGenerated analysis:")
    for path in (comparison_dir, ablation_dir):
        if path.exists():
            print(f" - {path}")


if __name__ == "__main__":
    main()

