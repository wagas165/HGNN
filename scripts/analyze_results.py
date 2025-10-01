#!/usr/bin/env python3
"""Utilities for aggregating HGNN experiment metrics into comparison plots."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_METRIC_KEYS = ("test_accuracy", "test_macro_f1", "test_roc_auc")


@dataclass
class ExperimentMetrics:
    """Container for metrics collected from a single experiment label."""

    label: str
    runs: List[Dict[str, float]]


def _parse_input(token: str) -> Tuple[str, Path]:
    if "=" in token:
        label, path = token.split("=", 1)
        return label.strip(), Path(path).expanduser().resolve()

    path = Path(token).expanduser().resolve()
    if path.is_file():
        parent = path.parent.parent if path.name == "metrics.json" else path.parent
        label = parent.name if parent.name else path.stem
    else:
        label = path.name or "experiment"
    return label, path


def _load_metrics_from_path(path: Path) -> List[Dict[str, float]]:
    if path.is_file():
        with path.open("r", encoding="utf-8") as fp:
            return [json.load(fp)]

    metrics: List[Dict[str, float]] = []
    direct_path = path / "reports" / "metrics.json"
    candidates = []
    if direct_path.exists():
        candidates.append(direct_path)
    candidates.extend(
        candidate
        for candidate in sorted(path.glob("**/reports/metrics.json"))
        if candidate != direct_path
    )

    for candidate in candidates:
        with candidate.open("r", encoding="utf-8") as fp:
            metrics.append(json.load(fp))

    if not metrics:
        raise FileNotFoundError(f"No metrics.json found under {path}")

    return metrics


def _collect_experiments(tokens: Iterable[str]) -> List[ExperimentMetrics]:
    experiments: List[ExperimentMetrics] = []
    for token in tokens:
        label, path = _parse_input(token)
        runs = _load_metrics_from_path(path)
        experiments.append(ExperimentMetrics(label=label, runs=runs))
    return experiments


def _prepare_long_frame(
    experiments: List[ExperimentMetrics], metric_keys: Tuple[str, ...]
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for exp in experiments:
        for run_idx, metrics in enumerate(exp.runs, start=1):
            for metric_key in metric_keys:
                if metric_key not in metrics:
                    continue
                records.append(
                    {
                        "Method": exp.label,
                        "Run": run_idx,
                        "Metric": metric_key,
                        "Score": float(metrics[metric_key]),
                    }
                )
    if not records:
        return pd.DataFrame(columns=["Method", "Run", "Metric", "Score"])

    df = pd.DataFrame(records)
    df["Metric"] = pd.Categorical(df["Metric"], categories=list(metric_keys), ordered=True)
    return df


def _configure_style() -> None:
    sns.set_theme(context="talk", style="whitegrid", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 6),
            "savefig.dpi": 300,
        }
    )


def _save_method_bars(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return

    summary = df.groupby(["Method", "Metric"], as_index=False)["Score"].mean()
    for method, subset in summary.groupby("Method"):
        plt.figure()
        ax = sns.barplot(data=subset, x="Metric", y="Score", color="#4c72b0")
        ax.set_ylim(0, 1)
        ax.set_title(f"{method} performance")
        ax.set_ylabel("Score")
        ax.set_xlabel("Metric")
        plt.tight_layout()
        filename_slug = method.lower().replace(" ", "_")
        plt.savefig(output_dir / f"{filename_slug}_metrics_bar.png")
        plt.close()

    plt.figure()
    ax = sns.barplot(data=summary, x="Metric", y="Score", hue="Method")
    ax.set_ylim(0, 1)
    ax.set_title("Method comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.legend(title="Method", loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "methods_comparison_bar.png")
    plt.close()


def _save_boxplot(df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    subset = df[df["Metric"] == metric]
    if subset.empty:
        return

    counts = subset.groupby("Method")["Score"].count()
    if (counts < 2).all():
        return

    plt.figure()
    ax = sns.boxplot(data=subset, x="Method", y="Score", width=0.5)
    ax.set_title(f"{metric} distribution")
    ax.set_ylabel(metric)
    ax.set_xlabel("Method")
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_boxplot.png")
    plt.close()


def _save_summary_table(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return

    summary = (
        df.groupby(["Method", "Metric"])["Score"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comparison plots from metrics.json files")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="List of result directories or metrics files. Use label=path to override labels.",
    )
    parser.add_argument(
        "--metrics-keys",
        nargs="+",
        default=list(DEFAULT_METRIC_KEYS),
        help="Metric keys to extract from metrics.json",
    )
    parser.add_argument(
        "--boxplot-metric",
        type=str,
        default="test_accuracy",
        help="Metric key to use for the optional boxplot",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis",
        help="Directory where plots will be written",
    )
    args = parser.parse_args()

    experiments = _collect_experiments(args.inputs)
    metric_keys = tuple(args.metrics_keys)
    df = _prepare_long_frame(experiments, metric_keys)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_style()
    _save_method_bars(df, output_dir)
    _save_boxplot(df, args.boxplot_metric, output_dir)
    _save_summary_table(df, output_dir)


if __name__ == "__main__":
    main()
