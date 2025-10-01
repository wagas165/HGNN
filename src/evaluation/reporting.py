"""Reporting utilities for DF-HGNN experiments."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


PredictionPlotCache = Dict[str, Dict[str, torch.Tensor]]


def save_metrics_report(
    metrics: Dict[str, float],
    output_dir: str,
    filename: str = "metrics.json",
    prediction_cache: Optional[PredictionPlotCache] = None,
) -> Path:
    """Persist metric values and optional diagnostic plots.

    Parameters
    ----------
    metrics:
        Mapping of metric names to values produced by the trainer.
    output_dir:
        Destination folder for reports and generated plots.
    filename:
        Filename for the JSON summary of scalar metrics.
    prediction_cache:
        Optional mapping of split names to cached probability/label tensors.
    """

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    report_path = path / filename
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if prediction_cache:
        _configure_plot_style()
        _save_metrics_bar_plot(metrics, path / "metrics_bar.png")
        for split_name, tensors in prediction_cache.items():
            probs = tensors["probs"].detach().cpu()
            labels = tensors["labels"].detach().cpu()
            _save_roc_curve(path, split_name, probs, labels)
            _save_precision_recall_curve(path, split_name, probs, labels)
            _save_confusion_matrix(path, split_name, probs, labels)

    return report_path


def _configure_plot_style() -> None:
    sns.set_theme(context="talk", style="whitegrid", palette="colorblind")
    plt.rcParams.update(
        {
            "figure.figsize": (7, 5),
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def _save_metrics_bar_plot(metrics: Dict[str, float], output_path: Path) -> None:
    df = _metrics_to_dataframe(metrics)
    if df is None or df.empty:
        return

    plt.figure()
    ax = sns.barplot(data=df, x="metric", y="value", hue="split")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Evaluation summary")
    ax.set_ylim(0, 1)
    ax.legend(title="Split", loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _metrics_to_dataframe(metrics: Dict[str, float]) -> Optional[pd.DataFrame]:
    records: List[Dict[str, object]] = []
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        parts = key.split("_")
        if len(parts) < 2:
            split = "overall"
            metric_name = key
        elif parts[0] == "transfer" and len(parts) >= 3:
            split = "_".join(parts[:2])
            metric_name = "_".join(parts[2:])
        else:
            split = parts[0]
            metric_name = "_".join(parts[1:])
        records.append({"split": split, "metric": metric_name, "value": float(value)})

    if not records:
        return None

    df = pd.DataFrame(records)
    metric_order = sorted(df["metric"].unique())
    df["metric"] = pd.Categorical(df["metric"], categories=metric_order, ordered=True)
    split_order = sorted(df["split"].unique())
    df["split"] = pd.Categorical(df["split"], categories=split_order, ordered=True)
    return df.sort_values(["metric", "split"]).reset_index(drop=True)


def _safe_split_name(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_").lower()


def _save_roc_curve(
    output_dir: Path,
    split_name: str,
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    y_true = labels.numpy()
    y_score = probs.numpy()
    num_classes = y_score.shape[1] if y_score.ndim > 1 else 1

    fig, ax = plt.subplots()
    if np.unique(y_true).size < 2 and num_classes <= 2:
        ax.text(
            0.5,
            0.5,
            "Single class detected",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(output_dir / f"roc_{_safe_split_name(split_name)}.png")
        plt.close(fig)
        return

    if num_classes <= 2:
        positive_scores = y_score[:, -1]
        fpr, tpr, _ = roc_curve(y_true, positive_scores)
        auc_score = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", linewidth=2.0)
    else:
        classes = np.arange(num_classes)
        y_true_bin = label_binarize(y_true, classes=classes)
        fpr_dict = {}
        tpr_dict = {}
        plotted_classes: List[int] = []
        for class_idx in classes:
            try:
                fpr_vals, tpr_vals, _ = roc_curve(
                    y_true_bin[:, class_idx], y_score[:, class_idx]
                )
            except ValueError:
                continue
            fpr_dict[class_idx] = fpr_vals
            tpr_dict[class_idx] = tpr_vals
            auc_score = np.trapz(tpr_vals, fpr_vals)
            ax.plot(
                fpr_vals,
                tpr_vals,
                label=f"Class {class_idx} (AUC={auc_score:.3f})",
                linewidth=1.5,
            )
            plotted_classes.append(class_idx)

        if not plotted_classes:
            ax.text(
                0.5,
                0.5,
                "Insufficient class diversity",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(output_dir / f"roc_{_safe_split_name(split_name)}.png")
            plt.close(fig)
            return

        all_fpr = np.unique(np.concatenate([fpr_dict[idx] for idx in plotted_classes]))
        mean_tpr = np.zeros_like(all_fpr)
        for class_idx in plotted_classes:
            mean_tpr += np.interp(all_fpr, fpr_dict[class_idx], tpr_dict[class_idx])
        mean_tpr /= len(plotted_classes)
        macro_auc = np.trapz(mean_tpr, all_fpr)
        ax.plot(
            all_fpr,
            mean_tpr,
            label=f"Macro-average (AUC={macro_auc:.3f})",
            linewidth=2.5,
            linestyle="--",
            color="black",
        )

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {split_name}")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / f"roc_{_safe_split_name(split_name)}.png")
    plt.close(fig)


def _save_precision_recall_curve(
    output_dir: Path,
    split_name: str,
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    y_true = labels.numpy()
    y_score = probs.numpy()
    num_classes = y_score.shape[1] if y_score.ndim > 1 else 1

    fig, ax = plt.subplots()
    if np.unique(y_true).size < 2 and num_classes <= 2:
        ax.text(
            0.5,
            0.5,
            "Single class detected",
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(output_dir / f"pr_{_safe_split_name(split_name)}.png")
        plt.close(fig)
        return

    if num_classes <= 2:
        positive_scores = y_score[:, -1]
        precision, recall, _ = precision_recall_curve(y_true, positive_scores)
        ap_score = average_precision_score(y_true, positive_scores)
        ax.plot(recall, precision, label=f"AP = {ap_score:.3f}", linewidth=2.0)
    else:
        classes = np.arange(num_classes)
        y_true_bin = label_binarize(y_true, classes=classes)
        precision_dict = {}
        recall_dict = {}
        plotted_classes: List[int] = []
        for class_idx in classes:
            if y_true_bin[:, class_idx].sum() == 0:
                continue
            precision_vals, recall_vals, _ = precision_recall_curve(
                y_true_bin[:, class_idx], y_score[:, class_idx]
            )
            precision_dict[class_idx] = precision_vals
            recall_dict[class_idx] = recall_vals
            ap_score = average_precision_score(y_true_bin[:, class_idx], y_score[:, class_idx])
            ax.plot(
                recall_vals,
                precision_vals,
                label=f"Class {class_idx} (AP={ap_score:.3f})",
                linewidth=1.5,
            )
            plotted_classes.append(class_idx)

        if plotted_classes:
            precision_micro, recall_micro, _ = precision_recall_curve(
                y_true_bin[:, plotted_classes].ravel(), y_score[:, plotted_classes].ravel()
            )
            ap_micro = average_precision_score(
                y_true_bin[:, plotted_classes], y_score[:, plotted_classes], average="micro"
            )
            ax.plot(
                recall_micro,
                precision_micro,
                label=f"Micro-average (AP={ap_micro:.3f})",
                linewidth=2.5,
                linestyle="--",
                color="black",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient class diversity",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            plt.tight_layout()
            plt.savefig(output_dir / f"pr_{_safe_split_name(split_name)}.png")
            plt.close(fig)
            return

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {split_name}")
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(output_dir / f"pr_{_safe_split_name(split_name)}.png")
    plt.close(fig)


def _save_confusion_matrix(
    output_dir: Path,
    split_name: str,
    probs: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    y_true = labels.numpy()
    y_score = probs.numpy()
    preds = y_score.argmax(axis=1)
    cm = confusion_matrix(y_true, preds)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix — {split_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{_safe_split_name(split_name)}.png")
    plt.close(fig)
