"""
Evaluation Visualization Module: Confusion Matrices and Performance Plots.
Generates publication-quality confusion matrix heatmaps at both the fine-grained
class level and the aggregated vehicle manufacturer (Make) level.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for CI and headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.classification.metrics import ClassMappingDecoder
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


def plot_make_confusion_matrix(
    y_true_makes: Sequence[str],
    y_pred_makes: Sequence[str],
    output_path: str | Path,
    title: str = "Vehicle Make Confusion Matrix",
    top_n_makes: int = 25,
) -> Path:
    """
    Generate and save normalized vehicle Make confusion matrix heatmap.

    Args:
        y_true_makes: Sequence of ground-truth make strings.
        y_pred_makes: Sequence of predicted make strings.
        output_path: Target image file path (.png).
        title: Plot title.
        top_n_makes: Limit to top N most frequent makes for visual clarity.

    Returns:
        Path to saved image.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # Determine unique makes sorted by frequency in ground-truth
    s_true = pd.Series(y_true_makes)
    top_makes = s_true.value_counts().index[:top_n_makes].tolist()

    # Filter to top makes or map others to 'Other'
    filtered_true = [m if m in top_makes else "Other" for m in y_true_makes]
    filtered_pred = [m if m in top_makes else "Other" for m in y_pred_makes]
    labels = top_makes + (["Other"] if "Other" in filtered_true or "Other" in filtered_pred else [])

    cm = confusion_matrix(filtered_true, filtered_pred, labels=labels)
    # Row normalize to percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    plt.figure(figsize=(14, 11))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Normalized Accuracy (%)"},
        linewidths=0.5,
    )
    plt.title(title, fontsize=15, pad=15, fontweight="bold")
    plt.xlabel("Predicted Make", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Make", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(out_p, dpi=300)
    plt.close()
    logger.info("Saved make confusion matrix plot to %s", out_p)
    return out_p


def plot_class_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_path: str | Path,
    class_decoder: ClassMappingDecoder | None = None,
    title: str = "Fine-Grained Class Confusion Matrix (Top Classes)",
    top_n_classes: int = 20,
) -> Path:
    """
    Generate and save normalized class-level confusion matrix heatmap.

    Args:
        y_true: Ground-truth class IDs (0-indexed).
        y_pred: Predicted class IDs (0-indexed).
        output_path: Target image file path (.png).
        class_decoder: Optional taxonomy decoder to map IDs to readable class names.
        title: Plot title.
        top_n_classes: Limit to top N most active/confused classes for readability.

    Returns:
        Path to saved image.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    decoder = class_decoder or ClassMappingDecoder()

    # Determine top active classes
    s_true = pd.Series(y_true)
    top_ids = s_true.value_counts().index[:top_n_classes].tolist()

    labels_id = top_ids
    labels_names = [decoder.get_info(cid)["class_name"] for cid in labels_id]
    # Shorten class names if too long
    labels_short = [n[:24] + "..." if len(n) > 24 else n for n in labels_names]

    cm = confusion_matrix(y_true, y_pred, labels=labels_id)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    plt.figure(figsize=(14, 11))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="mako",
        xticklabels=labels_short,
        yticklabels=labels_short,
        cbar_kws={"label": "Normalized Recall (%)"},
        linewidths=0.5,
    )
    plt.title(title, fontsize=15, pad=15, fontweight="bold")
    plt.xlabel("Predicted Vehicle Class", fontsize=12, labelpad=10)
    plt.ylabel("Ground Truth Vehicle Class", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    plt.savefig(out_p, dpi=300)
    plt.close()
    logger.info("Saved class confusion matrix plot to %s", out_p)
    return out_p

