"""
Class imbalance analysis engine for Stanford Cars (Cars196).
Analyzes distribution of 196 vehicle classes, 49 manufacturers, and production years.
Identifies rare and dominant categories, evaluates split coverage, and computes
class weights for Cross-Entropy loss and WeightedRandomSampler.
Outputs data/reports/class_imbalance_report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import find_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analyze_dataset")


def compute_class_weights(
    class_counts: dict[int, int],
    num_classes: int = 196,
    method: str = "balanced",
    beta: float = 0.999,
) -> dict[int, float]:
    """
    Calculate class weighting factors.

    Methods:
        - 'balanced': n_samples / (n_classes * count_c) [Standard sklearn formula]
        - 'inverse': 1.0 / count_c
        - 'effective_samples': (1 - beta) / (1 - beta^count_c) [Cui et al. CVPR 2019]

    Args:
        class_counts: Map from class_id to integer sample count.
        num_classes: Expected total classes (196).
        method: Weight calculation strategy.
        beta: Hyperparameter for effective samples (default: 0.999).

    Returns:
        Dict mapping class_id to float weight.
    """
    total_samples = sum(class_counts.values())
    weights: dict[int, float] = {}

    for c in range(num_classes):
        cnt = class_counts.get(c, 0)
        if cnt == 0:
            weights[c] = 0.0
            continue

        if method == "balanced":
            w = total_samples / (float(num_classes) * float(cnt))
        elif method == "inverse":
            w = 1.0 / float(cnt)
        elif method == "effective_samples":
            effective_num = 1.0 - np.power(beta, cnt)
            w = (1.0 - beta) / max(1e-8, effective_num)
        else:
            raise ValueError(f"Unknown weight method: {method}")

        weights[c] = float(w)

    # Normalize weights so mean weight equals 1.0
    valid_weights = [w for w in weights.values() if w > 0]
    if valid_weights:
        mean_w = sum(valid_weights) / len(valid_weights)
        weights = {c: round(w / mean_w, 4) for c, w in weights.items()}

    return weights


def build_weighted_sampler(
    dataset_targets: list[int] | np.ndarray | torch.Tensor,
    class_weights: dict[int, float],
) -> WeightedRandomSampler:
    """
    Construct a PyTorch WeightedRandomSampler from sample target labels and class weights.
    """
    if isinstance(dataset_targets, torch.Tensor):
        targets = dataset_targets.cpu().numpy()
    else:
        targets = np.array(dataset_targets)

    sample_weights = np.array([class_weights.get(int(t), 1.0) for t in targets], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def analyze_dataset_imbalance(
    splits_dir: str | Path | None = None,
    output_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Perform deep statistical analysis of class and metadata distributions across dataset splits.
    """
    start_time = time.time()
    root = find_project_root()
    splits_p = Path(splits_dir) if splits_dir else (root / "data" / "splits")
    reports_dir = root / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_csv = splits_p / "train.csv"
    val_csv = splits_p / "validation.csv"
    test_csv = splits_p / "test.csv"

    if not train_csv.is_file():
        raise FileNotFoundError(f"Train split not found at: {train_csv}")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv) if val_csv.is_file() else pd.DataFrame()
    df_test = pd.read_csv(test_csv) if test_csv.is_file() else pd.DataFrame()

    total_classes = 196
    train_counts_series = df_train["class_id"].value_counts().sort_index()
    train_counts: dict[int, int] = {int(k): int(v) for k, v in train_counts_series.items()}

    # Class statistics
    min_count = int(train_counts_series.min())
    max_count = int(train_counts_series.max())
    mean_count = float(round(train_counts_series.mean(), 2))
    median_count = float(round(train_counts_series.median(), 2))
    std_count = float(round(train_counts_series.std(), 2))
    imbalance_ratio = round(max_count / max(1, min_count), 2)

    # Threshold for rare / dominant classes
    p10_threshold = float(train_counts_series.quantile(0.10))
    p90_threshold = float(train_counts_series.quantile(0.90))

    rare_classes: list[dict[str, Any]] = []
    dominant_classes: list[dict[str, Any]] = []

    for cid, cnt in train_counts.items():
        sample_rows = df_train[df_train["class_id"] == cid]
        cname = sample_rows.iloc[0]["class_name"] if "class_name" in sample_rows.columns else f"Class_{cid}"
        cmake = sample_rows.iloc[0]["make"] if "make" in sample_rows.columns else "Unknown"

        if cnt <= p10_threshold:
            rare_classes.append({
                "class_id": cid,
                "class_name": cname,
                "make": cmake,
                "sample_count": cnt,
            })
        elif cnt >= p90_threshold:
            dominant_classes.append({
                "class_id": cid,
                "class_name": cname,
                "make": cmake,
                "sample_count": cnt,
            })

    # Manufacturer distribution
    make_counts = df_train["make"].value_counts().to_dict() if "make" in df_train.columns else {}
    make_stats = {
        "num_makes": len(make_counts),
        "top_5_makes": dict(list(make_counts.items())[:5]),
        "bottom_5_makes": dict(list(make_counts.items())[-5:]),
    }

    # Year distribution
    year_counts = (
        df_train["year"].dropna().astype(int).value_counts().sort_index().to_dict()
        if "year" in df_train.columns
        else {}
    )

    # Class weights
    balanced_weights = compute_class_weights(train_counts, total_classes, method="balanced")
    effective_weights = compute_class_weights(train_counts, total_classes, method="effective_samples")

    # Split coverage
    val_classes = set(df_val["class_id"].dropna().astype(int).unique()) if len(df_val) > 0 else set()
    train_classes = set(train_counts.keys())
    missing_in_train = sorted(set(range(total_classes)) - train_classes)
    missing_in_val = sorted(set(range(total_classes)) - val_classes)

    elapsed = round(time.time() - start_time, 2)

    report: dict[str, Any] = {
        "status": "SUCCESS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": elapsed,
        "dataset_summary": {
            "total_classes": total_classes,
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
        },
        "class_distribution_metrics": {
            "min_samples_per_class": min_count,
            "max_samples_per_class": max_count,
            "mean_samples_per_class": mean_count,
            "median_samples_per_class": median_count,
            "std_dev_samples": std_count,
            "imbalance_ratio": imbalance_ratio,
        },
        "rare_classes": {
            "threshold_10th_percentile": p10_threshold,
            "count": len(rare_classes),
            "classes": rare_classes[:10],  # first 10
        },
        "dominant_classes": {
            "threshold_90th_percentile": p90_threshold,
            "count": len(dominant_classes),
            "classes": dominant_classes[:10],
        },
        "manufacturer_distribution": make_stats,
        "year_distribution": year_counts,
        "coverage": {
            "train_classes_count": len(train_classes),
            "val_classes_count": len(val_classes),
            "missing_in_train": missing_in_train,
            "missing_in_val": missing_in_val,
            "all_196_classes_in_train": len(missing_in_train) == 0,
            "all_196_classes_in_val": len(missing_in_val) == 0,
        },
        "class_weights": {
            "method_balanced_sample": {str(k): balanced_weights[k] for k in list(balanced_weights.keys())[:5]},
            "method_effective_samples": {str(k): effective_weights[k] for k in list(effective_weights.keys())[:5]},
        },
    }

    report_p = Path(output_report_path) if output_report_path else (reports_dir / "class_imbalance_report.json")
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved class imbalance report to %s", report_p)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze class imbalance in Stanford Cars splits")
    parser.add_argument("--splits", type=str, default=None, help="Splits directory")
    parser.add_argument("--output", type=str, default=None, help="Output report path")
    args = parser.parse_args()

    report = analyze_dataset_imbalance(splits_dir=args.splits, output_report_path=args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
