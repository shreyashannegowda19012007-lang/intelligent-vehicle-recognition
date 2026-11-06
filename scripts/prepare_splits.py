"""
Reproducible dataset splitting pipeline for Stanford Cars (Cars196).
Partitions official training data (8,144 images) into stratified Train (85%) and
Validation (15%) splits, while strictly preserving the official holdout Test split (8,041 images).
Outputs data/splits/train.csv, validation.csv, test.csv, and split_summary.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import load_manifest_df
from src.utils.config import find_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_splits")


def create_splits(
    manifest_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Generate stratified train/validation/test splits from the unified manifest.

    Args:
        manifest_path: Path to vehicle_manifest.csv or vehicle_manifest.json.
        output_dir: Destination directory (defaults to data/splits/).
        val_ratio: Fraction of labeled training data for validation (default: 0.15).
        random_seed: Fixed random seed for complete reproducibility (default: 42).

    Returns:
        Summary dictionary with split counts, class coverage, and distribution stats.
    """
    root = find_project_root()
    splits_dir = Path(output_dir) if output_dir else (root / "data" / "splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    df_manifest = load_manifest_df(manifest_path)
    logger.info("Loaded manifest with %d records", len(df_manifest))

    # Separate official train (labeled) from official test (competition holdout)
    df_official_train = df_manifest[df_manifest["split"] == "train"].copy()
    df_official_test = df_manifest[df_manifest["split"] == "test"].copy()

    logger.info("Official train set: %d images", len(df_official_train))
    logger.info("Official test set: %d images", len(df_official_test))

    if len(df_official_train) == 0:
        raise ValueError("Manifest contains 0 records marked with split='train'")

    # Ensure class_id is non-null integer for stratification
    df_official_train["class_id"] = df_official_train["class_id"].astype(int)

    # Perform stratified split
    train_df, val_df = train_test_split(
        df_official_train,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=df_official_train["class_id"],
        shuffle=True,
    )

    # Update the split column to reflect 'train' vs 'val'
    train_df = train_df.copy()
    train_df["split"] = "train"

    val_df = val_df.copy()
    val_df["split"] = "val"

    test_df = df_official_test.copy()
    test_df["split"] = "test"

    # Save split CSVs
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "validation.csv"
    test_csv = splits_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    logger.info("Saved train split: %s (%d records)", train_csv, len(train_df))
    logger.info("Saved val split: %s (%d records)", val_csv, len(val_df))
    logger.info("Saved test split: %s (%d records)", test_csv, len(test_df))

    # Compute distribution statistics
    train_class_counts = train_df["class_id"].value_counts().to_dict()
    val_class_counts = val_df["class_id"].value_counts().to_dict()

    train_unique_classes = int(train_df["class_id"].nunique())
    val_unique_classes = int(val_df["class_id"].nunique())

    summary: dict[str, Any] = {
        "status": "SUCCESS",
        "random_seed": random_seed,
        "validation_ratio": val_ratio,
        "counts": {
            "total": len(df_manifest),
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "percentages": {
            "train_of_official_train": round(len(train_df) / len(df_official_train) * 100, 2),
            "val_of_official_train": round(len(val_df) / len(df_official_train) * 100, 2),
            "test_of_total": round(len(test_df) / len(df_manifest) * 100, 2),
        },
        "class_coverage": {
            "num_total_classes": 196,
            "train_unique_classes": train_unique_classes,
            "val_unique_classes": val_unique_classes,
            "all_classes_in_train": train_unique_classes == 196,
            "all_classes_in_val": val_unique_classes == 196,
        },
        "per_class_samples": {
            "train_min": int(min(train_class_counts.values())),
            "train_max": int(max(train_class_counts.values())),
            "train_mean": round(float(pd.Series(train_class_counts).mean()), 2),
            "val_min": int(min(val_class_counts.values())),
            "val_max": int(max(val_class_counts.values())),
            "val_mean": round(float(pd.Series(val_class_counts).mean()), 2),
        },
        "files": {
            "train_csv": str(train_csv.relative_to(root) if train_csv.is_relative_to(root) else train_csv).replace("\\", "/"),
            "validation_csv": str(val_csv.relative_to(root) if val_csv.is_relative_to(root) else val_csv).replace("\\", "/"),
            "test_csv": str(test_csv.relative_to(root) if test_csv.is_relative_to(root) else test_csv).replace("\\", "/"),
        },
    }

    summary_file = splits_dir / "split_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved split summary to %s", summary_file)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare stratified splits for Stanford Cars")
    parser.add_argument("--manifest", type=str, default=None, help="Manifest CSV/JSON path")
    parser.add_argument("--output", type=str, default=None, help="Output splits directory")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    summary = create_splits(
        manifest_path=args.manifest,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
