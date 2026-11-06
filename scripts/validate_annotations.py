"""
Annotation validation script for Stanford Cars (Cars196).
Verifies completeness of normalized annotations, bounding box geometry,
taxonomy consistency, coordinate bounds, and split integrity.
Outputs data/reports/annotation_validation_report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.annotation_parser import ALL_49_MAKES
from src.utils.config import find_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_annotations")


def validate_annotations(
    manifest_path: str | Path | None = None,
    output_report_path: str | Path | None = None,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """
    Perform deep validation of annotations, bounding boxes, and taxonomy.

    Args:
        manifest_path: Path to vehicle_manifest.csv or normalized_annotations.csv.
        output_report_path: Path to save the output report.
        sample_size: If specified, validates file existence/dimensions on subsample.

    Returns:
        Validation report dictionary.
    """
    start_time = time.time()
    root = find_project_root()

    if manifest_path is None:
        manifest_path = root / "data" / "processed" / "metadata" / "vehicle_manifest.csv"
        if not manifest_path.is_file():
            manifest_path = root / "data" / "processed" / "annotations" / "normalized_annotations.csv"

    p = Path(manifest_path)
    logger.info("Validating annotations from: %s", p)
    df = pd.read_csv(p)

    # Ensure standardized column names
    col_map = {
        "bbox_xmin": "bbox_x1",
        "bbox_ymin": "bbox_y1",
        "bbox_xmax": "bbox_x2",
        "bbox_ymax": "bbox_y2",
    }
    df = df.rename(columns=col_map)

    total_records = len(df)
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    df_train = df[train_mask]
    df_test = df[test_mask]

    # 1. Check splits
    train_count = len(df_train)
    test_count = len(df_test)

    # 2. Check labels
    train_missing_class = int(df_train["class_id"].isna().sum())
    train_missing_make = int(df_train["make"].isna().sum())
    train_missing_year = int(df_train["year"].isna().sum())

    test_unlabeled_class = int(df_test["class_id"].isna().sum())

    # Check class ID range in train
    invalid_class_ids: list[int] = []
    if train_missing_class == 0:
        min_class = int(df_train["class_id"].min())
        max_class = int(df_train["class_id"].max())
        num_unique_classes = int(df_train["class_id"].nunique())
        out_of_bounds = df_train[(df_train["class_id"] < 0) | (df_train["class_id"] >= 196)]
        invalid_class_ids = out_of_bounds["class_id"].tolist()
    else:
        min_class, max_class, num_unique_classes = None, None, 0

    # 3. Check Bounding Boxes
    # Condition: 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
    invalid_bboxes: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        x1 = row["bbox_x1"]
        y1 = row["bbox_y1"]
        x2 = row["bbox_x2"]
        y2 = row["bbox_y2"]
        w = row["image_width"]
        h = row["image_height"]

        errors = []
        if x1 < 0 or y1 < 0:
            errors.append("negative_coord")
        if x2 <= x1:
            errors.append("x2_not_greater_than_x1")
        if y2 <= y1:
            errors.append("y2_not_greater_than_y1")
        if x2 > w:
            errors.append(f"x2_{x2}_exceeds_width_{w}")
        if y2 > h:
            errors.append(f"y2_{y2}_exceeds_height_{h}")

        if errors:
            invalid_bboxes.append({
                "image_id": str(row.get("image_id", idx)),
                "bbox": [x1, y1, x2, y2],
                "dims": [w, h],
                "errors": errors,
            })

    # 4. Check Taxonomy Consistency
    all_makes_set = set(ALL_49_MAKES)
    unknown_makes: list[str] = []
    if "make" in df_train.columns:
        present_makes = set(df_train["make"].dropna().unique())
        unknown_makes = sorted(present_makes - all_makes_set)

    # Check years range
    invalid_years: list[int] = []
    if "year" in df_train.columns:
        train_years = df_train["year"].dropna().astype(int)
        invalid_years = train_years[(train_years < 1990) | (train_years > 2025)].tolist()

    # 5. Image existence and dimension verification
    check_rows = df if sample_size is None or sample_size >= len(df) else df.sample(sample_size, random_state=42)
    missing_files: list[str] = []
    dimension_mismatches: list[dict[str, Any]] = []

    for _, row in check_rows.iterrows():
        raw_path = Path(str(row["image_path"]))
        full_path = raw_path if raw_path.is_absolute() else (root / raw_path)

        if not full_path.is_file():
            missing_files.append(str(raw_path))
            continue

        # Optional dimension verification on sample
        try:
            with Image.open(full_path) as img:
                actual_w, actual_h = img.size
                if actual_w != int(row["image_width"]) or actual_h != int(row["image_height"]):
                    dimension_mismatches.append({
                        "file": str(raw_path),
                        "expected": [int(row["image_width"]), int(row["image_height"])],
                        "actual": [actual_w, actual_h],
                    })
        except Exception as e:  # noqa: BLE001
            missing_files.append(f"{raw_path} (read error: {e})")

    elapsed = round(time.time() - start_time, 2)
    is_valid = bool(
        len(invalid_bboxes) == 0
        and len(invalid_class_ids) == 0
        and train_missing_class == 0
        and len(missing_files) == 0
        and len(unknown_makes) == 0
    )

    report: dict[str, Any] = {
        "status": "VALID" if is_valid else "INVALID",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": elapsed,
        "manifest_path": str(p.relative_to(root) if p.is_relative_to(root) else p).replace("\\", "/"),
        "total_records": total_records,
        "splits": {
            "train_count": train_count,
            "test_count": test_count,
        },
        "labels": {
            "train_labeled_count": train_count - train_missing_class,
            "train_missing_class": train_missing_class,
            "train_missing_make": train_missing_make,
            "train_missing_year": train_missing_year,
            "test_unlabeled_count": test_unlabeled_class,
            "unique_classes_in_train": num_unique_classes,
            "class_id_range": [min_class, max_class],
            "invalid_class_ids": invalid_class_ids,
        },
        "bounding_boxes": {
            "all_valid": len(invalid_bboxes) == 0,
            "invalid_bbox_count": len(invalid_bboxes),
            "invalid_bboxes": invalid_bboxes[:20],
        },
        "taxonomy": {
            "unique_makes": len(all_makes_set),
            "unknown_makes": unknown_makes,
            "invalid_years_count": len(invalid_years),
            "invalid_years": invalid_years,
        },
        "files_and_dimensions": {
            "checked_files": len(check_rows),
            "missing_files_count": len(missing_files),
            "missing_files": missing_files[:20],
            "dimension_mismatch_count": len(dimension_mismatches),
            "dimension_mismatches": dimension_mismatches[:20],
        },
    }

    report_p = Path(output_report_path) if output_report_path else (root / "data" / "reports" / "annotation_validation_report.json")
    report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved annotation validation report to %s", report_p)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stanford Cars annotations")
    parser.add_argument("--manifest", type=str, default=None, help="Manifest CSV path")
    parser.add_argument("--output", type=str, default=None, help="Output report path")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for file check")
    args = parser.parse_args()

    report = validate_annotations(
        manifest_path=args.manifest,
        output_report_path=args.output,
        sample_size=args.sample,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "VALID" else 1


if __name__ == "__main__":
    sys.exit(main())
