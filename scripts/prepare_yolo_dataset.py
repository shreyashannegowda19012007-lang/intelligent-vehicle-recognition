"""
YOLO Dataset preparation pipeline for Stanford Cars (Cars196).
Converts Stanford Cars annotations into normalized YOLO format:
  0 <x_center> <y_center> <width> <height>
Creates symlinked image structures, label text files, split manifests, and dataset.yaml.
Strictly avoids physical image file duplication to save disk space.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.box_utils import box_to_yolo
from src.utils.config import find_project_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_yolo_dataset")


def prepare_yolo_split(
    df: pd.DataFrame,
    split_name: str,
    detection_dir: Path,
    project_root: Path,
    use_symlinks: bool = True,
) -> tuple[int, int, list[str]]:
    """
    Generate YOLO labels, symlink images, and create image list manifest for a split.

    Args:
        df: Split DataFrame.
        split_name: 'train', 'val', or 'test'.
        detection_dir: Root detection output directory.
        project_root: Workspace root directory.
        use_symlinks: If True, symlinks images into images/<split>/.

    Returns:
        tuple of (processed_count, valid_labels_count, list of image paths)
    """
    labels_dir = detection_dir / "labels" / split_name
    images_dir = detection_dir / "images" / split_name
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    image_paths_list: list[str] = []
    valid_labels_count = 0

    x1_col = "bbox_x1" if "bbox_x1" in df.columns else "bbox_xmin"
    y1_col = "bbox_y1" if "bbox_y1" in df.columns else "bbox_ymin"
    x2_col = "bbox_x2" if "bbox_x2" in df.columns else "bbox_xmax"
    y2_col = "bbox_y2" if "bbox_y2" in df.columns else "bbox_ymax"

    for _, row in df.iterrows():
        img_rel = str(row["image_path"]).replace("\\", "/")
        full_img_p = project_root / Path(img_rel)

        if not full_img_p.is_file():
            logger.warning("Image not found: %s", full_img_p)
            continue

        stem = full_img_p.stem
        # Ensure unique stem if collision possible, but train and test have distinct folders
        # Include prefix if desired or use image_id stem
        item_id = str(row.get("image_id", stem))
        txt_name = f"{item_id}.txt"
        img_symlink_name = f"{item_id}{full_img_p.suffix}"

        # 1. Compute YOLO normalized bounding box
        orig_w = int(row["image_width"])
        orig_h = int(row["image_height"])
        x1 = float(row[x1_col])
        y1 = float(row[y1_col])
        x2 = float(row[x2_col])
        y2 = float(row[y2_col])

        xc, yc, w, h = box_to_yolo(x1, y1, x2, y2, orig_w, orig_h, inclusive=True)

        # Validate bounds
        if 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
            valid_labels_count += 1
        else:
            logger.error("Out-of-bounds YOLO coordinates for %s: xc=%f, yc=%f, w=%f, h=%f", item_id, xc, yc, w, h)

        # 2. Write YOLO label file: class_id xc yc w h
        label_file = labels_dir / txt_name
        with open(label_file, "w", encoding="utf-8") as f:
            f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # 3. Create image symlink (or record path)
        symlink_dest = images_dir / img_symlink_name
        if use_symlinks and not symlink_dest.exists():
            try:
                os.symlink(full_img_p, symlink_dest)
            except (OSError, NotImplementedError) as e:
                logger.debug("Symlink creation skipped: %s", e)

        # Relative path from project root for manifest
        rel_img_manifest = str(symlink_dest.relative_to(project_root) if symlink_dest.exists() else full_img_p.relative_to(project_root)).replace("\\", "/")
        image_paths_list.append(rel_img_manifest)

    # 4. Write manifest text file
    manifest_txt = detection_dir / f"{split_name}.txt"
    with open(manifest_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(image_paths_list) + "\n")

    return len(df), valid_labels_count, image_paths_list


def prepare_yolo_dataset(
    splits_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    use_symlinks: bool = True,
) -> dict[str, Any]:
    """
    Convert Stanford Cars splits into complete YOLO dataset structure.
    """
    start_time = time.time()
    root = find_project_root()
    splits_p = Path(splits_dir) if splits_dir else (root / "data" / "splits")
    det_p = Path(output_dir) if output_dir else (root / "data" / "processed" / "detection")
    det_p.mkdir(parents=True, exist_ok=True)

    train_csv = splits_p / "train.csv"
    val_csv = splits_p / "validation.csv"
    test_csv = splits_p / "test.csv"

    if not train_csv.is_file() or not val_csv.is_file() or not test_csv.is_file():
        raise FileNotFoundError(f"Splits not found in {splits_p}. Run scripts/prepare_splits.py first.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    logger.info("Processing train split (%d items)...", len(train_df))
    t_tot, t_val, _ = prepare_yolo_split(train_df, "train", det_p, root, use_symlinks)

    logger.info("Processing validation split (%d items)...", len(val_df))
    v_tot, v_val, _ = prepare_yolo_split(val_df, "val", det_p, root, use_symlinks)

    logger.info("Processing test split (%d items)...", len(test_df))
    te_tot, te_val, _ = prepare_yolo_split(test_df, "test", det_p, root, use_symlinks)

    # 5. Generate dataset.yaml
    rel_det_path = str(det_p.relative_to(root)).replace("\\", "/")
    yaml_payload = {
        "path": rel_det_path,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "vehicle"
        },
        "nc": 1,
    }

    yaml_file = det_p / "dataset.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(yaml_payload, f, sort_keys=False)
    logger.info("Generated YOLO configuration file: %s", yaml_file)

    elapsed = round(time.time() - start_time, 2)
    summary: dict[str, Any] = {
        "status": "SUCCESS",
        "duration_seconds": elapsed,
        "detection_dir": str(det_p.relative_to(root) if det_p.is_relative_to(root) else det_p).replace("\\", "/"),
        "yaml_file": str(yaml_file.relative_to(root) if yaml_file.is_relative_to(root) else yaml_file).replace("\\", "/"),
        "splits": {
            "train": {"total": t_tot, "valid_labels": t_val},
            "val": {"total": v_tot, "valid_labels": v_val},
            "test": {"total": te_tot, "valid_labels": te_val},
        },
        "total_labels_generated": t_tot + v_tot + te_tot,
        "all_labels_valid": (t_tot == t_val and v_tot == v_val and te_tot == te_val),
    }

    summary_file = root / "data" / "reports" / "yolo_preparation_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved YOLO preparation summary to %s", summary_file)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare YOLO format detection dataset")
    parser.add_argument("--splits", type=str, default=None, help="Splits directory")
    parser.add_argument("--output", type=str, default=None, help="Detection output directory")
    parser.add_argument("--no-symlinks", action="store_true", help="Disable symlink generation")
    args = parser.parse_args()

    summary = prepare_yolo_dataset(
        splits_dir=args.splits,
        output_dir=args.output,
        use_symlinks=not args.no_symlinks,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_labels_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
