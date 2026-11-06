"""
CLI Script for Classical Computer Vision Visualizations.
Usage:
    python scripts/visualize_classical.py --sample-index 0
    python scripts/visualize_classical.py --image-path "Stanford Dataset/cars_train/cars_train/00001.jpg"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.classical_cv.visualizer import ClassicalVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("visualize_classical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate classical CV visual diagnostics")
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to specific image file",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Row index from train split to visualize",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to visualize from split",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated PNG figures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    visualizer = ClassicalVisualizer(output_dir=args.output_dir, project_root=PROJECT_ROOT)

    if args.image_path:
        img_p = Path(args.image_path)
        logger.info("Generating visualizations for single image: %s", img_p)
        saved = visualizer.generate_all_visualizations_for_sample(
            image_input=img_p,
            sample_name=img_p.stem,
            save_individual=True,
        )
        print(f"Generated {len(saved)} visualizations:")
        for name, path in saved.items():
            print(f"  - {name}: {path}")
        return 0

    train_csv = PROJECT_ROOT / "data" / "splits" / "train.csv"
    if not train_csv.is_file():
        logger.error("Train split not found at: %s", train_csv)
        return 1

    df = pd.read_csv(train_csv)
    start_idx = max(0, min(args.sample_index, len(df) - 1))
    end_idx = min(start_idx + args.num_samples, len(df))

    for idx in range(start_idx, end_idx):
        row = df.iloc[idx]
        sample_name = f"sample_{idx}_{row.get('image_id', 'car')}"
        logger.info("Processing sample %d: %s (%s)", idx, row.get("image_id", ""), row.get("class_name", ""))
        saved = visualizer.generate_all_visualizations_for_sample(
            image_input=str(row["image_path"]),
            bbox=row,
            sample_name=sample_name,
            save_individual=True,
        )
        print(f"\nSample {idx} ({row.get('class_name', '')}):")
        for name, path in saved.items():
            print(f"  - {name}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
