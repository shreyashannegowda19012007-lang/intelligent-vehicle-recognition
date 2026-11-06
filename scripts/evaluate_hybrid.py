"""
CLI Tool for Evaluating Hybrid Intelligence Vehicle Recognition System.
Executes the Ablation Study (Experiments A-E), Robustness Stress Tests,
Failure Case Attribution, and exports comprehensive metrics to outputs/hybrid/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid.ablation import AblationStudyRunner
from src.hybrid.config import HybridConfig
from src.hybrid.evaluator import HybridEvaluator
from src.hybrid.pipeline import HybridVehiclePipeline
from src.hybrid.robustness import HybridRobustnessEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Vehicle Recognition Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to hybrid.yaml config")
    parser.add_argument("--output-dir", type=str, default="outputs/hybrid", help="Directory for output metrics and reports")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of validation samples to evaluate")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip running the 5 ablation experiments")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip robustness perturbation benchmarks")
    return parser.parse_args()


def load_test_samples(num_samples: int = 20) -> list[dict]:
    """Load evaluation samples using manifest or validation CSV."""
    val_csv = PROJECT_ROOT / "data" / "splits" / "validation.csv"
    manifest_csv = PROJECT_ROOT / "data" / "processed" / "metadata" / "vehicle_manifest.csv"

    csv_path = val_csv if val_csv.is_file() else manifest_csv
    if not csv_path.is_file():
        logger.warning("No split CSV found; generating synthetic evaluation samples.")
        return generate_synthetic_samples(num_samples)

    df = pd.read_csv(csv_path)
    # Check if images exist on disk
    valid_records = []
    for _, row in df.iterrows():
        p = PROJECT_ROOT / str(row["image_path"])
        if p.is_file():
            valid_records.append({
                "image": p,
                "true_class_id": int(row.get("class_id", 0)),
                "bbox": (
                    int(row.get("bbox_x1", 0)),
                    int(row.get("bbox_y1", 0)),
                    int(row.get("bbox_x2", 100)),
                    int(row.get("bbox_y2", 100)),
                ),
            })
            if len(valid_records) >= num_samples:
                break

    if not valid_records:
        logger.info("Raw Stanford Cars image files not present on disk; generating representative synthetic vehicle samples.")
        return generate_synthetic_samples(num_samples)

    return valid_records


def generate_synthetic_samples(num_samples: int = 20) -> list[dict]:
    """Generate deterministic synthetic vehicle images for evaluation and tests."""
    samples = []
    rng = np.random.default_rng(42)

    # If sample car image exists in Stanford Dataset/cars_train/cars_train/00001.jpg, include it
    cand_sample = PROJECT_ROOT / "Stanford Dataset" / "cars_train" / "cars_train" / "00001.jpg"
    if cand_sample.is_file():
        samples.append({
            "image": cand_sample,
            "true_class_id": 0,
            "bbox": (22, 229, 805, 750),
        })

    for i in range(len(samples), num_samples):
        img_arr = np.zeros((480, 640, 3), dtype=np.uint8)
        # Background noise
        img_arr[:] = rng.integers(120, 160, (480, 640, 3), dtype=np.uint8)

        # Vehicle body
        bx1, by1, bx2, by2 = 80 + (i % 5) * 10, 100 + (i % 3) * 10, 560 - (i % 5) * 10, 380
        img_arr[by1:by2, bx1:bx2, :] = (40, 80, 180)  # Car body blue/orange
        img_arr[by1 + 20 : by1 + 100, bx1 + 60 : bx2 - 60, :] = (200, 220, 240)  # Windshield / cabin
        # Wheels
        img_arr[by2 - 40 : by2, bx1 + 30 : bx1 + 90, :] = (20, 20, 20)
        img_arr[by2 - 40 : by2, bx2 - 90 : bx2 - 30, :] = (20, 20, 20)

        cid = i % 196
        samples.append({
            "image": img_arr,
            "true_class_id": cid,
            "bbox": (bx1, by1, bx2, by2),
        })

    return samples


def main() -> int:
    args = parse_args()
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = HybridConfig.from_yaml(args.config)
    cfg.outputs_dir = str(out_dir)
    cfg.failure_cases_dir = str(out_dir / "failure_cases")

    pipeline = HybridVehiclePipeline(config=cfg)
    evaluator = HybridEvaluator(pipeline=pipeline, config=cfg)
    ablation_runner = AblationStudyRunner(pipeline=pipeline, config=cfg)
    robustness_evaluator = HybridRobustnessEvaluator(pipeline=pipeline, config=cfg)

    logger.info("Loading %d evaluation samples...", args.num_samples)
    samples = load_test_samples(args.num_samples)
    logger.info("Loaded %d evaluation samples.", len(samples))

    # 1. Standard Comparison
    logger.info("Evaluating on vehicle crops across decision strategies...")
    crop_eval = evaluator.evaluate_on_crops(samples)
    with open(out_dir / "crop_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(crop_eval, f, indent=2)

    # 2. End-to-End Pipeline Evaluation
    logger.info("Evaluating complete end-to-end detection and classification...")
    e2e_eval = evaluator.evaluate_pipeline_end_to_end(samples)
    with open(out_dir / "end_to_end_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(e2e_eval, f, indent=2)

    # 3. Ablation Study (Experiments A - E)
    if not args.skip_ablation:
        logger.info("Running Ablation Study (Experiments A through E)...")
        df_ablation = ablation_runner.run_experiments(samples, output_dir=out_dir)
        print("\n" + "=" * 80)
        print("ABLATION STUDY RESULTS:")
        print("=" * 80)
        print(df_ablation.to_string(index=False))
        print("=" * 80 + "\n")

    # 4. Robustness & Failure Analysis
    if not args.skip_robustness:
        logger.info("Running Robustness Stress Tests...")
        robustness_metrics = robustness_evaluator.evaluate_corruptions(samples, output_dir=out_dir)

        logger.info("Cataloging Representative End-to-End Failure Cases...")
        failure_cases = robustness_evaluator.generate_and_save_failure_cases(
            samples, output_dir=out_dir / "failure_cases", max_cases=4
        )
        print(f"Captured and saved {len(failure_cases)} systematic failure cases to {out_dir / 'failure_cases'}")

    logger.info("Evaluation complete! All artifacts saved to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
