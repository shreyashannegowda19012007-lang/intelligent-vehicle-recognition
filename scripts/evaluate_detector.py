"""
CLI Script to evaluate YOLO Vehicle Detector.
Computes mAP, Precision, Recall, F1, IoU distributions, and latency metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.detection.evaluator import DetectionEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_detector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO Vehicle Detector")
    parser.add_argument("--model", type=str, default=None, help="Path to YOLO checkpoint (default: models/detection/best.pt or yolov8n.pt)")
    parser.add_argument("--data", type=str, default="data/processed/detection/dataset.yaml", help="Path to dataset YAML")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker threads")
    parser.add_argument("--device", type=str, default="auto", help="Evaluation device")
    parser.add_argument("--gt-eval", action="store_true", help="Run direct GT IoU matching from manifest")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples for GT matching")
    parser.add_argument("--robustness", action="store_true", help="Run environmental robustness benchmarks")
    parser.add_argument("--optimization", action="store_true", help="Run FP16, resolution, and batch optimization benchmarks")
    parser.add_argument("--output", type=str, default="outputs/detection/metrics/eval_results.json", help="Output JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_arg = args.model if args.model is not None else "models/detection/best.pt"
    if not (PROJECT_ROOT / model_arg).is_file() and not Path(model_arg).is_file():
        model_arg = "yolov8n.pt"

    evaluator = DetectionEvaluator(
        model_path=model_arg,
        data_yaml=args.data,
        device=args.device,
    )

    out_data = {}

    # 1. Ultralytics mAP validation
    try:
        val_metrics = evaluator.evaluate_ultralytics(
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
        )
        out_data["ultralytics_validation"] = val_metrics
    except Exception as e:  # noqa: BLE001
        logger.error("Ultralytics validation encountered an issue: %s", e)
        out_data["ultralytics_validation"] = {"error": str(e)}

    # 2. Manifest / Split GT evaluation if requested
    split_name = "validation" if args.split == "val" else "test"
    split_csv = PROJECT_ROOT / "data" / "splits" / f"{split_name}.csv"
    manifest_csv = PROJECT_ROOT / "data" / "processed" / "metadata" / "vehicle_manifest.csv"

    split_df = pd.DataFrame()
    if split_csv.is_file():
        split_df = pd.read_csv(split_csv)
    elif manifest_csv.is_file():
        df = pd.read_csv(manifest_csv)
        split_df = df[df["split"] == args.split] if "split" in df.columns and args.split in df["split"].values else df

    if args.gt_eval and len(split_df) > 0:
        logger.info("Evaluating ground truth matching on %d samples from %s...", min(args.max_samples, len(split_df)), args.split)
        gt_metrics = evaluator.evaluate_ground_truth_matching(split_df, max_samples=args.max_samples)
        out_data["ground_truth_matching"] = gt_metrics

    # 3. Robustness perturbation evaluation if requested
    if args.robustness and len(split_df) > 0:
        logger.info("Running environmental robustness perturbation benchmarks...")
        sample_imgs = split_df["image_path"].head(10).tolist()
        robustness_summary = evaluator.evaluate_robustness(sample_imgs)
        out_data["robustness_diagnostics"] = robustness_summary

        rob_out = PROJECT_ROOT / "outputs" / "detection" / "metrics" / "robustness_metrics.json"
        rob_out.parent.mkdir(parents=True, exist_ok=True)
        with open(rob_out, "w", encoding="utf-8") as rf:
            json.dump(robustness_summary, rf, indent=2)
        logger.info("Saved robustness metrics to %s", rob_out)

    # 4. Optimization benchmarks if requested
    if args.optimization and len(split_df) > 0:
        logger.info("Running inference optimization benchmarks (FP32/FP16, resolutions, batching)...")
        sample_imgs = split_df["image_path"].head(16).tolist()
        opt_summary = evaluator.evaluate_optimization(sample_imgs)
        out_data["optimization_diagnostics"] = opt_summary

        opt_out = PROJECT_ROOT / "outputs" / "detection" / "metrics" / "optimization_benchmark.json"
        opt_out.parent.mkdir(parents=True, exist_ok=True)
        with open(opt_out, "w", encoding="utf-8") as of:
            json.dump(opt_summary, of, indent=2)
        logger.info("Saved optimization benchmark to %s", opt_out)

    # Save output JSON
    out_file = PROJECT_ROOT / args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    logger.info("Evaluation results saved to %s", out_file)
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(out_data, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
