"""
CLI Script for Evaluating Fine-Grained Vehicle Classification Models.
Computes Top-1, Top-5, macro/weighted F1, decoded Make/Model/Year attribute accuracies,
runs temperature scaling calibration, and optional robustness stress testing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.calibration import ModelWithTemperature
from src.classification.evaluate import ClassifierEvaluator
from src.classification.robustness import RobustnessAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Fine-Grained Vehicle Classifier")
    parser.add_argument("--checkpoint", type=str, default="outputs/classification/checkpoints/best_checkpoint.pt", help="Path to checkpoint .pt")
    parser.add_argument("--split", type=str, default="data/splits/validation.csv", help="Split CSV to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--output-json", type=str, default="outputs/classification/metrics/eval_results.json", help="Output JSON path")
    parser.add_argument("--output-markdown", type=str, default="outputs/classification/metrics/eval_report.md", help="Output Markdown report path")
    parser.add_argument("--calibrate", action="store_true", default=False, help="Run validation temperature scaling calibration")
    parser.add_argument("--robustness", action="store_true", default=False, help="Run environmental stress robustness benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info("Initializing classifier evaluation for checkpoint: %s...", args.checkpoint)

    evaluator = ClassifierEvaluator(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    metrics = evaluator.evaluate(
        data_source=args.split,
        batch_size=args.batch_size,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
    )

    logger.info("Evaluation Complete:")
    logger.info("  Top-1 Accuracy: %.2f%%", metrics.get("top1_accuracy", 0.0) * 100.0)
    logger.info("  Top-5 Accuracy: %.2f%%", metrics.get("top5_accuracy", 0.0) * 100.0)
    logger.info("  Macro F1-Score: %.4f", metrics.get("macro_f1", 0.0))
    attr = metrics.get("attribute_metrics", {})
    logger.info("  Make Accuracy:  %.2f%%", attr.get("make_accuracy", 0.0) * 100.0)
    logger.info("  Model Accuracy: %.2f%%", attr.get("model_accuracy", 0.0) * 100.0)
    logger.info("  Year Accuracy:  %.2f%%", attr.get("year_accuracy", 0.0) * 100.0)

    # Optional Temperature Calibration
    if args.calibrate:
        logger.info("Running Temperature Calibration on validation set...")
        calibrator = ModelWithTemperature(evaluator.model)
        loader_for_cal = getattr(evaluator, "last_loader", None)
        if loader_for_cal is not None:
            cal_stats = calibrator.calibrate(loader_for_cal, device=evaluator.device)
            logger.info("Calibration Results: %s", cal_stats)
            # Save calibration results to json
            cal_path = PROJECT_ROOT / "outputs" / "classification" / "metrics" / "calibration_report.json"
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(cal_path, "w", encoding="utf-8") as f:
                json.dump(cal_stats, f, indent=2)

    # Optional Robustness Benchmark
    if args.robustness:
        logger.info("Running Environmental Robustness Suite...")
        analyzer = RobustnessAnalyzer(checkpoint_path=args.checkpoint, device=args.device)
        rob_stats = analyzer.run_full_robustness_benchmark(split_path=args.split)
        logger.info("Robustness benchmark summary saved to outputs/classification/metrics/robustness_metrics.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
