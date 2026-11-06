"""
System Evaluation CLI Script for CV Hackathon P5.
Executes complete evaluation suite: metrics calculation, 4-way ablation study,
robustness testing, failure analysis, and artifact generation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import SystemEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_system")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation system for CV Hackathon P5 vehicle recognition"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Directory to save evaluation artifacts (default: results/evaluation)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of samples for live verification benchmark",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Initializing SystemEvaluator...")
    evaluator = SystemEvaluator(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
    )

    try:
        metrics = evaluator.run_full_evaluation()
        logger.info("Evaluation completed successfully!")
        logger.info("Saved deliverables under %s:", evaluator.output_dir)
        for item in evaluator.output_dir.iterdir():
            logger.info("  - %s (%d bytes)", item.name, item.stat().st_size)
        return 0
    except Exception as e:
        logger.exception("Evaluation encountered an unexpected error: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())

