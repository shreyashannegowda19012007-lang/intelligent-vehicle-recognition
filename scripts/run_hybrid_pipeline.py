"""
CLI Tool for Running Hybrid Vehicle Recognition Pipeline.
Usage:
    python scripts/run_hybrid_pipeline.py --image "Stanford Dataset/cars_train/cars_train/00001.jpg"
    python scripts/run_hybrid_pipeline.py --image test_car.jpg --strategy agreement_based --output-json output.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hybrid.config import DecisionStrategy, HybridConfig
from src.hybrid.pipeline import HybridVehiclePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_hybrid_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hybrid Intelligence Vehicle Recognition Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file or directory")
    parser.add_argument(
        "--strategy",
        type=str,
        default="deep_learning_primary",
        choices=["deep_learning_primary", "agreement_based", "score_fusion"],
        help="Hybrid decision strategy",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to hybrid.yaml config")
    parser.add_argument("--conf", type=float, default=None, help="YOLO detection confidence threshold override")
    parser.add_argument("--iou", type=float, default=None, help="YOLO IoU threshold override")
    parser.add_argument("--device", type=str, default=None, help="Inference device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--output-json", type=str, default=None, help="Path to write JSON response")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load configuration
    cfg = HybridConfig.from_yaml(args.config)
    if args.strategy:
        cfg.decision_strategy = DecisionStrategy(args.strategy)
    if args.device:
        cfg.detector_device = args.device
        cfg.dl_device = args.device

    # Initialize central pipeline
    pipeline = HybridVehiclePipeline(config=cfg)

    input_path = Path(args.image)
    if not input_path.is_file() and (PROJECT_ROOT / input_path).is_file():
        input_path = PROJECT_ROOT / input_path

    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    else:
        logger.error("Input path does not exist: %s", input_path)
        return 1

    all_outputs: list[dict] = []
    for img_p in image_files:
        logger.info("Processing %s...", img_p.name)
        result = pipeline.process_image(
            img_p,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            decision_strategy=cfg.decision_strategy,
        )
        all_outputs.append(result)

    final_result = all_outputs[0] if len(all_outputs) == 1 else all_outputs

    if args.output_json:
        out_p = Path(args.output_json)
        if not out_p.is_absolute():
            out_p = PROJECT_ROOT / out_p
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2)
        logger.info("Saved result to %s", out_p)
    else:
        print(json.dumps(final_result, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
