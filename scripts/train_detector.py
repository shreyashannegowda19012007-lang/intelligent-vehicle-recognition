"""
CLI Script to train YOLO Vehicle Detector on Stanford Cars dataset.
Supports CLI overrides for hyperparameters and hardware configuration.
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

from src.detection.trainer import DetectionTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_detector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 Vehicle Detector")
    parser.add_argument("--config", type=str, default="configs/detection.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (<= 16 recommended for RTX 3050)")
    parser.add_argument("--imgsz", type=int, default=None, help="Input image resolution")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu', '0', 'auto')")
    parser.add_argument("--workers", type=int, default=None, help="Data loader worker count")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Pretrained checkpoint path/name")
    parser.add_argument("--name", type=str, default=None, help="Experiment run name")
    parser.add_argument("--freeze", type=int, default=None, help="Backbone layers to freeze")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info("Initializing DetectionTrainer with config: %s", args.config)

    overrides = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch_size"] = args.batch
    if args.imgsz is not None:
        overrides["imgsz"] = args.imgsz
    if args.device is not None:
        overrides["device"] = args.device
    if args.workers is not None:
        overrides["workers"] = args.workers
    if args.lr0 is not None:
        overrides["lr0"] = args.lr0
    if args.checkpoint is not None:
        overrides["pretrained_checkpoint"] = args.checkpoint
    if args.name is not None:
        overrides["name"] = args.name
    if args.freeze is not None:
        overrides["freeze"] = args.freeze

    trainer = DetectionTrainer(config_path=args.config, **overrides)
    summary = trainer.train(resume=args.resume)

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
