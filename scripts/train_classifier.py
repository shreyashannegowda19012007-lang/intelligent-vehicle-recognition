"""
CLI Script for Training Fine-Grained Vehicle Classification Models.
Supports ResNet50, EfficientNet-B0/B2, ConvNeXt-Tiny, ResNet18, state resumption,
early stopping, mixed precision, and structured logging.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification.config import ClassificationConfig
from src.classification.trainer import ClassifierTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fine-Grained Vehicle Classifier")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet18", "efficientnet_b0", "efficientnet_b2", "convnext_tiny"], help="Backbone architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"], help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "plateau", "none"], help="LR scheduler")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True, help="Enable AMP mixed precision on GPU")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--freeze-backbone", action="store_true", default=False, help="Freeze backbone weights during initial training")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional maximum training batches per epoch (for fast debugging/CPU runs)")
    parser.add_argument("--max-val-batches", type=int, default=None, help="Optional maximum validation batches per epoch")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info("Initializing vehicle classification training pipeline...")

    config = ClassificationConfig(
        architecture=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        device=args.device,
        mixed_precision=args.mixed_precision,
        freeze_backbone=args.freeze_backbone,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    trainer = ClassifierTrainer(config=config)

    if args.resume:
        trainer.resume_from(args.resume)

    summary = trainer.train()
    logger.info(
        "Training finished successfully! Best Val Top-1: %.2f%% at epoch %d. Best checkpoint: %s",
        summary["best_val_top1"] * 100.0,
        summary["best_epoch"],
        summary["best_checkpoint_path"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
