"""
Deep Learning Training Pipeline for Fine-Grained Vehicle Classification.
Implements complete training and validation loops with AdamW optimizer, cosine/step LR scheduling,
best checkpoint tracking, early stopping, state resumption, mixed precision, and structured logging.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.classification.config import ClassificationConfig
from src.classification.models.backbones import build_classification_model
from src.classification.transforms import (
    get_training_transforms,
    get_validation_transforms,
)
from src.data.classification_dataset import StanfordCarsClassificationDataset
from src.utils.config import find_project_root
from src.utils.device import get_device

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ClassifierTrainer:
    """
    Modular training manager for fine-grained vehicle classification.
    """

    def __init__(
        self,
        config: ClassificationConfig | None = None,
        model: nn.Module | None = None,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
    ) -> None:
        self.root = find_project_root()
        self.config = config or ClassificationConfig()
        set_seed(self.config.seed)

        # 1. Resolve Device
        if self.config.device == "auto":
            self.device = get_device("auto")
        else:
            self.device = torch.device(self.config.device)

        self.mixed_precision = self.config.mixed_precision and (self.device.type == "cuda")
        logger.info("Initializing ClassifierTrainer on %s (mixed_precision=%s)...", self.device, self.mixed_precision)

        # 2. Build or assign model
        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = build_classification_model(
                architecture=self.config.architecture,
                num_classes=self.config.num_classes,
                pretrained=self.config.pretrained,
                dropout=self.config.dropout,
                freeze_backbone=self.config.freeze_backbone,
            ).to(self.device)

        # 3. Setup DataLoaders if not provided
        self.train_loader = train_loader or self._create_dataloader(
            split_rel_path=self.config.train_split,
            is_train=True,
            batch_size=self.config.batch_size,
        )
        self.val_loader = val_loader or self._create_dataloader(
            split_rel_path=self.config.val_split,
            is_train=False,
            batch_size=self.config.batch_size,
        )

        # 4. Loss Criterion
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        ).to(self.device)

        # 5. Optimizer
        self.optimizer = self._build_optimizer()

        # 6. Learning Rate Scheduler
        self.scheduler = self._build_scheduler()

        # 7. Scaler for mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)

        # 8. Tracking state
        self.start_epoch = 1
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history: list[dict[str, Any]] = []

        # Ensure output directories exist
        (self.root / self.config.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        (self.root / self.config.metrics_dir).mkdir(parents=True, exist_ok=True)

    def _create_dataloader(
        self, split_rel_path: str, is_train: bool, batch_size: int
    ) -> DataLoader:
        """Create DataLoader for specified CSV split."""
        split_path = self.root / split_rel_path
        if not split_path.is_file():
            raise FileNotFoundError(f"Dataset split not found at: {split_path}")

        if is_train:
            transform = get_training_transforms(
                image_size=self.config.image_size,
                mean=self.config.mean,
                std=self.config.std,
                random_crop_scale=self.config.random_crop_scale,
                horizontal_flip_p=self.config.horizontal_flip_p,
                rotation_degrees=self.config.rotation_degrees,
                color_jitter_brightness=self.config.color_jitter_brightness,
                color_jitter_contrast=self.config.color_jitter_contrast,
                color_jitter_saturation=self.config.color_jitter_saturation,
                color_jitter_hue=self.config.color_jitter_hue,
            )
            shuffle = True
        else:
            transform = get_validation_transforms(
                image_size=self.config.image_size,
                mean=self.config.mean,
                std=self.config.std,
            )
            shuffle = False

        dataset = StanfordCarsClassificationDataset(
            source=split_path,
            project_root=self.root,
            transform=transform,
            crop_bbox=self.config.crop_bbox,
            image_size=self.config.image_size,
        )

        pin_mem = (self.device.type == "cuda")
        num_workers = self.config.num_workers

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=is_train and (len(dataset) > batch_size),
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Configure AdamW, Adam, or SGD optimizer."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt_name = self.config.optimizer.lower()

        if opt_name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif opt_name == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif opt_name == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _build_scheduler(self) -> Any:
        """Configure learning rate scheduler."""
        sched_name = self.config.scheduler.lower()
        if sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.lr_min,
            )
        elif sched_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.step_size,
                gamma=self.config.gamma,
            )
        elif sched_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=self.config.gamma,
                patience=2,
            )
        elif sched_name == "none":
            return None
        raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")

    def train_epoch(self, epoch: int) -> tuple[float, float, float]:
        """
        Execute one training epoch.
        Returns (train_loss, train_top1_acc, train_top5_acc).
        """
        self.model.train()
        running_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total_samples = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            if self.config.max_train_batches and batch_idx >= self.config.max_train_batches:
                break
            if isinstance(batch, dict):
                images = batch["image"].to(self.device, non_blocking=True)
                targets = batch["class_id"].to(self.device, non_blocking=True)
            else:
                images, targets = batch[0].to(self.device), batch[1].to(self.device)

            # Skip unlabeled or invalid samples if present
            valid_mask = targets >= 0
            if not valid_mask.any():
                continue
            if not valid_mask.all():
                images = images[valid_mask]
                targets = targets[valid_mask]

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            # Accuracy calculations
            with torch.no_grad():
                top1_pred = logits.argmax(dim=1)
                correct_top1 += (top1_pred == targets).sum().item()

                k = min(5, logits.size(1))
                top5_pred = logits.topk(k, dim=1)[1]
                correct_top5 += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(self.train_loader):
                cur_top1 = (correct_top1 / total_samples) * 100.0 if total_samples > 0 else 0.0
                elapsed = time.time() - t0
                logger.info(
                    "Epoch %d [%d/%d] - Loss: %.4f - Top-1: %.2f%% - Speed: %.1f img/s",
                    epoch,
                    batch_idx + 1,
                    len(self.train_loader),
                    running_loss / total_samples,
                    cur_top1,
                    total_samples / max(0.001, elapsed),
                )

        epoch_loss = running_loss / max(1, total_samples)
        epoch_top1 = correct_top1 / max(1, total_samples)
        epoch_top5 = correct_top5 / max(1, total_samples)

        return epoch_loss, epoch_top1, epoch_top5

    def validate(self) -> tuple[float, float, float]:
        """
        Execute evaluation on the validation split.
        Returns (val_loss, val_top1_acc, val_top5_acc).
        """
        self.model.eval()
        running_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if self.config.max_val_batches and batch_idx >= self.config.max_val_batches:
                    break
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device, non_blocking=True)
                    targets = batch["class_id"].to(self.device, non_blocking=True)
                else:
                    images, targets = batch[0].to(self.device), batch[1].to(self.device)

                valid_mask = targets >= 0
                if not valid_mask.any():
                    continue
                if not valid_mask.all():
                    images = images[valid_mask]
                    targets = targets[valid_mask]

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size

                top1_pred = logits.argmax(dim=1)
                correct_top1 += (top1_pred == targets).sum().item()

                k = min(5, logits.size(1))
                top5_pred = logits.topk(k, dim=1)[1]
                correct_top5 += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

        val_loss = running_loss / max(1, total_samples)
        val_top1 = correct_top1 / max(1, total_samples)
        val_top5 = correct_top5 / max(1, total_samples)

        return val_loss, val_top1, val_top5

    def save_checkpoint(
        self,
        filepath: str | Path,
        epoch: int,
        val_acc: float,
        is_best: bool = False,
    ) -> None:
        """Save training state checkpoint."""
        save_path = self.root / filepath if not Path(filepath).is_absolute() else Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": epoch,
            "architecture": self.config.architecture,
            "num_classes": self.config.num_classes,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_acc": self.best_val_acc,
            "val_acc": val_acc,
            "config": self.config.to_dict(),
            "history": self.history,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_best": is_best,
        }
        torch.save(state, save_path)
        logger.info("Saved %scheckpoint to %s", "best " if is_best else "", save_path)

    def resume_from(self, checkpoint_path: str | Path) -> None:
        """Resume training state from an existing checkpoint."""
        p = self.root / checkpoint_path if not Path(checkpoint_path).is_file() else Path(checkpoint_path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found for resumption: {p}")

        logger.info("Resuming training state from %s...", p)
        checkpoint = torch.load(p, map_location=self.device)

        self.model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_val_acc = checkpoint.get("best_val_acc", checkpoint.get("val_acc", 0.0))
        self.history = checkpoint.get("history", [])
        logger.info("Resumed from epoch %d with prior best val acc: %.4f", self.start_epoch - 1, self.best_val_acc)

    def train(self, max_epochs: int | None = None) -> dict[str, Any]:
        """
        Execute full training loop up to target epochs with early stopping.
        """
        target_epochs = max_epochs or self.config.epochs
        logger.info(
            "Starting training for %d epochs (%s, lr=%.6f, batch_size=%d)...",
            target_epochs,
            self.config.architecture,
            self.config.learning_rate,
            self.config.batch_size,
        )

        # Save initial training configuration
        cfg_out = self.root / self.config.metrics_dir / "train_config.json"
        self.config.save_json(cfg_out)

        total_train_start = time.time()

        for epoch in range(self.start_epoch, target_epochs + 1):
            epoch_t0 = time.time()
            train_loss, train_top1, train_top5 = self.train_epoch(epoch)
            val_loss, val_top1, val_top5 = self.validate()
            epoch_time = time.time() - epoch_t0

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_top1)
                else:
                    self.scheduler.step()

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_top1": round(train_top1, 4),
                "train_top5": round(train_top5, 4),
                "val_loss": round(val_loss, 4),
                "val_top1": round(val_top1, 4),
                "val_top5": round(val_top5, 4),
                "learning_rate": current_lr,
                "epoch_time_s": round(epoch_time, 2),
            }
            self.history.append(epoch_record)

            logger.info(
                "Epoch %d/%d Complete (%.1fs) | Train Loss: %.4f | Train Acc: %.2f%% | "
                "Val Loss: %.4f | Val Top-1: %.2f%% | Val Top-5: %.2f%% | LR: %.2e",
                epoch,
                target_epochs,
                epoch_time,
                train_loss,
                train_top1 * 100.0,
                val_loss,
                val_top1 * 100.0,
                val_top5 * 100.0,
                current_lr,
            )

            # Check if this is the best epoch
            is_best = val_top1 > (self.best_val_acc + self.config.early_stopping_min_delta)
            if is_best:
                self.best_val_acc = val_top1
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(
                    self.config.best_checkpoint_path,
                    epoch=epoch,
                    val_acc=val_top1,
                    is_best=True,
                )
            else:
                self.patience_counter += 1

            # Always save last checkpoint
            self.save_checkpoint(
                self.config.last_checkpoint_path,
                epoch=epoch,
                val_acc=val_top1,
                is_best=False,
            )

            # Save updated history
            history_path = self.root / self.config.metrics_dir / "training_history.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)

            # Check early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    "Early stopping triggered at epoch %d. Best validation Top-1: %.4f at epoch %d.",
                    epoch,
                    self.best_val_acc,
                    self.best_epoch,
                )
                break

        total_time = time.time() - total_train_start

        summary = {
            "status": "COMPLETED",
            "architecture": self.config.architecture,
            "total_epochs_trained": len(self.history),
            "best_epoch": self.best_epoch,
            "best_val_top1": round(self.best_val_acc, 4),
            "total_training_time_s": round(total_time, 2),
            "best_checkpoint_path": self.config.best_checkpoint_path,
            "last_checkpoint_path": self.config.last_checkpoint_path,
        }

        # Save summary
        summary_path = self.root / self.config.metrics_dir / "training_metrics.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
