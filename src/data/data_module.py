"""
Unified DataModule for Stanford Cars (Cars196).
Coordinates dataset loading, train/val/test splits, transform pipelines,
class imbalance mitigation via WeightedRandomSampler or loss weighting,
and hardware-optimized DataLoader creation for both classification and detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.augmentation import (
    ALBUMENTATIONS_AVAILABLE,
    get_albumentations_train_transforms,
    get_albumentations_val_transforms,
    get_torchvision_train_transforms,
    get_torchvision_val_transforms,
)
from src.data.classification_dataset import StanfordCarsClassificationDataset
from src.data.detection_dataset import (
    StanfordCarsDetectionDataset,
    collate_detection_batch,
)
from src.utils.config import ProjectConfig, find_project_root, get_config

logger = logging.getLogger(__name__)


class StanfordCarsDataModule:
    """
    Centralized data management module for Stanford Cars pipelines.
    Encapsulates both Classification and Detection loaders.
    """

    def __init__(
        self,
        config: ProjectConfig | None = None,
        train_csv: str | Path | None = None,
        val_csv: str | Path | None = None,
        test_csv: str | Path | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        image_size: tuple[int, int] | None = None,
        crop_bbox: bool = True,
        train_transform: Any = None,
        val_transform: Any = None,
        train_det_transform: Any = None,
        val_det_transform: Any = None,
    ):
        self.config = config or get_config()
        self.root = self.config.project_root or find_project_root()

        # Custom transforms
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.train_det_transform = train_det_transform
        self.val_det_transform = val_det_transform

        # Paths
        splits_dir = self.config.dataset.splits_dir or (self.root / "data" / "splits")
        self.train_csv = Path(train_csv) if train_csv else splits_dir / "train.csv"
        self.val_csv = Path(val_csv) if val_csv else splits_dir / "validation.csv"
        self.test_csv = Path(test_csv) if test_csv else splits_dir / "test.csv"

        # Parameters
        self.batch_size = batch_size or self.config.hardware.batch_size
        self.num_workers = num_workers if num_workers is not None else self.config.hardware.num_workers
        raw_size = image_size or self.config.preprocessing.image_size
        self.image_size = (int(raw_size[0]), int(raw_size[1]))
        self.crop_bbox = crop_bbox

        # Hardware settings
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = self.num_workers > 0

        # State
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        self._train_cls_dataset: StanfordCarsClassificationDataset | None = None
        self._val_cls_dataset: StanfordCarsClassificationDataset | None = None
        self._test_cls_dataset: StanfordCarsClassificationDataset | None = None

        self._train_det_dataset: StanfordCarsDetectionDataset | None = None
        self._val_det_dataset: StanfordCarsDetectionDataset | None = None
        self._test_det_dataset: StanfordCarsDetectionDataset | None = None

        self._class_weights_cache: dict[str, torch.Tensor] = {}

    def setup(self, stage: str | None = None) -> None:
        """Load split data and initialize datasets."""
        if self.train_csv.is_file():
            self.train_df = pd.read_csv(self.train_csv)
        if self.val_csv.is_file():
            self.val_df = pd.read_csv(self.val_csv)
        if self.test_csv.is_file():
            self.test_df = pd.read_csv(self.test_csv)

        # Classification Transforms
        train_tf = self.train_transform or get_torchvision_train_transforms(self.image_size)
        val_tf = self.val_transform or get_torchvision_val_transforms(self.image_size)

        # Classification Datasets
        if self.train_df is not None:
            self._train_cls_dataset = StanfordCarsClassificationDataset(
                source=self.train_df,
                project_root=self.root,
                transform=train_tf,
                crop_bbox=self.crop_bbox,
                image_size=self.image_size,
            )
        if self.val_df is not None:
            self._val_cls_dataset = StanfordCarsClassificationDataset(
                source=self.val_df,
                project_root=self.root,
                transform=val_tf,
                crop_bbox=self.crop_bbox,
                image_size=self.image_size,
            )
        if self.test_df is not None:
            self._test_cls_dataset = StanfordCarsClassificationDataset(
                source=self.test_df,
                project_root=self.root,
                transform=val_tf,
                crop_bbox=self.crop_bbox,
                image_size=self.image_size,
            )

        # Detection Transforms
        det_train_tf = self.train_det_transform
        if det_train_tf is None and ALBUMENTATIONS_AVAILABLE:
            det_train_tf = get_albumentations_train_transforms((640, 640), bbox_format="pascal_voc")

        det_val_tf = self.val_det_transform
        if det_val_tf is None and ALBUMENTATIONS_AVAILABLE:
            det_val_tf = get_albumentations_val_transforms((640, 640), bbox_format="pascal_voc")

        # Detection Datasets
        if self.train_df is not None:
            self._train_det_dataset = StanfordCarsDetectionDataset(
                source=self.train_df,
                project_root=self.root,
                transform=det_train_tf,
                image_size=(640, 640),
            )
        if self.val_df is not None:
            self._val_det_dataset = StanfordCarsDetectionDataset(
                source=self.val_df,
                project_root=self.root,
                transform=det_val_tf,
                image_size=(640, 640),
            )
        if self.test_df is not None:
            self._test_det_dataset = StanfordCarsDetectionDataset(
                source=self.test_df,
                project_root=self.root,
                transform=det_val_tf,
                image_size=(640, 640),
            )

    def get_class_weights(self, method: str = "balanced") -> torch.Tensor:
        """
        Compute or retrieve cached class weights tensor of shape (196,) for loss weighting.
        """
        if method in self._class_weights_cache:
            return self._class_weights_cache[method]

        if self.train_df is None:
            self.setup()

        assert self.train_df is not None, "train_df must be loaded"
        counts_map = self.train_df["class_id"].value_counts().to_dict()
        total_samples = len(self.train_df)
        num_classes = 196

        weights = torch.ones(num_classes, dtype=torch.float32)
        for c in range(num_classes):
            cnt = counts_map.get(c, 0)
            if cnt > 0:
                if method == "balanced":
                    weights[c] = total_samples / (float(num_classes) * float(cnt))
                elif method == "inverse":
                    weights[c] = 1.0 / float(cnt)
                elif method == "effective_samples":
                    beta = 0.999
                    effective_num = 1.0 - (beta ** cnt)
                    weights[c] = (1.0 - beta) / max(1e-8, effective_num)

        # Normalize mean to 1.0
        weights = weights / weights.mean()
        self._class_weights_cache[method] = weights
        return weights

    def build_weighted_sampler(self) -> WeightedRandomSampler:
        """Construct WeightedRandomSampler for balanced class training."""
        if self.train_df is None:
            self.setup()
        assert self.train_df is not None
        class_weights = self.get_class_weights("balanced")
        sample_targets = self.train_df["class_id"].values
        sample_weights = torch.tensor([class_weights[int(t)] for t in sample_targets], dtype=torch.double)
        return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    def train_dataloader(
        self,
        batch_size: int | None = None,
        use_weighted_sampler: bool = False,
    ) -> DataLoader:
        """Get DataLoader for classification training."""
        if self._train_cls_dataset is None:
            self.setup()
        assert self._train_cls_dataset is not None

        bs = batch_size or self.batch_size
        sampler = self.build_weighted_sampler() if use_weighted_sampler else None
        shuffle = None if sampler is not None else True

        return DataLoader(
            self._train_cls_dataset,
            batch_size=bs,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=len(self._train_cls_dataset) > bs,
        )

    def val_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Get DataLoader for classification validation."""
        if self._val_cls_dataset is None:
            self.setup()
        assert self._val_cls_dataset is not None

        bs = batch_size or self.batch_size
        return DataLoader(
            self._val_cls_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=False,
        )

    def test_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Get DataLoader for classification testing."""
        if self._test_cls_dataset is None:
            self.setup()
        assert self._test_cls_dataset is not None

        bs = batch_size or self.batch_size
        return DataLoader(
            self._test_cls_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            drop_last=False,
        )

    def detection_train_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Get DataLoader for detection training."""
        if self._train_det_dataset is None:
            self.setup()
        assert self._train_det_dataset is not None

        bs = batch_size or self.batch_size
        return DataLoader(
            self._train_det_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=collate_detection_batch,
            drop_last=len(self._train_det_dataset) > bs,
        )

    def detection_val_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Get DataLoader for detection validation."""
        if self._val_det_dataset is None:
            self.setup()
        assert self._val_det_dataset is not None

        bs = batch_size or self.batch_size
        return DataLoader(
            self._val_det_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=collate_detection_batch,
            drop_last=False,
        )


__all__ = ["StanfordCarsDataModule"]
