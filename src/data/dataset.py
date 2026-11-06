"""
PyTorch Dataset implementation for Stanford Cars.
Supports full-image and bounding-box crop loading, robust RGB conversion,
multi-task hierarchical targets, and custom torchvision/albumentations transforms.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class StanfordCarsDataset(Dataset):
    """
    Stanford Cars Dataset with support for:
    - Bounding-box cropping or full-image loading
    - Robust RGB conversion handling 1-channel grayscale images
    - Single-task ('class', 'make', 'year', 'body_type') or multi-task targets
    - Torchvision and Albumentations transforms
    """

    def __init__(
        self,
        annotations: str | list[dict[str, Any]],
        image_root: str | None = None,
        transform: Callable | None = None,
        crop_bbox: bool = True,
        target_type: str = "class",
        is_test: bool = False,
    ):
        """
        Args:
            annotations: Path to JSON annotations file or pre-loaded records list.
            image_root: Optional root directory override for image paths.
            transform: Callable transform (torchvision or albumentations).
            crop_bbox: If True, crops the vehicle using ground truth bounding box.
            target_type: One of 'class', 'make', 'year', 'body_type', 'multi'.
            is_test: True if loading test set without ground truth labels.
        """
        if isinstance(annotations, (str, Path)):
            with open(annotations, "r", encoding="utf-8") as f:
                self.records: list[dict[str, Any]] = json.load(f)
        else:
            self.records = list(annotations)

        self.image_root = image_root
        self.transform = transform
        self.crop_bbox = crop_bbox
        self.target_type = target_type
        self.is_test = is_test

        valid_targets = {"class", "make", "model", "year", "body_type", "multi"}
        if target_type not in valid_targets:
            raise ValueError(
                f"target_type must be one of {valid_targets}, got '{target_type}'"
            )

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, record: dict[str, Any]) -> str:
        """Resolve absolute path to image file."""
        if self.image_root:
            fname = record.get("filename", record.get("original_filename", ""))
            return os.path.join(self.image_root, fname)
        if "file_path" in record and os.path.exists(record["file_path"]):
            return record["file_path"]
        if "image_path" in record and os.path.exists(record["image_path"]):
            return record["image_path"]
        # Fallback to current working directory relative path
        if "relative_path" in record and os.path.exists(record["relative_path"]):
            return record["relative_path"]
        return record.get(
            "image_path", record.get("file_path", record.get("filename", ""))
        )

    def __getitem__(self, idx: int) -> tuple[Any, Any] | Any:
        record = self.records[idx]
        img_path = self._resolve_image_path(record)

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found at: {img_path}")

        # Open and ensure 3-channel RGB (crucial guard against grayscale images)
        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")

            # Crop bounding box if requested
            if self.crop_bbox:
                if "bbox_crop" in record:
                    crop_box = tuple(record["bbox_crop"])  # [left, top, right, bottom]
                    img = img.crop(crop_box)
                elif (
                    "bbox_xmin" in record
                    and "bbox_ymin" in record
                    and "bbox_xmax" in record
                    and "bbox_ymax" in record
                ):
                    # Half-open crop box [left, top, right, bottom] for PIL
                    crop_box = (
                        int(record["bbox_xmin"]),
                        int(record["bbox_ymin"]),
                        int(record["bbox_xmax"]) + 1,
                        int(record["bbox_ymax"]) + 1,
                    )
                    img = img.crop(crop_box)

        # Apply transformation
        if self.transform is not None:
            # Detect albumentations vs torchvision
            if hasattr(self.transform, "__class__") and "albumentations" in str(
                self.transform.__class__
            ):
                np_img = np.array(img)
                transformed = self.transform(image=np_img)
                img = transformed["image"]
            else:
                img = self.transform(img)

        fname = record.get(
            "filename",
            record.get("original_filename", Path(record.get("image_path", "")).name),
        )

        if self.is_test:
            return img, fname

        # Build target according to target_type
        if self.target_type == "class":
            target = torch.tensor(
                record.get("class_id_0", record.get("class_id", 0)), dtype=torch.long
            )
            return img, target
        elif self.target_type == "make":
            target = torch.tensor(record.get("make_id", 0), dtype=torch.long)
            return img, target
        elif self.target_type == "model":
            target = torch.tensor(record.get("model_id", 0), dtype=torch.long)
            return img, target
        elif self.target_type == "year":
            target = torch.tensor(record.get("year_id", 0), dtype=torch.long)
            return img, target
        elif self.target_type == "body_type":
            target = torch.tensor(record.get("body_type_id", 0), dtype=torch.long)
            return img, target
        else:  # 'multi'
            bbox_vals = record.get(
                "bbox_pixel_inc",
                [
                    record.get("bbox_xmin", 0),
                    record.get("bbox_ymin", 0),
                    record.get("bbox_xmax", 0),
                    record.get("bbox_ymax", 0),
                ],
            )
            targets = {
                "class_id": torch.tensor(
                    record.get("class_id_0", record.get("class_id", 0)),
                    dtype=torch.long,
                ),
                "make_id": torch.tensor(record.get("make_id", 0), dtype=torch.long),
                "model_id": torch.tensor(record.get("model_id", 0), dtype=torch.long),
                "year_id": torch.tensor(record.get("year_id", 0), dtype=torch.long),
                "body_type_id": torch.tensor(
                    record.get("body_type_id", 0), dtype=torch.long
                ),
                "bbox": torch.tensor(bbox_vals, dtype=torch.float32),
                "bbox_yolo": torch.tensor(
                    record.get("bbox_yolo", [0.0, 0.0, 0.0, 0.0]), dtype=torch.float32
                ),
                "filename": fname,
                "index": idx,
            }
            return img, targets


def get_default_transforms(
    image_size: int = 224,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Generate standard ImageNet normalization transforms for train and validation splits.
    """
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, val_transform


def build_dataloaders(
    train_json: str,
    val_json: str,
    batch_size: int = 16,
    image_size: int = 224,
    num_workers: int = 2,
    crop_bbox: bool = True,
    target_type: str = "class",
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build standard PyTorch DataLoader instances for train and validation sets.
    Configures persistent_workers and prefetch_factor for optimal Windows multiprocessing.
    """
    train_tf, val_tf = get_default_transforms(image_size=image_size)

    train_dataset = StanfordCarsDataset(
        annotations=train_json,
        transform=train_tf,
        crop_bbox=crop_bbox,
        target_type=target_type,
    )

    val_dataset = StanfordCarsDataset(
        annotations=val_json,
        transform=val_tf,
        crop_bbox=crop_bbox,
        target_type=target_type,
    )

    use_pin = torch.cuda.is_available() if pin_memory is None else pin_memory
    persistent = num_workers > 0
    prefetch = 2 if num_workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        drop_last=False,
    )

    return train_loader, val_loader
