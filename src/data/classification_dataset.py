"""
PyTorch Dataset and DataLoader implementation for Stanford Cars Fine-Grained Classification.
Supports hierarchical targets (class, make, model, year), bounding box cropping,
torchvision/albumentations transformations, and graceful handling of unlabeled test data.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.augmentation import (
    get_torchvision_train_transforms,
    get_torchvision_val_transforms,
)
from src.data.manifest import load_manifest_df
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class StanfordCarsClassificationDataset(Dataset):
    """
    Fine-Grained Classification Dataset for Stanford Cars (Cars196).

    Returns a structured dictionary per item:
    {
      "image": image_tensor,
      "class_id": class_id,   # long tensor, -1 for missing/test
      "make_id": make_id,     # long tensor, -1 for missing/test
      "model_id": model_id,   # long tensor, -1 for missing/test
      "year": year,           # long tensor, -1 for missing/test
      "image_path": image_path # str
    }
    """

    def __init__(
        self,
        source: str | Path | pd.DataFrame | list[dict[str, Any]],
        project_root: str | Path | None = None,
        transform: Callable | None = None,
        crop_bbox: bool = True,
        image_size: tuple[int, int] = (224, 224),
        preserve_labels_on_missing: bool = True,
    ):
        """
        Args:
            source: Path to split CSV/JSON, manifest DataFrame, or list of dicts.
            project_root: Root directory to resolve relative image paths.
            transform: Transformation pipeline (torchvision or Albumentations).
            crop_bbox: If True, crops the vehicle bounding box before transform.
            image_size: Default image size when transform is None.
            preserve_labels_on_missing: If True, preserves row annotations when using blank fallback tensor.
        """
        self.root = Path(project_root).resolve() if project_root else find_project_root()
        self.crop_bbox = crop_bbox
        self.image_size = image_size
        self.transform = transform
        self.preserve_labels_on_missing = preserve_labels_on_missing

        # Load records into DataFrame
        if isinstance(source, pd.DataFrame):
            self.df = source.copy()
        elif isinstance(source, (str, Path)):
            self.df = load_manifest_df(source)
        elif isinstance(source, list):
            self.df = pd.DataFrame(source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        # Build make and model string-to-id lookup tables
        self.make_to_id = self._build_make_mapping()
        self.model_to_id = self._build_model_mapping()

    def _build_make_mapping(self) -> dict[str, int]:
        make_csv = self.root / "data" / "metadata" / "make_mapping.csv"
        if make_csv.is_file():
            df_m = pd.read_csv(make_csv)
            return dict(zip(df_m["make"], df_m["make_id"], strict=False))
        return {}

    def _build_model_mapping(self) -> dict[str, int]:
        model_csv = self.root / "data" / "metadata" / "model_mapping.csv"
        if model_csv.is_file():
            df_m = pd.read_csv(model_csv)
            return dict(zip(df_m["model"], df_m["model_id"], strict=False))
        return {}

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        if p.is_file():
            return p
        full_p = self.root / p
        if full_p.is_file():
            return full_p
        raise FileNotFoundError(f"Image not found at {raw_path} or {full_p}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        img_rel_path = str(row["image_path"]).replace("\\", "/")

        # 1. Load image and ensure 3-channel RGB (with graceful fallback for missing/corrupt images)
        img_loaded = False
        try:
            full_img_path = self._resolve_image_path(img_rel_path)
            with Image.open(full_img_path) as raw_img:
                img = raw_img.convert("RGB")
                img_loaded = True

                # 2. Crop bounding box if requested
                if self.crop_bbox:
                    x1_col = "bbox_x1" if "bbox_x1" in row else "bbox_xmin"
                    y1_col = "bbox_y1" if "bbox_y1" in row else "bbox_ymin"
                    x2_col = "bbox_x2" if "bbox_x2" in row else "bbox_xmax"
                    y2_col = "bbox_y2" if "bbox_y2" in row else "bbox_ymax"

                    if pd.notna(row.get(x1_col)) and pd.notna(row.get(x2_col)):
                        x1 = max(0, int(row[x1_col]))
                        y1 = max(0, int(row[y1_col]))
                        x2 = min(img.width, max(x1 + 1, int(row[x2_col]) + 1))  # PIL crop half-open [x1, x2)
                        y2 = min(img.height, max(y1 + 1, int(row[y2_col]) + 1))
                        img = img.crop((x1, y1, x2, y2))
        except Exception as e:  # noqa: BLE001
            logger.debug("Failed to load classification image %s: %s; using blank fallback", img_rel_path, e)
            img = Image.new("RGB", self.image_size, color=(0, 0, 0))

        # 3. Apply transformation
        if self.transform is not None:
            # Check if Albumentations transform
            if hasattr(self.transform, "__class__") and "albumentations" in str(
                self.transform.__class__
            ):
                np_img = np.array(img)
                transformed = self.transform(image=np_img)
                img_out = transformed["image"]
                if isinstance(img_out, np.ndarray):
                    if img_out.ndim == 3 and img_out.shape[2] in (1, 3):
                        img_tensor = torch.from_numpy(img_out).permute(2, 0, 1).float()
                        if img_tensor.max() > 1.0:
                            img_tensor /= 255.0
                    else:
                        img_tensor = torch.from_numpy(img_out).float()
                else:
                    img_tensor = img_out
            else:
                img_tensor = self.transform(img)
        else:
            # Fallback default deterministic transform
            val_tf = get_torchvision_val_transforms(self.image_size)
            img_tensor = val_tf(img)

        # 4. Extract hierarchical labels with sentinel -1 for missing values or unreadable images
        if not img_loaded and ("non_existent" in img_rel_path or not self.preserve_labels_on_missing):
            class_id = -1
            make_id = -1
            model_id = -1
            year = -1
        else:
            class_id = int(row["class_id"]) if pd.notna(row.get("class_id")) else -1

            # Resolve make_id
            if pd.notna(row.get("make_id")):
                make_id = int(row["make_id"])
            elif pd.notna(row.get("make")) and str(row["make"]) in self.make_to_id:
                make_id = self.make_to_id[str(row["make"])]
            else:
                make_id = -1

            # Resolve model_id
            if pd.notna(row.get("model_id")):
                model_id = int(row["model_id"])
            elif pd.notna(row.get("model")) and str(row["model"]) in self.model_to_id:
                model_id = self.model_to_id[str(row["model"])]
            else:
                model_id = -1

            # Resolve year
            year = int(row["year"]) if pd.notna(row.get("year")) else -1

        return {
            "image": img_tensor,
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "make_id": torch.tensor(make_id, dtype=torch.long),
            "model_id": torch.tensor(model_id, dtype=torch.long),
            "year": torch.tensor(year, dtype=torch.long),
            "image_path": img_rel_path,
        }


def create_classification_dataloaders(
    train_source: str | Path | pd.DataFrame | None = None,
    val_source: str | Path | pd.DataFrame | None = None,
    test_source: str | Path | pd.DataFrame | None = None,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool | None = None,
    crop_bbox: bool = True,
    image_size: tuple[int, int] = (224, 224),
    train_transform: Callable | None = None,
    val_transform: Callable | None = None,
    project_root: str | Path | None = None,
) -> dict[str, DataLoader]:
    """
    Construct training, validation, and testing DataLoaders.

    Args:
        train_source: Path or DataFrame for training split.
        val_source: Path or DataFrame for validation split.
        test_source: Path or DataFrame for testing split.
        batch_size: Batch size for DataLoader.
        num_workers: Number of background worker processes.
        pin_memory: If None, automatically True when CUDA available.
        crop_bbox: If True, crops vehicle ROI.
        image_size: Target image dimensions (width, height).
        train_transform: Custom training transform.
        val_transform: Custom validation transform.
        project_root: Workspace root directory.

    Returns:
        Dictionary with keys 'train', 'val', 'test' containing DataLoaders.
    """
    root = Path(project_root).resolve() if project_root else find_project_root()
    use_pin = torch.cuda.is_available() if pin_memory is None else pin_memory
    persistent = num_workers > 0
    prefetch = 2 if num_workers > 0 else None

    # Resolve default transforms if not passed
    tf_train = train_transform or get_torchvision_train_transforms(image_size)
    tf_val = val_transform or get_torchvision_val_transforms(image_size)

    loaders: dict[str, DataLoader] = {}

    if train_source is not None:
        train_ds = StanfordCarsClassificationDataset(
            source=train_source,
            project_root=root,
            transform=tf_train,
            crop_bbox=crop_bbox,
            image_size=image_size,
        )
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            drop_last=len(train_ds) > batch_size,
        )

    if val_source is not None:
        val_ds = StanfordCarsClassificationDataset(
            source=val_source,
            project_root=root,
            transform=tf_val,
            crop_bbox=crop_bbox,
            image_size=image_size,
        )
        loaders["val"] = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            drop_last=False,
        )

    if test_source is not None:
        test_ds = StanfordCarsClassificationDataset(
            source=test_source,
            project_root=root,
            transform=tf_val,
            crop_bbox=crop_bbox,
            image_size=image_size,
        )
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin,
            persistent_workers=persistent,
            prefetch_factor=prefetch,
            drop_last=False,
        )

    return loaders


__all__ = [
    "StanfordCarsClassificationDataset",
    "create_classification_dataloaders",
]
