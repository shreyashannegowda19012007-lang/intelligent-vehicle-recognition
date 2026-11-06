"""
PyTorch Detection Dataset and YOLO formatting pipeline for Stanford Cars.
Maps all vehicle instances to single class 'vehicle' (class_id = 0).
Converts pixel bounding boxes into normalized YOLO format [0, 1].
Supports custom transforms and bounding-box collation.
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
    get_torchvision_val_transforms,
)
from src.data.manifest import load_manifest_df
from src.utils.box_utils import box_to_yolo
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)

VEHICLE_DETECTION_CLASS_ID = 0
VEHICLE_DETECTION_CLASS_NAME = "vehicle"


class StanfordCarsDetectionDataset(Dataset):
    """
    Object Detection Dataset for Stanford Cars.
    Extracts vehicle bounding boxes, maps to single category 0 ('vehicle'),
    and outputs both pixel coordinates and normalized YOLO coordinates.
    """

    def __init__(
        self,
        source: str | Path | pd.DataFrame | list[dict[str, Any]],
        project_root: str | Path | None = None,
        transform: Callable | None = None,
        image_size: tuple[int, int] = (640, 640),
        yolo_normalized: bool = True,
    ):
        """
        Args:
            source: Path to manifest/split CSV, DataFrame, or records list.
            project_root: Workspace root directory.
            transform: Transformation function/pipeline.
            image_size: Target image dimensions (width, height).
            yolo_normalized: If True, box coordinates in target are normalized to [0, 1].
        """
        self.root = Path(project_root).resolve() if project_root else find_project_root()
        self.image_size = image_size
        self.transform = transform
        self.yolo_normalized = yolo_normalized

        if isinstance(source, pd.DataFrame):
            self.df = source.copy()
        elif isinstance(source, (str, Path)):
            self.df = load_manifest_df(source)
        elif isinstance(source, list):
            self.df = pd.DataFrame(source)
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        row = self.df.iloc[idx]
        img_rel_path = str(row["image_path"]).replace("\\", "/")

        # 1. Load image and ensure 3-channel RGB (with graceful fallback for missing/corrupt images)
        img_loaded = False
        try:
            full_img_path = self._resolve_image_path(img_rel_path)
            with Image.open(full_img_path) as raw_img:
                img = raw_img.convert("RGB")
                orig_w, orig_h = img.size
                img_loaded = True
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load detection image %s: %s; using blank fallback", img_rel_path, e)
            orig_w, orig_h = self.image_size[0], self.image_size[1]
            img = Image.new("RGB", (orig_w, orig_h), color=(0, 0, 0))

        if not img_loaded:
            val_tf = get_torchvision_val_transforms(self.image_size)
            img_tensor = val_tf(img)
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "boxes_pixel": torch.zeros((0, 4), dtype=torch.float32),
                "boxes_yolo": torch.zeros((0, 4), dtype=torch.float32),
                "orig_boxes_pixel": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
                "image_id": str(row.get("image_id", idx)),
                "image_path": img_rel_path,
                "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            }
            return img_tensor, target

        # 2. Extract bounding box
        x1_col = "bbox_x1" if "bbox_x1" in row else "bbox_xmin"
        y1_col = "bbox_y1" if "bbox_y1" in row else "bbox_ymin"
        x2_col = "bbox_x2" if "bbox_x2" in row else "bbox_xmax"
        y2_col = "bbox_y2" if "bbox_y2" in row else "bbox_ymax"

        x1 = float(row[x1_col])
        y1 = float(row[y1_col])
        x2 = float(row[x2_col])
        y2 = float(row[y2_col])
        orig_boxes_pixel = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)

        # Convert to initial YOLO format (normalized xc, yc, w, h)
        xc_norm, yc_norm, w_norm, h_norm = box_to_yolo(
            x1, y1, x2, y2, orig_w, orig_h, inclusive=True
        )

        # 3. Apply transformations
        if self.transform is not None:
            if hasattr(self.transform, "__class__") and "albumentations" in str(
                self.transform.__class__
            ):
                np_img = np.array(img)

                # Detect Albumentations bbox format if configured
                bbox_format = "pascal_voc"
                if hasattr(self.transform, "processors") and "bboxes" in self.transform.processors:
                    proc = self.transform.processors["bboxes"]
                    bbox_format = getattr(getattr(proc, "params", None), "format", "pascal_voc") or "pascal_voc"

                if bbox_format == "yolo":
                    in_boxes = [[xc_norm, yc_norm, w_norm, h_norm]]
                elif bbox_format == "albumentations":
                    in_boxes = [[x1 / orig_w, y1 / orig_h, x2 / orig_w, y2 / orig_h]]
                else:
                    in_boxes = [[x1, y1, x2, y2]]

                transformed = self.transform(
                    image=np_img,
                    bboxes=in_boxes,
                    category_ids=[VEHICLE_DETECTION_CLASS_ID],
                )
                img_out = transformed["image"]

                # Ensure image is a 3-channel PyTorch tensor [3, H, W]
                if isinstance(img_out, np.ndarray):
                    if img_out.ndim == 3 and img_out.shape[2] in (1, 3):
                        img_tensor = torch.from_numpy(img_out).permute(2, 0, 1).float()
                        if img_tensor.max() > 1.0:
                            img_tensor /= 255.0
                    else:
                        img_tensor = torch.from_numpy(img_out).float()
                else:
                    img_tensor = img_out

                new_h, new_w = int(img_tensor.shape[1]), int(img_tensor.shape[2])

                if "bboxes" in transformed and len(transformed["bboxes"]) > 0:
                    b = transformed["bboxes"][0]
                    if bbox_format == "yolo":
                        xc, yc, bw, bh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                        bx1 = max(0.0, (xc - bw / 2.0) * new_w)
                        by1 = max(0.0, (yc - bh / 2.0) * new_h)
                        bx2 = min(float(new_w), (xc + bw / 2.0) * new_w)
                        by2 = min(float(new_h), (yc + bh / 2.0) * new_h)
                    elif bbox_format == "albumentations":
                        bx1 = max(0.0, float(b[0]) * new_w)
                        by1 = max(0.0, float(b[1]) * new_h)
                        bx2 = min(float(new_w), float(b[2]) * new_w)
                        by2 = min(float(new_h), float(b[3]) * new_h)
                        xc, yc, bw, bh = box_to_yolo(bx1, by1, bx2, by2, new_w, new_h, inclusive=True)
                    else:  # pascal_voc
                        bx1 = max(0.0, min(float(new_w), float(b[0])))
                        by1 = max(0.0, min(float(new_h), float(b[1])))
                        bx2 = max(bx1, min(float(new_w), float(b[2])))
                        by2 = max(by1, min(float(new_h), float(b[3])))
                        xc, yc, bw, bh = box_to_yolo(bx1, by1, bx2, by2, new_w, new_h, inclusive=True)

                    boxes_pixel = torch.tensor([[bx1, by1, bx2, by2]], dtype=torch.float32)
                    boxes_yolo = torch.tensor([[xc, yc, bw, bh]], dtype=torch.float32)
                    labels = torch.tensor([VEHICLE_DETECTION_CLASS_ID], dtype=torch.long)
                else:
                    # Bounding box dropped by augmentations (e.g. out of frame / visibility threshold)
                    boxes_pixel = torch.zeros((0, 4), dtype=torch.float32)
                    boxes_yolo = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.long)
            else:
                img_tensor = self.transform(img)
                new_h, new_w = int(img_tensor.shape[1]), int(img_tensor.shape[2])
                boxes_yolo = torch.tensor([[xc_norm, yc_norm, w_norm, h_norm]], dtype=torch.float32)
                x1_scaled = x1 * (new_w / orig_w)
                y1_scaled = y1 * (new_h / orig_h)
                x2_scaled = x2 * (new_w / orig_w)
                y2_scaled = y2 * (new_h / orig_h)
                boxes_pixel = torch.tensor([[x1_scaled, y1_scaled, x2_scaled, y2_scaled]], dtype=torch.float32)
                labels = torch.tensor([VEHICLE_DETECTION_CLASS_ID], dtype=torch.long)
        else:
            val_tf = get_torchvision_val_transforms(self.image_size)
            img_tensor = val_tf(img)
            new_h, new_w = int(img_tensor.shape[1]), int(img_tensor.shape[2])
            boxes_yolo = torch.tensor([[xc_norm, yc_norm, w_norm, h_norm]], dtype=torch.float32)
            x1_scaled = x1 * (new_w / orig_w)
            y1_scaled = y1 * (new_h / orig_h)
            x2_scaled = x2 * (new_w / orig_w)
            y2_scaled = y2 * (new_h / orig_h)
            boxes_pixel = torch.tensor([[x1_scaled, y1_scaled, x2_scaled, y2_scaled]], dtype=torch.float32)
            labels = torch.tensor([VEHICLE_DETECTION_CLASS_ID], dtype=torch.long)

        target = {
            "boxes": boxes_yolo if self.yolo_normalized else boxes_pixel,
            "boxes_pixel": boxes_pixel,
            "boxes_yolo": boxes_yolo,
            "orig_boxes_pixel": orig_boxes_pixel,
            "labels": labels,
            "image_id": str(row.get("image_id", idx)),
            "image_path": img_rel_path,
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
        }

        return img_tensor, target


def collate_detection_batch(
    batch: list[tuple[torch.Tensor, dict[str, Any]]]
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Collate function for batching detection items with variable targets."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def create_detection_dataloaders(
    train_source: str | Path | pd.DataFrame | None = None,
    val_source: str | Path | pd.DataFrame | None = None,
    test_source: str | Path | pd.DataFrame | None = None,
    batch_size: int = 16,
    num_workers: int = 2,
    pin_memory: bool | None = None,
    image_size: tuple[int, int] = (640, 640),
    train_transform: Callable | None = None,
    val_transform: Callable | None = None,
    project_root: str | Path | None = None,
) -> dict[str, DataLoader]:
    """Build PyTorch DataLoaders for detection datasets."""
    root = Path(project_root).resolve() if project_root else find_project_root()
    use_pin = torch.cuda.is_available() if pin_memory is None else pin_memory
    persistent = num_workers > 0
    prefetch = 2 if num_workers > 0 else None

    loaders: dict[str, DataLoader] = {}

    for split_name, src, tf in [
        ("train", train_source, train_transform),
        ("val", val_source, val_transform),
        ("test", test_source, val_transform),
    ]:
        if src is not None:
            ds = StanfordCarsDetectionDataset(
                source=src,
                project_root=root,
                transform=tf,
                image_size=image_size,
            )
            loaders[split_name] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                num_workers=num_workers,
                pin_memory=use_pin,
                persistent_workers=persistent,
                prefetch_factor=prefetch,
                collate_fn=collate_detection_batch,
                drop_last=(split_name == "train" and len(ds) > batch_size),
            )

    return loaders


__all__ = [
    "VEHICLE_DETECTION_CLASS_ID",
    "VEHICLE_DETECTION_CLASS_NAME",
    "StanfordCarsDetectionDataset",
    "collate_detection_batch",
    "create_detection_dataloaders",
]
