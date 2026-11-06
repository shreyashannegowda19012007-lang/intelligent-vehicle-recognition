"""
Vehicle ROI Extraction Module for Classical Computer Vision.
Handles Stanford Cars bounding box parsing, coordinate clamping, optional controlled padding,
aspect-ratio preserving letterbox resizing, and fallback handling for invalid or missing bounding boxes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocessing import load_image, resize_image
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class ROIExtractor:
    """
    Extracts and standardizes vehicle Regions of Interest (ROIs) from images
    using ground-truth bounding box coordinates or full-frame fallback.
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (128, 128),
        keep_aspect_ratio: bool = True,
        padding_ratio: float = 0.05,
        interpolation: int | None = None,
    ) -> None:
        """
        Args:
            target_size: Output (width, height) for extracted ROI crops.
            keep_aspect_ratio: If True, uses letterboxing to preserve aspect ratio.
            padding_ratio: Fractional padding around the bounding box (e.g. 0.05 = 5% padding per side).
            interpolation: OpenCV interpolation mode (default INTER_AREA for downscale, INTER_LINEAR for upscale).
        """
        self.target_size = tuple(target_size)
        self.keep_aspect_ratio = keep_aspect_ratio
        self.padding_ratio = max(0.0, float(padding_ratio))
        self.interpolation = interpolation

    def parse_bbox_coordinates(
        self,
        bbox: tuple[int, int, int, int] | list[int] | dict[str, Any] | pd.Series | None,
        img_width: int,
        img_height: int,
        is_matlab_coord: bool = False,
        is_inclusive: bool = True,
    ) -> tuple[int, int, int, int] | None:
        """
        Parse and validate bounding box coordinates into 0-based half-open pixel bounds [x1, y1, x2, y2].

        Args:
            bbox: Bounding box as tuple/list [x1, y1, x2, y2] or dict/Series with bbox keys.
            img_width: Image width in pixels.
            img_height: Image height in pixels.
            is_matlab_coord: If True, treats input coordinates as 1-based MATLAB coordinates.
            is_inclusive: If True and not MATLAB coord, treats [x2, y2] as inclusive pixel coordinates.

        Returns:
            Tuple (x1, y1, x2, y2) or None if bbox is invalid/missing.
        """
        if bbox is None or img_width <= 0 or img_height <= 0:
            return None

        # Extract raw coordinates from dict / Series / sequence
        try:
            if isinstance(bbox, (dict, pd.Series)):
                # If explicit MATLAB keys exist and no pixel keys, mark as matlab coord
                if "bbox_matlab_x1" in bbox and pd.notna(bbox["bbox_matlab_x1"]) and "bbox_x1" not in bbox and "bbox_xmin" not in bbox:
                    x1_raw = float(bbox["bbox_matlab_x1"])
                    y1_raw = float(bbox["bbox_matlab_y1"])
                    x2_raw = float(bbox["bbox_matlab_x2"])
                    y2_raw = float(bbox["bbox_matlab_y2"])
                    is_matlab_coord = True
                elif "bbox_x1" in bbox and pd.notna(bbox["bbox_x1"]):
                    x1_raw = float(bbox["bbox_x1"])
                    y1_raw = float(bbox["bbox_y1"])
                    x2_raw = float(bbox["bbox_x2"])
                    y2_raw = float(bbox["bbox_y2"])
                    is_matlab_coord = False
                elif "bbox_xmin" in bbox and pd.notna(bbox["bbox_xmin"]):
                    x1_raw = float(bbox["bbox_xmin"])
                    y1_raw = float(bbox["bbox_ymin"])
                    x2_raw = float(bbox["bbox_xmax"])
                    y2_raw = float(bbox["bbox_ymax"])
                    is_matlab_coord = False
                elif "x1" in bbox and pd.notna(bbox["x1"]):
                    x1_raw = float(bbox["x1"])
                    y1_raw = float(bbox["y1"])
                    x2_raw = float(bbox["x2"])
                    y2_raw = float(bbox["y2"])
                else:
                    return None
            elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1_raw, y1_raw, x2_raw, y2_raw = [float(c) for c in bbox]
            else:
                return None
        except (ValueError, TypeError, KeyError):
            return None

        # Guard against NaN / Inf
        if not (np.isfinite(x1_raw) and np.isfinite(y1_raw) and np.isfinite(x2_raw) and np.isfinite(y2_raw)):
            return None

        # Convert indexing to 0-based half-open [x1, x2)
        if is_matlab_coord:
            # 1-based inclusive MATLAB [1, W] -> 0-based half-open [0, W)
            x1 = round(x1_raw) - 1
            y1 = round(y1_raw) - 1
            x2 = round(x2_raw)
            y2 = round(y2_raw)
        elif is_inclusive:
            # 0-based inclusive [0, W-1] -> 0-based half-open [0, W)
            x1 = round(x1_raw)
            y1 = round(y1_raw)
            x2 = round(x2_raw) + 1
            y2 = round(y2_raw) + 1
        else:
            # Already 0-based half-open
            x1 = round(x1_raw)
            y1 = round(y1_raw)
            x2 = round(x2_raw)
            y2 = round(y2_raw)

        # Check for degenerate or inverted coordinates
        if x2 <= x1 or y2 <= y1:
            return None

        # Check if box completely falls outside image boundaries
        if x1 >= img_width or y1 >= img_height or x2 <= 0 or y2 <= 0:
            return None

        # Clamp within image bounds
        x1_clamped = max(0, min(img_width - 1, x1))
        y1_clamped = max(0, min(img_height - 1, y1))
        x2_clamped = max(x1_clamped + 1, min(img_width, x2))
        y2_clamped = max(y1_clamped + 1, min(img_height, y2))

        # Check if clamped box has positive area
        if (x2_clamped - x1_clamped) < 1 or (y2_clamped - y1_clamped) < 1:
            return None

        return (x1_clamped, y1_clamped, x2_clamped, y2_clamped)

    def apply_padding(
        self,
        box: tuple[int, int, int, int],
        img_width: int,
        img_height: int,
    ) -> tuple[int, int, int, int]:
        """
        Apply controlled padding expansion around bounding box, clamped to image borders.
        """
        if self.padding_ratio <= 0.0:
            return box

        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        pad_x = round(bw * self.padding_ratio)
        pad_y = round(bh * self.padding_ratio)

        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(img_width, x2 + pad_x)
        py2 = min(img_height, y2 + pad_y)

        return (px1, py1, px2, py2)

    def extract_roi(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int] | list[int] | dict[str, Any] | pd.Series | None = None,
        use_bbox: bool = True,
        is_matlab_coord: bool = False,
        is_inclusive: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Extract vehicle ROI from image and resize to target_size.

        Args:
            image: Input image numpy array (H, W, C) or (H, W).
            bbox: Bounding box coordinates (optional).
            use_bbox: If True, crops to bounding box; if False, uses full image.
            is_matlab_coord: If True, adjusts 1-based MATLAB coordinates.
            is_inclusive: If True, treats pixel coordinates as inclusive [x1, y1, x2, y2].

        Returns:
            Tuple of (resized_crop_array, metadata_dict).
        """
        orig_h, orig_w = image.shape[:2]
        fallback_used = False
        parsed_box = None
        padded_box = None

        if orig_h <= 0 or orig_w <= 0:
            channels = image.shape[2] if image.ndim == 3 else 1
            out_shape = (self.target_size[1], self.target_size[0], channels) if channels > 1 else (self.target_size[1], self.target_size[0])
            return np.zeros(out_shape, dtype=np.uint8), {
                "original_width": orig_w,
                "original_height": orig_h,
                "parsed_bbox": None,
                "applied_bbox": (0, 0, 0, 0),
                "fallback_used": True,
                "target_size": self.target_size,
                "keep_aspect_ratio": self.keep_aspect_ratio,
            }

        if use_bbox and bbox is not None:
            parsed_box = self.parse_bbox_coordinates(
                bbox=bbox,
                img_width=orig_w,
                img_height=orig_h,
                is_matlab_coord=is_matlab_coord,
                is_inclusive=is_inclusive,
            )

        if parsed_box is not None:
            padded_box = self.apply_padding(parsed_box, orig_w, orig_h)
            x1, y1, x2, y2 = padded_box
            crop = image[y1:y2, x1:x2].copy()
        else:
            if use_bbox and bbox is not None:
                fallback_used = True
                logger.debug("Invalid bounding box %s; fell back to full image (%dx%d)", bbox, orig_w, orig_h)
            crop = image.copy()
            x1, y1, x2, y2 = 0, 0, orig_w, orig_h
            padded_box = (x1, y1, x2, y2)

        # Resize crop to target size
        resized_crop = resize_image(
            crop,
            target_size=self.target_size,
            keep_aspect_ratio=self.keep_aspect_ratio,
            interpolation=self.interpolation,
        )

        metadata = {
            "original_width": orig_w,
            "original_height": orig_h,
            "parsed_bbox": parsed_box,
            "applied_bbox": padded_box,
            "fallback_used": fallback_used,
            "target_size": self.target_size,
            "keep_aspect_ratio": self.keep_aspect_ratio,
        }

        return resized_crop, metadata

    def extract_from_row(
        self,
        row: dict[str, Any] | pd.Series,
        project_root: str | Path | None = None,
        use_bbox: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Load image from row['image_path'] and extract ROI using row annotations.

        Args:
            row: Manifest row containing 'image_path' and bounding box coordinates.
            project_root: Project root directory for relative paths.
            use_bbox: Whether to crop vehicle bounding box.

        Returns:
            Tuple of (resized_crop_array, metadata_dict).
        """
        root = Path(project_root).resolve() if project_root else find_project_root()
        raw_path = Path(str(row["image_path"]).replace("\\", "/"))
        full_path = raw_path if raw_path.is_file() else root / raw_path

        if not full_path.is_file():
            raise FileNotFoundError(f"Image not found at: {full_path}")

        image = load_image(full_path, as_rgb=True)

        # Stanford Cars manifest (train.csv/vehicle_manifest.csv) stores 0-based inclusive coordinates
        # (bbox_x1, bbox_y1, bbox_x2, bbox_y2). devkit stores 1-based MATLAB coordinates (bbox_matlab_x1).
        is_matlab = bool("bbox_matlab_x1" in row and "bbox_x1" not in row and "bbox_xmin" not in row)
        res_crop, meta = self.extract_roi(
            image=image,
            bbox=row,
            use_bbox=use_bbox,
            is_matlab_coord=is_matlab,
            is_inclusive=True,
        )
        meta["image_path"] = str(full_path)
        meta["image_id"] = str(row.get("image_id", ""))
        return res_crop, meta

    def batch_extract(
        self,
        images: list[np.ndarray],
        bboxes: list[Any] | None = None,
        use_bbox: bool = True,
    ) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        """
        Batch extract ROIs for a list of images.
        """
        crops: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []

        for idx, img in enumerate(images):
            box = bboxes[idx] if (bboxes is not None and idx < len(bboxes)) else None
            crop, meta = self.extract_roi(img, bbox=box, use_bbox=use_bbox)
            crops.append(crop)
            metas.append(meta)

        return crops, metas
