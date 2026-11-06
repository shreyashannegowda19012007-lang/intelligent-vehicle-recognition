"""
Vehicle Detection Visualizer.
Generates annotated bounding box overlays, confidence badges,
multi-vehicle views, and individual extracted ROI visual collections.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class VehicleVisualizer:
    """
    Visualization tool for vehicle detections and ROI extractions.
    """

    def __init__(
        self,
        box_color: tuple[int, int, int] = (0, 255, 128),  # Vibrant Green
        text_color: tuple[int, int, int] = (255, 255, 255),
        line_thickness: int = 2,
        font_scale: float = 0.6,
    ) -> None:
        self.root = find_project_root()
        self.box_color = box_color
        self.text_color = text_color
        self.line_thickness = line_thickness
        self.font_scale = font_scale

    def draw_detections(
        self,
        image: str | Path | np.ndarray | Image.Image,
        detections: list[dict[str, Any]],
        show_conf: bool = True,
        show_label: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes and label tags onto image.

        Args:
            image: Input image (Path, numpy RGB array, or PIL Image).
            detections: List of detection dictionaries matching schema.
            show_conf: Whether to render confidence percentage/score.
            show_label: Whether to render class name.

        Returns:
            Annotated RGB numpy array.
        """
        if isinstance(image, (str, Path)):
            p = self.root / image if not Path(image).is_file() else Path(image)
            with Image.open(p) as pil_img:
                img = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        elif isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"), dtype=np.uint8)
        elif isinstance(image, np.ndarray):
            if np.issubdtype(image.dtype, np.floating):
                if image.size > 0 and image.max() <= 1.0 + 1e-5:
                    img = (image * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img = image.clip(0, 255).astype(np.uint8)
            else:
                img = image.copy().astype(np.uint8)

            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Draw each detection
        for i, det in enumerate(detections):
            bbox = det.get("bbox", {})
            x1 = int(bbox.get("x1", 0))
            y1 = int(bbox.get("y1", 0))
            x2 = int(bbox.get("x2", 0))
            y2 = int(bbox.get("y2", 0))

            conf = float(det.get("confidence", 0.0))
            cls_name = str(det.get("class_name", "car"))

            # 1. Draw bounding box rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), self.box_color, self.line_thickness)

            # 2. Build tag text
            tag_parts = []
            if show_label:
                tag_parts.append(f"#{i+1} {cls_name}")
            if show_conf:
                tag_parts.append(f"{conf:.2f}")
            tag = " | ".join(tag_parts) if tag_parts else f"#{i+1}"

            # 3. Render label background banner
            (tw, th), baseline = cv2.getTextSize(
                tag, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            banner_h = th + baseline + 6
            # If box is near top border, draw banner inside box to prevent off-screen clipping
            if y1 - banner_h < 0:
                banner_y1 = y1
                banner_y2 = min(img.shape[0], y1 + banner_h)
                text_pos = (x1 + 3, min(img.shape[0] - 2, y1 + th + 3))
            else:
                banner_y1 = max(0, y1 - banner_h)
                banner_y2 = y1
                text_pos = (x1 + 3, banner_y2 - baseline - 2)

            banner_x2 = min(img.shape[1], x1 + tw + 6)

            cv2.rectangle(
                img,
                (x1, banner_y1),
                (banner_x2, banner_y2),
                self.box_color,
                thickness=-1,
            )

            # 4. Text overlay
            cv2.putText(
                img,
                tag,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (0, 0, 0),  # Dark text on vibrant banner
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return img

    def visualize_rois_grid(
        self,
        detections: list[dict[str, Any]],
        grid_size: tuple[int, int] = (224, 224),
        max_rois: int = 8,
    ) -> np.ndarray | None:
        """
        Create a horizontal strip or grid of extracted vehicle ROIs.
        """
        rois = [d["roi"] for d in detections if d.get("roi") is not None][:max_rois]
        if not rois:
            return None

        target_w, target_h = grid_size
        resized = []
        for r in rois:
            if (r.shape[1], r.shape[0]) != (target_w, target_h):
                r_resized = cv2.resize(r, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                r_resized = r
            resized.append(r_resized)

        # Concatenate horizontally
        return np.hstack(resized)

    def save_image(
        self,
        image_rgb: np.ndarray,
        save_path: str | Path,
    ) -> Path:
        """Save RGB numpy image to disk."""
        out_p = self.root / save_path if not Path(save_path).is_absolute() else Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if np.issubdtype(image_rgb.dtype, np.floating):
            if image_rgb.size > 0 and image_rgb.max() <= 1.0 + 1e-5:
                image_rgb = (image_rgb * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image_rgb = image_rgb.clip(0, 255).astype(np.uint8)
        elif image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)

        pil_img = Image.fromarray(image_rgb)
        pil_img.save(out_p)
        return out_p
