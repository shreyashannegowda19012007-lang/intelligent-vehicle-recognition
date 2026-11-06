"""
Web Application Utility Module.
Provides image decoding, bounding-box visual overlays, and multi-model agreement analysis.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    """
    Decode raw bytes into a RGB PIL Image.

    Raises:
        ValueError: If bytes cannot be decoded as an image.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Received empty image byte payload.")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}") from e


def draw_bounding_boxes(
    image: Image.Image,
    detections: Sequence[dict[str, Any]],
    box_color: tuple[int, int, int] = (0, 168, 255),  # Cyan/Blue
    text_color: tuple[int, int, int] = (255, 255, 255),
    line_width: int = 4,
) -> Image.Image:
    """
    Render bounding boxes and label badges onto a PIL Image.

    Args:
        image: Original RGB PIL image.
        detections: List of detection dicts containing 'bbox' and label metadata.
        box_color: RGB tuple for bounding box edges.
        text_color: RGB tuple for label text.
        line_width: Box stroke width.

    Returns:
        New PIL Image with drawn annotations.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    for idx, det in enumerate(detections):
        bbox = det.get("bbox", {})
        if isinstance(bbox, dict):
            x1 = max(0, int(bbox.get("x1", 0)))
            y1 = max(0, int(bbox.get("y1", 0)))
            x2 = min(w, int(bbox.get("x2", w)))
            y2 = min(h, int(bbox.get("y2", h)))
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        else:
            continue

        # Draw box rectangle
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)

        # Prepare label text
        cls_info = det.get("classification", {})
        pred_label = cls_info.get("predicted_class") or det.get("class_name", "Vehicle")
        conf = det.get("detection_confidence") or det.get("confidence", 0.0)
        label_text = f"#{idx+1} {pred_label} ({conf*100:.1f}%)" if conf else f"#{idx+1} {pred_label}"

        # Draw background badge for label text
        font = ImageFont.load_default()
        bbox_text = draw.textbbox((x1, y1), label_text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]

        badge_y1 = max(0, y1 - text_h - 8)
        badge_y2 = y1 if badge_y1 > 0 else y1 + text_h + 8
        badge_x2 = min(w, x1 + text_w + 10)

        draw.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=box_color)
        draw.text((x1 + 5, badge_y1 + 4), label_text, fill=text_color, font=font)

    return annotated


def analyze_model_agreement(
    classical_pred: dict[str, Any],
    resnet_pred: dict[str, Any],
    hybrid_pred: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute consensus and disagreement metrics between prediction streams.

    Args:
        classical_pred: Result from classical ML baseline.
        resnet_pred: Result from ResNet50 deep learning model.
        hybrid_pred: Result from hybrid fusion pipeline.

    Returns:
        Structured agreement dictionary.
    """
    c_make = str(classical_pred.get("make", "")).strip().lower()
    r_make = str(resnet_pred.get("make", "")).strip().lower()
    h_make = str(hybrid_pred.get("make", "")).strip().lower()

    c_class = str(classical_pred.get("predicted_class", "")).strip().lower()
    r_class = str(resnet_pred.get("predicted_class", "")).strip().lower()
    h_class = str(hybrid_pred.get("predicted_class", "")).strip().lower()

    # Exact class consensus
    all_agree_class = (c_class == r_class == h_class) and (c_class != "")
    partial_agree_class = (c_class == r_class) or (r_class == h_class) or (c_class == h_class)

    # Make level consensus
    all_agree_make = (c_make == r_make == h_make) and (c_make != "")
    partial_agree_make = (c_make == r_make) or (r_make == h_make) or (c_make == h_make)

    if all_agree_class:
        status = "Full Consensus"
        summary = f"All models fully agree on fine-grained class: '{resnet_pred.get('predicted_class')}'"
    elif all_agree_make:
        status = "Make Consensus"
        summary = f"Models agree on manufacturer ({resnet_pred.get('make')}) with fine-grained trim divergence."
    elif partial_agree_make:
        status = "Partial Consensus"
        summary = "Two out of three models agree on manufacturer styling cues."
    else:
        status = "Disagreement"
        summary = "Models generated divergent vehicle hypotheses across disparate manufacturers."

    return {
        "status": status,
        "summary": summary,
        "all_agree_class": all_agree_class,
        "all_agree_make": all_agree_make,
        "partial_agree_make": partial_agree_make,
    }

