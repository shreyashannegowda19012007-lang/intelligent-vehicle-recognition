"""
Vehicle Detector Module using Ultralytics YOLO.
Provides inference, bounding-box restoration, NMS thresholding,
and integrated vehicle ROI extraction for downstream classification.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from src.classical_cv.roi_extractor import ROIExtractor
from src.utils.config import find_project_root
from src.utils.device import get_device

logger = logging.getLogger(__name__)

# COCO vehicle class IDs for out-of-the-box pretrained YOLO models
COCO_VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
COCO_CAR_CLASS_ID = 2


class VehicleDetector:
    """
    Robust vehicle detector wrapping Ultralytics YOLOv8.
    Produces vehicle bounding boxes, detection confidences,
    and standardized ROI crops for downstream fine-grained classification.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | torch.device = "auto",
        half: bool = False,
        roi_target_size: tuple[int, int] = (224, 224),
        roi_padding_ratio: float = 0.05,
        roi_keep_aspect_ratio: bool = True,
        class_name: str = "car",
    ) -> None:
        """
        Initialize VehicleDetector.

        Args:
            model_path: Path to YOLO weights (.pt) or checkpoint name (defaults to best.pt or yolov8n.pt).
            conf_threshold: Minimum confidence score for detection.
            iou_threshold: IoU threshold for non-maximum suppression.
            device: 'auto', 'cuda', 'cpu', or torch.device.
            half: Enable FP16 half-precision inference if GPU available.
            roi_target_size: Output dimensions (width, height) for vehicle crops.
            roi_padding_ratio: Fractional padding around detected vehicle.
            roi_keep_aspect_ratio: If True, preserves aspect ratio via letterboxing.
            class_name: Target vehicle class label to emit in detection schema.
        """
        self.root = find_project_root()
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.class_name = class_name
        self.roi_target_size = tuple(roi_target_size)
        self.roi_padding_ratio = float(roi_padding_ratio)
        self.roi_keep_aspect_ratio = bool(roi_keep_aspect_ratio)

        # Resolve device
        if isinstance(device, torch.device):
            self.device_str = "cuda:0" if device.type == "cuda" else "cpu"
        elif device == "auto":
            resolved = get_device("auto")
            self.device_str = "cuda:0" if resolved.type == "cuda" else "cpu"
        else:
            self.device_str = str(device)

        self.half = bool(half) and ("cuda" in self.device_str)

        # Resolve model path
        self.model_path = self._resolve_model_path(model_path)
        logger.info("Loading YOLO model from %s on %s (half=%s)...", self.model_path, self.device_str, self.half)
        self.model = YOLO(str(self.model_path))

        # Check model class space
        self.num_classes = len(self.model.names)
        self.is_single_class = (self.num_classes == 1)
        logger.info(
            "YOLO model loaded: %d classes. Single-class mode: %s",
            self.num_classes,
            self.is_single_class,
        )

        # Initialize ROI Extractor for downstream pipeline
        self.roi_extractor = ROIExtractor(
            target_size=self.roi_target_size,
            keep_aspect_ratio=self.roi_keep_aspect_ratio,
            padding_ratio=self.roi_padding_ratio,
        )

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> VehicleDetector:
        """
        Instantiate VehicleDetector from configuration file (configs/detection.yaml).
        """
        import yaml

        root = find_project_root()
        cfg_p = root / "configs" / "detection.yaml" if config_path is None else Path(config_path)
        if not cfg_p.is_file() and (root / cfg_p).is_file():
            cfg_p = root / cfg_p

        cfg: dict[str, Any] = {}
        if cfg_p.is_file():
            with open(cfg_p, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

        inf_cfg = cfg.get("inference", {})
        roi_cfg = cfg.get("roi", {})
        mod_cfg = cfg.get("model", {})

        model_ckpt = inf_cfg.get("model_checkpoint") or mod_cfg.get("pretrained_checkpoint")
        # If best.pt exists in models/detection, prefer it for inference
        best_cand = root / "models" / "detection" / "best.pt"
        if best_cand.is_file():
            model_ckpt = best_cand

        return cls(
            model_path=model_ckpt,
            conf_threshold=inf_cfg.get("conf_threshold", 0.25),
            iou_threshold=inf_cfg.get("iou_threshold", 0.45),
            device=inf_cfg.get("device", "auto"),
            half=inf_cfg.get("half", False),
            roi_target_size=tuple(roi_cfg.get("target_size", (224, 224))),
            roi_padding_ratio=roi_cfg.get("padding_ratio", 0.05),
            roi_keep_aspect_ratio=roi_cfg.get("keep_aspect_ratio", True),
            class_name=mod_cfg.get("class_names", ["car"])[0] if mod_cfg.get("class_names") else "car",
        )

    def _resolve_model_path(self, model_path: str | Path | None) -> Path:
        """Resolve model checkpoint path with fallback search."""
        if model_path is None or str(model_path).lower() in {"auto", "best", "best.pt"}:
            # Check for fine-tuned best checkpoint first
            for cand in [
                self.root / "models" / "detection" / "best.pt",
                self.root / "models" / "detection" / "yolov8n_vehicle" / "weights" / "best.pt",
                self.root / "yolov8n.pt",
            ]:
                if cand.is_file():
                    return cand.resolve()
            return Path("yolov8n.pt")

        p = Path(model_path)
        if p.is_file():
            return p.resolve()

        # Check in project root
        p_root = self.root / p
        if p_root.is_file():
            return p_root.resolve()

        # Check in models/detection
        p_det = self.root / "models" / "detection" / p
        if p_det.is_file():
            return p_det.resolve()

        # Check in models/detection/yolov8n_vehicle/weights/
        p_sub = self.root / "models" / "detection" / "yolov8n_vehicle" / "weights" / p
        if p_sub.is_file():
            return p_sub.resolve()

        # Fallback to yolov8n.pt if candidate not found on disk
        yolo_fallback = self.root / "yolov8n.pt"
        if yolo_fallback.is_file():
            logger.warning("Checkpoint '%s' not found on disk; falling back to %s", model_path, yolo_fallback)
            return yolo_fallback.resolve()

        # If it's a standard Ultralytics name (e.g. 'yolov8n.pt'), allow Ultralytics to resolve/download
        return p

    def _load_image_as_rgb(
        self,
        image_input: str | Path | np.ndarray | Image.Image,
    ) -> tuple[np.ndarray, int, int]:
        """
        Normalize input image into RGB numpy array (H, W, 3) and return dimensions.

        Returns:
            Tuple of (rgb_image, width, height)
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                path = self.root / path
            if not path.is_file():
                raise FileNotFoundError(f"Image not found at {image_input}")
            with Image.open(path) as pil_img:
                rgb_img = np.array(pil_img.convert("RGB"), dtype=np.uint8)
        elif isinstance(image_input, Image.Image):
            rgb_img = np.array(image_input.convert("RGB"), dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                rgb_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 1:
                rgb_img = cv2.cvtColor(image_input[:, :, 0], cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 4:
                rgb_img = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                rgb_img = image_input.copy()
            else:
                raise ValueError(f"Unsupported numpy image shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Normalize floating-point representations to uint8 [0, 255]
        if np.issubdtype(rgb_img.dtype, np.floating):
            if rgb_img.size > 0 and rgb_img.max() <= 1.0 + 1e-5:
                rgb_img = (rgb_img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                rgb_img = rgb_img.clip(0, 255).astype(np.uint8)
        elif rgb_img.dtype != np.uint8:
            rgb_img = rgb_img.astype(np.uint8)

        h, w = rgb_img.shape[:2]
        return rgb_img, w, h

    def detect(
        self,
        image: str | Path | np.ndarray | Image.Image,
        conf: float | None = None,
        iou: float | None = None,
        extract_rois: bool = True,
    ) -> dict[str, Any]:
        """
        Perform vehicle detection and ROI extraction on a single image.

        Args:
            image: Image path, numpy array, or PIL Image.
            conf: Optional confidence threshold override.
            iou: Optional IoU NMS threshold override.
            extract_rois: Whether to extract cropped vehicle ROI arrays.

        Returns:
            Dictionary matching specification schema:
            {
                "detections": [
                    {
                        "bbox": {"x1": int, "y1": int, "x2": int, "y2": int},
                        "class_id": 0,
                        "class_name": "car",
                        "confidence": float,
                        "roi": np.ndarray (or None if extract_rois=False)
                    }
                ],
                "inference_time_ms": float,
                "image_width": int,
                "image_height": int
            }
        """
        conf_thresh = self.conf_threshold if conf is None else float(conf)
        iou_thresh = self.iou_threshold if iou is None else float(iou)

        # 1. Normalize image
        rgb_img, orig_w, orig_h = self._load_image_as_rgb(image)

        if orig_w <= 0 or orig_h <= 0:
            return {
                "detections": [],
                "inference_time_ms": 0.0,
                "image_width": max(0, orig_w),
                "image_height": max(0, orig_h),
            }

        # 2. Filter classes:
        # If single-class model, class 0 is car/vehicle
        # If multi-class COCO model, restrict to vehicles [2, 3, 5, 7]
        target_classes = None if self.is_single_class else list(COCO_VEHICLE_CLASS_IDS)

        # 3. Perform inference and time it
        predict_kwargs: dict[str, Any] = {
            "source": rgb_img,
            "conf": conf_thresh,
            "iou": iou_thresh,
            "device": self.device_str,
            "classes": target_classes,
            "verbose": False,
        }
        if self.half:
            predict_kwargs["half"] = True

        t_start = time.perf_counter()
        results = self.model.predict(**predict_kwargs)
        t_end = time.perf_counter()
        inference_time_ms = round((t_end - t_start) * 1000.0, 2)

        detections: list[dict[str, Any]] = []

        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            xyxy_tensor = boxes.xyxy.cpu().numpy()
            confs_tensor = boxes.conf.cpu().numpy()
            clss_tensor = boxes.cls.cpu().numpy()

            for i in range(len(xyxy_tensor)):
                raw_x1, raw_y1, raw_x2, raw_y2 = xyxy_tensor[i]
                c_score = float(confs_tensor[i])
                _raw_cls = int(clss_tensor[i])

                # Safe clamping and integer restoration to original image bounds
                x1 = int(max(0, min(orig_w - 1, round(raw_x1))))
                y1 = int(max(0, min(orig_h - 1, round(raw_y1))))
                x2 = int(max(x1 + 1, min(orig_w, round(raw_x2))))
                y2 = int(max(y1 + 1, min(orig_h, round(raw_y2))))

                # Verify non-degenerate area
                if (x2 <= x1) or (y2 <= y1):
                    continue

                bbox_dict = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

                # Extract vehicle ROI for downstream classifier
                roi_array = None
                if extract_rois:
                    roi_array, _ = self.roi_extractor.extract_roi(
                        image=rgb_img,
                        bbox=(x1, y1, x2, y2),
                        use_bbox=True,
                        is_inclusive=False,
                    )

                detections.append(
                    {
                        "bbox": bbox_dict,
                        "class_id": 0,  # Unified vehicle detection class 0
                        "class_name": self.class_name,
                        "confidence": round(c_score, 4),
                        "roi": roi_array,
                    }
                )

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        return {
            "detections": detections,
            "inference_time_ms": inference_time_ms,
            "image_width": orig_w,
            "image_height": orig_h,
        }

    def detect_batch(
        self,
        images: list[str | Path | np.ndarray | Image.Image],
        conf: float | None = None,
        iou: float | None = None,
        extract_rois: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Run efficient batch vehicle detection across a list of images.
        Uses native Ultralytics batching for accelerated GPU/CPU throughput.
        """
        if not images:
            return []

        conf_thresh = self.conf_threshold if conf is None else float(conf)
        iou_thresh = self.iou_threshold if iou is None else float(iou)
        target_classes = None if self.is_single_class else list(COCO_VEHICLE_CLASS_IDS)

        # 1. Normalize all images into RGB numpy arrays and record shapes
        rgb_list: list[np.ndarray] = []
        dims_list: list[tuple[int, int]] = []

        for idx, img_input in enumerate(images):
            try:
                rgb_img, orig_w, orig_h = self._load_image_as_rgb(img_input)
                rgb_list.append(rgb_img)
                dims_list.append((orig_w, orig_h))
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to load batch image %d: %s; using placeholder", idx, e)
                rgb_list.append(np.zeros((64, 64, 3), dtype=np.uint8))
                dims_list.append((0, 0))

        # 2. Run batched prediction
        predict_kwargs: dict[str, Any] = {
            "source": rgb_list,
            "conf": conf_thresh,
            "iou": iou_thresh,
            "device": self.device_str,
            "classes": target_classes,
            "verbose": False,
        }
        if self.half:
            predict_kwargs["half"] = True

        t_start = time.perf_counter()
        results = self.model.predict(**predict_kwargs)
        t_end = time.perf_counter()
        total_time_ms = (t_end - t_start) * 1000.0
        avg_time_ms = round(total_time_ms / max(1, len(images)), 2)

        batch_outputs: list[dict[str, Any]] = []

        for i, res in enumerate(results):
            orig_w, orig_h = dims_list[i]
            if orig_w <= 0 or orig_h <= 0:
                batch_outputs.append({
                    "detections": [],
                    "inference_time_ms": avg_time_ms,
                    "image_width": max(0, orig_w),
                    "image_height": max(0, orig_h),
                })
                continue

            dets: list[dict[str, Any]] = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy_tensor = res.boxes.xyxy.cpu().numpy()
                confs_tensor = res.boxes.conf.cpu().numpy()

                for j in range(len(xyxy_tensor)):
                    raw_x1, raw_y1, raw_x2, raw_y2 = xyxy_tensor[j]
                    c_score = float(confs_tensor[j])

                    x1 = int(max(0, min(orig_w - 1, round(raw_x1))))
                    y1 = int(max(0, min(orig_h - 1, round(raw_y1))))
                    x2 = int(max(x1 + 1, min(orig_w, round(raw_x2))))
                    y2 = int(max(y1 + 1, min(orig_h, round(raw_y2))))

                    if (x2 <= x1) or (y2 <= y1):
                        continue

                    roi_array = None
                    if extract_rois:
                        roi_array, _ = self.roi_extractor.extract_roi(
                            image=rgb_list[i],
                            bbox=(x1, y1, x2, y2),
                            use_bbox=True,
                            is_inclusive=False,
                        )

                    dets.append({
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "class_id": 0,
                        "class_name": self.class_name,
                        "confidence": round(c_score, 4),
                        "roi": roi_array,
                    })

            dets.sort(key=lambda d: d["confidence"], reverse=True)
            batch_outputs.append({
                "detections": dets,
                "inference_time_ms": avg_time_ms,
                "image_width": orig_w,
                "image_height": orig_h,
            })

        return batch_outputs
