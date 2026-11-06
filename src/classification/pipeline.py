"""
Integrated Vehicle Recognition Pipeline: YOLO Vehicle Detection + Deep Classification.
Connects the Phase 4 YOLO object detector with the Phase 5 fine-grained classifier,
extracts bounding-box crops, classifies each detected vehicle into Make, Model, and Year,
and provides comparative benchmark utilities (Ground Truth Crops vs. Detector Crops).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.classification.inference import VehicleClassifier
from src.detection.detector import VehicleDetector
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class VehicleRecognitionPipeline:
    """
    End-to-End Fine-Grained Vehicle Recognition System.
    Combines YOLO detector with deep classifier.
    """

    def __init__(
        self,
        detector: VehicleDetector | None = None,
        classifier: VehicleClassifier | None = None,
        detector_weights: str | Path | None = None,
        classifier_checkpoint: str | Path | None = None,
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.root = find_project_root()
        self.detector = detector or VehicleDetector(
            model_path=detector_weights,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            extract_rois=True,
        )
        self.classifier = classifier or VehicleClassifier(
            checkpoint_path=classifier_checkpoint,
            device=device,
        )

    def process_image(
        self,
        image_input: str | Path | Image.Image | np.ndarray,
        extract_rois: bool = True,
    ) -> dict[str, Any]:
        """
        Execute end-to-end detection and classification on an input scene image.

        Returns:
            Dictionary matching pipeline contract:
            {
                "image_width": int,
                "image_height": int,
                "num_vehicles_detected": int,
                "detections": [
                    {
                        "bbox": {"x1": int, "y1": int, "x2": int, "y2": int},
                        "detection_confidence": float,
                        "classification": {
                            "predicted_class": str,
                            "make": str,
                            "model": str,
                            "year": int,
                            "confidence": float,
                            "top_5": list
                        }
                    }
                ],
                "timing": {
                    "detection_ms": float,
                    "classification_ms": float,
                    "total_pipeline_ms": float
                }
            }
        """
        t_start = time.perf_counter()

        # 1. Detection Stage
        det_result = self.detector.detect(image_input, extract_rois=True)
        det_time_ms = det_result["inference_time_ms"]

        cls_start = time.perf_counter()
        pipeline_detections: list[dict[str, Any]] = []

        # 2. Classification Stage for each detected vehicle ROI
        for det in det_result["detections"]:
            roi = det.get("roi")
            if roi is not None and getattr(roi, "size", 0) > 0:
                cls_res = self.classifier.predict(roi)
            else:
                # If no ROI extracted or empty box, fallback to classifying the full image
                cls_res = self.classifier.predict(image_input)

            pipeline_detections.append({
                "bbox": det["bbox"],
                "detection_confidence": round(det["confidence"], 4),
                "detector_class": det.get("class_name", "car"),
                "classification": {
                    "predicted_class": cls_res["predicted_class"],
                    "class_id": cls_res.get("class_id", -1),
                    "make": cls_res["make"],
                    "model": cls_res["model"],
                    "year": cls_res["year"],
                    "confidence": cls_res["confidence"],
                    "top_5": cls_res.get("top_5", []),
                    "inference_time_ms": cls_res["inference_time_ms"],
                },
            })

        cls_time_ms = round((time.perf_counter() - cls_start) * 1000.0, 2)
        total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        return {
            "image_width": det_result["image_width"],
            "image_height": det_result["image_height"],
            "num_vehicles_detected": len(pipeline_detections),
            "detections": pipeline_detections,
            "timing": {
                "detection_ms": det_time_ms,
                "classification_ms": cls_time_ms,
                "total_pipeline_ms": total_ms,
            },
        }

    def compare_gt_vs_detector_crops(
        self,
        samples: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Compare classification accuracy on controlled Ground Truth crops vs. YOLO-generated crops.
        Separates controlled classification capability from end-to-end detection+classification errors.

        Args:
            samples: List of dictionaries with keys:
                     'image_path', 'true_class_id', 'bbox' (x1, y1, x2, y2).

        Returns:
            Structured comparative report.
        """
        gt_correct = 0
        yolo_correct = 0
        total = 0

        for s in samples:
            img_p = self.root / s["image_path"] if not Path(s["image_path"]).is_file() else Path(s["image_path"])
            if not img_p.is_file():
                continue

            true_id = int(s["true_class_id"])
            if true_id < 0:
                continue

            total += 1
            with Image.open(img_p) as pil_img:
                rgb = pil_img.convert("RGB")
                w, h = rgb.size

                # 1. Ground Truth Crop
                x1, y1, x2, y2 = s.get("bbox", (0, 0, w, h))
                gt_crop = rgb.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
                gt_pred = self.classifier.predict(gt_crop)
                if gt_pred.get("class_id") == true_id:
                    gt_correct += 1

                # 2. End-to-End YOLO Detection Crop
                pipe_res = self.process_image(rgb)
                if pipe_res["detections"]:
                    # Best detection by confidence
                    best_det = max(pipe_res["detections"], key=lambda d: d["detection_confidence"])
                    if best_det["classification"].get("class_id") == true_id:
                        yolo_correct += 1

        gt_acc = gt_correct / max(1, total)
        yolo_acc = yolo_correct / max(1, total)

        return {
            "samples_evaluated": total,
            "ground_truth_crop_accuracy": round(gt_acc, 4),
            "yolo_detector_crop_accuracy": round(yolo_acc, 4),
            "delta_accuracy": round(gt_acc - yolo_acc, 4),
            "note": (
                "Ground-truth crop accuracy represents isolated classifier accuracy. "
                "YOLO detector crop accuracy represents cascaded end-to-end pipeline accuracy "
                "which accounts for bounding-box IoU offsets and imperfect vehicle framing."
            ),
        }
