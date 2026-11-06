"""
Unified Central Pipeline Controller for Hybrid Vehicle Recognition.
Integrates YOLO vehicle detection, classical CV baseline, and deep fine-grained classification
into a robust end-to-end processing pipeline with detailed telemetry and failure isolation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.hybrid.adapters import (
    BaseClassifierAdapter,
    ClassicalAdapter,
    DeepLearningAdapter,
    DetectorAdapter,
)
from src.hybrid.config import DecisionStrategy, HybridConfig
from src.hybrid.decision_engine import HybridDecisionEngine
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class HybridVehiclePipeline:
    """
    Central Hybrid Intelligence Pipeline Controller.
    Locates vehicles via YOLO, extracts ROIs, runs classical and deep classifiers,
    fuses predictions via configurable decision strategies, and emits structured analysis.
    """

    def __init__(
        self,
        config: HybridConfig | None = None,
        detector: Any | None = None,
        dl_classifier: Any | None = None,
        classical_predictor: Any | None = None,
    ) -> None:
        self.root = find_project_root()
        self.config = config or HybridConfig.from_yaml()

        # Initialize or inject component adapters
        if isinstance(detector, DetectorAdapter) or hasattr(detector, "detect_vehicles"):
            self.detector_adapter = detector
        else:
            self.detector_adapter = DetectorAdapter(
                detector=detector,
                model_path=self.config.detector_checkpoint,
                conf_threshold=self.config.detector_conf_threshold,
                iou_threshold=self.config.detector_iou_threshold,
                device=self.config.detector_device,
                roi_target_size=self.config.roi_target_size,
            )

        if isinstance(dl_classifier, (BaseClassifierAdapter, DeepLearningAdapter)) or hasattr(dl_classifier, "predict_crop"):
            self.dl_adapter = dl_classifier
        else:
            self.dl_adapter = DeepLearningAdapter(
                classifier=dl_classifier,
                checkpoint_path=self.config.dl_checkpoint_path,
                class_mapping_path=self.config.class_mapping_path,
                device=self.config.dl_device,
                top_k=self.config.top_k,
                temperature=self.config.dl_temperature,
            )

        if isinstance(classical_predictor, (BaseClassifierAdapter, ClassicalAdapter)) or hasattr(classical_predictor, "predict_crop"):
            self.classical_adapter = classical_predictor
        else:
            self.classical_adapter = ClassicalAdapter(
                predictor=classical_predictor,
                model_path=self.config.classical_model_path,
                pipeline_path=self.config.classical_pipeline_path,
                class_mapping_path=self.config.class_mapping_path,
                uncertainty_threshold=self.config.classical_uncertainty_threshold,
            )

        # Initialize decision engine
        self.decision_engine = HybridDecisionEngine(config=self.config)

    def validate_image(
        self, image_input: str | Path | Image.Image | np.ndarray | Any
    ) -> tuple[np.ndarray | None, int, int, str | None]:
        """
        Validate and normalize input image.

        Returns:
            Tuple of (rgb_image_array, width, height, error_message).
        """
        if image_input is None:
            return None, 0, 0, "Image input is None."

        try:
            if isinstance(image_input, (str, Path)):
                p = Path(image_input)
                if not p.is_file():
                    p = self.root / p
                if not p.is_file():
                    return None, 0, 0, f"Image file not found at: {image_input}"

                try:
                    with Image.open(p) as pil_img:
                        pil_img.verify()
                except Exception as e:
                    return None, 0, 0, f"Image file is corrupted or invalid format: {e}"

                with Image.open(p) as pil_img:
                    rgb_img = np.array(pil_img.convert("RGB"), dtype=np.uint8)

            elif isinstance(image_input, Image.Image):
                rgb_img = np.array(image_input.convert("RGB"), dtype=np.uint8)

            elif isinstance(image_input, np.ndarray):
                if image_input.size == 0 or image_input.ndim < 2:
                    return None, 0, 0, f"Invalid numpy array shape for image: {image_input.shape}"

                if image_input.ndim == 2:
                    rgb_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
                elif image_input.ndim == 3 and image_input.shape[2] == 1:
                    rgb_img = cv2.cvtColor(image_input[:, :, 0], cv2.COLOR_GRAY2RGB)
                elif image_input.ndim == 3 and image_input.shape[2] == 4:
                    rgb_img = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
                elif image_input.ndim == 3 and image_input.shape[2] == 3:
                    rgb_img = image_input.copy()
                else:
                    return None, 0, 0, f"Unsupported image channels: {image_input.shape[2]}"

                # Normalize float images
                if np.issubdtype(rgb_img.dtype, np.floating):
                    if rgb_img.max() <= 1.0 + 1e-4:
                        rgb_img = (rgb_img * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        rgb_img = rgb_img.clip(0, 255).astype(np.uint8)
                elif rgb_img.dtype != np.uint8:
                    rgb_img = rgb_img.astype(np.uint8)

            else:
                return None, 0, 0, f"Unsupported image type: {type(image_input)}"

            h, w = rgb_img.shape[:2]
            if h <= 0 or w <= 0:
                return None, 0, 0, f"Image has non-positive dimensions: ({w}x{h})"

            return rgb_img, w, h, None

        except Exception as e:
            return None, 0, 0, f"Unexpected error during image validation: {e}"

    def process_image(
        self,
        image_input: str | Path | Image.Image | np.ndarray | Any,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        decision_strategy: DecisionStrategy | None = None,
    ) -> dict[str, Any]:
        """
        Execute full hybrid pipeline on a single image.

        1. Validate image.
        2. Record original image dimensions.
        3. Run YOLO detection and extract vehicle bounding boxes and crops.
        4. Classify each crop through DL and Classical models.
        5. Compare predictions and fuse decisions.
        6. Measure component and total latency.

        Returns:
            Unified response dictionary matching specification.
        """
        t_total_start = time.perf_counter()
        pipeline_warnings: list[str] = []

        # 1. Image Validation
        rgb_img, orig_w, orig_h, val_err = self.validate_image(image_input)
        if val_err is not None or rgb_img is None:
            logger.warning("Image validation failed: %s", val_err)
            total_elapsed = round((time.perf_counter() - t_total_start) * 1000.0, 2)
            return {
                "status": "error",
                "image": {"width": 0, "height": 0},
                "vehicle_count": 0,
                "vehicles": [],
                "performance": {
                    "detection_time_ms": 0.0,
                    "classical_time_ms": 0.0,
                    "deep_learning_time_ms": 0.0,
                    "classification_time_ms": 0.0,
                    "decision_time_ms": 0.0,
                    "total_time_ms": total_elapsed,
                },
                "warnings": [f"Image validation error: {val_err}"],
            }

        # 2. YOLO Vehicle Detection
        t_det_start = time.perf_counter()
        det_result = self.detector_adapter.detect_vehicles(
            image_input=rgb_img,
            conf=conf_threshold,
            iou=iou_threshold,
        )
        detection_time_ms = round((time.perf_counter() - t_det_start) * 1000.0, 2)

        detections = det_result.get("detections", [])
        if len(detections) == 0:
            pipeline_warnings.append("No vehicles detected in scene.")
            total_elapsed = round((time.perf_counter() - t_total_start) * 1000.0, 2)
            return {
                "status": "success",
                "image": {"width": orig_w, "height": orig_h},
                "vehicle_count": 0,
                "vehicles": [],
                "performance": {
                    "detection_time_ms": detection_time_ms,
                    "classical_time_ms": 0.0,
                    "deep_learning_time_ms": 0.0,
                    "classification_time_ms": 0.0,
                    "decision_time_ms": 0.0,
                    "total_time_ms": total_elapsed,
                },
                "warnings": pipeline_warnings,
            }

        # 3. Process Each Detected Vehicle
        vehicle_results: list[dict[str, Any]] = []
        total_dl_time_ms = 0.0
        total_cv_time_ms = 0.0
        total_dec_time_ms = 0.0

        for det in detections:
            v_id = det["vehicle_id"]
            bbox = det["bbox"]
            roi = det.get("roi")

            # Fallback crop if ROI extraction was None or empty
            if roi is None or getattr(roi, "size", 0) == 0:
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                roi = rgb_img[y1:y2, x1:x2]
                if roi.size == 0:
                    roi = rgb_img

            # A. Deep Learning Classification
            t_dl0 = time.perf_counter()
            try:
                dl_pred = self.dl_adapter.predict_crop(roi)
            except Exception as e:
                logger.error("DL classifier exception on vehicle %d: %s", v_id, e)
                dl_pred = {
                    "status": "error",
                    "class_id": -1,
                    "predicted_class": "Unknown",
                    "make": "Unknown",
                    "model": "Unknown",
                    "year": 0,
                    "confidence": 0.0,
                    "score": 0.0,
                    "is_calibrated": False,
                    "is_uncertain": True,
                    "top_5": [],
                    "inference_time_ms": 0.0,
                    "warning": f"DL classifier crashed: {e}",
                    "error": str(e),
                }
            dl_latency = round((time.perf_counter() - t_dl0) * 1000.0, 2)
            total_dl_time_ms += dl_latency

            # B. Classical CV Classification
            t_cv0 = time.perf_counter()
            try:
                cv_pred = self.classical_adapter.predict_crop(roi)
            except Exception as e:
                logger.error("Classical classifier exception on vehicle %d: %s", v_id, e)
                cv_pred = {
                    "status": "error",
                    "class_id": -1,
                    "predicted_class": "Unknown",
                    "make": "Unknown",
                    "model": "Unknown",
                    "year": 0,
                    "confidence": 0.0,
                    "score": 0.0,
                    "is_calibrated": False,
                    "is_uncertain": True,
                    "top_5": [],
                    "inference_time_ms": 0.0,
                    "warning": f"Classical classifier crashed: {e}",
                    "error": str(e),
                }
            cv_latency = round((time.perf_counter() - t_cv0) * 1000.0, 2)
            total_cv_time_ms += cv_latency

            # C. Hybrid Decision and Confidence Analysis
            t_dec0 = time.perf_counter()
            v_analysis = self.decision_engine.decide(
                vehicle_id=v_id,
                bbox=bbox,
                detection_info={
                    "class_name": det.get("class_name", "car"),
                    "confidence": det.get("confidence", 0.0),
                },
                dl_pred=dl_pred,
                classical_pred=cv_pred,
                strategy_override=decision_strategy,
            )
            dec_latency = round((time.perf_counter() - t_dec0) * 1000.0, 3)
            total_dec_time_ms += dec_latency

            vehicle_results.append(v_analysis)

        classification_time_ms = round(total_dl_time_ms + total_cv_time_ms, 2)
        total_time_ms = round((time.perf_counter() - t_total_start) * 1000.0, 2)

        return {
            "status": "success",
            "image": {
                "width": orig_w,
                "height": orig_h,
            },
            "vehicle_count": len(vehicle_results),
            "vehicles": vehicle_results,
            "performance": {
                "detection_time_ms": round(detection_time_ms, 2),
                "classical_time_ms": round(total_cv_time_ms, 2),
                "deep_learning_time_ms": round(total_dl_time_ms, 2),
                "classification_time_ms": classification_time_ms,
                "decision_time_ms": round(total_dec_time_ms, 3),
                "total_time_ms": total_time_ms,
            },
            "warnings": pipeline_warnings,
        }
