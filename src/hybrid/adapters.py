"""
Component Adapters for Hybrid Vehicle Recognition System.
Provides unified, decoupled interfaces for the YOLO detector, Classical CV classifier,
and Deep Learning fine-grained classifier to isolate model-specific quirks.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.classification.inference import VehicleClassifier
from src.classification.metrics import ClassMappingDecoder
from src.detection.detector import VehicleDetector
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class BaseClassifierAdapter(ABC):
    """Abstract interface defining the standardized contract for vehicle classifiers."""

    name: str = "base"
    is_available: bool = False

    @abstractmethod
    def predict_crop(self, crop: np.ndarray | Image.Image | str | Path) -> dict[str, Any]:
        """
        Classify a single vehicle crop.

        Returns:
            Standardized dictionary:
            {
                "status": "success" | "unavailable" | "error",
                "class_id": int,
                "predicted_class": str,
                "make": str,
                "model": str,
                "year": int,
                "confidence": float,
                "score": float,
                "is_calibrated": bool,
                "is_uncertain": bool,
                "top_5": list[dict[str, Any]],
                "inference_time_ms": float,
                "warning": str | None,
                "error": str | None
            }
        """
        raise NotImplementedError


class DeepLearningAdapter(BaseClassifierAdapter):
    """Adapter wrapping the deep learning fine-grained classifier (Phase 5)."""

    name: str = "deep_learning"

    def __init__(
        self,
        classifier: VehicleClassifier | None = None,
        checkpoint_path: str | Path | None = None,
        class_mapping_path: str | Path | None = None,
        device: str = "auto",
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> None:
        self.root = find_project_root()
        self.top_k = top_k
        self.device = device
        self.classifier: VehicleClassifier | None = None
        self.is_available = False
        self.init_error: str | None = None

        if classifier is not None:
            self.classifier = classifier
            self.is_available = True
        else:
            try:
                ckpt = self._resolve_checkpoint(checkpoint_path)
                self.classifier = VehicleClassifier(
                    checkpoint_path=ckpt,
                    class_mapping_path=class_mapping_path,
                    device=device,
                    top_k=top_k,
                    temperature=temperature,
                )
                self.is_available = True
                logger.info("DeepLearningAdapter successfully initialized with checkpoint: %s", ckpt)
            except Exception as e:
                self.init_error = str(e)
                logger.warning("DeepLearningAdapter unavailable: %s", e)
                self.is_available = False

    def _resolve_checkpoint(self, checkpoint_path: str | Path | None) -> Path | None:
        if checkpoint_path is None:
            cand = self.root / "outputs" / "classification" / "checkpoints" / "best_checkpoint.pt"
            return cand if cand.is_file() else None
        p = Path(checkpoint_path)
        if p.is_file():
            return p.resolve()
        p_root = self.root / p
        if p_root.is_file():
            return p_root.resolve()
        return p

    def predict_crop(self, crop: np.ndarray | Image.Image | str | Path) -> dict[str, Any]:
        if not self.is_available or self.classifier is None:
            return {
                "status": "unavailable",
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
                "warning": "Deep learning model checkpoint unavailable.",
                "error": self.init_error,
            }

        t0 = time.perf_counter()
        try:
            res = self.classifier.predict(crop, top_k=self.top_k)
            latency_ms = res.get("inference_time_ms", round((time.perf_counter() - t0) * 1000.0, 2))

            top_5 = res.get("top_5", [])
            conf = float(res.get("confidence", 0.0))

            return {
                "status": "success",
                "class_id": int(res.get("class_id", -1)),
                "predicted_class": str(res.get("predicted_class", "Unknown")),
                "make": str(res.get("make", "Unknown")),
                "model": str(res.get("model", "Unknown")),
                "year": int(res.get("year", 0)),
                "confidence": round(conf, 4),
                "score": round(conf, 4),
                "is_calibrated": True,  # Softmax output with temperature calibration
                "is_uncertain": bool(conf < 0.20),
                "top_5": top_5,
                "inference_time_ms": round(latency_ms, 2),
                "warning": None if conf >= 0.20 else "Low deep-learning classification confidence.",
                "error": None,
            }
        except Exception as e:
            logger.error("DeepLearningAdapter prediction failed: %s", e)
            return {
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
                "inference_time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "warning": "Deep learning prediction failed on this crop.",
                "error": str(e),
            }


class ClassicalAdapter(BaseClassifierAdapter):
    """Adapter wrapping the Classical CV + ML classifier (Phase 3)."""

    name: str = "classical_cv"

    def __init__(
        self,
        predictor: Any | None = None,
        model_path: str | Path | None = None,
        pipeline_path: str | Path | None = None,
        class_mapping_path: str | Path | None = None,
        uncertainty_threshold: float = 0.30,
    ) -> None:
        self.root = find_project_root()
        self.uncertainty_threshold = uncertainty_threshold
        self.predictor = predictor
        self.is_available = False
        self.init_error: str | None = None

        if self.predictor is not None:
            self.is_available = True
        else:
            resolved_model = self._resolve_model(model_path)
            if resolved_model and resolved_model.is_file():
                try:
                    from src.classical_cv.predictor import ClassicalVehiclePredictor

                    self.predictor = ClassicalVehiclePredictor(
                        model_path=resolved_model,
                        pipeline_path=pipeline_path,
                        class_mapping_path=class_mapping_path,
                        project_root=self.root,
                    )
                    self.is_available = True
                    logger.info("ClassicalAdapter successfully initialized from: %s", resolved_model)
                except Exception as e:
                    self.init_error = str(e)
                    logger.warning("ClassicalAdapter initialization failed: %s", e)
                    self.is_available = False
            else:
                self.init_error = f"Classical model artifact not found at {model_path}"
                logger.info("ClassicalAdapter inactive: %s", self.init_error)
                self.is_available = False

    def _resolve_model(self, model_path: str | Path | None) -> Path | None:
        if model_path is None:
            cand = self.root / "models" / "classical" / "svm" / "linear_svm_hog_color_lbp_model.joblib"
            if cand.is_file():
                return cand
            return None
        p = Path(model_path)
        if p.is_file():
            return p.resolve()
        p_root = self.root / p
        if p_root.is_file():
            return p_root.resolve()
        return None

    def predict_crop(self, crop: np.ndarray | Image.Image | str | Path) -> dict[str, Any]:
        if not self.is_available or self.predictor is None:
            return {
                "status": "unavailable",
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
                "warning": "Classical CV model checkpoint unavailable.",
                "error": self.init_error,
            }

        t0 = time.perf_counter()
        try:
            # Prepare image array
            if isinstance(crop, Image.Image):
                arr = np.array(crop.convert("RGB"), dtype=np.uint8)
            elif isinstance(crop, (str, Path)):
                arr = np.array(Image.open(crop).convert("RGB"), dtype=np.uint8)
            elif isinstance(crop, np.ndarray):
                arr = crop
            else:
                raise TypeError(f"Unsupported crop type for ClassicalAdapter: {type(crop)}")

            res = self.predictor.predict(arr, use_bbox=False, top_k=5)
            latency_ms = res.get("total_latency_ms", round((time.perf_counter() - t0) * 1000.0, 2))

            conf = float(res.get("confidence", 0.0))
            is_calibrated = bool(res.get("is_calibrated", False))
            is_uncertain = conf < self.uncertainty_threshold

            warning = None
            if not is_calibrated:
                warning = "Classical model score is an uncalibrated decision value, not a true probability."
            elif is_uncertain:
                warning = "Classical classifier score indicates high prediction uncertainty."

            return {
                "status": "success",
                "class_id": int(res.get("class_id", -1)),
                "predicted_class": str(res.get("class_name", "Unknown")),
                "make": str(res.get("make", "Unknown")),
                "model": str(res.get("model", "Unknown")),
                "year": int(res.get("year", 0)),
                "confidence": round(conf, 4) if is_calibrated else 0.0,
                "score": round(conf, 4),
                "is_calibrated": is_calibrated,
                "is_uncertain": is_uncertain,
                "top_5": res.get("top_5", []),
                "inference_time_ms": round(latency_ms, 2),
                "warning": warning,
                "error": None,
            }
        except Exception as e:
            logger.error("ClassicalAdapter prediction failed: %s", e)
            return {
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
                "inference_time_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "warning": "Classical CV inference encountered an exception.",
                "error": str(e),
            }


class DetectorAdapter:
    """Adapter wrapping the YOLO object detector (Phase 4)."""

    def __init__(
        self,
        detector: VehicleDetector | None = None,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        roi_target_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.root = find_project_root()
        if detector is not None:
            self.detector = detector
        else:
            resolved_pt = self._resolve_weights(model_path)
            self.detector = VehicleDetector(
                model_path=resolved_pt,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                device=device,
                roi_target_size=roi_target_size,
            )

    def _resolve_weights(self, model_path: str | Path | None) -> Path | str:
        if model_path is not None:
            p = Path(model_path)
            if p.is_file():
                return p.resolve()
            if (self.root / p).is_file():
                return (self.root / p).resolve()
        # Fallback search
        for cand in [self.root / "models" / "detection" / "best.pt", self.root / "yolov8n.pt"]:
            if cand.is_file():
                return cand.resolve()
        return "yolov8n.pt"

    def detect_vehicles(
        self,
        image_input: str | Path | Image.Image | np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
    ) -> dict[str, Any]:
        """
        Execute YOLO detection and return standardized bounding boxes with vehicle crops.

        Returns:
            {
                "image_width": int,
                "image_height": int,
                "detections": [
                    {
                        "vehicle_id": int,
                        "bbox": {"x1": int, "y1": int, "x2": int, "y2": int},
                        "confidence": float,
                        "class_name": str,
                        "roi": np.ndarray
                    }
                ],
                "inference_time_ms": float
            }
        """
        if hasattr(self.detector, "detect_vehicles"):
            det_result = self.detector.detect_vehicles(
                image_input=image_input,
                conf=conf,
                iou=iou,
            )
        else:
            det_result = self.detector.detect(
                image=image_input,
                conf=conf,
                iou=iou,
                extract_rois=True,
            )

        orig_w = det_result["image_width"]
        orig_h = det_result["image_height"]

        formatted_detections: list[dict[str, Any]] = []
        for idx, det in enumerate(det_result.get("detections", [])):
            b = det["bbox"]
            # Ensure coordinates are integers clamped strictly inside original image bounds
            x1 = max(0, min(orig_w - 1, int(round(b["x1"]))))
            y1 = max(0, min(orig_h - 1, int(round(b["y1"]))))
            x2 = max(x1 + 1, min(orig_w, int(round(b["x2"]))))
            y2 = max(y1 + 1, min(orig_h, int(round(b["y2"]))))

            formatted_detections.append({
                "vehicle_id": idx + 1,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "confidence": round(float(det["confidence"]), 4),
                "class_name": str(det.get("class_name", "car")),
                "roi": det.get("roi"),
            })

        return {
            "image_width": orig_w,
            "image_height": orig_h,
            "detections": formatted_detections,
            "inference_time_ms": det_result.get("inference_time_ms", 0.0),
        }
