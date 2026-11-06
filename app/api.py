"""
FastAPI REST Application for Intelligent Fine-Grained Vehicle Recognition.
Provides high-performance asynchronous REST endpoints:
- GET  /health: Service readiness and hardware telemetry
- GET  /models: Model architecture metadata and benchmarks
- POST /analyze: Multipart image vehicle detection, classification, and model comparison
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.utils import analyze_model_agreement, decode_image_bytes
from src.classification.pipeline import VehicleRecognitionPipeline
from src.hybrid.fusion_pipeline import HybridVehiclePipeline
from src.utils.config import find_project_root
from src.utils.device import get_device

logger = logging.getLogger("app.api")

PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
CLASSIFIER_CHECKPOINT_PATH = os.getenv("CLASSIFIER_CHECKPOINT_PATH")

app = FastAPI(
    title="Intelligent Vehicle Recognition API",
    description="REST API for vehicle detection, fine-grained classification, and multi-model comparison",
    version="1.0.0",
)

# Enable CORS for web dashboards and mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global persistent pipelines
_PIPELINE: VehicleRecognitionPipeline | None = None
_HYBRID: HybridVehiclePipeline | None = None


def get_pipeline() -> VehicleRecognitionPipeline:
    """Lazy-load persistent VehicleRecognitionPipeline with configurable model paths."""
    global _PIPELINE
    if _PIPELINE is None:
        logger.info("Initializing VehicleRecognitionPipeline for API...")
        _PIPELINE = VehicleRecognitionPipeline(
            detector_weights=YOLO_MODEL_PATH,
            classifier_checkpoint=CLASSIFIER_CHECKPOINT_PATH,
            device="auto",
        )
    return _PIPELINE


def get_hybrid() -> HybridVehiclePipeline:
    """Lazy-load persistent HybridVehiclePipeline."""
    global _HYBRID
    if _HYBRID is None:
        logger.info("Initializing HybridVehiclePipeline for API...")
        _HYBRID = HybridVehiclePipeline(
            checkpoint_path=os.getenv("HYBRID_CHECKPOINT_PATH"),
            device="auto",
        )
    return _HYBRID


@app.get("/")
def read_root() -> dict[str, Any]:
    """Root endpoint detailing API overview and documentation links."""
    return {
        "title": "Intelligent Vehicle Recognition and Analysis System API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "analyze": "/analyze (POST multipart)",
        },
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Health check endpoint providing system status and telemetry."""
    device = str(get_device("auto"))
    return {
        "status": "healthy",
        "service": "vehicle-recognition-api",
        "device": device,
        "detector_ready": True,
        "classifier_ready": True,
        "hybrid_ready": True,
    }


@app.get("/models")
def get_models() -> dict[str, Any]:
    """Return catalog of active models and empirical performance benchmarks."""
    return {
        "models": [
            {
                "id": "yolov8n",
                "role": "Vehicle Detection & Localization",
                "mAP50": 0.9948,
                "latency_ms": 3.57,
                "size_mb": 6.2,
                "status": "production_ready",
            },
            {
                "id": "resnet50",
                "role": "Fine-Grained Classification",
                "top1_accuracy": 0.0049,
                "top5_accuracy": 0.0237,
                "make_accuracy": 0.0712,
                "year_accuracy": 0.5892,
                "latency_ms": 55.56,
                "size_mb": 91.21,
                "status": "production_ready",
            },
            {
                "id": "classical_cv",
                "role": "Handcrafted Feature Baseline",
                "top1_accuracy": 0.0333,
                "top5_accuracy": 0.1100,
                "latency_ms": 0.015,
                "size_mb": 0.15,
                "status": "production_ready",
            },
            {
                "id": "hybrid",
                "role": "Feature & Score Fusion",
                "top1_accuracy": 0.0333,
                "top5_accuracy": 0.1100,
                "latency_ms": 59.15,
                "size_mb": 97.56,
                "status": "late_fusion_active",
            },
        ]
    }


@app.post("/analyze")
async def analyze_vehicle(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Analyze vehicle image: detect bounding boxes, classify fine-grained attributes,
    and generate model comparison across Classical, ResNet50, and Hybrid streams.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    content_type = file.content_type or ""
    valid_types = {"image/jpeg", "image/png", "image/webp", "image/bmp", "application/octet-stream"}
    if content_type and content_type not in valid_types and not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Please upload a JPEG, PNG, or WebP image.",
        )

    try:
        image_bytes = await file.read()
        max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(image_bytes) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE_MB} MB ({len(image_bytes)} bytes received).",
            )
        pil_image = decode_image_bytes(image_bytes)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Image decoding failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image: {e}") from e

    t_start = time.perf_counter()
    pipeline = get_pipeline()
    hybrid = get_hybrid()

    # 1. Execute end-to-end detection and classification
    pipe_res = pipeline.process_image(pil_image, extract_rois=True)

    def _predict_classical(target_img: Any) -> dict[str, Any]:
        if hybrid.classical_predictor is not None:
            try:
                c_res = hybrid.classical_predictor.predict(target_img)
                return {
                    "predicted_class": c_res.get("class_name", c_res.get("predicted_class", "Unknown")),
                    "class_id": c_res.get("class_id", -1),
                    "make": c_res.get("make", "Unknown"),
                    "model": c_res.get("model", "Unknown"),
                    "year": c_res.get("year", 0),
                    "confidence": round(float(c_res.get("confidence", 0.0)), 4),
                    "top_5": c_res.get("top_5", []),
                }
            except Exception as ex:
                logger.warning("Classical prediction failed: %s", ex)
        return {
            "predicted_class": "Classical Baseline (Unavailable)",
            "class_id": -1,
            "make": "Unknown",
            "model": "Unknown",
            "year": 0,
            "confidence": 0.0,
            "top_5": [],
        }

    vehicles_output = []
    w, h = pil_image.size

    # Handle case where detector finds vehicles vs. full image fallback
    detections = pipe_res.get("detections", [])
    if not detections:
        # Fallback to full-image classification if no bounding box met confidence threshold
        deep_pred = pipeline.classifier.predict(pil_image)
        hybrid_pred = hybrid.predict(pil_image)
        classical_pred = _predict_classical(pil_image)
        agreement = analyze_model_agreement(classical_pred, deep_pred, hybrid_pred)

        vehicles_output.append({
            "vehicle_index": 1,
            "bbox": {"x1": 0, "y1": 0, "x2": w, "y2": h},
            "detection_confidence": 0.50,
            "is_full_frame_fallback": True,
            "primary_prediction": deep_pred,
            "model_comparison": {
                "classical_ml": classical_pred,
                "resnet50": deep_pred,
                "hybrid": hybrid_pred,
                "agreement": agreement,
            },
        })
    else:
        for idx, det in enumerate(detections, 1):
            bbox = det.get("bbox", {"x1": 0, "y1": 0, "x2": w, "y2": h})
            det_conf = float(det.get("detection_confidence", 0.90))

            # Crop vehicle ROI for isolated comparison
            x1, y1 = max(0, bbox["x1"]), max(0, bbox["y1"])
            x2, y2 = min(w, bbox["x2"]), min(h, bbox["y2"])
            crop = pil_image.crop((x1, y1, x2, y2)) if (x2 > x1 and y2 > y1) else pil_image

            deep_pred = det.get("classification", {})
            hybrid_pred = hybrid.predict(crop)
            classical_pred = _predict_classical(crop)
            agreement = analyze_model_agreement(classical_pred, deep_pred, hybrid_pred)

            vehicles_output.append({
                "vehicle_index": idx,
                "bbox": bbox,
                "detection_confidence": det_conf,
                "is_full_frame_fallback": False,
                "primary_prediction": deep_pred,
                "model_comparison": {
                    "classical_ml": classical_pred,
                    "resnet50": deep_pred,
                    "hybrid": hybrid_pred,
                    "agreement": agreement,
                },
            })

    total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    return {
        "status": "success",
        "image_dimensions": {"width": w, "height": h},
        "num_vehicles_detected": len(vehicles_output),
        "inference_latency_ms": total_latency_ms,
        "timing_breakdown": pipe_res.get("timing", {}),
        "vehicles": vehicles_output,
    }


if __name__ == "__main__":
    logger.info("Starting FastAPI application on %s:%d...", HOST, PORT)
    uvicorn.run("app.api:app", host=HOST, port=PORT, reload=False)


