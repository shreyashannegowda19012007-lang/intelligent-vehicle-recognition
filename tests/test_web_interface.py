"""
Comprehensive Unit Tests for Web Application Interface (Streamlit & FastAPI).
Verifies image decoding, bounding-box rendering, model agreement heuristics,
FastAPI REST endpoints (/health, /models, /analyze), and error responses.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import app
from app.utils import (
    analyze_model_agreement,
    decode_image_bytes,
    draw_bounding_boxes,
)


@pytest.fixture
def test_client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generate in-memory valid JPEG image bytes."""
    img = Image.new("RGB", (320, 240), color=(120, 150, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_decode_image_bytes_valid(sample_image_bytes: bytes):
    """Verify decoding valid image bytes to RGB PIL Image."""
    img = decode_image_bytes(sample_image_bytes)
    assert isinstance(img, Image.Image)
    assert img.size == (320, 240)
    assert img.mode == "RGB"


def test_decode_image_bytes_invalid():
    """Verify graceful error on invalid or empty image bytes."""
    with pytest.raises(ValueError, match="empty image byte payload"):
        decode_image_bytes(b"")

    with pytest.raises(ValueError, match="Failed to decode image"):
        decode_image_bytes(b"corrupt-non-image-data-payload")


def test_draw_bounding_boxes(sample_image_bytes: bytes):
    """Verify drawing bounding boxes on PIL image."""
    img = decode_image_bytes(sample_image_bytes)
    detections = [
        {
            "bbox": {"x1": 20, "y1": 30, "x2": 150, "y2": 180},
            "detection_confidence": 0.95,
            "classification": {
                "predicted_class": "Audi S4 Sedan 2012",
            },
        }
    ]
    annotated = draw_bounding_boxes(img, detections)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == img.size
    # Verify image array modified by drawing
    assert not np.array_equal(np.array(img), np.array(annotated))


def test_analyze_model_agreement():
    """Verify consensus and disagreement logic across models."""
    # Scenario 1: Full consensus
    pred_c1 = {"make": "Audi", "model": "S4", "predicted_class": "Audi S4 Sedan 2012"}
    pred_r1 = {"make": "Audi", "model": "S4", "predicted_class": "Audi S4 Sedan 2012"}
    pred_h1 = {"make": "Audi", "model": "S4", "predicted_class": "Audi S4 Sedan 2012"}
    res1 = analyze_model_agreement(pred_c1, pred_r1, pred_h1)
    assert res1["status"] == "Full Consensus"
    assert res1["all_agree_class"] is True

    # Scenario 2: Make consensus with model/trim divergence
    pred_c2 = {"make": "Audi", "model": "A4", "predicted_class": "Audi A4 Sedan 2012"}
    pred_r2 = {"make": "Audi", "model": "S4", "predicted_class": "Audi S4 Sedan 2012"}
    pred_h2 = {"make": "Audi", "model": "RS 4", "predicted_class": "Audi RS 4 Convertible 2008"}
    res2 = analyze_model_agreement(pred_c2, pred_r2, pred_h2)
    assert res2["status"] == "Make Consensus"
    assert res2["all_agree_make"] is True
    assert res2["all_agree_class"] is False

    # Scenario 3: Complete Disagreement
    pred_c3 = {"make": "Ford", "model": "Expedition", "predicted_class": "Ford Expedition 2009"}
    pred_r3 = {"make": "Audi", "model": "S4", "predicted_class": "Audi S4 Sedan 2012"}
    pred_h3 = {"make": "BMW", "model": "M3", "predicted_class": "BMW M3 Coupe 2012"}
    res3 = analyze_model_agreement(pred_c3, pred_r3, pred_h3)
    assert res3["status"] == "Disagreement"


def test_api_health_endpoint(test_client: TestClient):
    """Verify /health endpoint returns HTTP 200 and healthy status."""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert data["detector_ready"] is True


def test_api_models_endpoint(test_client: TestClient):
    """Verify /models endpoint returns active models catalog."""
    resp = test_client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) >= 4
    model_ids = [m["id"] for m in data["models"]]
    assert "yolov8n" in model_ids
    assert "resnet50" in model_ids
    assert "classical_cv" in model_ids
    assert "hybrid" in model_ids


def test_api_analyze_endpoint_valid_image(test_client: TestClient, sample_image_bytes: bytes):
    """Verify /analyze endpoint processes image upload and returns valid response structure."""
    files = {"file": ("test_car.jpg", sample_image_bytes, "image/jpeg")}
    resp = test_client.post("/analyze", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "num_vehicles_detected" in data
    assert "inference_latency_ms" in data
    assert "vehicles" in data
    assert len(data["vehicles"]) >= 1

    first_vehicle = data["vehicles"][0]
    assert "bbox" in first_vehicle
    assert "detection_confidence" in first_vehicle
    assert "primary_prediction" in first_vehicle
    assert "model_comparison" in first_vehicle

    comparison = first_vehicle["model_comparison"]
    assert "classical_ml" in comparison
    assert "resnet50" in comparison
    assert "hybrid" in comparison
    assert "agreement" in comparison


def test_api_analyze_endpoint_invalid_file(test_client: TestClient):
    """Verify /analyze endpoint gracefully rejects corrupt or unsupported payloads."""
    # Corrupt image file
    files = {"file": ("bad_car.jpg", b"not-a-valid-image", "image/jpeg")}
    resp = test_client.post("/analyze", files=files)
    assert resp.status_code == 400
    assert "Invalid or corrupted image" in resp.json()["detail"]

