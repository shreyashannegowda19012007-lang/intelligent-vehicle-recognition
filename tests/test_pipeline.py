"""
Automated unit tests for End-to-End Vehicle Recognition Pipeline (Detection + Classification).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.classification.inference import VehicleClassifier
from src.classification.models.backbones import build_classification_model
from src.classification.pipeline import VehicleRecognitionPipeline
from src.detection.detector import VehicleDetector


def test_pipeline_integration():
    # Construct lightweight detector and classifier
    det = VehicleDetector(model_path="yolov8n.pt", device="cpu")
    cls_model = build_classification_model("resnet18", num_classes=196, pretrained=False)
    clf = VehicleClassifier(model=cls_model, device="cpu")

    pipeline = VehicleRecognitionPipeline(detector=det, classifier=clf)

    # Synthetic image
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    result = pipeline.process_image(img)

    assert "image_width" in result
    assert "image_height" in result
    assert "num_vehicles_detected" in result
    assert "detections" in result
    assert "timing" in result
    assert "detection_ms" in result["timing"]
    assert "classification_ms" in result["timing"]
    assert "total_pipeline_ms" in result["timing"]
