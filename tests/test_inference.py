"""
Automated unit tests for Unified Inference Interface (VehicleClassifier).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src.classification.inference import VehicleClassifier
from src.classification.models.backbones import build_classification_model


@pytest.fixture
def dummy_classifier() -> VehicleClassifier:
    """Fixture providing a persistent VehicleClassifier with lightweight model."""
    model = build_classification_model(
        architecture="resnet18",
        num_classes=196,
        pretrained=False,
    )
    return VehicleClassifier(model=model, device="cpu", top_k=5)


def test_inference_output_schema(dummy_classifier: VehicleClassifier):
    # Test with PIL image
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    res = dummy_classifier.predict(img)

    required_keys = {"predicted_class", "class_id", "make", "model", "year", "confidence", "top_5", "inference_time_ms"}
    assert required_keys.issubset(set(res.keys()))

    assert isinstance(res["predicted_class"], str)
    assert isinstance(res["class_id"], int)
    assert 0 <= res["class_id"] < 196
    assert isinstance(res["make"], str)
    assert isinstance(res["model"], str)
    assert isinstance(res["year"], int)
    assert isinstance(res["confidence"], float)
    assert 0.0 <= res["confidence"] <= 1.0
    assert isinstance(res["inference_time_ms"], float)
    assert res["inference_time_ms"] >= 0.0

    # Verify Top-5 predictions
    assert len(res["top_5"]) == 5
    for item in res["top_5"]:
        assert "rank" in item
        assert "class_id" in item
        assert "class" in item
        assert "make" in item
        assert "model" in item
        assert "year" in item
        assert "confidence" in item
        assert 0.0 <= item["confidence"] <= 1.0

    # Verify descending confidence
    confs = [item["confidence"] for item in res["top_5"]]
    assert confs == sorted(confs, reverse=True)


def test_inference_numpy_and_tensor_inputs(dummy_classifier: VehicleClassifier):
    # Test numpy array (H, W, 3) uint8
    arr = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    res_np = dummy_classifier.predict(arr)
    assert 0 <= res_np["class_id"] < 196

    # Test tensor input (3, H, W) float
    t = torch.randn(3, 224, 224)
    res_t = dummy_classifier.predict(t)
    assert 0 <= res_t["class_id"] < 196


def test_inference_graceful_handling_invalid_inputs(dummy_classifier: VehicleClassifier):
    # Empty numpy array
    empty_arr = np.array([])
    res_empty = dummy_classifier.predict(empty_arr)
    assert res_empty["confidence"] == 0.0
    assert res_empty["predicted_class"] == "Unknown"
    assert res_empty["class_id"] == -1
    assert "error" in res_empty

    # Non-existent file
    res_nofile = dummy_classifier.predict("non_existent/path/car_does_not_exist.jpg")
    assert res_nofile["confidence"] == 0.0
    assert res_nofile["predicted_class"] == "Unknown"
    assert res_nofile["class_id"] == -1

    # Empty 0-element tensor
    res_empty_t = dummy_classifier.predict(torch.zeros(0))
    assert res_empty_t["confidence"] == 0.0
    assert res_empty_t["class_id"] == -1


def test_inference_advanced_tensor_shapes_and_formats(dummy_classifier: VehicleClassifier):
    # 1. Channels-last 3D tensor (H, W, 3)
    t_cl = torch.randn(224, 224, 3)
    r1 = dummy_classifier.predict(t_cl)
    assert 0 <= r1["class_id"] < 196

    # 2. 2D grayscale tensor (H, W)
    t_gray = torch.randn(180, 180)
    r2 = dummy_classifier.predict(t_gray)
    assert 0 <= r2["class_id"] < 196

    # 3. 4D channels-last tensor (1, H, W, 3)
    t_4d_cl = torch.randn(1, 224, 224, 3)
    r3 = dummy_classifier.predict(t_4d_cl)
    assert 0 <= r3["class_id"] < 196

    # 4. Unnormalized uint8 tensor in [0, 255]
    t_uint8 = torch.randint(0, 256, (3, 200, 200), dtype=torch.uint8)
    r4 = dummy_classifier.predict(t_uint8)
    assert 0 <= r4["class_id"] < 196

    # 5. Non-square aspect ratio tensor (3, 100, 320)
    t_banner = torch.randn(3, 100, 320)
    r5 = dummy_classifier.predict(t_banner)
    assert 0 <= r5["class_id"] < 196


def test_inference_batch(dummy_classifier: VehicleClassifier):
    imgs = [
        Image.new("RGB", (64, 64), color=(255, 0, 0)),
        Image.new("RGB", (64, 64), color=(0, 255, 0)),
    ]
    results = dummy_classifier.predict_batch(imgs)
    assert len(results) == 2
    for r in results:
        assert 0 <= r["class_id"] < 196
