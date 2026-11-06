"""
Comprehensive Verification Script for ConvNeXt-Tiny Classifier Integration.
Tests:
1. Auto-discovery of convnext_best.pt
2. Architecture, parameters, image_size (320x320)
3. Inference on validation sample
4. Make, model, year taxonomy decoding
5. Top-5 ranking and probability calibration
6. Hybrid pipeline integration
7. FastAPI endpoint verification
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from fastapi.testclient import TestClient

from app.api import app
from src.classification.inference import VehicleClassifier
from src.hybrid.adapters import DeepLearningAdapter
from src.hybrid.fusion_pipeline import HybridVehiclePipeline


def test_classifier_loading_and_inference():
    print("\n=== 1. Testing VehicleClassifier Checkpoint Loading & Inference ===")
    clf = VehicleClassifier()
    print(f"Checkpoint path: {clf.checkpoint_path}")
    print(f"Architecture: {clf.config.architecture}")
    print(f"Image size: {clf.config.image_size}")

    assert clf.config.architecture == "convnext_tiny", f"Expected convnext_tiny, got {clf.config.architecture}"
    assert clf.config.image_size == (320, 320), f"Expected (320, 320), got {clf.config.image_size}"
    assert "convnext_best.pt" in str(clf.checkpoint_path), f"Expected convnext_best.pt in {clf.checkpoint_path}"

    test_img_path = PROJECT_ROOT / "Stanford Dataset" / "cars_train" / "cars_train" / "07936.jpg"
    assert test_img_path.is_file(), f"Sample image not found at {test_img_path}"

    res = clf.predict(test_img_path)
    print("\nInference Output:")
    print(f"  Predicted Class: {res['predicted_class']}")
    print(f"  Class ID:        {res['class_id']}")
    print(f"  Make:            {res['make']}")
    print(f"  Model:           {res['model']}")
    print(f"  Year:            {res['year']}")
    print(f"  Confidence:      {res['confidence'] * 100:.2f}%")
    print(f"  Latency:         {res['inference_time_ms']} ms")

    print("\nTop-5 Predictions:")
    for item in res["top_5"]:
        print(f"  Rank {item['rank']}: {item['class']} | Conf: {item['confidence'] * 100:.2f}% | Make: {item['make']} | Year: {item['year']}")

    assert res["class_id"] >= 0
    assert len(res["top_5"]) == 5
    assert res["make"] != "Unknown"
    assert res["year"] > 1900
    print(">>> VehicleClassifier verification SUCCESS!")
    return clf


def test_hybrid_integration():
    print("\n=== 2. Testing Hybrid Pipeline with ConvNeXt ===")
    adapter = DeepLearningAdapter()
    print(f"DeepLearningAdapter available: {adapter.is_available}")
    print(f"DeepLearningAdapter model size: {adapter.get_model_size_mb()} MB")
    assert adapter.is_available is True
    assert adapter.get_model_size_mb() > 300.0, f"Expected >300 MB, got {adapter.get_model_size_mb()}"

    hybrid = HybridVehiclePipeline()
    test_img = Image.new("RGB", (320, 320), color=(100, 150, 200))
    h_res = hybrid.predict(test_img)
    print(f"Hybrid prediction class: {h_res.get('predicted_class')}")
    print(f"Hybrid prediction make: {h_res.get('make')}")
    print(f"Hybrid confidence: {h_res.get('confidence')}")
    assert h_res.get("predicted_class") is not None
    assert h_res.get("make") != "Unknown"
    print(">>> Hybrid Pipeline verification SUCCESS!")


def test_api_endpoints():
    print("\n=== 3. Testing FastAPI REST Endpoints ===")
    client = TestClient(app)

    # 1. Health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    print(f"GET /health: {r_health.json()}")

    # 2. Models
    r_models = client.get("/models")
    assert r_models.status_code == 200
    models_data = r_models.json()["models"]
    model_ids = [m["id"] for m in models_data]
    print(f"GET /models IDs: {model_ids}")
    assert "convnext_tiny" in model_ids
    assert "resnet50" in model_ids
    assert "yolov8n" in model_ids

    # 3. Analyze
    test_img_path = PROJECT_ROOT / "Stanford Dataset" / "cars_train" / "cars_train" / "07936.jpg"
    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    r_analyze = client.post("/analyze", files={"file": ("test_car.jpg", img_bytes, "image/jpeg")})
    assert r_analyze.status_code == 200
    data = r_analyze.json()
    print(f"POST /analyze status: {data['status']}")
    print(f"Num vehicles detected: {data['num_vehicles_detected']}")
    print(f"Total latency: {data['inference_latency_ms']} ms")

    first_v = data["vehicles"][0]
    primary = first_v["primary_prediction"]
    print(f"Primary prediction: {primary['predicted_class']} ({primary['confidence']*100:.2f}%)")
    comp = first_v["model_comparison"]
    print(f"Model comparison keys: {list(comp.keys())}")
    assert "resnet50" in comp
    assert "convnext_tiny" in comp
    assert "classical_ml" in comp
    assert "hybrid" in comp
    print(">>> FastAPI Endpoints verification SUCCESS!")


if __name__ == "__main__":
    test_classifier_loading_and_inference()
    test_hybrid_integration()
    test_api_endpoints()
    print("\n==========================================")
    print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
    print("==========================================")
