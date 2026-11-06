"""
Automated Unit and Integration Tests for Hybrid Intelligence Vehicle Recognition Pipeline (Phase 6).
Validates:
1. Image validation (corrupted, empty, grayscale, RGBA, non-existent, float)
2. YOLO output adaptation & bounding-box coordinate restoration
3. ROI extraction & boundary clamping
4. Classical classifier integration & calibration status
5. Deep-learning classifier integration & Top-5 extraction
6. Hybrid Decision Logic: Strategy A (DL Primary)
7. Hybrid Decision Logic: Strategy B (Agreement & Disagreement handling)
8. Hybrid Decision Logic: Strategy C (Configurable Score Fusion & Uncalibrated Fallback)
9. Missing classifier artifacts handling (graceful degradation)
10. Multiple vehicle processing in a single scene
11. Zero vehicle detection handling
12. End-to-end pipeline response schema compliance
13. Vehicle failure isolation (single vehicle failure does not crash pipeline)
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.hybrid.adapters import (
    BaseClassifierAdapter,
    ClassicalAdapter,
    DeepLearningAdapter,
    DetectorAdapter,
)
from src.hybrid.config import DecisionStrategy, HybridConfig
from src.hybrid.decision_engine import HybridDecisionEngine
from src.hybrid.pipeline import HybridVehiclePipeline


# =============================================================================
# Mock Adapters for Fast, Isolated Unit Testing
# =============================================================================

class MockDetectorAdapter:
    """Mock detector returning configurable vehicle detections."""

    def __init__(self, detections: list[dict] | None = None) -> None:
        self.mock_detections = detections if detections is not None else [
            {
                "vehicle_id": 1,
                "bbox": {"x1": 50, "y1": 60, "x2": 250, "y2": 200},
                "confidence": 0.95,
                "class_name": "car",
                "roi": np.ones((140, 200, 3), dtype=np.uint8) * 100,
            }
        ]

    def detect_vehicles(self, image_input, conf=None, iou=None):
        return {
            "image_width": 640,
            "image_height": 480,
            "detections": self.mock_detections,
            "inference_time_ms": 12.5,
        }


class MockDLAdapter(BaseClassifierAdapter):
    """Mock deep learning classifier adapter."""

    name = "deep_learning"

    def __init__(
        self,
        class_id: int = 42,
        class_name: str | None = None,
        make: str = "BMW",
        model: str = "3 Series",
        year: int = 2018,
        confidence: float = 0.92,
        is_available: bool = True,
        should_raise: bool = False,
    ) -> None:
        self.is_available = is_available
        self.class_id = class_id
        self.make = make
        self.model = model
        self.year = year
        self.class_name = class_name or f"{make} {model} {year}"
        self.confidence = confidence
        self.should_raise = should_raise

    def predict_crop(self, crop):
        if self.should_raise:
            raise RuntimeError("Forced DL model forward error")
        if not self.is_available:
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
                "warning": "DL checkpoint missing",
                "error": "Checkpoint not found",
            }

        return {
            "status": "success",
            "class_id": self.class_id,
            "predicted_class": self.class_name,
            "make": self.make,
            "model": self.model,
            "year": self.year,
            "confidence": self.confidence,
            "score": self.confidence,
            "is_calibrated": True,
            "is_uncertain": self.confidence < 0.20,
            "top_5": [
                {"rank": 1, "class_id": self.class_id, "class": self.class_name, "make": self.make, "model": self.model, "year": self.year, "confidence": self.confidence},
                {"rank": 2, "class_id": 10, "class": "Audi A4 2015", "make": "Audi", "model": "A4", "year": 2015, "confidence": 0.05},
            ],
            "inference_time_ms": 15.2,
            "warning": None,
            "error": None,
        }


class MockClassicalAdapter(BaseClassifierAdapter):
    """Mock classical CV classifier adapter."""

    name = "classical_cv"

    def __init__(
        self,
        class_id: int = 42,
        class_name: str | None = None,
        make: str = "BMW",
        model: str = "3 Series",
        year: int = 2018,
        score: float = 0.81,
        is_calibrated: bool = False,
        is_available: bool = True,
        should_raise: bool = False,
    ) -> None:
        self.is_available = is_available
        self.class_id = class_id
        self.make = make
        self.model = model
        self.year = year
        self.class_name = class_name or f"{make} {model} {year}"
        self.score = score
        self.is_calibrated = is_calibrated
        self.should_raise = should_raise

    def predict_crop(self, crop):
        if self.should_raise:
            raise RuntimeError("Forced classical feature extraction error")
        if not self.is_available:
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
                "warning": "Classical checkpoint missing",
                "error": "Model not found",
            }

        return {
            "status": "success",
            "class_id": self.class_id,
            "predicted_class": self.class_name,
            "make": self.make,
            "model": self.model,
            "year": self.year,
            "confidence": self.score if self.is_calibrated else 0.0,
            "score": self.score,
            "is_calibrated": self.is_calibrated,
            "is_uncertain": self.score < 0.30,
            "top_5": [
                {"rank": 1, "class_id": self.class_id, "class": self.class_name, "score": self.score},
            ],
            "inference_time_ms": 8.4,
            "warning": "Uncalibrated SVM decision score" if not self.is_calibrated else None,
            "error": None,
        }


# =============================================================================
# Unit Tests
# =============================================================================

def test_image_validation_valid_and_invalid_inputs():
    pipeline = HybridVehiclePipeline(
        detector=MockDetectorAdapter(),
        dl_classifier=MockDLAdapter(),
        classical_predictor=MockClassicalAdapter(),
    )

    # 1. Valid PIL Image
    pil_img = Image.new("RGB", (320, 240), color=(120, 150, 180))
    arr, w, h, err = pipeline.validate_image(pil_img)
    assert err is None
    assert w == 320 and h == 240
    assert arr.shape == (240, 320, 3)

    # 2. Valid Numpy uint8 RGB
    np_img = np.zeros((100, 150, 3), dtype=np.uint8)
    arr, w, h, err = pipeline.validate_image(np_img)
    assert err is None
    assert w == 150 and h == 100

    # 3. Grayscale (H, W) -> converted to RGB (H, W, 3)
    gray = np.zeros((80, 80), dtype=np.uint8)
    arr, w, h, err = pipeline.validate_image(gray)
    assert err is None
    assert arr.shape == (80, 80, 3)

    # 4. RGBA (H, W, 4) -> converted to RGB (H, W, 3)
    rgba = np.zeros((50, 50, 4), dtype=np.uint8)
    arr, w, h, err = pipeline.validate_image(rgba)
    assert err is None
    assert arr.shape == (50, 50, 3)

    # 5. Non-existent path
    _, _, _, err = pipeline.validate_image("non_existent_car_image_path.jpg")
    assert err is not None
    assert "not found" in err.lower()

    # 6. None input
    _, _, _, err = pipeline.validate_image(None)
    assert err is not None

    # 7. Empty array
    empty = np.array([])
    _, _, _, err = pipeline.validate_image(empty)
    assert err is not None


def test_detector_output_adaptation():
    det = MockDetectorAdapter([
        {
            "vehicle_id": 1,
            "bbox": {"x1": 10.4, "y1": 20.6, "x2": 300.2, "y2": 400.8},
            "confidence": 0.88,
            "class_name": "car",
            "roi": np.zeros((100, 100, 3), dtype=np.uint8),
        }
    ])

    adapter = DetectorAdapter(detector=det)
    res = adapter.detect_vehicles(np.zeros((480, 640, 3), dtype=np.uint8))

    assert res["image_width"] == 640
    assert res["image_height"] == 480
    assert len(res["detections"]) == 1

    d = res["detections"][0]
    # Coordinates must be converted to rounded integers clamped inside original image
    assert isinstance(d["bbox"]["x1"], int)
    assert isinstance(d["bbox"]["y1"], int)
    assert isinstance(d["bbox"]["x2"], int)
    assert isinstance(d["bbox"]["y2"], int)
    assert d["bbox"]["x1"] == 10
    assert d["bbox"]["y1"] == 21
    assert d["bbox"]["x2"] == 300
    assert d["bbox"]["y2"] == 401
    assert d["confidence"] == 0.88


def test_classical_adapter_uncalibrated_score_handling():
    # Uncalibrated classical adapter should report is_calibrated=False and not fabricate probability
    cl_adapter = MockClassicalAdapter(score=0.75, is_calibrated=False)
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    res = cl_adapter.predict_crop(crop)

    assert res["status"] == "success"
    assert res["is_calibrated"] is False
    assert res["confidence"] == 0.0  # Must not treat SVM decision score as probability
    assert res["score"] == 0.75
    assert res["warning"] is not None


def test_hybrid_decision_strategy_a_deep_learning_primary():
    engine = HybridDecisionEngine()

    dl = MockDLAdapter(make="Audi", model="A4", year=2015, class_id=10, confidence=0.91).predict_crop(None)
    cl = MockClassicalAdapter(make="BMW", model="3 Series", year=2018, class_id=42, score=0.60).predict_crop(None)

    res = engine.decide(
        vehicle_id=1,
        bbox={"x1": 10, "y1": 10, "x2": 100, "y2": 100},
        detection_info={"class_name": "car", "confidence": 0.95},
        dl_pred=dl,
        classical_pred=cl,
        strategy_override=DecisionStrategy.DEEP_LEARNING_PRIMARY,
    )

    assert res["agreement"] is False
    assert res["final_prediction"]["make"] == "Audi"
    assert res["final_prediction"]["model"] == "A4"
    assert res["final_prediction"]["confidence"] == 0.91
    assert res["decision_strategy"] == "deep_learning_primary"
    assert any("Model disagreement" in w for w in res["warnings"])


def test_hybrid_decision_strategy_b_agreement():
    engine = HybridDecisionEngine()

    # Both models predict the exact same vehicle
    dl = MockDLAdapter(make="BMW", model="3 Series", year=2018, class_id=42, confidence=0.85).predict_crop(None)
    cl = MockClassicalAdapter(make="BMW", model="3 Series", year=2018, class_id=42, score=0.80, is_calibrated=True).predict_crop(None)

    res = engine.decide(
        vehicle_id=1,
        bbox={"x1": 10, "y1": 10, "x2": 100, "y2": 100},
        detection_info={"class_name": "car", "confidence": 0.95},
        dl_pred=dl,
        classical_pred=cl,
        strategy_override=DecisionStrategy.AGREEMENT_BASED,
    )

    assert res["agreement"] is True
    assert res["agreement_details"]["class_match"] is True
    assert res["agreement_details"]["make_match"] is True
    assert res["final_prediction"]["class_id"] == 42
    # Agreement with calibrated classical model yields consensus boost
    assert res["final_prediction"]["confidence"] > 0.85


def test_hybrid_decision_strategy_b_disagreement_no_invented_score():
    engine = HybridDecisionEngine()

    # Models disagree
    dl = MockDLAdapter(make="Tesla", model="Model S", year=2012, class_id=184, confidence=0.88).predict_crop(None)
    cl = MockClassicalAdapter(make="Audi", model="A5", year=2012, class_id=12, score=0.70).predict_crop(None)

    res = engine.decide(
        vehicle_id=1,
        bbox={"x1": 10, "y1": 10, "x2": 100, "y2": 100},
        detection_info={"class_name": "car", "confidence": 0.95},
        dl_pred=dl,
        classical_pred=cl,
        strategy_override=DecisionStrategy.AGREEMENT_BASED,
    )

    assert res["agreement"] is False
    assert res["final_prediction"]["make"] == "Tesla"
    # In disagreement, confidence must remain strictly the primary model's confidence (no fabricated weighted score)
    assert res["final_prediction"]["confidence"] == 0.88
    assert any("Model disagreement" in w for w in res["warnings"])


def test_hybrid_decision_strategy_c_score_fusion_calibrated_and_uncalibrated():
    engine = HybridDecisionEngine()

    # Case 1: Uncalibrated classical model -> should fall back safely without inventing probability
    dl = MockDLAdapter(class_id=1, confidence=0.90).predict_crop(None)
    cl_uncal = MockClassicalAdapter(class_id=2, score=0.80, is_calibrated=False).predict_crop(None)

    res_uncal = engine.decide(
        vehicle_id=1,
        bbox={"x1": 0, "y1": 0, "x2": 50, "y2": 50},
        detection_info={"class_name": "car", "confidence": 0.9},
        dl_pred=dl,
        classical_pred=cl_uncal,
        strategy_override=DecisionStrategy.SCORE_FUSION,
    )
    assert res_uncal["final_prediction"]["class_id"] == 1
    assert any("uncalibrated" in w.lower() for w in res_uncal["warnings"])

    # Case 2: Calibrated classical model -> correctly applies weighted fusion
    cl_cal = MockClassicalAdapter(class_id=1, score=0.80, is_calibrated=True).predict_crop(None)
    res_cal = engine.decide(
        vehicle_id=1,
        bbox={"x1": 0, "y1": 0, "x2": 50, "y2": 50},
        detection_info={"class_name": "car", "confidence": 0.9},
        dl_pred=dl,
        classical_pred=cl_cal,
        strategy_override=DecisionStrategy.SCORE_FUSION,
    )
    # Both predict class 1: w_dl=0.7, s_dl=0.9; w_cl=0.3, s_cl=0.8 -> 0.7*0.9 + 0.3*0.8 = 0.63 + 0.24 = 0.87
    assert res_cal["final_prediction"]["confidence"] == pytest.approx(0.87, abs=1e-3)
    assert res_cal["final_prediction"]["class_id"] == 1

    # Case 3: Calibrated classical model with DISAGREEMENT -> selects winning candidate with normalized relative confidence
    cl_disagree = MockClassicalAdapter(class_id=2, make="Audi", model="A4", year=2015, score=0.80, is_calibrated=True).predict_crop(None)
    res_disagree = engine.decide(
        vehicle_id=1,
        bbox={"x1": 0, "y1": 0, "x2": 50, "y2": 50},
        detection_info={"class_name": "car", "confidence": 0.9},
        dl_pred=dl,
        classical_pred=cl_disagree,
        strategy_override=DecisionStrategy.SCORE_FUSION,
    )
    # w_dl*s_dl = 0.63, w_cl*s_cl = 0.24 -> class 1 wins; normalized conf = 0.63 / (0.63 + 0.24) = 0.7241
    assert res_disagree["final_prediction"]["class_id"] == 1
    assert res_disagree["final_prediction"]["confidence"] == pytest.approx(0.7241, abs=1e-3)


def test_missing_classifier_artifacts_graceful_handling():
    # Pipeline with missing classical model
    pipeline_no_cl = HybridVehiclePipeline(
        detector=MockDetectorAdapter(),
        dl_classifier=MockDLAdapter(make="Ford", model="Mustang", year=2014, class_id=50, confidence=0.89),
        classical_predictor=MockClassicalAdapter(is_available=False),
    )

    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    res = pipeline_no_cl.process_image(test_img)

    assert res["status"] == "success"
    assert res["vehicle_count"] == 1
    v = res["vehicles"][0]
    assert v["final_prediction"]["make"] == "Ford"
    assert v["classical_prediction"]["status"] == "unavailable"
    assert any("Classical classifier unavailable" in w for w in v["warnings"])


def test_multiple_vehicle_processing():
    two_dets = [
        {
            "vehicle_id": 1,
            "bbox": {"x1": 20, "y1": 30, "x2": 200, "y2": 180},
            "confidence": 0.94,
            "class_name": "car",
            "roi": np.zeros((150, 180, 3), dtype=np.uint8),
        },
        {
            "vehicle_id": 2,
            "bbox": {"x1": 250, "y1": 40, "x2": 450, "y2": 220},
            "confidence": 0.89,
            "class_name": "car",
            "roi": np.zeros((180, 200, 3), dtype=np.uint8),
        },
    ]

    pipeline = HybridVehiclePipeline(
        detector=MockDetectorAdapter(two_dets),
        dl_classifier=MockDLAdapter(),
        classical_predictor=MockClassicalAdapter(),
    )

    res = pipeline.process_image(np.zeros((480, 640, 3), dtype=np.uint8))

    assert res["status"] == "success"
    assert res["vehicle_count"] == 2
    assert len(res["vehicles"]) == 2
    assert res["vehicles"][0]["vehicle_id"] == 1
    assert res["vehicles"][1]["vehicle_id"] == 2
    assert res["vehicles"][0]["bbox"]["x1"] == 20
    assert res["vehicles"][1]["bbox"]["x1"] == 250


def test_zero_vehicles_detected_scene():
    pipeline = HybridVehiclePipeline(
        detector=MockDetectorAdapter([]),  # 0 detections
        dl_classifier=MockDLAdapter(),
        classical_predictor=MockClassicalAdapter(),
    )

    res = pipeline.process_image(np.zeros((480, 640, 3), dtype=np.uint8))

    assert res["status"] == "success"
    assert res["vehicle_count"] == 0
    assert res["vehicles"] == []
    assert any("No vehicles detected" in w for w in res["warnings"])


def test_complete_pipeline_output_schema():
    pipeline = HybridVehiclePipeline(
        detector=MockDetectorAdapter(),
        dl_classifier=MockDLAdapter(),
        classical_predictor=MockClassicalAdapter(),
    )

    img = Image.new("RGB", (640, 480), color=(100, 120, 140))
    res = pipeline.process_image(img)

    # Required root keys
    root_keys = {"status", "image", "vehicle_count", "vehicles", "performance", "warnings"}
    assert root_keys.issubset(set(res.keys()))

    # Image metadata
    assert res["image"]["width"] == 640
    assert res["image"]["height"] == 480

    # Performance breakdown
    perf = res["performance"]
    perf_keys = {
        "detection_time_ms",
        "classical_time_ms",
        "deep_learning_time_ms",
        "classification_time_ms",
        "decision_time_ms",
        "total_time_ms",
    }
    assert perf_keys.issubset(set(perf.keys()))
    assert perf["total_time_ms"] >= 0.0

    # Vehicle schema
    v = res["vehicles"][0]
    v_keys = {
        "vehicle_id",
        "bbox",
        "detection",
        "classical_prediction",
        "deep_learning_prediction",
        "agreement",
        "agreement_details",
        "final_prediction",
        "decision_strategy",
        "decision_latency_ms",
        "warnings",
    }
    assert v_keys.issubset(set(v.keys()))

    assert isinstance(v["bbox"], dict)
    assert {"x1", "y1", "x2", "y2"}.issubset(set(v["bbox"].keys()))
    assert isinstance(v["final_prediction"], dict)
    assert {"make", "model", "year"}.issubset(set(v["final_prediction"].keys()))


def test_vehicle_failure_isolation():
    # If DL forward raises an exception, the pipeline should NOT crash and should return structured error
    pipeline = HybridVehiclePipeline(
        detector=MockDetectorAdapter(),
        dl_classifier=MockDLAdapter(should_raise=True),
        classical_predictor=MockClassicalAdapter(),
    )

    res = pipeline.process_image(np.zeros((480, 640, 3), dtype=np.uint8))
    assert res["status"] == "success"
    v = res["vehicles"][0]
    # Falls back gracefully to classical prediction
    assert v["final_prediction"]["make"] == "BMW"
    assert v["deep_learning_prediction"]["status"] == "error"


def test_pipeline_batch_processing():
    pipeline = HybridVehiclePipeline(
        detector=MockDetectorAdapter(),
        dl_classifier=MockDLAdapter(),
        classical_predictor=MockClassicalAdapter(),
    )
    imgs = [
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.ones((120, 120, 3), dtype=np.uint8) * 128,
    ]
    batch_res = pipeline.process_batch(imgs)
    assert len(batch_res) == 2
    for r in batch_res:
        assert r["status"] == "success"
        assert r["vehicle_count"] == 1


def test_detector_failure_isolation():
    class CrashingDetector:
        def detect_vehicles(self, image_input, conf=None, iou=None):
            raise RuntimeError("Fatal CUDA detector crash")

    pipeline = HybridVehiclePipeline(
        detector=CrashingDetector(),
        dl_classifier=MockDLAdapter(),
        classical_predictor=MockClassicalAdapter(),
    )
    res = pipeline.process_image(np.zeros((100, 100, 3), dtype=np.uint8))
    assert res["status"] == "error"
    assert res["vehicle_count"] == 0
    assert any("fatal cuda detector crash" in w.lower() for w in res["warnings"])


def test_detector_adapter_sequence_and_dict_bbox_formats():
    # Sequence [x1, y1, x2, y2]
    det_seq = MockDetectorAdapter([
        {
            "vehicle_id": 1,
            "bbox": [10.2, 20.8, 200.4, 300.9],
            "confidence": 0.90,
            "class_name": "car",
            "roi": None,
        }
    ])
    adapter = DetectorAdapter(detector=det_seq)
    res = adapter.detect_vehicles(np.zeros((480, 640, 3), dtype=np.uint8))
    b = res["detections"][0]["bbox"]
    assert b["x1"] == 10 and b["y1"] == 21 and b["x2"] == 200 and b["y2"] == 301

    # Dictionary with xmin, ymin, xmax, ymax
    det_xmin = MockDetectorAdapter([
        {
            "vehicle_id": 1,
            "bbox": {"xmin": 15, "ymin": 25, "xmax": 215, "ymax": 325},
            "confidence": 0.88,
            "class_name": "car",
            "roi": None,
        }
    ])
    adapter2 = DetectorAdapter(detector=det_xmin)
    res2 = adapter2.detect_vehicles(np.zeros((480, 640, 3), dtype=np.uint8))
    b2 = res2["detections"][0]["bbox"]
    assert b2["x1"] == 15 and b2["y1"] == 25 and b2["x2"] == 215 and b2["y2"] == 325


def test_evaluator_crop_metrics_complete_schema():
    from src.hybrid.evaluator import HybridEvaluator

    pipeline = HybridVehiclePipeline(
        detector=MockDetectorAdapter(),
        dl_classifier=MockDLAdapter(class_id=1, make="Audi", model="A4", year=2015, confidence=0.9),
        classical_predictor=MockClassicalAdapter(class_id=1, make="Audi", model="A4", year=2015, score=0.8, is_calibrated=True),
    )
    evaluator = HybridEvaluator(pipeline=pipeline)
    samples = [
        {"image": np.zeros((100, 100, 3), dtype=np.uint8), "true_class_id": 1},
        {"image": np.zeros((100, 100, 3), dtype=np.uint8), "true_class_id": 2},
    ]
    res = evaluator.evaluate_on_crops(samples)

    # Verify DL metrics schema completeness per Prompt Section 8
    dl_m = res["deep_learning"]
    assert "top1_accuracy" in dl_m
    assert "top5_accuracy" in dl_m
    assert "make_accuracy" in dl_m
    assert "model_accuracy" in dl_m
    assert "year_accuracy" in dl_m
    assert "macro_precision" in dl_m
    assert "weighted_precision" in dl_m
    assert "macro_recall" in dl_m
    assert "weighted_recall" in dl_m
    assert "macro_f1" in dl_m
    assert "weighted_f1" in dl_m
    assert "model_size_mb" in dl_m
    assert "confusion_matrix_top_errors" in dl_m

    # Verify Hybrid strategies contain full schema
    for strat_key, strat_m in res["hybrid_strategies"].items():
        assert "top1_accuracy" in strat_m
        assert "top5_accuracy" in strat_m
        assert "macro_precision" in strat_m
        assert "macro_recall" in strat_m
        assert "confusion_matrix_top_errors" in strat_m


def test_robustness_all_eight_conditions():
    from src.hybrid.robustness import HybridRobustnessEvaluator

    pipeline = HybridVehiclePipeline(
        detector=MockDetectorAdapter(),
        dl_classifier=MockDLAdapter(),
        classical_predictor=MockClassicalAdapter(),
    )
    robustness = HybridRobustnessEvaluator(pipeline=pipeline)
    test_img = np.zeros((120, 160, 3), dtype=np.uint8)
    samples = [{"image": test_img, "true_class_id": 42}]

    metrics = robustness.evaluate_corruptions(samples)

    expected_conditions = {
        "baseline",
        "low_resolution_4x",
        "mild_blur",
        "low_light",
        "overexposed",
        "low_contrast",
        "partial_occlusion",
        "multiple_vehicles",
        "detector_crop_error",
        "visually_similar_models",
    }
    assert expected_conditions.issubset(set(metrics.keys()))
