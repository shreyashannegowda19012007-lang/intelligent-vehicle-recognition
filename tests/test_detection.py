"""
Unit tests for YOLO Vehicle Detection Pipeline (Phase 4).
Validates configuration loading, checkpoint management, CPU/GPU inference,
bounding-box coordinate restoration, schema compliance, ROI extraction,
visualization, and ground-truth IoU correspondence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from src.detection.detector import VehicleDetector
from src.detection.evaluator import DetectionEvaluator, compute_iou
from src.detection.visualizer import VehicleVisualizer
from src.utils.config import find_project_root


@pytest.fixture(scope="session")
def project_root() -> Path:
    return find_project_root()


@pytest.fixture(scope="session")
def sample_car_image_path(project_root: Path) -> Path:
    # First image of Stanford Cars train set
    p = project_root / "Stanford Dataset" / "cars_train" / "cars_train" / "00001.jpg"
    assert p.is_file(), f"Sample car image not found at {p}"
    return p


@pytest.fixture(scope="session")
def detector_cpu() -> VehicleDetector:
    return VehicleDetector(model_path="yolov8n.pt", device="cpu", conf_threshold=0.25)


# 1. YOLO Configuration Loading
def test_detection_configuration_loading(project_root: Path):
    cfg_path = project_root / "configs" / "detection.yaml"
    assert cfg_path.is_file(), f"Detection config missing: {cfg_path}"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "model" in cfg
    assert "training" in cfg
    assert "augmentation" in cfg
    assert "inference" in cfg
    assert "roi" in cfg

    # Verify vertical flipping is strictly disabled
    assert cfg["augmentation"].get("flipud", 0.0) == 0.0
    # Verify single-class target
    assert cfg["model"].get("num_classes") == 1
    assert "car" in cfg["model"].get("class_names", [])


# 2. Checkpoint Loading
def test_checkpoint_loading(detector_cpu: VehicleDetector):
    assert detector_cpu.model is not None
    assert detector_cpu.num_classes > 0
    assert hasattr(detector_cpu.model, "predict")


# 3. Output Schema Compliance
def test_inference_output_schema(detector_cpu: VehicleDetector, sample_car_image_path: Path):
    res = detector_cpu.detect(sample_car_image_path, extract_rois=True)

    # Root schema keys
    assert "detections" in res
    assert "inference_time_ms" in res
    assert "image_width" in res
    assert "image_height" in res

    assert isinstance(res["detections"], list)
    assert isinstance(res["inference_time_ms"], float)
    assert isinstance(res["image_width"], int)
    assert isinstance(res["image_height"], int)
    assert res["image_width"] > 0
    assert res["image_height"] > 0

    # Detection item schema
    assert len(res["detections"]) > 0
    det = res["detections"][0]
    required_item_keys = {"bbox", "class_id", "class_name", "confidence", "roi"}
    assert required_item_keys.issubset(det.keys())

    bbox = det["bbox"]
    assert {"x1", "y1", "x2", "y2"}.issubset(bbox.keys())
    assert isinstance(bbox["x1"], int)
    assert isinstance(bbox["y1"], int)
    assert isinstance(bbox["x2"], int)
    assert isinstance(bbox["y2"], int)

    assert det["class_id"] == 0
    assert det["class_name"] == "car"
    assert 0.0 <= det["confidence"] <= 1.0
    assert isinstance(det["roi"], np.ndarray)


# 4. Single-Image Inference
def test_single_image_inference(detector_cpu: VehicleDetector, sample_car_image_path: Path):
    res = detector_cpu.detect(sample_car_image_path)
    assert len(res["detections"]) >= 1
    assert res["detections"][0]["class_name"] == "car"
    assert res["inference_time_ms"] > 0.0


# 5. Multiple Detections Support and Sorting
def test_multiple_detections_support(detector_cpu: VehicleDetector, sample_car_image_path: Path):
    # With lower confidence threshold, background cars in 00001.jpg are detected
    res = detector_cpu.detect(sample_car_image_path, conf=0.15)
    assert len(res["detections"]) >= 1

    # Verify descending confidence ordering
    confs = [d["confidence"] for d in res["detections"]]
    assert confs == sorted(confs, reverse=True)


# 6. Bounding-Box Coordinate Restoration
def test_bounding_box_coordinate_restoration(detector_cpu: VehicleDetector, sample_car_image_path: Path):
    res = detector_cpu.detect(sample_car_image_path)
    w = res["image_width"]
    h = res["image_height"]

    for det in res["detections"]:
        b = det["bbox"]
        assert 0 <= b["x1"] < b["x2"] <= w, f"Invalid x bounds: {b} for width {w}"
        assert 0 <= b["y1"] < b["y2"] <= h, f"Invalid y bounds: {b} for height {h}"


# 7. Confidence Filtering
def test_confidence_filtering(detector_cpu: VehicleDetector, sample_car_image_path: Path):
    res_low = detector_cpu.detect(sample_car_image_path, conf=0.1)
    res_high = detector_cpu.detect(sample_car_image_path, conf=0.95)
    assert len(res_low["detections"]) >= len(res_high["detections"])


# 8. Empty Detections Graceful Handling
def test_empty_detections_handling(detector_cpu: VehicleDetector):
    # Pure blank image (black frame)
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    res = detector_cpu.detect(blank_img, conf=0.8)

    assert isinstance(res, dict)
    assert res["detections"] == []
    assert res["image_width"] == 300
    assert res["image_height"] == 300
    assert res["inference_time_ms"] >= 0.0


# 9. ROI Extraction and Aspect-Ratio Preservation
def test_roi_extraction_and_letterbox(detector_cpu: VehicleDetector, sample_car_image_path: Path):
    res = detector_cpu.detect(sample_car_image_path, extract_rois=True)
    assert len(res["detections"]) > 0

    first_det = res["detections"][0]
    roi = first_det["roi"]
    assert roi is not None
    assert isinstance(roi, np.ndarray)
    assert roi.shape == (224, 224, 3)
    assert roi.dtype == np.uint8


# 10. Visualization Rendering
def test_visualization_rendering(detector_cpu: VehicleDetector, sample_car_image_path: Path):
    res = detector_cpu.detect(sample_car_image_path)
    visualizer = VehicleVisualizer()

    annotated = visualizer.draw_detections(sample_car_image_path, res["detections"])
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == (res["image_height"], res["image_width"], 3)
    assert annotated.dtype == np.uint8

    grid = visualizer.visualize_rois_grid(res["detections"])
    if len(res["detections"]) > 0:
        assert grid is not None
        assert grid.shape[0] == 224


# 11. CPU Inference
def test_cpu_inference(detector_cpu: VehicleDetector, sample_car_image_path: Path):
    assert "cpu" in detector_cpu.device_str.lower()
    res = detector_cpu.detect(sample_car_image_path)
    assert len(res["detections"]) > 0


# 12. GPU Inference (Conditional)
def test_gpu_inference_if_available(sample_car_image_path: Path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU not available in current environment")

    gpu_detector = VehicleDetector(model_path="yolov8n.pt", device="cuda:0")
    assert "cuda" in gpu_detector.device_str.lower()
    res = gpu_detector.detect(sample_car_image_path)
    assert len(res["detections"]) > 0
    assert res["inference_time_ms"] > 0.0


# 13. Real Stanford Cars Ground-Truth Alignment & IoU
def test_real_stanford_cars_ground_truth_alignment(detector_cpu: VehicleDetector, sample_car_image_path: Path):
    # Ground truth bounding box for 00001.jpg is [38, 115, 568, 374]
    gt_box = (38, 115, 568, 374)

    res = detector_cpu.detect(sample_car_image_path, conf=0.25)
    assert len(res["detections"]) > 0

    best_iou = 0.0
    for d in res["detections"]:
        b = d["bbox"]
        p_box = (b["x1"], b["y1"], b["x2"], b["y2"])
        iou = compute_iou(gt_box, p_box)
        best_iou = max(best_iou, iou)

    # Bounding box must align with ground truth with high IoU
    assert best_iou >= 0.70, f"Expected high IoU alignment with GT, got {best_iou:.3f}"


# 14. Compute IoU Unit Logic
def test_compute_iou_logic():
    # Identical boxes
    assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    # Disjoint boxes
    assert compute_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # Half-overlap
    # box1: (0, 0, 10, 10) area=100
    # box2: (5, 0, 15, 10) area=100
    # intersection: (5, 0, 10, 10) area=50
    # union: 100 + 100 - 50 = 150
    # iou = 50 / 150 = 1/3
    assert abs(compute_iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-5


# 15. Flexible Image Input Types
def test_detector_input_types(detector_cpu: VehicleDetector, sample_car_image_path: Path):
    # Path input
    res_path = detector_cpu.detect(sample_car_image_path)
    assert len(res_path["detections"]) > 0

    # PIL Image input
    with Image.open(sample_car_image_path) as pil_img:
        res_pil = detector_cpu.detect(pil_img)
        assert len(res_pil["detections"]) > 0

    # NumPy RGB input
    with Image.open(sample_car_image_path) as pil_img:
        np_img = np.array(pil_img.convert("RGB"))
        res_np = detector_cpu.detect(np_img)
        assert len(res_np["detections"]) > 0


# 16. Detection Evaluator Model Specs
def test_detection_evaluator_specs():
    evaluator = DetectionEvaluator(model_path="yolov8n.pt", device="cpu")
    specs = evaluator.get_model_specs()
    assert "size_mb" in specs
    assert specs["size_mb"] > 0
    assert "parameter_count" in specs
    assert specs["parameter_count"] > 0


# 17. Fine-Tuned Checkpoint Loading & Alignment
def test_fine_tuned_best_model_if_available(project_root: Path, sample_car_image_path: Path):
    best_pt = project_root / "models" / "detection" / "best.pt"
    if not best_pt.is_file():
        pytest.skip("Fine-tuned best.pt checkpoint not found")

    det_best = VehicleDetector(model_path=best_pt, device="cpu", conf_threshold=0.25)
    assert det_best.is_single_class is True
    res = det_best.detect(sample_car_image_path)
    assert len(res["detections"]) >= 1

    # GT for 00001.jpg is [38, 115, 568, 374]
    gt_box = (38, 115, 568, 374)
    best_iou = 0.0
    for d in res["detections"]:
        b = d["bbox"]
        iou = compute_iou(gt_box, (b["x1"], b["y1"], b["x2"], b["y2"]))
        best_iou = max(best_iou, iou)
    assert best_iou >= 0.80, f"Expected fine-tuned detector to achieve high IoU >= 0.80, got {best_iou:.3f}"


# 18. True Batched Inference
def test_detect_batch_execution(detector_cpu: VehicleDetector, sample_car_image_path: Path):
    with Image.open(sample_car_image_path) as pil_img:
        np_img = np.array(pil_img.convert("RGB"))

    batch_res = detector_cpu.detect_batch([sample_car_image_path, np_img], extract_rois=True)
    assert len(batch_res) == 2
    for entry in batch_res:
        assert "detections" in entry
        assert "inference_time_ms" in entry
        assert "image_width" in entry
        assert "image_height" in entry
        assert entry["image_width"] > 0
        assert entry["image_height"] > 0
        assert len(entry["detections"]) >= 1


# 19. 1-Channel Grayscale and Float Image Inputs
def test_detector_1channel_and_float_inputs(detector_cpu: VehicleDetector):
    # 1-Channel (H, W, 1)
    img_1ch = np.zeros((120, 160, 1), dtype=np.uint8)
    res_1ch = detector_cpu.detect(img_1ch)
    assert res_1ch["image_width"] == 160
    assert res_1ch["image_height"] == 120
    assert isinstance(res_1ch["detections"], list)

    # Float array in [0.0, 1.0]
    img_flt = np.random.rand(120, 160, 3).astype(np.float32)
    res_flt = detector_cpu.detect(img_flt)
    assert res_flt["image_width"] == 160
    assert res_flt["image_height"] == 120
    assert isinstance(res_flt["detections"], list)


# 20. Visualizer Float Normalization, Top Edge Banner, and Non-Square Grid
def test_visualizer_float_and_edge_cases():
    vis = VehicleVisualizer()

    # Float array image
    flt_img = np.random.rand(100, 100, 3).astype(np.float32)
    dets = [{"bbox": {"x1": 10, "y1": 2, "x2": 80, "y2": 80}, "class_name": "car", "confidence": 0.92}]
    # Top edge y1=2 should not raise OpenCV error and should keep uint8
    annotated = vis.draw_detections(flt_img, dets)
    assert isinstance(annotated, np.ndarray)
    assert annotated.dtype == np.uint8

    # Non-square ROI grid
    roi_item = np.zeros((80, 120, 3), dtype=np.uint8)
    grid = vis.visualize_rois_grid([{"roi": roi_item}], grid_size=(120, 80))
    assert grid is not None
    assert grid.shape == (80, 120, 3)


# 21. VehicleDetector.from_config Loading
def test_vehicle_detector_from_config(project_root: Path):
    cfg_p = project_root / "configs" / "detection.yaml"
    detector = VehicleDetector.from_config(cfg_p)
    assert detector is not None
    assert detector.roi_target_size == (224, 224)
    assert detector.conf_threshold == 0.25
    assert detector.iou_threshold == 0.45


# 22. Auto-Resolution to Best Checkpoint
def test_detector_auto_resolves_to_best_model(project_root: Path):
    best_pt = project_root / "models" / "detection" / "best.pt"
    detector = VehicleDetector(model_path=None)
    if best_pt.is_file():
        assert detector.model_path.resolve() == best_pt.resolve()
        assert detector.is_single_class is True


# 23. FileNotFoundError on Nonexistent Image
def test_detector_raises_on_missing_file(detector_cpu: VehicleDetector):
    with pytest.raises(FileNotFoundError):
        detector_cpu.detect("nonexistent_image_path_12345.jpg")


# 24. Evaluator Robustness All Conditions
def test_evaluate_robustness_all_conditions(sample_car_image_path: Path):
    evaluator = DetectionEvaluator(model_path=None, device="cpu")
    summary = evaluator.evaluate_robustness([sample_car_image_path])
    expected_conditions = {
        "baseline",
        "low_light",
        "mild_blur",
        "partial_occlusion",
        "multiple_vehicles",
        "cropped_vehicle",
        "small_vehicle",
        "low_resolution_320x240",
        "high_resolution_1280x720",
    }
    for cond in expected_conditions:
        assert cond in summary, f"Condition {cond} missing from robustness summary"
        assert "detection_rate" in summary[cond]
        assert "mean_confidence" in summary[cond]


# 25. Evaluator Optimization Diagnostics
def test_evaluate_optimization(sample_car_image_path: Path):
    evaluator = DetectionEvaluator(model_path=None, device="cpu")
    opt = evaluator.evaluate_optimization([sample_car_image_path], batch_sizes=[1, 2], resolutions=[320])
    assert "model_size_mb" in opt
    assert "fp32_single_image_ms" in opt
    assert "resolution_latency_ms" in opt
    assert "320x320" in opt["resolution_latency_ms"]
    assert "batch_throughput" in opt
    assert "batch_1" in opt["batch_throughput"]
