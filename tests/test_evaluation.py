"""
Comprehensive Unit Tests for Prompt 7 Evaluation Suite.
Verifies classification and detection metrics, performance monitoring,
robustness transforms, failure analysis categorization, and full artifact generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.evaluator import SystemEvaluator
from src.evaluation.failure_analyzer import FailureAnalyzer
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_detection_metrics,
    calculate_topk_accuracy,
    compute_iou,
    get_performance_metrics,
)
from src.evaluation.robustness import (
    apply_brightness,
    apply_gaussian_blur,
    apply_reduced_resolution,
    run_robustness_benchmark,
)
from src.evaluation.visualizer import (
    plot_class_confusion_matrix,
    plot_make_confusion_matrix,
)
from src.hybrid.fusion_pipeline import HybridVehiclePipeline


def test_topk_accuracy_computation():
    """Verify top-k accuracy computation logic."""
    y_true = [0, 1, 2]
    # Probs where 0 and 2 are in top-2, but 1 is not
    y_probs = np.array([
        [0.8, 0.1, 0.1],  # 0 is top-1
        [0.6, 0.1, 0.3],  # top-2 are classes 0 and 2, true class 1 is rank 3
        [0.2, 0.3, 0.5],  # 2 is top-1
    ])
    top1 = calculate_topk_accuracy(y_true, y_probs, k=1)
    top2 = calculate_topk_accuracy(y_true, y_probs, k=2)

    assert top1 == pytest.approx(2 / 3, rel=1e-3)
    assert top2 == pytest.approx(2 / 3, rel=1e-3)


def test_classification_metrics_multi_attribute():
    """Verify multi-level classification metrics (Top-1, Make, Model, Year, Precision, Recall, F1)."""
    # 22 is "Audi S4 Sedan 2012"
    # 23 is "Audi S4 Sedan 2007"
    # 118 is "GMC Savana Van 2012"
    y_true = [22, 23, 118]
    y_pred = [22, 22, 22]  # First correct, second same make&model diff year, third wrong make

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        confidences=[0.95, 0.85, 0.75],
    )

    assert metrics["total_samples"] == 3
    assert metrics["top1_accuracy"] == pytest.approx(1 / 3, rel=1e-2)
    assert metrics["make_accuracy"] >= 0.60  # At least 2 of 3 share Make "Audi"
    assert metrics["model_accuracy"] >= 0.60  # At least 2 of 3 share Model "S4"
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert metrics["confidence_statistics"]["mean_confidence"] == pytest.approx(0.85, rel=1e-2)


def test_detection_metrics_and_iou():
    """Verify IoU calculation and detection precision/recall/F1 metrics."""
    box_a = [10, 10, 50, 50]
    box_b = [10, 10, 50, 50]  # Exact match
    box_c = [30, 30, 70, 70]  # Partial overlap
    box_d = [100, 100, 200, 200]  # Disjoint

    assert compute_iou(box_a, box_b) == pytest.approx(1.0, rel=1e-3)
    assert 0.0 < compute_iou(box_a, box_c) < 1.0
    assert compute_iou(box_a, box_d) == 0.0

    gt_boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]
    pred_boxes = [[10, 10, 50, 50], [100, 100, 120, 120]]  # 1 TP, 1 FP, 1 FN

    det_metrics = calculate_detection_metrics(gt_boxes, pred_boxes, iou_threshold=0.5)
    assert det_metrics["true_positives"] == 1
    assert det_metrics["false_positives"] == 1
    assert det_metrics["false_negatives"] == 1
    assert det_metrics["precision"] == pytest.approx(0.5, rel=1e-2)
    assert det_metrics["recall"] == pytest.approx(0.5, rel=1e-2)
    assert det_metrics["f1_score"] == pytest.approx(0.5, rel=1e-2)


def test_performance_metrics_and_memory():
    """Verify performance latency, model size, and memory usage profiling."""
    latencies = [10.5, 12.0, 15.2, 9.8, 11.1]
    perf = get_performance_metrics(
        latencies_ms=latencies,
        parameter_count=3138003,
    )
    assert perf["latency"]["avg_inference_latency_ms"] > 0
    assert perf["latency"]["median_inference_latency_ms"] > 0
    assert perf["parameter_count"] == 3138003
    assert perf["approximate_ram_mb"] > 0


def test_robustness_perturbation_transforms():
    """Verify image perturbation operators preserve dimensions and dtype."""
    dummy_img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)

    blurred = apply_gaussian_blur(dummy_img, ksize=5, sigma=1.5)
    assert blurred.shape == dummy_img.shape
    assert blurred.dtype == np.uint8

    dimmed = apply_brightness(dummy_img, factor=0.5)
    assert dimmed.shape == dummy_img.shape
    assert dimmed.dtype == np.uint8
    assert dimmed.mean() < dummy_img.mean()

    bright = apply_brightness(dummy_img, factor=1.5)
    assert bright.shape == dummy_img.shape
    assert bright.dtype == np.uint8
    assert bright.mean() > dummy_img.mean()

    downscaled = apply_reduced_resolution(dummy_img, scale=0.5)
    assert downscaled.shape == dummy_img.shape
    assert downscaled.dtype == np.uint8


def test_failure_analyzer_categorization():
    """Verify failure taxonomy assigns meaningful root causes to error cases."""
    analyzer = FailureAnalyzer()

    # Case 1: Same make, different model
    true_audi_rs4 = {"class_name": "Audi RS 4 Convertible 2008", "make": "Audi", "model": "RS 4", "year": 2008}
    pred_audi_s4 = {"class_name": "Audi S4 Sedan 2012", "make": "Audi", "model": "S4", "year": 2012}
    cat1, reason1 = analyzer.categorize_failure(true_audi_rs4, pred_audi_s4, 0.15)
    assert cat1 == "visually_similar_models"
    assert "Audi" in reason1

    # Case 2: Same make, same model, different year
    true_audi_2007 = {"class_name": "Audi S4 Sedan 2007", "make": "Audi", "model": "S4", "year": 2007}
    cat2, reason2 = analyzer.categorize_failure(true_audi_2007, pred_audi_s4, 0.45)
    assert cat2 == "facelift_year_confusion"
    assert "production year" in reason2.lower()

    # Case 3: Perturbation condition
    cat3, reason3 = analyzer.categorize_failure(true_audi_rs4, pred_audi_s4, 0.05, condition="blur")
    assert cat3 == "severe_illumination_or_quality"


def test_hybrid_pipeline_initialization_and_status():
    """Verify HybridVehiclePipeline initializes gracefully with missing dedicated weights."""
    pipeline = HybridVehiclePipeline()
    status = pipeline.get_status()

    assert status["ready"] is True
    assert status["deep_classifier_available"] is True
    assert "fusion_weights" in status
    assert isinstance(status["missing_dependencies"], list)


def test_system_evaluator_end_to_end(tmp_path: Path):
    """Verify end-to-end evaluation runner produces all required Prompt 7 artifacts."""
    output_dir = tmp_path / "evaluation_results"
    evaluator = SystemEvaluator(output_dir=output_dir)

    metrics = evaluator.run_full_evaluation()

    assert "classification_benchmarks" in metrics
    assert "detection_benchmarks" in metrics
    assert "robustness_summary" in metrics

    # Verify all 7 required deliverables are generated
    expected_files = [
        "metrics.json",
        "comparison.csv",
        "ablation.csv",
        "robustness.csv",
        "failure_analysis.md",
        "confusion_matrix_classes.png",
        "confusion_matrix_makes.png",
        "eval_P5_TeamID.md",
    ]
    for fname in expected_files:
        fpath = output_dir / fname
        assert fpath.is_file(), f"Expected deliverable not found: {fname}"
        assert fpath.stat().st_size > 0, f"Deliverable is empty: {fname}"

