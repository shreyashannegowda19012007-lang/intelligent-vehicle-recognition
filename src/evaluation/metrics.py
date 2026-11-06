"""
Evaluation Metrics Suite for Fine-Grained Vehicle Recognition and Localization.
Computes multi-level classification metrics (Top-1, Top-5, Make, Model, Year,
Macro/Weighted Precision, Recall, F1), detection localization metrics (mAP, IoU,
Detection Precision/Recall), and hardware/runtime performance benchmarks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.classification.metrics import ClassMappingDecoder
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


def calculate_topk_accuracy(y_true: Sequence[int], y_probs: np.ndarray, k: int = 5) -> float:
    """
    Calculate Top-K classification accuracy given 2D probability/logit matrix.

    Args:
        y_true: Array of true class integers of shape (N,).
        y_probs: Matrix of prediction probabilities/logits of shape (N, C).
        k: Top-K rank cutoff.

    Returns:
        Fractional Top-K accuracy in [0.0, 1.0].
    """
    if len(y_true) == 0:
        return 0.0
    y_true_arr = np.asarray(y_true, dtype=int)
    probs_arr = np.asarray(y_probs)

    if probs_arr.ndim != 2 or probs_arr.shape[0] != len(y_true_arr):
        return 0.0

    num_classes = probs_arr.shape[1]
    actual_k = min(k, num_classes)
    # Get top-k indices along axis 1
    topk_indices = np.argpartition(probs_arr, -actual_k, axis=1)[:, -actual_k:]
    # Check if ground-truth matches any in top-k
    matches = (topk_indices == y_true_arr[:, None]).any(axis=1)
    return float(np.mean(matches))


def calculate_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_probs: np.ndarray | None = None,
    class_decoder: ClassMappingDecoder | None = None,
    confidences: Sequence[float] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Comprehensive multi-attribute vehicle classification metrics calculation.

    Computes:
    - Top-1 Accuracy
    - Top-5 Accuracy (if probabilities supplied)
    - Hierarchical attribute accuracies: Make, Model, Year
    - Precision, Recall, F1 (both macro and weighted)
    - Average and distribution of prediction confidence

    Args:
        y_true: Ground truth class IDs (0-indexed).
        y_pred: Predicted class IDs (0-indexed).
        y_probs: Optional (N, C) probability matrix for Top-K calculation.
        class_decoder: Optional ClassMappingDecoder for Make/Model/Year decoding.
        confidences: Optional list of top-1 confidence scores.
        top_k: Top-K cutoff for accuracy (default: 5).

    Returns:
        Structured dictionary of classification metrics.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)
    n_samples = len(y_t)

    if n_samples == 0:
        return {
            "total_samples": 0,
            "top1_accuracy": 0.0,
            "top5_accuracy": 0.0,
            "make_accuracy": 0.0,
            "model_accuracy": 0.0,
            "year_accuracy": 0.0,
            "precision_macro": 0.0,
            "precision_weighted": 0.0,
            "recall_macro": 0.0,
            "recall_weighted": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
        }

    # 1. Top-1 Accuracy
    top1_acc = float(accuracy_score(y_t, y_p))

    # 2. Top-5 Accuracy
    if y_probs is not None and y_probs.shape[0] == n_samples and y_probs.ndim == 2:
        top5_acc = calculate_topk_accuracy(y_t, y_probs, k=top_k)
    else:
        # If top-1 is correct, it's also in top-5
        top5_acc = top1_acc

    # 3. Precision, Recall, F1
    prec_macro = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_t, y_p, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_t, y_p, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    f1_w = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    # 4. Attribute-level Accuracies (Make, Model, Year)
    decoder = class_decoder or ClassMappingDecoder()
    make_correct = 0
    model_correct = 0
    year_correct = 0

    for t_id, p_id in zip(y_t, y_p, strict=False):
        t_info = decoder.get_info(int(t_id))
        p_info = decoder.get_info(int(p_id))

        if t_info["make"].strip().lower() == p_info["make"].strip().lower():
            make_correct += 1
        if t_info["model"].strip().lower() == p_info["model"].strip().lower():
            model_correct += 1
        if t_info["year"] != 0 and t_info["year"] == p_info["year"]:
            year_correct += 1

    make_acc = float(make_correct / n_samples)
    model_acc = float(model_correct / n_samples)
    year_acc = float(year_correct / n_samples)

    # 5. Confidence stats
    conf_stats = {}
    if confidences is not None and len(confidences) > 0:
        c_arr = np.asarray(confidences, dtype=float)
        conf_stats = {
            "mean_confidence": round(float(np.mean(c_arr)), 4),
            "std_confidence": round(float(np.std(c_arr)), 4),
            "min_confidence": round(float(np.min(c_arr)), 4),
            "max_confidence": round(float(np.max(c_arr)), 4),
        }

    return {
        "total_samples": n_samples,
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "make_accuracy": round(make_acc, 4),
        "model_accuracy": round(model_acc, 4),
        "year_accuracy": round(year_acc, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_m, 4),
        "f1_weighted": round(f1_w, 4),
        "confidence_statistics": conf_stats,
    }


def compute_iou(
    box1: Sequence[float],
    box2: Sequence[float],
) -> float:
    """
    Calculate IoU between box1 and box2 in format [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, float(x2 - x1))
    inter_h = max(0.0, float(y2 - y1))
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def calculate_detection_metrics(
    ground_truth_boxes: Sequence[Sequence[float]],
    predicted_boxes: Sequence[Sequence[float]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute detection precision, recall, F1, and mean IoU.

    Args:
        ground_truth_boxes: List of [x1, y1, x2, y2] GT bounding boxes.
        predicted_boxes: List of [x1, y1, x2, y2] predicted bounding boxes.
        iou_threshold: Minimum IoU to count as a True Positive.

    Returns:
        Structured detection metrics.
    """
    tp = 0
    fp = 0
    ious = []

    matched_gt = set()
    for pred_idx, pred_box in enumerate(predicted_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        ious.append(best_iou)
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truth_boxes) - len(matched_gt)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    mean_iou = float(np.mean(ious)) if ious else 0.0

    return {
        "total_ground_truth": len(ground_truth_boxes),
        "total_predictions": len(predicted_boxes),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "iou_threshold": iou_threshold,
    }


def get_performance_metrics(
    latencies_ms: Sequence[float] | None = None,
    model_path: str | Path | None = None,
    parameter_count: int | None = None,
) -> dict[str, Any]:
    """
    Compute system performance metrics: inference time, model size, memory usage.

    Args:
        latencies_ms: Sequence of latency measurements in milliseconds.
        model_path: Path to checkpoint or serialized model file.
        parameter_count: Total parameter count of neural network.

    Returns:
        Structured performance dictionary.
    """
    # 1. Latency statistics
    latency_stats = {
        "avg_inference_latency_ms": 0.0,
        "median_inference_latency_ms": 0.0,
        "p95_inference_latency_ms": 0.0,
    }
    if latencies_ms and len(latencies_ms) > 0:
        l_arr = np.asarray(latencies_ms, dtype=float)
        latency_stats = {
            "avg_inference_latency_ms": round(float(np.mean(l_arr)), 2),
            "median_inference_latency_ms": round(float(np.median(l_arr)), 2),
            "p95_inference_latency_ms": round(float(np.percentile(l_arr, 95)), 2),
        }

    # 2. Model Size on disk
    size_mb = 0.0
    if model_path is not None:
        p = Path(model_path)
        if p.is_file():
            size_mb = round(os.path.getsize(p) / (1024 * 1024), 2)

    # 3. System & GPU Memory
    process = psutil.Process(os.getpid())
    ram_mb = round(process.memory_info().rss / (1024 * 1024), 2)

    gpu_vram_gb = 0.0
    if torch.cuda.is_available():
        try:
            gpu_vram_gb = round(torch.cuda.memory_allocated() / (1024**3), 3)
        except Exception:
            gpu_vram_gb = 0.0

    return {
        "latency": latency_stats,
        "model_size_mb": size_mb,
        "parameter_count": parameter_count or 0,
        "approximate_ram_mb": ram_mb,
        "approximate_vram_gb": gpu_vram_gb,
    }

