"""
Vehicle Detection and Localization Package.
Provides YOLOv8 vehicle detection, custom training, evaluation,
bounding-box NMS filtering, ROI extraction, and visualization.
"""

from __future__ import annotations

from src.detection.detector import VehicleDetector
from src.detection.evaluator import DetectionEvaluator, compute_iou
from src.detection.trainer import DetectionTrainer
from src.detection.visualizer import VehicleVisualizer

__all__ = [
    "DetectionEvaluator",
    "DetectionTrainer",
    "VehicleDetector",
    "VehicleVisualizer",
    "compute_iou",
]
