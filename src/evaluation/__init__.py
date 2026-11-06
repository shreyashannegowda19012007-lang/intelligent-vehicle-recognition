"""
Evaluation Package for Intelligent Fine-Grained Vehicle Recognition System.
Provides multi-task classification and detection metrics, ablation benchmarking,
environmental robustness testing, failure analysis, and visualization heatmaps.
"""

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

__all__ = [
    "SystemEvaluator",
    "FailureAnalyzer",
    "calculate_classification_metrics",
    "calculate_detection_metrics",
    "calculate_topk_accuracy",
    "compute_iou",
    "get_performance_metrics",
    "apply_gaussian_blur",
    "apply_brightness",
    "apply_reduced_resolution",
    "run_robustness_benchmark",
    "plot_class_confusion_matrix",
    "plot_make_confusion_matrix",
]
