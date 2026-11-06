"""
Intelligent Vehicle Recognition - Deep Learning Fine-Grained Classification Package.
Provides pretrained backbones, training engine, evaluation, calibration, robustness analysis,
production inference engine, and YOLO detector pipeline integration.
"""

from src.classification.calibration import ModelWithTemperature, compute_ece
from src.classification.config import ClassificationConfig
from src.classification.evaluate import ClassifierEvaluator
from src.classification.inference import VehicleClassifier
from src.classification.metrics import (
    ClassMappingDecoder,
    calculate_classification_metrics,
)
from src.classification.models.backbones import (
    SUPPORTED_ARCHITECTURES,
    VehicleClassificationModel,
    build_classification_model,
)
from src.classification.pipeline import VehicleRecognitionPipeline
from src.classification.robustness import RobustnessAnalyzer
from src.classification.trainer import ClassifierTrainer, set_seed
from src.classification.transforms import (
    get_inference_transforms,
    get_robustness_transform,
    get_training_transforms,
    get_validation_transforms,
)

__all__ = [
    "ClassificationConfig",
    "VehicleClassificationModel",
    "build_classification_model",
    "SUPPORTED_ARCHITECTURES",
    "get_training_transforms",
    "get_validation_transforms",
    "get_inference_transforms",
    "get_robustness_transform",
    "ClassMappingDecoder",
    "calculate_classification_metrics",
    "ModelWithTemperature",
    "compute_ece",
    "ClassifierTrainer",
    "set_seed",
    "ClassifierEvaluator",
    "RobustnessAnalyzer",
    "VehicleClassifier",
    "VehicleRecognitionPipeline",
]
