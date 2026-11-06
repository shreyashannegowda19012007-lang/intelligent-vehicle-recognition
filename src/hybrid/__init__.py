"""
Hybrid Intelligence Vehicle Recognition System Package.
Integrates YOLO detection, Classical CV, and Deep Learning into a unified pipeline.
"""

from src.hybrid.adapters import (
    BaseClassifierAdapter,
    ClassicalAdapter,
    DeepLearningAdapter,
    DetectorAdapter,
)
from src.hybrid.config import DecisionStrategy, HybridConfig
from src.hybrid.decision_engine import HybridDecisionEngine
from src.hybrid.evaluator import HybridEvaluator
from src.hybrid.fusion_pipeline import HybridVehiclePipeline as HybridFusionPipeline
from src.hybrid.pipeline import HybridVehiclePipeline

__all__ = [
    "DecisionStrategy",
    "HybridConfig",
    "BaseClassifierAdapter",
    "DeepLearningAdapter",
    "ClassicalAdapter",
    "DetectorAdapter",
    "HybridDecisionEngine",
    "HybridVehiclePipeline",
    "HybridFusionPipeline",
    "HybridEvaluator",
]
