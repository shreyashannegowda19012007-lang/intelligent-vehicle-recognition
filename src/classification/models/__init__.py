"""Classification models and backbones package."""

from src.classification.models.backbones import (
    SUPPORTED_ARCHITECTURES,
    VehicleClassificationModel,
    build_classification_model,
)

__all__ = [
    "SUPPORTED_ARCHITECTURES",
    "VehicleClassificationModel",
    "build_classification_model",
]
