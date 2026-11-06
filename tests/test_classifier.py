"""
Automated unit tests for deep learning vehicle classification models and backbones.
"""

from __future__ import annotations

import pytest
import torch

from src.classification.models.backbones import (
    SUPPORTED_ARCHITECTURES,
    VehicleClassificationModel,
    build_classification_model,
)


@pytest.mark.parametrize("arch", ["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b2"])
def test_build_classification_model_architectures(arch: str):
    model = build_classification_model(
        architecture=arch,
        num_classes=196,
        pretrained=False,
        dropout=0.2,
    )
    assert isinstance(model, VehicleClassificationModel)
    assert model.num_classes == 196
    assert model.architecture == arch

    # Forward pass with synthetic batch
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 196)
    assert not torch.isnan(logits).any()


def test_model_parameter_counts_and_freezing():
    model = build_classification_model(
        architecture="resnet18",
        num_classes=196,
        pretrained=False,
        freeze_backbone=True,
    )
    counts = model.count_parameters()
    assert counts["total_parameters"] > 10_000_000
    # Since backbone is frozen, trainable parameters should be head only
    assert counts["trainable_parameters"] < counts["total_parameters"]
    assert counts["trainable_parameters"] == (512 * 196 + 196)

    # Unfreeze and verify
    model.unfreeze_backbone_parameters()
    counts_unfrozen = model.count_parameters()
    assert counts_unfrozen["trainable_parameters"] == counts["total_parameters"]


def test_model_forward_with_features():
    model = build_classification_model(
        architecture="resnet18",
        num_classes=196,
        pretrained=False,
    )
    x = torch.randn(2, 3, 224, 224)
    logits, feats = model.forward_with_features(x)
    assert logits.shape == (2, 196)
    assert feats.shape == (2, 512)
