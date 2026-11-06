"""
Pretrained Deep Learning Model Architectures for Fine-Grained Vehicle Classification.
Supports ResNet50 (baseline), EfficientNet-B0/B2 (efficient alternatives),
ConvNeXt-Tiny, and ResNet18 with configurable transfer learning and head replacement.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tv_models

logger = logging.getLogger(__name__)

SUPPORTED_ARCHITECTURES = {
    "resnet50",
    "resnet18",
    "efficientnet_b0",
    "efficientnet_b2",
    "convnext_tiny",
}


class VehicleClassificationModel(nn.Module):
    """
    Modular wrapper for fine-grained vehicle classification.
    Wraps standard convolutional or modern backbones, replaces the final classification head
    with a 196-class output layer, and supports feature extraction.
    """

    def __init__(
        self,
        architecture: str = "resnet50",
        num_classes: int = 196,
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.pretrained = pretrained

        if self.architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. "
                f"Supported options: {sorted(SUPPORTED_ARCHITECTURES)}"
            )

        self.backbone, self.in_features = self._build_backbone(
            self.architecture, pretrained=pretrained
        )

        if freeze_backbone:
            self.freeze_backbone_parameters()

        # Build custom classification head with dropout
        if self.dropout_rate > 0:
            self.classifier_head = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(self.in_features, self.num_classes),
            )
        else:
            self.classifier_head = nn.Linear(self.in_features, self.num_classes)

        self._integrate_head()

    def _build_backbone(
        self, arch: str, pretrained: bool
    ) -> tuple[nn.Module, int]:
        """Instantiate torchvision backbone and identify classifier feature dimension."""
        if arch == "resnet50":
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            model = tv_models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            return model, in_features

        elif arch == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            model = tv_models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            return model, in_features

        elif arch == "efficientnet_b0":
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = tv_models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, in_features

        elif arch == "efficientnet_b2":
            weights = tv_models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            model = tv_models.efficientnet_b2(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, in_features

        elif arch == "convnext_tiny":
            try:
                import timm
                model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=0)
                in_features = model.num_features
                return model, in_features
            except Exception:
                weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
                model = tv_models.convnext_tiny(weights=weights)
                in_features = model.classifier[2].in_features
                model.classifier = nn.Identity()
                return model, in_features

        raise ValueError(f"Unhandled architecture: {arch}")

    def _integrate_head(self) -> None:
        """Replace the placeholder identity with the custom head inside the backbone."""
        if self.architecture in {"resnet50", "resnet18"}:
            self.backbone.fc = self.classifier_head
        elif self.architecture in {"efficientnet_b0", "efficientnet_b2"}:
            self.backbone.classifier = self.classifier_head
        elif self.architecture == "convnext_tiny":
            if hasattr(self.backbone, "classifier"):
                # Torchvision ConvNeXt classifier has [LayerNorm2d, Flatten, Linear]
                self.backbone.classifier = nn.Sequential(
                    tv_models.convnext.LayerNorm2d(self.in_features, eps=1e-6),
                    nn.Flatten(1),
                    self.classifier_head,
                )
            # For timm backbone with num_classes=0, head is already Identity and classifier_head is called directly

    def freeze_backbone_parameters(self) -> None:
        """Freeze all backbone parameters except the classification head."""
        for name, param in self.named_parameters():
            if "classifier_head" not in name:
                param.requires_grad = False
        logger.info("Backbone parameters frozen. Only classifier head is trainable.")

    def unfreeze_backbone_parameters(self) -> None:
        """Unfreeze all parameters for full model fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("All model parameters unfrozen for end-to-end training.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw class logits of shape (B, num_classes)."""
        if self.architecture == "convnext_tiny" and not hasattr(self.backbone, "classifier"):
            feats = self.backbone(x)
            return self.classifier_head(feats)
        return self.backbone(x)

    def forward_with_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning both classification logits and penultimate feature embedding.
        """
        if self.architecture in {"resnet50", "resnet18"}:
            # Forward up to avgpool
            h = self.backbone.conv1(x)
            h = self.backbone.bn1(h)
            h = self.backbone.relu(h)
            h = self.backbone.maxpool(h)
            h = self.backbone.layer1(h)
            h = self.backbone.layer2(h)
            h = self.backbone.layer3(h)
            h = self.backbone.layer4(h)
            h = self.backbone.avgpool(h)
            feats = torch.flatten(h, 1)
            logits = self.classifier_head(feats)
            return logits, feats

        elif self.architecture in {"efficientnet_b0", "efficientnet_b2"}:
            h = self.backbone.features(x)
            h = self.backbone.avgpool(h)
            feats = torch.flatten(h, 1)
            logits = self.classifier_head(feats)
            return logits, feats

        elif self.architecture == "convnext_tiny":
            if hasattr(self.backbone, "classifier"):
                feats = self.backbone.features(x)
                feats = self.backbone.avgpool(feats)
                feats = torch.flatten(feats, 1)
                logits = self.classifier_head(feats)
                return logits, feats
            else:
                feats = self.backbone(x)
                logits = self.classifier_head(feats)
                return logits, feats

        # Generic fallback
        logits = self.forward(x)
        return logits, logits

    def count_parameters(self) -> dict[str, int]:
        """Return counts of total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}


def build_classification_model(
    architecture: str = "resnet50",
    num_classes: int = 196,
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> VehicleClassificationModel:
    """
    Factory function to construct a vehicle classification model.

    Args:
        architecture: Backbone name ('resnet50', 'efficientnet_b0', 'efficientnet_b2', 'convnext_tiny', 'resnet18').
        num_classes: Number of vehicle categories (Stanford Cars has 196).
        pretrained: If True, loads official ImageNet-1K pretrained weights.
        dropout: Dropout probability before classification linear layer.
        freeze_backbone: If True, freezes feature extractor layers for transfer learning.

    Returns:
        Instantiated VehicleClassificationModel.
    """
    logger.info(
        "Building model '%s' (classes=%d, pretrained=%s, dropout=%.2f, freeze=%s)...",
        architecture,
        num_classes,
        pretrained,
        dropout,
        freeze_backbone,
    )
    return VehicleClassificationModel(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
