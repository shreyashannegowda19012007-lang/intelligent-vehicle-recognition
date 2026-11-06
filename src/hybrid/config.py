"""
Configuration definitions for the Hybrid Intelligence Vehicle Recognition System.
Defines decision strategies, thresholds, component parameters, and configuration loaders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class DecisionStrategy(str, Enum):
    """Supported hybrid decision strategies."""
    DEEP_LEARNING_PRIMARY = "deep_learning_primary"
    AGREEMENT_BASED = "agreement_based"
    SCORE_FUSION = "score_fusion"


@dataclass
class HybridConfig:
    """Central configuration for the hybrid vehicle recognition pipeline."""

    # Decision Engine settings
    decision_strategy: DecisionStrategy = DecisionStrategy.DEEP_LEARNING_PRIMARY
    fallback_strategy: DecisionStrategy = DecisionStrategy.DEEP_LEARNING_PRIMARY
    fusion_weights: dict[str, float] = field(
        default_factory=lambda: {"deep_learning": 0.70, "classical": 0.30}
    )
    confidence_rejection_threshold: float = 0.20
    agreement_confidence_boost: float = 0.05
    uncertainty_threshold: float = 0.35

    # Detector settings
    detector_checkpoint: str | Path = "yolov8n.pt"
    detector_conf_threshold: float = 0.25
    detector_iou_threshold: float = 0.45
    detector_device: str = "auto"
    detector_half: bool = False
    roi_target_size: tuple[int, int] = (224, 224)
    roi_padding_ratio: float = 0.05
    roi_keep_aspect_ratio: bool = True

    # Classifier settings
    dl_checkpoint_path: str | Path | None = "outputs/classification/checkpoints/best_checkpoint.pt"
    dl_device: str = "auto"
    dl_temperature: float = 1.0
    classical_model_path: str | Path | None = "models/classical/svm/linear_svm_hog_color_lbp_pca128_model.joblib"
    classical_pipeline_path: str | Path | None = "models/classical/scalers/linear_svm_hog_color_lbp_pca128_pipeline.joblib"
    classical_uncertainty_threshold: float = 0.30
    top_k: int = 5

    # Paths
    class_mapping_path: str | Path | None = "data/processed/class_mapping.json"
    outputs_dir: str | Path = "outputs/hybrid"
    metrics_dir: str | Path = "outputs/hybrid/metrics"
    failure_cases_dir: str | Path = "outputs/hybrid/failure_cases"
    predictions_dir: str | Path = "outputs/hybrid/predictions"

    @classmethod
    def from_yaml(cls, yaml_path: str | Path | None = None) -> HybridConfig:
        """Load configuration from YAML file with fallback to defaults."""
        root = find_project_root()
        p = Path(yaml_path) if yaml_path else root / "configs" / "hybrid.yaml"
        if not p.is_file() and (root / p).is_file():
            p = root / p

        if not p.is_file():
            logger.info("Hybrid config YAML not found at %s. Using default HybridConfig.", p)
            return cls()

        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        dec_cfg = raw.get("decision", {})
        det_cfg = raw.get("detection", {})
        cls_cfg = raw.get("classification", {})
        dl_cfg = cls_cfg.get("deep_learning", {})
        cv_cfg = cls_cfg.get("classical", {})
        paths_cfg = raw.get("paths", {})

        raw_strategy = dec_cfg.get("strategy", "deep_learning_primary")
        try:
            strategy = DecisionStrategy(raw_strategy)
        except ValueError:
            logger.warning("Unknown strategy '%s', defaulting to deep_learning_primary", raw_strategy)
            strategy = DecisionStrategy.DEEP_LEARNING_PRIMARY

        return cls(
            decision_strategy=strategy,
            fallback_strategy=DecisionStrategy(dec_cfg.get("fallback_strategy", "deep_learning_primary")),
            fusion_weights=dec_cfg.get("fusion_weights", {"deep_learning": 0.70, "classical": 0.30}),
            confidence_rejection_threshold=float(dec_cfg.get("confidence_rejection_threshold", 0.20)),
            agreement_confidence_boost=float(dec_cfg.get("agreement_confidence_boost", 0.05)),
            uncertainty_threshold=float(dec_cfg.get("uncertainty_threshold", 0.35)),
            detector_checkpoint=det_cfg.get("model_checkpoint", "yolov8n.pt"),
            detector_conf_threshold=float(det_cfg.get("conf_threshold", 0.25)),
            detector_iou_threshold=float(det_cfg.get("iou_threshold", 0.45)),
            detector_device=det_cfg.get("device", "auto"),
            detector_half=bool(det_cfg.get("half", False)),
            roi_target_size=tuple(det_cfg.get("roi_target_size", (224, 224))),
            roi_padding_ratio=float(det_cfg.get("roi_padding_ratio", 0.05)),
            roi_keep_aspect_ratio=bool(det_cfg.get("roi_keep_aspect_ratio", True)),
            dl_checkpoint_path=dl_cfg.get("checkpoint_path", "outputs/classification/checkpoints/best_checkpoint.pt"),
            dl_device=dl_cfg.get("device", "auto"),
            dl_temperature=float(dl_cfg.get("temperature", 1.0)),
            classical_model_path=cv_cfg.get("model_path", "models/classical/linear_svm_model.joblib"),
            classical_pipeline_path=cv_cfg.get("pipeline_path"),
            classical_uncertainty_threshold=float(cv_cfg.get("uncertainty_threshold", 0.30)),
            top_k=int(dl_cfg.get("top_k", 5)),
            class_mapping_path=paths_cfg.get("class_mapping", "data/processed/class_mapping.json"),
            outputs_dir=paths_cfg.get("outputs_dir", "outputs/hybrid"),
            metrics_dir=paths_cfg.get("metrics_dir", "outputs/hybrid/metrics"),
            failure_cases_dir=paths_cfg.get("failure_cases_dir", "outputs/hybrid/failure_cases"),
            predictions_dir=paths_cfg.get("predictions_dir", "outputs/hybrid/predictions"),
        )
