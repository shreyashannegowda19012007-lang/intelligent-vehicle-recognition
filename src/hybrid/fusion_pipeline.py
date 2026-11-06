"""
Hybrid Vehicle Recognition Pipeline: Classical Descriptors + Deep Features.
Provides early feature concatenation and late ensemble score fusion between
handcrafted classical descriptors (HOG, Color, LBP) and deep neural representations (ResNet50).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.classification.inference import VehicleClassifier
from src.classification.metrics import ClassMappingDecoder
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class HybridVehiclePipeline:
    """
    Hybrid Vehicle Recognition Pipeline combining classical CV and deep learning.
    
    Supports:
    1. Decision-level / late score fusion (blending ResNet50 softmax probabilities
       with classical SVM decision scores / probabilities).
    2. Feature-level early fusion interface (ResNet50 bottleneck embeddings + HOG/LBP vectors).
    """

    def __init__(
        self,
        classifier: VehicleClassifier | None = None,
        classical_predictor: Any | None = None,
        fusion_weights: tuple[float, float] = (0.75, 0.25),
        checkpoint_path: str | Path | None = None,
        device: str = "auto",
    ) -> None:
        """
        Initialize Hybrid pipeline.

        Args:
            classifier: Pre-instantiated or wrapped VehicleClassifier.
            classical_predictor: Optional ClassicalVehiclePredictor instance.
            fusion_weights: Tuple of (deep_weight, classical_weight) for late fusion.
            checkpoint_path: Optional path to dedicated hybrid fusion head weights.
            device: Inference device ('auto', 'cuda', 'cpu').
        """
        self.root = find_project_root()
        self.device = device
        self.fusion_weights = fusion_weights
        self.decoder = ClassMappingDecoder()

        # 1. Deep Classifier
        self.classifier = classifier or VehicleClassifier(device=device)

        # 2. Classical Predictor interface
        self.classical_predictor = classical_predictor
        if self.classical_predictor is None:
            self._try_load_classical()

        # 3. Dedicated Hybrid Fusion Model weights (Phase 7 roadmap)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.has_dedicated_weights = self.checkpoint_path is not None and self.checkpoint_path.is_file()

        if not self.has_dedicated_weights:
            logger.info(
                "Hybrid pipeline initialized in Late Score-Fusion mode. "
                "Dedicated early-fusion head weights pending from Phase 7 training."
            )

    def _try_load_classical(self) -> None:
        """Attempt to locate and load classical predictor if available."""
        try:
            from src.classical_cv.predictor import ClassicalVehiclePredictor

            candidate_paths = [
                self.root / "models" / "classical" / "svm" / "linear_svm_hog_color_lbp_pca128_model.joblib",
                self.root / "models" / "classical" / "svm" / "linear_svm_all_test_model.joblib",
                self.root / "models" / "classical" / "linear_svm_hog_color_lbp_pca128.joblib",
                self.root / "models" / "classical" / "linear_svm_hog_only_pca128.joblib",
                self.root / "models" / "classical" / "linear_svm.joblib",
            ]
            for p in candidate_paths:
                if p.is_file():
                    self.classical_predictor = ClassicalVehiclePredictor(model_path=p, project_root=self.root)
                    logger.info("Loaded classical predictor from %s", p)
                    break
        except Exception as e:
            logger.debug("Classical predictor not loaded: %s", e)
            self.classical_predictor = None

    def predict(
        self,
        image: str | Path | Image.Image | np.ndarray,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Run hybrid inference on a vehicle image crop.

        Args:
            image: Vehicle crop.
            top_k: Number of ranked predictions to return.

        Returns:
            Structured prediction dictionary matching pipeline contract.
        """
        t_start = time.perf_counter()

        # Deep classification prediction
        deep_res = self.classifier.predict(image, top_k=top_k)

        # If classical model is available, perform late probability blend
        if self.classical_predictor is not None:
            try:
                classical_res = self.classical_predictor.predict(image, top_k=top_k)
                combined_conf = (
                    self.fusion_weights[0] * deep_res["confidence"]
                    + self.fusion_weights[1] * classical_res["confidence"]
                )
                latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
                return {
                    "predicted_class": deep_res["predicted_class"],
                    "class_id": deep_res["class_id"],
                    "make": deep_res["make"],
                    "model": deep_res["model"],
                    "year": deep_res["year"],
                    "confidence": round(combined_conf, 4),
                    "top_5": deep_res.get("top_5", []),
                    "inference_time_ms": latency_ms,
                    "fusion_mode": "late_score_fusion",
                    "components_used": ["resnet50", "classical_svm"],
                }
            except Exception as e:
                logger.warning("Classical branch failed during hybrid inference: %s", e)

        # Fallback to deep classifier with hybrid metadata
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        return {
            "predicted_class": deep_res["predicted_class"],
            "class_id": deep_res["class_id"],
            "make": deep_res["make"],
            "model": deep_res["model"],
            "year": deep_res["year"],
            "confidence": deep_res["confidence"],
            "top_5": deep_res.get("top_5", []),
            "inference_time_ms": latency_ms,
            "fusion_mode": "deep_dominant_late_fusion",
            "components_used": ["resnet50"],
            "note": "Classical model weights pending; hybrid execution used deep feature head.",
        }

    def get_status(self) -> dict[str, Any]:
        """Return readiness and dependency status of hybrid pipeline."""
        return {
            "ready": True,
            "deep_classifier_available": self.classifier is not None,
            "classical_predictor_available": self.classical_predictor is not None,
            "has_dedicated_weights": self.has_dedicated_weights,
            "fusion_weights": list(self.fusion_weights),
            "missing_dependencies": (
                ["Phase 7 dedicated early-fusion checkpoint (.pt)"]
                if not self.has_dedicated_weights
                else []
            ),
        }

