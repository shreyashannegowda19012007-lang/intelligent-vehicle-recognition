"""
Unified Inference Interface for Fine-Grained Deep Learning Vehicle Classification.
Loads model weights and taxonomy mappings once, preprocesses vehicle crops, runs inference,
and returns structured predictions with decoded Make, Model, Year, top-5 classes, and latency.
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

from src.classification.config import ClassificationConfig
from src.classification.metrics import ClassMappingDecoder
from src.classification.models.backbones import build_classification_model
from src.classification.transforms import get_inference_transforms
from src.utils.config import find_project_root
from src.utils.device import get_device

logger = logging.getLogger(__name__)


class VehicleClassifier:
    """
    Production-grade inference engine for fine-grained Stanford Cars classification.
    Accepts arbitrary vehicle crops (PIL Image, numpy ndarray, file path, or torch.Tensor)
    and produces standardized structured prediction outputs.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model: nn.Module | None = None,
        config: ClassificationConfig | None = None,
        class_mapping_path: str | Path | None = None,
        device: str | torch.device = "auto",
        top_k: int = 5,
        temperature: float = 1.0,
    ) -> None:
        """
        Initialize persistent classifier.

        Args:
            checkpoint_path: Path to checkpoint .pt file (defaults to best_checkpoint.pt).
            model: Optional pre-instantiated nn.Module.
            config: Optional ClassificationConfig instance.
            class_mapping_path: Optional path to class_mapping.json / class_mapping.csv.
            device: 'auto', 'cuda', 'cpu', or torch.device.
            top_k: Number of ranked predictions to return (default: 5).
            temperature: Softmax calibration temperature scalar (default: 1.0).
        """
        self.root = find_project_root()
        self.top_k = top_k
        self.temperature = max(0.01, float(temperature))

        # 1. Resolve Device
        if device == "auto":
            self.device = get_device("auto")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # 2. Load Class Taxonomy Decoder
        self.decoder = ClassMappingDecoder(class_mapping_path)

        # 3. Load or assign model
        if checkpoint_path is not None:
            self.checkpoint_path = (
                self.root / checkpoint_path if not Path(checkpoint_path).is_file() else Path(checkpoint_path)
            )
            self.model, self.config = self._load_checkpoint(self.checkpoint_path)
        elif model is not None:
            self.model = model.to(self.device)
            self.config = config or ClassificationConfig()
            self.checkpoint_path = None
        else:
            # Fallback to default best checkpoint if available, or build pretrained baseline
            default_ckpt = self.root / "outputs" / "classification" / "checkpoints" / "best_checkpoint.pt"
            if default_ckpt.is_file():
                self.checkpoint_path = default_ckpt
                self.model, self.config = self._load_checkpoint(self.checkpoint_path)
            else:
                logger.warning("No checkpoint found at %s. Initializing baseline ResNet50...", default_ckpt)
                self.config = config or ClassificationConfig()
                self.model = build_classification_model(
                    architecture=self.config.architecture,
                    num_classes=self.config.num_classes,
                    pretrained=True,
                    dropout=self.config.dropout,
                ).to(self.device)
                self.checkpoint_path = None

        self.model.eval()

        # 4. Initialize Preprocessing Pipeline
        self.transform = get_inference_transforms(
            image_size=self.config.image_size,
            mean=self.config.mean,
            std=self.config.std,
        )

    def _load_checkpoint(self, path: Path) -> tuple[nn.Module, ClassificationConfig]:
        """Load weights and configuration from disk."""
        if not path.is_file():
            raise FileNotFoundError(f"Classification checkpoint not found: {path}")

        logger.info("Loading VehicleClassifier weights from %s...", path)
        checkpoint = torch.load(path, map_location=self.device)

        cfg_dict = checkpoint.get("config", {})
        config = ClassificationConfig.from_dict(cfg_dict) if cfg_dict else ClassificationConfig()

        arch = checkpoint.get("architecture", config.architecture)
        num_classes = checkpoint.get("num_classes", config.num_classes)

        model = build_classification_model(
            architecture=arch,
            num_classes=num_classes,
            pretrained=False,
            dropout=config.dropout,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        return model, config

    def _prepare_image(
        self,
        image_input: str | Path | Image.Image | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert arbitrary input into normalized 4D tensor (1, 3, H, W).
        """
        if isinstance(image_input, torch.Tensor):
            if image_input.ndim == 3:
                return image_input.unsqueeze(0).to(self.device)
            elif image_input.ndim == 4:
                return image_input.to(self.device)
            else:
                raise ValueError(f"Invalid tensor shape: {image_input.shape}")

        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.is_file():
                p = self.root / p
            if not p.is_file():
                raise FileNotFoundError(f"Image not found at {image_input}")
            with Image.open(p) as raw:
                pil_img = raw.convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0 or image_input.ndim < 2:
                raise ValueError("Empty or invalid image array provided to classifier.")
            arr = image_input
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.concatenate([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            elif arr.ndim == 3 and arr.shape[2] != 3:
                raise ValueError(f"Unsupported channel count: {arr.shape[2]}")

            # Convert float to uint8 if needed
            if np.issubdtype(arr.dtype, np.floating):
                if arr.max() <= 1.0 + 1e-5:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)

            pil_img = Image.fromarray(arr, mode="RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        tensor = self.transform(pil_img)  # shape: (3, H, W)
        return tensor.unsqueeze(0).to(self.device)

    def predict(
        self,
        image: str | Path | Image.Image | np.ndarray | torch.Tensor,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Classify a single vehicle crop.

        Args:
            image: Vehicle crop image input.
            top_k: Optional override for top-K classes to return.

        Returns:
            Standardized dictionary matching contract:
            {
              "predicted_class": "2012 Tesla Model S",
              "make": "Tesla",
              "model": "Model S",
              "year": 2012,
              "confidence": 0.94,
              "top_5": [
                {
                  "class": "2012 Tesla Model S",
                  "confidence": 0.94
                }
              ],
              "inference_time_ms": 18.5
            }
        """
        k = top_k or self.top_k

        # Handle empty/invalid crops safely
        try:
            tensor = self._prepare_image(image)
        except Exception as e:
            logger.warning("Failed to prepare image for classification: %s", e)
            return {
                "predicted_class": "Unknown",
                "make": "Unknown",
                "model": "Unknown",
                "year": 0,
                "confidence": 0.0,
                "top_5": [],
                "inference_time_ms": 0.0,
                "error": str(e),
            }

        t_start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            if self.temperature != 1.0:
                logits = logits / self.temperature
            probs = torch.softmax(logits, dim=1).squeeze(0)
        t_end = time.perf_counter()
        latency_ms = round((t_end - t_start) * 1000.0, 2)

        # Extract top-K rankings
        num_avail = probs.size(0)
        actual_k = min(k, num_avail)
        top_probs, top_indices = torch.topk(probs, actual_k)

        top_probs_list = top_probs.cpu().tolist()
        top_indices_list = top_indices.cpu().tolist()

        best_id = top_indices_list[0]
        best_conf = round(float(top_probs_list[0]), 4)

        best_info = self.decoder.get_info(best_id)

        top_list = []
        for idx, (cid, p_val) in enumerate(zip(top_indices_list, top_probs_list, strict=False)):
            info = self.decoder.get_info(cid)
            top_list.append({
                "rank": idx + 1,
                "class_id": cid,
                "class": info["class_name"],
                "make": info["make"],
                "model": info["model"],
                "year": info["year"],
                "confidence": round(float(p_val), 4),
            })

        return {
            "predicted_class": best_info["class_name"],
            "class_id": best_id,
            "make": best_info["make"],
            "model": best_info["model"],
            "year": best_info["year"],
            "confidence": best_conf,
            "top_5": top_list,
            "inference_time_ms": latency_ms,
        }

    def predict_batch(
        self,
        images: list[str | Path | Image.Image | np.ndarray | torch.Tensor],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run batched inference across a collection of images."""
        return [self.predict(img, top_k=top_k) for img in images]
