"""
Configuration for Fine-Grained Deep Learning Vehicle Classification.
Provides structured dataclass and YAML/JSON serialization for model, training,
data, and evaluation parameters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.utils.config import find_project_root


@dataclass
class ClassificationConfig:
    # Model parameters
    architecture: str = "resnet50"
    num_classes: int = 196
    pretrained: bool = True
    dropout: float = 0.2
    freeze_backbone: bool = False

    # Image & Preprocessing parameters
    image_size: tuple[int, int] = (224, 224)
    crop_bbox: bool = True
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Augmentation parameters
    random_crop_scale: tuple[float, float] = (0.8, 1.0)
    horizontal_flip_p: float = 0.5
    rotation_degrees: float = 15.0
    color_jitter_brightness: float = 0.15
    color_jitter_contrast: float = 0.15
    color_jitter_saturation: float = 0.15
    color_jitter_hue: float = 0.05

    # Training parameters
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # 'adamw', 'adam', 'sgd'
    scheduler: str = "cosine"  # 'cosine', 'step', 'plateau', 'none'
    lr_min: float = 1e-6
    step_size: int = 5
    gamma: float = 0.5
    label_smoothing: float = 0.05
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    seed: int = 42
    max_train_batches: int | None = None
    max_val_batches: int | None = None

    # Hardware & Runtime
    device: str = "auto"
    mixed_precision: bool = True
    num_workers: int = 2

    # Dataset & Split Paths (relative to project root)
    train_split: str = "data/splits/train.csv"
    val_split: str = "data/splits/validation.csv"
    test_split: str = "data/splits/test.csv"
    class_mapping_csv: str = "data/metadata/class_mapping.csv"
    class_mapping_json: str = "data/processed/class_mapping.json"

    # Output paths
    checkpoints_dir: str = "outputs/classification/checkpoints"
    metrics_dir: str = "outputs/classification/metrics"
    plots_dir: str = "outputs/classification/plots"
    failure_cases_dir: str = "outputs/classification/failure_cases"
    predictions_dir: str = "outputs/classification/predictions"
    best_checkpoint_path: str = "outputs/classification/checkpoints/best_checkpoint.pt"
    last_checkpoint_path: str = "outputs/classification/checkpoints/last_checkpoint.pt"

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        d = asdict(self)
        # Convert tuples to lists for JSON compatibility
        d["image_size"] = list(self.image_size)
        d["mean"] = list(self.mean)
        d["std"] = list(self.std)
        d["random_crop_scale"] = list(self.random_crop_scale)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationConfig:
        """Create config from dictionary."""
        cfg_data = dict(data)
        if "image_size" in cfg_data and isinstance(cfg_data["image_size"], list):
            cfg_data["image_size"] = tuple(cfg_data["image_size"])
        if "mean" in cfg_data and isinstance(cfg_data["mean"], list):
            cfg_data["mean"] = tuple(cfg_data["mean"])
        if "std" in cfg_data and isinstance(cfg_data["std"], list):
            cfg_data["std"] = tuple(cfg_data["std"])
        if "random_crop_scale" in cfg_data and isinstance(cfg_data["random_crop_scale"], list):
            cfg_data["random_crop_scale"] = tuple(cfg_data["random_crop_scale"])

        # Filter out unknown keys
        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in cfg_data.items() if k in known_keys}
        return cls(**filtered)

    def save_json(self, output_path: str | Path) -> None:
        """Save configuration as JSON file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, json_path: str | Path) -> ClassificationConfig:
        """Load configuration from JSON file."""
        p = Path(json_path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_yaml(self, output_path: str | Path) -> None:
        """Save configuration as YAML file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load_yaml(cls, yaml_path: str | Path) -> ClassificationConfig:
        """Load configuration from YAML file."""
        p = Path(yaml_path)
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
