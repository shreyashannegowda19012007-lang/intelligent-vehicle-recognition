"""
Central configuration management module for P5 Vehicle Recognition.
Loads settings from configs/config.yaml with dynamic path resolution
and environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def find_project_root(anchor_file: str = "configs/config.yaml") -> Path:
    """Find repository root by ascending until anchor_file or .git is found."""
    current = Path(__file__).resolve().parent
    for _ in range(8):
        if (current / anchor_file).exists() or (current / ".git").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    return Path.cwd()


@dataclass
class DatasetConfig:
    name: str = "Stanford Cars (Cars196)"
    root_dir: Path = field(default_factory=Path)
    raw_dir: Path = field(default_factory=Path)
    processed_dir: Path = field(default_factory=Path)
    metadata_dir: Path = field(default_factory=Path)
    reports_dir: Path = field(default_factory=Path)
    manifest_csv: Path = field(default_factory=Path)
    manifest_json: Path = field(default_factory=Path)
    splits_dir: Path = field(default_factory=Path)
    train_split: Path = field(default_factory=Path)
    val_split: Path = field(default_factory=Path)
    test_split: Path = field(default_factory=Path)
    detection_dir: Path = field(default_factory=Path)
    detection_yaml: Path = field(default_factory=Path)
    num_classes: int = 196
    num_makes: int = 49
    expected_train_images: int = 8144
    expected_test_images: int = 8041
    expected_total_images: int = 16185
    train_split_ratio: float = 0.85
    val_split_ratio: float = 0.15


@dataclass
class AugmentationConfig:
    random_horizontal_flip_p: float = 0.5
    rotation_degrees: int = 15
    random_crop_scale: list[float] = field(default_factory=lambda: [0.8, 1.0])
    color_jitter: dict[str, float] = field(
        default_factory=lambda: {
            "brightness": 0.15,
            "contrast": 0.15,
            "saturation": 0.15,
            "hue": 0.05,
        }
    )
    gaussian_blur_p: float = 0.2
    gaussian_noise_p: float = 0.1
    perspective_p: float = 0.1


@dataclass
class PreprocessingConfig:
    image_size: list[int] = field(default_factory=lambda: [224, 224])
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: list[int] = field(default_factory=lambda: [8, 8])
    canny_low_thresh: int = 50
    canny_high_thresh: int = 150


@dataclass
class HardwareConfig:
    device: str = "auto"
    mixed_precision: bool = True
    num_workers: int = 2
    batch_size: int = 32
    max_vram_gb: float = 5.2


@dataclass
class InferenceConfig:
    detection_conf_thresh: float = 0.45
    classification_conf_thresh: float = 0.50
    rejection_thresh: float = 0.45
    temperature: float = 1.2


@dataclass
class ProjectConfig:
    name: str = "P5-Intelligent-Vehicle-Recognition"
    version: str = "1.0.0"
    seed: int = 42
    project_root: Path = field(default_factory=Path)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def load_config(config_path: Path | str | None = None) -> ProjectConfig:
    """
    Load project configuration from YAML file and environment variables.
    """
    root = find_project_root()
    if config_path is None:
        config_path = root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    raw_cfg: dict[str, Any] = {}
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

    proj = raw_cfg.get("project", {})
    ds = raw_cfg.get("dataset", {})
    prep = raw_cfg.get("preprocessing", {})
    aug = raw_cfg.get("augmentation", {})
    hw = raw_cfg.get("hardware", {})
    inf = raw_cfg.get("inference", {})

    # Resolve dataset root with env var priority
    env_root = os.environ.get("STANFORD_CARS_ROOT") or os.environ.get("DATASET_ROOT")
    if env_root:
        dataset_root = Path(env_root).resolve()
    elif "root_dir" in ds:
        p = Path(ds["root_dir"])
        dataset_root = p if p.is_absolute() else (root / p).resolve()
    else:
        dataset_root = (root / "Stanford Dataset").resolve()

    dataset_cfg = DatasetConfig(
        name=ds.get("name", "Stanford Cars (Cars196)"),
        root_dir=dataset_root,
        raw_dir=dataset_root,
        processed_dir=(root / ds.get("processed_dir", "data/processed")).resolve(),
        metadata_dir=(root / ds.get("metadata_dir", "data/metadata")).resolve(),
        reports_dir=(root / ds.get("reports_dir", "data/reports")).resolve(),
        manifest_csv=(root / ds.get("manifest_csv", "data/processed/metadata/vehicle_manifest.csv")).resolve(),
        manifest_json=(root / ds.get("manifest_json", "data/processed/metadata/vehicle_manifest.json")).resolve(),
        splits_dir=(root / ds.get("splits_dir", "data/splits")).resolve(),
        train_split=(root / ds.get("train_split", "data/splits/train.csv")).resolve(),
        val_split=(root / ds.get("val_split", "data/splits/validation.csv")).resolve(),
        test_split=(root / ds.get("test_split", "data/splits/test.csv")).resolve(),
        detection_dir=(root / ds.get("detection_dir", "data/processed/detection")).resolve(),
        detection_yaml=(root / ds.get("detection_yaml", "data/processed/detection/dataset.yaml")).resolve(),
        num_classes=int(ds.get("num_classes", 196)),
        num_makes=int(ds.get("num_makes", 49)),
        expected_train_images=int(ds.get("expected_train_images", 8144)),
        expected_test_images=int(ds.get("expected_test_images", 8041)),
        expected_total_images=int(ds.get("expected_total_images", 16185)),
        train_split_ratio=float(ds.get("train_split_ratio", 0.85)),
        val_split_ratio=float(ds.get("val_split_ratio", 0.15)),
    )

    preprocessing_cfg = PreprocessingConfig(
        image_size=prep.get("image_size", [224, 224]),
        mean=prep.get("mean", [0.485, 0.456, 0.406]),
        std=prep.get("std", [0.229, 0.224, 0.225]),
        clahe_clip_limit=float(prep.get("clahe_clip_limit", 2.0)),
        clahe_tile_grid=prep.get("clahe_tile_grid", [8, 8]),
        canny_low_thresh=int(prep.get("canny_low_thresh", 50)),
        canny_high_thresh=int(prep.get("canny_high_thresh", 150)),
    )

    augmentation_cfg = AugmentationConfig(
        random_horizontal_flip_p=float(aug.get("random_horizontal_flip_p", 0.5)),
        rotation_degrees=int(aug.get("rotation_degrees", 15)),
        random_crop_scale=aug.get("random_crop_scale", [0.8, 1.0]),
        color_jitter=aug.get(
            "color_jitter",
            {"brightness": 0.15, "contrast": 0.15, "saturation": 0.15, "hue": 0.05},
        ),
        gaussian_blur_p=float(aug.get("gaussian_blur_p", 0.2)),
        gaussian_noise_p=float(aug.get("gaussian_noise_p", 0.1)),
        perspective_p=float(aug.get("perspective_p", 0.1)),
    )

    hardware_cfg = HardwareConfig(
        device=hw.get("device", "auto"),
        mixed_precision=bool(hw.get("mixed_precision", True)),
        num_workers=int(hw.get("num_workers", 2)),
        batch_size=int(hw.get("batch_size", 32)),
        max_vram_gb=float(hw.get("max_vram_gb", 5.2)),
    )

    inference_cfg = InferenceConfig(
        detection_conf_thresh=float(inf.get("detection_conf_thresh", 0.45)),
        classification_conf_thresh=float(inf.get("classification_conf_thresh", 0.50)),
        rejection_thresh=float(inf.get("rejection_thresh", 0.45)),
        temperature=float(inf.get("temperature", 1.2)),
    )

    return ProjectConfig(
        name=proj.get("name", "P5-Intelligent-Vehicle-Recognition"),
        version=proj.get("version", "1.0.0"),
        seed=int(proj.get("seed", 42)),
        project_root=root,
        dataset=dataset_cfg,
        preprocessing=preprocessing_cfg,
        augmentation=augmentation_cfg,
        hardware=hardware_cfg,
        inference=inference_cfg,
    )


# Singleton instance accessor
_CONFIG_CACHE: ProjectConfig | None = None


def get_config(reload: bool = False) -> ProjectConfig:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None or reload:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE
