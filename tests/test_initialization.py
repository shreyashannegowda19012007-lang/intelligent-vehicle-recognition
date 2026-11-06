"""
Unit tests for Project Initialization (Phase 1).
Validates package structure, configuration loading, device telemetry,
preprocessing functions, PyTorch dataset loading, class mappings,
and dataset validation reports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.annotations import get_class_mapping_dict, get_normalized_annotations_df
from src.data.dataset import StanfordCarsDataset, get_default_transforms
from src.data.preprocessing import (
    apply_clahe,
    bgr_to_rgb,
    compute_image_quality_metrics,
    denoise_bilateral,
    denoise_gaussian,
    detect_edges_canny,
    normalize_image,
    resize_image,
    rgb_to_bgr,
    to_grayscale,
)
from src.utils.config import get_config
from src.utils.device import get_device, get_hardware_telemetry


@pytest.fixture(scope="session")
def workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_package_imports():
    """Verify that all core project packages and submodules import cleanly."""
    import src
    import src.classical_cv
    import src.classification
    import src.data
    import src.data.annotations
    import src.data.dataset
    import src.data.preprocessing
    import src.detection
    import src.evaluation
    import src.hybrid
    import src.utils
    import src.utils.config
    import src.utils.device

    assert src is not None


def test_configuration_loading(workspace_root: Path):
    """Verify loading config.yaml and default dataclass values."""
    cfg = get_config(reload=True)
    assert cfg.name == "P5-Intelligent-Vehicle-Recognition"
    assert cfg.dataset.num_classes == 196
    assert cfg.dataset.num_makes == 49
    assert cfg.dataset.root_dir.is_dir()
    assert cfg.preprocessing.image_size == [224, 224]


def test_hardware_device_detection():
    """Verify CPU/GPU detection utility."""
    dev = get_device("auto")
    assert isinstance(dev, torch.device)
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    else:
        assert dev.type == "cpu"

    telemetry = get_hardware_telemetry()
    assert "os" in telemetry
    assert "python_version" in telemetry
    assert telemetry["ram_total_gb"] > 0
    assert "cuda_available" in telemetry


def test_class_mappings_exist_and_valid(workspace_root: Path):
    """Verify data/processed/class_mapping.json and .csv files."""
    json_p = workspace_root / "data" / "processed" / "class_mapping.json"
    csv_p = workspace_root / "data" / "processed" / "class_mapping.csv"

    assert json_p.is_file(), f"Missing {json_p}"
    assert csv_p.is_file(), f"Missing {csv_p}"

    mapping_dict = get_class_mapping_dict(json_p)
    assert mapping_dict["num_classes"] == 196
    assert len(mapping_dict["classes"]) == 196
    assert "0" in mapping_dict["by_class_id"]

    df = pd.read_csv(csv_p)
    assert len(df) == 196
    required_cols = {"class_id", "class_name", "make", "model", "year"}
    assert required_cols.issubset(df.columns)


def test_normalized_annotations_df(workspace_root: Path):
    """Verify normalized annotations loading and bounding box validity."""
    df = get_normalized_annotations_df()
    assert len(df) == 16185
    assert df["annotation_valid"].all()

    # Coordinates within boundaries
    assert (df["bbox_xmin"] >= 0).all()
    assert (df["bbox_ymin"] >= 0).all()
    assert (df["bbox_xmax"] <= df["image_width"]).all()
    assert (df["bbox_ymax"] <= df["image_height"]).all()


def test_image_preprocessing_pipeline():
    """Verify all 8 OpenCV preprocessing functions on synthetic and edge arrays."""
    # Synthetic RGB image
    img = np.random.randint(0, 256, (120, 180, 3), dtype=np.uint8)

    # 1. Colorspace
    rgb = bgr_to_rgb(img)
    assert rgb.shape == (120, 180, 3)
    bgr = rgb_to_bgr(rgb)
    assert bgr.shape == (120, 180, 3)

    # 2. Grayscale
    gray = to_grayscale(img)
    assert gray.shape == (120, 180)

    # 3. Resize standard & letterbox
    resized = resize_image(img, (224, 224), keep_aspect_ratio=False)
    assert resized.shape == (224, 224, 3)
    letterboxed = resize_image(img, (224, 224), keep_aspect_ratio=True)
    assert letterboxed.shape == (224, 224, 3)

    # 4. Normalization
    norm = normalize_image(img)
    assert norm.max() <= 1.0 and norm.min() >= 0.0

    # 5. CLAHE
    clahe = apply_clahe(img)
    assert clahe.shape == (120, 180, 3)

    # 6. Denoising
    g_denoise = denoise_gaussian(img)
    assert g_denoise.shape == (120, 180, 3)
    b_denoise = denoise_bilateral(img)
    assert b_denoise.shape == (120, 180, 3)

    # 7. Canny Edges
    edges = detect_edges_canny(img)
    assert edges.shape == (120, 180)
    assert set(np.unique(edges)).issubset({0, 255})

    # 8. Quality metrics
    metrics = compute_image_quality_metrics(img)
    assert "mean_brightness" in metrics
    assert "contrast_std" in metrics
    assert "sharpness_laplacian_var" in metrics
    assert "is_blurry" in metrics


def test_pytorch_dataset_sample_loading(workspace_root: Path):
    """Verify PyTorch dataset and transforms with bounding box crop."""
    df = get_normalized_annotations_df()
    train_records = df[df["split"] == "train"].head(10).to_dict(orient="records")

    train_tf, _ = get_default_transforms(image_size=224)
    ds = StanfordCarsDataset(
        annotations=train_records,
        transform=train_tf,
        crop_bbox=True,
        target_type="multi",
    )

    assert len(ds) == 10
    img_tensor, targets = ds[0]

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 224, 224)
    assert "class_id" in targets
    assert "bbox" in targets
    assert targets["class_id"].dtype == torch.long


def test_validation_report_artifacts(workspace_root: Path):
    """Verify that dataset validation artifacts are generated on disk."""
    json_report = workspace_root / "data" / "reports" / "dataset_validation_report.json"
    md_report = workspace_root / "docs" / "dataset_validation_report.md"

    assert json_report.is_file(), f"Missing {json_report}"
    assert md_report.is_file(), f"Missing {md_report}"

    with open(json_report, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "VALID"
    assert data["dataset"]["total_images"] == 16185
    assert data["annotations"]["all_bounding_boxes_valid"] is True
    assert data["taxonomy"]["num_classes"] == 196
