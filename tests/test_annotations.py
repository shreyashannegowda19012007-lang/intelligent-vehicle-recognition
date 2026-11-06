"""
Automated unit tests for annotations, manifests, and YOLO coordinate conversion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.annotation_parser import ALL_49_MAKES
from src.data.manifest import load_manifest_df
from src.utils.box_utils import box_to_yolo, yolo_to_box
from src.utils.config import find_project_root


@pytest.fixture(scope="session")
def project_root() -> Path:
    return find_project_root()


@pytest.fixture(scope="session")
def manifest_df(project_root: Path) -> pd.DataFrame:
    manifest_csv = project_root / "data" / "processed" / "metadata" / "vehicle_manifest.csv"
    assert manifest_csv.is_file(), f"Manifest CSV not found: {manifest_csv}"
    return load_manifest_df(manifest_csv)


def test_manifest_schema_and_counts(manifest_df: pd.DataFrame):
    assert len(manifest_df) == 16185
    expected_cols = {
        "image_path",
        "image_id",
        "split",
        "class_id",
        "class_name",
        "make",
        "model",
        "year",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "image_width",
        "image_height",
        "is_valid",
    }
    assert expected_cols.issubset(set(manifest_df.columns))


def test_manifest_train_vs_test_labels(manifest_df: pd.DataFrame):
    train_df = manifest_df[manifest_df["split"] == "train"]
    test_df = manifest_df[manifest_df["split"] == "test"]

    assert len(train_df) == 8144
    assert len(test_df) == 8041

    # Train must have complete labels
    assert train_df["class_id"].notna().all()
    assert train_df["make"].notna().all()
    assert train_df["year"].notna().all()

    # Train class_id in range [0, 195]
    assert int(train_df["class_id"].min()) == 0
    assert int(train_df["class_id"].max()) == 195
    assert train_df["class_id"].nunique() == 196

    # Test set must have None / NaN labels (competition protocol)
    assert test_df["class_id"].isna().all()


def test_bounding_box_geometry(manifest_df: pd.DataFrame):
    x1 = manifest_df["bbox_x1"]
    y1 = manifest_df["bbox_y1"]
    x2 = manifest_df["bbox_x2"]
    y2 = manifest_df["bbox_y2"]
    w = manifest_df["image_width"]
    h = manifest_df["image_height"]

    assert (x1 >= 0).all(), "Found negative bbox_x1"
    assert (y1 >= 0).all(), "Found negative bbox_y1"
    assert (x2 > x1).all(), "Found bbox_x2 <= bbox_x1"
    assert (y2 > y1).all(), "Found bbox_y2 <= bbox_y1"
    assert (x2 <= w).all(), "Found bbox_x2 > image_width"
    assert (y2 <= h).all(), "Found bbox_y2 > image_height"


def test_box_to_yolo_conversion():
    # Test typical coordinate conversion
    img_w, img_h = 640, 480
    x1, y1, x2, y2 = 100, 100, 500, 300
    xc, yc, w, h = box_to_yolo(x1, y1, x2, y2, img_w, img_h, inclusive=True)

    assert 0.0 <= xc <= 1.0
    assert 0.0 <= yc <= 1.0
    assert 0.0 < w <= 1.0
    assert 0.0 < h <= 1.0

    # Invert back
    inv_x1, inv_y1, inv_x2, inv_y2 = yolo_to_box(xc, yc, w, h, img_w, img_h, inclusive=True)
    assert abs(inv_x1 - x1) <= 1
    assert abs(inv_y1 - y1) <= 1
    assert abs(inv_x2 - x2) <= 1
    assert abs(inv_y2 - y2) <= 1


def test_yolo_dataset_yaml(project_root: Path):
    yaml_p = project_root / "data" / "processed" / "detection" / "dataset.yaml"
    assert yaml_p.is_file(), f"YOLO dataset.yaml not found: {yaml_p}"

    with open(yaml_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "train" in cfg
    assert "val" in cfg
    assert "names" in cfg
    assert cfg["names"][0] == "vehicle"
    assert cfg["nc"] == 1


def test_taxonomy_manufacturers(manifest_df: pd.DataFrame):
    train_makes = set(manifest_df[manifest_df["split"] == "train"]["make"].dropna().unique())
    all_makes = set(ALL_49_MAKES)
    assert len(train_makes) == 49
    assert train_makes == all_makes
