"""
Automated unit tests for dataset split generation and integrity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.utils.config import find_project_root


@pytest.fixture(scope="session")
def project_root() -> Path:
    return find_project_root()


@pytest.fixture(scope="session")
def splits_data(project_root: Path) -> dict[str, pd.DataFrame]:
    splits_dir = project_root / "data" / "splits"
    return {
        "train": pd.read_csv(splits_dir / "train.csv"),
        "val": pd.read_csv(splits_dir / "validation.csv"),
        "test": pd.read_csv(splits_dir / "test.csv"),
    }


def test_split_files_existence(project_root: Path):
    splits_dir = project_root / "data" / "splits"
    assert (splits_dir / "train.csv").is_file()
    assert (splits_dir / "validation.csv").is_file()
    assert (splits_dir / "test.csv").is_file()
    assert (splits_dir / "split_summary.json").is_file()


def test_split_counts(splits_data: dict[str, pd.DataFrame]):
    train_df = splits_data["train"]
    val_df = splits_data["val"]
    test_df = splits_data["test"]

    assert len(train_df) == 6922
    assert len(val_df) == 1222
    assert len(test_df) == 8041
    assert len(train_df) + len(val_df) == 8144
    assert len(train_df) + len(val_df) + len(test_df) == 16185


def test_no_data_leakage(splits_data: dict[str, pd.DataFrame]):
    train_ids = set(splits_data["train"]["image_id"])
    val_ids = set(splits_data["val"]["image_id"])
    test_ids = set(splits_data["test"]["image_id"])

    # Disjoint assertions
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage between train and validation"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage between train and test"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage between validation and test"


def test_stratified_class_coverage(splits_data: dict[str, pd.DataFrame]):
    train_df = splits_data["train"]
    val_df = splits_data["val"]

    train_classes = set(train_df["class_id"].unique())
    val_classes = set(val_df["class_id"].unique())

    assert len(train_classes) == 196, f"Expected 196 classes in train, got {len(train_classes)}"
    assert len(val_classes) == 196, f"Expected 196 classes in val, got {len(val_classes)}"


def test_split_summary_json(project_root: Path):
    summary_path = project_root / "data" / "splits" / "split_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["status"] == "SUCCESS"
    assert summary["random_seed"] == 42
    assert summary["counts"]["train"] == 6922
    assert summary["counts"]["validation"] == 1222
    assert summary["counts"]["test"] == 8041
    assert summary["class_coverage"]["all_classes_in_train"] is True
    assert summary["class_coverage"]["all_classes_in_val"] is True
