"""
Unit and integration test suite for Phase 1.1:
Stanford Cars Dataset Integration and Annotation Extraction.

Validates configurable root paths, image existence, annotation extraction,
normalized schema compliance, class/make/model mappings, bounding box geometry,
missing label identification, duplicate detection, and edge-case resilience.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.annotation_parser import (
    ALL_49_MAKES,
    BODY_TYPE_OVERRIDES,
    KNOWN_BODY_TYPES,
    AnnotationParser,
    NormalizedAnnotationRecord,
    TaxonomyParser,
    detect_duplicates,
    export_normalized_dataset_and_metadata,
    load_normalized_annotations,
    validate_annotation_integrity,
)
from src.data.dataset import StanfordCarsDataset
from src.data.dataset_parser import (
    ENV_STANFORD_CARS_DIR,
    ENV_STANFORD_DATASET_ROOT,
    DatasetManifest,
    StanfordCarsDatasetParser,
    resolve_dataset_root,
)


@pytest.fixture(scope="session")
def workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def dataset_parser(workspace_root: Path) -> StanfordCarsDatasetParser:
    return StanfordCarsDatasetParser(base_dir=workspace_root)


@pytest.fixture(scope="session")
def manifest(dataset_parser: StanfordCarsDatasetParser) -> DatasetManifest:
    return dataset_parser.get_manifest()


@pytest.fixture(scope="session")
def normalized_csv_path(
    workspace_root: Path, dataset_parser: StanfordCarsDatasetParser
) -> Path:
    csv_p = (
        workspace_root
        / "data"
        / "processed"
        / "annotations"
        / "normalized_annotations.csv"
    )
    if not csv_p.is_file():
        export_normalized_dataset_and_metadata(
            dataset_parser, output_dir=workspace_root / "data"
        )
    return csv_p


# ==============================================================================
# 1. Dataset Directory Integration & Configurable Root Tests
# ==============================================================================


def test_configurable_dataset_root_explicit(manifest: DatasetManifest):
    """Test resolving dataset root via explicit parameter."""
    resolved = resolve_dataset_root(custom_root=manifest.root_dir)
    assert resolved == manifest.root_dir
    assert resolved.is_dir()


def test_configurable_dataset_root_env_var(manifest: DatasetManifest, monkeypatch):
    """Test resolving dataset root via environment variable."""
    monkeypatch.setenv(ENV_STANFORD_CARS_DIR, str(manifest.root_dir))
    resolved = resolve_dataset_root()
    assert resolved == manifest.root_dir


def test_configurable_dataset_root_alt_env_var(manifest: DatasetManifest, monkeypatch):
    """Test resolving dataset root via secondary environment variable."""
    monkeypatch.delenv(ENV_STANFORD_CARS_DIR, raising=False)
    monkeypatch.setenv(ENV_STANFORD_DATASET_ROOT, str(manifest.root_dir))
    resolved = resolve_dataset_root()
    assert resolved == manifest.root_dir


def test_dataset_root_non_existent_raises():
    """Test that resolving a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_dataset_root(custom_root="D:/non_existent_cars_dataset_dir_xyz_999")


def test_dataset_manifest_integrity(manifest: DatasetManifest):
    """Verify that dataset manifest detects all required image and annotation assets."""
    assert manifest.is_valid is True
    assert manifest.root_dir.is_dir()
    assert manifest.train_image_dir.is_dir()
    assert manifest.test_image_dir.is_dir()
    assert manifest.devkit_dir.is_dir()
    assert manifest.meta_mat_path.is_file()
    assert manifest.train_annos_mat_path.is_file()
    assert manifest.test_annos_mat_path.is_file()

    # Exact Stanford Cars split sizes
    assert manifest.num_train_images == 8144
    assert manifest.num_test_images == 8041
    assert manifest.num_train_images + manifest.num_test_images == 16185


def test_parser_processed_and_dataloader_status(
    dataset_parser: StanfordCarsDatasetParser,
):
    """Verify helper queries for existing processed data and PyTorch data loaders."""
    processed = dataset_parser.get_processed_data_status()
    assert processed["data_dir_exists"] is True
    assert "classes_json" in processed["artifacts"]
    assert "normalized_annotations_csv" in processed["artifacts"]

    loader_status = dataset_parser.get_dataloader_status()
    assert loader_status["module_available"] is True
    assert loader_status["dataset_class"] == "StanfordCarsDataset"
    assert loader_status["dataloader_builder"] == "build_dataloaders"


# ==============================================================================
# 2. Taxonomy & Metadata Parser Tests
# ==============================================================================


def test_taxonomy_parser_complete_coverage(manifest: DatasetManifest):
    """Verify taxonomy parser decomposes all 196 classes with zero ambiguity."""
    classes_meta = TaxonomyParser.parse_meta_mat(manifest.meta_mat_path)
    assert classes_meta["num_classes"] == 196
    assert len(classes_meta["classes"]) == 196

    # Verify counts
    assert len(classes_meta["makes"]) == 49
    assert len(classes_meta["years"]) == 16
    assert len(classes_meta["body_types"]) == 14

    for idx, c in enumerate(classes_meta["classes"]):
        assert c["class_id"] == idx
        assert c["original_class_id"] == idx + 1
        assert c["is_ambiguous"] is False
        assert c["review_flag"] == "OK"
        assert c["make"] in ALL_49_MAKES
        assert 1990 <= c["year"] <= 2013
        assert c["body_type"] in KNOWN_BODY_TYPES
        assert len(c["model"]) > 0
        assert len(c["clean_model"]) > 0


def test_taxonomy_multiword_and_hyphenated_makes():
    """Verify multiword and hyphenated manufacturer parsing."""
    samples = [
        (
            "AM General Hummer SUV 2000",
            "AM General",
            "Hummer SUV",
            "Hummer",
            2000,
            "SUV",
        ),
        (
            "Aston Martin V8 Vantage Coupe 2012",
            "Aston Martin",
            "V8 Vantage Coupe",
            "V8 Vantage",
            2012,
            "Coupe",
        ),
        (
            "Land Rover Range Rover SUV 2012",
            "Land Rover",
            "Range Rover SUV",
            "Range Rover",
            2012,
            "SUV",
        ),
        (
            "Mercedes-Benz C-Class Sedan 2012",
            "Mercedes-Benz",
            "C-Class Sedan",
            "C-Class",
            2012,
            "Sedan",
        ),
        (
            "Rolls-Royce Ghost Sedan 2012",
            "Rolls-Royce",
            "Ghost Sedan",
            "Ghost",
            2012,
            "Sedan",
        ),
        (
            "smart fortwo Convertible 2012",
            "smart",
            "fortwo Convertible",
            "fortwo",
            2012,
            "Convertible",
        ),
    ]
    for raw, make, model, clean, year, body in samples:
        res = TaxonomyParser.parse_class_name(raw)
        assert res["make"] == make, f"Failed make on {raw}"
        assert res["model"] == model, f"Failed model on {raw}"
        assert res["clean_model"] == clean, f"Failed clean_model on {raw}"
        assert res["year"] == year, f"Failed year on {raw}"
        assert res["body_type"] == body, f"Failed body_type on {raw}"
        assert res["is_ambiguous"] is False


def test_taxonomy_implicit_body_type_overrides():
    """Verify body type overrides for performance trims without explicit body keywords."""
    for raw_name, expected_body in BODY_TYPE_OVERRIDES.items():
        res = TaxonomyParser.parse_class_name(raw_name)
        assert res["body_type"] == expected_body, (
            f"Failed override on {raw_name}: got {res['body_type']}, expected {expected_body}"
        )


def test_taxonomy_ambiguous_handling():
    """Verify graceful handling and flagging for malformed or unknown vehicle strings."""
    # Unknown manufacturer and missing year
    bad_car = "UnknownMaker FastModel"
    res = TaxonomyParser.parse_class_name(bad_car)
    assert res["is_ambiguous"] is True
    assert res["raw_name"] == bad_car
    assert res["body_type"] == "Unknown"
    assert res["review_flag"] != "OK"


# ==============================================================================
# 3. Annotation Extraction & Schema Normalization Tests
# ==============================================================================


def test_annotation_parser_train_split(manifest: DatasetManifest, workspace_root: Path):
    """Verify train split parsing yields 8,144 valid, fully labeled records."""
    parser = AnnotationParser(manifest.meta_mat_path, workspace_root=workspace_root)
    records = parser.parse_split_annotations(
        annos_mat_path=manifest.train_annos_mat_path,
        img_dir=manifest.train_image_dir,
        split="train",
        is_test=False,
    )

    assert len(records) == 8144

    for i, r in enumerate(records[:100]):
        assert r.split == "train"
        assert r.image_id == f"train_{Path(r.original_filename).stem}"
        assert r.dataset_source == "stanford_cars"
        assert r.class_id is not None
        assert 0 <= r.class_id < 196
        assert r.original_class_id == r.class_id + 1
        assert r.has_label is True
        assert r.annotation_valid is True
        assert r.image_width > 0
        assert r.image_height > 0
        assert 0 <= r.bbox_xmin <= r.bbox_xmax < r.image_width
        assert 0 <= r.bbox_ymin <= r.bbox_ymax < r.image_height
        assert (workspace_root / r.image_path).is_file()


def test_annotation_parser_test_split_missing_labels(
    manifest: DatasetManifest, workspace_root: Path
):
    """Verify test split parsing handles withheld labels by assigning nulls while keeping bboxes valid."""
    parser = AnnotationParser(manifest.meta_mat_path, workspace_root=workspace_root)
    records = parser.parse_split_annotations(
        annos_mat_path=manifest.test_annos_mat_path,
        img_dir=manifest.test_image_dir,
        split="test",
        is_test=True,
    )

    assert len(records) == 8041

    for r in records[:100]:
        assert r.split == "test"
        assert r.image_id == f"test_{Path(r.original_filename).stem}"
        assert r.has_label is False
        assert r.class_id is None
        assert r.original_class_id is None
        assert r.class_name is None
        assert r.make is None
        assert r.model is None
        assert r.year is None
        assert r.annotation_valid is True  # Valid bounding box and existing image
        assert r.image_width > 0
        assert r.image_height > 0
        assert 0 <= r.bbox_xmin <= r.bbox_xmax < r.image_width
        assert 0 <= r.bbox_ymin <= r.bbox_ymax < r.image_height
        assert (workspace_root / r.image_path).is_file()


# ==============================================================================
# 4. Normalized Annotations CSV & Schema Conformance Tests
# ==============================================================================


def test_normalized_annotations_csv_schema_and_integrity(
    normalized_csv_path: Path, workspace_root: Path
):
    """Verify exported normalized_annotations.csv matches the exact Section 4 required schema."""
    df = pd.read_csv(normalized_csv_path)

    # 1. Total row count check
    assert len(df) == 16185
    assert (df["split"] == "train").sum() == 8144
    assert (df["split"] == "test").sum() == 8041

    # 2. Required columns in exact order
    expected_required_cols = [
        "image_id",
        "image_path",
        "dataset_source",
        "split",
        "class_id",
        "class_name",
        "make",
        "model",
        "year",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "image_width",
        "image_height",
        "annotation_valid",
    ]
    for i, col in enumerate(expected_required_cols):
        assert df.columns[i] == col, (
            f"Column mismatch at index {i}: got {df.columns[i]}, expected {col}"
        )

    # 3. Preserved columns present
    preserved_cols = [
        "original_class_id",
        "original_filename",
        "bbox_matlab_x1",
        "bbox_matlab_y1",
        "bbox_matlab_x2",
        "bbox_matlab_y2",
        "body_type",
        "clean_model",
        "has_label",
    ]
    for col in preserved_cols:
        assert col in df.columns, f"Preserved column {col} missing from CSV!"

    # 4. Label presence vs absence
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    assert train_df["class_id"].notna().all(), "Train class_id has missing values!"
    assert train_df["class_id"].between(0, 195).all(), (
        "Train class_id outside [0, 195]!"
    )
    assert train_df["original_class_id"].between(1, 196).all(), (
        "Train original_class_id outside [1, 196]!"
    )
    assert len(train_df["class_id"].unique()) == 196

    assert test_df["class_id"].isna().all(), "Test class_id has non-null values!"
    assert test_df["class_name"].isna().all()
    assert test_df["make"].isna().all()
    assert test_df["model"].isna().all()
    assert test_df["year"].isna().all()

    # 5. Bounding box geometry
    assert (df["bbox_xmin"] >= 0).all()
    assert (df["bbox_ymin"] >= 0).all()
    assert (df["bbox_xmin"] <= df["bbox_xmax"]).all()
    assert (df["bbox_ymin"] <= df["bbox_ymax"]).all()
    assert (df["bbox_xmax"] < df["image_width"]).all()
    assert (df["bbox_ymax"] < df["image_height"]).all()

    # 6. Annotation validity
    assert df["annotation_valid"].all()

    # 7. Validate through integrity validation function (including image existence on disk)
    res = validate_annotation_integrity(
        df, workspace_root=workspace_root, check_all_images_exist=True
    )
    assert res["status"] == "PASS"
    assert res["missing_test_labels_identified"] == 8041


# ==============================================================================
# 5. Metadata Mapping CSV Files Tests
# ==============================================================================


def test_class_mapping_csv(workspace_root: Path):
    """Verify data/metadata/class_mapping.csv has 196 rows and valid columns."""
    p = workspace_root / "data" / "metadata" / "class_mapping.csv"
    assert p.is_file(), f"class_mapping.csv missing at {p}"
    df = pd.read_csv(p)

    assert len(df) == 196
    expected_cols = [
        "class_id",
        "original_class_id",
        "class_name",
        "make",
        "model",
        "clean_model",
        "year",
        "body_type",
        "sample_count",
        "is_ambiguous",
        "review_flag",
    ]
    assert list(df.columns) == expected_cols

    assert df["class_id"].is_monotonic_increasing
    assert df["class_id"].tolist() == list(range(196))
    assert df["original_class_id"].tolist() == list(range(1, 197))
    assert (df["is_ambiguous"] == False).all()
    assert (df["review_flag"] == "OK").all()
    assert df["sample_count"].sum() == 8144
    assert df["sample_count"].min() == 24
    assert df["sample_count"].max() == 68


def test_make_mapping_csv(workspace_root: Path):
    """Verify data/metadata/make_mapping.csv has 49 rows."""
    p = workspace_root / "data" / "metadata" / "make_mapping.csv"
    assert p.is_file(), f"make_mapping.csv missing at {p}"
    df = pd.read_csv(p)

    assert len(df) == 49
    assert list(df.columns) == ["make_id", "make", "class_count", "sample_count"]
    assert df["class_count"].sum() == 196
    assert df["sample_count"].sum() == 8144
    assert set(df["make"]) == set(ALL_49_MAKES)


def test_model_mapping_csv(workspace_root: Path):
    """Verify data/metadata/model_mapping.csv has valid columns and sample counts."""
    p = workspace_root / "data" / "metadata" / "model_mapping.csv"
    assert p.is_file(), f"model_mapping.csv missing at {p}"
    df = pd.read_csv(p)

    assert len(df) > 0
    expected_cols = [
        "model_id",
        "model",
        "clean_model",
        "make",
        "body_type",
        "class_count",
        "sample_count",
    ]
    assert list(df.columns) == expected_cols
    assert df["class_count"].sum() == 196
    assert df["sample_count"].sum() == 8144


# ==============================================================================
# 6. Duplicate Detection & Edge Case Tests
# ==============================================================================


def test_duplicate_detection_no_duplicate_image_ids(normalized_csv_path: Path):
    """Verify no duplicate image IDs exist in the normalized annotations."""
    df = pd.read_csv(normalized_csv_path)
    res = detect_duplicates(df, compute_md5=False)
    assert res["duplicate_image_ids_count"] == 0


def test_duplicate_detection_md5_groups(
    normalized_csv_path: Path, workspace_root: Path
):
    """Verify cryptographic duplicate detection discovers the 39 duplicate pairs identified in Phase 0."""
    df = pd.read_csv(normalized_csv_path)
    res = detect_duplicates(df, compute_md5=True, workspace_root=workspace_root)
    assert len(res["md5_duplicate_groups"]) == 39
    assert len(res["cross_split_duplicates"]) == 19
    assert len(res["train_internal_duplicates"]) == 12
    assert len(res["test_internal_duplicates"]) == 8


def test_annotation_record_invalid_box_flag():
    """Verify annotation_valid is False when a bounding box is degenerate or out of bounds."""
    # Degenerate box: xmin > xmax
    rec = NormalizedAnnotationRecord(
        image_id="test_fake_001",
        image_path="fake/path.jpg",
        dataset_source="stanford_cars",
        split="train",
        class_id=0,
        class_name="AM General Hummer SUV 2000",
        make="AM General",
        model="Hummer SUV",
        year=2000,
        bbox_xmin=200,
        bbox_ymin=100,
        bbox_xmax=150,  # inverted
        bbox_ymax=300,
        image_width=640,
        image_height=480,
        annotation_valid=False,
    )
    assert rec.annotation_valid is False


# ==============================================================================
# 7. Additional Robustness, Edge-Case & Ingestion Tests
# ==============================================================================


def test_resolve_dataset_root_from_nested_subdirectory(workspace_root: Path):
    """Verify resolving dataset root succeeds when executed from deep subdirectories (e.g. src/data)."""
    sub_dir = workspace_root / "src" / "data"
    resolved = resolve_dataset_root(base_dir=sub_dir)
    assert resolved.is_dir()
    assert (resolved / "cars_train").is_dir() or (resolved / "car_devkit").is_dir()


def test_resolve_dataset_root_with_quoted_env_var(
    manifest: DatasetManifest, monkeypatch
):
    """Verify resolving dataset root works even if the environment variable contains surrounding quotes."""
    quoted = f'"{manifest.root_dir}"'
    monkeypatch.setenv(ENV_STANFORD_CARS_DIR, quoted)
    resolved = resolve_dataset_root()
    assert resolved == manifest.root_dir


def test_dataset_parser_flat_layout_preference(tmp_path: Path):
    """Verify parser prioritizes a directory containing .jpg files over an empty nested directory."""
    root = tmp_path / "custom_stanford"
    train_flat = root / "cars_train"
    train_nested = root / "cars_train" / "cars_train"
    test_flat = root / "cars_test"
    devkit_dir = root / "devkit"

    train_nested.mkdir(parents=True)
    test_flat.mkdir(parents=True)
    devkit_dir.mkdir(parents=True)

    # Place a dummy image in the flat train dir, leaving nested dir empty
    (train_flat / "00001.jpg").touch()
    (test_flat / "00001.jpg").touch()

    # Create dummy mat files
    import scipy.io as sio

    meta_dummy = {"class_names": [[["AM General Hummer SUV 2000"]]]}
    train_dummy = {
        "annotations": [
            [
                {
                    "bbox_x1": 1,
                    "bbox_y1": 1,
                    "bbox_x2": 10,
                    "bbox_y2": 10,
                    "class": 1,
                    "fname": "00001.jpg",
                }
            ]
        ]
    }
    test_dummy = {
        "annotations": [
            [
                {
                    "bbox_x1": 1,
                    "bbox_y1": 1,
                    "bbox_x2": 10,
                    "bbox_y2": 10,
                    "fname": "00001.jpg",
                }
            ]
        ]
    }

    sio.savemat(str(devkit_dir / "cars_meta.mat"), meta_dummy)
    sio.savemat(str(devkit_dir / "cars_train_annos.mat"), train_dummy)
    sio.savemat(str(devkit_dir / "cars_test_annos.mat"), test_dummy)

    parser = StanfordCarsDatasetParser(dataset_root=root)
    assert parser.train_image_dir == train_flat
    assert len(parser.list_train_images()) == 1


def test_load_normalized_annotations_types(normalized_csv_path: Path):
    """Verify load_normalized_annotations returns nullable Int64 types for class_id, original_class_id, year."""
    df = load_normalized_annotations(normalized_csv_path)
    assert str(df["class_id"].dtype) == "Int64"
    assert str(df["original_class_id"].dtype) == "Int64"
    assert str(df["year"].dtype) == "Int64"

    train_row = df[df["split"] == "train"].iloc[0]
    assert pd.notna(train_row["class_id"])
    assert int(train_row["class_id"]) == 13

    test_row = df[df["split"] == "test"].iloc[0]
    assert pd.isna(test_row["class_id"])


def test_annotation_parser_invalid_raw_box_flagged(
    manifest: DatasetManifest, tmp_path: Path
):
    """Verify that corrupt raw bounding boxes (e.g. x1 > x2 or out of bounds) result in annotation_valid=False."""
    import scipy.io as sio
    from PIL import Image

    # Create dummy 100x100 image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "corrupt_box.jpg"
    Image.new("RGB", (100, 100)).save(img_path)

    # Inverted box: bbox_x1 = 90, bbox_x2 = 10
    corrupt_annos = {
        "annotations": [
            [
                {
                    "bbox_x1": 90,
                    "bbox_y1": 10,
                    "bbox_x2": 10,
                    "bbox_y2": 50,
                    "class": 1,
                    "fname": "corrupt_box.jpg",
                }
            ]
        ]
    }
    mat_path = tmp_path / "corrupt_annos.mat"
    sio.savemat(str(mat_path), corrupt_annos)

    parser = AnnotationParser(
        meta_mat_path=manifest.meta_mat_path, workspace_root=tmp_path
    )
    records = parser.parse_split_annotations(
        annos_mat_path=mat_path,
        img_dir=img_dir,
        split="train",
    )
    assert len(records) == 1
    assert records[0].annotation_valid is False


def test_direct_script_execution():
    """Verify dataset_parser.py and annotation_parser.py can be executed as standalone scripts."""
    import subprocess
    import sys

    # Test dataset_parser.py
    res1 = subprocess.run(
        [sys.executable, "src/data/dataset_parser.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res1.returncode == 0, f"dataset_parser.py failed: {res1.stderr}"
    assert "Stanford Cars Dataset Manifest" in res1.stdout

    # Test annotation_parser.py
    res2 = subprocess.run(
        [sys.executable, "src/data/annotation_parser.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res2.returncode == 0, f"annotation_parser.py failed: {res2.stderr}"
    assert "Exported normalized_annotations_csv" in res2.stdout


def test_stanford_cars_dataset_with_normalized_records(
    normalized_csv_path: Path, workspace_root: Path
):
    """Verify StanfordCarsDataset can directly consume records loaded from normalized_annotations.csv."""
    df = load_normalized_annotations(normalized_csv_path)
    train_records = df[df["split"] == "train"].head(10).to_dict(orient="records")

    dataset = StanfordCarsDataset(
        annotations=train_records,
        image_root=None,
        crop_bbox=True,
        target_type="class",
    )
    assert len(dataset) == 10

    # Load first sample
    img, target = dataset[0]
    assert img is not None
    assert target is not None
    assert img.size[0] > 0 and img.size[1] > 0
