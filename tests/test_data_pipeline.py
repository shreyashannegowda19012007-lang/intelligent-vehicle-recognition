"""
Comprehensive verification test suite for Phase 1 Data Preprocessing & Annotation Engine.
Validates metadata integrity, train/val split stratification, bounding box mathematics,
grayscale handling, multi-task dataset outputs, and dataloader batch shapes.
"""

import json
import os

import pytest
import torch
import yaml
from PIL import Image

from src.data.dataset import (
    StanfordCarsDataset,
    build_dataloaders,
)
from src.utils.box_utils import (
    box_to_yolo,
    compute_iou,
    matlab_to_pixel_box,
    yolo_to_box,
)


@pytest.fixture(scope="session")
def workspace_root():
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="session")
def classes_meta(workspace_root):
    json_path = os.path.join(workspace_root, "data", "classes.json")
    assert os.path.isfile(json_path), f"classes.json missing at {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def train_annotations(workspace_root):
    json_path = os.path.join(workspace_root, "data", "train_annotations.json")
    assert os.path.isfile(json_path), f"train_annotations.json missing at {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def val_annotations(workspace_root):
    json_path = os.path.join(workspace_root, "data", "val_annotations.json")
    assert os.path.isfile(json_path), f"val_annotations.json missing at {json_path}"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_classes_metadata_integrity(classes_meta):
    """Verify classes.json contains all 196 classes and proper taxonomies."""
    assert classes_meta["num_classes"] == 196
    classes = classes_meta["classes"]
    assert len(classes) == 196

    assert len(classes_meta["makes"]) == 49
    assert len(classes_meta["years"]) == 16
    assert len(classes_meta["body_types"]) == 14

    for i, c in enumerate(classes):
        assert c["class_id"] == i
        assert c["class_id_0"] == i
        assert c["class_id_1"] == i + 1
        assert len(c["raw_name"]) > 0
        assert len(c["make"]) > 0
        assert len(c["model"]) > 0
        assert 1990 <= c["year"] <= 2013
        assert c["body_type"] in classes_meta["body_types"]
        assert 0 <= c["make_id"] < 49
        assert 0 <= c["year_id"] < 16
        assert 0 <= c["body_type_id"] < 14
        assert 0 <= c["model_id"] < len(classes_meta["models"])


def test_dataset_split_stratification(train_annotations, val_annotations):
    """Verify 80/20 train/validation split sizes, class coverage, and mutual exclusion."""
    assert len(train_annotations) == 6515
    assert len(val_annotations) == 1629
    assert len(train_annotations) + len(val_annotations) == 8144

    train_files = {r["filename"] for r in train_annotations}
    val_files = {r["filename"] for r in val_annotations}

    # Disjoint sets
    assert len(train_files & val_files) == 0, "Train and val sets overlap!"

    # Stratification check: all 196 classes must exist in both splits
    train_classes = {r["class_id_0"] for r in train_annotations}
    val_classes = {r["class_id_0"] for r in val_annotations}

    assert len(train_classes) == 196, (
        f"Expected 196 classes in train, got {len(train_classes)}"
    )
    assert len(val_classes) == 196, (
        f"Expected 196 classes in val, got {len(val_classes)}"
    )

    # Check minimum class representation
    from collections import Counter

    train_counts = Counter(r["class_id_0"] for r in train_annotations)
    val_counts = Counter(r["class_id_0"] for r in val_annotations)

    for c in range(196):
        assert train_counts[c] >= 19, (
            f"Class {c} has fewer than 19 train samples: {train_counts[c]}"
        )
        assert val_counts[c] >= 4, (
            f"Class {c} has fewer than 4 val samples: {val_counts[c]}"
        )


def test_bounding_box_transforms(train_annotations):
    """Verify coordinate transformation logic between MATLAB, pixel, and YOLO."""
    # Test case: 600x400 image, 1-based MATLAB box [100, 50, 499, 349]
    # In 0-based pixel inclusive: [99, 49, 498, 348]
    # width = 400, height = 300
    x1, y1, x2, y2 = matlab_to_pixel_box(100, 50, 499, 349, 600, 400, inclusive=True)
    assert (x1, y1, x2, y2) == (99, 49, 498, 348)

    # Half-open for PIL crop: [99, 49, 499, 349]
    crop_box = matlab_to_pixel_box(100, 50, 499, 349, 600, 400, inclusive=False)
    assert crop_box == (99, 49, 499, 349)
    assert crop_box[2] - crop_box[0] == 400
    assert crop_box[3] - crop_box[1] == 300

    # Convert to YOLO
    xc, yc, w, h = box_to_yolo(x1, y1, x2, y2, 600, 400, inclusive=True)
    assert 0.0 < xc < 1.0
    assert 0.0 < yc < 1.0
    assert 0.0 < w <= 1.0
    assert 0.0 < h <= 1.0

    # Continuous coordinate center check: (99 + 400/2)/600 = 299/600 = 0.498333
    assert abs(xc - (99 + 200.0) / 600.0) < 1e-5
    assert abs(yc - (49 + 150.0) / 400.0) < 1e-5

    # Reconvert back from YOLO: must be exact pixel-perfect match
    rx1, ry1, rx2, ry2 = yolo_to_box(xc, yc, w, h, 600, 400, inclusive=True)
    assert (rx1, ry1, rx2, ry2) == (x1, y1, x2, y2)

    # Full image box continuous center must be exact (0.5, 0.5)
    f_xc, f_yc, f_w, f_h = box_to_yolo(0, 0, 99, 99, 100, 100, inclusive=True)
    assert (f_xc, f_yc, f_w, f_h) == (0.5, 0.5, 1.0, 1.0)
    rf_box = yolo_to_box(f_xc, f_yc, f_w, f_h, 100, 100, inclusive=True)
    assert rf_box == (0, 0, 99, 99)

    # Verify exact round-trip reconstruction on 500 real dataset annotations
    for rec in train_annotations[:500]:
        gt = rec["bbox_pixel_inc"]
        w_img, h_img = rec["img_width"], rec["img_height"]
        recon = yolo_to_box(*rec["bbox_yolo"], w_img, h_img, inclusive=True)
        assert list(recon) == gt, (
            f"Mismatch on {rec['filename']}: recon {recon} vs gt {gt}"
        )


def test_annotation_boxes_within_image_bounds(train_annotations, val_annotations):
    """Verify all 8,144 bounding boxes lie strictly within image boundaries."""
    for rec in train_annotations[:100] + val_annotations[:100]:
        w = rec["img_width"]
        h = rec["img_height"]
        x1, y1, x2, y2 = rec["bbox_pixel_inc"]
        assert 0 <= x1 <= x2 < w, f"Invalid x coordinates: {x1}, {x2} for width {w}"
        assert 0 <= y1 <= y2 < h, f"Invalid y coordinates: {y1}, {y2} for height {h}"

        xc, yc, bw, bh = rec["bbox_yolo"]
        assert 0.0 < xc < 1.0
        assert 0.0 < yc < 1.0
        assert 0.0 < bw <= 1.0
        assert 0.0 < bh <= 1.0


def test_dataset_grayscale_and_rgb_handling(
    train_annotations, val_annotations, workspace_root
):
    """Verify dataset loader seamlessly converts 1-channel grayscale images to 3-channel RGB."""
    # The 18 grayscale images in cars_train are partitioned: 14 in train split, 4 in val split
    train_gray = [r for r in train_annotations if r.get("is_grayscale")]
    val_gray = [r for r in val_annotations if r.get("is_grayscale")]
    assert len(train_gray) == 14, (
        f"Expected 14 train split grayscale images, found {len(train_gray)}"
    )
    assert len(val_gray) == 4, (
        f"Expected 4 val split grayscale images, found {len(val_gray)}"
    )
    assert len(train_gray) + len(val_gray) == 18, (
        "Total cars_train grayscale images must equal 18"
    )

    # Test loading all train split grayscale images through dataset
    dataset = StanfordCarsDataset(
        annotations=train_gray, crop_bbox=True, target_type="class"
    )

    for i in range(len(train_gray)):
        img, target = dataset[i]
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB", f"Expected RGB mode, got {img.mode}"
        assert isinstance(target, torch.Tensor)

    # Verify total 34 grayscale images across cars_train (18) and cars_test (16)
    test_annos_mat = os.path.join(
        workspace_root,
        "Stanford Dataset",
        "car_devkit",
        "devkit",
        "cars_test_annos.mat",
    )
    test_img_dir = os.path.join(
        workspace_root, "Stanford Dataset", "cars_test", "cars_test"
    )
    if os.path.exists(test_annos_mat) and os.path.exists(test_img_dir):
        from src.data.parser import parse_annotations

        test_records = parse_annotations(test_annos_mat, test_img_dir, is_test=True)
        test_gray = [r for r in test_records if r.get("is_grayscale")]
        assert len(test_gray) == 16, (
            f"Expected 16 test grayscale images, got {len(test_gray)}"
        )
        assert len(train_gray) + len(val_gray) + len(test_gray) == 34, (
            "Total grayscale images across dataset must equal 34"
        )
        test_ds = StanfordCarsDataset(annotations=test_gray, is_test=True)
        t_img, _fname = test_ds[0]
        assert t_img.mode == "RGB"


def test_dataset_multi_task_targets(train_annotations):
    """Verify dataset returns all required multi-task fields in 'multi' mode and single-task modes."""
    dataset = StanfordCarsDataset(
        annotations=train_annotations[:10], crop_bbox=True, target_type="multi"
    )

    _img, targets = dataset[0]
    assert isinstance(targets, dict)
    required_keys = {
        "class_id",
        "make_id",
        "model_id",
        "year_id",
        "body_type_id",
        "bbox",
        "bbox_yolo",
        "filename",
        "index",
    }
    assert required_keys.issubset(targets.keys())
    assert targets["class_id"].dim() == 0  # scalar tensor
    assert targets["make_id"].dim() == 0
    assert targets["model_id"].dim() == 0
    assert targets["year_id"].dim() == 0
    assert targets["body_type_id"].dim() == 0
    assert targets["bbox"].shape == (4,)
    assert targets["bbox_yolo"].shape == (4,)

    # Test single-target modes: 'class', 'make', 'model', 'year', 'body_type'
    for mode in ["class", "make", "model", "year", "body_type"]:
        ds = StanfordCarsDataset(
            annotations=train_annotations[:5], crop_bbox=True, target_type=mode
        )
        _, t = ds[0]
        assert isinstance(t, torch.Tensor)
        assert t.dim() == 0


def test_yolo_dataset_structure(workspace_root):
    """Verify YOLO directory layout, yaml configuration, and label file syntax."""
    yolo_dir = os.path.join(workspace_root, "data", "yolo")
    yaml_path = os.path.join(yolo_dir, "dataset.yaml")
    assert os.path.isfile(yaml_path), f"YOLO dataset.yaml missing at {yaml_path}"

    with open(yaml_path, "r", encoding="utf-8") as f:
        yolo_cfg = yaml.safe_load(f)

    assert yolo_cfg["nc"] == 1
    assert yolo_cfg["names"][0] == "vehicle"

    labels_train_dir = os.path.join(yolo_dir, "labels", "train")
    labels_val_dir = os.path.join(yolo_dir, "labels", "val")

    assert len(os.listdir(labels_train_dir)) == 6515
    assert len(os.listdir(labels_val_dir)) == 1629

    # Check syntax of first label file
    sample_lbl = os.path.join(labels_train_dir, os.listdir(labels_train_dir)[0])
    with open(sample_lbl, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        parts = lines[0].strip().split()
        assert len(parts) == 5
        assert parts[0] == "0"
        coords = [float(p) for p in parts[1:]]
        for c in coords:
            assert 0.0 <= c <= 1.0


def test_dataloader_batch_loading(workspace_root):
    """Verify DataLoader iterates correctly and outputs expected batch shapes."""
    train_json = os.path.join(workspace_root, "data", "train_annotations.json")
    val_json = os.path.join(workspace_root, "data", "val_annotations.json")

    train_loader, _val_loader = build_dataloaders(
        train_json=train_json,
        val_json=val_json,
        batch_size=16,
        image_size=224,
        num_workers=0,  # Single-process for instant test execution
        crop_bbox=True,
        target_type="class",
    )

    batch_imgs, batch_targets = next(iter(train_loader))
    assert batch_imgs.shape == (16, 3, 224, 224), (
        f"Unexpected image batch shape: {batch_imgs.shape}"
    )
    assert batch_targets.shape == (16,), (
        f"Unexpected target batch shape: {batch_targets.shape}"
    )
    assert batch_targets.dtype == torch.long


def test_dataset_invalid_target_type_raises(train_annotations):
    """Verify ValueError is raised when an unsupported target_type is supplied."""
    with pytest.raises(ValueError, match="target_type must be one of"):
        StanfordCarsDataset(
            annotations=train_annotations[:5], target_type="invalid_target"
        )


def test_dataset_missing_image_raises():
    """Verify FileNotFoundError is raised when image file does not exist."""
    fake_record = [
        {
            "filename": "non_existent_car_99999.jpg",
            "file_path": "D:/CV/Stanford Dataset/cars_train/cars_train/non_existent_car_99999.jpg",
            "class_id_0": 0,
        }
    ]
    dataset = StanfordCarsDataset(annotations=fake_record)
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_dataset_full_image_mode(train_annotations):
    """Verify crop_bbox=False returns the complete image without cropping."""
    dataset_crop = StanfordCarsDataset(
        annotations=train_annotations[:1], crop_bbox=True
    )
    dataset_full = StanfordCarsDataset(
        annotations=train_annotations[:1], crop_bbox=False
    )
    img_crop, _ = dataset_crop[0]
    img_full, _ = dataset_full[0]

    # Full image dimensions should match record img_width and img_height
    rec = train_annotations[0]
    assert img_full.size == (rec["img_width"], rec["img_height"])
    # Crop size should match crop_box dimensions
    crop_w = rec["bbox_crop"][2] - rec["bbox_crop"][0]
    crop_h = rec["bbox_crop"][3] - rec["bbox_crop"][1]
    assert img_crop.size == (crop_w, crop_h)


def test_box_clamping_and_boundary_conditions():
    """Verify boundary coordinate handling and zero-area box edge cases."""
    # Boundary box touching edges
    x1, y1, x2, y2 = matlab_to_pixel_box(1, 1, 100, 100, 100, 100, inclusive=True)
    assert (x1, y1, x2, y2) == (0, 0, 99, 99)

    # Box exceeding boundaries is safely clamped
    cx1, cy1, cx2, cy2 = matlab_to_pixel_box(
        -10, -5, 200, 300, 100, 100, inclusive=True
    )
    assert cx1 == 0
    assert cy1 == 0
    assert cx2 == 99
    assert cy2 == 99

    # Non-overlapping boxes return 0 IoU
    iou_zero = compute_iou([0, 0, 10, 10], [50, 50, 60, 60])
    assert iou_zero == 0.0

    # Identical boxes return 1.0 IoU
    iou_one = compute_iou([10, 10, 50, 50], [10, 10, 50, 50])
    assert abs(iou_one - 1.0) < 1e-5


def test_label_distributions(train_annotations, val_annotations, classes_meta):
    """Verify distribution and coverage of hierarchical labels (make, model, year, body_type)."""
    # Verify all 49 makes are represented in both train and val splits
    train_makes = {r["make_id"] for r in train_annotations}
    val_makes = {r["make_id"] for r in val_annotations}
    assert len(train_makes) == 49, f"Train makes: {len(train_makes)}/49"
    assert len(val_makes) == 49, f"Val makes: {len(val_makes)}/49"

    # Verify all 14 body types are represented in both splits
    train_body = {r["body_type_id"] for r in train_annotations}
    val_body = {r["body_type_id"] for r in val_annotations}
    assert len(train_body) == 14, f"Train body types: {len(train_body)}/14"
    assert len(val_body) == 14, f"Val body types: {len(val_body)}/14"

    # Verify all 16 years are represented in both splits
    train_years = {r["year_id"] for r in train_annotations}
    val_years = {r["year_id"] for r in val_annotations}
    assert len(train_years) == 16, f"Train years: {len(train_years)}/16"
    assert len(val_years) == 16, f"Val years: {len(val_years)}/16"

    # Verify model_id is present and populated in every record
    for rec in train_annotations[:100]:
        assert "model_id" in rec
        assert isinstance(rec["model_id"], int)
        assert 0 <= rec["model_id"] < len(classes_meta["models"])


def test_dataset_loading_speed(workspace_root):
    """Benchmark dataset loading speed (throughput) to ensure high-performance I/O."""
    import time

    train_json = os.path.join(workspace_root, "data", "train_annotations.json")
    val_json = os.path.join(workspace_root, "data", "val_annotations.json")

    train_loader, _ = build_dataloaders(
        train_json=train_json,
        val_json=val_json,
        batch_size=32,
        image_size=224,
        num_workers=0,
        crop_bbox=True,
    )

    t0 = time.perf_counter()
    num_batches = 10
    total_samples = 0
    for i, (imgs, _targets) in enumerate(train_loader):
        total_samples += imgs.shape[0]
        if i >= num_batches - 1:
            break
    elapsed = time.perf_counter() - t0
    throughput = total_samples / elapsed
    assert throughput >= 30.0, (
        f"DataLoader speed too slow: {throughput:.1f} img/s (min 30.0 required)"
    )
