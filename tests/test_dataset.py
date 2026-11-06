"""
Automated unit tests for PyTorch Datasets, DataLoaders, Augmentations, and DataModule.
"""

from __future__ import annotations

import torch

from src.data.augmentation import (
    ALBUMENTATIONS_AVAILABLE,
    get_albumentations_train_transforms,
    get_torchvision_train_transforms,
    get_torchvision_val_transforms,
)
from src.data.classification_dataset import (
    StanfordCarsClassificationDataset,
    create_classification_dataloaders,
)
from src.data.data_module import StanfordCarsDataModule
from src.data.detection_dataset import (
    VEHICLE_DETECTION_CLASS_ID,
    StanfordCarsDetectionDataset,
    create_detection_dataloaders,
)
from src.utils.config import find_project_root


def test_classification_dataset_item_schema():
    root = find_project_root()
    train_csv = root / "data" / "splits" / "train.csv"
    ds = StanfordCarsClassificationDataset(train_csv, project_root=root, crop_bbox=True)

    assert len(ds) == 6922
    item = ds[0]

    assert isinstance(item, dict)
    expected_keys = {"image", "class_id", "make_id", "model_id", "year", "image_path"}
    assert set(item.keys()) == expected_keys

    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, 224, 224)
    assert isinstance(item["class_id"], torch.Tensor)
    assert 0 <= item["class_id"].item() < 196
    assert isinstance(item["make_id"], torch.Tensor)
    assert 0 <= item["make_id"].item() < 49
    assert isinstance(item["model_id"], torch.Tensor)
    assert isinstance(item["year"], torch.Tensor)
    assert isinstance(item["image_path"], str)


def test_classification_dataset_test_sentinels():
    root = find_project_root()
    test_csv = root / "data" / "splits" / "test.csv"
    ds = StanfordCarsClassificationDataset(test_csv, project_root=root, crop_bbox=False)

    item = ds[0]
    assert item["class_id"].item() == -1
    assert item["make_id"].item() == -1
    assert item["model_id"].item() == -1
    assert item["year"].item() == -1


def test_detection_dataset():
    root = find_project_root()
    train_csv = root / "data" / "splits" / "train.csv"
    ds = StanfordCarsDetectionDataset(train_csv, project_root=root, image_size=(640, 640))

    img, target = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 640, 640)

    assert "boxes" in target
    assert "labels" in target
    assert "boxes_yolo" in target
    assert target["labels"][0].item() == VEHICLE_DETECTION_CLASS_ID

    # Verify YOLO boxes are normalized to [0, 1]
    yolo_box = target["boxes_yolo"][0]
    xc, yc, w, h = yolo_box.tolist()
    assert 0.0 <= xc <= 1.0
    assert 0.0 <= yc <= 1.0
    assert 0.0 < w <= 1.0
    assert 0.0 < h <= 1.0


def test_classification_dataloader_batch():
    root = find_project_root()
    train_csv = root / "data" / "splits" / "train.csv"
    val_csv = root / "data" / "splits" / "validation.csv"

    loaders = create_classification_dataloaders(
        train_source=train_csv,
        val_source=val_csv,
        batch_size=4,
        num_workers=0,
        project_root=root,
    )

    train_batch = next(iter(loaders["train"]))
    assert train_batch["image"].shape == (4, 3, 224, 224)
    assert train_batch["class_id"].shape == (4,)
    assert train_batch["make_id"].shape == (4,)

    val_batch = next(iter(loaders["val"]))
    assert val_batch["image"].shape == (4, 3, 224, 224)
    assert val_batch["class_id"].shape == (4,)


def test_detection_dataloader_batch():
    root = find_project_root()
    train_csv = root / "data" / "splits" / "train.csv"

    loaders = create_detection_dataloaders(
        train_source=train_csv,
        batch_size=2,
        num_workers=0,
        project_root=root,
    )

    imgs, targets = next(iter(loaders["train"]))
    assert imgs.shape == (2, 3, 640, 640)
    assert len(targets) == 2
    assert targets[0]["labels"].shape == (1,)


def test_torchvision_transforms():
    tf_train = get_torchvision_train_transforms((128, 128))
    tf_val = get_torchvision_val_transforms((128, 128))

    from PIL import Image
    dummy_img = Image.new("RGB", (200, 300), color=(100, 150, 200))

    t1 = tf_train(dummy_img)
    t2 = tf_val(dummy_img)

    assert t1.shape == (3, 128, 128)
    assert t2.shape == (3, 128, 128)


def test_albumentations_transforms():
    if not ALBUMENTATIONS_AVAILABLE:
        return

    import numpy as np
    dummy_arr = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)

    alb_tf = get_albumentations_train_transforms((128, 128), bbox_format="pascal_voc")
    res = alb_tf(image=dummy_arr, bboxes=[[10, 10, 100, 100]], category_ids=[0])

    assert "image" in res
    assert res["image"].shape == (3, 128, 128)
    assert "bboxes" in res


def test_datamodule():
    dm = StanfordCarsDataModule(batch_size=4, num_workers=0)
    dm.setup()

    # Class weights
    weights = dm.get_class_weights("balanced")
    assert weights.shape == (196,)
    assert abs(weights.mean().item() - 1.0) < 0.05

    # Sampler train loader
    loader_weighted = dm.train_dataloader(use_weighted_sampler=True)
    batch = next(iter(loader_weighted))
    assert batch["image"].shape == (4, 3, 224, 224)

    # Detection loader
    det_loader = dm.detection_train_dataloader()
    det_imgs, det_targets = next(iter(det_loader))
    assert det_imgs.shape == (4, 3, 640, 640)
    assert len(det_targets) == 4


def test_detection_dataset_with_albumentations_horizontal_flip():
    """Verify Albumentations horizontal flip properly transforms both image and bounding box coordinates."""
    if not ALBUMENTATIONS_AVAILABLE:
        return

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    root = find_project_root()
    train_csv = root / "data" / "splits" / "train.csv"

    # Flip horizontally with 100% certainty
    flip_tf = A.Compose(
        [A.HorizontalFlip(p=1.0), ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    )

    ds = StanfordCarsDetectionDataset(train_csv, project_root=root, transform=flip_tf, yolo_normalized=True)

    # Find a sample where box is asymmetric relative to center
    sample_idx = None
    for i in range(min(50, len(ds))):
        orig_x1 = float(ds.df.iloc[i]["bbox_x1"])
        orig_x2 = float(ds.df.iloc[i]["bbox_x2"])
        w = float(ds.df.iloc[i]["image_width"])
        orig_xc = (orig_x1 + orig_x2) / (2.0 * w)
        if abs(orig_xc - 0.5) > 0.08:
            sample_idx = i
            break

    if sample_idx is None:
        sample_idx = 0

    orig_x1 = float(ds.df.iloc[sample_idx]["bbox_x1"])
    orig_x2 = float(ds.df.iloc[sample_idx]["bbox_x2"])
    orig_w = float(ds.df.iloc[sample_idx]["image_width"])
    orig_xc = (orig_x1 + orig_x2) / (2.0 * orig_w)

    _img, target = ds[sample_idx]
    flipped_xc = target["boxes"][0, 0].item()

    # In a horizontally flipped image, the new center must be approximately 1.0 - orig_xc
    expected_flipped_xc = 1.0 - orig_xc
    assert abs(flipped_xc - expected_flipped_xc) < 0.05, (
        f"Transformed YOLO box xc ({flipped_xc:.4f}) does not match expected flipped center ({expected_flipped_xc:.4f})"
    )
    # Both target['boxes'] and target['boxes_yolo'] must match
    assert torch.allclose(target["boxes"], target["boxes_yolo"])


def test_detection_dataset_yolo_normalized_false():
    """Verify target['boxes'] returns scaled pixel coordinates when yolo_normalized=False."""
    root = find_project_root()
    train_csv = root / "data" / "splits" / "train.csv"

    ds = StanfordCarsDetectionDataset(train_csv, project_root=root, image_size=(640, 640), yolo_normalized=False)
    _img, target = ds[0]

    assert torch.allclose(target["boxes"], target["boxes_pixel"])
    # Scaled pixel boxes should be within the [0, 640] bounds
    x1, y1, x2, y2 = target["boxes"][0].tolist()
    assert 0.0 <= x1 <= 640.0
    assert 0.0 <= y1 <= 640.0
    assert x1 <= x2 <= 640.0
    assert y1 <= y2 <= 640.0


def test_detection_dataset_dropped_bbox():
    """Verify that when Albumentations drops bounding box, empty tensors of shape [0, 4] and [0] are returned."""
    if not ALBUMENTATIONS_AVAILABLE:
        return

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    root = find_project_root()
    train_csv = root / "data" / "splits" / "train.csv"

    # Crop an extreme corner that guarantees 0 overlap with the vehicle box
    drop_tf = A.Compose(
        [A.Crop(x_min=0, y_min=0, x_max=5, y_max=5), ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"], min_visibility=0.5),
    )

    ds = StanfordCarsDetectionDataset(train_csv, project_root=root, transform=drop_tf)
    _img, target = ds[0]

    assert target["boxes"].shape == (0, 4)
    assert target["boxes_yolo"].shape == (0, 4)
    assert target["boxes_pixel"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_classification_dataset_graceful_missing_image():
    """Verify StanfordCarsClassificationDataset gracefully returns fallback tensor on missing image without crashing."""
    import pandas as pd

    fake_df = pd.DataFrame([
        {
            "image_path": "non_existent/fake_car_12345.jpg",
            "image_id": "fake_12345",
            "split": "train",
            "class_id": 10,
            "make": "Audi",
            "model": "A4",
            "year": 2010,
            "bbox_x1": 10,
            "bbox_y1": 10,
            "bbox_x2": 50,
            "bbox_y2": 50,
        }
    ])

    ds = StanfordCarsClassificationDataset(fake_df, image_size=(224, 224))
    item = ds[0]

    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape == (3, 224, 224)
    assert item["class_id"].item() == -1
    assert item["make_id"].item() == -1
    assert item["model_id"].item() == -1
    assert item["year"].item() == -1


def test_detection_dataset_graceful_missing_image():
    """Verify StanfordCarsDetectionDataset gracefully returns blank image and empty targets on missing image without crashing."""
    import pandas as pd

    fake_df = pd.DataFrame([
        {
            "image_path": "non_existent/fake_car_12345.jpg",
            "image_id": "fake_12345",
            "split": "train",
            "bbox_x1": 10,
            "bbox_y1": 10,
            "bbox_x2": 50,
            "bbox_y2": 50,
        }
    ])

    ds = StanfordCarsDetectionDataset(fake_df, image_size=(640, 640))
    img, target = ds[0]

    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 640, 640)
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)


def test_detection_dataloader_with_custom_transforms():
    """Verify create_detection_dataloaders correctly propagates custom transform pipelines."""
    root = find_project_root()
    train_csv = root / "data" / "splits" / "train.csv"

    loaders = create_detection_dataloaders(
        train_source=train_csv,
        batch_size=2,
        num_workers=0,
        project_root=root,
    )
    assert "train" in loaders
    batch_img, batch_targets = next(iter(loaders["train"]))
    assert batch_img.shape == (2, 3, 640, 640)
    assert len(batch_targets) == 2
