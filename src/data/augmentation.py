"""
Data augmentation pipelines for Stanford Cars (Cars196).
Provides training augmentations and deterministic validation transformations
using both torchvision and Albumentations frameworks.
Supports bounding-box aware transformations for object detection.
Enforces domain rules: orientation-sensitive (NO vertical flipping).
"""

from __future__ import annotations

from typing import Any

from torchvision import transforms

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    A = None
    ToTensorV2 = None
    ALBUMENTATIONS_AVAILABLE = False


# Default ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ==============================================================================
# 1. Torchvision Transform Pipelines
# ==============================================================================


def get_torchvision_train_transforms(
    image_size: tuple[int, int] | int = (224, 224),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    hflip_prob: float = 0.5,
    rotation_degrees: int = 15,
    color_jitter: dict[str, float] | None = None,
    random_crop_scale: tuple[float, float] = (0.8, 1.0),
) -> transforms.Compose:
    """
    Construct torchvision training transform pipeline.
    Notice: NO vertical flip is used because vehicles have strict vertical orientation.

    Args:
        image_size: (height, width) or int size.
        mean: Normalization mean.
        std: Normalization std.
        hflip_prob: Probability of horizontal flip.
        rotation_degrees: Maximum degree for random rotation.
        color_jitter: Dict of brightness, contrast, saturation, hue.
        random_crop_scale: Range of size of the origin size cropped.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    jitter_params = color_jitter or {
        "brightness": 0.15,
        "contrast": 0.15,
        "saturation": 0.15,
        "hue": 0.05,
    }

    transform_list = [
        transforms.Resize(size),
        transforms.RandomResizedCrop(size, scale=random_crop_scale, ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=hflip_prob),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.ColorJitter(
            brightness=jitter_params.get("brightness", 0.15),
            contrast=jitter_params.get("contrast", 0.15),
            saturation=jitter_params.get("saturation", 0.15),
            hue=jitter_params.get("hue", 0.05),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    return transforms.Compose(transform_list)


def get_torchvision_val_transforms(
    image_size: tuple[int, int] | int = (224, 224),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """
    Construct deterministic validation/test transformation pipeline.
    Only resizes and normalizes without stochastic alterations.
    """
    size = (image_size, image_size) if isinstance(image_size, int) else image_size

    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


# ==============================================================================
# 2. Albumentations Transform Pipelines (with Bounding Box support)
# ==============================================================================


def get_albumentations_train_transforms(
    image_size: tuple[int, int] | int = (224, 224),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    hflip_prob: float = 0.5,
    rotation_degrees: int = 15,
    bbox_format: str | None = None,
) -> Any:
    """
    Construct Albumentations training pipeline.
    Optionally supports bounding box transformation when bbox_format is specified
    (e.g., 'pascal_voc', 'albumentations', or 'yolo').

    Args:
        image_size: Target (height, width).
        mean: Channel means.
        std: Channel std deviations.
        hflip_prob: Probability of horizontal flip.
        rotation_degrees: Max rotation limit.
        bbox_format: One of 'pascal_voc', 'albumentations', 'yolo', or None.

    Returns:
        A.Compose transform object.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Albumentations is not installed in this environment.")

    h, w = (image_size, image_size) if isinstance(image_size, int) else image_size

    transform_list = [
        A.Resize(h, w),
        A.HorizontalFlip(p=hflip_prob),
        A.Rotate(limit=rotation_degrees, p=0.5),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.1),
        A.Perspective(scale=(0.02, 0.05), p=0.1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    bbox_params = None
    if bbox_format:
        bbox_params = A.BboxParams(
            format=bbox_format,
            label_fields=["category_ids"],
            min_visibility=0.2,
        )

    return A.Compose(transform_list, bbox_params=bbox_params)


def get_albumentations_val_transforms(
    image_size: tuple[int, int] | int = (224, 224),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    bbox_format: str | None = None,
) -> Any:
    """
    Construct deterministic Albumentations validation pipeline.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Albumentations is not installed in this environment.")

    h, w = (image_size, image_size) if isinstance(image_size, int) else image_size

    transform_list = [
        A.Resize(h, w),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    bbox_params = None
    if bbox_format:
        bbox_params = A.BboxParams(
            format=bbox_format,
            label_fields=["category_ids"],
        )

    return A.Compose(transform_list, bbox_params=bbox_params)


__all__ = [
    "ALBUMENTATIONS_AVAILABLE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "get_albumentations_train_transforms",
    "get_albumentations_val_transforms",
    "get_torchvision_train_transforms",
    "get_torchvision_val_transforms",
]
