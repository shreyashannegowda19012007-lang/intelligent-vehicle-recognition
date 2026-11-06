"""
Data Preprocessing, Augmentation, and Robustness Transforms for Classification.
Provides deterministic transforms for validation/testing, controlled training augmentations,
and synthetic stress transforms (blur, low resolution, lighting, occlusion).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_training_transforms(
    image_size: tuple[int, int] = (224, 224),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    random_crop_scale: tuple[float, float] = (0.8, 1.0),
    horizontal_flip_p: float = 0.5,
    rotation_degrees: float = 15.0,
    color_jitter_brightness: float = 0.15,
    color_jitter_contrast: float = 0.15,
    color_jitter_saturation: float = 0.15,
    color_jitter_hue: float = 0.05,
) -> T.Compose:
    """
    Construct training augmentation pipeline preserving vehicle badges, grilles, and structural silhouettes.
    """
    transform_list = [
        T.RandomResizedCrop(
            size=image_size,
            scale=random_crop_scale,
            ratio=(0.85, 1.15),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.RandomHorizontalFlip(p=horizontal_flip_p),
        T.RandomRotation(
            degrees=rotation_degrees,
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.ColorJitter(
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
            saturation=color_jitter_saturation,
            hue=color_jitter_hue,
        ),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    return T.Compose(transform_list)


def get_validation_transforms(
    image_size: tuple[int, int] = (224, 224),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    """
    Deterministic validation and test transformation pipeline.
    Resizes image directly or scales to size and normalizes.
    """
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def get_inference_transforms(
    image_size: tuple[int, int] = (224, 224),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> T.Compose:
    """Exact preprocessing required for unified inference interface."""
    return get_validation_transforms(image_size=image_size, mean=mean, std=std)


# -------------------------------------------------------------------------
# Robustness and Stress Perturbations
# -------------------------------------------------------------------------


class LowResolutionTransform:
    """Simulates degraded sensor resolution by downsampling then upsampling."""

    def __init__(self, downscale_factor: float = 0.4) -> None:
        self.downscale_factor = downscale_factor

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        small_w = max(16, int(w * self.downscale_factor))
        small_h = max(16, int(h * self.downscale_factor))
        down = img.resize((small_w, small_h), Image.Resampling.BILINEAR)
        return down.resize((w, h), Image.Resampling.BILINEAR)


class GaussianBlurTransform:
    """Applies mild or moderate Gaussian blur to simulate optical defocus."""

    def __init__(self, radius: float = 2.0) -> None:
        self.radius = radius

    def __call__(self, img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))


class BrightnessTransform:
    """Applies brightness scaling factor (e.g. 0.7 for dim lighting, 1.3 for glare)."""

    def __init__(self, factor: float = 1.3) -> None:
        self.factor = factor

    def __call__(self, img: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(self.factor)


class ContrastTransform:
    """Applies contrast scaling factor (e.g. 0.6 for foggy/washed out, 1.4 for harsh shadows)."""

    def __init__(self, factor: float = 1.4) -> None:
        self.factor = factor

    def __call__(self, img: Image.Image) -> Image.Image:
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(self.factor)


class PartialOcclusionTransform:
    """Applies a rectangular gray patch simulating partial occlusion of vehicle body/badge."""

    def __init__(self, patch_scale: float = 0.25) -> None:
        self.patch_scale = patch_scale

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        patch_w = int(w * self.patch_scale)
        patch_h = int(h * self.patch_scale)
        # Center the occlusion or offset slightly
        x1 = max(0, (w - patch_w) // 2)
        y1 = max(0, (h - patch_h) // 2)

        arr = np.array(img).copy()
        arr[y1 : y1 + patch_h, x1 : x1 + patch_w] = 128
        return Image.fromarray(arr)


def get_robustness_transform(
    perturbation: str,
    image_size: tuple[int, int] = (224, 224),
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    **kwargs: Any,
) -> T.Compose:
    """
    Construct a deterministic evaluation pipeline under a specific environmental stress condition.

    Args:
        perturbation: One of ['reduced_resolution', 'mild_blur', 'brightness_high',
                      'brightness_low', 'contrast_high', 'contrast_low', 'partial_occlusion'].
        image_size: Output dimensions (width, height).
        mean: Normalization mean.
        std: Normalization std.
    """
    p = perturbation.lower()
    stress_op: Any = None

    if p in {"reduced_resolution", "low_res"}:
        scale = kwargs.get("scale", 0.4)
        stress_op = LowResolutionTransform(downscale_factor=scale)
    elif p in {"mild_blur", "blur"}:
        radius = kwargs.get("radius", 2.0)
        stress_op = GaussianBlurTransform(radius=radius)
    elif p in {"brightness_high", "bright"}:
        factor = kwargs.get("factor", 1.35)
        stress_op = BrightnessTransform(factor=factor)
    elif p in {"brightness_low", "dark"}:
        factor = kwargs.get("factor", 0.65)
        stress_op = BrightnessTransform(factor=factor)
    elif p in {"contrast_high"}:
        factor = kwargs.get("factor", 1.4)
        stress_op = ContrastTransform(factor=factor)
    elif p in {"contrast_low"}:
        factor = kwargs.get("factor", 0.6)
        stress_op = ContrastTransform(factor=factor)
    elif p in {"partial_occlusion", "occlusion"}:
        scale = kwargs.get("scale", 0.25)
        stress_op = PartialOcclusionTransform(patch_scale=scale)
    else:
        raise ValueError(f"Unknown robustness condition: {perturbation}")

    return T.Compose([
        stress_op,
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
