"""
Automated unit tests for image preprocessing functions in src/data/preprocessing.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocessing import (
    apply_clahe,
    bgr_to_rgb,
    compute_image_quality_metrics,
    denoise_bilateral,
    denoise_gaussian,
    detect_edges_canny,
    estimate_blur,
    load_image,
    measure_brightness,
    measure_contrast,
    normalize_image,
    preprocess_pipeline,
    resize_image,
    rgb_to_bgr,
    to_grayscale,
)


@pytest.fixture
def synthetic_rgb_image() -> np.ndarray:
    np.random.seed(42)
    return np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_gray_image() -> np.ndarray:
    np.random.seed(42)
    return np.random.randint(0, 256, (100, 150), dtype=np.uint8)


def test_bgr_rgb_conversions(synthetic_rgb_image: np.ndarray):
    bgr = rgb_to_bgr(synthetic_rgb_image)
    assert bgr.shape == synthetic_rgb_image.shape
    # Inverting should recover original
    recovered = bgr_to_rgb(bgr)
    np.testing.assert_array_equal(recovered, synthetic_rgb_image)


def test_to_grayscale(synthetic_rgb_image: np.ndarray, synthetic_gray_image: np.ndarray):
    gray = to_grayscale(synthetic_rgb_image)
    assert gray.ndim == 2
    assert gray.shape == (100, 150)

    # Idempotent on 2D image
    gray_from_gray = to_grayscale(synthetic_gray_image)
    np.testing.assert_array_equal(gray_from_gray, synthetic_gray_image)


def test_resize_image_direct(synthetic_rgb_image: np.ndarray):
    resized = resize_image(synthetic_rgb_image, target_size=(224, 224), keep_aspect_ratio=False)
    assert resized.shape == (224, 224, 3)


def test_resize_image_letterbox(synthetic_rgb_image: np.ndarray):
    resized = resize_image(synthetic_rgb_image, target_size=(200, 200), keep_aspect_ratio=True)
    assert resized.shape == (200, 200, 3)


def test_normalize_image_range(synthetic_rgb_image: np.ndarray):
    norm = normalize_image(synthetic_rgb_image)
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_normalize_image_zscore(synthetic_rgb_image: np.ndarray):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    norm = normalize_image(synthetic_rgb_image, mean=mean, std=std)
    assert norm.shape == synthetic_rgb_image.shape
    assert norm.dtype == np.float32


def test_apply_clahe_color_and_gray(
    synthetic_rgb_image: np.ndarray, synthetic_gray_image: np.ndarray
):
    enhanced_rgb = apply_clahe(synthetic_rgb_image, clip_limit=2.0)
    assert enhanced_rgb.shape == synthetic_rgb_image.shape

    enhanced_gray = apply_clahe(synthetic_gray_image, clip_limit=2.0)
    assert enhanced_gray.shape == synthetic_gray_image.shape


def test_denoise_gaussian(synthetic_rgb_image: np.ndarray):
    denoised = denoise_gaussian(synthetic_rgb_image, ksize=(3, 3))
    assert denoised.shape == synthetic_rgb_image.shape


def test_denoise_bilateral(synthetic_rgb_image: np.ndarray):
    denoised = denoise_bilateral(synthetic_rgb_image, d=5, sigma_color=50, sigma_space=50)
    assert denoised.shape == synthetic_rgb_image.shape


def test_detect_edges_canny(synthetic_rgb_image: np.ndarray):
    edges = detect_edges_canny(synthetic_rgb_image, low_thresh=50, high_thresh=150)
    assert edges.ndim == 2
    assert edges.shape == (100, 150)
    assert set(np.unique(edges)).issubset({0, 255})


def test_image_quality_metrics(synthetic_rgb_image: np.ndarray):
    b = measure_brightness(synthetic_rgb_image)
    c = measure_contrast(synthetic_rgb_image)
    sharp = estimate_blur(synthetic_rgb_image)

    assert isinstance(b, float)
    assert 0.0 <= b <= 255.0
    assert isinstance(c, float)
    assert c >= 0.0
    assert isinstance(sharp, float)

    metrics = compute_image_quality_metrics(synthetic_rgb_image)
    assert "mean_brightness" in metrics
    assert "contrast_std" in metrics
    assert "sharpness_laplacian_var" in metrics
    assert "is_blurry" in metrics


def test_preprocess_pipeline(synthetic_rgb_image: np.ndarray):
    processed = preprocess_pipeline(
        synthetic_rgb_image,
        target_size=(128, 128),
        apply_enhancement=True,
        denoise_method="gaussian",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    )
    assert processed.shape == (128, 128, 3)
    assert processed.dtype == np.float32


def test_load_image_non_existent():
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_file_path_12345.jpg")
