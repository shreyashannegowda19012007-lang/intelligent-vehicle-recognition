"""
Reusable image preprocessing functions using OpenCV for P5 Vehicle Recognition.
Provides image loading, resizing, normalization, color space conversion, CLAHE enhancement,
denoising, Canny edge detection, and image quality measurements.
All functions preserve original image arrays without in-place modification.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(file_path: str | Path, as_rgb: bool = True) -> np.ndarray:
    """
    Load an image from disk safely using PIL / OpenCV.
    Guarantees 3-channel RGB representation (converting 1-channel grayscale or 4-channel RGBA).

    Args:
        file_path: Path to the image file.
        as_rgb: If True, returns image in RGB format; if False, returns in BGR format.

    Returns:
        uint8 numpy ndarray of shape (H, W, 3).
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found at path: {file_path}")

    # Use PIL to read image robustly (handles exotic EXIF, grayscale, corrupted headers)
    with Image.open(p) as pil_img:
        rgb_img = pil_img.convert("RGB")
        arr = np.array(rgb_img, dtype=np.uint8)

    if as_rgb:
        return arr
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an image from BGR to RGB color space."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an image from RGB to BGR color space."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB or BGR image to single-channel grayscale."""
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    keep_aspect_ratio: bool = False,
    interpolation: int | None = None,
) -> np.ndarray:
    """
    Resize image to target_size (width, height).

    Args:
        image: Input image array (H, W) or (H, W, C).
        target_size: (target_width, target_height).
        keep_aspect_ratio: If True, pads with zeros (letterboxing) to maintain aspect ratio.
        interpolation: OpenCV interpolation mode. If None, auto-selects INTER_AREA
            for downsampling and INTER_LINEAR for upsampling.

    Returns:
        Resized numpy ndarray with shape (target_height, target_width, C) or (target_height, target_width).
    """
    target_w, target_h = target_size
    orig_h, orig_w = image.shape[:2]

    if interpolation is None:
        if target_w < orig_w or target_h < orig_h:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR

    if not keep_aspect_ratio:
        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    # Letterbox resize
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2

    if image.ndim == 2:
        canvas = np.zeros((target_h, target_w), dtype=image.dtype)
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
    else:
        canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    return canvas


def normalize_image(
    image: np.ndarray,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """
    Normalize image values to float32 range [0, 1] or standard ImageNet z-scores.

    Args:
        image: uint8 or float32 image array.
        mean: Optional per-channel mean (e.g., ImageNet (0.485, 0.456, 0.406)).
        std: Optional per-channel std (e.g., ImageNet (0.229, 0.224, 0.225)).

    Returns:
        float32 normalized array.
    """
    img_float = image.astype(np.float32)
    if img_float.max() > 1.0:
        img_float /= 255.0

    if mean is not None and std is not None:
        mean_arr = np.array(mean, dtype=np.float32)
        std_arr = np.array(std, dtype=np.float32)
        return (img_float - mean_arr) / std_arr

    return img_float


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    For color images, operates on the L channel in LAB color space to preserve chromaticity.

    Args:
        image: uint8 image (RGB or grayscale).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Grid size for local histogram equalization (default: 8x8).

    Returns:
        Enhanced uint8 image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if image.ndim == 2:
        return clahe.apply(image)

    # Color image: convert to LAB, equalize L channel, convert back
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def denoise_gaussian(
    image: np.ndarray,
    ksize: tuple[int, int] = (5, 5),
    sigma_x: float = 0.0,
) -> np.ndarray:
    """Apply Gaussian blur denoising to an image."""
    return cv2.GaussianBlur(image, ksize, sigmaX=sigma_x)


def denoise_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Apply edge-preserving bilateral filtering.

    Args:
        image: uint8 input image.
        d: Diameter of each pixel neighborhood.
        sigma_color: Filter sigma in the color space.
        sigma_space: Filter sigma in the coordinate space.
    """
    return cv2.bilateralFilter(
        image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
    )


def detect_edges_canny(
    image: np.ndarray,
    low_thresh: int = 50,
    high_thresh: int = 150,
    aperture_size: int = 3,
    l2_gradient: bool = True,
) -> np.ndarray:
    """
    Perform Canny edge detection.

    Args:
        image: RGB or grayscale image.
        low_thresh: First threshold for the hysteresis procedure.
        high_thresh: Second threshold for the hysteresis procedure.
        aperture_size: Aperture size for the Sobel operator.
        l2_gradient: Flag indicating whether a more accurate L2 norm should be used.

    Returns:
        Single-channel uint8 binary edge map.
    """
    gray = to_grayscale(image)
    return cv2.Canny(
        gray,
        threshold1=low_thresh,
        threshold2=high_thresh,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )


def measure_brightness(image: np.ndarray) -> float:
    """Calculate mean brightness (luminance) of an image in range [0, 255]."""
    gray = to_grayscale(image)
    return float(round(float(np.mean(gray)), 2))


def measure_contrast(image: np.ndarray) -> float:
    """Calculate standard deviation of pixel intensities as contrast metric."""
    gray = to_grayscale(image)
    return float(round(float(np.std(gray)), 2))


def estimate_blur(image: np.ndarray) -> float:
    """
    Estimate image blur using the variance of the Laplacian.
    Higher values indicate sharp edges; values below 100 typically indicate blur.
    """
    gray = to_grayscale(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(round(float(laplacian.var()), 2))


def compute_image_quality_metrics(image: np.ndarray) -> dict[str, float | bool]:
    """
    Calculate fundamental image quality metrics:
    - mean_brightness: Mean pixel intensity in [0, 255].
    - contrast_std: Standard deviation of pixel intensities.
    - sharpness_laplacian_var: Variance of Laplacian (blur metric; higher = sharper).
    - is_blurry: True if sharpness_laplacian_var < 100.0.

    Args:
        image: Input RGB or grayscale image array.

    Returns:
        Dictionary containing quality metrics.
    """
    b = measure_brightness(image)
    c = measure_contrast(image)
    sharp = estimate_blur(image)

    return {
        "mean_brightness": b,
        "contrast_std": c,
        "sharpness_laplacian_var": sharp,
        "is_blurry": sharp < 100.0,
    }


def preprocess_pipeline(
    image: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    apply_enhancement: bool = False,
    denoise_method: str | None = None,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """
    Modular preprocessing pipeline.

    Args:
        image: Input RGB image.
        target_size: (width, height) for resizing.
        apply_enhancement: If True, applies CLAHE contrast enhancement.
        denoise_method: Optional denoising: 'gaussian' or 'bilateral'.
        mean: Optional normalization mean.
        std: Optional normalization std.

    Returns:
        Processed image ndarray.
    """
    processed = image.copy()

    if apply_enhancement:
        processed = apply_clahe(processed)

    if denoise_method == "gaussian":
        processed = denoise_gaussian(processed)
    elif denoise_method == "bilateral":
        processed = denoise_bilateral(processed)

    processed = resize_image(processed, target_size=target_size)

    if mean is not None and std is not None:
        processed = normalize_image(processed, mean=mean, std=std)

    return processed


__all__ = [
    "apply_clahe",
    "bgr_to_rgb",
    "compute_image_quality_metrics",
    "denoise_bilateral",
    "denoise_gaussian",
    "detect_edges_canny",
    "estimate_blur",
    "load_image",
    "measure_brightness",
    "measure_contrast",
    "normalize_image",
    "preprocess_pipeline",
    "resize_image",
    "rgb_to_bgr",
    "to_grayscale",
]
