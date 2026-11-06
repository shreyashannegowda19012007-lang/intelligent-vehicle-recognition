"""
Histogram of Oriented Gradients (HOG) Feature Extraction.
Extracts gradient orientation descriptors from vehicle images using scikit-image,
supporting configurable cell/block geometry, normalization, single-image, and batch operations.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from skimage.feature import hog

from src.data.preprocessing import resize_image, to_grayscale

logger = logging.getLogger(__name__)


class HOGFeatureExtractor:
    """
    Histogram of Oriented Gradients (HOG) descriptor extractor.
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (128, 128),
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
        transform_sqrt: bool = True,
        feature_vector: bool = True,
    ) -> None:
        """
        Args:
            image_size: (width, height) expected by extractor (resizes if different).
            orientations: Number of gradient orientation bins.
            pixels_per_cell: Number of pixels per cell (H_cell, W_cell).
            cells_per_block: Number of cells per block (H_block, W_block).
            block_norm: Normalization method ('L2-Hys', 'L1', 'L1-sqrt', 'L2').
            transform_sqrt: Power law compression (square root) before gradients.
            feature_vector: If True, returns 1D flattened feature vector.
        """
        self.image_size = tuple(image_size)
        self.orientations = int(orientations)
        self.pixels_per_cell = tuple(pixels_per_cell)
        self.cells_per_block = tuple(cells_per_block)
        self.block_norm = block_norm
        self.transform_sqrt = transform_sqrt
        self.feature_vector = feature_vector

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Ensure 2D grayscale image resized to expected image_size."""
        gray = to_grayscale(image)
        h, w = gray.shape[:2]
        if (w, h) != self.image_size:
            gray = resize_image(gray, target_size=self.image_size, keep_aspect_ratio=False)
        return gray

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 1D HOG feature descriptor from an input image.

        Args:
            image: 2D or 3D numpy image array.

        Returns:
            1D numpy array of float32 features.
        """
        gray = self._prepare_image(image)
        descriptor = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            transform_sqrt=self.transform_sqrt,
            feature_vector=self.feature_vector,
            visualize=False,
        )
        return np.asarray(descriptor, dtype=np.float32)

    def compute_hog_with_visualization(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute HOG features and return both descriptor and visualization image.

        Args:
            image: 2D or 3D numpy image array.

        Returns:
            Tuple of (1D float32 descriptor, 2D uint8 visualization image).
        """
        gray = self._prepare_image(image)
        descriptor, hog_image = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            transform_sqrt=self.transform_sqrt,
            feature_vector=self.feature_vector,
            visualize=True,
        )
        # Rescale hog_image to uint8 [0, 255]
        norm_viz = cv2.normalize(hog_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return np.asarray(descriptor, dtype=np.float32), norm_viz

    def extract_batch(self, images: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """
        Extract HOG features for a batch of images.

        Args:
            images: List or 4D array of images.

        Returns:
            2D numpy array of shape (N, feature_dim).
        """
        features = [self.extract_features(img) for img in images]
        if not features:
            dim = self.get_feature_dimension()
            return np.empty((0, dim), dtype=np.float32)
        return np.vstack(features).astype(np.float32)

    def get_feature_dimension(self) -> int:
        """Calculate the theoretical feature dimension based on settings."""
        w, h = self.image_size
        px, py = self.pixels_per_cell
        bx, by = self.cells_per_block

        cells_x = w // px
        cells_y = h // py

        blocks_x = cells_x - bx + 1
        blocks_y = cells_y - by + 1

        if blocks_x <= 0 or blocks_y <= 0:
            raise ValueError(f"Image size {self.image_size} too small for cell {self.pixels_per_cell} and block {self.cells_per_block}")

        return blocks_x * blocks_y * bx * by * self.orientations

    def get_config(self) -> dict[str, Any]:
        """Return configuration dictionary."""
        return {
            "image_size": self.image_size,
            "orientations": self.orientations,
            "pixels_per_cell": self.pixels_per_cell,
            "cells_per_block": self.cells_per_block,
            "block_norm": self.block_norm,
            "transform_sqrt": self.transform_sqrt,
            "feature_dim": self.get_feature_dimension(),
        }
