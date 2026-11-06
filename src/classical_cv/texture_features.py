"""
Local Binary Patterns (LBP) Texture Feature Extraction.
Extracts micro-texture descriptors using scikit-image's local_binary_pattern.
Supports uniform patterns, spatial grid partitioning, L1/L2 normalization,
and batch feature extraction.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from src.data.preprocessing import resize_image, to_grayscale

logger = logging.getLogger(__name__)


class LBPFeatureExtractor:
    """
    Local Binary Patterns (LBP) texture descriptor extractor.
    """

    def __init__(
        self,
        radius: int = 2,
        n_points: int = 16,
        method: str = "uniform",
        grid: tuple[int, int] = (4, 4),
        norm: str = "l2",
        image_size: tuple[int, int] = (128, 128),
    ) -> None:
        """
        Args:
            radius: Radius of circular neighborhood in pixels.
            n_points: Number of circularly symmetric neighbor set points.
            method: LBP operator method ('uniform', 'default', 'ror', 'var').
            grid: (grid_y, grid_x) spatial subdivisions (e.g. (4, 4) for 16 spatial cells).
            norm: Normalization method ('l2', 'l1', or 'none').
            image_size: Target (width, height) to ensure consistent cell sizes.
        """
        self.radius = int(radius)
        self.n_points = int(n_points)
        self.method = method.lower()
        self.grid = tuple(grid)
        self.norm = norm.lower()
        self.image_size = tuple(image_size)

        # Calculate number of bins based on method
        if self.method == "uniform":
            self.n_bins = self.n_points + 2
        elif self.method in ("default", "ror"):
            self.n_bins = 2 ** self.n_points
        elif self.method == "var":
            self.n_bins = 32  # continuous variance typically binned
        else:
            self.n_bins = self.n_points + 2

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Convert to 2D grayscale and resize to image_size."""
        gray = to_grayscale(image)
        h, w = gray.shape[:2]
        if (w, h) != self.image_size:
            gray = resize_image(gray, target_size=self.image_size, keep_aspect_ratio=False)
        return gray

    def compute_lbp_image(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the raw LBP map for the input image.

        Args:
            image: 2D or 3D image.

        Returns:
            2D float array containing LBP values.
        """
        gray = self._prepare_image(image)
        return local_binary_pattern(gray, P=self.n_points, R=self.radius, method=self.method)

    def compute_lbp_with_visualization(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute LBP feature vector and return normalized uint8 visualization map.

        Returns:
            Tuple of (1D float32 descriptor, 2D uint8 visualization image).
        """
        lbp_map = self.compute_lbp_image(image)
        features = self._extract_from_lbp_map(lbp_map)

        # Normalize LBP map for display
        norm_viz = cv2.normalize(lbp_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return features, norm_viz

    def _extract_from_lbp_map(self, lbp_map: np.ndarray) -> np.ndarray:
        """Calculate spatial histogram from an existing LBP map."""
        gy, gx = self.grid
        h, w = lbp_map.shape[:2]

        cell_h = h // gy
        cell_w = w // gx

        histograms: list[np.ndarray] = []

        for row in range(gy):
            for col in range(gx):
                y1 = row * cell_h
                y2 = (row + 1) * cell_h if row < gy - 1 else h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w if col < gx - 1 else w

                cell = lbp_map[y1:y2, x1:x2]
                hist, _ = np.histogram(cell, bins=self.n_bins, range=(0, self.n_bins))
                hist = hist.astype(np.float32)

                # Sub-region normalization
                cell_sum = hist.sum()
                if cell_sum > 0:
                    hist /= cell_sum

                histograms.append(hist)

        combined = np.concatenate(histograms)

        # Vector normalization
        if self.norm == "l2":
            norm_val = np.linalg.norm(combined)
            if norm_val > 1e-7:
                combined /= norm_val
        elif self.norm == "l1":
            norm_val = np.sum(np.abs(combined))
            if norm_val > 1e-7:
                combined /= norm_val

        return combined.astype(np.float32)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP spatial histogram descriptor from an image.

        Args:
            image: Input image array.

        Returns:
            1D float32 numpy array.
        """
        lbp_map = self.compute_lbp_image(image)
        return self._extract_from_lbp_map(lbp_map)

    def extract_batch(self, images: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """
        Extract LBP features for a batch of images.

        Returns:
            2D numpy array of shape (N, feature_dim).
        """
        features = [self.extract_features(img) for img in images]
        if not features:
            dim = self.get_feature_dimension()
            return np.empty((0, dim), dtype=np.float32)
        return np.vstack(features).astype(np.float32)

    def get_feature_dimension(self) -> int:
        """Calculate total output descriptor dimension."""
        gy, gx = self.grid
        return gy * gx * self.n_bins

    def get_config(self) -> dict[str, Any]:
        """Return configuration dictionary."""
        return {
            "radius": self.radius,
            "n_points": self.n_points,
            "method": self.method,
            "grid": self.grid,
            "norm": self.norm,
            "image_size": self.image_size,
            "n_bins": self.n_bins,
            "feature_dim": self.get_feature_dimension(),
        }
