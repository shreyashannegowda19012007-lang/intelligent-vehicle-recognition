"""
Color Histogram Feature Extraction for Classical Computer Vision.
Supports RGB, HSV, and LAB color spaces, flexible channel binning, L1/L2 normalization,
and batch feature extraction.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ColorHistogramExtractor:
    """
    Extracts multi-space color histogram descriptors.
    """

    CHANNEL_RANGES: ClassVar[dict[str, tuple[list[int], list[int], list[int]]]] = {
        "rgb": ([0, 256], [0, 256], [0, 256]),
        "hsv": ([0, 180], [0, 256], [0, 256]),
        "lab": ([0, 256], [0, 256], [0, 256]),
    }

    CHANNEL_NAMES: ClassVar[dict[str, tuple[str, str, str]]] = {
        "rgb": ("R", "G", "B"),
        "hsv": ("H", "S", "V"),
        "lab": ("L", "A", "B"),
    }

    def __init__(
        self,
        color_spaces: list[str] | tuple[str, ...] = ("hsv", "rgb"),
        bins_per_channel: int = 16,
        norm: str = "l2",
        mode: str = "concatenated",
    ) -> None:
        """
        Args:
            color_spaces: List of spaces to extract ('rgb', 'hsv', 'lab').
            bins_per_channel: Number of histogram bins per channel (e.g. 16 or 32).
            norm: Normalization method: 'l2', 'l1', or 'none'.
            mode: 'concatenated' (1D per channel) or 'joint' (3D joint histogram per space).
        """
        self.color_spaces = [cs.lower() for cs in color_spaces]
        for cs in self.color_spaces:
            if cs not in self.CHANNEL_RANGES:
                raise ValueError(f"Unsupported color space: {cs}. Must be one of {list(self.CHANNEL_RANGES.keys())}")

        self.bins_per_channel = int(bins_per_channel)
        self.norm = norm.lower()
        self.mode = mode.lower()

    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert input to 3-channel RGB uint8."""
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image

    def _convert_color_space(self, rgb_image: np.ndarray, color_space: str) -> np.ndarray:
        """Convert RGB image to target color space."""
        cs = color_space.lower()
        if cs == "rgb":
            return rgb_image
        if cs == "hsv":
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        if cs == "lab":
            return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        raise ValueError(f"Unknown color space: {cs}")

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract normalized color histogram feature vector from an image.

        Args:
            image: Input image (H, W, 3) RGB or grayscale.

        Returns:
            1D numpy array of float32 features.
        """
        rgb = self._ensure_rgb(image)
        feature_chunks: list[np.ndarray] = []

        for cs in self.color_spaces:
            converted = self._convert_color_space(rgb, cs)
            ranges = self.CHANNEL_RANGES[cs]

            if self.mode == "joint":
                hist = cv2.calcHist(
                    [converted],
                    [0, 1, 2],
                    None,
                    [self.bins_per_channel, self.bins_per_channel, self.bins_per_channel],
                    ranges[0] + ranges[1] + ranges[2],
                )
                chunk = hist.flatten()
            else:
                channel_chunks = []
                for ch_idx in range(3):
                    ch_hist = cv2.calcHist(
                        [converted],
                        [ch_idx],
                        None,
                        [self.bins_per_channel],
                        ranges[ch_idx],
                    )
                    channel_chunks.append(ch_hist.flatten())
                chunk = np.concatenate(channel_chunks)

            # Per-space or global normalization
            chunk = np.asarray(chunk, dtype=np.float32)
            feature_chunks.append(chunk)

        combined = np.concatenate(feature_chunks)

        # Apply final vector normalization
        if self.norm == "l2":
            norm_val = np.linalg.norm(combined)
            if norm_val > 1e-7:
                combined /= norm_val
        elif self.norm == "l1":
            norm_val = np.sum(np.abs(combined))
            if norm_val > 1e-7:
                combined /= norm_val

        return combined.astype(np.float32)

    def extract_batch(self, images: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """
        Extract color features for a batch of images.

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
        """Calculate total output feature vector dimension."""
        num_spaces = len(self.color_spaces)
        if self.mode == "joint":
            return num_spaces * (self.bins_per_channel ** 3)
        return num_spaces * 3 * self.bins_per_channel

    def compute_channel_histograms_dict(
        self,
        image: np.ndarray,
    ) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
        """
        Compute individual channel histograms suitable for visualization plotting.

        Returns:
            Dict mapping space -> channel_name -> (bin_centers, hist_values).
        """
        rgb = self._ensure_rgb(image)
        result: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

        for cs in self.color_spaces:
            converted = self._convert_color_space(rgb, cs)
            ranges = self.CHANNEL_RANGES[cs]
            names = self.CHANNEL_NAMES[cs]
            result[cs] = {}

            for ch_idx in range(3):
                hist = cv2.calcHist(
                    [converted],
                    [ch_idx],
                    None,
                    [self.bins_per_channel],
                    ranges[ch_idx],
                ).flatten()
                norm_hist = hist / (hist.sum() + 1e-7)
                bin_edges = np.linspace(ranges[ch_idx][0], ranges[ch_idx][1], self.bins_per_channel + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                result[cs][names[ch_idx]] = (bin_centers, norm_hist)

        return result

    def get_config(self) -> dict[str, Any]:
        """Return configuration dictionary."""
        return {
            "color_spaces": self.color_spaces,
            "bins_per_channel": self.bins_per_channel,
            "norm": self.norm,
            "mode": self.mode,
            "feature_dim": self.get_feature_dimension(),
        }
