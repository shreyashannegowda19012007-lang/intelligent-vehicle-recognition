"""
ORB (Oriented FAST and Rotated BRIEF) Feature Extraction and Bag-of-Visual-Words (BoVW).
Provides keypoint detection, descriptor extraction, visual vocabulary generation via MiniBatchKMeans,
histogram quantization, and keypoint visualization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.data.preprocessing import to_grayscale

logger = logging.getLogger(__name__)


class ORBFeatureExtractor:
    """
    ORB feature extractor with Bag-of-Visual-Words (BoVW) spatial quantization.
    """

    def __init__(
        self,
        n_features: int = 500,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        vocab_size: int = 100,
        norm: str = "l2",
    ) -> None:
        """
        Args:
            n_features: Maximum number of keypoints to retain.
            scale_factor: Pyramid decimation ratio.
            n_levels: Number of pyramid levels.
            vocab_size: Number of visual words in KMeans vocabulary.
            norm: Normalization method for BoVW histogram ('l2', 'l1', or 'none').
        """
        self.n_features = int(n_features)
        self.scale_factor = float(scale_factor)
        self.n_levels = int(n_levels)
        self.vocab_size = int(vocab_size)
        self.norm = norm.lower()

        self._orb: cv2.ORB | None = None
        self.kmeans: MiniBatchKMeans | None = None

    @property
    def orb(self) -> cv2.ORB:
        """Lazy-loaded OpenCV ORB detector (handles pickling/serialization)."""
        if self._orb is None:
            self._orb = cv2.ORB_create(
                nfeatures=self.n_features,
                scaleFactor=self.scale_factor,
                nlevels=self.n_levels,
            )
        return self._orb

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_orb"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._orb = None

    def detect_and_compute(
        self,
        image: np.ndarray,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        """
        Detect keypoints and compute ORB descriptors for an image.

        Args:
            image: RGB or grayscale image.

        Returns:
            Tuple of (list_of_keypoints, descriptor_array_or_None).
        """
        gray = to_grayscale(image)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def get_descriptor_statistics(
        self,
        image: np.ndarray,
    ) -> dict[str, Any]:
        """
        Compute summary statistics of detected keypoints and descriptors.
        """
        keypoints, descriptors = self.detect_and_compute(image)
        count = len(keypoints) if keypoints is not None else 0
        if count == 0 or descriptors is None:
            return {
                "keypoint_count": 0,
                "mean_response": 0.0,
                "mean_size": 0.0,
                "descriptor_shape": (0, 32),
            }

        responses = [kp.response for kp in keypoints]
        sizes = [kp.size for kp in keypoints]

        return {
            "keypoint_count": count,
            "mean_response": float(np.mean(responses)),
            "mean_size": float(np.mean(sizes)),
            "descriptor_shape": descriptors.shape,
        }

    def draw_keypoints(
        self,
        image: np.ndarray,
        max_keypoints: int = 100,
    ) -> np.ndarray:
        """
        Draw detected ORB keypoints on image for visualization.

        Args:
            image: RGB uint8 image.
            max_keypoints: Maximum number of strongest keypoints to draw.

        Returns:
            RGB visualization image.
        """
        keypoints, _ = self.detect_and_compute(image)
        if not keypoints:
            return image.copy()

        # Sort by response magnitude and keep top N
        sorted_kps = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:max_keypoints]

        # Draw keypoints with rich circles and orientation
        viz = cv2.drawKeypoints(
            image,
            sorted_kps,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        return viz

    def fit_vocabulary(
        self,
        training_images: list[np.ndarray],
        max_descriptors: int = 50000,
        random_state: int = 42,
    ) -> MiniBatchKMeans:
        """
        Fit a Bag-of-Visual-Words KMeans vocabulary using training images only.

        Args:
            training_images: List of training image arrays.
            max_descriptors: Maximum number of descriptors sampled for clustering.
            random_state: Random seed for reproducibility.

        Returns:
            Fitted MiniBatchKMeans model.
        """
        all_descriptors: list[np.ndarray] = []
        for img in training_images:
            _, des = self.detect_and_compute(img)
            if des is not None and len(des) > 0:
                all_descriptors.append(des.astype(np.float32))

        if not all_descriptors:
            raise ValueError("No ORB descriptors found in training images to fit vocabulary.")

        stacked = np.vstack(all_descriptors)

        if len(stacked) > max_descriptors:
            rng = np.random.default_rng(random_state)
            indices = rng.choice(len(stacked), size=max_descriptors, replace=False)
            stacked = stacked[indices]

        # Ensure clusters do not exceed available points
        n_clusters = min(self.vocab_size, len(stacked))
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1024,
            random_state=random_state,
            n_init="auto",
        )
        self.kmeans.fit(stacked)
        self.vocab_size = len(self.kmeans.cluster_centers_)
        logger.info("Fitted ORB visual vocabulary with %d words from %d descriptors", n_clusters, len(stacked))
        return self.kmeans

    def transform_to_bovw(
        self,
        descriptors: np.ndarray | None,
    ) -> np.ndarray:
        """
        Transform raw descriptors into a fixed-length visual word histogram.

        Args:
            descriptors: (N, 32) uint8 or float32 descriptors.

        Returns:
            1D float32 normalized histogram of length vocab_size.
        """
        hist = np.zeros(self.vocab_size, dtype=np.float32)
        if descriptors is None or len(descriptors) == 0 or self.kmeans is None:
            return hist

        des_float = descriptors.astype(np.float32)
        words = self.kmeans.predict(des_float)

        for w in words:
            if 0 <= w < self.vocab_size:
                hist[w] += 1.0

        if self.norm == "l2":
            norm_val = np.linalg.norm(hist)
            if norm_val > 1e-7:
                hist /= norm_val
        elif self.norm == "l1":
            norm_val = np.sum(np.abs(hist))
            if norm_val > 1e-7:
                hist /= norm_val

        return hist

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract BoVW histogram for a single image.

        Args:
            image: Input image array.

        Returns:
            1D float32 numpy array.
        """
        _, descriptors = self.detect_and_compute(image)
        return self.transform_to_bovw(descriptors)

    def extract_batch(self, images: list[np.ndarray] | np.ndarray) -> np.ndarray:
        """
        Extract BoVW features for a batch of images.
        """
        features = [self.extract_features(img) for img in images]
        if not features:
            return np.empty((0, self.vocab_size), dtype=np.float32)
        return np.vstack(features).astype(np.float32)

    def save_vocabulary(self, file_path: str | Path) -> None:
        """Save fitted KMeans vocabulary to disk."""
        if self.kmeans is None:
            raise ValueError("Vocabulary has not been fitted yet.")
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.kmeans, p)
        logger.info("Saved visual vocabulary to %s", p)

    def load_vocabulary(self, file_path: str | Path) -> None:
        """Load fitted KMeans vocabulary from disk."""
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Vocabulary file not found: {p}")
        self.kmeans = joblib.load(p)
        self.vocab_size = len(self.kmeans.cluster_centers_)
        logger.info("Loaded visual vocabulary with %d words from %s", self.vocab_size, p)

    def get_feature_dimension(self) -> int:
        """Return visual vocabulary histogram dimension."""
        return self.vocab_size
