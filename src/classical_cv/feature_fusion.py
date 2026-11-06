"""
Feature Fusion and Normalization Pipeline for Classical Computer Vision.
Combines HOG, Color Histograms, LBP, and ORB BoVW descriptors into a unified representation.
Applies StandardScaler and optional PCA fitted strictly on training data to prevent test leakage.
Implements disk caching for efficient repeated experimentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.classical_cv.color_features import ColorHistogramExtractor
from src.classical_cv.hog_features import HOGFeatureExtractor
from src.classical_cv.orb_features import ORBFeatureExtractor
from src.classical_cv.roi_extractor import ROIExtractor
from src.classical_cv.texture_features import LBPFeatureExtractor
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class FeatureFusionPipeline:
    """
    Orchestrates extraction, fusion, scaling, and dimensionality reduction
    of classical computer vision descriptors.
    """

    SUPPORTED_CONFIGS = (
        "hog",
        "color",
        "lbp",
        "orb",
        "hog_color",
        "hog_lbp",
        "hog_orb",
        "hog_color_lbp",
        "all",
        "hog_color_lbp_orb",
    )

    def __init__(
        self,
        feature_config: str = "hog_color_lbp",
        roi_extractor: ROIExtractor | None = None,
        hog_extractor: HOGFeatureExtractor | None = None,
        color_extractor: ColorHistogramExtractor | None = None,
        lbp_extractor: LBPFeatureExtractor | None = None,
        orb_extractor: ORBFeatureExtractor | None = None,
        use_scaler: bool = True,
        use_pca: bool = False,
        pca_components: float | None = 128,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            feature_config: Descriptor combination: 'hog', 'color', 'lbp', 'hog_color',
                            'hog_lbp', 'hog_color_lbp', 'all'.
            roi_extractor: Vehicle ROI extractor.
            hog_extractor: HOG feature extractor.
            color_extractor: Color histogram extractor.
            lbp_extractor: LBP texture extractor.
            orb_extractor: ORB BoVW extractor.
            use_scaler: If True, fits and applies StandardScaler.
            use_pca: If True, fits and applies PCA dimensionality reduction.
            pca_components: Number of PCA components or variance ratio.
            random_state: Random seed for PCA.
        """
        self.feature_config = feature_config.lower()
        if self.feature_config not in self.SUPPORTED_CONFIGS:
            raise ValueError(f"Unsupported feature_config: {feature_config}. Choose from {self.SUPPORTED_CONFIGS}")

        self.roi_extractor = roi_extractor or ROIExtractor()
        self.hog_extractor = hog_extractor or HOGFeatureExtractor()
        self.color_extractor = color_extractor or ColorHistogramExtractor()
        self.lbp_extractor = lbp_extractor or LBPFeatureExtractor()
        self.orb_extractor = orb_extractor or ORBFeatureExtractor()

        self.use_scaler = use_scaler
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.random_state = random_state

        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.is_fitted = False

    def extract_raw_descriptors(
        self,
        image: np.ndarray,
        bbox: Any = None,
        use_bbox: bool = True,
        is_matlab_coord: bool = False,
    ) -> np.ndarray:
        """
        Extract and concatenate raw feature descriptors from an image.

        Args:
            image: Full or cropped image array.
            bbox: Bounding box coordinates if cropping ROI.
            use_bbox: Whether to crop vehicle ROI.
            is_matlab_coord: Whether bounding box is in 1-based MATLAB coordinates.

        Returns:
            1D float32 numpy array.
        """
        # 1. Extract vehicle ROI
        roi, _ = self.roi_extractor.extract_roi(
            image=image,
            bbox=bbox,
            use_bbox=use_bbox,
            is_matlab_coord=is_matlab_coord,
        )

        # 2. Extract selected descriptors
        chunks: list[np.ndarray] = []

        if "hog" in self.feature_config or self.feature_config == "all":
            hog_feat = self.hog_extractor.extract_features(roi)
            chunks.append(hog_feat)

        if "color" in self.feature_config or self.feature_config == "all":
            color_feat = self.color_extractor.extract_features(roi)
            chunks.append(color_feat)

        if "lbp" in self.feature_config or self.feature_config == "all":
            lbp_feat = self.lbp_extractor.extract_features(roi)
            chunks.append(lbp_feat)

        if "orb" in self.feature_config or self.feature_config == "all" or self.feature_config == "hog_color_lbp_orb":
            orb_feat = self.orb_extractor.extract_features(roi)
            chunks.append(orb_feat)

        if not chunks:
            raise ValueError(f"No feature extractors active for config '{self.feature_config}'")

        return np.concatenate(chunks).astype(np.float32)

    def fit(self, X_train: np.ndarray) -> FeatureFusionPipeline:
        """
        Fit scaler and PCA on training features ONLY to prevent data leakage.
        """
        current_X = X_train

        if self.use_scaler:
            self.scaler = StandardScaler()
            current_X = self.scaler.fit_transform(current_X)
            logger.info("Fitted StandardScaler on %d training samples (dim=%d)", len(X_train), X_train.shape[1])

        if self.use_pca and self.pca_components is not None:
            # PCA n_components cannot exceed min(n_samples, n_features)
            n_samples, n_features = current_X.shape
            max_components = min(n_samples, n_features)
            n_comp = self.pca_components
            if isinstance(n_comp, int) and n_comp > max_components:
                n_comp = max_components
                logger.warning("Clamped PCA components to %d due to dataset dimensions (%d, %d)", n_comp, n_samples, n_features)

            self.pca = PCA(n_components=n_comp, random_state=self.random_state)
            self.pca.fit(current_X)
            explained_var = float(np.sum(self.pca.explained_variance_ratio_))
            logger.info("Fitted PCA with %s components (Explained variance: %.4f)", str(n_comp), explained_var)

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted scaler and PCA to feature matrix.
        """
        current_X = X.copy()
        if self.scaler is not None:
            current_X = self.scaler.transform(current_X)

        if self.pca is not None:
            current_X = self.pca.transform(current_X)

        return current_X.astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit and transform training features."""
        self.fit(X_train)
        return self.transform(X_train)

    def extract_and_transform_single(
        self,
        image: np.ndarray,
        bbox: Any = None,
        use_bbox: bool = True,
        is_matlab_coord: bool = False,
    ) -> np.ndarray:
        """
        Extract raw features and apply fitted transform for a single image.
        """
        raw = self.extract_raw_descriptors(
            image=image,
            bbox=bbox,
            use_bbox=use_bbox,
            is_matlab_coord=is_matlab_coord,
        )
        if not self.is_fitted:
            return raw

        matrix = raw.reshape(1, -1)
        transformed = self.transform(matrix)
        return transformed[0]

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        project_root: str | Path | None = None,
        cache_path: str | Path | None = None,
        use_bbox: bool = True,
        max_samples: int | None = None,
        fit_orb_vocab: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Extract raw features for all images in DataFrame with automatic disk caching.

        Args:
            df: Manifest split DataFrame.
            project_root: Project root to resolve image paths.
            cache_path: File path to save/load cached .npz features.
            use_bbox: Whether to crop vehicle ROI.
            max_samples: If set, limits number of samples for rapid benchmarking.
            fit_orb_vocab: If True and ORB is enabled, fits vocabulary on training images.

        Returns:
            Tuple of (features_array (N, D), labels_array (N,), image_ids_list).
        """
        root = Path(project_root).resolve() if project_root else find_project_root()
        sub_df = df.iloc[:max_samples] if max_samples is not None else df

        # Check disk cache
        if cache_path is not None:
            cp = Path(cache_path)
            if cp.is_file():
                try:
                    logger.info("Loading cached features from %s", cp)
                    data = np.load(cp, allow_pickle=True)
                    cached_x = data["features"]
                    cached_y = data["labels"]
                    cached_ids = list(data["image_ids"])
                    if len(cached_x) == len(sub_df):
                        return cached_x.astype(np.float32), cached_y, cached_ids
                    logger.warning("Cache size mismatch (%d vs %d); re-extracting", len(cached_x), len(sub_df))
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to load cache %s: %s; re-extracting", cp, e)

        # Pre-fit ORB vocabulary if required
        has_orb = ("orb" in self.feature_config or self.feature_config == "all")
        if fit_orb_vocab and has_orb and self.orb_extractor.kmeans is None:
            logger.info("Fitting ORB vocabulary on training set...")
            sample_images: list[np.ndarray] = []
            sample_count = min(300, len(sub_df))
            for _, row in sub_df.head(sample_count).iterrows():
                crop, _ = self.roi_extractor.extract_from_row(row, project_root=root, use_bbox=use_bbox)
                sample_images.append(crop)
            self.orb_extractor.fit_vocabulary(sample_images)

        features_list: list[np.ndarray] = []
        labels_list: list[int] = []
        image_ids_list: list[str] = []

        total_count = len(sub_df)
        logger.info("Extracting %s features for %d samples (use_bbox=%s)...", self.feature_config, total_count, use_bbox)

        for idx, (_, row) in enumerate(sub_df.iterrows()):
            try:
                crop, _meta = self.roi_extractor.extract_from_row(
                    row=row,
                    project_root=root,
                    use_bbox=use_bbox,
                )
                feat = self.extract_raw_descriptors(crop, use_bbox=False)
                features_list.append(feat)

                # Class label (-1 if missing/test)
                cid = int(row["class_id"]) if pd.notna(row.get("class_id")) else -1
                labels_list.append(cid)
                image_ids_list.append(str(row.get("image_id", f"img_{idx}")))
            except Exception as e:  # noqa: BLE001
                logger.error("Error extracting features for image %s: %s", row.get("image_path", idx), e)
                continue

            if (idx + 1) % 500 == 0 or (idx + 1) == total_count:
                logger.info("Processed %d / %d images", idx + 1, total_count)

        if not features_list:
            raise RuntimeError("No features could be extracted from DataFrame.")

        X = np.vstack(features_list).astype(np.float32)
        y = np.array(labels_list, dtype=np.int64)

        # Save to cache
        if cache_path is not None:
            cp = Path(cache_path)
            cp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cp, features=X, labels=y, image_ids=np.array(image_ids_list))
            logger.info("Saved feature cache to %s (%d samples, %d features)", cp, len(X), X.shape[1])

        return X, y, image_ids_list

    def save_pipeline(self, file_path: str | Path) -> None:
        """Serialize fitted pipeline to disk."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_config": self.feature_config,
            "use_scaler": self.use_scaler,
            "use_pca": self.use_pca,
            "pca_components": self.pca_components,
            "random_state": self.random_state,
            "scaler": self.scaler,
            "pca": self.pca,
            "is_fitted": self.is_fitted,
            "roi_config": {
                "target_size": self.roi_extractor.target_size,
                "keep_aspect_ratio": self.roi_extractor.keep_aspect_ratio,
                "padding_ratio": self.roi_extractor.padding_ratio,
            },
            "hog_config": self.hog_extractor.get_config(),
            "color_config": self.color_extractor.get_config(),
            "lbp_config": self.lbp_extractor.get_config(),
            "orb_config": {
                "n_features": self.orb_extractor.n_features,
                "scale_factor": self.orb_extractor.scale_factor,
                "n_levels": self.orb_extractor.n_levels,
                "vocab_size": self.orb_extractor.vocab_size,
                "norm": self.orb_extractor.norm,
            },
            "orb_vocab": self.orb_extractor.kmeans,
        }
        joblib.dump(state, p)
        logger.info("Saved FeatureFusionPipeline to %s", p)

    @classmethod
    def load_pipeline(cls, file_path: str | Path) -> FeatureFusionPipeline:
        """Load serialized pipeline from disk."""
        p = Path(file_path)
        if not p.is_file():
            raise FileNotFoundError(f"Pipeline file not found at: {p}")

        state = joblib.load(p)
        roi_cfg = state.get("roi_config", {})
        roi_ext = ROIExtractor(
            target_size=roi_cfg.get("target_size", (128, 128)),
            keep_aspect_ratio=roi_cfg.get("keep_aspect_ratio", True),
            padding_ratio=roi_cfg.get("padding_ratio", 0.05),
        )

        orb_cfg = state.get("orb_config", {})
        orb_ext = ORBFeatureExtractor(
            n_features=orb_cfg.get("n_features", 500),
            scale_factor=orb_cfg.get("scale_factor", 1.2),
            n_levels=orb_cfg.get("n_levels", 8),
            vocab_size=orb_cfg.get("vocab_size", 100),
            norm=orb_cfg.get("norm", "l2"),
        )
        if "orb_vocab" in state and state["orb_vocab"] is not None:
            orb_ext.kmeans = state["orb_vocab"]
            if hasattr(orb_ext.kmeans, "cluster_centers_"):
                orb_ext.vocab_size = len(orb_ext.kmeans.cluster_centers_)

        pipeline = cls(
            feature_config=state["feature_config"],
            roi_extractor=roi_ext,
            orb_extractor=orb_ext,
            use_scaler=state["use_scaler"],
            use_pca=state["use_pca"],
            pca_components=state["pca_components"],
            random_state=state["random_state"],
        )
        pipeline.scaler = state.get("scaler")
        pipeline.pca = state.get("pca")
        pipeline.is_fitted = state.get("is_fitted", False)
        logger.info("Loaded FeatureFusionPipeline from %s (fitted=%s)", p, pipeline.is_fitted)
        return pipeline
