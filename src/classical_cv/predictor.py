"""
Inference and Prediction Module for Classical Vehicle Recognition.
Produces Top-1 and Top-5 predictions with calibrated confidence scores,
hierarchical vehicle metadata (make, model, year), and timing benchmarks.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.classical_cv.evaluator import ClassicalEvaluator
from src.classical_cv.feature_fusion import FeatureFusionPipeline
from src.data.preprocessing import load_image
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class ClassicalVehiclePredictor:
    """
    Inference engine for classical CV vehicle recognition pipelines.
    """

    def __init__(
        self,
        model_path: str | Path,
        pipeline_path: str | Path | None = None,
        class_mapping_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        """
        Args:
            model_path: Path to serialized scikit-learn classifier (.joblib).
            pipeline_path: Path to serialized FeatureFusionPipeline (.joblib).
            class_mapping_path: Path to class mapping CSV/JSON.
            project_root: Project root path.
        """
        self.root = Path(project_root).resolve() if project_root else find_project_root()
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load classifier
        model_dict = joblib.load(self.model_path)
        if isinstance(model_dict, dict) and "classifier" in model_dict:
            self.classifier = model_dict["classifier"]
            self.model_metadata = model_dict.get("metadata", {})
            stored_pipe = model_dict.get("pipeline")
        else:
            self.classifier = model_dict
            self.model_metadata = {}
            stored_pipe = None

        # Load feature pipeline
        if pipeline_path is not None and Path(pipeline_path).is_file():
            self.pipeline = FeatureFusionPipeline.load_pipeline(pipeline_path)
        elif stored_pipe is not None:
            self.pipeline = stored_pipe
        else:
            # Look in default scalers dir based on model name
            candidate = self.root / "models" / "classical" / "scalers" / f"{self.model_path.stem}_pipeline.joblib"
            if candidate.is_file():
                self.pipeline = FeatureFusionPipeline.load_pipeline(candidate)
            else:
                logger.warning("No fitted pipeline found; creating default un-fitted pipeline")
                self.pipeline = FeatureFusionPipeline()

        self.evaluator = ClassicalEvaluator(class_mapping_path=class_mapping_path, project_root=self.root)
        self.classes_ = getattr(self.classifier, "classes_", np.arange(196))
        self.is_calibrated = hasattr(self.classifier, "predict_proba")

    def _ensure_image_array(self, image: np.ndarray | str | Path) -> np.ndarray:
        """Ensure input is loaded RGB numpy array."""
        if isinstance(image, (str, Path)):
            p = Path(image)
            full_p = p if p.is_file() else self.root / p
            return load_image(full_p, as_rgb=True)
        if isinstance(image, np.ndarray):
            return image
        raise TypeError(f"Unsupported image type: {type(image)}")

    def predict(
        self,
        image: np.ndarray | str | Path,
        bbox: Any = None,
        use_bbox: bool = True,
        is_matlab_coord: bool = False,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Execute full classical vehicle recognition on a single image.

        Args:
            image: Image array or file path.
            bbox: Optional bounding box [x1, y1, x2, y2] or dict.
            use_bbox: Whether to crop vehicle ROI.
            is_matlab_coord: Whether bounding box is 1-based MATLAB coordinates.
            top_k: Number of ranked predictions to return.

        Returns:
            Structured prediction dictionary.
        """
        img_arr = self._ensure_image_array(image)

        # 1. Feature extraction timing
        t0 = time.perf_counter()
        feat = self.pipeline.extract_and_transform_single(
            image=img_arr,
            bbox=bbox,
            use_bbox=use_bbox,
            is_matlab_coord=is_matlab_coord,
        )
        feat_time_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Classifier inference timing
        t1 = time.perf_counter()
        feat_2d = feat.reshape(1, -1)

        if self.is_calibrated:
            raw_scores = self.classifier.predict_proba(feat_2d)[0]
        elif hasattr(self.classifier, "decision_function"):
            decision = self.classifier.decision_function(feat_2d)[0]
            # Softmax or min-max normalization for relative score output
            exp_scores = np.exp(decision - np.max(decision))
            raw_scores = exp_scores / np.sum(exp_scores)
        else:
            pred_class = self.classifier.predict(feat_2d)[0]
            raw_scores = np.zeros(len(self.classes_))
            match_idx = np.where(self.classes_ == pred_class)[0]
            if len(match_idx) > 0:
                raw_scores[match_idx[0]] = 1.0

        infer_time_ms = (time.perf_counter() - t1) * 1000.0

        # 3. Top-K ranking
        k_val = min(top_k, len(self.classes_))
        ranked_indices = np.argsort(raw_scores)[::-1][:k_val]

        top_candidates: list[dict[str, Any]] = []
        for rank, idx in enumerate(ranked_indices):
            c_id = int(self.classes_[idx])
            c_score = float(raw_scores[idx])
            info = self.evaluator.get_class_info(c_id)
            top_candidates.append({
                "rank": rank + 1,
                "class_id": c_id,
                "original_class_id": info["original_class_id"],
                "class_name": info["class_name"],
                "make": info["make"],
                "model": info["model"],
                "year": info["year"],
                "score": round(c_score, 4),
            })

        best = top_candidates[0]

        return {
            "class_id": best["class_id"],
            "original_class_id": best["original_class_id"],
            "class_name": best["class_name"],
            "make": best["make"],
            "model": best["model"],
            "year": best["year"],
            "confidence": best["score"],
            "is_calibrated": self.is_calibrated,
            "top_5": top_candidates,
            "feature_extraction_time_ms": round(feat_time_ms, 2),
            "classifier_inference_time_ms": round(infer_time_ms, 2),
            "total_latency_ms": round(feat_time_ms + infer_time_ms, 2),
        }

    def predict_batch(
        self,
        images: list[np.ndarray | str | Path],
        bboxes: list[Any] | None = None,
        use_bbox: bool = True,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Batch prediction for a list of images.
        """
        results: list[dict[str, Any]] = []
        for idx, img in enumerate(images):
            box = bboxes[idx] if (bboxes is not None and idx < len(bboxes)) else None
            res = self.predict(image=img, bbox=box, use_bbox=use_bbox, top_k=top_k)
            results.append(res)
        return results

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        use_bbox: bool = True,
        top_k: int = 5,
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """
        Run predictions over a manifest DataFrame and return results as a DataFrame.
        """
        sub_df = df.iloc[:max_samples] if max_samples is not None else df
        rows: list[dict[str, Any]] = []

        for _, row in sub_df.iterrows():
            img_path = str(row["image_path"])
            is_matlab = bool("bbox_matlab_x1" in row and "bbox_x1" not in row and "bbox_xmin" not in row)
            pred = self.predict(
                image=img_path,
                bbox=row,
                use_bbox=use_bbox,
                is_matlab_coord=is_matlab,
                top_k=top_k,
            )
            top_candidates = pred.get("top_5", [])
            row_dict = {
                "image_id": row.get("image_id", ""),
                "image_path": img_path,
                "predicted_class_id": pred["class_id"],
                "predicted_class_name": pred["class_name"],
                "predicted_make": pred["make"],
                "predicted_model": pred["model"],
                "predicted_year": pred["year"],
                "confidence": pred["confidence"],
                "is_calibrated": pred["is_calibrated"],
                "feature_latency_ms": pred["feature_extraction_time_ms"],
                "classifier_latency_ms": pred["classifier_inference_time_ms"],
            }
            # Record Top-2 through Top-5 alternatives and scores
            for alt_rank in range(2, 6):
                cand_idx = alt_rank - 1
                if cand_idx < len(top_candidates):
                    cand = top_candidates[cand_idx]
                    row_dict[f"top{alt_rank}_class_id"] = cand["class_id"]
                    row_dict[f"top{alt_rank}_class_name"] = cand["class_name"]
                    row_dict[f"top{alt_rank}_score"] = cand["score"]
                else:
                    row_dict[f"top{alt_rank}_class_id"] = None
                    row_dict[f"top{alt_rank}_class_name"] = None
                    row_dict[f"top{alt_rank}_score"] = None

            rows.append(row_dict)

        return pd.DataFrame(rows)
