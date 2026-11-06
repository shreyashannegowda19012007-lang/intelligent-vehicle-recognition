"""
Evaluation and Metrics Module for Classical Computer Vision Baseline.
Computes Top-1, Top-5, Macro/Weighted Precision, Recall, F1, Confusion Matrix,
per-class accuracies, inference latency, and detailed failure analysis.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class ClassicalEvaluator:
    """
    Evaluator for classical computer vision models on Stanford Cars dataset.
    """

    def __init__(
        self,
        class_mapping_path: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        """
        Args:
            class_mapping_path: Path to class_mapping.csv or class_mapping.json.
            project_root: Root workspace path.
        """
        self.root = Path(project_root).resolve() if project_root else find_project_root()
        self.mapping_path = Path(class_mapping_path) if class_mapping_path else self.root / "data" / "processed" / "class_mapping.csv"
        self.class_df: pd.DataFrame | None = None
        self._load_class_mapping()

    def _load_class_mapping(self) -> None:
        """Load class metadata table."""
        if self.mapping_path.is_file():
            try:
                if self.mapping_path.suffix == ".json":
                    with open(self.mapping_path, encoding="utf-8") as f:
                        data = json.load(f)
                    self.class_df = pd.DataFrame(data.get("classes", []))
                else:
                    self.class_df = pd.read_csv(self.mapping_path)
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not load class mapping from %s: %s", self.mapping_path, e)

    def get_class_info(self, class_id: int) -> dict[str, Any]:
        """Lookup class name, make, model, year by class_id (0-indexed)."""
        if self.class_df is not None and "class_id" in self.class_df.columns:
            matches = self.class_df[self.class_df["class_id"] == class_id]
            if not matches.empty:
                row = matches.iloc[0]
                return {
                    "class_id": int(class_id),
                    "original_class_id": int(row.get("original_class_id", class_id + 1)),
                    "class_name": str(row.get("class_name", f"Class_{class_id}")),
                    "make": str(row.get("make", "Unknown")),
                    "model": str(row.get("model", "Unknown")),
                    "year": int(row.get("year", 0)) if pd.notna(row.get("year")) else 0,
                }
        return {
            "class_id": int(class_id),
            "original_class_id": int(class_id + 1),
            "class_name": f"Class_{class_id}",
            "make": "Unknown",
            "model": "Unknown",
            "year": 0,
        }

    def evaluate(
        self,
        clf: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        image_ids: list[str] | None = None,
        k: int = 5,
    ) -> dict[str, Any]:
        """
        Compute full benchmark evaluation metrics.

        Args:
            clf: Trained classifier (with predict and predict_proba or decision_function).
            X: Feature matrix (N, D).
            y_true: Ground truth class array (N,).
            image_ids: List of image IDs (optional).
            k: Top-K rank (default 5).

        Returns:
            Dict containing accuracy, top-5, precision, recall, F1, latency, and per-class stats.
        """
        n_samples = len(X)
        if n_samples == 0:
            return {"error": "Empty evaluation dataset"}

        # Class labels from classifier
        if hasattr(clf, "classes_"):
            classes = clf.classes_
        else:
            classes = np.unique(y_true)

        # Measure inference latency
        start_time = time.perf_counter()
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X)
            is_calibrated = True
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X)
            if scores.ndim == 1:
                # Binary case
                scores = np.column_stack([-scores, scores])
            is_calibrated = False
        else:
            y_pred_arr = clf.predict(X)
            scores = np.zeros((n_samples, len(classes)), dtype=np.float32)
            for idx, p in enumerate(y_pred_arr):
                match_pos = np.where(classes == p)[0]
                if len(match_pos) > 0:
                    scores[idx, match_pos[0]] = 1.0
            is_calibrated = False

        total_infer_time = time.perf_counter() - start_time
        latency_per_sample_ms = (total_infer_time / n_samples) * 1000.0

        # Rank predictions
        k_val = min(k, scores.shape[1])
        top_k_indices = np.argsort(scores, axis=1)[:, -k_val:][:, ::-1]
        top_k_classes = classes[top_k_indices]
        y_pred = top_k_classes[:, 0]

        # Top-1 and Top-K accuracy
        top1_acc = float(accuracy_score(y_true, y_pred))
        top_k_matches = np.any(top_k_classes == y_true[:, None], axis=1)
        topk_acc = float(np.mean(top_k_matches))

        # Precision, Recall, F1 (macro and weighted)
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Confusion Matrix
        unique_labels = np.unique(np.concatenate([y_true, classes]))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

        # Per-class accuracy
        per_class_acc: dict[int, float] = {}
        for c in np.unique(y_true):
            mask = y_true == c
            c_acc = float(np.mean(y_pred[mask] == c))
            per_class_acc[int(c)] = round(c_acc, 4)

        metrics = {
            "num_samples": n_samples,
            "num_classes_evaluated": len(np.unique(y_true)),
            "top1_accuracy": round(top1_acc, 4),
            "top5_accuracy": round(topk_acc, 4),
            "precision_macro": round(prec_macro, 4),
            "precision_weighted": round(prec_weighted, 4),
            "recall_macro": round(rec_macro, 4),
            "recall_weighted": round(rec_weighted, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "latency_per_sample_ms": round(latency_per_sample_ms, 3),
            "total_inference_time_sec": round(total_infer_time, 3),
            "is_calibrated": is_calibrated,
            "per_class_accuracy": per_class_acc,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": unique_labels.tolist(),
        }

        return metrics

    def analyze_failures(
        self,
        clf: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        image_ids: list[str] | None = None,
        max_samples: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Identify and diagnose genuine misclassified examples.

        Returns:
            List of misclassified item dictionaries.
        """
        n_samples = len(X)
        classes = getattr(clf, "classes_", np.unique(y_true))
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X)
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X)
            if scores.ndim == 1:
                scores = np.column_stack([-scores, scores])
        else:
            y_pred_raw = clf.predict(X)
            scores = np.zeros((n_samples, len(classes)), dtype=np.float32)
            for idx, p in enumerate(y_pred_raw):
                match_pos = np.where(classes == p)[0]
                if len(match_pos) > 0:
                    scores[idx, match_pos[0]] = 1.0

        k_val = min(5, scores.shape[1])
        top_k_indices = np.argsort(scores, axis=1)[:, -k_val:][:, ::-1]
        top_k_classes = classes[top_k_indices]
        y_pred = top_k_classes[:, 0]

        misclassified_indices = np.where(y_true != y_pred)[0]
        failures: list[dict[str, Any]] = []

        for idx in misclassified_indices[:max_samples]:
            gt_id = int(y_true[idx])
            pred_id = int(y_pred[idx])

            gt_info = self.get_class_info(gt_id)
            pred_info = self.get_class_info(pred_id)

            # Analyze failure type
            if gt_info["make"] == pred_info["make"] and gt_info["model"] == pred_info["model"]:
                failure_type = "Year Difference (Same Make & Model)"
            elif gt_info["make"] == pred_info["make"]:
                failure_type = "Model Confusion (Same Make)"
            else:
                failure_type = "Inter-Make Fine-Grained Confusion"

            # Generate relevant visual observations
            gt_year = gt_info.get("year", 0)
            pred_year = pred_info.get("year", 0)
            gt_make = gt_info.get("make", "")
            pred_make = pred_info.get("make", "")
            gt_model = gt_info.get("model", "")
            pred_model = pred_info.get("model", "")

            if failure_type == "Year Difference (Same Make & Model)":
                obs = (
                    f"Subtle facelift trim variance across {abs(gt_year - pred_year)} model years ({gt_year} vs {pred_year}) "
                    f"for {gt_make} {gt_model}; identical global silhouette and aspect ratio; minor cosmetic differences "
                    "in headlight clusters and grille mesh indiscernible by classical spatial histograms."
                )
            elif failure_type == "Model Confusion (Same Make)":
                obs = (
                    f"Shared corporate design language and front fascia within {gt_make} family ({gt_model} vs {pred_model}); "
                    "HOG gradient orientation captures similar brand contours; localized body line cues lack discriminative power."
                )
            else:
                obs = (
                    f"Inter-make silhouette ambiguity between {gt_make} {gt_model} ({gt_year}) and {pred_make} {pred_model} ({pred_year}); "
                    "similar vehicle proportions, viewpoint angle, and exterior lighting dominate hand-crafted descriptors."
                )

            top5_candidates = []
            for rank in range(k_val):
                c_id = int(top_k_classes[idx, rank])
                c_score = float(scores[idx, top_k_indices[idx, rank]])
                c_info = self.get_class_info(c_id)
                top5_candidates.append({
                    "rank": rank + 1,
                    "class_id": c_id,
                    "class_name": c_info["class_name"],
                    "score": round(c_score, 4),
                })

            sample_id = image_ids[idx] if (image_ids is not None and idx < len(image_ids)) else f"sample_{idx}"

            failures.append({
                "sample_index": int(idx),
                "image_id": sample_id,
                "ground_truth": gt_info,
                "predicted": pred_info,
                "predicted_score": round(float(scores[idx, top_k_indices[idx, 0]]), 4),
                "failure_type": failure_type,
                "visual_observations": obs,
                "top_5": top5_candidates,
            })

        return failures

    def save_metrics(self, metrics: dict[str, Any], output_path: str | Path) -> None:
        """Save metrics dictionary to JSON file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Exclude large confusion matrix from top-level summary if writing compact
        with open(p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved evaluation metrics to %s", p)

    def save_failures(self, failures: list[dict[str, Any]], output_path: str | Path) -> None:
        """Save failure analysis list to JSON and CSV."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)

        # Also write flattened CSV summary
        flat_records = []
        for fail in failures:
            flat_records.append({
                "sample_index": fail["sample_index"],
                "image_id": fail["image_id"],
                "gt_class_id": fail["ground_truth"]["class_id"],
                "gt_class_name": fail["ground_truth"]["class_name"],
                "pred_class_id": fail["predicted"]["class_id"],
                "pred_class_name": fail["predicted"]["class_name"],
                "pred_score": fail["predicted_score"],
                "failure_type": fail["failure_type"],
                "visual_observations": fail.get("visual_observations", ""),
            })
        csv_p = p.with_suffix(".csv")
        pd.DataFrame(flat_records).to_csv(csv_p, index=False)
        logger.info("Saved failure cases to %s and %s", p, csv_p)
