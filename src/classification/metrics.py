"""
Comprehensive Evaluation Metrics and Attribute-Level Decoders for Vehicle Classification.
Computes Top-1, Top-5, precision, recall, macro/weighted F1, confusion matrix,
per-class accuracy, and decodes 196-class predictions into Make, Model, and Year attributes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.utils.config import find_project_root


class ClassMappingDecoder:
    """
    Decodes Stanford Cars 196 class IDs into full class names, makes, models, and manufacturing years.
    Source of truth: data/processed/class_mapping.json or data/metadata/class_mapping.csv.
    """

    def __init__(self, mapping_path: str | Path | None = None) -> None:
        self.root = find_project_root()
        if mapping_path is None:
            p_json = self.root / "data" / "processed" / "class_mapping.json"
            p_csv = self.root / "data" / "metadata" / "class_mapping.csv"
            mapping_path = p_json if p_json.is_file() else p_csv

        self.mapping_path = Path(mapping_path)
        if not self.mapping_path.is_file():
            self.mapping_path = self.root / self.mapping_path
        if not self.mapping_path.is_file():
            raise FileNotFoundError(f"Class mapping file not found at: {mapping_path}")

        self.class_info: dict[int, dict[str, Any]] = {}
        self.class_names: list[str] = [""] * 196
        self.makes: list[str] = [""] * 196
        self.models: list[str] = [""] * 196
        self.years: list[int] = [0] * 196

        self._load()

    def _load(self) -> None:
        if self.mapping_path.suffix.lower() == ".json":
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("classes", data) if isinstance(data, dict) else data
        else:
            df = pd.read_csv(self.mapping_path)
            records = df.to_dict(orient="records")

        for item in records:
            cid = int(item["class_id"])
            cname = str(item.get("class_name", ""))
            make = str(item.get("make", ""))
            model = str(item.get("clean_model", item.get("model", "")))
            year = int(item.get("year", 0))

            info = {
                "class_id": cid,
                "class_name": cname,
                "make": make,
                "model": model,
                "year": year,
                "body_type": item.get("body_type", ""),
            }
            self.class_info[cid] = info
            if 0 <= cid < 196:
                self.class_names[cid] = cname
                self.makes[cid] = make
                self.models[cid] = model
                self.years[cid] = year

    def get_info(self, class_id: int) -> dict[str, Any]:
        """Return metadata dictionary for a class ID."""
        if class_id in self.class_info:
            return dict(self.class_info[class_id])
        return {
            "class_id": class_id,
            "class_name": f"Unknown_Class_{class_id}",
            "make": "Unknown",
            "model": "Unknown",
            "year": 0,
            "body_type": "Unknown",
        }

    def decode_attributes(self, class_id: int) -> tuple[str, str, int]:
        """Return (make, model, year) tuple for a class ID."""
        info = self.get_info(class_id)
        return info["make"], info["model"], info["year"]


def calculate_classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    y_probs: np.ndarray | None = None,
    decoder: ClassMappingDecoder | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Calculate full suite of classification and diagnostic metrics.

    Args:
        y_true: Ground truth class IDs (shape: [N]).
        y_pred: Predicted class IDs (shape: [N]).
        y_probs: Softmax prediction probabilities (shape: [N, C] or None).
        decoder: Optional ClassMappingDecoder for attribute decoding and names.
        top_k: Top-K rank for accuracy calculation (default: 5).

    Returns:
        Structured dictionary containing metrics, attribute accuracies, and diagnostics.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    total_samples = len(y_true_arr)

    if total_samples == 0:
        return {"total_samples": 0}

    if decoder is None:
        decoder = ClassMappingDecoder()

    # 1. Top-1 Accuracy
    correct_top1 = (y_pred_arr == y_true_arr)
    top1_accuracy = float(np.mean(correct_top1))

    # 2. Top-K Accuracy
    topk_accuracy = top1_accuracy
    if y_probs is not None and y_probs.ndim == 2:
        k = min(top_k, y_probs.shape[1])
        top_k_preds = np.argsort(y_probs, axis=1)[:, -k:]
        correct_topk = np.any(top_k_preds == y_true_arr[:, None], axis=1)
        topk_accuracy = float(np.mean(correct_topk))

    # 3. Precision, Recall, F1
    macro_precision = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_precision = float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    macro_recall = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_recall = float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

    # 4. Per-Class Accuracy
    num_classes = 196
    per_class_acc: dict[str, float] = {}
    for cid in range(num_classes):
        mask = (y_true_arr == cid)
        count = int(np.sum(mask))
        if count > 0:
            acc = float(np.mean(y_pred_arr[mask] == cid))
            cname = decoder.class_names[cid] if cid < len(decoder.class_names) else f"Class_{cid}"
            per_class_acc[cname] = round(acc, 4)

    # 5. Confusion Matrix & Top Confused Classes
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(num_classes)))
    confused_pairs = []
    np.fill_diagonal(cm, 0)  # zero diagonal to isolate errors
    error_indices = np.argwhere(cm > 0)
    for true_id, pred_id in error_indices:
        count = int(cm[true_id, pred_id])
        confused_pairs.append({
            "true_class_id": int(true_id),
            "true_class_name": decoder.class_names[true_id] if true_id < len(decoder.class_names) else f"Class_{true_id}",
            "pred_class_id": int(pred_id),
            "pred_class_name": decoder.class_names[pred_id] if pred_id < len(decoder.class_names) else f"Class_{pred_id}",
            "count": count,
        })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    # 6. Attribute-level Evaluation (Make, Model, Year)
    # Important: These are attribute-level metrics derived from the original 196-class prediction.
    true_makes = [decoder.makes[cid] if 0 <= cid < 196 else "" for cid in y_true_arr]
    pred_makes = [decoder.makes[cid] if 0 <= cid < 196 else "" for cid in y_pred_arr]
    make_accuracy = float(np.mean([t == p for t, p in zip(true_makes, pred_makes, strict=False)]))

    true_models = [decoder.models[cid] if 0 <= cid < 196 else "" for cid in y_true_arr]
    pred_models = [decoder.models[cid] if 0 <= cid < 196 else "" for cid in y_pred_arr]
    model_accuracy = float(np.mean([t == p for t, p in zip(true_models, pred_models, strict=False)]))

    true_years = [decoder.years[cid] if 0 <= cid < 196 else -1 for cid in y_true_arr]
    pred_years = [decoder.years[cid] if 0 <= cid < 196 else -1 for cid in y_pred_arr]
    year_accuracy = float(np.mean([t == p for t, p in zip(true_years, pred_years, strict=False)]))

    # 7. Confidence Distribution Diagnostics
    conf_stats: dict[str, Any] = {}
    if y_probs is not None:
        max_probs = np.max(y_probs, axis=1)
        conf_stats = {
            "mean_confidence": float(np.mean(max_probs)),
            "std_confidence": float(np.std(max_probs)),
            "median_confidence": float(np.median(max_probs)),
            "min_confidence": float(np.min(max_probs)),
            "max_confidence": float(np.max(max_probs)),
            "percentiles": {
                "25": float(np.percentile(max_probs, 25)),
                "50": float(np.percentile(max_probs, 50)),
                "75": float(np.percentile(max_probs, 75)),
                "90": float(np.percentile(max_probs, 90)),
                "95": float(np.percentile(max_probs, 95)),
            },
        }

    return {
        "total_samples": total_samples,
        "top1_accuracy": round(top1_accuracy, 4),
        f"top{top_k}_accuracy": round(topk_accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "weighted_precision": round(weighted_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "weighted_recall": round(weighted_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "attribute_metrics": {
            "make_accuracy": round(make_accuracy, 4),
            "model_accuracy": round(model_accuracy, 4),
            "year_accuracy": round(year_accuracy, 4),
            "derivation_note": (
                "Attribute-level metrics (Make, Model, Year) are strictly decoded "
                "from the primary 196-class predictions using the verified class mapping, "
                "not independently trained attribute heads."
            ),
        },
        "confidence_statistics": conf_stats,
        "top_confused_classes": confused_pairs[:20],
        "per_class_accuracy_summary": {
            "evaluated_classes": len(per_class_acc),
            "mean_per_class_acc": round(float(np.mean(list(per_class_acc.values()))) if per_class_acc else 0.0, 4),
        },
    }
