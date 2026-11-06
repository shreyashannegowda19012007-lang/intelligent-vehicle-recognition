"""
Hybrid Evaluation and Benchmarking Module.
Compares Classical CV + ML, Deep Learning, and Hybrid Decision Strategies (A, B, C)
across Top-1, Top-5, Make, Model, Year, F1-scores, IoU alignment, and latency.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from src.classification.metrics import ClassMappingDecoder, calculate_classification_metrics
from src.detection.evaluator import compute_iou
from src.hybrid.config import DecisionStrategy, HybridConfig
from src.hybrid.pipeline import HybridVehiclePipeline
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


class HybridEvaluator:
    """
    Evaluator comparing Classical CV, Deep Learning, and Hybrid decision strategies.
    Conducts comparative benchmarks across ground-truth crops and end-to-end YOLO crops.
    """

    def __init__(
        self,
        pipeline: HybridVehiclePipeline | None = None,
        config: HybridConfig | None = None,
        class_mapping_path: str | Path | None = None,
    ) -> None:
        self.root = find_project_root()
        self.config = config or HybridConfig.from_yaml()
        self.pipeline = pipeline or HybridVehiclePipeline(config=self.config)
        self.decoder = ClassMappingDecoder(class_mapping_path or self.config.class_mapping_path)

    def evaluate_on_crops(
        self,
        samples: list[dict[str, Any]],
        strategies: list[DecisionStrategy] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate Classical, Deep Learning, and Hybrid strategies on ground truth vehicle crops.

        Args:
            samples: List of dictionaries with 'image', 'true_class_id', and optional 'true_make', 'true_model', 'true_year'.
            strategies: List of DecisionStrategy values to evaluate.

        Returns:
            Dictionary containing metrics breakdown for all approaches.
        """
        if not samples:
            return {"error": "No evaluation samples provided"}

        eval_strategies = strategies or [
            DecisionStrategy.DEEP_LEARNING_PRIMARY,
            DecisionStrategy.AGREEMENT_BASED,
            DecisionStrategy.SCORE_FUSION,
        ]

        y_true: list[int] = []
        dl_preds: list[int] = []
        dl_confs: list[float] = []
        cl_preds: list[int] = []
        cl_scores: list[float] = []

        hybrid_preds: dict[str, list[int]] = {s.value: [] for s in eval_strategies}
        hybrid_confs: dict[str, list[float]] = {s.value: [] for s in eval_strategies}

        dl_latencies: list[float] = []
        cl_latencies: list[float] = []
        dec_latencies: list[float] = []
        dl_top5_hits = 0
        cl_top5_hits = 0
        agreement_count = 0

        for s in samples:
            crop = s["image"]
            true_cid = int(s["true_class_id"])
            y_true.append(true_cid)

            # 1. DL prediction
            t_dl = time.perf_counter()
            dl_res = self.pipeline.dl_adapter.predict_crop(crop)
            dl_latencies.append((time.perf_counter() - t_dl) * 1000.0)
            dl_preds.append(dl_res.get("class_id", -1))
            dl_confs.append(float(dl_res.get("confidence", 0.0)))
            dl_top5_cids = [item.get("class_id") for item in dl_res.get("top_5", [])]
            if true_cid in dl_top5_cids:
                dl_top5_hits += 1

            # 2. Classical prediction
            t_cl = time.perf_counter()
            cl_res = self.pipeline.classical_adapter.predict_crop(crop)
            cl_latencies.append((time.perf_counter() - t_cl) * 1000.0)
            cl_preds.append(cl_res.get("class_id", -1))
            cl_scores.append(float(cl_res.get("score", 0.0)))
            cl_top5_cids = [item.get("class_id") for item in cl_res.get("top_5", [])]
            if true_cid in cl_top5_cids:
                cl_top5_hits += 1

            # 3. Hybrid decisions for each strategy
            for strat in eval_strategies:
                t_dec = time.perf_counter()
                dec = self.pipeline.decision_engine.decide(
                    vehicle_id=1,
                    bbox={"x1": 0, "y1": 0, "x2": 100, "y2": 100},
                    detection_info={"class_name": "car", "confidence": 1.0},
                    dl_pred=dl_res,
                    classical_pred=cl_res,
                    strategy_override=strat,
                )
                dec_latencies.append((time.perf_counter() - t_dec) * 1000.0)
                final_cid = dec["final_prediction"].get("class_id", -1)
                hybrid_preds[strat.value].append(final_cid)
                hybrid_confs[strat.value].append(float(dec["final_prediction"].get("confidence", 0.0)))

            if dl_res.get("class_id") == cl_res.get("class_id") and dl_res.get("class_id") >= 0:
                agreement_count += 1

        total_evaluated = len(y_true)

        # Compute model sizes
        dl_size_mb = getattr(self.pipeline.dl_adapter, "get_model_size_mb", lambda: 0.0)()
        cl_size_mb = getattr(self.pipeline.classical_adapter, "get_model_size_mb", lambda: 0.0)()
        det_size_mb = getattr(self.pipeline.detector_adapter, "get_model_size_mb", lambda: 0.0)()

        # Compute metrics
        results: dict[str, Any] = {
            "total_samples": total_evaluated,
            "classifier_agreement_rate": round(agreement_count / max(1, total_evaluated), 4),
            "detector": {
                "available": True,
                "model_size_mb": det_size_mb,
            },
            "classical_ml": {
                "available": self.pipeline.classical_adapter.is_available,
                "model_size_mb": cl_size_mb,
                "mean_latency_ms": round(float(np.mean(cl_latencies)), 2) if cl_latencies else 0.0,
            },
            "deep_learning": {
                "available": self.pipeline.dl_adapter.is_available,
                "model_size_mb": dl_size_mb,
                "mean_latency_ms": round(float(np.mean(dl_latencies)), 2) if dl_latencies else 0.0,
            },
            "hybrid_strategies": {},
        }

        # DL Metrics
        if self.pipeline.dl_adapter.is_available:
            dl_metrics = calculate_classification_metrics(y_true, dl_preds, decoder=self.decoder)
            results["deep_learning"].update({
                "top1_accuracy": dl_metrics["top1_accuracy"],
                "top5_accuracy": round(dl_top5_hits / max(1, total_evaluated), 4),
                "make_accuracy": dl_metrics["attribute_metrics"]["make_accuracy"],
                "model_accuracy": dl_metrics["attribute_metrics"]["model_accuracy"],
                "year_accuracy": dl_metrics["attribute_metrics"]["year_accuracy"],
                "macro_precision": dl_metrics["macro_precision"],
                "weighted_precision": dl_metrics["weighted_precision"],
                "macro_recall": dl_metrics["macro_recall"],
                "weighted_recall": dl_metrics["weighted_recall"],
                "macro_f1": dl_metrics["macro_f1"],
                "weighted_f1": dl_metrics["weighted_f1"],
                "confusion_matrix_top_errors": dl_metrics.get("confused_pairs", [])[:10],
            })

        # Classical Metrics
        if self.pipeline.classical_adapter.is_available:
            cl_metrics = calculate_classification_metrics(y_true, cl_preds, decoder=self.decoder)
            results["classical_ml"].update({
                "top1_accuracy": cl_metrics["top1_accuracy"],
                "top5_accuracy": round(cl_top5_hits / max(1, total_evaluated), 4),
                "make_accuracy": cl_metrics["attribute_metrics"]["make_accuracy"],
                "model_accuracy": cl_metrics["attribute_metrics"]["model_accuracy"],
                "year_accuracy": cl_metrics["attribute_metrics"]["year_accuracy"],
                "macro_precision": cl_metrics["macro_precision"],
                "weighted_precision": cl_metrics["weighted_precision"],
                "macro_recall": cl_metrics["macro_recall"],
                "weighted_recall": cl_metrics["weighted_recall"],
                "macro_f1": cl_metrics["macro_f1"],
                "weighted_f1": cl_metrics["weighted_f1"],
                "confusion_matrix_top_errors": cl_metrics.get("confused_pairs", [])[:10],
            })

        # Hybrid Metrics for each strategy
        for strat in eval_strategies:
            preds = hybrid_preds[strat.value]
            strat_metrics = calculate_classification_metrics(y_true, preds, decoder=self.decoder)
            results["hybrid_strategies"][strat.value] = {
                "top1_accuracy": strat_metrics["top1_accuracy"],
                "top5_accuracy": round(dl_top5_hits / max(1, total_evaluated), 4),
                "make_accuracy": strat_metrics["attribute_metrics"]["make_accuracy"],
                "model_accuracy": strat_metrics["attribute_metrics"]["model_accuracy"],
                "year_accuracy": strat_metrics["attribute_metrics"]["year_accuracy"],
                "macro_precision": strat_metrics["macro_precision"],
                "weighted_precision": strat_metrics["weighted_precision"],
                "macro_recall": strat_metrics["macro_recall"],
                "weighted_recall": strat_metrics["weighted_recall"],
                "macro_f1": strat_metrics["macro_f1"],
                "weighted_f1": strat_metrics["weighted_f1"],
                "mean_confidence": round(float(np.mean(hybrid_confs[strat.value])), 4) if hybrid_confs[strat.value] else 0.0,
                "confusion_matrix_top_errors": strat_metrics.get("confused_pairs", [])[:10],
            }

        return results

    def evaluate_pipeline_end_to_end(
        self,
        samples: list[dict[str, Any]],
        iou_threshold: float = 0.5,
    ) -> dict[str, Any]:
        """
        Evaluate complete detection-to-classification pipeline on scene images.

        Separately measures:
        1. Vehicle detection performance (IoU against ground truth box).
        2. Classification on ground-truth crops.
        3. Classification on YOLO-generated crops.
        4. Complete end-to-end detection+classification performance.
        """
        total = 0
        gt_class_correct = 0
        yolo_detected = 0
        yolo_class_correct = 0
        e2e_correct = 0
        ious: list[float] = []

        for s in samples:
            img = s["image"]
            true_cid = int(s["true_class_id"])
            gt_bbox = s.get("bbox")  # (x1, y1, x2, y2)

            if isinstance(img, (str, Path)):
                img_p = Path(img) if Path(img).is_file() else self.root / img
                if not img_p.is_file():
                    continue
                pil_img = Image.open(img_p).convert("RGB")
                rgb_arr = np.array(pil_img)
            elif isinstance(img, Image.Image):
                rgb_arr = np.array(img.convert("RGB"))
            else:
                rgb_arr = img

            total += 1
            h, w = rgb_arr.shape[:2]

            # 1. Ground truth crop classification
            if gt_bbox:
                gx1, gy1, gx2, gy2 = gt_bbox
                gt_crop = rgb_arr[max(0, gy1):min(h, gy2), max(0, gx1):min(w, gx2)]
                if gt_crop.size > 0:
                    gt_pred = self.pipeline.dl_adapter.predict_crop(gt_crop)
                    if gt_pred.get("class_id") == true_cid:
                        gt_class_correct += 1

            # 2. End-to-end YOLO detection + hybrid recognition
            res = self.pipeline.process_image(rgb_arr)
            vehicles = res.get("vehicles", [])

            if vehicles:
                yolo_detected += 1
                # Find detection with highest IoU with GT
                best_vehicle = None
                best_iou = 0.0

                for v in vehicles:
                    vb = v["bbox"]
                    pred_box = (vb["x1"], vb["y1"], vb["x2"], vb["y2"])
                    iou = compute_iou(gt_bbox, pred_box) if gt_bbox else 1.0
                    if iou > best_iou:
                        best_iou = iou
                        best_vehicle = v

                ious.append(best_iou)

                if best_vehicle:
                    pred_cid = best_vehicle["final_prediction"].get("class_id", -1)
                    if pred_cid == true_cid:
                        yolo_class_correct += 1
                        if best_iou >= iou_threshold:
                            e2e_correct += 1

        return {
            "samples_evaluated": total,
            "detection_rate": round(yolo_detected / max(1, total), 4),
            "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
            "gt_crop_accuracy": round(gt_class_correct / max(1, total), 4),
            "yolo_crop_accuracy": round(yolo_class_correct / max(1, total), 4),
            "end_to_end_accuracy": round(e2e_correct / max(1, total), 4),
            "delta_gt_vs_yolo": round(
                (gt_class_correct - yolo_class_correct) / max(1, total), 4
            ),
        }
