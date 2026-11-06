"""
Vehicle Detection Evaluation and Benchmarking Suite.
Computes mAP50, mAP50-95, Precision, Recall, F1, IoU distributions,
latency profiling, GPU memory metrics, and perturbation robustness testing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

from src.detection.detector import VehicleDetector
from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


def compute_iou(box1: tuple[float, float, float, float] | list[float], box2: tuple[float, float, float, float] | list[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, float(x2 - x1))
    inter_h = max(0.0, float(y2 - y1))
    inter_area = inter_w * inter_h

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class DetectionEvaluator:
    """
    Evaluator for vehicle detection models on Stanford Cars.
    Provides standard COCO mAP validation via Ultralytics and
    granular ground-truth matching, latency, and robustness diagnostics.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        data_yaml: str | Path = "data/processed/detection/dataset.yaml",
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.root = find_project_root()
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Initialize detector for direct inference tests (auto-resolves best.pt or fallback)
        self.detector = VehicleDetector(
            model_path=model_path,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            device=self.device,
        )
        self.model_path = self.detector.model_path
        self.data_yaml = Path(data_yaml) if Path(data_yaml).is_absolute() else (self.root / data_yaml)

    def get_model_specs(self) -> dict[str, Any]:
        """Compute parameter count, file size, and architecture details."""
        pt_file = self.detector.model_path
        size_mb = round(pt_file.stat().st_size / (1024 * 1024), 2) if pt_file.is_file() else 0.0

        param_count = 0
        try:
            param_count = sum(p.numel() for p in self.detector.model.model.parameters())
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not count parameters via model.model: %s", e)

        return {
            "model_path": str(pt_file),
            "size_mb": size_mb,
            "parameter_count": param_count,
            "device": self.detector.device_str,
            "num_classes": self.detector.num_classes,
        }

    def evaluate_ultralytics(
        self,
        split: str = "val",
        imgsz: int = 640,
        batch: int = 16,
        workers: int = 0,
    ) -> dict[str, Any]:
        """
        Run official Ultralytics dataset validation to calculate mAP50, mAP50-95,
        precision, and recall.
        """
        if not self.data_yaml.is_file():
            raise FileNotFoundError(f"dataset.yaml not found at {self.data_yaml}")

        dev = "0" if "cuda" in self.detector.device_str else "cpu"
        yolo_model = YOLO(str(self.detector.model_path))

        # Reset peak GPU memory stats if using CUDA
        is_cuda = torch.cuda.is_available() and ("cuda" in self.detector.device_str or dev.isdigit())
        if is_cuda:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception as e:  # noqa: BLE001
                logger.debug("Could not reset CUDA peak memory stats: %s", e)

        logger.info("Running Ultralytics validation on split='%s' with %s...", split, self.detector.model_path)
        metrics = yolo_model.val(
            data=str(self.data_yaml),
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=dev,
            workers=workers,
            conf=0.001,  # Standard evaluation confidence threshold
            iou=0.6,
            verbose=False,
            plots=False,
        )

        mp = float(metrics.box.mp)
        mr = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map95 = float(metrics.box.map)
        f1 = float(2 * mp * mr / (mp + mr + 1e-9))

        speed = metrics.speed
        inference_latency_ms = round(float(speed.get("inference", 0.0)), 2)

        peak_vram_gb = 0.0
        if is_cuda:
            try:
                peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 3)
            except Exception as e:  # noqa: BLE001
                logger.debug("Could not read peak memory: %s", e)

        results = {
            "split": split,
            "precision": round(mp, 4),
            "recall": round(mr, 4),
            "f1_score": round(f1, 4),
            "mAP50": round(map50, 4),
            "mAP50-95": round(map95, 4),
            "inference_latency_ms": inference_latency_ms,
            "preprocess_latency_ms": round(float(speed.get("preprocess", 0.0)), 2),
            "postprocess_latency_ms": round(float(speed.get("postprocess", 0.0)), 2),
            "peak_vram_gb": peak_vram_gb,
            "model_specs": self.get_model_specs(),
        }
        return results

    def evaluate_ground_truth_matching(
        self,
        records: list[dict[str, Any]] | pd.DataFrame,
        iou_eval_threshold: float = 0.5,
        max_samples: int | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate detections directly against ground-truth annotations from vehicle manifest.
        Calculates empirical IoU distribution, TP, FP, FN, precision, recall, and per-image latency.

        Args:
            records: List of manifest dicts or DataFrame containing image_path and bbox coords.
            iou_eval_threshold: Minimum IoU to consider detection a True Positive.
            max_samples: Optional limit on number of samples for rapid evaluation.
        """
        if isinstance(records, pd.DataFrame):
            df = records.copy()
        else:
            df = pd.DataFrame(records)

        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        x1_col = "bbox_x1" if "bbox_x1" in df.columns else "bbox_xmin"
        y1_col = "bbox_y1" if "bbox_y1" in df.columns else "bbox_ymin"
        x2_col = "bbox_x2" if "bbox_x2" in df.columns else "bbox_xmax"
        y2_col = "bbox_y2" if "bbox_y2" in df.columns else "bbox_ymax"

        tp = 0
        fp = 0
        fn = 0
        ious: list[float] = []
        latencies: list[float] = []

        for _, row in df.iterrows():
            img_rel = str(row["image_path"]).replace("\\", "/")
            full_img_p = self.root / img_rel
            if not full_img_p.is_file():
                continue

            gt_box = (
                float(row[x1_col]),
                float(row[y1_col]),
                float(row[x2_col]),
                float(row[y2_col]),
            )

            # Predict
            pred = self.detector.detect(full_img_p, extract_rois=False)
            latencies.append(pred["inference_time_ms"])

            dets = pred["detections"]
            if not dets:
                fn += 1
                ious.append(0.0)
                continue

            # Compare each detection to GT box
            best_iou = 0.0
            for d in dets:
                b = d["bbox"]
                p_box = (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]))
                iou = compute_iou(gt_box, p_box)
                best_iou = max(best_iou, iou)

            ious.append(best_iou)
            if best_iou >= iou_eval_threshold:
                tp += 1
                # Any additional detections count as FP
                fp += max(0, len(dets) - 1)
            else:
                fn += 1
                fp += len(dets)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        mean_iou = float(np.mean(ious)) if ious else 0.0
        mean_lat = float(np.mean(latencies[1:])) if len(latencies) > 1 else (latencies[0] if latencies else 0.0)

        return {
            "total_evaluated": len(df),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "mean_iou": round(mean_iou, 4),
            "mean_inference_latency_ms": round(mean_lat, 2),
            "iou_threshold": iou_eval_threshold,
        }

    def evaluate_robustness(
        self,
        sample_images: list[Path | str],
    ) -> dict[str, Any]:
        """
        Evaluate detector under diverse challenging environmental conditions:
        1. Baseline (standard RGB)
        2. Low-light condition (gamma = 2.5 / brightness = 0.4)
        3. Mild motion / optical blur (Gaussian blur 7x7)
        4. Partial occlusion (center blackout 25%)
        5. Multiple vehicles (composite dual-vehicle scene)
        6. Cropped vehicles (side boundary cut-off)
        7. Small vehicles (distant vehicle placed on neutral canvas)
        8. Multi-resolution scaling (1280x720 down to 160x120)
        """
        results: dict[str, list[dict[str, Any]]] = {}

        loaded_images: list[np.ndarray] = []
        for img_input in sample_images[:10]:
            p = self.root / img_input if not Path(img_input).is_file() else Path(img_input)
            if not p.is_file():
                continue
            with Image.open(p) as pil_img:
                loaded_images.append(np.array(pil_img.convert("RGB"), dtype=np.uint8))

        if not loaded_images:
            return {}

        def _record_metrics(condition: str, det_result: dict[str, Any]) -> None:
            dets = det_result.get("detections", [])
            confs = [d["confidence"] for d in dets]
            results.setdefault(condition, []).append({
                "count": len(dets),
                "has_detection": len(dets) > 0,
                "mean_conf": float(np.mean(confs)) if confs else 0.0,
            })

        for idx, raw_np in enumerate(loaded_images):
            h, w = raw_np.shape[:2]

            # 1. Baseline
            _record_metrics("baseline", self.detector.detect(raw_np, extract_rois=False))

            # 2. Low Light (scale intensity by 0.4)
            low_light = (raw_np.astype(np.float32) * 0.4).astype(np.uint8)
            _record_metrics("low_light", self.detector.detect(low_light, extract_rois=False))

            # 3. Mild Blur (Gaussian 7x7)
            blurred = cv2.GaussianBlur(raw_np, (7, 7), 1.5)
            _record_metrics("mild_blur", self.detector.detect(blurred, extract_rois=False))

            # 4. Partial Occlusion (center blackout 25%)
            occluded = raw_np.copy()
            cy, cx = h // 2, w // 2
            bh, bw = h // 4, w // 4
            occluded[cy - bh // 2 : cy + bh // 2, cx - bw // 2 : cx + bw // 2] = 0
            _record_metrics("partial_occlusion", self.detector.detect(occluded, extract_rois=False))

            # 5. Multiple Vehicles (composite dual vehicle frame)
            next_img = loaded_images[(idx + 1) % len(loaded_images)]
            h_tgt = min(h, next_img.shape[0])
            r1 = cv2.resize(raw_np, (int(w * (h_tgt / h)), h_tgt))
            r2 = cv2.resize(next_img, (int(next_img.shape[1] * (h_tgt / next_img.shape[0])), h_tgt))
            dual_comp = np.hstack([r1, r2])
            _record_metrics("multiple_vehicles", self.detector.detect(dual_comp, extract_rois=False))

            # 6. Cropped Vehicle (side boundary clipping - 65% width)
            cropped_side = raw_np[:, : int(w * 0.65)]
            _record_metrics("cropped_vehicle", self.detector.detect(cropped_side, extract_rois=False))

            # 7. Small Vehicle (vehicle resized to 25% and placed on neutral canvas)
            small_w, small_h = max(32, w // 4), max(24, h // 4)
            mini_car = cv2.resize(raw_np, (small_w, small_h), interpolation=cv2.INTER_AREA)
            canvas = np.full((h, w, 3), 128, dtype=np.uint8)
            canvas[h // 4 : h // 4 + small_h, w // 4 : w // 4 + small_w] = mini_car
            _record_metrics("small_vehicle", self.detector.detect(canvas, extract_rois=False))

            # 8. Multi-Resolution: Low Resolution (320x240)
            downscaled = cv2.resize(raw_np, (320, 240), interpolation=cv2.INTER_AREA)
            _record_metrics("low_resolution_320x240", self.detector.detect(downscaled, extract_rois=False))

            # 9. Multi-Resolution: High Resolution (1280x720)
            upscaled = cv2.resize(raw_np, (1280, 720), interpolation=cv2.INTER_LINEAR)
            _record_metrics("high_resolution_1280x720", self.detector.detect(upscaled, extract_rois=False))

        summary: dict[str, Any] = {}
        for condition, records in results.items():
            counts = [r["count"] for r in records]
            has_dets = [1 if r["has_detection"] else 0 for r in records]
            confs = [r["mean_conf"] for r in records if r["has_detection"]]
            summary[condition] = {
                "detected_count": int(np.sum(counts)),
                "detection_rate": round(float(np.mean(has_dets)), 4) if has_dets else 0.0,
                "mean_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
            }

        return summary

    def evaluate_optimization(
        self,
        sample_images: list[Path | str],
        batch_sizes: list[int] | None = None,
        resolutions: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate efficient inference settings:
        - FP32 vs FP16 half-precision latency and speedup.
        - Input resolutions (640 vs 480 vs 320).
        - Batch inference latency per image across batch sizes.
        - Model parameter count and checkpoint size.
        """
        if batch_sizes is None:
            batch_sizes = [1, 4, 8, 16]
        if resolutions is None:
            resolutions = [640, 480, 320]

        loaded_images: list[np.ndarray] = []
        for img_input in sample_images[:16]:
            p = self.root / img_input if not Path(img_input).is_file() else Path(img_input)
            if not p.is_file():
                continue
            with Image.open(p) as pil_img:
                loaded_images.append(np.array(pil_img.convert("RGB"), dtype=np.uint8))

        if not loaded_images:
            return {"error": "No images available for optimization benchmarking"}

        specs = self.get_model_specs()
        test_img = loaded_images[0]

        # 1. FP32 vs FP16 Single-Image Latency
        # Warmup
        for _ in range(3):
            self.detector.detect(test_img, extract_rois=False)

        fp32_times = []
        for _ in range(5):
            t0 = cv2.getTickCount()
            self.detector.detect(test_img, extract_rois=False)
            t1 = cv2.getTickCount()
            fp32_times.append((t1 - t0) / cv2.getTickFrequency() * 1000.0)
        mean_fp32_ms = round(float(np.median(fp32_times)), 2)

        # FP16 test if CUDA
        is_cuda = "cuda" in self.detector.device_str
        mean_fp16_ms = mean_fp32_ms
        if is_cuda:
            fp16_detector = VehicleDetector(
                model_path=self.detector.model_path,
                device=self.detector.device_str,
                half=True,
            )
            # Warmup
            for _ in range(3):
                fp16_detector.detect(test_img, extract_rois=False)
            fp16_times = []
            for _ in range(5):
                t0 = cv2.getTickCount()
                fp16_detector.detect(test_img, extract_rois=False)
                t1 = cv2.getTickCount()
                fp16_times.append((t1 - t0) / cv2.getTickFrequency() * 1000.0)
            mean_fp16_ms = round(float(np.median(fp16_times)), 2)

        speedup_pct = round(((mean_fp32_ms - mean_fp16_ms) / mean_fp32_ms) * 100.0, 1) if mean_fp32_ms > 0 else 0.0

        # 2. Input Resolution Impact
        resolution_bench: dict[str, float] = {}
        for r_size in resolutions:
            scaled = cv2.resize(test_img, (r_size, r_size))
            times = []
            for _ in range(5):
                t0 = cv2.getTickCount()
                self.detector.detect(scaled, extract_rois=False)
                t1 = cv2.getTickCount()
                times.append((t1 - t0) / cv2.getTickFrequency() * 1000.0)
            resolution_bench[f"{r_size}x{r_size}"] = round(float(np.median(times)), 2)

        # 3. Batch Inference Throughput
        batch_bench: dict[str, Any] = {}
        for b_size in batch_sizes:
            sub_batch = (loaded_images * ((b_size // len(loaded_images)) + 1))[:b_size]
            # Warmup
            self.detector.detect_batch(sub_batch[:min(2, b_size)], extract_rois=False)
            t0 = cv2.getTickCount()
            self.detector.detect_batch(sub_batch, extract_rois=False)
            t1 = cv2.getTickCount()
            total_b_ms = (t1 - t0) / cv2.getTickFrequency() * 1000.0
            per_img_ms = round(total_b_ms / b_size, 2)
            batch_bench[f"batch_{b_size}"] = {
                "total_time_ms": round(total_b_ms, 2),
                "per_image_ms": per_img_ms,
                "fps": round(1000.0 / per_img_ms, 1) if per_img_ms > 0 else 0.0,
            }

        return {
            "model_size_mb": specs["size_mb"],
            "parameter_count": specs["parameter_count"],
            "device": specs["device"],
            "fp32_single_image_ms": mean_fp32_ms,
            "fp16_single_image_ms": mean_fp16_ms,
            "fp16_speedup_pct": speedup_pct,
            "resolution_latency_ms": resolution_bench,
            "batch_throughput": batch_bench,
        }
