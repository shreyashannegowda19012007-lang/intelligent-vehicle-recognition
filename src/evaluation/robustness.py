"""
Robustness Testing Suite for Fine-Grained Vehicle Recognition.
Evaluates recognition performance and accuracy degradation under visual perturbations:
- Gaussian Blur
- Low Brightness (dim illumination)
- High Brightness (glare / exposure)
- Reduced Resolution (downsampling)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.utils.config import find_project_root

logger = logging.getLogger(__name__)


def apply_gaussian_blur(image: np.ndarray, ksize: int = 7, sigma: float = 2.0) -> np.ndarray:
    """Apply Gaussian blur perturbation."""
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Adjust image brightness by multiplicative factor [0.1, 3.0]."""
    img_f = image.astype(np.float32) * factor
    return np.clip(img_f, 0, 255).astype(np.uint8)


def apply_reduced_resolution(image: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Downsample image to simulate low-resolution sensor, then scale back."""
    h, w = image.shape[:2]
    down_w = max(16, int(w * scale))
    down_h = max(16, int(h * scale))
    down = cv2.resize(image, (down_w, down_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(down, (w, h), interpolation=cv2.INTER_LINEAR)


PERTURBATIONS: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "blur": ("Gaussian Blur (k=7, sigma=2.0)", lambda img: apply_gaussian_blur(img, ksize=7, sigma=2.0)),
    "low_brightness": ("Low Brightness (factor=0.5)", lambda img: apply_brightness(img, factor=0.5)),
    "high_brightness": ("High Brightness (factor=1.5)", lambda img: apply_brightness(img, factor=1.5)),
    "reduced_resolution": ("Reduced Resolution (scale=0.5)", lambda img: apply_reduced_resolution(img, scale=0.5)),
}


def run_robustness_benchmark(
    predict_fn: Callable[[np.ndarray], dict[str, Any]],
    test_samples: Sequence[tuple[np.ndarray, int]],
    pipeline_name: str = "ResNet50",
) -> dict[str, Any]:
    """
    Run lightweight robustness experiment across standard visual perturbations.

    Args:
        predict_fn: Callable taking RGB numpy array and returning prediction dict with 'class_id'.
        test_samples: List of (rgb_image_array, true_class_id) tuples.
        pipeline_name: Identifying name of the pipeline being benchmarked.

    Returns:
        Structured dictionary of baseline and perturbed accuracy drops.
    """
    total = len(test_samples)
    if total == 0:
        return {"error": "No test samples provided for robustness benchmark."}

    # 1. Baseline Evaluation
    baseline_correct = 0
    for img, true_id in test_samples:
        try:
            res = predict_fn(img)
            pred_id = int(res.get("class_id", -1))
            if pred_id == true_id:
                baseline_correct += 1
        except Exception as e:
            logger.debug("Baseline inference failed: %s", e)

    baseline_acc = float(baseline_correct / total)
    results = {
        "pipeline": pipeline_name,
        "sample_count": total,
        "baseline_accuracy": round(baseline_acc, 4),
        "conditions": {},
    }

    # 2. Evaluate Each Perturbation Condition
    for cond_key, (cond_desc, transform_fn) in PERTURBATIONS.items():
        cond_correct = 0
        for img, true_id in test_samples:
            try:
                perturbed_img = transform_fn(img)
                res = predict_fn(perturbed_img)
                pred_id = int(res.get("class_id", -1))
                if pred_id == true_id:
                    cond_correct += 1
            except Exception as e:
                logger.debug("Perturbed inference failed for %s: %s", cond_key, e)

        cond_acc = float(cond_correct / total)
        abs_drop = round(baseline_acc - cond_acc, 4)
        rel_drop_pct = round((abs_drop / max(1e-6, baseline_acc)) * 100.0, 2) if baseline_acc > 0 else 0.0

        results["conditions"][cond_key] = {
            "description": cond_desc,
            "accuracy": round(cond_acc, 4),
            "absolute_accuracy_drop": abs_drop,
            "relative_drop_pct": rel_drop_pct,
        }

    return results


def robustness_results_to_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    """Convert robustness benchmark results into a flat pandas DataFrame."""
    rows = []
    pipe = results.get("pipeline", "Model")
    base_acc = results.get("baseline_accuracy", 0.0)

    rows.append({
        "pipeline": pipe,
        "condition": "Baseline (Clean)",
        "accuracy": base_acc,
        "accuracy_drop": 0.0,
        "relative_drop_pct": 0.0,
        "samples": results.get("sample_count", 0),
    })

    for cond_k, cond_data in results.get("conditions", {}).items():
        rows.append({
            "pipeline": pipe,
            "condition": cond_k,
            "accuracy": cond_data["accuracy"],
            "accuracy_drop": cond_data["absolute_accuracy_drop"],
            "relative_drop_pct": cond_data["relative_drop_pct"],
            "samples": results.get("sample_count", 0),
        })

    return pd.DataFrame(rows)

