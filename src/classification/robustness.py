"""
Robustness and Failure Analysis Suite for Vehicle Classification.
Evaluates classifier degradation under environmental stress (blur, resolution, lighting, occlusion),
extracts and categorizes concrete failure examples with root-cause explanations.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from src.classification.config import ClassificationConfig
from src.classification.metrics import ClassMappingDecoder
from src.classification.models.backbones import build_classification_model
from src.classification.transforms import (
    get_robustness_transform,
    get_validation_transforms,
)
from src.data.classification_dataset import StanfordCarsClassificationDataset
from src.utils.config import find_project_root
from src.utils.device import get_device

logger = logging.getLogger(__name__)

FAILURE_CATEGORIES = {
    "visually_similar_models": "Visually similar vehicle models with shared chassis/styling",
    "different_model_years": "Different model years with minor mid-cycle refresh changes",
    "poor_image_quality": "Degraded resolution, motion blur, or severe sensor noise",
    "occlusion_partial_view": "Partial vehicle occlusion or obscured distinctive badges/grilles",
    "lighting_extreme": "Extreme contrast or extreme specular highlights/shadows",
}


class RobustnessAnalyzer:
    """
    Evaluates fine-grained vehicle classifier stability across multiple image degradations.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        model: nn.Module | None = None,
        config: ClassificationConfig | None = None,
        device: str | torch.device = "auto",
        class_mapping_path: str | Path | None = None,
    ) -> None:
        self.root = find_project_root()
        if device == "auto":
            self.device = get_device("auto")
        else:
            self.device = torch.device(device)

        self.decoder = ClassMappingDecoder(class_mapping_path)

        if checkpoint_path is not None:
            p = self.root / checkpoint_path if not Path(checkpoint_path).is_file() else Path(checkpoint_path)
            checkpoint = torch.load(p, map_location=self.device)
            cfg_dict = checkpoint.get("config", {})
            self.config = ClassificationConfig.from_dict(cfg_dict) if cfg_dict else ClassificationConfig()
            arch = checkpoint.get("architecture", self.config.architecture)
            self.model = build_classification_model(
                architecture=arch,
                num_classes=checkpoint.get("num_classes", self.config.num_classes),
                pretrained=False,
                dropout=self.config.dropout,
            )
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(self.device)
        elif model is not None:
            self.model = model.to(self.device)
            self.config = config or ClassificationConfig()
        else:
            raise ValueError("Either checkpoint_path or model must be provided.")

        self.model.eval()

    def evaluate_condition(
        self,
        split_path: str | Path,
        perturbation: str,
        batch_size: int = 32,
        max_samples: int = 500,
        **kwargs: Any,
    ) -> tuple[float, float, list[dict[str, Any]]]:
        """
        Evaluate model under a specific perturbation condition.

        Returns:
            (top1_accuracy, top5_accuracy, sample_predictions)
        """
        p_path = self.root / split_path if not Path(split_path).is_file() else Path(split_path)
        tf = get_robustness_transform(
            perturbation=perturbation,
            image_size=self.config.image_size,
            mean=self.config.mean,
            std=self.config.std,
            **kwargs,
        )

        dataset = StanfordCarsClassificationDataset(
            source=p_path,
            project_root=self.root,
            transform=tf,
            crop_bbox=self.config.crop_bbox,
            image_size=self.config.image_size,
        )

        # Slice subset if max_samples specified
        if max_samples and max_samples < len(dataset):
            from torch.utils.data import Subset
            dataset = Subset(dataset, list(range(max_samples)))

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        correct_top1 = 0
        correct_top5 = 0
        total = 0
        predictions = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                targets = batch["class_id"].to(self.device)
                paths = batch.get("image_path", [""] * len(targets))

                valid = targets >= 0
                if not valid.any():
                    continue

                images = images[valid]
                targets = targets[valid]
                sub_paths = [paths[i] for i, v in enumerate(valid.cpu().tolist()) if v]

                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                batch_sz = targets.size(0)
                total += batch_sz
                correct_top1 += (preds == targets).sum().item()

                k = min(5, logits.size(1))
                top5_pred = logits.topk(k, dim=1)[1]
                correct_top5 += top5_pred.eq(targets.view(-1, 1).expand_as(top5_pred)).sum().item()

                probs_np = probs.cpu().numpy()
                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()

                for pred_id, target_id, pr, path in zip(preds_np, targets_np, probs_np, sub_paths, strict=False):
                    predictions.append({
                        "image_path": path,
                        "true_class_id": int(target_id),
                        "pred_class_id": int(pred_id),
                        "confidence": float(np.max(pr)),
                        "is_correct": bool(pred_id == target_id),
                    })

        top1 = correct_top1 / max(1, total)
        top5 = correct_top5 / max(1, total)
        return top1, top5, predictions

    def run_full_robustness_benchmark(
        self,
        split_path: str | Path | None = None,
        max_samples_per_condition: int = 400,
        output_dir: str | Path = "outputs/classification",
    ) -> dict[str, Any]:
        """
        Execute full robustness benchmark across all canonical environmental stress conditions.
        """
        split = split_path or self.config.val_split
        out_root = self.root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)
        metrics_dir = out_root / "metrics"
        failures_dir = out_root / "failure_cases"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        failures_dir.mkdir(parents=True, exist_ok=True)

        # Baseline clean performance
        logger.info("Evaluating baseline clean performance on %s...", split)
        base_clean_tf = get_validation_transforms(
            image_size=self.config.image_size,
            mean=self.config.mean,
            std=self.config.std,
        )
        base_ds = StanfordCarsClassificationDataset(
            source=self.root / split,
            project_root=self.root,
            transform=base_clean_tf,
            crop_bbox=self.config.crop_bbox,
            image_size=self.config.image_size,
        )
        if max_samples_per_condition and max_samples_per_condition < len(base_ds):
            from torch.utils.data import Subset
            base_ds = Subset(base_ds, list(range(max_samples_per_condition)))

        base_loader = DataLoader(base_ds, batch_size=32, shuffle=False)
        base_top1_sum = 0
        base_top5_sum = 0
        base_total = 0
        clean_samples = []

        with torch.no_grad():
            for batch in base_loader:
                imgs = batch["image"].to(self.device)
                tgts = batch["class_id"].to(self.device)
                paths = batch.get("image_path", [""] * len(tgts))
                valid = tgts >= 0
                if not valid.any():
                    continue
                imgs, tgts = imgs[valid], tgts[valid]
                sub_paths = [paths[i] for i, v in enumerate(valid.cpu().tolist()) if v]

                l = self.model(imgs)
                p = torch.softmax(l, dim=1)
                preds = l.argmax(dim=1)
                base_total += tgts.size(0)
                base_top1_sum += (preds == tgts).sum().item()
                k = min(5, l.size(1))
                t5 = l.topk(k, dim=1)[1]
                base_top5_sum += t5.eq(tgts.view(-1, 1).expand_as(t5)).sum().item()

                probs_np = p.cpu().numpy()
                preds_np = preds.cpu().numpy()
                tgts_np = tgts.cpu().numpy()
                for prd, tgt, pr, path in zip(preds_np, tgts_np, probs_np, sub_paths, strict=False):
                    clean_samples.append({
                        "image_path": path,
                        "true_class_id": int(tgt),
                        "pred_class_id": int(prd),
                        "confidence": float(np.max(pr)),
                        "is_correct": bool(prd == tgt),
                    })

        baseline_top1 = base_top1_sum / max(1, base_total)
        baseline_top5 = base_top5_sum / max(1, base_total)

        conditions = [
            ("reduced_resolution", {"scale": 0.4}),
            ("mild_blur", {"radius": 2.5}),
            ("brightness_high", {"factor": 1.35}),
            ("brightness_low", {"factor": 0.65}),
            ("contrast_high", {"factor": 1.4}),
            ("contrast_low", {"factor": 0.6}),
            ("partial_occlusion", {"scale": 0.25}),
        ]

        condition_results = {
            "baseline_clean": {
                "top1_accuracy": round(baseline_top1, 4),
                "top5_accuracy": round(baseline_top5, 4),
                "accuracy_drop": 0.0,
            }
        }

        all_condition_failures = []

        for cond_name, kwargs in conditions:
            logger.info("Evaluating robustness condition: %s...", cond_name)
            top1, top5, preds = self.evaluate_condition(
                split_path=split,
                perturbation=cond_name,
                max_samples=max_samples_per_condition,
                **kwargs,
            )
            drop = baseline_top1 - top1
            condition_results[cond_name] = {
                "top1_accuracy": round(top1, 4),
                "top5_accuracy": round(top5, 4),
                "accuracy_drop": round(drop, 4),
                "relative_drop_pct": round((drop / max(0.001, baseline_top1)) * 100.0, 2),
            }

            for item in preds:
                if not item["is_correct"]:
                    item["condition"] = cond_name
                    all_condition_failures.append(item)

        # Extract concrete failure cases with root causes
        failure_examples = self._curate_failure_cases(
            clean_failures=[s for s in clean_samples if not s["is_correct"]],
            stress_failures=all_condition_failures,
        )

        # Save failure cases JSON
        failures_json = failures_dir / "failure_cases.json"
        with open(failures_json, "w", encoding="utf-8") as f:
            json.dump(failure_examples, f, indent=2)

        # Save robustness metrics
        report = {
            "evaluated_samples_per_condition": min(max_samples_per_condition, base_total),
            "baseline_top1": round(baseline_top1, 4),
            "baseline_top5": round(baseline_top5, 4),
            "conditions": condition_results,
            "failure_cases_summary": {
                "total_failure_examples_documented": len(failure_examples),
                "cases_file": str(failures_json.relative_to(self.root) if failures_json.is_relative_to(self.root) else failures_json).replace("\\", "/"),
            },
        }

        report_json = metrics_dir / "robustness_metrics.json"
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Robustness benchmark complete. Saved to %s and %s", report_json, failures_json)
        return report

    def _curate_failure_cases(
        self,
        clean_failures: list[dict[str, Any]],
        stress_failures: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Categorize concrete failure cases across the 5 canonical failure modes specified by the design.
        Ensures diverse representation of visually similar models, differing model years,
        poor quality, occlusion, and lighting extremes.
        """
        by_category: dict[str, list[dict[str, Any]]] = {
            cat: [] for cat in FAILURE_CATEGORIES
        }

        # Interleave stress failures and clean failures to maximize perturbation diversity
        pool = stress_failures + clean_failures
        seen_pairs = set()

        for sample in pool:
            t_id = sample["true_class_id"]
            p_id = sample["pred_class_id"]
            cond = sample.get("condition", "baseline_clean")
            key = (t_id, p_id, cond)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            t_info = self.decoder.get_info(t_id)
            p_info = self.decoder.get_info(p_id)

            category = "poor_image_quality"
            reason = "Degraded image resolution or blurring obscured key high-frequency features."

            if t_info["make"] == p_info["make"] and t_info["model"] == p_info["model"]:
                category = "different_model_years"
                reason = (
                    f"Same vehicle model ({t_info['make']} {t_info['model']}) differing only in "
                    f"manufacturing year ({t_info['year']} vs {p_info['year']}) with nearly identical body design."
                )
            elif t_info["make"] == p_info["make"]:
                category = "visually_similar_models"
                reason = (
                    f"Shared corporate styling and badge aesthetic between {t_info['make']} {t_info['model']} "
                    f"and {p_info['make']} {p_info['model']}."
                )
            elif cond == "partial_occlusion":
                category = "occlusion_partial_view"
                reason = "Synthetic occlusion masked the distinguishing vehicle badge and front grille silhouette."
            elif cond in {"brightness_high", "brightness_low", "contrast_high", "contrast_low"}:
                category = "lighting_extreme"
                reason = "Extreme lighting perturbation degraded structural contour clarity and texture details."
            elif cond in {"reduced_resolution", "mild_blur"}:
                category = "poor_image_quality"
                reason = f"Environmental degradation ({cond}) destroyed fine-grained badges and distinctive lighting accents."

            entry = {
                "image_path": sample["image_path"],
                "condition": cond,
                "ground_truth": {
                    "class_id": t_id,
                    "class_name": t_info["class_name"],
                    "make": t_info["make"],
                    "model": t_info["model"],
                    "year": t_info["year"],
                },
                "predicted": {
                    "class_id": p_id,
                    "class_name": p_info["class_name"],
                    "make": p_info["make"],
                    "model": p_info["model"],
                    "year": p_info["year"],
                },
                "confidence": round(sample["confidence"], 4),
                "failure_category": category,
                "category_description": FAILURE_CATEGORIES.get(category, ""),
                "likely_reason": reason,
            }
            by_category[category].append(entry)

        # Curate up to 2-3 items from each represented category to ensure breadth
        curated: list[dict[str, Any]] = []
        for cat in FAILURE_CATEGORIES:
            entries = by_category.get(cat, [])
            curated.extend(entries[:2])

        # If total is less than 5, backfill from remaining
        if len(curated) < 5:
            for cat, entries in by_category.items():
                for e in entries[2:]:
                    if e not in curated:
                        curated.append(e)
                    if len(curated) >= 10:
                        break
                if len(curated) >= 10:
                    break

        return curated
