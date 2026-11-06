"""
Comprehensive Evaluation Pipeline for Vehicle Classification.
Computes Top-1/Top-5 accuracy, macro/weighted F1, attribute-level metrics (Make, Model, Year),
identifies failure cases and top confused pairs, and outputs JSON/Markdown reports.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.classification.config import ClassificationConfig
from src.classification.metrics import (
    ClassMappingDecoder,
    calculate_classification_metrics,
)
from src.classification.models.backbones import build_classification_model
from src.classification.transforms import get_validation_transforms
from src.data.classification_dataset import StanfordCarsClassificationDataset
from src.utils.config import find_project_root
from src.utils.device import get_device

logger = logging.getLogger(__name__)


class ClassifierEvaluator:
    """
    Evaluator for fine-grained vehicle classifiers.
    Evaluates checkpoints against designated test or validation splits.
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

        # Resolve device
        if device == "auto":
            self.device = get_device("auto")
        else:
            self.device = torch.device(device)

        self.decoder = ClassMappingDecoder(class_mapping_path)

        # Load model and config from checkpoint or direct parameters
        if checkpoint_path is not None:
            self.checkpoint_path = (
                self.root / checkpoint_path if not Path(checkpoint_path).is_file() else Path(checkpoint_path)
            )
            self.model, self.config = self._load_from_checkpoint(self.checkpoint_path)
        elif model is not None:
            self.model = model.to(self.device)
            self.config = config or ClassificationConfig()
            self.checkpoint_path = None
        else:
            raise ValueError("Either checkpoint_path or model must be provided.")

        self.model.eval()

    def _load_from_checkpoint(
        self, path: Path
    ) -> tuple[nn.Module, ClassificationConfig]:
        """Load model architecture, weights, and configuration from checkpoint file."""
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        logger.info("Loading evaluation model checkpoint from %s on %s...", path, self.device)
        checkpoint = torch.load(path, map_location=self.device)

        cfg_dict = checkpoint.get("config", {})
        config = ClassificationConfig.from_dict(cfg_dict) if cfg_dict else ClassificationConfig()

        arch = checkpoint.get("architecture", config.architecture)
        num_classes = checkpoint.get("num_classes", config.num_classes)

        model = build_classification_model(
            architecture=arch,
            num_classes=num_classes,
            pretrained=False,
            dropout=config.dropout,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        return model, config

    def evaluate(
        self,
        data_source: str | Path | DataLoader | None = None,
        batch_size: int = 32,
        num_workers: int = 2,
        output_json: str | Path | None = None,
        output_markdown: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        Run full evaluation on the dataset and produce structured metrics.

        Args:
            data_source: Split CSV path or DataLoader (defaults to val_split or test_split).
            batch_size: Batch size for DataLoader if data_source is a path.
            num_workers: DataLoader worker count.
            output_json: Optional destination to write JSON metrics.
            output_markdown: Optional destination to write Markdown summary.

        Returns:
            Dictionary containing metrics, attributes, diagnostics, and sample errors.
        """
        # Resolve DataLoader
        if isinstance(data_source, DataLoader):
            loader = data_source
        else:
            split_rel = str(data_source) if data_source else self.config.val_split
            split_p = self.root / split_rel if not Path(split_rel).is_file() else Path(split_rel)
            if not split_p.is_file():
                raise FileNotFoundError(f"Evaluation dataset split not found: {split_p}")

            transform = get_validation_transforms(
                image_size=self.config.image_size,
                mean=self.config.mean,
                std=self.config.std,
            )
            dataset = StanfordCarsClassificationDataset(
                source=split_p,
                project_root=self.root,
                transform=transform,
                crop_bbox=self.config.crop_bbox,
                image_size=self.config.image_size,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(self.device.type == "cuda"),
            )

        self.last_loader = loader

        logger.info("Evaluating classifier on %d samples...", len(loader.dataset))

        all_preds = []
        all_targets = []
        all_probs = []
        sample_records = []
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    targets = batch["class_id"].to(self.device)
                    paths = batch.get("image_path", [""] * len(targets))
                else:
                    images, targets = batch[0].to(self.device), batch[1].to(self.device)
                    paths = [""] * len(targets)

                valid_mask = targets >= 0
                if not valid_mask.any():
                    continue

                images = images[valid_mask]
                targets = targets[valid_mask]
                sub_paths = [paths[i] for i, v in enumerate(valid_mask.cpu().tolist()) if v]

                logits = self.model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                targs = targets.cpu().numpy()

                all_probs.append(probs)
                all_preds.extend(preds.tolist())
                all_targets.extend(targs.tolist())

                for p, t, pr, path in zip(preds, targs, probs, sub_paths, strict=False):
                    sample_records.append({
                        "image_path": path,
                        "true_class_id": int(t),
                        "pred_class_id": int(p),
                        "confidence": float(np.max(pr)),
                        "is_correct": bool(p == t),
                    })

        total_latency_s = time.perf_counter() - t0
        avg_latency_ms = (total_latency_s / max(1, len(all_targets))) * 1000.0

        if not all_targets:
            logger.warning("No valid labeled samples evaluated.")
            return {"total_samples": 0}

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_probs = np.vstack(all_probs)

        # Calculate metrics suite
        metrics = calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_probs=y_probs,
            decoder=self.decoder,
            top_k=5,
        )

        metrics["architecture"] = self.config.architecture
        metrics["device"] = str(self.device)
        metrics["inference_latency_ms_per_image"] = round(avg_latency_ms, 2)
        metrics["checkpoint_path"] = str(self.checkpoint_path) if self.checkpoint_path else "in_memory"

        # Separate correct and incorrect samples for diagnostic inspection
        correct_samples = [s for s in sample_records if s["is_correct"]]
        incorrect_samples = [s for s in sample_records if not s["is_correct"]]

        metrics["sample_diagnostics"] = {
            "correct_samples_count": len(correct_samples),
            "incorrect_samples_count": len(incorrect_samples),
            "sample_correct": correct_samples[:10],
            "sample_errors": incorrect_samples[:10],
        }

        # Save Confusion Matrix CSV and JSON
        metrics_dir = (
            Path(output_json).parent if output_json else self.root / self.config.metrics_dir
        )
        if not metrics_dir.is_absolute():
            metrics_dir = self.root / metrics_dir
        metrics_dir.mkdir(parents=True, exist_ok=True)

        cm_raw = metrics.get("confusion_matrix_raw")
        if cm_raw:
            cm_arr = np.array(cm_raw)
            cnames = [self.decoder.class_names[i] for i in range(cm_arr.shape[0])]
            df_cm = pd.DataFrame(cm_arr, index=cnames, columns=cnames)
            cm_csv_path = metrics_dir / "confusion_matrix.csv"
            df_cm.to_csv(cm_csv_path, index=True)

            cm_json_path = metrics_dir / "confusion_matrix.json"
            with open(cm_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "matrix_shape": list(cm_arr.shape),
                    "classes": cnames,
                    "top_confused_pairs": metrics.get("top_confused_classes", []),
                }, f, indent=2)
            logger.info("Saved confusion matrix artifacts to %s and %s", cm_csv_path, cm_json_path)

            metrics.pop("confusion_matrix_raw", None)
            metrics["confusion_matrix_path"] = str(cm_csv_path.relative_to(self.root) if cm_csv_path.is_relative_to(self.root) else cm_csv_path).replace("\\", "/")

        # Save Sample Predictions
        pred_dir = self.root / "outputs" / "classification" / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_json_path = pred_dir / "sample_predictions.json"
        with open(pred_json_path, "w", encoding="utf-8") as f:
            json.dump(sample_records[:100], f, indent=2)
        logger.info("Saved sample predictions to %s", pred_json_path)

        # Generate Diagnostic Visualizations
        plots_dir = self.root / "outputs" / "classification" / "plots"
        self._generate_diagnostic_plots(metrics, plots_dir)

        # Save JSON if specified
        if output_json:
            out_p = self.root / output_json if not Path(output_json).is_absolute() else Path(output_json)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logger.info("Saved evaluation JSON metrics to %s", out_p)

        # Save Markdown report if specified
        if output_markdown:
            out_md = self.root / output_markdown if not Path(output_markdown).is_absolute() else Path(output_markdown)
            self._save_markdown_report(metrics, out_md)

        return metrics

    def _generate_diagnostic_plots(self, metrics: dict[str, Any], plots_dir: Path) -> None:
        """Render publication-quality diagnostic plots for classification."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plots_dir.mkdir(parents=True, exist_ok=True)

            # 1. Top Confused Classes Horizontal Bar Chart
            top_pairs = metrics.get("top_confused_classes", [])[:10]
            if top_pairs:
                fig, ax = plt.subplots(figsize=(10, 6))
                labels = [
                    f"{p['true_class_name'][:25]} -> {p['pred_class_name'][:25]}"
                    for p in reversed(top_pairs)
                ]
                counts = [p["count"] for p in reversed(top_pairs)]
                bars = ax.barh(labels, counts, color="#2b5c8f", edgecolor="black", alpha=0.85)
                ax.set_xlabel("Confusion Count", fontsize=11, fontweight="bold")
                ax.set_title("Top 10 Most Frequently Confused Vehicle Classes", fontsize=13, fontweight="bold", pad=12)
                ax.bar_label(bars, padding=3, fontsize=10)
                plt.tight_layout()
                fig.savefig(plots_dir / "confusion_matrix_top20.png", dpi=180)
                plt.close(fig)

            # 2. Prediction Confidence Distribution
            conf_stats = metrics.get("confidence_statistics", {})
            if conf_stats:
                fig, ax = plt.subplots(figsize=(8, 5))
                mean_c = conf_stats.get("mean_confidence", 0.5)
                median_c = conf_stats.get("median_confidence", 0.5)
                ax.axvline(mean_c, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_c:.3f}")
                ax.axvline(median_c, color="green", linestyle=":", linewidth=1.5, label=f"Median: {median_c:.3f}")
                ax.set_xlim(0.0, 1.0)
                ax.set_xlabel("Softmax Confidence", fontsize=11, fontweight="bold")
                ax.set_ylabel("Density / Frequency", fontsize=11, fontweight="bold")
                ax.set_title("Prediction Confidence Distribution", fontsize=13, fontweight="bold", pad=12)
                ax.legend(loc="upper left")
                ax.grid(True, alpha=0.3, linestyle="--")
                plt.tight_layout()
                fig.savefig(plots_dir / "confidence_distribution.png", dpi=180)
                plt.close(fig)

            # 3. Training History Curves (if available)
            history_file = self.root / "outputs" / "classification" / "metrics" / "training_history.json"
            if history_file.is_file():
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                if history:
                    epochs = [h.get("epoch", idx + 1) for idx, h in enumerate(history)]
                    train_loss = [h.get("train_loss", 0.0) for h in history]
                    val_loss = [h.get("val_loss", 0.0) for h in history]
                    train_acc = [h.get("train_top1", 0.0) * 100.0 for h in history]
                    val_acc = [h.get("val_top1", 0.0) * 100.0 for h in history]

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
                    # Loss plot
                    ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="#1f77b4", linewidth=2)
                    ax1.plot(epochs, val_loss, "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
                    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
                    ax1.set_ylabel("Cross Entropy Loss", fontsize=11, fontweight="bold")
                    ax1.set_title("Training & Validation Loss", fontsize=12, fontweight="bold")
                    ax1.grid(True, alpha=0.3, linestyle="--")
                    ax1.legend()

                    # Accuracy plot
                    ax2.plot(epochs, train_acc, "o-", label="Train Top-1", color="#2ca02c", linewidth=2)
                    ax2.plot(epochs, val_acc, "s--", label="Val Top-1", color="#d62728", linewidth=2)
                    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
                    ax2.set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
                    ax2.set_title("Training & Validation Accuracy", fontsize=12, fontweight="bold")
                    ax2.grid(True, alpha=0.3, linestyle="--")
                    ax2.legend()

                    plt.tight_layout()
                    fig.savefig(plots_dir / "training_history.png", dpi=180)
                    plt.close(fig)

            logger.info("Generated diagnostic plots in %s", plots_dir)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to render diagnostic plots: %s", e)

    def _save_markdown_report(self, metrics: dict[str, Any], path: Path) -> None:
        """Render a readable Markdown evaluation report."""
        path.parent.mkdir(parents=True, exist_ok=True)
        attr = metrics.get("attribute_metrics", {})
        conf = metrics.get("confidence_statistics", {})
        conf_pct = conf.get("percentiles", {})

        lines = [
            "# Fine-Grained Vehicle Classification — Evaluation Report",
            "",
            f"**Architecture:** {metrics.get('architecture')}  ",
            f"**Checkpoint:** `{metrics.get('checkpoint_path')}`  ",
            f"**Device:** {metrics.get('device')}  ",
            f"**Evaluated Samples:** {metrics.get('total_samples')}  ",
            f"**Average Latency:** {metrics.get('inference_latency_ms_per_image')} ms/image  ",
            "",
            "## 1. Classification Metrics (196 Stanford Cars Classes)",
            "",
            f"- **Top-1 Accuracy:** `{metrics.get('top1_accuracy') * 100:.2f}%`",
            f"- **Top-5 Accuracy:** `{metrics.get('top5_accuracy', 0.0) * 100:.2f}%`",
            f"- **Macro Precision:** `{metrics.get('macro_precision', 0.0):.4f}`",
            f"- **Weighted Precision:** `{metrics.get('weighted_precision', 0.0):.4f}`",
            f"- **Macro Recall:** `{metrics.get('macro_recall', 0.0):.4f}`",
            f"- **Weighted Recall:** `{metrics.get('weighted_recall', 0.0):.4f}`",
            f"- **Macro F1-Score:** `{metrics.get('macro_f1', 0.0):.4f}`",
            f"- **Weighted F1-Score:** `{metrics.get('weighted_f1', 0.0):.4f}`",
            "",
            "## 2. Decoded Attribute-Level Accuracies",
            "",
            "> Note: Make, Model, and Year are decoded from the primary 196-class prediction using the class mapping.",
            "",
            f"- **Make Accuracy:** `{attr.get('make_accuracy', 0.0) * 100:.2f}%` (49 makes)",
            f"- **Model Accuracy:** `{attr.get('model_accuracy', 0.0) * 100:.2f}%`",
            f"- **Year Accuracy:** `{attr.get('year_accuracy', 0.0) * 100:.2f}%`",
            "",
            "## 3. Confidence Statistics",
            "",
            f"- **Mean Confidence:** `{conf.get('mean_confidence', 0.0):.4f}`",
            f"- **Median Confidence:** `{conf.get('median_confidence', 0.0):.4f}`",
            f"- **Min / Max:** `{conf.get('min_confidence', 0.0):.4f}` / `{conf.get('max_confidence', 0.0):.4f}`",
            f"- **25th / 75th Percentile:** `{conf_pct.get('25', 0.0):.4f}` / `{conf_pct.get('75', 0.0):.4f}`",
            "",
            "## 4. Top Confused Vehicle Classes",
            "",
            "| True Vehicle Class | Predicted Vehicle Class | Count |",
            "| :--- | :--- | :--- |",
        ]

        for pair in metrics.get("top_confused_classes", [])[:10]:
            lines.append(f"| {pair['true_class_name']} | {pair['pred_class_name']} | {pair['count']} |")

        lines.extend([
            "",
            "## 5. Diagnostic Artifacts & Visualizations",
            "",
            "- **Full Confusion Matrix:** `outputs/classification/metrics/confusion_matrix.csv`",
            "- **Top Confused Plot:** `outputs/classification/plots/confusion_matrix_top20.png`",
            "- **Confidence Histogram:** `outputs/classification/plots/confidence_distribution.png`",
            "- **Training Curves:** `outputs/classification/plots/training_history.png`",
            "- **Sample Predictions:** `outputs/classification/predictions/sample_predictions.json`",
            "",
        ])
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Saved evaluation Markdown report to %s", path)
