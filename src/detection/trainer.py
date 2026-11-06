"""
Vehicle Detection Training Pipeline using Ultralytics YOLOv8.
Handles dataset configuration, transfer learning, layer freezing,
hyperparameter management, checkpoint tracking, and metric logging.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from src.utils.config import find_project_root
from src.utils.device import get_device

logger = logging.getLogger(__name__)


class DetectionTrainer:
    """
    Manager for YOLO vehicle detector training, checkpointing, and fine-tuning.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize trainer with configuration file and optional overrides.

        Args:
            config_path: Path to detection config YAML (default: configs/detection.yaml).
            kwargs: Command-line or runtime overrides for any config parameter.
        """
        self.root = find_project_root()
        self.config_path = self._resolve_config_path(config_path)
        self.config = self._load_config()

        # Apply runtime kwargs overrides
        for key, value in kwargs.items():
            if value is not None:
                self._apply_override(key, value)

        self._setup_paths()

    def _resolve_config_path(self, config_path: str | Path | None) -> Path:
        if config_path is None:
            return self.root / "configs" / "detection.yaml"
        p = Path(config_path)
        return p if p.is_file() else (self.root / p)

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path.is_file():
            logger.warning("Config not found at %s; using internal defaults", self.config_path)
            return self._default_config()

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg

    def _apply_override(self, key: str, value: Any) -> None:
        """Apply top-level or nested override."""
        for section in ("training", "model", "data", "augmentation", "inference"):
            if section in self.config and isinstance(self.config[section], dict) and key in self.config[section]:
                self.config[section][key] = value
                return
        # If not found in subsections, place in training
        if "training" not in self.config:
            self.config["training"] = {}
        self.config["training"][key] = value

    def _setup_paths(self) -> None:
        paths = self.config.get("paths", {})
        self.checkpoints_dir = (self.root / paths.get("checkpoints_dir", "models/detection")).resolve()
        self.metrics_dir = (self.root / paths.get("metrics_dir", "outputs/detection/metrics")).resolve()
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def _default_config(self) -> dict[str, Any]:
        return {
            "model": {
                "architecture": "yolov8n",
                "pretrained_checkpoint": "yolov8n.pt",
                "num_classes": 1,
            },
            "data": {
                "dataset_yaml": "data/processed/detection/dataset.yaml",
            },
            "training": {
                "epochs": 10,
                "batch_size": 16,
                "imgsz": 640,
                "device": "auto",
                "workers": 2,
                "optimizer": "AdamW",
                "lr0": 0.001,
                "lrf": 0.01,
                "weight_decay": 0.0005,
                "patience": 5,
                "project": "models/detection",
                "name": "yolov8n_vehicle",
                "exist_ok": True,
                "pretrained": True,
                "amp": True,
                "freeze": 0,
            },
            "augmentation": {
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 10.0,
                "translate": 0.1,
                "scale": 0.5,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic": 1.0,
                "mixup": 0.0,
            },
        }

    def train(self, resume: bool = False) -> dict[str, Any]:
        """
        Execute training run using configured settings.

        Args:
            resume: If True, attempts to resume training from the last checkpoint.

        Returns:
            Dictionary with training outcome, metrics, and checkpoint paths.
        """
        t_start = time.time()
        m_cfg = self.config.get("model", {})
        d_cfg = self.config.get("data", {})
        t_cfg = self.config.get("training", {})
        a_cfg = self.config.get("augmentation", {})

        # Resolve checkpoint to start from
        ckpt_str = str(m_cfg.get("pretrained_checkpoint", "yolov8n.pt"))
        ckpt_path = self.root / ckpt_str if not Path(ckpt_str).is_absolute() else Path(ckpt_str)
        if not ckpt_path.is_file() and (self.root / "yolov8n.pt").is_file():
            ckpt_path = self.root / "yolov8n.pt"

        # Resolve data YAML
        data_yaml = d_cfg.get("dataset_yaml", "data/processed/detection/dataset.yaml")
        data_path = self.root / data_yaml if not Path(data_yaml).is_absolute() else Path(data_yaml)
        if not data_path.is_file():
            raise FileNotFoundError(f"Dataset YAML not found at {data_path}")

        # Resolve device
        dev_req = t_cfg.get("device", "auto")
        if dev_req == "auto":
            resolved_dev = get_device("auto")
            device_str = "0" if resolved_dev.type == "cuda" else "cpu"
        elif str(dev_req).startswith("cuda:"):
            device_str = str(dev_req).split(":")[-1]
        elif str(dev_req) == "cuda":
            device_str = "0"
        else:
            device_str = str(dev_req)

        # Output project & experiment name
        project_dir = self.root / t_cfg.get("project", "models/detection")
        exp_name = t_cfg.get("name", "yolov8n_vehicle")
        last_ckpt = project_dir / exp_name / "weights" / "last.pt"

        # Initialize model
        is_resuming = False
        if resume:
            if last_ckpt.is_file():
                logger.info("Resuming training from last checkpoint: %s", last_ckpt)
                model = YOLO(str(last_ckpt))
                is_resuming = True
            else:
                logger.warning(
                    "Resume requested but last checkpoint not found at %s; starting fresh training with %s",
                    last_ckpt,
                    ckpt_path,
                )
                model = YOLO(str(ckpt_path))
        else:
            logger.info("Starting training with base model: %s", ckpt_path)
            model = YOLO(str(ckpt_path))

        # Check layer freeze
        freeze_layers = t_cfg.get("freeze", 0)

        # Assemble Ultralytics train kwargs
        train_args: dict[str, Any] = {
            "data": str(data_path),
            "epochs": int(t_cfg.get("epochs", 10)),
            "batch": int(t_cfg.get("batch_size", 16)),
            "imgsz": int(t_cfg.get("imgsz", 640)),
            "device": device_str,
            "workers": int(t_cfg.get("workers", 2)),
            "optimizer": str(t_cfg.get("optimizer", "AdamW")),
            "lr0": float(t_cfg.get("lr0", 0.001)),
            "lrf": float(t_cfg.get("lrf", 0.01)),
            "weight_decay": float(t_cfg.get("weight_decay", 0.0005)),
            "patience": int(t_cfg.get("patience", 5)),
            "project": str(project_dir),
            "name": exp_name,
            "exist_ok": bool(t_cfg.get("exist_ok", True)),
            "pretrained": bool(t_cfg.get("pretrained", True)),
            "amp": bool(t_cfg.get("amp", True)),
            "save_period": int(t_cfg.get("save_period", 1)),
            "plots": bool(t_cfg.get("plots", True)),
            "val": bool(t_cfg.get("val", True)),
            "verbose": True,
        }

        # Apply resume if valid checkpoint exists
        if is_resuming:
            train_args["resume"] = True

        # Apply freeze if requested
        if freeze_layers and int(freeze_layers) > 0:
            train_args["freeze"] = int(freeze_layers)

        # Incorporate augmentation settings
        train_args.update(
            {
                "hsv_h": float(a_cfg.get("hsv_h", 0.015)),
                "hsv_s": float(a_cfg.get("hsv_s", 0.7)),
                "hsv_v": float(a_cfg.get("hsv_v", 0.4)),
                "degrees": float(a_cfg.get("degrees", 10.0)),
                "translate": float(a_cfg.get("translate", 0.1)),
                "scale": float(a_cfg.get("scale", 0.5)),
                "shear": float(a_cfg.get("shear", 0.0)),
                "perspective": float(a_cfg.get("perspective", 0.0)),
                "flipud": 0.0,  # Strictly disallow vertical flipping
                "fliplr": float(a_cfg.get("fliplr", 0.5)),
                "mosaic": float(a_cfg.get("mosaic", 1.0)),
                "mixup": float(a_cfg.get("mixup", 0.0)),
            }
        )

        logger.info("Starting Ultralytics YOLO training with parameters:\n%s", json.dumps(train_args, indent=2, default=str))

        # Launch training
        results = model.train(**train_args)
        duration_s = round(time.time() - t_start, 2)

        # Retrieve paths to resulting checkpoints
        best_pt = project_dir / exp_name / "weights" / "best.pt"
        last_pt = project_dir / exp_name / "weights" / "last.pt"

        # Also copy best.pt to models/detection/best.pt for direct convenience
        convenience_best = self.checkpoints_dir / "best.pt"
        if best_pt.is_file():
            shutil.copy2(best_pt, convenience_best)
            logger.info("Copied best checkpoint to %s", convenience_best)

        # Extract validation metrics
        results_dict: dict[str, Any] = {}
        if hasattr(results, "results_dict") and isinstance(results.results_dict, dict):
            for k, v in results.results_dict.items():
                results_dict[k] = float(v) if isinstance(v, (int, float)) else str(v)

        # Summarize key metrics
        metrics_summary: dict[str, Any] = {
            "status": "COMPLETED",
            "duration_seconds": duration_s,
            "epochs_completed": train_args["epochs"],
            "batch_size": train_args["batch"],
            "image_size": train_args["imgsz"],
            "device": device_str,
            "best_checkpoint": str(best_pt.relative_to(self.root) if best_pt.is_relative_to(self.root) else best_pt).replace("\\", "/"),
            "last_checkpoint": str(last_pt.relative_to(self.root) if last_pt.is_relative_to(self.root) else last_pt).replace("\\", "/"),
            "convenience_checkpoint": str(convenience_best.relative_to(self.root) if convenience_best.is_relative_to(self.root) else convenience_best).replace("\\", "/"),
            "results_dict": results_dict,
        }

        # Map standard metric keys if available
        if "metrics/precision(B)" in results_dict:
            metrics_summary["precision"] = round(results_dict["metrics/precision(B)"], 4)
        if "metrics/recall(B)" in results_dict:
            metrics_summary["recall"] = round(results_dict["metrics/recall(B)"], 4)
        if "metrics/mAP50(B)" in results_dict:
            metrics_summary["mAP50"] = round(results_dict["metrics/mAP50(B)"], 4)
        if "metrics/mAP50-95(B)" in results_dict:
            metrics_summary["mAP50-95"] = round(results_dict["metrics/mAP50-95(B)"], 4)

        # Save metrics JSON
        out_metrics_file = self.metrics_dir / "training_metrics.json"
        with open(out_metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info("Saved training summary to %s", out_metrics_file)

        return metrics_summary
